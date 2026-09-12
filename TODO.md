# TODO / session handoff — Strix Halo Vulkan inference benchmark suite

## Where we are

Goal: benchmark the Vulkan compute operations neural-network inference is
built from (GEMM, GEMV, elementwise, softmax/RMSNorm) across implementation
"flavours" on this specific machine's iGPU (AMD Radeon 8060S / RDNA3.5 /
RADV STRIX_HALO), to figure out how to build an inference engine tuned for
it — with a specific focus on whether ~4-bit quantized weights can be made
to perform well.

Everything below builds clean (`go build ./...`, `gofmt -l .`,
`go vet ./...` all clean) and has passed its correctness checks on real
hardware. `results.csv` was regenerated again this session, with the new
register-blocked WMMA GEMM rows.

Two documents carry the analysis: **`IDEAS.md`** is the prioritised
experiment backlog (~30 items, each with hypothesis / change / expected
gain / how to measure, and marked up with what has since been measured).
**`GOALS.md`** is the long-term target (four models, HTTP API in Go).

### This session: IDEAS §2.1, the register-blocked WMMA GEMM

`IDEAS.md`'s largest remaining item is done, and it beat its own forecast.
**A register-blocked cooperative-matrix GEMM reaches 25.2 TFLOP/s at
N=4096 — 6.0x the old coopmat kernel at the same shape and 45% of this
chip's measured 55.5 TFLOP/s of WMMA throughput, up from 8%.** (Against the
old kernel's own best figure anywhere in the sweep, 5.4 TFLOP/s at N=1024,
it is 4.6x.) §2.1 predicted 2-4x and 10-20 TFLOP/s.

**New code:**
- `shaders/gemm_wmma.comp`, compiled into **twelve** variants off one source
  via `glslc -D`. Four independent axes, so every effect is attributable
  rather than bundled: accumulator grid per wave (`WM`×`WN`), waves per
  workgroup (`WAVES_M`×`WAVES_N`), LDS staging (`USE_LDS`) with optional
  double buffering (`DOUBLE_BUFFER`), and B stored [N,K] + loaded
  column-major (`B_COLMAJOR`). Arithmetic intensity climbs 16 → 85 FLOP/byte
  against the old kernel's 8.
- `bench/ops_gemm_wmma.go` — the new `gemm` cases, each verified against a
  CPU reference at a 2×2-workgroup / 4-K-slab shape before timing (small
  enough for a Go reference, large enough that a wrong tile origin or
  wave-to-tile mapping fails). Its `Detail` column carries tile shape, total
  waves and arithmetic intensity, so a row can be placed against the
  bandwidth and MMA ceilings without re-deriving them.
- `cmd/probe/main.go` — dispatches a single `.spv` once so `RADV_DEBUG=asm`
  and `RADV_DEBUG=shaderstats` can be aimed at any shader. This is IDEAS
  §6.1, and it is what found the B-layout effect below.

**The prefill table, after** (N=4096; implied traffic is GFLOP/s ÷
arithmetic intensity, to place each row against the 805 GB/s MALL and
236 GB/s DRAM ceilings):

| kernel | tile | waves/wg | AI | GFLOP/s | % of 55.5 T | implied traffic |
|---|---|---|---|---|---|---|
| `coopmat` (before) | 16×16 | 1 | 8 | 4190 | 7.5% | 524 GB/s |
| `wmma_reg32` | 32×32 | 1 | 16 | 12233 | 22% | **765 GB/s** |
| `wmma_reg64` | 64×64 | 1 | 32 | 24276 | 44% | **759 GB/s** |
| **`wmma_reg64x128`** | 64×128 | 1 | 43 | **24458** | **44%** | 573 GB/s |
| **`wmma_wg128x256`** | 128×256 | 4 | ≤85 | **25245** | **45%** | ≤296 GB/s |
| `wmma_lds128_db` | 128×128 | 4 | 64 | 21223 | 38% | 332 GB/s |
| `wmma_lds128` | 128×128 | 4 | 64 | 17163 | 31% | 268 GB/s |
| `wmma_reg64_bt` | 64×64 | 1 | 32 | 8543 | 15% | 267 GB/s |

The `wg*` row's intensity is an upper bound, not a fact: its four waves
issue overlapping fragment loads and only reach it if L0/L1 dedups them.

**Five things worth carrying forward:**

1. **Arithmetic intensity was the right variable, and it pays linearly
   once the MALL saturates.** AI 8 → 16 → 32 gives 4.19 → 12.2 → 24.3
   TFLOP/s, and the traffic column says why: at AI 16 and 32 the kernel is
   pinned at 765 and 759 GB/s (766-814 across the sizes swept), i.e. at the
   measured 805 GB/s MALL, so the 16 → 32 step is *exactly* 2.0x — a
   doubling of reuse buying a doubling of throughput out of a fixed budget.
   The 8 → 16 step is 2.9x rather than 2x because the old kernel's 524 GB/s
   shows it lost twice over: with one accumulator per wave there is no
   independent work to hide load latency behind, so it was not even
   bandwidth-saturated.
2. **Getting rid of LDS staging is the win; grouping waves is neither here
   nor there.** The top two configurations tie within run-to-run noise —
   one wave per workgroup with no shared memory (`wmma_reg64x128`) and four
   waves with no shared memory and no barriers (`wmma_wg128x256`) — while
   *every* LDS variant loses, the best of them by 1.2x. And the LDS variants
   are nowhere near bandwidth-bound (149-332 of 805 GB/s); they are bound by
   their own shared-memory reads and barriers. **The 32 MiB MALL is already
   supplying the cross-wave reuse that LDS staging exists to provide**, so
   staging it explicitly just adds barriers and `ds_load`s on top. On a
   discrete part with a small L2 this ordering would very likely invert, so
   it is a property of the APU, not of the kernel.
3. **Double buffering works and is worth 1.24x** (17.2 → 21.2 TFLOP/s) — one
   barrier per K-step instead of two, with the next slab's global loads
   issued before the current slab's MMAs so the compiler sinks the
   `s_waitcnt` past them. It just is not enough to make the LDS path
   competitive here.
4. **§2.3's B layout is a sharp, unresolved contradiction, and it is the
   single most useful thing reading the ISA produced.** RADV's `coopMatLoad`
   emits **2 `buffer_load_b128`** for a row-major A or **column-major** B
   operand, and **16 scalar `buffer_load_d16_b16`** for the other two
   combinations — the hardware WMMA operand layout is K-contiguous for both
   operands, so the conventional [K,N] row-major B is gathered element by
   element. Storing B [N,K] takes `wmma_reg64`'s loop from **577
   instructions (232 of them address arithmetic) to 389 (144)** for the same
   16 `v_wmma`… **and is 2.8x slower at N=4096** (8543 vs 24276), across
   every tiling tried. It is *faster* at N=512 (14265 vs 12838) and degrades
   exactly as B stops being cache-resident, so it is a memory-system effect
   on a strided gather, not an instruction-count one. One mechanism was
   tested and refuted (partial cache-line consumption: deepening the K-slab
   to 64 so a whole 128-byte line is consumed per MMA block changes nothing,
   8583 vs 8543). The
   untested one is channel/bank aliasing from the power-of-two leading
   dimension — a column-major fragment load issues 16 addresses `K·2` bytes
   apart within one instruction, and every K swept here is a power of two.
5. **The occupancy floor never bound anything.** §0.1's warning was that
   8-16 accumulators per wave would drop below the 2-waves-per-CU WMMA needs.
   `shaderstats` says **144 VGPRs for 16 accumulators, 252 for 32, no spills
   and no scratch** — 3-5 waves per SIMD. Register blocking here is capped by
   the 256-VGPR wave64 limit, not by occupancy, which also makes §6.2
   (wave32) more interesting than it was: WMMA is natively a wave32 shape.

### This session: IDEAS §1.1, the W4A8 GEMV kernel

The highest-value item in `IDEAS.md` is done, and it delivered more than it
promised. **W4A8 GEMV — 4-bit weights against int8 activations, both fed to
the packed-int8 dot instruction — runs at 819 GFLOP/s and 211 GB/s against
DRAM-resident weights: 2.4x the previous best decode kernel, and 89% of
this machine's 236 GB/s of DRAM bandwidth.** Decode has gone from
ALU-bound-on-the-unpack to bandwidth-bound, which is the end state §1 was
aiming for.

**New code:**
- `shaders/gemv_w4a8.comp`, compiled in two variants off one source:
  `VEC=1` loads one `uint` (8 weights) per lane per step, `VEC=4` one
  `uvec4` (32 weights). They differ in nothing else, so the pair also
  measures IDEAS §1.3's "wider loads" hypothesis in isolation.
- `bench/ops_w4a8.go` — the `gemv`-family cases (variants `subgroup` and
  `subgroup_vec4`, format `w4a8`), each verified against a CPU reference
  before timing.
- `bench/quant.go` gains `repackQ4ToW4A8`, `int8BlockSums` and the
  int32/uint32 byte helpers; `bench/ops_gemv_cold.go` gains a `coldKind`
  for the three different shader calling conventions and runs W4A8 in the
  DRAM-resident sweep too.

**The decode table, after:** (`gemv_cold`, 256MB fp16-equivalent footprint,
M=32768, N=4096, block=128 — the regime real decode runs in)

| format | kernel | GFLOP/s | GB/s | % of its own DRAM ceiling |
|---|---|---|---|---|
| fp16 | subgroup | 177 | 177 | 75% |
| q8 | subgroup | 234 | 119 | 50% |
| q4 | subgroup, float unpack | 283 | 73 | 31% |
| w8a8 | packed dot | 339 | 172 | 73% |
| **w4a8** | packed dot, `uint` loads | **698** | 180 | 76% |
| **w4a8** | packed dot, `uvec4` loads | **819** | **211** | **89%** |

Cache-resident (`gemv`, N=1024, block=128) the same kernel does 1326
GFLOP/s against W8A8's 523 and Q4's 378, and 2538 GFLOP/s at 654 GB/s at
the 64MB footprint, where 4-bit weights still fit the 32MB MALL.

**Four things worth carrying forward:**

1. **The obvious de-bias bit trick does not work, and the fix is better
   than the trick.** `(v & 0x0F0F0F0F) - 0x08080808` borrows across byte
   lanes whenever a nibble is below 8, corrupting the neighbouring weight;
   there is no borrow-free per-byte subtract. So the kernel keeps the
   nibbles unsigned and uses the **mixed-signedness** overload
   `dotPacked4x8EXT(int, uint)` (this device reports
   `integerDotProduct4x8BitPackedMixedSignednessAccelerated = true`),
   cancelling the +8 bias afterwards with
   `sum (u-8)x = sum u*x - 8*sum x`. The per-block `sum x` is the same for
   every weight row, so it is precomputed once per GEMV into a fifth
   binding and costs one short loop over `blocksPerRow` — zero extra
   instructions in the inner loop. A real engine gets those sums free out
   of its activation-quantize pass.
2. **The weight layout is a repack, not a re-quantization.** One word holds
   8 consecutive weights with its 4 low nibbles carrying elements n..n+3
   and its high nibbles n+4..n+7, so masking yields two packed-int8 words
   that line up with two consecutive words of a plainly packed int8
   activation vector — no per-token permutation of `x`. It is a pure
   permutation of `quantizeQ4`'s output, so the CPU reference is still
   built from `dequantizeQ4` on the pre-repack array, which checks the
   repack as well as the kernel. Real models would pay this once, offline.
3. **The disassembly confirms all of it** (`RADV_DEBUG=asm` on the gemv
   family; the w4a8 pipelines are the last two compiled). The `VEC=4`
   inner loop is 3 `buffer_load_b128` + one `buffer_load_d16_b16` for the
   scale, 8 and/shift pairs, **8 chained
   `v_dot4_i32_iu8 ... neg_lo:[1,0,0] clamp`** (mixed-signedness hardware
   dot: signed activations, unsigned nibbles), one `v_cvt_f32_i32` and one
   `v_fma_mix_f32`. No integer division, no per-byte extraction. W8A8 emits
   `neg_lo:[1,1,0]`, one dot per iteration, for comparison.
4. **Wide loads are worth 1.17x** (698 → 819 GFLOP/s DRAM-resident,
   1204 → 1326 cache-resident) — the difference between 76% and 89% of the
   DRAM ceiling. That is IDEAS §1.3's load half confirmed, and the fp16 and
   W8A8 GEMV kernels (75% and 73%) are the obvious next places to apply it.
   §1.2's "no integer division" is also folded into this kernel: it takes
   `log2(block)` and shifts.

### Previous session: the §0 "measurement validity" block from IDEAS.md

The session before produced `IDEAS.md`. That session implemented its §0 —
the work that had to happen before optimising against any of the existing
numbers — and it changed several answers. **Every number predating it
should be considered superseded.**

**New code:**
- `shaders/alu_peak.comp` (5 variants: fp32 FMA, packed-fp16 FMA,
  `dotPacked4x8AccSatEXT`, WMMA fp16, WMMA int8) + `bench/ops_peak.go` —
  the `peak` family. Operands are register-resident before the loop and
  each iteration feeds the next accumulator, so the timed loop touches no
  memory and cannot be hoisted; 4-8 independent accumulator chains hide
  instruction latency. This establishes the *measured* ceilings every other
  kernel should be judged against.
- `shaders/empty.comp` + `bench/ops_overhead.go` — the `overhead` family:
  dispatch + pipeline-barrier cost with no work.
- `bench/ops_gemv_cold.go` — the `gemv_cold` family: decode-shape GEMV
  against weight matrices several times the last-level cache, swept by
  fp16-equivalent footprint so every format in a row holds the same number
  of weights. Fills buffers with cheap deterministic patterns rather than
  quantizing float32 weights (which would be gigabytes at these sizes) and
  deliberately skips the CPU reference, which `RunGEMV` already does at
  small sizes — only a cheap output sanity check runs.
- `bench/sysmon.go` — clock/power instrumentation. Every `Result` now
  carries mean/min/max sclk and package power for its timed batch (four new
  CSV columns plus a `detail` column), and `TimeDispatch` drives the GPU
  with ALU-heavy work before each measurement until sclk reaches 97% of the
  advertised maximum. **Closed-loop, not a fixed duration** — a first
  attempt with a fixed 200ms warmup still left expensive cases measured
  mid-ramp at 1068-2400 MHz while cheap ones ran at 2900.
- `cmd/bench`: new families registered, plus `-coldfootprints`, `-coldn`,
  `-warmclock`, `-clocksample`, `-cus`; `-bwsizes` now defaults to 12
  points from a 2MB to a 512MB footprint instead of 4.

**Measured hardware ceilings** (40 CUs at 2900 MHz), the main deliverable:

| ceiling | measured | ops/clk/CU | best real kernel | utilisation |
|---|---|---|---|---|
| DRAM bandwidth | 236 GB/s | — | 236 GB/s | ~100% |
| MALL (32 MiB) | ~805 GB/s | — | ~805 GB/s | ~100% |
| fp32 FMA | 22.9 TFLOP/s | 198 | 2.76 TFLOP/s | ~12% |
| packed fp16 FMA | 25.3 TFLOP/s | 218 | unused | — |
| `dotPacked4x8` | 54.0 TOP/s | 465 | 2.2 TOP/s | ~4% |
| WMMA fp16 | 55.5 TFLOP/s | 479 | 4.9 TFLOP/s | ~9% |
| WMMA int8 | 55.7 TOP/s | 480 | 4.7 TOP/s | ~8% |

**Five findings that change prior conclusions:**

1. **WMMA int8 is exactly as fast as WMMA fp16 here — not 2x.** 479 vs 480
   ops/clk/CU. The discrete-RDNA3 assumption that int8 matrix throughput
   doubles does not hold for RDNA3.5. The apparent 2x in the old data
   (`coopmat,q8` 9.6 TOPS vs `coopmat,fp16` 4.4 TFLOP/s at N=512) was a
   *memory* effect — int8 reads half the bytes and both kernels are
   memory-bound. This removes the main reason to write a W4A8 coopmat
   kernel (IDEAS §2.2, downgraded).
2. **Packed fp16 FMA is only 1.10x scalar fp32 FMA**, not the 2x the packed
   ALU should give. Needs an ISA dump to find out whether `v_pk_fma_f16` is
   even being emitted (IDEAS §6.1).
3. **The DVFS artefacts were real and large.** With warming,
   `gemv,subgroup,w8a8,block=128,N=1024` went 190 → **522** GFLOP/s,
   `gemv,subgroup,q4,block=128,N=2048` went 135 → **401**, and
   `bandwidth,copy` at 1M elements went 262 → **775** GB/s. Both GEMV
   sweeps are now monotonic in size. **The previous session's "the W8A8
   GEMV block-size effect is noisy rather than a clean trend" was this, not
   the kernel** — block size now barely matters anywhere in GEMV.
4. **Per-dispatch overhead is ~300 ns**, plus ~0.7 ns per workgroup
   scheduled — not the tens of microseconds that would have capped decode
   throughput. ~500 dispatches per token is under 1% of a ~20 ms token, so
   kernel fusion is justified by DRAM round-trips, not launch cost. (This
   measures dispatch + barrier *inside* a command buffer; CPU-side
   `vkQueueSubmit` + fence wait is still unmeasured.)
5. **The cache cliff is exactly 32 MiB, and it is a cliff, not a slope**:
   804 GB/s at a 32 MiB footprint, 235 GB/s at 48 MB. The in-place
   `elementwise` sweep agrees to the element — its last fully-cached fp32
   case is 8388608 × 4 B = exactly 32 MiB. So there is a **3.4x** bandwidth
   prize for any working set that can be kept under 32 MiB.

**The decode answer, corrected.** `gemv_cold` at a 256MB fp16-equivalent
footprint (M=32768, N=4096), which is the regime real decode runs in:

| format | bytes/weight | best achieved | DRAM-bound ceiling | % of ceiling |
|---|---|---|---|---|
| fp16 | 2 | 179 GFLOP/s @ 179 GB/s | 236 | 76% |
| q8 | 1 | 232 GFLOP/s @ 122 GB/s | 464 | 50% |
| **q4** | 0.5 | **284 GFLOP/s @ 73 GB/s** | **916** | **31%** |
| w8a8 | 1 | 344 GFLOP/s @ 173 GB/s | 464 | 73% |

- The old square sweep **overstates decode by 3.0x** (W8A8: 1033 vs 344).
- W8A8's lead over Q4 collapses from ~2.6x to **1.2x** once weights come
  from DRAM.
- But the prediction that Q4 would *overtake* W8A8 was wrong, and the
  reason is the useful part: **Q4 is ALU-bound in its nibble unpack even
  against DRAM**, moving only 73 of 236 GB/s. Its own bandwidth ceiling is
  916 GFLOP/s; matching W8A8's 73% efficiency would put it at ~670 —
  **2x the current champion**. That is now the highest-value item in
  `IDEAS.md` (§1.1, W4A8: load Q4 as `uint`, unpack 8 nibbles to two
  packed-int8 words with mask/subtract, feed `dotPacked4x8EXT`, scale once
  per block).

### What got built (earlier sessions)

- **`vk/`** — the engine: optional device features (fp16, int8, integer
  dot-product, cooperative-matrix) negotiated at `NewDevice`, N-buffer
  pipelines with push and specialization constants, GPU-timestamp timing
  (`ComputePipeline.DispatchTimed`), buffer allocation preferring this
  APU's `DEVICE_LOCAL|HOST_VISIBLE` unified memory,
  `PhysicalDevice.CooperativeMatrixShapes()` (this device reports only
  16x16x16, subgroup scope, fp16→fp32 and int8→int32).
- **`shaders/`** — 19 `.comp` sources compiling to 36 SPIR-V variants
  (precision/tile variants via `glslc -D`, not duplicated GLSL): bandwidth
  baseline, elementwise/relu, GEMV (naive/subgroup × fp32/fp16/q8/q4 plus
  subgroup W8A8), GEMM (naive/tiled × fp32/fp16/q8/q4, cooperative-matrix
  fp16/int8/q4, plus naive W8A8), RMSNorm/softmax (shared/subgroup), a
  standalone Q4→fp16 dequant kernel, and this session's ALU-peak and empty
  kernels.
- **`bench/`** — the harness: `bench.go` (timing + adaptive iteration
  capping), `quant.go` (fp16 conversion with round-to-nearest, GGML-style
  Q8/Q4 block quantize/dequantize), `sysmon.go` (clock/power), `ops_*.go`
  (one file per op family).
- **`cmd/bench`** — the CLI (`go run ./cmd/bench -h`).
- Original demo (`main.go`, `./strix-halo-vulkan`) still works unchanged.

### Bugs hit and fixed in earlier sessions (worth remembering)

1. **GPU driver hang watchdog**: naive GEMM at N=4096 batched into 20
   back-to-back dispatches took long enough to trip amdgpu's TDR →
   `VK_ERROR_DEVICE_LOST`, which poisons the device for every case still
   queued. `bench.TimeDispatch` now probes with 1 iteration and caps any
   batch to ~500ms worth.
2. **O(N³) CPU reference GEMM inside the size sweep** sat at 98% CPU for
   18+ minutes at N=4096. **If adding a new op/variant, verify once at a
   small fixed size, never inside the sweep loop.**
3. **`VK_KHR_shader_integer_dot_product` was requested but never enabled**:
   `vk/shim.c`'s `shim_create_device` built the feature struct into the
   `pNext` chain but never added the extension name to the enabled list
   (required since the instance targets Vulkan 1.2, where it is not yet
   core). Fixed.

## Next steps (pick up here)

`IDEAS.md` has the full backlog with its "Suggested order of attack"
updated for what §0, §1.1 and §2.1 found. In short:

1. **IDEAS §2.3 — resolve why the better instruction stream is slower.**
   Storing B [N,K] and loading it column-major cuts the best GEMM kernel's
   loop from 577 instructions to 389 and is **2.8x slower** at N=4096, only
   once B stops being cache-resident. §2.1's remaining gap to the ceiling
   looks like exactly the address-arithmetic pressure that layout removes,
   so one answer here unblocks the largest kernel on the chip. Untested
   mechanism: channel/bank aliasing from the power-of-two leading dimension.
   Testing it needs a separate stride push constant (the kernel currently
   reuses `pc.K` as both extent and stride) so the leading dimension can be
   padded off a power of two — an afternoon's work.
2. **IDEAS §2.4 — workgroup swizzle.** Upgraded to high value by §2.1: the
   best kernel is pinned at ~760 of the MALL's 805 GB/s, and a swizzle is
   the only remaining change that cuts MALL traffic without touching the
   tile shape or the 256-VGPR budget. §2.1 also showed all its cross-tile
   reuse comes from the MALL (neither LDS nor multi-wave workgroups help),
   which is precisely the resource a swizzle reorders. Cheapest item with a
   measured constraint behind it.
3. **IDEAS §2.2's int8 arm — a `PRECISION_I8` variant of
   `gemm_wmma.comp`**, accumulating in int32. Small change to a kernel that
   now reaches 45% of the ceiling, halves its operand bytes, and gives W8A8
   prefill its number. (The *Q4*-unpack version stays downgraded: there is
   no int8 matrix-rate bonus on this chip, and the winning kernel has no LDS
   tile whose fill cost a cheaper unpack would improve.)
4. **IDEAS §1.3 — wide loads on the other GEMV kernels.** Measured at 1.17x
   on W4A8. fp16 GEMV sits at 75% of its DRAM ceiling and W8A8 at 73%,
   where W4A8 now reaches 89%.
5. **IDEAS §1.2 — remove the runtime integer divisions** (`pc.N / pc.block`
   and `n / pc.block`) from everything W4A8 did not rewrite: the
   naive/tiled GEMM paths and the old quantized GEMV variants.
   `gemv,naive,q4` is still slower than naive fp32, which is the symptom.
6. **IDEAS §6.2 — wave32 vs wave64**, now more interesting than when it was
   filed: WMMA is natively a 16x16x16 wave32 shape, RADV runs these shaders
   at wave64, and §2.1's tile size is capped by the 256-VGPR wave64 limit
   rather than by occupancy. Needs a small `vk/shim.c` change to pass
   `VK_EXT_subgroup_size_control`'s `requiredSubgroupSize`.
7. **IDEAS §1.6 — the activation-quantize cost W8A8 and W4A8 both hide.**
   Both kernels get `x` quantized on the host, outside the timed loop, and
   W4A8 additionally gets its per-block activation sums for free. A real
   decode step must produce both on-GPU between layers. The sums are one
   extra reduction over N (cheap, and fusable into the quantize pass), but
   it should be measured rather than assumed — it is the one place where
   these numbers are friendlier than production would be.
8. **Still open from §0**: CPU-side submit/fence cost (the half of §4.1 the
   `overhead` family does not measure), and the last ISA question from §6.1
   (is `v_pk_fma_f16` actually emitted, given §0.1's 1.10x packed-fp16
   surprise). `-blocks` is worth re-sweeping only where a block-size effect
   survives warming — in GEMV it does not.
9. **Not committed to git.** The W4A8 work went in as 620f23f; this
   session's is still outstanding: new `shaders/gemm_wmma.comp`,
   `bench/ops_gemm_wmma.go` and `cmd/probe/main.go`, edits to
   `shaders/shaders.go` and `bench/ops_gemm.go`, plus `README.md`,
   `IDEAS.md`, this file, and a regenerated `results.csv`. Note `*.spv` is
   gitignored, so **`go generate ./...` is required after a fresh
   checkout** — the 12 WMMA variants are all built from the one `.comp`.
