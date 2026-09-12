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
hardware. `results.csv` (549 rows) was regenerated this session with the
new instrumentation; **every earlier number in it should be considered
superseded**, for the reasons in the next section.

Two documents carry the analysis: **`IDEAS.md`** is the prioritised
experiment backlog (~30 items, each with hypothesis / change / expected
gain / how to measure, and marked up with what has since been measured).
**`GOALS.md`** is the long-term target (four models, HTTP API in Go).

### This session: the §0 "measurement validity" block from IDEAS.md

Previous session produced `IDEAS.md`. This session implemented its §0 —
the work that had to happen before optimising against any of the existing
numbers — and it changed several answers.

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
section updated for what §0 found. In short:

1. **IDEAS §1.1 — W4A8 GEMV.** The highest-value item: a measured 2x on
   decode, from closing Q4's 31%-of-bandwidth-ceiling gap.
2. **IDEAS §2.1 — register-blocked coopmat GEMM.** 9% of a measured 55.5
   TFLOP/s, with 8 FLOP/byte of arithmetic intensity where 235 is needed.
   Largest absolute gain on the chip. Note the occupancy floor §0.1 found:
   **WMMA needs 2 waves per CU** for full rate (27.9 TFLOP/s at 40 waves,
   55.5 at 80), so register blocking must not drop occupancy below that.
3. **IDEAS §1.2 — remove the runtime integer divisions** (`pc.N / pc.block`
   and `n / pc.block` in every quantized inner loop). Cheapest real win,
   and it is on the critical path of the kernel §1.1 rewrites.
4. **IDEAS §6.1 — ISA dumps** (`RADV_DEBUG=asm`). Now has three specific
   questions: is `v_pk_fma_f16` emitted (finding 2 above), is
   `V_DOT4_I32_I8` emitted, and what are the VGPR counts that decide
   whether §2.1's register blocking fits in 2 waves per CU.
5. **Still open from §0**: CPU-side submit/fence cost (the half of §4.1 the
   `overhead` family does not measure), and `-blocks` is worth re-sweeping
   only where a block-size effect survives warming — in GEMV it does not.
6. **Not yet committed to git.** Last commit was `b362606`. This session's
   changes: 2 new shaders + 5 new SPIR-V variants, 4 new `bench/` files,
   modifications to `bench.go`/`quant.go`/all `ops_*.go` (TimeDispatch now
   returns clock stats)/`cmd/bench/main.go`/`shaders/shaders.go`, plus
   `README.md`, `IDEAS.md` and a regenerated `results.csv`.
