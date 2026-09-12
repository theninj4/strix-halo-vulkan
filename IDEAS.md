# IDEAS — experiments worth running to squeeze more out of Strix Halo

Written after reviewing `TODO.md`, every shader in `shaders/`, the full
`results.csv`, and the device's actual reported capabilities (`vulkaninfo`,
`/sys/class/drm/card1/device/pp_dpm_sclk`). This is a prioritised
experiment backlog, not a plan — each item states a *hypothesis*, the
*change*, the *expected gain*, and *how we'd know*.

**§0 has since been implemented and run** (549-row `results.csv`
regenerated with clock instrumentation). Items it confirmed, refuted, or
re-aimed are marked **[measured]** in place rather than rewritten away, so
the wrong predictions stay visible next to what actually happened.

**§1.1 (W4A8 GEMV) has since been implemented and run too**, and it
delivered: 819 GFLOP/s against DRAM-resident weights, **2.4x** the previous
best decode kernel and **89% of the DRAM bus**. Decode is now a solved
bandwidth problem at 4 bits/weight; see §1.1 for what it took and §1.3 for
the load-width half of it.

**§2.1 (register-blocked coopmat GEMM) is done as well**, and it delivered
more: **25.2 TFLOP/s, 6.0x the previous best GEMM kernel at the same shape
and 45% of the measured WMMA ceiling**, up from 8%. Two of its sub-hypotheses were wrong in
useful ways — LDS staging and multi-wave workgroups are both *unnecessary* on
this part, because the 32 MiB MALL already supplies the reuse they exist to
provide — and §2.3 turned into a sharp, unresolved contradiction: the B
layout that halves the instruction count is 2.8x slower. See §2.1, §2.3, §6.1.

## The roofline, and why it says there's a lot left on the table

Hardware numbers for this part (AMD Radeon 8060S, gfx1151, 40 CU, sclk
tops out at **2900 MHz** per `pp_dpm_sclk`; LPDDR5X-8000 on a 256-bit bus):

> **Updated after running §0.** The ceilings below are now *measured* on
> this chip (`go run ./cmd/bench -skip ...` → the `peak` family), not
> extrapolated from discrete RDNA3 parts, and several §0 hypotheses turned
> out to be wrong. Corrections are marked **[measured]** throughout.

| Ceiling | Measured peak | Ops/clk/CU | Best real kernel | Utilisation |
|---|---|---|---|---|
| DRAM bandwidth | **236 GB/s** (92% of the 256 GB/s bus) | — | 236 GB/s | **~100%** ✅ |
| MALL (32 MiB) bandwidth | **805 GB/s** | — | 805 GB/s | **~100%** ✅ |
| Vector FP32 FMA | **22.9 TFLOP/s** | 198 | 2.76 TFLOP/s (`tiled,fp16`) | ~12% |
| Packed fp16 FMA | **25.3 TFLOP/s** | 218 | — (no kernel uses it) | — |
| `dotPacked4x8` int8 | **54.0 TOP/s** | 465 | 2.2 TOP/s (`naive,w8a8`) | **~4%** ❌ |
| WMMA fp16→fp32 | **55.5 TFLOP/s** | 479 | **25.2 TFLOP/s** (`wmma_wg128x256`) | **45%** |
| WMMA int8→int32 | **55.7 TOP/s** | 480 | 4.7 TOP/s (`coopmat,q8`) | **~8%** ❌ |

(Real-kernel column is the best measured anywhere in `results.csv`; the
GEMM entries are at N=4096, the `naive,w8a8` entry at N=256 where it is
still cache-resident. The WMMA fp16 row was 4.9 TFLOP/s / 9% until §2.1
register-blocked the kernel; the int8 row is still the un-blocked one and
would move the same way if §2.2 were written. The `dotPacked4x8` row's 4% is not the indictment it
looks like: the kernels that use that instruction — W8A8 and now W4A8 GEMV
— are memory-bound by construction, and W4A8 runs at 89% of the *bandwidth*
ceiling, which is the one that binds it.)

Four things in that table were not what §0 predicted:

1. **WMMA int8 is exactly as fast as WMMA fp16 on this chip — not 2x.**
   Both measure 479-480 ops/clk/CU. The discrete-RDNA3 assumption that int8
   matrix throughput doubles does not hold for RDNA3.5 here. This falsifies
   the original reasoning below that "int8 coopmat has 2x the hardware rate
   yet lands at the same GFLOP/s, therefore the kernel is memory-bound" —
   the two land together because the *hardware* rates are equal. The
   memory-bound conclusion still stands, but now rests on the much stronger
   direct evidence in the last column: the best GEMM kernel reaches 9% of a
   ceiling that was measured on this very device.
2. **WMMA fp16 hits 479 of the assumed 512 FLOP/clk/CU** — so the one
   extrapolated figure that *was* roughly right is the fp16 WMMA peak
   (55.5 TFLOP/s measured vs ~59 predicted).
3. **Packed fp16 FMA is only 1.10x scalar fp32 FMA, not 2x** (218 vs 198
   ops/clk/CU). Either `v_pk_fma_f16` isn't being emitted, or it doesn't
   dual-issue the way fp32 does. This substantially weakens §1.3's
   packed-fp16 argument — see the note there.
4. **fp32 FMA reaches 198 ops/clk/CU**, above the 128 that single-issue
   64-lane FMA allows, so RDNA3's dual-issue (VOPD) is working — at 77% of
   the 256 it would allow. It is also *power*-limited rather than
   issue-limited: throughput peaks at 320 waves (22.9 TFLOP/s, 118 W) and
   then *falls* at 640 and 1280 waves (20.0 and 19.2 TFLOP/s) as package
   power climbs to 138 W. The WMMA paths, by contrast, are flat across wave
   count and draw only ~95-110 W at full rate — matrix-core work is both
   2.4x faster and lower-power than scalar FMA here.

One more result worth designing around: **WMMA needs 2 waves per CU to
reach full rate.** wmma_fp16 measures 27.9 TFLOP/s at 40 waves (1/CU) and
55.5 at 80 waves (2/CU), then stays flat to 1280. Any coopmat kernel must
keep ≥80 workgroups resident, which also means a register-blocked kernel
(§2.1) must not blow occupancy down to one wave per CU. **[measured]** In
practice this never bound §2.1: its heaviest variant uses 252 of the 256
VGPRs a wave64 may hold, with no spills, which is still 3 waves per SIMD,
and a 64×128 tile at N=4096 launches 2048 workgroups. Register blocking
here is capped by the VGPR *limit*, not by occupancy.

Two conclusions drive everything below, both strengthened by the
measurements:

1. **DRAM bandwidth is already saturated** (236 of 256 GB/s). No kernel
   trick will beat it. So for *decode*, the only lever is **reading fewer
   bytes per weight**. The corollary is sharper than expected: see §1.1,
   which is now backed by per-format DRAM-resident ceilings. **[measured]**
   §1.1 has since been built: the W4A8 kernel reads 4 bits/weight *and*
   reaches 211 of 236 GB/s, so decode has gone from "the format that reads
   fewest bytes can't consume them" to "the bus is the limit". The lever
   that remains for decode is the format itself, not the kernel.
2. **The matrix cores are ~91% idle.** The best GEMM kernel on this chip
   reaches 4.9 of 55.5 TFLOP/s. The coopmat kernels are nowhere near an
   MMA-issue limit; they are memory-bound because of how they are written
   (8 FLOP/byte of arithmetic intensity where ~235 is needed to saturate
   55.5 TFLOP/s at 236 GB/s), not because of the hardware. So for
   *prefill/batch*, the lever is **arithmetic intensity** (§2.1).
   **[measured]** §2.1 has since been built and this was right: raising
   arithmetic intensity from 8 to 32 FLOP/byte by register-blocking the
   accumulators took the kernel from 4.2 to 24.3 TFLOP/s, 8% → 44% of the
   ceiling, and the returns were *linear in intensity* because the kernel
   sits pinned at ~760 of the 805 GB/s the MALL delivers. Past AI 32 they
   stop: 25.2 TFLOP/s is the plateau, bound by neither ceiling. Note what
   did **not** matter — LDS staging, more waves per workgroup, and the
   occupancy floor below were all irrelevant or harmful here.

There is also a third, newly quantified lever: **the 32 MiB MALL delivers
805 GB/s, 3.4x DRAM.** Anything that can be restructured to work out of a
sub-32MiB tile — attention K/V tiles, a single quantized layer's weights,
activations — gets 3.4x the bandwidth of anything that streams.

---

## 0. Measurement-validity work — **DONE**, and it changed the answers

All four items are implemented and run. The harness now measures the
hardware's ceilings, records the clock every number was taken at, and has a
DRAM-resident decode benchmark. Three of the four hypotheses were confirmed;
one was wrong in an instructive way.

### 0.1 Real MMA ceiling from a memory-free microbenchmark — **done** ✅
`shaders/alu_peak.comp` + `bench/ops_peak.go`, run as the `peak` family.
Operands are loaded into registers before the loop and each iteration feeds
the next accumulator, so the loop touches no memory and cannot be hoisted;
4-8 independent accumulator chains hide instruction latency. Results and
the four surprises are in the roofline table at the top of this document.
The headline: **fp16 and int8 WMMA are both ~55.5 T-ops/s (479-480
ops/clk/CU), identical to each other**, and the best GEMM kernel achieves
9% of that.

### 0.2 Clock/power logging and warmup — **done** ✅, hypothesis **confirmed**
`bench/sysmon.go`. Every `Result` now carries the mean/min/max shader clock
and package power observed during its timed batch, in the table and in four
new CSV columns. Before each measurement the harness drives the GPU with
ALU-heavy work until the clock reaches 97% of the advertised maximum —
**closed-loop rather than a fixed duration**, because the ramp time depends
on how much host-side work (generating and quantizing weights) preceded the
case, and a fixed 200ms warmup left the expensive cases still mid-ramp at
1068-2400 MHz while cheap ones ran at 2900.

The suspected DVFS artefacts were exactly that:

| case | before | after | ratio |
|---|---|---|---|
| `gemv,subgroup,w8a8,block=128,N=1024` | 190 GFLOP/s | **522** | 2.75x |
| `gemv,subgroup,q4,block=128,N=2048` | 135 GFLOP/s | **401** | 2.98x |
| `bandwidth,copy,n=1048576` | 262 GB/s | **775** | 2.96x |

Both GEMV sweeps are now monotonic in size, and the bandwidth curve is
monotonic up to its cache cliff. Every measurement in the current
`results.csv` was taken between 2776 and 2900 MHz. Note the third row:
the small-buffer bandwidth anomaly was *also* clock, not the dispatch
overhead §4.1 suspected — cache-resident bandwidth scales with core clock.

### 0.3 Cache-resident vs DRAM-resident decode — **done** ✅, hypothesis **half right**
`bench/ops_gemv_cold.go`, run as the `gemv_cold` family: the same subgroup
GEMV kernels against weight matrices several times the 32 MiB cache, swept
over fp16-equivalent footprints so every format in a row holds the same
number of weights. One large matrix suffices rather than round-robining
several — within a dispatch each weight is read exactly once, so there is
no intra-dispatch reuse to defeat.

**Confirmed: the existing headline numbers overstate decode by ~3x.**
W8A8 GEMV reports 1033 GFLOP/s on the cache-resident square sweep and
**344 GFLOP/s** on a 256MB weight footprint — a **3.0x** overstatement.

**Wrong: the ranking did not invert.** The prediction was that Q4, reading
half the bytes, would overtake W8A8 once DRAM-bound. It did not — W8A8 still
wins (344 vs 284 GFLOP/s). The reason is the interesting part, and it
*redirects* rather than weakens §1.1: **Q4 is still not bandwidth-bound even
when DRAM-resident.** Per-format, at a 256MB fp16-equivalent footprint:

| format | bytes/weight | best achieved | DRAM-bound ceiling | fraction of its ceiling |
|---|---|---|---|---|
| fp16 | 2 | 179 GFLOP/s @ 179 GB/s | 236 GFLOP/s | 76% |
| q8 | 1 | 232 GFLOP/s @ 122 GB/s | 464 GFLOP/s | 50% |
| **q4** | 0.5 | **284 GFLOP/s @ 73 GB/s** | **916 GFLOP/s** | **31%** ❌ |
| w8a8 | 1 | 344 GFLOP/s @ 173 GB/s | 464 GFLOP/s | 73% |

Block size, swept 32→1024, now barely matters in any of these (q4 varies
267-284, w8a8 328-344) — further evidence that the old sweep's dramatic
block-size "sensitivity" in GEMV was the clock artefact of §0.2, not a
property of the kernels.

Q4 leaves the most on the table by far: it is ALU-bound in its nibble
unpack, moving only 74 of the available 236 GB/s. Fixing that (§1.1) should
land it near 670-900 GFLOP/s — **2-2.7x the current W8A8 champion** — which
is a stronger case for W4A8 than the original bytes-only argument, and a
quantified one. Secondary finding: *nothing* reaches 236 GB/s (best is 75%),
so there is another ~25% in the GEMV memory access pattern itself (§1.3,
§1.4).

### 0.4 Fine-grained bandwidth-vs-working-set curve — **done** ✅
`-bwsizes` now defaults to 12 points from a 2MB to a 512MB footprint. The
cache cliff is not just straddled but pinned exactly:

| footprint | copy GB/s |
|---|---|
| 2 / 4 / 8 MB | 673 / 740 / 775 |
| 16 / 24 / **32 MiB** | 797 / 802 / **804** |
| **48 MB** | **235** ← cliff |
| 64 / 128 / 256 / 512 MB | 237 / 235 / 237 / 236 |

The in-place `elementwise` sweep agrees to the element: its last
fully-cached fp32 case is 8388608 elements × 4 B = **exactly 32 MiB**, at
809 GB/s, and the next point (50 MB) falls to 236. So the MALL is 32 MiB,
delivers **~805 GB/s**, and DRAM delivers 236 GB/s — a **3.4x** cliff, and a
hard budget for any "keep it resident" strategy (§5).

### 0.5 Per-dispatch overhead — **done** ✅ (not originally in §0; see §4.1)
Added as the `overhead` family while the plumbing was open, because §0.4's
small-size anomaly needed it to be ruled out. An empty dispatch costs
**~300 ns**, plus ~0.7 ns per workgroup scheduled (359 ns at 1 workgroup,
689 ns at 640, 45.5 µs at 65536). This **falsifies the §4.1 worry**: at
~500 dispatches per token, launch overhead is ~0.15 ms against a ~20 ms
token — negligible. Fusion (§3.1-3.2) is justified by DRAM round-trips, not
by launch cost. Caveat: this measures dispatch + pipeline barrier *within*
one command buffer, which is what the harness does; it does not measure
CPU-side `vkQueueSubmit` + fence wait, so §4.2's question is still open.

## 1. Decode path (GEMV / memory-bound) — the tok/s lever

### 1.1 W4A8: Q4 weights fed to `dotPacked4x8EXT` — **DONE** ✅, **2.4x**
**Hypothesis**: the best decode kernel reads 4-bit weights *and* uses the
packed-int8 dot instruction. Today those are mutually exclusive — Q4 goes
through the float dequant path and W8A8 reads 8-bit weights.

**[measured]** §0.3 turned this from an inference into a quantity. With
weights held DRAM-resident, Q4 GEMV moves only **74 of the available 236
GB/s — 31% of its own bandwidth ceiling**, while W8A8 reaches 73%. Q4 is
not bandwidth-bound at all; it is **ALU-bound in the unpack**, reading one
`uint8_t` at a time and doing a float multiply per nibble
(`gemv_subgroup.comp:46-53`). The bytes are already being saved; the kernel
just cannot consume them fast enough to benefit.
**Change**: new `gemv_w4a8.comp`. Load weights as `uint` (8 nibbles/word),
unpack to two packed-int8 words with bit tricks — roughly
`lo = (v & 0x0F0F0F0F) - 0x08080808`, `hi = ((v >> 4) & 0x0F0F0F0F) -
0x08080808` (~6 ALU ops for 8 weights) — then two `dotPacked4x8EXT` calls.
Accumulate in **int32 per block** and apply the scale once per block, not
per element.
**Expected**: Q4's DRAM-bound ceiling is **916 GFLOP/s**. Reaching W8A8's
73% bandwidth efficiency would put it at ~670 GFLOP/s — **2x the current
W8A8 champion (344)** — and reaching fp16's 75% would be ~690. Even a
partial fix that gets Q4 from 31% to 50% wins outright.
**Measure**: `gemv_cold` GB/s for q4 should climb from 74 toward 170+; the
cache-resident `gemv` row is the wrong one to watch, since it was never
bandwidth-limited.
**Effort**: medium. **Value**: **highest remaining item in this document** —
the only decode change with a measured 2x behind it.

**[measured] Built and run** — `shaders/gemv_w4a8.comp`, `bench/ops_w4a8.go`,
in both the warm `gemv` and DRAM-resident `gemv_cold` families. It is now
the fastest decode kernel in the suite by a wide margin. At the 256MB
fp16-equivalent footprint (M=32768, N=4096, block=128), which is the regime
real decode runs in:

| format | kernel | GFLOP/s | GB/s | % of its own DRAM ceiling |
|---|---|---|---|---|
| q4 | float unpack (old) | 283 | 73 | 31% |
| w8a8 | packed dot | 339 | 172 | 73% |
| **w4a8** | packed dot, `uint` loads | **698** | 180 | 76% |
| **w4a8** | packed dot, `uvec4` loads | **819** | **211** | **89%** |

**2.4x the previous champion** (W8A8's 339) and 2.9x the old Q4 path —
comfortably past the ~670 GFLOP/s this section predicted. That prediction
was too conservative because it assumed W4A8 would land at W8A8's *efficiency*
(73%); it actually reached **89% of the 236 GB/s DRAM bus**, the highest
utilisation any kernel in the suite achieves apart from the pure bandwidth
benchmark. Decode is now genuinely bandwidth-bound at 4 bits/weight, which
is the end state this whole section was aiming at: there is at most 11%
left in this kernel, and further decode gains have to come from reading
fewer bytes (finer-grained or lower-bit formats), not from better
arithmetic. Cache-resident (`gemv`, N=1024, block=128) it is 1326 GFLOP/s
against W8A8's 523 and Q4's 378; at the 64MB footprint, where the q4
weights still fit the MALL, it reaches 2538 GFLOP/s at 654 GB/s.

Four implementation notes, since two of them differ from the sketch above:

1. **The bit trick above does not work.** `(v & 0x0F0F0F0F) - 0x08080808`
   borrows across byte lanes: a nibble below 8 underflows its byte and
   corrupts the next one. There is no borrow-free per-byte subtract. The
   kernel instead leaves the nibbles *unsigned* (`u ∈ [0,15]`), feeds them
   to the **mixed-signedness** overload `dotPacked4x8EXT(int, uint)` — this
   device reports `integerDotProduct4x8BitPackedMixedSignednessAccelerated`,
   so it is one hardware instruction — and removes the bias afterwards with
   `sum (u-8)x = sum u*x - 8*sum x`. The per-block `sum x` is identical for
   every weight row, so it is computed once per GEMV (a fifth binding) and
   costs one short loop over `blocksPerRow`, not a single instruction in
   the inner loop. A real engine gets those sums for free from the
   activation-quantize pass.
2. **The weights are repacked offline** (`bench.repackQ4ToW4A8`). Masking a
   word of `quantizeQ4`'s interleaved layout (low nibble = even element)
   would yield lanes that need the *activations* permuted per token to
   match. Instead one word holds 8 consecutive weights with its 4 low
   nibbles carrying elements n..n+3 and its high nibbles n+4..n+7, so the
   two masked halves line up with two consecutive words of a plainly
   packed int8 activation vector — the same x layout `gemv_w8a8.comp`
   already consumes. This is a one-time cost on a model's weights, and it
   is a pure permutation, so the CPU reference is still built from
   `dequantizeQ4` on the pre-repack array (which checks the repack too).
3. **The ISA is what was intended** (`RADV_DEBUG=asm`, and one of §6.1's
   three questions answered early). The `VEC=4` inner loop is 3
   `buffer_load_b128` (32 weights + 8 activation words), one
   `buffer_load_d16_b16` for the fp16 scale, 8 `v_and_b32`/`v_lshrrev_b32`
   pairs splitting the nibbles, **8 chained
   `v_dot4_i32_iu8 ... neg_lo:[1,0,0] clamp`** — the mixed-signedness
   hardware dot, signed first operand, unsigned second, exactly as asked
   for — then one `v_cvt_f32_i32` and one `v_fma_mix_f32` that applies the
   fp16 scale without a separate convert. No integer division, no
   scalarized byte extraction. The W8A8 kernel by comparison emits
   `neg_lo:[1,1,0]` (both signed), one dot per iteration.
4. **Scale per load, not per block.** A load's worth of weights is required
   to sit inside one quantization block (`block >= weightsPerLoad`), so the
   int32 accumulator covers 8 or 32 weights and the fp16 scale is applied
   once per load. Block size barely matters (660–819 GFLOP/s across
   32/64/128, the larger blocks slightly ahead on scale traffic).

### 1.2 Kill the runtime integer divisions in the inner loop
**Hypothesis**: every quantized kernel does **two runtime integer
divisions per element** in its innermost loop. E.g.
`gemv_subgroup.comp:44`, `gemm_tiled.comp:55-56`, `gemm_naive.comp:54-55`,
`gemm_coopmat_q4.comp:44-45`: `pc.N / pc.block` and `n / pc.block`, plus
`idx / 2u` for Q4. Integer division by a non-constant has no hardware
instruction on AMD — the compiler emits a float-reciprocal sequence of
~10-20 instructions. This is likely a large part of why naive q4/q8
(445–536 GFLOP/s) is *slower than naive fp32* (521) despite reading 4-8x
fewer bytes.
**Change**: pass `log2(block)` (all block sizes swept are powers of two)
and use `>>`; better, make it a **specialization constant** so the shader
compiles with the shift folded in — `vk.PipelineSpec` already supports spec
constants. Hoist `blocksPerRow` out of the loop entirely (it's loop-invariant
but the compiler can't prove it cheaply through the buffer indexing).
**Expected**: large on the naive/tiled quantized paths; meaningful on GEMV
Q4. Possibly also flattens the mysterious block-size sensitivity.
**Measure**: dump ISA (§6.1) and confirm the `v_rcp_f32`/`v_mul_hi_u32`
division sequences are gone; then re-benchmark all quantized variants.
**Effort**: low. **Value**: high — cheapest real win in this document.

**[measured] Done for the one kernel that mattered most, still open for the
rest.** `gemv_w4a8.comp` takes `log2(block)` in its push constants and
shifts, and every other divisor in it is a compile-time power of two, so it
contains no integer division at all. That is bundled into §1.1's 2.4x and
was not measured in isolation. The naive/tiled GEMM paths and the existing
`gemv_subgroup`/`gemv_naive` quantized variants still divide twice per
element — and `gemv,naive,q4` is still the slowest kernel in the suite
(16 GFLOP/s at N=1024, *half* naive fp32's 32), which remains the strongest
circumstantial evidence for this item.

### 1.3 Vectorise the loads (uvec4 / 128-bit per lane)
**Hypothesis**: GEMV is issuing 32-bit loads where it could issue 128-bit
ones, wasting memory-instruction issue slots and not filling cache lines
per instruction. `gemv_w8a8.comp:52` loads one `uint` per lane per step;
`gemv_subgroup.comp` loads a single `float16_t` or `uint8_t`.
**Change**: read `uvec4` (16 weights for W8A8, 32 for W4A8) per lane per
step; for fp16 use `f16vec2`/`f16vec4` and keep the math in packed fp16
(`v_pk_fma_f16`, 2 FLOP/lane/clk) rather than converting to fp32 scalar.
**Expected**: 1.3-2x on the ALU-bound variants; smaller on already
bandwidth-saturated ones.
**[measured] The packed-fp16 half of this argument does not hold.** §0.1
measures packed f16vec2 multiply-add at 25.3 TFLOP/s against scalar fp32
FMA's 22.9 — **1.10x, not the 2x the packed ALU should give** (218 vs 198
ops/clk/CU). So "switch the math to f16vec2" is worth ~10%, not 2x. Worth
one ISA dump (§6.1) to find out whether `v_pk_fma_f16` is even being
emitted before investing here. The *load* half of the idea stands on its
own: wider loads reduce memory-instruction issue regardless of what ALU
consumes them, and §0.3 showed every GEMV variant stuck at 50-75% of its
bandwidth ceiling.
**Effort**: low-medium. **Value**: medium (downgraded from high).

**[measured] The load half is confirmed, and it is worth ~15%.**
`gemv_w4a8.comp` compiles in two variants off one source, `VEC=1` (one
`uint`, 8 weights per lane per step) and `VEC=4` (one `uvec4`, 32 weights),
which differ in nothing else. Against DRAM-resident weights the wide loads
are **1.17x** (698 → 819 GFLOP/s, 180 → 211 GB/s at block=128) and
cache-resident **1.10x** (1204 → 1326 GFLOP/s at N=1024). That is the
difference between 76% and 89% of the DRAM ceiling — i.e. most of what was
left in the access pattern was memory-instruction issue, exactly as this
item argued. Worth repeating on the fp16 and W8A8 GEMV kernels, which are
still at 75% and 73%. The packed-fp16 *math* half remains unattractive
(1.10x on the ALU peak) and untested.

### 1.4 Multiple output rows per workgroup
**Hypothesis**: one workgroup (one wave64) per output row means the
activation vector `x` is re-read from cache by all 4096 workgroups, and
each wave does a full `subgroupAdd` tree for a single scalar output.
**Change**: have each workgroup compute R=4-8 rows, holding `x`'s tile in
registers/LDS once and reusing it across rows; emit R results per
`subgroupElect`.
**Expected**: R-fold reduction in activation traffic and 1/R the reduction
overhead. **[measured]** §0.3 gives this a target: no GEMV format exceeds
75% of its DRAM bandwidth ceiling, so ~25-50% is sitting in the access
pattern, and this plus §1.3 are the two candidates for it.
**Effort**: medium. **Value**: medium. **[measured]** §1.3's wide loads
took W4A8 to 89% of its ceiling on their own, leaving this only ~10% to
chase on the kernel that matters; it stays interesting for the other
formats and for the small-M batched shapes in §1.5.

### 1.5 Find the GEMV→coopmat crossover for small batch
**Hypothesis**: for M=2..16 (speculative decoding, beam search, batched
serving) padding M up to 16 and using the WMMA path beats M separate GEMV
dispatches, even though up to 15/16 of the MMA work is wasted — because
the MMA path has far more throughput available (**[measured]** 55.5
TFLOP/s of WMMA ceiling, and even today's unoptimised coopmat GEMM does
4.9 TFLOP/s against the best DRAM-resident GEMV's 0.34). §0.5's ~300 ns dispatch cost
also means the "M separate dispatches" alternative is not penalised by
launch overhead, so this is a pure work-efficiency question.
**Change**: benchmark `gemm_coopmat_*` and `gemv_*` at M = 1,2,4,8,16,32,64
with K=N=4096, and find the crossover.
**Expected**: a concrete batch threshold for the engine's scheduler to
switch kernels at. This is exactly the decision llama.cpp encodes as
separate `mul_mat_vec` vs `mul_mm` kernels.
**Effort**: low (it's a shape sweep over existing kernels).
**Value**: high for the serving story in `GOALS.md`.

### 1.6 Measure the activation-quantization cost that W8A8 currently hides
**Hypothesis**: W8A8's headline number excludes work a real engine must do
per token. `bench/ops_w8a8.go:65` quantizes `x` **on the host, once,
outside the timed loop**. In production, every decode step must quantize
the activation vector on-GPU (a pass over `x`: max-abs reduction, then
scale-and-pack) between every pair of layers.
**Change**: time the full chain RMSNorm → quantize-to-int8 → GEMV, and
compare against RMSNorm → fp16 GEMV. Then §3.1 fuses it away.
**Expected**: reveals W8A8's true cost. On small vectors the extra
dispatch's launch + barrier overhead may dominate the quantize itself.
**Effort**: low. **Value**: high — needed to fairly compare W8A8 vs Q4.

---

## 2. Prefill / batch path (GEMM) — the ~90%-idle matrix cores

### 2.1 Register-block the coopmat kernels — **DONE** ✅, **6.0x**
**Hypothesis**: `gemm_coopmat_fp16.comp` is memory-bound by construction.
One workgroup = one wave64 = **one** 16×16 accumulator, and the K-loop
loads A and B straight from global memory every iteration
(`gemm_coopmat_fp16.comp:46-48`). Arithmetic intensity: 2·16·16·K flops per
(16K + 16K)·2 bytes = **8 FLOP/byte**. **[measured]** Saturating the
measured 55.5 TFLOP/s at 236 GB/s needs **235 FLOP/byte**; the kernel
supplies 8. The measured 4.2-4.9 TFLOP/s is exactly what an 8 FLOP/byte
kernel gets with MALL help — the MMA units are starved.

The original supporting argument for this (int8 coopmat having 2x the
hardware rate yet matching fp16) was **wrong**: §0.1 shows fp16 and int8
WMMA are the same speed here. But the conclusion is now on far firmer
ground, because the ceiling is no longer extrapolated: **4.9 of 55.5
TFLOP/s, measured on this device, is 9%.**
**Change**: the standard tiled-WMMA structure:
- 4 waves per workgroup (256 threads), workgroup tile 128×128.
- Each wave holds a **2×4 or 4×4 grid of accumulator coopmats** (8-16
  accumulators) → AI rises to 32-64 FLOP/byte.
- Stage A and B K-slabs into LDS **once per workgroup** (65536 B of LDS
  available per `maxComputeSharedMemorySize`), then `coopMatLoad` from LDS
  — so global traffic is amortised across all 4 waves and all accumulators.
- Double-buffer the LDS slabs so there's one `barrier()` per K-step
  instead of the current two, and loads overlap MMA.
**Expected**: **2-4x** over the current 4.2-4.9 TFLOP/s, i.e. 10-20
TFLOP/s against a 55.5 TFLOP/s ceiling. This is the single largest absolute
gain available on the chip.

**[measured] Built and run** — `shaders/gemm_wmma.comp` + `bench/ops_gemm_wmma.go`,
twelve variants off one source forming an ablation over four axes
(accumulator grid, waves per workgroup, LDS staging, double buffering, B
layout). It beat the prediction: **25.2 TFLOP/s at N=4096, 6.0x the
baseline coopmat kernel at the same shape and 45% of the measured WMMA
ceiling**, up from 8%. (Against the baseline's own best figure anywhere in
the sweep — 5.4 TFLOP/s at N=1024 — it is 4.6x.) At N=4096, with implied
A+B traffic computed as GFLOP/s ÷ arithmetic intensity so each row can be
placed against the 805 GB/s MALL and 236 GB/s DRAM ceilings:

| kernel | tile | waves/wg | AI | GFLOP/s | % of 55.5 T | implied traffic |
|---|---|---|---|---|---|---|
| `coopmat` (baseline) | 16×16 | 1 | 8 | 4190 | 7.5% | 524 GB/s |
| `wmma_reg32` | 32×32 | 1 | 16 | 12233 | 22% | **765 GB/s** |
| `wmma_reg64` | 64×64 | 1 | 32 | 24276 | 44% | **759 GB/s** |
| **`wmma_reg64x128`** | 64×128 | 1 | 43 | **24458** | **44%** | 573 GB/s |
| `wmma_wg128` | 128×128 | 4 | ≤64 | 23409 | 42% | ≤366 GB/s |
| **`wmma_wg128x256`** | 128×256 | 4 | ≤85 | **25245** | **45%** | ≤296 GB/s |
| `wmma_lds128` | 128×128 | 4 | 64 | 17163 | 31% | 268 GB/s |
| `wmma_lds128_db` | 128×128 | 4 | 64 | 21223 | 38% | 332 GB/s |
| `wmma_lds128_db_bt` | 128×128 | 4 | 64 | 12944 | 23% | 202 GB/s |
| `wmma_lds256x128_db_bt` | 256×128 | 4 | 85 | 12716 | 23% | 149 GB/s |
| `wmma_reg64_bt` | 64×64 | 1 | 32 | 8543 | 15% | 267 GB/s |

The `wg*` rows' intensity is an upper bound rather than a fact: their four
waves issue overlapping fragment loads and only reach it if L0/L1 dedups
them, which is what those rows exist to test.

Four findings, two of which contradict the plan above.

1. **Arithmetic intensity was exactly the right variable, and the returns
   are linear once the MALL is saturated.** AI 8 → 16 → 32 gives 4.19 →
   12.2 → 24.3 TFLOP/s, and the traffic column explains the shape of that:
   at AI 16 and 32 the kernel is pinned at **765 and 759 GB/s** (766-814
   GB/s across the sizes swept), i.e. at the measured 805 GB/s MALL
   bandwidth, so the 16 → 32 step is *exactly* 2.0x — a doubling of reuse
   buying a doubling of throughput out of a fixed budget. The 8 → 16 step is
   2.9x rather than 2x because the baseline's 524 GB/s shows it was not even
   bandwidth-saturated: with one accumulator per wave there is no
   independent work to hide load latency behind, so it lost twice over.
2. **Getting rid of LDS staging is the win; grouping waves is neither here
   nor there.** The top two configurations tie within run-to-run noise —
   `wmma_reg64x128` (one wave per workgroup, no shared memory) and
   `wmma_wg128x256` (four waves, no shared memory, no barriers) — while
   *every* LDS variant loses, the best of them by 1.2x. And the LDS variants
   are nowhere near bandwidth-bound (149-332 GB/s of a 805 GB/s budget);
   they are bound by their own shared-memory reads and barriers. So **the 32
   MiB MALL is already supplying the cross-wave reuse that LDS staging
   exists to provide**, and explicitly staging it just adds barriers and
   `ds_load`s on top. That is an APU-shaped result: on a discrete part with a
   small L2 this ordering would very likely invert.
3. **Double buffering does work — 1.24x — on the path that loses anyway.**
   `wmma_lds128` → `wmma_lds128_db` is 17.2 → 21.2 TFLOP/s for one barrier
   per K-step instead of two, with the next slab's global loads issued
   before the current slab's MMAs (the compiler sinks the `s_waitcnt` to the
   first use, so the latency is spent under the matrix work). The mechanism
   is confirmed; it is just not enough to make LDS competitive here.
4. **The occupancy floor was never the constraint.** §0.1's worry was that
   8-16 accumulators per wave would drop below the 2 waves/CU WMMA needs.
   `RADV_DEBUG=shaderstats` says 144 VGPRs for the 16-accumulator variants
   and 252 for the 32-accumulator ones, **no spills and no scratch** in any
   of them — 3-5 waves per SIMD, comfortably above the floor. Register
   blocking on this part is limited by the 256-VGPR wave64 cap, not by
   occupancy, which makes §6.2 (wave32) more interesting than it was.

**What binds it at 25 TFLOP/s** is neither ceiling: 573 GB/s of a 805 GB/s
MALL and 45% of a 55.5 TFLOP/s MMA rate. The instruction mix names the
likely culprit — `wmma_reg64`'s loop body is **577 instructions for 16
`v_wmma_f32_16x16x16_f16`**, of which 232 are pure address arithmetic
(`v_lshlrev_b32` + `v_add3_u32`) feeding the scalarized B-fragment loads
described in §2.3. At roughly 34 SIMD-cycles per wave64 WMMA, ~36 VALU
instructions per MMA is the same order as the matrix work itself, so the
address math plausibly co-issues into the MMA shadow and then runs out of
room. The fix is exactly what §2.3's B layout does to the instruction
stream — 389 instructions and 144 address ops instead of 577 and 232 — and
it costs 2.8x in memory behaviour instead of winning. Closing that
contradiction is the highest-value follow-up in this section.
**Effort**: high (this is a real GEMM kernel). **Value**: highest for
prefill, image generation, and the Parakeet encoder. **Delivered.**

### 2.2 W4A8 coopmat — Q4 weights into the int8 MMA path — **downgraded**
**Hypothesis (as written, now falsified in part)**: that feeding Q4 into
the int8 MMA path would get both 1/4 the weight bytes *and* the 2x int8
matrix rate. **[measured] There is no 2x int8 matrix rate** — §0.1 measures
fp16 and int8 WMMA at 479 and 480 ops/clk/CU respectively. The apparent 2x
in the old data (`coopmat,q8` at 9.6 TOPS vs `coopmat,fp16` at 4.4 TFLOP/s
at N=512) was a *memory* effect: int8 reads half the bytes of fp16, and
both kernels are memory-bound. Consistent with §2.1, not with an MMA-rate
difference.
**What remains worth doing**: the unpack itself is cheaper into int8 than
into fp16 (mask/subtract, no float conversion), and the LDS tile is half
the size, which helps occupancy — but the payoff is now "somewhat cheaper
tile fill", not 2x. **Do §2.1 first**; once the kernel is
arithmetic-intensity-bound rather than memory-bound, re-measure whether the
tile-fill cost is even on the critical path before writing this.
**[measured] §2.1 is now done, and it re-aims this item.** The winning
kernel has no LDS tile at all, so "the LDS tile is half the size" is moot;
and the remaining gap to the ceiling looks like address-arithmetic and
load-issue pressure rather than operand-fill cost. What §2.1 *does* make
newly attractive is the plain **int8 register-blocked GEMM** — a
`PRECISION_I8` arm of `gemm_wmma.comp` accumulating in int32, which is a
small change to a kernel that now reaches 45% of the ceiling and would halve
its operand bytes. That, not the Q4-unpack version, is the next step here.
**Effort**: low for the int8 arm, high for the Q4 unpack.
**Value**: medium-high for int8, medium for Q4 (downgraded from very high).

### 2.3 B stored [N,K] with a column-major `coopMatLoad` — **DONE**, and it **loses** ❌
**Hypothesis**: B is stored K-major ([K,N]) and loaded row-major, but real
`Linear` weights are stored `[out_features, in_features]` = [N,K], and
RDNA's WMMA B-operand lane layout may favour the transposed load. The
W8A8 GEMM path already stores B transposed for packing reasons — the
coopmat paths don't.
**Change**: store B as [N,K] and load with
`gl_CooperativeMatrixLayoutColumnMajor`; benchmark both.
**Expected**: unclear sign, but it's nearly free to test and removes a
host-side transpose from the engine's weight loader if it wins.

**[measured] The codegen half of the hypothesis is emphatically right and
the performance half is emphatically wrong.** Tested as the `_bt` variants
of `gemm_wmma.comp` (§2.1), which differ from their pairs in nothing else.

RADV's `coopMatLoad` has a strong layout preference, isolated with a
four-case probe (`cmd/probe` + `RADV_DEBUG=asm`, one fragment load per case):

| operand | layout | instructions emitted |
|---|---|---|
| `gl_MatrixUseA` | RowMajor | 2 × `buffer_load_b128` ✅ |
| `gl_MatrixUseA` | ColumnMajor | 16 × `buffer_load_d16_b16` |
| `gl_MatrixUseB` | RowMajor | 16 × `buffer_load_d16_b16` |
| `gl_MatrixUseB` | **ColumnMajor** | 2 × `buffer_load_b128` ✅ |

So the hardware WMMA operand layout is K-contiguous for *both* operands,
and a [K,N] row-major B has to be gathered element by element — which is
what `gemm_coopmat_fp16.comp` has been paying on every fragment all along.
Storing B [N,K] removes it: `wmma_reg64`'s loop body goes from **577
instructions (8 wide loads + 64 scalar 16-bit loads, and 232 address-
arithmetic ops to compute them) to 389 (16 wide loads, 144 address ops)**,
for the same 16 `v_wmma`.

**And it is 2.8x slower at N=4096** — 8543 vs 24276 GFLOP/s — across every
tiling tried: the register-blocked pair, the LDS pair
(`wmma_lds128_db` 21223 → `wmma_lds128_db_bt` 12944), and the 256×128 tile.
The penalty is size-dependent and appears exactly when B stops being
cache-resident: at N=512 `_bt` is *faster* (14265 vs 12838), at N=1024 it
is 2.0x slower, at N≥2048 it is 2.5-2.8x slower. So this is a memory-system
effect on a strided gather, not an instruction-count effect — the
instruction count moves the *other* way.

One mechanism was tested and **refuted**: that a column-major B fragment
consumes only 32 of each 128-byte cache line and depends on the line
surviving three more K-steps. Deepening the K-slab to 64 so one MMA block
consumes a full line (`wmma_reg64_bt_k64`) changes nothing — 8583 vs 8543.
**Still untested, and the obvious next probe**: channel/bank aliasing from
the power-of-two leading dimension. A column-major fragment load issues 16
addresses `K·2` bytes apart *within one instruction*, and every K swept here
is a power of two; padding the leading dimension to break the alias would
distinguish this cleanly, and needs a separate stride push constant since
the kernel currently reuses `pc.K` as both extent and stride.
**Effort**: low. **Value**: high but blocked on the above — §2.1 identifies
this layout as the way to halve its address-arithmetic overhead, so the
contradiction is worth resolving rather than filing.

### 2.4 Workgroup swizzle / tile reordering for MALL locality
**Hypothesis**: linear `gl_WorkGroupID` ordering walks C row-by-row, so
concurrently-resident workgroups share A rows but stream all of B — poor
reuse in the 32MB MALL. Grouped/Morton ("super-tile") ordering makes the
in-flight set of workgroups a square block of C, maximising shared A and B.
**Change**: remap `gl_WorkGroupID.x/y` through a swizzle in-shader (group
tiles into e.g. 8×8 super-tiles); pure index arithmetic, no structural
change.
**Expected**: 10-30% on large N, more once §2.1 makes tiles bigger. A
standard, well-documented GEMM win.
**[measured] §2.1 promotes this.** Its best kernel is pinned at ~760 of the
805 GB/s the MALL delivers at AI 16 and 32, so it is *literally* MALL-
bandwidth-bound, and a swizzle is the one remaining change that reduces MALL
traffic without touching the tile shape or the register budget. §2.1 also
showed that neither LDS staging nor grouping waves into a workgroup buys
anything, which means all the cross-tile reuse this kernel gets is already
coming from the MALL — exactly the resource a swizzle reorders. It is now
the cheapest untried item with a measured constraint behind it.
**Effort**: low. **Value**: high (upgraded).

### 2.5 Split-K for skinny shapes
**Hypothesis**: real transformer GEMMs aren't square. When M·N is small
but K is large (e.g. the attention output projection at short sequence
length), there aren't enough tiles to fill 40 CUs, and the current kernels
under-occupy.
**Change**: partition K across G workgroups, each producing a partial C;
combine via a second reduce dispatch (or `atomicAdd` on fp32 C).
**Expected**: large on shapes where the tile count is below ~2x CU count.
**Measure**: needs §3.4's realistic shape sweep to know which shapes matter.
**Effort**: medium. **Value**: medium, shape-dependent.

### 2.6 fp16 output, and fuse epilogue work
**Hypothesis**: every GEMM writes `float c[]` (fp32), doubling store
traffic versus the fp16 the next layer wants, and every fused op (bias,
activation, residual add, the next layer's quantize) is a separate
full-round-trip dispatch through DRAM.
**Change**: an fp16-output variant, plus an epilogue hook in the GEMM
kernel (bias + activation + optional int8-quantize of the result).
**Expected**: ~M·N·2 bytes saved per GEMM, and one whole DRAM round-trip
per fused op eliminated — at 236 GB/s that's directly measurable.
**Effort**: low-medium. **Value**: medium-high.

---

## 3. Kernel fusion and the ops the suite doesn't cover yet

### 3.1 Fused RMSNorm → int8 quantize
Pairs with §1.6. One kernel that reads `x`, computes the RMS, and writes
*both* the normalised fp16 and the packed-int8 + scale. Eliminates a full
read+write of the activation vector and one dispatch+barrier per layer.
For W8A8 to be viable this fusion is mandatory, not an optimisation.
**Effort**: low. **Value**: high.

### 3.2 Fused SwiGLU / gated FFN
`silu(gate) * up` is currently three DRAM round-trips (two GEMM outputs
read, one written). Fuse into the GEMM epilogue (§2.6) or at minimum into
a single elementwise kernel taking both operands. Elementwise already hits
617 GB/s cache-resident / 236 GB/s DRAM — so these passes are pure
bandwidth and the only win is *not doing them*.
**Effort**: low. **Value**: medium-high.

### 3.3 Attention is completely absent from the suite — add it
**Gap**: there is no attention kernel at all. For decode at long context,
attention over the KV cache is the *dominant* memory consumer, more than
the weights. For prefill it's a big chunk of the FLOPs.
**Change**: benchmark (a) naive multi-dispatch attention — QKᵀ GEMM,
softmax, PV GEMM — versus (b) a fused flash-attention-style kernel with
online softmax keeping the K-tile in LDS and using coopmat for both
matmuls. Sweep sequence length 128…8192, plus GQA group sizes.
**Expected**: the standard flash-attention result is a large win, and it's
amplified here because the standalone `softmax` kernel measures a poor
**104 GB/s** — the fused version never materialises the score matrix at all.
**Effort**: high. **Value**: very high — it's a whole missing pillar.

### 3.4 Benchmark the *real* shapes from the target models
**Gap**: the sweep is square N×N×N, which no transformer layer is.
**Change**: pull the actual dims from the four models in `GOALS.md`
(Qwen3-Next hidden/FFN/expert dims and head config; Parakeet's conformer
encoder; Kokoro; Z-Image) and sweep those exact (M,N,K) triples plus
decode (M=1) and prefill (M=512/2048) variants.
**Expected**: reveals which kernels matter and which shapes are
pathological (odd dims, non-multiples of 16 needing padding/tail handling
— none of the current kernels handle a K that isn't a multiple of TILE_K,
`gemm_coopmat_fp16.comp:44` silently truncates `pc.K / TILE_K`).
**Effort**: low-medium. **Value**: high — refocuses all other work.

### 3.5 Grouped / MoE GEMM
Qwen3-Next is an MoE model: each token goes to a few of many experts, so
the FFN is a **grouped GEMM** over variable-sized token batches, not one
big GEMM. Benchmark: top-k routing, the gather/scatter of token rows, and a
grouped-GEMM kernel that reads per-expert (offset, count) from a buffer.
Also measure the "one dispatch per expert" naive alternative to quantify
the launch-overhead penalty (§4.1).
**Effort**: high. **Value**: high, and specific to the stated text-gen goal.

### 3.6 Gated DeltaNet / linear-attention chunk kernel
Qwen3-Next mixes linear attention with full attention. The chunked
linear-attention recurrence is a different primitive (chunk-local matmuls
plus a sequential state update) that nothing here covers. Worth a
prototype to find out whether it's matmul-bound (good — coopmat) or
serialised on the state update (bad — needs careful chunking).
**Effort**: high. **Value**: high but only for Qwen3-Next.

### 3.7 Fix the reduction kernels — the `subgroup` variants are 3.3x slower
**[measured]** Re-running these warmed changes the picture from what the
original data suggested. At N=4096, against the 236 GB/s DRAM ceiling:

| kernel | GB/s | % of DRAM ceiling |
|---|---|---|
| `rmsnorm,shared` | 224 | **95%** — essentially optimal, nothing to win |
| `softmax,shared` | 157 | 67% |
| `rmsnorm,subgroup` | 68 | 29% |
| `softmax,subgroup` | 48 | 20% |

So the original framing ("both are 2-3x off achievable bandwidth") was
wrong about the shared-memory RMSNorm, which is already at 95% of the bus.
Two narrower findings replace it:
1. **The `subgroup` variants are 3.3x slower than their `shared`
   counterparts at every size.** They use one wave64 per row, so a row is
   reduced by 64 lanes doing scalar loads — latency-bound, not
   bandwidth-bound. Either rewrite them as one 256-thread workgroup doing
   `uvec4` loads → per-subgroup `subgroupAdd` → a 4-element LDS combine, or
   delete them; as they stand they are a strictly worse option that the
   engine should never pick.
2. **Softmax is 1.4x off RMSNorm's efficiency** (67% vs 95%) despite being
   the same shape of traversal, because it makes two passes (max, then
   exp-sum) plus a normalise. An online/single-pass softmax would close
   most of that — and in the fused attention kernel (§3.3) it disappears
   entirely.
**Effort**: low. **Value**: medium — small ops, but the subgroup gap is
large and the fix is mechanical.

## 4. Dispatch overhead and engine-level plumbing

### 4.1 Per-dispatch launch + barrier cost — **DONE**, worry **falsified** ✅
Measured as the `overhead` family (§0.5). A dispatch plus the pipeline
barrier between iterations costs **~300 ns**, plus ~0.7 ns per workgroup
scheduled. The feared 20 µs would have capped decode throughput on its own;
300 ns does not — ~500 dispatches per token is ~0.15 ms against a ~20 ms
token, under 1%.
**Consequence**: kernel fusion (§3.1-3.2) should be justified by the DRAM
round-trips it removes, *not* by launch overhead, and §4.2/§4.3 drop in
priority accordingly. **Still open**: this measures dispatch + barrier
*inside* a command buffer. CPU-side `vkQueueSubmit` + fence wait is not
measured, and a real engine submits per token (or per layer), so that cost
— which §4.2 would remove — is the one still worth a number.

### 4.2 Pre-recorded command buffers for the decode step
**Hypothesis**: a decode step is the same dispatch graph every token, so it
should be recorded **once** and resubmitted, with only push constants (or
better, a descriptor-indexed buffer of per-step params) changing. The
current harness rebuilds the command buffer per call.
**Change**: record a whole-layer or whole-model command buffer; drive
per-step variation through buffer contents rather than re-recording.
**Expected**: removes the CPU-side submit/record cost §4.1 did *not*
measure. Measure that cost first — if it is also sub-microsecond, this is
unnecessary.
**Effort**: medium (engine change, not a shader). **Value**: medium
(downgraded — the GPU-side half of the concern is now ruled out).

### 4.3 Barrier granularity
Full `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` memory barriers between every
dispatch (`vk/shim.c:511-521`) drain the pipeline. Many adjacent ops in a
layer are independent (e.g. Q, K, V projections) and could run
concurrently with no barrier at all. Experiment: measure 3 independent
GEMMs with vs without barriers between them; expect meaningful overlap
gains on small shapes where a single GEMM can't fill the GPU.
**Effort**: low. **Value**: medium-high for decode.

---

## 5. APU-specific memory experiments (this is the unusual part of the chip)

### 5.1 Which memory type is fastest for GPU-read-only weights?
**Hypothesis**: the allocator currently prefers `DEVICE_LOCAL|HOST_VISIBLE`
(memory type 3/4 here) for its unified-memory convenience, but on an APU
that choice usually means host-coherent/uncached-write-combine pages, which
can read *slower* from the GPU than plain `DEVICE_LOCAL`. Given the whole
decode path is bandwidth-bound, a 10% difference here is a 10% tok/s
difference. This device exposes 11 memory types across 2 heaps:
- types 0,1: `DEVICE_LOCAL` only (heap 1, 83.8 GiB)
- type 2: `HOST_VISIBLE|HOST_COHERENT` (heap 0, 41.9 GiB)
- types 3,4: `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` ← **what we use now**
- types 5,6: `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` (heap 0)
- types 7-10: the above plus AMD's `DEVICE_COHERENT`/`DEVICE_UNCACHED`
  (`VK_AMD_device_coherent_memory` is supported here)
**Change**: parameterise the bandwidth benchmark by memory type index and
run the 64MB copy from each.
**Expected**: a ranked table. Plausibly `DEVICE_LOCAL`-only wins for
weights, with a staging upload at load time — which is a small change to
the weight loader for a possibly free few-percent on everything.
**Effort**: low. **Value**: high (bandwidth is the binding constraint).

### 5.1b Exploit the 3.4x MALL cliff deliberately
**[measured]** §0.4 pinned the last-level cache at exactly **32 MiB
delivering ~805 GB/s**, against 236 GB/s from DRAM. That is a 3.4x
bandwidth difference available to any kernel whose working set can be kept
under 32 MiB — and it is a *hard* cliff, not a gradual falloff (32 MiB:
804 GB/s, 48 MB: 235 GB/s).
**Change**: treat 32 MiB as a first-class budget in the engine's
scheduling. Candidates: tile attention so the K/V slab per pass stays under
it (§3.3); size MoE expert groups so one expert's Q4 weights plus
activations fit; for prefill, block the GEMM's B panel to a sub-32MiB
strip and sweep all tiles of A against it before moving on (this is §2.4's
swizzle, but sized against a now-known number rather than guessed).
**Measure**: the `gemv_cold` footprint sweep already shows the transition
for a real kernel — the same sweep shape applied to a blocked GEMM would
show whether the blocking holds.
**Effort**: varies by kernel. **Value**: high, and it multiplies with
everything in §2 and §3.

### 5.2 Does the BIOS VRAM carveout matter?
Heap 1 reports 83.8 GiB device-local on a machine with 117 GiB of usable
RAM (and heap 0 another 41.9 GiB) — the two heaps overlap the same physical
memory, so the driver is largely handing out GTT rather than a fixed
carveout.
Experiment: measure bandwidth for buffers small enough to fit a
conventionally-sized carveout vs much larger, and check whether page-size
effects appear. Also test whether huge/2MB pages are in play. Informs how
to lay out a multi-GB model.
**Effort**: medium. **Value**: medium, potentially high for big models.

### 5.3 Image/texture loads vs storage-buffer loads
On AMD, sampler/image reads go through a different path (and on some parts
a different cache) than buffer loads. For read-only weights, a
`readonly image2D`/texel-buffer path is worth one measurement. Long shot,
but cheap and occasionally a surprise win.
**Effort**: low. **Value**: low-medium (lottery ticket).

---

## 6. Occupancy, wave size, and looking at the actual machine code

### 6.1 Dump and read the ISA
`RADV_DEBUG=asm` (verify option names with `RADV_DEBUG=help`) dumps the
generated GCN/RDNA assembly. Use it to confirm:
- `dotPacked4x8EXT` really became `V_DOT4_I32_I8` (this is the whole point
  of the W8A8 path, and the extension was silently *not enabled* until
  last session's `vk/shim.c` fix — worth verifying the fix took effect at
  the instruction level).
- `coopMatMulAdd` really became `V_WMMA_*` and not a scalar fallback.
- The integer-division sequences of §1.2 exist (and then disappear).
- **VGPR/SGPR counts and LDS usage per kernel** → waves/SIMD occupancy.
**[measured] Done, and it is now a tool rather than a one-off.** `cmd/probe`
dispatches any single `.spv` once so `RADV_DEBUG=asm` (disassembly) and
`RADV_DEBUG=shaderstats` (VGPR/SGPR/LDS/spills) can be pointed at it.
Answers so far:
- `dotPacked4x8EXT` → `v_dot4_i32_iu8 ... neg_lo:[1,0,0]`, the
  mixed-signedness hardware dot, in W4A8 (§1.1); `neg_lo:[1,1,0]` in W8A8.
- `coopMatMulAdd` → `v_wmma_f32_16x16x16_f16`, no scalar fallback, and
  subgroup size is **64** even in a coopmat shader, so a 256-thread
  workgroup is 4 waves.
- VGPR counts: 144 for a 16-accumulator wave, **252** for 32 accumulators,
  no spills or scratch in either — §2.1's occupancy worry was unfounded.
- **And one finding nothing else would have surfaced**: `coopMatLoad` emits
  2 `buffer_load_b128` for a row-major A or column-major B operand and
  **16 scalar `buffer_load_d16_b16`** for the other two combinations. That
  is §2.3, and it is the single most useful thing reading the ISA has
  produced. `v_pk_fma_f16` (§1.3's 1.10x surprise) is still unchecked.
**Effort**: low. **Value**: high — it converts speculation into fact, and
every item above gets cheaper to evaluate once we can read the ISA.

### 6.2 wave32 vs wave64
**Hypothesis**: this device reports `minSubgroupSize=32`,
`maxSubgroupSize=64`, `subgroupSizeControl=true` with compute in
`requiredSubgroupSizeStages` — so we can *choose*. Everything is currently
written for wave64 (`local_size_x = 64`). RDNA's WMMA 16×16×16 shape maps
naturally onto wave32, and wave32 halves the latency of a dependent
instruction chain and reduces divergence cost; wave64 halves instruction
issue count. Which wins is empirical and differs per kernel.
**Change**: use `VK_EXT_subgroup_size_control`'s
`requiredSubgroupSize=32` pipeline creation flag and re-benchmark the
coopmat GEMMs and the subgroup GEMV/reduction kernels at both sizes.
**Expected**: 0-20% either way, per kernel. Cheap to test, and it's a
per-pipeline knob we'd want to tune once and hard-code.
**Effort**: low-medium (small `vk/shim.c` change). **Value**: medium-high.

### 6.3 Occupancy tuning via workgroup size and LDS budget
Once §6.1 gives VGPR counts, sweep workgroup sizes (64/128/256/512) and
LDS tile sizes for the tiled and coopmat kernels, and check where occupancy
cliffs are. `gemm_tiled.comp` uses TILE×TILE = 16×16 threads with two
fp32 LDS tiles; 65536 B of LDS allows far more. Note `gemm_tiled.comp:74`
stores dequantized Q4/Q8 into **fp32** LDS — using fp16 LDS tiles halves
LDS use and doubles the occupancy headroom for free.
**Effort**: low-medium. **Value**: medium.

### 6.4 Power and sustained-clock behaviour
The iGPU shares a TDP budget with 16 Zen5 cores. Every number in
`results.csv` is from a short kernel on an otherwise-idle machine.
Experiment: run a 60-second sustained GEMM loop while logging
`freq1_input`/`power1_average`, both with the CPU idle and with the CPU
loaded — then report GFLOP/s/W and the sustained-vs-burst ratio. A server
doing tokenisation/sampling on the CPU while the GPU runs will see the
degraded number, not the benchmark number.
**Effort**: low. **Value**: medium-high for the real serving goal.

---

## 7. Accuracy work that has to happen before any of this ships

Perf work is choosing between formats whose *accuracy* is unmeasured — the
`TODO.md` working conclusion already flags this. None of these are perf
experiments, but they gate the format decision:

- **Per-format error metrics**: for each of Q8/Q4/W8A8/W4A8 and each block
  size, measure RMS and max relative error of a full layer's output against
  an fp32 reference — the harness already builds CPU references, so this is
  mostly bookkeeping. This gives an accuracy-vs-GFLOP/s Pareto front
  instead of a speed ranking.
- **Asymmetric (zero-point) vs symmetric Q4**: the current scheme is
  symmetric `[-8,7]` (`gemm_coopmat_q4.comp:43`). Asymmetric costs one more
  value per block and a correction term but typically halves the error;
  worth measuring both cost and benefit.
- **Activation outliers**: W8A8's weak point is per-tensor/per-row dynamic
  activation quantization in the presence of outlier channels. Measure how
  bad it is on real activations before committing to W8A8 for decode.
- **Real perplexity**: ultimately the only test that matters. Needs a
  model loaded, so it comes after the engine exists — but it should be the
  acceptance criterion for the format choice, not benchmark GFLOP/s.

---

## Suggested order of attack

**§0 is done** — the harness now measures its own ceilings, records the
clock behind every number, and has a DRAM-resident decode benchmark
(`peak`, `overhead`, `gemv_cold` families). **§1.1 is done** — W4A8 GEMV at
819 GFLOP/s / 211 GB/s, 89% of the DRAM bus, so decode is finished as a
kernel problem. **§2.1 is done** — the register-blocked WMMA GEMM at 25.2
TFLOP/s, 45% of the matrix cores, so prefill is no longer the gaping hole
either. What those three left behind:

**Next (resolve the contradiction §2.1 ran into):**
- **§2.3, why the better instruction stream is slower.** Storing B [N,K]
  and loading it column-major cuts `wmma_reg64`'s loop from 577 instructions
  to 389 and is 2.8x slower at N=4096, only once B stops being
  cache-resident. §2.1's remaining gap to the ceiling looks like exactly the
  address-arithmetic pressure that layout removes, so this is the one place
  where a single answer unblocks the largest kernel on the chip. The
  untested mechanism is channel/bank aliasing from the power-of-two leading
  dimension; padding it needs a stride push constant, which is an afternoon.
- **§2.4 workgroup swizzle.** Upgraded to high value: §2.1 is pinned at ~760
  of the MALL's 805 GB/s, and a swizzle is the only remaining change that
  reduces MALL traffic without touching the tile shape or the register
  budget. Cheapest item in the document with a measured constraint behind it.
- **§2.2's int8 arm** — a `PRECISION_I8` variant of `gemm_wmma.comp`
  accumulating in int32. Small change to a kernel that now works, halves the
  operand bytes, and gives the W8A8 prefill story its number. (The *Q4*
  unpack version stays downgraded: there is no int8 matrix-rate bonus, and
  §2.1's winner has no LDS tile to make cheaper.)

**Then (cheap, and still untouched):**
- **§1.2 remove the runtime integer divisions** — still the cheapest real
  win on everything W4A8 didn't rewrite (the naive/tiled GEMM paths and the
  old quantized GEMV variants, where `gemv,naive,q4` is still slower than
  naive fp32).
- **§1.3 wide loads on the remaining GEMV kernels** — measured at 1.17x on
  W4A8; fp16 and W8A8 GEMV are still at 75% and 73% of their ceilings.
- **§5.1 memory-type bandwidth** — untouched, cheap, and bandwidth is the
  binding constraint on both of the paths that now work.
- **§3.7 fix the reduction kernels**, **§6.2 wave32 vs wave64** (now more
  interesting: WMMA is natively a wave32 shape and §2.1 runs at wave64
  against a 256-VGPR cap that is what limits its tile size), **§1.4 GEMV
  access pattern**, and the one ISA question §6.1 has left (`v_pk_fma_f16`).

**Then (the missing pillars):** §3.3 attention/flash-attention, §3.4 real
model shapes, §3.1-3.2 fusion (justified by DRAM round-trips, not launch
overhead), §3.5 MoE grouped GEMM. §3.3 in particular now has a working
register-blocked WMMA kernel to build its two matmuls out of.

**Dropped or downgraded by measurement:** §2.2's Q4 unpack (no 2x int8
matrix rate, and no LDS tile to make cheaper), §4.1 (answered: ~300 ns),
§4.2/§4.3 (the GPU-side half of the concern is ruled out), the packed-fp16
half of §1.3 (1.10x, not 2x), and — unexpectedly — the **LDS-staging and
multi-wave-workgroup half of §2.1 itself**, which the 32 MiB MALL makes
redundant on this part.

The one-line summary, now with measured numbers behind each clause: **DRAM
bandwidth is maxed at 236 GB/s so decode wins come only from reading fewer
bytes — and W4A8 now reads 4 bits/weight at 211 of those 236 GB/s, so
decode is finished as a kernel problem and continues only as a format
problem. Prefill was 8% of a measured 55.5 TFLOP/s and is now 45%, bought
entirely by raising arithmetic intensity from 8 to 32 FLOP/byte in
registers — with no shared memory, because the 32 MiB MALL already does
that job and hands the kernel 760 of its 805 GB/s. What is left
in prefill is address-arithmetic and MALL-traffic ordering, not tiling. And
neither dispatch overhead nor a nonexistent int8 matrix-rate bonus is worth
designing around.**
