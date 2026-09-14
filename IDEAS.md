# IDEAS — experiments worth running to squeeze more out of Strix Halo

Written after reviewing `TODO.md`, every shader in `shaders/`, the full
`results.csv`, and the device's actual reported capabilities (`vulkaninfo`,
`/sys/class/drm/card1/device/pp_dpm_sclk`). This is a prioritised
experiment backlog, not a plan — each item states a *hypothesis*, the
*change*, the *expected gain*, and *how we'd know*.

> Note on where the numbers live: every reference to `results.csv` below
> predates the split of that file into `results/<family>.csv` (one CSV per op
> family, same columns, same rows). Read `results.csv` as "the measurements";
> `tail -q -n +2 results/*.csv` is the flat table it used to be.

> **Note on where the write-ups live:** a completed item's full write-up has
> moved to **[`research/`](research/README.md)**, one file per section, and
> the heading here keeps a one-line result and a link. This file is the
> backlog, the roofline and the order of attack; `research/` is the archive.
> **`§N.M` is still the address** — it is cited 293 times from `shaders/`,
> `bench/`, `cmd/` and `vk/`, so a finding may be moved or rewritten but its
> number is permanent.

**§0 has since been implemented and run** (`results.csv` regenerated with
clock instrumentation behind every row). Items it confirmed, refuted, or
re-aimed are marked **[measured]** in place rather than rewritten away, so
the wrong predictions stay visible next to what actually happened.

**§1.1 (W4A8 GEMV) has since been implemented and run too**, and it
delivered: 819 GFLOP/s against DRAM-resident weights, **2.4x** the previous
best decode kernel and **89% of the DRAM bus** — since taken to **96%** by
pinning the pipeline to wave32 (§6.2), and then to **101%** of that measured
peak, at every reduction length rather than one, by §1.7. Decode is a solved
bandwidth problem at 4 bits/weight; see §1.1 for what it took, §1.3 for the
load-width half of it, §6.2 for the wave-size arm, and **§1.7 for the rule
that finished it**: make one lane-step cover a whole weight row
(`VEC = N/(8·WAVE)`) and the kernel reads 99-103% of the bus at N=2048, 4096
and 8192 alike. §6.2's unexplained wave32 1.07x is explained there and turns
out to be a deficit in one cell rather than a win — and the wave32 pin is
withdrawn with it.

**§1.8 (the grouped GEMV for MoE decode) is done as well**, and it is the
decode half of the text-generation goal: the same W4A8 kernel with a table of
routed (token, expert) pairs, which is **1.8-2.0x the grouped GEMM at M=1**
that every decode row in this file used to be measured with, reads **235 GB/s
of distinct weight bytes — 100% of the bus — on the gate/up projection**, and
takes one token's MoE FFN from 9.56 ms to **5.47 ms over 48 layers, 91% of the
5.00 ms its 1.18 GB of 4-bit weights cost at the bus**. Two of its results were
not in the plan: grouping is worth 1.29-1.81x even though every dispatch it
merges already fills the machine (the barrier's *drain* is 1163 ns against
§4.1's 300 ns launch), and **an expert bank should not be padded at all** — the
unpadded stride wins at both shapes, including `down`'s gcd-64 row, which
§3.5's [128, 256] plateau says should be the slow end.

**§1.9 gave that kernel an M block**, which is the one thing §1.8 said it was
missing, and it turns decode's *throughput* end into a separate answer from
its latency end. `MROWS` routed pairs of one expert sharing a workgroup's
weight loads is worth **1.39-1.66x at 256 sequences in flight** and **0.39x at
one row per expert**, so the block has to be chosen from the routing rather
than tuned once. What it takes off is the cache and not the bus: unblocked at
five rows per expert the kernel asks for 746 GB/s — 3.2x the bus — and DRAM
underneath runs at 148, which is the measurement that says an expert's
duplicate pairs were never free. Blocked, DRAM reaches 206 GB/s, 87% of the
bus, and a 256-sequence batch's FFN runs at **694 tok/s against 212 at 16**.

**§1.10 gave it the other block and §1.11 put both in one wave**, which
finishes the decode kernel. `NROWS` consecutive weight rows per subgroup
against one activation row closed §1.8's last open cell — `down`'s 198 GB/s
against `gate_up`'s 235 is a **fixed cost per output row**, 19.8% of down's
time and -0.4% of gate_up's, and dividing the workgroup count without the wave
count is worth only 1.02x — and unlike the M block it is a *latency* lever, so
a single token's FFN reached **5.14 ms over 48 layers, 195 tok/s, 97% of the
bus floor**. Combining the two is worth **1.12-1.19x over each of its own
axes, and 1.10x over the best single-blocked build of any width**, at 256
sequences; it takes `gate_up` to **96% of the DRAM bus** and the FFN to
**818 tok/s** — 1.50x the Q4 grouped GEMM, where §1.9's M block alone measured
1.27x. The two rules to size them by are independent: `VEC` and `NROWS` come
from the model (`VEC = K/(8*WAVE)` and `NROWS >= 8*VEC*WAVE/K`, the same
coverage rule read along the two axes of the matrix), `MROWS` comes from the
routing histogram, and the one place they interact is the pad — an N block
makes a pad slot 4x cheaper where a slot is few output rows.

**§2.1 (register-blocked coopmat GEMM) is done as well**, and it delivered
more: **25.2 TFLOP/s, 6.0x the previous best GEMM kernel at the same shape
and 45% of the measured WMMA ceiling**, up from 8%. Two of its sub-hypotheses were wrong in
useful ways — LDS staging and multi-wave workgroups are both *unnecessary* on
this part, because the 32 MiB MALL already supplies the reuse they exist to
provide — and §2.3 turned into a sharp contradiction: the B
layout that halves the instruction count is 2.8x slower. See §2.1, §2.3, §6.1.

**§2.3's contradiction is now resolved, and resolving it was worth more than
the layout question that raised it.** The cause is DRAM channel aliasing:
every fragment load in these kernels is K-strided, a power-of-two leading
dimension puts all 16 addresses of one load in the same channel, and the
penalty is periodic in the stride — with a **4 KB** period, per §5.1b's
direct measurement of it; §2.3 had only three strides and read it as 2 KB.
Padding each operand's leading dimension 256 B off a multiple of 4 KB — a
host-side allocation change, no shader restructuring — is worth 1.6x on the
transposed-B kernel,
**up to 1.35x on the kernels that already won**, and takes the suite's best
GEMM to **28.4 TFLOP/s (51% of the WMMA ceiling)**. It also retires the
"pinned at the MALL's 805 GB/s" reading of §2.1's plateau. See §2.3.

**§5.1b's strided-bandwidth probe is done too**, and it is the one item so
far that found something nobody was looking for. It reads a fixed set of
bytes at a swept row stride in five request shapes, with no tiling in the
way, and it separates two effects §2.3 had seen as one. The small one is
§2.3's: a load whose addresses are all one stride apart, with that stride a
multiple of the interleave rotation, loses up to **1.34x** — and the rotation
is **4 KB** (sixteen 256 B channels), not the 2 KB §2.3 inferred from three
strides. The large one is new and applies to *plainly contiguous* reads: when
a row is shorter than `gcd(stride, 4096)`, the access pattern **never
addresses some channels at all**, and the achievable fraction of peak is
exactly `min(1, rowBytes/gcd(stride, 4096))` — measured to within 2% at every
point. A K=1024 fp16 weight matrix whose rows someone aligned to 4 KB reads
at **half** this chip's DRAM bandwidth; 1024 B rows at an 8 KB stride, a
**quarter**. It also retires §2.3's leading suspect for its own residue:
at a de-aliased stride a 64-address gather reads within 2% of a contiguous
sweep, so request shape is free and the [N,K] weight layout is usable as-is.
See §5.1b.

**§5.1b's own follow-up is done too, and it turned the coverage law into a
law about concurrency.** One `-D` swaps the probe's traversal so consecutive
*waves* start on consecutive rows instead of walking along one — the same
bytes, the same instructions, the same checksum, and only which addresses
are outstanding together changes. It costs **3.9x** on a plainly contiguous
read at a K=4096 fp16 matrix's natural 8192 B stride, and up to **8x** on a
tensor with *no padding at all*, which retires this item's own "a densely
packed tensor is never at risk". The unification: the achievable fraction of
peak is `min(1, C/gcd(stride, 4096))` where **C is the contiguous run of
bytes the requests in flight at one moment hold in a row** — the whole row
when waves walk along it, one request's run when they do not — and a ladder
that gives each wave 1/2/4/8 KB of its row measures 0.25/0.50/1.00/1.00 of
peak exactly as that predicts. The top rung is one wave per row, which is
the GEMV/W4A8 shape, and it is clean at every stride: decode was never
exposed, and the 1.12x that bounded it is not there. What *is* exposed is
tiling, and rule 1 (pad to 256 B past a multiple of 4 KB) already defends
against all of it. See §5.1b's mechanism 3.

**§2.7 took that law back to the GEMM kernel, and it closed the last thing
this file could not explain.** A tiled GEMM is the exposed shape: a coopmat
fragment load holds 32 B of a row and its waves retire after one, so C is an
eighth of the `gcd` even a padded stride leaves. Issuing a whole K-slab's
fragment loads before the slab's first MMA — identical bytes, identical
instruction counts, identical tile and intensity, only the scheduling —
is worth **2.1x**, takes the suite's best GEMM to **38990 GFLOP/s, 70% of the
WMMA ceiling**, and makes the transposed-B kernel *faster* than row-major at
every size, closing and reversing the 1.6x residue §2.3 carried. The control
that makes it attributable had been sitting in the suite since §2.3: the same
slab depth *without* hoisting leaves 16 loads in flight instead of 45 and
measures as nothing. Depth is not the variable; concurrency is. See §2.7.

**§6.2 (wave32) is done, and it is the first item here whose payoff landed on
a different family from the one it was promoted for.** §2.7 asked for it
because its ladder ends on the 256-VGPR wave64 budget and wave32 was supposed
to halve what a fragment costs. Half of that is true — a 16×16 fragment is the
native wave32 WMMA operand, and the most fragment-heavy variant reports the
*same* VGPR count at both wave sizes, i.e. half the per-work cost — but the
accumulators do not shrink and the register ceiling halves along with the wave,
so §2.7's winning kernel spills at wave32 and loses 25%, and the suite's best
GEMM numbers are unchanged. What did move: the fragment-dominated AI-16 GEMM
grid gains **1.9-2.1x at every size**, wave32 takes the N=1024 crown at 33603
GFLOP/s, and — the one number that changes an engine decision — **the
DRAM-resident W4A8 decode GEMV at `VEC=4` reads 225.7 of 236 GB/s at wave32,
96% of the bus, against wave64's 89%**, reproducible to under 1% over three
runs. It also closed §3.7: the subgroup reductions scale linearly in threads
per row (256 → 224 GB/s, 64 → 68.5, 32 → 35.7), so their 3.3x gap is the lane
count and nothing to do with `subgroupAdd`. See §6.2.

## The roofline, and why it says there's a lot left on the table

Hardware numbers for this part (AMD Radeon 8060S, gfx1151, 40 CU, sclk
tops out at **2900 MHz** per `pp_dpm_sclk`; LPDDR5X-8000 on a 256-bit bus):

> **Updated after running §0.** The ceilings below are now *measured* on
> this chip (`go run ./cmd/bench -skip ...` → the `peak` family), not
> extrapolated from discrete RDNA3 parts, and several §0 hypotheses turned
> out to be wrong. Corrections are marked **[measured]** throughout.

| Ceiling | Measured peak | Ops/clk/CU | Best real kernel | Utilisation |
|---|---|---|---|---|
| DRAM bandwidth | **236 GB/s** (92% of the 256 GB/s bus) † | — | 239 GB/s (`gemv_cold` W4A8, §1.7) | **~100%** ✅ |
| MALL (32 MiB) bandwidth | **805 GB/s** copy, **965 GB/s** pure read † | — | 805 GB/s | **~100%** ✅ |
| Vector FP32 FMA | **22.9 TFLOP/s** | 198 | 2.76 TFLOP/s (`tiled,fp16`) | ~12% |
| Packed fp16 FMA | **25.3 TFLOP/s** | 218 | — (no kernel uses it) | — |
| `dotPacked4x8` int8 | **54.0 TOP/s** | 465 | 2.2 TOP/s (`naive,w8a8`) | **~4%** ❌ |
| WMMA fp16→fp32 | **55.5 TFLOP/s** | 479 | **39.0 TFLOP/s** (`wmma_reg64_bt_hka4_padab128`); 38.8 for attention (§3.3) | **70%** |
| WMMA int8→int32 | **55.7 TOP/s** | 480 | 4.7 TOP/s (`coopmat,q8`) | **~8%** ❌ |

† **[measured] §5.1b** Both bandwidth rows are *conditional on the access
pattern*, which is why they carry a dagger: they are what a contiguous sweep
gets. A row stride that leaves `gcd(stride, 4096) > rowBytes` cuts DRAM to
`rowBytes/gcd` of the figure above — half at 2 KB rows with a 4 KB stride, a
quarter at 1 KB rows with an 8 KB stride — and does so for a plainly
contiguous read; a stride that is a multiple of 4 KB additionally costs a
multi-row gather up to 1.32x. The MALL row is also two numbers, not one: the
805 GB/s §0.4 reported is a read+write copy, and a pure read reaches
940-965 GB/s. See §5.1b.

(Real-kernel column is the best measured anywhere in `results.csv`; the
GEMM entries are at N=4096 except the WMMA fp16 one, whose best shape is now
N=2048 (30.6 TFLOP/s at N=4096), and the `naive,w8a8` entry at N=256 where it
is still cache-resident. The WMMA fp16 row was 4.9 TFLOP/s / 9% until §2.1
register-blocked the kernel, 25.2 / 45% until §2.3 de-aliased its operand
strides, and 28.4 / 51% until §2.7 gave its inner loop a whole K-slab of
loads in flight at once; the int8 row is still the un-blocked one and
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
   **[measured] §1.7 then removed the remaining 11%, and the caveat that the
   211 was one shape's number.** Sizing the load width so that a lane-step
   covers a whole weight row reads **239 GB/s at N=4096, 243 at N=2048 and
   235 at N=8192** — 99-103% of this row's measured peak. The N=8192 figure
   is the one that mattered: the kernel behind the 211 gets **164** there,
   and N=8192 is the shape a real model decodes at. The lever that remains
   for decode is now genuinely only the format.
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
   **[measured] §2.3 then found a second, independent lever that is not
   arithmetic intensity at all: operand *stride*.** Every fragment load in
   these kernels is K-strided, and a power-of-two leading dimension puts all
   16 of its addresses in one memory channel; padding each stride 256 B off
   a multiple of the 4 KB interleave rotation (§5.1b's correction to the
   2 KB §2.3 inferred) is worth 0-5% at N=4096, 4-35% at N=2048 and 2.6x on
   the [N,K] weight layout, for a new best of 28.4 TFLOP/s (51%). It also breaks
   the "pinned at 760 GB/s" reading above: the de-aliased AI-32 kernels
   imply 742-887 GB/s of MALL traffic, i.e. at or *past* the 805 GB/s
   ceiling, so the L1s must be absorbing part of what that arithmetic
   attributes to the MALL — and the AI-85 plateau at 26.0 TFLOP/s implies
   only 306 GB/s, so whatever binds it is not bandwidth at any level.
   **[measured] §2.7 then found a third lever, and it is neither intensity
   nor stride: how many of the slab's fragment loads are in flight at once.**
   Issuing a whole K-slab's loads before its first MMA — same bytes, same
   instruction counts, same tile, same intensity, only the scheduling —
   is worth 2.1x on the transposed-B kernel, takes the suite to **39.0
   TFLOP/s (70%)**, and closes the 1.6x residue §2.3 left behind. At 38990
   GFLOP/s the AI-32 arithmetic implies 1218 GB/s of A+B traffic, past even
   the 965 GB/s the MALL delivers on a pure read, so the L1s are now demonstrably
   serving a large share of it and "implied traffic" has stopped being a
   useful ceiling check for these kernels.

There is also a third, newly quantified lever: **the 32 MiB MALL delivers
805 GB/s, 3.4x DRAM.** Anything that can be restructured to work out of a
sub-32MiB tile — attention K/V tiles, a single quantized layer's weights,
activations — gets 3.4x the bandwidth of anything that streams.

---

## 0. Measurement-validity work — **DONE**, and it changed the answers

**Written up in [`research/0-measurement-validity.md`](research/0-measurement-validity.md).** The harness now measures its own ceilings and records clock and power behind every row. It also falsified three predictions: WMMA int8 is **not** 2x fp16 on this part (both 479-480 ops/clk/CU), packed fp16 FMA is 1.10x scalar and not 2x, and fp32 FMA is power-limited rather than issue-limited.

## 1. Decode path (GEMV / memory-bound) — the tok/s lever

### 1.1 W4A8: Q4 weights fed to `dotPacked4x8EXT` — **DONE** ✅, **2.4x**

**Written up in [`research/1.1-w4a8-gemv.md`](research/1.1-w4a8-gemv.md).** Q4 weights fed to `dotPacked4x8EXT`: **819 GFLOP/s, 2.4x** the previous best decode kernel and 89% of the DRAM bus — the result that turned decode from a compute problem into a bandwidth one.

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

### 1.7 Match the load width to the weight row — **DONE** ✅, **and decode reaches the bus at every N**

**Written up in [`research/1.7-load-width.md`](research/1.7-load-width.md).** Size the load width so one lane-step covers a whole weight row, `VEC = N/(8*WAVE)`: **99-103% of the bus at N=2048, 4096 and 8192 alike**, and 1.43x at the N=8192 a real model decodes at. Also withdraws §6.2's wave32 pin.

### 1.8 A grouped GEMV for MoE decode — **DONE** ✅, **1.8-2.0x the GEMM at M=1, and decode reaches 91% of its bus floor**

**Written up in [`research/1.8-grouped-gemv.md`](research/1.8-grouped-gemv.md).** The W4A8 kernel with a table of routed (token, expert) pairs: **1.8-2.0x the grouped GEMM at M=1**, 100% of the bus on gate/up, and a token's MoE FFN at 5.47 ms over 48 layers. An expert bank should **not** be padded at all.

### 1.9 An M block for the grouped GEMV — **DONE** ✅, **1.66x at a serving batch, and what it takes off is the cache, not the bus**

**Written up in [`research/1.9-m-block.md`](research/1.9-m-block.md).** `MROWS` routed pairs of one expert per workgroup: **1.66x at 256 sequences, 0.39x at one**, so the block must be read off the routing. What it removes is cache traffic, not bus traffic.

### 1.10 An N block for the grouped GEMV — **DONE** ✅, **the down projection's deficit is a fixed cost per output row, and removing it closes §1.8's last open cell**

**Written up in [`research/1.10-n-block.md`](research/1.10-n-block.md).** `NROWS` consecutive weight rows per subgroup: `down`'s deficit is a **fixed cost per output row**, not a grid or occupancy effect. A single token's FFN reaches 5.14 ms, **195 tok/s, 97% of the bus floor**.

### 1.11 The corner: `MROWS` x `NROWS` in one wave — **DONE** ✅, **they compose, and decode's throughput end reaches the bus**

**Written up in [`research/1.11-mn-corner.md`](research/1.11-mn-corner.md).** The two blocks compose: **1.10-1.19x** over the best single-blocked build, `gate_up` at 96% of the bus, and the FFN at **818 tok/s**. `VEC` and `NROWS` come from the model, `MROWS` from the routing.

### 1.12 Sizing the M block from the routing histogram — **DONE** ✅, **1.09-1.13x on `down`'s last cell, and a locality law the cost model cannot see**

**Written up in [`research/1.12-m-block-from-histogram.md`](research/1.12-m-block-from-histogram.md).** Size `MROWS` per dispatch from the routing histogram: **1.09-1.13x** on `down`'s last cell, 848 tok/s at a serving batch. The pad-minimal plan *loses*, because an expert belongs to exactly one dispatch — a locality law no cost model written in groups and slots can see.

## 2. Prefill / batch path (GEMM) — the ~90%-idle matrix cores

### 2.1 Register-block the coopmat kernels — **DONE** ✅, **6.0x** (6.3x after §2.3)

**Written up in [`research/2.1-register-blocking.md`](research/2.1-register-blocking.md).** Register-blocking the accumulators raised arithmetic intensity from 8 to 32 FLOP/byte: **25.2 TFLOP/s, 6.0x**, 8% to 45% of the matrix cores. LDS staging and multi-wave workgroups both turned out to be **unnecessary** — the 32 MiB MALL already supplies that reuse.

### 2.2 Q4 weights into the coopmat path — **DONE** ✅, **2.10x a MoE block**

**Written up in [`research/2.2-q4-coopmat.md`](research/2.2-q4-coopmat.md).** Q4 weights through the coopmat path: **2.10x a whole MoE block**, a prompt chunk's 48 of them from 1.35 s to 0.64 s. The operand must round-trip through LDS (no extension exposes a fragment's lane layout), and that LDS tile's rows must be padded off the 32-bank rotation for 1.68-1.83x.

### 2.3 B stored [N,K] with a column-major `coopMatLoad` — **RESOLVED** ✅ (it was channel aliasing)

**Written up in [`research/2.3-channel-aliasing.md`](research/2.3-channel-aliasing.md).** The "better instruction stream is 2.8x slower" contradiction was **DRAM channel aliasing**: a power-of-two leading dimension puts all 16 addresses of a K-strided fragment load in one channel. Padding each stride 256 B off a multiple of 4 KB is worth up to 1.35x and took the suite to 28.4 TFLOP/s.

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
**[measured] §2.1 promotes this, and §2.3 sharpens what it would be
testing.** §2.1's best kernel sat at ~760 of the 805 GB/s the MALL delivers
at AI 16 and 32, which read as *literally* MALL-bandwidth-bound; §2.3 then
raised the same kernels past that figure (742-887 GB/s implied at AI 32) by
de-aliasing their strides, so part of that traffic must be served by the
L1s and "pinned at the MALL" is no longer the right description. What
survives is the shape of the argument: the AI-32 kernels are still within
~10% of a bandwidth ceiling at *some* level, a swizzle is still the only
remaining change that reduces cross-tile traffic without touching the tile
shape or the register budget, and §2.1's finding that neither LDS staging
nor multi-wave workgroups help still means every bit of cross-tile reuse
this kernel gets comes from a cache it does not manage. Note that the
AI-85 variant is *not* bandwidth-bound anywhere (306 GB/s implied), so a
swizzle should help the AI-32 kernels and do nothing for that one — which
is itself the cleanest way to tell whether the swizzle is doing what it
claims.
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

### 2.7 Hoist the K-slab's fragment loads — **DONE** ✅, **2.1x**, and it closed §2.3's residue

**Written up in [`research/2.7-kslab-hoist.md`](research/2.7-kslab-hoist.md).** Issuing a whole K-slab's fragment loads before its first MMA — same bytes, same instructions, same intensity, only the scheduling — is worth **2.1x** and takes the best GEMM to **38990 GFLOP/s, 70% of the WMMA ceiling**. Depth is not the variable; concurrency is.

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

### 3.3 Attention is completely absent from the suite — **DONE** ✅, **30.4x**, and the layout mattered more than the tiling

**Written up in
[`research/stage-3-dit-attention.md`](research/stage-3-dit-attention.md)**,
because it was built as pipeline stage 3 against Z-Image's real attention
rather than as a benchmark family. Both matmuls on `coopMatMulAdd`, online
softmax, fp16 operands and fp32 accumulators: **38.8 TFLOP/s, 70% of the WMMA
ceiling**, against the scalar flash kernel's 1.28 — one block at 4096 tokens
from 202 ms to **6.64 ms**.

Three results worth carrying forward, none of them the one this item
predicted:

- **Arithmetic intensity is inert here.** 16, 32 and 64 FLOP/byte land within
  1.06x. The caches already supply the cross-block reuse of k and v, so there
  is no DRAM traffic for register blocking to buy back — the same reason LDS
  staging lost in §2.1.
- **Store WMMA operands as 16x16 fragment tiles.** A fragment load holds 32 B
  of each of 16 rows, so any natural row stride puts §5.1b's coverage at
  32/gcd. Packing each operand as contiguous 512 B tiles was the difference
  between 1.56x and 30x, and it needs no registers, no LDS and no hoisting —
  it strictly dominates §2.7's HOIST lever and every future WMMA kernel should
  start here.
- **The register file and the wave size decide it**, as §2.7 said they would:
  every variant that spills loses monotonically in how much, and wave32 is a
  clean 1.40x at identical tiling (§6.2's third arm).

The LDS-keyed hypothesis above was wrong in both directions: the score matrix
never goes to LDS (the k tile is read straight from the fragment-tile layout),
but P *has* to, because `GL_KHR_cooperative_matrix` forbids using an
accumulator as a multiply operand.

### 3.4 Benchmark the *real* shapes from the target models — **DONE** ✅, **and it moved two answers**

**Written up in [`research/3.4-model-shapes.md`](research/3.4-model-shapes.md).** The 60 real weight matrices from the five models in `GOALS.md`, plus the per-model budget table. Retired `VEC=32`, split prefill into **two winners divided by M alone**, and showed the MoE model's prefill is 98% memory-bound.

### 3.5 Grouped / MoE GEMM — **DONE** ✅, **1.1-4.1x**, and it is occupancy

**Written up in [`research/3.5-grouped-moe-gemm.md`](research/3.5-grouped-moe-gemm.md).** One dispatch covering all 512 experts: **1.08-4.14x**, and the mechanism is **occupancy** — a monotone function of workgroups per expert dispatch — not the 1440 dispatches and not the 62% tile padding it was promoted on.

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
   **[measured] §6.2's wave32 arm turned that diagnosis from a guess into a
   line.** Same kernels at 32 lanes per row instead of 64, N=4096:
   `rmsnorm` goes 68.5 → 35.7 GB/s and `softmax` 47.7 → 24.7, both 0.52x for
   0.5x the threads. With `shared`'s 223.7 at 256 threads per row that is
   three points proportional to threads per row and nothing else: it is the
   lane count, not `subgroupAdd`, and not the wave size as such. So the
   rewrite above is the fix, and there is no wave-size knob to try first.
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

### 5.1b The MALL cliff, and the strided-bandwidth probe — **probe DONE** ✅

**Written up in [`research/5.1b-mall-cliff-and-stride.md`](research/5.1b-mall-cliff-and-stride.md).** The probe that measured the memory system directly, and the one item that found something nobody was looking for. The interleave rotation is **4 KB**, not 2 KB, and the achievable fraction of peak is `min(1, C/gcd(stride, 4096))` where C is the contiguous run the requests in flight hold in a row — worth 4x, and up to 8x on a tensor with no padding at all.

### 5.2 Heap topology, carveout size and page size

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

### 6.2 wave32 vs wave64 — **DONE** ✅, and it split three ways

**Written up in [`research/6.2-wave32-vs-wave64.md`](research/6.2-wave32-vs-wave64.md).** Wave size as a per-pipeline knob, and it split three ways: the best GEMM **spills** at wave32 and loses 25%, the fragment-dominated AI-16 grid gains 1.9-2.1x, and W4A8 decode reads 96% of the bus. Also closed §3.7 — the reduction kernels' 3.3x gap is lane count, not `subgroupAdd`.

**[measured] §3.3's attention kernel is a fourth point, and it behaves like the
second one**: 1.40x at wave32 (9.30 → 6.64 ms at 4096 tokens) at identical
tiling, because it is fragment-load dominated and has the registers to spare
where the GEMM's best kernel did not — 240 of 256 with nothing spilled, against
its wave64 sibling's 192. Every geometry that *does* spill at wave32 loses,
which is the same cliff from the other side.

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
819 GFLOP/s / 211 GB/s, 89% of the DRAM bus (226 GB/s and 96% once §6.2 pins
it to wave32), so decode is finished as a kernel problem. **§2.1 is done** — the register-blocked WMMA GEMM at 25.2
TFLOP/s, 45% of the matrix cores, so prefill is no longer the gaping hole
either. **§2.3 is done** — the contradiction §2.1 ran into was DRAM channel
aliasing on K-strided fragment loads, and padding the operand strides took
the best GEMM to 28.4 TFLOP/s / 51%. **§5.1b's strided-bandwidth probe is
done** — the `stride` family, which measured the memory system directly,
corrected §2.3's period from 2 KB to 4 KB, found a *larger* second effect
(whole channels left unaddressed when a row is shorter than
`gcd(stride, 4096)`, costing 2-4x and hitting contiguous reads too), and
retired the request-shape explanation for §2.3's residue. **§5.1b's own
follow-up — the traversal axis — is done as well**, and it generalised that
second effect into one law about *concurrency*: the achievable fraction of
peak is `min(1, C/gcd(stride, 4096))` with C the contiguous run the requests
in flight hold in a row, which is 3.9-8x rather than 2-4x, applies to
unpadded tensors, exonerates the GEMV shape outright, and points the finger
at tiling. **§2.7 is done**, and it took that law back to the GEMM: a whole
K-slab's fragment loads in flight instead of one K-tile's is worth 2.1x, and
at **38990 GFLOP/s / 70% of the WMMA ceiling** it closes and reverses §2.3's
residue, which is the last thing this file was carrying as unexplained.
**§6.2 is done**, the item §2.7 promoted: wave size is now a per-pipeline knob
(`VK_EXT_subgroup_size_control`), and it did not extend §2.7's ladder — the
fragment cost halves but the accumulator cost does not and the register ceiling
moves with the wave, so the suite's best GEMM *spills* at wave32 and loses 25%.
It paid off somewhere else instead: **the DRAM-resident W4A8 decode GEMV at
`VEC=4` reads 225.7 of 236 GB/s at wave32, 96% of the bus, up from 89%**, and
the fragment-dominated AI-16 GEMM grid gains 1.9-2.1x. It also closed §3.7 as a
side effect.
**§2.2 is done**, the item that stood at the top of this list, and it
delivered: the Q4 grouped GEMM takes one MoE block from 28.13 ms to
**13.37 ms**, a prompt chunk's 48 of them from 1.35 s to **0.64 s**, and it
does it by moving the phase off the memory system — 190 GB/s of bus at fp16
becomes 113, while the padded tiles' own rate goes from 22% to **51% of the
matrix cores**. 4x the bytes removed buys 2.1x, and the gap is the whole
finding. Three of its results were not in the plan: the LDS row pad is worth
1.68-1.83x (a `BK`-half slab row is exactly one 32-bank rotation), the scale
block is worth 1.24x between two instruction-identical binaries, and
§5.1b's coverage window *loses* to the traffic that buying it costs once the
kernel is no longer bandwidth-bound.
**§3.5 is done**, the item §3.4 promoted to the top of the file, and it
settled the MoE FFN as a kernel problem: the grouped GEMM is worth
1.08-4.14x over the expert-at-a-time loop, the mechanism is occupancy (a
monotone fall-off from 4.14x at 10 workgroups per expert dispatch to ~1.1x at
the 80 waves §0.1 calls full rate) rather than the dispatch count or the tile
padding it was promoted on, and at 93% of the memory-bound ceiling the fp16
bank implies there is nothing left in the kernel. It also amended §5.1b's
first engine rule from "pad by 256 B" to "land the gcd in [128, 256] B",
which is the same thing only at power-of-two K — and the first shapes in this
suite where it is not are the MoE projections, where the old statement costs
1.19x.
What those nine left behind:

**§1.7 is done**, the item that stood here: the wave32 W4A8 win is explained,
and explaining it finished the decode path. Two of the three probes it named
were run (`VEC=8`, later `VEC=16`, and `ROWS=2`) and they agree that the wave
size was never the cause — hold C, the bytes of a weight row a lane-step
holds, fixed and vary the wave size and the bandwidth is identical to 0.05%.
The 1.07x is a deficit in one cell (N=4096 at `VEC=4`), not a win. What
replaces it is §5.1b's coverage law applied to C: size the load width so one
lane-step covers a whole row, `VEC = N/(8·WAVE)`, and DRAM-resident decode
reads **99-103% of the bus at N=2048, 4096 and 8192** — including **234.6
GB/s at N=8192, 1.43x** the previous kernel at the shape a real model actually
decodes at. The third probe, the `stride` family at both wave sizes, was
**not run and is retired**: it existed to isolate a wave-size effect, and
there is no longer one to isolate.

**§3.4 is done**, the item the last handoff said to run before anything on
this list: the models' own shapes are now in `bench/modelshapes.go` and the
`shapes` family runs the suite's chosen kernels over them. It retired one item
on this list outright and re-aimed two. **Decode needs no wider load**: the
real reduction lengths are 640, 1024, 2560 and 6144, none a power of two, so
their 4-bit rows have small gcds with the 4 KB rotation and K=2560 — most of
the text model's traffic — reads 101% of the bus at *every* built width
including the narrowest. `VEC=32` has no model to serve. **Prefill has two
winners split by M**: §2.7's kernel takes every rectangle with M ≥ 1024 and
loses all 22 below it, by up to 2.8x, to §6.2's AI-16 wave32 arm — a
crossover a square sweep cannot express, since it never separates the batch
from the other two extents. And the budget the shapes add up to says the MoE
model's *prefill* is 98% memory-bound, because 2048 tokens reach an expert as
40 rows, which the best kernel runs at 14% of the WMMA ceiling.

**§3.5 is done**, the item that stood at the top of this list, and none of
the three numbers it was promoted on turned out to be the mechanism. The
grouped kernel — the same WMMA binary with its tile origin read from a table
instead of from `gl_WorkGroupID`, so one dispatch covers all 512 experts — is
worth **1.08-4.14x** over the expert-at-a-time loop, and the speedup is a
monotone function of **how many workgroups one expert's dispatch launches**,
collapsing to ~1.1x at the 80 waves §0.1 says this part needs resident. Not
the 1440 dispatches (a launch is 300 ns against a 20 µs dispatch) and not the
62% tile padding: a 16-row tile takes that to 84% useful and is *slower*,
because the grouped kernel already moves **190 GB/s of a 236 GB/s bus** and is
at **93% of the memory-bound ceiling** the fp16 bank implies. "14% of the WMMA
ceiling" was the wrong ceiling for a shape that never sees the matrix cores.
One MoE block costs 29.99 ms against 38.49 ms, 1.44 s against 1.85 s over 48
layers, and the gather and combine a grouped kernel makes necessary are 5% of
it. It also closed the second item on this list from an unexpected direction —
see below.

**§1.8 is done**, the item that stood at the top of this list, and it is the
one that finishes the decode path for a MoE model. The grouped GEMV — the same
W4A8 kernel with a table of routed (token, expert) pairs and a two-dimensional
grid — is **1.30-3.26x the grouped GEMM at M=1** the file had been measuring
decode with, reads **100% of the bus** on the gate/up projection, and takes a
token's MoE FFN to **5.47 ms over 48 layers, 91% of its 5.00 ms bus floor**,
against 9.56 ms for the GEMM. Three things it found that were not in the plan:
grouping is worth 1.29-1.81x even where every dispatch already fills the
machine, because a barrier costs the *drain* of a memory-bound dispatch
(1163 ns median) and not §4.1's 300 ns launch; the crossover back to the GEMM
is at about **1.2 rows per expert**, not at the M=16 a tile holds, because the
question is how many rows of *one expert* routing supplies; and an expert bank
should **not be padded at all** — every pad costs, including the ones §3.5's
[128, 256] plateau recommends, because a kernel that walks whole matrices wants
its rows contiguous. That also closes the "K=640 is explained, stop padding it"
item this list was carrying separately, and the load-width item with it: at the
expert shapes `VEC` is worth 3-5%, exactly as §3.4 predicted.

**§1.9 is done**, the item that stood here, and it settles the M block as a
*throughput* lever and nothing else. `MROWS` activation rows against one
weight row is worth **1.39-1.66x at 256 sequences in flight**, 1.07-1.17x at
64, a wash at 16 and a **0.39x disaster** at one row per expert, where an
over-blocked kernel does eight slots of work for one pair's output. The best
block is the one nearest the rows routing actually supplies, and the reason it
has to be chosen rather than tuned is a fitted cost model that is worth
carrying: a group (one expert's 845 KB through all its output rows) costs
**6.5-10.5x what a slot costs**, so blocking pays as soon as it removes one
group per 6-10 pads it creates. The mechanism is not the one the item was
promoted on. What the block removes is **cache** traffic: unblocked at 5.05
rows per expert the kernel asks for **746 GB/s** — 3.2x the bus — while DRAM
underneath it runs at 148, so the MALL serving the four-fifths of requests
that are re-reads *is* the bottleneck; blocked, issue falls 4.7x and DRAM
rises to **206 GB/s, 87% of the bus**. It moved the GEMM crossover out by 4x
and split it by projection: `down`'s t=64 tie is now a 1.16x win, but at 5.05
rows a 16-row tile covers an expert in one pass where an `MROWS=4` block needs
1.62, so `down` goes to the GEMM there and `gate_up` stays 1.70x ahead of it.

**§1.10 is done**, the item that stood at the top of this list, and it closes
the last cell §1.8 left open. `down` read 198 GB/s per distinct expert against
`gate_up`'s 235 for identical bytes, and the two candidates were the grid and
what a wave does per output row. Two arms that divide the workgroup count
identically settle it: dividing it *alone* (`ROWS`, same waves, same loads,
same stores) is worth **1.02x**, and dividing it *with the wave count*
(`NROWS`, several of an expert's weight rows per subgroup against one
activation row) is worth **1.17x** and takes down to **98% of gate_up's rate**.
A fit of `T = A/n + B` prices the difference: **19.8% of down's time is fixed
per output row and -0.4% of gate_up's is** — one `subgroupAdd`, one store and
one activation row, paid four times as often against a quarter of the loads.
Three things that were not in the plan. Lane occupancy is falsified a second
time and from the opposite direction — the one build in the grid that *fills*
down's lanes (VEC=1, 80 loads over 64) is **1.05x slower**, while the block
that leaves them idle is 1.17x faster. The N block **beats §1.9's M block at
every batch on down** (1.16-1.58x) while needing nothing from routing and
padding nothing, so the M axis is now the secondary knob. And unlike the M
block it is a **latency** lever: the single-stream MoE FFN goes from 5.43 ms
to **5.15 ms over 48 layers, 194 tok/s, 97% of its 5.00 ms bus floor**.

**§1.11 is done**, the item that stood at the top of this list, and the two
blocks **compose**: a wave carrying `MROWS` activation rows *and* `NROWS`
weight rows is worth **1.12x over each of its own axes on `gate_up` and 1.19x
on `down`** at 256 sequences in flight — 1.10x over the best single-blocked
build of any width, 1.54-2.11x over the unblocked kernel — and it is the
kernel to dispatch at a serving batch. What
that buys is the thing the decode path had left: `gate_up` now reads **96% of
the DRAM bus at a serving batch** (§1.9's best was 87%, the unblocked kernel
63%) and the FFN reaches **818 tok/s against the Q4 GEMM's 545**, closing the
per-projection crossover §1.9 opened — `down` at 5.05 rows per expert goes from
0.77x the GEMM to **0.98x**. Three things that were not in the plan: an N block
makes the M block's pad slots **4x cheaper on `gate_up`** and barely cheaper on
`down` (a pad costs a per-wave part, which `NROWS` divides, plus a
per-output-row part, which it does not — and down's slot is 2560 rows against
gate_up's 640), so the two engine rules interact and the M block may be sized
*high* where its rows are few; the row block **partly substitutes for the load
width**, with `VEC=1` + `NROWS=4` the fastest single-axis build at gate_up's
t=256 cell where §1.7's rule calls that width the worst possible; and the
corner never spills — 83 VGPRs at (4, 4) against the plain kernel's 47 — so
the register-competition branch of the hypothesis did not happen. The two
probes beside it also ran: wave32 + N block pays **1.065x at down's t=256** and
nowhere else, with §1.7's coverage rule explaining the rest.

**§1.12 is done as well**, the item that stood here, and it closed the cell:
sizing the M block off the routing histogram takes `down` at 256 sequences from
2.761 ms to **2.530 ms** (66% → 72% of the bus) and the MoE FFN at a serving
batch to **848 tok/s**, with `down` back from the Q4 GEMM (0.99x → 1.07x). The
selection half is free — a two-term cost model picks the oracle width in 28 of
40 cells and is within 1% in 32, and **calibrating it once at t=4 is as good as
fitting it at the batch**. The mixed-width half turned up the finding: a plan
may give an expert several widths (fewer pads) or one width repeated (fewer
groups), and the pad-minimal plan **loses by up to 1.4x** while the other wins,
because the two differ in the one quantity the model cannot see — whether an
expert's groups land in the same *dispatch*. Each group that does not costs
**3.3-5.9 us**, the price of a cold read of that expert (3.58 us), where a
group inside the part costs 0.45-1.67 us. Barriers between the parts cost
+0.2-1.3%, so it is locality and not schedule. §1.11's "a group is worth 45 pad
slots on gate_up, 6.5 on down" is corrected too: that is a t=4, MALL-resident
number, and at t=256 a group is worth **1.0-2.7 pad slots** on both.

**Next:**
- **Filling the lanes properly** — disjoint lane slices per output row, reduced
  with `subgroupClusteredAdd`. §1.10 finding 4 lowered the expected gain to
  near zero and §1.11 finding 4 raised it again from the other side: if what
  the memory system responds to is bytes in flight per wave, the lane map is
  the third way to buy them, and it is the only one untried. §1.12 finding 6
  now points at it directly: what is left of `down`'s 28% deficit is not the
  table — the best plan's own residual over its cost model is 5-13% and
  re-planning it to fewer pads is what *loses* — it is §1.10's per-output-row
  cost paid over four times the rows.
- **§2.2's own two leftovers, both cheap and both attribution rather than
  gain.** The scale plane's layout — 1.24x with no mechanism attached,
  between two binaries that are instruction-for-instruction identical, which
  a blocked scale layout would settle, and which §1.8 has narrowed by
  elimination: the same axis on the GEMV costs 1.045-1.116x, i.e. its bytes
  and nothing more, so whatever the GEMM's extra 1.1x is, it is not traffic — and the dequant's two packed fp16
  ops per two weights, whose cheap halving is numerically invalid (§2.2
  finding 9) and whose valid form is an fp32-epilogue row sum.
- **Carry the load-width rule to the other GEMV kernels.** fp16, W8A8 and the
  `gemv_subgroup` precisions were never given a load width at all, and they
  sit at 73-75% of their ceilings (§1.3). The rule is format-independent —
  it is about bytes of a row per lane-step — so each needs the same `VEC`
  treatment, and W8A8's row is twice as wide per weight, so its `VEC` is half.
  §3.4 lowered the ceiling on this: at the real reduction lengths the width
  that wins is usually the narrowest, so the expected gain is the gap to the
  bus on *these* kernels, not W4A8's 1.43x. §1.8 lowered it again on the
  W4A8 side — at the MoE shapes the spread across VEC 1-16 is 3-5% — so what
  is left here is the *other formats'* 25% gap, not the width itself.
- **Hoist the other winners, and finish §2.7's attribution.** The ladder was
  run on `reg64_bt` and `reg32_bt` only. `reg64x128` and `wg128x256` hold 32
  accumulators and already sit at 252 VGPRs, so they cannot hoist as they
  stand — but they are the AI-43/85 shapes, and whether the lever survives
  into them is what says how it composes with intensity. Also missing: a
  row-major `hkb4`, which would complete the 2x2 of (operand hoisted) x
  (layout) that §2.7 could only half-fill.
- **Explain the rung-4 cell where padding hurts.** `reg32_bt_hkab4` is
  slower padded than unpadded (0.88x at N=4096, 0.87x at N=2048),
  reproducibly, at both sizes, in both runs. Every other rung on that ladder
  goes the other way. It is small and it is the only cell in §2.7 that the
  coverage law gets backwards.
- **The MALL's own slice structure**, which §5.1b opened and could not
  close. Under *partial* channel coverage the 32 MiB MALL keeps a working
  set only while its span is also inside 32 MiB, and one case lands halfway
  — so the cliff is not a function of bytes touched alone, and "block it
  under 32 MiB for 3.4x" is conditional on the access pattern covering the
  channels. A footprint sweep at a fixed bad stride, which the `stride`
  family already runs from flags (`-stridefootprints 4,8,16,24,32,48
  -striderowbytes 1024 -stridepads 0,1024,3072`), should separate effective
  capacity from bus width. This blocks the §5.1b/§2.4/§3.3 blocking work
  from being sized correctly.
- **§2.4 workgroup swizzle.** Still a cheap structural item, but its premise
  has now been restated twice: §2.3 pushed the AI-32 kernels to 742-887 GB/s
  of implied MALL traffic, past the 805 GB/s ceiling, and §2.7 pushed the
  best of them to **1218 GB/s implied**, past even the 965 GB/s a pure MALL
  read delivers. So "implied traffic" no longer bounds anything for these
  kernels — the L1s are serving a large share — and the swizzle's
  cross-tile-reuse argument has lost the number it was sized against. Test it
  as a swizzle-helps-AI-32-and-not-AI-85 discriminator, not as a
  bandwidth fix. §3.4 supplies a better discriminator still: the real
  rectangles at M ≤ 512 launch few enough workgroups that *which* tile runs
  next is visible, which is where a swizzle acts, and they are 22 of the 38
  shapes the models actually use.
- **§2.2's int8 arm** — a `PRECISION_I8` variant of `gemm_wmma.comp`
  accumulating in int32. Still unbuilt, and §2.2's Q4 arm has made it much
  less interesting: int8 halves the operand bytes where Q4 quarters them,
  there is no int8 matrix-rate bonus (§0.1), and Q4 is the format a real
  4-bit checkpoint arrives in. Build it only if a W8A8 *prefill* story is
  wanted for its own sake.
- **Pad the strides everywhere else** — still demoted, but §3.5 has changed
  what the test is. §5.1b's mechanism 3 predicted these kernels *were*
  exposed (one wave per row, rows a stride apart) right up until its ladder
  measured the top rung — one wave streaming a whole row — at full bandwidth
  at every stride, so the prediction for GEMV/W4A8/W8A8 was **no effect**.
  §3.5 found a second reason the answer may not be no: the rule is a
  *window*, [128, 256] B of gcd, and a 4-bit weight row at the models' real
  K values sits below it (320 B at K=640 gives gcd 64, 1280 B at K=2560 gives
  gcd 256). The below-the-window end is worth 5-17% in the GEMM. Run it as a
  stride *sweep* on the GEMV kernels, not as a single pad.

**Then (cheap, and still untouched):**
- **§1.2 remove the runtime integer divisions** — still the cheapest real
  win on everything W4A8 didn't rewrite (the naive/tiled GEMM paths and the
  old quantized GEMV variants, where `gemv,naive,q4` is still slower than
  naive fp32).
- **§1.3 wide loads on the remaining GEMV kernels** — measured at 1.17x on
  W4A8; fp16 and W8A8 GEMV are still at 75% and 73% of their ceilings.
- **§5.1 memory-type bandwidth** — untouched, cheap, and bandwidth is the
  binding constraint on both of the paths that now work.
- **§3.7 fix the reduction kernels** — now with its cause measured rather than
  guessed (§6.2's wave32 arm made it a line through the origin in threads per
  row), so the rewrite is a 256-thread workgroup and nothing else. Plus
  **§1.4 GEMV access pattern** and the one ISA question §6.1 has left
  (`v_pk_fma_f16`).

**Then (the missing pillars):** §3.1-3.2 fusion (justified by DRAM
round-trips, not launch overhead), and §3.6 (gated DeltaNet), the last
primitive in qwen3.8-flash-next that nothing here covers. §3.3, §3.4 and §3.5
are done — and §3.3 came back with a lever that applies to all of them: store
every WMMA operand as 16x16 fragment tiles.

**Dropped or downgraded by measurement:** §2.2's Q4 unpack (no 2x int8
matrix rate, and no LDS tile to make cheaper), **wave32 as a register-headroom
lever for the GEMM** (§6.2: the fragment cost halves but the accumulator cost
does not and the ceiling halves with the wave, so §2.7's winner spills),
**wave32 as a decode lever** (§1.7: at equal bytes-of-row-per-step the two
wave sizes measure identically, and wave64 wins outright at N=2048), **the
`stride` family at both wave sizes** (§6.2 asked for it to isolate a wave-size
effect; §1.7 removed the effect),
§4.1 (answered: ~300 ns),
§4.2/§4.3 (the GPU-side half of the concern is ruled out), the packed-fp16
half of §1.3 (1.10x, not 2x), the **LDS-staging and multi-wave-workgroup
half of §2.1 itself**, which the 32 MiB MALL makes redundant on this part,
and now **"pad the strides in the GEMV kernels"** — mechanism 3 looked like
it had caught them (one wave per row, rows a stride apart) until its own
ladder measured that exact shape at the full bus at every stride, so that
item is a falsification test rather than an expected win, from the mechanism
most likely to have found one. Also **retired as an
explanation**: request shape as the cause of §2.3's residue, which §5.1b
measured at within 2% of contiguous.

The one-line summary, now with measured numbers behind each clause: **DRAM
bandwidth is maxed at 236 GB/s so decode wins come only from reading fewer
bytes — and W4A8 now reads 4 bits/weight at 99-103% of that bus at every
reduction length, once its load width is sized so one lane-step covers a
whole weight row, `VEC = N/(8·WAVE)` (§1.7); the wave32 pin §6.2 proposed is
withdrawn, because the 1.07x behind it was one cell being wrong at wave64 and
not the wave size doing anything. Decode is finished as a kernel problem and
continues only as a format problem — and the traversal probe that looked most
likely to reopen it instead confirmed the one-wave-per-row shape reads at the
full bus at every stride. Prefill was 8% of a measured 55.5 TFLOP/s and is now 51%, bought
first by raising arithmetic intensity from 8 to 32 FLOP/byte in registers —
with no shared memory, because the 32 MiB MALL already does that job — and
then, for free, by moving every operand's row stride 256 B off a multiple of
4 KB. That last number is the interleave rotation of sixteen 256 B channels,
and getting it wrong costs in two independent ways: a load whose addresses
are all one aliased stride apart aims them at a single channel (up to 1.34x
of bandwidth, up to 1.6x through a GEMM), and — the big one — whenever the
bytes the concurrent waves hold in flight cover less of the 4 KB rotation
than `gcd(stride, 4096)`, the remaining channels are never addressed at all
(2x, 4x, 8x; it hits a plainly contiguous read just as hard, it needs no
padding to happen, and aligning weight rows to a page is the most expensive
tidy-looking thing an engine could do here) — and the rule is a *window*
rather than a pad, `gcd(stride, 4096)` in [128, 256] B, which §3.5's
non-power-of-two shapes are the first in this suite able to distinguish from
"+256 B" and where getting it wrong in the other direction costs 1.19x. Request *shape* is free once
the stride is right, so the [N,K] layout real weights come in is usable
as-is — but request *timing* is not: one wave per row is safe at any stride,
a wave that grabs a fragment and retires is where the 4x lives, a tiled
kernel is the second kind, and §2.7 is what fixing that from inside the
kernel is worth (2.1x, paid for in registers, ending at a spill rather than
a slope). And neither dispatch overhead
nor a nonexistent int8 matrix-rate bonus is worth designing around — the MoE
FFN looked like the counterexample, 1440 of a token's 1861 matmuls, and it is
not: grouping its 512 experts into one dispatch is worth 1.08-4.14x and the
whole of that is occupancy, a monotone function of how many workgroups one
expert's dispatch launches against the 80 waves this part needs resident,
with the launches themselves 1.5% of it. What was left of MoE prefill was
weight bytes and only weight bytes — 93% of the ceiling its format implies,
5.03 GB per block at fp16 — and the 4-bit grouped GEMM that follows from that
is now built and is worth **2.10x a whole block**, 1.35 s of prompt chunk
down to 0.64 s. Removing 4x of the bytes buys 2.1x of the time because the
phase stops being a memory problem in the middle of the change: 80% of the
bus becomes 48%, and 22% of the matrix cores becomes 51%. What it cost was
one thing the plan had and three it did not — the operand has to round-trip
through LDS, because no cooperative-matrix extension exposes a fragment's
lane layout and 4-bit weights therefore cannot be unpacked into registers;
that LDS tile must have its rows padded off the 32-bank rotation, for
1.68-1.83x; the quantization block size is worth 1.24x between two binaries
that are instruction-for-instruction identical; and §5.1b's stride window
stops paying the moment the kernel stops being bandwidth-bound, because the
padding that buys it still costs bytes. Prefill is now a solved format
problem as well as a solved kernel one, and decode's own grouped kernel — the
W4A8 GEMV with the same table, 1.3-3.3x the GEMM it was being measured with
and 91% of its bus floor at a single stream — now has the M block that
separates its throughput end from its latency end: `MROWS` pairs of one expert
per workgroup is 1.66x at 256 sequences and 0.39x at one, because what it
takes off is not the bus but the cache serving four-fifths of a batched
kernel's requests, and the block has to be read off the routing rather than
compiled in once. The other axis of the same block is what finally explains
the one cell that had stood open since decode was first measured honestly:
`down` reads less of the bus than `gate_up` for identical bytes not because of
its grid and not because of its idle lanes — the build that fills them is
slower — but because a fifth of its time is a fixed cost paid once per output
row, and giving one wave four of them (`NROWS >= 8*VEC*WAVE/K`, the load-width
rule read along the other axis of the matrix) takes it to 98% of gate_up's
rate and a decode token's MoE FFN to 97% of its bus floor. And the last thing
the M block was still getting wrong was that it was a constant: read off the
routing histogram per dispatch instead — one width per expert, chosen by a
two-term cost model calibrated once on the cheapest batch there is — and the
last cell under the bus comes up to 72% and a serving batch to 848 tok/s. The
plan that minimises the *pad*, which is what that cell's deficit was made of,
is the one that loses, by up to 1.4x, and the reason is the sharpest
statement this file has of what a grouped kernel actually buys: a group
whose expert the previous dispatch already read costs half a microsecond, and
a group whose expert was last read a dispatch ago costs the 3.6 us its 845 KB
take off the bus — so an expert belongs to exactly one dispatch, and no cost
model written in groups and slots can see the difference.**
