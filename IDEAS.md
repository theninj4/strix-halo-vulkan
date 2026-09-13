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
| WMMA fp16→fp32 | **55.5 TFLOP/s** | 479 | **39.0 TFLOP/s** (`wmma_reg64_bt_hka4_padab128`) | **70%** |
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

**[measured] §5.1b qualifies both numbers.** 805 GB/s is what a *copy*
gets; the `stride` family's read-only kernel reaches **940-965 GB/s** from
the same 32 MiB, so the MALL cliff is closer to **4x** for read-dominated
work — which is what weight streaming is. And neither figure is a function
of footprint alone: at a row stride whose `gcd` with 4 KB exceeds the bytes
read per row, DRAM drops to half or a quarter of 236 GB/s and the MALL drops
*further* than the model predicts, because the same address bits that select
a channel also select a cache slice.

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

### 1.7 Match the load width to the weight row — **DONE** ✅, **and decode reaches the bus at every N**
**Hypothesis** (promoted out of §6.2, which ended on an unexplained 1.07x):
the DRAM-resident W4A8 GEMV read 225.7 of 236 GB/s at wave32 against 211 at
wave64, reproducibly, only in the `VEC=4` arm, and §5.1b's coverage law
predicted the wrong sign. §6.2 named three cheap probes — a `VEC=8` arm,
wave32 with two rows per workgroup, and the `stride` family at both wave
sizes — on the reasoning that the wave size changes three things at once
(lanes sweeping a row, bytes the row asks for per request round, and the
width of the grid) and each probe pins one of them.
**Change**: `gemv_w4a8.comp` took two new `-D`s. `VEC` now accepts 8 and 16
as well as 1 and 4 (two and four `uvec4` of weights per lane per step; the
wide path became a `W4A8_GROUP(G)` macro instantiated once per `uvec4`,
because a `for (g < VEC/4)` loop survives `glslc -O` rolled and a rolled
group loop indexes its operands dynamically, which is the one thing a
load-width arm must not do). `ROWS` puts several subgroups — several output
rows — in one workgroup, so a wave32 pipeline can be given back wave64's 64
threads per workgroup and wave64's grid width while 32 lanes still sweep each
row. Both are orthogonal to the wave size, so the wave64 cells are controls.
`bench/ops_w4a8.go` carries the resulting grid and `runGEMVColdCase` divides
the dispatch by `rowsPerWG`.
**Expected**: an attribution for the 1.07x. What came out is an attribution
*and* a rule that finishes the decode path.

**[measured] It was never the wave size. It is how much of one weight row a
wave asks for in a single step.** DRAM-resident (256 MB of fp16-equivalent
weights, 8x the MALL), block=256, GB/s, with **C** = the contiguous run of
one row that a lane-step holds = `min(WAVE, N/(8·VEC)) · VEC · 4` bytes:

| N (row bytes) | kernel | C | steps/row | GB/s |
|---|---|---|---|---|
| 2048 (1024 B) | `vec1` w64 | 256 | 4 | 201.1 |
| | `vec4` w32 | 512 | 2 | 214.8 |
| | **`vec4` w64** | **1024** | **1** | **243.0** |
| 4096 (2048 B) | `vec1` w64 | 256 | 8 | 180.1 |
| | `vec4` w64 | 1024 | 2 | 210.6 |
| | `vec4` w32 | 512 | 4 | 221.1 |
| | `vec8` w32 | 1024 | 2 | 224.9 |
| | `vec16` w64 (half the wave idle) | 2048 | 1 | 212.3 |
| | **`vec8` w64** | **2048** | **1** | **234.8** |
| | **`vec16` w32** | **2048** | **1** | **238.9** |
| 8192 (4096 B) | `vec1` w32 | 128 | 32 | 134.8 |
| | `vec1` w64 | 256 | 16 | 157.2 |
| | `vec4` w32 | 512 | 8 | 161.1 |
| | `vec8` w32 | 1024 | 4 | 162.3 |
| | `vec4` w64 | 1024 | 4 | 164.3 |
| | `vec8` w64 | 2048 | 2 | 198.8 |
| | `vec16` w32 | 2048 | 2 | 198.9 |
| | **`vec16` w64** | **4096** | **1** | **234.6** |

Four things fall out of that table.

**1. The best cell at every N is the one where a lane-step covers a whole
weight row, and it reads the bus.** 243.0, 238.9 and 234.6 GB/s against a
measured DRAM peak of 236 — 99-103%. The condition is a one-liner: one step
covers the row when `WAVE · VEC · 4 == N/2`, i.e. **`VEC = N / (8 · WAVE)`**,
which at wave64 is `VEC = N/512`: 4 at N=2048, 8 at N=4096, 16 at N=8192. The
previous committed best was 211 GB/s at N=4096 (`vec4` w64) and the kernel had
never been run at any other N; at N=8192 it was **164**, 70% of the bus, which
is the number a real model's decode would have got.

**2. It is the coverage law, applied to the right variable.** §5.1b's
`coverage = min(1, C/gcd(stride, 4096))` is stated for requests in flight; the
stride here is the weight row, `N/2` bytes, and at these N `gcd(N/2, 4096) =
N/2`, so the law says full bandwidth exactly when a step holds the whole row.
That threshold is hit on the nose in all three sweeps. What the law gets wrong
is everything *below* the threshold, where it is far too pessimistic (it
predicts 1/32 of peak at N=8192 `vec1` w32; the measurement is 57%) — expected,
because the law was derived for one wave's requests and here thousands of waves
on adjacent rows partially cover the rotation between them. So: **the law
predicts the cliff edge exactly and the slope not at all.** §6.2 said the law
"predicts the wrong sign" for the wave32 effect. It does, for the wave size.
It is right about the thing the wave size was standing in for.

**3. Same C, different wave size, same bandwidth — to 0.05%.** At N=8192,
`vec8` at wave64 and `vec16` at wave32 both hold 2048 B of a row per step, and
measure **198.83 and 198.93 GB/s**. At N=4096 the two whole-row cells,
`vec8` w64 and `vec16` w32, measure 234.8 and 238.9. That is the direct
falsification of §6.2's wave-size reading: hold C fixed and the wave size
buys nothing.

**4. The 1.07x was a deficit in one cell, not a win in another.** The
wave32-beats-wave64 inversion occurs at exactly one (N, VEC) pair — N=4096,
`VEC=4` — and not at N=2048 or N=8192, where wave64 leads at the same VEC. At
that pair the wave64 kernel takes exactly two steps per row and each step
holds exactly half of it, so every wave in the grid is addressing the same
half of the 4 KB rotation at the same time and the other half's channels sit
idle: §5.1b's mechanism 3, from inside the kernel. Two unrelated changes
each recover it — `ROWS=2` at wave64 (210.6 → 225.5, and ROWS does nothing
in any other cell) and `VEC=8` (→ 234.8, which is the bus). The wave32 arm
was a third. None of them is the fix to reach for; making C the whole row is.

**One more clause the rule needs: every lane must be issuing.** `vec16` at
wave64 and N=4096 has `loadsPerRow = 32` against 64 lanes, so half the wave
is idle. Its C is still a whole row, and it still loses — 212.3 against
234.8 — because it has half as many requests outstanding. The two clauses are
satisfied together by the same equality, which is why the rule is one line and
not two.

**Also: this is why wave64 is the default again.** §6.2's engine rule was
"pin wave32 for the DRAM-resident W4A8 decode GEMV at VEC=4". That is now
superseded: at every N, once VEC is right for wave64, wave32 is equal
(N=4096, 238.9 vs 234.8, inside run-to-run spread) or clearly worse (N=2048,
219.2 vs 243.0 — wave32 satisfies both clauses there and still loses 10%,
which is the one residue this item leaves behind: at equal C and equal lane
occupancy, 64 outstanding requests per wave beat 32).

**Committed numbers, and how to reproduce the rest**: `results/gemv_cold.csv`
holds the default `-coldn 4096`, where the best DRAM-resident decode row is
now **239.1 GB/s** (`subgroup_vec16_w32`, block=1024) against 225.6 before
this item and 211 before §6.2. The N=2048 and N=8192 columns of the table
above are not in that file — one CSV per family means a second N would
overwrite the first — and come from

    go run ./cmd/bench -resultsdir "" -coldfootprints 256 -coldn 8192 \
        -blocks 256 -warmup 5 -iters 20 gemv_cold

with `-coldn 2048` for the other. Both reproduce to ~1% across runs; the
`vec8` w64 cell at N=4096 measured 237.5, 234.8 and 237.8 in three.

**It carries into the cache-resident sweep too, which was not expected.**
The `gemv` family is MALL-resident at these sizes and §6.2 found wave32
*costing* 0.58-0.95x on every row of it. Under the new widths the same sweep's
best W4A8 row at N=4096 goes from **563.5 GB/s** (`vec4`, the committed
previous best) to **734.0** (`vec16_w32`) — **1.30x**. The rule was derived
against DRAM and the MALL has no 4 KB channel rotation to miss, so this is
presumably the plainer half of the same thing (fewer, larger requests per row),
but it was not predicted and it is not explained here.

**What this changes for the engine**: pick `VEC = N/512` per weight matrix at
wave64 — it is a compile-time `-D`, so a real engine ships the two or three
widths its layer shapes need and selects per matrix. The kernel currently
tops out at `VEC=16`, which is N=8192; an N=16384 matrix would want `VEC=32`
(eight `uvec4` per lane) and is a two-line extension of the same macro.

**Effort**: low (done). **Value**: high — it closes §6.2's open question, it
takes the decode path from 70-89% of the DRAM bus to 99-101% *at every shape
rather than one*, and the 1.39x at N=8192 is on the shape the target models
actually decode at.

### 1.8 A grouped GEMV for MoE decode — **DONE** ✅, **1.8-2.0x the GEMM at M=1, and decode reaches 91% of its bus floor**
**Hypothesis** (promoted out of §3.5's "not done", and the largest gap in this
file once §2.2 landed): every decode number the `moe` family reports is the
grouped *GEMM* run at M=1. A cooperative-matrix A fragment is 16 rows tall and
these tiles are 16-64, so a one-row expert group fills 1/16th to 1/64th of one:
the weights are read once either way — which is why §3.5 called the waste
free — but A's traffic and the executed FLOPs are multiplied by the tile
height. And the 10 experts one token routes to are 8 MB of 4-bit weights per
projection, *inside* the 32 MiB MALL, so the timing loop re-reads them out of
cache. The honest kernel is §1.1's W4A8 GEMV with §3.5's table, and decode is
the phase `GOALS.md` is actually asking about.

**Change**: `gemv_w4a8.comp` took a `GROUPED` mode — one extra binding holding
one entry per routed (token, expert) pair (the pair's row in the `[E*N, ldw]`
bank, its activation row, its output row), and a two-dimensional grid whose x
is the output row inside an expert's matrix and whose y is the pair, so the
pair index costs no division and consecutive workgroups sweep one expert's
rows in address order. Six builds (VEC 1/4/8/16 at wave64, VEC 4/8 at wave32);
the bank's row stride and the quantization block are pushed, so one binary
sweeps both. `GROUPED=0` is textually the file the twelve existing binaries
were built from and every one of them is byte-identical across the change
(`cmp`-verified), so `results/gemv{,_cold}.csv` and `results/shapes.csv` stay
comparable. Measured over §3.5's routing at decode batches of 1, 4, 16 and 64
sequences — the batch is what moves the footprint off the MALL (8, 34, 117,
315 MB) while rows per expert stay at 1.00-1.72, so every cell is still a GEMV
shape. `bench/ops_moe_gemv.go`, `results/moe.csv`, and the Q4 GEMM arm was
re-run at the same four batches so the comparison is the same routing through
two kernels rather than two measurements.

**Finding 1: the GEMV is 1.30-3.26x the GEMM at M=1, and what it removes is
activations, not weights.** Best-of-arm against best-of-arm, grouped, both Q4,
distinct weight bytes over the time (GB/s of the 236 GB/s bus):

| layer | tokens | footprint | GEMV ms | GEMV W | GEMV act | GEMM ms | GEMM W | GEMM act | GEMV/GEMM |
|---|---|---|---|---|---|---|---|---|---|
| gate_up | 1 | 8 MB (MALL) | 0.012 | 676 | 4 | 0.031 | 293 | 39 | **2.66x** |
| gate_up | 4 | 34 MB | 0.083 | 408 | 2 | 0.272 | 135 | 18 | **3.26x** |
| gate_up | 16 | 117 MB | 0.497 | **235** | 2 | 0.999 | 128 | 17 | **2.01x** |
| gate_up | 64 | 315 MB | 1.432 | 220 | 2 | 2.605 | 132 | 18 | **1.82x** |
| down | 1 | 8 MB (MALL) | 0.019 | 430 | 6 | 0.044 | 207 | 41 | **2.39x** |
| down | 4 | 34 MB | 0.115 | 295 | 4 | 0.204 | 181 | 36 | **1.77x** |
| down | 16 | 117 MB | 0.590 | **198** | 3 | 0.770 | 166 | 33 | **1.30x** |
| down | 64 | 315 MB | 1.997 | 158 | 3 | 2.005 | 171 | 34 | 1.00x |

The GEMM's best tile at this shape is the smallest built (16x32 at wave32) and
it is still **7-11% useful rows**; it moves 17-41 GB/s of activations where the
GEMV moves 2-6, and its weight rate tops out at 128-171 GB/s where the GEMV
reaches **235 GB/s — 100% of the bus** on gate_up. So the ceiling §3.5 said
this phase was at was the format's, not the kernel's, and at M=1 the kernel
was leaving half of it.

**Finding 2, which the table above also contains: the crossover is at about
1.2 rows per expert, and it is a re-read.** This GEMV runs one workgroup grid
per *pair*, so two tokens routed to the same expert read its weights twice,
while one GEMM tile amortizes them over 16-64 rows. At t=16 (1.15 pairs per
expert) that costs little and the GEMV wins 1.30-2.01x; at t=64 (1.72) the
down projection ties outright. That is §1.5's GEMV→coopmat crossover, measured
at last, and it lands *far* below the M=16 a tile can hold — because the
comparison is not "one row against sixteen" but "one row against sixteen rows
of the same expert", which routing only supplies at batch. The fix is not the
GEMM: it is an M-blocked GEMV (several activation rows against one weight row,
one extra accumulator each), which **§1.9 has since built** — it takes this
t=64 `down` cell from the 1.00x tie above to 1.16x, and the tie itself moves
out to 5 rows per expert.

**Finding 3: grouping is worth 1.29-1.81x, and at decode it is the barrier,
not occupancy.** §3.5 tied the grouped GEMM's 1.1-4.1x to how many workgroups
one expert's dispatch launches, with the effect gone by the 80 waves §0.1 says
this part needs resident. A GEMV launches one workgroup *per output row* — 640
or 2560 per pair, 8-32x that threshold — so on that mechanism grouping should
be worth the launch cost alone. It is worth much more: dividing the
grouped-vs-per-pair gap by the number of dispatches it removes gives **690-1555
ns per dispatch, median 1163**, against §4.1's **300 ns** for an empty shader
with a barrier. What the extra ~900 ns buys is the drain: a memory-bound
dispatch's last waves are still waiting on DRAM when the barrier stops the next
dispatch from starting, and §4.1's empty shader has nothing to drain. So §3.5's
occupancy law and this are two different costs of the same barrier, and an
engine pays the second even when every dispatch fills the machine.

**Finding 4: the load width is worth 3-5% here, and §1.7's rule explains why
it is not worth 1.43x.** DRAM-resident (t=16, t=64), the spread across VEC
1/4/8/16 is 269-377 GB/s of issued bytes at gate_up and 208-271 at down, with
VEC=16 the worst cell at both. The reason is the rule itself: a 4-bit expert
row is 1280 B (gate_up) or 320 B (down), whose gcds with the 4 KB rotation are
256 and 64 — small enough that even VEC=1's 256 B lane-step covers them. This
is §3.4's finding 1 again ("at the real reduction lengths the width that wins
is usually the narrowest"), now at the shapes a MoE model actually decodes,
and it means the `VEC` knob is a 5% engine decision here rather than the 1.43x
it is at N=8192.

**Finding 5: on a kernel that reads its scales the way it reads its weights,
the scale plane costs exactly its bytes.** QBLOCK=32 against QBLOCK=128 adds
4.7% of the bank (6.25% of it against 1.56%) and costs **1.045-1.116x** of
time. Worth recording next to §2.2's finding 4, where the same axis was worth
**1.24x between two instruction-identical GEMM binaries** and the bytes did not
explain it: whatever that was, it was not the scale traffic.

**Finding 6, the one that changes an allocator: do not pad an expert bank at
all.** The handoff carried "K=640 is explained, and the fix is to stop padding
it" as a separate item; this arm answers it, and the answer is stronger than
the item. Every stride in the sweep is a multiple of 64 B, so every row still
starts on a cache line and the padding bytes are **never read** — unlike
§2.2's arm, which had to buy coverage with 1.2-2.4x the traffic, this one buys
it with nothing but address span. Distinct bytes read are identical (315 MB) in
every row below (grouped, t=64, GB/s asked for):

| gcd(stride, 4096) | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| gate_up (1280 B row) | 356 | 371 | **376** (unpadded) | 323 | 244 |
| down (320 B row) | **268** (unpadded) | 239 | 233 | 203 | 110 |

The unpadded stride wins at both shapes, and on `down` it wins *at gcd 64*,
below the [128, 256] plateau §3.5 established on the GEMM — padding it into
that window costs **1.12-1.15x** and padding it to gcd 1024 costs 2.4x. So the
plateau is a property of a *tiled* read, where a fragment holds 32 B of a row
and the gcd is what the waves in flight span, and not of a read that walks a
whole matrix: a grouped GEMV at the natural stride makes each expert one
contiguous 845 KB run that consecutive workgroups continue, and any pad punches
a hole in every row of it. Neither the gcd nor the pad size orders the rest of
the sweep on its own (down's 768 B / gcd 256 beats its 512 B / gcd 512), which
is unexplained; what is not in doubt is the engine rule, and it is the cheapest
one in this file: **leave the bank alone.**

**Finding 7: what a decode token's MoE FFN costs.** Per *distinct* expert —
the quantity that transfers, because a real step's 10 experts have not been
touched for 47 layers — at the least-duplicated DRAM-resident batch:

| | gate+up / expert | down / expert | one block | x48 layers | FFN-only tok/s |
|---|---|---|---|---|---|
| grouped GEMV | 7.2 us | 4.2 us | 0.114 ms | **5.47 ms** | **183** |
| grouped GEMM at M=1 | 14.4 us | 5.5 us | 0.199 ms | 9.56 ms | 105 |
| the bus (1.18 GB at 236 GB/s) | — | — | 0.104 ms | 5.00 ms | 200 |

**91% of the only ceiling that can matter**, and 1.75x the kernel the suite has
been measuring decode with. For scale: §3.4's whole-token budget is 2.89 GB and
11.9 ms at the best measured rate, of which this is 1.18 GB — so the MoE FFN is
41% of a decode token's weight traffic and is now within 9% of reading it at
the bus.

**One cell left open.** The down projection reads 198 GB/s per distinct expert
where gate_up reads 235, for identical bytes (845 KB per expert either way).
Its rows are 4x as many and a quarter as long, so it launches 4x the workgroups
and issues 4x the output writes, and at VEC=4 its 640-nibble row is only 20
lane-steps for a 64-lane wave. The lane-count explanation is the testable one
and it **fails**: the wave32 arms, where the same row is 20 steps over 32
lanes, measure within 1% of wave64 (0.590 against 0.594 ms). So it is not lane
occupancy, and the remaining candidates are the workgroup count and the write
pattern.

**What this changes for the engine**: decode a MoE FFN with the grouped GEMV,
one dispatch per projection per layer, no gather pass at all (the gather exists
in the GEMM path only because a coopmat fragment reads 16 *consecutive* rows;
a subgroup reads whichever row the table names), and do not pad the bank.
Switch to the grouped GEMM when routing gives an expert more than ~1.2 rows,
or — better — give the GEMV an M block, which §1.9 has now built and which
moves that crossover to 5 rows on one of the two projections and past it on
the other.

**Effort**: medium (done). **Value**: high — it is the decode half of
`GOALS.md`'s text-generation goal, it is 1.75x on a phase that is 41% of a
token's bytes, and it retires the last measurement in this file that was
standing in for a kernel that did not exist.

### 1.9 An M block for the grouped GEMV — **DONE** ✅, **1.66x at a serving batch, and what it takes off is the cache, not the bus**
**Hypothesis** (§1.8 finding 2, the item at the top of this file's Next list):
the grouped GEMV runs one workgroup grid per routed *pair*, so two tokens
routed to the same expert read that expert's whole 4-bit matrix twice, while
one cooperative-matrix tile amortizes 16-64 rows over one read. §1.8 measured
the crossover back to the GEMM at **1.2 rows per expert** — the `down`
projection ties at t=64 — and named the fix: several activation rows against
one weight row, one extra int32 accumulator and one extra activation-row
pointer each. Expected: the GEMV's 1.8-2.0x holds out to the batches a serving
engine actually decodes at.

**Change**: `gemv_w4a8.comp` took an `MROWS` knob on the grouped arm. A
*group* is `MROWS` consecutive table slots naming one expert; a workgroup
loads that expert's weight row once per step and dots it against `MROWS`
activation rows, each with its own accumulator, activation base, block-sum
base and output row. An expert whose routed count is not a multiple of `MROWS`
gets a short group padded with slots that repeat the last real pair's
activation row and write a scratch output row, so the inner loop stays
branchless. Four builds (VEC=4 at MROWS 2/4/8, VEC=8 at MROWS=2); MROWS=1 is
textually the previous file and all eighteen existing binaries are
`cmp`-identical. The decode sweep gained **t=256**, where routing gives an
expert 5.05 rows — the first batch in the list with something for a block of 4
to amortize. `bench/ops_moe_gemv.go`, `results/moe.csv`.

**Finding 1: the block pays exactly in proportion to what routing gives it,
and the sign flips at about 1.7 rows per expert.** Against the same VEC at
MROWS=1, best cell per layer:

| pairs/expert | batch | gate_up MROWS=2 | 4 | 8 | down MROWS=2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| 1.00 | 1 | 0.92x | 0.64x | 0.41x | 0.98x | 0.69x | **0.39x** |
| 1.00 | 4 | 0.96x | 0.81x | 0.62x | 0.95x | 0.75x | 0.52x |
| 1.15 | 16 | 1.01x | 0.99x | 0.91x | 0.99x | 0.85x | 0.67x |
| 1.72 | 64 | 1.06x | **1.07x** | 1.00x | **1.17x** | 1.07x | 0.81x |
| 5.05 | 256 | 1.21x | 1.29x | **1.39x** | 1.47x | **1.66x** | 1.65x |

The best block is the one nearest the rows routing supplies, and the penalty
for over-blocking is steep and monotone: at one row per expert an MROWS=8
kernel does eight slots of work for one pair's worth of output and takes 2.4x
as long. There is no free setting — `MROWS` is a routing-dependent choice, not
a tuning constant.

**Finding 2, the mechanism: the block removes *cache* traffic, and at batch
that was the binding constraint rather than DRAM.** The rates the two columns
report are the whole story (t=256, 512 experts touched, 428 MB distinct per
projection either way):

| layer | MROWS | groups | issued GB/s | distinct GB/s | ms |
|---|---|---|---|---|---|
| gate_up | 1 | 2560 | **746** | 148 | 2.898 |
| gate_up | 2 | 1407 | 496 | 179 | 2.395 |
| gate_up | 4 | 829 | 312 | 191 | 2.245 |
| gate_up | 8 | 544 | 221 | **206** | **2.078** |
| down | 1 | 2560 | 369 | 73 | 5.860 |
| down | 2 | 1407 | 298 | 107 | 3.993 |
| down | 4 | 829 | 199 | **121** | **3.526** |
| down | 8 | 544 | 129 | 120 | 3.555 |

Unblocked, the kernel asks for **746 GB/s** — 3.2x the 236 GB/s bus — and DRAM
underneath it runs at **148**, 63% of the bus. Four fifths of the requests are
re-reads of an expert a previous pair already pulled in, the MALL serves them
at roughly the 750-965 GB/s §5.1b measured for a cache-resident read, and
*that* rate is what sets the time. Blocking to MROWS=8 cuts the issue by 4.7x
and DRAM rises to **206 GB/s, 87% of the bus**. So the answer to "are an
expert's duplicate pairs free because the cache serves them?" is no: they are
cheaper than a DRAM read and dearer than a register, and at five duplicates
deep the cache is the bottleneck. This also retires the reading of §1.8's
issued column that took 220-377 GB/s as evidence the bus was saturated — it
was evidence of the opposite.

**Finding 3: a pad slot costs a tenth of a pair, which is why the penalty
curve is as shallow as it is.** At t=1 and t=4 every expert has exactly one
routed pair, so an `MROWS=m` build reads the same weights as MROWS=1 and does
`m` slots of work — a controlled measurement of a slot with the weight traffic
held fixed. Fitting `T = groups*w + slots*s` at t=4 gives **w = 1.92 µs and
s = 0.182 µs per gate_up group/slot (10.5:1)** and **w = 2.53 µs, s = 0.390 µs
at down (6.5:1)**. A group is one expert's 845 KB of weights through 640 or
2560 output rows; a slot is one more accumulator, one more activation row
(cache-hot) and one more fp32 output row. So the rule an engine needs is
arithmetic: blocking pays as soon as it removes one group per 6-10 pads it
creates, which at Poisson-ish routing means **`MROWS` ≈ the mean rows per
expert**, and erring low is much cheaper than erring high.

Note what this says about the slot cost itself: down's slot is 2.1x gate_up's
where its output rows are 4x and its reduction length is a quarter. The cost
per slot is therefore neither the dot products (which would order it the other
way) nor purely the writes — it sits between, which is what a fixed
per-output-row cost (a `subgroupAdd` and a store, both independent of K) plus
a small K-proportional part looks like. That is the same suspect §1.8's open
cell has: `down` reads 198 GB/s against gate_up's 235 for identical bytes, and
it is the shape with 4x the output rows.

**Finding 4: the crossover with the GEMM moves out by 4x, and it is now a
per-projection decision.** Best GEMV (blocked or not) against best Q4 grouped
GEMM, same routing:

| batch | pairs/expert | gate_up | down | GEMV kernel |
|---|---|---|---|---|
| 1 | 1.00 | 2.76x | 2.35x | unblocked |
| 4 | 1.00 | 3.26x | 1.79x | unblocked |
| 16 | 1.15 | 2.04x | 1.31x | `v4_m2` at gate_up |
| 64 | 1.72 | 1.94x | **1.16x** (was 1.00x) | `v4_m4`, `v4_m2` |
| 256 | 5.05 | **1.70x** | **0.77x** | `v4_m8`, `v4_m4` |

The `down` tie §1.8 measured at t=64 is now a 1.16x win, and the GEMV holds
both projections out to 1.72 rows per expert. At 5.05 it splits: gate_up stays
1.70x ahead, `down` goes to the GEMM. The reason is visible in finding 2's
table — a 16-row tile covers all 5.05 of an expert's rows in *one* pass, so
the GEMM's groups equal its experts, while an MROWS=4 block still needs 1.62
passes and MROWS=8 pays 1792 pads to get to 1.06. The M block is a coarser
instrument than a tile: it is a compile-time constant against a per-expert
count, and the tile rounds up for free.

**Finding 5: what this is worth to a serving engine.** The FFN alone, all 48
blocks, per token of the batch:

| sequences in flight | GEMV block | x48 layers | FFN tok/s | GEMM tok/s | kernel |
|---|---|---|---|---|---|
| 1 | 0.041 ms | 1.98 ms | 504 (MALL) | 196 | unblocked |
| 16 | 1.573 ms | 75.5 ms | 212 | 120 | `v4_m2` |
| 64 | 4.403 ms | 211 ms | 303 | 185 | `v4_m4` |
| 256 | 7.681 ms | 369 ms | **694** | 546 | `v4_m8` |

At 256 sequences the FFN is 694 tok/s against 212 at 16, because the 1.18 GB a
token's experts cost is being shared: the distinct weight traffic per pass
stops growing once routing touches all 512 experts, and everything after that
is free. The t=1 row is the MALL flattering an 8 MB working set, as always.

**Finding 6: the single-stream decode budget is unchanged.** The headline
§1.8 set — one cold token's MoE FFN at **5.43 ms over 48 layers, 184 tok/s,
92% of the 5.00 ms bus floor** — is the same number with the block available
(it picks `v4_m2` at 1.15 rows per expert, worth 1.01x). The M block is a
throughput lever, not a latency one, which is what finding 1's top rows say:
at one row per expert there is nothing to amortize and the pads cost.

**What this changes for the engine**: keep one build per `MROWS` in {1, 2, 4,
8} and pick per dispatch from the routing's mean rows per expert — 1 below
~1.5, then the nearest block at or below the mean. Past ~5 rows per expert,
send `down` to the grouped GEMM and keep `gate_up` on the blocked GEMV; that
is two kernels for one FFN, and it is worth 1.7x on the half that stays.

**Still open**: `down`'s per-output-row cost, now with a second measurement
pointing at it (§1.8's open cell, and this section's finding 3). The probe both
suggest is the same one: give a workgroup several *output* rows rather than
several activation rows — the `ROWS` knob the non-grouped kernel already has,
which `GROUPED` currently forbids.

**Effort**: low (done — one `-D`, a table-builder change and four binaries).
**Value**: high at batch, zero at a single stream: 1.39-1.66x at 256 sequences,
1.07-1.17x at 64, and it is what lets one kernel serve both the latency and
the throughput end of decode.

## 2. Prefill / batch path (GEMM) — the ~90%-idle matrix cores

### 2.1 Register-block the coopmat kernels — **DONE** ✅, **6.0x** (6.3x after §2.3)
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
   **[measured] §6.2 has since run and the cap does not move**: RADV's 256
   ceiling is in units that halve with the wave, so wave32 buys headroom only
   for the *fragments*, whose per-work cost genuinely halves, and not for the
   accumulators. The 32-accumulator variants spill at wave32.

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

**[measured] It has since been closed (§2.3), and it moved this kernel
too.** The memory behaviour was DRAM channel aliasing on the K-strided
fragment loads, which both layouts have — A is K-strided in every variant
here — so de-aliasing the strides raised the register-blocked kernels
themselves: `wmma_wg128x256` 25331 → **26008** GFLOP/s at N=4096 and
`wmma_reg64` 20216 → **27260** at N=2048 (1.35x, the shape where aliasing
was costing the most). With *both* strides padded, the transposed-B kernel —
the 389-instruction one this paragraph wanted — reaches **28377 GFLOP/s at
N=1024, 51% of the ceiling** and the suite's best GEMM number, though it
still loses at N=4096. So the address-arithmetic diagnosis above is now partly
testable rather than hypothetical: the shorter instruction stream does win,
but only where its operand fits the caches.

**[measured] §2.7 has since finished that argument.** Giving the inner loop
a whole K-slab of fragment loads in flight instead of one K-tile's worth
takes the transposed-B kernel — the 389-instruction one — to **38990
GFLOP/s, 70% of the ceiling**, ahead of row-major at every size. So the
address-arithmetic diagnosis was right: the shorter instruction stream is
the better kernel, and everything that made it look otherwise was the memory
system, in two layers (channel aliasing, §2.3; loads-in-flight coverage,
§2.7). The plateau this paragraph describes was never an MMA-issue limit.
**Effort**: high (this is a real GEMM kernel). **Value**: highest for
prefill, image generation, and the Parakeet encoder. **Delivered.**

### 2.2 Q4 weights into the coopmat path — **DONE** ✅, **2.10x a MoE block**
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

**[measured] §3.5 re-promoted the Q4 arm to the top of the file, and it has
now been built and run.** What changed between those two readings is that
§3.5 put a number on the thing a 4-bit bank removes: the MoE prefill at 93%
of the ceiling its *format* implies, 5.03 GB of expert weights per block,
21.3 ms of unavoidable DRAM traffic inside a 28-30 ms block. That is not a
tile-fill argument, which is what this item kept being downgraded on — it is
the whole of the phase.

Built as `shaders/gemm_wmma_q4.comp`, a separate file rather than another -D
on `gemm_wmma.comp`, for one reason: **a cooperative-matrix fragment's lane
layout is not exposed**. `GL_KHR_cooperative_matrix` defines `m[i]` and
`m.length()` but not which (row, col) an element is, so 4-bit weights cannot
be unpacked into registers and declared a fragment. They have to be written
somewhere `coopMatLoad` can read with a defined layout, and the only such
place that is not DRAM is LDS. So the kernel is §2.7's winner with one
operand rerouted: A stays fp16 from global with `HOIST_A`, and B's 4-bit
slab is dequantized into an LDS tile per K-step and read back with the same
column-major fragment load the `_bt` variants already use.

**This is not the LDS that §2.1 measured losing.** That one was a *reuse*
mechanism — stage a slab so `WAVES_M x WAVES_N` waves share it — and it lost
because the 32 MiB MALL was already supplying the reuse for free. This one is
a *format conversion*: one wave per workgroup, the slab is that wave's own B
tile, nothing is shared. What is paid for is the round trip and the barriers.

The unpack avoids integer-to-float conversion entirely. fp16 `1024.0` is
`0x6400` and its ulp at that exponent is exactly 1.0, so OR-ing a 4-bit value
into the mantissa gives `1024+n` exactly; subtract 1032 and multiply by the
block scale, two nibbles at a time in one `f16vec2`. Two packed fp16 ops per
two weights, no conversions, and every intermediate exact.

Eight builds: §3.5's five tile geometries unchanged, so the two formats'
tables subtract row for row, plus one axis each for the scale block, the LDS
row pad and double buffering. A stride arm on top. `results/moe.csv`, same
routing, same tile table, same two schedules, weight bytes counted as packed
nibbles plus fp16 scales.

**Finding 1: 2.10x on a whole MoE block, and 2.24x on the matmuls.** Best
fp16 configuration measured anywhere in the family against best Q4, at 2048
tokens:

| | gate_up (N=640, K=2560) | down (N=2560, K=640) | block | x48 layers |
|---|---|---|---|---|
| fp16 | 8.77 ms | 9.18 ms | 28.13 ms | 1.35 s |
| Q4 | **3.82 ms** | **4.31 ms** | **13.37 ms** | **0.64 s** |
| | 2.29x | 2.13x | **2.10x** | |

Against the geometry arm alone — like for like, both formats at their arm's
default stride — it reads 2.32x and 2.52x, and 2.24x on the block; the
difference is that fp16 had 1.18x of stride left on `down` (§3.5 finding 3)
and Q4 has none, for the reason in finding 6. The whole-block ratio is below
the matmul ratio because the gather and combine do not shrink: they were 5%
of the fp16 block and are **10.6%** of the Q4 one.

**Finding 2: the constraint moved, which is what the 2x rather than 4x is.**
The byte count falls 4x and the time falls 2.1x, and the two numbers that say
why are in the same rows:

| | weight GB/s (of 236) | exec GFLOP/s (of 55500) |
|---|---|---|
| fp16, gate_up | 190 — **80% of the bus** | 12137 — 22% |
| Q4, gate_up | 113 — 48% | 28104 — **51% of the matrix cores** |
| Q4, down | 100 — 42% | 24900 — 45% |

At fp16 the kernel was pinned against the bus and the FLOPs were free. At Q4
it is at roughly *half* of each ceiling and pinned against neither — the
dequant path itself is what is left, which is 513 VALU instructions in a loop
body that issues 64 MMAs, plus two barriers per K-step. The 4-bit bank's own
floor is 5.5-6.0 ms per block against a 11.95 ms of matmul, so bytes are no
longer the story even in principle.

**Finding 3: §3.5's finding 2 inverts, as predicted, and then loses to
something else.** There, a 16-row tile bought 84% useful rows instead of 62%
and was *slower*, because the padding cost FLOPs and FLOPs were free. Here
they are not, and among the QBLOCK=32 geometries the 16-row tile is now the
fastest at both layers — `moe_q4_reg16x32_w32` at 4.63/4.32 ms against
`moe_q4_reg64`'s 4.73/4.53 — winning by *executing fewer FLOPs* (17247 exec
GFLOP/s against 22697, at a higher useful rate). The inversion is real and
it is small, 1.02-1.05x. It is beaten outright by a variant that is not a
geometry at all:

**Finding 4: the scale block is worth 1.24x, and the two binaries are
instruction-for-instruction identical.** `QBLOCK=128` against `QBLOCK=32` at
the same tile takes gate_up from 4.73 to 3.82 ms. `RADV_DEBUG=shaderstats`
prices them at the same 252 VGPRs, the same 5064-byte code, the same 802
instructions, the same 513 VALU and the same 100 VMEM — the only difference
in the source is the shift in the scale index. And the nominal bytes do not
explain it either: scales go from 6.2% of the bank to 1.6%, a 4.4% traffic
difference standing behind a 24% one. What is left is **where the scale loads
land**: at QBLOCK=32 a slab's 128 staging chunks read 64 rows of a scale
plane whose own row stride is 160 B, so each 2-byte scale drags in its own
cache line. This is measured, not attributed — see the open item below.
**[narrowed by §1.8]** The same axis on the grouped GEMV, which reads its
scales exactly the way it reads its weights, costs **1.045-1.116x** — its bytes
and nothing more. So the extra ~1.1x here belongs to how the *GEMM* stages
them, not to the traffic, which leaves the scale plane's layout as the one
candidate standing.

**Finding 5: the LDS row pad is the single biggest knob in the file, 1.68x
and 1.83x.** Without it (`LDS_PAD=0`) the B slab's rows are `BK` halves =
128 B apart, which is exactly one rotation of the 32 LDS banks, so a fragment
load touching 16 rows hits one bank 16 times: 7.93/7.88 ms against 4.73/4.53.
Eight halves of pad fixes it. Anyone building an LDS-staged dequant kernel
pays this before they pay anything else, and it is invisible in the
instruction stream (28 bytes of code and 1 KB of LDS separate the two).

**Finding 6: the coverage law loses to the traffic buying it costs.** The
stride arm sweeps `gcd(row bytes, 4096)` on the 4-bit bank, which is half the
bytes of the fp16 one and so lands somewhere else: gate_up's 1280 B row is at
gcd 256 *unpadded*, inside §5.1b's [128, 256] window, and down's 320 B row is
at gcd 64, below it — the end §3.5 measured costing 5-17%.

| layer | gcd | stride B | extra traffic | useful GFLOP/s |
|---|---|---|---|---|
| gate_up | 256 (natural) | 1280 | 1.00x | **14215** |
| gate_up | 128 | 1408 | 1.10x | 14151 |
| down | 64 (natural) | 320 | 1.00x | **14799** |
| down | 128 | 384 | 1.20x | 13826 |
| down | 256 | 768 | 2.40x | 12710 |

So on `down` climbing *into* the window costs 1.07x and 1.16x rather than
paying. The amendment to §5.1b rule 1 is: the window prices **bandwidth**,
and a kernel that is no longer bandwidth-bound has nothing to spend it on,
while the padding's extra bytes are charged either way. Read the rule as
"prefer a stride already in the window" — it is a layout choice, not a pad to
buy — and note that a 4-bit row is half an fp16 one, so the pad reaching a
given gcd is twice as large a share of it.

**Finding 7: grouping is worth *more* at Q4, not less.** §3.5's occupancy law
holds — the speedup is still monotone in workgroups per expert dispatch — but
every cell moves up: 5.2-8.1x at 10 workgroups where fp16 measured 3.2-4.2x,
2.8-2.9x at 30 where fp16 measured 2.6x. Which is what the occupancy
explanation predicts and the launch-cost one does not: the idle time an
under-filled dispatch leaves is roughly fixed, so it is a larger share of a
faster kernel. The per-expert loop gets *worse* the better the kernel gets.

**Finding 8: double buffering is worth nothing here** — 4.54 against 4.73 ms
on gate_up, 4.59 against 4.53 on down. It was worth 1.23x on
`gemm_wmma.comp`'s LDS path; the staged operand here is a quarter the size,
and it costs the last 4 VGPRs (256, at the wave64 ceiling) and half the
occupancy (4 subgroups per SIMD against 6).

**Finding 9, negative and worth keeping: folding the bias into the multiply
is wrong.** `fma(q, s, -1032*s)`, with the addend computed once per chunk,
would halve the dequant's arithmetic — and it **fails the correctness check
by 8%**, not by an ulp: -0.661 against -0.717. `1032*s` is ~145 where the
result `(n-8)*s` is ~1, and rounding that addend into fp16's 11-bit
significand puts 0.07 of error onto a quantity of magnitude 1. The bias has
to come off at nibble magnitude, before the scale, where it is exact. The
version that would work carries it out to the fp32 epilogue as a per-K-block
row sum of A — which is what `gemv_w4a8.comp` does, in int32, where it is
exact rather than merely bigger.

**A note on the clock.** The Q4 rows run at 2796-2870 MHz and 118-133 W
against fp16's 2880 MHz and 104-106 W: more arithmetic per byte is more
power, and this part answers that by dropping 1-3% of clock. The ratios above
are therefore very slightly *understated*.

**What is left here.**
- **The scale plane's layout.** Finding 4 is a 1.24x with no mechanism
  attached, from two identical binaries. The test is a blocked scale layout —
  one slab's 64 scales contiguous instead of 160 B apart — which would make
  QBLOCK=32 cost what QBLOCK=128 costs and settle whether it is locality.
  Accuracy wants the small block; this is what it would cost.
- **The dequant's two packed ops.** Finding 2 says the loop body is what
  binds now and finding 9 says the cheap way to halve it is invalid. The
  valid way is the epilogue row-sum; it needs a second buffer and a pass, and
  should be sized against finding 4 first.
- **The int8 arm is still unbuilt and is now much less interesting.** A
  `PRECISION_I8` `gemm_wmma.comp` halves operand bytes against fp16 — but Q4
  quarters them, has no matrix-rate penalty (§0.1), and is the format a real
  4-bit checkpoint arrives in. Build it only if a W8A8 prefill story is
  wanted for its own sake.
- **Decode.** All of this is prefill. The grouped *GEMV* for decode is still
  unwritten and is the phase `GOALS.md` is actually asking about.
**Effort**: was high; it was. **Value**: delivered — 2.10x on the largest
single number the file was carrying.

### 2.3 B stored [N,K] with a column-major `coopMatLoad` — **RESOLVED** ✅ (it was channel aliasing)
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

**[measured] The remaining hypothesis — channel/bank aliasing from the
power-of-two leading dimension — is confirmed, and it was the dominant
term.** B's leading dimension is now a push constant of its own (`pc.ldb`,
and `pc.lda` for A) rather than a reuse of `pc.K`/`pc.N`, so the row stride
can be padded at run time while the extents stay powers of two. Every padded
case below reuses its unpadded case's SPIR-V and differs in nothing but that
one push constant. `wmma_reg64_bt`, N=K=4096, as committed in
`results.csv`; five runs of this sweep agree to ≤2% on every row here, so
the ordering below is the finding and the third digit is not:

| pad (halves) | stride | stride mod 2 KB | GFLOP/s | vs unpadded |
|---|---|---|---|---|
| 0 | 8192 B | **0** | 8618 | — |
| 8 | 8208 B | 16 B | 11713 | 1.36x |
| 64 | 8320 B | 128 B | 13052 | 1.51x |
| **128** | 8448 B | 256 B | **14028** | **1.63x** |
| 256 | 8704 B | 512 B | 13132 | 1.52x |
| 512 | 9216 B | 1 KB | 11804 | 1.37x |
| 1024 | 10240 B | **0** | 9917 | 1.15x |
| 2048 | 12288 B | **0** | 9284 | 1.08x |

The effect is **periodic in the stride with a 2 KB period**, not monotone in
the pad: the three strides that are multiples of 2 KB are the three slowest
rows, every off-multiple stride beats all three, and the peak sits 256 B
past a multiple with a broad plateau from 128 to 512 B. 2 KB is exactly one
interleave rotation of eight 256 B channel chunks, so a fragment load whose
16 addresses are a multiple of 2 KB apart puts all 16 in the same channel —
and one K-strided fragment load is precisely that. Two controls place the
mechanism:

- **Padding a row-major B changes nothing** (`wmma_reg64_pad64`: 23538 at
  N=4096 against the unpadded kernel's 24530, and 0.2-2.7% lower at the
  other two sizes — inside the 23.5-24.9 TFLOP/s this kernel spans across
  five runs). Its B fragments are gathered element by element along N, so there
  is no 16-address strided load to de-alias. Padding per se is neither the
  fix nor a cost; the strided gather is.
- **Deepening the K-slab still does nothing, now that aliasing is out of the
  way** (`wmma_reg64_bt_k64_pad128`: 14026 vs `wmma_reg64_bt_pad128`'s
  14028 — the same number twice). Partial-line consumption is refuted a second time, on a kernel
  where it could no longer be masked.

One prediction made here was **wrong**, and instructively: the LDS path's
global B reads were expected to be immune, because they are contiguous
128-bit staging loads rather than `coopMatLoad` gathers. They are not —
`wmma_lds128_db_bt` goes 13286 → **20597** with the same pad (1.55x), which
closes essentially all of its gap to the row-major `wmma_lds128_db` (21214).
The reason is that a staging load is contiguous only *within* a row: thread
i reads 16 B from slab row i, and consecutive slab rows are `ldb*2` bytes
apart, so the wave's 64 addresses are the same aliased pattern. **What
matters is the address stride across the concurrent loads, not which
instruction issues them.**

**And the same aliasing was costing the kernels that already won.** A is
read with a K-stride in *every* variant here, so padding A alone (`_pada128`,
+256 B on A's stride, B untouched) speeds up the §2.1 winners — most at the
shape that was anomalously slow:

| kernel | N=1024 | N=2048 | N=4096 |
|---|---|---|---|
| `wmma_reg64` | 23229 | 20216 | 24530 |
| `wmma_reg64_pada128` | 24541 (1.06x) | **27260 (1.35x)** | 23747 (0.97x) |
| `wmma_reg64x128` | 18759 | 20008 | 25371 |
| `wmma_reg64x128_pada128` | 19595 (1.04x) | 23862 (1.19x) | 25498 (1.01x) |
| `wmma_wg128x256` | 18987 | 23363 | 25331 |
| `wmma_wg128x256_pada128` | 19858 (1.05x) | 24359 (1.04x) | **26008 (1.03x)** |
| `wmma_reg64_bt` | 11002 | 8070 | 8618 |
| `wmma_reg64_bt_padab128` (both padded) | **28377 (2.6x)** | 19127 (2.4x) | 15669 (1.8x) |

Unlike the stride sweep, these rows are only reproducible to 2-4% across
runs, and the pad-A effect at N=4096 is inside that: over four runs it is
+3-5% on `wg128x256`, +0.5-4.5% on `reg64x128` and −3 to +0.5% on `reg64`. So
the N=4096 claim is "small and kernel-dependent"; the solid ones are the
1.2-1.4x at N=2048 and the 1.8-2.6x on transposed-B, each reproduced in
every run.

So the §2.1 plateau was partly an aliasing artefact. **New bests: 28377
GFLOP/s (51% of the 55.5 TFLOP/s ceiling, up from 45%) and 26008 at N=4096
(47%)** — the first from the transposed-B kernel with both strides padded,
which is the fastest in the suite at N=1024 despite being the loser of the
original comparison, and closely matched by `reg64_pada128`'s 27260 at
N=2048. The N=2048 dip visible in every unpadded row (20-23 TFLOP/s against
24-25 at N=4096) was aliasing all along, which is why that shape moves most.

**What is still unexplained** — **[measured] and now explained; see §2.7,
which closed it and reversed it** — is the residue: with both strides padded,
the transposed-B kernel is still 1.6x behind row-major B at N=4096 (15669 vs
24530) while *beating* it at N=1024, so the footprint-dependent part of §2.3
survives de-aliasing. Two candidate explanations are already out:

- **Not request count.** Both kernels' *A* loads are the identical gather
  (16 rows, 32 B each, `lda*2` apart), so the difference is entirely in B —
  and the column-major B fragment is 2 `buffer_load_b128` where the
  row-major one is 64 scalar 2-byte loads. The faster kernel is the one
  issuing *more* and smaller requests.
- **Not cache-line survival, above MALL size.** A column-major fragment
  takes 32 B from each of 16 lines and needs the line to last three more
  K-steps, where the wave's four row-major fragments sweep a whole 128 B
  line per k. But deepening the K-slab so one slab consumes the full line
  measures as nothing at N≥2048, twice (`_bt_k64`: 8560 vs 8618;
  `_bt_k64_pad128`: 14026 vs 14028) — while *helping* at N=1024 (11002 →
  13321 unpadded, 17661 → 19212 padded). Line survival is real while the
  operand is cache-resident and is not the binding term once it isn't.

Discriminating what is left needs the memory system measured directly rather
than inferred through a GEMM: fixed bytes touched, sweeping request size,
request count and stride independently — §5.1b, which is the right place for
it. **[measured] That probe is now built, and it rules out the request-shape
explanation: see below.**

**[measured] And its traversal follow-up named the survivor, which §2.7 then
confirmed on the kernel.** The third thing ruled out above was request
*shape*; what was left was request *timing*. A coopmat fragment load holds
32 B of a row and a tiled kernel's waves retire after one, so C — the
contiguous run the in-flight requests hold — is an eighth of the `gcd` a
+256 B pad leaves. Hoisting a whole K-slab's loads raises C and nothing else,
and it takes this kernel from 15697 to **30609 at N=4096, past row-major's
24466**. The residue was the traversal. §2.7 has the ladder, the controls and
what it costs in registers.

**[measured] §5.1b has since measured the memory system directly, and it
corrects two things here without disturbing the conclusion.** The
strided-read probe reads a fixed set of bytes at a swept stride with no
tiling in the way (`shaders/strided_read.comp`), so it separates what this
sweep could not:

- **The period is 4 KB, not 2 KB.** With nine strides inside one rotation,
  10240 B and 14336 B — both multiples of 2 KB — read at the full rate,
  and only the multiples of **4 KB** are slow. The interleave is 256 B over
  **sixteen** channels, which is what a 256-bit LPDDR5X bus of 16-bit
  sub-channels gives. The mechanism named above is right; the rotation is
  twice as wide as three strides could show.
- **The decline past +256 B in the sweep above is footprint, not aliasing.**
  B is `N * (K+padB) * 2` bytes, so at N=K=4096 the unpadded case is
  *exactly* the 32 MiB MALL and every pad step pushes it further out: 8192 B
  → 32 MiB, 8448 → 33, 9216 → 36, 10240 → 40, 12288 → 48 MiB. At constant
  bytes touched the probe shows no such decline — a flat plateau from +128 B
  across a whole period, with only the 4 KB multiples down. So this sweep is
  the product of two terms: a de-aliasing step that is complete by
  +128-256 B, and a monotone footprint cost after it. Both are visible in
  it, which is why +256 B read as a peak rather than as the start of a
  plateau.

Two figures worth carrying: the *bus* penalty for a 4 KB-multiple stride is
at most **1.34x** and only for a gather (a contiguous read pays nothing),
while this kernel lost **1.63x** to it — so a GEMM amplifies the bandwidth
penalty rather than merely inheriting it. And request shape *per se* is free:
at a de-aliased stride a 64-address gather reads within 2% of a contiguous
sweep, which retires the leading candidate for the residue below.


**Engine implication, and it is free:** allocate every GEMM operand with its
leading dimension padded 256 B past a multiple of **4 KB** (§5.1b's
correction to the 2 KB written here originally, and +256 B is both the
measured optimum at K=4096 and the value that makes the rule shape-independent
— see §5.1b), including the weight tensors. It costs 128 halves of
padding per row — 3% of memory at K=4096 — needs no shader change now that
the strides are push constants, and is worth 0-5% at N=4096 (kernel-dependent, and inside the run-to-run
spread for one of the three), 4-35% at N=2048,
and 2.6x on the transposed-B layout that a real `Linear` weight already
has. Pad by that and no more: §5.1b shows the extra footprint is a real cost
and the extra de-aliasing is not. **[measured]** The pad composes with §2.7's
hoisting rather than being replaced by it — together they are 3.6x on the
transposed-B kernel at N=4096 — but note §2.7 also measures the pad's worth
*decaying* as the kernel holds more of a row in flight, from 1.44x at 32 B to
1.10x at 256 B. It stays free, so keep it; it is simply not the whole of the
stride story.
**Effort**: low (done). **Value**: high (delivered).

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
**Hypothesis** (§5.1b's mechanism 3, aimed at §2.3's leftover): the residue
§2.3 could not explain — transposed-B still 1.6x behind row-major at N=4096
*with both strides padded*, while beating it at N=1024 — is the same
coverage effect the strided probe measured, at the far end of its range. The
law is `coverage = min(1, C/gcd(stride, 4096))` with **C the contiguous run
of one row that the requests in flight hold**, and a 16x16 fp16 fragment
load holds **32 B** of each row it touches. A tiled GEMM's waves grab a
fragment and retire; they never walk along a row. Padding takes `gcd` to 256,
so even a de-aliased stride leaves C an eighth of it.
**Change**: raise C and change nothing else. `HOIST_A` / `HOIST_B` in
`gemm_wmma.comp` issue a whole K-slab of one operand's fragment loads before
the slab's first MMA, so a wave holds `32*BK_TILES` bytes of each row instead
of 32. The byte set, the MMA count, the tile shape, the accumulator grid and
the arithmetic intensity are all identical to the row each variant is
compared against.
**Expected**: if the mechanism is right, the gap moves and the transposed-B
kernel converges on row-major. If it is wrong, nothing happens — which is
exactly what the *existing* `_bt_k64` control already measured, twice.

**[measured] It moved, by more than the residue was worth.** New suite best:
**38990 GFLOP/s at N=2048 — 70% of the measured 55.5 TFLOP/s WMMA ceiling**,
up from 28510 and 51%. All figures from the committed `results.csv`; a second
independent run agrees to ≤2% on every row quoted here and ≤1% on the
headline ones.

The ladder, `wmma_reg64*`, WM=WN=4 (AI 32), every stride padded +256 B,
GFLOP/s. "C in flight" is the nominal rung — `32*BK_TILES`, what the source
asks for; the ISA table below has what the scheduler actually delivers, which
is lower:

| rung | C in flight | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|
| `reg64_bt_padab128` (none) | 32 B | 28510 | 18982 | 15697 |
| `reg64_bt_hkab2_padab128` | 64 B | **31504** | 20926 | 15167 |
| `reg64_bt_hka4_padab128` | 128 B | 28354 | **38990** | 30297 |
| `reg64_bt_hkb4_padab128` | 128 B | 27178 | 38424 | **30609** |
| `reg64_pada128` (row-major, none) | — | 24620 | 27895 | 24466 |
| `reg64_hka4_pada128` (row-major) | — | 21505 | 29070 | 26668 |

**§2.3's residue is gone, and it inverted.** At N=4096 the transposed-B
kernel was 15697 against row-major's 24466 — the 1.56x this file has carried
as unexplained since §2.3. At rung 4 it is **30609 against 26668, 1.15x
ahead**. The kernel with the better instruction stream is now also the faster
one, at every size, which is what §2.1's address-arithmetic diagnosis
predicted before the memory system got in the way.

**The control says it is concurrency, not depth.** `wmma_reg64_bt_k64` has
been in the suite since §2.3: BK_TILES=4, the same 64 `buffer_load_b128` and
64 `v_wmma` per loop body as the hoisted kernel, the same bytes — and no
hoisting. `RADV_DEBUG=asm` (via `cmd/probe`) counts the loads the scheduler
actually leaves outstanding before the first `s_waitcnt`:

| kernel | loads in flight | ≈ C | N=4096, unpadded |
|---|---|---|---|
| `reg64_bt` | 16 | 32 B | 8587 |
| `reg64_bt_k64` (depth, no hoist) | **16** | 32 B | 8734 |
| `reg64_bt_hkab2` | 32 | 64 B | 10358 |
| `reg64_bt_hka4` | **45** | ~90 B | **18363** |
| `reg64_bt_hkb4` | **45** | ~90 B | **19083** |

Deepening the slab without hoisting leaves the scheduler issuing exactly one
K-tile's worth of loads at a time, and measures as nothing — which is why it
measured as nothing twice before, and it was never a refutation of anything
except itself. Forcing the slab's fragments into a live register array is
what makes the loads concurrent, and that is worth **2.1x** over the
identical instruction stream. Note that the 2.1x above is measured
**unpadded**, on a stride that is a multiple of 4 KB — so this is §5.1b's
third engine rule arriving on a real kernel: a kernel whose stride is not its
to choose (a weight file mapped as-is) can buy its way out with depth. The
padded pair is the same size — 14152 → 30297 — though not exactly matched,
since `_bt_k64_pad128` pads only B where `_bt_hka4_padab128` pads both, so
take the unpadded pair as the clean one. It is also why the scheduler's 45
matters more than the nominal 128 B: the rung buys what it buys even short of
one full interleave rotation.

**Hoisting A and hoisting B are the same thing here**, 30297 vs 30609 and
18363 vs 19083, because at 252 VGPRs either one gives the scheduler the
headroom to pull the other forward too: both compile to 45 loads in flight
and 534 instructions. So this experiment does *not* attribute the residue to
an operand, only to the traversal. That is a limit of the lever, not a
finding.

**The differential control fires, and it is large.** Hoisting deepens A's
runs in both layouts (A is K-contiguous either way) but can only deepen B's
in the transposed one, because a row-major B fragment is gathered element by
element along N. At N=4096 the transposed-B arm gains **2.88x** from the
ladder (5604 → 16126, AI-16 grid) and the row-major arm **1.14x** (12363 →
14145). The lever acts on exactly the loads the mechanism says it should.

**And the padding benefit decays as C grows, which is the mechanism's own
signature.** The AI-16 grid (WM=WN=2) reaches rung 8, where C is 256 B — the
`gcd` a +256 B pad leaves, i.e. the rung at which the law says the stride
stops mattering. Ratio of padded to unpadded at N=4096:

| rung | C | unpadded | padded | pad worth |
|---|---|---|---|---|
| `reg32_bt` | 32 B | 5604 | 8042 | **1.44x** |
| `reg32_bt_hkab2` | 64 B | 11434 | 14157 | 1.24x |
| `reg32_bt_hkab4` | 128 B | 12939 | 11420 | *0.88x* |
| `reg32_bt_hka8` | 256 B | 16407 | 17811 | **1.09x** |
| `reg32_bt_hkb8` | 256 B | 16126 | 17792 | **1.10x** |

A stride pad is worth 1.44x to a kernel holding 32 B of a row and 1.10x to
the same kernel holding 256 B. That is the coverage law's prediction stated
as a trend rather than as a single number, and it is the closest this suite
comes to confirming the *mechanism* rather than the lever. Two caveats,
both honest: the same table at N=2048 does not decay monotonically (2.04x,
1.36x, 0.87x, 1.22x, 1.43x), and the rung-4 cell where padding *hurts* is
reproducible across runs at both sizes and is not explained. More loads in
flight also buys plain latency hiding, and this experiment does not separate
that from channel coverage; the decaying pad benefit is evidence for
coverage, the `_k64` control is evidence against depth, and neither rules
latency hiding out.

**What it costs, and where it stops: registers.** On wave64 an accumulator
is ~8 VGPRs and a fragment ~4, so WM=WN=4's sixteen accumulators take 128 of
the 256 a wave may hold. `RADV_DEBUG=shaderstats` prices every rung:
`reg64` 144 VGPR, `hkab2` 192, `hka4`/`hkb4` 252 — and hoisting *both*
operands at BK_TILES=4 spills (140 VGPRs, 3 KB of scratch), at BK_TILES=8
catastrophically (380-488 VGPRs, 70 KB). The ladder therefore stops at rung 4
at AI 32 and needs the four-accumulator AI-16 grid to reach rung 8. **This
makes §6.2 (wave32) materially more interesting than it was**: wave32 halves
the per-fragment register cost of the same tile, which is the exact currency
this lever spends.
**[measured] §6.2 has since run, and that last sentence is half right in a way
that matters.** The per-*fragment* cost does halve — a 16x16 fragment is the
native wave32 WMMA operand and wave64 carries it redundantly across the two
halves of the wave — but the per-accumulator cost does not, and RADV's 256
ceiling is in units that halve with the wave, so the absolute reach is
unchanged: `reg64_bt_hka4` fits wave64 at 252 VGPRs and **spills 62 at
wave32**, losing 18-25%. Where the lever does land is the fragment-dominated
AI-16 grid, where wave32 is worth **1.9-2.1x** at every size. So wave32 does
not extend this ladder; it makes a different, lower-intensity rung of it
competitive. See §6.2.

**Engine implication**: hoist the K-slab. It is a source-level change to the
inner loop with no layout, no shared memory and no host-side cost, it is
worth 2.1x on the [N,K] layout real `Linear` weights already have, and it
composes with the +256 B stride pad rather than replacing it — the two
together are 30609 against the unpadded, unhoisted kernel's 8587, **3.6x**.
Take it as far as the register file allows and no further; the cliff is a
spill, not a slope.
**Effort**: low (done). **Value**: high (delivered).

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

### 3.4 Benchmark the *real* shapes from the target models — **DONE** ✅, **and it moved two answers**
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

**[measured]** Now the `shapes` family: `bench/modelshapes.go` holds 60
weight matrices read out of the models' own configs (Z-Image from the local
checkout including the FFN width, which only its safetensors headers state;
the rest from their Hugging Face `config.json`), each with the token count it
runs at and how many times per forward pass. `bench/ops_shapes.go` runs the
W4A8 GEMV over every decode reduction length and the WMMA GEMM over every
deduplicated rectangle, and adds the whole thing up per model. 242 rows in
`results/shapes.csv`, ~5 minutes, median run-to-run spread 0.2%.

**What the square sweep got wrong is not a kernel, it is which variable is
free.** N×N×N ties the batch, the output width and the reduction length
together; every real layer separates them, and both of the suite's standing
answers turn out to depend on the separation.

**1. The load-width rule is satisfied for free at the real reduction
lengths, and `VEC=32` is dead.** §1.7's law wants a lane-step to cover
`gcd(rowBytes, 4096)`. The square sweep's power-of-two N made that gcd as
large as it can be; the models' reduction lengths are 640, 1024, 2560 and
6144, whose 4-bit rows are 320, 512, 1280 and 3072 B, with gcds of 64, 512,
256 and 1024. Measured DRAM-resident, GB/s of the 236 GB/s bus:

| K | rowB | gcd | `VEC=1` | `VEC=4` | `VEC=8` | `VEC=16` | who |
|---|---|---|---|---|---|---|---|
| 640 | 320 | 64 | 209 | **215** | 213 | 196 | MoE down-projections; Parakeet LSTM |
| 1024 | 512 | 512 | 211 | **239** | 237 | 222 | Parakeet joint |
| 2560 | 1280 | 256 | 239 | 241 | 241 | **242** | every Qwen projection |
| 6144 | 3072 | 1024 | 236 | **239** | 238 | 239 | attention and DeltaNet outputs |

At K=2560 — most of the decode traffic in `GOALS.md`'s text model — *every*
width reads 101-102% of the bus, the narrowest included. Only K=1024 gains
anything from a wider load (1.13x), and it is the one row where the law's
threshold actually binds. The largest reduction length any of the five models
decodes over is 6144, so the `VEC=32` item the last two handoffs carried
"for when a model needs it" has no model to need it: **retired**.

Two cells the law does not get right, in opposite directions. At K=6144 it
predicts `VEC=1` reaches 0.25 of the bus and it reaches 1.00 — the
below-threshold pessimism §1.7 already documented, and here total. At K=640
*nothing* reaches the bus: 209-215 GB/s, 88-91%, at every width, where the
law predicts full coverage from the narrowest load onward. 320-byte rows are
the shortest in the set and the only ones below the 4 KB rotation by more
than an order of magnitude; this is the one decode shape left with something
on the table, and it is 480 of a token's 1861 matmuls.

**2. Prefill has two winners, and which one applies is set by M alone.**
§2.7's `wmma_reg64_bt_hka4_padab128` wins **every** rectangle with M ≥ 1024
and **loses every rectangle below it** to `wmma_reg32_bt_hkab4_w32_padab128`
— §6.2's AI-16 wave32 arm, which the square sweep only ever crowned at
N=1024 and which is behind at N=2048 and N=4096. Across the 22 real
rectangles at M ≤ 512 it wins 21, by 1.06-2.80x; across the 16 at M ≥ 1024 it
wins none, losing by up to 2.8x the other way. The crossover is clean and it
is in the batch: N and K predict nothing once M is known.

The attribution is in the data too, because the same tile exists at both wave
sizes. Holding tile, hoist rung and pads fixed, wave32 is 1.2-2.3x the wave64
AI-16 arm at *every* shape measured (median 1.74x at M ≤ 512, 2.00x above) —
consistent with §6.2's 1.9-2.1x on that grid. And holding the wave at 64, the
AI-16 tile beats the AI-32 tile 1.43x at M=128 (11631 vs 8138 at N=K=768).
So it is both: the smaller tile for the occupancy a short batch cannot
otherwise fill, and wave32 for the fragment registers. One exception worth
keeping: at M=128 with N=9728 the wave32 arm *loses* 0.63x, the only
small-M cell it does, so the rule is "small M", not "small work".

**3. The padding problem is M, not K.** Every K in all five models is a
multiple of 64 — the predicted "K not a multiple of TILE_K" hazard does not
occur once, and the kernels never have to pad a reduction. Every N is a
multiple of 64 except Parakeet's 8198-wide joint output (vocab 8193 plus
five duration classes), which pads to 8224 for 0.3% waste. The one shape that
hurts is the MoE expert at prefill: routing 2048 tokens over 512 experts
gives each **40 rows**, which a 64-row tile runs at **62% useful**, and the
best kernel on it returns 7600 useful GFLOP/s — 14% of the WMMA ceiling and
5.1x behind the same matrix run densely at M=2048. That is §3.5's case made
from measurement rather than from first principles.
**[corrected by §3.5]** The 62% is right and it is not the cost, and the
other two numbers are against the wrong ceiling. Both of these cells were
measured against one expert's weights re-read out of the 32 MiB MALL; the
real thing streams 512 experts from DRAM, where the ceiling is 8.2 TFLOP/s
and the grouped kernel reaches 93% of it. §3.5 also removed the padding: a
16-row tile takes the same shape to 84% useful and runs *slower*, because at
this shape the bytes bind and the FLOPs do not.

**4. The budget, which is the thing the shapes were written down for.**

| model / phase | q4 weights | GFLOP | flop/byte | mem-bound share | dispatches |
|---|---|---|---|---|---|
| qwen3.8-flash-next decode | 2.89 GB | 11.6 | 4.0 | 100% | 1861 |
| qwen3.8-flash-next prefill (2048 tok) | 61.79 GB | 21067 | 341 | 98% | 74148 |
| parakeet encoder (30 s audio) | 0.49 GB | 1269 | 2586 | 0% | 336 |
| kokoro (128 phonemes) | 0.05 GB | 38 | 830 | 0% | 90 |
| z-image transformer (1024², 1 step) | 5.09 GB | 53455 | 10498 | 0% | 391 |
| z-image text encoder (128 tok) | 1.82 GB | 930 | 512 | 0% | 252 |
| qwen3-embedding (512 tok) | 0.35 GB | 1533 | 4352 | 0% | 280 |

One matrix's own intensity against 4-bit weights is `4*M` and nothing else,
so the aggregate ratio is the wrong statistic for a mixed pass — hence the
mem-bound column, the share of a pass's weight bytes sitting in matrices
below the 235 flop/byte crossover. It says the thing the totals hide:
**prefill of the MoE model is 98% memory-bound**, because the batch reaches
the experts divided by 512.

One matrix in that table is worth naming on its own. The text model's LM head
is 248320 x 2560, **312.6 MB at 4 bits**, and it is measured here at its real
size rather than through a twin — genuinely DRAM-resident, 240-242 GB/s at
every load width, **1.36 ms**. That is 11% of a token's weight traffic in a
single dispatch, and it is the one matrix a speculative or multi-token
decoder would amortise rather than pay per token.

At the best measured DRAM-resident decode rate (242 GB/s) the text model
reads 2.89 GB per token = **11.9 ms = 84 tok/s** before attention, and its
1861 matmul dispatches add 0.56 ms of launch cost on top (5%, at §4.1's
measured 300 ns) — the first time that number has been large enough to
matter, and it is an argument for §3.5's grouped GEMM (which collapses 1440
expert dispatches into 48) rather than for §4.2's pre-recorded command
buffers.

**What this re-aims.** §3.5 (grouped/MoE GEMM) is now the highest-value item
in the file on two independent counts — the 5.1x at M=40 and the 1440
dispatches — where it was previously argued for from the model card alone.
**[done; and both counts were the wrong reason.]** The grouped kernel is
worth 1.1-4.1x, the mechanism is occupancy rather than either padding or
launch overhead, and the dispatch count was never the lever — see §3.5.
§1.3/§1.7's "carry the load-width rule to the other GEMV kernels" keeps its
value but loses its urgency: the widths that matter at real K are the narrow
ones those kernels already issue. And §2.4's swizzle finally has a
discriminator worth running it on, since small-M shapes are where tile
scheduling, not bandwidth, should show up.

### 3.5 Grouped / MoE GEMM — **DONE** ✅, **1.1-4.1x**, and it is occupancy
Qwen3-Next is an MoE model: each token goes to a few of many experts, so
the FFN is a **grouped GEMM** over variable-sized token batches, not one
big GEMM. Benchmark: top-k routing, the gather/scatter of token rows, and a
grouped-GEMM kernel that reads per-expert (offset, count) from a buffer.
Also measure the "one dispatch per expert" naive alternative to quantify
the launch-overhead penalty (§4.1).
**Effort**: high. **Value**: high, and specific to the stated text-gen goal.

**[measured]** Built as `-DGROUPED=1` on `gemm_wmma.comp`: one extra binding
holding a tile table of (row in the gathered activations, row in the
`[E*N, K]` expert bank, column in the output), and the tile origin read out
of it instead of derived from `gl_WorkGroupID`. Nothing below the origin
changes — same inner loop, same operand layout, same hoisting rung, same VGPR
counts from `RADV_DEBUG=shaderstats` — so the grouped and per-expert arms run
*the same binary over the same table*, differing only in whether it is one
dispatch or 512. (The per-expert arm needed a command buffer whose dispatches
carry different push constants, which is `vk.DispatchSequenceTimed`.) The
`moe` family, `results/moe.csv`, 512 experts top-10 at 1/512/2048/8192 tokens.

**Finding 1: grouping is worth 1.08-4.14x, and the cause is occupancy, not
launch cost.** §3.4 sized this item on a dispatch count — 1440 of a token's
matmuls collapsing to 48 — and that turns out to be the wrong lever. Sort
every (kernel, layer, batch) cell by how many workgroups *one expert's*
dispatch launches, and the speedup falls off monotonically:

| workgroups in one expert's dispatch | grouped / per-expert |
|---|---|
| 10 | 3.18-4.14x |
| 20 | 1.58-3.27x |
| 30-41 | 1.12-2.54x |
| 80-119 | 1.09-1.27x |
| 209-837 | 0.75-1.14x |

80 waves is exactly what §0.1 measured this 40-CU part needs resident to
issue WMMA at rate, and one workgroup here is one wave. So a per-expert
dispatch is slow when it cannot fill the machine, the barrier after it makes
that idleness serial, and the effect is gone once an expert's own tile count
reaches occupancy. §4.1's 300 ns launch is 1.5% of a 20 µs dispatch and
explains none of it. Two cells at the top of that table **invert**
(0.74-0.75x, reproducibly across three runs, both `moe.gate_up` at 8192
tokens in the two narrowest-BN kernels) and are not explained.

**Finding 2: the tile padding §3.4 flagged is real and costs nothing.** At
40 rows an expert fills 62% of a 64-row tile; a 16-row tile takes that to
84%. It is *slower*:

| kernel | tile | AI | useful | gate_up GFLOP/s | down GFLOP/s |
|---|---|---|---|---|---|
| `reg64_bt_hka4` | 64x64x64 | 32 | 62% | **7596** | 6092 |
| `reg32_bt_hkab4` | 32x32x64 | 16 | 67% | 6484 | 5895 |
| `reg32_bt_hkab4_w32` | 32x32x64 | 16 | 67% | 6536 | 6055 |
| `reg16x32_bt_hkab4_w32` | 16x32x64 | 11 | 84% | 6890 | 6116 |
| `reg16x64_bt_hkab4_w32` | 16x64x64 | 13 | 84% | 7146 | **6164** |

Because the padding costs *FLOPs*, and at this shape FLOPs are free: the
grouped kernel moves 190 GB/s of a 236 GB/s bus, so what a smaller tile buys
in useful rows it more than loses in operand bytes re-read. §3.4's "62% of
that dispatch is tile padding" is true and is not the problem; **"14% of the
WMMA ceiling" was the wrong ceiling.** Against the memory-bound ceiling the
fp16 bank implies — 1.68 GB of weights plus 252 MB of activations in 8.2 ms
at 236 GB/s, i.e. 8.2 TFLOP/s useful — the grouped kernel is at **93%**.
§3.4's "5.1x behind the same matrix at M=2048" compared a MALL-resident
40-row dispatch against a MALL-resident 2048-row one and is retired.

**Finding 3, which nobody asked for: the +256 B stride pad is the wrong
statement of its own rule.** See §5.1b's amendment below — these are the
first two non-power-of-two reduction lengths anything in this suite has run a
GEMM at, and they separate "pad by 256 B" from "land on a gcd of 256". The
down projection's 640-wide rows are already at gcd 256 unpadded, and the pad
every other kernel here carries takes them to 512 and costs **1.19x**.

**Finding 4: the gather and combine are 5% of the block, not free and not
the problem.** At 2048 tokens the gather is 0.90 ms (368 GB/s — above the
DRAM bus, because it re-reads a 10 MB token buffer ten times out of the MALL)
and the combine 0.52 ms, against 28.1 ms of matmul.

**Finding 5: the budget.** One MoE block at 2048 tokens: 29.99 ms grouped
against 38.49 ms expert-at-a-time, and 5 dispatches against 1538. Times 48
layers that is **1.44 s against 1.85 s** for a prompt chunk, with the stride
fix taking the grouped figure to 28.09 ms/block. All of it at fp16, where the
bank alone is 5.03 GB per block and 21.3 ms of unavoidable DRAM traffic — so
**the whole prefill is a bandwidth problem and the case for a Q4 grouped
kernel (§2.2) is now quantified rather than argued.** (§2.2 has since built
it: 13.37 ms against 28.13, and the bandwidth problem becomes a matrix-core
one on the way.)

**Not done here, and since done in §1.8:** decode's grouped path was measured
with this GEMM kernel at M=1, which wastes 15 of every 16 rows of an MMA tile —
free on weight bytes, since they are read once either way, but not free on A's,
which the tile height multiplies. The honest kernel is a grouped *GEMV*, it now
exists, and it is **1.30-3.26x this one** at the decode batches, with the
activation traffic (2-6 GB/s against 17-41 GB/s) and the 7-11% useful rows
being where the difference is. The T=1 cases here touch 10 experts and 33 MB,
inside the 32 MiB MALL, so those rows were cache-resident as well; §1.8 sweeps
the decode batch out to 315 MB for that reason.

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

**[measured] §2.3 added a second axis to this family and one specific probe
it should own. That probe is now built, and it found two mechanisms rather
than one — the larger of which nothing in the suite was looking for, because
it needs neither a gather nor a kernel to trigger.**

§2.3 measured a GEMM gaining up to 1.6x purely from moving an operand's row
stride 256 B off a power-of-two, but the `bandwidth` family only ever
measures a contiguous sweep, so the 805/236 GB/s ceilings it reports are
best-case figures a strided kernel does not automatically get — and a GEMM
cannot separate the stride from its own tiling.

**Built**: `shaders/strided_read.comp` + `bench/ops_stride.go`, run as the
`stride` family. The pattern is `rows` rows of which the first `rowBytes`
are touched, spaced `stride` bytes apart, one `uvec4` per lane. Every
touched 16-byte chunk is read exactly once in every variant, so the byte
set, the footprint and the number of `buffer_load_b128`s are identical
across the sweep and only two things move:

- **the stride**, which adds a gap and changes nothing about what is read —
  exactly what padding a real matrix's leading dimension does, including
  that the pad bytes are never read;
- **the request shape**, via `LANES_PER_ROW`: how many consecutive 16-byte
  chunks of one row go to consecutive lanes, so one 64-lane load spans
  `64/LANES_PER_ROW` distinct rows. 1 row is a contiguous 1 KB sweep, 64
  rows is the 64-address/16-bytes-each gather, 32 rows is what a 16×16 fp16
  `coopMatLoad` issues.

Each case is checked against a host checksum over its (row, chunk) set,
computed from the geometry and deliberately *not* from the shader's lane
mapping, so a variant that covered a chunk twice or skipped one fails
instead of reporting a plausible bandwidth.

#### Mechanism 1: channel coverage — 2-4x, and it hits contiguous reads

The interleave on this part is 256 B across **sixteen** channels, i.e. a
**4 KB rotation** (what a 256-bit LPDDR5X bus of 16-bit sub-channels gives).
Row *k* starts at `k*stride`, so row starts land only on the multiples of
`g = gcd(stride, 4096)` inside one rotation, and each row covers `rowBytes`
from its start. When `rowBytes < g` the union of them **never addresses some
channels at all**, and the achievable fraction of peak is exactly the
fraction of channels touched:

> **coverage = min(1, rowBytes / gcd(stride, 4096))**

1024 B rows, 64 MiB touched, GB/s (`model` is the formula; rows are how many
distinct rows one request spans):

| rows/req | 1024 | 1280 | 2048 | 3072 | 4096* | 5120 | 9216 |
|---|---|---|---|---|---|---|---|
| **model** | 1.00 | 1.00 | **0.50** | 1.00 | **0.25** | 1.00 | 1.00 |
| 1 (contiguous) | 243 | 241 | **120** | 242 | **61** | 245 | 239 |
| 4 | 243 | 241 | 122 | 243 | 61 | 245 | 239 |
| 16 | 234 | 238 | 113 | 233 | 55 | 237 | 231 |
| 32 | 230 | 239 | 113 | 234 | 58 | 238 | 232 |
| 64 (gather) | 232 | 240 | 120 | 237 | 59 | 242 | 235 |

The model holds to within 2% at every point, for every shape. Note which
strides are the bad ones — 2048, 4096, 8192, the ones a tidy allocator picks
— while 1280 and 3072 are perfect. At 2048 B rows (a K=1024 fp16 weight row)
it is the same law one notch weaker: 243 GB/s at stride 2048, **123 at 4096**,
242 at 6144, **121 at 8192**.

None of this is visible to anything §2.3 measured: a K=4096 fp16 row is
8192 B, already ≥ any `g`, so its coverage is full at every stride.

#### Mechanism 2: per-request aliasing — up to 1.32x, and it needs a gather

With coverage full (8192 B rows), a stride that is a multiple of 4 KB costs a
contiguous read *nothing* and costs progressively more the more rows one
request spans. 64 MiB touched, GB/s:

| rows/req | 8192* | 8208 | 8256 | 8320 | 8448 | 8704 | 9216 | 10240 | 11264 | 12288* | 14336 | 16384* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (contiguous) | 243 | 238 | 238 | 240 | 242 | 243 | 244 | 244 | 243 | 242 | 243 | 243 |
| 4 | 231 | 226 | 228 | 230 | 236 | 240 | 244 | 245 | 242 | 232 | 245 | 233 |
| 16 | 205 | 210 | 216 | 229 | 236 | 237 | 237 | 238 | 236 | 202 | 236 | 202 |
| 32 | 204 | 205 | 214 | 238 | 235 | 236 | 238 | 238 | 236 | 201 | 236 | 200 |
| 64 (gather) | **182** | 198 | 230 | 237 | 230 | 233 | **239** | 237 | 238 | **174** | 238 | **180** |

(* = multiple of 4 KB. Identical to within 1% at 256 MiB touched, so this is
a property of DRAM and not of the footprint.)

**The period is 4 KB, not the 2 KB §2.3 inferred.** With nine strides inside
one rotation the sweep is unambiguous: 10240 and 14336 **are** multiples of
2 KB and run at the full rate, while 8192, 12288 and 16384 — the multiples of
4 KB — are the only slow strides. And recovery inside a period is complete by
+128-256 B, then flat: +16 B gets a third of it, +64 B most of it, and
everything from +128 B to +3072 B is within noise of the best.

#### The MALL behaves differently on both counts

At 16 MiB touched a pure read delivers **930-965 GB/s** — above the 805 GB/s
§0.4 measured with a read+write copy — and:

- it is **immune to the gather penalty**: the 64-row gather measures 824-896
  GB/s at *every* stride, aliased or not, i.e. a flat ~0.93x of the other
  shapes rather than DRAM's 1.32x spread;
- **the coverage law does not carry over to it**, and what replaces it is not
  yet pinned down. Six cases at 16 MiB touched, all with the same bytes read:

  | row | stride | coverage | span | measured |
  |---|---|---|---|---|
  | 1024 B | 3072 / 5120 / 9216 | 1.00 | 48-144 MiB | 780-968 (MALL) |
  | 2048 B | 2048 | 1.00 | 16 MiB | 782-945 (MALL) |
  | 2048 B | 4096 | 0.50 | 32 MiB | 769-924 (**MALL, no penalty**) |
  | 1024 B | 2048 | 0.50 | 32 MiB | 353-469 (**halfway**) |
  | 2048 B | 8192 | 0.50 | 64 MiB | 118-121 (DRAM × 0.50) |
  | 1024 B | 4096 | 0.25 | 64 MiB | 55-61 (DRAM × 0.25) |

  Read down it: **full coverage keeps the set resident however far it is
  spread** — a 16 MiB working set scattered over a 144 MiB span still gets
  MALL bandwidth, which is worth knowing on its own. **Partial coverage keeps
  it only while the span is also inside 32 MiB**, and past that the DRAM law
  reappears undiminished. The one case that is neither (1024 B rows, 2048 B
  stride: partial coverage, 32 MiB span) lands halfway. The plausible reading
  is that the address bits selecting a channel also select a cache slice, so
  partial coverage concentrates the touched lines onto a fraction of the sets
  and shrinks the effective capacity — but that is inference, and pinning it
  needs its own experiment: a footprint sweep at a fixed bad stride, which
  the `stride` family can already run from flags.

  Note the direction of the engine consequence: **§5.1b's own "keep it under
  32 MiB for 3.4x" premise only holds for a fully-covering access pattern.**
  A blocked panel with a bad stride does not get the MALL either.

One MALL-resident oddity looked reproducible and unexplained: the *purest*
pattern — the fully contiguous shape at zero pad, i.e. a plain linear sweep —
measured **782 GB/s in all three row-length groups** while every
fully-covering perturbed stride of the same shape reached 805-966. It was the
one cell where adding a pad helped a contiguous read, it came out at exactly
782 three times in one committed run, and two earlier runs put it at 794-839
twice and 942 once. **Mechanism 3 has since retired it, and not as
scatter**: at 1 KB rows the walk and cross-wave shaders compile to the *same*
mapping, and in the last run before the fix below that identical pair
measured 782 and 942. What 782 actually marked was *position* — it was the
first timed case of each 16 MiB group, in all three row lengths to three
digits, and only of the cache-resident ones (the 64 MiB groups start at a
normal 243). The buffer is filled by the host immediately before, so the
first cache-resident pass measures the residue of that fill rather than its
own access pattern; writeback competing for DRAM is the likeliest cause, and
the clock differs by 1% between the twins so it is not that. Measurement
lesson, not a memory-system finding: **discard the first cache-resident case
after a host fill.** `RunStride` now runs and throws away one case per (row
length, footprint) group, and those cells read 884-950 in the committed run
— inside the ~10% one-sided scatter every cache-resident case here carries —
with the identical twins 2.3% apart.

#### Mechanism 3: the traversal — the coverage law is about what is *in flight*

**[measured]** The one pattern the sweep above could not reach was
concurrent-but-contiguous requests from *different* waves at an aliased
stride: every shape there varies the addresses inside one request, and the
contiguous shape deliberately walks along a row, so consecutive waves are
never a stride apart. `CROSS_WAVE` on `strided_read.comp` swaps the two loop
orders over the (row-group, column-block) grid so consecutive requests
advance down the rows instead of along one. It is a permutation of exactly
the same (row, chunk) set — same bytes, same footprint, same instruction
count, same host checksum — and the only thing that moves is which addresses
are outstanding at the same moment.

It is the largest effect in this family, it **subsumes mechanism 1**, and it
fires with no padding involved at all.

8192 B rows, 64 MiB touched, contiguous 1 KB requests, GB/s:

| stride | 8192* | 8208 | 8256 | 8320 | 8448 | 8704 | 9216 | 10240 | 11264 | 12288* | 14336 | 16384* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gcd(stride,4096)` | 4096 | 16 | 64 | 128 | 256 | 512 | 1024 | 2048 | 1024 | 4096 | 2048 | 4096 |
| **model** | **0.25** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.50** | 1.00 | **0.25** | **0.50** | **0.25** |
| walk along a row | 243 | 238 | 238 | 240 | 242 | 243 | 244 | 244 | 243 | 243 | 243 | 243 |
| cross-wave | **63** | 208 | 220 | 217 | 229 | 236 | 238 | **124** | 238 | **63** | **123** | **62** |

The model is the same `min(1, C/gcd(stride, 4096))` as mechanism 1 with one
substitution: **C is not the row length, it is the contiguous run of bytes
that the requests outstanding at one moment hold in a single row.** Walking
along a row, consecutive requests continue where the last one stopped, so C
is the whole row and the law reads exactly as §5.1b first stated it.
Cross-wave, consecutive requests are a stride apart, so C is only what one
request holds — here 64 lanes × 16 B = 1024 B. It predicts every cell of
that row to within 2% (0.26/0.51/0.26/0.51/0.26 measured against
0.25/0.50/0.25/0.50/0.25), and a 3.9x loss on a plainly contiguous read at
the stride a K=4096 fp16 matrix has by default.

**The ladder that proves the mechanism and exonerates the GEMV kernels.**
`LOADS_PER_WAVE` makes each request issue *n* loads along its own row before
retiring, which is what a real kernel's inner loop does. Same bytes, same
strides, 8192 B rows, 64 MiB, GB/s:

| bytes of a row in flight per wave | 1 KB | 2 KB | 4 KB | 8 KB (whole row) |
|---|---|---|---|---|
| **model at a 4 KB-multiple stride** | 0.25 | 0.50 | 1.00 | 1.00 |
| stride 8192* | **63** | 122 | 239 | 238 |
| stride 12288* | **63** | 121 | 238 | 237 |
| stride 16384* | **62** | 121 | 237 | 237 |
| stride 8448 (de-aliased control) | 229 | 229 | 238 | 238 |

C is a property of the *wave*, not of the instruction: doubling what one
wave has outstanding doubles the bandwidth, and at 4 KB — one full
interleave rotation — the penalty is gone at every stride. The 2048 B-row
group runs the same ladder one rung shorter and says it again: at a dense
2048 B stride, one wave holding 1 KB of its row reads 120 GB/s and one
holding the whole 2 KB reads 237. The top rung is
one wave streaming a whole row with consecutive waves on consecutive rows,
which is precisely the GEMV/W4A8 access pattern, and it reads at the full
238-244 GB/s at every stride in the sweep. That is the answer to the
question this item opened: **the GEMV shape is not exposed** — consistent
with W4A8 measuring 89% of the bus against a 2048 B row stride — and the
1.12x that bounded the decode side of it is not there to be had.

At the top rung there is only one column group, so the two traversals
compile to the same mapping; those cells are printed with `=` and are a
repeatability check. At 64 MiB they come out at 1.00x in all twelve, which
also **retires the 782 GB/s oddity above**: the same degenerate pair at 16
MiB measured 782 and 942, and 782 turned out to be where in the *sweep* a
case sat (the first cache-resident case after the host fills the buffer),
not what it read. The harness now discards that case, and the committed run
has no 782 in it.

**What is exposed is a dense, unpadded tensor.** §5.1b's first corollary —
"a densely packed tensor is never at risk, because `stride == rowBytes`
forces `gcd ≤ rowBytes`" — was derived with C = rowBytes and does not
survive this axis. 64 MiB touched, **no padding anywhere**, GB/s:

| rows | stride | in flight per row | walk | cross-wave | model |
|---|---|---|---|---|---|
| 1024 B | 1024 | 1024 B | 243 | 243 | 1.00 |
| 1024 B | 1024 | 256 B | 243 | **61** | 0.25 |
| 2048 B | 2048 | 1024 B | 243 | **120** | 0.50 |
| 2048 B | 2048 | 256 B | 246 | **31** | 0.125 |

A tensor with nothing wrong with it, read by waves that each hold 256 B of a
row, runs at an eighth of this chip's DRAM bandwidth. The defect is in the
kernel's traversal, and the fix is in the tensor's stride.

**The model is exact for contiguous requests and only indicative for
gathers**, and it misses in *both* directions there. Where it predicts a
severe penalty the gathers beat it by 1.4-8x — at stride 8192 the
64-address gather measures 0.03x against a model of 0.004 — because their
rows start at every multiple of `gcd`, so neighbouring requests partly refill
the lines the model counts as untouched. Where it predicts no penalty at all
they fall short of it, to 0.53-0.93x: the second mechanism's per-request
cost is still there, and cross-wave it is far larger than the 1.32x a walking
traversal shows. Both are outside the law and neither affects a contiguous
request, which is the shape real kernels issue and where the model holds to
2%.

**The MALL is not immune to this one.** §5.1b found the MALL indifferent to
request *shape*; it is not indifferent to traversal. At 16 MiB touched,
2048 B rows at a dense 2048 B stride, the walk shapes read 868-942 GB/s and
the cross-wave traversal **484** with 1 KB in flight (model 0.50) and **128**
with 256 B (model 0.125) — the same law with the same C, scaled to the
MALL's own peak. The ladder's top rung recovers it there too: a wave holding
the whole 2 KB row reads 928. It does not explain every MALL cell — the same
rows at a 4096 B stride are at model 0.50 and measure 774-919 — so the
slice-structure question this item already had stays open, but the traversal
penalty itself carries across the cache.

#### The answer to (b) retires §2.3's open question

At a de-aliased stride a 64-address gather reads at **239 GB/s against a
contiguous sweep's 244** — within 2%, over identical bytes. **Request shape
costs essentially nothing once the stride is right.** So:

- The [N,K] layout a real `Linear` weight already has is **fully usable above
  MALL size**; there is no bandwidth reason to transpose weights at load
  time.
- The residue §2.3 could not explain — transposed-B still 1.6x behind
  row-major at N=4096 *with both strides padded* — is **not** a
  memory-request-shape effect, which was the leading candidate. §2.3 ruled
  out request count and cache-line survival; this rules out request shape.
  What is left is in the kernel, not the bus.

Worth carrying: the *bus* penalty for an aliased stride is at most 1.32x and
only for a gather, while §2.3's kernel lost 1.63x to it — a GEMM **amplifies**
the bandwidth penalty rather than merely inheriting it.

#### Engine rules, all three free

**One law, stated once:** the fraction of peak bandwidth an access pattern
can reach is

> **coverage = min(1, C / gcd(stride, 4096))**, where **C** is the contiguous
> run of bytes, inside one row, that the requests outstanding at any one
> moment hold.

Mechanisms 1 and 3 are that law with two different C's — the whole row when
consecutive waves walk along it, one wave's own run when they do not — and
it holds to within 2% wherever C is a genuine contiguous run. Mechanism 2,
the ≤1.32x per-request gather penalty, is a separate and much smaller
residue on top.

1. **Choose the row stride so that `gcd(stride, 4096)` is 128 or 256 B.**
   That is one rule covering all three mechanisms: the gcd is at or below
   the bytes *any* request shape in this family holds in one row, so
   coverage is full however the kernel traverses the tensor, and the stride
   is off the 4 KB multiple, so gathers de-alias.

   **[amended by §3.5]** This used to read "pad every leading dimension to
   256 B past a multiple of 4 KB", which is the same rule only when the
   unpadded stride is *already* a multiple of 4 KB — and every reduction
   length this suite had swept until §3.5 was a power of two, so the two
   statements could not be told apart. §3.5's MoE shapes are the first
   non-power-of-two ones, and they separate them. Sweeping the gcd over
   every power of two, at fixed tile, on two kernels and two shapes whose
   natural strides start at opposite ends (grouped `moe` family, 2048
   tokens, GB/s of the 236 GB/s bus):

   | gcd(stride, 4096) | 16 | 64 | **128** | **256** | 512 | 1024 | 2048 |
   |---|---|---|---|---|---|---|---|
   | gate_up K=2560, reg64 | 181 | 186 | **192** | 190 | 190 | 173 | 133 |
   | gate_up K=2560, reg16x64 | 149 | 165 | **179** | 179 | 173 | 137 | 89 |
   | down K=640, reg64 | 167 | 165 | 175 | **175** | 152 | 118 | 98 |
   | down K=640, reg16x64 | 164 | 166 | 182 | **183** | 155 | 114 | 86 |

   A plateau at 128-256 B with a cliff on **both** sides: over-fine is worth
   5-17%, and gcd 2048 is worth 1.4-2.0x. The 640-wide down projection's
   rows are 1280 B, gcd 256, already on the plateau — and adding the +256 B
   pad takes them to 1536 B, gcd 512, for a measured **1.19x loss**. So the
   pad is a *consequence* of the rule at power-of-two K, not the rule. An
   engine should compute the gcd and pad only if it is outside [128, 256].
   Note the confound is broken by construction: gate_up's gcd-1024 row is
   the *unpadded* stride and down's gcd-256 row is too, so "large gcd is
   slow" is not "large stride is slow".

   **[amended again by §1.8, and the two amendments point opposite ways]**
   The plateau above is a property of a *tiled* read. §1.8 ran the same sweep
   on the grouped GEMV, where a wave holds a whole weight row and consecutive
   workgroups continue the previous one's address, and every pad in it is
   64 B-aligned so the padding bytes are never read — a pure coverage
   experiment at constant traffic. There the **unpadded stride wins at both
   shapes**, and on the 320 B `down` row it wins at gcd **64**, below the
   plateau: padding into [128, 256] costs 1.12-1.15x and padding to gcd 1024
   costs 2.4x. So rule 1 applies to kernels whose requests span a *fraction*
   of a row; a kernel that walks whole matrices wants its rows contiguous and
   nothing else. An engine that knows which kernel will read a tensor should
   pad for the tiled one and leave the streamed one alone — and an MoE expert
   bank, which decode streams and prefill tiles, is read both ways, which is
   a trade this file has not priced.

   Mechanism 3 promotes this from a 1.3x tidy-up to the rule that defends
   against a **4x**, and against a case that needs no padding to occur (see
   the corollary below).
2. **Never round a row stride up to a page.** A K=1024 fp16 weight matrix
   aligned to 4096 B reads at **half** this chip's DRAM bandwidth; 1024 B
   rows at an 8192 B stride, a **quarter**. It is a loss a tidy allocator
   inflicts on itself while looking like it is doing the right thing.
3. **Do not over-pad, and do not under-pad either.** Recovery is complete by
   +128-256 B and then flat, so more pad buys nothing and costs footprint —
   which is what made §2.3's GEMM sweep appear to *decline* past +256 B (see
   the correction there). §3.5 adds the other end: a pad that takes the gcd
   *below* 128 B costs 5-17%, so "as small a gcd as possible" — which is what
   the coverage law on its own would recommend, since coverage saturates at
   1 — is also wrong. The coverage law predicts the order of these rows and
   gets it wrong at both ends; what it bounds is the cliff, not the plateau.

Three corollaries worth stating on their own, because they decide where
rule 1 actually earns its 3%:

- **A dense tensor is at risk after all** — the corollary this item first
  drew ("`stride == rowBytes` forces `gcd ≤ rowBytes`, so coverage is 1 by
  construction") was right about the row and wrong about what the law
  measures. It holds only for a kernel whose concurrent waves walk along a
  row. One wave per row with 256 B in flight reads a *dense* 1024 B-row
  tensor at **61 of 243 GB/s**, and a dense 2048 B-row tensor at 31. What
  saves the suite's GEMV kernels is not the packing, it is that each of
  their waves streams a whole row (mechanism 3's ladder), and W4A8's 89% of
  the bus is that and not luck.
- **Any kernel that reads a *strip* of a larger tensor is exposed, and that
  is what tiling is.** A blocked GEMM that sweeps a 1 KB-wide panel of a
  K=4096 fp16 matrix has C ≤ 1024 against a stride of 8192, so `gcd` is 4096
  and coverage is **0.25** — a quarter of DRAM bandwidth, for a kernel doing
  nothing wrong except tiling. Padding that matrix's stride to 8448 restores
  coverage to 1 for *every* panel width and every traversal (its `gcd` is
  256). This is the same regime as the measured 1024 B/4096 B point above,
  not an extrapolation from it. It also makes rule 1 a prerequisite for §2.4
  and for the MALL-blocking half of this item rather than a 3% optimisation:
  block a panel out of an unpadded weight matrix and the blocking can cost
  more than it saves.
- **A kernel with a bad stride can also buy its way out with depth.** Four
  KB of a row in flight per wave — one full interleave rotation — restores
  full bandwidth at every stride measured, without touching the layout. That
  is the escape hatch when the tensor's stride is not the engine's to choose
  (a weight file mapped as-is), and it is a second reason to prefer wide
  loads and deep unrolling in the inner loop beyond §1.3's instruction count.

**Effort**: low (done). **Value**: high (delivered) — the probe was written
to quantify a 1.35x, found a 2-4x cliff next to it, and then found that the
cliff is really about concurrency and can reach 8x.

**What this leaves open** is the original half of this item: deliberately
scheduling work to stay inside the 32 MiB MALL (attention K/V tiles,
per-expert MoE weights, a blocked GEMM B-panel), and the slice-structure
question under it — partial coverage keeps a working set resident only while
its span is also inside 32 MiB, one case lands halfway, and the traversal
axis has now added MALL cells the coverage law fits (2048 B rows at a dense
2048 B stride, cross-wave: 484 against the 868-942 the walk shapes read
there, model 0.50) next to cells it does not (the same rows at a 4096 B
stride, walk: 774-919 at model 0.50). A footprint sweep at a
fixed bad stride is still the experiment that separates effective capacity
from bus width, and the `stride` family already runs it from flags
(`-stridefootprints 4,8,16,24,32,48 -striderowbytes 1024
-stridepads 0,1024,3072`).

Also worth a cheap look now that traversal is a measured axis: **§2.3's
unexplained residue.** A coopmat fragment load holds 32 B of a row, so a
tiled GEMM sits at the far end of mechanism 3, and its waves do not walk
along rows — a mechanism §2.3 did not have. Padding to `gcd` 256 does not
fully clear that end of the range: the 32- and 64-row cross-wave shapes read
0.87x and 0.73x of their walk counterparts *at a de-aliased stride*, which
is the right order of magnitude for a residue §2.3 measured at 1.6x and
could not attribute.

**[measured] Done, in §2.7, and the prediction held.** Raising C inside the
WMMA kernel — a whole K-slab of fragment loads in flight instead of one
K-tile's worth, same bytes and same instruction counts — is worth **2.1x**
on the transposed-B kernel, closes the 1.6x residue outright and reverses
it, and takes the suite's best GEMM to **38990 GFLOP/s, 70% of the WMMA
ceiling**. Two things there are worth carrying back here. The stride pad's
value *decays with C* exactly as this law says it should (1.44x at C=32 B,
1.10x at C=256 B), which is the first confirmation of the mechanism from
outside this probe. And the escape hatch in corollary 3 is real on a real
kernel: the 2.1x is available with no padding at all, so a weight file whose
stride the engine does not control can still be read at rate by a deep enough
inner loop. What §2.7 could *not* do is separate coverage from plain latency
hiding, since more loads in flight buys both.

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
**Hypothesis**: this device reports `minSubgroupSize=32`,
`maxSubgroupSize=64`, `subgroupSizeControl=true` with compute in
`requiredSubgroupSizeStages` — so we can *choose*. Everything was written for
wave64 (`local_size_x = 64`). RDNA's WMMA 16×16×16 shape maps naturally onto
wave32, and wave32 halves the latency of a dependent instruction chain and
reduces divergence cost; wave64 halves instruction issue count. §2.7 promoted
this item by ending on the register file: its ladder stops at rung 4 because
hoisting both operands spills the 256-VGPR wave64 budget, and wave32 was
supposed to halve the per-fragment cost of the same tile, which is the exact
currency that lever spends.
**Change**: `WAVE` became a `-D` on every kernel whose workgroup is one
subgroup (`gemm_wmma.comp`, `gemv_subgroup.comp`, `gemv_w4a8.comp`,
`gemv_w8a8.comp`, `rmsnorm_subgroup.comp`, `softmax_subgroup.comp`), and the
host pins the pipeline to the matching size with
`VK_EXT_subgroup_size_control`'s `requiredSubgroupSize` plus
`REQUIRE_FULL_SUBGROUPS` (`vk/shim.c`, `PipelineSpec.RequiredSubgroupSize`).
`cmd/probe` takes the size as a second argument, so `RADV_DEBUG=shaderstats`
can price a variant at either. The wave64 SPIR-V is byte-identical to what it
was before `WAVE` existed, verified with `cmp`.
**Expected**: 0-20% either way, per kernel.

**[measured] The range is 0.58x to 2.13x, and which end a kernel lands on is
predictable.** Three families, three different answers.

**1. The GEMM: up to 2.13x, but not where §2.7 wanted it.** Same tile, same
accumulator grid, same hoist rung, same strides, same bytes — only the wave
size differs. GFLOP/s, wave64 → wave32:

| kernel (padded unless noted) | N=1024 | N=2048 | N=4096 |
|---|---|---|---|
| `reg32_bt` | 11693 → 20606 (1.76x) | 9011 → 10422 (1.16x) | 8034 → 8354 (1.04x) |
| `reg32_bt_hkab4` | 17200 → **32946** (1.92x) | 13569 → **28292** (2.08x) | 11417 → **24313** (2.13x) |
| `reg32_bt_hka8` | 18338 → 30776 (1.68x) | 24321 → 22035 (0.91x) | 17537 → 18552 (1.06x) |
| `reg64_bt` | 28408 → 31502 (1.11x) | 18946 → 18237 (0.96x) | 15667 → 14707 (0.94x) |
| `reg64_bt_hkab2` | 31664 → **33603** (1.06x) | 20966 → 19952 (0.95x) | 15178 → 15922 (1.05x) |
| `reg64_bt_hka4` | 28174 → 25730 (0.91x) | **39039** → 32127 (0.82x) | **30933** → 23297 (0.75x) |
| `reg64_hka4` (row-major) | 21494 → 26114 (1.21x) | 28905 → 31495 (1.09x) | 26558 → 28092 (1.06x) |
| `reg64_bt_hka4`, *unpadded* | 13205 → 23892 (1.81x) | 22176 → 15297 (0.69x) | 18494 → 15097 (0.82x) |

A second independent run agrees to ≤1.5% on every cell quoted, and the wave64
column reproduces the committed `results.csv` row for row.

**wave32 takes N=1024 and loses N=2048 and N=4096.** The new best at N=1024 is
`reg64_bt_hkab2_w32_padab128` at 33603 (60% of the 55.5 TFLOP/s ceiling),
against wave64's 31664. At the two larger sizes §2.7's `reg64_bt_hka4` /
`_hkb4` pair still wins at **39039** and **31183**, and wave32 costs them
18-25%. So the suite's headline number does not move; what moves is which
kernel to pick at which size.

**Why: the fragments get cheaper at wave32 and the accumulators do not.**
`RADV_DEBUG=shaderstats` prices every variant at both sizes, and the first
thing to establish is the unit, because RADV's wave32 VGPR figure is not
directly comparable to its wave64 one. The calibration is the three
*non-coopmat* kernels, whose register use per unit of work cannot change with
the wave size: `gemv_subgroup_f16`, `gemv_w4a8` and `gemv_w4a8_v4` all report
**48 → 96**, exactly 2x, with "Subgroups per SIMD" halving 32 → 16. That 2x is
the no-change baseline. Against it:

| variant | w64 VGPR | w32 VGPR | 2x w64 | w32 sg/SIMD | spill |
|---|---|---|---|---|---|
| `gemv_w4a8_v4` (control) | 48 | 96 | 96 | 16 | — |
| `reg32_bt` | 48 | 72 | 96 | 16 | — |
| `reg32_bt_hkab4` | 144 | 168 | 288 | 9 | — |
| `reg32_bt_hka8` | 192 | **192** | 384 | 8 | — |
| `reg64_bt` | 144 | 192 | 288 | 8 | — |
| `reg64_bt_hkab2` | 192 | 256 | 384 | 5 | — |
| `reg64_bt_hka4` | 252 | 256 | 504 | 5 | **62** |
| `reg64_bt_hkab4` | spills (140, 3 KB) | 256 | — | 5 | **324**, 12 KB |

Every coopmat kernel comes in *under* 2x, and the more fragment-heavy it is
the further under: `reg32_bt_hka8` holds eighteen fragments against four
accumulators and reports the identical 192 at both sizes, i.e. half the
per-work cost. That is §6.2's hypothesis, confirmed — a 16×16 fragment is the
native wave32 WMMA operand shape and wave64 carries it redundantly across the
two halves of the wave. **But the 256 ceiling is in the same reported units**,
so the absolute reach is not extended: `reg64_bt_hka4`, which fits wave64 at
252, spills 62 at wave32. The lever is real and it is aimed at fragments; it
does not lift the roof.

Which is exactly what the timings say. The AI-16 grid is fragment-dominated
(four accumulators), so at wave32 rung 4 becomes nearly free and is worth
**1.9-2.1x at every size**. The AI-32 grid is accumulator-dominated (sixteen),
so wave32 buys it little and costs it a spill at the rung that wins.

**2. The GEMV kernels: a uniform loss, 0.58-0.95x.** GB/s at N=4096, wave64 →
wave32: fp32 148 → 132, fp16 407 → 258, q8 199 → 116, q4 105 → 70, w8a8
508 → 390, w4a8 418 → 326, w4a8 vec4 563 → 491. These kernels are one
workgroup per output row and the workgroup *is* the subgroup, so halving the
wave halves the lanes sweeping each row while the grid stays at M workgroups —
half the threads, half the requests in flight, and no register benefit to show
for it since (per the table above) their register cost per unit work is flat.
The GEMM pays the same halving and wins anyway; these have nothing to buy with
it. Two caveats: these are the cache-resident square sweep, so they carry the
one-sided dropout noise README documents for MALL-resident cases — one wave32
cell in the committed run (`gemv,subgroup_w32,w4a8,block=64,N=1024`, 61.5 GB/s)
is such a dropout and reran at 270-281; and a second run put a different cell
(`subgroup_vec4_w32` at N=2048) low instead. Read the *pattern*, not any one
cell.

**3. The DRAM-resident decode kernel: 1.07x, and it is the one number here
that moves an engine decision.** `gemv_cold` at a 68 MB W4A8 weight matrix —
twice the MALL, so every iteration refills from DRAM — over three runs:

| kernel | run 1 | run 2 | run 3 | % of the 236 GB/s bus |
|---|---|---|---|---|
| `subgroup_vec4` (wave64) | 209.8 | 210.2 | 212.8 | 89% |
| `subgroup_vec4_w32` | **225.6** | **226.3** | **225.1** | **96%** |
| `subgroup` (wave64) | 182.2 | 182.5 | 181.0 | 77% |
| `subgroup_w32` | 177.8 | 177.1 | 179.1 | 76% |

Under 1% spread across runs, and the wave64 rows reproduce the committed
`results.csv` to 0.1%. **The wide-load W4A8 GEMV at wave32 reads 225.7 GB/s of
the 236 GB/s bus** — the decode path's best number, up from 89%. The effect is
specific to the wide-load arm: at `VEC=1` wave32 is 0.98x, i.e. nothing. So it
is not the wave size on its own, it is the wave size at a load width that
already saturates each lane; and it runs the opposite way to the cache-resident
square sweep above, which is why the two families needed separate rows. This is
the counter-example to the "wave32 halves your threads and your requests"
reading that explains family 2 — at 16 B per lane, 32 lanes is already enough
requests to fill the bus, and the shorter wave wins on something else. Not
attributed further; §5.1b's coverage law predicts wave32 *worse* here
(C 1024 B → 512 B), so whatever this is, it is not coverage.

> **[superseded by §1.7]** It is coverage, applied to the right variable. The
> three probes this section asked for were run, and they say the wave size is
> not the cause: hold C — the bytes of a weight row one lane-step holds —
> fixed and vary the wave size, and the bandwidth is the same to 0.05%. The
> 1.07x is a *deficit* in this cell (N=4096 at `VEC=4`, where a wave64 step
> holds exactly half a row, so every wave in the grid addresses the same half
> of the 4 KB rotation at once), not a win in the wave32 one, and it does not
> occur at N=2048 or N=8192. Raising C to a whole row instead —
> `VEC = N/(8·WAVE)` — reads 99-103% of the DRAM bus at every N, including
> 234.6 GB/s at N=8192 where the kernel measured here gets 164. See §1.7.

**4. It also settles §3.7.** The subgroup reductions at N=4096: `shared`
223.7 GB/s at 256 threads per row, `subgroup` 68.5 at 64, `subgroup_w32` 35.7
at 32. Halving the wave halves the bandwidth, to 0.52x — the third point on a
line through the origin in *threads per row*. §3.7 guessed the subgroup
variants were latency-bound on too few lanes; this measures it. The fix is
threads, not reduction ops, and there is nothing wave-size-shaped to tune.

**Engine rule**: it is a per-pipeline knob and the default should stay 64.
~~Pin 32 for the DRAM-resident W4A8 decode GEMV at `VEC=4`~~ — **withdrawn by
§1.7**: that 1.07x was the `VEC=4` cell being wrong at wave64, and widening
the load beats both wave sizes. Pin 32 for a fragment-heavy low-intensity
GEMM tile if one is ever the right shape. Leave it at 64 everywhere else, and
never pin it on a kernel that spills at 32 — that is the one case where it
costs 25%.
**Effort**: low-medium (done). **Value**: medium — one real decode win, one
retired hypothesis, and §3.7 closed as a side effect.

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

**Next:**
- **Explain the down projection's 198 GB/s against gate_up's 235**, for
  identical bytes per expert. It launches 4x the workgroups, each a quarter as
  long, and issues 4x the output writes. §1.8 killed the obvious explanation:
  at VEC=4 its 640-nibble row is 20 lane-steps for a 64-lane wave, but the
  wave32 arms — 20 steps over 32 lanes — measure within 1%. Candidates left are
  the workgroup count and the write pattern; the probe is a build that gives
  one workgroup several output rows. **§1.9 has narrowed it by half**: its pad
  slots price one more output row at constant weight traffic, and down's slot
  costs 2.1x gate_up's where its rows are 4x as many and its reduction a
  quarter as long — an ordering the dot products get backwards and a fixed
  per-output-row cost (one `subgroupAdd`, one store) gets right. The probe is
  now the same one from both directions: the `ROWS` knob the non-grouped
  kernel already has, which `GROUPED` forbids, and which would amortize that
  fixed cost over several rows instead of several tokens.
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

**Then (the missing pillars):** §3.3 attention/flash-attention and §3.1-3.2
fusion (justified by DRAM round-trips, not launch overhead). §3.3 in
particular now has a working register-blocked WMMA kernel to build its two
matmuls out of, and §3.6 (gated DeltaNet) is the last primitive in
qwen3.8-flash-next that nothing here covers. §3.4 and §3.5 are done.

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
compiled in once.**
