<!-- §0 — extracted verbatim from IDEAS.md. The section number is the
     stable identifier cited from shaders/, bench/, cmd/, vk/ and TODO.md.
     Do not renumber it. Edit this file, not a copy in IDEAS.md. -->

[← IDEAS.md](ideas.md) · [research index](README.md) · measurement validity

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


### 0.6 The WMMA ceiling re-pinned, and the clock under a real kernel — **done** ✅ (KERNELS.md G0, 2026-09-29)

**Hypothesis** (KERNELS.md G-o1): 478 FLOP/clk/CU might be the probe's
number, not the hardware's — §0.1's loop has 4 dependent chains and a
loop counter, and the spec says 512.
**Change**: the same loop at 8 and 16 chains (`-DACC`) and at wave32
(`-DWAVE`, pinned with `requiredSubgroupSize`), five new `peak` rows.
**[measured] It is the hardware's.** 481 / 481 / 480 / 479 / 478
FLOP/clk/CU (4, 8, 16 chains; wave32 at 4 and 8), int8 482: flat to 1%
and **15/16 of 512 exactly**. 55.5 TFLOP/s at 2899 MHz stands.

**[measured] But no real kernel runs at 2899.** A 100 ms sampler on
`freq1_input`/`power1_average` around H3's `TestGPUShapes`:

| load | sclk mode | package W |
|---|---|---|
| the probe (registers only) | 2890–2899 | 100–112 |
| the suite's 20 ms GEMM bursts (`gemm_wmma.csv`) | 2813 | 137 |
| **a 480p video forward** | **2600–2650** | **149** |
| a 768p forward, sustained ~2 min | 2550–2600 | 142–155 |
| a 480p forward beside 32 busy CPU threads | **1300–1400** | 121 (GPU) |

The package is limited at ~150 W; a GEMM's memory traffic costs ~40 W
over the probe and the clock gives 9–10%. So the ceiling *during a
forward* is 49.9 TFLOP/s and the shipped GEMM's 37 there is **74% per
clock**, not 67% of 55.5. And the CPU draws from the same budget: 32
busy threads take a forward from 30.0 to **50.1 s** (768p: 106.7 →
190 s). That is VIDEO.md M11f's "+2.4 s of device time beside the
32-thread audio decode" — power, not memory contention. Every
utilisation this repo quotes from here on carries the clock it ran at
(KERNELS.md decision 1).

### 0.7 The energy roofline — **opened** (KERNELS.md G0b, 2026-09-29)

**Observation**, from rows the harness had already recorded (`power_w`):
the `gemm_wmma` ladder at N = 2048 draws 136–137 W at every rung from 23
to 39 TFLOP/s. Under a ~150 W package cap (§0.6), rate is cap ÷ energy
per operation, so the ladder's winners are the rungs that spend fewer
joules a FLOP. Energy per operation, PPT above a ~20 W idle:

| | rate | PPT | GFLOP/J or GB/J | pJ each |
|---|---|---|---|---|
| WMMA fp16, registers only, 16 chains | 55.5 TFLOP/s | 98 W | 567 | 1.4 a FLOP |
| the same at 4 chains | 55.5 | 112 W | 495 | +14 W of loop issue |
| `dot4_int8` / packed fp16 FMA / fp32 FMA | 54 / 25 / 23 T | 118–121 | 450 / 215 / 189 | |
| MALL-resident copy | 800 GB/s | 105–136 | 6–8 | ~100 a byte |
| DRAM stream | 236 GB/s | 136 | 1.7 | **~490 a byte** |
| best suite GEMM | 39.0 TFLOP/s | 137 | 285 | |
| a video forward | 32.5 TFLOP/s | 149 | 218 | |
| LLM prefill, 8,192 tokens, 24 layers | 2,680 tok/s | 142 (2713 MHz) | ≈110 mJ a token at 48 layers | |
| LLM decode, 24 layers | 63.2 tok/s | 91 (2590 MHz) | ≈2.9 J a token at 48 layers | memory-bound |

A DRAM byte is ~340 matrix FLOPs of energy; a MALL byte ~70. The matrix
cores at full rate cost ~80 W over idle, leaving ~50 W under the cap for
every byte and every non-MMA instruction. **Open**: the pJ a byte at L0,
L1 and L2 (the `bandwidth` family stops at 2 MB footprints), and the
sensor split (`power1_label` is `PPT`, the package; the CPU's share is
in `/sys/class/powercap/intel-rapl:0`, so GPU = PPT − CPU).
