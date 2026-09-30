<!-- TODO.md P19. The decode step re-attributed on the shipped banks,
     2026-09-30: family, bytes, ms, GB/s, lost; the weightless dispatches;
     the host; one row and three; the clock. Cited from cmd/llm -gen -attrib,
     cmd/llm -batch and cmd/llm -graph, llm/gpu_head.go, llm/gpu.go. -->

[← TODO.md](../TODO.md) · [← LLM.md](llm-vertical.md) · [research index](README.md) · [P1](p1-decode-attribution.md) · [P16](p16-decode-arena-width.md) · [C5](../CONCURRENCY.md)

# P19 — the decode step, re-attributed: 26.7 ms in four parts, and two of the projections are still on the GEMM

**Result.** A decode token is **26.66 ms** on the shipped banks at depth
zero (37.5 tok/s through `-gen -attrib`, 256 tokens, two runs 26.68 and
26.64 ms, every label within 0.014 ms of itself). It is four parts and the
plan's reconstruction (17 / 5–6 / 3.3 / 1–2) was close:

    weight streams at 242 GB/s, the floor at this format     16.4 ms
    weight streams running under the bus (against 227)        4.3 ms   (5.4 against 242)
    dispatches that stream no weight                          3.0 ms
    the host, outside the command buffer                      1.1 ms
                                                             -------
                                                             26.7 ms

**And the largest two losses are not GEMV kernels.** The lm head and the
hyper-connection's up projection never got D15's treatment: at one row they
still run `llm_gemm.comp` with a sixteen-row fragment holding one row — the
head at **185 GB/s** (2.37 ms, the single largest dispatch of the step) and
`hc.up` at **125 GB/s** (1.75 ms over 97 dispatches). Together they are 4.1
ms of the step at a rate the decode GEMV beats by 15–70%. The head's GEMV
exists, was laddered at 210 GB/s in L8c (`results/l8c_head.csv`) and was never
wired into the graph; that is the first change below. `hc.up` has no GEMV,
because its bank is the ten-group record (`q4kScaleMin12`) and its epilogue
collapses the four streams — that is P21's first kernel.

## The instrument

`cmd/llm -gen -attrib -n 256 -prompt 'The capital of France is'`, 48 layers,
`LLM_DENSE_BANK`/`LLM_MOE_BANK` at the shipped plans and `LLM_BANK_CACHE` set
(P1's instrument, `research/p1-decode-attribution.md`), the `ai` user service
resident but idle (its swap slot empty, 9 GB), nothing else on the GPU, a
100 ms sampler on `hwmon2` beside it. The 256-token decode is what the
sampler needs: at `-n 64` the decode is one second and the hwmon average
never sees it.

The tables are `results/p19_decode_attrib_run{1,2}.csv` (the two 48-layer
runs), `p19_batch_rows.csv`, `p19_graph_rows123.csv`, `p19_head_gemv.csv`,
`p21a_hc_up_gemv.csv`, `p21b_hoisted_gemvs_dead.csv`,
`p22_shexp_fold_{split1,fold1,split2,fold2,folddown}.csv`,
`p21c_down_ladders.csv`, `p21c_moe_gemv_hoist_{ctl,hoist}{A,B,C}.csv` and
`p21c_moe_up_grouped_dead_{v1,v2}{A,B}.csv`.

The two 24-layer runs made first (16.20 and 15.29 ms a step; the difference
is all in the host gather, which at 64 tokens carries the two ~8 ms fault
tails P17 found) reproduce every per-layer label to the microsecond, so the
24-layer prefix is a valid instrument for anything per layer — but not for
the ratios, where the fixed costs (head, ple, host) are 25% of a step instead
of 14%.

**Bytes are the staged bank's** (P4a), from the tensor shapes and the plans:
deltanet at `q4_k/32` (0.5625 B a weight), hyper-connection, head, attention
and indexer at `q5_k/32` (0.6875), routed gate/up as the checkpoint's Q4_K
(0.5625), routed down at IQ4_NL (0.5625), the shared expert's gate/up at
`q4_k` and its down at `q5_1` (0.75), the router as halves. `dn.qkv` is the
fused `attn_qkv` + `attn_gate` (2560 × 16384); `attn.qkv` is q, k, v
(2560 × 13312) and may or may not carry the indexer's 1.1 MB, which moves its
rate 199 → 207.

## Where the token goes — the weight streams

Two 48-layer runs, means; `lost` is against 227 GB/s, the best any dispatch
in this graph has measured (L8c-7). The router's 2.9 MB a layer lives in the
MALL by design (D12) and is excluded from the total.

| label | n/pass | MB/step | ms/step | µs each | GB/s | of 242 | ms at 227 | lost |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| moe.up | 48 | 884.7 | 4.576 | 95.3 | 193 | 80% | 3.897 | +0.68 |
| dn.qkv | 36 | 849.3 | 4.030 | 111.9 | **211** | 87% | 3.741 | +0.29 |
| moe.down | 48 | 442.4 | 2.537 | 52.8 | 174 | 72% | 1.949 | +0.59 |
| **head.head** (GEMM) | 1 | 437.1 | 2.366 | 2366.0 | **185** | 76% | 1.926 | +0.44 |
| **hc.up** (GEMM) | 97 | 218.5 | 1.748 | 18.0 | **125** | 52% | 0.963 | **+0.79** |
| dn.out | 36 | 318.5 | 1.631 | 45.3 | 195 | 81% | 1.403 | +0.23 |
| attn.qkv | 12 | 281.1 | 1.416 | 118.0 | 199 | 82% | 1.238 | +0.18 |
| hc.down (GEMV s32) | 97 | 218.5 | 1.351 | 13.9 | 162 | 67% | 0.963 | +0.39 |
| moe.shexp.up | 48 | 88.5 | 0.708 | 14.8 | 125 | 52% | 0.390 | +0.32 |
| moe.router | 48 | 141.6 | 0.698 | 14.5 | *203* | *84%* | 0.624 | *+0.07* |
| attn.out | 12 | 129.8 | 0.662 | 55.2 | 196 | 81% | 0.572 | +0.09 |
| moe.shexp.down | 48 | 59.0 | 0.476 | 9.9 | 124 | 51% | 0.260 | +0.22 |
| ple.kv | 1 | ≤36 | 0.276 | 276.0 | ≤130 | ≤54% | 0.159 | +0.12 |
| **total, router excluded** | | **3963** | **21.78** | | **182** | **75%** | 17.46 | **+4.32** |

Read down the `lost` column and the step's rate gap has a shape:

- **The four big GEMVs are near the bus.** `dn.qkv` at 211 GB/s, `dn.out`,
  `attn.qkv`, `attn.out` at 195–199: between them 0.8 ms against 227 and
  they are the kernels P21 named first. The Q4/Q5 arm of `llm_gemv.comp`
  *as compiled* is not the four-scalar-load shape the plan suspected: RADV
  vectorises the record refresh to one `buffer_load_b128`, the two payload
  dwords to a `b64`, the sixteen activation halves to two `b128`; a step is
  seven loads for eight weight bytes a lane, and the loop ends in
  `s_waitcnt vmcnt(0)`. There is something there, but not 5 ms of it.
- **The MoE is 1.3 ms under**, `moe.up` at 193 and `moe.down` at 174 — the
  latter reads a 9.2 MB expert set in 53 µs through 1600 workgroups of one
  quad a lane. Both are C5's quad-load kernel; the loss is dispatch shape,
  not load count.
- **The two GEMMs are 1.2 ms under**, and the head alone is the largest
  dispatch of the step at 2.37 ms.
- **The small dispatches are half the bus or less**: `hc.up`, `hc.down`, the
  shared expert's two, `ple.kv` — 1.9 ms of the step for 620 MB, each a
  1–2.3 MB dispatch of 10–18 µs. At 242 GB/s a 2.25 MB dispatch is 9.3 µs; the
  rest is launch, first latency and tail, paid 340 times a step. This is
  P8's rule (decode is latency, not bandwidth) at the scale of a whole
  family: the fix is fewer, fatter waves that issue their whole slab at
  once, or fewer dispatches (the shared expert as an eleventh tile of the
  routed dispatch, P22).

## The dispatches that stream no weight — 3.0 ms

1419 dispatches a pass; 25 of the 38 labels stream no weight and add up to
3.03 ms, of which the sum kernels behind the split-K GEMVs are 0.26:

    dn.scan 0.636   move 0.537 (196)   moe.route 0.275   hc.cn 0.232
    attn.attn.split 0.153   moe.combine 0.148   dn.conv 0.146   moe.perm.up 0.136
    hc.down_reduce 0.122   dn.norm 0.116   attn.score 0.084   dn.out.sum 0.061
    moe.router.sum 0.058   dn.qkv.sum 0.058   attn.pack 0.058   dn.hist 0.049
    attn.attn.combine 0.049   attn.idx 0.038   attn.out.sum 0.020   attn.qkv.sum 0.020
    ple.gate 0.016   hc.norm 0.008   hc.combine 0.004   ple.conv 0.004   ple.hist 0.001

P16's count stands: the 196 moves are 0.54 ms and L6c's single arena is the
largest single item; `dn.scan` at 17.7 µs a layer is the recurrence and the
only one of these above 10 µs.

## The host — 1.1 ms

    gather (ple)    0.601   the n-gram table (0.598 in run 2)
    hand-over       0.236   submit + fence, minus the GPU time inside it
    sample          0.160   argmax over 248 320 logits
    record          0.098
    unattributed    0.051
    detokenize+emit 0.008

At 256 tokens the gather is 0.6 ms a token, not P17's 1.3: the ~8.7 ms
fault tails are there (run 1 of the 24-layer pair read 1.32 ms at 64
tokens) but amortised. P23's 1 ms is really 0.6 at any length a client
generates.

## The clock

The sampler (100 ms, hwmon2 `freq1_input` / `power1_average`, samples above
60 W = the decode) over both 48-layer runs: **97 W mean, p10/p50/p90 76 /
102 / 115 W; the shader clock 1105 MHz mean, p10/p50/p90 727 / 795 / 2758
MHz**. A decode step spends most of its time with the shader clock near its
floor and the power near 100 W: the step is memory-bound as the plan said,
and there are no matrix clocks to buy here. (The 24-layer runs read the same:
96 W, 1072 MHz.) The 3 W / 760 MHz idle floor is what the sampler sees between
tokens.

## One row and three

**Three conversations** (`cmd/llm -batch 1,2,3 -batch-depth 7000`, C5's
instrument, one slot a row):

| rows | ms | tok/s | steps | `moe.up` | `moe.down` | head |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.1 | 36.9 | 1.00 | 4.6 | 2.5 | 2.4 |
| 2 | 37.7 | 53.0 | 1.39 | | | 2.4 |
| 3 | 45.3 | 66.2 | 1.67 | 14.6 (303 µs) | 6.1 (128 µs) | 2.4 |

**One sequence** (`cmd/llm -graph -tokens 1,2,3`, P20b's instrument — a
verification pass's rows share experts):

| rows | ms | steps | GPU | `moe` block | `deltanet` | `hyper-conn` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.9 | 1.00 | | | | |
| 2 | 34.4 | **1.23** | 32.9 | 16.1 | 7.4 | 3.6 |
| 3 | 38.2 | **1.37** | 37.0 | 19.7 | 7.8 | 3.7 |

So a two-row verification pass costs **1.23 steps** today (P5c's was 1.36),
and P20's arithmetic follows: with a free draft, one drafted token breaks
even at **23% acceptance**, two at 37/2 = 19% a token. The head is 2.4 ms at
every row count in both tables, because at two and three rows it is the GEMM
either way.

## Finding 1 — the head at one row goes on its GEMV

`HeadGPU` staged a `llm_gemv.comp` K1 pipeline since L8c and only the bench
and a test ever set it (`SetGEMV`); the graph's `Resize(1)` picked
`OutGEMMKernelFor(1)`. Now `plan` puts one row on a quantised bank on the
GEMV (`GraphOpts.HeadGEMM` and `LLM_HEAD_GEMM=1` keep the GEMM, the control
arm), and a `SetGEMV` pins whichever arm it names.

The GEMV carries one fewer rounding a weight than the GEMM (L8c-4's
argument; rms 2.0e-4 relative on the block test, `TestHeadGPUQ4Gemv`), so
**four gates that compare one-row logits with multi-row ones to the last
place pin the GEMM** — `TestGraphForwardRows`, `TestGraphCheckpointIsThePrefill`,
`TestGraphSlotsAreSequences`, `TestGraphDecodeRowsIsEachSlot` — since they
test the graph's mechanics and not the head's kernel, and
`TestGraphHeadDecodeGEMV` prices the kernel on the whole 4-layer graph at a
tolerance and asserts the argmax.

**Measured, same hour, interleaved** (`-gen -attrib -n 256`, 48 layers, A =
the tree before this change, B = after; A's third arm ran beside a `go
build` and reads 1.7% slow on every label, which is P4c's lesson again and
is why it is quoted apart):

| arm | ms a step | tok/s | head |
|---|--:|--:|--:|
| A1, A2 | 26.68, 26.64 | 37.49, 37.53 | `head.head` 2.368, 2.364 (185 GB/s) |
| **B1, B2** | **26.37, 26.37** | **37.92, 37.92** | `head.head_gemv` **2.061, 2.065** (212 GB/s) |
| A3 (beside a compile) | 27.10 | 36.90 | 2.407 |

**−0.30 ms a step, 37.5 → 37.9 tok/s (+1.1%)**, the head at 212 GB/s, which
is L8c's ladder number to the percent; every other label within 0.01 ms of
A. It is the smallest of the items below and the only one that was wiring.

## Finding 2 — `hc.up` on a GEMV (P21a): 125 → 165 GB/s, and where the rest went

`shaders/llm_hc_up_gemv.comp`: one workgroup a feature block (nEmbd/16 =
160 of them), a lane a column of the block — lane c·16 + r is stream c of
feature r, which is what `packUpB`'s permutation makes of a 64-column block —
so the four-stream collapse is three shuffles into lane c = 0, summed in the
GEMM epilogue's order (c = 0, 1, 2, 3), and the wave stores its sixteen
`mixed` values. The record is the ten-group one (`q4kScaleMin12`, made
branch-free), `lo` is staged in LDS for the workgroup, ROWS is P5b's
specialization constant. `HCUpGemv`, mode 3 of `hcVariants`; `hcQ4DecodePlan`
names it on the K-quant banks (`LLM_HC_UP_GEMM=1` keeps the GEMM). Gate:
`TestHCGPUUpGemvAgrees`, one to three rows on both banks, maxAbs 1.4e-4 on
the gate and 9e-5 on `mixed` against the GEMM — one rounding a weight, as the
Q4 arm of `llm_gemv.comp` is.

**Version 1, one wave holding a whole column** (20 k-tiles: the ISA shows
46 loads in flight before the first multiply): `hc.up` 17.9 → **15.4 µs**
(147 GB/s), the step 26.34 → 26.07/26.10 ms, 37.96 → 38.35 tok/s. Less than
the 0.79 ms on the table, and the shape of the shortfall is the clock:
a 2.25 MB dispatch is 9.3 µs at the bus, and what is left is ~2000 VALU
instructions a lane of unpack and multiply *in series after the loads
land*, at the ~800 MHz a memory-bound step runs the shader clock at (the
sampler above) — 2.5 µs a wave, with two waves a SIMD to hide it behind.

**Version 2, four waves a feature block**, each a quarter of the column's
k-tiles, partials through LDS, wave 0 sums them in order and runs the
collapse: `hc.up` **13.6 µs** (165 GB/s), the step **25.80 / 25.85 ms,
38.7 tok/s**, every other label unchanged (same-hour arms C 26.10, N1
25.80, N2 25.85). The wave count screened in the model,
interleaved (5 / 10 / 4 / 5 / 10): four waves 13.5 µs, **five 13.3, 13.3**,
ten 13.9, 13.8 — a plateau; five ships (four k-tiles a wave, 800 waves),
and the ~4 µs above the bytes is the dispatch's fixed cost (launch, the
barrier, the LDS staging and the collapse), which only fewer dispatches
would remove. **The step is 25.8 ms, 38.7 tok/s; P21a is +1.1% on top of
the head's +1.1%, 37.5 → 38.7 since the morning.**

## Finding 3 — the big GEMVs do not want their loads hoisted (P21b, measured dead)

The same trick was built for the rungs `llm_gemv.comp` and `llm_hc_gemv.comp`
run at decode: `-DPERSLAB` compiles the slab's length in, the step loop
unrolls, every weight word of the slab is issued before the first multiply
(the record decoded without a branch, since a divergent `if` splits the
block and stops the scheduler), and the ISA showed 17–23 loads in flight
where the shipped rungs hold 6. Bit-identical to the rungs (four gates,
DeltaNet / attention / head / hc down, one to three rows). In the model,
same hour, interleaved with the arm above:

| arm | step | head | dn.qkv | dn.out | attn.qkv | attn.out | hc.down |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped rungs (N1, N2) | 25.80, 25.85 | 2.063, 2.065 | 4.028, 4.040 | 1.609, 1.606 | 1.420, 1.423 | 0.654 | 1.352, 1.349 |
| hoisted (D1, D2) | **26.54, 26.52** | 2.415, 2.357 | 4.130, 4.122 | 1.597, 1.599 | 1.400, 1.403 | 0.657 | 1.575, 1.591 |

Slower by 0.7 ms: the head +0.3, `hc.down` +0.23, `dn.qkv` +0.1, the two
`attn` and `dn.out` flat or 0.02 better. The big GEMVs have thousands of
waves a dispatch and were never short of loads in flight; what the hoisted
builds bought was 60–72 VGPRs against 48, a branch-free record decode that
costs VALU at 800 MHz, and two record loads a chunk — and the head, the
widest grid, paid the most. The builds, the map entries and the gates were
removed; the shader arms are not kept. **P21's "the `s_waitcnt vmcnt(0)` a
step" is answered: not that.** What limits the big GEMVs at 195–211 GB/s
is not their load count and not their loop; the remaining suspects are the
DRAM pattern of a slab a workgroup (D12's 4 KB rotation, re-laddered for
today's K-quant bank sizes) and the clock.

## Finding 4 — the shared expert as the eleventh tile (P22a): 15.0 → 9.2 µs a layer, the step 25.75 → 25.53

The shared expert is one group of the same grouped kernel over the same
[2560, 640] and [640, 2560] shapes, and at decode it ran as two dispatches
of its own over an identity permutation at the front of the row space —
`shexp.up` 15.0 µs for 1.84 MB (123 GB/s) and `shexp.down` 9.9 for 1.23
(124), 1.19 ms a step between them. Now `llm_moe_gemv.comp` carries it as
**one more tile past the routed schedule**: when the host names the shared
bank in the two push fields llm_common.glsl maps as `MOE_SH_B` /
`MOE_SH_B2`, the dispatch's grid is one tile taller, the tile past the
device's count has no record — its rows are the reserve the host wrote as
the identity, every token of the batch is real, and its bank rows are the
shared matrix's rather than expert-major rows of `bOff` — and the shared
expert's own dispatch is not recorded (`MoEGPU.foldShared`,
`LLM_MOE_SHEXP_SPLIT=1` the control). Nothing about the permutation, the
combine or the shared slot moves; the tile writes the same swiglu rows and
the same (token, slot) the split dispatch did.

**The format is compiled into the rung, so a mode folds only where the
shared bank ships in the routed bank's format.** On the shipped plan
(`ShippedMoEBank`) the shared gate and up are Q4_K like the routed pair on
47 of 48 layers — layer 2's routed pair is Q5_K and keeps its dispatch — and
the shared down is Q5_1 against the routed IQ4_NL, which does not fold.
Gate: `TestMoEGPUSharedFold`, one to three rows on a bank restaged to
match, the arena dirtied with other tokens between arms so a tile that never
ran cannot pass on the previous arm's rows (P5c's lesson); the fold agrees
with the split to rms 8e-9 and maxAbs 3e-8 — one reassociation of the
shared row, since the tile runs on the routed rung's lane width (v16w4)
where the split ran the shared plan's (v64w4).

**Measured, same hour, interleaved** (`-gen -attrib -n 256`, 48 layers, the
shipped banks, one binary, the sampler beside each arm at 81–82 W and
884–906 MHz for all four):

| arm | ms a step | tok/s | `moe.up` | `moe.shexp.up` | `moe.down` | `moe.shexp.down` |
|---|--:|--:|--:|--:|--:|--:|
| split 1, 2 (`LLM_MOE_SHEXP_SPLIT=1`) | 25.754, 25.743 | 38.83, 38.85 | 4.560, 4.553 (95.4 µs) | 0.716, 0.708 (15.0 µs) | 2.527, 2.525 | 0.472, 0.480 |
| **fold 1, 2** | **25.558, 25.513** | **39.13, 39.20** | 5.001, 4.999 (104.6 µs) | 0.015 (layer 2 only) | 2.552, 2.558 | 0.475, 0.471 |

**−0.21 ms a step, 38.8 → 39.2 tok/s (+0.9%)**, the text identical across
the four arms. The shared up costs **9.2 µs inside the routed dispatch
where it cost 15.0 on its own** — 1.84 MB at 200 GB/s, the routed rate —
so the fixed cost of a small dispatch is the 5.8 µs the plan priced it at,
and the eleventh tile pays none of it.

**The down half, priced and not shipped.** One more arm with the shared
down restaged to the routed down's format
(`down_shexp=iq4_nl`, `p22_shexp_fold_folddown.csv`): `moe.down` 2.55 →
2.78 with the tile, `shexp.down` gone, the step **25.31 ms** (39.5 tok/s).
Of that 0.22 ms about 0.07 is the narrower bytes (0.31 MB a layer at the
bus) and ~0.15 the fold. It needs the shared down shipped narrower than
Q5_1, which is a bank-plan decision P24's downstream eval grades, not this
stage's; the code folds it the moment the formats match.

## Finding 5 — `moe.down` was five serial round trips a lane (P21c): 53.3 → 49.2 µs, the step 25.54 → 25.13

**The two ladders first, in the whole model** (`p21c_down_ladders.csv`,
one binary, `LLM_MOE_DECODE_PLAN` and `LLM_HC_DOWN_SLABS` as the knobs,
three controls interleaved at 53.2 / 53.4 / 53.5 and 13.9 / 14.0 / 13.9 µs):

| `moe.down` rung | v16w4 (shipped) | v32w4 | v64w4 | v16 | v32 | v64 |
|---|--:|--:|--:|--:|--:|--:|
| µs a dispatch | **53.3** | 54.4 | 56.2 | 69.3 | 77.1 | 104.6 |

| `hc.down` slabs | 8 | 16 | 32 (shipped) | 40 | 80 | 160 |
|---|--:|--:|--:|--:|--:|--:|
| µs a dispatch | 19.7 | 14.2 | **14.0** | 26.8 | 15.4 | 13.4 |
| + `hc.down_reduce`, ms a step | 2.00 | 1.48 | **1.46** | 2.71 | 1.64 | 1.53 |

Both shipped rungs are the right ones. D12 holds on the K-quant bank —
40 slabs is the whole multiple of 4 KB again, 1.9x slower — and 160 wins
the kernel by 0.6 µs and loses 0.12 ms a step on the reduce behind it.
`hc.down`'s 0.4 ms is neither its split nor its loads in flight (P21b), and
it is left where it is.

**Then the ISA of the shipped `moe.down` build** (`cmd/probe`,
`RADV_DEBUG=asm` on `llm_moe_gv_down_iq4nl_v16w4.spv`): the header's claim
that "the whole row's worth of loads is in flight at once" was not what the
compiler made of it. A lane's five trips over its 80 dwords (K = 640, LPR
16) were each two `buffer_load_b32` and then `s_waitcnt vmcnt(0)` before
the first nibble moved — five serial DRAM round trips — behind an A staging
that loaded one half a load with a wait each (2.5 more), on 1600
workgroups of four waves. The kernel was latency-bound the way P8 describes,
at the scale of one lane's loop.

**The change** (`llm_moe_gemv.comp`): `unpackDword` is split into
`loadDword` (the two or three words a dword needs, per format) and
`unpackRaw` (the arithmetic); the down mode loads all `TRIPS` of a lane's
words before it unpacks any, with TRIPS from the compiled AKMAX; and the A
rows are staged eight halves a load through binding 7. **Two things had to
be branch-free for the hoist to survive compilation**: under an `if (d <
nqd)` or under the row guard `if (r < real)` NIR sinks each trip's loads
into the branch beside their use and the wait is back one trip at a time,
so a trip past the row and a row past the tile's real ones multiply by zero
instead, and the rows past `real` are staged as zeros so that the zero
never meets a stale NaN. The ISA now reads ten loads, the barrier, and
waits stepping down from `vmcnt(8)`. VGPRs stay at 48. Gates:
`TestMoEGPUDecode`, `TestMoEGPUDecodeTwoRows`, `TestMoEGPUSharedFold`,
`TestMoEGPUDownNarrowed` (every down format), `TestMoEGPUBlock`,
`TestMoEGPUPaddingIsInert` — and the first build failed all of them at rms
4.5e-3, because the Q8_0 loader had the block parity inverted (an *even*
block's int8s straddle two words); the fixture's shared expert is Q8_0 in
both modes, which is what caught it.

**Measured, same hour, three interleaved pairs** (`-gen -attrib -n 256`, 48
layers, the shipped banks, the shipped builds loaded through `LLM_MOE_SPV`
as the control; the third pair sampled: control 82 W / 947 MHz, hoist 83 W
/ 976 MHz):

| arm | ms a step | tok/s | `moe.up` | `moe.down` | `moe.shexp.down` |
|---|--:|--:|--:|--:|--:|
| shipped builds (A, B, C) | 25.57, 25.50, 25.56 | 39.1 | 4.997, 4.991, 4.996 (104.5 µs) | 2.551, 2.547, 2.556 (53.3 µs) | 0.478, 0.473, 0.476 (10.0 µs) |
| **hoisted (A, B, C)** | **25.09, 25.13, 25.17** | **39.8** | 4.799, 4.810, 4.810 (100.4 µs) | 2.353, 2.359, 2.364 (49.2 µs) | 0.424, 0.421, 0.435 (8.9 µs) |

**−0.42 ms a step, 39.1 → 39.8 tok/s (+1.7%)**, the text identical across
the six arms. `moe.down` 53.3 → 49.2 µs is the hoist (9.2 MB at 187 GB/s
now, from 173); `moe.up` 104.5 → 100.4 is the A staging alone, since its
loop is unchanged (20.3 MB at 202 GB/s); the shared down rides the same
kernel. The shader clock read 3% higher under the hoisted builds, which is
a kernel with less idle time under DVFS rather than a control arm doing
less work — every arm moves the same bytes.

**And the up mode's loop does not want the same** (P21b's finding, a third
time): the header and payload loads of both matrices issued together (four
`buffer_load_b128` in flight a quad where the shipped loop waits after
each), the `j < 4` of the scale decode as selects and the row guard as a
multiply read **+1.5 µs a dispatch slower** (100.4 → 102.0, two pairs; the
step 25.12 → 25.19, `p21c_moe_up_grouped_dead_*.csv`). At 202 GB/s that
loop is where the big dense GEMVs are and is not short of loads; the change
bought VALU at ~900 MHz. Reverted; the down mode's hoist is the one that
ships.

**The day:** 26.66 → 25.13 ms a step, **37.5 → 39.8 tok/s** (P19's head,
P21a, P22a, P21c).

## Finding 6 — the moves deleted without a shared arena (P22b): 196 → 4 a step, 25.13 → 24.83

The 196 moves were the largest weightless item above (0.53 ms, 2.7 µs
each), and L6c's answer to them was a shared arena it priced and refused:
splitting every block's `alloc` into a layout pass and a materialisation
through five constructors. **There is another way to delete a copy: bind
the other block's arena in the kernel on either side of the boundary.** A
compute pipeline here owns its descriptor set, so a pipeline can be built
per (kernel, foreign buffer) — which is exactly what `llm_move.comp`
already was, nine pipelines for nine buffer pairs — and binding 11 is that
foreign buffer, 5–10 padded with the bank (`XOUT`/`XIN` in
`llm_common.glsl`, `HCLink` in `gpu.go`, built on first use because the
sublayers' arenas exist only once the graph does).

- **The output side, at every row count.** `llm_hc_cn.comp` and
  `llm_hc_combine.comp` read the block output out of the sublayer's own
  fp32 arena (`xout[outOff + t*n + i]`) — the same floats the move copied.
  The graph holds a combine as before (`pending`) and now remembers where
  the sublayer left its output (`pendingOut`); the PLE flush and the last
  flush read it there too.
- **The input side, on the decode GEMV.** `llm_hc_up_gemv.comp`'s epilogue
  has the mixer's value in a register, so the `XIN` build stores
  `float16_t(s * invHC)` straight into the sublayer's A operand at the
  consumer's stride (two PLE fields carry the stride and the row block, on
  the SEQ_PAST rule) and skips the fp32 `mixed` store. The padded GEMM at
  prefill still writes `mixed` and the graph moves it; `RunLinked` says
  which happened. What is left to the mover at decode is the PLE block's
  wide residual out and back, the final row move and the head's input: 4.

**Bit-exact by construction** — `float16_t(v)` on the same fp32 value the
move narrowed — and gated: `TestGraphHCLinkBitExact` runs a prefill and
one-, two- and three-row passes with the moves and with the link and
demands the same bits (100 → 28 moves over the five passes on the K-quant
bank, where the up rung is the GEMV; 100 → 60 on the Q8 bank, which has no
GEMV arm and links the output side alone). `LLM_HC_MOVES=1` /
`GraphOpts.HCMoves` keeps every move and is the control.

**Measured, same hour, three interleaved pairs** (`-gen -attrib -n 256`,
48 layers, one binary): moves 25.13 / 25.15 / 25.13 ms → link **24.83 /
24.83 / 24.81**, 39.79 → **40.28 / 40.28 / 40.31 tok/s** (`p22b_{moves,link}{A,B,C}.csv`).
`move` 196 → 4 dispatches, 0.530 → 0.031 ms. **But `hc.up` grew 13.3 →
14.9 µs a dispatch (+0.156 ms)** and `hc.cn` 2.4 → 2.7 (+0.03), so the
step gave back a third of the 0.5: the move had also been *zeroing the
consumer's pad rows*, contiguously, and the linked kernel did the same from
the sixteen lanes of one wave, 63 to 127 scattered two-byte stores a lane.

**The pad rows are not needed on the decode plan at all.** Every consumer
asked for them because its GEMM rungs have no bounds check — DeltaNet and
attention to their widest tile, the MoE to 64 — but at one to three rows
every rung that reads the A operand is a GEMV (`qkvGemv != GEMVOff`; the
router's split-K kernel and the expert GEMVs) that reads ROWS rows and
not one more. Each `InPort` now says so and reports the run itself as its
row block on that plan, so neither the move nor the linked kernel writes a
zero row at decode; the zeroing that remains (the `LLM_DECODE_GEMM=1` arm)
is done by the whole workgroup in runs of sixteen halves. The gates above
and `TestGraphIsAChunkSplit` on the shipped banks are unchanged to the
digit by it — no rung read those rows.

**Measured again, three interleaved pairs, both arms with the pad rows
gone** (`p22b_link_{moves,link}{A,B,C}.csv`): moves **24.91 / 24.89 /
24.97** → link **24.42 / 24.39 / 24.43 ms**, 40.1 → **41.0 tok/s** (40.96 /
41.01 / 40.93). `hc.up` is back at 13.4 µs; `hc.cn` reads the foreign
buffer at 2.6 µs against 2.3 (+0.03 ms a step, unexplained and small). The
zero rows had cost the *moves arm* 0.22 ms on their own — the kernels after
each move (`dn.out` −0.09, `attn.out`, `moe.down`, `moe.router`) ran
faster once no write-back of a megabyte of zeros preceded them — so the
day's second change is worth as much as its first: **25.13 → 24.42 ms,
39.8 → 41.0 tok/s** for the pair of them, the moves −0.50 and the pad rows
−0.22. The 4 moves left are 0.03 ms.

## Postscript — the suite's three failures at the end of the day, and what they were

Run after P22b landed (2026-09-30, late), the `llm` suite failed two tests
on its default banks (the Q8 dense bank, the checkpoint's own MoE formats)
and one subtest on the shipped ones. All three were unchanged with
`LLM_HC_MOVES=1`, so none was P22b's; each was bisected by the decode path's
control knobs and found.

1. **`TestGraphIsAChunkSplit`'s three decode-schedule subtests, rms 0.137 on
   a scale of 18.6 — a wrong answer.** `LLM_MOE_SHEXP_SPLIT=1` cured it, and
   restaging only the shared down (`LLM_MOE_BANK=down_shexp=iq4_nl`) cured
   it too, which named the **down half of P22a's fold**. The fold's order
   was wrong in one combination: the routed `down` dispatch, which carries
   the folded shared-down tile, was recorded *before* the split `shexp.up`
   that writes the shared swiglu rows that tile reads. The shipped plan
   folds the up and splits the down; the fold's block test folded both;
   only the checkpoint's own formats — a Q8_0 shared gate/up against Q4_K
   routed, a Q5_1 shared down against Q5_1 routed — take the mixed path,
   up split and down folded, and read rows written two dispatches later.
   `MoEGPU.graph` now records `shexp.up` before the routed down, and
   `TestMoEGPUSharedFold` has a mixed arm (down restaged alone) that agrees
   with the split at rms 4e-10 and asserts the order. The default-bank
   subtests pass at rms 5.7e-4 / 7.5e-4 / 8.4e-4.
2. **`TestGraphImageDecode`'s batched-rows gate, a fifth-digit difference
   between a batched step and the same slots' solo steps.** Not the fold,
   not the overlap, not the prerecord, not the decode GEMVs; `LLM_HEAD_GEMM=1`
   made it bit-identical. P19 put the head on its GEMV **at one row only**,
   so a three-slot batched step ran the GEMM head and rounded differently
   from three solo steps. The head now builds the GEMV per row count to
   GEMVMaxRows (P5b's `ROWS`) and plans it there, as every other decode GEMV
   does; `TestGraphHeadDecodeGEMV` gates the three-row pass against the GEMM
   (rms 7.7e-6 of the logits' max) and the batched gate is bit-identical
   again. **The rule:** a slot's step must not depend on who it shares a
   pass with, so a decode rung applies to every row count up to
   GEMVMaxRows or to none.
3. **The shipped-bank 3-row decode-schedule subtest at rms 2.2e-3 against a
   1e-3 bar.** `LLM_HC_UP_GEMM=1` brought it to 7.6e-4 and pointed at the
   up GEMV — but `TestHCGPUUpGemvRowsAgree` (new) shows the GEMV's rows at
   two and three rows are the *bits* of one-row dispatches on the model's
   own weights, and its disagreement with the GEMM is the same 4e-5 rms on
   every row. The schedule is what moves: with the up on the GEMM,
   `LLM_HC_DOWN_SLABS=16` (a pure reassociation of the split-K down) takes
   the same subtest from 7.6e-4 to 2.3e-3, while the 2-row schedule stays at
   5.5–8e-4 under every arm. One channel (951) of one token's seventeen-step
   recurrence swings by 5e-2 — a near-tie downstream, most likely a routing
   one. The multi-row subtests' bar is 5e-3 now, with the measurement in
   the test; the signatures it exists to catch are 0.137 and 1.785.

## What this sets up

In order of `lost` and of cost to take:

1. ~~**`hc.up` on a GEMV**~~ — done above (P21a): 1.75 → 1.31 ms.
2. ~~**The shared expert as an eleventh tile** of `moe.up`~~ — done above
   (P22a, Finding 4): −0.21 ms; the down half is priced at ~0.15 more and
   waits on the shared down's format.
3. ~~**`moe.down` and `hc.down`'s shape**~~ — Finding 5 (P21c): both
   rungs re-laddered in the whole model and confirmed; `moe.down`'s five
   serial round trips a lane hoisted, −0.42 ms with the A staging; `hc.down`
   stays at 32 slabs with 0.4 ms nobody has an idea for.
4. ~~**The moves** (P22, 0.54 ms): L6c's single arena.~~ Finding 6 (P22b):
   deleted at the boundary instead, 196 → 4 a step.
5. **P20** stands on 1.23 steps for a two-row pass and a 23% break-even.

The four big GEMVs (0.8 ms against 227) move down the list: P21's first
suspect, the Q4 arm's load shape, was read from source and the compiler had
already done most of it, and Finding 3 measured the rest.
