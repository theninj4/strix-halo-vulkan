<!-- LLM.md L7d. The three decode kernels: a split-K GEMV for the
     hyper-connection down projection, narrow column blocks for the MoE, and
     one command buffer a pass with the attribution moved inside it. Cited
     from shaders/llm_hc_gemv.comp, shaders/llm_moe_gemm.comp, llm/record.go,
     llm/gpu.go, llm/gpu_moe.go, llm/graph.go, vk/shim.c and
     results/l7d_decode.csv. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L7c](l7c-decode.md) · [L5b](l5b-moe-gpu.md) · [§5.1b](5.1b-mall-cliff-and-stride.md) · phase 1

# L7d — the decode kernels: 7.46 → 11.89 tok/s, and the same text

`results/l7d_decode.csv` · `results/l7d_graph.csv` · `results/l7d_hc_gemv.csv` · `results/l7d_moe_decode.csv`

**Result: decode is 11.89 tok/s against L7c's 7.46 — 1.59x — with the
completion unchanged. Prefill comes along for free at 990.6 tok/s against
950.3, which is 2.53x llama.cpp's 391.42.** Two runs agree to **0.08%**
(11.89 / 11.90) and the ladder to 0.5%.

L7c-6 left an attribution rather than a direction: of 134.1 ms a token, the
hyper-connection block and the MoE were 55% and were **6.9x and 5.6x** off
their own bytes where every other block was 1.2-1.7x, and ~490 submits were
~15 ms more. This is those three, and the first two turned out to be one
finding said twice.

| | L7c | L7d | |
|---|---:|---:|---|
| decode | 7.46 tok/s | **11.89** | 1.59x |
| a token | 134.1 ms | **84.1** | |
| prefill, ubatch 2048 | 950.3 tok/s | **990.6** | 2.53x llama.cpp |
| against llama.cpp's 25.15 | 30% | **47%** | |
| against this bank's own 25.0 ceiling | 30% | **48%** | |

The gate is L7c's and it is unchanged: on `The capital of France is` at
temperature zero, 128 tokens, our completion is llama.cpp's — the same list of
capitals, the same `<think>` block, the same `Lisbon.` — diverging at the same
single token (`.` against ` for`, 0.161 logits apart) and re-converging within
a sentence. Both sides were re-run for this write-up.

## L7d-1: the down projection is a grid, and a split-K GEMV is 7.95x

At one token `llm_gemm.comp`'s MODE 0 runs the hyper-connection down
projection in **seven workgroups**. Its fused N is 336 — 320 low-rank columns
plus the tile carrying `inject` — and BN is 48, so the column blocking that
fills a 40-CU device at 512 tokens cannot fill it at one, whatever BM is:
measured, `down_m1`, `down_m2` and `down_m4` are 238.6, 256.4 and 303.7 µs,
all at 23-29 GB/s of a 242 GB/s bus, and the widest is the worst because its
A fragment is 63 rows of padding to one of data.

At M=1 the parallelism cannot come from N — there are only 336 of them and
they are the output — so it has to come from **K**, which is 10240.
`shaders/llm_hc_gemv.comp` is the ordinary split-K GEMV over the weight the
GEMM already staged, in two dispatches: `KSLABS × 21` workgroups writing
partial sums, then one that reduces them and carries MODE 0's own epilogue
(silu(x/hc) narrowed into the fp16 arena, `inject`'s tile written fp32).

**The §2.8 fragment tiling turns out to be exactly the layout this wants.**
Tile (nt, kt) is 256 contiguous halves holding element (k, n) at
(n%16)*16 + k%16, tiles ordered kt-fastest, so within an n-tile the tiles of
consecutive kt are contiguous: one workgroup's whole slab is **one contiguous
run**, and a wave of 64 lanes taking sixteen halves each covers 1024
consecutive halves with no gaps. The (n%16)*16 order then hands each lane a
single output column and sixteen consecutive k of it — one accumulator per
lane, no LDS, and the four lanes that share a column are `l`, `l^16`, `l^32`,
which is two shuffles.

| rung | 238.6 µs (GEMM `down_m1`) | | |
|---|---:|---:|---:|
| `down_gemv8` | 38.4 µs | 180 GB/s | |
| `down_gemv16` | 44.7 µs | 154 GB/s | |
| `down_gemv32` | 31.5 µs | 219 GB/s | |
| `down_gemv40` | 53.4 µs | 129 GB/s | |
| `down_gemv80` | 50.1 µs | 138 GB/s | |
| **`down_gemv160`** | **30.0 µs** | **230 GB/s** | **7.95x** |

The mixer goes from 293.9 µs to **79.5**, and the block from 28.5 ms a graph
to 7.7.

## L7d-2: which split is §5.1b's 4 KB period, not the workgroup count

That ladder is not monotonic and it is not noise — the two fastest rungs are
the one with 672 workgroups and the one with 3360, and the three in between
are slower than both. What separates them is the distance between two
workgroups' slabs, `(gemmK/16/KSLABS) * 512` bytes:

| KSLABS | 8 | 16 | 32 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|
| slab, bytes | 40960 | 20480 | 10240 | 8192 | 4096 | 2048 |
| × 4 KB | 10 | 5 | 2.5 | 2 | 1 | 0.5 |
| GB/s | 181 | 154 | **219** | 129 | 138 | **230** |

**Every rung whose slab is a whole multiple of 4 KB is slow, and both that are
not are fast.** 4 KB is the rotation §5.1b pinned for DRAM channel aliasing on
this part — a 256-bit LPDDR5X bus of 16-bit sub-channels interleaved 256 B
across sixteen — measured there across nine strides inside one rotation with
the byte set held constant, and measured here across six with the byte set
held constant again. §2.3 found the same period in a GEMM's leading dimension
and §5.1b's probe separated it from the gather penalty; this is the third
place it has decided a kernel's shape, and the first where the stride in
question is not a matrix's at all but **the distance between two workgroups'
starting addresses**.

So the rung to pick is not the widest split. It is the one whose slab stride
misses the rotation, and at 160 that is 230 GB/s — **95% of the bus**, against
the GEMM's 29. Swapping the grid axes so that consecutive workgroups walk
consecutive slabs rather than consecutive n-tiles was worth a further 4% and
is kept; it does not change which rungs are fast.

`gemv8` and `gemv80` are kept built for the same reason the GEMM rungs are:
the table above is the evidence for the rule, and a rule with two data points
either side is not one.

## L7d-3: the MoE is the same finding, and the narrow-N rungs are 1.28x

L7c-6 read the MoE's 70 GB/s as the 32-row padding. It is not, or not mainly.
At one token ten experts have one row each, so **every rung makes exactly ten
tiles**, and a narrower row block removes arithmetic that was never the
constraint. What the profiler shows at T=1 is a grid:

| dispatch | workgroups | GB/s of bank |
|---|---:|---:|
| `up` (BN 64, 10 tiles × 640/64) | 100 | 71 |
| `shexp.up` (BN 64, 1 tile × 640/64) | **10** | **23** |
| `down` (BN 64, 10 tiles × 2560/64) | 400 | 136 |

The same matrix shape, the same format, the same kernel — and the one with
400 workgroups reads six times faster than the one with ten. So the MoE's
answer is the down projection's answer on the other axis: **cut BN**, which
multiplies the grid and divides the slab each workgroup unpacks, for the same
total unpack. `n1m1` is BN 16 and BM 16:

| plan | up | shexp.up | block |
|---|---:|---:|---:|
| `m2/m2` (L5b's schedule) | 259.0 µs, 71 GB/s | 153.7 µs, 23 GB/s | 612.3 µs |
| `m1/m1` | 212.9, 87 | 123.0, 28 | 549.6 |
| **`n1m1/m1`** | **197.3, 93** | **76.0, 46** | **479.9** |

**1.28x on the block**, and the up mode is still at 93 GB/s where the down
mode beside it is at 136 for the same grid. Narrowing BN 4x moved it 71 → 93,
not 71 → 136, so the grid was part of it and not all of it — what is left is
that MODE 0 gathers its A rows into LDS per K-step and MODE 1 does not, and at
decode every workgroup gathers the **same** 2560-long row. That is the open
question this stage narrows rather than closes (below).

The narrow rungs are bit-exact against the wide ones at 4096 tokens, which is
what `TestMoEGPULadderAgrees` demands of every rung: they change how the
output columns are blocked and nothing about how a sum is associated.

## L7d-4: one command buffer a pass, and the attribution moved inside it

L7c-5 took the graph from a submit a dispatch to a submit a block — ~1130 to
~490 — and priced the rest at 31-43 µs each, ~15-21 ms of a token. Nothing in
the shim ever prevented going further: it binds each dispatch's own pipeline,
descriptor set and push constants, so a sequence may mix blocks as freely as
it mixes kernels of one block. **What stopped it was the measurement.** Every
number this vertical is tuned by came from the host wall clock around a
block's `Run`, and a block that no longer submits has no wall clock of its
own.

So both moved. `llm/record.go` is a recorder each block's `Run` appends to
instead of submitting; `Graph.Extend` installs it, records the layers, the
head mixer and the projection, and submits **once** — 1405 dispatches at
decode, 1309 at prefill, in one command buffer per 1024. And
`vk.DispatchMultiMarked` asks the shim for a timestamp after every dispatch
rather than only at the two ends, so the per-block figures come back as GPU
time *inside* that command buffer. That is the same quantity
`GGML_VK_PERF_LOGGER` reports for llama.cpp's graph — the baseline half of
every comparison in LLM.md — and it is strictly the better number: it cannot
be inflated by a fence wait, a Go allocation or a page fault.

Two things the merge needed. The moves' push block is 32 bytes against the
vertical's 256 and one command buffer pushes one size, so `movePush.bytes()`
pads to the shared size and its pipeline layout declares the same range. And
nothing may read a device arena between recording and submitting, which made
`HiddenExtend` and `Extend` into a record/flush pair with every `Mixed()`,
`Logits()` and `Res()` after the flush.

| | decode, ms a token |
|---|---:|
| L7c: a command buffer a block | 99.0 |
| L7d: a command buffer a pass | **84.1** |

**1.18x**, and the hand-over that is left is **0.96 ms of 84.1** against the
~15 that ~490 submits cost.

## L7d-5: the exactness gate is now conditional on the schedule, and says so

Every rung of every ladder in this vertical had been **bit-exact** against its
siblings: they tile the same arithmetic differently and never re-associate a
sum, which is why `TestHCGPULadderAgrees` and `TestMoEGPULadderAgrees` assert
`maxAbs == 0`. The split-K GEMV is the first that is not. Dividing a
10240-long dot product 160 ways and adding the pieces back is a different
association of the same fp32 products — and at one token it is the rung the
schedule picks, which is exactly the chunk length L7b's gate is about.

So `TestGraphIsAChunkSplit` now pins the schedule for its three equalities and
adds a fourth arm that runs the decode schedule:

| | result_norm against the 512-token run |
|---|---|
| 128 at a time, pinned | identical to the last place |
| 9 at a time, pinned | identical to the last place |
| 495 then 17 single tokens, pinned | identical to the last place |
| 495 then 17 single, **decode schedule** | rms 4.43e-04, 0.0024% of scale |
| control, every token a fresh sequence | rms 1.79e+00 |

The claim L7b made is untouched — the histories carry, to the last place —
and what the new kernel costs is measured beside it: **4000x smaller than a
dropped history**, and the same order as the 0.07-0.11% of scale
`TestGraphPrefix` reports against llama.cpp itself.

## L7d-6: where the 84.1 ms goes now

| block | ms/token | % | its bytes | at 242 GB/s | off by |
|---|---:|---:|---:|---:|---:|
| MoE | 28.50 | 33.9% | ~1.60 GB | 6.6 | **4.3x** |
| gated DeltaNet | 27.00 | 32.1% | 4.18 GB | 17.3 | 1.56x |
| hyper-connection | 8.93 | 10.6% | 1.31 GB | 5.4 | 1.65x |
| full attention | 7.85 | 9.3% | 1.24 GB | 5.1 | 1.53x |
| lm head | 6.32 | 7.5% | 1.27 GB | 5.2 | 1.20x |
| moves | 0.55 | 0.7% | — | — | 197 dispatches |
| PLE n-gram | 0.40 | 0.5% | — | — | |
| **on the GPU** | **79.57** | **94.6%** | **9.67 GB** | **40.0** | **1.99x** |
| n-gram gather (host) | 2.80 | 3.3% | 1.41 KB | — | latency |
| glue (host) | 0.79 | 0.9% | — | — | recording |
| hand-over | 0.96 | 1.1% | — | — | 2 submits |

The whole step now reads **122 GB/s, 50% of the bus**, against L7c's 72
GB/s — and the shape of the problem has changed. The hyper-connection block
went from 6.9x off to 1.65x and is no longer the place to look; the two blocks
left are the MoE, still 4.3x off and the only one that is, and the **DeltaNet,
which is now the largest single block by wall clock** at 1.56x off its own
4.18 GB. Underneath both, L8: 8.07 of those 9.67 GB are a dense half staged as
halves, and not expanding it is 1.66x on the ceiling before a single weight is
re-quantised.

## What this leaves

L7 is complete and decode is 47% of the reference. The three things between
here and the 25.0 tok/s this bank's own bytes allow are, in order of what they
are worth:

- **L8's bank.** 9.67 GB a token against the checkpoint's 6.334, because our
  dense half is fp16. Worth 1.66x for staging the shipped Q8_0 and 3.5x at
  D3's ~4.25 bits, and it would also leave page cache for the 28.80 GB n-gram
  table.
- **The MoE's up mode.** 93 GB/s of bank where the down mode beside it does
  136 on the same grid. The remaining difference is the A gather, and at
  decode it is a gather of one row that every workgroup repeats — which is an
  argument for a GEMV per expert (§1.8-§1.12 have 25 grouped builds of exactly
  that, against the checkpoint's own Q4_K rather than a repacked bank) and no
  longer an argument about padding.
- **The DeltaNet**, which nothing has yet looked at with one token in mind:
  36 layers, 4.18 GB, 1.56x off, and the largest block in the step.
