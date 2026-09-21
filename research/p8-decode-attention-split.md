# P8 — the decode attention was one wave's walk, not the cache's size

> Measured 2026-09-21 on this machine, interleaved A/B of the same binary
> under `LLM_ATTN_SPLITS=1` (the unsplit control) and the default, two passes
> of each arm in the same hour. It follows P7
> (`research/p7-context-depth.md`) and is the second half of the same
> question.

## What P7 left, and why its attribution was wrong

P7 closed the depth regression the 2026-09-19 sweep recorded — decode at 64k
from 4.05 to 16.27 tok/s — and named what was left as **the gather**: compact
the ≤2051 selected cells' K and V into a contiguous scratch and run a
2051-cell attention, "the only route to a rate that does not care how long the
conversation is". That was the remaining 1 948 ms of a 64k decode step.

The premise under it was that the kernel's cost is the *cells it reads*. It is
not, and one probe says so.

**Cutting the attention grid from 24 workgroups to 2 — a twelfth of the work
and a twelfth of the traffic — measured 1.00x at every depth.**

| depth | 0 | 16 384 | 65 536 |
|---|---|---|---|
| `attn.attn`, 24 workgroups (ms/token, one layer) | 0.142 | 1.930 | 1.975 |
| the same with the grid cut to `NHeadKV` | 0.142 | 1.927 | 1.976 |

The grid is (query blocks, heads). At decode that is (1, 24): **24 workgroups,
each a single wave**, and this device has 40 compute units, so all of them run
at once and the dispatch's wall clock is one wave's serial walk over the key
blocks. Twelve of those waves share each of the two KV planes and re-read it
in full, which looks like the obvious waste and is not the cost — removing it
removes nothing, because the twelve were never queued behind each other.
Neither is the QSA selection's scatter, and neither is bandwidth: with one
wave per CU there is nothing to hide a `coopMatLoad`'s latency behind.

So the lever is not what the walk reads. It is **how long the walk is**, and
the gather is only one way to shorten it — by making the cells fewer. Cutting
the axis across more workgroups shortens it without touching the cells at all,
works at every depth rather than only past the width, and needs no second
layout for K and V.

## The finding, in one line

A decode step's attention is latency-bound on a single wave, and the model has
**24 waves of work where the device wants hundreds**.

## What changed

**`llm_attn_wmma.comp` gains a `-DSPLITK` build.** The key axis is cut
`splits` ways, the grid's x carries (query block, slice), and each slice runs
its own online softmax over its own blocks and writes the result
**unnormalised** — beside the two numbers that let it be resumed, its row max
and its row sum. `llm_attn_combine.comp` folds the slices together, and does
the softmax divide and the output gate that the unsplit epilogue does. That is
flash-decoding's split.

**The slices are strided, not contiguous, and that is a correctness
requirement.** L7a's gate is that a prompt run in pieces equals the prompt run
whole, bit for bit. The natural cut — `[0, last)` into `splits` runs — fails
it, because `last` is `min(n, past + q0 + BM)` and both `n` and `q0` belong to
the *batch*: a seven-token chunk and a one-token chunk put the boundaries in
different places and fold the partial softmaxes in a different order. A stride
is immune — block b is slice `b % splits` however the run was cut — and the
blocks past a row's causal extent that the stride then walks cost nothing and
change nothing, for exactly the reason the unsplit loop can walk them: blkMax
stays -inf, the correction is exactly 1.0, and P is zeroed. The stride also
balances the slices where a contiguous cut would not, because P7's selection
is clustered.

**The slice count is a constant, and every other candidate is a bug.** It must
not move with `SEQ_PAST`, because a decode step's command buffer is recorded
once and replayed byte for byte (P1c). And it must not be read off `nKV` —
the tempting one, since a deep arena is where the split pays — because
`TestAttnGPUCacheSizeDoesNotChangeTheAnswer` gates that the cache's *size*
changes nothing about the answer, and a slice count read off `nKV` would
change the order the partial softmaxes fold and so the last places of the
result. That is P7's own class of bug, re-entered through the rounding rather
than through the cost. So the number is fixed at 16 and the slices' *contents*
are the live depth.

## It is not bit-identical, and the repo already had a place for that

It cannot be: the rescales happen in a different order and fp32 addition is
not associative. `PinSchedule` is where this vertical puts such kernels —
L7d's split-K GEMV, L8d's three MoE rungs, L8e's two projections — and the
split attention is the sixth. The equalities are asserted with the pin on; the
"one token at a time, on the decode schedule" subtest is what measures the
cost of taking it off, and its bar is unchanged at 1e-3 rms.

What the split adds there, measured:

| | result_norm rms against the one-shot prefill |
|---|---|
| decode schedule, split off | 6.626e-04 |
| decode schedule, split on | **7.067e-04** |

And at the layer's own output, where the comparison is against llama.cpp
rather than against the other kernel — which is the comparison that decides
whether it is *wrong*, since neither kernel is the truth:

| | rms against llama.cpp, one decode step at cell 4095 |
|---|---|
| unsplit | 9.211e-04 |
| 16 slices | **9.213e-04** |

The two are indistinguishable. Between the arms themselves the difference is
1.26e-04 rms, and it is **flat in the slice count** — 1.02e-04 at two slices,
1.26e-04 at sixteen — which is the signature of the fp16 `P` rounding under a
different max sequence rather than of accumulated reassociation. The selection
is identical, cell for cell, on all 2051.

## What it bought, on the whole model

**48 layers, the shipped banks (D19 + D20 + D21), P7's protocol exactly — one
sequence walked forwards, `-pp 512 -tg 64`, ctx 66 880, corpus
`wiki.test.raw` — three arms interleaved, two passes of each, one binary built
once and the arms selected by environment variable.** The `base` arm is P7's
own code path and reproduces P7's published table, which is what makes the
rest comparable to it. CSVs: `results/p10_depth_{base,p89,p8910}[_run2].csv`.

Decode, tok/s, means of two passes:

| depth | base (P7) | +P8+P9 | +P8+P9+P10 | |
|------:|----------:|-------:|-----------:|---|
| 0 | 33.16 | 35.06 | **35.08** | 1.06x |
| 8 000 | 22.74 | 32.18 | **32.40** | 1.42x |
| 16 000 | 20.53 | 31.12 | **31.35** | 1.53x |
| 32 000 | 18.06 | 29.14 | **29.68** | 1.64x |
| 64 000 | 15.91 | 26.76 | **27.69** | **1.74x** |

Per-arm, pass 1 / pass 2 at 64 000: base **15.83 / 15.99**, +P8+P9 **26.80 /
26.71**, +P8+P9+P10 **27.71 / 27.66**. The within-arm spread is ≤2.4% at
every depth and ≤0.2% on the final arm.

**The regression itself**, which is the falloff and not the level:

| | before P7 | P7 | P8+P9 | P10 |
|---|---|---|---|---|
| decode, depth 0 → 64 000 | 0.14x | 0.48x | 0.76x | **0.79x** |
| decode at 64 000, tok/s | 4.05 | 15.91 | 26.76 | **27.69** |
| a decode step at 64 000 | — | 62.9 ms | — | **36.1 ms** |

The attention block itself, ms a token across the twelve layers:

| depth | 0 | 8 000 | 16 000 | 32 000 | 64 000 |
|---|---|---|---|---|---|
| base | 4.58 | 16.61 | 20.58 | 25.87 | 30.80 |
| +P8+P9 | 3.01 | 4.76 | 5.51 | 6.63 | 8.13 |
| +P8+P9+P10 | **3.03** | **4.63** | **5.30** | **6.21** | **7.28** |
| | 1.51x | 3.58x | 3.88x | 4.17x | **4.23x** |

**Prefill is untouched, which is the control.** All three changes are decode
paths and nothing else: prefill reads 691.1 → 691.3, 579.8 → 579.1, 557.6 →
558.8, 532.6 → 531.8, 440.7 → 439.6 tok/s across the five depths — inside the
noise at every one. Prefill's 0.64x falloff is the quadratic term any dense
attention has and is not this bug; llama.cpp has the same one.

## What it bought, on the 4-layer probe

**The 4-layer prefix (one attention layer), two passes of each arm,
interleaved, `-ctx 131072 -pp 512 -tg 32`.** This is the instrument the work
was done on — three seconds to stage against ninety — and it is here because
it resolves the kernels the whole-model table can only total. Means:

| depth | 0 | 4 096 | 16 384 | 65 536 |
|---|---|---|---|---|
| attention block, split off (ms/token) | 0.463 | 1.365 | 2.356 | 2.734 |
| attention block, split on | **0.355** | **0.489** | **0.708** | **1.050** |
| | 1.30x | 2.79x | 3.33x | **2.60x** |

The flash kernel alone, which is what was cut:

| depth | 0 | 4 096 | 16 384 | 65 536 |
|---|---|---|---|---|
| `attn.attn`, unsplit | 0.142 | 1.021 | 1.927 | 1.978 |
| `attn.attn.split` + `attn.attn.combine` | 0.035 | 0.143 | 0.276 | **0.295** |
| | 4.1x | 7.1x | 7.0x | **6.7x** |

The combine is flat at 0.006 ms — one dispatch a layer, 24 workgroups of four
waves over a 393 KB arena — and it is the whole of what the split costs.

**It pays at depth zero too**, 1.30x, which the gather could not have: at a
few hundred cells there is nothing to compact, but there are still only 24
waves, and sixteen slices of one block each is 384.

# P9 — the indexer was on one compute unit of forty

Splitting the attention left `attn.score` as the **largest kernel in the
block at depth**, and it is the same structural bug one dispatch over:
`llm_attn_score.comp` was dispatched one workgroup a token, so a decode step
— one token — scored every pooled block of the context, and then wrote every
cell of a 131 072-cell row, on a single compute unit. After P8, at 65 536
cells and one attention layer:

| kernel | ms/token | what it is |
|---|---:|---|
| `attn.score` | 0.304 | one workgroup: `nBid` block dots **and** an `nKV` cell expansion |
| `attn.attn.split` | 0.289 | the split flash attention |
| `attn.qkv` | 0.182 | flat in depth |
| `attn.select` | 0.176 | one workgroup, four radix passes over the live cells |

**What pinned it was a barrier.** The expansion reads `score[t][blk]` for an
arbitrary block, so every block had to be scored before any cell was
expanded, and inside one workgroup that is a `barrier()` — which is exactly
what stopped the scoring from being spread over more than one workgroup. Two
dispatches with a pipeline barrier between them is the same ordering with the
workgroup count let go of. So the expansion moved to
`llm_attn_expand.comp`, and both halves are striped over the grid's y.

**It is bit-identical, and that is the gate.** Neither kernel sums anything
across the stripe — every output element is one read, one add and one store —
so which lane writes it is not observable.
`TestAttnGPUCellStripeDoesNotChangeTheAnswer` demands the score, the expanded
cells and the selection back **to the last bit** at 2, 7, 16 and 64 stripes,
and gets them. That is a stronger gate than a tolerance on the layer's output
would be, and deliberately: a dropped cell score is one -inf among 2051
selected cells, which moves the answer by almost nothing and moves the
*selection* by a cell. `PinSchedule` has no business with this one.

# P10 — the selection could not be split, so it was widened

`attn.select` was the last of the three and the only one that **cannot** be
spread over the grid. Its four radix passes are a *reduction* over the row:
each pass narrows a threshold by eight bits using a histogram of everything
that matched the bits already fixed, so pass n+1 cannot start until pass n has
finished everywhere. Striping it the way P9 stripes the score would need a
global barrier between every pass — two dispatches each, nine a layer,
~96 extra dispatches a decode step at twelve layers, which at the ~6 µs a
dispatch the combine costs is more than the kernel spends.

**So the parallelism it got is waves, not workgroups.** At 65 536 cells the
row is 64 000 uints and `SEL_LDS` holds 4 096, so every one of the five passes
— four radix and the emit — streams the row out of DRAM again: 1.28 MB, from
one workgroup, on one compute unit. The workgroup was 256 lanes, four wave64s,
and four waves have nothing to hide a DRAM round trip behind. The width was
tied to the radix size by the bucket search, which is one lane per bucket; it
is now a build knob, and the shipped rung is **1024 lanes, sixteen waves**.

Two things had to be true for that to be free. The bucket search is a suffix
sum over `histo[tid]`, so lanes past the radix carry a zero count — and they
have to be held out of the `atomicMax` that elects the bucket, because a pass
whose `desired` has fallen to zero would otherwise elect the highest *lane*
and put a bucket index that does not exist into `prefix`. The emit's running
`taken` is already computed identically on every lane through a wave scan and
`shWave[WG/64]`, so it widens without change.

**It is bit-identical, and that is the gate**:
`TestAttnGPUSelectWidthDoesNotChangeTheAnswer` runs a 4095-token prefill and
a decode step at both widths and demands the selected set **and** the layer's
output back to the last bit. It gets them, on 8 400 896 selected cells.

| depth | 0 | 16 384 | 65 536 |
|---|---|---|---|
| `attn.select`, 256 lanes (ms/token, one layer) | 0.0069 | 0.0491 | 0.1744 |
| `attn.select`, 1024 lanes | 0.0080 | **0.0287** | **0.0932** |
| | 0.86x | 1.72x | **1.87x** |

Depth zero pays 0.0011 ms for the wider launch over a row that fits LDS
anyway, which is the shape of the trade and is 0.01% of a step.

## What is still open

**The gather, and it is now the only large one left.** After P8, P9 and P10,
what a decode step still gains between depth 0 and 64 000 is 5.9 ms a token,
and `attn.attn.split` is 3.0 of it — the split shortened the walk sixteenfold
but did not stop it being a walk over every key block. The gather P7 named is
the way to make it a walk over 2 051 cells instead, and it is worth what that
3.0 ms is, not the 1.978 ms a layer P7 costed it against.

**More slices is not the answer, and that is measured.** At 65 536 cells
`attn.attn.split` runs 0.318 / 0.288 / 0.291 / 0.283 ms a token at 8 / 16 / 32
/ 64 slices while the combine behind it climbs 0.004 / 0.006 / 0.010 / 0.018.
Sixteen is a plateau: past it the walk stops getting shorter and only the fold
gets longer.

**128 000 still does not complete**, and nothing here touches it.

