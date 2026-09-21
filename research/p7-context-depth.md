# P7 — context depth: the cache's size was not the context's depth

> Measured 2026-09-21 on this machine, both arms the same hour, the shipped
> banks (D19 + D20 + D21). The baseline arm is the binary built from the
> commit this work started on; every number below is an interleaved A/B of
> the two binaries, never a comparison against a stored CSV.

## The finding

Decode fell **6.9x between depth 0 and 64 000 cells** (27.90 → 4.04 tok/s)
and prefill halved, and TODO.md recorded it as the vertical's largest
measured regression with the attribution open: *"all of it is attention"*,
with the note that the 2051-cell read cap says the selection is not the term
that grows.

It was two terms, not one, and neither was the selection.

**1. Three kernels walked the cache and not the context.** `nKV` is the
cell count the arenas were *allocated* for — 131 072 in a served process —
and `llm_attn_score.comp` scored all `nKV/ratio` pooled blocks, while
`llm_attn_select.comp` ran four radix passes and an emit over all `nKV`
cells. Neither number has anything to do with how deep the conversation is.
The cost is therefore paid **at depth zero, in full**, and it is linear in
the cache:

| cells allocated | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|
| attention, ms / 32 decode tokens, one layer, **empty cache** | 12.9 | 14.5 | 15.3 | 17.1 | 20.7 | 28.2 | **42.7** |

That is **6.94 ns per allocated cell, per token, per attention layer** —
**10.9 ms of a decode step** across the twelve attention layers at 131 072
cells, before a single cell is real. It is most of why the recorded sweep
starts at 24.27 tok/s where the headline decode is 36.19: the sweep
allocates a 131 456-cell cache and the headline does not.

**2. QSA was semantics and not a saving.** `llm_attn_wmma.comp` read the
selection bitmask exactly where it read the causal mask — out of the block
row max and out of P — and so walked **every** key block at every depth,
just as `build_attn_qsa` does. A decode step names 2051 cells however deep
the cache is; it was reading 64 000 of them. Measured slope: **0.25 µs per
live cell, per token, per attention layer** — 192 ms of a 248 ms token at
64k, which is the 85.6% attention share the sweep reported.

L4b-3 had written this down and left it: *"QSA becomes an optimisation at
decode, and only there… skipping unselected key blocks is the whole point.
L7's."* It was never L7's.

## The fix, and why it is bit-identical

Three changes, all of them *identities* rather than approximations, which is
what let the whole thing be gated on exact equality.

**The dead pooled blocks share one score.** `llm_attn_idx.comp` pools cell 0
`ratio` times for every block past the last whole one, and leaves the
position at 0 so the row is unrotated — so all `nBlocks - nBid` of them hold
*the same 128 halves* and therefore the same score. The kernel now computes
that one dot product and broadcasts it, instead of computing it tens of
thousands of times. **The tensor it writes is unchanged**, which is why
`TestAttnGPUIndexer`'s reference comparison still holds row for row.

**The selection is cut against the live cells.** Every cell past
`SEQ_PAST + T` is an `-inf` the causal mask wrote, and `-inf` is the
smallest key `f2ui` has: it can never be picked, so the four radix passes
and the emit skip it. The row *stride* stays `nKV`'s — the arenas are still
cut against the cache — and the row's remaining bitmask words are zeroed
rather than left stale, so `Selection()` still reads as a bitmask over the
whole cache. `desired` is clamped to the live count, since below the width
the selection is the identity and a `desired` the row cannot reach would
leave the bucket search with nothing to find. A useful side effect: at
shallow depth the row now fits `SEL_LDS` and the radix passes come out of
LDS instead of DRAM.

**A key block with nothing selected in it is skipped whole.** The bitmask is
staged *before* the block is read rather than after, and if no row has a bit
in it the iteration is skipped. This is exactly the existing masking, moved
earlier: an unselected cell already contributed nothing to O and nothing to
the row sum, so a block of them contributed nothing, and `blkMax` staying
`-inf` already had its `mo == mn` guard. One consequence had to be handled —
a pad row used to stage all-ones so its softmax denominator would not be
zero, which would have defeated the skip on every decode step (fifteen pad
rows of sixteen). It now stages zero and the epilogue turns a zero
denominator into a zero row, which is strictly better than the garbage it
wrote before, in an arena nothing reads either way.

## The gate

`TestAttnGPUCacheSizeDoesNotChangeTheAnswer` is new and it is the one that
matters: the 4k fixture staged twice, once in its own 4096-cell cache and
once in a cache three times too big, demanding **identical bits** out of
`attn_gated-3` and `attn_output-3` and the identical selected-cell set on
all 4096 rows. Every other test in `gpu_qsa4k_test.go` runs with `nKV`
equal to the live cell count, so nothing distinguished "the cache is this
big" from "the context is this deep" — which is the whole of what changed.
It passes, and so does the rest of `./llm`.

This is the shape P5c-6 and the two-row regime warn about: a block test that
cannot see the regime is not a gate on it.

And the graph gate above the block gate, for the same reason: `cmd/llm -gen
-n 48 -temp 0 -ctx 32768` on both binaries returns the **same 48 tokens**,
token for token, on a 131k-capable build in a cache 4 700 times the prompt.

## What it bought

**48 layers, the shipped banks, one sequence walked forwards, `-pp 512
-tg 64`, corpus `wiki.test.raw`.** Both arms in the same hour, the cache cut
at the sweep's own requirement (66 880 cells), **two passes of each arm** —
this is also the reproducibility pair TODO.md said the depth figures still
owed. Means below; the within-arm spread is ≤0.5% at every depth on the new
arm and ≤0.6% on the old except at 8 000, where the old arm moved 5%.

| depth | pp before | pp after | | tg before | tg after | |
|---:|---:|---:|---|---:|---:|---|
| 0 | 616.9 | **692.7** | 1.12x | 27.96 | **33.18** | 1.19x |
| 8 000 | 530.2 | **583.7** | 1.10x | 15.75 | **22.94** | 1.46x |
| 16 000 | 495.5 | **559.4** | 1.13x | 11.12 | **20.58** | 1.85x |
| 32 000 | 416.5 | **534.5** | 1.28x | 6.78 | **18.34** | 2.71x |
| 64 000 | 313.1 | **441.2** | 1.41x | 4.05 | **16.27** | **4.02x** |

Per-arm, tok/s, pass 1 / pass 2 — `tg` before **27.90 / 28.01**, **15.35 /
16.14**, **10.99 / 11.25**, **6.72 / 6.83**, **4.04 / 4.06**; after
**33.18 / 33.18**, **23.13 / 22.75**, **20.49 / 20.66**, **18.35 / 18.32**,
**16.32 / 16.22**.

**The regression itself**, which is the falloff and not the level:

| | before | after |
|---|---|---|
| prefill, depth 0 → 64 000 | 0.51x | **0.64x** |
| decode, depth 0 → 64 000 | **0.14x** (6.9x down) | **0.49x** (2.0x down) |
| attention share of a 64k decode step | 85.6% | 49.7% |
| attention, ms / 64 tokens at 64k | 13 581.8 | **1 948.0** (7.0x) |

And the level, not the falloff, moves too, because the cache-walk was
charged at every depth: `cmd/llm -gen -n 48 -temp 0 -ctx 32768` is
**32.43 → 35.92 tok/s**, which is 1.29x llama.cpp to **1.43x** and 85% of
the byte ceiling to **94%**.

The 4-layer probe says where each half of it came from, ms of attention per
32 decode tokens in a 131 072-cell cache, two interleaved passes:

| depth | 0 | 4 096 | 16 384 | 65 536 |
|---|---|---|---|---|
| before | 42.5 / 42.5 | 74.4 / 74.3 | 175.9 / 175.3 | 580.4 / 585.5 |
| after | 14.8 / 14.9 | 43.7 / 43.9 | 75.5 / 75.5 | **87.5 / 87.5** |

The depth-zero column is the cache-walk; the 65 536 column is the skip. The
skip does **better than the estimate** — ~513 selected blocks over 2 000 key
blocks predicts four in five empty, and the curve is flatter than that,
because the selection is clustered rather than uniform.

## What is still open

**Decode is still O(depth), just with a much smaller constant.** The skip
prunes key blocks; it does not make the cost independent of how many there
are to prune. The constant-cost version is the **gather** L4b declined:
compact the ≤2051 selected cells' K and V into a contiguous scratch and run
a 2051-cell attention, which is flat in depth by construction. That is the
remaining 1 948 ms of the 64k decode step above, and it is the only route to
a rate that does not care how long the conversation is.

**Prefill's falloff is not this bug.** At ubatch 512 and ~50% density a
32-cell key block is eight indexer blocks and the chance all eight miss is
~0.4% (L4b-3), so the skip is close to inert there; what prefill gained
(1.11-1.40x) is the cache-walk, and the 0.64x that remains is the quadratic
term any dense attention has. llama.cpp has the same one.

**128 000 still does not complete**, and nothing here touches it: the fill
dies at ~115k cells in P0's timestamp pathology reached by depth instead of
row count. The untried route is unchanged — make `DispatchMultiMarked`'s
per-dispatch marks optional, since they are pure instrumentation and
wall-clock tok/s would survive losing the per-block breakdown at that depth.

**The cell expansion is the last O(nKV) loop.** `llm_attn_score.comp` still
writes `-inf` into every cell of the row past the live count, because that
tensor is compared against `indexer_score_tokens-3` in full and the mask has
to agree exactly. It is 6.3 MB of stores a token across the twelve layers,
~0.1% of a step, and it is written down rather than done.
