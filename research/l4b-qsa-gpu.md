<!-- LLM.md L4b. The QSA selection as a kernel, the bitmask the attention
     reads, and what sparsity costs at prefill. Cited from
     llm/gpu_attn.go and shaders/llm_attn_select.comp. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L2f](l2f-attention-gpu.md) · [L4a](l4-qsa.md) · [L3b](l3b-deltanet-gpu.md) · phase 1

# L4b — the selection on the device: one dispatch against twenty-four, and what sparsity costs

**Result: the QSA selection runs as a single kernel writing a per-cell
bitmask, at 1.94x llama.cpp's TOP_K and its GET_ROWS together and 2.59x at
ubatch 2048; it reproduces `topK`'s selection cell for cell on all 4096 rows
of the 4 k fixture with no tolerance at all; and the layer's output at 4096
tokens is 9.87e-04 rms against llama.cpp — inside what the same comparison
gives at 7 tokens.** Two things it also measures rather than assumes: the
selection *costs* the attention kernel **1.21x** and saves it nothing, which
is what "the sparsity is semantics at prefill" means with a number on it; and
the kernel's disagreement with the reference's own selection is **17x** L4a's
CPU figure, for a reason that is upstream of this kernel and is L2f's fusion.

## What was left

L4a settled what the selection *is* — a radix select, ported pass for pass,
reproducing `indexer_top_k-3` set for set on the 2045 rows of the 4 k dump
where it bites — and left the kernel. Two pieces were missing:

- a **radix-select kernel**, one workgroup a token, reading the per-cell
  scores `llm_attn_score.comp` already writes;
- `llm_attn_wmma.comp` **reading the result** beside the causal mask.

Neither can be tested below 2051 cells. At the 256-cell cache the rest of this
vertical is checked against, the selection names every cell and running it is
the identity (L2e-5), so `AttnGPU` decides for itself: `sparse` is
`TopK > 0 && selWidth < nKV`, fixed when the layer is staged, and false
everywhere L2f measured. **None of L2f's numbers move** — the default shape
re-measures at 2.38x today against its published 2.40x.

## The shape of the kernel, and three departures from the reference

`shaders/llm_attn_select.comp`, one workgroup of 256 lanes a token:

1. the row's cells are mapped through `f2ui` and cached in LDS — 16 KB at a
   4096-cell cache, which turns five passes over DRAM into one;
2. four 8-bit radix passes narrow the threshold key's prefix;
3. one lane per **32-cell word** emits the strictly-above bits, then as many
   tie bits as the fill still needs, in ascending cell index.

> **It writes a bitmask, not an index list.** The reference emits `width`
> int32 cell indices and then spends a second op — `TOPK_QSA GET_ROWS`, 12
> dispatches and 207 us a graph — turning them back into an f16 mask, because
> `build_attn_qsa` runs **dense** flash attention over every cell and adds the
> mask to the scores. A bitmask is what the consumer wants, it is 32x smaller
> than the index list (nKV/8 bytes a token against 4·width), and it deletes
> that dispatch outright.
>
> **The tie fill is by ascending cell index, not by whoever won the atomic.**
> L4a-2 measured that the reference's order differs on 4096 of 4096 rows
> between two runs of the same prompt while its visible set differs on none. A
> bitmask has no order to get wrong, so ours is a function of the scores
> alone — which is what makes the test below exact rather than statistical.
>
> **It does not gather.** The reference's QSA arm recomputes
> `score[cell_blk[i]] + mask[i]` per element and caches it in a scratch buffer
> to avoid doing it five times. `llm_attn_score.comp` has already written that
> tensor, so this kernel reads it straight.

The bitmask lives in the fp32 activation arena read as uints through a second
descriptor — `binding = 4` in `llm_common.glsl`, the same `VkBuffer` bound
twice. Punning a mask through `uintBitsToFloat` would put NaN payloads in a
storage buffer and trust them to survive; a second descriptor costs nothing
and says what the tensor is.

## L4b-1: one dispatch against twenty-four, and 2.40 ms becomes 1.24

Per 512-token graph, against the two lines of llama.cpp's own graph that are
the selection (`results/l2a_prefill_ops.csv`, the same 2048-cell cache):

| | disp | llama.cpp | disp | ours | |
|---|---:|---:|---:|---:|---:|
| TOP_K K=2048 (2048,512,1,1) | 12 | 2.19 ms | | | |
| TOPK_QSA GET_ROWS | 12 | 0.21 ms | | | |
| **selection, ubatch 512** | **24** | **2.40 ms** | **12** | **1.24 ms** | **1.94x** |
| **selection, ubatch 2048** | 24 | 8.89 ms | 12 | 3.42 ms | **2.59x** |

Both sides are doing the identity at this width — the reference runs it
anyway, which is what makes the comparison like for like. Reproducibility
across two runs is **0.1%** (103.4 and 103.2 us a dispatch).

With the selection counted on both sides, the full-attention layer at `-ub
512` is **68.8 ms of llama.cpp's graph against 29.6 — 2.32x.** A real run at
that shape would not select at all, and then it is **68.8 against 27.9,
2.47x**: the reference spends 2.40 ms a graph on a selection that names every
cell, and `sparse` is what declines to.

## L4b-2: the bucket search was the kernel, and the histogram never was

The first build was **192.0 us** a dispatch, 22 GB/s, an order of magnitude
off the bus for a kernel whose whole input is 8 KB a workgroup. The cost was
not where a radix histogram usually puts it.

> **The reference walks 256 radix buckets serially on lane 0** while the other
> 255 wait at a barrier — four times a row, so 1024 dependent LDS reads on the
> critical path. The same answer is a suffix sum: `acc` at bucket *b* is
> exactly the count above *b*, so the bucket wanted is the **largest** *b*
> whose inclusive suffix still reaches `desired`, the suffix is non-increasing
> in *b*, and that is an `atomicMax` over the lanes that satisfy it. As a
> shared-memory scan: **192.0 → 119.7 us, 1.60x.** As a subgroup scan over
> four pinned 64-wide waves, two barriers instead of sixteen: **106.5**. The
> tie fill's scan took the same treatment: **103.3 us, 1.86x in total.**
>
> **Per-wave histograms are 1.06x *against*.** Four private 256-bucket
> histograms reduced at the end — the standard fix for LDS atomic contention,
> and the obvious suspect when a row's scores cluster into a handful of top
> bytes — measured **126.8 us** against 119.7. The atomics were never the
> cost, which is why it is worth measuring before rewriting them.

What is left is 42-61 GB/s, still 4-6x off the bus, and the shape of the
remaining cost is legible: three of the four radix passes walk every cell to
test a prefix that almost none of them match. It is **0.1% of a prefill
graph**, so it is written down rather than chased — see the open questions.

## L4b-3: the selection costs the attention kernel 1.21x, and saves it nothing

This is the number behind "the sparsity is semantics at prefill". The same
kernel, the same arithmetic, 4096 tokens in a 4096-cell cache, with and
without the bitmask:

| attention kernel | us a dispatch | TFLOP/s | |
|---|---:|---:|---:|
| dense causal | 9950 | 20.7 | |
| reading the bitmask | **12046** | 17.1 | **1.21x** |
| reading it per element, out of the arena | 18546 | 11.1 | 1.90x |

> **The first version cost 1.90x, and staging recovered it.** Bit-testing
> `actu` per element is BM·BN reads in the mask loop and another BM·TILE in
> the row max — 512 global reads for a block whose entire mask is
> BM·BN/32 = 16 words. Loading those 16 words into LDS once a block and
> reading bits out of them took 18546 → 12046 us.
>
> **The residual 1.21x is structural.** Without a selection only the diagonal
> key block passes through the P mask loop; every earlier block is entirely in
> the past of every row and is left alone. With one, cells are dropped
> anywhere, so all 128 blocks of a 4096-cell row pass through it. Both rungs
> behind the winner pay the same 1.20-1.25x, so this is a property of the mask
> and not of a schedule.

Two smaller things the selection forced, both of which are wrong answers
rather than slow ones if missed:

- **`exp2(-inf - -inf)` is a NaN.** A key block with nothing selected in it
  leaves both the block max and the running row max at -inf, and the rescale
  `exp2(mo - mn)` then poisons the accumulators for the rest of the row. The
  dense path can never reach it — block 0 always contains cell 0, which every
  row can see — so it is a case the selection creates. Guarded by `mo == mn`.
- **A pad row has no selection.** `llm_attn_select.comp` does not run past the
  prompt, so the attention kernel treats rows at or beyond `tokens` as dense.
  Their output is never read; what this avoids is a row with no visible cell
  at all and a zero softmax denominator.

## L4b-4: exact against the CPU, 0.0381% against llama.cpp — and the 17x is upstream

> **The bitmask is `topK`'s selection, cell for cell, on all 4096 rows, with
> no tolerance.** The kernel and `attn.go`'s `topK` are the same algorithm
> over the same numbers — the comparison feeds the CPU the cell scores the
> device itself wrote — so the only thing that could differ is the tie fill,
> and a bitmask has no order to get it wrong in. Past the biting point every
> row holds exactly `width` = 2051 cells. `TestAttnGPUSelectionIsTheCPUs`.
>
> **Against llama.cpp's own selection: 2400 of 6 298 621 visible cells,
> 0.0381%, over 490 of 4096 tokens, worst row 14 cells.** L4a-4 measured
> 138 cells, 0.0022%, over 30 tokens for the CPU path. **17x**, and the cause
> is not this kernel.

The cause is L2f-3's fusion, and it is worth stating plainly because it is the
first place in the vertical where a structural choice costs accuracy:

> **The device cannot take L4a-6's bf16 activation, because the indexer's two
> BF16 weights are column ranges of the layer's one fp16 matrix.** L4a-6 found
> that above the 8-column threshold a BF16 `src0` against an F32 `src1` sets
> `y_non_contig`, converts the *activation* to bf16 and runs a bf16 × bf16
> kernel — eight mantissa bits — and that modelling it was worth 2061x on
> `indexer_k_raw`. Six of llama.cpp's matrices being one of ours is what makes
> the projection 1.53x, and the indexer's two are two of the six.

Measured, at 4096 tokens, against llama.cpp:

| tensor | device | L4a's CPU model | |
|---|---:|---:|---|
| `indexer_k_raw-3` | **9.319e-04** | 4.454e-07 | 2093x — and 9.179e-04 is L4a-6's *unmodelled* figure |
| `indexer_k-3` | 6.122e-04 | — | |
| `indexer_q-3` | 9.138e-04 | — | |
| `indexer_score-3` | 1.929e-02 | 7.93e-04 | 24x, on values to 149.4 |

So the device sits exactly where an fp16 operand puts it, the CPU sits where a
modelled bf16 one does, and the 24x on the score becomes 17x on the selection
because a top-k is discontinuous. The trade is **one weight instead of six**,
and what it costs downstream is bounded by the next finding.

## L4b-5: the layer at 4 k, and the control that says the bitmask is read

`attn_output-3`, 4096 tokens, layer 3, from llama.cpp's own `hc_mixed-3`:

| | rms against llama.cpp |
|---|---:|
| `attn_gated-3` | 3.475e-04 |
| **`attn_output-3`** | **9.872e-04** |
| the same layer with the selection off | 1.958e-03 |
| sparse against dense | 1.696e-03 |

Three things follow. The layer's output at 4096 tokens is **inside** the
1.14e-03 the same comparison gives at 7 tokens, so the 17x on the selection
does not propagate: 0.0381% of cells entering a softmax over 2051 of them
moves the result less than the reference's own fp16 accumulation does (L4a-5).
The dense control is **2.0x further** from llama.cpp, which is what says the
bitmask is being read *and* that reading it is what moves the layer towards
the reference — a kernel that loaded the mask and ignored it would pass every
tolerance above by being a dense causal attention, which at 7 tokens is the
right answer and at 4096 is a different model.

## L4b-6: the attention ladder's top two invert with length, and the selection is not why

L2f-5 measured the ladder monotonic in how much a wave holds live: qt1_kt2
170 us, qt1_kt4 223, qt2_kt4 241. At 4096 tokens it is not:

| rung | dense | sparse | |
|---|---:|---:|---:|
| **qt1_kt2** | **9950** | **12046** | the winner at both lengths and both densities |
| qt2_kt4 | 10770 | 13033 | |
| qt1_kt4 | 13969 | 17513 | |

The inversion is in the **dense** column too, so it is a fact about length and
not about the selection — and the winner does not move, which is what the
`DefaultAttnKernel` rests on. §3.3's conclusion is unchanged; what a long
prompt changes is the order of the two rungs behind the winner.

## How to run it

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go run ./cmd/llm -attn -model $M -tokens 512                    # L2f's shape: no selection
    go run ./cmd/llm -attn -model $M -tokens 512,2048 -sel on       # price it where it is the identity
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096         # where it bites
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096 -sel off  # the dense control
    go run ./cmd/llm -attn -model $M -tokens 4096 -ctx 4096 -ladder -iters 5

    go test ./llm/ -v -run 'TestAttnGPUSelection|TestAttnGPULayer4k|TestAttnGPUIndexer4k'

`results/l4b_qsa.csv` carries all five runs, with a `selection` column.

## What is still open

- **The select kernel is at 42-61 GB/s**, 4-6x off the bus. Three of its four
  radix passes walk every cell to test a prefix almost none of them match; a
  compaction after pass 1 would fix that. So would a **block-granular**
  histogram: all `ratio` cells of a whole block carry one score, so a
  4096-cell row is 1024 distinct values plus a split tail, and the histogram
  could count blocks with a weight. It is 0.1% of a prefill graph.
- **The 1.21x the selection costs the attention kernel** is the P mask loop
  running on every key block. A per-block "nothing selected" test would let
  the kernel skip whole GEMMs — real sparsity, a real saving — but at ~50%
  density and 8 indexer blocks to a 32-cell key block the probability of an
  empty block is ~0.4%. It becomes interesting at a long context, where the
  selected 2051 cells are a shrinking fraction of the cache.
- **Should the indexer's two projections leave the fused matrix?** They are
  the only BF16 weights in the model, and being fp16 column ranges of one fp16
  matrix costs 2093x on `indexer_k_raw` and 17x on the selection. Un-fusing
  them into a bf16 matmul of their own would cost a dispatch — llama.cpp
  spends 5.03 ms a graph on those two lines — and buy 24x on the score. The
  layer's output does not currently care (L4b-5); whether the *model* does is
  L8's perplexity run.
- **The selection at decode.** Everything here is prefill, where the reference
  runs dense attention over a mask and the selection is semantics. At decode
  one token reads 2051 of up to 262 144 cells, the bitmask is 32 KB, and
  skipping unselected key blocks is the whole point rather than a 0.4% chance.
  That is L7's, and it is the first place QSA is an optimisation.
