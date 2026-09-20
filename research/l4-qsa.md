<!-- LLM.md L4a. The 4k dump, the QSA selection, and what the reference's
     arithmetic becomes at a real ubatch. Cited from llm/attn.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L2e](l2e-attention.md) · [L2f](l2f-attention-gpu.md) · [L3](l3-deltanet.md) · phase 1

# L4a — the 4 k dump: the selection settled, and the oracle's arithmetic changes underneath it

**Result: the QSA selection is reproduced exactly — set for set, on all 2045
rows of a 4096-token prefill where it actually bites — and the 4 k dump it
needed turns out to answer a second question that was not about QSA at all.
At a real ubatch the reference accumulates every quantised matmul in
`fp16`, and it meets the indexer's BF16 weights with a `bf16` activation.
Neither is visible at 7 tokens. Modelling the second is worth 2061x; the
first cannot be modelled and instead sets every tolerance at prefill to ~5e-3
rms.** Two bugs fell out of the second finding, both in half-precision
subnormals, both invisible to five stages of tensor comparison.

## Why a second dump

Everything in L2 and L3 was checked against a 7-token trace, and two separate
things were known to be untestable there:

- **The selection.** The reference asks for `top_k + ratio - 1` = 2051 cells
  out of a **256**-cell cache, so it names every one of them. `TestAttnTopK`
  `CannotBiteHere` asserts exactly that — running the attention with the
  selection is bit-identical to running it without (L2e-5).
- **Every tolerance in the vertical.** L3a-5 found `ggml_vk_mul_mat`'s
  threshold: at up to `mul_mat_vec_max_cols` = 8 output columns a matmul goes
  to the f32 *vector* path, above it to the coopmat GEMM. A 7-token dump is
  entirely on the first, and no real prefill ever is.

So: 4096 tokens of wikitext-2's test split in a 4096-cell cache, one ubatch,
the same pinned build. `reference/eval_dump.c` grew `-f` (a prompt from a file)
and `-nt` (truncate to an exact token count), because the fixture's length is
arithmetic the tests read back and wants to be a chosen number. 8.0 GB, 116
tensors, ~4 minutes.

> **A hazard the dump surfaced, worth writing down.** `/home/kube/repos/llama.cpp`
> has been pulled forward to `d1d3c3396` while `build/bin` is still
> **`cff184438`** — the build behind L1's `pp2048 388.60`, `tg128 25.15`,
> `PPL 4.0340` and the GDN normalisation bug (L3a-4). Reading
> `src/models/qwen4exp.cpp` today therefore reads a *different implementation*
> from the one the oracle runs, and `include/llama.h` and `ggml/include/ggml.h`
> have both moved. The dump is now built against headers extracted from the
> build's own commit (`git archive cff184438 include ggml/include`) rather than
> from the working tree, and every source citation in this file is
> `git show cff184438:…`.

The filter is layers 0 and 3 plus the entry tensors, which brings **the whole
MoE block along for free** — `ffn_moe_logits`, `probs`, `argsort`, `topk`,
`weights`, `gate`, `up`, `swiglu`, `down`, `weighted`, `out`, both layers. L5's
fixture exists before L5 starts.

## L4a-1: the selection is a radix select, and its threshold is exact

`build_qsa_top_k` ends in `ggml_top_k(expanded, min(n_kv, top_k + r - 1))`, and
on this backend that is `topk_radix_select.comp`. It is **not a sort**:

1. an order-preserving `float -> uint` map (`f2ui`: flip the sign bit of a
   positive, invert every bit of a negative, so `-inf` is the smallest key);
2. four 8-bit radix passes narrowing a prefix, each counting only the keys
   whose already-fixed high bits match, top-down to find the bucket holding
   the k-th largest;
3. emit every cell **strictly above** that threshold;
4. fill the rest from the cells **equal** to it, in `atomicAdd` order.

`llm/attn.go`'s `topK` is that, pass for pass. Fed llama.cpp's own
`indexer_score_tokens-3`, it reproduces `indexer_top_k-3` on **all 4096 rows**:
every strictly-above cell is in the reference's set, and every cell the
reference has that we do not is at our threshold. It is also what makes the
fixture affordable — four passes over a row where a partial selection sort
costs 2051 of them.

**And the tie fill is reproducible, which was not expected.** All `ratio` cells
of a block carry one block score, so the tie group at the threshold *is* a
block and the fill takes 1, 2 or 3 of its 4 cells — a block **split**. Taking
them in **ascending cell index** gives the reference's set on all 2045 rows
where the selection bites, with nothing over. That is a wave resolving an
atomic in lane order across `ratio` consecutive cells, not something the shader
promises, so the other half of the evidence is measured too (L4a-3).

## L4a-2: the selection is block-granular, and now that is checked rather than read

vLLM supplied the claim during L3 — top `top_k/ratio` = 512 **blocks** by
score, each expanded to its `ratio` cells, plus the causal tail — and llama.cpp
only implies it, by biasing the tail block by 1e9 and taking the top
`top_k + ratio - 1` **cells**. At 4096 tokens it is a property of the
reference's own output, and it holds exactly. For every token past the point
where the selection bites, the selected-and-visible cells are:

| | count | |
|---|---:|---|
| the causal tail | `(i+1) mod ratio` | cells `tail_start..i`, carrying the 1e9 marker |
| whole blocks | `(width - tail) / ratio` = **512** | always exactly 512 |
| one split block | `(width - tail) mod ratio` | 0, 1, 2 or 3 cells, and it is the **lowest-scoring block in the selection** |

Over the 2045 biting rows the split is 3 cells 512 times, 2 cells 511 times,
1 cell 511 times and **0 cells 511 times** — the benign case, where the
threshold lands on the last cell of a block and the whole block goes in.

So a cell-level top-k that split blocks freely would be a different model, and
**`top_k`'s tie order being unobservable is true by construction** (L2e's
closing worry) only because blocks stay whole apart from that one split.

## L4a-3: the reference's order is not reproducible and its selection is

A second dump of the same prompt, same binary, same machine:

| | |
|---|---|
| `indexer_score_tokens-3` | **byte-identical**, all 16.8 M cells |
| `indexer_top_k-3`, order | differs on **4096 of 4096** rows |
| `indexer_top_k-3`, set | differs on **1967 of 4096** rows |
| `indexer_top_k-3`, set restricted to **causally visible** cells | differs on **0 rows, 0 cells** |

All of the run-to-run set difference is among the `-inf` cells the causal mask
drops anyway — at token `i` the row has `i+1` finite cells and 2051 slots, so
below `i = 2051` most of the output is an arbitrary fill of masked cells. What
attention can actually read is identical.

**Which is the useful shape of the result**: a kernel of ours may emit the
selection in whatever order is cheapest, because nothing downstream reads the
order, and it must reproduce the *set*, which it can.

## L4a-4: how much of the selection survives our own scores

The gate above feeds the selection llama.cpp's scores, because what is under
test there is the rule. The engineering question is the other one: a top-k is a
**discontinuous** function of its input, so two blocks whose scores differ by
less than our error can swap across the threshold and then our attention reads
a cell the reference does not.

Run end to end from `hc_mixed-3` through our own indexer, counting only
causally visible cells:

| | cells differing | of | rows affected |
|---|---:|---:|---:|
| before L4a-6's bf16 model | 2354 | 6 298 621 | 471 of 4096 |
| **after** | **138** | 6 298 621 (0.0022%) | **30** of 4096 |

Worst row: 8 cells of 2051. The floor is not zero and cannot be — a block whose
score ties another to within 1e-5 is a coin flip — but at 0.0022% it is two
orders of magnitude below what would change a token.

## L4a-5: every quantised matmul accumulates in fp16 at a real ubatch

This is the larger finding and it needs no arithmetic to state. At 4096 columns
**every value the reference writes out of a quantised matmul is exactly
representable as an IEEE half**: 100.0% of a sample of `Qcur_full-3` (12288 x
4096), `Kcur`, `Vcur`, `attn_output`, the DeltaNet's `z-0` and
`linear_attn_out-0`, the MoE's `ffn_gate`/`ffn_up`, the shared expert's
`ffn_shexp`. Every one.

That is an fp16 **accumulator**, not an fp16 operand.
`ggml_vk_get_mul_mat_mat_pipeline` returns the `.f16acc` variant whenever the
precision is `GGML_PREC_DEFAULT`, the device has fp16 and its coopmat supports
an fp16 accumulator — all true on RADV STRIX_HALO. A K = 2560 dot product
summed in halves carries ~sqrt(K/16) roundings of 2^-11.

The exceptions confirm it rather than weakening it, and the test asserts them:
`hc_inject` ([10240, 4], F32), `shared_expert_gate` (F32, one column),
`ffn_moe_logits` (the F32 router) and `indexer_k_raw` (BF16) are **0.0%**
fp16-exact.

**The consequence is that the tolerance at prefill is the reference's error,
not ours.** Against `Qcur_full-3`, three models of the *operands* — exact f32,
fp16 operands, and the reference's own int8 activations (L2b-2) — land within
**1.06x** of each other and all of them sit ~**5.4e-03 rms** from the
reference, on a tensor whose own rms is 1.21. No chunked fp16-accumulator model
improves on it either (chunk 8/16/32/64/128 all sit at 5.8-6.3e-03), because
matching it would mean reproducing a tile schedule rather than an arithmetic.

So the reading L2c-3, L2f-6 and L3b-6 reached at 7 tokens holds an order of
magnitude larger and from a different cause: **at a real ubatch the reference
is the side losing precision.** A tolerance of 1e-6 on a quantised projection
is a fact about a 7-token dump; at prefill it is ~5e-3 rms, and it is the
oracle that moved.

## L4a-6: and the indexer's BF16 weights meet a bf16 activation

The smaller finding, and the one that changed the code. The indexer's
`index_q_proj` and `index_k_proj` are the only BF16 tensors in this model.
Above the 8-column threshold, `ggml_vk_mul_mat_q_f16` sees a BF16 `src0`
against an F32 `src1`, which sets

	y_non_contig = (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) || …

so the **activation** is converted to BF16 and a BF16 x BF16 kernel runs. Eight
mantissa bits, not ten. On `indexer_k_raw-3` at 4096 tokens:

| activation | rms vs llama.cpp | |
|---|---:|---|
| f32 (what L2e's model did) | 9.179e-04 | |
| **bf16** | **4.454e-07** | **2061x** |
| fp16 | 9.319e-04 | 1.02x *worse* than f32 |

The fp16 row is the control that matters: this is a **different** numeric from
L2e-3's "an F32 x F32 matmul is an fp16 matmul", not the same one seen again.
And the checkpoint's BF16 weight is already on that grid — 0 of 327 680 values
move — so only the activation does.

Modelled, the whole indexer chain comes back at a real ubatch:

| tensor | f32 activation | bf16 activation | |
|---|---:|---:|---|
| `indexer_k_raw-3` | 9.179e-04 | **4.454e-07** | 2061x |
| `indexer_k_pooled-3` | 4.774e-04 | **6.929e-06** | 69x — the fp16 cache (L2e-2) is the floor now |
| `indexer_k-3` | 5.799e-04 | **2.396e-05** | 24x |
| `indexer_q-3` | 8.774e-04 | **1.319e-05** | 67x |
| `indexer_score-3` | 1.930e-02 | **7.930e-04** | 24x — on values to 149.4, 5e-6 relative |

And with the inputs fixed, the score's own numeric can be re-asked at a real
ubatch, where it was previously buried: f32 gives 5.744e-03, **fp16 gives
7.930e-04**, bf16 gives 4.649e-02. **L2e-3 holds at 4096 columns** — the F32 x
F32 matmul is still an fp16 matmul, 7.2x better than f32 and 59x better than
bf16. One numeric confirmed at the width that matters, one replaced.

## Two bugs, both in half-precision subnormals

The fp16-exactness census is a sharp instrument, and it immediately caught
something: `ffn_shexp-3` came back **99.9%** fp16-exact where every other
quantised matmul was 100.0%. The 11 047 exceptions of 10.5 M values were all
the smallest magnitudes and all negative.

**`f16Round` returned `|x|` for every negative subnormal half.** It built the
result as `Float32frombits(sign) + q*2^-24` — and `-0.0` plus a positive
magnitude is positive. It is used for the reference's fp16 KV cache, the
indexer's fp16 key cache and every fp16-operand model in the package.

Fixing that exposed the second. **`f16`, the dump reader's widening, returned
half the correct value for every subnormal half**: a subnormal is `man * 2^-24`,
and shifting until bit 10 is set leaves `1.f * 2^(-14-k)`, so the exponent
field is `114 + e` where the code had `113 + e`. The smallest half came back as
2^-25.

Neither is reachable through a normal half, which is why five stages of
tensor comparison never saw them. `TestHalfPrecisionRoundTrips` now walks all
65 536 halves and asserts that each widens to a float32 that rounds straight
back, plus the two boundary values as numbers rather than bit patterns, plus
the sign of a negative subnormal — and the same round trip for bfloat16.

## What the tests are

    go test ./llm/ -run 'TestQSA'          # the selection, at 4096 tokens
    go test ./llm/ -run 'TestMatMuls|TestF16Acc|TestIndexerProjection|TestHalfPrecision'

| test | what it pins |
|---|---|
| `TestQSASelectionBitesAt4k` | the width is 2051 of 4096, the first token to drop a visible cell is **2051**, and the last drops 2045 |
| `TestQSASelectionIsARadixSelect` | our selection is the reference's, from the reference's scores — algorithmically on all 4096 rows, set-for-set on the 2045 biting ones |
| `TestQSASelectionIsStableAcrossRuns` | byte-identical scores, a different order every row, the same visible set every row |
| `TestQSASelectionIsBlockGranular` | 512 whole blocks + the tail + at most `ratio-1` split cells, and the split block is the lowest-scoring one |
| `TestQSAIndexer4k` | every indexer tensor at 4096 tokens, at the tolerances L4a-6 measures |
| `TestQSASelectionSurvivesOurScores` | 0.0022% of visible cells differ when the scores are ours |
| `TestMatMulsAccumulateInF16AtAWideUbatch` | the census, including the four matmuls that are *not* on it |
| `TestF16AccumulatorSetsTheTolerance` | three operand models within 1.06x, all ~5.4e-03 away |
| `TestIndexerProjectionIsBF16AtAWideUbatch` | bf16 2061x, fp16 *worse* than f32, the weight already on the grid |
| `TestHalfPrecisionRoundTrips` | all 65 536 halves, and all 65 536 bfloats |

## What is not built

**The GPU half — L4b.** `llm/gpu_attn.go` computes `indexer_score_tokens` on
the device already (`llm_attn_score.comp`), but `llm_attn_wmma.comp` applies
only the causal mask; the selection is not wired in. What that needs is a
radix-select kernel of this file's shape — one workgroup a token, four
histogram passes over `n_kv` — writing a per-cell **bitmask** rather than an
index list, since our attention walks key tiles and llama.cpp itself turns
`top_k` back into a mask (`build_attn_qsa` builds `kq_mask_all`, `ggml_set_rows`
zeros the selected rows, and then runs **dense** flash attention over every
cell). Then the 4096-token staging, and the gate LLM.md states: layer 3's
output at 4 k context.

**Everything past the layer.** `attn_output-3` at 4096 tokens has not been
compared end to end on either side, because the CPU reference's projections are
139 G multiply-adds over 126 MB of weights a token at that length. The layer's
pieces are each checked from llama.cpp's own input; the whole layer at 4 k is
L4b's, on the device, where it costs milliseconds.
