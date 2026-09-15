<!-- LLM.md L2e. The full-attention layer and the QSA indexer, CPU reference,
     against llama.cpp's own activations. Cited from llm/attn.go. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L2b](l2b-hyper-connections.md) · [L2d](l2d-ple.md) · phase 1

# L2e — the full-attention layer, the QSA indexer, and two more things the oracle does

**Result: layer 3 — the first of the 12 full-attention layers — runs in Go and
matches llama.cpp tensor for tensor, indexer included: the block-pooled scores
to 3.5e-07 relative, the fused query/gate split to 1.2e-06 rms, the attention
output to 2.4e-04.** And the chain composes: from the previous layer's output
through our mixer, our indexer, our attention and our combine, `hc_combine-3`
agrees to **9.7e-06 rms**.

**Two of the numerics are new, and neither is in the graph.** L2b found that a
Q8_0 matmul is evaluated over int8 activations; here the indexer's key cache
turns out to be **fp16** (worth **2072x** on the pooled key) and a matmul
between two **F32** tensors is evaluated on **fp16** matrix cores (worth
**69x** on the score). The same arithmetic in float64 lands in exactly the same
place, so neither is precision on our side.

| tensor | vs llama.cpp, rms | |
|---|---:|---|
| `indexer_k_raw-3` | 1.26e-06 | BF16 weights: no quantisation either side |
| `indexer_k_pooled-3` | **1.69e-07** | with the fp16 cache modelled; 3.49e-04 without |
| `indexer_k-3` | 1.71e-07 | normed and rotated |
| `indexer_q-3` | 8.63e-07 | |
| `indexer_score-3` | **5.11e-05** | on values to 145.8 — 3.5e-07 relative |
| `indexer_score_tokens-3` | mask exact | worst finite cell 3.2e-04 |
| `gate_reshaped-3` | **1.19e-06** | the fused projection's second half |
| `gate_sigmoid-3` | 1.43e-07 | |
| `attn_pregate-3` | 1.03e-04 | the flash-attention kernel's own accumulation |
| `attn_gated-3` | 2.25e-05 | |
| `attn_output-3` | 2.43e-04 | |
| **`hc_combine-3`, four stages composed** | **9.75e-06** | |

## Three things this layer does that no other transformer here does

**One projection produces the query and its gate.** `attn_q` is [2560, 12288]:
per token it is 24 heads of 256 query dims each immediately followed by 256
gate dims, so the query is a strided view and the gate is the same view shifted
by a head. The gate is a sigmoid on the attention *output*, not on the query.

**The rotary is interleaved M-RoPE — and on text it is NeoX.** `ggml_rope_multi`
with `GGML_ROPE_TYPE_IMROPE`, sections [11, 11, 10, 0] over n_rot = 64 of the
256 head dims. For a text batch llama.cpp puts the token position in the t, h
and w slots and zero in e, and no sector of [11, 11, 10] ever selects e — so
every pair rotates by the angle plain NeoX rope would give it.
`TestRoPEMultiIsNeoXOnText` asserts that **bit-for-bit** rather than leaving it
as an assumption, because the day an image batch arrives is the day it stops
being true. The second half of the contract has its own test: **64 of the 256
dims rotate and 192 are copied**, which is what a transcription that reads
n_rot as the head dim gets wrong without producing anything obviously broken.

**The indexer is a block-sparse scorer, not attention.** It projects the same
block input to a 128-wide key and four 128-wide query heads, pools the keys
over blocks of `compress_ratio` = 4 cells, scores every block against every
query with a **rectified** per-head dot product — a head that disagrees
contributes nothing rather than cancelling a head that agrees — and takes the
top `top_k + ratio - 1` cells. L2a priced it at 1.5% of a prefill graph; what
it buys is that a step reads at most 2048 keys however long the context is.

## The block structure is the cache's, not the prompt's

This is the part that cannot be inferred from the graph and has to be read out
of `set_input_qsa`. A block is `ratio` consecutive cell *positions*, and it
exists only if **all** of them are occupied. So a prompt of 7 tokens in a
256-cell cache gives:

- **one** real block, covering cells 0-3;
- **63** blocks that do not exist — and because `blk_cells` is zero-filled,
  each of them pools cell 0 four times, so `indexer_k_pooled` rows 1..63 are
  all exactly token 0's raw key. That is a check, not a curiosity, and it is
  one of the things that pinned the fp16 cache down;
- cells 4-6, which are occupied but unpooled, routed to a **spare block** that
  carries the 1e9 "always visible" marker — a finite marker rather than an
  infinity, so that it can never meet a `-inf` and produce a NaN.

The cell count is 256 and not 7 or 64 because **flash attention pads it**, and
every part of the indexer is cut against that number. The tests read it off
the dump rather than assuming it.

## The two new numerics

**1. The indexer's key cache is fp16, and modelling it is worth 2072x.**
`llama_memory_hybrid_idx` passes the context's `type_k` straight through to the
indexer cache, which is F16 by default. The keys are therefore rounded to
halves *between* being projected and being pooled — and `indexer_k_raw`, which
the callback sees before the cache write, is exact while `indexer_k_pooled`
three lines later is not. Rounding the cached copy takes the pooled key from
**3.49e-04 rms to 1.69e-07**.

**2. A matmul between two F32 tensors is an fp16 matmul, and modelling it is
worth 69x.** The indexer's score is `ggml_mul_mat(pooled, q)` with both
operands F32 tensors the callback hands us directly. Computing it in f32 gives
3.52e-03 rms against the reference; computing it in **float64** gives 3.52e-03
as well — the same answer, so it is not our precision. Rounding **both
operands to fp16** gives **5.11e-05**. The backend is putting an f32 matmul on
the fp16 matrix cores.

That is the third distinct thing this oracle does that its graph does not say,
after L2b's int8 activations and the cache above — and it is the most useful of
the three for this project, because it means **the fp16 kernels L2c and L2d
already run are not giving up anything the reference keeps.** They are the same
arithmetic.

## What is bounded rather than explained

`attn_pregate` sits at **1.03e-04 rms on values to 3.44** — 3e-05 relative —
where everything either side of it is 1e-06. Its inputs are modelled (int8
activations into the projections, an fp16 KV cache out of them, fp16 operands
into the score) and rounding the query to fp16 as well moves it by 3%. What is
left is the flash-attention kernel's own accumulation order, which the dump
enables by default and which is not reproducible from outside it. The same
category as L2b's three residual mixers: bounded, stated, not explained.

## And one thing about the oracle worth writing down

**llama.cpp's `TOP_K` is a selection, not a sort.** At token 4 cells 4-6 carry
the 1e9 marker and cells 0-3 carry 117.7, and `indexer_top_k` comes back as
the **identity permutation**. With `top_k + ratio - 1` = 2051 against a
256-cell cache the output names every cell, so the order is unobservable
downstream — it becomes an unmask, and unmasking everything is what the causal
mask already allows. `TestAttnTopKCannotBiteHere` therefore checks the set and
not the order, and asserts the property that makes that safe: running the
attention with the selection is **bit-identical** to running it without.

**So the selection cannot be validated at this prompt length at all**, and
that is L4's job rather than a gap here: at 4k context the width binds, the
order starts to matter, and a new dump is needed.

## How to reproduce

    go test ./llm/ -v -run 'TestAttn|TestRoPE'

Layer 3 is the only full-attention layer in the four the trace covers — the
interval is 4 — and `hc_mixed-3` is llama.cpp's own value for its input, so
every comparison above is that stage's own.

## What is not built

**The GPU port.** This is the CPU reference, in the order every vertical here
takes; the kernels are next, and three of the four pieces already exist in some
form — `shaders/qwen_attn_wmma_*` is causal GQA on the matrix cores at
headDim 128 (this model's is 256), `qwen_rope.comp` is NeoX rotary, and
`llm_gemm.comp`'s plain arm is the projections. What has no kernel anywhere in
this repo is the indexer's pool-score-select, which is small, oddly shaped, and
1.5% of a graph.

**Decode.** Everything here is prefill from position zero with a fresh cache.
The KV cache, the ring buffer and the indexer cache's own incremental pooling
are L7.
