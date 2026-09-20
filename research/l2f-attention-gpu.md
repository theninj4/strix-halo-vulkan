<!-- LLM.md L2f. The full-attention layer and the QSA indexer on the GPU, six
     dispatches against nineteen, checked against L2e's CPU reference and
     llama.cpp's own tensors. Cited from llm/gpu_attn.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L2c](l2c-hc-kernel.md) · [L2d](l2d-ple.md) · [L2e](l2e-attention.md) · phase 1

# L2f — the full-attention layer on the device, and a ladder that inverts

**Result: the layer runs in six dispatches where llama.cpp's graph spends
nineteen, and 66.3 ms of its 512-token prefill graph becomes 27.6 — 2.40x, or
1.60x once the one row that is not like for like is repriced.** Every tensor
agrees: the fused projection's six column ranges to 1.7e-04 rms, the indexer's
pooled key to 3.1e-04, its score to 3.8e-07 *relative*, the gated context to
3.3e-05 against the f32 model and the layer's output to 5.9e-05 — **7.4x and
19.3x nearer that model than llama.cpp is**, which is the int8 activations the
reference brings and we do not.

## The graph

| | llama.cpp | ours |
|---|---:|---:|
| dispatches per layer | **19** (17 excluding the selection) | **6** |
| per 512-token graph, 12 layers | 228 (204) | 72 |
| time per graph | **66.3 ms** | **27.6 ms** |
| share of the whole 1164.7 ms graph | 5.7% | 2.4% |

	qkv    xn -> query, gate, key, value, indexer query, indexer key
	pack   per-head norm + IMRoPE + the fragment tiling, for q, k and v
	idx    the indexer's pooled key and its query: pool, norm, rope, fp16
	score  the rectified block score, its bias, the cells and the mask
	attn   causal GQA on the matrix cores, with the output gate in its epilogue
	out    the output projection

| dispatch | disp | llama.cpp | disp | ours | |
|---|---:|---:|---:|---:|---:|
| qkv | 60 | 28.2 ms | 12 | **18.4 ms** | 1.53x |
| pack | 48 | 2.8 | 12 | **0.7** | 3.97x |
| idx | 48 | 0.9 | 12 | **0.1** | 12.20x |
| score | 24 | 0.7 | 12 | **0.8** | 0.93x |
| attn | 12 | 25.2 | 12 | **2.1** | 12.28x † |
| out | 12 | 8.6 | 12 | **5.6** | 1.52x |
| **layer** | **204** | **66.3 ms** | **72** | **27.6 ms** | **2.40x** |

† **not like for like, and it is the only row that is not.** llama.cpp's
flash-attention kernel issues the whole [512, 2048] rectangle its cache defines
— ggml's own FLOP count says so — where ours walks the causal triangle of a
512-token prefill, an eighth of the work. On *rate* the two kernels are
**18.9 TFLOP/s against 12.3, 1.54x**; at our shape and its own rate the
reference would spend 3.1 ms there, which makes the layer 44.3 ms against our
27.6 and the honest whole-layer figure **1.60x**. Both numbers are real: 2.40x
is what a prefill of 2048 in four chunks actually costs either side, 1.60x is
what the kernels are worth at equal work.

**Two more things are left out of the comparison, both in the conservative
direction.** The selection — `TOP_K` and its `GET_ROWS`, 24 dispatches and
2.4 ms — is work llama.cpp does and we do not, because at any prompt shorter
than `top_k` the indexer names every cell (L2e-5); counting it would make the
reference 68.8 ms and the ratio 2.49x. And the pooling, the gate's `SIGMOID`
and `MUL`, the `CONT` behind them and the `CPY`s into the KV cache are inside
`MUL` (479), `SIGMOID` (326), `CONT` (251) and `CPY` (85), which every other
block of the model also uses; none of it is attributed here and all of it is
**absent from our graph rather than faster in it**.

## Where the dispatches went

**Six matrices become one.** The query, its gate, the key, the value and both
of the indexer's projections all read the same normalised residual, so they are
one [13952, 2560] weight and one matmul — the same argument L2a made for
`inject` and L2d for the PLE block's value, applied to the largest instance of
it in the model. llama.cpp runs five separate `MUL_MAT`s over the same
activation, three of them `q8_0` and two `bf16`.

**The norm, the rotary and the staging become one.** All three of q, k and v
are column ranges of that projection's output and all three end in the same
fragment tiling, so the grid is "one (token tile, plane)" and the plane index
says which. llama.cpp spends two `RMS_NORM_MUL`s, two `ROPE`s and the cache
`CPY`s.

**The gate is fused into the attention epilogue.** `attn_q` is [2560, 12288]:
each head's 256 query dims are immediately followed by 256 *gate* dims, and the
gate is a sigmoid on the attention **output**. The epilogue already holds the
output tile in LDS for the softmax divide, so the gate costs one read of the
projection's own fp32 row and the context leaves as fp16 straight into the
output projection's A layout — deleting a `SIGMOID`, a `MUL`, a `CONT` and a
[512, 6144] round trip through DRAM. `TestAttnGPUGateIsFused` is the control:
the ungated output is a plausible tensor of the right shape, and the test
demands the two disagree.

## Two ladders that disagree, on one cache boundary

The plain GEMM arm runs both projections, and **they want opposite schedules**
— which is the finding here that was not predicted.

Per layer, microseconds, BM 32 / 64 / 128:

| T | fused projection (B = 71.4 MB) | output projection (B = 31.5 MB) |
|---:|---|---|
| 64 | 507 / **417** / 452 | **164** / 198 / 286 |
| 128 | 914 / 542 / **453** | **198** / 207 / 286 |
| 256 | 1476 / 1051 / **793** | **265** / 302 / 318 |
| 512 | 3097 / 2005 / **1517** | 508 / **466** / 519 |
| 1024 | 8271 / 4156 / **3031** | 1023 / 967 / **875** |
| 2048 | 19201 / 9710 / **6058** | 2200 / 1838 / **1651** |

The fused projection is L2d's story unchanged: its B is **2.2x the MALL**, a
workgroup carrying BM rows reads all of it M/BM times, and the widest rung wins
everywhere above 64 tokens by up to 3.2x on identical arithmetic.

The output projection's B is **31.5 MB, just inside the 32 MiB MALL**, and on
that side of the line the reuse argument buys nothing at all — what matters is
the occupancy a narrow row block leaves. At 64 tokens BM=128 is the **worst**
rung by 2.1x. Then as M grows the activation traffic takes over and the ladder
turns back: BM=64 wins at 512 and BM=128 at 1024 and above, by 1.32x at 2048.
**Same kernel, same arithmetic, inverted schedule, decided by which side of one
cache the weight falls on.** One schedule could not have served both, and a
single `GEMMKernelFor` would have cost 1.3-1.7x on one of the two at every
length.

Against the whole 27-plan ladder at six lengths (`results/l2f_attn.csv`, 1134
rows), the three schedules together are the **exact** winner at 128, 256, 512,
1024 and 2048, and **1.010x** off it at 64 — where the whole spread across
attention rungs is 1.0% because the attention dispatch is 11 us of a 617 us
layer. The worst plan in the table is 1.78x off at 512 and 2.33x at 2048.

## The attention ladder ends one rung earlier than every other in the repo

At 512 tokens, per layer: **qt1_kt2 170 us, qt1_kt4 223, qt2_kt4 241** — 19.0,
14.4 and 13.4 TFLOP/s, monotonic in how much a wave holds live. The DiT's
ladder (§3.3) and the text encoder's both end at qt1_kt4 or wider.

The reason is the head dim. This model's is **256 against their 128**, so HDT
doubles and a wave carries 16 query fragments and 16 output accumulators across
the whole key loop before KTIL adds a score accumulator per key tile. §3.3's
conclusion holds exactly as stated — arithmetic intensity is inert, the
register file decides — it just decides one rung sooner here.

## The numbers

Against L2e's CPU reference, and against llama.cpp behind it:

| tensor | vs CPU | vs llama.cpp | |
|---|---:|---:|---|
| `attn_q` + gate, fused col 0 | 1.02e-04 | — | six column ranges of one matrix |
| `attn_k`, col 12288 | 9.41e-05 | — | |
| `attn_v`, col 12800 | 9.74e-05 | — | |
| `indexer_q_raw`, col 13312 | 1.11e-04 | — | |
| `indexer_k_raw`, col 13824 | 1.65e-04 | **1.65e-04** | BF16 weights, no quantisation either side |
| `indexer_k-3` | 3.06e-04 | **3.06e-04** | pooled through an fp16 cache, normed, rotated |
| `indexer_q-3` | 2.26e-04 | 2.26e-04 | |
| `indexer_score-3` | 5.57e-03 | 5.55e-03 | on values to 145.8 — **3.8e-07 relative** |
| `indexer_score_tokens-3` | — | mask **exact** | worst finite cell 1.7e-02 |
| q, normed rotated scaled | 3.00e-05 | — | read back out of the fragment tiling |
| k, normed rotated | 3.67e-04 | — | |
| v | 1.47e-04 | — | |
| `attn_gated-3` | **3.29e-05** | 2.45e-04 | |
| `attn_output-3` | **5.92e-05** | 1.14e-03 | |

**The kernel is 7.4x and 19.3x nearer the f32 model than the oracle is.** The
layer has three Q8_0 projections in front of it and one behind, and llama.cpp
evaluates every one over int8 activations (L2b-2); the device evaluates them in
fp16 operands with fp32 accumulators, which L2e-3 showed is the arithmetic the
backend's own matrix cores use wherever it is not quantising. So the remaining
gap to llama.cpp is the reference's, and `TestAttnGPUIsNearerTheModelThanTheReference`
says so as a checked claim rather than an inference.

**One numeric is reproduced rather than modelled.** L2e found the indexer's key
cache is fp16 and that modelling it was worth 2072x on the pooled key. Here the
cache *is* fp16, because the arena is — one `float16_t()` on the way out of the
projection's fp32 output — so the kernel does the reference's arithmetic
instead of approximating it, and `indexer_k` lands where the CPU's `RefQ8` does
rather than where its `Exact` does.

## Two controls

**`TestAttnGPUEmptyBlocksPoolCellZero`.** A block exists only if all `ratio` of
its cells are occupied, and llama.cpp leaves `blk_cells` zero-filled — so at 7
tokens in a 256-cell cache there is one real block and **63 that pool cell 0
four times**, every one byte-identical. A kernel that pooled the cells it
found, or clamped to the last real one, or skipped the empty blocks would
produce a perfectly plausible tensor and fail nothing else, because the score
those blocks feed is masked to -inf.

**`TestAttnGPUGateIsFused`.** The ungated attention output is what a kernel
that forgot the sigmoid would write, and it is the right shape and a comparable
magnitude. The test demands the two disagree, and the margin — rms 2.8e-01
against a 1e-2 floor — is the gate doing something.

## The MALL cliff, again

`pack` runs at **379 GB/s at 512 tokens and 172 at 1024**, above the DRAM bus
on the near side and below it on the far. That is L2c-4's measurement on a
different kernel: the fp32 tensor it reads is 21 MB at T=512 and 42 at T=1024
against a 32 MiB MALL. Nothing here is tuned for it, and at L6 it is the same
decision — keep the wide activations fp16 and fuse the traffic away rather than
cache it.

## What is not built

**The selection.** `top_k + ratio - 1` is 2051 against a 256-cell cache, so the
indexer names every cell and running the attention with the selection is
bit-identical to running it without (L2e-5). The score is computed and checked;
the gather is L4's, along with the 4k dump that can test it.

**Decode.** Everything here is prefill from position zero with a fresh cache.
The KV cache's ring buffer and the indexer cache's own incremental pooling are
L7. The rotary table is built per run for `nKV` positions, which is the right
shape for prefill and the wrong one for a 262144-cell context.

**The score kernel is the one line we lose on**, 0.8 ms against llama.cpp's
0.7. It is a scalar workgroup per token at 4.1 TFLOP/s where the matrix cores
do 39, and it is the obvious cooperative-matrix rewrite — but it is 0.07% of a
prefill graph, so it is written down rather than done.

## Reproducibility

Two separate runs of the same plans — the auto-schedule sweep and the winning
rows of the 27-plan ladder, in different processes — agree on the per-layer
total to **1.008x median** over the five lengths from 128 up:

| T | sweep | ladder | |
|---:|---:|---:|---:|
| 128 | 720.4 us | 714.6 | 1.008x |
| 256 | 1194.3 | 1185.1 | 1.008x |
| 512 | 2301.1 | 2280.8 | 1.009x |
| 1024 | 4997.3 | 4940.4 | 1.012x |
| 2048 | 10741.6 | 10770.3 | 1.003x |

Which is the scale to read the ladder's own margins against: a 1.01x
difference between two rungs is inside it, and the 1.32x-3.2x spreads the
schedules are cut on are not.

## How to reproduce

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go test ./llm/ -v -run TestAttnGPU          # the layer against the trace
    go run ./cmd/llm -attn -model $M            # the sweep, 64..2048
    go run ./cmd/llm -attn -model $M -tokens 512 -ladder
    go run ./cmd/llm -attn -model $M -ladder -csv results/l2f_attn.csv

The bench stages two real layers and sizes the cache at **2048 cells**, which
is what llama.cpp's measured graph had — its `MUL_MAT f32 m=512 n=2048 k=128`
is the indexer scoring 512 blocks and its `FLASH_ATTN_EXT` reads
`k(256,2048,2,1)`. Sizing ours the same way is what makes the indexer rows
comparable at all.
