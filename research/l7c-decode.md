<!-- LLM.md L7c. The generation loop: sampling, the end-of-generation set, the
     text against llama.cpp, one madvise worth 176x on the n-gram gather, and
     the attribution that says decode is two kernels shaped for prefill.
     Cited from llm/sample.go, cmd/llm/generate.go, gguf/gguf.go and
     results/l7c_decode.csv. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L7a](l7a-kv-cache.md) · [L7b](l7b-sequence.md) · [L1](l1-baseline.md) · [L5b](l5b-moe-gpu.md) · phase 1

# L7c — decode: it generates llama.cpp's text, at 7.46 tok/s

**Result: the model generates. Given `The capital of France is` at
temperature zero it produces llama.cpp's completion — the same list of
capitals, the same `<think>` block, the same `Lisbon.` — diverging at exactly
one token where the top two logits are 0.161 apart, and re-converging
immediately. It does it at 7.46 tok/s against llama.cpp's 25.15, which is 30%
of the reference and 30% of what this bank's own bytes allow.**

Prefill is unchanged by any of this and is slightly better: **950.3 tok/s at
ubatch 2048, 2.43x**, and 587.1 at 512 against llama.cpp's 313.62.

## L7c-1: the loop, and the sampler that is an argmax

`llm/sample.go` is the `top_k -> top_p -> temp -> dist` chain with temperature
zero short-circuiting into `Argmax`, and greedy is the **default** because the
gate is "the same text as llama.cpp" and that is only a check if the sampler
is a function of the logits alone. All of it is on the host, on one row, as
llama.cpp's samplers are: 248320 floats is 1 MB of arithmetic against a
token's 9.67 GB of weights, so putting it on the device would optimise 0.01%
of a step. The one thing that needed care is that a 248320-wide sort is 30 ms
— most of a decode step — so `TopK` is a quickselect and the sort happens over
what survives it.

`cmd/llm -gen` is the driver: prefill in one batch, then `Graph.Extend` with
one token at a time, printing as it goes.

## L7c-2: the stop condition is not `eos_token_id`, and the first prompt says so

This checkpoint's `tokenizer.ggml.eos_token_id` is **248046**. What the model
actually emits to finish a completion is **248044**, `<|endoftext|>` — which
the metadata calls `bos_token_id` and `padding_token_id`. A loop that trusts
the key runs past the end of the text and then repeats `<|im_start|>` until
`-n` is exhausted, which is what the first full run did.

llama.cpp marks a control token end-of-generation by **name** as well as by
metadata key — `llama_vocab` flags `<|endoftext|>`, `<|im_end|>` and
`<|eot_id|>` wherever the vocabulary has them — so `endOfGeneration` builds
the set from both. It is the kind of thing that is invisible until a model
generates and then is the first thing anyone notices.

## L7c-3: the text, and the one token it disagrees on

Both at temperature zero, 128 tokens, prompt `The capital of France is`:

```
llama.cpp  … asking me to complete the pattern for Portugal. The capital of Portugal is Lisbon.
ours       … asking me to complete the pattern. The capital of Portugal is Lisbon. This is a
             straightforward factual completion.
```

and both then close `</think>` and answer `Lisbon.`

The divergence is one token and it is priced. At that position our top two are

```
*13 "."  25.362     364 " for"  25.201
```

— **0.161 logits apart**, where the tokens either side of it are decided by 3
to 10. llama.cpp takes `" for"`. This is the drift `TestGraphLogits` measured
at **0.881% of scale on `l_last-47`**, a clean geometric x1.085 a layer
(L6b-3), arriving at a place where the model itself is indifferent: the two
continuations say the same thing and the completion re-converges within a
sentence.

So the gate reads: **greedy generation reproduces llama.cpp's text except at
positions where the argmax is a near-tie**, and not "it generates the same
text", which would be a claim about a checkpoint neither implementation
computes exactly.

## L7c-4: one `madvise` is worth 176x on the n-gram gather

D2 leaves `per_layer_token_embd` — 28.80 GB, a quarter of the checkpoint — in
the mapping, on the grounds that a token reads sixteen rows of 160 values out
of it: **1.41 KB**. That is right about bandwidth and says nothing about
latency, and at decode the gather was **23.6 ms a token, 16% of the step**.

It is not the disk. Sixteen scattered 90-byte reads are sixteen page faults,
and the kernel's default readahead answers each of them with a 128 KB window:
**2 MB of I/O to deliver 1.41 KB**, 1450x amplification. `MADV_RANDOM` on that
tensor's byte range — one syscall, on the first gather, page-aligned down to
the tensor's own pages so that the rest of the shard keeps the readahead it
wants for staging — removes it:

| | gather, ms/token | decode tok/s |
|---|---:|---:|
| 48 layers | 23.6 → **8.5** | 6.68 → **7.46** |
| 4 layers, 7 GB resident | 17.6 → **0.1** | 27.57 → **57.08** |

**176x at four layers and 2.07x on the rate.** The residual 8.5 ms at 48
layers is the part that is real: with 85.47 GB of weights resident there is
not enough page cache left for a 28.80 GB table, so each of the sixteen pages
is a genuine read. At four layers the table stays cached and the gather
disappears entirely.

Prefill gains too, though it had far less to lose — the cost is per token
either way but amortised over a batch: the 2048-token gather goes from 23.2 ms
to 17.5, and the ladder from L6c's 941.3 tok/s to **950.3**.

The general form: **`TENSOR_READ_LAZY` is a statement about bytes and
`MADV_RANDOM` is the statement about pages that has to go with it.**

## L7c-5: a command buffer a block, not a submit a dispatch

Every block's `Run` submitted each dispatch as its own command buffer with its
own fence wait — fine at prefill, where a dispatch is milliseconds, and the
whole cost at decode, where a token is ~1130 of them. One `DispatchMultiTimed`
call per block, which the shim already records as one command buffer with
memory barriers between (and which binds each pipeline's own descriptor set,
so this was never a limitation of the shim), takes it to ~490:

| | decode tok/s | ms/token |
|---|---:|---:|
| a submit a dispatch | 5.99 | 166.9 |
| a command buffer a block | 6.68 | 149.7 |
| \+ `MADV_RANDOM` | **7.46** | **134.1** |

The per-submit cost falls out of the MoE column: 432 submits a token became
48, and the block went from 48.9 to 36.8 ms — **31 µs a submit**. There are
~490 left, so ~15 ms of the remaining 134 is still hand-over. One command
buffer for the **whole token** is available — nothing in the shim prevents
mixing blocks — and it is worth about 11%, which is not where the next 4x is.

Two runs of the final configuration agree to **0.27%** (7.46 / 7.48 tok/s) and
every column of `results/l7c_decode.csv` to within 0.5%.

## L7c-6: the ceiling is 25.0 tok/s, not 38.2, because our dense half is fp16

LLM.md's decode budget — 6.334 GB a token, 38.2 tok/s at 242 GB/s — is the
checkpoint's, with the dense tensors at the Q8_0 Unsloth shipped. **Ours are
staged as halves**, and L6a already measured the consequence on capacity
(6.79 GB of fp16 against 3.67 of Q8_0, 1.85x). At decode it is a bandwidth
statement:

| | GB a token |
|---|---:|
| hyper-connection mixers | 1.31 |
| gated DeltaNet, 36 layers | 4.18 |
| full attention, 12 layers | 1.24 |
| lm head | 1.27 |
| PLE projections | 0.07 |
| **dense, read every token** | **8.07** |
| eleven experts of 512, 48 layers | ~1.60 |
| **total** | **9.67** |

9.67 GB at 242 GB/s is **40.0 ms — a ceiling of 25.0 tok/s**, which is
llama.cpp's *measured* rate. So the reference is not leaving a third of the
bus unused relative to *this* bank; it is at the ceiling of the bank it reads,
which is a smaller one.

Two things follow. **We are at 30% of our own ceiling**, not 20% of a ceiling
we cannot reach. And **D3 is worth more than it says**: "target ~4.25 bits on
everything streamed" is 1.7x against the shipped checkpoint and **3.5x against
what we currently stage**, so L8's re-quantisation is the difference between a
25.0 tok/s ceiling and a 67 tok/s one. Keeping the dense half in the
checkpoint's own Q8_0 rather than expanding it is worth 1.66x on its own and
needs no re-quantisation at all.

## L7c-7: where the 134 ms goes, and it is two kernels shaped for prefill

| block | ms/token | % | its bytes | at 242 GB/s | off by |
|---|---:|---:|---:|---:|---:|
| hyper-connection | 37.0 | 27.6% | 1.31 GB | 5.4 | **6.9x** |
| MoE | 36.9 | 27.5% | ~1.60 GB | 6.6 | **5.6x** |
| gated DeltaNet | 28.7 | 21.4% | 4.18 GB | 17.3 | 1.7x |
| full attention | 8.4 | 6.3% | 1.24 GB | 5.1 | 1.6x |
| n-gram gather (host) | 8.5 | 6.3% | 1.41 KB | — | latency |
| moves | 7.1 | 5.3% | — | — | 197 dispatches |
| lm head | 6.4 | 4.8% | 1.27 GB | 5.2 | 1.2x |
| **total** | **134.1** | | **9.67 GB** | **40.0** | **3.4x** |

The two 6x blocks have one cause each, and both are visible in their own
profilers at `-tokens 1`.

**The hyper-connection block is one dispatch, and it is a grid problem.** At
T=1 a mixer is 293.5 µs, of which the **down projection is 237.6 µs at 29
GB/s** — 8.3x off the bus — while the up projection beside it does 46.0 µs at
143 GB/s. The difference is the output width: `up` writes 10240 columns and
launches 640 workgroups; `down` writes **336** (320 low-rank plus 16 inject)
and launches about **21**, on a 40-CU device. Half the machine is idle and the
6.9 MB of weights are being pulled by 21 workgroups. This is what
`shaders/gemv_w4a8.comp` exists for (§1.1, 99-103% of the bus): at M=1 the
parallelism has to come from splitting **K**, not N, and no rung of a GEMM
ladder can supply it.

**The MoE block is a padding problem.** At T=1 its own profiler prints
`10 rows: 10 up tiles (320 rows, 32.00x)` — the permutation pads each expert's
rows to the GEMM's 32-row block (L5b-3, which is what made the epilogue need
no LDS), so **one token does 32x the arithmetic it needs**. The bank read is
the same bytes either way, which is why this shows up as 70 GB/s of bank
rather than as 32x: the kernel is unpack-bound, each workgroup unpacking a
whole BN×BK slab of Q4_K into LDS to serve one useful row. L5b-7 already named
the unpack as what is left of prefill; at decode it is 32 times more of the
answer.

The DeltaNet, the attention and the head are 1.2-1.7x off, which is a normal
distance for a kernel doing a real amount of work, and together they are 32%
of the step. **The next 4x at decode is two kernels, and neither of them is
the one 97% of the parameters are in.**

## What this leaves

L7 is complete: the model generates, in Go, on Vulkan, from its own GGUF, and
the text is llama.cpp's. What it does not do is generate *fast*, and the
attribution above is specific enough to be a task list rather than a direction
— a K-split GEMV for the hyper-connection down projection, a row block of one
for the MoE, one command buffer a token, and, underneath all of it, L8's bank,
which is worth 3.5x on its own and would also let the n-gram table back into
the page cache.
