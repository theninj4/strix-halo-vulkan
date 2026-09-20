<!-- LLM.md L8c, first item. The instrument L8c's gate is written against.
     Cited from cmd/llm/ppl.go, llm/graph.go and llm/graph_rows_test.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · accuracy

# L8c-0 — perplexity on our own model, and what the bank we run is worth

**Result: 4.0289 ± 0.02279 against llama.cpp's 4.0340 ± 0.02283, on the same
corpus, the same context and the same chunking — −0.13%, a fifth of one
side's standard error.** That is the number every width L8c picks has to be a
delta from, and it is the first accuracy statement in this vertical that is
about the *model* rather than about a tensor.

The instrument is `go run ./cmd/llm -ppl`, 145 chunks of 2048 tokens over
`models/wikitext-2-raw/wiki.test.raw` in 7m58s, and it needed one thing
the graph did not have: **a logit per token**. Everything here computed the
last row alone, because that is what the reference computes.

Files: `cmd/llm/ppl.go` (new), `cmd/llm/main.go`, `llm/graph.go`,
`llm/gpu.go`, `llm/gpu_head.go`, `llm/graph_rows_test.go` (new). No shader
changed, and no weight is staged differently. Results:
`results/l8c_ppl.csv`.

---

## Why this is the first thing L8c does

Every stage from L2 to L8e has been graded against `llama-eval-callback`'s
dump, tensor for tensor, and could be: **nothing any of them did changed what
the model computes.** L8a and L8b narrowed the dense bank and argued the
change was an identity (D13, ggml's `d = amax/127`); L5b, L7d, L8d and L8e
changed how a matrix is tiled and where a sum is associated, which is
round-off and not a model. The one tolerance that ever mattered was
`TestGraphLogits`' — argmax 561 out of llama.cpp's own top ten.

L8c is different. Re-quantising a weight from 8.5 bits to ~4.25 is the first
step in this vertical that makes the model a *different function*, and a
top-ten agreement on one seven-token prompt cannot grade it: L0d's own
numbers say a 4-bit weight is 16x the reconstruction error of an 8-bit one,
which is far outside anything a tensor comparison here has had to hold. The
gate LLM.md writes for L8c is "perplexity within a stated delta of 4.0340 on
the same corpus, context and chunking", and until now there was no way to
produce the left-hand side.

## The protocol, because a perplexity is only comparable to its protocol

Taken from `tools/perplexity/perplexity.cpp` at the oracle's own build,
`cff184438` — not from the working tree, which has moved (LLM.md's open
question on the pinned build):

| | llama.cpp | ours |
|---|---|---|
| corpus | `wiki.test.raw`, tokenized once | the same file, the same tokenizer |
| BOS | substituted at each chunk **only if `add_bos`** | this checkpoint's `tokenizer.ggml.add_bos_token` is **false**, so neither does it |
| chunks | `len(tokens) / n_ctx` = **145** at 2048 | the same |
| each chunk | a fresh sequence, `llama_memory_clear` | `Graph.ForwardRows` resets |
| scored | positions `[n_ctx/2, n_ctx-1)`, predicting one to the right | the same |
| per chunk | `n_ctx - 1 - n_ctx/2` = **1023** tokens | the same |
| NLL | f32 max, f64 sum of `expf(x - max)` | the same, split over 16 cores |
| PPL | `exp(sum nll / count)` | the same |

The tokenizer is not assumed to agree, it is checked: `llama-tokenize
--no-parse-special` over the whole 1.29 MB file and `-tokenize-file
-ids-only` produce **297 193 identical ids**, byte for byte through `cmp`.
That is what makes "the same corpus" a fact rather than an intention — 145
chunks is a property of that count.

## What had to be built: a logit per token

`Forward` ends the way llama.cpp's graph does. `inp_out_ids` drops every row
but the last before the final mixer, so `hidden` moves one row of the
residual to the front, resizes the hyper-connection block to one token and
runs the mixer and the head over it. A prefill therefore produces **one** row
of 248 320 logits, which is the row `TestGraphLogits` compares and the row
`-gen` samples.

`Graph.ForwardRows` is that without the move:

1. **The final mixer runs over the whole batch.** It is per token — an
   RMSNorm and a low-rank projection a row — so this is the same arithmetic
   on nTok times the work, and no kernel changed to allow it.
2. **The head runs over slabs**, because a row of logits is 0.99 MB of f32
   and 2048 of them would be a 2.03 GB arena for a tensor the caller reduces
   to one number per row. `GraphOpts.HeadRows` is the slab — 256 by default,
   255 MB — and `HCGPU.MixedRowPort(t)` is the mixer's output from row t on,
   which is the one new port this needed.
3. **Each slab is its own command buffer.** Nothing may read a device arena
   while a pass is being recorded (L7d's one rule), and the logits have to be
   read between slabs, so the pass flushes after the mixer and once per slab
   — five submits for a 2048-token chunk against the one a `-gen` prefill
   uses.

### The claim that makes it an instrument, not just a bigger read

A perplexity over the second half of a window is only meaningful if **row t
is the model's prediction after t+1 tokens** — if nothing below the head lets
a row see its right-hand neighbours. Across five block types, three carried
histories and a QSA selection that is a property of the row, that is worth
asserting rather than assuming.

`TestGraphForwardRows` asserts it directly and without a reference: four rows
of a 128-token batch against the same prompt truncated there, on a pinned
schedule. It passes as an **equality**: rows 64, 65,
121 and 127 of the batch are identical to the 65-, 66-, 122- and 128-token
prompts' logits **to the last place**, all 248 320 of them.

L7b made the neighbouring claim — a prompt in chunks is the prompt whole, to
the last place — and this is the same property read down the batch instead of
across a split, through the head.

## Finding 1: the protocol reproduces, chunk for chunk

Four chunks, both sides, running estimate after each:

| chunk | llama.cpp | ours | ours / llama.cpp |
|---:|---:|---:|---:|
| 1 | 1.3814 | 1.3903 | 1.0064 |
| 2 | 1.3404 | 1.3475 | 1.0053 |
| 3 | 1.4289 | 1.4301 | 1.0008 |
| 4 | 1.9370 | 1.9432 | 1.0032 |

The *shape* is what says the protocol is right and not just the mean: chunk 4
is a 1.36x jump on both sides, chunk 3 a smaller one on both. A chunking
error — an off-by-one in `first`, a missing reset, the wrong target token —
does not track a corpus's own difficulty.

## Finding 2: 4.0289 over the whole corpus, which is the reference's number

| | PPL | standard error | tokens |
|---|---:|---:|---:|
| llama.cpp, `cff184438` (L1) | 4.0340 | ± 0.02283 | 148 335 |
| **ours** | **4.0289** | **± 0.02279** | 148 335 |
| delta | **−0.0051** | | **−0.13%** |

**The two are the same model to well inside the corpus's own noise.** The
difference is 0.22 of one side's standard error, on 148 335 scored tokens; the
honest statement is that forty-eight layers of five block types, twenty-three
hand-written kernels, a bank that is not the reference's arrangement of the
same bits, and a QSA selection that is ours rather than llama.cpp's, together
move perplexity by less than the corpus can resolve.

**The sign is the one the numerics predict.** L3a-5 and L4a-5: the reference
takes the fp16 *accumulator* path on every quantised matmul above 8 output
columns — which is every projection in the model at a real ubatch — and a bf16
activation on the indexer's two BF16 projections (L4a-6), where ours
accumulate in f32 throughout. L4a measured that as ~5.4e-03 rms between the
reference and the f32 model, and concluded ours are *nearer the model and
further from the oracle*. This is the first evidence that nearer the model is
also nearer the text, and it is worth 0.13% of perplexity — which is to say,
almost exactly nothing, which is itself the useful part: **nobody has to
choose between the tensor comparison and the corpus.**

### And a four-chunk screen is not calibrated for a fraction of a percent

Finding 1's four chunks put us **+0.32%** above the reference. The full corpus
puts us **−0.13%** below it. The sign reverses, so that +0.32% was chunk noise
and not a small systematic bias being estimated early:

| after | ours / llama.cpp |
|---:|---:|
| 4 chunks (4 092 tokens) | 1.0032 |
| 145 chunks (148 335 tokens) | **0.9987** |

`-chunks N` stays useful as a *screen* — a width that breaks the model shows
up in the first chunk — but a delta of less than about a percent has to be
read off the whole corpus. That is a constraint on how L8c compares candidate
widths, and it is the reason to know it now rather than after choosing one.

## Finding 3: what it costs

**36.8 s of staging and 7m58s for 145 chunks — 3.30 s a chunk**, of which
**0.30 s is the host reduce** and the rest the device. Two runs of the first
four chunks reproduce exactly (1.3903 / 1.3475 / 1.4301 / 1.9432 both times),
which is what a greedy corpus pass over a fixed schedule should do and is
worth having said.

Where a chunk goes, and what is new in it:

- **2048 tokens of prefill**, the same pass `-graph` runs — except that the
  final hyper-connection mixer runs over 2048 rows instead of one, because
  there is no `inp_out_ids` move. That is the whole structural cost of an
  all-rows graph and it is one mixer, not one a layer.
- **Four head dispatches**, 256 rows each, covering the 1024 scored rows. The
  head bank is unchanged and so is the kernel; what changes is `M`.
- **1.01 GB of logits read back a chunk** — 1024 rows of 248 320 f32 — out of
  a mapped arena. This is L6b-4's memory type doing work in a third place:
  25.06 GB/s on a HOST_CACHED arena is 40 ms, and the write-combined default
  it replaced would have been **5.6 seconds**, twice the rest of the chunk.
- **Five submits a chunk** rather than the one a `-gen` prefill uses: the pass
  flushes after the mixer and once per slab, because nothing may read a device
  arena while a pass is being recorded (L7d).

The 0.30 s reduce is 1023 log-softmaxes over 248 320 logits — 254 M `expf`
calls a chunk — spread over sixteen cores. It is llama.cpp's own arithmetic
(a f32 max, a f64 sum of f32 summands) with the sum split by core rather than
serial, which moves the total by ~1e-15 relative and is the one deliberate
departure from the reference's protocol in this file.

## What this settles for the rest of L8c

The gate can now be written as a number. **4.0289 is the bank we run today** — the
checkpoint's own Q4_K/Q5_K/Q5_1/Q8_0 experts, its own Q8_0 dense
weights at 8.5 bits, its F32 routers — and the question L8c asks is what
~4.25 bits does to it. Two things follow immediately:

- **The delta to state is against 4.0289, not against 4.0340.** Ours already
  differs from the reference by −0.13% at *identical* weights, for reasons
  that are settled and are not the bank: the reference
  accumulates every quantised matmul in fp16 above 8 output columns (L3a-5,
  L4a-5) and takes a bf16 activation on the indexer's two projections, where
  ours accumulate in f32 and are therefore nearer the model and further from
  the oracle. Charging that to the re-quantisation would hide it.
- **The instrument is 145 chunks, 7m58s and 36.8 s of staging**, which is
  cheap enough to run per candidate width and not cheap enough to run per
  tensor — and finding 2's second half says a short `-chunks N` run screens
  for *broken*, not for a percent. A candidate bank gets one chunk to show it
  still speaks English and the whole corpus to be priced.
