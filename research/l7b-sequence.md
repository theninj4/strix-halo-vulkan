<!-- LLM.md L7b. The whole graph continuing a sequence: the two convolution
     rings, the shared DeltaNet window that was wrong, the trigram that is the
     host's, and the chunk-split gate on result_norm. Cited from llm/graph.go,
     llm/gpu_ple.go, llm/gpu_deltanet.go, shaders/llm_seq_hist.comp and
     llm/graph_cache_test.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L7a](l7a-kv-cache.md) · [L3](l3-deltanet.md) · [L2d](l2d-ple.md) · [L7c](l7c-decode.md) · phase 1

# L7b — the model continues a sequence

**Result: the graph run over a 512-token prompt in chunks of 128, of 9, or as
495 tokens and then seventeen single ones, returns `result_norm` identical to
the last place in every case. The control — the same 512 tokens with the
sequence reset before each one — is 1.785e+00 rms away.**

[L7a](l7a-kv-cache.md) made the attention layer a chunk split. This stage is
the same claim about the model, and it needs three more histories to carry,
one of which was being carried wrong in a way nothing could see.

## L7b-1: four histories, and only one of them is a KV cache

| what carries | where | how far back |
|---|---|---|
| the KV cache and the indexer's pooled blocks | the attention layer's own planes | the whole context |
| the PLE convolution's normed rows | a ring, `(kern-1)*ngram` = **9** | 9 tokens |
| every DeltaNet layer's recurrent state | `[128,128,48]` a layer, 3.1 MB | the whole sequence |
| every DeltaNet layer's convolution window | a ring, `kern-1` = **3** | 3 tokens |
| the n-gram **trigram** | the host's id list | 2 tokens |

The last one is the one a cache design does not suggest. `PLERows` hashes a
token with its two predecessors, so the first token of a continuing run reads
two tokens that are not in it — a decode step's gather is a function of three
tokens, one of which it has. So `Graph` keeps the **sequence**, not the batch:
`PLERows` runs over the whole id list and the result is sliced at `past`.
`token_embd` needs no such thing; the n-gram table is the only per-token
lookup in this model with a context.

## L7b-2: the DeltaNet's convolution window was shared by all 36 layers

The window was `Conv-1` rows of *negative token index* laid out in front of
the fused projection's own output (L3b), so a tap reaching before token zero
was an ordinary read at a wrapped offset. That arena is **shared by every
staged layer** — it is sized for one batch, not 36 — and while every pass was
a fresh sequence nothing could see it: the graph zeroed the rows once per
layer, no layer wrote them, and all 36 read zeros, which is the right answer.

The moment a run leaves its tail behind for the next one, layer 1 reads layer
0's tail. `TestGraphPrefix` caught it on the first try, and it caught it at
the right depth:

| | baseline | shared window carried |
|---|---:|---:|
| `l_last-0` rms | 2.227e-04 | 2.227e-04 |
| **`l_last-1`** | **2.473e-04** | **3.062e-03** |
| `l_last-2` | 4.767e-04 | 4.798e-03 |
| `l_last-3` | 5.418e-04 | 5.071e-03 |

Layer 0 is exact and layer 1 is 12x worse, which is what "the first layer that
has a predecessor to be contaminated by" looks like.

So the window is now **per layer**, `Conv-1` rows of `qkvN` floats each — 198
KB a layer, 7.1 MB across the block — and it is a ring rather than a prefix,
for the reason below. The old prefix arrangement was not wrong at L3b; it was
a layout whose one assumption stopped holding the day the state had to
outlive the batch, and it is worth saying which assumption: *the arena in
front of the projection's output belongs to whoever ran last*.

## L7b-3: one kernel for both rings, and why a ring rather than a window

`shaders/llm_seq_hist.comp` is 15 lines and serves both convolutions, because
the shape is the same: copy the last `hist` rows of a `[tokens][width]` tensor
into a `[hist][width]` ring. The PLE block passes `hist = 9, width = 10240`
and the DeltaNet `hist = 3, width = qkvN`; the source, the ring, the length
and the width ride push fields both blocks leave free
(`SEQ_SRC`, `SEQ_HIST`, `gemmM`, `gemmK`), the same arrangement
[L7a-4](l7a-kv-cache.md) documents.

**The ring is addressed by absolute position, and that is what makes the store
a pure write.** The rows a run must leave behind are the last `hist` positions
of the sequence; any of those that predate the run are already in the slots
they belong in, because the run that produced them put them there and nothing
has overwritten them since. So the dispatch reads the source and never the
ring, and there is no slot it both reads and writes however short the run is.

A shifted window needs the opposite. At one token — the case decode spends all
of its time in — it would read `hist - 1` rows to write them back one slot
along, and those reads and writes race across workgroups, so it would want
either a ping-pong buffer or a host round trip. The host round trip is what
`DeltaNetGPU.Carry` was, and it is deleted: 198 KB a layer read and written
through a mapped arena, 36 layers a pass, and **wrong for any run shorter than
the window**, which is every decode step. (It read `aQKV + (rows-hist)*qkvN`,
which is in front of the arena when `rows < hist`. Nothing had ever called it
with fewer than three tokens.)

A tap that reaches before position zero contributes nothing — `back > past + t`
— so a fresh sequence needs no ring cleared. `DeltaNetGPU.Reset` clears it
anyway, because it is 198 KB against a 3.1 MB state and `State` is easier to
believe when the two agree about what an empty sequence looks like.

## L7b-4: what the graph now holds

`Graph` gains `past` and `ids`, and the API splits along the reset:

```
Reset()                     a fresh sequence
Append(ids) / AppendN       run the next tokens, advance past
Extend(ids)  -> logits      Append plus the final mixer and the head
Forward(ids) -> logits      Reset plus Extend            (unchanged meaning)
Prefill(ids), Hidden(ids)   Reset plus Append / HiddenExtend  (unchanged)
```

Every block is told `past` once a run. The wide residual is **not** carried
and this is worth stating: it is a function of the token, so each batch starts
it as `hc` copies of its own embedding, and what crosses a run boundary is the
five histories above and nothing else.

`Reset` writes only the DeltaNet's state and window. The KV cache and both
rings are *masked* rather than cleared — a cell past the end of the sequence
is excluded from every score, a tap before position zero contributes nothing —
so clearing them would be work that changes no output. The one exception is
the indexer's pooled block table, which a fresh sequence rebuilds in full for
[L7a-3](l7a-kv-cache.md)'s reason.

## L7b-5: the gate, and the control that comes free

`TestGraphIsAChunkSplit`, four layers over 512 tokens of the 4k fixture — four
layers being the whole architecture, three DeltaNet layers with the PLE block
in front of layer 1, one full-attention layer with its indexer, four MoE
blocks and eight mixers, at 7 GB instead of 85:

| schedule | chunks | `result_norm` against the single pass |
|---|---:|---|
| 128 at a time | 4 | identical to the last place |
| 9 at a time, the length of the PLE ring | 57 | identical |
| 495, then one token at a time | 18 | identical |
| **control: every token a fresh sequence** | 512 | **1.785e+00 rms**, on values to 18.5 |

The one-token schedule is the control as well as the gate, and more sharply
than a separate ablation would be. A chunk of one has nothing of its own to
read: every convolution tap reaches behind it, its trigram is two tokens it
does not contain, its attention is a cache it did not write, and its recurrent
state is 495 tokens of somebody else's arithmetic. A history that was not
carried cannot produce the right answer there by accident — which is why there
is no separate test that the PLE ring is read.

## What this leaves

The model continues a sequence, so a decode loop is a loop. That is
[L7c](l7c-decode.md), and it is where the number stops being an equality and
starts being a rate.
