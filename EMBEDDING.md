# EMBEDDING — the qwen3-embedding-0.6b vertical

> **Current work from 2026-09-18.** `LLM.md` (qwen3.8-flash-next) is the other
> live file; `SPEECH.md` is finished-ish and `PIPELINE.md` (z-image) is parked.
> Same rules as all three: this file is **rewritten** each session rather than
> appended to, history goes to `TODO.md` and closed findings to `research/`.
> Stage numbers are **E0, E1, …**; `§N.M` still addresses `IDEAS.md`.

**Target**: `Qwen/Qwen3-Embedding-0.6B` — text in, a 1024-dimensional unit
vector out, end to end in Go on Vulkan, served at `POST /v1/embeddings`. The
fifth and last of `GOALS.md`'s models, and the smallest: 596 M parameters,
1.19 GB of bf16.

**Status (2026-09-18, E8): the vertical is finished except for batching.** It
tokenizes, runs on the device, pools, normalises, and **reproduces the model
card's own query-by-document similarity matrix to 1.3e-4 — over HTTP**. A text
is **11.5 ms** on the device against 170 ms for the fp32 CPU reference, the
model is 0.88 GB of fp16 staged in 1.3 s, and `go run ./cmd/serve -embed`
serves it. E0–E6 and E8 are done; **E7, batching, is the one thing open**, and
the measurement below says what it is worth.

    go run ./cmd/embed            # the model card's example, on the device
    go run ./cmd/serve -embed     # POST /v1/embeddings

    the card's matrix, three ways            q0·d0   q0·d1   q1·d0   q1·d1
    the card (transformers, CUDA, fp32)     0.7646  0.1414  0.1355  0.6000
    reference/dump_qwen_embed.py (fp32 CPU) 0.7646  0.1414  0.1355  0.6000
    embed.Model   (Go, fp32, host)          0.7646  0.1414  0.1355  0.6000
    embed.GPU     (Go, fp16, Vulkan)        0.7646  0.1416  0.1355  0.6001
    POST /v1/embeddings                     0.7646  0.1416  0.1355  0.6001

    latency, one text at a time (best of 5)
    tokens    wall    GFLOP/s   weight GB/s   % of the 236 GB/s bus
        12  11.81ms      895         74.6      32%
        32  11.59ms     2431         76.0      32%
       122  13.15ms     8170         67.0      28%
       252  19.48ms    11394         45.2      19%
       382  23.57ms    14276         37.4      16%
       602  35.59ms    14901         24.8      10%

    so a text of up to ~32 tokens costs what the weights cost and nothing
    else; at 602 tokens the run has crossed to the matrix cores (14.9 of
    55.5 TFLOP/s, 27%) and the bus is no longer the story.

    one 27-token query, by dispatch (588 of them, 11.0 ms of GPU time)
      gemm down    2.52ms  22.9%     attention    167µs  1.5%
      gemm o       1.75ms  15.8%     swiglu        92µs  0.8%
      gemm up      1.45ms  13.2%     the 4 norms  271µs  2.5%
      gemm gate    1.42ms  12.9%     the 3 packs  213µs  1.9%
      gemm q       1.09ms   9.9%     rope q+k      62µs  0.6%
      gemm v        939µs   8.5%     resid, narrow 147µs 1.3%
      gemm k        918µs   8.3%
      the seven projections are 91.5% of it, which is why E7 is about them

## Why this one was cheap

It is the same architecture as a model this repository already runs. Z-Image's
text encoder is **Qwen3-4B** and lives in `zimage/qwen` — a CPU reference
(`model.go`) and a Vulkan encoder (`gpu.go`, PIPELINE.md stage 5c) doing
causal grouped-query attention, NeoX RoPE, per-head q/k norms and SwiGLU over
a fragment-tiled fp16 bank. Qwen3-Embedding-0.6B is the *same* `Qwen3Model`,
narrower and shorter, and `llm/` and `parakeet/` already import `zimage/qwen`,
so a third caller was the established shape rather than a new one.

    dims            Qwen3-4B (z-image)   Qwen3-Embedding-0.6B
    layers                  36                    28
    hidden                2560                  1024
    intermediate          9728                  3072
    heads / kv              32 / 8                16 / 8
    head_dim               128                   128     <- the kernels' constant
    vocab               151936                151669
    weights (fp16)      7.06 GB               0.88 GB

So `embed/` is not a transformer. It is **the four things an embedding model
has that a text encoder does not** — every layer runs rather than
`NumLayers-1`, the final `norm` runs, the vector is the last token's row, and
it is L2-normalised — plus a tokenizer post-processor and a kernel schedule.
The whole package is 500 lines, and the two changes it needed in `zimage/qwen`
are a tensor-name prefix (`Config.Prefix`) and two accessors (`RunIDs`,
`ReadRow`).

## Four traps, each of which would have been a silent bug

- **The tensor names have no `model.` prefix.** This checkpoint is a
  `Qwen3Model` export, so it is `layers.0.self_attn.q_proj.weight` and
  `norm.weight` where `zimage/qwen` hardcoded `model.layers.%d.`. `lm_head` is
  absent entirely (`tie_word_embeddings`), which is right: nothing here
  generates.
- **The tokenizer appends `<|endoftext|>` (151643) to every input**, via a
  `TemplateProcessing` post-processor that the usage examples never write out
  — and it is **not** the configured `eos_token_id`, which is 151645
  (`<|im_end|>`). The pooled vector is *that appended row's* hidden state, so
  an encoder that stopped at the last real token would return a
  plausible-looking vector that scores wrong. `TestTokenizer` pins it against
  the dump, both with and without `add_special_tokens`.
- **`transformers` v5 captures hidden states after the final norm.** `hs[-1]`
  *is* `last_hidden_state`, not layer 27's output; the two differ by 775 in
  absolute terms on the reference prompt. A port validated against `hs[-1]` as
  "the last layer" would be checking the wrong tensor, so the dump recomputes
  the pre-norm tensor by hand and `TestConfig` fails if the two ever coincide.
- **A query gets an instruction prefix and a document does not**
  (`Instruct: {task}\nQuery:{query}`, no space after the colon). It is part of
  the input *text*, so it lives at the API layer — the `"instruct"` extension
  in `API.md` — and not inside `Embed`.

## What fp16 does to a vector, measured

The device path was held to two bounds rather than a per-element one, because
worst-element error is a misleading statistic on this model. `TestGPULadder`
is why: the same run truncated to 1, 2, 14 and 28 layers, each against its own
dumped hidden state.

    layers   tensor       max abs   err RMS / tensor RMS   worst row cosine
         1   hidden_1      0.0007                 3.4e-4           1.000000
         2   hidden_2      0.0012                 3.8e-4           1.000000
        14   hidden_14        1.1                 1.9e-4           0.999999
        28   prenorm         1.29                 1.9e-3           0.999988

Fp16 projections leave **5.4e-2 of the tensor RMS on a single near-zero
component** of `prenorm` while the error *as a whole* is 1.9e-3 of it and
every row still points where the reference's does to 1.2e-5. The four embedded
texts come back at **cosine 0.999999-1.000000 against the fp32 reference
vectors**, which is the number that matters: an embedding is consumed as a
direction.

## The kernel schedule was the wrong one, and it was free to fix

`qwen.PlanFor` — which rung of the GEMM ladder each projection runs on — was
fitted on Qwen3-4B at hidden 2560. At hidden 1024 every projection is a
quarter the width, a tile that filled the machine there launches a quarter the
workgroups here, and the ladder moves. Swept at five lengths, **`qwen.PlanFor`
is 1.13-1.23x off the best rung at every one of them and never wins**:

    tokens   winner              vs qwen.PlanFor
        31   reg16x64_bt16_k8    1.22x
       122   reg16x64_bt16_k8    1.20x
       252   reg32x64_bt16       1.20x
       382   reg64_bt16          1.13x
       602   reg64_bt16          1.00x   (the tables agree at last)

`embed.PlanFor` is that table with the boundaries placed between the measured
points, and it costs nothing to install: every rung reads the same staged
weight, so re-planning changes which pipeline a dispatch names and nothing
else. A short text went from 14.5 ms to **11.5**.

## Where the work stands

| # | Stage | State |
|---|---|---|
| E0 | Survey: checkpoint, tokenizer, pooling, what `zimage/qwen` already gives | **done** |
| E1 | `reference/dump_qwen_embed.py` — the transformers oracle | **done** — 27 tensors, the card's matrix to 3.3e-7 |
| E2 | Tokenizer in Go, exact against the dump | **done** — 4 of 4 texts, post-processor and all |
| E3 | CPU forward: 28 layers + final norm + pool + normalise | **done** — 6.7e-5 relative through the stack |
| E4 | The model card's similarity matrix reproduced end to end | **done** — to 1e-6 on the host |
| E5 | Vulkan: the full stack on `qwen.GPUEncoder`, validated | **done** — cosine 1.000000, err/RMS 2.4e-3 |
| E6 | `cmd/embed` and a measured latency | **done** — 11.5 ms a text, 1.3 s to load |
| E7 | **Batching** — several texts per weight read | **open**, see below |
| E8 | `POST /v1/embeddings` over the existing `api` envelopes | **done** — float/base64, MRL, `instruct`, `-embed` |

## E7, the one thing open: a batch is worth up to 10x

A text of up to ~32 tokens costs **11.6 ms and 32% of the bus**, and so does a
text of 12. That is the whole finding: at one sequence the run is 0.88 GB of
weights read once and the arithmetic is nowhere near binding, so *the second
text in a batch would be nearly free*. The same weights at 602 tokens do 530
GFLOP in 35.6 ms, so there is an order of magnitude of throughput sitting in
the M axis — exactly the axis `LLM.md`'s MoE decode kernel found its 1.39-1.66x
in, for the same reason.

What stands in the way is not the GEMMs, which take M as a parameter already.
It is **attention**: the graph is one causal sequence, so two texts in one
residual stream would need a block-diagonal mask, and concatenating them
without one lets the second text attend to the first — a wrong answer rather
than a slow one. `backend.Embed` therefore runs inputs one at a time today,
with a comment saying so.

Two shapes for it, and the second is the cheap one:

- **A segmented attention kernel.** `shaders/dit_attn.comp` compiled with a
  segment table, so a key outside the query's own segment is masked. One
  `-D`, one extra buffer, and the projections need no change at all.
- **Attention per segment, projections batched.** Attention is **1.5% of the
  graph** and is already one dispatch per layer over whole planes; issuing it
  once per text with its own offsets (the push block already carries
  `InOff`/`KOff`/`VOff`/`Tokens`) leaves the other 98.5% batched. No shader
  changes whatsoever — this is the one to measure first.

Either way the arena sizing becomes a sum rather than a max, and the pooled
rows become a gather of B rows rather than one `ReadRow`.

## What is not planned, and why

**Quantisation.** The model is 0.88 GB of fp16, resident with 90 GB to spare,
and E7 above will move the binding constraint to the matrix cores rather than
the bus. `LLM.md`'s 4.5-bit machinery exists for a 180 B model; there is
nothing here for it to buy until a batch is running and the weights are the
limit again.

**Long inputs.** The checkpoint handles 32k positions and `-embed-tokens`
sizes the arenas for 512; longer inputs are truncated at the front with the
end-of-text token re-appended, which is what HF does. Raising it is a flag and
some memory, not work.

**The reranker.** `Qwen3-Reranker-0.6B` is the same checkpoint shape with a
yes/no head. Out of `GOALS.md`'s scope.
