# IMAGE — the z-image-turbo vertical

> **ARCHIVED 2026-09-20** — frozen as the closing record of the z-image-turbo vertical (I0–I7): serving, previews, edits, the re-checked hypotheses. It was
> `IMAGE.md` at the repo root; the live state of play is now [`../TODO.md`](../TODO.md).

> **Rewritten 2026-09-18 (I7), and this is the file to read first for this
> vertical.**
> It supersedes [`zimage-pipeline.md`](zimage-pipeline.md), which carries a banner saying so
> and whose stage table (1-10) and validation rules remain the record of how
> the slice was built. Same rules as `llm-vertical.md`, `speech-vertical.md` and `embedding-vertical.md`:
> this file is **rewritten** each session rather than appended to, history goes
> to `../TODO.md`, closed findings to `research/`. Stage numbers are **I0, I1, …**
> — the original slice's stages 1-10 keep their bare numbers in
> `stage-*.md` — and `§N.M` still addresses `ideas.md`.

**Target**: `Tongyi-MAI/Z-Image-Turbo` — prompt → PNG at up to 1024x1024, 8
steps, fp16, end to end in Go on Vulkan, served at
`POST /v1/images/generations`, with in-progress previews from
`madebyollin/taef1`.

**Status**: **every capability `../GOALS.md` names for this vertical is built and
served, and `../API.md` now carries no refusal that a flag does not fix.** The
kernel work was parked 2026-09-14 with the budget below; the serving work
unparked 2026-09-18. **I1** made width and height a *ceiling and a default
rather than a fixed size* and wired `/v1/images/generations`. **I2** ported
taef1 and turned on streaming. **I7** ported the VAE's **encoder** and wired
`/v1/images/edits` over SDEdit: a 1024x1024 edit is **11.3 s** against a
generation's 14.3, and it reproduces diffusers to **1.1e-2**. What is left is
four percentage items and one capability nobody has asked for (masks).

## Where the work stands

| # | Stage | State |
|---|---|---|
| I0 | The slice: stages 1-10, prompt → PNG in **14.26 s** | **done** — `zimage-pipeline.md`, `stage-*.md` |
| I1 | Served: variable geometry + `/v1/images/generations` | **done 2026-09-18** — `../API.md` carries the row and the flags |
| I2 | taef1 previews, and `stream: true` | **done 2026-09-18** — measured below |
| I7 | `/v1/images/edits`: the VAE encoder, and SDEdit | **done 2026-09-18** — `i7-vae-encoder-and-edits.md` |
| I3 | The eight elementwise passes, 1.6 s, fused into their GEMMs | open — the L2c/L2f pattern, proven since parking |
| I4 | The GEMMs' missing quarter, 2.6 s | open — two new probes since parking, no known lever |
| I5 | The VAE decoder's elementwise fusions, ~200 ms | open |
| I6 | Attention's last 1.1x, ~0.2 s | parked — register-bound, twice measured |
| I8 | taef1's pack pass, **38 ms of an 87 ms preview** | open |
| I9 | The **encoder's** group norms, **128 ms of a 425 ms encode** | open, and new — see I7 |

The order was **capabilities before percents**, and with I7 done it has run
out of capabilities: everything left is a percentage, except masked edits,
which nothing asked for. I3, I5, I8 and I9 are now visibly *the same item in
four shapes* — a bandwidth-bound elementwise pass that its consumer could
absorb — and together they are 1.9 s of the 14.3.

## The model, as the checkpoint describes it

`go run ./cmd/inspect models/Z-Image-Turbo/<part>`; full table in
`zimage-pipeline.md`. What the open items rest on:

| | |
|---|---|
| DiT | **34** blocks (30 `layers` + 2 context + 2 noise refiners), dim 3840, 30 heads of 128, SwiGLU 10240, adaLN `[15360, 256]` |
| Weights, fp16 | text encoder (Qwen3-4B) **7.07 GB**, transformer **12.54 GB**, VAE decoder 0.30 GB, VAE **encoder 0.21 GB**, activations 4.90 GB (6.80 with `-edits`) |
| Preview decoder | `madebyollin/taef1`, 2.46 M parameters of which the decoder is **1.229 M**, 9.8 MB fp32 on disk |
| VAE encoder | 106 tensors, **34.27 M params**, 4 down blocks over `[128, 256, 512, 512]`, three stride-2 downsamplers, `conv_out` `[32, 512, 3, 3]` = 16 mean + 16 log-variance |
| Geometry | 16-ch latents, patch 2 → both sides a multiple of **16**; 1024² is 4096 image tokens, 512² is 1024 |
| Schedule | FlowMatchEuler, shift 3.0, **8 NFE** (a turbo distillation — `steps` is a knob for looking at the trade, not tuning) |
| Weight sizes vs the 32 MiB MALL | attention projections 3840² fp16 = **29.5 MB, under it**; the three FFN matrices 78.6 MB, **over it** (I4) |

## What a request costs

`curl` wall clock including the base64 body, two runs each. The generation rows
are I1's and I2's, carried over unrepeated; the edit rows are I7's, measured on
`go run ./cmd/zimage -init` at 1024² and over HTTP at 512².

| size | image tokens | generate | with 3 partial frames | edit, strength 0.8 |
|---|---|---|---|---|
| 1024x1024 | 4096 | **14.72 s** / 14.80 | **15.40 s** / 15.41 | **11.29 s** / 12.06 |
| 1024x576 | 2304 | 7.95 s | | |
| 512x512 | 1024 | **3.65 s** / 3.64 | 3.79 / 3.87 | **2.51 s** (strength 0.7, over HTTP) |
| 256x256 | 256 | 1.33 s | | |

`stream: true` with no partials measures 14.76/14.75 — **the framing is free**,
and `partial_images` is the only field that costs anything. A `-image` process
*without* the preview decoder renders the same non-streamed image in
14.64/14.72 against this one's 14.72/14.80: 0.6% apart, which is the
run-to-run spread, so **holding taef1 costs the ordinary path nothing**
measurable beyond its 1.0 GB of arena. The same is true of the encoder: it runs
only when `Init` is set.

**An edit is cheaper than a generation, and the strength is why.** At 1024²:

| strength | steps run | first sigma | wall |
|---|---|---|---|
| 1.0 | 0..7 (8) | 1.000 | 14.66 s — and bit-identical to a generation |
| 0.8 (default) | 2..7 (6) | 0.900 | **11.29 s** |
| 0.5 | 4..7 (4) | 0.750 | 7.94 s |
| 0.3 | 6..7 (2) | 0.500 | 4.63 s |

The encode is a flat 430 ms in every row — 3.6% at the default strength, 9.3%
at 0.3 — and everything else is the steps that did not run.

## I7 — the encoder, and edits

Full write-up in `i7-vae-encoder-and-edits.md`. The three things worth
carrying in this file:

**The stride-2 convolution needed no kernel.** `zimage-vertical.md` listed it as the one
genuinely new piece. It is not: a 3x3 stride-2 convolution under diffusers'
`(0, 1, 0, 1)` pad computes exactly what the **stride-1 pad-1 form computes at
the odd pixels**, far edge included, because the stride-1 form's own zero
padding supplies the pixel the asymmetric pad would have. So the encoder runs
its own filter on stage 8's unmodified kernel and throws three pixels in four
away (`shaders/vae_downsample2x.comp`, 30 lines). It costs 4x the
downsamplers' arithmetic — 927 GFLOP against 232 at 1024², about 18 ms of a
425 ms encode — and buys not touching the packed layout or the ladder.
`TestStrideTwoIsStrideOneSubsampled` asserts the identity on the real filter.

**fp16 operands: stage 8's precondition was range, and range was never the
question.** The encoder's worst convolution input is **284, 231x inside fp16** —
more headroom than the decoder's 497 — and the matrix-core path is still 0.58
out at the latent by max-over-RMS. It is not a shader bug: simulating the same
narrowing on the CPU port reproduces the GPU's number to four figures. The
encoder *contracts*, and a perturbation at `conv_in` comes out of
`conv_norm_out` amplified by **~4e5** — which diffusers' own float32 does too
(6.3e-4 from its float64 at the same tensor). So the bound had to be measured
rather than inherited, and **which bound** is the point: the same latent is
2.2e-2 in relative L2 and its round-tripped picture is 4.2e-4 of a [-1, 1]
range, a twentieth of one 8-bit level. Matrix cores are the default at **425 ms
against the fp32 graph's 1.78 s**, and the fp32 graph is kept, selectable and
used as the oracle.

**SDEdit works on an 8-NFE turbo distillation**, which was the open question.
Relative L2 to the input at 256²: a round trip with no denoising is 0.073, and
strengths 0.2/0.4/0.6/0.8/1.0 give 0.178/0.291/0.376/0.630/1.441 — monotone,
which is the only part of that worth asserting. The composition against
diffusers is **1.06e-2 on the decoded image**, *better* than a generation's
2.2e-2, because six of the eight steps of fp16 compounding never happen.

## Where the image budget stands

`go run ./cmd/zimage -width 1024`, two runs to 14.18/14.34 s. Unchanged since
stage 10; none of I1, I2 or I7 moved any arithmetic on the default path.

| | per image, 1024², 8 steps | share |
|---|---|---|
| the 34 DiT blocks | 13.3 s | 93.1% |
| — of which the seven projections | **9.9 s** at 73-76% of the WMMA ceiling | 69% |
| — of which attention | 1.9 s at 38 TFLOP/s | 13% |
| — of which eight elementwise passes | 1.6 s | 11% |
| VAE decode | 0.81 s | 5.7% |
| text encoder + host + refiners | 0.15 s | 1.0% |
| **total** | **14.26 s** | |
| *(with 8 previews)* | *+0.70 s, 4.5%* | |
| *(an edit adds a 0.43 s encode and removes the steps it skips)* | | |

Per step there is also a **~50 ms remainder no dispatch accounts for** (wall
1.67-1.70 s against 1.66 s of timestamps; stage 9 ruled out submit batching).
L7d built the tool that would attribute it: a timestamp after *every* dispatch,
which is also what llama.cpp's perf logger reports.

## The hypotheses, re-checked (2026-09-18)

1. **"`aspect_ratio` is a residency question, not a parameter"** (`../API.md`) —
   **disproven, constructively**: I1 made size a per-request parameter under a
   ceiling. **And the ceiling is each side, not the area**, because the VAE's
   fp16 arena holds *blocked* copies padded per axis.
   `TestSameAreaTallerShapeIsRefused` pins it.
2. **"Do not quantize the DiT for speed"** (§3.4) — **still true at every
   servable size**, and the footprint half is settled the other way; see "What
   is not planned".
3. **"The missing GEMM quarter is the top item"** (stage 10's parting ranking)
   — **still demoted**, and now behind four instances of one fusion pattern
   (I3, I5, I8, I9) worth 1.9 s between them.
4. **The error margin is thinner than it looks.** End to end against diffusers
   at 256² the pipeline sits at **2.2e-2 of a 3e-2 bound**. I2 found the same
   one level out; I7 found something sharper — see 6.
5. **One kernel wins all three DiT shapes** (stage 4) — **measured at M=4096
   only**, and both I2 and I7 found the analogous assumption false one graph
   over. Variable size makes M=1152 a real DiT workload; `cmd/ditstack -image
   1024` is one run and `qwen.PlanFor` is the precedent.
6. **NEW — "the activations fit fp16, so the operands narrow safely"** (stage
   8's precondition, inherited by I2). **Insufficient, and I7 is the
   counterexample.** Range is necessary and says nothing about *conditioning*,
   and the encoder passes the range test with more room than the decoder while
   being unusable in fp16 by the decoder's own metric. Any future narrowing in
   this repository owes two measurements, not one: does it overflow, and how
   much does the graph downstream multiply what it does not overflow by. The
   cheap way to ask the second is the one I7 used — run the reference in
   float64 and diff its own float32 against it, which prices the graph's
   conditioning in four lines of Python.

## What is not planned, and why

**Quantisation.** Decided 2026-09-18: the deployment is **two of these
machines** — the language model alone on one, the other four verticals on the
other — so this pipeline's ~25 GB never has to coexist with an ~80 GB
neighbour, and footprint stops being a reason. Speed was never one: §3.4 stands
(compute-bound at every servable size), and §0 says int8 WMMA runs at fp16
rate. Until that changes no weight in this vertical leaves fp16.

**A preview at a smaller size than the image.** taef1's upsampling factor is
fixed at 8, so a preview is the image's size or it is a resize, and a resize is
the client's business. What a server could do instead is send fewer frames,
which `partial_images` already is.

**Masked edits.** OpenAI's `/v1/images/edits` takes a `mask`, and this one
refuses it with a 501 that says what it would be: a blend into the latent at
every denoising step, which is a mechanism rather than a parameter. It is the
only capability this vertical is missing and `../GOALS.md` does not ask for it —
the endpoint exists because OpenAI's surface does. Tongyi's dedicated edit
checkpoint would be a new vertical and is out of scope.

**Sampling the posterior.** `Encoder.Encode` returns the DiagonalGaussian's
*mode*, never its variance. Sampling would put a different latent behind every
edit of the same picture and a seed would stop reproducing one.

## Harnesses and oracles

Everything from the slice still runs and is still the measurement kit:
`cmd/zimage` (`-reps`, `-unfused-layout`, `-width`, `-preview`, and now
**`-init`/`-strength`**, which edits a PNG and prints the encode beside the
steps it skipped), `cmd/ditstack`, `cmd/vaebench` (`-tiny`, `-profile`, and
now **`-encode`**, which times the encoder on both convolution paths beside
the decode of the same size), `cmd/textenc -gpu`.

`reference/dump_zimage_run.py` is the half-minute 256² composition oracle and
**`reference/dump_zimage_edit.py`** is I7's — the same thing for an edit, with
the encoded latent, the noised one and every step of the tail. The stagewise
ones are `dump_vae.py`, **`dump_vae_encoder.py`** (I7's, on a *real image*),
`dump_taef1.py`, `dump_dit_block.py` and `dump_qwen.py`.
**`GPUEncoder.RunTo`** is the encoder's `GPUBlock.RunTo`: it re-runs the graph's
prefix to a named submodule and reads the tensor back, which is the only way to
see an intermediate in a 122-dispatch graph over a bump-allocated arena. It is
what found that I7's error was smooth compounding rather than one wrong
dispatch.

The five validation rules are stated in full in `zimage-pipeline.md` and every one of
them has caught something. I7 used all five, and leaned hardest on the third:
its whole finding is that the decoder's denominator and the decoder's bound are
both wrong for the encoder, in opposite directions.
