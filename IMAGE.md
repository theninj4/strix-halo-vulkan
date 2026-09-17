# IMAGE — the z-image-turbo vertical

> **Formed 2026-09-18, and this is now the file to read first for this
> vertical.** It supersedes [`PIPELINE.md`](PIPELINE.md), which carries a
> banner saying so and whose stage table (1-10) and validation rules remain
> the record of how the slice was built. Same rules as `LLM.md`, `SPEECH.md`
> and `EMBEDDING.md`: this file is **rewritten** each session rather than
> appended to, history goes to `TODO.md`, closed findings to `research/`.
> Stage numbers are **I0, I1, …** — the original slice's stages 1-10 keep
> their bare numbers in `research/stage-*.md` — and `§N.M` still addresses
> `IDEAS.md`.

**Target**: `Tongyi-MAI/Z-Image-Turbo` — prompt → PNG at up to 1024x1024, 8
steps, fp16, end to end in Go on Vulkan, served at
`POST /v1/images/generations`, with in-progress previews from
`madebyollin/taef1` (`GOALS.md`'s one line about this vertical that nothing
had scoped until now).

**Status**: **the model generates in 14.26 s, and it is served.** The kernel
work was parked 2026-09-14 with the budget below; the serving work unparked
2026-09-18: **I1** rebuilt the pipeline's geometry so width and height are a
*ceiling and a default rather than a fixed size* — every arena is sized at
construction, none of the three graphs is built for one geometry — and wired
`/v1/images/generations` with `size`, `aspect_ratio`, `seed`, `steps` and `n`
resolved at the HTTP layer (`api/images.go`, `backend.NewImage`, `-image`).
`go run ./cmd/serve -image` stages 24.9 GB in **20.4 s** and then serves any
size the arenas hold, out of one process: **14.55 s** at 1024², 7.95 at
1024x576, 3.65 at 512², 1.33 at 256², each reproducible to a second run.
Two refusals state their own missing pieces: `stream: true` needs the small
decoder (I2), `/v1/images/edits` needs a VAE *encoder* (I7) — the one 501 on
the server that no flag fixes.

## Where the work stands

| # | Stage | State |
|---|---|---|
| I0 | The slice: stages 1-10, prompt → PNG in **14.26 s** | **done** — `PIPELINE.md`, `research/stage-*.md` |
| I1 | Served: variable geometry + `/v1/images/generations` | **done 2026-09-18** — measured below; `API.md` carries the row, the flags and the refusals |
| I2 | taef1 previews, and `stream: true` | open — the last capability `GOALS.md` names |
| I3 | The eight elementwise passes, 1.6 s, fused into their GEMMs | open — the L2c/L2f pattern, proven since parking |
| I4 | The GEMMs' missing quarter, 2.6 s | open — two new probes since parking, no known lever |
| I5 | The VAE's elementwise fusions, ~200 ms | open |
| I6 | Attention's last 1.1x, ~0.2 s | parked — register-bound, twice measured |
| I7 | `/v1/images/edits`: the VAE encoder, and whether at all | decision first — weights are in the checkpoint, see below |

The order is deliberate: **capabilities before percents**, the same argument
that parked this vertical for the speech ones.

## The model, as the checkpoint describes it

`go run ./cmd/inspect models/Z-Image-Turbo/<part>`; full table in
`PIPELINE.md`. What the open items rest on:

| | |
|---|---|
| DiT | **34** blocks (30 `layers` + 2 context + 2 noise refiners), dim 3840, 30 heads of 128, SwiGLU 10240, adaLN `[15360, 256]` |
| Weights, fp16 | text encoder (Qwen3-4B) **7.07 GB**, transformer **12.54 GB**, VAE 0.30 GB, activations 4.90 GB — ~25 GB resident, 21.9 s to load |
| Geometry | 16-ch latents, patch 2 → both sides a multiple of **16**; 1024² is 4096 image tokens, 512² is 1024 |
| Schedule | FlowMatchEuler, shift 3.0, **8 NFE** (a turbo distillation — `steps` is a knob for looking at the trade, not tuning) |
| Weight sizes vs the 32 MiB MALL | attention projections 3840² fp16 = **29.5 MB, under it**; the three FFN matrices 78.6 MB, **over it** (I4) |

## What a request costs, by size

One `go run ./cmd/serve -image` process, `curl` wall clock including the
base64 body, two runs at each of the two sizes that have them.

| size | image tokens | unified rows | request |
|---|---|---|---|
| 1024x1024 | 4096 | 4128 | **14.55 s** / 14.62 |
| 1024x576 | 2304 | 2336 | 7.95 s |
| 512x512 | 1024 | 1056 | **3.65 s** / 3.64 |
| 256x256 | 256 | 288 | 1.33 s |

The 1024² figure is stage 10's 14.26 s plus ~0.3 s of PNG and base64, so the
serving layer costs 2% and the rest of the column is the DiT being linear in
its token count over this range. **Nothing in it is staging**: the arenas are
the 1024² ones in every row.

I1's own gates, all in `zimage/pipeline` and `api`: `TestOnePipelineManySizes`
runs four geometries through one pipeline and asserts the decoded tensor's
shape and that the picture is a picture rather than a stale arena;
`TestGeomFor`/`TestGeomForRefuses` pin the token-count bound, including the
1024x1088 case whose sides both look small enough; `TestPipelineAgainstDiffusers`
is unchanged and still lands at 2.2e-2, which is what says the default path
did not move.

## Where the image budget stands

`go run ./cmd/zimage -width 1024`, two runs to 14.18/14.34 s. Unchanged since
stage 10; I1 moved no arithmetic.

| | per image, 1024², 8 steps | share |
|---|---|---|
| the 34 DiT blocks | 13.3 s | 93.1% |
| — of which the seven projections | **9.9 s** at 73-76% of the WMMA ceiling | 69% |
| — of which attention | 1.9 s at 38 TFLOP/s | 13% |
| — of which eight elementwise passes | 1.6 s | 11% |
| VAE decode | 0.81 s | 5.7% |
| text encoder + host + refiners | 0.15 s | 1.0% |
| **total** | **14.26 s** | |

Per step there is also a **~50 ms remainder no dispatch accounts for** (wall
1.67-1.70 s against 1.66 s of timestamps; stage 9 ruled out submit batching).
L7d built the tool that would attribute it: a timestamp after *every*
dispatch, which is also what llama.cpp's perf logger reports.

## The hypotheses, re-checked (2026-09-18)

Four sessions of parking and forty of LLM/speech work later, what this
vertical believed:

1. **"`aspect_ratio` is a residency question, not a parameter"** (`API.md`) —
   **disproven, constructively**: I1 made size a per-request parameter under a
   ceiling, the same arrangement `-max-audio` gives parakeet.
   **And the ceiling is each side, not the area** — which is the one thing in
   I1 that was not obvious and had to be measured. The transformer's bound is
   its row count, so by its arithmetic 512x2048 is the same 4096 tokens as the
   square and should run. The VAE's is not: stage 8's fp16 arena holds
   *blocked* copies of each convolution's input, padded per axis, so the same
   area redistributed over a taller grid needs **33 MB against 32** and fails
   in the decoder after the whole denoising loop. Both sides inside the
   ceiling is the condition that holds — every tensor smaller elementwise, a
   per-axis round-up being monotone — and `TestSameAreaTallerShapeIsRefused`
   pins it. The `max_pixels` an earlier draft of this file reported is gone;
   `max_size` is the whole rule.
2. **"Do not quantize the DiT for speed"** (§3.4) — **still true at every
   servable size**: intensity is `4*M` flop/byte against the 235 crossover,
   and even 512² (M=1024) is 4096. And the "footprint only, someday" half is
   now settled the other way — see "What is not planned". Quantisation is off
   this vertical's list entirely.
3. **"The missing GEMM quarter is the top item"** (stage 10's parting
   ranking) — **demoted, deliberately**. It is still the biggest number
   (2.6 s), but the ablation behind `results/gemm_wmma.csv` is 1.25x *behind*
   what the pipeline already does — no known lever — while the elementwise
   fusions (I3) now have a proven pattern: L2c consumed a 21 MB SwiGLU gate
   inside the up projection's epilogue so the tensor was never written, and
   L2f fused two norms, two ropes and the tiling into one pass. Ranked by
   size the quarter wins; ranked by expected value it no longer does.
4. **The error margin is thinner than it looks.** End to end against
   diffusers at 256² the pipeline sits at **2.2e-2 of a 3e-2 bound** — stage
   10 alone ate a third of the margin, as a chaotic trajectory landing
   elsewhere rather than an accuracy loss. Anything that perturbs an operand
   — I3's fp16 epilogue stores above all — re-runs
   `reference/dump_zimage_run.py`'s composition first, and should expect to
   need the bound re-derived the way L8c had to build its own perplexity
   grade — a trajectory that lands elsewhere is not wrong, and the test that
   can say so does not exist yet.
5. **One kernel wins all three DiT shapes** (stage 4) — **measured at M=4096
   only**. Variable size makes M=1152 a real workload; still compute-bound,
   but `cmd/ditstack -image 1024` is one run and `qwen.PlanFor` is the
   precedent if the answer moves.

## The open stages

**I2 — taef1 previews.** OpenAI's streaming image endpoint sends partial
images as the denoiser goes; the pipeline already calls `Progress(Step)` after
every step and `Request.Latents`/`GenerateFrom` already exist. What is missing
is only the decoder: the full VAE costs 0.81 s against a 1.7 s step, so a
preview through it would cost half the image, and taef1 is the same 16-channel
Flux latent space at roughly a hundredth of the VAE's size — stage 8's conv
kernels should put it in the low milliseconds. The oracle is diffusers'
`AutoencoderTiny` (the established dump pattern), and the dump decides the
scale/shift convention rather than the model card. Then `stream: true` answers
with `partial_image` events and the refusal comes out.

**I3 — the elementwise passes.** Eight passes, 5.93 ms a block, 1.6 s an
image, each reading the residual stream and writing it. `swiglu` (1.45 ms) is
L2c's gate trick verbatim; the two tanh-gated residuals (0.89, 0.85) belong in
the epilogues of `to_out` and `w2`; the adaLN scales in the norms. Not all of
1.6 s comes out — the norms still have to read the stream — but the pattern
is proven and the bandwidth arithmetic (stage 10's exactness: bytes removed =
time returned) prices each fusion before it is written.

**I4 — the missing quarter.** Two probes that did not exist at parking: L2f
found the *same kernel* wanting opposite BM schedules on weights either side
of the 32 MiB MALL, and this DiT straddles it (table above) while running one
schedule for everything; and L7d's per-dispatch timestamps would finally
attribute the ~50 ms/step remainder, which is 0.4 s an image on its own.
Neither is promised more than ~1.05x; both are cheap to ask.

**I5 — the VAE's fusions.** 63% of the decode is bandwidth-bound elementwise
back to back over the same tensors; a SiLU whose only consumer is a conv can
write the blocked fp16 form instead of its fp32 output. ~200 ms, 1.4%.

**I7 — edits.** A decision before any code: `GOALS.md` does not ask for
edits — the endpoint exists because OpenAI's surface does.

**The encoder's weights are already here**, which is worth stating because
the 501 could be read as "this model has no encoder". z-image ships a stock
Flux `AutoencoderKL`: `config.json` declares four `DownEncoderBlock2D`s, and
the 167 MB file `vae.LoadDecoder` already opens holds **106 encoder tensors,
34.27 M params, 68.5 MB bf16** beside the decoder's 138 and 49.55 M. Every
`-image` process maps them and reads none. So I7 is a port, not a download.

Most of that port is reuse — the encoder is the decoder's mirror and wants
conv2d (including stage 8's implicit GEMM), group norm, SiLU, the residual
add, and a mid block with attention that is the *same* 512-channel block
stage 7 already put on WMMA. Genuinely new:

- **Stride-2 convolution.** `vae.Conv2D` has `Pad` and no `Stride` at all,
  and the implicit-GEMM gather assumes 1. Three downsamplers need it, with
  diffusers' **asymmetric `(0,1,0,1)` pad** — `Downsample2D` pads that way
  whenever `padding == 0`, which is what `Encoder` passes.
- **The DiagonalGaussian head.** `encoder.conv_out` is `[32, 512, 3, 3]`:
  16 channels of mean and 16 of log-variance. Then `(z - shift) * scale`,
  the inverse of what the pipeline already does before decoding.
- Residency: +68.5 MB of weights and an encoder-side arena, against 25 GB.

**And the encoder is not the whole endpoint.** It yields a latent; an *edit*
needs a strategy on top. SDEdit is the cheap one and needs no new weights —
noise the encoded latent to an intermediate sigma, run the tail of the
schedule — but whether an 8-NFE turbo distillation edits acceptably that way
is unmeasured, and is exactly what the 256² composition run answers in half a
minute. OpenAI's endpoint also takes a **mask**, which is masked blending per
step and a third piece again. Tongyi's dedicated edit checkpoint would be a
new vertical and is out of scope.

## What is not planned, and why

**Quantisation.** Decided 2026-09-18: the deployment is **two of these
machines** — the language model alone on one, the other four verticals on the
other — so this pipeline's ~25 GB never has to coexist with an ~80 GB
neighbour, and footprint stops being a reason. Speed was never one: §3.4
stands (compute-bound at every servable size), and §0 says int8 WMMA runs at
fp16 rate, so bits buy arithmetic nothing here. `LLM.md`'s L8 toolchain
exists if this ever changes; until then no weight in this vertical leaves
fp16.

## Harnesses and oracles

Everything from the slice still runs and is still the measurement kit:
`cmd/zimage` (`-reps`, `-unfused-layout`, `-width`), `cmd/ditstack`,
`cmd/vaebench`, `cmd/textenc -gpu`; `reference/dump_zimage_run.py` is the
half-minute 256² composition oracle, `dump_vae.py`/`dump_dit_block.py`/
`dump_qwen.py` the stagewise ones. The five validation rules — negative
controls, keep the slow path, choose the error denominator per model,
validate the composition small, use a real prompt — are stated in full in
`PIPELINE.md` and every one of them has caught something; I2 and I3 will
want all five.
