# IMAGE — the z-image-turbo vertical

> **Formed 2026-09-18, and this is the file to read first for this vertical.**
> It supersedes [`PIPELINE.md`](PIPELINE.md), which carries a banner saying so
> and whose stage table (1-10) and validation rules remain the record of how
> the slice was built. Same rules as `LLM.md`, `SPEECH.md` and `EMBEDDING.md`:
> this file is **rewritten** each session rather than appended to, history goes
> to `TODO.md`, closed findings to `research/`. Stage numbers are **I0, I1, …**
> — the original slice's stages 1-10 keep their bare numbers in
> `research/stage-*.md` — and `§N.M` still addresses `IDEAS.md`.

**Target**: `Tongyi-MAI/Z-Image-Turbo` — prompt → PNG at up to 1024x1024, 8
steps, fp16, end to end in Go on Vulkan, served at
`POST /v1/images/generations`, with in-progress previews from
`madebyollin/taef1`.

**Status**: **every capability `GOALS.md` names for this vertical is
built and served.** The kernel work was parked 2026-09-14 with the budget
below; the serving work unparked 2026-09-18. **I1** made width and height a
*ceiling and a default rather than a fixed size* and wired
`/v1/images/generations`. **I2** ported taef1 and turned on streaming: a
preview decode is **87 ms at 1024x1024** against the full VAE's 876, a
streamed request with three partial frames costs **4.4%** over one without,
and the first picture reaches the client at **3.6 s** instead of at fifteen.
One refusal is left and no flag fixes it: `/v1/images/edits` needs a VAE
*encoder* (I7).

## Where the work stands

| # | Stage | State |
|---|---|---|
| I0 | The slice: stages 1-10, prompt → PNG in **14.26 s** | **done** — `PIPELINE.md`, `research/stage-*.md` |
| I1 | Served: variable geometry + `/v1/images/generations` | **done 2026-09-18** — `API.md` carries the row, the flags and the refusals |
| I2 | taef1 previews, and `stream: true` | **done 2026-09-18** — measured below |
| I3 | The eight elementwise passes, 1.6 s, fused into their GEMMs | open — the L2c/L2f pattern, proven since parking |
| I4 | The GEMMs' missing quarter, 2.6 s | open — two new probes since parking, no known lever |
| I5 | The VAE's elementwise fusions, ~200 ms | open |
| I6 | Attention's last 1.1x, ~0.2 s | parked — register-bound, twice measured |
| I7 | `/v1/images/edits`: the VAE encoder, and whether at all | decision first — weights are in the checkpoint, see below |
| I8 | taef1's pack pass, **38 ms of an 87 ms preview** | open, and new — see I2's findings |

The order was **capabilities before percents**, the same argument that parked
this vertical for the speech ones. With I2 done that ordering has run out of
capabilities: what is left is I7, which is a scope decision rather than work,
and four percentage items.

## The model, as the checkpoint describes it

`go run ./cmd/inspect models/Z-Image-Turbo/<part>`; full table in
`PIPELINE.md`. What the open items rest on:

| | |
|---|---|
| DiT | **34** blocks (30 `layers` + 2 context + 2 noise refiners), dim 3840, 30 heads of 128, SwiGLU 10240, adaLN `[15360, 256]` |
| Weights, fp16 | text encoder (Qwen3-4B) **7.07 GB**, transformer **12.54 GB**, VAE 0.30 GB, activations 4.90 GB — ~25 GB resident, 21.2 s to load |
| Preview decoder | `madebyollin/taef1`, 2.46 M parameters of which the decoder is **1.229 M**, 9.8 MB fp32 on disk |
| Geometry | 16-ch latents, patch 2 → both sides a multiple of **16**; 1024² is 4096 image tokens, 512² is 1024 |
| Schedule | FlowMatchEuler, shift 3.0, **8 NFE** (a turbo distillation — `steps` is a knob for looking at the trade, not tuning) |
| Weight sizes vs the 32 MiB MALL | attention projections 3840² fp16 = **29.5 MB, under it**; the three FFN matrices 78.6 MB, **over it** (I4) |

## What a request costs

`curl` wall clock including the base64 body, two runs each. The streamed
column and the 1024x1024 request beside it are this session's, on one `go run
./cmd/serve -image -previews` process; the rest of the request column is I1's
and is carried over unrepeated.

| size | image tokens | request | with 3 partial frames |
|---|---|---|---|
| 1024x1024 | 4096 | **14.72 s** / 14.80 | **15.40 s** / 15.41 |
| 1024x576 | 2304 | 7.95 s | |
| 512x512 | 1024 | **3.65 s** / 3.64 | 3.79 / 3.87 |
| 256x256 | 256 | 1.33 s | |

`stream: true` with no partials measures 14.76/14.75 — **the framing is
free**, and `partial_images` is the only field that costs anything. A `-image`
process *without* the preview decoder renders the same non-streamed image in
14.64/14.72 against this one's 14.72/14.80: 0.6% apart, which is the
run-to-run spread, so **holding taef1 costs the ordinary path nothing**
measurable beyond its 1.0 GB of arena.

The three frames land at **3.6, 7.1 and 10.7 s**, from steps 1, 3 and 5 of the
eight.

## I2 — taef1 previews, and what they cost

The capability is in three pieces and each has its own file:
`zimage/vae/tiny.go` is the CPU port and the oracle, `zimage/vae/tiny_gpu.go`
the Vulkan graph, `zimage/pipeline`'s `Step.X0`/`Step.Preview` the composition.
`reference/dump_taef1.py` is the reference, and it decided both conventions.

**The measurement**, `go run ./cmd/vaebench -tiny -sizes 16,32,64,128`:

| image | dispatches | arena | taef1 | full VAE | ratio |
|---|---|---|---|---|---|
| 1024x1024 | 114 | 872 MB + 135 fp16 | **87.4 ms** | 876 ms | 10.0x |
| 512x512 | 114 | 218 + 34 | 21.4 ms | 202 ms | 9.4x |
| 256x256 | 114 | 55 + 8.5 | 4.8 ms | 46 ms | 9.4x |
| 128x128 | 114 | 14 + 2.2 | 2.4 ms | 13 ms | 5.6x |

**The 10x is a flop ratio and not a parameter ratio, and the difference
matters.** taef1's decoder is 1.229 M parameters against the full one's 49.55
M — 40x — and an earlier draft of this file expected "low milliseconds" from
that number. The right figure is arithmetic: 566 GFLOP against 10.47 TFLOP at
1024x1024, **18.5x**, because taef1 does all of its work at full resolution in
64 channels where the big decoder has already narrowed to 128. Delivered is
10x of that 18.5, and the gap is I8.

### Four things the port found

1. **taef1 takes the raw diffusion latent**, not the `(z/scale + shift)` the
   full decoder is handed. Its config says `scaling_factor: 1.0` and the model
   card says "the same latent API as FLUX.1's VAE", and both are right — but
   neither is what settled it, because **the wrong convention still produces a
   recognisable picture.** The decoder opens with `tanh(x/3)*3`, so a latent
   2.8x too large is squashed rather than blown out. What separates them is the
   distance from the full VAE's image for the same latent: **2.0% against
   6.7%**, a 3.3x separation, on a real 256x256 trajectory.
   `TestTinyTakesTheRawLatent` is that ordering as an assertion.
2. **A preview decodes the denoised estimate, not the state the loop holds** —
   and this is the finding, because it is the one that would have been wrong.
   The schedule is flow matching, so `latents` at step k is
   `x_t = (1-σ)x0 + σε`, and with shift 3.0 the sigmas are 1.00, 0.95, 0.90,
   0.83, 0.75, 0.64, 0.50, 0.30: **four steps into eight the state is still
   75% noise**, and decoding it gives a picture of noise. The estimate is
   `x0 = x_t - σv`, which the loop forms for nothing because it has the
   velocity in hand, and it is **a recognisable fox after the first step of
   eight**. Measured as mean distance from the finished image over the run,
   x0 goes 0.29 → 0.02 and x_t goes 0.48 → 0.02.
   `TestPreviewIsNotTheRawLatent` decodes both sequences and pins the ordering.
3. **The preview's error bound is not the finished image's.** A preview
   carries σ times the model's single-step error on top of the latent error
   the final image carries alone, and σ is 0.95 at the first step. Measured
   against the reference the previews run 0.003 to **0.030** relative L2, worst
   at step 4, where the same run's finished image is 0.022. The bound is 5e-2
   and says so.
4. **The container costs more than the model.** At 1024x1024 Go's default PNG
   encoder is **344 ms** — four times the decode it is wrapping — against
   `png.BestSpeed`'s 60 for a file 15% larger, and JPEG q90's 23. Partial
   frames therefore go out at BestSpeed and the finished image does not. This
   was found by the wall clock not adding up, not by suspecting it.

### Two decisions below the graph

**The activation arena is host-cached and the full decoder's is not — and the
control says it changes nothing.** A preview's whole output crosses back to the
host, 12.6 MB at 1024x1024 and eight times per image rather than once, so
L6b's 0.18 GB/s write-combined read rate would put 70 ms in front of an 87 ms
decode. It does not, and this vertical already knew why: **stage 9** found that
0.18 GB/s is a property of the *device-local* heap, whose budget here is about
8 GB, and that a program past it gets ordinary cached memory whichever type it
asks for. A pipeline holding 25 GB is well past it.

`ZIMAGE_TINY_UNCACHED=1` is the control and it measures the null result: 87.7 /
88.2 / 87.1 ms against the cached arena's 90.9 / 87.7 / 88.3 in the full
pipeline, and 87.7 against 86.6 in `cmd/vaebench`. The cached type is kept
because it is free and because the arithmetic *would* bite in a process small
enough to stay in that heap — which the next benchmark to be written could
easily be. What is worth carrying forward is the rule's exact scope, since the
engine spent stages 3c to 8 citing the wrong version of it.

**taef1 gets a different convolution build**, `Conv64x64W32` rather than stage
8's `Conv128x64W32`. Every convolution in taef1 produces 64 output channels,
so half of a 128-row A tile is padding. Measured at 1024x1024: 64x64_w32
**86.1 ms**, 64x64 88.6, 64x128 88.5, 128x64_w32 97.2, 128x128_w32 97.8,
256x64 126.4 — 1.13x for reading the shape rather than inheriting the default.

### I8 — the pack pass, 38 ms of 87

`-tiny -profile` at 1024x1024, and the distribution is the opposite of the
full decoder's:

| | | |
|---|---|---|
| packconv | 35 x | **38.3 ms, 47.4%** |
| conv3x3 | 35 x | 21.8 ms, 27.0% — 25.9 TFLOP/s |
| relu | 31 x | 11.9 ms, 14.7% |
| add | 10 x | 6.9 ms, 8.5% |
| upsample2x | 3 x | 1.9 ms, 2.4% |

Stage 8 priced the pack at 7 ms against a 30 ms convolution and called it the
price of the layout. Here it is *larger than the convolution it feeds*, and
the reason is exact: the pack reads C·H·W floats and writes C·H·W halves
whatever the filter is, while the convolution's work scales with OC·K, and
taef1's K is 576 against the full decoder's 4608. So the same layout costs
eight times as much per unit of arithmetic.

The fix is the one stage 8 left on the table and I5 wants anyway: **every
convolution in taef1 is fed by a ReLU or an add**, so the producer can write
the blocked fp16 form directly and the pack disappears. That is 38 ms of 87
plus most of the 19 ms those two passes cost, i.e. a preview at ~35-45 ms and
a 15-20x ratio against the full decoder. It is ranked below I3 only because a
preview is already 5% of a step.

## Where the image budget stands

`go run ./cmd/zimage -width 1024`, two runs to 14.18/14.34 s. Unchanged since
stage 10; neither I1 nor I2 moved any arithmetic on the default path.

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

Per step there is also a **~50 ms remainder no dispatch accounts for** (wall
1.67-1.70 s against 1.66 s of timestamps; stage 9 ruled out submit batching).
L7d built the tool that would attribute it: a timestamp after *every*
dispatch, which is also what llama.cpp's perf logger reports.

## The hypotheses, re-checked (2026-09-18)

1. **"`aspect_ratio` is a residency question, not a parameter"** (`API.md`) —
   **disproven, constructively**: I1 made size a per-request parameter under a
   ceiling, the same arrangement `-max-audio` gives parakeet. **And the ceiling
   is each side, not the area** — the transformer's bound is its row count, so
   512x2048 is the same 4096 tokens as the square, but the VAE's fp16 arena
   holds *blocked* copies padded per axis and the same area in a taller grid
   needs 33 MB against 32. `TestSameAreaTallerShapeIsRefused` pins it.
2. **"Do not quantize the DiT for speed"** (§3.4) — **still true at every
   servable size**, and the footprint half is settled the other way; see "What
   is not planned".
3. **"The missing GEMM quarter is the top item"** (stage 10's parting ranking)
   — **still demoted.** It is the biggest number (2.6 s) but the ablation
   behind `results/gemm_wmma.csv` is 1.25x *behind* what the pipeline already
   does, while I3 and now I8 both have a proven pattern behind them.
4. **The error margin is thinner than it looks.** End to end against diffusers
   at 256² the pipeline sits at **2.2e-2 of a 3e-2 bound**, and I2 found the
   same thing one level out: a preview of an *estimate* runs to 3.0e-2 and
   needed a bound of its own rather than the image's. Anything that perturbs an
   operand — I3's fp16 epilogue stores above all — re-runs
   `reference/dump_zimage_run.py`'s composition first.
5. **One kernel wins all three DiT shapes** (stage 4) — **measured at M=4096
   only**, and I2 found the analogous assumption false one graph over: stage
   8's convolution ladder was swept on the full decoder's channel counts and
   loses 1.13x on taef1's. Variable size makes M=1152 a real DiT workload;
   `cmd/ditstack -image 1024` is one run and `qwen.PlanFor` is the precedent.

## The open stages

**I3 — the elementwise passes.** Eight passes, 5.93 ms a block, 1.6 s an
image, each reading the residual stream and writing it. `swiglu` (1.45 ms) is
L2c's gate trick verbatim; the two tanh-gated residuals (0.89, 0.85) belong in
the epilogues of `to_out` and `w2`; the adaLN scales in the norms. Not all of
1.6 s comes out — the norms still have to read the stream — but the pattern is
proven and the bandwidth arithmetic prices each fusion before it is written.
**I8 is the same fusion in the VAE's shape** and is cheaper to do first.

**I4 — the missing quarter.** Two probes that did not exist at parking: L2f
found the *same kernel* wanting opposite BM schedules on weights either side
of the 32 MiB MALL, and this DiT straddles it; and L7d's per-dispatch
timestamps would attribute the ~50 ms/step remainder, 0.4 s an image on its
own. Neither is promised more than ~1.05x; both are cheap to ask.

**I5 — the VAE's fusions.** 63% of the decode is bandwidth-bound elementwise
back to back over the same tensors; a SiLU whose only consumer is a conv can
write the blocked fp16 form instead of its fp32 output. ~200 ms, 1.4%. Its
readback is *not* an item — stage 9 measured the pipeline's at 6.5 ms and I2's
control reconfirmed the reason.

**I7 — edits.** A decision before any code: `GOALS.md` does not ask for edits
— the endpoint exists because OpenAI's surface does.

**The encoder's weights are already here**, which is worth stating because the
501 could be read as "this model has no encoder". z-image ships a stock Flux
`AutoencoderKL`: the 167 MB file `vae.LoadDecoder` already opens holds **106
encoder tensors, 34.27 M params, 68.5 MB bf16** beside the decoder's 138 and
49.55 M. Every `-image` process maps them and reads none. So I7 is a port, not
a download.

Most of that port is reuse — the encoder is the decoder's mirror and wants
conv2d (including stage 8's implicit GEMM), group norm, SiLU, the residual add,
and a mid block with attention that is the *same* 512-channel block stage 7
already put on WMMA. Genuinely new:

- **Stride-2 convolution.** `vae.Conv2D` has `Pad` and no `Stride` at all, and
  the implicit-GEMM gather assumes 1. Three downsamplers need it, with
  diffusers' **asymmetric `(0,1,0,1)` pad**.
- **The DiagonalGaussian head.** `encoder.conv_out` is `[32, 512, 3, 3]`: 16
  channels of mean and 16 of log-variance. Then `(z - shift) * scale`.
- Residency: +68.5 MB of weights and an encoder-side arena, against 25 GB.

**And the encoder is not the whole endpoint.** It yields a latent; an *edit*
needs a strategy on top. SDEdit is the cheap one and needs no new weights —
noise the encoded latent to an intermediate sigma, run the tail of the schedule
— but whether an 8-NFE turbo distillation edits acceptably that way is
unmeasured, and is exactly what the 256² composition run answers in half a
minute. **I2 makes that question cheaper to ask than it was**: `Step.X0` is
already the pipeline's estimate of where a partial trajectory is heading, and
taef1 decodes one in 5 ms at 256². OpenAI's endpoint also takes a **mask**,
which is masked blending per step and a third piece again. Tongyi's dedicated
edit checkpoint would be a new vertical and is out of scope.

## What is not planned, and why

**Quantisation.** Decided 2026-09-18: the deployment is **two of these
machines** — the language model alone on one, the other four verticals on the
other — so this pipeline's ~25 GB never has to coexist with an ~80 GB
neighbour, and footprint stops being a reason. Speed was never one: §3.4 stands
(compute-bound at every servable size), and §0 says int8 WMMA runs at fp16
rate, so bits buy arithmetic nothing here. Until that changes no weight in this
vertical leaves fp16 — **including taef1's**, which is fp32 on disk, 4.9 MB,
and not worth a thought either way.

**A preview at a smaller size than the image.** taef1's upsampling factor is
fixed at 8, so a preview is the image's size or it is a resize, and a resize is
the client's business. What a server could do instead is send fewer frames,
which `partial_images` already is.

## Harnesses and oracles

Everything from the slice still runs and is still the measurement kit:
`cmd/zimage` (`-reps`, `-unfused-layout`, `-width`, and now **`-preview`**,
which writes a frame per step and prints what each cost against the step it
interrupted), `cmd/ditstack`, `cmd/vaebench` (and now **`-tiny`**, which times
taef1 beside the full decoder at every size, with `-profile` for the dispatch
distribution), `cmd/textenc -gpu`.

`reference/dump_zimage_run.py` is the half-minute 256² composition oracle,
`dump_vae.py`/`dump_dit_block.py`/`dump_qwen.py` the stagewise ones, and
**`reference/dump_taef1.py`** is I2's: `--latent-size` dumps all nineteen
layers for the stagewise test, `--from-run` decodes the *same* trajectory
`dump_zimage_run.py` already saved and prints both convention tables.

The five validation rules are stated in full in `PIPELINE.md` and every one of
them has caught something. I2 used all five: the negative control is
`TestPreviewIsNotTheRawLatent` (the raw state decodes to a picture, so only a
control says it is the wrong one), the slow path is the CPU `TinyDecoder` the
Vulkan graph is checked against, the error denominator is per-model (a preview
got a bound of its own), the composition was validated at 256², and the prompt
is a real one.
