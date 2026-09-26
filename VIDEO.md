# VIDEO — text (and keyframes) to video with sound (MiniMax-H3)

> **Live tracking and session handoff doc, opened 2026-09-26.** Stage letters
> are **M** (for MiniMax; `H` would read as the model's own name). When the
> vertical closes, this file is frozen to `research/video-vertical.md` like
> the others and `TODO.md` gets the one-line summary. Until then: tick a
> stage when its gate passes, put its measured numbers under it, and keep
> **§ Handoff** at the bottom current. A new session should be able to start
> from there.

## What we are building

`GOALS.md` item 7: video generation via
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
(read at `42ed227e`, 2026-08-13). A prompt, and optionally a first and/or
last keyframe, go in. **A 24 fps video with a 32 kHz stereo soundtrack** comes
out, generated jointly by one transformer: sound is not a separate pass. The
served door is OpenAI's asynchronous video API, which is also what SGLang
serves this model behind (`scripts/readme/*.sh` in the repo):
`POST /v1/videos` → id, `GET /v1/videos/{id}` → status,
`GET /v1/videos/{id}/content` → mp4.

### What the release is, and what it is not

MiniMax's product is three modules. **Only the middle one is open.**

| module | what it does | here |
|---|---|---|
| H3-Context-IR | hosted multi-model service that rewrites a free-form request into a long structured prompt (`integrated_multimodal_description:` / `overall_soundscape:` / `non_diegetic_music:`, shot markers, `<d>` dialogue tags) | **not released.** The card says output quality depends on it. M12 substitutes our own LLM with MiniMax's published prompt-writing guide |
| **H3-Base** | the generator: 768p short edge, 5–15 s, 24 fps, stereo | **this vertical** |
| H3-Regenerate-2K | re-generates the 768p result at 2K in context | not released ("once it is ready") |

H3-Base ships as two task checkpoints that share everything but the
transformer:

- **`transformer/`: `t2va` (text only) and `fl2va` (first and/or last
  keyframe).** This is the target.
- `transformer_ref/`: `ref2va` (up to 9 images, 3 videos and 3 audio clips as
  references). Another 66 GB, and deferred (M13).

The released transformers are **CFG-distilled**: one forward per step, no
negative prompt, no guidance scale.

### The reference implementation

diffusers carries the model as a **modular pipeline only**, and the pinned
diffusers in `.venv` (0.41.0.dev0, git `80c7ed26`, pinned for Qwen-Image)
already contains all of it:

- `models/transformers/transformer_minimax_h3.py`: the transformer (664 lines);
- `models/autoencoders/autoencoder_kl_minimax_h3.py`: the video VAE (922);
- `models/autoencoders/autoencoder_kl_minimax_h3_audio.py`: the audio VAE (679);
- `schedulers/scheduling_minimax_h3.py`: the scheduler (283);
- `modular_pipelines/minimax_h3/*`: the prompt presentation, layout, noise,
  loop and decode (4,541).

The repo's `FL2VA/` and `Ref2VA/` trees are the *original* checkpoint format
for SGLang/vLLM, with the VAE Python sources. They duplicate the diffusers
tree and are not downloaded.

## The model, from the configs and the safetensors headers

Every shape below was read from the safetensors headers by HTTP range
request, before any weight was downloaded (3,486 tensors).

### Parameters, and which of them a step touches

| component | params | stored | needed at run time |
|---|---|---|---|
| transformer | 33.12 B | bf16, fp32 heads | **19.27 B** in the 50-block stack (see below) |
| text encoder, Qwen3-VL-32B | 33.36 B | bf16 | layers 0–49 + embeddings = **25.2 B**; plus the 0.6 B vision tower for keyframes |
| video VAE | 2.60 B | fp32 | decoder **2.42 B** (a ViT); encoder 0.18 B (a CNN, keyframes only) |
| audio VAE | 0.15 B | fp32 | decoder ~0.07 B |

**13.0 B of the transformer is AdaLN projection, and the card says to
precompute it.** Each block's `adaln_proj.linear` is `[96768, 2688]`: the
timestep embedding goes to 6 modulation vectors × 3 modalities × 5376. The
input depends only on the timestep. A request's steps are known before the
first one runs, so the whole table is `steps × (≤4 distinct t) × 50 blocks ×
18 × 5376`. That is under 1 GB in fp16 for 50 steps, and computing it is one
26 GB read. **The adaln weights never go to the device.** The same holds for
`norm_out.linear` and the timestep MLP.

**The two text-refiner blocks run once a request, not once a step.** They
refine the projected text rows and take no timestep. diffusers recomputes
them in every forward. We hoist them.

### The transformer (`MiniMaxH3Transformer3DModel`)

- **One packed sequence, full self-attention, no cross-attention.** The rows
  are `[text | keyframe conditions | audio (L then R) | video]`. Every block
  attends over all of them, text included, so text rows are *not* a cacheable
  prefix: they change every step, like Qwen-Image's step-0 regime and unlike
  its cached one.
- 50 blocks, hidden 5376, **56 heads × 128 = 7168** (wider than the residual),
  SwiGLU 14336, no biases in the block. RMSNorm pre-norms (eps 1e-5) and a
  per-head q/k RMSNorm.
- **Per-row AdaLN.** Every row picks its modulation row by
  `timestep_index × 3 + modality` (0 video, 1 text, 2 audio): shift/scale for
  the norm, then a gate on the residual. Six vectors per block, per row class.
- **3-axis RoPE, on only 96 of the 128 head channels.** `rope_freq_dim 16`:
  16 frequencies are shared by t, h and w, concatenated to 48 and doubled to
  96 (rotate-half), and channels 96–127 pass through. Positions are **float64
  and non-integer**: the spatial axes are aspect-normalised onto `[0, 32)`,
  and time runs `5/3 × (1,4,4,4,4)` per latent frame, offset by the text
  length. Audio rows share the video clock at 1 unit a latent, pinned to the
  two extremes of the width grid (L left, R right).
- Input heads: video patch `1×2×2×24 = 96 → 5376`, audio `32 → 5376`, text
  `5120 → 5376` (then the refiner). Output heads run over every row, and the
  video and audio rows are selected after. The heads and the timestep MLP are
  fp32 in the checkpoint.
- **The velocity sign is reversed** against diffusers' flow-match
  convention: `x0 = x_t + σ·v`. The timesteps are `t = 1 − σ` in `[0, 1]`,
  where 1 means clean.

### The schedulers

Rectified-flow Euler, `η = 0`. There are **two schedules a request**, video
(`shift 12`) and audio (`shift 3`), stepped inside one forward. The grid is
`linspace(1, 0, N)`, shifted by `σ' = sσ/(1+(s−1)σ)` and deduplicated, and
it includes the terminal 0, so `N` steps are **N − 1 forwards**. SGLang's
default is `N = 50`. The update is `x' = r·x + (1−r)·x0` with `r = σ'/σ`,
evaluated in fp32. The σ in `x0` is recovered from the *timestep*, while `r`
comes from the grid, and the two differ in the last ulp below σ = 0.5.
Keyframe rows sit at `t = max(t_video, 0.999)`, and text rows inherit the
video timestep.

### The text encoder: our Qwen-Image encoder, four times the size

Qwen3-VL-32B: 64 layers, hidden 5120, 64 q heads over 8 kv heads, head 128,
SwiGLU 25600, interleaved mrope `[24,20,20]`, θ 5e6. H3 reads
**`hidden_states[50]`**, the *unnormalised* output of decoder layer 49. It
takes no final norm and no LM head, and layers 50–63 never run. For `t2va`
the prompt is tokenised **verbatim, with no chat template and no special
tokens**. The vision tower (depth 27, hidden 1152, deepstack 8/16/24) has the
same config as Qwen-Image-2.1's except `out_hidden_size` (5120 against 4096).

That is exactly what `qimage/textenc` (a config adapter over `zimage/qwen`)
and `qimage/vision` already run for Qwen3-VL-8B. Q0 measured there that a
text-only prompt's interleaved mrope *is* plain NeoX RoPE. **The encoder is
a reuse, not a port**. What is new is its size (50 GB fp16) and where it
lives (decision 2).

### The video VAE (`AutoencoderKLMiniMaxH3`)

f16 t4 d24: 16× spatial, 4× temporal, 24 latent channels, causal in time,
and ImageNet-normalised pixels. Frame counts are `17n + 5`, which gives
`5n + 2` latent frames. 124 frames (5.17 s) give 37. The longest request
the pipeline accepts is 345 frames (14.4 s, 102 latent frames): 362 would be
15.08 s, past the 15 s ceiling, and diffusers checks the *aligned* count.

- **The decoder is a 36-layer ViT**, not a conv stack: hidden 2048, 32 heads
  × 64, FFN ×4 (GEGLU/SwiGLU 16384 → 8192), 4 register tokens, RoPE θ 100 on
  0.75 of each head, and layer-scale vectors. It decodes 17-frame clips
  (`clip_length 17`, `token_drop 3`), and `proj_out` gives
  `3 × 4 × 16 × 16 = 3072` values a token. That is 2.42 B fp32 params, and
  attention-and-GEMM work, which is the shape this repo's kernels are good at.
  Qwen-Image's VAE refused fp16 anywhere; whether this one does is an M5
  measurement, not an assumption.
- The encoder is a conventional CNN (`down_blocks`, 0.18 B), needed only for
  `fl2va`. The keyframe latent is **sampled from the posterior with a fixed
  seed of 42 and rounded to fp16** before normalisation (`encode_vae_condition`).

### The audio VAE

DAC/BigVGAN-style. One mono codec is applied to L and R separately, at 32 kHz
and 800-sample hops, which gives **40 latents a second**, 32 channels. The
decoder upsamples `5·5·2·2·2·2·2 = 800` through snake/alias-free activations
and dilated residual stacks. That is the family of Kokoro's vocoder (T4).

## What it costs on this machine

> **Measured since (M7):** 35.6 s a forward at the served 480p (11.3 min
> for 20 steps) and 143 s at the trained 768p (1 h 57 min for 50). That is
> 25–28 TFLOP/s against the 37 assumed below, because the attention runs
> at 20–26. The table below is the planning estimate, kept for its shape.

**Compute.** One forward is `2 × 19.27 B × L` for the GEMMs plus
`4 × L² × 7168 × 50` for attention. Attention dominates past ~30k rows. The
image DiT runs both at **35–40 TFLOP/s** on the matrix cores
(research/qimage-vertical.md, Q9). At 37 TFLOP/s, with 1,000 text tokens:

| canvas | length | rows | TFLOP/forward (GEMM + attn) | s/forward | 49 forwards | 19 forwards |
|---|---|---|---|---|---|---|
| 1344×768 (trained) | 5.2 s | 38,710 | 1,490 + 2,150 | 98 | **80 min** | 31 min |
| 1344×768 | 10.1 s | 74,386 | 2,870 + 7,930 | 292 | 4.0 h | 1.5 h |
| 1344×768 | 14.4 s | 104,966 | 4,050 + 15,800 | 536 | 7.3 h | 2.8 h |
| 960×544 | 5.2 s | 20,284 | 780 + 590 | 37 | 30 min | 12 min |
| 864×480 | 5.2 s | 16,399 | 630 + 390 | 27.5 | 22 min | **8.7 min** |
| 672×384 | 5.2 s | 10,738 | 410 + 170 | 15.7 | 13 min | 5.0 min |
| 448×256 | 5.2 s | 5,558 | 210 + 40 | 7.0 | 5.7 min | 2.2 min |

For scale: SGLang's cookbook measures an RTX 5090 at 5.15 s a step at
864×480 × 124 frames × 20 steps, and 112 s a request. We should expect about
5× that. **This vertical is minutes a clip at 480p, and over an hour at the
trained 768p.** The levers, in order: canvas and step count (request
parameters); MiniMax's sparse attention ("will be released"), which is the
only thing that bends the L² term; and int8 GEMMs, which are worth nothing
where attention dominates.

**Memory**, in fp16 on the device:

| piece | GB |
|---|---|
| DiT block stack (19.27 B) + refiner + embedder | 38.5 + 1.5 + 0.06 |
| AdaLN table, 50 steps | < 1 |
| activations at 39k rows (fp32 residual, qkv, SwiGLU in/out) | ~6; ~16 at 105k |
| text encoder, 50 layers + embeddings (25.2 B) | 50.4 |
| video VAE decoder (fp32 / fp16) | 9.7 / 4.8 |
| audio VAE | 0.3 |

Everything resident is **~106 GB**, which does not fit beside anything.
Decision 2 below brings the peak to ~50 GB.

**The 4 GiB descriptor range** (memory: *A benchmark that gets faster*).
maxStorageBufferRange is silent when exceeded. SwiGLU's `[L, 28672]` fp16
input is 2.1 GiB at 39k rows, **3.97 GiB at a 10 s trained canvas** (74k:
one row-chunk from the edge) and 5.6 GiB at 14.4 s, which must be
row-chunked. Every arena buffer is asserted under the limit at planning time
(M7 gate), not discovered.

## Decisions (so future sessions don't relitigate)

1. **Target `t2va` first, then `fl2va`; `ref2va` is last and optional.** One
   transformer (`transformer/`) serves the first two. `ref2va` is another
   66 GB checkpoint and an omni-reference presentation (video and audio
   references into the Qwen3-VL conditioner at 2 fps), for a third of the
   use.
2. **The text encoder is staged per request and freed before the DiT's
   arena is allocated.** It runs once, and a request takes minutes.
   **Measured (M2): the forward is 0.4–1.0 s, but staging from bf16 is 51 s.**
   That is ~10% of a served 9-minute request, not the "few percent" first
   assumed. Before M9: stage from pre-narrowed fp16 banks on disk (one read,
   no conversion), or keep the encoder resident whenever the machine has the
   50 GB. The int8 bank (Kev K7.1's route, 25 GB resident) stays the
   fallback.
3. **AdaLN is host-precomputed per request** from the fp32 time MLP and the
   bf16 projections, which is diffusers' own arithmetic. The device gets the
   table and never the 26 GB of projections. The table is keyed by
   `(steps, shift, shift_audio)` and cached.
4. **fp16 on the device, bf16 → fp16 checked, not assumed.** The weights are
   bf16. M0 records every tensor's absmax against fp16's range, and M3/M7
   measure the activations. A block whose residual overflows fp16 keeps its
   residual in fp32, as the image DiT does.
5. **Oracles are diffusers' code on the CPU, with fp32 where it fits.** torch
   in `.venv` is CPU-only. Loaded naively, the transformer in fp32 is 132 GB
   and does not fit. With the AdaLN projections folded into the table first,
   it is 80 GB and does. The text encoder's oracle runs in bf16 and fp32 layer
   by layer.
6. **The served canvas and step count are request parameters with honest
   defaults**: 864×480 (16:9) and 20 steps (~9 min) until the profile says
   otherwise. The trained 1344×768 × 50 steps (~80 min) is allowed, and the
   response says what it costs. Duration is 5–15 s as the model requires.
7. **Video out through ffmpeg** (`/usr/bin/ffmpeg` is installed): raw RGB
   frames and float PCM piped to one H.264/AAC mp4 mux. We write no encoder
   of our own.
8. **Our own RNG, with noise injectable for gates.** torch's CPU generator is
   not reproduced. Every end-to-end gate feeds the oracle's own noise, as the
   image vertical did.

## The stages

| # | Stage | State |
|---|---|---|
| M0 | Weights, reference env, dumps (`reference/dump_h3_*.py`) | **done 2026-09-26** for what M1–M3 need: 135 GB in, every shard's size checked against the repo tree; no weight overflows fp16. Oracles for M4–M8 still to write |
| M1 | Packed layout, canvas/frame arithmetic, both schedulers, row timesteps, RoPE tables in Go (`h3/plan`) | **done 2026-09-26**: bit-exact on 7 layouts, 10 canvases, 7 frame counts, 4 schedules, 14 Euler steps; cos/sin within 1 ulp |
| M2 | Text encoder: Qwen3-VL-32B through `qimage/textenc`, `hidden_states[50]`, no template; GPU, staged per request | **done 2026-09-26**: the unchanged `zimage/qwen` GPU encoder at **rel ≤ 1.1e-4** against fp32 (official bf16: 1.2–1.4e-2); 537 tokens in 1.0 s; staging 50 GB takes 51 s (see decision 2) |
| M3 | Front, AdaLN tables, two blocks and tail on the CPU (`h3/dit`) against `dump_h3_dit_block.py` | **done 2026-09-26**: 13 stages at ≤ 8e-7 of absmax (temb 5e-6); residual absmax **3.1e4 by block 1** |
| M4 | The whole-stack oracle: torch fp32 with AdaLN folded (80 GB) at 448×256, per-block outputs of one forward and per-step latents of a short run | **done 2026-09-26** (`dump_h3_dit.py`): 5,095 rows, N = 8, ~175 s a forward on the CPU; per-block ranges answer M-o1 |
| M5 | Video VAE decoder (36-layer ViT) on the CPU, then the GPU; the fp16 question | |
| M6 | Audio VAE decoder, stereo, on the CPU, then the GPU | |
| M7 | DiT on the GPU: GEMMs and WMMA attention at 5k–105k rows, the 4 GiB plan, per-step profile | **done 2026-09-26** (`h3/dit/gpu.go`): a forward at ≤ 6.7e-3 of the fp32 oracle, teacher-forced steps ≤ 1.7e-3 rms; **35.6 s a forward at 480p, 143 s at the trained 768p** |
| M8 | End to end `t2va`: prompt → mp4 against the oracle's run at a tiny canvas; the ffmpeg mux | |
| M9 | Serve: `/v1/videos` async jobs, cancellation, **yielding the device between steps** so other verticals are not starved for minutes | |
| M10 | `fl2va`: video VAE encoder + vision tower + keyframe rows | |
| M11 | Performance: the profile's winners; sparse attention if MiniMax publishes it | |
| M12 | A Context-IR stand-in: the LLM rewrites the request under MiniMax's prompt-writing guide (`docs/VIDEO_PROMPT_WRITING_GUIDE_*.md`) | |
| M13 | `ref2va` (optional) | |

### M0 — weights, environment, oracles

**Download, diffusers layout only** (~144 GB of the repo's 498):

```sh
.venv/bin/hf download MiniMaxAI/MiniMax-H3 \
  --exclude "*.safetensors" --exclude "assets/*" --local-dir models/MiniMax-H3   # configs, code, tokenizers, docs
.venv/bin/hf download MiniMaxAI/MiniMax-H3 \
  --include "transformer/*" --include "text_encoder/*" --include "vae/*" --include "audio_vae/*" \
  --include "assets/t2va.mp4" --include "assets/fl2va.mp4" --local-dir models/MiniMax-H3
```

Not fetched: `transformer_ref/` (M13), `FL2VA/` and `Ref2VA/` (the same
weights in the original layout), and `assets/` (demo videos). The text
encoder's shards also hold layers 50–63 and the LM head, which we never load.
They are downloaded because the diffusers oracle loads the whole
`Qwen3VLForConditionalGeneration`.

**Oracles, each its own script and manifest, as `dump_qi21_*` were:**

- `dump_h3_plan.py`: canvas, frame counts, the packed layout (positions,
  tags, indices), both sigma grids, row timesteps, RoPE cos/sin, and one
  scheduler step. No weights. (M1)
- `dump_h3_textenc.py`: token ids and `hidden_states[50]` for three prompts,
  including the README's full Context-IR t2va prompt (537 tokens). (M2)
- `dump_h3_dit_block.py`: one block's input and output, with its AdaLN rows,
  at a small packed layout, fp32. (M3)
- `dump_h3_dit.py`: the stack at 448×256 × 124 frames (5,558 rows) × a few
  steps, and per-step latents. (M4)
- `dump_h3_vae.py` and `dump_h3_audio.py`: decode stagewise. (M5, M6)
- `dump_h3_run.py`: one full `t2va` run at the tiny canvas with its noise
  saved. This is the end-to-end oracle. (M8)

Gate: every oracle run twice, byte-identical. An absmax table of every
transformer tensor against fp16.

### M1 — the plan (no weights)

`h3/plan` in Go: `Canvas(aspect, short, maxPixels)`, `AlignFrames`,
`LatentFrames`, `AudioLatents`, `Layout(textTags, lf, lh, lw, audio, anchors)`
→ float64 `(t,h,w)` per row, tags, and the three index lists;
`Schedule(N, shift)` → σ grid and timesteps; `Step`; `RowTimesteps` → unique
sorted t and per-row index; RoPE cos/sin in fp32 as the model computes them
(float64 positions cast to fp32, then times the fp32 `inv_freq`).

Gate: **bit-exact** against `dump_h3_plan.py` on every case, including
t2va, first, last and first+last anchors, the 16:9/1:1/9:16/21:9 canvases,
5/10/14.4 s, and N = 2/8/20/50. The numpy pairwise-sum anchor for `last` is
reproduced as numpy computes it: the reference keeps two summation orders on
purpose.

### M0 — the fp16 audit (`dump_h3_ranges.py`, 2026-09-26)

**No weight in either component overflows fp16.** The transformer's absmax is
37 (block 49's q/k norm weights) and the text encoder's 26.9. The only
tensors with real subnormal mass are the fp32 heads (`proj_out` 17%,
`time_embedder.linear_2` 14%, `audio_proj_out` 13%), which stay fp32 because
they are tiny. In the language model, 9% of layer 1's `gate_proj` is
subnormal and 0.017% flushes, and M2's 1e-4 already prices that in. The
vision tower's first MLP flushes 0.23%: note it for M10. **So the fp16
question is about activations, not weights** (M-o1).

### M1 — result (2026-09-26)

Every gate bit-exact on the first run, and three details had to be
reproduced rather than re-derived:

- **torch's float32 `linspace` is a fused multiply-add** on a float32 step
  (`start + step·i`, one rounding). Each operation rounded separately gets
  270 of 298 grids wrong. In float64 the product and sum are exact at these
  sizes, so one narrowing reproduces it.
- **numpy's pairwise sum** for a `last` keyframe's anchor. At 102 latent
  frames it differs from the sequential sum used for the frame grid
  (`TestLastAnchorOrder` is the negative control that keeps the layout case
  meaningful).
- Canvas and audio counts use **round-half-to-even**.

cos/sin differ from torch by one ulp in ~5% of values (Sleef against Go's
correctly rounded math). That is the only non-exact table, and it sits
far below the bf16/fp16 rounding the model applies to it.

### M2 — the tokeniser, and a silent id shift it avoided

**`<d>` and `</d>` are not in `tokenizer.json`.** They and five more
(`<|cutoff|>`, `<|lyrics_*|>`, `<|caption_*|>`) are only listed in
`tokenizer_config.json`'s `additional_special_tokens`. transformers appends
the missing ones as new ids after the highest (151669…151675). Our tokeniser
read `tokenizer.json` alone, so every Context-IR dialogue line would have
tokenised as `<`, `d`, `>`: plausible text, and the wrong conditioning.
`zimage/tokenizer` now does what transformers does, and also reads the older
`"a b"` merge format this checkpoint uses. No other checkpoint in `models/`
lists a token its `tokenizer.json` lacks, so nothing else moves.

### M2 — result (2026-09-26)

`h3/textenc` is config and presentation only. `qimage/textenc`'s adapter
and `zimage/qwen`'s GPU encoder ran the 32 B model unchanged, at 50 of 64
layers:

| prompt | tokens | forward | fp16 vs fp32 oracle (rel of absmax) | official bf16 vs fp32 |
|---|---|---|---|---|
| en | 16 | 411 ms | 6.5e-5 | 1.44e-2 |
| cjk | 17 | 409 ms | 2.4e-6 | 1.35e-2 |
| README Context-IR | 537 | 1.01 s | 1.1e-4 | 1.15e-2 |

The conditioning carries a **~1.5e4 massive-activation channel** (absmax
1.5–1.7e4). That is within fp16, but only 4× from its top, and it is the
channel that cost Qwen-Image's 8 B encoder its precision (Q1: 0.024). There
it collapsed in layers 34–35. Here layer 50 reads it before any collapse, so
fp16 holds at 1e-4. The fp32 oracle comes from widening one decoder layer at
a time, so the 67 GB model never holds more than one fp32 layer (1m28s for
all three prompts on the CPU).

### M3 — result (2026-09-26)

The front (timestep MLP, projections, refiner, packing), both blocks' AdaLN
tables, blocks 0–1 and the tail (norm and both heads) all match diffusers at
**≤ 8e-7 of absmax** (temb 5e-6). Each stage is fed the oracle's own input,
so a failure names the piece. Two findings:

- **The residual stream is at 2.7e4 after block 0 and 3.1e4 after block 1**,
  half of fp16's range two blocks into fifty (M-o1). The GPU residual is
  fp32 from the start, as decision 4 allowed for.
- **The CPU oracle needed float64 accumulation.** `qwen.Linear` sums in one
  float32 accumulator. On an audio-head output of 72, built from terms summing
  to 137 in magnitude, that lands 2e-3 off, where torch's blocked fp32 lands
  4e-5 off and float64 lands exact. The oracle must be the more accurate side
  of a comparison, so `h3/dit.Linear` accumulates in float64. The GPU's GEMMs
  accumulate fp32 in 16-wide fragments, which is torch's regime and not the
  sequential one.

### M4 — the oracle, and what it says about fp16 (2026-09-26)

`dump_h3_dit.py` builds the transformer a block at a time. It applies each
block's AdaLN projection, in fp32, to every forward's timestep embedding,
then drops the projection and widens the rest to fp32. It peaks at 85 GB RSS
(2 min to load) and takes ~175 s a forward over a 5,095-row t2va (256×448 ×
124 frames on M2's real 537-token conditioning; 537 text, 414 audio and
4,144 video rows). It dumps the noise, every forward's velocities and
latents, the input and output of blocks 0/1/2/25/49 at forward 0, every
block's table at every forward, and the absmax of every GEMM operand in
every block (`block_stats` in the manifest).

**M-o1, answered:**

| tensor | peak | where | on the device |
|---|---|---|---|
| residual | **8.5e6** | a massive channel appears at block 13 (2.1e6) and 39 (8.3e6) | fp32, never narrowed |
| FFN down-projection input (value·silu(gate)) | **1.27e5** | block 39; 7.5e4 at 45 | fp16 **after ×1/16** (qimage's ffScale), 7.9e3 |
| attention output projection's output | 5.3e4 | block 44 | fp32 GEMM output |
| k before its norm | 1.7e3 | block 48 | fp32, normed before the pack |
| every other fp16 GEMM operand | ≤ 433 | | fp16 |

So the fp16 plan holds, with two conditions, and both are in `h3/dit/gpu.go`:
the residual is fp32 end to end, and SwiGLU's output is scaled before it is
narrowed. Without the scale, block 39 overflows to infinity.

### M7 — the transformer on the GPU (2026-09-26)

`h3/dit/gpu.go` runs the transformer on the image DiT's validated kernels:
the fragment-tiled fp16 GEMM at 128×256, the WMMA flash attention writing
fp16 context, the packs, SwiGLU and the gated residual. Three kernels are new
(`shaders/h3_*.comp`): the RMS pre-norm with per-run AdaLN vectors, the q/k
pack with the 96-channel rotate-half RoPE, and a bias copy. The graph's shape
comes from H3's sizes:

- **Row chunks.** A row costs ~300 KB of activations at these widths, so the
  image DiT's one fp32 arena would stop at ~14k rows. Every row-local stage
  runs over chunks (8192 served) through one scratch, and only the fp32
  residual and the fp16 q/k/v planes span the sequence. A block is two
  passes: q/k/v for every chunk, then attention onward chunk by chunk. At
  38k rows the arenas are 4.33 GB. The planes cap a staging near 90k rows.
- **Per-run AdaLN.** The packed rows fall into ≤ 4 runs sharing a
  (timestep, modality), so each modulated stage is one dispatch per run,
  reading host-folded vectors: a = norm·(1+scale), b = shift, and gate (the
  FFN's carries 1/ffScale).
- **The refiner runs as a plain block** (a = weight, b = 0, unit gates,
  identity rotary tables), once per request. The host does the timestep MLP,
  the refiner's final norm and the heads' biases, all in fp32.
- **AdaLN tables on the host** (`dit.Tables`): 13 timesteps in 28.8 s, most
  of it reading 26 GB of bf16. Matches the oracle's tables to 1e-6.

**Gates** (`TestGPUForward`, `TestGPURun`, against M4's oracle; chunk 2048,
so 5,095 rows run as three chunks):

| what | result |
|---|---|
| refiner | rel 5.4e-4 |
| blocks 0/1/2/25/49, teacher-forced | rel 2e-6 … 1.9e-4 of absmax (block 49 rms 1.1e-3) |
| forward 0, end to end | video velocity rel 6.7e-3 (rms 1.8e-3), audio 1.2e-3 |
| 7 forwards, teacher-forced | latents rms 7.9e-5 … 1.7e-3 per step, no growth |
| 7 forwards, free-running | 2.6e-4 → 0.12 rms: **the sampler's, not the port's** (below) |

**Free-running is not a precision measurement for this model.** Two runs of
the *same* device path, whose starting noise differs by 1e-4 of its own
scale, diverge to latents rms 0.096 by step 6 of N = 8
(`TestGPUSensitivity`). fp16-vs-fp32 ends at 0.121, the same size. So
`TestGPURun` gates teacher-forced, as the image vertical's Q4 did, and M8's
end-to-end gate has to be teacher-forced or perceptual, never a free-running
max-abs.

**Two bugs the gates caught:**

- **Stale keys turn rows into NaN.** The attention reads whole key blocks
  and masks the rows past its count out of the softmax weights, but not out
  of the row max. The refiner's 537-key attention, run after a 38k-row
  forward, read keys 544–575 from that forward, and every text row came back
  NaN. This only showed up on the *second* request of a process. Every
  attention length a request uses now gets its plane tail zeroed
  (`zeroKeyTail`), and `TestGPUForward` carries the regression and its
  negative control.
- **The 2-second ring watchdog** (research/p0-ring-watchdog.md). Submitting
  16 dispatches at a time put one block's attention, 2.15 s at 38k rows, in
  one submission, and the shim reported it as `VK_TIMEOUT`. Submissions now
  close on estimated device time (800 ms at a conservative 12 TFLOP/s).

**Speed** (`TestGPUShapes`, `H3_PROFILE=1`; 537 text tokens):

| shape | rows | forward | rate | attention share | 20 steps (19 fwd) | 50 steps (49 fwd) |
|---|---|---|---|---|---|---|
| 256×448 × 124 | 5,095 | 7.4 s | 31 TFLOP/s | | 2.3 min | 6.0 min |
| **864×480 × 124 (served)** | 15,936 | **35.6 s** | 27.5 | 42% | **11.3 min** | 29 min |
| 1344×768 × 124 (trained) | 38,247 | **143 s** | 24.9 | 68% | 45 min | **1 h 57 min** |

The GEMMs run at 35–37 TFLOP/s, except the down projection (K = 14336) at
22. The attention is the long pole: 26 TFLOP/s at 16k keys and 20–22 at 38k,
against 55.5 peak. The attention screen (`TestGPUAttentionScreen`): the
image DiT's QT1 KTIL4 is best at 16k keys, and **QT2 KTIL4 is +14% at 38k**,
with bit-identical output. The build now switches by key count at 24k, which
took the trained forward from 157 s to 143 s. QT4 spills (5 TFLOP/s) and
KTIL8 loses everywhere.

### Planning correction: no Go CPU stack

A Go CPU forward at the smallest canvas (5,558 rows) is ~214 TFLOP: about
an hour a forward at the CPU reference's speed. The Go CPU port therefore
stops at what M3 gates (front, tables, blocks, tail). The whole stack is
gated first on the GPU (M7), teacher-forced block by block against M4's torch
oracle. That oracle takes a few minutes a forward in fp32 once the AdaLN
projections are folded out.

## Open questions

- **M-o1. Does anything in the stack overflow fp16?** *The residual does, or
  nearly.* It reaches 3.1e4 by block 1 (M3, on stand-in text), so it is fp32
  on the device. Still open: the per-block absmax of every GEMM input and
  output over the full 50 blocks with real conditioning, which M4's oracle
  dumps and M7's fp16 operands have to fit.
- **M-o2. Does the ViT decoder tolerate fp16?** Qwen-Image's conv VAE did
  not. A ViT is a different animal, and at 9.7 GB fp32 the answer matters for
  memory.
- **M-o3. How good is the output without Context-IR?** The card is blunt that
  it matters. The oracle comparisons don't care; the product does. M12.
- **M-o4. Is the sparse attention coming?** It is the only lever on the L²
  term, which is 59% of a trained-canvas 5 s forward and 80% of a 14.4 s one.
- **M-o6. Forward 2's video velocity is 2.6% rms off** (teacher-forced),
  where the other six forwards are 0.2–0.5%. The step's latents still land
  at 1.2e-3. It is either a sensitivity of the model at t ≈ 0.03 or a
  localised fp16 loss, and a per-row/per-channel breakdown would say which.
- **M-o5. Machine placement.** At ~50 GB peak, video fits on the second
  machine beside image (~32 GB) and the small verticals. It then holds the
  device for minutes at a time, and M9's yielding is what makes that
  acceptable.

## Handoff

**2026-09-26, session 1: M0–M4 and M7 done** (M5, M6 and M8 onward open).

- Weights: `models/MiniMax-H3` (135 GB, gitignored): `transformer/`,
  `text_encoder/`, `vae/`, `audio_vae/`, plus configs, tokenizers, docs and
  the README's request scripts.
- Code: `h3/plan` (M1), `h3/textenc` (M2), `h3/dit` (M3 CPU,
  `Tables`, and the M7 GPU transformer `gpu.go`), `shaders/h3_*.comp`.
  `zimage/tokenizer` also reads `tokenizer_config.json`'s extra specials and
  the `"a b"` merge form.
- Oracles: `reference/dump_h3_{plan,tokens,textenc,dit_block,dit,ranges}.py`,
  outputs in `reference/out/h3*`. `dump_h3_dit.py` needs ~85 GB and ~23 min,
  so nothing else big can run beside it.
- Tests: `go test -short ./h3/...` is quick. Without `-short`, the M2 gate
  stages 50 GB (~1 min) and the M7 gates stage 42 GB (~1.5–2 min each). The
  opt-in instruments are `H3_SHAPES=1` (+`H3_PROFILE=1`, `H3_CHUNK`),
  `H3_SCREEN=<h>x<w>`, `H3_SENSITIVITY=1` and `H3_FREE=1`.

**Next, in order:**

1. **M5, the video VAE decoder** (the 36-layer ViT; M-o2's fp16 question), and
   **M6, the audio decoder**. They are small next to M7, both oracles are
   cheap on the CPU, and M8 needs them.
2. **M8, end to end**: a pipeline package holding the text encoder (staged
   per request), `Tables`, the GPU transformer and the decoders, plus the
   ffmpeg mux. Its gate is teacher-forced or perceptual (see M7 on free
   running). The memory plan has to sequence the stagings: text encoder
   50 GB → freed → transformer 40 GB + 4.3 GB of arenas → VAE.
3. M11 levers, measured and priced: the attention (20–26 TFLOP/s against
   55.5 peak; 68% of a trained forward), the down projection's GEMM (22
   against 37), and `Tables`' 29 s (cache it by schedule, or narrow the reads).
