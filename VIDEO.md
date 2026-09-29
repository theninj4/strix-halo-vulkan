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
| M5 | Video VAE decoder (36-layer ViT) on the CPU, then the GPU; the fp16 question | **done 2026-09-26** (`h3/vae`): the decode at **PSNR 81.7 dB** against fp32 (≤ 1.3 8-bit levels); **38 s at 480p, 72 s at 768p**; M-o2 answered, fp16 is fine |
| M6 | Audio VAE decoder, stereo, on the CPU, then the GPU | **done 2026-09-26** (`h3/audiovae`, CPU fp32): every stage ≤ 2.8e-6 of the oracle, the stereo decode at **SNR 106 dB**; 7.2 s for 5.2 s of audio (the GPU is an M11 lever) |
| M7 | DiT on the GPU: GEMMs and WMMA attention at 5k–105k rows, the 4 GiB plan, per-step profile | **done 2026-09-26** (`h3/dit/gpu.go`): a forward at ≤ 6.7e-3 of the fp32 oracle, teacher-forced steps ≤ 1.7e-3 rms; **35.6 s a forward at 480p, 143 s at the trained 768p** |
| M8 | End to end `t2va`: prompt → mp4 against the oracle's run at a tiny canvas; the ffmpeg mux | **done 2026-09-26** (`h3/pipeline`, `cmd/h3`): the README prompt to mp4 in 3 m 28 s at 448×256 × N = 8; frames **PSNR 25.9 dB** / soundtrack SNR 21.5 dB against the oracle's own free-running run |
| M9 | Serve: `/v1/videos` async jobs, cancellation, **yielding the device between steps** so other verticals are not starved for minutes | **done 2026-09-26** (`api/videos.go`, `backend/video.go`, `-video`): the README prompt served in 3 m 19 s at 448×256 × N = 8 (estimate 206 s); speech beside it **≤ 0.27 s** through every forward, against 17 ms idle; DELETE stops a running job within one submission |
| M10 | `fl2va`: video VAE encoder + vision tower + keyframe rows | **done 2026-09-26** (`h3/vae/encoder.go`, `h3/textenc` Presentation, `h3/pipeline/keyframe.go`, `-first`/`-last`, `input_reference` / `conditions`): encoder moments at ≤ 1.6e-5 of fp32, torch's seed-42 draw reproduced; conditioner 9.3e-4 teacher-forced (bf16: 1.2e-2); teacher-forced steps ≤ 1.3e-4 rms; frame 0 of a served run lands on the keyframe at **27.1 dB** |
| M11 | Performance: the profile's winners; sparse attention if MiniMax publishes it | **M11a done 2026-09-26**: int8 banks for the text encoder and the transformer (`qwen.BankQ8`, served default): 50 → 26.9 GB and 42 → 22.9 GB staged, measured request peak 57 → 32.6 GB at 480p, no speed cost, every teacher-forced step inside the released bf16 pipeline's error. **M11b done 2026-09-29**: the int8 banks and AdaLN tables cached beside the checkpoint (`qwen.Q8Cache`, 47 GB of disk), staging ~94 s → ~12 s of a request, output byte-identical. **M11c done 2026-09-29**: the attention transposed (`h3_attn_t.comp`), so P reaches the matrix cores without LDS: 1.55–1.61x on the kernel, a forward 35.6 → 30.6 s at 480p and 144 → 111 s at 768p, gated teacher-forced. **M11d**: the down projection in two K passes, bit-identical, 1.02x a 480p forward; its swizzle band and row pad screened and already optimal. The rest is open |
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

### M5 — the video VAE decoder (2026-09-26)

`h3/vae` decodes on the device, and the host does what diffusers does
around the decoder: unpatchify and denormalise, post_quant_conv (a 1x1
conv, float64), the 256-pixel tiles with their widened overlaps
(`_split_tiles`), the 7-latent temporal clips, and the linear cross-fades
that put them back together (`_stitch_tiles`, `_decode`'s loop).
diffusers' order is reproduced as written: a tile blends with the
*unblended* tile above it and then the unblended tile to its left. The
oracle is `reference/dump_h3_vae.py`, diffusers' own decode in fp32 on M4's
final latents (the real 8-step sample). It takes 10.8 s a tile-clip on the
CPU, and its repeat run is byte-identical.

**The shape decides the graph.** A decoder call is one tile-clip:
7 × 16 × 16 latents, 4 registers and a zero token, 1,797 tokens of full
attention. A 5 s clip is 105 of them at 480p (7 clips × 15 tiles) and 196
at 768p (7 × 28), all the same length and all independent. So they run
**batched**, S = 8 sequences at a 1,808-row stride. Every row-local stage is
one dispatch over all the rows, and only the attention runs a sequence at a
time. The kernels are M7's, three rebuilt at head 64 (`shaders.H3VAE*`).
The biases ride in the GEMMs as a ones column (the transformer has none),
and proj_in's register tokens are one-hot columns, so the input stage is one
GEMM into the residual.

**M-o2, answered: fp16 is fine.** The oracle's per-block absmax
(`block_stats`) peaks at 832 (the FFN's down projection output, block 30;
its input is 354). The residual stays under 22, because the layer-scale
vectors keep every update small. This is not Qwen-Image's conv VAE. The
pipeline itself decodes under fp16 autocast, so fp16 is the released recipe
and not a compromise.

| gate (`TestGPUDecoder`, `TestGPUDecodeFull`) | result |
|---|---|
| unpatchify + denormalise vs the oracle's z | bit-exact |
| post_quant_conv | rel 3e-7 |
| blocks 0/1/18/35, teacher-forced | rel 1.7e-5 … 1.4e-4 |
| one tile-clip, whole decoder | rms 2.2e-4; ≤ 1.28 levels, PSNR 81.7 dB |
| 12 latent frames (2 clips × 2 tiles: both blends) | rms 2.2e-4; ≤ 1.12 levels, PSNR 81.7 dB |
| all 37 latent frames, 124 frames | rms 2.3e-4; ≤ 1.08 levels, PSNR 81.7 dB, 5.1 s |

**Speed** (`H3_VAE_SHAPES=1`, S = 8): 105 tile-clips in **38 s at 480p**
and 196 in **72 s at 768p**, both at 26.4 TFLOP/s. That is ~6% of a served
20-step request and ~1% of a trained one. The arenas are 1.9 GB at S = 8,
and the weights 4.9 GB fp16.

The frames are coherent: M4's 8-step sample of the README prompt is a
starship bridge that cuts to a close-up of a commander. That is the first
video out of this vertical.

### M6 — the audio VAE decoder (2026-09-26)

`h3/audiovae` is the BigVGAN in Go, **fp32 on the CPU**. A 5 s stereo clip
is ~480 GFLOP of 1-D convolutions (about 1% of a served request's device
time), and diffusers pins this model to fp32 on purpose: bf16 decodes come
out ~20 dB quieter. So it is the one piece that needed neither the device
nor fp16 to start with. Activations are channel-last, and every conv is
4 × 4 register-blocked dot products over contiguous channels. The oracle is
`reference/dump_h3_audio.py`: diffusers' decode of M4's final audio rows,
dumped stage by stage (its staged run equals `vae.decode` bit for bit).

| gate (`TestStages`, `TestDecode`) | result |
|---|---|
| dec_in_proj + conv_pre | rel 1.1e-6 |
| one alias-free SnakeBeta alone (the replicate-padded ×2 resampling) | rel 2.6e-7 |
| stages 0–6, teacher-forced | rel 3.6e-7 … 2.8e-6 |
| tail (activation, conv_post, clamp) | rel 2.8e-7 |
| the whole stereo decode from the transformer's rows | rms 6.6e-6 / 1.4e-6, **SNR 106 dB** |

**7.2 s for 5.17 s of stereo** on 32 threads. torch does it in 1.1 s
(oneDNN, ~440 GFLOP/s), so this is where the Go port is weakest: stage 0 runs
at ~80 GFLOP/s. It can overlap the video decode, which is on the device, and
M11 decides whether it moves there. The sample's soundtrack is plausible:
−25.8 dB mean, −9.8 dB peak, steady harmonic lines under broadband ambience
that swells from ~2 s. It is muxed with M5's frames in the scratch
`av.mp4` (not kept).

### M8 — end to end (2026-09-26)

`h3/pipeline` runs a t2va request from the prompt string, and `cmd/h3` is
its CLI (`go run ./cmd/h3 -prompt … -out clip.mp4`). Its own job is the
order in which the pieces hold the machine. Everything at once is ~106 GB,
so a request is three stagings, each freed before the next is allocated:
the text encoder (50 GB, one forward), the transformer (44 GB), then the
video VAE (7 GB). The AdaLN tables are computed on the host while the
encoder stages, and the audio decode runs on the CPU beside the video
decode. `Options.Resident` keeps the transformer and VAE between requests
(51 GB held) for M9. The noise is our own PCG (video, then audio), and it
is injectable. The mux pipes rgb24 frames to ffmpeg's stdin, with the
soundtrack as a float32 file (H.264 crf 18 + AAC 192k, faststart).

**The gate** (`TestE2E`) starts from the README's Context-IR prompt as a
string, at M4's oracle shape (448×256, 124 frames, N = 8) and from M4's own
noise. It runs free, so the latents are not held to a tolerance: they end
0.115 rms (video) and 0.031 (audio) from the oracle's, the size M7 measured
for two runs whose noise differs by 1e-4. What is gated is the output
against the oracle's decode of its own run: **frames PSNR 25.9 dB**,
**soundtrack SNR 21.5 dB**, and an mp4 whose streams ffprobe reads back
(h264 448×256 × 124 frames, aac 32 kHz stereo). Side by side, the two
clips are the same scene, the same shots and the same composition. The
visible difference is a collar that is red in one and navy in the other.

**Where the 3 m 28 s went:**

| stage | wall | note |
|---|---|---|
| encode (stage 50 GB + forward), tables beside it | 81.6 s (tables 48.6 s) | both reading bf16 from disk at ~0.9 GB/s |
| transformer staging + Begin | 52.8 s | 40 GB narrowed from bf16 |
| 7 forwards | 55.9 s | 7.98 s each, as M7 |
| both decoders | 17.2 s | VAE staging 7 s + decode 5 s; audio 7 s beside it |

At the served 480p × 20 steps, the forwards are 11.3 min and the fixed
costs above are ~2.5 min: **~14 min a request**, ~18% of it staging. The
cure is decision 2's pre-narrowed fp16 banks on disk, or residency (M9/M11).

### M9 — serving (2026-09-26)

`-video` serves OpenAI's asynchronous video API (API.md, *Videos are
jobs*): `POST /v1/videos` → a queued job, `GET /v1/videos[/{id}]`,
`GET /v1/videos/{id}/content` → the mp4, `DELETE` to cancel and forget. It
also reads SGLang's H3 envelope (`task`, `target.{short_edge, aspect_ratio,
duration_seconds}`, `seed`), so the README's request scripts work against it
unchanged but for the host (M10 added keyframes; see there). The pieces:

- **The queue is in `api`** (`VideoJobs`): one job at a time, 16 waiting at
  most (429 past it), jobs in memory, files in `-video-dir` for `-video-ttl`
  (24 h). The backend (`backend.Video`) is synchronous: plan a request on the
  host, then generate one mp4 to a path. `api/videos_test.go` covers the
  queue against a fake: both envelopes and a multipart form, the refusals,
  one-at-a-time, 409 on unfinished content, 429, cancel of a running job,
  failure, shutdown, expiry.
- **A request is planned at submit** (`Pipeline.Resolve`): canvas, frames,
  prompt tokens (≤ 4096) and **the transformer's arena check**
  (`dit.CheckArenas`, split out of `allocActivations`). The fp16 q/k/v planes
  span the sequence, so **87,296 rows** is the ceiling: 480p to 14.4 s, the
  trained 768p to ~11.8 s (10 s is 73k rows; 12 s is 88.7k and refused). The
  refusal is a 400 at submit, not a failed job after a 50 GB encode. Under
  5 s is raised to 5 (OpenAI clients send "4").
- **Only submissions hold the device.** Staging here is allocation, pipeline
  creation and writes into mapped memory; none of it touches the queue
  (checked: no dispatch in `newEncoder`, `dit.NewGPU`, `vae.NewGPU`). So
  `pipeline.Options.Hold` wraps just the encoder's forward, `Begin`, each
  forward and the VAE decode, and ~2 of a request's minutes of bf16 reads run
  unlocked. Inside a held section, `dit.GPU.Between` / `vae.GPU.Between` run
  between the ≤ 800 ms submissions: the backend yields the device there, and
  the request's context is checked there, so a DELETE or a shutdown stops a
  forward within one submission.
- The job reports `stage`, `progress` (weighted by the estimate) and
  `estimated_seconds`: forward = 1.159e-3·L + 6.745e-8·L² s (fitted through
  M7's 480p and 768p), plus M8's fixed stagings. 206 s estimated, 199 s run.

**The gate** (`serve -video -tts`, the README prompt at 448×256 × N = 8 in
SGLang's envelope, a speech request every ~2 s beside it):

| stage | job | speech round trip (idle: 17–18 ms) |
|---|---|---|
| encode (encoder staging unlocked + forward) | ~75 s | 19–92 ms, but **1.18 and 1.09 s in its first 4 s** |
| transformer staging | ~60 s | 18–267 ms, one **0.50 s** at its start |
| 7 forwards (7.7 s each, held) | ~55 s | **28–255 ms** |
| decode + mux | ~17 s | 20–201 ms |
| total | **3 m 19 s** (M8 direct: 3 m 28 s) | |

A speech request is several device units, and each waits at most one
submission. The two long waits at the start of the encode, and the 0.5 s at
the transformer's, line up with allocating 50 and 44 GB. That is the kernel
making the memory (the lock is not held then), and it is not something the
lock order can fix. The mp4 reads back as h264 448×256 × 124 frames plus aac
32 kHz stereo.

**Cancel, then again.** A second job was DELETEd 3 s into its forwards, and
it stopped **0.33 s** later, inside a forward. A third job on the same
process then ran in 3 m 17 s, and its mp4 is **byte-identical** to the
first's (same seed). So an abandoned forward leaves nothing behind, and a
served request reproduces from its seed.

### M10 — keyframes, fl2va (2026-09-26)

A keyframe reaches the transformer **twice**, and both routes had to be
reproduced (diffusers' `MiniMaxH3ResizeStep`, `FL2VATextEncoderStep`,
`KeyframeVaeEncoderStep`, `PrepareConditionLatentsStep`):

1. **In the prompt.** The presentation is `"<Picture i>: "` + a Qwen3-VL
   vision block per keyframe, then the prompt verbatim, still with no
   template. The block's rows are tagged **video**, not text, which is what
   the transformer's AdaLN keys off. The conditioner reads the picture
   exactly as a Qwen-Image edit reads a reference: the tower's merged rows in
   the pads, deepstack after layers 0–2, 3-D mrope positions. So this is
   `qimage/vision` and `qimage/textenc`'s edit path, unchanged but for a
   constructor that takes already-built ids (`NewPromptIDs`) and a zero
   system-prefix drop.
2. **As anchor rows.** The picture through the video VAE's **encoder**, a
   posterior *draw*, rounded to fp16, normalised, noised to t = 0.999 with
   the request's generator, patchified and packed ahead of the generated
   video rows. They sit at `max(t_video, 0.999)` in every forward and are
   never stepped. M1 already laid them out; `dit.GPU` needed no change (the
   four modulation keys — text, vision-as-video, keyframe, audio — fit its
   eight).

**The encoder** (`h3/vae/encoder.go`) is a causal 3-D CNN, and at one frame
it *is* 2-D: every temporal filter is front-padded with two zero frames, so
only its last tap ever meets the picture, and diffusers runs a single frame
through `_encode_clip` alone. It runs on the device in fp32 on the image
VAE's direct-convolution kernel, with two additions: a `REFLECT` build
(the checkpoint pads by reflection; the non-reflect SPIR-V is byte-identical
to before), and a GroupNorm(32)+SiLU kernel (`h3_groupnorm.comp`). The
reflect-padded (0, 1, 0, 1) stride-2 downsampler is the stride-1 filter read
at the odd pixels, as with zeros. 256-pixel tiles, widened overlaps and the
blend are the decoder's `splitTiles`/`stitch` in latents. 0.24 GB of weights
(one tap in three), **~100 ms a tile**: 0.2 s at 448×256, 1.4 s at 864×480.

**The posterior draw is load-bearing.** Sampling moves the normalised
anchor by up to 0.16–0.33 against the posterior mean (`sample_vs_mode`), so
torch's CPU `randn` under seed 42 is reproduced (`vae.TorchRandn`: mt19937,
24-bit uniforms, Box–Muller in blocks of 16 with the tail redrawn) to ≤ 1
float32 ulp, and the sample, fp16 rounding and normalisation are then
**bit-exact** on the oracle's moments. The request's own noise (the
augmentation, then video, then audio, in the pipeline's order) stays our
PCG (decision 8) and is injectable (`Request.CondNoise`).

**Placement** (`h3/pipeline/keyframe.go`) is `MiniMaxH3ResizeStep`: with no
canvas given, the canvas takes the first keyframe's aspect; that keyframe is
*stretched* onto it with PIL's LANCZOS (qimage's port), and a second one is
cover-cropped (Python rounding, `(size − canvas) // 2`). Alpha is dropped,
as `convert("RGB")` drops it. Byte-identical to PIL on all three dumped
placements.

**Gates** (`reference/dump_h3_fl2va.py`: the README fl2va request's keyframe
and prompt at 256×448 × 124, 1,039 presentation tokens, 5,709 rows):

| what | result |
|---|---|
| placement, presentation ids + tags, processor pixel_values, auto canvas, keyframe rows from latent + noise | bit-exact |
| encoder, first tile, 15 stages | rel 1.3e-7 … 5.6e-6 (norm_out 2.8e-4: a small group variance divided out) |
| moments, 256×448 (2 tiles) and 480×864 (15) | rel ≤ 1.6e-5; anchor latents ≤ 5e-4 (fp16 roundings that flip) |
| `TorchRandn(42)` against torch | ≤ 4.8e-7 |
| vision tower (fp16, unchanged) | merged 3.9e-2, deepstack 4.5e-3 … 3.0e-2 of absmax |
| conditioner, oracle tower rows fed in | **9.3e-4** (f), 3.3e-4 (fl); 1-D positions control: 0.93 / 0.11 |
| conditioner, device tower | 7.4e-3 (f), 1.0e-3 (fl); the official bf16 pipeline: **1.2e-2 / 1.1e-2** |
| transformer blocks 0 / 49, teacher-forced | rel 1.2e-4 / 2.5e-4 |
| forwards 0–1, teacher-forced | velocity rel 2.8e-2 (rms 5.6e-3); **latents rms 1.1e-4, 1.3e-4**; keyframe rows unchanged |

The conditioner's teacher-forced figure is 8× t2va's 1.1e-4 (the image rows
enter fp16 activations at the tower's scale); the control shows a wrong
mechanism lands 100–1000× further off. The velocity's rel is ~3× M7's t2va
figure at the same shape, and the steps it produces are inside M7's bound.

**End to end** (`cmd/h3 -first … -short 256 -steps 8`, the README prompt):
3 m 32 s, against t2va's 3 m 28 s at the same shape. The encode is 88 s
against 82 (the VAE encoder and the tower staged and run inside it). **Frame
0 matches the keyframe at 27.1 dB PSNR**, letterbox bars and all, and the
clip then does what the prompt says: steam rises, the family eats, the bowl
holds its place (21.0, 18.7, 18.5 dB at frames 40, 80, 123 against the
keyframe). **First and last** (the dump's portrait crop around the bowl as
the last keyframe, cover-cropped into a close-up): 3 m 22 s; frame 0 is the
first keyframe at **27.8 dB** and frame 123 the last at **28.2 dB**
(11.2–11.6 dB against the other one). The model gets there by a cut to the
close-up about two thirds in, not a push-in, which is a fair reading of two
keyframes and a prompt that says nothing about the camera moving.

**Served** (`serve -video`, the README fl2va request with its keyframe
inlined as a data: URI, `short_edge 256`, 5 s, N = 8): the canvas resolves
to 448×256 from the keyframe (`aspect_ratio: "auto"`), estimated 220 s, run
in 3 m 21 s, and the mp4 is **byte-identical** to the CLI's at the same
seed. (The README's own request, CDN link and all, was a 400 that day; it
is fetched now, below.)

**Served:** OpenAI's `input_reference` (multipart, or OpenAI's JSON
`{"image_url": …}`, or a bare string) is the first frame; SGLang's
`conditions` are up to two images with `frame_index` 0 (first) and -1 or ≥
the last requested frame (last). **URLs are fetched, as OpenAI and SGLang
fetch them** (the API follows OpenAI's standard: `util.FetchImage`, 50 MB,
60 s), at submit, so the README scripts' CDN links work unchanged and an
unreadable picture is a 400. `file_id` is a 400 (no Files API). `task:
fl2va` with no keyframe is t2va, as the model card has it; `t2va` with one
and `ref2va` are 400s.

### M11a — int8 banks for the encoder and the transformer (2026-09-26)

The deployed service leaves ~35 GB free, and a request staged 50 GB (the
encoder) then 42 GB (the transformer). Both now hold their projections as
**int8, one fp16 scale per 32 k of a row**. That is Kev K7.1's
quantisation (q rounded against the fp16 scale) in the LLM's L8 fragment
layout, `qwen.PackQ8`.

**Not an int8 GEMM.** A layer's (or block's) first seven dispatches expand
its projections into one fp16 scratch (`shaders/dit_dequant_q8.comp`, 64
threads a 16×16 tile), and the validated fp16 GEMMs read the scratch as
before. The expansion is **bit-identical** to the host's
(`qwen.TestGPUDequantQ8`). At these row counts the GEMMs are compute-bound,
so an int8 kernel would buy nothing (TODO.md: int8 WMMA runs at fp16 rate).
Writing 0.77 GB of fp16 a block is invisible next to a 7 s forward:

| | fp16 | int8 |
|---|---|---|
| encoder staged | 50 GB, 51 s | **26.9 GB**, 60 s |
| encoder, 537 tokens | 1.0 s | 1.4 s |
| transformer staged (weights) | ~40 GB, 53 s | **22.1 GB** (21.3 int8 + 0.77 scratch + heads), 50 s |
| a forward, 5,095 rows, same day | 8.26 s | **7.29 s** |

At 5,095 rows int8 is **12% faster** a forward. (M7's 7.4 s for fp16 did
not reproduce on the day.) Every block's GEMMs read the same hot 0.77 GB
scratch rather than a stack spread over 40 GB, which is the likely reason,
but that is untested. At the served 480p, where attention dominates, it
is even: 35.7 s against M7's 35.6.

`pipeline.Options.Bank`, `cmd/h3 -bank q8|fp16` (default q8), and `serve
-video-fp16` for the control (default int8). Tests take `H3_BANK=q8`.

**Gates.** The encoder is held to its fp32 oracle as M2 was. The transformer
is held to the **released pipeline**: `reference/dump_h3_dit_bf16.py` runs
M4's seven forwards teacher-forced at diffusers' `torch_dtype=bfloat16` (its
`_keep_in_fp32_modules` split). That is the error MiniMax ships, and fp16
was never the right yardstick for a quantised bank.

| what | fp16 | **int8** | official bf16 |
|---|---|---|---|
| encoder, README prompt, rel of absmax (en / cjk) | 1.1e-4 | **3.0e-4** (2.1e-4 / 1.7e-4) | 1.2e-2 |
| refiner | 5.4e-4 | 1.7e-3 | 1.3e-2 |
| blocks 0/1/2/25/49, teacher-forced | ≤ 1.9e-4 | ≤ 4.2e-4 | |
| forward 0 video velocity, rms | 1.8e-3 | 1.8e-2 | 2.0e-2 |
| step latents rms (video), steps 0…6 | 7.9e-5 … 1.7e-3 | 7.9e-4, 1.0e-3, 2.1e-3, 2.1e-3, 2.2e-3, 3.2e-3, **6.9e-3** | 8.6e-4, 1.7e-3, 3.0e-3, 3.3e-3, 3.5e-3, 6.6e-3, **1.6e-2** |
| step latents rms (audio), steps 0…6 | ≤ 4.0e-4 | 6.9e-4 … 4.8e-3 | 1.1e-3 … 8.5e-3 |

**Every int8 step is at or inside bf16's, on both modalities** (0.46×–0.99×;
step 1's audio is the 0.99). `TestGPURun` now gates int8 at 1.25× bf16's
step, a step at a time, and fp16 at its old 3e-3.

**Where the error comes from** (`TestGPUQ8Ablation`, `H3_Q8_ABLATION=1`:
chosen projections of an fp16 bank held at their int8 values; the "all" arm
reproduces the real bank's numbers exactly). Putting any one of the seven
back to fp16 moves the worst step by at most 11% (6.1e-3 … 7.0e-3 against
6.9e-3). The error is spread evenly, so a mixed bank buys nothing worth its
memory.

**End to end** (`TestE2E`, free-running from the oracle's noise, 448×256 ×
N = 8): 3 m 4 s against 3 m 28 s, most of it staging. Frames PSNR **19.8 dB**
against the oracle's decode (fp16: 25.9), soundtrack SNR 16.5 dB (21.5), and
final latents rms 0.263. That is one draw of a chaotic divergence.
`TestGPUSensitivity` now takes `H3_SENSITIVITY_DRAWS`, and 8 draws of a
1e-4 move of the starting noise alone end 0.056–0.212 rms apart (median
0.10). fp16 lands at 0.121 and bf16 (`dump_h3_dit_bf16.py --free`) at 0.146.
Int8's draw is 1.24× past the chaos spread's top, while its every step is
inside bf16's. Frames 0/60/120 against the oracle's show the same shots,
character and lighting; framing and ship placement drift, as free runs do.
`TestE2E`'s floor for int8 is 18 dB (`q8PSNRFloor`): it catches a broken
run, and `TestGPURun` holds the precision.

**Memory at a request's peak**, measured, not summed: MemAvailable
sampled every 0.5 s against idle, over `TestE2E`. The int8 banks alone
took the peak only from 57.1 to 39.3 GB, because the host was holding the
rest. Two host-side fixes brought it to 31:

| | peak | encoder phase | transformer phase | wall (448×256, N = 8) |
|---|---|---|---|---|
| fp16 | 57.1 GB | | | 3 m 20 s |
| int8 | 39.3 GB | 33–39 | 29 | 3 m 3 s |
| + `dit.LoadHost`, a GC a staged layer | 34.0 GB | 30–34 | 21 | 3 m 19 s |
| + tables beside the transformer's staging | **31.2 GB** | 30 | 21–24 | 3 m 22 s |

- **`dit.LoadHost`.** `Tables` and `NewGPU` each called `Load(dir, 0)`,
  which materialises the two refiner blocks in fp32 (3.1 GB). Neither
  needs them: `Tables` wants the time MLP, and the device stages the
  refiner from the checkpoint.
- **`runtime.GC()` once per staged layer or block** (`qwen.stageWeights`,
  the DiT's staging, `Tables`). A 32 B encoder layer is ~2 GB of fp32 on
  the host, dead once quantised, and at the default GOGC the heap's
  doubling headroom kept several of them. `GOGC=25` showed the size of it
  (39.3 → 35.9 GB). The collection is pointer-free and cheap, and it
  changes no process-wide setting in a server that runs every vertical.
- **The AdaLN tables moved from beside the encoder's staging to beside the
  transformer's**, which now has the headroom. With `Resident` the
  transformer is already staged, so they stay beside the encoder.

That costs ~20 s of wall at this shape (the tables contend with the
transformer's staging for disk, plus the collections), ~2% of a served
480p request. The frames are the same bits throughout.

**The served shape** (`cmd/h3`, README prompt, 864×480 × 124 frames,
N = 20, 15,936 rows): **14 m 25 s**, peak **32.6 GB** (in the encoder's
staging), the transformer phase 19–24 GB, 35.7 s a forward, decode 52 s.
That looked like it fit inside the ~35 GB the deployed service left, with
~2 GB to spare. **It did not.**

**Served, it was OOM-killed** (2026-09-26 19:34:56, 7 s after submit;
systemd restarted the unit and the in-memory job was lost). The kernel's
report had 79.7 GB of GPU pages and 40.6 GB of anon, which is all of RAM.
The standalone measurements had missed the host side of a *server*.
Rerun by hand with `GODEBUG=gctrace=1`, `H3_MEMLOG=1` (the pipeline logs
the Go heap and RssAnon every 250 ms) and a guard that kills the server
below 3 GB free, the Go heap went **14.6 → 28.6 GB in 2 s**, before the
encoder's first layer, with no collection run. Two causes:

- **`safetensors.Tensor.F32(nil)` grew its output by `append`.** A large
  slice grows ~1.25× at a time, so the 3.1 GB embedding table churned
  ~15 GB of copies, and every fp32 load of every model paid ~5× its size.
  Standalone, the heap was small and the GC collected constantly, which
  hid it. In the server, with a 14 GB live heap, the next target was
  29 GB. `F32` and `F16` now reserve their capacity (`slices.Grow`).
- **The server kept ~11 GB of collected garbage resident** from staging
  its other models: 24 GB RssAnon against a 13 GB live heap. `serve` now
  calls `debug.FreeOSMemory()` once every backend is up, and the pipeline
  calls it at the start of a request and after each staging is freed.

| the deployed configuration (image, kev, speech, video) | available at rest | a 480p request's low point |
|---|---|---|
| before | 36.9 GB | OOM at +7 s |
| after | **47.5 GB** | **15.4 GB free** (+17 s, encoder staging); 18.7 GB in the transformer phase |

The server now gives every vertical ~10 GB more room at rest, not only
video. **Served end to end** (hand-run server with the unit's flags, the
README prompt, 864×480 × 20 steps): completed in **826 s**, 124 frames of
h264 and stereo AAC, and the frames follow the prompt.

**Through `ai.service` itself** (the unit's line restored to `-image
-edits 3 … -video`, the same request on :11434): completed in **834 s**,
low point **15.4 GB free**. The deployed machine now serves video beside
the image model. Don't measure headroom with anything else running: a
background `go test -short` sweep stages models too, and it tripped the
guard on the first attempt.

### M11b — the stagings from a bank cache (2026-09-29)

M11a left every request re-making its int8 banks from the checkpoint: 48 GB
of bf16 for the encoder and 42 GB for the transformer read, widened to fp32
and `PackQ8`'d, plus 17 GB more for the AdaLN tables beside it. At 3.7 GB/s
off this disk that is ~30 s of reads. **The rest was CPU**, and all of it
produces the same bytes every time.

**The cache** (`qwen.Q8Cache`, `zimage/qwen/q8cache.go`) keeps a staging's
int8 banks on disk byte for byte as the device holds them, one file a bank,
in `bank-cache/` beside each component (`text_encoder/`, `transformer/`).
A miss stages as before and writes each packed matrix at its bank offset
into a temporary file, renamed into place once the staging is complete. A
hit reads the files straight into the mapped banks, 64 MB chunks on 8
readers, and loads only the norms from the checkpoint. The key hashes the
layout (every matrix's shape, bank and offset, so a different layer count
or fp16 keep list is a different file set), `q8CacheVersion`, and each
shard's name, size and mtime. It does not hash the shards' bytes (the LLM's
bank cache makes the same trade, P4c). The AdaLN tables are cached the same
way (`dit.TablesCached`), keyed by the request's timestep bits: one file a
step count and t2va/fl2va, 19 MB a distinct timestep (0.25 GB at N = 8, 0.72 GB at N = 20).

On by default: `pipeline.Options.BankCache`, `serve -video-bank-cache`,
`cmd/h3 -bank-cache`. The tests leave it off. It takes **47 GB of disk**
(25.9 GB encoder, 21.3 GB transformer), written by the first request, which
pays ~14 s for the writes.

| 448×256 × N = 2, `cmd/h3` | encode (staging + forward) | transformer staging | tables | request |
|---|---|---|---|---|
| M11a (no cache) | 40.5 s | 34.3 s | 19.2 s (beside it) | 97 s |
| first run (writes the cache) | 48.2 s | 40.4 s | 19.3 s | 111 s |
| cached, files evicted from the page cache | **7.5 s** | **4.5 s** | **8 ms** | **33 s** |

The cold row is honest: the files were evicted with
`posix_fadvise(DONTNEED)` first (no root needed), and buff/cache fell from
71 to 13 GB. That is 26 GB in ~6 s and 21 GB in 4.5 s, ~4.7 GB/s. Every
run's mp4 is **byte-identical** to the uncached one (same seed).

Served (hand-run `serve -video`, README-style request at `short_edge 256`,
`7:4`, N = 8): **74 s** end to end, against M9's 3 m 19 s. The default shape
(864×480 × 124 frames, N = 20): **741 s** against M11a's 826 s. That run
also computed the N = 20 tables for the first time (0.7 GB, 19 s, which
hide behind the transformer's staging), so a repeat should come in ~10 s
under. The forwards are now ~95% of a served request.

What is left of a request's staging is the video VAE's (inside the decode's
14 s at the small canvas) and the embedding table (3.1 GB of fp32 widened
from bf16 on every encoder staging). Neither has been timed on its own yet.

### M11c — the attention, transposed (2026-09-29)

The attention was 42% of a served forward and 68% of a trained one, at
20–26 TFLOP/s against 55.5. **It is now 1.55–1.61x faster, and a forward
is 1.16x (480p) and 1.30x (768p).**

**Where the time went.** The kernel was priced by gutting it one phase at
a time on H3's own planes (block 49 after a real forward, 480p, 15,936
keys, QT1 KTIL4; `H3_ATTN_SPV` loads a build from a file into
`TestGPUAttentionScreen`):

| arm | time | TFLOP/s |
|---|---|---|
| the plain kernel | 288 ms | 25.3 |
| no row max (LDS scan, 2 barriers a key tile) | 288 ms | 25.3 |
| and no exp2 | 296 ms | 24.6 |
| and no LDS round trip for P (store, barrier, reload as A) | **111 ms** | 65.7 |

The softmax costs nothing. The round trip that turns P from an
accumulator back into an A operand is **~60% of the kernel**. The
extension has no other path from an accumulator to an operand.

**Refused first: LLM's P12-1 lane split** (the row max over all 32
lanes, one clustered max, the running max in a register). Bit-identical,
and **slower in every arm**: 23.8 → 23.1 TFLOP/s at QT1 KTIL4, 480p, and
22.4 → 19.8 at QT2 KTIL4, 38k keys. That is what the table predicts: the
row max was never the cost. It is not in the tree.

**The fix: compute the transpose.** `shaders/h3_attn_t.comp` runs
S^T = K·Q^T and O^T += V^T·P^T. On this device an accumulator's column
and a B operand's column are the same lane. The accumulator splits a
column's 16 rows between lanes l and l^16 (even rows in the first
half-wave, odd in the second), where B wants all 16 in both. So P^T
becomes the B operand with one `subgroupShuffleXor(16)` per packed pair
of halves. A lane then owns one query of a tile, and the row max, the
online correction, the row sum and the final 1/sum are all per-lane
scalars. **No LDS and no barrier.** The memory reads are the plain
kernel's, byte for byte: K as a row-major A, Q^T as a column-major B, the
packed v tile as V^T row-major, and O^T stored column-major is O at the
output projection's stride.

That relies on an element order `GL_KHR_cooperative_matrix` leaves to
the implementation, so it is checked, not assumed.
`shaders/coopmat_layout_probe.comp` reads back which (row, column) every
lane's element is, for A, B and both accumulators. `checkCoopMatLayout`
runs it at the first `Begin` (held in a server; staging still submits
nothing), and on any other order the build falls back to the plain
kernel. `TestCoopMatLayout` pins it: A lane l = row l%16, element e =
column e; B lane l = column l%16, element e = row e; accumulators lane l
= column l%16, element e = row 2e + l/16.

**Two numeric differences, neither bit-for-bit with the plain kernel.**
The denominator sums the fp16-narrowed weights in fp32 registers instead
of through a ones column on the matrix cores. The key tail is masked out
of the max as well as out of P. The second makes the kernel immune to
stale keys by itself: `TestGPUForward`'s negative control (no
`zeroKeyTail`) now pins the plain kernel, which still goes NaN, while the
transposed one stays at rel 5.3e-4. The zeroing stays, because P = 0
still multiplies whatever stale v rows hold.

**The screen** (`TestGPUAttentionScreen` + `H3_ATTN_SPV`, block 49's
planes; rms relative to the plain QT1 KTIL4 context):

| build | 15,936 keys | 38,247 keys | VGPRs / spilled |
|---|---|---|---|
| plain QT1 KTIL4 | 297 ms (24.5) | 2.16 s (19.4) | |
| plain QT2 KTIL4 (was picked ≥ 24k) | 297 ms | 1.87 s (22.4) | |
| T QT1 KTIL4 | 246 ms (29.6) | 2.02 s (20.8) | 216 / 0 |
| T QT2 KTIL2 | 227 ms (32.1) | | 256 / 64 |
| **T QT2 KTIL4** | **194 ms (37.6)** | **1.17 s (36.0)** | 256 / 112 |
| T QT2 KTIL4, Q reloaded a block (`QREG=0`) | 215 ms (33.8) | | 256 / 0 |
| T QT2 KTIL8, QT3, QT4 | 301–881 ms | | spills 266–1341 |

rms against the plain kernel **9.7e-5 / 1.1e-4**, about the size of fp16's
own rounding of the context. QT2 KTIL4 spills 112 VGPRs and still wins.
Reloading Q every block removes the spill and loses 11%. `attnFor` runs
T QT2 KTIL4 at every key count.

**Gates** (fp16 bank unless noted):

| gate | result |
|---|---|
| `TestGPUForward` forward 0 | video rel 6.43e-3 (M7: 6.7e-3), audio 1.25e-3; blocks ≤ 1.9e-4 of absmax |
| `TestGPURun`, 7 teacher-forced steps | latents rms 9.8e-5 … 1.68e-3 (M7: 7.9e-5 … 1.7e-3) |
| `TestGPURun`, int8 | every step inside the released bf16 pipeline's error (step 6: 6.9e-3 against 1.59e-2) |
| `TestGPUFL2VA`, int8 | steps 1.4e-3 rms, as M10 |
| `TestE2E`, int8, free-running | frames 20.3 dB, soundtrack 16.0 dB (M11a's int8 draw: 19.8 / 16.5) |

**Forwards** (`TestGPUShapes`, int8, `H3_ATTN_PLAIN=1` as the control, two
passes of each arm interleaved):

| shape | plain | transposed | |
|---|---|---|---|
| 864×480 × 124 (served), 15,936 rows | 35.80 / 35.37 s | **30.77 / 30.50 s** | 1.16x |
| 1344×768 × 124 (trained), 38,247 rows | 143.9 / 145.1 s | **108.1 / 113.6 s** | 1.30x |

A served 480p × 20 request (19 forwards) should lose ~95 s of M11b's
741 s. That is estimated from the forwards, not yet measured served.

### M11d — the down projection (2026-09-29)

After M11c, a 480p forward's profile (`TestGPUShapes`, `H3_PROFILE=1`,
int8) is attention 32%, qkv 16%, **down 16%**, gate and up 11% each, `o`
6%. Every GEMM runs at 35–38 TFLOP/s except the down projection, at
24–28. Its grid is `o`'s: 21 × 64 workgroups of 128×256 at chunk 8192,
N = 5376 for both. What differs is K: 14,336 against 7,168.

**Three screens, 480p** (`gemm down`'s share of one forward's profile):

| arm | gemm down | |
|---|---|---|
| swizzle band 8 (shipped) | 4.33–4.71 s, 26–28.5 TFLOP/s | |
| band 16 / 4 / 2 | 22.6 / 14.1 / 9.3 TFLOP/s | refused |
| A row pad 0 / 64 / 256 / 512 halves (shipped 128) | 19.8 / 24.0 / 23.7 / 25.6 | refused |
| **K in 2 passes** (`-DKRANGE`) | **4.24–4.27 s, 28.9–29.1** | shipped |
| K in 3 / 4 passes | 29.3 / 29.2 | no better than 2 |

The band result refutes the first guess, which was that a band of 8
columns of B slabs (59 MB at this K, against 29 MB for `o`) overflows the
32 MB MALL. Narrower bands are *worse*, because every band re-streams the
235 MB A chunk. The shipped band and pad are both already the optimum.

**K in two passes.** `dit_gemm.comp -DKRANGE` runs the K loop over
`[aux0, aux1)`. With `aux2 = 1`, its accumulators start from the fp32 C
already in place. An fp32 store and reload is exact, so the two passes
make the same MMAs in the same order as one: **bit-identical**, which
`TestGPUForward` now asserts (`dit.GPU.downSplit`, default 2;
`H3_DOWN_SPLIT=n` in `TestGPUShapes`). Two passes also steady the
kernel: one pass measured 24.0–28.5 TFLOP/s across runs, two passes
28.9–29.1. It still stops short of `o`'s 35, so K's length is part of the
gap, not all of it.

**Forwards** (int8, two passes of each arm, interleaved): 480p **29.81 /
29.91 s** against 30.41 / 30.42 s one-pass (1.02x). 768p 106.1 / 109.6
against 107.2 / 114.4, inside that shape's run-to-run spread.

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
- ~~M-o2. Does the ViT decoder tolerate fp16?~~ **Yes** (M5): its activations
  peak at 832, and the decode lands at PSNR 81.7 dB against fp32.
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

**2026-09-29, session 8: M11d done.** The down projection runs as two K
passes (`-DKRANGE`, `downSplit`), bit-identical, 29 TFLOP/s against
24–28. It is 1.02x a 480p forward, now ~29.9 s. Its swizzle band and A row
pad were screened, and the shipped values are the optimum. `H3_SHAPES=480`
times only the served shape. Not deployed.

**2026-09-29, session 7: M11c done.** The attention runs transposed
(`shaders/h3_attn_t.comp`, `attnFor`) wherever `checkCoopMatLayout` finds
the element order it is written against, which it does on this device. A
480p forward takes 30.6 s (35.6 before), a 768p one 111 s (144). Output is
no longer byte-identical to earlier builds (rms 1e-4 on the context), and
every gate holds. Not deployed. The screen now takes `H3_ATTN_SPV=qt:ktil:path,...`
to time a candidate build without a rebuild, and `TestGPUShapes` takes
`H3_ATTN_PLAIN=1` as the control arm.

**2026-09-29, session 6: M11b done.** The stagings read their int8 banks
and AdaLN tables from `bank-cache/` beside the checkpoint (`qwen.Q8Cache`,
`dit.TablesCached`; on by default in `serve` and `cmd/h3`, off in the
tests). The first request writes the cache (47 GB, ~14 s extra). After that
a request's stagings are ~12 s instead of ~94 s, and the output is
byte-identical. **Not yet deployed**: `ai.service` runs the binary from
before this change. The next deploy's first video request writes the cache.

**2026-09-26, session 5: M11a done.** The text encoder and the
transformer stage as int8 banks by default (`qwen.BankQ8`: `serve`
unless `-video-fp16`, `cmd/h3 -bank`, `pipeline.Options.Bank`). A
request's measured peak is 31–33 GB (fp16: 57), inside the ~35 GB the
deployed service leaves free, and every teacher-forced step is inside the
released bf16 pipeline's error. **A request through the deployed
`ai.service`, beside the image model, completed in 834 s with 15.4 GB
still free**, after two host-side fixes the first attempt's OOM
kill exposed (`safetensors` F32/F16 `append` growth, and
`debug.FreeOSMemory` in `serve` and between the stagings).

**Session 4: M10 done.** `fl2va` is served: a first and/or last
keyframe through `input_reference` or SGLang's `conditions` (URLs fetched,
as OpenAI's API does),
and `cmd/h3 -first/-last`. M11 onward is open.

**Session 3: M9 done.** The vertical is served:
`serve -video` answers `/v1/videos` as OpenAI's async jobs (and SGLang's H3
envelope), and shares the device with the other verticals while it runs.

- Weights: `models/MiniMax-H3` (135 GB, gitignored): `transformer/`,
  `text_encoder/`, `vae/`, `audio_vae/`, plus configs, tokenizers, docs and
  the README's request scripts (`scripts/readme/*.sh`).
- Code: `h3/plan` (M1), `h3/textenc` (M2), `h3/dit` (M3 CPU, `Tables`, M7
  GPU `gpu.go`, `CheckArenas`), `h3/vae` (M5), `h3/audiovae` (M6, CPU fp32),
  `h3/pipeline` (M8: staging order, `Generate`, `WriteMP4`; M9: `Resolve`
  with limits, `Options.Hold`/`Between`; M10: `keyframe.go`, keyframes in
  `Resolve`/`Generate`), M10's `h3/vae/encoder.go` + `posterior.go` and
  `h3/textenc`'s `Presentation`, `cmd/h3` (CLI), and for serving
  `backend/video.go` + `api/videos.go` + the `-video*` flags in `cmd/serve`.
- Oracles: `reference/dump_h3_{plan,tokens,textenc,dit_block,dit,ranges,vae,audio,fl2va}.py`,
  outputs in `reference/out/h3*`. `dump_h3_fl2va.py` runs in phases
  (`prep vae text dit`) and reads `models/MiniMax-H3/assets/fl2va_keyframe.png`,
  the README fl2va request's keyframe (fetched from its CDN link; gitignored).
- Tests: `go test -short ./h3/...` and `go test ./api/` are quick. Without
  `-short`: `TestGPUDecoder` (h3/vae, 10 s), `TestStages`/`TestDecode`
  (h3/audiovae, ~10 s), `TestE2E` (h3/pipeline, ~3.5 min, 57 GB peak in fp16, 31 GB with `H3_BANK=q8`), and
  M10's `TestGPUEncoder*` (h3/vae, seconds), `TestGPUPresentation`
  (h3/textenc, ~70 s, 51 GB), `TestGPUFL2VA` (h3/dit, ~100 s, 44 GB;
  `H3_ENC_SCREEN=1` times the encoder's conv builds). The
  opt-ins are `H3_VAE_FULL=1`, `H3_VAE_SHAPES=1` (+`H3_VAE_SEQS`),
  `H3_AUDIO_RAW=path`, `H3_E2E_MP4=path`, and M7's (`H3_SHAPES`,
  `H3_PROFILE`, `H3_CHUNK`, `H3_SCREEN`, `H3_SENSITIVITY`, `H3_FREE`).
- Served gate by hand: `serve -addr 127.0.0.1:18080 -token= -video -tts`, then
  POST the README request with `target.short_edge 256, aspect_ratio "7:4"`,
  `num_inference_steps 8` and poll (M9's table).
- **Deployed** (2026-09-26): `ai.service` runs `-video` beside `-image
  -edits 3` (49.7 GB), `-kev` and the speech models, which leaves ~35 GB
  available at rest, 47.5 GB since M11a's host fixes. A 480p request's
  low point there is 15.4 GB free (it was OOM-killed before M11a).
  `H3_MEMLOG=1` logs a request's Go heap and RssAnon every 250 ms.
- M11a's tools: `H3_BANK=q8` on every device test,
  `reference/dump_h3_dit_bf16.py [--free]` (the released precision's
  teacher-forced and free-running distance from fp32; `TestGPURun` prices
  int8 on it), `H3_Q8_ABLATION=1` (`TestGPUQ8Ablation`) and
  `H3_SENSITIVITY_DRAWS=n`.
- M11b's: `qwen.TestGPUQ8Cache` (round trip and the three misses), and
  `-bank-cache=false` / `-video-bank-cache=false` as the control. Delete a
  `bank-cache/` to re-pack. Time a cold hit by evicting the files first
  (`posix_fadvise(POSIX_FADV_DONTNEED)` on each, e.g. from Python).

**Next, in order:**

1. ~~The staging cost~~: done, M11b. What staging is left (the video VAE,
   the fp32 embedding table) is not yet timed on its own.
2. M11 levers, measured and priced: ~~the attention~~ (M11c), ~~the down
   projection~~ (M11d; still 29 against `o`'s 35, and nothing cheap is
   left there), the video VAE's attention
   (`h3vae_attn_hd64`, the plain kernel at head 64: a HEAD_DIM=64 build of
   `h3_attn_t.comp` is the obvious try), and the audio decode on the
   device (7 s on the CPU against torch's 1.1). Outside this vertical,
   every `dit_attention_wmma.comp` user (the image DiT, Kev, OCR, ACE) pays
   the same LDS round trip for P.
3. M12: a Context-IR stand-in, since plain prompts are what users will send.
   For fl2va it has to write the `<Picture 1>` references the README's
   prompts carry ("at 0.00 seconds … <Picture 1> … is fully referenced").
