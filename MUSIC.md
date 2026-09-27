# MUSIC — text (and lyrics) to a song (ACE-Step 1.5 XL turbo)

> **Live tracking and session handoff doc, opened 2026-09-27.** Stage letters
> are **A** (for ACE; `M` is the video vertical's). When the vertical closes,
> this file is frozen to `research/music-vertical.md` like the others and
> `TODO.md` gets the one-line summary. Until then: tick a stage when its gate
> passes, put its measured numbers under it, and keep **§ Handoff** at the
> bottom current. A new session should be able to start from there.

## What we are building

`GOALS.md` item 8: music generation via ACE-Step 1.5, with two checkpoints:

- the DiT, [`ACE-Step/acestep-v15-xl-turbo`](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo)
  (read at `d4a0b28`);
- the planner LM, [`ACE-Step/acestep-5Hz-lm-4B`](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B)
  (read at `0a3ec94`).

The pipeline code is [`ace-step/ACE-Step-1.5`](https://github.com/ace-step/ACE-Step-1.5),
read at `ca1e85f` (2026-08-29). A caption, optional lyrics and optional
metadata (bpm, key, time signature, duration, language) go in. **A 48 kHz
stereo song of 10 s to 10 min** comes out.

### The pipeline, as upstream runs it by default

ACE-Step 1.5 is two generators in series, and a request can use one or both:

1. **The 5 Hz LM ("thinking", on by default)** is a Qwen3-4B fine-tune
   with 65,535 audio-code tokens added to its vocabulary. It runs in two
   phases:
   - **CoT phase** (no CFG): it writes a YAML block inside `<think>`: bpm,
     a rewritten caption, duration, key, language and time signature. It
     skips any field the user already supplied.
   - **Codes phase** (CFG 2.0 against a `NO USER INPUT` prompt): it writes
     **5 audio codes a second** until the target duration. EOS is blocked
     until then.

   Both phases sample at temperature 0.85 and top-p 0.9 through a
   constrained-decoding FSM (`constrained_logits_processor.py`).
2. **The DiT** takes three inputs:
   - the caption and metas through Qwen3-Embedding-0.6B;
   - the lyrics as raw token embeddings through its own lyric encoder;
   - a "timbre" reference (the silence latent when there is none).

   If the LM ran, its codes are detokenised to 25 Hz "LM hints" and replace
   the silence as the source latents. That is the `is_covers` path. The DiT
   then denoises 25 Hz × 64-channel latents in **8 Euler steps, CFG-free**
   (distilled).
3. **The VAE** (Oobleck, 48 kHz stereo, hop 1920) decodes the latents.
   Then comes peak normalisation to −1 dB.

With thinking off, step 1 is skipped. The DiT then sees silence as its
source, and "N/A" metas unless the user gave them. This is the DiT-only
path, and **it is the build order's first milestone**. The LM adds
structure, and it is also most of the request's time (see below).

### What we download, and what we do not

| what | from | size | why |
|---|---|---|---|
| DiT + condition encoder + audio (de)tokenizer | `ACE-Step/acestep-v15-xl-turbo` (whole repo) | 19.9 GB (fp32) | the generator |
| 5 Hz LM | `ACE-Step/acestep-5Hz-lm-4B` (whole repo) | 8.4 GB (bf16) | the planner |
| VAE | `ACE-Step/Ace-Step1.5`: `vae/` only | 337 MB (bf16) | latents → 48 kHz stereo |
| text-encoder configs | `ACE-Step/Ace-Step1.5`: `Qwen3-Embedding-0.6B/*.json`, `*.jinja`, `merges.txt` | < 20 MB | to confirm it is ours |
| text-encoder weights | **not downloaded** | (1.19 GB) | **byte-identical to `models/Qwen3-Embedding-0.6B`** (sha256 `0437e45c…` both). Only the config differs: `Qwen3Model` rather than `Qwen3ForCausalLM`, and the same weights |
| `acestep-v15-turbo/` (2 B DiT), `acestep-5Hz-lm-1.7B/` | `ACE-Step/Ace-Step1.5` | 8.5 GB | not the models `GOALS.md` names |

Both model repos were already present in `models/` as **git-LFS clones**, so
every weight is on disk twice (`.git/lfs/objects` plus the working tree:
36 GB for 28 GB of weights). Removing the two `.git` directories would
reclaim ~28 GB, but that is the user's call. The rest was fetched with:

```sh
.venv/bin/hf download ACE-Step/Ace-Step1.5 vae/config.json vae/diffusion_pytorch_model.safetensors \
  --local-dir models/Ace-Step1.5
.venv/bin/hf download ACE-Step/Ace-Step1.5 --include "config.json" "README.md" \
  "Qwen3-Embedding-0.6B/*.json" "Qwen3-Embedding-0.6B/*.jinja" "Qwen3-Embedding-0.6B/merges.txt" \
  --local-dir models/Ace-Step1.5
git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git models/ACE-Step-1.5-src   # the pipeline, ca1e85f
```

(`hf download --include a b c` takes only the first pattern after a
positional file list; name the VAE files explicitly.)

## The model, from the configs and the safetensors headers

### Parameters

| component | params | stored | on the device, fp16 | runs |
|---|---|---|---|---|
| DiT decoder: 32 layers + time embeds + patch in/out | 4.17 B (layers 4.07) | fp32 | 8.3 GB | 8 times a request |
| condition encoder: lyric encoder (8 layers), timbre encoder (4), text projector | 0.61 B | fp32 | 1.2 GB | once a request |
| detokenizer: 5 Hz codes → 25 Hz hints (2 layers + FSQ `project_out`) | 0.10 B | fp32 | 0.2 GB | once, LM path only |
| audio tokenizer: 25 Hz latents → codes (pooler + FSQ) | 0.10 B | fp32 | 0.2 GB | cover/repaint only (A11) |
| 5 Hz LM: Qwen3-4B, 36 layers, vocab 217,204, tied embeddings | 4.02 B | bf16 | 8.0 GB | ~5 tokens a second of song, ×2 rows for CFG |
| text encoder: Qwen3-Embedding-0.6B | 0.60 B | bf16 | 1.2 GB | once |
| VAE decoder (Oobleck) | 0.084 B | bf16 | 0.17 GB | once |

Everything resident is **~19.5 GB in fp16**, and activations are small
(below). Nothing here is big by this repo's standards. The questions are
speed and fidelity, not fit.

### The DiT (`AceStepDiTModel`, `modeling_acestep_v15_xl_turbo.py`)

- **Input.** `[context_latents | x_t]` on channels: 64 source latents + 64
  chunk mask + 64 noisy = 192. They are patched ×2 in time by a Conv1d(k2,
  s2) to hidden **2560**, so **a token is 80 ms**. There are 12.5 tokens a
  second, 3,000 for a 4-minute song and 7,500 at the 10-minute ceiling.
  Output is a ConvTranspose1d(k2, s2) back to 64 channels.
- **32 layers**, each **self-attention → cross-attention → SwiGLU**:
  - 32 q heads over 8 kv heads, head 128 (q is 4096 wide, wider than the
    residual);
  - q/k RMSNorm per head, RoPE θ 1e6 on plain token positions;
  - SwiGLU 9728, no biases.
- **Self-attention is bidirectional. Even layers use a ±128-token sliding
  window (|i−j| ≤ 128) and odd layers are full.** The model code passes *no*
  padding mask to the DiT (`attention_mask = None`), so with one request a
  batch there is none to reproduce.
- **Cross-attention** reads the packed condition sequence after a
  `condition_embedder` Linear (2048 → 2560). It uses the same q/k norms and
  no RoPE, with no residual gate. **Its K/V are the same at every step**, and
  upstream caches them in the first forward (`EncoderDecoderCache`). We
  compute them once a request.
- **AdaLN from a per-layer `scale_shift_table` (6 × 2560) plus a shared
  timestep projection.** The timestep projection is
  `time_embed(t) + time_embed_r(t − r)`, each a sinusoid(256) → MLP →
  `time_proj` to 6 × 2560. Upstream always calls it with `r = t`, so the
  second term is a constant. It covers shift/scale/gate for self-attention
  and for the MLP. The final norm uses a 2 × 2560 table on `temb`. With 8
  timesteps and one table per request, all of it is host arithmetic.
- **Velocity convention**: `x_t = t·noise + (1−t)·x0`, the model predicts
  `v = noise − x0`, and Euler is `x ← x − v·(t − t_next)`. **The last step
  jumps straight to `x0 = x − v·t`.** So `N` schedule entries are **N
  forwards**. This differs from H3, where the terminal 0 was an entry.

### The turbo sampler

The schedule is fixed per shift. Upstream accepts only shift ∈ {1, 2, 3} and
maps anything else to the nearest. These are its 8-entry tables:

| shift | timesteps |
|---|---|
| 1 | 1, .875, .75, .625, .5, .375, .25, .125 |
| 2 | 1, .9333, .8571, .7692, .6667, .5455, .4, .2222 |
| **3** | **1, .9545, .9, .8333, .75, .6429, .5, .3** |

**The shift default is a trap.** `generate_audio` defaults to 3.0, but the
pipeline's `GenerationParams.shift` defaults to 1.0 and is passed through
as-is. `docs/en/INFERENCE.md` says to set 3.0 for turbo "manually". We use
**3** (decision 3).

**DCW is on by default for turbo** (`_resolve_dcw_enabled`: `None` → the
model's `is_turbo`). DCW is a CVPR 2026 sampler correction, and the HF
card's copy of the model file predates it. After each Euler step:

- take a one-level **Haar** DWT along time of both `x_next` and
  `denoised = x − v·t`;
- push the low band by `t·0.05·(xL − yL)` and the high band by
  `(1−t)·0.02·(xH − yH)` ("double" mode);
- invert the DWT.

This is host-side arithmetic on a `[T, 64]` tensor.

### The conditioning (`AceStepConditionEncoder`, width 2048)

- **Caption → Qwen3-Embedding-0.6B `last_hidden_state`** (final-normed, all
  tokens, ≤ 256). The input is the upstream template, verbatim:

  ```
  # Instruction
  Fill the audio semantic mask based on the given conditions:

  # Caption
  {caption}

  # Metas
  {metas}<|endoftext|>
  ```

  Then `text_projector` (1024 → 2048, no bias). **`embed.GPU.Hidden`
  already returns exactly this tensor.**
- **Lyrics → the same encoder's `embed_tokens` only** (a table lookup, no
  transformer). The text is `# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>`,
  ≤ 2048 tokens. That goes through the **lyric encoder**: Linear 1024 →
  2048, then 8 bidirectional Qwen3-style layers (16 heads over 8, head 128,
  SwiGLU 6144, sliding window on even layers as above), then a final norm.
- **Timbre**: reference-audio latents, or with none `silence_latent[:750]`
  (30 s of the shipped silence). They pass through Linear 64 → 2048 and a
  prepended CLS token, then 4 layers, and **the CLS row** is the timbre
  embedding: one token.
- **Packing**: `[lyrics | timbre]` then `[… | text]`, valid rows first. With
  one request and no padding, that is plain concatenation. The cross-attention
  sequence is ≤ 2048 + 1 + 256 rows.
- **Context latents**: `src_latents` is `silence_latent[:T]`, or the LM
  hints when codes exist. `chunk_masks` is **1.0 everywhere**. The default
  `chunk_mask_mode="auto"` means to write 2.0 ("model decides"), but it
  writes it into a *bool* tensor, so it becomes True and then 1.0 when
  `prepare_condition` casts it (A1, from the oracle; this corrects the
  first draft of this plan).
- `silence_latent.pt` is `[1, 64, 15000]` fp32 (10 min, channel-first on
  disk; absmax 6.7, mean −0.21, std 0.96). A1 converts it once to
  safetensors, so the Go side reads no pickle.

### The 5 Hz LM (`acestep-5Hz-lm-4B`)

`Qwen3ForCausalLM` with the dimensions of Z-Image's text encoder: hidden
2560, 36 layers, 32 q heads over 8, head 128, SwiGLU 9728, θ 1e6, tied
embeddings. The vocabulary is 217,204. Ids 151,669–217,203 are
`<|audio_code_0…65534|>`, but only 0–63,999 are valid codes (FSQ levels
8·8·8·5·5·5 = 64,000). It uses the Qwen3 chat template with a system prompt
`# Instruction\n{instruction}\n\n` and a user turn
`# Caption\n{caption}\n\n# Lyric\n{lyrics}\n`.

- Phase 1 stops at `</think>`.
- Phase 2 reopens the assistant turn with the CoT and `\n\n` (uncond:
  `NO USER INPUT` and `<think>\n\n</think>\n\n`) and generates codes to EOS.
- The codes are mapped through `ResidualFSQ.get_output_from_indices` to
  2048-d, then the **detokenizer** runs: each code is repeated ×5 plus 5
  learned tokens, 2 encoder layers run over each 5-row patch, and
  `proj_out` gives 64 channels. The result is 25 Hz hints, cropped to `T`.

**The constrained-decoding FSM is 2,339 lines.** How much of it is live in
the default path (the field order, digit-only bpm, duration-forced EOS,
code-only phase 2) is A-o5. A port must reproduce whatever the default path
actually masks.

### The VAE (`AutoencoderOobleck`, diffusers)

48 kHz stereo, `downsampling_ratios [2,4,4,6,10]` (hop 1920 → **25 latents
a second**), 64 latent channels. The decoder is:

- conv1 (k7, 64 → 2048);
- five blocks, each Snake → weight-normed ConvTranspose1d (k = 2·stride)
  → three residual units (dilations 1, 3, 9: Snake, k7 conv, Snake, k1
  conv), with channels 2048 → 1024 → 512 → 256 → 128 → 128;
- a final Snake and conv2 (k7, 128 → 2, no bias).

Every conv is weight-normed, so we fold `g·v/‖v‖` at load. Snake uses
per-channel α, β (log-scale). By our estimate that is ~120 GFLOP a second of
audio, most of it in the ×2 and ×4 stages at 24–48 kHz: **~29 TFLOP for a
4-minute song**. That is too much for the CPU (H3's audio VAE ran at ~67
GFLOP/s), so the decode goes on the device.

Upstream decodes in **overlap-discard tiles** (64-latent overlap) to bound
VRAM. The untiled decode is the defined answer, and A5 measures what tiling
changes.

## What it costs on this machine (estimate, to be replaced by measurement)

**DiT.** A token costs 7.8 GFLOP in GEMMs (3.9 B params a token: self-attn
26 M, cross q/o 21 M, SwiGLU 75 M, per layer). On top of that:

- full attention costs `16 layers × 4·L²·4096`;
- the windowed layers are negligible;
- cross-attention costs `32 × 4·L·L_enc·4096`, with L_enc ≈ 700 for a
  typical lyric sheet.

At the image DiT's ~30 TFLOP/s:

| song | tokens | TFLOP a forward | s a forward | 8 forwards |
|---|---|---|---|---|
| 30 s | 375 | 3.1 | 0.10 | 0.8 s |
| 60 s | 750 | 6.3 | 0.21 | 1.7 s |
| 2 min | 1,500 | 12.8 | 0.43 | 3.4 s |
| 4 min | 3,000 | 26.9 | 0.90 | 7.2 s |
| 10 min | 7,500 | 76 | 2.5 | 20 s |

Short songs will run below 30 TFLOP/s: 375 rows is a thin GEMM.

**LM.** Phase 2 is 5 tokens a second of song, CFG as two rows sharing every
weight read. A dense 4 B decode reads 8 GB a token in fp16 and 4 GB in int8.
At the ~200 GB/s this machine sustains, that is ~25 and ~45 tokens/s, so a
**4-minute song is 27–48 s of codes**, and phase 1 adds ~150–400 CoT tokens.
**In the default (thinking) path, the LM is most of the request.** Its
decode speed is the vertical's main performance lever, and the LLM
vertical's decode work is what we have to draw on (A-o2).

**VAE.** ~120 GFLOP a second of audio. At 15 TFLOP/s of conv that is ~2 s
for 4 minutes.

**Activations** at 7,500 tokens: SwiGLU's input `[7500, 19456]` fp16 is
0.29 GB, so nothing comes near the 4 GiB descriptor range. Attention is
flash, so no scores are materialised.

## Decisions (so future sessions don't relitigate)

1. **Target: `text2music` with thinking, as upstream serves it by default.
   Build the DiT-only path first.** It is a strict subset: the LM's
   outputs are metas (text) and codes (hints), and both enter the DiT
   through inputs that the DiT-only path already builds. Cover, repaint,
   reference-audio timbre and the other task types come last (A11).
2. **The oracle is upstream's own code, in its own pinned venv.**
   `.venv-acestep` has torch 2.10 CPU, transformers **4.57.1**, diffusers
   0.37.0, vector-quantize-pytorch and pytorch-wavelets. It is gitignored,
   and created as in § Handoff. Upstream pins `transformers<4.58`. Under the
   repo's `.venv` (transformers 5.17), the model's `__init__` fails:
   `ResidualFSQ` calls `.item()` while transformers 5 builds modules on the
   meta device. The DiT loads in fp32 in 2.5 s and a forward is CPU-cheap
   (≤ 76 TFLOP), so **every oracle is plain fp32 torch** with no folding
   tricks. Upstream runs **bf16**, so a quantised or fp16 path is priced
   against a bf16 run of the same oracle, not only against fp32 (the lesson
   of H3's M11a).
3. **Upstream's pipeline defaults, not the model file's.** 8 steps, **shift
   3** (the documented turbo value; see § The turbo sampler), Euler ODE,
   **DCW double on**, `chunk_masks` 1.0 (what "auto" becomes), LM temperature 0.85, top-p 0.9,
   CFG 2.0 in phase 2 only, constrained decoding on, −1 dB peak
   normalisation. Where upstream's default and its docs disagree, the doc
   records which one we took and why.
4. **fp16 on the device, checked and not assumed.** A0 records every
   weight's absmax. The activation question (does the residual grow the way
   H3's did?) is A-o1, answered by per-layer dumps.
5. **Reuse before writing:**
   - the text encoder is `embed` (same bytes);
   - the tokenizer is `zimage/tokenizer`;
   - the DiT and encoders run on the image/video DiT's GEMM and WMMA
     attention kernels, plus a window bound and GQA;
   - the LM starts from `zimage/qwen`'s Qwen3-4B, which has the same
     dimensions (prefill), and adds decode.
6. **Our own RNG, with noise injectable for gates.** Upstream draws with a
   CUDA `torch.Generator`, which we cannot reproduce anyway. Every
   end-to-end gate feeds the oracle's noise, and the LM gates feed the
   oracle's sampled tokens (teacher-forced).
7. **Audio out through ffmpeg** (`/usr/bin/ffmpeg`): float PCM in; wav,
   flac and mp3 out. We write no encoder of our own.

## The stages

| # | Stage | State |
|---|---|---|
| A0 | Weights, oracle venv, fp16 weight audit, the silence latent to safetensors | **done 2026-09-27**: no weight in the DiT, LM or VAE overflows fp16 (absmax 31.6 / 43.8 / 5.6); the oracle is upstream's handler on the CPU |
| A1 | The plan in Go (`ace/plan`): prompt, lyric and metas formatting, tokenisation, latent/token counts, the three schedules, DCW's Haar step, the FSQ codebook | **done 2026-09-27**: six requests' prompts, token ids and latent lengths exact; the whole upstream sampler loop (3 shifts × DCW on/off × odd/even T) **bit-exact**; noise within 1 ulp; FSQ codes exact |
| A2 | Conditioning on the GPU: caption through `embed`, the lyric lookup, lyric and timbre encoders, packing, cross-K/V | oracle done (`dump_ace_dit.py`) |
| A3 | DiT on the GPU: GEMMs, windowed/full self-attention, cross-attention, AdaLN; teacher-forced per layer and per step against the fp32 oracle; per-step profile | oracle done; **A-o1 answered: fp32 residual, fp16 operands** |
| A4 | Detokenizer: codes → 25 Hz hints | |
| A5 | VAE decoder (Oobleck) on the GPU: folded weight norm, Snake, conv/convT; fp16 question; tiled vs untiled | |
| A6 | End to end, DiT only: caption + lyrics → wav/mp3 (`cmd/ace`), against the oracle's run with its noise | |
| A7 | The 5 Hz LM: Qwen3-4B prefill + KV-cached decode, CFG rows, sampling, the constrained FSM | |
| A8 | End to end with thinking: CoT → codes → hints → DiT → VAE | |
| A9 | Serve: the endpoint (A-o3), queueing, yielding the device between DiT steps and LM chunks | |
| A10 | Performance: the LM's decode first (int8 bank?), then the DiT's small-M GEMMs and the VAE's convs | |
| A11 | Other tasks (optional): reference-audio timbre, cover, repaint, via the VAE encoder and the audio tokenizer | |

### A0 — weights, environment, oracles

Oracles, each its own script and manifest, as `dump_h3_*` were. They run
under `.venv-acestep` and write to `reference/out/ace*`:

- `dump_ace_ranges.py`: every tensor's absmax and fp16 subnormal/flush mass,
  across the DiT, LM and VAE. (A0)
- `dump_ace_plan.py`: formatted prompts, token ids, the three schedules, a
  DCW step on known tensors, and FSQ `get_output_from_indices` on known
  codes. No big weights. (A1)
- `dump_ace_cond.py`: text hidden states, lyric embeddings, the lyric and
  timbre encoders' outputs, the packed sequence, context latents. (A2)
- `dump_ace_dit.py`: one forward's per-layer outputs, then per-step latents
  of a full 8-step run with its noise, at 30 s and at 4 min, fp32 and bf16.
  (A3)
- `dump_ace_detok.py` (A4), `dump_ace_vae.py` (A5, stagewise, tiled and
  untiled), and `dump_ace_run.py` (A6: the whole DiT-only request).
- `dump_ace_lm.py`: prompts, teacher-forced logits and the FSM's masks along
  a recorded sampled run. (A7)

Gate: every oracle run twice, byte-identical.

### A0 — result (2026-09-27)

`reference/dump_ace_ranges.py` (run twice, byte-identical). **No weight
overflows fp16 in any component.** The largest are q/k-norm weights (DiT
layer 0: 31.6; LM layer 0 `k_norm`: 43.8), and the VAE's largest is 5.6.
Subnormal mass is small everywhere, and the heaviest is the FSQ
`project_out` bias at 11%. That tensor is tiny and stays fp32. The only
flushes are 0.1% of one Snake β, which is stored in log scale (Snake uses
exp β), so flushing it changes nothing. As with H3, the fp16 question is
about activations (A-o1).

`reference/convert_ace_silence.py` writes
`models/acestep-v15-xl-turbo/silence_latent.safetensors`: `[15000, 64]`
fp32, time-major, as upstream uses it after its `transpose(1, 2)`.

**The oracle is upstream's own handler, which needs its own checkpoint
directory.** `AceStepHandler.initialize_service` compares the checkpoint's
model code with `acestep/models/xl_turbo/` and **overwrites the
checkpoint's copy** if they differ. Pointed at our HF clone, it replaced
the clone's `modeling_acestep_v15_xl_turbo.py` (restored with `git
checkout`). The upstream copy is the one with the DCW and flow-edit hooks.
The code upstream actually runs is therefore the *upstream* file, not the
HF card's. `models/ace-checkpoints/checkpoints/` holds:

- symlinks to the weights, the VAE and the LM;
- a text-encoder directory: the Ace-Step1.5 configs, and our
  `Qwen3-Embedding-0.6B/model.safetensors` (same bytes);
- a copy of upstream's model code, so the sync is a no-op.

Everything `.venv-acestep` needed beyond decision 2's list: torchaudio 2.10
(CPU), peft, lightning, diskcache, numba, toml, accelerate, matplotlib,
lycoris-lora, typer-slim and tensorboard. The handler imports the training
stack.

### A1 — result (2026-09-27)

`reference/dump_ace_plan.py` (run twice, byte-identical) drives
`acestep.inference.generate_music` with thinking off, and with shift 3 and
seed 42 given explicitly. It uses six requests: full metas, instrumental,
all defaults, Chinese lyrics at 12.34 s, a 3 s request, and a 256-token
caption with 563-token lyrics. It captures what reaches `generate_audio`
and stops. It then reruns the **real** `generate_audio` with the decoder
swapped for a stub (`v = 0.5·x + t·c`) and records the whole sampler. Last,
it decodes FSQ indices. `ace/plan` (`go test ./ace/plan`) reproduces all of
it:

| gate | result |
|---|---|
| caption prompt, lyrics prompt, 6 requests | exact strings |
| caption and lyric token ids | exact (`zimage/tokenizer` + `plan.Tokens`) |
| latent length | exact: 750, 1500, 3000, 307, 128, 5000 |
| sampler: shifts 1/2/3 × DCW off/on × T = 37/40 | **bit-exact**, all 12 runs |
| seed-42 noise (CPU `torch.Generator`) | within 4.8e-7 (`vae.TorchRandn`, 1 ulp of Sleef) |
| FSQ codebook rows / projected outputs | exact / within 7.5e-6 |

What had to be reproduced rather than assumed:

- **The text encoder's tokenizer appends `<|endoftext|>`.** Its
  post-processor does this for Qwen3-Embedding's last-token pooling, on top
  of the one written into the prompt. Truncation keeps room for it: a
  256-token caption keeps 255 ids plus the appended token, and loses the
  prompt's own `<|endoftext|>`.
- **Upstream's duration rules.** The metas take `int(duration)` of the raw
  request, while the length takes `round(duration, 1)`: 12.34 s gives 307
  latents under "12 seconds". A missing duration means **120 s** of audio
  (`padding_utils`'s fallback, issue #929) under metas that still say "30
  seconds". The target is never shorter than 128 latents (5.12 s).
- **Turbo ignores `inference_steps` and `guidance_scale`.** The handler
  passes `infer_steps`, which the model's `generate_audio` swallows in
  `**kwargs`. The schedule is always the 8-entry table.
- **DCW's Haar analysis is an FMA.** oneDNN's direct conv rounds the first
  tap's product and fuses the second: `fma(h[1], x[2k+1], h[0]·x[2k])`.
  Rounding both products first is 2 ulp off in ~9% of values. The synthesis
  rounds each product. On the last step DCW is applied to `x0` against the
  same `x0`, so all it does there is round-trip through the transform.
- **FSQ is symmetric.** ResidualFSQ builds its FSQs with
  `preserve_symmetry=True`, so a digit is `d·(2/(L−1)) − 1` (8 levels: −1,
  −5/7, …, 1), not `(d − ⌊L/2⌋)/⌊L/2⌋`.

### A2/A3 — the oracle, and the fp16 answer (2026-09-27)

`reference/dump_ace_dit.py` (run twice, byte-identical, 197 MB). It
captures A1's requests at the DiT's door and runs upstream's own
`generate_audio` in fp32 on the CPU, with hooks. Two requests:

- `full_metas`: 30 s, T = 750, 375 tokens, 121 condition rows (57 lyric +
  1 timbre + 63 caption); 26 s on the CPU;
- `defaults`: 120 s, T = 3000, 1,500 tokens, 71 rows; 83 s.

For each it dumps:

- the text projector, every lyric and timbre layer, the packed condition
  sequence and the context latents;
- the timestep tables, the patch and condition embeddings, and layers
  0/1/2/15/31 at forward 0;
- every forward's input latents and velocity, and the final latents with
  the noise;
- the absmax of every GEMM operand in every layer at every forward.

**A-o1, answered:**

| tensor | peak | where | on the device |
|---|---|---|---|
| residual | **5.7e5** (4.4e5 at 30 s) | passes 65,504 at layer 10 of every forward | **fp32**, never narrowed |
| cross-attention output projection's output | 8.5e3 | layers 21/27 | fp32 GEMM output |
| FFN down projection's output | 6.0e3 | layer 14 | fp32 GEMM output |
| self-attention output projection's output | 5.1e3 | layer 25 | fp32 GEMM output |
| SwiGLU product (down projection's input) | **2.45e3** | layer 14 | fp16, **no scale needed** (H3 needed ×1/16) |
| attention contexts (o_proj inputs) | ≤ 329 | | fp16 |
| normed GEMM inputs (q/kv, cross q, gate/up) | ≤ 67 | | fp16 |

So the plan is H3's shape without its scale factor: an fp32 residual end to
end, fp16 everywhere a GEMM reads. The residual grows over the steps
(3.5e5 at forward 0 to 4.4e5 at forward 7 for 30 s), so the gate has to
cover every forward, not only the first.

## Open questions

- **A-o1: does fp16 hold in the DiT's activations?** Answered above: fp32
  residual (5.7e5), fp16 operands (≤ 2.45e3).
- **A-o2: the LM's decode path.** Options:
  - a dense GQA decode on `zimage/qwen`'s arena (new GEMV kernels);
  - borrow the LLM vertical's attention/decode kernels.

  The quantisation question (int8 bank for bandwidth) is separate. The LM is
  most of a thinking request's time, so this is the vertical's main lever.
- **A-o3: the served door.** OpenAI has no music endpoint. Upstream ships
  two:
  - its own async `/release_task` + `/query_result`;
  - an OpenAI-chat-compatible server (`openrouter/`): `/v1/chat/completions`
    whose reply carries the audio, plus `audio_config`, `lyrics` and
    `thinking` fields.

  Our `/v1/chat/completions` belongs to the LLM. To decide at A9 against
  OpenAI's reference ([[api-follows-openai-standard]]); ask before
  inventing.
- **A-o4: resident or staged per request?** It is ~19.5 GB in fp16, beside
  the image and video models on the non-LLM machine.
- **A-o5: how much of the constrained-decoding FSM is live** in the default
  thinking path, and what it masks.
- **A-o6: does untiled decode differ audibly from upstream's tiled one?**
  (A5)

## Handoff

**2026-09-27, session 1: opened; A0 and A1 done, A2/A3's oracle written, A-o1 answered.** The plan is read from
the model files, headers and upstream code, and corrected where the A1
oracle disagreed (chunk masks are 1.0, not 2.0).

- Code: `ace/plan` (A1): prompts, metas, tokens, latent length,
  schedules, `Sample` (Euler + x0 + DCW), FSQ.
- Oracles: `reference/dump_ace_ranges.py` (A0),
  `reference/convert_ace_silence.py` (A0),
  `reference/dump_ace_plan.py` (A1; needs `HF_HUB_OFFLINE=1`; ~1 min),
  `reference/dump_ace_dit.py` (A2/A3 and A-o1; imports the plan script's
  cases; ~2 min). Outputs are in `reference/out/ace{ranges,plan,dit}`.
- The oracle's checkpoint directory: `models/ace-checkpoints/checkpoints`
  (see § A0 result; **never point the handler at the HF clones**).

- Weights, all gitignored under `models/`:
  - `acestep-v15-xl-turbo/` (git-LFS clone; includes `silence_latent.pt`
    and the HF `modeling_acestep_v15_xl_turbo.py`);
  - `acestep-5Hz-lm-4B/` (git-LFS clone);
  - `Ace-Step1.5/` (`vae/`, the text encoder's configs);
  - `ACE-Step-1.5-src/` (upstream pipeline at `ca1e85f`; its
    `acestep/models/xl_turbo/` model file is the HF one plus DCW and
    flow-edit hooks).
- Oracle env:
  `uv venv --python 3.12 .venv-acestep`, then
  `uv pip install --python .venv-acestep/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.10.0`
  and `uv pip install --python .venv-acestep/bin/python transformers==4.57.1 diffusers==0.37.0 vector-quantize-pytorch einops loguru pyyaml numpy safetensors soundfile scipy pytorch-wavelets PyWavelets`.
  (`vector-quantize-pytorch` also went into `.venv`, harmlessly.)

**Next, in order:**

1. **The attention kernel's two new bounds.**
   `shaders/dit_attention_wmma.comp` already does GQA, and a query count
   different from the key count (the REL_BIAS split). It needs two more
   things:
   - a **±128 window** (skip key blocks outside it, mask inside it, with the
     row max masked too, as its CAUSAL build does);
   - **cross-attention planes** whose key count and plane stride differ
     from the queries'.

   The encoders need the window and the DiT needs both. This is the one
   new piece of kernel work before A2/A3 are plumbing.
2. A2 on the GPU (`ace/cond`?):
   - `embed.GPU.Hidden` for the caption, gated against
     `reference/out/aceplan/*_text_hidden.bin`;
   - the lyric lookup (`*_lyric_embeds.bin`);
   - the lyric (8-layer) and timbre (4-layer) encoders against
     `reference/out/acedit/*_{lyric,timbre}_layer*.bin`, and the packed
     `*_encoder_states.bin`.
3. A3 (`ace/dit`): on H3's block machinery (fp32 residual, per-run AdaLN
   vectors, RMS pre-norms, q/k norm + full-width RoPE, SwiGLU), plus the
   cross-attention K/V computed once a request. Gate teacher-forced per
   layer and per forward against `acedit`, as H3's M7 did.
