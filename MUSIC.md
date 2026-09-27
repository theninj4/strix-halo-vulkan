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

Both model repos were already present in `models/` as git-LFS clones, so
every weight was on disk twice. By 2026-09-27 11:35 their `.git`
directories had been removed (19 GB + 7.9 GB remain). **So nothing in them
can be restored with `git checkout` any more**, which is one more reason
never to point the upstream handler at them (§ A0 result). The rest was
fetched with:

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
| A2 | Conditioning on the GPU: caption through `embed`, the lyric lookup, lyric and timbre encoders, packing, cross-K/V | **done 2026-09-27** (`ace/dit`): both encoders ≤2.2e-4 rms a layer, the packed sequence ≤3.1e-4; the caption's `embed` pass is wired in A6 |
| A3 | DiT on the GPU: GEMMs, windowed/full self-attention, cross-attention, AdaLN; teacher-forced per layer and per step against the fp32 oracle; per-step profile | **done 2026-09-27**: layers ≤2.5e-4 rms except layer 0 (3.8e-3, fp16 q/k); latents 5–9% rms from fp32, **3.6–16× closer than upstream's own bf16**; 8 forwards 0.97 s (30 s) / 6.9 s (4 min) / 19.1 s (10 min) |
| A4 | Detokenizer: codes → 25 Hz hints | **done 2026-09-27** (`dit.GPU.Detokenize`): real codes to hints within 1.3e-4 rms, whole or chunked; a `GROUP` attention build (block-diagonal 5-row sequences) |
| A5 | VAE decoder (Oobleck) on the GPU: folded weight norm, Snake, conv/convT; fp16 question; tiled vs untiled | **done 2026-09-27** (`ace/vae`): every stage ≥59 dB SNR against fp32, whole songs 60–65 dB; 30 s in 0.29 s, 2 min in 1.19 s; tiling is free (A-o6) |
| A6 | End to end, DiT only: caption + lyrics → wav/mp3 (`cmd/ace`), against the oracle's run with its noise | **done 2026-09-27** (`ace/pipeline`, `cmd/ace`): seed-42 noise 1e-7, caption hidden ≤2e-3 rms, latents 3.6–17× closer to fp32 than upstream bf16; **4 min of audio in 9.6 s** |
| A7 | The 5 Hz LM: Qwen3-4B prefill + KV-cached decode, CFG rows, sampling, the constrained FSM | **done 2026-09-27** (`ace/lm`): prompts and 3,100/3,100 fuzzed CoTs exact; teacher-forced logits over 101 steps of both phases ≤2e-3 rel, KL ≤1.4e-5, every argmax; the FSM exact over 857 steps; top-p survivors exact (one fp32 boundary); 45 ms a step for 1 or 2 rows |
| A8 | End to end with thinking: CoT → codes → hints → DiT → VAE | **done 2026-09-27** (`ace/pipeline`, `cmd/ace` thinks by default): from the oracle's CoT and codes, caption states ≤1.0e-3 rms, lyrics exact, hints ≤1.6e-4; **60 s of song in 26 s**, 90% of it the LM |
| A9 | Serve: the endpoint (A-o3), queueing, yielding the device between DiT steps and LM chunks | **done 2026-09-27** (`serve -music`, `/v1/music`): submit-then-poll jobs in `/v1/videos`' shape, ACE-Step's `release_task` fields read; estimates within 10% of the run; speech beside a song waits ≤ one LM step or DiT forward |
| A10 | Performance: the LM's decode first (int8 bank?), then the DiT's small-M GEMMs and the VAE's convs | |
| A11 | Other tasks (optional): reference-audio timbre, cover, repaint, via the VAE encoder and the audio tokenizer | |
| A12 | Sample mode: from a description alone the LM writes the caption, metas, genres and lyrics (upstream's `sample_query` / "simple mode"), then the thinking path runs on them | **done 2026-09-27** (`ace/lm` `Sample`, `pipeline.Options.Sample`, `sample_query` on `/v1/music`, `cmd/ace -sample`): host side exact on 358 hints, 24 prompts, 20 lyric cuts; the FSM exact over 2,001 scripted and 1,208 real steps; upstream's own song parsed field for field and its codes prompts rebuilt id for id; teacher-forced KL ≤ 5.7e-5, every argmax; served, 45 s of song in 31 s |

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

### A2/A3 — result (2026-09-27)

`ace/dit` (`go test ./ace/dit`, ~35 s, stages everything once at 7,500
tokens: 10.1 GB of fp16 weights). It holds the lyric encoder, the timbre
encoder and the DiT as one layer graph (h3/dit's machinery without row
chunks or runs):

- **The encoders are the DiT layer without modulation**: a = the norm
  weight, b = 0, a plain residual add, no cross-attention.
- **Cross-attention K/V are computed once a request** (`Begin`):
  `condition_embedder`, then each layer's k_proj → k_norm (no rotation) and
  v_proj, packed into that layer's planes, ~10 MB a layer.
- **Three new attention builds** (`shaders/ace_attn_*`, two new flags on
  `dit_attention_wmma.comp`; **every existing build's SPIR-V is
  byte-identical**, checked against all 40 of them):
  - `WINDOW`: the ±128 band. The key loop covers only the blocks the band
    reaches, and the band is masked on the row max as well as on P. A row
    with no key yet keeps its max at 0, not −inf, so no 0/0.
  - `CROSS`: key and query counts split, and the kv planes' own row stride
    (`pc.ldb`).
  - The full build is GQA with `TAIL_MAX`.
- **The q/k pack** is H3's with `ROPE_WIDTH=128` (Qwen3's full-width
  rotate-half). Cross q/k read identity tables.
- The final norms (lyric, timbre CLS), the timestep MLPs (1e-6 against the
  oracle) and the output's bias run on the host in fp32.

Gates, teacher-forced from the fp32 oracle:

| what | full_metas (375 tokens) | defaults (1,500) |
|---|---|---|
| lyric / timbre layers, worst rms | 2.2e-4 / 5.5e-5 | 4.3e-5 / 5.5e-5 |
| packed condition sequence | 5.3e-4 max, 3.1e-4 rms | 9.2e-5 max, 3.0e-4 rms |
| DiT layer 0 | 8.1e-3 max, 3.8e-3 rms | 9.4e-3, 2.3e-3 |
| DiT layers 1, 2, 3–15, 16–31 | ≤3.8e-4 max, ≤2.2e-4 rms | ≤9.2e-4, ≤2.5e-4 |
| velocity, forward 0 (t = 1) | 6.7e-2 rms | 2.9e-2 |
| velocity, forwards 1–7 | 1.2e-3 – 2.7e-2 | 1.3e-3 – 1.6e-2 |
| final latents, 8 steps from the oracle's noise | 8.6e-2 rms | 5.2e-2 |
| **upstream bf16's** v0 / latents, same noise | 1.85e-1 / 3.08e-1 | 3.94e-1 / 8.24e-1 |

**Layer 0 is the whole of the error.** A host fp32 layer 0 matches the
oracle to 8e-6. Rounding only its q, k and v to fp16 gives 8.2e-3, the
device's error exactly. Rounding the normed input, the context, the SwiGLU
product or P instead costs ≤3e-4 each. The cause is that layer 0's q/k-norm
weights reach 31.6, so its logits are large and fp16's 11 bits move the
softmax. That error rides the residual's outlier channel (5.8e3 after layer
0) through every later layer's norm.

**It is well inside what upstream ships.** `dump_ace_dit.py --bf16` runs
the same requests from the same noise the way upstream serves them on CUDA
(`init_service_orchestrator`: bf16 on CUDA; **fp32 on ROCm**, i.e. this
machine, and on the CPU). It even rounds the timesteps to bf16 (0.953125 for
0.9545). The bf16 run drifts 2.8–16× further from fp32 than we do, and the
test holds ours under half of it (`underBF16`). A split-precision QK for
layer 0 only (hi/lo fp16 planes, three products a score tile) would remove
the error for ~1/64 of the attention time. It is not needed for the gate;
it is recorded as A-o7.

Timing, one forward with the device to itself, best of 3:

| song | tokens | a forward | 8 forwards | estimate |
|---|---|---|---|---|
| 30 s | 375 | 121 ms | 0.97 s | 0.8 s |
| 2 min | 1,500 | 403 ms | 3.22 s | 3.4 s |
| 4 min | 3,000 | 859 ms | 6.87 s | 7.2 s |
| 10 min | 7,500 | 2,389 ms | 19.1 s | 20 s |

At 4 minutes the GEMMs are 77% of the forward and run at 37–41 TFLOP/s.
Self-attention is 9% at 32 TFLOP/s, cross-attention 1% at 22, SwiGLU 6%,
and the norms, packs and residual adds the remaining ~12%.

### A4 — result (2026-09-27)

`reference/dump_ace_detok.py` (run twice, byte-identical) runs upstream's
`_decode_audio_codes_to_latents` in fp32. Its codes are real: the fp32 DiT
oracle's latents through the model's own audio tokenizer (`tokenize`), 150
and 600 codes. The handler's entry from the code string equals the direct
call exactly.

The detokenizer runs **every code as its own 5-row sequence**: the code's
projection ×5 plus 5 learned tokens, RoPE positions 0–4, then 2 encoder
layers, the norm and proj_out 2048 → 64. `dit.GPU.Detokenize`:

- FSQ digits and project_out on the host (`plan.FSQOutput`);
- embed_tokens as a GEMM;
- the ×5 and the learned tokens on the host;
- the two layers as `ace/dit` encoder layers with `group = 5`;
- the norm and proj_out on the device.

It runs in chunks of `rows/5` codes, since a 10-minute song is 15,000 rows,
twice the DiT's planes.

- **`GROUP`** is a fourth flag on `dit_attention_wmma.comp` that rides
  `WINDOW`'s machinery. The band predicate becomes `kj/G == qi/G`, the key
  range the query block's groups, and every block is masked. Every non-ACE
  build is still byte-identical; the window build's SPIR-V moved with the
  refactor and its gates are unchanged.
- **The rope tables restart every group**: a table at position `row mod 5`
  beside the plain one.

| gate | full_metas (150 codes) | defaults (600) |
|---|---|---|
| layer 1, teacher-forced | 4.3e-5 rms | 4.0e-5 |
| hints, whole | 2.3e-4 max, 1.3e-4 rms | 3.1e-4, 1.2e-4 |
| hints, 97-code chunks | the same | the same |

Device time is small (the whole test is 0.4 s with both cases twice). The
hints are the DiT's source latents on the thinking path (A8).

### A5 — result (2026-09-27)

`reference/dump_ace_vae.py` (~45 s, run twice, byte-identical) decodes the
fp32 DiT oracle's latents with diffusers' `AutoencoderOobleck` in fp32. It
dumps per stage on a 4 s excerpt, whole songs untiled and through
upstream's own tiling (`VaeDecodeChunksMixin`, chunk 512, overlap 64), and
the audio upstream writes.

`ace/vae` (`go test ./ace/vae`) runs every convolution on kokoro's
`A_CONV=1` GEMM (32×64, wave32; 32×32 for conv2's two channels). The
activations are channel-last fp16 in a zero-bordered arena. The pieces:

- **The transposed convolutions** are kokoro's 2-tap GEMM with an
  s-times-wider output.
- **The glue is one new pass**, `shaders/ace_vae.comp`: the preceding bias,
  Snake with β, the fp16 narrowing, and the zero rows the next
  convolution's taps and M padding read.
- **Weight norm is folded at load.** For a ConvTranspose1d it is over the
  *input* channel (`weight_g` is `[in, 1, 1]`).
- **The decode is tiled**: 512-latent windows with a 32-latent margin. The
  last stage is 1920 × 128 fp32 a latent, so a 4-minute song cannot be one
  buffer.

| gate | result |
|---|---|
| conv1, blocks 0–4, output (4 s excerpt) | 70.9, 63.6, 59.4, 63.1, 69.7, 69.4, 67.4 dB SNR |
| full_metas, 30 s, tiled vs the untiled oracle | 63.6 dB (max 4.9e-3 of peak) |
| defaults, 120 s, same | 65.0 dB |
| after the peak clamp and −1 dBFS | 60.6 / 63.9 dB |
| **A-o6: upstream's tiling vs untiled** | max 2.4e-6 / 8.3e-6 abs: tiling changes nothing |

Device time: 285 ms for 30 s, 1,189 ms for 120 s (~12 TFLOP/s on the
estimated 120 GFLOP a second of audio). One VAE staging is ~1.9 GB, most
of it the 512-latent window's activations.

What upstream writes, and it is the same everywhere: after the decode,
divide by the peak if it passes 1 (`generate_music_decode`); then
`normalize_audio` to −1 dBFS (`enable_normalization` is on by default).
`latent_shift`/`latent_rescale` default to 0 and 1, and the fades to 0.

### A6 — result (2026-09-27)

`ace/pipeline` puts it together:

- `ace/plan` formats the prompts and fixes the length;
- `embed.GPU.Hidden` (Qwen3-Embedding-0.6B, the same bytes) encodes the
  caption;
- the lyrics are rows of the same model's embedding table;
- `ace/dit` runs Encode, then Begin, then `plan.Sample` over `Step`, with
  seeded noise (`h3/vae.TorchRandn`, torch's CPU randn) and the context
  [silence | 1.0];
- `ace/vae` decodes, then Normalize.

`cmd/ace` writes wav/flac/mp3/opus through ffmpeg. `go test ./ace/pipeline`
(~20 s, 240 s ceiling):

| gate | full_metas (30 s) | defaults (120 s) |
|---|---|---|
| seed-42 noise vs the oracle's | 1.0e-7 of absmax | |
| caption last_hidden_state (also long_caption at 256 tokens, instrumental_60) | 1.0e-3 rms | 9.2e-4 (worst 2.0e-3, long_caption) |
| lyric embeddings | exact | exact |
| final latents vs the fp32 oracle | 8.5e-2 rms | 4.7e-2 |
| upstream bf16's own drift, same request and noise | 3.1e-1 (3.6×) | 8.2e-1 (17×) |
| −1 dBFS audio vs the fp32 oracle's | 14.1 dB SNR | 20.8 dB |

The audio SNR is the latent drift from layer 0's fp16 logits (A-o7)
carried through the decoder. It is not a VAE error: the VAE alone is ≥60
dB. Timings, request to samples with the device to itself (`cmd/ace`,
warm):

| song | text | encoders | DiT (8 steps) | VAE | total |
|---|---|---|---|---|---|
| 30 s | 14 ms | 49 ms | 1.10 s | 0.30 s | 1.47 s |
| 45 s | 15 ms | 46 ms | 1.58 s | 0.46 s | 2.10 s |
| 2 min | 15 ms | 41 ms | 3.41 s | 1.23 s | 4.70 s |
| 4 min | 16 ms | 42 ms | 7.08 s | 2.49 s | 9.63 s |

`cmd/ace` for 4 minutes is 18.8 s of wall time, ~9 s of it staging 17 GB
of fp32 checkpoint into 10 GB of fp16 (page cache warm). A 45 s mp3 comes
out at 48 kHz stereo with its peak at −1.0 dB (volumedetect). Nobody has
listened yet. The gate is the oracle's audio at 14–21 dB SNR, with the
drift priced against bf16.

### A7 — the oracle, and what the FSM masks (2026-09-27)

`reference/dump_ace_lm.py` runs upstream's `generate_music` with thinking
on: `LLMHandler`, PyTorch backend, fp32 on the CPU, 75–170 s a case. It
stops at the DiT's door. At every sampling step it records the FSM state,
how many tokens the FSM alone allows (and which, when ≤ 64), how many are
left after top-p (and which), and the sampled token. It records the prompts
as text and ids, the CFG pair's padding mask, and the raw logits of both
rows at steps 0, 1, 2, 3, 10, 50 and 100 of each phase. It also dumps what
reaches the DiT. Two cases:

- `given_duration`: caption, lyrics, 30 s; the LM plans the rest.
- `all_metas`: A1's full_metas request.

**Upstream's LM sampling is not seeded**: two runs of `given_duration`
wrote 161- and 120-token CoTs. So this oracle is a trace to teacher-force
against, not a byte-reproducible run. The manifest holds the last run;
regenerating it moves every gate's tokens.

**A-o5, answered: the default path's FSM is small.**

| phase 1 state | what the FSM allows |
|---|---|
| `<think>`, field names (`bpm:`, `caption:`…), newlines, `</think>` | exactly one token (forced) |
| bpm value | a space, then 1–9, then 0–9, then newline (30–300 by range check) |
| caption value | **everything but the 64,001 audio-code tokens** (153,203 of 217,204); the model's own YAML folding (`\n  ` continuations) and its newline end the field |
| duration value | a user's duration, forced digit by digit; otherwise the LM's (10–600) |
| keyscale value | a root (C, A, D, B, F, E, G, Ab, Db, Eb as tokens), then `#`/`b`/`♯`/`♭`/` major`/` minor`, then newline |
| language value | forced here (` en`); to check against a request with other lyrics |
| timesignature value | a space, then 2/3/4/6, then newline |
| end | `<|im_end|>` (EOS) at `THINK_END_TAG` |
| **phase 2** | **exactly the 64,000 valid codes** until 5 × seconds codes, then EOS alone |

What a port must reproduce around it, all read from `llm_inference.py`:

- **Top-p (0.9) filters the untempered logits.** Temperature (0.85) is
  applied only inside `_sample_tokens`. The FSM's phase temperatures are
  unset by default. Repetition penalty is 1.0 and top-k is off.
- **CFG (2.0) runs in phase 2 only**, over the 64,000 code tokens:
  `uncond + 2·(cond − uncond)`, everything else −inf. The sampled token is
  appended to both rows.
- **The unconditional row** is the chat prompt with `NO USER INPUT` as the
  user turn and `<think>\n\n</think>\n\n`. It is *left-padded* with
  `<|endoftext|>` to the conditional row's length, attention-masked, and
  given no position ids. Its real tokens' positions are therefore shifted
  by the pad count, which RoPE makes a numerical no-op.
- **Phase 2's CoT is re-serialised, not phase 1's text.**
  `_format_metadata_as_cot` builds `{bpm, caption, duration, keyscale,
  language, timesignature}` (digit strings → ints, `N/4` → `N`) and emits
  it with `yaml.dump(allow_unicode=True, sort_keys=True)`. The caption's
  line folding is **PyYAML's plain-scalar emitter at width 80**, and a
  caption with `: ` or a leading indicator switches to a quoted style. A
  port needs that emitter for strings, fuzzed against Python.
- **With every meta given (`all_metas`), phase 1 is skipped**
  (`has_all_metas`). The CoT then holds only the user's bpm, duration,
  keyscale and timesignature: no caption and no language, and the DiT
  keeps the user's caption.
- The DiT reads the hints: `is_covers` is True, the source latents are the
  detokenized codes (A4), and the chunk mask stays 1.0.

**The plan for A7, in pieces:**

1. **A7a, prompts and prefill.** Build the phase prompts and tokens (the
   LM's tokenizer has 65k added tokens), exact against the dumped ids. The
   prefill is `zimage/qwen.GPUEncoder`, which is this model's shape: dense
   Qwen3-4B, causal GQA, prefill only. The lm_head is the tied 217k-row
   embedding. Gate: logits at step 0 of each phase, both rows.
2. **A7b, KV-cached decode for two rows (A-o2).** Nothing in the repo
   decodes a *dense* Qwen3 today: Kev is Qwen3.5 (GDN + attention) and the
   LLM is MoE. The candidate is the GPUEncoder's staged weights with
   narrow-M GEMM rungs at M = 2 plus a decode attention over a KV cache.
   At ~200 GB/s, 8 GB of fp16 is ~25 tokens/s, so 1,200 codes (4 min) is
   ~50 s. That makes int8 (A10) the lever. Gate: teacher-forced logits at
   steps 1–100 of both phases.
3. **A7c, sampling and the FSM** (the table above), the YAML emitter,
   phase 1 → phase 2 → `Detokenize` → the DiT's covers path (A8).

### A7 — result (2026-09-27)

`ace/lm` is the LM in four files:

- `prompt.go`: the phase prompts, `ParseCoT` (parse_lm_output), and
  `Meta.CoT`, a port of PyYAML's emitter for one top-level string value.
  It folds a plain scalar at width 80 and falls back to single or double
  quotes. It also runs the implicit-resolver check that writes the
  language `no` as `'no'`.
- `gpu.go`: the model on the device (below).
- `fsm.go`: the phase-1 FSM, state for state.
- `sample.go` and `planner.go`: top-p, the draw, the CFG/EOS rule, and the
  two phases.

**The device graph** is a dense Qwen3-4B with a KV cache. Nothing in the
repo decoded one before. `zimage/qwen` is prefill-only, Kev is GDN plus
attention, and the LLM is MoE. The pieces already existed, though, and they
all read the same staged weight:

- the projections are staged once, fp16 in the §2.8 fragment tiling. That
  is zimage/qwen's layout 2, which the DiT GEMM rungs read for a pass of
  many rows. It is also `llm_gemv.comp`'s fp16 bank, for 1–3 rows (split-K
  GEMV, `ROWS` specialised). So there is no second copy of 7.3 GB;
- the norms, SwiGLU and residual adds are the DiT's scalar kernels;
- two new shaders:
  - `ace_lm_prep.comp` does the q/k norm, the NeoX rope at each row's own
    position, and the KV write;
  - `ace_lm_attn.comp` is causal GQA over the cache, split over keys, plus
    its combine. Q carries 1/√128.

A row is (token, slot, position). Attention reads keys only from the cache,
so a prompt prefills in any number of passes. The CFG pair's two prompts
share one pass, and a decode step is one row per slot. A prefill leaves its
last token for the first step, the only pass that runs the head. The head is
the tied embedding tiled into a bank of its own, and it reads only the rows
asked for: the whole vocabulary in phase 1 (the FSM needs it), and
`<|im_end|>` through the last code in phase 2 (64,024 rows).

**Two things the gates found:**

- **Qwen3's massive activations reach fp16's ceiling.** On a newline of the
  phase-2 prompt, layer 16's SwiGLU product is 6.39e4, 2.4% under 65,504.
  Layer 6's is 1.2e4, and the residual reaches 1.1e6 (fp32). The product is
  narrowed at ×1/16 and the residual add undoes it, using the DiT add
  shader's per-column gate (H3's factor, VIDEO.md).
- **A race in the new prep shader.** Every thread read the norm's `red[0]`,
  and thread 0 then overwrote it for the rope exchange with no barrier in
  between. It NaN'd about one layer in twelve. It hid behind the overflow
  above and passed a tolerance gate most of the time. `TestDeterministic`
  now wants 8 runs of prefill + decode bit-identical.

Gates (`go test ./ace/lm`, ~40 s):

| gate | result |
|---|---|
| prompts: phase 1, phase 2 cond, uncond (both cases) | 5/5 rows' ids exact |
| phase-2 CoT against `yaml.dump` (`dump_ace_yaml.py`, 3,100 fuzzed metadata dicts: folding, quoting, `no`/`123`/`''`, unicode, control chars) | 3,100/3,100 exact |
| logits, teacher-forced from the oracle's sampled tokens, steps 0–100 of each phase | phase 1: ≤2.0e-3 rel, KL(T=0.85) ≤3e-8; phase 2 rows and CFG: ≤5.1e-4 rel, KL ≤1.4e-5; **every argmax agrees** |
| the FSM against upstream's processor, driven through 7 scripted CoTs (`dump_ace_fsm.py`: free, user metas injected, extremes, fields out of order, invalid values drifting into the free caption path) | 737 steps, every allowed set exact |
| the FSM along the oracle's real phase 1 | 120 steps exact |
| FSM + top-p on the oracle's own logits, every dumped step | 20/21 exact; 1 keeps 3,830 where torch keeps 3,829 (below) |
| 8 × (prefill + 4 decode steps) | bit-identical |

**The top-p boundary** is torch's rounding, not ours. Its fp32 softmax
denominator came out 8.7e-6 low (from its vectorised reduction), and that
pushed a token of probability 3.6e-5 across 0.9. Our sums are fp64. The gate
accepts one extra token only when the boundary sits within 2e-5 of p, and it
reports the case. Reproducing one CPU build's SIMD reduction order is not
worth it: upstream on CUDA reduces differently again.

**Phase 1 is upstream's FSM, quirks included:**

- a caption ends when the *raw* argmax after a newline is not indentation;
- the model then writes the next field name unmasked and may skip fields;
- the language is the top-1 candidate, forced;
- the request's metas are injected as the tokens of `" value\n"`.

One divergence: a `genres:` value is left free. Upstream constrains it to a
vocabulary file, but the field is skipped by default, so the state is
reachable only if the model writes `genres:` itself.

Speed, with the device to itself:

- a decode step is **45 ms for one row or two**: the CFG row is free, and
  that is ~187 GB/s over the 8.4 GB a step reads with the full head;
- the planner's step is 50 ms wall; the other 5 ms is host sampling;
- prefill is 60 ms for 86 tokens and 97 ms for the 239-row CFG pair;
- staging is 4.9 s.

A 30 s song is ~130 CoT steps plus 151 code steps, **~14 s of LM**. The
estimate was 25 tokens/s from 8 GB at 200 GB/s. That lever is A10's
(int8: 4 GB a step).

### A8 — result (2026-09-27)

`pipeline.Generate` with `Options.Think`, after `LoadLM`, runs phase 1 and
phase 2, then `DiTRequest` does upstream's `_update_metadata_from_lm`:

- the CoT's bpm, key, time signature and duration fill what the request left
  out;
- the rewritten caption replaces the request's when phase 1 ran;
- **the language stays the request's.** Upstream reads the CoT's `language`
  as `vocal_language`, a key its parser never writes, so the default
  `"unknown"` survives;
- **the instruction becomes the cover task's.** Audio codes switch
  text2music to `cover` (`_resolve_generate_music_task`), and its
  instruction is `Generate audio semantic tokens…`. With the default cover
  strength 1.0 and cover noise 0, that line is all the switch changes. It
  was worth one token of the caption prompt, and the gate caught it
  (132 rows against 131).

The length is the codes' (T = 5 per code, `conditioning_target`). The codes
through `Detokenize` are the source latents, padded or cropped with the
silence. The chunk mask stays 1.0.

`go test ./ace/pipeline -run TestThinkingInputs` rebuilds the DiT's inputs
from the oracle's own CoT and codes:

| | given_duration (phase 1 ran) | all_metas (skipped) |
|---|---|---|
| caption last_hidden_state | [131 1024], 1.00e-3 rms | [62 1024], 9.4e-4 |
| lyric embeddings | exact | exact |
| hints (150 codes → 750 latents) | 1.55e-4 rms | 1.38e-4 |

`cmd/ace` thinks by default (`-think=false` is A6's path). A 60 s request
(synthwave caption, the oracle's lyrics, `-duration 60`) is **26.0 s** from
staged models:

| LM phase 1 | LM phase 2 | text + encoders | DiT | VAE |
|---|---|---|---|---|
| 162 steps, 8.2 s | 300 codes, 15.1 s | 83 ms | 1.77 s | 0.62 s |

The mp3 is at −15.8 dB mean with a −1.4 dB peak, and its loudness is steady
across the minute. Its zero-crossing rate is 0.05 (noise would be 0.5). It
is `out/ace-think-synthwave-60s.mp3`, and nobody has listened to it yet.

### Listening (2026-09-27)

The user listened to `out/ace-think-synthwave-60s.mp3` (thinking, A8): **it
sounds great.** That is the first ear on the vertical's output, and it
closes the handoff's "listen" item.

### A9 — result (2026-09-27)

A-o3, answered by the user: **a submit-then-poll endpoint.** It is
`/v1/music`, in `/v1/videos`' shape (API.md § Music is a job):

- `POST` answers with a job;
- `GET /v1/music/{id}` polls it;
- `GET …/content` fetches the audio;
- `DELETE` cancels it;
- `GET /v1/music` lists the jobs.

**The request reads ACE-Step's own `release_task` fields**: the names, the
aliases, the nested `metas` object, JSON or a form. What that API offers and
this server does not run is a 400 naming the field: other tasks, reference
audio, batches, sample and format modes, top-k and the rest. Upstream's own
routes (`/release_task`, `/query_result`) are not served. Their `{data, code}`
wrapper, integer statuses and path-based file URLs would be a second door to
add if a client of upstream's server needs one.

The pieces:

- `api/music.go` (a queue after `videos.go`'s, kept separate rather than
  refactoring the served video one);
- `backend/music.go`, which is everything resident (~21 GB, A-o4: resident);
- `serve -music` (`-music-lm=false` serves only `thinking: false`, ~10 GB
  lighter).

What the job shows while it runs:

- `seconds` is null until the length is known;
- `metadata` is the DiT's inputs once the LM has planned them;
- `stage` is `think`, `codes`, `dit`, `vae` or `write`;
- `estimated_seconds` comes from the measured rates and is redone when the
  length is known.

`ace/lm`'s `Planner` and `pipeline.Options` gained hooks for this:
`Between` (yield the device and check for cancellation after every LM step
and DiT forward), `Progress` and `Planned`.

Measured live (`serve -music -tts`, three jobs submitted at once, then a
3-minute song with speech requests beside it):

| | result |
|---|---|
| folk, 30 s given, flac | estimated 17 s, ran 16.2 s |
| techno, no duration: the LM chose 233 s, 136 bpm, E♭ minor | estimated 43 s, then 76 s once planned; ran 78.7 s |
| piano, `thinking: false`, 20 s | estimated 1 s, ran 1.4 s |
| a speech request, idle / during LM planning / during DiT forwards (3-min song) | 32 ms / 32–71 ms / 0.14–0.31 s |

Each fetched file had the right container and length and peaked at −1 dB.
Drawn seeds are 32-bit, as upstream draws them. The first run drew int63
seeds, which a JavaScript client reads wrong past 2^53.

`go test ./api -run Music` covers:

- the lifecycle, with the plan arriving mid-run;
- upstream's fields (flat, nested, form);
- the refusals, and the upstream defaults they accept;
- cancel, and a 409 on content before completion;
- the 501 without `-music`.

### A12 — sample mode (2026-09-27)

**What upstream does.** A `release_task` with `sample_query` (or
`sample_mode: true` and no query, which is `NO USER INPUT`) goes through
`llm_generation_inputs.py`:

1. `parse_description_hints(query)`: the first language named in the query
   (a fixed word table, in dict order; names of ≤ 2 letters must sit between
   whitespace or `.,;:!?`, longer ones between `\b`s), and "instrumental",
   "pure music", "pure instrument" or a trailing "solo" for an instrumental.
   The request's `vocal_language` overrides the hint unless it is `en`,
   `unknown` or empty.
2. `create_sample` → `create_sample_from_query`: one row, no CFG, the prompt
   `# Instruction\nExpand the user's input into a more detailed and specific
   musical description:` over `{query}\n\ninstrumental: {true|false}`, and
   the FSM in its **understand** phase:
   - genres are **not** skipped, so every run writes `genres:`, held to
     `acestep/genres_vocab.txt` (178,571 lines, 4.8 MB) by a character trie
     over the lower-cased accumulated value. The trie never allows a leading
     space, so the model writes `genres:synthwave`;
   - the language, when there is one, is injected as a user meta;
   - no stop at reasoning: it writes `</think>` and goes free, with only the
     64,000 audio codes masked, until EOS (3,500 tokens at most);
   - the lyrics are what follows `</think>`, stripped, less a leading
     `^#\s*Lyri[c|cs]?\s*\n` (so `# Lyric` goes and `# Lyrics` stays) and a
     trailing `<|im_end|>`; `[Instrumental]` when empty and instrumental.
3. The song's caption, lyrics, bpm, key, time signature and duration
   **replace the request's**, and `generate_music` runs as ever. The song
   nearly always has all four metas, so phase 1 is skipped (`has_all_metas`)
   and the codes phase reads the song's caption and lyrics directly.
   `sample_mode: true` also turns `use_cot_metas` off, so phase 1 is skipped
   even for a song that lacks a meta; a bare `sample_query` leaves it on.

**What we do.** The same, with three departures, each where upstream's API
would silently drop what a client sent:

- a request's own `audio_duration`, `bpm`, `key_scale` and
  `time_signature` are injected into the sample pass as the language is
  (upstream overwrites them with the song's);
- an explicit `vocal_language: "en"` holds the lyrics to English (upstream
  skips `en` because it is its API's default; ours is empty);
- the DiT's vocal language is the song's when the request gave none (the
  Gradio app's choice; upstream's API keeps its default `en`).

A caption or lyrics sent with a query is a 400: upstream would discard them.
`use_format` (format mode) is still refused.

**Code:**

- `ace/lm/genres.go`: the vocabulary as one sorted list (a trie node is a
  binary search, its children the distinct runes after the prefix), and
  `_get_allowed_genres_tokens` over the precomputed first-character token
  map. It loads in ~0.4 s from `models/ACE-Step-1.5-src/` (the upstream clone
  every deployment has), and the thinking path now uses it too, which closes
  A7's one recorded FSM divergence.
- `ace/lm/fsm.go`: `newFSM(…, sample)`, the understand phase.
- `ace/lm/inspire.go`: `DescriptionHints`, `SamplePrompt`, `ExtractLyrics`,
  `SongOf` (create_sample's conversion), `Planner.Sample`, and `decode`, the
  one-row loop that `Think` now shares.
- `ace/pipeline`: `Options.Sample` (`Query`, `SkipCoT`), `Options.Sampled`
  (the song, as soon as it is written), `Result.Song`/`SampleText`.
- `api/music.go`: `sample_query`/`description`/`desc` and `sample_mode`; the
  job reports them, `metadata` gains `lyrics` and `genres`, and `stage` gains
  `sample`. `backend/music.go`: the sample pass in the estimate (~450 steps)
  and progress.
- `cmd/ace -sample "…"` (and `-sample-mode`).

**Oracles:**

- `reference/dump_ace_sample.py` (`reference/out/acesample`, seconds): the
  hints over 358 queries (58 written, 300 fuzzed), 26 prompts, 20 lyric
  extractions and 16 conversions.
- `reference/dump_ace_fsm.py` now also writes `fsm_sample.json`: five
  scripted understand-phase runs (a clean genre, a multi-word one, one that
  drifts off the vocabulary, the language injected, a duration injected with
  genres reached through the caption, an audio code in the lyrics). Its
  `fsm.json` is byte-identical to before.
- `reference/dump_ace_lm.py sample sample_ja` (~8 min each on the CPU)
  records upstream's real sample pass and stops at the codes phase's door.
  It now merges into the existing manifest instead of rewriting it.

**Gates:**

| gate | result |
|---|---|
| hints, prompts (text and ids), lyric cuts, conversions (`TestSampleHost`) | 358 + 24 + 20 + 16, all exact |
| the FSM on the scripts (`TestFSMSample`) | 2,001 steps exact (allowed sets of 1 to 153,204 tokens; genres 1–2,386) |
| the FSM on upstream's real passes (`TestFSMSampleTrace`) | `sample` 750 and `sample_ja` (language injected) 458 steps, exact |
| upstream's songs, parsed from its own text (`TestSamplePrompts`) | `sample` (97 bpm, 209 s, D minor, Czech) and `sample_ja` (120 bpm, 163 s, C minor, Japanese): field for field |
| the codes phase's prompts rebuilt from those songs | 755 + 33 and 460 + 33 ids exact |
| teacher-forced logits over the sample pass (`TestLogits`) | rel ≤ 2.5e-3, KL ≤ 5.7e-5, every argmax |
| live, on the device (`TestSamplePlanner`) | a Japanese city-pop song (language injected, genres "J-pop ballad") and an ambient-piano instrumental, every field valid |

`TestLogits`' relative bound went from 2e-3 to 3e-3. A7 set it at its own
worst, given_duration's phase-1 step 3 (the bpm value, at exactly 2.00e-3).
The sample pass's step 3 is the same position, at 2.5e-3 with KL 5.7e-5.
Nothing on the device changed; KL is the bound that matters.

**Measured:**

| run | result |
|---|---|
| `cmd/ace -sample "a melancholy synthwave song about driving through a neon city at night"` | the LM chose an instrumental, 158 bpm, D minor, 187 s: 183 sample steps (10.4 s), 935 codes (48.7 s); 187 s of song in 68 s |
| `-sample "an upbeat pop-punk song about the last day of school, female vocals" -duration 60` | duration held at 60; no language named, so the LM chose Chinese; 60 s in 39 s |
| served: `{"sample_query": "a cheerful acoustic folk song about a road trip with friends, in english", "audio_duration": 45}` | English lyrics, "Acoustic Folk", 167 bpm; estimated 36 s, ran 31 s; the lyrics were in `metadata` after 15 s |

A sample step is ~55 ms, like any other one-row LM step. The pass is a
caption, the metas and a song's lyrics: 100 to 750 tokens, so 5 to 40 s.

**What to know:**

- A description that names no language gets whatever the LM samples
  (Czech, Chinese…). Name one ("… in english") or send `vocal_language`.
- The lyrics are not known at submit time, so the codes phase's cache fit
  is checked when it starts, not at submit. The worst case (3,500 tokens of
  lyrics and a 10-minute song) would not fit 6,144 positions and fails the
  job with the LM's own message.

## Open questions

- **A-o1: does fp16 hold in the DiT's activations?** Answered above: fp32
  residual (5.7e5), fp16 operands (≤ 2.45e3).
- **A-o2: the LM's decode path.** Answered (A7): both options at once. The
  weight is staged once in the fragment tiling that the DiT GEMM rungs
  (prefill) and `llm_gemv.comp` (1–3 rows) both read. Attention is a new
  split-key kernel over a KV cache. 45 ms a step, bandwidth-bound; int8
  (A10) is the lever.
- **A-o3: the served door.** Answered by the user (A9): submit-then-poll,
  as `/v1/music` in `/v1/videos`' shape, reading upstream's request fields.
  OpenAI has no music endpoint. Upstream ships
  two:
  - its own async `/release_task` + `/query_result`;
  - an OpenAI-chat-compatible server (`openrouter/`): `/v1/chat/completions`
    whose reply carries the audio, plus `audio_config`, `lyrics` and
    `thinking` fields.

  Our `/v1/chat/completions` belongs to the LLM. To decide at A9 against
  OpenAI's reference ([[api-follows-openai-standard]]); ask before
  inventing.
- **A-o4: resident or staged per request?** Resident (A9): ~21 GB with the
  LM's cache, against ~15 s of staging for a request of 16–80 s. It sits
  beside image (~32 GB) and video (~50 GB at a request's peak) on the
  non-LLM machine.
- **A-o5: how much of the constrained-decoding FSM is live** in the default
  thinking path, and what it masks. Answered (§ A7 — the oracle): a
  forced-token YAML skeleton, digit/range and key/time-signature prefix
  sets, captions free except audio codes, and codes-only with a forced EOS
  in phase 2.
- **A-o6: does untiled decode differ audibly from upstream's tiled one?**
  Answered (A5): no. Upstream's own tiling is within 8.3e-6 of untiled,
  and ours (512/32) is within the fp16 path's 60+ dB.
- **A-o7: split-precision QK for DiT layer 0?** fp16 q/k there are the
  whole of the path's drift from fp32 (§ A2/A3 result), still 3.6–16×
  under upstream bf16's. It is worth doing if an end-to-end listen or a
  later gate wants the path closer to fp32.

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
  - `acestep-v15-xl-turbo/` (HF clone, `.git` since removed; includes `silence_latent.pt`
    and the HF `modeling_acestep_v15_xl_turbo.py`);
  - `acestep-5Hz-lm-4B/` (HF clone, `.git` since removed);
  - `Ace-Step1.5/` (`vae/`, the text encoder's configs);
  - `ACE-Step-1.5-src/` (upstream pipeline at `ca1e85f`; its
    `acestep/models/xl_turbo/` model file is the HF one plus DCW and
    flow-edit hooks).
- Oracle env:
  `uv venv --python 3.12 .venv-acestep`, then
  `uv pip install --python .venv-acestep/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.10.0`
  and `uv pip install --python .venv-acestep/bin/python transformers==4.57.1 diffusers==0.37.0 vector-quantize-pytorch einops loguru pyyaml numpy safetensors soundfile scipy pytorch-wavelets PyWavelets`.
  (`vector-quantize-pytorch` also went into `.venv`, harmlessly.)

**2026-09-27, session 2: A2–A6 done, A7's oracle written, A-o5 answered.** The DiT-only path
runs end to end: `go run ./cmd/ace -caption … -lyrics-file … -duration 90
-out song.mp3`.

- Code:
  - `ace/dit`: the three stacks, and the `WINDOW`/`CROSS` builds of
    `dit_attention_wmma.comp`;
  - `ace/vae`: Oobleck on kokoro's `A_CONV` GEMM, plus
    `shaders/ace_vae.comp`;
  - `ace/pipeline`;
  - `cmd/ace`.
- Oracles:
  - `dump_ace_dit.py --bf16` (upstream's CUDA dtype, from the fp32 run's
    noise, into `reference/out/acedit_bf16`);
  - `dump_ace_vae.py` (`reference/out/acevae`);
  - `dump_ace_detok.py` (A4, `reference/out/acedetok`);
  - `dump_ace_lm.py` (A7, `reference/out/acelm`; run it with
    `PYTHONDONTWRITEBYTECODE=1`, as the scripts import each other and
    would leave a `reference/__pycache__`).
- Tests: `go test ./ace/...`, ~1 min in all. Each package stages its own
  models and they don't share a device, so run them one package at a time
  if the machine is busy.
- The fp16 drift is all DiT layer 0's q/k (§ A2/A3 result, A-o7), priced
  against upstream's bf16.

**2026-09-27, session 3: A7 and A8 done.** The thinking path runs end to
end: `go run ./cmd/ace -caption … -lyrics-file … -duration 60 -out
song.mp3` (`-think=false` for the DiT-only path).

- Code: `ace/lm` (prompts + PyYAML emitter, the device graph, the FSM,
  sampling, the planner), `shaders/ace_lm_{prep,attn}.comp` (and
  `ace_lm_attn_combine.spv` from the second), `ace/pipeline` (`LoadLM`,
  `Options.Think`, `DiTRequest`, `Hints`), `cmd/ace` (`-think`,
  `-lm-seed`).
- Oracles: `reference/dump_ace_yaml.py` (`reference/out/aceyaml`, seconds)
  and `reference/dump_ace_fsm.py` (`reference/out/acefsm`, ~1 min), both
  under `.venv-acestep`, both byte-identical twice.
- Tests: `go test ./ace/lm` (~40 s, stages the LM at MaxLen 1024) and
  `ace/pipeline`'s `TestThinkingInputs`. `go test ./ace/...` is ~90 s, one
  package at a time.

**2026-09-27, session 4: listened (sounds great) and A9 done.** Songs are
served: `go run ./cmd/serve -music -token=`, then `POST /v1/music` (API.md
§ Music is a job).

- Code: `api/music.go` (+ `music_test.go`), `backend/music.go`,
  `cmd/serve` (`-music*` flags), and hooks in `ace/lm/planner.go` and
  `ace/pipeline` (`Between`, `Progress`, `Planned`, `Sampling`, `Shift`,
  `Timesteps`).
- Not deployed: `ai.service`'s unit line does not have `-music` yet. It
  belongs on the non-LLM machine (deployment is two machines), and adding it
  is the user's call.

**2026-09-27, session 5: A12 done.** Sample mode (upstream's "simple
mode"): `POST /v1/music {"sample_query": "…"}` or `cmd/ace -sample "…"`,
and the LM writes the caption, metas, genres and lyrics before the song
(§ A12). Needs `models/ACE-Step-1.5-src/acestep/genres_vocab.txt` beside
the LM (the upstream clone); without it sample mode is a 400 and the rest
serves as before. Songs to hear: `out/ace-sample-*.mp3`.

**Next, in order:**

1. **A10, the LM's decode**, which is 90% of a request (a 233 s song was 79
   s, ~60 s of it codes):
   - an int8 bank (`llm_gemv -DQ8B` reads the L8a bank at 1–3 rows; it
     halves the 8 GB a step);
   - host sampling (5 of 50 ms);
   - the phase-1 head over all 217k rows.

   Price int8 against bf16 logits (the M11a lesson).
2. A11 (optional): other tasks (cover, repaint, reference timbre). Each is a
   field `/v1/music` refuses today.
3. Close the vertical: freeze this file to `research/music-vertical.md` and
   leave `TODO.md` a one-line summary.
