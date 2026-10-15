# MUSIC — text (and lyrics) to a song (ACE-Step 1.5 XL turbo)

> **ARCHIVED 2026-09-27 — the vertical is closed, not unfinished.** This is
> the closing record of ACE-Step 1.5 (A0–A12, all done in one day). It was
> `MUSIC.md` at the repo root; what is open after closing is the
> last section, § "After closing" (moved from the retired root `TODO.md`).
>
> **Stage letters `A0…A12`, decisions `1–7` and open questions
> `A-o1…A-o7` resolve here.** They are cited from comments in `ace/`,
> `api/`, `backend/`, `cmd/ace`, `shaders/` and `reference/` that still
> say `MUSIC.md`, the same way live code still cites `LLM.md` and
> `IMAGE.md`. Nothing below was rewritten on archiving: the tenses are the
> working sessions'.
>
> **What was built**:
>
> - upstream's default thinking path (the 5 Hz LM's CoT and codes, the
>   DiT, the VAE), and sample mode;
> - the turbo model's audio-in tasks (cover, cover-nofsq, repaint) and a
>   reference's timbre;
> - all of it served as `/v1/music` jobs.
>
> Every gate is against upstream's own handler in fp32. Every drift is
> priced against upstream's bf16 (its CUDA dtype), and each lands under it,
> 2.8–39× closer to fp32. A 60 s song takes 14.4 s, and a 30 s cover or
> repaint ~2 s. The user listened after A8: "sounds great".
>
> **Where to pick it up**: the stage table, then "Open questions" and
> § Handoff. Three facts have each already caught a port, so read them
> before touching this code again:
>
> - the VAE encoder needs its audio and first block as fp16 pairs, and the
>   compiler folds the naive split to nothing (§ A11a);
> - a cover's fidelity is FSQ code flips (§ A11b–e);
> - the LM is priced against bf16 only over steps the FSM leaves open
>   (§ A10).
>
> **Left open when it closed**:
>
> - listening to the A10 (int8 LM) and A11 songs in `out/`;
> - the DiT's and the VAE's speed (≤20% of a request);
> - upstream's retake and flow-edit, which are refused;
> - deployment: `-music` is not in `ai.service`, which is the user's call.

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
| A10 | Performance: the LM's decode first (int8 bank?), then the DiT's small-M GEMMs and the VAE's convs | **done 2026-09-27 for the LM** (`ace/lm` `BankQ8`, the default): int8 layers priced against upstream bf16 over every teacher-forced step (text 0.88× its mean KL, codes 0.02×); a step 45 → 22 ms on the device, host sampling 10 → 1 ms; **a 60 s song 26.0 → 14.4 s**, 3 min served in 34.9 s. The DiT and VAE (≤20% of a request) are left |
| A11 | Other tasks: reference-audio timbre, cover, cover-nofsq, repaint, via the VAE encoder and the audio tokenizer (§ A11 — the plan) | **done 2026-09-27** (`ace/vae.Encoder`, `dit.GPU.Tokenize`, `ace/pipeline/tasks.go`, `/v1/music` multipart): encoder mean 56–63 dB from fp32 (bf16 16–19), with the audio and block 0 as fp16 pairs; tokenizer codes exact but 7 of 900 at a boundary; all six oracle cases under half of bf16's drift (2.8–39×); a 30 s cover or repaint in ~2 s, served |
| A12 | Sample mode: from a description alone the LM writes the caption, metas, genres and lyrics (upstream's `sample_query` / "simple mode"), then the thinking path runs on them | **done 2026-09-27** (`ace/lm` `Sample`, `pipeline.Options.Sample`, `sample_query` on `/v1/music`, `cmd/ace -sample`): host side exact on 358 hints, 24 prompts, 20 lyric cuts; the FSM exact over 2,001 scripted and 1,208 real steps; upstream's own song parsed field for field and its codes prompts rebuilt id for id; teacher-forced KL ≤ 5.7e-5, every argmax; served, 45 s of song in 31 s |
| A13 | Understand (captioning): audio → VAE encoder → audio tokenizer → 5 Hz codes → the LM writes caption, genres, metas and lyrics (upstream's `understand_music`, the API's `full_analysis_only`) (§ A13 — the plan) | **done 2026-10-12** (`ace/lm` `Understand`, `pipeline.Understand`, `cmd/ace -understand`, `full_analysis_only` on `/v1/music`): prompt ids, parse and FSM exact against upstream; teacher-forced KL ≤ 1.1e-5, every argmax; int8 under bf16's mean KL; codes within boundary flips; 30 s captioned in ~15 s, most of it the lyrics |

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

### A10 — the LM's decode (2026-09-27)

A request was 90% LM. At the end of A9 a step was 45 ms on the device and
50 ms wall: 7.3 GB of fp16 layers plus the head, read at ~187 GB/s, and 5 ms
of host sampling. A10 attacked both: fewer bytes, and less host work.

**The int8 bank.** `ace/lm` stages the layers in the LLM's L8a layout
(`BankQ8`): int8 in the §2.8 fragment tiling, one fp16 scale per 32 k of a
row. It is quantised from the bf16 checkpoint as Kev's is (K7.1), with q
rounded against the fp16 scale the kernel multiplies by. That is **3.9 GB
of layers against 7.3**. Decode reads it with `llm_gemv -DQ8B` and prefill
with the LLM's `llm_gemm -DQ8B` rungs (Kev's `q8m2`/`q8m4` schedule), both
unchanged. The tied head stays fp16 (below). `Options.Bank = BankFP16`,
`cmd/ace -lm-fp16` and `serve -music-lm-fp16` run the control.

**The price, against bf16 (the M11a lesson).**
`reference/dump_ace_lm_tf.py` (`.venv-acestep`, 19 s) teacher-forces
each of the A7/A12 oracle's sampled phases in one causal forward, in fp32
and again in bf16 (upstream's CUDA dtype). It writes fp32 logits at all
101 steps and bf16's KL(fp32 ‖ bf16) at T = 0.85 per step. Phase 1's KL is
over what the FSM allowed at that step, from the oracle's trace. **A step the
FSM forces is left out**: its logits choose nothing, and bf16's worst
whole-vocabulary steps are exactly those (steps 3–4 of a sample pass, where
the codes also hold up to 1% of the mass). `TestBankAgainstBF16` then runs
the bank over the same steps.

| KL from fp32, pooled | fp16 bank | int8 bank | upstream bf16 |
|---|---|---|---|
| phase 1, 273 unforced steps of 3 cases: mean / worst | 9.4e-7 / 3.2e-5 | **6.6e-4 / 4.6e-2** | 7.4e-4 / 3.4e-2 |
| phase 2, 2 × 101 steps (each row and after CFG): mean / worst | 3.1e-6 / 2.5e-5 | **6.6e-4 / 6.8e-3** | 3.4e-2 / 1.25e-1 |
| phase 1 argmax agreement | 273/273 | 271/273 | 271/273 |
| phase 2 CFG argmax agreement | 200/202 | 187/202 | 89/202 |

So int8 is **upstream's own bf16 in the CoT** and **50× under it in the
codes**. The gate pools each kind of phase: the bank's mean ≤ bf16's, and
its worst step ≤ 2× bf16's (int8: 0.88× and 1.36×; codes 0.02× and 0.05×).
One sensitive `sample_ja` step (4.6e-2) makes the phase-1 worst. Per case it
is noise: one case's mean is 1.4× bf16's, another's 0.44×.
`TestLogits`, the implementation gate against fp32, now always stages the
fp16 bank, so `go test ./ace/lm` runs both (~2 min, two stagings).

Tried and measured, and not kept:

- **Layers 6 and 16 in fp16**, Qwen3's massive-activation layers and Q13's
  exceptions. Phase 1 moved from 7.6e-4 to 5.9e-4 on one case and 3.6e-4
  to 3.5e-4 on another, and the worst step stayed. A per-layer sweep
  (`sample_ja`, one layer in int8 and the rest fp16, layers 0–26) shows why:
  layer 6 is 1.1e-4 of mean KL, layer 16 4.4e-5, and every other layer
  3e-6 to 4e-5. The error is spread thin, and no exception list buys much.
- **MSE-chosen scales** (the least-squares scale among amax/127 × 0.80–1.00
  per group): KL unchanged within noise, and staging took 19 s instead of 5.
- **An int8 head.** It moved phase 2's KL 25× (to 1.6e-2, half of bf16's)
  and its CFG argmax to 69/101, because the code logits are sensitive to
  the head and CFG doubles the difference. As a split head (int8 vocabulary
  for phase 1, an fp16 codes block for phase 2) it was clean, but it saved
  only 0.7 ms of a 24.8 ms phase-1 step: the int8 GEMV reads a 217k-row
  head at ~110 GB/s. Not worth two head banks.
- **Skipping the 64,000 code rows in phase 1's head** (328 MB a step). The
  FSM masks them in every constrained state, but upstream leaves the field
  name after a caption unmasked, and there the codes can hold real mass
  (1% on one of the oracle's steps). The skip would change upstream's
  behaviour, so the head reads the whole vocabulary.

**The split-K ladder** (`TestSlabLadder`, `ACE_LM_LADDER=1`) re-rungs the
int8 GEMVs, one projection at a time, whole steps at a fixed position:

| | fp16's rung → int8's | a step |
|---|---|---|
| (the fp16 table on int8) | | 27.1 ms |
| o (K 4096) | k16 → **k2** | 24.9 |
| gate, up | k4 → **k8** | 25.8 each |
| k, v | k20 → **k40** | 26.8 each |
| q, down | k8, k8 | unchanged |
| all of them | | **21.5 ms** (21.7 at two rows) |

21.7 ms for the 4.2 GB a codes step reads (3.9 layers + 0.33 head) is 194
GB/s, this bus. A decode step is also one command buffer now instead of
six: wall 22.7 → 22.3 ms against 21.7 on the device.

**Host sampling, 10 → 1 ms.** `topP` sorted every candidate within 30
nats of the max. That is ~64,000 on a flat codes step, 8 of the step's 10
ms (`BenchmarkCodeStep`). Now it buckets the candidates by distance below
the max (2,048 buckets over 30 nats). It keeps whole buckets while their
mass stays at most p, bins the rest once (a counting sort), and sorts only
the bucket that crosses. Equal scores share a bucket, so the kept set is
the sort's. `TestTopPBucketed` wants it identical on 343 rows (every
oracle codes step, text steps, 200 random rows full of ties) at p = 0.5,
0.9 and 0.99, and `TestTopP` still passes against upstream. Two traps:

- The first version walked a peaked phase-1 step through hundreds of empty
  buckets, rescanning 217k scores each time. It was 90 ms a step, and only
  the end-to-end run showed it. That's the reason for the counting sort and
  `BenchmarkTextStep`.
- `math.Max` in the max scans was half of what remained (NaN handling). A
  plain comparison replaced it.

A codes step's host work is 0.95 ms and a text step's 1.5 ms (fp16 was
10.1 and ~5).

**End to end** (`cmd/ace`, A8's request: synthwave caption, the oracle's
lyrics, `-duration 60`, seeds 42; staged models, the device to itself):

| | LM phase 1 | LM phase 2 | DiT | VAE | total |
|---|---|---|---|---|---|
| A8 (fp16) | 162 steps, 8.2 s (50 ms) | 300 codes, 15.1 s (50 ms) | 1.77 s | 0.62 s | 26.0 s |
| fp16 + A10's host | 162 steps, 7.6 s (47 ms) | 13.1 s (44 ms) | 1.79 s | 0.63 s | 23.5 s |
| **int8** | 163 steps, **4.5 s (27.7 ms)** | **7.2 s (24.0 ms)** | 1.79 s | 0.63 s | **14.4 s** |

Served (`serve -music`, one job at a time, device to itself): a 30 s folk
song with lyrics (flac) in **9.6 s** (A9: 16.2 s), and a 3-minute
instrumental techno track in **34.9 s** (A9's 233 s one was 78.7 s). The
job's `estimated_seconds` now uses the bank's rates (28 and 25 ms a step
int8, 47 and 44 fp16). Resident music is ~18 GB (was ~21). The songs are
`out/ace-a10-int8-synthwave-60s.mp3` and `out/ace-a10-int8-techno-180s.mp3`,
at −15.8 and −15.1 dB mean with peaks of −1.9 and −1.4 dB. Nobody has
listened to them yet.

**What is left of A10.** The LM is still ~80% of a request, and its step is
at the bus. Fewer bytes a step would mean a 4-bit bank: `llm_gemv -DQ4B`
reads one, but a K-quant quantiser from bf16 would be new, and phase 1 has
no headroom under bf16 left to spend. The DiT (1.8 s of a 60 s song, 6.9 s
of 4 minutes) and the VAE (0.6 s) are ≤ 20% of any request. They were the
plan's second and third items, and are left until something wants them.

### A11 — the plan: audio in (2026-09-27)

Every task that takes audio needs the one piece A0–A10 never built, the
**VAE encoder**, and then a different wiring of what is already on the
device. Upstream's turbo model serves four tasks (`TASK_TYPES_TURBO`):
text2music, cover, cover-nofsq and repaint. Lego, extract and complete are
base-model tasks, so they stay refused. A reference audio (timbre) can go
with any of them. What upstream does, read at `ca1e85f`:

- **Audio in** (`io_audio.py`): soundfile, or torchaudio for what it cannot
  read; mono is doubled, a third channel dropped; resampled to 48 kHz
  (torchaudio's `Resample`); clamped to ±1. All-silent (|x| < 1e-6
  everywhere) is refused.
- **The VAE encoder** (`OobleckEncoder`, diffusers): conv1 (k7, 2 → 128),
  five blocks (three residual units at dilation 1/3/9, Snake, then a
  strided conv of kernel 2s, stride s, padding s/2; s = 2, 4, 4, 6, 10;
  channels 128 → 128 → 256 → 512 → 1024 → 2048), Snake, conv2 (k3,
  2048 → 128). The 128 are a mean and a scale, and upstream **samples** the
  posterior, `mean + (softplus(scale) + 1e-4)·randn`, unseeded
  (`latent_dist.sample()`). It tiles at 30 s with 2 s overlap.
- **Reference audio** (`process_reference_audio`, `infer_refer_latent`):
  repeated to at least 30 s, then three 10 s segments at random offsets in
  its front, middle and back thirds, concatenated; encoded and sampled; the
  timbre encoder reads those 750 latents where it reads the silence today.
- **cover**: the source's latents (encoded, sampled) are the DiT's source,
  and `is_covers` sends them through the audio tokenizer (a pooler over
  5-latent groups plus a CLS row, FSQ) and back through the detokenizer:
  the A8 hints path, from audio instead of from the LM. **cover-nofsq** skips
  that bottleneck: the raw latents are the source. Both use the LM's
  instruction ("Generate audio semantic tokens…"), the length is the
  source's, and the LM does not run (`DIRECT_CONDITIONING_TASKS`).
  - `audio_cover_strength` < 1 switches to a text2music conditioning
    (instruction "Fill the audio semantic mask…", silence source) after
    `int(steps × strength)` steps.
  - `cover_noise_strength` > 0 starts from `t·noise + (1−t)·src` at the
    schedule entry nearest `1 − strength` and drops the entries before it.
- **repaint**: the source's latents with the span [start, end) replaced by
  silence are the source; `is_covers` is off. The chunk mask is the span as
  0/1 only with `chunk_mask_mode: "explicit"`; "auto" (the default) makes
  it all ones. Then:
  - after each of the first `round(ratio × steps)` steps, everything outside
    the span is reset to `t_next·noise + (1−t_next)·src`;
  - the result is blended with the clean source over `crossfade` latents at
    each edge;
  - after the decode, the source *waveform* is spliced back outside the span
    with a linear crossfade.
  `repaint_mode` and `repaint_strength` set the ratio and both crossfades
  (conservative 1.0/25/50 ms, aggressive 0/0/0, balanced (1−s) of each).
  A span past either end of the source pads the source with silence
  (outpainting).

The sub-stages, each gated against the fp32 oracle as before:

| # | what | gate |
|---|---|---|
| A11a | `ace/vae` encoder: posterior mean and scale | **done**: mean 55.9–62.9 dB from fp32, where upstream bf16 is 15.7–19.3 |
| A11b | reference timbre: segments, encode, sample, timbre encoder | **done**: clip exact, latents 4.1e-2 (bf16 0.40) |
| A11c | the audio tokenizer (pooler + FSQ) and cover / cover-nofsq, both strengths | **done**: 7 of 900 codes flip, all at a boundary; latents 3.5–39× under bf16 |
| A11d | repaint: padding, masks, injection, both blends, the splice | **done**: spans and masks exact; latents 2.8× / 13× under bf16 |
| A11e | serve: `/v1/music` takes `task_type`, `src_audio`/`reference_audio` uploads and the repaint and cover fields; `cmd/ace` flags | **done**: `TestMusicAudioTasks`; served repaint and cover in 2.0 and 2.7 s |

Our audio is read by ffmpeg (decision 7), resampled by its `swr`. That is
not torchaudio's sinc resampler, so the gates feed 48 kHz audio, and a
source at another rate is a documented departure. Noise (the posterior's
and the reference segments' offsets) is ours and seeded; the gates feed the
oracle's.

### A11a — result (2026-09-27)

`reference/dump_ace_vae_enc.py` (~1 min) encodes A5's two upstream-written
songs, plus the 120 s one cut to 100 s and 1,234 frames (not whole
latents), with diffusers' `OobleckEncoder` in fp32 and again in bf16
(upstream's CUDA dtype, `_get_vae_dtype`). It dumps per stage on 4 s, and
the whole posterior (mean, scale) untiled and through upstream's own
tiling (30 s chunks, 2 s overlap, the mean standing in for its unseeded
sample).

`ace/vae.Encoder` (`encoder.go`, `go test ./ace/vae`) is the decoder's
graph mirrored, on the same kernels and the same glue passes:

- **The strided convolutions are the A_CONV GEMM over overlapping rows.**
  Output frame t reads 2s consecutive input rows, which is one contiguous
  run of 2s·C values. The row stride is s·C and the tap stride C, so there
  is no im2col and no new GEMM.
- **The audio and block 0's activations are carried as two fp16s**, hi and
  lo = v − hi, against weights duplicated across the pair. The GEMM then
  sums W·v to ~22 bits of v (below).
  - The audio's pair rides in conv1's padding channels (its 2 channels are
    padded to 16 and its 7 taps to 8), so it is free.
  - Block 0's convolutions read a 2C-wide operand that a new `SPLIT` build
    of `shaders/ace_vae.comp` writes.
- **It borrows the decoder's activation arenas.** The two never run at
  once. Block 0's split operand is twice the decoder's widest fp16 stage,
  so the encoder's window is 255 latents, derived from the arena. It adds
  only its weights: 0.16 GB.
- **The margin is 16 latents.** The receptive field is ~±7: block 4's
  residual units reach ±3.9 latents, and its strided convolution and
  conv2 one each. A window holds the frames past the last whole latent
  too, which the unpadded convolutions read.

**Why the split: the encoder is precision-hungry at 48 kHz, not the
port.** The first fp16 build reached only 37.7 dB on the mean. That
matched, within 0.1 dB at every stage, torch's fp32 encoder with every
convolution's input and weights rounded to fp16. Rounding one site at a
time locates the loss:

| site rounded to fp16 (torch, 4 s) | posterior mean |
|---|---|
| the audio (conv1's input) | 45.5 dB |
| block 0's inputs | 51.1 dB |
| blocks 1–4, conv2 | 72–87 dB each |
| every weight | 67.4 dB |

Carrying the audio and block 0 exactly predicts 64 dB, and the port lands
on the prediction stage by stage. One trap: the obvious split,
`float16_t(v)` and then `v − float(hi)`, is folded by the compiler to
lo = 0. The shader instead cuts v to fp16's mantissa with a bit mask. A
teacher-forced block-0 convolution showed the fold: it matched the
activations-rounded model at 79 dB, not the exact one. After the fix it
matches the fp16-weight model at 105 dB.

| gate | ours (fp16, split) | upstream bf16 |
|---|---|---|
| conv1, blocks 0–4 (4 s excerpt) | 82.8, 67.6, 59.2, 54.7, 67.1, 65.9 dB | |
| posterior (4 s) | 79.9 dB | |
| mean: 30 s / 120 s tiled / 100 s + 1,234 frames | **59.5 / 55.9 / 62.9 dB** | 15.7 / 17.7 / 19.3 dB |
| scale: same | 80.9 / 77.1 / 80.9 dB | 32.4 / 39.8 / 46.5 dB |
| upstream's tiling vs untiled (mean) | | 107–130 dB: tiling changes nothing |

The stages are gated at 50 dB, and the whole encode at bf16's SNR or
50 dB, whichever is higher. The posterior's std is ~3e-4 against a mean
rms of ~1, so upstream's sampling barely moves a latent. `vae.Sample`
reproduces it, softplus threshold included. Device time: 30 s in 343 ms,
120 s in 1.39 s (0.28 s and 1.17 s before the split).

### A11b–e — the tasks, gated and served (2026-09-27)

**The oracle** is `reference/dump_ace_tasks.py` (~4 min; `--bf16` for
upstream's CUDA dtype, from the fp32 run's DiT noise). It runs upstream's
own `generate_music` on six requests, with the instruction upstream's API
resolves for each task. The source is the 30 s synthwave song and the
reference the 120 s lofi one, both from A5's oracle and written as 48 kHz
wavs. Hooks record:

- the processed audio and the reference's three `random.randint` draws;
- every VAE encode's mean and the sample upstream went on with;
- what `generate_audio` was handed;
- every `prepare_condition`'s packed sequence and context;
- the tokenizer's pooled rows and codes;
- the noise, the final latents and the audio.

| case | what it is |
|---|---|
| `ref_t2m` | text2music, 30 s, the lofi song as the reference |
| `cover` | cover of the synthwave song as a folk ballad |
| `cover_mix` | the same, `audio_cover_strength` 0.5, `cover_noise_strength` 0.3 |
| `cover_nofsq` | cover-nofsq, with the reference |
| `repaint` | repaint 10–20 s, balanced 0.5 (auto mask) |
| `repaint_out` | repaint 24–34 s of the 30 s song (outpainting 4 s), conservative, explicit mask |

**The tokenizer** (`dit.GPU.Tokenize`, A11c) is the detokenizer's mirror:

- `audio_acoustic_proj` and the pooler's `embed_tokens` have nothing
  between them, so they are folded into one [2048, 64] projection;
- a CLS row goes before every 5 latents;
- the two layers run over 6-row groups, on the GROUP attention at span 6
  with a row-mod-6 rope table;
- the final norm runs on the CLS rows;
- the FSQ runs on the host (`plan.FSQIndex`): project_in, ResidualFSQ's
  soft clamp tanh(z/c)·c with c = 1 + 1/(L−1), the hard clamp, and
  floor((L−1)(z+1)/2 + 0.5).

`TestGPUTokenize` feeds it A4's latents (the fp32 DiT's, whose codes A4's
detokenizer gate reads) and the cover's source latents. The pooled rows
are 5.8e-4 rms from fp32. 7 of 900 codes differ, each within 0.0008 of a
floor boundary.

**The wiring** (`ace/pipeline/tasks.go`, `Options.Audio`):

- **Reference** (`ReferenceClip`, exact against the oracle's draws): the
  30 s clip is encoded and sampled, and the timbre encoder reads it where
  it read the silence.
- **cover**: the source's sampled latents are padded to whole 5-latent
  groups with the silence's *first* rows, as upstream pads. They go through
  `Tokenize` and `Detokenize`, and the hints are cut back to T.
- **cover-nofsq**: the latents as they are.
- **audio_cover_strength**: after `int(steps × s)` steps the DiT is
  re-conditioned (`Begin`) on the same caption with text2music's
  instruction, over silence.
- **cover_noise_strength**: the start is the entry nearest 1 − s, from
  the renoised source (before the hints).
- **repaint**:
  - padding, and the span in latents with upstream's float floor
    division;
  - the context with the span silenced; the explicit or all-ones chunk
    mask;
  - the injection after the first `round(ratio·steps)` steps (half-even,
    as Python rounds; after DCW, never on the last step);
  - the latent crossfade (torch's fp32 `linspace`);
  - after the decode: the peak clamp, the waveform splice with its
    crossfade, then −1 dBFS. `vae.ClampPeak` splits Normalize for this.
- **Length and prompt**: T is the source's whole latents, padded to 128
  with the silence's first rows. The prompt's duration is the *unpadded*
  source's.

| gate | result |
|---|---|
| reference clip from the oracle's draws | exact |
| caption prompt through the text encoder (all six; and the non-cover one) | 0.9–1.0e-3 rms, A6's |
| chunk masks, repaint spans, context rows (non-cover) | exact |
| cover hints: outside the 1 flipped code of 150 | 1.3e-4 rms |

End to end (`TestTasks`), from the oracle's sampled latents and noise.
Every case is held under half of bf16's drift, as A6's gate is:

| case | latents rms from fp32 | upstream bf16 | audio SNR | device |
|---|---|---|---|---|
| ref_t2m | 4.1e-2 | 0.40 (9.7×) | 20.7 dB | 1.43 s / 30 s |
| cover | 1.91e-1 | 0.66 (3.5×) | 11.3 dB | 1.44 s |
| cover_mix | 6.1e-3 | 0.24 (39×) | 41.4 dB | 0.96 s (4 steps) |
| cover_nofsq | 2.1e-2 | 0.23 (11×) | 32.0 dB | 1.43 s |
| repaint | 9.8e-2 | 0.27 (2.8×) | 16.3 dB | 1.43 s |
| repaint_out | 1.2e-2 | 0.15 (13×) | 32.8 dB | 1.74 s / 34 s |

The full path, with our clip, encoder and posterior sampling and no oracle
latents (`TestTasksOwnEncode`), gives ref_t2m 4.1e-2, cover 2.37e-1 and
repaint 8.7e-2. All are under half of bf16's. bf16's own run encodes for
itself too.

**A cover is the fragile case, and the reason is the codes.** A code at a
floor boundary flips, and a flip replaces 5 rows of hints outright:

- Even from the oracle's own hints the cover drifts 0.13, against
  text2music's 0.085 (A6).
- Upstream's own unseeded posterior noise, at std 3e-4, flips 2 codes and
  moves the song 0.155. We measured that as two of our runs that differ
  only in the posterior seed.
- Before the encoder's split, our encode flipped 13 codes (0.465). bf16
  flips 73 of 150 (0.66).

So the encoder's precision is what a cover's fidelity rides on.

**Served** (`/v1/music`, A11e): `task_type`, multipart `src_audio` /
`ctx_audio` and `reference_audio` / `ref_audio`, and the cover and repaint
fields (API.md § Music). A source task skips the LM as upstream's does,
and runs as long as its source. The backend decodes the uploads at submit
(`pipeline.ReadAudioBytes`, ffmpeg pinned to local files) to check and
measure them, and again when the job runs. Refused where upstream would
silently drop a field:

- `*_path` fields (this server reads no files of its own);
- the base model's tasks;
- `audio_duration` on a source task;
- a task's fields on another task, a custom `instruction`, and the
  crossfade fields `repaint_mode` overrides.

Through a side server:

- **A 30 s repaint (mp3 in, flac out): 2.0 s.** Outside the span the
  result is the source at a constant gain (the −1 dBFS normalisation)
  within 27–33 dB, which is the mp3 upload's own error. Inside the span it
  is new audio.
- **A 30 s cover with a reference: 2.7 s**, against a 2 s estimate.

Staging still makes zero queue submits (`TestStagingSkipsTheQueue`, music
16 s).

Songs to hear, from `cmd/ace` (the synthwave song as the source):

- `out/ace-a11-cover-folk.mp3`
- `out/ace-a11-repaint-sax.mp3` (10–20 s)
- `out/ace-a11-ref-lofi.mp3` (thinking, with the lofi song as the
  reference)
- the served `out/ace-a11-served-{repaint.flac,cover-ref.mp3}`

Their lyrics lost the `[Verse 1]` tag to a test script's extraction; the
rest is A1's `LYRICS`.

### A13 — the plan: understand, i.e. captioning a song (2026-10-12)

**What upstream does** (read at `ca1e85f`). Three entry points share one
path: the Gradio "analyse source audio" / "transcribe codes" buttons
(`llm_analysis_actions.py`), the API's `full_analysis_only` job
(`api/job_analysis_runtime.py`), and the dataset labeller
(`training/dataset_builder_modules/label_single.py`). The path is:

1. `convert_src_audio_to_codes` (`core/generation/handler/audio_codes.py`):
   `process_src_audio` (stereo 48 kHz), a silence check, the VAE encoder,
   then `model.tokenize` with the silence latent's first rows padding to a
   whole 5-latent group. This is A11's `coverHints` up to `Tokenize`, so
   the codes already exist on our side (A11: exact but 7 of 900 at a
   boundary).
2. `understand_audio_from_codes` (`llm_inference.py:1850`): the chat
   template with the system turn `# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n`
   (`"Understand the given musical conditions and describe the audio
   semantics accordingly:"`), and the user turn **the codes string alone**
   (`<|audio_code_N|>` × T/5, nothing else). No CFG, one row.
3. `generate_from_formatted_prompt` with `generation_phase: "understand"`,
   no user metadata, caption, language and genres all generated,
   `stop_at_reasoning=False`, `target_duration=None`. This is A12's FSM
   in exactly the state `Sample` drives it with an empty `Meta`.
4. `parse_lm_output` for the metas and `_extract_lyrics_from_output`
   (A12's `ExtractLyrics`, the same regexes). `understand_music` then turns
   bpm into an int and duration into a float, and blanks `N/A` values.

The sampling settings differ between entry points. The API job uses
temperature **0.3**. Gradio and `understand_music` use **0.85**. Neither
sets top-k or top-p, and repetition penalty is 1.0.

**What we would build.**

- `ace/lm`: `UnderstandPrompt(codes []int) []int32`. The template's text
  parts are tokenised, and the codes are spliced in as `CodeBase+c` ids
  rather than encoded as a 5,000-token string. The gate is id-for-id
  against upstream's tokenizer on the oracle's prompt. Then
  `Planner.Understand(codes, temperature, rng)` is `Sample` without the
  query: `decode(ids, newFSM(v, Meta{}, true), …)`, then `ParseCoT` and
  `ExtractLyrics`, with no `[Instrumental]` fill (upstream adds none
  here).
- `ace/pipeline`: `Understand(audio)` runs the A11 decode, encoder and
  `Tokenize`, then the LM. This needs the VAE encoder, the DiT's tokenizer
  weights and the LM, but not the DiT forward or the VAE decoder.
- `cmd/ace -understand song.mp3` prints the caption, metas and lyrics as
  JSON.
- Serving is an open decision (A-o13 below).

**Gates** (the A7/A12 kind; sampling is not compared output for output):

- the oracle `reference/dump_ace_understand.py`: two or three real songs
  (one with vocals, one instrumental, one non-English) through upstream's
  `convert_src_audio_to_codes` and `understand_audio_from_codes` at
  temperature 0.3, saving the codes, the prompt ids, the raw text and the
  parsed result;
- codes exact against the oracle's, A11's boundary flips aside;
- prompt ids exact;
- teacher-forced on the oracle's text: KL and every argmax over each step,
  under the int8 bank as A10 priced it. The understand prompt is mostly
  audio codes, which A10 priced only as *outputs*. The bank needs pricing
  against bf16 here too, in case codes as *inputs* behave differently;
- the FSM exact over the oracle's steps (A12's harness);
- parsing field for field.

**Costs to expect.** The prompt is T/5 codes: 300 tokens for a minute,
1,200 for 4 min, 3,000 for 10 min. Prefill is cheap. The output is about
100–750 tokens, which is 2–17 s at 22 ms a step. `MaxLen` is 6,144, so a
10-minute song (3,000 codes plus up to 3,500 generated) can outgrow the
cache. Either cap understand at about 8 minutes or raise `MaxLen` for this
call.

**A-o13: where it is served.** The options are a field on `/v1/music`
(upstream's shape: `full_analysis_only` with `src_audio`, returning a job),
a synchronous endpoint, or `/v1/audio/transcriptions`-style multipart.
OpenAI has no music-analysis endpoint ([[api-follows-openai-standard]]),
so this is the user's call.

**Limits to state up front.** The LM writes lyrics from 5 Hz semantic
codes, so lyrics are a guess, not ASR. For words, Parakeet (the speech
vertical) on separated vocals would do better. Duration is the LM's own
estimate. We could overwrite it with the measured length, but upstream
does not.

### A13 — result (2026-10-12)

**Done, as planned, with two changes.** The oracle is two more cases of
`reference/dump_ace_lm.py` (`understand`, `understand_lofi`) rather than a
script of its own: the LM run, its hooks and its manifest are the ones
TestLogits, TestFSMSampleTrace and `dump_ace_lm_tf.py` already read. Each
case seeds torch, runs `convert_src_audio_to_codes` on A11's sources (the
30 s synthwave `src.wav` and the 120 s lofi `ref.wav`), dumps the posterior
sample it encoded (`{label}_latents`) and the codes, then runs
`understand_audio_from_codes` at 0.3 (163 s and 128 s on the CPU). And the
A-o13 serving decision went the user's way: `/v1/music` with
`full_analysis_only`, as upstream.

**Code:**

- `ace/lm/understand.go`: `UnderstandInstruction`, `UnderstandSampling`
  (0.3, top-p off), `Planner.UnderstandPrompt` (the template's text
  tokenised, the codes spliced in as `CodeBase+c`), `Planner.Understand`
  (A12's `decode` under `newFSM(v, Meta{}, true)`, step hook phase 3).
  Encoding the codes as text gives the same ids, but our tokenizer is
  slow over thousands of added tokens (TestUnderstandPrompt, which encodes
  both prompts that way, takes 70 s), so they are spliced.
- `ace/pipeline/understand.go`: `AudioCodes(latents)` (A11's padding and
  `Tokenize`, now shared with `coverHints` as `tokenize`) and
  `Understand(wav, Options)`: bounds (a latent to 600 s, `Silent` refused),
  the encoder with `Options.Seed`'s posterior noise, the tokenizer, the LM
  with `Options.LMSeed`; `Between` and `Progress` ("understand") at every
  LM step.
- `cmd/ace -understand song.mp3 [-lm-temp t]`: the JSON on stdout.
- `/v1/music` (`api/music.go`, `backend/music.go`): `full_analysis_only`
  (or `fullAnalysisOnly`) with `src_audio` uploaded is a job that writes no
  audio. Its `metadata` is the result (caption, genres, bpm, keyscale,
  timesignature, duration, vocal_language, lyrics), `seconds` the source's
  length, `full_analysis_only: true` on the job, and `/content` is a 409.
  It reads `lm_temperature` (default 0.3, upstream's constant), `lm_top_p`,
  `lm_seed`; `thinking` and upstream's defaults of the generation fields
  are tolerated, and anything that would shape a song (a caption, metas, a
  task, a reference, sample mode) is a 400. `extract_codes_only` and
  `analysis_only` were silently ignored before and are now 400s.
  `GET /v1/models` reports `full_analysis_only` when the LM and the genres
  vocabulary are loaded.

**Gates:**

| Gate | Result |
|---|---|
| prompt ids (`TestUnderstandPrompt`), spliced and text-encoded | exact: 179 and 629 ids |
| the song parsed from upstream's text vs `understand_audio_from_codes`' dict | field for field, both cases |
| the FSM replayed over upstream's steps (`TestFSMSampleTrace`) | exact: 252 and 167 steps |
| teacher-forced logits vs fp32 (`TestLogits`, fp16 bank) | KL ≤ 1.1e-5, every argmax; rel ≤ 3.3e-3 (see below) |
| int8 bank vs bf16 (`TestBankAgainstBF16`) | KL mean 0.95x / 0.43x bf16's, every argmax (bf16: 90/92); worst step 2.1x bf16's worst; text pooled 0.80x mean, 1.36x max |
| codes from the oracle's latents (`TestUnderstandCodes`) | 0 of 150; 10 of 600 flipped, all within 0.0061 of a boundary |
| live, end to end (`TestUnderstand`, `TestUnderstandPlanner`) | every field the FSM holds, genres in the vocabulary |
| the API (`TestMusicAnalysis`) | job, metadata, 409 content, 8 refusals |

Three things the gates turned up:

- **The duration's bound is a GPU tier.** Upstream bounds `duration:` by
  `gpu_config.max_duration_with_lm`: 240 s on tier 1, which is what the
  CPU oracle runs, and 600 s on the top tier, which is ours. After the
  understand case's "30", the oracle allowed only the newline (300 > 240)
  and we allowed digits. A12's cases never showed it: their durations
  were three digits already. `fsmVocab.setDurationMax` is
  `set_max_duration`, and the trace test replays at 240.
- **TestLogits' rel bound is 4e-3 now**, from 3e-3: `understand_lofi`'s
  step 3 (the bpm value, the step that set the bound before) is 3.3e-3
  after 628 tokens of codes, at KL 6.3e-6 and the right argmax.
- **TestLogits and TestBankAgainstBF16 prefilled in one pass**, which the
  test LM's 256 rows cannot take for 600 codes; both now chunk as
  `Planner.prefill` does.

**What it hears.** On `src.wav` (full_metas: synthwave, female vocals,
120 bpm, C major, 4/4, English, A1's lyrics), upstream's oracle wrote
"Chiptune", A major, 120 bpm, 4, French, 30 s; ours (int8, seed 42) wrote
"Synth-pop J-pop", G major, 120 bpm, 4, English, 30 s, "a clear, powerful
female vocal". Tempo, metre, length and the vocal are right in both; the
key is wrong in both. The lyrics are a loop of one line ("I'm on the
road" — the real ones are "Neon rivers run beneath the city light"): as
planned, the codes do not carry the words. On the lofi reference, upstream
heard instrumental funk-rock at 143 bpm.

**Cost.** 30 s of audio: encode 0.4–0.6 s, tokenize 12 ms, a 179-token
prefill in 73–90 ms, then ~490 steps at 28–30 ms: ~15 s, nearly all of it
the lyrics.
Served (a `serve -music` instance beside `ai.service`, the slot already
resident): A12's 60 s pop-punk song, submitted multipart, completed in
15 s against a 14 s estimate. It heard "high-energy pop-punk with J-rock
influences… lead male vocal", 100 bpm, F# major, 2, 60 s, and wrote
Catalan-ish lyrics: again, the words are the weak part.

### A14 — the plan: a quality score on every song (2026-10-13)

**What the user asked for.** An automatic score on a finished song, plus a
flag a client can read when it polls the job, to judge the result before
listening. How it is computed is open. The user mentioned MusicCNN and CLAP
as examples, not as requirements.

**What "good" can mean here.** No single number covers it. There are three
separate questions, and published music-generation evaluations score each
one separately:

1. **Did it follow the request?** Does the song match the caption, the
   lyrics and the metas that were asked for? This is adherence.
2. **Does it sound well made?** Is it clean, with no artefacts, mush or
   clipping? This is production or signal quality.
3. **Is it a good song?** Is it coherent, with a clear structure, natural
   vocals and something memorable? This is musical or aesthetic quality.

ACE-Step 1.5's own report (arXiv 2602.00744) scores (2) and (3) with Meta's
**Audiobox Aesthetics** and **SongEval**. For (1) it uses its own
ACE-Caption-Reward and ACE-Lyric-Reward models, not CLAP or MuQ-MuLan. The
v1 report used CLAP and MuLan, and v1.5 dropped them. The report quotes
AudioBox CU 8.09, PQ 8.35 and SongEval Coherence 4.72 for this model.
That gives us a reference distribution for the same model on the same
metrics, so a port that scores our songs far from those numbers is
suspect.

**What upstream already ships (read at `ca1e85f`).** Gradio's
"Auto Score" checkbox (`auto_score`, `ui/gradio/events/results/scoring.py`)
runs two scorers. Its API (`release_task`, `openrouter/`) exposes neither,
so how we serve it is our own call (A-o14).

- **LM PMI** (`core/scoring/lm_score.py`,
  `calculate_pmi_score_per_condition`). The 5 Hz LM reads the song's
  codes in the A13 understand prompt and is teacher-forced on the
  conditions. It runs one forward over the whole sequence and takes the
  log-softmax at each target position, with no sampling.
  - The caption and the lyrics are scored by **PMI**: the mean log p of
    `<think>\ncaption: …\n</think>\n` (or `<think>\n</think>\n# Lyric\n…`)
    given the codes, minus the same given the codes string
    `"NO USER INPUT"`. The result is squashed by `sigmoid(PMI / scale)`.
  - bpm, duration, genres, keyscale, language and timesignature are
    scored by **top-10 recall**: the rank-weighted share of the field's
    YAML tokens that sit in the LM's top 10.
  - `calculate_reward_score` combines them 0.5 caption, 0.3 lyrics and
    0.2 mean metadata, and renormalises the weights when one is absent.
  - `scale` is 0.1 in the function and 0.5 in Gradio's saved preferences
    (`user_preferences.py`), so the 0–1 number depends on that knob.
  - This is question (1) only. It also scores the **codes the LM planned**,
    not the audio the DiT rendered, and Gradio skips it for cover and
    repaint, which have no LM codes.
- **DiT lyric alignment** (`core/generation/handler/lyric_score.py`,
  `core/scoring/dit_score.py`). This runs only when there are lyrics.
  - One decoder forward on two rows: pure noise at t = 1 (`lm_score`), and
    the final latent noised to t = 1/steps (`dit_score`). Both use the
    song's conditioning and seed 42, and the forward exits early after the
    last listed layer.
  - It reads the cross-attention probabilities of the heads in the
    checkpoint's `lyric_alignment_layers_config`. For XL turbo those are
    `{3: [18, 27], 4: [22], 5: [5, 6, 7], 6: [2, 12, 13], 7: [20, 21]}`,
    not the handler's default.
  - The heads are averaged over the lyric tokens (structure tags masked)
    and min-max normalised, and a DTW path is found on energy².
  - The score is **coverage² × monotonicity² × path confidence**, in 0–1.
    It asks whether every lyric line is sung once, in order. That is half
    of question (1), the half that PMI cannot hear, because the codes do
    not carry words (A13).

**The outside models, and why each is in or out:**

| model | answers | input | size | weights' licence | verdict |
|---|---|---|---|---|---|
| Audiobox Aesthetics (`facebook/audiobox-aesthetics`) | (2), partly (3): PQ production quality, PC complexity, CE enjoyment, CU usefulness, each 1–10 | 16 kHz mono, 10 s windows at a 10 s hop, averaged | WavLM-base: CNN front end + 12 × 768 transformer, plus four layer-weighted MLP heads, ~95 M | CC-BY 4.0 (code CC-BY 4.0 + MIT) | **in** (A14c) |
| SongEval (`ASLP-lab/SongEval`) | (3): coherence, musicality, memorability, structure clarity, vocal naturalness, each 1–5 | 24 kHz mono, the whole song in one pass | MuQ-large-msd-iter (~300 M; only its first 6 layers are read) + a small head (FFN 1024→4096→1024, 4 × 8-head self-attention, mean+max pool, `tanh·2+3`) | MuQ weights CC-BY-NC 4.0; SongEval's README says CC BY-NC-SA 4.0, its LICENSE file says Apache-2.0 | **in, if the licence is acceptable** (A14d, A-o15) |
| CLAP (LAION) / MuQ-MuLan | (1): text–audio cosine | 48 kHz / 24 kHz | ~200 M / ~700 M | CLAP permissive; MuLan CC-BY-NC | **out** for now: same question as the LM PMI, with a general-purpose text encoder; ACE-Step dropped them in v1.5 |
| ACE-Caption-Reward / ACE-Lyric-Reward | (1) | ? | ? | ? | **check** whether they are released (A14a); if so, they replace CLAP |
| MusicCNN | a tagger (genre, mood, instruments) | 16 kHz | small | ISC | **out**: it labels, it does not judge. Tags could feed an adherence check, but the PMI already does that with the model's own vocabulary |
| FAD / KAD | distribution distance | — | — | — | **out**: a property of a set of songs, not of one song |
| Parakeet ASR → WER against the lyrics | (1), intelligibility | 16 kHz | resident already | — | **parked**: words over a full mix need source separation first; DiT alignment covers "sung in order" without it |

**What we would build.** The stages run cheapest first. Each one can ship
on its own and adds fields to the same object.

- **A14a — survey close-out (no code).** Look for released ACE reward
  models (HF `ACE-Step/*`, the 1.5 report's links). Read SongEval's
  actual weights licence (the `ckpt/` folder, the HF dataset card) and
  MuQ's `LICENSE_weights`. Put both to the user as A-o15.
- **A14b — upstream's auto-score, ported.** No new weights; it is the LM
  and the DiT we already have resident.
  - `ace/lm`: `Planner.Score(codes, Meta, caption, lyrics)`. It prefills
    the understand prompt once and keeps the KV prefix (A13's
    `UnderstandPrompt`), then teacher-forces each target with
    full-vocabulary logits at every target row. TestLogits'
    harness already teacher-forces this way. The unconditional prompt is the same template over
    `"NO USER INPUT"`, prefilled once. That means about 10 short forwards
    after one prefill of T/5 + ~30 tokens. Expect well under a second for
    a 60 s song; A13 measured a 179-token prefill at 73–90 ms.
  - `ace/dit`: an attention-probability output for the listed heads of
    layers 3–7 from the `CROSS` attention build: the softmax over the
    encoder tokens, written out for those 11 heads only. Then an
    early-exit forward, layers 0–7 of 32 on two rows. Expect about a
    quarter of one denoising step.
  - The DTW, median filter, mask and metrics on the host in Go
    (`core/scoring/_dtw.py` is 110 lines).
  - **A departure to measure, not assume:** also run PMI on codes
    re-tokenized from the *rendered* latent (A13's `AudioCodes`). That
    scores what was made rather than what was planned, and it gives
    cover, repaint and DiT-only songs a PMI. Compare the two on A14e's
    set before choosing.
  - Oracle `reference/dump_ace_score.py` (`.venv-acestep`): upstream's
    `calculate_pmi_score_per_condition` and `get_lyric_score` on A7/A12's
    cases, dumping the per-target mean log-probs, the recalls, the
    attention energy matrices, the DTW path and the scores.
  - Gates: log-probs to the TestLogits bound under the fp16 bank, then
    priced against bf16 under int8 as in A10. Recall exact. The energy
    matrix to fp16 tolerance. The DTW path exact on the oracle's matrix.
    Scores to 1e-3.
- **A14c — Audiobox Aesthetics.** This is the cheapest outside model and
  the safest licence, and it is the metric the report quotes.
  - Port WavLM-base to the device in a new `aes/` package. The conv front
    end goes through kokoro's/ace's conv GEMM. The 12 transformer layers
    use the DiT GEMM rungs, plus WavLM's gated relative position bias,
    which is the only unusual piece. Then the per-axis layer weighting
    and the MLPs.
  - Input: resample 48 kHz stereo to 16 kHz mono (the reference's
    torchaudio resample, matched to the oracle), then 10 s windows.
    Batch the windows: a 4-minute song is 24 rows of ~500 frames.
  - Oracle `reference/dump_aes.py` in its own venv (`audiobox_aesthetics`
    from PyPI, torch CPU). Gates: per-layer hidden states, then the four
    axes to ~1e-3 on A10/A11's songs and a few non-generated clips.
  - Sanity check: our songs' mean PQ and CU should land near the report's
    8.35 and 8.09.
- **A14d — SongEval** (A-o15: yes, 2026-10-13). It is the only model on the list
  that judges whole-song qualities such as structure, memorability and
  vocal phrasing, which is closest to "is it good".
  - Port MuQ's first 6 layers (a mel front end with a Conformer stack at
    25 Hz; to be confirmed against `muq` at A14a). MuQ wants fp32
    ("NaN otherwise", its README), so price fp16 against fp32 the way A2
    did. Then the SongEval head.
  - The whole song goes in as one sequence: 4 min is about 6,000 frames.
    The attention is long but cheap at 6 layers, and the DiT attention
    kernels already handle that length.
  - Oracle `reference/dump_songeval.py` (`muq`, `hydra`, the repo's
    `ckpt/`), gated the same way as A14c, with the report's Coherence 4.72
    as the sanity check.
- **A14e — calibration: does the score track what a listener hears?**
  - Build a set:
    - ~40 normal requests (varied genres, with and without vocals,
      30 s–4 min, two seeds each);
    - known-bad controls: 1–2 DiT steps, a caption swapped with another
      song's, lyrics swapped, A12's degenerate loops, a cover at
      strength 0, white noise and silence.
  - Score all of it, and have the user rate ~30 songs blind on a 1–5
    scale.
  - Gates:
    - every control is flagged;
    - Spearman correlation of each component with the ratings, recorded;
    - choose the components and the weights for the headline number, the
      PMI `scale` (0.1 or 0.5), and the flag's thresholds from this
      set, not invented in advance.
- **A14f — serve it.**
  - A `quality` object on `MusicJob`, filled before the job turns
    `completed`, so a client resolving the job sees it alongside the
    audio. A shape to propose (A-o14):
    `{"flag": "good"|"fair"|"poor", "score": 0–1, "adherence":
    {"caption", "lyrics", "metadata", "lyrics_alignment"}, "production":
    {"pq", "pc", "ce", "cu"}, "song": {"coherence", "musicality",
    "memorability", "clarity", "naturalness"}}`. Components that did not
    run are omitted, not zero.
  - `cmd/ace -score` prints it, and `cmd/ace -score-only song.mp3` scores
    any file. The audio models need no request, so this also works on
    uploads.
  - The estimate in `backend/music.go` gains the scoring cost.
  - Memory: Audiobox is ~0.2 GB fp16 and MuQ's 6 layers ~0.3 GB, against
    the music pipeline's ~21 GB resident. Both stay resident with the music
    pipeline.

**Costs to expect.** For a 60 s song, A14b should take under ~0.5 s. A14c
is six 10 s windows through a 95 M model, well under a second. A14d is one
6-layer pass, about a second. In total that is ≤ ~2 s against a 14.4 s
request (15%), and the order to build them in is the cost order. If the
total comes in over that, the flag can be computed after `completed`, with
`quality` null until then (A-o14).

**Limits to state up front.**

- No automatic score is a listener. Audiobox and SongEval were trained on
  human ratings of human-made and other models' music. They can be fooled
  by artefacts that never appeared in their training data, which is what
  A14e's controls are for.
- PMI depends on the LM's own vocabulary and biases: a caption the LM
  writes badly will score badly whatever the audio.
- Lyric alignment says the words are *placed* in order, not that they
  are *intelligible*. That would take ASR on separated vocals (parked).

### A14a — result (2026-10-13)

**The survey is closed, and the user answered A-o14 and A-o15 the same
day: non-commercial weights are fine, so A14d is in, and the score is on
by default.**

- **ACE's reward models are not released.** HF's `ACE-Step` organisation
  has no ACE-Caption-Reward or ACE-Lyric-Reward; a search for both names
  finds nothing. The organisation does publish two Qwen2.5-Omni models,
  `acestep-captioner` and `acestep-transcriber`. The transcriber is a
  lyrics ASR, a candidate for intelligibility (lyric WER) if the
  alignment score proves too weak, and it would not need source
  separation.
- **Licences:**
  - MuQ's weights are CC-BY-NC 4.0 (`LICENSE_weights`, and the HF card
    for `OpenMuQ/MuQ-large-msd-iter`);
  - SongEval's head is a 100 MB `ckpt/model.safetensors` in the GitHub
    repo, whose README and HF dataset card both say CC BY-NC-SA 4.0 (the
    repo's LICENSE file is Apache-2.0);
  - Audiobox Aesthetics is CC-BY 4.0.
- **MuQ, for A14d** (`config.json`):
  - input: a 128-mel spectrogram at 24 kHz with hop 240 (100 Hz),
    normalised by the stats in the config;
  - a conv front end down to 25 Hz, then a 12-layer, 1024-wide
    wav2vec2-Conformer: rotary positions, a depthwise conv of kernel 31,
    16 heads, FFN 4096;
  - SongEval reads `hidden_states[6]`, the output of layer 6, so six
    layers are needed. Its head is FFN 1024→4096→1024, four 8-head
    self-attention layers, mean and max pooling, then a linear to five
    scores through `tanh·2+3`.
  - the whole song goes through at 24 kHz in one pass (`eval.py`).

### A14b — result (2026-10-13)

**Done.** Upstream's Gradio "Auto Score" is ported, both halves, gated
against upstream's own functions in fp32 and priced against its bf16. It
is served by default as the job's `quality` (research/api-jobs.md § Music
is a job).

**The oracle** is `reference/dump_ace_score.py` (`--bf16` for the twin),
which writes to `reference/out/acescore{,_bf16}`; each run takes ~3 min
on the CPU. It has five cases:

- `pmi_match`: A13's `understand` codes, which are A1's full_metas song,
  scored against the full_metas request.
- `pmi_mismatch`: the lofi reference's codes against the same request,
  as the control.
- `align_full_metas`, `align_zh_odd`, `align_instrumental_60`: A1's
  requests, generated with seed 42 and then scored. The script records
  the decoder's inputs, the noise, the attention probabilities of the
  11 heads for both rows, the energy, the DTW path and the metrics.

It spies on `_get_logits_and_target_for_scoring` and on the decoder's
forward, and otherwise calls upstream's code unchanged.

**Code:**

- **`ace/lm/score.go`.**
  - `Planner.Score(codes, Conditions, scale)` is
    `calculate_pmi_score_per_condition` and `calculate_reward_score`.
  - `ScoreTargets` gives the YAML of one field and the lyric block, in
    upstream's order.
  - `ScorePrompts` gives the understand prompt and the `NO USER INPUT`
    prompt.
  - `Teach` is one target's teacher-forced pass.
  - `Reward`.
- **`ace/lm/gpu.go`.**
  - `GPU.Teacher` runs a pass of up to 64 rows (`ScoreRows`) and returns
    each row's log-softmax and rank over the whole vocabulary. The head
    runs as the fp16 `reg32x128` GEMM rung rather than `llm_gemv`'s
    three rows.
  - The head bank is padded to 128 rows for it (`headRows`), and its
    logits take a 56 MB region of the host-cached arena.
  - `Pass` is now `run` plus a tail.
- **`ace/dit/align.go`.**
  - `GPU.CrossProbs` runs a forward up to the last listed layer and
    rebuilds the listed heads' cross-attention softmax on the host, from
    the packed cross q and k (fp16, qkpack's fragment tiling, base-2
    logits).
  - `Config.AlignHeads` reads `lyric_alignment_layers_config`.
  - `stepGraph`'s input stage is split out as `inputGraph`.
- **`ace/pipeline/score.go`.**
  - `LyricAlignment` covers the seed-42 noise, the two rows, `LyricSpan`,
    `TypeMask`, `HeadEnergy`, `Normalize`, `DTW` (fp32 with numba's
    tie-breaking) and `Metrics`.
  - `ScoreConditions` is Gradio's `_build_score_metadata`.
  - `Options.Score` and `Result.Quality`. The alignment runs at the end of
    `generate`, while the DiT still holds the song's condition, and the
    PMI runs in `Generate` after the codes path.
- **`api/music.go` and `backend/music.go`.**
  - `MusicQuality` on the job (`quality`).
  - `auto_score`/`autoScore` (Gradio's name) turns the score off; it is
    on by default.
  - A `score` stage and a `scoreCost` in the estimate.
- **`cmd/ace -score`**, on by default, logs the score.

**Gates:**

| gate | result |
|---|---|
| targets and ids, prompt + target (`TestScoreTargets`) | exact, 18 forwards, incl. `timesignature: '4'` |
| teacher-forced mean log p, fp16 bank (`TestScore`, `ACE_LM_BANK=fp16`) | ≤ 5.2e-3 a forward (bf16: 5e-3–2.5e-1); 1.9e-3 / 2.0e-3 pooled; rewards 0.6454 / 0.4454 vs fp32 0.6448 / 0.4454 |
| int8 bank (`TestScore`), pooled as A10 prices it | 3.76e-2 / 5.76e-2 a forward vs bf16 3.89e-2 / 8.96e-2; every condition and the reward within bf16's drift of fp32 (or 0.01) |
| our seed-42 noise | TorchRandn to an ulp (TestNoise's 1e-6) |
| lyric ids, span, type mask (`TestAlignHost`) | exact, 3 cases |
| head average and min-max energy from upstream's probabilities | ≤ 2.2e-8 |
| DTW path, over upstream's energy and over ours | exact, 6 paths (180–753 steps) |
| metrics from upstream's probabilities | coverage and monotonicity exact; confidence ≤ 1e-6; score to upstream's 4 places |
| attention probabilities from upstream's decoder inputs (`TestLyricAlignment`) | rms 6.7e-4–1.7e-2 of fp32's, against bf16's 4.0e-2–0.75 (19–60× closer) |
| alignment scores end to end | within 0.0012 of fp32 (bf16 is up to 0.041 off) |
| the API (`TestMusicQuality`) | on by default, both spellings of `auto_score: false`, JSON shape |
| served (`serve -music` beside ai.service) | 30 s song with lyrics in 11.1 s against a 10 s estimate, `quality` on the job |

**What the gates turned up:**

- **Upstream's "early exit" never exits.** `AceStepDiTModel.forward` works
  out the last layer it needs and then runs all 32. Layers 3–7's
  attention depends only on layers 0–7, so we stop after layer 7.
- **The alignment heads are the checkpoint's, not the handler's.** The XL
  turbo `config.json` names 11 heads in layers 3–7, and the handler's
  `_DEFAULT_LAYERS_CONFIG` (layers 2–6) only applies when a config has
  none.
- **The alignment score favours an instrumental.** `[Instrumental]`
  scores 0.33/0.34, because it has no sung rows: coverage and
  monotonicity are 1, and the score is the path's confidence alone. The
  30 s sung song scores 0.17/0.14, the Chinese one 0.09/0.0005, and the
  200 s song with 12 identical verses 0.014/0.018. As upstream defines it,
  the number is not comparable between a song and an instrumental. A14e
  has to decide how to read it.
- **The LM scores its own plan.** `adherence` reads the CoT's metas, so a
  meta the request left out is one the LM chose and then recalls: bpm
  0.91 on a song whose bpm the LM picked. That is upstream's behaviour,
  but it is "the codes agree with the plan", not "the song is good".
- **PMI separates matched from mismatched codes,** though not by much on
  everything. The reward is 0.645 on the matched song and 0.445 on the
  mismatched one, and the caption 0.36 against 0.015, but the lyrics only
  0.988 against 0.944. The caption's PMI is negative even when the song
  matches. The user's short caption is not how this LM writes captions;
  served with thinking, the LM's own rewritten caption scored 0.48.
- The bf16 run is a noisy bar on the alignment. bf16 alone moves
  `instrumental_60`'s `lm` score 0.327 → 0.287 and flips coverage counts.

**Departures:**

- **The scale is 0.5,** Gradio's shipped `score_scale`, not the
  function's default of 0.1.
- **A language of "unknown" is not scored.** Gradio would recall its own
  default `language: unknown`, and that measures nothing.
- **The two prompts are prefilled once** (slots 0 and 1), and each target
  is a teacher-forced pass from the prompt's last token. That gives the
  same logits as upstream's full forward per target, at a tenth of the
  cost.
- **The probabilities are rebuilt on the host** from fp16 q and k. Our
  attention is a flash kernel and never forms them.

**Cost** (TestScoreGenerate, both halves):

| song | LM part | alignment | total | share of the request |
|---|---|---|---|---|
| 30 s | 0.56 s: 211-token prefill 90 ms, 185 target tokens in 9 passes, 466 ms | ~0.11 s | 0.67 s | 5.7 s, 12% |
| 200 s, 12 verses | 2.65 s: 1,061-token prefill 0.53 s, 1,481 target tokens, 2.1 s | ~0.6 s | 3.3 s | 44.7 s, 7% |

A teacher-forced pass costs ~45 ms whatever its row count up to 64,
because the int8 layers are read once a pass. If this ever matters, the
nine short targets could share passes across more slots.

**Not done, by design:**

- the plan-versus-render question (A-o16) and a PMI for the DiT-only,
  cover and repaint paths, which upstream does not score either;
- the good/fair/poor flag, which waits for A14e;
- the deployment, which the user then asked for: deployed 2026-10-13. A
  20 s folk song with lyrics came back from `ai.service` in 10 s with its
  `quality`.

### A14c — result (2026-10-13)

**Done and deployed.** Meta's Audiobox Aesthetics runs on the device, and
every song's `quality` now carries its four axes as `aesthetics` (1–10).
Every score lands within 0.0015 of upstream's fp32 predictor.

**What upstream runs** (`audiobox_aesthetics` 0.0.4, `infer.AesPredictor`):

- The audio is resampled to 16 kHz with torchaudio's Hann sinc, and the
  channels are averaged.
- It is cut into 10 s windows at a 10 s hop. The last window is
  zero-padded, and its padding is masked: a frame of 320 samples is padding
  when all of them are.
- Each window goes through WavLM-base:
  - a 7-layer conv extractor, with GroupNorm(512, 512) and GELU on the
    first layer and GELU on the rest;
  - LayerNorm(512) and a 512→768 projection;
  - a positional conv (kernel 128, 16 groups, weight-normed) with GELU,
    added to the input, then LayerNorm;
  - 12 post-LN layers whose attention adds layer 0's bucketed
    relative-position bias, gated per layer from the query's input
    (`gru_rel_pos`).
- Each of the four axes takes a softmax-weighted sum of the 13 hidden
  states, a mean over the valid frames, L2 normalisation, a 5-layer MLP
  (Linear–LayerNorm–GELU ×4, then Linear) and the checkpoint's mean and
  std.
- The song's score is the windows' mean, weighted by their length.
- It is fp32 even though the predictor says "bf16": `enable_autocast`
  compares a torch dtype to strings and is never true.

**Code:**

- **`audiobox/` (new package).**
  - `model.go`: weights, weight norm folded, `bucket`.
  - `heads.go`: the four MLPs on the host.
  - `resample.go`: `Mono16k`.
  - `gpu.go`: arenas, staging and the graph.
  - `score.go`: `Load`, `Score`, `ScoreMono`, the windows.
- **Shaders**, `shaders/audiobox_*.comp`:
  - `conv0`: the first conv in fp32 (raw audio in fp16 is the A11a trap);
  - `colstats`: the column mean and M2 in 1,024-row chunks, combined by
    Chan's formula. It serves the GroupNorm over 31,999 frames and the
    heads' pooling.
  - `act`: an optional norm, then the exact erf GELU (the repo's other
    GELUs are the tanh approximation), out as fp16 or fp32;
  - `posconv`: the grouped conv, fp32 with LDS-staged input, fused with
    the GELU and the residual;
  - `relbias`: the gated bias plane, in log2 units.
- **Reused kernels:**
  - kokoro's A_CONV GEMM runs conv layers 1–6 as strided convolutions
    (A row stride = stride·512, tap stride one input row);
  - the LDS-staged GEMM runs every projection, with the bias as an extra
    K column (kokoro's PL-BERT trick);
  - parakeet's LayerNorm and residual, and kokoro's HD64 pack;
  - `dit_attention_wmma`'s REL_BIAS build, rebuilt at head width 64
    (`audiobox_attn_hd64_w32`). It takes each window's valid frames as
    its key count.
- **Wiring.**
  - `pipeline.LoadAesthetics`, `Quality.Aesthetics`, scored over the final
    -1 dBFS audio in `Options.Score`.
  - The backend loads it when `models/audiobox-aesthetics` exists
    (`-music-aes`, default on, ~0.6 GB).
  - `MusicAesthetics` in the API.
  - `cmd/ace -aesthetics file` scores any file and prints JSON.
- **Oracle:** `reference/dump_aes.py` in `.venv-aes` (torch 2.10 CPU,
  torchaudio, `audiobox_aesthetics==0.0.4`). The weights are loaded from
  `models/audiobox-aesthetics/model.safetensors` (HF), and the source is
  cloned at `models/audiobox-aesthetics-src` (`2618e9d`). The four cases:
  - A11's 30 s synthwave source and 120 s lofi reference;
  - the source cut to 23.7 s, which ends in a padded window;
  - 12.5 s of seeded white noise.

**Gates:**

| gate | result |
|---|---|
| relative-position buckets, every offset −498..498 (`TestBuckets`) | exact |
| 48 kHz stereo → 16 kHz mono (`TestResample`) | rms 1.2e-7, max 1.8e-7 |
| extractor output, projection (`TestEncoder`, first and last window of each case) | rms 5.6e-4–9.0e-4 |
| the 13 hidden states over each window's valid frames | rms 6.5e-4–5.1e-3 (synthwave's first window peaks at layers 9–10), max 0.36 |
| valid frames and window weights | exact |
| every window's four scores (`TestScores`) | within 0.0015 |
| each song's four scores | within 0.001: synthwave PQ 8.476 PC 6.619 CE 7.774 CU 8.062 |
| end to end (`TestScoreGenerate`), served (`ai.service`) | present, on the 1–10 scale |

**What it says.**

- The synthwave song scores PQ 8.48 and CU 8.06. ACE-Step 1.5's report
  quotes PQ 8.35 and CU 8.09 for the model, so our port's songs sit where
  the paper's do.
- White noise scores PQ 5.16, PC 2.07, CE 2.91 and CU 6.11.
- The lofi song scores PQ 7.92 and CE 7.15. The 200 s song with twelve
  identical verses scores PQ 7.78 and CE 7.21. The A10 techno track scores
  PQ 8.32.
- CU barely separates noise from music (6.1 against 7.8–8.1). PC and CE
  carry the difference, and PQ halfway.

**Cost.** On the device, a 10 s window costs ~17 ms over a batch of eight.
By stage:

| stage | share |
|---|---|
| the naive positional conv | 26% |
| conv1–conv3 | 23% |
| relbias | 6.5% |
| attention | 6% |

The 180 s techno song took 373 ms on the device. Adding it to the auto
score took a 30 s song's score from 0.67 to 0.77 s and a 200 s song's from
3.3 to 3.7 s, including the host resample. That is under 1% of a request,
so nothing was tuned. The positional conv as a GEMM is the first thing to
try if it ever matters.

### A14d — result (2026-10-13)

**Done and deployed.** SongEval runs on the device over MuQ's first six
layers. Every song's `quality` now carries its five whole-song scores as
`song` (1–5). They land within 0.0001 of upstream's fp32 from the same
input. The resampler choice alone moves them up to 0.05.

**What upstream runs.**

- **Input:** SongEval's `eval.py` (`models/SongEval-src` at `848fb2f`)
  does `librosa.load(sr=24000)`, which is mono and then soxr_hq.
- **MuQ** (`models/MuQ-large-msd-iter`, `muq` 0.1.0) runs over the whole
  song in one pass:
  - a 128-band log-mel: n_fft 2048, hop 240, reflect-centred, HTK,
    AmplitudeToDB, the last frame dropped, normalised by the checkpoint's
    stats;
  - two Res2d blocks: 3×3 convs at stride (2, 2), BatchNorm and ReLU, with
    a strided shortcut. They take 128×100 Hz down to 32×25 Hz at 512
    channels, then `rearrange("b c f t -> b t (c f)")` and a Linear
    16384→1024;
  - a wav2vec2-conformer. Each layer has a half-step FFN with SiLU,
    attention with a rotary embedding applied to the normed input before
    q and k, a convolution module (pointwise to 2048, GLU, depthwise
    kernel 31, BatchNorm, SiLU, pointwise), a second half-step FFN and a
    final LayerNorm.
- **`hidden_states[6]`** is the sixth layer's output, checked to be equal
  under transformers 4.57.1. muq 0.1.0 does not run under transformers 5:
  it builds its config as an `EasyDict`.
- **The head:**
  - an FFN 1024→4096→1024 with ReLU;
  - four 8-head self-attention layers, chained with no residuals;
  - the mean of the last over time beside the FFN's max over time;
  - fc to 5, tanh·2 + 3.

**Code:**

- **`songeval/` (new package).**
  - `model.go`: weights, BatchNorm folded into the convs and the depthwise
    conv.
  - `mel.go`: `Mono24k` (torchaudio's sinc at 2:1) and `Mel`, on the
    repo's `audio.STFT` and `audio.MelFilters`.
  - `gpu.go`: the graph.
  - `songeval.go`: `Load`, `Score`, `ScoreMono`, and the head's tail on
    the host.
- **The front end runs in chunks of 512 tokens with a 3-token halo.**
  - Each level is computed over its local range, and positions outside
    the song are zeroed, which is each conv's padding. Activations stay a
    few hundred MB at any length.
  - The 512→512 convs are a new `dit_gemm` mode, `A_CONV=3`: a 3×3
    implicit im2col with the tap offset carried across the K loop, one
    output frequency row per dispatch (`muq_conv2d_reg64`). The
    A_CONV=2/1 builds are byte-identical to before.
  - `muq_conv_in` runs the 1-channel convs in fp32, and `muq_res` the
    blocks' epilogues.
- **The conformer:**
  - `muq_act` (SiLU, ReLU, GLU), `muq_rotary` and `muq_dwconv`;
  - parakeet's LayerNorm and residual, and the LDS GEMM with
    bias-as-K-column;
  - the kokoro HD64 pack, and the DiT HD128 pack for the head.
- **Attention builds with `TAIL_MAX`** (`muq_attn_hd64_tailmax_w32`,
  `songeval_attn_hd128_tailmax_w32`). The plain builds took every score
  NaN in the head; the trap is in the findings below.
- **Wiring.**
  - `pipeline.LoadSongEval`, `Quality.Song`.
  - The backend loads it when the MuQ checkpoint and the head exist
    (`-music-songeval`, default on).
  - `MusicSongEval` in the API.
  - `cmd/ace -aesthetics file` now prints both models' scores.
- **Oracle:** `reference/dump_songeval.py` in `.venv-aes`, with muq,
  librosa 0.11.0, hydra-core and transformers 4.57.1 added. SongEval's
  head is restated inline rather than imported from the clone. The three
  cases:
  - the 30 s synthwave song;
  - the 120 s lofi song;
  - 12.5 s of noise.

  Each is scored twice, once from librosa's soxr input and once from
  torchaudio's.

**Gates:**

| gate | result |
|---|---|
| 48 → 24 kHz resample, normalised mel (`TestHost`) | rms 9e-8; mel rms 5.4e-6 |
| conv front end through the chunks and their halos, hidden_states[0] (`TestStages`) | rms 1.7e-4–1.9e-4 |
| hidden_states 1–6 | rms 2.0e-4–4.2e-4, max 7e-3 |
| the head's FFN | rms 1.4e-4–1.6e-4 |
| the five scores (`TestScores`) | within 0.0001 of fp32 from the same input; the resampler gap (soxr against torchaudio) is 0.024–0.049 |
| end to end (`TestScoreGenerate`), served (`ai.service`) | present, on 1–5 |

**What the gates turned up:**

- **The plain attention builds keep padding keys in the row max.** A
  padding key scores exactly 0. On a row whose real scores are all
  strongly negative, the max is 0, every real exp2 underflows, and the row
  is 0/0. SongEval's head attends over unnormalised features, whose
  scores run to thousands of log2 units. Every score came out NaN, and the
  first `TestScores` passed anyway, because `worst > bar` is false for
  NaN. The gate is now `!(worst <= bar)`.
- **The score is sensitive to tiny input changes, not just to the
  resampler.** A resampler that agrees with soxr to 45 dB, Kaiser-best
  64 taps, still moves a score by 0.02–0.03, and torchaudio's 6-zero Hann
  by 0.024–0.049. No resampler reproduces `eval.py` exactly. We keep
  torchaudio's, already ported, and treat ±0.05 as the score's own
  resolution.
- `muq` pulls in torchvision, whose PyPI build does not load against the
  CPU torch. It has to come from the CPU index, `0.25.0+cpu`.

**What it says:**

| song | coherence | musicality | memorability | clarity | naturalness |
|---|---|---|---|---|---|
| synthwave, 30 s (A11's source) | 3.73 | 3.67 | 3.63 | 3.61 | 3.53 |
| synthwave, generated by TestScoreGenerate | 3.91 | 3.81 | 3.96 | 3.76 | 3.68 |
| lofi, 120 s | 3.26 | 2.94 | 3.08 | 2.84 | 2.84 |
| 200 s, twelve identical verses | 3.83 | 3.58 | 3.83 | 3.49 | 3.61 |
| A10's techno, 180 s | 3.65 | 3.41 | 3.35 | 3.15 | 3.18 |
| white noise | about 1.07–1.10 on all five | | | | |

So it separates noise from music by a wide margin, as Audiobox's PC and CE
do. Among songs, the instrumentals (lofi, techno) score lower on clarity
and naturalness, the vocal dimensions.

**Cost.**

- On the device: 95 ms for 30 s, 315 ms for 120 s, 575 ms for 180 s.
- The host's resample and mel add about 0.1–0.5 s more.
- The whole auto score now takes 0.94 s on a 30 s song (was 0.77 s) and
  4.9 s on a 200 s one (was 3.7 s, 11% of that request). The LM's
  teacher-forced passes are still most of it.
- Memory: ~2.6 GB staged at the 600 s ceiling. Of that, ~0.4 GB is fp16
  weights, and the rest is activations for one pass over a 10-minute
  song.
- `ai.service`'s memory peak during the deploy's smoke test was 83.8 GB,
  against 70.2 GB earlier the same day. The swap slot's music footprint
  grew by Audiobox and SongEval.

### A14e — the set, built and scored (2026-10-13)

- **`cmd/acecal`** builds the set from `cmd/acecal/set.json` (20 requests,
  seeds 11 and 12, 30-240 s, 6 instrumental, one Spanish; 17 controls) and
  scores every item into `out/a14e/<id>.{mp3,flac,lat,json}`, resuming from
  whatever records exist. 57 items in ~20 min beside `ai.service`.
- **`pipeline.Rescore`** (`ace/pipeline/rescore.go`) scores a finished
  song against any request, or audio no request made (it encodes it):
  the alignment against the given request, the LM's half over the plan's
  codes, and over codes re-tokenized from the render (`render` in the
  record, for A-o16). For every song it reproduced the generation's
  score bit for bit (alignment, PMI, Audiobox, SongEval), and a rerun of
  `n02-11` matched the smoke run's record exactly.
- **What the controls are.** steps (1 or 2 DiT steps, the LM's plan kept);
  caption and lyric swaps (the song's audio, the other's conditions);
  a lyric loop (one line 24 times); audio loops (2-3 s tiled); cover at
  strength 0; white noise; silence. Noise, loops and silence are scored
  against their source song's request and plan.
- **First look, before the ratings.** The audio models catch noise and
  silence (PQ 5.0/6.7, CE 2.7/3.2, musicality 1.1-1.3 against 7.5-8.5,
  7-8 and 2.6-4.5 for songs). The LM's reward over the *plan* misses all
  of them (0.72-0.75: the plan is the source's); over the *render* it
  drops for noise and a 3 s loop (0.61, 0.60) but not silence (0.72).
  The 1- and 2-step songs score like songs on every component (latents
  60-70% off the 8-step render, so they are real renders): the listening
  says whether they are bad. Lyric swaps show in the DiT alignment
  (0.025, 0.034 against their sources' 0.28); caption swaps cost the
  reward only 0.04 and 0.14, lyric swaps 0.02 and 0.07. Silence's
  alignment is high (0.54): the instrumental issue.
- **The listening panel**: https://claude.ai/artifact/H2WVUjBnUW1F6gP5uwAJgz
  (private; db collections `clips` and `ratings`, doc id `cNN` = clip
  NN). 30 clips: the 20 seed-12 songs and controls c01-c04, c09-c14,
  shuffled; the clip-to-item map is `out/a14e/panel.json` (kept out of
  the page). Read the ratings with ArtifactData `list` on `ratings`.
- **Next**: the user rates; then Spearman per component, the weights,
  the PMI scale (from the stored raw `PMI`), plan vs render (A-o16), the
  flag's thresholds.

### A14e — the first ratings (2026-10-13)

The user rated all 30 panel clips in ~11 minutes (skimming 45 min of
audio); `out/a14e/ratings.json`, mean 3.6 for songs and 2.7 for controls.

- **The rater is consistent where the audio is plainly broken**: both
  audio loops and the noise got 1. The prompts with a song and a control
  side by side mostly agree too (1-step synth-pop 2 vs its song 4, lyric
  loop 2 vs 4, the 2-step country 4 = 4). Two exceptions: the 1-step
  orchestral rock got 5 where its 8-step sibling got 2, and both covers at
  strength 0 got 4-5. The covers were never broken (strength 0 is a
  text2music render over silence, a song like any other), a flaw in the
  control, not the rating.
- **Spearman against the ratings** (chance |rho| at 95%: 0.36 for 30,
  0.44 for the 20 songs):

  | component | all 30 | songs 20 |
  |---|---|---|
  | PMI caption, render codes | **+0.60** | **+0.56** |
  | PMI caption, plan codes | +0.39 | +0.42 |
  | LM reward, render / plan | +0.37 / +0.24 | +0.41 / +0.37 |
  | Audiobox CE | +0.44 | +0.33 |
  | Audiobox PQ, CU, PC | +0.30, +0.20, -0.05 | +0.04, +0.02, -0.08 |
  | SongEval (five) | +0.34 to +0.38 | +0.18 to +0.31 |
  | lyric alignment DiT / LM | -0.28 / -0.33 | -0.48 / -0.30 |
  | PMI lyrics, render / plan | -0.27 / -0.31 | -0.20 / -0.26 |

- **Readings.** Only the caption PMI over the render's codes clears
  chance on both sets: A-o16 leans to the render. The negative alignment
  and lyric PMI are the instrumental confound: instrumentals score high
  on alignment and the user rated them lower (3.1 against 3.85 for the
  sung songs). No component sees a 2 s loop (rated 1): every score of
  `c11` sits among the songs'. A loop needs its own detector
  (self-similarity), which no scorer here has. The caption PMI alone is
  not a flag either: the Spanish reggaeton (`n11-12`, rated 3) sits at
  -0.24, below the noise control `c14`.
- **Thin data.** One rater, 30 clips, many ties; enough to rank the
  components roughly, not to fit weights or thresholds.

### A14e — the loop measure (2026-10-13)

The ratings showed a 2 s tiled clip (rated 1) scoring like a song on every
component. **`ace/loops`** measures repetition on the audio, on the CPU, no
model: 64 log-mel bands (30 Hz-8 kHz) every 1024 samples, centred on the
song's mean and normalised; each frame against the frame 0.5-12 s later
(or one frame after that, so a period between frames costs nothing);
frames 50 dB under the loudest never match. Two measures:

- **repetition**: the share of frames ≥ 0.9 alike at the best lag;
- **the longest loop**: the longest stretch ≥ 0.98 alike at one lag (gaps
  under 0.25 s bridged) plus the lag, in seconds and as a share.

Chosen on a numpy prototype over the set, then ported (the Go numbers
within 0.01 of it). In `Quality.Loops` on every scored song and in
`Rescore`; `cmd/ace -aesthetics` prints it; `cmd/acecal -loops` added it to
the 57 records without the device. ~0.13 s for 60 s, 0.23 s for 120 s.
Not in the served `quality` yet (A14f maps fields one by one).

- **Tiles**: every 1-8 s tile of six songs scores 1.000, and 1.000 again
  after an mp3 round trip (the 8 kHz cap is what survives mp3; at 16 kHz
  and 0.99 an mp3'd 2 s tile fell to 0.34 coverage). A song whose second
  half is a 4 s loop scores 0.50. Silence 0. `go test ./ace/loops` holds
  these on synthetic audio.
- **Real songs repeat too, near-exactly.** The deep house `n04-12` has a
  48 s stretch 0.98 alike at 8 s (0.40 of it), the chiptune `n16-12` 16 of
  its 30 s (0.54). The LM's plans do not loop (the longest exact code
  repeat in any plan is 6.8 s): the DiT renders near-copies from codes
  that differ. The user rated both 2.
- **Against the ratings** (chance |rho| 0.44 at 20, 0.36 at 30):

  | measure | all 30 | songs 20 |
  |---|---|---|
  | loop share | -0.41 | **-0.69** |
  | loop seconds | -0.40 | -0.67 |
  | repetition | -0.35 | -0.57 |

  The strongest single component yet among the songs (the render's
  caption PMI was +0.56). Over all 30 it is weaker: the noise control
  (rated 1) has no loop at all, which CE catches instead.
- **No flag yet.** Loops score 1.00; the most looped real song 0.54, and a
  half-looped song 0.50. Where "poor" starts is the fit's, with the second
  panel's data.

### A15 — the plan: the XL base model (2026-10-15)

**What the user asked for.** Pivot the music vertical from
`acestep-v15-xl-turbo` to the base model, then look for ways to cut
generation time or raise concurrent throughput. The user's answers on
2026-10-15:

- the checkpoint is **`ACE-Step/acestep-v15-xl-base`** (read at
  `220c116`), not `-xl-sft`;
- the default is **Gradio's for a pure base model: 32 steps**, and S3
  prices fewer;
- **turbo stays available as a flag**, as the control and the fallback.

**What changes, and what doesn't.** The architecture is the same.
`config.json` differs from turbo's in `is_turbo` (false) and
`model_version` ("base") only. The safetensors index is the same 829
tensors at 19,949,242,904 bytes, so `ace/dit`, `ace/vae` and the LM
carry over unchanged. `silence_latent.pt` ships again, byte-identical
to turbo's. The weights were already on disk:
`models/ace-checkpoints/checkpoints/acestep-v15-xl-base` (and `-xl-sft`),
fetched 2026-10-15 00:00–00:10. All four shards match HF's LFS sha256s
(checked 2026-10-15). What changes is the
sampler:

| | turbo (A0–A14) | base (Gradio's pure-base defaults) |
|---|---|---|
| steps | 8, a fixed table per shift | **32**: `linspace(1, 0, N+1)` then `shift·t/(1+(shift−1)·t)` |
| shift | 3 | 3 |
| guidance | none (CFG distilled in) | **CFG 7.0 through APG** (`apg_guidance.py`) |
| forwards a song | 8 | **64**: a conditional and an unconditional row a step |
| DCW | on | **off** (`_resolve_dcw_enabled` → `is_turbo`) |
| CFG interval | n/a | 0–1 (every step) |
| tasks | text2music, cover, repaint | adds lego, extract, complete |

- **The step default is a trap, as the shift was.** `GenerationParams`
  and `release_task` default `inference_steps` to 8 on any model, so
  upstream's literal API default on base is 8 CFG steps. Gradio sets 32
  for a pure base model, 50 for SFT and 8 for turbo
  (`ui/gradio/events/generation/model_config.py`). The HF card says 50.
- **APG.** The unconditional row's condition is `null_condition_emb`
  broadcast to every row of the packed condition, with the same mask.
  Then, per step:
  - `diff = cond − uncond`;
  - a momentum buffer, `avg = diff + (−0.75)·avg`;
  - the diff's L2 norm is clamped to 2.5, `dims=[1]`, which is the
    **time** axis of `[B, T, 64]`;
  - the clamped diff's component parallel to `cond` is projected out
    (fp64 in upstream);
  - `v = cond + (g − 1)·orthogonal`.

  This is host arithmetic on `[T, 64]`, like DCW was.
- **Euler throughout.** The last entry is t = 0, so the last step's
  `x − v·(t − 0)` is the turbo path's x0 jump in other words. N entries
  are still N steps.

**What it costs (estimate; A15e's baseline replaces it).** A forward is
~0.22 s at 60 s of song (750 tokens) and ~0.86 s at 4 minutes (6.9 s / 8,
§ After closing). At 64 forwards:

| song | turbo DiT | base DiT | LM (int8, A10) | request, turbo → base |
|---|---|---|---|---|
| 60 s | 1.8 s | ~14 s | ~7 s | 14.4 → ~27 s |
| 4 min | 6.9 s | ~55 s | ~26 s | ~35 → ~85 s |

**The DiT becomes the request**, where the LM was before. The speed work
moves with it.

**The stages:**

| stage | what | gate |
|---|---|---|
| A15a | Weights and oracle. Go reads `models/acestep-v15-xl-base` (the files moved there, the checkpoint dir symlinks them as turbo's does; sha256s already checked); `dump_ace_dit.py`, `dump_ace_detok.py` and `dump_ace_tasks.py` grow a `--model` and rerun on base, in fp32 and `--bf16` | the oracle's outputs exist, and the turbo dumps are untouched |
| A15b | The sampler in `ace/plan`: any N with shift, CFG through APG (ADG behind a flag, plain CFG for completeness), the CFG interval, DCW off; the stepper takes a two-row velocity | **bit-exact** against upstream's own loop with a stub decoder, as A1 did, over N ∈ {8, 32, 50} × shift {1, 3} × APG/ADG × an interval |
| A15c | The DiT with CFG, as two forwards a step (the control the speed stages are measured against): `Begin` holds both condition sets' cross planes, `null_condition_emb` staged | end to end against upstream fp32, priced against its bf16. **The risk stage**: CFG scales the fp16 drift in `cond − uncond` by up to g − 1 = 6, less APG's clamp. If base's layer-0 q/k drift lands over bf16, A-o7's split-precision QK is built here |
| A15d | Everything else that reads the DiT checkpoint, re-gated on base: the detokenizer (the LM's hints), the tokenizer, cover/cover-nofsq/repaint (`cover_steps` and the injection cutoff now follow `infer_steps`), the A14 score (`CrossProbs`; `t_last = 1/steps` moves with N) | the A11/A14 gates' oracles rerun on base |
| A15e | Served: `serve -music-dit base\|turbo` (base the default), `cmd/ace -dit`. `/v1/music` honours `inference_steps`, `guidance_scale`, `use_adg`, `cfg_interval_start/end` and `shift` on base, which turbo ignores. The estimate is rated per model. A baseline table (turbo vs base, 30 s / 60 s / 4 min, per stage) | the API tests; a listen; deployment is the user's call |
| A15f | Optional: lego, extract, complete (the base model's tasks, refused today) | upstream's handler, one case each |

**Speed: one request (after A15e's baseline).**

- **S1 — skip the unconditional half's cross-attention.** Every row of
  the unconditional condition is the same vector. After
  `condition_embedder` and the k norm, every key and every value is the
  same, and no row is masked with one request. So the softmax is uniform,
  and the cross-attention's output is `o_proj(v_null)`: **one constant
  vector a layer, whatever x and t are**. We compute it once at load and
  add it as a bias. The unconditional half then skips its cross q, the
  attention and the o projection. Estimate ~8–10% of a step. It is the
  same maths in a different order, so it is gated against A15c to
  rounding.
- **S2 — run the CFG pair as one forward of 2T rows.** The weights are
  read once, and a 60 s song's 750 rows become 1,500, past the wave32
  LDS GEMM's ~1,000-row knee ([[lds-gemm-needs-a-full-machine]]).
  Self-attention runs as two groups. Most of the gain is at ≤ 60 s, and
  about none at 4 minutes. (A side note: layer 0's self-attention comes
  before the first cross-attention, so it is the same for both rows.)
- **S3 — the step budget, priced by A14.** 32 against 24 and 16 steps,
  and a CFG interval that drops the unconditional row on the last steps
  (each such step costs half). This is the biggest lever, and it changes
  the output, so it is decided by A14's components on the `acecal` set
  and by the user's ears, not by a gate.

**Speed: concurrent throughput.** Today there is one music worker
(`MusicJobs`), and a job holds the device a step at a time.

- **T1 — overlap the LM and the DiT across two jobs.** Job B's LM step is
  latency-bound (22 ms of thin GEMVs), and job A's DiT forward is
  compute-bound. Overlapped, the LM's time disappears into the DiT's:
  throughput up to (LM + DiT) / DiT, about 1.5× at 60 s. First, measure
  whether two submissions overlap on this GPU at all. Then the cost: a
  second KV cache and a second activation set.
- **T2 — batch the LM across jobs.** Two jobs' CFG pairs make 4 rows. At
  the bus, a step should cost about the same 22 ms, which would double
  the LM's throughput. `llm_gemv` stops at 3 rows, so it needs a 4-row
  rung. This only pays if T1 shows the LM still showing through.
- **T3 — batch DiT rows across jobs.** This only helps short songs, after
  S2 has already doubled their rows. Lowest priority.

### A15a–c, S1 — result (2026-10-15)

**A15a.** The weights were already on disk and match HF's sha256s. They
moved to `models/acestep-v15-xl-base/` (with turbo's
`silence_latent.safetensors`; the `.pt` is byte-identical), and the
oracle's checkpoint directory symlinks them, as it does turbo's. Most of
the checkpoint is shared with turbo, byte for byte: the detokenizer, the
audio tokenizer (pooler, FSQ) and nearly all of both encoders. What
differs is the DiT, `condition_embedder`, `text_projector`,
`null_condition_emb`, and the first and last tensors of each encoder. So
A4's and A11c's gates (detokenizer, tokenizer) carry over unchanged.

**A15b: the sampler is bit-exact.** `plan.BaseSchedule`,
`plan.SampleGuided` and `plan.Guidance` (`ace/plan/cfg.go`) were run
against `reference/dump_ace_base_sampler.py`: upstream's base
`generate_audio` over a stub decoder whose conditional and null rows
differ. All 11 cases come back bit for bit: 8–50 steps, shifts 1–3,
guidance 1/3/7, an interval, DCW, and 37 to 3,000 latents. Four torch CPU
details had to be matched:

- `linspace` is one fused multiply-add per entry (its vectorised loop),
  counted down from the end for the second half;
- `tensor.norm(dim=1)` sums its squares in order in fp32;
- `tensor.sum(dim=1)` over a non-innermost dimension is SumKernel's
  four-level cascade (`plan.cascadeSum`); this applies to the projection's
  fp64 dot product;
- `2.5 / tensor` is `reciprocal() · 2.5`, so it rounds twice.

The oracle also records what `generate_music` hands the base model at
Gradio's settings: 32 steps, guidance 7, APG, interval 0–1, DCW off, Euler
(`TestBaseRequest`). ADG is not ported: a `use_adg: true` is a 400.

**A15c: the DiT with CFG, without split precision.**

- Code: `dit.GPU.BeginNull` stages the null condition's cross planes and
  `StepNull` reads them. They are only allocated on a guided checkpoint
  (`is_turbo` false in its config.json, `Guided()`).
- The oracle is `dump_ace_dit.py --model base [--bf16] full_metas`
  (30 s, 375 tokens, 32 CFG steps; about 3 min on the CPU).
- The gate is `ace/dit`'s `TestBaseDiT`. Run the base tests on their own
  (`-run Base`), since they stage a second checkpoint.

| | ours vs fp32 (rms) | upstream bf16 vs fp32 | bf16 / ours |
|---|---|---|---|
| encoder states | 3.2e-4 | | |
| first forward, conditional row | 4.4e-3 | 1.54e-2 | 3.5× |
| first forward, null row | 3.8e-3 | 1.71e-2 | 4.5× |
| forwards 1–31, either row | 0.5–2.7e-3 | | |
| latents after 32 guided steps | 7.4e-2 | 0.302 | 4.1× |

- **Base's forwards are 10× tighter than turbo's** (turbo: 3–7% rms on a
  velocity, all from layer 0's q/k).
- The trajectory still amplifies drift: 0.4% on a forward becomes 7% on
  the latents over 32 steps, and upstream's bf16 lands at 30%.
- The activations stay in turbo's range. fp16 operands peak at 5.4e3, and
  the residual reaches 6.35e5 (fp32).
- **A-o7's split-precision q/k is not needed.**

**S1, done: the null row's cross-attention as one constant a layer.**
`GPU.nullCross` computes `o_proj · (each q head's kv head of v_proj ·
(condition_embedder · null_condition_emb + bias))` once at load, in fp32
from the checkpoint's fp32 weights with float64 sums. The null forward
then adds it through `dit_gate_add_hoist_bcast.spv` (the gate kernel's
`BCAST` build: y is one row), in place of the cross norm, q projection,
attention and o projection.

- It is on by default; `nullConst` off runs the null planes as the control.
- `TestBaseNullCross` holds it against the control and the oracle: S1 is
  1.5–3.4e-4 rms from the control, and as close to upstream as the
  control or closer.
- The null forward goes **111.4 → 90.8 ms at 375 tokens (1.23×)**, so a
  CFG step is 9% shorter.

**What a forward costs now** (`TestBaseTiming`, the median of 5):

| song | tokens | conditional | null (S1) | CFG step | per token-row |
|---|---|---|---|---|---|
| 15 s | 188 | 90.1 ms | 74.0 ms | 164 ms | 0.437 ms |
| 30 s | 375 | 111.0 | 89.7 | 201 | 0.268 |
| 60 s | 750 | 178.8 | 144.5 | 323 | 0.216 |
| 2 min | 1,500 | 345.7 | 282.0 | 628 | 0.209 |
| 4 min | 3,000 | 731.4 | 604.1 | 1,336 | 0.223 |

The big GEMMs run at ~42 TFLOP/s (gate|up at 60 s), so the DiT is
compute-bound from 750 tokens up. **S2 (the CFG pair as one forward of 2n
rows) is therefore worth little.** At the 2n rate it would save ~1.6× at
15 s, ~1.24× at 30 s, ~1.03× at 60 s and nothing beyond. It is not built;
it is worth building only if short songs matter. Past 60 s, only fewer
FLOPs help (S3).

**The first base song** (`cmd/ace -dit base`, a 60 s synthwave song,
thinking, before S1) took 28.1 s: the LM 11.9 s, the DiT 12.7 s (64
forwards), the VAE 0.6 s. Turbo rendering the same plan took 17.3 s (DiT
1.7 s). Their scores, base then turbo:

| | base | turbo |
|---|---|---|
| PQ | 8.42 | 8.32 |
| CE | 7.47 | 7.60 |
| SongEval coherence | 3.95 | 3.84 |
| SongEval naturalness | 3.47 | 3.69 |
| DiT lyric alignment | 0.40 | 0.18 |

The songs are `out/a15/{base,turbo}-synthwave-60.mp3`. S3's sweep scores
the whole A14e set.

**A15d: the audio tasks on base, gated.** The oracle is
`dump_ace_tasks.py --model base [--bf16] cover cover_mix repaint` (32
steps, CFG 7; about 3 min a case on the CPU). The gate is `ace/pipeline`'s
`TestBaseTasks`. Results, against upstream fp32:

| case | ours (rms) | upstream bf16 | bf16 / ours |
|---|---|---|---|
| cover_mix (renoised source, condition switch at step 7 of 14) | 4.85e-3 | 0.186 | 38× |
| repaint | 5.7e-2 | 0.416 | 7.3× |
| cover, from the oracle's hints | 0.116 | 0.721 | 6.2× |
| cover, through our tokenizer | 0.46 | 0.721 | 1.6× |

- **The cover's gap is one code.** One code in 150 flips at an FSQ
  boundary (as on turbo, § A11b–e). That replaces 5 hint rows, and 32
  steps of CFG 7 carry it from turbo's 0.13 to 0.46. Upstream's own
  unseeded posterior noise flips ~2, and bf16 flips half the codes. So
  the full cover is held from the oracle's hints, and the flips stay with
  the tokenizer's gate (`TestTaskInputs`).
- The pipeline changes behind these:
  - a base schedule has steps + 1 entries;
  - a cover's renoise searches all but the last;
  - `cover_steps` and the injection cutoff count steps;
  - base injects after every step under the cutoff (turbo stops before
    its last);
  - `condition()` rebuilds the null planes at the new condition's length
    when a cover switches to text2music.
- **The A14b score on base:** `t_last = 1/steps` follows the request
  (`Rescore` uses 32 on base). `CrossProbs` reads the conditional planes
  only, as upstream's `lyric_score` does. The implementation is gated on
  turbo; base runs the same code over its own weights. Its alignment
  heads are the same config entry.

**A15e: served, not deployed.**

- `serve -music-dit base|turbo` (default base) and `cmd/ace -dit`,
  `-steps`, `-guidance`, `-cfg-start`, `-cfg-end`, `-shift`.
- `/v1/music` reads `inference_steps` (1–200, default 32),
  `guidance_scale` (0–30, default 7) and `cfg_interval_start`/`_end`. On
  base, a `timesteps` list is taken as given (at least 2 entries).
  `use_adg: true` is a 400.
- **A request picks the model (2026-10-15, the user's request).**
  - `-music-dit base,turbo` (the default) stages both DiTs through
    `Pipeline.AddDiT`. `UseDiT` switches between them per job, which is
    safe because jobs are serial on the device.
  - `model` (and upstream's `model_name`/`dit_model`) matches an id, the
    short name, or a word in the name. Anything else runs the first
    model, as upstream's `select_generation_handler` falls back to its
    primary, and the job's `model` reports which ran.
  - Measured served: a 20 s song in 1.5 s on turbo (8 steps) and 7.0 s
    on base (32 steps), one after the other in one server.
  - Music's residency is ~31 GB with both DiTs and the LM (scorers off),
    ~10 GB over base alone. That is still under video's 60 GB in the same
    swap slot.
- The model ids are the checkpoints' directories (`acestep-v15-xl-base`,
  `acestep-v15-xl-turbo`).
- The estimate counts steps × 1.81 forwards.
- `research/api-jobs.md` is updated.
- `ai.service` has not been redeployed: that is the user's call, and it
  will change the deployed model's default.

**S3, the sweep (2026-10-15): fewer steps cost nothing measurable, and
turbo outscores base.** `cmd/acesteps -out out/a15/steps` covered the
A14e set's 20 songs (30 s to 4 min, 29 min of audio, seed 11). Each song
was planned once by the LM. Every arm rendered the same codes from the
same noise and went through the whole auto score. The table gives each
arm's mean, then the mean paired difference from b32 and on how many of
the 20 songs the arm scored at least b32's:

| component | b32 | b24 | b16 | b32, CFG for t ≥ 0.5 | turbo |
|---|---|---|---|---|---|
| Audiobox PQ | 7.91 | −0.02 (6) | −0.06 (4) | −0.01 (11) | **+0.28 (17)** |
| Audiobox CE | 7.20 | −0.02 (5) | −0.07 (4) | −0.02 (7) | **+0.29 (18)** |
| Audiobox CU | 7.77 | −0.01 (6) | −0.04 (6) | −0.01 (13) | +0.14 (12) |
| SongEval coherence | 3.60 | +0.02 (14) | +0.07 (16) | +0.04 (15) | **+0.29 (17)** |
| SongEval musicality | 3.39 | +0.02 (14) | +0.07 (15) | +0.05 (16) | **+0.27 (17)** |
| SongEval naturalness | 3.27 | 0.00 (12) | +0.05 (15) | +0.03 (17) | **+0.24 (16)** |
| render reward (adherence) | 0.725 | 0.000 | +0.002 | −0.001 | −0.004 |
| DiT lyric alignment | 0.233 | +0.003 | −0.001 | 0.000 | −0.001 |
| loop share | 0.047 | −0.013 | −0.004 | −0.001 | −0.007 |
| DiT s a second of song | 0.227 | 0.171 | 0.113 | 0.206 | 0.028 |

(PC, memorability and clarity move the same way. The DiT times of the
first ~half of the base pass ran beside a CPU oracle and read ~1.5×
slow, so compare them across arms, not with `TestBaseTiming`.)

- **The step budget.** Between 32 and 16 steps, Audiobox drops 0.04–0.07
  and SongEval rises 0.05–0.07. Both are well inside the spread between
  songs, and adherence does not move. 24 steps is indistinguishable from
  32. Restricting CFG to t ≥ 0.5 is neutral, but it only saves 10% of a
  step's work.
- **On these metrics, base is not better than turbo.** Turbo leads every
  production and musicality score, by ~0.25 on 16–18 of the 20 songs.
  Adherence (the LM's verdict over the render, lyric alignment) is a
  tie. This matches the HF card, which rates turbo's quality "Very
  High" and base's "High", with base's own advantage in diversity and
  tasks.
- These are automatic scores. Panel 1 found they track the user's
  ratings only partly (§ A14e). **Which model to serve is the user's
  ears' call**, and the renders are there to be heard:
  `out/a15/steps/<song>-<arm>.mp3`.

**T1, measured: the LM and the DiT do overlap.** A 4-minute song's LM
phase 2 (`cmd/ace`, turbo) ran in one process while `TestBaseTiming`
looped in another:

- the LM step went 25.1 → 68.9 ms;
- the DiT's forwards ran within 0–5% of their speed alone (0.223 ms a
  token-row at 4 min, as alone);
- the pair's work, 30 s of LM and ~79 s of DiT alone, finished in ~83 s,
  against ~109 s back to back.

So the LM's work fits into the DiT's gaps. That is ~1.3× the throughput
of the two, at the cost of the planning job's latency.
`backend.Device`'s premise ("two of them interleaved would not go
faster") holds for two compute-bound models, but not for a latency-bound
decode beside a compute-bound one. Building T1 means a second queue (or
device) for the LM, which changes the server's one-queue, one-lock design
for every vertical. **That is the user's call** (A-o17).

## Open questions

- **A-o1: does fp16 hold in the DiT's activations?** Answered above: fp32
  residual (5.7e5), fp16 operands (≤ 2.45e3).
- **A-o2: the LM's decode path.** Answered (A7, A10): both options at once. The
  weight is staged once in the fragment tiling that the DiT GEMM rungs
  (prefill) and `llm_gemv.comp` (1–3 rows) both read. Attention is a new
  split-key kernel over a KV cache. 45 ms a step, bandwidth-bound. A10 made
  the layers int8 (the L8a bank, the same two kernels' `-DQ8B` arms): 22 ms
  a step, at the bus.
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
- **A-o14: the served shape of the score (A14f).** Upstream's API has
  none (Gradio only) and OpenAI has no music endpoint
  ([[api-follows-openai-standard]]). **Partly answered by the user
  (2026-10-13): on by default.** A14b serves it as `quality` on the job,
  filled before `completed`, with Gradio's `auto_score: false` to turn it
  off. The flag and the outside models' fields are still to come.
- **A-o15: are non-commercial weights acceptable?** SongEval's head
  (CC BY-NC-SA per its README) sits on MuQ (CC-BY-NC 4.0). **Answered by
  the user (2026-10-13): yes**, so A14d is in.
- **A-o17: a second queue for the music LM (A15 T1)?** Measured: a
  latency-bound LM decode beside the compute-bound base DiT costs the DiT
  0–5% and finishes the pair ~1.3× sooner than back to back, but each LM
  step is 2.75× slower while they overlap. Serving it means two queues,
  or two devices, where `backend.Device` has one queue behind one lock.
  That is the user's call.
- **A-o16: score the plan or the render?** Upstream's PMI reads the LM's
  codes. Codes re-tokenized from the rendered latent score what was
  actually made. A14e decides between them.

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

**2026-09-27, session 6: A10 done for the LM.** The layers are int8 by
default and priced against upstream bf16. Host sampling is 1 ms, and a 60 s
song is 14.4 s (§ A10).

- Code: `ace/lm/gpu.go` (`Bank`, `BankQ8`, `tileBQ8`, `gemvSlabsQ8`, the
  q8 GEMM rungs, one submit a decode step), `ace/lm/sample.go` (bucketed
  top-p, `maxScore`), `-lm-fp16` on `cmd/ace`, `-music-lm-fp16` on
  `serve`, and `backend/music.go`'s estimate rates by bank.
- Oracle: `reference/dump_ace_lm_tf.py` → `reference/out/acelmtf` (fp32
  logits at every teacher-forced step, bf16's KL per step, FSM-restricted in
  phase 1). It reads `reference/out/acelm`, so rerun it after
  `dump_ace_lm.py`.
- Tests: `ace/lm/bank_test.go` (`TestBankAgainstBF16`; `TestSlabLadder`
  under `ACE_LM_LADDER=1`), `ace/lm/sample_bench_test.go`
  (`TestTopPBucketed`, `BenchmarkCodeStep`, `BenchmarkTextStep`).
  `ACE_LM_BANK=fp16` runs the package's shared staging as fp16. `TestLogits`
  always stages fp16.
- Not deployed: `ai.service`'s unit line still has no `-music`. That is the
  user's call (session 4).

**2026-09-27, session 7: A11 done.** Audio in: a reference's timbre on
any request, and cover, cover-nofsq and repaint (§ A11). Try
`go run ./cmd/ace -task cover -src song.mp3 -caption "…"`, or
`POST /v1/music -F task_type=repaint -F src_audio=@song.mp3 …`.

- Code:
  - `ace/vae/encoder.go` (`Encoder`, `Sample`, `ClampPeak`);
  - `shaders/ace_vae.comp` `SPLIT` (`ace_vae_snake_split.spv`);
  - `ace/dit` `Tokenize` (+ `poolL`, `tokIn`, the row-mod-6 rope tables,
    `Host.TokB/PoolSpecial/PoolNorm/FSQInW/FSQInB`);
  - `ace/plan.FSQIndex`;
  - `ace/pipeline/tasks.go` (`Audio`, `ReadAudio`, `ReadAudioBytes`,
    `ReferenceClip`, `taskPlan`), and `generate` now runs from a `ditPlan`;
  - `api/music.go` (multipart audio, the task fields, `readTask`);
  - `backend/music.go` (`musicAudio`, `encodeCost`);
  - `cmd/ace` (`-task -src -ref -start -end -cover-strength -cover-noise
    -repaint-mode -repaint-strength -explicit-mask`).
- Oracles:
  - `reference/dump_ace_vae_enc.py` → `reference/out/acevaeenc`;
  - `reference/dump_ace_tasks.py [--bf16] [label …]` →
    `reference/out/acetasks{,_bf16}`. It merges into its manifest, reads
    `acevae`'s audio, and `--bf16` reads the fp32 run's noise.
- Tests:
  - `ace/vae/encoder_test.go`;
  - `ace/dit/tokenize_test.go`;
  - `ace/pipeline/tasks_test.go` (`TestReferenceClip`, `TestTaskInputs`,
    `TestTasks`, `TestTasksOwnEncode`, ~40 s);
  - `api` `TestMusicAudioTasks`.
- Not deployed: as before, `-music` is not in `ai.service`.

**2026-09-27: closed.** This file was frozen to `research/music-vertical.md`
at the user's request. What was left is in the archive note at the top and
in `TODO.md`'s music section.

## After closing: where it stands, what is open

*Moved here from the root `TODO.md` when it was retired on 2026-10-02.*

**Where it stands.** Closed 2026-09-27: A0–A12 are all done, and the plan
ran out. ACE-Step 1.5's three generators (the 5 Hz LM, the 4 B DiT, the
Oobleck VAE) run upstream's default thinking path, sample mode, and the
turbo model's audio-in tasks (cover, cover-nofsq, repaint, a reference's
timbre). Every gate is against upstream's handler in fp32, with the drift
priced against its bf16. `serve -music` answers `/v1/music` as
submit-then-poll jobs (multipart audio in), sharing the swap slot with
image and video; `-music` is on the deployed `ai.service` line. Code: `ace/` (`plan`, `dit`, `vae`, `lm`, `pipeline`),
`api/music.go`, `backend/music.go`, `cmd/ace`.

**Open, none of it planned:**

- **Listen** to `out/ace-a10-int8-*` (the int8 LM) and `out/ace-a11-*`
  (a cover, a repaint, a referenced song). The user listened after A8
  ("sounds great") but not since.
- **Speed past the LM**: the DiT (1.8 s of a 60 s song, 6.9 s of 4 min) and
  the VAE are ≤ 20% of a request, and the LM's step is at the bus. A 4-bit
  bank would be the next byte cut, and phase 1 has no headroom under bf16
  for it.
- Upstream's retake and flow-edit, and the base model's tasks (lego,
  extract, complete), stay refusals.
- **A14, a quality score on every song**: planned 2026-10-13; see
  § A14 — the plan (A14a–f).
  - **A14a and A14b are done** (same day): upstream's auto score is
    served as the job's `quality`, on by default. See § A14b — result.
  - **Deployed** on `ai.service` the same day.
  - **A14c is done and deployed** (same day): Audiobox Aesthetics' PQ,
    PC, CE and CU on every song, within 0.0015 of upstream. See § A14c —
    result.
  - **A14d is done and deployed** (same day): SongEval's five whole-song
    scores on every song, within 0.0001 of upstream from the same input.
    See § A14d — result.
  - **A14e under way**: the set is built and scored, the user rated the
    panel (§ A14e — the first ratings), and `ace/loops` measures the
    repetition nothing else saw (§ A14e — the loop measure). Calibration
    paused 2026-10-13 at the user's call; next, when it resumes: a second
    panel (repeats for consistency, more songs), then the fit, the headline
    number and the good/fair/poor flag.
  - **A14f in part (2026-10-13)**: `quality` now carries every component:
    `render` (the LM's verdict over the render's own codes, A-o16's
    numbers, on any path with the LM loaded, covers included) and `loops`
    beside the rest; no headline number, no flag, by the user's choice.
    Scoring a 30 s song went 0.9 → 1.8 s. `cmd/ace` logs both. Not
    deployed yet. Still to do from A14f: `-score-only` on any file.
- **A15, the XL base model** (2026-10-15, the user's request):
  - A15a–e are done: the sampler is bit-exact, the DiT and the tasks are
    gated under bf16, and it is served as `-music-dit base` (now the
    default), not deployed;
  - S1 is done (the null cross-attention is a constant, 1.23× on the
    null forward);
  - S2 is priced and not built (≤ 30 s only);
  - S3 swept: 16 steps cost nothing measurable, and **turbo outscores
    base on every quality component**, so which to serve awaits the
    user's listening;
  - T1 is measured (1.3×) and needs a second queue (A-o17);
  - A15f is not started.

  See § A15 — the plan, and the result sections after it.
- **A13, understand (captioning existing audio)**: done 2026-10-12, see
  § A13 — result. Served as `full_analysis_only` on `/v1/music`, and
  deployed: the `ai` binary of 2026-10-13 20:15 postdates it.
