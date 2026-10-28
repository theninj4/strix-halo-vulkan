# MUSIC 2 — MiniMax-Music3 beside ACE-Step 1.5 (N-stages)

> **Live.** Planned 2026-10-23; N0 a go (N-d8), N1–N6 done on the device
> 2026-10-23/24, N7 end to end. Stage letters `N0…N10`,
> decisions `N-d1…` and open questions `N-o1…` resolve here. ACE-Step's
> record (A-stages) is [`music-vertical.md`](music-vertical.md); nothing
> in it changes. ACE stays the default and keeps every task it serves.

## What we are building

A second music model on `/v1/music`, chosen per request by `model`:
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3).
Caption + lyrics in, a stereo song of up to 6 min out. Text to music only:
the open weights take no audio.

The reference is the diffusers modular pipeline in
[huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456),
read at `dafe3733` (unmerged as of 2026-10-23). SGLang-Omni and ComfyUI
also run it; diffusers is the readable one.

## The model, from the configs and the reference

| part | shape | file (dtype) |
|---|---|---|
| **Global LM** | Qwen3-8B dense: 36 layers, hidden 4096, 32/8 heads × 128, FFN 12288, vocab 200,000, untied head | `language_model/` 17.2 GB (bf16) |
| **Local LM** (`rvq_depth_decoder`) | 4 layers, hidden 4096, 16 heads × 256, FFN 6144, RMSNorm, causal; 7 heads of 1,024 codes, 7 code embedding tables | 1.29 GB (bf16, ~0.65 B) |
| **Condition encoder** | softmax-weighted sum of 8 hidden states a frame (layer_weight_logits), a scale, Conv1d k3 4096 → 2048, nearest resample 25 Hz → 86.13 Hz (`frames · 44100/24000 · 960/512`) | 100 MB |
| **DiT** (`transformer`) | 36 blocks, 32 heads × 64, FFN 8192 gated, LayerNorm, rotary on 32 of 64 dims, Fourier timestep embedding prepended as a token, **self-attention only**; input is `[latents 128, zeros 128, condition 2048]` through a 1×1 conv + residual | 9.7 GB (fp32, ~2.4 B) |
| **Vocoder** (Flow-VAE decoder) | DAC-style: Snake, weight-normed Conv1d, ConvTranspose ×8·8·4·2 = 512 hop, tanh; the 128 latent channels are two 64-channel mono latents decoded separately | 216 MB |
| not downloaded | `qwen_7B/` (the same LM + Local LM as `AbabForCausalLM`, ~17 GB), `dav.pth`, `flowmatching_vae.pth` (pickles, ~10 GB: the conversion script's sources) | — |

**Output rate is unresolved (N-o1)**: the card says "32 kHz, 16-bit
stereo"; the diffusers vocoder config and decode step say 44.1 kHz. N0
settles it from the reference's output and `assets/minimax_ttm.wav`.

### The pipeline, as the reference runs it

1. **Prompt.** `_clean_caption` (markdown stripped, special tags rewritten)
   and `_normalize_lyrics` (tags lowercased, text on a tag line dropped,
   `[start]` prepended), assembled with `<|caption_start|>`,
   `<|lyrics_start|>`, `<|audio_start|>`. The card: whitespace-level changes
   change the audio. The unconditional row is the same ids with
   `[1:-2]` replaced by `_AUDIO_CFG_TOKEN_ID` (151654). ≤ 5,000 tokens.
2. **Frame loop** (25 frames/s, ≤ 9,000 frames, ~360 s), two rows
   throughout:
   - Global LM head masked to the 16,384 semantic codes (offset 151675)
     plus end-of-audio (151670); guided = `u + 1.5·(c − u)`, restricted to
     the conditional row's top 50, then top-k 50 sampling. End-of-audio
     stops the loop. The first step only advances past `<|audio_start|>`.
   - Local LM: `[proj(last_hidden), proj(embed(semantic))]`, then 7
     sequential passes, each over the whole ≤ 9-token sequence (no cache
     upstream), each guided 1.5 and top-k 50, the code repeated into both
     rows. The conditional row's 7 hidden states are kept.
   - Feedback: `(embed(semantic) + Σ audio_embeddings(c1..c7)) · 8^-0.5`.
   - Per frame the condition input is `[last_hidden, 7 depth hiddens]`,
     8 × 4096.
3. **Windows**: 200 frames (8 s, ~689 latents) every 100 frames (4 s).
   Each window: condition encoder, fresh noise, 30 Euler steps on
   `linspace(1, 1/30, 30)` (scheduler `invert_sigmas`, t = 0 is noise),
   CFG 1.7 against **zeros** for the condition. The first 172 latents of
   the overlap are blended toward the previous window's latents at every
   step and restored after; the carry is latents `[L−344, L−172)`.
4. **Vocoder** per window; crop 86 latents left (not the first window) and
   258 right (not the last); concatenate; clamp to [−1, 1].

## What it costs on this machine (estimate, to be replaced by N3/N5)

From our measured rates: ~187 GB/s for single-row decode reads (A10), 37–41
TFLOP/s for the dit GEMM (A3).

| part | per unit | per second of audio |
|---|---|---|
| Global LM step, int8 body + 16k-row head (~7.1 GB) | ~38 ms a frame | ~0.95 s |
| Local LM, 7 passes of ~0.65 GB int8 | ~25 ms a frame | ~0.6 s |
| DiT, 60 forwards of 2.4 B at ~689 rows (2 rows batched: 1,378) ≈ 200 TFLOP a window | ~5.2 s a window | ~1.3 s (every second is denoised twice) |

A 60 s song is ~2.7 min serial; 4 min is ~11 min serial, ~6.5 min if the
LM and DiT overlap (second queue, A-o17). ACE today: 14 s and ~40 s
(turbo), ~85 s (base, 32 steps). **The frame loop's ms a frame (N3) is
the number the vertical lives or dies on.**

Resident, int8 LM: ~7 GB layers, 0.13 GB head, 1.6 GB embeddings, 0.65 GB
Local LM, 4.8 GB DiT (fp16), ~0.3 GB condition + vocoder, KV cache 147 KB a
token a row (≤ 4.1 GB at 5,000 + 9,000 tokens × 2 rows), scorers 3.2 GB:
**~20 GB**.

## Decisions (so future sessions don't relitigate)

- **N-d1 (the user, 2026-10-23): ACE-Step stays the default.** A request
  that names no model, or a name nothing matches, runs ACE's default DiT,
  as today; the job's `model` says which ran.
- **N-d2 (the user): separate swap entries.** MiniMax is its own
  `Swappable` (`music-mm3`) in the slot beside music (ACE), image and
  video. Switching model is an eviction and a stage, not co-residency.
- **N-d3 (the user): metadata goes into the caption.** `bpm`, `keyscale`,
  `timesignature` (and `language`, if given) are written as a Global
  Metadata line in the card's own phrasing ("bpm is 92. key is E, and
  scale is minor."), prepended to the caption, not refused.
- **N-d4: selection is by `model`, through a router.** `backend.MusicRouter`
  implements `api.MusicBackend` and dispatches on the request's model:
  `MiniMax-Music3`, or a name with `minimax` or `music3` as a word, goes
  to MiniMax; everything else to ACE's `resolveModel` (`backend/music.go`).
  One `MusicJobs` queue for both. `serve -music-models
  ace-base,ace-turbo,minimax` (first is the default); `-music-dit` stays
  as ACE's alias.
- **N-d5: refusals name ACE.** On MiniMax, `thinking`, sample mode,
  `full_analysis_only`, the audio-in tasks (`src_audio`,
  `reference_audio`, cover, repaint) and the DiT schedule fields
  (`inference_steps`, `shift`, `guidance_scale`) are a 400 that says to
  use ACE.
- **N-d6: the scorers are shared, staged twice.** Audiobox, SongEval and
  `ace/loops` move out of `ace/pipeline` into a package both pipelines
  load; only one is resident at a time. MiniMax's `quality` has no
  `render` (that is ACE's LM). Resample to the scorers' rate if they need
  48 kHz.
- **N-d8 (the user, 2026-10-23): go.** After N0b (guitar: ACE turbo
  "incapable of rendering a guitar") and the card check, the user stopped
  N0c's MiniMax renders and called for the GPU implementation. N0c's ACE
  renders stay in `out/mm3-n0c/` for a later panel.
- **N-d7: N0 is a go/no-go.** If the matched renders don't clearly beat
  ACE on the A14 scores and the user's listening, the vertical stops at
  the download and the oracle.

## The stages

| stage | what | gate | status |
|---|---|---|---|
| **N0** | Weights (safetensors only, ~28.5 GB, sha256 against LFS); `.venv-mm3` with diffusers `dafe3733`; `reference/dump_mm3.py` (CPU fp32 and bf16, fixed seeds, dumps prompt ids, per-frame codes and hiddens, conditions, latents per step, waveform); N-o1. **Go/no-go**: 4–6 matched caption + lyrics pairs rendered on the CPU (~1 h per 30 s song), scored with the A14 scorers beside ACE base and turbo, and listened to | the user's call | panel rated 2026-10-23: **a tie with ACE**, not a clear win (N-d7); guitar panel N0b rated (MiniMax 2.50, turbo 2.00, base 1.00, all poor); pipeline matches the card; N0c stopped unrendered for MiniMax; **go (the user, 2026-10-23)** |
| **N1** | Global LM on the device from the `ace/lm` Qwen3 path at 8B: exact prompt, 2 rows, frame-embedding sum, head cut to 16,385 rows; int8 bank priced against bf16 (as A10) | teacher-forced logits per frame vs fp32; int8 drift under bf16's |**done 2026-10-23** (`mm3/lm.go`, `TestLMFrames`): fp16 CFG KL ≤ 7.8e-6; int8 mean 6.8e-4 vs bf16's 2.4e-3; 63 ms a 2-row step (int8, untuned; 45 after N3's ladder) |
| **N2** | Local LM with a KV cache (upstream recomputes; check bit-identity — `wmma-negative-zero-products`), 7 heads, 2 rows | teacher-forced codes and hiddens |**done 2026-10-23** (`mm3/depth.go`, `TestDepthFrames`): fp16 CFG KL ≤ 7.5e-7, int8 ≤ 8.1e-5; 42 ms a frame (int8, 7 passes, latency-bound) |
| **N3** | The frame loop on the device: masked guided top-k sampler, end-of-audio, per-frame yield, the condition's weighted sum folded in (4 KB a frame kept, not 32 KB). **Time it** | codes under the oracle's draws; ms a frame |**done 2026-10-23** (`mm3/frames.go`): oracle draws inside our support 392/392; **75 ms a frame, 1.88 s a second of audio** (int8, k32 ladder) |
| **N4** | Condition encoder + DiT forward on the LDS wave32 GEMM (1,378 rows, above its floor) and the G4b attention at head 64; partial rotary | per layer vs fp32 | **done 2026-10-23** (`mm3/dit.go`, `TestDiTForward`): guided velocity rel 3.3e-4; 216 ms a CFG forward |
| **N5** | Window sampler: 30 steps, CFG 1.7 vs zeros, overlap blend and carry | latents vs the oracle's, its noise | **done 2026-10-23** (`mm3/window.go`, `TestWindows`): final latents rel 4.6e-4 / 7.4e-4 (2 windows); 6.7 s a window |
| **N6** | Vocoder from `ace/vae`'s Snake/conv kernels, per-channel, per-window crop | waveform vs the oracle |**done 2026-10-24** (`mm3/vocoder.go`, `TestVocoder`): SNR 61–64 dB a window and on the cropped song; 220 ms an 8 s window |
| **N7** | End to end in `cmd/ace -model minimax` (or `cmd/mm3`) | the oracle's run reproduced; listening |**running 2026-10-24** (`cmd/mm3`, `mm3/pipeline.go`, `mm3/prompt.go`): prompt ids 44/44 cases; 30 s in 107 s (3.57 s a second), PQ 8.18 CE 7.60; N0c full-length panel rated 2026-10-24: MiniMax 4.50, turbo 3.50, base 1.75 |
| **N8** | Served: router, `music-mm3` swap entry, the request mapping (N-d3, N-d5), duration default 120 s and cap 360 s, `estimated_seconds` from measured rates, progress by frame and window, `quality` (N-d6) | API tests; one served song |**done 2026-10-24** (`backend/music_mm3.go`, `music_router.go`, `serve -music-mm3`, in `ai.service`): a 20 s song served in 75 s (estimate 66); `quality` refused for now |
| **N9** | Speed, each priced: fewer DiT steps (as S3), 4-bit LM bank, LM/DiT overlap (needs A-o17's second queue), streaming window by window | measured served | |
| **N10** | Deploy: one flag on the `ai.service` line | the user's call | |

### N0 — progress (2026-10-23)

- **Weights**: `models/MiniMax-Music3`, 28.5 GB, the nine safetensors
  match HF's LFS sha256s. `qwen_7B/` and the `.pth` pickles were not
  fetched.
- **Oracle**: `.venv-mm3` (see Handoff). `reference/render_mm3.py`
  renders set songs (`cmd/acecal/set.json`) at the set's seed 11. The
  modular index names the hub repo for every component, so the script
  repoints each `ComponentSpec` at the local copy; with `HF_HUB_OFFLINE=1`
  and no repoint, every component silently fails to load and the run dies
  at `language_model.model` (`NoneType`).
- **Scoring**: `cmd/acecal -ext DIR` scores another model's renders
  (`<id>-<seed>*.wav`) against their set songs through `pipeline.Rescore`,
  into `DIR/score`, the way the noise and loop controls are scored: no
  CoT, the request as asked, the render's own codes. Checked on ACE's own
  `n09-11`: Audiobox, SongEval and the loops reproduce its record within
  1e-4 (the flac→wav trip). The ACE-model parts move (LM reward 0.766 →
  0.630) because the CoT and plan are gone, **so ACE's renders are scored
  through `-ext` too**, under the same conditions: `out/mm3-n0-ace`
  (turbo from `out/a14e` flac, base b32 from `out/a15/steps` mp3).
- **The mp3 floor**: the same turbo song from its mp3 rather than its flac
  moves SongEval musicality +0.11 and the LM reward −0.045; Audiobox ≤
  0.01. Base's renders exist only as mp3, so differences that small
  between base and anything else are not differences.
- **The probe** (10 s of n09, bf16, 16 threads, beside `ai.service`):
  173 s wall, ~22 GB RSS. The frame loop is ~84 s for 250 frames (~336 ms
  a frame on the CPU) and the DiT 89 s for 2 windows × 30 steps (1.48 s a
  guided step at ~689 latents). About 19 s of CPU a second of song.
- **The subset**: n02 (piano, instrumental, 30 s), n05 (folk, 45 s), n09
  (jazz, 30 s), n11 (reggaeton, Spanish, 60 s), n13 (metal,
  instrumental, 45 s), n18 (gospel, 60 s); 270 s of audio. Captions as
  the set gives them (short, not the card's structured form), so this is
  MiniMax at a disadvantage it would not have behind a caption rewrite.

### N0 — the first pass: the set's own captions (2026-10-23)

Six renders, bf16 on the CPU, the set's one-line captions and lyrics as
given, seed 11; ACE's renders of the same requests scored the same way
(`-ext`). Means over the six:

| | seconds | PQ | CE | CU | coherence | musicality | naturalness | reward | DiT align | loop share |
|---|---|---|---|---|---|---|---|---|---|---|
| ACE turbo | 45.0 | 8.17 | 7.56 | 7.95 | 3.79 | 3.51 | 3.39 | 0.603 | 0.218 | 0.063 |
| ACE base (b32, mp3) | 45.0 | 8.02 | 7.34 | 7.87 | 3.48 | 3.27 | 3.18 | 0.584 | 0.219 | 0.035 |
| MiniMax, set caption | 32.2 | 7.30 | 6.34 | 7.09 | 3.23 | 3.05 | 3.00 | 0.605 | 0.192 | 0.059 |
| *the card's own render* | *240 of 285* | *8.15* | *7.62* | *7.79* | *4.18* | *4.00* | *3.88* | — | — | — |

- **MiniMax lost every production and musicality score to turbo on all
  six songs** (PQ −0.87, CE −1.22, coherence −0.56, musicality −0.46 a
  song); adherence tied (reward +0.002, 3/6). n02 (solo piano, PQ 6.09)
  and n18 (gospel, PQ 6.07, CE 4.40, −11.1 LUFS) were the worst.
- **The control says it is the prompting, not the model**: the card's
  `assets/minimax_ttm.wav` (a full structured caption and full lyrics,
  SGLang on CUDA; scored on its first 240 s, the set's ceiling) is level
  with turbo on Audiobox and above it on SongEval. Not clipping either:
  n18 has 0.004% of samples at full scale, the rest none.
- **The model chooses its own length (N-o4).** Duration is not in the
  prompt (`encoders.py`: caption and lyrics only); `audio_duration` caps
  the frames. Five of six ended early on end-of-audio: n02 22.5 of 30 s,
  n05 31.1/45, n13 30.7/45, n11 18.6/60; n09 hit its 30 s cap and n18 its
  60. The set's lyrics are 1–43 words, which ACE stretches over the
  length and MiniMax sings once.
- Hence the second pass: the same six with structured captions written
  per the card's `music-caption-rewriter` skill (`out/mm3-n0/captions.json`:
  ~260–330 words, ACE's own CoT BPM and key except n18 and n11, a section
  timeline sized to the requested length), the role ACE's CoT rewrite
  plays for ACE. `render_mm3.py --captions` tags them `-sc`.

### N0 — the second pass: structured captions (2026-10-23)

The same six from `out/mm3-n0/captions.json` (`-sc`). Lengths: n09 30/30,
n02 30/30, n05 45/45, n13 45/45, n11 60.1/60, n18 40.4/60 — **a sized
arrangement timeline sets the length** (N-o4), closely but not exactly.
Means over the six, beside the first pass:

| | seconds | PQ | CE | CU | coherence | musicality | naturalness | reward | DiT align | loop share |
|---|---|---|---|---|---|---|---|---|---|---|
| ACE turbo | 45.0 | 8.17 | 7.56 | 7.95 | 3.79 | 3.51 | 3.39 | 0.603 | 0.218 | 0.063 |
| MiniMax, set caption | 32.2 | 7.30 | 6.34 | 7.09 | 3.23 | 3.05 | 3.00 | 0.605 | 0.192 | 0.059 |
| **MiniMax, structured** | 41.8 | 7.72 | 7.37 | 7.71 | **4.03** | **3.76** | **3.67** | **0.640** | 0.159 | **0.024** |

- Against turbo, per song: SongEval coherence +0.24, musicality +0.25,
  naturalness +0.29, each on 5 of 6; Audiobox PQ −0.46 (1/6), CE −0.19
  (1/6), CU −0.24 (0/6); the ACE-LM reward +0.04 (3/6); the loop share
  lower on 5/6. ACE's DiT alignment −0.06 (2/6), but that is ACE's DiT
  judging MiniMax.
- Against MiniMax on the set caption: better on every production and
  song score (CE +1.03 on 5/6, coherence +0.80 on 6/6). **A structured
  caption is part of the model**, as the CoT rewrite is part of ACE: N8
  needs a rewrite step (the caller's, as video's M12, or ours).
- So the scores split: MiniMax writes the better song (SongEval, less
  looping), turbo produces the cleaner audio (Audiobox). A14e found the
  scores track the user's ratings only partly; the panel decides.
- **The panel**: https://claude.ai/artifact/3XmVEhc8XE4PfSSiAjS47G
  (private; db collections `ratings`, doc id `<song>-<letter>`, field
  `score` 1–5, and `picks`, doc id `<song>`, fields `best` and `note`).
  24 clips: each song's turbo, base, MiniMax and MiniMax-structured,
  lettered A–D in a seeded shuffle, all loudness-matched to −16 LUFS /
  −1 dBTP at 192 kb/s mp3. The clip-to-model map is
  `out/mm3-n0/panel.json` (kept out of the page); the builder is
  `out/mm3-n0/panel_build.py`.

### N0 — the panel (2026-10-23)

The user rated all 24 clips blind (1–5); one keep pick (n13: MiniMax
structured). Unblinded with `out/mm3-n0/panel.json`:

| song | ACE turbo | ACE base | MiniMax, set caption | MiniMax, structured |
|---|---|---|---|---|
| n09 smooth jazz | 2 | **5** | 3 | 4 |
| n02 solo piano | **5** | 3 | 1 | 2 |
| n05 acoustic folk | **5** | 4 | 1 | 2 |
| n13 speed metal | 2 | 2 | 1 | **5** |
| n11 reggaeton | **5** | 3 | 1 | 2 |
| n18 gospel | 2 | 3 | 1 | **5** |
| **mean** | **3.50** | 3.33 | 1.33 | 3.33 |

- **A tie, not a clear win**, so by N-d7 the port does not pay: MiniMax
  structured tops 2 of 6 (metal, gospel), turbo 3, base 1, and the means
  are within one rating. MiniMax on the set's one-liners is last on every
  song (five 1s), which confirms the first pass.
- The split is by song, not by model quality across the board: MiniMax
  takes the dense full-band genres, ACE the sparse ones (piano, folk) and
  reggaeton.
- The scores against the ratings (Spearman, 24 clips; mean within a
  song): SongEval coherence 0.35 (0.59), musicality 0.34 (0.56), Audiobox
  CE 0.50 (0.39), PQ 0.36 (0.36). Within a song SongEval ranks best, as in
  A14e, but it rated MiniMax structured over turbo on n02 and n05 where
  the user put it 3 points under: its 5-of-6 SongEval lead did not hold.
- **The user, after unblinding (2026-10-23):** the metal and gospel
  ratings stand, and ACE turbo cannot render a guitar (acoustic, clean
  electric or distorted), which rules out whole classes of music. So the
  mean understates the gap: MiniMax covers guitar-led genres ACE cannot.
  One tension in the set: n05 (a fingerpicked acoustic folk song) has
  turbo at 5 and MiniMax structured at 2.
- On the means alone, a tie at ~10x the render time would not justify
  N1–N10. Unless the user decides otherwise, the vertical stops here at
  the oracle (`render_mm3.py`, `acecal -ext`) and the weights.

### N0b — a guitar panel (2026-10-23)

After N0 the user reported that ACE turbo cannot render a guitar
(acoustic, clean electric or distorted), and chose a guitar-led panel
before any port work. `out/mm3-n0b/set.json`: six guitar-led songs (g1
fingerpicked steel-string, g2 strummed acoustic heartland rock, g3 clean
electric neo-soul, g4 overdriven Texas blues, g5 distorted grunge, g6
flamenco nylon), seed 11. ACE turbo and base render through ai.service's
`/v1/music` with thinking on (`out/mm3-n0b/ace_render.py`), MiniMax on the
CPU with structured captions (`out/mm3-n0b/captions.json`, BPM and key
from turbo's CoT plan); `render_mm3.py --set` and `acecal -set` read the
new set. MiniMax hit every requested length (12.6 min CPU per 40 s).

- ACE's CoT rewrote g1's "fingerpicked steel-string acoustic guitar" as a
  **ukulele**: the planner drops the guitar before the DiT sees it.
- Scores (`acecal -set out/mm3-n0b/set.json -ext out/mm3-n0b`) are mixed:
  MiniMax leads SongEval musicality on g1 (4.41 vs 3.65 turbo) and g3,
  turbo on g2, g4, g5 and g6; Audiobox PQ is turbo's on 4 of 6.
- **Panel**: https://claude.ai/artifact/WoVhq5ASqXwD2aFdH7HHAd (db
  `ratings`/`picks` as N0's), three takes a song (turbo, base, MiniMax
  structured), map `out/mm3-n0b/panel.json`, builder
  `out/mm3-n0b/panel_build.py`.

- **Rated 2026-10-23** (user: "most of those were quite bad"), unblinded:

  | song | ACE turbo | ACE base | MiniMax structured |
  |---|---|---|---|
  | g1 fingerpicked | 2 | 1 | **3** |
  | g2 strummed | 2 | 1 | **4** |
  | g3 clean electric | 1 | 1 | 1 |
  | g4 blues | **4** | 1 | 2 |
  | g5 grunge | 2 | 1 | 2 |
  | g6 flamenco | 1 | 1 | **3** |
  | **mean** | 2.00 | 1.00 | **2.50** |

  MiniMax is best or tied on 5 of 6, but the whole panel is poor.
- **Are we prompting MiniMax properly? Not fully.** Against the skill's
  1,000 templates (`skills/music-caption-rewriter/templates`, cloned
  2026-10-23) and the card's own example
  (`scripts/end_to_end/minimax_ttm_test.py`):
  1. **Length**: templates run 416–723 words (median ~570), and the
     card's caption is 561. Ours ran 212–325, roughly one sentence for
     each field where the templates write a paragraph.
  2. **Our trailing section line** ("Intro: … Verse: …") appears in none
     of the 1,000 templates. Their timeline lives in the prose of each
     field.
  3. **Instrumentals**: every template describes a singer (998 of 1,000
     carry an exact BPM). n02, n13, g3 and g6 were instrumental, which the
     captions MiniMax was trained on hardly cover.
  4. **Song length**: the card's example is a 285 s song with full
     lyrics; templates describe whole songs. We asked for 30–45 s with
     6–8 lyric lines.
  5. **The skill's workflow** (genre router, then family index, then up
     to 3 templates as Foundation/Modifier/Arrangement) was not followed:
     the captions were written from SKILL.md alone.
  6. **Pipeline parity is unverified**: the card's wav came from
     SGLang-Omni, and the diffusers PR (`dafe3733`) has never reproduced
     it here. The lyrics are normalised as upstream does: tags are
     lowercased and `[start]` is prepended.
- The diffusers load warns "incorrect regex pattern … will lead to
  incorrect tokenization" (transformers' Mistral check). It is a false
  positive: the tokenizer is Qwen2's standard pre-tokenizer, and
  `fix_mistral_regex=True` changes no id on any N0c caption or lyric
  (checked 2026-10-23).
- **Check (done 2026-10-23): the pipeline reproduces the card.** The
  card's exact caption and lyrics through `render_mm3.py`
  (`out/mm3-n0b/card-set.json`, seed 11) gave a 300.2 s song in 7,348 s
  (1.52 s/step, 2,220 AR steps). Scored beside the card's own
  `assets/minimax_ttm.wav` (copied in as `card-11-ref.wav`):

  | | PQ | CE | musicality | reward |
  |---|---|---|---|---|
  | ours (diffusers, CPU bf16) | 7.90 | 7.44 | 3.78 | 0.602 |
  | card (SGLang-Omni) | 8.09 | 7.59 | 3.98 | 0.618 |

  The gap is about one seed's spread (N0's earlier card control scored
  PQ 8.15 / CE 7.62), so gap 6 is closed. What hurt N0b was the prompting
  (gaps 1–5), not the pipeline.

### N0c — full-length guitar songs, prompted the card's way (2026-10-23)

- The user: go ahead with a full-length guitar set after the check. The
  set is `out/mm3-n0c/set.json`, all 180 s, English, sung, seed 11, with
  full [Intro]/[Verse]/[Pre-Chorus]/[Chorus]/[Bridge]/[Solo]/[Outro]
  lyrics:
  - h1: indie folk ballad (fingerpicked steel-string, female);
  - h2: heartland rock (strummed acoustic, raspy male, gang vocals);
  - h3: Texas blues rock (overdriven guitar and solos, male);
  - h4: 90s grunge (clean to distorted, husky female).
- ACE (turbo and base, served, `thinking:true`) gets the set's short
  natural caption. MiniMax gets `out/mm3-n0c/captions.json`:
  - 472–550-word captions written through the skill's workflow. Each has
    an exact bpm and key: h1 76 D major, h2 112 G major, h3 96 A minor,
    h4 100 E minor.
  - No trailing section line. The templates used are listed in its `_note`.
- ACE's CoT re-picked the tempo on every render, and on five of eight it
  chose 41–43 bpm (h2 turbo 41, h3 42/43, h4 42/42), half-time or less of
  a rock tempo. h1 stayed a fingerpicked acoustic guitar this time, not
  N0b's ukulele.
- ACE took 42–84 s per song served. MiniMax runs on the CPU via
  `scratchpad/mm3_n0c.sh`, about an hour per song.
- Scoring (`acecal -set out/mm3-n0c/set.json -ext out/mm3-n0c`) and a
  three-way blind panel follow the renders.

### N1 — the global LM on the device (2026-10-23)

- **Oracle**: `reference/dump_mm3.py` runs the reference's own functions
  (`encoders.py` at `dafe3733`) in fp32 on N0c's h3 (a 944-token prompt a
  row) for 48 frames from seed 11, then bf16 teacher-forced on the same
  codes. It writes `reference/out/mm3lm/`: the head span's logits, the
  hidden, the 8 codes, the 7 depth hiddens and logits, and the feedback for
  every step. Upstream bf16's drift from fp32, after CFG 1.5 and before
  the top-50 cut: KL mean 2.36e-3, worst 1.10e-2; guided argmax 47/49.
  (The KL is over the CFG logits because each side's top-50 gives it its
  own support, which would make the guided KL infinite.)
- **`mm3/lm.go`** is ace/lm's dense Qwen3 at 8B. The heads match (32/8 ×
  128), so `ace_lm_prep` and `ace_lm_attn` run as built. Changes:
  - hidden 4096 and FFN 12288;
  - the untied `lm_head` is staged only over [151670, 168059), 16,400
    rows padded, 134 MB;
  - a row's input is a token id or a caller's vector (the feedback);
  - Pass returns the final-normed hidden (fp16) with the logits;
  - the fp16 arena is host-cached so the host can read the hidden back;
  - the rope base is read from `rope_parameters`, because transformers 5
    no longer writes a top-level `rope_theta`, so `qwen.Config` read 0;
  - split-K tables moved to rungs that divide K's 256 or 768 tiles (40
    does not divide 256).
  
  The default MaxLen is 10,240, the model's limit: 3.0 GB of cache for
  the pair.
- **Gate** (`TestLMFrames`, teacher-forced, every step):

  | bank | CFG KL mean | worst | logits rel | hidden rel | top-50 identical | 2-row step |
  |---|---|---|---|---|---|---|
  | fp16 | 1.62e-6 | 7.84e-6 | 9.1e-4 | 1.1e-3 | 49/49 | 73.7 ms |
  | int8 | 6.75e-4 | 7.41e-3 | 1.7e-2 | 2.2e-2 | 25/49 | 63.1 ms |
  | (bf16, upstream) | 2.36e-3 | 1.10e-2 | | | | |

  fp16 passes ace/lm's A7 bound (1e-3) by 100x. int8 passes A10's rule
  (mean no worse than bf16's, worst within 2x of bf16's worst) and is 3.5x
  under bf16 on the mean. **int8 is the default.**
- **Speed**: 63.1 ms a step is 1.58 s of LM a second of audio; real time
  is 40 ms. The int8 step reads 7.4 GB at 117 GB/s, against fp16's
  189 GB/s, so the slab table is the first suspect (ace/lm's A10 ladder
  was at hidden 2560). That ladder is N3's.

### N2 — the local LM (2026-10-23)

- The local LM has 4 layers, 16 heads × 256, full MHA, FFN 6144, a learned
  16-row position table added after the input projection, no rope and no
  q/k norm. Upstream re-runs the whole depth sequence for each of the 7
  codes. **`mm3/depth.go` keeps a KV cache instead**, so a frame is 7 passes:
  positions 0 and 1 as one 4-row pass, then one position a pass, both CFG
  rows each time.
  - Without rope or q/k norm there is no prep step: the k and v GEMVs write
    straight into fp32 caches `[pos][slot][4096]` (rows in (pos, slot)
    order are contiguous).
  - A new kernel, `mm3_depth_attn.comp`, attends over ≤ 16 keys with one
    workgroup per (row, head) and a lane per dim.
  - The 7 heads stay fp16 (58.7 MB).
  - The feedback is built on the host from the bf16 tables.
- **Gate** (`TestDepthFrames`, all 49 frames teacher-forced from the
  oracle's global hidden and codes; 343 heads):

  | bank | CFG KL mean | worst | hidden rel | top-50 identical | a frame |
  |---|---|---|---|---|---|
  | fp16 | 2.2e-7 | 7.5e-7 | 3.4e-4 | 338/343 | 44.6 ms |
  | int8 | 2.1e-5 | 8.1e-5 | 6.0e-3 | 304/343 | 42.0 ms |

  The feedback matches the oracle's to 3.6e-9. The cache is exact in all
  but rounding (upstream's recompute and our cache are the same causal
  sums). There is no bf16 depth reference yet, but int8's worst is 100x
  under the global LM's bf16 drift, so int8 is the default.
- **Speed**: fp16 runs near bandwidth (8.2 GB a frame at 184 GB/s), but
  int8, at half the bytes, is only 6% faster. The frame is ~400 small
  dispatches in 7 submits, so it is latency-bound. **A frame is
  63 + 42 = 105 ms, 2.6 s a second of audio**, against the plan's 1.55 s.
  The levers (fused depth passes, one submit a frame, the slab ladder,
  `decode-is-latency-not-bandwidth`) are N3's timing and N9's.

### N3 — the frame loop (2026-10-23)

- **`mm3/frames.go`**: `Generate` prefills the CFG pair, then runs a step
  a frame:
  1. the global LM pass (2 rows, head span, hidden);
  2. upstream's global draw: CFG 1.5, the conditional row's top 50, the
     mask, then top-50 sampling;
  3. the local LM's 7 passes, each code drawn by CFG 1.5 then top-50;
  4. the feedback.
  
  Step 0's codes are fed back but are not a frame, and AudioEnd ends the
  song. The draws use our RNG (PCG), not torch's.
- **The condition's mix is folded per frame.** The encoder's softmax
  weights times `layer_scale` come to `[0.0621, 0.00089, 0.00087, 0.00088,
  0.00096, 0.00093, 0.00094, 0.00097]`, global LM first. The mix is linear,
  so each frame keeps one 4096 vector (16 KB, 147 MB at 9,000 frames).
  **The global LM's hidden carries ~93% of the weight**: the local LM's
  hiddens barely enter the condition, though its codes still drive the
  feedback.
- **Gate**: `TestSamplerSupport` checks that on the oracle's fp32 logits
  every draw it made (49 semantic, 343 residual) is inside our sampler's
  support. `TestGenerate` free-runs 250 frames from the oracle's prompt:
  finite conditions throughout, no early end.
- **The split-K ladder** (`TestSlabLadder`, one projection at a time, then
  whole tables): **the int8 GEMV's k8 and k16 rungs are slow at 2–4 rows.**

  | int8 table | global LM step | local LM frame |
  |---|---|---|
  | N1/N2's (mostly k8) | 63.1 ms | 42.0 ms |
  | all k16 | 70.5 | 40.5 |
  | all k1 | — | 27.7 |
  | all k32 | **45.3** | **25.4** |

  k32 everywhere is the table now. Single-projection moves read within
  noise (±3 ms), but whole tables reproduce. **This carries to ace/lm**:
  its int8 table (A10) is k8 on q, gate, up and down, and its ladder was
  read one projection at a time. A two-row check there is worth an
  afternoon.
- **Timing** (`TestGenerate`, int8, 250 frames, wall): 45.8 ms LM +
  27.2 ms local LM + 2.4 ms host sampling = **75 ms a frame, 1.88 s of
  compute a second of audio**, so a 3-minute song spends ~5.6 min in the
  frame loop. The LM reads 7.4 GB at 163 GB/s. The local LM's 7 submits
  and ~400 dispatches a frame are now the soft target: one submit a frame
  needs the draws on the device. That and the 4-bit bank are N9's.

### N4, N5 — the DiT and the window sampler (2026-10-23)

- **Oracle**: `reference/dump_mm3_dit.py` writes `reference/out/mm3dit/`.
  1. 300 frames of h3 from seed 11 through the bf16 LMs, kept as
     `frame_hiddens [300, 8, 4096]`.
  2. Upstream's denoise blocks run by hand in fp32 over both windows
     (starts 0 and 100). Per window: condition, noise, every step's
     latents and guided velocity, step 0's two predictions, final latents,
     the vocoder's wave.
  3. The cropped 12.00 s song.
  
  Over the whole run the largest |input| into any block's `ff_out` is 232
  (fp16 is fine, no FFN scale). The schedule is t = i/30, with t = 0 as
  noise.
- **The DiT is simpler than planned**: no adaLN anywhere. The timestep is
  only the prepended token (Fourier features, then a SiLU MLP). Blocks are
  LayerNorm (weight + bias, eps 1e-5) → q, k, v (no bias) → partial rope →
  attention → o → residual, then LayerNorm → ff_in (bias; `chunk(2)` =
  [value | gate], the gate silu'd) → ff_out (bias) → residual. There is no
  final norm before `proj_out`.
- **`mm3/dit.go`** runs on the H3 video VAE's ViT kernels (h3/vae, M5):
  - the LDS wave32 GEMM with the bias in a ones column, and its fused
    SwiGLU epilogue (`qwen.InterleaveGLU`, the silu'd half second, as
    here);
  - the head-64 v pack and the transposed head-64 attention (QT2 KTIL4,
    taken when `dit.TransposedAttentionOK`);
  - `parakeet_layernorm`'s OUT_F16 build for the norms.
  
  New kernels:
  - `h3_qk_pack.comp` gets a `QK_NORM=0` switch (`mm3_qk_pack_hd64`, rope
    32 of 64). The VAE's build compiles byte-identical with the switch in.
  - `mm3_narrow.comp`, the residual into `proj_out`'s A operand.
  
  Folds, in float64 at load:
  - proj_in · (preprocess_conv + I), with the zero input channels' columns
    dropped, is one [2048, 2176] weight;
  - (postprocess_conv + I) · proj_out is one [128, 2048] weight, on the
    reg32x128 GEMM (N = 128).
  
  A forward is the CFG pair as two sequences at a 16-row stride (2 × 704
  rows at L = 689). The condition encoder's k3 conv is one GEMM over
  im2col rows `[x(f−1) | x(f) | x(f+1) | 1]`, then the nearest resample on
  the host. 4.9 GB of fp16 in 2 banks, staged in 5.1 s.
- **Gates**: `TestDiTForward` takes window 0's step 0 at 689 latents: rel
  cond 3.0e-4, uncond 3.9e-4, guided 3.3e-4. `TestWindows` runs both
  windows from the oracle's frame hiddens and noise, window 1 on our own
  carry. Condition rel 2.0e-4. Final latents rel 4.6e-4 (window 0) and
  7.4e-4 (window 1), so 30 steps don't amplify the error.
- **Speed**: 216 ms a CFG forward (6.6 TFLOP, ~31 TFLOP/s), 6.6 s a
  window, **1.65 s of compute a second of audio** at the 4 s hop. With
  N3's 1.88 that is ~3.5 s a second: about 10.5 min for a 3-minute song,
  serial. N9 has the levers: fewer DiT steps, the LM and DiT overlapped
  (second queue), int8 DiT.

### N6 — the vocoder (2026-10-24)

- **`mm3/vocoder.go`** is ACE's Oobleck decoder host (ace/vae, A5) with
  four changes:
  1. **Widths are 3·2^k** (1536 → 96), and A_CONV and `ace_vae.comp`
     address by powers of two. Every width is padded to the next power of
     two with zero weights, zero biases and α = 0 Snakes, so pad channels
     stay 0. That is up to 1.8x the FLOPs of a ~110 GFLOP-a-second decode.
  2. **Snake has α only**: `ace_vae.comp`'s x + sin²(αx)·γ with
     γ = 1/(α + 1e-9).
  3. **The extra k1 `dec_in_proj`** (no weight norm) has its bias in an
     α = 0 Snake pass.
  4. **`conv_out` is one channel** (on the 32-column rung) with bias and
     tanh on the host; each stereo channel is a decode of its 64 latent
     channels.
  
  As upstream, a window is decoded whole and the crops are the caller's.
- **Gate** (`TestVocoder`, both oracle windows' final latents): SNR 63.2 /
  60.7 dB (window 0, L/R), 63.8 / 61.4 dB (window 1). The cropped,
  concatenated 12.00 s song is 63.8 / 61.3 dB against upstream's.
  220 ms an 8 s window for both channels (27 ms a second of audio), staged
  in 0.3 s.

### N7 — end to end (2026-10-24)

- **`mm3/prompt.go`** is upstream's `_clean_caption` and
  `_normalize_lyrics`, rule for rule, through the repo's Qwen2 BPE
  tokenizer (`zimage/tokenizer`, which reads `tokenizer.json` and the
  config's specials). Three spots where Go and Python differ are handled
  by hand:
  - Python's `\s` is Unicode whitespace;
  - `str.splitlines` has more line boundaries than `\n`;
  - the italics pattern needs look-around, which RE2 lacks, so it is a
    scan.
  
  `reference/dump_mm3_prompt.py` dumps 44 cases (the A14e, N0b and N0c
  sets, the card's, and edge cases for every rule). `TestPrompt` matches
  all 44: the cleaned caption, the normalised lyrics and every token id.
- **`mm3/pipeline.go`** is the whole model, resident at ~18 GB with the
  int8 LMs. It runs prompt → `Generate` → windows (`Condition`, `Noise`,
  `Denoise` with the carry) → `Vocoder.Decode` → decoders.py's crops (86
  latents left, 258 right) → clamp. One PCG seeded per request drives both
  the draws and the noise. `cmd/mm3` writes a 16-bit 44.1 kHz wav and a
  json of the request and timings.
- **First song** (h3, capped at 30 s, seed 11): 750 frames, 7 windows.
  It took 107 s: 58.4 s frames, 47.1 s DiT, 1.5 s vocoder, **3.57 s a
  second of audio** (the CPU oracle was ~19). Scores: PQ 8.18, CE 7.60,
  musicality 3.99, against the card's own SGLang render at 8.09 / 7.59 /
  3.98.
- N0c's four full-length guitar songs (h1–h4, 180 s, the skill-style
  captions) are rendering on the device into `out/mm3-n0c/*-gpu-sc.wav`,
  for the panel N0c could not finish on the CPU.

### N8 — serving (2026-10-24)

- **`backend.MusicMM3`** is an `api.MusicBackend` and its own `Swappable`
  ("music-mm3", N-d2), resident at ~19 GB while it holds the slot. Planning
  tokenises the prompt on the host. The device is held a step at a time:
  `mm3.Pipeline.Between` runs after every frame, DiT step and window, and
  yields to waiting work and checks for cancellation.
- **`backend.MusicRouter`** (N-d4) dispatches on `model`. `IsMusicMM3` is
  the id `MiniMax-Music3`, or a name with `minimax` or `music3` as a word.
  Everything else goes to ACE's `resolveModel` (N-d1), and with only one
  backend, that one takes every request. One `MusicJobs` queue serves both.
  **The flag is `serve -music-mm3`**, not N-d4's `-music-models a,b,c`:
  that name was already ACE's checkpoint root. MiniMax stages before ACE,
  so ACE is the one resident after start-up.
- **The request mapping**:
  - **N-d3**: bpm and keyscale go on a `Basic Attributes:` line in the
    skill templates' phrasing (`bpm is 92. key is E, and scale is minor.`).
    The templates never state a time signature or language there, so those
    are short sentences after it (`The time signature is 3/4. Sung in
    English.`). A caption with its own `Basic Attributes:` line takes the
    request's metas there, in place of the bpm and key it states; any other
    caption gets a `Global Metadata` block prepended.
  - Empty lyrics become `[Instrumental]`: upstream refuses an empty string,
    and ACE's convention is the tag.
  - `duration` caps the song (default 120 s, at most 360 s, and no more
    than the prompt leaves of the cache). The `planned` progress event
    reports the length the LM actually made, with a re-estimate.
  - **N-d5**: thinking, sample mode, captioning, the audio-in tasks,
    `src_audio`/`reference_audio`, the DiT schedule fields, the LM's
    sampling and `lm_seed` are 400s that name ACE.
  - `auto_score` is a 400 until N-d6's shared scorers exist.
- **The estimate** (`mm3Estimate`, from N3–N7's rates): prefill 0.75 ms a
  token, 75 ms a frame, 6.7 s a window, 0.22 s of vocoder a window, plus
  the encode. Audio is written through ACE's ffmpeg writer, which now takes
  a rate (`pipeline.WriteRate`, 44.1 kHz).
- **The cache grows to 13,312 positions** (3.9 GB, under one binding's
  4 GB): a 6-minute song after a prompt of up to 4,311 tokens. Upstream
  allows 5,000 + 9,000, past the model's trained 10,240, so it extrapolates
  too.
- **Tests**: `TestIsMusicMM3`, `TestMusicMM3Caption`,
  `TestMusicMM3Refusals` (before any device work) and `TestMusicRouter`
  pass. **Served**: a `serve -music-mm3` instance on a spare port, POST
  `/v1/music` (model `MiniMax-Music3`, h1's caption and lyrics, 20 s,
  bpm 76, keyscale "D major", seed 11). It completed in 75 s against an
  estimate of 66 s and wrote a 20.0 s 44.1 kHz stereo mp3. The request's
  metas replaced the ones on the caption's own `Basic Attributes:` line. An
  explicit `thinking: true` got the 400 naming ACE.
- **Found by the served test**: the API defaults `thinking` and
  `auto_score` to on, so N-d5's refusals caught every request.
  `api.MusicRequest` now carries `ThinkingSet`/`ScoreSet`, and MiniMax
  refuses only a client's explicit true. A job still echoes
  `"thinking": true` (the default); that is cosmetic.
- `ai.service` now carries `-music-mm3` (the user, 2026-10-24); not yet
  deployed.

### N0c — the full-length panel (2026-10-24)

- MiniMax rendered on the device (`cmd/mm3`, seed 11, the skill-style
  captions). Each song took ~11.2 min: 6.0 min of frames, 5.0 of DiT,
  10 s of vocoder. h2 and h4 ended by the LM at 177.7 s and 179.0 s. h1
  and h3 ran to the 180.2 s cap and were trimmed to 180 s, because
  acecal's ACE alignment scorer is sized by the set's duration and
  overflowed by 5 latents (originals in `out/mm3-n0c/full/`). ACE turbo
  and base were served earlier (`ace_n0c.py`).

  | song | turbo PQ / CE / mus | base | MiniMax |
  |---|---|---|---|
  | h1 folk | 8.42 / 7.86 / 4.18 | 8.02 / 7.61 / 4.08 | **8.48 / 8.01 / 4.48** |
  | h2 heartland | 8.02 / 7.39 / 4.06 | **8.33** / 7.51 / 4.16 | 8.24 / **7.51 / 4.44** |
  | h3 blues rock | 8.23 / 7.06 / **4.26** | 6.91 / 7.22 / 2.80 | **8.30 / 7.59** / 4.13 |
  | h4 grunge | **8.05** / 7.02 / 3.92 | 7.63 / 7.03 / **4.29** | 7.83 / **7.44** / 3.95 |

  MiniMax has the best CE on all four and the best musicality on two. The
  blind panel (three takes a song, `scratchpad/panel-h`, hidden map
  `out/mm3-n0c/panel.json`) is https://claude.ai/artifact/7EBvFbRo3zREVKkVarVgzV.
- **Rated 2026-10-24** (12 of 12 clips; no keep picks saved), unblinded:

  | song | ACE turbo | ACE base | MiniMax (skill captions) |
  |---|---|---|---|
  | h1 folk | 4 | 2 | **5** |
  | h2 heartland | 1 | 3 | **5** |
  | h3 blues rock | **5** | 1 | 3 |
  | h4 grunge | 4 | 1 | **5** |
  | **mean** | 3.50 | 1.75 | **4.50** |

  MiniMax is top on 3 of 4 and rated "would keep" on all three; turbo
  wins the blues again (as on N0b's g4). Against N0b (MiniMax 2.50,
  turbo 2.00, base 1.00) every arm rose, but the songs, lengths and
  captions all changed, so the rise is not one cause. Scorers: MiniMax's
  best-CE-on-all-four overcalls h3; SongEval musicality picked the user's
  top take on 3 of 4 (missed h4, where it favoured base, rated 1).

## Open questions

- **N-o1** (closed 2026-10-23): **44.1 kHz**. The diffusers pipeline
  writes 44,100 Hz stereo (`pipe.sampling_rate`, N0's probe). Before
  that: `assets/minimax_ttm.wav` (the card's SGLang render) is
  32 kHz s16 stereo, 284.85 s. The latent rate is 44100/512 = 86.13 Hz
  and the vocoder's hop is 512, so the decoder is natively 44.1 kHz and
  32 kHz looks like SGLang's output resample. To confirm by running the
  diffusers vocoder.
- **N-o2** (closed 2026-10-23): the community license's clause 4 requires
  safeguards when generation is offered to third parties. The user:
  `ai.service` is reachable only by them, so clause 4 does not apply;
  revisit if that changes. (Clause 3, the "MiniMax-Music3" UI credit, is
  for commercial products.)
- **N-o3**: languages are not stated anywhere.
- **N-o4**: the model picks the song's length; `duration` only caps it.
  Serving has to say so (a job's `duration` is the render's, not the
  request's), and N0's second pass checks whether a sized arrangement
  timeline in the caption moves it.

## Handoff

Download: `.venv/bin/hf download MiniMaxAI/MiniMax-Music3 --exclude
"qwen_7B/*" --exclude "*.pth" --local-dir models/MiniMax-Music3`. **One
`--exclude` per pattern**: `--exclude a b c` takes `b c` as file names,
warns "Ignoring `--exclude` since filenames have been explicitly set" and
fetches only those (the same gotcha as A0's `--include`). The oracle venv
is `.venv-mm3`: `uv venv --python 3.12`, torch 2.14.1+cpu from the
PyTorch CPU index, diffusers at `dafe3733` (0.40.0.dev0), transformers
5.20.2. The reference's sources were read from the PR
commit; its constants are in `encoders.py` (`_AR_CFG_SCALE`, token ids),
`denoise.py` (`_OVERLAP_LATENT_LENGTH`, guidance 1.7, 30 steps),
`before_denoise.py` (`_CHUNK_FRAMES`, `_CHUNK_HOP`) and `decoders.py`
(crops).
