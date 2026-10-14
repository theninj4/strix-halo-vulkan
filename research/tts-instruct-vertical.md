# Instructed TTS — Breeze-TTS-2 beside kokoro (B-stages)

> **Live (2026-10-14).** B0–B5 are done and every part runs on the device
> to its gate; `cmd/breeze` speaks end to end (the radio-DJ case
> transcribes exactly), at 0.78x real time on fp16 banks before any of B7.
> Next: B6 (serving), B7 (int8, one submit a frame), B8 (cloning; the user
> asked for it, 2026-10-14). Stage letters are **B0–B8**; code in
> `breeze/`, `cmd/breeze`, `shaders/breeze*.{go,comp}`.

**Target**: `/v1/audio/speech` takes OpenAI's own `instructions` field ("An
enthusiastic radio DJ") and answers it with
[`BreezeBlue/Breeze-TTS-2`](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
end to end in Go on Vulkan, beside kokoro rather than instead of it. Kokoro
stays the default for a request without `instructions` and the only voice
on Wyoming.

## What the model is

It is a CSM-shaped stack: a text encoder, an autoregressive backbone that
emits one 16-codebook audio frame per step, a small depth decoder that fills
in codebooks 1–15 within a frame, and Qwen3-TTS's 12.5 Hz codec. It outputs
24 kHz mono, in English and Chinese.

| part | shape | params | used |
|---|---|---|---|
| text encoder | T5Gemma2 (Gemma-3 style): 26 layers, 1152 wide, 4 q heads / 1 kv, head 256, sliding 512 (±256) with full attention at layers 5/11/17/23, **bidirectional**, sandwich norms, `(1+w)` RMSNorm, GELU-tanh | 1000 M (302 M of it embedding) | once a request |
| `text_encoder_proj` | Linear 1152 → 2048 | 2.4 M | once |
| backbone | **dense Qwen3-1.7B shape**: 28 layers, 2048 wide, 16 q / 8 kv, head 128, qk-norm, rope θ 1e6 (the top-level llama3 rope is not used by the qwen3 adapter), MLP 6144 | 1409 M | 1 step a frame |
| `lm_head` | 2048 → 2052, fp32; 2051 = EOS, 2048–2050 suppressed | 4.2 M | 1 a frame |
| audio embedding | `[16·2051, 2048]`, shared with the depth decoder; a frame's input is the **sum** of its 16 rows | 67 M | gather |
| depth decoder | Llama: 12 layers, 1024 wide, 8 q / 2 kv, head 128, MLP 8192, llama3 rope θ 5e5; input projector 2048 → 1024; 15 heads `[15,1024,2051]` in fp32 | 367 M | **15 steps a frame** |
| codec decoder | Qwen3TTSTokenizerV2: RVQ (1 semantic + 15 acoustic) → causal conv → 8-layer transformer (512 wide, sliding 72) → 2× (convT 2 + ConvNeXt) → snake + convT 8,5,4,3 → 1920 samples a frame; fp32 | 114 M | streaming, causal |
| codec encoder | Mimi-like | 55 M | voice cloning only |
| *dead weight* | `embed_text_tokens` (537 M) and `codec_model.*` (a Mimi, 79 M) are in the checkpoint but never called | 616 M | never |

The total is 3.65 B parameters and 7.7 GB on disk. The live weights are
**2.85 B, 5.7 GB in bf16**.

### The prompt

From `breeze_infer/templates.py`, the text is tokenised with Gemma-3's 262k
vocabulary, which adds 14 tokens:

- plain: `[S0]{text}`
- **voice design**: `[S0]<ins_bos>{instruction}<ins_eos>{text}`, with CFG
  whose negative prompt is the plain prompt
- clone: `[S0]{ref_text}`, reference codes, `<|audio_eos|>`, then
  `[S0]{text}`
- direction: the clone prompt with the instruction added; its CFG negative
  is the clone prompt

Each text segment runs through the encoder **separately**, with positions
from 0. Its projected states replace those positions in the backbone's
input, so the text arrives as a **prefix, not through cross-attention**.

Vocal events are plain text, `(laugh)` and `(sigh)`. **No G2P**, so none of
kokoro's T5 gap is inherited.

Every code path uses `[S0]`, and nothing documents what S1–S9 mean.

### Sampling

For the backbone and every depth step: temperature 0.9 and top-k 50. The
backbone alone also gets a repetition penalty of 1.1 over codebook 0's
history.

CFG is a batch of 2. The scale is applied to the backbone logits and to
every depth codebook's logits, and one sampled token is shared by both rows.
Upstream's CLI default is 1.0, which means no CFG; its README uses 4.

The length cap is 1500 frames, which is 120 s.

## What it will cost here — the one number to know before starting

Every frame is **16 dependent forwards**: one backbone step and 15 depth
steps, each of which needs the previous step's *sampled* token. That is 200
dependent forwards and about 2600 layer evaluations per second of audio.

These are the prices from what the repo has already measured. They are
**estimates, not measurements**:

- **backbone**: ace/lm's int8 Qwen3-4B step is 22 ms for about 3.6 GB of
  layers, which is 165 GB/s effective. 1.41 GB int8 at that rate is **about
  8–9 ms a frame**.
- **depth**: 0.34 GB int8 a step, about 2 ms, which is close to the
  per-layer latency floor ([[decode-is-latency-not-bandwidth]]). 15 steps
  make **about 30 ms a frame**.
- **codec, encoder, prefill**: small next to the depth decoder.

That is **about 40–45 ms per 80 ms of audio, roughly 2x real time** for one
stream. Kokoro runs at 120x. CFG's second row costs almost nothing at decode,
because ace/lm runs 1 or 2 rows at the same speed.

**The depth decoder is where the time goes**, and it is latency-bound. The
plan therefore turns on keeping a whole frame inside one submit:
**sampling on the device** (a new shader: penalty, mask, temperature,
top-k 50, multinomial from a host-seeded uniform) and the next step's
embedding gather by index, so a frame never comes back to the host. A
readback per depth step would be about 190 fences a second. Upstream needs
CUDA graphs on an H100 for the same reason (its RTF is 0.32 there).

The model is **autoregressive, so streaming is native**. The first audio is
about the encoder, the prefill and one or two frames plus a codec chunk.
That should be well under 300 ms (estimate). Kokoro has to refuse
`stream: true`; this model can honour it, with OpenAI's
`stream_format: "audio"` or `"sse"`.

## What the repo already has

- **The dense Qwen3 KV-cached decode, with an int8 bank**: `ace/lm` (music
  A7/A10). It is 4B there and needs re-shaping to 1.7B. Its prefill takes
  token ids, so a path that takes **input embeddings** is needed.
- **The Gemma-3 tokenizer scheme**: `gemma4/tokenizer.go` is the same
  262k-vocabulary BPE family. Check it against this tokenizer.json before
  reusing it.
- **The Gemma layer pieces** in `gemma4/` (sandwich norms, `(1+w)` norm,
  GELU-tanh, head 256): the encoder is bidirectional with a *symmetric*
  sliding window, which is new.
- **Snake and transposed convolutions**: `ace/vae` (Oobleck) and kokoro's
  generator. The codec's sliding-window transformer is a small dense one.
- **Not here**: any sampler on the device, a Llama depth decoder, ConvNeXt
  upsampling, the RVQ dequantiser.

## Stages

| # | stage | gate |
|---|---|---|
| B0 | Oracle: `reference/dump_breeze.py` (`.venv-breeze`: torch 2.9.1+cpu, transformers 4.57.3, qwen-tts 0.1.1; upstream source in `models/breeze-tts-src` at 6eba901). Upstream's streaming runtime refuses non-CUDA devices, so the script drives upstream's own modules in fp32 on the CPU with `iter_audio_chunks`'s loop, then teacher-forces the sampled frames. Four cases (design, plain, zh, long = 348 frames) | **done** — byte-identical on a second run; the design clip transcribes exactly (parakeet) |
| B1 | `breeze/prompt.go`: the four templates over `gemma4.Tokenizer` (the same Gemma BPE file format, read as-is); nothing to convert, the checkpoint is read directly and the dead tensors are never staged | **done** — `TestPromptIDs`: 44/44 branches exact (English, Chinese, vocal events, emoji, byte fallback, literal special tokens, whitespace runs, clone placeholders) |
| B2 | `breeze/encoder.go`: Rune's vision tower's arrangement (DiT GEMM, GLU_GELU epilogue, gemma4_vit JOIN with the (1 + w) norms staged as such) plus `breeze_enc.comp` (prep, windowed bidirectional attention) | **done** — encoder ≤7.5e-4, projection ≤1.3e-3 relative; 14 ms a segment |
| B3 | `breeze/decoder.go`: ace/lm's decoder generalised to two stacks in one set of buffers; prep/attn rebuilt at 16/8 heads | **done** — teacher-forced over 415 rows: hidden ≤5.1e-4, KL ≤1.3e-6, every argmax |
| B4 | Depth decoder (same file, 8/2 heads, no q/k norm, llama3 rope from a host table, per-position heads) and `breeze/generate.go` (the loop, a host sampler) | **done (host sampler)** — logits ≤1.1e-3, KL ≤6.8e-6, 12 argmax missed of 15,030 (near-ties). The device sampler moved to B7 |
| B5 | `breeze/codec.go` + `breeze_codec.comp`: streamed in chunks with every causal state carried (KV cache, conv tails, depthwise tails); pre_conv∘input_proj and output_proj∘ConvT folded at load; convs on kokoro's A_CONV=2 GEMM | **done** — 57.9–63.7 dB against fp32, identical at chunks of 1, 7, 25 and whole; 27.8 s in 104 ms |
| B6 | Serving (`backend/breeze.go`, `api/speech.go`, `cmd/serve -tts-instruct`): OpenAI's `instructions`, `stream_format` (`audio`: a chunked body, WAV with streaming sizes; `sse`: `speech.audio.delta`/`.done`), `voice` as a name or `{"id": ...}`; routed by model (`breeze-tts-2`, `gpt-4o-mini-tts*`) or by those fields, kokoro the default; instructions sent to kokoro are refused as OpenAI refuses them for tts-1; a `seed` extension. The device lock is taken a pass at a time, so an utterance does not hold other models off the GPU | **done** (test server; ai.service not redeployed) — served radio-DJ clip transcribes exactly; first streamed byte 147 ms; kokoro unchanged |
| B7 | Performance: int8 layer banks (fp16 heads), fused q\|k\|v and gate\|up projections and the residual join folded into the next norm (`breeze_dec.comp`), the split-K ladder (`TestSlabLadder`), a heap top-k on the host. **Not done**: the device sampler and one submit a frame | **in part** — int8 KL against fp32 is 0.03–0.33x upstream bf16's own, fewer argmax misses; **1.65x real time**, first audio 150 ms (was 0.78x) |
| B8 | Cloning (`breeze/clone.go`: the codec encoder, Mimi, on the host in fp32; `api/voices.go` + `backend/breeze_voices.go`: OpenAI's consent → voice → `voice: {"id"}` flow, a file store) | **done** — codes exact against `Qwen3TTSTokenizer.encode` (3,264 of 3,264, jfk past the nominal 250-frame window); served clone and direction transcribe exactly |

## Decisions to make before B0

1. **Licence.** The code is Apache-2.0, but the **weights, derivatives and
   self-hosted outputs are under the BreezeBlue research and
   non-commercial licence**. That is fine for personal use and not fine for
   anything sold.
2. **No fixed voices.** Without reference audio, every request **samples a
   new speaker**: "radio DJ" gives a different DJ every time, and a seed
   only pins it for the same text. A stable designed voice needs B8: design
   once, keep the clip, and clone from it as a named voice. That makes B8
   the step that turns "describe a voice" into "have a voice".
3. **Residency.** About 3 GB int8 or 5.7 GB bf16. It belongs on the
   non-LLM machine ([[deployment-two-machines]]): resident like kokoro, or
   in the swap slot with image, music and video. Its ~2x real time leaves
   little room for contention.
4. **API spelling.** `model: "breeze-tts-2"` with `instructions` follows
   OpenAI (`gpt-4o-mini-tts` takes `instructions`). Decide what `voice`
   means for this model: ignored, a seed, or a saved clone after B8.

## Decisions taken (2026-10-14)

1. Non-commercial licence: accepted by the user.
2. B8 (cloning) is in scope.
3. Residency and the `voice` field: open, settled at B6.

## The first end-to-end numbers (fp16 banks, host loop)

`cmd/breeze`, the radio-DJ prompt, CFG 4, GPU otherwise idle: 65 frames =
5.20 s in 6.68 s (**0.78x real time**), first audio 260 ms. A frame:
backbone 19.0 ms, depth decoder 73.4 ms (15 passes of ~4.9 ms), host
sampling 2.9 ms, codec 1.6 ms. Both decoders read their fp16 weights at
~150 GB/s, so the int8 banks are the first move (B7), then one submit a
frame. **A run beside a busy ai.service is not a measurement**: the same
command read 0.28x while a music job finished.

## B7 — where the time went, and what moved it

The int8 bank alone barely moved the clock (backbone 19 → 18 ms, depth 73 →
65 ms a frame): decode here is dispatch latency, not bytes
([[decode-is-latency-not-bandwidth]]). `TestProfileStep` (a timestamp after
every dispatch) then put **90% of both passes in the GEMVs** — at 77–157
GB/s, because the first slab table was a guess. Two changes:

1. **Fewer dispatches a layer** (22 → 13): q|k|v and gate|up are one
   projection each (stacked rows; `breeze_dec.comp` PREP reads q/k/v from
   the fused row, SWIGLU gate/up from it), and the residual add is folded
   into the next norm (ADDNORM).
2. **The split-K ladder** (`TestSlabLadder`, every layer in one submit so
   no weight is cached): the backbone wants 32 slabs on all four
   projections (204–230 GB/s), the depth decoder 16 except its down
   projection (K 8192) at 4. A backbone step 13.2 → 8.0 ms, a depth pass
   3.3 → 2.0 ms.

A frame is now backbone 7.9 + depth 15 x 2.26 + sampling 0.37 + codec 1.5
ms, and ~4.5 ms of submit gaps: 48 ms for 80 ms of audio, **1.65x real
time** on the radio-DJ prompt with CFG, first audio ~150 ms.

**Pricing the int8 bank** (`TestBackboneTeacher`/`TestDepthTeacher` with
`reference/dump_breeze.py --bf16`: upstream's own bf16 teacher-forced
logits, heads in fp32 as its graphs run them): mean KL against the fp32
oracle at temperature 0.9 is 0.03–0.22x bf16's on the backbone and
0.24–0.33x on the depth decoder, and int8 misses fewer argmaxes than bf16
(backbone 3 vs 11 of 1,322; depth 177 vs 365 of 15,030). The gate is "no
worse than bf16" (`q8VsBF16 = 1`). `BREEZE_BANK=fp16` runs the control.

**What is left, by size**: the depth decoder is 70% of a frame, 15
dependent passes. One submit a frame — the sampler on the device (top-k
50 over 2,051, CFG, a host-seeded uniform) and the next code's row gathered
there — removes the ~4.5 ms of gaps and the 15 readbacks; after it the
floor is the passes themselves (~2 ms each, GEMVs at ~200 GB/s). Two
concurrent requests could share a pass (rows of four), which is how the
LLM's batching paid.

## B8 — voices, consent, and what is not checked

`POST /v1/audio/voice_consents` transcribes the recording (parakeet; `-stt`
is required) and accepts it only if it is the consent phrase within one
word (`GET /v1/audio/consent_phrases`; OpenAI's English phrase with this
server in place of OpenAI). `POST /v1/audio/voices` (≤30 s, WAV) encodes
the sample to codes on the host (~0.25 s a second of audio) and transcribes
it — the model clones from the reference *and its exact transcript*, and
OpenAI transcribes server-side too; a `transcript` field overrides (the
recogniser does not read Chinese). Voices persist under
`-tts-instruct-voices` (default `data/voices`, gitignored); list and delete
are extensions. **Not checked: that consent and sample are one speaker** —
OpenAI rejects a mismatch and nothing here can tell two speakers apart.
The consent recording is kept beside the voice.

## Deployment (not done)

ai.service's line lacks `-tts-instruct`. Adding it costs **4.34 GB of
device memory** (measured, `Model.Bytes`: the int8 layers, the fp16
encoder, heads and codec, the backbone's 2 x 2048-position KV cache, the
arenas) and ~0.9 GB of host memory (the encoder's embedding table and the
code embedding), and ~4 s at startup. A restart
was not done in this session: the server was serving other clients.

## Handoff

Next: B7's one submit a frame (device sampler), then concurrency. The
oracle's venv is `.venv-breeze` (gitignored); `models/breeze-tts-src` is
upstream at 6eba901. Samples to listen to are in `out/breeze-samples/`.
