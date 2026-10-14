# Instructed TTS — Breeze-TTS-2 beside kokoro (B-stages)

> **Live (2026-10-14).** B0–B8 are done and every part runs on the device
> to its gate; **deployed on ai.service** (`-tts-instruct`) the same day.
> The device frame (B7) runs it at 1.93x real time, first audio ~160 ms,
> and two requests share every frame (1.59x the throughput). Stage letters are **B0–B8**;
> code in `breeze/`, `cmd/breeze`, `shaders/breeze*.{go,comp}`.

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
| B6 | Serving (`backend/breeze.go`, `api/speech.go`, `cmd/serve -tts-instruct`): OpenAI's `instructions`, `stream_format` (`audio`: a chunked body, WAV with streaming sizes; `sse`: `speech.audio.delta`/`.done`), `voice` as a name or `{"id": ...}`; routed by model (`breeze-tts-2`, `gpt-4o-mini-tts*`) or by those fields, kokoro the default; instructions sent to kokoro are refused as OpenAI refuses them for tts-1; a `seed` extension. The device lock is taken a pass at a time, so an utterance does not hold other models off the GPU | **done** (deployed 2026-10-14) — served radio-DJ clip transcribes exactly; first streamed byte 147 ms; kokoro unchanged |
| B7 | Performance: int8 layer banks (fp16 heads), fused q\|k\|v and gate\|up projections and the residual join folded into the next norm (`breeze_dec.comp`), the split-K ladder (`TestSlabLadder`), then the **device frame** (`breeze/frame.go`, `breeze_frame.comp`): the sampler, the input gathers and the frame embedding on the device, a frame recorded once (`vk.Prerecorded`) and replayed as four submits | **done** — int8 KL against fp32 is 0.03–0.33x upstream bf16's own; device sampler 0 of 3,000 draws off the host's; seeded audio identical to the host loop; two streams share a frame, each bit-identical to alone; **1.93x real time** alone, 1.59x throughput with two (was 0.78x, then 1.65x, 1.80x) |
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

**The device frame (done 2026-10-14).** `breeze_frame.comp` does what the
host did between passes: GATHER (the backbone's hidden rows and the next
code's bf16 embedding row into the depth decoder's fp16 input), FRAME (the
16 code rows summed in fp32, frameEmbed's order, into the backbone's input),
COPYH (the hidden rows kept for the next frame) and SAMPLE (CFG, the
penalty and the temperature with float64 constants as fp32 hi/lo pairs, a
radix select of the 50th-largest key a byte at a time, softmax, one draw
against a host uniform by an inclusive scan). The frame's state is data —
codes `fr[0..16]`, the penalty's marks, sixteen uniforms a frame from the
same PCG stream in the host loop's order, the backbone rows' positions — so
the dispatches never change and are recorded once: three recordings of five
depth passes and one backbone step, keyed by the backbone's key-chunk count
(it moves every 64 positions) and the sampler settings. Four submits a
frame, each under `GenOptions.Lock`, at most ~11 ms of device time each, so
a waiting model is never held off for a whole frame.

Gates: `TestDeviceSampler` (0 of 3,000 random draws differ from the host
sampler, CFG on and off, penalty, EOS column) and `TestDeviceFrames` (a
seeded utterance gives **identical audio** from both loops, design with CFG
and plain). `GenOptions.HostSampler` / `cmd/breeze -host-sampler` runs the
old loop as the control.

Radio-DJ prompt, CFG 4, GPU idle: a frame is **41.6 ms of device time + 1.5
ms codec, 43.9 ms wall** (was 48), **1.80x real time**, first audio 150
ms. Plain (no CFG) 40 ms wall.

**What is left is the passes themselves.** The depth decoder reads its
whole 367 M int8 weights once a pass, fifteen times a frame (~5.5 GB), at
~160 GB/s; the backbone ~1.4 GB at ~175 GB/s. That is close to the
bandwidth floor (~31 ms a frame at 220 GB/s), so one request will not get
much faster; throughput comes from sharing the passes.

**Shared passes (done 2026-10-14).** Two utterances run at once and ride
the same frame (`frame.go`; `Model.Generate`'s scheduler in
`generate.go`). Each has a stream slot: two KV slots in each stack, its own
fr/marks/uniforms and its own codec stream (`Codec.Stream`: new arenas over
the shared weights). A request prefills in its own goroutine between frame
submits (every device use goes through `Model.run`: the model's mutex,
then the device lock), seeds its slot, and joins the scheduler, which runs
every active stream's frame as one: a gather and a sample a stream, every
row through one set of passes. The recordings are keyed by the
composition. A third request waits for a slot.

Two things this needed:

1. **The split-K scratch was too small.** One partials buffer serves every
   GEMV, and it was sized for the backbone at three rows; the depth
   decoder's gate|up at four rows (16 slabs x 4 x 16,384) ran past it into
   the depth decoder's own residual stream. Two rows fitted, so nothing
   showed until a four-row GEMV. It is now sized for the widest projection
   at `MaxRows`.
2. **Eight-row GEMVs** (`breeze_gemv*`: llm_gemv at MAXROWS 8). A depth
   pass 0 is a hidden row and a code row a branch, eight rows for two CFG
   pairs; past MaxRows it fell to the GEMM, whose sums round differently,
   so a request's audio depended on its neighbour. On the GEMV every row's
   arithmetic is independent of the others: `TestSharedFrames` gets each
   request's audio **bit-identical** alone and shared. It is also faster
   alone: a CFG request's four-row pass 0 used to be a GEMM.

Numbers (GPU idle): a CFG + a plain request, 2.58 s together against 4.11
s one after the other (**1.59x the throughput**); a frame of three branches
is ~40-41 ms of device time against ~38 alone. Alone, the radio-DJ line is
now 40.8 ms a frame, **1.93x real time**, first audio 165 ms. Served (a
test server): two instructed requests at once both stream from ~0.25 s and
transcribe exactly; a third waits for a slot. Device memory **5.08 GB** (the
second KV pair and codec stream: +0.6 GB).

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

## The closing breath (2026-10-14)

The served narrator line "...with a light breeze from the west." came back
from parakeet as "...from the west, and". It is the model, not the port:

| run (narrator, CFG 4) | ends in an extra word |
| --- | --- |
| upstream's own loop, fp32 on the CPU (`dump_breeze.generate`) | 3 of 8 seeds |
| ours, int8 (served) | 5 of 8 |
| ours, fp16 layers (`cmd/breeze -fp16`) | 10 of 12 |
| ours, temperature 0.6 / 0.3 (`-temp`) | 4 of 8 / 4 of 8 |
| CFG 1 | 3 of 6 ("turning" once) |

The extra "word" ("and", "said", "turning") is an exhale: the 0.2-0.8 s
after the last word's voicing is unvoiced (pitch autocorrelation 0.1-0.3
against 0.7-0.9 in speech) at speech level (-25 to -33 dB). Clips that
transcribe clean end on the same kind of tail. Text with no final full stop
can draw a real voiced word ("west to"); that is not touched.

Ruled out:
- **Stopping on EOS sooner.** EOS is all or nothing: its log10 probability
  is -5 to -10 on every frame, the breath's included, then ~0 on the last.
- **Telling the breath from the signal.** A voicing rule cut 25 of 27 test
  clips clean, but on a wider set it cut a rough, excited "breakfast show"
  (the DJ voice has voicing 0.2-0.5, like a breath) and a whispered "us".
- **Parakeet on the last 3 s alone.** It heard "west." and nothing after.
  It heard the breath as a word only with the sentence before it.

The fix (`breeze/tail.go`, `GenOptions.Ending`; `backend/breeze.go`
`ending`/`speechEnd`):
- **Hold back the unvoiced tail.** `Generate` passes audio on as it comes,
  holding back only what follows the last voiced run (two 40 ms windows),
  at most 1 s.
- **Find where the words end.** At the end the backend runs the server's
  parakeet (`-stt`) on the last 30 s. When the text's last word is heard
  with more words after it, the speech ends at that word.
- **Cut only held audio.** The held audio past that point is cut behind a
  40 ms fade. Audio already passed on is never cut.
- **No STT, no change.** Without `-stt`, or if the check fails, the ending
  stays as it is.

Served on a test server (port 18088), seeded:

| check | result |
| --- | --- |
| narrator line | **12 of 12 clean**, was 7 of 12 with "and" |
| four more texts (a DJ, plain "...the facts before you post", whisper, podcast host) | unchanged, all endings kept |
| first audio, streamed | 164 ms (live 164 ms) |
| end of request | +30-40 ms for the check |
| streamed bytes | equal to the whole response |

The six transcripts on the four texts that still differ were already off
without the fix, and none at the end: a "sighing" voice opening with
"Uh"/"Ah", whisper misreads, "See ya". Tests: `TestTailTrim`,
`TestSpeechEnd`.

## Deployment (2026-10-14)

`-tts-instruct` is on ai.service's line (the `ai.service` template),
deployed on the user's go. It costs **~4.5 GB of device memory**
(`Model.Bytes`: the int8 layers, the fp16 encoder, heads and codec, the
code embedding's bf16 copy the device frame gathers from (134 MB), the
backbone's 2 x 2048-position KV cache, the arenas), ~0.9 GB of host memory,
and 3.7 s at startup. Custom voices live in `data/voices` under the unit's
working directory. The first deployment ran the host-sampler loop (1.65x);
the device frame was redeployed the same day, and shared passes (5.08 GB)
at 12:18 that day: two instructed requests sent at once both streamed from
one set of frames (9.3 s of audio in 3.0 s of wall time), and both transcribed
correctly. A trailing breath that parakeet hears as "and" shows up on some
seeds whether the request runs alone or shared: see the closing breath above
(cut since, not yet deployed).

## Handoff

Next: nothing planned. B7 closed with the device frame and shared passes (1.93x alone, two streams at once). The
oracle's venv is `.venv-breeze` (gitignored); `models/breeze-tts-src` is
upstream at 6eba901. Samples to listen to are in `out/breeze-samples/`.
