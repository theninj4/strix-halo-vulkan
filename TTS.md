# TTS — text to speech, where it stands

> The kokoro slice of `SPEECH.md`, on its own. That file carries both audio
> verticals and is the place stage write-ups land; this one is the recap:
> what is built, what it costs, and what is left. Same rules — rewritten
> rather than appended to, history in `TODO.md`, closed findings in
> `research/`.

**Target**: `text -> wav` for `Kokoro-82M`, end to end in Go on Vulkan, with
no Python on the path and every stage validated against a reference dump
before it was optimised.

**Status (2026-09-19, T8)**: **done and serving.** `go run ./cmd/tts -gpu -text
'Hello there.'` speaks; `POST /v1/audio/speech` answers. An utterance is
**44 ms for 3.25 s of audio — 74.2x real time**, from 3848 ms on the CPU
reference. The model side is finished through T6d and T8 closed the last missing feature —
a voice may name a mixture of packs — so what is left is optimisation: T7,
the excitation.

    "The quick brown fox jumps over the lazy dog." (af_heart)
    48 phonemes -> 50 tokens -> 130 frames -> 78000 samples = 3.250 s

    stage          gpu     cpu (T3)
    bert            4ms      104ms   12 ALBERT layers, 46.6x
    dur encoder     3ms       24ms   + the whole text encoder
    durations       0ms        8ms   the head, on the host, in fp32
    prosody         1ms       66ms   gather, shared, both AdaIN stacks
    text encoder    0ms       15ms   now only a readback
    phoneme side    8ms      222ms   18% of the utterance, 1.96ms of GPU time
    decoder         1ms      132ms
    generator      12ms     3160ms
    excitation     14ms       24ms   float64 on the host, see T7
    tail            8ms       22ms
    vocoder        36ms     3626ms   82% of the utterance
    total          44ms     3848ms   74.20x real time

## The stages

| # | Stage | State |
|---|---|---|
| T1 | `convert_kokoro.py` + `dump_kokoro.py` | **done** — 513 tensors readable from Go, 63 dumped, every self-check 0 |
| T2 | Phoneme encoder + ALBERT + predictor, CPU | **done** — 5e-7 relative, durations exact |
| T3 | iSTFTNet decoder, CPU; **first waveform** | **done** — 98 dB against the reference |
| T4a | The generator's residual blocks on Vulkan | **done** — 385x, 22.4 TFLOP/s, [write-up](research/t4-kokoro-vocoder.md) |
| T4b | The upsamplers | **done** — vocoder 487 → 244 ms |
| T4c | The decoder, the tail, every readback between | **done** — vocoder 244 → 38 ms |
| T5a–d | The whole of misaki's English G2P in Go | **done** — corpus 24/24 exact, [write-up](research/t5-kokoro-g2p.md) |
| T6a | PL-BERT on Vulkan | **done** — 104 ms → 4 ms, no duration changes |
| T6b | The F0/N AdaIN stacks | **done** — 40 ms → 1 ms, one new shader |
| T6c | The six bidirectional LSTMs | **done** — 73 ms → 6 ms |
| T6d | The chain between them | **done** — 17 ms → 8 ms, two submits, two readbacks |
| A | `/v1/audio/speech` and `backend/tts.go` | **done** — wav and pcm, voices in `/v1/models` |
| T7 | The excitation on the device | **next** — 14 ms, the largest single stage in the model |
| T8 | Voice blending | **done** — upstream's spelling, checked against `load_voice` at every row |

## What exists

**`kokoro/`** — the model on the CPU (`config.go`, `model.go`, `conv.go`,
`lstm.go`, `albert.go`, `textencoder.go`, `adain.go`, `predictor.go`,
`vocoder.go`, `load.go`, `infer.go`) and on the device over one shared fp32
arena: `adainset.go` (`blockSet`, the `AdainResBlk1d` every stack reuses),
`gpu.go` (a generator upsampling stage, and the tail), `gpudec.go` (the five
decoder AdaIN blocks), `gpuprosody.go` (the F0/N stacks, 62 dispatches in one
submit), `gpubert.go` (PL-BERT), `gpulstm.go` (one bidirectional recurrence,
one dispatch a timestep) and `gpuphonemes.go` (the whole phoneme side, two
submits and two readbacks). 45 tests.

**`g2p/`** — English text to kokoro's 178 IPA symbols, ported from misaki:
`lexicon.go`, `stress.go`, `number.go`, `getnumber.go`, `subtoken.go`,
`tokenize.go`, `homograph.go`, and `espeak.go`/`fallback.go` — a cgo `dlopen`
binding to libespeak-ng with misaki's 25-rule rewrite of its IPA. 18 tests,
each against its own dumped oracle.

**`audio/`** — WAV in and out (16-bit PCM only, on purpose), the radix-2 FFT
and the arbitrary-size float64 DFT that `n_fft = 20` needs, a centred STFT
reproducing `torch.stft` under reflection padding, and `ISTFT` as weighted
overlap-add.

**`shaders/kokoro_*`** — the generator's eight kernels plus T4c's eight, T6a's
attention and `gelu_new` build, T6b's `kokoro_proj.comp` (a 256-to-1
convolution, the one shape the GEMM ladder cannot express), and T6d's gather.

**`cmd/tts`** — phonemes or `-text` in, a WAV and the stage profile out.
`-gpu`, `-voice` (54, or a mixture), `-speed`, `-noise <seed>`, `-list`.

**`kokoro/blend.go`** — a voice specification: one name, `af_bella,af_sky` for
the equal mean (upstream's spelling and upstream's meaning), or
`af_bella:3,af_sky:1` for a weighted one (ours). `reference/dump_blend.py` is
the oracle — `KPipeline.load_voice` run offline over the local packs — and
`TestBlendAgainstUpstream` checks all 510 rows of six mixes against it.

**`backend/tts.go` + `api/speech.go`** — the server adapter and the endpoint.
One mutex serialises utterances; `Speak` takes `input` or the non-OpenAI
`phonemes` field, and `/v1/models` carries the voice list so a client needs no
second call.

## What is left

**T7 — the excitation, 14 ms and 32% of an utterance.** It is on the host
because upstream integrates phase in radians and multiplies by 300: three
seconds holds 1.3e5 radians where one float32 ulp is 0.016, and fp16 cannot
represent the number at all. But `HarmonicSource` already keeps the phase in
*cycles* and wraps before the sine, so nothing it computes is ever large —
which is why it is more accurate than the dump rather than less. The open
question is therefore not whether upstream's accumulator can move, but
**whether the wrapped form is float32-safe on the device**, and that is a
measurement: 78000 samples x eight harmonics of `sin(2*pi*frac(phi))`, the F0
curve upsampled 300:1 and a uv mask, is embarrassingly parallel if the wrap is
exact.

**The remaining 6 ms of the phoneme side's 8, none of it arithmetic** —
ALBERT's `[T, 768]` readback and its host-side embedding stack, the two
readbacks T6d could not remove, and four submits. Closing any of it means
moving the vocoder's input boundary, which is one change rather than four, and
is the thing to do *after* T7.

**`-tts-gpu` is off by default**, because the device path **stages per
request**: kokoro's arenas are sized for one utterance's frame count, the
durations decide that, and so the prosody runs on the host to settle the
length before anything can be staged — which is why `synthesize` computes it
twice. Two things would fix it, both measurements rather than arguments:
**bucketed residency** (size the arenas above the utterance, zero-pad the
alignment, trim by `Prosody.Samples`; the phoneme side already accepts any
count up to the one it was built for, and whether the generator's stages
tolerate the padding is the open question), and **skipping the second prosody
pass** (the vocoder reads `asr`, `f0` and `energy` from the host anyway, so the
first pass could feed the device directly and the phoneme side would not need
staging at all).

**The style row is indexed by the character count** of the phoneme string minus
whatever fell outside the vocabulary — which is how `KPipeline` does it, and is
*not* the token count. A front end that emits a different number of characters
for the same utterance therefore picks a different style row. Known, unsettled.

**G2P on unseen text.** The designed corpus is exact, 24 of 24; on 400
sentences nobody chose, **68.8% of sentences and 92.4% of phoneme words**
agree with misaki. The homograph tagger is 88.4% against spacy over 1962
occurrences, against an 83.8% baseline. Closing the remainder means more
measured rules, not more reasoning — the rules that looked syntactically
obvious scored worse than no tagger at all.

**Serving defaults worth knowing.** `-noise 0` leaves the excitation noise
*off*, which is the reproducible configuration the reference dump was taken
with; the noise is worth 13.8 dB and an utterance meant to be listened to
wants it on. Without libespeak-ng, words outside the lexicon are **dropped**
with a warning rather than mispronounced — a wrong utterance rather than a
slow one, which is why `-espeak` defaults on and a failure to open it is a
warning and not fatal.

**Refused rather than faked.** `stream: true` — kokoro decides the whole
utterance's durations before a single sample exists, so there is nothing to
send early and a whole body dressed as a stream would be a lie about latency.
`mp3`/`opus`/`aac`/`flac` — there is no audio encoder here; the 400 names
`wav` and `pcm` and points out that mp3 is OpenAI's default rather than
something the client chose.
