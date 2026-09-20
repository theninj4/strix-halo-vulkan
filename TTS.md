# TTS — text to speech, where it stands

> The kokoro slice of `SPEECH.md`, on its own. That file carries both audio
> verticals and is the place stage write-ups land; this one is the recap:
> what is built, what it costs, and what is left. Same rules — rewritten
> rather than appended to, history in `TODO.md`, closed findings in
> `research/`.

**Target**: `text -> wav` for `Kokoro-82M`, end to end in Go on Vulkan, with
no Python on the path and every stage validated against a reference dump
before it was optimised.

**Status (2026-09-20, T10)**: **done and serving, and it now costs the same
per second of audio at any length.** `go run ./cmd/tts -gpu -text 'Hello
there.'` speaks; `POST /v1/audio/speech` answers in **59 ms for 6.45 s of
audio**. An utterance is **31 ms for 3.25 s — 105x real time**, from 3848 ms
on the CPU reference, and a nineteen-second one is **162 ms — 120x**. Every
stage of the model runs on the device: T7 closed the last stage that was still
on the host, T8 the last missing feature (a voice may name a mixture of
packs), T9 the last place where the *server* cost more than the model, and
**T10 the last place where the model's cost depended on how much you said** —
PL-BERT's attention was a scalar kernel sized for fifty tokens and is now
stage 3c's matrix-core one, 56x faster a layer at the model's ceiling.

    "The quick brown fox jumps over the lazy dog." (af_heart)
    48 phonemes -> 50 tokens -> 130 frames -> 78000 samples = 3.250 s

    stage          gpu     cpu (T3)
    bert            4ms      104ms   12 ALBERT layers, 46.6x
    dur encoder     3ms       24ms   + the whole text encoder
    durations       0ms        8ms   the head, on the host, in fp32
    prosody         1ms       66ms   gather, shared, both AdaIN stacks
    text encoder    0ms       15ms   now only a readback
    phoneme side    8ms      222ms   26% of the utterance, 1.96ms of GPU time
    decoder         1ms      132ms
    generator      12ms     3160ms
    excitation      0ms       24ms   0.2ms measured, 23us of it GPU
    tail            8ms       22ms
    vocoder        22ms     3626ms   73% of the utterance
    total          31ms     3848ms   105x real time

A short utterance is not where the work is any more, because the short one is
the length every stage was tuned at. The same table at a paragraph — 319
phonemes, 19.525 s — is what T10 moved, and it is the one to read when
changing anything here:

    stage           T9      T10
    bert            75ms    12ms    2.5 ms of it on the device; the rest is
                                    the host embedding stack
    phonemes        96ms    33ms    43% of an utterance -> 20%
    generator       64ms    66ms
    tail            59ms    57ms
    vocoder        128ms   129ms    57% -> 80%
    total          225ms   162ms    86.9x -> 120.5x real time

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
| T7 | The excitation and its transform on Vulkan | **done** — 14 ms → 0.2 ms, [write-up](SPEECH.md#t7--the-excitation-done) |
| T8 | Voice blending | **done** — upstream's spelling, checked against `load_voice` at every row |
| T9 | One staging for the life of the server | **done** — the endpoint 550 ms → 59, the same bytes, [write-up](SPEECH.md#t9--one-staging-for-the-life-of-the-server-done) |
| T10 | PL-BERT's attention on the matrix cores | **done** — a layer **56x** at 510 tokens, an utterance 225 ms → **162**, no duration moved, [write-up](SPEECH.md#t10--pl-berts-attention-on-the-matrix-cores-done) |

## What exists

**`kokoro/`** — the model on the CPU (`config.go`, `model.go`, `conv.go`,
`lstm.go`, `albert.go`, `textencoder.go`, `adain.go`, `predictor.go`,
`vocoder.go`, `load.go`, `infer.go`) and on the device over one shared fp32
arena: `adainset.go` (`blockSet`, the `AdainResBlk1d` every stack reuses),
`gpu.go` (a generator upsampling stage, and the tail), `gpudec.go` (the five
decoder AdaIN blocks), `gpuprosody.go` (the F0/N stacks, 62 dispatches in one
submit), `gpubert.go` (PL-BERT, with stage 3c's matrix-core attention at
`HEAD_DIM=64` since T10), `gpulstm.go` (one bidirectional recurrence,
one dispatch a timestep), `gpuphonemes.go` (the whole phoneme side, two
submits and two readbacks) and `gpusource.go` (the excitation and its forward
transform, two dispatches over the only arena here that is HOST_CACHED).
Every one of those objects is sized for a *ceiling* rather than for an
utterance: `SetFrames` takes a clip within it (T9). 60 tests.

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
convolution, the one shape the GEMM ladder cannot express), T6d's gather, and
T7's `kokoro_source.comp` and `kokoro_srcstft.comp` — the sine bank, and the
forward transform that mirrors `kokoro_istft.comp`.

**`cmd/tts`** — phonemes or `-text` in, a WAV and the stage profile out.
`-gpu`, `-voice` (54, or a mixture), `-speed`, `-noise <seed>`, `-list`, and
`-frames N` to stage for a ceiling instead of this utterance — which is how
the claim that a ceiling costs nothing per request is checked from the command
line.

**`kokoro/blend.go`** — a voice specification: one name, `af_bella,af_sky` for
the equal mean (upstream's spelling and upstream's meaning), or
`af_bella:3,af_sky:1` for a weighted one (ours). `reference/dump_blend.py` is
the oracle — `KPipeline.load_voice` run offline over the local packs — and
`TestBlendAgainstUpstream` checks all 510 rows of six mixes against it.

**`backend/tts.go` + `api/speech.go`** — the server adapter and the endpoint.
One mutex serialises utterances; `Speak` takes `input` or the non-OpenAI
`phonemes` field, and `/v1/models` carries the voice list so a client needs no
second call. Since T9 the device attachment is staged **once at startup** for
`-tts-frames` (1000 frames, 25 s) and every request is the model alone: a
request re-conditions the voice, sizes the arenas with `Vocoder.SetFrames`,
and runs.

## What is left

**The generator's 12 ms and the tail's 8**, which T4 left measured and which
are the only parts of this model that were ever arithmetic-bound. Together
they are 65% of an utterance.

**The excitation's remaining 0.2 ms is 88% submit and readback** — 23
microseconds of GPU against 155 of submit and 70 of download. Closing it means
moving the two noise convolutions that consume the spectrogram onto the device
too, after which nothing of the stage would touch the bus.

**The remaining 6 ms of the phoneme side's 8, none of it arithmetic** —
ALBERT's `[T, 768]` readback and its host-side embedding stack, the two
readbacks T6d could not remove, and four submits. Closing any of it means
moving the vocoder's input boundary, which is one change rather than four.

**`-tts-frames` is a memory budget, not a latency one.** The arenas are
~150 MB plus 0.7 MB a frame, so the 1000-frame default is 848 MB and a
100-second ceiling would be 3.0 GB. It costs nothing per request: every
dispatch extent is the utterance's, and the Montana sentence is 48 ms staged
for its own 258 frames and 48 ms staged for 1000. An utterance past the
ceiling restages once, rounded up to the next 256 frames, and keeps the larger
arenas — though the voice packs have only 510 style rows, so only `speed < 1`
reaches it.

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
wants it on. Since T7 the noise is drawn on the device from a hash of (seed,
sample, harmonic) rather than from Go's generator, so **`-noise n` is
reproducible on a given path but the CPU and GPU paths do not produce the same
samples for the same seed** — the same distribution and the same rule, not the
same sequence. And with the noise off, the excitation's phase spectrum is
analytically zero over 27% of its bins, which makes the waveform's agreement
with the dump a draw spanning 13.8 to 20.2 dB; the device draws 13.8 where the
host draws 18.2, and neither is more correct than the other. SPEECH.md T7 has
the measurement. Without libespeak-ng, words outside the lexicon are **dropped**
with a warning rather than mispronounced — a wrong utterance rather than a
slow one, which is why `-espeak` defaults on and a failure to open it is a
warning and not fatal.

**Refused rather than faked.** `stream: true` — kokoro decides the whole
utterance's durations before a single sample exists, so there is nothing to
send early and a whole body dressed as a stream would be a lie about latency.
`mp3`/`opus`/`aac`/`flac` — there is no audio encoder here; the 400 names
`wav` and `pcm` and points out that mp3 is OpenAI's default rather than
something the client chose.
