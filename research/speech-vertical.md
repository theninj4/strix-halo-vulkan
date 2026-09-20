# SPEECH — the two audio verticals

> **ARCHIVED 2026-09-20** — frozen as the closing record of the two audio verticals (parakeet S1–S8, kokoro T1–T10, R1, W1 — the T6–T10/R1/W1 write-ups live only here). It was
> `SPEECH.md` at the repo root; the live state of play is now [`../TODO.md`](../TODO.md).

> **Current work.** `zimage-pipeline.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `../TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `../GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-20, W1)**: **both verticals are on the device, the loop
between them closes over HTTP, synthesis now costs the same per second of
audio however long the utterance is, and the pair answers Home Assistant on a
port of its own.**
W1 added the second door: `-wyoming 0.0.0.0:10300` serves the same two
backends over the protocol Home Assistant's voice pipeline speaks, out of the
same process and the same staging — checked against upstream's own client,
and byte for byte the same waveform as `/v1/audio/speech`.
See [W1](#w1--the-wyoming-door-done).
T9 staged kokoro once for the life of the server instead of once per request:
`POST /v1/audio/speech` went from **550 ms to 59** for 6.45 s of audio, byte
for byte the same waveform.
Speech-to-text transcribes an 11 s clip in 43 ms, 257x real time, whole model
resident, S1–S8 done. **Text-to-speech is 31 ms for 3.25 s of audio**, 105x
real time. T6 is finished: ALBERT 104 ms to 4 (T6a), the F0/N AdaIN stacks 40
to 1 (T6b), the six bidirectional LSTMs 73 to 6 (T6c) and the chain between
them 17 to 8 (T6d) — so **the phoneme side went from 222 ms to 8 and is now
26% of an utterance, over 1.96 ms of GPU time**. The other 74% is the vocoder,
whose 22 ms T4c and T7 cut from 3626.
T8 closed the last open feature: a voice may name a **mixture** of packs, in
upstream's own spelling, and T9 closed the last place where the server cost
more than the model.

**R1 closed the loop and moved the target.** `cmd/roundtrip` runs text through
both endpoints and back — six prose cases return exactly — and what it
measured is that the two verticals scaled in opposite directions:
transcription amortises its fixed cost away to 257x, while synthesis bottomed
out at 8.3 ms per second of audio and then got **36% worse per second** by a
nineteen-second utterance. `cmd/tts` localised that to **PL-BERT, 4 ms at 50
tokens and 75 at 321**. See [R1](#r1--the-round-trip-and-what-it-says-to-do-next).

**T10 removed it.** PL-BERT's attention was a scalar kernel chosen when the
sequence was fifty tokens; on the matrix cores it is **216x faster at 321
tokens and 420x at 510**, which takes the whole encoder from 66.8 ms to
**2.5** and an ALBERT layer from 16.2 ms to **288 us** at the model's ceiling.
A nineteen-second utterance is **225 ms → 162** (86.9x → **120.5x** real
time), the round trip is **61.6x → 70.1x**, and the speech leg is a straight
line instead of a curve: 8.0 ms per second of audio from five seconds to
nineteen, where it used to climb to 11.3. Not one duration moved.
See [T10](#t10--pl-berts-attention-on-the-matrix-cores-done).

`go run ./cmd/tts -gpu -text 'Hello there.'` speaks, with no Python on the
path. T5 opened with a measurement instead of code — **91% of running-text
tokens are a dictionary lookup and a part-of-speech tagger decides 1.74% of
them** — and ended with the whole of misaki's English G2P ported: five
components exact against their own dumped oracles, twelve measured rules where
spacy's tagger mattered, and a cgo binding to libespeak-ng for the rest.
**The designed corpus is exact, 24 of 24; on 400 sentences nobody chose, 92.4%
of phoneme words agree.** [Write-up](t5-kokoro-g2p.md).

    "In 2024 the team shipped 1,024 kernels and spent $3.5 million, up 12%."
    -> ɪn twˈɛnti twˈɛnti fˈɔɹ ðə tˈim ʃˈɪpt wˈʌn θˈWzᵊnd twˈɛnti fˈɔɹ
       kˈɜɹnᵊlz ænd spˈɛnt θɹˈi pYnt fˈIv mˈɪljᵊn dˈɑləɹz, ˌʌp twˈɛlv pəɹsˈɛnt.
    character for character what misaki gives

    text to speech, T7 (kokoro)
    "The quick brown fox jumps over the lazy dog." (af_heart)
    48 phonemes -> 50 tokens -> 130 frames -> 78000 samples = 3.250 s

    stage          gpu    (T6d)   cpu (T3)
    bert            4ms      4ms     104ms   12 ALBERT layers, 46.6x
    dur encoder     3ms      3ms      24ms   + the whole text encoder
    durations       0ms      0ms       8ms   the head, on the host, in fp32
    prosody         1ms      1ms      66ms   gather, shared, both AdaIN stacks
    text encoder    0ms      0ms      15ms   now only a readback
    phoneme side    8ms      8ms     222ms   26% of the utterance
    decoder         1ms      1ms     132ms
    generator      12ms     12ms    3160ms
    excitation      0ms     14ms      24ms   0.2ms measured, 23us of it GPU
    tail            8ms      8ms      22ms
    vocoder        22ms     36ms    3626ms   73% of the utterance
    total          31ms     44ms    3848ms   105x real time, from 74.20x

    the phoneme side on the device, by family   1.957 ms of GPU time
      six recurrences, steps only  1.478ms  76%   shared alone is 0.508
      F0/N AdaIN stacks            0.204ms  10%
      input projections            0.072ms   4%   six GEMMs, out of the loops
      text encoder convolutions    0.076ms   4%
      everything else              0.127ms   6%   norms, narrows, the gather

    speech to text, S8 (parakeet, Vulkan)
    testdata/jfk.wav: 11.000 s at 16000 Hz
    gpu: 24 layers, 1162 MB of encoder weights + 24 MB of decoder weights

    stage         time  share      (S7)     (S6)    (CPU, S5)
    front end     21ms   48.0%     25ms     21ms       20ms
    encoder       17ms   39.8%     20ms    106ms     2454ms
    decode         5ms   12.2%    209ms    206ms      209ms
    total         43ms             255ms    331ms      2.68s

    257x real time, 43x at S7, 33x at S6, 4.1x on the CPU

## Where the work stands

| # | Stage | State |
|---|---|---|
| S1 | `safetensors` reads the parakeet checkpoint; WAV reader | **done** |
| S2 | `reference/dump_parakeet.py` — mel, layer outputs, joint, decode trace | **done** — 59 tensors, four self-checks |
| S3 | Mel front end in Go, against S2's mel | **done** — 3.1e-4 absolute |
| S4 | Encoder, CPU reference, layer by layer | **done** — 1.7e-4 relative |
| S5 | Prediction net + joint + TDT greedy decode; **first transcript** | **done** — exact string |
| S6 | Encoder on Vulkan (GEMMs, LayerNorm, attention, conv) | **done** — 178x, [write-up](s6-parakeet-encoder.md) |
| S7 | The subsampling stack on Vulkan | **done** — 95x, [write-up](s7-parakeet-subsampling.md) |
| S8 | The projector, the prediction net, the joint and the TDT loop | **done** — 43x, 257x real time, [write-up](s8-parakeet-decode.md) |
| S9 | Long clips: chunking, or full attention at T=3000 | open, see below |
| S10 | The front end on the device, or a faster one on the host | open — **48% of the pipeline** |
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | **done** — 513 tensors readable from Go, 63 dumped, every self-check 0 |
| T2 | Phoneme encoder + ALBERT + predictor, CPU, against T1 | **done** — 5e-7 relative, durations exact |
| T3 | iSTFTNet decoder, CPU; **first waveform** | **done** — 98 dB against the reference, `cmd/tts` speaks |
| T4a | The generator's residual blocks on Vulkan | **done** — 385x, 22.4 TFLOP/s, [write-up](t4-kokoro-vocoder.md) |
| T4b | The upsamplers; a stage that does not come back | **done** — vocoder 487 → 244 ms |
| T4c | The decoder, the tail, and every readback between them | **done** — vocoder 244 → 38 ms, 12.90x real time |
| T5a | The G2P oracle: `convert_misaki.py`, `dump_g2p.py`, the coverage survey | **done** — 91% is a dictionary, a tagger is worth 1.74% |
| T5b | The lexicon, the suffix rules, the special cases and the stress rules in Go | **done** — 204 of 204 tokens exact |
| T5c | `num2words`, `get_number`, the subtokenizer, the tokenizer and the merge loop | **done** — each exact; `-text` speaks |
| T5d | The espeak fallback and the homograph tagger | **done** — 39 words exact, tagger 88.4% against spacy; corpus 24/24 |
| T6a | PL-BERT on Vulkan | **done** — 104 ms to 4 ms, 46.6x, no duration changes |
| T6b | The F0/N AdaIN stacks, 40 ms of the prosody predictor's 67 | **done** — 40 ms to 1 ms, one new shader |
| T6c | The six bidirectional LSTMs, 73 ms — 96% of the phoneme side | **done** — 73 ms to 6 ms, durations unchanged |
| T6d | The chain between them: readbacks, the regulator, the text encoder | **done** — 17 ms to 8 ms, two submits and two readbacks |
| T7 | The excitation: float64 phase accumulation on the host | **done** — 14 ms to 0.2 ms, two new shaders; the wrapped phase is float32-safe |
| T8 | Voice blending: a request that names several voices | **done** — upstream's own spelling, checked against `load_voice` at every row |
| T9 | The server: one staging for the life of the process | **done** — `/v1/audio/speech` 550 ms to 59, and the bytes are unchanged |
| R1 | The round trip: `cmd/roundtrip`, text → speech → text over HTTP | **done** — six cases exact, 61.6x real time, and it named what was next |
| T10 | PL-BERT's attention on the matrix cores | **done** — a layer **56x** at the model's ceiling, an utterance 225 ms → **162**, the loop **70.1x** |
| W1 | The Wyoming door: both verticals to Home Assistant | **done** — one port, 28 voices, byte-identical to the HTTP door |

## What exists

**`wyoming/`** — the second door: the two backends over the protocol Home
Assistant's voice pipeline speaks, on its own TCP port beside the HTTP API
(W1, below). `event.go` is the framing, `info.go` the events, `voices.go`
what a kokoro pack's name says about it and `server.go` the conversation.
It imports `api` for the two backend interfaces and `audio` for the filter,
and nothing else — no model and no Vulkan.

**`audio/`** — WAV in and out (16-bit PCM only, on purpose), a radix-2 FFT and
an arbitrary-size direct DFT in float64, a centred STFT that reproduces
`torch.stft` under either padding — zeros for parakeet's front end,
**reflection** for kokoro's vocoder, which is torch's undocumented default —
`MagnitudePhase`, an `ISTFT` doing weighted overlap-add with the window-square
envelope divided out, and a Slaney mel filterbank that reproduces
`librosa.filters.mel` to a float32 ulp. The DFT exists because **n_fft is 20**
and 20 is not a power of two.

**`parakeet/`** — the model on the CPU (`frontend.go`, `subsampling.go`,
`encoder.go`, `decoder.go`, `decode.go`, `tokenizer.go`, `load.go`) and on the
device:

  - `gpu.go`, `gpugraph.go`, `gpusub.go` — a `GPUEncoder` holding 1.16 GB of
    fp16 and running a clip as **946 dispatches** over six shared arenas: ten
    for the subsampling convolutions, 936 for the 24 conformer layers.
    `Encoder.Apply` and `GPUEncoder.ApplyMel` have the same signature, and
    `HostSubsampling` puts the convolutions back on the CPU.
  - `gpudecode.go` — a `GPUDecoder` holding 24 MB of fp16 over its own four
    buffers, running the projector once per clip and **eleven dispatches per
    emission**. `Attach` lets it read the encoder's residual stream in place,
    so the hidden states never cross the bus.

**`shaders/parakeet_*`** — fifteen kernels. Seven from S6 (the LayerNorm with
a mean and an affine, silu, the scalar-weighted residual, the GLU, the
depthwise convolution over time with its folded BatchNorm, the `bias_v`
narrowing, the relative-position shift), four from S7 (the stride-2
convolution over a single-channel input, the depthwise form of it, the
bias-and-mask epilogue, the flatten) and four from S8 (the LSTM's concatenated
A operand, the LSTM cell, the joint's addition and rectification, the two
argmaxes). Plus two builds of shaders that already existed: the fragment pack
with `bias_u` folded in, and the score kernel with the position term as an
additive bias (`REL_BIAS`).

**`cmd/asr`** — a WAV in, a transcript and a stage profile out. `-gpu` runs
the whole model on the device; `-hostsub` and `-hostdec` put the convolutions
and the transducer tail back on the CPU, which is what the S7 and S8
measurements are against; `-profile` times every dispatch; `-v` prints the
per-emission trace.

**`reference/dump_parakeet.py`** — the parakeet oracle, which walks the whole
model rather than its ends and checks itself in four places.

**`reference/convert_kokoro.py`, `reference/dump_kokoro.py`** — T1, below.

**`kokoro/`** — the model on the CPU: `config.go` (and `Phonemes`, the
vocabulary lookup), `model.go`, `conv.go`, `lstm.go`, `albert.go`,
`textencoder.go`, `adain.go`, `predictor.go`, `vocoder.go`, `load.go` and
`infer.go` (`Prosody`, `Synthesize`, `Speak`). 18 tests — sixteen against the
dump, two against the definitions over shapes the dump does not reach.

On the device, `blockSet` and four objects built on or beside it, over
**one shared fp32 arena**:

  - `adainset.go` — `blockSet`, the part of StyleTTS2 that is the same
    wherever it appears: `AdainResBlk1d` staged on the device. The padded row
    stride, the bordered fp16 operand, the two-phase AdaIN reduction, the
    weight bank, the style projection and the eleven-dispatch graph. It owns
    no input and no output; **two objects embed it** and supply only where a
    block's input comes from and where its output goes.
  - `gpu.go` — `GPUBlocks` is one generator upsampling stage: its transposed
    convolution, its three Snake residual blocks and the excitation's, staged
    as fp16 fragment tiles. The last one also carries the **tail**
    (`tailWeights`): `conv_post`, the two nonlinearities and the inverse
    transform, so `RunStageWave` returns 78000 samples rather than an 8 MB
    activation.
  - `gpudec.go` — `GPUDecoder` is the five AdaIN blocks, 31 M parameters of
    fp16 over channel counts of 514, 1090, 1024 and 512. Its plumbing is the
    **concatenation as a row stride**: 66 side channels written once, and each
    block's epilogue writing its 1024 beside them.
  - `gpuprosody.go` — `GPUProsody` is the predictor's F0 and energy stacks:
    the same block again, six of them at 512 and 256 channels in two parallel
    stacks off one input, and the two 1-wide projections that make them
    curves. **62 dispatches in one submit**, 0.39 ms on the device. Four of the
    six keep their channel count and so have no shortcut convolution at all —
    their epilogue reads the block's input where it already lies.
  - `gpubert.go` — `GPUAlbert` is PL-BERT: one shared weight group, fifteen
    dispatches a layer, arenas sized for the model's 512-token limit so it is
    staged once for any utterance. Its biases are columns of K.
  - `sharedArena` — one fp32 buffer laid out across the decoder and both
    generator stages before any of them is built, so `stage0.aXin` *is*
    `decoder.aOut` and `stage1.aXin` is `stage0.aSum`. `Model.AttachGPU` does
    the layout, the commit and the aliasing; `Resident()` is how
    `Vocoder.Apply` knows there is nothing to upload. `GPUProsody` keeps an
    arena of its own: its output is two curves the host convolves on the way
    into the vocoder, so nothing of its would stay on the device anyway.

**`kokoro/gpulstm.go` and `shaders/kokoro_lstm.comp`** — `GPULSTM`, one
bidirectional recurrence on the device: a narrow, the input projection as one
GEMM over both directions, and **one dispatch a timestep** covering both.
`SetTail` is what turns the style concatenation into part of the operand.
Four builds of the step kernel over two knobs, of which only one matters — see
T6c below.

**`kokoro/gpuphonemes.go`** — `GPUPhonemes`, the whole phoneme side over one
shared arena: the duration encoder's three LSTM/AdaLayerNorm pairs, the
recurrence before the duration head, the text encoder's three convolutions and
its own recurrence, the length regulator, `shared`, and the F0/N stacks.
**Two submits and two readbacks**, and the only thing in it that is not an
offset is `kokoro_gather.comp`. The duration head is deliberately *not* here —
see T6d.

**`shaders/kokoro_*`** — the generator's eight kernels (four builds of
`dit_gemm.comp` with `A_CONV=1`, the AdaIN reduction's two halves, the fused
affine/Snake/narrow and its leaky build, the residual add and its copy
variant, the upsampler's epilogue with and without the reflection pad) and
T4c's eight: four `A_CONV=2` rungs, an NPOT reduction, three more activation
builds, the shortcut epilogue, the depthwise `pool`, `conv_post`'s epilogue
and the inverse transform. T6a added the attention and a `gelu_new` build;
T6b added exactly one, `kokoro_proj.comp` — a 256-to-**1** convolution, which
is the one output shape the ladder cannot express because a fragment tile is
16 columns wide.

**`g2p/`** — English text to kokoro's phonemes, in seven files.
`stress.go` (the stress rewrite rules, the vowel and consonant sets, `Context`
and its reverse walk), `lexicon.go` (both dictionaries, `IsKnown`, `lookup`,
`NNP`, the special cases, the three suffix rules and `Word`), `number.go`
(`num2words`' cardinal, ordinal, year and decimal spellings), `getnumber.go`
(misaki's eight-branch `get_number`, with the currency and ordinal rules) and
`subtoken.go` / `tokenize.go` (the subtoken scanner, the tokenizer, the
progressive-merge loop and `Phonemize`), `homograph.go` (twelve measured rules
where spacy's tagger mattered) and `espeak.go` / `fallback.go` (a cgo `dlopen`
binding to libespeak-ng, and misaki's 25-rule rewrite of its IPA). Nine tests,
each against its own dumped oracle.

**`reference/convert_misaki.py`, `reference/dump_g2p.py`** — the lexicon
export and the G2P oracle. The first writes `models/misaki/{us,gb}_{gold,
silver}.txt`; the second records a 24-sentence corpus token by token, with
each token's tag, its context and what `Lexicon.get_word` and `get_number`
each made of it *in isolation*, plus three tables — `num2words` over 10115
integers, `get_number` over a 1760-case cross product, and the subtoken regex
over 67 word shapes — and the coverage survey above.

**`cmd/tts`** — phonemes in, a WAV out, with the stage profile above.
`-gpu` runs the whole model on Vulkan except the embeddings, the duration
head and the excitation, each of which is on the host for a reason given in
T6d and T7; `-voice` picks
one of 54 or a mixture of them (T8), `-speed` divides the durations, `-noise <seed>` switches the
excitation noise on (off is the reproducible configuration; on is what an
utterance meant to be listened to wants), `-list` prints the voices.

## T1, in one page

`kokoro-v1_0.pth` is a pickle of five `state_dict`s, every key prefixed
`module.` from the `DataParallel` it was trained under, so nothing about the
model could be compared against anything until it was a mapping.
`convert_kokoro.py` writes `models/Kokoro-82M/model.safetensors` (459 tensors,
81.731 M params, 327 MB fp32) and `voices.safetensors` (54 voices, `[510,
256]`, under `voice.<name>`). Both go in the same directory because
`safetensors.OpenSet` globs `*.safetensors`: **one open gives Go the model and
the voices**, 513 tensors and 355 MB, cross-checked value for value.

`dump_kokoro.py` runs `KModel` on a fixed phoneme string — misaki's output for
"The quick brown fox…", recorded verbatim in the manifest so the dump
reproduces with no G2P installed — and walks it: ALBERT layer by layer, the
duration encoder block by block, the length regulator, F0/N, the text encoder,
every decoder block, the source module, both generator stages and the iSTFT.
63 tensors. The by-hand chain matches `forward_with_tokens` to **0**, and the
by-hand ALBERT layer and AdaIN block to 2.9e-6 and 7.2e-7.

Three findings, none of them in `config.json`:

**The vocoder is stochastic, and the noise is worth 13.8 dB.** `SineGen` draws
the eight harmonics' initial phases from `torch.rand` and adds Gaussian noise
to the excitation, so two runs of the reference differ. The dump zeroes every
draw — that deterministic run is what Go is checked against — and writes a
seeded run beside it: the difference is 0.00956 rms against 0.0469 rms of
signal. Loud enough that **Go needs its own excitation noise to sound right**,
so it is an explicit input to the vocoder rather than a residual to chase.

**The AdaIN `InstanceNorm1d` affine is identity.** All 70 are built
`affine=True` (an upstream ONNX workaround) with no weights in the checkpoint,
and `KModel` loads `strict=False` and logs at *debug* — so a genuinely missing
tensor would load as noise, silently. The key sets are diffed both ways: 140
of 688 tensors are uncovered and every one is an identity affine or a
`num_batches_tracked`. AdaIN is `(1 + gamma) * instance_norm(x) + beta`, and
the normalisation is over **time**, per channel.

**`weight_norm` folds at conversion**, 89 convolutions, agreeing with
`remove_weight_norm` to 1.19e-7 — the same check parakeet's BatchNorm fold got,
and why the safetensors has 32 K parameters fewer than the pickle.

## Kokoro — the shape of the port

**One alignment frame is exactly 600 samples = 25 ms at 24 kHz.** For the dump
run: 50 tokens → `pred_dur` sums to 130 frames → 2x through the predictor's
upsampling block to 260 F0/N frames → 2x again in `decode.3` → 10x and 6x
through the generator to 15600 STFT frames → hop 5 → 78000 samples, 3.250 s.
An off-by-one anywhere in that chain desynchronises everything after it, and
the chain is the first **data-dependent output length** in the engine.

**The length regulator is a gather, not a matmul.** torch builds an `[N, L]`
one-hot and does `d @ aln`; frame *f* just reads token `indices[f]`. Two
`[512, 50] x [50, 130]` GEMMs that do not need to exist.

**The five modules**, 82 M params: `bert` (ALBERT, 12 layers sharing *one*
3.5 M weight group, `layer_norm_eps` **1e-12** — not the 1e-5 everything else
in this repo uses — and `gelu_new` in the FFN), `bert_encoder` (one linear),
`text_encoder` (embedding, three weight-normed conv1d + LayerNorm + LeakyReLU,
one bidirectional LSTM), `predictor` (a duration encoder of three
LSTM/AdaLayerNorm pairs, a duration head, and F0/N stacks of AdaIN resblocks),
`decoder` (the iSTFTNet vocoder, 53 M of the 82 M).

**fp16 survey**: largest activation **314** (the F0 curve, in Hz), largest
weight **12.2**. Nothing on the path is near 65504. The two places to watch
are the generator's `exp()` on the spectrogram half of `conv_post` (argument
max 3.0 here) and the instance-norm variances.

## What cannot be reproduced, and how much it is worth

Three quantities in the vocoder are ill-conditioned *in the reference*, and
finding out which of them mattered was most of T3. They are the reason the
device is validated against the CPU path *and* against the dump separately.

**The phase accumulator.** Upstream integrates the excitation's phase in
radians and then multiplies by 300, so three seconds in it holds **1.3e5
radians, where one float32 ulp is 0.016 radians** — a 1% error in the sine.
`HarmonicSource` keeps the phase in cycles and wraps before the sine, so
nothing is ever large, which makes it *more* accurate than the dump and
therefore different from it by 0.3%. **fp16 cannot represent 1.3e5 at all**;
at 6e4 its ulp is 64 radians. This is why the excitation is still on the host,
and it is not a thing to fix.

**The phase spectrogram.** With the noise off, an unvoiced region's excitation
is a constant — tanh of the mixer's bias — and a constant's windowed spectrum
is exactly zero outside the three bins a Hann window occupies, so `atan2(0, 0)`
decides a quarter of the channels the network reads. Measured: **47704 of
171611 phases disagree** with the reference; 46957 still disagree modulo 2*pi
and the loudest of those sits **3593x below the spectrum's rms**, while the
other 747 are pure ±pi branch flips, the loudest 13x below. Every disagreement
is a phase of nothing — and it is still **18.2 dB** in the output, because
nothing trained the network to ignore those channels.

**`rand_ini` is dead code.** `SineGen` draws a random initial phase per
harmonic and adds it to sample 0 of the phase curve, which is then decimated
300:1 with a half-sample offset — output frame 0 reads samples 149 and 150,
and sample 0 is never read. Setting it to 0, 0.37 or 0.99 changes the
reference bit for bit not at all. So the only randomness that matters is the
additive Gaussian noise, which `-noise <seed>` provides.

So `TestVocoder` runs the chain **three ways**, and the spread is the finding:
from the F0 curve **18.2 dB**, from the reference's excitation **18.7 dB** (the
accumulator is worth almost nothing), from the reference's excitation
spectrogram **98.0 dB** (the port's own error). Without the middle run the
accumulator would have looked like the culprit.

**And this is how the device path is checked.** `TestGPUTail` compares the
Vulkan waveform against the CPU path (59.1 dB) *and* both of them against the
dump — where they agree to three digits at 18.2 dB. A tail that were wrong
would move the second number and nothing else could hide it, because 18.2 dB
is a floor set by the reference and not by any arithmetic here.

## T5 — grapheme to phoneme, done

`cmd/tts` takes text. The oracle is `misaki.en.G2P`, because it is what kokoro
was trained on; the full write-up is
[`t5-kokoro-g2p.md`](t5-kokoro-g2p.md) and the short version
is the table:

    the word path         204 corpus tokens, phonemes and rating     0 wrong
    num2words             10115 integers x 3 spellings, 11 decimals  0 wrong
    get_number            1760 cases over 45 digit strings           0 wrong
    subtokenize           67 word shapes                             0 wrong
    the espeak fallback   39 words, raw and rewritten                0 wrong
    the homograph tagger  1962 occurrences against spacy            88.4%  (baseline 83.8%)
    end to end, designed  24 sentences                          24 exact, 100% of words
    end to end, unseen    400 sentences of this repo's prose    68.8%, 92.4% of words

Three things from it are worth keeping in mind anywhere else in this engine.

**Test the oracle's decomposition, not just its output.** The dump records what
`get_word` and `get_number` each made of a token *in isolation*, with the
context replayed — misaki does not keep it — which turned an all-or-nothing
port into a pure function with 204 cases. One level down, dump the *cross
product* rather than the corpus: 21 number tokens cannot pin down eight
branches, and 1760 cases cost nothing.

**A table catches what reading does not.** The subtoken port looked correct and
split `3.50` into `3.5` and `0`, because a regex repetition matches a run's
last digit with both its optional parts empty and a hand loop does not unless
it tries the alternatives in the engine's order.

**Measure the rule, do not reason about it.** The homograph rules scored
*worse than no tagger at all* on the first attempt, and the fix — `that` is a
determiner when a boundary or a function word is on its left — beat the
syntactically obvious refinement by 8.5 points, because "that is" is
overwhelmingly a determiner in running prose. And a rule that helps one word is
not a rule until it is measured on the others: "a content word in front means a
tensed verb" is worth seven occurrences of `read` and would wreck `fragment`,
so it is restricted to the entries that carry a VBD key.

## T6 — the phoneme side, finished

**Three stages, 222 ms to 17, and the profile decided the order of all three
of them.** `speech-vertical.md` had carried "60% of the phoneme side is recurrences" since
T2. Timing the six stages put **ALBERT at 104 ms of 222** — 47% of the phoneme
side and 40% of a whole utterance — against 6.7 GFLOP of arithmetic, which is a
transformer this repository has had kernels for since stage 3. Then T6b split
`Prosody`'s remaining 67 ms and found **40 ms of AdaIN stacks** where this file
had written 33. Both corrections were three lines of `time.Now()`. **The first
commit of a stage should be the instrument, not the kernel** — and the
instrument has to be one level finer than the plan, because a stage's number
inferred by subtraction is not a measurement. T6c is the case where the
instrument agreed with the plan — 73 ms of 76 in the recurrences, exactly where
T2 had said — and it is worth noticing that this was only knowable the same
way.

**T6a put PL-BERT on Vulkan: 104 ms to 4 ms, 46.6x**, 0.204 ms of GPU time a
layer. Nothing about the model's behaviour moved: **none of the fifty durations
changed** (the largest unrounded drift is 0.005 of a frame) and the F0 curve is
2.2e-4 relative, so the alignment and everything downstream of it are
identical.

Almost all of it ran on kernels that already existed. The projections and the
feed-forward are `dit_gemm.comp`'s narrow-M rung — M is the token count, fifty,
which is the same short-M regime the decoder's ladder picked a 32-wide tile for
— the post-norms are parakeet's LayerNorm, which has a mean *and* an affine and
so is the one this model's 1e-12 norm needs, and the residual adds are
parakeet's. Two things are new: the attention, and the feed-forward's GELU as
another build of `kokoro_act.comp`.

Three things worth keeping:

  - **A bias is an extra column of K** (stage 9's trick, second outing). The
    fp16 A operand's row stride is padded anyway for §2.3, so column `width`
    of every row holds a 1 and the packed B holds the bias at that K index —
    and `y = Wx + b` comes out of a biasless GEMM. Six projections a layer,
    twelve layers, no epilogue and no dispatch. The pad is 64 rather than one
    tile because the K loop steps by BK.
  - **ALBERT's twelve layers are one weight group**, so 5.5 M parameters are
    staged once and the graph is the same block appended twelve times. The
    arenas are sized for `max_position_embeddings` rather than per utterance —
    25 MB at 512 tokens — so the object is built once and any clip runs on it.
  - **The attention is deliberately scalar.** Twelve heads over fifty tokens at
    a head width of 64 is 7.7 MFLOP a layer, 0.1% of the 558 a layer does. A
    WMMA kernel would spend most of its tiles on padding — fifty tokens is
    four 16-wide tiles with fourteen rows of nothing — and cost a pack pass
    per operand to save a tenth of a millisecond over the whole stack.

**T6b took the F0/N stacks from 40 ms to 1**, and the instrument moved the
number before the kernel did: `speech-vertical.md` had carried "~33 ms" for them,
inferred by subtraction, and splitting `Prosody` measured 24 ms of recurrence
against **40 ms of stacks** — a third of the whole phoneme side. They are six
`AdainResBlk1d` at 512 and 256 channels over 130 and 260 frames, which is *the
same block `GPUDecoder` already runs*, so the stage was a refactor and some
plumbing: `blockSet` lifted out of `gpudec.go` (which shrank from 729 lines to
322, with its numbers unmoved), `GPUProsody` wrapped around it, and **one new
shader in the whole stage**.

Three things from it are worth keeping:

  - **The third kind of shortcut is no shortcut.** Four of the six blocks keep
    their channel count, so upstream builds them with `conv1x1 = None`. On the
    device that is not a case to implement but three dispatches and an fp16
    sub-arena that do not happen — the epilogue reads the block's input where
    it already lies. It cost one thing: the epilogue had assumed its second
    operand carried the block's own row stride, which is true only when the
    block made it.
  - **A 1-wide output is a kernel, not a rung.** The projections are
    [260, 256] x [256, 1]. Every rung writes a 16-wide cooperative-matrix
    fragment, so the narrowest GEMM available is 16 columns of which 15 would
    be zero weight, *plus* a narrow into fp16 for an operand already in fp32.
    Reuse is the default and the fragment tile is a hard floor.
  - **The projection is on the device for the boundary, not the arithmetic.**
    It is 0.7% of the stacks' GPU time. What it buys is that the last block's
    [260, 256] activation never comes back: this device reads host-visible
    memory at 210 MB/s, so that tensor is 1.3 ms a curve against 1 KB and 5 µs
    for the curve itself.

**T6c took the six recurrences from 73 ms to 6**, and this time the plan was
right: the instrument measured **73 ms of a 76 ms phoneme side** in the six
LSTMs, 96%, with everything else on that side — ALBERT, the stacks, the
convolutions, the length regulator — sharing 3 ms between them.

**One algebraic move does most of it.** Torch's step is
`W_ih x_t + W_hh h_{t-1} + b`, and the first term depends on nothing the loop
produces. `P = X W_ih^T + b` is therefore computed for the whole sequence up
front, as **one GEMM with both directions' weights concatenated along N**, and
the loop keeps only `W_hh h`. The per-step B operand goes from 1.8 MB to 0.5 —
small enough to stay MALL-resident for every step of every LSTM in the model —
and the pass that would otherwise assemble `[x_t | h_{t-1}]` as an fp16 operand
stops existing.

**The recurrence is a static graph, and that is the difference from S8.** A
transducer decides what to do next from what it just emitted, so its loop
costs a host round trip a step. Nothing is decided here: the sequence length is
known before the first dispatch, so the whole thing is T+2 dispatches in one
command buffer with no fence in the middle, and one dispatch a step covers
both directions.

Three things from it are worth keeping anywhere else in this engine:

  - **The knob was not the one being turned.** A step is 0.5 MB of weights
    against 0.26 MFLOP — 500 bytes a FLOP — so it reads as a bandwidth problem,
    and the obvious fix was to spread it over more CUs. Two workgroups, eight,
    sixteen: **12.1 microseconds either way**. What it was short of was
    outstanding loads, and deepening the unroll from 4 taps to 32 took it to
    **3.7**. A recurrence has only 4H threads to issue from, so depth per
    thread is the only parallelism left.
  - **Measure the step on the critical path.** The same dispatch 130 times
    with barriers is 11.9 microseconds; without them it is 0.53, because they
    overlap and hit the same hot megabyte. The second number is a throughput
    figure for work that cannot be run in parallel, and it is not the step.
  - **One field beat four signatures.** `LSTM.GPU` hangs off the recurrence,
    not off the four objects that own one, because all four reach it through
    `Apply`.

**T6d closed it**: the whole phoneme side is one arena, two submits and two
readbacks, 8 ms of wall clock over **1.96 ms of GPU time**. T6c's recurrences
had been 6 ms of wall clock over 1.4 of GPU — the rest six readbacks and six
submits, because each one returned its output so the host could do the three
lines between it and the next. This was those three lines, and it needed one
new kernel: a gather.

Four things from it are worth keeping anywhere else in this engine:

  - **fp16 has exactly one place it cannot go in this model.** The duration
    head was on the device first — one narrow, one GEMM, 13 KB back instead of
    102 — and a duration moved by **0.29 of a frame**, where T6a's whole
    ALBERT port moved the largest of fifty by 0.005. Its logits have an rms of
    27, so a thousandth of relative error is two hundredths of a logit; a
    duration is the *sum of fifty sigmoids* of them, the terms near zero have
    a derivative of a quarter, and the total is then **rounded**. The head
    went back to the host in fp32 and the drift returned to 0.006. **A dot
    product whose output is rounded is not the same kind of tensor as one
    whose output is added** — and this is the model's only one.
  - **A constant input channel is part of the operand, not part of the data.**
    Five of the six recurrences take `[h | style]`, the same 128 numbers in
    every row of every utterance. `GPULSTM.SetTail` writes them once into the
    fp16 A operand, in columns `lstmPad` had already reserved, so torch's
    `Concat(h, s)` becomes nothing at all. Stage 9's bias trick is the special
    case where the constant is 1. It also **removed a kernel before it was
    written**: with the style out of the data, every normalisation runs over a
    512-wide row, which is the shape parakeet's LayerNorm already reads — so
    AdaLayerNorm is that kernel with `(1+gamma)` and `beta` computed on the
    host once an utterance, exactly as every AdaIN in the vocoder is.
  - **Read back the shorter tensor.** `t_en` can be expanded here and returned
    as [L, 512], or returned as [T, 512] and expanded on the host. The
    expansion is a gather either way; 102 KB is cheaper to read than 266. At
    210 MB/s the *direction* of a gather is a bandwidth decision.
  - **Two independent paths are one submit.** The text encoder depends on
    nothing the duration path produces, so its convolutions and its recurrence
    ride in the same command buffer and cost only their own dispatches.

The `parallelFor` lesson from T2 is worth repeating before anyone profiles it:
pushing one index per worker through a channel costs more than the work when a
recurrence calls it once per timestep. Contiguous chunks took the phoneme side
from 420 ms to 200 ms, and the obvious-looking refinement — skipping the
fan-out for small n — is wrong, because the short loops are short in indices,
not in work.

One thing still to settle. `Model.Style` indexes the voice pack by the
**character** count of the phoneme string minus whatever fell outside the
vocabulary, which is how `KPipeline` does it and is not the token count — so a
G2P that emits a different number of characters for the same utterance picks a
different style row.

## What is left on parakeet, in order of what it would buy

**S10 — the front end is 48% of the pipeline.** 21 ms of a few thousand
512-point FFTs in float64 on the host, untouched since S3 because it was 0.8%
of the clip. Two directions and they are not exclusive: a float32 radix-4 on
the host, or the STFT and the mel filterbank as two dispatches. The filterbank
is a `[T, 257] x [257, 128]` GEMM, which is the ladder's own shape; the FFT is
not, and is the interesting half — and kokoro's iSTFT is now a **worked
example of the inverse**, one thread per output sample with a twiddle table,
which the forward direction can mirror.

**The submit and the fence are 38% of an emission.** 40 µs of the 105 µs the
decode loop spends per token is the round trip, not the work. Two things would
remove it — a **persistent kernel** with the loop's control flow on the device
(which is also what `api/transcription.go`'s `transcript.text.delta` would
want), and **speculating on blanks**, since during a run of blanks the
prediction state does not change and the joints for consecutive frames are
independent, i.e. one GEMM at M = 16 for the price of the M = 1 one. This clip
has only 8 blanks in 46 emissions; a clip with silence in it has many more.

**S9 — long clips.** Full attention over the clip means a chunk boundary
changes every frame's hidden state, so chunking belongs in the design rather
than after it. The ladder's 1024-frame row (111 ms against 59 at 768) is the
quadratic term arriving: at 82 s the position projection is 2T-1 = 2047 rows
and the score matrices are 1024x1024 per head. It would also bound the
subsampling arena, whose largest tensor is `[4T, 64, 256]` fp32 — 36 MB at 138
frames and 268 MB at 1024. Narrowing that to fp16 halves it *and* halves the
two kernels that are 80% of the subsampling stack, both bandwidth-bound; it
has not been done because the whole stack is 1 ms.

**Smaller, and measured.** `UploadMel` is 1 ms, almost all of it computing
2T-1 rows of sinusoidal position embeddings in float64 on the host — they
depend on nothing but T and could be cached. The encoder's eight per-head
position-score GEMMs are the same shape at different offsets, which is §3.5's
grouped GEMM.

## T7 — the excitation, done

With T6 closed the vocoder was **82% of an utterance** and the largest single
stage in the whole model was the **excitation at 14 ms** — 32% of a 44 ms
utterance, and nearly twice the whole phoneme side. It is now **0.2 ms**, of
which 23 microseconds is GPU time and the rest is one submit and one readback,
and the utterance is **31 ms — 105x real time** (five runs: 30, 31, 31, 32,
33 ms).

**The first thing measured was what the 14 ms actually was, and it was not
what this file said.** Half of it is not the sine bank at all:

    Source.Apply    6.4 ms     upsample 0.14, frac 0.75, cumsum 0.008,
                               interpolate 0.70, sin 4.2, mix 0.6
    Harmonic        8.2 ms     15601 frames of a 20-point DFT, 6.3;
                               171611 hypot and atan2, 1.5

So T7 is two kernels, not one: `kokoro_source.comp` for the sine bank and
`kokoro_srcstft.comp` for the forward transform, which is the mirror of the
`kokoro_istft.comp` T4c already had. One submit, one download, and the
waveform between them never crosses the bus.

### The phase question, answered

T3 said this could not move: upstream integrates the phase in radians and
multiplies by 300, so three seconds holds 1.3e5 radians where one float32 ulp
is 0.016. The answer is that **the accumulator never had to move**. The
cumulative sum is 260 frames by 9 harmonics and takes **8 microseconds** in
float64 on the host — 0.06% of the stage — so it stays there, and what goes
to the device is the integral already wrapped into [0, 1) together with its
forward difference. The largest number any thread evaluates is one frame's
worth of phase, at most 301, where one ulp is 3e-5 cycles rather than 0.016
radians.

**And three of the reference's stages turned out to be the identity.** The F0
curve is upsampled 300:1, converted to cycles per sample, and decimated back
down — and the decimation reads coordinate 300t + 149.5, so both samples it
averages lie in F0 frame t and nearest upsampling made them equal. Half of x
plus half of x is x. So the whole round trip is `frac(f0[t]*(d+1)/sr)` on 260
frames instead of 702000, the 5.6 MB float64 intermediate is never formed, and
`TestT7DecimationIsIdentity` measured the gap at exactly zero before anything
was built on it.

The result: the device waveform agrees with the float64 host to **1.6e-7 rms,
3.5e-6 worst**, and both land at 50.7 dB against the reference dump. The
wrapped form is float32-safe, and that is now a measurement rather than an
argument.

### The phase that is not a question but a lottery

The magnitude spectrum agrees to 7.3e-6 and the phase, wherever the magnitude
defines it, to 4.9e-6 in the complex value — float32 over twenty windowed
terms, which is what it should be. But the *waveform* through the rest of the
vocoder lands at **13.8 dB against the dump where the host path lands at
18.2**, and that number took most of this stage's work to understand.

It is not an error in either half. Decomposed four ways:

    host src   + host transform      0.123   (18.2 dB)
    host src   + device transform    0.110   (19.2 dB)
    device src + host transform      0.118   (18.6 dB)
    device src + device transform    0.203   (13.8 dB)
    reference src + host transform   0.116   (18.7 dB)
    reference src + device transform 0.092   (20.8 dB)

Each half is fine on its own and the device transform is the *better* of the
two given identical input — which is what one would expect, since torch.stft
runs on the float32 tensor and the host's float64 is the outlier. Only the
combination is bad.

The reason is the one `Generator.Harmonic` already names. With the noise off
the excitation is a **constant** wherever the signal is unvoiced — exactly
constant, 26100 samples of it, 0 of which differ — and a constant's windowed
spectrum is analytically zero outside the three bins a Hann window occupies.
So 27% of this phase spectrum is the angle of a rounding residual, one
arbitrary value per bin repeated over thousands of frames, and the network
consumes it as an ordinary number.

That makes the waveform's agreement with the dump a **draw**, and the draw was
measured rather than assumed. Walking the unvoiced constant by five ulps under
the device's own transform moves it over **0.097 to 0.203 — 13.8 to 20.2 dB**,
and the value that ships is the unluckiest of the eleven. Drawing one uniform
angle per bin instead spans 16.3 to 21.8 dB over eight seeds, with the host's
own float64 sitting at 18.2 in the middle of it.

Two null models were wrong before the right one was found, and both are worth
recording. Perturbing every sample of the host excitation by 1e-7 makes the
number *better* (0.109 to 0.114, tightly clustered) — because independent
noise breaks the constancy and so defines the phase. Walking the constant
under the *host* transform barely moves it either (0.118 to 0.130). It is the
constant and the transform's precision together that select the ticket.

**So the bound in the tests is the lottery's width and not an arithmetic
tolerance**, and everything that is defined is bounded separately and tightly:
the waveform at 1e-4 absolute, the magnitude at 1e-4, the phase at 1e-5 in the
complex value. `TestGPUTail` and `TestGPUVocoder` now run their
device-against-CPU comparison with the host excitation on both sides, because
what they are about is the blocks and this would swamp it.

**With the noise on — the configuration an utterance meant to be listened to
uses — the question does not arise**: 2 bins of 171611 fall below 1e-6 instead
of 47075, and the phase is defined everywhere.

### The noise, and what it is not

The noise is drawn on the device from a counter-based hash of (seed, sample,
harmonic) rather than uploaded. Upstream adds a Gaussian to every one of the
nine sinusoids, so an utterance needs 702000 draws — 7 ms of Go's
`NormFloat64`, which is half of what this whole stage cost *before* it moved,
and 2.8 MB across the bus.

What that costs is the sequence: **`-noise n` does not give sample-identical
audio on the two paths.** It gives the same distribution, the same rule — loud
where the signal is unvoiced and quiet where it is not — and the same audio
for the same seed on the same path. `TestGPUSourceNoise` checks it by that
rule, since the rule is the only thing about it that is defined.

### The arena is HOST_CACHED

The spectrogram is 1.4 MB and comes back every utterance. kokoro's other
arenas use the write-combined type `NewBuffer` prefers, which this host reads
at **0.18 GB/s** — 7.7 ms for this download, which would have been the entire
stage. `vk.NewHostCachedBuffer` reads at 25 and the readback becomes 74
microseconds. It is the first place in kokoro where that distinction has
mattered; the other readbacks are small enough not to care.

## What is left after T7

**The generator is 12 ms and the tail 8**, which T4 left measured and which are
the only parts of this model that were ever arithmetic-bound. Together they
are now 65% of an utterance.

**The phoneme side's 8 ms, none of it arithmetic** — ALBERT's [T, 768]
readback and its host-side embedding stack, the two readbacks T6d could not
remove, and four submits. Closing any of it means moving the *vocoder's* input
boundary, which is one change rather than four.

**The excitation's remaining 0.2 ms is 88% submit and download**, not
arithmetic: 23 microseconds of GPU against 155 of submit and 70 of readback.
The way to close it is to stop reading the spectrogram back at all — the two
noise convolutions that consume it are host-side, and moving them onto the
device would leave nothing of this stage on the bus.

## T8 — voice blending, done

A client asked `/v1/audio/speech` for
`"voice": "af_alloy,af_bella,af_heart"` and got a 400:

    api: 400: no voice "af_alloy,af_bella,af_heart"; this checkpoint has 54: unsupported request

It now speaks, and so does `af_bella:3,af_sky:1`. `kokoro/blend.go` parses the
specification, `Model.Style` resolves it, and `CheckVoice` is what
`backend/tts.go` validates `-voice` and a request with. The measurement is
`TestBlendAgainstUpstream`: **every one of the 510 rows of six mixed packs,
against `KPipeline.load_voice` itself**, dumped by `reference/dump_blend.py`.

**The spelling was not ours to choose, and finding that out was the stage.**
This file had said "OpenAI's API has no notion of mixing voices, so nothing
constrains this except that a single name must keep meaning what it means
now" — and that was true about OpenAI and wrong about the constraint, because
hexgrad's own `KPipeline.load_voice` splits a voice on commas and returns
`torch.mean(torch.stack(packs), dim=0)`. So `af_bella,af_sky` already means
the equal mean of two packs everywhere else kokoro runs; a server free to
choose would have been free to answer that request with a *different voice*
than every other client gives it. The weight — `af_bella:3,af_sky:1` — is the
part upstream has no spelling for, and is where the decisions actually were.

Four things worth keeping:

  - **Look for the upstream spelling before designing one.** The oracle for
    this stage is `load_voice` run over the local `.pt` files, offline: its
    `load_single_voice` takes a path when the name ends in `.pt`, and
    `KPipeline(lang_code='a', model=False)` builds no model and downloads
    nothing. That is upstream's own code path rather than a reimplementation
    of the line quoted from it, which is the difference between checking an
    implementation and checking a reading.
  - **Normalise the weights, and require all or none.** `3,1` and
    `0.75,0.25` are the same mix, so bare weights need not add up — an error
    class removed rather than added. But `af_bella:0.7,af_sky` is refused,
    because it reads as either 0.3 for the rest or one share before
    normalising, and a mix that silently picked one would still sound like a
    voice. That is this model's failure mode in one sentence, and it is the
    same argument that put the phoneme count in `Style`'s doc comment.
  - **The mix is per row, and the whole pack is what checks it.** A weighted
    mean is linear, so the row of the mean is the mean of the rows: 256
    numbers rather than 130560. The dump writes the whole `[510, 256]` packs
    anyway and the test walks every row, because "linear" is a claim that
    holds at every length or is a bug a single-length check would not see.
  - **Bound the gap in ulps of the *terms*, not of the answer.** torch sums
    the packs in float32 and divides; this accumulates in float64 against
    weights normalised once. The two-way and four-way equal mixes agree **bit
    for bit** (the divisor is a power of two); the three-way and the weighted
    ones differ in 21–42% of channels, at up to two thirds of the N-ulp bound
    for a float32 sum. A per-channel ulp bound was tried first and read
    **2.18e4 ulp**, because three style channels of order 1 cancel to 1e-11
    and the error of the sum is set by the size of its terms, not of its
    result.

And a blend is **not a crossfade of two utterances**: half the style vector
conditions the predictor, so the durations move with it. The dump sentence is
3.575 s in `af_bella`, 3.425 s in `af_sky` and **3.450 s in their equal mean**,
which is a different alignment and not a mix of two waveforms. The path from a
style vector to samples is unchanged and already measured against the
reference at 18.2 dB, so what T8 had to check was the vector — and it is
checked against upstream exactly, at every length.

## T9 — one staging for the life of the server, done

`POST /v1/audio/speech` answered in **3.285 s** for 3.35 s of audio. The model
had been on the device since T4 and the utterance itself was 48 ms; the
endpoint was not running it. `-tts-gpu` defaulted **off**, and the reason it
defaulted off was this file's own note: the device path "stages per request".

It does not any more. The same request is **31 ms**, and the audio is byte for
byte what the old path produced.

    POST /v1/audio/speech, af_heart, three runs each

    utterance            audio     CPU     T4-T8    T9      device
    "Hello there, ..."    3.35 s   3.285   0.302   0.031    0.025
    "A guy is driving     6.45 s   5.430   0.550   0.059    0.048
     around the
     backwoods ..."

**The 550 ms was not the model.** It decomposes exactly, and only the last
row of it was ever arithmetic:

    host prosody, to learn the frame count   411 ms
    AttachGPU, sized by that count            88 ms
    the whole utterance on the device         48 ms
                                             547 ms, and the endpoint measured 550

Kokoro's arenas were sized for one utterance's alignment frames. The
*durations* decide that count, so the phoneme side had to run on the **host**
— 411 ms of ALBERT and six recurrences, the very work T6 moved — before the
device could be built to run the same phoneme side in 13. Then the arenas were
torn down again, because the next request would be a different length. Every
request paid 499 ms of protocol to save 502 ms of arithmetic.

**The frame count is now a ceiling.** `Model.AttachGPU` stages for
`-tts-frames` (1000 frames, 25 s of speech, by default), every shorter
utterance is a prefix of the same arenas, and what a request costs is
`Vocoder.SetFrames` — a walk down the chain AttachGPU already walks, writing a
frame count into a push constant. The server stages once at startup and the
host prosody pass is gone entirely: `prosodyGPU` computes the durations on the
device and nothing needs them earlier.

**A ceiling costs memory and nothing else.** The arenas are ~150 MB plus
0.7 MB a frame — 240 MB at 130 frames, 848 MB at 1000, 3.0 GB at 4000 — and
staging is 87 ms at the bottom of that range and 230 ms at the top. Per
utterance the ceiling is free, because every dispatch extent is the
utterance's: the Montana sentence is **48 ms staged for its own 258 frames and
48 ms staged for 1000** (`go run ./cmd/tts -gpu -frames 1000`).

### Padding to the bucket would have been wrong

The note this stage was written from proposed sizing "for a frame count above
the utterance's and zero-pad the alignment up to it, trimming the tail by
`Prosody.Samples`". That is the obvious way to do it and it would have
produced a wrong utterance, quietly.

**AdaIN normalises over time.** Every one of the ~40 blocks between the
durations and the waveform subtracts a per-channel mean and divides by a
per-channel standard deviation taken over *the whole alignment* — that is what
`kokoro_stats.comp` is and why it exists. Pad 130 frames of speech out to 1000
frames of silence and every statistic in the model moves, so every sample
changes, including the ones inside the utterance. The trimmed output would
have been a plausible waveform of the right length that was not the utterance
anybody asked for — and no length check or shape assertion would have caught
it.

So the frame count is a *runtime* quantity rather than padding: `run` beside
`frames` in `GPUBlocks`, `GPUDecoder` and `GPUProsody`, and every dispatch
extent, reduction bound and readback length taken from it. The reductions were
already clamped (`min(t0 + aux0, pc.tokens)`), which is why this was a change
of arithmetic in the *host* and of nothing in the shaders.

### The one thing bucketing can break

A shorter utterance leaves the rows past its end holding a longer one's
activations, and **kokoro's convolutions are branchless because their padding
is in the data**: a k-tap filter at dilation d reads (k-1)d/2 rows either side
of its output and those rows are a zero border rather than a bounds check
(`convBorder`, 32 rows, which covers k=11 at d=5). The border at the start of
an arena is written once at staging. The border at the *end* moves with the
utterance.

Restoring it is 8 KB a block from the host, in `zeroBorders` and
`GPUBlocks.zeroBorder`, and it is the whole of what a shorter utterance costs.
Getting it right took two corrections, both off-by-one and both found by the
same test:

  - **The upsampler's zero row is T_in, not T_in+1.** Its GEMM runs T_in+1
    rows because the last output row reads input rows T_in-1 and T_in, while
    the rectifier that fills the arena writes only T_in of them. Row T_in is
    padding that had always been zero because nothing ever wrote it. Getting
    this wrong moved the *whole* waveform by 2.5% relative, not the five
    frames at the seam — because five wrong frames enter an AdaIN and the
    statistics carry them across every other frame. A 4% error at the first
    sample of an utterance whose seam is 2600 frames away is what a
    time-normalised model does with a local mistake.
  - **The text encoder's border was already wrong, and had been since T6d.**
    Its arena is sized by the position embedding's 512 tokens rather than by
    the utterance — it is the one thing in this package that was bucketed
    before T9 — and its convolutions are five taps wide, so they read two rows
    past the last token. Nothing could observe it: no attachment had ever
    survived two utterances. The first thing bucketing does is make one
    survive several, and the second utterance of a pair came out wrong.

### The bound is equality

`TestGPUBucketedFrames` and `TestGPUBucketedSpeak` do not compare against a
tolerance. Two attachments of different sizes issue **the same dispatches with
the same push constants over the same weights**; all that differs is where the
arenas sit. Anything that reads past the live rows is a wrong number and not a
rounding error, so the test requires the waveforms to agree **sample for
sample**, and it runs the utterances in the order a server sees them — short,
long, short — so that the short one runs against arenas full of somebody
else's activations. The 550-to-59 ms result above is the same claim from the
outside: the redeployed endpoint's bytes `cmp` clean against the old path's.

### And the voice is not baked in either

`AttachGPU` conditions every AdaIN on a style vector, which is why the server
used to restage when the voice changed. It does not need to: `Model.SetVoice`
rewrites gamma and beta in the arenas that are already there, which is two
projections of one vector per block on the host. `TestGPUSetVoice` holds it to
the same bound — an utterance in a re-conditioned voice is the one a fresh
attachment in that voice gives, sample for sample.

The style row is indexed by the **phoneme count** as well as by the voice (see
`Model.Style`), so it changes with the length of the utterance and not only
with the name in the request; `backend.TTS.attach` compares the vectors rather
than the names, and re-conditions when either moves.

### What is left

**An utterance past the ceiling restages once and keeps the larger arenas.**
`kokoro.FramesOverflowError` carries the frame count the durations came to, so
`backend.TTS.synthesize` rounds it up to the next 256 frames, restages, and
runs — 1665 frames became arenas for 1792 in 151 ms, and the next request of
that length found them already there. It is a rare path: the voice packs have
**510 style rows**, so 510 phonemes is the model's own ceiling on an utterance
and only `speed < 1` stretches one past 1000 frames.

**What did not change is the 25 ms.** The endpoint is now the model plus
~5 ms, and the model is where T4 and T6 left it — the generator at 12 ms, the
tail at 8, the phoneme side at 8 with 1.96 ms of GPU inside it. The next
millisecond on this vertical is still the one "What is left after T7" names:
the noise convolutions are on the host, so the excitation's spectrogram
crosses the bus for no other reason.

## R1 — the round trip, and what it says to do next

`go run ./cmd/roundtrip` is the loop both verticals have to close and neither
of them could measure on its own: text → `POST /v1/audio/speech` → resample →
`POST /v1/audio/transcriptions` → text, with the text that comes back checked
against the text that went in. It is a client — it imports `audio` and
`net/http` and nothing else from here — so what it reports is what a caller
waits on rather than what a stage table says.

    go run ./cmd/serve -tts -stt
    go run ./cmd/roundtrip -reps 2 -csv results/roundtrip.csv

**Six prose cases, one word to a paragraph, come back exactly.** Two runs
agree at 61.6x and 62.6x real time over 51.6 s of audio. Each request is
timed from the last byte written to the first byte read, which is the closest
a client gets to "what the model cost" — and it is within a millisecond of
what the handlers log for `Speak` and `Transcribe` themselves (221 against
222, 73 against 74), so **neither endpoint has protocol left in it**: HTTP,
JSON, multipart and the WAV container together are 0.2% of the loop.

| | total | ms/s of audio | share |
|---|---|---|---|
| speech, server | 530 ms | 10.3 | **63.2%** |
| transcribe, server | 236 ms | 4.6 | 28.1% |
| resample, client | 70 ms | 1.4 | 8.4% |
| http and encoding | 2 ms | 0.0 | 0.2% |

The per-case rows are what the totals hide, and they point in opposite
directions:

    ms per second of audio     1.35s   3.25s   5.05s   8.80s  13.60s  19.52s
    speech, server              10.8     8.8     8.3     9.1    10.6    11.3
    transcribe, server          11.5     6.6     5.8     4.4     4.1     3.9
    resample, client             1.3     1.5     1.3     1.5     1.3     1.3

**Parakeet amortises and kokoro does not.** Transcription's fixed 11 ms falls
away as the clip lengthens and it ends at 257x real time; synthesis bottoms
out at 8.3 ms/s around five seconds and then **gets 36% worse per second of
audio** by nineteen. A least-squares fit says the same thing in one line: the
speech leg's intercept is *negative* (-10.7 ms, 11.5 ms/s, r²=0.99), which is
a curve and not a line.

`cmd/tts -gpu -frames 1000` localises it. The same two utterances, 50 tokens
and 321:

    stage        3.25 s   19.52 s   growth
    bert            4ms      75ms     18.8x     ← 6.4x the tokens
    phoneme side    8ms      96ms     12.0x
    generator      12ms      64ms      5.3x
    tail            8ms      59ms      7.4x
    vocoder        21ms     128ms      6.1x
    total          29ms     225ms      7.8x

The vocoder scales with its frames, as it has to. **PL-BERT does not** — 6.4x
the tokens for 18.8x the time — so T6a's 46.6x was measured at the one length
where the phoneme side was cheap, and at paragraph length the phoneme side is
**43% of an utterance rather than 26%**. That is where the next millisecond
on this vertical is, and it displaces "What is left after T7": the excitation
is 1 ms of a 225 ms utterance.

### It is the attention, and the kernel says so itself

`TestGPUAlbertScaling` profiles one ALBERT layer per dispatch over six
sequence lengths. Microseconds a layer, on GPU timestamps:

    us a layer           50      100      200      321      410      510
    q / k / v            34/14/9  11/11/11 13/13/13 17/17/17 18/18/18 22/22/21
    attention            44      313     1786     5385     9844    16029
    dense                 9       10       13       17       18       22
    ffn                  14       17       23       34       41       60
    ffn out              22       23       31       39       42       47
    layer               164      412     1915     5560    10039    16266
    12 layers, ms       2.0      4.9     23.0     66.7    120.5    195.2

Every projection is **flat per token** — 17 us at 321 against 14 at 50, which
is a GEMM doing T times one row of work — so the tile is not the problem and
the ladder needs nothing. Attention is **0.87 us a token at 50 and 31.43 at
510**: 364x the time for 10.2x the sequence, which is O(T^2.5) rather than the
O(T²) the arithmetic asks for. It goes from **27% of a layer to 98.5%**.

`shaders/kokoro_bert_attn.comp` documents the decision that became wrong:

> It is deliberately scalar rather than a matrix-core flash kernel, and the
> arithmetic says why: twelve heads over fifty tokens at a head width of 64 is
> **7.7 MFLOP a layer** … A WMMA attention would spend most of its tiles on
> padding — a 50-token sequence is four 16-wide tiles with 14 rows of nothing.

That is sound at fifty tokens and false at three hundred. The extra half-power
above quadratic is in the kernel's shape rather than its work: one workgroup
per (head, query), and its weighted sum runs `if (t < hd)` — **64 of 256 lanes
busy, walking every key serially** — while the score loop re-reads the query
row from global memory once per key. The model's own ceiling is 510 phonemes
(the voice packs have 510 style rows), and there PL-BERT alone is **195 ms**.

So, in order of what it would buy a round trip:

1. ~~**PL-BERT's attention.**~~ **Done, T10** — 64.6 ms of a 225 ms paragraph
   and 195 ms at the model's ceiling, against 9.6 GFLOP that stage 3c's
   matrix-core kernel does in microseconds. It was a build of two existing
   shaders at `-DHEAD_DIM=64`, and the gate was not the tensor bound but
   `TestGPUAlbertDurations`: durations are integers and did not move.
2. **The host embedding stack, now that the attention is gone.** `bert` is 12
   ms of the paragraph and only **2.5 of it is on the device** —
   `GPUAlbert.Apply` says the embeddings are "a table lookup, two adds and a
   128->768 projection over fifty rows, and none of that is worth a dispatch",
   which is the same sentence the attention kernel had and stale for the same
   reason. At 321 rows it is 79% of what PL-BERT costs.
3. **The phoneme side's other 21 ms** — the six recurrences are one dispatch
   a timestep (T6c), and a timestep is a phoneme.
4. **A 16 kHz path out of the vocoder.** The client resample is 9.7% of the
   loop and pure waste in a pipeline whose consumer is a 16 kHz model: it is
   thirty times what HTTP costs. `audio.Resample` is the filter; the question
   is whether the iSTFT head can be asked for the lower rate directly.
5. **Nothing on parakeet.** S10's front end is 48% of *its* pipeline and
   still only 33% of the loop, and it is the leg that already amortises.

**What T10 did not touch is a short utterance**, and it was never going to.
At fifty tokens PL-BERT is 2 ms of a 29 ms utterance, so "Hello." is 31 ms
before and after. The 1.4x is on a paragraph, the 56x is on a layer at the
ceiling, and what is actually gone is a term that got worse the longer anyone
dictated.

What the benchmark will not do is rule on **text normalisation**, and
`-stress` is why: kokoro says "one thousand and twenty four" for 1,024 and
parakeet writes `1,024` back, which is two correct components disagreeing
about spelling. Those three cases are measured and never counted as failures.

## T10 — PL-BERT's attention on the matrix cores, done

R1 said synthesis got 36% worse per second of audio as an utterance
lengthened, and `TestGPUAlbertScaling` said all of it was one dispatch. The
fix is not a new kernel. `shaders/dit_attention_wmma.comp` is stage 3c's
flash attention and already carries `HEAD_DIM` as a compile-time define, so
ALBERT's 768-over-12 head width is a *build* of it — `-DHEAD_DIM=64
-DQT=1 -DKTIL=4 -DWAVE=32`, with `CAUSAL`, `GQA`, `REL_BIAS` and `OUT_F16`
all zero — plus the same build of `dit_pack_f16.comp` to lay q, k and v out as
16x16 fp16 fragment tiles. Two `//go:generate` lines and no new GLSL.

Microseconds per ALBERT layer, per dispatch, on GPU timestamps:

    us a layer          50      100      200      321      410      510
    scalar attention    43      320     1774     5394    10052    15976
    wmma attention       3        6       11       25       29       38
    + three packs        6        6        7       10       12       15
    scalar, layer      161      420     1903     5569    10247    16218
    wmma, layer         81      105      148      206      231      288
    speedup, layer    2.00x    4.01x   12.88x   27.03x   44.44x   56.27x

    PL-BERT, 12 layers, ms
    scalar             1.9      5.0     22.8     66.8    123.0    194.6
    wmma               1.0      1.3      1.8      2.5      2.8      3.5

**The kernel that was quadratic now amortises.** Attention was 0.87 us a token
at fifty and 31.33 at five hundred and ten — 36x the cost per token for 10.2x
the sequence. On the matrix cores it is **0.19 us a token at fifty and 0.10 at
510**, packs included: it gets *cheaper* per token, because a flash kernel's
per-query-tile cost is fixed and the tile is 16 rows whether the utterance
fills it or not. That is also why there is no crossover and no fallback — at
the reference fifty tokens, where the tiles are mostly padding and the three
packs buy nothing, it is still 2.0x faster a layer.

What it cost: 2.4 MB of fp16 arena for three per-head plane sets at the 512
ceiling, and three pack dispatches a layer at 3-5 us each. `ScalarAttn` keeps
the old kernel as the oracle.

### The numerics did not move, and that is the whole gate

fp16 operands with fp32 accumulators is what the matrix cores implement, so
the scores are the exposure. They are also bounded: the running max is folded
in before every exponential and P is at most 1 by construction.

The measurements, none of which changed from the scalar path:

| | |
|---|---|
| Layer by layer against the CPU reference | **4.2e-4** at layer 0 to **2.4e-3** at layer 11, bound 6e-3 |
| The whole encoder, `Apply` | **2.4e-3** — the same figure the scalar kernel gave |
| **Durations** | **0 of 50 differ**; largest unrounded drift **0.0048 frames** |
| The F0 curve the durations feed | 2.6e-4 |

The durations are the one that matters and the reason this stage was safe to
do: they are a sum of sigmoids rounded to integers, they decide the length of
every phoneme, and a single flipped rounding shifts the whole waveform after
it — so no sample-wise comparison downstream would mean anything. 60 kokoro
tests pass, including T3's reference waveform.

### End to end

    "Speech synthesis and speech recognition are ..." (319 phonemes, 19.525 s)

    stage           T9      T10
    bert            75ms    12ms     2.5 ms of it on the device
    phonemes        96ms    33ms     43% of an utterance -> 20%
    generator       64ms    66ms
    tail            59ms    57ms
    vocoder        128ms   129ms     57% -> 80%
    total          225ms   162ms     86.9x -> 120.5x real time

The vocoder is the majority again, which is where T4 left it and where a
fixed-cost-per-frame stage belongs. And the round trip, two runs each:

| | T9 | T10 |
|---|---|---|
| The loop over 51.6 s of audio | 61.6x / 62.6x | **70.1x / 70.1x** |
| Speech leg, ms per second of audio | 10.3 | **8.2** |
| … at 19.5 s specifically | 11.3 | **8.1** |
| Fit against audio length | -10.5 ms + 11.5 ms/s | **+2.7 ms + 7.9 ms/s** |

A negative intercept is a curve pretending to be a line. The intercept is now
positive and small, the slope is flat from five seconds to nineteen, and
`r²` is 0.999 — which is the actual result of this stage: not the 1.4x, but
that there is no longer a length at which synthesis gets worse.

## W1 — the Wyoming door, done

The two verticals now answer Home Assistant directly, on a TCP port beside
the HTTP one and out of the same process:

    go run ./cmd/serve -tts -stt -wyoming 0.0.0.0:10300

    Settings -> Devices & services -> Add integration -> Wyoming Protocol
    -> the host, port 10300

One listener carries both services, because a Wyoming client asks what is
behind a port rather than inferring it from the number: `describe` is
answered with one `asr` program and one `tts` program, and Home Assistant
makes a speech-to-text entity and a text-to-speech entity out of the single
entry. The protocol itself is a newline-terminated JSON header, an optional
JSON data block and an optional binary payload, framed by lengths in the
header because the payload is raw PCM and may contain a newline.

**It is a door, not a second server.** `wyoming.Server` holds the same
`api.SpeechBackend` and `api.TranscriptionBackend` the HTTP handlers do, so
there is one staging of kokoro, one residency of parakeet and one GPU queue;
two clients on the two protocols are two goroutines serialising at the same
place two HTTP requests would. The check that this is true is that the same
text and voice through both doors is the same waveform:

    wyoming: 24000 Hz, 78000 frames   http: 24000 Hz, 78000 frames
    byte for byte the same waveform

Driven by upstream's own Python client rather than by our own reader, with
the two conversations written the way `homeassistant/components/wyoming`
writes them:

    describe    1 asr, 1 tts, 28 voices, 25 asr languages
    synthesize  44 characters -> 77 chunks, 3.25 s of audio at 24000 Hz, 35 ms
    transcribe  testdata/jfk.wav, 11.00 s -> 108 characters, 46 ms, 236.9x
    the loop    speech at 24 kHz -> resampled -> "The quick brown fox jumps
                over the lazy dog." exactly, 140.4x

**28 voices and not 54**, because the front end is the limit rather than the
checkpoint: misaki's English G2P is the one that is ported (T5), and there is
no phoneme field in this protocol to route around it with, so only the packs
whose language can be pronounced are advertised. `-wyoming-all-voices` offers
the other 26 to a deployment that has a reason to want them. Each is labelled
for a menu — `af_heart` is "Heart (American English, female)" — while the id
Home Assistant sends back is the pack's own filename, so a blend still
reaches the backend by being spelled into the voice field.

**Neither streaming direction is implemented, and both are measurements.**
`supports_synthesize_streaming` buys the gap between the first sentence of an
answer and the last; T10 put a nineteen-second utterance at 162 ms, which is
less than a satellite would take to play the first sentence of a streamed
one — and the durations for the whole utterance are decided at once, so the
pieces could not agree about prosody they had not seen. The transcript side
is 257x real time and the same argument applies. What is left open is what
another program owns: wake words, intent handling, satellite control, all
reported as empty lists rather than claimed.

**The conversions happen at the door this time.** `backend.STT` refuses a
clip at the wrong rate and always has — there the caller chose a file and can
convert it — but here the caller is a microphone and there is no conversation
to have, so this side reads 8-, 16- and 32-bit PCM at any rate and any
channel count and hands over the 16 kHz mono parakeet reads, through
`audio.Resample` and not an interpolation. The one thing it will not convert
is a stream that changes format halfway through: two rates concatenated is a
clip whose timeline is wrong in a way the transcript would not show.

**There is no authentication and there cannot be.** Wyoming has nowhere to
put a credential, so `-token` does not reach this listener and the startup
line says so. That is why the flag takes an address instead of a boolean —
which interface it binds is the whole of the control, and it should be
something somebody chose.
