# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-19, T8)**: **both verticals are on the device, text-to-speech
takes text, and an utterance is 74.2x real time.**
Speech-to-text transcribes an 11 s clip in 43 ms, 257x real time, whole model
resident, S1–S8 done. **Text-to-speech is 44 ms for 3.25 s of audio.** T6 is
finished: ALBERT 104 ms to 4 (T6a), the F0/N AdaIN stacks 40 to 1 (T6b), the
six bidirectional LSTMs 73 to 6 (T6c) and the chain between them 17 to 8
(T6d) — so **the phoneme side went from 222 ms to 8 and is now 18% of an
utterance, over 1.96 ms of GPU time**. The other 82% is the vocoder, whose
36 ms T4c already cut from 3626, and whose largest remaining piece is **14 ms
of float64 phase accumulation on the host** that T3 put there on purpose.
T8 closed the last open feature: a voice may name a **mixture** of packs, in
upstream's own spelling.

`go run ./cmd/tts -gpu -text 'Hello there.'` speaks, with no Python on the
path. T5 opened with a measurement instead of code — **91% of running-text
tokens are a dictionary lookup and a part-of-speech tagger decides 1.74% of
them** — and ended with the whole of misaki's English G2P ported: five
components exact against their own dumped oracles, twelve measured rules where
spacy's tagger mattered, and a cgo binding to libespeak-ng for the rest.
**The designed corpus is exact, 24 of 24; on 400 sentences nobody chose, 92.4%
of phoneme words agree.** [Write-up](research/t5-kokoro-g2p.md).

    "In 2024 the team shipped 1,024 kernels and spent $3.5 million, up 12%."
    -> ɪn twˈɛnti twˈɛnti fˈɔɹ ðə tˈim ʃˈɪpt wˈʌn θˈWzᵊnd twˈɛnti fˈɔɹ
       kˈɜɹnᵊlz ænd spˈɛnt θɹˈi pYnt fˈIv mˈɪljᵊn dˈɑləɹz, ˌʌp twˈɛlv pəɹsˈɛnt.
    character for character what misaki gives

    text to speech, T6d (kokoro)
    "The quick brown fox jumps over the lazy dog." (af_heart)
    48 phonemes -> 50 tokens -> 130 frames -> 78000 samples = 3.250 s

    stage          gpu    (T6c)   cpu (T3)
    bert            4ms      4ms     104ms   12 ALBERT layers, 46.6x
    dur encoder     3ms      3ms      24ms   + the whole text encoder
    durations       0ms      1ms       8ms   the head, on the host, in fp32
    prosody         1ms      2ms      66ms   gather, shared, both AdaIN stacks
    text encoder    0ms      8ms      15ms   now only a readback
    phoneme side    8ms     17ms     222ms   18% of the utterance
    decoder         1ms      1ms     132ms
    generator      12ms     12ms    3160ms
    excitation     14ms     15ms      24ms   float64 on the host, see T7
    tail            8ms      8ms      22ms
    vocoder        36ms     36ms    3626ms   82% of the utterance
    total          44ms     53ms    3848ms   74.20x real time, from 61.44x

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
| S6 | Encoder on Vulkan (GEMMs, LayerNorm, attention, conv) | **done** — 178x, [write-up](research/s6-parakeet-encoder.md) |
| S7 | The subsampling stack on Vulkan | **done** — 95x, [write-up](research/s7-parakeet-subsampling.md) |
| S8 | The projector, the prediction net, the joint and the TDT loop | **done** — 43x, 257x real time, [write-up](research/s8-parakeet-decode.md) |
| S9 | Long clips: chunking, or full attention at T=3000 | open, see below |
| S10 | The front end on the device, or a faster one on the host | open — **48% of the pipeline** |
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | **done** — 513 tensors readable from Go, 63 dumped, every self-check 0 |
| T2 | Phoneme encoder + ALBERT + predictor, CPU, against T1 | **done** — 5e-7 relative, durations exact |
| T3 | iSTFTNet decoder, CPU; **first waveform** | **done** — 98 dB against the reference, `cmd/tts` speaks |
| T4a | The generator's residual blocks on Vulkan | **done** — 385x, 22.4 TFLOP/s, [write-up](research/t4-kokoro-vocoder.md) |
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
| T7 | The excitation: float64 phase accumulation on the host | **next** — the largest single stage in the model, 14 ms |
| T8 | Voice blending: a request that names several voices | **done** — upstream's own spelling, checked against `load_voice` at every row |

## What exists

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
[`research/t5-kokoro-g2p.md`](research/t5-kokoro-g2p.md) and the short version
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
of them.** `SPEECH.md` had carried "60% of the phoneme side is recurrences" since
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
number before the kernel did: `SPEECH.md` had carried "~33 ms" for them,
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

## T7 — the excitation, and what is actually left

With T6 closed the vocoder is **82% of an utterance** again, and the largest
single stage in the whole model is the **excitation at 14 ms** — 32% of a
44 ms utterance, and nearly twice the whole phoneme side.

It is on the host on purpose and T3 explains why: upstream integrates the
excitation's phase in radians and multiplies by 300, so three seconds holds
**1.3e5 radians, where one float32 ulp is 0.016** and fp16 cannot represent the
number at all. But that is an argument about *upstream's* formulation.
`HarmonicSource` already keeps the phase in cycles and wraps before the sine,
so nothing it computes is ever large — which is why it is more accurate than
the dump rather than less. The open question is therefore not "can the
reference's accumulator move to the device" but **"is the wrapped form
float32-safe, and if so what does it cost there"**, and it is a measurement
rather than an argument: 78000 samples times eight harmonics of
`sin(2*pi*frac(phi))`, with the F0 curve upsampled 300:1 and a uv mask, is an
embarrassingly parallel kernel if the wrap is exact.

Beside it: **the generator is 12 ms and the tail 8**, which T4 left measured
and which are the only parts of this model that were ever arithmetic-bound.

And what is left on the phoneme side is 6 ms of the 8, none of it arithmetic:
ALBERT's [T, 768] readback and its host-side embedding stack, the two
readbacks T6d could not remove, and four submits. Closing any of it means
moving the *vocoder's* input boundary — `asr` currently goes host-side into
`GPUDecoder.Upload` — which is one change, not four, and is the thing to do
after T7 rather than before it.

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
