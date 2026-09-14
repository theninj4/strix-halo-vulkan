# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-14)**: **parakeet transcribes on the CPU, exactly.**
`go run ./cmd/asr testdata/jfk.wav` turns 11 s of audio into the reference's
transcript, emission for emission, in 2.68 s — 4.1x real time before a single
shader is written. S1–S5 are done and S6 (the Vulkan port) is next. Kokoro is
untouched.

    testdata/jfk.wav: 11.000 s at 16000 Hz
    1101 mel frames (1100 valid) -> 138 encoder frames (138 valid)

    stage             time  share
    front end         20ms   0.7%
    encoder         2.454s  91.5%
    projector          2ms   0.1%
    decode           207ms   7.7%

    And so, my fellow Americans, ask not what your country can do for you,
    ask what you can do for your country.

## Where the work stands

| # | Stage | State |
|---|---|---|
| S1 | `safetensors` reads the parakeet checkpoint; WAV reader | **done** — `DType` gained `I64`, `audio.DecodeWAV` |
| S2 | `reference/dump_parakeet.py` — mel, layer outputs, joint, decode trace | **done** — 59 tensors, four self-checks in the manifest |
| S3 | Mel front end in Go, against S2's mel | **done** — 3.1e-4 absolute on unit-variance features |
| S4 | Encoder, CPU reference, layer by layer | **done** — 1.7e-4 relative at the encoder output |
| S5 | Prediction net + joint + TDT greedy decode; **first transcript** | **done** — exact string, identical trace |
| S6 | Encoder on Vulkan (GEMMs, LayerNorm, attention, conv) | next |
| S7 | `cmd/asr`, profile, then optimise | CPU half done (the table above) |
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | |
| T2 | Phoneme encoder + ALBERT + predictor, CPU, against T1 | |
| T3 | iSTFTNet decoder, CPU; **first waveform** | |
| T4 | Vulkan port; `cmd/tts` | |
| T5 | G2P — `espeak-ng` shell-out, then a port if it is worth one | |

## What was built

**`audio/`** — the host-side signal processing both verticals need. WAV in and
out (16-bit PCM only, on purpose), a radix-2 FFT in float64, a centred STFT
that reproduces `torch.stft(center=True, pad_mode="constant")`, and a Slaney
mel filterbank that reproduces `librosa.filters.mel` to a float32 ulp. The
inverse FFT is written and tested but not yet used; it is kokoro's.

**`parakeet/`** — the model. `frontend.go` (preemphasis → STFT → mel → log →
per-utterance normalisation), `subsampling.go` (the strided conv stack),
`encoder.go` (24 FastConformer layers, including the relative-position
attention), `decoder.go` (the LSTM prediction network and the joint),
`decode.go` (the TDT greedy loop), `tokenizer.go` (decode only), `load.go`.

**`cmd/asr`** — a WAV in, a transcript and a stage profile out, `-v` for the
per-emission trace with timestamps.

**`reference/dump_parakeet.py`** — the oracle. It walks the whole model rather
than its ends, and it checks itself in four places, all recorded in the
manifest: the by-hand encoder stack reproduces `encoder()` exactly
(`stack_gap` 0), the BatchNorm fold is an affine to 1.5e-5, the hand-written
TDT loop emits what `model.generate` does, and the transcript is the clip's
known one.

**`testdata/jfk.wav`** — the fixture, 11.000 s of 16 kHz mono speech, public
domain. Its decode is pinned against CPython's `wave` by a sha256 in
`audio/wav_test.go`, so a change to the decoder cannot silently move every
downstream comparison.

## What the checkpoint said that the paper reading did not

Four corrections to this file's own inventory, all of them found by running
the thing rather than by reading about it:

- **The subsampling is separable, not three plain convolutions.** It is
  NeMo's `dw_striding`: `conv2d(1→256, stride 2)` and then *twice*
  `depthwise(256, stride 2) + pointwise(256→256)`, five weight tensors at
  ModuleList indices 0, 2, 3, 5, 6 with ReLUs in the gaps. The collapsed
  shape table read `[256, 1, 3, 3]` five times and hid the grouping.
- **A clip is 8x shorter in frames than it is in mel frames, and the two
  length formulas disagree by one.** 176000 samples → 1101 mel frames of
  which **1100** are valid → **138** encoder frames, all valid. The tensor
  grows by `1 + n/hop` and the valid count by `n/hop`; the trailing frame is
  zeroed, excluded from the normalisation statistics, and absorbed by the
  subsampling.
- **The attention is full, not windowed.** `config.json` names no
  `att_context_size` and the mask transformers builds is padding only — a
  rectangle, not a band. At 138 frames that is nothing; at 3000 frames (30 s)
  it is a 3000x3000 score matrix per head, and it is the reason a long clip is
  a different problem from a short one.
- **The blank's embedding row is zeros.** So the first prediction step cannot
  be validated through its input at all — only through the LSTM state it
  leaves behind, which is what `TestPredictionMatchesReference` does.

## The one real kernel question, answered on the CPU

Relative-position attention is two score matrices summed, not one:

    (q + bias_u)·k^T          content, over key positions
    (q + bias_v)·rel_k^T      position, over *relative* offsets

`rel_k` projects a sinusoidal embedding of every offset from +(T-1) to -(T-1),
so the position term is `[T, 2T-1]` and has to be folded onto the `[T, T]`
grid. transformers does that with `_rel_shift` — a left pad, a reinterpret of
the flat buffer, a dropped row — and the closed form, checked exactly against
the reference for all eight heads, is

    shifted[i][j] = raw[i][T-1-i+j]

i.e. **column j of row i holds the score for relative offset i-j**, read
straight off a diagonal. A shader does not need the pad-and-reinterpret dance;
it needs that index. `TestRelShiftIsNotASlice` pins it against the mistake it
is most often confused with — taking the middle T columns — which agrees with
it on exactly one row.

The position term also costs something worth knowing: `relative_k_proj` is a
`[2T-1, 1024] x [1024, 1024]` GEMM per layer, which at T=138 is *twice* the
work of the q projection and at T=3000 is 43x it. It is the same matrix for
every layer's input but a different weight per layer, so it cannot be hoisted
— but it is independent of the activations, so it can be computed once per
layer ahead of the attention rather than inside it.

## Accuracy, and what the bounds are worth

Every stage is compared against the dump with a *relative* bound, because the
stages' rms spans 0.02 to 400 and an absolute one would mean nothing across
them. The measured drift, worst case per stage:

| stage | drift | against |
|---|---|---|
| mel filterbank | 3.7e-9 | a peak of 0.042 (one float32 ulp) |
| power spectrum | 1.1e-5 relative | bins within six decades of the peak |
| log mel | 3.0e-4 | rms 10.1 |
| normalised mel | 3.1e-4 | rms 1.0 |
| subsampled | 1.7e-2 | rms 373 (4.5e-5 relative) |
| layer 0 out | 4.3e-4 | rms 10.0 (4.2e-5 relative) |
| encoder out | 3.4e-6 | rms 0.021 (1.7e-4 relative) |
| joint logits | 4.6e-5 | rms 36.6 |
| transcript | **exact** | 46 emissions, identical frames and durations |

Two of those bounds are set by the *reference* being the imprecise side rather
than this code: the power spectrum, where float32 loses three digits between
the loudest bins and those six decades down, and the position embeddings,
where torch holds the angle in float32 and one ulp at an offset of 137 frames
is already 1e-5. Both are computed in float64 here and rounded once.

Every bound has a negative control beside it — `TestFrontEndDetectsErrors` and
`TestEncoderDetectsErrors` — that breaks the thing the bound is supposed to
protect (the window convention, the preemphasis, the mel scale, the content
bias, the BatchNorm fold, the GLU halves, the sin/cos interleave) and requires
each to miss by at least 10x the bound. They miss by 119x to 6411x.

## S6: what the Vulkan port is walking into

The encoder is 91.5% of the CPU time and ~180 GFLOP (89 G multiply-adds) for
eleven seconds of audio. Per layer, at T=138:

| op | shape | G MAC | share |
|---|---|---|---|
| feed forward 1 and 2 | `[T,1024]x[1024,4096]` and back, x2 | 2.32 | 63% |
| q/k/v/o projections | `[T,1024]x[1024,1024]` x4 | 0.58 | 16% |
| `relative_k_proj` | `[2T-1,1024]x[1024,1024]` | 0.29 | 8% |
| conv pointwise 1 and 2 | `[T,1024]x[1024,2048]`, `[T,1024]x[1024,1024]` | 0.43 | 12% |
| attention scores and context | `[8,T,T]` | 0.08 | 2% |

which is **the GEMM ladder this repository already has**, at a narrow M — and
the ladder was re-measured at these exact shapes this session (`bench`'s
`shapes` family; the parakeet rows in `bench/modelshapes.go` were corrected
against the checkpoint first, since they were missing `relative_k_proj` and
the subsampling linear and had the projector running per emitted token rather
than once per frame):

| shape | M=138 (11 s) | M=384 (30 s) | M=767 (rel_k) | M=1024 |
|---|---|---|---|---|
| `[M,1024]x[1024,1024]` | **23.1** | 32.2 | 34.0 | — |
| `[M,1024]x[1024,4096]` | **29.2** | 31.7 | — | 30.2 |
| `[M,4096]x[4096,1024]` | **27.4** | 33.2 | — | 26.4 |

TFLOP/s of *useful* work, i.e. charged for the tile padding. The winner at
every one of those is the same rung — `wmma_reg32_bt_hkab4_w32_padab128`, the
**wave32 narrow-M** build — and it wins by nearly 2x over the wave64 rungs at
M=138 (23.1 against 12.7). At M=1024 `reg64_bt_hka4` takes the lead back. So
stage 5c's finding holds here and the plan is the same shape as
`qwen.PlanFor`: pick the rung from the clip length.

The padding is the other thing M=138 pays: BM=32 rounds it to 160 (86%
useful), BM=64 to 192 (72%), and the DiT's 128x256 tile to 256 (54%, and
10.6 TFLOP/s — half of what the narrow rung gets). "Use the kernel we already
have" would cost a factor of two here.

**The target that follows.** 177 GFLOP at the measured ~26 TFLOP/s is 6.8 ms,
and the bandwidth floor — 1.25 GB of fp16 weights read once — is 5.3 ms at
236 GB/s. So an 11 s clip should encode in **~7 ms**, against 2.45 s on the
CPU: a 350x speedup, and ~1500x real time. The decode loop is already
negligible and gets more so — the joint's `[8198, 640]` head measures 1.6
TFLOP/s as a GEMV, i.e. 6.5 µs a step, 0.3 ms for the fixture's 46 steps.

Three things the ladder does not have:

1. **stride-2 conv2d over a 1-channel input**, plus the depthwise form of it.
   The stage-8 implicit GEMM is stride 1. This is 2.8 GFLOP of the 180 and it
   runs once, so it is a correctness problem rather than a performance one.
2. **The rel-shift index above**, inside the score kernel.
3. **LayerNorm with a mean**, five per layer. `dit_final_norm.comp` has it.

And one thing to decide rather than port: **M moves with the clip**. 11 s is
T=138 and 30 s is T=375, against a DiT tuned at M=4096 — stage 5c's finding
(the winning tile moves with the sequence length, and re-planning per run is
free) is the precedent, and `qwen.PlanFor` is the shape of the answer.

**fp16 headroom, surveyed and then tested:** the largest activation anywhere
on the path is **7284** (the subsampling's output and the residual after the
first feed forward, layer 0) against fp16's 65504, and the largest weight is
45.3 — a factor of 9 of headroom, but fp16's resolution at 7000 is 4, so the
residual stream was the tensor to watch rather than the overflow. Narrowing
every operand on the CPU reference and asking for the transcript settles it:
**the words and the whole decode trace are unchanged**, at a cost of 4.3%
relative drift at the encoder output against fp32's 0.017%. The transducer has
that much margin; a tensor-level bound does not.

## Open, and to settle by measurement

- ~~**Does fp16 hold through 24 layers?**~~ **Answered: yes, and by the only
  bound that counts.** `Model.SetF16` narrows every matrix-core operand —
  weights in place, activations on the way into each projection, accumulation
  and norms and residuals left in float32, which is the shape the port will
  have — and the transcript is unchanged *and the decode trace is identical*,
  step for step (`TestF16SurvivesTheTranscript`). What it costs is real
  though: the encoder output drifts **4.3% relative** against the reference
  where fp32 drifts 0.017%, growing layer by layer (0.7% at layer 0, 1.4% at
  layer 12). **So S6 cannot be validated by a tight tensor bound.** Its bound
  is the transcript, with per-stage tensors used to locate a fault rather than
  to certify its absence.
- ~~**Where is the roofline crossover for this T?**~~ **Answered
  arithmetically and then measured.** A projection's intensity counting only
  weight traffic is `2M/bytes-per-weight`, so at fp16 it is exactly M flop per
  byte against this device's 235 crossover (`research/3.4-model-shapes.md`):
  the encoder is **memory-bound below 235 frames — 18.7 s of audio — and
  compute-bound above it**. At Q4 the crossover moves to 59 frames (4.7 s).
  The measured rates above are the confirmation: at M=138 the best rung
  reaches 29.2 TFLOP/s on `[138,1024]x[1024,4096]`, which is 90% of that
  shape's *bandwidth* ceiling (32.6 TFLOP/s) and 53% of the matrix rate.
  This inverts the DiT's situation, where every shape was compute-bound.
- **What does streaming mean here?** `api/transcription.go` already defines
  `transcript.text.delta`. The TDT loop is naturally incremental — it emits at
  a frame and jumps forward — but the encoder is not: full attention over the
  clip means a chunk boundary changes every frame's hidden state. Chunking
  belongs in the design rather than after it.
- **Is kokoro's voice pack indexed by `len(phonemes)`?** Unchanged from last
  session: 510 rows, and the reference will say.

## Kokoro — unchanged, and next after S6

`models/Kokoro-82M/`, 82 M params, a StyleTTS2 derivative in a PyTorch pickle
(`kokoro-v1_0.pth`, 327 MB F32), five modules: `bert` (ALBERT, 12 layers x
768, weights shared), `bert_encoder`, `text_encoder`, `predictor` (duration +
F0/N prosody, AdaIN-conditioned), `decoder` (the iSTFTNet vocoder, 53 M of the
82 M). `config.json`: `n_token` 178 with the IPA vocabulary inline,
`style_dim` 128, `hidden_dim` 512, `n_mels` 80, `upsample_rates [10, 6]`,
resblocks `(3,7,11) x (1,3,5)`, final iSTFT at `n_fft` 20 / `hop` 5, 24 kHz
out. `voices/*.pt`, 54 of them, each `[510, 1, 256]` split 128/128 between the
predictor and the decoder.

Three things it needs that parakeet did not: a **pickle-to-safetensors
conversion** before a single number can be compared, **1-D transposed
convolution** and the AdaIN resblock stack (a new operator family, two thirds
of the parameters), and a **length regulator** that makes the output length
data-dependent, which no stage of z-image or parakeet was. G2P stays last and
outside the model: **the first vertical takes phonemes, not text**.

`audio/` already has the inverse FFT the vocoder's iSTFT needs, tested by a
round trip.
