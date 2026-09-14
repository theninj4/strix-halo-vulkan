# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-14, later)**: **parakeet transcribes on the GPU, exactly.**
The 24 conformer layers went from 2.45 s to **13.8 ms** on the device for an
11 s clip — 178x — with the transcript and the whole decode trace unchanged.
S1–S6 are done. What that exposes is everything that is *not* the layers: the
subsampling stack on the host is now 92 ms, and the TDT decode loop is 205 ms,
so the clip that took 2.68 s takes 332 ms and **the encoder is no longer the
problem**. Kokoro is untouched.

    testdata/jfk.wav: 11.000 s at 16000 Hz
    gpu: 24 layers, 1154 MB of weights, 26 MB of arenas, plan reg32x32_bt16_w32
    1101 mel frames (1100 valid) -> 138 encoder frames (138 valid)

    stage             time  share       (CPU, S5)
    front end         20ms   5.9%        20ms
    encoder          103ms  31.1%       2454ms   <- 92ms of it is CPU subsampling
    projector          2ms   0.5%          2ms
    decode           205ms  62.5%        207ms
    total            332ms               2.68s

    33.1x real time, 4.1x before

## Where the work stands

| # | Stage | State |
|---|---|---|
| S1 | `safetensors` reads the parakeet checkpoint; WAV reader | **done** |
| S2 | `reference/dump_parakeet.py` — mel, layer outputs, joint, decode trace | **done** — 59 tensors, four self-checks |
| S3 | Mel front end in Go, against S2's mel | **done** — 3.1e-4 absolute |
| S4 | Encoder, CPU reference, layer by layer | **done** — 1.7e-4 relative |
| S5 | Prediction net + joint + TDT greedy decode; **first transcript** | **done** — exact string |
| S6 | Encoder on Vulkan (GEMMs, LayerNorm, attention, conv) | **done** — 178x, exact transcript, [write-up](research/s6-parakeet-encoder.md) |
| S7 | The subsampling stack on Vulkan | **next** — 92 ms, 87% of the encoder's wall clock |
| S8 | The projector, the joint and the TDT loop on Vulkan | 205 ms, 62% of the pipeline |
| S9 | Long clips: chunking, or full attention at T=3000 | undecided, see below |
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | |
| T2 | Phoneme encoder + ALBERT + predictor, CPU, against T1 | |
| T3 | iSTFTNet decoder, CPU; **first waveform** | |
| T4 | Vulkan port; `cmd/tts` | |
| T5 | G2P — `espeak-ng` shell-out, then a port if it is worth one | |

## What exists

**`audio/`** — WAV in and out (16-bit PCM only, on purpose), a radix-2 FFT in
float64, a centred STFT that reproduces `torch.stft(center=True,
pad_mode="constant")`, and a Slaney mel filterbank that reproduces
`librosa.filters.mel` to a float32 ulp. The inverse FFT is written and tested
but not yet used; it is kokoro's.

**`parakeet/`** — the model on the CPU (`frontend.go`, `subsampling.go`,
`encoder.go`, `decoder.go`, `decode.go`, `tokenizer.go`, `load.go`) and on the
device (`gpu.go`, `gpugraph.go`): a `GPUEncoder` that stages all 24 layers as
1.15 GB of fp16 into one bank and runs a clip as 960 dispatches over four
shared arenas. `Encoder.Apply` and `GPUEncoder.ApplyMel` have the same
signature, so a caller switches path by switching which one it calls.

**`shaders/parakeet_*`** — the seven kernels the GEMM ladder did not have: the
LayerNorm (with a mean *and* an affine, two builds), silu, the scalar-weighted
residual, the GLU, the depthwise convolution with its folded BatchNorm, the
narrowing pass that adds `bias_v`, and the relative-position shift. Plus two
builds of shaders that already existed: the fragment pack with `bias_u` folded
in, and the score kernel with the position term as an additive bias
(`REL_BIAS`).

**`cmd/asr`** — a WAV in, a transcript and a stage profile out. `-gpu` moves
the layers to the device, `-profile` times every dispatch, `-v` prints the
per-emission trace.

**`reference/dump_parakeet.py`** — the oracle, which walks the whole model
rather than its ends and checks itself in four places.

## S6, in one page

The full write-up is [`research/s6-parakeet-encoder.md`](research/s6-parakeet-encoder.md).
The four things worth carrying forward:

**The score kernel's tail handling encoded an assumption about the size of a
score.** The flash kernel masks pad keys out of `P` but takes the row max
before that mask, which is safe when a real score is a q·k product of
RMS-normalised vectors and never far from a pad key's 0. With the position
bias the real scores are tens of log2 units wide, so on any row whose scores
are all negative a key that does not exist set the scale and the whole
softmax underflowed fp16's smallest subnormal. One `break`, under `REL_BIAS`;
a factor of 100 in the error.

**Two alignments, not one.** The GEMM's M is padded to the plan's tile (160 at
138 frames) and the attention geometry to the key block (192). Using one
number for both padded a clip to 256 and cost 1.28x.

**wave32 wins on a whole model.** Every wave32 rung beats its wave64 twin at
every clip length measured — 1.28x at 138 frames on the identical tile — and
the DiT's own winning tile is 1.83x off the pace, because 86 of its 128 rows
would be padding. §6.2 and stage 3c measured that on one kernel each; this is
the same lever on 960 dispatches.

**A transcript is not a kernel test.** Two of the four negative controls
change the encoder output by 33% and 3.9% and leave the words alone. The
transcript says the port is right; the per-stage tensors say a kernel is, and
they need the rms-of-the-difference measure rather than max-abs-against-rms,
because fp16 puts the latter at 2-3% on tensors that are correct.

## S7 and S8: what is left, and in what order

The profile above is the whole argument, and it inverts the one that ordered
S6:

**S7, the subsampling stack — 92 ms, and it is 2.8 GFLOP.** That is 30
GFLOP/s, on a part that has just done 177 GFLOP in 13.8 ms. It is three
kernels the ladder does not have — a stride-2 `conv2d` over a *single-channel*
input, the depthwise form of it, and a pointwise 256→256 that is a GEMM over
positions — plus the `[T, 4096] x [4096, 1024]` linear, which is already a
GEMM and is half the stack's arithmetic. Stage 8's implicit-GEMM conv
(`research/stage-8-vae-conv.md`) is stride 1 and channel-tiled; the layout
question there (a patch fragment is contiguous only if the channel axis is the
tiled one) is the same question here with a 1-channel input, where it has a
different answer.

**S8, the decode loop — 205 ms, 62% of the pipeline.** The TDT greedy loop is
one LSTM step and one 8198-wide joint per emission, 46 of them for this clip.
The joint's `[8198, 640]` head measures 1.6 TFLOP/s as a GEMV — 6.5 µs a step,
0.3 ms for the whole clip — so this is not an arithmetic problem: it is 46
round trips between the host and a model that lives on the device. The
projector goes with it, and then nothing crosses the bus per clip but the
logits.

Together those two are 297 ms of the 332, so the clip should land near 35 ms —
**310x real time**, against 33x now and 4.1x on the CPU.

## Open, and to settle by measurement

- **What does streaming mean here?** `api/transcription.go` already defines
  `transcript.text.delta`. The TDT loop is naturally incremental — it emits at
  a frame and jumps forward — but the encoder is not: full attention over the
  clip means a chunk boundary changes every frame's hidden state. Chunking
  belongs in the design rather than after it. The ladder's 1024-frame row
  (111 ms, against 59 ms at 768) is the first sign of the quadratic term
  arriving: at 82 s of audio the position projection is 2T-1 = 2047 rows and
  the score matrices are 1024x1024 per head.
- **Is the position term worth hoisting?** `rel_k` is 5.4% of the encoder and
  the eight per-head score GEMMs another 5.2%, and both are recomputed every
  layer because the weight is per layer. Nothing can be hoisted, but the eight
  dispatches could be one: they are the same shape with a different offset,
  which is §3.5's grouped GEMM.
- **Is kokoro's voice pack indexed by `len(phonemes)`?** Unchanged: 510 rows,
  and the reference will say.

## Kokoro — unchanged, and next after the parakeet pipeline closes

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
