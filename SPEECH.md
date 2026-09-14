# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-14, later still)**: **the speech-to-text vertical is
finished.** S8 moved the transducer tail onto the device — 208 ms to 4.8 ms —
so an 11 s clip goes from 2.68 s on the CPU to **43 ms**, at **257x real
time**, with the transcript and all 46 emissions unchanged. S1–S8 are done.
The whole model is resident: what crosses the bus per clip is the mel
spectrogram in and two floats per emitted token out. Kokoro is next.

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
| T1 | `reference/convert_kokoro.py` + `reference/dump_kokoro.py` | **next** |
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

**`reference/dump_parakeet.py`** — the oracle, which walks the whole model
rather than its ends and checks itself in four places.

## S8, in one page

The full write-up is [`research/s8-parakeet-decode.md`](research/s8-parakeet-decode.md).
Three things worth carrying forward:

**The first latency-bound stage in the engine.** Greedy transducer decoding is
sequential by construction — step n+1's input is step n's output — so there is
no batch and no tile to pick. 1.1 GFLOP took 208 ms because each of 46
emissions was a round trip between a host holding the decision and a device
holding the model. The whole design is about what crosses the bus: the model,
the state and the logits stay resident, and what moves is 2.5 KB of embedding
in and **two floats out** per token.

**M = 1 needs no new kernel, and the ladder proves it.** The four projections
run on the existing GEMM rungs with M padded from 1 up to the tile, because a
GEMM reads its B operand once whatever M is. `BM = 32` ties `BM = 16`
**exactly**, twice, reproducibly — double the arithmetic for the same time —
which is the direct statement that the tail is bandwidth-bound at M = 1 and
that a dedicated GEMV would win back something nobody is paying for.

**24 MB of weights against a 32 MiB MALL.** The LSTM and joint GEMMs measure
449 and 702 GB/s, both *above* this part's DRAM bus and below §5.1b's 930-965
GB/s cache ceiling: the loop re-reads the entire model out of the last-level
cache, 46 times. This is the first place in the engine where §0.4's 32 MiB is
a design constraint that was **met** rather than a cliff that was fallen off —
and it says exactly what would break it: a bigger prediction network, or a
second hypothesis in a beam.

## What is left, in order of what it would buy

**S10 — the front end is 48% of the pipeline.** 21 ms of a few thousand
512-point FFTs in float64 on the host, which has been untouched since S3
because it was 0.8% of the clip. Two directions and they are not exclusive: a
float32 radix-4 on the host, or the STFT and the mel filterbank as two
dispatches. The filterbank is a `[T, 257] x [257, 128]` GEMM, which is the
ladder's own shape; the FFT is not, and is the interesting half.

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

## Kokoro — next, and unchanged

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

Two findings transfer directly. S7's: the vocoder is 1-D convolutions over
`[T, C]`, and holding them channel-last is what makes the pointwise ones GEMMs
and the depthwise ones coalesced. S8's: 82 M params is 164 MB in fp16, five
times the MALL, so unlike the transducer tail it will be on the DRAM law and
the budget is bytes per sample.

`audio/` already has the inverse FFT the vocoder's iSTFT needs, tested by a
round trip — and S10 would give it a device FFT to share.
