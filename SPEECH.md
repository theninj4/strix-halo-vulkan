# SPEECH — the two audio verticals

> **Current work.** `PIPELINE.md` (the z-image slice) is **parked** at 14.26 s
> an image; this is what happens next. Same rules as that file: it is
> rewritten rather than appended to, history goes to `TODO.md` and closed
> findings to `research/`.

**Targets**: `wav → text` for `parakeet-tdt-0.6b-v3` and `text → wav` for
`Kokoro-82M`, both end to end in Go on Vulkan, both validated against a Python
reference dump before anything is optimised. Two of `GOALS.md`'s five models,
and the two smallest.

**Status (2026-09-15, later still)**: **speech-to-text is finished** — an 11 s
clip transcribes in 43 ms, 257x real time, whole model resident, S1–S8 done.
**Text-to-speech speaks, at 7.3x real time.** `go run ./cmd/tts -gpu -o out.wav`
takes phonemes to a 24 kHz WAV with no PyTorch on the path: T1 converted the
pickle and built the oracle, T2 put the phoneme side in Go, T3 the vocoder,
T4a moved the eight residual blocks that are 97% of the vocoder's arithmetic
onto Vulkan (**2669 ms to 6.9 ms**), and T4b the upsamplers, so a generator
stage is one upload and one download.

    text to speech, T4b (kokoro)
    "The quick brown fox jumps over the lazy dog." (af_heart)
    48 phonemes -> 50 tokens -> 130 frames -> 78000 samples = 3.250 s

    stage            gpu     cpu (T3)
    phoneme side    205ms      207ms
    decoder         132ms      132ms   54% of the vocoder, 5.5% of its flops
    generator        62ms     3160ms   two stages, incl. 51 ms of readback
    tail + source    46ms       46ms   conv_post, iSTFT, excitation, noise
    total           446ms     3626ms   7.28x real time, from 0.90x

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
| T4a | The generator's residual blocks on Vulkan | **done** — 385x, 22.4 TFLOP/s, 5.7e-4 on the waveform |
| T4b | The upsamplers; a stage that does not come back | **done** — vocoder 487 → 244 ms, 7.28x real time |
| T4c | The decoder, the tail, and the last readback | **next** — 132 ms of 244 is the decoder |
| T5 | G2P — `espeak-ng` shell-out, then a port if it is worth one | |

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

**`kokoro/`** — everything up to the vocoder, on the CPU. `config.go` (and
`Phonemes`, the vocabulary lookup), `model.go` (`Mat`, `Linear`, `Embedding`,
`LayerNorm`, `InstanceNormInPlace`, `GELUNew`), `conv.go` (`Conv1D` with
stride/pad/dilation/groups, `ConvTranspose1D`, `UpsampleNearest`), `lstm.go`
(the bidirectional layer, which appears six times), `albert.go`,
`textencoder.go`, `adain.go` (`AdaIN1d`, `AdaLayerNorm`, `AdainResBlk1d`),
`predictor.go` (`DurationEncoder`, `Durations`, `Expand`, `Prosody`),
`vocoder.go` (`Snake`, `SnakeResBlock`, `HarmonicSource`, `Generator`,
`Vocoder`), `load.go` and `infer.go` (`Prosody`, `Synthesize`, `Speak`).
2340 lines, 18 tests — sixteen against the dump, two against the definitions
over shapes the dump does not reach.

`gpu.go` adds `GPUBlocks`, `NewGPUStage` and `Model.AttachGPU`: the eight
residual blocks and the two upsamplers staged as fp16 fragment tiles, four
rungs of the convolution ladder, and the AdaIN reduction. 23 tests, four of
them on the device.

**`shaders/kokoro_*`** — four builds of `dit_gemm.comp` with **`A_CONV=1`**
(the dilated convolution as an implicit im2col; see below) and seven scalar
kernels: the AdaIN reduction's two halves, the fused affine/Snake/narrow and
its leaky-rectifier build, the residual add and its copy variant, and the
upsampler's epilogue with and without the reflection pad.

**`cmd/tts`** — phonemes in, a WAV out, with the stage profile above.
`-gpu` runs the generator's residual blocks on Vulkan, `-voice` picks one of
54, `-speed` divides the durations, `-noise <seed>` switches the excitation
noise on (off is the reproducible configuration; on is what an utterance meant
to be listened to wants), `-list` prints the voices.

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

**Scale**: `KModel` synthesises this clip on the CPU in 0.23 s — **14x real
time before any of it is on the device** — against Go's 3.63 s, which is what
a reference implementation costs and what T4 is for.

Two findings transfer directly. S7's: the vocoder is 1-D convolutions over
`[T, C]`, and holding them channel-last is what makes the pointwise ones GEMMs
and the depthwise ones coalesced. S8's: the M = 1 rungs of the GEMM ladder
already cover the per-frame projections.

`audio/` already has the inverse FFT the vocoder's iSTFT needs, tested by a
round trip — and S10 would give it a device FFT to share.

## T2 and T3, and what they say about T4

`kokoro.Model.Speak` takes a phoneme string to samples. The phoneme side
matches the T1 dump to **5e-7 relative** at twenty checkpoints and predicts all
50 durations exactly; the decoder matches to **1.2e-6**; the generator, given
the reference's own excitation spectrogram, to **1.3e-5 — 98 dB**.

**The two halves want opposite treatment.** The phoneme side is 207 ms of
which 60% is recurrences, each of their 50 or 130 steps a GEMV at M = 1 that
cannot start until the last one finished — latency-bound, the same shape of
problem S8 found in the transducer loop. The vocoder is **160 GFLOP for 3.25 s
of audio, 49 GFLOP per second of speech**, against 53 M parameters read once:
arithmetic intensity in the hundreds of FLOP per byte, so **compute-bound**.
(T2 guessed the opposite from "164 MB of fp16 weights against a 32 MiB MALL".
That is a statement about the model, not about the vocoder, which re-reads its
53 M parameters once while doing 80 G MACs.)

**97% of the vocoder is one kernel family.** Eight `SnakeResBlock`s, six
convolutions each, every one of them `[T, C] x [C, C]` per tap at C = 128 or
256 — the ladder's own shape, with T in the thousands. The CPU reference gets
55 GFLOP/s and predicted single-digit milliseconds on the device; T4a measured
**6.93 ms at 22.4 TFLOP/s**, which is the prediction and half the WMMA path's
rate on a much narrower N.

**The CPU reference's own fan-out was half its phoneme side.** `parallelFor`
as copied from `parakeet/` pushes one index per worker through a channel,
which a feed-forward stack calls a few dozen times a clip and a recurrence
calls once per timestep — a thousand times, over 2048 gates. Contiguous chunks
instead: 420 ms → 200 ms. The obvious-looking refinement, skipping the fan-out
for small n, is wrong here: the short loops are short in indices, not in work.

## What cannot be reproduced, and how much it is worth

Three quantities in the vocoder are ill-conditioned *in the reference*, and
finding out which of them mattered was most of T3.

**The phase accumulator.** Upstream integrates the excitation's phase in
radians and then multiplies by 300, so three seconds in it holds **1.3e5
radians, where one float32 ulp is 0.016 radians** — a 1% error in the sine.
`HarmonicSource` keeps the phase in cycles and wraps before the sine, so
nothing is ever large, which makes it *more* accurate than the dump and
therefore different from it by 0.3%. **fp16 cannot represent 1.3e5 at all**;
at 6e4 its ulp is 64 radians. T4 must not narrow this.

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

## T4a — the convolution is a layout, again

Every convolution in a generator residual block is a k-tap filter over a
channel-last `[T, C]` activation — k of 3, 7 or 11, dilations of 1, 3 and 5,
C of 128 or 256. Written out that is the GEMM

    C[t, o] = sum over j, c of A[t, j*C + c] * B[o, j*C + c]
    with     A[t, j*C + c] = x[t + j*d - p, c]

so the A address is the ordinary one **plus `j*(d*lda)`**, and because C is a
power of two, j and c are a shift and a mask of the K index. One 16-wide
fragment therefore lies entirely inside one tap. `A_CONV=1` is twelve lines in
`dit_gemm.comp`: **the im2col is an addressing rule, not a pass**, so a dilated
convolution is one dispatch with no packing and no accumulate flag. The
padding is in the data — a 32-frame zero border at each end of the fp16 arena,
with `inOff` at row `border - p` — which is stage 8's conv2d trick one
dimension down, and it leaves the kernel branchless. Every existing build of
`dit_gemm.comp` is byte-for-byte unchanged.

**The readback is the pipeline now, and it is 140x asymmetric.** Writing into a
`DEVICE_LOCAL|HOST_VISIBLE` buffer runs at **29 GB/s**; reading the same buffer
back runs at **210 MB/s**. An 8 MB `[15601, 128]` stage output uploads in
0.3 ms, computes a whole residual block in 0.8-1.3 ms, and **downloads in
38 ms**. About 100 of the vocoder's remaining 487 ms is four such downloads.
S8 reached the same conclusion about the transducer loop from the opposite
direction — there the round trip was latency, here it is a write-combined
read — and the answer is the same: stop coming back to the host.

**The arithmetic has collapsed and the elementwise passes have not.** On the
device the convolutions are 65% of a block and the two AdaINs are 28%, where on
the CPU the convolutions were essentially all of it. Stage 8's VAE decoder
ended in the same place, and the same two levers apply: narrow the activations
to fp16, and fuse the residual add into the second convolution's epilogue.

**Two things that are free because they are per utterance, not per frame.**
Each AdaIN's `fc` is a `[2C, 128]` projection of one style vector, so the host
computes gamma and beta once and 48 of the 70 `fc` layers leave the graph. And
`conv1`'s bias cancels identically inside the AdaIN that follows it — dropping
all three in a block moves the output by 2.7e-7 relative, against 0.031 for the
same experiment on `conv2`, whose output reaches the residual add — so it is
not staged at all.

**AdaIN reduces down a column and is still coalesced**, because the workgroup
is laid out across channels rather than along the column: 256 threads cover
`256/C` row groups, so a wave's addresses are one contiguous run of a row and
the column direction is the loop. The ladder's winner is `reg32x64_w32` at both
rates by 1.10-1.28x, which is §6.2's wave32 result holding on a fourth set of
shapes.

## T4b — the transposed convolution is one GEMM

Both upsamplers have `kernel = 2*stride` and `padding = stride/2`, so every
output frame is reached by exactly two taps and which two is the residue of the
frame index:

    output frame q*s + r - s/2  =  sum_i x[q-1, i]*W[i, n, r+s] + x[q, i]*W[i, n, r]

The obvious reading is s GEMMs writing a strided subset of the rows, which
needs a store stride the kernel does not have. The other reading is **one**
GEMM whose N is `s*C_out`: column `r*C_out + n` of a `[T+1, s*C_out]` result
*is* output frame `q*s + r`, so the same bytes read as `[(T+1)*s, C_out]` are
already the upsampled signal — offset by `stride/2`, which is a pointer. No
strided store, no residue loop, no second kernel, and the A operand is the
existing `A_CONV` addressing at two taps with the arena's zero border supplying
`x[-1]` and `x[T]`.

**The reflection pad is an index map, not a copy.** Folded into the epilogue
that was already adding the bias and the excitation, it is
`src = t == 0 ? 1 : t - 1` — one pass over the tensor for what the reference
does in three.

With the upsampler on the device a stage is **one upload of its input, one of
the excitation projection, and one download of its output**. That removed
192 ms of host convolution and two of the four readbacks at once, which is why
the vocoder halved rather than dropping by the upsamplers' share.

## T4c — what to build next

**The decoder is 132 ms of the vocoder's 244 — 54% of the time on 5.5% of the
arithmetic**, and T3's by-difference estimate had it at 29 ms. It is not a
kernel problem: `AdainResBlk1d` is the same shape of block as the generator's,
and the only reason it is still on the host is that its channel counts are
**514, 1090, 1024 and 512**, and two of those are not powers of two, so
`A_CONV`'s shift and mask do not apply. An `A_CONV=2` build with a real integer
division — the index is wave-uniform, so it is scalar work — plus the three
things the generator's block does not have: a leaky rectifier instead of a
Snake, a `1/sqrt(2)` on the sum, and a shortcut path (`conv1x1`, and a
depthwise transposed convolution on the one block that upsamples).

**Then `conv_post` and the iSTFT** (22 ms). `conv_post`'s N is **22**, which is
not a multiple of the 16-wide tile — pad it to 32 with zero weight columns and
let the epilogue read 22.

**Then the last download** (38 ms). Once the tail is on the device what comes
back is 78000 samples, 312 KB instead of 8 MB.

After that the vocoder is ~40 ms and the phoneme side's 205 ms of recurrences
is the whole cost — which is S8's problem again: six bidirectional LSTMs whose
steps are GEMVs at M = 1, sequential by construction, wanting a resident state
and the loop's control flow on the device.

**Do not narrow the excitation's phase.** It reaches 1.3e5 radians; fp16 cannot
represent it. See above.

## What is left on parakeet, in order of what it would buy

**S10 — the front end is 48% of the pipeline.** 21 ms of a few thousand
512-point FFTs in float64 on the host, untouched since S3 because it was 0.8%
of the clip. Two directions and they are not exclusive: a float32 radix-4 on
the host, or the STFT and the mel filterbank as two dispatches. The filterbank
is a `[T, 257] x [257, 128]` GEMM, which is the ladder's own shape; the FFT is
not, and is the interesting half — and kokoro's iSTFT would share it.

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
