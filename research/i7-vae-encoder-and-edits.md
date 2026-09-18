# I7 — the VAE encoder, and `/v1/images/edits`

**Result: `POST /v1/images/edits` answers.** A 1024x1024 edit at the default
strength is **11.3-12.1 s** against a generation's 14.3, the composition
reproduces diffusers to **1.1e-2** relative L2 on the decoded image, and the
last refusal `API.md` carried is gone.

| | |
|---|---|
| encoder, 1024x1024, matrix cores | **425-431 ms**, 122 dispatches |
| encoder, 1024x1024, fp32 convolutions | 1.78 s — the oracle, and selectable |
| an edit, 1024², strength 0.8 (6 of 8 steps) | **11.29 / 12.06 s** |
| an edit, 1024², strength 0.5 (4 steps) | 7.94 s |
| an edit, 1024², strength 0.3 (2 steps) | 4.63 s |
| an edit, 512², strength 0.7, over HTTP | **2.51 s** |
| residency, 1024² ceiling | +0.21 GB weights, +1.55 GB fp32 arena, +0.26 GB fp16 |
| composition vs diffusers, 256² | **1.06e-2** relative L2 on the image |

The port itself was the easy half and went as `IMAGE.md` predicted: the
encoder is the decoder's mirror, so `ResnetBlock`, `Attention`, `MidBlock`,
`GroupNorm`, `SiLU` and `Conv2D` are shared and the graph reuses every kernel
stage 7 and stage 8 wrote. Three things were not predictable from the shape,
and two of them changed a decision.

## 1. The stride-2 convolution is not a kernel

`IMAGE.md` listed it as the genuinely new piece — `vae.Conv2D` had no stride
field, three downsamplers need one, and the implicit-GEMM gather assumes 1.
It turns out not to need a kernel at all.

diffusers' `Downsample2D` pads `(0, 1, 0, 1)` and convolves 3x3 with stride 2
and no padding of its own, so output `(oh, ow)` reads input rows `2oh..2oh+2`
and columns `2ow..2ow+2`. That is **exactly what the stride-1 pad-1 form
computes at `(2oh+1, 2ow+1)`** — at the far edge included, where the stride-1
form's own zero padding supplies the pixel the asymmetric pad would have. So
the device runs the same filter at stride 1 on stage 8's kernel and keeps one
pixel in four (`shaders/vae_downsample2x.comp`, 30 lines).

It costs four times the downsamplers' arithmetic: 927 GFLOP against 232 at
1024x1024, which the profile below puts at about 18 ms of a 425 ms encode. What
it buys is that no kernel, no packed layout and no ladder in stage 8 has to
learn about a stride. `TestStrideTwoIsStrideOneSubsampled` asserts the identity
on the real filter and its negative control takes the *even* pixels instead,
which lands 32000x above the bound.

The CPU path does implement the real strided convolution (`Conv2D.Stride`,
`Conv2D.PadEnd`), because it is the oracle and it should mirror diffusers
rather than the trick.

## 2. fp16 convolution operands: range was never the question

**This is the finding, and it is the opposite of the one stage 8 had to make.**

Stage 8's precondition was *range*: the decoder's activations peak at 497, 132x
inside fp16, so the operands narrow safely. The encoder passes that test with
more room — `TestEncoderConvInputsFitFP16` measures its worst convolution input
at **284, 231x inside fp16**. And the matrix-core path was still wrong by 0.158
relative at `conv_out` and **0.58 at the latent**.

It is not a shader bug. Simulating the same narrowing on the CPU port —
operands to fp16, accumulation in fp32, nothing else changed — reproduces
**0.1577 and 0.582**, the GPU's numbers to four figures. What the encoder has
and the decoder does not is *conditioning*: it contracts a 3-channel image into
a 32-channel latent at an eighth of the resolution, and a perturbation at
`conv_in` comes out of `conv_norm_out` amplified by about 4e5.

That amplification is a property of the arithmetic, not of this port, and the
reference has it too. Asking diffusers for the same encode in float64 and
diffing its own float32 against it:

| stage | rel | stage | rel |
|---|---|---|---|
| `down.0.resnets.0` | 4.8e-06 | `down.3.resnets.1` | 8.1e-05 |
| `down.1.downsample` | 1.2e-05 | `mid.resnets.0` | 2.0e-04 |
| `down.2.downsample` | 3.4e-05 | `conv_norm_out` | **6.3e-04** |
| | | `conv_out` | 1.5e-05 |

The Go CPU port lands within about 3x of those everywhere, which is what two
independent float32 summation orders cost. So **the encoder's own stagewise
bound had to be measured rather than inherited** from the decoder's 2e-4, and
`conv_norm_out` — a group norm dividing by a standard deviation that the
accumulated error has already moved — is where both paths are worst.

### What made the fast path the default anyway

The honest question was not the intermediate, it was the answer. Both metrics
on the same 256x256 latent:

| | |
|---|---|
| latent, max element over RMS | **0.58** |
| latent, relative L2 | **2.2e-2** |
| decoded back to a picture | 3.0e-3 relative L2, mean **4.2e-4** of a [-1, 1] range |

A twentieth of one 8-bit level. The two metrics disagree by 27x on the same
tensor because the first is set by a handful of outliers in a latent whose RMS
is 2.5 — which is `PIPELINE.md`'s third rule exactly: *which denominator is
right is a property of the model*. A latent is the chaotic-trajectory case and
is bounded in relative L2, the same as the pipeline's own composition.

And in an *edit* it matters less still: SDEdit multiplies the encoded latent by
`(1 - sigma)`, which is 0.1 at the default strength, before the loop ever sees
it. Measured end to end at 256x256, the edit's image is **1.06e-2** from
diffusers' — *better* than the plain generation's 2.2e-2, because six of its
eight steps of fp16 error compounding never happen.

So: matrix cores by default, **1.78 s to 425 ms**, with the fp32 graph kept as
the oracle and selectable (`vae.Options{Conv: vae.ConvScalar}`,
`go run ./cmd/vaebench -encode`). If an edit ever has to be exact rather than
fast, the flag is there and it costs 1.35 s.

## 3. Where the encode's time goes, and it is not the convolution

`go run ./cmd/vaebench -encode -sizes 128 -profile`, 1024x1024:

| | | |
|---|---|---|
| groupnorm | 22 x | **128.1 ms, 30.2%** |
| conv3x3 | 25 x | 126.1 ms, 29.7% — 39.3 TFLOP/s |
| packconv | 27 x | 73.8 ms, 17.4% |
| silu | 21 x | 30.7 ms, 7.2% |
| add | 10 x | 26.8 ms, 6.3% |
| attention | 1 x | 16.1 ms, 3.8% — 34.1 TFLOP/s |
| to_nchw, downsample, conv1x1, … | | 22 ms, 5.2% |

The decoder's distribution is convolution-first and this one is not. Group norm
is the largest item because the encoder's early tensors are its largest — 128
channels at full resolution, 537 MB — and a group norm reads them twice and
writes them once. That is the same bandwidth-bound elementwise back-to-back
pattern I5 and I8 already name, in a third shape, and it is now the biggest of
the three at 128 ms.

## 4. SDEdit, and the question that was open

`IMAGE.md` said "whether an 8-NFE turbo distillation edits acceptably that way
is unmeasured". **It does.** Relative L2 of the edited image against the input,
at 256x256, with an encode/decode round trip as the floor:

| | relative L2 to the input | steps run |
|---|---|---|
| round trip, no denoising | 0.073 | — |
| strength 0.2 | 0.178 | 7..7 |
| strength 0.4 | 0.291 | 5..7 |
| strength 0.6 | 0.376 | 4..7 |
| strength 0.8 | 0.630 | 2..7 |
| strength 1.0 | 1.441 | 0..7 |

Monotone, which is the only part of that table worth asserting — a strength
knob that is not monotone does not mean what it says, whatever the pictures
look like — and `TestEditStaysNearTheInput` asserts it. The pictures agree: at
0.3 the input is essentially unchanged, at 0.8 a "full moon" prompt puts a moon
in the sky and starts the snow falling while the fox keeps its pose.

Two composition checks beyond the diffusers walk, and both were worth having:

- **`TestStrengthOneIsAGeneration`.** At strength 1 the first sigma is exactly
  1, so `(1-sigma) x0 + sigma eps` is `eps` and an edit *is* the generation
  from that noise. It is asserted **bit-identical**, not merely close. An
  off-by-one in `StartStep`, a sigma read from the wrong end, or the noising
  written the other way round all still produce an image and all fail this.
- **The `StartStep` floor.** diffusers' arithmetic is
  `keep = int(steps * strength)`, and `int(8 * 0.1)` is zero: a small edit
  would start at the terminal sigma and run *no* steps, handing back a round
  trip of the input. An edit always runs at least one step.

## 5. What the endpoint decides that the pipeline does not

Two policy questions the model has no business answering, both resolved at the
HTTP layer and both written down there:

- **The size an edit with no `size` gets is the input picture's own shape**,
  fitted inside the ceiling and rounded to a multiple of 16 — not the server's
  default, which would silently reframe what was sent. It is never scaled
  *up*: a 512x512 picture on a 1024x1024 server renders at 512, because four
  times the tokens cannot paint detail the input never had.
- **Fitting is cover-crop** (`backend/resize.go`), CSS's `object-fit: cover`,
  because an edit whose point is to keep the composition should not begin by
  stretching it. With the default geometry the crop is at most fifteen pixels
  and it is a pure scale. The resampler is a separable triangle whose support
  widens when shrinking, which is the cheap form of area averaging;
  `TestFitImageIsAPartitionOfUnity` is the check that catches every way a
  kernel's weights can fail to sum to 1.

`mask` is parsed and refused with a 501 that says why: masked blending is a
per-step operation on the latent and a different mechanism from SDEdit. More
than one input image is a 400 for the same reason — OpenAI's field is a list
because their model composites, and this one does not.

## What this leaves open

- **The encoder's group norms**, 128 ms of 425. Same fusion as I5 and I8 and
  now the largest instance of it.
- **A real stride-2 convolution**, worth the ~18 ms above if the encode ever
  matters. It does not today: the encode is 3.6% of an edit.
- **Masked edits**, which are a third piece and a real capability rather than
  a percent.
