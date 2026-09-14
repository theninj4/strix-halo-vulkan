# Stage 6 — the scheduler and the driver

**Result: `prompt → PNG`.** A 1024x1024 image from Z-Image-Turbo in **19.8 s**
on the iGPU, reproducing diffusers' fp32 CPU pipeline to **2.5e-2** of the
decoded image. Two runs agree to 19.79 s and 19.79 s.

| stage | wall | share | |
|---|---|---|---|
| text encoder | **62 ms** | 0.3% | 17 tokens, tokenizer included |
| caption refiners | **6 ms** | 0.03% | once per image, not once per step |
| denoising | **14.05 s** | 71.0% | 8 steps x 1.76 s over 4128 tokens |
| VAE decode | **5.65 s** | 28.6% | |
| **total** | **19.79 s** | | +21.6 s of one-time loading |

Inside a step: **1.70 s** in the transformer's blocks (of which ~0.10 s is the
residual stream crossing the bus), **0.06 s** in the CPU head and tail.

## What this stage adds, and why it had its own bugs

Stages 1-5 each ended with a component that matched its library reference. This
stage adds no component: it is the order they run in, the conventions they hand
each other, and the loop that turns eight forward passes into a picture. Every
bug it had was in that gap, and **not one of them was visible to any component's
own oracle** — two of the three produced a plausible tensor from every module
in isolation.

So the finding that matters first is methodological.

### A whole-pipeline oracle is affordable, because the composition does not know how big the image is

`reference/dump_zimage_run.py` runs diffusers' `ZImagePipeline` in **fp32 on
the CPU** at a 256x256 image: 288 unified tokens instead of 4128, 3.2 s per
step, all eight steps and the VAE in **half a minute**. It dumps the initial
latent — the two RNGs do not agree and nothing else in the pipeline is random —
the per-step latents, and the decoded image.

That is the whole test. `zimage/pipeline`'s `TestPipelineAgainstDiffusers`
walks the same eight steps from the same noise in 2 s and compares latent by
latent. Everything below was found with it, in the order below, in one sitting;
without it the only signal was "the picture is wrong".

Two companions make it a debugger rather than a pass/fail:

- `reference/dump_zimage_step.py` runs **one** transformer forward with hooks on
  the last module of each of the three phases, so the residual stream is
  visible at every boundary the Go implementation also has one. It is what
  turned "the image is wrong" into "the caption embedding is wrong" in a single
  run.
- `reference/dump_zimage.py` is the head's own oracle: the real
  `ZImageTransformer2DModel` instantiated with **no blocks at all**
  (`n_layers=0`), so its patchify, embedders, position ids, final layer and
  unpatchify run for 130 MB instead of 24.6 GB.

**The error bound for this stage is relative L2, not max-over-RMS.** A denoising
trajectory is chaotic: one element of the last latent can be far from the
reference's while the image is the same image. The measured per-step sequence is

    1.4e-4  7.3e-4  2.2e-3  3.0e-3  5.2e-3  7.9e-3  1.3e-2  2.1e-2

i.e. the fp16 chain's error entering at 1.4e-4 and the feedback multiplying it
by about **1.6 a step**. What the bound is really checking is that the growth is
*geometric from rounding* rather than a constant offset from a convention — a
wrong convention does not start at 1.4e-4.

## Finding 1: SwiGLU overflows fp16 on a real prompt, and never on a random one

The first real prompt produced a **black image**, from a latent that was
entirely NaN. Walking it back: the NaN appeared at step 2 of 8, in
`layers.0`, in exactly **five rows of 288**, each row entirely NaN while its
input was clean.

The mechanism, and it is worth stating in full because every step of it is a
property of this graph rather than a mistake:

1. `w2`'s A operand is the SwiGLU output, and it is fp16 because the matrix
   cores take nothing else.
2. On a real prompt the gate and up projections peak at **276 and 256**. Their
   product is **7.1e4** against fp16's 65504, so a handful of elements in a few
   million become infinity.
3. **One infinity anywhere in a row of an A operand makes that whole row of the
   GEMM's output a NaN** — 3840 columns from one element.
4. The NaN is in the residual stream, so it is in the latent, so it is in the
   next step's input, and two steps later everything is NaN.

Stage 4b measured those two halves at **250 and 508** on the reference fixture
and never multiplied them. Measuring it now: on that same *random* fixture the
unscaled SwiGLU peak is already **5.12e4 — 78% of fp16's range**. The margin
was 28%, and a real prompt is what spends it.

**The fix costs nothing, because it never has to be undone.** `w2`'s output is
read by an RMS norm and by nothing else, and an RMS norm is invariant to a
positive scale on its input. So `GPUStack.FFScale` (1/16, a power of two —
exact in the exponent) divides the SwiGLU output on its way into fp16 and *no
consumer is told*. `TestGPUBlockFFScale` is what keeps that true: three decades
of scale move the intermediate by exactly the factor they say and move the
block's output by **8e-7 to 6.4e-6** of its RMS.

The stagewise walk against diffusers now scales that one tensor back before
comparing — and the stage right after it, `ffn_norm2`, is compared *unscaled*,
which is where the invariance claim is actually checked rather than asserted.

## Finding 2: a shared shader with a new push constant is a silent trap

Adding the scale to `shaders/dit_swiglu_f16.comp` **zeroed the text encoder's
feed-forward**. `zimage/qwen` builds the same shader, its push-constant block
left `scale` at its zero default, and `silu(g) * up * 0` is zero.

What makes it worth a paragraph is the *shape* of the failure. The encoder did
not produce zeros or NaNs; it produced a tensor with a plausible RMS and a
plausible shape, missing only its massive activations — absmax **56.8 where the
reference has 13753**. Everything downstream ran happily and the image was
colourful noise. A push constant whose zero value is a legal-looking multiplier
is a landmine: **there is no default that means "not set"**.

Qwen's scale has to be 1 and it is now stated so, with the asymmetry written
down at the site: the DiT's SwiGLU output feeds a GEMM whose output feeds an RMS
norm, and the encoder's feeds a GEMM whose output feeds a **residual add**,
which is not scale-invariant. The same tensor, the same shader, and only one of
them has the headroom for free.

(`go test ./zimage/qwen` would have caught this. It was found by the end-to-end
reference instead, because the end-to-end reference was the thing being run.)

## Finding 3: the VAE's watchdog limit is per *batch*, and the graph is not flat

At 1024x1024 the decode came back `VK_ERROR_DEVICE_LOST` on its second submit —
`amdgpu: ring gfx_0.0.0 timeout`, a GPU reset in the kernel log. Stage 2 had
already found the watchdog and set `dispatchesPerSubmit = 8`, which is fine at
every size stage 2 measured.

The reason it is not fine at the full image is that the decoder's dispatches
differ by two orders of magnitude. Profiling the 119 of them at a 128x128
latent:

| dispatch | time | share |
|---|---|---|
| **mid-block attention, 16384 rows x 512** | **1.51 s** | **27.1%** |
| conv3x3 512→512 @512x512 | 408 ms | 7.3% |
| conv3x3 256→256 @1024x1024 | 392 ms | 7.0% |
| ... 115 others | | |

A batch of eight around the 1.5 s one is over the limit on its own. **4 works,
and is the same wall clock as 2 to within 1%** (5.591 s against 5.545 s), so
the halving is free. The lesson generalises past this decoder: a
dispatches-per-submit cap is a proxy for a *time* cap, and it is only a good
proxy while the dispatches are the same size.

## Finding 4: two of the transformer's three phases are per-image, not per-step

The transformer runs its 34 blocks as three phases over three different
sequences. The **context refiners** — the two blocks diffusers builds with
`modulation=False` — take neither the timestep nor the latents. Nothing about
them changes between steps.

So the pipeline runs them **once per image**: `cap_embedder`, two blocks over
the 32-row caption stream, and the result is held on the host and written back
behind the image stream every step. It is 6 ms against 8 x 6 ms, which is
nothing at 1024x1024 — but it is a third of the caption's cost at 256x256, and
it is the structure that makes the arena work: one residual stream, one arena,
three phases, and the concatenation is the arena's own layout rather than a
copy.

`GPUStack.Upload(x, at, rows)` is what that needs and `Apply` could not
express: write a tensor at a row offset and state the run's length. The two
shorter phases then run as prefixes of the same arena the long one uses.

## Finding 5: where the seconds go, and what is left

**71% denoising, 28.6% VAE, 0.3% text encoder.** The step is 1.76 s against
`cmd/ditstack`'s 1.60 s of pure dispatches, and the 0.16 s between them is the
pipeline's own overhead, all of it addressable:

| | per step | per image | what it is |
|---|---|---|---|
| CPU head and tail | 58 ms | 0.47 s | patchify, `x_embedder`, the final layer, unpatchify |
| stream across the bus | ~100 ms | 0.8 s | 63 MB up and 63 MB back, per step |

Both exist because the head and tail are on the CPU, and both go away together:
put `x_embedder` and the final layer on the device and what crosses the bus per
step is the **[4096, 64] latent, 1 MB, instead of the [4128, 3840] residual
stream, 63 MB**. That is 1.3 s of a 19.8 s image, or 6.4%, for two GEMMs at
shapes the existing kernel already covers (K=64 and N=64) plus a LayerNorm.

It is still not the largest item. **The VAE is 28.6% of the image and has had
no optimisation pass at all** — it is fp32 throughout, its conv3x3 runs at
3.0-3.2 TFLOP/s against a 55.5 TFLOP/s fp16 ceiling, and one attention dispatch
is 27% of it.

## What the head had to get right

`zimage/dit/head.go` is 130 MB of small linear layers and almost entirely
convention. Each of these produces a tensor of the right shape and a picture of
noise, and each has a test and a control in `head_test.go` that breaks exactly
it:

| | the trap | control lands at |
|---|---|---|
| Patch feature order | `(ph, pw, c)`, channel fastest — and the channel-major order **round-trips through unpatchify**, so the round trip proves nothing | 26149x |
| Timestep embedding | **cos then sin**, not the sin-then-cos most diffusion code uses; every norm is identical | 38293x |
| Caption pad token | replaces the row **after** the embedder; diffusers pads the *input* by repeating the last row, which nothing then reads | 48474x |
| Final layer's norm | a **LayerNorm** — the mean is subtracted — where every other norm in this model is an RMS norm | 219x |
| Caption position ids | start at **1**, and the image's axis-0 id is the **padded** caption length plus one, so 40 tokens and 64 tokens put the image at different rotary positions | asserted as ids |

And one that is diffusers' own and only shows up if you read the code:
`_pad_with_ids` builds the caption's ids from the **already-padded** length and
then appends `pad_len` more, so the id list is longer than the stream and
`_prepare_sequence` truncates it. Reproducing the bug is what matching requires.

## The pipeline's negative controls, and one that is weak

`TestPipelineDetectsErrors` breaks three things the composition gets to decide.
Two are decisive, one is worth knowing is not:

| control | final latent | image |
|---|---|---|
| model output not negated | 40x the bound | 54x |
| caption not refined (skip the context refiners) | 27x | 56x |
| **caption positions from 0 instead of 1** | **2x** | **3x** |

The last one is the off-by-one the checkpoint invites and it is the one the
tests barely catch: shifting every caption token one place along the rotary
axis moves the image by three times the fp16 chain's own error and no more. It
is caught here, and it would not be caught at a looser bound — which is the
argument for keeping the end-to-end bound as tight as the measurement allows
rather than rounding it up to a comfortable number.
