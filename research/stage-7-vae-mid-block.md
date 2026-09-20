<!-- Pipeline stage findings, not an IDEAS section: these come from building
     the z-image slice rather than from a numbered experiment. Referenced
     from PIPELINE.md, which stays short by pointing here. -->

[← PIPELINE.md](zimage-pipeline.md) · [research index](README.md) · pipeline stage 7

# Stage 7 — the VAE mid block on the matrix cores

The first optimisation pass chosen by a profiler rather than by a backlog.
`cmd/vaeprof` said the decode's second- and third-largest costs were the mid
block's attention (**1.42 s in one dispatch, 26%**) and its four projections
(544 ms at **62 GFLOP/s**), both of them the fp32 correctness kernels stage 2b
left. Moving both onto fp16 operands and `coopMatMulAdd`:

| | before | after | |
|---|---|---|---|
| mid-block attention | 1.429 s | **15.1 ms** | 94x, 36.4 TFLOP/s |
| the four projections | 503 ms | **1.12 ms** | 449x, 30.6 TFLOP/s |
| VAE decode, 1024² | 5.53 s | **3.56 s** | GPU total |
| VAE decode, wall | 5.65 s | **3.62 s** | |
| **an image** | **19.79 s** | **17.82 s** | two runs, 17.84 and 17.79 |

The decode is now **85% one convolution kernel**, and the mid block is 0.5% of
it.

## What was reused, and the one thing that could not be

Both halves are GEMMs, and the engine already had a tuned one, so the
projections run on **`shaders/dit_gemm.comp` unmodified**. What that costs is
one convention: `vae_common.glsl`'s push-constant block now mirrors
`dit_common.glsl`'s GEMM fields at the same byte offsets, and the two blocks
are the same 88 bytes, because `vk.DispatchMultiTimed` records one
push-constant size across a command buffer. `inOff` and `outOff` already
coincided. That is a smaller price than forking a kernel that carries §2.1,
§2.3, §2.4 and §2.7 behind it.

The attention kernel could not be reused, and the reason is a register file:

- **`dit_attention_wmma.comp` is one wave per workgroup and keeps the whole
  output tile in accumulators** — `HEAD_DIM/16` of them per query tile. At the
  DiT's 128 that is 8. The VAE's head is the full 512-channel width, where it
  is 32, and the build asks for **424 VGPRs against a 256-register file and
  spills 170 into 30 KB of scratch**. Measured with `RADV_DEBUG=shaderstats`
  before a line of host code was written, which is what stopped this being a
  day of wondering why the kernel was slow.

So `shaders/vae_attention_wmma.comp` gives the head to the **workgroup**:
WAVES=4 waves, wave *w* owning component tiles `[w*DW, (w+1)*DW)` of the 512.
That splits the two matmuls in opposite directions, and the asymmetry is the
whole design:

- `S = Q.K^T` reduces **over** the component axis, so each wave computes a
  *partial* S over its own tiles and the workgroup sums the four partials in
  LDS. No wave repeats another's work.
- `O += P.V` is *indexed by* the component axis, so each wave owns DW output
  tiles outright and never sees another wave's.

Having paid for the LDS round trip the cross-wave sum needs, the softmax is
done scalar in LDS as well — the row max, the exponential and the row sum, as
16-lane `subgroupClustered` reductions — rather than through the DiT kernel's
row-constant-matrix trick. That is simpler *and* one barrier cheaper, because
S is already there. The one piece of the trick that survives is the rescale of
O between key blocks: O lives in accumulators, a per-row scale is not a
scalar, and a matrix whose every row is constant multiplies it componentwise.

At QT=1, KTIL=4, WAVES=4 that is **192 VGPRs, no spill, 24 KB of LDS**.

## wave32 again, and the ladder

| build | attention | TFLOP/s |
|---|---|---|
| `qt1_kt4_w32` | **15.1 ms** | **36.4** |
| `qt1_kt2_w32` | 15.3 ms | 36.0 |
| `qt1_kt2` | 19.9 ms | 27.6 |
| `qt1_kt4` | 23.6 ms | 23.3 |
| `qt1_kt8` | 29.7 ms | 18.5 |
| `qt2_kt2` | 32.8 ms | 16.7 |
| scalar (stage 2b) | 1429 ms | 0.6 |

**wave32 is 1.56x at identical tiling**, which is [§6.2](6.2-wave32-vs-wave64.md)
and stage 3c's third arm for a third time — and here it is the *only* thing
that separates the ladder's winner from the middle of it. Two runs of the
whole ladder agree to 1.003 on the winner.

The two knobs behave as stage 3c found: KTIL past 4 loses monotonically (more
score accumulators live, more LDS, and the partial-score buffer grows with it
— 43 KB at KTIL=8), and QT=2 loses more than KTIL=8 does, because at DW=8 an
extra query tile costs 8 more accumulators and 8 more Q fragments. The
arithmetic intensity argument that decides a GEMM decides nothing here, again.

**The projections**: four builds, all fragment-tiled B, spanning 1.12 to
1.52 ms. `reg64_hka4_bt16` wins by 4% over `wg128x256_bt16_swz8` and
`reg64_bt16`, which tie — and it is §2.7's **hoisted K-slab on a tiled B**,
which stage 4a measured as a *loss* on the DiT's `ff.w13`. Same lever, other
sign, different shape: M=16384, N=512, K=512 here against M=4096, N=10240.
At 1.1 ms of a 3.6 s decode the choice is worth nothing and is recorded only
so the default is the measured one.

## The finding: this attention defeats end-to-end validation

Stage 2 already knew the mid block's softmax is a **hard one-hot with entropy
0**, its argmax set by `||k_j||` rather than by q's direction, and that
transposing `to_q` therefore changes the decoded image by 1.5e-5. Stage 7 hit
the same wall one level harder.

The kernel ships with two negative controls. `NO_RESCALE` drops the
online-softmax correction; `NO_CROSS_WAVE` drops every wave's partial score
but its own, i.e. **throws away three quarters of every dot product**. Against
the reference image:

| control | image | verdict |
|---|---|---|
| `NO_RESCALE` | 1.59 | 634x the bound, caught |
| `NO_CROSS_WAVE` | **8.2e-4** | *inside* the bound the correct kernel meets |

A quarter of each score is enough to pick the same key. **No end-to-end
tolerance can catch a score-side bug in this block**, and the danger is
specific: a kernel whose Q or K addressing ignored the wave index — every wave
summing the same 128 components — would have passed every image test in this
package, including the one against diffusers.

So the kernel is validated on its own, on **unit-normal rows**, where the
scores land near 1 and the distribution is soft:

| | image (real activations) | kernel (soft scores) |
|---|---|---|
| correct builds | 8.2e-4 | 2.0e-3 |
| `NO_CROSS_WAVE` | 8.2e-4 | **2.5, 833x** |
| `NO_RESCALE` | 1.59 | 8.15, 2718x |

`TestGPUWMMAControls` now asserts *both* directions — that `NO_RESCALE` fires
and that `NO_CROSS_WAVE` does not — so that if the image ever becomes
sensitive to the scores the test fails and says to promote the control rather
than widen a bound. The general rule: **a saturated softmax is an information
bottleneck, and everything upstream of it is untested by anything downstream
of it.**

## What fp16 costs here, and where the error actually is

The decoded image moves by **8.2e-4** of its RMS, against the fp32 graph's
6.2e-5 — 100x inside the pipeline's own end-to-end bound (2.5e-2), and the
256² comparison against diffusers is unchanged at 0.0252.

On the kernel test the number is 2.0e-3, and it is **v, not the softmax**: v's
entries reach 4.5, where the fp16 quantum is 2e-3, and a weighted average of
144 of them keeps a few times 1e-4 of that. P is bounded by 1 by construction,
so narrowing it is 5e-4 at worst, and the scores accumulate in fp32 — which
they must, since this model's reach 1.16e7 (stage 2).

That bound is measured with a **per-element denominator floored at the RMS**,
which is stage 5's lesson paying off a second time: the worst element is 4.1e-4
off a value of 0.556 — two fp16 ulp — and against the tensor's 0.137 RMS the
same error reads as 3.0e-3, indistinguishable from a real bug.

## Two smaller things

- **The biases went where the passes already were.** `dit_gemm.comp` computes
  `A*B` and has no bias, so rather than fork it or add four elementwise
  dispatches, the four projection biases were folded into the passes that were
  already reading those tensors: three into `vae_pack_f16.comp` and the fourth
  into `vae_rows_to_nchw_add.comp`, which was already fusing the residual.
  Zero extra dispatches. The fp32 path then has to pass `NO_BIAS` explicitly,
  which is stage 6's "there is no default that means not set" in its mildest
  form.
- **The batch cap is no longer load-bearing.** `dispatchesPerSubmit = 4`
  existed because the attention dispatch was 1.5 s and eight of them tripped
  the driver's reset watchdog. The largest dispatch is now a 404 ms
  convolution; 8 measures 3.62 s against 4's 3.64 and completes cleanly. It
  stays at 4 because the difference is inside the noise and the reason to
  raise it is gone with the reason to lower it.

## What is left in this decoder

**conv3x3 is 85% of the decode**, 3.02 s at 3.0-3.2 TFLOP/s against a
55.5 TFLOP/s fp16 ceiling, over 40-odd dispatches of which the largest is
404 ms. That is now the whole of the VAE's remaining headroom, and it is a
different kind of problem from this one: not a GEMM with the wrong kernel in
front of it, but an operator whose implicit GEMM has never been written.
Stage 2's other finding still stands in the way — **fp16 cannot hold this
model's VAE intermediates** (1.16e7 against 65504) — so a narrowing needs the
per-tensor range check stage 2 called for, and the one that matters for a
`conv` is that its *inputs* peak at 497 while its accumulators need fp32,
which is exactly what the matrix cores do natively.
