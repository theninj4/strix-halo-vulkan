# Stage 8 — conv2d on the matrix cores

**Result: conv3x3 from 3.02 s to 244 ms and 3.1 to 41-42 TFLOP/s, the VAE
decode from 3.62 s to 876 ms, and a 1024x1024 image from 17.82 s to 15.0 s.**
The largest single dispatch in the pipeline, 404 ms, is now 30.3 ms.

| | before | after | |
|---|---|---|---|
| `conv3x3 512->512 @512x512` | 404.2 ms, 3060 GFLOP/s | **30.3 ms, 40841** | 13.3x |
| `conv3x3 256->256 @1024x1024` | 391.9 ms, 3156 GFLOP/s | **29.5 ms, 41983** | 13.3x |
| all conv3x3 | 3.017 s | **244 ms** | 12.4x |
| all conv1x1 | 153 ms | **12 ms** | 12.8x |
| VAE decode, GPU total | 3.557 s | **801 ms** | 4.4x |
| VAE decode, wall | 3.62 s | **876 ms** | 4.1x |
| VAE decode, in an image | 3.60 s | **805 ms** | 4.5x |
| an image, 1024², 8 steps | 17.82 s | **15.00 s** | 1.19x |

The two decode rows differ by 71 ms because `cmd/vaebench`'s wall clock
includes a read-back of the image and the pipeline's does not pay the same
price for it: 12.58 MB at 0.18 GB/s is 70 ms, and *which* 0.18 is itself a
property of how much the program has allocated. Stage 9's
[note](stage-9-head-and-tail.md) has the mechanism.

Measured with `go run ./cmd/vaebench -convladder -size 128` (GPU timestamps,
best of two then best of three, the two runs agreeing to 1.005 on the winner),
`go run ./cmd/vaeprof -size 128` for the distribution, `go run ./cmd/vaebench
-sizes 16,32,64,128` and `go run ./cmd/zimage -reps 2` for wall clock (15.00 s
and 15.04 s).

## Why this one was different

Every optimisation in stages 3 through 7 was the same move: an operator that
*is* a GEMM, running on a kernel that did not know it. conv3x3 was not. Stage
2b's `vae_conv2d.comp` is a good direct convolution — IDEAS §2.1's register
blocking applied to a different operator, 6.1x over the version before it —
and it still only reached **3.0-3.2 TFLOP/s against a 55.5 TFLOP/s fp16
ceiling**, because a thread holding `OC_BLOCK` accumulators and reading one
activation at a time is doing scalar FMAs. The matrix cores were never
involved. What had to be written was the implicit GEMM:

    C[OC, H*W] = A[OC, taps*C] * B[taps*C, H*W]

with `A` the filter (a reshape, nothing moves) and `B` the im2col patch
matrix, which cannot be materialised — at 1024x1024 with 256 channels it is
**4.7 GB**, against the 0.5 GB the activation already is. So `B` has to be
*addressed* rather than built, and that is the whole of the design problem.

## The layout, which is the finding

Every WMMA stage in this engine has ended at the same place: a 16x16 fp16
fragment load covers 512 bytes, and §5.1b prices it at `C/gcd(stride, 4096)`
of the bus unless those 512 bytes are contiguous. Stage 3c measured the
difference at **1.56x against 30x**. So the question for conv is: what layout
makes a patch fragment contiguous?

A patch fragment is 16 input channels by 16 output pixels at one tap:

    B[k, n] = act[c0 + k, oh + dh, ow0 + n + dw]

Two candidate layouts fail, and they fail for the same reason:

- **NCHW as it stands.** The 16 channels are `H*W` apart — 4 MB at 1024x1024
  — so the fragment is sixteen 32-byte chunks a stride apart, `32/4096` of
  the bus.
- **Tiling the pixel axis in 16s**, which is what every other operand in this
  engine does. It makes the aligned window contiguous and the **`dw = ±1`
  window straddle two tiles**, which a fragment load cannot express. The
  horizontal tap shift is not an edge case here; it is two thirds of the taps.

What works is to tile the *channel* axis and leave the pixel axis alone:

    [ceil(C/16)][H+2][W+2][16]

16 channels contiguous per pixel, pixels consecutive within a padded row. Now
a patch fragment is 16 pixels of 32 contiguous bytes at 32-byte spacing —
**one fully contiguous 512 B block, at any pixel offset** — which is exactly
a column-major `coopMatLoad` of stride 16. The `±1` shift is an add. The
filter gets the same treatment on the other side (16x16 tiles, packed once at
load, k = `tap*C + ic` so a k-tile is 16 channels at one tap), so **both
operands of every MMA are a single fully covered 512 B read**, and the kernel
has no LDS in its inner loop at all.

This is the third distinct shape the "store operands as fragment tiles" rule
has taken — §2.8's tiled weight, stage 3c's tiled activation, and now a
*blocked channel axis* — and the conv case is the one where the tiling has to
be chosen against an access pattern rather than for one.

### The border is what makes the taps branchless

The packed layout carries a one-pixel zero frame, so every address the kernel
forms holds either a real activation or a zero: the convolution's padding is
**in the data**, and the nine taps are nine adds with no masking, no `if` and
no K-axis bounds check. It costs `(H+2)(W+2)/HW`, which at 1024x1024 is 0.4%.

The same trick pads the channel axis to 16 with zeros, which turns `conv_in`'s
16 input channels into a padded K rather than a special case.

## What it cost: the pack

The blocked layout is not what the rest of the decoder uses, so each
convolution is preceded by a pass that writes it. That pass is now **160 ms,
20% of the decode** — 35 packs moving 16.04 GB at 99 GB/s, or 42% of the bus.
It is the honest price of the layout and it is the next item (below).

Two things it is not: it is not the fp16 narrowing (that is free, the write is
half the read), and it is not avoidable by packing once — a convolution's
input is produced by the dispatch before it.

## The ladder

`BM x BN` is the workgroup's output tile, output channels by pixels. The two
knobs mean different things from the DiT's: **BM sets how many times the
activation is streamed** (once per `ceil(OC/BM)`), **BN how much of the filter
slab a workgroup amortises**.

| build | conv | TFLOP/s | pack | conv+pack | decode |
|---|---|---|---|---|---|
| scalar (stage 2b) | 3.184 s | 3.1 | — | 3.184 s | 3.568 s |
| 64x64 | 310.1 ms | 31.9 | 159.0 ms | 469.1 ms | 851 ms |
| 64x64, BK=2 | 308.6 ms | 32.0 | 158.9 ms | 467.5 ms | 848 ms |
| 64x128 | 297.7 ms | 33.2 | 160.0 ms | 457.7 ms | 840 ms |
| 128x64 | 275.7 ms | 35.9 | 159.1 ms | 434.8 ms | 818 ms |
| 128x128 | 280.3 ms | 35.3 | 161.3 ms | 441.6 ms | 826 ms |
| 256x64 | 327.7 ms | 30.2 | 160.5 ms | 488.2 ms | 874 ms |
| 64x64 wave32 | 397.3 ms | 24.9 | 160.6 ms | 557.9 ms | 940 ms |
| **128x64 wave32** | **258.3 ms** | **38.3** | 159.0 ms | **417.3 ms** | **801 ms** |
| 128x128 wave32 | 258.5 ms | 38.3 | 159.1 ms | 417.6 ms | 801 ms |

Four things in that table:

1. **BM matters and BN barely does.** 64→128 in the M direction is 1.13x;
   64→128 in the N direction is 1.04x. That is the activation-streaming
   argument showing up as a measurement: at OC=256 a BM of 64 reads the whole
   input four times and a BM of 128 twice.
2. **256x64 is a loss**, and it is the first row where BM stops paying: OC is
   128 at the largest shapes in this decode, so a 256-row tile is half empty
   and the grid loses half its parallelism to get there.
3. **wave32 again, a fifth time** (§6.2) — but only above 64 rows. At 128x64
   it is 1.07x, at 128x128 1.08x, and at 64x64 it is **0.78x**, a loss. Two
   things change together there and the measurement does not separate them: a
   wave32 accumulator tile costs 8 VGPRs a lane instead of 4, so every wave32
   build asks for 192 VGPRs against 144 and gets 8 subgroups per SIMD against
   10 — and the 64x64 wave32 arm is the only build in the ladder whose
   *workgroup* is a single 32-thread wave.
4. **The K-slab knob does nothing** (BK=2 is 1.00x). Both operands are already
   fully covered 512 B loads, so §2.7's mechanism — more bytes of one row in
   flight — has nothing left to fix. That is the same result stage 3c got for
   the same reason, and it is worth stating as a rule: **§2.7 is a fix for an
   under-covered load, not a general lever.**

The default is 128x64 wave32 rather than 128x128 wave32 because they tie at
1024x1024 and a 64-wide tile leaves fewer waves idle at the small ones.

No build spills: 144 VGPRs at wave64, 192 at wave32, 1-4 KB of LDS (the
masked-store scratch, below), 10 subgroups per SIMD.

### One thing that looks like a tiling problem and is not

`conv_out` is 128 channels to **three**, so a 128-row workgroup tile is 42x
taller than the answer and the dispatch measures **6.88 ms at 1053 GFLOP/s** —
by far the worst efficiency in the decode. Skipping the waves whose whole row
band is past `OC` (legal here: the kernel has no workgroup barrier) measured
**6.82 ms**, i.e. nothing. That dispatch is bound by streaming its 270 MB
input, not by the multiplies it wastes, so the fix was reverted rather than
kept for its story. It is 0.9% of the decode either way.

## The bug the odd size caught

The kernel clamps every pixel address to the last whole fragment of its
channel plane, so that nothing it forms can leave the buffer whatever `W` and
`BN` are. The reasoning was that the clamp lands on the bottom border row,
which is zeros, and that anything it displaces belongs to an output column
past `W` and is masked away at the store.

That is true **when the padded row is at least 16 pixels wide**. At the
12x12 latent `TestGPUConvOddSize` decodes, the first convolution's rows are
14 pixels, so "the last 16 pixels of the plane" reaches back into the last
*real* row — and the last output row got two columns of somebody else's
pixels. Every build failed identically, at the same element, which is what
said the bug was in the shared addressing rather than in a tiling.

The fix is one `max`: the packed row width is `max(W+2, 16)`. It costs
nothing at any real size (1026 stays 1026) and it is the kind of invariant
that only a test at a deliberately awkward size can find — the 1024x1024
image cannot have this bug, and neither can the 16x16 reference latent used
by every other test in the package.

## The masked epilogue, and why it is not a fast path with a hole in it

Output tiles that hang off an edge — `OC` not a multiple of `BM`, `W` not a
multiple of `BN` — go through LDS and are written element by element, with
subgroup-scoped sync because the branch is wave-uniform and not
workgroup-uniform. Both cases are real: `conv_out` has **three** output
channels, and at the 16x16 reference latent `W` is as small as 16.

The point of paying for it is that **every convolution in the decoder now runs
on this kernel**, including those two. A fast path that the tests cannot reach
is a fast path nothing checks, and stage 8's own bug was found at 96x96.

## The controls

Two, and both fire:

| control | what it breaks | image | |
|---|---|---|---|
| `no_tap_shift` | drops the horizontal tap offset | 4.69 | 312x the bound |
| `pad_clamp` | replicates the edge pixel instead of zeroing the border | 4.07 | 271x |

`no_tap_shift` is the control for the layout's central claim — a 16-pixel
window that starts at an arbitrary pixel — and a kernel that had quietly
rounded that window down to a tile boundary would compute exactly it.

`pad_clamp` was the interesting one to write, because it was not obvious it
would be caught: it touches only the frame, 6% of the 128x128 image this test
decodes and 0.4% of a 1024x1024 one. It lands 271x outside the bound anyway.
That is the opposite of stage 7's result, where `NO_CROSS_WAVE` threw away
three quarters of every dot product and decoded the same image — and the
difference is the model, not the test. A VAE decoder's border is not
information-bottlenecked the way its mid-block softmax is.

## What fp16 costs here

| comparison | rel |
|---|---|
| fp32 GPU graph vs diffusers | 1.5e-5 |
| **fp16 convolutions vs the fp32 graph** | **8.2e-3** |
| fp16 convolutions + fp16 mid block vs diffusers | 3.3e-3 |
| at a 96x96 image (12x12 latent) | 7.5e-3 |
| the pipeline's 256x256 end-to-end number vs diffusers | **0.0252, unchanged** |

8.2e-3 is a chain and not a step: **40-odd convolutions in series, each
rounding its input to fp16 before reading it**, where stage 7's 8.2e-4 was one
block. The bound is `max|got-want| / rms`, a max over elements normalised by
the tensor, which is why the third row is *smaller* than the second — a different mid block makes a
different pixel the worst one, and the two narrowings are not additive.

The precondition is asserted rather than remembered. `TestConvInputsFitFP16`
runs the real reference latent through the CPU decoder and measures every
convolution's input: **35 of them, worst 184, which is 356x inside fp16's
65504.** `TestConvWeightsFitFP16` does the same for the 48.5 M filter weights
(worst 1.91). Stage 2's "absmax 497 through the whole decoder, 1.16e7 in the
mid block's scores" is now a test that fails if a checkpoint arrives whose
activations grew.

## A bump allocator that never gave anything back

Unrelated to the matrix cores and found by looking at the fp16 arena: the
packed activations are allocated and freed one at a time in *increasing* size
— 17 MB at 128x128, then 68, then 270, then 539 — and `arena.release` put each
freed block on the free list, where only a request of its own size or smaller
could ever use it. So the arena was the **sum** of them, 933 MB, instead of
the largest, 539 MB.

Four lines fix it: anything that ends at the bump pointer goes back to the
bump pointer. The fp32 arena gains 168 MB from the same change (3.12 GB →
2.95 GB), which matters for a different reason — a single Vulkan storage
buffer on this device tops out at 4.29 GB.

## What the decode is now, which is a different shape entirely

| kind | scalar conv | matrix-core conv |
|---|---|---|
| conv3x3 | 3.017 s, **84.8%** | 244 ms, 30.5% |
| groupnorm | 246 ms, 6.9% | 244 ms, **30.4%** |
| packconv | — | 160 ms, 20.0% |
| conv1x1 | 153 ms, 4.3% | 12 ms, 1.5% |
| silu | 59 ms, 1.7% | 61 ms, 7.6% |
| add | 42 ms, 1.2% | 40 ms, 5.0% |
| mid-block attention | 15 ms, 0.4% | 15 ms, 1.9% |
| **total** | **3.557 s** | **801 ms** |

For seven stages the answer has been "find the arithmetic and put it on the
matrix cores". There is no arithmetic left in this decoder: after conv3x3 the
largest item is **a group norm**, and the four bandwidth-bound passes —
groupnorm, pack, silu, add — are **63%** of what remains. Every one of them
reads a tensor and writes a tensor, and they run back to back:

    groupnorm -> silu -> pack -> conv -> groupnorm -> silu -> pack -> conv

Four passes over the same 1 GB where one would do. The next item in this
decoder is not a kernel, it is **fusion**, and the most valuable single piece
of it is visible in that chain: a SiLU whose only consumer is a convolution
could write the blocked fp16 form *instead of* its fp32 output, which removes
a 1 GB write and a 1 GB read at once. On the numbers above that is worth
~200 ms of a 801 ms decode, and it is worth more than making any of these
kernels individually faster.

Below that, the pack's own 99 GB/s against a ~235 GB/s bus is 2x of headroom
in a pass that is pure streaming on both sides; the likely cause is that each
thread has one load in flight per channel of the 16-channel LDS stage.

And in the image as a whole the VAE is no longer the question: at 805 ms it is
**5.4%** of a 15.0 s image, against the DiT's 94%.
