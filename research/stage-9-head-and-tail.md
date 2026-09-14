# Stage 9 — the transformer's head and tail on the device

**Result: the denoising loop no longer runs any model arithmetic on the host.**
The patch embedder and the final layer moved onto the matrix cores, the host's
share of a step went from **59 ms to 5-10 ms**, a step from 1.75 s to
**1.70 s** and an image from **14.89 s to 14.49 s**.

It is a smaller win than the item was priced at, and *why* is the more useful
half of this stage: the 1.24 s `PIPELINE.md` predicted was 0.44 s of host
arithmetic (real, and recovered) plus 0.8 s of "the residual stream crossing
the bus" (not real — it is 0.07 s). That second number had never been
measured. It was a subtraction, and the thing it was subtracting from was
wrong by 83x for a reason nothing in the engine knew about.

| | host head | device head |
|---|---|---|
| host per step | 59.0 ms | **5-10 ms** |
| one step, wall | 1.75 s | **1.70 s** |
| an image, 1024², 8 steps | 14.89 s | **14.49 s**, 14.54 s |

Both columns are the same binary: `go run ./cmd/zimage -cpuhead` keeps the
host path, which is stage 6's and is the oracle the device path is compared
against.

## What moved

Per step, the host used to do this: embed 4096 patches through a
[3840, 64] linear, write the [4096, 3840] fp32 stream across the bus, and at
the other end read a [4128, 3840] stream back, LayerNorm it, scale it, and
project it back down to [4096, 64]. Two GEMMs of 2.0 GFLOP each and 126 MB of
traffic, for a latent that is 1 MB.

Now the host patchifies (index arithmetic on a megabyte), uploads **1 MB**,
and reads **1 MB** back. Everything between is four dispatches on the device:
the embedder's GEMM, the final layer's own adaLN projection, a
LayerNorm-scale-narrow pass, and the tail's GEMM.

Neither GEMM needed a kernel. `dit_gemm.comp` already covers both shapes; what
they needed was two rungs of its ladder that the block never uses, because the
constraint at the head is not a tile's efficiency but a tile's *divisibility*:
N must divide 3840 at one end and 64 at the other, and M must divide a token
count that is only ever guaranteed to be a multiple of `SeqMultiOf` = 32. A
16-row tile satisfies that at both ends, so `reg16x256_bt16` and
`reg16x64_bt16` are the two builds, and neither choice is a tuning result.

## The bias and the pad token are columns of K

`dit_gemm.comp` computes `A*B` and has no bias. Stage 7's rule — put a bias in
a pass that is already reading the tensor — has nowhere to go at the head: the
patch embedder's output is read next by the block graph, on the device, in a
kernel that must not learn about it.

So the bias became **an extra column of K**. A's column 64 holds 1 for every
real token; B's row 64 holds the embedder's bias. The multiply-accumulate
that was going to run anyway adds `1 * b`.

The same trick then does something the host used to do with a memcpy. A
padded image stream's rows past the last real token must carry the learned
`x_pad_token`, and diffusers writes them after the embedder. Here **column 65
holds 1 for a padded row and 0 for a real one**, against a B row holding
`x_pad_token`, so `y = Wx + b` and `y = x_pad_token` come out of the same
GEMM with no branch, no second dispatch and no host write. The cost is K going
from 64 to 128 — the slab is 64 wide, so the padding was free anyway — which
is 2 GFLOP on a 1.8 s step.

The tail's bias went the other way, back to stage 7's rule: its output is 64
values per token and the host reads them to unpatchify, so the host adds the
bias while it is copying. Two biases, two mechanisms, and the difference
between them is only *who was already touching the tensor*.

## The one shader this stage needed

`dit_final_norm.comp`: LayerNorm, the adaLN scale, and the narrowing into the
fp16 arena, in one pass over the residual stream instead of three.

It is `dit_norm_scale_f16.comp`'s shape with one difference that makes it a
different file rather than a flag — **the mean is subtracted**. Every other
norm in this model is an RMS norm, and diffusers' final layer is a LayerNorm
with `elementwise_affine=False` and its own epsilon (1e-6, not the config's
1e-5). `head.go` already listed that as one of the four conventions that
produce a plausible tensor when they are wrong.

It reduces twice — the mean, then the variance about it — rather than using
`E[x²] - mean²`, which reads the row once less and cancels catastrophically
when the mean approaches the RMS. That is the CPU reference's arithmetic, and
this tensor is a residual stream after 34 blocks; there is no reason to assume
anything about its mean.

## What fp16 costs, and a bound that had to be earned

| | rel |
|---|---|
| the embedder against diffusers' `x_prepared` | **3.4e-3** (bound 6e-3) |
| the same at a token count that needs padding, against the host | 3.4e-3 |
| the tail against diffusers' `final` | **8.6e-4** (bound 2e-3) |
| the `NO_MEAN` control | 4.4e-2, **22x the bound** |
| the whole pipeline against diffusers at 256² | 0.0252 → **0.0158** |
| the device head against the host head, whole image | latent 0.0135, image 0.0187 |

The two bounds are different and the difference is the finding. Both GEMMs
narrow A to fp16 and accumulate in fp32, but the embedder reduces over K = 66
into an output whose RMS is 0.41, so what it measures is almost entirely the
rounding of the *patches themselves*; the tail reduces over 3840 into an
output whose RMS is 11, and the same per-element rounding averages away.

Inheriting the block's `fp16RelTol` (1.5e-2) would have left the `NO_MEAN`
control **3x** outside the bound, which is not a control. Setting each bound
from its own measurement puts it at 22x. The host's version of the same break
lands 219x outside the fp32 bound in `head_test.go` — same 4.4e-2, a different
denominator — and that is what a tolerance inherited from a looser path costs:
not a wrong answer, a control that has stopped controlling.

The end-to-end number against diffusers *improved*, 0.0252 to 0.0158. That is
not an accuracy claim. A denoising trajectory is chaotic and the bound is a
relative L2 over eight steps of feedback; what it says is that the device head
is well inside the composition's tolerance, not that it is better than fp32.

## The read-back number the engine had been carrying was wrong here

Since stage 3c the engine has repeated one measurement: *a host read of the
device-local host-visible arena runs at 0.2 GB/s, so any wall clock with a
read-back in it is measuring the read-back*. It is cited in four commands and
three research notes, and stage 6 used it to price this stage's other half at
0.8 s an image.

The pipeline's 63 MB read-back takes **6.5 ms**. Not 344.

`cmd/bus` is the probe that settles it. `vk.NewBuffer` asks for
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` and falls back to
`HOST_VISIBLE | HOST_COHERENT` when that heap cannot serve the request — and
on this device the first heap's budget is about **8 GB**, against the 83.79
GiB it reports. The fallback is ordinary cached system memory:

| allocated first | 63 MB write | 63 MB read |
|---|---|---|
| 0 GB | 2.3 ms, 29 GB/s | 361 ms, **0.18 GB/s** |
| 6 GB | 2.3 ms, 29 GB/s | 363 ms, 0.18 GB/s |
| **7 GB** | 2.4 ms, 28 GB/s | **4.7 ms, 14.1 GB/s** |
| 8 GB | 2.2 ms, 30 GB/s | 4.3 ms, 15.4 GB/s |
| 20 GB | 2.1 ms, 32 GB/s | 4.7 ms, 14.0 GB/s |

Writes do not move, because they were already going through write-combining
buffers. Reads move **83x**, at a threshold nothing in the program can see.

So a single-stage benchmark and the real pipeline are on opposite sides of a
line: the pipeline has 20.5 GB of weights resident before it allocates an
activation arena, and every read-back it does is a fast one. Two things this
explains, both already in the repo's own numbers:

- **`cmd/vaebench` reports a 1024² decode at 876 ms and the pipeline reports
  805.** The difference is 71 ms; the image it reads back is 12.58 MB, and
  12.58 MB at 0.18 GB/s is 70 ms. `vaebench` allocates 3.5 GB and is on the
  slow side of the line, the pipeline is not. Stage 8's wall-clock column is
  that 876, and 805 is what the decode costs in an image.
- **The 0.8 s of "stream crossing the bus" never existed.** It was the
  remainder of `1.76 - 1.60 - 0.058` attributed to the bus by elimination,
  and the actual traffic is 2.3 ms of upload and 6.5 ms of read.

The rule that survives is narrower and worth stating exactly: **a read-back
costs 0.18 GB/s only while your program is small enough to still be in the
device-local heap.** The rule that does not survive is using it to price
anything in the pipeline.

## What is left in a step, and what is not

A step is 1.70 s of wall clock. The host is 5-10 ms of it, the bus 9 ms, and
the 34 blocks the rest. GPU timestamps put those blocks at 1.60 s
(`cmd/ditstack`), so ~100 ms is unaccounted for — and it is **not** submit
batching: raising `perSubmit` from 8 to 18 (one submit per block) and to 64
measures 1.69 s and 1.68 s against 1.70, which is inside the noise. That is a
measurement rather than a theory, which is more than the 0.8 s had.

What the loop has left is the DiT itself: 72% GEMM at 73-76% of the WMMA
ceiling, ~1.0 s an image of pure layout inside the block (`pack v` and
`narrow ctx`, both of which a kernel epilogue could absorb), and attention at
38 TFLOP/s. There is no boundary work left to remove.
