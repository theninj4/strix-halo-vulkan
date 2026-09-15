# T4 — kokoro's vocoder on Vulkan

*The whole of `Kokoro-82M`'s `Decoder` — the generator's eight Snake residual
blocks and two upsamplers (T4a, T4b), then the five AdaIN blocks before them,
`conv_post` and the inverse short-time transform (T4c) — ported from the CPU
reference to the device. SPEECH.md's T4; the code is `kokoro/gpu.go` and
`kokoro/gpudec.go`, the `shaders/kokoro_*` family, and `A_CONV` in
`dit_gemm.comp`. Validated by `kokoro/gpu_test.go` and
`kokoro/gpudec_test.go`.*

**Headline: the vocoder goes from 3626 ms on the CPU to 38 ms, and an
utterance from 0.90x real time to 12.9x. Nothing between the decoder's first
block and the waveform crosses the bus.**

| | CPU (T3) | T4a | T4b | T4c |
|---|---|---|---|---|
| decoder | 132 ms | 132 ms | 132 ms | **3 ms** |
| generator, two stages | 3160 ms | 355 ms | 62 ms | **13 ms** |
| tail + excitation | 46 ms | 46 ms | 46 ms | **22 ms** |
| vocoder | 3626 ms | 487 ms | 244 ms | **38 ms** |
| real time | 0.90x | 3.9x | 7.28x | **12.90x** |

## T4a — the convolution is a layout

Every convolution in a generator residual block is a k-tap filter over a
channel-last `[T, C]` activation — k of 3, 7 or 11, dilations of 1, 3 and 5,
C of 128 or 256. Written out that is the GEMM

    C[t, o] = sum over j, c of A[t, j*C + c] * B[o, j*C + c]
    with     A[t, j*C + c] = x[t + j*d - p, c]

so the A address is the ordinary one **plus `j*(d*lda)`**, and because C is a
power of two, j and c are a shift and a mask of the K index. One 16-wide
fragment therefore lies entirely inside one tap. `A_CONV=1` is twelve lines
in `dit_gemm.comp`: **the im2col is an addressing rule, not a pass**, so a
dilated convolution is one dispatch with no packing and no accumulate flag.
The padding is in the data — a 32-frame zero border at each end of the fp16
arena, with `inOff` at row `border - p` — which is stage 8's conv2d trick one
dimension down, and it leaves the kernel branchless. Every existing build of
`dit_gemm.comp` is byte-for-byte unchanged. **The eight blocks went from
2669 ms to 6.9 ms, 22.4 TFLOP/s.**

Two things are free because they are per utterance, not per frame. Each
AdaIN's `fc` is a `[2C, 128]` projection of one style vector, so the host
computes gamma and beta once and 48 of the 70 `fc` layers leave the graph.
And `conv1`'s bias cancels identically inside the AdaIN that follows it —
dropping all three in a block moves the output by 2.7e-7 relative, against
0.031 for the same experiment on `conv2`, whose output reaches the residual
add.

**AdaIN reduces down a column and is still coalesced**, because the workgroup
is laid out across channels rather than along the column: 256 threads cover
`256/C` row groups, so a wave's addresses are one contiguous run of a row.
The ladder's winner is `reg32x64_w32` at both rates by 1.10-1.28x, which is
§6.2's wave32 result holding on a fourth set of shapes.

## T4b — the transposed convolution is one GEMM

Both upsamplers have `kernel = 2*stride` and `padding = stride/2`, so every
output frame is reached by exactly two taps and which two is the residue of
the frame index:

    output frame q*s + r  =  sum_i x[q-1, i]*W[i, n, r+s] + x[q, i]*W[i, n, r]

The obvious reading is s GEMMs writing a strided subset of the rows, which
needs a store stride the kernel does not have. The other reading is **one**
GEMM whose N is `s*C_out`: column `r*C_out + n` of a `[T+1, s*C_out]` result
*is* output frame `q*s + r`, so the same bytes read as `[(T+1)*s, C_out]` are
already the upsampled signal — offset by `stride/2`, which is a pointer. No
strided store, no residue loop, no second kernel, and the A operand is the
existing `A_CONV` addressing at two taps with the arena's zero border
supplying `x[-1]` and `x[T]`.

**The reflection pad is an index map, not a copy.** Folded into the epilogue
that was already adding the bias and the excitation, it is
`src = t == 0 ? 1 : t - 1` — one pass over the tensor for what the reference
does in three. With the upsampler on the device a stage is one upload of its
input, one of the excitation projection, and one download of its output.

## T4c — the decoder, the tail, and the last readback

**The decoder alone goes from 132 ms to 0.74 ms of GPU time.**

| | T4b | T4c |
|---|---|---|
| decoder | 132 ms | **3 ms** (0.74 ms of it on the device) |
| generator, two stages | 62 ms | **13 ms** |
| excitation + STFT (host) | 24 ms | 14 ms |
| tail — conv_post, iSTFT | 22 ms | **8 ms** |
| vocoder | 244 ms | **38 ms** |
| whole utterance | 446 ms | **252 ms** |
| real time | 7.28x | **12.90x** |
| against the CPU path | — | 59.1 dB |
| against the reference dump | 18.2 dB | **18.2 dB** |

The last row is the one that says it is correct. 18.2 dB is T3's measured
floor — the reference's own excitation phases are undefined wherever its
magnitude is zero, and with the dump's noise switched off a quarter of them
are — so *matching the CPU path's distance from torch exactly* is the
strongest statement available. The device is not 0.01 dB further away.

## Three things kept the decoder on the host, and only one was interesting

The blocks are the same shape of thing the generator's are: k-tap
convolutions over a channel-last `[T, C]` activation with an AdaIN between
each pair, which T4a established is one dispatch on `dit_gemm.comp`'s
`A_CONV` build. Two of the three differences were bookkeeping — a leaky
rectifier where the generator has a Snake, and a shortcut branch averaged in
by `1/sqrt(2)` rather than added. The third was addressing.

**A_CONV=1 reads the tap index off the K index with a shift and a mask**,
which works because the generator's channel counts are 128 and 256. The
decoder's are **514, 1090, 1024 and 512**. The literal fix is `K/C` and
`K%C`, and it is a bad one: RDNA has no scalar integer division, so a
non-constant divisor becomes a float-reciprocal sequence of about ten VALU
instructions — inside the K loop, which for these shapes runs 54 times per
workgroup against 32 MMAs per iteration.

**A_CONV=2 does not divide at all.** The host pads the row stride up to a
multiple of BK (the 64-wide K slab), so a slab never straddles a tap
boundary; then the tap's row offset and the column inside it are *carried*
across the loop, advancing by BK and rolling over into the next tap exactly
when the column offset reaches the padded width. A compare and a subtract.
The inner loop is byte-for-byte the cost of A_CONV=1.

    aCol += BK;
    if (aCol >= pc.aux0) { aCol -= pc.aux0; aTap += pc.aux1; }

Padding to 64 rather than to a power of two is the other half. 1090 rounds
to **1152**, a 5.7% widening; rounding to 2048 would have been 88% more
arithmetic for the same kernel.

## The concatenation is a row stride

Upstream hands every decode block its own output *concatenated* with the
phoneme residual and the two curves — `torch.cat` four times over a
`[130, 1090]` tensor, three quarters of which is the same 66 columns every
time. On the device those 66 columns are written once into
`[1024, 1090)` of the running arena and each block's epilogue writes its 1024
beside them. There is no concatenation pass, and the four blocks read one
buffer.

This is the same observation the length regulator gave in T2 (a gather, not a
matmul) and the transposed convolution gave in T4b (a column permutation, not
a strided store): **the shape change is an addressing rule.** Three times
now, in one model.

## Where the decoder's 0.74 ms goes

| | ms | share |
|---|---|---|
| conv1 (K = 3*1152) | 0.281 | 37.8% |
| conv2 (K = 3*1024) | 0.190 | 25.5% |
| conv1x1 (K = 1152) | 0.079 | 10.6% |
| the two AdaIN reductions | 0.142 | 19.2% |
| everything else | 0.048 | 6.5% |

Three quarters is the convolutions, which is what a block should look like.
The ladder disagrees with the generator's, and that is the point of running
it: **`reg32x32_w32` wins here at 0.79 ms against `reg32x64_w32`'s 0.91 and
`reg64`'s 1.43**, where the generator picked the 64-wide tile. The shape is
the other way up — M of 130 or 260 by N of 512 or 1024, against the
generator's M in the thousands by N of 128 — and a short M cannot feed a wide
tile. §6.2's wave32 result holds for a fifth time: both wave32 rungs beat
both wave64 ones.

The AdaIN reduction needed a second formulation. The generator's walks 256
threads across `256/C` row groups, which is coalesced *because* C is 128 or
256 and a power of two; at C = 1090 there is nothing to arrange, so the NPOT
build is one workgroup per 256 consecutive channels of one row and needs no
shared memory and no final reduction. Simpler and faster, and it only exists
because the first one could not be made to address 1090.

## The inverse transform is a gather

The reference inverts each frame, windows it, and accumulates it into a
shared buffer. At `n_fft = 20` and `hop = 5` that is a scatter with four
frames landing on every sample — atomics, or a second pass. Turned around,
**sample i reads the four frames that cover it and evaluates only the one
point of each frame's transform that reaches it**:

    x_t[n] = sum_b Re'[t,b] cos(2*pi*b*n/N) - Im'[t,b] sin(2*pi*b*n/N)

One dispatch, 44 multiply-adds a sample, no atomics. The twiddles are a
`[20, 11]` table of 220 entries that depends on nothing but the geometry, and
the Hermitian factor of two, the `1/N` and `conv_post`'s bias are all folded
into `Re'` and `Im'` by a per-*frame* pass — so the per-sample kernel has no
transcendental in it, and it runs twenty times for every time that pass runs
once.

The window-square envelope is accumulated in the same loop, because it has
the same support. That division is the whole of an iSTFT; the transform is
the easy half.

**20 is not a power of two**, which is why `audio/` has a direct DFT at all.
At this size a table beats any factoring.

## The readback was most of what was left, twice

T4a priced the asymmetry: a `DEVICE_LOCAL|HOST_VISIBLE` buffer writes at
29 GB/s and reads at 154-219 MB/s. T4c paid it three more times and then
stopped.

| what came back | size | cost |
|---|---|---|
| the generator's last stage, `[15601, 128]` | 8.0 MB | **38 ms** |
| stage 0 to stage 1, `[2600, 256]` | 2.5 MB | **14.4 ms** |
| the decoder to stage 0, `[260, 512]` | 0.5 MB | **2.3 ms** |
| the waveform, 78000 samples | 0.3 MB | 1.9 ms |

The first went away with the tail: once `conv_post` and the transform are on
the device, what comes back is the last row of that table instead of the
first. The other two went away with **one fp32 arena shared by the decoder
and both stages**, laid out before any of them is built, so that each
object's output *is* the next one's input — `stage0.aXin` and `dec.aOut` are
the same region under two names, and `UploadInput` has nothing to do.

**A scalar can always be moved into a weight.** Three separate places wanted
to divide by something between two objects, and none of them cost a pass:

- the generator averages its three residual blocks by `1/NumKernels`, and the
  rectifier in front of the next upsampler is positively homogeneous — so the
  division lives in that upsampler's *weights*, and not in its bias, which is
  added after the projection rather than before it.
- the same division, for the last stage, lives in `conv_post`'s weights.
- `conv1`'s bias in every block is followed immediately by an AdaIN, which
  subtracts the per-channel mean over time, so it cancels identically and is
  not staged at all. (The `pool`'s bias is *not* dropped: it enters before a
  convolution whose zero padding makes its contribution differ on the two
  boundary frames, so it is not quite a constant.)

## `conv_post`'s N is 22

Which is not a multiple of the 16-wide fragment tile, and not a multiple of
any rung's BN. It is padded to 32 with zero weight columns — free, because
`packConvBPad` writes into a zeroed bank and the tiles it does not touch stay
zero — and the epilogue reads the 22 that mean anything. That also forces the
rung: the 64- and 128-wide tiles cannot divide 32, so the tail picks the
widest rung that can.

## What this leaves

The vocoder is 38 ms of a 252 ms utterance and **the phoneme side is 214 of
it**. That is six bidirectional LSTMs whose steps are GEMVs at M = 1,
sequential by construction — S8's problem in a different model, wanting a
resident state and the loop's control flow on the device.

The 14 ms of excitation that remains on the host is float64 signal
processing, and it stays there: the phase accumulator reaches 1.3e5 radians
and fp16 cannot represent it at all.
