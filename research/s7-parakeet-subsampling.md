# S7 — the parakeet subsampling stack on Vulkan

*The five convolutions and the linear in front of the FastConformer encoder,
ported from the CPU reference to the device. SPEECH.md's seventh stage; the
code is `parakeet/gpusub.go` and the four `shaders/parakeet_sub_*` files, and
it is validated by `parakeet/gpusub_test.go`.*

**Headline: 92 ms on the host becomes 0.97 ms on the device — 95x — and the
whole encoder goes from 106 ms wall to 20 ms, with the transcript and the
decode trace unchanged. The pipeline for an 11 s clip is 331 ms → 247 ms,
33.2x real time → 44.6x. The choice that produced it is a layout, not a
kernel.**

| | host (S6) | device (S7) |
|---|---|---|
| subsampling | 92.3 ms | **0.97 ms** on the device |
| 24 conformer layers | 13.4 ms | 13.4 ms |
| encoder, end to end (wall) | 106 ms | **20 ms** |
| whole pipeline | 331 ms | **247 ms** |
| transcript | exact | **exact, identical trace** |

2.82 GFLOP in 0.97 ms is **2.9 TFLOP/s**, against the 30 GFLOP/s the host was
getting on the same arithmetic.

## What the stack is

NeMo's `dw_striding`, read off the checkpoint rather than off the collapsed
shape table (which said three plain convolutions and was wrong):

    mel [1101, 128]
      conv 0   dense 1 -> 256, 3x3, stride 2, pad 1, bias   ReLU after
      conv 2   depthwise 256,  3x3, stride 2, pad 1, bias
      conv 3   pointwise 256 -> 256, 1x1, bias              ReLU after
      conv 5   depthwise 256,  3x3, stride 2, pad 1, bias
      conv 6   pointwise 256 -> 256, 1x1, bias              ReLU after
      linear   [T, 256*16] -> [T, 1024], bias
    subsampled [138, 1024]

The tensor shrinks 1101 → 551 → 276 → 138 in time and 128 → 64 → 32 → 16 in
frequency, and the *valid* length shrinks along a second chain that disagrees
with it by a frame at each stage: 1100 → 550 → 275 → 138. Everything past the
valid length is zeroed after each convolution. That is library bookkeeping,
not an arithmetic identity, and it is the reason five of the ten dispatches
carry a masking bound.

Of the 2.82 GFLOP, the linear is 1.16 and the first pointwise convolution
1.16; the dense one is 0.16 and the two depthwise ones together 0.05. So
**three quarters of the stack is already on the GEMM ladder** — if the
feature map is laid out so that it can see it.

## The finding: the layout is the port

Held the way PyTorch holds it — `[C, T, F]`, channel slowest — every operator
here fights the machine:

- the depthwise convolution's 256 lanes are a *plane* apart, so each of nine
  taps is a 256-address gather;
- the 1x1 convolutions are not GEMMs in any useful sense, because the
  reduction axis is the slowest one;
- the flatten into the linear is a transpose.

Held channel-**last**, as `[T, F, C]`, all three become things the engine
already has, and the two kernels that remain are the two the ladder genuinely
did not have:

| operator | channel-last form | cost |
|---|---|---|
| dense 1 → 256, stride 2 | one lane per output channel; the nine mel samples are wave-uniform scalars shared by all 256 lanes | 481 µs |
| depthwise 3x3, stride 2 | 256 lanes walking 256 contiguous floats per tap — one fully covered 1 KB request | 302 + 23 µs |
| pointwise 1x1 | a `[P, 256] x [256, 256]` GEMM over positions, on `dit_gemm.comp`, with **no packing pass** | 40 + 13 µs |
| flatten + linear | a contiguous copy into the A operand, then `[138, 4096] x [4096, 1024]` | 5 + 80 µs |

The last row is the one that could have gone the other way. PyTorch's module
does `h.transpose(1, 2).reshape(...)`, so its feature index is `c*F + f` —
channel slower. A `[T, F, C]` map's feature index is `f*C + c`. One of the two
has to move. Moving the **activation** is a transpose per clip; moving the
**weight** is a permutation of a `[1024, 4096]` matrix done once at upload,
after which `sub flatten` is a 5 µs row copy. This is §2.8's argument about
the B layout in a different costume: a retiling that would cost an activation
a pass per clip costs a weight nothing per clip.

It is also the sharpest failure mode in the stage, because the wrong order is
not an error. The shapes agree, the GEMM runs, and a `[138, 1024]` tensor of
entirely ordinary magnitude comes out. Only the words change — the
`subFlatOrder` control moves the encoder output by **1.22 relative** and
empties the transcript.

## Stage 8's question, with a different answer

`research/stage-8-vae-conv.md` asked where a convolution's fragment should
come from: a patch is contiguous only if the channel axis is the tiled one.
That kernel is stride 1 over 512 channels and answers "tile the channels and
pack the patch".

Here the dense convolution's input has **one** channel. There is nothing to
tile and nothing to reduce over: an output element is nine scalars against
nine weights, and all 256 output channels of a position want the *same* nine
scalars. So the implicit-GEMM framing is the wrong one, and the kernel that
wins is the one that never builds a fragment at all — a lane per output
channel, the nine weights hoisted into registers (`[9, C]` tap-major, so one
tap's 256 channels load in one request), and the mel samples read as
wave-uniform scalars. Same operator family, opposite answer, and the thing
that decided it was the channel count of the *input*.

## The ladder, and wave32 for a third time

The expectation was that the stack's two GEMM shapes would disagree. The
pointwise convolutions are **8832 rows deep** at N = K = 256, which is the one
place in this whole model with enough rows to fill a wide four-wave tile; the
linear is 138 rows, the narrow-M regime `PlanFor` exists for. `TestGPUSubLadder`
times the whole stack once per rung:

    kernel                 pointwise      linear
    reg32x32_bt16_w32       1.00x          1.00x
    reg32x64_bt16_w32       1.00x          1.12x
    reg16x64_bt16_w32       1.11x          1.25x
    reg16x64_bt16           1.42x          1.31x
    reg32x64_bt16           1.17x          1.67x
    reg32x128_bt16          1.32x          1.63x
    reg64_bt16              1.17x          1.90x
    wg128x256_bt16_swz8     1.24x          2.80x

They do not disagree, and the tile that was supposed to own the deep shape
loses it. **Every wave32 rung beats every wave64 rung on both shapes** — §6.2
again, on a third set of dimensions — and the margin at 8832 rows is
1.17-1.24x rather than the 1.28x the conformer measures at 138. Depth does
not rescue the wide tile because N = 256 is only four of its tiles wide: a
128x256 workgroup reads 256 K-deep rows of B to serve 128 rows of A, and the
reuse it was built for is not there.

So `DefaultSubPlan` is `reg32x32_bt16_w32` uniformly, which is what
`DefaultGEMMPlan` is. Three shapes, one tile.

## What is left in the stack, and why it is not being chased

    sub conv0        481 µs   49%
    sub dw1          302 µs   31%
    gemm sub.pw1      40 µs    4%
    sub bias1         23 µs    2%
    sub dw2           23 µs    2%
    gemm sub.pw2      13 µs    1%
    sub bias2          5 µs    1%
    sub flatten        5 µs    1%
    gemm sub.linear   80 µs    8%
    sub scale          3 µs    0%

The inversion is complete: 80% of the stack is now the two kernels that are
0.21 GFLOP of its 2.82, and the three GEMMs that are 2.6 GFLOP of it are 13%.
Both of the expensive ones are bandwidth, not arithmetic — the dense
convolution writes 36 MB (`[551, 64, 256]` fp32) at 75 GB/s and the first
depthwise reads it back. Narrowing that intermediate to fp16 would halve both
and halve the largest arena in the encoder, and it is the obvious next move
*if this stage ever matters again*. It does not yet: the whole stack is 0.4%
of a 247 ms pipeline, of which the TDT decode loop is now 82%.

## Validating it

Six tensors, walked in one run — unlike a conformer layer, whose four
branches share one output buffer, no two stages of this stack alias, so
stopping early buys nothing:

    sub_0       [256 551 64]   7.1e-06 relative
    sub_2       [256 276 32]   2.1e-04
    sub_3       [256 276 32]   1.6e-04
    sub_5       [256 138 16]   3.1e-04
    sub_6       [256 138 16]   2.4e-04
    subsampled  [138 1024]     3.4e-04

The first row is the one worth reading: the dense convolution is the only
stage here with no fp16 operand anywhere in it, and it lands at 7e-6 — which
is the Go front end's own drift from the reference mel, and nothing else. The
rest sit at 2-3e-4, which is the fp16 A operand of the GEMM that follows.

The end-to-end assertion is S6's, for S6's reason — fp16 moves the encoder
output by percent and the transducer does not care, so the trace is the bound
and the tensors locate a fault rather than certify its absence. `ApplyMel`
produces the reference's 46 emissions at the same frames with the same
durations, and sits **4.6e-04 relative** to the same graph run with the host
stack in front of it, which is the comparison that isolates S7 from S6.

## The negative control

Two breakages, each required to move the encoder output by 10x the intact
run's 2.2e-3:

| control | what it breaks | effect | transcript |
|---|---|---|---|
| `subFlatOrder` | the linear reads the feature axis channel-slower | **1.22** relative | **empty** |
| `subNoMask` | padding survives all three stride-2 convolutions | **3.1e-2** relative | unchanged |

The second is S6's lesson holding for a third time: a control that changes the
encoder output by 14x the drift leaves the words alone. Three of the six
controls this model now has do that. The transcript says the port is right; it
is not sensitive enough to say a kernel is.

## What this leaves

The encoder is finished. 177 GFLOP of layers in 13.4 ms and 2.8 GFLOP of
convolutions in 0.97 ms, 20 ms wall from mel frames to hidden states, against
2.68 s on the CPU. What remains in the pipeline is entirely the ends:

    front end     25 ms   10%   (host FFTs)
    encoder       20 ms    8%
    projector      2 ms    1%
    decode       207 ms   82%   <- S8

and the decode loop is not an arithmetic problem. The joint's `[8198, 640]`
head measures 1.6 TFLOP/s as a GEMV — 6.5 µs a step, 0.3 ms for the clip's 46
emissions. It is 46 round trips between the host and a model that lives on the
device.
