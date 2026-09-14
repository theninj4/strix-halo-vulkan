# S6 — the parakeet encoder on Vulkan

*The FastConformer stack of `parakeet-tdt-0.6b-v3`, ported from the CPU
reference to the device. SPEECH.md's sixth stage; the code is
`parakeet/gpu.go`, `parakeet/gpugraph.go` and the eight `shaders/parakeet_*`
files, and it is validated by `parakeet/gpu_test.go`.*

**Headline: the 24 conformer layers go from 2.45 s on the CPU to 13.8 ms on
the device for an 11 s clip — 178x — and the transcript is unchanged,
emission for emission. What that exposes is the *rest* of the encoder: the
subsampling stack, still on the host, is now 92 ms, or 6.7x everything the
GPU does.**

| | CPU (S5) | GPU (S6) |
|---|---|---|
| subsampling (host, both) | — | 92.3 ms |
| 24 conformer layers | 2.45 s | **13.8 ms** on the device, 14.9 ms wall |
| upload + read-back | — | 0.8 + 2.2 ms |
| encoder, end to end | 2.45 s | **110 ms** |
| transcript | exact | **exact, identical trace** |

177 GFLOP in 13.8 ms is **12.9 TFLOP/s** of useful work, i.e. charged for the
tile padding but not for it.

## The graph

Forty dispatches per layer, 960 for the stack, recorded into four command
buffers. Per layer, in order:

    norm ff1 → gemm ff1.linear1 → silu → gemm ff1.linear2 → x += 0.5·y
    norm attn → gemm q, k, v → pack q(+bias_u), k, v → bias_v narrow
               → gemm rel_k → narrow rel_k → 8 × gemm pos → rel shift
               → attention → gemm o → x += y
    norm conv → gemm conv.pw1 → glu → dwconv → gemm conv.pw2 → x += y
    norm ff2 → gemm ff2.linear1 → silu → gemm ff2.linear2 → x += 0.5·y
    norm out (in place, fp32)

Eleven of those are the DiT's GEMM (`shaders/dit_gemm.comp`) and one is its
attention kernel; the other seven shaders are new and small. The profile at
138 frames, summed over 24 layers:

| kind | n | total | share |
|---|---|---|---|
| `gemm ff1.linear1` / `ff2.linear1` | 48 | 4.0 ms | 29% |
| `gemm ff1.linear2` / `ff2.linear2` | 48 | 4.0 ms | 29% |
| `gemm q/k/v/o` | 96 | 2.1 ms | 15% |
| `gemm conv.pw1` + `conv.pw2` | 48 | 1.4 ms | 10% |
| `gemm rel_k` | 24 | 0.74 ms | 5.4% |
| 8 × `gemm pos` + `rel shift` | 216 | 0.72 ms | 5.2% |
| `attention` | 24 | 0.31 ms | 2.3% |
| `dwconv` | 24 | 0.19 ms | 1.4% |
| the five norms | 120 | 0.47 ms | 3.4% |
| everything else elementwise | 288 | 0.60 ms | 4.4% |

The feed forwards are 58% of it, which is exactly their share of the
arithmetic — so the graph is not spending its time anywhere surprising.

## What the port had to invent

Three of the four are what SPEECH.md predicted; the fourth was not on the
list and is the one that cost a day of being wrong.

**LayerNorm with a mean and an affine** (`parakeet_layernorm.comp`). Five per
layer, and there is no RMS norm in this model at all. It is a third norm
kernel rather than a flag on one of the DiT's two, because the DiT's
mean-subtracting norm has no affine and its affine norm has no mean. Two
reduction passes, as the CPU reference does: the single-pass
`E[x²] − mean²` identity cancels on a residual stream. Four of the five write
the fp16 A operand their branch's first GEMM reads — the narrowing is fused
into the norm, never a pass of its own — and the fifth writes fp32 in place.

**The position term as an additive bias.** Transformer-XL attention is two
score matrices summed, and the second one is computed over *relative offsets*
and folded onto the query/key grid by `shifted[i][j] = raw[i][T−1−i+j]`. The
port does that in three dispatches ahead of the score kernel — the
`[2T−1, 1024]` projection, eight per-head `[T,128]×[2T−1,128]ᵀ` products, and
a shift pass that writes an fp32 `[heads][T, T]` plane — and then the score
kernel *adds a 16×16 tile of that plane to each score tile*
(`dit_attention_wmma.comp`'s `REL_BIAS` build). A cooperative-matrix load
plus an accumulator-to-accumulator add: no second pass over S, and no
materialised score matrix, so the flash structure survives intact. The whole
position term costs 5.2% of the encoder.

The shift is the closed form and not transformers' pad-and-reinterpret, which
is three operations that exist to move data a shader can simply address.

**The convolution branch.** A GLU over the *channel* axis (`parakeet_glu.comp`)
and a 9-tap depthwise convolution over time with the folded BatchNorm and
silu in one pass (`parakeet_dwconv_f16.comp`). The depthwise convolution has
no reduction over channels at all — 9 multiply-adds per element against a
pointwise convolution's 1024 — so what it costs is the access pattern, and
256 lanes walking a 4 KB row make each of the nine taps a covered read. It is
1.4% of the encoder.

**One GEMM whose B operand is an activation.** `rel_k` is projected from the
position embeddings, so it is different every clip and cannot be staged as
fragment tiles. It reads the natural `[N, ldb]` layout out of the *fp16
activation arena* — same shader, a descriptor set that binds the activation
arena where the others bind the weight bank. That is one more pipeline and no
new kernel.

## The bug that only a bias could produce

The first end-to-end run was wrong by 2% rms in the attention context, with a
few rows out by 100%. Everything feeding it — q, k, v, `rel_k`, the raw
position scores, the shifted bias — matched the reference to 3.5e-4.

**The row max was being set by keys that do not exist.** The flash kernel
masks pad keys out of `P`, which removes them from the output and from the
softmax denominator, but it takes the row max *before* that mask — and that
was safe for every model this repository had run, because a pad key scores
exactly 0 and a real key's score is a q·k product of RMS-normalised vectors,
which is never far from 0 either. Here the position bias makes the real
scores tens of log2 units wide (`matrix_bd`'s rms is 13.7 against the content
term's 1). So on any row whose real scores are *all negative* — and there are
many — a key that does not exist set the scale, every real weight became
`exp2(real − 0)`, and in a row where that gap exceeded 24 log2 units the whole
softmax underflowed **fp16's smallest subnormal** and took the denominator to
zero with it.

Masking the max as well as `P` (under `REL_BIAS`, so no other build changes)
takes the error from 2.2e-2 to 2.2e-4 relative — a factor of 100, from one
`break`.

The lesson is not about attention. It is that **the kernel's tail handling
encoded an assumption about the size of a score**, the assumption held for
two models, and nothing in the kernel said so.

## Padding M is not free, and neither alignment is the other

A clip is 138 frames. The first version padded that to 256 — one number for
every alignment in the graph, taken from the widest tile in the ladder — and
so computed 86% more rows than the clip has. Splitting it in two:

- **`rowsRun`**, the GEMM's M, is padded to the *plan's* BM: 160 at the
  wave32 32×32 tile.
- **`planeRows`**, the attention geometry, is padded to the key block (64),
  because the score kernel reads whole key blocks and the packed planes and
  the bias plane have to have those columns.

17.6 ms → 13.8 ms, for two constants and no arithmetic. This is stage 5c's
finding one level down: the tile follows the sequence, and so does everything
sized from it.

## The ladder, and wave32 on a whole model

`TestGPUGEMMLadder` times the entire encoder once per rung at four clip
lengths. It is a ladder over the *graph*, not over a GEMM in isolation, so it
is allowed to disagree with `results/shapes.csv` — that table measures one
shape with its operands hot, and this streams 1.15 GB of weights past the
cores once per clip.

| frames | reg32x32_w32 | reg32x64_w32 | reg16x64_w32 | reg32x64 | reg64 | wg128x256 |
|---|---|---|---|---|---|---|
| 138 (11 s) | **15.7 ms** | 16.4 | 17.0 | 21.0 | 21.8 | 28.8 |
| 384 (31 s) | **28.3** | 29.2 | 30.7 | 33.4 | 32.4 | 36.9 |
| 768 (61 s) | 59.4 | **59.0** | 62.8 | 65.9 | 61.4 | 67.3 |
| 1024 (82 s) | 111.5 | **105.1** | 114.3 | 121.2 | 110.4 | 111.6 |

Two things fall out of it:

1. **Every wave32 rung beats its wave64 twin at every length** — 1.28x at 138
   frames on the identical tile, 1.15x at 1024. §6.2 measured that on a GEMM
   in isolation and stage 3c on one attention kernel; this is the same lever
   applied to a whole model, and it holds.
2. **The DiT's own winner is 1.83x off the pace** at 138 frames, because 86 of
   its 128 rows would be padding. "Use the kernel we already have" costs
   nearly two.

`PlanFor` is those boundaries: the 32×32 tile up to 512 frames, the 32×64 one
above. Getting it wrong costs 1-6% either side, which is why the boundary sits
between measured points.

## Validating a path that fp16 moves more than the bug would

SPEECH.md settled this before a shader was written: narrowing every
matrix-core operand of the *CPU* reference leaves the transcript and the whole
decode trace unchanged while moving the encoder output by 4.3% on a
max-abs-against-rms measure. So that measure cannot certify this port, and the
bound has to be chosen to catch a wrong kernel rather than to catch fp16.

Two changes make the stage walk work:

- **The measure is the rms of the difference over the rms of the tensor**
  (`deviation.Rel`). Max-abs-against-rms reads 2-3% on tensors that are
  correct here: fp16 rounds every element by its own magnitude, and the
  largest element of these tensors is 30x their rms, so that ratio measures
  the dynamic range and not the error.
- **`RunTo` re-uploads the clip.** The residual stream is updated in place, so
  running a prefix over the arena a previous prefix left adds every residual
  in it twice. Nothing about that failure looks like a dispatch being wrong —
  it presented as "the second norm in the layer is 84% off while the first is
  exact".

The measured walk, layer 0, every dispatch against the same reference tensors
the CPU implementation is held to:

| stage | relative | stage | relative |
|---|---|---|---|
| `norm_ff1` | 2.1e-4 | `matrix_bd_raw` | 3.5e-4 |
| `ff1` | 3.6e-4 | `matrix_bd` | 3.5e-4 |
| `resid1` | 5.0e-5 | `attn_ctx` | **9.5e-4** |
| `q` / `k` / `v` | 2.8e-4 / 1.8e-4 / 1.3e-4 | `conv_pw1` | 1.8e-4 |
| `rel_k` | 2.9e-5 | `silu(conv_bn)` | 3.5e-4 |
| `norm_conv` | 2.5e-4 | `layer_out` | 1.7e-4 |

and down the stack: 2.4e-4 at layer 1, 5.6e-4 at layer 12, **2.1e-3 at the
encoder output** — where the last layer's norm divides by a small variance
(rms 0.02 against the stack's 11) and magnifies what came before it.

The widest stage is the attention context, and that is the one tensor whose
inputs have been through a softmax: a score's fp16 error is exponentiated
before it reaches the output.

**The bound that counts is still the transcript**, and it is exact: 46
emissions, identical tokens, frames and durations.

## The negative control, and what the transducer does not notice

Four breakages, each required to move the encoder output by 10x what the
intact graph does. The clip is padded by eight frames, because one of the four
only acts on padding and the fixture has none.

| control | what it breaks | output moves | transcript |
|---|---|---|---|
| centre slice | the position scores read down the middle T columns instead of the diagonal | **0.77** | changed |
| no `bias_u` | the content term loses its learned query bias | **0.46** | changed |
| no pad zero | padding walks back into real frames through the 9-tap kernel | **0.33** | *unchanged* |
| no mean | every LayerNorm becomes an RMS norm | **0.039** | *unchanged* |

The two that leave the words alone are the finding. A 24-layer stack
normalised without its mean moves the encoder output by 3.9% and the
transducer does not care — which is why the mean gets a second check at the
tensor where it acts (layer 0's first norm, whose input is not zero-mean:
1.9e-2 against 2.1e-4 intact, a factor of 90). **The transcript is what says
the port is right; it is not sensitive enough to say a kernel is.**

## What this leaves

**The subsampling stack is now the encoder.** 92.3 ms on the host against
13.8 ms on the device — 87% of the wall clock, for 2.8 GFLOP of the 180. S6's
plan called it "a correctness problem rather than a performance one" because
it runs once per clip; the measurement says otherwise, and it is the next
thing to move. It wants three kernels the ladder does not have — a stride-2
`conv2d` over a single-channel input, the depthwise form of it, and the
`[T, 4096] × [4096, 1024]` linear that follows, which is already a GEMM.

**And then the decode loop is the pipeline.** With the layers at 14 ms, an
11 s clip is front end 20 ms, subsampling 92 ms, encoder 14 ms, projector
2 ms, **decode 205 ms** — the TDT greedy loop, on the host, one 8198-wide
joint per step with an LSTM between. It was 7.7% of the CPU pipeline and it is
62% of this one. Nothing about it is hard; it is simply next.

The read-back is worth 2.2 ms of the 110 and will not survive the projector
moving to the device: what crosses the bus then is 46 emissions' worth of
logits rather than a `[138, 1024]` tensor.
