<!-- LLM.md L3b. The gated DeltaNet on the GPU, five dispatches against about
     twenty, and the scan-versus-chunk question priced. Cited from
     llm/gpu_deltanet.go and shaders/llm_dn_scan.comp. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L3a](l3-deltanet.md) · [L2c](l2c-hc-kernel.md) · [L2f](l2f-attention-gpu.md) · phase 1

# L3b — the gated DeltaNet on the device, and what the scan actually wants

**Result: the layer runs in five dispatches where llama.cpp's graph spends
eleven, and 152.8 ms of its 512-token prefill graph becomes 111.5 — 1.37x.**
The recurrence itself, which is the piece L3a left open, is **413 us a layer
against the reference's 438 — 1.06x** — and getting there took a fifteen-rung
ladder whose two ends are 6.3x apart on identical arithmetic. Two separate
runs give **111.5 and 113.2 ms**, 1.37x and 1.35x, and 413.1 and 410.3 us on
the scan; the tables below are the first of them.

**And the answer to "scan or chunk" is: the question was mis-scoped.** The
recurrence is **13% of this layer**. The fused input projection is **64%**.
L2a-1 said the architecture was innocent of llama.cpp's prefill gap; this says
the same thing one level down — the thing that looked like the hard part of
the hard layer is not where the layer's time is.

Every tensor agrees. Against L3a's CPU reference, from the same input:
`attn_output` at **2.4e-06 rms**, `q_conv_predelta` at 1.2e-04, the layer's
output at 1.1e-03. Against llama.cpp itself: `attn_output` **2.4e-06**,
`linear_attn_out` 1.1e-03. Layers 1 and 2 land in the same place (1.9e-06 on
the recurrence). And **3 + 4 tokens reproduce 7 bit-identically**, output and
state both, with the control showing the convolution's window matters.

## The graph

| | llama.cpp | ours |
|---|---:|---:|
| dispatches per layer | **11** (396 per graph over 36 layers) | **5** (180) |
| time per 512-token graph | **152.8 ms** | **111.5 ms** |
| share of the whole 1164.7 ms graph | 13.1% | **9.6%** |

	qkv    xn -> q, k, v, z, alpha, beta          (the plain GEMM arm)
	conv   depthwise conv + SiLU + L2 norm + the two per-head scalars
	scan   the delta rule, state in registers across the token loop
	norm   the gated RMS norm, into the output matmul's A operand
	out    the output projection                  (the plain GEMM again)

Per 512-token graph, against the lines of llama.cpp's own graph that are
nameably this layer (`results/l2a_prefill_ops.csv`):

| | disp | llama.cpp | disp | ours | |
|---|---:|---:|---:|---:|---:|
| `qkv` | 144 | 91.3 ms | 36 | **71.3 ms** | 1.28x |
| `conv` | 144 | 11.4 ms | 36 | **7.0 ms** | 1.63x |
| `scan` | 36 | 15.8 ms | 36 | **14.9 ms** | 1.06x |
| `norm` | 36 | 8.6 ms | 36 | **2.2 ms** | 3.90x |
| `out` | 36 | 25.7 ms | 36 | **16.1 ms** | 1.60x |
| **layer** | **396** | **152.8 ms** | **180** | **111.5 ms** | **1.37x** |

(Second run: 72.6 / 7.0 / 14.8 / 2.4 / 16.4, **113.2 ms, 1.35x**.)

Two attributions are the host file's rather than the CSV's, and both are
stated in `cmd/llm/bench_deltanet.go` so they can be disagreed with: the
`m=10240 k=2560` line runs 37 times and only 36 are this layer (the 37th is
layer 1's `ple_key`, same shape), and the `m=2560 k=6144` line runs 48 times
and only 36 are `ssm_out` (the other twelve are L2f's `attn_output`, same
shape). What is **not** attributed is the SiLU the convolution carries, the
sigmoid of z, the multiply against it and the CONTs behind all three: they
live in the shared MUL (479), SIGMOID (326) and CONT (251) lines that every
block of the model uses, and every one of those dispatches is **absent** from
our graph rather than faster in it. Leaving them out is the conservative
direction.

## Finding 1 — four of llama.cpp's matrices are one, and two of them are the F32 scandal

`attn_qkv` [2560, 10240], `attn_gate` [2560, 6144], `ssm_alpha` [2560, 48] and
`ssm_beta` [2560, 48] all read the same block input, so they are one
[16512, 2560] weight. Same argument as L2a's for `inject`, L2d's for the PLE
value and L2f's for the attention layer's six matrices, and it is worth 1.28x
on the line.

The two F32 ones are the interesting half. They are L2a's "tiny-N F32 matmul"
in miniature — **72 dispatches a graph, 7.8 ms, for 0.25 MB of weights**,
1155 GFLOP/s — and here they are 96 more columns on a matrix that already has
16384. The 0.25 MB rides along free.

**That puts them on the fp16 matrix cores, which is a deviation from the dump
and not from the reference.** L3a-5 established the rule: `ggml_vk_mul_mat`
takes the f32 *vector* path up to `mul_mat_vec_max_cols = 8` output columns
and the fp16 coopmat GEMM above it. The dump has 7 tokens; any real ubatch has
512. So llama.cpp evaluates alpha and beta in fp16 too, and **L1's
`PPL 4.0340` was measured with it** — this is the reference's arithmetic, seen
for the first time from the side the dump cannot show.

`TestDeltaNetGPUGateIsTheFp16Path` prices it rather than hiding it in a
tolerance, because `gate` is a **log** decay and an absolute error on alpha is
multiplied by |`ssm_a`| — which reaches 23 in layer 0 — before `exp` turns it
into a per-token forgetting factor:

| | rms on `gate` | worst | |
|---|---:|---:|---|
| CPU fp16 vs CPU f32 | 2.15e-03 | 2.16e-02 | what the fp16 path costs |
| GPU vs CPU f32 | **1.23e-03** | 1.11e-02 | ours, and *nearer* than the CPU's emulation |
| GPU vs CPU fp16 | 1.73e-03 | 1.86e-02 | two fp16 orders, not one |

**Worst per-token decay ratio: 1.105%.** The GPU is closer to the f32 model
than the CPU's own fp16 emulation is, because a cooperative-matrix multiply
accumulates in f32 across tiles where `f16Copy` + a sequential dot does not.
So the test asserts what is true — that ours is an fp16 matmul of the same
quality — rather than that it reproduces one particular evaluation order,
which no two implementations share.

## Finding 2 — the scan's ladder is a re-read-bandwidth to ALU crossover, and it is 6.3x end to end

The state is [128, 128] per head and **LPC** is how many lanes own one column
of it. That one number moves three things at once:

- a lane holds **128/LPC** state elements in registers;
- a wave owns **64/LPC** columns, so the layer dispatches **48·128/(cols)**
  workgroups;
- every workgroup of a head re-reads that head's whole q and k row **once per
  token**, so the re-read factor is **128/cols**.

llama.cpp's own kernel is the top rung: `ggml_vk_gated_delta_net` dispatches
`{H, n_seqs, S_v}` with one column per workgroup, which on this device's
64-wide wave is **6144 workgroups and a 128-fold re-read** — not the "48
workgroups" L3a inferred from the source. Fifteen rungs, us per layer at 512
tokens, two runs 1-3% apart:

| rung | regs/lane | workgroups | q/k re-read | operands | us | GFLOP/s | GB/s | vs 438.4 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| l64s | 2 | 6144 | 128x | LDS | 2599 | 1240 | **1252** | 0.17x |
| l64 | 6 | 6144 | 128x | reg | 2766 | 1165 | 1176 | 0.16x |
| l32 | 12 | 3072 | 64x | reg | 1355 | 2378 | **1212** | 0.32x |
| l16s | 8 | 1536 | 32x | LDS | 694 | 4642 | **1206** | 0.63x |
| l16 | 24 | 1536 | 32x | reg | 605 | 5045 | 1311 | 0.72x |
| l8s | 16 | 768 | 16x | LDS | 447 | 7204 | 971 | 0.98x |
| **l8** | **48** | **768** | **16x** | **reg** | **413** | **7892** | 1051 | **1.06x** |
| l4s | 32 | 384 | 8x | LDS | 454 | 7092 | 512 | 0.97x |
| l4 | 96 | 384 | 8x | reg | 602 | 5352 | 387 | 0.73x |
| l2 | 64 | 192 | 4x | LDS | 486 | 6632 | 272 | 0.90x |
| l1 | 128 | 96 | 2x | LDS | 868 | 3710 | 94 | 0.50x |
| l8w4 | 48 | 192 | 4x | reg | 415 | 7773 | 319 | **1.06x** |
| l4w4 | 96 | 96 | 2x | reg | 588 | 5483 | 139 | 0.75x |
| l2w4 | 64 | 48 | 1x | LDS | 678 | 4749 | 84 | 0.65x |

**Five of the rungs are pinned at 1.18-1.31 TB/s while their arithmetic rate
varies 4x**, which is a flat ceiling and not a coincidence: it sits above the
965 GB/s the MALL reads, so it is a cache level this repo has not measured
directly, and it is what the re-read runs into. Those rungs are not computing;
they are re-reading operands they did not need to re-read. The direct ablation says so: **deleting the q,
k and v loads from the l64 rung takes it from 2766 us to 1013**, so 63% of the
widest rung is that traffic. Deleting both subgroup reductions instead
changes it by 5%, so it is not the cross-lane cost.

Coming down the ladder cuts the re-read in half per rung and the time follows
— 2766, 1355, 605 — until at **l8** the arithmetic takes over at 7.9 TFLOP/s
of f32, and below that the register file starts costing more occupancy than
the saved traffic is worth. **6.3x between the two ends, on identical
arithmetic.**

The winner is l8 at every length from 128 to 2048, with l8w4 inside 1-3% of
it, which is inside the run-to-run spread:

| tokens | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|
| l8, us a layer | 52.8 | 99.9 | 199.2 | **413** | 856 | 1767 |
| llama.cpp | — | — | — | **438** | — | 2058 |

At 2048 tokens the reference costs 2057.5 us a layer and ours 1767 — **1.16x**,
a little better than at 512, because the re-read is a *per-token* cost on both
sides and our schedule pays it 16 times where theirs pays it 128.

## Finding 3 — registers beat LDS for the operands, until they do not

Where q and k live is the second dial (`QKREG`), and it is not free either
way. In registers a lane holds 2·128/LPC more floats beside the state; in LDS
they are staged once per token by the whole workgroup and read back, which
costs a barrier per token and an LDS round trip per element.

| LPC | registers | LDS | |
|---:|---:|---:|---|
| 64 | 2766 | **2599** | LDS wins: at 2 state elements the staging is the work |
| 16 | **605** | 694 | 1.15x for registers |
| 8 | **413** | 447 | 1.08x, and this is the winner |
| 4 | 602 | **454** | **1.33x the other way** — 96 registers a lane costs occupancy |

So the crossover is at LPC 8 in both dials at once, and the two arms cross in
*opposite* directions on either side of it. That is the same shape of result
as L2f-4 — one kernel wanting opposite schedules on two weights that differ
only in which side of a cache they fall on — and the same moral: the rung is a
property of the shape, not of the kernel.

Two things that sound like they should help and do not:

- **Sharing the staged operands across four waves of a workgroup** (the `w4`
  arm) cuts the re-read by 4x and buys nothing: l8w4 ties l8, l4w4 ties l4.
  Once the re-read is small enough to stop being the bottleneck, removing more
  of it is free in the wrong direction.
- **A one-deep software pipeline** — issuing token t+1's operands before token
  t's arithmetic — is worth **nothing** at l64 (2775 against 2766) and costs
  1.4x at l8, where the extra 2·RPL registers hurt. The hypothesis it was
  testing, that the write-back to a read-write SSBO stops the compiler
  hoisting loads across it, is **not** what is happening.

## Finding 4 — the recurrence is 13% of this layer, and the projections are 64%

This is the finding that decides what to build next, and it is the reason the
chunked form is written down rather than built.

| dispatch | us a layer at 512 | share |
|---|---:|---:|
| `qkv` | 1982 | **64.0%** |
| `out` | 447 | 14.4% |
| `scan` | **413** | **13.3%** |
| `conv` | 195 | 6.3% |
| `norm` | 61 | 2.0% |

L2a-1 found that `GATED_DELTA_NET` is 1.6% of llama.cpp's prefill graph and
concluded the architecture was innocent. One level down, the same thing is
true of the layer: **the delta rule is not where the linear-attention layer's
time goes.** The fused projection is, and it is already running at 21.8
TFLOP/s against a 39 TFLOP/s best — a matmul question and not a recurrence
one.

### The chunked form, priced

L3a-2 left a real open question: llama.cpp runs a sequential scan and declines
its own `build_delta_net_chunking`; vLLM does the opposite, `chunk_gated_delta_rule`
at chunk size 64. flash-linear-attention's four stages give the arithmetic, so
it can be costed without being built. Per (chunk, head) at C = 64, D = 128:

| stage | shape | MFLOP |
|---|---|---:|
| `chunk_scaled_dot_kkt` | (KβKᵀ), C x C x D | 1.05 |
| `solve_tril` | (I + tril A)⁻¹, C³/3 | 0.18 |
| `chunk_delta_h` | state carry, 2 x D x D x C | 4.20 |
| `chunk_fwd_o` | qH + causal (qkᵀ)v | 4.20 |
| | | **9.63** |

At 512 tokens that is 8 chunks x 48 heads x 9.63 = **3.70 GFLOP** against the
scan's 3.22 — **1.15x the arithmetic, all of it parallel and all of it matmul
shaped.** At this repo's best measured WMMA rate (39.0 TFLOP/s, §2.7) the
chunked layer would be **95 us against the scan's 413, a 4.3x ceiling.** At a
more honest rate for 64x64x128 and 128x128x64 tiles with a serial triangular
solve in the middle and an 8-step serial carry — say 10-15 TFLOP/s — it is
250-370 us, **1.1x to 1.6x.**

Against the layer that is 3097 us, a perfect chunked kernel saves **10%** of
the layer and **1.0% of the whole prefill graph**; the realistic version saves
1-5% of the layer. And it costs something real: the matrix cores are fp16-in,
so `KβKᵀ`, `qH` and the state carry would be evaluated with fp16 operands.
L2e-3 established that the reference already does that to its *projections*,
but a **recurrent state** accumulated through fp16 matmuls across 8 chunks is
a different proposition from one projection, and the scan's 2.4e-06 on
`attn_output` would not survive it.

**So: not built, and the reason is a number rather than a preference.** If it
is ever worth revisiting, the trigger is decode and not prefill — at one token
a chunk is a token and the whole argument changes — and L7 is where that gets
measured.

## Finding 5 — the convolution's window is three rows of negative token index

The fused projection's output is allocated with `Conv - 1` rows *in front of*
token zero, so a tap that reaches before the batch is an ordinary read at a
wrapped `uint` offset and a fresh sequence is those rows zeroed. No branch, no
second tensor, no separate history binding.

That is what makes `TestDeltaNetGPUStateCarries` pass **bit-identically**: 3 +
4 tokens reproduce 7 exactly, in both the output and the 786 432-value state,
with only `DeltaNetState`'s two members carried between them. The control —
the same run with the recurrent state carried and the window zeroed — moves
the output by **4.0e-02 rms**, four orders of magnitude, so the test would
have caught a kernel that forgot it. L3a-6 made this point on the CPU; at L7,
where every decoded token is its own batch, a kernel that carries the 3.1 MB
state and drops the 120 KB window would be wrong on every token.

## Finding 6 — the GEMM schedule carries, and it carries by the MALL

L2f-4 found the plain GEMM arm wanting opposite row-block schedules on two
weights that differ only in which side of the 32 MiB MALL they fall on. This
layer has a third and a fourth weight, and both land where that rule predicts:

| weight | size | 512: m2 / m4 / m8 | 2048: m2 / m4 / m8 |
|---|---:|---|---|
| fused qkv, [16512, 2560] | **84.5 MB** | 5141 / 2899 / **1965** | 24490 / 12949 / **7348** |
| `ssm_out`, [2560, 6144] | **31.5 MB** | 489 / **443** / 519 | 2146 / 1824 / **1605** |

The 84.5 MB weight is 2.6x the MALL and wants the widest row block everywhere,
by 2.6x at 512 and 3.3x at 2048 — a workgroup carrying BM token rows reads all
of B M/BM times, and that is the whole shape of the dispatch. The 31.5 MB one
fits, and on that side reuse buys nothing while occupancy does, until M is
large enough that the activation traffic dominates and the widest rung wins
back. `ssm_out` is the same shape as `attn_output`, and it picks the same
rungs at the same lengths, which is the cheapest possible confirmation that
L2f's schedule is about the weight and not about the layer.

## What the tolerances mean

Every number in this write-up is against **fp16 operands with f32
accumulators**, where L3a's CPU reference was against f32 throughout. Five of
this layer's matmuls move to the matrix cores, and everything downstream
inherits it: the projections land at 1.3e-04 to 2.3e-04 rms, `q_conv_predelta`
at 1.2e-04, the layer output at 1.1e-03.

What the tests are really asserting is that **nothing beyond that is
happening**. The two places it would show are the recurrence, which has no
matmul in it at all — 2.4e-06 rms on `attn_output`, 200x tighter than the
tensors feeding it — and the ladder, where fifteen decompositions of the same
[128, 128] state over lanes and workgroups agree with each other to
**2.6e-10 to 7.1e-10 rms**, with the l64/l64s pair bit-identical because
`QKREG` moves where an operand is read from and nothing else.

## How to reproduce

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    go test ./llm/ -v -run TestDeltaNetGPU        # every tensor, the ladder, the carry
    go run ./cmd/llm -dn -model $M -tokens 512    # the table above
    go run ./cmd/llm -dn -model $M -ladder -csv results/l3b_dn.csv
    go run ./cmd/llm -dn -model $M -tokens 512,2048 -gemm-ladder   # finding 6

`results/l3b_dn.csv` is the fifteen-rung ladder at six lengths, 684 rows.

## What is not built

**Decode.** Everything here is prefill. The state carries correctly across a
batch boundary, which is the piece L7 needs, but the one-token path has its
own problem: at T = 1 the scan's 768 workgroups each run a single iteration,
the 3.1 MB state is read and written whole for one token, and the fused
projection becomes a GEMV. 113 MB of state traffic a token across the 36
layers against a 6.334 GB weight budget is 1.8%, so it is not a bandwidth
problem — it is a dispatch and occupancy problem, and it is the same one
L2a-5 found for the whole decode graph.

**The chunked form**, for the reason finding 4 gives.

**Q8_0 weights read directly.** The bank dequantises to fp16 at upload: the
fused projection is 84.5 MB where the checkpoint's Q8_0 is 42.2, and `qkv` is
64% of this layer and weight-read-bound at the wide end. It is the same open
lever L2c and L2d wrote down, now with the largest weight in the model behind
it.
