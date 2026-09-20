<!-- LLM.md L5b. The MoE block on the device: a grouped GEMM over the
     checkpoint's own quantised blocks, the schedule L5a-4's routing forces,
     and four optimisations of which two were worth nothing. Cited from
     llm/gpu_moe.go and shaders/llm_moe_*.comp. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L5a](l5a-moe.md) · [L4b](l4b-qsa-gpu.md) · [L3b](l3b-deltanet-gpu.md) · phase 1

# L5b — the MoE block on the device: the last kernel, and the first one whose weights stay quantised

**Result: nine dispatches against the ~845 llama.cpp spends on the same work,
567.4 ms of its 512-token prefill graph against 522.3 — 1.09x at the
reference's own best ubatch and 1.55x at 2048 — with `ffn_out` at 8.132e-05
rms over all 4096 tokens of the 4 k dump and the routing reproduced set for
set and *order for order*.** Every layer of this model now runs on the GPU.

Two things fell out that are not about the MoE. The router turns out to be
**bit-exact** against llama.cpp on all 2 097 152 logits, and so does
`shared_expert_gate` — which **corrects L5a-1**: that gate does not stay on
the f32 vector path at a real ubatch, so fusing it onto the router's matrix is
not a deviation but a reproduction. And of the four things tried to make the
grouped GEMM faster, **two of the three that should obviously have worked
made it slower**, for reasons that are now measured rather than guessed.

## The block

Nine dispatches, against a reference that spends fifteen distinguishable op
lines and about 845 dispatches a graph:

    router    [nExpert+1, nEmbd] * xn -> logits          llm_gemm.comp MODE 2
    route     softmax, top-10, normalise, shared gate    one workgroup a token
    perm x2   counting sort + the two tile schedules     one workgroup
    up        gate and up, silu(gate)*up fused           grouped, Q4_K/Q5_K
    down      down, in permutation order                 grouped, Q5_1/Q8_0
    shexp x2  the same two kernels, one group, Q8_0
    combine   the eleven contributions weighted and summed

Four tensors the reference materialises never exist. `ffn_moe_gate` and
`ffn_moe_up` are consumed on the accumulators, where llama.cpp writes both and
reads them back through `ggml_swiglu_split` (96 dispatches, 7.0 ms a graph).
`ffn_moe_weighted` is folded into the combine's read. And
`shared_expert_gate` — a [2560, 1] F32 matrix run as its own matmul, 47
dispatches at **13.6 GFLOP/s and 9.1 ms a graph**, which is L2a's `inject`
scandal in miniature — is a 513th column of the router's matrix, the same
fusion argument at a fifth site after L2a's `inject`, L2d's PLE value, L2f-3's
six attention matrices and L3b-2's alpha and beta.

## L5b-1: 567.4 ms becomes 522.3, and at ubatch 2048 1758.4 becomes 1131.6

Per prefill graph, against the lines of llama.cpp's own graph that are
nameably this block. Our side is one measured layer times the 48 that have
one.

| | llama.cpp disp | llama.cpp | ours | |
|---|---:|---:|---:|---:|
| router | 94 | 15.2 ms | 4.7 ms | **3.22x** |
| route | 238 | 2.6 ms | 0.7 ms | **3.70x** |
| perm (x2) | 135 | 2.0 ms | 1.8 ms | 1.11x |
| up | 190 | 372.3 ms | 345.4 ms | 1.08x |
| down | 47 | 158.9 ms | 134.0 ms | 1.19x |
| shexp.up | 94 | 12.3 ms | 10.4 ms | 1.18x |
| shexp.down | 47 | 4.1 ms | 3.2 ms | 1.31x |
| combine | 0 | — | 22.1 ms | — |
| **block** | **845** | **567.4 ms** | **522.3 ms** | **1.09x** |

**48.7% of llama.cpp's 1164.7 ms graph becomes 44.8%.** A second run gives
522.2 ms, 0.3% apart.

At `-ub 2048` the same comparison is **1758.4 ms against 1131.6, 1.55x** —
38.6% of that graph becoming 25.4% — because the MoE is the one block that
gets cheaper per token on *both* sides as the chunk grows, and the glue that
makes 2048 a worse ubatch overall (L2a-4) belongs to other blocks. Per token
the block costs 21.2 us at 512, 11.5 at 2048 and 8.7 at 4096.

What is **not** attributed on either side: the `MUL` that applies the ten
weights, the `MUL` that applies the shared gate, the `SIGMOID` behind it and
the `ADD` that sums the two halves all live in the shared MUL (479), SIGMOID
(326) and ADD (231) lines every other block also uses. Every one of those
dispatches is absent from our graph rather than faster in it — the weights
ride the combine's read, the gate rides the router — so leaving them out is
the conservative direction. The one line that runs the other way is
`combine`, a dispatch the reference does not have at all and which is counted
in full.

## L5b-2: the bank never becomes floats, and that is what makes the kernel possible

Every other GEMM in this vertical takes a B operand the host has dequantised
to halves and packed into §2.8 fragment tiles. That cannot happen here. One
layer's three routed banks are **2.52 billion weights**: 5.03 GB at fp16 and
**241 GB across the model**, against 1.57 GB and 75 GB as they ship. So the
bank is staged **byte for byte out of the mmap'd checkpoint** and each
workgroup unpacks its own BN x BK slab into LDS per K-step — §2.2's
arrangement, for the reason it gave: a cooperative-matrix fragment's lane
layout is not exposed by `GL_KHR_cooperative_matrix`, so nibbles cannot be
unpacked into registers and declared a fragment.

Four formats, because which one a tensor is, is a fact about the layer:
**Q4_K** gate and up on 46 layers, **Q5_K** on layer 2, **Q5_1** down on 43,
**Q8_0** down on four and the shared expert everywhere. They are one kernel
with a `-DQFMT`, and a row's byte length is `(K / QK) * QB` so nothing below
the format arm knows which axis it is walking.

## L5b-3: the schedule is a list, and padding the rows is what buys the epilogue

L5a-4 measured the routing and it is not balanced: at a 512-token ubatch
**274 of 512 experts are touched, one takes 484 of the 512 tokens and the
other 273 have fewer than 19 rows**. A grid of (experts x fixed row blocks)
is almost entirely empty, so `llm_moe_perm.comp` emits one record per
(expert, row block) that has rows and the GEMM dispatches a static upper
bound with an early return — 372 real records against a bound of 1216 at
ubatch 512.

The less obvious half is the **padding**. A cooperative-matrix store covers
sixteen rows and cannot be masked, so a tile whose expert runs out of rows
part-way has to stage its accumulators in LDS and copy them out under a
bound. Rounding every expert's range up to the schedule's alignment instead —
and naming the slack with a **sentinel** permutation entry pointing at a token
one past the batch, whose activation row is zero — makes every store a
fragment store. It costs arithmetic on rows nothing reads:

| ubatch | experts touched | real rows | executed at BM 16 / 32 / 64 |
|---:|---:|---:|---|
| 512 | 274 | 5 120 | 8 064 (1.57x) / 11 904 (2.33x) / 19 968 (3.90x) |
| 2048 | 399 | 20 480 | 24 304 (1.19x) / 28 832 (1.41x) / 38 912 (1.90x) |
| 4096 | 435 | 40 960 | 44 816 (1.09x) / 49 536 (1.21x) / 59 904 (1.46x) |

> **And it is inert, which is a control and not an assumption.** Poisoning the
> sentinel's activation row with 1000 and re-running moves `ffn_out` by
> **exactly zero** at every rung. Nothing in a tensor comparison tests that:
> the padding rows are invisible, and a kernel whose padding overlapped a
> *real* row would be wrong only where the overlap happened to matter.

The permutation's row index is `token * (used + 1) + slot`, not the token,
and that one choice does four things: the shared expert's row is the last slot
so it is one more group of the same grouped GEMM rather than a dense special
case; the up projection's gathered rows become the down projection's
**contiguous** ones; the routing weight is indexed by the same number as the
row; and a token's eleven contributions are contiguous for the combine.

## L5b-4: the ladder moves outward with the chunk, and sharing the slab across waves loses

`results/l5b_moe_ladder.csv`, 25 plans at two lengths. A rung is how many
accumulator tiles a wave holds and how many waves split a tile's rows.

| | 512 | 2048 |
|---|---:|---:|
| m1/m1 | 11 510 us | 28 471 us |
| **m2/m2** | **10 874** | 24 257 |
| m4/m4 | 12 340 | **23 645** |
| w2m1/w2m1 | 12 527 | 29 460 |
| w4m1/w4m1 | 13 533 | 27 018 |

The winner moves from BM 32 to BM 64 between 1024 and 2048 tokens, and
`MoEPlanFor` is that boundary: a wider row block amortises the slab a
workgroup unpacks over more rows and costs padding, and how much padding
depends on how skewed the routing is — 3.90x at 512 against 1.90x at 2048 for
the same BM 64.

> **The multi-wave rungs are 1.15-1.24x *against* at the same row block**, and
> that is the finding. `w2m1` and `m2` execute identical tiles over identical
> rows; the only difference is that `w2m1` splits them across two waves of one
> workgroup that share the dequantised slab, which halves the unpack per row
> and **doubles the waves resident per CU** on a kernel running at about one
> wave a SIMD. It loses anyway. Occupancy is not what this kernel is short of,
> which is the same shape of answer L3b-4 got for the scan's staged operands
> ("sharing across four waves of a workgroup cuts the re-read 4x and ties").

## L5b-5: the LDS pad is pinned by alignment, and the textbook fix is 5.4x worse

A slab row of `BK + 8` halves is 20 dwords and `gcd(20, 32)` is 4, so the 64
lanes of the unpack — one output column each — land on eight of the 32 LDS
banks. A four-way bank conflict on every store of the hottest loop in the
block is exactly the thing §2.2 measured at 1.68-1.83x, and the fix is
arithmetic: two halves of pad make the stride 17 dwords, coprime with 32,
conflict-free.

**It is 5.4x slower.** At 34 halves a row starts on a 4-byte boundary and the
cooperative-matrix load *off* LDS loses its wide path, which costs far more
than the conflict it removes. Every pad that keeps rows 16-byte aligned is a
multiple of four dwords and so has `gcd ≥ 4` with the bank count: a four-way
conflict is the price of the fragment load, and eight is as good as it gets.

## L5b-6: the apparent 2x read amplification is not one

Q4_K and Q5_K pack a 64-element group as the low nibbles of 32 bytes followed
by the high nibbles of the **same** 32 bytes. A 32-element K-step therefore
reads the group's bytes, uses half of every one, and reads them again — an
apparent 2x amplification on the 44.4 GB of Q4_K in this checkpoint, which
one step per group removes exactly.

**Measured, one step per group is 1.16x slower** (up: 7 209 us against 8 363
at ubatch 512). The doubled slab takes MODE 0's LDS from 12.8 KB to 23 and
the workgroups a CU can hold from five to two, and the second read of those
32 bytes was being served out of cache rather than off the bus. The loop is
written to cover any BK and the ladder runs at 32.

## L5b-7: the two that did work were the epilogue and the load width

Both are about instruction count rather than about bytes or arithmetic.

> **Deleting the LDS epilogue is 1.35x on the whole block** — 14 785 us to
> 10 930 at m2/m2. That is L5b-3's padding paying for itself: the staging
> buffer it removes is 8 KB at BM 32 and 16 at BM 64, and with it gone MODE 0
> holds five workgroups a CU where it held three.
>
> **Widening the unpack's loads is 3.1x on Q8_0 and 1.05x on Q4_K.**
> `shexp.up` goes from 669 us to 217 and `shexp.down` from 124 to 67. Three
> changes: the block header is one word rather than four byte reads (`d` and
> its companion are the first four bytes of every one of these formats and the
> block base is four-aligned); Q4_K's and Q5_K's 32 bytes of nibbles are two
> aligned `uvec4` loads through a second view of the same buffer rather than
> eight `uint` ones; and Q8_0's int8s, which sit at an offset of 2 mod 4 on
> every other block, come out of nine words through a rotating window rather
> than out of **32 separate byte reads**. The Q8_0 arm is why the shared
> expert moved 3x and the routed half barely did.

## L5b-8: the router is bit-exact, and that corrects L5a-1

`ffn_moe_logits-3` at 4096 tokens is **0 ulp** against llama.cpp on all
2 097 152 values — `maxAbs` exactly zero — where L5a's CPU reference sits at
3.606e-05 rms on the same tensor. Same operands (fp16), same accumulator
(f32), and evidently the same 16-wide accumulation order: the coopmat path
reduces K in tiles and so do we, while `matvecW16` sums 2560 terms in a line.

**And `shared_expert_gate-3` is bit-exact too, which L5a-1 did not expect.**
That finding read the reference as keeping this one on the f32 *vector* path
"at any prompt length", because it has one output column — and inferred that
fusing it onto the router's matrix would be a deviation to be priced, in the
spirit of L3b-2's alpha and beta. It is not. `ggml_vk_mul_mat`'s
`mul_mat_vec_max_cols` threshold counts the columns of the **output**, which
is the token count, so at 4096 tokens a [2560, 1] F32 weight goes to the same
fp16 coopmat GEMM as everything else. The evidence is two-sided: our fused
column reproduces the reference exactly, and the CPU reference's honest f32
dot product is the one that differs, at 4.152e-05. L4a-5's census
independently put this tensor at 0.0% exactly-representable halves, which is
what an F32 (non-`.f16acc`) GEMM gives — it was never evidence for the vector
path.

So the fifth fusion site costs nothing at all, and L5a-2's reading of
`shared_expert_gate` as "the tightest tensor in the block because it is on the
f32 vector path" should be read instead as "the tightest tensor in the block
because it has no quantised weight in it".

## L5b-9: the block at 4 k, and a selection that now matches in order too

Every tensor of both halves, over all 4096 tokens, from llama.cpp's own
`hc_mixed-3` (the **second** occurrence — the hyper-connection block runs
twice a layer):

| tensor | rms | ref \|max\| | |
|---|---:|---:|---|
| `ffn_moe_logits-3` | **0** | 9.920 | bit-exact, L5b-8 |
| `shared_expert_gate-3` | **0** | 2.753 | |
| `ffn_moe_weights_norm-3` | 9.342e-09 | 0.582 | the softmax and the divide |
| `ffn_moe_swiglu-3` | 5.495e-04 | 8.844 | Q4_K weights against an fp16 activation |
| `ffn_moe_weighted-3` | 2.441e-05 | 0.141 | Q5_1 |
| `ffn_swiglu-3` | 4.983e-04 | 4.609 | the shared expert, Q8_0 |
| `ffn_shexp_gated-3` | 2.568e-05 | 0.124 | |
| **`ffn_out-3`** | **8.132e-05** | 0.165 | the block |

These are L4a-5's tolerances and not L2's: every matmul here is a quantised
weight against an activation the reference quantises to int8, and at 4096
columns it accumulates all of them in **fp16**, so ~5e-04 rms on a tensor
whose own values reach 8.8 is the oracle's precision and not ours. L5a-3
measured that modelling the reference's arithmetic makes the fit *worse* here
by 1.4-1.8x; the f32 accumulator this kernel uses is the nearer of the two to
the model, and `ffn_out` at 8.132e-05 over 4096 tokens sits inside L5a's
1.378e-04 over 48.

> **The selection is the reference's on all 4096 tokens — and now in order as
> well.** 0 of 40 960 slots differ as a set, and **0 differ in sequence**,
> where L5a's CPU reference had two slots on one row swapped. That is L5b-8
> propagating: the two probabilities that tie are 4.35e-07 apart relative, the
> CPU's own logits differ from the reference's by more than that, and the
> device's do not differ at all. The ten workgroup argmaxes resolve a tie
> towards the lower expert index, which is what `ggml_argsort`'s DESC
> comparator does under a stable sort.

The permutation has a gate of its own and it has no tolerance: it is a
bijection over all 40 960 (token, slot) pairs, every row sits in the range of
the expert that chose it, the histogram is the selection's bucket for bucket,
and the inverse round-trips on all 45 056 slots including the shared
expert's.

## L5b-10: what it costs, and where the next 2.5x is

At ubatch 512 the `up` dispatch is 7 195 us, **10.8 TFLOP/s and 70 GB/s** of
quantised bank. Neither number is near a wall: §2.7's best measured kernel is
39 TFLOP/s and L0a's bus is 236 GB/s.

The floor is arithmetic. 274 experts of gate and up are **505 MB**, and the
schedule reads 372 tiles' worth of them — 1.36x, because a tile covers one
expert's whole 640 columns and the hot expert has sixteen tiles — so ~686 MB
at 236 GB/s is **2.9 ms**, against 7.2 measured. The MMA at the executed row
count is 78 GFLOP, 2.0 ms at 39 TFLOP/s. So the kernel is **2.5x off its own
byte floor** with the arithmetic able to hide under it, and what sits in
between is the unpack: 1.22 G weights a dispatch, each a shift, a mask, a
convert, a fused multiply-add and a 16-bit LDS store, issued by a wave that
is also the one waiting for the load.

Three things are open and two of them are cheap:

- **The combine is the price of having no float atomic.** 22.1 ms a graph at
  512 and 66.4 at 2048, at 137-198 GB/s, reading eleven rows a token and
  writing one. `VK_EXT_shader_atomic_float` would delete the dispatch and the
  [T][used+1][nEmbd] tensor behind it — 58 MB at 512 tokens — at the cost of a
  non-deterministic summation order, which is the thing L5a's tie analysis
  says to be careful about.
- **The arithmetic intensity is the routing's, not the kernel's.** 274
  experts read 505 MB to serve 5120 rows: ~17 rows an expert, and the whole
  block is memory-bound at prefill for the same reason decode is. The lever
  is not a better tile but a **bigger ubatch** — which is why the block is
  1.55x at 2048 and 1.09x at 512, and why L2a-4's "do not raise the ubatch"
  is a statement about the *glue*, not about this.
- **The hot expert is still routed.** L5a-4's expert 454 is chosen by 95% of
  tokens at ubatch 512 and is the same shape as the shared expert. Running
  the two together as one dense pair and routing only the other nine would
  delete a gather for 95% of the block's tokens, and would be wrong on the
  other 5%, so it is a fast path with a correction rather than a
  simplification. Still unmeasured.

## How to run it

    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    go run ./cmd/llm -moe -model $M -tokens 512,2048
    go run ./cmd/llm -moe -model $M -tokens 512,2048 -ladder -iters 5 \
        -csv results/l5b_moe_ladder.csv
    go test ./llm/ -v -run TestMoEGPU

The benchmark reads `hc_mixed-3` out of `reference/out/llm4k/` and takes its
first T rows, and that is not a convenience: everything this kernel costs
depends on the routing, and a synthetic activation routes very nearly
uniformly — which reads twice the bank and hides the imbalance the schedule
exists for. Without the trace it falls back to synthetic input and says so.

One staged layer is **1.57 GB**, so `-layers` defaults to one; two is the most
a 4 GiB buffer holds. Repetition warms nothing that matters here — a layer's
bank is 50x the MALL — which is the one way this block's profiler differs
from the other four's.
