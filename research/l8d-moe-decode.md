# L8d — the decode kernels: the MoE, the router, the combine and the DeltaNet

**Result: decode is 23.15 tok/s against L8b's 14.71 — 1.57x — with prefill
1049.8 tok/s at ubatch 2048 against 1041.1 and 653.9 at 512 against 643.4, and
no bank change of any kind.** The MoE block goes 27.6 ms a token to 12.05
(2.29x, **58 GB/s of bank to 133**), the gated DeltaNet 18.9 to 11.0 (1.72x,
119 GB/s to 205), and a token 66.7 ms to 43.0.

Everything below is a **grid or a kernel shape**. Every weight this stage
reads is staged exactly as L5b and L8a staged it, so there is no accuracy
question to argue and none is argued: the difference between the old kernels
and the new ones is the *order of a sum*, measured at rms 1.7e-05 on a
projection whose scale is 31.5.

Files: `shaders/llm_moe_gemv.comp`, `shaders/llm_moe_router.comp`,
`shaders/llm_gemv.comp`, `shaders/llm_moe_combine.comp`, `llm/gpu_moe.go`,
`llm/gpu_deltanet.go`, `llm/bank.go`, `llm/graph.go`, `cmd/llm/bench_moe.go`,
`cmd/llm/bench_deltanet.go`.
Results: `results/l8d_decode.csv`, `results/l8d_decode_gemm.csv`,
`results/l8d_graph.csv`, `results/l8d_moe.csv`, `results/l8d_dn.csv`.

---

## The control, and what it is a control for

`LLM_DECODE_GEMM=1` puts every kernel this stage added back on the
cooperative-matrix GEMM that ran before it, on the precedent of L8a's
`LLM_DENSE_FP16`. It is the only way to run the two side by side over one
prompt, because a block chooses its rung from the batch size and not from a
flag at the call site.

|  | decode | MoE a token | DeltaNet a token |
|---|---:|---:|---:|
| L8b, as published | 14.71 tok/s | 28.6 ms | 18.9 ms |
| the control (GEMM rungs, L8d's grids) | **14.99** | 27.6 ms | 18.9 ms |
| L8d | **23.15** | **12.05** | **11.0** |

The control is 1.9% faster than L8b rather than identical to it, and that gap
is L8d-1 and L8d-2 — a grid bound and a dispatch shape that are not kernels
and that help the GEMM too.

---

## L8d-1 — the tile grid was a static bound of 2052 where eleven tiles exist

`llm_moe_gemm.comp` dispatches a **static upper bound** of (expert, row block)
records and every workgroup past the schedule's real length returns before it
reads a weight, because the real count is built on the device by
`llm_moe_perm.comp` (L5b). The bound was `maxRows / bm`, and `maxRows` allows
**every one of the 512 experts** an alignment of slack — which is right for
the row space, because a 512-token ubatch really does touch 274 of them.

It is not right for the tile *list*. A batch of `rows` tokens routes
`rows * used` rows and so can make at most that many padded experts, however
wide the padding is. At one token that is ten, and the bound was

    maxRows(1)/16 = (64 + 10 + 512*64)/16 = 2052

tile records for a schedule that holds **eleven**. At the GEMM's own BN of 64
the down mode's grid is then 40 x 2052 = **82 080 workgroups to run 11**, and
an empty workgroup on this device is about 0.5 ns — so it cost ~33 us a layer,
1.6 ms a token, and it was invisible because it was inside the dispatch it
padded.

The new bound is exact at one token:

    reserve/bm + ceil(rows*used / bm) + min(nExpert, rows*used) * (pad/bm)

which is 1 + 0 + 10 = 11. At 512 tokens it is 688 against 1200 and at 2048 it
is 864 against 864 — unchanged, because `min(512, 20480)` is 512 and the old
bound's worst case is the real one there.

**It is the whole reason the first GEMV ladder looked like a failure.** On the
old bound the routed down mode's GEMV at NCOL=16 was **481 us against the
GEMM's 84**, and at NCOL=1 it was 3666 — which reads as a catastrophic kernel
and is in fact 2560 x 2052 = 5.25 M empty launches. A kernel whose whole
purpose is a narrower column block cannot be measured against a grid bound
that multiplies with it.

> **D11 gets a corollary.** *Read a dispatch's shape off its grid* now includes
> the part of the grid that does nothing: a static upper bound is a dispatch,
> not a bookkeeping number, and it has to be tightened on the same axis the
> kernel narrows.

## L8d-2 — the combine was one workgroup, and its grid has an axis order

`llm_moe_combine.comp` sums a token's eleven contributions. Its grid was
`(tokens)` with a 256-lane workgroup walking all 2560 columns — so **at one
token the entire dispatch was one workgroup**: 112 KB of reads on a 40-CU
device, 22 us at 6 GB/s, 9.5% of the block. Splitting the column axis into
64-column blocks is one line and takes it to **2.8 us**, 7.8x.

**The axis order is not cosmetic and cost a 9.2x regression before it was
found.** The obvious spelling puts the token on x and the column block on y.
Vulkan varies x fastest, so consecutive workgroups are then consecutive
*tokens* at the same 256-byte column slice — and a token's eleven rows are
10 KB apart in a 566 MB arena. At a 2048-token ubatch that dispatch went from
815 us to **7505**: 34 GB/s, where the single workgroup it replaced had been
streaming. With the column block on x, the forty workgroups of one token walk
its rows end to end and it is **1104.9 us at 228 GB/s** — 1.35x *faster* than
the version it replaced, at prefill as well as decode.

## L8d-3 — the router was nine workgroups, and split-K is 11.8x

`ffn_gate_inp` is [513, 2560] fp16 — 512 experts and the shared expert's gate
as a 513th column (L5b) — 2.63 MB, the smallest weight in the block by two
orders of magnitude, and **27.7% of it at decode**: 64 us at 41 GB/s where the
two expert GEMVs beside it reach 235 and 345.

D11 for the fourth time. `llm_gemm.comp` blocks the *output* columns at 64, so
576 padded columns are **nine workgroups**. At M=1 the parallelism cannot come
from N — N is the output — so it comes from K, exactly as L7d's down
projection did. `shaders/llm_moe_router.comp` is llm_hc_gemv.comp's split-K
over the same §2.8 fragment tiling with the epilogue removed:
**64 us to 4.4 plus a 1.0 us reduce, 11.8x on the pair.**

**D12 does not apply here, and that is the finding rather than an omission.**
The KSLABS ladder is flat — 8/10/20/40 slabs give 4.6/4.5/4.9/4.3 us — where
the same kernel's ladder on the hyper-connection block spans 2.4x. The 4 KB
rotation is a **DRAM channel** effect (§5.1b), and this matrix is 2.63 MB: it
never leaves the 32 MiB MALL. A stride rule about DRAM says nothing about a
weight that does not reach DRAM.

The logits are compared as numbers *and* the top-ten as a sequence, because
the router's output is discrete: rms 3.4e-05 on a |max| of 7.46, and **0 of 10
slots differ** at every rung.

## L8d-4 — the expert GEMM at one token is the wrong kernel, and the GEMV is 2.0x

At prefill `llm_moe_gemm.comp` is the right kernel and L5b's reasoning holds:
a workgroup unpacks a BN x BK slab of the checkpoint's own Q4_K into LDS per
K-step and multiplies it by BM rows of A, so the unpack — which is most of
what the block does — is amortised over sixteen to sixty-four rows.

**At decode there is one row.** Every row block is fifteen sixteenths padding,
the slab is unpacked for one real row, and the K-loop is a barrier, a
dependent load and a consume with one wave in flight per workgroup. L7d had
already narrowed BN from 64 to 16 and moved the up mode from 71 to 93 GB/s and
no further; the remaining 2.5x is not the grid.

`shaders/llm_moe_gemv.comp` changes all three of the things L8d's plan named:

- **the grid** — LPR lanes share one output column and a workgroup owns
  `waves*64/LPR` of them, so the up mode is 40-640 workgroups per expert
  rather than 10-40;
- **the LDS round trip** — there is none. §2.2's rule that nibbles cannot be
  unpacked into a cooperative-matrix *fragment* still holds; it says nothing
  about a dot product, which has no fragment. A lane unpacks its own dwords
  straight into registers. The A vector is the only thing in LDS, staged once
  per workgroup rather than once per K-step;
- **the loads** — a lane's dwords are independent of each other and of every
  barrier, so a whole row's worth is in flight at once instead of one K-step's
  being issued and immediately consumed (§2.7).

Microseconds a layer at one token, each dispatch at its own best rung:

| | GEMM | GEMV | |
|---|---:|---:|---|
| routed up (Q4_K, gate+up fused) | 196.5 | **78.1** | 2.52x, 94 GB/s to 236 |
| routed down (Q5_1) | 84.9 | **34.5** | 2.46x, 145 to 356 |
| shared up (Q8_0) | 75.9 | **12.3** | **6.17x**, 46 to 284 |
| shared down (Q8_0) | 11.8 | **7.3** | 1.62x, 147 to 239 |
| router | 63.1 | **5.4** | 11.8x (L8d-3) |
| combine | 21.9 | **2.8** | 7.8x (L8d-2) |
| **block** | **464.5** | **152.0** | **3.06x** |

The rates above 242 GB/s are the bench's one-layer loop hitting the MALL; the
end-to-end number is the honest one and it is **133 GB/s**, 2.3x the 58 the
block ran at.

> **The lane count is a ladder and it inverts between two dispatches of the
> same kernel.** The routed up mode's best rung is LPR=16 (65 us) and the
> shared expert's is LPR=64 (12 us) — the same MODE 0 over the same shape,
> differing only in format: Q4_K packs 320 payload dwords into a 2560-long row
> and Q8_0 packs 640, so the wider read is the one with twice the bytes to
> coalesce. The plan runs both on LPR=64 because they are one rung today
> (78.1 + 12.3 = 90.4 against a split plan's 77.3), and splitting them is
> 13 us a layer still on the table.

**It is a decode kernel and the host says so.** A tile is (expert, row block)
and the GEMV reads *one* row of it — the first, which is the only real one
when the batch is a single token, because a token's top-k names ten
**distinct** experts with one row each and the permutation puts that row at
the tile's base. `SetPlan` and `Upload` refuse the rung above one token rather
than let it drop rows silently, and `TestMoEGPUDecode` asserts the refusal.

## L8d-5 — the same finding on the dense projections, and it is worth 1.72x

`shaders/llm_gemv.comp` is llm_hc_gemv.comp's split-K with the
hyper-connection epilogue removed, so it serves any `llm_gemm.comp` MODE 2
caller: MODE 0 writes `KSLABS` partials, MODE 1 sums them, and at KSLABS=1
MODE 0 stores the row itself. Both banks, and L8a's fp16 tail — the DeltaNet's
F32 alpha and beta, 128 rows of 16512 kept as halves at `gateOff` and numbered
from `lowRank/16`.

The gated DeltaNet is 36 of the 48 layers. Microseconds a layer at one token:

| | GEMM | GEMV | |
|---|---:|---:|---|
| qkv (16512 x 2560, with the tail) | 316.8 | **199.1** + 1.4 | 1.58x |
| out (2560 x 6144) | 124.3 | **26.7** + 1.2 | 4.45x |
| **layer** | **454.0** | **242.5** | **1.87x** |

**D12 does apply here, and the rung moves with K exactly as it says.** A slab
is `(gemmK/16/KSLABS) * 256` bytes on the int8 bank:

| qkv, K = 2560 | 1 | 2 | 4 | 8 | 20 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| slab / 4 KB | 10 | 5 | 2.5 | 1.25 | 0.5 | 0.25 |
| us | 275.8 | 308.9 | 244.0 | **196.2** | 225.4 | 207.3 |

| out, K = 6144 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| slab / 4 KB | 24 | 12 | 6 | 3 | 1.5 | 0.75 |
| us | 32.8 | 25.4 | 25.8 | 30.7 | 26.1 | **22.0** |

(each axis laddered against the other's default, so the winning pair measures
200.5 and 27.9 us when they run together rather than 196.2 and 22.0 apart —
the two dispatches share a bus.)

The two whole multiples that are also the two extremes — 1 and 2 slabs on a K
of 2560 — are the two slow rungs, and the plan is `k8` for one projection and
`k32` for the other. **A ladder measured on one matrix does not carry to
another of the same kernel.** A rung also has to cut K into whole four-tile
steps, which is why 16 and 32 do not exist for a K of 2560; `GEMVFits` is what
says so and the ladder filters rather than failing.

---

## What it costs in exactness, and where

**Nothing in the bank and nothing in a weight.** Every kernel here reads the
bytes L5b and L8a staged, and the GEMV multiplies in fp16 for L8b-2's reason
so that the halves are the GEMM's halves.

What changes is **the order of a sum**. A cooperative-matrix accumulator adds
sixteen k an instruction in an order the extension does not define; a GEMV
lane adds them serially; a split-K reduce adds the slabs afterwards. Measured
against the GEMM over one token:

| | rms | reference scale |
|---|---:|---:|
| MoE block, `ffn_out` | 2.5e-06 | 0.032 |
| DeltaNet, fused projection | 1.75e-05 | 31.5 |
| DeltaNet, its fp16 tail alone | 1.77e-05 | 8.79 |
| DeltaNet, output projection | 1.96e-06 | 0.998 |
| router logits | 3.4e-05 | 7.46 |

What makes those measurements rather than alibis is that they **do not move
with the rung**: every split of the DeltaNet's qkv lands within 0.2% of the
same rms, where a mis-read scale plane or a tail taken from the wrong row
would be a different number at every one. The MoE's `ffn_out` is 50x closer to
the GEMM than the GEMM is to llama.cpp's own tensor, and neither is further
from the reference than the other.

**Two places had to be told.** `Graph.PinSchedule` existed for L7d's split-K
down projection — the first rung in this vertical that reassociates — and now
covers four blocks rather than one, so `TestGraphIsAChunkSplit`'s three
equalities are still exact to the last place. Its fourth case, the unpinned
decode schedule, is what now measures all four kernels at once: **4.6e-04 rms
on a `result_norm` of |18.5|**, against the 1.785e+00 the no-history control
reports.

### And the text moves at one token

`TestGraphLogits` is unchanged — llama.cpp's own argmax (**561**) out of
llama.cpp's own top ten, with the drift the same clean x1.085 a layer, 0.881%
at 48 layers. Prefill is untouched because none of these kernels runs there.

At temperature zero over a 69-token completion the reassociation is
**enough to re-word one sentence**, and it is worth being exact about where:

    the GEMM   ... asking me to complete the pattern. The capital of
               Portugal is Lisbon. This is a straightforward factual
               completion.
    the GEMV   ... asking me to complete the pattern for Portugal. The
               capital of Portugal is Lisbon.

The capitals list is identical, the `<think>` block opens identically, and the
answer is `Lisbon.` on both. They diverge at **completion token 47 and nowhere
else that is not downstream of it**, and that token is a tie: the GEMM's top
two are `.` 25.177 and ` for` 25.165, **0.012 apart**, where the tokens either
side of it are decided by 3 to 25. The GEMV's are ` for` 25.180 and `.` 25.127,
0.053 the other way.

That is the same class of thing L7c's own divergence from llama.cpp is — one
token, 0.161 apart — and an order of magnitude tighter. It is the honest
reading of "the same text" at this precision, and it is the reason the control
exists: `LLM_DECODE_GEMM=1` reproduces L7c's completion exactly.

---

## What is left

- **The full-attention layer is the same finding a fifth time**, and it is
  now the third-largest block at decode (354.5 ms of a 2478 ms GPU total,
  12.9%, ~126 GB/s). Its two projections are the same `llm_gemm.comp` MODE 2
  dispatches the DeltaNet's are, and `llm_gemv.comp` is already generic: the
  work is the wiring and a ladder, and it is worth ~1.5 tok/s.
- **The MoE's shared expert wants its own rung.** 13 us a layer, 0.6 ms a
  token, and it needs a second pair of fields on the plan.
- **The up mode is 51% of what is left of the block** at 236 GB/s, which is
  97% of the bus. It is finished as a bandwidth problem; the next thing that
  moves it is L8c's narrower bank.
- **`route` and the two `perm` dispatches are 11.6 us a layer**, 0.56 ms a
  token, and all three are one workgroup. The same fix as L8d-2 if it is ever
  worth 1 tok/s.
