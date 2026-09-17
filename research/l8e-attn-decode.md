# L8e — the projections L8d did not reach, and a ladder that lied

**Result: decode is 24.66 tok/s against L8d's 23.15 — 1.065x — with prefill
1052.8 tok/s at ubatch 2048 against 1049.8 and 655.2 at 512 against 653.9, and
no bank change of any kind.** The full-attention layer goes 5.5 ms a token to
**3.5** (1.57x, 354.5 ms of a run to 223.3), the MoE block 12.1 to **11.7**,
and a token 43.2 ms to **40.5**. A decode pass is **1501** dispatches and a
prefill pass **1261**, 48 fewer at both than before L8e-3.

Everything here is a **grid, a lane group or a dispatch that did not need to
exist**. Every weight is staged exactly as L8a, L8b and L5b staged it, so
there is no accuracy question to argue — and the completion this time comes
back **token for token identical to `LLM_DECODE_GEMM=1`'s**, which L8d's did
not.

Files: `llm/gpu_attn.go`, `llm/gpu_moe.go`, `llm/graph.go`,
`llm/gpu_attn_test.go`, `llm/gpu_moe_test.go`, `cmd/llm/bench_attn.go`,
`cmd/llm/bench_moe.go`, `cmd/llm/main.go`. No shader changed.
Results: `results/l8e_decode.csv`, `results/l8e_graph.csv`,
`results/l8e_attn.csv`, `results/l8e_moe.csv`.

---

## What moved

| block | L8d, a token | L8e | |
|---|---:|---:|---|
| hyper-connections | 5.7 ms | 5.6 ms | — |
| PLE n-gram | 0.4 | 0.4 | — |
| gated DeltaNet | 11.0 | 11.0 | — |
| **full attention** | **5.5** | **3.5** | L8e-1 |
| **MoE** | **12.1** | **11.7** | L8e-2, L8e-3 |
| lm head | 3.5 | 3.5 | — |
| move, gather, glue | 4.2 | 4.0 | — |
| **a token** | **43.2 ms** | **40.5 ms** | **24.66 tok/s** |

Two runs of `-gen -n 64` agree to 0.7% (24.66 and 24.48); the attention ladder
reproduces at median 1.0000 over 104 common rows (p10 0.976, p90 1.015) and
the MoE ladder at median 1.0003 over 993 (p10 0.987, p90 1.019).

---

## L8e-1 — the full-attention layer is D11 a fifth time, and it is 1.84x

The layer's two projections were the same `llm_gemm.comp` MODE 2 dispatches
the DeltaNet's were before L8d-5: a sixteen-row cooperative-matrix fragment
holding **one** real row, an LDS slab per K-step behind a barrier, and a grid
of `gemmN/64` workgroups — **218** for the fused projection and **40** for the
output one, on a 40-CU device.

`llm_gemv.comp` was already generic over both banks and both tail cases, so
the work was the wiring and a KSLABS ladder per projection. The two reduction
extents are the DeltaNet's exactly — nEmbd = 2560 on the fused projection and
6144 on the output one — but D12 says a ladder measured on one *matrix* does
not carry to another, so it was re-measured rather than copied. It landed on
the same two rungs, which is a fact and not an assumption:

    us a layer, twelve staged, one token        (results/l8e_attn.csv)

    qkv   [2560, 13952]   GEMM 263.8
          k1 208.4   k2 246.5   k4 201.7   k8 174.3   k20 201.4   k40 180.8
    out   [6144, 2560]    GEMM 124.9
          k1 34.4   k2 27.3   k4 28.0   k8 33.4   k16 29.2   k32 24.6

    layer  414.8 us  ->  225.9   (1.84x, and 1.92x at the two-layer fixture)

D12's 4 KB rotation is visible on the fused projection exactly as L8b-3 found
it on the int8 bank: a slab is `(2560/16/KSLABS) * 256` bytes, so k2's slab is
20 480 — five whole 4 KB pages — and it is the worst rung on the ladder, 1.41x
behind k8's 5120-byte slab which is not a multiple.

**The fused projection is the one that is really 1.51x, and it is at the
bus.** Its bank is 39.5 MB a layer — 13 312 int8 rows with an fp16 scale per
32, plus the indexer's 640 BF16 rows in the tail — and 174.3 us to read that
is **227 GB/s** of a bus this machine has been measured at 242. (The CSV's
`gbps` column says 411, because it counts a weight as the two bytes the fp16
bank held before L8a; every rate in this section is recomputed against the
bank that is actually read.) The output projection's 16.7 MB is *inside* the
32 MiB MALL and the sweep re-reads it, so its 24.6 us — 679 GB/s of real bank,
1266 as the CSV counts it — is not a DRAM rate and cannot be: see D16 below.
In the whole model it costs about 69 us a layer of DRAM time whatever kernel
reads it, which is why the block's real cost is **291 us a layer** against the
micro-bench's 226.

The tail is the check worth stating. The fused matrix's last 640 rows are the
indexer's `q_proj` and `k_proj`, the model's only BF16 weights, and L8a left
them as halves at `gateOff` numbered from `lowRank/16`. A GEMV column read out
of the wrong plane is not a tolerance, it is somebody else's indexer query —
so `TestAttnGPUGemvAgrees` compares the tail on its own as well as the whole
matrix. Every rung agrees with the GEMM at **rms 2.93-2.95e-06** over 13 952
columns (tail alone 9.89-9.98e-06) and **3.62e-06** over the output
projection's 2560, and the number does not move with the rung — a 0.4% spread
across six splits, where a mis-read scale plane would be a different number at
every one.

---

## L8e-2 — the shared expert wanted its own lane group, and the routed pair did not want what the ladder said

The MoE's shared expert is one group of the same grouped GEMM: the same
[2560, 640] and [640, 2560] shapes as a routed expert, the same two kernels,
running for every token. So it rode the routed pair's rung — one field, one
choice — and L8d's ladder picked that field by **block total**.

Read the same ladder one dispatch at a time and the two want opposite ends of
it. A GEMV rung is LPR: how many lanes share one output column and walk its
row of the bank in stride. What that wants is the row's **payload words**,
which is a fact about the format and not about the shape — and the routed
banks are Q4_K (320 payload dwords in a 2560-long row) where the shared
expert's three are Q8_0 (640).

    us a layer at one token                     (results/l8e_moe.csv)

    up      v16  93.2   v32 118.3   v64 131.5   v16w4  63.7   v32w4  78.0   v64w4  78.1
    sh.up   v16  40.5   v32  23.5   v64  30.0   v16w4  35.9   v32w4  19.4   v64w4  12.0
    down    v16  35.4   v32  41.8   v64  56.1   v16w4  34.7   v32w4  38.1   v64w4  45.2
    sh.down v16   8.2   v32   6.5   v64   8.4   v16w4   7.2   v32w4   5.7   v64w4   7.3

One field had to take the better *joint* rung — v64w4 at 90.1 us against
v16w4's 100.7. Two fields take 63.7 + 12.0 = 75.7.

**And then the whole model said no to half of it.** The split itself is
`shUp`/`shDown`, a host-built tile list cut to its own row block, and `pad`
widened to the widest of four dispatches rather than two. Carrying the routed
up mode to v16w4 alongside it is what the table above asks for, and in the
whole model it is **42.1 ms worse over 64 tokens** where the micro-bench
predicted 43.6 better:

| routed up | shared pair | perm passes | MoE over 64 tokens |
|---|---|---|---:|
| v64w4 | on the routed field | two | 759.9 ms |
| **v64w4** | **v64w4 / v32w4** | **one** | **743.4 ms** |
| v16w4 | v64w4 / v32w4 | one | 785.5 ms |

So the routed pair keeps v64w4/v16w4 and only the shared expert moves — its
down mode from v16w4 to v32w4, its up mode already on the rung it wanted for
the wrong reason. What is left of item two is 1.5 us a layer, and the reason
the other 14.4 was never there is the next paragraph.

### D16 — a micro-bench rung whose rate is above the bus is not a DRAM measurement

`-moe -tokens 1 -ladder` stages two layers and re-runs one dispatch twenty
times. A token routes ten experts, so the bank that dispatch actually touches
is about 32 MB — **the MALL exactly** — and every repetition after the first
reads it at 805-965 GB/s rather than at 242. The ladder is then measuring
kernel shape against an L3 hit, and the rung that wins there is the one with
the most parallelism, not the one with the best DRAM locality. In the whole
model each expert is read once from DRAM a token and the wider lane group's
coalescing is what matters.

The rate says so on its face: v16w4's 63.7 us over the 18.4 MB of Q4_K the up
mode reads is **288 GB/s**, which this machine does not have. **Trust a micro-bench rung
only when its measured rate is below the 242 GB/s bus; above it, confirm in
the whole model.** By that test the fused attention projection's ladder above
is sound (227 GB/s at the winner), its output projection's is not (679), and
both of the shared expert's are not (285 and 305) — which is why the one
change that was kept was kept on an end-to-end measurement and not on the
table.

This is D12 widened once more. L7d-2 found that a ladder does not carry
across a *bank*, L8b-3 that it does not carry across a *width*, L8d-5 that it
does not carry across a *matrix*; D16 is that it does not carry across a
**residency**, and that one is a property of the harness rather than of the
kernel.

---

## L8e-3 — the second counting sort was the first one again

`llm_moe_perm.comp` ran twice a layer because the two expert modes are cut to
different row blocks and a tile list is only consistent with the permutation
that outlives it. At decode they are cut to the **same** row block: all six
GEMV rungs fix BM at sixteen so that a GEMV plan may be mixed with a GEMM one.
So the second pass recomputed the first pass's permutation and emitted the
same records into a second buffer, for 2.9 us a layer at one token and 17.8
at ubatch 512.

The fix is two lines and a `downTiles()` accessor: run one pass when
`moeBM(down) == moeBM(up)` and point the down mode at the up mode's list.
**-48 dispatches a pass at every length** — a decode pass is 1501 and a
prefill pass 1261 — worth 0.14 ms a token and 0.9 ms of the 512-token prefill
graph. It is not a decode-only change and it is not conditional on the GEMV:
`m2/m2` at ubatch 512 and `m4/m4` at 2048 have equal row blocks too.

**What was left alone is `route`.** It is 5.8 us a layer at one token, one
workgroup of 256 lanes doing a softmax over 512 experts and then ten
sequential workgroup argmaxes. It is D11's symptom — one workgroup on a 40-CU
device — but not D11's disease: the only axis it has across is the token, and
at decode there is one. Splitting the top-k across workgroups needs a second
dispatch to merge, and a dispatch here is 1-2 us of the 5.8. The item's own
condition was "only worth it if the token gets short enough for it to matter";
at 40.5 ms the 0.28 ms it could plausibly buy is 0.7%, and it is not.

---

## What it costs in exactness, and where

Nothing reads a different weight and no bank moved. What changes is the order
of a sum: the attention layer's two projections now split a 2560- or 6144-long
dot product 8 and 32 ways and add the pieces back, and the shared expert's
down mode groups its lanes 32 to a column rather than 16.

`TestGraphLogits` is unchanged — llama.cpp's own argmax (**561**) out of
llama.cpp's own top ten, the drift the same clean x1.085 a layer and 0.881% at
48 layers. `TestGraphIsAChunkSplit` is unchanged: `result_norm` identical to
the last place on the pinned schedule at every chunking, and rms 7.55e-04 on
the decode schedule, which is the reassociation the pin exists to separate.
`PinSchedule` now covers **five** blocks rather than four.

And the completion is the interesting one. At temperature zero over 128
tokens it is **identical to `LLM_DECODE_GEMM=1`'s, token for token** —
diffed, not eyeballed — which means L8e lands back on L7c's text. L8d's
decode kernels had re-worded one sentence of the `<think>` block at completion
token 47, a tie whose top two logits were 0.012 apart; L8e's extra
reassociation flips that tie back. That is luck rather than a property, and it
is worth saying as such: the honest claim is still "the same text at this
precision", with a tie somewhere in it that either ordering may win.

---

## What is left

- **L8c, the re-quantisation**, onto a baseline that is 1.65x faster than the
  one it was planned against. Everything L8a and L8b stage is still the
  checkpoint's arithmetic at 8.5 bits a weight; D3's ~4.25 is where the next
  1.9x of the dense half is, and it is the first step in this vertical that
  changes what the model computes.
- **The hyper-connection block is now the third-largest at decode** — 5.6 ms
  a token, 13.8% — and the gated DeltaNet the second at 11.0. Neither is a
  grid any more: the DeltaNet's two projections are at 205 GB/s and the
  hyper-connection block's GEMV at 326.
- `route`'s 5.8 us a layer and the three dispatches around it, if a token
  ever gets short enough (above).
