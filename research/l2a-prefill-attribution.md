<!-- LLM.md L2a. Where llama.cpp's 393 tok/s prefill actually goes, per op,
     measured with GGML_VK_PERF_LOGGER. Cited from LLM.md's open questions. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L1](l1-baseline.md) · [§2.2](2.2-q4-coopmat.md) · [§3.5](3.5-grouped-moe-gemm.md) · phase 1

# L2a — the 5x is attributed: it is not the architecture

**Result: the two candidates LLM.md offered for "why is prefill 393 and not
~2000" were both wrong, and the answer is better than either.** DeltaNet, the
QSA indexer and full attention together are **4.5%** of a prefill graph, so
the architecture is not hiding a pile of non-matmul work. Nor is llama.cpp
losing it to host overhead — **93% of the wall clock is GPU kernel time** at
the ubatch it runs best at. The time goes to four places, and three of them
this repo already has a faster kernel for:

| | ms per 2048 tokens | share | ours, measured | |
|---|---:|---:|---:|---:|
| MoE experts (`MUL_MAT_ID`) | 1626 | 35.7% | 642 (§2.2 Q4 grouped) | **2.53x** |
| dense + f32 matmul (`MUL_MAT`) | 1192 | 26.2% | 529 (`results/shapes.csv`) | **2.25x** |
| elementwise glue | 1396 | 30.6% | fusible | — |
| norms, DeltaNet, attention, routing, rope | 342 | 7.5% | 248 | 1.38x |
| **total GPU** | **4556** | | **1768-2879** | |
| | **450 tok/s** | | **711-1159 tok/s** | |

So **L6 is an afternoon, not a fortnight**: ~2.6-3.0x against llama.cpp's
391.4 tok/s is reachable with kernels that exist, and the honest ceiling is
**~1150 tok/s, not the ~2000** §3.4 and §2.2 were added up into. Raw data:
`results/l2a_prefill_ops.csv`, 195 rows.

## How it was measured

llama.cpp's Vulkan backend carries a per-op timestamp logger. With
`GGML_VK_PERF_LOGGER=1` it writes a timestamp query around every dispatch,
inserts a buffer barrier between them, and prints per-op-shape totals for
each graph. The serialisation is the one thing to be careful of, and it is
small here: at `-ub 512` the logger's own sum over the graph is **1211.8 ms**
(8 graphs, 1.90% spread) against an un-instrumented **5.232 s** for the four
graphs a 2048-token chunk needs — **93% of the wall clock**, so the barriers
cost little and the attribution is if anything conservative about how much is
GPU-bound.

    M=models/Qwen3.8-Flash-Next-GGUF/…UD-Q4_K_XL-00001-of-00004.gguf
    GGML_VK_PERF_LOGGER=1 llama-bench -m $M -p 2048 -n 16 -r 1 -ub 512
    GGML_VK_PERF_LOGGER=1 llama-bench -m $M -p 2048 -n 0  -r 1 -ub 2048 -b 2048

Reproducibility: the two `-ub 2048` graphs agree to **0.09%** on the total and
to 1-3% line for line; the eight `-ub 512` graphs to 1.90%. The un-instrumented
`pp2048` re-measured this session is **391.42 ± 1.13** against L1's separate
**388.60 ± 1.89** — **0.7%** apart, which is the two-run agreement `TODO.md`
asks for.

## The prefill graph, by category

Per token, so the two ubatches are comparable; `-ub 512` is what llama-bench
and the L1 baseline run at, `-ub 2048` is one graph for the whole chunk.

| category | µs/tok @ub512 | µs/tok @ub2048 | ratio | dispatches |
|---|---:|---:|---:|---:|
| MoE experts | 1024.3 | 794.0 | 0.78 | 144 |
| dense matmul (q8_0/bf16) | 460.7 | 482.4 | 1.05 | 521 |
| **elementwise glue** | 432.7 | **681.5** | **1.57** | 2453 |
| **matmul f32 (tiny-N)** | **290.5** | 101.2 | 0.35 | 276 |
| norms | 54.9 | 63.0 | 1.15 | 232 |
| attention + QSA | 53.9 | 32.6 | 0.60 | 36 |
| DeltaNet core | 53.1 | 61.1 | 1.15 | 180 |
| routing / gather | 10.4 | 7.5 | 0.72 | 302 |
| rope | 2.2 | 3.0 | 1.35 | 48 |
| **total** | **2382.7** | **2226.1** | 0.93 | ~4150 |
| **GPU-only rate** | **420 tok/s** | **449 tok/s** | | |

### Finding 1: the architecture is innocent

**`GATED_DELTA_NET` is 1.6% of the graph** — 36 dispatches, 2057 µs each at
2048 tokens, 74.1 ms of 4556. Add `SSM_CONV`, `L2_NORM` and `SOFTPLUS` and
the whole DeltaNet core is **2.7%**. Full attention plus the QSA indexer and
its top-2048 selection is **1.5%** (`FLASH_ATTN_EXT` runs at 21.4 TFLOP/s,
which is respectable). LLM.md's first candidate — *"this architecture has a
great deal of prefill that is not matmul — 36 DeltaNet layers, the QSA
indexer"* — accounts for **4.5%** of prefill and cannot explain a 5x.

The hyper-connections, the third item on that list, *are* expensive, but not
because they are narrow: see finding 3.

### Finding 2: it is not host overhead either, at `-ub 512`

93% of the un-instrumented wall clock is inside timed GPU dispatches. Whatever
llama.cpp is losing, it is losing in kernels.

### Finding 3: `hc_*_inject` is 10.3% of prefill for 15.6 MB of weights

The single worst line in the graph:

    MUL_MAT f32 m=4 n=512 k=10240: 95 x 1318.2 us = 125231 us (31.8 GFLOPS/s)

`build_hc_mix` projects the normalised 10240-wide residual `xn` down to **4**
scatter weights per token with an F32 `[10240, 4]` matrix, 95 times a graph
(48 layers x 2, less one). It is 0.016 TFLOP of arithmetic and **10.3% of the
prefill graph**, running at **31.8 GFLOP/s — 0.06% of this part's fp16 peak**
and 16 GB/s of activation bandwidth. An N=4 F32 GEMM has nothing to fill the
machine with.

It is also free to remove. `w_down` ([10240, 320], 34.9 TFLOP/s in our own
kernel) reads *the same* `xn`; `w_inject` is four more output columns on a
matrix that already has 320. **Fusing inject into hc.down deletes 10% of
prefill.** The same applies to the shared-expert gate — `MUL_MAT f32 m=1
n=2048 k=2560`, 11.0 ms for a single output column — which belongs in
`moe.shared`'s epilogue.

Between them the tiny-N F32 matmuls are **12.2% of the graph at `-ub 512`**
and 4.5% at `-ub 2048`, where there are more tokens to amortise the launch
against.

### Finding 4: 30% of the graph is glue, and `CONCAT` runs at 13 GB/s

2453 dispatches of `MUL`, `ADD`, `MULTI_ADD`, `CONCAT`, `REPEAT`, `SIGMOID`,
`CONT`, `SCALE`. This is the hyper-connection tax made concrete: a 10240-wide
F32 residual carrying 4 branches, normalised, gated, collapsed by a `CONT`
plus three `ADD`s, and recombined through a `REPEAT` and a `MUL` **every
layer**. The four largest:

| op | @ub512 | @ub2048 | effective bandwidth |
|---|---:|---:|---:|
| `MUL` (479x) | 83.2 ms | 410.2 ms | 362 -> 294 GB/s |
| `CONCAT` (37x) | 32.6 ms | 475.4 ms | **48 -> 13 GB/s** |
| `MULTI_ADD` (170x) | 31.7 ms | 120.2 ms | |
| `ADD` (231x) | 25.3 ms | 138.0 ms | |

`CONCAT` is 14.6x slower for 4x the data — it is DeltaNet's conv input being
joined to the 3-column conv state along the token axis, and at 13 GB/s it is
**5% of the bus**. Nothing here is arithmetic; all of it is either an epilogue
on a matmul we already run or a kernel that should be at the bus.

### Finding 5: `-ub 512` is a MALL-sized sweet spot, and it is why bigger is worse

The obvious lever on a 512-expert MoE — raise the ubatch so the whole expert
bank is read once for more tokens — **does not work**:

| `-ub` | pp2048 tok/s |
|---:|---:|
| 256 | 347.71 ± 1.72 |
| **512** | **391.42 ± 1.13** |
| 1024 | 380.91 ± 0.67 |
| 2048 | 336.72 ± 0.11 |

Per token the MoE does get cheaper (1024.3 -> 794.0 µs, 0.78x) and so do the
tiny-N F32 matmuls (0.35x) — but **the glue gets 1.57x more expensive**, and
that is the larger term. The mechanism is the MALL: the residual stream is
`[10240, T]` F32, which is **20.97 MB at T=512 — just inside the 32 MiB
MALL** — and 83.9 MB at T=2048, which is not. `MUL` drops from 362 GB/s
(above the DRAM bus, so it was reading out of cache) to 294, `CONCAT` from 48
to 13.

Two consequences. **For us**: a 10240-wide F32 residual is the wrong
representation — fp16 halves it, and fusing the gate/collapse/recombine into
the matmul epilogues removes most of the traffic rather than caching it. Then
the ubatch can be chosen for the MoE bank instead of for the MALL. **For the
baseline**: 391.4 tok/s is llama.cpp at its best ubatch, so it is the right
number to quote.

A loose end, stated rather than explained: at `-ub 2048` the GPU graph is
4.556 s but the wall clock is 6.083 s, so **at least 25% is host-side** there,
against ≤7% at `-ub 512`. Whatever it is, it does not scale with graph count
(both ubatches leave ~1.7 s unaccounted in their logger-enabled runs) and it
is llama.cpp's problem, not the architecture's.

## What our own kernels do on these shapes

Every row is `results/shapes.csv`'s best measured variant for that exact
`(M, N, K)`, at **fp16** — §2.2 measures Q4 WMMA at **2.10x fp16** on a MoE
block, so the quantised versions should be at worst this:

| shape | llama.cpp (q8_0) | ours (fp16) | |
|---|---:|---:|---:|
| `hc.inject` [10240, 4] F32 | 164.8 ms | fused into hc.down | **~0** |
| `hc.down` M=2048 N=320 K=10240 | 185.3 ms | 36.5 ms (34943 GF/s) | **5.08x** |
| `qsa.k` (bf16) | 3.8 ms | 0.5 ms | 7.83x |
| `moe.shared` | 46.0 ms | 15.5 ms (40592 GF/s) | 2.96x |
| `hc.up` | 136.5 ms | 57.4 ms | 2.38x |
| `attn.o + la.out` | 193.0 ms | 89.9 ms (34384 GF/s) | 2.15x |
| `la.gate` | 119.1 ms | 65.6 ms (35364 GF/s) | 1.82x |
| `la.qkv + ple.key` | 192.0 ms | 167.4 ms (23726 GF/s) | 1.15x |
| `attn.q` | 75.8 ms | 68.3 ms (22654 GF/s) | 1.11x |
| **all matmul** | **1192.0 ms** | **529.0 ms** | **2.25x** |

llama.cpp's dense GEMM is not bad — 20.7 TFLOP/s on `la.qkv`, which is 53% of
§2.7's best-real-kernel 39.0 — and on the two widest-N shapes we barely beat
it. Where we win by 2-5x is exactly where N is small (320, 640, 4): our WMMA
kernels hold 34-40 TFLOP/s down to N=320 and llama.cpp falls to 6.9.

And the MoE, from §2.2's already-built Q4 grouped kernel at 2048 tokens —
**13.37 ms a block, 48 blocks, 642 ms** against llama.cpp's 1626, **2.53x**.
The bank floor is 64 GB at 4.25 bits, i.e. **273 ms at 236 GB/s**, so even
§2.2's kernel is 2.4x above the bus and there is more there later.

## The revised prefill target

    llama.cpp, GPU graph only, -ub 2048       4556 ms    450 tok/s
    llama.cpp, wall clock, -ub 512                       391 tok/s
    ours, matmul from results/shapes.csv,
      MoE from §2.2, glue unfused              2879 ms    711 tok/s   1.8x
      … with 3/4 of the glue fused             1768 ms   1159 tok/s   3.0x
    matmul-only floor (24.7 TFLOP @ 39 TF/s)    633 ms   3234 tok/s
    expert-bank floor (64 GB @ 236 GB/s)        273 ms

**~1150 tok/s is the number L6 should be scoped against**, and how close it
gets is decided by epilogue fusion, not by the matmuls. LLM.md's ~2000 was
optimistic for a reason it can now name: it priced the matmuls and not the
10240-wide residual they hand to each other.

## What this changes for L2-L7

1. **The hyper-connection block is the design problem, not DeltaNet.** 30.6%
   glue + 12.2% tiny-N F32 + the `hc.down`/`hc.up` pair is over half the
   graph, and almost all of it is one fused kernel: normalise, gate, collapse
   the 4 branches, project down to 320 **and** 4 in the same pass. Build that
   before the stack, not after.
2. **Keep the residual in fp16 and keep it off DRAM.** Finding 5 is a warning
   that at any realistic ubatch this tensor does not fit the MALL in F32.
3. **DeltaNet can be built for correctness first and tuned later.** It is 2.7%
   of prefill. L3's risk is that it is hard to get *right*, not that it is
   slow — which is the opposite of how LLM.md sized it.
4. **The expert kernel is already competitive**; §2.2's Q4 grouped GEMM is
   2.53x the reference and 2.4x off the bus. That gap is the one worth
   revisiting after L6, not before.
5. **Do not raise the ubatch to amortise the expert bank** until the residual
   traffic is fixed. It is a net loss today.

## Decode, for free

The same run timed 17 decode graphs, mean **41.73 ms** (24.0 tok/s of GPU time
against the 25.15 measured — the logger costs 4%). The block broken out below
is the last of them, 41.3 ms, and it tells a different story from prefill:

| category | ms | share |
|---|---:|---:|
| dense matmul (`MUL_MAT_VEC`) | 21.1 | 51.2% |
| MoE experts | 8.9 | 21.5% |
| elementwise glue (2239 dispatches) | 5.9 | 14.3% |
| matmul f32 (tiny-N) | 2.5 | 5.9% |
| routing, DeltaNet, norms, attention, rope | 3.0 | 7.1% |

**llama.cpp's decode matmuls are already near the bus**: `la.qkv` reads 27.9
MB of q8_0 in 124.9 µs = **223 GB/s**, `lm_head` 675 MB in 2.93 ms = **230
GB/s**, `hc.down`/`hc.up` 198 GB/s. The experts are softer at 159-185 GB/s.
So the 6.334 GB a token needs **26.2 ms at 242 GB/s** and llama.cpp spends
41.3 — and the missing **15.1 ms is almost entirely ops that read no weights
at all**. The 2896 dispatches that are not a weight matmul cost **8.9 ms,
22% of the step**, ~3.1 µs apiece; the tiny-N F32 matmuls add 2.5 ms more.
That floor is not the logger's barriers — the un-instrumented step is 39.8 ms
against the logger's 41.3, **3.6% apart** — it is what a dependent chain of
small kernels costs here, against §4.1's 300 ns for an empty shader and
§1.8's 1163 ns median in a real grouped GEMV. L1 priced 2081 decode
dispatches at 0.62 ms of launch cost, 5% of a step; measured, the real
number is **an order of magnitude larger**.

**That is the whole of the 25.15 -> 38.2 tok/s gap, and it is L7's brief**:
decode is not won by a better GEMV — llama.cpp's is fine and §1.1's is better
— it is won by not dispatching 3837 kernels.

## How to reproduce

    L=/home/kube/repos/llama.cpp
    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    # the two perf-logger runs behind results/l2a_prefill_ops.csv
    GGML_VK_PERF_LOGGER=1 $L/build/bin/llama-bench -m $M -p 2048 -n 16 -r 1 -ub 512  2> ub512.err
    GGML_VK_PERF_LOGGER=1 $L/build/bin/llama-bench -m $M -p 2048 -n 0  -r 1 -ub 2048 -b 2048 2> ub2048.err

    # the ubatch sweep of finding 5
    $L/build/bin/llama-bench -m $M -p 2048 -n 0 -ub 256,512,1024,2048 -b 2048 -r 2

    # the kernels compared against, already committed
    grep qwen3.8-flash-next results/shapes.csv     # per-shape best, fp16 WMMA
    research/2.2-q4-coopmat.md                     # 13.37 ms/MoE block at Q4
    research/3.5-grouped-moe-gemm.md               # the fp16 block it replaced
