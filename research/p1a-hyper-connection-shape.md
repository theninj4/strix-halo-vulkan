<!-- LLM.md P1a. The hyper-connection block's 485 dispatches: what fusing the
     weightless half is worth, and why the other half is where it is. Cited
     from shaders/llm_hc_cn.comp, llm/gpu.go, llm/graph.go and
     results/p1a_*.csv. -->

[← LLM.md](llm-vertical.md) · [← LLM2.md](llm-review.md) · [research index](README.md) · [P1](p1-decode-attribution.md) · [L7d](llm-vertical.md) · phase 2

# P1a — the hyper-connection block: a boundary that was two dispatches, and a grid that is pinned at 160 workgroups

**Result: 94 dispatches out of the step, bit for bit, decode 32.59 →
32.93 tok/s and prefill at ubatch 2048 1070.8 → 1089.2** — both arms measured on this build, an hour apart, at the
shipped bank. The scatter that closes one mixer and the grouped RMSNorm that
opens the next are the same 2560 values per (token, stream) written and
immediately read back, so `llm_hc_cn.comp` does both in one workgroup. The
pass goes **1501 dispatches to 1407** and the step **30.69 ms to 30.37**.

**And the other half of P1a's 1.56 ms is priced and not taken.** P1 read
`hyper_conn`'s 114.6 GB/s as "shape, not bytes" and proposed fusing or
batching the two projections. The ladder says what the shape *is*, and it is
neither: **a one-token dispatch on this device is short of workgroups, and
`up` is pinned at 160 of them by the collapse's 64-column block.** Unpinning
it is worth **2.3 us a mixer, 0.22 ms a token, +0.24 tok/s** — measured
against the down projection's own grid sweep rather than guessed — and it
changes the row permutation every `up` rung reads, so it is written down here
instead of built.

## What the block costs, before and after

`-gen -attrib` over 64 tokens, `LLM_DENSE_BANK` at the five 4.5-bit families,
the same binary either way; `LLM_HC_NOFUSE=1` is the control arm.
(`results/p1a_decode_attrib.csv`, `results/p1a_decode_attrib_nofuse.csv`.)

| label | n/pass | us each | ms/token | | n/pass | us each | ms/token |
|---|---:|---:|---:|---|---:|---:|---:|
| | **the pair** | | | | **fused** | | |
| `hc.up` | 97 | 16.7 | 1.593 | | 97 | 16.5 | 1.573 |
| `hc.down` | 97 | 15.0 | 1.435 | | 97 | 15.0 | 1.437 |
| `hc.norm` | 97 | 6.5 | 0.617 | | 3 | 9.7 | 0.029 |
| `hc.combine` | 96 | 4.0 | 0.380 | | 2 | 5.8 | 0.011 |
| **`hc.cn`** | — | — | — | | **94** | **7.3** | **0.676** |
| `hc.down_reduce` | 97 | 1.3 | 0.122 | | 97 | 1.3 | 0.124 |
| **block** | **484** | | **4.147** | | **390** | | **3.850** |
| step | 1501 | | 30.69 | | 1407 | | 30.37 |
| decode | | | **32.59 tok/s** | | | | **32.93 tok/s** |

A boundary was **10.5 us** and is **7.3**. Ninety-four of the 96 combines
fuse; the two that do not are the PLE block's layer, where the wide residual
is moved out of this arena and back, and the end of the pass, where the final
mixer is reached through a row move. The block falls from 13.5% of the step to
12.7%.

**The weight-streaming half did not move and was not expected to**: 0.360 GB
a token over `up` + `down` + the reduce is **114.9 GB/s** against P1's 114.6.
P1a's gate asked for the label count and the GB/s to move together. The label
count moved; the GB/s did not, and the rest of this file is why.

## Finding 1 — the boundary was one workgroup's work said twice

Both kernels were already one workgroup per (token, stream) over the same
2560 values. `llm_hc_combine.comp` reads `res`, reads the block output,
writes `res`; `llm_hc_norm.comp` then reads `res` back, reduces it, and writes
`xn`. So the fusion needs no new decomposition — a workgroup combines its own
stream, keeps the result in registers, and norms it there:

    res + out*w   ->  registers  ->  sum of squares  ->  xn
                  \-> res

191 KB of arena traffic a boundary becomes 151, and one dispatch becomes none.

**It is bit-exact against the pair**, which is the gate
(`TestHCFusionIsTheCombineThenTheNorm`, and the same equality through four
layers of the graph in `TestHCFusionRunsThePass`). That is why the workgroup
is 256 threads and not wider: the norm's sum over 2560 squares is a per-thread
strided partial and a 256-way tree, and both depend on how many threads there
are. A 1024-wide workgroup would have been a different sum — and, as finding 2
shows, not a faster dispatch either, because four workgroups is four
workgroups.

### And it is 1.7% of prefill, for a different reason

At one token the fusion saves a dispatch and 40 KB. At **ubatch 2048** the
wide residual is 84 MB, so what it saves is one 84 MB read a boundary — 7.9 GB
a pass — and that is what shows: the block goes **330.1 ms to 294.6** and the
pass **1912.6 to 1880.3, 1070.8 to 1089.2 tok/s, +1.7%**, same build, control
arm on the same run of the machine.

**At ubatch 512 it is worth nothing** (66.0 ms against 66.8, 671.5 tok/s
against 668.1 — noise either way), and that is **D14** again from the
activation side: 512 rows of the residual are 21 MB, inside the 32 MiB MALL,
so the read the fusion removes was never going to DRAM in the first place.
The lever only exists once the tensor stops fitting.

### The ulp that cost an afternoon

The first version failed the equality by **one ulp on 23% of the residual**,
and the SPIR-V of the two kernels is identical where it matters — `OpFMul`
then `OpFAdd`, in both. The difference is the *backend*: in the scatter the
sum has one reader and ACO leaves it a multiply and an add; in the fused
kernel it has two — the store and the square — and ACO contracts the pair into
a fused multiply-add, which rounds once where the scatter rounds twice.
`precise` on the register array is the only thing that says no to it, and it
costs nothing. Written down because nothing else in this vertical has hit it:
**a fusion can change a value by giving it a second reader.**

## Finding 2 — a one-token dispatch is short of workgroups, and the down GEMV's ladder is the ruler

The down projection's decode rungs differ only in **how many ways they split
K**, so they read the same 1.94 MB of the same bank with a grid that varies
twenty-fold. That makes the ladder a measurement of the grid itself
(`results/p1a_hc_grid.csv`, 24 mixers staged, `-tokens 1 -ladder`; by **D16**
these are MALL numbers and the point is the ratio, not the rate):

| rung | slabs | **workgroups** | KB a workgroup | us | GB/s |
|---|---:|---:|---:|---:|---:|
| `down_gemv8` | 8 | **168** | 11.8 | **9.26** | 291 |
| `down_gemv16` | 16 | 336 | 5.9 | 6.47 | 416 |
| `down_gemv32` | 32 | 672 | 3.0 | **5.54** | 486 |
| `down_gemv40` | 40 | 840 | 2.4 | 7.59 | 358 |
| `down_gemv80` | 80 | 1680 | 1.2 | 6.08 | 447 |
| `down_gemv160` | 160 | 3360 | 0.6 | 5.90 | 461 |
| **`up_m1`** | — | **160** | **11.5** | **10.87** | 171 |

**168 workgroups is 9.26 us and 672 is 5.54, for identical bytes off an
identical bank.** (`gemv40` is L8c-6's 4 KB-rotation outlier and is the one
rung the trend does not fit.) And the last row is the finding: `up_m1` reads
**11.5 KB a workgroup in 160 workgroups and takes 10.87 us** — which is
`down_gemv8`'s 11.8 KB in 168 workgroups taking 9.26, to within the
difference in bytes. Two different kernels, two different modes, one
cooperative-matrix GEMM and one split-K GEMV, land on the same number because
they are launched at the same width.

So **`up`'s cost is not MODE 1's M=1 waste.** The `up` ladder's four rungs
(`m1` 10.88, `m2` 15.41, `m4` 24.48, `m8` 42.94 us) vary `BM` — 16 to 128 rows of
which one is real — and they all launch **160 workgroups**, because a rung's
grid is `wide / BN` and `BN` is 64 for every one of them. The ladder has never
varied the thing that matters.

## Finding 3 — what pins the grid is the collapse, and the price of unpinning it

`BN` cannot fall below 64 in MODE 1 because the epilogue is a reduction over
the hyper-connection streams: `mixed[i] = mean_c(xn[c][i] * sigmoid(gate[c][i]))`
needs the four streams of one feature inside one workgroup, and `packUpB`
buys that by permuting the up projection's output rows so that **a 64-column
block is 16 features x 4 streams**. `build` checks it — `bn % (coopMatTile * hc)`
— and every rung obeys.

A 16-column block of **4 features x 4 streams** would satisfy the collapse just
as well and would let `BN` be 16, which is `10240/16 = 640` workgroups. What
that is worth, from the table above: in the MALL `up` would go from 10.87 to
about `5.54 * 1.84/1.94 = 5.3` us; in the whole model the DRAM component is the
same either way — `down` costs 15.0 us in the model against 5.54 in the MALL,
so 1.84 MB carries about 9.0 us of DRAM — and `up` lands at **~14.2 us against
16.5**. That is **2.3 us a mixer, 0.22 ms a token, +0.24 tok/s**, or 0.7%.

**It is not taken.** The permutation is baked into the staged bank, so the
index change is not a new rung beside the old ones — it is every `up` rung's
epilogue, on the prefill path, where the projection is 638-694 us a mixer at
ubatch 2048 and where the collapse has been correct since L2c. 0.7%
of decode does not buy that, and the number is here so the next person does
not have to re-derive it.

## What this leaves, and where it re-ranks

The block is **3.850 ms, 12.7% of the step, for 8.4% of the bytes**, and the
remaining 3.01 ms of it is two projections running at the width their grids
allow. Against the other items on P1's list:

| ms | item | status |
|---:|---|---|
| ~~0.30~~ | `hc.norm` + `hc.combine` | **done**, and bit-exact |
| 1.19 | `moe.down` at 147.5 GB/s against `moe.up`'s 200.6 | **P1b**, and a ladder rather than a kernel |
| 1.96 | record + hand-over | P1c |
| 0.73 | the shared expert's pair at 136.5 GB/s | P1b |
| 0.22 | `up`'s grid, if `BN` could be 16 | priced above; not taken |

**P1b is now the largest item on the list by a factor of four**, and finding 2
is the reason to expect it to answer: `moe.down` and `moe.up` read the same
bank in the same layer, and the one thing nobody has varied for `down` is the
grid its row block implies.

## Files

- `shaders/llm_hc_cn.comp` — the fused scatter and norm, and the `precise` note
- `llm/gpu.go` — `graph(mixer, prev, combine)`, `RunCombineMix`, the `cn` pipeline
- `llm/graph.go` — the held combine, `flushHC`/`mixHC`, and `HCFuse()` / `LLM_HC_NOFUSE`
- `llm/graph_fuse_test.go` — the two equalities
- `results/p1a_decode_attrib.csv`, `results/p1a_decode_attrib_nofuse.csv` — both arms
- `results/p1a_hc_grid.csv` — the ladder finding 2 reads
