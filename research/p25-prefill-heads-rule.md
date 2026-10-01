# P25 — which gathered attention a prefill chunk takes

**2026-10-01.** One rule changed and nothing else. At the served
`-llm-batch 8192`, a prompt's first chunk prefills at **1518.5 / 1523.1 →
1605.6 / 1601.4 tok/s (1.054x)**. 2048 and 4096 are unmoved, by
construction.

## What was wrong

P17 put a token's twelve query heads on the gathered attention's fragment
rows, so a workgroup reads one token's 2051-cell selection rather than a
sixteen-token union. It measured the loss at depth zero on a 2048-row chunk
(−1.1%) and gated the build on `past >= selWidth`, which is where the chunk
*starts*. A wide chunk from a cold cache is mostly rows past the width,
though. At 8192 rows from cell zero, three quarters of them select 2051 of up
to 8191 cells, and their sixteen-token union is most of the prefix. The
token-row kernel then walked that union plus a 3 ms mask pass, and
`attn.attn` was the largest label of the pass: 69.5 ms a layer, 15.5% of
5.38 s.

## The measurement

`cmd/llm -attn` from cell zero (layers 3 and 7, the shipped `q5_k` bank),
µs a layer, two runs agreeing to 1%:

| rows | share past the width | token rows | heads on rows | |
|---:|---:|---:|---:|---:|
| 2048 | 0.00 | 12 286 | 13 410 | 0.92x |
| 3072 | 0.33 | 22 079 | 23 572 | 0.94x |
| 4096 | 0.50 | 34 620 | 33 757 | 1.03x |
| 5120 | 0.60 | 50 996 | 44 865 | 1.14x |
| 6144 | 0.67 | 69 267 | 56 544 | 1.22x |
| 8192 | 0.75 | 118 590 | 79 603 | 1.49x |

The cost follows the share of the chunk's rows past the width, and the
crossover is between a third and a half. So `AttnGPU.headRows` now takes
the heads build when `5·(past + rows − width) ≥ 2·rows`. A chunk that
starts past the width is all of its rows, which is P17's rule unchanged.
Decode is untouched (it always takes heads). `LLM_ATTN_GATHER_HROWS` and
`LLM_ATTN_GATHER_HROWS_MIN` still override.

## Whole model

`cmd/llm -graph -tokens 2048,4096,8192 -ctx 8192`, 48 layers, shipped banks
under `LLM_BANK_CACHE`, old and new binaries interleaved twice
(`results/p25_graph_{old,new}{1,2}.csv`, labels beside them):

| tokens | old | new | |
|---:|---:|---:|---:|
| 2048 | 1567.8 / 1567.3 | 1569.7 / 1568.5 | 1.00x |
| 4096 | 1592.1 / 1596.8 | 1599.4 / 1598.7 | 1.00x |
| 8192 | 1518.5 / 1523.1 | **1605.6 / 1601.4** | **1.054x** |

`attn.attn` at 8192 goes from 69.5 to 48.6 ms a layer.

## Exactness

Prefill under the gathered kernel was already a reassociation (P14-2,
P17): the same cells, folded in different chunks. This changes which of the
two gated builds a chunk runs and nothing about either. Two dispatch-shape
tests read the token-row chain on a full 4096-row pass from cell zero, and
now pin it (`SetHeadRows(false)`, as `TestAttnGPUGatherSelectsTheSameCells`
already did): `TestAttnGPUGatherIsDispatched` and
`TestAttnGPUSelectionEngagesAt4k`. The heads chain's own dispatch gate is
`TestAttnGPUHeadRowsIsTheGather`. The full `llm` suite passes (296 s).

## What is left at 8192

Labels, 5.10 s a pass after the change: `moe.up` 16.2%, `dn.qkv` 12.4%
(~40 TFLOP/s, the dense GEMM's rate), `attn.attn` 11.4%, `moe.down` 10.9%,
`hc.cn` 8.5% (at the bus), `dn.scan` 6.2%, `hc.up` 5.4%, `dn.out` 4.5%,
`moe.combine` 4.3% (~220 GB/s), `hc.down` 4.0%.

- **The heads build is the attention lever.** About 361 GFLOP of useful
  work a layer at 8192 rows in 48.6 ms is ~7.4 TFLOP/s: twelve of sixteen
  rows useful, S and P through LDS, and K/V staged 32 bytes a cell a
  head-dim group. M11c's transposed shape (P^T on the B side, no LDS) and
  G4b's shared K/V blocks are what took H3's attention to 78% per clock. At
  most ~5% of an 8192 pass, and about as much at depth, where P17 left
  `attn.attn` at ~9% of a token.
- **A split at the width inside one chunk** (token rows below, heads
  above) would add ~1.2% at 4096 and ~0.3% at 8192. It needs a row base,
  and the push block is full.
- **`moe.combine` is not a fusion target.** It runs at the bus. The down
  GEMM's fp32 rows (840 MB a layer at 8192) are about half the down's
  traffic, and the ordered ten-way sum is what keeps it bit-exact.
- **`hc.up` / `hc.down` are small-N dequant GEMMs**, not streaming kernels:
  K = 320 or N = 324, at ~19–27 TFLOP/s.
