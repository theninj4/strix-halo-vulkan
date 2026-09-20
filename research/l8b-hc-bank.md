<!-- LLM.md L8b. The hyper-connection block on L8's dense bank: a Q8 split
     that falls on a column block rather than on a row, a -DQ8B arm of
     llm_hc_gemv.comp that rounds its weights in fp16 because there is no LDS
     to round them through, a collapse scratch cut to one m-tile, and the two
     ladders that moved as a result. Cited from llm/gpu.go,
     shaders/llm_gemm.comp, shaders/llm_hc_gemv.comp, llm/bank.go,
     llm/gpu_test.go, cmd/llm/bench.go and results/l8b_*.csv. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L8a](l8a-dense-bank.md) · [L7d](l7d-decode-kernels.md) · [L2c](l2c-hc-kernel.md) · phase 2

# L8b — the last dense family: the block is 2.14x at one token, bit for bit

`results/l8b_decode.csv` · `results/l8b_graph.csv` · `results/l8b_graph_fp16.csv` · `results/l8b_hc.csv` · `results/l8b_hc_fp16.csv`

**Result: the hyper-connection block at one token is 37.0 us a mixer against
79.2 — 2.14x — and the arithmetic is unchanged.** As in L8a this is an
equality rather than a tolerance: every tensor the block produces, on every
rung of both ladders and every split of the GEMV, is identical to the fp16
bank's to the last bit (`TestHCGPUQ8IsTheHalves`), and the model's completion
is the same token list down to `Lisbon.`

End to end: **decode 14.71 tok/s against 14.07**, a token 71.1 ms to 68.0,
residency 82.40 GB to **81.89**, and the block's own share of a step 13.3% to
**8.7%**. Prefill is **1041.1 tok/s at ubatch 2048** against L8a's 1044.8 and
643.4 at 512 against 644.1 — unchanged inside the 0.36% two runs of this
benchmark agree to, which is the gate this stage had to clear rather than a
number it set out to move. Two decode runs agree to **0.34%** (14.71 / 14.76).

| | L8a | L8b | |
|---|---:|---:|---|
| decode | 14.07 tok/s | **14.71** | 1.045x |
| a token | 71.1 ms | **68.0** | |
| the block, at one token | 79.2 us a mixer | **37.0** | 2.14x |
| the block, a token | 9.5 ms | **5.9** | 1.61x |
| the block's bytes a token | 1.303 GB | **0.740** | 1.76x |
| a token, all in | 6.61 GB | **6.05** | |
| this bank's own ceiling | 36.6 tok/s | **40.0** | |
| prefill, ubatch 2048 | 1044.8 tok/s | 1041.1 | 2.66x llama.cpp |
| resident | 82.40 GB | **81.89** | |

L8a left this block out for three reasons and each one turned into a finding.
Its down projection is MODE 0 with `inject` fused into its last tile — four
F32 rows that do not begin on a column block, so L8a-2's tail does not fit as
it stands (L8b-1). Its up projection is MODE 1, whose epilogue collapses the
gate through LDS the unpack would have to share (L8b-3). And at one token
neither runs: L7d replaced the down projection with a split-K GEMV that reads
the fragment tiling directly and has **no LDS stage to round a weight
through** (L8b-2).

---

## L8b-1: the split is a column block, not a row, and 32 rows are staged twice

`inject` is a [10240, 4] F32 matrix applied to the same normalised activation
`down` reads, and since L2c it has been four more output columns on a matrix
that already has 320 — rows 320..323 of a fused N padded to 336. L8a-2's tail
takes the rows a checkpoint does not ship as Q8_0 out of the int8 plane and
stages them as a fp16 fragment-tiled matrix of their own, on one condition:
**a whole column block has to fall on one side of the split**, because a
compare inside the k-loop cost 1.27x on a matrix with no tail at all (L8a-3).

320 is not a multiple of the down ladder's BN of 48, and 48 is not free to
change: the fused N is 336, which is 21 tiles, and 21 is 3 x 7.

So the split is the column block that **contains** the first non-Q8 row —
`pc.lowRank` rounded down to BN, 288 here — and the 32 low-rank rows between
288 and `inject` are staged twice: once in the int8 plane, where they are
never read, and once in the tail, where they are. Both kernels derive it from
`pc.lowRank` and their own column block, so it costs no push field, and the
push block has been full since L5b.

What the duplication costs, per mixer:

| | bytes | |
|---|---:|---|
| down, halves | 6.88 MB | 336 x 10240 |
| down, int8 + scale plane | 3.65 MB | staged for all 336 columns |
| down, fp16 tail | 0.98 MB | 48 columns, of which 4 are inject |
| **down, read a token** | **4.15 MB** | the int8 past 288 is skipped |
| ideal, a 16-column tail | 3.82 MB | 0.33 MB less |

0.33 MB a mixer, 32 MB a decode token, 0.5% of a step — against a branch that
stays per workgroup. The narrower tail is available whenever it is worth 0.5%;
it is not worth a second unpack arm today.

The up projection needs none of this. `hc_up.weight` is Q8_0 for all 10240 of
its rows, so MODE 1's tail branch is **compiled out** rather than guarded —
which also keeps its k-loop to one `coopMatLoad` of B, the thing L8a-3 was
about. That matters twice over here, because in MODE 1 `pc.gateOff` is not a
weight at all: it is the arena the validation gate is optionally written to.

## L8b-2: the GEMV has no LDS to round through, so it rounds in fp16

The decode rung is `llm_hc_gemv.comp`, and the Q8 arm of it is almost free:
the §2.8 tiling hands each lane one output column and sixteen consecutive k of
it, which as int8 is sixteen consecutive *bytes* — four `uint` loads — and one
scale, because a 32-element group is two whole k-tiles and the tile a lane
owns sits inside one of them.

The catch is the rounding. The GEMM's Q8 arm is bit-exact because it *stores*
`float16_t(float(q) * d)` into LDS, and what the matrix cores then multiply is
a half. This kernel has no LDS stage at all, and `float(float16_t(x))` in a
register is folded away by RADV (D10) — so the rounding would simply not
happen, the GEMV's effective weight would be the unrounded `q*d`, and the two
kernels would disagree by a weight ulp accumulated over a 10240-long dot
product. On `inject`, whose values reach |71.7|, that is ~1e-2: an order of
magnitude past the bound L7d's own agreement test asserts.

The fix is to do the multiply **in fp16**:

```glsl
float16_t d = w16[sbase + ((kt0 + c + sub) >> 1) * TILE];
sum += float(float16_t(bitfieldExtract(int(word), t * 8, 8)) * d) * float(hact[x]);
```

`float16_t(q) * d` is a real f16 instruction with two exact f16 operands, so
there is no f32→f16→f32 pair for the compiler to fold, and its correctly
rounded result is the **same half** the GEMM's LDS store produces: q carries
at most 8 significant bits and d at most 11, so their f32 product is exact and
the conversion to fp16 is the only rounding either path performs.

That is what makes `TestHCGPUQ8IsTheHalves` an equality over the decode rungs
as well as the GEMM ones — 9 GEMM pairs and 6 splits, every tensor identical.

**And the GEMV ladder slides one rung along §5.1b's 4 KB rotation**, which is
D12 rather than a surprise: a workgroup's slab is now
`(gemmK/16/KSLABS) * 256` bytes and not 512, so the whole table shifts.

| KSLABS | 8 | 16 | 32 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|
| slab, bytes | 20480 | 10240 | 5120 | 4096 | 2048 | 1024 |
| x 4 KB | 5 | 2.5 | **1.25** | 1 | 0.5 | 0.25 |
| us a mixer | 18.0 | 15.5 | **12.8** | 30.9 | 21.6 | 13.5 |
| GB/s | 231 | 269 | **326** | 135 | 193 | 308 |

The two whole multiples of 4 KB are 8 and 40, and they are the two slow rungs
— 40 by 2.4x against 32. On halves the same law picked 160 (230 GB/s) and 32
(221); on bytes it picks 32 (326) and 160 (308), which are the same two
strides one rung along. `PlanFor` now names the bank.

## L8b-3: the collapse's scratch was [BM][BN], and the unpack wanted the widest row block there is

MODE 1's epilogue stores its gate tiles to LDS and reads them back with
ordinary indexing, because the cooperative-matrix extension does not define
how matrix elements map to lanes. At [BM][BN] floats that is 4, 8 and 16 KB
for the three rungs — and this kernel gets slower as its LDS grows, because it
is latency-bound on two streams it cannot prefetch. On the **fp16** bank, with
no unpack involved at all, BM=64 was 1.38x BM=32 at 2048 tokens.

The Q8 arm wants the opposite of what that ladder says. Its unpack costs
`BN*BK` element conversions per slab against `WM*WN*BK_TILES` matrix steps —
**256/WM conversions per step, and nothing else in the shape changes it**. At
WM=1 that is four VALU-ish instructions a lane for every 16x16x16 multiply,
and the measured cost is exactly what that predicts: with the old epilogue the
Q8 up projection at 2048 tokens was 1003 us at its best rung against the fp16
arm's 721, **1.39x slower for reading 1.8x fewer bytes**. Deeper K-slabs do
not touch the ratio and confirmed it — BK_TILES=4 was 1334 us and BK_TILES=1
was 984.

So the scratch was cut to **one m-tile**: store `acc[i][*]`, collapse those 16
token rows, move on. It is 4 KB at every rung instead of BM x BN, at the price
of two barriers a tile instead of one, and the arithmetic is untouched — the
same four gate values of the same feature, in the same order, into the same
output, which is why the equality test still holds across the change.

With 4 KB of gate beside 5 KB of slab, a rung wide enough to amortise the
unpack fits. The up projection's best rung, microseconds a mixer:

| T | | up_m1 | up_m2 | up_m4 | up_m8 |
|---|---|---:|---:|---:|---:|
| 512 | fp16 | 235.2 | 152.6 | **142.0** | 159.2 |
| 512 | Q8 | 224.0 | 166.0 | **152.8** | 159.4 |
| 2048 | fp16 | 947.3 | **638.2** | 647.8 | 793.6 |
| 2048 | Q8 | 1005.8 | 774.2 | **694.5** | 778.6 |

Both banks gained: the fp16 arm's own best went 176.3 → 142.0 at 512 and
720.8 → 638.2 at 2048, so the control this stage is measured against moved
too, and the table above is both sides after the change. m8 is new and wins
nowhere, but it is what shows the ladder has a top: past BM=64 the A operand's
re-reads stop paying.

## L8b-4: what the block costs, both banks, at every length

`results/l8b_hc.csv` and `results/l8b_hc_fp16.csv`, microseconds a mixer at
each projection's best rung. `norm` and `combine` read no weight and are the
same on both banks.

| T | down fp16 | down Q8 | | up fp16 | up Q8 | |
|---:|---:|---:|---|---:|---:|---|
| 1 | 30.0 | **12.8** | 2.34x | 40.0 | **16.1** | 2.48x |
| 64 | 234.7 | **204.2** | 1.15x | 39.5 | 30.9 | 1.28x |
| 512 | 254.5 | 253.2 | 1.01x | 142.0 | 152.8 | 0.93x |
| 2048 | 457.3 | 525.6 | 0.87x | 638.2 | 694.5 | 0.92x |

The shape of that is the whole story of this stage. **At one token the bank is
the kernel**: nothing is re-read, both projections stream their weight once,
and halving the bytes is 2.3-2.5x. **At 2048 tokens the bank is nearly
irrelevant**: the down projection's 6.9 MB is read 32 times and the up's 6.6
MB 32 times, so most of that traffic is served by the 32 MiB MALL, and what is
left is the unpack's ALU — 1.15x and 1.09x against the fp16 arm, for 1.7-1.9x
fewer DRAM bytes that the cache had already hidden.

Against llama.cpp's own graph at ubatch 512 the block is unchanged as a
comparison — 226.6 ms of its 1164.7 ms prefill graph becomes 60.7 — because
neither side of that moved.

## L8b-5: end to end

`results/l8b_decode.csv`, `results/l8b_graph.csv`. The whole model, 48 layers,
81.89 GB resident:

| | L7d | L8a | **L8b** |
|---|---:|---:|---:|
| decode | 11.89 tok/s | 14.07 | **14.71** |
| a token | 84.1 ms | 71.1 | **68.0** |
| prefill, ubatch 512 | 622.5 tok/s | 644.1 | 643.4 |
| prefill, ubatch 2048 | 990.6 tok/s | 1044.8 | 1041.1 |
| resident | 84.20 GB | 82.40 | **81.89** |

Where a token goes now, milliseconds and percent:

| block | L8a | L8b |
|---|---:|---:|
| MoE | 28.5 (40.1%) | 28.6 (42.1%) |
| DeltaNet | 18.8 (26.5%) | 18.9 (27.8%) |
| hyper-connection | 9.5 (13.3%) | **5.9 (8.7%)** |
| attention | 5.6 (7.8%) | 5.6 (8.2%) |
| lm head | 3.5 (4.9%) | 3.5 (5.1%) |

The gate is the text, and it is the same text: `TestGraphLogits` returns
llama.cpp's own argmax (**561**) out of llama.cpp's own top ten with the drift
unchanged at 0.881% over 48 layers, and `-gen -n 128` produces L7c's
completion exactly — the same list of capitals, the same `<think>` block, the
same `Lisbon.`

## L8b-6: what is left

**Every dense weight in this model is now staged in the width the checkpoint
ships it in.** A token reads **6.05 GB** against the 9.67 of L7d; the
ceiling this bank allows is **40.0 tok/s** and we are at 37% of it. What is
still halves anywhere is the PLE block's one projection, which runs once at
layer 1 for 0.6% of a step, and the three families the checkpoint does not
ship as Q8_0 — the DeltaNet's F32 alpha and beta, the indexer's BF16
projections, and now the hyper-connection block's F32 `inject` with the 32
low-rank rows the split leaves beside it.

So phase 2's first half is finished and it bought 11.89 → 14.71 tok/s, 1.24x,
without changing a single number the model computes.

**And the attribution has not moved since L8a said it — it has sharpened, and
it reordered the plan.** We are at **89 GB/s of a 242 GB/s bus**, lower than
L8a's 93 because the bytes fell faster than the time did. The MoE is 42% of a
token at **56 GB/s, 4.3x off its own bytes**; the DeltaNet is 28% at 119,
2.0x off; this block is now 125 and the lm head — the same kernel family over
the same kind of bank — is **194 and 1.25x off**. If every block ran at the
head's rate a token would be 31 ms rather than 68: **32 tok/s from kernel work
alone**, with no bank change and so nothing to argue about on accuracy.

That is worth more than L8c's ~1.9x on the dense half, so the task list now
runs **L8d — the MoE at decode — before L8c**, and the re-quantisation lands
on a faster baseline for having waited. **The bytes stopped being the thing to
fix**, and L8b is the stage that proved it twice over: once by taking 1.76x of
them out of this block for a 2.14x that only decode saw, and once by costing
1.09-1.15x at ubatch 2048 for the same bytes (D14).
