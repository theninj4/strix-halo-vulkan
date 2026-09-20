<!-- LLM.md L8a. The dense bank in the width the checkpoint already ships:
     int8 fragment tiles with an fp16 scale per 32 elements, a -DQ8B arm of
     llm_gemm.comp that unpacks a slab into LDS, and the fp16 tail the three
     non-Q8_0 families keep. Cited from llm/bank.go, shaders/llm_gemm.comp,
     shaders/llm_common.glsl, llm/gpu_head.go, llm/gpu_deltanet.go,
     llm/gpu_attn.go, cmd/llm/bench_head.go and results/l8a_*.csv. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L7d](l7d-decode-kernels.md) · [L7c](l7c-decode.md) · [L6a](l6a-residency.md) · phase 2

# L8a — the dense bank stops being halves: 11.90 → 14.07 tok/s, bit for bit

`results/l8a_decode.csv` · `results/l8a_decode_fp16.csv` · `results/l8a_graph.csv` · `results/l8a_head.csv`

**Result: decode is 14.07 tok/s against 11.90 — 1.18x — and the arithmetic is
unchanged.** Not "within a tolerance": the halves the matrix cores multiply
are the same halves, tile for tile, and the model's completion is the same
token list. Prefill comes along at **1044.8 tok/s against 990.6** (2.67x
llama.cpp's 391.42), residency falls from 84.20 GB to **82.40**, and three of
the four dense families now read the width the checkpoint ships them at. Two
runs agree to **0.36%** (14.07 / 14.02) and the fp16 control reproduces L7d
exactly (11.90 against 11.89).

L7c-5 is the whole motivation. A decode step reads 8.07 GB of dense weight
where the checkpoint holds 3.67, because every GEMM in this vertical but the
MoE's took a B operand the host had dequantised to halves. The dense half is
83% of a token's bytes, so **the first 1.66x of phase 2 needs no
re-quantisation at all** — only not expanding what is already there.

| | L7d | L8a | |
|---|---:|---:|---|
| decode | 11.89 tok/s | **14.07** | 1.18x |
| a token | 84.1 ms | **71.1** | |
| prefill, ubatch 2048 | 990.6 tok/s | **1044.8** | 2.67x llama.cpp |
| dense bytes a token | 8.07 GB | **5.01** | 1.61x |
| a token, all in | 9.67 GB | **6.61** | |
| this bank's own ceiling | 25.0 tok/s | **36.6** | |
| resident | 84.20 GB | **82.40** | |

## L8a-1: the round trip is an identity, so the gate is an equality

A dense weight is staged as int8 in the same §2.8 fragment tiling — tile
(nt, kt) is 256 contiguous *bytes* holding element (k, n) at (n%16)*16 + k%16
— with one fp16 scale per 32 elements of a row, in a k-major plane behind the
tiles (D8). 8.5 bits a weight against 16.

The scales are not new information. ggml's `quantize_row_q8_0` picks
`d = amax/127`, so **the largest level in a Q8_0 block is always 127**, and
re-deriving (d, q) from the dequantised floats returns exactly the pair the
checkpoint stored: `127*d` is exact in f32 (11 mantissa bits times 7),
dividing it by 127 is a correctly-rounded division whose exact result is
representable, and `round(q*d*(1/d))` is `q` for every |q| ≤ 127. Then
`float(q) * float(d)` is exact — an fp16 times an fp16 needs 22 bits — and
rounding *that* to fp16 is precisely what `tileB` wrote into the old bank.

So the kernel puts the same halves in LDS that the fp16 arm loaded from
global, and the tests say so as equalities rather than tolerances:
`TestBankQ8RoundTrip` recovers the levels and scales a synthetic Q8_0 tensor
was built from; `TestBankQ8IsTheHalves` compares the two banks element for
element through a row permutation; `TestHeadGPUQ8IsTheHalves` runs the head
twice on one device and demands identical logits. On the real
`output.weight`, `-head` reports **0 of 248320 logits differ**.

## L8a-2: three families are not Q8_0, and they cost a tail rather than a tolerance

The checkpoint's dense half is Q8_0 with three exceptions, and all three are
fused into the middle of somebody else's matrix:

| family | type | where | rows |
|---|---|---|---:|
| `ssm_alpha`, `ssm_beta` | F32 | DeltaNet qkv, from column 16384 | 96 of 16512 |
| `indexer.{q,k}_proj` | BF16 | attention qkv, from column 13312 | 640 of 13952 |
| `hc_inject` | F32 | hyper-connection down, from row 320 | 4 of 336 |
| `ffn_gate_inp` | F32 | the MoE's router | all |

For these, int8 is a real re-quantisation where it is an identity everywhere
else, and the first attempt — quantise the whole fused matrix and see —
**failed the layer's own oracle**: `ssm_alpha` went from 1.31e-04 rms against
the CPU reference to **2.99e-03** and `ssm_beta` from 1.28e-04 to 2.95e-03,
both past the 2.0e-03 the test has asserted since L3b. That is the arithmetic
one would predict: a 32-element int8 group carries about seven bits relative
to the group's own maximum, so a 2560-long dot product lands ~1% from the f32
one, against fp16's ~0.05%.

They are 0.6% and 4.6% of their matrices and both begin **on a column-block
boundary**, so the answer is a split rather than a compromise. The Q8 arm
reads two push fields MODE 2 does not otherwise use — `lowRank`, the first row
that is not int8, which is what it already means in MODE 0, and `gateOff`,
where those rows' halves are, or `NO_W` for a matrix that is int8 throughout —
and a whole 64-column workgroup falls on one side or the other, so the branch
is per workgroup and free. With the tail restored, every tensor of the
DeltaNet layer and of the attention layer is **identical to the fp16 bank's**,
to the last digit printed, including `ssm_alpha`, `ssm_beta`, `indexer_k_raw`
and `indexer_score`.

The routers stay fp16 entirely: they are the MoE's, their ties decide which
experts run (L5a), and 0.125 GB a token is 1.9% of the budget.

## L8a-3: a compare in the k-loop cost 1.27x on a matrix that has no tail

The first spelling of L8a-2 put the branch *inside* the reduction, choosing
per tile between a fragment out of LDS and one out of the fp16 bank. It is
correct and it is expensive: the DeltaNet's output projection — `ssm_out`,
Q8_0 all the way through, a matrix whose `tail` is false in every workgroup —
went from **125.9 µs to 160.0** at one token, which is the fp16 arm's own
163.3. A branch nothing takes cost the whole saving.

Two `coopMatLoad`s in one loop body are two fragment allocations the compiler
must keep live, and register pressure is what both projection ladders were
already bound by (L2f-5, L3b-4). Writing the loop twice — the tail's B out of
the fp16 bank in one, the LDS slab in the other, hoisted above the k-loop —
puts it back at 125.9 µs exactly. It is the third time in this vertical that
the cost of a kernel turned out to be its shape rather than its work.

## L8a-4: the unpack is 4% of the bus, and a tiny second dispatch is not free

`-head` stages `output.weight` on both banks in one process and profiles the
same dispatch against the same activation. At the shape the model actually
runs it — one row, 248320 columns, 2560 deep:

| rows | fp16 best | q8 best | | q8 GB/s |
|---:|---:|---:|---:|---:|
| 1 | 6454.4 µs | **3585.7** | 1.80x | 188.4 |
| 8 | 6449.5 | 3546.4 | 1.82x | 190.5 |
| 64 | 6522.7 | 3725.1 | 1.75x | 181.3 |
| 512 | 26916.5 | 18763.7 | 1.43x | 36.0 |

The bank is **1.87x** smaller and the dispatch is **1.80x** faster, so the
unpack — a byte extract, a convert, a multiply and an LDS store per weight,
plus a barrier per K-step — costs 4% of the bandwidth and nothing else. 188.4
GB/s against the fp16 arm's 197.0 is the whole price of the format. At 512
rows the weight is re-read M/BM times and the ratio falls to 1.43x, which is
the same curve every projection ladder here has.

The tail was a *dispatch* before it was a branch, and that is worth recording
as a second data point for D11. A [128, 2560] fp16 GEMM at one token is two
workgroups on a 40-CU device: **70.2 µs for 0.8% of the rows**, against 325.1
for the other 99.2%. Folding it into the same grid took the fused projection
from 395.3 µs to 329.6 — 1.20x — for 128 rows of arithmetic that were never
the point.

## L8a-5: the three blocks, measured

Per layer, GPU time inside the command buffer, at one token and at 512:

| | fp16 | q8 | | | fp16 @512 | q8 @512 | |
|---|---:|---:|---:|---|---:|---:|---:|
| DeltaNet qkv | 527.4 µs | **329.6** | 1.60x | | 1967.9 | 1436.5 | 1.37x |
| DeltaNet out | 163.3 | **125.9** | 1.30x | | 446.8 | 446.0 | 1.00x |
| DeltaNet layer | 705.0 | **469.7** | 1.50x | | 3103.2 | 2560.6 | 1.21x |
| attention qkv | 427.2 | **266.2** | 1.60x | | 1530.2 | 1129.9 | 1.35x |
| attention out | 156.5 | **125.1** | 1.25x | | 461.0 | 454.4 | 1.01x |
| attention layer | 611.3 | **417.6** | 1.46x | | 2339.6 | 1927.4 | 1.21x |
| lm head | 6454.4 | **3585.7** | 1.80x | | 26916.5 | 18763.7 | 1.43x |

And end to end, milliseconds a decode token, from `results/l8a_decode.csv`
beside `results/l8a_decode_fp16.csv` — the same binary, one environment
variable apart:

| block | fp16 | q8 | GB a token | GB/s | off the bus |
|---|---:|---:|---:|---:|---:|
| MoE | 28.4 | 28.5 | 1.60 | 56 | 4.31x |
| gated DeltaNet | 27.1 | **18.8** | 2.25 | 120 | 2.02x |
| hyper-connection | 8.9 | 9.5 | 1.31 | 138 | 1.75x |
| attention | 7.8 | **5.6** | 0.70 | 125 | 1.94x |
| lm head | 6.3 | **3.5** | 0.68 | 194 | 1.25x |
| PLE, moves, gather, glue | 4.6 | 4.3 | 0.07 | | |
| **a token** | **84.0** | **71.1** | **6.61** | 93 | |

## L8a-6: what is left

The gate is L7c's, unchanged and re-run on both banks: on `The capital of
France is` at temperature zero the completion is the same list of capitals,
the same `<think>` block and the same `Lisbon.`, token for token — which it
has to be, because for every tensor but the four families of L8a-2 the two
banks hold the same numbers, and those four are still halves.

Three things the table above says.

**The hyper-connection block is the last dense family on halves**, 1.31 GB a
token and 13.3% of the step. It is not here because it is the only one that
does not use the plain arm: its down projection is MODE 0 with `inject` fused
into its last tile — four F32 rows that do *not* start on a column block —
its up projection is MODE 1 with the gate collapsed in the epilogue, and at
one token it runs neither, but `llm_hc_gemv.comp`'s split-K GEMV. Three more
arms, and 1.31 GB becomes ~0.70: that is L8b.

**The MoE is now 40.1% of the step and 4.3x off its own bytes**, which is
where it was before and is now the largest thing in the token by a factor of
1.5. Its bank was never expanded — L5b staged it verbatim — so L8 does not
touch it, and what is wrong with it is L5b-7's unpack and the open question
about a GEMV per expert.

**The ceiling moved.** A token now reads 6.61 GB against 9.67, so this bank's
own limit is **36.6 tok/s** rather than 25.0 — within 4% of the 38.2 the
checkpoint's own width allows, because 8.5 bits a weight *is* the checkpoint's
width. We are at 38% of it. The bytes are no longer the thing to fix; what is
left is the same 2-4x off the bus that L7d left in the two big blocks.
