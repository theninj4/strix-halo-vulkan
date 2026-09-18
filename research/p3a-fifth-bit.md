# P3a — the fifth bit, and D19

> 2026-09-18. P3 closed the widths question with **D18** and one thing it
> could not buy: `bank_q4.go` is nibbles by construction, so every arm on the
> accuracy frontier better than the one it shipped needed a kernel that did
> not exist. This stage builds it — ggml's `qh` bit-plane, three kernels, one
> new `DenseBank` value — and then measures the four families in the order P3
> ranked them.
>
> **The plane is exact**: the bank equals the simulation **chunk for chunk
> over all 145 chunks**, nll to six decimals, on the first family. **D19
> takes three of the four**: `full_attn` (with `qsa_indexer`), `lm_head` and
> `hyper_conn` at `q5_k` — **4.0948 (+1.63%) against D18's 4.1850 (+3.87%)**,
> for **1.13 tok/s** of measured decode over two interleaved passes.
> `deltanet`'s fifth bit is refused: 0.41 pp for 1.44 tok/s.

## 1. The format, and the one decision in it

ggml's `Q5_K` is its `Q4_K` plus a `qh` bit-plane and **the same record** — the
fp16 (d, dmin) pair and twelve bytes of six-bit (scale, min) pairs — so
nothing L8c-4 built for the record moves, including L8c-6's ten-group
packing for the 320-wide hyper-connection family. What is new is one bit a
weight, and the only question was where to put it.

**The plane is one 32-byte tile per 128-byte nibble tile, in the same order,
and byte i of it is the top bit of each of the eight levels in word i of the
nibble tile** — bit e for the level at k offset e inside that word.

That is the whole design, and it is chosen so that **the plane byte index is
the nibble word index**. Every kernel already computes that index:

    llm_gemm.comp     `u`, a lane's share of the WN run's BK_TILES*32 words
    llm_gemv.comp     `col * 2 + u`, the two words a lane owns in a tile
    llm_hc_gemv.comp  the same

so the unpack gains a load and two bit tests and **no second addressing
scheme**. The plane sits *between* the tiles and the records — bOff, then
n*k/2, then n*k/8 — which keeps the record base derivable from `gemmN *
gemmK` and needs no push field; 64 uints is this device's whole push range
and the block has been full since L5b.

5 bits of level, 12 bits a group of 32 and 32 bits a super-block is
**5.500 bits a weight**, ggml's own number.

### What it cost in code

`-DQ5B` beside `-DQ4B`, four macros in `shaders/llm_q4k.glsl`, and 25 new
`.spv` — every Q4B build of the three kernels has a Q5 twin. The four-bit
arm is **unchanged**: rebuilt from the edited sources, `llm_gemm_q4_m4.spv`
disassembles identically modulo SSA numbering and `llm_gemv_q4_k1.spv`
differs only in where one `OpIMul` is hoisted.

On the host, `bank_q4.go:144`'s four-bit refusal became `qkBits` (4 or 5) and
the two-plane staging became `qkStage` — three planes, their order stated
once instead of at each of the six blocks that stage a dense matrix. That
mattered: with a plane that is sometimes absent, "the planes are contiguous,
in this order" is exactly the sort of fact that goes wrong in one place and
is found by a perplexity run rather than by a test. The lm head is the one
site that does **not** write them contiguously — it stages slabs of a matrix
that is 2.54 GB as floats, so each slab's three planes go to three separate
bases — and it says so.

`BankQ5K` is a fourth `DenseBank`. The alternative was to let the format's
`Bits` field decide inside a single `BankQ4K`, and it was rejected because
the bank value is what names a pipeline (`bankPipe`, `gemvBankPipe`) and
sizes a buffer: a bank that stages one width and builds another's SPIR-V
would be a wrong answer, not a slow one.

## 2. The gates

**Host, an equality.** `TestBankQKIsTheSim` now runs twelve rows: both
widths x both super-block lengths (256 and the 320-wide family's whole row)
x `rtn`, `imatrix` and `search`, through a row permutation, every value
compared to the simulation's with `!=`. What it really checks is that the
packing and the addressing are each other's inverse, which is the thing a
tolerance would hide.

**Device, five blocks, both widths.** Every existing bank-against-simulation
test is now a subtest pair:

| block | what is compared | q5_k result |
|---|---|---|
| attention | fused projection + layer output, GEMM | **identical**, 115 584 values |
| attention | every split-K GEMV rung against the GEMM | one rounding apart, rms ≤ 1.4e-3 |
| DeltaNet | fused projection + layer output | **identical**, 133 280 values |
| hyper-conn | 16 GEMM rung pairs, six outputs each | **identical**, 2 688 320 values |
| hyper-conn | six `down_gemv` rungs | 1.12e-4 rms against the GEMM |
| lm head | logits, GEMM; and the GEMV against it | **identical**; rms 1.96e-4 |
| PLE n-gram | key, value and the residual | **identical**, 161 280 values |

The hyper-connection block is the one that matters most here, because it is
the only one staging **two** record packings — so the fifth bit had to
survive a super-block that is the whole row.

**Corpus, chunk for chunk.** The real gate. P3 simulated the complete
uniform plan with `full_attn` at `q5_k` and got 4.1560 over 145 chunks. The
same plan on the *bank*:

    PPL = 4.1560 +/- 0.02361 over 148335 tokens, +3.16% against our 4.0289

and `results/p3a_ppl_attn_q5k.csv` against `results/p3_ppl_attn_q5k.csv` is
**0 of 145 chunks different** — equal in nll to all six recorded decimals.
The plane is the format the simulation measured, and P2's "the bank is the
simulation" survives the fifth bit.

## 3. The ladder, on D18's basis

Four plans, each a **complete plan measured over 145 chunks** rather than
composed — P2's additivity leak has not gone away. Bytes are `cmd/gguf
-width`'s arithmetic, which reproduces the group table exactly; decode is the
mean of two interleaved passes of `-gen -n 64` on one binary within one hour.

| plan | PPL (145 chunks) | delta | GB/token | ceiling | decode | pp recovered | **pp/GB** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **D18** — the control | 4.1850 ± 0.02389 | +3.87% | 4.281 | 56.5 | **35.70** | — | — |
| + `full_attn`, `qsa_indexer` | 4.1386 ± 0.02349 | **+2.72%** | 4.359 | 55.5 | 35.49 | 1.15 | **14.9** |
| + `lm_head` | 4.1136 ± 0.02327 | **+2.10%** | 4.438 | 54.5 | 35.11 | 0.62 | **7.8** |
| + `hyper_conn` = **D19** | 4.0948 ± 0.02322 | **+1.63%** | 4.518 | 53.6 | **34.57** | 0.47 | **5.9** |
| + `deltanet` | 4.0780 ± 0.02313 | +1.22% | 4.778 | 50.6 | 33.13 | 0.41 | **1.6** |

Per-arm decode, both passes, so the spread is visible rather than asserted:

    D18    35.63  35.77   mean 35.70   spread 0.14
    +attn  35.35  35.62        35.49          0.27
    +head  35.08  35.14        35.11          0.06
    +hc    34.43  34.71        34.57          0.28
    all4   33.08  33.18        33.13          0.10

Every step but the first is resolved several times over against the
within-arm spread, and the first is the one whose direction does not matter
to the decision. Both passes agree in sign at every step.

### Finding 1: P3's estimates held, except where they were inferred

| family | P3's pp/GB | measured | |
|---|---:|---:|---|
| `full_attn` | 14.1 | **14.9** | simulated |
| `lm_head` | 8.0 | **7.8** | simulated |
| `hyper_conn` | ~5.6 | **5.9** | simulated, from L8c-3's pair |
| `deltanet` | ~2.8 | **1.6** | *inferred, not simulated* |

Three families landed within 6% of a number produced before the kernel
existed, which is a good report on `sim.go` as an instrument. The fourth was
the one P3 marked "inferred" — and it is **1.75x worse** than the inference,
in the direction that matters, on the family that carries five times the
bytes of any other. The lesson is the one P3 already wrote down one level up:
*a plan is measured, never composed* — and an estimate that was never
simulated is a composition wearing a number.

### Finding 2: the fifth bit costs less decode than its bytes

At 178 GB/s — P1's measured rate for the weight-streaming dispatches — the
0.078 GB `full_attn` adds is 0.44 ms a step, which at D18's 28.01 ms is
−0.55 tok/s. Measured: **−0.21**. The whole ladder is the same way: the four
families add 0.497 GB, which the same arithmetic prices at −3.1 tok/s
against a measured −2.57.

This is not a correction to P1's attribution, which was per dispatch. It is
that the fifth bit is a **second stream** at a quarter of the first's width,
perfectly sequential, issued from the same loop — so it lands nearer the
227 GB/s a dispatch can reach than the 178 the nibble stream averages. It
is written down rather than chased: the item that would chase it is a
per-dispatch attribution of a q5 bank against a q4 one, and nothing on the
list currently turns on it.

### Finding 3: the line D18 already drew puts `deltanet` on the far side

D18 refused an arm at **4.6 pp/GB** (`full_attn` → int8) and took one at 22.
Applying that same line without adjusting it: `full_attn` (14.9), `lm_head`
(7.8) and `hyper_conn` (5.9) are all **above** the trade D18 refused;
`deltanet` at 1.6 is below it by three times. The cut needs no new principle
and no new judgement about what a point of perplexity is worth — which is
why it is the cut D19 takes.

In decode terms the same cut reads: the three families together cost
**1.13 tok/s (3.2%)** and take the quantisation damage from **+3.87% to
+1.63%**, less than half. `deltanet`'s fifth bit alone costs **1.44 tok/s
(4.2%)** for a further 0.41 pp.

## 4. D19

**D19 — ship `full_attn`, `qsa_indexer`, `lm_head` and `hyper_conn` at
`q5_k`; leave `deltanet` at `q4_k` and `ple_proj` at int8.**

It amends D18 rather than replacing it: `ple_proj`'s int8 row is untouched
and still the plan's best trade at 22 pp/GB, and `deltanet` stays where D18
put it. `llm.ShippedDenseBank` carries it and `cmd/serve -llm` stages it by
default; `cmd/llm` still stages nothing by default, so no CSV in `results/`
becomes ambiguous about what it measured.

    4.0948 +/- 0.02322 over 145 chunks — +1.63% against our own 4.0289
    4.518 GB a token, a 53.6 tok/s ceiling
    34.57 tok/s decode, 1.37x llama.cpp's 25.15

Against where P2 left the vertical — 4.2010, +4.27% — the accuracy cost of
the whole dense quantisation is now **38% of what it was**, for 6% of the
throughput.

## 5. What this hands the next stage

**The accuracy frontier is now flat.** Everything left on it is either
refused on measurement (`deltanet`'s fifth bit, 1.6 pp/GB) or needs a sixth
bit, and there is no reason to expect a sixth to rank differently from the
fifth. P3's carried-forward item — the **downstream task eval** — is now the
only unpriced thing about the widths, and it is more valuable than it was:
a plan at +1.63% of wikitext perplexity is close enough to the unquantised
model that perplexity may no longer separate the arms at all.

**Two things were priced and not taken.**

- **`ple_proj` at `q5_k`** rides free now that the plane exists — P3 put it
  at 47.8 pp/GB, the steepest slope on the board, for an absolute prize of
  0.196 pp. It is not in D19 because D18 put that family on **int8**, which
  is strictly better on both axes; the `q5_k` row was only ever the
  alternative to a *four*-bit `ple_proj`.
- **The GEMV rungs were not re-screened at the new width.** D12 says a
  split's stride must miss the 4 KB rotation and that the rung moves when the
  weight's width does — and a q5 slab is 1.22x a q4 slab plus a second
  stream. Every rung was *measured correct* (section 2) and the decode
  numbers above are on the rungs the q4 bank chose, so this is a possible
  tok/s left on the table, not a risk. `-attn -tokens 1 -gemm-ladder` on the
  q5 bank is the run, and the honest-floor rules from P1b apply: `-layers`
  high, `-iters 1`, and no row above 242 GB/s.

Files: `llm/bank_q4.go` (`qkBits`, `qkStage`, `qkHighPlane`, `qkDequant`),
`llm/bank.go` (`BankQ5K`, `qkBank`, `bankOfSim`), `shaders/llm_q4k.glsl`
(the `QK_*` macros), `shaders/llm_{gemm,gemv,hc_gemv}.comp`,
`llm/sim.go` (`ShippedDenseBank` = D19).
Results: `results/p3a_ppl_attn_q5k.csv` (the gate, against
`p3_ppl_attn_q5k.csv`), `p3a_ppl_d18_attn.csv`, `p3a_ppl_d18_attn_head.csv`,
`p3a_ppl_d18_attn_head_hc.csv` (D19), `p3a_ppl_d18_all4.csv`.
