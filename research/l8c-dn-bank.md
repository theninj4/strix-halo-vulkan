<!-- LLM.md L8c-5. The gated DeltaNet on the 4.5-bit bank: the second family,
     the largest one, and the first corpus-scale measurement of a family whose
     eight-chunk screen said it was free.
     Cited from llm/gpu_deltanet.go, llm/bank_q4.go, llm/graph.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · bank

# L8c-5 — the gated DeltaNet at 4.5 bits, and a screen that had the sign wrong

**Result: 36 of the 48 layers are on L8c-4's bank, decode is 28.9 tok/s
against 25.75, prefill is faster at both ubatches, and the family costs
+0.93% of perplexity where its own eight-chunk screen said −0.30%.**

The gated DeltaNet is the largest dense thing in the model — 2.087 B
parameters, 2.247 GB of a 6.334 GB decode token on the checkpoint's own
widths, 46% of the dense half — so it is the family whose bytes justified
going second. It needed no new kernel and no new format: `llm_gemm.comp
-DQ4B` and `llm_gemv.comp -DQ4B` already had the arms, including the fp16
tail arm that nothing had exercised, and this block is what exercises it.

| a decode token | L8e (q8) | L8c-4 (+ head) | **L8c-5 (+ deltanet)** |
|---|---:|---:|---:|
| decode | 24.66 tok/s | 25.75 | **28.73 / 29.07** |
| against llama.cpp's 25.15 | 0.981x | 1.024x | **1.15x** |
| a token | 40.5 ms | 38.8 ms | **34.4 ms** |
| the DeltaNet block, per token | 11.0 ms | 10.9 ms | **6.8 ms** |
| residency | 81.89 GB | 81.57 GB | **80.53 GB** |
| prefill, ubatch 2048 | 1052.8 tok/s | — | **1070.1** |
| prefill, ubatch 512 | 655.2 tok/s | — | **667.0** |
| perplexity, 145 chunks | 4.0289 | 4.0621 (+0.82%) | **4.1012 (+1.79%)** |

Files: `llm/gpu_deltanet.go`, `llm/bank_q4.go`, `llm/graph.go`,
`llm/deltanet.go`, `llm/model.go`, `cmd/llm/bench_deltanet.go`. Results:
`results/l8c_decode_dn.csv`, `results/l8c_graph_dn.csv`,
`results/l8c_ppl_dn_q4k.csv`, `results/l8c_ppl_dn_only.csv`,
`results/l8c_dn_gemv.csv`.

---

## 1. The screen was not merely optimistic; it had the wrong sign

L8c-4 found that a per-family width screened over eight chunks under-reads
its corpus cost by 2.05x, on the one family that had been measured both ways.
This family is the second, and it is worse than a factor:

| `deltanet` at `q4_k/32`, imatrix | baseline | measured | delta |
|---|---:|---:|---:|
| the simulation, 8 chunks (L8c-3) | 2.0189 | 2.0128 | **−0.30%** |
| **the bank**, 145 chunks | 4.0289 | **4.0665** | **+0.93%** |

At eight chunks the calibrated 4.5-bit DeltaNet is *better than the
checkpoint's own Q8_0* — which was one of L8c-3's headline results, quoted
in LLM.md as "it now helps every family it covers (`deltanet` +1.48% to
−0.30%)". Over the whole corpus it costs +0.93%. The screen is not a noisy
estimate of the corpus number; on the first 8184 tokens of `wiki.test.raw`
the family's cost is *negative*, and over 148 335 tokens it is positive and
five times either side's standard error.

So L8c-4's warning generalises and hardens. **A per-family row in
`results/l8c_asym.csv` does not bound its family's corpus cost, and does not
even fix its sign.** The uniform plan's +4.24% is a corpus number and still
stands — the rows it was assembled from do not, and the two families now
measured both ways came out +0.42 pp and +1.23 pp above their screens.

What does not go wrong is the other half of L8c-1's rule. At corpus scale
these two families are very nearly **additive**:

| 145 chunks, against 4.0289 | ppl | delta |
|---|---:|---:|
| `lm_head` alone (L8c-4) | 4.0621 | +0.82% |
| `deltanet` alone | **4.0665** | **+0.93%** |
| the two together | **4.1012** | **+1.79%** |
| their two deltas summed | — | +1.75% |

1.79 against 1.75 is compounding of 0.04 pp over two families. L8c-1
measured four symmetric families summing to 11.7% where the plan measured
18.5% — a 1.58x gap — and L8c-3 already found the asymmetric form much
milder. This is the first *built* confirmation of that: the errors the
asymmetric form injects at 36 and at 1 depth do not multiply through
L6b-3's x1.085 a layer the way the symmetric form's did.

## 2. The bank is the format, said twice

The stage's one load-bearing property is that the thing we store is the thing
L8c-3 measured. It is checked at two scales, and both are equalities.

**On the device, value for value.** `TestDeltaNetGPUQ4IsTheSim` stages layer
0 twice from the same checkpoint weights — once on the fp16 bank with the
format applied through `sim.go`, once on the real 4.5-bit bank — and runs the
whole layer. The fused projection's 115 360 values and the layer's 17 920
output values are **identical**, not close: the halves the matrix cores
multiply are the same halves, and the fragment order is the same because both
arms are `llm_gemm.comp` MODE 2 at `BK_TILES = 2`. The comparison is taken at
the layer's *output* as well as at the projection, so the convolution, the
recurrence, the gated norm and the second projection have all run on top of
it — a wrong address in the record plane cannot hide behind a tolerance.

**Over the corpus, chunk for chunk.** The bank and its simulation were run at
eight chunks:

    bank   LLM_DENSE_BANK=lm_head=q4_k/32,deltanet=q4_k/32
    sim    LLM_DENSE_SIM=lm_head=q4_k/32,deltanet=q4_k/32
           LLM_DENSE_SIM_SRC=q8 LLM_DENSE_SIM_QUANT=imatrix

and every per-chunk figure printed is the same to four decimals — 1.9139,
2.1107, 2.0253, 1.9160 and a final **2.0189** on both sides. `SRC=q8` is what
makes them the same question: the simulation's default scope includes the two
F32 matrices the bank leaves in its fp16 tail, so without it the two arms
would be quantising different sets of weights (§3).

## 3. The tail is where it was, and the plane covers it anyway

`ssm_alpha` and `ssm_beta` are the checkpoint's only F32 matrices in this
block — 48 rows of 2560 each, the decay and the delta-rule gate — and L8a-2
measured what int8 does to them: 1.3e-04 rms to 3.0e-03, past this layer's
own tolerance. A nibble is not a reason to revisit that, so the split is
exactly L8a's: the quantised plane holds rows `[0, 16384)` and the 96 rows
above it are a fp16 fragment-tiled matrix of its own at `pc.gateOff`. The
kernel's `Q8_SPLIT` in MODE 2 is `pc.lowRank` unrounded, because the host
chose the fused row order and `16384 % 64 == 0` — this is the easy case L8b's
hyper-connection block was not.

The quantised plane still covers **all** 16512 rows, tail included, because
the kernel derives the record plane's base from `pc.gemmN * pc.gemmK` and
`gemmN` is the stride of the output the store writes. That is 0.12 MB a layer
of nibbles and records nothing reads, against the 29 MB a layer the width
saves. It also means the matrix as staged is **4.590 bits a weight**, not
4.500: the tail's 96 rows are carried at 16 bits *and* at 4.5.

| a DeltaNet layer, staged | bytes | bits/weight |
|---|---:|---:|
| halves (pre-L8) | 116.00 MB | 16.000 |
| the checkpoint's Q8_0 (L8a) | 62.28 MB | 8.590 |
| **`q4_k/32` (L8c-5)** | **33.28 MB** | **4.590** |
| 36 layers | **1.20 GB** against 2.24 and 4.18 | |

`TestDeltaNetGPUQ4BankSize` states those three and checks the quantised half
alone is 4.500 bits plus the tail's records, rather than asserting a round
number a tail makes untrue.

## 4. Four of llama.cpp's matrices, three imatrix rows

The fused projection is `attn_qkv`, `attn_gate`, `ssm_alpha` and `ssm_beta`
written as column ranges of one [16512, 2560] weight (L3b). The published
importance matrix is keyed by **tensor name**, and the simulation L8c-3's
number came out of looked each source up separately — so the bank quantises
each source under its own name too, even though all four read the same block
input and their `in_sum2` rows are therefore nearly the same numbers.

That is not a micro-optimisation; it is what makes "the bank is the format the
simulation measured" true rather than nearly true, and it is why
`DeltaNetWeights` now carries its layer index: without it a staged layer
cannot name its own tensors. `bankImatrix` is the lookup, and it carries
ggml's own fallback — an uncovered tensor is **round-to-nearest**, not the
scale search with unit weights, because `quantize_row_q4_K_impl` returns
`quantize_row_q4_K_ref` on a null `quant_weights` and that third arm measured
worse than either (L8c-3). Every tensor of this family is covered, so the
fallback did not fire here; it exists so that the first family where it does
behaves like the simulation rather than like a guess.

`tileBQ4K` grew one property to allow this: **the destination's row count is
the plane's, not the source's**. The head stages one matrix into a plane of
its own size; this block writes four sources into one wider plane through
`row`, and the buffer length is what says how wide it is.

## 5. What the width buys, at both ends

**At decode the block is 1.60x.** 10.9 ms a token to **6.8**, which is 28.1%
of a token to 19.8%, and a token 38.8 ms to 34.4. On the bytes: 2.24 GB of
staged bank to 1.20, so the block moves **176 GB/s** where the Q8 bank moved
205. About 0.5 ms a token of that is the scan, the convolution, the gated
norm and the history write, which read no bank at all; netting them out puts
the two projections at roughly 190 GB/s against 215. That gap is the unpack —
a nibble costs two shifts and an affine term where a byte costs one multiply —
and it is the same shape as the head's, which holds 210 GB/s on a matrix with
no fusion and no tail.

**At prefill it is 1.09-1.11x *faster*, where D14 as written says it should
be slower — and the reason is that this weight crosses the MALL.**

| the block, per graph | 512 tokens | 2048 tokens |
|---|---:|---:|
| q8 (L8e) | 96.5 ms | 350.7 ms |
| **q4_k** | **86.7 ms** | **320.5 ms** |

and the whole graph comes along: **1070.1 tok/s at ubatch 2048 against
1052.8, and 667.0 at 512 against 655.2** — 2.73x and 1.70x llama.cpp's own
391.42, and 2.13x its `pp512` of 313.62.

D14 says a narrower bank is a decode decision that costs at prefill, because
at ubatch 2048 a workgroup reads all of B M/BM times and those re-reads come
out of a 32 MiB MALL, so the DRAM bytes were already hidden and all the
narrower bank adds is the unpack's ALU. L8c-4 added the clause that this is a
statement about a weight that **fits** the MALL — the head's 357 MB never
does at any width, so it is 1.08x faster at 512 rows too. This block is the
third case and the one that names the boundary, because the fused projection
**changes side**:

| the fused [16512, 2560] projection | staged | 32 MiB MALL |
|---|---:|---|
| halves | 84.5 MB | no |
| the checkpoint's Q8_0 | 44.9 MB | no |
| **`q4_k/32`** | **23.8 MB** | **yes** |

So going from int8 to nibbles does not only halve the bytes; it takes this
matrix's sixteen re-reads at ubatch 2048 off DRAM entirely. The output
projection fits at both widths (16.7 MB and 8.9), and it is the quarter of the
block where D14's hazard could still fire; the sum comes out on the fused
projection's side. Three blocks, three answers, one rule: **the hyper-connection
block's weights fit at every width and it was 1.09-1.15x slower; the head's
fit at none and it was faster; the DeltaNet's fused projection changes side
and it is faster.**

## 6. The decode rungs did not move, and the ladder that says so is an L3 hit

D12 says a split-K ladder does not carry across a bank: L8b-3 re-ran it going
from halves to int8 and the winner moved one rung along, because a slab is
half the bytes and §5.1b's 4 KB rotation lands somewhere else. Going from int8
to nibbles halves it again, so the ladder was re-run —
`LLM_DENSE_BANK=deltanet=q4_k/32 go run ./cmd/llm -dn -tokens 1 -gemm-ladder`,
`results/l8c_dn_gemv.csv`:

| rung | fused projection, us/layer | output projection, us/layer |
|---|---:|---:|
| k1 | 52.5 | 38.7 |
| k2 | 55.5 | 25.3 |
| k4 | 48.2 | 22.1 |
| **k8** | **47.1** | 21.1 |
| k16 | — | 18.3 |
| k20 | 50.0 | — |
| **k32** | — | **17.8** |
| k40 | 50.7 | — |
| the GEMM control | 180.3 | 119.9 |

k8 and k32 are the rungs L8d chose on the int8 bank, so nothing moved and
nothing needed to. **But the ladder is not evidence, and D16 is why**: every
row above reads 1700-1800 GB/s, seven times a 242 GB/s bus, because the bench
stages a handful of layers and repeats one dispatch over a bank that fits the
MALL. What it measures is kernel shape against an L3 hit — which is also why
the spread is 1.18x here where L7d-2 saw 2.4x in DRAM, and why the rung whose
slab stride *is* a whole multiple of 4 KB (k1, 20 480 bytes) is only the
second-worst rather than the worst. A rule about DRAM channels has nothing to
say about a weight that never reaches DRAM. The honest statement is that the
incumbent rungs are at the top of a ladder that cannot rank them properly, and
that a whole-model A/B is the only thing that could — which was not run,
because there is no candidate the ladder disagrees about.

## 7. What this does not settle

The text parted from L7c's completion at L8c-4, at a 0.075-wide tie in the
head's logits, and it has not come back: this bank writes the same Paris
syllogism the head-only bank does. That is what D17 and L8c's gate were
written for — the grade is §1's perplexity.

Three families are left, and the order their bytes justify now reads
`hyper_conn` (0.695 GB a token), `full_attn` (0.635) and `ple_proj` (0.035),
with `qsa_indexer` riding along. The first of them is the one that needs a
**packing of its own** — `hc_{attn,ffn}_up` and `output_hc_up` are 320 wide
and `get_scale_min_k4`'s twelve-byte scheme *is* eight groups, four low and
four high, not a length — and it is the family the whole imatrix result turns
on, so it should be re-screened at 145 chunks before its width is called. On
the two data points that exist, the screen it would be called from under-reads
by between 0.42 and 1.23 percentage points.

    cmd/gguf -width basis, the checkpoint's own widths
    a token   6.334 GB  as shipped
              6.016     lm_head at 4.5          (L8c-4)
              4.943     + deltanet at 4.5       (L8c-5, a 49.0 tok/s ceiling)
              4.264     + the other three       (L8c-3's plan, 56.7 tok/s)

Measured, this stage is 28.9 tok/s of that 49.0 — 59%, against L8c-4's 61% of
its own — so the bank is not where the remaining decode time is. On our own
staged widths a token is 4.69 GB and the ceiling 51.6.
