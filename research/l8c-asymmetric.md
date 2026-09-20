<!-- LLM.md L8c-3. D7, reopened and reversed: the asymmetric form is the one
     calibration works with, and it is the one to build.
     Cited from llm/sim.go, llm/sim_test.go and reference/quant_ref.c. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · accuracy

# L8c-3 — the asymmetric form, and the calibration that only works on it

**Result: D7 was decided on the wrong metric, and reversing it halves the cost
of 4-bit dense weights twice over.** At identical bits and identical bytes —
4.500 either way — ggml's asymmetric `q4_k` costs **+7.04%** of perplexity over
the whole corpus where the symmetric `q4_0` costs **+14.49%**. And the lever
L8c-2 found broken turns out to be a property of the *format*, not of the
model: on the asymmetric form the imatrix **helps**, taking 4.5 bits on every
streamed dense family to **+4.24%**, where on the symmetric form the same
matrix makes it worse (+14.49% → +15.63%). The 2x2 is the stage:

| 145 chunks, from 4.0289 | `rtn` | `imatrix` |
|---|---:|---:|
| symmetric `q4_0/32` | 4.6127 (+14.49%) | 4.6588 (+15.63%) |
| **asymmetric `q4_k/32`** | 4.3124 (+7.04%) | **4.1998 (+4.24%)** |

**That is better than L8c-1's best mixed plan, at fewer bits.** The plan
L8c-1 recommended was 5.30 bits for +5.98% and a 52.9 tok/s ceiling; uniform
asymmetric 4.5-bit with calibration is **4.50 bits for +4.24% and a 56.7 tok/s
ceiling**, and a mixed asymmetric plan at 4.75 bits is **+2.70% at 54.8**. **So
the dense re-quantisation is worth building after all**, which is the opposite
of what L8c-1 and L8c-2 concluded, and the reason is one that neither could
see: every rung either of them measured was in a format whose group has a
single free parameter.

Files: `llm/sim.go`, `llm/sim_test.go`, `llm/quantref_test.go`,
`llm/testdata/quant_{in,ref}.bin`. No shader and no kernel changed — this is
still the simulation, which stages a candidate format's own halves through the
kernels that exist. Results: `results/l8c_asym.csv`,
`results/l8c_ppl_asym.csv`, `results/l8c_ppl_asym_mixed.csv`.

---

## What was actually open

L8c-2 closed the calibration question and left one caveat, in its own words:

> This is Q4_0's symmetric form, not Q4_K's. The checkpoint's own experts are
> Q4_K — an asymmetric super-block with its own min and scale — and that is
> the format unsloth's imatrix was collected for and used against. Nothing
> here says calibration fails for K-quants; it says it fails for the
> *symmetric* 4-bit format D7 chose, on the families whose importance is
> concentrated.

D7 chose symmetric on L0d's evidence: asymmetric was **1.043-1.053x** better
at equal bits, against a §7 prediction of 2x, and a min per group is a second
number to store. The number was real; what it was a number *about* was weight
reconstruction under int8 activations, and L8c-1 had already found that
reconstruction does not carry to perplexity (it is what retired D3).

## The format, and the one place it is not ggml's

A symmetric group has one number. An asymmetric one has two — a scale and a
min, with unsigned levels in `[0, 2^b - 1]` — and ggml does not spend an fp16
on each of them per group of 32. It **nests**: a super-block of eight groups
carries one fp16 `d` and one fp16 `dmin`, and each group carries a 6-bit scale
and a 6-bit min quantised against that pair. Twelve bits a group plus 32 a
super-block is **0.5 bits a weight at eight groups of 32 — exactly what a
symmetric fp16 scale per 32 costs.**

So `q4_k/32` and `q4_0/32` are both 4.500 bits and `q5_k/32` and `q5sym/32`
both 5.500, and **every comparison below is free of a width argument.**

The one departure: ggml's K-quants require `k % 256 == 0`, and
`llama-quantize` falls back to another type where a row is not a whole number
of super-blocks. Every dense row in this model is — **except the one family
the question is about.** `hc_{attn,ffn}_up` and `output_hc_up` read the
low-rank space, so their k is 320, and `hc_attn_up` is L8c-2's outlier, the
tensor whose importance is concentrated in 4.4 effective columns of 32 and
which carried the whole imatrix regression. Falling back to `q4_0` there would
have answered a different question. So the super-block is ggml's eight where
eight fits and **the whole row** where it does not: ten groups rather than
eight, which is **4.475 bits a weight rather than 4.500**, stated rather than
rounded off. Nothing in `make_qkx3_quants` or `make_qp_quants` knows the
count.

## Finding 1: the port is bit-identical to ggml in f32, not just in the half

Same discipline as L8c-2, same oracle: `reference/quant_ref.c` runs
`ggml_quantize_chunk` — `llama-quantize`'s own entry point — and dequantises
straight back. The header already carried the ggml type, so the only change
was emitting Q4_K and Q5_K records beside the Q4_0 ones.

Four new arms — two formats, calibrated and not — over two real shapes, and
all four are **identical to ggml's output over 204 800 values apiece, and
identical in f32 and not merely in the half** (`TestQuantSimMatchesGGML`).
The stronger equality is available here because the asymmetric arm reproduces
ggml's dequant expression term for term — `d*sc*l - dmin*m`, which is also
what `llm_moe_gemv.comp`'s Q4_K unpack already computes, in f32, for this
checkpoint's own experts.

It matched first try, which is worth recording only because L8c-2's did not:
the three details that cost that stage a day — `nearest_int` rounding half to
**even**, and two `float` accumulators — are shared, and had already been
paid for.

Two functions are ports rather than transcriptions:

- **`makeQkxQuants` is `make_qkx2_quants` and `make_qkx3_quants` at once**,
  because they are the same function. They differ in two places and neither is
  reachable: qkx3 accepts a null weight vector where qkx2 requires one (every
  call site here passes one), and qkx3's degenerate test is `max <= min` where
  qkx2's is `max == min`, after a line that has already forced
  `min <= 0 <= max`.
- **`makeQpQuants` is `make_qp_quants`**, the non-negative quantiser the
  calibrated path uses for the super-block's own eight scales and eight mins —
  nine rungs, then five sweeps of coordinate descent. It is the part of the
  format that has no symmetric counterpart at all.

Three arms as before, because ggml still searches and calibrates in one
function: `rtn` is `quantize_row_q4_K_ref`, `search` is
`quantize_row_q4_K_impl` with a null imatrix, `imatrix` is the same with
unsloth's columns.

## Finding 2: at identical bits the asymmetric form is 2.1x cheaper

Over the corpus it is **+7.04% against +14.49%, 2.06x**. Per family the screen
is eight chunks, baseline 2.0189, and the symmetric column is L8c-1's and
L8c-2's own numbers — **4.500 bits both**:

| scope | symmetric | asymmetric | |
|---|---:|---:|---|
| every streamed dense family, rtn | 2.3927 (+18.51%) | **2.1968 (+8.81%)** | 2.1x |
| hyper_conn, rtn | 2.1396 (+5.98%) | **2.0563 (+1.85%)** | 3.2x |
| full_attn, rtn | 2.0990 (+3.97%) | **2.0601 (+2.04%)** | 1.9x |
| deltanet, rtn | 2.0489 (+1.49%) | **2.0422 (+1.15%)** | 1.3x |
| lm_head, rtn | **2.0245 (+0.28%)** | 2.0269 (+0.40%) | 0.7x |
| hyper_conn at 5.5 bits, rtn | 2.0935 (+3.70%) | **2.0169 (−0.10%)** | — |

**The gap is widest exactly where L8c-1 found the model most sensitive**, and
that is the shape of the result rather than a detail: the hyper-connection
block is the family whose cost ran *inverse* to its bytes, and it is the one
the asymmetric form helps most. The lm head is the one family that prefers
symmetric, by 0.12 points of 0.4.

On reconstruction alone (`TestQuantSimAsym`, 256 rows) the gap is
**1.22-1.30x** on relative rms, not L0d's 1.043-1.053x — so D7's own metric
was also understating it, on a hierarchical two-level scale L0d never
measured. But reconstruction is still the wrong instrument, and it under-reads
the end-to-end gap by another factor of two.

## Finding 3: the imatrix works on this form — and that is the real result

Three arms, every streamed dense family at 4.5 bits. Eight chunks, so that the
middle arm sits beside L8c-2's own — the corpus numbers for the outer two are
in the 2x2 above and agree in sign, order and rough magnitude:

| arm | symmetric `q4_0/32` | asymmetric `q4_k/32` |
|---|---:|---:|
| `rtn` | 2.3927 (+18.51%) | 2.1968 (+8.81%) |
| `search` | 2.4802 (+22.85%) | 2.2366 (+10.78%) |
| **`imatrix`** | 2.4702 (+22.35%) | **2.0981 (+3.92%)** |

**The scale search alone still loses on both forms** — which is why the middle
arm exists, and it means the win is calibration and not ggml's extra
machinery. What changes is what happens when the importance matrix goes in:
symmetric, it recovers 0.5 points of the search's 4.3-point regression and
ends 3.8 points worse than round-to-nearest; asymmetric, it goes **4.9 points
past round-to-nearest**.

Per family, and calibration now helps everything it covers:

| family | `rtn` | `imatrix` | symmetric `imatrix` |
|---|---:|---:|---:|
| deltanet | +1.15% | **−0.30%** | +1.48% |
| hyper_conn | +1.85% | **+0.90%** | **+9.04%** |
| full_attn | +2.04% | **+1.05%** | — |
| lm_head | +0.40% | +0.40% | — |

`lm_head` does not move because the published matrix has no entry for it — the
run prints that rather than absorbing it — and `hyper_conn`, L8c-2's
catastrophe at +9.04%, is **+0.90%**.

## Finding 4: the mechanism is the second parameter, measured

L8c-2's explanation was a gain error: `make_qx_quants` minimises a weighted
squared error with the levels clamped, that optimum trades a little bias for a
lot of variance, and **bias compounds on L6b-3's x1.085 a layer where residual
noise averages out**. Where importance is concentrated, the weighted fit is
effectively over a handful of columns and the bias is large.

That explanation makes a prediction about the asymmetric form, and it is the
reason this stage is worth running rather than guessing: **a symmetric group
has one free parameter and it *is* the gain**, so a calibrated fit has nowhere
to express its preference except by moving the scale. An asymmetric group has
two, and the min can absorb an offset without dragging the scale. If the
mechanism is right, calibration should cost less systematic shrinkage on the
asymmetric form. `TestQuantSimGain`, 256 rows of layer 0:

| tensor | form | `rtn` gain | `imatrix` gain | **what calibration costs** |
|---|---|---:|---:|---:|
| **`hc_attn_up`** | symmetric | −0.110% | **−1.346%** | **−1.24 pp** |
| **`hc_attn_up`** | asymmetric | −0.352% | −0.822% | **−0.47 pp** |
| `hc_attn_down` | symmetric | −0.124% | −0.312% | −0.19 pp |
| `hc_attn_down` | asymmetric | −0.291% | −0.321% | −0.03 pp |
| `attn_qkv` | symmetric | −0.084% | −0.237% | −0.15 pp |
| `attn_qkv` | asymmetric | −0.287% | −0.284% | **+0.00 pp** |

**The extra shrinkage calibration costs on `hc_attn_up` is 2.6x smaller on the
asymmetric form, and on the other two it is gone.** That is the mechanism, and
it is enough to flip the sign: calibration buys a better fit on both forms and
pays for it in bias on one of them.

Note what it is *not*. The asymmetric `imatrix` arm still has a **worse**
unweighted reconstruction than its own `rtn` (8.90e-02 against 7.55e-02 on
`hc_attn_up`) — as the symmetric one did — and it wins the model anyway. The
quantity that predicts the model's behaviour is the systematic part, not the
total, on both forms. L8c-2's rule survives with a qualifier:

> An imatrix helps in proportion to how evenly importance is spread inside a
> scale group — **and how many free parameters that group has to express the
> preference with.** Where importance is concentrated and the group has one
> parameter, what goes up is the systematic error. Where it has two, it does
> not.

## Finding 5: the plan, and what it is worth

`cmd/gguf -width 4.5` is the bytes half. Dense at 4.50 bits is **4.264
GB/token against 6.334** — 1.49x — for a **56.7 tok/s ceiling** against 38.2.
A family moved to 5.5 bits costs its own row times 5.5/4.5: `hyper_conn` is
+0.080 GB and `full_attn` +0.075.

Residency moves much less than the token does, and it is worth saying why:
the dense half is 5.5 GB of an 82.52 GB resident core, so re-pricing all of it
to 4.5 bits leaves **80.5 GB** — the experts are 77.02 of it and this stage
does not touch them. **A decode token is the number that moved**, because a
token reads every dense weight and ten experts of 512.

| plan | bits | GB/token | ceiling | PPL (8 chunks) | delta |
|---|---:|---:|---:|---:|---:|
| as shipped (L8c-0) | 8.50 | 6.334 | 38.2 | 2.0189 | — |
| **L8c-1's symmetric mixed plan** | 5.30 | 4.575 | **52.9** | — | **+5.98%** |
| uniform `q4_k/32`, imatrix | **4.50** | **4.264** | **56.7** | 2.0981 | **+3.92%** |
| hc at `q5_k`, rest `q4_k` | 4.63 | 4.344 | 55.7 | 2.0740 | +2.73% |
| hc + attn at `q5_k`, rest `q4_k` | 4.75 | 4.419 | 54.8 | 2.0561 | **+1.84%** |

**Every asymmetric plan dominates L8c-1's recommendation on both axes at
once** — fewer bytes *and* less perplexity — which is not a trade-off and is
why this stage changes the direction rather than refining it.

L8c-1's compounding rule survives and is much milder: the four families at
uniform 4.5 bits sum to +2.05% where the plan measures +3.92%, against a
symmetric 11.7% summing to a measured 18.5%.

### Over the whole corpus

Eight chunks screens for broken, not for a percent (L8c-1's gate). Six rows
got the full 145, against our own 4.0289 — the two plans, and the 2x2 that is
the stage's central claim:

| plan | bits | GB/token | ceiling | PPL (145 chunks) | delta |
|---|---:|---:|---:|---:|---:|
| as shipped (L8c-0) | 8.50 | 6.334 | 38.2 | **4.0289** ± 0.02279 | — |
| L8c-1's symmetric mixed plan | 5.30 | 4.575 | 52.9 | 4.2699 ± 0.0243 | +5.98% |
| uniform `q4_k/32`, imatrix | **4.50** | **4.264** | **56.7** | **4.1998** ± 0.02399 | **+4.24%** |
| hc + attn at `q5_k`, rest `q4_k`, imatrix | 4.75 | 4.419 | 54.8 | **4.1377** ± 0.02356 | **+2.70%** |
| uniform `q4_k/32`, **rtn** (the calibration control) | 4.50 | 4.264 | 56.7 | 4.3124 ± 0.02482 | +7.04% |
| uniform `q4_0/32`, rtn — **the symmetric form at the same bits** | 4.50 | 4.264 | 56.7 | 4.6127 ± 0.02697 | +14.49% |
| uniform `q4_0/32`, imatrix | 4.50 | 4.264 | 56.7 | 4.6588 ± 0.02760 | +15.63% |

**Both asymmetric plans beat L8c-1's recommendation on both axes at once.**

**The eight-chunk screen ranked all five rows in the corpus's own order** —
mixed < asymmetric+imatrix < asymmetric+rtn < symmetric+rtn < symmetric+imatrix
at both lengths — which is more than L8c-1's gate note promises of it, and the
reason to trust the per-family tables above. It is still not calibrated for a
percent, and it errs in **both directions**: it over-read the `rtn` arms
(+8.81% and +18.51% against +7.04% and +14.49%) and under-read the two plans
(+3.92% and +1.84% against +4.24% and +2.70%). Nothing here should be quoted
to a tenth of a point from the screen alone.

**Calibration is worth 2.80 points of the 7.04 at corpus scale** — 4.3124 to
4.1998 — where on the symmetric form it added 1.14 points instead. The
bottom two rows are the same comparison the eight-chunk table makes, at the
length the gate is written for, and they are the stage's central claim: at
**identical bits and identical bytes** the two forms are **2.06x** apart
uncalibrated and **3.69x** apart with the matrix.

Wall clock is 8m19s-8m26s a plan, against L8c-0's 7m58s for the unsimulated
bank: the simulation's cost is in staging, not in the corpus.

## What this settles

**D7 is reversed.** A 4-bit rung here is `q4_k`, not `q4_0`, and the reason is
not the 5% reconstruction gap it was argued about in either direction — it is
that the asymmetric group is the one an importance matrix can be fitted into
without turning into a gain error.

**D3 is back, at its own width.** ~4.25-4.5 bits on everything streamed was
retired at L8c-1 as +18.5% of perplexity. In the format the checkpoint's own
experts already use, calibrated with the matrix its authors already published,
it is **4.1998 against 4.0289, +4.24%** — and at 4.264 GB a token that is
1.49x the bytes and a **56.7 tok/s ceiling**, for 80.5 GB resident. Spending a quarter of a bit on
the two sensitive families takes it to **+2.70% at 54.8 tok/s**, which is the
row to build against.

**So the dense kernel is the next thing to build**, which reverses L8c-1's and
L8c-2's recommendation. The W4A8 layout §1.1 describes is for a *symmetric*
weight; an asymmetric one needs the min folded into the epilogue as a
per-group correction times the activation's column sum, which is one extra
reduction over A and not a change to the matrix core. That is L8c-4's shape.

**Two things are still out of the simulation's reach and are worth more
together than any of the rows above**: the F32 router at fp16 (+1.5 tok/s of
ceiling) and the 512 expert banks at ~4.25 (+3.0). The experts are the
*calibrated* part of this checkpoint and they are already Q4_K — which, after
this stage, is the reason to expect re-quantising them to behave, rather than
the reason to fear it.

**Caveats.**

- The gain table is 256 rows of layer 0 of three tensors. It is a mechanism
  check, not a census.
- Nothing at `-c 2048` reads the QSA indexer's projections (L8c-1's finding
  2), so `qsa_indexer` rides along in every plan untested. It is 0.011 GB a
  token.
- The 320-wide hyper-connection rows use a ten-group super-block, which is
  our format and not ggml's. It is bit-exactness-checked only where ggml will
  run, which is the eight-group case; `TestQuantSimAsym` covers the rest
  structurally.
- Every rung here has an **fp16 A operand**. §1.1's W4A8 is an
  int8-activation format, and L0d priced int8-per-token at 1.13x the error of
  fp16 activations on reconstruction — which, as this stage and L8c-1 both
  show, is not a number that carries to perplexity. It has to be measured
  where it lands.
