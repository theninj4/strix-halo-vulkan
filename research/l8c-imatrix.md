<!-- LLM.md L8c-2. The calibration question, answered — and answered no.
     Cited from llm/imatrix.go, llm/sim.go and reference/quant_ref.c. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · accuracy

# L8c-2 — the imatrix, and why calibration makes this model worse

**Result: calibration does not rescue 4-bit dense weights here. It costs
another 3.8 points on top of round-to-nearest — 22.4% against 18.5% — and the
mechanism is not noise, it is a systematic gain error that is worst exactly
where the calibration data is most concentrated.** L8c-1 recommended trying
this before building any kernel, on the grounds that every rung it measured
was uncalibrated against a checkpoint whose experts are not. The lever has now
been pulled and it moves the wrong way, which settles phase 2's direction more
firmly than a win would have.

Files: `llm/imatrix.go` (new), `llm/sim.go`, `llm/quantref_test.go` (new),
`llm/sim_test.go`, `reference/quant_ref.c` (new),
`reference/fetch_llm_checkpoint.sh`, `llm/testdata/quant_{in,ref}.bin` (new).
No shader changed. Results: `results/l8c_imatrix.csv`.

---

## The matrix, and a fact about it that matters

`imatrix_unsloth.gguf`, 580 MB, is a GGUF — `general.type = "imatrix"`, 1852
F32 tensors, 45 chunks of unsloth's own calibration set — so `gguf/` opened it
with no changes. Two tensors per weight: `<name>.in_sum2`, the sum over
calibration tokens of the squared activation on each *input column*, and
`.counts`. Importance is `in_sum2 / counts` (llama.cpp's own normalisation),
and its length is the matrix's **k** — exactly the axis a scale group runs
along.

**It covers `blk.N.*` only.** There is no entry for `output.weight` or for the
head mixer's two projections, so **0.642 B weights — 16% of the streamed dense
half, the whole lm head among them — have no calibration data at all.** ggml's
behaviour there is to fall back to round-to-nearest (`quantize_row_q4_0_impl`
short-circuits when `quant_weights` is null), so that is what the plan does,
and every run prints it rather than absorbing it:

    simulated: 4.010 B weights — deltanet 2.085B, full_attn 0.598B, …
               0.642 B of them have no imatrix entry and fell back to rtn
               — hyper_conn 0.007B, lm_head 0.636B

## Three arms, because ggml does two things in one function

`quantize_row_q4_0_impl` reaches `make_qx_quants`, which replaces
round-to-nearest with a weighted least-squares scale **and** a nineteen-rung
sweep of the initial scale, and only then folds in the imatrix. Measuring
"imatrix against RTN" alone would credit calibration with both, so
`LLM_DENSE_SIM_QUANT` has three settings:

| arm | what it is |
|---|---|
| `rtn` | `d = -maxval/8`, round — ggml's **uncalibrated** path, and what every L8c-1 rung used |
| `search` | `make_qx_quants` with rmse_type 1, no imatrix — the scale search alone |
| `imatrix` | the same with unsloth's columns in the weight — `quantize_row_q4_0_impl` verbatim |

## Finding 1: the port is bit-identical to ggml, and getting there mattered

A transcription that optimises *something* looks fine on a
reconstruction-error table — it reduces error, just not ggml's. So
`reference/quant_ref.c` is the oracle, on `dequant_ref.c`'s precedent:
`ggml_quantize_chunk`, the entry point `llama-quantize` uses, then
`ggml_get_type_traits(t)->to_float` straight back. Passing an imatrix or not
selects both arms out of one binary.

The first port disagreed with it on **0.07%** of values. Three details had to
match, and every one is a case where the more accurate choice is the wrong
one:

1. **`nearest_int` rounds half to even.** It is the `+1.5*2^23` magic-number
   trick, so the FPU's rounding mode decides ties — where
   `quantize_row_q4_0_ref` and every quantiser in this repo round half away
   from zero. That is why `rtn` matched bit for bit without it and the
   nineteen-rung search did not.
2. **`make_qx_quants` accumulates in `float`.** The rung is chosen by
   `sumlx*sumlx > best*suml2`, a comparison between near-equal quantities; a
   float64 accumulator picks a different rung on about one group in a
   thousand.
3. **`sigma2` is a `float` accumulator too**, and widening it shifts every
   weight in the row and with it the rung.

With all three matched, all four cases — two widths, calibrated and not — are
**identical to ggml's own output, half for half, over 819 200 values**
(`TestQuantSimMatchesGGML`). That also validates every L8c-1 rung
retroactively: our `rtn` q4_0 *is* `quantize_row_q4_0_ref`.

The comparison is against the **half**: ggml's `to_float` returns the f32
product where the simulation stores `fp16(q*d)`, because that is what a kernel
reading the tile forms (L8b-2).

## Finding 2: calibration is worse, end to end, and the loss is concentrated

Eight chunks, baseline 2.0189:

| scope | rtn | search | imatrix |
|---|---:|---:|---:|
| every streamed dense family at q4_0/32 | **2.3927** (+18.5%) | 2.4802 (+22.9%) | 2.4702 (+22.4%) |
| hyper_conn alone | **2.1396** (+5.98%) | — | 2.2014 (+9.04%) |
| deltanet alone | 2.0489 (+1.49%) | — | 2.0487 (+1.48%) |

**The scale search alone already loses**, and the imatrix recovers only a
little of it. Per family the loss is entirely the hyper-connection block: the
DeltaNet does not move at all, and `hyper_conn` goes from +5.98% to +9.04%.

That is the sharpest inversion in this stage, because on *weight
reconstruction* the calibrated arm wins everywhere, and wins **most** on the
hyper-connection block — 1.148x on `hc_attn_down` against 1.065-1.074x
elsewhere. The family calibration helps most on the metric it optimises is the
family it damages most in the model.

## Finding 3: it is a gain error, not noise — and gain compounds

A quantiser's error splits into two parts that do not cost the same. The
**gain** is the least-squares scalar relating the reconstruction to the
original, `sum(w_hat*w)/sum(w*w)`; a gain below one is a systematic shrinkage
of the whole matrix. The **residual** is what is left after removing it, and
it averages out across a wide sum. `TestQuantSimGain`:

| tensor | rtn gain | search | imatrix |
|---|---:|---:|---:|
| `hc_attn_down` | −0.124% | −0.355% | −0.312% |
| **`hc_attn_up`** | **−0.110%** | −0.368% | **−1.346%** |
| `attn_qkv` | −0.084% | −0.335% | −0.237% |
| `output` | −0.089% | −0.316% | *no entry* |

**Round-to-nearest has three to four times less systematic shrinkage than the
search, and on `hc_attn_up` the imatrix has twelve times more.** The residual
goes the other way — the search really is the better fit — but a bias does not
average out, and L6b-3 measured the amplifier: a clean geometric **x1.085 a
layer**, over 48 of them, on a weight the hyper-connection block reads 97
times a pass.

That is the mechanism. `make_qx_quants` minimises squared error with the
levels clamped, and that optimum trades a little bias for a lot of variance —
the right trade for one group in isolation, the wrong one for a residual
stream.

## Finding 4: why `hc_attn_up`, and the number that predicts it

An importance-weighted scale is a least-squares fit over the columns of one
scale group, and how well conditioned it is depends on how evenly the
importance is spread inside that group. The **participation ratio**
`(sum w)^2 / (n * sum w^2)` is exactly that, and times 32 it is the effective
number of columns the scale is being fitted to. `TestImatrixSkew`:

| tensor | k | max/median | top-1% share | **effective columns per group** |
|---|---:|---:|---:|---:|
| **`hc_attn_up`** | 320 | **24 740** | **89.9%** | **4.4** |
| `hc_attn_down` | 10240 | 198 | 30.5% | 11.6 |
| `ssm_out` | 6144 | 1033 | 56.0% | 13.3 |
| `attn_qkv` | 2560 | 46 | 13.5% | 20.0 |
| `attn_q` | 2560 | 34 | 7.8% | 22.5 |

**The hyper-connection up projection's scale is fitted to an effective 4.4 of
its 32 columns**, because its input *is* the low-rank space — the down
projection's output, which is a gate — and 1% of those 320 columns carry 90%
of the activation energy. Twenty-eight of every thirty-two weights get a scale
chosen to serve the other four.

So the rule this stage yields, and it is not the one the literature suggests:

> **An imatrix helps in proportion to how evenly importance is spread inside a
> scale group.** Where it is concentrated, an importance-weighted scale is fit
> to an effective handful of samples, and what goes up is the systematic
> error — the part that compounds.

## Finding 5: correcting the gain recovers a third, and not the rest

If the mechanism is gain, the correction is one line: keep the levels
`make_qx_quants` chose and replace its scale with the unbiased one,
`d = sum(x*x) / sum(l*x)`, which forces the reconstruction to have no
systematic component along the original. `LLM_DENSE_SIM_QUANT=imatrix+gain`:

| scope | rtn | imatrix | imatrix+gain |
|---|---:|---:|---:|
| hyper_conn alone | **+5.98%** | +9.04% | +7.82% |
| every family | **+18.5%** | +22.4% | +22.8% |

**It recovers a third of the hyper-connection regression, which confirms the
mechanism, and it does not close the gap to round-to-nearest.** Gain is a real
part of the story and not all of it; what else the search costs is written
down rather than chased, because the conclusion does not turn on it.

## What this settles

**L8c-1's recommendation stands, and stands on firmer ground.** The one
untested lever has been tested: calibration as llama.cpp does it does not make
4-bit dense weights viable on this model, so the mixed plan's **+5.98% for
1.38x the bytes** (L8c-1) is the honest end of the dense re-quantisation, and
whether that is worth its kernel is a decision about throughput elsewhere —
the router, the experts, MTP speculation, batching — rather than about widths.

**Two scoping caveats, because neither is closed by this.**

- **This is Q4_0's symmetric form, not Q4_K's.** The checkpoint's own experts
  are Q4_K — an asymmetric super-block with its own min and scale — and that
  is the format unsloth's imatrix was collected for and used against. Nothing
  here says calibration fails for K-quants; it says it fails for the
  *symmetric* 4-bit format D7 chose, on the families whose importance is
  concentrated. **That reopens D7 with a different argument than L0d's**: the
  case for asymmetric is no longer a 5% error reduction, it is that the format
  calibration actually works with may be the asymmetric one.
- **The lm head was never calibrated**, here or in principle: the published
  matrix has no entry for it. It is 14% of a dense token and L8c-1 measured it
  at +0.28% uncalibrated, so nothing hangs on it — but "calibration does not
  help" is a statement about the 84% that had data.
