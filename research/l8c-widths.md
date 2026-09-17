<!-- LLM.md L8c, second item. The width question, answered by measurement
     rather than by L0d's proxy. Cited from llm/sim.go and cmd/gguf/main.go. -->

[← LLM.md](../LLM.md) · [research index](README.md) · accuracy

# L8c-1 — the widths, measured, and D3 does not survive them

**Result: ~4.25 bits on everything streamed costs 18.5% of perplexity, and the
best mixed plan measured here costs 5.98% for 1.38x the bytes.** The one
change that *is* free is the one D13 deferred: L8a-2's fp16 tail can be int8
for **+0.01%**, though it is worth only 0.6% of a token. D3 has been
the vertical's stated phase-2 target since L0; it was an inference from L0d's
weight-reconstruction ladder and from a bandwidth argument, and neither had a
perplexity beside it. Now one does, and the honest reading is that **the
re-quantisation is not ready to be built**: what it buys at the widths that
keep the model intact is a third of what D3 promised, and the one lever that
could change that — calibration — has not been pulled.

Four things are settled underneath that, and two of them are about the
instrument rather than the model.

Files: `llm/sim.go` (new), `llm/sim_test.go` (new), `llm/model.go`,
`llm/gpu_head.go`, `llm/graph.go`, `cmd/llm/ppl.go`, `cmd/gguf/main.go`.
No shader changed. Results: `results/l8c_widths.csv`,
`results/l8c_ppl_mixed.csv`, `results/l8c_ppl_tail.csv`.

---

## The instrument: a width, graded before its kernel exists

L8c has two halves that are usually done together and do not have to be. One
is **which widths**; the other is **the kernel and the layout** that make a
narrow bank fast. The second is weeks of work whose only purpose is
throughput. The first is a question about the model, and L8c-0 built the
instrument that answers it.

`Model.F32` is the seam every streamed dense weight crosses on its way to the
device, so a candidate format is one round trip inserted there:

    w  ->  (d, q) at the candidate width  ->  fp16(float(q) * float(d))

**The last step is why this is the format's own perplexity and not a model of
it.** L8b-2 settled that a kernel reading a quantised tile multiplies in
fp16 — `float16_t(q) * d` is one f16 instruction — and an f32 product of a
4-bit level and an 11-bit significand is exact, so rounding it once gives the
same correctly-rounded half. The simulation stages exactly the values the
kernel would multiply. What it cannot say is what the format *costs*, because
it stages halves; bytes are arithmetic and that is `cmd/gguf -width`.

A sim therefore forces the fp16 arm (`DenseQ8()`), because L8a's int8 bank is
an identity only for a Q8_0 tensor and staging a simulated Q4 weight through
it would quantise twice.

### It is checked against the case whose answer is known

**At `q8sym/32` over a Q8_0 tensor the round trip must be an identity**, by
D13: ggml picks `d = amax/127`, so re-deriving `(d, q)` from the dequantised
floats returns the checkpoint's own pair. `TestQuantSimIsIdentityAtQ8` holds
it on **696 M real weights** across all four families, bit for bit. If the
scale convention, the rounding rule, the group boundary or the fp16 store
were wrong anywhere, it would not.

And the same claim is then made *through the whole graph*, which is the row
that matters:

| 8 chunks, 8184 tokens | PPL |
|---|---:|
| the default bank (L8's int8 arm) | **2.0189** |
| the fp16 arm, no simulation | **2.0189** |
| `q8sym/32` restricted to the Q8_0 tensors | **2.0189** |

**A three-way identity.** The first two rows are L8a's and L8b's
bit-exactness claim measured at the corpus level rather than tensor by tensor;
the third is the simulator's exactness on the real model. One batch, three
controls, and every number below rests on them.

## Finding 1: the screen found a bug, and the bug is the reason for the tally

`lm_head` screened as **exactly** unchanged at 4.5 bits — 0.636 B parameters
re-quantised, and not one logit moved. That is not a result. The head is the
one dense weight that does not reach the device through `Model.F32`: it is
2.54 GB as floats, so `gpu_head.go` dequantises it a slab at a time and the
hook was not there.

**A family the simulation silently skips reads as "this family tolerates 4
bits", which is the most expensive wrong answer available in this stage.** So
the fix is not only the missing hook. Every run now prints what it actually
touched —

    simulated: 4.010 B weights — deltanet 2.085B, full_attn 0.598B,
    hyper_conn 0.640B, lm_head 0.636B, ple_proj 0.033B, qsa_indexer 0.020B

— which makes a zero visible rather than inferred, and
`TestSimFamilyCoversEveryDenseMatmul` is the static half: every family the
whitelist names has to match real tensors of the checkpoint, and the
exclusions (the router, the norms, `ssm_a`, `conv1d`, the two gathers) are
asserted as exclusions.

## Finding 2: perplexity at `-c 2048` cannot grade the QSA indexer at all

`qsa_indexer` also read as exactly 0.00%, and that one **is** a result — about
the instrument. L4a measured the selection as beginning to bite at token
**2051**, the width; a chunk is 2048 cells, so `top_k + ratio - 1` never
exceeds the cache, the selection is the identity, and `llm_attn_select.comp`
is not dispatched. The indexer's two projections are computed and their output
is not read.

So the indexer's width is free *at this context* and **unmeasurable by this
instrument**, and it will stay that way until a perplexity run uses `-c 4096`
or longer. It is 0.020 B parameters and 0.8% of a dense token, so the cost of
guessing is small — but it should be recorded as a guess.

## Finding 3: the cost of 4 bits is inverse to the bytes

Per family, alone, against the 2.0189 baseline. The right-hand columns are
`cmd/gguf`'s, so an accuracy row and a byte row name the same thing.

| family | 4.50 bits | 5.50 | 6.50 | GB/token | of dense |
|---|---:|---:|---:|---:|---:|
| **deltanet** | **+1.49%** | +0.14% | +0.37% | **2.247** | **46%** |
| full_attn | +3.97% | +0.80% | −0.46% | 0.635 | 13% |
| **hyper_conn** | **+5.98%** | +3.70% | +0.30% | 0.695 | 14% |
| lm_head | +0.28% | +0.05% | +0.08% | 0.675 | 14% |
| qsa_indexer | 0.00% | — | — | 0.039 | 0.8% |
| ple_proj | −0.51% | — | — | 0.035 | 0.7% |

**The family holding 46% of a dense token is the cheapest to narrow, and the
one that refuses 4 bits is 14% of it.** That is the whole argument for a plan
rather than a width: D3's single number is the one arrangement guaranteed to
be wrong at both ends.

**The columns have a floor and it is about 0.4%.** `deltanet` reads worse at
6.5 bits than at 5.5, `full_attn` at 6.5 reads better than the unquantised
baseline. More bits cannot help, so those are noise — the same resolution
L8c-0 found comparing four chunks to the corpus. Anything under ~0.4% in that
table means *indistinguishable*, not *measured*.

A mechanism worth writing down as a hypothesis rather than a finding: the
hyper-connection mixers **are** the model's plumbing — low-rank projections
deciding how four residual streams mix, 97 of them a pass, with L6b-3's
x1.085-a-layer drift compounding on top — where the DeltaNet's fused
projection feeds an L2 norm and a gated delta rule, which absorb error.

## Finding 4: for the sensitive family it is levels, not scale granularity

The obvious way to rescue `hyper_conn` cheaply is a finer scale group: L0c
made block-32 granularity cost 1.0% instead of 14.2% at prefill, so the bank
can afford it. It does not rescue it. Sixteen chunks, baseline 3.0436:

| format | bits/w | levels | group | Δ |
|---|---:|---:|---:|---:|
| q4_0/16 | 5.00 | 16 | 16 | +5.04% |
| q5sym/32 | 5.50 | 31 | 32 | +3.48% |
| q5sym/16 | 6.00 | 31 | 16 | +2.42% |

Halving the group at fixed levels is worth ~1.06 points (5.04 → ... at 16
levels, and 3.48 → 2.42 at 31); going from 16 levels to 31 at the *coarser*
group is worth 1.56. **The hyper-connection block wants resolution, not
outlier handling**, which is what low-rank mixer weights being well-behaved
rather than spiky would predict. D8's fp16 scale per 32 is the right
granularity and the bits are the axis to spend on.

## Finding 5: ggml's 16-level convention is 1.13x L0d's 15-level one, free

Every number in `research/l0d-quant-error.md` was measured with `q4sym`:
`d = amax/7`, levels clamped to [−8, 7], of which only fifteen are reachable
because `amax/7 * -8` is below `-amax`. ggml's Q4_0 uses `d = -maxval/8` and
reaches all sixteen. On a real DeltaNet projection the reconstruction error is
**9.85e-02 against 1.115e-01 at identical bits**, and end to end at four
chunks the gap is much larger than that suggests — **2.2813 against 2.4688**.

That is bigger than L0d's asymmetric-vs-symmetric 1.05x, which was the
comparison D7 was decided on, and it costs nothing. Every 4-bit rung here is
therefore `q4_0`, and **D7's 5% should be re-read as "against the wrong
symmetric baseline"**.

## Finding 6: per-family deltas do not add, and the plan is what gets measured

The plan the table above argues for — the two cheap families narrow, the two
sensitive ones wide:

| family | bits | GB/token | was |
|---|---:|---:|---:|
| deltanet | 4.50 | 1.174 | 2.247 |
| lm_head | 4.50 | 0.358 | 0.675 |
| full_attn | 6.50 | 0.486 | 0.635 |
| hyper_conn | 6.50 | 0.521 | 0.695 |
| qsa_indexer, ple_proj | 4.50 | 0.030 | 0.074 |
| router + shared expert | *out of reach* | 0.503 | 0.503 |

**4.575 GB a token against 6.334 — 1.38x — for a 52.9 tok/s ceiling** against
the checkpoint's 38.2 and this bank's 40.0. At L8e's measured 62% of ceiling
that projects to ~33 tok/s against today's 24.66.

Over the whole corpus it is **4.2699 ± 0.02432 against 4.0289 — +5.98%.**

The parts predicted about +2%. **A width plan is not the sum of its
families**: uniform 4.5 bits is 18.51% where its four families sum to 11.7%
(1.6x), and the mixed plan is 5.98% where its resolved parts sum to ~2%.
Errors injected at 36 or 48 depths do not add, they compound — L6b-3 measured
the carrier, a clean geometric x1.085 a layer. So a plan has to be measured as
a plan, and a per-family screen is for **ranking**, not for budgeting.

## Finding 7: the fp16 tail is unnecessary, and it is worth almost nothing

L8a-2 left four families as halves because the checkpoint does not ship them
as Q8_0 — the DeltaNet's F32 `ssm_alpha` and `ssm_beta`, the attention layer's
two BF16 indexer projections, the hyper-connection block's F32 `inject` — and
D13 deferred the decision: *"they keep their halves in a tail rather than
being re-quantised early — that is L8c's decision to make, with a perplexity
number beside it."* Here is the number.

| whole corpus | PPL | vs 4.0289 |
|---|---:|---:|
| the tail as halves (today's bank) | 4.0289 | — |
| the tail as `q8sym/32` | **4.0294** | **+0.01%** |

**Nothing, to five times inside the error bar.** So D13's tail can go: those
four families can be staged as int8 with a scale per 32 like everything else,
which deletes the `lowRank`/`gateOff` two-plane machinery from `llm_gemm.comp`
and `llm_gemv.comp`, the 0.33 MB a mixer L8b-1 pays to stage 32 low-rank rows
in both planes, and the branch L8a-3 measured at 1.27x when it was inside the
k-loop.

**The reason to do it is simplification and not speed**: the tail is 0.032 B
weights, so int8 saves **0.030 GB a token — 0.6%**.

And this is the finding that most needed the whole corpus. At eight chunks the
same comparison read **−0.53%**, which would have been reported as a free
accuracy *improvement* with a mechanism attached (fp16 cannot hold these
families' dynamic range; a per-group scale can). It was noise, in a direction
that invited a story. L8c-0's rule — a short run screens for broken, not for a
percent — earned its keep here.

One caveat on the scope: of the 0.032 B weights, **0.020 B are the indexer's**,
and finding 2 says the indexer does not affect the output at `-c 2048` at all.
So this result covers `ssm_alpha`, `ssm_beta` and `inject` — 0.013 B weights —
and the indexer's two projections ride along untested.

## What this settles, and what it stops

**D3 as written is retired.** ~4.25 bits on everything streamed is +18.5% of
perplexity for a 1.49x ceiling, and unsloth's choice to put every dense tensor
at Q8_0 now reads as deliberate rather than lazy. The mixed plan is the
defensible end of it and it is +5.98% for 1.38x.

**And L8c should not build its kernel yet**, which is the recommendation this
stage exists to make. Two reasons, in order of size:

1. **Nothing here is calibrated.** Every rung above is round-to-nearest with
   no importance weighting at all, against a checkpoint whose *experts* are
   imatrix-quantised — and unsloth publish the imatrix
   (`imatrix_unsloth.gguf`, 580 MB), which LLM.md has recorded as free since
   L0. The family that decides the whole plan, `hyper_conn`, is exactly the
   kind of low-rank, high-leverage tensor an imatrix helps most. Pulling that
   lever is days; building the W4A8 kernel and its layout is weeks, and its
   value is set by the widths calibration allows.
2. **The remaining bytes are not where the sim can see them.** The plan above
   leaves 0.503 GB a token in the router and the shared expert, and 1.504 in
   the routed experts, neither reachable through `Model.F32`. Router → fp16 is
   +1.5 tok/s of ceiling and is D3's own line; the experts at 4.25 bits are
   +3.0. Together they are worth more than narrowing `hyper_conn` and
   `full_attn` put together, and both are untested for accuracy.
