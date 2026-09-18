# P3 — the shipped-widths decision, and D18

> 2026-09-18. L8c closed at P2 with every streamed dense family on one
> 4.5-bit bank, and it closed with a question rather than a recommendation:
> the uniform plan is **provably not optimal in pp-per-GB**, additivity had
> just broken at the sixth family, and the knapsack had never been solved
> with measurements. This stage measures it — six 145-chunk arms, five of
> them new — and the answer is smaller than the question, in a useful way.
>
> **D18: ship the uniform 4.5-bit plan with `ple_proj` on int8.** 0.0165 GB a
> token, for 0.367 points of perplexity: **4.1850 (+3.87%) against the
> uniform plan's 4.2010 as built (+4.27%; 4.1998 simulated)**. Everything better
> than that on the frontier needs a kernel that does not exist, and the
> stage's second output is the measured case for building it.

## 1. What the knapsack actually looked like

P2 handed over a frontier with six rows and one tool (additivity) that had
just been shown to leak. The costs are arithmetic — `cmd/gguf`'s group table
re-prices the decode budget exactly, reproducing L8c-3's 4.264 GB a token to
three decimals — so the only unknown was the accuracy column, and the only
instrument that can read it before a kernel exists is `sim.go`.

Two things shaped which arms were worth eight minutes each.

**The real bank is nibbles, and that is a hard floor.** `llm/bank_q4.go:144`
refuses anything but four bits, and the comment says why: a fifth bit is a
plane of its own the way ggml's `qh` is — a third stream through the unpack,
and a tile that is no longer one byte per two elements. So `q5_k` is not a
width the shipped code can stage; it is a kernel stage. **int8, by contrast,
costs nothing to build**: a family left off `LLM_DENSE_BANK` stages on L8a's
int8 bank, which is how every L8c stage advanced one family at a time. The
decision is therefore three-way per family — q4_k (built), int8 (built),
q5_k (a stage) — and the int8 arms deserved measurements rather than the
estimates idea 4 carried.

**`ple_proj`'s headline number was quoted on the wrong basis.** LLM2 ranks it
at ~12 pp/GB, from 0.047 GB — but that 0.047 is the saving *from halves*,
where every other family's pp-per-GB in the same table is quoted from int8
(`full_attn` 0.299, `deltanet` 1.07, `lm_head` 0.318 are all int8 → q4_k).
On the consistent basis `ple_proj` buys **0.0165 GB**, and its standalone
+0.59% is **~35 pp/GB**, three times worse than the number the review
carried. It is the plan's worst trade by a wider margin than anyone thought,
and — because the bytes are so few — the cheapest thing on the board to buy
back.

## 2. The six arms

Every arm is a **complete plan** run over the whole 145 chunks, not a family
in isolation: P2 showed additivity deviating 0.22 pp once the n-gram block is
in play, so composing a plan from per-family deltas is no longer sound. All
six are `LLM_DENSE_SIM_QUANT=imatrix`, full scope (no `SRC=q8` — after P2's
tail deletion the bank quantises exactly what the simulation does), and all
are against our own 4.0289.

| plan | PPL (145 chunks) | delta | Δ GB/token | pp recovered | **pp/GB** | buildable? |
|---|---:|---:|---:|---:|---:|:--|
| uniform `q4_k/32` — the control | 4.1998 ± 0.02399 | +4.24% | — | — | — | shipped |
| `ple_proj` → `q5_k` | 4.1919 ± 0.02394 | +4.05% | +0.0041 | 0.196 | **47.8** | needs `qh` |
| **`ple_proj` → int8** | **4.1850 ± 0.02389** | **+3.87%** | **+0.0165** | **0.367** | **22.0** | **today** |
| `full_attn` → `q5_k` | **4.1560 ± 0.02361** | **+3.15%** | +0.0773 | 1.087 | **14.1** | needs `qh` |
| `lm_head` → `q5_k` | 4.1743 ± 0.02374 | +3.61% | +0.0795 | 0.633 | 8.0 | needs `qh` |
| `hyper_conn` → `q5_k` (L8c-3, with attn) | *4.1377* | *+2.70%* | +0.0801 | ~0.45 | ~5.6 | needs `qh` |
| `full_attn` → int8 | 4.1385 ± 0.02348 | +2.72% | +0.3277 | 1.522 | 4.6 | today |

Four findings come straight off that table.

**The fifth bit is worth two-thirds to three-quarters of the whole width, for
a quarter of the bytes.** `full_attn` at `q5_k` recovers 1.087 pp of the 1.65
the family costs at 4.5 bits — 66% — for 26% of what int8 costs. `lm_head`
recovers 77%, and the reason is visible in the staging line: unsloth's matrix
has **no row for `output.weight`**, so all 0.636 B of `lm_head` is quantised
round-to-nearest, and an uncalibrated weight has more to gain from a level
than a calibrated one does. The *marginal* step from `q5_k` up to int8 on
`full_attn` is 0.435 pp for 0.225 GB — **1.9 pp/GB**, worse than every other
dial on the board and the clearest "stop here" on the frontier.

**Idea 4's int8 option was 0.65 pp optimistic, and it is strictly
dominated.** The review priced `full_attn` back to int8 at ~+2.07% from
additivity; measured it is **+2.72%**. And L8c-3 already measured a plan that
beats it on *both* axes at once: `hc + attn` at `q5_k` is 4.1377 (+2.70%) at
**4.419 GB**, against int8's 4.1385 (+2.72%) at **4.592** — the same accuracy
to a thousandth of a point, 0.17 GB dearer, ~2 tok/s of ceiling worse. Idea 4
can be closed: **no int8 arm belongs in the plan except `ple_proj`'s.**

**Additivity leaks in both directions, and both leaks are the n-gram block.**
`ple_proj` standalone costs +0.59%; measured as a marginal upgrade inside the
plan it is worth only **0.367 pp** — a 0.22 pp shortfall that is *exactly*
P2's 0.22 pp deviation, now observed from the other side. In the other
direction, `hc + attn` at `q5_k` recovers 1.54 pp where `full_attn` alone
recovers 1.087 and `hyper_conn`'s entire q4_k cost is 0.31 — the two cannot
sum to 1.54, so there is a **~0.15 pp super-additive interaction** there too.
Standing rule, and it is now load-bearing: **a plan is measured, never
composed.** The per-family table remains a ranking instrument and nothing
more.

**The ranking of the screens held, the magnitudes did not — again.** Nothing
here was screened at 8 chunks, because L8c-4 through P2 recorded five
consecutive screen failures including two sign flips. Every number above is a
145-chunk run, ~6.5 minutes on the bank and ~8 on the simulation, which is
cheap enough that the screen has no remaining job.

## 3. D18, and why the answer is the small one

**`ple_proj` at int8 is the only upgrade that is free.** It is 0.0165 GB a
token — 0.4% of the bank — and both its tensors ship Q8_0, so the int8 stage
is bit-identical arithmetic rather than a re-quantisation. The simulation and
the real bank agree to four decimals *and to the same standard error*:

| | PPL (145 chunks) | delta |
|---|---:|---:|
| sim, `ple_proj` off the plan | 4.1850 ± 0.02389 | +3.87% |
| **the bank, same plan** | **4.1850 ± 0.02389** | **+3.87%** |

which is what P2's tail deletion bought and the third demonstration of it.

**And the bytes are not resolvable.** Three interleaved A/B pairs, all in
one hour on one machine state (P1c's rule), `-gen -n 64 -prompt 'The capital
of France is'`:

| pair | uniform `q4_k` | D18 | diff |
|---|---:|---:|---:|
| 1 | 35.86 | 35.80 | −0.06 |
| 2 | 35.55 | 35.69 | +0.14 |
| 3 | 35.82 | 35.87 | +0.05 |
| **mean** | **35.74** | **35.79** | **+0.04** |

The sign flips across pairs and D18 finishes marginally *ahead* on the mean.
The **within-arm** spread is 0.31 tok/s — seven times the between-arm
difference — so the honest statement is not "it is free" but "**the cost is
below this instrument's floor**": 0.0165 GB at the 178 GB/s the streaming
dispatches reach predicts 0.093 ms a token, 0.12 tok/s, and three pairs
cannot resolve that. Interleaving mattered: a single pair would have read
−0.06 and a different single pair +0.14, and either alone would have been a
number rather than a bound.


The fresh decode inventory P2 asked for, from `cmd/gguf`'s group table at
D18's widths:

| group | bits/w | GB/token |
|---|---:|---:|
| moe_experts (as shipped) | 5.10 | 1.504 |
| deltanet | 4.50 | 1.174 |
| hyper_conn (ten-group super-block) | 4.475 | 0.359 |
| lm_head | 4.50 | 0.358 |
| full_attn | 4.50 | 0.336 |
| moe_router (F32) | 32.00 | 0.252 |
| moe_shared | 8.51 | 0.251 |
| **ple_proj (int8)** | **8.55** | **0.035** |
| qsa_indexer | 4.50 | 0.011 |
| **total** | | **4.281** |

**56.5 tok/s at 242 GB/s, 53.0 at the 227 the dispatches reach.** One
warning, because the collision is unfortunate: this **4.281 is arithmetic**,
and P1's 4.281 GB is a *measured* dispatch-byte total at a different bank
state. They are not the same quantity and agree by coincidence. P2's "~4.15
GB by subtraction" chained off the measured one; the table above supersedes
that chain.

## 3b. A second corpus: the ranking is invariant, the magnitude is not

Idea 5's first half. +4.27% rests entirely on wikitext-2, and the instrument
is corpus-agnostic, so the plan was re-graded on **Go**: the Go 1.x standard
library's non-test sources, concatenated in path order and cut to
wiki.test.raw's own byte count (1 290 527 B), capped at the same 145 chunks
so every run is the same length. It is a large domain shift — prose to code
— and it is regenerable in one line (see `LLM.md`'s command block).

| plan | wikitext | delta | Go stdlib | delta | pp back (wiki) | pp back (code) |
|---|---:|---:|---:|---:|---:|---:|
| as shipped, int8 | 4.0289 | — | 1.6288 ± 0.00631 | — | — | — |
| uniform `q4_k`, the bank | 4.2010 | +4.27% | 1.6560 ± 0.00645 | **+1.67%** | — | — |
| **D18**, the bank | 4.1850 | +3.87% | **1.6506 ± 0.00642** | **+1.34%** | 0.367 | 0.33 |
| `full_attn` at `q5_k`, sim | 4.1560 | +3.15% | 1.6450 ± 0.00636 | **+0.99%** | 1.087 | 0.68 |

(Percent of perplexity is comparable across two corpora whose baselines
differ this much because `dppl/ppl = exp(dnll) - 1 ~ dnll`; the code
baseline's 1.6288 does not inflate or deflate the column. Recoveries are
quoted against each corpus's own uniform-`q4_k` row.)

**Both decisions survive, and both look better off wikitext.** The ranking is
identical on the two corpora — uniform worst, D18 better, `full_attn` at
`q5_k` best — which is the thing idea 5 was insurance against. What changes
is scale and share:

- **The whole plan costs 2.5x less on code** (dnll 0.0166 against 0.0418).
  **+4.27% is the pessimistic end of the range, not a universal number**, and
  it should be quoted as a wikitext figure from here on.
- **`ple_proj` is proportionally worse on code, not better**: 0.33 pp of a
  1.67 pp plan is **20% of the whole damage**, against 8.7% on prose, for
  0.4% of the bytes. The one family present at a single layer, run once, is
  the n-gram block's — and the n-gram block matters more to code than to
  prose. D18 strengthens.
- **So does the `q5_k` case**: `full_attn`'s fifth bit recovers **41%** of
  the plan's cost on code against 26% on prose.

The three bank arms and the one simulated arm mix instruments, which is only
sound because P2's tail deletion made bank and simulation the same
measurement — and section 3 re-demonstrates it to four decimals on this very
plan.


## 4. What this hands the next stage

**Build the `qh` plane.** This is the stage's second output and it is now a
measured recommendation rather than a guess. `full_attn` at `q5_k` is 14.1
pp/GB — three times the best int8 arm — and a plan carrying it lands near
**+2.5% at ~4.36 GB**, which no combination of widths the current kernel can
stage comes close to on both axes. The cost is known and bounded: a third
stream through the unpack, a tile that is no longer one byte per two
elements, and `bank_q4.go`'s four-bit check becomes a two-format branch.
Order the families by measured return: `full_attn` (14.1), `lm_head` (8.0),
`hyper_conn` (~5.6), `deltanet` (~2.8 inferred, and 0.26 GB — the one family
where a fifth bit is genuinely expensive).

**`ple_proj` at `q5_k` is not worth its own kernel** (47.8 pp/GB is the
steepest slope on the board, but the absolute prize is 0.196 pp), and it
should ride along free once the plane exists.

**What is still open from idea 5**: the downstream task-level eval. There is
no multiple-choice set on this machine and fetching one was out of this
stage's scope; the cross-corpus half is done above.

Files: `llm/sim.go` (`ShippedDenseBank`), `cmd/serve/main.go` (the opt-in);
the second corpus is `models/code-corpus/go.test.raw`, gitignored with the
rest of `models/` and regenerated by the line in `LLM.md`'s command block.
Results: `results/p3_ppl_attn_q5k.csv`, `p3_ppl_ple_q5k.csv`,
`p3_ppl_head_q5k.csv`, `p3_ppl_attn_int8.csv`, `p3_ppl_ple_int8.csv`,
`p3_ppl_d18_bank.csv`, and the code-corpus set `p3_code_*.csv`.
