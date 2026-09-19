# P5a — the draft head runs, the wiring is settled, and the acceptance rate is 74%

> **2026-09-19.** P5's design pass said the one number that decides the stage
> is the acceptance rate, and that it can be had from the *side* of an
> ordinary decode with no rollback, no verification pass and no new kernel.
> It can, and it has been.
>
> **`a₁ = 74.0%` over 1024 rounds at n_ctx 2048, against a negative control
> at 9.8%**, and `E[tokens]` per round of 1.74 at depth 1 rising to 3.02 at
> depth 6. With P5b's projected kernel that is **~1.33x, optimum at depth 1
> or 2**; on today's kernels it is **0.52x — still a loss**, exactly as the
> design pass predicted.

Data: `results/p5a_wire_teacher_r{1,2}.csv` (the wiring screen, twice),
`results/p5a_ctx2048.csv` (the long-context arm),
`results/p5a_free_{unprimed,primed}.csv` (the arm that had to be thrown away).
Code: `llm/mtp.go`, `cmd/llm/mtp.go`, `go run ./cmd/llm -mtp`.

---

## 1. The wiring, which had no oracle and now has a measurement

D5 — *llama.cpp is the oracle* — had an answer for every other block in this
vertical and none for this one. `src/models/qwen4exp.cpp` contains no `nextn`
symbol at all, so `blk.48` is never created and never loaded; transformers'
`models/qwen4_exp/modeling_qwen4_exp.py` carries
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and discards the weights on
load. The nearest implemented relative is `qwen3next.cpp`'s `graph_mtp`, and
it is a model without hyper-connections.

What the checkpoint states is a contradiction on its face. `nextn.eh_proj` is
`[5120, 2560]` — K = 2·n_embd, the shape every architecture in llama.cpp
creates — but `nextn.hnorm` is `[10240]`, not `[2560]`, because this model's
residual is four hyper-connection streams. 2560 + 10240 is not 5120.

So the readings were **measured against each other**, all five from one walk
of the trunk (§4 says how), 256 rounds of wikitext, depth 6:

| wiring | what it hands the block | a₁ | E[tokens] @6 |
|---|---|---:|---:|
| **`res`** | the **wide 10240 residual**, `hnorm` as `[2560, 4]`, `eh_proj` **once per stream**, embedding first | **73.4%** | **2.652** |
| `mixed` | the collapsed `result_norm` broadcast into four streams | 55.5% | 1.930 |
| `res-flip` | as `res`, concatenation the other way round | **0.0%** | 1.000 |
| `mixed-flip` | as `mixed`, flipped | 0.4% | 1.000 |
| `zero` | **a zeroed hidden state — the negative control** | 9.8-10.5% | 1.105 |

**Three things fall out of one table.**

The per-stream reading is right and it is not close: `res` beats `mixed` by
18 points, and `mixed` is the reading that would have been chosen by analogy
with `hc_init` (which broadcasts the token embedding into every stream at the
top of the trunk). Both *work*, which is why the control matters; only one is
the model.

**The concatenation order is load-bearing and it is llama.cpp's.**
`res-flip` scores **zero out of 256** — not degraded, extinguished. So
`ggml_concat(ctx0, e_norm, h_norm, /*dim=*/0)` in `graph_mtp`, which puts the
embedding first, describes this checkpoint too. A converter that had done it
the other way would have produced a file that looks exactly like this one and
a head that predicts nothing.

**And the control is what makes the 73.4% mean something.** A zeroed hidden
state leaves the block nothing but the embedding of the current token, and it
still gets 9.8-10.5% — that is the base rate for guessing a token two ahead
from one token of context, and it is the number every wiring arm has to be
read against. `res` is **7.5x** it.

### 1.1 The rest of the staging, and two gates that came with it

**The head is borrowed, and the checkpoint says it should be.**
`mtp_use_dedicated_embeddings` is false, and the draft shard's bundled
`token_embd` and `output` are the trunk's at coarser widths — measured, not
assumed: over six sampled rows of each, `output.weight` is **cosine 0.999792**
between the trunk's Q8_0 and the draft's Q6_K (relative rms 2.04%) and
`token_embd.weight` is **cosine 0.997280** between Q8_0 and Q4_K (7.38%).
Those are the two widths' own quantisation costs and nothing else, so the
draft shares the trunk's head and its embeddings. That is 1.27 GB of
residency not spent and a *more* accurate head than the shard ships; the real
added residency is **1.92 GB of weights and 0.43 GB of arenas**.

**Two formats had to be added to read the shard at all.** `mtp-…-Q4_K_M.gguf`
is a stock `Q4_K_M` mix rather than unsloth's `UD-Q4_K_XL`, so `blk.48` ships
its three `hc_*_up` matrices as **Q5_0** and `hc_ffn_down`, `attn_v` and its
own copy of `output` as **Q6_K** — neither of which is anywhere in the trunk.
Both are now in `gguf/dequant.go` and both are gated the way every other
format in this repo is: **element-for-element against llama.cpp's own
`to_float`**, through `reference/dequant_ref.c` and the committed
`testdata/dequant_{in,ref}.bin` (`TestDequantMatchesGGML`: *Q5_0 256 elements
exact*, *Q6_K 1024 elements exact*).

**One deviation from the checkpoint, stated rather than hidden.**
`attention.compress_ratios[48]` is **0**, which is what the trunk's *linear*
layers say, even though `blk.48` ships the indexer's `q_proj`/`k_proj` and
their norms. The array does not describe this layer and nothing else in the
checkpoint says what the MTP layer's selection budget is, so `NewMTPHead`
borrows the trunk's 4 — and refuses to stage above n_ctx 2051, where
`selWidth = min(top_k + ratio - 1, nKV)` stops naming every cell and the
choice would start to matter. Every number here is at or below 2048, where
the selection is the identity and is not dispatched at all.

---

## 2. The acceptance profile

Teacher-forced on wikitext — the trunk advances on the corpus's tokens so the
context stays on-distribution, and the draft's guess is compared against the
**trunk's own argmax**, which is what a speculative loop verifies against.

**n_ctx 2048, 1024 rounds** (`results/p5a_ctx2048.csv`):

| k | marginal aₖ | chained aₖ | E[tokens] at depth k |
|---:|---:|---:|---:|
| 1 | **74.0%** | **74.0%** | 1.740 |
| 2 | 53.0% | 65.0% | 2.222 |
| 3 | 36.5% | 64.7% | 2.533 |
| 4 | 28.4% | 70.5% | 2.753 |
| 5 | 22.0% | 72.0% | 2.911 |
| 6 | 16.6% | 70.4% | 3.022 |

*chained* is conditional on every earlier guess in the round being right,
which is the rate a speculative round actually runs at; E[tokens] is built
from the measured survival fractions rather than from a power of a₁, because
the chain is visibly **not geometric** — it drops to 65% at k = 2 and then
*flattens*, so a round that gets two right tends to keep going.

**Acceptance does not move with context.** The same arm at n_ctx 320 over 256
rounds is **73.4%** against 74.0 at 2048 over 1024 — 0.6 points apart on four
times the context and four times the sample. What does improve is the tail:
E[tokens] at depth 6 is 2.652 at ctx 320 against **3.022** at 2048.

**The instrument is exactly reproducible.** The five-arm screen run twice is
identical count for count in all thirty cells — 188/256, 120/256, 76/256, …
— which follows from greedy sampling, a fixed corpus and P1c's byte-identical
decode step, and means any difference between arms above is real rather than
noise.

### 2.1 Finding 1 — the free-running arm is an artefact, and the text says so

The faithful mode for a speculative loop is **free-running**: the trunk
advances on its own argmax, because that is the context a real loop sees.
Measured that way the head looks spectacular — **a₁ = 92.2%**, chained
90-94% all the way to depth 6, E[tokens] **5.56**, with the negative control
at 11.3%. Against P5b's projected kernel that would be **1.8-2.0x**.

**It is not a measurement of the draft head.** Printing what the trunk
generated (`-mtp-free` now does, and this is why) shows greedy decoding at
temperature zero falling into a verbatim loop within a few dozen tokens: the
same paragraph about Robert Boulter repeated three times in 423 tokens. A
looping trunk is trivially predictable, so the 92.2% is mostly the
degeneration. Priming with 96 real tokens first — which is what the run
above already did — does not prevent it; the loop starts after the prompt
runs out.

So the free-running arm is recorded and **discarded**, and the number this
stage reports is the teacher-forced one. The lasting part is a rule for
P5c's gate, which asks for "a measured multiplier at ctx 512 *and* at a long
context": **a speculation multiplier measured on greedy self-generated text
is measuring the sampler, not the speculation**, and it has to be read
against the text it ran on or taken teacher-forced.

The honest range stays open at the top end, and only a real workload closes
it: wikitext continuation is the hard case, and chat or code — more
templated, more predictable — plausibly sits between 74% and the artefact.
Nothing here measures that, and nothing should be claimed from it.

---

## 3. What it is worth

A round of depth M drafts M candidates, verifies them in a pass of **M+1
rows** (the committed token plus the M candidates) and emits `j+1` tokens
where `j` is the number of leading matches. Against a 27.63 ms decode step,
with the draft priced at `c_d` steps each:

    speedup(M) = E[tokens](M) / (pass(M+1) + M·c_d)

**The draft step costs 6.99 ms in this harness, and the split is the point**:
**2.21 ms is the host `nextn` block** — four 13.1 M-weight matvecs of
`eh_proj` on the CPU — and **4.63 ms is the layer's thirteen separate
submits**, against a byte floor of 0.538 GB / 4.132 = **0.130 of a step, 3.6
ms**. A product path moves `nextn` onto the device and folds the submits into
one recorded buffer, so `c_d` is between **0.130 and 0.166** (the measured GPU
half alone). Both are quoted below.

Using the measured ctx-2048 profile, P5's design-pass projection for an R-row
decode kernel, and today's measured pass costs:

| depth M | rows | E[tokens] | **with P5b** (c_d 0.130 / 0.166) | today |
|---:|---:|---:|---:|---:|
| **1** | 2 | 1.740 | **1.33x / 1.29x** | 0.52x |
| **2** | 3 | 2.222 | **1.32x / 1.27x** | 0.59x |
| 3 | 4 | 2.533 | 1.22x / 1.16x | 0.64x |
| 4 | 5 | 2.753 | 1.19x / 1.12x | 0.64x |
| 6 | 7 | 3.022 | 1.10x / 1.02x | 0.62x |

### Finding 2 — the optimum is depth 1, and that makes P5b much smaller

Every row of the "today" column is a loss, which is P5's design pass
confirmed end to end rather than projected: **speculation cannot be built on
the kernels that exist**, at any depth, even with a 74% draft.

And the depth that wins is **1** — because the MoE's expert set grows with
the row count (10 experts at 1 row, 17 at 2, 26 at 3, 36 at 4) while the
draft's own cost grows linearly in M and is four-fifths lm head. Both punish
depth, from opposite ends, and the acceptance profile's flat tail is not
enough to pay for either.

**A verify pass at depth 1 is two rows.** So P5b — "the R-row decode kernels"
— does not need arbitrary R at all to capture the whole multiplier: an
**R = 2 arm** of the four decode GEMVs is the cheapest possible version of
the feature and gives up 0.01x against R = 3. That is a materially smaller
stage than the design pass assumed, and it is the recommendation.

The caveat that cuts the other way: **P6 wants large R**, since R concurrent
sequences are R rows through the same weights, and there the expert-set growth
is the price of serving more streams rather than a tax on one. If both are
wanted, build R general; if only P5 is, build R = 2.

---

## 4. How it is measured, and why it is arranged that way

`-mtp` stages the trunk and the draft head side by side, walks a sequence one
token at a time, and **records** the walk; the draft arms are then replayed
over the recording. Three reasons, all of which turned out to matter:

- **No rollback is needed to measure acceptance.** The draft is a passive
  observer of a loop that already works. Its own state is one full-attention
  layer's KV cache, and `Step` sets `past` from the position it is given, so
  a chain that is abandoned leaves cells that are masked rather than stale.
  The whole of P5's rollback design is untouched here.
- **One trunk walk, many arms.** Restaging the trunk per arm is a minute each
  and would compare arms across different machine states, which is exactly
  what P1c's environment finding forbids. Recording `(h_i, result_norm_i,
  token, argmax_i)` is 13 MB for 256 steps, and each arm replays against it
  with a **fresh draft cache**, so no arm can contaminate another's history.
- **The trunk is not perturbed by the draft.** The walk is 28.0-29.1 ms a
  step across every run here, against the 27.63 ms the product decode
  measures — so the residual read-back and the draft's 2.35 GB of buffers
  cost the trunk about 2%, and the acceptance numbers are measured on a trunk
  behaving normally.

The draft head stages on **L8a's int8 bank**, not on `LLM_DENSE_BANK`'s D19
plan, and its experts on the checkpoint's own widths rather than D20/D21's:
both of those plans were chosen by 145-chunk perplexity runs over the trunk,
and the published imatrix has no row for any `blk.48` tensor, so narrowing
here would be an uncalibrated fit nobody has graded. It is 1.9 GB either way.

---

## 5. What is owed

- **The acceptance rate on a real workload.** 74% is wikitext continuation,
  the hard case; the free-running arm cannot answer it (§2.1) and chat or code
  through the HTTP API can. This is the one number that could still move the
  stage's value, and it is an evening's work now that the instrument exists.
- **`nextn` on the device.** 2.21 ms of every draft step is four matvecs on
  the CPU. It is an `eh_proj` of 13.1 M weights against a block that already
  has a GEMV kernel for exactly that shape, and it is 32% of the draft step.
- **A recorded draft step.** Thirteen submits where P1c showed the trunk's
  1407 dispatches fit in one; worth ~1 ms of the 4.63.
- Carried from the design pass, unchanged: **P5b** (now scoped to R = 2 for
  P5, R general for P6) and **P5c** (the rollback and the loop, with §2.1's
  rule about what the multiplier may be measured on).
