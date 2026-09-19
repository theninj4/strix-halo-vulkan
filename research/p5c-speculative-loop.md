# P5c — the rollback and the loop: lossless, and 0.95x

> **2026-09-19.** P5's design pass said what a rollback would have to restore,
> P5a measured the draft head at `a₁ = 74.0%` and set the optimum depth to
> one, and P5b built the two-row decode kernels. This stage is the round the
> three of them imply: the ping-pong state, the ping-pong rings, `Commit` and
> `Rewind`, `TestSpeculationRewindIsTheSequence`, and the loop itself.
>
> **It is built, it is lossless, and it does not pay.** Speculation generates
> the plain loop's text token for token and runs at **0.95x** of it on a short
> context — 34.65 against 36.38 tok/s — and **0.89x** on a long one, 28.20
> against 31.56, each over two runs on the **shipped banks** (D19 + D20 +
> D21). It reads 0.98x and 0.98x on the checkpoint's own widths, and §4.2 is
> about why narrowing the trunk makes speculation *worse* rather than better.
>
> The gap to P5a's projected 1.33x is three numbers, and none of them is the
> rollback: **the verification pass is 1.36 steps, not 1.17**; **acceptance on
> real generated text is 47-66%, not 74%**; and **at R = 2 a rejection costs a
> whole extra pass**, which the design pass priced as free and is not. Before
> this stage found three defects in the kernel family P5b shipped — two of them
> silently corrupting *every* two-row pass — it read 0.91x on the checkpoint's
> widths.

Data: `results/p5c_spec_c512_r{1,2}.csv`, `results/p5c_spec_long_r{1,2}.csv`.
Code: `llm/spec.go`, `Graph.Speculate`/`Commit`/`Rewind`/`ExtendRows` in
`llm/graph.go`, the slots in `llm/gpu_deltanet.go` and `llm/gpu_ple.go`,
`SnapshotBlocks` in `llm/gpu_attn.go`, `shaders/llm_dn_scan.comp`,
`shaders/llm_seq_hist.comp`, `cmd/llm -spec`.

---

## 1. What was built

### 1.1 The rollback, and the one place the design was wrong

§3 of the design pass inventoried what a pass mutates that outlives it. Three
of the four entries survived contact:

| thing | design | built |
|---|---|---|
| DeltaNet recurrent state, 113.25 MB | ping-pong, free | **as designed** — `llm_dn_scan.comp` loads S once and stores it once, so `SSM_STATE_DST` costs nothing |
| KV cache, indexer raw keys | free | **free** — a cell past the end is masked out of every score |
| host id list | free | **free** |
| pooled indexer blocks | free, "self-healing" | **not free** — see §1.2 |
| the two convolution rings, 7.5 MB | *defer* the `hist` dispatch and re-issue it with the accepted count | **impossible; ping-ponged instead** — see below |

**The deferred ring write cannot work, and the reason is one shared arena.**
The design's argument was that `llm_seq_hist.comp` is one dispatch at the end
of a block reading the pass's own activation arena, so a commit could re-issue
it with `tokens = j`. That is true of the PLE block, which runs once. It is
false of the gated DeltaNet: `aQKV` is **one arena shared by all 36 layers**
(`llm/gpu_deltanet.go`'s `alloc` — the ring is per layer precisely because the
projection is not), so by the time a pass ends it holds layer 47's projection
and nothing else. Thirty-five layers' worth of rows to re-issue from are gone.

So the rings ping-pong like the state. That needed one change to
`llm_seq_hist.comp`, and it is the change the kernel's own header argues
against: the pure write is right when the ring written **is** the ring read —
a slot this run did not produce is already in it — and wrong the moment the
destination is a second ring, which starts empty. The ping-pong arm therefore
writes every slot, carrying the ones it did not produce over from
`SEQ_HIST_PREV`. It is a *second* arm rather than a replacement: decode and
prefill still take the pure write, so the one-row path writes one row and not
three.

Cost of the whole rollback: **+120.75 MB of residency** (a second state slot
and a second pair of rings), **one extra ring row** per speculative pass per
layer, and **nothing per rejection** — a rewind is not copying anything back,
it is declining to flip two integers.

**And the residency is opt-in, so a product run does not pay it** (§7). The
second slot is staged only for `GraphOpts.Speculative`; without it every block
has one slot and `Speculate(true)` refuses rather than writing over the state
it is supposed to be preserving (`TestSpeculationNeedsItsSlots`). Measured
against `main` with everything in this stage applied and speculation off,
`-gen -n 64` on the shipped banks is **36.09 / 36.04 tok/s against a pre-stage
control's 35.90 / 35.93, on identical arenas of 795.6 MB** — so what P5c costs
the decode path when it is not used is nothing, and the `ctxZero` fix of §2.4
gives a little back.

### 1.2 Finding 1 — the pooled indexer table does not self-heal, and the reason is the word "final"

The design pass called the pooled block table free: "a block is only written
once complete, and re-running the positions rebuilds it from the re-written
cells". The first half is exactly why the second half fails. A block is
written **the one time** a run completes its `ratio` cells and never again, so
a pass that completes a block out of a token that is then rejected leaves a
row that nothing will rebuild until all of that block's cells are re-written —
and the selection reads it in between, for the cells of that block that are
real.

`Rewind` therefore snapshots it: the run's `[lo, hi)` rows of `hIdxK` in every
staged layer, which is at most `rows/ratio + 1` blocks of `idxDim` halves —
a few kilobytes, read out of a HOST_CACHED arena.

It cannot bite at the contexts this stage measures, and that is the point of
saying so: `selWidth` is `min(top_k + ratio - 1, nKV)` = 2051, so at
`n_ctx <= 2048` the selection names every cell, `AttnGPU.sparse` is false, and
the pooled table feeds nothing the attention kernel reads. It would bite at the
first context where speculation became worth turning on, which is a bad place
to discover it.

### 1.3 The loop, and why it is a two-state chain rather than a formula

At depth one the trunk is at position P with `x_P` known and not consumed. The
draft head is handed the trunk's residual and `x_P` and names a candidate for
`x_{P+1}`; the verification pass runs both as rows P and P+1; row P's logits
say what `x_{P+1}` really was.

- **accepted** — the pass *is* the sequence. Commit it, and row P+1's logits
  name `x_{P+2}` for nothing. **Two tokens, one pass, one draft.**
- **rejected** — rewind. The round still learned `x_{P+1}` from row P.
  **One token, one pass, one draft.**

And then the part that is not in the design pass's model: after a rejection
**two** tokens are known and neither is in the trunk, so the next pass is
`[x_P, x_{P+1}]` — two certain rows, no draft, and one new token. At general M
that re-run is free, because the next pass is M+1 rows whatever happens and
the accepted prefix rides in front of the drafts (§3.3c). At **R = 2 it is
not**: the prefix fills the pass and displaces the speculation.

So the round is a chain with two states, and its multiplier is not
`E[tokens]/cost` of one round type:

	speculating   pass + draft   2 tokens w.p. a, else 1 and a recovery next
	recovering    pass           1 token, then speculating again

which over a super-round is `(pass + draft) + (1−a)·pass` steps for exactly
**2** tokens. §4 puts the measured numbers in it.

### 1.4 The loop carries its own check on the rollback

A recovery pass re-runs a token whose value is already known, so row 0's
argmax has to name that token again. If a rejected pass had left anything in a
ring, in a recurrent state or in the pooled table, it would not. `Speculator.
Next` makes that an error rather than a comment; over the runs in §4 it fired
on none of the 348 recovery passes.

---

## 2. Three defects in P5b, and the instrument that found them

The verification pass is a two-row pass, and **two rows is a schedule nothing
in the suite exercised**. `TestGraphIsAChunkSplit` ran 128, 9 and 1; P5b's four
block tests ran two rows against the GEMM in isolation. The first thing this
stage did was run the whole graph two tokens at a time, and it did not
reproduce the whole prompt.

### 2.1 Finding 2 — `PinSchedule` stopped pinning at two rows

`MoEGPU.moePlan`'s pinned branch asked `MoEPlanFor(nTok)` for anything past one
token, which before P5b was guaranteed to be a GEMM because the GEMV was named
at one row only. P5b raised that bound to `GEMVMaxRows` and the pin quietly
stopped pinning: `Graph.PinSchedule(true)` went on returning nil while the
schedule changed under it. Every bit-exactness gate in this vertical is
asserted against the pinned schedule, so this is a hole in the instrument and
not only in the block. `moeGEMMPlanFor` is now the function the pin asks.

### 2.2 Finding 3 — the R-row expert kernels were never dispatched

P5b made `ROWS` a specialization constant and built one pipeline per row count.
The router's two dispatches named the specialized pipeline. **The four expert
dispatches did not** — `up`, `down`, `shexp.up` and `shexp.down` went on
naming the bare pipeline, which is the one specialized to a single row. So
every two-row batch in the model ran the one-row kernel over a two-row
schedule and each tile's second row kept whatever the arena last held.

At the whole-graph level it is a 45x error on the residual. In
`TestMoEGPUDecodeTwoRows` it was invisible, and the reason is worth keeping:
the test's `rowMoved` control is a control on the **combine**, not on the
GEMV. A different second token routes to a different expert set, so `Out()`
moves even when the expert kernels leave a row unwritten — and the row-against-
row comparison passes trivially when the reference pass is the last thing that
wrote the row being compared. The fix to the test is one line: **dirty the
arenas with a different token between the reference and the arm**, so a kernel
that skips a row is compared against the other token's values instead of
against its own reference. With that line and without the pipeline fix the
test fails by rms 4.8e-03; with both it passes at 3.4e-06.

This is P5b's own finding one stage later. *A control has to be able to fail*,
and "run the same thing twice and compare" is not one.

### 2.3 Finding 4 — the R-row GEMV re-read the bank, and the L1 argument does not survive 48 layers

With the two above fixed, a two-row pass was correct and cost **1.43 decode
steps** where P5b measured 1.17. The attribution names one dispatch:

| dispatch | plain, ms a token | spec, ms a token | per **round** |
|---|---:|---:|---:|
| `moe.up` | 6.26 | 10.23 | **2.42x** |
| `moe.down` | 3.67 | 4.50 | 1.82x |
| everything else | — | — | 1.02-1.05x |

against an expert set that grows **1.74x** (§3). `moe.up` was reading its bank
at **101 GB/s at two rows against 141 at one** — a second DRAM read, not an L1
hit.

`llm_moe_gemv.comp` put the row loop *outside* the dword loop, on the argument
that "the bank row is in the L1 after the first pass over it, where MAXROWS
pairs of live accumulators plus the unpack's temporaries are not free". On
P5b's two-layer fixture that is true. In the whole model it is not: forty-eight
layers of expert bank do not stay in anything, and the second pass over a row
goes back to DRAM.

Moving the row loop inside — a dword loaded once and consumed by every row
before the next is issued, at the cost of `MAXROWS` accumulators and the
unpack's arithmetic repeated per row — takes the block from 16.74 to **14.40
ms a token** and the whole loop from **0.91x to 0.98x**. The one-row path is
unregressed (27.30 against 27.37 tok/s). The unpack repeat is the cheap half:
this kernel is bound by how many loads it has outstanding, not by the shifts
between them.

### 2.4 Finding 5 — 453 MB of host memset a pass, and only a decode-shaped pass sees it

At the long context the verification pass's host `glue` was **3.93 ms a token
against the plain step's 0.30** — thirteen times — on the same dispatch count.
It is `DeltaNetGPU.Resize`, which zeroed the output projection's pad rows
(`arenaRows - nTok` of them) on **every call**, and `Graph.sublayer` calls it
once a layer: 12.6 MB × 36 = **453 MB of memset a pass** at a 1024-token
prompt.

Neither existing path could see it. A prefill has `nTok == arenaRows`, so the
pad is empty; P1c's pre-recorded decode step resizes once rather than per
layer. A two-row recorded pass over arenas sized for a long prompt is the
first thing in this vertical that is both short and re-recorded.

The fix is to track how many rows the last run dirtied and clear only those:
`glue` 3.93 → **0.49 ms a token**, and the long-context multiplier **0.91 →
0.98x**. Nothing about it is speculation's; it is a decode-path cost that only
speculation's shape exposed.

---

## 3. What the numbers are

Two runs each, `-spec -n 128 -spec-passes 2`, 48 layers, the two arms
interleaved in one process. **Two configurations, because they do not give the
same answer**: the shipped banks (D19 + D20 + D21, 78.53 GB, what
`cmd/serve -llm` stages) and the checkpoint's own widths (L8a's int8, 81.70 GB,
what `cmd/llm` stages unless told otherwise — see §4.2).

| bank | arm | prompt | cells | plain tok/s | spec tok/s | **ratio** | a₁ | tokens a round |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **shipped** | short, r1 | 34 | 512 | 36.34 | 34.56 | **0.95x** | 65.6% | 1.506 |
| **shipped** | short, r2 | 34 | 512 | 36.41 | 34.73 | **0.95x** | 65.6% | 1.506 |
| **shipped** | long, r1 | 1024 | 2048 | 31.54 | 28.20 | **0.89x** | 46.9% | 1.320 |
| **shipped** | long, r2 | 1024 | 2048 | 31.58 | 28.20 | **0.89x** | 46.9% | 1.320 |
| checkpoint | short, r1 | 34 | 512 | 27.30 | 26.71 | 0.98x | 64.1% | 1.483 |
| checkpoint | short, r2 | 34 | 512 | 27.38 | 26.65 | 0.97x | 64.1% | 1.483 |
| checkpoint | long, r1 | 1024 | 2048 | 24.71 | 24.30 | 0.98x | 60.9% | 1.449 |
| checkpoint | long, r2 | 1024 | 2048 | 24.72 | 24.33 | 0.98x | 60.9% | 1.449 |

**The shipped rows are the ones that count**, and the plain column is the check
that says so: 36.34-36.41 tok/s against `LLM2.md`'s carried 36.19, a same-hour
control agreeing with the headline to 0.6%.

**Lossless, and the control says so.** Every speculative arm emitted the same
128 ids as the plain arms beside it, token for token — see §5 for the one
qualification, which is about the plain loop and not about speculation.

A round, and where it goes (shipped banks, short context, means of the two
runs; the checkpoint-width column is beside it because §4.2 is about the
difference):

| | ms | steps | steps, checkpoint widths |
|---|---:|---:|---:|
| the draft head | 5.71 | **0.208** | 0.176 |
| the verification pass | 37.46 | **1.362** | 1.324 |
| the host decide (two argmaxes over 248320 logits, the residual copy) | 0.31 | 0.011 | 0.008 |
| **a round** | 46.20 | 1.581 | 1.508 |
| a plain decode step | 27.50 | 1.000 | 1.000 |

Per block, per **output** token — so a number under 1.00 is a block the
speculative arm runs *less* of per token of text:

| block | plain ms | spec ms | spec/plain |
|---|---:|---:|---:|
| hyper-conn | 5.04 | 3.51 | 0.70x |
| deltanet | 10.97 | 7.73 | 0.70x |
| attention | 3.50 | 2.47 | 0.71x |
| **moe** | **11.76** | **14.40** | **1.22x** |
| lm head | 3.51 | 2.41 | 0.69x |
| = on the GPU | 35.58 | 31.08 | **0.87x** |
| glue | 0.14 | 0.88 | 6.3x |

**The GPU does 13% less work per token of output, and the loop still loses.**
Four of the five blocks read the same bytes at two rows as at one, so per token
of text they cost what the acceptance rate says: 1/1.483 = 0.67, and they
measure 0.69-0.71. Only the MoE grows, and the rest of the deficit is the draft
head (0.176 of a step) and the recovery rounds.

### 3.1 The expert set, measured on decode traffic at last

The design pass read **17 experts at two tokens** off a two-token prefill of a
repeated prompt, and flagged that a repeated prompt is not the question. The
loop reads it off its own traffic — the distinct experts one layer's router
names across a pass's two rows, sampled every round:

| arm | E(2) | E(2)/10 |
|---|---:|---:|
| short (34-token prompt) | **17.4** | 1.74 |
| long (1024-token prompt) | **16.8** | 1.68 |

So the design pass's 17 was right, and it is the one number in P5's model that
did not move. With `moe.up` fixed, the MoE's per-round growth is 2.05x at the
short context and **1.57x at the long one** against a byte growth of 1.68-1.74
— at the long context the block is now *below* its bytes, which is the second
row overlapping loads the way the dense blocks already do.

### 3.2 The chain, checked against itself

From the measured `a = 0.656`, pass = 1.362 and draft = 0.208, the two-state
chain of §1.3 predicts

	(1.362 + 0.208) + (1 − 0.656) × 1.362  =  2.039 steps for 2 tokens
	1.019 steps a token  →  0.98x

against a measured 0.95x. The 3% is the host: the chain counts GPU-shaped
steps and the loop also pays `glue` and the draft's host `nextn` block.

**Break-even is `a = 0.686` in the chain and about 0.72 measured**, from
`(pass + draft) + (1 − a)·pass = 2`. On the checkpoint's own widths it is
0.622 and 0.66 — which is §4.2.

---

## 4. What it would take to make it pay, priced

Three things, in the order of what they are worth. All three are named by the
measurement above rather than guessed at.

**1. The recovery round — worth 0.95x → ~1.05x.** 25 of every 89 rounds carry
no draft and emit one token, and they exist only because R = 2 leaves no room
for a draft behind a re-run prefix. Removing them means committing a partial
accept, which means the state at P+1 as well as the state at P+2. The design
pass declined the snapshot at 1.4 ms a round for general M; at depth one it is
**one extra write of 113 MB, about 0.02 of a step**, and it buys `1.609/1.455`
instead of `1.609/1.575 · (1 + (1−a))`. Two ways to get it, both real work:

- a store inside `llm_dn_scan.comp`'s token loop, guarded by a uniform — which
  is a store inside the one loop in this model with a 512-deep carried
  dependency, and that kernel's header is about what a store there costs; or
- **split the scan into two one-row dispatches**, which is host-side only —
  every row offset the scan reads is a push field (`normOff`, `ssmGateOff`,
  `ssmBetaOff`, `outOff`) so a second dispatch is the same kernel with four
  offsets moved. It costs a third state slot, one extra read and write of the
  state (+0.034 of a step), 36 extra dispatches — and the same treatment for
  the rings, which would have to be written at both row counts during the pass
  because §1.1 says they cannot be re-issued afterwards.

**2. Pre-record the verification pass — worth about 2%.** `glue` is 0.88 ms a
token and the pass re-encodes 955 dispatches every round where P1c's decode
step replays one command buffer. The obstacle is that the state destination
flips with the committed slot, so it is two recorded buffers rather than one
— which is exactly the shape P1c already supports.

**3. Acceptance.** 65.6% on generated prose and **46.9%** on wikitext
continuation at a long context, against P5a's 74.0% teacher-forced. This is the
one lever that is not engineering: it is the draft head, and P5a's carried item
— *acceptance on a real workload, where chat or code is more templated than
prose* — is still the largest unknown in the stage.

With item 1 the multiplier is `(pass + draft)/(1 + a)`:

| a | today | + item 1 | + items 1 and 2 |
|---:|---:|---:|---:|
| 0.47 (long context) | 0.89x | 0.93x | 0.96x |
| 0.66 (short context) | 0.95x | 1.05x | 1.08x |
| 0.74 (P5a, teacher-forced) | 1.00x | 1.11x | 1.14x |
| 0.80 | 1.04x | 1.16x | 1.20x |

So **the whole remaining programme is worth ~1.08x at the acceptance this
model actually gives**, and speculation only becomes interesting above
`a ≈ 0.8` — which is why the stage ships the loop with the rollback rather
than shipping it turned on, and why the next move is a measurement of the
draft head and not a kernel.

### 4.2 Finding 7 — narrowing the trunk makes speculation worse, twice

The two configurations in §3 are not a detail. Going from the checkpoint's own
widths to the shipped banks — D19 + D20 + D21, 0.85 GB a token out of the
trunk, **27.4 → 36.4 tok/s** — moves the multiplier the *wrong* way, 0.98x to
0.95x short and 0.98x to 0.89x long, and it does it through two independent
channels.

**The fixed cost does not narrow with the trunk.** A draft step is 5.7 ms
either way, because `blk.48` stages at the checkpoint's own widths on purpose
(the published imatrix has no row for any of its tensors, so a 4.5-bit stage
there would be an uncalibrated fit nobody has graded — `llm/mtp.go`). But the
step it is a fraction *of* shrank from 36.5 ms to 27.5, so the draft went from
**0.176 of a step to 0.208** without changing. Break-even moves from a = 0.62
to a = 0.69 for that reason alone.

**And acceptance itself drops at the long context**, 60.9% to **46.9%**. The
two arms generate different text under different banks, so text and bank are
confounded and this is not a clean measurement — but the mechanism is
plausible and it is the one to test: the draft head predicts the *checkpoint's*
next token, and D19/D20/D21 move the trunk 1.74% of perplexity away from the
checkpoint. Acceptance is agreement between the draft and the trunk, so
quantising the trunk and not the draft is spending exactly that agreement.

**This generalises past P5.** Every stage on the throughput frontier that takes
bytes out of the trunk makes speculation's fixed costs proportionally larger,
and every stage on the accuracy frontier that moves the trunk away from the
checkpoint makes the draft head a worse predictor of it. The two programmes are
in tension, and the width work has already happened.

### 4.1 And a hard ceiling nobody had priced: 2051 cells

The draft head **refuses to stage past 2051 cells**, and this is the first
stage where that matters. `blk.48`'s `attention.compress_ratios` entry is 0 —
what the trunk's *linear* layers say — so the checkpoint does not state the
MTP layer's selection budget, and `NewMTPHead` borrows the trunk's ratio only
where the selection is provably the identity (`top_k + ratio - 1` = 2051).
Past that the number would be a guess.

So **speculation as built has a context ceiling of 2048 cells**, against a
trunk that runs to 8192 rows of prefill (P0) and prices 262k of KV. Lifting it
needs either a number from somewhere the checkpoint does not have one, or a
measurement that the selection is insensitive to the ratio on this one layer.
The long-context arm above is the deepest run this allows.

---

## 5. Finding 6 — a fresh sequence is not independent of the one before it

The lossless claim is only a claim about speculation if the plain loop
reproduces itself, so `-spec` prints the pairwise table rather than comparing
against one arm. It does reproduce at the long context: all four arms of both
runs emitted the same 128 ids.

At the short context it does not. `SPEC_PLAIN_ONLY=1` drops the speculative arm
and runs the plain loop three times over the same prompt in one process:

	plain0 ≠ plain1   first at token 24:  30713 against 41275
	plain0 ≠ plain2   first at token 24:  30713 against 41275
	plain1 ≠ plain2   first at token 98:  10597 against  4655

**It is deterministic** — the same three texts in every invocation, and
`plain0`'s token 24 is 30713 every time — so decode is not racy; a "fresh"
sequence is carrying something from the run before it. Each run leaves
different leftovers past the end of the next run's prompt, and at the short
context (34 tokens of prompt in 512 cells) there are leftovers to leave. At the
long context (1024 in 2048) the next prompt overwrites them.

What it is not: the recurrent state (`Reset` zeroes both slots), the
convolution rings (at position zero every tap reaches before the sequence), the
KV cache or the pooled table (masked, and `sparse` is false below 2051 cells).
What is left is an arena a kernel reads past the run's rows — the MoE's
sentinel row is the documented candidate and `TestMoEGPUPaddingIsInert` says it
is not that one.

It does not touch this stage's result: the speculative arms reproduced the
plain arms they were interleaved with, in every run, at both contexts. But
**any future claim of the form "token for token" in this vertical needs the
plain-against-plain row printed beside it**, and this is owed its own stage —
a `Reset` that is a reset, with a test that runs the same prompt after
different predecessors and requires the same logits.

---

## 6. The gate

`TestSpeculationRewindIsTheSequence` (`llm/spec_rewind_test.go`), four layers
and 512 tokens of the 4k trace, on the pinned schedule.

The schedule is the loop's own shape rather than "run some tokens and rewind":
a 480-token prefill, then rounds of a rewindable pass whose tail is a token the
model rejects, the rewind, and the accepted prefix re-run and committed —
cycling `keep` 1 and 2 against `draft` 1 and 2, so both the two-row pass P5 is
built around and the three-row case that puts the ring's oldest tap inside the
pass are covered. 22 committed rounds, 22 state-slot flips, and
**`result_norm` identical to the last place** against the prompt run whole.

**And its control.** The same schedule with the position rewound and the
histories left where the rejected pass put them — `Graph.rewindPosition`, which
exists for this and which nothing in the product calls — moves `result_norm` by
**rms 8.5e-01** against a reference whose scale is 18.6. Without that arm the
equality above would not be evidence that anything was rolled back, which is
the shape of mistake P5b made and wrote down.

`TestGraphIsAChunkSplit` gains two rungs for the same reason: **2 at a time,
which is a verification pass**, and 3 beside it, because two is also the bound
`GEMVMaxRows` names and a gate that only tests the bound tests the wrong thing.
Findings 2 and 3 are both visible there and nowhere else in the suite.

---

## 7. What is owed

**The loop is parked, not deleted.** It is off by default and costs the
product path nothing measurable; `cmd/llm -spec` is how it runs, and P5a's
observer is how the one number that would restart the stage gets measured.

- **Acceptance on a real workload**, and it is now the *first* thing rather
  than a carried item: 65.6% on prose and 46.9% on wikitext continuation
  against a break-even of ~0.72, so the table in §4 says the rest of the
  programme is not worth starting until this number is known on chat or code.
  P5a's observer measures it with none of this machinery (`cmd/llm -mtp`,
  ~3 minutes).
- **The partial-accept commit** (§4 item 1), which is the difference between
  0.95x and ~1.05x and the only structural lever left — worth building only if
  the item above comes back high.
- **A draft head at the trunk's widths** (§4.2), which would need an imatrix
  for `blk.48` that nobody publishes, or a graded fit of our own.
- **The 2051-cell ceiling** (§4.1) — either a ratio for `blk.48` or a
  measurement that its selection does not need one.
- **`Reset` that is a reset** (§5), with its own test.
- **`nextn` on the device.** Still four host matvecs and thirteen unrecorded
  submits: 6.42 ms a draft step, of which 2.2 is the host block. A recorded
  draft step would take the draft from 0.176 of a step toward the 0.130 byte
  floor.
- The `moe.up` unpack is still repeated per row (§2.3). Sharing it across rows
  means restructuring `dotDword` over six formats, and the measurement says
  what is left there is small: the block is at or below its byte growth now.
