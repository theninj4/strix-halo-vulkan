# P20 — llama.cpp's MTP for this model, read against P5's loop

> **2026-10-01.** llama.cpp PR [#29761](https://github.com/ggml-org/llama.cpp/pull/29761)
> ("Qwen4Exp: add MTP", am17an, opened 2026-09-30, **open and under review**
> as of this note; it supersedes unsloth's #28243) implements the draft head
> P5a built without an oracle and runs it through llama.cpp's `draft-mtp`
> speculation. **It reads 1.55x on a DGX Spark at depth 3**, where P5c's loop
> reads 0.95x at depth 1. The head and its wiring are the same. The loop is
> not, in four places — and one of the four is a defect in ours.

Sources read: the PR's diff (273/−50 over 11 files), its thread, and master's
`common/speculative.cpp` (`common_speculative_impl_draft_mtp`),
`src/llama-memory-recurrent.cpp`, `src/models/delta-net-base.cpp`,
`tools/server/server-context.cpp` and `tools/server/bench/speed-bench`.
Nothing here was run; the numbers in §1 are the PR's and the costs in §4 are
P19's and P5c's.

---

## 1. What the PR measures

DGX Spark, Qwen3.8-Flash-Next **IQ4_XS**, `llama-server -np 1 -lzm on
--spec-type draft-mtp --spec-draft-n-max 3`, SPEED-Bench's `qualitative`
prompts (`nvidia/SPEED-Bench` on Hugging Face, temperature 0, 1024 output
tokens), 24 samples:

| category | n | baseline tok/s | MTP tok/s | decode | acceptance |
|---|---:|---:|---:|---:|---:|
| coding | 4 | 28.27 | 41.91 | 1.48x | 0.605 |
| qa | 4 | 28.43 | 42.85 | 1.51x | 0.579 |
| rag | 4 | 27.96 | 46.48 | 1.66x | 0.691 |
| writing | 4 | 28.30 | 42.83 | 1.51x | 0.671 |
| multilingual | 4 | 28.69 | 48.05 | 1.67x | 0.778 |
| roleplay | 3 | 28.56 | 41.36 | 1.45x | 0.610 |
| humanities | 1 | 28.33 | 40.43 | 1.43x | 0.582 |
| **overall** | 24 | 28.36 | 43.88 | **1.55x** | **0.640** |

`acceptance` is speed-bench's `accepted / drafted`, so at a fixed three
drafts a round (`p_min` defaults to 0 and was not set) it is **1.92 accepted
of 3, ≈2.92 tokens a round**, and 2.92 / 1.55 says a round — a four-row
verification pass, the draft layer's catch-up rows and three draft steps
each with the lm head — costs **≈1.9 decode steps** there. Per drafted
position that averages higher than P5a's wikitext profile (74 / 53 / 36.5%
marginal, mean 0.545 over three positions, E[tokens] 2.53 at depth 3): a
real workload accepts more, which is the direction P5a guessed and the number
nobody had.

The thread: ggerganov started a conversion job and proposed an `is_empty()`
tidy-up of the recurrent-memory change; one user hit `token_embd.weight not
found` with unsloth's `mtp-…-shared-Q8_0.gguf`, and the answer is that the
MTP-only file has to carry its own embeddings and head (a requant). Neither
touches us: the head is borrowed from the trunk here (P5a §1.1).

---

## 2. What the PR builds

**The head, and its wiring.** `graph_mtp` is: `hnorm` loaded as `[n_embd,
hc]` and applied as a per-stream RMS of the trunk's **hc-wide residual before
the final mixer** (`res_hc`, exported as `h_nextn`); `enorm(tok_embd)`
repeated across the `hc` streams; `ggml_concat(e_norm, h_norm, 0)` —
embedding first — into `eh_proj` once, giving a fresh `[n_embd, hc]` state;
then `blk.48`'s own `hc_attn` mix → full attention with the indexer →
combine → `hc_ffn` mix → MoE → combine; then `nextn.hc_head_{norm,down,up}`
as the output mixer and the model's `output`. The next draft step chains on
the block's wide residual *before* its head mixer. The converter builds
`eh_proj` as `cat(fc_embedding, fc_hidden)` on dim 1 and names the mixer
`blk.48.nextn.hc_head_*`. **That is P5a's `res` arm to the letter** — the
per-stream `hnorm`, the embedding-first concatenation that scored 0 of 256
flipped, the block's own mixer at the output, the chain on the wide
residual — and the tensor names are the ones `llm/mtp.go` reads. D5 is back
for this block: once a build carrying the PR exists, `MTPHead.Step`'s logits
have an oracle.

One more line of the converter matters: `add_attention_compress_ratios(…
+ [ratio] * (block_count − n_layer))`. llama.cpp gives `blk.48` **the
trunk's compress ratio** and runs its selection at the trunk's budget. That
is the number P5c §4.1 said the checkpoint does not state; the reference
supplies it by assumption, and borrowing it past 2051 cells is now what the
oracle does.

**Two contexts, one model.** The target context runs with
`embeddings_nextn` on, so `h_nextn` (10240 floats a row) is read back for
every row of every batch. The draft is a second `llama_context` on the same
model with `LLAMA_CONTEXT_TYPE_MTP`: its memory is filtered to `blk.48`'s
attention and indexer caches only — no recurrent layer at all, which is the
PR's `ctxs_bufs.empty()` branch in `seq_rm` (a rollback there only moves the
position).

**The catch-up.** `process()` runs after *every* target batch — the prompt
and each verification batch — and decodes the draft layer over all of that
batch's tokens, each paired with the trunk's residual from the position
before it (the first row of a batch with `pending_h`, carried across
batches), **output flag off** (no lm head). The draft's KV cache therefore
holds a cell for every position the trunk has, written from the trunk's
residual; the rejected rows are then trimmed with `seq_rm` on the draft
context. `accept(n)` sets `pending_h` to the trunk's residual at the last
accepted row, which is the next draft's seed.

**The draft.** `n_max` steps of one row each, the layer and the lm head,
greedy under a top-k 10 sampler, with an optional confidence stop
(`p_min`, off in the table above) and `n_min`.

**Partial accept, by snapshot planes.** The target context is created with
`n_rs_seq = n_max`. The recurrent memory allocates `(1 + n_rs_seq) ×
mem_size` rows for the state `s`, the conv state `r` **and the PLE ring
`p`**; `ggml_gated_delta_net(…, K = n_rs_seq + 1)` returns the state after
each of the ubatch's last K tokens and the graph copies the K planes into
the cache (`[TAG_RECURRENT_ROLLBACK_SPLITS]`: `split_equal` keeps a
sequence's last K tokens in one ubatch); the conv ring is copied K times,
the windows ending at each of the last K tokens. `seq_rm(p0)` with `1 ≤
rollback ≤ n_rs_seq` sets `rs_idx[seq] = rollback` and the next graph's
`s_copy` reads `rollback × size + cell`, single use. **No ping-pong**: row 0
of a verification pass is the sampled token and is always committed, so the
pre-pass state is never wanted again. A rollback deeper than `n_rs_seq`
falls back to a full `llama_state_seq_*` checkpoint in the server.

---

## 3. Against P5c's loop

| | llama.cpp (#29761) | P5c (`llm/spec.go`) |
|---|---|---|
| head wiring | P5a's `res` arm | the same |
| the head's embeddings and lm head | the file's own (a requant) | the trunk's, borrowed |
| `blk.48`'s selection budget | the trunk's ratio, any context | the trunk's ratio **to 2051 cells, refused above** |
| draft input | the trunk's `res_hc`, read back a row at a time | the same, `Graph.Residual()` |
| **the draft's KV cache** | **every position, from the trunk's residual** (the catch-up) | **the prompt never, and a cell only on a speculating round** |
| depth | `n_max` = 3, a confidence stop available | 1, fixed |
| verification pass | `n_max + 1` rows, batched MoE (CUDA) | 2 rows, R-row GEMVs (`GEMVMaxRows` 3) |
| partial accept | K snapshot planes written by the scan; a plane index at the next read | none: a rejection is a recovery pass of two known rows, no draft |
| draft step | the layer + lm head on the device, one graph | the layer on the device, `nextn` on the host (2.2 ms), 13 submits |

### 3.1 Finding 1 — the draft head in P5c's loop attends over cells nobody wrote

Read off the code, not yet measured. `Speculator.Start` runs the trunk's
prefill, keeps the last row's residual and calls `MTPHead.Reset()`; nothing
runs the draft layer over the prompt. `Speculator.Next` calls
`MTPHead.Step` only on a speculating round, at position `at`, and `Step`
writes the one cell at `at`. On an accept the trunk advances two and the
next draft writes `at + 2`; on a rejection the recovery pass drafts nothing
and the next draft writes `at + 2` as well. So the draft's one-layer
attention, at every step, sees **no cell for the prompt and no cell at every
other position** — whatever the allocation or the previous arm left there.

P5a's instrument did not have this: the observer stepped the draft at every
position of its recorded walk, so its cache was complete, and it measured
74.0%. The loop then measured **65.6% at a 34-token prompt and 46.9% at a
1024-token one** and P5c §4.2 charged the gap to the narrowed trunk — a
confounded reading, as it said. A 1024-token prompt is 1024 empty cells in
front of the draft's attention; that is the simpler explanation of the
long-context number, and it is the first thing to test.

The lossless claim is untouched (the draft only picks rows; every emitted
token is the trunk's argmax). What is wrong is the acceptance rate, and
with it the stage's whole pricing.

### 3.2 Why depth 3 pays there and depth 1 was chosen here

P5a's optimum was computed on 2026-09-19's pass costs (a two-row pass at
1.17 projected, 1.36 measured) and the wikitext profile. Both inputs have
moved: P19 measured a one-sequence pass at **1.23 steps for two rows and 1.37
for three** on today's kernels, and §1 says a real workload accepts ~0.64 a
drafted token at depth 3. The MoE's expert growth is still what bounds the
depth — it is the reason their 1.9-step round is not 1.5 — and a four-row
pass here is unmeasured (`GEMVMaxRows` is 3; four rows fall onto the GEMM).

### 3.3 The partial accept, priced again

P5 §3.3 priced "snapshot S after every token" at M−1 writes of 113 MB and
declined it for the re-run; P5c §4 item 1 found the re-run is a whole pass
at R = 2 and named the snapshot the one structural lever (0.95x → ~1.05x at
depth 1). llama.cpp's form is the per-row store, done inside the scan op.
`llm_dn_scan.comp` loads S once and stores it once, so K planes are (K−1)
extra stores of 113 MB a pass: ~0.5 ms each at the bus, **~1.7 ms at K = 4,
~0.07 of a step**, against a recovery pass of ~1.2 steps. The rings are 7.5
MB a copy. Residency goes from the two ping-pong slots (241 MB) to K planes
(483 MB at K = 4), still behind `GraphOpts.Speculative`. The alternative
P5c named — the pass's scan as K one-row dispatches — costs a state read
per row and is about 0.1 of a step at K = 4; the in-loop store is the
reference's choice and the cheaper one.

### 3.4 The draft step

5.71 ms was 0.208 of P5c's 27.5 ms step and is **0.236 of today's 24.19**
(§4.2's effect again: the step shrank and the draft did not). The floor is
0.538 / 4.132 = **0.13** (3.1 ms). At depth 3 the drafts are 0.71 steps a
round as they are and 0.39 at the floor — a 0.3-step difference on a
~1.9-step round, so P5a's two owed items (`nextn` on the device, the draft
step recorded) are on the critical path. The catch-up rows (§2) ride in the
same dispatch: rows = the committed tokens since the last draft plus the new
one, logits for the last row only — the layer is 0.07 GB, so the extra rows
are a few hundred microseconds. One thing the reference does through the
host that we must not: at 128k cells the prompt's residual is 5 GB, and the
draft's prefill over it has to read the trunk's residual arena on the device
a ubatch at a time, not through a mapped buffer.

---

## 4. The plan, re-cut

`TODO.md`'s P20 is rewritten against this note. In order:

- **P20a — the catch-up** (the fix for §3.1): the draft layer over the
  prompt at `Start`, from the prefill's residual arena a ubatch at a time,
  and over every committed row each round, no lm head but for the drafting
  row. Gate: the loop's acceptance on a teacher-forced sequence equals the
  observer's, count for count — the two instruments on one text.
- **P20b — the pass at 1–4 rows** (`cmd/llm -graph -tokens 1,2,3,4`, the
  shipped banks, one sequence) and `GEMVMaxRows` 4 (`MAXROWS` in
  `llm_gemv.comp`, `llm_hc_gemv.comp`, `llm_moe_gemv.comp`, the fused
  router's one-to-three rows, a four-at-a-time rung in
  `TestGraphIsAChunkSplit` — a new row count needs a graph gate).
- **P20c — K snapshot planes** in `llm_dn_scan.comp` and K rings in
  `llm_seq_hist.comp`'s ping-pong arm, `Rewind(j)`, the ping-pong retired;
  `TestSpeculationRewindIsTheSequence` gains j = 1..3 of 4.
- **P20d — the draft at the floor**: `nextn` on the device, the step
  recorded, depth as a parameter.
- **P20e — the measurement, the PR's way**: free-running on SPEED-Bench's
  qualitative prompts through the chat template at temperature zero, the
  text read for P5a's loop artefact, two runs, the plain-against-plain row
  printed (P5c finding 6's `Reset` test first), the shipped banks. The
  2051-cell refusal comes out (§2) and the long arm runs past 2048.
- **P20f — prompt lookup**, demoted to an extra draft combined with the head.

Unverified arithmetic at today's 24.19 ms step, the PR's acceptance (E ≈
1.74 / ~2.4 / 2.92 tokens a round at depth 1 / 2 / 3), the draft at the
floor and 0.02 of catch-up:

| depth | rows | pass (steps) | round (steps) | tokens a round | multiplier | tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.23 (P19) | 1.38 | 1.74 | **1.26x** | 52 |
| 2 | 3 | 1.37 (P19) | 1.65 | ~2.4 | **~1.4x** | 58 |
| 3 | 4 | ~1.5 (P20b) | ~1.9 | 2.92 | **~1.5x** | 63 |

With P5a's wikitext profile in place of the PR's, depths 2 and 3 read 1.34x
and 1.32x, so the workload's acceptance is the swing, exactly as P5c said —
and the outside number for it is now 0.64. The 46–56 tok/s TODO.md quotes
from `halogen-flash-server` sits inside this table. **A plan is measured,
never composed** (P3): every row above is owed its run.

---

## 5. Caveats

The PR is open; its graph may still change (the thread's `is_empty()`
refactor does not touch the mechanisms above). Its numbers are CUDA on a
DGX Spark at an IQ4_XS trunk, a slower baseline than ours (28.4 against 41.4
tok/s), so its fixed costs are proportionally smaller — P5c §4.2's rule —
and the 1.55x is a ceiling for the same loop here, not a floor. The
acceptance figures are the transferable part: the same head on a trunk
quantised to a similar width, on a public prompt set we can run.

---

## 6. Measured, 2026-10-01 — P20a (the catch-up) and P20d's first half

`cmd/llm -spec`, 48 layers, the shipped banks with `LLM_BANK_CACHE`, `-n
256`, nothing else on the GPU. **long** is `-spec-prefix 1024 -ctx 2048`
(the first 1024 tokens of wikitext-2's test split, generated on from there);
**short** is the default 34-token prose prompt at `-ctx 512`.
`SPEC_CATCHUP=0` is P5c's loop and the control. CSVs
`results/p20a_{long,short}_cu{0,1}.csv` (one pass a run) and
`results/p20d_{long,short}_cu1.csv` (two interleaved passes).

**P20a — the catch-up.** `Speculator.Start` runs the prompt through
`Graph.ForwardResidual` (the layers, a copy of every row's residual, then
the mixer and head on the last row) and primes the draft at every prompt
position, `DraftRows` = 64 a pass, position p from the trunk's residual at
p−1 and the token at p (position 0 from a zero residual). Each committed
two-row pass then owes the draft exactly **one** cell — P+1, from row 0's
residual and row 1's token, because the draft step at P was already seeded
with the true pair — and the owed row rides in front of the next drafting
row in one `MTPHead.Rows` call. The draft never holds a cell past the
committed sequence at depth one, so a rejection needs nothing from it.

| arm | catch-up | drafted | accepted | a₁ | multiplier | draft ms a round |
|---|---|---:|---:|---:|---:|---:|
| long | off (P5c) | 85 | 58 | 68.2% | 1.00x | 5.56 |
| long | **on** | 85 | 77 | **90.6%** | 1.10x | 9.01 |
| short | off (P5c) | 128 | 85 | 66.4% | 1.00x | 5.27 |
| short | on | 128 | 85 | 66.4% | 0.94x | 7.07 |

§3.1 was right where it had room to be: at 1024 prompt cells the draft's
attention without them is 22 points down. At 34 the counts are identical —
the prompt is too short for its cells to move an argmax. **The long arm's
90.6% is an upper bound, not a workload number:** the text the trunk
continues is wikitext's first article (Robert Boulter), which it largely
recites; P20e's SPEED-Bench prompts are the number to quote. With `nextn`
on the host the extra row cost 2.2 ms a round and ate the gain on the short
arm.

**P20d, first half — `eh_proj` on the device and the draft as one
submit.** `MTPHead.eh` is `nextn.eh_proj` staged as the head block's plain
projection (`NewHeadGPU`, K = 5120, N = 2560, the int8 bank) over one A row a
(draft row, stream): the two RMS norms and the concatenation stay on the
host (2560-wide elementwise work), the projection writes [n·hc][2560] and
one move puts it in the draft layer's residual arena, which is the same
bytes as [n][hc·2560]. The whole step — `eh_proj`, the move, the layer, the
head mixer, the borrowed lm head — is recorded into one command buffer
(`MTPHead.record`/`flush`, the trunk's recorder) where it was fifteen
submit-and-wait round trips. `LLM_MTP_HOST_NEXTN=1` and
`LLM_MTP_UNRECORDED=1` are the controls; `TestMTPSeedDeviceIsTheHost` is the
gate (rms 6e-3 of the seed's rms, the int8 rounding of a 13.1 M-weight
matrix; a misplaced stream reads ~1.4).

| arm | plain tok/s | speculative tok/s | multiplier | a₁ | draft (of a step) | verify (of a step) |
|---|---:|---:|---:|---:|---:|---:|
| long | 41.70 (41.55 / 41.85) | **53.95** (53.52 / 54.38) | **1.29x** | 77/85 | 3.50 ms (0.146) | 1.270 |
| short | 42.00 (41.44 / 42.55) | **45.26** (44.54 / 45.98) | **1.08x** | 85/128 | 2.89 ms (0.121) | 1.265 |

The acceptance is the host path's **count for count** on both arms, which is
the gate the seed's tolerance cannot be. Priming is 81 ms at 1024 cells
(1.99 s on the host). A drafting round is now pass + 0.12–0.15 of a step,
and what is left of the draft is the borrowed lm head (2 ms, the step's
own) — the floor §3.4 priced at 0.13.

**Lossless, qualified.** Both arms reproduce themselves (plain0 = plain1,
spec0 = spec1, token for token — P5c finding 6's non-reproducing control
is gone), and the speculative text leaves the plain one at **one** token in
each arm (long at 72, short at 109) and is coherent after it. Every emitted
token is still an argmax of the trunk's own logits; what differs is that a
two-row pass rounds row 0 unlike a one-row pass (the decode schedule's
row-count sensitivity, the 2.2e-3 subtest of the P19 postscript), and a
near-tie went the other way. That is the same class of difference as
llama.cpp's batched verification; it is recorded rather than fixed.

**What bounds it now** — the chain line of the short arm: speculating 1.386
steps for 1.664 tokens, recovering 1.265 for one. The recovery round is
`1 − a₁` of the rounds and the whole of the remaining gap to the
partial-accept form: with P20c's snapshot planes a rejected round commits
row 0 and costs nothing more, so the short arm would read (1.265 + 0.121) /
1.664 = 0.83 steps a token, **~1.20x**, and depth beyond one becomes
possible at all. **Next is P20c**, then P20b's four-row pass.

## 7. Measured, 2026-10-01 — P20c (partial accept), at depth one

**The form built is simpler than §3.3's planes, and it is free in memory.**
Row 0 of a verification pass is the token the trunk already emitted, so it
is committed whatever the draft row behind it is. With
`Graph.SpeculateFirst` on, the pass folds row 0 alone into the slots it
**read** — the scan stores S after its first token back over the state it
loaded (each lane stores exactly the elements it loaded, so no race), and the
ring write (`llm_seq_hist.comp`'s ping-pong arm) also stores row 0 into
`SEQ_HIST_PREV` at its own position, the slot of the position `hist` back,
which row 0's convolution has already read and which no invocation carries
over. The whole pass still goes to the other slot. `Commit` takes the other
slot as before; **`Graph.KeepFirst`** keeps the read slot, which is now the
sequence at P+1, and puts back the pooled indexer blocks only the rejected
rows completed (`AttnGPU.RestoreBlocksAfter`). No third slot: residency is
P5c's 241 MB. The flag is `SEQ_KEEP_FIRST` (= `ldaLo`, zero in every push
block that does not set it, so every other path runs the kernels as they
were). A keep-first pass cannot be rewound (the read slot has row 0 in it),
and `Graph.arm` refuses one wider than either ring (row 0 would not be
stored) — so this is depth one and two; P20b's four-row pass needs the
ring's 3 rows to grow or real planes.

The loop (`Speculator.Partial`, `SPEC_PARTIAL=0` the control): a rejection
is `KeepFirst`, the next round drafts at P+1 from row 0's residual and the
token row 0 named, and the draft owes nothing (its cell at P was written this
round from the true pair). Every round drafts; there is no recovery round.

**Gate** `TestSpeculationKeepFirstIsTheSequence` (4 layers, 480-token
prefix, two- and three-row passes kept to row 0 between passes accepted
whole, 10 partial keeps in 32 tokens): `result_norm` **identical to the last
place** against the one-pass prompt; the control (the in-place store off in
both blocks) misses by rms 0.93; `Rewind` of a keep-first pass refused. The
P5c gate is unchanged and passes. **Not guarded:** a sabotage run with the
partial block restore removed also passes, because the next pass always
starts at P+1 and rewrites the block a rejected row completed before
anything reads it (and at 528 cells the selection may be the identity). The
restore is kept, as `Rewind` keeps its own, for correctness by construction.

`cmd/llm -spec`, same protocol as §6 (shipped banks, `-n 256`, two
interleaved passes, a same-process plain control), one process an arm,
results `results/p20c_{short,long}_p{0,1}.csv`:

| arm | P20a (`SPEC_PARTIAL=0`) | **P20c** | rounds | a₁ | draft a drafted round |
|---|---:|---:|---|---:|---:|
| short (34-token prose, ctx 512) | 42.01 → 45.17, 1.08x | 42.00 → **49.82, 1.19x** | 170 (128 drafted) → 148 (all) | 66.4 → 73.0% | 3.85 → 3.72 ms |
| long (1024 wikitext cells, ctx 2048) | 41.72 → 53.98, 1.29x | 41.57 → **54.65, 1.31x** | 92 (85) → 88 (all) | 90.6 → 92.0% | — |

Short is §6's prediction to the hundredth (**~1.20x**): a round is 1.29
steps of pass + 0.16 of draft for 1.73 tokens. Long barely moves because at
90% only 8 of 85 rounds were rejections to begin with. a₁ differs between the
arms only because the text does after the first near-tie (short leaves plain
at token 67 here, 109 under P20a — the same two-row rounding, §6); both arms
reproduce themselves token for token (plain0 = plain1, spec0 = spec1).

Unpriced cost: the scan's extra store is ~113 MB a pass; `dn.scan` read 0.47
ms a pass per output token here against 0.46 under P20a, inside noise. The
uniform `t == 0` branch inside the prefill scan loop was not timed.

(The summary's "a round is" line on the long arm includes Start's 1024-token
prefill in `Stats.Wall`, ~8.5 ms a round; tok/s is timed after Start and is
unaffected.)

**Next is P20b**: depth beyond one. The verify pass is now 1.27–1.30 steps at
two rows and the draft 0.16; with every round drafting, a third row is worth
it only if a₂ × (rows-3 cost) beats it — §3.2's arithmetic on the shipped
banks, after `cmd/llm -graph -tokens 1,2,3,4` prices the passes. The keep-
first form covers keeping row 0 of any pass up to three rows wide; keeping
rows 0–1 of three needs a second stored state (a plane), which is §3.3's
general form.

## 8. Measured, 2026-10-01 — P20b, depth two: correct, and not worth it here

**Built.** P20c's keep-first generalised to keep any prefix of a pass up to
three rows: `GraphOpts.SpecRows` stages that many slots of every carried
tensor (three at depth two, +~120 MB, still only under `Speculative`); a
pass reads slot c, stores row 0 back into c, the state/ring after row 1 into
a *mid* slot, and the whole pass into the destination (`SEQ_KEEP_FIRST` = 1
+ mids, the mid offsets on `gateOff`/`bOff`; the ring write has its own
keep-prefix arm). `Graph.Keep(k)` picks c, the mid slot or the destination.
`Speculator.Depth` (`cmd/llm -spec-depth N`) chains the draft on its own
residual, verifies `[x_P, d₁, d₂]` in one 3-row pass, keeps the agreeing
prefix, and owes the draft the kept rows' cells (each from the trunk residual
of the row before it). `SPEC_TRACE=<path>` writes each round of the last pass.

**Gates.** `TestSpeculationKeepFirstIsTheSequence` now runs SpecRows 3 with
keeps of 1/2 of two rows and 1/2/3 of three: bit-exact; a sabotage keeping
the read slot where the mid slot is due fails on every value. At 48 layers
the depth-2 loop is lossless as before (self-reproducing; it leaves plain at
the same near-tie, token 67 short). The trace settles the a₁ question below:
the depth-1 and depth-2 sequences agree to position 242, and at all **78**
rounds that start at a common (position, pending token) the first draft is
identical — the draft's cache is the same at both depths.

**Measured** (`results/p20b_{short,long}_d{1,2}.csv`, shipped banks, two
passes, same-process plain control):

| arm | depth 1 | depth 2 | 3-row pass | draft a round | tokens a round |
|---|---:|---:|---:|---:|---:|
| short | 41.88 → 49.64, **1.19x** | 41.78 → 46.45, 1.11x | 1.29 → 1.52 steps | 0.16 → 0.31 | 1.73 → 2.05 |
| long | 41.63 → 54.43, **1.31x** | 41.54 → 54.49, 1.31x | 1.30 → 1.53 | 0.16 → 0.31 | 1.93 → 2.43 |

a₂ given a₁ is **62.5%** short (50/80). The per-round a₁ falls with depth
(73 → 64% short) for a selection reason, not a defect: a depth-2 round starts
after up to three accepted tokens, so its rounds sit at different positions.

**Why it loses: the third row is MoE bytes.** Per pass, `moe.up` is 5.0 /
8.9 / 12.6 ms at one / two / three rows — 1.78x and 2.5x for 17.3 and ~24
distinct experts against 10 — and `moe.down` 2.4 / 3.6 / 4.6. The 3-row
pass's +0.23 step over two rows is the experts' weight, which no kernel
recovers, and the second draft step is another full 0.15 (its lm head is
2 ms of 3.7). Break-even for depth 2 is tokens a round ≥ 1.73 × 1.83 / 1.45
= 2.19 short (got 2.05) — about a₁a₂ ≥ 0.46 — which the long arm's
memorised text reaches (a tie) and the short prose does not. llama.cpp's
1.55x at depth 3 was on a DGX Spark whose MoE pays less for a wider pass;
this machine's pass cost is the expert set.

**So depth one stays the default.** Depth two becomes worth re-measuring
only if the draft gets cheaper or acceptance rises (P20e's workload number).
The lever for both depths is now the draft's **lm head** — 2 ms of a 3.7 ms
draft step over the full 248 320-token vocabulary: a draft head over the
most frequent ~32k tokens (FR-Spec's trimmed vocabulary; a draft outside it
is simply never proposed, so the loop stays lossless) would take the draft
from 0.16 to ~0.08 of a step — short 1.20 → ~1.27x on paper, and depth 2's
break-even down to a₁a₂ ≈ 0.40. Next: price that on the observer
(`-mtp`, acceptance with the vocabulary cut) before building it.

## 9. Measured, 2026-10-01 — the draft's lm head over a vocabulary prefix

After P20c the draft step was 3.7 ms a round, 2.07 ms of it the borrowed lm
head's GEMV over all 248 320 ids. FR-Spec's idea: the draft only has to
*propose*, so it may propose from the frequent tokens alone; the trunk still
decides every token and the loop stays lossless.

**Which tokens.** `cmd/vocabfreq` (coverage of held-out text by the top K
under three rankings): counts from wikitext-2 train are useless on code (49%
of Go source at 32k), and **BPE id order** — merge order, a frequency ranking
on the tokenizer's own broad corpus — needs no data and covers both: 90.8 /
92.5% (wikitext test / Go) at 32k, **96.6 / 98.3% at 64k**. With id order the
subset is a prefix, which is what makes the device side free:

**The device side is a smaller grid.** `output.weight`'s bank is tiled
sixteen output rows a block, contiguous, so the first K rows are the first
K/16 workgroups' bytes. `HeadGPU.RunCols(K)` dispatches the trunk's own
head GEMV with `GroupsY = K/16` and the push constants unchanged (the K-quant
planes are addressed from the full N): no new weights, no new kernel.
`MTPHead.SetVocab`, `Speculator.DraftVocab`, `llm.DraftVocabDefault` =
65 536; `SPEC_DRAFT_VOCAB=K` (0 = whole vocabulary, the control),
`SPEC_DRAFT_VOCAB_HOST=1` cuts on the host instead. Gate `TestHeadGPURunCols`
(q8, q4_k, q5_k): the prefix bit-identical to the full run's, the columns past
K untouched.

**Priced on the loop first** (host cut, acceptance only, one pass):

| K | short a₁ | long a₁ |
|---:|---:|---:|
| all | 108/148, 73.0% | 81/88, 92.0% |
| 65 536 | **108/148, 73.0%** | 78/91, 85.7% |
| 32 768 | 106/149, 71.1% | 74/95, 77.9% |
| 16 384 | 100/155, 64.5% | 73/96, 76.0% |

**Measured** (device cut, two passes, same-process plain,
`results/p20v_{short,long}_{full,65536}.csv`): the acceptance is the host
cut's count for count; the draft is **3.71 → 2.22 ms (0.155 → 0.093 of a
step)**.

| arm | whole vocabulary | **65 536** |
|---|---:|---:|
| short | 41.84 → 49.76, 1.19x | 41.91 → **51.93, 1.24x** |
| long | 41.55 → 54.51, 1.31x | 41.65 → **55.14, 1.32x** |

The long arm gives back most of its saving in acceptance (wikitext's proper
nouns sit past id 65 536) and still nets +0.6 tok/s; prose keeps all of it.
What is left of the draft is its one layer (~1.7 ms). Depth two re-priced on
these costs: a 1.52-step pass + 2 × 0.093 = 1.71 steps for ~2.05 tokens
(short) is ~1.20x against depth one's 1.25x — still not adopted.

## 10. Measured, 2026-10-01 — P20e, SPEED-Bench: 1.48x at depth two

**The set.** `nvidia/SPEED-Bench`'s `qualitative` split
(`models/speed-bench/qualitative.parquet`) is 880 prompts in 11 categories,
but most rows ship as a placeholder ("FULL BENCHMARK DATA SHOULD BE FETCHED
FROM THE SOURCE USING SPECDEC_BENCH") for licensed sources; complete
single-turn rows exist only for coding, multilingual, qa, rag and writing —
which is why the PR's 24 samples are those categories plus a few roleplay
and humanities rows. `models/speed-bench/qualitative-20.jsonl` is the first
four complete single-turn rows of each of the five, file order, first turn
only: 18–1586 prompt tokens through `tokenizer.ChatPrompt`, temperature
zero, **1024 output tokens** (the PR's), two passes, plain and speculative
interleaved per prompt, shipped banks, the 64k draft vocabulary.
`cmd/llm -spec -spec-set <jsonl>` (`cmd/llm/specset.go`); `SPEC_SET_TEXT`
writes the texts. Contexts reach 2614 cells, so the 2051-cell refusal in
`NewMTPHead` came out first: the PR's converter writes `blk.48` the
trunk's compress ratio, which is the value borrowed here.

**The texts are not P5a's loop artefact**: every one is coherent, most end
on `<|im_end|>`, and at most 2.7% of a text's 40-character windows repeat an
earlier one. Plain reproduces itself on 20/20 prompts at both depths, spec
on 20/20; spec is the plain text on 10 and leaves it at a near-tie on the
rest (the two-row rounding of §6).

| category | plain | depth 1 | a₁ | **depth 2** | tok/round |
|---|---:|---:|---:|---:|---:|
| coding | 41.4 | 57.5, 1.39x | 93.4% | **65.3, 1.58x** | 2.73 |
| multilingual | 41.6 | 52.7, 1.27x | 75.6% | 56.1, 1.35x | 2.31 |
| qa | 41.6 | 55.1, 1.33x | 83.7% | 59.5, 1.43x | 2.46 |
| rag | 41.2 | 55.6, 1.35x | 87.7% | 62.7, 1.52x | 2.64 |
| writing | 41.1 | 54.8, 1.33x | 84.9% | 60.6, 1.47x | 2.53 |
| **overall** | 41.3 | **55.4, 1.34x** | 86.2% | **61.3, 1.48x** | 2.56 |

(`results/p20e_speedbench_d{1,2}.csv`, per prompt and pass.) a₂ given a₁ is
~0.84 here (2.56 = 1 + 0.85 + 0.85·a₂), against 0.625 on the short prose
prompt — which is why §8 and §9 priced depth two dead and this workload
does not: at depth two the round is 1.52 + 2 × 0.09 = 1.70 steps, and
break-even is a₁a₂ ≈ 0.40. On the prose prompt depth two now reads 1.20x
against depth one's 1.24x (`results/p20e_short_d2_v64k.csv`), the one place
it gives anything back. **`llm.SpecDepthDefault` = 2**, `-spec-depth 1` the
control.

Against the PR: 1.48x at depth two on an AMD iGPU against 1.55x at depth
three on a DGX Spark, with a first-draft acceptance (86%) well above its
per-token 0.64 average over three positions. **Depth three** is priced at
~+5% more (≈3.15 tokens for a ~1.75-step four-row pass + 0.28 of draft) and
needs a four-row pass: `GEMVMaxRows` 4 in the decode GEMVs and the fused
router, and a keep-prefix form past the DeltaNet ring's three rows (a ring
of hist + R − 1 slots, or a stored plane per row). The larger lever is
serving the loop at all — the server does not speculate yet, and its
sequence slots share the carried state's slot index with speculation.

## 11. Measured, 2026-10-01 — P20f, the loop served

**What the server needed.** `-llm-draft <mtp gguf>` (`LLMOptions.Draft`,
`-llm-spec-depth`, default `llm.SpecDepthDefault`):

- **Sequence slots and speculation planes compose.** They shared the carried
  state's slot index and refused each other (`GraphOpts.Slots` vs
  `Speculative`). The DeltaNet and PLE blocks now index (sequence, plane):
  sequence s owns slots `[s·P, (s+1)·P)`, each sequence remembers its
  committed plane across `UseSlot`, and a batched row (C5) reads its
  sequence's committed plane, not its first. Gate
  `TestSpeculationSlotsAreSequences` (keep-prefix rounds on slot 0, slot 1
  stepping between, batched passes of both, bit-exact; sabotaging the batched
  plane lookup fails it, rms 0.51). `Commit`/`Keep` drop the live slot's
  pre-recorded step, which names the old plane.
- **A draft cache a slot** (`NewMTPHeadSlots`, `MTPHead.UseSlot`).
- **The loop in pieces** (`NewSlotSpeculator`; `Prime`, `Begin`, `Round`,
  `Owe`, `Settle`, `Rewind`): a prefill chunk runs `Graph.ExtendResidual`
  and primes the draft from its residual; a round draws each verified row's
  token with a `pick` function — **the request's own sampler, in order**, so
  the loop is lossless at any temperature (row i's token is drawn over a
  prefix that is the sequence), and it stops at end-of-generation or the
  budget so the trunk never runs past what the plain loop would; a plain step
  the slot takes instead (batched with other conversations) is **owed** to
  its draft (`Graph.ResidualRow`), so the draft never falls behind; a
  checkpoint keeps the residual it resumes from. Gate
  `TestServedSpeculationIsPlain` (two slots, chunked priming, rounds
  alternating with batched steps, a second turn; token for token).
- **The scheduler** (`backend/llm_sched.go`): a decode is submitted as a
  *round*; it speculates when its conversation is the only one of its class
  decoding and its slot's draft is caught up (`llmSlot.track`), and otherwise
  runs as a plain step that batches (C5) — three conversations' rows are
  worth more than one conversation's speculation. Gate
  `TestLLMSpeculationIsPlain` (server with and without the draft: greedy,
  seeded sampling, two turns, two at once then a second turn, a checkpoint
  restore — identical streams; also on the shipped banks).

**Served, 48 layers, shipped banks, SPEED-Bench's 20 (§10's set) through
`/v1/chat/completions`, 512 tokens, ctx 32768, 3 slots, one client**
(`results/p20f_served.csv`; decode rates from the server's own line):

| | plain | speculative | |
|---|---:|---:|---:|
| greedy | 41.06 | **56.83** (2.38 tok/round) | **1.38x** |
| greedy, again | 41.06 | 56.62 | |
| default sampling (temp 1, top-k 20, top-p 0.95) | 37.93 | 46.22 (2.08) | 1.22x |
| — after the top-k fix below | **41.06** | **50.08** | **1.22x** |

by category, greedy: coding 1.45x, writing 1.41x, rag 1.38x, qa 1.37x,
multilingual 1.30x. Prefill pays **+9%** time for priming the draft (8.49 →
9.28 s over the 6960 prompt tokens): the residual comes to the host and goes
back a 64-row draft arena at a time.

**Against `cmd/llm -spec-set` at the same 512 tokens: 1.48x, 2.53
tok/round.** The served loop costs the same per round; it accepts less,
because the server's template opens a thinking block and terse reasoning
drafts worse (2.38 against 2.53 tokens a round is 56.7 against 60.3 tok/s).
Sampling lowers acceptance again (2.08) — the draft proposes the argmax, the
sampler draws from the top 20.

**Lossless, and the near-ties measured.** Each arm reproduces itself 20/20
(plain A/B, spec A/B). Spec is plain's greedy text on 4 of 20 served and 12
of 20 in `cmd/llm`. `SPEC_MARGIN=1` now prints the plain loop's top-2 margin
where spec first leaves it: **0.003–0.033 logits, against medians of
2.7–11** on all eight (`results/p20f_spec_set_margins.txt`) — the shipped
K-quant banks' multi-row GEMVs round unlike one-row steps (P19's 2.2e-3 rms),
and a near-tie goes the other way. §10 asserted this; it is measured now. On
the Q8 test banks every gate is bit-exact.

**The sampler cost 2 ms a token** (plain sampled 37.93 against greedy 41.06):
`Sample` built a 1 MB index array and quickselected all 248 320 logits every
token, and in a round the scheduler samples every verified row in turn.
`topKOf` keeps the k largest in one pass (0.115 ms at k = 20;
`TestTopKOfIsTheSort` against a stable sort, ties included): **plain sampled
37.93 → 41.06, speculative sampled 46.22 → 50.08 tok/s**, the same texts 20/20.

**Memory.** The draft is 1.96 GB of weights; at ctx 32768 × 3 slots its
arenas and caches are 0.70 GB. At the deployed 262 144 × 3 the caches grow
eightfold (~2.3 GB) and the planes add 2 × 120.75 MB a slot: **~+5 GB on the
LLM line**, and the host keeps a residual buffer of a prefill chunk (335 MB at
`-llm-batch 8192`).

**Next.** Priming on the device (the +9% prefill; the residual is already in
an arena the draft could read, as §3.1 said); depth 3 (§10, ~+5%, a four-row
pass); and an acceptance lever on thinking text. `DecodeRows` ignores
`Graph.PinSchedule` — a pinned batched row rounds unlike a pinned solo step —
which only a test notices (`TestServedSpeculationIsPlain` runs unpinned).
