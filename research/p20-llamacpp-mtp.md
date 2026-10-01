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
