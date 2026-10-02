# C5 — batched decode (done 2026-09-23)

*The C5 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


`llm.Graph.DecodeRows(slots, ids)` advances several sequences one token each
in one pass. The scheduler batches the pending decode steps of one class into
it.

**How, and why not row-aware kernels.** The plan was to make attention,
DeltaNet and PLE read a per-row slot and position from an arena table. The
build runs **those three blocks' per-sequence kernels once per row instead**,
as a one-token pass of that row's sequence: its slot's cache, state and
ring, its row of the projection in and its row of the context out. The
projections, the hyper-connections, the MoE and the head run over all R rows
at once. Three things make this nearly free:

- **The position was the only thing a push field had no room for.**
  `SEQ_PAST` is now `actu[pc.lowRank]`: dword 1+r of the block's arena for
  row r, and dword 0 for every ordinary dispatch, whose `lowRank` was
  already zero. None of the thirteen kernels that read the position uses
  `lowRank` as itself. So a single-sequence pass and its recorded replay are
  the same bytes as before. `TestGraphLogits` reproduces P16's diagnostics
  to the digit (maxAbs 1.987e+00 at 19208, rms 3.314e-01).
- **The recorded pass fences every dispatch**, so every scratch arena
  between the projection and the context is reused row after row. A row's
  pass may write the context's pad rows past itself, and the next row's pass
  overwrites them, because rows run in ascending order.
- **Per-row cost is small at decode**: attention and DeltaNet add ~2.4 ms a
  row at 48 layers (the batch profile below). Row-aware kernels would win
  back part of that, and it is the smaller lever.

**`moe.up` computed padding rows.** The GEMV rung computes `ROWS` rows of
every expert tile, and the permutation's sentinel makes the extra ones
harmless. At P5b's two rows of one sequence that was nearly free, because
most tiles had two real rows. With three conversations most of the ~27
experts serve one row, so two-thirds of `moe.up`'s arithmetic was padding.
The tile record's third word (the block size, which nothing read) is now the
real row count (`llm_moe_perm.comp`, and on the host for the shared expert's
list), and `llm_moe_gemv.comp` stops at it. `moe.up` at three rows went from
**28.9 to 18.8 ms**, and a three-row pass from **64.2 to 53.4 ms**. The
shared expert reads ~0.25 ms slower at three rows in both runs, which is not
explained.

**The instrument is `cmd/llm -batch 1,2,3 -batch-depth 7000`.** It uses one
slot a row, each prefilled with its own wikitext, and decodes greedily. A
fixed token per row would read the same experts every step and understate
the MoE; that was C5a's error in another form. `-batch-chunk 8192 -batch-ctx
65536` gives the server's shape, which costs the same to 0.6 ms.

| rows | pass | a row | aggregate | moe | dn | attn | distinct experts (layer 47) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.2 ms | 34.2 tok/s | 34.2 | 11.1 | 6.8 | 3.4 | 10.0 |
| 2 | 42.9 (1.47x) | 23.3 | 46.6 | 20.5 | 7.8 | 4.8 | 19.2 |
| 3 | **53.4 (1.83x)** | 18.7 | **56.2** | 28.7 | 8.9 | 6.1 | 26.8 |

**Gates:**
- `TestGraphDecodeRowsIsEachSlot` (4 layers, 5 s): three sequences of 700,
  300 and 500 tokens decode 8 steps batched. Every row's logits are
  **bit-identical** to that sequence stepped alone, at three rows and at two
  (slots 0 and 2). There is nothing to reassociate: the R-row GEMV sums each
  row in the one-row order, and the per-sequence kernels run a row as a
  one-token pass. The three controls each make row 1 read row 0's position
  in one block, and each moves the logits: attention 0.169 rms, DeltaNet
  0.127, PLE 0.055.
- `GEMVMaxRows` 3 with its rungs: `TestGraphIsAChunkSplit` gained the
  unpinned "2 / 3 at a time, on the decode schedule" subtests (C5a's).
  `TestDeltaNetGPUStateCarries` now pins the GEMM, because its 3 + 4 split
  made the first half a GEMV. `TestMoEGPUDecode` and `TestMoEGPUSharedPlan`
  now stage one past the bound. They staged a literal two, so their
  "refused past the bound" control was passing on the staging's refusal and
  not the plan's.
- `TestLLMConcurrentIsSolo` (backend) now asserts that batching happened:
  23 batched passes of 2.91 rows each, and every conversation streams
  exactly its solo text.

**At 48 layers** (C0's harness, same hour, two runs agreeing within 0.7
tok/s):

| | agents' decode, each | agents done | aggregate | voice |
|---|---|---|---|---|
| 3 agents, batching off | 11.0-11.5 tok/s | 88.0 s | 26.2 tok/s | — |
| **3 agents, batching on** | **17.6-18.8** | **62.0 s** | **37.1** | — |
| C0's mix, batching off | 16.9 / 16.4 | 61.1 s | 26.2 | as C4 |
| **C0's mix, batching on** | **23.2 / 22.3** | **48.6 s** | **32.9** | as C4 (0.20 s in decode) |

- The loop's gap between batched steps is **0.3 ms** (an env-gated trace in
  a scratch build). The coalescing wait (up to 5 ms for peers that are still
  sampling) costs nothing measurable, and without it the first conversation
  back from sampling would run alone.
- **Background rows still do not ride in an interactive pass** (rule 4).
  Per the table, one rider would cost a voice stream 1.47x and two 1.83x.
  That stays off: a voice reply is 10-30 tokens, and its latency is its TTFT.
- `-llm-batch-decode=false` is the control.
- **Solo decode is 33.8 tok/s this afternoon against 34.7 this morning, and
  the same-hour A/B/C says that is not this work.** The one-row pass is 29.0
  ms before today's changes, 29.3 with `MAXROWS` 3 and 29.5 now, inside the
  0.5 ms spread.
