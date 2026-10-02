# C4 — checkpoints (done 2026-09-23)

*The C4 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


`llm.Graph.Checkpoint` / `Restore`, and a checkpoint per slot in the
scheduler.

- **What a checkpoint is.** It holds the live slot's DeltaNet state and rings
  for every layer (~120 MB at 36 layers), the PLE ring, the position and the
  ids, copied on the host. The arenas are host-cached and mapped, so this is
  a memcpy, not a readback. The KV cells and raw indexer keys below the
  position stay on the device, because a continuation from there never
  rewrites them. So a checkpoint restores only into its own slot, and only
  while that slot's ids still start with the checkpoint's. `Restore` checks
  both and refuses otherwise.
- **The pooled indexer table needs no restore**, although P5c's rollback
  snapshots it. The first version restored it, and its control did not
  move. The reason is in `llm_attn_score.comp`: a pass rebuilds the blocks it
  completes before it scores any. Every block at or past the last whole one
  is scored as one broadcast value under a −inf or 1e9 bias (P7), and no
  stored row can change that selection. So the restore was dropped, and the
  gate keeps the case that could have disproved this (below).
- **Where the checkpoint goes.** It sits at the end of everything before the
  last `<|im_start|>user` (`backend.checkpointAt`). That point is found on
  the rendered text and re-encoded, and refused if the result is not a token
  prefix of the prompt. A prefix under 256 tokens is not checkpointed. The
  scheduler cuts the prefill chunk at the boundary and checkpoints after it,
  reusing the slot's host buffers. Slot choice counts a restorable checkpoint
  as reuse, and among slots that save a job nothing it overwrites the one
  with the least checkpointed first. Without that, agents arriving after a
  voice command take its slot by LRU. `-llm-checkpoints=false` is the
  control. The log says `(N from a checkpoint)` where it said `(N cached)`.

**Gates:**
- `TestGraphCheckpointIsThePrefill` (4 layers, 7 s): a 2600-token prefix in
  a 4096-cell cache (sparse: the selection is past its width), checkpointed
  in slot 1. The slot then runs a 600-token detour and 12 steps, slot 0 runs
  another conversation, and slot 1 restores and runs a 300-token tail and 12
  steps. **All 13 logit rows are bit-identical** to prefix + tail run
  directly. The detour runs further than the real continuation, so stale
  pooled blocks are present at the decode frontier, and they still change
  nothing. The stale *values* past the frontier did change something, one
  ulp at one row in twelve, until the pack started clearing them ("Stale
  values past the frontier", below). Controls: a restore without the
  DeltaNet state moves row 0 (logit 0 −0.8748 against −0.8638), and one
  without the PLE ring moves row 0 too (−0.8640). `Restore` refuses a slot that has since been re-run from token
  zero.
- `TestLLMCheckpointIsTheSystemPrompt` (backend, 4 layers): a voice-shaped
  request restores the system prompt's checkpoint (566 of 585 tokens,
  prefill 148 → 24 ms) and streams exactly the text of a fresh prefill. The
  scheduler's restore counter is asserted, so a run that never restored
  cannot pass.

**At 48 layers** (C0's harness, 3 slots, same hour, two runs agreeing within
0.02 s):

| | voice alone | voice during agent prefill (2 s / 9 s) | during agent decode (25 s / 40 s) | agents done | aggregate |
|---|---|---|---|---|---|
| chunk 2048, checkpoints off | 1.83 s | 3.41 / 2.44 | 1.80 / 1.79 | 66.3 s | 24.1 tok/s |
| **chunk 2048, checkpoints on** | **0.19 s** | 1.82 / 1.11 | **0.22 / 0.20** | 59.9 s | 26.7 |
| chunk 1024, checkpoints on | 0.19 s | **0.50 / 1.14** | 0.22 / 0.21 | 63.5 s | 25.2 |

- **A voice command's TTFT goes from 1.83 s to 0.19 s**: 1917 of its ~1940
  tokens come from the checkpoint, and what is left is a ~22-token pass plus
  the restore. The first command after a start pays **+0.2 s once** (2.03 s):
  the checkpoint copy and the extra chunk the cut makes.
- Voice turns are cheaper, so the agents finish 6.4 s sooner and the aggregate
  rises 11%.
- **The chunk wait is now nearly all of a voice command's latency while an
  agent prefills.** At 2048 the wait is up to ~2.0 s, plus ~0.2 s. At 1024
  it is up to ~1.3 s, and the price is the +25% background prefill C3
  measured (agents 3.6 s later here, aggregate −6%). **Decided: 2048**:
  prefill speed over a second of voice latency.

**Not built: cross-slot restore.** A checkpoint restores only into its own
slot. Copying the KV cells too (27.8 KB a cell, ~53 MB for a 1.9k system
prompt) would let a voice command use a checkpoint while another voice
command holds its slot. It is not needed while one slot is reserved for the
interactive class and commands arrive one at a time.
