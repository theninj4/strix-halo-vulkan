# CONCURRENCY — more than one completion at a time

> **Moved to `research/` on 2026-10-02** — this was `CONCURRENCY.md` at the repo
> root, and code comments citing `CONCURRENCY.md` resolve here. C0–C6 are done and
> C7 is conditional; § Next is still where a session resumes. Each stage's
> write-up is broken out into a `c*.md` file beside this one, linked where its
> section was; C-stages still resolve here.

> **Opened 2026-09-23.** This is the live file for P6 ("batching"), which
> `TODO.md` and `research/llm-vertical.md` carried as *blocked on the product
> question: will the API serve more than one stream?* **The answer is yes**:
> **three concurrent completions**, as fair as the hardware allows, with a
> **priority lane** for a latency-bound caller (a voice command) over
> callers that can wait a few seconds (background agents). Like the other live files, this one is rewritten rather than appended to; a closed stage's write-up
> goes to `research/` when it is long.

## Where it started (before C2)

`backend.LLM` held **one `llm.Graph`, one sequence and one mutex for the
whole request** (`backend/llm.go`, `Complete`). A second request waits for
the first to finish generating, and the log line says so (`(Ns queued)`). The
device lock is taken per forward pass, but the graph lock is held per request,
because the graph's carried state *is* the conversation:

| per-sequence state | where | size |
|---|---|---|
| KV cache: K, V, indexer raw + pooled keys | `AttnGPU.kbuf/vbuf/ibuf`, 12 layers | **27.8 KB a cell**: 7.25 GB at 262 144 cells |
| DeltaNet recurrent state | `DeltaNetGPU.aState`, 36 layers | 113.25 MB |
| DeltaNet convolution rings | `DeltaNetGPU.aWin` | 7.13 MB |
| PLE convolution ring | `PLEGPU.aHist` | small |
| position | `SEQ_PAST`, dword 0 of three blocks' arenas | 4 bytes |
| token ids (n-gram gather) | `Graph.ids` | host |
| the recorded decode step (P1c) | `Graph.pre` | its push constants hold the offsets above |

Everything else in the graph is **per pass**, not per sequence: the hyper-connection
residual, the MoE, the head and every activation arena are rewritten from
scratch by each pass. So a second sequence needs a second copy of the table
above and nothing else.

Prefix reuse was **one sequence deep and all-or-nothing** (and within a slot it still is). A
DeltaNet state has no inverse, so a conversation that diverges from the held
one anywhere before its end is prefilled from token zero. That already costs
the voice case today: a Home Assistant command is the same long system prompt
plus a new utterance. The held sequence is the *previous* utterance and its
answer, so every command re-reads the system prompt.

### The numbers this plan is priced against

- **Decode: 34.2 tok/s through the server** (~29 ms a step; 35.6 since the
  MoE GEMV's quad loads, below), and the step is
  *latency, not bandwidth*: single-wave workgroups that all fit at once (P8).
  That is why rows are cheap. **A two-row pass costs 1.32 steps at 48 layers**
  (P5c), and the MoE is the part that keeps growing (10 experts touched at
  M = 1, 17 at 2, 36 at 4).
- **Prefill: ~1200 tok/s**, and a unit of work is a whole `-llm-batch` chunk.
  At the served 8192 that is **~6-7 s that nothing else can interrupt**.
- **Memory:** the served 262k configuration stages at ~83 GB and leaves ~33 GB.
  Two more full-context sequences would need 14.5 GB of KV plus 0.24 GB of
  DeltaNet state. That fits, with ~18 GB to spare.
- **`maxStorageBufferRange` is 4 GiB - 4**, and a K or V plane is 12 KB a cell,
  so **one plane buffer holds ~349k cells in total**. If slots share a buffer
  by offset, `slots × ctx ≤ ~349k` (for example 3 × 112k). Three full 262k
  contexts need a buffer per slot, bound as a descriptor array (C6), like the
  MoE bank's.

## The design, in one paragraph

**Slots, then a scheduler, then batching.** The graph gets `S` *sequence
slots*: `S` copies of the per-sequence state above, one live at a time,
switched by `Graph.UseSlot` (C1). The backend replaces its mutex with a
**scheduler that owns the graph** and interleaves *work units*: a decode step
or a prefill chunk, each on some slot. It picks the next unit by priority
class and then round-robin (C2). This is concurrency without extra
throughput: three streams share ~34 tok/s. What it buys is that nobody waits
for someone else's whole generation, and a voice command waits at most one
work unit. Making that unit short is C3. **Batching (C5)** then runs the
decoding slots' next tokens as **R rows of one pass**. Every block but
attention, DeltaNet and PLE is already row-independent, and P5b built the
R-row decode GEMVs. It is the step that adds throughput: if three rows cost
~1.5 steps, three streams get ~23 tok/s each instead of ~11.
**Checkpoints (C4)** make prefix caching work across more than one live
conversation, and between the turns of a voice session.

## Priority

Two classes, **interactive** and **background**. The API names the class:

- `service_tier` on the request body (OpenAI's field on chat completions and
  responses): `"priority"` is interactive; `"flex"`, `"default"` or absent
  are background, unless the server's default says otherwise.
- An `X-Priority: interactive|background` header, for a client that cannot
  add a body field (Home Assistant's OpenAI integration is the likely one).
- A server flag for the default class.

Scheduling rules (C2, tuned in C3):

1. An interactive unit always runs before a background one. Preemption
   happens at a **unit boundary**, never mid-pass. The activation arenas are
   shared, so a pass cannot be paused.
2. Within a class, round-robin **by GPU time**, not by unit count (deficit
   round-robin). A prefill chunk is worth ~200 decode steps.
3. Background prefill is cut into `-llm-preempt-chunk` pieces whenever there
   is more than one slot. It cannot wait until an interactive request is live,
   because by then the chunk it waits behind has already started. So the worst
   wait an interactive request sees is one short chunk rather than 6-7 s, and
   what that costs background prefill is C3's measurement.
4. Decode steps of one class share a pass (C5, done). Background rows do
   **not** ride in an interactive pass: one rider would cost the voice
   stream 1.47x and two 1.83x, and a voice reply's latency is its TTFT.
5. At most `S` requests hold slots. A fourth waits for one, and it is logged
   as queued, as today.

## Stages

Each stage has a gate that can fail, and the gate lands in the same commit as
the change.

| # | stage | what it is | gate |
|---|---|---|---|
| ~~**C0**~~ | ~~**Instrument**~~ **done 2026-09-23** (below): `cmd/loadgen` | A load generator (`cmd/bench -llm-concurrent` or similar) that fires one voice-shaped request (a fixed ~2k-token system prompt + a short utterance, short answer) against two agent-shaped ones (long prompts, long answers). It records per-stream TTFT, token rate and total throughput. Run it against today's server for the baseline. | Numbers in this file, two runs agreeing. |
| ~~**C1**~~ | ~~**Sequence slots in the graph**~~ **done 2026-09-23** (below) | `GraphOpts.Slots`, `Graph.UseSlot(s)`. Attention allocates `S×` the cache and offsets it by slot; DeltaNet and PLE allocate `S` state/ring slots, and the *committed slot* is the sequence slot. The graph keeps `past`, `ids` and the recorded decode step per slot. Refused together with `Speculative` (the ping-pong and the slots share an index). | `TestGraphSlotsAreSequences`: two sequences interleaved token by token through one graph give **bit-identical** logits to each run alone, prerecorded decode included. Its control (the same interleave without switching slots) must differ. |
| ~~**C2**~~ | ~~**The scheduler**~~ **done 2026-09-23** (below); `-llm-slots` stays 1 in `ai.service` until C0/C3 are measured | `backend.LLM` gets a scheduler goroutine that owns the graph. `Complete` submits work and receives tokens. Slot choice: longest reusable held prefix, else the least recently used idle slot. `held` becomes per slot. Priority from `service_tier`/header/flag. `-llm-slots N` (default 1 until C3 lands). | An `api`-level test with three concurrent requests on the 4-layer prefix: each one's output equals its solo run at temperature 0. The interactive one's TTFT is bounded by one unit, not by the other requests' generations. |
| ~~**C3**~~ | ~~**Bounded work units**~~ **measured 2026-09-23** (below); chunk stays 2048 until C4 | Price prefill against chunk size at depth (512 / 1024 / 2048 / 4096 / 8192). While an interactive request is live, cap background chunks at the chosen size. Measure the switch cost between slots (per-slot recorded decode steps should make it ~0). | C0's harness: interactive TTFT and token rate with two background streams live, against the same request alone. |
| ~~**C4**~~ | ~~**Checkpoints: prefix caching across threads**~~ **done 2026-09-23** within a slot (below); cross-slot restore not built | Snapshot a slot's carried state at a **stable boundary**: the end of everything before the last user message. That is DeltaNet state + rings (~120 MB, ~1 ms on-device), the PLE ring, the pooled indexer block that straddles the boundary, and the position. KV cells below the boundary are never rewritten by a continuation, so within a slot they need no copy. A request whose prompt shares the checkpointed prefix restores it and prefills only the tail. That fixes the voice case (same system prompt, new utterance) and agents that retry or branch. Cross-slot restore copies the KV cells too (27.8 KB a cell, ~0.11 GB for 4k). | Restore-then-continue is **bit-identical** to a fresh prefill of the whole prompt, and a control that skips the DeltaNet restore differs. |
| ~~**C5**~~ | ~~**Batched decode**~~ **done 2026-09-23** (below): per-row passes rather than row-aware kernels, and `moe.up` skips padding rows | R decoding slots advance one token each in one pass. Per-row position and slot come from an arena table (P1c's trick again: the recorded buffer stays byte-identical). The row-aware kernels are attention (`pack`, `idx`, `score`, `select`, the gathered `wmma`), DeltaNet (`conv`, `scan`, `seq_hist`) and PLE's `hist`. Raise `GEMVMaxRows` 2 → 3 with its rung in `TestGraphIsAChunkSplit` in the same commit (see the memory note on new row counts). **C5a prices it first**: a 3-row pass at 48 layers against a 1-row one, which needs the machine (stop `ai.service`). | Each row of a batched pass equals that slot's solo step (to the R-row GEMV's documented tolerance), whole graph, at R = 2 and 3. |
| ~~**C6**~~ | ~~**Full context in every slot**~~ **done 2026-09-23** (below): a buffer set and a pipeline set a slot, no descriptor array | K, V and indexer planes as a descriptor array of per-slot buffers (`PipelineSpec.Counts`, as the MoE bank does), which lifts the `slots × ctx ≤ ~349k` cap to 3 × 262k (~22 GB of KV). | `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` extended across slots; P18's recall at 237k in a non-zero slot. |
| C7 | *Mixed passes* (maybe) | Decode rows riding in another slot's prefill chunk, so a background prefill does not stall the other streams' decode. Only worth building if C3's numbers leave background decode visibly starved. | — |

Not planned: a second Vulkan queue or global-priority queues to preempt a
pass mid-flight. The arenas are shared, so preemption would need a second set
of them, and C3's short chunks bound the same latency far more cheaply.

## Open questions for later stages

- **How long is Home Assistant's system prompt?** C4's value and C3's chunk
  cap both depend on it. One real request in the journal answers it. The
  harness assumes 1.94k tokens (40 devices).
- ~~**The preempt chunk, 2048 or 1024?**~~ **Decided 2026-09-23: 2048.**
  A voice command's worst wait of ~2 s while an agent prefills is fine;
  what is not fine is minutes, so prefill speed wins over a second of
  latency.
- ~~**Default slot count and context per slot.**~~ **Decided 2026-09-23:
  three slots**, and with C6 each holds the full 262 144 cells
  (`ai.service`'s LLM line, `-llm-slots 3`).
- ~~**Does a background stream riding in a voice decode pass (rule 4) cost
  the voice reply too much?**~~ C5's numbers: one rider costs it 1.47x and
  two cost 1.83x. It stays off, because a voice reply is short and its
  latency is its TTFT, which riding does not help.

## C1 — sequence slots (done 2026-09-23)

Broken out to [`c1-sequence-slots.md`](c1-sequence-slots.md).

## C2 — the scheduler (done 2026-09-23)

Broken out to [`c2-scheduler.md`](c2-scheduler.md).

## C0 + C3 — the harness and the chunk curve (measured 2026-09-23)

Broken out to [`c0-c3-harness-and-chunk-curve.md`](c0-c3-harness-and-chunk-curve.md).

## C5a — what a three-row pass costs (measured 2026-09-23)

Broken out to [`c5a-three-row-pass.md`](c5a-three-row-pass.md).

## C4 — checkpoints (done 2026-09-23)

Broken out to [`c4-checkpoints.md`](c4-checkpoints.md).

## C5 — batched decode (done 2026-09-23)

Broken out to [`c5-batched-decode.md`](c5-batched-decode.md).

## C6 — full context in every slot (done 2026-09-23)

Broken out to [`c6-full-context-slots.md`](c6-full-context-slots.md).

## The cache-size cost in prefill (done 2026-09-23)

Broken out to [`c-prefill-cache-size.md`](c-prefill-cache-size.md).

## The MoE GEMV's load count (done 2026-09-23)

Broken out to [`c-moe-gemv-loads.md`](c-moe-gemv-loads.md).

## Stale values past the frontier (done 2026-09-23)

Broken out to [`c-stale-values.md`](c-stale-values.md).

## Rows side by side (done 2026-09-23)

Broken out to [`c-rows-side-by-side.md`](c-rows-side-by-side.md).

## Next

- **What is left in a batched pass (46.2 ms at three rows):**
  - `moe.up` is ~2.2 ms over what its bytes predict at three rows (163 GB/s
    against 192 at one). It is **not** occupancy: the shipped v16w4 Q4_K
    build is 96 VGPRs and 16 waves a SIMD at 1, 2 and 3 rows (RADV
    shaderstats). The multi-row tiles repeat the nibble unpack per row, and
    holding it unpacked is measured dead (above), so this wants a new idea.
  - The host: a batched pass is recorded every step, ~1.5 ms more than the
    replayed one-row step. Replaying it needs the per-row slots and offsets
    out of the push constants, as P1c did for the position.
  - DeltaNet's per-row scan is bandwidth (a slot's state each), and PLE's
    per-row work is 0.1 ms.
- **C7 (mixed passes)** stays conditional on background decode starving
  behind prefill, which C0's numbers have not shown.

## Log

- **2026-09-23**: file opened; plan above. **C1 and C2 done** the same day.
- **2026-09-23**: C0 (`cmd/loadgen`), C3's chunk curve, C1's depth check at
  99k in slot 1, and C5a, all on the 48-layer model with `ai.service`
  stopped. The chunk stays at 2048.
- **2026-09-23**: **C4 done** (within a slot). The voice TTFT goes from 1.83
  to 0.19 s at 48 layers.
- **2026-09-23**: **C6 done**, three slots of the full 262 144 cells, and
  `ai.service`'s LLM line is `-llm-slots 3`.
- **2026-09-23**: chunk decided at 2048 (prefill speed over a second of
  voice latency). **C5 done**: batched decode, per-row passes for the three
  per-sequence blocks, and `moe.up` skipping padding rows. Three agents get
  18 tok/s each instead of 11 and finish in 62 s instead of 88.
- **2026-09-23**: **the cache-size cost in prefill is fixed**: the indexer
  score filled its dead tail across the whole cache. Bit-exact, and prefill
  at the served 262k cache goes from 1037 to 1172 tok/s at depth zero and
  985 to 1047 at 64k. Decode does not move.
- **2026-09-23**: **the MoE GEMV loads once per quad**: Q4_K takes `uvec4`
  header and payload loads, the load is out of the row loop, and the routed
  up mode is v16w4. A three-row pass goes 53.6 → 48.5 ms (61.9 tok/s
  aggregate) and a one-row step 29.2 → 27.6 ms (36.2 tok/s). It exposed
  `TestGraphCheckpointIsThePrefill` as arithmetic-fragile: it fails at row
  11 by 3.5e-4, cause open.
- **2026-09-23**: **the checkpoint gate is green again**, and the cause was
  not the checkpoint. A masked value cell with a negative stale value makes
  a −0 product, and the WMMA's sum differs by an ulp when one of its zero
  products is −0. `llm_attn_pack.comp` now writes +0 over the value cells
  past `n` to the end of a 64-cell block, and `TestGraphStaleValuesAreUnread`
  is the gate.
- **2026-09-23**: **re-measured through the server**, A/B/A/B against
  `8a2b175`: solo decode 33.2-33.6 → 35.5-35.6 tok/s, C0's mix aggregate
  32.6-32.9 → 34.9, agents done 48.5-49.1 → 45.8 s. Voice TTFT unchanged.
- **2026-09-23**: **batched rows side by side**: `vk.MultiDispatch.Overlap`,
  per-row attention scratch, and the per-sequence stages emitted row-major
  within a stage. A three-row pass goes 48.4 → 46.2 ms (62.0 → 65.0 tok/s
  aggregate) and two rows 39.9 → 38.5, bit-identical. `moe.down` and the
  IQ4_NL layout are closed as not worth it.
