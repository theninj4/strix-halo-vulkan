# CONCURRENCY — more than one completion at a time

> **Opened 2026-09-23.** This is the live file for P6 ("batching"), which
> `TODO.md` and `research/llm-vertical.md` carried as *blocked on the product
> question: will the API serve more than one stream?* **The answer is yes**:
> **three concurrent completions**, as fair as the hardware allows, with a
> **priority lane** for a latency-bound caller (a voice command) over
> callers that can wait a few seconds (background agents). Like `TODO.md`,
> this file is rewritten rather than appended to; a closed stage's write-up
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

- **Decode: 34.2 tok/s through the server** (~29 ms a step), and the step is
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
4. Once batching exists (C5), background decode rows **ride along** in the
   interactive request's decode pass whenever that costs the interactive
   stream less than a set fraction of its rate. Whether 1.32x a step for a
   free second stream is acceptable in a voice reply is a measurement and a
   knob, not a decision to make now.
5. At most `S` requests hold slots. A fourth waits for one, and it is logged
   as queued, as today.

## Stages

Each stage has a gate that can fail, and the gate lands in the same commit as
the change.

| # | stage | what it is | gate |
|---|---|---|---|
| **C0** | **Instrument** | A load generator (`cmd/bench -llm-concurrent` or similar) that fires one voice-shaped request (a fixed ~2k-token system prompt + a short utterance, short answer) against two agent-shaped ones (long prompts, long answers). It records per-stream TTFT, token rate and total throughput. Run it against today's server for the baseline. | Numbers in this file, two runs agreeing. |
| ~~**C1**~~ | ~~**Sequence slots in the graph**~~ **done 2026-09-23** (below) | `GraphOpts.Slots`, `Graph.UseSlot(s)`. Attention allocates `S×` the cache and offsets it by slot; DeltaNet and PLE allocate `S` state/ring slots, and the *committed slot* is the sequence slot. The graph keeps `past`, `ids` and the recorded decode step per slot. Refused together with `Speculative` (the ping-pong and the slots share an index). | `TestGraphSlotsAreSequences`: two sequences interleaved token by token through one graph give **bit-identical** logits to each run alone, prerecorded decode included. Its control (the same interleave without switching slots) must differ. |
| ~~**C2**~~ | ~~**The scheduler**~~ **done 2026-09-23** (below); `-llm-slots` stays 1 in `ai.service` until C0/C3 are measured | `backend.LLM` gets a scheduler goroutine that owns the graph. `Complete` submits work and receives tokens. Slot choice: longest reusable held prefix, else the least recently used idle slot. `held` becomes per slot. Priority from `service_tier`/header/flag. `-llm-slots N` (default 1 until C3 lands). | An `api`-level test with three concurrent requests on the 4-layer prefix: each one's output equals its solo run at temperature 0. The interactive one's TTFT is bounded by one unit, not by the other requests' generations. |
| **C3** | **Bounded work units** | Price prefill against chunk size at depth (512 / 1024 / 2048 / 4096 / 8192). While an interactive request is live, cap background chunks at the chosen size. Measure the switch cost between slots (per-slot recorded decode steps should make it ~0). | C0's harness: interactive TTFT and token rate with two background streams live, against the same request alone. |
| **C4** | **Checkpoints: prefix caching across threads** | Snapshot a slot's carried state at a **stable boundary**: the end of everything before the last user message. That is DeltaNet state + rings (~120 MB, ~1 ms on-device), the PLE ring, the pooled indexer block that straddles the boundary, and the position. KV cells below the boundary are never rewritten by a continuation, so within a slot they need no copy. A request whose prompt shares the checkpointed prefix restores it and prefills only the tail. That fixes the voice case (same system prompt, new utterance) and agents that retry or branch. Cross-slot restore copies the KV cells too (27.8 KB a cell, ~0.11 GB for 4k). | Restore-then-continue is **bit-identical** to a fresh prefill of the whole prompt, and a control that skips the DeltaNet restore differs. |
| **C5** | **Batched decode** | R decoding slots advance one token each in one pass. Per-row position and slot come from an arena table (P1c's trick again: the recorded buffer stays byte-identical). The row-aware kernels are attention (`pack`, `idx`, `score`, `select`, the gathered `wmma`), DeltaNet (`conv`, `scan`, `seq_hist`) and PLE's `hist`. Raise `GEMVMaxRows` 2 → 3 with its rung in `TestGraphIsAChunkSplit` in the same commit (see the memory note on new row counts). **C5a prices it first**: a 3-row pass at 48 layers against a 1-row one, which needs the machine (stop `ai.service`). | Each row of a batched pass equals that slot's solo step (to the R-row GEMV's documented tolerance), whole graph, at R = 2 and 3. |
| **C6** | **Full context in every slot** | K, V and indexer planes as a descriptor array of per-slot buffers (`PipelineSpec.Counts`, as the MoE bank does), which lifts the `slots × ctx ≤ ~349k` cap to 3 × 262k (~22 GB of KV). | `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` extended across slots; P18's recall at 237k in a non-zero slot. |
| C7 | *Mixed passes* (maybe) | Decode rows riding in another slot's prefill chunk, so a background prefill does not stall the other streams' decode. Only worth building if C3's numbers leave background decode visibly starved. | — |

Not planned: a second Vulkan queue or global-priority queues to preempt a
pass mid-flight. The arenas are shared, so preemption would need a second set
of them, and C3's short chunks bound the same latency far more cheaply.

## Open questions for later stages

- **How long is Home Assistant's system prompt?** C4's value and C3's chunk
  cap both depend on it. One real request in the journal answers it.
- **Default slot count and context per slot.** Until C6, `3 × 112k` or
  `2 × 174k` against today's `1 × 262k`. What do the agents actually use?
- **Does a background stream riding in a voice decode pass (rule 4) cost the
  voice reply too much?** This waits on C5a's number.

## C1 — sequence slots (done 2026-09-23)

`GraphOpts.Slots` and `Graph.UseSlot(s)`. It turned out to be almost free to
build, because P5c's rollback had already put a slot index on every carried
tensor but one:

- **DeltaNet and PLE** allocate `Slots` state/ring slots where the ping-pong
  allocated two (`WithDNSlots`, `PLEOpts.Slots`). The *committed slot* is the
  live sequence, a pass reads and writes it, and `Reset` clears that slot
  alone. Because the two uses share the index, `Slots > 1` refuses
  `Speculative`.
- **Attention** allocates `Slots x` each cache plane (`WithAttnSlots`) and
  `UseSlot` moves `hK/hV/hIdxRaw/hIdxK` to the slot's base. Every push
  constant and read-back follows them. No shader changed. `checkBufferRange`
  now reports the cap over all slots.
- **The graph** keeps `past`, `ids` and the P1c recorded decode step per slot
  (`parked`), because the recorded buffer's push constants carry the slot's
  offsets. A switch moves nothing on the device: four blocks re-point, three
  arena dwords get the position, and the host swaps a slice and a buffer
  handle.

**Gate:** `TestGraphSlotsAreSequences` (4 layers, 6 s). Two sequences of 7
and 5 prompt tokens, 24 greedy steps each, are interleaved a pass at a time
with the recorded decode step live in both slots. **All 50 logit rows are
bit-identical to the solo runs.** Three controls each sabotage one block's
switch (that block stays on slot 0), and **each moves sequence A at its first
decode row**: attention, DeltaNet and PLE are each load-bearing and each
visible to the gate. The existing gates (`TestAttnGPU`,
`TestGraphIsAChunkSplit`, `TestPrerecordedDecode`, `TestSpec*`, `TestRewind`,
`TestSnapshot`, `TestGraphPrefix`, `TestDeltaNetGPU`, `TestPLE`) pass
unchanged at one slot.

*Not covered yet:* the 4-layer fixture's cache is 256 cells, so the attention
is dense and the gathered arm (P15's depth epoch) never ran in a non-zero
slot. That arm reads the same four offsets, but it is owed a run at depth
(C3's measurements will exercise it).

## C2 — the scheduler (done 2026-09-23)

`backend/llm_sched.go`. One goroutine owns the graph. `Complete` acquires a
slot, submits its prefill and each decode step as units, and keeps its own
sampling, stop sequences and streaming, so it samples while another
conversation's pass runs. `held` is per slot. Slot choice is the longest
usable held prefix, else least recently used. The policy is the three rules in
"Priority" above, plus a newcomer starting level with the least-served job of
its class (otherwise it would take the device from a conversation that has
been running for a minute until it caught up). Flags: `-llm-slots` (1),
`-llm-preempt-chunk` (2048), `-llm-reserve` (1), `-llm-class` (background).
Priority comes from `X-Priority`, then `service_tier`, then `-llm-class`. The
log line gains `[slot N, class, Xs stalled]` when there is more than one slot.

**Gates** (`backend/llm_sched_test.go`, 4 layers, ~15 s together):

- `TestLLMConcurrentIsSolo`: three greedy conversations run concurrently
  through three slots stream **exactly** their solo text.
- `TestLLMInteractiveGoesFirst`: two background conversations are decoding
  (160 tokens each), and a 16-token request arrives. With
  `service_tier: "priority"`, the background ones stream **0 deltas** while
  it runs, and it finishes in **166 ms**. **The control**, the same request
  with no tier, sees them stream **19 and 20** deltas beside it, and it takes
  **419 ms**. The window is long enough to see a priority, and the priority
  holds.
- Through HTTP (`cmd/serve -llm-layers 4 -llm-slots 3`, by hand): two
  background curls and one with `X-Priority: interactive`. The interactive
  one took the reserved slot 2, a 42 ms TTFT and 126 ms total, and the two
  background lines report ~1.2 s `stalled`.

## Next

- **C0 + C3 need the machine** (a 48-layer server is ~83 GB, and `ai.service`
  currently holds the image/speech side on this box). The load generator is
  the first thing to write, then the preempt-chunk curve at depth, then the
  gathered arm in a non-zero slot at 128k.
- **C4 (checkpoints)** can be built and gated on the 4-layer prefix now.
- **C5a** (price a 3-row pass at 48 layers) needs the machine. C5 itself is
  kernel work in attention, DeltaNet and PLE, and its gate runs at 4 layers.

## Log

- **2026-09-23**: file opened; plan above. **C1 and C2 done** the same day.
