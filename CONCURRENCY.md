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

*At depth (2026-09-23, 48 layers):* the 4-layer fixture's cache is 256
cells, so the gate's attention is dense and never takes P15's gathered arm.
That arm was checked through the server instead: `-llm-slots 2 -llm-ctx
131072 -llm-reserve 0`, and the same **98 937-token** prompt at temperature 0
sent three times, which lands in slot 1, slot 0 and slot 1 (`[slot N]` in the
log). All three answers are byte-identical, and they are P18's recall ("Robert
Boulter", plus the corpus's last heading). The gathered prefill and the
gathered, split decode both run above ~4k live cells, so C0's 7-8k agents
exercised them in slots 1 and 2 as well.

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

## C0 + C3 — the harness and the chunk curve (measured 2026-09-23)

**The harness is `cmd/loadgen`.** It sends two background agents at t=0
(a 7.3k/7.8k-token wikitext document each, "summarise it at length",
`max_tokens` 768, thinking on) and four interactive voice commands at 2, 9,
25 and 40 s. Each voice command is a Home-Assistant-shaped system prompt of 40
devices (**1.94k tokens**), a short utterance, `reasoning_effort: none` and
`max_tokens` 32. The voice commands at 2 s and 9 s land while the agents are
prefilling, and the ones at 25 s and 40 s land while they are decoding.
Before each concurrent phase it runs every stream alone as the same-hour
control. Every arm below ran the concurrent phase twice, and **the two runs
agree within 0.03 s on every TTFT and 0.3 tok/s on every rate** (the
baseline's within 0.07 s). The server was the 48-layer model at
`-llm-ctx 65536 -llm-batch 8192`, with `ai.service` stopped. The sweep script
restarts the server per arm from one binary.

| arm | voice TTFT at 2 s / 9 s (in prefill) | at 25 s / 40 s (in decode) | agent decode, each | agent prefill alone (7338 tok) | aggregate |
|---|---|---|---|---|---|
| alone (control) | 1.84 / 1.79 | 1.80 / 1.79 | 34.8 | — | — |
| **1 slot (before C2)** | **27.4 / 22.7** | 9.0 / 24.1 | 35.4, but agent1 waits 40.6 s for its first token | 6.32 s (8192 chunk) | 24.8 tok/s |
| 3 slots, chunk 512 | 2.05 / 1.83 | 1.90 / 1.79 | 16.8 | 11.71 s (**+85%**) | 21.3 |
| 3 slots, chunk 1024 | 2.09 / 2.01 | 1.81 / 1.79 | 16.1 | 9.10 s (+44%) | 22.9 |
| **3 slots, chunk 2048** (the default) | 3.41 / 2.44 | 1.81 / 1.79 | 16.2 / 15.7 | 7.29 s (+15%) | 24.2 |
| 3 slots, chunk 4096 | 3.11 / 4.80 | 1.82 / 1.81 | 16.2 / 14.6 | 6.43 s (+2%) | 24.7 |
| 3 slots, chunk 8192 | 5.74 / 7.35 | 1.83 / 1.79 | 13.1 / 15.4 | 6.32 s | 24.8 |

(TTFTs in seconds and rates in tok/s, all as the client sees them.)

What it says:

1. **C2 does what it was for.** A voice command that arrives while the agents
   are decoding gets its solo TTFT and its solo rate (35.4 tok/s against 34.4
   alone) at every chunk size. The agents stall for those ~2 s and nothing
   else happens to them. One that arrives during an agent's prefill waits for
   the chunk on the device and no longer. Before C2 it waited for a whole
   generation, 9-27 s.
2. **Switching slots costs nothing measurable.** The two agents' decode
   windows (~48.8 s for 768 tokens each, minus ~4.4 s of voice turns inside
   them) come to 34.6 tok/s combined, which is the solo rate. Aggregate
   throughput is the same as one slot's to within the prefill-chunk cost, as
   the design said: C2 buys fairness, not tokens. C5 buys tokens.
3. **A prefill chunk has a fixed cost of ~0.39 s**, and it is the MoE bank.
   Fitting chunk time = c0 + n·c1 over the solo prefills gives c0 ≈ 0.39 s
   and c1 ≈ 0.80 ms a token. 0.39 s is one read of the ~76 GB expert bank at
   ~200 GB/s: every chunk of 512 or more rows touches essentially every
   expert. So a chunk under ~2k is mostly bank reading. At 512 the cost is
   +85% on background prefill, and the aggregate falls 14%.
4. **The worst voice wait is one chunk's duration**, not what the two sample
   arrivals happened to hit: ~0.8 s at 512, **~1.3 s at 1024, ~2.0 s at
   2048**, ~3.3 s at 4096 and ~6.3 s at 8192.

**Decision: the default stays 2048.** Going to 1024 saves ~0.7 s of worst-case
voice wait and costs every background prefill 25% (whenever there is more than
one slot, whether or not a voice command ever comes). The larger part of a
voice command's TTFT today is its **own** 1.8 s prefill of a system prompt
that is identical every time, and C4 removes that. Revisit the chunk after C4,
when the chunk wait will be most of what is left. A different lever is the
chunk's fixed cost: if an MoE pass over 1024 rows read less than the whole
bank, small chunks would be cheap, but at 1024 rows × 8 experts of 256 it
does not.

## C5a — what a three-row pass costs (measured 2026-09-23)

`cmd/llm -graph -tokens 1,2,3 -layers 48 -ctx 2048` on the shipped banks
(`LLM_DENSE_BANK`/`LLM_MOE_BANK` as `cmd/serve` sets them, plus
`LLM_BANK_CACHE`). Two builds, interleaved A/B/A/B: HEAD, where
`GEMVMaxRows` is 2 and a three-row pass falls back to the GEMM, and a scratch
worktree with `MAXROWS 3` in the four GEMV shaders and `GEMVMaxRows = 3`.
The two pairs agree within 1 ms.

| rows | HEAD | MAXROWS 3 | hc | dn | attn | **moe** | head |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.3 ms | 29.3 ms | 3.9 | 7.2 | 2.5 | 11.0 | 2.4 |
| 2 | 38.9 (1.33x) | 41.2 (1.41x) | 4.0 | 7.6 | 2.5 | 22.3 | 2.4 |
| 3 | 106.3 (3.6x) | **52.2 (1.78x)** | 4.2 | 7.7 | 2.6 | **33.1** | 2.4 |

(block columns are the MAXROWS 3 build)

- **Three rows cost 1.78 steps, not the ~1.5 the plan hoped for**, and C5
  found that even this was optimistic. **These three rows are three
  consecutive tokens of one sequence, and consecutive tokens share
  experts.** Three conversations touch 26.8 distinct experts a layer, not
  ~24, and at C5's first build their pass was 64.2 ms (2.22 steps). The
  `moe.up` fix below brought it to 53.4 ms (1.83). See C5 for the numbers
  that stand.
- **Everything but the MoE is flat in rows**, as P5b found at two: hc, dn,
  attn, ple and head are all within 0.4 ms from one row to three. **The MoE
  is the whole cost**, and `moe.up` alone goes 6.0 → 14.8 → 24.0 ms. More
  rows touch more distinct experts, so some of that growth is bytes that must
  be read. But `moe.up` grows 4.0x over three rows, while `moe.down` over the
  same expert set grows 2.3x (2.6 → 6.0 ms). `moe.up` at R rows was the
  kernel to look at, and C5 did: most of it was padding rows.
- **Raising MAXROWS costs two rows 2.3 ms** (38.9 → 41.2, mostly `moe.up`'s
  extra accumulator). Nothing ships two-row passes today (the speculative
  loop is parked), and one row is unchanged, so C5 can raise it. It has to
  land with its rungs.
- **Correctness in the scratch build:** `TestGraphIsAChunkSplit` passes, and
  new *unpinned* "2 / 3 at a time, on the decode schedule" subtests read
  result_norm at 5.2e-4 and 6.2e-4 rms (one row: 4.5e-4; the gate's bound is
  1e-3). The existing "3 at a time" rung runs on the pinned schedule, so it
  would not see a broken three-row GEMV. **C5's commit should carry these two
  subtests.**
- **Rule 4 (background rows riding in a voice decode):** one rider costs the
  voice stream 1.33x (35 → ~26 tok/s) and two cost it 1.78x (→ ~20 tok/s). A
  spoken reply is consumed at speech rate, which is well under either, so
  riding is plausible for TTS-bound replies. It stays a knob, and it defaults
  to off, until C5 exists.

## C4 — checkpoints (done 2026-09-23)

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

## C5 — batched decode (done 2026-09-23)

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

## C6 — full context in every slot (done 2026-09-23)

**Each slot's cache is its own three buffers (key, value, indexer), and each
slot has its own attention pipelines.** A pipeline's descriptor set names its
buffers, so a slot with buffers of its own needs pipelines of its own. The
plan's descriptor array (`PipelineSpec.Counts`, as the MoE bank does) would
have needed the slot index in every cache-reading kernel, and the push block
has no room for it. Instead `AttnGPU.build` runs once per slot against that
slot's buffers, and `UseSlot` swaps the buffers and the pipeline map. The
cache offsets no longer depend on the slot, **no shader changed**, and a
recorded pass was already mixing pipelines of different blocks. So:

- `maxStorageBufferRange` caps a slot's cells (~349k), not all slots'
  together. **3 × 262 144 cells**, where C1's shared buffers allowed
  ~349k in total.
- The pipeline builds cost nothing measurable. The 48-layer server came up
  in 68 s against P18's ~66, because identical SPIR-V hits Mesa's shader
  cache. Residency is **~98 GB, with 19 GB available** on this 117.7 GiB
  machine.
- Batched decode (C5) needed nothing: a row's per-sequence core already
  calls `UseSlot` for its slot.

**Gates:**
- `TestAttnGPUSlotsHaveTheirOwnCache` (block, 0.5 s): slot 2 prefills half
  the 4k fixture, slot 0 runs a different input from cell zero, and slot 2
  continues. The continuation is **bit-identical** to the same two chunks
  with nothing in between. The control puts slot 2's continuation on slot
  0's pipelines, and it moves from the first continued token. A fresh run
  cannot see a shared cache, because it writes every cell it reads; the
  first version of this control put one inside
  `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` and passed unchanged, which
  is how that was found.
- `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` now also runs slot 2 of a
  3-slot block with a 3x cache, after slot 0 ran something else, and demands
  the same bits.
- **At 48 layers**, `-llm-slots 3 -llm-ctx 262144 -llm-batch 8192`: two short
  requests take slots 0 and 1, and a **237 231-token** prompt lands in slot
  2 and recalls P18's "Robert Boulter" (plus the last heading, "IRA
  resurgence"). The same prompt again lands in slot 0, and **the two answers
  are byte-identical**. Decode at 237k is 30.1 tok/s in both slots (P18:
  29.9).
- C0's mix on the same server gives agents 21.4-22.9 tok/s each and voice as
  before (0.20-0.26 s TTFT outside an agent's prefill chunk).

**Two costs, and neither is C6's:**
- **The 237k prefill runs at 901 tok/s, against P18's ~1022.** With more
  than one slot a background prefill is cut into `-llm-preempt-chunk` (2048)
  chunks, and at this length that is ~116 chunks of C3's ~0.39 s fixed
  cost. It is the trade decided above. A client that wants P18's rate for
  one long prompt can send it as interactive, which is not chunked.
- **A 262k-cell cache made prefill attention twice as dear, at any
  depth.** A 2048-token pass at depth zero cost 366 ms of attention against
  178 ms in a 65k-cell cache (1926 against 1729 ms a pass). It predated C6
  and every 262k configuration since P18 paid it. **Fixed the same day**
  (the next section): it was the indexer score's tail fill.

## The cache-size cost in prefill (done 2026-09-23)

**The indexer's score wrote one float a token for every pooled block the
cache holds, not the context.** Every block at or past `nBid` (the first
that is not whole) pools cell 0 and so shares one score, and P7 already
computed that score once. But `llm_attn_score_wmma.comp` then *filled* it
into all `nBlocks - nBid` entries of the row. That is 65 535 floats a token
at depth zero in a 262 144-cell cache, **512 MB a layer for a 2048-row
pass**, streamed out at ~24 GB/s. Nothing on the device reads those entries:
P14-1's block select stops at the last block a cell can reach, and the
expansion (the cell-select control arm) maps every tail cell to `deadBid`.
Only the host's `AttnGPU.Score()` readback, which the tests compare with the
reference's whole `indexer_score_tokens`, ever saw them.

So the kernel writes the one entry at `nBid`, and `Score()` fills the rest on
the host, so the tensor keeps the reference's shape. The scalar control
kernel (`LLM_ATTN_SCORE=scalar`) got the same change, so the two arms keep
one contract. `-attn -tokens 2048` found it in one run: `score` was 4.46 ms
a layer at 65k cells and 20.6 at 262k. After the fix it is **0.29 and 0.30**,
because the fill was most of the kernel at *any* cache size.

**It is bit-exact.** The selection reads the same scores. `TestGraphLogits`
reproduces its diagnostics to the digit through 48 layers (maxAbs 1.987e+00
at 19208, rms 3.314e-01, argmax 11751 against 561). `TestAttnGPUIndexer`
and `TestAttnGPUIndexer4k` compare the full tensor, tail included, against
the reference, and they pass because of the host fill. The attention, QSA
and graph gates pass (`-run 'TestAttn|TestQSA|TestGraphIsAChunkSplit|
TestGraphSlots|TestGraphDecodeRows'`, 49 s).

**At 48 layers**, shipped banks, `cmd/llm -graph -tokens 2048`, the old and new
binaries interleaved, two passes each (agreeing to 6 ms):

| cache | before | after | attention before → after |
|---|---:|---:|---:|
| 262 144 cells | 1922 / 1916 ms (1067 tok/s) | **1696 / 1696 ms (1208 tok/s), +13%** | 365 → 149 ms |
| 65 536 cells | 1728 / 1723 ms (1187 tok/s) | **1688 / 1688 ms (1213 tok/s), +2.3%** | 178 → 143 ms |

The 262k cache now costs 8 ms more a pass than the 65k one, where it cost
194 ms.

**Against depth** at the served cache (`-depth -depths 0,64000,128000,192000
-pp 2048 -ctx 262144`, one sweep each). The fill was `nBlocks - nBid` a
token, so the gain shrinks as the context fills:

| depth | pp before | pp after | |
|---:|---:|---:|---:|
| 0 | 1043.4 | **1170.2** | +12.2% |
| 64 000 | 984.4 | **1050.8** | +6.7% |
| 128 000 | 953.2 | **968.1** | +1.6% |
| 192 000 | 892.2 | **901.7** | +1.1% |

**Decode did not move, and the prefill gain reproduces.** The sweep's
4-token decode read 0.2-0.6 tok/s lower in the new arm, so three
interleaved pairs of `-depths 0,64000 -tg 128` settled it:

| | pp at 0 | pp at 64 000 | tg at 0 | tg at 64 000 |
|---|---:|---:|---:|---:|
| before (3 runs) | 1035.4-1038.6 | 984.7-986.0 | 34.06-34.12 | 32.61-32.82 |
| after (3 runs) | **1170.6-1174.6** | **1046.4-1048.0** | 34.13-34.18 | 32.72-32.79 |

Decode is +0.07 / +0.05 tok/s, inside the spread. A decode step's tail fill
was 65k floats a layer, which is small but not nothing, and the dead score is
now computed by one workgroup rather than every stripe. The 237k agent
prefill C6 measured at 901 tok/s is not re-measured. Linear in the tail,
the saving averages roughly half of depth zero's 226 ms a chunk over a
0-237k walk, so an estimated ~5%.

**What still scales with the cache, and is left:** `idx` at `past == 0`
pools every block of the table (0.09 → 0.31 ms a layer, once a sequence),
because a later pass's dead block reads that fill. `select` is 0.21 →
0.38 ms a layer, probably the score rows' wider stride. Together they are
the 8 ms above.

## The MoE GEMV's load count (done 2026-09-23)

**The expert GEMV issued one scalar load per four bytes of Q4_K nibbles,
plus four more for the block header, and it was bound by the load count.**
`moe.up` and `moe.down` run over the same tiles, but at three rows `moe.up`
streamed its Q4_K gate and up banks at ~128 GB/s while `moe.down` read its
IQ4_NL bank at ~190 (1.84 and 0.92 MB an expert). This is P12's finding
about the GEMM's slab, made again for `llm_moe_gemv.comp`, which never got
it. Two changes:

- **The load is outside the row loop.** `dotDword` had loaded, decoded and
  dotted one row, and the row loop called it once per row. So a tile with R
  real rows issued every load R times. The shared expert is the same bytes at
  any row count, and it showed this most clearly: `shexp.up` took 15.6 /
  23.4 / 32.2 µs a dispatch at 1/2/3 rows. Now `unpackDword` loads once and
  `dotRow` is the per-row half, and every product and sum stays in the order
  it had. Three rows: `shexp.up` 1.55 → 0.98 ms, `shexp.down` 0.73 → 0.54,
  and the pass 53.6 → 53.0 ms.
- **Q4_K is walked a quad at a time.** Four consecutive payload dwords are 16
  bytes of one 64-element group and share both scale pairs, so a quad is one
  `uvec4` header load plus one `uvec4` payload load, where the dword walk
  issued twenty scalar loads. Every Q4_K block is 16-byte aligned (P12
  checks it). The sum is now grouped by quad, so the decode path is **not**
  bit-identical to before. `TestMoEGPUDecode` still reads rms 2.54e-06
  against the GEMM, the batched-rows gate is still bit-identical row for
  row, and prefill (`TestGraphLogits`) is untouched to the digit.
- **The routed up mode moved from v64w4 to v16w4.** A row is 80 quads, which
  16 divides and 64 does not. L8e-2 once measured v16w4 42 ms worse in the
  whole model on the dword kernel, so the choice was made in the graph, not
  on the cache-hot block ladder: v16w4 < v32w4 < v64w4, two runs each. The
  shared expert's rung (v64w4) was within 0.04 ms of the alternatives and
  stays.

`cmd/llm -batch 1,2,3 -batch-depth 7000`, 48 layers, shipped banks; "after"
is four runs from two sweeps of the final configuration:

| rows | before | after | a row | aggregate | `moe.up` | `shexp.up` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.2 ms | **27.6-27.7** | 34.2 → **36.2 tok/s** | | 6.02 → 4.60 | 0.75 → 0.72 |
| 2 | 42.5 | **39.0-40.0** | 23.5 → 25.3 | 47.1 → **50.6** | 12.6 → 10.0 | 1.12 → 0.92 |
| 3 | 53.6 | **48.3-48.7** | 18.7 → 20.6 | 56.0 → **61.9** | 18.9 → 14.3 | 1.55 → 1.15 |

**Single-stream decode gains too, about 1.6 ms a step (+6%).** Through the
server it is +7%.

**Through the server** (C0's harness, the served `-llm-ctx 262144
-llm-batch 8192 -llm-slots 3`, `ai.service` stopped). A is `8a2b175`,
before this change, and B is the tree, which also carries the stale-value
fix. The arms ran A/B/A/B in one half hour, with two concurrent runs each:

| | A (two arms) | B (two arms) |
|---|---:|---:|
| solo decode | 33.20-33.60 tok/s | **35.51-35.62** |
| solo voice TTFT (from the checkpoint) | 0.187-0.192 s | 0.188-0.189 |
| agents' decode in the mix, each | 22.8 / 21.9 - 23.1 / 22.2 | **25.1 / 24.1** |
| agents done | 49.1 / 48.5 s | **45.9 / 45.8** |
| aggregate | 32.58 / 32.93 | **34.89 / 34.93** |
| voice TTFT in the mix (2 s / 9 s / 25 s / 40 s) | 1.77-1.79 / 0.98-1.05 / 0.24-0.25 / 0.21-0.24 | 1.77-1.78 / 0.96-0.99 / 0.22-0.25 / 0.20-0.22 |

The gap is the same in both pairs, and the within-arm spread (0.4 tok/s solo
in A, 0.1 in B) is a fifth of it. Voice latency does not move, because it is
the prefill chunk the command lands behind. Prefill TTFT is 7.2 s for a
7.3k-token agent prompt in both arms.

**Measured and dropped:** unpacking each Q4_K weight once and applying it to
every row inside the element loop, which keeps the shared expert's
three-row gain. It cost more than it saved at every row count (29.0 ms at
one row, `shexp.up` 1.25 against 0.72): P11's rule again, a branch inside an
unrolled loop. So a three-row shared expert gives back 0.17 ms of the first
change's gain (0.98 → 1.15).

### A gate this exposed: `TestGraphCheckpointIsThePrefill` failed

The quad change made C4's gate fail at row 11 by 3.5e-4. The defect was
older than the change, and it is written up in the next section.

## Stale values past the frontier (done 2026-09-23)

**A decode pass's context depended on the sign of stale values in cells it
cannot see.** The split block kernel (`qt1_kt2.split`, BN = 32) loads whole
16-cell tiles of the value plane straight into a fragment. On a cell past
the row's extent P is +0, so a stale value there is multiplied by zero. The
product should add nothing, but it does not: **the WMMA's sum comes out one
ulp different when one of its zero products is −0**, which is what a
negative stale value makes. How it was found, on the 4-layer gate:

- **Only the value plane.** Writing back the reference's K, raw-indexer
  or pooled planes changed nothing. Writing back its V plane fixed it, and
  so did writing back **one cell (2911, position + 1 at row 11)**, then one
  kv head's one element (dim 138).
- **The sign and not the magnitude.** That element at +0, 1.0, 1000 or
  any positive value passes bit for bit. At −0, −6e-8, −0.17, −2048 or any
  negative value it fails, and with the same digits every time. −Inf gives
  NaN, which shows that the cell is read and multiplied by zero.
- **It depends on where the slices fall.** `LLM_ATTN_SPLITS=1` and 2 and 8
  pass. 4 fails at row 7 (cell 2907, again position + 1), and 64 fails at
  row 11. The split partials show it: only slice 10 (the block 2880-2911)
  differs, with its row max and sum bit-identical and its context off in
  a few dims per head. **The two reference runs already differed there**
  (V = 0 in the first, the real value in the second), just not by enough
  to reach the logits.
- So HEAD passed only because of the values that happened to be there, and
  why the failure began at exactly a 300-token detour is now explained. That
  detour is the first to leave its own value in cell 2911.

**The fix is in `llm_attn_pack.comp`.** One workgroup per value head writes
+0 over cells `[n, roundUp(n, 64))` (64 is the widest key block, KTIL = 4)
before any attention reads the plane. Nothing past `n` is a cell of this
sequence yet, and the next run to reach one writes it before any row can
see it. The gathered kernels were already immune: they stage through LDS
and zero anything past the list. The key needs nothing, because a masked
score is replaced rather than summed.

**Gates:** `TestGraphCheckpointIsThePrefill` passes again at 16 (default),
4 and 64 splits. **`TestGraphStaleValuesAreUnread`** (new, 5 s) decodes 48
steps from 2600 cells twice, with every value cell past the frontier set to
−0 before each step in one arm and +0 in the other. The arms must agree bit
for bit. With the old pack both tests fail: the new one at step 11 (−1.764984
against −1.764973).

**It re-scoped L7a's chunk gate.** `TestAttnGPUCacheIsAChunkSplit` said a
token's output is bit-identical whether it arrived alone or inside a bigger
chunk. That held only because each subtest reran the same prompt over the
last one's cells, so the stale values were the real ones. With the V plane
cleared first, HEAD's own shader failed at 7 at a time (34 726 values) and
one at a time (4891). The gate is now exact for every token whose 64-cell
key block its chunk closes, which covers all of the 512- and 64-token
schedules. A token whose block the chunk leaves open is held to 1e-4 of its
row's largest output. Measured: 2.17e-5 at 7 at a time, 7.29e-6 at one at a
time. A cache off by one position is ~1e-1.

What it covers in production: any restore, speculative rewind or slot reuse
could shift a decode by an ulp at some positions. That is harmless to
quality but broke every bit-exact comparison across a rewind. Prefill is
unaffected, because its future cells in the diagonal block are this pass's
own.

## Rows side by side (done 2026-09-23)

**A batched pass ran its rows' per-sequence work one after another, and
most of it is too small to fill the device.** At three rows, going from one
row to three added 19.1 ms. The profile (`cmd/llm -batch 1,2,3
-batch-depth 7000`, shipped banks, two runs agreeing to 0.4 ms) splits it:

| | added, 1 → 3 rows | what it is |
|---|---:|---|
| `moe.up` | +9.75 ms | 2.64x the experts. The bytes alone predict +7.5, so ~2.2 ms is excess (below) |
| `moe.down` | +3.62 | 2.4x for 2.64x the experts: nothing to take, so the IQ4_NL layout item is closed |
| attention, per row | +2.6 | `attn.split` 1.79, then score, select, combine, pack, idx, gather |
| DeltaNet, per row | +2.0 | `dn.scan` 1.40, then conv, norm, hist |
| host | ~+1.5 | a batched pass is recorded every step; the one-row step is replayed |

**The change: `vk.MultiDispatch.Overlap`**, meaning no barrier between a
dispatch and the one before it. It goes to both recording paths (the shim's
per-dispatch byte array). A group's timestamps are written together after
its last dispatch, so every query slot is still written and the group's
time is reported on its first dispatch. The per-sequence blocks now emit
their rows **stage by stage** (every row's conv, then every row's scan, and
so on), with the rows of a stage overlapped:

- **DeltaNet** needed only the reorder. Its rows already touch only their own
  rows of the per-token tensors and their own slot's state and ring.
- **Attention** shared its scratch across rows (query plane, indexer query,
  scores, selection, gathered list, split partials, block list). So each
  row now takes its own 64-row slice of every one of them
  (`AttnGPU.rowScratchFor`). The arenas are cut for the prompt batch, so a
  decode batch fits many times over, and anything that does not fit falls
  back to the old order. **The combine stays serialized in ascending row
  order**, because it writes the context's pad rows past its own row, and
  those are the next rows' real ones.
- PLE's per-row conv and hist are left as they were (0.1 ms).

`LLM_BATCH_OVERLAP=0` is the control: the same order with every barrier.

**Measured** (48 layers, control/overlap/control/overlap, same binary):

| | 1 row | 2 rows | 3 rows | attention at 3 | DeltaNet at 3 |
|---|---:|---:|---:|---:|---:|
| control | 27.7 ms | 39.8-39.9 | 48.4 (62.0 tok/s) | 6.1 | 8.8-8.9 |
| **overlap** | 27.6 | **38.5-38.6** | **46.2 (65.0 tok/s)** | **4.1** | **8.2** |

DeltaNet alone (the first version) was 0.7 ms of this. The scan streams a
slot's recurrent state, ~3.1 MB a layer, so three rows are bandwidth and
not latency, and an overlapped group of three scans still takes ~3x one.
Attention's split is 24 single-wave workgroups, and three of those side by
side cost about what one did.

**Gates:** `TestGraphDecodeRowsIsEachSlot` now runs dense (2048 cells, as
before) and **sparse** (4096 cells, prompts of 2600, 2200 and 2400 tokens,
so every row scores, selects and gathers). It asserts the rows really got
their own scratch, and every row stays bit-identical to its solo step.
Its negative control was run by hand: with every row given row 0's scratch
and the overlap left on, the sparse case failed 3 of 3 times at 0.25-0.27
rms, a different figure each run, which is a race and shows the rows really
do run at once. The full `llm` and `backend` suites pass.

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
