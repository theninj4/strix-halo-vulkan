# C2 — the scheduler (done 2026-09-23)

*The C2 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
