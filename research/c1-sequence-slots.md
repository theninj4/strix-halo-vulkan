# C1 — sequence slots (done 2026-09-23)

*The C1 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
