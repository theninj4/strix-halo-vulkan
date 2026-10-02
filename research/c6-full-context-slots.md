# C6 — full context in every slot (done 2026-09-23)

*The C6 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
