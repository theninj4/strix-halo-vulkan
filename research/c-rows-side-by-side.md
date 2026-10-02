# Rows side by side (done 2026-09-23)

*Broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
