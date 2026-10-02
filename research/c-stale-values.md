# Stale values past the frontier (done 2026-09-23)

*Broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
