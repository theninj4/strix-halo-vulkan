# C5a — what a three-row pass costs (measured 2026-09-23)

*The C5a section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
