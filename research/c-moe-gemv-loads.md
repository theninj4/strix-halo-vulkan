# The MoE GEMV's load count (done 2026-09-23)

*Broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


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
