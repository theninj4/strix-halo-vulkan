# G5c — the short last tile (2026-09-30)

*The G5c stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§2.12** in `research/2.12-moe-short-tiles.md`.

**The rows were the cost now.** P11-4 padded to the fragment three ways
and lost every time because the slab unpack was the kernel; after §2.11
the MMAs are ~78% of it. The permutation's record already said how many
of a block's rows are real (C5 wrote it for the decode GEMV); the
single-wave builds of `llm_moe_gemm.comp` are now `SHORT` builds that
round it to the schedule's alignment (`pc.gemmM`) and run one of WM
copies of the K loop and store on a uniform `switch` — no bound inside
the unrolled loops, the same MMAs per real row in the same order —
and the alignment (`MoEGPU.pad`, `moeAlign`) is sixteen rows unless a
plan names a multi-wave rung. Bit-identical: `TestMoEGPULadderAgrees` at
`maxAbs == 0` across every mixed plan, `TestMoEGPUPaddingIsInert`, and
`TestGraphLogits`'s whole-model line reproduced to the digit against the
old test binary.

| 2048 tokens, m4, µs a layer (old / new, two interleaved runs) | rows | `moe.up` | `moe.down` | block |
|---|---:|---:|---:|---:|
| the padded schedule | 1.90x | 6 994 / 7 029 | 6 577 / 6 609 (Q5_1) | 15 733 / 15 793 |
| **SHORT** | **1.19x** (up executes 1.41x) | **6 314 / 6 255** | **5 370 / 5 339** | **13 822 / 13 721** |
| … with the served IQ4_NL down | | 6 922 → 6 374 | 5 373 → **4 209** | 14 441 → **12 727** |

**What decided the form.** Any third copy of the Q4_K up loop spills
11–15 registers its epilogue needs across the K loop, and a control run
of the four-copy build with `LLM_MOE_PAD=64` (every block whole) is 9%
slower than the shipped kernel on the same work; the two-copy form
(`SHORT_STEP=2`: 2 and 4 tiles, the store still bounded by the real
count — the first cut stored the rounded tile over the next expert's
rows, a race the ladder test caught) costs nothing on full blocks and
executes a 16-row tail as 32. The down builds have 120–144 registers and
keep all four. **A short tile's rows are cheap rows**: on the four-copy
build the alignment 64 → 16 sheds 14 608 rows for 1 255 µs, 0.086 µs a
row against the MMA-only control's 0.137, because the tile still unpacks
two whole slabs (~300 VALU a lane a step, a matrix clock each, §2.10)
beside its sixteen MMAs. That unpack is now the largest term in the up
projection at every rung.

**In the whole model** (`cmd/llm -graph`, old and new interleaved twice):
**1416.5 / 1418.6 → 1514.0 / 1513.2 tok/s at 2048 (1.07x)**, 1499.7 /
1502.1 → 1549.4 / 1542.8 at 4096, 1461.4 / 1466.7 → 1480.8 / 1482.7 at
8192. The rung ladder on the SHORT builds puts `m4/m4` ahead of `m2/m2`
from 512 tokens (6 380 against 6 460; 8 999 against 9 361 at 1024) and
behind it at 64–128, so `moeGEMMPlanFor`'s boundary is 256, not 1024.
Tools: `-DSHORT`, `-DSHORT_STEP` on `llm_moe_gemm.comp`, `LLM_MOE_PAD` on
the host (a pre-§2.12 build loaded through `LLM_MOE_SPV` needs it at 64).
