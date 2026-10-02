# G5a — the MoE up GEMM's K loop as a pipeline (2026-09-29)

*The G5a (and G5b) stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§2.11** in `research/2.11-moe-gemm-pipeline.md`.

**Read the ISA first, then price the parts.** The shipped `up_q4k_m4`
(one wave64, a 4×4 tile grid over two 64-column slabs, BK 32) is per
K-step four permutation loads, four gathered `uvec4` of A, per slab
column a sixteen-byte header and two `uvec4` of nibbles with ~150 VALU
of decode, twelve `ds_store_b128`, then 96 `ds_load_b64` and 64 MMAs —
256 VGPRs, four subgroups a SIMD, and no barrier at all (one wave). Every
load is issued after the previous step's MMAs and waited for before this
step's; every bank load is sixty-four lanes at sixty-four rows 1440 B
apart, sixteen bytes used of each line, and the same lines again next
step. The controls (`-D` knobs, wrong by construction, loaded with the
new `LLM_MOE_SPV` into `cmd/llm -moe -tokens 2048`, the m4 rung, layer 3,
the real routing):

| taken away | `moe.up` | vs 11.53 ms |
|---|---:|---:|
| the A gather's fetch (`MOE_L0_A`) / the A staging (`MOE_NO_ASTAGE`) | 10.53 / 10.48 | 1.09x / 1.10x |
| the bank's fetch, same sixty-four lines a load (`MOE_L0_BANK=2`) | 8.81 | **1.31x** |
| the scale path (`MOE_NIB_ONLY`, P12-3) | 8.43 | 1.37x |
| the bank's fetch and its divergence (`MOE_L0_BANK=1`) | 8.02 | **1.44x** |
| the B unpack (`MOE_NO_UNPACK`) | 6.56 | 1.76x |
| both stagings: MMAs and fragment reads alone | 5.33 | 2.16x (47.8 TFLOP/s, the pipe at that clock) |
| (added) every fragment read twice (`MOE_FRAG2`) | 12.51 | 0.92x |

So the LDS round trip §2.2 blamed is ~8 points, and the bank's fetch —
latency in front of every step plus the re-reading of lines — is 25–30.
G5's register-built fragment would replicate the unpack 4x at wave64 to
save those 8; not built.

**The arm** (`-DPIPE=2` on `llm_moe_gemm.comp`, the Q4_K up mode): the
bytes a lane unpacks at step k+1 are fetched into registers before step
k's MMAs, the permutation entries once before the loop, **the header
once a super-block and the nibble group once per two steps** (9 bank
loads a super-block instead of 24); the unpack, slabs and fragment reads
are the shipped ones, so the MMAs see the same halves in the same order.
Bit-identical (`TestMoEGPULadderAgrees` on each of the seven geometries,
the whole `TestMoEGPU*` suite with all seven loaded):

| `moe.up`, µs a layer | 512 (m2) | 2048 (m4) | 4096 (m4) |
|---|---:|---:|---:|
| shipped | 6 508 | 11 512 | 17 106 |
| a step ahead, every load every step (`PIPE=1`) | 6 185 | 11 054 | 15 008 |
| **… and each block's bytes once (`PIPE=2`, shipped)** | **3 241 (2.01x)** | **6 837 (1.68x)** | **10 925 (1.57x)** |

Executed 22.1 → 37.3 TFLOP/s at 2048 (~75% per clock; useful 11.7 →
19.6 under the m4 rung's 1.90x padding). `PIPE=1` alone is 4–14%
because the allocator copies the fetched registers inside the MMA block
and waits there (§2.10's moves again); a two-set form spills at m4 and
buys nothing where it fits. Refused: the wave32 m4 (spills 217), the
two-set pipeline, the register fragment. The seven `llm_moe_up_q4k_*`
generate lines carry `-DPIPE=2`; every other build keeps the plain loop.

**In the whole model** (`cmd/llm -graph -tokens 2048,4096,8192 -ctx 8192`,
48 layers, the shipped banks, old and new binaries interleaved twice the
same hour): prefill **1194 / 1206 → 1407 / 1408 tok/s at 2048 (1.17x)**,
1317 / 1339 → 1494 / 1495 at 4096 (1.13x), 1306 / 1345 → 1459 / 1462 at
8192 (1.10x); the embedded builds pass the whole `TestMoEGPU*` suite and
the 1024-token ladder keeps `moeGEMMPlanFor`'s m2/m4 boundary.

**The down mode, the same session (G5b):** the pipeline written for the
block-32 down formats (B only; twenty builds, all spill-free, the whole
suite passing) moves **Q5_1 by nothing** (6 587 → 6 607 µs at 2048,
though its `L0_BANK` control read 1.29x: that control removed the load's
own cost, not latency, and a 32-element block a step has nothing to
hoist) and **IQ4_NL, the served down format, by 1.05x at 2048, 1.20x at
512, 1.04x at 4096** (5 634 → 5 366; 2 718 → 2 266; 8 264 → 7 946), which
is its nine-word pair read once per two steps. The five IQ4_NL down
builds ship with `-DPIPE=2` (embedded, the suite passing; whole-model
prefill 1408 → 1422 tok/s at 2048 and 1502 → 1507 at 4096, two runs
each); Q5_1, Q4_1 and Q8_0 keep the plain loop.
What binds the down mode now is its A fragment loads inside the MMA
block, which a fragment's back-edge copy (§2.9) keeps from pipelining.

**Next in G5:** ~~the up rung's padding factor (1.90x executed rows at
m4, 2048 tokens: the largest term left in the up kernel)~~ (G5c), the
Q5_K up build (layer 2), and the down mode's A side.
