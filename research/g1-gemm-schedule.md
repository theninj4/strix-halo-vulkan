# G1 — the shipped GEMM's schedule (2026-09-29)

*The G1 stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§6.5** in `research/6.5-gemm-schedule.md`.

**The loop, from the disassembly** (`dit_gemm_wg128x256_bt16_swz8`,
wave64, 240 VGPRs, no spill, no LDS; ACO says 6 subgroups a SIMD, which
is not the arithmetic of a 1536-register file at wave64 and is not
relied on). One iteration is one 16-wide K tile:

| instructions | count | what |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` | 32 | the 4×8 accumulator grid |
| `buffer_load_b128` | 24 | 4 A + 8 B fragments, 2 loads each, in 3 clauses |
| `s_add/s_mul/s_lshr/s_lshl/s_load/s_mov` | 56 | the B tile addresses, **recomputed every iteration** |
| `v_lshlrev/v_add3/v_mul_lo/v_mbcnt/v_and` | 29 | the A addresses and lane offset, likewise |
| `s_delay_alu` | 20 | dependency stalls in that arithmetic |
| `s_waitcnt` | 9 | `vmcnt(14)`, then 12, 10, … 0, four MMAs between each |
| `s_clause`, branch, compare | 5 | |

Order: address arithmetic → all 24 loads → `s_waitcnt vmcnt(14)` (A and
the first B landed) → 4 MMAs → wait → 4 MMAs … → `vmcnt(0)` → the last 4
→ loop. **G-o3 answered: no.** ACO does not hoist the next tile's loads
above this tile's MMAs; each iteration pays the load latency once, from
the address arithmetic to the tenth load landing, with only the other
waves on the SIMD to fill it. 4.5 non-MMA instructions an MMA, so it is
not issue-bound by count (H3); but ~85 of them sit on the critical path
in front of the loads.

**Occupancy does not move it** (`-DLDS_PAD`, an unused shared array that
caps workgroups a WGP; `H3_GEMM_SPV`, 480p, int8):

| build | LDS | workgroups a CU | gemm qkv | gate / up | o | forward |
|---|---|---|---|---|---|---|
| shipped | 0 | ≥ 3 | 36.9 | 36.6 / 36.6 | 34.5 | 30.10 s |
| `LDS_PAD=4096` | 16 KB | 4 | 37.2 | 36.7 / 36.6 | 34.1 | 30.21 s |
| `LDS_PAD=8192` | 32 KB | 2 | 37.3 | 36.6 / 36.5 | 34.2 | 30.04 s |
| `LDS_PAD=16384` | 64 KB | **1** | 37.6 | 37.0 / 36.9 | 34.7 | 30.00 s |

Flat to 1%, down to one 4-wave workgroup a CU. So the exposed latency is
not being covered by other waves at *any* of these occupancies, and
adding waves is not a lever (G-o2 is moot for this kernel).

**Take the memory system away** (`-DL0_LOADS=1`: every fragment load
reads one of two K tiles, so the 24 loads are issued and waited for as
before but every one is an L0 hit; the result is garbage by
construction; the `down` projection keeps the shipped KRANGE build, so
it is a control):

| | shipped, real loads | `L0_LOADS` | |
|---|---|---|---|
| gemm qkv | 36.9 | **43.6** | 1.18x |
| gemm gate / up | 36.6 / 36.6 | **44.2 / 44.0** | 1.21x |
| gemm o | 34.5 | **42.8** | 1.24x |
| gemm down (control, real loads) | 26.7 | 29.9 | the clock |
| attention (unchanged kernel) | 37.2 | 39.4 | the clock |
| sclk during the forward | 2600–2650 | 2650–2700 (less memory power) | |
| **per clock, FLOP/clk/CU** | **353 (74%)** | **~405 (82–85%)** | |

So the gap decomposes: **~10 points are the memory system** (the
difference between L0 hits and the real slabs, at the same schedule),
and **~16 points are the in-wave schedule** with the loads perfect: the
once-a-tile stall from the address arithmetic through the L0 latency to
the tenth load, against ~1,090 clocks of MMA. Neither part responds to
occupancy, which is what the LDS arms say.

**What G2 builds, in order** (re-ranked by the energy roofline, §0.7).
(1) Strength-reduce the addresses: every fragment's offset advances by a
constant per iteration, so the 85 instructions are 12 adds; cheap,
bit-identical, shortens the critical path in front of every iteration's
loads, and issue costs watts (the 4-chain probe burns 14 W more than the
16-chain one for the same MMAs). (2) Fewer bytes a FLOP: LDS sharing of
each slab across the workgroup's 4 waves (the two waves that share an A
slab each load it today; H6, untested at this tiling) and the largest
per-wave tile the registers allow, measured in TFLOP/s *and* W, because
traffic is the power and the power is the clock. (3) Prefetch one K
tile ahead into a second register set, so the loads a wave waits on
were issued 1,090 clocks earlier: +24 VGPRs, which at WN = 8 means a 4×4
or 8×4 grid instead; it buys the in-wave stall and no clock. Bit-identical
C for all three, `TestGPUForward`; every arm reports GFLOP/J.

**Hypotheses after G0/G1:** H1 confirmed and larger than thought (10%
of the ceiling is the clock; G7 asks whether anything can be done about
it). H2 alive as the in-wave stall, not as an occupancy story. H3 alive
only as the address arithmetic on the critical path. H4, H5 unmeasured.
H6 is the ~10 memory points. H7 dead.
