# G5d — the nibble as an f16 denormal (2026-09-30)

*The G5d stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§2.13** in `research/2.13-moe-nibble-denormal.md`.

**The estimate was wrong by the compiler.** §2.12 priced the slab unpack
at ~300 VALU a lane a step and G5's plan was `v_perm`/`v_cvt_pk`
sequences. The ISA of the shipped `up_q4k_m4` says an element is three
instructions, not five — ACO already fuses `ds * nib - dm` and the
`float16_t()` into one `v_fma_mixlo/hi_f16` — so the only work in front
of the floor is a `v_bfe_u32` and a `v_cvt_f32_u32` an element. Both go
by reading the nibble as a half: a halfword whose low nibble is the
value and whose other bits are zero is the f16 denormal `nib × 2^-24`,
`v_fma_mix` takes a half from either half of a word through op_sel, and
`d × sc × 2^24` is as exact as `d × sc`. Three instructions for four
elements, the same fma operands to the bit, and the slab is byte-identical
(`TestMoEGPULadderAgrees` at `maxAbs == 0` with the new m4 against the
old m2/m2; the negative control through the same override fails).

| four-tile K-step, VALU beside 64 MMAs | shipped | `NIB_F16` |
|---|---:|---:|
| extract + convert | 128 | 0 |
| masks | — | 48 |
| `v_fma_mix` (floor) | 64 | 64 |
| allocator moves | 65 | 71 |
| scale path, addresses, A store | ~74 | ~76 |
| **non-MMA VALU** | **331** | **259** |

| `moe.up`, µs a layer (two interleaved runs) | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| shipped | 2 960 / 2 964 | 4 075 / 4 118 | 6 236 / 6 298 | 10 235 / 10 205 |
| **`NIB_F16`** | **2 922 / 2 921** | **4 022 / 4 016** | **6 180 / 6 115** | **9 941 / 10 014** |
| `NIB_F16` + `PIPE_A=0` | 3 290 / 3 280 | 4 546 / 4 508 | 7 014 / 6 887 | 11 225 / 11 311 |

**Why 2% for a 22% cut in VALU.** The count is an upper bound: four
waves a SIMD overlap one wave's unpack with another's MMAs part of the
time, which §2.10's single-kernel probe does not see. **And the move
storm is closed**: the 71 moves are the allocator building eight-register
fragments from scattered `ds_load_b64` pairs at the 256-VGPR limit
(m1/m2/n-rungs at 96–240 VGPRs carry 4–6); the K tiles as a loop
(`MOE_KT_LOOP`, §2.10's fix) adds 38 moves a tile here, and freeing the
A prefetch's sixteen registers (`PIPE_A=0`) removes the moves and loses
12% to the exposed latency. Whole model: **1 513.1 / 1 512.7 → 1 521.1 / 1 523.3 tok/s at 2048 (1.006x)**, 1 549.3 / 1 546.7 → 1 553.9 / 1 556.6 at 4096 (1.005x), 1 483.9 / 1 483.4 → 1 491.4 / 1 487.5 at 8192 (1.004x), the clock 2 671–2 674 MHz at 141–142 W on every arm. The ladder at the
plan boundary: on the served IQ4_NL down, one run of 30 iters, `m2/m2` against `m4/m4` is 4 057 against 3 994 µs a block at 256 tokens and 5 616 against 5 431 at 512, so the wide rung leads at both and `moeGEMMPlanFor`'s boundary at 256 stands (the 1.6% at 256 is inside a run's spread and 256 is not a served ubatch).
