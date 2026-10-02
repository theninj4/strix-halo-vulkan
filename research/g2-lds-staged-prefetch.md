# G2 — the prefetch that fits (2026-09-29)

*The G2 stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§2.9** in `research/2.9-lds-staged-wave32-gemm.md`, with the
whole ladder. The shipped build is now `dit_gemm_wg128x256_lds_w32` (and
`_krange`): eight wave32 waves of 4×4 tiles in the same 128×256
workgroup, the K slab (two tiles) fetched by the workgroup as 6 `uvec4`
a lane a slab ahead, written to the other of two LDS planes behind the
MMAs, one barrier a slab, fragments read from LDS at a conflict-free
pitch of 36 halves. Bit-identical to the wave64 build (`H3_GEMM_REF`,
`TestGPUForward`).

| 480p, int8 | qkv | gate / up | o | down | forward | sclk | per clock |
|---|---|---|---|---|---|---|---|
| wave64 4×8 (was shipped) | 37.2 | 37.3 / 37.1 | 35.4 | 28.7 | 29.86 s | 2620 | 74% |
| **`lds_w32`, two runs** | **41.1 / 40.9** | **40.7 / 40.7** | **38.9 / 39.1** | **40.8 / 40.6** | **27.16 / 27.11 s** | 2665 | **80%** |

The three findings that decided it: **a wave64 fragment is 8 VGPRs**
(4x replicated across the lane groups), so §6.5's "+24 VGPRs" for a
prefetched tile was +96 and no register prefetch fits beside a 4×8
grid; **ACO copies a fragment carried across a loop back-edge element
by element** (200 `v_mov_b16` a trip), so the two register-pipelining
arms were dead before they ran, and only a build that carries nothing
(a hoisted two-tile slab: +18% at the same intensity; or LDS) measures
the idea; and **wave32 and LDS are one lever**: wave32 halves the bytes a
fragment read moves but caps the tile at 4×4 and, fed from global memory,
runs at 21.5; fed from LDS it runs at 40.9. Refused on the way, all
bit-identical: strength-reduced addresses (nothing: the arithmetic was
never the cost), four hoisted tiles (no better than two), the read
order inside a slab (nothing), LDS pitches of 16 and 24 halves (the
reads are `ds_load_b64`, 16 lanes a clock: 4-way and 2-way conflicted).

**What is left** (§2.9 item 5): with the staging loads all L0 hits the
build reaches 82% per clock, so the global side is ~3 points and the
LDS-side schedule (64 `ds_load_b64`, 6 stores, a barrier and the
`lgkmcnt` ladder a 544-clock slab) is ~18. G-o8.
