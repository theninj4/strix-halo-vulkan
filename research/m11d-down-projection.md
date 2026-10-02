# M11d — the down projection (2026-09-29)

*The M11d stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


After M11c, a 480p forward's profile (`TestGPUShapes`, `H3_PROFILE=1`,
int8) is attention 32%, qkv 16%, **down 16%**, gate and up 11% each, `o`
6%. Every GEMM runs at 35–38 TFLOP/s except the down projection, at
24–28. Its grid is `o`'s: 21 × 64 workgroups of 128×256 at chunk 8192,
N = 5376 for both. What differs is K: 14,336 against 7,168.

**Three screens, 480p** (`gemm down`'s share of one forward's profile):

| arm | gemm down | |
|---|---|---|
| swizzle band 8 (shipped) | 4.33–4.71 s, 26–28.5 TFLOP/s | |
| band 16 / 4 / 2 | 22.6 / 14.1 / 9.3 TFLOP/s | refused |
| A row pad 0 / 64 / 256 / 512 halves (shipped 128) | 19.8 / 24.0 / 23.7 / 25.6 | refused |
| **K in 2 passes** (`-DKRANGE`) | **4.24–4.27 s, 28.9–29.1** | shipped |
| K in 3 / 4 passes | 29.3 / 29.2 | no better than 2 |

The band result refutes the first guess, which was that a band of 8
columns of B slabs (59 MB at this K, against 29 MB for `o`) overflows the
32 MB MALL. Narrower bands are *worse*, because every band re-streams the
235 MB A chunk. The shipped band and pad are both already the optimum.

**K in two passes.** `dit_gemm.comp -DKRANGE` runs the K loop over
`[aux0, aux1)`. With `aux2 = 1`, its accumulators start from the fp32 C
already in place. An fp32 store and reload is exact, so the two passes
make the same MMAs in the same order as one: **bit-identical**, which
`TestGPUForward` now asserts (`dit.GPU.downSplit`, default 2;
`H3_DOWN_SPLIT=n` in `TestGPUShapes`). Two passes also steady the
kernel: one pass measured 24.0–28.5 TFLOP/s across runs, two passes
28.9–29.1. It still stops short of `o`'s 35, so K's length is part of the
gap, not all of it.

**Forwards** (int8, two passes of each arm, interleaved): 480p **29.81 /
29.91 s** against 30.41 / 30.42 s one-pass (1.02x). 768p 106.1 / 109.6
against 107.2 / 114.4, inside that shape's run-to-run spread.
