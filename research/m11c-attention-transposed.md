# M11c — the attention, transposed (2026-09-29)

*The M11c stage, broken out of `VIDEO.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`video-vertical.md`](video-vertical.md).*


The attention was 42% of a served forward and 68% of a trained one, at
20–26 TFLOP/s against 55.5. **It is now 1.55–1.61x faster, and a forward
is 1.16x (480p) and 1.30x (768p).**

**Where the time went.** The kernel was priced by gutting it one phase at
a time on H3's own planes (block 49 after a real forward, 480p, 15,936
keys, QT1 KTIL4; `H3_ATTN_SPV` loads a build from a file into
`TestGPUAttentionScreen`):

| arm | time | TFLOP/s |
|---|---|---|
| the plain kernel | 288 ms | 25.3 |
| no row max (LDS scan, 2 barriers a key tile) | 288 ms | 25.3 |
| and no exp2 | 296 ms | 24.6 |
| and no LDS round trip for P (store, barrier, reload as A) | **111 ms** | 65.7 |

The softmax costs nothing. The round trip that turns P from an
accumulator back into an A operand is **~60% of the kernel**. The
extension has no other path from an accumulator to an operand.

**Refused first: LLM's P12-1 lane split** (the row max over all 32
lanes, one clustered max, the running max in a register). Bit-identical,
and **slower in every arm**: 23.8 → 23.1 TFLOP/s at QT1 KTIL4, 480p, and
22.4 → 19.8 at QT2 KTIL4, 38k keys. That is what the table predicts: the
row max was never the cost. It is not in the tree.

**The fix: compute the transpose.** `shaders/h3_attn_t.comp` runs
S^T = K·Q^T and O^T += V^T·P^T. On this device an accumulator's column
and a B operand's column are the same lane. The accumulator splits a
column's 16 rows between lanes l and l^16 (even rows in the first
half-wave, odd in the second), where B wants all 16 in both. So P^T
becomes the B operand with one `subgroupShuffleXor(16)` per packed pair
of halves. A lane then owns one query of a tile, and the row max, the
online correction, the row sum and the final 1/sum are all per-lane
scalars. **No LDS and no barrier.** The memory reads are the plain
kernel's, byte for byte: K as a row-major A, Q^T as a column-major B, the
packed v tile as V^T row-major, and O^T stored column-major is O at the
output projection's stride.

That relies on an element order `GL_KHR_cooperative_matrix` leaves to
the implementation, so it is checked, not assumed.
`shaders/coopmat_layout_probe.comp` reads back which (row, column) every
lane's element is, for A, B and both accumulators. `checkCoopMatLayout`
runs it at the first `Begin` (held in a server; staging still submits
nothing), and on any other order the build falls back to the plain
kernel. `TestCoopMatLayout` pins it: A lane l = row l%16, element e =
column e; B lane l = column l%16, element e = row e; accumulators lane l
= column l%16, element e = row 2e + l/16.

**Two numeric differences, neither bit-for-bit with the plain kernel.**
The denominator sums the fp16-narrowed weights in fp32 registers instead
of through a ones column on the matrix cores. The key tail is masked out
of the max as well as out of P. The second makes the kernel immune to
stale keys by itself: `TestGPUForward`'s negative control (no
`zeroKeyTail`) now pins the plain kernel, which still goes NaN, while the
transposed one stays at rel 5.3e-4. The zeroing stays, because P = 0
still multiplies whatever stale v rows hold.

**The screen** (`TestGPUAttentionScreen` + `H3_ATTN_SPV`, block 49's
planes; rms relative to the plain QT1 KTIL4 context):

| build | 15,936 keys | 38,247 keys | VGPRs / spilled |
|---|---|---|---|
| plain QT1 KTIL4 | 297 ms (24.5) | 2.16 s (19.4) | |
| plain QT2 KTIL4 (was picked ≥ 24k) | 297 ms | 1.87 s (22.4) | |
| T QT1 KTIL4 | 246 ms (29.6) | 2.02 s (20.8) | 216 / 0 |
| T QT2 KTIL2 | 227 ms (32.1) | | 256 / 64 |
| **T QT2 KTIL4** | **194 ms (37.6)** | **1.17 s (36.0)** | 256 / 112 |
| T QT2 KTIL4, Q reloaded a block (`QREG=0`) | 215 ms (33.8) | | 256 / 0 |
| T QT2 KTIL8, QT3, QT4 | 301–881 ms | | spills 266–1341 |

rms against the plain kernel **9.7e-5 / 1.1e-4**, about the size of fp16's
own rounding of the context. QT2 KTIL4 spills 112 VGPRs and still wins.
Reloading Q every block removes the spill and loses 11%. `attnFor` runs
T QT2 KTIL4 at every key count.

**Gates** (fp16 bank unless noted):

| gate | result |
|---|---|
| `TestGPUForward` forward 0 | video rel 6.43e-3 (M7: 6.7e-3), audio 1.25e-3; blocks ≤ 1.9e-4 of absmax |
| `TestGPURun`, 7 teacher-forced steps | latents rms 9.8e-5 … 1.68e-3 (M7: 7.9e-5 … 1.7e-3) |
| `TestGPURun`, int8 | every step inside the released bf16 pipeline's error (step 6: 6.9e-3 against 1.59e-2) |
| `TestGPUFL2VA`, int8 | steps 1.4e-3 rms, as M10 |
| `TestE2E`, int8, free-running | frames 20.3 dB, soundtrack 16.0 dB (M11a's int8 draw: 19.8 / 16.5) |

**Forwards** (`TestGPUShapes`, int8, `H3_ATTN_PLAIN=1` as the control, two
passes of each arm interleaved):

| shape | plain | transposed | |
|---|---|---|---|
| 864×480 × 124 (served), 15,936 rows | 35.80 / 35.37 s | **30.77 / 30.50 s** | 1.16x |
| 1344×768 × 124 (trained), 38,247 rows | 143.9 / 145.1 s | **108.1 / 113.6 s** | 1.30x |

A served 480p × 20 request (19 forwards) should lose ~95 s of M11b's
741 s. That is estimated from the forwards, not yet measured served.
