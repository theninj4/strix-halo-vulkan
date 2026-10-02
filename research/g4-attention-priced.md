# G4 — the attention priced, and carried to the image DiT (2026-09-30)

*The G4 stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§3.8** in `research/3.8-attention-priced.md`.

**The ISA first (G-o9).** The shipped `h3_attn_t_qt2_kt4` executes, a
key block of 128 MMAs, **911 non-MMA VALU**: 197 register moves (the
allocator rotating registers under the next V fragments' loads), 176
selects (the tail mask on P compiled into every block, and the half-wave
exchange), a 16-deep dependent `v_max3` chain a query tile, 88 address
adds, the softmax's 66 exp / 64 cvt / 92 mul; plus 113 scratch and LDS
reloads of the spilled Q^T and 128 fragment loads. The head-64 VAE build
carries 925 VALU against 64 MMAs.

**The gutting, honestly** (`GUT_*` knobs, each result kept live; the
screen with the sampler, 480p, 15,936 keys, 2650–2690 MHz, ceiling ~51.2):

| | ms | TFLOP/s | per clock |
|---|---:|---:|---:|
| shipped (three runs) | 192.6 – 195.0 | 37.3–37.8 | 73% |
| no rescale / no exp2 / no row max (on the spill-free base at 202.1) | 196.3 / 199.9 / 187.3 | | 3 / 1 / 7 points |
| no softmax, exchange kept | 179.7 | 40.5 | 79% |
| **the floor: MMAs and loads** | **176.3** | **41.3** | **81%** |
| the same four with every K/V load an L0 hit (`L0_LOADS`) | 197.5 → 173.8 / 168.4 / **162.4** | 41.9 / 43.3 / **44.8** | 82 / 85 / **87.5%** |
| 768p, shipped → floor + L0 | 1.192 → 0.933 s | 35.2 → 45.0 | 70 → 90% |

**The exact arms** (bit-identical, every one; the screen now checks it):
`TAIL_SPLIT` (the last block peeled, −111 VALU) 201–208; `QLDS` (Q^T in
8 KB of LDS, no spill) 202–204; `MAX_TREE` 200.6; `LAZY` (skip the
rescale when no lane's max moved: corr is exactly 1) 201.0; `PV_ORDER`
209.9; the best combination `QLDS TAIL_SPLIT PV_ORDER MAX_TREE LAZY`
**192.7 / 197.5** with 776 VALU, 239 moves and no reloads — the shipped
build's time with 135 fewer instructions and 113 fewer loads a block.
`QT1` 337, `KTIL2` 204, `Q_LPITCH=24` 223. Decision 7's plateau, five
deep. The instruction count is not the price here (as §2.13 found): the
allocator's moves at 256 VGPRs redraw with every edit.

**Carry-in.** `qimage/dit` on the transposed kernel (`STORE_TAIL=0` for
its segment-by-segment prefix repair; the element order probed at the
first graph; the context fp16 in place, "narrow ctx" only for the causal
text rows; `QIMAGE_ATTN_PLAIN=1` the control): attention **248 → 232 ms**
a 1024² step (35.7 → 38.3 TFLOP/s, 1.07x — its plain kernel was already
at 35.7 on 4k keys), a cached step **1.908 → 1.876 s**, a prefill 1.966
→ 1.934; `TestGPUOracle256` / `TestGPUOracle1024` pass; **`TestGPUEditOracle`
does not** (the prefill 7.9e-2 against 2e-2, the plain kernel 7.7e-3),
and the block-by-block dumps in §3.8 say why: on identical inputs the
kernel is exact at every block (rms 8.1e-5 at block 0, 1.1e-4 at block
19 run alone), and this edit amplifies fp16-level differences in the
prefix rows ~100x at three target rows between blocks 16 and 20 — a
plain-family perturbation of those rows reads 1.73e-2 against the bound.
The gate stands, so the build is **opt-in** (`QIMAGE_ATTN_T=1`) and the
served image path is unchanged; the VAE's head-64 builds carry the new
per-query bound on the last tile (PSNR 81.7 dB, unchanged). Not carried: `ace/dit` (grouped-query heads the
kernel does not map, a small share of a music step), the speech
encoders, the VAE's exact arms.

**What is left (G4b):** the K/V block staged through LDS for a
workgroup's waves, §2.9's lever, at `QT1` a wave (the staging registers
do not fit beside `QT2`'s accumulators): the 12 memory points less ~6 of
LDS reads at one fragment an MMA and the barrier's lockstep, ≤ 2% of a
video forward for a session. Recorded, not built. *(Built the same day:
G4b below, §3.9 — 1.10x on the kernel and 3.7% of a forward, with one
query tile a wave at eight waves.)*
