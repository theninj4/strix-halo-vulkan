# G8 — the carry-in

*The G8 stage's two write-ups, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*

## G8 — the carry-in (2026-09-29)

Every host of the 128×256 kernel now creates `dit_gemm_wg128x256_lds_w32`
pinned to wave32 (a `wave` field on each variant table; `K % 32` where a
host checks its tiles), through its own gate and screen, the same
session, the machine otherwise idle:

| host | gate | screen, old → new | note |
|---|---|---|---|
| `h3/dit` | `TestGPUForward`, bit-identical | 480p forward 29.86 → 27.11 s | G2 |
| `h3/vae` | `TestGPUDecoder`, PSNR 81.7 dB unchanged | 480p decode 37.1 → **35.5 s** device, 768p 66.1 s | its bias column made K = H + 16; K now rounds to the big GEMM's tile (`gemmVariants[gemmBig].bk`), the pad columns zero in A and B |
| `qimage/dit` | `TestGPUOracle1024` | `TestGPUGEMMScreen`, one staging: swz8 2179 → **`lds_w32` 2058 ms a step** (−5.6%; swz2/4/16 +30/+18/+10%) | the fifth arm of the screen, the default |
| `ace/dit` | `TestGPUDiT` | 8 forwards at 30 s / 2 / 4 / 10 min: **wave64 0.96** / lds 3.27 (w64 3.24) / **6.76** (6.93) / **18.79** (19.37) s | **the LDS build loses at 375 rows** (130 against 120 ms: 30 workgroups, the barrier shows), ties at 1500, wins 2.5–3% from 3000; `bigW64Rows = 1024` picks the wave64 build below that; the 64×64 rung is 131 / 488 / 1002 / 3215 ms, never the answer |
| `kev` (fp16 bank) | `TestGPURowMatchesKev` | `TestGPUGEMMLadder`, 494 tokens: 154.2 → **132.4 ms** (reg64 at 37 tokens 49.7 both days, so the machine is the same) | serves int8 (`q8m4`, 155.3 at 494), which the fp16 rung now beats there: G5's argument |
| `parakeet` | `TestGPUEncoderStack` | ladder, `wg128x256`: 138 / 384 / 768 / 1024 frames 28.8 / 36.9 / 67.3 / 111.6 → 31.1 / 40.9 / 66.9 / 109.1 ms | its plan never picks the rung (`reg32x32_w32` wins every clip); at 8–12 workgroups the build loses 8–11% |
| `zimage/qwen` | `TestGPUEncoder`, `TestGPUKernelsAgree` | — (its rung is the >192-token one) | rung name now `wg128x256_lds_w32` |
| `ocr` | `TestLM`, `TestEngine` (all six cases token-identical) | the page's 1,234-row prefill 100 → 102 ms | the prefill is 2.5% of a page |
| `ace/lm` | `TestLogits`, `TestDeterministic` | prefill 239 rows 106 ms | rows ≤ 256 are GEMVs |

**The rule it adds:** the LDS-staged build needs the machine full. Its
per-slab barrier and staging cost show whenever the grid is under ~40
workgroups (rows × N / 32,768), and the hosts whose inputs are short
(parakeet, the LM prefills, a 30 s clip) already run smaller rungs or a
wave64 build there. Where a host's screen gave a number, it is above.

## G8 — G3's epilogues carried to the music DiT and the video VAE (2026-09-30)

The two passes that read a GEMM's fp32 C (`pack v`, `swiglu`) are the
GEMM's epilogue on two more hosts, as host changes only (the interleaved
gate|up weight at staging, a `gemmWith`, `packV` / `fuseGLU` with their
controls); `dit_gemm.comp` and its builds are untouched. Both hosts are
**bit-identical to the binary before the change** (a hash of the outputs
written by the old test binary and compared by the new one) and to their
in-tree controls.

| `ace/dit`, ms a forward (old / fused / control, interleaved twice) | 375 rows (30 s) | 1500 (2 min) | 3000 (4 min) | 7500 (10 min) |
|---|---:|---:|---:|---:|
| before (gate, up, `swiglu`, `pack v`) | 119.5 / 119.7 | 381.7 / 382.0 | 802.4 / 804.6 | 2258.8 / 2262.2 |
| the control (fused GEMM + the SwiGLU pass, the pack) | 119.9 / 120.9 | 379.7 / 377.3 | 804.8 / 805.3 | 2255.1 / 2258.9 |
| **fused** | **111.4** (one run, the final build) | **347.2 / 347.8** | **742.4 / 746.5** | **2121.8 / 2126.8** |
| | 1.07x | 1.10x | 1.08x | 1.06x |

2627–2631 MHz at 145–146 W fused against 2672–2683 at 143–145 before
(the sampler over the whole timing test), so the gain is not clock. At
4 minutes the `swiglu` pass was 50.4 ms of 801 (6.3%) and `pack v` 3.3;
the fused gate|up runs at 44.9 TFLOP/s where gate and up ran 43.5 and
42.2. **The row rule moved for this one projection**: the fused gate|up
runs on the LDS build at every row count, because its grid is twice a
projection's and there is no pass behind it — against the wave64 build's
fused GEMM plus the pass, 110.1 / 110.9 against 120.5 / 119.7 ms at 375
rows, 181 against 200 at 750, 249 against 274 at 1000, a tie at 125
(75.5–76.6). Every other projection keeps `bigW64Rows`. v is stored by
its GEMM on either build (`gemmBigPack` / `gemmBigW64Pack`), the
request's 32 cross-attention v planes too; the A pad rows are zeroed
first (`zero_f16`), as in H3. Gates: `TestGPUBits` (new: the condition
encoder, a forward at 30 s and at 4 min and the detokenizer hashed, the
same four hashes from the old binary, and fused against control),
`TestGPUEncode`, `TestGPUDiT`, `TestGPUDetok`, `TestGPUTokenize`, and
`ace/pipeline`'s `TestGenerate` and `TestTasks`, all passing.

| `h3/vae`, 124 frames, batches of 8 (`TestGPUShapes`, every run) | 480p device | 768p device | useful TFLOP/s |
|---|---:|---:|---:|
| before | 34.41 / 34.47 / 34.40 s | 64.23 / 64.28 / 64.26 s | 29.4–29.5 |
| the control | 34.28 s | 64.03 s | 29.6 |
| **fused** | **30.02 / 29.94 s (1.15x)** | **55.97 / 57.42 s (1.12–1.15x)** | **33.0–33.9** |

2762 MHz at 144 W fused against 2786 at 138–139 before; the second
fused run sat at 2696 MHz / 129 W over its 768p half and read 57.4 s,
the machine's and recorded as such. 1.15x is more than H3's DiT took
from the same two fusions (1.03x) because the VAE's rows are 14,464 a
batch in fp16 throughout with 36 blocks and nothing else between them:
the passes were a larger share and their DRAM traffic a larger share of
the power. Two things are the VAE's own. **Its biases ride in the
GEMM**, so the interleaved weight is built from the two halves each with
its bias column, and the epilogue's fp16 store stops at the FFN's width,
under the down projection's column of ones. **And v is stored by its
GEMM only when the batch's rows are whole 128-row tiles** (8 sequences
of 1,808 are; the last partial batch of a decode runs the pack): a pad
row of this GEMM is the bias, not +0, and zeroing the A operand's pad
rows would take the ones column with them. Gates: `TestGPUDecoder` (now
staged for 8 sequences; the short decode and a batch of eight tile-clips
fused against control, bit for bit) and `TestGPUDecodeFull`, PSNR 81.7 dB
unchanged, both decodes' hashes equal to the old binary's.

**The speech encoders, priced and left.** Parakeet's feed forward is
linear → SiLU → linear, not gated, so there is nothing to interleave;
its plan runs the 32×32 wave32 rung, not the LDS build; and
`TestGPUProfile` puts the whole 24-layer encoder at 13.5 ms for an 11 s
clip with `silu ff1` + `silu ff2` at 1.8% and `pack v` at 0.5% — 48 and
24 dispatches of 3–5 µs each, which is the dispatch floor. A `C_SILU`
epilogue on that rung is a new build for ≤ 0.3 ms a clip. Kokoro's
ALBERT has one GELU pass a layer over a few hundred rows and is smaller
again (not measured). Neither is carried.
