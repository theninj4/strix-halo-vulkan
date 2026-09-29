# KERNELS — the matrix cores to their ceiling

> **Live tracking and session handoff doc, opened 2026-09-29.** Stage
> letters are **G** (GEMM; K is classification, W and R are speech). This
> is the project's first vertical, reopened: not a model, but the kernels
> every vertical stands on. A closed finding still gets a `§N.M` number and
> a file in `research/` as before (README there: the number is the address,
> never renumber); the G-stage here is where it is tracked while open. When
> the vertical closes, this file is frozen to `research/kernels-vertical.md`.

## What we are building

`GOALS.md`'s first sentence: *the most efficient ways to run these models
on Vulkan on this part*. Every vertical since has been a model brought up
on the kernels this one produced, and each of them ended on the same
sentence: "the GEMMs are at their ceiling, the bigger fish is elsewhere".
That ceiling was **42.0 TFLOP/s against a measured 55.5**, and it has not
moved since stage 4 (2026-09-21). Eight verticals later, the matrix cores
are the one place where a win lands in *every* served request at once: a
480p video forward is 63% GEMM and 33% attention on these kernels, an
image step 72% GEMM, a prefill chunk mostly GEMM. **A 1.15x on the GEMM is
worth more than any single vertical's remaining backlog.**

The deliverable is three kernel classes at a higher fraction of the
per-clock ceiling than today, carried into the verticals through the
screens they already have, and the ceiling itself pinned better than it
is:

| class | file | today | target |
|---|---|---|---|
| fp16 WMMA GEMM (activation × fp16 weight) | `shaders/dit_gemm.comp` (the vertical kernel), `shaders/gemm_wmma.comp` (the ablation suite) | 39.0 suite / **42.0** pipeline, 72–78% | ≥ 85% per clock |
| WMMA flash attention | `shaders/h3_attn_t.comp` (transposed, wave32), `shaders/dit_attention_wmma.comp` (plain) | **37.6–38.8**, 68–70% | ≥ 80% |
| int8 / Q4-weight WMMA GEMM | `shaders/llm_moe_gemm.comp`, `shaders/llm_gemm.comp`, `shaders/kev_gemm_q8_glu.comp`, `dit_dequant_q8` + fp16 GEMM (video) | MoE Q4 up: 22 → **37 TFLOP/s executed** at 2048 tokens (§2.11, ~75% per clock; 51% in §2.2), down 19–21, the rest unmeasured against the ceiling | the fp16 GEMM's rate at the same M, N, K |
| streaming kernels (norms, packs, gates, SwiGLU, copies) | everywhere | 6–14% of every block; `hc.cn` at 134 GB/s (TODO) | ≥ 200 GB/s each, or fused away |

Decode is not in the table: §1.7–§1.12 took the GEMV to 96–103% of the
bus, and the LLM's decode has been a *format* question since (L8).

## What we did (the record, so it is not re-run)

The kernel work of 2026-09-13 to 09-21 is filed under `§N.M` in
`research/ideas.md` and its per-item files. The ladder, on the fp16 GEMM:

| step | lever | best GEMM | of 55.5 | file |
|---|---|---|---|---|
| §0 | the ceilings measured on this chip, not extrapolated | — | — | `0-measurement-validity.md` |
| baseline | one 16×16 accumulator a wave, `gemm_coopmat_fp16.comp` | 4.9 | 9% | |
| §2.1 | register blocking: a WM×WN accumulator grid a wave (AI 8 → 32) | 25.2 | 45% | `2.1-register-blocking.md` |
| §2.3 | padded leading dimensions: a K-strided fragment load off one DRAM channel | 28.4 | 51% | `2.3-channel-aliasing.md` |
| §2.7 | a whole K-slab's fragment loads in flight before its first MMA | 39.0 | 70% | `2.7-kslab-hoist.md` |
| §2.8 | the weight stored as 16×16 fragment tiles (packed once at upload) | 41.0 on two of three shapes | 74% | `stage-4-dit-graph.md` |
| §2.4 | the grid walked in bands of 8 columns (MALL locality) | **42.0 / 41.1 / 40.6** on the DiT's three shapes | 73–76% | `stage-4-dit-graph.md` |

And on the others:

- **Attention** (§3.3, `stage-3-dit-attention.md`): both matmuls on the
  matrix cores, online softmax, every operand as fragment tiles: 1.28 →
  **38.8 TFLOP/s** (30x), wave32 a clean 1.40x over wave64 at the same
  tiling. Then VIDEO.md **M11c**: the LDS round trip that turned P from an
  accumulator into an operand was ~60% of the kernel; computing S^T and
  O^T instead puts P^T on the B side with one shuffle per element pair, no
  LDS, no barrier, 1.55–1.61x on H3's kernel. M11e carried it to head 64.
- **Q4 weights on the matrix cores** (§2.2): 2.10x a MoE block, at **51%**
  of the matrix cores, because the operand must round-trip through LDS
  ("no extension exposes a fragment's lane layout"). **That premise is now
  false**: M11c's `coopmat_layout_probe.comp` reads the element order back
  and it is fixed and simple on this device.
- **Grouped MoE GEMM** (§3.5): 1.1–4.1x, and the mechanism was occupancy.
- **conv2d as an implicit GEMM** (stage 8): conv3x3 at **41–42 TFLOP/s**;
  the dilated 1-D convolutions of kokoro's vocoder on `dit_gemm`'s
  `A_CONV` at 22.4.
- **Decode GEMV** (§1.1, §1.7–§1.12): 96–103% of the 236 GB/s bus at
  every N; done.
- **Wave size** (§6.2): a per-pipeline knob; the GEMM spills at wave32 and
  loses, the fragment-dominated kernels gain 1.4–2.1x.
- **The ISA** (§6.1): `cmd/probe` + `RADV_DEBUG=asm|shaderstats`. It found
  §2.3 (a row-major B fragment is sixteen scalar loads). **It was deleted
  by accident on 2026-09-22** (collateral in a speech commit, README still
  documented it) and **restored today**.

**Refused, measured, so not to be re-run without a new reason:** LDS
staging of A/B slabs (§2.1, lost at every tile; the MALL supplies the
reuse), LDS double-buffering (same), split-K on the big shapes (M11d: two K
passes buy 1.02x on `down` only because its A chunk overflows the MALL,
and a chunk of 4096 gets the same without it), swizzle bands other than 8
(§2.4, M11d), A row pads other than 128 (M11d), QT4 and KTIL8 attention
builds (spill), the lane-split row max (P12-1, slower everywhere), wave32
for the register-blocked GEMM (spills, −25%), an int8 *kernel* where the
GEMM is compute-bound (WMMA int8 runs at the fp16 rate, §0.1).

## What we have today

The ceiling, restated per clock, because the clock moves under load:

| | rate | sclk | power | FLOP/clk/CU |
|---|---|---|---|---|
| spec, 512 FLOP/clk/CU × 40 × 2.9 GHz | 59.4 | 2900 | | 512 |
| `peak` probe, wmma_fp16, wave64, 8 chains (`results/peak.csv`) | **55.5** | 2899 | 107–119 W | **478** |
| best suite GEMM, `wmma_reg64_bt_hka4_padab128` at 2048³ | 39.0 | **2813** | **137 W** | 347 (72.6%) |
| best pipeline GEMM, `dit_gemm_wg128x256_bt16_swz8` on `dit.qkv` at M = 4096 | 42.0 | *not recorded* | | 362–373 (76–78%) |
| H3's GEMMs, 480p (VIDEO.md M11h) | 37–38 | not recorded | | ~70% |
| H3's GEMMs on `dit_gemm_wg128x256_lds_w32` (G2, §2.9) | 40.7–41.1 (down 40.8) | 2665 | 154 | 385 (80%) |
| **… with the K tiles as a loop (G-o8, §2.10; the shipped build, two runs)** | **42.8–43.5** (down 41.9–42.2) | **2637–2672** | 153 | **406–407 (85%)** |
| H3's attention, transposed | 37.4–38.3 | not recorded | | ~69% |

Two things this table says that the old one did not. **The GEMM runs 3%
under the probe's clock** (2813 against 2899, at 137 W against 110), so
the ceiling it should be held to is 53.9, not 55.5, and the gap is 22–25%,
not 24–30%. And **the pipeline numbers have no clock behind them**: the
DiT's 42.0 and every vertical's TFLOP/s since came from GPU timestamps
without `bench/sysmon.go`, so nobody knows whether a 30 s video forward
holds 2813 or sags to 2600. §6.4 (sustained clocks and power) was never
run.

**Outside calibration.** The best public number on this exact chip is
36.9 TFLOP/s (PyTorch bf16 through ROCm, 64% of the 59.4 spec;
rocBLAS 6–20) per llm-tracker's Strix Halo page, so the tree's 42.0 is
already ahead of the vendor stack here. On the same architecture with a
mature library, rocBLAS on a 7900 XTX reports ~113.6 TFLOP/s of a 123
TFLOP/s dense fp16 peak at 4096³, **~92%** (ROCm/Tensile#1935; whether that
run accumulated in fp16 is not stated). So ~90% is what RDNA3 silicon does
for a hand-scheduled kernel, and 76% is not the architecture's limit.

**The shipped kernels' registers** (`cmd/probe`, `RADV_DEBUG=shaderstats`,
Mesa 26.2.2, 2026-09-29):

| build | wave | VGPRs | spilled | scratch | LDS |
|---|---|---|---|---|---|
| `dit_gemm_wg128x256_bt16_swz8` (WM 4 × WN 8, 2×2 waves, BK_TILES 1, no hoist) | 64 | **240** | 0 | 0 | 0 |
| `gemm_wmma_reg64_bt_hka4` (suite's best) | 64 | 252 | 0 | 0 | 0 |
| `h3_attn_t_qt2_kt4` (H3, VAE-free) | 32 | 256 | **112** | 8 KB | 6 KB |
| `dit_attn_wmma_qt2_kt4_w32` (image, plain) | 32 | 256 | **126** | 16 KB | 9 KB |
| `h3vae_attn_t_hd64_qt2_kt4` (head 64) | 32 | 240 | 0 | 0 | 0 |

At 240 VGPRs a wave64 gets **3 waves a SIMD**, and a 4-wave workgroup
lands 2 or 3 of them on each SIMD depending on whether it is spread over
the WGP's four SIMDs or its CU's two. Either way the matrix pipe on a SIMD
has at most two other waves to run while one waits on a load. Both head-128
attentions spill more than 100 registers to scratch.

## The energy roofline (2026-09-29, §0.7)

The GEMM ladder at N = 2048 drew **136–137 W at every rung from 23 to 39
TFLOP/s**; the winner won by needing less energy a FLOP under the same
cap. So the roofline that binds a real kernel on this part is joules,
not clocks, and it is already in `results/*.csv` (the `power_w` column):

| what | rate | PPT | energy | above ~20 W idle |
|---|---|---|---|---|
| matrix pipe, registers only, 16 chains (`peak`) | 55.5 TFLOP/s | 98 W | **567 GFLOP/J** | 1.4 pJ a FLOP |
| the same at 4 chains (the old probe) | 55.5 | 112 W | 495 | +14 W of loop issue |
| `dot4_int8` / packed fp16 FMA / fp32 FMA | 54 / 25 / 23 T | 118–121 W | 450 / 215 / 189 | |
| MALL-resident copy (1–4 MB) | 800 GB/s | 105–136 W | 6–8 GB/J | ~100 pJ a byte |
| DRAM stream (≥ 6 MB) | 236 GB/s | 136 W | **1.7 GB/J** | **~490 pJ a byte** |
| best suite GEMM | 39.0 TFLOP/s | 137 W | 285 GFLOP/J | |
| a video forward, every kernel | 32.5 TFLOP/s | 149 W | 218 GFLOP/J | |
| LLM prefill, 8,192 tokens, 24 of 48 layers (`cmd/llm -graph`) | 2,680 tok/s (≈1,340 at 48) | **142 W** at 2713 MHz | ≈ 110 mJ a token at 48 layers | |
| LLM decode, 24 of 48 layers (`cmd/llm -gen`) | 63.2 tok/s (≈31.6 at 48) | **91 W** at 2590 MHz | ≈ 2.9 J a token at 48 layers | memory-bound: the bytes, not the clock |

A DRAM byte costs the energy of ~340 matrix FLOPs, a MALL byte ~70.
The matrix cores at full rate are ~80 W over idle, so **at a 150 W cap a
real GEMM has ~50 W for everything else** (bytes at every level, address
arithmetic, issue), which is 100 GB/s of DRAM-priced traffic or ~500 GB/s
of MALL-priced: a kernel at 55 TFLOP/s and AI 43 moves 1.3 TB/s through
the L0/L1, and whether that fits depends on the pJ a byte at *those*
levels, which are not measured (G0b). This is also why a
register-resident loop "reaches peak at ~90 W" wherever it is claimed:
that is the probe, and the other 50–70 W in a real kernel is bytes.

**An outside calibration** (2026-09-29): `peonist-ai/halogen-flash-server`
claims, for the same model on the same part, prefill at 1,567–1,584 tok/s
(8k–32k prompts) and 1,517 at 131k, serial decode 37.6 tok/s at 1.5k and
34.1 at 32k, and 46–56 tok/s with a draft head plus prompt lookup, at
"about 85 W of sustained package power" and a 2,229 MHz median clock
during a 32k prefill. Against ours: prefill 1,213 at 8k and 1,011–1,022
at 128k (1.25–1.5x behind) at 142 W (so ~2x the energy a token); serial
decode 34.4 at depth 0 and 31.5 at 128k (within 10%) at 91 W (about
the same joules a token as their 108–118 W would give); our speculative
loop is parked at 0.95x where theirs is 1.35–1.5x. The repository is
documentation and a container image, not source ("Shell", no kernel
files), so the how is not inspectable. The image was not run here, on
purpose: the hardware is the same part under the same ~150 W cap, so
their 85 W at 2229 MHz is what their prefill *draws*, not a limit it
runs under, and the comparison needs no more than the two power
readings. The prefill gap is the one this vertical can act on: a 2x in
joules a token at a *lower* clock says their prefill moves far fewer
bytes a token through the memory system than ours, which is G5's
territory (4-bit weights straight into the matrix cores without the LDS
round trip) and G6's (the streaming kernels between the GEMMs).

What it changes: an arm that hides latency without moving fewer bytes
(a register prefetch) buys the in-wave stall and no clock; an arm that
moves fewer bytes a FLOP (LDS sharing across a workgroup's waves, bigger
effective tiles, fewer wasted instructions) buys twice. G2's order below
is set by that.

## Why 76% and not 90%: the hypotheses, each with its measurement

Every GEMM lever in the tree so far changed *what* is loaded (intensity,
stride, layout, order). None changed *when*, except §2.7, which changed only
the scheduling and was worth 2.1x. The remaining gap is most likely
scheduling again.

- **H1 — the clock.** 3% is already measured (2813 against 2899). Whether
  a sustained kernel loses more is §6.4, never run. *Measure*: sysmon over
  a 60 s GEMM loop and over one H3 forward, CPU idle and CPU busy (the
  server tokenises and decodes audio beside the device). Cheap, and it
  fixes the denominator for everything below.
- **H2 — latency exposed at the K-tile boundary.** The shipped loop is
  `BK_TILES 1`, no hoist: per K tile a wave issues 4 A + 8 B fragment loads
  and 32 MMAs, then the next tile's loads. 32 wave64 MMAs are ~1,100 clocks
  of matrix-pipe time (8,192 FLOP each at 239 FLOP/clk/SIMD); a fragment
  load from L2 or the MALL is several hundred to ~2,000 clocks. Unless the
  compiler already issues tile k+1's loads before tile k's MMAs, each wave
  stalls once a tile, and with 2–3 waves a SIMD the pipe idles part of the
  time. A 20–25% shortfall is exactly the size of one exposed latency per
  tile at this occupancy. *Measure*: G1 reads the ISA (where the
  `s_waitcnt` sits relative to the `v_wmma` block), then G2 builds the
  register-prefetch arm: tile k+1's 12 fragments loaded into a second
  register set before tile k's MMAs issue. That is +24 VGPRs; at WN = 8 it
  is 264 and spills, so the arm is WM 4 × WN 4 (16 accumulators, 64 VGPRs)
  in a 4-wave 128×128 workgroup, or WM 8 × WN 4, and the AI drop (43 → 32)
  is part of what is measured. **This is the lever the plan is built
  around.** *G2's answer (§2.9): the latency is the cost, but a fragment
  is 8 VGPRs (4x replicated at wave64), ACO copies any fragment carried
  across the loop's back-edge element by element, and the prefetch that
  fits is a whole slab through LDS at wave32: 74 → 80% per clock.*
- **H3 — the issue budget.** A wave64 MMA holds the matrix pipe ~34 clocks;
  if the loop spends more than ~30 VALU/SALU issue slots per MMA on
  address arithmetic, waitcnts and loads, the SIMD is issue-bound, not
  latency-bound. *Measure*: instruction count per K tile from the ISA
  (§2.3's route: the B-layout finding was an instruction count). If it is
  under ~8 non-MMA instructions an MMA, this is not it.
- **H4 — the epilogue.** 32 fp32 tiles a wave stored row-major with no
  overlap inside the wave: 63 MB of C at M = 4096 × N = 3840, 0.27 ms at
  the bus, of a 2.9 ms GEMM. **~9% if nothing overlaps it**, and other
  workgroups' MMAs only overlap it if a second workgroup fits on the CU.
  *Measure*: the `C_F16` build (halves it; already exists, allowed where
  the consumer narrows anyway) against the fp32 store, and a control that
  stores one tile of the 32 (the compiler must not be allowed to drop the
  MMAs: keep the store conditional on a push constant).
- **H5 — the grid tail.** 480 workgroups of 128×256 at the DiT's shape, one
  or two resident a CU, so 6–12 rounds; a shape whose last round is a
  quarter full loses up to a round. *Measure*: rate against M in steps of
  128 rows on one N, K; the tail's size falls out of the sawtooth.
- **H6 — L0/L1 traffic.** At 42 TFLOP/s and AI 43 the fragment loads move
  ~1 TB/s, above the MALL's 965 GB/s pure read, so the L0/L1 hit rate is
  carrying the kernel. The two waves of a workgroup that share an A slab
  each load it. LDS staging lost in §2.1 at AI 8–32 on an unhoisted,
  untiled kernel; it has not been asked at the current tiling. *Measure*:
  only after G2, and only if the prefetch arm lands short of 85%: one LDS
  arm on the *tiled* B (a slab is 512 B contiguous tiles, so the staging
  is a copy, not a gather). *G2: built, and it is the shipped kernel;
  with the staging loads made L0 hits the build reaches 82%, so the memory
  system is now ~3 points and the LDS-side schedule ~18 (G-o8).*
- **H7 — the accumulator chain.** 32 independent accumulators a K tile;
  the MMA latency is hidden. Not it; listed so no one re-tests it.

**Attention** (68–70%, and both shipped head-128 builds spill 112–126):

- **A1 — the spill.** `T QT2 KTIL4` spills 112 and wins; `T QT2 KTIL2` is
  spill-free at 256 and loses 15%; reloading Q each block removes the
  spill and loses 11%. Untried: **a head-dimension split** (two passes of
  64 channels; the head-64 build is spill-free at 240 and runs 34
  TFLOP/s on the VAE), and **wave64** for the transposed kernel (its
  element order is wave32's; the probe says what wave64's is, and at wave64
  the same tile needs half the VGPRs per fragment). *Measure*: the existing
  screen (`TestGPUAttentionScreen`, `H3_ATTN_SPV=qt:ktil:path`).
- **A2 — the online rescale.** Every key block rescales the 8 O^T
  accumulator tiles of a query tile, 8 elements a lane each, plus the
  running sum, whether or not the row max moved. FlashAttention-3's lazy
  rescaling skips it while the max grows by less than a threshold and
  corrects once at the end; exact up to fp32 rounding order. *Measure*:
  the screen, and rms against the plain kernel ≤ 1.1e-4 as M11c gated.
- **A3 — an honest gutting.** M11c's "no LDS round trip" arm read 65.7
  TFLOP/s, above the ceiling, so the compiler dropped MMAs and that arm is
  not a measurement. *Redo*: each phase removed with its result kept live
  (folded into the stored O), so the MMA-only floor of the transposed loop
  is a real number. If it is ~40, the softmax is not the cost and A1 is
  the whole story; if 45+, A2 is worth up to 15%.

**Int8 and Q4 weights** (the LLM's prefill, Kev, embeddings, video's
dequant pass): **G5** builds the B fragment in registers from the bytes
the lane loads itself, using the element order the probe pins (A: lane l
holds row l%16, element e is column e; B: lane l holds column l%16,
element e is row e). No LDS, no barrier, and the plain path stays as the
fallback the probe selects. The MoE Q4 GEMM's 51% is the first customer;
the LLM's prefill graph runs at 9.8 TFLOP/s of useful work (P11), most of
it on these kernels.

## Decisions (so future sessions don't relitigate)

1. **The ceiling is per clock, and the clock is set by energy.** A
   kernel's utilisation is FLOP/clk/CU against the probe's 480, at the
   clock sysmon saw *during that kernel*. Every timing this vertical
   records carries its sclk and package power (PPT), and every kernel
   table carries **GFLOP/J** (or GB/J) beside its rate. The package is
   capped near 150 W (§0.6), so rate = cap ÷ energy per operation: a
   kernel that spends fewer joules a FLOP runs at a higher clock and is
   faster for that reason alone. Power efficiency is not a second
   objective here; it is the same one (§0.7).
2. **The suite is the ablation instrument; the vertical kernels are where
   a winner ships.** `gemm_wmma.comp` + `results/gemm_wmma.csv` stay a
   measured artifact of a kernel that does not move under them (the
   header in `dit_gemm.comp` says why). A lever is proved in the suite on
   the `shapes` family (the models' real M, N, K), then ported to
   `dit_gemm.comp` as a `-D` and screened in the verticals with their own
   tools: `TestGPUShapes`/`H3_PROFILE` (video), the zimage DiT ladder
   (image), `cmd/kevload`, the LLM prefill bench.
3. **Bit-identical or the vertical's own gate.** A scheduling change
   (prefetch, tile order, epilogue store) makes the same MMAs in the same
   order and must be byte-identical (`TestGPUForward` asserts it, as
   M11d's split did). A numeric change (lazy rescale, fp16 C) goes through
   the consumer's gate (rms ≤ 1.1e-4 on a context; the teacher-forced step
   bounds), never a max-abs on a free run.
4. **Read the ISA before building the arm.** `cmd/probe` is back; a
   hypothesis about scheduling or issue is checked in the disassembly
   first, because §2.3 and §6.1 showed that is where the answers were.
5. **Wave size and element order are per kernel, probed, with a plain
   fallback.** Nothing written against a lane layout ships without
   `checkCoopMatLayout`'s equivalent selecting it at first use.
6. **A finding is filed under the next free `§N.M`** (§2.9 the prefetch,
   §3.8 the attention rescale, §2.10 the register dequant …) with its
   file in `research/`, and the G-stage here points at it. Numbers are
   never reused.
7. **Stop when three consecutive well-formed hypotheses move nothing by
   2%.** The exit is a measured plateau, not a percentage: 85% is the
   target, not the gate.

## The stages

| # | Stage | State |
|---|---|---|
| G0 | Instruments: `cmd/probe` restored; the `peak` probe at wave32 and at 16 chains (is 478/clk the hardware or the probe?); sysmon around every in-pipeline timing this vertical takes; §6.4's sustained-clock run (H1) | **done 2026-09-29** (§0.6): the ceiling is the hardware's, 478–482 FLOP/clk/CU at every chain count and wave size; **a video forward runs at 2600–2650 MHz and ~149 W**, so its ceiling is 49.9 TFLOP/s and the shipped GEMM is at 74% of it; 32 busy CPU threads take the GPU to 1350 MHz and a forward to 1.67x |
| G0b | The energy roofline: pJ a byte at L0, L1, L2, MALL and DRAM (the `bandwidth` family below its 2 MB floor, with the power column), pJ a FLOP per instruction class (done, §0.7), the sensor settled (PPT is the package; the CPU's share from the RAPL counter in `/sys/class/powercap`, GPU = PPT − CPU), and GFLOP/J in every profile this vertical prints | **§0.7 opened 2026-09-29** from the recorded rows; the cache levels and the sensor split are open |
| G1 | The shipped GEMM's ISA: one K iteration's schedule (loads, waitcnts, MMAs), instructions per MMA, occupancy as the driver places it (H2, H3) | **done 2026-09-29** (§6.5): no prefetch across K tiles, 85 address instructions recomputed an iteration; **the rate is flat from 1 to 3+ workgroups a CU**; with every load an L0 hit the GEMMs reach 82–85% per clock, so the memory system is ~10 points and the in-wave schedule ~16 |
| G2b | G-o8, the LDS-side points of `lds_w32`: the parts priced one at a time with `wmma_issue_probe.comp` through `cmd/bench peak` (`PEAK_SPV`), then the arms on H3's shapes | **done 2026-09-29** (§2.10): **every VALU instruction costs the matrix pipe a clock**, and ~40 `v_swap`/`v_mov` a tile from the register allocator were 7% of it; the slab's two K tiles kept as a loop (`KT_LOOP` + `STAGE_RW`) removes them, bit-identical: **projections 40.4 → 43.5, 80 → 85% per clock, a 480p forward 27.3 → 26.7 s**; shipped under the same build name to all nine hosts. Priced and left: the barrier's lockstep ≤ 9 points, the LDS reads ~4, the global side ~3; the 16-lane exchange, `LOAD_ORDER` and a 128×128 workgroup measured dead |
| G2 | The register-prefetch GEMM (H2): `PREFETCH=1` in `gemm_wmma.comp` and `dit_gemm.comp`, the tile geometries it fits in, screened on the `shapes` family and on the DiT's and H3's shapes; bit-identical | **done 2026-09-29** (§2.9): not registers, **LDS**: the K slab staged a slab ahead through 24 staging registers into a double-buffered LDS plane, at **wave32** (a fragment is 8 VGPRs and 4x its bytes at wave64, 2x at wave32), 4×4 tiles, eight waves. Bit-identical; H3's projections **37 → 41 TFLOP/s**, the split down projection **29 → 41**, a 480p forward **29.9 → 27.1 s**; 74 → 80% per clock. Shipped for H3; the other eight hosts are G8 |
| G3 | The epilogue and the tail (H4, H5): the store's cost priced with a live control; `C_F16` and the fragment-tile C for attention's operands (stage 4's open item) where the consumer allows; the M sawtooth | |
| G4 | Attention (A1–A3): the honest gutting first, then a head-split and a wave64 transposed build, then lazy rescaling; every build through the H3 screen and the image DiT's; carried to the VAE, Kev, OCR, ACE (every `dit_attention_wmma.comp` user still pays the LDS round trip M11c removed) | |
| G5 | Int8/Q4 B fragments built in registers from the probed layout: the MoE Q4 GEMM (51% → ?), `llm_gemm`'s Q8 arm, `kev_gemm_q8_glu`, and H3's dequant pass folded away | **G5a done 2026-09-29** (§2.11), and not as written: the controls on the MoE up GEMM say the LDS round trip is ~8% and the loop without its stagings already runs at the pipe; the cost was the per-step chain of divergent bank loads, re-reading each block's lines two to eight times. **The K loop as a software pipeline** (`-DPIPE=2`: the next step's bytes fetched behind this step's MMAs, a header once a super-block, a nibble group once per two steps) is bit-identical and takes `moe.up` **11.5 → 6.8 ms a layer at 2048 tokens (1.68x), 2.01x at 512, 1.57x at 4096**; shipped on the seven Q4_K up builds. **G5b the same day:** the down mode's pipeline moves Q5_1 by nothing and IQ4_NL (served) by 1.05x at 2048 / 1.20x at 512, shipped for IQ4_NL alone. In the whole model, prefill **1194 → 1407 tok/s at 2048 (1.17x)**, 1317 → 1494 at 4096, 1306 → 1459 at 8192. Left: the m4 rung's 1.90x padding, the Q5_K up build, the down mode's A fragment loads; the register-fragment idea itself has nothing to buy on this kernel |
| G6 | The streaming kernels: a bandwidth column in every vertical's profile, the ones under 200 GB/s listed and fixed or fused (`hc.cn` at 134 first) | |
| G7 | The budget: what the server's CPU work beside a device job costs in GPU clock (the audio decode's 32 threads, staging on 32 cores, tokenising; G0 measured 32 busy threads at 1.67x), whether a thread cap or a CPU power limit is a net win, and the sampler folded into `bench/sysmon.go` around the vertical tests | G0 measured the extremes; the server's own load is open |
| G8 | Carry-in: each vertical's screen re-run on the new builds, the numbers into VIDEO.md M11, the image and LLM records, and TODO.md's table | continuous. **G2 carried to all nine hosts 2026-09-29** (table below): every gate passes; the image step 2179 → 2058 ms, the VAE decode 37.1 → 35.5 s, Kev's fp16 pass at 494 tokens 154 → 132 ms; **the build loses below ~1000 rows** (a 30-workgroup grid), so `ace/dit` keeps the wave64 build up to 1024 rows and the ladders' schedules already keep it off short inputs |

**Order:** G0 (a day: it fixes the denominator), G1 (a session: it decides
between H2 and H3 before anything is built), G2 (the big one), then G4 and
G5 in either order, G3 and G6 as the profiles point, G7 when a long run is
happening anyway, G8 after every win.

**Gates, per stage:** G0 a ceiling table reproduced twice with clocks. G1 a
written schedule of one K iteration from the disassembly, with the
instruction counts. G2 the suite's best and `dit.qkv`'s rate, two runs
each, bit-identical C, and the register/spill line from `cmd/probe`. G3
the priced store and the sawtooth, both with clocks. G4 the H3 screen
table and rms against the plain kernel, plus `TestGPUForward`/`TestGPURun`
unchanged. G5 bit-identical to the LDS path on the MoE bench, and the
prefill rate. G6 a table of every streaming kernel over 1 ms in a block
with its GB/s. G7 the trace.

### G0 — instruments and the ceiling (2026-09-29)

Filed as **§0.6** in `research/0-measurement-validity.md`.

**The ceiling is the hardware's.** The `peak` family gained the WMMA loop
at 8 and 16 chains and at wave32 (`-DACC`, `-DWAVE`; `bench/ops_peak.go`
pins the size), so G-o1 could be answered instead of assumed:

| probe | best | sclk | W | FLOP/clk/CU |
|---|---|---|---|---|
| wmma_fp16, 4 chains, wave64 (the old probe) | 55,554 | 2890 | 112 | 481 |
| 8 chains | 55,615 | 2890 | 99 | 481 |
| 16 chains | 55,524 | 2890 | 98 | 480 |
| wave32, 4 chains | 55,436 | 2892 | 98 | 479 |
| wave32, 8 chains | 55,405 | 2896 | 100 | 478 |
| wmma_int8 | 55,538 | 2879 | 98 | 482 |

Flat to 1% across all of them, at **480 of the spec's 512**, which is
15/16 exactly: the matrix pipe's dense rate on this silicon, not a probe
artefact. 55.5 TFLOP/s at 2899 MHz stands, and every rate below is read
against 480 FLOP/clk/CU at the clock the kernel ran at.

**The clock under a real kernel.** A 100 ms sampler on
`hwmon/freq1_input` and `power1_average` around `TestGPUShapes`
(`H3_SHAPES=1 H3_PROFILE=1`, int8, the machine otherwise idle):

| what | sclk (mode) | package W | a 480p forward | a 768p forward |
|---|---|---|---|---|
| `peak` probe | 2890–2899 | 100–112 | | |
| suite GEMM, 20 ms bursts (`results/gemm_wmma.csv`) | 2813 | 137 | | |
| **H3 forward, CPU idle** | **2600–2650** | **149** | 30.0 s | |
| H3 768p forward, CPU idle (sustained ~2 min) | 2550–2600 | 142–155 | | 106.7 s |
| **H3 forward, 32 busy CPU threads** | **1300–1400** | 121 (GPU side) | **50.1 s** | **190 s** |

Three things. **The package is power-limited at ~150 W**, and a GEMM's
memory traffic costs ~40 W over the register-only probe, so the clock
sags 9–10% under any real kernel: the ceiling during a video forward is
**49.9 TFLOP/s**, not 55.5, and the shipped GEMM's 37 TFLOP/s there is
**74% per clock** (353 FLOP/clk/CU), the attention's 37.3 is 75%, the
whole forward's 32.5 is 65%. **The 768p run sags further** (2550–2600),
so long forwards are ~2% slower per FLOP than short ones for the clock
alone. And **the CPU shares the budget**: 32 busy host threads halve the
GPU clock and make a forward 1.67x slower. That is the mechanism behind
VIDEO.md M11f's "+2.4 s of device time while the audio decodes on 32
threads": not memory contention, the power budget. Anything the server
runs on the CPU beside a device job costs GPU clock in proportion to its
power, which is a serving rule this repo did not have.

**Instruments.** `cmd/probe` is back (`RADV_DEBUG=shaderstats|asm`),
`H3_GEMM_SPV=path` runs a big-GEMM build from disk in `TestGPUShapes`
(and times it even when it is wrong by construction), and the sampler
is `scratchpad/sample.sh`-shaped: 100 ms of `freq1_input`,
`power1_average`; a phase is the samples between two timestamps. It is
not in the tree yet; G7 should fold it into `bench/sysmon.go` around the
vertical tests.

### G1 — the shipped GEMM's schedule (2026-09-29)

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

### G2 — the prefetch that fits (2026-09-29)

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

### G8 — the carry-in (2026-09-29)

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

### G5a — the MoE up GEMM's K loop as a pipeline (2026-09-29)

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

**Next in G5:** the up rung's padding factor (1.90x executed rows at
m4, 2048 tokens: the largest term left in the up kernel), the Q5_K up
build (layer 2), and the down mode's A side.

## How to run

- The suite: `go run ./cmd/bench -h`; families `peak`, `gemm_wmma`,
  `shapes`, `stride`, `moe`, `bank`. Results append to `results/*.csv`
  with sclk and power columns; the harness warms the clock to 97% first
  (§0.2).
- The ISA: `RADV_DEBUG=shaderstats go run ./cmd/probe shaders/X.spv [32|64]`
  and `RADV_DEBUG=asm …` (`MESA_SHADER_CACHE_DISABLE=1` for a repeat).
  Builds are `go generate ./shaders` lines in `shaders/shaders.go`; `.spv`
  is gitignored.
- A GEMM build from disk through H3: `H3_GEMM_SPV=path` (`H3_GEMM_WAVE=32`
  to pin the wave, `H3_GEMM_KRANGE_SPV=path` for the split down
  projection's companion), `H3_GEMM_REF=file` the bit-identical gate
  (written by the first run, compared by every later one); a loaded
  build that is wrong by construction (`L0_LOADS`, `STAGE_L0`) is still
  timed. The clock and power beside it: a 100 ms loop over
  `hwmon/freq1_input` and `power1_average` (G7 will fold it into
  `bench/sysmon.go`).
- The vertical screens: VIDEO.md M7/M11c/M11h (`H3_SHAPES=480 H3_PROFILE=1
  H3_BANK=q8`, `H3_ATTN_SPV`, `H3_CHUNK`), the image DiT's ladder in
  `research/stage-4-dit-graph.md`, Kev's `cmd/kevload`, the LLM's
  `cmd/llm -bench` (memory: it needs `LLM_BANK_CACHE`).
- A MoE GEMM build from disk: `LLM_MOE_SPV=up_q4k_m4=path.spv[,…]`
  (`LLM_MOE_SPV_WAVE=32` for a wave32 build) on `cmd/llm -moe` and on the
  `llm` tests, which run from `llm/`; `cmd/llm -moe -model <shard>
  -tokens 2048 -iters 60` is ~2 s of device time an arm, so the clock is
  the median of the samples above 100 W.
- Never edit a shader during a sweep; build the test binary to the
  scratchpad first, and check `ps` for another device run.
- Driver pinned in the record: Mesa 26.2.2 (RADV), glslc 2026.3, Vulkan
  1.4.354, kernel 7.1.8. A driver upgrade re-runs G0.

## Open questions

- ~~**G-o1.** Is 478 FLOP/clk/CU the hardware's dense WMMA rate or the
  probe's?~~ **The hardware's** (G0): 478–482 at 4/8/16 chains and at
  wave32, 15/16 of the spec's 512.
- ~~**G-o2.** Does RADV place a 4-wave workgroup across the WGP's four
  SIMDs or the CU's two?~~ **Moot for this kernel** (G1): its rate is
  flat from ≥ 3 workgroups a CU down to 1. ACO's "6 subgroups a SIMD" at
  240 wave64 VGPRs is still unexplained and still not relied on.
- ~~**G-o3.** Does ACO already hoist the next tile's fragment loads above
  the current tile's MMAs?~~ **No** (G1): the loop is address math, 24
  loads, a waitcnt ladder, 32 MMAs, branch.
- ~~**G-o5.** Which cache is the ~10 memory points (G1)? Latency (then G2's
  prefetch takes it) or L2/MALL bandwidth (then only traffic per FLOP
  does)?~~ **Latency, mostly** (G2): with the slab fetched a slab ahead
  the memory side is ~3 points (`STAGE_L0` control), and the down
  projection's MALL overflow went with it because the slab is read once
  a workgroup.
- ~~**G-o8.** The LDS-side schedule is the remaining ~18 points of
  `lds_w32` (§2.9): 64 `ds_load_b64`, 6 `ds_store`, a barrier and a
  `lgkmcnt` ladder a slab. Which of those is it?~~ **Priced, one part at
  a time** (§2.10, `wmma_issue_probe.comp`): the reads 3%, a barrier
  alone nothing, the two together 9% (the lockstep burst), the LDS rate
  not a limit; and the biggest part was none of them: **~40 register
  moves a tile from the allocator, at a clock of matrix pipe each**,
  removed by keeping the K tiles as a loop (+5 points, shipped). Left:
  the barrier's lockstep (≤ 9, bounded by `LDS_NOBAR`), which needs a
  slab structure that keeps the waves or the two workgroups on a WGP out
  of phase; the LDS geometry admits no third tile a slab and a 128×128
  workgroup loses to the global side first.
- **G-o9.** The wave32 4×4 tile's non-MMA budget is a few instructions a
  MMA (§2.10: ~3% each). Where else in the tree does an unrolled
  cooperative-matrix loop carry a move storm? The attention kernels
  (`h3_attn_t`, `dit_attention_wmma`) spill as well as move; G4 should
  count `v_swap`/`v_mov`/`v_cndmask` a MMA in their disassembly before
  building anything.
- **G-o7.** What does a byte cost at L0, L1 and L2? DRAM is ~490 pJ and
  the MALL ~100 (§0.7); the GEMM's 1.3 TB/s of fragment traffic lives in
  the levels below, and their price decides whether LDS sharing (which
  moves bytes from L1 to LDS reads) is a saving or a wash. G0b.
- **G-o6.** The package budget is ~150 W and the CPU takes from it (G0:
  32 threads halve the GPU clock). What does the server's actual CPU
  work beside a device job cost (the audio decode's 32 threads, staging
  on 32 cores, tokenising), and is a lower thread cap or a lower CPU
  power limit a net win for the device? G7.
- **G-o4.** What is wave64's accumulator element order? The probe reads
  it; if it is the wave32 order with rows split across lane halves the
  transposed attention has a wave64 build for free (G4).

## Handoff

**2026-09-29, session 6: G5a, the MoE up GEMM 1.68x.** G5 was started as
written (register-built Q4 fragments) and measured out of it before a
line of it was built: the controls on `llm_moe_gemm.comp`'s up mode
(`research/2.11-…`) say the LDS round trip is ~8 points and the loop
without its stagings already runs at the pipe, while the bank's fetch —
a dependent, divergent load chain in front of every K-step, each block's
lines re-read two to eight times — is 25–30. The K loop as a software
pipeline (`-DPIPE=2`: the next step's bytes behind this step's MMAs, a
header once a super-block, a nibble group once per two steps) is
bit-identical and takes `moe.up` **11.5 → 6.8 ms a layer at 2048 tokens,
2.01x at 512, 1.57x at 4096**; shipped on the seven Q4_K up builds, the
whole `TestMoEGPU*` suite passing. Tools added: `LLM_MOE_SPV` /
`LLM_MOE_SPV_WAVE` (this kernel's `H3_GEMM_SPV`), the `MOE_*` screening
knobs and `-DWAVE` on `llm_moe_gemm.comp`. Two compiler facts: the
one-set pipeline's fetched registers get copied inside the MMA block
with `vmcnt` waits between the MMAs (so `PIPE=1` alone is 4–14%), and a
two-set (ping-pong) form spills at m4 and loses where it fits. In the whole model, prefill **1194 → 1407 tok/s at 2048 (1.17x)**, 1317 →
1494 at 4096, 1306 → 1459 at 8192 (same-hour A/B, twice); TODO.md's
table carries it. The down mode's pipeline (G5b) was built for all four
block-32 formats and measured: Q5_1 flat, IQ4_NL 1.05x at 2048, shipped
for IQ4_NL alone. **Next:** the m4 rung's 1.90x padding at 2048 (the
largest term left in the up kernel), the Q5_K up build, the down mode's
A fragment loads; and G4 or G6 as KERNELS.md's order says.

**2026-09-29, session 5: G-o8 answered, the GEMM at 85% per clock.**
The remaining points of `lds_w32` were not the LDS's: a probe that adds
the kernel's parts back to the register-only WMMA loop
(`shaders/wmma_issue_probe.comp`, `PEAK_SPV` on `cmd/bench peak`) shows
**every VALU instruction costs the matrix pipe about a clock** on this
part, the GEMM's LDS reads cost 3%, a barrier alone nothing and the two
together 9%. The shipped build's disassembly carried ~40 `v_swap`/`v_mov`
a tile (the allocator assembling the second K tile's fragments under
hoisted staging loads), ~7% of the pipe. Keeping the slab's two tiles as
a loop (`-DKT_LOOP=1 -DSTAGE_RW=1`, now in the `dit_gemm_wg128x256_lds_w32`
generate lines) removes them, bit-identical: **projections 40.4 → 43.5
TFLOP/s, 80 → 85% per clock, a 480p forward 27.30 → 26.71 s**; `research/
2.10-…` has the ladder. Measured dead the same session: the 16-lane
exchange that halves LDS bytes (`v_permlanex16` is VALU: −5%), a finer
load order (a wash), a four-wave 128×128 workgroup (−11%). **Left in the
GEMM**: the barrier's lockstep, bounded at ≤ 9 points by the no-barrier
arm, with no LDS room for a deeper slab. **Next:** G4 (attention, whose
head-128 builds spill 112–126 *and* will carry the same kind of moves:
count them first, G-o9), or G5 (int8 in registers, now knowing each VALU
op of a dequant costs a matrix clock: the budget is a few instructions a
MMA, which argues for `v_perm`/`v_cvt_pk` sequences over per-element
work). Gate chain for the new build (every host's own gate) ran this
session; the table in G8 stands for the per-host numbers, this build is
the same kernel one loop deeper.

**2026-09-29, session 4: G8 carried to every host.** All nine hosts run
`dit_gemm_wg128x256_lds_w32` pinned to wave32, every gate passing (table
in G8): image step −5.6%, VAE decode −4.3%, Kev's fp16 pass 1.16x at 494
tokens; the build loses below ~1000 rows, so `ace/dit` keeps the wave64
build up to `bigW64Rows` (1024), and the VAE's bias-padded K rounds to
32. Also added: `ACE_DIT_BIG=w64|small` on `TestGPUStepTiming`; the
served binary still needs a redeploy. **Next:** G-o8 (the LDS-side 18
points), or G4 (attention), or G5 (int8 in registers: Kev's fp16 rung
now beats its int8 one at 494 tokens, which says the int8 kernels are
where the LDS round trip still costs).

**2026-09-29, session 3: G2 done and shipped for H3.** The prefetch is
LDS at wave32, not registers: `dit_gemm_wg128x256_lds_w32` (+ `_krange`)
is bit-identical to the wave64 build and takes H3's projections 37 → 41
TFLOP/s, the split down projection 29 → 41 and a 480p forward 29.9 →
27.1 s (two runs, 0.2%), 74 → 80% per clock; `research/2.9-…` has the
twelve-rung ladder. Three facts for anyone touching the GEMM next: a
wave64 fragment is 8 VGPRs (4x replicated), ACO cannot carry a fragment
across a loop back-edge without copying it element by element (so
register pipelining of coopmat operands is dead on Mesa 26.2.2), and
LDS fragment reads are `ds_load_b64` (pitch 36 halves). Tools: `SR`,
`PIPE`, `DB`, `LDS_STAGE` (`A_LPITCH`/`B_LPITCH`, `BK_TILES` 1–2),
`STAGE_L0`, `LOAD_ORDER` in `dit_gemm.comp`; `H3_GEMM_WAVE`,
`H3_GEMM_KRANGE_SPV`, `H3_GEMM_REF` in `TestGPUShapes`; a `wave` field
on H3's GEMM variant table. **Next:** G8 for the eight other hosts of
the 128×256 kernel (pin wave32, K % 32, each vertical's screen; the
image DiT's `dit.qkv` at M = 4096 is the number the record quotes), then
G-o8 or G4 (attention, whose head-128 builds spill 112–126 and whose
fragments are the same 8 VGPRs). The suite (`gemm_wmma.comp`) was not
ported; the lever was proved on H3's shapes with the bit-identical gate
and the clock instead, which G2's gate asked for in the other order.

**2026-09-29, session 2: G0 and G1 done.** The ceiling is the hardware's
(480 FLOP/clk/CU, flat across chains and wave size). **The clock under a
real kernel is 2600–2650 MHz at ~149 W**, so a video forward's ceiling is
49.9 TFLOP/s and the shipped GEMM is at 74% of it, the attention 75%;
32 busy CPU threads halve the clock (a forward 1.67x slower), which is
M11f's mechanism. The GEMM's loop has no cross-tile prefetch and
recomputes 85 address instructions an iteration; its rate is flat from
1 to 3+ workgroups a CU; with every load an L0 hit it reaches 82–85%
per clock. So: ~10 points memory, ~16 points in-wave schedule. **Next is
G2** in the order above: strength-reduced addresses, then a one-tile
register prefetch, screened with `H3_GEMM_SPV` at 480p and in the suite,
bit-identical via `TestGPUForward`. Tools added: `cmd/probe` (restored),
`-DWAVE`/`-DACC` peak variants, `-DLDS_PAD` and `-DL0_LOADS` screening
knobs in `dit_gemm.comp` (never shipped builds), `H3_GEMM_SPV`. The
scratchpad sampler is not in the tree (G7).

**2026-09-29, session 1: the vertical reopened.** Record assembled from
`research/ideas.md`, `stage-4-dit-graph.md`, `stage-3-dit-attention.md`,
VIDEO.md M11c–M11h and `results/*.csv`. New facts today: the suite's
best GEMM ran at **2813 MHz and 137 W** where the peak probe ran at 2899
and 110, so the per-clock utilisation is 72.6% (39.0) and 76–78% (42.0,
clock unrecorded); the shipped GEMM is **240 VGPRs, no spill, no LDS,
`BK_TILES 1`, no hoist**; both shipped head-128 attentions spill 112–126
VGPRs; `cmd/probe` was deleted on 2026-09-22 by accident and is
**restored** (`cmd/probe/main.go`, from `501053a^`). Outside numbers:
36.9 TFLOP/s is the best public figure on this chip, 92% of peak is what
rocBLAS reaches on a 7900 XTX. Nothing else built. Next: G0, then G1.
