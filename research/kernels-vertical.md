# KERNELS — the matrix cores to their ceiling

> **Moved to `research/` on 2026-10-02** — this was `KERNELS.md` at the repo
> root, and code comments citing `KERNELS.md` resolve here. The vertical is **not closed**: [`../TODO.md`](../TODO.md)
> carries its open items in summary, and § Handoff below is still where a
> session resumes. The long G-stage write-ups are broken out into `g*.md`
> files beside this one, linked where each stage's section was; G-stages,
> G-o numbers and decisions 1–7 still resolve here.

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
| WMMA flash attention | `shaders/h3_attn_t.comp` (transposed, wave32; since G4b eight waves a workgroup sharing each key block through LDS), `shaders/dit_attention_wmma.comp` (plain) | **40.8–40.9** at 2717 MHz in a 480p forward, **78%** per clock (G4b, §3.9; the one-wave build 37.1–37.8, 73%: G4, §3.8) | ≥ 80% |
| int8 / Q4-weight WMMA GEMM | `shaders/llm_moe_gemm.comp`, `shaders/llm_gemm.comp`, `shaders/kev_gemm_q8_glu.comp`, `dit_dequant_q8` + fp16 GEMM (video) | MoE Q4 up: 22 → **37 TFLOP/s executed** at 2048 tokens (§2.11, ~75% per clock; 51% in §2.2), down 19–21, the rest unmeasured against the ceiling | the fp16 GEMM's rate at the same M, N, K |
| streaming kernels (norms, packs, gates, SwiGLU, copies) | everywhere | 6–14% of every block; H3's SwiGLU and v pack fused into the GEMM's epilogue (G3, §2.14); **G6 (§5.4): the packs 93–151 → 173–209 GB/s, the norms 181–193 → 205–217, the gates 193–213, `hc.cn` 184 → 200 (its "134" left the write-back out)** | ≥ 200 GB/s each, or fused away |

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
| … the same build with no C store at all (`STORE_TILES=0`, G3, §2.14: wrong by construction) | 47.1 | **2900** | **130** | 406 (85%): the store costs the pipe nothing, the 9% is clock |
| H3's attention, transposed (G4, §3.8, three runs) | 37.3–37.8 (480p), 35.2 (768p) | 2650–2690 / 2600 | 154 | 350–355 (**73%**); the honest floor 81%, with L0 hits 87–90% |
| **… its K/V blocks through LDS, eight waves a workgroup (G4b, §3.9; the shipped build, two forwards)** | **40.8–40.9** (480p), 38.6 (768p) | **2717** / 2704 | 154 | **375–376 (78%)**; its controls: softmax 6%, global side ≤ 6%, barriers 3% |

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
| G3 | The epilogue and the tail (H4, H5): the store's cost priced with a live control; `C_F16` and the fragment-tile C for attention's operands (stage 4's open item) where the consumer allows; the M sawtooth | **done 2026-09-30** (§2.14): **H4 dead on the pipe** — a live control that keeps every MMA and skips the stores (`STORE_TILES`) runs the same 404–409 FLOP/clk/CU as the shipped build at every arm; its +9% in TFLOP/s is the clock (2900 MHz at 130 W with every downstream kernel on zeros, against 2671 at the 154 W cap), so the fp32 C is paid for in energy, not issue slots. **H5**: `TestGPUGEMMSawtooth` — the plateau by ~640 rows on gate and ~1300 on q and down, then a ±5% / ±2% / ±3% swing with the fractional round of 40; the served 480p chunks sit at its top; under ~600 rows the ramp is 2x (G8's "loses below ~1000 rows"). **The consumers fused**: the v projection stored as the attention's fragment tiles (`C_PACK` on the LDS build, the A pad rows zeroed by `zero_f16.comp` so the pad keys stay +0) and gate|up as one GEMM over a host-interleaved weight (`projGateUp`, 64-row groups) with the SwiGLU as its epilogue (`C_SWIGLU`, kev's K7.6 pattern), both **bit-identical** to the passes they replace (the morning's `H3_GEMM_REF`, `TestGPUForward`'s run-time controls): the `swiglu` (570 ms) and `pack v` (244 ms) dispatches gone, **a 480p forward 26.22 / 26.33 → 25.59 / 25.50 s (1.03x)** at the same clock (two controls 26.26 / 26.26); `TestGPUForward` and `TestGPURun` pass; the fused GEMM and the fp16 stores run at the shipped rate. **The SwiGLU epilogue carried to `qimage/dit` the same day** (`projW13`, `qwen.InterleaveGLU`; bit-identical to its control on both banks, the 256² oracle passing): a 1024² cached step **2044 / 2046 → 1929 / 1968 ms wall (1.04–1.06x)** on the served int8 bank, 1.99 → 1.905 s per dispatch, the 83 ms `swiglu` gone; the v store not carried there (a 19-row prefix puts the fresh rows mid-tile, for 1.2%). **Carried to `ace/dit` and the video VAE the same day** (G8's section below): a music forward 1.07–1.10x, **a 480p VAE decode 34.4 → 30.0 s (1.15x)**, both bit-identical across binaries; the speech encoders priced and left (their FFN is not gated and the two passes are 2.3% of a 13.5 ms encoder). `C_F16` has no other consumer: o and down feed the fp32 residual |
| G4 | Attention (A1–A3): the honest gutting first, then a head-split and a wave64 transposed build, then lazy rescaling; every build through the H3 screen and the image DiT's; carried to the VAE, Kev, OCR, ACE (every `dit_attention_wmma.comp` user still pays the LDS round trip M11c removed) | **done 2026-09-30** (§3.8): the honest floor (softmax gutted, results live) is **81% per clock with real loads, 87–90% with L0 hits**; the softmax is 6–7 points (the row max's 16-deep chain 7, the rescale 3), the K/V loads 11–12. Five exact, bit-identical arms (peeled tail, Q^T in LDS instead of the 112-register spill, a tree max, an exact lazy rescale, P·V a tile at a time) all land inside ±2% of the shipped build: the allocator's ~200–390 moves a block at 256 VGPRs change with every edit and cost what the edit saves. Head split (1.5x MMAs) and wave64 (VALU two passes) dead by arithmetic. **Carried to the image DiT, opt-in** (`QIMAGE_ATTN_T=1`; `qimage_attn_t_qt2_kt4`, `STORE_TAIL=0` with a per-query bound on the last tile, the fp16 context written in place): attention 248 → 232 ms a 1024² step (35.7 → 38.3 TFLOP/s), a cached step 1.908 → 1.876 s, the t2i oracles passing; the edit oracle's prefill reads 7.9e-2 against its 2e-2 bound where a plain-family fp16 perturbation of the same prefix rows reads 1.7e-2 — the kernel is exact on identical inputs at every block (rms 8e-5 / 1.1e-4), the gate is near its sensitivity floor for edits, and it is the vertical's gate, so the default stays plain. Left: G4b, the K/V block staged through LDS for a workgroup's waves (§2.9's lever; the 12 memory points less 6–9 of LDS reads and barrier, ≤ 2% of a forward) |
| G4b | The attention's K/V block staged through LDS for a workgroup's waves (§2.9's lever on the key side) | **done 2026-09-30** (§3.9): `h3_attn_t_kvlds_w8_kt4` — eight wave32 waves of one query tile, a key block's K and V fetched once a workgroup as four `uvec4` a lane, half a block ahead, into one LDS plane each (pitches 132 and 68 halves, 34.8 KB), two barriers a block, spill-free, with §3.8's `LAZY` and `TAIL_SPLIT` (dead on the spilled build, 6–10% here). **Bit-identical** to the one-wave build (`H3_GEMM_REF` at both shapes, the screen per arm): the kernel **195 → 175 ms at 480p (1.11x), 1.195 → 1.046 s at 768p (1.14x)**, in the forward 37.2 → 40.8 TFLOP/s, **73 → 78% per clock; a 480p forward 25.22 → 24.31 s, 768p 95.9 → 92.5 s (1.037x)** with the clock 2657 → 2717 MHz. The first cut tied: staging registers carried across the back-edge are copied there behind `vmcnt(0)`, so its block-ahead fetch was waited out every block. `KV_DEEP`, `KV_VTOP`, `KV_BSOFT`, sixteen waves, `QREG=0`, `KTIL2` all lose. The default in `h3/dit` (`H3_ATTN_LDS=0` the control); the VAE's head-64 build priced and left (6.10 against 6.22 ms); the image DiT's stays behind its edit gate |
| G5 | Int8/Q4 B fragments built in registers from the probed layout: the MoE Q4 GEMM (51% → ?), `llm_gemm`'s Q8 arm, `kev_gemm_q8_glu`, and H3's dequant pass folded away | **G5a done 2026-09-29** (§2.11), and not as written: the controls on the MoE up GEMM say the LDS round trip is ~8% and the loop without its stagings already runs at the pipe; the cost was the per-step chain of divergent bank loads, re-reading each block's lines two to eight times. **The K loop as a software pipeline** (`-DPIPE=2`: the next step's bytes fetched behind this step's MMAs, a header once a super-block, a nibble group once per two steps) is bit-identical and takes `moe.up` **11.5 → 6.8 ms a layer at 2048 tokens (1.68x), 2.01x at 512, 1.57x at 4096**; shipped on the seven Q4_K up builds. **G5b the same day:** the down mode's pipeline moves Q5_1 by nothing and IQ4_NL (served) by 1.05x at 2048 / 1.20x at 512, shipped for IQ4_NL alone. In the whole model, prefill **1194 → 1407 tok/s at 2048 (1.17x)**, 1317 → 1494 at 4096, 1306 → 1459 at 8192. **G5c 2026-09-30** (§2.12): the padding. The alignment is one fragment, the record's real-row count picks one of WM copies of the K loop (`SHORT` builds, bit-identical): the padding factor at 2048 tokens 1.90x → 1.19x, `moe.down` **1.23x** (1.28x on the served IQ4_NL), `moe.up` 1.11x (any third copy of its loop spills, so it keeps the 2- and 4-tile ones and executes as 1.41x), the block 1.13–1.14x; whole-model prefill **1417 → 1514 tok/s at 2048 (1.07x)**, 1500 → 1549 at 4096, 1461 → 1481 at 8192; the m2/m4 boundary moves from 1024 tokens to 256. Left: the slab unpack, now the largest term in the up projection (~300 VALU a lane a step at a matrix clock each), the Q5_K up build, the down mode's A fragment loads. **G5d 2026-09-30** (§2.13): the unpack read from the ISA is three VALU an element, not five (the compiler already fuses affine and half conversion into `v_fma_mix`); **the nibble read as an f16 denormal** (`NIB_F16`, a masked halfword *is* the half `nib × 2^-24`, the scale carries the power of two) takes the extract-and-convert pair away, bit-identical, the four-tile step's non-MMA VALU 331 → 259 and `moe.up` **1.02x** at 512–4096 tokens (the count predicted 3–5%: the VALU was only partly on the critical path); whole-model prefill 1 513 → 1 522 tok/s at 2048 (1.006x), 1 548 → 1 555 at 4096, 1 484 → 1 489 at 8192, same-hour A/B twice. The m4 build's 71 allocator moves a step are register pressure at 256 VGPRs (the m2 carries four), and both arms on them are dead: the K tiles as a loop adds 38 moves a tile, A fetched at the top of the step removes the moves for **−12%**. There is no 5% left in the unpack; G5 is closed on this kernel |
| G6 | The streaming kernels: a bandwidth column in every vertical's profile, the ones under 200 GB/s listed and fixed or fused (`hc.cn` at 134 first) | **done 2026-09-30** (§5.4), for the LLM's hyper-connection kernels and the four DiT-family hosts: **the grid's fast axis decides which DRAM channels are busy** (§5.1b's law between workgroups). `hc.cn` was never at 134 GB/s (the residual's write-back was not counted: 184); walked stream-fastest it is 200, `hc.norm` 183 → 217, `hc.combine` 1.48x, and with a row's loads hoisted `hc.cn` is 2.9x at decode: **prefill 1520 → 1538 tok/s at 2048, decode 36.87 → 37.35 tok/s**. The q/k packs walked head-fastest: 117 → 209 GB/s (H3), 120 → 208 (ACE), a sixteenth of the channels → 198 (VAE), 126–151 → 173–189 (image): **a 480p VAE decode 29.95 → 27.5 s, a video forward 25.5 → 25.2 s, a 10-min music forward 2113 → 2050 ms, an image step 1.895 → 1.886 s**. All bit-identical. Left: gate+norm fusion (~1.5% of a music forward), the hosts not profiled (kev, ocr, zimage, the speech encoders, the LLM's other passes) |
| G7 | The budget: what the server's CPU work beside a device job costs in GPU clock (the audio decode's 32 threads, staging on 32 cores, tokenising; G0 measured 32 busy threads at 1.67x), whether a thread cap or a CPU power limit is a net win, and the sampler folded into `bench/sysmon.go` around the vertical tests | G0 measured the extremes; the server's own load is open |
| G8 | Carry-in: each vertical's screen re-run on the new builds, the numbers into VIDEO.md M11, the image and LLM records, and TODO.md's table | continuous. **G2 carried to all nine hosts 2026-09-29** (table below): every gate passes; the image step 2179 → 2058 ms, the VAE decode 37.1 → 35.5 s, Kev's fp16 pass at 494 tokens 154 → 132 ms; **the build loses below ~1000 rows** (a 30-workgroup grid), so `ace/dit` keeps the wave64 build up to 1024 rows and the ladders' schedules already keep it off short inputs. **G3's epilogues carried 2026-09-30** (section below): `qimage/dit` (SwiGLU, 1.04–1.06x a step), `ace/dit` (both; a forward **1.07–1.10x** at every length, the fused gate\|up on the LDS build even below 1024 rows), `h3/vae` (both; **a 480p decode 34.4 → 30.0 s, 768p 64.3 → 56.0 s, 1.15x**); parakeet and kokoro priced and left |

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

Broken out to [`g0-instruments-and-ceiling.md`](g0-instruments-and-ceiling.md).

### G1 — the shipped GEMM's schedule (2026-09-29)

Broken out to [`g1-gemm-schedule.md`](g1-gemm-schedule.md).

### G2 — the prefetch that fits (2026-09-29)

Broken out to [`g2-lds-staged-prefetch.md`](g2-lds-staged-prefetch.md).

### G8 — the carry-in (2026-09-29)

Broken out to [`g8-carry-in.md`](g8-carry-in.md).

### G5a — the MoE up GEMM's K loop as a pipeline (2026-09-29)

Broken out to [`g5a-moe-up-pipeline.md`](g5a-moe-up-pipeline.md).

### G5c — the short last tile (2026-09-30)

Broken out to [`g5c-moe-short-tile.md`](g5c-moe-short-tile.md).

### G5d — the nibble as an f16 denormal (2026-09-30)

Broken out to [`g5d-moe-nibble-denormal.md`](g5d-moe-nibble-denormal.md).

### G4 — the attention priced, and carried to the image DiT (2026-09-30)

Broken out to [`g4-attention-priced.md`](g4-attention-priced.md).

### G8 — G3's epilogues carried to the music DiT and the video VAE (2026-09-30)

Broken out to [`g8-carry-in.md`](g8-carry-in.md).

### G6 — the streaming kernels (2026-09-30)

Broken out to [`g6-streaming-kernels.md`](g6-streaming-kernels.md).

### G4b — the attention's key blocks through LDS (2026-09-30)

Broken out to [`g4b-attention-kv-lds.md`](g4b-attention-kv-lds.md).

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
  (`cmd/llm` needs `-model` as the **shard**, not the checkpoint
  directory, which holds four GGUFs and is refused; before trusting a
  pass, load a wrong-by-construction build through the same override
  and see the gate fail)
  (`LLM_MOE_SPV_WAVE=32` for a wave32 build; `LLM_MOE_PAD=64` for a build
  without `-DSHORT`, since §2.12 the schedule aligns to 16 and a plain
  build would write the next expert's rows; the shared expert runs the
  `q80` builds of the same rungs) on `cmd/llm -moe` and on the
  `llm` tests, which run from `llm/`; `cmd/llm -moe -model <shard>
  -tokens 2048 -iters 60` is ~2 s of device time an arm, so the clock is
  the median of the samples above 100 W.
- The GEMM alone against its row count: `H3_SAWTOOTH=480x864
  ./dit.test -test.run TestGPUGEMMSawtooth` (§2.14; 80 s after staging,
  the same `H3_GEMM_SPV` override); `H3_FUSE=0` on `TestGPUShapes` runs
  the fused epilogues' controls (the v pack, the SwiGLU pass), and a
  fused-path run is gated by `H3_GEMM_REF` against a shipped-build
  reference from the same day.
- The fused epilogues' controls on the other hosts: `ACE_DIT_FUSE=0` on
  `ace/dit`'s `TestGPUStepTiming`, `H3_VAE_FUSE=0` on `h3/vae`'s
  `TestGPUShapes`; `ACE_DIT_REF=file` on `TestGPUBits` writes the output
  hashes on the first run and compares them on every later one (build the
  old test binary first, as with `H3_GEMM_REF`); `h3/vae`'s decode tests
  log a `bits` hash to compare across binaries by eye.
- The streaming kernels (G6): the profiles of `h3/dit` (`H3_PROFILE=1`),
  `h3/vae` (`H3_VAE_SHAPES=1 -test.run TestGPUProfile`) and `ace/dit`
  (`ACE_DIT_PROFILE=<seconds>` on `TestGPUStepTiming`) print GB/s for every
  pass that moves bytes; `LLM_HC_SPV=cn=path,norm=path,combine=path` loads
  an hc kernel from disk, and `-DTOKEN_FAST=1` / `-DHOIST=0` on those three
  shaders are the controls (`cmd/llm -hc` times the norm and the combine,
  `cmd/llm -graph -layers 4` the fused `hc.cn`); `QIMAGE_PACK_TF=1` runs
  the image DiT's packs on the old grid. A pass near 120 GB/s that does no
  arithmetic is walking its grid the wrong way: check
  `gcd(address step along X, 4096)` against the slice a workgroup reads.
- The attention (G4, G4b): `H3_SCREEN=480x864 H3_BANK=q8
  H3_ATTN_ONLY_SPV=1 H3_SCREEN_REPS=12 H3_ATTN_SPV=qt:ktil:path,…` on
  `h3/dit`'s `TestGPUAttentionScreen`, where `qt` is the query tiles a
  *workgroup* (8 for a `-DKVLDS=1 -DWAVES=8 -DQT=1` arm); every arm is
  checked bit for bit against the one-wave transposed build, runs
  `H3_SCREEN_REPS` times in one submission and logs `window <from> <to>`
  (epoch ms) for the sampler. `H3_ATTN_LDS=0` on `TestGPUShapes` is the
  one-wave control of a forward (run it first with `H3_GEMM_REF`).
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
- ~~**G-o9.** The wave32 4×4 tile's non-MMA budget is a few instructions a
  MMA (§2.10: ~3% each). Where else in the tree does an unrolled
  cooperative-matrix loop carry a move storm? The attention kernels
  (`h3_attn_t`, `dit_attention_wmma`) spill as well as move; G4 should
  count `v_swap`/`v_mov`/`v_cndmask` a MMA in their disassembly before
  building anything.~~ **Counted** (G4, §3.8): 197 moves and 176 selects
  beside 128 MMAs in the shipped attention, 925 VALU beside 64 in the
  head-64 VAE build. And the count is not the price there: the best
  spill-free arm executes 135 fewer VALU a block for the same time.
- **G-o7.** What does a byte cost at L0, L1 and L2? DRAM is ~490 pJ and
  the MALL ~100 (§0.7); the GEMM's 1.3 TB/s of fragment traffic lives in
  the levels below, and their price decides whether LDS sharing (which
  moves bytes from L1 to LDS reads) is a saving or a wash. G0b.
- **G-o6.** The package budget is ~150 W and the CPU takes from it (G0:
  32 threads halve the GPU clock). What does the server's actual CPU
  work beside a device job cost (the audio decode's 32 threads, staging
  on 32 cores, tokenising), and is a lower thread cap or a lower CPU
  power limit a net win for the device? G7.
- ~~**G-o4.** What is wave64's accumulator element order? The probe reads
  it; if it is the wave32 order with rows split across lane halves the
  transposed attention has a wave64 build for free (G4).~~ **Moot** (G4):
  a wave64 VALU instruction is two passes and the softmax is 7 VALU an
  MMA, and the fragments move 4x their bytes; §6.2 measured the plain
  kernel 1.40x slower at wave64 for the same reasons. Not read.

## Handoff

**2026-09-30, session 13: G4b, the attention's key blocks through LDS;
a video forward 1.037x.** `research/3.9-…` has it. `h3_attn_t.comp` has a
`KVLDS` mode: a workgroup of eight wave32 waves, a query tile each, fetches
a key block's K and V once (four `uvec4` a lane a plane, half a block
ahead) into one LDS plane each, two barriers a block, and the build that
ships (`h3_attn_t_kvlds_w8_kt4`, `-DLAZY=1 -DTAIL_SPLIT=1`) is
**bit-identical** to the one-wave build and the default in `h3/dit`:
the kernel 195 → 175 ms on the 480p screen and 1.195 → 1.046 s at 768p,
in the forward 37.2 → 40.8 TFLOP/s (73 → 78% per clock), **a 480p forward
25.22 → 24.31 s and 768p 95.9 → 92.5 s**, the clock 2657 → 2717 MHz;
`TestGPUForward`, `TestGPURun` and `H3_GEMM_REF` at both shapes pass. Two
facts for whoever stages anything through registers next: **a register
that crosses a loop's back-edge is copied there behind a wait for its
load** (the first cut fetched a block ahead, tied the old build, and its
disassembly showed `vmcnt(0)` in front of the back-edge copies — fetch and
store inside one trip), and **§3.8's dead exact knobs come alive on a
spill-free build** (`LAZY` 3–4%, `TAIL_SPLIT` 6%). Measured dead: both
fetches a whole block deep with back-to-back barriers, V fetched at the
top, the barrier behind the softmax, sixteen waves (a 192-VGPR cap),
`QREG=0`, `KTIL2`. Left in the kernel by its controls: softmax 6%, global
side ≤ 6%, barriers 3%. Not carried: the VAE (6.10 against 6.22 ms on its
screen), the image DiT (opt-in behind its edit gate; same bits). The
screen now repeats an arm inside one submission and logs a sampler window
(`H3_SCREEN_REPS`); its first `H3_ATTN_SPV` field is query tiles a
workgroup. Results: the session scratchpad only (`s3`–`s10` screens,
`fw_{ctl,new}`, `fw768_{ctl,new}` logs and samples). **The served binary
needs a redeploy** (everything since G3; no restaging for this one).
**Next:** G7 (the CPU's share of the power budget, and the sampler into
`bench/sysmon.go` — this session wrote it as a scratchpad script again),
or the rest of G6 (gate+norm fusion; a bytes column for kev, ocr,
`zimage/qwen` and the LLM's other passes).

Older sessions (1–12): [`kernels-sessions.md`](kernels-sessions.md).
