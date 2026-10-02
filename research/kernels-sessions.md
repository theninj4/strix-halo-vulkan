# KERNELS — the session handoffs

*Sessions 1–12 of `KERNELS.md`'s § Handoff, newest first, broken out when it moved to `research/` on 2026-10-02. The newest handoff stays in [`kernels-vertical.md`](kernels-vertical.md).*

**2026-09-30, session 12: G6, the streaming kernels; the grid's fast axis
was the defect.** `research/5.4-…` has it. A bytes column went into the
profiles, and the slow passes turned out to share one cause that is not in
any kernel's body: the workgroups in flight together are neighbours along
the grid's X axis, and where a step along X is a whole number of the 4 KB
channel rotation they all load the same DRAM channels. Swapping the two
grid axes — the LLM's hyper-connection kernels to a stream across and a
token down, the DiT family's q/k packs to a head across — is the same
workgroups and the same bits, and takes `hc.norm` 183 → 217 GB/s, the packs
93–151 → 173–209. `hc.cn`'s long-standing "134 GB/s" was also a third
accounting (the residual's write-back uncounted; it was at 184 and is at
200), and the three hc kernels and the DiT norm and gate now issue a row's
loads before waiting on any, which is nothing off DRAM and 2.9x on
`hc.cn` at decode. Bit-identical everywhere (`TestGraphLogits`'s line to
the digit, `H3_GEMM_REF`, the VAE's two decode hashes, `ACE_DIT_REF`, the
image DiT's in-process control). Whole-path numbers, old and new the same
hour: LLM prefill 1520 → 1538 tok/s at 2048 and decode 36.87 → 37.35;
a 480p video forward 25.5 → 25.2 s; **a 480p VAE decode 29.95 → 27.5 s**
(34.4 this morning); a 10-minute music forward 2113 → 2050 ms; an image
step 1.895 → 1.886 s. Refused: `vec4` lanes, a one-wave workgroup, the
rotated walk. **The served binary needs a redeploy** (everything since G3).
Results are in the session scratchpad only. **Next:** G4b (the attention's
K/V block through LDS), G7 (the CPU's share of the power budget), or the
rest of G6 — the gate+norm fusion, and a bytes column for kev, ocr,
`zimage/qwen` and the LLM's other passes, where the same one-line test
(walk the grid the other way) is the first thing to run.

**2026-09-30, session 11: G8, G3's epilogues carried to the music DiT and
the video VAE; the speech encoders priced and left.** Host changes only
(the section above): `ace/dit` stores v from its GEMM on both rungs and
runs gate|up as one GEMM ending in the SwiGLU — on the LDS build at
every row count, which is a new row rule for that one projection (110
against 120 ms at 375 rows) — for **a forward 1.07x at 30 s, 1.10x at
2 min, 1.08x at 4 min, 1.06x at 10 min**; `h3/vae` does the same (its
biases in the interleaved weight; v from the GEMM only when the batch is
whole 128-row tiles, since its pad rows are the bias) for **a 480p
decode 34.4 → 30.0 s and 768p 64.3 → 56.0 s (1.15x)**, which is 4.4 s
of every 480p video request. Both bit-identical to the old binaries
(hashes) and to their controls; every gate of both packages and the
music pipeline's passes. Parakeet's two SiLU passes and its v pack are
2.3% of a 13.5 ms encoder at the dispatch floor, on a rung the epilogues
are not built for: left, with kokoro. New tests and knobs: `TestGPUBits`
(`ACE_DIT_REF`), `ACE_DIT_FUSE=0`, `H3_VAE_FUSE=0`, the `bits` lines in
`h3/vae`. Results: the session scratchpad only (`t_{old,new,ctl}_{1,2}`,
`vs_{old,new,ctl}_{1,2}` logs and samples), the numbers above. **The
served binary needs a redeploy** for all of G3 (video restages its int8
cache once; the VAE and music have no cache). Not looked at: the other
gated FFNs on the big GEMM (`zimage/qwen`'s text encoder, `ace/lm`'s and
`ocr`'s prefills, `qimage/vision`) — each runs once a request on short
inputs. **Next:** G6 (the streaming kernels; H3's `qkpack` q/k at 1.1%
each, and with the VAE's SwiGLU and pack gone its `qkpack` and norms are
the next thing its profile would show — it has no per-kind profile test
yet, write that first) or G4b.

**2026-09-30, session 10: G3, the epilogue priced and its consumers fused
into it.** H4 is dead on the pipe: `-DSTORE_TILES=n` keeps every MMA and
skips the stores at run time, and every arm (0, 1, 8 of 16 tiles) runs
the shipped build's 404–409 FLOP/clk/CU — the +9% the store-free arm
reads is 2900 MHz at 130 W against 2671 at the cap, because everything
downstream ran on zeros (a memory note: price arms per clock, and make a
wrong-by-construction control leave *stale* data, as `STORE_TILES=8`
does at 2746 MHz / 152 W). H5 is `TestGPUGEMMSawtooth`
(`H3_SAWTOOTH=480x864`): the rate plateaus by ~640 rows on gate and
~1300 on q and down and then swings ±5% / ±2% / ±3% with the fractional
round; the served chunks land at the top; nothing to fix. What the fp32
C costs is the passes that read it, so both are now the GEMM's epilogue
on the LDS build (`research/2.14-…`): **v stored as the attention's
fragment tiles** (`C_PACK`, whose guard is lifted; `zero_f16.comp` zeroes
the A operand's pad rows of the last chunk first, so the pad keys are the
pack's +0) and **gate|up as one GEMM whose epilogue is the SwiGLU**
(`C_SWIGLU`; the host interleaves `ff.net.0.proj` in 64-row groups as
`projGateUp`, K7.6's trick; `pc.aux0` the output stride, `pc.scale`
ffScale). The K loop of both builds is the shipped loop instruction for
instruction; the epilogue's exp/rcp/fma_mix lower as the SwiGLU kernel's
do. **Bit-identical** to the shipped build's forward
(`H3_GEMM_REF=ref480.bits` from the same morning) on every fused run,
and `TestGPUForward` now compares the fused paths against run-time
controls on the same weights (`packV`, `fuseGLU` off: the pack, and the
SwiGLU pass over the fused GEMM's fp32 output through its interleaved
columns, `dit_swiglu_f16 -DINTERLEAVE=32`). **A 480p forward 26.22 /
26.33 → 25.59 / 25.50 s (1.03x)** at 2660–2667 MHz / 156 W (the controls
26.26 / 26.26; `TestGPUForward` and `TestGPURun` pass), the `swiglu` and `pack v`
dispatches gone and the fused GEMM at the shipped rate (5.66 s against
5.72 for gate+up); one fused run read 32.9 s with its first 20 s at
1760 MHz / 159 W and nothing in the graph to explain it, bracketed by
clean runs, recorded as the machine's. The int8 bank cache's key
changes with the projection, so **the served binary restages once on
redeploy** (and still needs one). Tools: `STORE_TILES`, `C_SWIGLU`,
`INTERLEAVE`, `H3_FUSE=0`, `loadGEMMOverride` (shared by the screens).
**The SwiGLU epilogue carried to `qimage/dit` the same session**: the
checkpoint's `img_mlp.gate_layer` and `proj` interleaved at staging by
`qwen.InterleaveGLU` (now shared with H3) as `projW13`, the
`gemmLDSW32GLU` kernel when the big kernel is the LDS build and the fused
GEMM plus the interleaved SwiGLU pass on every other arm and as the
control (`SetFuseGLU`, `QIMAGE_FUSE=0`); `TestGPUFusedEpilogues` holds a
prefill and a cached step at 256² bit-identical to the control on the
int8 and fp16 banks, `TestGPUOracle256` passes unchanged, and at 1024²
on the served int8 bank a cached step is **2044 / 2046 → 1929 / 1968 ms
wall (1.04–1.06x)**, 1.99 → 1.905 s per dispatch with the 83 ms `swiglu`
gone and the fused GEMM 3% faster than its fp32-storing twin. The v store
is not carried there: a t2i prefix is 19 text rows and the fresh rows
land under it at their global rows, mid-tile for `C_PACK`, and `pack v`
is 1.2% of a step. The served image binary needs a redeploy (no bank
cache there; staging is the same). **Next:** G8 for the rest — `ace/dit`,
the VAE and the speech encoders, each with the same two passes and its
own gate; then G6 (the streaming kernels: H3's `qkpack` q/k at 1.1% each
are the next largest, and need a cross-wave head norm to fuse) or G4b.

**2026-09-30, session 9: G4, the attention priced; nothing exact beats
the shipped build; the kernel carried to the image DiT.** The ISA of
`h3_attn_t_qt2_kt4` (G-o9): 911 non-MMA VALU a key block of 128 MMAs,
197 of them allocator moves, 176 selects, a 16-deep row-max chain, plus
113 reloads of the spilled Q^T. The honest gutting on a spill-free base
(`research/3.8-…`, `GUT_*` knobs with every result kept live): the
softmax is 6–7 points (the row max 7, the rescale 3, the exp2 1), the
floor **81% per clock with real loads and 87–90% with L0 hits**
(`L0_LOADS`), so the K/V loads are 11–12 points. Five exact arms
(`TAIL_SPLIT`, `QLDS`, `MAX_TREE`, `LAZY`, `PV_ORDER`, and their
combinations; all bit-identical, the screen checks it now) land within
±2% of the shipped build at 480p and 768p — the best executes 135 fewer
VALU and no reloads a block for the same time, because at 256 VGPRs the
allocator's moves redraw with every edit. Head split and wave64 dead by
arithmetic. Carried to `qimage/dit` (`qimage_attn_t_qt2_kt4`,
`STORE_TAIL=0`, the element order probed at the first graph, the context
fp16 in place): attention **248 → 232 ms** a 1024² step, a cached step
**1.908 → 1.876 s**, the t2i oracles passing — and **opt-in only**
(`QIMAGE_ATTN_T=1`): the edit oracle's prefill reads 7.9e-2 against 2e-2
where the plain kernel reads 7.7e-3, the kernel is exact on identical
inputs at every block (rms 8e-5), and a plain-family fp16 perturbation of
the prefix rows reads 1.73e-2 on the same gate, so the bound is near its
sensitivity floor for edits and it is the vertical's call. `STORE_TAIL=0`
now bounds the last tile's store per query (the VAE's builds too, PSNR
unchanged). Not carried: `ace/dit` (grouped-query heads), the speech
encoders, the VAE's exact arms. Left: **G4b**, the
K/V block staged through LDS for a workgroup's waves at `QT1` a wave
(§2.9's lever; the 12 memory points less ~6–9, ≤ 2% of a video forward
for a session). The default H3 build's ISA is unchanged; the key block
lives in `shaders/h3_attn_t_block.glsl`, included twice. Results:
`scratchpad/screen{1,2,3,768}.samples` (clocks), the arms' `.spv` and
`.asm` in the session scratchpad only. Nothing served changes. **Next:** G6 (the streaming kernels) or G3
(the epilogue), as the order says; G4b if a session wants the last 2%
of a video forward.

**2026-09-30, session 8: G5d, the unpack, and G5 closed on this kernel.**
The ISA said the estimate was wrong: an element of the Q4_K unpack is
three VALU, not five, because ACO already fuses the affine and the half
conversion into `v_fma_mix`. The nibble read as an f16 denormal
(`-DNIB_F16=1` on the seven Q4_K up builds, `research/2.13-…`) removes
the extract-and-convert pair in front of it — three instructions for
four elements, the scale carrying the 2^24, bit-identical (the ladder at
`maxAbs == 0`, the whole-model logits line to the digit against the old test binary) — and the four-tile step's non-MMA
VALU goes 331 → 259 for `moe.up` **1.02x** at every count (the count
predicted 3–5%: the VALU is only partly on the critical path with four
waves a SIMD); **1 513.1 / 1 512.7 → 1 521.1 / 1 523.3 tok/s at 2048 (1.006x)**, 1 549.3 / 1 546.7 → 1 553.9 / 1 556.6 at 4096 (1.005x), 1 483.9 / 1 483.4 → 1 491.4 / 1 487.5 at 8192 (1.004x), the clock 2 671–2 674 MHz at 141–142 W on every arm. Two arms dead on the m4 build's 71 allocator
moves a step: the K tiles as a loop adds 38 moves a tile (dropped on the
count), and fetching A at the top of the step frees the sixteen prefetch
registers, removes the moves and costs **12%** (the m2 build at 240
VGPRs carries four moves, so it is pressure, and the pipeline is worth
six times the moves). Three consecutive arms at ≤ 2% is decision 7's
plateau: **G5 is closed on this kernel** — the up projection's step is
64 MMAs, 64 fma_mix, ~50 masks, ~70 moves that cannot be bought, ~75 of
scale path and addresses, 24 fragment reads and 12 LDS stores. Left
unbuilt and small: the Q5_K up build (layer 2 of 48), the down mode's A
fragment loads inside its MMA block (§2.11), and the scale path (~30 a
step, ≤ 1%). Tools: `-DNIB_F16`, `-DPIPE_A`, `-DMOE_KT_LOOP` on
`llm_moe_gemm.comp`; `cmd/llm -moe` wants `-model <shard>`. Results:
`results/g5d_moe_{shipped,nib,nib_pipea0}_r{1,2}.csv`. The served binary still needs a redeploy.
**Next:** G4 (attention) or G6 (the streaming kernels), as the order
says; G3's epilogue also still open.

**2026-09-30, session 7: G5c, the MoE GEMM's padding.** The schedule's
alignment is one fragment instead of the widest row block, and the
single-wave builds of `llm_moe_gemm.comp` are `SHORT` builds that pick
one of WM copies of their K loop from the record's real-row count
(`research/2.12-…`): bit-identical (the ladder at `maxAbs == 0`, the
whole-model logits line to the digit), the padding factor at 2048 tokens
1.90x → 1.19x, `moe.down` **1.23x** (1.28x on the served IQ4_NL),
`moe.up` **1.11x**, the block 1.13–1.14x, whole-model prefill **1417 →
1514 tok/s at 2048 (1.07x)**, 1500 → 1549 at 4096, 1461 → 1481 at 8192
(same-hour A/B, twice); `moeGEMMPlanFor`'s m2/m4 boundary moved from 1024
tokens to 256 on the new ladder; TODO.md's table carries it. Two facts
for whoever touches the kernel next: any third copy of the Q4_K up m4
loop spills 11–15 registers and costs 9% on full blocks, so that build
ships two copies (a 16-row tail runs as 32; its store is bounded by the
real count, and storing the rounded tile is a race the ladder test
catches); and a short tile's rows are cheap rows (0.086 µs against 0.137)
because the slab unpack — ~300 VALU a lane a step, a matrix clock each —
is now the largest term in the up projection at every rung. Tools:
`-DSHORT`/`-DSHORT_STEP`, `LLM_MOE_PAD` (a pre-§2.12 build through
`LLM_MOE_SPV` needs 64). Results: `results/g5c_moe_short_*.csv`. The
served binary still needs a redeploy. **Next:** the unpack's VALU (G5's
`v_perm`/`v_cvt_pk` argument, and it lifts the short tiles most), the
Q5_K up build, the down mode's A fragment loads; or G4 (attention) as the
order says.

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
