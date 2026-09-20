# Phase 1 — the engine, the bugs, and the microbenchmark backlog

> Archived 2026-09-20 from the old `TODO.md` session log (the rest of that
> log was distilled into the per-stage files in this directory as each
> stage closed; the full log is in git history). This file keeps the three
> parts that were never written up anywhere else: what phase 1 built, the
> bugs worth remembering, and the microbenchmark backlog as it stood when
> phase 2 began. `§N.M` addresses [`ideas.md`](ideas.md).

### What got built (earlier sessions)

- **`vk/`** — the engine: optional device features (fp16, int8, integer
  dot-product, cooperative-matrix) negotiated at `NewDevice`, N-buffer
  pipelines with push and specialization constants, GPU-timestamp timing
  (`ComputePipeline.DispatchTimed`), buffer allocation preferring this
  APU's `DEVICE_LOCAL|HOST_VISIBLE` unified memory,
  `PhysicalDevice.CooperativeMatrixShapes()` (this device reports only
  16x16x16, subgroup scope, fp16→fp32 and int8→int32).
- **`shaders/`** — 19 `.comp` sources compiling to 36 SPIR-V variants
  (precision/tile variants via `glslc -D`, not duplicated GLSL): bandwidth
  baseline, elementwise/relu, GEMV (naive/subgroup × fp32/fp16/q8/q4 plus
  subgroup W8A8), GEMM (naive/tiled × fp32/fp16/q8/q4, cooperative-matrix
  fp16/int8/q4, plus naive W8A8), RMSNorm/softmax (shared/subgroup), a
  standalone Q4→fp16 dequant kernel, and this session's ALU-peak and empty
  kernels.
- **`bench/`** — the harness: `bench.go` (timing + adaptive iteration
  capping), `quant.go` (fp16 conversion with round-to-nearest, GGML-style
  Q8/Q4 block quantize/dequantize), `sysmon.go` (clock/power), `ops_*.go`
  (one file per op family).
- **`cmd/bench`** — the CLI (`go run ./cmd/bench -h`).
- Original demo (`main.go`, `./strix-halo-vulkan`) still works unchanged.

### Bugs hit and fixed in earlier sessions (worth remembering)

1. **GPU driver hang watchdog**: naive GEMM at N=4096 batched into 20
   back-to-back dispatches took long enough to trip amdgpu's TDR →
   `VK_ERROR_DEVICE_LOST`, which poisons the device for every case still
   queued. `bench.TimeDispatch` now probes with 1 iteration and caps any
   batch to ~500ms worth.
2. **O(N³) CPU reference GEMM inside the size sweep** sat at 98% CPU for
   18+ minutes at N=4096. **If adding a new op/variant, verify once at a
   small fixed size, never inside the sweep loop.**
3. **`VK_KHR_shader_integer_dot_product` was requested but never enabled**:
   `vk/shim.c`'s `shim_create_device` built the feature struct into the
   `pNext` chain but never added the extension name to the enabled list
   (required since the instance targets Vulkan 1.2, where it is not yet
   core). Fixed.

## Phase 1's backlog, still open

> This is the microbenchmark phase's list, left as it stood when phase 2
> began. **It is not what to pick up next** — the newest session entry above
> carries that, and `PIPELINE.md` carries the plan. Nothing here is on the
> pipeline's critical path; it is where to look when a profile points at one
> of these kernels.

`IDEAS.md` has the full backlog with its "Suggested order of attack"
updated for what §0, §1.1, §2.1, §2.2, §2.3, §5.1b, §5.1b's traversal
follow-up, §2.7, §6.2, §1.7, §3.4, §3.5, §1.8, §1.9, §1.10, §1.11 and §1.12
found. In short:

1. **Filling the GEMV's idle lanes — the one arm none of §1.10, §1.11 or
   §1.12 built, and now the only thing left pointing at `down`'s deficit.**
   At down's K=640 a VEC=4 lane-step is 20 loads, so 44 of a wave's 64 lanes
   sit out the whole kernel, and no knob that exists changes that: `NROWS`
   gives the same 20 lanes more rows. The build that would is disjoint lane
   slices per output row reduced with `subgroupClusteredAdd`. §1.10 finding 4
   had lowered the expected gain to near zero (halving the wave is a wash;
   *filling* the lanes via VEC=1 is 1.05x slower), §1.11 finding 4 raised it
   again (the row block partly substitutes for the load width, so what the
   memory system responds to looks like bytes in flight per wave, and the lane
   map is the third way to buy them), and §1.12 finding 6 now says it is the
   *only* thing left: `down` at t=256 is at 72% of the bus, the best plan's
   own residual over its cost model is 5-13%, its pad is 617 slots of 3177,
   and the plan that removes more pad than that **loses** by up to 1.4x. What
   is left is §1.10's per-output-row cost paid over four times the rows.

2. **§2.2's two leftovers — both attribution, both cheap.** (a) The scale
   plane's layout: `QBLOCK=128` beats `QBLOCK=32` by **1.24x** between two
   binaries that are instruction-for-instruction identical, and the nominal
   byte difference is 4.4% standing behind a 24% one. The candidate is
   locality — at QBLOCK=32 a slab's 128 staging chunks read 64 rows of a
   scale plane whose row stride is 160 B, so each 2-byte scale drags in its
   own line — and the test is a blocked scale layout that makes one slab's
   scales contiguous. §1.8 has since narrowed it by elimination: the same axis
   on the GEMV, which reads its scales the way it reads its weights, costs
   1.045-1.116x — its bytes and nothing more — so the GEMM's extra 1.1x is
   about staging, not traffic. Accuracy wants the small block, so this is worth
   knowing the price of. (b) The dequant's two packed fp16 ops per two
   weights, now that the loop body is what binds; the cheap halving is
   numerically invalid (§2.2's finding 9) and the valid form is an
   fp32-epilogue per-K-block row sum of A.
3. **The selection arm's own two leftovers** (IDEAS §1.12, "still open"), both
   cheap. **A width-1 part is a big part**: at t=256 the winning plan puts 17
   experts in the width-1 bucket and 248 in the width-8 one, and at t=16 it is
   121 of 139 in width-1 — nothing measures whether a part below some group
   count is worth folding into its neighbour rather than dispatched. And **the
   calibration transfers across batches but nothing tests whether it transfers
   across shapes**, which is what an engine that fits once would actually want.

   (The item that used to stand here — "explain the down projection's 198 GB/s
   against gate_up's 235" — is **done**: it is a fixed cost per output row
   inside the wave, 19.8% of down's time against -0.4% of gate_up's, and the
   N block takes down to 98% of gate_up's rate. Dividing the workgroup count
   alone is worth 1.02x, so the grid was never it. See IDEAS §1.10. The item
   *before* that one — "carry §3.5's stride window back to the GEMV kernels" —
   is also done and came out backwards: at constant traffic the *unpadded* bank
   wins at both shapes, including down's gcd-64 row, and every pad costs
   1.12-2.4x. The window is a tiled-read rule. See §1.8 finding 6 and §5.1b
   rule 1's second amendment.)

4. **`VEC=32` is retired.** Left here so it is not re-proposed: the largest
   reduction length any of the five models decodes over is 6144, and at
   6144 every built width already reads the bus. There is no matrix for it.
5. **The two inverted cells in §3.5.** `moe.gate_up` at 8192 tokens in
   `reg32_bt_hkab4` and `reg16x32_bt_hkab4_w32`: the *grouped* dispatch is
   0.74-0.75x the per-expert loop, reproducibly across three runs, where
   every other cell goes the other way and the occupancy story predicts a
   tie. Both are the narrow-BN kernels at the largest dispatch (51k and 102k
   workgroups). Small, and the only cell in `moe` that the attribution gets
   backwards.
6. **Finish §2.7's ladder on the other winners.** It was run on `reg64_bt`
   and `reg32_bt` only. `reg64x128` and `wg128x256` are the AI-43/85 shapes
   and already sit at 252 VGPRs with 32 accumulators, so they cannot hoist as
   they stand — whether the lever survives into them is what says how it
   composes with arithmetic intensity. §6.2 has now closed off the wave32
   route to that: the register ceiling halves with the wave, so a
   32-accumulator variant spills at wave32 rather than gaining headroom.
   Also missing: a row-major `hkb4`, which completes the 2x2 of (operand
   hoisted) x (layout) that §2.7 could only half-fill, and an explanation for
   the one reproducible cell where padding *hurts* (`reg32_bt_hkab4`, 0.88x).
7. **The MALL's own slice structure — still open, and now with one more
   clue.** Under partial channel coverage the MALL keeps a working set only
   while its *span* is also inside 32 MiB, one case lands halfway, and the
   traversal axis has added MALL cells the coverage law fits (2048 B rows at
   a dense 2048 B stride, cross-wave: 484 of 868-942) next to cells it does
   not (the same rows at a 4096 B stride, walk: 774-919 at model 0.50). The
   `stride` family already runs the experiment from flags:
   `-stridefootprints 4,8,16,24,32,48 -striderowbytes 1024 -stridepads
   0,1024,3072` separates effective capacity from bus width. Worth doing
   before any MALL-blocking work in §5.1b/§2.4/§3.3, which needs a number to
   size against.
8. **IDEAS §2.4 — workgroup swizzle.** Still cheap and structural, and the
   traversal axis sharpens what it would be testing: a swizzle changes
   exactly which addresses are in flight together, which is now a measured
   axis with a model behind it. But its bandwidth premise is gone — §2.3
   pushed the AI-32 kernels to 742-887 GB/s of implied MALL traffic and §2.7
   pushed the best of them to **1218 GB/s**, past even the 965 GB/s a pure
   MALL read delivers, so implied traffic no longer bounds these kernels at
   all. Run it as a discriminator (helps AI-32, does nothing for AI-85), not
   as a bandwidth fix.
9. **Pad the strides in the other kernels — answered on the expert bank, open
   everywhere else.** This
   used to be a falsification test that the model predicted would find
   nothing: mechanism 3 looked like it had caught the GEMV kernels red-handed
   (one wave per row, rows a stride apart) until its own ladder measured that
   exact shape at the full bus at every stride. §3.5 changed the prediction —
   the rule is a window, [128, 256] B of gcd, and the real 4-bit weight rows
   sit *below* it — so it became a sweep with an expected win rather than a
   check. §1.8 finding 6 then ran that sweep on the MoE expert bank and it came
   out backwards: at constant traffic the unpadded stride wins at both shapes,
   including down's gcd-64 row, and every pad costs 1.12-2.4x (see item 3's
   note). What is left here is the same sweep on the kernels the `moe` family
   does not cover — the plain GEMV/W8A8/fp16 paths of item 10 — where the
   prediction is now "no effect, and any pad costs its own address span".
10. **IDEAS §1.3/§1.7 — the load-width rule on the other GEMV kernels.**
   fp16 GEMV sits at 75% of its DRAM ceiling and W8A8 at 73%, where W4A8 now
   reaches 101% at every N it was swept at. This is no longer "try wider
   loads and see": §1.7 gives the target width in closed form — make one
   lane-step cover a whole weight row — and it is format-independent, since
   it is about bytes of a row, not weights. A W8A8 row is twice the bytes per
   weight, so its `VEC` is half W4A8's at the same N; an fp16 row is four
   times, so a quarter. §3.4 shrank the expected gain without removing it:
   at the models' real reduction lengths the width that reaches the bus is
   usually the narrowest one, so what is on offer here is these kernels' own
   27% gap, not the 1.43x W4A8 collected at N=8192. §1.8 confirmed that at the
   MoE shapes: across VEC 1-16 the spread is 3-5%.
11. **IDEAS §1.2 — remove the runtime integer divisions** (`pc.N / pc.block`
   and `n / pc.block`) from everything W4A8 did not rewrite: the
   naive/tiled GEMM paths and the old quantized GEMV variants.
   `gemv,naive,q4` is still slower than naive fp32, which is the symptom.
12. **IDEAS §1.6 — the activation-quantize cost W8A8 and W4A8 both hide.**
   Both kernels get `x` quantized on the host, outside the timed loop, and
   W4A8 additionally gets its per-block activation sums for free. A real
   decode step must produce both on-GPU between layers. The sums are one
   extra reduction over N (cheap, and fusable into the quantize pass), but
   it should be measured rather than assumed — it is the one place where
   these numbers are friendlier than production would be.
13. **Still open from §0**: CPU-side submit/fence cost (the half of §4.1 the
   `overhead` family does not measure), and the last ISA question from §6.1
   (is `v_pk_fma_f16` actually emitted, given §0.1's 1.10x packed-fp16
   surprise). `-blocks` is worth re-sweeping only where a block-size effect
   survives warming — in GEMV it does not.
14. **IDEAS §3.7 — rewrite the subgroup reductions**, now that §6.2 has
   measured *why* they lose rather than guessing: bandwidth is linear in
   threads per row (256 → 223.7 GB/s, 64 → 68.5, 32 → 35.7), so the fix is a
   256-thread workgroup doing `uvec4` loads → per-subgroup `subgroupAdd` → a
   small LDS combine, and there is no wave-size knob worth trying first.
   Mechanical, and it deletes a variant the engine should never pick.
15. **Note on a fresh checkout.** `*.spv` is gitignored, so
   **`go generate ./...` is required** before anything runs — the thirteen
   `strided_read` variants and the 30 WMMA ones are each built from a single
   `.comp`. **This session added one**, `shaders/gemm_wmma_q4.comp`, built
   into eight binaries, so a checkout that skips `go generate` will not
   compile `shaders/shaders.go` at all.
   Worth knowing when reading ISA: RADV caches compiled pipelines on disk, so
   `RADV_DEBUG=shaderstats` and `RADV_DEBUG=asm` print **nothing** on a
   second run of the same shader. Set `MESA_SHADER_CACHE_DISABLE=true`
   alongside them.
