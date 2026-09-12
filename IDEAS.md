# IDEAS — experiments worth running to squeeze more out of Strix Halo

Written after reviewing `TODO.md`, all 433 rows of `results.csv`, every
shader in `shaders/`, and the device's actual reported capabilities
(`vulkaninfo`, `/sys/class/drm/card1/device/pp_dpm_sclk`). This is a
prioritised experiment backlog, not a plan — each item states a
*hypothesis*, the *change*, the *expected gain*, and *how we'd know*.

## The roofline, and why it says there's a lot left on the table

Hardware numbers for this part (AMD Radeon 8060S, gfx1151, 40 CU, sclk
tops out at **2900 MHz** per `pp_dpm_sclk`; LPDDR5X-8000 on a 256-bit bus):

| Ceiling | Theoretical | Best measured | Utilisation |
|---|---|---|---|
| DRAM bandwidth | 256 GB/s | **236 GB/s** (`bandwidth,copy` @64MB) | **92%** ✅ |
| Vector FP32 FMA | ~14.8 TFLOP/s (29.7 dual-issue) | 2.76 TFLOP/s (`tiled,fp16`) | ~19% |
| WMMA fp16→fp32 | ~59 TFLOP/s (512 FLOP/clk/CU) | 5.84 TFLOP/s (`coopmat,fp16` @N=1024) | **~10%** ❌ |
| WMMA int8→int32 | ~119 TOPS | 9.62 TOPS (`coopmat,q8` @N=512) | **~8%** ❌ |

Two conclusions drive everything below:

1. **DRAM bandwidth is already saturated** (92% of theoretical). No kernel
   trick will beat it. So for *decode*, the only lever is **reading fewer
   bytes per weight** — which makes the current "W8A8 beats Q4 by 2.5x"
   conclusion suspect (see §1.1, the highest-priority item here).
2. **The matrix cores are ~90% idle.** The coopmat kernels are nowhere near
   an MMA-issue limit; they're memory/latency-bound because of how they're
   written, not because of the hardware. That the int8 coopmat path (2x the
   hardware MMA rate of fp16) lands at roughly the *same* GFLOP/s as fp16
   (4729 vs 4200 @N=4096) is direct evidence: if MMA throughput were the
   limit, int8 would be ~2x ahead. So for *prefill/batch*, the lever is
   **arithmetic intensity** (§2.1).

Caveat worth resolving first: the 512 FLOP/clk/CU WMMA figure is derived
from RDNA3 discrete parts (7900 XTX: 122.8 TFLOP/s ÷ 96 CU ÷ 2.5 GHz), and
RDNA3.5's WMMA may differ. §0.1 measures it directly rather than trusting
the arithmetic.

---

## 0. Measurement-validity work (cheap, and it gates the rest)

Several existing numbers are probably not measuring what we think. Fix
these before optimising against them.

### 0.1 Establish the real MMA ceiling with a memory-free microbenchmark
**Hypothesis**: we don't actually know this chip's peak WMMA rate, so we
can't say how much headroom the coopmat kernels have.
**Change**: a shader that loads one `matA`/`matB` into registers *before*
the loop, then runs N thousand `coopMatMulAdd`s on register-resident
operands (accumulator chain, with enough independent accumulators to hide
latency), writing the result once at the end. Zero memory traffic in the
loop. Same for the int8 shape and for `dotPacked4x8EXT`.
**Expected**: a hard number for fp16 WMMA, int8 WMMA, packed-dot, and
plain FMA rates on gfx1151 — the denominator for every other experiment.
**Measure**: GFLOP/s vs the table above; also divide by 2.9 GHz × 40 CU to
recover FLOP/clk/CU and compare against the 512 assumption.
**Effort**: low. **Value**: high — it turns guesses into a roofline.

### 0.2 Log GPU clock and power during the sweep
**Hypothesis**: some results are DVFS artefacts, not kernel properties.
The GPU idles at **636 MHz** out of 2900, and `bench.TimeDispatch` probes
with a *single* dispatch before batching — on short kernels that probe (and
possibly the timed batch) can land before the clocks ramp. Smoking guns in
`results.csv`: `gemv,subgroup,w8a8,128` goes 382 → **190** → 592 GFLOP/s at
N=512/1024/2048; `gemv,subgroup,q4,128` goes 381 → **135** → 373 at
N=1024/2048/4096. Those dips are not plausible kernel behaviour.
**Change**: sample `hwmon*/freq1_input` and `power1_average` (both exist
on this card) plus `gpu_busy_percent` around each timed batch; record
min/max sclk as CSV columns. Add a fixed pre-sweep "clock warmer" dispatch
loop (~200ms of heavy work) and increase warmup so every case is measured
at boost.
**Expected**: the non-monotonic dips disappear; numbers become comparable
across sizes. Possibly a uniform uplift on all short kernels.
**Measure**: re-run and check monotonicity; compare pre/post GFLOP/s.
**Effort**: low. **Value**: high — the W8A8 block-size "noise" flagged as
TODO item 4 is very likely this, not the kernel.

### 0.3 Distinguish cache-resident from DRAM-resident measurements
**Hypothesis**: most of the interesting results are measuring the 32MB
MALL/Infinity Cache, not DRAM, and therefore overstate real inference.
Evidence: `bandwidth,copy` reads **908 GB/s** at 16MB and 236 GB/s at
64MB. A 4096² fp16 weight matrix is 33.5MB, int8 16.8MB, Q4 8.4MB — so the
GEMV sweep's weights are *entirely cache-resident* at every size, and the
timing loop re-reads the same matrix 100s of times. `gemv,subgroup,w8a8`
reports **518 GB/s** — more than 2x DRAM bandwidth, which is only possible
from cache. In real decode the weights are gigabytes and every byte comes
from DRAM, once per token.
**Change**: add a "cold weights" GEMV/GEMM mode — allocate K weight
matrices totalling >>32MB (say 8×64MB) and round-robin which one each
iteration touches, so no iteration hits a warm line. Report both numbers.
**Expected**: all GEMV variants collapse toward `bytes ÷ 236 GB/s`, and
the ranking **reorders by bytes-per-weight**: Q4 (0.5 B/weight + scales)
should beat W8A8 (1 B/weight) by ~2x, the opposite of the current
conclusion.
**Measure**: GB/s should pin at ~236 for every format; GFLOP/s should then
be ~2x for Q4 vs Q8/W8A8 vs 4x vs fp16.
**Effort**: medium. **Value**: **highest in this document** — it directly
decides the weight format for the whole engine, and the current answer
(W8A8) may be an artefact.

### 0.4 Fine-grained bandwidth-vs-working-set curve
**Change**: sweep `-bwsizes` at 2/4/8/12/16/20/24/28/32/40/48/64/128 MB
instead of the current four points.
**Expected**: maps the MALL cliff precisely, giving a hard budget for "how
much of a layer's weights can stay resident" and for KV-cache sizing.
**Effort**: trivial (a flag + a re-run). **Value**: high, informs §5.

---

## 1. Decode path (GEMV / memory-bound) — the tok/s lever

### 1.1 W4A8: Q4 weights fed to `dotPacked4x8EXT`
**Hypothesis**: the best decode kernel reads 4-bit weights *and* uses the
packed-int8 dot instruction. Today those are mutually exclusive — Q4 goes
through the float dequant path and W8A8 reads 8-bit weights.
Supporting evidence: Q4 GEMV achieves only **96–112 GB/s** while W8A8 hits
518 GB/s — Q4 is nowhere near any bandwidth limit, it's **ALU-bound in the
unpack**, reading one `uint8_t` at a time and doing a float multiply per
nibble (`gemv_subgroup.comp:46-53`).
**Change**: new `gemv_w4a8.comp`. Load weights as `uint` (8 nibbles/word),
unpack to two packed-int8 words with bit tricks — roughly
`lo = (v & 0x0F0F0F0F) - 0x08080808`, `hi = ((v >> 4) & 0x0F0F0F0F) -
0x08080808` (~6 ALU ops for 8 weights) — then two `dotPacked4x8EXT` calls.
Accumulate in **int32 per block** and apply the scale once per block, not
per element.
**Expected**: half the weight bytes of W8A8 at comparable ALU cost. Under
DRAM-bound conditions (§0.3) that's ~**2x W8A8**; even cache-resident it
should beat Q4's current 400 GFLOP/s by 2x+.
**Measure**: GB/s should approach W8A8's, GFLOP/s should roughly double.
**Effort**: medium. **Value**: very high — this is plausibly *the*
production decode kernel.

### 1.2 Kill the runtime integer divisions in the inner loop
**Hypothesis**: every quantized kernel does **two runtime integer
divisions per element** in its innermost loop. E.g.
`gemv_subgroup.comp:44`, `gemm_tiled.comp:55-56`, `gemm_naive.comp:54-55`,
`gemm_coopmat_q4.comp:44-45`: `pc.N / pc.block` and `n / pc.block`, plus
`idx / 2u` for Q4. Integer division by a non-constant has no hardware
instruction on AMD — the compiler emits a float-reciprocal sequence of
~10-20 instructions. This is likely a large part of why naive q4/q8
(445–536 GFLOP/s) is *slower than naive fp32* (521) despite reading 4-8x
fewer bytes.
**Change**: pass `log2(block)` (all block sizes swept are powers of two)
and use `>>`; better, make it a **specialization constant** so the shader
compiles with the shift folded in — `vk.PipelineSpec` already supports spec
constants. Hoist `blocksPerRow` out of the loop entirely (it's loop-invariant
but the compiler can't prove it cheaply through the buffer indexing).
**Expected**: large on the naive/tiled quantized paths; meaningful on GEMV
Q4. Possibly also flattens the mysterious block-size sensitivity.
**Measure**: dump ISA (§6.1) and confirm the `v_rcp_f32`/`v_mul_hi_u32`
division sequences are gone; then re-benchmark all quantized variants.
**Effort**: low. **Value**: high — cheapest real win in this document.

### 1.3 Vectorise the loads (uvec4 / 128-bit per lane)
**Hypothesis**: GEMV is issuing 32-bit loads where it could issue 128-bit
ones, wasting memory-instruction issue slots and not filling cache lines
per instruction. `gemv_w8a8.comp:52` loads one `uint` per lane per step;
`gemv_subgroup.comp` loads a single `float16_t` or `uint8_t`.
**Change**: read `uvec4` (16 weights for W8A8, 32 for W4A8) per lane per
step; for fp16 use `f16vec2`/`f16vec4` and keep the math in packed fp16
(`v_pk_fma_f16`, 2 FLOP/lane/clk) rather than converting to fp32 scalar.
**Expected**: 1.3-2x on the ALU-bound variants; smaller on already
bandwidth-saturated ones. Also engages RDNA3's packed-fp16 ALU which the
current code never touches.
**Effort**: low-medium. **Value**: high.

### 1.4 Multiple output rows per workgroup
**Hypothesis**: one workgroup (one wave64) per output row means the
activation vector `x` is re-read from cache by all 4096 workgroups, and
each wave does a full `subgroupAdd` tree for a single scalar output.
**Change**: have each workgroup compute R=4-8 rows, holding `x`'s tile in
registers/LDS once and reusing it across rows; emit R results per
`subgroupElect`.
**Expected**: R-fold reduction in activation traffic and 1/R the reduction
overhead. Modest when cache-resident, more once §0.3 makes traffic honest.
**Effort**: medium. **Value**: medium.

### 1.5 Find the GEMV→coopmat crossover for small batch
**Hypothesis**: for M=2..16 (speculative decoding, beam search, batched
serving) padding M up to 16 and using the WMMA path beats M separate GEMV
dispatches, even though up to 15/16 of the MMA work is wasted — because
the MMA path has ~10x the throughput.
**Change**: benchmark `gemm_coopmat_*` and `gemv_*` at M = 1,2,4,8,16,32,64
with K=N=4096, and find the crossover.
**Expected**: a concrete batch threshold for the engine's scheduler to
switch kernels at. This is exactly the decision llama.cpp encodes as
separate `mul_mat_vec` vs `mul_mm` kernels.
**Effort**: low (it's a shape sweep over existing kernels).
**Value**: high for the serving story in `GOALS.md`.

### 1.6 Measure the activation-quantization cost that W8A8 currently hides
**Hypothesis**: W8A8's headline number excludes work a real engine must do
per token. `bench/ops_w8a8.go:65` quantizes `x` **on the host, once,
outside the timed loop**. In production, every decode step must quantize
the activation vector on-GPU (a pass over `x`: max-abs reduction, then
scale-and-pack) between every pair of layers.
**Change**: time the full chain RMSNorm → quantize-to-int8 → GEMV, and
compare against RMSNorm → fp16 GEMV. Then §3.1 fuses it away.
**Expected**: reveals W8A8's true cost. On small vectors the extra
dispatch's launch + barrier overhead may dominate the quantize itself.
**Effort**: low. **Value**: high — needed to fairly compare W8A8 vs Q4.

---

## 2. Prefill / batch path (GEMM) — the ~90%-idle matrix cores

### 2.1 Register-block the coopmat kernels (the big one)
**Hypothesis**: `gemm_coopmat_fp16.comp` is memory-bound by construction.
One workgroup = one wave64 = **one** 16×16 accumulator, and the K-loop
loads A and B straight from global memory every iteration
(`gemm_coopmat_fp16.comp:46-48`). Arithmetic intensity: 2·16·16·K flops per
(16K + 16K)·2 bytes = **8 FLOP/byte**. Saturating ~59 TFLOP/s at 236 GB/s
needs ~250 FLOP/byte. The measured 4.2-5.8 TFLOP/s is exactly what an
8 FLOP/byte kernel gets with MALL help — the MMA units are starved.
**Change**: the standard tiled-WMMA structure:
- 4 waves per workgroup (256 threads), workgroup tile 128×128.
- Each wave holds a **2×4 or 4×4 grid of accumulator coopmats** (8-16
  accumulators) → AI rises to 32-64 FLOP/byte.
- Stage A and B K-slabs into LDS **once per workgroup** (65536 B of LDS
  available per `maxComputeSharedMemorySize`), then `coopMatLoad` from LDS
  — so global traffic is amortised across all 4 waves and all accumulators.
- Double-buffer the LDS slabs so there's one `barrier()` per K-step
  instead of the current two, and loads overlap MMA.
**Expected**: **2-4x** over the current 4.2-5.0 TFLOP/s, i.e. 10-20
TFLOP/s. This is the single largest absolute gain available on the chip.
**Measure**: GFLOP/s at N=4096 vs §0.1's ceiling; expect utilisation to go
from ~10% to 25-40%.
**Effort**: high (this is a real GEMM kernel). **Value**: highest for
prefill, image generation, and the Parakeet encoder.

### 2.2 W4A8 coopmat — Q4 weights into the *int8* MMA path
**Hypothesis**: the two best GEMM results are being left un-combined.
`coopmat,q8` (int8 MMA, 2x hardware rate) hits 9.6 TOPS cache-resident,
and `coopmat_fused,q4` (4-bit weights, 1/4 the bytes) hits 5.0 TFLOP/s —
but the Q4 path dequantizes to **fp16** and feeds the slower MMA shape
(`gemm_coopmat_q4.comp:67`). Unpacking Q4 nibbles to int8 is *cheaper*
than to fp16 (no float conversion at all — just the mask/subtract from
§1.1) and feeds the 2x-rate MMA.
**Change**: `gemm_coopmat_w4a8.comp` — unpack Q4 → int8 into LDS,
`coopmat<int8_t,...>` × `coopmat<int8_t,...>` → `coopmat<int32_t,...>`,
with per-block scales applied to the int32 accumulator after each block's
worth of K. (Requires block size to be a multiple of TILE_K=16, which
every swept block size already is.)
**Expected**: up to 2x over `coopmat_fused,q4`, at 1/4 the weight bytes of
fp16. Stacked with §2.1 this is the intended production prefill kernel.
**Effort**: high. **Value**: very high.

### 2.3 Test B stored [N,K] with a column-major `coopMatLoad`
**Hypothesis**: B is stored K-major ([K,N]) and loaded row-major, but real
`Linear` weights are stored `[out_features, in_features]` = [N,K], and
RDNA's WMMA B-operand lane layout may favour the transposed load. The
W8A8 GEMM path already stores B transposed for packing reasons — the
coopmat paths don't.
**Change**: store B as [N,K] and load with
`gl_CooperativeMatrixLayoutColumnMajor`; benchmark both.
**Expected**: unclear sign, but it's nearly free to test and removes a
host-side transpose from the engine's weight loader if it wins.
**Effort**: low. **Value**: medium.

### 2.4 Workgroup swizzle / tile reordering for MALL locality
**Hypothesis**: linear `gl_WorkGroupID` ordering walks C row-by-row, so
concurrently-resident workgroups share A rows but stream all of B — poor
reuse in the 32MB MALL. Grouped/Morton ("super-tile") ordering makes the
in-flight set of workgroups a square block of C, maximising shared A and B.
**Change**: remap `gl_WorkGroupID.x/y` through a swizzle in-shader (group
tiles into e.g. 8×8 super-tiles); pure index arithmetic, no structural
change.
**Expected**: 10-30% on large N, more once §2.1 makes tiles bigger. A
standard, well-documented GEMM win.
**Effort**: low. **Value**: medium-high (great effort/reward ratio).

### 2.5 Split-K for skinny shapes
**Hypothesis**: real transformer GEMMs aren't square. When M·N is small
but K is large (e.g. the attention output projection at short sequence
length), there aren't enough tiles to fill 40 CUs, and the current kernels
under-occupy.
**Change**: partition K across G workgroups, each producing a partial C;
combine via a second reduce dispatch (or `atomicAdd` on fp32 C).
**Expected**: large on shapes where the tile count is below ~2x CU count.
**Measure**: needs §3.4's realistic shape sweep to know which shapes matter.
**Effort**: medium. **Value**: medium, shape-dependent.

### 2.6 fp16 output, and fuse epilogue work
**Hypothesis**: every GEMM writes `float c[]` (fp32), doubling store
traffic versus the fp16 the next layer wants, and every fused op (bias,
activation, residual add, the next layer's quantize) is a separate
full-round-trip dispatch through DRAM.
**Change**: an fp16-output variant, plus an epilogue hook in the GEMM
kernel (bias + activation + optional int8-quantize of the result).
**Expected**: ~M·N·2 bytes saved per GEMM, and one whole DRAM round-trip
per fused op eliminated — at 236 GB/s that's directly measurable.
**Effort**: low-medium. **Value**: medium-high.

---

## 3. Kernel fusion and the ops the suite doesn't cover yet

### 3.1 Fused RMSNorm → int8 quantize
Pairs with §1.6. One kernel that reads `x`, computes the RMS, and writes
*both* the normalised fp16 and the packed-int8 + scale. Eliminates a full
read+write of the activation vector and one dispatch+barrier per layer.
For W8A8 to be viable this fusion is mandatory, not an optimisation.
**Effort**: low. **Value**: high.

### 3.2 Fused SwiGLU / gated FFN
`silu(gate) * up` is currently three DRAM round-trips (two GEMM outputs
read, one written). Fuse into the GEMM epilogue (§2.6) or at minimum into
a single elementwise kernel taking both operands. Elementwise already hits
617 GB/s cache-resident / 236 GB/s DRAM — so these passes are pure
bandwidth and the only win is *not doing them*.
**Effort**: low. **Value**: medium-high.

### 3.3 Attention is completely absent from the suite — add it
**Gap**: there is no attention kernel at all. For decode at long context,
attention over the KV cache is the *dominant* memory consumer, more than
the weights. For prefill it's a big chunk of the FLOPs.
**Change**: benchmark (a) naive multi-dispatch attention — QKᵀ GEMM,
softmax, PV GEMM — versus (b) a fused flash-attention-style kernel with
online softmax keeping the K-tile in LDS and using coopmat for both
matmuls. Sweep sequence length 128…8192, plus GQA group sizes.
**Expected**: the standard flash-attention result is a large win, and it's
amplified here because the standalone `softmax` kernel measures a poor
**104 GB/s** — the fused version never materialises the score matrix at all.
**Effort**: high. **Value**: very high — it's a whole missing pillar.

### 3.4 Benchmark the *real* shapes from the target models
**Gap**: the sweep is square N×N×N, which no transformer layer is.
**Change**: pull the actual dims from the four models in `GOALS.md`
(Qwen3-Next hidden/FFN/expert dims and head config; Parakeet's conformer
encoder; Kokoro; Z-Image) and sweep those exact (M,N,K) triples plus
decode (M=1) and prefill (M=512/2048) variants.
**Expected**: reveals which kernels matter and which shapes are
pathological (odd dims, non-multiples of 16 needing padding/tail handling
— none of the current kernels handle a K that isn't a multiple of TILE_K,
`gemm_coopmat_fp16.comp:44` silently truncates `pc.K / TILE_K`).
**Effort**: low-medium. **Value**: high — refocuses all other work.

### 3.5 Grouped / MoE GEMM
Qwen3-Next is an MoE model: each token goes to a few of many experts, so
the FFN is a **grouped GEMM** over variable-sized token batches, not one
big GEMM. Benchmark: top-k routing, the gather/scatter of token rows, and a
grouped-GEMM kernel that reads per-expert (offset, count) from a buffer.
Also measure the "one dispatch per expert" naive alternative to quantify
the launch-overhead penalty (§4.1).
**Effort**: high. **Value**: high, and specific to the stated text-gen goal.

### 3.6 Gated DeltaNet / linear-attention chunk kernel
Qwen3-Next mixes linear attention with full attention. The chunked
linear-attention recurrence is a different primitive (chunk-local matmuls
plus a sequential state update) that nothing here covers. Worth a
prototype to find out whether it's matmul-bound (good — coopmat) or
serialised on the state update (bad — needs careful chunking).
**Effort**: high. **Value**: high but only for Qwen3-Next.

### 3.7 Fix the reduction kernels — `subgroup` is *slower* than `shared`
**Anomaly in the data**: `rmsnorm,subgroup` gets 64 GB/s where
`rmsnorm,shared` gets 202 GB/s, and softmax tops out at 104 GB/s. Both are
far below the 236 GB/s these purely-bandwidth ops should reach.
**Hypothesis**: the subgroup variants use one wave64 per row → only 64
lanes and scalar loads per row, so they're latency-bound, not
bandwidth-bound.
**Change**: one workgroup of 256 doing `uvec4`-vectorised loads →
per-subgroup `subgroupAdd` → tiny LDS combine across the 4 subgroups.
**Expected**: 2-3x on both, up to the ~236 GB/s bandwidth wall.
**Effort**: low. **Value**: medium (small ops, but they're per-layer and
currently leaving 2-3x on the floor).

---

## 4. Dispatch overhead and engine-level plumbing

### 4.1 Measure per-dispatch launch + barrier cost
**Hypothesis**: decode is a long chain of *small* kernels. A 48-layer model
with ~10 dispatches per layer is ~500 dispatches per token; at 20 tok/s
that's 10k dispatches/s. If a dispatch+barrier costs even 20µs, overhead
alone caps throughput — and nothing in the suite measures it.
**Change**: time an empty (immediate-return) kernel at 1/10/100/1000
iterations, with and without the `vkCmdPipelineBarrier` that
`shim_dispatch_timed` inserts between iterations (`vk/shim.c:516-522`);
separately time CPU-side command-buffer build + submit + fence wait.
**Expected**: a hard per-dispatch floor in µs. This sets the fusion budget
(§3.1-3.2) and tells us whether pre-recorded/reused command buffers are
mandatory.
**Effort**: low. **Value**: high — an easy-to-miss ceiling.

### 4.2 Pre-recorded command buffers for the decode step
**Hypothesis**: a decode step is the same dispatch graph every token, so it
should be recorded **once** and resubmitted, with only push constants (or
better, a descriptor-indexed buffer of per-step params) changing. The
current harness rebuilds the command buffer per call.
**Change**: record a whole-layer or whole-model command buffer; drive
per-step variation through buffer contents rather than re-recording.
**Expected**: removes most CPU-side overhead from §4.1.
**Effort**: medium (engine change, not a shader).

### 4.3 Barrier granularity
Full `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` memory barriers between every
dispatch (`vk/shim.c:511-521`) drain the pipeline. Many adjacent ops in a
layer are independent (e.g. Q, K, V projections) and could run
concurrently with no barrier at all. Experiment: measure 3 independent
GEMMs with vs without barriers between them; expect meaningful overlap
gains on small shapes where a single GEMM can't fill the GPU.
**Effort**: low. **Value**: medium-high for decode.

---

## 5. APU-specific memory experiments (this is the unusual part of the chip)

### 5.1 Which memory type is fastest for GPU-read-only weights?
**Hypothesis**: the allocator currently prefers `DEVICE_LOCAL|HOST_VISIBLE`
(memory type 3/4 here) for its unified-memory convenience, but on an APU
that choice usually means host-coherent/uncached-write-combine pages, which
can read *slower* from the GPU than plain `DEVICE_LOCAL`. Given the whole
decode path is bandwidth-bound, a 10% difference here is a 10% tok/s
difference. This device exposes 11 memory types across 2 heaps:
- types 0,1: `DEVICE_LOCAL` only (heap 1, 83.8 GiB)
- type 2: `HOST_VISIBLE|HOST_COHERENT` (heap 0, 41.9 GiB)
- types 3,4: `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` ← **what we use now**
- types 5,6: `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` (heap 0)
- types 7-10: the above plus AMD's `DEVICE_COHERENT`/`DEVICE_UNCACHED`
  (`VK_AMD_device_coherent_memory` is supported here)
**Change**: parameterise the bandwidth benchmark by memory type index and
run the 64MB copy from each.
**Expected**: a ranked table. Plausibly `DEVICE_LOCAL`-only wins for
weights, with a staging upload at load time — which is a small change to
the weight loader for a possibly free few-percent on everything.
**Effort**: low. **Value**: high (bandwidth is the binding constraint).

### 5.2 Does the BIOS VRAM carveout matter?
Heap 1 reports 83.8 GiB device-local on a machine with 117 GiB of usable
RAM (and heap 0 another 41.9 GiB) — the two heaps overlap the same physical
memory, so the driver is largely handing out GTT rather than a fixed
carveout.
Experiment: measure bandwidth for buffers small enough to fit a
conventionally-sized carveout vs much larger, and check whether page-size
effects appear. Also test whether huge/2MB pages are in play. Informs how
to lay out a multi-GB model.
**Effort**: medium. **Value**: medium, potentially high for big models.

### 5.3 Image/texture loads vs storage-buffer loads
On AMD, sampler/image reads go through a different path (and on some parts
a different cache) than buffer loads. For read-only weights, a
`readonly image2D`/texel-buffer path is worth one measurement. Long shot,
but cheap and occasionally a surprise win.
**Effort**: low. **Value**: low-medium (lottery ticket).

---

## 6. Occupancy, wave size, and looking at the actual machine code

### 6.1 Dump and read the ISA
`RADV_DEBUG=asm` (verify option names with `RADV_DEBUG=help`) dumps the
generated GCN/RDNA assembly. Use it to confirm:
- `dotPacked4x8EXT` really became `V_DOT4_I32_I8` (this is the whole point
  of the W8A8 path, and the extension was silently *not enabled* until
  last session's `vk/shim.c` fix — worth verifying the fix took effect at
  the instruction level).
- `coopMatMulAdd` really became `V_WMMA_*` and not a scalar fallback.
- The integer-division sequences of §1.2 exist (and then disappear).
- **VGPR/SGPR counts and LDS usage per kernel** → waves/SIMD occupancy.
**Effort**: low. **Value**: high — it converts speculation into fact, and
every item above gets cheaper to evaluate once we can read the ISA.

### 6.2 wave32 vs wave64
**Hypothesis**: this device reports `minSubgroupSize=32`,
`maxSubgroupSize=64`, `subgroupSizeControl=true` with compute in
`requiredSubgroupSizeStages` — so we can *choose*. Everything is currently
written for wave64 (`local_size_x = 64`). RDNA's WMMA 16×16×16 shape maps
naturally onto wave32, and wave32 halves the latency of a dependent
instruction chain and reduces divergence cost; wave64 halves instruction
issue count. Which wins is empirical and differs per kernel.
**Change**: use `VK_EXT_subgroup_size_control`'s
`requiredSubgroupSize=32` pipeline creation flag and re-benchmark the
coopmat GEMMs and the subgroup GEMV/reduction kernels at both sizes.
**Expected**: 0-20% either way, per kernel. Cheap to test, and it's a
per-pipeline knob we'd want to tune once and hard-code.
**Effort**: low-medium (small `vk/shim.c` change). **Value**: medium-high.

### 6.3 Occupancy tuning via workgroup size and LDS budget
Once §6.1 gives VGPR counts, sweep workgroup sizes (64/128/256/512) and
LDS tile sizes for the tiled and coopmat kernels, and check where occupancy
cliffs are. `gemm_tiled.comp` uses TILE×TILE = 16×16 threads with two
fp32 LDS tiles; 65536 B of LDS allows far more. Note `gemm_tiled.comp:74`
stores dequantized Q4/Q8 into **fp32** LDS — using fp16 LDS tiles halves
LDS use and doubles the occupancy headroom for free.
**Effort**: low-medium. **Value**: medium.

### 6.4 Power and sustained-clock behaviour
The iGPU shares a TDP budget with 16 Zen5 cores. Every number in
`results.csv` is from a short kernel on an otherwise-idle machine.
Experiment: run a 60-second sustained GEMM loop while logging
`freq1_input`/`power1_average`, both with the CPU idle and with the CPU
loaded — then report GFLOP/s/W and the sustained-vs-burst ratio. A server
doing tokenisation/sampling on the CPU while the GPU runs will see the
degraded number, not the benchmark number.
**Effort**: low. **Value**: medium-high for the real serving goal.

---

## 7. Accuracy work that has to happen before any of this ships

Perf work is choosing between formats whose *accuracy* is unmeasured — the
`TODO.md` working conclusion already flags this. None of these are perf
experiments, but they gate the format decision:

- **Per-format error metrics**: for each of Q8/Q4/W8A8/W4A8 and each block
  size, measure RMS and max relative error of a full layer's output against
  an fp32 reference — the harness already builds CPU references, so this is
  mostly bookkeeping. This gives an accuracy-vs-GFLOP/s Pareto front
  instead of a speed ranking.
- **Asymmetric (zero-point) vs symmetric Q4**: the current scheme is
  symmetric `[-8,7]` (`gemm_coopmat_q4.comp:43`). Asymmetric costs one more
  value per block and a correction term but typically halves the error;
  worth measuring both cost and benefit.
- **Activation outliers**: W8A8's weak point is per-tensor/per-row dynamic
  activation quantization in the presence of outlier channels. Measure how
  bad it is on real activations before committing to W8A8 for decode.
- **Real perplexity**: ultimately the only test that matters. Needs a
  model loaded, so it comes after the engine exists — but it should be the
  acceptance criterion for the format choice, not benchmark GFLOP/s.

---

## Suggested order of attack

**First (a day's work, unblocks everything):** §0.3 cold/DRAM-resident
benchmarking, §0.2 clock logging, §0.1 the MMA-ceiling microbenchmark,
§6.1 ISA dumps, §1.2 remove integer divisions, §5.1 memory-type bandwidth,
§4.1 dispatch-overhead measurement.

**Then (the two big kernels):** §1.1 W4A8 GEMV for decode, §2.1
register-blocked coopmat GEMM for prefill — and if §2.1 works, §2.2 W4A8
coopmat to combine it with 4-bit weights.

**Then (the missing pillars):** §3.3 attention/flash-attention, §3.4 real
model shapes, §3.1-3.2 fusion, §3.5 MoE grouped GEMM.

**Opportunistically (cheap, independent):** §2.4 workgroup swizzle, §6.2
wave32 vs wave64, §3.7 fixed reduction kernels, §1.3 vectorised loads.

The one-line summary: **DRAM bandwidth is maxed, so decode wins come only
from reading fewer bytes (→ W4A8); the matrix cores are ~90% idle, so
prefill wins come from arithmetic intensity (→ register-blocked WMMA); and
before either, fix the benchmark so it measures DRAM and boost clocks
rather than cache and idle clocks.**
