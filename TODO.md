# TODO / session handoff — Strix Halo Vulkan inference benchmark suite

## Where we are

Goal: benchmark the Vulkan compute operations neural-network inference is
built from (GEMM, GEMV, elementwise, softmax/RMSNorm) across implementation
"flavours" on this specific machine's iGPU (AMD Radeon 8060S / RDNA3.5 /
RADV STRIX_HALO), to figure out how to build an inference engine tuned for
it — with a specific focus on whether ~4-bit quantized weights can be made
to perform well.

**Everything below is implemented, builds clean (`go build ./...`,
`gofmt -l .`, `go vet ./...` all clean), and has passed its correctness
checks against a CPU reference on real hardware.** `results.csv` (433 rows)
is current as of this session's full sweep. Not yet committed — the
previous commit was `a2bf6e7` (block-size follow-up); this session's W8A8 +
tiled-Q8/Q4 work (see below) is new on top of that.

### This session: bonus item 3 from the prior TODO — W8A8 + tiled Q8/Q4

Picked up "Next steps" item 3 (scoped-out bonus items). Both sub-items are
now done, plus a real bug fix found along the way:

- **Bug fix**: `vk/shim.c`'s `shim_create_device` built the
  `VkPhysicalDeviceShaderIntegerDotProductFeatures` struct into the
  `pNext` chain and set `shaderIntegerDotProduct` on it, but never added
  `VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME` to the enabled
  extensions list — required since this instance targets Vulkan 1.2 (the
  feature/extension was only promoted to core in 1.3). The feature was
  being silently requested-but-not-actually-enabled. Fixed by adding it
  alongside the existing coopMatrix extension request.
- **W8A8 GEMV/GEMM** (`shaders/gemv_w8a8.comp`, `shaders/gemm_w8a8.comp`,
  `bench/ops_w8a8.go`): both weights and activations quantized to int8,
  reduced via `VK_KHR_shader_integer_dot_product`'s `dotPacked4x8EXT` —
  RDNA's `V_DOT4_I32_I8`, one instruction for 4 int8 MACs — instead of the
  existing Q8 path's dequantize-then-float-multiply. Both operands are
  stored as plain packed-`uint32` buffers (4 signed int8 lanes per word,
  the same byte layout `quantizeQ8` already produces — no new packing code
  needed, no 8-bit-storage extension needed since nothing is read back as
  `int8_t`). GEMV uses subgroup reduction (compared directly against the
  existing subgroup champion); GEMM uses a naive one-thread-per-element
  kernel — not pitted against the coopmat variants, which go through a
  completely different hardware path already covered by
  `gemm_coopmat_int8.comp`. GEMM's weight operand is stored **transposed**
  (`[N,K]` instead of `[K,N]`) so a column's K-values are contiguous and
  packable — exactly how a real engine stores a `Linear` layer's weight
  (`[out_features, in_features]`), not a benchmark-only contrivance.
  Gated on `PhysicalDevice.SupportedFeatures().IntegerDotProduct`, skipped
  with a message if unsupported (same pattern as the coopmat checks).
- **Q8/Q4 for tiled GEMM** (`shaders/gemm_tiled.comp`): extended with the
  same `PRECISION_Q8`/`PRECISION_Q4` block-dequant-into-shared-memory
  scheme `gemm_naive.comp` already used — B is dequantized once per tile
  fill, the inner product loop is unchanged regardless of weight format.
  Closes the gap where only naive GEMM had Q4.

**Results, both genuinely surprising:**

- **New GEMV decode champion: subgroup W8A8, ~1020 GFLOP/s at N=4096**
  (block=128) — **~2.5-2.6x the previous champion**, subgroup Q4 at ~398
  GFLOP/s. Block-size effect is noisy here (994/903/1020/690/1012/1013
  GFLOP/s at blocks 32/64/128/256/512/1024) rather than the clean
  monotonic trend seen in the coopmat-fused-Q4 case — worth another look
  if this becomes the production path, but even the worst block (690)
  beats every non-W8A8 GEMV format measured.
- **Naive W8A8 GEMM is extremely block-size-sensitive**: 454 GFLOP/s at
  block=32 climbing to 1421 GFLOP/s at block=1024 at N=4096 (~3.1x) —
  same "fewer scale-lookups per coarser block" story as the coopmat-fused
  Q4 finding, but on a kernel with no shared-memory tiling at all. At its
  best block, naive W8A8 (1421) beats every other *naive* GEMM format by
  1.7-3x (naive fp16 825, naive q8/q4 445-536) and gets within ~13% of
  *tiled* fp32 (1623) — the packed-dot instruction alone recovers most of
  what tiling buys elsewhere.
- **Tiled Q8/Q4 GEMM lands right where expected**: 2400-2620 GFLOP/s at
  N=4096 across all block sizes (flat — block size barely matters here,
  unlike naive W8A8 or coopmat-fused Q4, because the dequant happens once
  per shared-memory tile-fill regardless of block granularity), matching
  tiled fp32's 1623-2757 fp16 range and confirming quantized weights carry
  no penalty once tiled — same story tiled already told for fp32 vs fp16.

**This changes the "Working conclusion" below**: W8A8 (packed dot-product,
not coopmat) is now the recommended decode-shape GEMV path, not Q4.

### What got built

- **`vk/`** — the engine, moved out of `package main` so `cmd/bench` can
  import it too. Extended well beyond the original PoC: optional device
  features (fp16, int8, integer dot-product, cooperative-matrix) negotiated
  at `NewDevice`, N-buffer pipelines with push constants and specialization
  constants (`Device.NewPipeline`/`vk.PipelineSpec`), GPU-timestamp-based
  timing (`ComputePipeline.DispatchTimed`), buffer allocation now prefers
  this APU's `DEVICE_LOCAL|HOST_VISIBLE` unified-memory type,
  `PhysicalDevice.CooperativeMatrixShapes()` queries which MxNxK/type/scope
  combos `VK_KHR_cooperative_matrix` actually supports here (discovered:
  16x16x16, subgroup scope, both fp16→fp32 and int8→int32).
- **`shaders/`** — 17 `.comp` sources compiling to 30 SPIR-V variants
  (precision/tile variants generated via `glslc -D`, not duplicated GLSL):
  bandwidth baseline, elementwise/relu, GEMV (naive/subgroup ×
  fp32/fp16/q8/q4, plus a subgroup W8A8 variant), GEMM (naive/tiled ×
  fp32/fp16/q8/q4, cooperative-matrix fp16/int8/q4, plus a naive W8A8
  variant), RMSNorm/softmax (shared-memory/subgroup reduction), and a
  standalone Q4→fp16 dequant kernel. W8A8 (both operands int8) reduces via
  `VK_KHR_shader_integer_dot_product`'s `dotPacked4x8EXT` instead of
  dequant-and-multiply — see this session's notes above.
- **`bench/`** — the harness: `bench.go` (timing + adaptive iteration
  capping, see "GPU watchdog" below), `quant.go` (fp16 conversion with
  correct round-to-nearest, GGML-style Q8/Q4 block quantize/dequantize),
  `ops_*.go` (one file per op family wiring shaders + buffers + a CPU
  correctness check to a size sweep).
- **`cmd/bench`** — the CLI. `go run ./cmd/bench -h` for flags (`-sizes`,
  `-bwsizes`, `-blocks`, `-warmup`, `-iters`, `-csv`, `-skip`).
- Original demo (`main.go`, `./strix-halo-vulkan`) still works unchanged.

### Two real bugs hit and fixed along the way (worth remembering)

1. **GPU driver hang watchdog**: naive GEMM at N=4096 batched into 20
   back-to-back dispatches in one command buffer took long enough to trip
   amdgpu's TDR → `VK_ERROR_DEVICE_LOST`, which poisons the device for
   every case still queued. Fixed in `bench.TimeDispatch`: it now probes
   with 1 iteration first and caps any batch to ~500ms worth of iterations.
2. **O(N³) CPU reference GEMM inside the size sweep**: `runGEMMCoopMatQ4TwoPass`
   originally ran a full CPU-side `cpuGEMM` correctness check at *every*
   swept size instead of once at a small size like every other variant does.
   At N=4096 that's ~137 billion scalar float ops in pure Go — the process
   sat at 98% CPU for 18+ minutes doing nothing but this before I caught it.
   Fixed by moving it to a one-time `verifyCoopMatQ4TwoPass` before the
   sweep. **If adding a new op/variant, verify once at a small fixed size,
   never inside the sweep loop.**

### Headline findings (full run: `results.csv` in the repo root, 433 rows)

- **Sustained memory bandwidth: ~236 GB/s** (measured at 64MB buffers, past
  cache effects — smaller buffers read 700+ GB/s from cache, not DRAM).
- **GEMV (decode-shape, N=4096)**: subgroup reduction beats naive by
  ~5-8x at every precision. Q4 was the previous champion at 399 GFLOP/s;
  **W8A8 (packed dot-product) is now the champion at ~1020 GFLOP/s**
  (block=128) — see this session's notes above for the full block-size
  breakdown and its caveats.
- **GEMM (N=4096)**: naive ~445-825 GFLOP/s depending on precision (fp16
  fastest of the non-W8A8 naive formats, q8/q4 slowest — dequant overhead
  dominates at this granularity), naive W8A8 445-1421 GFLOP/s depending
  heavily on quantization block size, tiled ~1600-2760 GFLOP/s (now
  including q8/q4, flat across block size — see this session's notes),
  cooperative-matrix 4200-7700 GFLOP/s — coopmat is still the dominant
  lever for GEMM, 1.5-5x over hand-tiled.
- **The Q4 answer** (what we set out to find): dequantizing Q4→fp16 once
  and reusing for coopmat (`coopmat_dequant`) runs at **~4170-4210 GFLOP/s
  at N=4096 regardless of block size — indistinguishable from plain fp16
  coopmat (4196 GFLOP/s)**, because the dequant itself is cheap (~215µs at
  N=4096, ~200GB/s, vs. ~32.7ms for the matmul — under 1% overhead,
  one-time per weight load). It's flat across block size because block
  size only affects the one-time dequant pass, not the matmul that follows.
  The single-pass fused kernel (`coopmat_fused`, dequantizes each K-tile
  into shared memory inline, no scratch buffer) climbs hard with block
  size and **wins outright once block ≥256: 3267 (32) → 4558 (64) → 4753
  (128) → 4871 (256) → 4915 (512) → 4878 (1024) GFLOP/s at N=4096** —
  beating plain fp16 coopmat (4196) *and* coopmat int8 (4705), making
  fused Q4 at block 256-512 the single fastest GEMM variant measured on
  this chip. Confirmed as a real, plateauing trend, not noise (see
  resolved item below) — driven by fewer scale-lookups/branches per
  shared-memory tile fill as blocks get coarser; it flattens (and dips
  fractionally) past ~512, plausibly register/shared-memory pressure from
  holding a larger per-tile scale table offsetting the win.

**Working conclusion (updated this session)**: for an inference engine on
this chip, store weights as Q4 (8x smaller than fp32) for the *matmul*
(prefill/large-batch) path — the fused single-pass Q4 coopmat kernel with
block size 256-512 is still the fastest GEMM variant found here (~4900
GFLOP/s @ N=4096) — but for *decode* (the memory-bound GEMV shape that
dominates autoregressive generation), **W8A8 via packed dot-product now
beats Q4 by ~2.5x** (~1020 vs ~399 GFLOP/s @ N=4096) and should be the
default there instead. Both quantization schemes trade off weight/activation
accuracy in a real model that this benchmark doesn't measure (Q4's wider
dynamic range per block; W8A8's dynamic activation quantization introducing
its own error) — worth validating against a real model's perplexity before
committing either into production. If small Q4 blocks (32-64) are needed
for matmul accuracy, the two-pass dequant-once approach is the better
choice there since it's block-size-agnostic (~4200 GFLOP/s regardless).

## Next steps (pick up here)

1. ~~Investigate the fused-kernel block-size effect~~ **Done.** Real,
   plateauing trend, not noise — see headline findings above.
2. ~~Scoped-out bonus items: W8A8 GEMV/GEMM, Q8/Q4 tiled GEMM~~ **Done this
   session** — see the notes at the top of this file. Also fixed a real
   bug along the way: `VK_KHR_shader_integer_dot_product` was queried and
   requested via the feature struct but never actually enabled as a device
   extension (`vk/shim.c`).
3. `results.csv` in the repo root is current as of this session's full
   sweep (433 rows; re-run via
   `go run ./cmd/bench -blocks 32,64,128,256,512,1024 -csv results.csv`).
   Re-run again if shaders/harness change before the next session.
4. The W8A8 GEMV block-size effect (994/903/1020/690/1012/1013 GFLOP/s
   across blocks 32-1024) is noisy rather than a clean trend — worth
   investigating if W8A8 GEMV becomes the production decode path, same way
   the coopmat-fused-Q4 block effect got investigated and confirmed real
   last session.
5. Consider a visual report/dashboard from the CSV (offered earlier, not
   done) — with W8A8 now in the mix this is probably the right next
   deliverable if the user wants one.
6. Not yet committed to git — the last commit was `a2bf6e7`. Commit this
   session's work (shim.c fix, new shaders, bench harness, README/TODO,
   results.csv) if the user wants a commit.
