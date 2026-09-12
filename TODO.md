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
checks against a CPU reference on real hardware.** Not yet committed to
git — `git status` shows the rename (`engine.go`/`shim.*` → `vk/`) staged
and everything else (bench/, cmd/, new shaders, README) untracked/modified.

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
- **`shaders/`** — 20 `.comp` sources compiling to ~28 SPIR-V variants
  (precision/tile variants generated via `glslc -D`, not duplicated GLSL):
  bandwidth baseline, elementwise/relu, GEMV (naive/subgroup ×
  fp32/fp16/q8/q4), GEMM (naive/tiled × fp32/fp16/q8, naive q4, cooperative-
  matrix fp16/int8/q4), RMSNorm/softmax (shared-memory/subgroup reduction),
  and a standalone Q4→fp16 dequant kernel.
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

### Headline findings (full run: `results.csv` in the repo root, 217 rows)

- **Sustained memory bandwidth: ~236 GB/s** (measured at 64MB buffers, past
  cache effects — smaller buffers read 700+ GB/s from cache, not DRAM).
- **GEMV (decode-shape, N=4096)**: subgroup reduction beats naive by
  ~5-8x at every precision. Best: **subgroup + q4, 399 GFLOP/s** at only
  ~112 GB/s (vs. subgroup fp16's 355 GFLOP/s at 355 GB/s) — Q4 is not just
  smaller, it's actually the *fastest* GEMV format here, not only the most
  memory-efficient.
- **GEMM (N=4096)**: naive ~430-530 GFLOP/s regardless of precision, tiled
  ~1600-2760 GFLOP/s, cooperative-matrix 4200-7700 GFLOP/s — coopmat is the
  dominant lever, 3-9x over hand-tiled.
- **The Q4 answer** (what we set out to find): dequantizing Q4→fp16 once
  and reusing for coopmat (`coopmat_dequant`) runs at **~4170-4210 GFLOP/s
  at N=4096 — indistinguishable from plain fp16 coopmat (4201 GFLOP/s)**,
  because the dequant itself is cheap (~215µs at N=4096, ~200GB/s, vs.
  ~32.7ms for the matmul — under 1% overhead, one-time per weight load).
  The single-pass fused kernel (`coopmat_fused`, dequantizes each K-tile
  into shared memory inline, no scratch buffer) is competitive and, with
  **block size 128, actually wins at N=4096: 4828 GFLOP/s**, edging out
  even the two-pass approach and plain fp16 coopmat. Smaller blocks (32)
  hurt the fused kernel more (3278 GFLOP/s) — worth another look before
  calling that final (see below).

**Working conclusion**: for an inference engine on this chip, store weights
as Q4 (8x smaller than fp32) and either (a) dequantize once to fp16 at load
time and run coopmat fp16 matmul, or (b) use the fused single-pass Q4
coopmat kernel with block size ≥128 — both get full cooperative-matrix
throughput. Use subgroup-reduced Q4 GEMV for decode.

## Next steps (pick up here)

1. **Investigate the fused-kernel block-size effect** (32 → 3278, 64 →
   4547, 128 → 4828 GFLOP/s at N=4096) — is this a real trend (larger block
   = fewer scale lookups/branches per shared-memory fill) or noise? Try
   block=256 to see if it keeps climbing or plateaus/reverses.
2. `results.csv` in the repo root is current as of this session; re-run
   (`go run ./cmd/bench -csv results.csv`, ~3-4 min now that the O(N³) bug
   is fixed) if shaders/harness change before the next session.
3. Scoped-out bonus items from the original plan, not yet built:
   - W8A8 full-integer GEMV/GEMM (both operands int8, using
     `VK_KHR_shader_integer_dot_product`'s packed dot instructions instead
     of dequant-and-multiply) — flagged as bonus in the plan, still open.
   - Q8/Q4 for the shared-memory-tiled GEMM variant (only naive has Q4;
     tiled only has fp32/fp16 today).
4. Consider a visual report/dashboard from the CSV (offered earlier, not
   done) — probably the right final deliverable once the block-size
   question above is settled.
5. Nothing has been committed to git yet — decide on commit structure
   (probably: one commit for the `vk/` restructure, one for the shader
   library + bench harness) when ready.
