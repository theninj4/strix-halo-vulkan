# strix-halo-vulkan

A Go + Vulkan compute harness for exploring shaders on the Strix Halo iGPU
(gfx1151 / AMD Radeon 8060S, RADV driver), plus a benchmark suite
(`cmd/bench`) that measures the operations neural-network inference is
built from — GEMM, GEMV, elementwise/activation, softmax/RMSNorm — across
the implementation "flavours" that matter on this hardware: naive vs.
shared-memory-tiled vs. subgroup-reduced vs. cooperative-matrix
(`VK_KHR_cooperative_matrix`, RDNA3.5's matrix-multiply accelerator), and
fp32 vs. fp16 vs. quantized (int8/int4, GGML-style block scales) weights vs.
W8A8 and W4A8 (int8 or 4-bit weights against int8 activations, reduced via
`VK_KHR_shader_integer_dot_product`'s packed-dot instructions).

The fastest decode kernel here so far is **W4A8 GEMV**
(`shaders/gemv_w4a8.comp`): 4-bit weights fed to the packed-int8 dot
instruction through its mixed-signedness overload, at **819 GFLOP/s and 211
GB/s against DRAM-resident weights — 89% of this machine's 236 GB/s memory
bandwidth**, 2.4x the int8-weight kernel it replaces. `IDEAS.md` explains
how it gets there and what is still on the table.

No third-party Go modules — `go.mod` has no dependencies. Vulkan access is a
hand-written cgo binding straight against the system Vulkan loader
(`vulkan/vulkan.h` + `libvulkan.so`), with the struct-heavy parts of the
Vulkan API built in a small C shim (`vk/shim.c`/`vk/shim.h`) rather than Go,
since cgo forbids passing a Go struct across the boundary when one of its
fields is itself a pointer into other Go memory — and Vulkan's
`*CreateInfo` structs are built almost entirely out of such chains.

## Layout

- `vk/` — the engine: `shim.c`/`shim.h` do the actual Vulkan calls
  (instance/device setup with optional fp16/int8/cooperative-matrix device
  features, buffer allocation preferring device-local+host-visible memory,
  N-buffer pipeline creation with push constants and specialization
  constants, GPU-timestamp-timed dispatch); `engine.go` is the idiomatic Go
  wrapper (`Instance`, `Device`, `Buffer`, `ShaderModule`, `ComputePipeline`).
- `main.go` — the original demo: picks the Strix Halo iGPU, uploads a float
  array, runs `shaders/double.comp`, reads it back, verifies it.
- `shaders/*.comp` — the compute shaders (GLSL); `shaders/shaders.go`
  embeds their compiled SPIR-V via `go:embed`. Precision/tile-size variants
  of the same source are generated via `glslc -D` flags, not duplicated GLSL.
- `bench/` — the benchmark harness: GPU-timestamp-based timing
  (`bench.go`), fp16/int8/int4 quantization helpers matching GGML-style
  block scales (`quant.go`), clock/power instrumentation (`sysmon.go`), and
  one file per op family (`ops_*.go`) wiring shader variants + buffer
  layouts + a CPU-reference correctness check to a size sweep.
- `cmd/bench/main.go` — the benchmark CLI.

## Requirements

- Go, gcc/clang, `pkg-config`
- Vulkan loader + headers (`vulkan-icd-loader`, `vulkan-headers` on Arch)
- `glslc` (from `shaderc`) to compile shaders — only needed when a `.comp`
  file changes, not to build/run the Go programs themselves.

## Usage

```sh
go generate ./...   # only needed after editing a .comp file
go build ./...

./strix-halo-vulkan   # original demo

go run ./cmd/bench    # benchmark suite; see -h for flags (sizes, blocks,
                      # warmup/iters, -csv output, -skip op families)
```

`cmd/bench` prints a results table per op family and, with `-csv`, writes
`op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps` rows,
followed by `sclk_mhz,sclk_mhz_min,sclk_mhz_max,power_w,detail`, for further
analysis (e.g. "which quantization block size + matrix shape wins").

## Reading the numbers honestly

Three op families exist to keep the rest of the suite interpretable. They
are cheap to run and worth running first.

- **`peak`** measures the hardware's instruction-issue ceilings with kernels
  whose operands are register-resident before the loop, so nothing in the
  timed loop touches memory (`shaders/alu_peak.comp`): scalar fp32 FMA,
  packed-fp16 FMA, `dotPacked4x8` (int8 packed dot), and cooperative-matrix
  fp16 and int8. Every other kernel's GFLOP/s should be read as a fraction
  of the relevant one of these — not of a spec-sheet figure extrapolated
  from a different chip. `-cus` (default 40) expresses each ceiling as
  ops/clock/CU, the form comparable against published RDNA3 figures.
- **`overhead`** measures a shader that does nothing, isolating the cost of
  a dispatch plus the pipeline barrier between iterations from any real
  work. It bounds how much a long chain of small kernels (decode) can lose
  to launch cost, and explains bandwidth measurements at sizes small enough
  to be dominated by a fixed cost.
- **`gemv_cold`** measures the decode-shape GEMV against weights several
  times larger than the last-level cache. The square `gemv` sweep's largest
  case holds 33.5MB of fp16 weights (16.8MB at int8, 8.4MB at int4), which
  fits this chip's ~32MB MALL, and then re-reads it hundreds of times — so
  it reports bandwidth above what the DRAM bus can deliver and flatters the
  formats that read *more* bytes, since cached bytes are nearly free. Real
  decode streams a whole model from DRAM per token. `-coldfootprints` sets
  the swept weight footprints in MB (fp16-equivalent, so every format in a
  row holds the same number of weights), `-coldn` the reduction length.

Two measurement hazards the harness handles rather than leaves to the
reader:

- **Clock.** This GPU idles at 637 MHz out of 2900 (see
  `/sys/class/drm/cardN/device/pp_dpm_sclk`), and host-side work between
  cases — generating and quantizing hundreds of MB of weights — lets it
  drop back. Before each timed measurement the harness drives the GPU with
  ALU-heavy work until the clock reaches 97% of its advertised maximum
  (`-warmclock` bounds how long it will try; it stops as soon as the target
  is reached, so a hot GPU costs one 20ms burst), and records the clock and
  power actually observed during the measurement in every result row. An
  earlier sweep without this reported the same kernel at 190 GFLOP/s and
  522 GFLOP/s depending only on how long the preceding case had kept the
  CPU busy.
- **GPU-hang watchdog.** `bench.TimeDispatch` probes each case with one
  dispatch before batching more into a single timed submission, capping the
  batch so it can't run long enough to trip the kernel driver's GPU-hang
  timeout (amdgpu's default TDR is commonly ~10s) — a slow kernel (e.g.
  naive GEMM at large sizes) is simply measured with fewer iterations
  rather than risking `VK_ERROR_DEVICE_LOST`, which would otherwise poison
  the device for every case still queued.

## Adding a new shader

1. Write `shaders/foo.comp`.
2. Add a `//go:generate glslc ... -o foo.spv foo.comp` line and a
   `//go:embed foo.spv` var in `shaders/shaders.go`.
3. Run `go generate ./...`.
4. Build a pipeline for it via `Device.NewPipeline(shader, vk.PipelineSpec{...})`
   in `vk/engine.go` — it takes any number of storage-buffer bindings, an
   optional push-constant size, and optional specialization constants.
5. To benchmark it, add a `bench/ops_*.go` case following the existing
   ones: build buffers/pipeline, run one untimed dispatch and compare
   against a CPU reference before trusting any timing, then call
   `bench.TimeDispatch`. Verify once at a small fixed size, never inside
   the size sweep — an O(N³) CPU reference GEMM at N=4096 is ~137 billion
   scalar operations in pure Go.
