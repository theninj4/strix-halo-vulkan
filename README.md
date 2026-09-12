# strix-halo-vulkan

A Go + Vulkan compute harness for exploring shaders on the Strix Halo iGPU
(gfx1151 / AMD Radeon 8060S, RADV driver), plus a benchmark suite
(`cmd/bench`) that measures the operations neural-network inference is
built from — GEMM, GEMV, elementwise/activation, softmax/RMSNorm — across
the implementation "flavours" that matter on this hardware: naive vs.
shared-memory-tiled vs. subgroup-reduced vs. cooperative-matrix
(`VK_KHR_cooperative_matrix`, RDNA3.5's matrix-multiply accelerator), and
fp32 vs. fp16 vs. quantized (int8/int4, GGML-style block scales) weights.

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
  block scales (`quant.go`), and one file per op family (`ops_*.go`) wiring
  shader variants + buffer layouts + a CPU-reference correctness check to
  a size sweep.
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
`op,variant,weight_format,block_size,size,ns_per_iter,gflops,gbps` rows for
further analysis (e.g. "which quantization block size + matrix shape wins").

A GPU-hang watchdog note: `bench.TimeDispatch` probes each case with one
dispatch before batching more into a single timed submission, capping the
batch so it can't run long enough to trip the kernel driver's GPU-hang
timeout (amdgpu's default TDR is commonly ~10s) — a slow kernel (e.g. naive
GEMM at large sizes) is simply measured with fewer iterations rather than
risking `VK_ERROR_DEVICE_LOST`, which would otherwise poison the device for
every case still queued.

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
   `bench.TimeDispatch`.
