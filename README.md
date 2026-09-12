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

Two kernels here are the current answers for the two shapes inference comes
in. For **decode** (GEMV, memory-bound), **W4A8**
(`shaders/gemv_w4a8.comp`): 4-bit weights fed to the packed-int8 dot
instruction through its mixed-signedness overload, at **819 GFLOP/s and 211
GB/s against DRAM-resident weights — 89% of this machine's 236 GB/s memory
bandwidth**, 2.4x the int8-weight kernel it replaces. For **prefill** (GEMM,
compute-bound), the register-blocked cooperative-matrix GEMM
(`shaders/gemm_wmma.comp`): **28.4 TFLOP/s, 51% of this chip's measured
55.5 TFLOP/s of matrix-core throughput** (26.0 TFLOP/s at N=4096), 6.2x the
straightforward coopmat kernel it replaces at that shape. The last 1.13x of
that came from neither tiling nor instruction selection but from *operand
strides*: every fragment load in a WMMA GEMM is K-strided, a power-of-two
leading dimension aims all the addresses of one load at the same memory
channel, and padding each stride 256 B off a 4 KB multiple — a host-side
allocation change — is worth up to 1.35x on the kernels that were already
winning and 2.6x on the [N,K] weight layout.

The `stride` family then measured that memory system directly
(`shaders/strided_read.comp`) and found a second, larger effect with no
kernel-side cause at all: channel interleave on this part is 256 B across
sixteen channels, so a row shorter than `gcd(stride, 4096)` bytes **never
addresses some of them**, and the achievable fraction of DRAM bandwidth is
exactly `min(1, rowBytes/gcd(stride, 4096))`. A K=1024 fp16 weight matrix
whose rows were aligned to 4 KB reads at **half** this chip's bandwidth; 1 KB
rows at an 8 KB stride, a **quarter** — and unlike the aliasing above this
hits a plainly contiguous read as hard as a gather.

Swapping the probe's traversal so consecutive *waves* start on consecutive
rows — the same bytes, the same instructions, the same checksum — then
generalised that into one law about concurrency: the achievable
fraction of peak is `min(1, C/gcd(stride, 4096))` where **C is the
contiguous run of bytes the requests in flight at one moment hold in a
row**, which is the whole row only when consecutive waves walk along it. A
*densely* packed tensor is not safe after all: 1 KB rows packed at a 1 KB
stride, read with 256 B of each row in flight, run at **61 of 243 GB/s**,
with no padding anywhere to blame. One wave streaming a *whole* row
— the GEMV kernels' shape — is clean at every stride, and 4 KB of a row in
flight per wave restores full bandwidth whatever the stride is. What is
exposed is tiling: a 1 KB-wide panel of a K=4096 fp16 matrix gets a quarter
of the bus. Pad every row stride to 256 B past a multiple of 4 KB and all
three effects go away, for every panel width and every traversal. `IDEAS.md`
explains how each kernel gets where it is and what is still on the table.

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
- `cmd/probe/main.go` — dispatches a single `.spv` once, so
  `RADV_DEBUG=asm` / `RADV_DEBUG=shaderstats` can be pointed at any shader to
  read its disassembly or its VGPR/LDS/spill counts. Compilation-time
  questions only; it binds dummy buffers and computes nothing meaningful.
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

Four op families exist to keep the rest of the suite interpretable. They
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
- **`stride`** measures what the memory system delivers for a *fixed* set of
  bytes as the row stride and the request shape vary — the qualifier on
  `bandwidth`, which only ever measures a contiguous sweep and therefore
  reports best-case ceilings a strided kernel does not automatically get. One
  kernel (`shaders/strided_read.comp`) reads every touched 16-byte chunk
  exactly once in every variant, so the byte set and the load count are
  identical and only the stride and the lane→address mapping change;
  `LANES_PER_ROW` sets how many consecutive chunks of a row go to
  consecutive lanes, so one request spans 1 row (contiguous) up to 64 rows (a
  gather); `CROSS_WAVE` swaps the traversal so consecutive *requests* advance
  down the rows rather than along one, which is the only way to put
  concurrent waves a stride apart; and `LOADS_PER_WAVE` gives each request
  more of its row in flight, up to a whole row, which is what a real kernel's
  inner loop does. `-stridepads`, `-striderowbytes` and `-stridefootprints`
  set the three sweeps; the printed grid carries a `model` row giving
  `min(1, C/gcd(stride, 4096))`, and the cross-wave grid under it carries the
  same model per shape, which the measurements track to within 2% wherever C
  is a genuine contiguous run and only indicatively for the gather shapes. Each case is checked against a host checksum
  over its (row, chunk) set, so a mis-derived index fails instead of
  reporting a plausible bandwidth — and the shapes where both traversals
  compile to the same mapping are marked, since those cells measure
  repeatability rather than traversal.
- **`gemv_cold`** measures the decode-shape GEMV against weights several
  times larger than the last-level cache. The square `gemv` sweep's largest
  case holds 33.5MB of fp16 weights (16.8MB at int8, 8.4MB at int4), which
  fits this chip's ~32MB MALL, and then re-reads it hundreds of times — so
  it reports bandwidth above what the DRAM bus can deliver and flatters the
  formats that read *more* bytes, since cached bytes are nearly free. Real
  decode streams a whole model from DRAM per token. `-coldfootprints` sets
  the swept weight footprints in MB (fp16-equivalent, so every format in a
  row holds the same number of weights), `-coldn` the reduction length.

Three measurement hazards the harness handles rather than leaves to the
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
- **Operand stride.** A sweep over powers of two sweeps, by construction,
  only the strides that alias worst on this memory system. Two distinct
  things go wrong there (IDEAS §2.3, §5.1b): a load whose addresses are all
  one stride apart aims them at a single 256 B-interleaved channel when that
  stride is a multiple of the 4 KB interleave rotation, costing up to 1.34x
  of bandwidth and up to 1.6x on a WMMA GEMM; and — the larger one —
  whenever the bytes the concurrent waves hold in flight cover less of the
  4 KB rotation than `gcd(stride, 4096)`, the remaining channels are never
  addressed at all, costing 2x, 4x or 8x even for a contiguous read and even
  with no padding present. The `gemm` family's WMMA cases therefore take
  both leading dimensions as push constants and carry `strideA`/`strideB` in
  their `detail` column, and the `_pada128`/`_pad*` rows are the same SPIR-V
  at a padded stride — so a row's stride is visible rather than implied by
  its size. Nothing outside that family has been measured against a padded
  stride yet; assume its numbers are the aliased ones.
- **Cache-resident measurements are contended.** A MALL-resident working
  set is shared with everything else touching memory — the display this iGPU
  also drives, or a second benchmark process — and losing part of it drops a
  case towards DRAM speed. With another benchmark running alongside, one in
  ten MALL-resident `stride` cases lands at 0.3-0.5x, on a different cell
  each run, with sclk and fclk both pinned; with sole use of the GPU the
  large dropouts vanish and ~10% one-sided scatter remains. Contention can
  only make a case slower, so that family runs each case three times and
  keeps the fastest. DRAM-resident cases reproduce to within 2% and are
  unaffected. Run one benchmark at a time, and treat cache-resident numbers
  elsewhere in the suite as carrying the same one-sided noise.
- **The first cache-resident case after a host fill is not a measurement.**
  Filling a 128 MiB buffer from the CPU cost the next case 13-17%,
  reproducibly by *position* rather than by access pattern: it came out at
  exactly 782 GB/s in all three row-length groups of one run, while a
  shape-identical twin measured immediately afterwards read 942 and the
  clock differed by 1%. `RunStride` now runs and discards one case per
  group, which puts those cells back at 884-950. Writeback from the fill
  competing for DRAM is the likeliest cause; it was never visible in the
  DRAM-resident groups.

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
