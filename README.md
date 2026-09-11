# strix-halo-vulkan

A minimal Go + Vulkan compute harness for exploring shaders on the Strix Halo
iGPU (gfx1151 / AMD Radeon 8060S, RADV driver).

No third-party Go modules — `go.mod` has no dependencies. Vulkan access is a
hand-written cgo binding straight against the system Vulkan loader
(`vulkan/vulkan.h` + `libvulkan.so`), with the struct-heavy parts of the
Vulkan API built in a small C shim (`shim.c`/`shim.h`) rather than Go, since
cgo forbids passing a Go struct across the boundary when one of its fields
is itself a pointer into other Go memory — and Vulkan's `*CreateInfo`
structs are built almost entirely out of such chains.

## Layout

- `shim.c` / `shim.h` — the actual Vulkan calls (instance/device setup,
  buffer + memory allocation, pipeline creation, dispatch).
- `engine.go` — thin Go wrapper giving the shim idiomatic types
  (`Instance`, `Device`, `Buffer`, `ShaderModule`, `ComputePipeline`).
- `main.go` — the demo: picks the Strix Halo iGPU, uploads a float array,
  runs `shaders/double.comp`, reads it back, verifies it.
- `shaders/double.comp` — the compute shader (GLSL). `shaders/shaders.go`
  embeds its compiled SPIR-V via `go:embed`.

## Requirements

- Go, gcc/clang, `pkg-config`
- Vulkan loader + headers (`vulkan-icd-loader`, `vulkan-headers` on Arch)
- `glslc` (from `shaderc`) to compile shaders — only needed when a `.comp`
  file changes, not to build/run the Go program itself.

## Usage

```sh
go generate ./...   # only needed after editing a .comp file
go build .
./strix-halo-vulkan
```

## Adding a new shader

1. Write `shaders/foo.comp`.
2. Add a `//go:generate glslc ... -o foo.spv foo.comp` line and a
   `//go:embed foo.spv` var in `shaders/shaders.go`.
3. Run `go generate ./...`.
4. Build a pipeline for it — `NewComputePipeline` in `engine.go` currently
   assumes one storage-buffer binding (binding 0); extend `shim.c`'s
   `shim_create_compute_pipeline` if a shader needs a different layout
   (more bindings, push constants, etc).
