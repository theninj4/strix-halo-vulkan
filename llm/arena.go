package llm

// Where an activation arena's memory comes from (LLM.md L6b).
//
// Every block in this vertical is four buffers: an fp32 weight arena, an fp32
// activation arena, an fp16 activation arena and an fp16 weight bank. The two
// weight buffers are written once at staging and never read back; the two
// activation arenas are written and read every run, and until L6b nothing had
// ever read one on a hot path — the block tests read a tensor once at the end
// and the profilers time the GPU, not the wall clock.
//
// The graph does read them, twice a sublayer, and on this device that turned
// out to be the whole cost of it. `vkGetPhysicalDeviceMemoryProperties` here
// offers six host-visible types that can back a storage buffer, and the one
// `NewBuffer` prefers — DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT — is
// write-combined:
//
//	type  flags                                              read      write
//	   2  HOST_VISIBLE|HOST_COHERENT                     0.18 GB/s  50.8 GB/s
//	   3  DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT        0.18 GB/s  49.6 GB/s
//	   5  HOST_VISIBLE|HOST_COHERENT|HOST_CACHED        24.76 GB/s  70.7 GB/s
//	   8  ...|DEVICE_COHERENT|DEVICE_UNCACHED            0.18 GB/s  49.9 GB/s
//	   9  DEVICE_LOCAL|...|DEVICE_UNCACHED               0.18 GB/s  50.3 GB/s
//	  10  ...|HOST_CACHED|DEVICE_COHERENT|DEVICE_UNCACHED  26.18     66.2 GB/s
//
// **138x on the read**, and the write is faster too. L0b already established
// that the *device* reads every one of these types at 236-237 GB/s, so this
// is a free choice on the GPU side and a decisive one on the host side.
//
// So the activation arenas come from a HOST_CACHED type and the weight banks
// do not — a bank is written once, where the cached type buys only its 1.4x
// write, and leaving it on the device-local type keeps L0a's and L6a's
// residency measurements measuring the same allocation they did before.
//
// `LLM_ARENA_UNCACHED=1` puts the arenas back on the write-combined type. It
// is the control: "the cached type costs the kernels nothing" is a claim
// about GPU time, and a claim like that needs a run with the knob the other
// way rather than an appeal to L0b.

import (
	"os"
	"sync"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

var arenaUncached = sync.OnceValue(func() bool {
	return os.Getenv("LLM_ARENA_UNCACHED") == "1"
})

// newArena allocates an activation arena: HOST_CACHED where the device has
// such a type, which is where a graph's read-backs get their bandwidth.
func newArena(dev *vk.Device, bytes int) (*vk.Buffer, error) {
	if arenaUncached() {
		return dev.NewBuffer(bytes)
	}
	return dev.NewHostCachedBuffer(bytes)
}

// narrowRows converts an activation into the fp16 A operand the matrix cores
// want: [rows][width] f32 in, [rows][lda] halves out, pad columns left zero.
//
// It is parallel over the rows because in the graph it is the glue, and at
// L6b the glue was most of the wall clock. Every block's Upload used to run
// this loop on one core — 1.31 million conversions a sublayer at llama.cpp's
// best ubatch, 96 sublayers a graph, 126 million in all — and the conversion
// itself is the cost, not the store: a mapped HOST_CACHED arena takes 5.2 MB
// in 0.1 ms and the narrowing of the same tensor took about 6.
//
// The result is the same values in the same places; the rows of a prefill are
// independent, so this changes the wall clock and nothing else.
func narrowRows(dst []uint16, src []float32, rows, width, lda int) {
	const chunk = 32
	parallelFor((rows+chunk-1)/chunk, func(c int) {
		for t := c * chunk; t < minInt((c+1)*chunk, rows); t++ {
			row, out := src[t*width:(t+1)*width], dst[t*lda:]
			for i, v := range row {
				out[i] = safetensors.F32ToF16(v)
			}
		}
	})
}
