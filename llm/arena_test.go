package llm

// What an activation arena costs the host, by memory type (LLM.md L6b).
//
// This is the measurement `arena.go` exists because of, and it is a test
// rather than a note because the claim it supports is load-bearing: the whole
// graph got 5.0x faster from one line choosing a different `memoryTypeIndex`,
// and a future change that quietly puts the arenas back on the write-combined
// type would cost that again with nothing else to show for it.
//
//	go test ./llm/ -v -run TestArenaMemoryTypes

import (
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// arenaProbeElems is one `hc_mixed` at llama.cpp's best ubatch: the tensor
// that actually crosses a block boundary in the graph, 512 tokens of 2560.
const arenaProbeElems = 512 * 2560

// TestArenaMemoryTypes prices a host read and a host write of a mapped
// storage buffer out of every memory type that can back one.
//
// L0b measured the same table from the *device* and found 236.0-237.4 GB/s
// across all of it, 0.57% over 32 cells — so which type a buffer lands in is
// free for a kernel. It is not free for the host: the type `NewBuffer`
// prefers is write-combined, and an uncached mapping turns a read into one
// uncached load per line.
func TestArenaMemoryTypes(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	types, err := dev.MemoryTypes()
	if err != nil {
		t.Fatal(err)
	}
	const n = arenaProbeElems
	src := make([]float32, n)
	var sink float32

	rate := func(b *vk.Buffer) (read, write float64) {
		const iters = 20
		b.WriteFloat32(make([]float32, n))
		st := time.Now()
		for i := 0; i < iters; i++ {
			sink += b.ReadFloat32At(0, n)[0]
		}
		rd := time.Since(st) / iters
		st = time.Now()
		for i := 0; i < iters; i++ {
			b.WriteFloat32At(0, src)
		}
		wr := time.Since(st) / iters
		return float64(n*4) / rd.Seconds() / 1e9, float64(n*4) / wr.Seconds() / 1e9
	}

	t.Logf("%-5s %-5s %-62s %10s %10s", "type", "heap", "flags", "read GB/s", "write GB/s")
	best, bestCached := 0.0, 0.0
	for _, mt := range types {
		if !mt.BufferCompatible || !mt.Has(vk.MemoryHostVisible) {
			continue
		}
		b, err := dev.NewBufferOfType(n*4, mt.Index)
		if err != nil {
			t.Logf("%-5d %-5d %-62s  %v", mt.Index, mt.HeapIndex, mt, err)
			continue
		}
		rd, wr := rate(b)
		b.Destroy()
		t.Logf("%-5d %-5d %-62s %10.2f %10.2f", mt.Index, mt.HeapIndex, mt, rd, wr)
		best = maxFloat(best, rd)
		if mt.Has(MemoryHostCachedFlag) {
			bestCached = maxFloat(bestCached, rd)
		}
	}
	if bestCached == 0 {
		t.Skip("this device has no HOST_CACHED type that can back a storage buffer")
	}
	if bestCached < best {
		t.Errorf("the fastest read is %.2f GB/s and the fastest HOST_CACHED one %.2f: newArena picks the wrong type",
			best, bestCached)
	}

	// And the arena the blocks actually allocate, which is the thing the
	// graph reads. A one-line regression in newArena shows up here.
	b, err := newArena(dev, n*4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Destroy()
	rd, wr := rate(b)
	t.Logf("newArena: read %.2f GB/s, write %.2f GB/s (sink %v)", rd, wr, sink)
	// A tenth of the best cached type is still two orders past the
	// write-combined 0.18 GB/s, so this catches the wrong type and not a
	// noisy run.
	if rd < bestCached/10 {
		t.Errorf("newArena reads at %.2f GB/s, the best HOST_CACHED type does %.2f", rd, bestCached)
	}
}

// MemoryHostCachedFlag is vk.MemoryHostCached, named here so the test reads
// without a package qualifier in the middle of a condition.
const MemoryHostCachedFlag = vk.MemoryHostCached

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
