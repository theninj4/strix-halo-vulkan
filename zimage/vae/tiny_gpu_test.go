package vae

import (
	"fmt"
	"testing"
)

// TestGPUTinyMatchesCPU is the preview decoder's correctness gate: the Vulkan
// graph against the same latent through the CPU port, which
// TestTinyDecoderAgainstDiffusers has already held to the diffusers dump.
//
// Comparing against the CPU rather than against the dump is what makes a
// failure readable. A graph built out of the wrong dispatches is wrong by a
// lot and shows here; the small difference that is left is float32 summation
// order, and the CPU port is the only thing that shares taef1's exact
// arithmetic with it.
func TestGPUTinyMatchesCPU(t *testing.T) {
	cpu := loadTiny(t)
	m := loadManifestFrom(t, tinyRefDir, tinyGen+" --latent-size 16")
	dev, done := newTestDevice(t)
	defer done()

	latent := loadRef(t, m, "latent")
	g, err := NewGPUTiny(dev, cpu, latent.H, latent.W)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("conv kernel %s, %d dispatches", g.ConvKernel(), mustDispatches(t, g, latent.H, latent.W))

	got, err := g.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	// Against the reference as well as against the CPU. The fp16 convolution
	// narrows its operands, so this is not the 2e-4 the fp32 graph holds to;
	// the bound is the one gpu_conv_test.go uses for the same kernel.
	compareTol(t, "image (gpu vs diffusers)", got, loadRef(t, m, "image"), 3e-2)

	want, err := cpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "image (gpu vs cpu)", got, want, 3e-2)
}

// TestGPUTinyOneDecoderManySizes is the residency claim, as a test: the arenas
// are built once for the largest latent and every smaller one runs in them.
// That is what a preview needs, because a server's preview decoder is sized
// for its ceiling and a request may be any size under it.
//
// The negative control is in here too: a latent larger than the arenas is
// refused rather than allowed to overrun. The graph is re-recorded per decode,
// so nothing else would catch it until the picture came back wrong.
func TestGPUTinyOneDecoderManySizes(t *testing.T) {
	cpu := loadTiny(t)
	dev, done := newTestDevice(t)
	defer done()

	const maxH, maxW = 32, 32
	g, err := NewGPUTiny(dev, cpu, maxH, maxW)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	for _, sz := range [][2]int{{32, 32}, {16, 32}, {16, 16}, {32, 16}} {
		h, w := sz[0], sz[1]
		latent := randomLatent(h, w, int64(h*1000+w))
		got, err := g.Apply(latent)
		if err != nil {
			t.Fatalf("%dx%d: %v", h, w, err)
		}
		if got.C != 3 || got.H != h*8 || got.W != w*8 {
			t.Errorf("%dx%d latent decoded to %s, want [1 3 %d %d]", h, w, got, h*8, w*8)
		}
		want, err := cpu.Apply(latent)
		if err != nil {
			t.Fatal(err)
		}
		compareTol(t, fmt.Sprintf("%dx%d", h, w), got, want, 3e-2)
	}

	if _, err := g.Apply(randomLatent(maxH*2, maxW, 1)); err == nil {
		t.Error("a latent twice the arenas' height was accepted")
	}
}

func randomLatent(h, w int, seed int64) *Tensor {
	t := NewTensor(1, 16, h, w)
	// A deterministic, latent-shaped input: values in roughly [-3, 3], which
	// is where a diffusion latent lives and where taef1's clamp starts to
	// bite.
	x := uint64(seed)*6364136223846793005 + 1442695040888963407
	for i := range t.Data {
		x = x*6364136223846793005 + 1442695040888963407
		t.Data[i] = (float32(x>>40)/float32(1<<24) - 0.5) * 6
	}
	return t
}

func mustDispatches(t *testing.T, g *GPUTiny, h, w int) int {
	t.Helper()
	n, err := g.Dispatches(h, w)
	if err != nil {
		t.Fatal(err)
	}
	return n
}
