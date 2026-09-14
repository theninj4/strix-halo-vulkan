package vae

import (
	"math"
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

// convTol is what fp16 conv operands cost the decoded image, measured rather
// than assumed: TestGPUConvMatchesScalar reports the same number against the
// kernel that narrows nothing.
//
// It is larger than the mid block's (wmmaTol, 2.5e-3) for a reason that has
// nothing to do with either kernel's arithmetic: the mid block is one place
// in the graph, and the convolutions are *every* place. A 1024x1024 decode
// runs 40-odd of them in series, each rounding its input to fp16 before
// reading it, so what this bound holds is a chain and not a step.
const convTol = 1.5e-2

// TestConvInputsFitFP16 is the precondition the whole matrix-core conv path
// rests on, asserted on real activations rather than on a random fixture
// (stage 6: "a real prompt is not a random tensor"). Stage 2 measured this
// decoder's absmax at 497 and its *scores* at 1.16e7; if a checkpoint ever
// arrives whose activations reach fp16's 65504, the pack silently produces
// infinities and every bound in this file goes with it.
func TestConvInputsFitFP16(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	type peak struct {
		shape string
		max   float32
	}
	var peaks []peak
	convTap = func(c *Conv2D, x *Tensor) {
		var mx float32
		for _, v := range x.Data {
			if a := float32(math.Abs(float64(v))); a > mx {
				mx = a
			}
		}
		peaks = append(peaks, peak{x.String(), mx})
	}
	defer func() { convTap = nil }()
	if _, err := cpu.Apply(latent); err != nil {
		t.Fatal(err)
	}

	// The margin is against fp16's largest finite value. A factor of 8 is
	// arbitrary and deliberate: it is far enough below 65504 that a model
	// whose activations grew would fail here rather than in a decoded image,
	// and far enough above 497 that this decoder passes with 16x to spare.
	const limit = 65504.0 / 8
	var worst peak
	for _, p := range peaks {
		if p.max > worst.max {
			worst = p
		}
		if float64(p.max) > limit {
			t.Errorf("conv input %s peaks at %g, past the %g this path narrows to fp16 at", p.shape, p.max, limit)
		}
	}
	t.Logf("%d conv inputs, worst %g at %s (%.0fx inside fp16's 65504)",
		len(peaks), worst.max, worst.shape, 65504/float64(worst.max))
}

// TestConvWeightsFitFP16 is the same question for the other operand. The
// filters are narrowed once at load, so an overflow there would be silent and
// permanent rather than image-dependent.
func TestConvWeightsFitFP16(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	var worst float32
	var count int
	check := func(w []float32) {
		for _, v := range w {
			count++
			if a := float32(math.Abs(float64(v))); a > worst {
				worst = a
			}
			if safetensors.F32ToF16(v)&0x7c00 == 0x7c00 {
				t.Fatalf("filter weight %g does not survive fp16", v)
			}
		}
	}
	for _, cr := range convRefs(cpu) {
		check(cr.conv.Weight)
	}
	t.Logf("%d filter weights, worst |w| = %g", count, worst)
}

// convRefs walks the decoder's convolutions the way flattenWeights does, for
// tests that want them without a device.
func convRefs(d *Decoder) []convRef {
	var out []convRef
	add := func(name string, c *Conv2D) {
		if c != nil {
			out = append(out, convRef{name, c})
		}
	}
	add("conv_in", d.ConvIn)
	for _, m := range []*ResnetBlock{d.Mid.Resnet1, d.Mid.Resnet2} {
		add("r.conv1", m.Conv1)
		add("r.conv2", m.Conv2)
		add("r.shortcut", m.Shortcut)
	}
	for _, up := range d.UpBlocks {
		for _, r := range up.Resnets {
			add("r.conv1", r.Conv1)
			add("r.conv2", r.Conv2)
			add("r.shortcut", r.Shortcut)
		}
		add("upconv", up.Upsampler)
	}
	add("conv_out", d.ConvOut)
	return out
}

// newConvDecoder builds a decoder whose convolutions run on the matrix cores
// and whose mid block does not, so that what a comparison measures is this
// stage and not stage 7's. It skips on a device without matrix cores.
func newConvDecoder(t *testing.T, dev *vk.Device, cpu *Decoder, h, w int, k ConvKernel) *GPUDecoder {
	t.Helper()
	g, err := NewGPUDecoderOpts(dev, cpu, h, w, Options{Attn: AttnScalar, Conv: k})
	if err != nil {
		t.Fatal(err)
	}
	if !g.convCores() {
		g.Destroy()
		t.Skip("device has no matrix cores")
	}
	return g
}

// TestGPUConvMatchesScalar is where convTol comes from: the same graph with
// and without the narrowing, on the same device, so the only difference
// between the two answers is fp16 and the implicit GEMM.
//
// Every build in the ladder is checked, not only the default. They differ in
// how the output tile is cut, which is exactly what decides whether the
// masked epilogue runs -- at a 16x16 reference latent most of this decode's
// rows are narrower than BN -- so a build that got the mask wrong would pass
// at one tiling and fail at another.
func TestGPUConvMatchesScalar(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	scalar, err := NewGPUDecoderOpts(dev, cpu, latent.H, latent.W,
		Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Destroy()
	want, err := scalar.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}

	for _, k := range ConvKernels() {
		if k == ConvScalar {
			continue
		}
		t.Run(string(k), func(t *testing.T) {
			g := newConvDecoder(t, dev, cpu, latent.H, latent.W, k)
			defer g.Destroy()
			got, err := g.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "conv "+string(k), got, want, convTol)
		})
	}
}

// TestGPUConvAgainstDiffusers holds the matrix-core decoder to the reference
// the fp32 CPU implementation was built against, with the mid block on the
// old kernels and on the new ones, because the two narrowings compose in a
// way worth having on the record: the convolutions alone land 8.2e-3 from
// the fp32 graph, and the whole fp16 decoder lands 3.3e-3 from diffusers.
// That is not an inconsistency and it is not cancellation -- the bound is a
// *max over elements* rather than an L2, so it reports whichever single
// pixel is worst, and a different mid block makes it a different pixel.
func TestGPUConvAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")
	want := loadRef(t, m, "image")
	for _, o := range []Options{{Attn: AttnScalar}, {}} {
		name := "conv only"
		if o.Attn != AttnScalar {
			name = "conv + mid block"
		}
		g, err := NewGPUDecoderOpts(dev, cpu, latent.H, latent.W, o)
		if err != nil {
			t.Fatal(err)
		}
		if !g.convCores() {
			g.Destroy()
			t.Skip("device has no matrix cores")
		}
		got, err := g.Apply(latent)
		if err != nil {
			g.Destroy()
			t.Fatal(err)
		}
		compareTol(t, name, got, want, convTol)
		g.Destroy()
	}
}

// TestGPUConvOddSize decodes at a latent whose width is not a multiple of
// anything the kernel tiles by: 12 makes every pixel tile a partial one and
// every image row 12 wide against a BN of 64 or 128. It is the masked
// epilogue's own test, and it is also the one place the pixel-tile-per-row
// rule is load-bearing -- at W=12 a tile that ran on past the row's end would
// wrap a tap into the next row and be off by one pixel everywhere.
func TestGPUConvOddSize(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	const n = 12
	latent := NewTensor(1, cpu.ConvIn.InC, n, n)
	for i := range latent.Data {
		latent.Data[i] = float32(math.Sin(float64(i)*0.37)) * 2
	}

	scalar, err := NewGPUDecoderOpts(dev, cpu, n, n, Options{Attn: AttnScalar, Conv: ConvScalar})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Destroy()
	want, err := scalar.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}

	for _, k := range ConvKernels() {
		if k == ConvScalar {
			continue
		}
		t.Run(string(k), func(t *testing.T) {
			g := newConvDecoder(t, dev, cpu, n, n, k)
			defer g.Destroy()
			got, err := g.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "odd "+string(k), got, want, convTol)
		})
	}
}

// TestGPUConvControls breaks the two things this path does that stage 2b's
// convolution did not, and asserts what each break does to the image.
//
//   - ConvNoTapShift drops the horizontal tap offset. The blocked layout
//     exists so that a 16-pixel fragment window can start at *any* pixel;
//     a kernel that had rounded the window down to a tile boundary would
//     compute this, and it has to be loud.
//   - ConvPadClamp replicates the edge pixel instead of zeroing the border.
//     It is the padding bug a conv port actually has, and it touches only
//     the frame -- 0.4% of a 1024x1024 image and 6% of the 128x128 one this
//     test decodes -- so whether an RMS-normalised image bound catches it is
//     a real question rather than a rhetorical one.
//
// Both are asserted in the direction they measure. If one of them stops
// firing, the right response is to find the test that does catch it, not to
// tighten this bound until it does.
func TestGPUConvControls(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")
	want := loadRef(t, m, "image")

	for _, c := range []struct {
		kernel ConvKernel
		caught bool
	}{
		{ConvNoTapShift, true},
		{ConvPadClamp, true},
	} {
		t.Run(string(c.kernel), func(t *testing.T) {
			g := newConvDecoder(t, dev, cpu, latent.H, latent.W, c.kernel)
			defer g.Destroy()
			got, err := g.Apply(latent)
			if err != nil {
				t.Fatal(err)
			}
			_, _, rel, _ := deviation(got, want)
			switch {
			case c.caught && rel <= convTol:
				t.Errorf("%s: rel %.3g is inside the tolerance %.1e -- the control is not controlling anything", c.kernel, rel, convTol)
			case !c.caught && rel > convTol:
				t.Errorf("%s: rel %.3g is outside the tolerance %.1e -- this control now works and should be asserted as one", c.kernel, rel, convTol)
			default:
				t.Logf("%-14s rel %.3g, %.3gx the bound", c.kernel, rel, rel/convTol)
			}
		})
	}
}
