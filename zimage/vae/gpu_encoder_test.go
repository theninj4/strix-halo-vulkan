package vae

import (
	"math"
	"os"
	"testing"

	"strix-halo-vulkan/vk"
)

// The encoder on the matrix cores needs a bound of its own, and getting there
// took a measurement that stage 8 never had to make.
//
// **Stage 8's precondition was range, and range was never the question here.**
// TestEncoderConvInputsFitFP16 below measures this graph's largest convolution
// input at 284, which is 230x inside fp16 -- more headroom than the decoder's
// 497. What the encoder has and the decoder does not is *conditioning*: it
// contracts a 3-channel image into a 32-channel latent at an eighth of the
// resolution, and a perturbation at conv_in comes out of conv_norm_out
// amplified by about 4e5. That is a property of the arithmetic and not of this
// port -- asking diffusers for the same encode in float64 and diffing its own
// float32 shows the same amplification, 4.8e-6 at down.0 to 6.3e-4 at
// conv_norm_out -- so an fp16-sized perturbation at the front comes out large
// at the back whatever computes it.
//
// So the matrix-core path's stagewise error is not small, and the honest
// question is what it does to the *answer*. Both numbers, on the 256x256
// reference image:
//
//	latent, max element over RMS   0.58
//	latent, relative L2            2.2e-2
//	decoded back to a picture      3.0e-3 relative L2, mean 4.2e-4 of [-1, 1]
//
// A twentieth of one 8-bit level. The two metrics disagree by 27x on the same
// tensor because the first is set by a handful of outliers in a latent whose
// RMS is 2.5, which is PIPELINE.md's third rule exactly: a latent is the
// chaotic-trajectory case and is bounded in relative L2 over the whole tensor.
// 3e-2 is the bound the pipeline's own composition uses and this sits inside
// it.
//
// **The fp32 path is kept and is the oracle** (PIPELINE.md's second rule), it
// meets encOutTol, and it is selectable -- 1.764 s against 431 ms at
// 1024x1024, `go run ./cmd/vaebench -encode`, which is what makes the fast one
// the default rather than the only one.
const (
	// gpuEncL2Tol bounds the latent in relative L2.
	gpuEncL2Tol = 3e-2
	// gpuEncStageTol bounds the *intermediates*, which the amplification
	// above puts an order of magnitude looser than the answer. It is here so
	// that the stagewise walk still fails on a wrong dispatch -- the
	// scalar-path walk beside it is what holds the graph tight.
	gpuEncStageTol = 10.0
)

func loadEncoder(t *testing.T) *Encoder {
	t.Helper()
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	enc, err := LoadEncoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

func newTestEncoder(t *testing.T, dev *vk.Device, cpu *Encoder, h, w int, opt Options) *GPUEncoder {
	t.Helper()
	g, err := NewGPUEncoderOpts(dev, cpu, h, w, opt)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// relL2Tensor is the whole-tensor relative L2 of an error, which is the right
// denominator for a latent; deviation's max-over-RMS is the right one for an
// activation. Which of the two a tensor wants is a property of the tensor.
func relL2Tensor(got, want *Tensor) float64 {
	var num, den float64
	for i := range want.Data {
		d := float64(got.Data[i]) - float64(want.Data[i])
		num += d * d
		den += float64(want.Data[i]) * float64(want.Data[i])
	}
	return math.Sqrt(num / math.Max(den, 1e-30))
}

// TestGPUEncoderScalarPath holds the fp32 graph -- no matrix cores anywhere --
// to the same bound the CPU port meets, stage by stage.
//
// **This is the test that says the graph is right**, and it comes first for
// that reason: it shares no arithmetic with the CPU port but every dispatch of
// it with the fast path, so a wrong downsample offset, a mis-ordered resnet or
// an arena that aliases fails here at 5e-4 rather than hiding under the fp16
// path's looser bound.
func TestGPUEncoderScalarPath(t *testing.T) {
	cpu := loadEncoder(t)
	m := loadEncManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	img := loadRef(t, m, "image_in")
	g := newTestEncoder(t, dev, cpu, img.H, img.W, Options{Attn: AttnScalar, Conv: ConvScalar})
	defer g.Destroy()

	stages, err := g.Stages(img.H, img.W)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range stages {
		got, err := g.RunTo(img, name)
		if err != nil {
			t.Fatal(err)
		}
		// Same split as the CPU walk: the ill-conditioned middle gets
		// encRelTol and the tensors the pipeline consumes get encOutTol.
		tol := encRelTol
		if name == "conv_out" {
			tol = encOutTol
		}
		compareTol(t, "fp32 "+name, got, loadRef(t, m, name), tol)
	}

	mode, err := g.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "fp32 mode", mode, loadRef(t, m, "mode"), encOutTol)
}

// TestGPUEncoderMatrixCores is the fast path, held to the bound the finding
// above says is the right one -- relative L2 on the latent -- with the
// stagewise walk beside it so the shape of the amplification is on the record
// rather than in a comment.
func TestGPUEncoderMatrixCores(t *testing.T) {
	cpu := loadEncoder(t)
	m := loadEncManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	img := loadRef(t, m, "image_in")
	g := newTestEncoder(t, dev, cpu, img.H, img.W, Options{})
	if !g.convCores() {
		g.Destroy()
		t.Skip("device has no matrix cores")
	}
	defer g.Destroy()
	n, err := g.Dispatches(img.H, img.W)
	if err != nil {
		t.Fatal(err)
	}
	attn, gemm, conv := g.Kernels()
	t.Logf("kernels attn=%s gemm=%s conv=%s, %d dispatches, %d MB fp32 + %d MB fp16 arena",
		attn, gemm, conv, n, g.ActivationBytes()>>20, g.F16ActivationBytes()>>20)

	stages, err := g.Stages(img.H, img.W)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range stages {
		got, err := g.RunTo(img, name)
		if err != nil {
			t.Fatal(err)
		}
		want := loadRef(t, m, name)
		_, _, rel, _ := deviation(got, want)
		t.Logf("%-22s %-20s max/rms %8.4g   relL2 %.4g", name, got.String(), rel, relL2Tensor(got, want))
		if rel > gpuEncStageTol {
			t.Errorf("%s: max/rms %.4g is past %.0f, which is more than the amplification accounts for",
				name, rel, gpuEncStageTol)
		}
	}

	// The answer, in the metric a latent takes.
	mode, err := g.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	wantMode := loadRef(t, m, "mode")
	if l2 := relL2Tensor(mode, wantMode); l2 > gpuEncL2Tol {
		t.Errorf("mode: relative L2 %.4g > %.0e", l2, gpuEncL2Tol)
	} else {
		t.Logf("mode relative L2 %.4g (bound %.0e)", l2, gpuEncL2Tol)
	}

	// And the picture, which is the number the bound above is justified by.
	// Round-tripping through the fp32 decoder puts the encoder's error where
	// a caller would actually see it.
	dec, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	a, err := dec.Apply(mode)
	if err != nil {
		t.Fatal(err)
	}
	b, err := dec.Apply(wantMode)
	if err != nil {
		t.Fatal(err)
	}
	l2, mean := relL2Tensor(a, b), meanAbsDiff(a, b)
	t.Logf("round-tripped image: relative L2 %.4g, mean |diff| %.4g of a [-1, 1] range (%.3f of one 8-bit level)",
		l2, mean, mean*127.5)
	if mean*127.5 > 0.5 {
		t.Errorf("the narrowing moves the decoded picture by %.3f of an 8-bit level; it is meant to be invisible", mean*127.5)
	}
}

// TestEncoderConvInputsFitFP16 is stage 8's precondition, re-measured on this
// graph. It passes with more headroom than the decoder's -- and the note at
// the top of this file is why that was not the end of the question.
func TestEncoderConvInputsFitFP16(t *testing.T) {
	cpu := loadEncoder(t)
	m := loadEncManifest(t)
	var peak float64
	var where string
	old := convTap
	defer func() { convTap = old }()
	convTap = func(c *Conv2D, x *Tensor) {
		if v := absMax(x); v > peak {
			peak, where = v, x.String()
		}
	}
	if _, err := cpu.Apply(loadRef(t, m, "image_in")); err != nil {
		t.Fatal(err)
	}
	const limit = 65504.0 / 8
	t.Logf("worst convolution input %g at %s (%.0fx inside fp16's 65504)", peak, where, 65504/peak)
	if peak > limit {
		t.Errorf("conv input peaks at %g, past the %g this path narrows to fp16 at", peak, limit)
	}
}

// TestGPUEncoderOneEncoderManySizes is the residency claim as a test: the
// arenas are built once for the largest image and every smaller one runs in
// them. A server sizes its encoder for its ceiling and a request may be any
// size under it, so this is the property /v1/images/edits rests on.
//
// The negative controls are in here: an image larger than the arenas, and a
// side that three halvings do not divide, are both refused rather than allowed
// to overrun or to silently truncate. The graph is re-recorded per encode, so
// nothing else would catch either until the latent came back wrong.
func TestGPUEncoderOneEncoderManySizes(t *testing.T) {
	cpu := loadEncoder(t)
	dev, done := newTestDevice(t)
	defer done()

	const maxH, maxW = 256, 256
	g := newTestEncoder(t, dev, cpu, maxH, maxW, Options{})
	defer g.Destroy()

	for _, sz := range [][2]int{{256, 256}, {128, 256}, {256, 128}, {64, 64}} {
		h, w := sz[0], sz[1]
		img := NewTensor(1, 3, h, w)
		for i := range img.Data {
			img.Data[i] = float32((i%211)-105) / 105
		}
		got, err := g.Encode(img)
		if err != nil {
			t.Fatalf("%dx%d: %v", h, w, err)
		}
		if got.C != cpu.LatentChannels || got.H != h/8 || got.W != w/8 {
			t.Errorf("%dx%d encoded to %s, want [1 %d %d %d]", h, w, got, cpu.LatentChannels, h/8, w/8)
		}
	}

	big := NewTensor(1, 3, maxH*2, maxW)
	if _, err := g.Encode(big); err == nil {
		t.Error("an image past the arenas was accepted; it would overrun them silently")
	} else {
		t.Logf("refused, as it must be: %v", err)
	}

	odd := NewTensor(1, 3, 60, 64)
	if _, err := g.Encode(odd); err == nil {
		t.Error("a 60-pixel side was accepted; three halvings do not divide it")
	}
}
