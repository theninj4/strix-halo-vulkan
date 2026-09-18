package vae

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// The references are produced by reference/dump_taef1.py. Regenerate with:
//
//	.venv/bin/python reference/dump_taef1.py --latent-size 16
//	.venv/bin/python reference/dump_taef1.py --from-run reference/out/zimagerun
const (
	tinyRefDir = "../../reference/out/taef1"
	tinyRunDir = "../../reference/out/taef1run"
	tinyDir    = "../../models/taef1"
	tinyGen    = "reference/dump_taef1.py"
)

func loadTiny(t *testing.T) *TinyDecoder {
	t.Helper()
	if _, err := os.Stat(tinyDir); err != nil {
		t.Skipf("no taef1 checkpoint at %s", tinyDir)
	}
	d, err := LoadTinyDecoder(tinyDir, TAEF1Config())
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// TestTinyDecoderAgainstDiffusers walks taef1 layer by layer against the
// dump. Same argument as TestDecoderAgainstDiffusers: nineteen layers of
// convolution compose into an image, and an end-to-end check can only say
// that the image is wrong.
func TestTinyDecoderAgainstDiffusers(t *testing.T) {
	d := loadTiny(t)
	m := loadManifestFrom(t, tinyRefDir, tinyGen+" --latent-size 16")

	latent := loadRef(t, m, "latent")
	// The clamp, which diffusers puts in DecoderTiny.forward rather than in
	// the Sequential, and which is therefore the one piece of this decoder
	// that no weight name would reveal.
	h := &Tensor{N: latent.N, C: latent.C, H: latent.H, W: latent.W,
		Data: make([]float32, latent.Len())}
	for i, v := range latent.Data {
		h.Data[i] = tanh32(v/3) * 3
	}
	compare(t, "clamped", h, loadRef(t, m, "clamped"))

	var err error
	for _, l := range d.Layers {
		switch {
		case l.Conv != nil:
			if h, err = l.Conv.Apply(h); err != nil {
				t.Fatal(err)
			}
		case l.Block != nil:
			if h, err = l.Block.Apply(h); err != nil {
				t.Fatal(err)
			}
		case l.Upsample:
			h = UpsampleNearest2x(h)
		case l.ReLU:
			ReLUInPlace(h)
		}
		compare(t, fmt.Sprintf("layers.%d", l.Index), h, loadRef(t, m, fmt.Sprintf("layers.%d", l.Index)))
	}
}

// TestTinyDecoderEndToEnd runs Apply the way a preview will, so the clamp and
// the [0,1] -> [-1,1] tail are covered as well as the layers.
func TestTinyDecoderEndToEnd(t *testing.T) {
	d := loadTiny(t)
	m := loadManifestFrom(t, tinyRefDir, tinyGen+" --latent-size 16")
	got, err := d.Apply(loadRef(t, m, "latent"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "image", got, loadRef(t, m, "image"))
	if s := d.Scale(); s != 8 {
		t.Errorf("Scale() = %d, want 8", s)
	}
}

// TestTinyDecoderShape pins the graph the loader builds against the
// checkpoint, rather than against the code that built it: nineteen layers in
// diffusers' order, with the four bare convolutions and the three upsamples
// where the config says they are. A checkpoint whose layer numbering moved
// would load and produce a plausible, wrong image without this.
func TestTinyDecoderShape(t *testing.T) {
	d := loadTiny(t)
	if len(d.Layers) != 19 {
		t.Fatalf("%d layers, want 19", len(d.Layers))
	}
	kind := func(l TinyLayer) string {
		switch {
		case l.Conv != nil:
			return "conv"
		case l.Block != nil:
			return "block"
		case l.Upsample:
			return "up"
		default:
			return "relu"
		}
	}
	want := []string{
		"conv", "relu",
		"block", "block", "block", "up", "conv",
		"block", "block", "block", "up", "conv",
		"block", "block", "block", "up", "conv",
		"block", "conv",
	}
	for i, l := range d.Layers {
		if l.Index != i {
			t.Errorf("layer %d is indexed %d", i, l.Index)
		}
		if got := kind(l); got != want[i] {
			t.Errorf("layers.%d is a %s, want %s", i, got, want[i])
		}
	}
	// 1.23 M parameters. The number is worth pinning because it is the whole
	// argument for the preview decoder existing: the full decoder is 49.55 M.
	if n := d.TinyParams(); n != 1229443 {
		t.Errorf("TinyParams() = %d, want 1229443", n)
	}
}

// TestTinyTakesTheRawLatent is the convention, as an assertion.
//
// reference/dump_taef1.py --from-run measured it on a real 256x256 z-image
// trajectory: the raw diffusion latent lands 3.3x closer to the full VAE's
// image than the (z/scale + shift) the full decoder is handed. This decodes
// the same latent both ways and checks the ordering still holds, because the
// wrong one is *not* visibly wrong -- the tanh clamp squashes it into a
// recognisable picture -- so nothing else in the pipeline would catch it.
func TestTinyTakesTheRawLatent(t *testing.T) {
	d := loadTiny(t)
	m := loadManifestFrom(t, tinyRunDir, tinyGen+" --from-run reference/out/zimagerun")
	z := loadRef(t, m, "latent_final")
	truth := loadRef(t, m, "vae_final")

	const scale, shift = 0.3611, 0.1159
	unscaled := &Tensor{N: z.N, C: z.C, H: z.H, W: z.W, Data: make([]float32, z.Len())}
	for i, v := range z.Data {
		unscaled.Data[i] = float32(float64(v)/scale + shift)
	}

	raw, err := d.Apply(z)
	if err != nil {
		t.Fatal(err)
	}
	wrong, err := d.Apply(unscaled)
	if err != nil {
		t.Fatal(err)
	}
	rawErr, wrongErr := meanAbsDiff(raw, truth), meanAbsDiff(wrong, truth)
	t.Logf("mean|taef1 - vae|: raw %.4f, unscaled %.4f (%.1fx)", rawErr, wrongErr, wrongErr/rawErr)
	if wrongErr < 2*rawErr {
		t.Errorf("the raw latent is meant to be the convention, and it is only %.2fx better "+
			"than the unscaled one (%.4f vs %.4f); one of the two dumps has moved", wrongErr/rawErr, rawErr, wrongErr)
	}
	// And the raw one has to actually be close, not merely closer. 2.0% of
	// the full VAE's range is what the dump measured; the bound is loose
	// because taef1 is a distillation and not a reproduction.
	if rel := rawErr / absMax(truth); rel > 0.03 {
		t.Errorf("the raw latent decodes %.1f%% of the VAE's range away from it, want under 3%%", 100*rel)
	}
}

// TestTinyPreviewsApproachTheImage decodes every step's denoised estimate from
// a real run and checks the sequence against the reference dump, tensor for
// tensor, and then checks the *shape* of the sequence: a preview gets closer
// to the finished image at every step.
//
// The second half is the part that no per-tensor comparison gives. A decoder
// that matched the dump on each x0 latent and a pipeline that handed it x_t
// instead would both pass everything else in this file; what says the preview
// is a preview is that it converges.
func TestTinyPreviewsApproachTheImage(t *testing.T) {
	d := loadTiny(t)
	m := loadManifestFrom(t, tinyRunDir, tinyGen+" --from-run reference/out/zimagerun")
	truth := loadRef(t, m, "vae_final")

	var errs []float64
	for k := 0; ; k++ {
		name := fmt.Sprintf("x0_%d", k)
		if _, ok := m.Tensors[name]; !ok {
			break
		}
		got, err := d.Apply(loadRef(t, m, name))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, fmt.Sprintf("preview_%d", k), got, loadRef(t, m, fmt.Sprintf("preview_%d", k)))
		errs = append(errs, meanAbsDiff(got, truth))
	}
	if len(errs) < 4 {
		t.Fatalf("only %d steps in the dump", len(errs))
	}
	t.Logf("mean|preview - vae| by step: %.4f", errs)
	for k := 1; k < len(errs); k++ {
		if errs[k] > errs[k-1] {
			t.Errorf("step %d's preview is further from the image than step %d's (%.4f > %.4f); "+
				"a preview sequence that does not converge is decoding x_t rather than x0",
				k, k-1, errs[k], errs[k-1])
		}
	}
	// The first preview is already a picture -- that is the measured claim
	// behind previewing x0 rather than x_t, and it is what makes a preview
	// worth sending after step 0 of 8.
	if errs[0] > 0.35 {
		t.Errorf("the first preview is %.4f from the finished image; the dump measured 0.29 for x0 "+
			"and 0.48 for x_t, so this is the x_t trajectory", errs[0])
	}
}

// TestTinyActivationsFitFP16 is TestConvInputsFitFP16 for the preview decoder:
// the device path narrows every convolution's operands to fp16, and that is
// only safe if nothing in the graph gets near 65504.
//
// It runs at the reference size because what it measures is the *range* the
// activations reach, and that is set by the weights and the clamp rather than
// by the image size. The clamp is in fact the reason there is so much room:
// the input can never exceed 3.
func TestTinyActivationsFitFP16(t *testing.T) {
	d := loadTiny(t)
	m := loadManifestFrom(t, tinyRefDir, tinyGen+" --latent-size 16")

	peak, where := 0.0, ""
	old := convTap
	defer func() { convTap = old }()
	convTap = func(c *Conv2D, x *Tensor) {
		if v := absMax(x); v > peak {
			peak, where = v, fmt.Sprintf("conv %d->%d at %dx%d", c.InC, c.OutC, x.H, x.W)
		}
	}
	if _, err := d.Apply(loadRef(t, m, "latent")); err != nil {
		t.Fatal(err)
	}
	t.Logf("largest convolution input: %.4g (%s), fp16 max is 65504", peak, where)
	// Two orders of magnitude of headroom is the claim, not merely "under
	// the maximum": fp16's relative precision at 8 is 1/1024, and an
	// activation at 6.6e4 would be accumulating in a format with none left.
	if peak > 655 {
		t.Errorf("peak convolution input %.4g is within 100x of fp16's maximum (%s)", peak, where)
	}
}

func meanAbsDiff(a, b *Tensor) float64 {
	var sum float64
	for i := range a.Data {
		sum += math.Abs(float64(a.Data[i]) - float64(b.Data[i]))
	}
	return sum / float64(len(a.Data))
}

func absMax(t *Tensor) float64 {
	m := 0.0
	for _, v := range t.Data {
		if a := math.Abs(float64(v)); a > m {
			m = a
		}
	}
	return m
}
