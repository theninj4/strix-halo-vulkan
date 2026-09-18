package pipeline

import (
	"fmt"
	"math"
	"testing"

	"strix-halo-vulkan/zimage/vae"
)

// IMAGE.md I7's composition tests: an SDEdit is the encoder, a noising and the
// tail of the schedule, and each of those three has its own oracle already.
// What is here is the claim nothing below can make -- that they are composed
// in the right order with the right sigma.
//
// The reference is reference/dump_zimage_edit.py, which runs the same edit
// through diffusers in fp32 on the CPU:
//
//	.venv/bin/python reference/dump_zimage_edit.py --size 256 --strength 0.8
//
// Affordable for the same reason dump_zimage_run.py is: the composition does
// not know how big the image is.

// editTol bounds the whole edit, and it is the generation's own bound.
//
// That is the claim worth making: an edit's error is *not* the encoder's error
// plus the loop's. The encoder's latent is 2.2e-2 in relative L2 (zimage/vae's
// TestGPUEncoderMatrixCores), but the noising multiplies it by (1 - sigma) --
// 0.1 at strength 0.8 -- before the loop ever sees it, so what the trajectory
// starts from is closer to diffusers' than the encoder alone is.
const editTol = 3e-2

func loadEditManifest(t *testing.T) *manifest {
	t.Helper()
	m := loadManifest(t, editDir)
	if m.Strength == 0 {
		t.Skipf("%s is not a dump_zimage_edit.py manifest", editDir)
	}
	return m
}

// initImage reads the dumped input picture as the tensor Request.Init takes.
func initImage(t *testing.T, m *manifest, size int) *vae.Tensor {
	t.Helper()
	data := loadRef(t, m, "image_in")
	img := vae.NewTensor(1, 3, size, size)
	if len(data) != len(img.Data) {
		t.Fatalf("image_in has %d values, want %d", len(data), len(img.Data))
	}
	copy(img.Data, data)
	return img
}

// TestEditAgainstDiffusers walks the tail of the schedule beside diffusers',
// and checks the two things upstream of it separately: the encoded latent and
// the noised one the loop actually starts from.
//
// Doing it in three pieces is the point. An edit that came out wrong could be
// a wrong encode, a wrong sigma, a wrong starting index or a wrong loop, and
// only the first of those has an oracle anywhere else in the repository.
func TestEditAgainstDiffusers(t *testing.T) {
	m := loadEditManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps, Encoder: true})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	img := initImage(t, m, m.Size)

	// One: the encode. This is the encoder's own bound rather than the
	// edit's, because it is the encoder's own output.
	got, err := p.EncodeImage(img)
	if err != nil {
		t.Fatal(err)
	}
	if e := relL2(got, loadRef(t, m, "latent_in")); e > editTol {
		t.Errorf("encoded latent: relative L2 %.3g > %.0e", e, editTol)
	} else {
		t.Logf("encoded latent: relative L2 %.3g", e)
	}

	// Two: where the schedule starts. These are integers and a float, and
	// getting any of them wrong is an edit that is too strong or too weak
	// rather than one that fails.
	if s := StartStep(m.Steps, m.Strength); s != m.Start {
		t.Fatalf("strength %g of %d steps starts at %d, diffusers starts at %d", m.Strength, m.Steps, s, m.Start)
	}
	sched, err := NewFlowMatchEuler(m.Steps, p.schedCfg)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(sched.Sigmas[m.Start] - m.Sigma); d > 1e-6 {
		t.Fatalf("sigma at step %d is %g, diffusers has %g", m.Start, sched.Sigmas[m.Start], m.Sigma)
	}

	// Three: the run, from the same eps the reference drew.
	eps := loadRef(t, m, "eps")
	latents := append([]float32(nil), eps...)
	var steps []float64
	var first int
	out, tm, err := p.Run(Request{
		Prompt: m.Prompt, Init: img, Strength: m.Strength, Latents: latents,
		Progress: func(s Step) {
			if len(steps) == 0 {
				first = s.Index
			}
			steps = append(steps, relL2(s.Latents, loadRef(t, m, fmt.Sprintf("latents_%d", s.Index))))
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first != m.Start || len(steps) != m.Steps-m.Start {
		t.Errorf("ran steps %d..%d, diffusers ran %d..%d", first, first+len(steps)-1, m.Start, m.Steps-1)
	}
	if tm.First != m.Start || len(tm.Steps) != m.Steps-m.Start {
		t.Errorf("Timings says first=%d over %d steps, want %d over %d", tm.First, len(tm.Steps), m.Start, m.Steps-m.Start)
	}
	// The step bound is the generation's, for the same reason: a trajectory
	// amplifies, and what has to hold at the end is the picture.
	const stepTol = 6e-2
	for i, e := range steps {
		if e > stepTol {
			t.Errorf("step %d: relative L2 %.3g > %.0e", m.Start+i, e, stepTol)
		}
	}
	t.Logf("strength %g: steps %d..%d from sigma %.4f, per-step relative L2 %s",
		m.Strength, m.Start, m.Steps-1, tm.Sigma, fmtErrs(steps))

	if e := relL2(latents, loadRef(t, m, "latents_final")); e > stepTol {
		t.Errorf("final latent: relative L2 %.3g > %.0e", e, stepTol)
	}
	e := relL2(out.Data, loadRef(t, m, "image"))
	if e > editTol {
		t.Errorf("image: relative L2 %.3g > %.0e", e, editTol)
	}
	t.Logf("edited image %dx%d: relative L2 %.3g against diffusers, encode %s of %s",
		out.W, out.H, e, tm.VAEEncode.Round(1e6), tm.Total.Round(1e6))
}

// TestStrengthOneIsAGeneration is the identity that pins the noising formula's
// endpoint and the loop's starting index at once.
//
// At strength 1 the schedule's first sigma is exactly 1, so
// (1 - sigma) x0 + sigma eps is eps: the input picture contributes nothing and
// the run is the ordinary generation from that noise. Nothing about the code
// makes that true by construction -- an off-by-one in StartStep, a sigma read
// from the wrong end of the schedule, or a noising written the other way round
// all still produce an image -- so it is asserted.
//
// It is also the only test in here that needs no reference dump.
func TestStrengthOneIsAGeneration(t *testing.T) {
	m := loadEditManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps, Encoder: true})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	img := initImage(t, m, m.Size)
	eps := loadRef(t, m, "eps")

	plain := append([]float32(nil), eps...)
	wantImg, wantTm, err := p.GenerateFrom(m.Prompt, plain, nil)
	if err != nil {
		t.Fatal(err)
	}
	edited := append([]float32(nil), eps...)
	gotImg, gotTm, err := p.Run(Request{Prompt: m.Prompt, Init: img, Strength: 1, Latents: edited})
	if err != nil {
		t.Fatal(err)
	}
	if gotTm.First != 0 || len(gotTm.Steps) != len(wantTm.Steps) {
		t.Fatalf("strength 1 ran steps %d..%d of %d; a generation runs all of them",
			gotTm.First, gotTm.First+len(gotTm.Steps)-1, len(wantTm.Steps))
	}
	// Bit-identical is the claim, not merely close: the two runs execute the
	// same dispatches on the same numbers. Anything above zero here means the
	// edit path perturbed the latent it was meant to leave alone.
	if e := relL2(edited, plain); e != 0 {
		t.Errorf("latents differ by %.3g; at strength 1 the input image contributes exactly nothing", e)
	}
	if e := relL2(gotImg.Data, wantImg.Data); e != 0 {
		t.Errorf("images differ by %.3g at strength 1", e)
	}
	t.Logf("strength 1 reproduced the generation exactly")
}

// TestEditStaysNearTheInput is the question IMAGE.md I7 said was unmeasured:
// whether an eight-step turbo distillation edits acceptably through SDEdit at
// all. It is a measurement rather than a threshold on most of its output --
// what it asserts is only the *ordering*, which is the part that would be a
// bug rather than a property of the checkpoint.
//
// The ordering is the whole contract of a strength knob: a weaker edit has to
// come back closer to the picture it was given. If that is not monotone the
// knob does not mean what it says, whatever the pictures look like.
func TestEditStaysNearTheInput(t *testing.T) {
	m := loadEditManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps, Encoder: true})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	img := initImage(t, m, m.Size)

	// The floor: a round trip through the encoder and the decoder with no
	// denoising at all. No edit can be closer to the input than this, so it is
	// what the strengths below are read against.
	z, err := p.EncodeImage(img)
	if err != nil {
		t.Fatal(err)
	}
	round, err := p.DecodeLatent(z, m.Size/8, m.Size/8)
	if err != nil {
		t.Fatal(err)
	}
	floor := relL2(round.Data, img.Data)
	t.Logf("%-10s relative L2 to the input %.4g   (the encode/decode round trip, no denoising)", "floor", floor)

	prev := floor
	for _, s := range []float64{0.2, 0.4, 0.6, 0.8, 1.0} {
		out, tm, err := p.Run(Request{Prompt: m.Prompt, Init: img, Strength: s, Seed: 11})
		if err != nil {
			t.Fatalf("strength %g: %v", s, err)
		}
		e := relL2(out.Data, img.Data)
		t.Logf("strength %-4g relative L2 to the input %.4g   steps %d..%d from sigma %.3f",
			s, e, tm.First, tm.First+len(tm.Steps)-1, tm.Sigma)
		if e < prev {
			t.Errorf("strength %g came back *closer* to the input than the strength below it (%.4g < %.4g); "+
				"the knob does not mean what it says", s, e, prev)
		}
		prev = e
	}
}

// TestEditRefusals covers the three ways an edit can be asked for wrongly.
// Each of them would otherwise be a picture: an unresized image would be
// encoded at the wrong geometry, a strength without an image would be silently
// ignored, and an edit on a pipeline with no encoder would have to invent one.
func TestEditRefusals(t *testing.T) {
	m := loadEditManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	img := initImage(t, m, m.Size)

	t.Run("no encoder", func(t *testing.T) {
		p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: 1})
		if err != nil {
			t.Skipf("cannot build the pipeline (%v)", err)
		}
		defer p.Destroy()
		if p.HasEncoder() {
			t.Fatal("a pipeline built without Options.Encoder reports one")
		}
		if _, _, err := p.Run(Request{Prompt: "p", Init: img}); err == nil {
			t.Error("an edit was accepted on a pipeline with no encoder")
		}
	})

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: 1, Encoder: true})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	t.Run("wrong size", func(t *testing.T) {
		small := vae.NewTensor(1, 3, m.Size/2, m.Size/2)
		if _, _, err := p.Run(Request{Prompt: "p", Init: small}); err == nil {
			t.Error("an init image of the wrong size was accepted; it would be encoded at the wrong geometry")
		}
	})
	t.Run("strength without an image", func(t *testing.T) {
		if _, _, err := p.Run(Request{Prompt: "p", Strength: 0.5}); err == nil {
			t.Error("a strength with no init image was accepted; nothing in the run would use it")
		}
	})
	t.Run("strength out of range", func(t *testing.T) {
		if _, _, err := p.Run(Request{Prompt: "p", Init: img, Strength: 1.5}); err == nil {
			t.Error("strength 1.5 was accepted")
		}
	})
}

// TestStartStep pins the arithmetic on its own, including the floor that is
// not diffusers': a strength that rounds down to no steps at all would hand
// back a round trip of the input with no denoising, which is not an edit.
func TestStartStep(t *testing.T) {
	for _, c := range []struct {
		steps, want int
		strength    float64
	}{
		{8, 0, 1.0}, {8, 2, 0.8}, {8, 4, 0.5}, {8, 6, 0.3}, {8, 7, 0.1}, {8, 7, 0.01},
		{4, 0, 1.0}, {4, 2, 0.5}, {1, 0, 0.5},
	} {
		if got := StartStep(c.steps, c.strength); got != c.want {
			t.Errorf("StartStep(%d, %g) = %d, want %d", c.steps, c.strength, got, c.want)
		}
	}
}
