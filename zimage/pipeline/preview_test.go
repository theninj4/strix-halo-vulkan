package pipeline

import (
	"fmt"
	"testing"
)

// The preview references are reference/dump_taef1.py, which decodes the *same*
// eight latents dump_zimage_run.py produced:
//
//	.venv/bin/python reference/dump_taef1.py --from-run reference/out/zimagerun
//
// That shared origin is what makes this a composition test rather than a
// second decoder test. zimage/vae already holds taef1 to the dump on a fixed
// latent; what is only checkable here is that the pipeline forms the *right*
// latent at each step and hands it over.
const (
	previewDir   = "../../reference/out/taef1run"
	taef1Dir     = "../../models/taef1"
	previewRegen = "reference/dump_taef1.py --from-run reference/out/zimagerun"
)

// TestPipelinePreviews runs the eight steps and checks every preview against
// the reference, both as a latent and as an image.
//
// The latent comparison is the one that can only be made here: Step.X0 is the
// denoised estimate, which the loop forms from the velocity it has in hand,
// and getting it wrong -- by a sign, by an off-by-one in the sigma, or by
// handing over the raw state instead -- produces a preview that is a
// perfectly good picture of the wrong thing. The reference derives the same
// estimate from the dumped trajectory, independently.
func TestPipelinePreviews(t *testing.T) {
	m := loadManifest(t, runDir)
	if m.Size == 0 {
		t.Skipf("%s is not a dump_zimage_run.py manifest", runDir)
	}
	pm := loadManifest(t, previewDir)
	if _, ok := pm.Tensors["x0_0"]; !ok {
		t.Skipf("%s has no previews; run %s", previewDir, previewRegen)
	}
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Preview: taef1Dir,
		Width: m.Size, Height: m.Size, Steps: m.Steps})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()
	if !p.HasPreview() {
		t.Fatal("Options.Preview named a checkpoint and HasPreview() is false")
	}

	latents := append([]float32(nil), loadRef(t, m, "latents_init")...)
	// The latent bound is the trajectory's own. The preview bound is *not*
	// the finished image's 3e-2, and the difference is worth stating: a
	// preview is a decode of an estimate formed from one velocity prediction,
	// so it carries sigma times the model's single-step error on top of the
	// latent error the finished image carries alone -- and sigma is 0.95 at
	// the first step. Measured here the previews run 0.003 to 0.030 with the
	// worst at step 4, against 0.022 for the image the same run produces. 5e-2
	// is above that and still below where the raw-state control in the next
	// test lands.
	const stepTol, previewTol = 6e-2, 5e-2
	var x0Errs, imgErrs []float64
	_, _, err = p.GenerateFrom(m.Prompt, latents, func(s Step) {
		if s.Preview == nil {
			t.Errorf("step %d: Step.Preview is nil on a pipeline with a preview decoder", s.Index)
			return
		}
		x0Errs = append(x0Errs, relL2(s.X0, loadRef(t, pm, fmt.Sprintf("x0_%d", s.Index))))
		img, err := s.Preview()
		if err != nil {
			t.Fatalf("step %d: %v", s.Index, err)
		}
		if img.C != 3 || img.H != m.Size || img.W != m.Size {
			t.Fatalf("step %d: preview is %s, want a %dx%d image", s.Index, img, m.Size, m.Size)
		}
		imgErrs = append(imgErrs, relL2(img.Data, loadRef(t, pm, fmt.Sprintf("preview_%d", s.Index))))
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(x0Errs) != m.Steps {
		t.Fatalf("%d previews over %d steps", len(x0Errs), m.Steps)
	}
	for i := range x0Errs {
		if x0Errs[i] > stepTol {
			t.Errorf("step %d: x0 relative L2 %.3g > %.0e", i, x0Errs[i], stepTol)
		}
		if imgErrs[i] > previewTol {
			t.Errorf("step %d: preview relative L2 %.3g > %.0e", i, imgErrs[i], previewTol)
		}
	}
	t.Logf("x0 relative L2      %s", fmtErrs(x0Errs))
	t.Logf("preview relative L2 %s", fmtErrs(imgErrs))
}

// TestPreviewIsNotTheRawLatent is the negative control for the one decision
// that has no other check.
//
// Every tensor comparison above would still pass if the pipeline decoded
// Step.Latents instead of Step.X0 *and* the reference had been generated the
// same way, because both are plausible pictures and neither is empty. What
// says x0 is the right choice is that its previews converge on the finished
// image and the raw state's do not: measured over the eight steps, x0 runs
// 0.29 -> 0.02 away from the final image and x_t runs 0.48 -> 0.02.
//
// So this decodes both sequences through the same decoder and asserts the
// ordering, which is a claim about the schedule rather than about taef1.
func TestPreviewIsNotTheRawLatent(t *testing.T) {
	m := loadManifest(t, runDir)
	if m.Size == 0 {
		t.Skipf("%s is not a dump_zimage_run.py manifest", runDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Preview: taef1Dir,
		Width: m.Size, Height: m.Size, Steps: m.Steps})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	_, latentH, latentW, err := p.LatentFor(m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	latents := append([]float32(nil), loadRef(t, m, "latents_init")...)
	var x0Imgs, xtImgs [][]float32
	final, _, err := p.GenerateFrom(m.Prompt, latents, func(s Step) {
		a, err := s.Preview()
		if err != nil {
			t.Fatal(err)
		}
		b, err := p.PreviewDecode(s.Latents, latentH, latentW)
		if err != nil {
			t.Fatal(err)
		}
		x0Imgs = append(x0Imgs, append([]float32(nil), a.Data...))
		xtImgs = append(xtImgs, append([]float32(nil), b.Data...))
	})
	if err != nil {
		t.Fatal(err)
	}

	// Against the pipeline's own finished image rather than the reference's,
	// so this control says nothing about accuracy and everything about which
	// tensor a preview is of.
	var x0Err, xtErr []float64
	for i := range x0Imgs {
		x0Err = append(x0Err, meanAbs(x0Imgs[i], final.Data))
		xtErr = append(xtErr, meanAbs(xtImgs[i], final.Data))
	}
	t.Logf("mean|preview - image|, x0  %s", fmtErrs(x0Err))
	t.Logf("mean|preview - image|, x_t %s", fmtErrs(xtErr))

	// The first step is where the two differ most, and it is the step a
	// streaming client sees first.
	if x0Err[0] >= xtErr[0] {
		t.Errorf("after one step the denoised estimate is %.3g from the image and the raw state %.3g; "+
			"they are the wrong way round", x0Err[0], xtErr[0])
	}
	if r := xtErr[0] / x0Err[0]; r < 1.3 {
		t.Errorf("the raw state is only %.2fx further from the image than the estimate after one step; "+
			"the measured separation is 1.6x, so something has flattened the schedule", r)
	}
	for i := 1; i < len(x0Err); i++ {
		if x0Err[i] > x0Err[i-1] {
			t.Errorf("preview %d is further from the image than preview %d (%.3g > %.3g); "+
				"the sequence does not converge", i, i-1, x0Err[i], x0Err[i-1])
		}
	}
}

// TestPreviewRefusedWithoutACheckpoint pins the other half of the flag: a
// pipeline built without one has no Step.Preview and says so, rather than
// returning a blank image or decoding through the full VAE at ten times the
// price.
func TestPreviewRefusedWithoutACheckpoint(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	p, err := New(dev, Options{Model: modelDir, Width: 256, Height: 256, Steps: 1})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	if p.HasPreview() {
		t.Error("HasPreview() is true on a pipeline built with no taef1 checkpoint")
	}
	if _, err := p.PreviewDecode(make([]float32, 16*32*32), 32, 32); err == nil {
		t.Error("PreviewDecode succeeded with no preview decoder")
	}
	seen := 0
	if _, _, err := p.Run(Request{Prompt: "a fox", Seed: 1, Progress: func(s Step) {
		seen++
		if s.Preview != nil {
			t.Error("Step.Preview is set on a pipeline with no preview decoder")
		}
		if s.X0 == nil {
			t.Error("Step.X0 is nil; the estimate costs one pass and does not need a decoder")
		}
	}}); err != nil {
		t.Fatal(err)
	}
	if seen != 1 {
		t.Errorf("%d progress callbacks over one step", seen)
	}
}

func meanAbs(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum / float64(len(a))
}
