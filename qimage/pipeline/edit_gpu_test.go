package pipeline

import (
	"fmt"
	"image"
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/zimage/qwen"
)

// nrgbaFrom turns a dumped [1, 4, H, W] tensor in [-1, 1] back into the
// 8-bit image a server is handed. The dump's card was built from uint8 in the
// first place, so this round-trips exactly — which is the point: the gate has
// to feed the served path the *same pixels* the reference saw, and Q8.3
// measured that half a level of difference is worth rel 11 on the prompt
// embedding by the time the tower is done with it.
func nrgbaFrom(t *qwen.Mat, h, w int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	plane := h * w
	for i := 0; i < plane; i++ {
		for c := 0; c < 4; c++ {
			v := (float64(t.Data[c*plane+i]) + 1) / 2
			img.Pix[i*4+c] = uint8(math.Round(math.Min(255, math.Max(0, v*255))))
		}
	}
	return img
}

// TestServedEditOracle256 is Q8.6's composition gate: the served edit path,
// exactly as /v1/images/edits drives it, over the 256²/4-step edit oracle.
//
// It is the edit twin of TestServedOracle256 and is measured the same way,
// in both regimes. What it composes that no earlier gate did is the whole
// request: the reference image resized by our Lanczos port, composited over
// white on the 8-bit levels, through the GPU vision tower and the GPU VAE
// encoder, positioned by our mrope, injected into the GPU text encoder, laid
// out from the pad mask, denoised by the GPU DiT over a real prefix and
// decoded. Only the noise is the oracle's.
func TestServedEditOracle256(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole edit pipeline, ~30 GB")
	}
	m := loadEditManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	side := m.Size / 16
	p := newPipeline(t, dev, Options{
		Model: model, Width: m.Size, Height: m.Size, Steps: m.Steps,
		Refs: 1, CondSize: m.Size,
	})
	want := loadEditMat(t, m, "image")

	// Teacher-forced: the oracle's own final latents through our decoder,
	// which bounds everything downstream of the sampler.
	final := loadEditMat(t, m, fmt.Sprintf("step%d_latents", m.Steps-1))
	img, err := p.Decode(final.Clone(), side, side)
	if err != nil {
		t.Fatal(err)
	}
	maxAbs, mean := imageDistance(img, want)
	if maxAbs > teacherImgTol {
		t.Errorf("teacher-forced image: max abs %.5f (mean %.6f) > %.3g", maxAbs, mean, teacherImgTol)
	} else {
		t.Logf("teacher-forced image: max abs %.5f, mean %.6f (bound %.3g)", maxAbs, mean, teacherImgTol)
	}

	// Free-running: the served edit, from the oracle's noise.
	src := nrgbaFrom(loadEditMat(t, m, "cond_rgba"), m.CondInputSize[1], m.CondInputSize[0])
	noise := loadEditMat(t, m, "noise").Clone()
	var latentRel float64
	img, tm, err := p.Edit(EditRequest{
		Prompt: m.Prompt, Images: []*image.NRGBA{src},
		Width: m.Size, Height: m.Size, Steps: m.Steps, Latents: noise,
		Progress: func(st Step) {
			ref := loadEditMat(t, m, fmt.Sprintf("step%d_latents", st.Index))
			latentRel = matRel(st.Latents, ref)
			t.Logf("step %d: %7.1f ms  latents rel %.3g", st.Index,
				float64(st.Wall.Microseconds())/1000, latentRel)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("condition %v, encode %v, prefill %v, %d steps, decode %v, total %v (%d prompt rows, %d refs)",
		tm.Condition.Round(time.Millisecond), tm.Encode.Round(time.Millisecond),
		tm.Prefill.Round(time.Millisecond), len(tm.Steps), tm.Decode.Round(time.Millisecond),
		tm.Total.Round(time.Millisecond), tm.Tokens, tm.Refs)

	maxAbs, mean = imageDistance(img, want)
	servedMean := mean
	writeArtifact(t, "oracle256_edited.png", img)
	if maxAbs > servedImgTol {
		t.Errorf("edited image: max abs %.4f (mean %.5f) > %.2g, latents rel %.3g",
			maxAbs, mean, servedImgTol, latentRel)
		return
	}
	t.Logf("edited image: max abs %.4f, mean %.5f (bound %.2g); final latents rel %.3g",
		maxAbs, mean, servedImgTol, latentRel)

	// The controls. An edit that ignored its reference, or read a different
	// one, still produces a full-shaped image — so the bound has to be shown
	// to separate them.
	t.Run("controls", func(t *testing.T) {
		blank := image.NewNRGBA(image.Rect(0, 0, m.CondInputSize[0], m.CondInputSize[1]))
		for i := range blank.Pix {
			blank.Pix[i] = 255
		}
		out, _, err := p.Edit(EditRequest{
			Prompt: m.Prompt, Images: []*image.NRGBA{blank},
			Width: m.Size, Height: m.Size, Steps: m.Steps, Latents: loadEditMat(t, m, "noise").Clone(),
		})
		if err != nil {
			t.Fatal(err)
		}
		maxAbs, mean := imageDistance(out, want)
		// Measured against the run above rather than against the gate's
		// bound: an edit's trajectory is pinned by its prefix, so the *run*
		// is two orders of magnitude inside a bound that has to survive fp16
		// through 36 encoder layers, 27 tower blocks and 32 DiT blocks, and
		// a control that only had to beat the bound would prove much less.
		if mean <= 100*servedMean {
			t.Errorf("a blank reference lands at mean %.5f against the run's %.5f (%.0fx); "+
				"the reference is not reaching the image", mean, servedMean, mean/servedMean)
		} else {
			t.Logf("control %-28s max abs %.4f, mean %.5f — %.0fx the served run's",
				"a blank white reference", maxAbs, mean, mean/servedMean)
		}

		// Past the staged budget: a refusal naming the number, not a resize.
		if _, _, err := p.Edit(EditRequest{
			Prompt: m.Prompt,
			Images: []*image.NRGBA{src, src},
		}); err == nil {
			t.Error("two references were accepted by a pipeline staged for one")
		}
		if _, _, err := p.Edit(EditRequest{Prompt: m.Prompt}); err == nil {
			t.Error("an edit with no reference image was accepted")
		}
	})
}

// TestServedEditWallClock is what a served edit costs at the size the
// endpoint defaults to: one 1024² reference, a 1024² output, 40 steps, twice.
// It writes the picture out, which is this stage's eyeball.
func TestServedEditWallClock(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole edit pipeline and runs 40 steps")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	p := newPipeline(t, dev, Options{Model: model, Refs: 1})

	// A reference the eye can judge an edit against: a generated image, so
	// the test needs no fixture on disk.
	base, _, err := p.Generate("a red fox sitting in fresh snow at dawn, photograph, shallow depth of field",
		7, nil)
	if err != nil {
		t.Fatal(err)
	}
	writeArtifact(t, "edit1024_reference.png", base)
	ref := ToImage(base, true)

	for run := 0; run < 2; run++ {
		img, tm, err := p.Edit(EditRequest{
			Prompt: "make it a summer meadow at noon, keep the fox exactly as it is",
			Images: []*image.NRGBA{ref}, Seed: 11,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("run %d: %dx%d in %v — condition %v, encode %v, prefill %v, %d steps mean %v, decode %v",
			run, tm.Width, tm.Height, tm.Total.Round(time.Millisecond),
			tm.Condition.Round(time.Millisecond), tm.Encode.Round(time.Millisecond),
			tm.Prefill.Round(time.Millisecond), len(tm.Steps), meanStep(tm).Round(time.Millisecond),
			tm.Decode.Round(time.Millisecond))
		if run == 0 {
			writeArtifact(t, "edit1024_edited.png", img)
		}
	}
	weights, cache := p.EditResidency()
	t.Logf("the edit half holds %d MB of weights and a %d MB prefix KV cache", weights>>20, cache>>20)
}

func meanStep(tm *Timings) time.Duration {
	if len(tm.Steps) < 2 {
		return 0
	}
	var total time.Duration
	for _, w := range tm.Steps[1:] {
		total += w
	}
	return total / time.Duration(len(tm.Steps)-1)
}
