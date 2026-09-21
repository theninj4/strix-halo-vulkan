package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	zvae "strix-halo-vulkan/zimage/vae"
)

const (
	strixHaloDeviceID    = 0x1586
	servedArtifactsDir   = "../../out/qi21served"
	defaultSweepPrompt   = "a red fox sitting in fresh snow at dawn, photograph, shallow depth of field"
	secondSweepPrompt    = "an oil painting of a harbour at sunset, thick impasto brushwork"
	sweepPromptThirdText = "a technical diagram of a bicycle drivetrain, clean white background, labelled parts"
)

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-pipeline-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// newPipeline stages the served pipeline, or skips.
func newPipeline(t *testing.T, dev *vk.Device, opt Options) *Pipeline {
	t.Helper()
	if _, err := os.Stat(opt.Model); err != nil {
		t.Skipf("no checkpoint at %s", opt.Model)
	}
	start := time.Now()
	p, err := New(dev, opt)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(p.Destroy)
	enc, dt, vae, act := p.Residency()
	edit, cache := p.EditResidency()
	t.Logf("staged in %v: text encoder %d MB, transformer %d MB, VAE %d MB, edit %d MB "+
		"(cache %d MB), activations %d MB (%.1f GB total)",
		time.Since(start).Round(time.Second), enc>>20, dt>>20, vae>>20, edit>>20, cache>>20,
		act>>20, float64(enc+dt+vae+edit+act)/(1<<30))
	return p
}

// writeArtifact saves a PNG for the eyeball, and says where.
func writeArtifact(t *testing.T, name string, img *zvae.Tensor) {
	t.Helper()
	if err := os.MkdirAll(servedArtifactsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(servedArtifactsDir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, ToImage(img, true)); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", path)
}

// imageDistance is the oracle comparison in the reference's own [0, 1]
// space, the same instrument TestEndToEndImage uses.
func imageDistance(got *zvae.Tensor, want *qwen.Mat) (maxAbs, mean float64) {
	var sumAbs float64
	for h := 0; h < got.H; h++ {
		for w := 0; w < got.W; w++ {
			for c := 0; c < got.C; c++ {
				g := (float64(got.Plane(0, c)[h*got.W+w]) + 1) / 2
				d := math.Abs(g - float64(want.Row(h*got.W + w)[c]))
				if d > maxAbs {
					maxAbs = d
				}
				sumAbs += d
			}
		}
	}
	return maxAbs, sumAbs / float64(got.C*got.H*got.W)
}

// The served path's bounds, in the two regimes the project's rule asks for
// (two-row-regime-needs-a-graph-gate, and Q4's own pair):
//
//   - teacherImgTol bounds the *composition*. Fed the oracle's own final
//     latents, everything downstream of the sampler is our decoder, and Q5g
//     measured that at max abs 7.3e-4 on this very case. 2.2e-2 is Z-Image's
//     end-to-end precedent and is kept as the bound so a regression here
//     reads against the same number the CPU gate does.
//   - servedImgTol bounds the *trajectory*. The served run is free-running
//     fp16: Q4 measured its latents drifting to rel 0.050 over 40 steps at
//     1024² against a 0.12 bound, and diffusers documents cache-on/off
//     producing visibly different but equally valid samples for the same
//     reason. So the image cannot be held to a fp32 tolerance, and what this
//     bound is for is catching a *break* — a wrong image is off by O(1) on
//     mean, not by hundredths.
const (
	teacherImgTol = 2.2e-2
	servedImgTol  = 0.35
)

// TestServedOracle256 is Q6's composition gate: the served pipeline, exactly
// as /v1/images/generations will drive it, over the 256²/4-step oracle's
// prompt and noise.
//
// Both regimes, and the split matters. Teacher-forced — the oracle's final
// latents through our decoder — bounds everything that is not the sampler,
// tightly. Free-running is the served path end to end, where fp16 through 36
// encoder layers and 32 DiT blocks perturbs an ODE and the trajectory
// separates by construction; what is checked there is that the picture is
// still the oracle's picture.
func TestServedOracle256(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole pipeline, ~30 GB")
	}
	m := loadRunManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	side := m.Size / 16
	p := newPipeline(t, dev, Options{Model: model, Width: m.Size, Height: m.Size, Steps: m.Steps})
	want := loadRunMat(t, m, "image")

	// Teacher-forced: the oracle's final latents, our decoder.
	final := loadRunMat(t, m, fmt.Sprintf("step%d_latents", m.Steps-1))
	img, err := p.Decode(t.Context(), final.Clone(), side, side)
	if err != nil {
		t.Fatal(err)
	}
	maxAbs, mean := imageDistance(img, want)
	if maxAbs > teacherImgTol {
		t.Errorf("teacher-forced image: max abs %.5f (mean %.6f) > %.3g", maxAbs, mean, teacherImgTol)
	} else {
		t.Logf("teacher-forced image: max abs %.5f, mean %.6f (bound %.3g)", maxAbs, mean, teacherImgTol)
	}

	// Free-running: the served path, from the oracle's noise so that only
	// the models differ.
	noise := loadRunMat(t, m, "noise").Clone()
	var latentRel float64
	img, tm, err := p.Run(t.Context(), Request{
		Prompt: m.Prompt, Width: m.Size, Height: m.Size, Steps: m.Steps, Latents: noise,
		Progress: func(st Step) {
			ref := loadRunMat(t, m, fmt.Sprintf("step%d_latents", st.Index))
			latentRel = matRel(st.Latents, ref)
			t.Logf("step %d: %7.1f ms  latents rel %.3g", st.Index,
				float64(st.Wall.Microseconds())/1000, latentRel)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("encode %v, prefill %v, %d steps, decode %v, total %v (%d prompt tokens)",
		tm.Encode.Round(time.Millisecond), tm.Prefill.Round(time.Millisecond),
		len(tm.Steps), tm.Decode.Round(time.Millisecond), tm.Total.Round(time.Millisecond), tm.Tokens)

	maxAbs, mean = imageDistance(img, want)
	writeArtifact(t, "oracle256_served.png", img)
	if maxAbs > servedImgTol {
		t.Errorf("served image: max abs %.4f (mean %.5f) > %.2g, latents rel %.3g",
			maxAbs, mean, servedImgTol, latentRel)
		return
	}
	t.Logf("served image: max abs %.4f, mean %.5f (bound %.2g); final latents rel %.3g",
		maxAbs, mean, servedImgTol, latentRel)
}

// matRel is the relative distance two latent matrices are apart, with an rms
// floor — pipeline_test.go's instrument, without the assertion.
func matRel(got, want *qwen.Mat) float64 {
	var sumSq float64
	for _, v := range want.Data {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	return rel
}

// TestServedWallClock is the number IMAGE.md's Q6 asked for: what a served
// 1024²/40-step image actually costs, end to end, twice. It writes the
// picture out, which is also the free-run eyeball Q4 deferred to this stage.
func TestServedWallClock(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole pipeline and runs 40 steps")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	p := newPipeline(t, dev, Options{Model: model})

	for run := 0; run < 2; run++ {
		img, tm, err := p.Generate(t.Context(), defaultSweepPrompt, 42, nil)
		if err != nil {
			t.Fatal(err)
		}
		var steps time.Duration
		for _, d := range tm.Steps[1:] {
			steps += d
		}
		t.Logf("run %d: %v total — encode %v, prefill %v, %d cached steps mean %v, decode %v",
			run, tm.Total.Round(time.Millisecond), tm.Encode.Round(time.Millisecond),
			tm.Prefill.Round(time.Millisecond), len(tm.Steps)-1,
			(steps / time.Duration(len(tm.Steps)-1)).Round(time.Millisecond),
			tm.Decode.Round(time.Millisecond))
		if run == 1 {
			writeArtifact(t, "served1024_40steps.png", img)
		}
	}
}

// TestStepSweep is the measurement that decides the served default: the same
// seed and the same prompts at 40, 24, 16 and 12 steps, written out to be
// looked at. It reports the cost of each and the distance from the 40-step
// image, which is a proxy and not the judgement — the judgement is the
// eyeball, and the PNGs are what it is made on.
func TestStepSweep(t *testing.T) {
	if testing.Short() || os.Getenv("QI21_SWEEP") == "" {
		t.Skip("set QI21_SWEEP=1; this is ~10 minutes of device time")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	p := newPipeline(t, dev, Options{Model: model})

	prompts := map[string]string{
		"fox":     defaultSweepPrompt,
		"harbour": secondSweepPrompt,
		"diagram": sweepPromptThirdText,
	}
	for name, prompt := range prompts {
		var ref *zvae.Tensor
		for _, steps := range []int{40, 24, 16, 12} {
			img, tm, err := p.Run(t.Context(), Request{Prompt: prompt, Steps: steps, Seed: 42})
			if err != nil {
				t.Fatal(err)
			}
			line := fmt.Sprintf("%-8s %2d steps: %v", name, steps, tm.Total.Round(time.Millisecond))
			if ref == nil {
				ref = img
			} else {
				var maxAbs, sum float64
				for i := range img.Data {
					d := math.Abs(float64(img.Data[i]-ref.Data[i])) / 2
					if d > maxAbs {
						maxAbs = d
					}
					sum += d
				}
				line += fmt.Sprintf("  vs 40: max abs %.3f, mean %.4f", maxAbs, sum/float64(len(img.Data)))
			}
			t.Log(line)
			writeArtifact(t, fmt.Sprintf("sweep_%s_%02dsteps.png", name, steps), img)
		}
	}
}

// TestGeometryShapes is the graph gate the area ceiling needs, and the reason
// it exists is the project's own rule about a new regime: the arithmetic in
// TestArenaShape and TestAspectRatioFitsTheCeiling says a request of the
// staged *area* runs in the staged arenas whatever its shape, and that is a
// claim about recorded graphs — the DiT's row count, the VAE's re-planned
// decode, the RoPE grids — which only the device can settle.
//
// So: stage one square pipeline, then render the shapes the served geometry
// now hands out. The square is the control, 1344x768 is what `16:9` resolves
// to at this ceiling, and 2048x512 has a side twice the staged one at exactly
// the staged token count. Eight steps, because what is under test is the
// geometry and not the picture — the PNGs are written anyway, since a graph
// that ran with a transposed grid would produce a *plausible* tensor and only
// the eyeball catches that.
func TestGeometryShapes(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole pipeline")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	p := newPipeline(t, dev, Options{Model: model, Width: 1024, Height: 1024, Steps: 8})
	maxTokens := p.MaxPixels() / (VAEScale * VAEScale)

	for _, c := range []struct{ w, h int }{{1024, 1024}, {1344, 768}, {768, 1344}, {2048, 512}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			img, tm, err := p.Run(t.Context(), Request{
				Prompt: defaultSweepPrompt, Width: c.w, Height: c.h, Steps: 8, Seed: 42,
			})
			if err != nil {
				t.Fatal(err)
			}
			if img.W != c.w || img.H != c.h {
				t.Fatalf("asked for %dx%d, decoded %dx%d", c.w, c.h, img.W, img.H)
			}
			tokens := (c.w / VAEScale) * (c.h / VAEScale)
			t.Logf("%dx%d: %d tokens of the staged %d, %v total (prefill %v, step %v, decode %v)",
				c.w, c.h, tokens, maxTokens, tm.Total.Round(time.Millisecond),
				tm.Prefill.Round(time.Millisecond), tm.Steps[1].Round(time.Millisecond),
				tm.Decode.Round(time.Millisecond))
			writeArtifact(t, fmt.Sprintf("shape_%dx%d_8steps.png", c.w, c.h), img)
		})
	}

	// And the refusal is the area, not the side: one step past the staged
	// token count is a 400 even though both sides are inside the square.
	if _, _, err := p.Run(t.Context(), Request{Prompt: "p", Width: 1024, Height: 1088, Steps: 1, Seed: 1}); err == nil {
		t.Error("1024x1088 ran; it is 6% more pixels than the arenas hold")
	} else {
		t.Logf("1024x1088 refused: %v", err)
	}
}

// TestCancellation is the gate on what a client that hangs up costs, and it
// has to be a device test because what is under test is *when* the run stops
// rather than that it eventually errors.
//
// Three things are asserted, and the third is the one that would hurt:
//
//   - the sampler stops at the step boundary, so a cancel two steps into
//     eight costs about two steps and not eight;
//   - the decoder stops between submit batches, so a cancel during the VAE
//     costs a fraction of the decode rather than all of it;
//   - **the pipeline still works afterwards**. A cancelled run leaves a
//     recorded graph half-submitted and the arenas holding an abandoned
//     image, so the next request is the real question, and it is answered by
//     running one and comparing it against the same image generated without
//     any cancellation in front of it. Bit-identical, not merely plausible.
//
// The error identity matters as much as the timing: `errors.Is(err,
// context.Canceled)` is what api.backendError keys on to answer with nothing
// instead of a 500, so it is asserted rather than assumed.
func TestCancellation(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole pipeline")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	const (
		side  = 512
		steps = 8
	)
	p := newPipeline(t, dev, Options{Model: model, Width: side, Height: side, Steps: steps})

	// The uncancelled run first: it is both the reference image and the wall
	// clock everything below is read against.
	full := time.Now()
	want, tm, err := p.Run(t.Context(), Request{
		Prompt: defaultSweepPrompt, Width: side, Height: side, Steps: steps, Seed: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	baseline := time.Since(full)
	t.Logf("uncancelled: %v total (%d steps, decode %v)",
		baseline.Round(time.Millisecond), len(tm.Steps), tm.Decode.Round(time.Millisecond))

	// ---- Cancelled two steps in.
	t.Run("sampler", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var reached int
		start := time.Now()
		_, _, err := p.Run(ctx, Request{
			Prompt: defaultSweepPrompt, Width: side, Height: side, Steps: steps, Seed: 42,
			Progress: func(st Step) {
				reached = st.Index
				if st.Index == 1 {
					cancel()
				}
			},
		})
		elapsed := time.Since(start)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err %v, want context.Canceled", err)
		}
		// Two steps ran and the third was refused, so the run cost about a
		// quarter of the schedule. Half the baseline is a loose bound that a
		// run-to-completion cannot pass.
		if elapsed > baseline/2 {
			t.Errorf("cancelled at step 1 of %d but took %v of a %v run",
				steps, elapsed.Round(time.Millisecond), baseline.Round(time.Millisecond))
		}
		t.Logf("cancelled after step %d: %v (%.0f%% of the full run): %v",
			reached, elapsed.Round(time.Millisecond), 100*float64(elapsed)/float64(baseline), err)
	})

	// ---- Cancelled inside the VAE, which is one call and 30 submits.
	t.Run("decoder", func(t *testing.T) {
		latents := p.noise(geom{latentH: side / VAEScale, latentW: side / VAEScale,
			imgTokens: (side / VAEScale) * (side / VAEScale)}, 7)
		clean := time.Now()
		if _, err := p.Decode(t.Context(), latents.Clone(), side/VAEScale, side/VAEScale); err != nil {
			t.Fatal(err)
		}
		decode := time.Since(clean)

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		time.AfterFunc(decode/8, cancel)
		start := time.Now()
		_, err := p.Decode(ctx, latents.Clone(), side/VAEScale, side/VAEScale)
		elapsed := time.Since(start)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err %v, want context.Canceled", err)
		}
		if elapsed > decode/2 {
			t.Errorf("cancelled an eighth into a %v decode but took %v",
				decode.Round(time.Millisecond), elapsed.Round(time.Millisecond))
		}
		t.Logf("decode %v, cancelled an eighth in after %v: %v",
			decode.Round(time.Millisecond), elapsed.Round(time.Millisecond), err)
	})

	// ---- And the device is not left in a state that poisons the next
	// request, which is the property the two above would be worthless
	// without.
	after, _, err := p.Run(t.Context(), Request{
		Prompt: defaultSweepPrompt, Width: side, Height: side, Steps: steps, Seed: 42,
	})
	if err != nil {
		t.Fatalf("the run after two cancellations failed: %v", err)
	}
	for i := range want.Data {
		if after.Data[i] != want.Data[i] {
			t.Fatalf("the run after two cancellations differs at element %d: %g against %g",
				i, after.Data[i], want.Data[i])
		}
	}
	t.Log("the run after two cancellations is bit-identical to the one before them")
}

// TestGeometryCeiling is the ceiling as a served behaviour rather than an
// arena number: a size past it is refused with a reason, and the refusal
// happens before anything is staged.
func TestGeometryCeiling(t *testing.T) {
	if _, err := os.Stat(model); err != nil {
		t.Skipf("no checkpoint at %s", model)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	_, err := New(dev, Options{Model: model, Width: 2048, Height: 2048})
	if err == nil {
		t.Fatal("a 2048x2048 ceiling was accepted; the VAE arena cannot hold it")
	}
	t.Logf("2048x2048 ceiling refused: %v", err)
}

// TestPreviewFrames is Q7's gate, and it is an eyeball with a number beside
// it rather than a tolerance: a 64x4 linear map cannot reproduce a
// convolutional decoder, so what has to be true is that an in-progress frame
// is *recognisably the picture being made*. The number is the distance
// between the last preview and the finished image box-filtered to the same
// 1/16 grid — the preview's own job, measured against the only thing that
// could grade it.
//
// It also pins the mechanism the frames depend on: the tensor decoded is the
// denoised estimate x0 and not the sample x_t. At step 1 of 8 the sample is
// still 80% noise, so a run of this test with that substitution produces
// noise, and the numbers below separate by an order of magnitude.
func TestPreviewFrames(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole pipeline")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	const side = 512
	p := newPipeline(t, dev, Options{Model: model, Width: side, Height: side, Steps: 8})

	var frames, samples []*zvae.Tensor
	var at []int
	img, tm, err := p.Run(t.Context(), Request{
		Prompt: defaultSweepPrompt, Width: side, Height: side, Steps: 8, Seed: 42,
		Progress: func(st Step) {
			if st.Preview == nil {
				t.Error("no preview closure on the step")
				return
			}
			start := time.Now()
			f, err := st.Preview()
			if err != nil {
				t.Fatal(err)
			}
			if st.Index == 0 {
				t.Logf("a preview decode costs %v against a %v step", time.Since(start).Round(time.Microsecond), st.Wall)
			}
			frames = append(frames, f)
			at = append(at, st.Index)
			// The negative control: the same matrix over the *sample* rather
			// than the denoised estimate. It is the substitution the field
			// name invites and the one that would silently ship noise.
			sc, err := PreviewDecode(st.Latents, side/16, side/16)
			if err != nil {
				t.Fatal(err)
			}
			samples = append(samples, sc)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 8 {
		t.Fatalf("%d frames for 8 steps", len(frames))
	}
	if frames[0].H != side/16 || frames[0].W != side/16 {
		t.Fatalf("preview is %dx%d, want the %d latent grid", frames[0].W, frames[0].H, side/16)
	}

	// The finished image on the preview's own grid, which is what a frame is
	// trying to be.
	want := boxDown(img, 16)
	for i, f := range frames {
		var sum float64
		for j := range f.Data {
			sum += math.Abs(float64(f.Data[j] - want.Data[j]))
		}
		t.Logf("step %d: mean abs %.4f from the finished image, in [-1,1]", at[i], sum/float64(len(f.Data)))
		writeArtifact(t, fmt.Sprintf("preview_step%d.png", at[i]), f)
	}
	writeArtifact(t, "preview_final_full.png", img)
	writeArtifact(t, "preview_final_grid.png", want)
	t.Logf("%dx%d/8 steps in %v", tm.Width, tm.Height, tm.Total.Round(time.Millisecond))

	// The last frame is the finished latent (the terminal sigma makes x0 the
	// sample exactly), so it is the fit's own error and nothing else. The
	// first is a preview of a picture that barely exists. Both are logged;
	// only the ordering is asserted, because that ordering is the claim a
	// preview makes.
	first, last := meanAbs(frames[0], want), meanAbs(frames[len(frames)-1], want)
	if last >= first {
		t.Errorf("previews do not converge: step 0 is %.4f from the image and the last is %.4f", first, last)
	}

	// The control, at the step where it matters most: early, where the
	// sample is still mostly noise and the estimate is already a picture.
	// They coincide at the end by construction (the terminal sigma is zero),
	// which is itself worth asserting -- it is what says the two really are
	// the same quantity at the end and only the middle is a choice.
	ctrl := meanAbs(samples[0], want)
	writeArtifact(t, "preview_control_step0_sample.png", samples[0])
	if ctrl < first*2 {
		t.Errorf("decoding the sample at step 0 is %.4f from the image against the estimate's %.4f; "+
			"the control should be far worse", ctrl, first)
	}
	t.Logf("control: step 0 from the sample x_t is %.4f, from the estimate x0 %.4f (%.1fx)", ctrl, first, ctrl/first)
	if tail := meanAbs(samples[len(samples)-1], want); tail != last {
		t.Errorf("at the last step the sample and the estimate differ (%.6f vs %.6f); "+
			"the terminal sigma should make them identical", tail, last)
	}
}

func meanAbs(a, b *zvae.Tensor) float64 {
	var sum float64
	for i := range a.Data {
		sum += math.Abs(float64(a.Data[i] - b.Data[i]))
	}
	return sum / float64(len(a.Data))
}

// boxDown averages n x n blocks, which is how cmd/previewfit paired a latent
// pixel with the colour it produced.
func boxDown(t *zvae.Tensor, n int) *zvae.Tensor {
	out := zvae.NewTensor(1, t.C, t.H/n, t.W/n)
	inPlane, outPlane := t.H*t.W, out.H*out.W
	for c := 0; c < t.C; c++ {
		for y := 0; y < out.H; y++ {
			for x := 0; x < out.W; x++ {
				var acc float64
				for dy := 0; dy < n; dy++ {
					for dx := 0; dx < n; dx++ {
						acc += float64(t.Data[c*inPlane+(y*n+dy)*t.W+x*n+dx])
					}
				}
				out.Data[c*outPlane+y*out.W+x] = float32(acc / float64(n*n))
			}
		}
	}
	return out
}
