package pipeline

import (
	"fmt"
	"math"
	"testing"

	"strix-halo-vulkan/vk"
)

// Stage 6's end-to-end check, and the only one that can exist for what this
// stage adds. The head, the blocks, the scheduler and the VAE each have their
// own oracle; "the four of them run in the right order with the right
// conventions" is a claim about the *composition*, and nothing below it can
// test that.
//
// The reference is reference/dump_zimage_run.py, which runs diffusers'
// ZImagePipeline in fp32 on the CPU from the same initial latent:
//
//	.venv/bin/python reference/dump_zimage_run.py --size 256
//
// It is affordable because the composition does not know how big the image is.
// At 256x256 the unified sequence is 288 tokens rather than 4128, so diffusers
// walks all eight steps on the CPU in half a minute and this walks them in two
// seconds.

const modelDir = "../../models/Z-Image-Turbo"

// strixHaloDeviceID picks the iGPU when the machine also enumerates something
// else; devs[0] otherwise.
const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("pipeline-test")
	if err != nil {
		t.Skipf("no Vulkan instance (%v)", err)
	}
	devs, err := inst.PhysicalDevices()
	if err != nil || len(devs) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan device (%v)", err)
	}
	phys := &devs[0]
	for i := range devs {
		if devs[i].DeviceID == strixHaloDeviceID {
			phys = &devs[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue (%v)", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Fatal(err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no fp16 cooperative-matrix device (%v)", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// relL2 is the whole-tensor error this stage is judged by, rather than the
// max-over-RMS the earlier stages use.
//
// The reason is the object: a denoising trajectory is chaotic, so a single
// element of the last latent can be far from the reference's while the image
// is the same image. What has to be bounded is the tensor as a whole, and
// relative L2 is that -- it is also what says how much of the *signal* the
// difference is, which max-over-RMS does not.
func relL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// TestPipelineAgainstDiffusers walks the eight steps beside diffusers'.
//
// Two bounds, and they say different things. The per-step latent bound is the
// fp16 chain's error compounding through the loop -- it starts at 1.4e-4 and
// the feedback multiplies it by about 1.6 a step -- and what it is really
// checking is that the growth is *geometric from rounding* rather than a
// constant offset from a convention. The image bound is the one that matters:
// after the VAE the two decodes are the same picture.
func TestPipelineAgainstDiffusers(t *testing.T) {
	m := loadManifest(t, runDir)
	if m.Size == 0 {
		t.Skipf("%s is not a dump_zimage_run.py manifest", runDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	init := loadRef(t, m, "latents_init")
	latents := append([]float32(nil), init...)

	// The bound per step. The measured sequence is 1.4e-4 rising to 2.1e-2;
	// 6e-2 leaves room for the run-to-run variation a GPU reduction order has
	// and is still 16x below where the last step's control lands.
	const stepTol = 6e-2
	var steps []float64
	img, tm, err := p.GenerateFrom(m.Prompt, latents, func(s Step) {
		steps = append(steps, relL2(s.Latents, loadRef(t, m, fmt.Sprintf("latents_%d", s.Index))))
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range steps {
		if e > stepTol {
			t.Errorf("step %d: relative L2 %.3g > %.0e", i, e, stepTol)
		}
	}
	t.Logf("%d tokens, %d caption rows, %d unified; per-step relative L2 %s",
		tm.Tokens, tm.CapTotal, tm.Unified, fmtErrs(steps))

	if e := relL2(latents, loadRef(t, m, "latents_final")); e > stepTol {
		t.Errorf("final latent: relative L2 %.3g > %.0e", e, stepTol)
	}

	// The image. 3e-2 is where the measurement lands and it is a bound on a
	// tensor in [-1, 1], so it is about a quarter of an 8-bit level.
	const imageTol = 3e-2
	want := loadRef(t, m, "image")
	if len(img.Data) != len(want) {
		t.Fatalf("decoded %s, reference has %d values", img, len(want))
	}
	e := relL2(img.Data, want)
	if e > imageTol {
		t.Errorf("image: relative L2 %.3g > %.0e", e, imageTol)
	}
	t.Logf("image %dx%d: relative L2 %.3g, max abs %.3g", img.W, img.H, e, maxAbs(img.Data, want))
}

// TestPipelineHeadOnDevice is stage 9's "keep the slow path" check: the same
// image generated with the patch embedder and the final layer on the device
// and on the host, from the same starting latent.
//
// It is the only test that exercises the device head *in the composition* --
// with the real caption behind the image, the padded rows in the stream, and
// eight steps of feedback through the scheduler. Each half is separately
// held to diffusers (TestPipelineAgainstDiffusers here, TestGPUHead* in
// zimage/dit); what this adds is that moving the boundary did not move the
// picture.
//
// The bound is the image bound rather than the step bound, because a
// denoising trajectory amplifies: what the head's fp16 operands cost at step
// one is 1e-4, and by step eight the two runs have taken slightly different
// paths through the same basin.
func TestPipelineHeadOnDevice(t *testing.T) {
	m := loadManifest(t, runDir)
	if m.Size == 0 {
		t.Skipf("%s is not a dump_zimage_run.py manifest", runDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()
	if p.gpuHead == nil {
		t.Skip("this pipeline has no device head")
	}
	init := loadRef(t, m, "latents_init")

	run := func(cpu bool) ([]float32, []float32) {
		p.ctl = controls{cpuHead: cpu}
		latents := append([]float32(nil), init...)
		img, _, err := p.GenerateFrom(m.Prompt, latents, nil)
		p.ctl = controls{}
		if err != nil {
			t.Fatalf("cpuHead=%v: %v", cpu, err)
		}
		return latents, img.Data
	}
	hostLatent, hostImage := run(true)
	devLatent, devImage := run(false)

	const tol = 3e-2
	le, ie := relL2(devLatent, hostLatent), relL2(devImage, hostImage)
	if le > tol || ie > tol {
		t.Errorf("device head vs host head: latent %.3g, image %.3g > %.0e", le, ie, tol)
		return
	}
	t.Logf("device head vs host head: latent %.3g, image %.3g", le, ie)
}

// TestPipelineDetectsErrors is the negative control. Each case breaks one
// thing the composition gets to decide and asserts the break is caught: the
// direction the schedule integrates in, whether the context refiners ran at
// all, and where the caption sits on the rotary axis.
//
// Without it the test above proves only that the pipeline runs, and each of
// these produces an image.
func TestPipelineDetectsErrors(t *testing.T) {
	m := loadManifest(t, runDir)
	if m.Size == 0 {
		t.Skipf("%s is not a dump_zimage_run.py manifest", runDir)
	}
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: m.Size, Height: m.Size, Steps: m.Steps})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	init := loadRef(t, m, "latents_init")
	wantLatent := loadRef(t, m, "latents_final")
	wantImage := loadRef(t, m, "image")

	const stepTol, imageTol = 6e-2, 3e-2
	for _, c := range []struct {
		name string
		ctl  controls
	}{
		{"model output not negated", controls{noNegate: true}},
		{"caption not refined", controls{staleCaption: true}},
		{"caption positions from 0", controls{capPosFromZero: true}},
	} {
		p.ctl = c.ctl
		latents := append([]float32(nil), init...)
		img, _, err := p.GenerateFrom(m.Prompt, latents, nil)
		p.ctl = controls{}
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		le, ie := relL2(latents, wantLatent), relL2(img.Data, wantImage)
		if le <= stepTol || ie <= imageTol {
			t.Errorf("%s: latent %.3g, image %.3g -- inside the bounds, the test would not catch it", c.name, le, ie)
			continue
		}
		t.Logf("%-26s latent %.3g = %.0fx the bound, image %.3g = %.0fx",
			c.name, le, le/stepTol, ie, ie/imageTol)
	}
}

func maxAbs(got, want []float32) float64 {
	var m float64
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > m {
			m = d
		}
	}
	return m
}

func fmtErrs(v []float64) string {
	s := ""
	for _, e := range v {
		s += fmt.Sprintf("%.2g ", e)
	}
	return s
}
