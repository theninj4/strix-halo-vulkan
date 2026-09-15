package kokoro

import (
	"fmt"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("kokoro-test")
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

// gpuTol is what the matrix-core path is held to: the rms of the difference
// against the CPU reference as a fraction of the tensor's own rms.
//
// fp16 operands carry a relative error of their own magnitude, and a residual
// block is six convolutions deep with an instance norm between each pair —
// which *renormalises* after every one, so the error does not compound the way
// it would down a plain stack. The bound is the measured worst times three.
const gpuTol = 3e-3

// gpuBlocks builds the device object for the generator's blocks at one stage.
func gpuBlocks(t *testing.T, dev *vk.Device, stage int) (*GPUBlocks, []*SnakeResBlock, *Model, []float32) {
	t.Helper()
	m := loadModel(t)
	man := loadManifest(t)
	decStyle, _, err := m.Style(man.Voice, len([]rune(man.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	g := m.Vocoder.Generator
	frames := 2 * man.LengthRegulator.Frames * 10
	blocks := []*SnakeResBlock{g.ResBlocks[0], g.ResBlocks[1], g.ResBlocks[2], g.NoiseRes[0]}
	if stage == 1 {
		frames = frames*6 + 1
		blocks = []*SnakeResBlock{g.ResBlocks[3], g.ResBlocks[4], g.ResBlocks[5], g.NoiseRes[1]}
	}
	gb, err := NewGPUBlocks(dev, blocks, frames, DefaultConvKernel)
	if err != nil {
		t.Fatal(err)
	}
	if err := gb.SetStyle(decStyle); err != nil {
		t.Fatal(err)
	}
	return gb, blocks, m, decStyle
}

// TestGPUResBlock runs every residual block in the generator on the device
// against the CPU reference, at both of the generator's rates.
//
// The input is the reference's own `gen_up_*` tensor, so a block is checked
// on the activations it actually sees rather than on noise — which matters
// here because AdaIN divides by a per-channel variance, and a synthetic input
// with a different one would test a different conditioning.
func TestGPUResBlock(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)

	for _, stage := range []int{0, 1} {
		t.Run(fmt.Sprintf("stage%d", stage), func(t *testing.T) {
			gb, blocks, _, style := gpuBlocks(t, dev, stage)
			defer gb.Destroy()
			x := refMat(t, m, fmt.Sprintf("gen_up_%d", stage), true)
			if x.Rows != gb.Frames() {
				t.Fatalf("reference has %d frames, the device is built for %d", x.Rows, gb.Frames())
			}
			for i, b := range blocks {
				want, err := b.Apply(x, style)
				if err != nil {
					t.Fatal(err)
				}
				got, err := gb.ApplyOne(i, x)
				if err != nil {
					t.Fatal(err)
				}
				d := compare(t, got.Data, want.Data)
				if d.Rel() > gpuTol {
					t.Errorf("block %d: relative %.3g (max abs %.3g at %d, rms %.3g)",
						i, d.Rel(), d.MaxAbs, d.At, d.RMS)
				} else {
					t.Logf("block %d %-11s k=%d d=%v  relative %.3g  rms %.3g",
						i, got, b.Convs1[0].Kernel, []int{b.Convs1[0].Dilation, b.Convs1[1].Dilation,
							b.Convs1[2].Dilation}, d.Rel(), d.RMS)
				}
			}
		})
	}
}

// TestGPUBiasCancels holds the claim the staging relies on: the bias of a
// convolution whose output goes straight into an AdaIN has no effect at all,
// because the normalisation subtracts the per-channel mean over time.
//
// It is checked on the CPU, where the bias can be zeroed without restaging
// anything, and it is what lets the device path leave convs1's bias out of the
// weight arena entirely.
func TestGPUBiasCancels(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	man := m
	style, _, err := model.Style(man.Voice, len([]rune(man.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	b := model.Vocoder.Generator.ResBlocks[3]
	x := refMat(t, m, "gen_up_1", true)
	want, err := b.Apply(x, style)
	if err != nil {
		t.Fatal(err)
	}

	saved := make([][]float32, len(b.Convs1))
	for i, c := range b.Convs1 {
		saved[i] = c.Bias
		c.Bias = nil
	}
	got, err := b.Apply(x, style)
	for i, c := range b.Convs1 {
		c.Bias = saved[i]
	}
	if err != nil {
		t.Fatal(err)
	}
	d := compare(t, got.Data, want.Data)
	if d.Rel() > 1e-6 {
		t.Errorf("dropping convs1's bias changed the block by %.3g relative", d.Rel())
	}
	t.Logf("convs1 bias dropped: relative %.3g (max abs %.3g)", d.Rel(), d.MaxAbs)

	// And the control: convs2's bias reaches the residual add, so dropping it
	// must change the output. Without this the test above would pass on a
	// block whose biases were all zero.
	saved2 := make([][]float32, len(b.Convs2))
	for i, c := range b.Convs2 {
		saved2[i] = c.Bias
		c.Bias = nil
	}
	got2, err := b.Apply(x, style)
	for i, c := range b.Convs2 {
		c.Bias = saved2[i]
	}
	if err != nil {
		t.Fatal(err)
	}
	if d2 := compare(t, got2.Data, want.Data); d2.Rel() < 1e-3 {
		t.Errorf("dropping convs2's bias changed the block by only %.3g; the control is degenerate", d2.Rel())
	} else {
		t.Logf("convs2 bias dropped: relative %.3g (the control)", d2.Rel())
	}
}

// TestGPUConvLadder times every rung over a whole residual block at both of
// the generator's rates.
//
// The shape is new to this repository: M in the thousands and N of 128 or 256,
// where the DiT's projections are M=4096 by N=3840 and parakeet's are M=138 by
// N=1024. A tall narrow GEMM has few tiles across, so the rung that wins is
// the one that keeps enough workgroups in flight down the M direction.
func TestGPUConvLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("ladder sweep")
	}
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)

	for _, stage := range []int{0, 1} {
		gb, _, _, _ := gpuBlocks(t, dev, stage)
		x := refMat(t, m, fmt.Sprintf("gen_up_%d", stage), true)
		if err := gb.Upload(x); err != nil {
			t.Fatal(err)
		}
		type row struct {
			k ConvKernel
			d time.Duration
		}
		var rows []row
		for _, k := range ConvKernels() {
			if err := gb.SetKernel(k); err != nil {
				t.Logf("stage %d %-18s unavailable: %v", stage, k, err)
				continue
			}
			if _, err := gb.ApplyOne(0, x); err != nil { // warm
				t.Fatal(err)
			}
			best := time.Duration(1 << 62)
			for r := 0; r < 3; r++ {
				t0 := time.Now()
				if err := gb.Run(0); err != nil {
					t.Fatal(err)
				}
				if d := time.Since(t0); d < best {
					best = d
				}
			}
			rows = append(rows, row{k, best})
		}
		for _, r := range rows {
			t.Logf("stage %d  %-18s %7.2f ms  %.2fx", stage, r.k,
				float64(r.d.Microseconds())/1000, float64(r.d)/float64(rows[0].d))
		}
		gb.Destroy()
	}
}

// TestGPUProfile prints where a block's time goes on the device, which is the
// number that says whether the scalar passes around the convolutions have
// become the cost.
func TestGPUProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("profile")
	}
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	gb, _, _, _ := gpuBlocks(t, dev, 1)
	defer gb.Destroy()
	if err := gb.Upload(refMat(t, m, "gen_up_1", true)); err != nil {
		t.Fatal(err)
	}
	stages, err := gb.Profile(0)
	if err != nil {
		t.Fatal(err)
	}
	byKind := map[string]float64{}
	var total float64
	for _, s := range stages {
		kind := s.Kind
		for _, p := range []string{"conv1", "conv2", "norm1", "norm2", "residual"} {
			if len(kind) >= len(p) && kind[:len(p)] == p {
				kind = p
				break
			}
		}
		byKind[kind] += s.Time
		total += s.Time
	}
	t.Logf("%d dispatches, %.3f ms total", len(stages), total*1000)
	for _, k := range []string{"conv1", "conv2", "norm1", "norm2", "residual"} {
		t.Logf("  %-10s %7.3f ms  %5.1f%%", k, byKind[k]*1000, 100*byKind[k]/total)
	}
}

// TestGPUVocoder runs the whole vocoder with the residual blocks on the
// device and compares the waveform against the CPU reference's.
//
// The bound is the same relative measure the per-block test uses, applied to
// the samples: everything downstream of the blocks is identical code on both
// paths, so what this adds over TestGPUResBlock is that the two stages are
// wired up in the right order and that the averaging and the noise block go
// where they belong.
func TestGPUVocoder(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr := refMat(t, m, "asr", true)
	f0, energy := refCurve(t, m, "f0_pred"), refCurve(t, m, "n_pred")

	t0 := time.Now()
	want, cpuTrace, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	cpu := time.Since(t0)

	if err := model.AttachGPU(dev, m.LengthRegulator.Frames, decStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()
	if _, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle); err != nil { // warm
		t.Fatal(err)
	}
	t0 = time.Now()
	got, tr, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	gpu := time.Since(t0)

	// Against the CPU path's own stages, not against the reference dump: the
	// dump's `gen_up_*` are 4% and 2% away from *both* paths because the
	// excitation's phase spectrogram is not reproducible (see
	// TestHarmonicSource), and that deviation would swamp the one this test
	// is about.
	for i, s := range tr.Generator.Stages {
		d := compare(t, s.Data, cpuTrace.Generator.Stages[i].Data)
		if d.Rel() > gpuTol {
			t.Errorf("gen_up_%d: relative %.3g against the CPU path", i, d.Rel())
		} else {
			t.Logf("gen_up_%d %-12s relative %.3g against the CPU path", i, s, d.Rel())
		}
	}
	d := compare(t, got, want)
	if d.Rel() > gpuTol {
		t.Errorf("waveform: relative %.3g against the CPU path", d.Rel())
	}
	t.Logf("vocoder %.0f ms on the CPU, %.0f ms with the blocks on the device — %.2fx; "+
		"waveform relative %.3g", float64(cpu.Microseconds())/1000,
		float64(gpu.Microseconds())/1000, float64(cpu)/float64(gpu), d.Rel())
}
