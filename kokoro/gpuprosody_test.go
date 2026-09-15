package kokoro

import (
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// prosTol is what the F0/N stacks' device path is held to as an rms difference
// against the reference dump over the tensor's own rms.
//
// It is the decoder's decTol, for the decoder's reason: these are the same
// block, and their convolutions have a K of 3*512 and 3*256 against the
// decoder's 3*1152 — shallower, so if anything the bound is generous. What it
// is *not* is a bound on the curves, which are checked separately and more
// tightly: a projection to one channel averages 256 of these errors.
const prosTol = 6e-3

// prosInputs is the stacks' single operand — the shared recurrence's output,
// taken from the dump so the blocks are checked on the activations they
// actually see. AdaIN divides by a per-channel variance over time, so an
// input that is merely close is not the same test.
func prosInputs(t *testing.T, m *manifest) *Mat {
	t.Helper()
	return refMat(t, m, "shared_out", false)
}

func newProsody(t *testing.T, dev *vk.Device, model *Model, m *manifest, frames int) *GPUProsody {
	t.Helper()
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUProsody(dev, model.Predictor, frames, DefaultDecoderKernel)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetStyle(predStyle); err != nil {
		g.Destroy()
		t.Fatal(err)
	}
	return g
}

// TestGPUProsody runs the six AdaIN blocks on the device and compares each one
// against the reference dump's own `f0_blk_*` and `n_blk_*`.
//
// Block by block rather than only at the curves, for the decoder's reason and
// one more of this stage's own: two of the three shapes here are new. A block
// whose shortcut is its input rather than a convolution, and a block that
// upsamples in the *middle* of a stack rather than at its end, are exactly
// the two cases the decoder never exercises — and a single number at the end
// of a 1-wide projection could not say which of them was wrong.
func TestGPUProsody(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	shared := prosInputs(t, m)

	g := newProsody(t, dev, model, m, shared.Rows)
	defer g.Destroy()

	names := []string{"f0_blk_0", "f0_blk_1", "f0_blk_2", "n_blk_0", "n_blk_1", "n_blk_2"}
	for i, name := range names {
		got, err := g.ApplyOne(shared, i)
		if err != nil {
			t.Fatal(err)
		}
		want := refMat(t, m, name, true)
		if got.Rows != want.Rows || got.Cols != want.Cols {
			t.Fatalf("%s: device gives %v, reference has [%d %d]", name, got, want.Rows, want.Cols)
		}
		dv := compare(t, got.Data, want.Data)
		if dv.Rel() > prosTol {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d, rms %.3g)", name, dv.Rel(), dv.MaxAbs, dv.At, dv.RMS)
		} else {
			t.Logf("%-10s relative %.3g", name, dv.Rel())
		}
	}
}

// TestGPUProsodyCurves runs both stacks as the pipeline does — one submit, one
// download of 2 KB — and checks the two curves against the dump and against
// the CPU path.
//
// The two comparisons answer different questions and the second is the
// stricter one. Against the dump: is the port still as close to torch as T2
// left it. Against the CPU reference: does the device compute the same
// function, fp16 and all — and that is the number that would move if the
// projection, which is this stage's only new kernel, were wrong.
func TestGPUProsodyCurves(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	en := refMat(t, m, "en", true)
	shared := prosInputs(t, m)

	hostF0, hostN, err := model.Predictor.Prosody(en, predStyle, nil)
	if err != nil {
		t.Fatal(err)
	}

	g := newProsody(t, dev, model, m, shared.Rows)
	defer g.Destroy()
	f0, energy, err := g.Apply(shared)
	if err != nil {
		t.Fatal(err)
	}
	if len(f0) != 2*shared.Rows || len(energy) != 2*shared.Rows {
		t.Fatalf("curves are %d and %d frames, want %d", len(f0), len(energy), 2*shared.Rows)
	}
	for _, c := range []struct {
		name      string
		got, want []float32
		bound     float64
	}{
		{"f0 against the dump", f0, refCurve(t, m, "f0_pred"), prosTol},
		{"n  against the dump", energy, refCurve(t, m, "n_pred"), prosTol},
		{"f0 against the CPU", f0, hostF0, prosTol},
		{"n  against the CPU", energy, hostN, prosTol},
	} {
		dv := compare(t, c.got, c.want)
		if dv.Rel() > c.bound {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d, rms %.3g)",
				c.name, dv.Rel(), dv.MaxAbs, dv.At, dv.RMS)
		} else {
			t.Logf("%-22s relative %.3g", c.name, dv.Rel())
		}
	}
}

// TestGPUProsodyLadder measures every rung of the A_CONV=2 ladder over these
// stacks' shapes.
//
// The decoder's winner has no claim here and is not assumed to carry: its
// convolutions are M of 130 or 260 against N of 512 and 1024 at K = 3*1152,
// and these are the same M against N of 512 and 256 at K = 3*512 and 3*256 —
// a third of the K and a quarter of the output tile count, which is the
// regime where a rung's occupancy matters more than its K loop.
func TestGPUProsodyLadder(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	shared := prosInputs(t, m)

	g := newProsody(t, dev, model, m, shared.Rows)
	defer g.Destroy()
	if err := g.Upload(shared); err != nil {
		t.Fatal(err)
	}
	want := refCurve(t, m, "f0_pred")

	for _, k := range ConvKernels() {
		if err := g.SetKernel(k); err != nil {
			t.Logf("%-20s unavailable: %v", k, err)
			continue
		}
		dis, _, err := g.graph()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := vk.DispatchMultiTimed(dis, 1, 1, true); err != nil {
			t.Fatal(err)
		}
		best := time.Hour
		for r := 0; r < 5; r++ {
			t0 := time.Now()
			if _, err := vk.DispatchMultiTimed(dis, 1, 1, true); err != nil {
				t.Fatal(err)
			}
			if e := time.Since(t0); e < best {
				best = e
			}
		}
		got := g.abuf.ReadFloat32At(int(g.aOut[0]), g.OutFrames())
		dv := compare(t, got, want)
		t.Logf("%-20s %7.2f ms   relative %.3g", k, float64(best.Microseconds())/1000, dv.Rel())
		if dv.Rel() > prosTol {
			t.Errorf("%s: relative %.3g", k, dv.Rel())
		}
	}
}

// TestGPUProsodyProfile prints where the stacks' time goes, dispatch by
// dispatch, on GPU timestamps rather than wall clock.
func TestGPUProsodyProfile(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	shared := prosInputs(t, m)

	g := newProsody(t, dev, model, m, shared.Rows)
	defer g.Destroy()
	if err := g.Upload(shared); err != nil {
		t.Fatal(err)
	}
	stages, err := g.Profile()
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[string]float64{}
	var total float64
	for _, s := range stages {
		total += s.Time
		kind := s.Kind
		if i := indexByte(kind, ' '); i >= 0 {
			kind = kind[i+1:]
		}
		kinds[kind] += s.Time
	}
	for _, s := range stages {
		t.Logf("  %-24s %7.3f ms", s.Kind, s.Time*1000)
	}
	for k, v := range kinds {
		t.Logf("%-24s %7.3f ms  %4.1f%%", k, v*1000, 100*v/total)
	}
	t.Logf("%-24s %7.3f ms", "total", total*1000)
	if math.IsNaN(total) {
		t.Fatal("no timings")
	}
}
