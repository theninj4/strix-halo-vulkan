package kokoro

import (
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// decTol is what the decoder's matrix-core path is held to, block by block,
// as an rms difference against the CPU reference over the tensor's own rms.
//
// It is looser than the generator's gpuTol because a decode block is deeper in
// fp16 terms: its convolutions have a K of 3*1152 against the generator's
// 3*128 to 11*256, so a single dot product accumulates 3456 fp16 products, and
// the AdaIN between them renormalises the scale but not the error.
const decTol = 6e-3

// decInputs is the decoder's four operands, taken from the reference dump so
// that a block is checked on the activations it actually sees — which matters
// because AdaIN divides by a per-channel variance over time.
func decInputs(t *testing.T, m *manifest, model *Model) (asr, f0c, nc, asrRes *Mat) {
	t.Helper()
	v := model.Vocoder
	asr = refMat(t, m, "asr", true)
	f0 := refCurve(t, m, "f0_pred")
	energy := refCurve(t, m, "n_pred")
	curve := func(c *Conv1D, in []float32) *Mat {
		out, err := c.Apply(&Mat{Rows: len(in), Cols: 1, Data: in})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	f0c, nc = curve(v.F0Conv, f0), curve(v.NConv, energy)
	var err error
	if asrRes, err = v.ASRRes.Apply(asr); err != nil {
		t.Fatal(err)
	}
	return asr, f0c, nc, asrRes
}

// TestGPUDecoder runs the five AdaIN blocks on the device and compares each
// one against the reference dump's own `dec_encode` and `dec_decode_*`.
//
// Comparing block by block rather than only at the end is the point: the
// blocks are chained on the device, so an error in the first one would reach
// the last, and a single end-to-end number could not say which of the three
// things that are new here — the non-power-of-two addressing, the leaky
// rectifier after the normalisation, or the shortcut and its 1/sqrt(2) — was
// wrong.
func TestGPUDecoder(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr, f0c, nc, asrRes := decInputs(t, m, model)

	d, err := NewGPUDecoder(dev, model.Vocoder, asr.Rows, DefaultDecoderKernel)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	if err := d.SetStyle(decStyle); err != nil {
		t.Fatal(err)
	}
	if err := d.Upload(asr, f0c, nc, asrRes); err != nil {
		t.Fatal(err)
	}

	names := []string{"dec_encode", "dec_decode_0", "dec_decode_1", "dec_decode_2", "dec_decode_3"}
	for i, name := range names {
		got, err := d.ApplyOne(i)
		if err != nil {
			t.Fatal(err)
		}
		want := refMat(t, m, name, true)
		if got.Rows != want.Rows || got.Cols != want.Cols {
			t.Fatalf("%s: device gives %v, reference has [%d %d]", name, got, want.Rows, want.Cols)
		}
		dv := compare(t, got.Data, want.Data)
		if dv.Rel() > decTol {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d, rms %.3g)", name, dv.Rel(), dv.MaxAbs, dv.At, dv.RMS)
		} else {
			t.Logf("%s: relative %.3g", name, dv.Rel())
		}
	}
}

// TestGPUDecoderChain runs the whole decoder as the pipeline does — one
// submit, one download — and checks its output against the reference's last
// decode block, which is the generator's input.
func TestGPUDecoderChain(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr, f0c, nc, asrRes := decInputs(t, m, model)

	d, err := NewGPUDecoder(dev, model.Vocoder, asr.Rows, DefaultDecoderKernel)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	if err := d.SetStyle(decStyle); err != nil {
		t.Fatal(err)
	}
	got, err := d.Apply(asr, f0c, nc, asrRes)
	if err != nil {
		t.Fatal(err)
	}
	want := refMat(t, m, "dec_decode_3", true)
	dv := compare(t, got.Data, want.Data)
	if dv.Rel() > decTol {
		t.Errorf("chained: relative %.3g (max abs %.3g at %d)", dv.Rel(), dv.MaxAbs, dv.At)
	} else {
		t.Logf("chained: relative %.3g", dv.Rel())
	}
}

// TestGPUDecoderLadder measures every rung of the A_CONV=2 ladder over the
// decoder's own shapes and prints the table DefaultDecoderKernel is chosen
// from.
//
// The shape is the opposite of the generator's — M of 130 or 260 against N of
// 512 or 1024 — so the rung that wins there has no claim here, and §6.2's
// wave32 result is worth re-testing on a fourth geometry.
func TestGPUDecoderLadder(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr, f0c, nc, asrRes := decInputs(t, m, model)

	d, err := NewGPUDecoder(dev, model.Vocoder, asr.Rows, DefaultDecoderKernel)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	if err := d.SetStyle(decStyle); err != nil {
		t.Fatal(err)
	}
	if err := d.Upload(asr, f0c, nc, asrRes); err != nil {
		t.Fatal(err)
	}
	want := refMat(t, m, "dec_decode_3", true)

	for _, k := range ConvKernels() {
		if err := d.SetKernel(k); err != nil {
			t.Logf("%-20s unavailable: %v", k, err)
			continue
		}
		var dis []vk.MultiDispatch
		for i := range d.blocks {
			g, _, err := d.graph(i)
			if err != nil {
				t.Fatal(err)
			}
			dis = append(dis, g...)
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
		dv := compare(t, d.Download().Data, want.Data)
		t.Logf("%-20s %7.2f ms   relative %.3g", k, float64(best.Microseconds())/1000, dv.Rel())
		if dv.Rel() > decTol {
			t.Errorf("%s: relative %.3g", k, dv.Rel())
		}
	}
}

// TestGPUDecoderProfile prints where the decoder's time goes, dispatch by
// dispatch, on GPU timestamps rather than wall clock.
func TestGPUDecoderProfile(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr, f0c, nc, asrRes := decInputs(t, m, model)

	d, err := NewGPUDecoder(dev, model.Vocoder, asr.Rows, DefaultDecoderKernel)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	if err := d.SetStyle(decStyle); err != nil {
		t.Fatal(err)
	}
	if err := d.Upload(asr, f0c, nc, asrRes); err != nil {
		t.Fatal(err)
	}
	stages, err := d.Profile()
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[string]float64{}
	var total float64
	for _, s := range stages {
		total += s.Time
		// The label is "<block> <what>"; the family is what comes after.
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

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// TestGPUTail runs the whole vocoder on the device — decoder, generator,
// conv_post and the inverse transform — and compares the waveform against both
// the CPU reference and the reference dump.
//
// Two comparisons rather than one, because they answer different questions.
// Against the CPU path: does the device compute the same function, fp16 and
// all. Against the dump: is the whole chain still as close to torch as T3
// left it — 18.2 dB from the F0 curve, which is a floor set by the
// reference's own undefined phases and not by anything here. The second is
// what says the *tail* is right, because the tail is the last thing before
// the samples and an error in it would move that number and nothing else.
func TestGPUTail(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr := refMat(t, m, "asr", true)
	f0 := refCurve(t, m, "f0_pred")
	energy := refCurve(t, m, "n_pred")

	host, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.AttachGPU(dev, asr.Rows, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()

	// With the host's excitation on both sides, which is what makes the
	// comparison against the CPU path mean the tail. T7 moved the excitation
	// to the device as well, and it is the one quantity here that is
	// ill-conditioned -- see the second half of this test.
	src := model.Vocoder.Generator.SrcGPU
	model.Vocoder.Generator.SrcGPU = nil
	got, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	model.Vocoder.Generator.SrcGPU = src
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(host) {
		t.Fatalf("device gives %d samples, the host gives %d", len(got), len(host))
	}

	d := compare(t, got, host)
	t.Logf("device against the CPU path      relative %.3g (%.1f dB)", d.Rel(), -20*math.Log10(d.Rel()))
	if d.Rel() > 2e-2 {
		t.Errorf("device against the CPU path: relative %.3g (max abs %.3g at %d)", d.Rel(), d.MaxAbs, d.At)
	}

	want := refCurve(t, m, "audio")
	dev0 := compare(t, got, want)
	cpu := compare(t, host, want)
	t.Logf("device against the dump          relative %.3g (%.1f dB)", dev0.Rel(), -20*math.Log10(dev0.Rel()))
	t.Logf("CPU against the dump             relative %.3g (%.1f dB)", cpu.Rel(), -20*math.Log10(cpu.Rel()))
	// The device must not be meaningfully further from torch than the CPU
	// reference is; the dump's own floor dominates both.
	if dev0.Rel() > 1.5*cpu.Rel() {
		t.Errorf("device is %.2fx further from the dump than the CPU path", dev0.Rel()/cpu.Rel())
	}

	// And now the whole path, excitation included. This number is a *draw*
	// rather than a tolerance: with the noise off the excitation is a
	// constant wherever the signal is unvoiced, its windowed spectrum is
	// analytically zero outside three bins, and the angle of the rounding
	// residual is one arbitrary value per bin repeated over thousands of
	// frames. Walking that constant by five ulps under the device's own
	// transform moves this over 0.097 to 0.203; the CPU path draws 0.123 and
	// the device draws the unluckiest of the eleven. Each half is fine on its
	// own -- the device excitation through the host transform is 0.118, the
	// host excitation through the device transform 0.110 -- so what the bound
	// below records is the width of the lottery and not an error in either.
	full, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	fd := compare(t, full, want)
	t.Logf("device+excitation against dump   relative %.3g (%.1f dB)", fd.Rel(), -20*math.Log10(fd.Rel()))
	if fd.Rel() > 0.25 {
		t.Errorf("the whole device path is %.3g from the dump, outside the measured lottery", fd.Rel())
	}
}
