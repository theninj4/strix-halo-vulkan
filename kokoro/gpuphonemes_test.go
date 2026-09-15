package kokoro

import (
	"testing"

	"strix-halo-vulkan/vk"
)

// phonemeTol is what the chained phoneme side is held to against the host,
// as an rms difference over the tensor's own rms.
//
// It is the recurrences' own bound, because they are what dominates the error:
// a normalisation and a convolution in fp32 add nothing a 130-step LSTM in
// fp16 has not already contributed. What it does *not* cover is the durations,
// which are integers and are compared as integers — see below, and see
// GPUPhonemes for the one place in this model where this bound was not good
// enough.
const phonemeTol = 3e-3

// newPhonemes stages the whole phoneme side and the ALBERT in front of it.
func newPhonemes(t *testing.T, dev *vk.Device, model *Model, m *manifest, frames int) func() {
	t.Helper()
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUPhonemes(dev, model, frames)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetStyle(predStyle); err != nil {
		g.Destroy()
		t.Fatal(err)
	}
	bert, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		g.Destroy()
		t.Fatal(err)
	}
	model.BERTGPU, model.PhonemesGPU = bert, g
	return func() {
		model.BERTGPU, model.PhonemesGPU = nil, nil
		bert.Destroy()
		g.Destroy()
	}
}

// TestGPUPhonemes walks the chain link by link against the host.
//
// Link by link rather than only at the end, because T6d is *entirely* a
// question of whether each object reads the previous one's output at the right
// offset and stride. An error in the wiring is not a small error — it is the
// wrong tensor — and the one number at the end that would catch it could not
// say which link it was.
func TestGPUPhonemes(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}

	// The host chain, from the dump's own `d_en` so the two start identical.
	dEn := refMat(t, m, "d_en", true)
	hostEnc, err := model.Predictor.TextEncoder.Apply(dEn, predStyle, nil)
	if err != nil {
		t.Fatal(err)
	}
	hostX, err := model.Predictor.LSTM.Apply(hostEnc)
	if err != nil {
		t.Fatal(err)
	}
	hostTE, err := model.TextEncoder.Apply(m.InputIDs, nil)
	if err != nil {
		t.Fatal(err)
	}

	cleanup := newPhonemes(t, dev, model, m, m.LengthRegulator.Frames)
	defer cleanup()
	g := model.PhonemesGPU

	emb, err := model.TextEncoder.Embedding.Rows(m.InputIDs)
	if err != nil {
		t.Fatal(err)
	}
	gotX, err := g.Encode(dEn, emb)
	if err != nil {
		t.Fatal(err)
	}

	// The duration encoder's output is the tensor the length regulator
	// gathers from, so it is read where it lies rather than where it would
	// have been returned.
	encOff, encLDA := g.Encoded()
	gotEnc := NewMat(dEn.Rows, encLDA)
	copy(gotEnc.Data, g.abuf.ReadFloat32At(int(encOff), dEn.Rows*encLDA))
	// The host keeps the style concatenated; the device keeps it in the
	// operand, so only the first 2H channels are the same tensor.
	hostHead := NewMat(hostEnc.Rows, encLDA)
	for r := 0; r < hostEnc.Rows; r++ {
		copy(hostHead.Row(r), hostEnc.Row(r))
	}

	for _, c := range []struct {
		name      string
		got, want []float32
	}{
		{"duration encoder", gotEnc.Data, hostHead.Data},
		{"the head's input", gotX.Data, hostX.Data},
		{"text encoder", g.TextEncoded().Data, hostTE.Data},
	} {
		dv := compare(t, c.got, c.want)
		if dv.Rel() > phonemeTol {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d, rms %.3g)",
				c.name, dv.Rel(), dv.MaxAbs, dv.At, dv.RMS)
		} else {
			t.Logf("%-18s relative %.3g", c.name, dv.Rel())
		}
	}

	// And the curves, which need the durations the head decides.
	logits, err := model.Predictor.DurationHead.Apply(gotX)
	if err != nil {
		t.Fatal(err)
	}
	durations, _ := durationsFrom(logits, m.Speed)
	f0, energy, err := g.Prosody(durations)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		got  []float32
		ref  string
	}{{"f0", f0, "f0_pred"}, {"n", energy, "n_pred"}} {
		dv := compare(t, c.got, refCurve(t, m, c.ref))
		if dv.Rel() > phonemeTol {
			t.Errorf("%s against the dump: relative %.3g", c.name, dv.Rel())
		} else {
			t.Logf("%-18s relative %.3g against the dump", c.name, dv.Rel())
		}
	}
}

// TestGPUPhonemesProfile prints where the phoneme side's GPU time goes, by
// family.
//
// By family rather than by dispatch because seven hundred of them are LSTM
// steps, and the question the table has to answer is whether anything other
// than the recurrences is worth looking at.
func TestGPUPhonemesProfile(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	cleanup := newPhonemes(t, dev, model, m, m.LengthRegulator.Frames)
	defer cleanup()

	stages, err := model.PhonemesGPU.Profile(len(m.InputIDs), m.LengthRegulator.Frames)
	if err != nil {
		t.Fatal(err)
	}
	var total float64
	for _, s := range stages {
		total += s.Time
	}
	for _, s := range stages {
		if s.Time*1000 < 0.005 {
			continue
		}
		t.Logf("  %-18s %7.3f ms  %4.1f%%", s.Kind, s.Time*1000, 100*s.Time/total)
	}
	t.Logf("  %-18s %7.3f ms", "total", total*1000)
}

// TestGPUPhonemesDurations runs the whole phoneme side on the device and
// checks that nothing the alignment depends on moved.
//
// The same test T6a and T6b each ended with, and for a stronger reason here:
// the durations are a rounded sum of fifty sigmoids of a *recurrent* network's
// output, so an error in the state does not merely perturb them, it
// accumulates along the sequence before it reaches them. Fifty integers that
// do not move is the claim; the curves agreeing to a few parts in ten
// thousand is the supporting evidence.
func TestGPUPhonemesDurations(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}

	host, err := model.Prosody(m.InputIDs, predStyle, m.Speed)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUPhonemes(dev, model, host.Frames)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.SetStyle(predStyle); err != nil {
		t.Fatal(err)
	}
	bert, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer bert.Destroy()
	model.BERTGPU, model.PhonemesGPU = bert, g
	defer func() { model.BERTGPU, model.PhonemesGPU = nil, nil }()

	got, err := model.Prosody(m.InputIDs, predStyle, m.Speed)
	if err != nil {
		t.Fatal(err)
	}
	if got.Frames != host.Frames {
		t.Fatalf("%d frames on the device, %d on the host", got.Frames, host.Frames)
	}
	var moved int
	for i := range host.Durations {
		if got.Durations[i] != host.Durations[i] {
			moved++
			t.Errorf("token %d: %d frames on the device, %d on the host",
				i, got.Durations[i], host.Durations[i])
		}
	}
	t.Logf("%d of %d durations unchanged", len(host.Durations)-moved, len(host.Durations))
	for _, c := range []struct {
		name      string
		got, want []float32
	}{
		{"f0", got.F0, host.F0},
		{"n", got.Energy, host.Energy},
		{"asr", got.ASR.Data, host.ASR.Data},
	} {
		dv := compare(t, c.got, c.want)
		t.Logf("%-4s relative %.3g", c.name, dv.Rel())
		if dv.Rel() > lstmTol {
			t.Errorf("%s: relative %.3g", c.name, dv.Rel())
		}
	}
	t.Logf("phoneme side %v on the device against %v on the host (recurrences %v against %v)",
		got.Times.Total(), host.Times.Total(), got.Times.Recurrence, host.Times.Recurrence)
}
