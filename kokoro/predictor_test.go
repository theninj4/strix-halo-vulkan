package kokoro

import (
	"fmt"
	"testing"
)

// TestDurationEncoder walks the prosody predictor's text side block by block.
// Each LSTM's output is checked before the normalisation and each
// normalisation's after the style vector is re-concatenated, which is the
// shape the reference dumps them in and the shape the next LSTM consumes.
func TestDurationEncoder(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	_, style, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}

	dEn := refMat(t, m, "d_en", true)
	de := model.Predictor.TextEncoder
	s := Broadcast(style, dEn.Rows)
	h, err := Concat(dEn, s)
	if err != nil {
		t.Fatal(err)
	}
	for i := range de.LSTMs {
		if h, err = de.LSTMs[i].Apply(h); err != nil {
			t.Fatal(err)
		}
		check(t, m, fmt.Sprintf("dur_lstm_%d", i), h, true, 1e-5)
		if err = de.Norms[i].Apply(h, style); err != nil {
			t.Fatal(err)
		}
		if h, err = Concat(h, s); err != nil {
			t.Fatal(err)
		}
		check(t, m, fmt.Sprintf("dur_norm_%d", i), h, true, 1e-5)
	}

	d, err := de.Apply(dEn, style, nil)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "dur_encoded", d, false, 1e-5)
}

// TestDurations checks the head and, more importantly, the rounding.
//
// The durations are integers, so "close" is not a measure that applies to
// them: every frame boundary downstream is placed by these numbers, and one
// token rounding the other way changes the length of the waveform. The float
// sums are bounded loosely and the integers have to match exactly.
func TestDurations(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	d := refMat(t, m, "dur_encoded", false)

	x, err := model.Predictor.LSTM.Apply(d)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "dur_lstm_out", x, false, 1e-5)
	logits, err := model.Predictor.DurationHead.Apply(x)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "dur_logits", logits, false, 1e-5)

	durations, raw, err := model.Predictor.Durations(d, m.Speed, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantRaw, _ := loadRef(t, m, "durations")
	if dev := compare(t, raw, wantRaw); dev.Rel() > 1e-5 {
		t.Errorf("duration sums: relative %.3g", dev.Rel())
	} else {
		t.Logf("durations      [%d]         max abs %.3g  relative %.3g", len(raw), dev.MaxAbs, dev.Rel())
	}

	if len(durations) != len(m.LengthRegulator.Durations) {
		t.Fatalf("%d durations, reference has %d", len(durations), len(m.LengthRegulator.Durations))
	}
	total := 0
	for i, got := range durations {
		if want := m.LengthRegulator.Durations[i]; got != want {
			t.Errorf("token %d: duration %d, reference has %d (raw %.6f vs %.6f)",
				i, got, want, raw[i], wantRaw[i])
		}
		total += got
	}
	if total != m.LengthRegulator.Frames {
		t.Fatalf("%d frames, reference has %d", total, m.LengthRegulator.Frames)
	}
	t.Logf("%d tokens -> %d frames -> %d samples, exactly",
		len(durations), total, total*model.Config.SamplesPerFrame())
}

// TestLengthRegulator checks the gather against the matmul the reference
// writes: `en` there is the duration encoder's output times a [T, L] one-hot,
// and here it is a row copied per frame.
func TestLengthRegulator(t *testing.T) {
	m := loadManifest(t)
	d := refMat(t, m, "dur_encoded", false)
	en, err := Expand(d, m.LengthRegulator.Durations)
	if err != nil {
		t.Fatal(err)
	}
	if dev := check(t, m, "en", en, true, 0); dev.MaxAbs != 0 {
		t.Errorf("the gather is not a copy: %.3g", dev.MaxAbs)
	}

	// The alignment matrix is dumped too; it is the same statement in the
	// reference's own form, so a disagreement between the two would be a
	// disagreement about which token owns which frame.
	aln, _ := loadRef(t, m, "alignment")
	frames := m.LengthRegulator.Frames
	f := 0
	for tok, dur := range m.LengthRegulator.Durations {
		for i := 0; i < dur; i++ {
			if aln[tok*frames+f] != 1 {
				t.Fatalf("frame %d is not owned by token %d in the reference alignment", f, tok)
			}
			f++
		}
	}
	if f != frames {
		t.Fatalf("the durations cover %d of %d frames", f, frames)
	}
	if _, err := Expand(d, m.LengthRegulator.Durations[:1]); err == nil {
		t.Error("a short duration list expanded")
	}
}

// TestProsody walks the expanded side: the shared recurrence, both AdaIN
// stacks and the two curves they project to.
//
// This is where the first upsampling AdaIN block runs, so it is also the
// check on the two halves of that block agreeing about length — a depthwise
// transposed convolution on the residual path and a nearest-neighbour repeat
// on the shortcut, which only line up at stride 2 with output padding 1.
func TestProsody(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	_, style, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	en := refMat(t, m, "en", true)

	shared, err := model.Predictor.Shared.Apply(en)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "shared_out", shared, false, 1e-5)

	for _, arm := range []struct {
		name   string
		blocks []*AdainResBlk1d
	}{{"f0", model.Predictor.F0}, {"n", model.Predictor.N}} {
		h := shared
		for i, b := range arm.blocks {
			if h, err = b.Apply(h, style); err != nil {
				t.Fatal(err)
			}
			check(t, m, fmt.Sprintf("%s_blk_%d", arm.name, i), h, true, 1e-5)
		}
	}

	f0, energy, err := model.Predictor.Prosody(en, style, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * m.LengthRegulator.Frames; len(f0) != want || len(energy) != want {
		t.Fatalf("f0 %d and energy %d curves, want %d each", len(f0), len(energy), want)
	}
	for _, c := range []struct {
		name string
		got  []float32
	}{{"f0_pred", f0}, {"n_pred", energy}} {
		want, _ := loadRef(t, m, c.name)
		dev := compare(t, c.got, want)
		if dev.Rel() > 1e-5 {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d)", c.name, dev.Rel(), dev.MaxAbs, dev.At)
		} else {
			t.Logf("%-14s [%d]        max abs %.3g  rms %.3g  relative %.3g",
				c.name, len(c.got), dev.MaxAbs, dev.RMS, dev.Rel())
		}
	}
}

// TestProsodyChain runs T2 end to end from the phoneme ids alone — no
// reference tensor anywhere on the path — and checks where it lands. This is
// the test that would catch a stage being fed the wrong thing, which the
// stage-by-stage checks above cannot.
func TestProsodyChain(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	_, style, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	ids, dropped := model.Config.Phonemes(m.Phonemes)
	if dropped != 0 {
		t.Fatalf("%d phonemes dropped", dropped)
	}

	p, err := model.Prosody(ids, style, m.Speed)
	if err != nil {
		t.Fatal(err)
	}
	for i, got := range p.Durations {
		if want := m.LengthRegulator.Durations[i]; got != want {
			t.Errorf("token %d: duration %d, reference has %d", i, got, want)
		}
	}
	if p.Frames != m.LengthRegulator.Frames {
		t.Fatalf("%d frames, reference has %d", p.Frames, m.LengthRegulator.Frames)
	}
	if got := p.Samples(model.Config); got != m.Decoder.Samples {
		t.Errorf("%d samples, the reference's waveform is %d", got, m.Decoder.Samples)
	}
	check(t, m, "asr", p.ASR, true, 1e-5)
	check(t, m, "en", p.Encoded, true, 1e-5)
	for _, c := range []struct {
		name string
		got  []float32
	}{{"f0_pred", p.F0}, {"n_pred", p.Energy}} {
		want, _ := loadRef(t, m, c.name)
		if dev := compare(t, c.got, want); dev.Rel() > 1e-5 {
			t.Errorf("%s: relative %.3g", c.name, dev.Rel())
		} else {
			t.Logf("%-14s [%d]        relative %.3g", c.name, len(c.got), dev.Rel())
		}
	}
	t.Logf("%q -> %d tokens -> %d frames -> %.3f s",
		m.Text, len(p.Tokens), p.Frames, p.Seconds(model.Config))
}
