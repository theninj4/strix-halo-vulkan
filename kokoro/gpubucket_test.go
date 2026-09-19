package kokoro

import (
	"sort"
	"testing"
)

// TestGPUBucketedFrames is the claim T9 rests on: an attachment staged for a
// ceiling produces, for a shorter utterance, exactly what an attachment
// staged for that utterance's own length produces.
//
// Exactly — not within a tolerance. The two runs issue the same dispatches
// with the same push constants over the same weights; all that differs is
// where the arenas sit and how much of each one is live. Anything that read
// past the live rows would show up here as a difference rather than as a
// tolerance, which is the point: the one thing bucketing can break is the
// zero padding the convolutions assume, and that is not a rounding error.
//
// The long utterance runs *first*, so the short one runs against arenas full
// of somebody else's activations — which is the state a server is in on every
// request after its first.
func TestGPUBucketedFrames(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr := refMat(t, m, "asr", true)
	f0, energy := refCurve(t, m, "f0_pred"), refCurve(t, m, "n_pred")
	frames := asr.Rows

	// Staged for exactly this utterance: the pre-T9 arrangement, and what
	// every existing GPU test measures.
	if err := model.AttachGPU(dev, frames, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	want, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	model.DetachGPU()

	// Staged for a ceiling that is neither a multiple of the utterance nor of
	// the 64-frame padding, so no coincidence of alignment can hide a stale
	// row.
	ceiling := 2*frames + 13
	if err := model.AttachGPU(dev, ceiling, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()
	if got := model.MaxFrames(); got != ceiling {
		t.Errorf("MaxFrames is %d, staged for %d", got, ceiling)
	}

	// A full-length utterance first, to leave every arena dirty.
	long, longF0, longEnergy := tile(asr, ceiling), tileCurve(f0, 2*ceiling), tileCurve(energy, 2*ceiling)
	longOut, _, err := model.Vocoder.Apply(long, longF0, longEnergy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	if want := ceiling * model.Config.SamplesPerFrame(); len(longOut) != want {
		t.Errorf("the ceiling utterance is %d samples, the geometry says %d", len(longOut), want)
	}

	got, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("%d samples from the bucketed run, %d from the exact one", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			d := compare(t, got, want)
			t.Fatalf("sample %d is %g bucketed and %g exact; relative %.3g over the waveform",
				i, got[i], want[i], d.Rel())
		}
	}
	t.Logf("%d frames through arenas built for %d: %d samples, bit for bit",
		frames, ceiling, len(got))
}

// TestGPUBucketedRejectsOverflow pins the other half of the contract: an
// utterance longer than the ceiling is an error and not a wrong waveform.
func TestGPUBucketedRejectsOverflow(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	asr := refMat(t, m, "asr", true)
	f0, energy := refCurve(t, m, "f0_pred"), refCurve(t, m, "n_pred")

	if err := model.AttachGPU(dev, asr.Rows-1, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()
	if _, _, err := model.Vocoder.Apply(asr, f0, energy, decStyle); err == nil {
		t.Fatalf("%d frames ran against arenas for %d", asr.Rows, asr.Rows-1)
	}
}

// tile repeats a tensor's rows until it is `rows` long.
func tile(x *Mat, rows int) *Mat {
	out := NewMat(rows, x.Cols)
	for r := 0; r < rows; r++ {
		copy(out.Row(r), x.Row(r%x.Rows))
	}
	return out
}

// tileCurve is tile for one of the two curves, which are one value a frame.
func tileCurve(x []float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = x[i%len(x)]
	}
	return out
}

// TestGPUBucketedSpeak is TestGPUBucketedFrames over the whole model rather
// than the vocoder: the phoneme side, the length regulator and both AdaIN
// stacks resize with everything else, and an utterance spoken through a
// ceiling is the one spoken through arenas its own size.
//
// Three utterances through one attachment, in the order a server sees them —
// short, long, short — and then the same short one through an attachment
// built for exactly its frame count.
func TestGPUBucketedSpeak(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}

	// Generous enough for both utterances; the point of a ceiling is that it
	// is nobody's exact length.
	const ceiling = 1000
	if err := model.AttachGPU(dev, ceiling, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	first, p, err := model.Speak(m.Phonemes, m.Voice, 1)
	if err != nil {
		t.Fatal(err)
	}
	long, lp, err := model.Speak(m.Phonemes+" "+m.Phonemes, m.Voice, 1)
	if err != nil {
		t.Fatal(err)
	}
	if lp.Frames <= p.Frames {
		t.Fatalf("the long utterance is %d frames against the short one's %d", lp.Frames, p.Frames)
	}
	_ = long
	again, _, err := model.Speak(m.Phonemes, m.Voice, 1)
	if err != nil {
		t.Fatal(err)
	}
	same(t, "the same utterance twice through one attachment", again, first)
	model.DetachGPU()

	// Now the pre-T9 arrangement: arenas for exactly this utterance.
	if err := model.AttachGPU(dev, p.Frames, decStyle, predStyle, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()
	exact, _, err := model.Speak(m.Phonemes, m.Voice, 1)
	if err != nil {
		t.Fatal(err)
	}
	same(t, "a bucketed utterance against an exactly staged one", again, exact)
	t.Logf("%d frames through arenas built for %d, after a %d-frame utterance: %d samples, bit for bit",
		p.Frames, ceiling, lp.Frames, len(again))
}

// same requires two waveforms to agree exactly. See TestGPUBucketedFrames for
// why the bound is equality and not a tolerance.
func same(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d samples against %d", what, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			d := compare(t, got, want)
			t.Fatalf("%s: sample %d is %g against %g; relative %.3g over the waveform",
				what, i, got[i], want[i], d.Rel())
		}
	}
}

// TestGPUSetVoice is the other half of what a staged attachment has to
// survive: a request in a voice the attachment was not conditioned on.
//
// Same bound as the rest of this file. Re-conditioning writes the same gamma
// and beta into the same arenas that staging would have, so the utterance is
// the one a fresh attachment in that voice produces, sample for sample.
func TestGPUSetVoice(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)

	other := ""
	for _, name := range sortedVoices(model) {
		if name != m.Voice {
			other = name
			break
		}
	}
	if other == "" {
		t.Skip("the checkpoint has one voice")
	}
	phonemes := m.Phonemes
	style := func(voice string) (dec, pred []float32) {
		d, p, err := model.Style(voice, len([]rune(phonemes)))
		if err != nil {
			t.Fatal(err)
		}
		return d, p
	}
	decA, predA := style(m.Voice)
	decB, predB := style(other)

	// Staged on one voice, re-conditioned onto the other.
	if err := model.AttachGPU(dev, 600, decA, predA, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.Speak(phonemes, m.Voice, 1); err != nil {
		t.Fatal(err)
	}
	if err := model.SetVoice(decB, predB); err != nil {
		t.Fatal(err)
	}
	got, _, err := model.Speak(phonemes, other, 1)
	if err != nil {
		t.Fatal(err)
	}
	model.DetachGPU()

	// Staged on that voice to begin with.
	if err := model.AttachGPU(dev, 600, decB, predB, DefaultConvKernel); err != nil {
		t.Fatal(err)
	}
	defer model.DetachGPU()
	want, _, err := model.Speak(phonemes, other, 1)
	if err != nil {
		t.Fatal(err)
	}
	same(t, "a re-conditioned voice against one staged with it", got, want)
	t.Logf("%s spoken through an attachment staged for %s: %d samples, bit for bit",
		other, m.Voice, len(got))
}

// sortedVoices is the checkpoint's voice names in a stable order, so that
// "some other voice" means the same one on every run.
func sortedVoices(m *Model) []string {
	out := make([]string, 0, len(m.Voices))
	for name := range m.Voices {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
