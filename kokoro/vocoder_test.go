package kokoro

import (
	"fmt"
	"math"
	"testing"
)

// refCurve reads a 1-D dumped tensor.
func refCurve(t *testing.T, m *manifest, name string) []float32 {
	t.Helper()
	v, _ := loadRef(t, m, name)
	return v
}

// TestSTFTGeometry pins the frame arithmetic the whole vocoder hangs off,
// before any weights are involved. Every count here is a function of the
// three numbers in config.json's istftnet block and the durations, and an
// off-by-one in any of them is a silent desynchronisation rather than an
// error.
func TestSTFTGeometry(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	g := model.Vocoder.Generator
	frames := m.LengthRegulator.Frames
	samples := m.Decoder.Samples

	if got := g.STFT.Frames(samples); got != m.Tensors["gen_post"].Shape[1] {
		t.Errorf("%d samples make %d STFT frames, the reference has %d",
			samples, got, m.Tensors["gen_post"].Shape[1])
	}
	if got := g.ISTFT.Samples(g.STFT.Frames(samples)); got != samples {
		t.Errorf("the round trip gives %d samples, want %d", got, samples)
	}
	// 130 alignment frames -> 260 through the last decode block -> 2600 and
	// 15600 through the two upsamplers -> 15601 after the reflection pad.
	n := 2 * frames
	for i, up := range g.Ups {
		n = up.OutFrames(n)
		want := []int{2600, 15600}[i]
		if n != want {
			t.Fatalf("upsampler %d takes %d frames to %d, want %d", i, n, want, n)
		}
	}
	if n+1 != g.STFT.Frames(samples) {
		t.Errorf("the generator makes %d frames and the excitation %d", n+1, g.STFT.Frames(samples))
	}
	if got := g.Source.UpsampleScale * frames * 2; got != samples {
		t.Errorf("the source module makes %d samples, want %d", got, samples)
	}
}

// TestHarmonicSource walks the excitation, and is the one test in this
// package that does not bound its error at float noise.
//
// Two separate things are loose here and both are the *reference's*, not the
// port's. The integrated phase reaches 1.3e5 radians in float32, where one ulp
// is 0.016 radians; and in unvoiced regions the excitation is a constant, so
// its spectrum is exactly zero outside three bins and the phase of the rest is
// whatever the rounding says. The upsampled F0 and the voiced mask, which
// have neither problem, are checked as exact.
func TestHarmonicSource(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	g := model.Vocoder.Generator

	f0 := refCurve(t, m, "f0_pred")
	source, f0Up, uv, sines := g.Source.Apply(f0)

	for _, c := range []struct {
		name string
		got  []float32
	}{{"f0_upsampled", f0Up}, {"uv", uv}} {
		want := refCurve(t, m, c.name)
		if d := compare(t, c.got, want); d.MaxAbs != 0 {
			t.Errorf("%s differs by %.3g at %d; it should be exact", c.name, d.MaxAbs, d.At)
		}
	}
	// sine_waves is dumped [samples, harmonics] — already channel-last.
	if d := compare(t, sines.Data, refCurve(t, m, "sine_waves")); d.Rel() > 0.05 {
		t.Errorf("sine waves: relative %.3g", d.Rel())
	} else {
		t.Logf("sine_waves     relative %.3g (float32 phase at 1.3e5 radians)", d.Rel())
	}
	d := compare(t, source, refCurve(t, m, "har_source"))
	if d.Rel() > 0.02 {
		t.Errorf("har_source: relative %.3g", d.Rel())
	}
	t.Logf("har_source     max abs %.3g  rms %.3g  relative %.3g", d.MaxAbs, d.RMS, d.Rel())

	// The excitation's *spectrogram* is where the reference stops being
	// reproducible. Taken from the reference's own excitation, so that the
	// phase accumulator above is out of the picture: the magnitudes then
	// agree to float noise and the phases still do not.
	har := g.Harmonic(refCurve(t, m, "har_source"))
	bins := g.STFT.Bins()
	var worstMag, loudFlip, loudWrapped, sq float64
	flips, wrapped := 0, 0
	for tt := 0; tt < har.Rows; tt++ {
		row := har.Row(tt)
		for b := 0; b < bins; b++ {
			want := math.Abs(float64(refPlane(t, m, "har_spec", b, tt)))
			sq += want * want
			if diff := math.Abs(float64(row[b])) - want; math.Abs(diff) > worstMag {
				worstMag = math.Abs(diff)
			}
			d := float64(row[bins+b]) - float64(refPlane(t, m, "har_phase", b, tt))
			if math.Abs(d) <= 1e-3 {
				continue
			}
			flips++
			if want > loudFlip {
				loudFlip = want
			}
			// Modulo 2*pi the two agree wherever the difference is only which
			// side of the branch cut atan2 landed on. What is left after
			// wrapping is a phase that genuinely differs.
			if w := math.Mod(d+3*math.Pi, 2*math.Pi) - math.Pi; math.Abs(w) > 1e-3 {
				wrapped++
				if want > loudWrapped {
					loudWrapped = want
				}
			}
		}
	}
	rms := math.Sqrt(sq / float64(har.Rows*bins))
	t.Logf("har_spec       max abs %.3g against an rms of %.3g", worstMag, rms)
	t.Logf("har_phase      %d of %d bins disagree, the loudest at magnitude %.3g (%.0fx below the rms)",
		flips, har.Rows*bins, loudFlip, rms/loudFlip)
	t.Logf("               %d of those still disagree modulo 2*pi, the loudest at %.3g (%.0fx below)",
		wrapped, loudWrapped, rms/loudWrapped)
	if worstMag > 1e-5 {
		t.Errorf("the excitation's magnitude spectrum should be exact, max abs %.3g", worstMag)
	}
	// Two different irreproducibilities, and the claim about each is that it
	// only happens where there is nothing to be reproducible about. A phase
	// that differs modulo 2*pi at a real magnitude would be a bug; one that
	// differs only by which side of the cut it landed on is atan2 doing its
	// job on a value the network should not have been shown.
	if loudWrapped > rms/1e3 {
		t.Errorf("a phase differs modulo 2*pi at magnitude %.3g, only %.0fx below the rms %.3g",
			loudWrapped, rms/loudWrapped, rms)
	}
}

// refPlane reads one element of a dumped [C, T] tensor without transposing
// the whole thing, which for har_phase is 343k floats read a quarter of a
// million times.
var planeCache = map[string][]float32{}

func refPlane(t *testing.T, m *manifest, name string, c, frame int) float32 {
	t.Helper()
	v, ok := planeCache[name]
	if !ok {
		v, _ = loadRef(t, m, name)
		planeCache[name] = v
	}
	return v[c*m.Tensors[name].Shape[1]+frame]
}

// TestVocoder walks the decoder and the generator, and then separates the
// port's error from the reference's.
//
// The chain is run three ways. From the F0 curve, which is what the model
// does and which inherits everything TestHarmonicSource measured. From the
// reference's excitation, which removes the phase accumulator. And from the
// reference's excitation *spectrogram*, which removes the undefined phases as
// well and leaves only this package's own arithmetic — that last one is the
// number that says whether the vocoder is correct.
func TestVocoder(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	decStyle, _, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	v := model.Vocoder
	asr := refMat(t, m, "asr", true)
	f0 := refCurve(t, m, "f0_pred")
	energy := refCurve(t, m, "n_pred")

	wav, tr, err := v.Apply(asr, f0, energy, decStyle)
	if err != nil {
		t.Fatal(err)
	}
	check(t, m, "f0_conv", tr.F0, true, 1e-6)
	check(t, m, "n_conv", tr.N, true, 1e-6)
	check(t, m, "asr_res", tr.ASRRes, true, 1e-6)
	check(t, m, "dec_encode", tr.Encode, true, 1e-5)
	for i, d := range tr.Decode {
		check(t, m, fmt.Sprintf("dec_decode_%d", i), d, true, 1e-5)
	}

	want := refCurve(t, m, "audio")
	if len(wav) != m.Decoder.Samples {
		t.Fatalf("%d samples, reference has %d", len(wav), m.Decoder.Samples)
	}
	own := compare(t, wav, want)
	t.Logf("audio, own excitation          relative %.3g (%.1f dB)", own.Rel(), -20*math.Log10(own.Rel()))

	x := tr.Decode[len(tr.Decode)-1]
	sub, _, err := v.Generator.ApplyWithSource(x, decStyle, refCurve(t, m, "har_source"))
	if err != nil {
		t.Fatal(err)
	}
	viaSource := compare(t, sub, want)
	t.Logf("audio, reference excitation    relative %.3g (%.1f dB)", viaSource.Rel(), -20*math.Log10(viaSource.Rel()))

	har := NewMat(m.Tensors["har_spec"].Shape[1], 2*v.Generator.STFT.Bins())
	bins := v.Generator.STFT.Bins()
	for tt := 0; tt < har.Rows; tt++ {
		for b := 0; b < bins; b++ {
			har.Data[tt*2*bins+b] = refPlane(t, m, "har_spec", b, tt)
			har.Data[tt*2*bins+bins+b] = refPlane(t, m, "har_phase", b, tt)
		}
	}
	exact, gtr, err := v.Generator.ApplyWithHarmonic(x, decStyle, har)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range gtr.Stages {
		check(t, m, fmt.Sprintf("gen_up_%d", i), s, true, 1e-5)
	}
	check(t, m, "gen_post", gtr.Post, true, 1e-5)
	d := compare(t, exact, want)
	t.Logf("audio, reference spectrogram   relative %.3g (%.1f dB)", d.Rel(), -20*math.Log10(d.Rel()))
	if d.Rel() > 1e-4 {
		t.Errorf("the vocoder's own error is %.3g; given the reference's excitation it should be float noise", d.Rel())
	}
	// And the two looser numbers, bounded where the reference's own
	// conditioning puts them rather than pretended away.
	if own.Rel() > 0.2 {
		t.Errorf("audio from the model's own excitation: relative %.3g", own.Rel())
	}
}

// TestSpeak runs the whole model from a phoneme string, which is the only
// test here with no reference tensor anywhere on the path.
func TestSpeak(t *testing.T) {
	m := loadManifest(t)
	model := loadModel(t)
	wav, p, err := model.Speak(m.Phonemes, m.Voice, m.Speed)
	if err != nil {
		t.Fatal(err)
	}
	if len(wav) != m.Decoder.Samples {
		t.Fatalf("%d samples, reference has %d", len(wav), m.Decoder.Samples)
	}
	d := compare(t, wav, refCurve(t, m, "audio"))
	t.Logf("%q -> %.3f s, relative %.3g (%.1f dB)",
		m.Text, p.Seconds(model.Config), d.Rel(), -20*math.Log10(d.Rel()))
	if d.Rel() > 0.2 {
		t.Errorf("relative %.3g", d.Rel())
	}

	// With the noise on, the output is a different sample from the same
	// distribution: the same length, a comparable level, and not the same
	// numbers. All three are worth pinning, because a broken RNG path would
	// show up as one of them.
	model.SetExcitationNoise(1)
	noisy, _, err := model.Speak(m.Phonemes, m.Voice, m.Speed)
	if err != nil {
		t.Fatal(err)
	}
	if len(noisy) != len(wav) {
		t.Fatalf("the noise changed the length: %d against %d", len(noisy), len(wav))
	}
	q, n := rms(wav), rms(noisy)
	if n < 0.5*q || n > 2*q {
		t.Errorf("with noise the level is %.4g against %.4g", n, q)
	}
	if compare(t, noisy, wav).MaxAbs == 0 {
		t.Error("the noise changed nothing")
	}
	t.Logf("excitation noise on: rms %.4g against %.4g", n, q)
}

func rms(v []float32) float64 {
	var sq float64
	for _, x := range v {
		sq += float64(x) * float64(x)
	}
	return math.Sqrt(sq / float64(len(v)))
}

// TestSnake pins the generator's activation, which is the only nonlinearity
// in the model that is neither monotonic nor bounded.
func TestSnake(t *testing.T) {
	// x + sin(ax)^2/a is x at every multiple of pi/a, and x + 1/a at the odd
	// multiples of pi/(2a) — so it rides above the identity and touches it
	// periodically. Both are checked, since a port that dropped the square or
	// the 1/a would still look like a plausible activation.
	for _, a := range []float32{0.5, 1, 2.5} {
		for k := 0; k < 4; k++ {
			x := float32(float64(k) * math.Pi / float64(a))
			if got := Snake(x, a); math.Abs(float64(got-x)) > 1e-5 {
				t.Errorf("Snake(%g, %g) = %g, want %g at a zero of the ripple", x, a, got, x)
			}
			peak := float32((float64(k) + 0.5) * math.Pi / float64(a))
			if got, want := Snake(peak, a), peak+1/a; math.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Snake(%g, %g) = %g, want %g at a peak", peak, a, got, want)
			}
		}
	}
}
