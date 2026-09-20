package audio

import (
	"math"
	"testing"
)

// tone is s seconds of a sine at hz, at rate.
func tone(hz float64, rate int, s float64) *Clip {
	n := int(s * float64(rate))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * hz * float64(i) / float64(rate)))
	}
	return &Clip{Samples: out, Rate: rate}
}

// rms over an interior window, skipping the filter's edge transient at each
// end: the kernel is ~48 input samples wide, and the first and last outputs
// see zero-padding rather than signal.
func rms(x []float32, skip int) float64 {
	if len(x) <= 2*skip {
		return 0
	}
	var sum float64
	for _, v := range x[skip : len(x)-skip] {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)-2*skip))
}

func TestResampleSameRateIsACopy(t *testing.T) {
	in := tone(440, 24000, 0.01)
	out, err := Resample(in, 24000)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Samples) != len(in.Samples) {
		t.Fatalf("length = %d, want %d", len(out.Samples), len(in.Samples))
	}
	for i := range out.Samples {
		if out.Samples[i] != in.Samples[i] {
			t.Fatalf("sample %d = %v, want %v", i, out.Samples[i], in.Samples[i])
		}
	}
	// A copy and not an alias, so a caller may hand the result to a
	// destructive front end.
	out.Samples[0] = 42
	if in.Samples[0] == 42 {
		t.Error("the result shares its backing array with the input")
	}
}

// TestResampleKeepsDuration is the property the round trip depends on: the
// clip the transcription endpoint receives has to be the same length in
// seconds as the one the speech endpoint produced, because both the
// benchmark's real-time factors and parakeet's word timings are read against
// it.
func TestResampleKeepsDuration(t *testing.T) {
	for _, tc := range []struct{ from, to int }{
		{24000, 16000}, // kokoro -> parakeet, the one that matters
		{16000, 24000},
		{44100, 16000},
		{8000, 16000},
	} {
		in := tone(440, tc.from, 1.5)
		out, err := Resample(in, tc.to)
		if err != nil {
			t.Fatal(err)
		}
		if out.Rate != tc.to {
			t.Errorf("%d -> %d: rate = %d", tc.from, tc.to, out.Rate)
		}
		// Within one output sample.
		if d := math.Abs(out.Duration() - in.Duration()); d > 1/float64(tc.to) {
			t.Errorf("%d -> %d: duration %.6f s, want %.6f s",
				tc.from, tc.to, out.Duration(), in.Duration())
		}
	}
}

// TestResamplePassesSpeechBand checks a tone well inside both Nyquists comes
// through with its amplitude and its phase: the resampled clip is compared
// sample for sample against the same sine generated at the target rate, which
// is the only comparison that catches a phase error in the polyphase
// indexing. A kernel whose phases were off by one would still have the right
// spectrum.
func TestResamplePassesSpeechBand(t *testing.T) {
	for _, hz := range []float64{100, 440, 1000, 3000} {
		in := tone(hz, 24000, 0.2)
		out, err := Resample(in, 16000)
		if err != nil {
			t.Fatal(err)
		}
		want := tone(hz, 16000, 0.2)
		// Skip 100 samples each end: zero-padding makes the edges roll on.
		var worst float64
		for i := 100; i < len(out.Samples)-100 && i < len(want.Samples); i++ {
			if d := math.Abs(float64(out.Samples[i] - want.Samples[i])); d > worst {
				worst = d
			}
		}
		if worst > 1e-2 {
			t.Errorf("%.0f Hz: worst sample error %.4f against the analytic sine", hz, worst)
		}
	}
}

// TestResampleRejectsAboveNyquist is why this is a filter and not an
// interpolation. 10 kHz exists at 24 kHz and cannot exist at 16 kHz; a naive
// decimation would fold it to 6 kHz at full amplitude, straight into the
// middle of parakeet's mel bank.
func TestResampleRejectsAboveNyquist(t *testing.T) {
	for _, hz := range []float64{8400, 10000, 11000} {
		in := tone(hz, 24000, 0.2)
		out, err := Resample(in, 16000)
		if err != nil {
			t.Fatal(err)
		}
		level := rms(out.Samples, 200) / rms(in.Samples, 200)
		db := 20 * math.Log10(level+1e-12)
		if db > -40 {
			t.Errorf("%.0f Hz survives at %.1f dB, want below -40", hz, db)
		}
	}
}

// TestResampleUpsamplesWithoutRipple is the normalisation of each phase: with
// unnormalised kernels a steady tone comes out amplitude-modulated at
// rate/L Hz, which is a much harder thing to notice downstream than an
// outright wrong sample.
func TestResampleUpsamplesWithoutRipple(t *testing.T) {
	in := tone(440, 16000, 0.2)
	out, err := Resample(in, 24000)
	if err != nil {
		t.Fatal(err)
	}
	// The peak of every cycle should be the same height. Take the envelope
	// as the max over each 10 ms window and check its spread.
	win := out.Rate / 100
	var lo, hi float64 = 1, 0
	for s := win; s+2*win < len(out.Samples); s += win {
		var peak float64
		for _, v := range out.Samples[s : s+win] {
			if a := math.Abs(float64(v)); a > peak {
				peak = a
			}
		}
		lo, hi = math.Min(lo, peak), math.Max(hi, peak)
	}
	if hi-lo > 0.02 {
		t.Errorf("envelope spans [%.4f, %.4f]; a normalised kernel set should hold it flat", lo, hi)
	}
}

func TestResampleRejectsBadRates(t *testing.T) {
	in := tone(440, 24000, 0.01)
	if _, err := Resample(in, 0); err == nil {
		t.Error("a zero target rate was accepted")
	}
	if _, err := Resample(&Clip{Samples: in.Samples}, 16000); err == nil {
		t.Error("a clip with no rate was accepted")
	}
	if _, err := Resample(nil, 16000); err == nil {
		t.Error("a nil clip was accepted")
	}
}
