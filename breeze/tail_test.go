package breeze

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestTailTrim: the unvoiced tail is held and cut where Ending says, at any
// chunking; voiced audio is never held, and an ending at the end keeps all.
func TestTailTrim(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	tone := func(sec float64) []float32 {
		s := make([]float32, int(sec*SampleRate))
		for i := range s {
			s[i] = float32(0.2 * math.Sin(2*math.Pi*180*float64(i)/SampleRate))
		}
		return s
	}
	noise := func(sec float64) []float32 {
		s := make([]float32, int(sec*SampleRate))
		for i := range s {
			s[i] = float32(0.05 * rng.NormFloat64())
		}
		return s
	}
	// run trims in with Ending at `at` seconds before the end (0: keep all).
	run := func(in []float32, chunk int, at float64) ([]float32, float64) {
		var out []float32
		tr := &tailTrim{
			emit: func(s []float32) error { out = append(out, s...); return nil },
			ending: func(wav []float32) (int, error) {
				if len(wav) < SampleRate/2 {
					t.Errorf("Ending saw %d samples, not the held tail and its context", len(wav))
				}
				return len(wav) - int(at*SampleRate), nil
			},
		}
		for i := 0; i < len(in); i += chunk {
			if err := tr.push(in[i:min(i+chunk, len(in))]); err != nil {
				t.Fatal(err)
			}
			// Nothing voiced waits for the end.
			if v := len(in[:min(i+chunk, len(in))]); len(out) < v-tailHold && v > SampleRate {
				t.Errorf("chunk %d: %d of %d samples held", chunk, v-len(out), v)
			}
		}
		d, err := tr.finish()
		if err != nil {
			t.Fatal(err)
		}
		return out, d.Seconds()
	}
	speech := slices.Concat(tone(4), noise(0.5))
	var first []float32
	for _, chunk := range []int{2 * FrameSamples, 777, len(speech)} {
		// The cut is at the ending or, past it, where holding began: the
		// end of the last voiced window, which may reach into the noise.
		out, d := run(speech, chunk, 0.5)
		if k := len(out) - 4*SampleRate - tailFade; k < 0 || k > tailWin || math.Abs(float64(len(out))/SampleRate+d-4.5) > 1e-6 {
			t.Errorf("chunk %d: %d samples kept, %.3f s dropped", chunk, len(out), d)
		}
		if v := out[len(out)-1]; math.Abs(float64(v)) > 0.01 {
			t.Errorf("chunk %d: ends at %g, not faded", chunk, v)
		}
		if first == nil {
			first = out
		} else if !slices.Equal(out, first) {
			t.Errorf("chunk %d: the output depends on the chunking", chunk)
		}
	}
	for name, in := range map[string][]float32{
		"breath kept": speech,
		"whisper":     noise(2),
		"pause":       slices.Concat(tone(0.5), noise(1.5), tone(0.5)),
	} {
		if out, d := run(in, 2*FrameSamples, 0); !slices.Equal(out, in) || d != 0 {
			t.Errorf("%s: %d of %d samples out, %v dropped", name, len(out), len(in), d)
		}
	}
	// An ending inside audio already passed on cuts only what is held.
	if out, _ := run(speech, 2*FrameSamples, 2); !slices.Equal(out, first) {
		t.Errorf("an early ending: %d samples kept, not %d", len(out), len(first))
	}
}
