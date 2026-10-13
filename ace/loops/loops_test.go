package loops

import (
	"math"
	"math/rand/v2"
	"testing"
)

// melody is seconds of stereo that never repeats: a random note every
// quarter second over a noise hiss, so no two stretches are alike.
func melody(seconds float64, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, 1))
	n := int(seconds * SampleRate)
	wav := make([]float32, 2*n)
	note := SampleRate / 4
	var f1, f2 float64
	for i := 0; i < n; i++ {
		if i%note == 0 {
			f1 = 110 * math.Pow(2, float64(rng.IntN(48))/12)
			f2 = 110 * math.Pow(2, float64(rng.IntN(48))/12)
		}
		env := math.Exp(-float64(i%note) / SampleRate * 6)
		t := float64(i) / SampleRate
		v := 0.3*env*(math.Sin(2*math.Pi*f1*t)+0.5*math.Sin(2*math.Pi*f2*t)) + 0.02*rng.NormFloat64()
		wav[2*i], wav[2*i+1] = float32(v), float32(v)
	}
	return wav
}

// tile repeats seconds of wav from at, to wav's length, from frame from on.
func tile(wav []float32, at, seconds float64, from int) []float32 {
	out := append([]float32(nil), wav...)
	a, p := 2*int(at*SampleRate), 2*int(seconds*SampleRate)
	clip := wav[a : a+p]
	for i := from * 2; i < len(out); i += len(clip) {
		copy(out[i:], clip)
	}
	return out
}

func TestMeasure(t *testing.T) {
	song := melody(60, 7)
	for _, c := range []struct {
		name               string
		wav                []float32
		minShare, maxShare float64
		minRep, maxRep     float64
	}{
		{"song", song, 0, 0.1, 0, 0.2},
		{"tile 2 s", tile(song, 20, 2, 0), 0.99, 1, 0.99, 1},
		{"tile 1.37 s", tile(song, 20, 1.37, 0), 0.99, 1, 0.99, 1},
		{"tile 8 s", tile(song, 20, 8, 0), 0.99, 1, 0.99, 1},
		{"second half a 4 s loop", tile(song, 30, 4, 30*SampleRate), 0.45, 0.55, 0.45, 0.55},
		{"silence", make([]float32, len(song)), 0, 0, 0, 0},
	} {
		r := Measure(c.wav)
		t.Logf("%-24s repetition %.3f at %.2f s, loop %.1f s (%.3f) at %.2f s, %v", c.name, r.Repetition, r.RepetitionLag,
			r.LoopSeconds, r.LoopShare, r.LoopLag, r.Took)
		if r.LoopShare < c.minShare || r.LoopShare > c.maxShare || math.IsNaN(r.LoopShare) {
			t.Errorf("%s: loop share %.3f outside [%.2f, %.2f]", c.name, r.LoopShare, c.minShare, c.maxShare)
		}
		if r.Repetition < c.minRep || r.Repetition > c.maxRep || math.IsNaN(r.Repetition) {
			t.Errorf("%s: repetition %.3f outside [%.2f, %.2f]", c.name, r.Repetition, c.minRep, c.maxRep)
		}
	}
}
