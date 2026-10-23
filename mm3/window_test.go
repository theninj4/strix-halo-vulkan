package mm3

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

// mixed applies the condition encoder's mix to the oracle's frame hiddens
// ([frames][8][4096]) as Generate does.
func mixed(t testing.TB, m *ditManifest) [][]float32 {
	_, fh := ditTensor(t, m, "frame_hiddens")
	mix, err := LoadMix(filepath.Join(modelDir, "condition_encoder"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([][]float32, m.Frames)
	for f := range out {
		out[f] = make([]float32, hidden)
		for k := 0; k < len(mix); k++ {
			row := fh[(f*len(mix)+k)*hidden : (f*len(mix)+k+1)*hidden]
			for j, v := range row {
				out[f][j] += mix[k] * v
			}
		}
	}
	return out
}

func rel(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

// TestWindows is N5's gate: the oracle's two windows (300 frames, starts 0
// and 100) through the condition encoder and the sampler, with its noise;
// the condition, every step's latents and each window's result against
// fp32's. The second window runs on our own first window's carry.
func TestWindows(t *testing.T) {
	m := loadDiTManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	if got := WindowStarts(m.Frames); len(got) != len(m.Starts) || got[0] != m.Starts[0] || got[len(got)-1] != m.Starts[len(m.Starts)-1] {
		t.Fatalf("window starts %v, upstream %v", got, m.Starts)
	}
	for i, ts := range m.Scheduler.Timesteps {
		if d := math.Abs(ts - float64(i)/DiTSteps); d > 1e-6 {
			t.Fatalf("timestep %d: upstream %.8f, ours %.8f", i, ts, float64(i)/DiTSteps)
		}
	}
	g, err := LoadDiT(dev, modelDir, LatentsFor(WindowFrames))
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	frames := mixed(t, m)
	var prev *Carry
	for k, start := range m.Starts {
		end := min(start+WindowFrames, m.Frames)
		cond, _, err := g.Condition(frames[start:end])
		if err != nil {
			t.Fatal(err)
		}
		_, wantCond := ditTensor(t, m, "w"+itoa(k)+"_condition")
		shape, noise := ditTensor(t, m, "w"+itoa(k)+"_noise")
		L := shape[1]
		if len(cond) != L*CondChannels {
			t.Fatalf("window %d: %d latents, upstream %d", k, len(cond)/CondChannels, L)
		}
		overlap := 0
		if prev != nil {
			overlap = len(prev.Latents[0])
		}
		// The encoder's own output is compared past the overlap, which the
		// splice replaces.
		ce := rel(cond[overlap*CondChannels:], wantCond[overlap*CondChannels:])
		t0 := time.Now()
		var worst float64
		lat, carry, took, err := g.Denoise(cond, channels(noise, L), prev, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, final := ditTensor(t, m, "w"+itoa(k)+"_final")
		fe := flatRel(lat, final)
		worst = math.Max(worst, fe)
		ce2 := rel(cond, wantCond)
		t.Logf("window %d: %d latents, overlap %d; condition rel %.2e (spliced %.2e); final latents rel %.2e; %v (GPU %v, %d steps)",
			k, L, overlap, ce, ce2, fe, time.Since(t0), took, DiTSteps)
		if !(ce <= 2e-3 && fe <= 1e-2) {
			t.Errorf("window %d: condition rel %.2e, final rel %.2e", k, ce, fe)
		}
		prev = carry
	}
}

func itoa(i int) string {
	return string(rune('0' + i))
}
