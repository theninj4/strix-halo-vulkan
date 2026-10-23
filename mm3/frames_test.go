package mm3

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// TestSamplerSupport holds the sampler to upstream's on the oracle's fp32
// logits: every code the oracle drew (49 semantic, 343 residual) is inside
// the support our draw picks from. Our RNG is not torch's, so the draws
// themselves are not compared; N1 and N2 gate the logits.
func TestSamplerSupport(t *testing.T) {
	m := loadManifest(t)
	_, codes := refInt32(t, m, "codes")
	_, hc := refFloat32(t, m, "head_cond")
	_, hu := refFloat32(t, m, "head_uncond")
	_, dc := refFloat32(t, m, "depth_cond")
	_, du := refFloat32(t, m, "depth_uncond")
	span := HeadHi - HeadLo
	sizes := 0
	for f := 0; f <= m.Frames; f++ {
		g, keep := globalCandidates(hc[f*span:(f+1)*span], hu[f*span:(f+1)*span])
		sup := topKSupport(g, keep)
		sizes += len(sup)
		want := int(codes[f*8]) + CodeOffset - HeadLo
		if !contains(sup, want) {
			t.Errorf("step %d: semantic code %d outside our support of %d", f, codes[f*8], len(sup))
		}
		for i := 0; i < dCodes; i++ {
			o := (f*dCodes + i) * dVocab
			x := cfgSpan(dc[o:o+dVocab], du[o:o+dVocab])
			idx := make([]int, dVocab)
			for j := range idx {
				idx[j] = j
			}
			if !contains(topKSupport(x, idx), int(codes[f*8+1+i])) {
				t.Errorf("step %d c%d: code %d outside our support", f, i+1, codes[f*8+1+i])
			}
		}
	}
	t.Logf("all %d draws inside the support; the global support averages %.1f codes", (m.Frames+1)*8, float64(sizes)/float64(m.Frames+1))
}

func contains(xs []int, x int) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// TestGenerate is N3's run: the oracle's prompt, free-running on our RNG
// for MM3_FRAMES frames (default 250, 10 s of audio), timed per part.
func TestGenerate(t *testing.T) {
	m := loadManifest(t)
	dev, err := openDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	bank := testBank()
	g, err := LoadLM(dev, filepath.Join(modelDir, "language_model"), Options{MaxLen: 4096, Slots: 2, Rows: 512, Bank: bank})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer g.Destroy()
	d, err := LoadDepth(dev, filepath.Join(modelDir, "rvq_depth_decoder"), bank)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	mix, err := LoadMix(filepath.Join(modelDir, "condition_encoder"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("mix %v", mix)
	shape, ids := refInt32(t, m, "prompt_ids")
	T := shape[1]
	frames := envInt("MM3_FRAMES", 250)
	t0 := time.Now()
	s, err := Generate(g, d, mix, ids[:T], ids[T:], frames, rand.New(rand.NewPCG(11, 1)), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(t0)
	n := float64(len(s.Cond))
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 / n }
	t.Logf("%s: %d frames (ended %v) in %v: prefill %v; a frame %.1f ms LM + %.1f depth + %.1f sampling = %.2f s a second of audio",
		bank, len(s.Cond), s.Ended, wall, s.Prefill, ms(s.LMTime), ms(s.DepthTime), ms(s.SampleTime),
		(wall-s.Prefill).Seconds()/(n/FrameRate))
	for f, c := range s.Cond {
		for _, v := range c {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("frame %d's condition is not finite", f)
			}
		}
	}
	t.Logf("first codes %v, last %v", s.Codes[0], s.Codes[len(s.Codes)-1])
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}
