package lm

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// BenchmarkCodeStep is phase 2's host work a step, on the oracle's own
// logits at step 50 (a real, flat code distribution): CFG, top-p, the draw.
func BenchmarkCodeStep(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join(tfRef, "given_duration_p1_logits.bin"))
	if err != nil {
		b.Skip(err)
	}
	row := func(step, r int) []float32 {
		off := (step*2 + r) * NumCodes * 4
		out := make([]float32, CodeBase+NumCodes-ImEndID)
		for i := 0; i < NumCodes; i++ {
			out[CodeBase-ImEndID+i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off+4*i:]))
		}
		return out
	}
	cond, uncond := row(50, 0), row(50, 1)
	rng := rand.New(rand.NewPCG(1, 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := codeScores(cond, uncond, 2, 10, 100)
		topP(s, 0.9)
		draw(s, 0.85, rng)
	}
}

// BenchmarkTextStep is phase 1's top-p and draw over the whole vocabulary
// on a peaked step (the oracle's step 50 of given_duration's CoT).
func BenchmarkTextStep(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join(tfRef, "given_duration_p0_logits.bin"))
	if err != nil {
		b.Skip(err)
	}
	row := make([]float32, 217204)
	for i := range row {
		row[i] = negInf
	}
	for i := 0; i < CodeBase; i++ {
		row[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[(50*CodeBase+i)*4:]))
	}
	rng := rand.New(rand.NewPCG(1, 2))
	s := make([]float32, len(row))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(s, row)
		topP(s, 0.9)
		draw(s, 0.85, rng)
	}
}

// topPSorted is topP as A7 wrote it, a full sort of the candidates: the
// reference for the bucketed walk.
func topPSorted(scores []float32, p float64) float64 {
	m := math.Inf(-1)
	for _, s := range scores {
		m = math.Max(m, float64(s))
	}
	var z float64
	var cand []int
	for i, s := range scores {
		if math.IsInf(float64(s), -1) {
			continue
		}
		z += math.Exp(float64(s) - m)
		if float64(s) > m-tailCut {
			cand = append(cand, i)
		}
	}
	sort.Slice(cand, func(a, b int) bool {
		if scores[cand[a]] != scores[cand[b]] {
			return scores[cand[a]] > scores[cand[b]]
		}
		return cand[a] < cand[b]
	})
	keep := map[int]bool{}
	cum, before := 0.0, 0.0
	for _, i := range cand {
		if len(keep) > 0 && cum > p {
			break
		}
		keep[i] = true
		before = cum
		cum += math.Exp(float64(scores[i])-m) / z
	}
	for i := range scores {
		if !keep[i] {
			scores[i] = negInf
		}
	}
	return before
}

// TestTopPBucketed wants the bucketed walk to keep exactly the sorted
// walk's tokens: on every teacher-forced step of the oracle's phases (CFG'd
// codes, and the text steps whole), and on quantised random logits full of
// ties, at several p.
func TestTopPBucketed(t *testing.T) {
	var rows [][]float32
	for _, name := range []string{"given_duration_p1", "given_duration_p0", "sample_ja_p0"} {
		raw, err := os.ReadFile(filepath.Join(tfRef, name+"_logits.bin"))
		if err != nil {
			t.Skip(err)
		}
		f := make([]float32, len(raw)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		if name == "given_duration_p1" {
			for s := 0; s < 101; s++ {
				c, u := f[(2*s)*NumCodes:(2*s+1)*NumCodes], f[(2*s+1)*NumCodes:(2*s+2)*NumCodes]
				rows = append(rows, cfg(c, u, 2))
			}
			continue
		}
		for s := 0; s < 101; s += 5 {
			rows = append(rows, f[s*CodeBase:(s+1)*CodeBase])
		}
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for i := 0; i < 200; i++ {
		r := make([]float32, 1+rng.IntN(5000))
		scale := []float64{0.01, 1, 5, 40}[i%4]
		for j := range r {
			r[j] = float32(math.Round(rng.NormFloat64()*scale*4) / 4)
			if rng.IntN(10) == 0 {
				r[j] = negInf
			}
		}
		rows = append(rows, r)
	}
	for n, r := range rows {
		for _, p := range []float64{0.5, 0.9, 0.99} {
			a, b := append([]float32(nil), r...), append([]float32(nil), r...)
			ca, cb := topPCum(a, p), topPSorted(b, p)
			for j := range a {
				if (a[j] == negInf) != (b[j] == negInf) {
					t.Fatalf("row %d p %v: token %d kept %v by the walk, %v by the sort", n, p, j, a[j] != negInf, b[j] != negInf)
				}
			}
			if math.Abs(ca-cb) > 1e-12 {
				t.Fatalf("row %d p %v: cum before %v vs %v", n, p, ca, cb)
			}
		}
	}
	t.Logf("%d rows x 3 p: identical", len(rows))
}
