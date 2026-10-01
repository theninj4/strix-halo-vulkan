package llm

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// L7c: the sampler. Small tests, because the only one that matters for the
// gate is the argmax — but `quickselect` is the kind of code that is wrong on
// one input in a thousand and produces a plausible token every time.

// TestArgmax checks the greedy sampler against a linear scan, including the
// tie rule: `ggml_argmax` keeps the first, so a later equal value must not
// displace it.
func TestArgmax(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 200; trial++ {
		n := 1 + rng.Intn(500)
		v := make([]float32, n)
		for i := range v {
			// A coarse quantisation, so ties happen on purpose.
			v[i] = float32(rng.Intn(20))
		}
		want := 0
		for i, x := range v {
			if x > v[want] {
				want = i
			}
			_ = x
		}
		if got := Argmax(v); int(got) != want {
			t.Fatalf("trial %d: argmax %d (%v), want %d (%v)", trial, got, v[got], want, v[want])
		}
	}
}

// TestTopN is the check quickselect needs: the n it returns have to be the n
// largest, in order, for every length and every n.
func TestTopN(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 300; trial++ {
		n := 1 + rng.Intn(300)
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		k := 1 + rng.Intn(n)
		got := TopN(v, k)
		if len(got) != k {
			t.Fatalf("trial %d: %d of %d, want %d", trial, len(got), n, k)
		}
		sorted := append([]float32(nil), v...)
		sort.Slice(sorted, func(a, b int) bool { return sorted[a] > sorted[b] })
		for i, id := range got {
			if v[id] != sorted[i] {
				t.Fatalf("trial %d of %d, rank %d: %v, want %v", trial, n, i, v[id], sorted[i])
			}
		}
	}
}

// TestSamplerTemperatureZeroIsGreedy is the one the gate rests on: a
// comparison against another implementation is only a comparison if the
// sampler is a function of the logits alone.
func TestSamplerTemperatureZeroIsGreedy(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	v := make([]float32, 1000)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	for _, s := range []*Sampler{
		NewSampler(0, 0, 0, 1),
		NewSampler(0, 40, 0.9, 2),
	} {
		if !s.Greedy() {
			t.Fatalf("temperature %v is not greedy", s.Temp)
		}
		if got := s.Sample(v); got != Argmax(v) {
			t.Errorf("sampled %d, argmax is %d", got, Argmax(v))
		}
	}
}

// TestSamplerTopKIsAWall: nothing outside the k highest can ever come out,
// however many times it is asked.
func TestSamplerTopKIsAWall(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	v := make([]float32, 500)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	const k = 5
	allowed := map[int32]bool{}
	for _, id := range TopN(v, k) {
		allowed[id] = true
	}
	s := NewSampler(1.0, k, 0, 3)
	for i := 0; i < 2000; i++ {
		if id := s.Sample(v); !allowed[id] {
			t.Fatalf("sampled %d (%v), which is not in the top %d", id, v[id], k)
		}
	}
}

// TestSamplerDistribution: at temperature 1 with no cut, the empirical
// frequencies have to be the softmax. It is a loose bound — 2000 draws over
// eight outcomes — because what would fail it is a wrong *shape*, not a
// sampling fluctuation.
func TestSamplerDistribution(t *testing.T) {
	v := []float32{2, 1, 0, -1, 0.5, 1.5, -2, 0.25}
	var sum float64
	want := make([]float64, len(v))
	for i, x := range v {
		want[i] = math.Exp(float64(x))
		sum += want[i]
	}
	for i := range want {
		want[i] /= sum
	}
	s := NewSampler(1.0, 0, 0, 5)
	const draws = 20000
	count := make([]int, len(v))
	for i := 0; i < draws; i++ {
		count[s.Sample(v)]++
	}
	for i := range want {
		got := float64(count[i]) / draws
		if math.Abs(got-want[i]) > 0.02 {
			t.Errorf("token %d: %.4f of the draws, softmax says %.4f", i, got, want[i])
		}
	}
}

// TestSamplerPenalties: a token in the window loses to one just below it once
// the presence penalty is larger than the gap, the row comes back untouched,
// and a token that has slid out of the window is not penalised any more.
func TestSamplerPenalties(t *testing.T) {
	v := []float32{0, 2.0, 1.5, -1}
	s := NewSampler(0, 0, 0, 1)
	s.PresencePenalty = 1
	s.Accept(1)
	if got := s.Sample(v); got != 2 {
		t.Fatalf("with token 1 penalised, sampled %d; want 2", got)
	}
	if v[1] != 2.0 {
		t.Fatalf("the row was left penalised: %v", v)
	}
	for i := 0; i < PenaltyWindow; i++ {
		s.Accept(3)
	}
	if got := s.Sample(v); got != 1 {
		t.Fatalf("token 1 is out of the window, sampled %d; want 1", got)
	}

	r := NewSampler(0, 0, 0, 1)
	r.RepeatPenalty = 2
	r.Accept(1)
	if got := r.Sample(v); got != 2 { // 2.0/2 = 1.0 < 1.5
		t.Fatalf("with repeat penalty 2 on token 1, sampled %d; want 2", got)
	}
}

// TestSamplerMinP: a token below min_p times the best one never comes out.
func TestSamplerMinP(t *testing.T) {
	v := []float32{0, 0, 3} // probabilities ~0.045, 0.045, 0.91
	s := NewSampler(1, 0, 0, 5)
	s.MinP = 0.1
	for i := 0; i < 500; i++ {
		if id := s.Sample(v); id != 2 {
			t.Fatalf("sampled %d, below min_p", id)
		}
	}
}

// TestTopKOfIsTheSort: the one-pass selection (P20f) names the same ids in
// the same order as sorting the whole row, ties to the lower id, for the k a
// request asks for and on rows with many exact ties.
func TestTopKOfIsTheSort(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := range 40 {
		n := 1000 + rng.Intn(5000)
		row := make([]float32, n)
		for i := range row {
			row[i] = float32(rng.NormFloat64())
			if trial%2 == 1 {
				row[i] = float32(rng.Intn(50)) // ties everywhere
			}
		}
		all := make([]int32, n)
		for i := range all {
			all[i] = int32(i)
		}
		sort.SliceStable(all, func(a, b int) bool { return row[all[a]] > row[all[b]] })
		for _, k := range []int{1, 20, 40, 256} {
			got := topKOf(row, k, nil)
			if len(got) != k {
				t.Fatalf("k %d: %d ids", k, len(got))
			}
			for i := range got {
				if got[i] != all[i] {
					t.Fatalf("trial %d, k %d: id %d is %d (%v), the sort says %d (%v)", trial, k, i, got[i],
						row[got[i]], all[i], row[all[i]])
				}
			}
		}
	}
}

func BenchmarkSampleTopK20(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	row := make([]float32, 248320)
	for i := range row {
		row[i] = float32(rng.NormFloat64()) * 3
	}
	s := NewSampler(1, 20, 0.95, 1)
	b.ResetTimer()
	for range b.N {
		s.Sample(row)
	}
}

// TestSamplerVerifyIsTheDistribution (P20g): a row verified by speculative
// sampling is distributed as Sample's, whatever the draft's distribution,
// and keeps the draft with probability Σ min(p, q) — more often than an
// argmax draft is kept (p at q's argmax). Under top-k/top-p cuts and with
// none (the map path past 64 ids), and with a draft whose support differs.
func TestSamplerVerifyIsTheDistribution(t *testing.T) {
	cases := []struct {
		name string
		n    int
		topK int
		topP float32
		temp float32
	}{
		{"topk20-topp95", 400, 20, 0.95, 1},
		{"no-cuts", 200, 0, 0, 0.7},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(3))
			pl := make([]float32, c.n)
			ql := make([]float32, c.n)
			for i := range pl {
				pl[i] = float32(r.NormFloat64() * 2)
				// The draft: the trunk's row, noised.
				ql[i] = pl[i] + float32(r.NormFloat64()*0.8)
			}
			s := NewSampler(c.temp, c.topK, c.topP, 7)
			ds := s.DraftSampler(9)
			p := s.dist(pl)
			p = Dist{IDs: append([]int32(nil), p.IDs...), P: p.P}
			_, q := ds.Propose(ql)
			var keep float64
			for i, id := range p.IDs {
				keep += math.Min(float64(p.P[i]), float64(q.Of(id)))
			}
			argmaxKeep := float64(p.Of(Argmax(ql[:])))
			if p.Of(q.IDs[0]) > 0 {
				argmaxKeep = float64(p.Of(q.IDs[0]))
			}

			const draws = 200000
			count := map[int32]int{}
			kept := 0
			for i := 0; i < draws; i++ {
				d, q := ds.Propose(ql)
				g := s.Verify(pl, d, q)
				count[g]++
				if g == d {
					kept++
				}
			}
			for id := range count {
				if p.Of(id) == 0 {
					t.Fatalf("drew %d, which the cuts remove", id)
				}
			}
			var worst float64
			for i, id := range p.IDs {
				want := float64(p.P[i])
				got := float64(count[id]) / draws
				// Four standard errors.
				if tol := 4*math.Sqrt(want*(1-want)/draws) + 1e-4; math.Abs(got-want) > tol {
					t.Errorf("id %d: %.4f of the draws, p says %.4f", id, got, want)
				}
				worst = math.Max(worst, math.Abs(got-want))
			}
			got := float64(kept) / draws
			if math.Abs(got-keep) > 0.01 {
				t.Errorf("kept the draft %.4f of the time, Σ min(p, q) is %.4f", got, keep)
			}
			t.Logf("kept %.3f (Σ min(p,q) %.3f) against an argmax draft's %.3f; worst |Δp| %.4f",
				got, keep, argmaxKeep, worst)
		})
	}
}

// TestSamplerVerifyNilIsSample: without a draft distribution Verify is
// Sample, draw for draw — the seeded stream the server keeps for a client
// that named a seed.
func TestSamplerVerifyNilIsSample(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	row := make([]float32, 1000)
	a, b := NewSampler(1, 20, 0.95, 11), NewSampler(1, 20, 0.95, 11)
	for i := 0; i < 500; i++ {
		for j := range row {
			row[j] = float32(r.NormFloat64() * 3)
		}
		if x, y := a.Sample(row), b.Verify(row, 5, nil); x != y {
			t.Fatalf("draw %d: Sample %d, Verify %d", i, x, y)
		}
	}
}
