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
