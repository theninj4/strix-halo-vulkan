package llm

// Sampling: one row of 248320 logits to one token (LLM.md L7c).
//
// The gate this vertical is held to is "it generates the same text as
// llama.cpp", and the only way that is a *check* rather than a impression is
// at temperature zero, where the sampler is an argmax and the text is a
// function of the weights alone. So greedy is the default here and the rest
// of the ladder exists because a model that can only be run greedily is not
// finished — not because any of it is on the critical path of the number.
//
// Everything below is on the host, on one row. llama.cpp does the same: its
// samplers are CPU code over the logits the graph returns, and at 248320
// floats a row it is 1 MB of arithmetic against a token's 6.3 GB of weights.
// Putting it on the device would be an optimisation of 0.02% of a step.

import (
	"math"
	"math/rand"
	"sort"
)

// Sampler is the chain llama.cpp calls `penalties -> top_k -> top_p -> min_p
// -> temp -> dist`, with temperature zero short-circuiting everything after
// the penalties into an argmax.
type Sampler struct {
	// Temp scales the logits before the softmax. Zero — the default — is
	// greedy, and is what a comparison against another implementation needs.
	Temp float32
	// TopK keeps the k highest logits, or every one of them at zero.
	TopK int
	// TopP keeps the smallest prefix of the sorted distribution whose mass
	// reaches p, or every token at zero or one.
	TopP float32
	// MinP drops every token whose probability is below MinP times the most
	// likely one's, after TopP. Zero keeps them all.
	MinP float32
	// RepeatPenalty and PresencePenalty are llama.cpp's penalties, over the
	// last PenaltyWindow tokens Accept was given: a token in the window has
	// a positive logit divided by RepeatPenalty (a negative one multiplied),
	// then PresencePenalty subtracted. They apply before the argmax too, so a
	// greedy run with a penalty is not a plain argmax. One and zero are off.
	RepeatPenalty   float32
	PresencePenalty float32
	history         []int32
	rng             *rand.Rand
}

// PenaltyWindow is how many trailing tokens the penalties look at. It is
// llama.cpp's default `penalty_last_n`, which is what the presets this server
// takes from llama-server were tuned against.
const PenaltyWindow = 64

// NewSampler builds one. A seed of zero is still deterministic — this is a
// reproducibility-first repo — and callers that want variety pass a clock.
func NewSampler(temp float32, topK int, topP float32, seed int64) *Sampler {
	return &Sampler{Temp: temp, TopK: topK, TopP: topP, RepeatPenalty: 1, rng: rand.New(rand.NewSource(seed))}
}

// Accept records tokens for the penalties' window: the prompt before the
// first Sample, as llama-server does, and each sampled token after it.
func (s *Sampler) Accept(ids ...int32) {
	if !s.penalised() {
		return
	}
	s.history = append(s.history, ids...)
	if n := len(s.history); n > PenaltyWindow {
		s.history = append(s.history[:0], s.history[n-PenaltyWindow:]...)
	}
}

func (s *Sampler) penalised() bool {
	return s != nil && ((s.RepeatPenalty != 1 && s.RepeatPenalty > 0) || s.PresencePenalty != 0)
}

// penalise applies the penalties to logits in place and returns a function
// that undoes it, so the caller's row is left as the graph wrote it.
func (s *Sampler) penalise(logits []float32) func() {
	if !s.penalised() || len(s.history) == 0 {
		return func() {}
	}
	saved := make(map[int32]float32, len(s.history))
	for _, id := range s.history {
		if _, done := saved[id]; done || int(id) >= len(logits) {
			continue
		}
		v := logits[id]
		saved[id] = v
		if s.RepeatPenalty > 0 && s.RepeatPenalty != 1 {
			if v > 0 {
				v /= s.RepeatPenalty
			} else {
				v *= s.RepeatPenalty
			}
		}
		logits[id] = v - s.PresencePenalty
	}
	return func() {
		for id, v := range saved {
			logits[id] = v
		}
	}
}

// Greedy reports whether this sampler is an argmax.
func (s *Sampler) Greedy() bool { return s == nil || s.Temp <= 0 }

// Sample picks a token from one row of logits.
func (s *Sampler) Sample(logits []float32) int32 {
	if s != nil {
		defer s.penalise(logits)()
	}
	if s.Greedy() {
		return Argmax(logits)
	}
	// The candidates, sorted by logit descending. A full sort of 248320 is
	// 30 ms and would be half a decode step, so the cut comes first: TopK
	// partitions, and TopP needs the order only over what survives it.
	idx := make([]int32, len(logits))
	for i := range idx {
		idx[i] = int32(i)
	}
	k := s.TopK
	if k <= 0 || k > len(idx) {
		k = len(idx)
	}
	if k < len(idx) {
		// Partial: the k largest, in no particular order, then sorted.
		quickselect(logits, idx, k)
		idx = idx[:k]
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })

	// Softmax over the survivors, at temperature.
	maxL := logits[idx[0]]
	probs := make([]float32, len(idx))
	var sum float32
	for i, id := range idx {
		p := float32(math.Exp(float64((logits[id] - maxL) / s.Temp)))
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	if s.TopP > 0 && s.TopP < 1 {
		var acc float32
		for i := range probs {
			acc += probs[i]
			if acc >= s.TopP {
				probs, idx = probs[:i+1], idx[:i+1]
				break
			}
		}
		var re float32
		for _, p := range probs {
			re += p
		}
		for i := range probs {
			probs[i] /= re
		}
	}

	if s.MinP > 0 && s.MinP < 1 {
		// probs is sorted descending, so the survivors are a prefix.
		cut := s.MinP * probs[0]
		n := 1
		for n < len(probs) && probs[n] >= cut {
			n++
		}
		probs, idx = probs[:n], idx[:n]
		var re float32
		for _, p := range probs {
			re += p
		}
		for i := range probs {
			probs[i] /= re
		}
	}

	r := s.rng.Float32()
	var acc float32
	for i, p := range probs {
		acc += p
		if r < acc {
			return idx[i]
		}
	}
	return idx[len(idx)-1]
}

// Argmax is the greedy sampler, and the only one a comparison against
// llama.cpp uses. Ties go to the lowest id, which is what `ggml_argmax` does.
func Argmax(logits []float32) int32 {
	best, bestV := int32(0), float32(math.Inf(-1))
	for i, v := range logits {
		if v > bestV {
			best, bestV = int32(i), v
		}
	}
	return best
}

// TopN is the n highest logits of a row, descending — what a caller prints
// beside a generated token, and what L6b's gate compared against llama.cpp's
// own ten.
func TopN(logits []float32, n int) []int32 {
	idx := make([]int32, len(logits))
	for i := range idx {
		idx[i] = int32(i)
	}
	if n > len(idx) {
		n = len(idx)
	}
	quickselect(logits, idx, n)
	idx = idx[:n]
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	return idx
}

// quickselect partitions idx so that its first k entries are the k largest by
// key, in no particular order. It is here rather than a sort because a decode
// step's logit row is 248320 wide and sorting it is longer than the forward
// pass that produced it.
func quickselect(key []float32, idx []int32, k int) {
	lo, hi := 0, len(idx)-1
	for lo < hi {
		// Median of three, so a row that arrives nearly sorted — which a
		// logit row does not, but a test's might — is not quadratic.
		mid := lo + (hi-lo)/2
		if key[idx[mid]] > key[idx[lo]] {
			idx[lo], idx[mid] = idx[mid], idx[lo]
		}
		if key[idx[hi]] > key[idx[lo]] {
			idx[lo], idx[hi] = idx[hi], idx[lo]
		}
		if key[idx[mid]] > key[idx[hi]] {
			idx[mid], idx[hi] = idx[hi], idx[mid]
		}
		pivot := key[idx[hi]]
		i := lo
		for j := lo; j < hi; j++ {
			if key[idx[j]] > pivot {
				idx[i], idx[j] = idx[j], idx[i]
				i++
			}
		}
		idx[i], idx[hi] = idx[hi], idx[i]
		switch {
		case i == k-1:
			return
		case i >= k:
			hi = i - 1
		default:
			lo = i + 1
		}
	}
}
