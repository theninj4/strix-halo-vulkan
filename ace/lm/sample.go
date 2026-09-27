package lm

// Sampling as upstream's PyTorch path does it (llm_inference.py): the FSM's
// mask, then top-p on the *untempered* logits, then softmax(logits / T) and
// one draw. Our own RNG (decision 6): upstream's LM sampling is unseeded
// anyway.

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Sampling holds upstream's defaults: temperature 0.85, top-p 0.9, CFG 2.0
// in phase 2 only; repetition penalty 1 and top-k off.
type Sampling struct {
	Temperature float64
	TopP        float64
	CFG         float32
}

// DefaultSampling is the thinking path's.
var DefaultSampling = Sampling{Temperature: 0.85, TopP: 0.9, CFG: 2.0}

// tailCut is how far below the max a logit may be and still be sorted into
// the nucleus: e^-30 of the max's weight, so even 217,204 such tokens hold
// under 2e-8 of the mass.
const tailCut = 30

// topP is _apply_top_p_filter: sort descending, keep every token whose
// preceding cumulative probability is at most p (the first always), and
// mask the rest.
//
// The sums are exact (fp64). Torch's are fp32 with a vectorised reduction
// whose error puts the boundary a token either side: on one of the oracle's
// 21 steps its softmax denominator was 8.7e-6 low and it cut at 3,829 of
// 3,830, over a token of probability 3.6e-5. That is the reduction order of
// one CPU build, and upstream on CUDA has another; TestTopP allows exactly
// that, a boundary within 2e-5 of p.
func topP(scores []float32, p float64) {
	topPCum(scores, p)
}

// topPCum is topP returning the cumulative probability before the last
// token kept, for the test's boundary check.
func topPCum(scores []float32, p float64) float64 {
	if p <= 0 || p >= 1 {
		return 0
	}
	m := math.Inf(-1)
	for _, s := range scores {
		m = math.Max(m, float64(s))
	}
	if math.IsInf(m, -1) {
		return 0
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
	keep := make(map[int]bool, 64)
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

// draw is _sample_tokens: softmax(scores / T) and one multinomial draw, or
// the argmax at T ≤ 0.
func draw(scores []float32, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		return int(argmax(scores))
	}
	m := math.Inf(-1)
	for _, s := range scores {
		m = math.Max(m, float64(s))
	}
	var z float64
	for _, s := range scores {
		if !math.IsInf(float64(s), -1) {
			z += math.Exp((float64(s) - m) / temp)
		}
	}
	u := rng.Float64() * z
	last := -1
	for i, s := range scores {
		if math.IsInf(float64(s), -1) {
			continue
		}
		last = i
		u -= math.Exp((float64(s) - m) / temp)
		if u < 0 {
			return i
		}
	}
	return last
}

// codeScores is phase 2's step over the head's [ImEndID, CodeBase+NumCodes)
// rows: CFG over the 64,000 codes and EOS, EOS masked until target codes
// have been written and then the only choice. The result is indexed by
// code, with EOS at NumCodes.
func codeScores(cond, uncond []float32, cfg float32, written, target int) []float32 {
	out := make([]float32, NumCodes+1)
	for i := 0; i < NumCodes; i++ {
		c, u := cond[CodeBase-ImEndID+i], uncond[CodeBase-ImEndID+i]
		out[i] = u + cfg*(c-u)
	}
	c, u := cond[0], uncond[0]
	eos := u + cfg*(c-u)
	if written < target {
		out[NumCodes] = negInf
		return out
	}
	for i := range out {
		out[i] = negInf
	}
	out[NumCodes] = eos
	return out
}
