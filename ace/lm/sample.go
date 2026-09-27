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
//
// A token is kept when the mass sorted ahead of it is at most p, so only
// the order near the boundary matters (MUSIC.md A10: a full sort of a flat
// 64,000-code step was 8 of its 10 ms). The candidates are bucketed by
// distance below the max; whole buckets are kept while their mass stays
// under p, and only the bucket that crosses it is sorted. Equal scores share
// a bucket, so the order and the tie-break are the sort's. What differs is
// the order of the fp64 sums, at the 1e-16 level.
func topPCum(scores []float32, p float64) float64 {
	if p <= 0 || p >= 1 {
		return 0
	}
	m := maxScore(scores)
	if math.IsInf(m, -1) {
		return 0
	}
	const nb = 2048
	var mass [nb]float64
	bucket := make([]int16, len(scores)) // -1: not a candidate
	var z float64
	for i, s := range scores {
		bucket[i] = -1
		if math.IsInf(float64(s), -1) {
			continue
		}
		e := math.Exp(float64(s) - m)
		z += e
		if d := m - float64(s); d < tailCut {
			b := int16(min(int(d*(nb/tailCut)), nb-1))
			bucket[i] = b
			mass[b] += e
		}
	}
	keep := make([]bool, len(scores))
	kept := 0
	cum, before := 0.0, 0.0
	b := 0
	// Whole buckets while the mass through them stays at most p: every
	// token in one has at most that much ahead of it.
	for ; b < nb && cum+mass[b]/z <= p; b++ {
		cum += mass[b] / z
	}
	// Every token in a bucket before b is kept; before is the mass ahead of
	// the last of them, which the sorted walk below overwrites.
	lastScore, last := math.Inf(1), -1
	for i, bi := range bucket {
		if bi >= 0 && int(bi) < b {
			keep[i] = true
			kept++
			if s := float64(scores[i]); s < lastScore || (s == lastScore && i > last) {
				lastScore, last = s, i
			}
		}
	}
	if last >= 0 {
		before = cum - math.Exp(lastScore-m)/z
	}
	// Then token by token, in sorted order, from the bucket that crosses.
	// The candidates from there on are binned once (a counting sort), so a
	// bucket costs its own tokens: a peaked phase-1 step walks through many
	// empty ones.
	var start [nb + 1]int
	for _, bi := range bucket {
		if int(bi) >= b {
			start[bi+1]++
		}
	}
	for j := b; j < nb; j++ {
		start[j+1] += start[j]
	}
	order := make([]int, start[nb])
	fill := start
	for i, bi := range bucket {
		if int(bi) >= b {
			order[fill[bi]] = i
			fill[bi]++
		}
	}
walk:
	for ; b < nb; b++ {
		cand := order[start[b]:start[b+1]]
		sort.Slice(cand, func(x, y int) bool {
			if scores[cand[x]] != scores[cand[y]] {
				return scores[cand[x]] > scores[cand[y]]
			}
			return cand[x] < cand[y]
		})
		for _, i := range cand {
			if kept > 0 && cum > p {
				break walk
			}
			keep[i] = true
			kept++
			before = cum
			cum += math.Exp(float64(scores[i])-m) / z
		}
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
	m := maxScore(scores)
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

// maxScore is the largest score (-inf for none). A plain comparison:
// math.Max's NaN handling was half of a phase-1 step's host time.
func maxScore(scores []float32) float64 {
	m := negInf
	for _, s := range scores {
		if s > m {
			m = s
		}
	}
	return float64(m)
}
