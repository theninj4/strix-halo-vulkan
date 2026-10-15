package plan

// The base model's sampler (AceStepConditionGenerationModel.generate_audio
// in upstream's acestep/models/xl_base; research/music-vertical.md A15): a
// linspace schedule bent by the shift, Euler between every pair of entries
// (the last pair ends at 0, so the last step lands on x0 as turbo's jump
// does), and classifier-free guidance through APG on a conditional and a
// null-condition velocity. DCW runs after each step only when asked; for
// base it is off by default.
//
// As in sampler.go, every fp32 product is rounded before the sum it feeds,
// and float64 is used where upstream's arithmetic is float64 (APG's
// projection). The reductions follow torch's CPU kernels: a norm sums its
// squares in order, a sum cascades (cascadeSum). A run over the same
// velocities lands on upstream's bits (TestBaseSampler).

import (
	"fmt"
	"math"
)

// BaseSteps and BaseGuidance are what upstream's Gradio UI sets for a pure
// base model; its API's own default of 8 steps is turbo's (A15).
const (
	BaseSteps    = 32
	BaseGuidance = 7.0
)

// BaseSchedule is torch.linspace(1, 0, steps+1) in fp32, then
// shift·t/(1+(shift−1)·t) when the shift is not 1, as fp32 tensor ops: steps
// + 1 entries, the last 0.
func BaseSchedule(steps int, shift float64) []float32 {
	n := steps + 1
	out := make([]float32, n)
	if n == 1 {
		out[0] = 1
		return out
	}
	// at::linspace's CPU kernel: step = (end − start)/(steps − 1) in the
	// scalar type, the first half counted up from start and the second down
	// from end, each entry one fused multiply-add (its vectorised loop).
	step := float32(-1) / float32(n-1)
	half := n / 2
	for i := range out {
		if i < half {
			out[i] = fma32(step, float32(i), 1)
		} else {
			out[i] = fma32(-step, float32(n-i-1), 0)
		}
	}
	if shift != 1 {
		s, s1 := float32(shift), float32(shift-1)
		for i, t := range out {
			out[i] = float32(s*t) / (1 + float32(s1*t))
		}
	}
	return out
}

// Guidance is a base request's CFG: the scale (≤ 1 runs the conditional row
// alone), the interval of timesteps it applies over (upstream compares the
// fp32 t with both ends, inclusive), and DCW.
type Guidance struct {
	Scale              float64
	IntervalStart, End float64
	DCW                DCW
}

// DefaultGuidance is a base request's: scale 7 over every step, no DCW.
var DefaultGuidance = Guidance{Scale: BaseGuidance, IntervalStart: 0, End: 1}

// on reports whether CFG applies at timestep t.
func (g Guidance) on(t float32) bool {
	return g.Scale > 1 && t >= float32(g.IntervalStart) && t <= float32(g.End)
}

// PairVelocity is the model for a guided run: the conditional velocity at
// latents x and timestep t, and, when uncond is set, the null-condition
// velocity as well (nil otherwise). Upstream runs the null row at every step
// once the scale is over 1 and discards it outside the interval; skipping it
// there changes nothing.
type PairVelocity func(x []float32, t float32, uncond bool) (cond, null []float32, err error)

// SampleGuided runs a base schedule (BaseSchedule's steps + 1 entries) from
// x0 and returns the final latents. step, if not nil, sees the latents after
// every step.
func SampleGuided(x0 []float32, sched []float32, g Guidance, v PairVelocity, step func(i int, x []float32)) ([]float32, error) {
	if len(x0)%LatentChannels != 0 {
		return nil, fmt.Errorf("plan: %d latent values is not a whole number of %d-channel frames", len(x0), LatentChannels)
	}
	if len(sched) < 2 {
		return nil, fmt.Errorf("plan: a schedule of %d entries has no step", len(sched))
	}
	x := append([]float32(nil), x0...)
	var apg apgState
	for i := 0; i+1 < len(sched); i++ {
		t, tn := sched[i], sched[i+1]
		guided := g.on(t)
		cond, null, err := v(x, t, guided)
		if err != nil {
			return nil, err
		}
		if len(cond) != len(x) || (guided && len(null) != len(x)) {
			return nil, fmt.Errorf("plan: velocity has %d values (null %d), want %d", len(cond), len(null), len(x))
		}
		vt := cond
		if guided {
			vt = apg.forward(cond, null, g.Scale)
		}
		// dt is the difference of two fp32 entries, an fp32 tensor.
		dt := t - tn
		next := make([]float32, len(x))
		for j := range next {
			next[j] = x[j] - float32(vt[j]*dt)
		}
		if g.DCW.Enabled {
			denoised := make([]float32, len(x))
			for j := range denoised {
				denoised[j] = x[j] - float32(vt[j]*t)
			}
			next = g.DCW.apply(next, denoised, float64(t))
		}
		x = next
		if step != nil {
			step(i, x)
		}
	}
	return x, nil
}

// apgState is APG's MomentumBuffer (momentum −0.75), carried across the
// steps CFG applies at.
type apgState struct {
	avg []float32
}

// apgNorm is apg_forward's norm_threshold.
const apgNorm = 2.5

// forward is apg_forward with dims=[1], the time axis of [1, T, 64]: the
// difference through the momentum buffer, its per-channel norm over time
// clamped to 2.5, its component along cond projected out (float64), and
// cond + (scale − 1)·the rest.
func (s *apgState) forward(cond, null []float32, scale float64) []float32 {
	const C = LatentChannels
	n := len(cond)
	T := n / C
	diff := make([]float32, n)
	for i := range diff {
		diff[i] = cond[i] - null[i]
	}
	// MomentumBuffer.update: running_average starts as the int 0, and
	// diff + (−0.75·0) is diff.
	if s.avg != nil {
		for i := range diff {
			diff[i] += float32(-0.75 * float64(s.avg[i]))
		}
	}
	s.avg = diff
	// diff.norm(p=2, dim=1): squares summed in time order in fp32.
	var sq [C]float32
	for r := 0; r < T; r++ {
		for c := 0; c < C; c++ {
			d := diff[r*C+c]
			sq[c] += float32(d * d)
		}
	}
	// minimum(1, 2.5 / norm), where a scalar over a tensor is
	// reciprocal() · scalar: two roundings.
	var sf [C]float32
	for c := range sf {
		nrm := float32(math.Sqrt(float64(sq[c])))
		sf[c] = min(1, float32(float32(1/float64(nrm))*apgNorm))
	}
	// project(v0 = diff·sf, v1 = cond) in float64: v1 normalised over time
	// (its norm summed in order, clamped at 1e-12), the dot product summed
	// as torch's cascade, v0 − dot·v1.
	v0 := make([]float64, n)
	v1 := make([]float64, n)
	var n1 [C]float64
	for r := 0; r < T; r++ {
		for c := 0; c < C; c++ {
			i := r*C + c
			v0[i] = float64(diff[i] * sf[c])
			v1[i] = float64(cond[i])
			n1[c] += float64(v1[i] * v1[i])
		}
	}
	for c := range n1 {
		n1[c] = max(math.Sqrt(n1[c]), 1e-12)
	}
	prod := make([]float64, n)
	for i := range v1 {
		v1[i] /= n1[i%C]
		prod[i] = float64(v0[i] * v1[i])
	}
	dot := cascadeSum(prod, T, C)
	w := float32(scale - 1)
	out := make([]float32, n)
	for i := range out {
		orth := float32(v0[i] - float64(dot[i%C]*v1[i]))
		out[i] = cond[i] + float32(w*orth)
	}
	return out
}

// cascadeSum sums a [rows, cols] row-major array over its rows as torch's
// CPU sum does when the reduced dimension is not the innermost
// (SumKernel.cpp's multi_row_sum): four levels of accumulators, the first
// taking 2^max(4, ⌈log2 rows⌉/4) rows before it is folded into the next,
// and so on up while the row count is a multiple of each level's span.
func cascadeSum(a []float64, rows, cols int) []float64 {
	const levels = 4
	power := max(4, ceilLog2(rows)/levels)
	stepRows := 1 << power
	mask := stepRows - 1
	acc := make([][]float64, levels)
	for l := range acc {
		acc[l] = make([]float64, cols)
	}
	i := 0
	for i+stepRows <= rows {
		for j := 0; j < stepRows; j, i = j+1, i+1 {
			for c := 0; c < cols; c++ {
				acc[0][c] += a[i*cols+c]
			}
		}
		for l := 1; l < levels; l++ {
			for c := 0; c < cols; c++ {
				acc[l][c] += acc[l-1][c]
				acc[l-1][c] = 0
			}
			if i&(mask<<(l*power)) != 0 {
				break
			}
		}
	}
	for ; i < rows; i++ {
		for c := 0; c < cols; c++ {
			acc[0][c] += a[i*cols+c]
		}
	}
	for l := 1; l < levels; l++ {
		for c := 0; c < cols; c++ {
			acc[0][c] += acc[l][c]
		}
	}
	return acc[0]
}

// ceilLog2 is c10's CeilLog2: 0 for 0 and 1.
func ceilLog2(x int) int {
	if x <= 1 {
		return 0
	}
	n := 0
	for v := x - 1; v > 0; v >>= 1 {
		n++
	}
	return n
}
