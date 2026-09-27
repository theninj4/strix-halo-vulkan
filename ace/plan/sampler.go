package plan

// The turbo sampler (AceStepConditionGenerationModel.generate_audio in
// upstream's acestep/models/xl_turbo): a fixed 8-entry schedule per shift,
// Euler steps between entries, a jump straight to x0 on the last one, and
// after every step the DCW correction, which is on by default for turbo.
//
// Arithmetic is fp32 where upstream's is (the latents, velocities and the
// dt/t broadcasts) and float64 where upstream's is Python floats (dt itself,
// the DCW scalers), so a run over the same velocities lands on the same bits.
// Every product is converted explicitly: torch rounds it before the sum,
// and the conversion keeps a compiler from fusing the two into an FMA.

import (
	"fmt"
	"math"
)

// ShiftTimesteps are upstream's SHIFT_TIMESTEPS: t = s·u/(1+(s−1)·u) on
// u = 1, 7/8, …, 1/8, written out as the literals upstream uses.
var ShiftTimesteps = map[float64][]float64{
	1: {1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125},
	2: {1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222},
	3: {1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3},
}

// ValidTimesteps are the 20 values a custom schedule is snapped to.
var ValidTimesteps = []float64{
	1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
	0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
	0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
	0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
}

// DefaultShift is the documented turbo value; upstream's GenerationParams
// default of 1.0 is for the base/SFT models (MUSIC.md decision 3).
const DefaultShift = 3.0

// nearest returns the first of vals closest to x, as Python's
// min(vals, key=abs(v − x)) does on ties.
func nearest(vals []float64, x float64) float64 {
	best := vals[0]
	for _, v := range vals[1:] {
		if math.Abs(v-x) < math.Abs(best-x) {
			best = v
		}
	}
	return best
}

// Schedule is the timestep list a request runs, as fp32 values (upstream
// builds `torch.tensor(list, dtype)` from it; the oracle's dtype is fp32).
// A custom list loses its trailing zeros, is cut to 20 and snapped to
// ValidTimesteps; otherwise the shift is snapped to 1, 2 or 3.
func Schedule(shift float64, custom []float64) []float32 {
	var list []float64
	if custom != nil {
		c := append([]float64(nil), custom...)
		for len(c) > 0 && c[len(c)-1] == 0 {
			c = c[:len(c)-1]
		}
		if len(c) > 20 {
			c = c[:20]
		}
		for _, t := range c {
			list = append(list, nearest(ValidTimesteps, t))
		}
	}
	if len(list) == 0 {
		list = ShiftTimesteps[nearest([]float64{1, 2, 3}, shift)]
	}
	out := make([]float32, len(list))
	for i, t := range list {
		out[i] = float32(t)
	}
	return out
}

// DCW is the Differential Correction in Wavelet domain settings; upstream's
// turbo default is Double with 0.05 and 0.02 on the Haar wavelet.
type DCW struct {
	Enabled    bool
	Low, High  float64
	ModeDouble bool
}

// DefaultDCW is what a turbo request runs with.
var DefaultDCW = DCW{Enabled: true, Low: 0.05, High: 0.02, ModeDouble: true}

// Velocity is the model: the velocity at latents x ([T·64], time-major) and
// timestep t.
type Velocity func(x []float32, t float32) ([]float32, error)

// Sample runs the schedule from noise and returns the final latents. step,
// if not nil, sees the latents after every step.
func Sample(noise []float32, sched []float32, dcw DCW, v Velocity, step func(i int, x []float32)) ([]float32, error) {
	if len(noise)%LatentChannels != 0 {
		return nil, fmt.Errorf("plan: %d latent values is not a whole number of %d-channel frames", len(noise), LatentChannels)
	}
	x := append([]float32(nil), noise...)
	for i, t := range sched {
		vt, err := v(x, t)
		if err != nil {
			return nil, err
		}
		if len(vt) != len(x) {
			return nil, fmt.Errorf("plan: velocity has %d values, want %d", len(vt), len(x))
		}
		before := x
		next := make([]float32, len(x))
		if i == len(sched)-1 {
			// get_x0_from_noise: x − v·t.
			for j := range next {
				next[j] = before[j] - float32(vt[j]*t)
			}
		} else {
			// dt is a Python float from the fp32 entries, then broadcast
			// back to fp32.
			dt := float32(float64(t) - float64(sched[i+1]))
			for j := range next {
				next[j] = before[j] - float32(vt[j]*dt)
			}
		}
		if dcw.Enabled {
			denoised := make([]float32, len(x))
			for j := range denoised {
				denoised[j] = before[j] - float32(vt[j]*t)
			}
			next = dcw.apply(next, denoised, float64(t))
		}
		x = next
		if step != nil {
			step(i, x)
		}
	}
	return x, nil
}

// haar is pytorch_wavelets' haar filter tap: pywt's float64
// 0.7071067811865476, narrowed to the fp32 filter tensor.
var haar = float32(0.7071067811865476)

// apply is DCWCorrector.apply in "double" mode: a one-level zero-padded Haar
// DWT along time of x and y (each channel separately), the low band pushed
// by t·Low·(xL − yL) and the high by (1−t)·High·(xH − yH), and the inverse
// transform cut back to x's length. An odd length gets one zero appended
// before the analysis, as pytorch_wavelets pads for mode "zero".
func (d DCW) apply(x, y []float32, t float64) []float32 {
	low := t * d.Low
	high := (1 - t) * d.High
	if !d.ModeDouble {
		high = 0
	}
	if low == 0 && high == 0 {
		return x
	}
	T := len(x) / LatentChannels
	half := (T + 1) / 2
	at := func(a []float32, i, c int) float32 {
		if i >= T {
			return 0
		}
		return a[i*LatentChannels+c]
	}
	out := make([]float32, len(x))
	ls, hs := float32(low), float32(high)
	for c := 0; c < LatentChannels; c++ {
		for k := 0; k < half; k++ {
			x0, x1 := at(x, 2*k, c), at(x, 2*k+1, c)
			y0, y1 := at(y, 2*k, c), at(y, 2*k+1, c)
			// conv with the reversed analysis filters h0 = [c, c] and
			// h1 = [c, −c]. oneDNN's direct conv rounds the first tap's
			// product and fuses the second into it: fma(h[1], x1, h[0]·x0).
			// (Emulated through float64, the double rounding has not been
			// seen to differ.)
			xl := fma32(haar, x1, float32(haar*x0))
			xh := fma32(-haar, x1, float32(haar*x0))
			if low != 0 {
				yl := fma32(haar, y1, float32(haar*y0))
				xl = xl + float32(ls*(xl-yl))
			}
			if high != 0 {
				yh := fma32(-haar, y1, float32(haar*y0))
				xh = xh + float32(hs*(xh-yh))
			}
			// conv_transpose with rec_lo = [c, c] and rec_hi = [c, −c]: one
			// product per output each, the two bands summed after.
			if 2*k < T {
				out[2*k*LatentChannels+c] = float32(haar*xl) + float32(haar*xh)
			}
			if 2*k+1 < T {
				out[(2*k+1)*LatentChannels+c] = float32(haar*xl) - float32(haar*xh)
			}
		}
	}
	return out
}

// fma32 is a·b + c with one rounding, as the fp32 FMA upstream's conv uses.
func fma32(a, b, c float32) float32 {
	return float32(math.FMA(float64(a), float64(b), float64(c)))
}
