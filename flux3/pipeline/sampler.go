// Package pipeline is FLUX 3 Image Klein's sampler (X3,
// research/flux3-vertical.md): the flow-map schedule, the seeded noise and
// the Euler loop around flux3/dit.
//
// The schedule is diffusers' FlowMatchEulerDiscreteScheduler as the
// Flux3 modular pipeline configures it (constant shift 6.93, sigmas
// linspace(1, 1/N, N), a 0 appended) and BFL's sampling.timesteps formula
// α s / (1 + (α−1) s) — the same points, which X0 measured to 1.2e-7. It is
// computed here the way numpy computes it, in float32 after a float64
// linspace, so the times the model sees and the step sizes are bit for bit
// the oracle's; every float32 product is an explicit conversion so the
// compiler cannot fuse it into an FMA and round differently.
package pipeline

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"strix-halo-vulkan/flux3/dit"
	"strix-halo-vulkan/zimage/qwen"
)

const (
	// Alpha is the schedule's constant shift (config.json sampler.alpha).
	Alpha = 6.93
	// DefaultSteps is the distilled checkpoint's own step count.
	DefaultSteps = 16
)

// Schedule holds steps+1 sigmas, the last 0.
type Schedule struct {
	Sigmas []float32
}

// NewSchedule builds the shifted schedule for a step count.
func NewSchedule(steps int, alpha float64) (*Schedule, error) {
	if steps < 1 {
		return nil, fmt.Errorf("pipeline: %d steps", steps)
	}
	a, b := float32(alpha), float32(alpha-1) // numpy's weak scalars, cast to the array's float32
	s := &Schedule{Sigmas: make([]float32, steps+1)}
	for i := 0; i < steps; i++ {
		// np.linspace(1, 1/N, N) in float64, then astype(float32)
		x := 1.0
		if steps > 1 {
			step := (1/float64(steps) - 1) / float64(steps-1)
			x = float64(float64(i)*step) + 1 // arange*step + start, unfused
		}
		if i == steps-1 {
			x = 1 / float64(steps)
		}
		f := float32(x)
		num := float32(a * f)
		den := float32(1 + float32(b*f))
		s.Sigmas[i] = float32(num / den)
	}
	return s, nil
}

// Steps is the number of model calls.
func (s *Schedule) Steps() int { return len(s.Sigmas) - 1 }

// T is the time the model is handed at step i (and, at i+1, the time step i
// jumps to): diffusers passes timesteps/1000 where timesteps = sigmas*1000,
// both float32, so it is σ through *1000 and /1000 — not quite σ.
func (s *Schedule) T(i int) float32 {
	if i >= s.Steps() {
		return 0
	}
	return float32(float32(s.Sigmas[i]*1000) / 1000)
}

// Step is the Euler update x += (σ_{i+1} − σ_i) v in float32, in place.
func (s *Schedule) Step(x, v *qwen.Mat, i int) error {
	if x.Rows != v.Rows || x.Cols != v.Cols {
		return fmt.Errorf("pipeline: step %s by %s", x, v)
	}
	dt := float32(s.Sigmas[i+1] - s.Sigmas[i])
	for j, vv := range v.Data {
		x.Data[j] = float32(x.Data[j] + float32(dt*vv))
	}
	return nil
}

// Noise is the initial latent: standard normal draws from a seeded Go
// generator (this repo's reproducibility, not torch's — decision 7), in the
// packed [tokens, channels] order, rounded to bf16 values as BFL's
// Pipeline.generate does (diffusers keeps fp32; X0 finding 7).
func Noise(tokens, channels int, seed int64) *qwen.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := qwen.NewMat(tokens, channels)
	for i := range m.Data {
		m.Data[i] = roundBF16(float32(rng.NormFloat64()))
	}
	return m
}

// roundBF16 rounds to the nearest bf16 value, ties to even.
func roundBF16(f float32) float32 {
	u := math.Float32bits(f)
	u += 0x7fff + (u>>16)&1
	return math.Float32frombits(u &^ 0xffff)
}

// Denoise runs the schedule from in.Img (the noise) through the CPU model
// and returns the final latent. in's T and TNext are overwritten each step;
// progress, if not nil, sees each step's result.
func Denoise(ctx context.Context, m *dit.Model, in *dit.Input, s *Schedule, progress func(step int, x *qwen.Mat)) (*qwen.Mat, error) {
	step := *in
	return DenoiseWith(ctx, in.Img, s, func(x *qwen.Mat, t, tNext float64) (*qwen.Mat, error) {
		step.Img, step.T, step.TNext = x, t, tNext
		return m.Forward(&step)
	}, progress)
}

// Velocity is one model call: the velocity of x at time t in a step that
// jumps to tNext. The CPU model and dit.GPU's Forward both are one.
type Velocity func(x *qwen.Mat, t, tNext float64) (*qwen.Mat, error)

// DenoiseWith runs the schedule from noise with any model. The context is
// checked between steps.
func DenoiseWith(ctx context.Context, noise *qwen.Mat, s *Schedule, model Velocity, progress func(step int, x *qwen.Mat)) (*qwen.Mat, error) {
	x := noise.Clone()
	for i := 0; i < s.Steps(); i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, err := model(x, float64(s.T(i)), float64(s.T(i+1)))
		if err != nil {
			return nil, fmt.Errorf("pipeline: step %d: %w", i, err)
		}
		if err := s.Step(x, v, i); err != nil {
			return nil, err
		}
		if progress != nil {
			progress(i, x)
		}
	}
	return x, nil
}
