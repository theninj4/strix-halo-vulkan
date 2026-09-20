// Package pipeline runs Qwen-Image-2.1's denoising loop on the CPU: the
// dynamic-shift flow-matching scheduler and the sampler that walks the DiT
// with the prefix KV cache — the oracle the served pipeline (Q6) is built
// from, validated against reference/out/qi21sched and the end-to-end runs.
package pipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// SchedConfig mirrors scheduler/scheduler_config.json. Unlike Z-Image's
// fixed shift 3.0, 2.1 shifts *dynamically*: mu is linear in the target
// token count between (BaseImageSeqLen, BaseShift) and (MaxImageSeqLen,
// MaxShift) — and extrapolates unclamped past the maximum, which 2048²
// (16384 tokens, mu 1.3129) relies on.
type SchedConfig struct {
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	BaseImageSeqLen   int     `json:"base_image_seq_len"`
	MaxImageSeqLen    int     `json:"max_image_seq_len"`
	BaseShift         float64 `json:"base_shift"`
	MaxShift          float64 `json:"max_shift"`
	ShiftTerminal     float64 `json:"shift_terminal"`
	TimeShiftType     string  `json:"time_shift_type"`
	UseDynamicShift   bool    `json:"use_dynamic_shifting"`
	Stochastic        bool    `json:"stochastic_sampling"`
	InvertSigmas      bool    `json:"invert_sigmas"`
}

// LoadSchedConfig reads the checkpoint's scheduler config and refuses the
// variants this port does not implement, rather than silently sampling a
// different schedule.
func LoadSchedConfig(dir string) (*SchedConfig, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "scheduler_config.json"))
	if err != nil {
		return nil, err
	}
	var c SchedConfig
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("pipeline: parsing scheduler_config.json: %w", err)
	}
	if !c.UseDynamicShift || c.TimeShiftType != "exponential" {
		return nil, fmt.Errorf("pipeline: scheduler wants dynamic exponential shifting, config says dynamic=%v type=%q",
			c.UseDynamicShift, c.TimeShiftType)
	}
	if c.Stochastic || c.InvertSigmas {
		return nil, fmt.Errorf("pipeline: stochastic_sampling/invert_sigmas are set and unimplemented")
	}
	if c.NumTrainTimesteps == 0 {
		return nil, fmt.Errorf("pipeline: num_train_timesteps is missing")
	}
	return &c, nil
}

// Mu is the dynamic shift for a target image of the given token count —
// linear between the config's two anchor points, deliberately unclamped.
func (c *SchedConfig) Mu(imageTokens int) float64 {
	m := (c.MaxShift - c.BaseShift) / float64(c.MaxImageSeqLen-c.BaseImageSeqLen)
	b := c.BaseShift - m*float64(c.BaseImageSeqLen)
	return float64(imageTokens)*m + b
}

// Schedule is one sampling run's noise levels: N+1 sigmas (terminal zero
// appended) and N timesteps (sigma times the training count). The DiT's `t`
// input is Timesteps[i]/1000, which is Sigmas[i].
type Schedule struct {
	Sigmas    []float32
	Timesteps []float32
}

// Timesteps builds the schedule for a step count and target token count:
// sigmas linspace(1, 1/N, N), exponentially time-shifted by mu, then
// stretched so the last live sigma is exactly ShiftTerminal.
//
// The arithmetic is float64 until the final narrowing — diffusers holds
// float32 tensors but computes the shift elementwise in the same order, and
// the dump comparison is what pins the two within float32 noise.
func (c *SchedConfig) Timesteps(steps, imageTokens int) (*Schedule, error) {
	if steps < 1 {
		return nil, fmt.Errorf("pipeline: %d steps", steps)
	}
	mu := c.Mu(imageTokens)
	sigmas := make([]float64, steps)
	for i := range sigmas {
		// np.linspace(1, 1/N, N)
		if steps == 1 {
			sigmas[i] = 1
			continue
		}
		sigmas[i] = 1 + float64(i)*((1/float64(steps))-1)/float64(steps-1)
	}
	// The exponential time shift: sigma' = e^mu / (e^mu + (1/sigma - 1)).
	emu := math.Exp(mu)
	for i, s := range sigmas {
		sigmas[i] = emu / (emu + (1/s - 1))
	}
	// Stretch so the last live sigma lands on ShiftTerminal exactly.
	if c.ShiftTerminal != 0 {
		scale := (1 - sigmas[len(sigmas)-1]) / (1 - c.ShiftTerminal)
		for i, s := range sigmas {
			sigmas[i] = 1 - (1-s)/scale
		}
	}
	sched := &Schedule{
		Sigmas:    make([]float32, steps+1),
		Timesteps: make([]float32, steps),
	}
	for i, s := range sigmas {
		sched.Sigmas[i] = float32(s)
		sched.Timesteps[i] = float32(s * float64(c.NumTrainTimesteps))
	}
	return sched, nil
}

// Steps is the live step count.
func (s *Schedule) Steps() int { return len(s.Timesteps) }

// T is the DiT's timestep input at step i: sigma, i.e. timestep/1000.
func (s *Schedule) T(i int) float64 { return float64(s.Timesteps[i]) / 1000 }

// Step advances the sample in place: x += (sigma_next - sigma) * modelOut,
// the plain Euler move over the flow. The difference is formed in float32,
// as diffusers' float32 sigma tensors form it.
func (s *Schedule) Step(i int, sample, modelOut []float32) error {
	if i < 0 || i >= s.Steps() {
		return fmt.Errorf("pipeline: step %d of %d", i, s.Steps())
	}
	if len(sample) != len(modelOut) {
		return fmt.Errorf("pipeline: %d sample values against %d model outputs", len(sample), len(modelOut))
	}
	dt := s.Sigmas[i+1] - s.Sigmas[i]
	for j := range sample {
		sample[j] += dt * modelOut[j]
	}
	return nil
}
