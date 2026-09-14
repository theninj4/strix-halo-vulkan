// Package pipeline drives Z-Image-Turbo end to end: a prompt in, a PNG out.
//
// The three model stages each have their own package and their own oracle --
// zimage/tokenizer and zimage/qwen for the text encoder, zimage/dit for the
// transformer, zimage/vae for the decoder -- and each was built and validated
// on its own. What is here is the thing that runs them in order, which is
// nothing any of them could contain: the noise schedule, the transformer's
// three phases over one stack of blocks, and the loop that turns eight
// forward passes into an image.
package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// FlowMatchEuler is Z-Image's scheduler: a flow-matching Euler integrator over
// a shifted sigma schedule.
//
// Flow matching makes this the simplest part of the pipeline -- one linear
// step per sigma, no noise re-injection, no predictor-corrector -- and the two
// things that are not obvious are both conventions:
//
//   - The schedule is not the scheduler's default. The pipeline hands it
//     sigmas of its own, linspace(1, 1/N, N), and the scheduler's `shift`
//     bends them towards the noisy end. The default would give a different
//     eight steps and a plausible, worse image.
//   - The timestep the transformer sees is not the schedule's. The pipeline
//     passes (1000 - timestep)/1000, i.e. 1-sigma, so the model's conditioning
//     runs the *other way* from the schedule it is integrating.
type FlowMatchEuler struct {
	// Sigmas has one entry per step plus a terminal zero, so a step reads
	// Sigmas[i] and Sigmas[i+1].
	Sigmas []float64
	// Timesteps is Sigmas[:N] in the model's training units.
	Timesteps []float64

	TrainTimesteps int
	Shift          float64
}

// SchedulerConfig mirrors the checkpoint's scheduler_config.json.
type SchedulerConfig struct {
	Class          string  `json:"_class_name"`
	TrainTimesteps int     `json:"num_train_timesteps"`
	Shift          float64 `json:"shift"`
	DynamicShift   bool    `json:"use_dynamic_shifting"`
}

// LoadSchedulerConfig reads a scheduler directory's config.
func LoadSchedulerConfig(dir string) (*SchedulerConfig, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "scheduler_config.json"))
	if err != nil {
		return nil, err
	}
	var c SchedulerConfig
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("pipeline: parsing scheduler_config.json: %w", err)
	}
	if c.Class != "" && c.Class != "FlowMatchEulerDiscreteScheduler" {
		return nil, fmt.Errorf("pipeline: scheduler is %s, this drives FlowMatchEulerDiscreteScheduler", c.Class)
	}
	if c.DynamicShift {
		return nil, fmt.Errorf("pipeline: use_dynamic_shifting is set; the shift would have to come from the image size")
	}
	if c.TrainTimesteps == 0 {
		c.TrainTimesteps = 1000
	}
	return &c, nil
}

// NewFlowMatchEuler builds the schedule for a number of steps.
func NewFlowMatchEuler(steps int, cfg *SchedulerConfig) (*FlowMatchEuler, error) {
	if steps < 1 {
		return nil, fmt.Errorf("pipeline: %d steps", steps)
	}
	s := &FlowMatchEuler{
		Sigmas:         make([]float64, steps+1),
		Timesteps:      make([]float64, steps),
		TrainTimesteps: cfg.TrainTimesteps,
		Shift:          cfg.Shift,
	}
	// The pipeline's own sigmas: linspace(1, 1/N, N), then the scheduler's
	// static shift. The terminal zero is the scheduler's, and is what makes
	// the last step land on a clean latent.
	lo := 1 / float64(steps)
	for i := 0; i < steps; i++ {
		raw := 1.0
		if steps > 1 {
			raw = 1 + (lo-1)*float64(i)/float64(steps-1)
		}
		sigma := cfg.Shift * raw / (1 + (cfg.Shift-1)*raw)
		s.Sigmas[i] = sigma
		s.Timesteps[i] = sigma * float64(cfg.TrainTimesteps)
	}
	s.Sigmas[steps] = 0
	return s, nil
}

// Steps is how many denoising steps the schedule holds.
func (s *FlowMatchEuler) Steps() int { return len(s.Timesteps) }

// ModelT is the conditioning value the transformer is given at step i, which
// is not the step's timestep: the pipeline sends (1000 - timestep)/1000.
func (s *FlowMatchEuler) ModelT(i int) float64 {
	return (float64(s.TrainTimesteps) - s.Timesteps[i]) / float64(s.TrainTimesteps)
}

// Step advances the latents by one Euler step, in place. out is the velocity
// the transformer predicted -- already negated by the caller, as the pipeline
// does -- and dt is negative, since the schedule runs down to zero.
func (s *FlowMatchEuler) Step(i int, latents, out []float32) error {
	if i < 0 || i >= len(s.Timesteps) {
		return fmt.Errorf("pipeline: step %d of %d", i, len(s.Timesteps))
	}
	if len(latents) != len(out) {
		return fmt.Errorf("pipeline: %d latents against %d predicted", len(latents), len(out))
	}
	dt := float32(s.Sigmas[i+1] - s.Sigmas[i])
	for j := range latents {
		latents[j] += dt * out[j]
	}
	return nil
}
