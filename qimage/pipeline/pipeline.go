package pipeline

import (
	"fmt"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/zimage/qwen"
)

// Denoise walks the sampler over the target latents: the first step runs the
// block-causal prefill and extracts every block's prefix K/V, the remaining
// steps decode only the target rows against the cache — the shipped
// configuration (IMAGE.md decision 2; cache-off would sample a *different
// valid image*, so there is exactly one oracle and this is it).
//
// txt is the raw VLM embedding (one row per VLM token), cond the packed
// condition-image latents (nil for t2i), noise the packed target latents the
// caller seeded, and the returned Mat the final target latents — still in
// the normalised space the VAE's per-channel denormalisation undoes.
// progress, if non-nil, sees the latents after every step.
func Denoise(m *dit.Model, txt *qwen.Mat, lay *dit.Layout, cond, noise *qwen.Mat, sched *Schedule, progress func(step int, latents *qwen.Mat)) (*qwen.Mat, error) {
	target := lay.ImgShapes[len(lay.ImgShapes)-1]
	targetTokens := target[0] * target[1] * target[2]
	if noise.Rows != targetTokens {
		return nil, fmt.Errorf("pipeline: %d noise rows for a %d-token target", noise.Rows, targetTokens)
	}
	latents := noise.Clone()
	cache := dit.NewCache(len(m.Blocks))

	for i := 0; i < sched.Steps(); i++ {
		input := latents
		if cond != nil {
			input = qwen.NewMat(cond.Rows+latents.Rows, latents.Cols)
			copy(input.Data, cond.Data)
			copy(input.Data[cond.Rows*cond.Cols:], latents.Data)
		}
		mode := dit.ModeCached
		if i == 0 {
			mode = dit.ModeExtract
		}
		out, err := m.Forward(input, txt, lay, sched.T(i), mode, cache)
		if err != nil {
			return nil, fmt.Errorf("pipeline: step %d: %w", i, err)
		}
		// The prefill returns the whole joint sequence; the model's rows for
		// the prefix are discarded and only the target's prediction steps.
		if out.Rows != targetTokens {
			out = &qwen.Mat{
				Rows: targetTokens, Cols: out.Cols,
				Data: out.Data[(out.Rows-targetTokens)*out.Cols:],
			}
		}
		if err := sched.Step(i, latents.Data, out.Data); err != nil {
			return nil, err
		}
		if progress != nil {
			progress(i, latents)
		}
	}
	return latents, nil
}
