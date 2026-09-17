package embed

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// LoadConfig reads the checkpoint's config.json as a qwen.Config, with the
// two differences this checkpoint has from Z-Image's text encoder written
// down rather than discovered at the first missing tensor.
//
// The names carry no `model.` prefix: `model.safetensors` here is a
// `Qwen3Model` export (sentence-transformers' `0_Transformer`), so it is
// `layers.0.self_attn.q_proj.weight` and `norm.weight`, and there is no
// `lm_head` at all -- `tie_word_embeddings` is true and nothing here
// generates.
func LoadConfig(dir string) (*qwen.Config, error) {
	cfg, err := qwen.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	cfg.Prefix = ""
	return cfg, nil
}

// LoadFinalNorm reads `norm.weight`, the RMS norm between the last decoder
// layer and the pooled vector.
//
// The text encoder next door never loads this tensor, because
// `hidden_states[-2]` is taken before it runs. An embedding model's output
// *is* its output: the pre-norm and post-norm tensors differ by 775 in
// absolute terms on the reference prompt, so this is not a rounding-level
// mistake to make.
func LoadFinalNorm(set *safetensors.Set, cfg *qwen.Config) (*qwen.RMSNorm, error) {
	name := cfg.Prefix + "norm.weight"
	t, err := set.Get(name)
	if err != nil {
		return nil, err
	}
	w, err := t.F32(nil)
	if err != nil {
		return nil, err
	}
	if len(w) != cfg.HiddenSize {
		return nil, fmt.Errorf("embed: %s is %d wide, want %d", name, len(w), cfg.HiddenSize)
	}
	return &qwen.RMSNorm{Weight: w, Eps: cfg.RMSEps}, nil
}
