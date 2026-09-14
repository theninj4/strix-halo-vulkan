package dit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

// Config mirrors the transformer's config.json.
type Config struct {
	Dim       int     `json:"dim"`
	NHeads    int     `json:"n_heads"`
	NKVHeads  int     `json:"n_kv_heads"`
	NLayers   int     `json:"n_layers"`
	NRefiner  int     `json:"n_refiner_layers"`
	NormEps   float64 `json:"norm_eps"`
	QKNorm    bool    `json:"qk_norm"`
	AxesDims  [3]int  `json:"axes_dims"`
	AxesLens  [3]int  `json:"axes_lens"`
	RopeTheta float64 `json:"rope_theta"`
	CapFeat   int     `json:"cap_feat_dim"`
	InChan    int     `json:"in_channels"`
	PatchSize []int   `json:"all_patch_size"`
	TScale    float64 `json:"t_scale"`
}

// LoadConfig reads a transformer checkpoint's config.json.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("dit: parsing config.json: %w", err)
	}
	if c.Dim == 0 || c.NHeads == 0 {
		return nil, fmt.Errorf("dit: config.json is missing dim or n_heads")
	}
	return &c, nil
}

// qkNormEps is the epsilon diffusers hardcodes for the q/k RMS norms, which
// is separate from the config's norm_eps used by the block's four norms.
const qkNormEps = 1e-5

// loader pulls named tensors as float32, holding the first error.
type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
	}
	return v
}

func (l *loader) shape(name string) []int {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	return t.Shape
}

func (l *loader) linear(prefix string) *Linear {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 2 {
		l.err = fmt.Errorf("dit: %s.weight has shape %v, want 2 dims", prefix, sh)
		return nil
	}
	lin := &Linear{Out: sh[0], In: sh[1], Weight: l.f32(prefix + ".weight")}
	if l.set.Has(prefix + ".bias") {
		lin.Bias = l.f32(prefix + ".bias")
	}
	return lin
}

func (l *loader) rms(name string, eps float64) *RMSNorm {
	return &RMSNorm{Weight: l.f32(name + ".weight"), Eps: eps}
}

// LoadBlock reads one transformer block out of a checkpoint. prefix is the
// tensor-name prefix, e.g. "layers.0", "noise_refiner.1" or
// "context_refiner.0" -- the refiners are the same module, which is why the
// DiT has 34 blocks and not 30.
func LoadBlock(set *safetensors.Set, prefix string, cfg *Config) (*Block, error) {
	l := &loader{set: set}
	headDim := cfg.Dim / cfg.NHeads

	b := &Block{
		Dim:       cfg.Dim,
		AttnNorm1: l.rms(prefix+".attention_norm1", cfg.NormEps),
		AttnNorm2: l.rms(prefix+".attention_norm2", cfg.NormEps),
		FFNNorm1:  l.rms(prefix+".ffn_norm1", cfg.NormEps),
		FFNNorm2:  l.rms(prefix+".ffn_norm2", cfg.NormEps),
		Attn: &Attention{
			Q:       l.linear(prefix + ".attention.to_q"),
			K:       l.linear(prefix + ".attention.to_k"),
			V:       l.linear(prefix + ".attention.to_v"),
			Out:     l.linear(prefix + ".attention.to_out.0"),
			Heads:   cfg.NHeads,
			HeadDim: headDim,
		},
		FFN: &FeedForward{
			W1: l.linear(prefix + ".feed_forward.w1"),
			W2: l.linear(prefix + ".feed_forward.w2"),
			W3: l.linear(prefix + ".feed_forward.w3"),
		},
	}
	if cfg.QKNorm {
		b.Attn.NormQ = l.rms(prefix+".attention.norm_q", qkNormEps)
		b.Attn.NormK = l.rms(prefix+".attention.norm_k", qkNormEps)
	}
	// The refiner blocks carry modulation too; a block without it would have
	// no adaLN tensors at all.
	if set.Has(prefix + ".adaLN_modulation.0.weight") {
		b.AdaLN = l.linear(prefix + ".adaLN_modulation.0")
	}
	if l.err != nil {
		return nil, l.err
	}
	if b.AdaLN != nil && b.AdaLN.Out != 4*cfg.Dim {
		return nil, fmt.Errorf("dit: %s adaLN produces %d, want 4*dim = %d", prefix, b.AdaLN.Out, 4*cfg.Dim)
	}
	if b.Attn.Q.Out != cfg.NHeads*headDim {
		return nil, fmt.Errorf("dit: %s to_q produces %d, want heads*head_dim = %d",
			prefix, b.Attn.Q.Out, cfg.NHeads*headDim)
	}
	return b, nil
}
