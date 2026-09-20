package dit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// Config mirrors transformer/config.json.
type Config struct {
	PatchSize    int     `json:"patch_size"`
	InChannels   int     `json:"in_channels"`
	OutChannels  int     `json:"out_channels"`
	NumLayers    int     `json:"num_layers"`
	HeadDim      int     `json:"attention_head_dim"`
	NumHeads     int     `json:"num_attention_heads"`
	ContextIn    int     `json:"context_in_dim"`
	MLPRatio     int     `json:"mlp_ratio"`
	AxesDimsRope []int   `json:"axes_dims_rope"`
	Eps          float64 `json:"eps"`
	CausalCond   bool    `json:"causal_condition"`

	AxesDims [3]int `json:"-"` // AxesDimsRope as the fixed-size array NewRope takes
}

// Dim is the model width, heads times head_dim.
func (c *Config) Dim() int { return c.NumHeads * c.HeadDim }

// LoadConfig reads the transformer's config.json.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("dit: parsing config.json: %w", err)
	}
	if c.NumHeads == 0 || c.HeadDim == 0 || c.InChannels == 0 {
		return nil, fmt.Errorf("dit: config.json is missing num_attention_heads, attention_head_dim or in_channels")
	}
	if len(c.AxesDimsRope) != 3 {
		return nil, fmt.Errorf("dit: axes_dims_rope is %v, want 3 axes", c.AxesDimsRope)
	}
	copy(c.AxesDims[:], c.AxesDimsRope)
	if c.PatchSize != 1 {
		// 2.1 consumes latents unpatched, and Assemble's 2x2 slot expansion
		// counts on one token being one latent pixel.
		return nil, fmt.Errorf("dit: patch_size %d is not the 1 this port implements", c.PatchSize)
	}
	if !c.CausalCond {
		return nil, fmt.Errorf("dit: causal_condition=false has no t=0 modulation row; nothing here supports it")
	}
	return &c, nil
}

// loader mirrors zimage/qwen's: named tensors as float32, first error held.
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

func (l *loader) linear(prefix string, out, in int) *qwen.Linear {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(prefix + ".weight")
	if err != nil {
		l.err = err
		return nil
	}
	if len(t.Shape) != 2 || t.Shape[0] != out || t.Shape[1] != in {
		l.err = fmt.Errorf("dit: %s.weight is %v, want [%d %d]", prefix, t.Shape, out, in)
		return nil
	}
	return &qwen.Linear{Out: out, In: in, Weight: l.f32(prefix + ".weight")}
}

// Load reads the transformer. blocks is how many of the 32 to bring in — 0
// for the head/tail alone, 1 for the block oracle, NumLayers for the model.
// fp32 is 28 GB at the full count; this is the oracle, not the fast path.
func Load(dir string, blocks int) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if blocks < 0 || blocks > cfg.NumLayers {
		return nil, fmt.Errorf("dit: asked for %d of %d blocks", blocks, cfg.NumLayers)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	dim := cfg.Dim()
	l := &loader{set: set}
	m := &Model{
		Cfg:        cfg,
		TimeL1:     l.linear("time_text_embed.timestep_embedder.linear_1", dim, 256),
		TimeL2:     l.linear("time_text_embed.timestep_embedder.linear_2", dim, dim),
		TxtNorm:    &ZeroRMSNorm{Weight: l.f32("txt_in.text_norm.weight"), Eps: cfg.Eps},
		TxtIn:      l.linear("txt_in.in_layer", dim, cfg.ContextIn),
		TxtOut:     l.linear("txt_in.out_layer", dim, dim),
		ImgIn:      l.linear("img_in", dim, cfg.InChannels),
		Modulation: l.linear("modulation.1", 4*dim, dim),
		NormOut:    l.linear("norm_out.linear", dim, dim),
		ProjOut:    l.linear("proj_out", cfg.OutChannels, dim),
	}
	for i := 0; i < blocks; i++ {
		b, err := LoadBlock(set, i, cfg)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, b)
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// LoadBlock reads one block as float32 — also what the GPU port (Q4) will
// stage from, one block at a time.
func LoadBlock(set *safetensors.Set, i int, cfg *Config) (*Block, error) {
	dim := cfg.Dim()
	mlp := dim * cfg.MLPRatio
	l := &loader{set: set}
	p := fmt.Sprintf("transformer_blocks.%d.", i)
	b := &Block{
		Q:       l.linear(p+"attn.to_q", dim, dim),
		K:       l.linear(p+"attn.to_k", dim, dim),
		V:       l.linear(p+"attn.to_v", dim, dim),
		O:       l.linear(p+"attn.to_out.0", dim, dim),
		QNorm:   &qwen.RMSNorm{Weight: l.f32(p + "attn.norm_q.weight"), Eps: cfg.Eps},
		KNorm:   &qwen.RMSNorm{Weight: l.f32(p + "attn.norm_k.weight"), Eps: cfg.Eps},
		GateL:   l.linear(p+"img_mlp.gate_layer", mlp, dim),
		Proj:    l.linear(p+"img_mlp.proj", mlp, dim),
		Out:     l.linear(p+"img_mlp.out", dim, mlp),
		Heads:   cfg.NumHeads,
		HeadDim: cfg.HeadDim,
		Eps:     cfg.Eps,
	}
	if l.err != nil {
		return nil, fmt.Errorf("dit: block %d: %w", i, l.err)
	}
	if len(b.QNorm.Weight) != cfg.HeadDim || len(b.KNorm.Weight) != cfg.HeadDim {
		return nil, fmt.Errorf("dit: block %d q/k norms are %d/%d wide, want head_dim %d",
			i, len(b.QNorm.Weight), len(b.KNorm.Weight), cfg.HeadDim)
	}
	return b, nil
}
