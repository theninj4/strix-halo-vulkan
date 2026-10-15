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
	InChannels     int     `json:"in_channels"`
	CondInChannels int     `json:"cond_in_channels"`
	NumLayers      int     `json:"num_layers"`        // mode blocks per mode
	NumSingle      int     `json:"num_single_layers"` // shared single-stream blocks
	HeadDim        int     `json:"attention_head_dim"`
	NumHeads       int     `json:"num_attention_heads"`
	JointDim       int     `json:"joint_attention_dim"` // the text embedding's width, 8 x 2560
	TimeChannels   int     `json:"timestep_channels"`
	MLPRatio       float64 `json:"mlp_ratio"`
	AxesDimsRope   []int   `json:"axes_dims_rope"`
	RopeTheta      float64 `json:"rope_theta"`
	Eps            float64 `json:"eps"`
	FlowMap        bool    `json:"flow_map"`
}

// Dim is the model width, heads times head_dim.
func (c *Config) Dim() int { return c.NumHeads * c.HeadDim }

// MLP is the SwiGLU's inner width.
func (c *Config) MLP() int { return int(float64(c.Dim()) * c.MLPRatio) }

// LoadConfig reads the transformer's config.json and refuses what this port
// does not implement.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("dit: parsing config.json: %w", err)
	}
	if c.NumHeads == 0 || c.HeadDim == 0 || c.InChannels == 0 || c.JointDim == 0 {
		return nil, fmt.Errorf("dit: config.json is missing heads, head_dim, in_channels or joint_attention_dim")
	}
	sum := 0
	for _, d := range c.AxesDimsRope {
		sum += d
	}
	if sum != c.HeadDim {
		return nil, fmt.Errorf("dit: rope axes %v do not fill head dim %d", c.AxesDimsRope, c.HeadDim)
	}
	if !c.FlowMap {
		// The base (CFG) checkpoint has no target_timestep_embedder; the
		// distilled one is what is served.
		return nil, fmt.Errorf("dit: flow_map=false (the undistilled base) is not implemented")
	}
	return &c, nil
}

type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) linear(name string, out, in int) *qwen.Linear {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name + ".weight")
	if err != nil {
		l.err = err
		return nil
	}
	if len(t.Shape) != 2 || t.Shape[0] != out || t.Shape[1] != in {
		l.err = fmt.Errorf("dit: %s.weight is %v, want [%d %d]", name, t.Shape, out, in)
		return nil
	}
	w, err := t.F32(nil)
	if err != nil {
		l.err = err
		return nil
	}
	return &qwen.Linear{Out: out, In: in, Weight: w}
}

func (l *loader) rms(name string, n int, eps float64) *qwen.RMSNorm {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name + ".weight")
	if err != nil {
		l.err = err
		return nil
	}
	w, err := t.F32(nil)
	if err != nil {
		l.err = err
		return nil
	}
	if len(w) != n {
		l.err = fmt.Errorf("dit: %s.weight has %d values, want %d", name, len(w), n)
		return nil
	}
	return &qwen.RMSNorm{Weight: w, Eps: eps}
}

func (l *loader) block(prefix string, c *Config) *Block {
	d, m := c.Dim(), c.MLP()
	b := &Block{
		Q:       l.linear(prefix+".attn.to_q", d, d),
		K:       l.linear(prefix+".attn.to_k", d, d),
		V:       l.linear(prefix+".attn.to_v", d, d),
		NormQ:   l.rms(prefix+".attn.norm_q", c.HeadDim, c.Eps),
		NormK:   l.rms(prefix+".attn.norm_k", c.HeadDim, c.Eps),
		Out:     l.linear(prefix+".attn.to_out.0", d, d),
		FFIn:    l.linear(prefix+".ff.linear_in", 2*m, d),
		FFOut:   l.linear(prefix+".ff.linear_out", d, m),
		Heads:   c.NumHeads,
		HeadDim: c.HeadDim,
		Eps:     c.Eps,
	}
	return b
}

// Parts says which blocks Load brings in; fp32 is 22 GB for all of them.
type Parts struct {
	Mode   int // mode blocks per mode (0..NumLayers)
	Single int // single blocks (0..NumSingle)
}

// All is every block.
func (c *Config) All() Parts { return Parts{Mode: c.NumLayers, Single: c.NumSingle} }

// Load reads the transformer: the embedders, modulations and output layer
// always, and the first p.Mode mode blocks of each mode and first p.Single
// single blocks.
func Load(dir string, p Parts) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if p.Mode < 0 || p.Mode > cfg.NumLayers || p.Single < 0 || p.Single > cfg.NumSingle {
		return nil, fmt.Errorf("dit: asked for %+v of %d mode / %d single blocks", p, cfg.NumLayers, cfg.NumSingle)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}
	d, tc := cfg.Dim(), cfg.TimeChannels
	m := &Model{
		Cfg:        cfg,
		ImgIn:      l.linear("x_embedders.image", d, cfg.InChannels),
		CondIn:     l.linear("x_embedders.image_cond", d, cfg.CondInChannels),
		TxtIn:      l.linear("context_embedder", d, cfg.JointDim),
		Time1:      l.linear("timestep_embedder.linear_1", d, tc),
		Time2:      l.linear("timestep_embedder.linear_2", d, d),
		Target1:    l.linear("target_timestep_embedder.linear_1", d, tc),
		Target2:    l.linear("target_timestep_embedder.linear_2", d, d),
		NormOut:    l.linear("norm_out.linear", 2*d, d),
		ProjOut:    l.linear("proj_out", cfg.InChannels, d),
		EarlyMod:   map[Mode]*qwen.Linear{},
		SingleMod:  map[Mode]*qwen.Linear{},
		ModeBlocks: map[Mode][]*Block{},
	}
	for _, md := range Modes {
		m.EarlyMod[md] = l.linear("early_stream_modulations."+md.key()+".linear", 3*d, d)
		m.SingleMod[md] = l.linear("single_stream_modulations."+md.key()+".linear", 3*d, d)
		for i := 0; i < p.Mode; i++ {
			m.ModeBlocks[md] = append(m.ModeBlocks[md], l.block(fmt.Sprintf("%s.%d", md.blockPrefix(), i), cfg))
		}
	}
	for i := 0; i < p.Single; i++ {
		m.Single = append(m.Single, l.block(fmt.Sprintf("single_transformer_blocks.%d", i), cfg))
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// LoadBlocks reads single blocks by safetensors prefix (e.g.
// "single_transformer_blocks.27", "mode_blocks.image.0") — for a block gate
// that should not hold the whole model.
func LoadBlocks(dir string, prefixes ...string) ([]*Block, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}
	var out []*Block
	for _, p := range prefixes {
		out = append(out, l.block(p, cfg))
	}
	if l.err != nil {
		return nil, l.err
	}
	return out, nil
}
