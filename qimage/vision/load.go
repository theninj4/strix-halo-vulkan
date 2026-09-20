package vision

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

// LoadConfig reads the vision half of the text encoder's config.json.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var wrapper struct {
		Vision Config `json:"vision_config"`
	}
	if err := json.Unmarshal(buf, &wrapper); err != nil {
		return nil, fmt.Errorf("vision: parsing config.json: %w", err)
	}
	c := wrapper.Vision
	if c.Depth == 0 || c.HiddenSize == 0 {
		return nil, fmt.Errorf("vision: config.json has no vision_config")
	}
	if c.HiddenAct != "gelu_pytorch_tanh" {
		return nil, fmt.Errorf("vision: hidden_act %q; this port implements gelu_pytorch_tanh", c.HiddenAct)
	}
	if c.SpatialMergeSize != 2 {
		return nil, fmt.Errorf("vision: spatial_merge_size %d; this port implements 2", c.SpatialMergeSize)
	}
	if side := c.GridPerSide(); side*side != c.NumPositionEmbeddings {
		return nil, fmt.Errorf("vision: %d position embeddings is not a square grid", c.NumPositionEmbeddings)
	}
	return &c, nil
}

type loader struct {
	set    *safetensors.Set
	prefix string
	err    error
}

func (l *loader) f32(name string, want int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(l.prefix + name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
		return nil
	}
	if want > 0 && len(v) != want {
		l.err = fmt.Errorf("vision: %s has %d values, want %d", name, len(v), want)
		return nil
	}
	return v
}

func (l *loader) linear(name string, out, in int) Linear {
	return Linear{In: in, Out: out,
		Weight: l.f32(name+".weight", out*in),
		Bias:   l.f32(name+".bias", out)}
}

func (l *loader) norm(name string, width int) LayerNorm {
	return LayerNorm{Weight: l.f32(name+".weight", width), Bias: l.f32(name+".bias", width)}
}

// merger loads one of the four patch mergers. `post` decides where the norm
// sits, and the norm's own width in the checkpoint is what proves it: a
// post-shuffle merger normalizes the concatenated 4608, a pre-shuffle one
// the 1152 it came from. Loading the wrong width is an error here rather
// than a wrong answer later.
func (l *loader) merger(name string, cfg *Config, post bool) Merger {
	merge := cfg.SpatialMergeSize * cfg.SpatialMergeSize
	wide := cfg.HiddenSize * merge
	normWidth := cfg.HiddenSize
	if post {
		normWidth = wide
	}
	return Merger{
		PostShuffleNorm: post,
		Merge:           merge,
		Norm:            l.norm(name+".norm", normWidth),
		FC1:             l.linear(name+".linear_fc1", wide, wide),
		FC2:             l.linear(name+".linear_fc2", cfg.OutHiddenSize, wide),
		// nn.GELU(), the exact one -- not the block MLP's tanh
		// approximation. Both are in this model and they are not the same
		// function.
		Act: geluErf,
	}
}

// Load reads the tower out of the text encoder's safetensors.
//
// `blocks` caps how many of the 27 are loaded, which is what lets a test
// walk the first two without staging 1.0 GB of ViT. Zero or negative means
// all of them; a deepstack merger whose layer is past the cap is skipped.
func Load(dir string, cfg *Config, blocks int) (*Model, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	if blocks <= 0 || blocks > cfg.Depth {
		blocks = cfg.Depth
	}
	l := &loader{set: set, prefix: "model.visual."}

	m := &Model{Cfg: *cfg}
	// The Conv3d patch embedding is one linear: its kernel is exactly one
	// patch, so [1152, 3, 2, 16, 16] is [1152, 1536] row-major already.
	m.PatchProj = l.linear("patch_embed.proj", cfg.HiddenSize, cfg.PatchElems())
	m.PosEmbed = l.f32("pos_embed.weight", cfg.NumPositionEmbeddings*cfg.HiddenSize)

	for i := 0; i < blocks; i++ {
		p := fmt.Sprintf("blocks.%d.", i)
		m.Blocks = append(m.Blocks, Block{
			Norm1: l.norm(p+"norm1", cfg.HiddenSize),
			Norm2: l.norm(p+"norm2", cfg.HiddenSize),
			Attn: Attention{
				QKV:      l.linear(p+"attn.qkv", 3*cfg.HiddenSize, cfg.HiddenSize),
				Proj:     l.linear(p+"attn.proj", cfg.HiddenSize, cfg.HiddenSize),
				NumHeads: cfg.NumHeads,
			},
			MLP: MLP{
				FC1: l.linear(p+"mlp.linear_fc1", cfg.IntermediateSize, cfg.HiddenSize),
				FC2: l.linear(p+"mlp.linear_fc2", cfg.HiddenSize, cfg.IntermediateSize),
				Act: geluTanh, // config hidden_act; the mergers use the erf one
			},
		})
	}
	for j, idx := range cfg.DeepstackIndexes {
		if idx >= blocks {
			continue
		}
		m.Deepstack = append(m.Deepstack, l.merger(fmt.Sprintf("deepstack_merger_list.%d", j), cfg, true))
	}
	m.Merger = l.merger("merger", cfg, false)

	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}
