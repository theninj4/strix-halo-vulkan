package qwen

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

// Config mirrors the text encoder's config.json.
type Config struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	VocabSize        int     `json:"vocab_size"`
	RMSEps           float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
}

// LoadConfig reads a text encoder checkpoint's config.json.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("qwen: parsing config.json: %w", err)
	}
	if c.HiddenSize == 0 || c.NumLayers == 0 || c.HeadDim == 0 {
		return nil, fmt.Errorf("qwen: config.json is missing hidden_size, num_hidden_layers or head_dim")
	}
	return &c, nil
}

// EncoderLayers is how many of the model's layers the pipeline needs.
//
// ZImagePipeline._encode_prompt takes `hidden_states[-2]`, and a
// hidden_states tuple is the embedding output followed by each layer's
// output, so [-2] is layer NumLayers-2's output: the last layer never runs,
// the final norm never runs, and there is no lm_head. That is 35 of 36
// layers here, worth 2.8% of the compute and 111 M of the 4.02 B parameters.
func (c *Config) EncoderLayers() int { return c.NumLayers - 1 }

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
		l.err = fmt.Errorf("qwen: %s.weight has shape %v, want 2 dims", prefix, sh)
		return nil
	}
	return &Linear{Out: sh[0], In: sh[1], Weight: l.f32(prefix + ".weight")}
}

func (l *loader) rms(name string, eps float64) *RMSNorm {
	return &RMSNorm{Weight: l.f32(name + ".weight"), Eps: eps}
}

// Load reads the text encoder. layers is how many decoder layers to bring
// in, counted from 0; pass cfg.EncoderLayers() for what the pipeline uses
// and cfg.NumLayers for the whole model. The weights land as float32, which
// is 404 MB per layer -- this is the oracle, not the fast path.
func Load(dir string, layers int) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if layers <= 0 || layers > cfg.NumLayers {
		return nil, fmt.Errorf("qwen: asked for %d of %d layers", layers, cfg.NumLayers)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	m := &Model{Cfg: cfg}
	if m.Rows, m.Embed, err = LoadEmbedding(set, cfg); err != nil {
		return nil, err
	}
	for i := 0; i < layers; i++ {
		l, err := LoadLayer(set, i, cfg)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, l)
	}
	return m, nil
}

// LoadEmbedding reads the token embedding table as float32, returning its row
// count alongside it -- vocab_size is padded above the tokenizer's id space,
// and the table is the authority on how far.
func LoadEmbedding(set *safetensors.Set, cfg *Config) (int, []float32, error) {
	l := &loader{set: set}
	shape := l.shape(embedName)
	if l.err != nil {
		return 0, nil, l.err
	}
	if len(shape) != 2 || shape[1] != cfg.HiddenSize {
		return 0, nil, fmt.Errorf("qwen: %s is %v, want [rows %d]", embedName, shape, cfg.HiddenSize)
	}
	data := l.f32(embedName)
	return shape[0], data, l.err
}

// LoadLayer reads one decoder layer as float32. It is what both the CPU
// model and the GPU encoder build from: the GPU path stages one layer at a
// time into its fp16 banks and drops it, so the host never holds more than
// the 404 MB one layer costs rather than the 14.1 GB all 35 would.
func LoadLayer(set *safetensors.Set, i int, cfg *Config) (*Layer, error) {
	l := &loader{set: set}
	p := fmt.Sprintf("model.layers.%d.", i)
	layer := &Layer{
		AttnNorm: l.rms(p+"input_layernorm", cfg.RMSEps),
		Q:        l.linear(p + "self_attn.q_proj"),
		K:        l.linear(p + "self_attn.k_proj"),
		V:        l.linear(p + "self_attn.v_proj"),
		QNorm:    l.rms(p+"self_attn.q_norm", cfg.RMSEps),
		KNorm:    l.rms(p+"self_attn.k_norm", cfg.RMSEps),
		O:        l.linear(p + "self_attn.o_proj"),
		FFNNorm:  l.rms(p+"post_attention_layernorm", cfg.RMSEps),
		Gate:     l.linear(p + "mlp.gate_proj"),
		Up:       l.linear(p + "mlp.up_proj"),
		Down:     l.linear(p + "mlp.down_proj"),

		Heads:   cfg.NumHeads,
		KVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim,
	}
	if l.err != nil {
		return nil, l.err
	}
	// The shapes are checked on every layer rather than on the first: a
	// checkpoint that changes width halfway through is not a thing that
	// happens, but a *name* that resolves to the wrong tensor is, and this is
	// where it would show.
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"q_proj", layer.Q.Out, cfg.NumHeads * cfg.HeadDim},
		{"k_proj", layer.K.Out, cfg.NumKVHeads * cfg.HeadDim},
		{"v_proj", layer.V.Out, cfg.NumKVHeads * cfg.HeadDim},
		{"o_proj", layer.O.Out, cfg.HiddenSize},
		{"gate_proj", layer.Gate.Out, cfg.IntermediateSize},
		{"up_proj", layer.Up.Out, cfg.IntermediateSize},
		{"down_proj", layer.Down.Out, cfg.HiddenSize},
	} {
		if c.got != c.want {
			return nil, fmt.Errorf("qwen: layer %d %s produces %d, want %d", i, c.name, c.got, c.want)
		}
	}
	return layer, nil
}

// embedName is the one tensor outside the layers that the encoder reads. The
// final norm and the lm_head are in the checkpoint and are never loaded:
// hidden_states[-2] is taken before either of them runs.
const embedName = "model.embed_tokens.weight"
