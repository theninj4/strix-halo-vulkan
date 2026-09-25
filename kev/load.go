package kev

// The checkpoint: Qwen3.5-4B-Base with Kev's LoRA folded in, and the pointer
// head (CLASSIFICATION.md K3).
//
// Kev serves by merging: `W += scale * B @ A` in fp32, rounded once to the
// serving dtype (kev/checkpoint.py, LoadOptions.merge). This does the same,
// from the base's bf16 (exact in fp32) and the adapter's fp32, into the fp16
// bank. The adapter covers every projection of every layer: q/k/v/o and the
// MLP on attention layers, in_proj_qkv/z/a/b and out_proj and the MLP on
// DeltaNet layers.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// Config is the text half of Qwen3.5's config.json, the fields this graph is
// built from. The kernels compile in the 4B's head shapes, so New checks them.
type Config struct {
	Hidden       int     `json:"hidden_size"`
	Intermediate int     `json:"intermediate_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	KVHeads      int     `json:"num_key_value_heads"`
	HeadDim      int     `json:"head_dim"`
	LinKHeads    int     `json:"linear_num_key_heads"`
	LinVHeads    int     `json:"linear_num_value_heads"`
	LinKDim      int     `json:"linear_key_head_dim"`
	LinVDim      int     `json:"linear_value_head_dim"`
	ConvKernel   int     `json:"linear_conv_kernel_dim"`
	RMSEps       float64 `json:"rms_norm_eps"`
	Vocab        int     `json:"vocab_size"`
	LayerTypes   []string
	RopeTheta    float64
	RotaryFactor float64
}

// Attention reports whether layer i is a full-attention layer (the rest are
// Gated DeltaNet).
func (c *Config) Attention(i int) bool { return c.LayerTypes[i] == "full_attention" }

// LoadConfig reads config.json's text_config.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var raw struct {
		Text json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(raw.Text, &c); err != nil {
		return nil, err
	}
	var rest struct {
		LayerTypes []string `json:"layer_types"`
		Rope       struct {
			Theta   float64 `json:"rope_theta"`
			Partial float64 `json:"partial_rotary_factor"`
		} `json:"rope_parameters"`
	}
	if err := json.Unmarshal(raw.Text, &rest); err != nil {
		return nil, err
	}
	c.LayerTypes, c.RopeTheta, c.RotaryFactor = rest.LayerTypes, rest.Rope.Theta, rest.Rope.Partial
	want := Config{Hidden: 2560, Intermediate: 9216, Layers: 32, Heads: 16, KVHeads: 4, HeadDim: 256,
		LinKHeads: 16, LinVHeads: 32, LinKDim: 128, LinVDim: 128, ConvKernel: 4}
	got := c
	got.RMSEps, got.Vocab, got.LayerTypes, got.RopeTheta, got.RotaryFactor = 0, 0, nil, 0, 0
	if fmt.Sprint(got) != fmt.Sprint(want) {
		return nil, fmt.Errorf("kev: %s is not Qwen3.5-4B's shape (%+v); the kernels are built for it", dir, got)
	}
	if len(c.LayerTypes) != c.Layers || c.RotaryFactor != 0.25 {
		return nil, fmt.Errorf("kev: %d layer types for %d layers, rotary factor %v", len(c.LayerTypes), c.Layers, c.RotaryFactor)
	}
	return &c, nil
}

// Head is the pointer head and its calibration: z_i = k(h_opt_i).q(h_decide)
// / sqrt(dp) / T.
type Head struct {
	QW, KW      []float32 // [dp][hidden]
	QB, KB      []float32 // [dp]
	Dim         int       // dp, 256
	Temperature float64
}

// LoadHead reads head.safetensors and head.json, which
// reference/convert_kev_head.py writes from head.pt.
func LoadHead(dir string) (*Head, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "head.json"))
	if err != nil {
		return nil, fmt.Errorf("kev: %w (run reference/convert_kev_head.py %s)", err, dir)
	}
	var meta struct {
		Temperature      float64 `json:"temperature"`
		HeadDim          int     `json:"head_dim"`
		OptionIsolation  bool    `json:"option_isolation"`
		SpecialEmbedding bool    `json:"special_embeddings"`
		Weights          string  `json:"weights"`
	}
	if err := json.Unmarshal(buf, &meta); err != nil {
		return nil, err
	}
	if meta.OptionIsolation || meta.SpecialEmbedding || (meta.Weights != "" && meta.Weights != "lora") {
		return nil, fmt.Errorf("kev: head.json asks for option_isolation/special_embeddings/full weights, none of which this implements")
	}
	f, err := safetensors.Open(filepath.Join(dir, "head.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := &Head{Dim: meta.HeadDim, Temperature: meta.Temperature}
	for name, dst := range map[string]*[]float32{"q.weight": &h.QW, "q.bias": &h.QB, "k.weight": &h.KW, "k.bias": &h.KB} {
		t, err := f.Get(name)
		if err != nil {
			return nil, err
		}
		if *dst, err = t.F32(nil); err != nil {
			return nil, err
		}
	}
	if len(h.QB) != h.Dim || len(h.KB) != h.Dim {
		return nil, fmt.Errorf("kev: head biases are %d/%d wide, head_dim is %d", len(h.QB), len(h.KB), h.Dim)
	}
	return h, nil
}

// Logits scores one question: the <decide> row and its options' </opt> rows,
// all already final-normed. Divided by the temperature; T = 1 is the raw
// head. Computed in float64 from Kev's fp32 weights.
func (h *Head) Logits(decide []float32, opts [][]float32, temperature float64) []float64 {
	q := h.project(h.QW, h.QB, decide)
	scale := 1 / math.Sqrt(float64(h.Dim)) / temperature
	z := make([]float64, len(opts))
	for i, o := range opts {
		k := h.project(h.KW, h.KB, o)
		var s float64
		for j := range k {
			s += k[j] * q[j]
		}
		z[i] = s * scale
	}
	return z
}

func (h *Head) project(w, b, x []float32) []float64 {
	out := make([]float64, h.Dim)
	d := len(x)
	for i := range out {
		s := float64(b[i])
		row := w[i*d : (i+1)*d]
		for j, v := range x {
			s += float64(row[j]) * float64(v)
		}
		out[i] = s
	}
	return out
}

// Softmax of logits, in float64.
func Softmax(z []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range z {
		m = math.Max(m, v)
	}
	p := make([]float64, len(z))
	var s float64
	for i, v := range z {
		p[i] = math.Exp(v - m)
		s += p[i]
	}
	for i := range p {
		p[i] /= s
	}
	return p
}

// Adapter is the LoRA: per target module, A [r][in] and B [out][r], and the
// scale alpha / r.
type Adapter struct {
	set   *safetensors.File
	Scale float32
	Rank  int
}

// OpenAdapter reads adapter_config.json and maps adapter_model.safetensors.
func OpenAdapter(dir string) (*Adapter, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		R        int    `json:"r"`
		Alpha    int    `json:"lora_alpha"`
		DoRA     bool   `json:"use_dora"`
		RSLoRA   bool   `json:"use_rslora"`
		PeftType string `json:"peft_type"`
		Bias     string `json:"bias"`
	}
	if err := json.Unmarshal(buf, &cfg); err != nil {
		return nil, err
	}
	if cfg.PeftType != "LORA" || cfg.DoRA || cfg.RSLoRA || cfg.Bias != "none" || cfg.R <= 0 {
		return nil, fmt.Errorf("kev: adapter is not plain LoRA (%+v)", cfg)
	}
	f, err := safetensors.Open(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return nil, err
	}
	return &Adapter{set: f, Scale: float32(cfg.Alpha) / float32(cfg.R), Rank: cfg.R}, nil
}

// Close unmaps the adapter.
func (a *Adapter) Close() error { return a.set.Close() }

// Count is the number of adapter tensors, 496 for Kev-4B.
func (a *Adapter) Count() int { return len(a.set.Names()) }

// Merge adds scale * B @ A into w ([out][in], fp32, in place) for module,
// e.g. "layers.3.self_attn.q_proj". Every projection is expected to have an
// adapter; a missing one is an error, because a half-merged model still
// answers.
func (a *Adapter) Merge(module string, w []float32, out, in int) error {
	ta, errA := a.set.Get("base_model.model." + module + ".lora_A.weight")
	tb, errB := a.set.Get("base_model.model." + module + ".lora_B.weight")
	if errA != nil || errB != nil {
		return fmt.Errorf("kev: the adapter has no %s", module)
	}
	A, err := ta.F32(nil)
	if err != nil {
		return err
	}
	B, err := tb.F32(nil)
	if err != nil {
		return err
	}
	r := a.Rank
	if len(A) != r*in || len(B) != out*r {
		return fmt.Errorf("kev: %s adapter is A %v B %v for a [%d %d] weight", module, ta.Shape, tb.Shape, out, in)
	}
	// peft's order: delta = (B @ A) * scale, then W + delta, all fp32.
	parallelRows(out, func(o int) {
		row := w[o*in : (o+1)*in]
		delta := make([]float32, in)
		for k := 0; k < r; k++ {
			b := B[o*r+k]
			ar := A[k*in : (k+1)*in]
			for j := range delta {
				delta[j] += b * ar[j]
			}
		}
		for j := range row {
			row[j] += delta[j] * a.Scale
		}
	})
	return nil
}

func parallelRows(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
