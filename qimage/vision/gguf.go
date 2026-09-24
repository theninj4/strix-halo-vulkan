package vision

// The same tower out of a llama.cpp `mmproj` GGUF (LLM-VISION.md V1).
//
// qwen3.8-flash-next ships its vision encoder as `mmproj-BF16.gguf`, and it
// is this package's tower with two differences: the merger projects to the
// language model's 2560 rather than Qwen-Image's 4096, and there is no
// deepstack. The config is not a config.json but `clip.vision.*` keys, and the
// names are llama.cpp's. One tensor is also laid out differently. The Conv3d
// patch kernel [1152, 3, 2, 16, 16] is stored split along time, as
// `v.patch_embd.weight` (frame 0) and `v.patch_embd.weight.1` (frame 1),
// each ggml [16, 16, 3, 1152]. Patchify's row is channels, then time, then
// pixels, so the two have to be interleaved per channel rather than
// concatenated. Concatenating them gives a plausible embedding of a
// different image.
//
// TestGGUFMatchesSafetensors holds every tensor this loader produces against
// the HF checkpoint's `model.visual.*`, bit for bit.

import (
	"fmt"

	"strix-halo-vulkan/gguf"
)

// LoadGGUF reads the tower and its config out of an mmproj file. `blocks`
// caps the layers loaded, as in Load.
func LoadGGUF(path string, blocks int) (*Config, *Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	cfg, err := ggufConfig(f)
	if err != nil {
		return nil, nil, err
	}
	if blocks <= 0 || blocks > cfg.Depth {
		blocks = cfg.Depth
	}
	l := &ggufLoader{f: f}

	m := &Model{Cfg: *cfg}
	m.PatchProj = Linear{In: cfg.PatchElems(), Out: cfg.HiddenSize,
		Weight: l.patchKernel(cfg),
		Bias:   l.f32("v.patch_embd.bias", cfg.HiddenSize)}
	m.PosEmbed = l.f32("v.position_embd.weight", cfg.NumPositionEmbeddings*cfg.HiddenSize)

	for i := 0; i < blocks; i++ {
		p := fmt.Sprintf("v.blk.%d.", i)
		m.Blocks = append(m.Blocks, Block{
			Norm1: l.norm(p+"ln1", cfg.HiddenSize),
			Norm2: l.norm(p+"ln2", cfg.HiddenSize),
			Attn: Attention{
				QKV:      l.linear(p+"attn_qkv", 3*cfg.HiddenSize, cfg.HiddenSize),
				Proj:     l.linear(p+"attn_out", cfg.HiddenSize, cfg.HiddenSize),
				NumHeads: cfg.NumHeads,
			},
			MLP: MLP{
				FC1: l.linear(p+"ffn_up", cfg.IntermediateSize, cfg.HiddenSize),
				FC2: l.linear(p+"ffn_down", cfg.HiddenSize, cfg.IntermediateSize),
				Act: geluTanh,
			},
		})
	}
	// The output merger is the pre-shuffle kind, so its norm is the 1152-wide
	// `v.post_ln` and llama.cpp's `mm.0`/`mm.2` are its two linears (`mm.1`
	// is the GELU between them, which has no weights).
	wide := cfg.HiddenSize * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	m.Merger = Merger{
		Merge: cfg.SpatialMergeSize * cfg.SpatialMergeSize,
		Norm:  l.norm("v.post_ln", cfg.HiddenSize),
		FC1:   l.linear("mm.0", wide, wide),
		FC2:   l.linear("mm.2", cfg.OutHiddenSize, wide),
		Act:   geluErf,
	}
	if l.err != nil {
		return nil, nil, l.err
	}
	return cfg, m, nil
}

// ggufConfig builds the Config LoadConfig would have read from config.json.
// The keys llama.cpp does not carry are constants of the architecture, and
// each is checked against something the file does carry.
func ggufConfig(f *gguf.File) (*Config, error) {
	if s, _ := f.Str("clip.projector_type"); s != "qwen3vl_merger" {
		return nil, fmt.Errorf("vision: projector %q; this port implements qwen3vl_merger", s)
	}
	get := func(key string) int {
		v, _ := f.Uint("clip.vision." + key)
		return int(v)
	}
	c := &Config{
		Depth:             get("block_count"),
		HiddenSize:        get("embedding_length"),
		NumHeads:          get("attention.head_count"),
		IntermediateSize:  get("feed_forward_length"),
		HiddenAct:         "gelu_pytorch_tanh",
		PatchSize:         get("patch_size"),
		SpatialMergeSize:  get("spatial_merge_size"),
		TemporalPatchSize: 2,
		InChannels:        3,
		OutHiddenSize:     get("projection_dim"),
	}
	if c.Depth == 0 || c.HiddenSize == 0 || c.NumHeads == 0 || c.OutHiddenSize == 0 {
		return nil, fmt.Errorf("vision: mmproj is missing clip.vision.* keys")
	}
	if ok, _ := f.Bool("clip.use_gelu"); !ok {
		return nil, fmt.Errorf("vision: clip.use_gelu is not set; this port implements gelu_pytorch_tanh")
	}
	if c.SpatialMergeSize != 2 {
		return nil, fmt.Errorf("vision: spatial_merge_size %d; this port implements 2", c.SpatialMergeSize)
	}
	// Temporal patch 2 is the two kernels existing. A third would be a
	// different model.
	if !f.Has("v.patch_embd.weight.1") || f.Has("v.patch_embd.weight.2") {
		return nil, fmt.Errorf("vision: mmproj does not split the patch kernel in two along time")
	}
	pos, err := f.Get("v.position_embd.weight")
	if err != nil {
		return nil, err
	}
	c.NumPositionEmbeddings = int(pos.Rows())
	if side := c.GridPerSide(); side*side != c.NumPositionEmbeddings {
		return nil, fmt.Errorf("vision: %d position embeddings is not a square grid", c.NumPositionEmbeddings)
	}
	// Deepstack mergers would be tensors this loader never reads; refuse
	// rather than run a tower with its taps silently missing.
	if flags, ok := f.KV["clip.vision.is_deepstack_layers"].([]bool); ok {
		for _, on := range flags {
			if on {
				return nil, fmt.Errorf("vision: mmproj has deepstack layers; this loader has none")
			}
		}
	}
	for i := 0; i < c.Depth; i++ {
		if f.Has(fmt.Sprintf("v.deepstack.%d.fc1.weight", i)) {
			return nil, fmt.Errorf("vision: mmproj has a deepstack merger at block %d; this loader has none", i)
		}
	}
	return c, nil
}

type ggufLoader struct {
	f   *gguf.File
	err error
}

func (l *ggufLoader) f32(name string, want int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.f.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.Dequantize(nil)
	if err != nil {
		l.err = fmt.Errorf("vision: %s: %w", name, err)
		return nil
	}
	if want > 0 && len(v) != want {
		l.err = fmt.Errorf("vision: %s has %d values, want %d", name, len(v), want)
		return nil
	}
	return v
}

// linear reads a ggml [in, out] weight, which in memory is row-major
// [out][in]: the same layout as the HF Linear, so nothing is transposed.
func (l *ggufLoader) linear(name string, out, in int) Linear {
	return Linear{In: in, Out: out,
		Weight: l.f32(name+".weight", out*in),
		Bias:   l.f32(name+".bias", out)}
}

func (l *ggufLoader) norm(name string, width int) LayerNorm {
	return LayerNorm{Weight: l.f32(name+".weight", width), Bias: l.f32(name+".bias", width)}
}

// patchKernel reassembles [out][c][t][py][px] from the two per-frame
// kernels, each [out][c][py][px].
func (l *ggufLoader) patchKernel(cfg *Config) []float32 {
	pp := cfg.PatchSize * cfg.PatchSize
	per := cfg.HiddenSize * cfg.InChannels * pp
	frames := [2][]float32{
		l.f32("v.patch_embd.weight", per),
		l.f32("v.patch_embd.weight.1", per),
	}
	if l.err != nil {
		return nil
	}
	w := make([]float32, cfg.HiddenSize*cfg.PatchElems())
	for o := 0; o < cfg.HiddenSize; o++ {
		for c := 0; c < cfg.InChannels; c++ {
			for t, fr := range frames {
				src := fr[(o*cfg.InChannels+c)*pp:][:pp]
				dst := w[((o*cfg.InChannels+c)*cfg.TemporalPatchSize+t)*pp:][:pp]
				copy(dst, src)
			}
		}
	}
	return w
}
