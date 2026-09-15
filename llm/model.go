package llm

import (
	"fmt"

	"strix-halo-vulkan/gguf"
)

// Config is the model's shape, read from the checkpoint's own metadata rather
// than transcribed from a config.json — which is the mistake L1 found in
// bench/modelshapes.go, four rows wrong and four families missing.
type Config struct {
	Arch        string
	NLayer      int
	NEmbd       int
	NHead       int
	NHeadKV     int
	HC          int // hyper-connection streams
	HCLowRank   int
	NExpert     int
	NExpertUsed int
	FFNExpert   int
	RMSEps      float32
	RopeBase    float32
	RopeDims    int
	VocabSize   int
}

// HCConfig returns the hyper-connection block's shape.
func (c Config) HCConfig() HCConfig {
	return HCConfig{NEmbd: c.NEmbd, HC: c.HC, LowRank: c.HCLowRank, Eps: c.RMSEps}
}

// Model is a checkpoint held open: the mmap'd shards plus their metadata.
//
// Weights are dequantised on demand. At L2 the CPU reference touches a few
// tensors of a few layers, and materialising 82 GB of f32 to reach them would
// be absurd — so nothing is cached here and the caller keeps what it wants.
type Model struct {
	Config Config
	Set    *gguf.Set
}

// Open maps a checkpoint: a directory, or any one of its shards.
func Open(path string) (*Model, error) {
	s, err := gguf.OpenSet(path)
	if err != nil {
		return nil, err
	}
	m := &Model{Set: s}
	if err := m.readConfig(); err != nil {
		s.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) readConfig() error {
	arch := m.Set.Arch()
	if arch == "" {
		return fmt.Errorf("llm: checkpoint states no general.architecture")
	}
	get := func(k string) int { v, _ := m.Set.Uint(arch + "." + k); return int(v) }
	c := Config{
		Arch:        arch,
		NLayer:      get("block_count"),
		NEmbd:       get("embedding_length"),
		NHead:       get("attention.head_count"),
		NHeadKV:     get("attention.head_count_kv"),
		HC:          get("hyper_connection.count"),
		HCLowRank:   get("hyper_connection.low_rank"),
		NExpert:     get("expert_count"),
		NExpertUsed: get("expert_used_count"),
		FFNExpert:   get("expert_feed_forward_length"),
		RopeDims:    get("rope.dimension_count"),
	}
	if v, ok := m.Set.Float(arch + ".attention.layer_norm_rms_epsilon"); ok {
		c.RMSEps = float32(v)
	}
	if v, ok := m.Set.Float(arch + ".rope.freq_base"); ok {
		c.RopeBase = float32(v)
	}
	if t, err := m.Set.Get("token_embd.weight"); err == nil && len(t.Dims) > 1 {
		c.VocabSize = int(t.Dims[1])
	}
	m.Config = c
	return nil
}

// F32 dequantises a whole tensor. The result is in ggml's memory order, so a
// weight the GGUF states as [in, out] reads back as out rows of in values —
// the row-major [out][in] a dot product wants, with no transpose.
func (m *Model) F32(name string) ([]float32, error) {
	t, err := m.Set.Get(name)
	if err != nil {
		return nil, err
	}
	return t.Dequantize(nil)
}

// Embedding gathers one row of token_embd, which is a lookup and not a
// matmul: 675 MB of Q8_0 that only ever has n_tokens rows read out of it.
func (m *Model) Embedding(id int32) ([]float32, error) {
	t, err := m.Set.Get("token_embd.weight")
	if err != nil {
		return nil, err
	}
	return t.DequantizeRow(int64(id), nil)
}

// Embeddings gathers a whole prompt, rows in token order: [T][nEmbd].
func (m *Model) Embeddings(ids []int32) ([]float32, error) {
	out := make([]float32, 0, len(ids)*m.Config.NEmbd)
	for _, id := range ids {
		row, err := m.Embedding(id)
		if err != nil {
			return nil, err
		}
		out = append(out, row...)
	}
	return out, nil
}

// HCWeights loads one hyper-connection mixer. side is "attn" or "ffn"; a
// negative layer selects the head mixer, which has no inject because there is
// no block after it to scatter into.
func (m *Model) HCWeights(layer int, side string) (HCWeights, error) {
	prefix := "output_hc_"
	if layer >= 0 {
		prefix = fmt.Sprintf("blk.%d.hc_%s_", layer, side)
	}
	var w HCWeights
	var err error
	if w.Norm, err = m.F32(prefix + "norm.weight"); err != nil {
		return w, err
	}
	if w.Down, err = m.F32(prefix + "down.weight"); err != nil {
		return w, err
	}
	if w.Up, err = m.F32(prefix + "up.weight"); err != nil {
		return w, err
	}
	if layer >= 0 {
		if w.Inject, err = m.F32(prefix + "inject.weight"); err != nil {
			return w, err
		}
	}
	return w, nil
}

// Close unmaps the shards.
func (m *Model) Close() error { return m.Set.Close() }

// PLEConfig reads the per-layer-embedding block's shape and its hash
// constants out of the checkpoint's `qwen4exp.ple.*` keys. ok is false when
// the checkpoint has no PLE module at all, which is how llama.cpp treats the
// key group: absent means every field stays zero.
//
// The constants are not optional detail. `layer_multipliers`,
// `head_offsets` and `head_vocab_sizes` *are* the hash — sixteen near-prime
// vocabularies around 20 000 0xx over disjoint ranges of a 320 M-row table —
// so a transcription of them would be a silent wrong answer, and they are
// read rather than written down.
func (m *Model) PLEConfig() (PLEConfig, bool, error) {
	arch := m.Set.Arch()
	layers, ok := m.Set.Ints(arch + ".ple.layers")
	if !ok || len(layers) == 0 {
		return PLEConfig{}, false, nil
	}
	c := PLEConfig{
		NEmbd: m.Config.NEmbd,
		HC:    m.Config.HC,
		Eps:   m.Config.RMSEps,
	}
	for _, l := range layers {
		c.Layers = append(c.Layers, int(l))
	}
	get := func(k string) (int, error) {
		v, ok := m.Set.Uint(arch + ".ple." + k)
		if !ok {
			return 0, fmt.Errorf("llm: the checkpoint has a PLE module but no %s.ple.%s", arch, k)
		}
		return int(v), nil
	}
	var err error
	if c.NGram, err = get("ngram_size"); err != nil {
		return c, true, err
	}
	if c.PerGram, err = get("heads_per_ngram"); err != nil {
		return c, true, err
	}
	if c.Conv, err = get("conv_kernel"); err != nil {
		return c, true, err
	}
	eos, err := get("eos_token_id")
	if err != nil {
		return c, true, err
	}
	c.EOS = int32(eos)
	if v, ok := m.Set.Uint(arch + ".ple.image_token_id"); ok {
		c.Image = int32(v)
	}
	dim, ok := m.Set.Uint(arch + ".embedding_length_per_layer_input")
	if !ok {
		return c, true, fmt.Errorf("llm: the checkpoint states no %s.embedding_length_per_layer_input", arch)
	}
	c.HeadDim = int(dim)
	c.NHeads = (c.NGram - 1) * c.PerGram

	if c.Mult, ok = m.Set.Uints(arch + ".ple.layer_multipliers"); !ok || len(c.Mult) != c.NGram {
		return c, true, fmt.Errorf("llm: %s.ple.layer_multipliers is %d values, want %d", arch, len(c.Mult), c.NGram)
	}
	for _, k := range []string{"head_offsets", "head_vocab_sizes"} {
		v, ok := m.Set.Uints(arch + ".ple." + k)
		if !ok || len(v) != c.NHeads {
			return c, true, fmt.Errorf("llm: %s.ple.%s is %d values, want %d", arch, k, len(v), c.NHeads)
		}
		u := make([]uint32, len(v))
		for i, x := range v {
			u[i] = uint32(x)
		}
		if k == "head_offsets" {
			c.Offsets = u
		} else {
			c.Vocabs = u
		}
	}
	if c.NHeads*c.HeadDim != c.NEmbd {
		return c, true, fmt.Errorf("llm: %d PLE heads of %d do not make n_embd %d, which the key and value projections assume",
			c.NHeads, c.HeadDim, c.NEmbd)
	}
	return c, true, nil
}

// PLEWeights loads the n-gram block's tensors for a layer.
func (m *Model) PLEWeights(layer int) (PLEWeights, error) {
	p := fmt.Sprintf("blk.%d.ple_", layer)
	var w PLEWeights
	for _, t := range []struct {
		name string
		dst  *[]float32
	}{
		{"key", &w.Key}, {"value", &w.Value},
		{"norm_key", &w.NormKey}, {"norm_query", &w.NormQuery}, {"norm_conv", &w.NormConv},
		{"conv1d", &w.Conv1d},
	} {
		v, err := m.F32(p + t.name + ".weight")
		if err != nil {
			return w, err
		}
		*t.dst = v
	}
	return w, nil
}

// PLEGather reads the rows PLERows named, as [T][NHeads*HeadDim].
//
// This is D2 in one function: `per_layer_token_embd` is 28.80 GB of IQ4_NL —
// a quarter of the whole checkpoint — and a token reads 16 rows of 160
// values out of it, 1.41 KB. It stays in the mapping and is never staged
// anywhere, which is also what llama.cpp does (`TENSOR_READ_LAZY`).
func (m *Model) PLEGather(rows []int32, nHeads, headDim int) ([]float32, error) {
	t, err := m.Set.Get("per_layer_token_embd.weight")
	if err != nil {
		return nil, err
	}
	if int(t.Dims[0]) != headDim {
		return nil, fmt.Errorf("llm: per_layer_token_embd rows are %d wide, want %d", t.Dims[0], headDim)
	}
	out := make([]float32, len(rows)*headDim)
	// gguf.Dequantize appends, so the scratch row is handed over empty and
	// kept only for its capacity.
	buf := make([]float32, 0, headDim)
	for i, r := range rows {
		if int64(r) < 0 || int64(r) >= t.Dims[1] {
			return nil, fmt.Errorf("llm: PLE row %d (head %d of token %d) is outside the table's %d rows",
				r, i%nHeads, i/nHeads, t.Dims[1])
		}
		v, err := t.DequantizeRow(int64(r), buf[:0])
		if err != nil {
			return nil, err
		}
		copy(out[i*headDim:], v)
	}
	return out, nil
}
