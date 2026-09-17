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

	// pleAdvised is whether the n-gram table's mapping has been told it is
	// read at random (L7c). Once a Model, on the first gather.
	pleAdvised bool
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
	x, err := t.Dequantize(nil)
	if err != nil {
		return nil, err
	}
	// L8c's width simulation, if one is running: a streamed dense weight is
	// round-tripped through the candidate format on its way to the device,
	// so that a width can be graded before there is a kernel that reads it
	// (sim.go). Off by default, and off is an identity.
	// ggml memory order: a tensor stated [in, out] reads back as out rows of
	// in values, so a row is Dims[0] long and a group runs along k, which is
	// what a format groups along and what an imatrix column indexes.
	if err := DensePlan().ApplyTo(name, t.Type == gguf.Q8_0, x, int(t.Dims[0])); err != nil {
		return nil, err
	}
	return x, nil
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
	w.Name = prefix
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
	// Once, on the first gather: sixteen scattered 90-byte reads a token are
	// what this mapping is for, and the kernel's default readahead answers
	// each of them with 128 KB (L7c). It is set here rather than at open,
	// because it is a fact about how *this* tensor is read and the rest of
	// the shard is read exactly once, sequentially, and wants the readahead.
	if !m.pleAdvised {
		m.pleAdvised = true
		if err := t.AdviseRandom(); err != nil {
			return nil, fmt.Errorf("llm: per_layer_token_embd madvise: %w", err)
		}
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

// AttnConfig returns a full-attention layer's shape. ok is false for the
// linear-attention layers, which are 36 of the 48 and have no attention
// tensors at all.
//
// The compress ratio is per layer and comes from an array rather than a
// scalar: `attention.compress_ratios` is 48 entries and only the
// full-attention ones are non-zero, so the layer index is not optional.
func (m *Model) AttnConfig(layer int) (AttnConfig, bool, error) {
	arch := m.Set.Arch()
	if !m.Set.Has(fmt.Sprintf("blk.%d.attn_q.weight", layer)) {
		return AttnConfig{}, false, nil
	}
	c := m.Config
	get := func(k string) int { v, _ := m.Set.Uint(arch + "." + k); return int(v) }
	a := AttnConfig{
		NEmbd: c.NEmbd, NHead: c.NHead, NHeadKV: c.NHeadKV,
		HeadDim:  get("attention.key_length"),
		RopeDims: c.RopeDims,
		RopeBase: c.RopeBase,
		Eps:      c.RMSEps,
		IdxHeads: get("attention.indexer.head_count"),
		IdxDim:   get("attention.indexer.key_length"),
		TopK:     get("attention.indexer.top_k"),
	}
	if v, ok := m.Set.Ints(arch + ".rope.dimension_sections"); ok {
		for i := 0; i < len(a.Sections) && i < len(v); i++ {
			a.Sections[i] = int(v[i])
		}
	}
	if v, ok := m.Set.Ints(arch + ".attention.compress_ratios"); ok {
		if layer < len(v) {
			a.Ratio = int(v[layer])
		}
	}
	if a.HeadDim == 0 || a.NHead == 0 || a.NHeadKV == 0 {
		return a, true, fmt.Errorf("llm: layer %d states %d heads of %d over %d kv heads",
			layer, a.NHead, a.HeadDim, a.NHeadKV)
	}
	return a, true, nil
}

// AttnWeights loads one full-attention layer's tensors, the indexer's
// included.
func (m *Model) AttnWeights(layer int) (AttnWeights, error) {
	p := fmt.Sprintf("blk.%d.", layer)
	var w AttnWeights
	for _, t := range []struct {
		name string
		dst  *[]float32
	}{
		{"attn_q", &w.Q}, {"attn_k", &w.K}, {"attn_v", &w.V}, {"attn_output", &w.O},
		{"attn_q_norm", &w.QNorm}, {"attn_k_norm", &w.KNorm},
		{"indexer.q_proj", &w.IdxQ}, {"indexer.k_proj", &w.IdxK},
		{"indexer.q_norm", &w.IdxQNorm}, {"indexer.k_norm", &w.IdxKNorm},
	} {
		v, err := m.F32(p + t.name + ".weight")
		if err != nil {
			return w, err
		}
		*t.dst = v
	}
	return w, nil
}

// DeltaNetConfig returns a linear-attention layer's shape. ok is false for
// the 12 full-attention layers, which have no `ssm_*` tensors at all — the
// mirror of AttnConfig, and between them the two cover all 48.
//
// The three head counts come from keys whose names are ggml's Mamba
// vocabulary rather than this architecture's: `time_step_rank` is the value
// head count, `group_count` the key head count, and `state_size` the width of
// a head on both sides. Reading them is not optional — the 16/48 split is
// what the whole layer is shaped around.
func (m *Model) DeltaNetConfig(layer int) (DeltaNetConfig, bool, error) {
	arch := m.Set.Arch()
	if !m.Set.Has(fmt.Sprintf("blk.%d.attn_qkv.weight", layer)) {
		return DeltaNetConfig{}, false, nil
	}
	get := func(k string) int { v, _ := m.Set.Uint(arch + ".ssm." + k); return int(v) }
	c := DeltaNetConfig{
		NEmbd:   m.Config.NEmbd,
		HeadDim: get("state_size"),
		NHeadK:  get("group_count"),
		NHeadV:  get("time_step_rank"),
		Conv:    get("conv_kernel"),
		Inner:   get("inner_size"),
		Eps:     m.Config.RMSEps,
	}
	switch {
	case c.HeadDim == 0 || c.NHeadK == 0 || c.NHeadV == 0 || c.Conv == 0:
		return c, true, fmt.Errorf("llm: layer %d states %d key heads and %d value heads of %d, conv %d",
			layer, c.NHeadK, c.NHeadV, c.HeadDim, c.Conv)
	case c.NHeadV*c.HeadDim != c.Inner:
		return c, true, fmt.Errorf("llm: %d value heads of %d do not make ssm.inner_size %d",
			c.NHeadV, c.HeadDim, c.Inner)
	case c.NHeadV%c.NHeadK != 0:
		return c, true, fmt.Errorf("llm: %d value heads do not divide into %d key heads",
			c.NHeadV, c.NHeadK)
	}
	return c, true, nil
}

// DeltaNetWeights loads one linear-attention layer's tensors.
func (m *Model) DeltaNetWeights(layer int) (DeltaNetWeights, error) {
	p := fmt.Sprintf("blk.%d.", layer)
	w := DeltaNetWeights{Layer: layer}
	for _, t := range []struct {
		name string
		dst  *[]float32
	}{
		{"attn_qkv.weight", &w.QKV}, {"attn_gate.weight", &w.Z}, {"ssm_out.weight", &w.Out},
		{"ssm_alpha.weight", &w.Alpha}, {"ssm_beta.weight", &w.Beta},
		{"ssm_conv1d.weight", &w.Conv1d}, {"ssm_norm.weight", &w.Norm},
		{"ssm_a", &w.A}, {"ssm_dt.bias", &w.DTBias},
	} {
		v, err := m.F32(p + t.name)
		if err != nil {
			return w, err
		}
		*t.dst = v
	}
	return w, nil
}
