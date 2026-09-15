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
