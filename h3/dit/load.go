package dit

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// loader reads named tensors as float32, holding the first error.
type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string, want ...int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	if len(t.Shape) != len(want) {
		l.err = fmt.Errorf("dit: %s is %v, want %v", name, t.Shape, want)
		return nil
	}
	for i := range want {
		if t.Shape[i] != want[i] {
			l.err = fmt.Errorf("dit: %s is %v, want %v", name, t.Shape, want)
			return nil
		}
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
	}
	return v
}

func (l *loader) linear(prefix string, out, in int, bias bool) *Linear {
	lin := &Linear{Linear: qwen.Linear{Out: out, In: in, Weight: l.f32(prefix+".weight", out, in)}}
	if bias {
		lin.Bias = l.f32(prefix+".bias", out)
	}
	return lin
}

func (l *loader) rms(name string, width int, eps float64) *qwen.RMSNorm {
	return &qwen.RMSNorm{Weight: l.f32(name, width), Eps: eps}
}

func (l *loader) attention(p string, c *Config) *Attention {
	return &Attention{
		Q:       l.linear(p+"to_q", c.Inner(), c.Hidden, false),
		K:       l.linear(p+"to_k", c.Inner(), c.Hidden, false),
		V:       l.linear(p+"to_v", c.Inner(), c.Hidden, false),
		O:       l.linear(p+"to_out.0", c.Hidden, c.Inner(), false),
		QNorm:   l.rms(p+"norm_q.weight", c.HeadDim, c.QKNormEps),
		KNorm:   l.rms(p+"norm_k.weight", c.HeadDim, c.QKNormEps),
		Heads:   c.Heads,
		HeadDim: c.HeadDim,
	}
}

func (l *loader) swiglu(p string, c *Config) *SwiGLU {
	return &SwiGLU{
		Up:    l.linear(p+"net.0.proj", 2*c.FFN, c.Hidden, false),
		Down:  l.linear(p+"net.2", c.Hidden, c.FFN, false),
		Inner: c.FFN,
	}
}

// Load reads the transformer in float32 with the first blocks of its
// blocks. This is the oracle, not the fast path: every block is 1.8 GB here
// with its AdaLN projection, and the whole model would be 132 GB.
func Load(dir string, blocks int) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if blocks < 0 || blocks > cfg.Layers {
		return nil, fmt.Errorf("dit: asked for %d of %d blocks", blocks, cfg.Layers)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	c := cfg
	l := &loader{set: set}
	m := &Model{
		Cfg:           c,
		TimeL1:        l.linear("time_embedder.linear_1", c.TimeHidden, c.FreqDim, true),
		TimeL2:        l.linear("time_embedder.linear_2", c.TimeDim, c.TimeHidden, true),
		ProjIn:        l.linear("proj_in", c.Hidden, c.Patch(), true),
		AudioProjIn:   l.linear("audio_proj_in", c.Hidden, c.AudioChannels, true),
		TextIn:        l.linear("context_embedder", c.Hidden, c.TextDim, true),
		RefinerNorm:   l.rms("token_refiner.final_norm.weight", c.Hidden, c.FinalNormEps),
		NormOut:       l.rms("norm_out.norm.weight", c.Hidden, c.FinalNormEps),
		NormOutLinear: l.linear("norm_out.linear", 2*c.Hidden, c.TimeDim, true),
		ProjOut:       l.linear("proj_out", c.Patch(), c.Hidden, true),
		AudioProjOut:  l.linear("audio_proj_out", c.AudioChannels, c.Hidden, true),
	}
	for i := 0; i < c.RefinerLayers; i++ {
		p := fmt.Sprintf("token_refiner.refiner_blocks.%d.", i)
		m.Refiner = append(m.Refiner, &RefinerBlock{
			Norm1: l.rms(p+"norm1.weight", c.Hidden, c.NormEps),
			Norm2: l.rms(p+"norm2.weight", c.Hidden, c.NormEps),
			Attn:  l.attention(p+"attn.", c),
			FF:    l.swiglu(p+"ff.", c),
		})
	}
	for i := 0; i < blocks; i++ {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		m.Blocks = append(m.Blocks, &Block{
			Norm1:     l.rms(p+"norm1.weight", c.Hidden, c.NormEps),
			Norm2:     l.rms(p+"norm2.weight", c.Hidden, c.NormEps),
			Attn:      l.attention(p+"attn.", c),
			FF:        l.swiglu(p+"ff.", c),
			AdaLN:     l.linear(p+"adaln_proj.linear", 6*Modalities*c.Hidden, c.TimeDim, true),
			Hidden:    c.Hidden,
			RopeWidth: c.RopeWidth(),
		})
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}
