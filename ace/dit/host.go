package dit

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// Linear is a projection with an optional bias, fp32 on the host.
type Linear struct {
	qwen.Linear
	Bias []float32
}

// Apply is W·x + b over every row.
func (l *Linear) Apply(x *qwen.Mat) (*qwen.Mat, error) {
	y, err := l.Linear.Apply(x)
	if err != nil || l.Bias == nil {
		return y, err
	}
	for r := 0; r < y.Rows; r++ {
		row := y.Row(r)
		for i := range row {
			row[i] += l.Bias[i]
		}
	}
	return y, nil
}

// TimeMLP is upstream's TimestepEmbedding: sinusoid(256) of t·1000 →
// linear_1 → silu → linear_2 = temb, and time_proj(silu(temb)) = the six
// AdaLN rows.
type TimeMLP struct {
	L1, L2, Proj *Linear
}

// Host holds everything the host runs in fp32: a few rows of arithmetic a
// request or a step, next to the device's billions.
type Host struct {
	Cfg *Config
	// The DiT's two timestep MLPs: time_embed(t) + time_embed_r(t − r).
	Time, TimeR TimeMLP
	// Per layer [6, hidden]: shift, scale, gate (self-attention), then
	// shift, scale, gate (MLP), added to the timestep projection.
	Tables [][]float32
	// Per layer: self_attn_norm and mlp_norm, which the device's modulated
	// norm folds into a = w·(1 + scale).
	Norms [][2][]float32
	// norm_out and its [2, hidden] (shift, scale) table over temb.
	NormOut  []float32
	OutTable []float32
	ProjOutB []float32 // [Latent]
	CondB    []float32 // condition_embedder bias
	// The encoders' final norms and the timbre encoder's CLS row.
	LyricNorm, TimbreNorm *qwen.RMSNorm
	TimbreCLS             []float32
	// The audio detokenizer's host pieces (A4): the FSQ project_out
	// ([2048, 6] and bias), embed_tokens' bias, the 5 learned tokens
	// [5, 2048], and proj_out's bias.
	FSQW, FSQB   []float32
	DetokEmbedB  []float32
	DetokSpecial []float32
	DetokProjB   []float32
	// The audio tokenizer's host pieces (A11c): the folded input bias
	// (embed_tokens·proj bias + embed_tokens bias), the pooler's CLS row
	// and final norm, and the FSQ project_in ([6, 2048] and bias).
	TokB        []float32
	PoolSpecial []float32
	PoolNorm    *qwen.RMSNorm
	FSQInW      []float32
	FSQInB      []float32
}

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
	n := 1
	for _, w := range want {
		n *= w
	}
	total := 1
	for _, s := range t.Shape {
		total *= s
	}
	if total != n {
		l.err = fmt.Errorf("dit: %s is %v, want %v", name, t.Shape, want)
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
	}
	return v
}

func (l *loader) linear(name string, out, in int, bias bool) *Linear {
	lin := &Linear{Linear: qwen.Linear{Out: out, In: in, Weight: l.f32(name+".weight", out, in)}}
	if bias {
		lin.Bias = l.f32(name+".bias", out)
	}
	return lin
}

func (l *loader) timeMLP(p string, h int) TimeMLP {
	return TimeMLP{
		L1:   l.linear(p+".linear_1", h, 256, true),
		L2:   l.linear(p+".linear_2", h, h, true),
		Proj: l.linear(p+".time_proj", 6*h, h, true),
	}
}

// LoadHost reads the host's pieces from an open checkpoint.
func LoadHost(set *safetensors.Set, c *Config) (*Host, error) {
	l := &loader{set: set}
	h := &Host{
		Cfg:          c,
		Time:         l.timeMLP("decoder.time_embed", c.Hidden),
		TimeR:        l.timeMLP("decoder.time_embed_r", c.Hidden),
		NormOut:      l.f32("decoder.norm_out.weight", c.Hidden),
		OutTable:     l.f32("decoder.scale_shift_table", 2, c.Hidden),
		ProjOutB:     l.f32("decoder.proj_out.1.bias", c.Latent),
		CondB:        l.f32("decoder.condition_embedder.bias", c.Hidden),
		LyricNorm:    &qwen.RMSNorm{Weight: l.f32("encoder.lyric_encoder.norm.weight", c.EncHidden), Eps: c.Eps},
		TimbreNorm:   &qwen.RMSNorm{Weight: l.f32("encoder.timbre_encoder.norm.weight", c.EncHidden), Eps: c.Eps},
		TimbreCLS:    l.f32("encoder.timbre_encoder.special_token", c.EncHidden),
		FSQW:         l.f32("tokenizer.quantizer.project_out.weight", c.EncHidden, 6),
		FSQB:         l.f32("tokenizer.quantizer.project_out.bias", c.EncHidden),
		DetokEmbedB:  l.f32("detokenizer.embed_tokens.bias", c.EncHidden),
		DetokSpecial: l.f32("detokenizer.special_tokens", c.PoolWindow, c.EncHidden),
		DetokProjB:   l.f32("detokenizer.proj_out.bias", c.Latent),
		PoolSpecial:  l.f32("tokenizer.attention_pooler.special_token", c.EncHidden),
		PoolNorm:     &qwen.RMSNorm{Weight: l.f32("tokenizer.attention_pooler.norm.weight", c.EncHidden), Eps: c.Eps},
		FSQInW:       l.f32("tokenizer.quantizer.project_in.weight", 6, c.EncHidden),
		FSQInB:       l.f32("tokenizer.quantizer.project_in.bias", 6),
	}
	// The folded input bias: embed_tokens(audio_acoustic_proj.bias) +
	// embed_tokens.bias, in fp64.
	{
		H := c.EncHidden
		we := l.f32("tokenizer.attention_pooler.embed_tokens.weight", H, H)
		be := l.f32("tokenizer.attention_pooler.embed_tokens.bias", H)
		bp := l.f32("tokenizer.audio_acoustic_proj.bias", H)
		if l.err == nil {
			h.TokB = make([]float32, H)
			for o := 0; o < H; o++ {
				s := float64(be[o])
				for j := 0; j < H; j++ {
					s += float64(we[o*H+j]) * float64(bp[j])
				}
				h.TokB[o] = float32(s)
			}
		}
	}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("decoder.layers.%d.", i)
		h.Tables = append(h.Tables, l.f32(p+"scale_shift_table", 6, c.Hidden))
		h.Norms = append(h.Norms, [2][]float32{l.f32(p+"self_attn_norm.weight", c.Hidden), l.f32(p+"mlp_norm.weight", c.Hidden)})
	}
	if l.err != nil {
		return nil, l.err
	}
	return h, nil
}

// sinusoid is upstream's timestep_embedding(t·1000, 256) in fp32, as torch
// forms it: freqs = exp(−ln(10⁴)·i/128) and args = t·freqs, both rounded to
// fp32, then [cos | sin].
func sinusoid(t float32, dim int) []float32 {
	half := dim / 2
	ts := t * 1000
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		e := float32(-math.Log(10000)) * float32(i) / float32(half)
		f := float32(math.Exp(float64(e)))
		a := ts * f
		out[i] = float32(math.Cos(float64(a)))
		out[half+i] = float32(math.Sin(float64(a)))
	}
	return out
}

func silu(m *qwen.Mat) *qwen.Mat {
	out := m.Clone()
	for i, v := range out.Data {
		out.Data[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
	return out
}

// embed runs one TimestepEmbedding: temb [hidden] and proj [6·hidden].
func (m TimeMLP) embed(t float32) (temb, proj []float32, err error) {
	x := &qwen.Mat{Rows: 1, Cols: 256, Data: sinusoid(t, 256)}
	h, err := m.L1.Apply(x)
	if err != nil {
		return nil, nil, err
	}
	e, err := m.L2.Apply(silu(h))
	if err != nil {
		return nil, nil, err
	}
	p, err := m.Proj.Apply(silu(e))
	if err != nil {
		return nil, nil, err
	}
	return e.Data, p.Data, nil
}

// Timestep is one forward's conditioning on t (with r = t, as upstream's
// turbo sampler always calls it): temb = time_embed(t) + time_embed_r(0),
// and the [6, hidden] projection likewise.
func (h *Host) Timestep(t float32) (temb, proj []float32, err error) {
	e1, p1, err := h.Time.embed(t)
	if err != nil {
		return nil, nil, err
	}
	e2, p2, err := h.TimeR.embed(0)
	if err != nil {
		return nil, nil, err
	}
	for i := range e1 {
		e1[i] += e2[i]
	}
	for i := range p1 {
		p1[i] += p2[i]
	}
	return e1, p1, nil
}
