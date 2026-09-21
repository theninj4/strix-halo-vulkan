package vae

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

// Config mirrors vae/config.json.
type Config struct {
	BaseDim        int       `json:"base_dim"`
	DecoderBaseDim int       `json:"decoder_base_dim"`
	ZDim           int       `json:"z_dim"`
	DimMult        []int     `json:"dim_mult"`
	NumResBlocks   int       `json:"num_res_blocks"`
	AttnScales     []float64 `json:"attn_scales"`
	TemporalDown   []bool    `json:"temperal_downsample"`
	InChannels     int       `json:"in_channels"`
	OutChannels    int       `json:"out_channels"`
	IsResidual     bool      `json:"is_residual"`
	LatentsMean    []float32 `json:"latents_mean"`
	LatentsStd     []float32 `json:"latents_std"`
	ScaleSpatial   int       `json:"scale_factor_spatial"`
}

// LoadConfig reads the VAE's config.json, refusing shapes this port does not
// implement rather than decoding a different model.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("qvae: parsing config.json: %w", err)
	}
	if !c.IsResidual {
		return nil, fmt.Errorf("qvae: is_residual=false is the plain Wan layout, which this port does not build")
	}
	if len(c.AttnScales) != 0 {
		return nil, fmt.Errorf("qvae: attn_scales %v want attention outside the mid block, unimplemented", c.AttnScales)
	}
	if len(c.TemporalDown) != len(c.DimMult)-1 {
		return nil, fmt.Errorf("qvae: %d temporal flags for %d blocks", len(c.TemporalDown), len(c.DimMult))
	}
	if len(c.LatentsMean) != c.ZDim || len(c.LatentsStd) != c.ZDim {
		return nil, fmt.Errorf("qvae: latents mean/std are %d/%d values for z_dim %d",
			len(c.LatentsMean), len(c.LatentsStd), c.ZDim)
	}
	if c.DecoderBaseDim == 0 {
		c.DecoderBaseDim = c.BaseDim
	}
	return &c, nil
}

// Normalize maps a raw latent to the DiT's space: (z - mean) / std, per
// channel. Denormalize is the inverse the decoder wants.
func (c *Config) Normalize(z *Tensor) {
	for ch := 0; ch < z.C; ch++ {
		m, s := c.LatentsMean[ch], c.LatentsStd[ch]
		p := z.Plane(0, ch)
		for i := range p {
			p[i] = (p[i] - m) / s
		}
	}
}

// Denormalize maps a DiT-space latent back: z*std + mean.
func (c *Config) Denormalize(z *Tensor) {
	for ch := 0; ch < z.C; ch++ {
		m, s := c.LatentsMean[ch], c.LatentsStd[ch]
		p := z.Plane(0, ch)
		for i := range p {
			p[i] = p[i]*s + m
		}
	}
}

type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string, want int) []float32 {
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
		return nil
	}
	if len(v) != want {
		l.err = fmt.Errorf("qvae: %s has %d values, want %d", name, len(v), want)
		return nil
	}
	return v
}

// conv reads a Conv2d, validating the full shape.
func (l *loader) conv(name string, out, in, k, pad, padEnd, stride int) Conv2D {
	if l.err != nil {
		return Conv2D{}
	}
	t, err := l.set.Get(name + ".weight")
	if err != nil {
		l.err = err
		return Conv2D{}
	}
	sh := t.Shape
	if len(sh) != 4 || sh[0] != out || sh[1] != in || sh[2] != k || sh[3] != k {
		l.err = fmt.Errorf("qvae: %s.weight is %v, want [%d %d %d %d]", name, sh, out, in, k, k)
		return Conv2D{}
	}
	return Conv2D{
		InC: in, OutC: out, KH: k, KW: k, Pad: pad, PadEnd: padEnd, Stride: stride,
		Weight: l.f32(name+".weight", out*in*k*k),
		Bias:   l.f32(name+".bias", out),
	}
}

func (l *loader) norm(name string, c int) ChannelNorm {
	return ChannelNorm{Gamma: l.f32(name+".gamma", c)}
}

func (l *loader) res(prefix string, in, out int) ResBlock {
	b := ResBlock{
		Norm1: l.norm(prefix+".norm1", in),
		Conv1: l.conv(prefix+".conv1", out, in, 3, 1, 0, 1),
		Norm2: l.norm(prefix+".norm2", out),
		Conv2: l.conv(prefix+".conv2", out, out, 3, 1, 0, 1),
	}
	if in != out {
		sc := l.conv(prefix+".conv_shortcut", out, in, 1, 0, 0, 1)
		b.Shortcut = &sc
	}
	return b
}

func (l *loader) mid(prefix string, dim int) Mid {
	return Mid{
		Res1: l.res(prefix+".resnets.0", dim, dim),
		Attn: Attention{
			Norm: l.norm(prefix+".attentions.0.norm", dim),
			QKV:  l.conv(prefix+".attentions.0.to_qkv", 3*dim, dim, 1, 0, 0, 1),
			Proj: l.conv(prefix+".attentions.0.proj", dim, dim, 1, 0, 0, 1),
		},
		Res2: l.res(prefix+".resnets.1", dim, dim),
	}
}

// LoadDecoder reads the decoder (plus post_quant_conv) as float32.
func LoadDecoder(dir string, cfg *Config) (*Decoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}

	n := len(cfg.DimMult)
	dims := make([]int, 0, n+1)
	dims = append(dims, cfg.DecoderBaseDim*cfg.DimMult[n-1])
	for i := n - 1; i >= 0; i-- {
		dims = append(dims, cfg.DecoderBaseDim*cfg.DimMult[i])
	}
	// temporal upsample flags are the downsample ones reversed
	tus := make([]bool, len(cfg.TemporalDown))
	for i, b := range cfg.TemporalDown {
		tus[len(tus)-1-i] = b
	}

	d := &Decoder{
		PostQuant: l.conv("post_quant_conv", cfg.ZDim, cfg.ZDim, 1, 0, 0, 1),
		ConvIn:    l.conv("decoder.conv_in", dims[0], cfg.ZDim, 3, 1, 0, 1),
		Mid:       l.mid("decoder.mid_block", dims[0]),
		NormOut:   l.norm("decoder.norm_out", dims[n]),
		ConvOut:   l.conv("decoder.conv_out", cfg.OutChannels, dims[n], 3, 1, 0, 1),
	}
	for i := 0; i < n; i++ {
		in, out := dims[i], dims[i+1]
		up := i != n-1
		b := UpBlock{}
		cur := in
		for r := 0; r <= cfg.NumResBlocks; r++ {
			b.Resnets = append(b.Resnets, l.res(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", i, r), cur, out))
			cur = out
		}
		if up {
			b.Up = &Resample{Up: true, Conv: l.conv(fmt.Sprintf("decoder.up_blocks.%d.upsampler.resample.1", i), out, out, 3, 1, 0, 1)}
			ft := 1
			if tus[i] {
				ft = 2
			}
			b.Shortcut = &DupUp{In: in, Out: out, FactorT: ft}
		}
		d.Ups = append(d.Ups, b)
	}
	if l.err != nil {
		return nil, l.err
	}
	return d, nil
}

// LoadEncoder reads the encoder (plus quant_conv) as float32.
func LoadEncoder(dir string, cfg *Config) (*Encoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}

	n := len(cfg.DimMult)
	dims := make([]int, 0, n+1)
	dims = append(dims, cfg.BaseDim)
	for _, m := range cfg.DimMult {
		dims = append(dims, cfg.BaseDim*m)
	}

	e := &Encoder{
		ConvIn:  l.conv("encoder.conv_in", dims[0], cfg.InChannels, 3, 1, 0, 1),
		Mid:     l.mid("encoder.mid_block", dims[n]),
		NormOut: l.norm("encoder.norm_out", dims[n]),
		ConvOut: l.conv("encoder.conv_out", 2*cfg.ZDim, dims[n], 3, 1, 0, 1),
		Quant:   l.conv("quant_conv", 2*cfg.ZDim, 2*cfg.ZDim, 1, 0, 0, 1),
	}
	for i := 0; i < n; i++ {
		in, out := dims[i], dims[i+1]
		down := i != n-1
		temporal := down && cfg.TemporalDown[i]
		b := DownBlock{Shortcut: AvgDown{In: in, Out: out, FactorT: 1, FactorS: 1}}
		if down {
			b.Shortcut.FactorS = 2
		}
		if temporal {
			b.Shortcut.FactorT = 2
		}
		cur := in
		for r := 0; r < cfg.NumResBlocks; r++ {
			b.Resnets = append(b.Resnets, l.res(fmt.Sprintf("encoder.down_blocks.%d.resnets.%d", i, r), cur, out))
			cur = out
		}
		if down {
			b.Down = &Resample{Conv: l.conv(fmt.Sprintf("encoder.down_blocks.%d.downsampler.resample.1", i), out, out, 3, 0, 1, 2)}
		}
		e.Downs = append(e.Downs, b)
	}
	if l.err != nil {
		return nil, l.err
	}
	return e, nil
}
