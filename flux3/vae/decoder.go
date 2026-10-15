package vae

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

// Decoder is AutoencoderKLFlux2's decoder: post_quant_conv (1x1), conv_in, a
// mid block with attention between two resnets, four up blocks (three of
// which upsample 2x), then a group norm, SiLU and a 3-channel output
// convolution. It is the FLUX.1 autoencoder's decoder (this file is
// zimage/vae's, restored from 77f6627^ for X4, research/flux3-vertical.md)
// with a 32-channel latent and the 1x1 post-quant convolution in front.
type Decoder struct {
	PostQuantConv *Conv2D
	ConvIn        *Conv2D
	Mid           *MidBlock
	UpBlocks      []*UpBlock
	ConvNormOut   *GroupNorm
	ConvOut       *Conv2D
}

// ResnetBlock is diffusers' ResnetBlock2D with output_scale_factor 1: two
// norm/SiLU/conv pairs added to the input, through a 1x1 convolution when
// the channel count changes.
type ResnetBlock struct {
	Norm1, Norm2 *GroupNorm
	Conv1, Conv2 *Conv2D
	Shortcut     *Conv2D // nil when in and out channels match
}

// Apply runs the block. x is not modified.
func (r *ResnetBlock) Apply(x *Tensor) (*Tensor, error) {
	h := &Tensor{N: x.N, C: x.C, H: x.H, W: x.W, Data: append([]float32(nil), x.Data...)}
	var err error
	if h, err = r.Norm1.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	if h, err = r.Conv1.Apply(h); err != nil {
		return nil, err
	}
	if h, err = r.Norm2.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	if h, err = r.Conv2.Apply(h); err != nil {
		return nil, err
	}
	skip := x
	if r.Shortcut != nil {
		if skip, err = r.Shortcut.Apply(x); err != nil {
			return nil, err
		}
	}
	return AddInPlace(h, skip)
}

// Attention is the mid block's spatial self-attention: one head over the
// whole channel width, with the H*W pixels as the sequence, a group norm in
// front and a residual connection around it.
type Attention struct {
	GroupNorm    *GroupNorm
	Q, K, V, Out *Linear
	Heads        int
	Scale        float64
}

// Linear is a [out, in] weight with an optional bias, applied to the channel
// axis of an NCHW tensor viewed as (H*W) rows of C.
type Linear struct {
	In, Out int
	Weight  []float32 // [Out][In]
	Bias    []float32 // [Out], may be nil
}

// applyRows computes y[r] = W x[r] + b for rows of length In.
func (l *Linear) applyRows(src []float32, rows int) []float32 {
	dst := make([]float32, rows*l.Out)
	parallelFor(l.Out, func(o int) {
		w := l.Weight[o*l.In : (o+1)*l.In]
		var bias float32
		if l.Bias != nil {
			bias = l.Bias[o]
		}
		for r := 0; r < rows; r++ {
			x := src[r*l.In : (r+1)*l.In]
			sum := bias
			for i, wv := range w {
				sum += wv * x[i]
			}
			dst[r*l.Out+o] = sum
		}
	})
	return dst
}

// Apply runs attention and adds the residual.
func (a *Attention) Apply(x *Tensor) (*Tensor, error) {
	if a.Heads != 1 {
		return nil, fmt.Errorf("vae: attention with %d heads is not implemented; the VAE uses 1", a.Heads)
	}
	h := &Tensor{N: x.N, C: x.C, H: x.H, W: x.W, Data: append([]float32(nil), x.Data...)}
	var err error
	if h, err = a.GroupNorm.ApplyInPlace(h); err != nil {
		return nil, err
	}

	// NCHW -> (H*W, C): attention runs over pixels, with the channel axis
	// as the feature. This is the transpose diffusers does with
	// view(B, C, H*W).transpose(1, 2).
	rows := h.H * h.W
	seq := make([]float32, rows*h.C)
	for c := 0; c < h.C; c++ {
		p := h.Plane(0, c)
		for i, v := range p {
			seq[i*h.C+c] = v
		}
	}

	q := a.Q.applyRows(seq, rows)
	k := a.K.applyRows(seq, rows)
	v := a.V.applyRows(seq, rows)

	// Scores, the softmax denominator and the P·V sums accumulate in
	// float64. At 1024² the sequence is 16,384 pixels, and float32
	// sequential sums there put the mid block at rel 1.1e-4 from the dump
	// while torch's own fp32 sits 1.8e-6 from a float64 run (X4): this is
	// the oracle, so it pays for the precision.
	dim := a.Q.Out
	ctx := make([]float32, rows*dim)
	parallelFor(rows, func(i int) {
		qi := q[i*dim : (i+1)*dim]
		scores := make([]float64, rows)
		maxScore := math.Inf(-1)
		for j := 0; j < rows; j++ {
			kj := k[j*dim : (j+1)*dim]
			var s float64
			for d, qv := range qi {
				s += float64(qv) * float64(kj[d])
			}
			s *= a.Scale
			scores[j] = s
			if s > maxScore {
				maxScore = s
			}
		}
		var denom float64
		for j, s := range scores {
			e := math.Exp(s - maxScore)
			scores[j] = e
			denom += e
		}
		acc := make([]float64, dim)
		for j, w := range scores {
			if w == 0 {
				continue
			}
			w /= denom
			vj := v[j*dim : (j+1)*dim]
			for d := range acc {
				acc[d] += w * float64(vj[d])
			}
		}
		out := ctx[i*dim : (i+1)*dim]
		for d, x := range acc {
			out[d] = float32(x)
		}
	})

	proj := a.Out.applyRows(ctx, rows)

	// Back to NCHW, adding the residual as we go.
	out := NewTensor(x.N, x.C, x.H, x.W)
	for c := 0; c < x.C; c++ {
		dst, res := out.Plane(0, c), x.Plane(0, c)
		for i := range dst {
			dst[i] = proj[i*x.C+c] + res[i]
		}
	}
	return out, nil
}

// MidBlock is resnet, attention, resnet at the decoder's lowest resolution.
type MidBlock struct {
	Resnet1 *ResnetBlock
	Attn    *Attention
	Resnet2 *ResnetBlock
}

// Apply runs the mid block.
func (m *MidBlock) Apply(x *Tensor) (*Tensor, error) {
	h, err := m.Resnet1.Apply(x)
	if err != nil {
		return nil, err
	}
	if h, err = m.Attn.Apply(h); err != nil {
		return nil, err
	}
	return m.Resnet2.Apply(h)
}

// UpBlock is a run of resnets followed by an optional 2x nearest upsample
// and its smoothing convolution.
type UpBlock struct {
	Resnets   []*ResnetBlock
	Upsampler *Conv2D // nil on the last block
}

// Apply runs the block.
func (u *UpBlock) Apply(x *Tensor) (*Tensor, error) {
	var err error
	h := x
	for _, r := range u.Resnets {
		if h, err = r.Apply(h); err != nil {
			return nil, err
		}
	}
	if u.Upsampler == nil {
		return h, nil
	}
	return u.Upsampler.Apply(UpsampleNearest2x(h))
}

// Apply decodes a latent [1, 32, H/8, W/8] to an image in [-1, 1]-ish RGB.
// The latent is the VAE's own — the DiT's tokens through
// pipeline.BN.ToVAE (de-normalise, unpatchify) — not the DiT's.
func (d *Decoder) Apply(latent *Tensor) (*Tensor, error) {
	h, err := d.PostQuantConv.Apply(latent)
	if err != nil {
		return nil, err
	}
	if h, err = d.ConvIn.Apply(h); err != nil {
		return nil, err
	}
	if h, err = d.Mid.Apply(h); err != nil {
		return nil, err
	}
	for i, up := range d.UpBlocks {
		if h, err = up.Apply(h); err != nil {
			return nil, fmt.Errorf("vae: up block %d: %w", i, err)
		}
	}
	if h, err = d.ConvNormOut.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	return d.ConvOut.Apply(h)
}

// loader pulls named tensors out of a checkpoint as float32, recording the
// first error so a long chain of lookups can be written without a check on
// every line.
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
		return nil
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

func (l *loader) conv(prefix string, pad int) *Conv2D {
	c := l.convOptBias(prefix, pad)
	if l.err == nil && c.Bias == nil {
		l.err = fmt.Errorf("vae: %s has no bias", prefix)
	}
	return c
}

// convOptBias is conv where the bias may legitimately be absent. Only the
// tiny decoder needs it (taef1 then, taef2 at X6) -- the three convolutions
// after its upsamples are built with bias=False -- and it is separate from
// conv so that a missing bias stays an error everywhere it would be a
// silently wrong answer.
func (l *loader) convOptBias(prefix string, pad int) *Conv2D {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 4 {
		l.err = fmt.Errorf("vae: %s.weight has shape %v, want 4 dims", prefix, sh)
		return nil
	}
	c := &Conv2D{
		OutC: sh[0], InC: sh[1], KH: sh[2], KW: sh[3], Pad: pad,
		Weight: l.f32(prefix + ".weight"),
	}
	if l.set.Has(prefix + ".bias") {
		c.Bias = l.f32(prefix + ".bias")
	}
	return c
}

func (l *loader) groupNorm(prefix string, groups int, eps float64) *GroupNorm {
	return &GroupNorm{
		Groups: groups, Eps: eps,
		Weight: l.f32(prefix + ".weight"),
		Bias:   l.f32(prefix + ".bias"),
	}
}

func (l *loader) linear(prefix string) *Linear {
	sh := l.shape(prefix + ".weight")
	if l.err != nil {
		return nil
	}
	if len(sh) != 2 {
		l.err = fmt.Errorf("vae: %s.weight has shape %v, want 2 dims", prefix, sh)
		return nil
	}
	lin := &Linear{Out: sh[0], In: sh[1], Weight: l.f32(prefix + ".weight")}
	if l.set.Has(prefix + ".bias") {
		lin.Bias = l.f32(prefix + ".bias")
	}
	return lin
}

func (l *loader) resnet(prefix string, groups int, eps float64) *ResnetBlock {
	r := &ResnetBlock{
		Norm1: l.groupNorm(prefix+".norm1", groups, eps),
		Conv1: l.conv(prefix+".conv1", 1),
		Norm2: l.groupNorm(prefix+".norm2", groups, eps),
		Conv2: l.conv(prefix+".conv2", 1),
	}
	if l.set.Has(prefix + ".conv_shortcut.weight") {
		r.Shortcut = l.conv(prefix+".conv_shortcut", 0)
	}
	return r
}

// Config mirrors the parts of vae/config.json the port needs.
type Config struct {
	LatentChannels   int   `json:"latent_channels"`
	BlockOutChannels []int `json:"block_out_channels"`
	LayersPerBlock   int   `json:"layers_per_block"`
	NormNumGroups    int   `json:"norm_num_groups"`
	UseQuantConv     bool  `json:"use_quant_conv"`
	UsePostQuantConv bool  `json:"use_post_quant_conv"`
	MidAttention     bool  `json:"mid_block_add_attention"`
}

// LoadConfig reads vae/config.json and refuses the shapes this port does
// not implement.
func LoadConfig(dir string) (Config, error) {
	var c Config
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(buf, &c); err != nil {
		return c, fmt.Errorf("vae: parsing config.json: %w", err)
	}
	if !c.UseQuantConv || !c.UsePostQuantConv || !c.MidAttention {
		return c, fmt.Errorf("vae: config wants quant convs and mid attention on, got %+v", c)
	}
	if c.LatentChannels == 0 || len(c.BlockOutChannels) == 0 || c.LayersPerBlock == 0 || c.NormNumGroups == 0 {
		return c, fmt.Errorf("vae: config.json is missing a shape: %+v", c)
	}
	return c, nil
}

// LoadDecoder reads the decoder weights out of a VAE checkpoint directory.
func LoadDecoder(dir string, cfg Config) (*Decoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	// The weights are copied into float32 slices here, so the mapping does
	// not need to outlive this call.
	defer set.Close()

	const eps = 1e-6
	l := &loader{set: set}

	d := &Decoder{
		PostQuantConv: l.conv("post_quant_conv", 0),
		ConvIn:        l.conv("decoder.conv_in", 1),
		ConvNormOut:   l.groupNorm("decoder.conv_norm_out", cfg.NormNumGroups, eps),
		ConvOut:       l.conv("decoder.conv_out", 1),
	}
	d.Mid = &MidBlock{
		Resnet1: l.resnet("decoder.mid_block.resnets.0", cfg.NormNumGroups, eps),
		Resnet2: l.resnet("decoder.mid_block.resnets.1", cfg.NormNumGroups, eps),
	}
	attnPrefix := "decoder.mid_block.attentions.0"
	q := l.linear(attnPrefix + ".to_q")
	if l.err != nil {
		return nil, l.err
	}
	d.Mid.Attn = &Attention{
		GroupNorm: l.groupNorm(attnPrefix+".group_norm", cfg.NormNumGroups, eps),
		Q:         q,
		K:         l.linear(attnPrefix + ".to_k"),
		V:         l.linear(attnPrefix + ".to_v"),
		Out:       l.linear(attnPrefix + ".to_out.0"),
		Heads:     1,
		// diffusers sets scale from dim_head, which for the VAE's single
		// head is the full channel width.
		Scale: 1 / math.Sqrt(float64(q.Out)),
	}

	// The decoder walks block_out_channels in reverse, and has
	// layers_per_block+1 resnets per block. Every block but the last
	// upsamples.
	nBlocks := len(cfg.BlockOutChannels)
	for i := 0; i < nBlocks; i++ {
		prefix := fmt.Sprintf("decoder.up_blocks.%d", i)
		up := &UpBlock{}
		for j := 0; j <= cfg.LayersPerBlock; j++ {
			up.Resnets = append(up.Resnets, l.resnet(fmt.Sprintf("%s.resnets.%d", prefix, j), cfg.NormNumGroups, eps))
		}
		if set.Has(prefix + ".upsamplers.0.conv.weight") {
			up.Upsampler = l.conv(prefix+".upsamplers.0.conv", 1)
		} else if i != nBlocks-1 {
			l.err = fmt.Errorf("vae: up block %d has no upsampler but is not the last of %d", i, nBlocks)
		}
		d.UpBlocks = append(d.UpBlocks, up)
	}
	if l.err != nil {
		return nil, l.err
	}
	if d.PostQuantConv.InC != cfg.LatentChannels || d.PostQuantConv.OutC != cfg.LatentChannels {
		return nil, fmt.Errorf("vae: post_quant_conv is %d -> %d, want %d", d.PostQuantConv.InC, d.PostQuantConv.OutC, cfg.LatentChannels)
	}
	if d.ConvIn.InC != cfg.LatentChannels {
		return nil, fmt.Errorf("vae: conv_in takes %d channels, config says %d latent channels", d.ConvIn.InC, cfg.LatentChannels)
	}
	if d.ConvOut.OutC != 3 {
		return nil, fmt.Errorf("vae: conv_out produces %d channels, want 3", d.ConvOut.OutC)
	}
	return d, nil
}

// RGB8 turns a decoded [1, 3, H, W] image into 8-bit RGB, row-major HWC, the
// way diffusers' VaeImageProcessor.postprocess does: (x*0.5 + 0.5) clamped
// to [0, 1] in float32, then *255 rounded half to even (numpy's round).
func RGB8(img *Tensor) []uint8 {
	out := make([]uint8, img.H*img.W*3)
	for c := 0; c < 3; c++ {
		for i, v := range img.Plane(0, c) {
			u := float32(float32(v*0.5) + 0.5)
			u = min(max(u, 0), 1)
			out[i*3+c] = uint8(math.RoundToEven(float64(u * 255)))
		}
	}
	return out
}
