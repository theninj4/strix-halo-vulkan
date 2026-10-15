package vae

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// AutoencoderKLFlux2's *encoder*: RGB in [-1, 1] to the 64 moments of a
// 32-channel diagonal gaussian at an eighth of the resolution. It is
// zimage/vae's FLUX.1 encoder (z-image I7, restored from 77f6627^) with the
// 1x1 quant_conv after conv_out.
//
// It shares ResnetBlock, Attention, MidBlock, GroupNorm, SiLU and Conv2D with
// the decoder. Its own pieces are the downsampler — diffusers' Downsample2D
// pads (0, 1, 0, 1) and then convolves 3x3 at stride 2 with no padding
// (Conv2D's PadEnd and Stride) — and the gaussian head: Encode returns the
// mean, the distribution's *mode*, as diffusers' Flux3VaeEncoderStep does
// with latent_dist.mode(); sampling would make an edit irreproducible.
//
// The caller is handed the VAE's own latent; pipeline.BN.FromVAE turns it
// into the DiT's tokens (patchify, BatchNorm).

// Encoder is the AutoencoderKL encoder: conv_in, four down blocks (three of
// which halve the resolution), a mid block with attention between two
// resnets, then a group norm, SiLU and a 32-channel output convolution.
type Encoder struct {
	ConvIn      *Conv2D
	QuantConv   *Conv2D // 1x1 over the moments, after conv_out
	DownBlocks  []*DownBlock
	Mid         *MidBlock
	ConvNormOut *GroupNorm
	ConvOut     *Conv2D
	// LatentChannels is half of ConvOut's output, i.e. the width of the mean
	// once the log-variance has been dropped.
	LatentChannels int
}

// DownBlock is a run of resnets followed by an optional stride-2 convolution.
// It is UpBlock read backwards, with the resolution change *after* the
// resnets rather than after them and through an upsample.
type DownBlock struct {
	Resnets     []*ResnetBlock
	Downsampler *Conv2D // nil on the last block
}

// Apply runs the block.
func (d *DownBlock) Apply(x *Tensor) (*Tensor, error) {
	var err error
	h := x
	for _, r := range d.Resnets {
		if h, err = r.Apply(h); err != nil {
			return nil, err
		}
	}
	if d.Downsampler == nil {
		return h, nil
	}
	return d.Downsampler.Apply(h)
}

// Apply encodes an image to the distribution's parameters: [1, 2*C, H/8, W/8]
// holding the mean and the log-variance. Encode is what a caller normally
// wants; this is here because it is the tensor the reference dumps.
func (e *Encoder) Apply(img *Tensor) (*Tensor, error) {
	h, err := e.ConvIn.Apply(img)
	if err != nil {
		return nil, err
	}
	for i, down := range e.DownBlocks {
		if h, err = down.Apply(h); err != nil {
			return nil, fmt.Errorf("vae: down block %d: %w", i, err)
		}
	}
	if h, err = e.Mid.Apply(h); err != nil {
		return nil, err
	}
	if h, err = e.ConvNormOut.ApplyInPlace(h); err != nil {
		return nil, err
	}
	SiLUInPlace(h)
	if h, err = e.ConvOut.Apply(h); err != nil {
		return nil, err
	}
	return e.QuantConv.Apply(h)
}

// Encode runs the encoder and takes the posterior's mode, which is its mean:
// the first half of Apply's channels.
func (e *Encoder) Encode(img *Tensor) (*Tensor, error) {
	moments, err := e.Apply(img)
	if err != nil {
		return nil, err
	}
	return Mode(moments, e.LatentChannels)
}

// Mode drops the log-variance, returning the first c channels of a
// DiagonalGaussian's parameters.
func Mode(moments *Tensor, c int) (*Tensor, error) {
	if moments.C != 2*c {
		return nil, fmt.Errorf("vae: %d moment channels for a %d-channel latent, want %d", moments.C, c, 2*c)
	}
	out := NewTensor(moments.N, c, moments.H, moments.W)
	for n := 0; n < moments.N; n++ {
		for ch := 0; ch < c; ch++ {
			copy(out.Plane(n, ch), moments.Plane(n, ch))
		}
	}
	return out, nil
}

// LoadEncoder reads the encoder weights out of a VAE checkpoint directory.
// It is the same file LoadDecoder opens; a process that wants both pays two
// mappings and one copy each, which for a 0.34 GB file is not worth a
// combined loader.
func LoadEncoder(dir string, cfg Config) (*Encoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	const eps = 1e-6
	l := &loader{set: set}

	e := &Encoder{
		ConvIn:         l.conv("encoder.conv_in", 1),
		ConvNormOut:    l.groupNorm("encoder.conv_norm_out", cfg.NormNumGroups, eps),
		ConvOut:        l.conv("encoder.conv_out", 1),
		QuantConv:      l.conv("quant_conv", 0),
		LatentChannels: cfg.LatentChannels,
	}
	e.Mid = &MidBlock{
		Resnet1: l.resnet("encoder.mid_block.resnets.0", cfg.NormNumGroups, eps),
		Resnet2: l.resnet("encoder.mid_block.resnets.1", cfg.NormNumGroups, eps),
	}
	attnPrefix := "encoder.mid_block.attentions.0"
	q := l.linear(attnPrefix + ".to_q")
	if l.err != nil {
		return nil, l.err
	}
	e.Mid.Attn = &Attention{
		GroupNorm: l.groupNorm(attnPrefix+".group_norm", cfg.NormNumGroups, eps),
		Q:         q,
		K:         l.linear(attnPrefix + ".to_k"),
		V:         l.linear(attnPrefix + ".to_v"),
		Out:       l.linear(attnPrefix + ".to_out.0"),
		Heads:     1,
		Scale:     1 / math.Sqrt(float64(q.Out)),
	}

	// The encoder walks block_out_channels forwards, unlike the decoder, and
	// has layers_per_block resnets per block rather than one more. Every
	// block but the last downsamples.
	nBlocks := len(cfg.BlockOutChannels)
	for i := 0; i < nBlocks; i++ {
		prefix := fmt.Sprintf("encoder.down_blocks.%d", i)
		down := &DownBlock{}
		for j := 0; j < cfg.LayersPerBlock; j++ {
			down.Resnets = append(down.Resnets, l.resnet(fmt.Sprintf("%s.resnets.%d", prefix, j), cfg.NormNumGroups, eps))
		}
		if set.Has(prefix + ".downsamplers.0.conv.weight") {
			c := l.conv(prefix+".downsamplers.0.conv", 0)
			if l.err == nil {
				// diffusers' Downsample2D with padding=0: F.pad(0,1,0,1) and
				// then stride 2. Said here rather than in the loader, because
				// it is this module's shape and not a property of the file.
				c.PadEnd, c.Stride = 1, 2
			}
			down.Downsampler = c
		} else if i != nBlocks-1 {
			l.err = fmt.Errorf("vae: down block %d has no downsampler but is not the last of %d", i, nBlocks)
		}
		e.DownBlocks = append(e.DownBlocks, down)
	}
	if l.err != nil {
		return nil, l.err
	}
	if e.ConvIn.InC != 3 {
		return nil, fmt.Errorf("vae: encoder conv_in takes %d channels, want 3", e.ConvIn.InC)
	}
	if e.ConvOut.OutC != 2*cfg.LatentChannels {
		return nil, fmt.Errorf("vae: encoder conv_out produces %d channels, want %d for a %d-channel diagonal gaussian",
			e.ConvOut.OutC, 2*cfg.LatentChannels, cfg.LatentChannels)
	}
	return e, nil
}
