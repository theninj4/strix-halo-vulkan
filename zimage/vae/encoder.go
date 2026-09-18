package vae

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// The AutoencoderKL *encoder*: RGB in [-1, 1] to a 16-channel latent at an
// eighth of the resolution -- IMAGE.md I7.
//
// It is the decoder's mirror and it is written here as one, sharing
// ResnetBlock, Attention, MidBlock, GroupNorm, SiLU and Conv2D with it rather
// than restating any of them. Two things are its own:
//
//   - **The downsampler.** diffusers' Downsample2D pads (0, 1, 0, 1) and then
//     convolves 3x3 with stride 2 and no padding, which is the one shape
//     Conv2D did not have. Both halves of it are in Conv2D now (Stride,
//     PadEnd); see also builder.downsample, which runs the *same filter* at
//     stride 1 on the device and throws three pixels in four away, because
//     that needs no new matrix-core kernel and costs 18 ms an image.
//   - **The DiagonalGaussian head.** conv_out produces 32 channels, which are
//     16 of mean and 16 of log-variance. Encode returns the mean -- the
//     distribution's *mode* -- and never reads the variance. That is a
//     decision and not an omission: sampling would make the same picture
//     encode to a different latent every time, and an edit that cannot be
//     reproduced from its seed is not much use. reference/encode_image.py
//     takes `.mode()` for the same reason and is the oracle.
//
// The caller is handed the VAE's own latent. Turning that into the
// transformer's is `(z - shift_factor) * scaling_factor`, which is
// Config.ToDiffusion -- the exact inverse of what Pipeline.Run undoes before
// the decode.

// Encoder is the AutoencoderKL encoder: conv_in, four down blocks (three of
// which halve the resolution), a mid block with attention between two
// resnets, then a group norm, SiLU and a 32-channel output convolution.
type Encoder struct {
	ConvIn      *Conv2D
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
	return e.ConvOut.Apply(h)
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

// ToDiffusion maps the VAE's own latent into the one the transformer
// denoises, in place: (z - shift_factor) * scaling_factor.
//
// FromDiffusion is the inverse, and it is the arithmetic Pipeline.Run already
// does in front of the decoder. They are here as a pair so that an edit
// cannot pick up one convention and the decode the other -- which is the bug
// this shape invites, and which produces a plausible washed-out image rather
// than an error.
func (c Config) ToDiffusion(z []float32) {
	for i, v := range z {
		z[i] = float32((float64(v) - c.ShiftFactor) * c.ScalingFactor)
	}
}

// FromDiffusion is ToDiffusion's inverse: z/scale + shift.
func (c Config) FromDiffusion(z []float32) {
	for i, v := range z {
		z[i] = float32(float64(v)/c.ScalingFactor + c.ShiftFactor)
	}
}

// LoadEncoder reads the encoder weights out of a VAE checkpoint directory.
// It is the same file LoadDecoder opens -- 106 tensors of encoder beside the
// decoder's 138 -- and a process that wants both pays two mappings and one
// copy each, which at 68 MB is not worth a combined loader.
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
