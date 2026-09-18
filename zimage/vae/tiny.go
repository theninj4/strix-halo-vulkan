package vae

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
)

// madebyollin/taef1 -- the preview decoder, IMAGE.md I2.
//
// It is in this package rather than beside it because it is the *same*
// operator set as the AutoencoderKL decoder with the expensive half removed:
// 3x3 convolutions, an upsample, a residual add. No group norms and no
// attention, which is most of what makes the full decoder 0.81 s. So it
// reuses Conv2D, Tensor and, on the device, stage 8's implicit-GEMM
// convolution unmodified -- the only kernel this file adds is a ReLU.
//
// Two conventions, both settled by reference/dump_taef1.py --from-run against
// a real z-image trajectory rather than by the model card:
//
//   - **It takes the raw diffusion latent**, not the `(z/scale + shift)` the
//     full decoder is handed. Its own config says scaling_factor 1.0 and
//     shift_factor 0.0, and the measurement agrees: on the final latent of a
//     256x256 run the raw latent lands 2.0% from the full VAE's image and the
//     unscaled one 6.7%, a 3.3x separation. It is worth knowing that this is
//     *not* visible by eye -- the decoder opens with tanh(x/3)*3, so a latent
//     2.8x too large is squashed rather than blown out, and both conventions
//     produce a recognisable picture.
//   - **A preview decodes the denoised estimate, not the state the loop
//     holds.** See pipeline.Step.
//
// Sizes: 1.23 M parameters in the decoder against the full VAE's 49.55 M, but
// the flop count is the number that decides what a preview costs, and there
// the ratio is 18.5x rather than 40x -- 566 GFLOP at 1024x1024 against
// 10.5 TFLOP -- because taef1 does its work at full resolution where the big
// decoder has already narrowed to 128 channels.

// TinyConfig mirrors the parts of taef1's config.json the decoder needs. As
// with FluxConfig it is spelled out rather than parsed, so a checkpoint that
// disagrees is a load error rather than a silent reshape.
type TinyConfig struct {
	LatentChannels int
	OutChannels    int
	BlockChannels  []int
	NumBlocks      []int
	UpsampleFactor int
	// LatentMagnitude is the 3 in tanh(x/3)*3. diffusers spells it as a
	// config field used by scale_latents, and the same number is the clamp's;
	// it is here so the two cannot drift.
	LatentMagnitude float64
}

// TAEF1Config is madebyollin/taef1's configuration.
func TAEF1Config() TinyConfig {
	return TinyConfig{
		LatentChannels:  16,
		OutChannels:     3,
		BlockChannels:   []int{64, 64, 64, 64},
		NumBlocks:       []int{3, 3, 3, 1},
		UpsampleFactor:  2,
		LatentMagnitude: 3,
	}
}

// TinyBlock is diffusers' AutoencoderTinyBlock: three convolutions with a ReLU
// between them, added to the input and passed through a fourth ReLU. The skip
// is an identity here and always will be -- every block in taef1 is 64 to 64
// -- so the 1x1 projection diffusers builds for a channel change is not
// ported. A checkpoint that needed one would fail to load rather than be
// quietly wrong.
type TinyBlock struct {
	Conv0, Conv2, Conv4 *Conv2D
}

// Apply runs the block. x is not modified.
func (b *TinyBlock) Apply(x *Tensor) (*Tensor, error) {
	h, err := b.Conv0.Apply(x)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(h)
	if h, err = b.Conv2.Apply(h); err != nil {
		return nil, err
	}
	ReLUInPlace(h)
	if h, err = b.Conv4.Apply(h); err != nil {
		return nil, err
	}
	if h, err = AddInPlace(h, x); err != nil {
		return nil, err
	}
	return ReLUInPlace(h), nil
}

// TinyLayer is one entry of the decoder's flat Sequential. Exactly one field
// is set. The graph is a straight line of nineteen of these, and keeping it
// flat -- rather than grouping it into up-blocks the way the full decoder is
// -- is what lets a stagewise test address a tensor by the same index the
// checkpoint names it with.
type TinyLayer struct {
	// Index is the layer's position in diffusers' Sequential, i.e. the number
	// in "decoder.layers.12".
	Index int
	Conv  *Conv2D
	Block *TinyBlock
	// Upsample is the nearest 2x, which has no weights.
	Upsample bool
	// ReLU is layer 1, the activation after conv_in. It is a layer of the
	// Sequential rather than part of the convolution, so it is one here too.
	ReLU bool
}

// TinyDecoder is taef1's decoder: a latent in, an image in [-1, 1] out.
type TinyDecoder struct {
	Cfg    TinyConfig
	Layers []TinyLayer
}

// Scale is how much larger the image is than the latent: 2 per upsample, and
// there is one after every block group but the last.
func (d *TinyDecoder) Scale() int {
	s := 1
	for _, l := range d.Layers {
		if l.Upsample {
			s *= d.Cfg.UpsampleFactor
		}
	}
	return s
}

// Apply decodes a latent.
//
// The latent is the pipeline's own, in the diffusion space -- see the package
// note above. The output is [3, H*8, W*8] in [-1, 1], which is the same range
// the full decoder produces, so backend.toRGBA maps it with no second case.
func (d *TinyDecoder) Apply(latent *Tensor) (*Tensor, error) {
	if latent.C != d.Cfg.LatentChannels {
		return nil, fmt.Errorf("vae: taef1 takes %d latent channels, got %d",
			d.Cfg.LatentChannels, latent.C)
	}
	// The clamp, which is in DecoderTiny.forward rather than in the layers.
	h := &Tensor{N: latent.N, C: latent.C, H: latent.H, W: latent.W,
		Data: make([]float32, latent.Len())}
	mag := float32(d.Cfg.LatentMagnitude)
	for i, v := range latent.Data {
		h.Data[i] = tanh32(v/mag) * mag
	}

	var err error
	for _, l := range d.Layers {
		switch {
		case l.Conv != nil:
			if h, err = l.Conv.Apply(h); err != nil {
				return nil, fmt.Errorf("vae: taef1 layers.%d: %w", l.Index, err)
			}
		case l.Block != nil:
			if h, err = l.Block.Apply(h); err != nil {
				return nil, fmt.Errorf("vae: taef1 layers.%d: %w", l.Index, err)
			}
		case l.Upsample:
			h = UpsampleNearest2x(h)
		case l.ReLU:
			ReLUInPlace(h)
		}
	}
	// [0, 1] to [-1, 1], diffusers' convention and this pipeline's.
	for i, v := range h.Data {
		h.Data[i] = v*2 - 1
	}
	return h, nil
}

// LoadTinyDecoder reads taef1's decoder out of a checkpoint directory. The
// encoder's 1.23 M parameters are in the same file and are not read: an
// encoder is what /v1/images/edits wants (I7) and it is the *full* VAE's that
// an edit needs, not a preview decoder's.
func LoadTinyDecoder(dir string, cfg TinyConfig) (*TinyDecoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}
	d := &TinyDecoder{Cfg: cfg}

	idx := 0
	next := func() int { i := idx; idx++; return i }
	conv := func(i int) *Conv2D {
		c := l.convOptBias(fmt.Sprintf("decoder.layers.%d", i), 1)
		// Every convolution in this decoder is 3x3 pad 1 and the packed
		// layout's border assumes it, so it is checked rather than trusted.
		if l.err == nil && (c.KH != 3 || c.KW != 3) {
			l.err = fmt.Errorf("vae: taef1 layers.%d is %dx%d, want 3x3", i, c.KH, c.KW)
		}
		return c
	}

	i := next()
	d.Layers = append(d.Layers, TinyLayer{Index: i, Conv: conv(i)})
	d.Layers = append(d.Layers, TinyLayer{Index: next(), ReLU: true})

	for gi, n := range cfg.NumBlocks {
		last := gi == len(cfg.NumBlocks)-1
		for j := 0; j < n; j++ {
			bi := next()
			p := fmt.Sprintf("decoder.layers.%d", bi)
			b := &TinyBlock{
				Conv0: l.conv(p+".conv.0", 1),
				Conv2: l.conv(p+".conv.2", 1),
				Conv4: l.conv(p+".conv.4", 1),
			}
			if l.err == nil && set.Has(p+".skip.weight") {
				l.err = fmt.Errorf("vae: taef1 layers.%d has a skip projection; "+
					"every block in this checkpoint is %d to %d and the port assumes it", bi, b.Conv0.InC, b.Conv0.OutC)
			}
			d.Layers = append(d.Layers, TinyLayer{Index: bi, Block: b})
		}
		if !last {
			d.Layers = append(d.Layers, TinyLayer{Index: next(), Upsample: true})
		}
		ci := next()
		d.Layers = append(d.Layers, TinyLayer{Index: ci, Conv: conv(ci)})
	}
	if l.err != nil {
		return nil, l.err
	}

	// The shapes the config claims, against the shapes the file holds. Both
	// ends are checked because both are load-bearing: the latent channel count
	// is what the pipeline hands it and the output channel count is what a PNG
	// writer assumes.
	first, lastConv := d.Layers[0].Conv, d.Layers[len(d.Layers)-1].Conv
	if first.InC != cfg.LatentChannels {
		return nil, fmt.Errorf("vae: taef1 layers.0 takes %d channels, config says %d latent channels",
			first.InC, cfg.LatentChannels)
	}
	if first.OutC != cfg.BlockChannels[0] {
		return nil, fmt.Errorf("vae: taef1 layers.0 produces %d channels, config says %d",
			first.OutC, cfg.BlockChannels[0])
	}
	if lastConv.OutC != cfg.OutChannels {
		return nil, fmt.Errorf("vae: taef1's last layer produces %d channels, want %d",
			lastConv.OutC, cfg.OutChannels)
	}
	if cfg.UpsampleFactor != 2 {
		return nil, fmt.Errorf("vae: taef1 upsamples %dx; UpsampleNearest2x does 2", cfg.UpsampleFactor)
	}
	return d, nil
}

// TinyParams is the decoder's parameter count, for a residency line.
func (d *TinyDecoder) TinyParams() int {
	n := 0
	count := func(c *Conv2D) {
		n += len(c.Weight) + len(c.Bias)
	}
	for _, l := range d.Layers {
		switch {
		case l.Conv != nil:
			count(l.Conv)
		case l.Block != nil:
			count(l.Block.Conv0)
			count(l.Block.Conv2)
			count(l.Block.Conv4)
		}
	}
	return n
}

// ReLUInPlace applies max(x, 0).
func ReLUInPlace(x *Tensor) *Tensor {
	parallelFor(x.N*x.C, func(p int) {
		plane := x.Plane(p/x.C, p%x.C)
		for i, v := range plane {
			if v < 0 {
				plane[i] = 0
			}
		}
	})
	return x
}
