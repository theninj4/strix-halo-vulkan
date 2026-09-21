// Package vae implements Qwen-Image-2.1's autoencoder on the CPU — the
// 64-channel, 16x-spatial, RGBA-native Wan-lineage design — validated
// against reference/out/qi21vae stage by stage.
//
// The reference (reference/qwenimage21/autoencoder_kl_qwenimage21.py) is a
// *video* VAE run here at a single frame, and the port folds the temporal
// dimension away rather than carrying it. What "T=1" means, made explicit
// because each is a numeric behaviour the dumps pin:
//
//   - every causal 3D conv is a plain 2D conv (the reference's own image
//     specialization squeezes the frame axis and refuses a feature cache);
//   - the time_conv weights inside the 3D resamplers exist in the checkpoint
//     and never run — on the first chunk the reference skips them;
//   - AvgDown3D with a temporal factor zero-pads the missing frame *in
//     front* and means it in, so half of a temporal down-shortcut's
//     contributions are zeros. That is what the trained weights expect; it
//     is not a bug to fix;
//   - DupUp3D duplicates the frame temporally and `first_chunk` keeps only
//     the last copy, so only the spatial duplication survives.
//
// Norms are not the group norms of z-image's VAE: RMS_norm here is an L2
// normalize *across channels per pixel* (F.normalize over dim 1, eps 1e-12),
// scaled by sqrt(C) and a per-channel gamma. Every residual block is
// pre-norm silu-conv twice with a 1x1 (or identity) shortcut, and each
// down/up block carries a *parameter-free* resampling shortcut around its
// whole body — the "is_residual" design.
//
// Tensors are zimage/vae's NCHW [1, C, H, W] float32; that package's Conv2D
// (whose Pad/PadEnd/Stride were built for the identical (0,1,0,1)+stride-2
// downsampler) does the convolving.
package vae

import (
	"fmt"
	"math"
)

// ChannelNorm is the reference's RMS_norm: per pixel, the channel vector is
// L2-normalized (norm clamped below at 1e-12), scaled by sqrt(C) and a
// per-channel gamma. Mean is not subtracted.
type ChannelNorm struct {
	Gamma []float32
}

// Apply returns a normalized copy.
func (n *ChannelNorm) Apply(x *Tensor) (*Tensor, error) {
	if x.C != len(n.Gamma) {
		return nil, fmt.Errorf("qvae: channel norm of width %d over %d channels", len(n.Gamma), x.C)
	}
	out := NewTensor(x.N, x.C, x.H, x.W)
	scale := math.Sqrt(float64(x.C))
	plane := x.H * x.W
	for i := 0; i < plane; i++ {
		var sum float64
		for c := 0; c < x.C; c++ {
			v := float64(x.Data[c*plane+i])
			sum += v * v
		}
		s := scale / math.Max(math.Sqrt(sum), 1e-12)
		for c := 0; c < x.C; c++ {
			out.Data[c*plane+i] = float32(float64(x.Data[c*plane+i]) * s * float64(n.Gamma[c]))
		}
	}
	return out, nil
}

// ResBlock is one residual block: pre-norm silu conv, twice, plus a 1x1 (or
// identity) shortcut.
type ResBlock struct {
	Norm1, Norm2 ChannelNorm
	Conv1, Conv2 Conv2D
	Shortcut     *Conv2D // nil when in == out
}

func (b *ResBlock) Forward(x *Tensor) (*Tensor, error) {
	h := x
	if b.Shortcut != nil {
		var err error
		if h, err = b.Shortcut.Apply(x); err != nil {
			return nil, err
		}
	}
	t, err := b.Norm1.Apply(x)
	if err != nil {
		return nil, err
	}
	SiLUInPlace(t)
	if t, err = b.Conv1.Apply(t); err != nil {
		return nil, err
	}
	if t, err = b.Norm2.Apply(t); err != nil {
		return nil, err
	}
	SiLUInPlace(t)
	if t, err = b.Conv2.Apply(t); err != nil {
		return nil, err
	}
	if b.Shortcut != nil {
		return AddInPlace(t, h)
	}
	return AddInPlace(t, x)
}

// Attention is the mid block's single-head spatial self-attention: channel
// norm, a 1x1 qkv conv, softmax(q kᵀ/√C) v over the H*W tokens, a 1x1
// projection, and the residual.
type Attention struct {
	Norm      ChannelNorm
	QKV, Proj Conv2D
}

func (a *Attention) Forward(x *Tensor) (*Tensor, error) {
	normed, err := a.Norm.Apply(x)
	if err != nil {
		return nil, err
	}
	qkv, err := a.QKV.Apply(normed)
	if err != nil {
		return nil, err
	}
	C, plane := x.C, x.H*x.W
	q := qkv.Data[:C*plane]
	k := qkv.Data[C*plane : 2*C*plane]
	v := qkv.Data[2*C*plane:]
	scale := 1 / math.Sqrt(float64(C))

	ctx := NewTensor(1, C, x.H, x.W)
	scores := make([]float64, plane)
	for qi := 0; qi < plane; qi++ {
		max := math.Inf(-1)
		for ki := 0; ki < plane; ki++ {
			var dot float64
			for c := 0; c < C; c++ {
				dot += float64(q[c*plane+qi]) * float64(k[c*plane+ki])
			}
			scores[ki] = dot * scale
			if scores[ki] > max {
				max = scores[ki]
			}
		}
		var sum float64
		for ki := range scores {
			scores[ki] = math.Exp(scores[ki] - max)
			sum += scores[ki]
		}
		for c := 0; c < C; c++ {
			var acc float64
			for ki := 0; ki < plane; ki++ {
				acc += scores[ki] * float64(v[c*plane+ki])
			}
			ctx.Data[c*plane+qi] = float32(acc / sum)
		}
	}
	out, err := a.Proj.Apply(ctx)
	if err != nil {
		return nil, err
	}
	return AddInPlace(out, x)
}

// Mid is resnet, attention, resnet at the innermost resolution.
type Mid struct {
	Res1 ResBlock
	Attn Attention
	Res2 ResBlock
}

func (m *Mid) Forward(x *Tensor) (*Tensor, error) {
	x, err := m.Res1.Forward(x)
	if err != nil {
		return nil, err
	}
	if x, err = m.Attn.Forward(x); err != nil {
		return nil, err
	}
	return m.Res2.Forward(x)
}

// AvgDown is the parameter-free downsampling shortcut: space-to-channel at
// the spatial factor, the missing temporal frame zero-padded in front, and
// groups of channels averaged down to the output width. FactorT 1 and
// FactorS 1 make it the identity, which is what the last encoder block has.
type AvgDown struct {
	In, Out          int
	FactorT, FactorS int
}

func (d *AvgDown) Forward(x *Tensor) (*Tensor, error) {
	if x.C != d.In {
		return nil, fmt.Errorf("qvae: avg-down expects %d channels, got %d", d.In, x.C)
	}
	fs, ft := d.FactorS, d.FactorT
	factor := ft * fs * fs
	group := d.In * factor / d.Out
	outH, outW := x.H/fs, x.W/fs
	out := NewTensor(1, d.Out, outH, outW)
	// At T=1 a temporal factor of 2 pads one zero frame in front: temporal
	// slot 0 is the zero frame and only slot ft-1 holds the image.
	for o := 0; o < d.Out; o++ {
		dst := out.Plane(0, o)
		for g := 0; g < group; g++ {
			idx := o*group + g
			c := idx / factor
			rem := idx % factor
			kt := rem / (fs * fs)
			fh := rem / fs % fs
			fw := rem % fs
			if kt != ft-1 {
				continue // the zero-padded frame contributes nothing
			}
			src := x.Plane(0, c)
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					dst[oh*outW+ow] += src[(oh*fs+fh)*x.W+(ow*fs+fw)]
				}
			}
		}
		inv := 1 / float32(group)
		for i := range dst {
			dst[i] *= inv
		}
	}
	return out, nil
}

// DupUp is the parameter-free upsampling shortcut: channels replicated and
// redistributed channel-to-space. At T=1 the reference expands the frame
// axis by FactorT and `first_chunk` keeps only the last copy, so only the
// spatial duplication survives — but *which* source channel feeds an output
// pixel still depends on FactorT through the replication layout.
type DupUp struct {
	In, Out int
	FactorT int
}

func (u *DupUp) Forward(x *Tensor) (*Tensor, error) {
	if x.C != u.In {
		return nil, fmt.Errorf("qvae: dup-up expects %d channels, got %d", u.In, x.C)
	}
	const fs = 2
	factor := u.FactorT * fs * fs
	repeats := u.Out * factor / u.In
	out := NewTensor(1, u.Out, x.H*fs, x.W*fs)
	for o := 0; o < u.Out; o++ {
		dst := out.Plane(0, o)
		for i := 0; i < fs; i++ {
			for j := 0; j < fs; j++ {
				flat := ((o*u.FactorT+(u.FactorT-1))*fs+i)*fs + j
				src := x.Plane(0, flat/repeats)
				for h := 0; h < x.H; h++ {
					for w := 0; w < x.W; w++ {
						dst[(h*fs+i)*out.W+(w*fs+j)] = src[h*x.W+w]
					}
				}
			}
		}
	}
	return out, nil
}

// Resample is the learned half of a down/up block's resampling: nearest 2x
// then a 3x3 conv going up, a (0,1,0,1) zero-pad and a stride-2 3x3 going
// down. The 3D variants' time_conv never runs at T=1.
type Resample struct {
	Up   bool
	Conv Conv2D
}

func (r *Resample) Forward(x *Tensor) (*Tensor, error) {
	if r.Up {
		return r.Conv.Apply(UpsampleNearest2x(x))
	}
	return r.Conv.Apply(x)
}

// DownBlock is one encoder stage: the residual blocks and learned
// downsampler, plus the AvgDown shortcut around the whole body.
type DownBlock struct {
	Resnets  []ResBlock
	Down     *Resample // nil on the last block
	Shortcut AvgDown
}

func (b *DownBlock) Forward(x *Tensor) (*Tensor, error) {
	short, err := b.Shortcut.Forward(x)
	if err != nil {
		return nil, err
	}
	for i := range b.Resnets {
		if x, err = b.Resnets[i].Forward(x); err != nil {
			return nil, err
		}
	}
	if b.Down != nil {
		if x, err = b.Down.Forward(x); err != nil {
			return nil, err
		}
	}
	return AddInPlace(x, short)
}

// UpBlock is one decoder stage: the residual blocks and learned upsampler,
// plus the DupUp shortcut. The last block has neither.
type UpBlock struct {
	Resnets  []ResBlock
	Up       *Resample
	Shortcut *DupUp
}

func (b *UpBlock) Forward(x *Tensor) (*Tensor, error) {
	in := x
	var err error
	for i := range b.Resnets {
		if x, err = b.Resnets[i].Forward(x); err != nil {
			return nil, err
		}
	}
	if b.Up != nil {
		if x, err = b.Up.Forward(x); err != nil {
			return nil, err
		}
	}
	if b.Shortcut != nil {
		short, err := b.Shortcut.Forward(in)
		if err != nil {
			return nil, err
		}
		return AddInPlace(x, short)
	}
	return x, nil
}

// Decoder decodes denormalized latents [1, z, h, w] to an RGBA image
// [1, 4, 16h, 16w] in [-1, 1] (the reference clamps).
type Decoder struct {
	PostQuant Conv2D // 1x1, z -> z
	ConvIn    Conv2D
	Mid       Mid
	Ups       []UpBlock
	NormOut   ChannelNorm
	ConvOut   Conv2D

	// Tap, when set, sees each stage's output by name — the stagewise test
	// walks the dump's hooks with it. Nil costs nothing.
	Tap func(name string, t *Tensor)
}

func (d *Decoder) tap(name string, t *Tensor) {
	if d.Tap != nil {
		d.Tap(name, t)
	}
}

// Decode runs the decoder.
func (d *Decoder) Decode(z *Tensor) (*Tensor, error) {
	x, err := d.PostQuant.Apply(z)
	if err != nil {
		return nil, err
	}
	if x, err = d.ConvIn.Apply(x); err != nil {
		return nil, err
	}
	d.tap("conv_in", x)
	if x, err = d.Mid.Forward(x); err != nil {
		return nil, err
	}
	d.tap("mid_block", x)
	for i := range d.Ups {
		if x, err = d.Ups[i].Forward(x); err != nil {
			return nil, fmt.Errorf("qvae: up block %d: %w", i, err)
		}
		d.tap(fmt.Sprintf("up_blocks.%d", i), x)
	}
	if x, err = d.NormOut.Apply(x); err != nil {
		return nil, err
	}
	d.tap("norm_out", x)
	SiLUInPlace(x)
	d.tap("nonlinearity", x)
	if x, err = d.ConvOut.Apply(x); err != nil {
		return nil, err
	}
	d.tap("conv_out", x)
	for i, v := range x.Data {
		if v > 1 {
			x.Data[i] = 1
		} else if v < -1 {
			x.Data[i] = -1
		}
	}
	return x, nil
}

// Encoder encodes an RGBA image [1, 4, H, W] in [-1, 1] to the posterior
// *mode* [1, z, H/16, W/16] — the pipeline's argmax path; sampling would
// break seed reproducibility and is not implemented.
type Encoder struct {
	ConvIn  Conv2D
	Downs   []DownBlock
	Mid     Mid
	NormOut ChannelNorm
	ConvOut Conv2D // -> 2z: mean then logvar
	Quant   Conv2D // 1x1, 2z -> 2z

	Tap func(name string, t *Tensor)
}

func (e *Encoder) tap(name string, t *Tensor) {
	if e.Tap != nil {
		e.Tap(name, t)
	}
}

// Encode returns the posterior mode: the mean half of the quantized output.
func (e *Encoder) Encode(img *Tensor) (*Tensor, error) {
	x, err := e.ConvIn.Apply(img)
	if err != nil {
		return nil, err
	}
	e.tap("conv_in", x)
	for i := range e.Downs {
		if x, err = e.Downs[i].Forward(x); err != nil {
			return nil, fmt.Errorf("qvae: down block %d: %w", i, err)
		}
		e.tap(fmt.Sprintf("down_blocks.%d", i), x)
	}
	if x, err = e.Mid.Forward(x); err != nil {
		return nil, err
	}
	e.tap("mid_block", x)
	if x, err = e.NormOut.Apply(x); err != nil {
		return nil, err
	}
	e.tap("norm_out", x)
	SiLUInPlace(x)
	e.tap("nonlinearity", x)
	if x, err = e.ConvOut.Apply(x); err != nil {
		return nil, err
	}
	e.tap("conv_out", x)
	if x, err = e.Quant.Apply(x); err != nil {
		return nil, err
	}
	mode := NewTensor(1, x.C/2, x.H, x.W)
	copy(mode.Data, x.Data[:len(mode.Data)])
	return mode, nil
}
