package vae

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// TAEQI2.1, the preview decoder — the answer to Q-o4.
//
// madebyollin/taesd shipped `taeqi2_1_decoder.pth` on 2026-09-25 (commit
// 401ce45e, taesd issue #38): a tiny decoder distilled against *this* VAE,
// 64 channels in, 16x, RGBA out. It replaces Q7's fitted 64x4 matrix, which
// was only ever standing in for it. reference/taesd/taesd.py's F16Decoder is
// the source and reference/dump_taeqi.py the oracle; the .pth is converted to
// safetensors there, keys unchanged.
//
// It is in this package rather than beside it for the reason taef1 was in
// zimage/vae: it is the *same* operator set as the full decoder with the
// expensive half removed — 3x3 convolutions, a nearest 2x upsample, a
// residual add — so it reuses Conv2D here and the conv builds on the device.
// What it adds is a ReLU, the tanh clamp on its input, and a 2x2 pixel
// shuffle on its output, which is how it reaches 16x with three upsamples.
//
// Two conventions, both measured rather than read off a card:
//
//   - **It takes the normalized latent**, the space the DiT works in, not the
//     denormalized one the full decoder is handed. Through the real VAE
//     encoder on a served picture the normalized latent decodes rms 0.035
//     from the full VAE's own decode, the raw one 0.198.
//   - **Its output is [0, 1]**, taesd's convention; Decode moves it to the
//     pipeline's [-1, 1] so a preview and a finished image are the same type.
//
// Cost: 7.6 M parameters, and at 1024² ~230 GFLOP, three quarters of it in
// the last two stages at 256² and 512² where the width is 64.

// TinyLatentMagnitude is the 3 in the input clamp tanh(x/3)*3.
const TinyLatentMagnitude = 3

// TinyScale is how much larger the image is than the latent.
const TinyScale = 16

// TinyBlock is taesd's Block for n_in == n_out: three 3x3 convolutions with a
// ReLU between them, added to the input, then a ReLU. Every block in the
// checkpoint keeps its width — the width changes happen in the bias-free
// convolution after each upsample — so the 1x1 skip taesd builds for a width
// change is not ported, and a checkpoint that needed one fails to load.
type TinyBlock struct {
	Conv0, Conv2, Conv4 Conv2D
}

// TinyStage is one resolution of the decoder: its blocks, then (except for
// the last) a nearest 2x upsample and the bias-free convolution that narrows
// the width.
type TinyStage struct {
	Blocks []TinyBlock
	Up     *Conv2D
}

// TinyDecoder is TAEQI2.1's decoder.
type TinyDecoder struct {
	LatentChannels int
	// ConvIn is layer 1, followed by a ReLU.
	ConvIn Conv2D
	Stages []TinyStage
	// ConvOut produces 4 * 4 channels, which the pixel shuffle folds into a
	// 2x2 block of RGBA.
	ConvOut Conv2D
}

// tinyLayout is F16Decoder's Sequential, as (block count, width) per stage:
// Clamp, conv, ReLU, then these stages each followed by Upsample + conv
// except the last, then conv and PixelShuffle. The layer indices the
// checkpoint names follow from it.
var tinyLayout = []struct{ blocks, width int }{{3, 256}, {3, 128}, {3, 64}, {1, 64}}

// LoadTinyDecoder reads models/taeqi2_1's converted decoder.
func LoadTinyDecoder(dir string) (*TinyDecoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, fmt.Errorf("qvae: tiny decoder: %w (convert the .pth with reference/dump_taeqi.py)", err)
	}
	defer set.Close()
	l := &loader{set: set}

	const z, rgba = 64, 4
	d := &TinyDecoder{LatentChannels: z}
	idx := 1 // 0 is the clamp
	d.ConvIn = l.conv(fmt.Sprint(idx), tinyLayout[0].width, z, 3, 1, 0, 1)
	idx += 2 // the ReLU
	for s, st := range tinyLayout {
		stage := TinyStage{}
		for range st.blocks {
			p := fmt.Sprint(idx)
			stage.Blocks = append(stage.Blocks, TinyBlock{
				Conv0: l.conv(p+".conv.0", st.width, st.width, 3, 1, 0, 1),
				Conv2: l.conv(p+".conv.2", st.width, st.width, 3, 1, 0, 1),
				Conv4: l.conv(p+".conv.4", st.width, st.width, 3, 1, 0, 1),
			})
			idx++
		}
		if s < len(tinyLayout)-1 {
			idx++ // the upsample
			up := l.convNoBias(fmt.Sprint(idx), tinyLayout[s+1].width, st.width, 3, 1)
			stage.Up = &up
			idx++
		}
		d.Stages = append(d.Stages, stage)
	}
	d.ConvOut = l.conv(fmt.Sprint(idx), rgba*4, tinyLayout[len(tinyLayout)-1].width, 3, 1, 0, 1)
	if l.err != nil {
		return nil, fmt.Errorf("qvae: tiny decoder: %w", l.err)
	}
	if n := len(set.Names()); n != d.tensorCount() {
		return nil, fmt.Errorf("qvae: tiny decoder checkpoint has %d tensors, the layout reads %d", n, d.tensorCount())
	}
	return d, nil
}

// convNoBias reads a bias-free Conv2d, refusing one that has a bias: a bias
// the layout does not expect is a checkpoint the layout does not describe.
func (l *loader) convNoBias(name string, out, in, k, pad int) Conv2D {
	if l.err != nil {
		return Conv2D{}
	}
	if _, err := l.set.Get(name + ".bias"); err == nil {
		l.err = fmt.Errorf("qvae: %s has a bias the layout does not expect", name)
		return Conv2D{}
	}
	t, err := l.set.Get(name + ".weight")
	if err != nil {
		l.err = err
		return Conv2D{}
	}
	if sh := t.Shape; len(sh) != 4 || sh[0] != out || sh[1] != in || sh[2] != k || sh[3] != k {
		l.err = fmt.Errorf("qvae: %s.weight is %v, want [%d %d %d %d]", name, sh, out, in, k, k)
		return Conv2D{}
	}
	return Conv2D{InC: in, OutC: out, KH: k, KW: k, Pad: pad, Weight: l.f32(name+".weight", out*in*k*k)}
}

// tensorCount is how many checkpoint tensors the layout reads.
func (d *TinyDecoder) tensorCount() int {
	n := 2 + 2 // conv_in, conv_out
	for _, s := range d.Stages {
		n += 6 * len(s.Blocks)
		if s.Up != nil {
			n++
		}
	}
	return n
}

// TinyClamp is the decoder's first layer, tanh(x/3)*3, applied in place. It
// is a function of one element and the latent is small, so both the CPU and
// the GPU decoder run it on the host.
func TinyClamp(z []float32) {
	for i, v := range z {
		z[i] = float32(math.Tanh(float64(v)/TinyLatentMagnitude) * TinyLatentMagnitude)
	}
}

// ReLUInPlace applies max(x, 0).
func ReLUInPlace(x *Tensor) *Tensor {
	for i, v := range x.Data {
		if v < 0 {
			x.Data[i] = 0
		}
	}
	return x
}

// PixelShuffle2x folds [N, 4C, H, W] into [N, C, 2H, 2W] the way
// nn.PixelShuffle(2) does: channel c*4 + i*2 + j lands at row offset i and
// column offset j.
func PixelShuffle2x(x *Tensor) (*Tensor, error) {
	if x.C%4 != 0 {
		return nil, fmt.Errorf("qvae: pixel shuffle needs a multiple of 4 channels, got %d", x.C)
	}
	out := NewTensor(x.N, x.C/4, x.H*2, x.W*2)
	for n := 0; n < x.N; n++ {
		for c := 0; c < out.C; c++ {
			dst := out.Plane(n, c)
			for k := 0; k < 4; k++ {
				i, j := k/2, k%2
				src := x.Plane(n, c*4+k)
				for h := 0; h < x.H; h++ {
					drow := dst[(2*h+i)*out.W:]
					for w, v := range src[h*x.W : (h+1)*x.W] {
						drow[2*w+j] = v
					}
				}
			}
		}
	}
	return out, nil
}

// TinyToImage maps the decoder's pre-shuffle output — [1, 16, h, w], taesd's
// [0, 1] convention — to the pipeline's image: shuffled to [1, 4, 2h, 2w],
// clamped and moved to [-1, 1]. Both decoders end here.
func TinyToImage(x *Tensor) (*Tensor, error) {
	img, err := PixelShuffle2x(x)
	if err != nil {
		return nil, err
	}
	for i, v := range img.Data {
		img.Data[i] = 2*min(max(v, 0), 1) - 1
	}
	return img, nil
}

// Apply runs one block. x is not modified.
func (b *TinyBlock) Apply(x *Tensor) (*Tensor, error) {
	h, err := b.Conv0.Apply(x)
	if err != nil {
		return nil, err
	}
	if h, err = b.Conv2.Apply(ReLUInPlace(h)); err != nil {
		return nil, err
	}
	if h, err = b.Conv4.Apply(ReLUInPlace(h)); err != nil {
		return nil, err
	}
	if h, err = AddInPlace(h, x); err != nil {
		return nil, err
	}
	return ReLUInPlace(h), nil
}

// Decode runs the decoder on the CPU: a normalized latent [1, 64, h, w] in,
// an RGBA image [1, 4, 16h, 16w] in [-1, 1] out. It is the GPU decoder's
// oracle; tap, when set, sees each stage's output under the name the dump
// gives the Sequential layer that produced it.
func (d *TinyDecoder) Decode(z *Tensor, tap func(layer int, t *Tensor)) (*Tensor, error) {
	if z.C != d.LatentChannels {
		return nil, fmt.Errorf("qvae: tiny decoder takes %d latent channels, got %d", d.LatentChannels, z.C)
	}
	if tap == nil {
		tap = func(int, *Tensor) {}
	}
	x := &Tensor{N: z.N, C: z.C, H: z.H, W: z.W, Data: append([]float32(nil), z.Data...)}
	TinyClamp(x.Data)
	tap(0, x)
	x, err := d.ConvIn.Apply(x)
	if err != nil {
		return nil, err
	}
	tap(1, x)
	tap(2, ReLUInPlace(x))
	idx := 3
	for _, s := range d.Stages {
		for i := range s.Blocks {
			if x, err = s.Blocks[i].Apply(x); err != nil {
				return nil, err
			}
			tap(idx, x)
			idx++
		}
		if s.Up != nil {
			x = UpsampleNearest2x(x)
			tap(idx, x)
			if x, err = s.Up.Apply(x); err != nil {
				return nil, err
			}
			tap(idx+1, x)
			idx += 2
		}
	}
	if x, err = d.ConvOut.Apply(x); err != nil {
		return nil, err
	}
	tap(idx, x)
	return TinyToImage(x)
}
