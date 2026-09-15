package kokoro

import (
	"fmt"
	"math"
)

// AdaIN1d is the operator the whole decoder is built from: normalise each
// channel over *time*, then apply a per-channel scale and shift that the
// 128-wide style vector produces.
//
// Two things about it are easy to get wrong and both are settled in
// reference/convert_kokoro.py. The normalisation is over time, not over
// channels — it is an instance norm, not a layer norm, and this model uses
// both. And the InstanceNorm1d's own affine is identity: it is constructed
// `affine=True` as an ONNX export workaround but the checkpoint has no
// weights for it, so there is no learned scale underneath the style's.
//
// The style affine is computed once per utterance, not per frame: FC is a
// [2C, 128] projection of a single vector, so it costs 2C*128 multiplies for
// the whole clip however long it is.
type AdaIN1d struct {
	Channels int
	FC       *Linear // 128 -> 2*Channels, gamma then beta
}

// Apply normalises x in place and applies the style affine.
func (a *AdaIN1d) Apply(x *Mat, style []float32) error {
	if x.Cols != a.Channels {
		return fmt.Errorf("kokoro: adain over %d channels, got %d", a.Channels, x.Cols)
	}
	h := a.FC.ApplyRow(make([]float32, a.FC.Out), style)
	gamma, beta := h[:a.Channels], h[a.Channels:]
	InstanceNormInPlace(x, instanceNormEps)
	for r := 0; r < x.Rows; r++ {
		row := x.Row(r)
		for c, v := range row {
			row[c] = (1+gamma[c])*v + beta[c]
		}
	}
	return nil
}

// AdaLayerNorm is the duration encoder's normalisation: the same style affine
// over a *layer* norm rather than an instance norm.
//
// The pairing is deliberate upstream and worth keeping straight — the prosody
// predictor's text side normalises each frame across its channels, and
// everything downstream of the length regulator normalises each channel
// across the frames. Nothing in the model does both.
type AdaLayerNorm struct {
	Channels int
	FC       *Linear // 128 -> 2*Channels
	norm     LayerNorm
}

// Apply normalises x in place and applies the style affine.
func (a *AdaLayerNorm) Apply(x *Mat, style []float32) error {
	if x.Cols != a.Channels {
		return fmt.Errorf("kokoro: ada layer norm over %d channels, got %d", a.Channels, x.Cols)
	}
	h := a.FC.ApplyRow(make([]float32, a.FC.Out), style)
	gamma, beta := h[:a.Channels], h[a.Channels:]
	if err := a.norm.ApplyInPlace(x); err != nil {
		return err
	}
	for r := 0; r < x.Rows; r++ {
		row := x.Row(r)
		for c, v := range row {
			row[c] = (1+gamma[c])*v + beta[c]
		}
	}
	return nil
}

// AdainResBlk1d is StyleTTS2's residual block: two style-conditioned
// normalisations around two 3-tap convolutions, added to a shortcut and
// scaled by 1/sqrt(2).
//
// The 1/sqrt(2) is not a detail. It is applied to the *sum*, so the block is
// a variance-preserving average of its two branches rather than the plain
// residual add every other model in this repository uses, and leaving it out
// makes the output grow by sqrt(2) per block — four blocks deep in the
// decoder, that is 4x.
//
// Upsampling blocks do it twice over: a depthwise transposed convolution on
// the residual path and a nearest-neighbour repeat on the shortcut. The two
// must agree on the output length, which they do at stride 2 with
// output_padding 1.
type AdainResBlk1d struct {
	In, Out  int
	Upsample bool

	Norm1   *AdaIN1d
	Pool    *ConvTranspose1D // nil unless Upsample
	Conv1   *Conv1D
	Norm2   *AdaIN1d
	Conv2   *Conv1D
	Conv1x1 *Conv1D // nil unless In != Out
}

// Apply runs the block over a [T, In] activation.
func (b *AdainResBlk1d) Apply(x *Mat, style []float32) (*Mat, error) {
	r := x.Clone()
	if err := b.Norm1.Apply(r, style); err != nil {
		return nil, err
	}
	leakyReLUInPlace(r, 0.2)
	if b.Pool != nil {
		var err error
		if r, err = b.Pool.Apply(r); err != nil {
			return nil, err
		}
	}
	r, err := b.Conv1.Apply(r)
	if err != nil {
		return nil, err
	}
	if err := b.Norm2.Apply(r, style); err != nil {
		return nil, err
	}
	leakyReLUInPlace(r, 0.2)
	if r, err = b.Conv2.Apply(r); err != nil {
		return nil, err
	}

	sc := x
	if b.Upsample {
		sc = UpsampleNearest(sc, 2)
	}
	if b.Conv1x1 != nil {
		if sc, err = b.Conv1x1.Apply(sc); err != nil {
			return nil, err
		}
	}
	if sc.Rows != r.Rows || sc.Cols != r.Cols {
		return nil, fmt.Errorf("kokoro: adain block shortcut %v against residual %v", sc, r)
	}
	scale := float32(1 / math.Sqrt2)
	for i, v := range sc.Data {
		r.Data[i] = (r.Data[i] + v) * scale
	}
	return r, nil
}
