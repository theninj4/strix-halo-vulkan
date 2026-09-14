package parakeet

import (
	"fmt"
	"math"
)

// FeedForward is the conformer's macaron half-step: a widening linear, SiLU,
// and a narrowing one. Neither has a bias (`attention_bias` is false and this
// checkpoint reuses that flag for the feed forwards).
type FeedForward struct {
	Linear1 *Linear
	Linear2 *Linear
}

// Apply runs the block.
func (f *FeedForward) Apply(x *Mat) (*Mat, error) {
	h, err := f.Linear1.Apply(x)
	if err != nil {
		return nil, err
	}
	for i, v := range h.Data {
		h.Data[i] = SiLU(v)
	}
	return f.Linear2.Apply(h)
}

// Attention is Transformer-XL style relative-position multi-head attention,
// as Conformer uses it.
//
// It is the one place the encoder needs arithmetic the engine has not run
// before. A standard attention is one score matrix; this is two, summed:
//
//	(q + bias_u)·k^T          the content term, over key positions
//	(q + bias_v)·rel_k^T      the position term, over *relative* positions
//
// where rel_k projects a sinusoidal embedding of every relative offset from
// +(T-1) down to -(T-1), and `relShift` is the trick that turns that
// 2T-1-wide matrix into the T-wide one the first term is added to, by reading
// it off a diagonal instead of computing T^2 offsets.
type Attention struct {
	Heads, HeadDim int
	Q, K, V, O     *Linear
	RelK           *Linear
	BiasU, BiasV   []float32 // [Heads*HeadDim]
}

// relShift reimplements transformers' `_rel_shift` literally — a left pad by
// one, a reinterpretation of the flat buffer, and a drop of the first row —
// rather than as the closed-form index map it is equivalent to. The literal
// version is the one that can be checked line by line against the Python; the
// closed form is a stage-6 concern, when this becomes a shader.
//
// in is [T, P] with P = 2T-1, row major. out is [T, T].
func relShift(in []float32, t, p int) []float32 {
	padded := make([]float32, t*(p+1))
	for i := 0; i < t; i++ {
		copy(padded[i*(p+1)+1:], in[i*p:(i+1)*p])
	}
	// view(2T, T), drop the first row, view(T, P), then keep the first T
	// columns: all of that is one offset into the flat buffer.
	out := make([]float32, t*t)
	for i := 0; i < t; i++ {
		for j := 0; j < t; j++ {
			out[i*t+j] = padded[t+i*p+j]
		}
	}
	return out
}

// Apply runs attention over frames, with posEmbed the [2T-1, dim] sinusoidal
// relative-position embedding and valid the number of non-padding frames.
func (a *Attention) Apply(x *Mat, posEmbed *Mat, valid int) (*Mat, error) {
	t, dim := x.Rows, x.Cols
	h, d := a.Heads, a.HeadDim
	if dim != h*d {
		return nil, fmt.Errorf("parakeet: attention over %d features with %d heads of %d", dim, h, d)
	}
	if posEmbed.Rows != 2*t-1 {
		return nil, fmt.Errorf("parakeet: %d position embeddings for %d frames, want %d", posEmbed.Rows, t, 2*t-1)
	}

	q, err := a.Q.Apply(x)
	if err != nil {
		return nil, err
	}
	k, err := a.K.Apply(x)
	if err != nil {
		return nil, err
	}
	v, err := a.V.Apply(x)
	if err != nil {
		return nil, err
	}
	relK, err := a.RelK.Apply(posEmbed)
	if err != nil {
		return nil, err
	}

	scale := float32(1 / math.Sqrt(float64(d)))
	ctx := NewMat(t, dim)
	p := 2*t - 1

	parallelFor(h, func(head int) {
		off := head * d
		qu := make([]float32, t*d)
		qv := make([]float32, t*d)
		for i := 0; i < t; i++ {
			row := q.Row(i)[off : off+d]
			for c := 0; c < d; c++ {
				qu[i*d+c] = row[c] + a.BiasU[off+c]
				qv[i*d+c] = row[c] + a.BiasV[off+c]
			}
		}

		// The position term, over all 2T-1 relative offsets, then shifted
		// onto the query/key grid.
		bd := make([]float32, t*p)
		for i := 0; i < t; i++ {
			qi := qv[i*d : (i+1)*d]
			for j := 0; j < p; j++ {
				rk := relK.Row(j)[off : off+d]
				var sum float32
				for c := 0; c < d; c++ {
					sum += qi[c] * rk[c]
				}
				bd[i*p+j] = sum
			}
		}
		shifted := relShift(bd, t, p)

		scores := make([]float32, t)
		for i := 0; i < t; i++ {
			qi := qu[i*d : (i+1)*d]
			max := float32(math.Inf(-1))
			for j := 0; j < t; j++ {
				if j >= valid {
					scores[j] = float32(math.Inf(-1))
					continue
				}
				kj := k.Row(j)[off : off+d]
				var sum float32
				for c := 0; c < d; c++ {
					sum += qi[c] * kj[c]
				}
				s := (sum + shifted[i*t+j]) * scale
				scores[j] = s
				if s > max {
					max = s
				}
			}
			var total float32
			for j := 0; j < t; j++ {
				e := float32(math.Exp(float64(scores[j] - max)))
				scores[j] = e
				total += e
			}
			inv := 1 / total
			dst := ctx.Row(i)[off : off+d]
			for j := 0; j < t; j++ {
				w := scores[j] * inv
				if w == 0 {
					continue
				}
				vj := v.Row(j)[off : off+d]
				for c := 0; c < d; c++ {
					dst[c] += w * vj[c]
				}
			}
		}
	})
	return a.O.Apply(ctx)
}

// ConvModule is the conformer's convolution branch: a pointwise convolution
// into a GLU, a depthwise convolution over time, a normalisation, SiLU, and a
// pointwise convolution back.
//
// The pointwise convolutions have kernel 1, so they are linears over the
// channel axis and are stored as such. BatchNorm is folded at load into the
// per-channel affine it becomes at inference — see loadConvModule — so there
// is no batch statistic anywhere in this struct.
type ConvModule struct {
	Channels int
	Kernel   int
	PW1      *Linear   // [2C, C]
	DW       []float32 // [C, K], one filter per channel
	BNScale  []float32
	BNShift  []float32
	PW2      *Linear // [C, C]
}

// Apply runs the branch over [T, C] frames.
func (c *ConvModule) Apply(x *Mat, valid int) (*Mat, error) {
	gated, err := c.PW1.Apply(x)
	if err != nil {
		return nil, err
	}
	t, ch := x.Rows, c.Channels

	// GLU over the channel axis: the first half gated by a sigmoid of the
	// second. In PyTorch this is `glu(dim=1)` on a [B, 2C, T] tensor, which
	// is the channel axis there too.
	glu := NewMat(t, ch)
	for i := 0; i < t; i++ {
		row := gated.Row(i)
		dst := glu.Row(i)
		for j := 0; j < ch; j++ {
			dst[j] = row[j] * sigmoid(row[ch+j])
		}
	}
	// Frames past the valid length are zeroed before the depthwise
	// convolution, so that padding cannot walk backwards into real frames
	// through the kernel.
	for i := valid; i < t; i++ {
		row := glu.Row(i)
		for j := range row {
			row[j] = 0
		}
	}

	pad := (c.Kernel - 1) / 2
	conv := NewMat(t, ch)
	parallelFor(ch, func(j int) {
		filt := c.DW[j*c.Kernel : (j+1)*c.Kernel]
		for i := 0; i < t; i++ {
			var sum float32
			for k, w := range filt {
				s := i + k - pad
				if s < 0 || s >= t {
					continue
				}
				sum += w * glu.Data[s*ch+j]
			}
			conv.Data[i*ch+j] = SiLU(sum*c.BNScale[j] + c.BNShift[j])
		}
	})
	return c.PW2.Apply(conv)
}

// EncoderLayer is one FastConformer block: two half-weighted feed forwards
// around an attention and a convolution, each pre-normed, with a final norm
// on the way out.
type EncoderLayer struct {
	NormFF1  *LayerNorm
	FF1      *FeedForward
	NormAttn *LayerNorm
	Attn     *Attention
	NormConv *LayerNorm
	Conv     *ConvModule
	NormFF2  *LayerNorm
	FF2      *FeedForward
	NormOut  *LayerNorm
}

// Apply runs the block. x is consumed: the residuals accumulate into it.
func (l *EncoderLayer) Apply(x *Mat, posEmbed *Mat, valid int) (*Mat, error) {
	n1, err := l.NormFF1.Apply(x)
	if err != nil {
		return nil, err
	}
	ff1, err := l.FF1.Apply(n1)
	if err != nil {
		return nil, err
	}
	x.AddInPlace(ff1, 0.5) // the macaron half-step

	n2, err := l.NormAttn.Apply(x)
	if err != nil {
		return nil, err
	}
	attn, err := l.Attn.Apply(n2, posEmbed, valid)
	if err != nil {
		return nil, err
	}
	x.AddInPlace(attn, 1)

	n3, err := l.NormConv.Apply(x)
	if err != nil {
		return nil, err
	}
	conv, err := l.Conv.Apply(n3, valid)
	if err != nil {
		return nil, err
	}
	x.AddInPlace(conv, 1)

	n4, err := l.NormFF2.Apply(x)
	if err != nil {
		return nil, err
	}
	ff2, err := l.FF2.Apply(n4)
	if err != nil {
		return nil, err
	}
	x.AddInPlace(ff2, 0.5)

	if err := l.NormOut.ApplyInPlace(x); err != nil {
		return nil, err
	}
	return x, nil
}

// Encoder is the FastConformer stack.
type Encoder struct {
	Config      EncoderConfig
	Subsampling *Subsampling
	Layers      []*EncoderLayer
}

// PositionEmbeddings builds the [2T-1, dim] sinusoidal embedding of every
// relative offset, from +(T-1) down to -(T-1).
//
// The layout is interleaved sin/cos — `stack([sin, cos], -1)` — which is not
// the half-and-half layout RoPE uses anywhere else in this repository, and
// the two are only distinguishable by what they do to the position term, so
// it is pinned by a test rather than by inspection.
func (e *Encoder) PositionEmbeddings(t int) *Mat {
	dim := e.Config.HiddenSize
	out := NewMat(2*t-1, dim)
	for p := 0; p < out.Rows; p++ {
		pos := float64(t - 1 - p)
		row := out.Row(p)
		for i := 0; i < dim/2; i++ {
			freq := pos / math.Pow(10000, float64(2*i)/float64(dim))
			row[2*i] = float32(math.Sin(freq))
			row[2*i+1] = float32(math.Cos(freq))
		}
	}
	return out
}

// Apply runs the whole encoder over a log-mel spectrogram, returning the
// per-frame hidden states and how many frames are valid.
func (e *Encoder) Apply(mel *Mat, validFrames int) (*Mat, int, error) {
	return e.forward(mel, validFrames, nil)
}

// forward is Apply with a hook that sees every layer's output.
func (e *Encoder) forward(mel *Mat, validFrames int, trace func(layer int, h *Mat)) (*Mat, int, error) {
	h, valid, err := e.Subsampling.Apply(mel, validFrames)
	if err != nil {
		return nil, 0, err
	}
	if e.Config.ScaleInput {
		s := float32(math.Sqrt(float64(e.Config.HiddenSize)))
		for i := range h.Data {
			h.Data[i] *= s
		}
	}
	pos := e.PositionEmbeddings(h.Rows)
	for i, layer := range e.Layers {
		h, err = layer.Apply(h, pos, valid)
		if err != nil {
			return nil, 0, fmt.Errorf("parakeet: encoder layer %d: %w", i, err)
		}
		if trace != nil {
			trace(i, h)
		}
	}
	return h, valid, nil
}
