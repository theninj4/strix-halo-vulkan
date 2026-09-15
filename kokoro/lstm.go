package kokoro

import "fmt"

// LSTMDirection is one direction of one layer of PyTorch's nn.LSTM, with the
// gate order its weights are stored in: input, forget, cell, output, stacked
// down the rows of both weight matrices.
//
// PyTorch keeps `bias_ih` and `bias_hh` separately even though only their sum
// can matter at inference — a CuDNN compatibility artefact — so they are
// summed once at load and carried as one vector.
type LSTMDirection struct {
	In, Hidden int
	WIH        []float32 // [4H, In]
	WHH        []float32 // [4H, H]
	Bias       []float32 // [4H], bias_ih + bias_hh
}

// step advances one token, writing into h and c in place.
func (d *LSTMDirection) step(x, h, c, gates []float32) {
	hs := d.Hidden
	copy(gates, d.Bias)
	parallelFor(4*hs, func(g int) {
		wi := d.WIH[g*d.In : (g+1)*d.In]
		var sum float32
		for i, w := range wi {
			sum += w * x[i]
		}
		wh := d.WHH[g*hs : (g+1)*hs]
		for i, w := range wh {
			sum += w * h[i]
		}
		gates[g] += sum
	})
	for i := 0; i < hs; i++ {
		in := sigmoid(gates[i])
		forget := sigmoid(gates[hs+i])
		cell := tanh(gates[2*hs+i])
		out := sigmoid(gates[3*hs+i])
		c[i] = forget*c[i] + in*cell
		h[i] = out * tanh(c[i])
	}
}

// LSTM is a single-layer bidirectional nn.LSTM, which is the only shape of
// recurrence in this model: the text encoder's, the duration encoder's three,
// the duration head's and the prosody predictor's `shared` are all one layer,
// bidirectional, batch-first.
//
// Bidirectional is the structural difference from parakeet, whose prediction
// network is causal by construction. Here every recurrence sees the whole
// utterance, which is why the phoneme side of this model cannot be streamed
// and the vocoder side can.
type LSTM struct {
	In, Hidden int // Hidden is per direction; the output is 2*Hidden wide
	Fwd, Rev   *LSTMDirection
}

// Apply runs both directions over a [T, In] activation and returns
// [T, 2*Hidden], forward in the first Hidden channels and reverse in the
// last — torch's concatenation order.
func (l *LSTM) Apply(x *Mat) (*Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("kokoro: lstm takes %d channels, got %d", l.In, x.Cols)
	}
	out := NewMat(x.Rows, 2*l.Hidden)
	gates := make([]float32, 4*l.Hidden)
	h := make([]float32, l.Hidden)
	c := make([]float32, l.Hidden)
	for t := 0; t < x.Rows; t++ {
		l.Fwd.step(x.Row(t), h, c, gates)
		copy(out.Row(t)[:l.Hidden], h)
	}
	for i := range h {
		h[i], c[i] = 0, 0
	}
	for t := x.Rows - 1; t >= 0; t-- {
		l.Rev.step(x.Row(t), h, c, gates)
		copy(out.Row(t)[l.Hidden:], h)
	}
	return out, nil
}
