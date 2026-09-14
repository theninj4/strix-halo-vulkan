package parakeet

import (
	"fmt"
	"math"
)

// LSTMLayer is one layer of PyTorch's nn.LSTM, with the gate order its
// weights are stored in: input, forget, cell, output, stacked down the rows
// of both weight matrices.
//
// Note the two bias vectors. PyTorch keeps `bias_ih` and `bias_hh` separately
// even though only their sum can ever matter at inference — a CuDNN
// compatibility artefact — so they are summed once at load and carried as one
// vector here.
type LSTMLayer struct {
	Hidden int
	WIH    []float32 // [4H, In]
	WHH    []float32 // [4H, H]
	Bias   []float32 // [4H], bias_ih + bias_hh
	In     int
}

// Step advances the layer by one token, writing into h and c in place.
func (l *LSTMLayer) Step(x, h, c []float32) {
	hs := l.Hidden
	gates := make([]float32, 4*hs)
	copy(gates, l.Bias)
	for g := 0; g < 4*hs; g++ {
		wi := l.WIH[g*l.In : (g+1)*l.In]
		var sum float32
		for i, w := range wi {
			sum += w * x[i]
		}
		wh := l.WHH[g*hs : (g+1)*hs]
		for i, w := range wh {
			sum += w * h[i]
		}
		gates[g] += sum
	}
	for i := 0; i < hs; i++ {
		in := sigmoid(gates[i])
		forget := sigmoid(gates[hs+i])
		cell := float32(math.Tanh(float64(gates[2*hs+i])))
		out := sigmoid(gates[3*hs+i])
		c[i] = forget*c[i] + in*cell
		h[i] = out * float32(math.Tanh(float64(c[i])))
	}
}

// Prediction is the transducer's prediction network: an embedding, a stack of
// LSTM layers and a projection. It is a language model over the emitted
// tokens — it never sees the audio — so it advances once per emitted symbol
// rather than once per frame, which is what makes it a latency problem and
// not a throughput one.
type Prediction struct {
	Vocab, Hidden int
	Embedding     []float32 // [Vocab, Hidden]
	Layers        []*LSTMLayer
	Projector     *Linear
}

// PredictionState is the LSTM's carried state, one h and one c per layer.
type PredictionState struct {
	H, C [][]float32
}

// NewState allocates a zeroed state, which is what the first step starts from.
func (p *Prediction) NewState() *PredictionState {
	st := &PredictionState{}
	for range p.Layers {
		st.H = append(st.H, make([]float32, p.Hidden))
		st.C = append(st.C, make([]float32, p.Hidden))
	}
	return st
}

// Clone copies a state, so a caller can advance a hypothesis without
// destroying the one it came from.
func (s *PredictionState) Clone() *PredictionState {
	out := &PredictionState{}
	for i := range s.H {
		out.H = append(out.H, append([]float32(nil), s.H[i]...))
		out.C = append(out.C, append([]float32(nil), s.C[i]...))
	}
	return out
}

// Step advances the network by one token and returns the projected output.
func (p *Prediction) Step(token int, st *PredictionState) ([]float32, error) {
	if token < 0 || token >= p.Vocab {
		return nil, fmt.Errorf("parakeet: token %d outside a vocabulary of %d", token, p.Vocab)
	}
	x := p.Embedding[token*p.Hidden : (token+1)*p.Hidden]
	for i, layer := range p.Layers {
		layer.Step(x, st.H[i], st.C[i])
		x = st.H[i]
	}
	return p.Projector.ApplyRow(make([]float32, p.Projector.Out), x), nil
}

// Joint combines one encoder frame with one prediction state into logits over
// the vocabulary *and* over the duration to jump.
//
// The head is [8198, 640]: 8193 token logits with blank at 8192, followed by
// five duration logits for [0, 1, 2, 3, 4]. That layout is the whole of TDT —
// emit a token, then advance the encoder cursor by the predicted duration
// instead of by one frame.
type Joint struct {
	Head      *Linear
	Vocab     int
	Durations []int
}

// Logits runs the joint for one (frame, state) pair.
func (j *Joint) Logits(dst, enc, dec []float32) []float32 {
	sum := make([]float32, len(enc))
	for i := range enc {
		sum[i] = ReLU(enc[i] + dec[i])
	}
	return j.Head.ApplyRow(dst, sum)
}

// Argmax returns the most likely token and the duration to advance by.
//
// The two argmaxes are independent — over the first Vocab logits and over the
// duration logits after them — and the one rule joining them is the guard
// against a stalled loop: a blank that asks for a zero-frame jump is forced
// to one frame.
func (j *Joint) Argmax(logits []float32, blank int) (token, duration int) {
	token = 0
	for i := 1; i < j.Vocab; i++ {
		if logits[i] > logits[token] {
			token = i
		}
	}
	best := 0
	for i := 1; i < len(j.Durations); i++ {
		if logits[j.Vocab+i] > logits[j.Vocab+best] {
			best = i
		}
	}
	duration = j.Durations[best]
	if token == blank && duration == 0 {
		duration = 1
	}
	return token, duration
}
