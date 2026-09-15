package kokoro

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// Mat is a dense row-major matrix, [frames, channels] throughout — the
// transpose of the [C, T] a PyTorch conv1d activation has. See the package
// comment: channel-last is what makes the pointwise convolutions GEMMs.
type Mat struct {
	Rows, Cols int
	Data       []float32
}

// NewMat allocates a zeroed matrix.
func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

// Row returns one frame, aliasing Data.
func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Clone copies the matrix.
func (m *Mat) Clone() *Mat {
	return &Mat{Rows: m.Rows, Cols: m.Cols, Data: append([]float32(nil), m.Data...)}
}

func (m *Mat) String() string { return fmt.Sprintf("[%d %d]", m.Rows, m.Cols) }

// AddInPlace accumulates another matrix of the same shape, which is every
// residual connection in this package.
func (m *Mat) AddInPlace(o *Mat) {
	for i, v := range o.Data {
		m.Data[i] += v
	}
}

// Concat joins two matrices along the channel axis, which is what the
// duration encoder does to the style vector after every normalisation and
// what the decoder does to its three side channels before every block.
func Concat(mats ...*Mat) (*Mat, error) {
	if len(mats) == 0 {
		return nil, fmt.Errorf("kokoro: concat of nothing")
	}
	rows, cols := mats[0].Rows, 0
	for _, m := range mats {
		if m.Rows != rows {
			return nil, fmt.Errorf("kokoro: concat of %d and %d frames", rows, m.Rows)
		}
		cols += m.Cols
	}
	out := NewMat(rows, cols)
	for r := 0; r < rows; r++ {
		dst := out.Row(r)
		for _, m := range mats {
			copy(dst, m.Row(r))
			dst = dst[m.Cols:]
		}
	}
	return out, nil
}

// Broadcast repeats one vector down `rows` frames, which is how the 128-wide
// style vector is joined to a [T, C] activation.
func Broadcast(v []float32, rows int) *Mat {
	out := NewMat(rows, len(v))
	for r := 0; r < rows; r++ {
		copy(out.Row(r), v)
	}
	return out
}

// parallelFor runs fn over 0..n-1, splitting the range across GOMAXPROCS
// goroutines that claim contiguous chunks.
//
// Chunks rather than one index per worker hand-off, which is what the
// recurrences need: an LSTM calls this once per timestep — a thousand times
// over an utterance — over 2048 gates of a few hundred multiplies each, and a
// per-index channel makes the synchronisation cost more than the arithmetic.
// Measured: the duration encoder goes from 145 ms to 32 ms on this change
// alone. Contiguous rather than strided, so each worker walks one span of the
// weight matrix.
//
// There is deliberately no lower bound on n. The loops here that are short
// are short in *indices*, not in work — twelve attention heads over a whole
// utterance — so a threshold that skipped the fan-out for them would cost
// more than it saved.
func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	// Several chunks per worker, so an uneven cost per index still balances.
	chunk := (n + workers*4 - 1) / (workers * 4)
	if chunk < 1 {
		chunk = 1
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				start := int(next.Add(int64(chunk))) - chunk
				if start >= n {
					return
				}
				end := start + chunk
				if end > n {
					end = n
				}
				for i := start; i < end; i++ {
					fn(i)
				}
			}
		}()
	}
	wg.Wait()
}

// Linear is y = Wx + b with W stored [Out, In], PyTorch's order.
//
// Every pointwise convolution in this model is one of these: convert folds
// the trailing kernel axis off a [Out, In, 1] weight, because a kernel of one
// over a channel-last activation is exactly a matrix multiply.
type Linear struct {
	In, Out int
	Weight  []float32
	Bias    []float32
}

// Apply runs the projection over every frame of x.
func (l *Linear) Apply(x *Mat) (*Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("kokoro: linear takes %d channels, got %d", l.In, x.Cols)
	}
	out := NewMat(x.Rows, l.Out)
	parallelFor(l.Out, func(o int) {
		w := l.Weight[o*l.In : (o+1)*l.In]
		var bias float32
		if l.Bias != nil {
			bias = l.Bias[o]
		}
		for r := 0; r < x.Rows; r++ {
			row := x.Row(r)
			sum := bias
			for i, wv := range w {
				sum += wv * row[i]
			}
			out.Data[r*l.Out+o] = sum
		}
	})
	return out, nil
}

// ApplyRow projects a single vector, which is what the style vector's fc
// layers do — one 128-wide input per block, not per frame.
func (l *Linear) ApplyRow(dst, x []float32) []float32 {
	for o := 0; o < l.Out; o++ {
		w := l.Weight[o*l.In : (o+1)*l.In]
		var sum float32
		if l.Bias != nil {
			sum = l.Bias[o]
		}
		for i, wv := range w {
			sum += wv * x[i]
		}
		dst[o] = sum
	}
	return dst
}

// Embedding is a [Vocab, Dim] lookup table.
type Embedding struct {
	Vocab, Dim int
	Weight     []float32
}

// Rows gathers one row per id.
func (e *Embedding) Rows(ids []int) (*Mat, error) {
	out := NewMat(len(ids), e.Dim)
	for i, id := range ids {
		if id < 0 || id >= e.Vocab {
			return nil, fmt.Errorf("kokoro: token %d outside a vocabulary of %d", id, e.Vocab)
		}
		copy(out.Row(i), e.Weight[id*e.Dim:(id+1)*e.Dim])
	}
	return out, nil
}

// LayerNorm is the mean-subtracting normalisation over a frame's channels,
// with an optional weight and bias.
//
// Three of this model's normalisations are this shape and they differ only in
// where the affine comes from: nn.LayerNorm carries its own (ALBERT, at
// eps 1e-12), modules.py's LayerNorm carries `gamma`/`beta` under different
// names (the text encoder), and AdaLayerNorm has none of its own because the
// style vector supplies one per utterance. Weight and Bias are nil for that
// last case.
type LayerNorm struct {
	Weight []float32
	Bias   []float32
	Width  int
	Eps    float64
}

// ApplyInPlace normalises every frame over its channels.
//
// The variance is the population one (divided by N), which is what
// torch.nn.LayerNorm uses.
func (n *LayerNorm) ApplyInPlace(x *Mat) error {
	if n.Width != x.Cols {
		return fmt.Errorf("kokoro: layer norm of width %d applied to %d channels", n.Width, x.Cols)
	}
	parallelFor(x.Rows, func(r int) {
		row := x.Row(r)
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(len(row))
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		scale := 1 / math.Sqrt(variance/float64(len(row))+n.Eps)
		for i, v := range row {
			out := float32((float64(v) - mean) * scale)
			if n.Weight != nil {
				out = out*n.Weight[i] + n.Bias[i]
			}
			row[i] = out
		}
	})
	return nil
}

// Apply returns a normalised copy.
func (n *LayerNorm) Apply(x *Mat) (*Mat, error) {
	out := x.Clone()
	if err := n.ApplyInPlace(out); err != nil {
		return nil, err
	}
	return out, nil
}

// InstanceNormInPlace normalises every *channel* over the frames — the
// transpose of what LayerNorm does, and the one thing a port of AdaIN gets
// wrong. nn.InstanceNorm1d is per (batch, channel) over time, with the
// population variance and, in this checkpoint, no affine of its own: see
// reference/convert_kokoro.py, which asserts all 70 of them are identity.
func InstanceNormInPlace(x *Mat, eps float64) {
	if x.Rows == 0 {
		return
	}
	parallelFor(x.Cols, func(c int) {
		var mean float64
		for r := 0; r < x.Rows; r++ {
			mean += float64(x.Data[r*x.Cols+c])
		}
		mean /= float64(x.Rows)
		var variance float64
		for r := 0; r < x.Rows; r++ {
			d := float64(x.Data[r*x.Cols+c]) - mean
			variance += d * d
		}
		scale := 1 / math.Sqrt(variance/float64(x.Rows)+eps)
		for r := 0; r < x.Rows; r++ {
			x.Data[r*x.Cols+c] = float32((float64(x.Data[r*x.Cols+c]) - mean) * scale)
		}
	})
}

// LeakyReLU is the activation of every StyleTTS2 block, at the 0.2 slope both
// the text encoder and the AdaIN blocks are built with.
func LeakyReLU(x, slope float32) float32 {
	if x < 0 {
		return slope * x
	}
	return x
}

func leakyReLUInPlace(x *Mat, slope float32) {
	for i, v := range x.Data {
		x.Data[i] = LeakyReLU(v, slope)
	}
}

// leakyReLU returns a rectified copy. The generator needs the copy: its
// activations are applied to a value the caller still holds — the previous
// stage's output, which a trace keeps a pointer to — where the encoders'
// are applied to a fresh convolution result.
func leakyReLU(x *Mat, slope float32) *Mat {
	out := x.Clone()
	leakyReLUInPlace(out, slope)
	return out
}

// GELUNew is HuggingFace's "gelu_new", the tanh approximation, which is what
// ALBERT's feed-forward uses here. It is not the erf GELU and the two differ
// in the fourth decimal.
func GELUNew(x float32) float32 {
	d := float64(x)
	return float32(0.5 * d * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(d+0.044715*d*d*d))))
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

func tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }
