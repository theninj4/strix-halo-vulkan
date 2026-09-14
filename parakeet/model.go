package parakeet

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// Mat is a dense row-major matrix, [frames, features] throughout — a PyTorch
// tensor with the batch axis squeezed out, so a tensor here compares against
// a reference dump with no transpose in the way.
type Mat struct {
	Rows, Cols int
	Data       []float32
}

// NewMat allocates a zeroed matrix.
func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

// Row returns one row, aliasing Data.
func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Clone copies the matrix.
func (m *Mat) Clone() *Mat {
	return &Mat{Rows: m.Rows, Cols: m.Cols, Data: append([]float32(nil), m.Data...)}
}

func (m *Mat) String() string { return fmt.Sprintf("[%d %d]", m.Rows, m.Cols) }

// AddInPlace accumulates another matrix of the same shape, which is every
// residual connection in the encoder.
func (m *Mat) AddInPlace(o *Mat, scale float32) {
	for i, v := range o.Data {
		m.Data[i] += scale * v
	}
}

// Volume is a [C, T, F] feature map with F fastest — the layout a PyTorch
// conv2d activation has once its batch axis is squeezed out, and the layout
// the reference dumps the subsampling stack in.
type Volume struct {
	C, T, F int
	Data    []float32
}

// NewVolume allocates a zeroed feature map.
func NewVolume(c, t, f int) *Volume {
	return &Volume{C: c, T: t, F: f, Data: make([]float32, c*t*f)}
}

// Plane returns one channel, aliasing Data.
func (v *Volume) Plane(c int) []float32 { return v.Data[c*v.T*v.F : (c+1)*v.T*v.F] }

func (v *Volume) String() string { return fmt.Sprintf("[%d %d %d]", v.C, v.T, v.F) }

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
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// Linear is y = Wx + b with W stored [Out, In], PyTorch's order. Bias is nil
// where the checkpoint has none — which is most of the encoder, since
// `attention_bias` and `convolution_bias` are both false, and present on the
// three projections that join stages: the subsampling's linear, the encoder
// projector and the joint head.
type Linear struct {
	In, Out int
	Weight  []float32
	Bias    []float32

	// Narrow rounds the *input* row to fp16 before multiplying, which is the
	// other half of what the matrix cores do — the weights are narrowed in
	// place by Model.SetF16 rather than on every use. The accumulation stays
	// float32, as `coopMatMulAdd`'s does.
	Narrow bool
}

// Apply runs the projection over every row of x.
func (l *Linear) Apply(x *Mat) (*Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("parakeet: linear takes %d features, got %d", l.In, x.Cols)
	}
	if l.Narrow {
		x = x.Clone()
		narrowSliceF16(x.Data)
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

// ApplyRow projects a single vector, which is what the prediction network and
// the joint do — one frame and one token at a time inside the decode loop.
func (l *Linear) ApplyRow(dst, x []float32) []float32 {
	if l.Narrow {
		x = append([]float32(nil), x...)
		narrowSliceF16(x)
	}
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

// LayerNorm is the mean-subtracting normalisation, with a weight and a bias.
//
// Every norm in the conformer is one of these — there is no RMSNorm in this
// model at all, which is the first structural difference from everything the
// engine has run so far.
type LayerNorm struct {
	Weight []float32
	Bias   []float32
	Eps    float64
}

// Apply returns a normalised copy.
func (n *LayerNorm) Apply(x *Mat) (*Mat, error) {
	out := x.Clone()
	if err := n.ApplyInPlace(out); err != nil {
		return nil, err
	}
	return out, nil
}

// ApplyInPlace normalises every row.
//
// The variance is the population one (divided by N, not N-1) — the
// convention torch.nn.LayerNorm uses, and the opposite of the one the feature
// extractor's per-utterance normalisation uses. Both appear in this model,
// eight lines apart in the Python; they are not the same statistic.
func (n *LayerNorm) ApplyInPlace(x *Mat) error {
	if len(n.Weight) != x.Cols {
		return fmt.Errorf("parakeet: layer norm of width %d applied to %d columns", len(n.Weight), x.Cols)
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
			row[i] = float32((float64(v)-mean)*scale)*n.Weight[i] + n.Bias[i]
		}
	})
	return nil
}

// SiLU is x*sigmoid(x), the encoder's activation ("silu" in config.json, and
// what NeMo calls swish).
func SiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// ReLU is the subsampling stack's and the joint network's activation.
func ReLU(x float32) float32 {
	if x < 0 {
		return 0
	}
	return x
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// NarrowF16 rounds a float32 through IEEE binary16 and back, which is what a
// number costs when it crosses into the matrix cores: the Vulkan path stores
// weights as fp16 and feeds fp16 operands to `coopMatMulAdd`, accumulating in
// fp32.
//
// It is here rather than only in a test because the question it answers —
// does the transcript survive fp16? — is a design question for stage S6, and
// the CPU reference is the cheapest place to ask it. See Model.SetF16.
func NarrowF16(x float32) float32 {
	return safetensors.F16ToF32(safetensors.F32ToF16(x))
}

func narrowSliceF16(v []float32) {
	for i, x := range v {
		v[i] = NarrowF16(x)
	}
}
