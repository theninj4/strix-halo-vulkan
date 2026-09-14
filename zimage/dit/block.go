// Package dit implements Z-Image's DiT transformer block.
//
// This is the CPU reference, written before the Vulkan one for the same
// reason the VAE's was (PIPELINE.md): it separates understanding the
// architecture from getting a shader right, and once it matches diffusers it
// is the oracle the GPU port is debugged against.
//
// The block is Z-Image's "single stream" variant: one sequence carrying both
// caption and image tokens, self-attention over all of it, adaLN modulation
// driven by the timestep embedding, and a SwiGLU feed-forward. Two details
// are easy to get wrong and are called out at their use sites -- RoPE pairs
// *adjacent* components as one complex number rather than splitting the head
// in half, and the q/k RMS norms are per head rather than over the full
// width.
//
// Layout is [tokens, features] row-major float32, matching a PyTorch tensor
// with the batch axis squeezed out, so a tensor here compares to a reference
// dump with no transpose in the way.
package dit

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Mat is a dense row-major matrix.
type Mat struct {
	Rows, Cols int
	Data       []float32
}

// NewMat allocates a zeroed matrix.
func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

// Row returns one row.
func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Clone copies the matrix.
func (m *Mat) Clone() *Mat {
	return &Mat{Rows: m.Rows, Cols: m.Cols, Data: append([]float32(nil), m.Data...)}
}

func (m *Mat) String() string { return fmt.Sprintf("[%d %d]", m.Rows, m.Cols) }

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

// Linear is y = W x + b with W stored [Out, In], PyTorch's order.
type Linear struct {
	In, Out int
	Weight  []float32
	Bias    []float32 // may be nil
}

// Apply runs the projection over every row of x.
func (l *Linear) Apply(x *Mat) (*Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("dit: linear takes %d features, got %d", l.In, x.Cols)
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

// RMSNorm is x * rsqrt(mean(x^2) + eps) * weight, applied over the last
// `len(Weight)` components of each row. When a row is several times that
// width -- as it is for the per-head q and k norms, where the row holds
// every head -- the norm is applied to each span independently, which is
// what makes it a *per-head* norm.
type RMSNorm struct {
	Weight []float32
	Eps    float64
}

// ApplyInPlace normalises every span of len(Weight) in every row.
func (n *RMSNorm) ApplyInPlace(x *Mat) (*Mat, error) {
	w := len(n.Weight)
	if w == 0 || x.Cols%w != 0 {
		return nil, fmt.Errorf("dit: rmsnorm width %d does not divide %d columns", w, x.Cols)
	}
	spans := x.Cols / w
	parallelFor(x.Rows, func(r int) {
		row := x.Row(r)
		for s := 0; s < spans; s++ {
			seg := row[s*w : (s+1)*w]
			// float64 accumulation: a 3840-wide sum of squares in float32
			// loses the low bits well before the end.
			var sum float64
			for _, v := range seg {
				sum += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(sum/float64(w)+n.Eps))
			for i, v := range seg {
				seg[i] = v * inv * n.Weight[i]
			}
		}
	})
	return x, nil
}

// RoPE holds the precomputed cos/sin table, [tokens, headDim/2].
type RoPE struct {
	Cos, Sin []float32
	Tokens   int
	Pairs    int // headDim/2
}

// NewRoPE builds the table for Z-Image's 3-D rotary embedding: each axis
// contributes axesDims[a]/2 complex frequencies, looked up at that token's
// position along that axis, and the three are concatenated.
//
// ids is [tokens, len(axesDims)] positions.
func NewRoPE(ids [][3]int32, axesDims [3]int, axesLens [3]int, theta float64) (*RoPE, error) {
	pairs := 0
	for _, d := range axesDims {
		if d%2 != 0 {
			return nil, fmt.Errorf("dit: rope axis dim %d is odd", d)
		}
		pairs += d / 2
	}
	r := &RoPE{Tokens: len(ids), Pairs: pairs,
		Cos: make([]float32, len(ids)*pairs), Sin: make([]float32, len(ids)*pairs)}
	for t, id := range ids {
		off := 0
		for a, d := range axesDims {
			pos := int(id[a])
			if pos < 0 || pos >= axesLens[a] {
				return nil, fmt.Errorf("dit: token %d axis %d position %d outside [0,%d)", t, a, pos, axesLens[a])
			}
			for j := 0; j < d/2; j++ {
				// freq = 1 / theta^(2j/d), computed in float64 to match the
				// reference, which builds the table in float64 and casts.
				freq := 1 / math.Pow(theta, float64(2*j)/float64(d))
				angle := float64(pos) * freq
				r.Cos[t*pairs+off+j] = float32(math.Cos(angle))
				r.Sin[t*pairs+off+j] = float32(math.Sin(angle))
			}
			off += d / 2
		}
	}
	return r, nil
}

// ApplyInPlace rotates every head of every row.
//
// The pairing is (2j, 2j+1) -- adjacent components form one complex number.
// The other common convention splits the head in half and pairs j with
// j+headDim/2; using it here produces a plausible-looking tensor that is
// wrong everywhere, which is why this has its own test against the
// reference's q_roped.
func (r *RoPE) ApplyInPlace(x *Mat, heads, headDim int) error {
	if headDim != 2*r.Pairs {
		return fmt.Errorf("dit: rope table is %d pairs, head dim is %d", r.Pairs, headDim)
	}
	if x.Cols != heads*headDim {
		return fmt.Errorf("dit: rope on %s, want %d columns", x, heads*headDim)
	}
	if x.Rows != r.Tokens {
		return fmt.Errorf("dit: rope table has %d tokens, input has %d", r.Tokens, x.Rows)
	}
	parallelFor(x.Rows, func(t int) {
		row := x.Row(t)
		cos, sin := r.Cos[t*r.Pairs:(t+1)*r.Pairs], r.Sin[t*r.Pairs:(t+1)*r.Pairs]
		for h := 0; h < heads; h++ {
			seg := row[h*headDim : (h+1)*headDim]
			for j := 0; j < r.Pairs; j++ {
				re, im := seg[2*j], seg[2*j+1]
				c, s := cos[j], sin[j]
				seg[2*j] = re*c - im*s
				seg[2*j+1] = re*s + im*c
			}
		}
	})
	return nil
}
