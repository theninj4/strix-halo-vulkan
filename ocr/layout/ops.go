package layout

// The CPU ops the fp32 oracle port is made of: convolutions (with the batch
// norm folded in at load), the resamplers, and the transformer's rows. Every
// one is written the way the reference computes it, and parallel over output
// channels or rows, because the backbone and the hybrid encoder are ~80
// GFLOP at 800x800 and a scalar Go loop would take minutes.

import (
	"math"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// fp16Operands, when set, rounds every conv's and linear's operands to
// half precision and keeps the sums in fp32: what a matrix-core port would
// compute, measured on the oracle before one is written (TestFP16Ladder).
// absMax records the largest value any conv or linear produced, against
// half's 65,504.
var (
	fp16Operands bool
	fp16Linears  = true // with fp16Operands: the linears too, or the convs alone
	absMaxMu     sync.Mutex
	absMax       float64
)

func half(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = safetensors.F16ToF32(safetensors.F32ToF16(x))
	}
	return out
}

func noteMax(v []float32) {
	var m float64
	for _, x := range v {
		m = max(m, math.Abs(float64(x)))
	}
	absMaxMu.Lock()
	absMax = max(absMax, m)
	absMaxMu.Unlock()
}

// Map is a [C, H, W] fp32 feature map.
type Map struct {
	C, H, W int
	D       []float32
}

func newMap(c, h, w int) *Map { return &Map{C: c, H: h, W: w, D: make([]float32, c*h*w)} }

// Plane is channel c's H*W values.
func (m *Map) Plane(c int) []float32 { return m.D[c*m.H*m.W : (c+1)*m.H*m.W] }

// parallel runs f(i) for i in [0, n) over GOMAXPROCS goroutines.
func parallel(n int, f func(i int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers <= 1 {
		for i := 0; i < n; i++ {
			f(i)
		}
		return
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	next := 0
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= n {
					return
				}
				f(i)
			}
		}()
	}
	wg.Wait()
}

// Act is an activation.
type Act int

const (
	ActNone Act = iota
	ActReLU
	ActSiLU
	ActGELU // exact erf
)

func (a Act) apply(v []float32) {
	switch a {
	case ActReLU:
		for i, x := range v {
			if x < 0 {
				v[i] = 0
			}
		}
	case ActSiLU:
		for i, x := range v {
			v[i] = float32(float64(x) / (1 + math.Exp(-float64(x))))
		}
	case ActGELU:
		for i, x := range v {
			v[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
		}
	}
}

// Conv is a Conv2d with the eval-mode batch norm after it folded in: W is
// [Out, In/Groups, K, K] scaled by gamma/sqrt(var+eps), and B is
// beta - mean*scale (plus the conv's own bias, if it had one).
type Conv struct {
	In, Out, K, Stride, Pad, Groups int
	W, B                            []float32
	Act                             Act
}

// Apply runs the convolution, zero padding Pad on every side.
func (c *Conv) Apply(x *Map) *Map {
	oh := (x.H+2*c.Pad-c.K)/c.Stride + 1
	ow := (x.W+2*c.Pad-c.K)/c.Stride + 1
	out := newMap(c.Out, oh, ow)
	cc := c
	if fp16Operands {
		h := *c
		h.W = half(c.W)
		cc = &h
		x = &Map{C: x.C, H: x.H, W: x.W, D: half(x.D)}
	}
	if c.Groups == c.In && c.Groups == c.Out {
		cc.depthwise(x, out)
	} else {
		cc.gemm(x, out)
	}
	parallel(c.Out, func(o int) { c.Act.apply(out.Plane(o)) })
	noteMax(out.D)
	return out
}

// depthwise is one filter a channel.
func (c *Conv) depthwise(x, out *Map) {
	k := c.K
	parallel(c.Out, func(ch int) {
		src, dst := x.Plane(ch), out.Plane(ch)
		w := c.W[ch*k*k : (ch+1)*k*k]
		b := c.B[ch]
		for oy := 0; oy < out.H; oy++ {
			for ox := 0; ox < out.W; ox++ {
				acc := b
				for ky := 0; ky < k; ky++ {
					iy := oy*c.Stride - c.Pad + ky
					if iy < 0 || iy >= x.H {
						continue
					}
					row := src[iy*x.W:]
					for kx := 0; kx < k; kx++ {
						ix := ox*c.Stride - c.Pad + kx
						if ix < 0 || ix >= x.W {
							continue
						}
						acc += w[ky*k+kx] * row[ix]
					}
				}
				dst[oy*out.W+ox] = acc
			}
		}
	})
}

// gemm is im2col and a product, group by group: out[o] = sum_k W[o][k]
// col[k], with col's rows the (input channel, ky, kx) taps.
func (c *Conv) gemm(x, out *Map) {
	cin := c.In / c.Groups
	cout := c.Out / c.Groups
	kk := cin * c.K * c.K
	n := out.H * out.W
	for g := 0; g < c.Groups; g++ {
		var col []float32
		if c.K == 1 && c.Stride == 1 && c.Pad == 0 {
			col = x.D[g*cin*n : (g+1)*cin*n] // the input planes are the columns
		} else {
			col = make([]float32, kk*n)
			parallel(cin, func(ci int) {
				src := x.Plane(g*cin + ci)
				for ky := 0; ky < c.K; ky++ {
					for kx := 0; kx < c.K; kx++ {
						dst := col[((ci*c.K+ky)*c.K+kx)*n:][:n]
						for oy := 0; oy < out.H; oy++ {
							iy := oy*c.Stride - c.Pad + ky
							d := dst[oy*out.W:][:out.W]
							if iy < 0 || iy >= x.H {
								clear(d)
								continue
							}
							row := src[iy*x.W:][:x.W]
							for ox := range d {
								ix := ox*c.Stride - c.Pad + kx
								if ix < 0 || ix >= x.W {
									d[ox] = 0
								} else {
									d[ox] = row[ix]
								}
							}
						}
					}
				}
			})
		}
		parallel(cout, func(o int) {
			oc := g*cout + o
			dst := out.Plane(oc)
			b := c.B[oc]
			for i := range dst {
				dst[i] = b
			}
			w := c.W[oc*kk : (oc+1)*kk]
			for k, wk := range w {
				if wk == 0 {
					continue
				}
				src := col[k*n : (k+1)*n]
				for i, v := range src {
					dst[i] += wk * v
				}
			}
		})
	}
}

// concat stacks maps along channels.
func concat(ms ...*Map) *Map {
	c := 0
	for _, m := range ms {
		c += m.C
	}
	out := newMap(c, ms[0].H, ms[0].W)
	off := 0
	for _, m := range ms {
		copy(out.D[off:], m.D)
		off += len(m.D)
	}
	return out
}

// add returns a + b.
func add(a, b *Map) *Map {
	out := newMap(a.C, a.H, a.W)
	for i := range out.D {
		out.D[i] = a.D[i] + b.D[i]
	}
	return out
}

// padBR zero-pads one row at the bottom and one column at the right:
// F.pad(x, (0, 1, 0, 1)).
func padBR(x *Map) *Map {
	out := newMap(x.C, x.H+1, x.W+1)
	for c := 0; c < x.C; c++ {
		src, dst := x.Plane(c), out.Plane(c)
		for y := 0; y < x.H; y++ {
			copy(dst[y*out.W:], src[y*x.W:(y+1)*x.W])
		}
	}
	return out
}

// maxPool2 is MaxPool2d(kernel 2, stride 1, ceil_mode): the output is one
// smaller each way.
func maxPool2(x *Map) *Map {
	out := newMap(x.C, x.H-1, x.W-1)
	parallel(x.C, func(c int) {
		src, dst := x.Plane(c), out.Plane(c)
		for y := 0; y < out.H; y++ {
			for xx := 0; xx < out.W; xx++ {
				a := src[y*x.W+xx]
				a = max(a, src[y*x.W+xx+1], src[(y+1)*x.W+xx], src[(y+1)*x.W+xx+1])
				dst[y*out.W+xx] = a
			}
		}
	})
	return out
}

// upNearest2 is F.interpolate(scale_factor=2, mode="nearest").
func upNearest2(x *Map) *Map {
	out := newMap(x.C, x.H*2, x.W*2)
	parallel(x.C, func(c int) {
		src, dst := x.Plane(c), out.Plane(c)
		for y := 0; y < out.H; y++ {
			for xx := 0; xx < out.W; xx++ {
				dst[y*out.W+xx] = src[(y/2)*x.W+xx/2]
			}
		}
	})
	return out
}

// upBilinear2 is F.interpolate(scale_factor=2, mode="bilinear",
// align_corners=False): source coordinate (i + 0.5)/2 - 0.5, clamped at 0.
func upBilinear2(x *Map) *Map {
	out := newMap(x.C, x.H*2, x.W*2)
	tap := func(i, n int) (int, int, float32) {
		s := (float32(i)+0.5)/2 - 0.5
		if s < 0 {
			s = 0
		}
		i0 := int(s)
		return i0, min(i0+1, n-1), s - float32(i0)
	}
	parallel(x.C, func(c int) {
		src, dst := x.Plane(c), out.Plane(c)
		for y := 0; y < out.H; y++ {
			y0, y1, fy := tap(y, x.H)
			for xx := 0; xx < out.W; xx++ {
				x0, x1, fx := tap(xx, x.W)
				top := src[y0*x.W+x0]*(1-fx) + src[y0*x.W+x1]*fx
				bot := src[y1*x.W+x0]*(1-fx) + src[y1*x.W+x1]*fx
				dst[y*out.W+xx] = top*(1-fy) + bot*fy
			}
		}
	})
	return out
}

// Rows is a [N, D] row-major matrix.
type Rows struct {
	N, D int
	V    []float32
}

func newRows(n, d int) *Rows { return &Rows{N: n, D: d, V: make([]float32, n*d)} }

// Row is row i.
func (r *Rows) Row(i int) []float32 { return r.V[i*r.D : (i+1)*r.D] }

// tokens flattens a map to [H*W, C], the reference's flatten(2).transpose(1, 2).
func tokens(m *Map) *Rows {
	out := newRows(m.H*m.W, m.C)
	parallel(m.C, func(c int) {
		for i, v := range m.Plane(c) {
			out.V[i*m.C+c] = v
		}
	})
	return out
}

// untokens is the inverse of tokens.
func untokens(r *Rows, h, w int) *Map {
	out := newMap(r.D, h, w)
	parallel(r.D, func(c int) {
		dst := out.Plane(c)
		for i := range dst {
			dst[i] = r.V[i*r.D+c]
		}
	})
	return out
}

// Linear is a row-major [Out, In] weight and a bias.
type Linear struct {
	In, Out int
	W, B    []float32
}

// Apply computes x W^T + b, with float64 sums.
func (l *Linear) Apply(x *Rows) *Rows {
	out := newRows(x.N, l.Out)
	W := l.W
	if fp16Operands && fp16Linears {
		W = half(l.W)
		x = &Rows{N: x.N, D: x.D, V: half(x.V)}
	}
	defer noteMax(out.V)
	parallel(x.N, func(i int) {
		row, dst := x.Row(i), out.Row(i)
		for o := 0; o < l.Out; o++ {
			w := W[o*l.In : (o+1)*l.In]
			var acc float64
			for k, v := range row {
				acc += float64(v) * float64(w[k])
			}
			if l.B != nil {
				acc += float64(l.B[o])
			}
			dst[o] = float32(acc)
		}
	})
	return out
}

// LayerNorm is affine, over each row.
type LayerNorm struct {
	W, B []float32
	Eps  float64
}

// Apply returns the normalised rows.
func (n *LayerNorm) Apply(x *Rows) *Rows {
	out := newRows(x.N, x.D)
	parallel(x.N, func(i int) {
		row, dst := x.Row(i), out.Row(i)
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(len(row))
		var vs float64
		for _, v := range row {
			d := float64(v) - mean
			vs += d * d
		}
		inv := 1 / math.Sqrt(vs/float64(len(row))+n.Eps)
		for k, v := range row {
			dst[k] = float32((float64(v)-mean)*inv*float64(n.W[k]) + float64(n.B[k]))
		}
	})
	return out
}

func actRows(r *Rows, a Act) *Rows {
	parallel(r.N, func(i int) { a.apply(r.Row(i)) })
	return r
}

func addRows(a, b *Rows) *Rows {
	out := newRows(a.N, a.D)
	for i := range out.V {
		out.V[i] = a.V[i] + b.V[i]
	}
	return out
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// inverseSigmoid is the reference's: clamp to [0, 1], then log(x/(1-x))
// with both sides floored at 1e-5.
func inverseSigmoid(x float64) float64 {
	x = min(max(x, 0), 1)
	return math.Log(max(x, 1e-5) / max(1-x, 1e-5))
}
