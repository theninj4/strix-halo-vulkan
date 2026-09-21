// tensor.go holds the dense-tensor primitives the CPU reference is built
// from: an NCHW float32 tensor, a direct convolution, and the three
// elementwise ops the graph needs. They were `zimage/vae`'s, shared by both
// autoencoders while Z-Image still shipped; that package was deleted once
// Q9b refused its matrix-core convolution on precision, and these two files
// are what survived it.
//
// They are written for clarity over speed — a direct convolution
// parallelised over output channels, no im2col, no blocking — because the
// fast version of this is a compute shader and not a smarter Go loop.
//
// Layout is NCHW float32 throughout, matching PyTorch, so a tensor here can
// be compared to a reference dump byte for byte with no transpose in the
// way.
package vae

import (
	"fmt"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// Tensor is a dense NCHW float32 tensor.
type Tensor struct {
	N, C, H, W int
	Data       []float32
}

// NewTensor allocates a zeroed tensor.
func NewTensor(n, c, h, w int) *Tensor {
	return &Tensor{N: n, C: c, H: h, W: w, Data: make([]float32, n*c*h*w)}
}

// Len is the element count.
func (t *Tensor) Len() int { return t.N * t.C * t.H * t.W }

// Plane returns the h*w slice for one image and channel.
func (t *Tensor) Plane(n, c int) []float32 {
	off := ((n * t.C) + c) * t.H * t.W
	return t.Data[off : off+t.H*t.W]
}

// Shape renders the shape the way the reference manifest writes it.
func (t *Tensor) Shape() []int { return []int{t.N, t.C, t.H, t.W} }

func (t *Tensor) String() string {
	return fmt.Sprintf("[%d %d %d %d]", t.N, t.C, t.H, t.W)
}

// parallelFor runs fn over [0,n) across GOMAXPROCS workers. Every heavy loop
// here is embarrassingly parallel over output channels, which is both the
// outermost loop and the one with no write aliasing between iterations.
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

// Conv2D is a 2-D convolution with zero padding, weights in PyTorch's
// [outC, inC, kh, kw] order.
//
// Stride and PadEnd exist for the encoder's three downsamplers and are zero
// everywhere else. **They are one shape between them, not two knobs**:
// diffusers' Downsample2D pads (0, 1, 0, 1) -- nothing on the top and left,
// one pixel on the bottom and right -- and then convolves with stride 2 and
// no padding of its own. So Pad stays the symmetric padding every other
// convolution in this package uses, PadEnd is the extra row and column on the
// far side, and a zero in either is the ordinary case.
type Conv2D struct {
	InC, OutC int
	KH, KW    int
	Pad       int
	// PadEnd is extra zero padding on the bottom and right only.
	PadEnd int
	// Stride is the step between output pixels; 0 and 1 both mean 1, so that
	// every Conv2D built before this field existed still means what it did.
	Stride int
	Weight []float32 // [OutC][InC][KH][KW]
	Bias   []float32 // [OutC], may be nil
}

// stride is Stride with the zero value read as 1.
func (c *Conv2D) stride() int {
	if c.Stride < 1 {
		return 1
	}
	return c.Stride
}

// OutSize is the spatial size this convolution produces from an input side of
// n, which is PyTorch's formula with the two paddings separated.
func (c *Conv2D) OutSize(n, k int) int {
	return (n+2*c.Pad+c.PadEnd-k)/c.stride() + 1
}

// FP16ConvOperands, when set, is asked of each convolution before it runs;
// the ones it accepts narrow *both* operands to IEEE binary16 and accumulate
// in float32. That is exactly what a matrix core does, and this is the
// instrument IMAGE.md's Q9 ledger says the conv port owes before it is
// written: z-image narrowed its decoder's convolutions on a graph whose
// activations peak at 497, Qwen's peak at 1.1e4, and "20x less headroom is
// still inside fp16" is an argument where the project's method wants a
// measurement.
//
// It is a predicate rather than a flag because the port is one too -- Qwen's
// graph narrows its 3x3 convolutions and leaves the 1x1 shortcuts alone,
// since those are the only ones reading a tensor the channel norm does not
// bound -- and an instrument that narrowed a different set from the port
// would be predicting the wrong thing.
//
// It narrows the *tensors*, once each, rather than each product: that is both
// faster and more faithful, because the device does the same -- the filter is
// packed to halves at load and shaders/vae_pack_conv.comp narrows the
// activation once into the blocked layout the fragment loads read.
//
// It is a package-level hook because the port it predicts is a property of
// the convolution and not of a caller; nothing in the library sets it and it
// is not safe to change while a decode is running.
var FP16ConvOperands func(c *Conv2D) bool

// narrowF16 returns a copy of v with every element rounded to binary16 and
// back.
func narrowF16(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = safetensors.F16ToF32(safetensors.F32ToF16(x))
	}
	return out
}

// Apply runs the convolution. Output is [N, OutC, H, W] for the 3x3 pad-1 and
// 1x1 pad-0 cases this decoder uses, both of which preserve spatial size.
func (c *Conv2D) Apply(x *Tensor) (*Tensor, error) {
	if x.C != c.InC {
		return nil, fmt.Errorf("vae: conv expects %d input channels, got %d", c.InC, x.C)
	}
	weight := c.Weight
	if FP16ConvOperands != nil && FP16ConvOperands(c) {
		weight = narrowF16(weight)
		x = &Tensor{N: x.N, C: x.C, H: x.H, W: x.W, Data: narrowF16(x.Data)}
	}
	stride := c.stride()
	outH := c.OutSize(x.H, c.KH)
	outW := c.OutSize(x.W, c.KW)
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("vae: conv on %s with kernel %dx%d pad %d leaves nothing", x, c.KH, c.KW, c.Pad)
	}
	out := NewTensor(x.N, c.OutC, outH, outW)

	for n := 0; n < x.N; n++ {
		parallelFor(c.OutC, func(oc int) {
			dst := out.Plane(n, oc)
			var bias float32
			if c.Bias != nil {
				bias = c.Bias[oc]
			}
			for i := range dst {
				dst[i] = bias
			}
			// Accumulate one (input channel, kernel tap) plane at a time.
			// Holding the tap fixed makes the inner loop a contiguous
			// strided add over a whole output row.
			for ic := 0; ic < c.InC; ic++ {
				src := x.Plane(n, ic)
				wbase := ((oc * c.InC) + ic) * c.KH * c.KW
				for kh := 0; kh < c.KH; kh++ {
					for kw := 0; kw < c.KW; kw++ {
						wv := weight[wbase+kh*c.KW+kw]
						if wv == 0 {
							continue
						}
						for oh := 0; oh < outH; oh++ {
							ih := oh*stride + kh - c.Pad
							if ih < 0 || ih >= x.H {
								continue
							}
							// Clip the output-column range to where the
							// input column is in bounds, so the inner loop
							// carries no branch. With a stride the two
							// bounds are the same inequality divided by it:
							// lo rounds up and hi rounds down.
							lo, hi := 0, outW
							if v := ceilDiv(c.Pad-kw, stride); v > lo {
								lo = v
							}
							if v := floorDiv(x.W-1+c.Pad-kw, stride) + 1; v < hi {
								hi = v
							}
							if lo >= hi {
								continue
							}
							drow := dst[oh*outW : (oh+1)*outW]
							srow := src[ih*x.W : (ih+1)*x.W]
							base := kw - c.Pad
							for ow := lo; ow < hi; ow++ {
								drow[ow] += wv * srow[ow*stride+base]
							}
						}
					}
				}
			}
		})
	}
	return out, nil
}

// SiLUInPlace applies x * sigmoid(x).
func SiLUInPlace(x *Tensor) *Tensor {
	parallelFor(x.N*x.C, func(p int) {
		plane := x.Plane(p/x.C, p%x.C)
		for i, v := range plane {
			plane[i] = v / (1 + exp32(-v))
		}
	})
	return x
}

// UpsampleNearest2x doubles H and W by pixel replication, which is what
// diffusers' Upsample2D does before its convolution.
func UpsampleNearest2x(x *Tensor) *Tensor {
	out := NewTensor(x.N, x.C, x.H*2, x.W*2)
	for n := 0; n < x.N; n++ {
		parallelFor(x.C, func(c int) {
			src, dst := x.Plane(n, c), out.Plane(n, c)
			for h := 0; h < x.H; h++ {
				srow := src[h*x.W : (h+1)*x.W]
				for _, dh := range [2]int{h * 2, h*2 + 1} {
					drow := dst[dh*out.W : (dh+1)*out.W]
					for w, v := range srow {
						drow[w*2] = v
						drow[w*2+1] = v
					}
				}
			}
		})
	}
	return out
}

// AddInPlace computes a += b.
func AddInPlace(a, b *Tensor) (*Tensor, error) {
	if a.Len() != b.Len() {
		return nil, fmt.Errorf("vae: cannot add %s to %s", b, a)
	}
	parallelFor(a.N*a.C, func(p int) {
		ap, bp := a.Plane(p/a.C, p%a.C), b.Plane(p/a.C, p%a.C)
		for i := range ap {
			ap[i] += bp[i]
		}
	})
	return a, nil
}
