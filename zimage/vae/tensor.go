// Package vae implements the Z-Image / Flux AutoencoderKL decoder: 16
// latent channels to RGB at 8x resolution.
//
// This is the CPU reference implementation. It exists before the Vulkan one
// on purpose: it separates "do I understand the architecture" from "is the
// shader right", and once it matches diffusers it becomes the oracle the GPU
// port is debugged against. It is written for clarity over speed — a direct
// convolution parallelised over output channels, no im2col, no blocking —
// because the fast version of this is a compute shader and not a smarter
// Go loop.
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

// convTap, when set, is called with every convolution's input before it runs.
// It is how TestConvInputsFitFP16 measures the headroom the matrix-core
// convolution depends on -- stage 2's "absmax 497 through the whole decoder"
// as an assertion rather than as a memory -- without keeping a second copy
// of the decoder's graph in the test. Nothing in the library sets it.
var convTap func(c *Conv2D, x *Tensor)

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
// It is a package-level hook, like convTap above, because the port it
// predicts is a property of the convolution and not of a caller; nothing in
// the library sets it and it is not safe to change while a decode is running.
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
	if convTap != nil {
		convTap(c, x)
	}
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

// GroupNorm normalises over (C/groups, H, W) per group, then scales and
// shifts per channel. The decoder uses 32 groups and eps 1e-6 everywhere.
type GroupNorm struct {
	Groups int
	Eps    float64
	Weight []float32 // [C]
	Bias   []float32 // [C]
}

// ApplyInPlace normalises x and returns it, so a caller can chain without
// allocating a tensor per norm.
func (g *GroupNorm) ApplyInPlace(x *Tensor) (*Tensor, error) {
	if x.C%g.Groups != 0 {
		return nil, fmt.Errorf("vae: %d channels do not divide into %d groups", x.C, g.Groups)
	}
	perGroup := x.C / g.Groups
	hw := x.H * x.W
	for n := 0; n < x.N; n++ {
		parallelFor(g.Groups, func(gi int) {
			c0 := gi * perGroup
			// Mean and variance in float64: a group here is up to 512*128*128
			// elements and a float32 running sum loses the low bits long
			// before the end of that.
			var sum, sumSq float64
			for c := c0; c < c0+perGroup; c++ {
				p := x.Plane(n, c)
				for _, v := range p {
					sum += float64(v)
					sumSq += float64(v) * float64(v)
				}
			}
			count := float64(perGroup * hw)
			mean := sum / count
			variance := sumSq/count - mean*mean
			if variance < 0 {
				variance = 0
			}
			inv := float32(1 / sqrt64(variance+g.Eps))
			fmean := float32(mean)
			for c := c0; c < c0+perGroup; c++ {
				p := x.Plane(n, c)
				w, b := g.Weight[c], g.Bias[c]
				scale := w * inv
				shift := b - fmean*scale
				for i, v := range p {
					p[i] = v*scale + shift
				}
			}
		})
	}
	return x, nil
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

// Subsample2x keeps the pixel at (2h+1, 2w+1) of every 2x2 block, halving H
// and W. It is the second half of the device's stride-2 convolution: the
// filter runs at stride 1 with symmetric padding and this throws three
// pixels in four away. TestStrideTwoIsStrideOneSubsampled is why that offset
// and not the other, and builder.downsample is what it costs.
func Subsample2x(x *Tensor) *Tensor {
	out := NewTensor(x.N, x.C, x.H/2, x.W/2)
	for n := 0; n < x.N; n++ {
		parallelFor(x.C, func(c int) {
			src, dst := x.Plane(n, c), out.Plane(n, c)
			for h := 0; h < out.H; h++ {
				srow := src[(h*2+1)*x.W:]
				drow := dst[h*out.W : (h+1)*out.W]
				for w := range drow {
					drow[w] = srow[w*2+1]
				}
			}
		})
	}
	return out
}
