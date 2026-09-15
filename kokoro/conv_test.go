package kokoro

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The convolutions are the one part of this package written as something
// other than the definition: Conv1D parallelises over output channels and
// ConvTranspose1D is a *gather* — output frame j sums the inputs that would
// have contributed to it — where the natural form of a transposed
// convolution is a scatter. Both rewrites are checked here against the
// straightforward form, over shapes the reference dump does not reach:
// dilation, which only the vocoder's resblocks use (T3), and a dense
// transposed convolution, which only its upsamplers do.
//
// The dump checks the shapes the model actually runs; this checks the
// operators either side of them.

func randSlice(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// naiveConv1D is torch's definition, transcribed: out[t, o] is a sum over the
// kernel and over the group's input channels, zero outside the input.
func naiveConv1D(c *Conv1D, x *Mat) *Mat {
	inPer, outPer := c.In/c.Groups, c.Out/c.Groups
	out := NewMat(c.OutFrames(x.Rows), c.Out)
	for t := 0; t < out.Rows; t++ {
		for o := 0; o < c.Out; o++ {
			g := o / outPer
			var sum float32
			if c.Bias != nil {
				sum = c.Bias[o]
			}
			for i := 0; i < inPer; i++ {
				for k := 0; k < c.Kernel; k++ {
					src := t*c.Stride - c.Pad + k*c.Dilation
					if src < 0 || src >= x.Rows {
						continue
					}
					sum += c.Weight[(o*inPer+i)*c.Kernel+k] * x.Data[src*x.Cols+g*inPer+i]
				}
			}
			out.Data[t*c.Out+o] = sum
		}
	}
	return out
}

// scatterConvTranspose is the other definition: every input frame writes its
// kernel into the output, which is how the operator is usually stated and the
// opposite traversal from the one Apply uses.
func scatterConvTranspose(c *ConvTranspose1D, x *Mat) *Mat {
	inPer, outPer := c.In/c.Groups, c.Out/c.Groups
	out := NewMat(c.OutFrames(x.Rows), c.Out)
	if c.Bias != nil {
		for j := 0; j < out.Rows; j++ {
			copy(out.Row(j), c.Bias)
		}
	}
	for t := 0; t < x.Rows; t++ {
		for in := 0; in < c.In; in++ {
			g := in / inPer
			for og := 0; og < outPer; og++ {
				for k := 0; k < c.Kernel; k++ {
					j := t*c.Stride + k - c.Pad
					if j < 0 || j >= out.Rows {
						continue
					}
					out.Data[j*c.Out+g*outPer+og] +=
						c.Weight[(in*outPer+og)*c.Kernel+k] * x.Data[t*x.Cols+in]
				}
			}
		}
	}
	return out
}

func maxAbs(a, b *Mat, t *testing.T) float64 {
	t.Helper()
	if a.Rows != b.Rows || a.Cols != b.Cols {
		t.Fatalf("shapes %v and %v", a, b)
	}
	var worst float64
	for i := range a.Data {
		if d := math.Abs(float64(a.Data[i]) - float64(b.Data[i])); d > worst {
			worst = d
		}
	}
	return worst
}

func TestConv1D(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	cases := []struct {
		name                                  string
		in, out, kernel, stride, pad, dil, gr int
		frames                                int
		bias                                  bool
	}{
		{name: "text encoder", in: 8, out: 8, kernel: 5, stride: 1, pad: 2, dil: 1, gr: 1, frames: 11, bias: true},
		{name: "adain 3-tap", in: 6, out: 4, kernel: 3, stride: 1, pad: 1, dil: 1, gr: 1, frames: 9, bias: true},
		{name: "pointwise", in: 5, out: 3, kernel: 1, stride: 1, pad: 0, dil: 1, gr: 1, frames: 7},
		{name: "dilated 5", in: 4, out: 4, kernel: 3, stride: 1, pad: 5, dil: 5, gr: 1, frames: 13, bias: true},
		{name: "dilated 3 wide", in: 4, out: 4, kernel: 7, stride: 1, pad: 9, dil: 3, gr: 1, frames: 13},
		{name: "F0 decimation", in: 1, out: 1, kernel: 3, stride: 2, pad: 1, dil: 1, gr: 1, frames: 10, bias: true},
		{name: "grouped", in: 6, out: 9, kernel: 3, stride: 1, pad: 1, dil: 1, gr: 3, frames: 8, bias: true},
		{name: "depthwise", in: 4, out: 4, kernel: 3, stride: 1, pad: 1, dil: 1, gr: 4, frames: 6},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			conv := &Conv1D{
				In: c.in, Out: c.out, Kernel: c.kernel, Stride: c.stride,
				Pad: c.pad, Dilation: c.dil, Groups: c.gr,
				Weight: randSlice(r, c.out*(c.in/c.gr)*c.kernel),
			}
			if c.bias {
				conv.Bias = randSlice(r, c.out)
			}
			x := &Mat{Rows: c.frames, Cols: c.in, Data: randSlice(r, c.frames*c.in)}
			got, err := conv.Apply(x)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbs(got, naiveConv1D(conv, x), t); d > 1e-5 {
				t.Errorf("max abs %g against the definition", d)
			}
		})
	}
	bad := &Conv1D{In: 4, Out: 4, Kernel: 1, Stride: 1, Dilation: 1, Groups: 3, Weight: make([]float32, 4)}
	if _, err := bad.Apply(NewMat(2, 4)); err == nil {
		t.Error("4 channels split into 3 groups")
	}
}

func TestConvTranspose1D(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	cases := []struct {
		name                                 string
		in, out, kernel, stride, pad, op, gr int
		frames                               int
		bias                                 bool
	}{
		// The AdaIN blocks' `pool`: depthwise, and the one combination that
		// takes T frames to exactly 2T.
		{name: "adain pool", in: 6, out: 6, kernel: 3, stride: 2, pad: 1, op: 1, gr: 6, frames: 9, bias: true},
		// The generator's two upsamplers, at T3's rates.
		{name: "ups 10x", in: 8, out: 4, kernel: 20, stride: 10, pad: 5, gr: 1, frames: 5, bias: true},
		{name: "ups 6x", in: 4, out: 2, kernel: 12, stride: 6, pad: 3, gr: 1, frames: 7, bias: true},
		{name: "no bias", in: 3, out: 3, kernel: 3, stride: 2, pad: 1, op: 1, gr: 3, frames: 4},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			conv := &ConvTranspose1D{
				In: c.in, Out: c.out, Kernel: c.kernel, Stride: c.stride,
				Pad: c.pad, OutputPadding: c.op, Groups: c.gr,
				Weight: randSlice(r, c.in*(c.out/c.gr)*c.kernel),
			}
			if c.bias {
				conv.Bias = randSlice(r, c.out)
			}
			x := &Mat{Rows: c.frames, Cols: c.in, Data: randSlice(r, c.frames*c.in)}
			got, err := conv.Apply(x)
			if err != nil {
				t.Fatal(err)
			}
			want := scatterConvTranspose(conv, x)
			if d := maxAbs(got, want, t); d > 1e-5 {
				t.Errorf("max abs %g between the gather and the scatter", d)
			}
			t.Logf("%d frames -> %d", x.Rows, got.Rows)
		})
	}

	// The length agreement the upsampling AdaIN block depends on: its
	// residual path is this convolution and its shortcut is a plain repeat,
	// and they have to produce the same number of frames.
	pool := &ConvTranspose1D{In: 4, Out: 4, Kernel: 3, Stride: 2, Pad: 1, OutputPadding: 1, Groups: 4,
		Weight: randSlice(r, 4*1*3)}
	for _, frames := range []int{1, 2, 5, 130} {
		if got := pool.OutFrames(frames); got != 2*frames {
			t.Errorf("pool takes %d frames to %d, want %d", frames, got, 2*frames)
		}
		if got := UpsampleNearest(NewMat(frames, 4), 2).Rows; got != 2*frames {
			t.Errorf("nearest repeat takes %d frames to %d", frames, got)
		}
	}
}

// TestInstanceNormIsNotLayerNorm pins the axis that AdaIN normalises over.
// The two are the same operation transposed, so a port that picks the wrong
// one still produces plausible numbers of the right shape.
func TestInstanceNormIsNotLayerNorm(t *testing.T) {
	x := &Mat{Rows: 3, Cols: 2, Data: []float32{1, 10, 2, 20, 3, 30}}
	got := x.Clone()
	InstanceNormInPlace(got, 0)
	// Each column is (1,2,3) and (10,20,30): both normalise to the same
	// standardised triple, which a layer norm over the rows could not do.
	want := []float32{-1.2247449, -1.2247449, 0, 0, 1.2247449, 1.2247449}
	for i, w := range want {
		if math.Abs(float64(got.Data[i]-w)) > 1e-5 {
			t.Fatalf("instance norm = %v, want %v", got.Data, want)
		}
	}
	ln := &LayerNorm{Width: 2, Eps: 0}
	row := x.Clone()
	if err := ln.ApplyInPlace(row); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(row.Data[0]-got.Data[0])) < 1e-3 {
		t.Error("layer norm and instance norm agree; the test shape is degenerate")
	}
}

// TestTransposedConvDecomposition checks the identity the device upsampler is
// built on, on the CPU and with no Vulkan in sight.
//
// A ConvTranspose1d with kernel = 2*stride and padding = stride/2 — which both
// of the generator's upsamplers are — reaches every output frame with exactly
// two taps, and which two is decided by the residue of the frame index. So
// output frame q*s + r - s/2 is
//
//	sum_i x[q-1, i] * W[i, n, r+s]  +  x[q, i] * W[i, n, r]
//
// which is an ordinary two-tap convolution whose output is s times wider:
// [T+1, s*C_out], read back as [(T+1)*s, C_out]. That is what turns the
// upsampler into one GEMM with no strided store and no residue loop, and it is
// the claim worth pinning, because getting it wrong produces a signal of the
// right length that is wrong everywhere.
func TestTransposedConvDecomposition(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for _, c := range []struct{ in, out, stride, frames int }{
		{8, 4, 10, 7}, // the shape of ups[0], narrowed
		{4, 2, 6, 11}, // ups[1]
		{3, 3, 2, 5},  // the smallest case the identity covers
	} {
		name := fmt.Sprintf("%dx%d_s%d", c.in, c.out, c.stride)
		t.Run(name, func(t *testing.T) {
			k := 2 * c.stride
			conv := &ConvTranspose1D{
				In: c.in, Out: c.out, Kernel: k, Stride: c.stride, Pad: c.stride / 2, Groups: 1,
				Weight: randSlice(r, c.in*c.out*k), Bias: randSlice(r, c.out),
			}
			x := &Mat{Rows: c.frames, Cols: c.in, Data: randSlice(r, c.frames*c.in)}
			want, err := conv.Apply(x)
			if err != nil {
				t.Fatal(err)
			}
			if want.Rows != c.frames*c.stride {
				t.Fatalf("%d frames out, want %d", want.Rows, c.frames*c.stride)
			}

			// The two-tap form: row q reads input rows q-1 and q (zero outside
			// the signal, which is what the arena's border provides) and
			// writes s*C_out columns.
			wide := NewMat(c.frames+1, c.stride*c.out)
			for q := 0; q <= c.frames; q++ {
				for rr := 0; rr < c.stride; rr++ {
					for n := 0; n < c.out; n++ {
						var sum float32
						for i := 0; i < c.in; i++ {
							if q > 0 {
								sum += x.Data[(q-1)*c.in+i] * conv.Weight[(i*c.out+n)*k+rr+c.stride]
							}
							if q < c.frames {
								sum += x.Data[q*c.in+i] * conv.Weight[(i*c.out+n)*k+rr]
							}
						}
						wide.Data[q*c.stride*c.out+rr*c.out+n] = sum
					}
				}
			}
			// Read as [(T+1)*s, C_out] and drop the leading stride/2 rows.
			off := c.stride / 2
			var worst float64
			for o := 0; o < want.Rows; o++ {
				for n := 0; n < c.out; n++ {
					got := wide.Data[(off+o)*c.out+n] + conv.Bias[n]
					if d := math.Abs(float64(got - want.Data[o*c.out+n])); d > worst {
						worst = d
					}
				}
			}
			if worst > 1e-4 {
				t.Errorf("max abs %g against ConvTranspose1D", worst)
			}
		})
	}
}
