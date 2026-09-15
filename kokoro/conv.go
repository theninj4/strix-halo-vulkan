package kokoro

import "fmt"

// Conv1D is a 1-D convolution over time, weights stored [Out, In/Groups, K] —
// PyTorch's order, with `weight_norm` already folded in by
// reference/convert_kokoro.py.
//
// Held against a channel-last activation this is a GEMM when K is 1 (which is
// why the loader turns those into a Linear instead) and a K-tap filter over
// contiguous channel vectors otherwise. Padding is zero padding, as torch's
// `padding=n` is.
type Conv1D struct {
	In, Out  int
	Kernel   int
	Stride   int
	Pad      int
	Dilation int
	Groups   int
	Weight   []float32 // [Out, In/Groups, K]
	Bias     []float32 // [Out], nil where the checkpoint has none
}

// OutFrames is the output length for an input of t frames.
func (c *Conv1D) OutFrames(t int) int {
	eff := c.Dilation*(c.Kernel-1) + 1
	if n := (t+2*c.Pad-eff)/c.Stride + 1; n > 0 {
		return n
	}
	return 0
}

// Apply runs the convolution over a [T, In] activation.
func (c *Conv1D) Apply(x *Mat) (*Mat, error) {
	if x.Cols != c.In {
		return nil, fmt.Errorf("kokoro: conv1d takes %d channels, got %d", c.In, x.Cols)
	}
	if c.Groups <= 0 || c.In%c.Groups != 0 || c.Out%c.Groups != 0 {
		return nil, fmt.Errorf("kokoro: conv1d with %d groups over %d->%d channels", c.Groups, c.In, c.Out)
	}
	inPer, outPer := c.In/c.Groups, c.Out/c.Groups
	out := NewMat(c.OutFrames(x.Rows), c.Out)
	parallelFor(c.Out, func(o int) {
		g := o / outPer
		w := c.Weight[o*inPer*c.Kernel : (o+1)*inPer*c.Kernel]
		var bias float32
		if c.Bias != nil {
			bias = c.Bias[o]
		}
		for t := 0; t < out.Rows; t++ {
			sum := bias
			base := t*c.Stride - c.Pad
			for k := 0; k < c.Kernel; k++ {
				src := base + k*c.Dilation
				if src < 0 || src >= x.Rows {
					continue
				}
				row := x.Data[src*x.Cols+g*inPer:]
				for i := 0; i < inPer; i++ {
					sum += w[i*c.Kernel+k] * row[i]
				}
			}
			out.Data[t*c.Out+o] = sum
		}
	})
	return out, nil
}

// ConvTranspose1D is the transposed convolution, weights stored
// [In, Out/Groups, K] — note the axis order is the *opposite* of Conv1D's,
// which is why convert_kokoro.py's weight_norm fold takes the norm over
// dimensions 1 and 2 of that rather than assuming a flattened output row.
//
// Two of these exist in the model and they are different animals: the AdaIN
// blocks' `pool` is depthwise (groups = channels, one tap per channel) and
// doubles the frame count, and the generator's `ups` are dense and multiply
// it by 10 and by 6. Both are covered here.
type ConvTranspose1D struct {
	In, Out       int
	Kernel        int
	Stride        int
	Pad           int
	OutputPadding int
	Groups        int
	Weight        []float32 // [In, Out/Groups, K]
	Bias          []float32
}

// OutFrames is the output length for an input of t frames.
func (c *ConvTranspose1D) OutFrames(t int) int {
	return (t-1)*c.Stride - 2*c.Pad + c.Kernel + c.OutputPadding
}

// Apply runs the transposed convolution over a [T, In] activation.
//
// Written as a gather rather than a scatter: output frame j sums the input
// frames that would have contributed to it, so the loop over outputs is
// parallel and nothing is accumulated across goroutines.
func (c *ConvTranspose1D) Apply(x *Mat) (*Mat, error) {
	if x.Cols != c.In {
		return nil, fmt.Errorf("kokoro: conv transpose takes %d channels, got %d", c.In, x.Cols)
	}
	if c.Groups <= 0 || c.In%c.Groups != 0 || c.Out%c.Groups != 0 {
		return nil, fmt.Errorf("kokoro: conv transpose with %d groups over %d->%d channels", c.Groups, c.In, c.Out)
	}
	inPer, outPer := c.In/c.Groups, c.Out/c.Groups
	out := NewMat(c.OutFrames(x.Rows), c.Out)
	parallelFor(c.Out, func(o int) {
		g := o / outPer
		og := o % outPer
		var bias float32
		if c.Bias != nil {
			bias = c.Bias[o]
		}
		for j := 0; j < out.Rows; j++ {
			sum := bias
			for k := 0; k < c.Kernel; k++ {
				// j = t*stride + k - pad, so t = (j + pad - k)/stride.
				n := j + c.Pad - k
				if n < 0 || n%c.Stride != 0 {
					continue
				}
				t := n / c.Stride
				if t >= x.Rows {
					continue
				}
				for i := 0; i < inPer; i++ {
					in := g*inPer + i
					sum += c.Weight[(in*outPer+og)*c.Kernel+k] * x.Data[t*x.Cols+in]
				}
			}
			out.Data[j*c.Out+o] = sum
		}
	})
	return out, nil
}

// UpsampleNearest repeats every frame `factor` times, which is
// F.interpolate(mode="nearest") over the time axis: the shortcut path of an
// upsampling AdaIN block, and the only resampler in the model that is not a
// convolution.
func UpsampleNearest(x *Mat, factor int) *Mat {
	out := NewMat(x.Rows*factor, x.Cols)
	for r := 0; r < out.Rows; r++ {
		copy(out.Row(r), x.Row(r/factor))
	}
	return out
}
