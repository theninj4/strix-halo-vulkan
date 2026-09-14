package parakeet

import "fmt"

// Conv2D is a 2-D convolution over a Volume, with the geometry PyTorch's
// nn.Conv2d has: a square kernel, a square stride, symmetric zero padding,
// and weights stored [Out, In/groups, K, K].
//
// Depthwise is groups == channels, the only grouping this model uses. The
// subsampling stack is `1 -> 256` and then twice `depthwise + pointwise`,
// which is the separable form NeMo calls `dw_striding` — SPEECH.md's
// inventory read it as three plain convolutions from the collapsed shape
// table, and the checkpoint says otherwise: layers 2 and 5 are [256, 1, 3, 3],
// one filter per channel.
type Conv2D struct {
	Index     int // position in the checkpoint's ModuleList, for error messages
	In, Out   int
	Kernel    int
	Stride    int
	Pad       int
	Depthwise bool
	ReLUAfter bool

	Weight []float32
	Bias   []float32
}

// OutLength is the number of output positions along an axis of n inputs.
func (c *Conv2D) OutLength(n int) int { return (n+2*c.Pad-c.Kernel)/c.Stride + 1 }

// Apply convolves a feature map.
func (c *Conv2D) Apply(v *Volume) (*Volume, error) {
	if v.C != c.In {
		return nil, fmt.Errorf("parakeet: conv2d %d takes %d channels, got %d", c.Index, c.In, v.C)
	}
	if c.Depthwise && c.In != c.Out {
		return nil, fmt.Errorf("parakeet: depthwise conv2d %d cannot map %d channels to %d", c.Index, c.In, c.Out)
	}
	outT, outF := c.OutLength(v.T), c.OutLength(v.F)
	out := NewVolume(c.Out, outT, outF)

	k, s, p := c.Kernel, c.Stride, c.Pad
	inPerOut := c.In
	if c.Depthwise {
		inPerOut = 1
	}
	parallelFor(c.Out, func(o int) {
		w := c.Weight[o*inPerOut*k*k : (o+1)*inPerOut*k*k]
		dst := out.Plane(o)
		var bias float32
		if c.Bias != nil {
			bias = c.Bias[o]
		}
		for i := range dst {
			dst[i] = bias
		}
		for ci := 0; ci < inPerOut; ci++ {
			src := ci
			if c.Depthwise {
				src = o
			}
			plane := v.Plane(src)
			kw := w[ci*k*k : (ci+1)*k*k]
			for kt := 0; kt < k; kt++ {
				for kf := 0; kf < k; kf++ {
					weight := kw[kt*k+kf]
					if weight == 0 {
						continue
					}
					for ot := 0; ot < outT; ot++ {
						it := ot*s + kt - p
						if it < 0 || it >= v.T {
							continue
						}
						row := plane[it*v.F:]
						drow := dst[ot*outF:]
						for of := 0; of < outF; of++ {
							iff := of*s + kf - p
							if iff < 0 || iff >= v.F {
								continue
							}
							drow[of] += weight * row[iff]
						}
					}
				}
			}
		}
	})
	return out, nil
}

// Subsampling is the convolutional front of the encoder: three strided
// convolutions that take the mel spectrogram down by 8 in both time and
// frequency, then a linear that folds the surviving 256 channels x 16 mel
// bins into the encoder's 1024 features.
//
// The length bookkeeping is the subtle part and it is the library's, not an
// arithmetic identity: the *tensor* shrinks by the convolution formula over
// the padded input, while the *valid length* shrinks by
// `(L + 2*pad - kernel)/stride + 1` over the unpadded one, and the two
// disagree by a frame at each stage. Everything past the valid length is
// zeroed after each strided convolution, which is what stops the padding at
// the end of the clip from leaking into the encoder.
type Subsampling struct {
	Convs  []*Conv2D
	Linear *Linear
}

// ValidLength maps a count of valid mel frames to valid encoder frames.
func (s *Subsampling) ValidLength(n int) int {
	for _, c := range s.Convs {
		if c.Stride > 1 {
			n = c.OutLength(n)
		}
	}
	return n
}

// Apply runs the stack over a log-mel spectrogram, returning the encoder's
// input frames and how many of them are valid.
func (s *Subsampling) Apply(mel *Mat, valid int) (*Mat, int, error) {
	return s.forward(mel, valid, nil)
}

// forward is Apply with a hook that sees each convolution's output, which is
// what the stage tests walk.
func (s *Subsampling) forward(mel *Mat, valid int, trace func(conv *Conv2D, v *Volume)) (*Mat, int, error) {
	v := &Volume{C: 1, T: mel.Rows, F: mel.Cols, Data: mel.Data}
	for _, c := range s.Convs {
		out, err := c.Apply(v)
		if err != nil {
			return nil, 0, err
		}
		if c.Stride > 1 {
			valid = c.OutLength(valid)
		}
		// Zero every frame past the valid length, in every channel.
		if valid < out.T {
			for ch := 0; ch < out.C; ch++ {
				plane := out.Plane(ch)
				for i := valid * out.F; i < len(plane); i++ {
					plane[i] = 0
				}
			}
		}
		if trace != nil {
			trace(c, out)
		}
		if c.ReLUAfter {
			for i, x := range out.Data {
				out.Data[i] = ReLU(x)
			}
		}
		v = out
	}

	// [C, T, F] -> [T, C*F]: the transpose-and-flatten the module does before
	// its linear, with the channel the slower axis of the feature index.
	flat := NewMat(v.T, v.C*v.F)
	for t := 0; t < v.T; t++ {
		row := flat.Row(t)
		for c := 0; c < v.C; c++ {
			copy(row[c*v.F:(c+1)*v.F], v.Plane(c)[t*v.F:(t+1)*v.F])
		}
	}
	out, err := s.Linear.Apply(flat)
	if err != nil {
		return nil, 0, err
	}
	return out, valid, nil
}
