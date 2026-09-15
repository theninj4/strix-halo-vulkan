package audio

import "math"

// transformer is what STFT and ISTFT need of a discrete Fourier transform:
// an in-place complex transform of a fixed size. FFT implements it for powers
// of two and DFT for everything else.
type transformer interface {
	Transform(buf []complex128)
	Size() int
}

// DFT is the direct O(n^2) discrete Fourier transform, for sizes the radix-2
// FFT cannot take.
//
// It exists for exactly one size: kokoro's vocoder ends in a **20-point**
// transform, one per output hop, and 20 is 4x5. At that size the direct form
// is not a compromise — a 20-point transform is 220 complex multiplies
// against the bookkeeping of a mixed-radix decomposition, and the whole
// vocoder's 15601 frames come to a few million operations against the
// billions in the convolutions above it. Writing Bluestein for this would be
// slower and would have to be tested.
type DFT struct {
	n  int
	tw []complex128 // e^(±2πi jk/n) for jk mod n, one row of n
}

// NewDFT builds a transform of size n. Unlike NewFFT it takes any n >= 1.
func NewDFT(n int, inverse bool) *DFT {
	sign := -2 * math.Pi / float64(n)
	if inverse {
		sign = -sign
	}
	d := &DFT{n: n, tw: make([]complex128, n)}
	for i := range d.tw {
		d.tw[i] = complex(math.Cos(sign*float64(i)), math.Sin(sign*float64(i)))
	}
	return d
}

// Size is the transform length.
func (d *DFT) Size() int { return d.n }

// Transform rewrites buf in place. The inverse transform is unnormalised, the
// same convention FFT uses.
func (d *DFT) Transform(buf []complex128) {
	if len(buf) != d.n {
		panic("audio: DFT applied to the wrong length")
	}
	out := make([]complex128, d.n)
	for k := range out {
		var sum complex128
		// The twiddle index is j*k mod n, accumulated rather than multiplied
		// so the inner loop has no integer division.
		idx := 0
		for j := 0; j < d.n; j++ {
			sum += buf[j] * d.tw[idx]
			if idx += k; idx >= d.n {
				idx -= d.n
			}
		}
		out[k] = sum
	}
	copy(buf, out)
}

// newTransform picks the FFT when it can and the DFT when it cannot.
func newTransform(n int, inverse bool) transformer {
	if n >= 2 && n&(n-1) == 0 {
		f, err := NewFFT(n, inverse)
		if err == nil {
			return f
		}
	}
	return NewDFT(n, inverse)
}
