package audio

import (
	"fmt"
	"math"
)

// FFT is a radix-2 complex discrete Fourier transform of a fixed size, with
// its twiddle factors and bit-reversal permutation computed once.
//
// Both verticals need one: parakeet's front end is 512-point forward
// transforms at one per 10 ms of audio, and kokoro's vocoder ends in a
// 20-point inverse transform per output hop. Neither is anywhere near a
// bottleneck — a 30 s clip is 3000 forward transforms, microseconds of work
// against an encoder that is billions of flops a frame — so this is the plain
// iterative Cooley-Tukey rather than anything clever, in float64 so that the
// transform is not the thing limiting how closely the Go front end tracks
// torch.stft.
type FFT struct {
	n       int
	rev     []int32      // bit-reversal permutation
	tw      []complex128 // twiddles, one per (stage, k) pair, laid out by stage
	inverse bool
}

// NewFFT builds a transform of size n, which must be a power of two.
func NewFFT(n int, inverse bool) (*FFT, error) {
	if n < 2 || n&(n-1) != 0 {
		return nil, fmt.Errorf("audio: FFT size %d is not a power of two", n)
	}
	f := &FFT{n: n, inverse: inverse, rev: make([]int32, n)}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := range f.rev {
		var r int
		for b := 0; b < bits; b++ {
			r |= (i >> b & 1) << (bits - 1 - b)
		}
		f.rev[i] = int32(r)
	}
	// One twiddle per butterfly offset per stage: stage of half-length h
	// needs h of them, so the whole table is n-1 entries.
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for h := 1; h < n; h <<= 1 {
		for k := 0; k < h; k++ {
			ang := sign * math.Pi * float64(k) / float64(h)
			f.tw = append(f.tw, complex(math.Cos(ang), math.Sin(ang)))
		}
	}
	return f, nil
}

// Size is the transform length.
func (f *FFT) Size() int { return f.n }

// Transform rewrites buf in place. len(buf) must be Size(). The inverse
// transform is unnormalised — it is the forward transform with a conjugated
// twiddle — so a caller that wants a round trip divides by Size().
func (f *FFT) Transform(buf []complex128) {
	if len(buf) != f.n {
		panic(fmt.Sprintf("audio: FFT of size %d applied to %d points", f.n, len(buf)))
	}
	for i, r := range f.rev {
		if int32(i) < r {
			buf[i], buf[r] = buf[r], buf[i]
		}
	}
	base := 0
	for h := 1; h < f.n; h <<= 1 {
		tw := f.tw[base : base+h]
		for off := 0; off < f.n; off += h << 1 {
			for k := 0; k < h; k++ {
				a := buf[off+k]
				b := buf[off+k+h] * tw[k]
				buf[off+k] = a + b
				buf[off+k+h] = a - b
			}
		}
		base += h
	}
}

// RealTransform is Transform for a real-valued input, returning the n/2+1
// bins that are not redundant. dst is appended to and may be nil.
func (f *FFT) RealTransform(dst []complex128, x []float64, scratch []complex128) []complex128 {
	if len(x) != f.n {
		panic(fmt.Sprintf("audio: FFT of size %d applied to %d real points", f.n, len(x)))
	}
	if cap(scratch) < f.n {
		scratch = make([]complex128, f.n)
	}
	scratch = scratch[:f.n]
	for i, v := range x {
		scratch[i] = complex(v, 0)
	}
	f.Transform(scratch)
	return append(dst, scratch[:f.n/2+1]...)
}
