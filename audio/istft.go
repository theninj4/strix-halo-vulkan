package audio

import (
	"fmt"
	"math"
)

// ISTFT is the inverse short-time Fourier transform: weighted overlap-add
// with the window-square envelope divided back out.
//
// It reproduces `torch.istft(..., center=True, normalized=False,
// onesided=True)`, which is how kokoro's vocoder turns its predicted
// magnitude and phase into samples. The forward STFT above is not enough on
// its own — the analysis window has already been applied once, the synthesis
// applies it again, and what makes the round trip exact is dividing by the
// sum of the squared windows at each sample rather than by their sum. That
// division is the whole of this type; the transform is the easy half.
//
// The geometry is fixed by the frame count: a run of F frames covers
// n_fft + hop*(F-1) samples before the centring pad is trimmed, and
// hop*(F-1) after. Kokoro's 15601 frames at hop 5 come to exactly 78000
// samples, which is 130 alignment frames of 600.
type ISTFT struct {
	NFFT   int
	Hop    int
	Center bool

	window []float64 // length NFFT, already zero-padded
	fft    transformer
}

// NewISTFT builds an inverse transform. The window is the *analysis* window —
// the same one the forward transform used — because the normalisation below
// assumes the two are the same.
func NewISTFT(nFFT, hop int, window []float64, center bool) (*ISTFT, error) {
	if len(window) > nFFT {
		return nil, fmt.Errorf("audio: window of %d samples does not fit in an %d-point transform", len(window), nFFT)
	}
	if hop <= 0 {
		return nil, fmt.Errorf("audio: hop %d", hop)
	}
	padded := make([]float64, nFFT)
	copy(padded[(nFFT-len(window))/2:], window)
	return &ISTFT{NFFT: nFFT, Hop: hop, Center: center, window: padded, fft: newTransform(nFFT, true)}, nil
}

// Bins is the number of non-redundant frequency bins the input carries.
func (s *ISTFT) Bins() int { return s.NFFT/2 + 1 }

// Window is the zero-padded analysis window, which is also the synthesis
// window and the thing the envelope is the square of. It is returned so that
// a device port can build the same overlap-add from the same numbers rather
// than reconstructing the padding rule.
func (s *ISTFT) Window() []float64 {
	out := make([]float64, len(s.window))
	copy(out, s.window)
	return out
}

// Samples is how many samples a run of frames produces.
func (s *ISTFT) Samples(frames int) int {
	if frames <= 0 {
		return 0
	}
	n := s.NFFT + s.Hop*(frames-1)
	if s.Center {
		n -= 2 * (s.NFFT / 2)
	}
	return n
}

// Apply reconstructs a signal from magnitude and phase, each frames x Bins()
// row major — the form kokoro's generator produces, where the two halves of
// `conv_post`'s output become exp(x) and sin(x) rather than a real and an
// imaginary part.
//
// The one-sided spectrum is mirrored back to a full one with the Hermitian
// symmetry an inverse real transform assumes, so bins 1..n/2-1 are counted
// twice and the DC and Nyquist bins once.
func (s *ISTFT) Apply(magnitude, phase []float64, frames int) ([]float32, error) {
	bins := s.Bins()
	if len(magnitude) != frames*bins || len(phase) != frames*bins {
		return nil, fmt.Errorf("audio: istft of %d frames wants %d values, got %d and %d",
			frames, frames*bins, len(magnitude), len(phase))
	}
	full := s.NFFT + s.Hop*(frames-1)
	if frames <= 0 {
		return nil, fmt.Errorf("audio: istft of %d frames", frames)
	}
	acc := make([]float64, full)
	env := make([]float64, full)
	buf := make([]complex128, s.NFFT)

	// The window-square envelope depends on nothing but the geometry, so it
	// is accumulated alongside rather than recomputed per frame.
	for t := 0; t < frames; t++ {
		row := t * bins
		for b := 0; b < s.NFFT; b++ {
			// Hermitian mirror: bin n-b is the conjugate of bin b.
			k, conj := b, false
			if k >= bins {
				k, conj = s.NFFT-b, true
			}
			m, p := magnitude[row+k], phase[row+k]
			if conj {
				p = -p
			}
			buf[b] = complex(m*math.Cos(p), m*math.Sin(p))
		}
		s.fft.Transform(buf)
		off := t * s.Hop
		norm := 1 / float64(s.NFFT)
		for i := 0; i < s.NFFT; i++ {
			w := s.window[i]
			acc[off+i] += real(buf[i]) * norm * w
			env[off+i] += w * w
		}
	}

	start, end := 0, full
	if s.Center {
		start, end = s.NFFT/2, full-s.NFFT/2
	}
	out := make([]float32, end-start)
	for i := range out {
		// torch rejects a window whose squares do not cover the signal
		// (the NOLA condition); here the uncovered samples are left at zero
		// rather than dividing by something near zero.
		if e := env[start+i]; e > 1e-11 {
			out[i] = float32(acc[start+i] / e)
		}
	}
	return out, nil
}
