package audio

import (
	"fmt"
	"math"
)

// HannWindow is the raised cosine window.
//
// `periodic` picks between the two conventions that differ by one sample of
// denominator: periodic (`sym=False`, the STFT default in most libraries)
// divides by n, symmetric divides by n-1. Parakeet's feature extractor asks
// torch for `hann_window(400, periodic=False)`, i.e. the symmetric one, which
// is not the default in either torch or librosa — so it is a parameter here
// rather than a constant.
func HannWindow(n int, periodic bool) []float64 {
	den := float64(n - 1)
	if periodic {
		den = float64(n)
	}
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/den)
	}
	return w
}

// PadMode is how a centred transform extends the signal past its ends.
type PadMode int

const (
	// PadZero is torch's `pad_mode="constant"`, which parakeet's feature
	// extractor asks for explicitly.
	PadZero PadMode = iota
	// PadReflect is torch.stft's *default*, and therefore what kokoro's
	// vocoder uses — it never names a pad mode. The reflection does not
	// repeat the edge sample: index -1 reads sample 1.
	PadReflect
)

// STFT is a short-time Fourier transform with a fixed frame geometry.
//
// It reproduces `torch.stft(..., center=True)`: the signal is padded by
// n_fft/2 on both sides so that frame t is centred on sample t*hop, a window
// shorter than n_fft is zero-padded symmetrically into it, and the frame
// count is 1 + len(padded-n_fft)/hop. Pad selects between torch's default
// reflection and the zero padding parakeet asks for; it is a field rather
// than a constructor argument because the zero value is the one the front end
// was written against.
type STFT struct {
	NFFT   int
	Hop    int
	Center bool
	Pad    PadMode

	window []float64 // length NFFT, already zero-padded
	fft    transformer
}

// NewSTFT builds a transform. window may be shorter than nFFT, in which case
// it is centred inside a zero-padded frame the way torch pads it.
func NewSTFT(nFFT, hop int, window []float64, center bool) (*STFT, error) {
	if len(window) > nFFT {
		return nil, fmt.Errorf("audio: window of %d samples does not fit in an %d-point transform", len(window), nFFT)
	}
	if hop <= 0 {
		return nil, fmt.Errorf("audio: hop %d", hop)
	}
	f := newTransform(nFFT, false)
	padded := make([]float64, nFFT)
	copy(padded[(nFFT-len(window))/2:], window)
	return &STFT{NFFT: nFFT, Hop: hop, Center: center, window: padded, fft: f}, nil
}

// Bins is the number of non-redundant frequency bins, n_fft/2 + 1.
func (s *STFT) Bins() int { return s.NFFT/2 + 1 }

// Frames is how many frames a signal of n samples produces.
func (s *STFT) Frames(n int) int {
	if s.Center {
		n += 2 * (s.NFFT / 2)
	}
	if n < s.NFFT {
		return 0
	}
	return 1 + (n-s.NFFT)/s.Hop
}

// sample reads x at an index that may fall outside it, applying Pad.
func (s *STFT) sample(x []float32, j int) float64 {
	if j >= 0 && j < len(x) {
		return float64(x[j])
	}
	if s.Pad != PadReflect || len(x) < 2 {
		return 0
	}
	// Mirror without repeating the edge: -1 -> 1, n -> n-2. Folded in a loop
	// so a pad wider than the signal still terminates, which torch would
	// reject outright.
	n := len(x)
	for j < 0 || j >= n {
		if j < 0 {
			j = -j
		}
		if j >= n {
			j = 2*(n-1) - j
		}
	}
	return float64(x[j])
}

// Power returns the power spectrogram |X|^2 as frames x Bins(), row major.
//
// Power rather than magnitude because that is what a mel filterbank is
// applied to here: parakeet's extractor takes `abs(stft)` and squares it,
// which is the same number by a different route.
func (s *STFT) Power(x []float32) []float64 {
	frames := s.Frames(len(x))
	out := make([]float64, frames*s.Bins())
	s.forEachFrame(x, frames, func(t int, buf []complex128) {
		row := out[t*s.Bins():]
		for b := 0; b < s.Bins(); b++ {
			re, im := real(buf[b]), imag(buf[b])
			row[b] = re*re + im*im
		}
	})
	return out
}

// MagnitudePhase returns |X| and arg(X) as two frames x Bins() arrays, which
// is what `TorchSTFT.transform` hands kokoro's generator: the harmonic
// excitation is fed to the noise convolutions as a magnitude spectrogram and
// a phase spectrogram stacked, not as a complex one.
func (s *STFT) MagnitudePhase(x []float32) (magnitude, phase []float64) {
	frames := s.Frames(len(x))
	magnitude = make([]float64, frames*s.Bins())
	phase = make([]float64, frames*s.Bins())
	s.forEachFrame(x, frames, func(t int, buf []complex128) {
		for b := 0; b < s.Bins(); b++ {
			re, im := real(buf[b]), imag(buf[b])
			magnitude[t*s.Bins()+b] = math.Hypot(re, im)
			phase[t*s.Bins()+b] = math.Atan2(im, re)
		}
	})
	return magnitude, phase
}

// forEachFrame windows, transforms and hands each frame's spectrum to fn.
func (s *STFT) forEachFrame(x []float32, frames int, fn func(t int, buf []complex128)) {
	buf := make([]complex128, s.NFFT)
	pad := 0
	if s.Center {
		pad = s.NFFT / 2
	}
	for t := 0; t < frames; t++ {
		start := t*s.Hop - pad
		for i := 0; i < s.NFFT; i++ {
			buf[i] = complex(s.sample(x, start+i)*s.window[i], 0)
		}
		s.fft.Transform(buf)
		fn(t, buf)
	}
}

// MelScale selects the frequency warping.
type MelScale int

const (
	// Slaney is the piecewise linear-then-log scale from the Auditory
	// Toolbox, which is librosa's default and what parakeet's filters are.
	Slaney MelScale = iota
	// HTK is the single log formula, 2595*log10(1+f/700).
	HTK
)

// HzToMel converts a frequency to the mel scale.
func HzToMel(f float64, scale MelScale) float64 {
	if scale == HTK {
		return 2595 * math.Log10(1+f/700)
	}
	const fSp = 200.0 / 3
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27
	if f >= minLogHz {
		return minLogMel + math.Log(f/minLogHz)/logStep
	}
	return f / fSp
}

// MelToHz is the inverse of HzToMel.
func MelToHz(m float64, scale MelScale) float64 {
	if scale == HTK {
		return 700 * (math.Pow(10, m/2595) - 1)
	}
	const fSp = 200.0 / 3
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27
	if m >= minLogMel {
		return minLogHz * math.Exp(logStep*(m-minLogMel))
	}
	return fSp * m
}

// MelFilters builds a triangular mel filterbank as nMels x (nFFT/2+1), row
// major — the matrix librosa's `filters.mel` returns.
//
// `slaneyNorm` divides each filter by its width in Hz so that the bank is
// approximately area-normalised (librosa's `norm="slaney"`, which is its
// default and what parakeet was trained with); without it every triangle
// peaks at 1.
//
// The arithmetic is float64 and the result is float32, which is what librosa
// does — and the difference matters enough that transformers' feature
// extractor keeps a comment about it, having found that computing the bank in
// float64 end to end moves the features.
func MelFilters(sampleRate, nFFT, nMels int, fMin, fMax float64, scale MelScale, slaneyNorm bool) []float32 {
	bins := nFFT/2 + 1
	fftFreqs := make([]float64, bins)
	for i := range fftFreqs {
		// linspace(0, sr/2, bins), which is rfftfreq to the last bit.
		fftFreqs[i] = float64(sampleRate) / 2 * float64(i) / float64(bins-1)
	}
	// nMels+2 band edges: filter i rises from edge i to i+1 and falls to i+2.
	edges := make([]float64, nMels+2)
	lo, hi := HzToMel(fMin, scale), HzToMel(fMax, scale)
	for i := range edges {
		edges[i] = MelToHz(lo+(hi-lo)*float64(i)/float64(nMels+1), scale)
	}

	w := make([]float32, nMels*bins)
	for m := 0; m < nMels; m++ {
		lowDiff := edges[m+1] - edges[m]
		highDiff := edges[m+2] - edges[m+1]
		norm := 1.0
		if slaneyNorm {
			norm = 2 / (edges[m+2] - edges[m])
		}
		for b, f := range fftFreqs {
			lower := (f - edges[m]) / lowDiff
			upper := (edges[m+2] - f) / highDiff
			v := math.Min(lower, upper)
			if v > 0 {
				w[m*bins+b] = float32(v * norm)
			}
		}
	}
	return w
}
