package audio

import (
	"fmt"
	"math"
)

// Resample converts a clip to another sample rate with a windowed-sinc
// polyphase filter.
//
// It exists because the two speech verticals do not agree on a rate and
// nothing else in the repository crosses them: kokoro's vocoder emits 24 kHz
// (`kokoro.Config.SamplingRate`, fixed by the iSTFT head) and parakeet's
// front end reads 16 kHz, so `text -> speech -> text` needs a filter in the
// middle. DecodeWAV still reports a rate rather than converting it, and
// backend.STT still refuses a clip at the wrong one -- a resample is a
// decision a caller makes, not something a door does quietly.
//
// The filter is the standard one rather than an interpolation: downsampling
// 24 kHz to 16 kHz moves the Nyquist from 12 kHz to 8, so everything the
// source held between the two has to be removed *before* the decimation or it
// folds back as a tone that was never spoken. That matters here more than it
// looks like it should -- the consumer is an ASR model whose mel front end
// has 80 filters up to 8 kHz, and aliased energy lands in them as a feature.
//
// Cost is one kernel per output phase times its taps: 24 kHz to 16 kHz is
// L=2, M=3, 97 taps, about 4 ms per second of audio on this part.
func Resample(c *Clip, rate int) (*Clip, error) {
	if c == nil {
		return nil, fmt.Errorf("audio: resample: no clip")
	}
	if rate <= 0 {
		return nil, fmt.Errorf("audio: resample: target rate is %d", rate)
	}
	if c.Rate <= 0 {
		return nil, fmt.Errorf("audio: resample: the clip's rate is %d", c.Rate)
	}
	if c.Rate == rate {
		out := make([]float32, len(c.Samples))
		copy(out, c.Samples)
		return &Clip{Samples: out, Rate: rate}, nil
	}

	// The ratio in lowest terms: L outputs for every M inputs. Reducing it
	// is what makes this polyphase -- the fractional part of an output's
	// position in the input repeats with period L, so there are L distinct
	// kernels and not one per sample.
	g := gcd(c.Rate, rate)
	L, M := rate/g, c.Rate/g

	kernels, taps := sincKernels(L, M)
	n := len(c.Samples)
	// The output holds the same span of time: n/c.Rate seconds is n*L/M
	// samples at the new rate, rounded up so a clip that does not divide
	// evenly keeps its last partial period.
	nOut := (n*L + M - 1) / M
	out := make([]float32, nOut)
	half := (taps - 1) / 2
	for i := range out {
		// Output i sits at input position (i*M)/L, whose integer part
		// chooses the tap window and whose remainder chooses the kernel.
		q, r := i*M/L, i*M%L
		k := kernels[r]
		// Zero outside the clip rather than clamped: a waveform's
		// neighbourhood before it starts is silence, and holding the first
		// sample instead would put a step into the filter.
		lo, hi := q-half, q-half+taps
		j := 0
		if lo < 0 {
			j = -lo
			lo = 0
		}
		if hi > n {
			hi = n
		}
		var sum float32
		for s := lo; s < hi; s, j = s+1, j+1 {
			sum += k[j] * c.Samples[s]
		}
		out[i] = sum
	}
	return &Clip{Samples: out, Rate: rate}, nil
}

// zeros is how many sinc zero crossings the kernel keeps each side of centre,
// and rolloff is where the passband ends as a fraction of the new Nyquist.
//
// 32 and 0.95 are libsoxr's "high quality" neighbourhood and are chosen for
// the stopband: the point of the filter is that content above the new Nyquist
// does not survive the decimation, and a short kernel leaks it. At these two
// a 10 kHz tone resampled 24 kHz -> 16 kHz comes out 60 dB down, which
// `TestResampleRejectsAboveNyquist` is the measurement of.
const (
	zeros   = 32
	rolloff = 0.95
)

// sincKernels builds one kernel per output phase for an L/M conversion, and
// reports how many taps each has.
//
// The cutoff is expressed against the *input* Nyquist because that is the
// axis the taps are spaced on: upsampling keeps the whole input band (the new
// Nyquist is higher, so there is nothing to remove), and downsampling has to
// stop at the output's.
func sincKernels(L, M int) ([][]float32, int) {
	cutoff := rolloff
	if M > L {
		cutoff *= float64(L) / float64(M)
	}
	// The kernel spans `zeros` crossings of sinc(cutoff*x), so it widens as
	// the cutoff falls. Odd, so it has a centre tap.
	half := int(math.Ceil(zeros / cutoff))
	taps := 2*half + 1

	kernels := make([][]float32, L)
	for p := range kernels {
		// This phase's centre is p/L of a sample past an input sample.
		frac := float64(p) / float64(L)
		k := make([]float32, taps)
		var sum float64
		for i := range k {
			x := float64(i-half) - frac
			w := sinc(cutoff*x) * blackman(x, float64(half))
			k[i] = float32(w)
			sum += w
		}
		// Normalised per phase so the DC gain is exactly one. Without it
		// each phase has its own slightly different sum and a steady tone
		// picks up an L-periodic amplitude ripple -- which is a modulation
		// at rate/L Hz, i.e. an artefact inside the band.
		if sum != 0 {
			for i := range k {
				k[i] = float32(float64(k[i]) / sum)
			}
		}
		kernels[p] = k
	}
	return kernels, taps
}

// sinc is the normalised sinc, sin(pi x)/(pi x), with the removable
// singularity at zero filled in.
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// blackman is the window that tapers the kernel to zero at +/- half. A
// rectangular window would give the truncated sinc's first sidelobe at
// -13 dB, which is audible aliasing; Blackman's is -58.
func blackman(x, half float64) float64 {
	if x < -half || x > half {
		return 0
	}
	t := math.Pi * x / half
	return 0.42 + 0.5*math.Cos(t) + 0.08*math.Cos(2*t)
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
