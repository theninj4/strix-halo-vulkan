package audiobox

import "math"

// The model hears 16 kHz mono. Upstream (infer.AesPredictor) resamples each
// channel with torchaudio.functional.resample at its defaults -- a Hann-
// windowed sinc, lowpass_filter_width 6, rolloff 0.99 -- and then averages
// the channels. From 48 kHz that is one phase of 41 taps at stride 3.

// resampleKernel is _get_sinc_resample_kernel(orig, new=1): the taps for
// output sample n over input samples orig·n − width .. orig·n + width +
// orig − 1. torch builds it in the waveform's fp32; this is the float64
// expression rounded once, which differs from that by an ulp at most.
func resampleKernel(orig int) ([]float32, int) {
	const filterWidth, rolloff = 6.0, 0.99
	base := rolloff // min(orig, 1) · rolloff
	width := int(math.Ceil(filterWidth * float64(orig) / base))
	k := make([]float32, 2*width+orig)
	for i := range k {
		t := float64(i-width) / float64(orig) * base
		t = math.Max(-filterWidth, math.Min(filterWidth, t))
		w := math.Cos(t * math.Pi / filterWidth / 2)
		w *= w
		t *= math.Pi
		s := 1.0
		if t != 0 {
			s = math.Sin(t) / t
		}
		k[i] = float32(s * w * base / float64(orig))
	}
	return k, width
}

// Mono16k is audio_resample_mono for interleaved stereo at 48 kHz: each
// channel resampled to 16 kHz (zero-padded at both ends, ceil(n/3)
// samples), then the two averaged.
func Mono16k(stereo []float32) []float32 {
	const orig = 3
	k, width := resampleKernel(orig)
	n := len(stereo) / 2
	out := make([]float32, (n+orig-1)/orig)
	parallel(len(out), func(lo, hi int) {
		for o := lo; o < hi; o++ {
			var l, r float32
			for j, w := range k {
				i := orig*o + j - width
				if i < 0 || i >= n {
					continue
				}
				l += w * stereo[2*i]
				r += w * stereo[2*i+1]
			}
			out[o] = (l + r) / 2
		}
	})
	return out
}
