package songeval

import (
	"math"

	"strix-halo-vulkan/audio"
)

// Mono24k is the song at MuQ's rate: each channel of interleaved 48 kHz
// stereo through torchaudio.functional.resample's Hann-windowed sinc
// (lowpass_filter_width 6, rolloff 0.99; 2:1, so one phase of 27 taps),
// then the two averaged. SongEval's eval.py loads with librosa's soxr
// instead; no resampler is that one, and A14d prices the choice (a
// resampler 45 dB from soxr still moves a score 0.02).
func Mono24k(stereo []float32) []float32 {
	const orig, filterWidth, rolloff = 2, 6.0, 0.99
	base := rolloff
	width := int(math.Ceil(filterWidth * orig / base))
	k := make([]float32, 2*width+orig)
	for i := range k {
		t := float64(i-width) / orig * base
		t = math.Max(-filterWidth, math.Min(filterWidth, t))
		w := math.Cos(t * math.Pi / filterWidth / 2)
		w *= w
		t *= math.Pi
		s := 1.0
		if t != 0 {
			s = math.Sin(t) / t
		}
		k[i] = float32(s * w * base / orig)
	}
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

// Mel is MuQ's input: torchaudio's MelSpectrogram (n_fft 2048, hop 240, a
// periodic Hann window, centred with reflection, power 2, 128 HTK bands
// over 0-12 kHz, no norm) through AmplitudeToDB (10·log10, floored at
// 1e-10), its last frame dropped, normalised by the checkpoint's mean and
// std: [128][frames], band-major, and the frame count (len/240).
func (w *weights) Mel(wav []float32) ([]float32, int) {
	st, err := audio.NewSTFT(nFFT, hop, audio.HannWindow(nFFT, true), true)
	if err != nil {
		panic(err)
	}
	st.Pad = audio.PadReflect
	frames := st.Frames(len(wav)) - 1
	if frames <= 0 {
		return nil, 0
	}
	bins := st.Bins()
	fb := audio.MelFilters(SampleRate, nFFT, nMels, 0, SampleRate/2, audio.HTK, false)
	pow := st.Power(wav)
	out := make([]float32, nMels*frames)
	parallel(frames, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := pow[t*bins : (t+1)*bins]
			for m := 0; m < nMels; m++ {
				var s float64
				for b, f := range fb[m*bins : (m+1)*bins] {
					if f != 0 {
						s += float64(f) * row[b]
					}
				}
				db := 10 * math.Log10(math.Max(s, 1e-10))
				out[m*frames+t] = float32((db - w.melMean) / w.melStd)
			}
		}
	})
	return out, frames
}
