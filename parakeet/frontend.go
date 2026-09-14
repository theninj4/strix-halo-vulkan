package parakeet

import (
	"fmt"
	"math"

	"strix-halo-vulkan/audio"
)

// Front-end constants that live in transformers' feature extractor rather
// than in any config file: the floor added before the logarithm, and the
// epsilon in the per-utterance normalisation.
const (
	logZeroGuard = 1.0 / (1 << 24) // 2^-24
	normEps      = 1e-5
)

// FrontEnd turns a waveform into the normalised log-mel features the encoder
// takes: preemphasis, a centred STFT, a Slaney mel filterbank, a logarithm,
// and a per-utterance mean/variance normalisation.
//
// Every step is `ParakeetFeatureExtractor`'s, including the two places it
// differs from the obvious choice — the Hann window is symmetric rather than
// periodic, and the normalisation divides the variance by N-1 rather than N.
type FrontEnd struct {
	cfg     FeatureConfig
	stft    *audio.STFT
	filters []float32 // NMels x bins, row major
}

// NewFrontEnd builds the transform for a feature configuration.
func NewFrontEnd(cfg FeatureConfig) (*FrontEnd, error) {
	if cfg.NFFT <= 0 || cfg.HopLength <= 0 || cfg.WinLength <= 0 || cfg.NMels <= 0 {
		return nil, fmt.Errorf("parakeet: incomplete feature config %+v", cfg)
	}
	st, err := audio.NewSTFT(cfg.NFFT, cfg.HopLength, audio.HannWindow(cfg.WinLength, false), true)
	if err != nil {
		return nil, err
	}
	return &FrontEnd{
		cfg:  cfg,
		stft: st,
		// fmin 0, fmax Nyquist, Slaney scale and Slaney normalisation: the
		// arguments transformers passes librosa.filters.mel.
		filters: audio.MelFilters(cfg.SamplingRate, cfg.NFFT, cfg.NMels,
			0, float64(cfg.SamplingRate)/2, audio.Slaney, true),
	}, nil
}

// Features is a log-mel spectrogram: Frames rows of Mels values, row major.
//
// Valid is how many leading frames hold audio rather than the tail of the
// centred padding. It is one less than Frames — the extractor's frame count
// and its length formula disagree by exactly one — and the difference is not
// cosmetic: the trailing frame is zeroed, it is excluded from the
// normalisation statistics, and it is still fed to the encoder, where the
// subsampling happens to absorb it.
type Features struct {
	Data   []float32
	Frames int
	Valid  int
	Mels   int
}

// At returns frame t's mel vector, aliasing Data.
func (f *Features) At(t int) []float32 { return f.Data[t*f.Mels : (t+1)*f.Mels] }

// Features computes the encoder input for a clip.
func (f *FrontEnd) Features(c *audio.Clip) (*Features, error) {
	if c.Rate != f.cfg.SamplingRate {
		return nil, fmt.Errorf("parakeet: clip is %d Hz, the model wants %d", c.Rate, f.cfg.SamplingRate)
	}
	if len(c.Samples) < f.cfg.NFFT {
		return nil, fmt.Errorf("parakeet: clip is %d samples, shorter than one %d-point frame",
			len(c.Samples), f.cfg.NFFT)
	}

	mels := f.cfg.NMels
	frames := f.stft.Frames(len(c.Samples))
	// The extractor's own length formula, which is one short of the frame
	// count that torch.stft produces from the same padding.
	valid := (len(c.Samples) + 2*(f.cfg.NFFT/2) - f.cfg.NFFT) / f.cfg.HopLength
	if valid > frames {
		valid = frames
	}
	out := &Features{Data: f.logMel(c.Samples), Frames: frames, Valid: valid, Mels: mels}

	// Per-utterance normalisation over the valid frames, per mel bin. The
	// variance divides by N-1, which is the sample rather than population
	// convention and is what the checkpoint was trained with.
	if valid < 2 {
		return nil, fmt.Errorf("parakeet: %d valid frames is too few to normalise", valid)
	}
	mean := make([]float64, mels)
	for t := 0; t < valid; t++ {
		row := out.At(t)
		for m, v := range row {
			mean[m] += float64(v)
		}
	}
	variance := make([]float64, mels)
	for m := range mean {
		mean[m] /= float64(valid)
	}
	for t := 0; t < valid; t++ {
		row := out.At(t)
		for m, v := range row {
			d := float64(v) - mean[m]
			variance[m] += d * d
		}
	}
	scale := make([]float32, mels)
	offset := make([]float32, mels)
	for m := range variance {
		std := math.Sqrt(variance[m] / float64(valid-1))
		scale[m] = float32(1 / (std + normEps))
		offset[m] = float32(mean[m])
	}
	for t := 0; t < frames; t++ {
		row := out.At(t)
		if t >= valid {
			// Padding frames are zeroed rather than normalised, which is
			// what the mask multiply in the extractor amounts to.
			for m := range row {
				row[m] = 0
			}
			continue
		}
		for m := range row {
			row[m] = (row[m] - offset[m]) * scale[m]
		}
	}
	return out, nil
}

// logMel is the front end up to but not including the normalisation: the
// preemphasis, the STFT, the filterbank and the logarithm. It is separate so
// that a test can compare it against the reference's `log_mel` — the stage
// where a window or a filterbank mistake is still visible, before the
// normalisation rescales everything into the same range.
func (f *FrontEnd) logMel(samples []float32) []float32 {
	// Preemphasis: a first-difference high-pass, with the first sample kept
	// as is because there is nothing before it.
	pre := make([]float32, len(samples))
	pre[0] = samples[0]
	a := float32(f.cfg.Preemphasis)
	for i := 1; i < len(pre); i++ {
		pre[i] = samples[i] - a*samples[i-1]
	}

	power := f.stft.Power(pre)
	bins := f.stft.Bins()
	frames := f.stft.Frames(len(pre))
	mels := f.cfg.NMels

	out := make([]float32, frames*mels)
	for t := 0; t < frames; t++ {
		row := power[t*bins : (t+1)*bins]
		dst := out[t*mels : (t+1)*mels]
		for m := 0; m < mels; m++ {
			filt := f.filters[m*bins : (m+1)*bins]
			var acc float64
			for b, w := range filt {
				if w != 0 {
					acc += float64(w) * row[b]
				}
			}
			dst[m] = float32(math.Log(acc + logZeroGuard))
		}
	}
	return out
}
