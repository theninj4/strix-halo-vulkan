package kokoro

import (
	"fmt"
	"math/rand"
)

// Prosody is everything the vocoder needs and nothing it does not: the
// expanded phoneme features, the two curves that condition them, and the
// alignment that produced all three.
type Prosody struct {
	Tokens    []int     // the input ids, boundary tokens included
	Durations []int     // one frame count per token, summing to Frames
	Frames    int       // alignment frames, 25 ms each
	ASR       *Mat      // [Frames, 512], the text encoder expanded
	Encoded   *Mat      // [Frames, 640], the duration encoder expanded
	F0        []float32 // [2*Frames]
	Energy    []float32 // [2*Frames]
	RawDur    []float32 // the durations before rounding, for diagnostics
}

// Samples is how long the waveform will be, which is fixed the moment the
// durations are rounded — 600 samples an alignment frame at 24 kHz.
func (p *Prosody) Samples(cfg *Config) int {
	return p.Frames * cfg.SamplesPerFrame()
}

// Seconds is Samples in wall-clock terms.
func (p *Prosody) Seconds(cfg *Config) float64 {
	return float64(p.Samples(cfg)) / float64(cfg.SamplingRate)
}

// Prosody runs everything up to the vocoder: both encoders, the duration
// head, the length regulator and the F0/energy stacks.
//
// The order matters and is not the order the modules are declared in. ALBERT
// and the duration encoder decide *how long* the utterance is before the text
// encoder's output can be expanded, so the cheap path (five convolutions and
// an LSTM over 50 frames) waits on the expensive one (twelve transformer
// layers). Nothing downstream of here has a shape that was known when the
// call started.
func (m *Model) Prosody(ids []int, style []float32, speed float32) (*Prosody, error) {
	if speed <= 0 {
		return nil, fmt.Errorf("kokoro: speed %g", speed)
	}
	if len(style) != m.Config.StyleDim {
		return nil, fmt.Errorf("kokoro: predictor style is %d wide, want %d",
			len(style), m.Config.StyleDim)
	}
	hiddens, err := m.BERT.Apply(ids)
	if err != nil {
		return nil, err
	}
	dEn, err := m.BERTEncoder.Apply(hiddens[len(hiddens)-1])
	if err != nil {
		return nil, err
	}
	encoded, err := m.Predictor.TextEncoder.Apply(dEn, style)
	if err != nil {
		return nil, err
	}
	durations, raw, err := m.Predictor.Durations(encoded, speed)
	if err != nil {
		return nil, err
	}
	en, err := Expand(encoded, durations)
	if err != nil {
		return nil, err
	}
	f0, energy, err := m.Predictor.Prosody(en, style)
	if err != nil {
		return nil, err
	}
	tEn, err := m.TextEncoder.Apply(ids)
	if err != nil {
		return nil, err
	}
	asr, err := Expand(tEn, durations)
	if err != nil {
		return nil, err
	}
	return &Prosody{
		Tokens: ids, Durations: durations, Frames: en.Rows,
		ASR: asr, Encoded: en, F0: f0, Energy: energy, RawDur: raw,
	}, nil
}

// Synthesize runs the whole model: phonemes in, samples out at 24 kHz.
//
// The two halves of the style vector go to different places and are not
// interchangeable — the predictor's 128 condition every AdaLayerNorm before
// the length regulator, the decoder's condition every AdaIN after it — so
// Style returns them in the order they are used here.
func (m *Model) Synthesize(ids []int, decStyle, predStyle []float32, speed float32) ([]float32, *Prosody, error) {
	p, err := m.Prosody(ids, predStyle, speed)
	if err != nil {
		return nil, nil, err
	}
	audio, _, err := m.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle)
	if err != nil {
		return nil, p, err
	}
	if want := p.Samples(m.Config); len(audio) != want {
		return nil, p, fmt.Errorf("kokoro: %d samples, the durations say %d", len(audio), want)
	}
	return audio, p, nil
}

// SetExcitationNoise switches the vocoder's excitation noise on, seeded.
//
// Off is the default because it is what the reference dump was taken with and
// the only configuration in which two runs agree. On is what an utterance
// meant to be listened to wants: see HarmonicSource.
func (m *Model) SetExcitationNoise(seed int64) {
	m.Vocoder.Generator.Source.Noise = rand.New(rand.NewSource(seed))
}

// Speak is Synthesize from a phoneme string and a voice name, which is the
// whole API the HTTP layer needs.
func (m *Model) Speak(phonemes, voice string, speed float32) ([]float32, *Prosody, error) {
	ids, dropped := m.Config.Phonemes(phonemes)
	if len(ids) <= 2 {
		return nil, nil, fmt.Errorf("kokoro: no phonemes in %q", phonemes)
	}
	// The style row is indexed by the phoneme count the *caller* supplied,
	// minus whatever fell outside the vocabulary — which is how KPipeline
	// indexes it, and is not the token count.
	dec, pred, err := m.Style(voice, len([]rune(phonemes))-dropped)
	if err != nil {
		return nil, nil, err
	}
	return m.Synthesize(ids, dec, pred, speed)
}
