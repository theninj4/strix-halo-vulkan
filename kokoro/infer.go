package kokoro

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Prosody is everything the vocoder needs and nothing it does not: the
// expanded phoneme features, the two curves that condition them, and the
// alignment that produced all three.
type Prosody struct {
	Tokens    []int // the input ids, boundary tokens included
	Durations []int // one frame count per token, summing to Frames
	Frames    int   // alignment frames, 25 ms each
	ASR       *Mat  // [Frames, 512], the text encoder expanded
	// Encoded is [Frames, 640], the duration encoder expanded. It is a
	// diagnostic: nothing downstream reads it, and on the device path it is
	// **nil**, because there the length regulator is a gather that happens
	// between two dispatches and the tensor never leaves the arena.
	Encoded *Mat
	F0      []float32 // [2*Frames]
	Energy  []float32 // [2*Frames]
	RawDur  []float32 // the durations before rounding, for diagnostics

	// Where the time went. The phoneme side is 85% of an utterance and 60% of
	// *it* is recurrences, so the split that matters is ALBERT against the six
	// bidirectional LSTMs (SPEECH.md T6).
	Times ProsodyTimes
}

// ProsodyTimes is the phoneme side, stage by stage.
type ProsodyTimes struct {
	BERT       time.Duration // 12 ALBERT layers over the tokens
	DurEncoder time.Duration // three LSTM/AdaLayerNorm pairs
	Durations  time.Duration // one LSTM and the duration head
	Prosody    time.Duration // the shared LSTM and the F0/N stacks
	Shared     time.Duration // of Prosody: the recurrence across the regulator
	Stacks     time.Duration // of Prosody: the six F0/N AdaIN blocks
	// Recurrence is the six bidirectional LSTMs, summed across whichever of
	// the stages above contain them: three in DurEncoder, one in Durations,
	// Shared, and one in TextEncoder. It cuts across the stage split rather
	// than partitioning it, because what T6c can move is the recurrences and
	// not the stages (SPEECH.md T6c).
	Recurrence  time.Duration
	TextEncoder time.Duration // an embedding, three convolutions and one LSTM
	Expand      time.Duration // the length regulator, twice
}

// Total is the whole phoneme side.
func (t ProsodyTimes) Total() time.Duration {
	return t.BERT + t.DurEncoder + t.Durations + t.Prosody + t.TextEncoder + t.Expand
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
	if m.PhonemesGPU != nil {
		return m.prosodyGPU(ids, style, speed)
	}
	var times ProsodyTimes
	t0 := time.Now()
	var last *Mat
	if m.BERTGPU != nil {
		// The embedding stack stays on the host: a table lookup, two adds and
		// a 128->768 projection over fifty rows is not worth a dispatch.
		_, hidden, err := m.BERT.Embed(ids)
		if err != nil {
			return nil, err
		}
		if last, err = m.BERTGPU.Apply(hidden); err != nil {
			return nil, err
		}
	} else {
		hiddens, err := m.BERT.Apply(ids)
		if err != nil {
			return nil, err
		}
		last = hiddens[len(hiddens)-1]
	}
	dEn, err := m.BERTEncoder.Apply(last)
	if err != nil {
		return nil, err
	}
	times.BERT = time.Since(t0)

	t0 = time.Now()
	encoded, err := m.Predictor.TextEncoder.Apply(dEn, style, &times)
	if err != nil {
		return nil, err
	}
	times.DurEncoder = time.Since(t0)

	t0 = time.Now()
	durations, raw, err := m.Predictor.Durations(encoded, speed, &times)
	if err != nil {
		return nil, err
	}
	times.Durations = time.Since(t0)

	t0 = time.Now()
	en, err := Expand(encoded, durations)
	if err != nil {
		return nil, err
	}
	times.Expand = time.Since(t0)

	t0 = time.Now()
	var f0, energy []float32
	if f0, energy, err = m.Predictor.Prosody(en, style, &times); err != nil {
		return nil, err
	}
	times.Prosody = time.Since(t0)

	t0 = time.Now()
	tEn, err := m.TextEncoder.Apply(ids, &times)
	if err != nil {
		return nil, err
	}
	times.TextEncoder = time.Since(t0)

	t0 = time.Now()
	asr, err := Expand(tEn, durations)
	if err != nil {
		return nil, err
	}
	times.Expand += time.Since(t0)

	return &Prosody{
		Tokens: ids, Durations: durations, Frames: en.Rows,
		ASR: asr, Encoded: en, F0: f0, Energy: energy, RawDur: raw,
		Times: times,
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

// prosodyGPU is Model.Prosody with the phoneme side on the device.
//
// It is a separate function rather than a set of branches because almost
// nothing survives the move: the length regulator, the concatenations, the
// three normalisations and the convolutions are all offsets in an arena now,
// and what is left on the host is the two decisions and the two table
// lookups that feed them.
//
// The host still does four things, and each of them is here for a reason:
//
//   - **the embeddings**, PL-BERT's and the text encoder's, because a gather
//     of fifty rows out of a table is not worth a dispatch and putting the
//     table on the device would cost 21 MB to save a copy;
//   - **`bert_encoder`**, one 768->512 projection over fifty rows, for the
//     same reason;
//   - **the duration head and its rounding**, which is the decision every
//     frame boundary downstream is placed by, and which fp16 cannot make;
//   - **expanding `t_en`**, because the vocoder still takes its input from
//     the host, and [T, 512] is cheaper to read back than [L, 512].
func (m *Model) prosodyGPU(ids []int, style []float32, speed float32) (*Prosody, error) {
	g := m.PhonemesGPU
	var times ProsodyTimes

	t0 := time.Now()
	_, hidden, err := m.BERT.Embed(ids)
	if err != nil {
		return nil, err
	}
	last, err := m.BERTGPU.Apply(hidden)
	if err != nil {
		return nil, err
	}
	dEn, err := m.BERTEncoder.Apply(last)
	if err != nil {
		return nil, err
	}
	times.BERT = time.Since(t0)

	t0 = time.Now()
	emb, err := m.TextEncoder.Embedding.Rows(ids)
	if err != nil {
		return nil, err
	}
	// One submit: the duration encoder, the recurrence before the head, and
	// the whole text encoder, which depends on nothing the first two produce.
	x, err := g.Encode(dEn, emb)
	if err != nil {
		return nil, err
	}
	times.DurEncoder = time.Since(t0)

	// The head stays on the host, in fp32, on the same code the CPU reference
	// runs: its logits have an rms of 27 and a duration is a *rounded* sum of
	// fifty sigmoids of them, so a thousandth of fp16 relative error is a
	// third of a frame. See GPUPhonemes.
	t0 = time.Now()
	logits, err := m.Predictor.DurationHead.Apply(x)
	if err != nil {
		return nil, err
	}
	durations, raw := durationsFrom(logits, speed)
	times.Durations = time.Since(t0)

	t0 = time.Now()
	f0, energy, err := g.Prosody(durations)
	if err != nil {
		return nil, err
	}
	times.Prosody = time.Since(t0)
	times.Stacks = times.Prosody

	t0 = time.Now()
	tEn := g.TextEncoded()
	times.TextEncoder = time.Since(t0)

	t0 = time.Now()
	asr, err := Expand(tEn, durations)
	if err != nil {
		return nil, err
	}
	times.Expand = time.Since(t0)

	frames := 0
	for _, d := range durations {
		frames += d
	}
	// Encoded is the duration encoder's expanded output, which on this path
	// never leaves the device: it is a diagnostic the CPU reference fills in
	// and nothing downstream reads.
	return &Prosody{
		Tokens: ids, Durations: durations, Frames: frames,
		ASR: asr, F0: f0, Energy: energy, RawDur: raw,
		Times: times,
	}, nil
}

// durationsFrom is the rounding, which is the same arithmetic wherever the
// logits came from: a duration is the *sum* of fifty sigmoids — a soft count
// of how many of fifty ticks are on — divided by the speed, rounded, and
// clamped to at least one frame so no token can vanish.
func durationsFrom(logits *Mat, speed float32) ([]int, []float32) {
	raw := make([]float32, logits.Rows)
	out := make([]int, logits.Rows)
	for t := 0; t < logits.Rows; t++ {
		var sum float32
		for _, v := range logits.Row(t) {
			sum += sigmoid(v)
		}
		raw[t] = sum / speed
		n := int(math.Round(float64(raw[t])))
		if n < 1 {
			n = 1
		}
		out[t] = n
	}
	return out, raw
}
