package kokoro

import (
	"fmt"
	"math"
)

// DurationEncoder is the prosody predictor's text side: three bidirectional
// LSTMs with a style-conditioned layer norm between them, and the style
// vector re-concatenated after every one of them so each LSTM sees
// 512 + 128 = 640 inputs.
//
// Re-concatenating rather than conditioning once is what makes the style
// reach the duration head at all: the AdaLayerNorm affine is per channel and
// per utterance, so on its own it could only rescale what the LSTM already
// decided. The 128 raw channels riding alongside are what a duration is
// actually predicted from.
type DurationEncoder struct {
	Channels, StyleDim int
	LSTMs              []*LSTM
	Norms              []*AdaLayerNorm
}

// Apply runs the encoder over a [T, Channels] activation and returns
// [T, Channels+StyleDim] — the style is still attached on the way out, which
// is what the duration head and the length regulator both consume.
func (d *DurationEncoder) Apply(x *Mat, style []float32) (*Mat, error) {
	s := Broadcast(style, x.Rows)
	h, err := Concat(x, s)
	if err != nil {
		return nil, err
	}
	for i := range d.LSTMs {
		if h, err = d.LSTMs[i].Apply(h); err != nil {
			return nil, fmt.Errorf("kokoro: duration lstm %d: %w", i, err)
		}
		if err = d.Norms[i].Apply(h, style); err != nil {
			return nil, fmt.Errorf("kokoro: duration norm %d: %w", i, err)
		}
		if h, err = Concat(h, s); err != nil {
			return nil, err
		}
	}
	return h, nil
}

// Predictor is the whole prosody predictor: durations out of the text side,
// then F0 and energy out of the *expanded* side.
//
// The split is the interesting part of the architecture. Everything before
// the length regulator runs at one frame per phoneme; everything after it
// runs at one frame per 25 ms of output. `Shared` is the recurrence that
// crosses the boundary, and F0 and N are two stacks of AdaIN blocks that
// branch off it.
type Predictor struct {
	TextEncoder  *DurationEncoder
	LSTM         *LSTM   // 640 -> 512
	DurationHead *Linear // 512 -> MaxDur

	Shared *LSTM // 640 -> 512, after the expansion
	F0     []*AdainResBlk1d
	N      []*AdainResBlk1d
	F0Proj *Conv1D // 256 -> 1, kernel 1
	NProj  *Conv1D
}

// Durations predicts one integer frame count per token.
//
// The head emits MaxDur=50 logits per token and the duration is the *sum* of
// their sigmoids — a soft count of how many of 50 ticks are on, not an
// argmax and not a regression. It is then rounded and clamped to at least
// one frame, so no token can vanish.
//
// Speed divides the duration before rounding, which is why it is a parameter
// here rather than a resampling of the output.
func (p *Predictor) Durations(d *Mat, speed float32) ([]int, []float32, error) {
	x, err := p.LSTM.Apply(d)
	if err != nil {
		return nil, nil, err
	}
	logits, err := p.DurationHead.Apply(x)
	if err != nil {
		return nil, nil, err
	}
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
	return out, raw, nil
}

// Expand is the length regulator: every token's row repeated for as many
// frames as its duration.
//
// torch writes this as a [T, L] one-hot matrix and a matmul. It is a gather:
// frame f reads token Index(durations)[f], and the two [512, T] x [T, L]
// GEMMs the reference spends on it do not need to exist. This is also the
// first data-dependent output length in the engine — L is a function of the
// *content* of the input, not its shape.
func Expand(x *Mat, durations []int) (*Mat, error) {
	if len(durations) != x.Rows {
		return nil, fmt.Errorf("kokoro: %d durations for %d frames", len(durations), x.Rows)
	}
	total := 0
	for _, d := range durations {
		total += d
	}
	out := NewMat(total, x.Cols)
	f := 0
	for t, d := range durations {
		for i := 0; i < d; i++ {
			copy(out.Row(f), x.Row(t))
			f++
		}
	}
	return out, nil
}

// Prosody runs the expanded side: the shared recurrence, then the F0 and
// energy stacks, each of which doubles the frame count in its middle block.
//
// Both curves come back at 2L frames — twice the alignment rate, 12.5 ms a
// frame — because the vocoder's F0 conditioning is decimated by a stride-2
// convolution on the way in. The round trip is not an accident: the doubling
// happens inside an AdaIN block where it can be learned, and the halving
// happens in a plain convolution where it cannot.
func (p *Predictor) Prosody(en *Mat, style []float32) (f0, energy []float32, err error) {
	shared, err := p.Shared.Apply(en)
	if err != nil {
		return nil, nil, err
	}
	run := func(blocks []*AdainResBlk1d, proj *Conv1D) ([]float32, error) {
		h, err := shared, error(nil)
		for i, b := range blocks {
			if h, err = b.Apply(h, style); err != nil {
				return nil, fmt.Errorf("kokoro: prosody block %d: %w", i, err)
			}
		}
		out, err := proj.Apply(h)
		if err != nil {
			return nil, err
		}
		return out.Data, nil
	}
	if f0, err = run(p.F0, p.F0Proj); err != nil {
		return nil, nil, err
	}
	if energy, err = run(p.N, p.NProj); err != nil {
		return nil, nil, err
	}
	return f0, energy, nil
}
