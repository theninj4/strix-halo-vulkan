package parakeet

import (
	"fmt"

	"strix-halo-vulkan/audio"
)

// Step is one emission of the TDT loop: the encoder frame it was made at, the
// token emitted there (possibly the blank), and the number of frames the
// cursor then jumped.
type Step struct {
	Frame    int
	Token    int
	Duration int
}

// Transcript is a decoded clip.
type Transcript struct {
	Text   string
	Tokens []int // emitted tokens, blanks removed
	Steps  []Step
	Frames int // valid encoder frames, i.e. the length of the audio in 80 ms units
}

// Transcribe runs the whole model over a clip.
func (m *Model) Transcribe(clip *audio.Clip) (*Transcript, error) {
	feats, err := m.FrontEnd.Features(clip)
	if err != nil {
		return nil, err
	}
	mel := &Mat{Rows: feats.Frames, Cols: feats.Mels, Data: feats.Data}
	hidden, valid, err := m.Encoder.Apply(mel, feats.Valid)
	if err != nil {
		return nil, err
	}
	enc, err := m.Projector.Apply(hidden)
	if err != nil {
		return nil, err
	}
	return m.Decode(enc, valid)
}

// Decode runs the TDT greedy loop over projected encoder frames.
//
// It is a port of transformers' ParakeetTDTGenerationMixin rather than of
// `generate`, and the three rules that are easy to get wrong are all here:
//
//   - the prediction network advances only on a *non-blank* emission. A blank
//     leaves its state and its output untouched, which is why the loop can
//     emit several blanks in a row for the price of one LSTM step.
//   - the encoder cursor advances by the predicted duration, not by one
//     frame, and a blank predicting duration 0 is forced to 1 so the loop
//     cannot stall. A non-blank predicting 0 is allowed: that is how several
//     tokens are emitted at the same frame.
//   - decoding stops when the cursor passes the last valid frame, with
//     `max_symbols_per_step` frames' worth of emissions as the backstop.
func (m *Model) Decode(enc *Mat, valid int) (*Transcript, error) {
	if enc.Cols != m.Config.DecoderHiddenSize {
		return nil, fmt.Errorf("parakeet: encoder frames are %d wide, the joint wants %d",
			enc.Cols, m.Config.DecoderHiddenSize)
	}
	if valid > enc.Rows {
		return nil, fmt.Errorf("parakeet: %d valid frames of %d", valid, enc.Rows)
	}
	blank := m.Config.BlankTokenID
	limit := m.Config.MaxSymbolsPerStep * valid

	out := &Transcript{Frames: valid}
	state := m.Prediction.NewState()
	logits := make([]float32, m.Joint.Head.Out)
	prev := blank
	var dec []float32

	for t := 0; t < valid && len(out.Steps) < limit; {
		if dec == nil || prev != blank {
			var err error
			if dec, err = m.Prediction.Step(prev, state); err != nil {
				return nil, err
			}
		}
		m.Joint.Logits(logits, enc.Row(t), dec)
		token, duration := m.Joint.Argmax(logits, blank)
		out.Steps = append(out.Steps, Step{Frame: t, Token: token, Duration: duration})
		if token != blank {
			out.Tokens = append(out.Tokens, token)
		}
		prev = token
		t += duration
	}

	text, err := m.Tokenizer.Decode(out.Tokens)
	if err != nil {
		return nil, err
	}
	out.Text = text
	return out, nil
}
