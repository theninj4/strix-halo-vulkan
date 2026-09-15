package kokoro

import (
	"fmt"
	"time"
)

// TextEncoder is the second, shallower path from the phonemes: an embedding,
// three 5-tap convolutions with a layer norm and a leaky ReLU, and one
// bidirectional LSTM.
//
// The model reads its input twice — once through ALBERT to decide *how long*
// each phoneme lasts and what pitch it carries, and once through here to
// decide what it sounds like. This path is the one whose output is expanded
// and handed to the vocoder, and it never sees the style vector at all:
// timbre arrives later, through the AdaIN blocks.
type TextEncoder struct {
	Channels int

	Embedding *Embedding
	Convs     []*Conv1D
	Norms     []*LayerNorm
	LSTM      *LSTM
}

// Apply runs the encoder over a token sequence and returns [T, Channels].
func (t *TextEncoder) Apply(ids []int, times *ProsodyTimes) (*Mat, error) {
	x, err := t.Embedding.Rows(ids)
	if err != nil {
		return nil, err
	}
	for i := range t.Convs {
		if x, err = t.Convs[i].Apply(x); err != nil {
			return nil, fmt.Errorf("kokoro: text encoder conv %d: %w", i, err)
		}
		if err = t.Norms[i].ApplyInPlace(x); err != nil {
			return nil, fmt.Errorf("kokoro: text encoder norm %d: %w", i, err)
		}
		leakyReLUInPlace(x, 0.2)
	}
	t0 := time.Now()
	out, err := t.LSTM.Apply(x)
	if times != nil {
		times.Recurrence += time.Since(t0)
	}
	return out, err
}
