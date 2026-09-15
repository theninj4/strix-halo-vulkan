package kokoro

import (
	"fmt"
	"math"
)

// ALBERT is the phoneme encoder — StyleTTS2 calls it PL-BERT, a BERT
// pre-trained on phonemes rather than words.
//
// Two things make it cheap and both are ALBERT's, not this model's. The
// embedding is 128 wide and projected once to 768, so the 178-entry table is
// a fifth of what a BERT's would be; and all twelve layers are *one* weight
// group applied twelve times, so the whole encoder is 3.5 M parameters read
// twelve times rather than 42 M read once. On a part whose last-level cache
// is 32 MiB that difference is the whole cost model: 14 MB of fp32 weights
// stay resident across the entire stack.
//
// It is post-LayerNorm — the normalisation is after each residual add, not
// before it — and its eps is 1e-12, six orders of magnitude tighter than
// everything StyleTTS2 built around it.
type ALBERT struct {
	Config PLBERTConfig

	Word      *Embedding
	Position  *Embedding
	TokenType *Embedding
	EmbedNorm *LayerNorm
	MapIn     *Linear // 128 -> 768

	Layer *ALBERTLayer // one group, applied NumHiddenLayers times
}

// ALBERTLayer is the shared weight group: self-attention with a
// post-normalised residual, then a feed-forward with another.
type ALBERTLayer struct {
	Heads, HeadDim int

	Q, K, V, Dense *Linear
	AttnNorm       *LayerNorm

	FFN, FFNOut *Linear
	OutNorm     *LayerNorm
}

// Embed runs the embedding stack: word + token type + position, then the
// 1e-12 LayerNorm, then the projection up to the hidden width.
//
// Positions are 0..T-1 with no offset — there is no padding index in play,
// because the model runs one utterance at a time and the boundary tokens are
// real id-0 entries rather than padding.
func (a *ALBERT) Embed(ids []int) (embed, hidden *Mat, err error) {
	embed, err = a.Word.Rows(ids)
	if err != nil {
		return nil, nil, err
	}
	if len(ids) > a.Position.Vocab {
		return nil, nil, fmt.Errorf("kokoro: %d tokens against %d position embeddings",
			len(ids), a.Position.Vocab)
	}
	for t := 0; t < embed.Rows; t++ {
		row := embed.Row(t)
		pos := a.Position.Weight[t*a.Position.Dim : (t+1)*a.Position.Dim]
		typ := a.TokenType.Weight[:a.TokenType.Dim] // token type 0 for every token
		for i := range row {
			row[i] += typ[i] + pos[i]
		}
	}
	if err := a.EmbedNorm.ApplyInPlace(embed); err != nil {
		return nil, nil, err
	}
	hidden, err = a.MapIn.Apply(embed)
	if err != nil {
		return nil, nil, err
	}
	return embed, hidden, nil
}

// Apply runs the whole encoder and returns every layer's output, newest last.
// The caller wants the last one; the rest are what a divergence is bisected
// with.
func (a *ALBERT) Apply(ids []int) (hiddens []*Mat, err error) {
	_, h, err := a.Embed(ids)
	if err != nil {
		return nil, err
	}
	for i := 0; i < a.Config.NumHiddenLayers; i++ {
		h, err = a.Layer.Apply(h)
		if err != nil {
			return nil, fmt.Errorf("kokoro: albert layer %d: %w", i, err)
		}
		hiddens = append(hiddens, h)
	}
	return hiddens, nil
}

// Apply runs one layer. Attention is full and unmasked: the model runs a
// single utterance with no padding, so there is nothing to mask and the
// [T, T] score matrix is dense.
func (l *ALBERTLayer) Apply(x *Mat) (*Mat, error) {
	attn, err := l.Attention(x)
	if err != nil {
		return nil, err
	}
	ffn, err := l.FFN.Apply(attn)
	if err != nil {
		return nil, err
	}
	for i, v := range ffn.Data {
		ffn.Data[i] = GELUNew(v)
	}
	out, err := l.FFNOut.Apply(ffn)
	if err != nil {
		return nil, err
	}
	out.AddInPlace(attn)
	return out, l.OutNorm.ApplyInPlace(out)
}

// Attention runs the self-attention half, including its post-normalised
// residual.
func (l *ALBERTLayer) Attention(x *Mat) (*Mat, error) {
	q, err := l.Q.Apply(x)
	if err != nil {
		return nil, err
	}
	k, err := l.K.Apply(x)
	if err != nil {
		return nil, err
	}
	v, err := l.V.Apply(x)
	if err != nil {
		return nil, err
	}
	ctx, err := l.Scores(q, k, v)
	if err != nil {
		return nil, err
	}
	out, err := l.Dense.Apply(ctx)
	if err != nil {
		return nil, err
	}
	out.AddInPlace(x)
	return out, l.AttnNorm.ApplyInPlace(out)
}

// Scores is the softmax attention itself, over q/k/v held [T, Heads*HeadDim].
func (l *ALBERTLayer) Scores(q, k, v *Mat) (*Mat, error) {
	t, width := q.Rows, l.Heads*l.HeadDim
	if q.Cols != width || k.Cols != width || v.Cols != width {
		return nil, fmt.Errorf("kokoro: attention over %d channels, want %d", q.Cols, width)
	}
	scale := float32(1 / math.Sqrt(float64(l.HeadDim)))
	out := NewMat(t, width)
	parallelFor(l.Heads, func(h int) {
		off := h * l.HeadDim
		scores := make([]float32, t)
		for i := 0; i < t; i++ {
			qr := q.Row(i)[off : off+l.HeadDim]
			max := float32(math.Inf(-1))
			for j := 0; j < t; j++ {
				kr := k.Row(j)[off : off+l.HeadDim]
				var dot float32
				for d, qv := range qr {
					dot += qv * kr[d]
				}
				scores[j] = dot * scale
				if scores[j] > max {
					max = scores[j]
				}
			}
			var sum float32
			for j, s := range scores {
				e := float32(math.Exp(float64(s - max)))
				scores[j] = e
				sum += e
			}
			dst := out.Row(i)[off : off+l.HeadDim]
			for j, w := range scores {
				w /= sum
				vr := v.Row(j)[off : off+l.HeadDim]
				for d := range dst {
					dst[d] += w * vr[d]
				}
			}
		}
	})
	return out, nil
}
