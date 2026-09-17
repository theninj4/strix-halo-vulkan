// Package embed implements Qwen3-Embedding-0.6B: text in, a 1024-dimensional
// unit vector out.
//
// The transformer is not written here. It is the same `Qwen3Model` as
// Z-Image's text encoder, which `zimage/qwen` already implements on the CPU
// and on the device, 36 layers of 2560 against 28 of 1024 -- so this package
// is the four things an *embedding* model has that a text encoder does not,
// and nothing else:
//
//   - every layer runs, where the encoder stops one short (`hidden_states[-2]`);
//   - the final `norm` runs after them;
//   - the vector is the **last token's** row (`1_Pooling` asks for
//     pooling_mode_lasttoken), and the last token is the `<|endoftext|>` the
//     tokenizer's post-processor appends -- see tokenizer.go, it is the
//     likeliest silent bug in the whole vertical;
//   - the row is L2-normalised, so a dot product is a cosine.
//
// Instructions (`Instruct: …\nQuery:…`) are part of the input *text* rather
// than of the model, so they are a function here and a request field at the
// API layer, not something Embed does behind the caller's back.
package embed

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// Dim is the model's embedding width, and the largest vector it produces.
// The card supports 32..1024 by truncating and renormalising (MRL); Truncate
// is that operation.
const Dim = 1024

// Model is the CPU reference: the oracle the Vulkan path is debugged
// against, not the fast path. Weights are float32 -- 2.38 GB, of which the
// embedding table is 0.62 -- and a text is 170 ms against the device's 11.5.
type Model struct {
	Cfg  *qwen.Config
	Enc  *qwen.Model   // embedding table and all NumLayers decoder layers
	Norm *qwen.RMSNorm // the final norm, which the text encoder never runs
	Tok  *Tokenizer
}

// Load reads a checkpoint directory -- models/Qwen3-Embedding-0.6B -- with
// its tokenizer.
func Load(dir string) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	enc, err := qwen.LoadWith(dir, cfg, cfg.NumLayers)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	norm, err := LoadFinalNorm(set, cfg)
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	return &Model{Cfg: cfg, Enc: enc, Norm: norm, Tok: tok}, nil
}

// Hidden runs the whole stack over a token sequence and returns the normed
// hidden states, [tokens, HiddenSize] -- transformers'
// `last_hidden_state`. tr may be nil.
func (m *Model) Hidden(ids []int32, tr qwen.Trace) (*qwen.Mat, error) {
	x, err := m.Enc.Forward(ids, tr)
	if err != nil {
		return nil, err
	}
	out, err := m.Norm.Apply(x)
	if err != nil {
		return nil, fmt.Errorf("embed: final norm: %w", err)
	}
	if tr != nil {
		tr["final"] = out.Clone()
	}
	return out, nil
}

// EmbedIDs pools and normalises a run of the model. The ids must already
// carry the appended end-of-text token; Tokenizer.Encode puts it there.
func (m *Model) EmbedIDs(ids []int32) ([]float32, error) {
	h, err := m.Hidden(ids, nil)
	if err != nil {
		return nil, err
	}
	return Normalize(Pool(h)), nil
}

// Embed is the whole thing: text to a unit vector.
func (m *Model) Embed(text string) ([]float32, error) {
	ids, err := m.Tok.Encode(text)
	if err != nil {
		return nil, err
	}
	return m.EmbedIDs(ids)
}

// Pool takes the last row, which is `pooling_mode_lasttoken` over a sequence
// that is not left-padded. Nothing in this repository pads: a Go caller runs
// one sequence at a time, and the reference dump measures that a right-padded
// batch gives a real token the same hidden state to 2.5e-05.
func Pool(h *qwen.Mat) []float32 {
	return append([]float32(nil), h.Row(h.Rows-1)...)
}

// Normalize scales a vector to unit length in place and returns it. A zero
// vector is left alone rather than turned into NaNs.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= scale
	}
	return v
}

// Truncate is Matryoshka (MRL) output: keep the first d components and
// renormalise. The card supports 32..1024 this way. d <= 0 or d >= len(v)
// returns v unchanged.
func Truncate(v []float32, d int) []float32 {
	if d <= 0 || d >= len(v) {
		return v
	}
	return Normalize(v[:d])
}

// Cosine is the similarity between two unit vectors, which is their dot
// product. It does not normalise: an embedding out of this package already
// is a unit vector, and quietly renormalising would hide a caller that had
// truncated one and not the other.
func Cosine(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Instruct is how the card asks a *query* to be written -- documents get no
// instruction at all. The string is exactly
// `config_sentence_transformers.json`'s "query" prompt with the query
// appended, and note there is no space after "Query:".
func Instruct(task, query string) string {
	return "Instruct: " + task + "\nQuery:" + query
}

// DefaultTask is the card's own instruction, and what an API request that
// names no task gets.
const DefaultTask = "Given a web search query, retrieve relevant passages that answer the query"
