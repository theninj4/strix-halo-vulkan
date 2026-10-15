// Package textenc implements FLUX 3 Image Klein's text encoding (X1,
// research/flux3-vertical.md): Qwen3-VL-4B's language model over a
// user-only chat template, right-padded to a multiple of 80 tokens, with
// hidden_states 4, 8, ..., 32 stacked per token into 20480 columns.
//
// It is the Go counterpart of diffusers'
// Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds
// (reference/flux3/diffusers/modular/encoders.py) and of BFL's
// TextEncoder.encode (reference/flux3/bfl/text_encoder.py). The model is
// zimage/qwen's Qwen3 transformer through qimage/textenc's Qwen3-VL config
// adapter; only 32 of the 36 layers are loaded, because the last tap is 32.
//
// The window's pad rows are DiT tokens, so they cannot be dropped — but they
// do not need to be computed one by one. Under the causal mask a real row
// never sees a pad, and every pad row has the same token, the same position
// (1: diffusers pins it, because transformers 5.x would give 0) and the same
// visible keys (the real ones, not its own — the padding mask removes it).
// So every pad row is the same row, which X0 measured as bit-identical in
// the oracle, and Encode runs the real tokens plus one pad row and copies it.
package textenc

import (
	"fmt"
	"math"

	qtextenc "strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

const (
	// PadMultiple is the window's granularity; the window is part of the
	// recipe because the DiT attends to the padding.
	PadMultiple = 80
	// MaxTokens is diffusers' max_sequence_length (BFL's own code says 4096;
	// the checkpoint's config.json says 8192, and so does diffusers).
	MaxTokens = 8192
	// PadID is <|endoftext|>, the tokenizer's pad token.
	PadID int32 = 151643
	// PadPosition is the rotary position every pad row takes.
	PadPosition = 1
)

// Taps are the hidden_states indices stacked into a text embedding;
// hidden_states[0] is the embedding table's output, [k] layer k-1's.
var Taps = []int{4, 8, 12, 16, 20, 24, 28, 32}

// Template renders a prompt as apply_chat_template does for a single user
// turn with add_generation_prompt: no system message, nothing dropped.
func Template(prompt string) string {
	return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// Window is the padded length for n real tokens.
func Window(n int) int {
	w := (n + PadMultiple - 1) / PadMultiple * PadMultiple
	if w > MaxTokens {
		w = MaxTokens
	}
	return w
}

// Encoder holds the language model's first Taps[len-1] layers and the tokenizer.
type Encoder struct {
	Model *qwen.Model
	Tok   *tokenizer.Tokenizer
}

// Load reads the checkpoint's text_encoder directory.
func Load(dir string) (*Encoder, error) {
	return LoadLayers(dir, Taps[len(Taps)-1])
}

// LoadLayers loads only the first n layers — for a walk that stops early.
func LoadLayers(dir string, n int) (*Encoder, error) {
	cfg, err := qtextenc.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	m, err := qwen.LoadWith(dir, cfg, n)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		return nil, err
	}
	return &Encoder{Model: m, Tok: tok}, nil
}

// IDs tokenizes a prompt under the template, truncated to MaxTokens as the
// reference's tokenizer call truncates it.
func (e *Encoder) IDs(prompt string) ([]int32, error) {
	ids, err := e.Tok.Encode(Template(prompt))
	if err != nil {
		return nil, err
	}
	if len(ids) > MaxTokens {
		ids = ids[:MaxTokens]
	}
	return ids, nil
}

// Hidden runs the loaded layers over the real ids plus one pad row and
// returns hidden_states[0..layers] (each [len(ids)+1, hidden]); the last row
// of each is the pad row.
func (e *Encoder) Hidden(ids []int32) ([]*qwen.Mat, error) {
	n := len(ids)
	if n == 0 || n > MaxTokens {
		return nil, fmt.Errorf("textenc: %d tokens, want 1..%d", n, MaxTokens)
	}
	cfg := e.Model.Cfg
	x, err := e.Model.Embeddings(append(append([]int32(nil), ids...), PadID))
	if err != nil {
		return nil, err
	}
	rope := padRoPE(cfg.HeadDim, n, cfg.RopeTheta)
	hs := []*qwen.Mat{x}
	for i, l := range e.Model.Layers {
		if x, err = l.ForwardAttn(x, rope, PadAttention, nil, ""); err != nil {
			return nil, fmt.Errorf("textenc: layer %d: %w", i, err)
		}
		hs = append(hs, x)
	}
	return hs, nil
}

// Encode returns the DiT's text input for real ids: [Window(n), 8*hidden],
// the pad row copied into every row past n.
func (e *Encoder) Encode(ids []int32) (*qwen.Mat, error) {
	last := Taps[len(Taps)-1]
	if len(e.Model.Layers) < last {
		return nil, fmt.Errorf("textenc: %d layers loaded, the taps need %d", len(e.Model.Layers), last)
	}
	hs, err := e.Hidden(ids)
	if err != nil {
		return nil, err
	}
	n, h := len(ids), e.Model.Cfg.HiddenSize
	out := qwen.NewMat(Window(n), len(Taps)*h)
	for r := 0; r < out.Rows; r++ {
		src := r
		if src > n {
			src = n // every pad row is row n
		}
		dst := out.Row(r)
		for j, k := range Taps {
			copy(dst[j*h:(j+1)*h], hs[k].Row(src))
		}
	}
	return out, nil
}

// padRoPE is the plain NeoX table for positions 0..n-1 followed by one row
// at PadPosition.
func padRoPE(headDim, n int, theta float64) *qwen.RoPE {
	r := qwen.NewRoPE(headDim, n, theta)
	half := headDim / 2
	for i := 0; i < half; i++ {
		angle := float64(PadPosition) / math.Pow(theta, float64(2*i)/float64(headDim))
		r.Cos = append(r.Cos, float32(math.Cos(angle)))
		r.Sin = append(r.Sin, float32(math.Sin(angle)))
	}
	return r
}

// PadAttention is causal attention over the first T-1 rows, plus a last row
// (the pad query) that attends to those T-1 keys and not to its own.
func PadAttention(q, k, v *qwen.Mat, heads, kvHeads, headDim int) (*qwen.Mat, error) {
	n := q.Rows - 1
	if n < 1 || k.Rows != q.Rows || v.Rows != q.Rows {
		return nil, fmt.Errorf("textenc: pad attention over %d/%d/%d rows", q.Rows, k.Rows, v.Rows)
	}
	head := func(m *qwen.Mat) *qwen.Mat { return &qwen.Mat{Rows: n, Cols: m.Cols, Data: m.Data[:n*m.Cols]} }
	real, err := qwen.Attention(head(q), head(k), head(v), heads, kvHeads, headDim)
	if err != nil {
		return nil, err
	}
	out := qwen.NewMat(n+1, real.Cols)
	copy(out.Data, real.Data)

	rep := heads / kvHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	qp, dst := q.Row(n), out.Row(n)
	scores := make([]float32, n)
	for h := 0; h < heads; h++ {
		kv := h / rep
		qh := qp[h*headDim : (h+1)*headDim]
		max := float32(math.Inf(-1))
		for s := 0; s < n; s++ {
			kr := k.Row(s)[kv*headDim : (kv+1)*headDim]
			var dot float32
			for i, qi := range qh {
				dot += qi * kr[i]
			}
			scores[s] = dot * scale
			if scores[s] > max {
				max = scores[s]
			}
		}
		var sum float32
		for s := range scores {
			scores[s] = float32(math.Exp(float64(scores[s] - max)))
			sum += scores[s]
		}
		dh := dst[h*headDim : (h+1)*headDim]
		for s := 0; s < n; s++ {
			p := scores[s] / sum
			vr := v.Row(s)[kv*headDim : (kv+1)*headDim]
			for i, vv := range vr {
				dh[i] += p * vv
			}
		}
	}
	return out, nil
}
