// Package qwen implements Z-Image's text encoder: Qwen3-4B, run as an
// encoder rather than as a language model.
//
// This is the CPU reference, written before the Vulkan one for the reason
// every stage in PIPELINE.md gives -- it separates understanding the
// architecture from getting a shader right, and once it matches transformers
// it is the oracle the GPU port is debugged against.
//
// What the pipeline wants out of this model is narrower than the model:
// ZImagePipeline._encode_prompt asks for `hidden_states[-2]`, which is the
// output of decoder layer 34 of 36 -- no final norm, no lm_head, and the
// last layer never runs. EncoderLayers is that count, and the reference dump
// checks the identity rather than assuming it.
//
// Two details differ from the DiT next door and are the likely porting
// mistakes:
//
//   - RoPE here is NeoX style: component i pairs with i+head_dim/2, the two
//     halves of the head rotating against each other. The DiT pairs
//     *adjacent* components. Same rotation, different memory order.
//   - Attention is causal and grouped: 32 query heads over 8 key/value
//     heads, four queries per kv head.
//
// Layout is [tokens, features] row-major float32 throughout, matching a
// PyTorch tensor with the batch axis squeezed out, so a tensor here compares
// to a reference dump with no transpose in the way.
package qwen

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Mat is a dense row-major matrix.
type Mat struct {
	Rows, Cols int
	Data       []float32
}

// NewMat allocates a zeroed matrix.
func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

// Row returns one row.
func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Clone copies the matrix.
func (m *Mat) Clone() *Mat {
	return &Mat{Rows: m.Rows, Cols: m.Cols, Data: append([]float32(nil), m.Data...)}
}

func (m *Mat) String() string { return fmt.Sprintf("[%d %d]", m.Rows, m.Cols) }

func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// Linear is y = W x with W stored [Out, In], PyTorch's order. Qwen3 has
// attention_bias false and no bias anywhere else either, so there is no bias
// term here at all.
type Linear struct {
	In, Out int
	Weight  []float32
}

// Apply runs the projection over every row of x.
func (l *Linear) Apply(x *Mat) (*Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("qwen: linear takes %d features, got %d", l.In, x.Cols)
	}
	out := NewMat(x.Rows, l.Out)
	parallelFor(l.Out, func(o int) {
		w := l.Weight[o*l.In : (o+1)*l.In]
		for r := 0; r < x.Rows; r++ {
			row := x.Row(r)
			var sum float32
			for i, wv := range w {
				sum += wv * row[i]
			}
			out.Data[r*l.Out+o] = sum
		}
	})
	return out, nil
}

// RMSNorm is x * rsqrt(mean(x^2) + eps) * weight over the last len(Weight)
// components of a row. As in the DiT, a row several times that width is
// normalised span by span, which is what makes the q and k norms *per head*.
//
// transformers computes the reduction in float32 and multiplies the weight
// in the input dtype; everything here is float32 already, so there is no
// cast to reproduce.
type RMSNorm struct {
	Weight []float32
	Eps    float64
}

// Apply returns a normalised copy.
func (n *RMSNorm) Apply(x *Mat) (*Mat, error) {
	out := x.Clone()
	if err := n.ApplyInPlace(out); err != nil {
		return nil, err
	}
	return out, nil
}

// ApplyInPlace normalises every span of len(Weight) in every row.
func (n *RMSNorm) ApplyInPlace(x *Mat) error {
	w := len(n.Weight)
	if w == 0 || x.Cols%w != 0 {
		return fmt.Errorf("qwen: rms norm of width %d does not divide %d columns", w, x.Cols)
	}
	spans := x.Cols / w
	parallelFor(x.Rows, func(r int) {
		row := x.Row(r)
		for s := 0; s < spans; s++ {
			span := row[s*w : (s+1)*w]
			var sum float64
			for _, v := range span {
				sum += float64(v) * float64(v)
			}
			scale := float32(1 / math.Sqrt(sum/float64(w)+n.Eps))
			for i := range span {
				span[i] *= scale * n.Weight[i]
			}
		}
	})
	return nil
}

// RoPE holds the rotary tables for one sequence length.
//
// transformers builds cos and sin as cat(freqs, freqs) over the full head
// width; only half of that is distinct, so this keeps the half and Apply
// indexes it twice. Positions are 0..T-1: the encoder sees one sequence with
// no cache.
type RoPE struct {
	HeadDim  int
	Cos, Sin []float32 // [T, HeadDim/2]
}

// NewRoPE builds the tables for T positions.
func NewRoPE(headDim, T int, theta float64) *RoPE {
	half := headDim / 2
	r := &RoPE{HeadDim: headDim, Cos: make([]float32, T*half), Sin: make([]float32, T*half)}
	for i := 0; i < half; i++ {
		freq := 1 / math.Pow(theta, float64(2*i)/float64(headDim))
		for p := 0; p < T; p++ {
			angle := float64(p) * freq
			r.Cos[p*half+i] = float32(math.Cos(angle))
			r.Sin[p*half+i] = float32(math.Sin(angle))
		}
	}
	return r
}

// Apply rotates every head of every row in place. x is [T, heads*HeadDim].
//
// The pairing is NeoX's: component i moves against component i+half, not
// against its neighbour. Getting this wrong leaves the norms and the
// magnitudes intact and only the attention pattern wrong, which is why it is
// checked against a dumped q_roped rather than eyeballed.
func (r *RoPE) Apply(x *Mat) error {
	half := r.HeadDim / 2
	if x.Cols%r.HeadDim != 0 {
		return fmt.Errorf("qwen: rope of head dim %d does not divide %d columns", r.HeadDim, x.Cols)
	}
	if x.Rows*half > len(r.Cos) {
		return fmt.Errorf("qwen: rope table holds %d positions, need %d", len(r.Cos)/half, x.Rows)
	}
	heads := x.Cols / r.HeadDim
	parallelFor(x.Rows, func(p int) {
		cos, sin := r.Cos[p*half:(p+1)*half], r.Sin[p*half:(p+1)*half]
		row := x.Row(p)
		for h := 0; h < heads; h++ {
			head := row[h*r.HeadDim : (h+1)*r.HeadDim]
			for i := 0; i < half; i++ {
				lo, hi := head[i], head[i+half]
				head[i] = lo*cos[i] - hi*sin[i]
				head[i+half] = hi*cos[i] + lo*sin[i]
			}
		}
	})
	return nil
}

// silu is x * sigmoid(x), Qwen3's hidden_act.
func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// Attention is causal grouped-query attention over one sequence.
//
// q is [T, heads*headDim] and k, v are [T, kvHeads*headDim]; each key/value
// head serves heads/kvHeads consecutive query heads, which is
// repeat_interleave and not a tile.
func Attention(q, k, v *Mat, heads, kvHeads, headDim int) (*Mat, error) {
	T := q.Rows
	if k.Rows != T || v.Rows != T {
		return nil, fmt.Errorf("qwen: attention got %d q rows against %d k and %d v", T, k.Rows, v.Rows)
	}
	if q.Cols != heads*headDim || k.Cols != kvHeads*headDim || v.Cols != kvHeads*headDim {
		return nil, fmt.Errorf("qwen: attention shapes %s %s %s do not match %d/%d heads of %d",
			q, k, v, heads, kvHeads, headDim)
	}
	if heads%kvHeads != 0 {
		return nil, fmt.Errorf("qwen: %d heads do not divide into %d kv heads", heads, kvHeads)
	}
	rep := heads / kvHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := NewMat(T, heads*headDim)
	parallelFor(heads, func(h int) {
		kv := h / rep
		scores := make([]float32, T)
		for t := 0; t < T; t++ {
			qv := q.Row(t)[h*headDim : (h+1)*headDim]
			// Causal: position t attends to 0..t and nothing later, so the
			// masked scores are never formed rather than set to -inf.
			max := float32(math.Inf(-1))
			for s := 0; s <= t; s++ {
				kvRow := k.Row(s)[kv*headDim : (kv+1)*headDim]
				var dot float32
				for i, qi := range qv {
					dot += qi * kvRow[i]
				}
				scores[s] = dot * scale
				if scores[s] > max {
					max = scores[s]
				}
			}
			var sum float32
			for s := 0; s <= t; s++ {
				scores[s] = float32(math.Exp(float64(scores[s] - max)))
				sum += scores[s]
			}
			dst := out.Row(t)[h*headDim : (h+1)*headDim]
			for s := 0; s <= t; s++ {
				p := scores[s] / sum
				vRow := v.Row(s)[kv*headDim : (kv+1)*headDim]
				for i, vv := range vRow {
					dst[i] += p * vv
				}
			}
		}
	})
	return out, nil
}

// Trace collects named intermediates of a forward pass. A nil Trace costs
// nothing; a non-nil one is how a stagewise test walks the graph against the
// reference dump.
type Trace map[string]*Mat

func (tr Trace) put(name string, m *Mat) {
	if tr != nil {
		tr[name] = m.Clone()
	}
}

// Layer is one Qwen3 decoder layer.
type Layer struct {
	AttnNorm *RMSNorm
	Q, K, V  *Linear
	QNorm    *RMSNorm // per head, over head_dim
	KNorm    *RMSNorm
	O        *Linear
	FFNNorm  *RMSNorm
	Gate     *Linear
	Up       *Linear
	Down     *Linear

	Heads, KVHeads, HeadDim int
}

// Forward runs the layer: pre-norm attention with a residual, then pre-norm
// SwiGLU with a residual. prefix names the trace entries.
func (l *Layer) Forward(x *Mat, rope *RoPE, tr Trace, prefix string) (*Mat, error) {
	normed, err := l.AttnNorm.Apply(x)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"input_layernorm", normed)

	q, err := l.Q.Apply(normed)
	if err != nil {
		return nil, err
	}
	k, err := l.K.Apply(normed)
	if err != nil {
		return nil, err
	}
	v, err := l.V.Apply(normed)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"q", q)
	tr.put(prefix+"k", k)
	tr.put(prefix+"v", v)

	// The q/k norms are per head: the weight is [head_dim] and the row holds
	// every head, so ApplyInPlace normalises each head's span on its own.
	if err := l.QNorm.ApplyInPlace(q); err != nil {
		return nil, err
	}
	if err := l.KNorm.ApplyInPlace(k); err != nil {
		return nil, err
	}
	tr.put(prefix+"q_normed", q)
	tr.put(prefix+"k_normed", k)

	if err := rope.Apply(q); err != nil {
		return nil, err
	}
	if err := rope.Apply(k); err != nil {
		return nil, err
	}
	tr.put(prefix+"q_roped", q)
	tr.put(prefix+"k_roped", k)

	ctx, err := Attention(q, k, v, l.Heads, l.KVHeads, l.HeadDim)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"attn_ctx", ctx)

	attn, err := l.O.Apply(ctx)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"attn_out", attn)

	resid := x.Clone()
	for i, v := range attn.Data {
		resid.Data[i] += v
	}
	tr.put(prefix+"resid1", resid)

	normed2, err := l.FFNNorm.Apply(resid)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"post_attention_layernorm", normed2)

	gate, err := l.Gate.Apply(normed2)
	if err != nil {
		return nil, err
	}
	up, err := l.Up.Apply(normed2)
	if err != nil {
		return nil, err
	}
	for i := range gate.Data {
		gate.Data[i] = silu(gate.Data[i]) * up.Data[i]
	}
	mlp, err := l.Down.Apply(gate)
	if err != nil {
		return nil, err
	}
	tr.put(prefix+"mlp", mlp)

	for i, v := range mlp.Data {
		resid.Data[i] += v
	}
	tr.put(prefix+"layer_out", resid)
	return resid, nil
}

// Model is the text encoder.
type Model struct {
	Cfg    *Config
	Embed  []float32 // [rows, HiddenSize]; rows may be fewer than vocab_size
	Rows   int
	Layers []*Layer
}

// Embeddings looks the token ids up in the embedding table.
func (m *Model) Embeddings(ids []int32) (*Mat, error) {
	out := NewMat(len(ids), m.Cfg.HiddenSize)
	for i, id := range ids {
		if id < 0 || int(id) >= m.Rows {
			return nil, fmt.Errorf("qwen: token id %d is outside the %d embedding rows", id, m.Rows)
		}
		copy(out.Row(i), m.Embed[int(id)*m.Cfg.HiddenSize:(int(id)+1)*m.Cfg.HiddenSize])
	}
	return out, nil
}

// Forward runs the loaded layers over a token sequence and returns the
// hidden state. With EncoderLayers layers loaded that is `hidden_states[-2]`,
// which is what the pipeline hands the DiT's cap_embedder.
func (m *Model) Forward(ids []int32, tr Trace) (*Mat, error) {
	x, err := m.Embeddings(ids)
	if err != nil {
		return nil, err
	}
	tr.put("embeddings", x)
	rope := NewRoPE(m.Cfg.HeadDim, len(ids), m.Cfg.RopeTheta)
	for i, l := range m.Layers {
		prefix := ""
		if tr != nil {
			prefix = fmt.Sprintf("layer%d.", i)
		}
		x, err = l.Forward(x, rope, tr, prefix)
		if err != nil {
			return nil, fmt.Errorf("qwen: layer %d: %w", i, err)
		}
		tr.put(fmt.Sprintf("hidden_%d", i+1), x)
	}
	return x, nil
}
