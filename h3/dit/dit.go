// Package dit implements MiniMax-H3's omni transformer on the CPU (VIDEO.md
// M3): the oracle the Vulkan port is debugged against, written against
// diffusers' transformer_minimax_h3.py and gated on
// reference/dump_h3_dit_block.py before any shader exists.
//
// Where this model differs from the image DiT next door (qimage/dit), which
// is where the porting mistakes live:
//
//   - One packed sequence and *full* self-attention over it: text, keyframe,
//     audio and video rows all attend to all. There is no mask, no
//     cross-attention and no cacheable prefix — text rows are modulated and
//     updated every step like everything else.
//   - Per-row AdaLN. Every block owns a projection from the timestep
//     embedding to a table of six modulation vectors per (timestep,
//     modality); a row reads row `timestep_index·3 + modality` of it. The
//     projection's input is the timestep alone, so the table is computed
//     once per request (Table) and the 26 GB of projections never need to be
//     near the device.
//   - The attention is wider than the stream: 56 heads × 128 = 7168 against
//     a 5376 residual.
//   - RoPE covers 96 of the 128 head channels, in the rotate-half (NeoX)
//     convention, from float64 (t, h, w) positions narrowed to float32; the
//     tables come from h3/plan. Channels 96–127 pass through.
//   - SwiGLU's projection is [value | gate], value first: `value · silu(gate)`.
//   - Pre-norms are RMSNorms *with* a weight, then `·(1 + scale) + shift`.
//
// Layout is [rows, features] row-major float32, reusing zimage/qwen's Mat,
// Linear and RMSNorm.
package dit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// Modalities is how many modulation rows each timestep has: video, text,
// audio, in the order of plan's tags.
const Modalities = 3

// Config mirrors transformer/config.json.
type Config struct {
	Heads         int     `json:"num_attention_heads"`
	HeadDim       int     `json:"attention_head_dim"`
	Hidden        int     `json:"hidden_size"`
	Layers        int     `json:"num_layers"`
	RefinerLayers int     `json:"num_refiner_layers"`
	FFN           int     `json:"ffn_dim"`
	InChannels    int     `json:"in_channels"`
	AudioChannels int     `json:"audio_in_channels"`
	PatchSize     []int   `json:"patch_size"`
	TextDim       int     `json:"text_dim"`
	FreqDim       int     `json:"freq_dim"`
	TimeHidden    int     `json:"time_embed_hidden_dim"`
	TimeDim       int     `json:"time_embed_dim"`
	RopeFreqDim   int     `json:"rope_freq_dim"`
	RopeTheta     float64 `json:"rope_theta"`
	NormEps       float64 `json:"norm_eps"`
	QKNormEps     float64 `json:"qk_norm_eps"`
	FinalNormEps  float64 `json:"final_norm_eps"`
}

// Inner is the attention width, heads × head_dim.
func (c *Config) Inner() int { return c.Heads * c.HeadDim }

// Patch is the width of one packed video row, in_channels × prod(patch).
func (c *Config) Patch() int {
	n := c.InChannels
	for _, p := range c.PatchSize {
		n *= p
	}
	return n
}

// RopeWidth is how many head channels the rotary embedding covers.
func (c *Config) RopeWidth() int { return 2 * 3 * c.RopeFreqDim }

// LoadConfig reads the transformer's config.json and checks it against what
// this port and h3/plan implement.
func LoadConfig(dir string) (*Config, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(buf, &c); err != nil {
		return nil, fmt.Errorf("dit: parsing config.json: %w", err)
	}
	switch {
	case c.Heads == 0 || c.HeadDim == 0 || c.Hidden == 0 || c.FFN == 0:
		return nil, fmt.Errorf("dit: config.json is missing a dimension")
	case len(c.PatchSize) != 3 || c.PatchSize[0] != 1 || c.PatchSize[1] != plan.Patch || c.PatchSize[2] != plan.Patch:
		return nil, fmt.Errorf("dit: patch_size %v is not the (1, %d, %d) h3/plan lays out", c.PatchSize, plan.Patch, plan.Patch)
	case c.RopeFreqDim != plan.RopeFreqDim || c.RopeTheta != plan.RopeTheta:
		return nil, fmt.Errorf("dit: rope %d/%g is not h3/plan's %d/%g", c.RopeFreqDim, c.RopeTheta, plan.RopeFreqDim, plan.RopeTheta)
	case c.RopeWidth() > c.HeadDim:
		return nil, fmt.Errorf("dit: rope covers %d channels of a %d head", c.RopeWidth(), c.HeadDim)
	case c.FreqDim%2 != 0:
		return nil, fmt.Errorf("dit: freq_dim %d is odd", c.FreqDim)
	}
	return &c, nil
}

// Linear is qwen.Linear's weight layout plus an optional bias, which the
// input and output projections, the timestep MLP and the AdaLN projections
// carry.
//
// Apply accumulates in float64, unlike qwen.Linear's single float32
// accumulator. This model's rows carry massive-activation channels whose
// products cancel: one audio-head output is 72 from terms summing to 137 in
// magnitude, and a sequential float32 sum over 5376 of them lands 2e-3 off
// where torch's blocked fp32 lands 4e-5 off (M3). An oracle has to be the
// more accurate side of the comparison.
type Linear struct {
	qwen.Linear
	Bias []float32
}

// Apply runs the projection over every row of x.
func (l *Linear) Apply(x *qwen.Mat) (*qwen.Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("dit: linear takes %d features, got %d", l.In, x.Cols)
	}
	out := qwen.NewMat(x.Rows, l.Out)
	parallel(l.Out, func(o int) {
		w := l.Weight[o*l.In : (o+1)*l.In]
		var b float64
		if l.Bias != nil {
			b = float64(l.Bias[o])
		}
		for r := 0; r < x.Rows; r++ {
			row := x.Row(r)
			var s0, s1, s2, s3 float64
			i := 0
			for ; i+4 <= len(w); i += 4 {
				s0 += float64(w[i]) * float64(row[i])
				s1 += float64(w[i+1]) * float64(row[i+1])
				s2 += float64(w[i+2]) * float64(row[i+2])
				s3 += float64(w[i+3]) * float64(row[i+3])
			}
			for ; i < len(w); i++ {
				s0 += float64(w[i]) * float64(row[i])
			}
			out.Data[r*l.Out+o] = float32((s0 + s1) + (s2 + s3) + b)
		}
	})
	return out, nil
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// siluMat returns silu applied elementwise.
func siluMat(x *qwen.Mat) *qwen.Mat {
	out := x.Clone()
	for i, v := range out.Data {
		out.Data[i] = silu(v)
	}
	return out
}

// Attention is the block's (and the refiner's) self-attention: q/k/v with no
// bias, a per-head RMSNorm on q and k, optional rotary tables, full
// unmasked softmax attention, and the output projection.
type Attention struct {
	Q, K, V, O   *Linear
	QNorm, KNorm *qwen.RMSNorm
	Heads        int
	HeadDim      int
}

// Forward runs attention over x. cos and sin are [rows, ropeWidth] or nil.
func (a *Attention) Forward(x *qwen.Mat, cos, sin []float32, ropeWidth int) (*qwen.Mat, error) {
	q, err := a.Q.Apply(x)
	if err != nil {
		return nil, err
	}
	k, err := a.K.Apply(x)
	if err != nil {
		return nil, err
	}
	v, err := a.V.Apply(x)
	if err != nil {
		return nil, err
	}
	if err := a.QNorm.ApplyInPlace(q); err != nil {
		return nil, err
	}
	if err := a.KNorm.ApplyInPlace(k); err != nil {
		return nil, err
	}
	if cos != nil {
		rotate(q, cos, sin, a.Heads, a.HeadDim, ropeWidth)
		rotate(k, cos, sin, a.Heads, a.HeadDim, ropeWidth)
	}
	return a.O.Apply(attend(q, k, v, a.Heads, a.HeadDim))
}

// rotate applies rotate-half RoPE to the leading width channels of every
// head: x·cos + cat(−x₂, x₁)·sin, with x₁, x₂ the two halves of that span.
func rotate(x *qwen.Mat, cos, sin []float32, heads, headDim, width int) {
	half := width / 2
	parallel(x.Rows, func(r int) {
		c, s := cos[r*width:(r+1)*width], sin[r*width:(r+1)*width]
		row := x.Row(r)
		for h := 0; h < heads; h++ {
			v := row[h*headDim : h*headDim+width]
			var tmp [128]float32
			copy(tmp[:width], v)
			for i := 0; i < half; i++ {
				v[i] = tmp[i]*c[i] - tmp[i+half]*s[i]
				v[i+half] = tmp[i+half]*c[i+half] + tmp[i]*s[i+half]
			}
		}
	})
}

// attend is softmax(q kᵀ / √d) v per head over the whole sequence, in
// float64 accumulation so the oracle's own error stays below fp32 noise.
func attend(q, k, v *qwen.Mat, heads, headDim int) *qwen.Mat {
	n := q.Rows
	out := qwen.NewMat(n, heads*headDim)
	scale := 1 / math.Sqrt(float64(headDim))
	parallel(heads*n, func(job int) {
		h, i := job/n, job%n
		qi := q.Row(i)[h*headDim : (h+1)*headDim]
		scores := make([]float64, n)
		best := math.Inf(-1)
		for j := 0; j < n; j++ {
			kj := k.Row(j)[h*headDim : (h+1)*headDim]
			var dot float64
			for d, qv := range qi {
				dot += float64(qv) * float64(kj[d])
			}
			scores[j] = dot * scale
			best = math.Max(best, scores[j])
		}
		var sum float64
		for j := range scores {
			scores[j] = math.Exp(scores[j] - best)
			sum += scores[j]
		}
		acc := make([]float64, headDim)
		for j, p := range scores {
			vj := v.Row(j)[h*headDim : (h+1)*headDim]
			for d, vv := range vj {
				acc[d] += p * float64(vv)
			}
		}
		o := out.Row(i)[h*headDim : (h+1)*headDim]
		for d := range o {
			o[d] = float32(acc[d] / sum)
		}
	})
	return out
}

// SwiGLU is the feed-forward: Up projects to [value | gate], the product
// value·silu(gate) goes through Down. Neither projection has a bias.
type SwiGLU struct {
	Up, Down *Linear
	Inner    int
}

// Forward runs the feed-forward over x.
func (f *SwiGLU) Forward(x *qwen.Mat) (*qwen.Mat, error) {
	up, err := f.Up.Apply(x)
	if err != nil {
		return nil, err
	}
	mid := qwen.NewMat(x.Rows, f.Inner)
	for r := 0; r < x.Rows; r++ {
		u, m := up.Row(r), mid.Row(r)
		for i := range m {
			m[i] = u[i] * silu(u[f.Inner+i])
		}
	}
	return f.Down.Apply(mid)
}

// RefinerBlock is a plain pre-norm transformer block over the text rows:
// no AdaLN and no rotary embedding.
type RefinerBlock struct {
	Norm1, Norm2 *qwen.RMSNorm
	Attn         *Attention
	FF           *SwiGLU
}

// Forward runs the block.
func (b *RefinerBlock) Forward(x *qwen.Mat) (*qwen.Mat, error) {
	n, err := b.Norm1.Apply(x)
	if err != nil {
		return nil, err
	}
	a, err := b.Attn.Forward(n, nil, nil, 0)
	if err != nil {
		return nil, err
	}
	x = add(x, a, nil)
	if n, err = b.Norm2.Apply(x); err != nil {
		return nil, err
	}
	f, err := b.FF.Forward(n)
	if err != nil {
		return nil, err
	}
	return add(x, f, nil), nil
}

// Table is one block's AdaLN modulation for a request: for every distinct
// timestep t and modality m, the six vectors shift_msa, scale_msa, gate_msa,
// shift_mlp, scale_mlp, gate_mlp, each Hidden wide. Row t·3+m.
type Table struct {
	Hidden int
	Rows   int
	Data   []float32 // [Rows][6][Hidden]
}

// Vec returns modulation vector j (0..5, in the order above) of table row r.
func (t *Table) Vec(r, j int) []float32 {
	o := (r*6 + j) * t.Hidden
	return t.Data[o : o+t.Hidden]
}

// Block is one of the 50 transformer blocks.
type Block struct {
	Norm1, Norm2 *qwen.RMSNorm
	Attn         *Attention
	FF           *SwiGLU
	AdaLN        *Linear // time_embed_dim → 6·3·hidden
	Hidden       int
	RopeWidth    int
}

// Table computes this block's modulation for temb, [T, time_embed_dim]:
// the projection of silu(temb), viewed as T·3 rows of 6·hidden.
func (b *Block) Table(temb *qwen.Mat) (*Table, error) {
	out, err := b.AdaLN.Apply(siluMat(temb))
	if err != nil {
		return nil, err
	}
	return &Table{Hidden: b.Hidden, Rows: temb.Rows * Modalities, Data: out.Data}, nil
}

// Forward runs the block over the packed sequence x. rowMod[i] is row i's
// table row, timestep_index·3 + tag.
func (b *Block) Forward(x *qwen.Mat, tab *Table, rowMod []int32, cos, sin []float32) (*qwen.Mat, error) {
	n, err := b.Norm1.Apply(x)
	if err != nil {
		return nil, err
	}
	modulate(n, tab, rowMod, 0)
	a, err := b.Attn.Forward(n, cos, sin, b.RopeWidth)
	if err != nil {
		return nil, err
	}
	x = add(x, a, func(r int) []float32 { return tab.Vec(int(rowMod[r]), 2) })
	if n, err = b.Norm2.Apply(x); err != nil {
		return nil, err
	}
	modulate(n, tab, rowMod, 3)
	f, err := b.FF.Forward(n)
	if err != nil {
		return nil, err
	}
	return add(x, f, func(r int) []float32 { return tab.Vec(int(rowMod[r]), 5) }), nil
}

// modulate applies n·(1 + scale) + shift in place, with shift and scale
// vectors first and first+1 of each row's table row.
func modulate(n *qwen.Mat, tab *Table, rowMod []int32, first int) {
	parallel(n.Rows, func(r int) {
		shift, scale := tab.Vec(int(rowMod[r]), first), tab.Vec(int(rowMod[r]), first+1)
		row := n.Row(r)
		for i := range row {
			row[i] = row[i]*(1+scale[i]) + shift[i]
		}
	})
}

// add returns x + gate(r)·d, or x + d when gate is nil.
func add(x, d *qwen.Mat, gate func(r int) []float32) *qwen.Mat {
	out := x.Clone()
	for r := 0; r < out.Rows; r++ {
		o, dr := out.Row(r), d.Row(r)
		if gate == nil {
			for i := range o {
				o[i] += dr[i]
			}
			continue
		}
		g := gate(r)
		for i := range o {
			o[i] += g[i] * dr[i]
		}
	}
	return out
}

// TimeProj is the sinusoidal timestep embedding, flip_sin_to_cos with no
// frequency shift: [cos(t·f), sin(t·f)] with f = exp(−ln(10⁴)·i/half), in
// float32 as diffusers computes it. H3 feeds t in [0, 1] unscaled.
func TimeProj(ts []float32, dim int) *qwen.Mat {
	half := dim / 2
	freq := make([]float32, half)
	for i := range freq {
		e := float32(-math.Log(10000)) * float32(i)
		freq[i] = float32(math.Exp(float64(e / float32(half))))
	}
	out := qwen.NewMat(len(ts), dim)
	for r, t := range ts {
		row := out.Row(r)
		for i, f := range freq {
			a := float64(t * f)
			row[i] = float32(math.Cos(a))
			row[half+i] = float32(math.Sin(a))
		}
	}
	return out
}

// Model is the transformer's pieces. Blocks may hold fewer than the config's
// layers — the block gate loads two.
type Model struct {
	Cfg *Config

	TimeL1, TimeL2              *Linear
	ProjIn, AudioProjIn, TextIn *Linear
	Refiner                     []*RefinerBlock
	RefinerNorm                 *qwen.RMSNorm
	Blocks                      []*Block
	NormOut                     *qwen.RMSNorm
	NormOutLinear               *Linear // time_embed_dim → [shift | scale]
	ProjOut, AudioProjOut       *Linear
}

// TimeEmbed is temb for the distinct timesteps of one forward,
// [T, time_embed_dim]: linear_2(silu(linear_1(TimeProj(t)))).
func (m *Model) TimeEmbed(ts []float32) (*qwen.Mat, error) {
	h, err := m.TimeL1.Apply(TimeProj(ts, m.Cfg.FreqDim))
	if err != nil {
		return nil, err
	}
	return m.TimeL2.Apply(siluMat(h))
}

// Text projects the encoder's conditioning into the stream and refines it:
// what the text rows of the packed sequence start every forward from. It
// depends on the prompt alone, so it runs once per request.
func (m *Model) Text(cond *qwen.Mat) (*qwen.Mat, error) {
	x, err := m.TextIn.Apply(cond)
	if err != nil {
		return nil, err
	}
	for _, b := range m.Refiner {
		if x, err = b.Forward(x); err != nil {
			return nil, err
		}
	}
	return m.RefinerNorm.Apply(x)
}

// Pack projects the video and audio rows and scatters them, with the
// refined text rows, into the packed sequence of l.
func (m *Model) Pack(l *plan.Layout, text, video, audio *qwen.Mat) (*qwen.Mat, error) {
	v, err := m.ProjIn.Apply(video)
	if err != nil {
		return nil, err
	}
	a, err := m.AudioProjIn.Apply(audio)
	if err != nil {
		return nil, err
	}
	if text.Rows != len(l.Text) || v.Rows != len(l.Video) || a.Rows != len(l.Audio) {
		return nil, fmt.Errorf("dit: %d/%d/%d text/video/audio rows for a layout of %d/%d/%d",
			text.Rows, v.Rows, a.Rows, len(l.Text), len(l.Video), len(l.Audio))
	}
	x := qwen.NewMat(len(l.Pos), m.Cfg.Hidden)
	for src, idx := range map[*qwen.Mat][]int32{text: l.Text, v: l.Video, a: l.Audio} {
		for i, r := range idx {
			copy(x.Row(int(r)), src.Row(i))
		}
	}
	return x, nil
}

// Tail is the output norm and both heads over the last block's output:
// RMSNorm, ·(1 + scale) + shift per row's timestep, then the video head over
// l's video rows and the audio head over its audio rows.
func (m *Model) Tail(x *qwen.Mat, temb *qwen.Mat, rowT []int32, l *plan.Layout) (normed, video, audio *qwen.Mat, err error) {
	mod, err := m.NormOutLinear.Apply(siluMat(temb))
	if err != nil {
		return nil, nil, nil, err
	}
	if normed, err = m.NormOut.Apply(x); err != nil {
		return nil, nil, nil, err
	}
	H := m.Cfg.Hidden
	for r := 0; r < normed.Rows; r++ {
		mr := mod.Row(int(rowT[r]))
		shift, scale := mr[:H], mr[H:]
		row := normed.Row(r)
		for i := range row {
			row[i] = row[i]*(1+scale[i]) + shift[i]
		}
	}
	if video, err = m.ProjOut.Apply(gather(normed, l.Video)); err != nil {
		return nil, nil, nil, err
	}
	audio, err = m.AudioProjOut.Apply(gather(normed, l.Audio))
	return normed, video, audio, err
}

func gather(x *qwen.Mat, idx []int32) *qwen.Mat {
	out := qwen.NewMat(len(idx), x.Cols)
	for i, r := range idx {
		copy(out.Row(i), x.Row(int(r)))
	}
	return out
}

// RowMod is every row's AdaLN table row, timestep_index·3 + tag.
func RowMod(rowT, tags []int32) []int32 {
	out := make([]int32, len(rowT))
	for i := range out {
		out[i] = rowT[i]*Modalities + tags[i]
	}
	return out
}

func parallel(n int, fn func(i int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
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
