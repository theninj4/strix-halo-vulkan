// Package dit is FLUX 3 Image Klein's transformer on the CPU, in fp32: the
// oracle the GPU port is debugged against (X2, research/flux3-vertical.md).
//
// It follows diffusers' Flux3Transformer2DModel
// (reference/flux3/diffusers/transformer_flux3.py) — the oracle the dumps
// come from — and BFL's DiT (reference/flux3/bfl/dit.py), which is the same
// function with the joint sequence in another order.
//
// Tokens come in three modes. Each mode has its own NumLayers mode blocks
// and attends only to itself there; then the modes are concatenated as
// [text, image, image_cond] and run through NumSingle shared single blocks
// with full attention and no mask. Every block is the same parallel
// attention + SwiGLU, so one Block type serves both kinds: a block is
// handed its rows as segments, each with the (shift, scale, gate) of its
// mode, and a mode block is simply a block whose sequence is one segment.
//
// Modulation is per mode, not per block: one Linear per mode for the early
// stage and one for the single stage, on silu(temb), where temb is the
// timestep embedding of t for the image and of 0 for text and references,
// plus — every mode — the target-timestep embedding of the time this step
// jumps to (the flow map). Nothing is cacheable across steps.
//
// The residual stream is far outside fp16 here (text rows reach 1.1e7, X0
// finding 2); this package is fp32 throughout and says nothing about that.
package dit

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"strix-halo-vulkan/zimage/qwen"
)

// Mode is a token mode.
type Mode int

const (
	Text Mode = iota
	Image
	Cond
)

// Modes in the joint sequence's order (diffusers': text first).
var Modes = []Mode{Text, Image, Cond}

func (m Mode) key() string { return [...]string{"txt", "image", "image_cond"}[m] }

func (m Mode) blockPrefix() string {
	return [...]string{"txt_mode_blocks", "mode_blocks.image", "mode_blocks.image_cond"}[m]
}

func (m Mode) String() string { return m.key() }

// Rope is the rotary table for a sequence: [seq, headDim/2] cos and sin.
type Rope struct {
	Cos, Sin *qwen.Mat
}

// NewRope builds the table from position ids [seq, len(axes)] — float,
// because image ids are scaled by 0.5. Each axis of width d takes d/2
// adjacent pairs, axis-major, with frequencies 1/theta^(2k/d) and angles in
// float64, as diffusers' Flux3PosEmbed computes them.
func NewRope(ids *qwen.Mat, axes []int, theta float64) (*Rope, error) {
	if ids.Cols != len(axes) {
		return nil, fmt.Errorf("dit: ids have %d axes, rope has %d", ids.Cols, len(axes))
	}
	half := 0
	for _, d := range axes {
		half += d / 2
	}
	r := &Rope{Cos: qwen.NewMat(ids.Rows, half), Sin: qwen.NewMat(ids.Rows, half)}
	for t := 0; t < ids.Rows; t++ {
		cos, sin, pos := r.Cos.Row(t), r.Sin.Row(t), ids.Row(t)
		at := 0
		for a, d := range axes {
			for k := 0; k < d/2; k++ {
				angle := float64(pos[a]) / math.Pow(theta, float64(2*k)/float64(d))
				cos[at] = float32(math.Cos(angle))
				sin[at] = float32(math.Sin(angle))
				at++
			}
		}
	}
	return r, nil
}

// Concat stacks tables in sequence order.
func Concat(rs ...*Rope) *Rope {
	var c, s []*qwen.Mat
	for _, r := range rs {
		if r != nil {
			c, s = append(c, r.Cos), append(s, r.Sin)
		}
	}
	return &Rope{Cos: stack(c...), Sin: stack(s...)}
}

// Apply rotates every head of every row in place, pairing adjacent
// components (x[2j], x[2j+1]) — diffusers' repeat_interleave_real.
func (r *Rope) Apply(x *qwen.Mat, headDim int) error {
	half := headDim / 2
	if r.Cos.Cols != half || x.Cols%headDim != 0 || x.Rows != r.Cos.Rows {
		return fmt.Errorf("dit: rope %s against %s with head dim %d", r.Cos, x, headDim)
	}
	heads := x.Cols / headDim
	parallel(x.Rows, func(t int) {
		cos, sin, row := r.Cos.Row(t), r.Sin.Row(t), x.Row(t)
		for h := 0; h < heads; h++ {
			seg := row[h*headDim : (h+1)*headDim]
			for j := 0; j < half; j++ {
				re, im := seg[2*j], seg[2*j+1]
				seg[2*j] = re*cos[j] - im*sin[j]
				seg[2*j+1] = im*cos[j] + re*sin[j]
			}
		}
	})
	return nil
}

// Mod is one mode's modulation: shift, scale and gate, each [dim].
type Mod struct {
	Shift, Scale, Gate []float32
}

// Segment is a run of rows sharing one modulation.
type Segment struct {
	Rows int
	Mod  Mod
}

// Block is a parallel attention + SwiGLU block with no biases.
type Block struct {
	Q, K, V, Out *qwen.Linear
	NormQ, NormK *qwen.RMSNorm // per head, over head_dim
	FFIn, FFOut  *qwen.Linear  // FFIn is gate|up, 2*mlp wide
	Heads        int
	HeadDim      int
	Eps          float64
}

// Forward returns x + gate * (attn(m) + ff(m)) with m = (1+scale)*LN(x)+shift,
// each row taking its segment's modulation; attention is full over all rows.
func (b *Block) Forward(x *qwen.Mat, segs []Segment, rope *Rope) (*qwen.Mat, error) {
	if n := rows(segs); n != x.Rows {
		return nil, fmt.Errorf("dit: segments cover %d rows of %d", n, x.Rows)
	}
	m := layerNorm(x, b.Eps)
	eachRow(segs, func(r int, mod Mod) {
		row := m.Row(r)
		for i, v := range row {
			row[i] = (1+mod.Scale[i])*v + mod.Shift[i]
		}
	})
	q, err := b.Q.Apply(m)
	if err != nil {
		return nil, err
	}
	k, err := b.K.Apply(m)
	if err != nil {
		return nil, err
	}
	v, err := b.V.Apply(m)
	if err != nil {
		return nil, err
	}
	if err := b.NormQ.ApplyInPlace(q); err != nil {
		return nil, err
	}
	if err := b.NormK.ApplyInPlace(k); err != nil {
		return nil, err
	}
	if err := rope.Apply(q, b.HeadDim); err != nil {
		return nil, err
	}
	if err := rope.Apply(k, b.HeadDim); err != nil {
		return nil, err
	}
	attn, err := b.Out.Apply(attend(q, k, v, b.Heads, b.HeadDim))
	if err != nil {
		return nil, err
	}
	h, err := b.FFIn.Apply(m)
	if err != nil {
		return nil, err
	}
	ff, err := b.FFOut.Apply(swiglu(h))
	if err != nil {
		return nil, err
	}
	out := x.Clone()
	eachRow(segs, func(r int, mod Mod) {
		dst, a, f := out.Row(r), attn.Row(r), ff.Row(r)
		for i := range dst {
			dst[i] += mod.Gate[i] * (a[i] + f[i])
		}
	})
	return out, nil
}

// Model is the transformer.
type Model struct {
	Cfg                  *Config
	ImgIn, CondIn, TxtIn *qwen.Linear
	Time1, Time2         *qwen.Linear // timestep_embedder
	Target1, Target2     *qwen.Linear // target_timestep_embedder (flow map)
	EarlyMod, SingleMod  map[Mode]*qwen.Linear
	ModeBlocks           map[Mode][]*Block
	Single               []*Block
	NormOut, ProjOut     *qwen.Linear
}

// TimestepEmbedding is the sinusoidal embedding of 1000*t: cos half first,
// frequencies exp(-ln(10000) k / half), as diffusers' Timesteps with
// flip_sin_to_cos and no frequency shift (and BFL's timestep_embedding).
func TimestepEmbedding(t float64, dim int) []float32 {
	half := dim / 2
	out := make([]float32, dim)
	ts := float32(t) * 1000
	for k := 0; k < half; k++ {
		freq := float32(math.Exp(-math.Log(10000) * float64(k) / float64(half)))
		arg := float64(ts * freq)
		out[k] = float32(math.Cos(arg))
		out[half+k] = float32(math.Sin(arg))
	}
	return out
}

// gemv is one row through a Linear with float64 accumulation. The
// conditioning path is all single rows, so the precision is free, and it is
// needed: where 1+scale nearly cancels against a large normalised channel, a
// float32 sequential sum's 1e-6 in scale became 1e-4 of norm_out (X2).
func gemv(l *qwen.Linear, x []float32) ([]float32, error) {
	if len(x) != l.In {
		return nil, fmt.Errorf("dit: linear takes %d features, got %d", l.In, len(x))
	}
	out := make([]float32, l.Out)
	parallel(l.Out, func(o int) {
		w := l.Weight[o*l.In : (o+1)*l.In]
		var sum float64
		for i, wv := range w {
			sum += float64(wv) * float64(x[i])
		}
		out[o] = float32(sum)
	})
	return out, nil
}

func mlp(l1, l2 *qwen.Linear, x []float32) ([]float32, error) {
	h, err := gemv(l1, x)
	if err != nil {
		return nil, err
	}
	for i, v := range h {
		h[i] = silu(v)
	}
	return gemv(l2, h)
}

// Temb is the conditioning embedding of a token at time t in a step that
// jumps to tNext: timestep_embedder(t) + target_timestep_embedder(tNext).
func (m *Model) Temb(t, tNext float64) ([]float32, error) {
	tc := m.Cfg.TimeChannels
	a, err := mlp(m.Time1, m.Time2, TimestepEmbedding(t, tc))
	if err != nil {
		return nil, err
	}
	b, err := mlp(m.Target1, m.Target2, TimestepEmbedding(tNext, tc))
	if err != nil {
		return nil, err
	}
	for i := range a {
		a[i] += b[i]
	}
	return a, nil
}

// modulate splits lin(silu(temb)) into (shift, scale, gate).
func modulate(lin *qwen.Linear, temb []float32) (Mod, error) {
	act := make([]float32, len(temb))
	for i, v := range temb {
		act[i] = silu(v)
	}
	o, err := gemv(lin, act)
	if err != nil {
		return Mod{}, err
	}
	d := len(temb)
	return Mod{Shift: o[:d], Scale: o[d : 2*d], Gate: o[2*d:]}, nil
}

// Mods are a step's modulations: per mode, the early and single sets.
type Mods struct {
	Early, Single map[Mode]Mod
	ImageTemb     []float32 // the output layer's conditioning
}

// Modulations computes a step's modulations at time t jumping to tNext.
func (m *Model) Modulations(t, tNext float64) (*Mods, error) {
	img, err := m.Temb(t, tNext)
	if err != nil {
		return nil, err
	}
	zero, err := m.Temb(0, tNext)
	if err != nil {
		return nil, err
	}
	out := &Mods{Early: map[Mode]Mod{}, Single: map[Mode]Mod{}, ImageTemb: img}
	for _, md := range Modes {
		temb := zero
		if md == Image {
			temb = img
		}
		if out.Early[md], err = modulate(m.EarlyMod[md], temb); err != nil {
			return nil, err
		}
		if out.Single[md], err = modulate(m.SingleMod[md], temb); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Input is one forward's tokens. Cond (and CondIDs) may be nil for t2i.
// Ids are [rows, 4]: (t, h, w, l), image h/w already scaled by 0.5.
type Input struct {
	Img, Txt, Cond          *qwen.Mat
	ImgIDs, TxtIDs, CondIDs *qwen.Mat
	T, TNext                float64
}

// Forward returns the velocity of the image tokens, [Img.Rows, InChannels].
func (m *Model) Forward(in *Input) (*qwen.Mat, error) {
	if len(m.Single) != m.Cfg.NumSingle || len(m.ModeBlocks[Image]) != m.Cfg.NumLayers {
		return nil, fmt.Errorf("dit: forward needs every block loaded")
	}
	mods, err := m.Modulations(in.T, in.TNext)
	if err != nil {
		return nil, err
	}
	axes, theta := m.Cfg.AxesDimsRope, m.Cfg.RopeTheta
	type stream struct {
		x    *qwen.Mat
		rope *Rope
	}
	streams := map[Mode]*stream{}
	add := func(md Mode, x, ids *qwen.Mat, lin *qwen.Linear) error {
		if x == nil {
			return nil
		}
		h, err := lin.Apply(x)
		if err != nil {
			return fmt.Errorf("dit: %s embedder: %w", md, err)
		}
		r, err := NewRope(ids, axes, theta)
		if err != nil {
			return err
		}
		if r.Cos.Rows != h.Rows {
			return fmt.Errorf("dit: %s has %d ids for %d tokens", md, r.Cos.Rows, h.Rows)
		}
		streams[md] = &stream{h, r}
		return nil
	}
	if err := add(Text, in.Txt, in.TxtIDs, m.TxtIn); err != nil {
		return nil, err
	}
	if err := add(Image, in.Img, in.ImgIDs, m.ImgIn); err != nil {
		return nil, err
	}
	if err := add(Cond, in.Cond, in.CondIDs, m.CondIn); err != nil {
		return nil, err
	}

	for _, md := range Modes {
		s := streams[md]
		if s == nil {
			continue
		}
		for i, b := range m.ModeBlocks[md] {
			if s.x, err = b.Forward(s.x, []Segment{{s.x.Rows, mods.Early[md]}}, s.rope); err != nil {
				return nil, fmt.Errorf("dit: %s mode block %d: %w", md, i, err)
			}
		}
	}

	var xs []*qwen.Mat
	var ropes []*Rope
	var segs []Segment
	for _, md := range Modes {
		if s := streams[md]; s != nil {
			xs, ropes = append(xs, s.x), append(ropes, s.rope)
			segs = append(segs, Segment{s.x.Rows, mods.Single[md]})
		}
	}
	x, rope := stack(xs...), Concat(ropes...)
	for i, b := range m.Single {
		if x, err = b.Forward(x, segs, rope); err != nil {
			return nil, fmt.Errorf("dit: single block %d: %w", i, err)
		}
	}

	from := 0
	if s := streams[Text]; s != nil {
		from = s.x.Rows
	}
	img := &qwen.Mat{Rows: in.Img.Rows, Cols: x.Cols, Data: x.Data[from*x.Cols : (from+in.Img.Rows)*x.Cols]}
	return m.Final(img, mods.ImageTemb)
}

// Final is the output layer: FinalNorm, then proj_out.
func (m *Model) Final(x *qwen.Mat, imageTemb []float32) (*qwen.Mat, error) {
	n, err := m.FinalNorm(x, imageTemb)
	if err != nil {
		return nil, err
	}
	return m.ProjOut.Apply(n)
}

// FinalNorm is norm_out: LN(x)*(1+scale)+shift with (scale, shift) =
// linear(silu(temb)) — scale first, unlike the blocks' shift first.
func (m *Model) FinalNorm(x *qwen.Mat, imageTemb []float32) (*qwen.Mat, error) {
	act := make([]float32, len(imageTemb))
	for i, v := range imageTemb {
		act[i] = silu(v)
	}
	e, err := gemv(m.NormOut, act)
	if err != nil {
		return nil, err
	}
	d := x.Cols
	scale, shift := e[:d], e[d:]
	n := layerNorm(x, m.Cfg.Eps)
	for r := 0; r < n.Rows; r++ {
		row := n.Row(r)
		for i, v := range row {
			row[i] = v*(1+scale[i]) + shift[i]
		}
	}
	return n, nil
}

// --- helpers ------------------------------------------------------------------

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// swiglu is silu(first half) * second half, row by row.
func swiglu(h *qwen.Mat) *qwen.Mat {
	n := h.Cols / 2
	out := qwen.NewMat(h.Rows, n)
	parallel(h.Rows, func(r int) {
		src, dst := h.Row(r), out.Row(r)
		for i := range dst {
			dst[i] = silu(src[i]) * src[n+i]
		}
	})
	return out
}

// layerNorm is a non-affine LayerNorm, accumulated in float64.
func layerNorm(x *qwen.Mat, eps float64) *qwen.Mat {
	out := qwen.NewMat(x.Rows, x.Cols)
	parallel(x.Rows, func(r int) {
		row, dst := x.Row(r), out.Row(r)
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(len(row))
		var vs float64
		for _, v := range row {
			d := float64(v) - mean
			vs += d * d
		}
		inv := 1 / math.Sqrt(vs/float64(len(row))+eps)
		for i, v := range row {
			dst[i] = float32((float64(v) - mean) * inv)
		}
	})
	return out
}

// attend is softmax(q kᵀ/√d) v per head, no mask.
func attend(q, k, v *qwen.Mat, heads, headDim int) *qwen.Mat {
	out := qwen.NewMat(q.Rows, heads*headDim)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	const chunk = 64
	nChunks := (q.Rows + chunk - 1) / chunk
	parallel(heads*nChunks, func(job int) {
		h, c := job/nChunks, job%nChunks
		scores := make([]float32, k.Rows)
		for qi := c * chunk; qi < min((c+1)*chunk, q.Rows); qi++ {
			qrow := q.Row(qi)[h*headDim : (h+1)*headDim]
			mx := float32(math.Inf(-1))
			for s := 0; s < k.Rows; s++ {
				krow := k.Row(s)[h*headDim : (h+1)*headDim]
				var dot float32
				for i, qv := range qrow {
					dot += qv * krow[i]
				}
				scores[s] = dot * scale
				if scores[s] > mx {
					mx = scores[s]
				}
			}
			var sum float32
			for s := range scores {
				scores[s] = float32(math.Exp(float64(scores[s] - mx)))
				sum += scores[s]
			}
			dst := out.Row(qi)[h*headDim : (h+1)*headDim]
			for s := range scores {
				p := scores[s] / sum
				vrow := v.Row(s)[h*headDim : (h+1)*headDim]
				for i, vv := range vrow {
					dst[i] += p * vv
				}
			}
		}
	})
	return out
}

func rows(segs []Segment) int {
	n := 0
	for _, s := range segs {
		n += s.Rows
	}
	return n
}

// eachRow calls fn for every row with its segment's modulation, in parallel.
func eachRow(segs []Segment, fn func(r int, mod Mod)) {
	start := 0
	for _, s := range segs {
		base, mod := start, s.Mod
		parallel(s.Rows, func(i int) { fn(base+i, mod) })
		start += s.Rows
	}
}

func stack(ms ...*qwen.Mat) *qwen.Mat {
	if len(ms) == 1 {
		return ms[0]
	}
	n := 0
	for _, m := range ms {
		n += m.Rows
	}
	out := qwen.NewMat(n, ms[0].Cols)
	at := 0
	for _, m := range ms {
		copy(out.Data[at:], m.Data)
		at += len(m.Data)
	}
	return out
}

// parallel runs fn(0..n-1) over GOMAXPROCS workers.
func parallel(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	var next int64
	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if int(i) >= n {
					return
				}
				fn(int(i))
			}
		}()
	}
	wg.Wait()
}
