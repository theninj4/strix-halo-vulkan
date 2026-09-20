// Package dit implements Qwen-Image-2.1's single-stream diffusion
// transformer on the CPU: the oracle the Vulkan port (Q4) is debugged
// against, written the way every stage in IMAGE.md is — against the
// reference dumps, piece by piece, before any shader exists.
//
// The reference is reference/qwenimage21/transformer_qwenimage21.py, vendored
// at the diffusers commit the dumps came from. The shapes come from
// transformer/config.json; nothing below hardcodes a dimension.
//
// Where this model differs from the two transformers next door, which is
// where the porting mistakes live:
//
//   - One joint sequence. Text embeddings and image latents share the
//     stream: each VLM `<|image_pad|>` slot stands for a 2x2 group of latent
//     tokens, so slot positions expand four-fold and the latents are
//     scattered into them (overwriting the VLM's own slot rows — vision
//     content reaches this model only through the *text* tokens). The
//     target image's slots are appended after the text.
//   - Block-causal attention, decomposed. The mask is
//     `(q >= kv) OR same_image_block`; the reference's exact non-flex path
//     runs one attention call per prefix segment — text segments get a
//     causal triangle, image blocks attend bidirectionally over [0, end) —
//     plus one unmasked call for the target rows, and that decomposition is
//     what Forward reproduces.
//   - The t=0 modulation row. With causal_condition the timestep vector
//     carries an extra zero entry; text and condition-image tokens read the
//     t=0 row of every modulation tensor and target tokens read their own.
//     That is what makes the prefix step-independent, hence the KV cache:
//     ModeExtract stores each block's post-RoPE prefix K/V, ModeCached
//     recomputes only the target rows against [cached prefix ++ fresh
//     target].
//   - One shared modulation. A single SiLU→Linear(dim→4*dim) feeds every
//     block; blocks own no adaLN parameters and slice scale/gate out of the
//     same tensor. Gates pass through tanh. The final norm is scale-only.
//   - RoPE is 3-axis (frame, height, width; dims 16/56/56 of the 128-wide
//     head), theta 10000, *adjacent-pair* complex rotation — the DiT
//     convention, not the text encoder's NeoX halves. Text advances all
//     three axes together; an image block freezes the frame axis and lays
//     h/w on zero-centred grids with genuinely negative indices, and the
//     position cursor advances by max(h, w) past it.
//
// Layout is [tokens, features] row-major float32 with the batch axis
// squeezed, reusing zimage/qwen's Mat/Linear/RMSNorm — that package is the
// shared substrate (embed, llm and parakeet already build on it).
package dit

import (
	"fmt"
	"math"

	"strix-halo-vulkan/zimage/qwen"
)

// Mode is which of the denoising loop's three shapes a forward pass runs.
type Mode int

const (
	// ModePrefill runs the full joint sequence with block-causal attention.
	ModePrefill Mode = iota
	// ModeExtract is ModePrefill plus storing the prefix K/V per block.
	ModeExtract
	// ModeCached recomputes only the target rows; K/V is the cached prefix
	// concatenated with the fresh target rows, and no mask is needed — a
	// target row sees the whole prefix and its own block either way.
	ModeCached
)

// Segment is one run of equal image-block membership in the prefix:
// [Start, End) of the joint sequence, text or one image block.
type Segment struct {
	Start, End int
	Text       bool
}

// Layout is the joint sequence's geometry: everything the model derives from
// the VLM mask and the image shapes before any weight is touched.
type Layout struct {
	ImgShapes [][3]int // (frame, h, w) in latent tokens, target last
	VLMMask   []bool   // one entry per VLM slot/token, target slots appended
	PadMask   []bool   // VLMMask with slots expanded 4x: true at latent rows
	ImageIDs  []int    // -1 at text rows, else the image block's index
	Target    []bool   // true at the target image's rows
	PrefixLen int      // rows before the target block
	Segments  []Segment
}

// NewLayout builds the geometry from the text runs and image shapes:
// textLens[i] tokens precede image block i, textLens[len(shapes)-1] sit
// between the last condition image and the appended target slots. For t2i
// that is NewLayout([]int{promptTokens}, [][3]int{{1, h, w}}).
func NewLayout(textLens []int, imgShapes [][3]int) (*Layout, error) {
	if len(textLens) != len(imgShapes) {
		return nil, fmt.Errorf("dit: %d text runs for %d image blocks", len(textLens), len(imgShapes))
	}
	l := &Layout{ImgShapes: imgShapes}
	for b, sh := range imgShapes {
		tokens := sh[0] * sh[1] * sh[2]
		if tokens%4 != 0 {
			return nil, fmt.Errorf("dit: image block %d has %d tokens, not a multiple of 4 (2x2 slots)", b, tokens)
		}
		for i := 0; i < textLens[b]; i++ {
			l.VLMMask = append(l.VLMMask, false)
			l.PadMask = append(l.PadMask, false)
			l.ImageIDs = append(l.ImageIDs, -1)
		}
		for i := 0; i < tokens/4; i++ {
			l.VLMMask = append(l.VLMMask, true)
		}
		for i := 0; i < tokens; i++ {
			l.PadMask = append(l.PadMask, true)
			l.ImageIDs = append(l.ImageIDs, b)
		}
	}
	seq := len(l.PadMask)
	target := imgShapes[len(imgShapes)-1]
	l.PrefixLen = seq - target[0]*target[1]*target[2]
	l.Target = make([]bool, seq)
	for i := l.PrefixLen; i < seq; i++ {
		l.Target[i] = true
	}
	// Segments: runs of equal ImageIDs over the prefix. Two adjacent image
	// blocks with no text between them are separate runs because their ids
	// differ — the id, not the mask, is what keeps them from attending to
	// each other bidirectionally.
	for start := 0; start < l.PrefixLen; {
		end := start
		for end < l.PrefixLen && l.ImageIDs[end] == l.ImageIDs[start] {
			end++
		}
		l.Segments = append(l.Segments, Segment{Start: start, End: end, Text: l.ImageIDs[start] < 0})
		start = end
	}
	return l, nil
}

// LayoutFromPadMask builds the geometry from the text encoder's own
// image-pad mask, which is where an edit's text runs actually come from:
// the mask has one entry per prompt-embedding row, false at a text token and
// true at a slot standing for a 2x2 group of one condition image's latents.
// Runs of true are the condition images in order, and what is between them
// is the text.
//
// It generalises NewLayout rather than sitting beside it: a t2i prompt has
// no true entries, so condShapes is empty and this is exactly
// NewLayout([]int{len(mask)}, [][3]int{target}).
func LayoutFromPadMask(padMask []bool, condShapes [][3]int, target [3]int) (*Layout, error) {
	var textLens []int
	run, images := 0, 0
	for i := 0; i <= len(padMask); i++ {
		if i < len(padMask) && !padMask[i] {
			run++
			continue
		}
		if i == len(padMask) {
			textLens = append(textLens, run)
			break
		}
		// A run of slots: one condition image, whose length the shape has to
		// agree with. Four latent tokens to a slot.
		textLens = append(textLens, run)
		run = 0
		slots := 0
		for ; i < len(padMask) && padMask[i]; i++ {
			slots++
		}
		i--
		if images >= len(condShapes) {
			return nil, fmt.Errorf("dit: the pad mask holds %d image runs, got %d condition shapes",
				images+1, len(condShapes))
		}
		sh := condShapes[images]
		if want := sh[0] * sh[1] * sh[2] / 4; slots != want {
			return nil, fmt.Errorf("dit: image %d occupies %d slots, shape %v wants %d",
				images, slots, sh, want)
		}
		images++
	}
	if images != len(condShapes) {
		return nil, fmt.Errorf("dit: the pad mask holds %d image runs for %d condition shapes",
			images, len(condShapes))
	}
	return NewLayout(textLens, append(append([][3]int{}, condShapes...), target))
}

// Seq is the joint sequence length.
func (l *Layout) Seq() int { return len(l.PadMask) }

// Rope holds the 3-axis rotary table for one layout: cos and sin per token
// over the head's 64 complex pairs, adjacent-pair convention.
type Rope struct {
	Cos, Sin *qwen.Mat // [seq, headDim/2]
}

// NewRope builds the table. axesDims are the per-axis rotary widths (16, 56,
// 56 — their halves 8+28+28 fill the 64 pairs of a 128-wide head).
func NewRope(l *Layout, axesDims [3]int, headDim int, theta float64) (*Rope, error) {
	if axesDims[0]+axesDims[1]+axesDims[2] != headDim {
		return nil, fmt.Errorf("dit: rope axes %v do not fill head dim %d", axesDims, headDim)
	}
	seq := l.Seq()
	frame := make([]int, seq)
	height := make([]int, seq)
	width := make([]int, seq)

	pos, cursor := 0, 0
	for _, sh := range l.ImgShapes {
		h, w := sh[1], sh[2]
		blockStart := cursor
		for blockStart < seq && !l.PadMask[blockStart] {
			blockStart++
		}
		for i := cursor; i < blockStart; i++ {
			frame[i], height[i], width[i] = pos, pos, pos
			pos++
		}
		// The image block: frame frozen at the position the text reached,
		// h/w on zero-centred grids — [-(n - n/2), n/2) — and the cursor
		// advanced by max(h, w) so the next text does not collide with it.
		i := blockStart
		for hh := -(h - h/2); hh < h/2; hh++ {
			for ww := -(w - w/2); ww < w/2; ww++ {
				frame[i], height[i], width[i] = pos, hh, ww
				i++
			}
		}
		cursor = blockStart + h*w
		if h > w {
			pos += h
		} else {
			pos += w
		}
	}
	for i := cursor; i < seq; i++ {
		frame[i], height[i], width[i] = pos, pos, pos
		pos++
	}

	half := headDim / 2
	r := &Rope{Cos: qwen.NewMat(seq, half), Sin: qwen.NewMat(seq, half)}
	for t := 0; t < seq; t++ {
		cos, sin := r.Cos.Row(t), r.Sin.Row(t)
		at := 0
		for axis, dim := range axesDims {
			p := [3]int{frame[t], height[t], width[t]}[axis]
			for k := 0; k < dim/2; k++ {
				angle := float64(p) / math.Pow(theta, float64(2*k)/float64(dim))
				cos[at] = float32(math.Cos(angle))
				sin[at] = float32(math.Sin(angle))
				at++
			}
		}
	}
	return r, nil
}

// Slice returns the table's rows from `from` on — what ModeCached applies to
// the target rows.
func (r *Rope) Slice(from int) *Rope {
	return &Rope{
		Cos: &qwen.Mat{Rows: r.Cos.Rows - from, Cols: r.Cos.Cols, Data: r.Cos.Data[from*r.Cos.Cols:]},
		Sin: &qwen.Mat{Rows: r.Sin.Rows - from, Cols: r.Sin.Cols, Data: r.Sin.Data[from*r.Sin.Cols:]},
	}
}

// Apply rotates every head of every row in place, pairing *adjacent*
// components: (x[2j], x[2j+1]) is one complex number. The text encoder next
// door pairs by halves; mixing the two conventions leaves every norm intact
// and only the attention pattern wrong.
func (r *Rope) Apply(x *qwen.Mat, headDim int) error {
	half := headDim / 2
	if x.Cols%headDim != 0 {
		return fmt.Errorf("dit: rope of head dim %d does not divide %d columns", headDim, x.Cols)
	}
	if x.Rows > r.Cos.Rows {
		return fmt.Errorf("dit: rope table holds %d rows, need %d", r.Cos.Rows, x.Rows)
	}
	heads := x.Cols / headDim
	for t := 0; t < x.Rows; t++ {
		cos, sin := r.Cos.Row(t), r.Sin.Row(t)
		row := x.Row(t)
		for h := 0; h < heads; h++ {
			seg := row[h*headDim : (h+1)*headDim]
			for j := 0; j < half; j++ {
				re, im := seg[2*j], seg[2*j+1]
				seg[2*j] = re*cos[j] - im*sin[j]
				seg[2*j+1] = re*sin[j] + im*cos[j]
			}
		}
	}
	return nil
}

// layerNorm is x normalised over the row — mean subtracted, unlike every RMS
// norm in this model — with no learned parameters.
func layerNorm(x *qwen.Mat, eps float64) *qwen.Mat {
	out := qwen.NewMat(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
		row, dst := x.Row(r), out.Row(r)
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(len(row))
		var varsum float64
		for _, v := range row {
			d := float64(v) - mean
			varsum += d * d
		}
		scale := 1 / math.Sqrt(varsum/float64(len(row))+eps)
		for i, v := range row {
			dst[i] = float32((float64(v) - mean) * scale)
		}
	}
	return out
}

// ZeroRMSNorm is an RMS norm whose learned weight is stored zero-centred:
// the effective scale is weight+1. Only txt_in's text_norm uses it.
type ZeroRMSNorm struct {
	Weight []float32
	Eps    float64
}

// Apply returns a normalised copy.
func (n *ZeroRMSNorm) Apply(x *qwen.Mat) (*qwen.Mat, error) {
	if x.Cols != len(n.Weight) {
		return nil, fmt.Errorf("dit: zero-centred norm of width %d over %d columns", len(n.Weight), x.Cols)
	}
	out := qwen.NewMat(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
		row, dst := x.Row(r), out.Row(r)
		var sum float64
		for _, v := range row {
			sum += float64(v) * float64(v)
		}
		scale := 1 / math.Sqrt(sum/float64(x.Cols)+n.Eps)
		for i, v := range row {
			dst[i] = float32(float64(v) * scale * (float64(n.Weight[i]) + 1))
		}
	}
	return out, nil
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// geluTanh is the tanh approximation, txt_in's activation.
func geluTanh(x float32) float32 {
	x64 := float64(x)
	return float32(0.5 * x64 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x64+0.044715*x64*x64*x64))))
}

// modRow picks a token's modulation row: with causal_condition the tensor
// holds [the sampled timestep's row, the t=0 row], target tokens read the
// first and everything else the second. target is the token's Target bit.
func modRow(mod *qwen.Mat, target bool) []float32 {
	if target {
		return mod.Row(0)
	}
	return mod.Row(mod.Rows - 1)
}

// LayerCache is one block's prefix K and V, post-RoPE, [prefix, heads*headDim].
type LayerCache struct {
	K, V *qwen.Mat
}

// Cache holds every block's LayerCache, indexed by block.
type Cache []LayerCache

// NewCache sizes a cache for a model.
func NewCache(blocks int) Cache { return make(Cache, blocks) }

// Block is one single-stream layer. It owns no modulation parameters — the
// model's one shared modulation tensor is passed in and sliced here.
type Block struct {
	Q, K, V, O   *qwen.Linear
	QNorm, KNorm *qwen.RMSNorm // per head_dim
	GateL        *qwen.Linear  // img_mlp.gate_layer — the silu'd half
	Proj         *qwen.Linear  // img_mlp.proj — the linear half
	Out          *qwen.Linear  // img_mlp.out

	Heads, HeadDim int
	Eps            float64
}

// modulate is norm(x) * (1 + scale) with per-token row selection, returning
// the modulated tensor and the (tanh-less) gate rows to apply afterwards.
// modHalf is [rows, 2*dim]: scale then gate.
func (b *Block) modulate(normed *qwen.Mat, modHalf *qwen.Mat, target []bool, from int) *qwen.Mat {
	dim := normed.Cols
	out := qwen.NewMat(normed.Rows, dim)
	for r := 0; r < normed.Rows; r++ {
		mrow := modRow(modHalf, target[from+r])
		row, dst := normed.Row(r), out.Row(r)
		for i, v := range row {
			dst[i] = v * (1 + mrow[i])
		}
	}
	return out
}

func (b *Block) gateAdd(x, delta *qwen.Mat, modHalf *qwen.Mat, target []bool, from int) {
	dim := delta.Cols
	for r := 0; r < delta.Rows; r++ {
		gate := modRow(modHalf, target[from+r])[dim:]
		xrow, drow := x.Row(r), delta.Row(r)
		for i := range drow {
			xrow[i] += float32(math.Tanh(float64(gate[i]))) * drow[i]
		}
	}
}

// attend runs softmax(q kᵀ/√d)v for every head. q covers global rows
// [qStart, qStart+q.Rows); keys are k's first kLimit rows. causal true adds
// the text triangle: q's global row i attends keys 0..i.
func attend(q, k, v *qwen.Mat, heads, headDim, qStart, kLimit int, causal bool) *qwen.Mat {
	out := qwen.NewMat(q.Rows, heads*headDim)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	parallelHeads(heads, func(h int) {
		scores := make([]float32, kLimit)
		for qi := 0; qi < q.Rows; qi++ {
			limit := kLimit
			if causal && qStart+qi+1 < limit {
				limit = qStart + qi + 1
			}
			qrow := q.Row(qi)[h*headDim : (h+1)*headDim]
			max := float32(math.Inf(-1))
			for s := 0; s < limit; s++ {
				krow := k.Row(s)[h*headDim : (h+1)*headDim]
				var dot float32
				for i, qv := range qrow {
					dot += qv * krow[i]
				}
				scores[s] = dot * scale
				if scores[s] > max {
					max = scores[s]
				}
			}
			var sum float32
			for s := 0; s < limit; s++ {
				scores[s] = float32(math.Exp(float64(scores[s] - max)))
				sum += scores[s]
			}
			dst := out.Row(qi)[h*headDim : (h+1)*headDim]
			for s := 0; s < limit; s++ {
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

// Forward runs the block. x is the joint stream — the full sequence in
// ModePrefill/ModeExtract, the target rows alone in ModeCached. mod is the
// model's shared modulation, [2, 4*dim]. rope must cover x's rows (the
// caller slices it for ModeCached). cache is written by ModeExtract and read
// by ModeCached; nil otherwise.
func (b *Block) Forward(x *qwen.Mat, mod *qwen.Mat, rope *Rope, lay *Layout, mode Mode, cache *LayerCache) (*qwen.Mat, error) {
	dim := b.Heads * b.HeadDim
	if mod.Cols != 4*dim {
		return nil, fmt.Errorf("dit: modulation is %d wide, want %d", mod.Cols, 4*dim)
	}
	from := 0
	if mode == ModeCached {
		from = lay.PrefixLen
		if x.Rows != lay.Seq()-lay.PrefixLen {
			return nil, fmt.Errorf("dit: cached mode got %d rows, want the %d target rows", x.Rows, lay.Seq()-lay.PrefixLen)
		}
	} else if x.Rows != lay.Seq() {
		return nil, fmt.Errorf("dit: got %d rows for a %d-token layout", x.Rows, lay.Seq())
	}
	// chunk(2, dim=-1) splits each row, so both halves have to be gathered
	// row by row — rewrapping the backing array at half the width would hand
	// row 1 the second half of row 0.
	mod1 := qwen.NewMat(mod.Rows, 2*dim)
	mod2 := qwen.NewMat(mod.Rows, 2*dim)
	for r := 0; r < mod.Rows; r++ {
		copy(mod1.Row(r), mod.Row(r)[:2*dim])
		copy(mod2.Row(r), mod.Row(r)[2*dim:])
	}

	// ---- Attention, with its residual.
	modulated := b.modulate(layerNorm(x, b.Eps), mod1, lay.Target, from)
	q, err := b.Q.Apply(modulated)
	if err != nil {
		return nil, err
	}
	k, err := b.K.Apply(modulated)
	if err != nil {
		return nil, err
	}
	v, err := b.V.Apply(modulated)
	if err != nil {
		return nil, err
	}
	if err := b.QNorm.ApplyInPlace(q); err != nil {
		return nil, err
	}
	if err := b.KNorm.ApplyInPlace(k); err != nil {
		return nil, err
	}
	if err := rope.Apply(q, b.HeadDim); err != nil {
		return nil, err
	}
	if err := rope.Apply(k, b.HeadDim); err != nil {
		return nil, err
	}

	var ctx *qwen.Mat
	switch mode {
	case ModeCached:
		if cache == nil || cache.K == nil {
			return nil, fmt.Errorf("dit: cached mode with an empty cache")
		}
		kAll := concatRows(cache.K, k)
		vAll := concatRows(cache.V, v)
		ctx = attend(q, kAll, vAll, b.Heads, b.HeadDim, 0, kAll.Rows, false)
	default:
		if mode == ModeExtract {
			if cache == nil {
				return nil, fmt.Errorf("dit: extract mode with nowhere to store")
			}
			cache.K = cloneRows(k, 0, lay.PrefixLen)
			cache.V = cloneRows(v, 0, lay.PrefixLen)
		}
		// The prefix, one attention call per segment: keys are everything up
		// to the segment's end, text segments causal over their own span.
		parts := make([]*qwen.Mat, 0, len(lay.Segments)+1)
		for _, s := range lay.Segments {
			qs := &qwen.Mat{Rows: s.End - s.Start, Cols: q.Cols, Data: q.Data[s.Start*q.Cols : s.End*q.Cols]}
			parts = append(parts, attend(qs, k, v, b.Heads, b.HeadDim, s.Start, s.End, s.Text))
		}
		// And the target rows, unmasked over the whole sequence.
		qt := &qwen.Mat{Rows: lay.Seq() - lay.PrefixLen, Cols: q.Cols, Data: q.Data[lay.PrefixLen*q.Cols:]}
		parts = append(parts, attend(qt, k, v, b.Heads, b.HeadDim, 0, lay.Seq(), false))
		ctx = parts[0]
		for _, p := range parts[1:] {
			ctx = concatRows(ctx, p)
		}
	}

	attnOut, err := b.O.Apply(ctx)
	if err != nil {
		return nil, err
	}
	resid := x.Clone()
	b.gateAdd(resid, attnOut, mod1, lay.Target, from)

	// ---- Feed forward, with its residual. proj is the plain half and
	// gate_layer the silu'd one: out(silu(gate_layer(x)) * proj(x)).
	modulated2 := b.modulate(layerNorm(resid, b.Eps), mod2, lay.Target, from)
	gate, err := b.GateL.Apply(modulated2)
	if err != nil {
		return nil, err
	}
	up, err := b.Proj.Apply(modulated2)
	if err != nil {
		return nil, err
	}
	for i := range gate.Data {
		gate.Data[i] = silu(gate.Data[i]) * up.Data[i]
	}
	mlp, err := b.Out.Apply(gate)
	if err != nil {
		return nil, err
	}
	b.gateAdd(resid, mlp, mod2, lay.Target, from)
	return resid, nil
}

func concatRows(a, b *qwen.Mat) *qwen.Mat {
	out := qwen.NewMat(a.Rows+b.Rows, a.Cols)
	copy(out.Data, a.Data)
	copy(out.Data[a.Rows*a.Cols:], b.Data)
	return out
}

func cloneRows(m *qwen.Mat, from, to int) *qwen.Mat {
	out := qwen.NewMat(to-from, m.Cols)
	copy(out.Data, m.Data[from*m.Cols:to*m.Cols])
	return out
}

// parallelHeads fans a per-head loop out over the cores.
func parallelHeads(n int, fn func(int)) {
	done := make(chan struct{}, n)
	for h := 0; h < n; h++ {
		go func(h int) {
			fn(h)
			done <- struct{}{}
		}(h)
	}
	for i := 0; i < n; i++ {
		<-done
	}
}

// Model is the transformer: the head that assembles the joint sequence, the
// shared modulation, the blocks, and the scale-only final norm.
type Model struct {
	Cfg *Config

	TimeL1, TimeL2 *qwen.Linear // timestep_embedder linear_1/linear_2
	TxtNorm        *ZeroRMSNorm
	TxtIn, TxtOut  *qwen.Linear // txt_in in_layer/out_layer
	ImgIn          *qwen.Linear
	Modulation     *qwen.Linear // dim -> 4*dim, after silu
	NormOut        *qwen.Linear // scale-only adaLN, dim -> dim
	ProjOut        *qwen.Linear
	Blocks         []*Block
}

// TimeEmbed is the timestep embedding for t in [0, 1] (the pipeline's
// timestep/1000): sinusoidal 256 with cos in the first half, then the
// two-linear MLP. With causal_condition the returned Mat has two rows —
// [t, 0] — and every modulation consumer row-selects from it.
func (m *Model) TimeEmbed(t float64) (*qwen.Mat, error) {
	const timestepDim = 256
	half := timestepDim / 2
	proj := qwen.NewMat(2, timestepDim)
	for r, tv := range []float64{t, 0} {
		row := proj.Row(r)
		for i := 0; i < half; i++ {
			freq := math.Exp(-math.Log(10000) * float64(i) / float64(half))
			angle := tv * 1000 * freq
			row[i] = float32(math.Cos(angle))
			row[i+half] = float32(math.Sin(angle))
		}
	}
	h, err := m.TimeL1.Apply(proj)
	if err != nil {
		return nil, err
	}
	for i := range h.Data {
		h.Data[i] = silu(h.Data[i])
	}
	return m.TimeL2.Apply(h)
}

// TxtProject is txt_in: zero-centred RMS norm, linear, GELU(tanh), linear.
func (m *Model) TxtProject(txt *qwen.Mat) (*qwen.Mat, error) {
	normed, err := m.TxtNorm.Apply(txt)
	if err != nil {
		return nil, err
	}
	h, err := m.TxtIn.Apply(normed)
	if err != nil {
		return nil, err
	}
	for i := range h.Data {
		h.Data[i] = geluTanh(h.Data[i])
	}
	return m.TxtOut.Apply(h)
}

// Assemble builds the joint stream from the projected text (one row per VLM
// token — condition-image slot rows included, because the VLM emitted rows
// for them) and the projected latents (condition images first, target last).
// Text rows pass through; each slot becomes its 2x2 group's four latent
// rows. A condition slot's text row is never read: the reference builds the
// joint stream from the text and then scatters the latents *over* the slot
// positions, so the VLM's vision rows are discarded by construction.
func (m *Model) Assemble(txt, img *qwen.Mat, lay *Layout) (*qwen.Mat, error) {
	target := lay.ImgShapes[len(lay.ImgShapes)-1]
	targetSlots := target[0] * target[1] * target[2] / 4
	vlmLen := len(lay.VLMMask) - targetSlots
	if txt.Rows != vlmLen {
		return nil, fmt.Errorf("dit: %d text rows for a layout with %d VLM tokens", txt.Rows, vlmLen)
	}
	slots := 0
	for _, s := range lay.VLMMask {
		if s {
			slots++
		}
	}
	if img.Rows != slots*4 {
		return nil, fmt.Errorf("dit: %d latent rows for %d slots", img.Rows, slots)
	}
	out := qwen.NewMat(lay.Seq(), txt.Cols)
	outAt, imgAt := 0, 0
	for i, slot := range lay.VLMMask {
		if slot {
			for j := 0; j < 4; j++ {
				copy(out.Row(outAt), img.Row(imgAt))
				outAt, imgAt = outAt+1, imgAt+1
			}
			continue
		}
		// Only entries below vlmLen can be text — the appended target slots
		// are all true — so txt.Row(i) is in range.
		copy(out.Row(outAt), txt.Row(i))
		outAt++
	}
	return out, nil
}

// Forward runs the model. latents is the packed latent stream [tokens,
// in_channels] (condition images then target, always the full set, every
// mode); txt is the raw VLM embedding [vlm rows, context_in_dim]; t is the
// pipeline's timestep/1000. The output is [rows, out_channels] — the full
// sequence in prefill (callers slice the target), target rows in ModeCached.
func (m *Model) Forward(latents, txt *qwen.Mat, lay *Layout, t float64, mode Mode, cache Cache) (*qwen.Mat, error) {
	if cache != nil && len(cache) != len(m.Blocks) {
		return nil, fmt.Errorf("dit: cache of %d layers for %d blocks", len(cache), len(m.Blocks))
	}
	img, err := m.ImgIn.Apply(latents)
	if err != nil {
		return nil, err
	}
	txtP, err := m.TxtProject(txt)
	if err != nil {
		return nil, err
	}
	x, err := m.Assemble(txtP, img, lay)
	if err != nil {
		return nil, err
	}
	rope, err := NewRope(lay, m.Cfg.AxesDims, m.Cfg.HeadDim, 10000)
	if err != nil {
		return nil, err
	}
	temb, err := m.TimeEmbed(t)
	if err != nil {
		return nil, err
	}
	// Both the shared modulation and the final norm read silu(temb); the
	// silu lives outside the linears in the reference (nn.Sequential(SiLU,
	// Linear) and AdaLayerNormContinuous's silu-then-linear).
	smod := temb.Clone()
	for i := range smod.Data {
		smod.Data[i] = silu(smod.Data[i])
	}
	mod, err := m.Modulation.Apply(smod)
	if err != nil {
		return nil, err
	}

	if mode == ModeCached {
		x = cloneRows(x, lay.PrefixLen, lay.Seq())
		rope = rope.Slice(lay.PrefixLen)
	}
	for i, b := range m.Blocks {
		var lc *LayerCache
		if cache != nil {
			lc = &cache[i]
		}
		x, err = b.Forward(x, mod, rope, lay, mode, lc)
		if err != nil {
			return nil, fmt.Errorf("dit: block %d: %w", i, err)
		}
	}

	// The tail: scale-only adaLN over the same [t, 0] rows, then proj_out.
	scaleIn, err := m.NormOut.Apply(smod)
	if err != nil {
		return nil, err
	}
	from := 0
	if mode == ModeCached {
		from = lay.PrefixLen
	}
	normed := layerNorm(x, m.Cfg.Eps)
	for r := 0; r < normed.Rows; r++ {
		srow := modRow(scaleIn, lay.Target[from+r])
		row := normed.Row(r)
		for i := range row {
			row[i] *= 1 + srow[i]
		}
	}
	return m.ProjOut.Apply(normed)
}
