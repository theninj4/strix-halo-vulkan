// Package vision implements Qwen-Image-2.1's vision tower on the CPU — the
// 27-layer ViT inside the Qwen3-VL text encoder that turns a condition
// image into the context an *edit* is conditioned on (IMAGE.md Q8).
//
// It is the largest genuinely new port of the Z-Image migration, and the
// thing to understand before reading it is what its output is for. The DiT
// overwrites every image-slot row of the joint sequence with VAE latents,
// so **none of this tensor reaches the transformer directly**. What reaches
// it is whatever the surrounding *text* tokens absorbed from these rows
// inside the text encoder's own attention. The tower conditions the text;
// the VAE conditions the pixels. That is why an edit needs both and why
// neither substitutes for the other.
//
// Five things here are not what a plain ViT would do, and each is a place a
// plausible-looking port gives a wrong answer:
//
//  1. **Token order is 2x2-block-major, not raster.** Patches are laid out
//     so that every four consecutive rows are one spatial merge block, which
//     is what lets the merger concatenate rows rather than gather. The
//     position ids carry that order (blockOrder below) and the packing of
//     pixels has to agree with it.
//  2. **The position embedding is interpolated.** The checkpoint holds
//     48x48 = 2304 learned positions and any image grid is a bilinear
//     resample of them — four (index, weight) pairs per patch, align_corners
//     true. Nothing about the grid is stored; it is recomputed per image.
//  3. **Two different GELUs.** The block MLP uses the tanh approximation
//     (config hidden_act is gelu_pytorch_tanh); both mergers use nn.GELU(),
//     which is the exact erf one. They differ by ~1e-3 at the knee, which is
//     small enough to look like noise and large enough to matter after 27
//     layers.
//  4. **Two different mergers.** The output merger normalizes each 1152-wide
//     row and *then* concatenates four of them; the three deepstack mergers
//     concatenate first and normalize the 4608-wide result
//     (use_postshuffle_norm). Their LayerNorms are different widths in the
//     checkpoint, so loading proves which is which — but running the wrong
//     one produces numbers, not an error.
//  5. **Axial rope over halves.** cos/sin are built from 18 inverse
//     frequencies per axis, concatenated h-then-w to 36 and duplicated to
//     the 72-wide head, and the rotation is the NeoX halves convention —
//     *not* the adjacent-pair complex rotation the DiT uses. The two live in
//     the same repository and are not interchangeable.
//
// Everything is gated stage by stage against reference/out/qi21vision.
package vision

import (
	"fmt"
	"math"

	"strix-halo-vulkan/zimage/qwen"
)

// Config mirrors the vision half of text_encoder/config.json.
type Config struct {
	Depth                 int    `json:"depth"`
	HiddenSize            int    `json:"hidden_size"`
	NumHeads              int    `json:"num_heads"`
	IntermediateSize      int    `json:"intermediate_size"`
	HiddenAct             string `json:"hidden_act"`
	PatchSize             int    `json:"patch_size"`
	SpatialMergeSize      int    `json:"spatial_merge_size"`
	TemporalPatchSize     int    `json:"temporal_patch_size"`
	InChannels            int    `json:"in_channels"`
	NumPositionEmbeddings int    `json:"num_position_embeddings"`
	OutHiddenSize         int    `json:"out_hidden_size"`
	DeepstackIndexes      []int  `json:"deepstack_visual_indexes"`
}

// HeadDim is the attention head width, 72 here.
func (c *Config) HeadDim() int { return c.HiddenSize / c.NumHeads }

// GridPerSide is the side of the learned position grid, 48 here.
func (c *Config) GridPerSide() int {
	return int(math.Round(math.Sqrt(float64(c.NumPositionEmbeddings))))
}

// PatchElems is how many values one patch of pixel_values carries:
// channels x temporal x patch x patch, 1536 here.
func (c *Config) PatchElems() int {
	return c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
}

// normEps is every LayerNorm's epsilon in the tower.
const normEps = 1e-6

// Linear is a row-major [Out, In] weight and an optional bias.
type Linear struct {
	In, Out int
	Weight  []float32 // [Out, In]
	Bias    []float32 // [Out] or nil

	// FP16 rounds both operands to half precision and accumulates in
	// float32, which is what a matrix-core GEMM does. It is off by default
	// and Model.SetFP16 is its only caller: this port is the fp32 oracle,
	// and the flag exists so the *cost* of the device's precision can be
	// measured before a device port is written to pay it. See
	// TestFP16Ladder.
	FP16 bool
}

// Apply computes x W^T + b over the rows of x.
func (l *Linear) Apply(x *qwen.Mat) (*qwen.Mat, error) {
	if x.Cols != l.In {
		return nil, fmt.Errorf("vision: linear %d->%d applied to %d columns", l.In, l.Out, x.Cols)
	}
	out := qwen.NewMat(x.Rows, l.Out)
	for r := 0; r < x.Rows; r++ {
		row := x.Row(r)
		dst := out.Row(r)
		if l.FP16 {
			for o := 0; o < l.Out; o++ {
				w := l.Weight[o*l.In : (o+1)*l.In]
				var acc float32
				for i, v := range row {
					acc += FP16(v) * FP16(w[i])
				}
				if l.Bias != nil {
					acc += l.Bias[o]
				}
				dst[o] = acc
			}
			continue
		}
		for o := 0; o < l.Out; o++ {
			w := l.Weight[o*l.In : (o+1)*l.In]
			var acc float64
			for i, v := range row {
				acc += float64(v) * float64(w[i])
			}
			if l.Bias != nil {
				acc += float64(l.Bias[o])
			}
			dst[o] = float32(acc)
		}
	}
	return out, nil
}

// FP16 rounds a float32 to the nearest representable half and back, ties to
// even — the rounding a value takes on when it is handed to a matrix core as
// an fp16 operand. Values past half's 65504 saturate to infinity, which is
// exactly the failure this tower has to be checked for: its residual stream
// reaches absmax 1.4e4 by the last hidden state.
func FP16(f float32) float32 {
	bits := math.Float32bits(f)
	sign := bits & 0x80000000
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	switch {
	case exp >= 0x1F: // overflow, or an input that was already inf/NaN
		return math.Float32frombits(sign | 0x7F800000)
	case exp <= 0: // subnormal in half, which this model never reaches
		return math.Float32frombits(sign)
	}
	half := mant >> 13
	if rem := mant & 0x1FFF; rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
		half++
		if half == 0x400 {
			half, exp = 0, exp+1
			if exp >= 0x1F {
				return math.Float32frombits(sign | 0x7F800000)
			}
		}
	}
	return math.Float32frombits(sign | uint32(exp+127-15)<<23 | half<<13)
}

// LayerNorm is the affine, mean-subtracting norm the tower uses everywhere.
type LayerNorm struct {
	Weight, Bias []float32
}

// Apply normalizes each row in place-safe fashion, returning a fresh matrix.
func (n *LayerNorm) Apply(x *qwen.Mat) (*qwen.Mat, error) {
	if len(n.Weight) != x.Cols {
		return nil, fmt.Errorf("vision: layer norm of width %d over %d columns", len(n.Weight), x.Cols)
	}
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
		inv := 1 / math.Sqrt(varsum/float64(len(row))+normEps)
		for i, v := range row {
			dst[i] = float32((float64(v)-mean)*inv*float64(n.Weight[i]) + float64(n.Bias[i]))
		}
	}
	return out, nil
}

// geluTanh is the block MLP's activation (config hidden_act).
func geluTanh(x float64) float64 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
}

// geluErf is nn.GELU()'s, which is what both mergers use.
func geluErf(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

func applyAct(m *qwen.Mat, f func(float64) float64) {
	for i, v := range m.Data {
		m.Data[i] = float32(f(float64(v)))
	}
}

// MLP is the block's feed-forward: fc1, GELU, fc2.
//
// Act is a field rather than a constant because the tower uses *two*
// different GELUs -- the tanh approximation here and the exact erf one in
// the mergers -- and which is which is a property of the checkpoint that
// deserves to be visible rather than buried in a call. Load sets it; the
// negative control in vision_test.go swaps it.
type MLP struct {
	FC1, FC2 Linear
	Act      func(float64) float64
}

func (m *MLP) Forward(x *qwen.Mat) (*qwen.Mat, error) {
	h, err := m.FC1.Apply(x)
	if err != nil {
		return nil, err
	}
	applyAct(h, m.Act)
	return m.FC2.Apply(h)
}

// Attention is the block's full (unmasked) self-attention over the image's
// patches. One condition image is one sequence, so the reference's
// cu_seqlens chunking degenerates to a single chunk and there is no mask.
type Attention struct {
	QKV, Proj Linear
	NumHeads  int
}

func (a *Attention) Forward(x *qwen.Mat, rope *Rope) (*qwen.Mat, error) {
	n, dim := x.Rows, x.Cols
	hd := dim / a.NumHeads
	qkv, err := a.QKV.Apply(x)
	if err != nil {
		return nil, err
	}
	// qkv rows are [3, heads, headDim] — the reference reshapes to
	// (seq, 3, heads, -1) and permutes, so q is the first `dim` values of
	// each row, k the second and v the third.
	q := qwen.NewMat(n, dim)
	k := qwen.NewMat(n, dim)
	v := qwen.NewMat(n, dim)
	for r := 0; r < n; r++ {
		row := qkv.Row(r)
		copy(q.Row(r), row[0:dim])
		copy(k.Row(r), row[dim:2*dim])
		copy(v.Row(r), row[2*dim:3*dim])
	}
	rope.Apply(q, a.NumHeads, hd)
	rope.Apply(k, a.NumHeads, hd)

	ctx := qwen.NewMat(n, dim)
	scale := 1 / math.Sqrt(float64(hd))
	scores := make([]float64, n)
	for h := 0; h < a.NumHeads; h++ {
		off := h * hd
		for i := 0; i < n; i++ {
			qi := q.Row(i)[off : off+hd]
			max := math.Inf(-1)
			for j := 0; j < n; j++ {
				kj := k.Row(j)[off : off+hd]
				var dot float64
				for d := 0; d < hd; d++ {
					dot += float64(qi[d]) * float64(kj[d])
				}
				scores[j] = dot * scale
				if scores[j] > max {
					max = scores[j]
				}
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - max)
				sum += scores[j]
			}
			dst := ctx.Row(i)[off : off+hd]
			for d := 0; d < hd; d++ {
				var acc float64
				for j := 0; j < n; j++ {
					acc += scores[j] * float64(v.Row(j)[off+d])
				}
				dst[d] = float32(acc / sum)
			}
		}
	}
	return a.Proj.Apply(ctx)
}

// Block is one of the 27: pre-norm attention and pre-norm MLP, each with a
// residual.
type Block struct {
	Norm1, Norm2 LayerNorm
	Attn         Attention
	MLP          MLP
}

func (b *Block) Forward(x *qwen.Mat, rope *Rope) (*qwen.Mat, error) {
	h, err := b.Norm1.Apply(x)
	if err != nil {
		return nil, err
	}
	if h, err = b.Attn.Forward(h, rope); err != nil {
		return nil, err
	}
	addInto(x, h)
	if h, err = b.Norm2.Apply(x); err != nil {
		return nil, err
	}
	if h, err = b.MLP.Forward(h); err != nil {
		return nil, err
	}
	addInto(x, h)
	return x, nil
}

func addInto(dst, src *qwen.Mat) {
	for i := range dst.Data {
		dst.Data[i] += src.Data[i]
	}
}

// Merger turns `merge` consecutive tower rows into one 4096-wide row.
//
// PostShuffleNorm is the whole difference between the output merger and the
// three deepstack ones, and it moves the LayerNorm across the concatenation:
// false normalizes each 1152-wide row first, true normalizes the 4608-wide
// concatenation. The checkpoint's norm widths say which a given merger is.
type Merger struct {
	PostShuffleNorm bool
	Merge           int // rows per output row, 4 here
	Norm            LayerNorm
	FC1, FC2        Linear
	Act             func(float64) float64
}

func (m *Merger) Forward(x *qwen.Mat) (*qwen.Mat, error) {
	if x.Rows%m.Merge != 0 {
		return nil, fmt.Errorf("vision: %d rows do not group into %ds", x.Rows, m.Merge)
	}
	var shuffled *qwen.Mat
	if m.PostShuffleNorm {
		shuffled = reshape(x, m.Merge)
		normed, err := m.Norm.Apply(shuffled)
		if err != nil {
			return nil, err
		}
		shuffled = normed
	} else {
		normed, err := m.Norm.Apply(x)
		if err != nil {
			return nil, err
		}
		shuffled = reshape(normed, m.Merge)
	}
	h, err := m.FC1.Apply(shuffled)
	if err != nil {
		return nil, err
	}
	applyAct(h, m.Act)
	return m.FC2.Apply(h)
}

// reshape concatenates every `merge` consecutive rows into one. It is a
// view in torch and a copy here; at 256 rows of 1152 that is 1 MB.
func reshape(x *qwen.Mat, merge int) *qwen.Mat {
	out := qwen.NewMat(x.Rows/merge, x.Cols*merge)
	copy(out.Data, x.Data)
	return out
}

// Model is the whole tower.
type Model struct {
	Cfg Config
	// PatchProj is the Conv3d patch embedding, which at one patch per
	// window is exactly a linear over the 1536 flattened values.
	PatchProj Linear
	// PosEmbed is the [2304, 1152] learned position grid.
	PosEmbed  []float32
	Blocks    []Block
	Deepstack []Merger
	Merger    Merger
	// RopePairs forces the rotation convention qimage/dit uses, which is
	// wrong here. See Rope.Pairs: it exists so the gate can be shown to
	// catch the swap.
	RopePairs bool
}

// SetFP16 switches every linear in the tower between the fp32 oracle and the
// matrix core's arithmetic — fp16 operands, fp32 accumulation. It is the
// instrument that prices a device port rather than a mode anything serves.
func (m *Model) SetFP16(on bool) {
	m.PatchProj.FP16 = on
	for i := range m.Blocks {
		b := &m.Blocks[i]
		b.Attn.QKV.FP16 = on
		b.Attn.Proj.FP16 = on
		b.MLP.FC1.FP16 = on
		b.MLP.FC2.FP16 = on
	}
	for i := range m.Deepstack {
		m.Deepstack[i].FC1.FP16 = on
		m.Deepstack[i].FC2.FP16 = on
	}
	m.Merger.FC1.FP16 = on
	m.Merger.FC2.FP16 = on
}

// Output is what one condition image becomes.
type Output struct {
	// Merged is [tokens, 4096] — one row per 2x2 patch group, which is what
	// is scattered into the `<|image_pad|>` slots of the text sequence.
	Merged *qwen.Mat
	// Deepstack is three [tokens, 4096] tensors, added into the text
	// model's first three layers at the visual positions.
	Deepstack []*qwen.Mat
	// Last is the tower's final hidden state, before the merger. Kept
	// because it is what the stagewise gate compares.
	Last *qwen.Mat
}

// Forward runs the tower over one image's patches.
//
// pixels is [patches, PatchElems] in the processor's own layout and order —
// see Patchify, which produces it from an image and is gated against the
// dump's pixel_values.
func (m *Model) Forward(pixels *qwen.Mat, gridH, gridW int, tap func(name string, x *qwen.Mat)) (*Output, error) {
	if pixels.Cols != m.Cfg.PatchElems() {
		return nil, fmt.Errorf("vision: pixel rows are %d wide, want %d", pixels.Cols, m.Cfg.PatchElems())
	}
	if pixels.Rows != gridH*gridW {
		return nil, fmt.Errorf("vision: %d patches for a %dx%d grid", pixels.Rows, gridH, gridW)
	}
	h, err := m.PatchProj.Apply(pixels)
	if err != nil {
		return nil, err
	}
	if tap != nil {
		tap("patch_embed", h)
	}

	pos, err := m.positionEmbedding(gridH, gridW)
	if err != nil {
		return nil, err
	}
	if tap != nil {
		tap("pos_embeds", pos)
	}
	addInto(h, pos)
	if tap != nil {
		tap("block_input", h)
	}

	rope := NewRope(&m.Cfg, gridH, gridW)
	rope.Pairs = m.RopePairs
	deep := make([]*qwen.Mat, 0, len(m.Deepstack))
	for i := range m.Blocks {
		if h, err = m.Blocks[i].Forward(h, rope); err != nil {
			return nil, fmt.Errorf("vision: block %d: %w", i, err)
		}
		if tap != nil {
			tap(fmt.Sprintf("block%d_out", i), h)
		}
		for j, idx := range m.Cfg.DeepstackIndexes {
			if idx != i {
				continue
			}
			f, err := m.Deepstack[j].Forward(h)
			if err != nil {
				return nil, fmt.Errorf("vision: deepstack merger %d: %w", j, err)
			}
			deep = append(deep, f)
		}
	}
	if tap != nil {
		tap("last_hidden", h)
	}
	merged, err := m.Merger.Forward(h)
	if err != nil {
		return nil, err
	}
	return &Output{Merged: merged, Deepstack: deep, Last: h}, nil
}
