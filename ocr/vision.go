package ocr

// The vision tower and projector on the CPU (OCR.md O2): the fp32 oracle the
// device port is debugged against, not a serving path. A full-bound crop is
// 5,120 patches and the attention here is scalar Go, so the gates run on the
// small cases.
//
// The blocks are qimage/vision's, unchanged: the same SigLIP-so400m widths
// (1152, 16 heads x 72, FFN 4304, gelu-tanh, LayerNorm eps 1e-6) and the same
// axial NeoX rope. What is this model's own, and each is a place the
// Qwen-VL tower next door would give plausible numbers:
//
//  1. **Raster token order.** Patches stay row-major through the tower; the
//     rope table and the position grid are indexed that way. The 2x2 merge
//     is a gather in the projector, not consecutive rows.
//  2. **A 27x27 position grid resized with align_corners=False**, the
//     authors' convention (OCR.md decision 3), where qimage/vision's is 48x48
//     with align_corners=True.
//  3. **Two LayerNorms in a row**: the tower's post_layernorm (eps 1e-6),
//     then the projector's pre_norm (eps 1e-5), both per 1152-wide row.
//  4. **Separate q/k/v projections with biases**, fused here at load into
//     the [3*1152, 1152] qkv qimage/vision's attention reads.
//  5. The projector's GELU is the exact erf one; the blocks' is tanh.

import (
	"fmt"
	"math"
	"path/filepath"

	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// The tower's shape (config.json vision_config).
const (
	VisionHidden = 1152
	VisionHeads  = 16
	VisionFFN    = 4304
	VisionDepth  = 27
	visionGrid   = 27 // sqrt of the 729 learned positions
	ropeTheta    = 10000.0
	projNormEps  = 1e-5
	// TextHidden is ERNIE's width, the projector's output.
	TextHidden = 1024
)

// Tower is the vision tower plus projector.
type Tower struct {
	PatchProj vision.Linear // Conv2d [1152, 3, 14, 14] as [1152, 588]
	PosEmbed  []float32     // [729, 1152]
	Blocks    []vision.Block
	PostLN    vision.LayerNorm
	ProjNorm  vision.LayerNorm // eps 1e-5, applied by layerNorm here
	FC1, FC2  vision.Linear
}

// LoadTower reads the tower and projector from dir/model.safetensors.
// blocks caps how many of the 27 load (0 is all), which is what lets a test
// walk the first ones without the rest; the post-LN and projector are only
// meaningful after all 27.
func LoadTower(dir string, blocks int) (*Tower, error) {
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if blocks <= 0 || blocks > VisionDepth {
		blocks = VisionDepth
	}
	var lerr error
	get := func(name string, want int) []float32 {
		if lerr != nil {
			return nil
		}
		t, err := f.Get(name)
		if err != nil {
			lerr = err
			return nil
		}
		v, err := t.F32(nil)
		if err != nil {
			lerr = err
			return nil
		}
		if len(v) != want {
			lerr = fmt.Errorf("ocr: %s has %d values, want %d", name, len(v), want)
		}
		return v
	}
	lin := func(name string, out, in int) vision.Linear {
		return vision.Linear{In: in, Out: out, Weight: get(name+".weight", out*in), Bias: get(name+".bias", out)}
	}
	norm := func(name string, w int) vision.LayerNorm {
		return vision.LayerNorm{Weight: get(name+".weight", w), Bias: get(name+".bias", w)}
	}
	const vp = "visual.vision_model."
	t := &Tower{
		PatchProj: lin(vp+"embeddings.patch_embedding", VisionHidden, PatchElems),
		PosEmbed:  get(vp+"embeddings.position_embedding.weight", visionGrid*visionGrid*VisionHidden),
	}
	for i := 0; i < blocks; i++ {
		p := fmt.Sprintf(vp+"encoder.layers.%d.", i)
		q, k, v := lin(p+"self_attn.q_proj", VisionHidden, VisionHidden),
			lin(p+"self_attn.k_proj", VisionHidden, VisionHidden),
			lin(p+"self_attn.v_proj", VisionHidden, VisionHidden)
		qkv := vision.Linear{In: VisionHidden, Out: 3 * VisionHidden}
		for _, l := range []vision.Linear{q, k, v} {
			qkv.Weight = append(qkv.Weight, l.Weight...)
			qkv.Bias = append(qkv.Bias, l.Bias...)
		}
		t.Blocks = append(t.Blocks, vision.Block{
			Norm1: norm(p+"layer_norm1", VisionHidden),
			Norm2: norm(p+"layer_norm2", VisionHidden),
			Attn:  vision.Attention{QKV: qkv, Proj: lin(p+"self_attn.out_proj", VisionHidden, VisionHidden), NumHeads: VisionHeads},
			MLP: vision.MLP{
				FC1: lin(p+"mlp.fc1", VisionFFN, VisionHidden),
				FC2: lin(p+"mlp.fc2", VisionHidden, VisionFFN),
				Act: geluTanh,
			},
		})
	}
	t.PostLN = norm(vp+"post_layernorm", VisionHidden)
	const wide = VisionHidden * MergeSize * MergeSize
	t.ProjNorm = norm("mlp_AR.pre_norm", VisionHidden)
	t.FC1 = lin("mlp_AR.linear_1", wide, wide)
	t.FC2 = lin("mlp_AR.linear_2", TextHidden, wide)
	if lerr != nil {
		return nil, lerr
	}
	return t, nil
}

func geluTanh(x float64) float64 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
}

func geluErf(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// positionEmbedding is F.interpolate(grid, (gh, gw), mode="bilinear",
// align_corners=False) of the 27x27 table, read out in raster order.
func (t *Tower) positionEmbedding(gh, gw int) *qwen.Mat {
	out := qwen.NewMat(gh*gw, VisionHidden)
	for r := 0; r < gh; r++ {
		y0, y1, fy := srcTaps(r, gh, visionGrid)
		for c := 0; c < gw; c++ {
			x0, x1, fx := srcTaps(c, gw, visionGrid)
			dst := out.Row(r*gw + c)
			for _, tap := range [4]struct {
				idx int
				w   float64
			}{
				{y0*visionGrid + x0, (1 - fy) * (1 - fx)},
				{y0*visionGrid + x1, (1 - fy) * fx},
				{y1*visionGrid + x0, fy * (1 - fx)},
				{y1*visionGrid + x1, fy * fx},
			} {
				if tap.w == 0 {
					continue
				}
				row := t.PosEmbed[tap.idx*VisionHidden : (tap.idx+1)*VisionHidden]
				for i, v := range row {
					dst[i] += float32(tap.w * float64(v))
				}
			}
		}
	}
	return out
}

// srcTaps is align_corners=False's source coordinate for destination i of n
// over a side of `side`: (i + 0.5)·side/n − 0.5, clamped below at 0, and
// its two taps and the far tap's weight.
func srcTaps(i, n, side int) (int, int, float64) {
	s := (float64(i)+0.5)*float64(side)/float64(n) - 0.5
	if s < 0 {
		s = 0
	}
	i0 := int(math.Floor(s))
	return i0, min(i0+1, side-1), s - float64(i0)
}

// rasterRope is the axial table in raster order: token (r, c) rotates by r
// in the first 18 frequencies of each half and by c in the second 18.
func rasterRope(gh, gw int) *vision.Rope {
	const hd = VisionHidden / VisionHeads // 72
	const spatial = hd / 2                // 36
	const nf = spatial / 2                // 18
	var inv [nf]float64
	for i := range inv {
		inv[i] = 1 / math.Pow(ropeTheta, float64(2*i)/float64(spatial))
	}
	cos, sin := qwen.NewMat(gh*gw, hd), qwen.NewMat(gh*gw, hd)
	for r := 0; r < gh; r++ {
		for c := 0; c < gw; c++ {
			cr, sr := cos.Row(r*gw+c), sin.Row(r*gw+c)
			for i, f := range inv {
				ah, aw := float64(r)*f, float64(c)*f
				for _, base := range [2]int{0, spatial} {
					cr[base+i], sr[base+i] = float32(math.Cos(ah)), float32(math.Sin(ah))
					cr[base+nf+i], sr[base+nf+i] = float32(math.Cos(aw)), float32(math.Sin(aw))
				}
			}
		}
	}
	return &vision.Rope{Cos: cos, Sin: sin}
}

// layerNorm is vision.LayerNorm with its own epsilon; the projector's is
// 1e-5 where the tower's is 1e-6.
func layerNorm(n *vision.LayerNorm, x *qwen.Mat, eps float64) *qwen.Mat {
	out := qwen.NewMat(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
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
			dst[i] = float32((float64(v)-mean)*inv*float64(n.Weight[i]) + float64(n.Bias[i]))
		}
	}
	return out
}

// Forward runs the tower and projector over one processed image and
// returns [MergedH*MergedW, 1024], the rows the image tokens take. tap, if
// set, sees each stage under the dump's names ("vis.embed", "vis.block{i}",
// "vis.post_ln", "proj").
func (t *Tower) Forward(im *Image, tap func(string, *qwen.Mat)) (*qwen.Mat, error) {
	gh, gw := im.GridH, im.GridW
	if gh%MergeSize != 0 || gw%MergeSize != 0 || len(im.Patches) != gh*gw*PatchElems {
		return nil, fmt.Errorf("ocr: %d values for a %dx%d patch grid", len(im.Patches), gw, gh)
	}
	px := &qwen.Mat{Rows: gh * gw, Cols: PatchElems, Data: im.Patches}
	h, err := t.PatchProj.Apply(px)
	if err != nil {
		return nil, err
	}
	pos := t.positionEmbedding(gh, gw)
	for i := range h.Data {
		h.Data[i] += pos.Data[i]
	}
	emit := func(name string, m *qwen.Mat) {
		if tap != nil {
			tap(name, m)
		}
	}
	emit("vis.embed", h)
	rope := rasterRope(gh, gw)
	for i := range t.Blocks {
		if h, err = t.Blocks[i].Forward(h, rope); err != nil {
			return nil, fmt.Errorf("ocr: vision block %d: %w", i, err)
		}
		emit(fmt.Sprintf("vis.block%d", i), h)
	}
	if len(t.Blocks) < VisionDepth {
		return nil, nil
	}
	if h, err = t.PostLN.Apply(h); err != nil {
		return nil, err
	}
	emit("vis.post_ln", h)
	if h, err = t.Project(h, gh, gw); err != nil {
		return nil, err
	}
	emit("proj", h)
	return h, nil
}

// Project is the projector alone, over the tower's post-LN rows of a
// gh x gw patch grid: pre_norm, the 2x2 gather, linear_1, GELU, linear_2.
func (t *Tower) Project(h *qwen.Mat, gh, gw int) (*qwen.Mat, error) {
	if h.Rows != gh*gw || h.Cols != VisionHidden {
		return nil, fmt.Errorf("ocr: projector given %dx%d for a %dx%d grid", h.Rows, h.Cols, gw, gh)
	}
	h = layerNorm(&t.ProjNorm, h, projNormEps)
	// Gather each 2x2 block: rows (r, c), (r, c+1), (r+1, c), (r+1, c+1),
	// which is the reference's reshape (h/2, 2, w/2, 2, d) -> transpose(2, 3).
	mh, mw := gh/MergeSize, gw/MergeSize
	const wide = VisionHidden * MergeSize * MergeSize
	cat := qwen.NewMat(mh*mw, wide)
	for r := 0; r < mh; r++ {
		for c := 0; c < mw; c++ {
			dst := cat.Row(r*mw + c)
			for dy := 0; dy < MergeSize; dy++ {
				for dx := 0; dx < MergeSize; dx++ {
					src := h.Row((2*r+dy)*gw + 2*c + dx)
					copy(dst[(dy*MergeSize+dx)*VisionHidden:], src)
				}
			}
		}
	}
	h, err := t.FC1.Apply(cat)
	if err != nil {
		return nil, err
	}
	for i, v := range h.Data {
		h.Data[i] = float32(geluErf(float64(v)))
	}
	return t.FC2.Apply(h)
}
