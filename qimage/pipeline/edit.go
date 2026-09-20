package pipeline

// The condition-image half of an edit — IMAGE.md Q8.3.
//
// A generation's prefix is a prompt. An edit's is a prompt *and* one to ten
// reference images, and each of those is encoded twice: by the vision tower
// into the context the text tokens absorb (qimage/textenc's edit path), and
// by the VAE into latents the transformer attends over directly. This file
// is the second half — the geometry a condition image is resized to, and the
// packing that turns its latent grid into the `cond` rows Denoise prepends to
// the target at every step.
//
// One resize feeds both encoders, which is the fact that keeps the two
// halves consistent: the vision tower's patch grid and the VAE's latent grid
// are the *same numbers* (a 16x16 pixel tile is one latent token and one
// vision patch, and four of either make one VLM slot), so a condition image
// has one geometry and not two.

import (
	"fmt"
	"image"
	"math"
	"time"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/qimage/textenc"
	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/zimage/qwen"
	zvae "strix-halo-vulkan/zimage/vae"
)

// CondMultiple is the pixel granularity every condition image is snapped to.
// It is the VAE's 16 times the VLM's 2x2 grouping: below it a latent grid
// cannot be cut into whole slots.
const CondMultiple = 32

// CalcDimensions is diffusers' calculate_dimensions: the pixel width and
// height a condition image is resized to, preserving its aspect ratio at a
// target area and snapped to CondMultiple.
//
// The rounding is Python's, which breaks halves to even rather than away
// from zero — math.RoundToEven and not math.Round. It only bites on exact
// halves, which at this granularity means a source whose scaled side lands
// on a multiple of 16 that is not one of 32; rare, and free to get right.
func CalcDimensions(targetArea int, ratio float64) (int, int) {
	w := math.Sqrt(float64(targetArea) * ratio)
	h := w / ratio
	return int(math.RoundToEven(w/CondMultiple)) * CondMultiple,
		int(math.RoundToEven(h/CondMultiple)) * CondMultiple
}

// PackLatents flattens a [1, C, h, w] latent grid into one row per token, in
// raster order — diffusers' _pack_latents, which for 2.1 is a plain spatial
// flatten because the DiT takes latents unpatched.
func PackLatents(z *zvae.Tensor) *qwen.Mat {
	out := qwen.NewMat(z.H*z.W, z.C)
	for c := 0; c < z.C; c++ {
		plane := z.Plane(0, c)
		for i, v := range plane {
			out.Row(i)[c] = v
		}
	}
	return out
}

// UnpackLatents is the inverse, back to a [1, C, h, w] grid.
func UnpackLatents(m *qwen.Mat, h, w int) (*zvae.Tensor, error) {
	if m.Rows != h*w {
		return nil, fmt.Errorf("pipeline: %d latent rows for a %dx%d grid", m.Rows, h, w)
	}
	z := zvae.NewTensor(1, m.Cols, h, w)
	for c := 0; c < m.Cols; c++ {
		plane := z.Plane(0, c)
		for i := 0; i < m.Rows; i++ {
			plane[i] = m.Row(i)[c]
		}
	}
	return z, nil
}

// EncodeCondition turns one condition image into the rows the denoiser
// prepends: the VAE's posterior *mode* (never a sample — IMAGE.md decision
// 3, and it is what keeps a seeded edit reproducible), normalised per
// channel into the DiT's space, and packed.
//
// img is [1, 4, H, W] in [-1, 1] — all four channels, alpha included. The
// vision tower reads a copy of the same image flattened over white; this one
// is not flattened, and a port that flattens once and uses it twice is
// wrong in a way that only shows on transparent references.
func EncodeCondition(enc *qvae.Encoder, cfg *qvae.Config, img *zvae.Tensor) (*qwen.Mat, [3]int, error) {
	if img.H%CondMultiple != 0 || img.W%CondMultiple != 0 {
		return nil, [3]int{}, fmt.Errorf("pipeline: a %dx%d condition image is not a multiple of %d",
			img.W, img.H, CondMultiple)
	}
	z, err := enc.Encode(img)
	if err != nil {
		return nil, [3]int{}, fmt.Errorf("pipeline: encoding a condition image: %w", err)
	}
	cfg.Normalize(z)
	return PackLatents(z), [3]int{1, z.H, z.W}, nil
}

// ConcatConditions stacks several condition images' latents into the single
// `cond` matrix Denoise takes, in the order their runs appear in the prompt.
func ConcatConditions(conds []*qwen.Mat) (*qwen.Mat, error) {
	if len(conds) == 0 {
		return nil, nil
	}
	rows, cols := 0, conds[0].Cols
	for i, c := range conds {
		if c.Cols != cols {
			return nil, fmt.Errorf("pipeline: condition %d is %d wide, condition 0 is %d", i, c.Cols, cols)
		}
		rows += c.Rows
	}
	out := qwen.NewMat(rows, cols)
	at := 0
	for _, c := range conds {
		copy(out.Data[at:], c.Data)
		at += len(c.Data)
	}
	return out, nil
}

// EditRequest is one edit: a prompt and one to Refs() reference images.
//
// The images arrive as 8-bit NRGBA because that is what the reference works
// in and what the gates are exact against — PIL resizes 8-bit and composites
// 8-bit, and Q8.3 measured what doing either in float instead costs (rel 11
// on the prompt embedding, through a tower that amplifies an input
// perturbation by ~10³). A server hands this the decoded PNG, unchanged.
type EditRequest struct {
	Prompt string
	Images []*image.NRGBA
	// Width and Height are the output size. Zero takes diffusers' default:
	// the *last* reference image's aspect ratio at the condition area.
	Width, Height int
	Steps         int
	Seed          int64
	// Latents replaces the seeded noise, as Request.Latents does.
	Latents  *qwen.Mat
	Progress func(Step)
}

// ErrNoEdits is a pipeline that was not staged for editing. It is
// distinguishable because it is a server configuration, not a bad request.
var ErrNoEdits = fmt.Errorf("this pipeline was not staged for editing")

// Edit renders one edited image: reference images through the resampler, the
// vision tower and the VAE encoder into the transformer's prefix, the prompt
// through the text encoder with those references' context injected, and then
// the same sampler and decoder a generation uses.
//
// An edit is not a truncated generation — there is no `strength`, and every
// step runs. It is a *conditional* generation: the references are rows of the
// same sequence, modulated from t = 0 and therefore computed once, and the
// target starts from pure noise like any other.
func (p *Pipeline) Edit(req EditRequest) (*zvae.Tensor, *Timings, error) {
	if p.refs == 0 {
		return nil, nil, ErrNoEdits
	}
	if p.enc == nil || p.dt == nil || p.dec == nil {
		return nil, nil, fmt.Errorf("pipeline: destroyed")
	}
	if len(req.Images) == 0 {
		return nil, nil, fmt.Errorf("pipeline: an edit needs at least one reference image")
	}
	if len(req.Images) > p.refs {
		return nil, nil, fmt.Errorf("pipeline: %d reference images; this server is staged for %d",
			len(req.Images), p.refs)
	}
	steps := req.Steps
	if steps == 0 {
		steps = p.steps
	}
	if steps < 1 {
		return nil, nil, fmt.Errorf("pipeline: %d steps", steps)
	}

	tm := &Timings{Refs: len(req.Images)}
	start := time.Now()

	// ---- Every reference image, twice over. One resize feeds both encoders,
	// which is what keeps the tower's patch grid and the VAE's latent grid
	// the same numbers.
	grids := make([]textenc.Grid, len(req.Images))
	conds := make([]*qwen.Mat, len(req.Images))
	shapes := make([][3]int, len(req.Images))
	ctx := make([]textenc.Condition, len(req.Images))
	for i, src := range req.Images {
		b := src.Bounds()
		if b.Dx() <= 0 || b.Dy() <= 0 {
			return nil, nil, fmt.Errorf("pipeline: reference %d is empty", i)
		}
		w, h := CalcDimensions(p.condSize*p.condSize, float64(b.Dx())/float64(b.Dy()))
		if tokens := (w / VAEScale) * (h / VAEScale); tokens > p.condTokens {
			return nil, nil, fmt.Errorf("pipeline: reference %d is %dx%d, which resizes to %dx%d "+
				"and occupies %d tokens; this server's per-reference budget is %d",
				i, b.Dx(), b.Dy(), w, h, tokens, p.condTokens)
		}
		resized := Resize(src, w, h)

		// The vision copy: flattened over white on the 8-bit levels, which is
		// what PIL's own composite does and what the tower was gated on.
		pixels, gridH, gridW, err := p.towerCfg.Patchify(flattenNRGBA(resized), h, w)
		if err != nil {
			return nil, nil, fmt.Errorf("pipeline: reference %d: %w", i, err)
		}
		out, err := p.tower.Forward(pixels, gridH, gridW)
		if err != nil {
			return nil, nil, fmt.Errorf("pipeline: reference %d through the vision tower: %w", i, err)
		}
		ctx[i] = textenc.Condition{Merged: out.Merged, Deepstack: out.Deepstack}
		grids[i] = textenc.Grid{T: 1, H: gridH, W: gridW}

		// The VAE copy keeps all four channels: a port that flattens once and
		// uses it twice is wrong only on transparent references, which is
		// exactly what this model exists to serve.
		z, err := p.venc.Encode(tensorNRGBA(resized))
		if err != nil {
			return nil, nil, fmt.Errorf("pipeline: reference %d through the VAE encoder: %w", i, err)
		}
		p.vcfg.Normalize(z)
		conds[i] = PackLatents(z)
		shapes[i] = [3]int{1, z.H, z.W}
	}
	cond, err := ConcatConditions(conds)
	if err != nil {
		return nil, nil, err
	}
	tm.Condition = time.Since(start)

	// ---- The output geometry. diffusers takes it from the *last* reference
	// image when the caller names none, which is what makes "make this
	// wider" produce something the same shape as what it was given.
	width, height := req.Width, req.Height
	if width == 0 && height == 0 {
		b := req.Images[len(req.Images)-1].Bounds()
		width, height = p.targetFor(float64(b.Dx()) / float64(b.Dy()))
	}
	g, err := p.geomFor(width, height)
	if err != nil {
		return nil, nil, err
	}
	tm.Width, tm.Height = g.width, g.height

	// ---- The prompt, with each reference's slots expanded and its context
	// injected.
	encStart := time.Now()
	prompt, err := textenc.NewEditPrompt(p.tok, req.Prompt, grids, p.towerCfg.SpatialMergeSize)
	if err != nil {
		return nil, nil, err
	}
	if len(prompt.IDs) > p.enc.Tokens() {
		return nil, nil, fmt.Errorf("%w: %d tokens against %d", ErrPromptTooLong, len(prompt.IDs), p.enc.Tokens())
	}
	rope, err := prompt.MRope(p.tcfg.HeadDim, p.tcfg.RopeTheta, p.mrope)
	if err != nil {
		return nil, nil, err
	}
	embeds, padMask, err := prompt.EncodeGPU(p.enc, rope, ctx, p.drop, nil)
	if err != nil {
		return nil, nil, err
	}
	tm.Encode = time.Since(encStart)
	tm.Tokens = embeds.Rows

	lay, err := dit.LayoutFromPadMask(padMask, shapes, [3]int{1, g.latentH, g.latentW})
	if err != nil {
		return nil, nil, err
	}
	// The shift is a function of the *target* token count: a bigger prefix
	// does not move the schedule.
	sched, err := p.scfg.Timesteps(steps, g.imgTokens)
	if err != nil {
		return nil, nil, err
	}
	latents := req.Latents
	if latents == nil {
		latents = p.noise(g, req.Seed)
	} else if latents.Rows != g.imgTokens || latents.Cols != p.vcfg.ZDim {
		return nil, nil, fmt.Errorf("pipeline: latents are %dx%d for a %dx%d target",
			latents.Rows, latents.Cols, g.imgTokens, p.vcfg.ZDim)
	}
	if err := p.dt.BeginImage(embeds, lay, cond); err != nil {
		return nil, nil, err
	}
	if err := p.denoise(latents, g, sched, steps, tm, req.Progress); err != nil {
		return nil, nil, err
	}

	decodeStart := time.Now()
	img, err := p.Decode(latents, g.latentH, g.latentW)
	if err != nil {
		return nil, nil, err
	}
	tm.Decode = time.Since(decodeStart)
	tm.Total = time.Since(start)
	return img, tm, nil
}

// tensorNRGBA is the VAE's copy of a reference image: all four channels in
// [-1, 1], which is the range the encoder was trained and gated on.
func tensorNRGBA(img *image.NRGBA) *zvae.Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	t := zvae.NewTensor(1, 4, h, w)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			for c := 0; c < 4; c++ {
				t.Plane(0, c)[y*w+x] = float32(row[x*4+c])/127.5 - 1
			}
		}
	}
	return t
}

// flattenNRGBA is the *tower's* copy: composited over white and reduced to
// three channels, with the composite done on the 8-bit levels because PIL's
// is — `paste` works on the uint8 image, so every composited pixel lands on a
// level. Doing the same arithmetic in float leaves 97% of the pixels half a
// level away, and qimage/pipeline's own control measures what the tower then
// does with that: rel 11 on the prompt embedding.
func flattenNRGBA(img *image.NRGBA) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			a := float64(row[x*4+3]) / 255
			for c := 0; c < 3; c++ {
				v := float64(row[x*4+c]) / 255
				out[c*h*w+y*w+x] = float32(math.Round((v*a+(1-a))*255)/127.5 - 1)
			}
		}
	}
	return out
}

// targetFor is the output size an edit that named none gets: diffusers'
// `calculate_dimensions` at the condition area, fitted inside this server's
// pixel budget.
//
// **It is now almost always the reference's own answer**, which is what
// changed when the ceiling stopped being a box. `calculate_dimensions` fixes
// the area and lets the sides run, so a 4:3 picture at a 1024² condition
// area becomes 1184x896 — whose long side used to be outside a 1024x1024
// ceiling even though its area was 1% past it, and every non-square edit was
// shrunk a whole step for that. Against an area budget only the 1% is left
// to deal with: the snap to 32 rounds up, so the result can sit one step
// over, and the longer side comes down one step to pay for it. That moves
// the aspect by less than the granularity the model accepts in the first
// place, and a budget above the condition area leaves diffusers' answer
// untouched.
func (p *Pipeline) targetFor(ratio float64) (int, int) {
	budget := p.MaxPixels()
	area := p.condSize * p.condSize
	if area > budget {
		area = budget
	}
	w, h := CalcDimensions(area, ratio)
	for w*h > budget && (w > CondMultiple || h > CondMultiple) {
		if w >= h && w > CondMultiple {
			w -= CondMultiple
		} else {
			h -= CondMultiple
		}
	}
	return w, h
}
