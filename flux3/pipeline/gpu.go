package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"path/filepath"
	"time"

	"strix-halo-vulkan/flux3/dit"
	"strix-halo-vulkan/flux3/textenc"
	"strix-halo-vulkan/flux3/vae"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU is FLUX 3 on the device (X5; served at X6, edits at X7): the Qwen3-VL
// encoder with its pad row, the DiT, the FLUX.1-family decoder on the matrix
// cores and taef2 for previews, with the latent bridge (BatchNorm,
// unpatchify) on the host between the DiT and either decoder; and, staged
// for edits only, the VAE encoder that turns references into image_cond
// tokens.
type GPU struct {
	Enc    *textenc.GPU
	DiT    *dit.GPU
	Dec    *vae.GPUDecoder
	Tiny   *vae.GPUTiny
	RefEnc *vae.GPUEncoder // nil unless Config.Refs > 0
	BN     *BN

	cfg Config
}

const (
	// SizeMultiple is what both sides must be a multiple of: one DiT token is
	// a 16x16 pixel tile.
	SizeMultiple = 16
	// MaxPixels is the area ceiling (X6, X-o3): the full decoder's fp32
	// activation arena at 1232² is 4076 MB, the last square under one
	// storage buffer's 4 GiB, and the arenas scale with the area alone
	// (flux3/vae TestArenaCeiling). The reference accepts up to 4 MiP; a
	// tiled decode is what would lift this.
	MaxPixels = 1232 * 1232
)

// ErrPromptTooLong is a prompt past the encoder's staging: the client's to fix.
var ErrPromptTooLong = errors.New("pipeline: prompt too long")

// Config sizes a staging. Sides are pixels, multiples of 16; what a request
// has to fit is their area, in any shape.
type Config struct {
	Model     string // checkpoint root: transformer/, text_encoder/, vae/
	Preview   string // taef2's directory; "" is taef2/ beside Model
	MaxWidth  int
	MaxHeight int
	MaxPrompt int // real tokens the encoder is staged for
	Chunk     int // the DiT's row chunk; 0 is 4096
	// Refs is how many reference pictures an edit may carry, 0 for no edits.
	// It sizes the DiT's rows (CondRows) and stages the VAE encoder.
	Refs int
	// RefEncoder picks the VAE encoder's kernels; the zero value is the fp16
	// matrix-core path.
	RefEncoder vae.Options
}

// CondRows is the most image_cond tokens an edit can bring to a staging of
// area pixels with up to refs references: one reference's canvas is capped
// at the area (PrepareReference), several at min(1 MiP, area) each.
func CondRows(area, refs int) int {
	if refs <= 0 {
		return 0
	}
	one := area / (16 * 16)
	return max(one, refs*min(MultiReferenceMaxPixels/(16*16), one))
}

// NewGPU stages the four models: ~11 GB DiT, ~7 GB encoder, the decoder's
// and taef2's weights and arenas, all fp16.
func NewGPU(dev *vk.Device, c Config) (*GPU, error) {
	if c.MaxWidth%SizeMultiple != 0 || c.MaxHeight%SizeMultiple != 0 || c.MaxWidth <= 0 || c.MaxHeight <= 0 {
		return nil, fmt.Errorf("pipeline: %dx%d is not a positive multiple of %d", c.MaxWidth, c.MaxHeight, SizeMultiple)
	}
	if c.MaxWidth*c.MaxHeight > MaxPixels {
		return nil, fmt.Errorf("pipeline: %dx%d is %d pixels, past the %d (1232²) the decoder's single-buffer arena holds",
			c.MaxWidth, c.MaxHeight, c.MaxWidth*c.MaxHeight, MaxPixels)
	}
	if c.MaxPrompt <= 0 {
		return nil, fmt.Errorf("pipeline: MaxPrompt %d", c.MaxPrompt)
	}
	if c.Refs < 0 || c.Refs > MaxReferences {
		return nil, fmt.Errorf("pipeline: %d references; the model takes 0 to %d", c.Refs, MaxReferences)
	}
	if c.Chunk == 0 {
		c.Chunk = 4096
	}
	if c.Preview == "" {
		c.Preview = filepath.Join(filepath.Dir(filepath.Clean(c.Model)), "taef2")
	}
	p := &GPU{cfg: c}
	fail := func(err error) (*GPU, error) {
		p.Destroy()
		return nil, err
	}
	var err error
	if p.BN, err = LoadBN(filepath.Join(c.Model, "vae")); err != nil {
		return nil, err
	}
	if p.Enc, err = textenc.NewGPU(dev, filepath.Join(c.Model, "text_encoder"), c.MaxPrompt); err != nil {
		return fail(err)
	}
	maxText := textenc.Window(c.MaxPrompt)
	if p.DiT, err = dit.NewGPU(dev, filepath.Join(c.Model, "transformer"), maxText+p.tokens()+CondRows(p.Area(), c.Refs), maxText, c.Chunk); err != nil {
		return fail(err)
	}
	vdir := filepath.Join(c.Model, "vae")
	vcfg, err := vae.LoadConfig(vdir)
	if err != nil {
		return fail(err)
	}
	cpu, err := vae.LoadDecoder(vdir, vcfg)
	if err != nil {
		return fail(err)
	}
	if p.Dec, err = vae.NewGPUDecoder(dev, cpu, c.MaxHeight/8, c.MaxWidth/8); err != nil {
		return fail(err)
	}
	tiny, err := vae.LoadTinyDecoder(c.Preview, vae.TAEF2Config())
	if err != nil {
		return fail(fmt.Errorf("pipeline: the preview decoder: %w", err))
	}
	if p.Tiny, err = vae.NewGPUTiny(dev, tiny, c.MaxHeight/8, c.MaxWidth/8); err != nil {
		return fail(err)
	}
	if c.Refs > 0 {
		enc, err := vae.LoadEncoder(vdir, vcfg)
		if err != nil {
			return fail(err)
		}
		// Sized by the area: every reference is capped at it.
		if p.RefEnc, err = vae.NewGPUEncoderOpts(dev, enc, c.MaxHeight, c.MaxWidth, c.RefEncoder); err != nil {
			return fail(err)
		}
	}
	return p, nil
}

func (p *GPU) tokens() int { return p.cfg.MaxWidth * p.cfg.MaxHeight / (16 * 16) }

// Destroy releases everything staged.
func (p *GPU) Destroy() {
	if p.RefEnc != nil {
		p.RefEnc.Destroy()
	}
	if p.Tiny != nil {
		p.Tiny.Destroy()
	}
	if p.Dec != nil {
		p.Dec.Destroy()
	}
	if p.DiT != nil {
		p.DiT.Destroy()
	}
	if p.Enc != nil {
		p.Enc.Destroy()
	}
}

// Size is the staging's own size, a request's default.
func (p *GPU) Size() (w, h int) { return p.cfg.MaxWidth, p.cfg.MaxHeight }

// Area is the most pixels a request may ask for.
func (p *GPU) Area() int { return p.cfg.MaxWidth * p.cfg.MaxHeight }

// Refs is how many references an edit may carry; 0 means no edits.
func (p *GPU) Refs() int { return p.cfg.Refs }

// Residency is what the staging holds on the device, in bytes.
func (p *GPU) Residency() (encoder, transformer, vaes, activations int) {
	df32, df16 := p.Dec.WeightBytes()
	tf32, tf16 := p.Tiny.WeightBytes()
	vaes = df32 + df16 + tf32 + tf16
	activations = p.Enc.Enc.ActivationBytes() + p.DiT.ActivationBytes() + p.Dec.ActivationBytes() +
		p.Dec.F16ActivationBytes() + p.Tiny.ActivationBytes() + p.Tiny.F16ActivationBytes()
	if p.RefEnc != nil {
		ef32, ef16 := p.RefEnc.WeightBytes()
		vaes += ef32 + ef16
		activations += p.RefEnc.ActivationBytes() + p.RefEnc.F16ActivationBytes()
	}
	return p.Enc.Enc.WeightBytes(), p.DiT.WeightBytes(), vaes, activations
}

// Timings is a generation's wall clock by stage; Device is the DiT's own
// device time inside Denoise, Previews the time spent on preview decodes
// (not counted in Denoise), References the preparation and VAE encode of an
// edit's pictures. Width and Height are the image returned: an edit's is
// cropped back to its reference's fitted box.
type Timings struct {
	Width, Height, Steps    int
	Encode, Denoise, Decode time.Duration
	Device, Previews        time.Duration
	References              time.Duration
	CondTokens              int
}

// Step is one finished step, as a Request's Progress sees it.
type Step struct {
	Index, Steps int
	// Latent is the state after the step (the next step's input).
	Latent *qwen.Mat
	// Preview decodes this step's denoised estimate, x − t·v (the state the
	// step started from, less its time times its velocity), through taef2.
	// Under the flow map v is the average velocity to the step's target, so
	// this is an estimate of the end, not the end; it is what a client wants
	// to watch. It costs ~0.1 s at 1024², so it runs only when called.
	Preview func() (*vae.Tensor, error)
}

// Request is one generation, or an edit when it carries references.
type Request struct {
	Prompt string
	// Width and Height zero take the staging's size, or an edit's default:
	// its first prepared reference's, cropped back to the fitted box when
	// there is one reference (BFL's generate.py).
	Width, Height int
	Steps         int // zero takes DefaultSteps
	Seed          int64
	// References are an edit's pictures as the client sent them, at most
	// Config.Refs; PrepareReferences fits them.
	References []image.Image
	// Noise, when set, replaces the seeded noise (a gate's dumped noise).
	Noise *qwen.Mat
	// Progress, when set, sees each step as it lands.
	Progress func(Step)
}

// Resolve fills a request's zero sides and checks the shape against the
// staging: both sides multiples of 16, the area within the staging's.
func (p *GPU) Resolve(width, height int) (int, int, error) {
	if width == 0 && height == 0 {
		width, height = p.Size()
	}
	if width <= 0 || height <= 0 {
		return 0, 0, fmt.Errorf("pipeline: name both sides or neither (got %dx%d)", width, height)
	}
	if width%SizeMultiple != 0 || height%SizeMultiple != 0 {
		return 0, 0, fmt.Errorf("pipeline: %dx%d: both sides must be multiples of %d", width, height, SizeMultiple)
	}
	if width*height > p.Area() {
		return 0, 0, fmt.Errorf("pipeline: %dx%d is %d pixels; this server was staged for %d", width, height, width*height, p.Area())
	}
	return width, height, nil
}

// Generate runs a request to a decoded [1, 3, H, W] image in [-1, 1]. The
// context is checked between steps.
func (p *GPU) Generate(ctx context.Context, r Request) (*vae.Tensor, *Timings, error) {
	if len(r.References) > p.cfg.Refs {
		if p.cfg.Refs == 0 {
			return nil, nil, fmt.Errorf("%w: this staging does not edit", ErrReferences)
		}
		return nil, nil, fmt.Errorf("%w: %d reference images, staged for %d", ErrReferences, len(r.References), p.cfg.Refs)
	}
	start := time.Now()
	refs, err := PrepareReferences(r.References, p.Area())
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %v", ErrReferences, err)
	}
	var crop image.Rectangle
	if len(refs) > 0 && r.Width == 0 && r.Height == 0 {
		r.Width, r.Height = refs[0].Image.Rect.Dx(), refs[0].Image.Rect.Dy()
		crop = refs[0].Box
	}
	w, h, err := p.Resolve(r.Width, r.Height)
	if err != nil {
		return nil, nil, err
	}
	if r.Steps == 0 {
		r.Steps = DefaultSteps
	}
	tm := &Timings{Width: w, Height: h, Steps: r.Steps}
	cond, condIDs, err := p.encodeReferences(refs)
	if err != nil {
		return nil, nil, err
	}
	if cond != nil {
		tm.CondTokens = cond.Rows
	}
	tm.References = time.Since(start)
	start = time.Now()
	ids, err := p.Enc.IDs(r.Prompt)
	if err != nil {
		return nil, nil, err
	}
	if len(ids) > p.cfg.MaxPrompt {
		return nil, nil, fmt.Errorf("%w: %d tokens, the encoder is staged for %d", ErrPromptTooLong, len(ids), p.cfg.MaxPrompt)
	}
	txt, err := p.Enc.Encode(ids)
	if err != nil {
		return nil, nil, err
	}
	tm.Encode = time.Since(start)

	th, tw := h/16, w/16
	noise := r.Noise
	if noise == nil {
		noise = Noise(th*tw, 128, r.Seed)
	}
	if noise.Rows != th*tw {
		return nil, nil, fmt.Errorf("pipeline: %d noise rows for a %dx%d grid", noise.Rows, th, tw)
	}
	if err := p.DiT.Begin(txt, TextIDs(txt.Rows), ImageIDs(th, tw, 0), cond, condIDs); err != nil {
		return nil, nil, err
	}
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		return nil, nil, err
	}
	// The step's input, time and velocity, kept for the preview's estimate.
	var xIn, vLast *qwen.Mat
	var tLast float64
	start = time.Now()
	x, err := DenoiseWith(ctx, noise, s, func(x *qwen.Mat, t, tNext float64) (*qwen.Mat, error) {
		if r.Progress != nil {
			xIn, tLast = x.Clone(), t
		}
		v, took, err := p.DiT.Forward(x, t, tNext)
		tm.Device += took
		vLast = v
		return v, err
	}, func(i int, x *qwen.Mat) {
		if r.Progress == nil {
			return
		}
		xi, vi, ti := xIn, vLast, float32(tLast)
		r.Progress(Step{Index: i, Steps: r.Steps, Latent: x, Preview: func() (*vae.Tensor, error) {
			t0 := time.Now()
			est := xi.Clone()
			for j, v := range vi.Data {
				est.Data[j] -= ti * v
			}
			img, err := p.decode(p.Tiny.Apply, Unpatchify, est, th, tw)
			tm.Previews += time.Since(t0)
			if err != nil {
				return nil, err
			}
			return cropTensor(img, crop), nil
		}})
	})
	if err != nil {
		return nil, nil, err
	}
	tm.Denoise = time.Since(start) - tm.Previews

	start = time.Now()
	img, err := p.decode(p.Dec.Apply, p.BN.ToVAE, x, th, tw)
	if err != nil {
		return nil, nil, err
	}
	tm.Decode = time.Since(start)
	img = cropTensor(img, crop)
	tm.Width, tm.Height = img.W, img.H
	return img, tm, nil
}

// ErrReferences is an edit's pictures the staging cannot take: the client's
// to fix.
var ErrReferences = errors.New("pipeline: reference images")

// encodeReferences runs prepared pictures through the VAE encoder to the
// DiT's image_cond tokens [Σ hk*wk, 128], normalised, in reference order,
// and their ids: reference k's grid at l = −(1+k).
func (p *GPU) encodeReferences(refs []Reference) (*qwen.Mat, *qwen.Mat, error) {
	if len(refs) == 0 {
		return nil, nil, nil
	}
	var toks, ids []*qwen.Mat
	rows := 0
	for k, ref := range refs {
		rw, rh := ref.Image.Rect.Dx(), ref.Image.Rect.Dy()
		mode, err := p.RefEnc.Encode(&vae.Tensor{N: 1, C: 3, H: rh, W: rw, Data: ToTensor(ref.Image)})
		if err != nil {
			return nil, nil, fmt.Errorf("reference %d: %w", k, err)
		}
		t, err := p.BN.FromVAE(mode.Data, rh/16, rw/16)
		if err != nil {
			return nil, nil, err
		}
		toks = append(toks, t)
		ids = append(ids, ImageIDs(rh/16, rw/16, float32(-(1+k))))
		rows += t.Rows
	}
	if rows > CondRows(p.Area(), p.cfg.Refs) {
		return nil, nil, fmt.Errorf("%w: %d reference tokens, staged for %d", ErrReferences, rows, CondRows(p.Area(), p.cfg.Refs))
	}
	return concatRows(toks), concatRows(ids), nil
}

func concatRows(ms []*qwen.Mat) *qwen.Mat {
	n := 0
	for _, m := range ms {
		n += m.Rows
	}
	out := qwen.NewMat(n, ms[0].Cols)
	off := 0
	for _, m := range ms {
		copy(out.Data[off:], m.Data)
		off += len(m.Data)
	}
	return out
}

// cropTensor cuts a [1, C, H, W] image to box; an empty box is no crop.
func cropTensor(t *vae.Tensor, box image.Rectangle) *vae.Tensor {
	if box.Empty() || box == image.Rect(0, 0, t.W, t.H) {
		return t
	}
	out := vae.NewTensor(1, t.C, box.Dy(), box.Dx())
	for c := 0; c < t.C; c++ {
		src, dst := t.Plane(0, c), out.Plane(0, c)
		for y := 0; y < box.Dy(); y++ {
			copy(dst[y*box.Dx():(y+1)*box.Dx()], src[(box.Min.Y+y)*t.W+box.Min.X:])
		}
	}
	return out
}

// decode is a latent bridge and a decoder: DiT tokens [th*tw, 128] to a
// [32, 2th, 2tw] latent -- raw (BN.ToVAE) for the full decoder, still
// normalised (Unpatchify) for taef2 -- and through it.
func (p *GPU) decode(apply func(*vae.Tensor) (*vae.Tensor, error), bridge func(*qwen.Mat, int, int) ([]float32, error),
	x *qwen.Mat, th, tw int) (*vae.Tensor, error) {
	z, err := bridge(x, th, tw)
	if err != nil {
		return nil, err
	}
	return apply(&vae.Tensor{N: 1, C: len(z) / (4 * th * tw), H: 2 * th, W: 2 * tw, Data: z})
}
