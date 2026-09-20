package pipeline

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"time"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/qimage/textenc"
	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
	zvae "strix-halo-vulkan/zimage/vae"
)

// The resident text-to-image pipeline — IMAGE.md Q6.
//
// Three staged models and nothing else: Q1's text encoder (zimage/qwen's
// GPUEncoder over the Qwen3-VL config), Q4's DiT, and Q5g's VAE decoder.
// Everything between them is arithmetic the host does per request — the
// template, the token drop, the layout, the schedule, the packing — so a
// request names a size and a step count rather than choosing a process.
//
// **The resolution is not part of what is resident.** The three arenas are
// sized once for the ceiling the flags asked for; a smaller image is the
// same three graphs with smaller numbers in them. That is the same split
// z-image's pipeline made and the same one api.ImageBackend's Geometry()
// expects: a size is a request parameter, a ceiling is residency.
//
// **The ceiling is the VAE's, and it is 1184², not 2048².** The decoder's
// activation arena is one storage buffer and this device caps one at 4 GiB
// − 4 (qimage/vae's TestArenaCeiling). New refuses a larger ceiling here,
// naming the number, rather than letting a buffer allocation fail three
// models into staging.

const (
	// SizeMultiple is what both sides of an image must be a multiple of: the
	// VAE is 16x spatial and the VLM groups latent tokens 2x2, so a latent
	// grid has to be even on both axes.
	SizeMultiple = 32
	// VAEScale is how much larger the image is than the latent grid.
	VAEScale = 16

	// DefaultSteps is the checkpoint's own default schedule. Unlike
	// Z-Image-Turbo's 8, this is not a distillation — 2.1 is a full model and
	// 40 is what diffusers runs.
	DefaultSteps = 40
	// DefaultSize is the pipeline's default resolution, which is also
	// diffusers' (the README's examples use 2048² explicitly instead).
	DefaultSize = 1024
	// DefaultMaxPrompt is the text encoder's token ceiling. A t2i prompt is
	// the template plus the user's text, tens of tokens in practice.
	DefaultMaxPrompt = 512
)

// Options is what cmd/serve's flags come to.
type Options struct {
	// Model is the Qwen-Image-2.1 checkpoint root, holding transformer/,
	// text_encoder/, vae/, processor/ and scheduler/.
	Model string
	// Width and Height are the largest image the arenas are built for, and
	// the size a request that names none gets.
	Width, Height int
	// Steps is the schedule a request that names none gets.
	Steps int
	// MaxPrompt is the longest prompt the text encoder is built for.
	MaxPrompt int
}

// geom is a resolved size, all the way down to the transformer's row count.
type geom struct {
	width, height    int
	latentH, latentW int
	imgTokens        int
}

func newGeom(width, height int) (geom, error) {
	if width <= 0 || height <= 0 {
		return geom{}, fmt.Errorf("pipeline: %dx%d is not an image", width, height)
	}
	if width%SizeMultiple != 0 || height%SizeMultiple != 0 {
		return geom{}, fmt.Errorf("pipeline: %dx%d; both sides must be a multiple of %d",
			width, height, SizeMultiple)
	}
	g := geom{width: width, height: height, latentH: height / VAEScale, latentW: width / VAEScale}
	g.imgTokens = g.latentH * g.latentW
	return g, nil
}

// Pipeline is the three staged models plus what a request needs to drive
// them. One image at a time: the arenas are one image's.
type Pipeline struct {
	opt Options

	tok  *tokenizer.Tokenizer
	drop int
	enc  *qwen.GPUEncoder
	dt   *dit.GPU
	dec  *qvae.GPUDecoder
	vcfg *qvae.Config
	scfg *SchedConfig

	def, max geom
	steps    int

	// residency, for the startup banner.
	encBytes, ditBytes, vaeBytes, actBytes int
}

// New stages the whole pipeline on the device.
func New(dev *vk.Device, opt Options) (*Pipeline, error) {
	if opt.Width == 0 && opt.Height == 0 {
		opt.Width, opt.Height = DefaultSize, DefaultSize
	}
	if opt.Width == 0 {
		opt.Width = opt.Height
	}
	if opt.Height == 0 {
		opt.Height = opt.Width
	}
	if opt.Steps == 0 {
		opt.Steps = DefaultSteps
	}
	if opt.MaxPrompt == 0 {
		opt.MaxPrompt = DefaultMaxPrompt
	}
	max, err := newGeom(opt.Width, opt.Height)
	if err != nil {
		return nil, err
	}

	p := &Pipeline{opt: opt, def: max, max: max, steps: opt.Steps}

	// The tokenizer and the two configs first: they are cheap, and a
	// checkpoint that is missing a piece should say so before 30 GB of
	// staging rather than after.
	if p.tok, err = tokenizer.Load(opt.Model + "/processor"); err != nil {
		return nil, fmt.Errorf("pipeline: tokenizer: %w", err)
	}
	if p.drop, err = textenc.DropTokens(p.tok); err != nil {
		return nil, fmt.Errorf("pipeline: system-message length: %w", err)
	}
	tcfg, err := textenc.LoadConfig(opt.Model + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("pipeline: text encoder config: %w", err)
	}
	if p.vcfg, err = qvae.LoadConfig(opt.Model + "/vae"); err != nil {
		return nil, fmt.Errorf("pipeline: VAE config: %w", err)
	}
	if p.scfg, err = LoadSchedConfig(opt.Model + "/scheduler"); err != nil {
		return nil, fmt.Errorf("pipeline: scheduler config: %w", err)
	}

	// The VAE's ceiling, checked before anything is staged. It is a property
	// of the graph and the device, so it costs a plan rather than an
	// allocation — and the alternative is a caller discovering it from a
	// failed 12 GB buffer after the other two models are already resident.
	cpuDec, err := qvae.LoadDecoder(opt.Model+"/vae", p.vcfg)
	if err != nil {
		return nil, fmt.Errorf("pipeline: VAE decoder: %w", err)
	}
	need, err := qvae.ArenaBytes(cpuDec, max.latentH, max.latentW)
	if err != nil {
		return nil, err
	}
	if need > qvae.MaxStorageBufferBytes {
		return nil, fmt.Errorf("pipeline: a %dx%d ceiling needs %d MB of VAE activations, "+
			"past this device's %d MB single-buffer limit; the largest square is 1184x1184",
			max.width, max.height, need>>20, qvae.MaxStorageBufferBytes>>20)
	}

	set, err := safetensors.OpenSet(opt.Model + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("pipeline: text encoder weights: %w", err)
	}
	if p.enc, err = qwen.NewGPUEncoder(dev, set, tcfg, tcfg.NumLayers, opt.MaxPrompt, nil); err != nil {
		set.Close()
		return nil, fmt.Errorf("pipeline: staging the text encoder: %w", err)
	}
	set.Close()

	// The DiT's row ceiling is the target's tokens plus the longest prompt:
	// for t2i the prefix is the prompt and nothing else.
	if p.dt, err = dit.NewGPU(dev, opt.Model+"/transformer", max.imgTokens+opt.MaxPrompt, opt.MaxPrompt, opt.MaxPrompt); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: staging the transformer: %w", err)
	}
	if p.dec, err = qvae.NewGPUDecoder(dev, cpuDec, max.latentH, max.latentW); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: staging the VAE decoder: %w", err)
	}

	p.encBytes = p.enc.WeightBytes()
	p.vaeBytes = p.dec.WeightBytes()
	p.ditBytes = p.dt.WeightBytes()
	p.actBytes = p.enc.ActivationBytes() + p.dt.ActivationBytes() + p.dec.ActivationBytes()
	return p, nil
}

// Destroy releases the device residency.
func (p *Pipeline) Destroy() {
	if p.dec != nil {
		p.dec.Destroy()
		p.dec = nil
	}
	if p.dt != nil {
		p.dt.Destroy()
		p.dt = nil
	}
	if p.enc != nil {
		p.enc.Destroy()
		p.enc = nil
	}
}

// Size is the image a request that names none gets; MaxSize is the ceiling.
func (p *Pipeline) Size() (width, height int)    { return p.def.width, p.def.height }
func (p *Pipeline) MaxSize() (width, height int) { return p.max.width, p.max.height }

// Steps is the default schedule length.
func (p *Pipeline) Steps() int { return p.steps }

// MaxPrompt is the text encoder's token ceiling.
func (p *Pipeline) MaxPrompt() int { return p.opt.MaxPrompt }

// Residency is what the pipeline holds on the device, for a startup banner.
func (p *Pipeline) Residency() (encoder, transformer, vaeWeights, activations int) {
	return p.encBytes, p.ditBytes, p.vaeBytes, p.actBytes
}

// LatentFor resolves a request's size to the latent grid the transformer
// will denoise. A size this pipeline cannot run is an error rather than a
// rounded-up answer.
func (p *Pipeline) LatentFor(width, height int) (channels, h, w int, err error) {
	g, err := p.geomFor(width, height)
	if err != nil {
		return 0, 0, 0, err
	}
	return p.vcfg.ZDim, g.latentH, g.latentW, nil
}

func (p *Pipeline) geomFor(width, height int) (geom, error) {
	if width == 0 {
		width = p.def.width
	}
	if height == 0 {
		height = p.def.height
	}
	g, err := newGeom(width, height)
	if err != nil {
		return geom{}, err
	}
	if g.width > p.max.width || g.height > p.max.height {
		return geom{}, fmt.Errorf("pipeline: %dx%d is past this server's %dx%d ceiling",
			g.width, g.height, p.max.width, p.max.height)
	}
	return g, nil
}

// ErrPromptTooLong is a prompt past the text encoder's arena. It is the
// client's to fix, which is why it is distinguishable.
var ErrPromptTooLong = fmt.Errorf("prompt is longer than the text encoder's token ceiling")

// Request is one image to generate.
type Request struct {
	Prompt        string
	Width, Height int
	Steps         int
	Seed          int64
	// Latents, when given, replaces the seeded noise. It is the packed
	// [tokens, z] layout Run works in, and it is written through: the loop
	// integrates in place and a caller usually wants the final latent too.
	Latents  *qwen.Mat
	Progress func(Step)
}

// Step is what a progress callback is told after each denoising step.
//
// Latents is the loop's own buffer and the next step overwrites it, so a
// callback that keeps it has to copy.
type Step struct {
	Index int
	Steps int
	Wall  time.Duration
	// Sigma and NextSigma are the noise levels this step moved between.
	Sigma, NextSigma float64
	Latents          *qwen.Mat

	// Preview decodes this step's *denoised estimate* through the fitted
	// linear map (preview.go) into an image at latent resolution, in the
	// same [-1, 1] RGBA layout the real decoder produces. It is valid only
	// for the duration of the callback.
	//
	// It is a closure rather than a decoded tensor because the estimate has
	// to be formed -- one pass over the latents -- and a callback that only
	// wants the wall clock should not pay for it. That it is cheap enough
	// not to need a flag is the point of Q7; that it is not *free* is why it
	// is still lazy.
	Preview func() (*zvae.Tensor, error)
}

// Timings is what one image cost, by stage.
type Timings struct {
	Encode  time.Duration // tokenizer and the text encoder
	Prefill time.Duration // the DiT's step 0, which also fills the prefix cache
	Steps   []time.Duration
	Decode  time.Duration // the VAE
	Total   time.Duration

	Tokens        int // the prompt's length after the template and the drop
	Width, Height int
}

// Generate is Run for the common case.
func (p *Pipeline) Generate(prompt string, seed int64, progress func(Step)) (*zvae.Tensor, *Timings, error) {
	return p.Run(Request{Prompt: prompt, Seed: seed, Progress: progress})
}

// Encode runs the tokenizer and the text encoder, returning the DiT's text
// embedding. It is separate because it is the one stage whose cost does not
// scale with the image, and because a caller sweeping step counts over one
// prompt should not pay for it each time.
func (p *Pipeline) Encode(prompt string) (*qwen.Mat, error) {
	ids, err := textenc.EncodePrompt(p.tok, prompt)
	if err != nil {
		return nil, err
	}
	if len(ids) > p.opt.MaxPrompt {
		return nil, fmt.Errorf("%w: %d tokens against %d", ErrPromptTooLong, len(ids), p.opt.MaxPrompt)
	}
	out, err := p.enc.Forward(ids)
	if err != nil {
		return nil, err
	}
	return textenc.Drop(out, p.drop)
}

// Noise draws the initial packed latents for a geometry from a seeded
// Gaussian, in the [tokens, z] layout the DiT works in.
//
// The order is the packed buffer's own — token-major, channel-minor — which
// is this repo's convention rather than the reference's: diffusers seeds a
// torch generator over [B, C, H, W], and reproducing *its* image from a seed
// was never available to us. What this does guarantee is our own
// reproducibility, which is what a client re-sending a seed is asking for.
func (p *Pipeline) Noise(width, height int, seed int64) (*qwen.Mat, error) {
	g, err := p.geomFor(width, height)
	if err != nil {
		return nil, err
	}
	return p.noise(g, seed), nil
}

func (p *Pipeline) noise(g geom, seed int64) *qwen.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := qwen.NewMat(g.imgTokens, p.vcfg.ZDim)
	for i := range m.Data {
		m.Data[i] = float32(rng.NormFloat64())
	}
	return m
}

// Run renders one image: prompt through the text encoder, the DiT's sampler
// with the prefix KV cache, and the VAE decoder. The returned tensor is
// [1, 4, H, W] RGBA in [-1, 1], which is the decoder's own range.
func (p *Pipeline) Run(req Request) (*zvae.Tensor, *Timings, error) {
	if p.enc == nil || p.dt == nil || p.dec == nil {
		return nil, nil, fmt.Errorf("pipeline: destroyed")
	}
	g, err := p.geomFor(req.Width, req.Height)
	if err != nil {
		return nil, nil, err
	}
	steps := req.Steps
	if steps == 0 {
		steps = p.steps
	}
	if steps < 1 {
		return nil, nil, fmt.Errorf("pipeline: %d steps", steps)
	}

	tm := &Timings{Width: g.width, Height: g.height}
	start := time.Now()

	embeds, err := p.Encode(req.Prompt)
	if err != nil {
		return nil, nil, err
	}
	tm.Encode = time.Since(start)
	tm.Tokens = embeds.Rows

	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, g.latentH, g.latentW}})
	if err != nil {
		return nil, nil, err
	}
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

	if err := p.dt.BeginImage(embeds, lay, nil); err != nil {
		return nil, nil, err
	}
	for i := 0; i < steps; i++ {
		stepStart := time.Now()
		// Step 0 is the block-causal prefill, which also extracts every
		// block's prefix K/V; the rest decode only the target rows against
		// that cache (IMAGE.md decision 2).
		out, err := p.dt.Step(latents, sched.T(i), i == 0)
		if err != nil {
			return nil, nil, fmt.Errorf("pipeline: step %d: %w", i, err)
		}
		if err := sched.Step(i, latents.Data, out.Data); err != nil {
			return nil, nil, err
		}
		wall := time.Since(stepStart)
		tm.Steps = append(tm.Steps, wall)
		if i == 0 {
			tm.Prefill = wall
		}
		if req.Progress != nil {
			// The denoised estimate, after the Euler move rather than before
			// it, which is the cheaper of the two identities: the step has
			// already written x_{t+1} = x_t + (s' - s) v, so
			// x0 = x_t - s v = x_{t+1} - s' v with s' the *next* sigma. No
			// copy of the pre-step latents is needed, and at the final step
			// s' is zero -- correctly making x0 the latent that gets decoded.
			sigmaNext := float64(sched.Sigmas[i+1])
			v := out
			req.Progress(Step{
				Index: i, Steps: steps, Wall: wall,
				Sigma:     float64(sched.Sigmas[i]),
				NextSigma: sigmaNext,
				Latents:   latents,
				Preview: func() (*zvae.Tensor, error) {
					x0 := latents.Clone()
					if sigmaNext != 0 {
						s := float32(sigmaNext)
						for j := range x0.Data {
							x0.Data[j] -= s * v.Data[j]
						}
					}
					return PreviewDecode(x0, g.latentH, g.latentW)
				},
			})
		}
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

// Decode unpacks the DiT's [tokens, z] latents into the VAE's [1, z, h, w],
// denormalizes them and decodes. It is exported because the step sweep and
// the CPU oracle both want to decode a latent they already have.
func (p *Pipeline) Decode(latents *qwen.Mat, latentH, latentW int) (*zvae.Tensor, error) {
	if latents.Rows != latentH*latentW {
		return nil, fmt.Errorf("pipeline: %d latent rows for a %dx%d grid", latents.Rows, latentH, latentW)
	}
	z := zvae.NewTensor(1, latents.Cols, latentH, latentW)
	for tk := 0; tk < latents.Rows; tk++ {
		row := latents.Row(tk)
		for c := 0; c < latents.Cols; c++ {
			z.Plane(0, c)[tk] = row[c]
		}
	}
	p.vcfg.Denormalize(z)
	return p.dec.Decode(z)
}

// ToImage maps the decoder's [1, 4, H, W] RGBA in [-1, 1] to a picture.
//
// It is here rather than in an adapter because the fourth channel's meaning
// is the model's: Qwen-Image-2.1's VAE emits alpha always, and transparency
// is asked for *in the prompt* (the model card's phrasing) rather than by a
// flag. So a caller that did not ask for it still gets an alpha plane, and
// what to do with it is the one policy this function carries — `opaque`
// composites over white, which is what an ordinary PNG or a JPEG wants, and
// otherwise the alpha is kept.
//
// The clamp is the decoder's and has already happened; this rounds.
func ToImage(t *zvae.Tensor, opaque bool) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, t.W, t.H))
	plane := t.H * t.W
	hasAlpha := t.C >= 4
	at := func(c, i int) float64 {
		// [-1, 1] to [0, 1].
		return (float64(t.Data[c*plane+i]) + 1) / 2
	}
	b := func(v float64) uint8 { return uint8(math.Round(math.Min(255, math.Max(0, v*255)))) }
	for i := 0; i < plane; i++ {
		a := 1.0
		if hasAlpha {
			a = math.Min(1, math.Max(0, at(3, i)))
		}
		px := img.Pix[i*4:]
		for c := 0; c < 3; c++ {
			v := at(c, i)
			if opaque && hasAlpha {
				// Over white, in the straight-alpha space the tensor is in.
				v = a*v + (1 - a)
			}
			px[c] = b(v)
		}
		if opaque {
			px[3] = 255
		} else {
			px[3] = b(a)
		}
	}
	return img
}
