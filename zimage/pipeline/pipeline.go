package pipeline

import (
	"fmt"
	"math/rand"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/dit"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
	"strix-halo-vulkan/zimage/vae"
)

// Options configures a Pipeline. The sizes are fixed at construction because
// every arena in it is: the text encoder is built for a maximum prompt, the
// transformer for a maximum unified sequence, and the VAE decoder for one
// latent grid.
type Options struct {
	Model     string // the checkpoint root, holding transformer/, vae/, ...
	Width     int    // image pixels; a multiple of 16
	Height    int
	Steps     int
	MaxPrompt int // the longest prompt the text encoder is built for
}

// Defaults fills in what was left zero.
func (o Options) Defaults() Options {
	if o.Model == "" {
		o.Model = "models/Z-Image-Turbo"
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
	if o.Steps == 0 {
		o.Steps = 8
	}
	if o.MaxPrompt == 0 {
		o.MaxPrompt = 512
	}
	return o
}

// vaeScale is how much larger the image is than the latent grid: the decoder
// upsamples 8x and the transformer's patch size is 2, so both sides must be a
// multiple of 16.
const vaeScale = 8

// Pipeline holds every stage resident: 7.07 GB of text encoder, 12.54 GB of
// transformer and the VAE, plus their activation arenas. Construction is what
// costs -- a little over twenty seconds, nearly all of it staging weights --
// and a Pipeline then generates images without touching the disk again.
type Pipeline struct {
	opt Options

	tok *tokenizer.Tokenizer
	enc *qwen.GPUEncoder

	cfg   *dit.Config
	head  *dit.Head
	stack *dit.GPUStack

	dec *vae.GPUDecoder

	sched *FlowMatchEuler
	scale float64 // the VAE's scaling_factor
	shift float64 // its shift_factor

	// The three phases' block indices, in the order the transformer runs
	// them. StackBlocks lays the stack out this way, so they follow from the
	// config's counts.
	noiseRefiner, contextRefiner, layers []int

	// ctl is empty in every real use; see controls.
	ctl controls

	latentH, latentW int
	imgTokens        int // the image stream before padding
	imgTotal         int // and after
	maxUnified       int
}

// Timings is what one image cost, by stage.
type Timings struct {
	Encode   time.Duration // tokenizer and the text encoder
	Caption  time.Duration // cap_embedder and the context refiners, once per image
	Steps    []time.Duration
	Decode   time.Duration // the VAE
	Total    time.Duration
	Tokens   int // the prompt's length
	CapTotal int // the caption stream after padding
	Unified  int // the sequence the layers ran over
}

// controls are the deliberate breakages the negative control switches on.
// They live here rather than in the test because what each of them breaks is
// the *composition* -- the order the three stages run in, the sign the
// scheduler is handed, which rows of the unified sequence the caption's rotary
// positions belong to -- and none of that is reachable from outside
// GenerateFrom. Every one of them still produces an image.
type controls struct {
	// noNegate leaves the transformer's output as it is instead of negating
	// it before the Euler step. The model predicts the velocity towards the
	// noise and the schedule runs away from it; nothing about the tensor says
	// which way it points.
	noNegate bool
	// staleCaption puts the *embedded* caption behind the image instead of
	// the one the context refiners produced, i.e. skips the phase whose whole
	// output is 32 rows of a 4128-row sequence.
	staleCaption bool
	// capPosFromZero starts the caption's rotary positions at 0 rather than
	// 1, which is the off-by-one the checkpoint invites: nothing downstream
	// of a rotary table can tell that every caption token moved one place.
	capPosFromZero bool
}

// Step is what a progress callback is told after each denoising step.
//
// Blocks is the time in the transformer's blocks and Head the time in the CPU
// head and tail; they are separated because the head is the pipeline's one
// unoptimised piece. Latents is the state the step left, [C, H, W] -- it is the
// pipeline's own buffer and the next step overwrites it, so a callback that
// wants to keep it (to decode a preview, say) has to copy it.
type Step struct {
	Index          int
	Blocks, Head   time.Duration
	Latents        []float32
	Sigma, NextSig float64
}

// New builds the pipeline, staging every weight onto the device.
func New(dev *vk.Device, opt Options) (*Pipeline, error) {
	opt = opt.Defaults()
	if opt.Width%(2*vaeScale) != 0 || opt.Height%(2*vaeScale) != 0 {
		return nil, fmt.Errorf("pipeline: %dx%d; both sides must be a multiple of %d", opt.Width, opt.Height, 2*vaeScale)
	}
	p := &Pipeline{
		opt:     opt,
		latentH: opt.Height / vaeScale,
		latentW: opt.Width / vaeScale,
	}

	schedCfg, err := LoadSchedulerConfig(opt.Model + "/scheduler")
	if err != nil {
		return nil, err
	}
	if p.sched, err = NewFlowMatchEuler(opt.Steps, schedCfg); err != nil {
		return nil, err
	}

	if p.tok, err = tokenizer.Load(opt.Model + "/tokenizer"); err != nil {
		return nil, fmt.Errorf("pipeline: tokenizer: %w", err)
	}
	encCfg, err := qwen.LoadConfig(opt.Model + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("pipeline: text encoder config: %w", err)
	}
	encSet, err := safetensors.OpenSet(opt.Model + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("pipeline: text encoder: %w", err)
	}
	p.enc, err = qwen.NewGPUEncoder(dev, encSet, encCfg, encCfg.EncoderLayers(), opt.MaxPrompt, nil)
	encSet.Close()
	if err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: text encoder: %w", err)
	}

	if p.cfg, err = dit.LoadConfig(opt.Model + "/transformer"); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: transformer config: %w", err)
	}
	ditSet, err := safetensors.OpenSet(opt.Model + "/transformer")
	if err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: transformer: %w", err)
	}
	defer ditSet.Close()
	if p.head, err = dit.LoadHead(ditSet, p.cfg); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: transformer head: %w", err)
	}
	if p.cfg.CapFeat != encCfg.HiddenSize {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: the transformer wants %d caption features, the encoder produces %d",
			p.cfg.CapFeat, encCfg.HiddenSize)
	}

	p.imgTokens = (p.latentH / p.head.Patch) * (p.latentW / p.head.Patch)
	p.imgTotal = dit.PadTo(p.imgTokens)
	p.maxUnified = p.imgTotal + dit.PadTo(opt.MaxPrompt)
	// The rotary table is replaced per run and per phase; this one only has
	// to be the right size and inside the axes' ranges.
	ids, err := p.head.PositionIDs(opt.MaxPrompt, p.latentH, p.latentW)
	if err != nil {
		p.Destroy()
		return nil, err
	}
	rope, err := dit.NewRoPE(ids, p.cfg.AxesDims, p.cfg.AxesLens, p.cfg.RopeTheta)
	if err != nil {
		p.Destroy()
		return nil, err
	}
	if p.stack, err = dit.NewGPUStack(dev, ditSet, p.cfg, rope, p.maxUnified, nil); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: transformer: %w", err)
	}
	nr := p.cfg.NRefiner
	p.noiseRefiner = seq(0, nr)
	p.contextRefiner = seq(nr, 2*nr)
	p.layers = seq(2*nr, 2*nr+p.cfg.NLayers)

	vaeCfg := vae.FluxConfig()
	p.scale, p.shift = vaeCfg.ScalingFactor, vaeCfg.ShiftFactor
	cpu, err := vae.LoadDecoder(opt.Model+"/vae", vaeCfg)
	if err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: vae: %w", err)
	}
	if p.dec, err = vae.NewGPUDecoder(dev, cpu, p.latentH, p.latentW); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: vae: %w", err)
	}
	return p, nil
}

// Destroy releases every device resource.
func (p *Pipeline) Destroy() {
	if p.dec != nil {
		p.dec.Destroy()
	}
	if p.stack != nil {
		p.stack.Destroy()
	}
	if p.enc != nil {
		p.enc.Destroy()
	}
	p.dec, p.stack, p.enc = nil, nil, nil
}

// Scheduler is the noise schedule the pipeline walks.
func (p *Pipeline) Scheduler() *FlowMatchEuler { return p.sched }

// Latent is the grid the transformer denoises: 16 channels at an eighth of
// the image's size.
func (p *Pipeline) Latent() (channels, height, width int) {
	return p.cfg.InChan, p.latentH, p.latentW
}

// Noise draws an initial latent from a seeded Gaussian.
func (p *Pipeline) Noise(seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, p.cfg.InChan*p.latentH*p.latentW)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// Encode runs the tokenizer and the text encoder, and returns the caption
// stream the transformer's context refiners take: the hidden state embedded
// by cap_embedder, padded with the learned token.
func (p *Pipeline) Encode(prompt string) (*dit.Mat, int, error) {
	ids, err := p.tok.EncodePrompt(prompt)
	if err != nil {
		return nil, 0, fmt.Errorf("pipeline: tokenizing: %w", err)
	}
	if len(ids) > p.opt.MaxPrompt {
		return nil, 0, fmt.Errorf("pipeline: the prompt is %d tokens, the encoder was built for %d",
			len(ids), p.opt.MaxPrompt)
	}
	hidden, err := p.enc.Forward(ids)
	if err != nil {
		return nil, 0, fmt.Errorf("pipeline: text encoder: %w", err)
	}
	cap, err := p.head.EmbedCaption(&dit.Mat{Rows: hidden.Rows, Cols: hidden.Cols, Data: hidden.Data})
	if err != nil {
		return nil, 0, err
	}
	return cap, len(ids), nil
}

// Generate runs the whole pipeline: prompt in, decoded image out. The image
// is [3, H, W] in [-1, 1], which is what the VAE produces and what a PNG
// writer has to map.
func (p *Pipeline) Generate(prompt string, seed int64, progress func(Step)) (*vae.Tensor, *Timings, error) {
	latents := p.Noise(seed)
	return p.GenerateFrom(prompt, latents, progress)
}

// GenerateFrom is Generate over an initial latent that is given rather than
// drawn, which is what makes an end-to-end comparison against diffusers
// possible: the two RNGs do not agree and nothing else in the pipeline is
// random.
func (p *Pipeline) GenerateFrom(prompt string, latents []float32, progress func(Step)) (*vae.Tensor, *Timings, error) {
	if want := p.cfg.InChan * p.latentH * p.latentW; len(latents) != want {
		return nil, nil, fmt.Errorf("pipeline: %d latents, want C*H*W = %d", len(latents), want)
	}
	tm := &Timings{}
	whole := time.Now()

	// --- the caption, once per image ---------------------------------
	t0 := time.Now()
	cap, tokens, err := p.Encode(prompt)
	if err != nil {
		return nil, nil, err
	}
	tm.Encode = time.Since(t0)
	tm.Tokens, tm.CapTotal = tokens, cap.Rows
	unified := p.imgTotal + cap.Rows
	tm.Unified = unified

	ids, err := p.head.PositionIDs(tokens, p.latentH, p.latentW)
	if err != nil {
		return nil, nil, err
	}
	if p.ctl.capPosFromZero {
		for i := p.imgTotal; i < len(ids); i++ {
			ids[i][0]--
		}
	}
	ropeUnified, err := dit.NewRoPE(ids, p.cfg.AxesDims, p.cfg.AxesLens, p.cfg.RopeTheta)
	if err != nil {
		return nil, nil, err
	}
	ropeCap, err := dit.NewRoPE(ids[p.imgTotal:], p.cfg.AxesDims, p.cfg.AxesLens, p.cfg.RopeTheta)
	if err != nil {
		return nil, nil, err
	}

	// The context refiners are the transformer's one phase that depends on
	// neither the timestep nor the latents -- diffusers builds them with
	// modulation=False -- so they run once per image and not once per step.
	t0 = time.Now()
	if err := p.stack.SetRoPE(ropeCap); err != nil {
		return nil, nil, err
	}
	if err := p.stack.Upload(cap, 0, cap.Rows); err != nil {
		return nil, nil, err
	}
	if err := p.stack.Run(p.contextRefiner); err != nil {
		return nil, nil, fmt.Errorf("pipeline: context refiners: %w", err)
	}
	refined := p.stack.Read(p.stack.TensorX(), p.cfg.Dim)
	tm.Caption = time.Since(t0)

	if err := p.stack.SetRoPE(ropeUnified); err != nil {
		return nil, nil, err
	}

	// --- the denoising loop -------------------------------------------
	for step := 0; step < p.sched.Steps(); step++ {
		t0 = time.Now()
		d := Step{Index: step, Latents: latents,
			Sigma: p.sched.Sigmas[step], NextSig: p.sched.Sigmas[step+1]}

		adaln, err := p.head.Timestep(p.sched.ModelT(step))
		if err != nil {
			return nil, nil, err
		}
		patches, err := p.head.Patchify(latents, p.latentH, p.latentW)
		if err != nil {
			return nil, nil, err
		}
		x, err := p.head.EmbedImage(patches)
		if err != nil {
			return nil, nil, err
		}
		d.Head = time.Since(t0)

		blocks := time.Now()
		if err := p.stack.SetAdaLN(adaln); err != nil {
			return nil, nil, err
		}
		// Phase one: the noise refiners over the image stream alone. The
		// unified rotary table's first rows are the image's, so the same
		// table serves both this phase and the layers.
		if err := p.stack.Upload(x, 0, p.imgTotal); err != nil {
			return nil, nil, err
		}
		if err := p.stack.Run(p.noiseRefiner); err != nil {
			return nil, nil, fmt.Errorf("pipeline: step %d noise refiners: %w", step, err)
		}
		// Phase three: the refined caption is appended behind the refined
		// image, in place, and the 30 layers run over the two.
		tail := refined
		if p.ctl.staleCaption {
			tail = cap
		}
		if err := p.stack.Upload(tail, p.imgTotal, unified); err != nil {
			return nil, nil, err
		}
		if err := p.stack.Run(p.layers); err != nil {
			return nil, nil, fmt.Errorf("pipeline: step %d layers: %w", step, err)
		}
		stream := p.stack.Read(p.stack.TensorX(), p.cfg.Dim)
		d.Blocks = time.Since(blocks)

		t0 = time.Now()
		// Only the image tokens reach the final layer; the caption's rows are
		// produced and discarded, as they are in diffusers' unpatchify.
		image := &dit.Mat{Rows: p.imgTokens, Cols: stream.Cols, Data: stream.Data[:p.imgTokens*stream.Cols]}
		final, err := p.head.Final(image, adaln)
		if err != nil {
			return nil, nil, err
		}
		out, err := p.head.Unpatchify(final, p.latentH, p.latentW)
		if err != nil {
			return nil, nil, err
		}
		// The pipeline negates the transformer's output before the step: the
		// model predicts the velocity towards the noise and the schedule runs
		// away from it.
		if !p.ctl.noNegate {
			for i := range out {
				out[i] = -out[i]
			}
		}
		if err := p.sched.Step(step, latents, out); err != nil {
			return nil, nil, err
		}
		d.Head += time.Since(t0)

		tm.Steps = append(tm.Steps, d.Blocks+d.Head)
		if progress != nil {
			progress(d)
		}
	}

	// --- the decode ----------------------------------------------------
	t0 = time.Now()
	latent := vae.NewTensor(1, p.cfg.InChan, p.latentH, p.latentW)
	for i, v := range latents {
		latent.Data[i] = float32(float64(v)/p.scale + p.shift)
	}
	img, err := p.dec.Apply(latent)
	if err != nil {
		return nil, nil, fmt.Errorf("pipeline: vae: %w", err)
	}
	tm.Decode = time.Since(t0)
	tm.Total = time.Since(whole)
	return img, tm, nil
}

// Residency reports what the pipeline holds on the device, by stage.
func (p *Pipeline) Residency() (encoder, transformer, activations int) {
	return p.enc.WeightBytes() + p.enc.ActivationBytes(),
		p.stack.WeightBytes(),
		p.stack.ActivationBytes() + p.dec.ActivationBytes()
}

func seq(lo, hi int) []int {
	out := make([]int, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, i)
	}
	return out
}
