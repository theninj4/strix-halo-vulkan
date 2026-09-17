package pipeline

import (
	"errors"
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

// Options configures a Pipeline.
//
// **Width and Height are a ceiling and a default, not a fixed size.** Every
// arena in the pipeline is sized at construction -- the text encoder for a
// maximum prompt, the transformer for a maximum unified sequence, the VAE
// decoder for the largest latent grid -- but none of the three graphs is
// *built* for one geometry: the transformer's run length is per dispatch
// (`GPUStack.Upload` states it), the decoder re-records its graph from the
// latent it is handed, and the rotary table is replaced per run anyway. So
// these two numbers decide the residency and the size a request that names
// none gets, and any smaller image runs in the same arenas. That is the same
// arrangement `-max-audio` gives parakeet, and for the same reason: a ceiling
// is a residency decision and a request's size is not.
type Options struct {
	Model     string // the checkpoint root, holding transformer/, vae/, ...
	Width     int    // the largest image, in pixels; a multiple of 16
	Height    int
	Steps     int // the schedule a request that names none gets
	MaxPrompt int // the longest prompt the text encoder is built for
	// CPUHead runs the patch embedder and the final layer on the host, which
	// is what stage 6 did. It is the slow path kept beside the device one
	// (PIPELINE.md's rule), and it is a flag rather than a test-only control
	// because the measurement that justifies stage 9 is the difference
	// between the two on the same image.
	CPUHead bool
	// UnfusedLayout keeps the two pure-layout dispatches stage 10 removed --
	// `pack v` and `narrow ctx` -- as their own passes. Same reason as
	// CPUHead: the slow path is what the fast one is measured against, and
	// the measurement is the difference between the two on the same image.
	UnfusedLayout bool
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

// SizeMultiple is what both sides of an image have to be a multiple of, for
// the reason above. It is exported because a server validating a request's
// size should say the number rather than discover it from an error.
const SizeMultiple = 2 * vaeScale

// ErrPromptTooLong is a prompt past the length the text encoder's arenas were
// built for.
//
// It is a value rather than a string because it is the one failure in here
// that the *caller* can fix: a server turns this into a 400 and everything
// else the pipeline returns into a 500, and telling them apart by matching on
// a message is a test that passes until someone rewords it.
var ErrPromptTooLong = errors.New("the prompt is longer than the text encoder was built for")

// promptTooLong carries both numbers as well as the sentinel, because what the
// caller needs in order to act on it is how far over they were.
func promptTooLong(limit, got int) error {
	return fmt.Errorf("pipeline: the prompt is %d tokens, the encoder was built for %d: %w",
		got, limit, ErrPromptTooLong)
}

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
	// gpuHead is the patch embedder and the final layer on the device (stage
	// 9). nil is the host path of stage 6, which is what it is measured
	// against and what runs if the shapes do not suit the kernels.
	gpuHead *dit.GPUHead

	dec *vae.GPUDecoder

	schedCfg *SchedulerConfig
	sched    *FlowMatchEuler // the default schedule, for Options.Steps
	scale    float64         // the VAE's scaling_factor
	shift    float64         // its shift_factor

	// The three phases' block indices, in the order the transformer runs
	// them. StackBlocks lays the stack out this way, so they follow from the
	// config's counts.
	noiseRefiner, contextRefiner, layers []int

	// ctl is empty in every real use except for Options.CPUHead; see controls.
	ctl controls

	// max is the geometry the arenas were built for and def the one a run
	// that names no size gets. Options sets them from the same pair, so they
	// are equal today; they are two fields rather than one because they are
	// two questions -- one is residency and the other is a default -- and
	// everything downstream already reports them separately.
	max, def   geom
	maxUnified int
}

// geom is one image's shape, all the way down: the pixels asked for, the
// latent grid the VAE decodes from, and the number of rows the transformer
// runs over. Everything in it follows from the width and the height, and it
// is derived per run rather than stored on the Pipeline because the Pipeline
// serves more than one size.
type geom struct {
	width, height    int
	latentH, latentW int
	imgTokens        int // the image stream before padding
	imgTotal         int // and after
}

// Request is one image to generate.
//
// Width, Height and Steps take the pipeline's own defaults when they are
// zero, which is what makes `Generate` a one-liner over this. Latents, when
// given, replaces the seeded noise -- and is written through, because the
// denoising loop integrates in place and the caller usually wants the final
// latent as well as the picture.
type Request struct {
	Prompt        string
	Width, Height int
	Steps         int
	Seed          int64
	Latents       []float32
	Progress      func(Step)
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
	// Width and Height are the image this run actually produced, which is
	// the request's size or the pipeline's default and not necessarily what
	// the arenas were built for.
	Width, Height int
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
	// cpuHead runs the patch embedder and the final layer on the host, which
	// is what stage 6 did and what stage 9's device path is compared against.
	// Not a breakage -- it is the slow path, kept (PIPELINE.md's rule).
	cpuHead bool
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
	// Checked before anything is staged: an unusable size should not cost a
	// 7 GB text encoder first.
	if err := checkSize(opt.Width, opt.Height); err != nil {
		return nil, err
	}
	p := &Pipeline{opt: opt}

	var err error
	if p.schedCfg, err = LoadSchedulerConfig(opt.Model + "/scheduler"); err != nil {
		return nil, err
	}
	if p.sched, err = NewFlowMatchEuler(opt.Steps, p.schedCfg); err != nil {
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

	if p.max, err = newGeom(opt.Width, opt.Height, p.head.Patch); err != nil {
		p.Destroy()
		return nil, err
	}
	p.def = p.max
	p.maxUnified = p.max.imgTotal + dit.PadTo(opt.MaxPrompt)
	// The rotary table is replaced per run and per phase; this one only has
	// to be the right size and inside the axes' ranges.
	ids, err := p.head.PositionIDs(opt.MaxPrompt, p.max.latentH, p.max.latentW)
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
	// The head on the device. It is a small object beside the stack -- 5 MB
	// of weights and four pipelines -- and it needs the stack to exist first,
	// because the residual stream it writes and reads is the stack's arena.
	p.ctl.cpuHead = opt.CPUHead
	p.stack.FuseLayout = !opt.UnfusedLayout
	if p.gpuHead, err = dit.NewGPUHead(p.stack, p.head); err != nil {
		p.Destroy()
		return nil, fmt.Errorf("pipeline: transformer head: %w", err)
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
	if p.dec, err = vae.NewGPUDecoder(dev, cpu, p.max.latentH, p.max.latentW); err != nil {
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
	// Before the stack: the head's pipelines are bound to the stack's arenas.
	if p.gpuHead != nil {
		p.gpuHead.Destroy()
	}
	if p.stack != nil {
		p.stack.Destroy()
	}
	if p.enc != nil {
		p.enc.Destroy()
	}
	p.dec, p.stack, p.enc, p.gpuHead = nil, nil, nil, nil
}

// Scheduler is the noise schedule the pipeline walks by default. A request
// that names its own step count gets a schedule of its own, built from the
// same config -- the schedule is a few dozen host floats, so a step count is
// a parameter and not a residency question the way a size is.
func (p *Pipeline) Scheduler() *FlowMatchEuler { return p.sched }

// Size is the image a request that names none gets, and MaxSize the largest
// the arenas hold -- the largest on *each side*; see geomFor.
func (p *Pipeline) Size() (width, height int)    { return p.def.width, p.def.height }
func (p *Pipeline) MaxSize() (width, height int) { return p.max.width, p.max.height }

// Steps is the default schedule's length.
func (p *Pipeline) Steps() int { return p.sched.Steps() }

// Latent is the grid the transformer denoises at the default size: 16
// channels at an eighth of the image's sides.
func (p *Pipeline) Latent() (channels, height, width int) {
	return p.cfg.InChan, p.def.latentH, p.def.latentW
}

// LatentFor is Latent for a size this pipeline can run but was not built
// around. A size it cannot run is an error rather than a rounded-up answer.
func (p *Pipeline) LatentFor(width, height int) (channels, h, w int, err error) {
	g, err := p.geomFor(width, height)
	if err != nil {
		return 0, 0, 0, err
	}
	return p.cfg.InChan, g.latentH, g.latentW, nil
}

// Noise draws an initial latent for the default size from a seeded Gaussian.
func (p *Pipeline) Noise(seed int64) []float32 { return p.noise(p.def, seed) }

// NoiseFor is Noise at a named size.
func (p *Pipeline) NoiseFor(width, height int, seed int64) ([]float32, error) {
	g, err := p.geomFor(width, height)
	if err != nil {
		return nil, err
	}
	return p.noise(g, seed), nil
}

func (p *Pipeline) noise(g geom, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, p.cfg.InChan*g.latentH*g.latentW)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// checkSize is the half of the geometry that needs no checkpoint: both sides
// a positive multiple of SizeMultiple, because the VAE upsamples 8x and the
// transformer's patch is 2.
func checkSize(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("pipeline: %dx%d is not an image", width, height)
	}
	if width%SizeMultiple != 0 || height%SizeMultiple != 0 {
		return fmt.Errorf("pipeline: %dx%d; both sides must be a multiple of %d",
			width, height, SizeMultiple)
	}
	return nil
}

// newGeom resolves a size all the way down to the transformer's row count.
func newGeom(width, height, patch int) (geom, error) {
	if err := checkSize(width, height); err != nil {
		return geom{}, err
	}
	g := geom{
		width: width, height: height,
		latentH: height / vaeScale, latentW: width / vaeScale,
	}
	g.imgTokens = (g.latentH / patch) * (g.latentW / patch)
	g.imgTotal = dit.PadTo(g.imgTokens)
	return g, nil
}

// geomFor resolves a request's size against the pipeline's. A zero side takes
// the default, and one side given alone takes the other from it -- which is
// what "512" means when a client says it.
//
// **The bound is each side, not the area, and that is a measurement rather
// than a conservative choice.** The transformer would take any shape with few
// enough rows -- imgTotal + caption against maxUnified -- so by the DiT's
// arithmetic a pipeline built for 1024x1024 could run 512x2048, the same 4096
// tokens. The VAE cannot: its fp16 arena holds *blocked* copies of each
// convolution's input (stage 8), and a block is padded on each axis
// separately, so the same area in a different shape needs more of it. A
// pipeline built for 256x256 measures 33 MB against a 32 MB arena at 128x512
// -- the identical area -- and fails in the decoder after the whole denoising
// loop has run.
//
// With both sides inside the ceiling every tensor in both graphs is smaller
// elementwise, and every padded tensor is too, since a per-axis round-up is
// monotone in the axis. So this one check is sufficient for all three graphs,
// and it is the one worth making up front: the alternative is discovering it
// eight denoising steps later.
func (p *Pipeline) geomFor(width, height int) (geom, error) {
	switch {
	case width == 0 && height == 0:
		return p.def, nil
	case width == 0:
		width = height
	case height == 0:
		height = width
	}
	if width == p.def.width && height == p.def.height {
		return p.def, nil
	}
	g, err := newGeom(width, height, p.head.Patch)
	if err != nil {
		return geom{}, err
	}
	if g.width > p.max.width || g.height > p.max.height {
		return geom{}, fmt.Errorf(
			"pipeline: %dx%d; these arenas were built for %dx%d and every side has to be inside it "+
				"(the VAE's blocked copies are padded per axis, so the same area in another shape does not fit)",
			width, height, p.max.width, p.max.height)
	}
	return g, nil
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
		return nil, 0, promptTooLong(p.opt.MaxPrompt, len(ids))
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

// Generate runs the whole pipeline at the default size: prompt in, decoded
// image out. The image is [3, H, W] in [-1, 1], which is what the VAE
// produces and what a PNG writer has to map.
func (p *Pipeline) Generate(prompt string, seed int64, progress func(Step)) (*vae.Tensor, *Timings, error) {
	return p.Run(Request{Prompt: prompt, Seed: seed, Progress: progress})
}

// GenerateFrom is Generate over an initial latent that is given rather than
// drawn, which is what makes an end-to-end comparison against diffusers
// possible: the two RNGs do not agree and nothing else in the pipeline is
// random. The latent's size decides the image's, so nothing has to say it
// twice.
func (p *Pipeline) GenerateFrom(prompt string, latents []float32, progress func(Step)) (*vae.Tensor, *Timings, error) {
	return p.Run(Request{Prompt: prompt, Latents: latents, Progress: progress})
}

// Run generates one image.
//
// The request's size and step count are resolved against the pipeline's own
// and then nothing else in here reads Options: every arena was sized for the
// ceiling at construction, the transformer's run length is stated per upload,
// the rotary table is rebuilt per run and the VAE re-records its graph from
// the latent it is given. So a 512x512 image out of a 1024x1024 pipeline is
// the same code path with smaller numbers in it, not a second one.
func (p *Pipeline) Run(req Request) (*vae.Tensor, *Timings, error) {
	g, err := p.geomFor(req.Width, req.Height)
	if err != nil {
		return nil, nil, err
	}
	sched := p.sched
	if req.Steps > 0 && req.Steps != sched.Steps() {
		if sched, err = NewFlowMatchEuler(req.Steps, p.schedCfg); err != nil {
			return nil, nil, err
		}
	}
	latents := req.Latents
	want := p.cfg.InChan * g.latentH * g.latentW
	if latents == nil {
		latents = p.noise(g, req.Seed)
	} else if len(latents) != want {
		// A latent that is the wrong length for the size asked for is almost
		// always a latent from another size, so the message names both.
		return nil, nil, fmt.Errorf("pipeline: %d latents for a %dx%d image, want C*H*W = %d",
			len(latents), g.width, g.height, want)
	}

	tm := &Timings{Width: g.width, Height: g.height}
	whole := time.Now()

	// --- the caption, once per image ---------------------------------
	t0 := time.Now()
	cap, tokens, err := p.Encode(req.Prompt)
	if err != nil {
		return nil, nil, err
	}
	tm.Encode = time.Since(t0)
	tm.Tokens, tm.CapTotal = tokens, cap.Rows
	unified := g.imgTotal + cap.Rows
	tm.Unified = unified

	ids, err := p.head.PositionIDs(tokens, g.latentH, g.latentW)
	if err != nil {
		return nil, nil, err
	}
	if p.ctl.capPosFromZero {
		for i := g.imgTotal; i < len(ids); i++ {
			ids[i][0]--
		}
	}
	ropeUnified, err := dit.NewRoPE(ids, p.cfg.AxesDims, p.cfg.AxesLens, p.cfg.RopeTheta)
	if err != nil {
		return nil, nil, err
	}
	ropeCap, err := dit.NewRoPE(ids[g.imgTotal:], p.cfg.AxesDims, p.cfg.AxesLens, p.cfg.RopeTheta)
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
	for step := 0; step < sched.Steps(); step++ {
		t0 = time.Now()
		d := Step{Index: step, Latents: latents,
			Sigma: sched.Sigmas[step], NextSig: sched.Sigmas[step+1]}

		adaln, err := p.head.Timestep(sched.ModelT(step))
		if err != nil {
			return nil, nil, err
		}
		patches, err := p.head.Patchify(latents, g.latentH, g.latentW)
		if err != nil {
			return nil, nil, err
		}
		var x *dit.Mat
		if p.gpuHead == nil || p.ctl.cpuHead {
			if x, err = p.head.EmbedImage(patches); err != nil {
				return nil, nil, err
			}
		}
		d.Head = time.Since(t0)

		blocks := time.Now()
		if err := p.stack.SetAdaLN(adaln); err != nil {
			return nil, nil, err
		}
		// Phase one: the noise refiners over the image stream alone. The
		// unified rotary table's first rows are the image's, so the same
		// table serves both this phase and the layers.
		//
		// On the device path the embedder *is* the upload: the [4096, 64]
		// patches go up as fp16 and the [4096, 3840] stream is written by a
		// GEMM that never leaves the device.
		if x != nil {
			if err := p.stack.Upload(x, 0, g.imgTotal); err != nil {
				return nil, nil, err
			}
		} else if err := p.gpuHead.Embed(patches, g.imgTotal); err != nil {
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
		if err := p.stack.Upload(tail, g.imgTotal, unified); err != nil {
			return nil, nil, err
		}
		if err := p.stack.Run(p.layers); err != nil {
			return nil, nil, fmt.Errorf("pipeline: step %d layers: %w", step, err)
		}
		d.Blocks = time.Since(blocks)

		t0 = time.Now()
		// Only the image tokens reach the final layer; the caption's rows are
		// produced and discarded, as they are in diffusers' unpatchify. On the
		// device path the tail runs over the padded image stream, because the
		// GEMM's tile divides that and not the token count, and the extra
		// rows are dropped here rather than not computed.
		var final *dit.Mat
		if p.gpuHead == nil || p.ctl.cpuHead {
			stream := p.stack.Read(p.stack.TensorX(), p.cfg.Dim)
			image := &dit.Mat{Rows: g.imgTokens, Cols: stream.Cols, Data: stream.Data[:g.imgTokens*stream.Cols]}
			if final, err = p.head.Final(image, adaln); err != nil {
				return nil, nil, err
			}
		} else {
			if final, err = p.gpuHead.Final(g.imgTotal); err != nil {
				return nil, nil, err
			}
			final = &dit.Mat{Rows: g.imgTokens, Cols: final.Cols, Data: final.Data[:g.imgTokens*final.Cols]}
		}
		out, err := p.head.Unpatchify(final, g.latentH, g.latentW)
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
		if err := sched.Step(step, latents, out); err != nil {
			return nil, nil, err
		}
		d.Head += time.Since(t0)

		tm.Steps = append(tm.Steps, d.Blocks+d.Head)
		if req.Progress != nil {
			req.Progress(d)
		}
	}

	// --- the decode ----------------------------------------------------
	t0 = time.Now()
	latent := vae.NewTensor(1, p.cfg.InChan, g.latentH, g.latentW)
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

// Residency reports what the pipeline holds on the device, by stage. The VAE
// is its own column because stage 8 gave it two arenas of each kind: the
// convolutions read fp16 fragment-tile copies of the filters and a blocked
// fp16 copy of each input, beside the fp32 originals the scalar kernels and
// the rest of the graph still use.
func (p *Pipeline) Residency() (encoder, transformer, vae, activations int) {
	vaeW32, vaeW16 := p.dec.WeightBytes()
	return p.enc.WeightBytes() + p.enc.ActivationBytes(),
		p.stack.WeightBytes(),
		vaeW32 + vaeW16,
		p.stack.ActivationBytes() + p.dec.ActivationBytes() + p.dec.F16ActivationBytes()
}

func seq(lo, hi int) []int {
	out := make([]int, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, i)
	}
	return out
}
