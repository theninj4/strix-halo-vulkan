package backend

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/flux3/pipeline"
	"strix-halo-vulkan/flux3/vae"
	"strix-halo-vulkan/vk"
)

// Flux3Options is what cmd/serve's flags come to for FLUX 3 Image Klein
// (research/flux3-vertical.md X6).
type Flux3Options struct {
	// Model is the checkpoint root in diffusers layout (transformer/,
	// text_encoder/, vae/); Preview is taef2's directory, "" for taef2/
	// beside it.
	Model, Preview string
	Device         *Device
	// Width and Height are the staging: a request's default size, and, as an
	// area, its ceiling -- any shape of no larger area is served. At most
	// pipeline.MaxPixels (1232², the decoder's single-buffer arena).
	Width, Height int
	// Steps is the default schedule; the checkpoint is distilled to 16.
	Steps int
	// MaxPrompt is the longest prompt, in tokens, the encoder is staged for.
	MaxPrompt int
	// Refs is how many reference images an edit may carry, 0 for no edits
	// (pipeline.Config.Refs).
	Refs int
	ID   string
	Swap      *Swap
}

const defaultFlux3ID = "flux-3-image-klein"

// Flux3 is FLUX 3 Image Klein's text-to-image: an api.ImageBackend over
// flux3/pipeline, taking turns in the swap slot like Image. Staging is
// ~22 GB (fp16: 6.5 encoder, 10.6 transformer, 0.3 decoders, 5 activations
// at 1024²; edits add the VAE encoder and their rows) and ~11 s from a warm
// page cache. It answers generations and, with Refs, edits (X7).
//
// One mutex guards the pipeline, a correctness lock as in Image: the DiT's
// arenas are one image's.
type Flux3 struct {
	opt Flux3Options

	mu   sync.Mutex
	pipe *pipeline.GPU // nil while another vertical holds the slot
	geo  api.ImageGeometry
	res  [4]int
	rng  *rand.Rand
}

// NewFlux3 stages the pipeline through the slot: resident when this returns.
func NewFlux3(opt Flux3Options) (*Flux3, error) {
	if opt.ID == "" {
		opt.ID = defaultFlux3ID
	}
	if opt.Steps == 0 {
		opt.Steps = pipeline.DefaultSteps
	}
	if opt.Device == nil || opt.Swap == nil {
		return nil, fmt.Errorf("backend: FLUX 3 needs a device and a swap slot")
	}
	if opt.Width*opt.Height > pipeline.MaxPixels {
		return nil, fmt.Errorf("backend: FLUX 3 at %dx%d is past the %d-pixel ceiling (1232²)", opt.Width, opt.Height, pipeline.MaxPixels)
	}
	b := &Flux3{opt: opt, rng: rand.New(rand.NewSource(rand.Int63()))}
	b.geo = api.ImageGeometry{
		Width: opt.Width, Height: opt.Height, MaxPixels: opt.Width * opt.Height,
		Multiple: pipeline.SizeMultiple, Steps: opt.Steps,
		Previews: true, MaxPartials: maxPartialImages,
		Edits: opt.Refs > 0, MaxRefs: opt.Refs,
	}
	if err := opt.Swap.Load(b); err != nil {
		return nil, err
	}
	return b, nil
}

// SwapName is the vertical's name in the swap log.
func (b *Flux3) SwapName() string { return "image (flux 3)" }

// Load stages the pipeline; it submits nothing, so it takes no device lock.
func (b *Flux3) Load() error {
	p, err := pipeline.NewGPU(b.opt.Device.dev, pipeline.Config{
		Model: b.opt.Model, Preview: b.opt.Preview,
		MaxWidth: b.opt.Width, MaxHeight: b.opt.Height, MaxPrompt: b.opt.MaxPrompt, Refs: b.opt.Refs,
	})
	if err != nil {
		return fmt.Errorf("backend: loading %s: %w", b.opt.Model, err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pipe = p
	b.res[0], b.res[1], b.res[2], b.res[3] = p.Residency()
	return nil
}

// Unload frees the pipeline; the slot guarantees nothing runs on it.
func (b *Flux3) Unload() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pipe != nil {
		b.pipe.Destroy()
		b.pipe = nil
	}
}

// Close releases the device residency.
func (b *Flux3) Close() { b.Unload() }

// Models reports the one model this backend serves.
func (b *Flux3) Models() []api.Model {
	return []api.Model{{ID: b.opt.ID, Object: "model", OwnedBy: "local"}}
}

// Geometry is what the process was started for, resident or not.
func (b *Flux3) Geometry() api.ImageGeometry { return b.geo }

// Residency is the staging's device bytes, for the startup banner.
func (b *Flux3) Residency() (encoder, transformer, vaes, activations int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.res[0], b.res[1], b.res[2], b.res[3]
}

// Generate renders one image, or an edit when the request carries
// references. The device lock is held for the run; the context is checked
// between steps, so a client that hangs up stops paying within one step
// (~1.15 s at 1024², ~2.5 s for a one-reference edit).
func (b *Flux3) Generate(ctx context.Context, req *api.ImageRequest) (*api.ImageResult, error) {
	if len(req.Init) > b.opt.Refs {
		if b.opt.Refs == 0 {
			return nil, fmt.Errorf("this server was not started for editing; restart it with -edits N: %w", api.ErrUnsupported)
		}
		return nil, fmt.Errorf("%d reference images; this server is started for %d: %w", len(req.Init), b.opt.Refs, api.ErrUnsupported)
	}
	// Decision 4: this VAE is RGB. Qwen-Image-2.1 painted alpha natively;
	// FLUX 3 has no mechanism for it, and a matte cut from an opaque picture
	// would be a different feature than the one the field names.
	if req.Transparent {
		return nil, fmt.Errorf("background \"transparent\" is not supported by %s, whose autoencoder is RGB; "+
			"ask for an opaque image: %w", b.opt.ID, api.ErrUnsupported)
	}
	release, err := b.opt.Swap.Hold(ctx, b)
	if err != nil {
		return nil, err
	}
	defer release()
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.pipe == nil {
		return nil, fmt.Errorf("the image pipeline is closed")
	}
	// An edit that names no size is the pipeline's to resolve: its first
	// reference's prepared size, cropped back to the fitted box.
	width, height := 0, 0
	if len(req.Init) == 0 || req.Width != 0 || req.Height != 0 {
		var err error
		if width, height, err = b.pipe.Resolve(req.Width, req.Height); err != nil {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
	}
	seed := b.rng.Int63()
	if req.Seed != nil {
		seed = *req.Seed
	}
	steps := req.Steps
	if steps == 0 {
		steps = b.opt.Steps
	}

	var perr error
	var progress func(pipeline.Step)
	if req.Partial != nil && req.PartialImages > 0 {
		partialAt := partialSteps(0, steps, min(req.PartialImages, maxPartialImages))
		progress = func(st pipeline.Step) {
			idx, want := partialAt[st.Index]
			if !want || perr != nil {
				return
			}
			t, err := st.Preview()
			if err != nil {
				perr = fmt.Errorf("preview at step %d: %w", st.Index, err)
				return
			}
			frame := opaqueImage(t)
			drawProgress(frame, st.Index+1, steps)
			perr = req.Partial(api.ImagePartial{
				Index: idx, Step: st.Index, Steps: steps, Image: frame, Width: t.W, Height: t.H,
			})
		}
	}

	var (
		img *vae.Tensor
		tm  *pipeline.Timings
	)
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		img, tm, err = b.pipe.Generate(ctx, pipeline.Request{
			Prompt: req.Prompt, Width: width, Height: height, Steps: steps, Seed: seed, Progress: progress,
			References: req.Init,
		})
		return err
	})
	if perr != nil {
		return nil, perr
	}
	if err != nil {
		if errors.Is(err, pipeline.ErrPromptTooLong) || errors.Is(err, pipeline.ErrReferences) {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		return nil, err
	}
	return &api.ImageResult{Image: opaqueImage(img), Width: tm.Width, Height: tm.Height, Steps: tm.Steps, Seed: seed}, nil
}

// opaqueImage is a decoded [1, 3, H, W] tensor in [-1, 1] as 8-bit RGB,
// rounded as diffusers' postprocess rounds (vae.RGB8).
func opaqueImage(t *vae.Tensor) *image.NRGBA {
	rgb := vae.RGB8(t)
	img := image.NewNRGBA(image.Rect(0, 0, t.W, t.H))
	for i := 0; i < t.W*t.H; i++ {
		copy(img.Pix[i*4:], rgb[i*3:i*3+3])
		img.Pix[i*4+3] = 255
	}
	return img
}
