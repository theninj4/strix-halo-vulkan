package backend

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/pipeline"
	"strix-halo-vulkan/zimage/vae"
)

// ImageOptions is what cmd/serve's flags come to.
type ImageOptions struct {
	// Model is the Z-Image-Turbo checkpoint root, holding transformer/,
	// text_encoder/, vae/, tokenizer/ and scheduler/.
	Model string
	// Device is required: there is no host path for this pipeline. The DiT is
	// 94% of an image and 13 s of it on the device, so a CPU fallback would
	// not be a slow answer, it would be a timeout.
	Device *Device
	// Width and Height are the largest image the arenas are built for, and the
	// size a request that names none gets.
	//
	// This is a residency decision and the only one this vertical has. The
	// arenas are ~20.5 GB of weights plus 4.9 GB of activations at
	// 1024x1024, they are allocated once, and every smaller image runs inside
	// them -- so what this number buys is the ceiling and nothing else.
	Width, Height int
	// Steps is the schedule a request that names none gets. Z-Image-Turbo is
	// a distilled model whose NFE is 8.
	Steps int
	// MaxPrompt is the longest prompt the text encoder is built for.
	MaxPrompt int
	// Preview is a madebyollin/taef1 checkpoint directory. Naming one loads
	// the small decoder beside the big one and is what makes `stream: true`
	// answerable; leaving it empty makes that request a 400 that says so.
	//
	// It is a flag rather than always-on because it is residency: 4.9 MB of
	// weights, but 1.0 GB of activation arena at a 1024x1024 ceiling.
	Preview string
	// Edits holds the VAE's *encoder* resident, which is what makes
	// /v1/images/edits answerable; without it that endpoint is a 501 naming
	// this flag.
	//
	// Residency again, and a larger bill than the preview decoder's: 0.21 GB
	// of weights and, at a 1024x1024 ceiling, 1.5 GB of fp32 activation arena
	// and 0.26 GB of fp16. About 7% on top of what the pipeline already
	// holds, for a capability a server that only generates never uses.
	Edits bool
	// ID is the model id this backend answers to in /v1/models.
	ID string
}

const (
	defaultImageModelID = "z-image-turbo"
	defaultImageSize    = 1024
	defaultImageSteps   = 8
	defaultMaxPrompt    = 512
	// maxPartialImages is OpenAI's bound on `partial_images`, kept because
	// the cost argument agrees with it: a preview is 87 ms at 1024x1024
	// against a 14.5 s image, so three of them is 1.8% and a frame per step
	// would be 4.8%. It is a policy, not a limit of the decoder.
	maxPartialImages = 3
	// vaeScale is how much larger the image is than the latent grid, which is
	// how this adapter reads a resolved size back out of pipeline.LatentFor.
	// zimage/pipeline states it too, unexported, and the two agreeing is
	// checked by the pipeline itself refusing a mis-sized init image.
	vaeScale = 8
)

// Image is the z-image adapter: an api.ImageBackend over the pipeline
// cmd/zimage drives.
//
// **It is resident, and the resolution is not part of what is resident.**
// Construction stages 7.17 GB of text encoder, 12.54 GB of transformer and
// 0.30 GB of VAE, sizes 4.9 GB of activation arenas for the largest image the
// flags asked for, and then every request is arithmetic: the transformer
// states its run length per upload, the VAE re-records its graph from the
// latent it is handed, and the rotary table is rebuilt per run anyway. So a
// 512x512 request out of a 1024x1024 server costs a quarter of the tokens and
// stages nothing.
//
// One mutex guards the pipeline, and it is a correctness lock rather than a
// contention one: the stack's activation arena, the caption's refined rows and
// the decoder's arena are one image's, so two interleaved would not be slower,
// they would be one image.
type Image struct {
	opt ImageOptions
	id  string

	mu   sync.Mutex
	pipe *pipeline.Pipeline
	// rng draws a seed for a request that named none. It is seeded from the
	// process's default source rather than the clock, and it is here rather
	// than in the handler because the seed that was used is part of what the
	// model produced -- a client asking for the same image again sends it
	// back.
	rng *rand.Rand
}

// NewImage stages the whole pipeline on the device.
func NewImage(opt ImageOptions) (*Image, error) {
	if opt.ID == "" {
		opt.ID = defaultImageModelID
	}
	if opt.Width == 0 && opt.Height == 0 {
		opt.Width, opt.Height = defaultImageSize, defaultImageSize
	}
	if opt.Width == 0 {
		opt.Width = opt.Height
	}
	if opt.Height == 0 {
		opt.Height = opt.Width
	}
	if opt.Steps == 0 {
		opt.Steps = defaultImageSteps
	}
	if opt.MaxPrompt == 0 {
		opt.MaxPrompt = defaultMaxPrompt
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the image pipeline needs a device; there is no host path for it")
	}

	b := &Image{opt: opt, id: opt.ID, rng: rand.New(rand.NewSource(rand.Int63()))}
	err := opt.Device.Do(func(dev *vk.Device) error {
		p, err := pipeline.New(dev, pipeline.Options{
			Model: opt.Model, Width: opt.Width, Height: opt.Height,
			Steps: opt.Steps, MaxPrompt: opt.MaxPrompt, Preview: opt.Preview,
			Encoder: opt.Edits,
		})
		if err != nil {
			return err
		}
		b.pipe = p
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("backend: loading %s: %w", opt.Model, err)
	}
	return b, nil
}

// Models reports the one model this backend serves.
func (b *Image) Models() []api.Model {
	return []api.Model{{ID: b.id, Object: "model", OwnedBy: "local"}}
}

// Geometry is what this process was started for.
func (b *Image) Geometry() api.ImageGeometry {
	w, h := b.pipe.Size()
	mw, mh := b.pipe.MaxSize()
	geo := api.ImageGeometry{
		Width: w, Height: h,
		MaxWidth: mw, MaxHeight: mh,
		Multiple: pipeline.SizeMultiple,
		Steps:    b.pipe.Steps(),
		Previews: b.pipe.HasPreview(),
	}
	// Zero when there is no preview decoder, so the two fields cannot
	// disagree: a client reading `max_partial_images: 3` beside
	// `previews: false` would reasonably conclude it could ask for three.
	if geo.Previews {
		geo.MaxPartials = maxPartialImages
	}
	// Same argument as MaxPartials: a client reading a default strength
	// beside `edits: false` would reasonably conclude it could send one.
	if geo.Edits = b.pipe.HasEncoder(); geo.Edits {
		geo.DefaultStrength = pipeline.DefaultStrength
	}
	return geo
}

// Residency reports what the pipeline holds on the device, for the startup
// banner.
func (b *Image) Residency() (encoder, transformer, vaeWeights, activations int) {
	return b.pipe.Residency()
}

// Close releases the device residency.
func (b *Image) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pipe == nil {
		return
	}
	_ = b.opt.Device.Do(func(*vk.Device) error {
		b.pipe.Destroy()
		b.pipe = nil
		return nil
	})
}

// Generate renders one image.
//
// **The device lock is held for the whole run**, which is the one place this
// adapter differs from the language model's -- that one takes it per forward
// pass, so a 20 ms transcription does not wait behind a 20 s completion. The
// same argument applies here and the same fix is available (between two
// denoising steps there is no work in flight), but it would mean the pipeline
// calling back out around every submit, and what it buys is untested: a step
// at 1024x1024 is 1.7 s, so the granularity a speech request would actually
// get is seconds either way. It is written down in API.md rather than done.
//
// The context is checked on the way in and not again. A denoising step is a
// submit-and-fence with no cancellation point in it, so a client that hangs up
// mid-run still costs the run -- what it does not cost is the reply, which
// api.backendError drops.
func (b *Image) Generate(ctx context.Context, req *api.ImageRequest) (*api.ImageResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.pipe == nil {
		return nil, fmt.Errorf("the image pipeline is closed")
	}

	seed := int64(0)
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		seed = b.rng.Int63()
	}
	steps := req.Steps
	if steps == 0 {
		steps = b.pipe.Steps()
	}

	// The geometry is validated before the device is touched, so a bad size
	// costs a comparison rather than a text encoder run. The handler has
	// already checked the same thing against Geometry(); this is the check
	// that is true by construction rather than by agreement.
	// The latent grid is also how the resolved size is read back: a request
	// that named one side, or neither, has it filled in by the pipeline, and
	// an edit's picture has to be fitted to *that* rather than to what the
	// request said. Deriving it here rather than repeating geomFor's rules is
	// what stops the two from disagreeing.
	_, latentH, latentW, err := b.pipe.LatentFor(req.Width, req.Height)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	width, height := latentW*vaeScale, latentH*vaeScale

	// The init image, if this is an edit. Fitting it to the geometry is this
	// adapter's, because a resample is arithmetic on pixels; which geometry
	// it is fitted *to* was the handler's.
	var init *vae.Tensor
	first := 0
	if req.Init != nil {
		if !b.pipe.HasEncoder() {
			return nil, fmt.Errorf("this server holds no VAE encoder, so it cannot edit: %w", api.ErrUnsupported)
		}
		if init, err = fitImage(req.Init, width, height); err != nil {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		strength := req.Strength
		if strength == 0 {
			strength = pipeline.DefaultStrength
		}
		first = pipeline.StartStep(steps, strength)
	} else if req.Strength != 0 {
		return nil, fmt.Errorf("strength %g with no image to edit: %w", req.Strength, api.ErrUnsupported)
	}

	// Which steps a partial comes from. The handler said how many it wants;
	// this is the half that needs the step count, and it is why the decision
	// is here (api.ImageRequest.PartialImages says so).
	//
	// An edit runs only the tail, so the frames are spread over *that* --
	// spreading them over the whole schedule would put every one of them
	// before the run started and send none at all.
	var (
		partialAt map[int]int
		partialN  int
		perr      error
	)
	if req.Partial != nil && req.PartialImages > 0 && b.pipe.HasPreview() {
		partialAt = partialSteps(first, steps, min(req.PartialImages, maxPartialImages))
		partialN = len(partialAt)
	}
	progress := func(st pipeline.Step) {
		idx, want := partialAt[st.Index]
		if !want || perr != nil || st.Preview == nil {
			return
		}
		t, err := st.Preview()
		if err != nil {
			perr = fmt.Errorf("preview at step %d: %w", st.Index, err)
			return
		}
		perr = req.Partial(api.ImagePartial{
			Index: idx, Step: st.Index, Steps: steps,
			Image: toRGBA(t), Width: t.W, Height: t.H,
		})
	}
	if partialN == 0 {
		progress = nil
	}

	var img *vae.Tensor
	var tm *pipeline.Timings
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		img, tm, err = b.pipe.Run(pipeline.Request{
			Prompt: req.Prompt, Width: req.Width, Height: req.Height,
			Steps: steps, Seed: seed, Progress: progress,
			Init: init, Strength: req.Strength,
		})
		return err
	})
	// A frame the client could not be given -- they hung up -- is the failure,
	// not whatever the rest of the run did, so it is reported first.
	if perr != nil {
		return nil, perr
	}
	if err != nil {
		// A prompt past the text encoder's arena is the client's fault and
		// fixable by them, which is the whole of what ErrUnsupported means.
		if errors.Is(err, pipeline.ErrPromptTooLong) {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		return nil, err
	}
	return &api.ImageResult{
		Image:  toRGBA(img),
		Width:  tm.Width,
		Height: tm.Height,
		Steps:  len(tm.Steps),
		Seed:   seed,
	}, nil
}

// toRGBA maps the decoder's output to an image.
//
// The VAE produces [3, H, W] in [-1, 1] and the clamp is load-bearing rather
// than defensive: the last denoising step lands near the range and not inside
// it, so a few pixels of a normal image are past ±1 and an unclamped
// conversion wraps them to the opposite end -- a white highlight comes out
// black.
func toRGBA(t *vae.Tensor) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, t.W, t.H))
	plane := t.H * t.W
	for y := 0; y < t.H; y++ {
		for x := 0; x < t.W; x++ {
			var px [3]uint8
			for ch := 0; ch < 3; ch++ {
				v := (float64(t.Data[ch*plane+y*t.W+x]) + 1) * 127.5
				px[ch] = uint8(math.Round(math.Min(255, math.Max(0, v))))
			}
			img.Set(x, y, color.RGBA{px[0], px[1], px[2], 255})
		}
	}
	return img
}

// partialSteps picks which denoising steps a partial image comes from: n
// frames spread evenly over the steps that will run, as a map from step index
// to the frame's own index.
//
// `first` is where the run starts -- 0 for a generation and the SDEdit start
// for an edit -- and it is a parameter rather than an assumption because an
// edit at strength 0.5 runs the last four steps of eight, and frames spread
// over all eight would all fall before it began.
//
// The last step is excluded, and that is the only judgement in here. Its
// denoised estimate *is* the final latent -- the terminal sigma is zero -- so
// a partial there would be the finished image sent twice, once through the
// preview decoder and once through the real one, and the two are not quite
// the same picture. Everything else follows: with fewer steps than frames
// asked for there are simply fewer frames, and with one step there are none.
func partialSteps(first, steps, n int) map[int]int {
	run := steps - first
	if run < 2 || n <= 0 {
		return nil
	}
	out := make(map[int]int, n)
	last := steps - 2 // the last step a partial may come from
	idx := 0
	for j := 1; j <= n; j++ {
		k := first + j*run/(n+1) - 1
		if k < first {
			k = first
		}
		if k > last {
			k = last
		}
		if _, seen := out[k]; seen {
			continue
		}
		out[k] = idx
		idx++
	}
	return out
}
