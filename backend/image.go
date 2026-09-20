package backend

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/qimage/pipeline"
	"strix-halo-vulkan/vk"
	zvae "strix-halo-vulkan/zimage/vae"
)

// ImageOptions is what cmd/serve's flags come to.
type ImageOptions struct {
	// Model is the Qwen-Image-2.1 checkpoint root, holding transformer/,
	// text_encoder/, vae/, processor/ and scheduler/.
	Model string
	// Device is required: there is no host path for this pipeline. The DiT is
	// 2.27 s a step on the device and hours on the host, so a CPU fallback
	// would not be a slow answer, it would be a timeout.
	Device *Device
	// Width and Height are the largest image the arenas are built for, and the
	// size a request that names none gets.
	//
	// This is a residency decision and the only one this vertical has. The
	// arenas are ~27 GB, they are allocated once, and every smaller image runs
	// inside them -- so what this number buys is the ceiling and nothing else.
	//
	// **The ceiling has a hard limit at 1184x1184** and it is not a budget:
	// the VAE decoder's activation arena is one storage buffer and this device
	// caps one at 4 GiB - 4 (IMAGE.md Q5g/Q6). Anything larger is refused at
	// startup, naming the number.
	Width, Height int
	// Steps is the schedule a request that names none gets. Unlike
	// Z-Image-Turbo's 8, Qwen-Image-2.1 is not a distillation and 40 is what
	// diffusers runs.
	Steps int
	// MaxPrompt is the longest prompt the text encoder is built for.
	MaxPrompt int
	// Edits is the one capability this vertical had under Z-Image and does
	// not have yet under Qwen-Image-2.1. It is kept as a field so that a
	// server started with the old flag fails at startup with the stage that
	// owes it, rather than silently serving without.
	//
	// There is no Preview field any more: previews were residency under
	// Z-Image (taef1 was 1.0 GB of activation arena) and are 260 float32s
	// here, so they are always on. A flag that cannot be turned off is not a
	// flag.
	Edits bool
	// ID is the model id this backend answers to in /v1/models.
	ID string
}

const (
	defaultImageModelID = "qwen-image-2.1"
	// maxPartialImages is OpenAI's bound on `partial_images`, kept because
	// the cost argument agrees with it -- though for a different reason than
	// it did under Z-Image. There a frame was 87 ms of taef1 against a
	// 1.67 s step, so three were 1.8% of the image. Here the decode is 260
	// multiply-adds a latent pixel on the host, and what actually costs
	// anything is *encoding* the PNG and writing it to a client. Three is a
	// policy about the stream, not a limit of the decoder.
	maxPartialImages = 3
)

// Image is the Qwen-Image-2.1 adapter: an api.ImageBackend over
// qimage/pipeline.
//
// **It is resident, and the resolution is not part of what is resident.**
// Construction stages ~13.2 GB of text encoder, ~13.3 GB of transformer and
// ~1.0 GB of VAE, sizes the activation arenas for the largest image the flags
// asked for, and then every request is arithmetic: the transformer states its
// run length per image, the VAE re-records its graph from the latent it is
// handed. So a 512x512 request out of a 1024x1024 server costs a quarter of
// the tokens and stages nothing.
//
// One mutex guards the pipeline, and it is a correctness lock rather than a
// contention one: the DiT's prefix KV cache and both activation arenas are
// one image's, so two interleaved would not be slower, they would be one
// image.
type Image struct {
	opt ImageOptions
	id  string

	mu   sync.Mutex
	pipe *pipeline.Pipeline
	// rng draws a seed for a request that named none. It is here rather than
	// in the handler because the seed that was used is part of what the model
	// produced -- a client asking for the same image again sends it back.
	rng *rand.Rand
}

// NewImage stages the whole pipeline on the device.
func NewImage(opt ImageOptions) (*Image, error) {
	if opt.ID == "" {
		opt.ID = defaultImageModelID
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the image pipeline needs a device; there is no host path for it")
	}
	// The one capability Z-Image had and this model's port does not yet.
	// Refused rather than ignored: a server told to edit would otherwise come
	// up and answer every edit request with a 501.
	if opt.Edits {
		return nil, fmt.Errorf("backend: image editing is not ported to Qwen-Image-2.1 yet " +
			"(IMAGE.md Q8: 2.1 edits by conditional generation over a vision tower, not by SDEdit, " +
			"and the tower is unported); start without -edits")
	}

	b := &Image{opt: opt, id: opt.ID, rng: rand.New(rand.NewSource(rand.Int63()))}
	err := opt.Device.Do(func(dev *vk.Device) error {
		p, err := pipeline.New(dev, pipeline.Options{
			Model: opt.Model, Width: opt.Width, Height: opt.Height,
			Steps: opt.Steps, MaxPrompt: opt.MaxPrompt,
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
	// Previews are unconditional -- the decoder is a constant matrix, not
	// residency -- so unlike under Z-Image there is no flag for the two
	// fields to disagree about. Edits stay false with DefaultStrength zero
	// beside them, so a client reading a default strength cannot conclude it
	// may send one.
	return api.ImageGeometry{
		Width: w, Height: h,
		MaxWidth: mw, MaxHeight: mh,
		Multiple:    pipeline.SizeMultiple,
		Steps:       b.pipe.Steps(),
		Previews:    true,
		MaxPartials: maxPartialImages,
	}
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

// transparentPrefix and transparentSuffix are the model card's recommended
// phrasing for an RGBA image, verbatim. Transparency in this model is asked
// for in the *prompt* -- the VAE emits four channels either way -- so
// `background: "transparent"` is a prompt rewrite plus a decision about the
// alpha plane, and not a mode.
const (
	transparentPrefix = "This is an RGBA image with transparency. "
	transparentSuffix = " The image has alpha channel and the background is transparent."
)

// Generate renders one image.
//
// **The device lock is held for the whole run.** A denoising step is a
// submit-and-fence with no cancellation point in it, so the context is
// checked on the way in and not again: a client that hangs up mid-run still
// costs the run, and what it stops costing is the encoding and the writing.
func (b *Image) Generate(ctx context.Context, req *api.ImageRequest) (*api.ImageResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.pipe == nil {
		return nil, fmt.Errorf("the image pipeline is closed")
	}

	// The two request shapes this model does not yet answer. Both are the
	// client's to avoid and both name the stage that owes them, which is the
	// whole of what api.ErrUnsupported means.
	if req.Init != nil {
		return nil, fmt.Errorf("this server cannot edit images: Qwen-Image-2.1 edits by conditional "+
			"generation over a vision tower, which is unported (IMAGE.md Q8): %w", api.ErrUnsupported)
	}
	if req.Strength != 0 {
		return nil, fmt.Errorf("strength %g with no image to edit: %w", req.Strength, api.ErrUnsupported)
	}

	seed := int64(0)
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		seed = b.rng.Int63()
	}

	// The geometry is validated before the device is touched, so a bad size
	// costs a comparison rather than a text encoder run. The handler has
	// already checked the same thing against Geometry(); this is the check
	// that is true by construction rather than by agreement, and it is also
	// how the resolved size is read back for a request that named one side or
	// neither.
	_, latentH, latentW, err := b.pipe.LatentFor(req.Width, req.Height)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	width, height := latentW*pipeline.VAEScale, latentH*pipeline.VAEScale

	prompt := req.Prompt
	if req.Transparent {
		prompt = transparentPrefix + prompt + transparentSuffix
	}

	// Which steps a partial comes from. The handler said how many it wants;
	// this is the half that needs the step count, and it is why the decision
	// is here (api.ImageRequest.PartialImages says so).
	steps := req.Steps
	if steps == 0 {
		steps = b.pipe.Steps()
	}
	var (
		partialAt map[int]int
		perr      error
	)
	if req.Partial != nil && req.PartialImages > 0 {
		partialAt = partialSteps(0, steps, min(req.PartialImages, maxPartialImages))
	}
	var progress func(pipeline.Step)
	if len(partialAt) > 0 {
		progress = func(st pipeline.Step) {
			idx, want := partialAt[st.Index]
			// A frame the client could not take stops any *further* frames.
			// It does not stop the run: a denoising step is a
			// submit-and-fence with no cancellation point in it.
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
				// A preview is opaque whatever the request asked for. Its
				// alpha comes from the same fitted matrix as its colour and
				// is the least reliable of the four channels; a transparent
				// in-progress frame would show the client holes that the
				// finished image does not have.
				Image: pipeline.ToImage(t, true),
				Width: t.W, Height: t.H,
			})
		}
	}

	var (
		img *zvae.Tensor
		tm  *pipeline.Timings
	)
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		img, tm, err = b.pipe.Run(pipeline.Request{
			Prompt: prompt, Width: width, Height: height,
			Steps: req.Steps, Seed: seed, Progress: progress,
		})
		return err
	})
	// A frame the client could not be given -- they hung up -- is the
	// failure, not whatever the rest of the run did, so it is reported first.
	if perr != nil {
		return nil, perr
	}
	if err != nil {
		// A prompt past the text encoder's arena is the client's fault and
		// fixable by them.
		if errors.Is(err, pipeline.ErrPromptTooLong) {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		return nil, err
	}

	return &api.ImageResult{
		// The alpha plane is kept only when the caller asked for
		// transparency. It exists either way -- the VAE has four channels --
		// but on an ordinary prompt it is whatever the model painted there,
		// and compositing over white is what makes an opaque request opaque.
		Image:  pipeline.ToImage(img, !req.Transparent),
		Width:  tm.Width,
		Height: tm.Height,
		Steps:  len(tm.Steps),
		Seed:   seed,
	}, nil
}

// partialSteps picks which denoising steps an in-progress frame comes from:
// n frames spread evenly over the steps that will run, as a map from step
// index to the frame's own index.
//
// It is host arithmetic with three edges -- fewer steps than frames asked
// for, a one-step schedule, and the last step -- and none of them would show
// up in an image, which is why it has a test of its own. It survived the
// migration untouched: the policy did not change with the model.
//
// `first` is where the run starts. It is 0 for every generation; it is a
// parameter because Q8's edits do not start at 0.
//
// The last step is excluded, and that is the only judgement in here. Its
// denoised estimate *is* the final latent -- the terminal sigma is the
// scheduler's shift_terminal -- so a partial there would be the finished
// image sent twice, through two different decoders.
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
