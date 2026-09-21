package backend

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"math/rand"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/qimage/pipeline"
	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/vk"
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
	// **What a request has to fit inside is this rectangle's area, not its
	// sides.** Every arena here is sized by the pixel count alone (measured:
	// `qimage/vae`'s TestArenaShape), so a server started at 1024x1024 serves
	// 1344x768 and 2048x512 as readily as its own square, and a 16:9 request
	// gets 16:9's share of the arenas instead of 56% of it.
	//
	// **The area has a hard limit of 1,403,584 pixels** -- 1184x1184, or
	// 1536x864 at 16:9 -- and it is not a budget: the VAE decoder's
	// activation arena is one storage buffer and this device caps one at
	// 4 GiB - 4 (IMAGE.md Q5g/Q6). Anything larger is refused at startup,
	// naming the number.
	Width, Height int
	// Steps is the schedule a request that names none gets. Unlike
	// Z-Image-Turbo's 8, Qwen-Image-2.1 is not a distillation and 40 is what
	// diffusers runs.
	Steps int
	// MaxPrompt is the longest prompt the text encoder is built for.
	MaxPrompt int
	// Refs is how many reference images an edit may carry, and zero means
	// this server does not answer /v1/images/edits at all.
	//
	// It is residency, which is why it is a number and not a bool. Editing
	// stages two more models -- the 27-layer vision tower and the VAE's
	// encoder, ~1.4 GB together -- and every reference then adds its latent
	// rows to the transformer's prefix and its own share of the prefix KV
	// cache, which at one 1024² reference is ~2.1 GB. Ten of them is the
	// model's limit and a different machine's decision.
	//
	// There is no Preview field any more: previews were residency under
	// Z-Image (taef1 was 1.0 GB of activation arena) and are 260 float32s
	// here, so they are always on. A flag that cannot be turned off is not a
	// flag.
	Refs int
	// CondSize is the square whose *area* every reference image is resized
	// to before it is encoded, and the default output size of an edit that
	// names none. It is diffusers' `output_resolution`; zero takes the
	// model's own default of 1024.
	CondSize int
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
	b := &Image{opt: opt, id: opt.ID, rng: rand.New(rand.NewSource(rand.Int63()))}
	err := opt.Device.Do(func(dev *vk.Device) error {
		p, err := pipeline.New(dev, pipeline.Options{
			Model: opt.Model, Width: opt.Width, Height: opt.Height,
			Steps: opt.Steps, MaxPrompt: opt.MaxPrompt,
			Refs: opt.Refs, CondSize: opt.CondSize,
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
	// Previews are unconditional -- the decoder is a constant matrix, not
	// residency -- so unlike under Z-Image there is no flag for the two
	// fields to disagree about. Edits report the *count* rather than a
	// strength, because 2.1 has no strength: a client reads MaxRefs to know
	// how many pictures it may send.
	return api.ImageGeometry{
		Width: w, Height: h,
		MaxPixels:   b.pipe.MaxPixels(),
		Multiple:    pipeline.SizeMultiple,
		Steps:       b.pipe.Steps(),
		Previews:    true,
		MaxPartials: maxPartialImages,
		Edits:       b.pipe.Refs() > 0,
		MaxRefs:     b.pipe.Refs(),
	}
}

// Residency reports what the pipeline holds on the device, for the startup
// banner.
func (b *Image) Residency() (encoder, transformer, vaeWeights, activations int) {
	return b.pipe.Residency()
}

// EditResidency is what the edit half holds: the vision tower's and the VAE
// encoder's weights, and the transformer's prefix KV cache. Zero when this
// server does not edit.
func (b *Image) EditResidency() (weights, cache int) { return b.pipe.EditResidency() }

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
// **The device lock is held for the whole run, and the context is what cuts
// it short.** A denoising *step* is a submit-and-fence with nothing to
// abandon inside it, but between two steps there is nothing in flight, and
// the same is true between the VAE's submit batches. So the context reaches
// all the way down (`pipeline.Run`, `Edit`, `Decode`, and the vision tower
// and VAE encoder an edit runs first) and a client that hangs up stops paying
// within one step — about 2.2 s of a 92 s image, or a quarter-second of the
// decode — instead of paying for the whole picture.
//
// That matters past the wasted arithmetic: Device.Do is one lock for every
// vertical in the process, so an abandoned image used to block the queued
// speech and embedding requests behind it for its full duration too.
func (b *Image) Generate(ctx context.Context, req *api.ImageRequest) (*api.ImageResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.pipe == nil {
		return nil, fmt.Errorf("the image pipeline is closed")
	}

	if len(req.Init) > 0 && b.pipe.Refs() == 0 {
		return nil, fmt.Errorf("this server was not started for editing; restart it with -edits N: %w",
			api.ErrUnsupported)
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
	//
	// An edit may name neither, and then the size is *not* the server's
	// default: it is the last reference image's aspect ratio at the model's
	// condition area, which only the pipeline can work out. So zero is
	// passed through rather than resolved here.
	width, height := 0, 0
	if len(req.Init) == 0 || req.Width != 0 || req.Height != 0 {
		_, latentH, latentW, err := b.pipe.LatentFor(req.Width, req.Height)
		if err != nil {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		width, height = latentW*pipeline.VAEScale, latentH*pipeline.VAEScale
	}

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
		img *qvae.Tensor
		tm  *pipeline.Timings
	)
	refs, err := nrgbaRefs(req.Init)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		if len(refs) > 0 {
			img, tm, err = b.pipe.Edit(ctx, pipeline.EditRequest{
				Prompt: prompt, Images: refs, Width: width, Height: height,
				Steps: req.Steps, Seed: seed, Progress: progress,
			})
			return err
		}
		img, tm, err = b.pipe.Run(ctx, pipeline.Request{
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
// `first` is where the run starts, and under Qwen-Image-2.1 it is 0 for every
// request: an edit conditions on its references rather than starting part-way
// up the schedule, so there is no truncated run to offset. It stays a
// parameter because it is the one thing that would change if a partially
// renoised path ever came back, and because a constant argument at the two
// call sites is cheaper to read than a comment explaining its absence.
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

// nrgbaRefs converts the decoded reference images to the 8-bit NRGBA the
// pipeline's resampler and compositor work in.
//
// It is a conversion and not a resize: the resize is the *model's*, exact to
// the 8-bit level against Pillow's Lanczos (IMAGE.md Q8.3b), and doing any of
// it here would be a second, worse one. `image.Decode` already hands back
// NRGBA for a PNG with alpha, and this is a copy for everything else --
// including the JPEG case, where the alpha it fills in is opaque, which is
// what a JPEG means.
func nrgbaRefs(in []image.Image) ([]*image.NRGBA, error) {
	if len(in) == 0 {
		return nil, nil
	}
	out := make([]*image.NRGBA, len(in))
	for i, src := range in {
		b := src.Bounds()
		if b.Dx() <= 0 || b.Dy() <= 0 {
			return nil, fmt.Errorf("reference image %d is empty", i)
		}
		if n, ok := src.(*image.NRGBA); ok && n.Bounds().Min == (image.Point{}) {
			out[i] = n
			continue
		}
		dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
		out[i] = dst
	}
	return out, nil
}
