package backend

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/h3/pipeline"
	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/vk"
)

// VideoOptions is what cmd/serve's flags come to.
type VideoOptions struct {
	// Model is the MiniMax-H3 checkpoint root (transformer/, text_encoder/,
	// vae/, audio_vae/, tokenizer/).
	Model string
	// Device is required: a forward is 35 s on the device at 480p.
	Device *Device
	// MaxPrompt caps a prompt's tokens (0: pipeline.DefaultMaxPrompt).
	MaxPrompt int
	// ID is the model id this backend answers to.
	ID string
}

const defaultVideoModelID = "minimax-h3"

// Video is the MiniMax-H3 adapter: an api.VideoBackend over h3/pipeline.
//
// **Nothing is resident between requests.** A request stages the text
// encoder (50 GB), frees it, stages the transformer (44 GB), frees it, and
// stages the video VAE (7 GB), so the process holds the host-side pieces
// (tokenizer, audio decoder: 0.3 GB) at rest and ~50 GB at a request's peak
// (VIDEO.md decision 2, M-o5).
//
// **The device is shared, in two ways** (VIDEO.md M9). The stagings do not
// take it at all: they allocate, build pipelines and write mapped memory,
// and none of that touches the queue, so ~2 minutes of a request's disk
// reads run while speech and images have the device. What does take it --
// the encoder's forward, each transformer forward, the VAE decode -- yields
// it between submissions (each ≤ 800 ms by construction) to whoever is
// waiting, so a speech request queued behind a 35 s forward waits for one
// submission rather than the forward.
type Video struct {
	opt  VideoOptions
	id   string
	pipe *pipeline.Pipeline

	mu  sync.Mutex // rng
	rng *rand.Rand
}

// NewVideo loads the host-side pieces. Nothing is staged on the device.
func NewVideo(opt VideoOptions) (*Video, error) {
	if opt.ID == "" {
		opt.ID = defaultVideoModelID
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the video pipeline needs a device; there is no host path for it")
	}
	d := opt.Device
	p, err := pipeline.New(d.dev, opt.Model, pipeline.Options{
		MaxPrompt: opt.MaxPrompt,
		Hold: func(fn func() error) error {
			return d.Do(func(*vk.Device) error { return fn() })
		},
		Between: func() { d.yield() },
	})
	if err != nil {
		return nil, fmt.Errorf("backend: loading %s: %w", opt.Model, err)
	}
	return &Video{opt: opt, id: opt.ID, pipe: p, rng: rand.New(rand.NewSource(rand.Int63()))}, nil
}

// Models reports the one model this backend serves.
func (b *Video) Models() []api.Model {
	return []api.Model{{ID: b.id, Object: "model", OwnedBy: "local"}}
}

// VideoGeometry is what a request may ask for.
func (b *Video) VideoGeometry() api.VideoGeometry {
	h, w, _ := plan.Canvas(16, 9, pipeline.DefaultShortEdge, plan.MaxPixels)
	return api.VideoGeometry{
		DefaultSize:    fmt.Sprintf("%dx%d", w, h),
		DefaultSeconds: pipeline.DefaultSeconds,
		MinSeconds:     plan.MinDuration, MaxSeconds: plan.MaxDuration,
		MaxShortEdge: plan.ShortEdge,
		DefaultSteps: pipeline.DefaultSteps,
		FPS:          plan.FPS,
		SampleRate:   b.pipe.SampleRate(),
	}
}

// Close frees whatever is staged. The job queue must be closed first.
func (b *Video) Close() {
	b.pipe.Close()
}

// request is the pipeline's request for an API one; seed is the resolved
// seed.
//
// **A duration under the model's 5 s floor is raised to it** rather than
// refused. OpenAI's default is 4 s and its clients send "4"; what they get
// is 5.17 s, and the job's `seconds` says so. Past 15 s is refused: that is
// a clip of a different length, not a rounding.
func request(req *api.VideoRequest, seed int64) *pipeline.Request {
	secs := req.Seconds
	if secs != 0 && secs < plan.MinDuration {
		secs = plan.MinDuration
	}
	return &pipeline.Request{
		Prompt: req.Prompt, AspectW: req.AspectW, AspectH: req.AspectH, ShortEdge: req.ShortEdge,
		Seconds: secs, Steps: req.Steps, Seed: uint64(seed),
	}
}

// PlanVideo resolves a request on the host: tokenises the prompt, fixes the
// canvas and frame count, and checks the transformer's arenas can span it.
func (b *Video) PlanVideo(req *api.VideoRequest) (*api.VideoPlan, error) {
	seed := int64(0)
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		b.mu.Lock()
		seed = b.rng.Int63()
		b.mu.Unlock()
	}
	if seed < 0 {
		return nil, fmt.Errorf("seed %d is negative: %w", seed, api.ErrUnsupported)
	}
	r, err := b.pipe.Resolve(request(req, seed))
	if err != nil {
		// Every Resolve failure is the request's: steps, canvas, duration,
		// a prompt past the cap, rows past the arenas.
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	return &api.VideoPlan{
		Width: r.Width, Height: r.Height, Frames: r.Frames,
		Seconds: float64(r.Frames) / plan.FPS,
		Steps:   r.Steps, Seed: seed, PromptTokens: len(r.Tokens),
		Estimate: estimate(r).total(),
	}, nil
}

// cost is a request's expected wall time by stage.
type cost struct {
	encode, stage, forward time.Duration // forward is one of Steps − 1
	forwards               int
	decode                 time.Duration
}

func (c cost) total() time.Duration {
	return c.encode + c.stage + time.Duration(c.forwards)*c.forward + c.decode
}

// estimate prices a request from M7/M8's measurements. A forward is
// a·L + b·L² in its L rows, fitted through 35.6 s at 15,936 rows and 143 s at
// 38,247 (7.7 s predicted at 5,095, where 7.4 was measured). The fixed costs
// are M8's: the encoder staged and run beside the tables (82 s), the
// transformer's staging (53 s), and the VAE's (7 s) plus 2.5 ms a video row
// of decode (38 s at 480p).
func estimate(r *pipeline.Resolved) cost {
	L := float64(len(r.Layout.Pos))
	fwd := 1.159e-3*L + 6.745e-8*L*L
	return cost{
		encode:   82 * time.Second,
		stage:    53 * time.Second,
		forward:  time.Duration(fwd * float64(time.Second)),
		forwards: r.Steps - 1,
		decode:   7*time.Second + time.Duration(2.5e-3*float64(len(r.Layout.Video))*float64(time.Second)),
	}
}

// GenerateVideo runs a planned request and muxes it to dst.
func (b *Video) GenerateVideo(ctx context.Context, req *api.VideoRequest, vp *api.VideoPlan, dst string, progress func(api.VideoProgress)) error {
	preq := request(req, vp.Seed)
	r, err := b.pipe.Resolve(preq)
	if err != nil {
		return fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	c := estimate(r)
	total := c.total().Seconds()
	at := func(d time.Duration) float64 { return d.Seconds() / total }
	res, err := b.pipe.Generate(ctx, preq, func(s pipeline.Stage, done, n int) {
		var f float64
		switch s {
		case pipeline.StageEncode:
			f = 0
		case pipeline.StageTransform:
			f = at(c.encode)
		case pipeline.StageStep:
			f = at(c.encode+c.stage) + at(time.Duration(done)*c.forward)
		case pipeline.StageDecode:
			f = at(c.total() - c.decode)
		}
		progress(api.VideoProgress{Stage: string(s), Fraction: f})
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return fmt.Errorf("h3: %w", err)
	}
	progress(api.VideoProgress{Stage: "mux", Fraction: 0.99})
	return pipeline.WriteMP4(dst, res)
}
