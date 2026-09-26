// Package pipeline runs a MiniMax-H3 t2va request end to end (VIDEO.md M8):
// a prompt goes in, and 24 fps frames with a 32 kHz stereo soundtrack come
// out, muxed to mp4 by ffmpeg.
//
// It composes what M1–M7 built, and its own job is the order in which they
// hold the machine. Everything resident at once is ~106 GB, so a request
// runs as three stagings, each freed before the next is allocated
// (decision 2):
//
//	text encoder  50 GB  staged, one forward, freed
//	transformer   44 GB  staged (weights + arenas), N − 1 forwards, freed
//	video VAE      7 GB  staged, the decode — the audio decode runs beside
//	                     it on the CPU
//
// The AdaLN tables (29 s of host work over 26 GB of bf16) are computed while
// the text encoder stages, since neither needs the other. A caller that
// holds the machine for many requests can keep the transformer and the VAE
// staged between them (Options.Resident): 51 GB, which is then the floor.
package pipeline

import (
	"context"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"sync"
	"time"

	"strix-halo-vulkan/h3/audiovae"
	"strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/h3/textenc"
	"strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// Request is one t2va generation. Zero values take the served defaults
// (decision 6): 16:9 at a 480 short edge (864×480), 5 s, 20 steps.
type Request struct {
	Prompt string
	// AspectW:AspectH and ShortEdge resolve the canvas as the pipeline's
	// resolve_canvas_size does; Height and Width, when both are set, are
	// taken as they are (multiples of 32).
	AspectW, AspectH float64
	ShortEdge        int
	Height, Width    int
	// Frames is the requested frame count, snapped up to 17n + 5 and held
	// to 5–15 s; Seconds sets it when Frames is zero.
	Frames  int
	Seconds float64
	// Steps is N: N − 1 forwards.
	Steps int
	Seed  uint64
	// VideoNoise and AudioNoise replace the seeded draws — the transformer's
	// video rows [rows, 96] and audio rows [2·n, 32] — so a gate can start
	// from an oracle's noise (decision 8).
	VideoNoise, AudioNoise []float32
}

const (
	DefaultShortEdge = 480
	DefaultSeconds   = 5.0
	DefaultSteps     = 20
	videoShift       = 12
	audioShift       = 3
)

// Resolved is a request's shape, fixed before anything is staged.
type Resolved struct {
	Height, Width int
	Frames        int // aligned, 17n + 5
	LatentFrames  int // 5n + 2
	AudioLatents  int // per channel
	Steps         int
	Tokens        []int32
	Layout        *plan.Layout
}

// Resolve validates a request and works out its shape.
func Resolve(tok *tokenizer.Tokenizer, req *Request) (*Resolved, error) {
	if req.Prompt == "" {
		return nil, fmt.Errorf("h3: an empty prompt")
	}
	r := &Resolved{Height: req.Height, Width: req.Width, Steps: req.Steps}
	if r.Steps == 0 {
		r.Steps = DefaultSteps
	}
	if r.Steps < 2 || r.Steps > 100 {
		return nil, fmt.Errorf("h3: %d steps; 2–100", r.Steps)
	}
	if r.Height == 0 || r.Width == 0 {
		aw, ah, se := req.AspectW, req.AspectH, req.ShortEdge
		if aw == 0 || ah == 0 {
			aw, ah = 16, 9
		}
		if se == 0 {
			se = DefaultShortEdge
		}
		if se < 256 || se > plan.ShortEdge {
			return nil, fmt.Errorf("h3: short edge %d; 256–%d", se, plan.ShortEdge)
		}
		h, w, err := plan.Canvas(aw, ah, se, plan.MaxPixels)
		if err != nil {
			return nil, err
		}
		r.Height, r.Width = h, w
	}
	if r.Height%plan.CanvasMultiple != 0 || r.Width%plan.CanvasMultiple != 0 || r.Height*r.Width > plan.MaxPixels ||
		min(r.Height, r.Width) < 256 {
		return nil, fmt.Errorf("h3: canvas %dx%d must be multiples of %d, at least 256 a side and at most %d pixels",
			r.Width, r.Height, plan.CanvasMultiple, plan.MaxPixels)
	}
	frames := req.Frames
	if frames == 0 {
		s := req.Seconds
		if s == 0 {
			s = DefaultSeconds
		}
		frames = int(s * plan.FPS)
	}
	var err error
	if r.Frames, err = plan.Frames(frames); err != nil {
		return nil, err
	}
	r.LatentFrames = plan.LatentFrames(r.Frames)
	r.AudioLatents = plan.AudioLatents(r.Frames)
	if r.Tokens, err = textenc.EncodePrompt(tok, req.Prompt); err != nil {
		return nil, err
	}
	tags := make([]int32, len(r.Tokens))
	for i := range tags {
		tags[i] = plan.TextTag
	}
	lh, lw := r.Height/plan.SpatialCompression, r.Width/plan.SpatialCompression
	if r.Layout, err = plan.NewLayout(tags, r.LatentFrames, lh, lw, r.AudioLatents, nil); err != nil {
		return nil, err
	}
	return r, nil
}

// Options configures a Pipeline.
type Options struct {
	// Resident keeps the transformer and the video VAE staged between
	// requests (51 GB held); otherwise every request stages and frees them.
	Resident bool
	// Chunk is the transformer's row chunk (0: 8192).
	Chunk int
}

// Pipeline holds the checkpoint's host-side pieces and, when resident, its
// device stagings.
type Pipeline struct {
	dir  string
	dev  *vk.Device
	opt  Options
	tok  *tokenizer.Tokenizer
	pqc  *vae.PostQuantConv
	avae *audiovae.Decoder

	mu      sync.Mutex // one request holds the stagings at a time
	dit     *dit.GPU
	ditRows int
	ditText int
	vae     *vae.GPU
	vaeTile [2]int
}

// New loads the host-side pieces: the tokenizer, the post-quant conv and
// the audio decoder (0.3 GB, fp32).
func New(dev *vk.Device, dir string, opt Options) (*Pipeline, error) {
	if opt.Chunk == 0 {
		opt.Chunk = 8192
	}
	p := &Pipeline{dir: dir, dev: dev, opt: opt}
	var err error
	if p.tok, err = tokenizer.Load(filepath.Join(dir, "tokenizer")); err != nil {
		return nil, err
	}
	if p.pqc, err = vae.LoadPostQuantConv(filepath.Join(dir, "vae")); err != nil {
		return nil, err
	}
	if p.avae, err = audiovae.Load(filepath.Join(dir, "audio_vae")); err != nil {
		return nil, err
	}
	return p, nil
}

// Tokenizer is the prompt tokenizer, for Resolve.
func (p *Pipeline) Tokenizer() *tokenizer.Tokenizer { return p.tok }

// Close frees whatever is staged.
func (p *Pipeline) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.freeDiT()
	p.freeVAE()
}

func (p *Pipeline) freeDiT() {
	if p.dit != nil {
		p.dit.Destroy()
		p.dit = nil
	}
}

func (p *Pipeline) freeVAE() {
	if p.vae != nil {
		p.vae.Destroy()
		p.vae = nil
	}
}

// Stage names a request's phases, for progress and timings.
type Stage string

const (
	StageEncode    Stage = "encode"    // staging the text encoder and running it
	StageTables    Stage = "tables"    // the AdaLN tables, beside the encoder
	StageTransform Stage = "transform" // staging the transformer
	StageStep      Stage = "step"      // one forward
	StageDecode    Stage = "decode"    // both decoders
)

// Progress is called as a request moves: stage, and for StageStep the
// forward just finished of total.
type Progress func(stage Stage, done, total int)

// Result is a finished generation.
type Result struct {
	*Resolved
	// Video is the frames, rgb24, Height × Width each.
	Video [][]byte
	// Left and Right are the soundtrack at SampleRate.
	Left, Right []float32
	SampleRate  int
	// Latents are the transformer's final rows: video [rows, 96] and audio
	// [2·n, 32], normalised.
	VideoLatents, AudioLatents []float32
	// Timings has the wall time of each stage; StepDevice the forwards'
	// summed device time.
	Timings    map[Stage]time.Duration
	StepDevice time.Duration
}

// Generate runs one request. ctx is checked between forwards; a cancelled
// request frees what it staged (unless resident) and returns ctx.Err().
func (p *Pipeline) Generate(ctx context.Context, req *Request, progress Progress) (*Result, error) {
	if progress == nil {
		progress = func(Stage, int, int) {}
	}
	r, err := Resolve(p.tok, req)
	if err != nil {
		return nil, err
	}
	lay := r.Layout
	p.mu.Lock()
	defer p.mu.Unlock()
	res := &Result{Resolved: r, Timings: map[Stage]time.Duration{}}
	if !p.opt.Resident {
		defer p.freeDiT()
		defer p.freeVAE()
	}

	// The schedules, and every row's timestep index into the distinct
	// timesteps the whole request uses.
	vs, err := plan.NewSchedule(r.Steps, videoShift)
	if err != nil {
		return nil, err
	}
	as, err := plan.NewSchedule(r.Steps, audioShift)
	if err != nil {
		return nil, err
	}
	var tvals []float32
	index := map[float32]int32{}
	rowTs := make([][]int32, len(vs.Timesteps))
	for f := range vs.Timesteps {
		u, idx := lay.RowTimesteps(vs.Timesteps[f], as.Timesteps[f])
		rowTs[f] = make([]int32, len(idx))
		for i, j := range idx {
			k, ok := index[u[j]]
			if !ok {
				k = int32(len(tvals))
				index[u[j]] = k
				tvals = append(tvals, u[j])
			}
			rowTs[f][i] = k
		}
	}

	// The text encoder and, beside it, the AdaLN tables. The transformer is
	// not staged yet, whether or not it will stay: the encoder's 50 GB and
	// its 44 do not fit together beside anything else.
	p.freeDiT()
	var tabs []*dit.Table
	var tabErr error
	var tabTook time.Duration
	var tabWG sync.WaitGroup
	tabWG.Add(1)
	go func() {
		defer tabWG.Done()
		start := time.Now()
		tabs, tabErr = dit.Tables(filepath.Join(p.dir, "transformer"), tvals)
		tabTook = time.Since(start)
	}()
	start := time.Now()
	progress(StageEncode, 0, 1)
	cond, err := p.encode(r.Tokens)
	res.Timings[StageEncode] = time.Since(start)
	tabWG.Wait()
	res.Timings[StageTables] = tabTook
	if err != nil {
		return nil, err
	}
	if tabErr != nil {
		return nil, tabErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// The transformer.
	start = time.Now()
	progress(StageTransform, 0, 1)
	g, err := p.stageDiT(len(lay.Pos), len(r.Tokens))
	if err != nil {
		return nil, err
	}
	if err := g.Begin(lay, cond, tvals, tabs); err != nil {
		return nil, err
	}
	res.Timings[StageTransform] = time.Since(start)
	video, audio := p.noise(req, lay)
	start = time.Now()
	for f := range vs.Timesteps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, a, took, err := g.Step(video, audio, rowTs[f])
		if err != nil {
			return nil, err
		}
		res.StepDevice += took
		vs.Step(f, v.Data, video.Data)
		as.Step(f, a.Data, audio.Data)
		progress(StageStep, f+1, len(vs.Timesteps))
	}
	res.Timings[StageStep] = time.Since(start)
	res.VideoLatents, res.AudioLatents = video.Data, audio.Data
	if !p.opt.Resident {
		p.freeDiT()
	}

	// Both decoders: the video's on the device, the audio's on the CPU
	// beside it.
	start = time.Now()
	progress(StageDecode, 0, 1)
	var audioErr error
	var audioWG sync.WaitGroup
	audioWG.Add(1)
	go func() {
		defer audioWG.Done()
		res.Left, res.Right, audioErr = p.avae.Decode(audio.Data)
	}()
	frames, err := p.decodeVideo(r, video.Data)
	audioWG.Wait()
	if err != nil {
		return nil, err
	}
	if audioErr != nil {
		return nil, audioErr
	}
	res.Video = frames
	res.SampleRate = p.avae.Cfg.SamplingRate
	res.Timings[StageDecode] = time.Since(start)
	return res, nil
}

// encode stages the text encoder, runs the prompt through it and frees it.
func (p *Pipeline) encode(ids []int32) (*qwen.Mat, error) {
	dir := filepath.Join(p.dir, "text_encoder")
	cfg, err := textenc.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	enc, err := qwen.NewGPUEncoder(p.dev, set, cfg, textenc.Layers, len(ids), nil)
	if err != nil {
		return nil, err
	}
	defer enc.Destroy()
	return enc.Forward(ids)
}

// stageDiT returns a transformer staged for at least rows rows and text
// text rows, restaging a resident one that is too small.
func (p *Pipeline) stageDiT(rows, text int) (*dit.GPU, error) {
	if p.dit != nil && (p.ditRows < rows || p.ditText < text) {
		p.freeDiT()
	}
	if p.dit == nil {
		// Staged for the request's own sizes; a resident one is sized up to
		// a text budget so the next prompt fits.
		if p.opt.Resident {
			text = max(text, 2048)
		}
		g, err := dit.NewGPU(p.dev, filepath.Join(p.dir, "transformer"), rows, text, p.opt.Chunk)
		if err != nil {
			return nil, err
		}
		p.dit, p.ditRows, p.ditText = g, rows, text
	}
	return p.dit, nil
}

// noise is the request's starting rows: the injected ones, or seeded
// Gaussians, video first then audio, as the pipeline draws them.
func (p *Pipeline) noise(req *Request, lay *plan.Layout) (video, audio *qwen.Mat) {
	video = qwen.NewMat(len(lay.Video), 96)
	audio = qwen.NewMat(len(lay.Audio), 32)
	rng := rand.New(rand.NewPCG(req.Seed, 0x4833))
	for _, m := range []struct {
		dst, inj []float32
	}{{video.Data, req.VideoNoise}, {audio.Data, req.AudioNoise}} {
		if m.inj != nil {
			copy(m.dst, m.inj)
			continue
		}
		for i := range m.dst {
			m.dst[i] = float32(rng.NormFloat64())
		}
	}
	return video, audio
}

// decodeVideo unpatchifies and denormalises the video rows and decodes
// them, staging the VAE for the plan's tile size.
func (p *Pipeline) decodeVideo(r *Resolved, rows []float32) ([][]byte, error) {
	dir := filepath.Join(p.dir, "vae")
	cfg, err := vae.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	lh, lw := r.Height/plan.SpatialCompression, r.Width/plan.SpatialCompression
	z, err := vae.Unpatchify(rows, cfg.LatentChannels, r.LatentFrames, lh, lw)
	if err != nil {
		return nil, err
	}
	cfg.Denormalize(z)
	vp, err := cfg.NewPlan(r.LatentFrames, lh, lw)
	if err != nil {
		return nil, err
	}
	tile := [2]int{vp.TileH, vp.TileW}
	if p.vae != nil && p.vaeTile != tile {
		p.freeVAE()
	}
	if p.vae == nil {
		if p.vae, err = vae.NewGPU(p.dev, dir, tile[0], tile[1], 8); err != nil {
			return nil, err
		}
		p.vaeTile = tile
	}
	dec, _, err := p.vae.Decode(z, p.pqc)
	if err != nil {
		return nil, err
	}
	return vae.ToRGB8(dec), nil
}
