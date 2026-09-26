// Package pipeline runs a MiniMax-H3 request end to end (VIDEO.md M8, M10):
// a prompt, and optionally a first and/or last keyframe (fl2va), go in, and
// 24 fps frames with a 32 kHz stereo soundtrack come out, muxed to mp4 by
// ffmpeg.
//
// It composes what M1–M7 built, and its own job is the order in which they
// hold the machine. Everything resident at once is ~106 GB, so a request
// runs as three stagings, each freed before the next is allocated
// (decision 2):
//
//	keyframes    0.4 GB  (fl2va) the video VAE's encoder, freed
//	text encoder  50 GB  staged with the vision tower for fl2va, one
//	                     forward, freed
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
	"errors"
	"fmt"
	"image"
	"math/rand/v2"
	"path/filepath"
	"sync"
	"time"

	"strix-halo-vulkan/h3/audiovae"
	"strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/h3/textenc"
	"strix-halo-vulkan/h3/vae"
	qtextenc "strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// Request is one t2va or fl2va generation. Zero values take the served
// defaults (decision 6): 16:9 at a 480 short edge (864×480), 5 s, 20 steps.
type Request struct {
	Prompt string
	// First and Last are fl2va's keyframes: the picture the video starts
	// from and the one it ends on, either, both or neither (t2va). The
	// first of them present sets the canvas's aspect ratio unless the
	// request gives one, and is stretched onto the canvas; a second is
	// cover-cropped onto it.
	First, Last image.Image
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
	// generated video rows [rows, 96] and audio rows [2·n, 32] — so a gate
	// can start from an oracle's noise (decision 8). CondNoise does the same
	// for the keyframe rows' noise augmentation, patchified like them.
	VideoNoise, AudioNoise, CondNoise []float32
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
	// Keyframes are fl2va's pictures on the canvas, H·W·3 RGB in packed
	// order, and Presentation is the conditioner's input with their vision
	// blocks; both are nil for t2va.
	Keyframes    [][]byte
	Presentation *textenc.Presentation
}

// fl2va's vision blocks are laid out for the processor's 16-pixel patches
// merged 2×2 before anything is staged; New checks the checkpoint agrees.
const (
	visionPatch = 16
	visionMerge = 2
)

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
	keys, anchors := req.keyframeList()
	if r.Height == 0 || r.Width == 0 {
		aw, ah, se := req.AspectW, req.AspectH, req.ShortEdge
		if (aw == 0 || ah == 0) && len(keys) > 0 {
			// The canvas follows the geometry anchor's own aspect.
			b := keys[0].Bounds()
			aw, ah = float64(b.Dx()), float64(b.Dy())
		}
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
	var tags []int32
	if len(keys) == 0 {
		if r.Tokens, err = textenc.EncodePrompt(tok, req.Prompt); err != nil {
			return nil, err
		}
		tags = make([]int32, len(r.Tokens))
		for i := range tags {
			tags[i] = plan.TextTag
		}
	} else {
		for i, k := range keys {
			rgb, err := PlaceKeyframe(k, i, r.Height, r.Width)
			if err != nil {
				return nil, err
			}
			r.Keyframes = append(r.Keyframes, rgb)
		}
		grids := make([]qtextenc.Grid, len(keys))
		for i := range grids {
			grids[i] = textenc.ImageGrid(r.Height, r.Width, visionPatch)
		}
		if r.Presentation, err = textenc.NewPresentation(tok, req.Prompt, grids, visionMerge); err != nil {
			return nil, err
		}
		r.Tokens, tags = r.Presentation.IDs, r.Presentation.Tags
	}
	lh, lw := r.Height/plan.SpatialCompression, r.Width/plan.SpatialCompression
	if r.Layout, err = plan.NewLayout(tags, r.LatentFrames, lh, lw, r.AudioLatents, anchors); err != nil {
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
	// MaxPrompt caps a prompt's tokens (0: DefaultMaxPrompt).
	MaxPrompt int

	// Hold runs fn with the device's queue held, and nil runs it directly,
	// for a caller that owns the device outright (cmd/h3, the tests).
	//
	// **Only submissions are held.** Staging a model here is allocation,
	// pipeline creation and writes into mapped memory, none of which touch
	// the queue, so the three stagings (~100 s of a request, most of it
	// reading bf16 from disk) run while other work has the device. What is
	// held is the encoder's forward, the transformer's Begin and each
	// forward, and the video decode.
	Hold func(fn func() error) error
	// Between is called inside Hold between two of a forward's or the
	// decode's submissions (dit.GPU.Between, vae.GPU.Between); a server
	// yields the device there. A 480p forward is 35 s. The request's
	// context is checked at the same points, so a cancelled request stops
	// within one submission (≤ 800 ms) and not one forward.
	Between func()
}

// DefaultMaxPrompt is the longest prompt a request may carry, in tokens.
// MiniMax's Context-IR prompts run to ~500–1,500; the cost of a long one is
// its text rows in every forward's attention, not the encoder.
const DefaultMaxPrompt = 4096

// Pipeline holds the checkpoint's host-side pieces and, when resident, its
// device stagings.
type Pipeline struct {
	dir   string
	dev   *vk.Device
	opt   Options
	tok   *tokenizer.Tokenizer
	dcfg  *dit.Config
	vcfg  *vision.Config
	mrope qtextenc.MRopeSection
	pqc   *vae.PostQuantConv
	avae  *audiovae.Decoder

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
	if opt.MaxPrompt == 0 {
		opt.MaxPrompt = DefaultMaxPrompt
	}
	if opt.Hold == nil {
		opt.Hold = func(fn func() error) error { return fn() }
	}
	p := &Pipeline{dir: dir, dev: dev, opt: opt}
	var err error
	if p.tok, err = tokenizer.Load(filepath.Join(dir, "tokenizer")); err != nil {
		return nil, err
	}
	if p.dcfg, err = dit.LoadConfig(filepath.Join(dir, "transformer")); err != nil {
		return nil, err
	}
	// fl2va's conditioner: the vision tower's geometry, which Resolve
	// assumes, and the interleaved mrope an image block needs.
	if p.vcfg, err = vision.LoadConfig(filepath.Join(dir, "text_encoder")); err != nil {
		return nil, err
	}
	if p.vcfg.PatchSize != visionPatch || p.vcfg.SpatialMergeSize != visionMerge {
		return nil, fmt.Errorf("h3: the vision tower patches %d merged %d; fl2va is built for %d merged %d",
			p.vcfg.PatchSize, p.vcfg.SpatialMergeSize, visionPatch, visionMerge)
	}
	if p.mrope, err = qtextenc.LoadMRope(filepath.Join(dir, "text_encoder")); err != nil {
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

// SampleRate is the soundtrack's rate.
func (p *Pipeline) SampleRate() int { return p.avae.Cfg.SamplingRate }

// Tokenizer is the prompt tokenizer, for Resolve.
func (p *Pipeline) Tokenizer() *tokenizer.Tokenizer { return p.tok }

// ErrTooLarge is a request this pipeline cannot run as asked: a prompt past
// MaxPrompt, or more rows than the transformer's arenas can span. Resolve
// wraps it, so a server can refuse the request before anything is staged.
var ErrTooLarge = errors.New("h3: request too large")

// Resolve is the package's Resolve plus this pipeline's limits.
func (p *Pipeline) Resolve(req *Request) (*Resolved, error) {
	r, err := Resolve(p.tok, req)
	if err != nil {
		return nil, err
	}
	if len(r.Tokens) > p.opt.MaxPrompt {
		return nil, fmt.Errorf("%w: the prompt is %d tokens, past this server's %d", ErrTooLarge, len(r.Tokens), p.opt.MaxPrompt)
	}
	if err := dit.CheckArenas(p.dcfg, len(r.Layout.Pos), p.textBudget(len(r.Tokens)), p.opt.Chunk); err != nil {
		return nil, fmt.Errorf("%w: %dx%d for %d frames is %d rows: %v", ErrTooLarge, r.Width, r.Height, r.Frames, len(r.Layout.Pos), err)
	}
	return r, nil
}

// textBudget is the text rows a transformer is staged for: the request's
// own, or for a resident one a budget the next prompt will likely fit.
func (p *Pipeline) textBudget(text int) int {
	if p.opt.Resident {
		return max(text, 2048)
	}
	return text
}

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
	r, err := p.Resolve(req)
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
	between := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if p.opt.Between != nil {
			p.opt.Between()
		}
		return nil
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
	var anchors *qwen.Mat
	cond, err := func() (*qwen.Mat, error) {
		if len(r.Keyframes) == 0 {
			return p.encode(ctx, r)
		}
		var err error
		if anchors, err = p.encodeKeyframes(ctx, between, r); err != nil {
			return nil, err
		}
		return p.encode(ctx, r)
	}()
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	g.Between = between
	if err := p.opt.Hold(func() error { return g.Begin(lay, cond, tvals, tabs) }); err != nil {
		return nil, err
	}
	res.Timings[StageTransform] = time.Since(start)
	video, audio, err := p.noise(req, lay, anchors)
	if err != nil {
		return nil, err
	}
	// The keyframe rows lead the video rows and are never stepped: the
	// loop only writes the generated ones.
	gen := lay.CondVideoRows * video.Cols
	start = time.Now()
	for f := range vs.Timesteps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var v, a *qwen.Mat
		var took time.Duration
		err := p.opt.Hold(func() error {
			var err error
			v, a, took, err = g.Step(video, audio, rowTs[f])
			return err
		})
		if err != nil {
			return nil, err
		}
		res.StepDevice += took
		vs.Step(f, v.Data[gen:], video.Data[gen:])
		as.Step(f, a.Data, audio.Data)
		progress(StageStep, f+1, len(vs.Timesteps))
	}
	res.Timings[StageStep] = time.Since(start)
	res.VideoLatents, res.AudioLatents = video.Data[gen:], audio.Data
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
	frames, err := p.decodeVideo(ctx, between, r, video.Data[gen:])
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
// An fl2va presentation first runs its keyframes through the vision tower,
// staged beside it, and the encoder then reads them as a Qwen-Image edit
// reads its references: merged rows in the pads, deepstack after layers
// 0–2, 3-D positions.
func (p *Pipeline) encode(ctx context.Context, r *Resolved) (*qwen.Mat, error) {
	ids := r.Tokens
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
	var conds []qtextenc.Condition
	if r.Presentation != nil {
		if conds, err = p.towerKeyframes(ctx, dir, r); err != nil {
			return nil, err
		}
	}
	enc, err := qwen.NewGPUEncoder(p.dev, set, cfg, textenc.Layers, len(ids), nil)
	if err != nil {
		return nil, err
	}
	defer enc.Destroy()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var cond *qwen.Mat
	err = p.opt.Hold(func() error {
		var err error
		if r.Presentation == nil {
			cond, err = enc.Forward(ids)
			return err
		}
		rope, err := r.Presentation.MRope(cfg.HeadDim, cfg.RopeTheta, p.mrope)
		if err != nil {
			return err
		}
		cond, _, err = r.Presentation.EncodeGPU(enc, rope, conds, 0, nil)
		return err
	})
	return cond, err
}

// towerKeyframes stages the vision tower, runs every keyframe through it
// and frees it.
func (p *Pipeline) towerKeyframes(ctx context.Context, dir string, r *Resolved) ([]qtextenc.Condition, error) {
	cpu, err := vision.Load(dir, p.vcfg, 0)
	if err != nil {
		return nil, err
	}
	g := textenc.ImageGrid(r.Height, r.Width, visionPatch)
	tower, err := vision.NewGPU(p.dev, cpu, g.H*g.W)
	if err != nil {
		return nil, err
	}
	defer tower.Destroy()
	var conds []qtextenc.Condition
	for _, rgb := range r.Keyframes {
		planes, err := textenc.VisionPixels(rgb, r.Height, r.Width)
		if err != nil {
			return nil, err
		}
		pix, gh, gw, err := p.vcfg.Patchify(planes, r.Height, r.Width)
		if err != nil {
			return nil, err
		}
		var out *vision.Output
		err = p.opt.Hold(func() error {
			var err error
			out, err = tower.Forward(ctx, pix, gh, gw)
			return err
		})
		if err != nil {
			return nil, err
		}
		conds = append(conds, qtextenc.Condition{Merged: out.Merged, Deepstack: out.Deepstack})
	}
	return conds, nil
}

// encodeKeyframes stages the video VAE's encoder, encodes every keyframe to
// its anchor latent (the posterior drawn under vae.KeyframeSeed, fp16,
// normalised) and frees it. The result is the keyframe rows, patchified and
// clean, [condRows, 96] in packed order.
func (p *Pipeline) encodeKeyframes(ctx context.Context, between func() error, r *Resolved) (*qwen.Mat, error) {
	dir := filepath.Join(p.dir, "vae")
	cfg, err := vae.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	th, tw := vae.TileExtent(r.Height, r.Width)
	enc, err := vae.NewEncoder(p.dev, dir, th, tw, "")
	if err != nil {
		return nil, err
	}
	defer enc.Destroy()
	enc.Between = between
	var rows []float32
	for _, rgb := range r.Keyframes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		x, err := vae.NormalizePixels(rgb, r.Height, r.Width)
		if err != nil {
			return nil, err
		}
		var moments *vae.Tensor
		err = p.opt.Hold(func() error {
			var err error
			moments, _, err = enc.Encode(x)
			return err
		})
		if err != nil {
			return nil, err
		}
		z, err := cfg.SampleLatent(moments, vae.KeyframeSeed, nil)
		if err != nil {
			return nil, err
		}
		patched, err := vae.Patchify(z)
		if err != nil {
			return nil, err
		}
		rows = append(rows, patched...)
	}
	cols := 4 * cfg.LatentChannels
	if len(rows) != r.Layout.CondVideoRows*cols {
		return nil, fmt.Errorf("h3: %d keyframe values for %d keyframe rows", len(rows), r.Layout.CondVideoRows)
	}
	return &qwen.Mat{Rows: len(rows) / cols, Cols: cols, Data: rows}, nil
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
		text = p.textBudget(text)
		g, err := dit.NewGPU(p.dev, filepath.Join(p.dir, "transformer"), rows, text, p.opt.Chunk)
		if err != nil {
			return nil, err
		}
		p.dit, p.ditRows, p.ditText = g, rows, text
	}
	return p.dit, nil
}

// noise is the request's starting rows: the injected ones, or seeded
// Gaussians drawn in the pipeline's order — the keyframes' augmentation,
// then the video, then the audio. The keyframe rows are the anchors noised
// to t = 0.999 (`scale_noise`: t·x + (1 − t)·n, in float32), and they stay
// that way for the whole run.
func (p *Pipeline) noise(req *Request, lay *plan.Layout, anchors *qwen.Mat) (video, audio *qwen.Mat, err error) {
	video = qwen.NewMat(len(lay.Video), 96)
	audio = qwen.NewMat(len(lay.Audio), 32)
	nc := lay.CondVideoRows * video.Cols
	if (anchors == nil) != (nc == 0) || (anchors != nil && len(anchors.Data) != nc) {
		return nil, nil, fmt.Errorf("h3: keyframe latents %v for %d keyframe rows", anchors, lay.CondVideoRows)
	}
	rng := rand.New(rand.NewPCG(req.Seed, 0x4833))
	for _, m := range []struct {
		dst, inj []float32
	}{{video.Data[:nc], req.CondNoise}, {video.Data[nc:], req.VideoNoise}, {audio.Data, req.AudioNoise}} {
		if m.inj != nil {
			if len(m.inj) != len(m.dst) {
				return nil, nil, fmt.Errorf("h3: %d injected noise values for %d", len(m.inj), len(m.dst))
			}
			copy(m.dst, m.inj)
			continue
		}
		for i := range m.dst {
			m.dst[i] = float32(rng.NormFloat64())
		}
	}
	t := float32(plan.KeyframeNoiseAug)
	for i := 0; i < nc; i++ {
		video.Data[i] = float32(t*anchors.Data[i]) + float32((1-t)*video.Data[i])
	}
	return video, audio, nil
}

// decodeVideo unpatchifies and denormalises the video rows and decodes
// them, staging the VAE for the plan's tile size.
func (p *Pipeline) decodeVideo(ctx context.Context, between func() error, r *Resolved, rows []float32) ([][]byte, error) {
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var dec *vae.Tensor
	p.vae.Between = between
	err = p.opt.Hold(func() error {
		var err error
		dec, _, err = p.vae.Decode(z, p.pqc)
		return err
	})
	if err != nil {
		return nil, err
	}
	return vae.ToRGB8(dec), nil
}
