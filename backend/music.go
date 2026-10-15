package backend

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/api"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/tokenizer"
)

// MusicOptions is what cmd/serve's flags come to.
type MusicOptions struct {
	// Models is the checkpoint root: acestep-v15-xl-base/ (or -turbo/),
	// Ace-Step1.5/vae, Qwen3-Embedding-0.6B/ and acestep-5Hz-lm-4B/ under it.
	Models string
	// DiTs are the DiT checkpoints to stage, pipeline.DiTModels' names:
	// "base" (research/music-vertical.md A15: CFG over 32 steps) and
	// "turbo" (8 distilled steps). The first is the default; a request's
	// model picks another (resolveModel). Each is ~10 GB resident. Empty is
	// base alone.
	DiTs []string
	// Device is required.
	Device *Device
	// NoLM leaves the 5 Hz LM out (5 GB of weights and 1.8 GB of KV
	// cache): the DiT-only path, and a thinking request is a 400.
	NoLM bool
	// LMFP16 stages the LM's layers as fp16 instead of int8 (MUSIC.md A10):
	// the control, 2x slower a step and 3.4 GB more.
	LMFP16 bool
	// NoAesthetics leaves Audiobox Aesthetics out of the auto score
	// (MUSIC.md A14c). It is loaded when models/audiobox-aesthetics exists.
	NoAesthetics bool
	// NoSongEval leaves SongEval out of the auto score (MUSIC.md A14d). It
	// is loaded when models/MuQ-large-msd-iter and SongEval's head exist.
	NoSongEval bool
	// ID is the model id this backend answers to.
	ID string
	// Swap is the residency slot this pipeline takes turns in with image
	// and video. Required.
	Swap *Swap
}

// Music is the ACE-Step 1.5 adapter: an api.MusicBackend over ace/pipeline
// (MUSIC.md A9).
//
// **Everything is resident while it holds the Swap** (A-o4): the DiT, its
// encoders, the VAE, the text encoder and the LM with its cache, ~18 GB
// (~21 with the LM in fp16). Staging them is ~16 s and a 60 s song is
// 14 s, so they stay staged
// between songs, until image or video is asked for or the slot's idle
// timer frees them. Planning a request needs none of it: the LM's
// tokenizer and genres vocabulary are held on the host for that.
//
// **The device is shared a step at a time.** A request holds it only
// between yields: after every LM step (~25 ms) and every DiT forward
// (0.1-2.4 s by length), so speech waits for one of those, not the song.
type Music struct {
	opt  MusicOptions
	id   string
	pipe *pipeline.Pipeline // nil while another vertical holds the slot
	// planner is the LM's tokenizer and genres vocabulary without the LM,
	// for PlanMusic; nil under NoLM.
	planner   *lm.Planner
	genresErr error
	lmOpt     lm.Options

	// dits are the staged DiT checkpoints' directory names, the default
	// first (MusicOptions.DiTs).
	dits []string

	mu  sync.Mutex // rng
	rng *rand.Rand
}

// NewMusic stages the models on the device, through the slot: they are
// resident when this returns, until something else is asked for.
func NewMusic(opt MusicOptions) (*Music, error) {
	if len(opt.DiTs) == 0 {
		opt.DiTs = []string{"base"}
	}
	var dits []string
	for _, n := range opt.DiTs {
		dir, ok := pipeline.DiTModels[n]
		if !ok {
			return nil, fmt.Errorf("backend: no music DiT %q (base or turbo)", n)
		}
		if !slices.Contains(dits, dir) {
			dits = append(dits, dir)
		}
	}
	if opt.ID == "" {
		opt.ID = dits[0]
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the music pipeline needs a device; there is no host path for it")
	}
	if opt.Swap == nil {
		return nil, fmt.Errorf("backend: the music pipeline needs a swap slot")
	}
	b := &Music{opt: opt, id: opt.ID, dits: dits, lmOpt: lm.DefaultOptions(), rng: rand.New(rand.NewSource(rand.Int63()))}
	if opt.LMFP16 {
		b.lmOpt.Bank = lm.BankFP16
	}
	if !opt.NoLM {
		dir := pipeline.LMDir(opt.Models)
		tk, err := tokenizer.Load(dir)
		if err != nil {
			return nil, fmt.Errorf("backend: the 5 Hz LM's tokenizer: %w", err)
		}
		if b.planner, err = lm.NewPlanner(nil, tk); err != nil {
			return nil, err
		}
		// A missing vocabulary refuses sample mode, and PlanMusic says why.
		b.genresErr = b.planner.LoadGenres(filepath.Join(filepath.Dir(dir), lm.GenresFile))
	}
	if err := opt.Swap.Load(b); err != nil {
		return nil, err
	}
	return b, nil
}

// SwapName is the vertical's name in the swap log.
func (b *Music) SwapName() string { return "music" }

// Load stages the pipeline. It takes no device lock (TestStagingSkipsTheQueue).
func (b *Music) Load() error {
	d := b.opt.Device.dev
	dirs := pipeline.DefaultDirs(b.opt.Models)
	dirs.DiT = filepath.Join(b.opt.Models, b.dits[0])
	p, err := pipeline.New(d, dirs, plan.MaxSeconds)
	for _, name := range b.dits[1:] {
		if err == nil {
			if err = p.AddDiT(d, filepath.Join(b.opt.Models, name)); err != nil {
				p.Destroy()
			}
		}
	}
	if err == nil && !b.opt.NoLM {
		if err = p.LoadLM(d, pipeline.LMDir(b.opt.Models), b.lmOpt); err != nil {
			p.Destroy()
		}
	}
	if aes := pipeline.AesDir(b.opt.Models); err == nil && !b.opt.NoAesthetics {
		if _, serr := os.Stat(aes); serr == nil {
			if err = p.LoadAesthetics(d, aes); err != nil {
				p.Destroy()
			}
		}
	}
	if muq, head := pipeline.SongEvalDirs(b.opt.Models); err == nil && !b.opt.NoSongEval {
		_, e1 := os.Stat(muq)
		_, e2 := os.Stat(head)
		if e1 == nil && e2 == nil {
			if err = p.LoadSongEval(d, b.opt.Models, plan.MaxSeconds); err != nil {
				p.Destroy()
			}
		}
	}
	if err != nil {
		return fmt.Errorf("backend: loading ACE-Step from %s: %w", b.opt.Models, err)
	}
	b.pipe = p
	return nil
}

// Unload frees the pipeline. The slot guarantees nothing is running on it,
// and destroying touches no queue.
func (b *Music) Unload() {
	if b.pipe == nil {
		return
	}
	b.pipe.Destroy()
	b.pipe = nil
}

// Models reports the DiT checkpoints this backend serves, the default
// first (its id is MusicOptions.ID when one was given).
func (b *Music) Models() []api.Model {
	out := []api.Model{{ID: b.id, Object: "model", OwnedBy: "local"}}
	for _, d := range b.dits[1:] {
		out = append(out, api.Model{ID: d, Object: "model", OwnedBy: "local"})
	}
	return out
}

// resolveModel is the staged DiT a request's model names, as upstream's
// select_generation_handler matches a loaded model's name: the id, or
// "base"/"turbo", or a name with either as a word ("acestep-v15-turbo",
// upstream's 2B default, runs XL turbo). Anything else -- an empty model,
// a client's own default, a checkpoint not staged -- runs the default, as
// upstream falls back to its primary; the job's model says which ran.
func (b *Music) resolveModel(model string) string {
	if model == "" || model == b.id {
		return b.dits[0]
	}
	if slices.Contains(b.dits, model) {
		return model
	}
	words := strings.FieldsFunc(strings.ToLower(model), func(r rune) bool { return r == '-' || r == '_' || r == '/' || r == ' ' })
	for _, short := range []string{"turbo", "base"} {
		if slices.Contains(words, short) && slices.Contains(b.dits, pipeline.DiTModels[short]) {
			return pipeline.DiTModels[short]
		}
	}
	return b.dits[0]
}

// modelID is the id a resolved model is reported as.
func (b *Music) modelID(model string) string {
	if model == b.dits[0] {
		return b.id
	}
	return model
}

// formats is pipeline.Formats' keys, sorted.
func formats() []string {
	var out []string
	for f := range pipeline.Formats {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// MusicInfo is what a request may ask for.
func (b *Music) MusicInfo() api.MusicInfo {
	return api.MusicInfo{MinSeconds: plan.MinSeconds, MaxSeconds: plan.MaxSeconds,
		FallbackSeconds: plan.FallbackSeconds, SampleRate: plan.SampleRate, Formats: formats(),
		Thinking: b.planner != nil, Analysis: b.planner != nil && b.planner.HasGenres(),
		Steps: b.steps(b.dits[0], &api.MusicRequest{})}
}

// guided reports whether the model (a staged DiT's name) is the base
// model, which runs CFG.
func guided(model string) bool { return model != pipeline.DiTModels["turbo"] }

// steps is how many DiT steps a request runs on the model: turbo's table
// (snapped timesteps or the shift's 8), or on base its timesteps' pairs,
// its inference_steps, or Gradio's 32.
func (b *Music) steps(model string, req *api.MusicRequest) int {
	switch {
	case !guided(model):
		return len(plan.Schedule(cmpOr(req.Shift, plan.DefaultShift), req.Timesteps))
	case req.Timesteps != nil:
		return len(req.Timesteps) - 1
	}
	return cmp.Or(req.InferenceSteps, plan.BaseSteps)
}

// guidance is a base request's CFG.
func guidance(req *api.MusicRequest) *plan.Guidance {
	g := plan.DefaultGuidance
	if v := req.GuidanceScale; v != nil {
		g.Scale = *v
	}
	if v := req.CFGStart; v != nil {
		g.IntervalStart = *v
	}
	if v := req.CFGEnd; v != nil {
		g.End = *v
	}
	return &g
}

// forwards is a step's DiT forwards on the model, in conditional-forward
// units: the null row costs ~0.81 of one under S1 (A15).
func forwards(model string, req *api.MusicRequest) float64 {
	if guided(model) && guidance(req).Scale > 1 {
		return 1.81
	}
	return 1
}

// Close frees the device objects. The job queue must be closed first, and
// the slot is closed after it.
func (b *Music) Close() { b.Unload() }

// request is the pipeline's request for an API one.
func musicRequest(req *api.MusicRequest) *plan.Request {
	return &plan.Request{Caption: req.Caption, Lyrics: req.Lyrics, BPM: req.BPM, KeyScale: req.KeyScale,
		TimeSignature: req.TimeSignature, Duration: req.Duration, Language: req.Language}
}

var validTimesigs = map[string]bool{"2": true, "3": true, "4": true, "6": true,
	"2/4": true, "3/4": true, "4/4": true, "6/8": true}

func unsupported(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, api.ErrUnsupported)...)
}

// PlanMusic validates a request and resolves its seeds, format and, when
// given, its length. Nothing touches the device; the LM's prompt is
// tokenised to check it fits the cache with the codes behind it.
func (b *Music) PlanMusic(req *api.MusicRequest) (*api.MusicPlan, error) {
	if req.Analysis {
		return b.planAnalysis(req)
	}
	if req.Thinking && b.planner == nil {
		return nil, unsupported("thinking needs the 5 Hz LM, which this server did not load (-music-lm)")
	}
	if req.Sample {
		if b.planner == nil {
			return nil, unsupported("sample mode needs the 5 Hz LM, which this server did not load (-music-lm)")
		}
		if !b.planner.HasGenres() {
			return nil, unsupported("sample mode needs upstream's genres vocabulary (models/%s): %v", lm.GenresFile, b.genresErr)
		}
	}
	if req.Duration != 0 && (req.Duration < plan.MinSeconds || req.Duration > plan.MaxSeconds) {
		return nil, unsupported("duration %g s is outside %d-%d s", req.Duration, plan.MinSeconds, plan.MaxSeconds)
	}
	if req.BPM != 0 && (req.BPM < 30 || req.BPM > 300) {
		return nil, unsupported("bpm %d is outside 30-300", req.BPM)
	}
	if ts := strings.TrimSpace(req.TimeSignature); ts != "" && !validTimesigs[ts] {
		return nil, unsupported("time signature %q: 2, 3, 4 or 6 (or 2/4, 3/4, 4/4, 6/8)", req.TimeSignature)
	}
	if req.Language != "" {
		ok := false
		for _, l := range lm.ValidLanguages {
			ok = ok || l == req.Language
		}
		if !ok {
			return nil, unsupported("vocal language %q is not one the model knows (%s)", req.Language,
				strings.Join(lm.ValidLanguages, ", "))
		}
	}
	format := req.Format
	if format == "" {
		format = "mp3"
	}
	if _, ok := pipeline.Formats[format]; !ok {
		return nil, unsupported("audio format %q: %s", req.Format, strings.Join(formats(), ", "))
	}
	if req.Shift != 0 && (req.Shift < 1 || req.Shift > 5) {
		return nil, unsupported("shift %g is outside 1-5", req.Shift)
	}
	if v := req.LMTemperature; v != nil && *v < 0 {
		return nil, unsupported("lm_temperature %g is negative", *v)
	}
	if v := req.LMCFGScale; v != nil && *v < 1 {
		return nil, unsupported("lm_cfg_scale %g is below 1", *v)
	}
	if v := req.LMTopP; v != nil && (*v <= 0 || *v > 1) {
		return nil, unsupported("lm_top_p %g is outside (0, 1]", *v)
	}
	// Audio in (MUSIC.md A11): decoded here to be checked and measured,
	// and again when the job runs.
	audio, err := musicAudio(req)
	if err != nil {
		return nil, err
	}
	model := b.resolveModel(req.Model)
	if guided(model) && req.Timesteps != nil && len(req.Timesteps) < 2 {
		return nil, unsupported("timesteps: the base model steps between entries, so it needs at least 2 (end with 0 to land on the song)")
	}
	p := &api.MusicPlan{Format: format, Model: b.modelID(model), Steps: b.steps(model, req)}
	if audio != nil && audio.Task != pipeline.Text2Music {
		p.Seconds = audio.Seconds()
		if p.Seconds > plan.MaxSeconds {
			return nil, unsupported("the %s would be %.0f s, past the %d s ceiling", audio.Task, p.Seconds, plan.MaxSeconds)
		}
		if cs := p.Seconds; cs < 1 {
			return nil, unsupported("the source audio is %.2f s; a latent is 0.04 s and upstream pads to 5.12 s, but this is too short to hear", cs)
		}
	}
	// Drawn seeds are 32-bit, as upstream draws them: a client in
	// JavaScript reads a JSON number past 2^53 wrong, and could not replay
	// the song it was given the seed of.
	b.mu.Lock()
	p.Seed, p.LMSeed = int64(b.rng.Uint32()), int64(b.rng.Uint32())
	b.mu.Unlock()
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	if req.LMSeed != nil {
		p.LMSeed = *req.LMSeed
	}
	r := musicRequest(req)
	switch {
	case p.Seconds > 0:
	case req.Duration > 0:
		p.Seconds = float64(musicLatents(r, req.Thinking)*plan.Hop) / plan.SampleRate
	case !req.Thinking:
		p.Seconds = plan.FallbackSeconds
	}
	if req.Thinking && !req.Sample {
		// The codes phase's prompt is the CoT's plus the CoT, and the codes
		// follow it in the same cache. (Sample mode's lyrics are not known
		// yet; the codes phase checks its own fit when it starts.)
		secs := cmpOr(p.Seconds, plan.MaxSeconds)
		need, err := b.planner.Positions(r.Caption, r.Lyrics, secs)
		if err != nil {
			return nil, err
		}
		if need > b.lmOpt.MaxLen {
			return nil, unsupported("the LM would need up to %d positions for %.0f s of codes after these lyrics, and holds %d; shorten the lyrics or the song",
				need, secs, b.lmOpt.MaxLen)
		}
	}
	c := musicEstimate(r, req.Thinking, req.Sample, cmpOr(p.Seconds, 120), b.lmOpt.Bank, p.Steps, forwards(model, req))
	c.encode = encodeCost(audio)
	p.Estimate = c.total()
	return p, nil
}

// musicAudio decodes a request's uploads into the pipeline's audio-in
// options, or nil for plain text2music. An unreadable or silent file is
// the client's.
func musicAudio(req *api.MusicRequest) (*pipeline.Audio, error) {
	if req.Task == "" || (req.Task == pipeline.Text2Music && req.ReferenceAudio == nil) {
		return nil, nil
	}
	a := pipeline.NewAudio(req.Task)
	var err error
	if req.SourceAudio != nil {
		if a.Source, err = pipeline.ReadAudioBytes(req.SourceAudio); err != nil {
			return nil, unsupported("src_audio: %v", err)
		}
		if len(a.Source)/2 < plan.Hop {
			return nil, unsupported("src_audio is shorter than one latent (%d samples)", plan.Hop)
		}
	}
	if req.ReferenceAudio != nil {
		if a.Reference, err = pipeline.ReadAudioBytes(req.ReferenceAudio); err != nil {
			return nil, unsupported("reference_audio: %v", err)
		}
	}
	if req.CoverStrength != nil {
		a.CoverStrength = *req.CoverStrength
	}
	a.CoverNoise = req.CoverNoise
	a.RepaintStart, a.RepaintEnd, a.RepaintMode, a.ExplicitMask = req.RepaintStart, req.RepaintEnd, req.RepaintMode, req.ExplicitMask
	if req.RepaintStrength != nil {
		a.RepaintStrength = *req.RepaintStrength
	}
	if err := a.Check(); err != nil {
		return nil, unsupported("%v", err)
	}
	return a, nil
}

func cmpOr(a, b float64) float64 {
	if a != 0 {
		return a
	}
	return b
}

// musicLatents is the latent length a request with a duration runs at:
// the codes' when thinking (5 a second, int(duration·5) of them), the
// request's own otherwise.
func musicLatents(r *plan.Request, thinking bool) int {
	if thinking {
		return max(plan.MinLatents, lm.CodesPerSecond*int(r.Duration*lm.CodesPerSecond))
	}
	return r.LatentLength()
}

// musicCost is a request's expected wall time by stage, from MUSIC.md's
// measurements with the device to itself: an LM step is 28 ms writing text
// and 25 ms writing codes on the int8 bank, 47 and 44 on fp16 (A10), a CoT
// ~150 of them; a DiT forward is 2.65e-4·L + 7.16e-9·L² s over L = 12.5 a
// second tokens (A3's 0.86 s at 3,000 and 2.39 s at 7,500) plus 35 ms of
// host; the VAE is 10.4 ms a second of audio (A6); encoding the file is
// ffmpeg's. Sample mode's pass is ~sampleSteps (a caption, the metas and
// a song's lyrics), and the song it writes usually has every meta, so
// phase 1 is then skipped. The auto score (A14b) is two parts: the LM's,
// a prefill of the codes at ~0.5 ms a token and a teacher-forced pass of
// ~45 ms for each 64 tokens of a target (the lyrics twice, ~200 characters
// a pass); the DiT's alignment, two forwards to layer 7 and the host's
// softmax, ~3 ms a second of audio (TestScoreGenerate: 0.67 s for a 30 s
// song, 3.3 s for a 200 s one with 12 verses).
type musicCost struct {
	sample, think, codes, encode, dit, vae, score, write time.Duration
}

const sampleSteps = 450

func (c musicCost) total() time.Duration {
	return c.sample + c.think + c.codes + c.encode + c.dit + c.vae + c.score + c.write
}

// scoreCost is the auto score's share of a request.
func scoreCost(r *plan.Request, thinking, withLM bool, seconds float64) time.Duration {
	s := 0.0
	lmScore := 0.35 + 0.0027*seconds + 0.13*math.Ceil(float64(len(r.Lyrics))/200)
	if thinking {
		s += lmScore
	}
	if withLM {
		// The render's own codes: the tokenizer (~0.4 ms a second) and
		// the same LM pass again.
		s += lmScore + 0.0004*seconds
	}
	// The loop measure, on the host.
	s += 0.002 * seconds
	if strings.TrimSpace(r.Lyrics) != "" {
		s += 0.05 + 0.003*seconds
	}
	// Audiobox Aesthetics: ~17 ms of device a 10 s window (A14c), and the
	// host's resample.
	s += 0.03 + 0.002*seconds
	// SongEval: the host's resample and mel, and ~3 ms of device a second.
	s += 0.05 + 0.004*seconds
	return time.Duration(s * float64(time.Second))
}

func musicEstimate(r *plan.Request, thinking, sample bool, seconds float64, bank lm.Bank, steps int, forwards float64) musicCost {
	sec := func(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }
	text, code := 0.028, 0.025
	if bank == lm.BankFP16 {
		text, code = 0.047, 0.044
	}
	var c musicCost
	if sample {
		c.sample = sec(sampleSteps * text)
	}
	if thinking {
		if !sample && !lm.UserMeta(r).HasAll() {
			c.think = sec(150 * text)
		}
		c.codes = sec((seconds*lm.CodesPerSecond + 1) * code)
	}
	L := seconds * 12.5
	c.dit = sec(float64(steps) * forwards * (2.65e-4*L + 7.16e-9*L*L + 0.035))
	c.vae = sec(0.0104*seconds + 0.1)
	c.write = sec(0.004 * seconds)
	return c
}

// GenerateMusic runs a planned request and writes the audio to dst.
func (b *Music) GenerateMusic(ctx context.Context, req *api.MusicRequest, mp *api.MusicPlan, dst string, progress func(api.MusicProgress)) error {
	if req.Analysis {
		return b.analyse(ctx, req, mp, progress)
	}
	model := b.resolveModel(req.Model)
	r := musicRequest(req)
	secs := cmpOr(mp.Seconds, 120)
	audio, err := musicAudio(req)
	if err != nil {
		return err
	}
	c := musicEstimate(r, req.Thinking, req.Sample, secs, b.lmOpt.Bank, b.steps(model, req), forwards(model, req))
	c.encode = encodeCost(audio)
	if req.Score {
		c.score = scoreCost(r, req.Thinking, !b.opt.NoLM, secs)
	}
	var sampling *lm.Sampling
	if req.LMTemperature != nil || req.LMCFGScale != nil || req.LMTopP != nil {
		s := lm.DefaultSampling
		if v := req.LMTemperature; v != nil {
			s.Temperature = *v
		}
		if v := req.LMCFGScale; v != nil {
			s.CFG = float32(*v)
		}
		if v := req.LMTopP; v != nil {
			s.TopP = *v
		}
		sampling = &s
	}
	frac := func(stage string, done, total int) float64 {
		t := c.total().Seconds()
		part := func(d time.Duration, done, total int) float64 {
			if total <= 0 {
				return 0
			}
			return d.Seconds() * float64(min(done, total)) / float64(total)
		}
		switch stage {
		case "sample":
			return part(c.sample, done, sampleSteps) / t
		case "think":
			return (c.sample.Seconds() + part(c.think, done, 150)) / t
		case "codes":
			return (c.sample+c.think).Seconds()/t + part(c.codes, done, total)/t
		case "dit":
			return (c.sample+c.think+c.codes+c.encode).Seconds()/t + part(c.dit, done, total)/t
		case "vae":
			return (c.sample + c.think + c.codes + c.encode + c.dit).Seconds() / t
		case "score":
			return (c.sample + c.think + c.codes + c.encode + c.dit + c.vae).Seconds() / t
		}
		return 0
	}
	var smp *pipeline.Sample
	if req.Sample {
		smp = &pipeline.Sample{Query: req.SampleQuery, SkipCoT: req.SampleMode}
	}
	if b.opt.Swap.Resident() != Swappable(b) {
		progress(api.MusicProgress{Stage: "load"})
	}
	release, err := b.opt.Swap.Hold(ctx, b)
	if err != nil {
		return err
	}
	defer release()
	var song *lm.Song
	var res *pipeline.Result
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		if err := b.pipe.UseDiT(model); err != nil {
			return err
		}
		res, err = b.pipe.Generate(r, pipeline.Options{
			Seed: uint64(mp.Seed), Think: req.Thinking, LMSeed: uint64(mp.LMSeed), Sampling: sampling, Sample: smp,
			Shift: req.Shift, Timesteps: req.Timesteps, Steps: req.InferenceSteps, Guidance: guidance(req),
			Audio: audio, Score: req.Score,
			Between: func() error {
				if err := ctx.Err(); err != nil {
					return err
				}
				b.opt.Device.yield()
				return nil
			},
			Progress: func(stage string, done, total int) {
				progress(api.MusicProgress{Stage: stage, Fraction: frac(stage, done, total)})
			},
			Sampled: func(s lm.Song) {
				song = &s
				progress(api.MusicProgress{Stage: "sampled", Fraction: frac("think", 0, 1),
					Plan: &api.MusicMetadata{Caption: s.Caption, Lyrics: s.Lyrics, Genres: s.Genres, BPM: s.BPM,
						KeyScale: s.KeyScale, TimeSignature: s.TimeSignature, Duration: s.Duration,
						Language: cmpStr(s.Language, "unknown")}})
			},
			Planned: func(d *plan.Request, seconds float64) {
				// The plan fixes the length, so the estimate is redone
				// around it; what is spent is kept.
				spent, sampled, enc := c.think, c.sample, c.encode
				c = musicEstimate(d, req.Thinking, req.Sample, seconds, b.lmOpt.Bank, b.steps(model, req), forwards(model, req))
				c.think, c.sample, c.encode = spent, sampled, enc
				if req.Score {
					c.score = scoreCost(d, req.Thinking, !b.opt.NoLM, seconds)
				}
				m := &api.MusicMetadata{Caption: d.Caption, BPM: d.BPM, KeyScale: d.KeyScale,
					TimeSignature: d.TimeSignature, Duration: seconds, Language: cmpStr(d.Language, "unknown")}
				if song != nil {
					m.Lyrics, m.Genres = d.Lyrics, song.Genres
				}
				progress(api.MusicProgress{Stage: "planned", Fraction: frac("codes", 0, 1), Seconds: seconds,
					Estimate: c.total(), Plan: m})
			},
		})
		return err
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return fmt.Errorf("ace: %w", err)
	}
	progress(api.MusicProgress{Stage: "write", Fraction: 1 - c.write.Seconds()/c.total().Seconds(),
		Quality: musicQuality(res.Quality)})
	return pipeline.Write(dst, res.Audio)
}

// musicQuality is the auto score as the API reports it, or nil.
func musicQuality(q *pipeline.Quality) *api.MusicQuality {
	if q == nil || (q.LM == nil && q.Lyrics == nil && q.Aesthetics == nil && q.Song == nil && q.Render == nil && q.Loops == nil) {
		return nil
	}
	out := &api.MusicQuality{}
	if s := q.LM; s != nil {
		r := s.Reward
		out.Score, out.Adherence, out.PMI = &r, s.Conditions, s.PMI
	}
	if a := q.Lyrics; a != nil {
		out.LyricsAlignment = &api.MusicAlignment{LM: a.LM.Score, DiT: a.DiT.Score}
	}
	if a := q.Aesthetics; a != nil {
		out.Aesthetics = &api.MusicAesthetics{PQ: a.PQ, PC: a.PC, CE: a.CE, CU: a.CU}
	}
	if s := q.Song; s != nil {
		out.Song = &api.MusicSongEval{Coherence: s.Coherence, Musicality: s.Musicality, Memorability: s.Memorability,
			Clarity: s.Clarity, Naturalness: s.Naturalness}
	}
	if s := q.Render; s != nil {
		out.Render = &api.MusicAdherence{Score: s.Reward, Adherence: s.Conditions, PMI: s.PMI}
	}
	if l := q.Loops; l != nil {
		out.Loops = &api.MusicLoops{Repetition: l.Repetition, RepetitionLag: l.RepetitionLag, LoopSeconds: l.LoopSeconds,
			LoopShare: l.LoopShare, LoopLag: l.LoopLag}
	}
	return out
}

func cmpStr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// encodeCost is the VAE encoder's share of a request: the source and a
// reference's 30 s, at 11.4 ms a second of audio (A11a), plus ffmpeg's
// decode (~2 ms a second) and, for a cover, the tokenizer.
func encodeCost(a *pipeline.Audio) time.Duration {
	if a == nil {
		return 0
	}
	secs := float64(len(a.Source)/2) / plan.SampleRate
	if a.Reference != nil {
		secs += 30
	}
	return time.Duration((0.0134*secs + 0.05) * float64(time.Second))
}

// analysisSource decodes an analysis's upload and checks it is one the
// pipeline captions (pipeline.Understand's bounds).
func analysisSource(req *api.MusicRequest) ([]float32, error) {
	wav, err := pipeline.ReadAudioBytes(req.SourceAudio)
	if err != nil {
		return nil, unsupported("src_audio: %v", err)
	}
	secs := float64(len(wav)/2) / plan.SampleRate
	switch {
	case len(wav)/2 < plan.Hop:
		return nil, unsupported("src_audio is shorter than one latent (%d samples)", plan.Hop)
	case secs > plan.MaxSeconds:
		return nil, unsupported("src_audio is %.0f s, past the %d s ceiling", secs, plan.MaxSeconds)
	case pipeline.Silent(wav):
		return nil, unsupported("src_audio is silent")
	}
	return wav, nil
}

// planAnalysis plans a full_analysis_only request (MUSIC.md A13): the
// source decoded and measured, the seeds drawn (the VAE posterior's and
// the LM's), and the estimate: the encode, and the LM's pass at text
// speed for about as many steps as sample mode's.
func (b *Music) planAnalysis(req *api.MusicRequest) (*api.MusicPlan, error) {
	if b.planner == nil {
		return nil, unsupported("full_analysis_only needs the 5 Hz LM, which this server did not load (-music-lm)")
	}
	if !b.planner.HasGenres() {
		return nil, unsupported("full_analysis_only needs upstream's genres vocabulary (models/%s): %v", lm.GenresFile, b.genresErr)
	}
	if v := req.LMTemperature; v != nil && *v < 0 {
		return nil, unsupported("lm_temperature %g is negative", *v)
	}
	if v := req.LMTopP; v != nil && (*v <= 0 || *v > 1) {
		return nil, unsupported("lm_top_p %g is outside (0, 1]", *v)
	}
	wav, err := analysisSource(req)
	if err != nil {
		return nil, err
	}
	p := &api.MusicPlan{Seconds: float64(len(wav)/2) / plan.SampleRate}
	b.mu.Lock()
	p.Seed, p.LMSeed = int64(b.rng.Uint32()), int64(b.rng.Uint32())
	b.mu.Unlock()
	if req.LMSeed != nil {
		p.LMSeed = *req.LMSeed
	}
	p.Estimate = analysisEstimate(p.Seconds, b.lmOpt.Bank)
	return p, nil
}

// analysisEstimate is an analysis's expected wall time: the encode
// (encodeCost's rate), the codes' prefill (~5 ms a second) and ~sampleSteps
// steps of text.
func analysisEstimate(seconds float64, bank lm.Bank) time.Duration {
	text := 0.028
	if bank == lm.BankFP16 {
		text = 0.047
	}
	return time.Duration((0.0134*seconds + 0.05 + 0.005*seconds + sampleSteps*text) * float64(time.Second))
}

// analyse runs a planned full_analysis_only request: the source captioned
// (pipeline.Understand), and what the LM heard reported as the job's plan.
// It writes no file.
func (b *Music) analyse(ctx context.Context, req *api.MusicRequest, mp *api.MusicPlan, progress func(api.MusicProgress)) error {
	wav, err := analysisSource(req)
	if err != nil {
		return err
	}
	s := lm.UnderstandSampling
	if v := req.LMTemperature; v != nil {
		s.Temperature = *v
	}
	if v := req.LMTopP; v != nil {
		s.TopP = *v
	}
	total := analysisEstimate(mp.Seconds, b.lmOpt.Bank).Seconds()
	enc := (0.0134*mp.Seconds + 0.05 + 0.005*mp.Seconds) / total
	if b.opt.Swap.Resident() != Swappable(b) {
		progress(api.MusicProgress{Stage: "load"})
	}
	release, err := b.opt.Swap.Hold(ctx, b)
	if err != nil {
		return err
	}
	defer release()
	progress(api.MusicProgress{Stage: "encode"})
	var u *pipeline.Understanding
	err = b.opt.Device.Do(func(*vk.Device) error {
		var err error
		u, err = b.pipe.Understand(wav, pipeline.Options{
			Seed: uint64(mp.Seed), LMSeed: uint64(mp.LMSeed), Sampling: &s,
			Between: func() error {
				if err := ctx.Err(); err != nil {
					return err
				}
				b.opt.Device.yield()
				return nil
			},
			Progress: func(stage string, done, _ int) {
				progress(api.MusicProgress{Stage: stage, Fraction: enc + (1-enc)*min(1, float64(done)/sampleSteps)})
			},
		})
		return err
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return fmt.Errorf("ace: %w", err)
	}
	g := u.Song
	progress(api.MusicProgress{Stage: "understood", Fraction: 1, Seconds: u.Seconds, Plan: &api.MusicMetadata{
		Caption: g.Caption, Lyrics: g.Lyrics, Genres: g.Genres, BPM: g.BPM, KeyScale: g.KeyScale,
		TimeSignature: g.TimeSignature, Duration: g.Duration, Language: cmpStr(g.Language, "unknown")}})
	return nil
}
