// Package pipeline is ACE-Step 1.5's text2music request end to end: a
// caption, lyrics and metadata in, a 48 kHz stereo song out. The DiT-only
// path is MUSIC.md A6; with the 5 Hz LM loaded (LoadLM) and Options.Think,
// the thinking path (A8) puts the LM in front:
//
//  0. ace/lm writes a CoT of the metadata the request left out (phase 1)
//     and then 5 audio codes a second under CFG (phase 2). The DiT's
//     caption and metas come from the CoT where the request gave none, the
//     length is the codes', and the codes, detokenized to 25 Hz hints,
//     replace the silence as the source latents (upstream's is_covers
//     path; the chunk mask stays 1.0).
//
//  1. ace/plan formats the caption and lyric prompts and fixes the length.
//
//  2. The caption runs through Qwen3-Embedding-0.6B (`embed`, the same bytes
//     as upstream's text encoder), last_hidden_state; the lyrics are that
//     model's embedding rows only.
//
//  3. ace/dit encodes the lyrics and the timbre (the shipped silence, with
//     no reference audio), packs the condition sequence, and runs 8 Euler
//     steps with DCW from seeded noise (ace/plan.Sample) over the context
//     [silence | chunk mask 1.0].
//
//  4. ace/vae decodes the latents; the audio is peak-normalised to -1 dBFS,
//     as upstream writes it.
package pipeline

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/ace/dit"
	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/ace/vae"
	"strix-halo-vulkan/embed"
	h3vae "strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// Dirs are the checkpoints.
type Dirs struct {
	DiT  string // acestep-v15-xl-turbo: DiT, encoders, silence_latent.safetensors
	VAE  string // Ace-Step1.5/vae
	Text string // Qwen3-Embedding-0.6B
}

// DefaultDirs is the layout under models/.
func DefaultDirs(models string) Dirs {
	return Dirs{
		DiT:  filepath.Join(models, "acestep-v15-xl-turbo"),
		VAE:  filepath.Join(models, "Ace-Step1.5", "vae"),
		Text: filepath.Join(models, "Qwen3-Embedding-0.6B"),
	}
}

// VAEWindow is the decode tile, in latents (20 s of audio).
const VAEWindow = 512

// Pipeline holds every model of the DiT-only path on the device.
type Pipeline struct {
	Text  *embed.GPU
	Tok   *tokenizer.Tokenizer
	DiT   *dit.GPU
	VAE   *vae.GPU
	table []float32 // the text encoder's embedding rows, for the lyrics
	width int
	// silence is silence_latent [15000, 64], time-major.
	silence []float32
	// The 5 Hz LM, when loaded.
	LM    *lm.Planner
	lmGPU *lm.GPU
}

// LMDir is the 5 Hz LM's checkpoint under models/.
func LMDir(models string) string { return filepath.Join(models, "acestep-5Hz-lm-4B") }

// LoadLM stages the 5 Hz LM (8 GB of fp16 and its KV cache) for the
// thinking path.
func (p *Pipeline) LoadLM(dev *vk.Device, dir string, o lm.Options) error {
	g, err := lm.Load(dev, dir, o)
	if err != nil {
		return err
	}
	tk, err := tokenizer.Load(dir)
	if err != nil {
		g.Destroy()
		return err
	}
	pl, err := lm.NewPlanner(g, tk)
	if err != nil {
		g.Destroy()
		return err
	}
	p.LM, p.lmGPU = pl, g
	return nil
}

// New stages the text encoder, the DiT and the VAE for songs of up to
// maxSeconds.
func New(dev *vk.Device, d Dirs, maxSeconds float64) (*Pipeline, error) {
	if maxSeconds <= 0 || maxSeconds > plan.MaxSeconds {
		return nil, fmt.Errorf("ace: a %.0f s ceiling is outside (0, %d]", maxSeconds, plan.MaxSeconds)
	}
	p := &Pipeline{}
	var err error
	if p.Tok, err = tokenizer.Load(d.Text); err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(d.Text)
	if err != nil {
		return nil, err
	}
	cfg, err := embed.LoadConfig(d.Text)
	if err != nil {
		set.Close()
		return nil, err
	}
	_, p.table, err = qwen.LoadEmbedding(set, cfg)
	set.Close()
	if err != nil {
		return nil, err
	}
	p.width = cfg.HiddenSize
	f, err := safetensors.Open(filepath.Join(d.DiT, "silence_latent.safetensors"))
	if err != nil {
		return nil, err
	}
	t, err := f.Get("silence_latent")
	if err == nil {
		p.silence, err = t.F32(nil)
	}
	f.Close()
	if err != nil {
		return nil, err
	}
	if p.Text, err = embed.NewGPU(dev, d.Text, plan.TextMaxTokens); err != nil {
		return nil, err
	}
	latents := int(maxSeconds*plan.SampleRate) / plan.Hop
	if p.DiT, err = dit.NewGPU(dev, d.DiT, (max(latents, plan.MinLatents)+1)/2); err != nil {
		p.Destroy()
		return nil, err
	}
	if p.VAE, err = vae.NewGPU(dev, d.VAE, VAEWindow); err != nil {
		p.Destroy()
		return nil, err
	}
	return p, nil
}

// Destroy releases the device objects.
func (p *Pipeline) Destroy() {
	if p.lmGPU != nil {
		p.lmGPU.Destroy()
	}
	if p.VAE != nil {
		p.VAE.Destroy()
	}
	if p.DiT != nil {
		p.DiT.Destroy()
	}
	if p.Text != nil {
		p.Text.Destroy()
	}
}

// Timings is where a request's time went.
type Timings struct {
	Think, Codes                  lm.Stats // the LM's two phases (thinking path)
	Text, Encode, DiT, VAE, Total time.Duration
	Steps                         []time.Duration // device time a forward
}

// Result is one generated song.
type Result struct {
	// Audio is interleaved stereo at 48 kHz, peak-normalised to -1 dBFS.
	Audio   []float32
	Latents []float32 // [T, 64], time-major
	Seconds float64
	Timings Timings
	// The thinking path's plan: the metadata the DiT was given (the CoT's
	// where the request left a field out), what the LM wrote in phase 1
	// (empty when the request gave every meta), and the codes.
	Meta  lm.Meta
	CoT   string
	Codes []int
	DiT   *plan.Request
}

// Options are the per-request knobs beyond the request itself.
type Options struct {
	Seed uint64
	// Noise, when set, replaces the seeded noise ([T, 64]): the gates feed
	// the oracle's.
	Noise []float32
	// Step, when set, sees the latents after every step.
	Step func(i int, x []float32)
	// Think runs the 5 Hz LM first (LoadLM): upstream's default.
	Think bool
	// LMSeed seeds the LM's sampling.
	LMSeed uint64
	// Sampling overrides the LM's (lm.DefaultSampling when zero).
	Sampling *lm.Sampling
	// Shift and Timesteps are the DiT schedule: plan.Schedule's (0 and nil
	// take shift 3's table).
	Shift     float64
	Timesteps []float64

	// Between, when set, runs between units of device work -- every LM
	// step and every DiT forward, and before the decode: a server yields
	// the device there, and an error (a cancelled request) stops the run.
	Between func() error
	// Progress, when set, sees the run move: a stage ("think", "codes",
	// "dit", "vae"), units done and the stage's length (0 when the model
	// decides it, as phase 1's does).
	Progress func(stage string, done, total int)
	// Planned, when set, is told what the DiT will be asked for once the
	// LM has planned it: the request with the CoT's metas and the length.
	Planned func(d *plan.Request, seconds float64)
}

func (o *Options) between() error {
	if o.Between != nil {
		return o.Between()
	}
	return nil
}

func (o *Options) progress(stage string, done, total int) {
	if o.Progress != nil {
		o.Progress(stage, done, total)
	}
}

// Ids tokenises the caption and lyric prompts as the text encoder's
// tokenizer does (plan.Tokens: truncation with room for the appended
// end-of-text).
func (p *Pipeline) Ids(r *plan.Request) (caption, lyrics []int32, err error) {
	c, err := p.Tok.Encode(r.CaptionPrompt())
	if err != nil {
		return nil, nil, err
	}
	l, err := p.Tok.Encode(r.LyricsPrompt())
	if err != nil {
		return nil, nil, err
	}
	return plan.Tokens(c, plan.TextMaxTokens), plan.Tokens(l, plan.LyricMaxTokens), nil
}

// Condition runs the caption through the text encoder and looks the
// lyrics up: the condition encoder's two text inputs.
func (p *Pipeline) Condition(r *plan.Request) (text, lyric *qwen.Mat, err error) {
	cids, lids, err := p.Ids(r)
	if err != nil {
		return nil, nil, err
	}
	if text, err = p.Text.Hidden(cids); err != nil {
		return nil, nil, err
	}
	lyric = qwen.NewMat(len(lids), p.width)
	for i, id := range lids {
		copy(lyric.Row(i), p.table[int(id)*p.width:(int(id)+1)*p.width])
	}
	return text, lyric, nil
}

// Generate runs one request.
func (p *Pipeline) Generate(r *plan.Request, o Options) (*Result, error) {
	if !o.Think {
		if o.Planned != nil {
			o.Planned(r, float64(r.LatentLength()*plan.Hop)/plan.SampleRate)
		}
		return p.generate(r, o, r.LatentLength(), nil, &Result{})
	}
	if p.LM == nil {
		return nil, fmt.Errorf("ace: thinking without the LM loaded")
	}
	start := time.Now()
	res := &Result{}
	rng := rand.New(rand.NewPCG(o.LMSeed, 0x5a))
	p.LM.S = lm.DefaultSampling
	if o.Sampling != nil {
		p.LM.S = *o.Sampling
	}
	p.LM.Between = o.Between
	p.LM.Step = func(phase, done, total int) {
		o.progress(map[int]string{1: "think", 2: "codes"}[phase], done, total)
	}
	defer func() { p.LM.Between, p.LM.Step = nil, nil }()
	meta, cot, st, err := p.LM.Think(r.Caption, r.Lyrics, lm.UserMeta(r), rng)
	if err != nil {
		return nil, err
	}
	res.Meta, res.CoT, res.Timings.Think = meta, cot, st
	target, err := lm.TargetCodes(r.Duration, meta)
	if err != nil {
		return nil, err
	}
	if o.Planned != nil {
		o.Planned(DiTRequest(r, meta, cot != ""), float64(max(plan.MinLatents, lm.CodesPerSecond*target)*plan.Hop)/plan.SampleRate)
	}
	codes, st, err := p.LM.Codes(r.Caption, r.Lyrics, meta, target, rng)
	if err != nil {
		return nil, err
	}
	res.Codes, res.Timings.Codes = codes, st
	res.DiT = DiTRequest(r, meta, cot != "")
	T := max(plan.MinLatents, lm.CodesPerSecond*len(codes))
	hints, err := p.Hints(codes, T)
	if err != nil {
		return nil, err
	}
	out, err := p.generate(res.DiT, o, T, hints, res)
	if err != nil {
		return nil, err
	}
	out.Timings.Total = time.Since(start)
	return out, nil
}

// DiTRequest is what the DiT is asked for after the LM has planned
// (_update_metadata_from_lm): the CoT's bpm, key, time signature and
// duration where the request left them out, and -- when phase 1 ran -- its
// rewritten caption (use_cot_caption). The language stays the request's:
// upstream reads the CoT's `language` as `vocal_language`, a key the parse
// never writes. And the task is no longer text2music: audio codes switch it
// to `cover` (_resolve_generate_music_task), whose instruction is the LM's
// own -- with the default cover strength 1.0 and cover noise 0, that line
// of the caption prompt is all the switch changes.
func DiTRequest(r *plan.Request, meta lm.Meta, cotRan bool) *plan.Request {
	d := *r
	d.Instruction = lm.Instruction
	if d.BPM == 0 {
		if v, err := strconv.Atoi(meta["bpm"]); err == nil && v > 0 {
			d.BPM = v
		}
	}
	if d.KeyScale == "" && meta["keyscale"] != "" && meta["keyscale"] != "N/A" {
		d.KeyScale = meta["keyscale"]
	}
	if d.TimeSignature == "" && meta["timesignature"] != "" && meta["timesignature"] != "N/A" {
		d.TimeSignature = meta["timesignature"]
	}
	if d.Duration <= 0 {
		if v, err := strconv.ParseFloat(meta["duration"], 64); err == nil {
			d.Duration = v
		}
	}
	if cotRan && meta["caption"] != "" {
		d.Caption = meta["caption"]
	}
	return &d
}

// Hints detokenizes codes into T rows of 25 Hz source latents, padded with
// the silence latent or cropped (_prepare_precomputed_lm_hints).
func (p *Pipeline) Hints(codes []int, T int) ([]float32, error) {
	ids := make([]int32, len(codes))
	for i, c := range codes {
		ids[i] = int32(c)
	}
	h, err := p.DiT.Detokenize(ids)
	if err != nil {
		return nil, err
	}
	out := make([]float32, T*plan.LatentChannels)
	n := copy(out, h.Data)
	copy(out[n:], p.silence[:len(out)-n])
	return out, nil
}

// generate is the DiT and the VAE over T latents, from the silence or from
// the LM's hints as the source.
func (p *Pipeline) generate(r *plan.Request, o Options, T int, hints []float32, res *Result) (*Result, error) {
	start := time.Now()
	tm := res.Timings
	text, lyric, err := p.Condition(r)
	if err != nil {
		return nil, err
	}
	tm.Text = time.Since(start)

	t0 := time.Now()
	timbre := &qwen.Mat{Rows: plan.TimbreLatents, Cols: plan.LatentChannels, Data: p.silence[:plan.TimbreLatents*plan.LatentChannels]}
	enc, err := p.DiT.Encode(text, lyric, timbre)
	if err != nil {
		return nil, err
	}
	if err := p.DiT.Begin(enc); err != nil {
		return nil, err
	}
	tm.Encode = time.Since(t0)

	if T*plan.LatentChannels > len(p.silence) {
		return nil, fmt.Errorf("ace: %d latents is past the %d-latent silence", T, len(p.silence)/plan.LatentChannels)
	}
	src := p.silence
	if hints != nil {
		src = hints
	}
	ctx := qwen.NewMat(T, 2*plan.LatentChannels)
	for i := 0; i < T; i++ {
		row := ctx.Row(i)
		copy(row, src[i*plan.LatentChannels:(i+1)*plan.LatentChannels])
		for j := plan.LatentChannels; j < len(row); j++ {
			row[j] = plan.ChunkMask
		}
	}
	noise := o.Noise
	if noise == nil {
		if noise, err = h3vae.TorchRandn(o.Seed, T*plan.LatentChannels); err != nil {
			return nil, err
		}
	}
	if len(noise) != T*plan.LatentChannels {
		return nil, fmt.Errorf("ace: %d noise values for %d latents", len(noise), T)
	}

	t0 = time.Now()
	shift := o.Shift
	if shift == 0 {
		shift = plan.DefaultShift
	}
	sched := plan.Schedule(shift, o.Timesteps)
	lat, err := plan.Sample(noise, sched, plan.DefaultDCW, func(x []float32, t float32) ([]float32, error) {
		if err := o.between(); err != nil {
			return nil, err
		}
		v, took, err := p.DiT.Step(&qwen.Mat{Rows: T, Cols: plan.LatentChannels, Data: x}, ctx, t)
		if err != nil {
			return nil, err
		}
		tm.Steps = append(tm.Steps, took)
		o.progress("dit", len(tm.Steps), len(sched))
		return v.Data, nil
	}, o.Step)
	if err != nil {
		return nil, err
	}
	tm.DiT = time.Since(t0)

	if err := o.between(); err != nil {
		return nil, err
	}
	o.progress("vae", 0, 1)
	t0 = time.Now()
	wav, _, err := p.VAE.Decode(lat)
	if err != nil {
		return nil, err
	}
	vae.Normalize(wav, -1)
	tm.VAE = time.Since(t0)
	tm.Total = time.Since(start)
	res.Audio, res.Latents, res.Seconds, res.Timings = wav, lat, float64(T*plan.Hop)/plan.SampleRate, tm
	return res, nil
}

// FFmpeg is the encoder (decision 7): we write none of our own.
var FFmpeg = "ffmpeg"

// Formats are the containers Write knows, by extension.
var Formats = map[string][]string{
	"wav":  {"-c:a", "pcm_s16le"},
	"flac": {"-c:a", "flac"},
	"mp3":  {"-c:a", "libmp3lame", "-b:a", "128k"},
	"opus": {"-c:a", "libopus", "-b:a", "128k"},
}

// Write encodes interleaved 48 kHz stereo to path, in the format its
// extension names, through ffmpeg.
func Write(path string, audio []float32) error {
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
	codec, ok := Formats[ext]
	if !ok {
		return fmt.Errorf("ace: no encoder for %q (wav, flac, mp3, opus)", ext)
	}
	args := append([]string{"-hide_banner", "-loglevel", "error", "-y",
		"-f", "f32le", "-ar", fmt.Sprint(plan.SampleRate), "-ac", "2", "-i", "pipe:0"}, codec...)
	args = append(args, path)
	cmd := exec.Command(FFmpeg, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("ace: %s: %w", FFmpeg, err)
	}
	w := bufio.NewWriterSize(stdin, 1<<20)
	var b [4]byte
	var werr error
	for _, v := range audio {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		if _, werr = w.Write(b[:]); werr != nil {
			break
		}
	}
	if werr == nil {
		werr = w.Flush()
	}
	stdin.Close()
	if err := cmd.Wait(); err != nil {
		os.Remove(path)
		return fmt.Errorf("ace: %s: %w: %s", FFmpeg, err, stderr.String())
	}
	return werr
}
