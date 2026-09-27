// Package pipeline is ACE-Step 1.5's text2music request end to end, the
// DiT-only path (MUSIC.md A6): a caption, lyrics and metadata in, a 48 kHz
// stereo song out.
//
//  1. ace/plan formats the caption and lyric prompts and fixes the length.
//  2. The caption runs through Qwen3-Embedding-0.6B (`embed`, the same bytes
//     as upstream's text encoder), last_hidden_state; the lyrics are that
//     model's embedding rows only.
//  3. ace/dit encodes the lyrics and the timbre (the shipped silence, with
//     no reference audio), packs the condition sequence, and runs 8 Euler
//     steps with DCW from seeded noise (ace/plan.Sample) over the context
//     [silence | chunk mask 1.0].
//  4. ace/vae decodes the latents; the audio is peak-normalised to -1 dBFS,
//     as upstream writes it.
package pipeline

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"strix-halo-vulkan/ace/dit"
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
}

// Options are the per-request knobs beyond the request itself.
type Options struct {
	Seed uint64
	// Noise, when set, replaces the seeded noise ([T, 64]): the gates feed
	// the oracle's.
	Noise []float32
	// Step, when set, sees the latents after every step.
	Step func(i int, x []float32)
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
	start := time.Now()
	var tm Timings
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

	T := r.LatentLength()
	if T*plan.LatentChannels > len(p.silence) {
		return nil, fmt.Errorf("ace: %d latents is past the %d-latent silence", T, len(p.silence)/plan.LatentChannels)
	}
	ctx := qwen.NewMat(T, 2*plan.LatentChannels)
	for i := 0; i < T; i++ {
		row := ctx.Row(i)
		copy(row, p.silence[i*plan.LatentChannels:(i+1)*plan.LatentChannels])
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
	lat, err := plan.Sample(noise, plan.Schedule(plan.DefaultShift, nil), plan.DefaultDCW, func(x []float32, t float32) ([]float32, error) {
		v, took, err := p.DiT.Step(&qwen.Mat{Rows: T, Cols: plan.LatentChannels, Data: x}, ctx, t)
		if err != nil {
			return nil, err
		}
		tm.Steps = append(tm.Steps, took)
		return v.Data, nil
	}, o.Step)
	if err != nil {
		return nil, err
	}
	tm.DiT = time.Since(t0)

	t0 = time.Now()
	wav, _, err := p.VAE.Decode(lat)
	if err != nil {
		return nil, err
	}
	vae.Normalize(wav, -1)
	tm.VAE = time.Since(t0)
	tm.Total = time.Since(start)
	return &Result{Audio: wav, Latents: lat, Seconds: float64(T*plan.Hop) / plan.SampleRate, Timings: tm}, nil
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
