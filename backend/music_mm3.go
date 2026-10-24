package backend

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/api"
	"strix-halo-vulkan/mm3"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/tokenizer"
)

// MusicMM3ID is the id MiniMax-Music3 answers to.
const MusicMM3ID = "MiniMax-Music3"

// MusicMM3Options is what cmd/serve's -music-mm3 flags come to.
type MusicMM3Options struct {
	// Models is the checkpoint root; the model is MiniMax-Music3/ under it.
	Models string
	// Device and Swap are required.
	Device *Device
	Swap   *Swap
	// LMFP16 stages the two LMs' layers as fp16, the control (2x the
	// bytes, ~10% slower a frame).
	LMFP16 bool
}

// MusicMM3 is MiniMax-Music3 (research/music-minimax.md N8): text to music
// only, beside ACE-Step on /v1/music, chosen by model (MusicRouter).
//
// **Everything is resident while it holds the Swap**, its own entry
// ("music-mm3", N-d2): the two LMs (int8), the DiT and the vocoder, ~19 GB.
// Planning tokenises the prompt on the host, to say how long a song it
// leaves room for.
//
// **The device is shared a step at a time**: the pipeline yields after
// every frame (~75 ms), DiT step (~0.2 s) and window.
//
// What a request maps to (N-d3, N-d5):
//
//   - bpm, keyscale, timesignature and vocal_language are written into the
//     caption as its Global Metadata, in the phrasing of the model's own
//     caption skill ("Basic Attributes: bpm is 92. key is E, and scale is
//     minor."), not refused;
//   - empty lyrics are "[Instrumental]" (the model needs lyrics; upstream
//     refuses an empty string);
//   - duration caps the song (the model ends it), 120 s when not given,
//     at most 360 s and what the prompt leaves of the cache;
//   - everything that is ACE's -- thinking, sample mode, captioning, the
//     audio-in tasks, the DiT schedule and the LM's sampling -- is a 400
//     that names ACE, and so is auto_score until the scorers are shared
//     (N-d6). thinking and auto_score default to on in the API, so only a
//     client's explicit true is refused; the defaults are ignored.
type MusicMM3 struct {
	opt  MusicMM3Options
	tok  *tokenizer.Tokenizer
	pipe *mm3.Pipeline // nil while another vertical holds the slot

	mu  sync.Mutex
	rng *rand.Rand
}

// NewMusicMM3 stages the model through the slot.
func NewMusicMM3(opt MusicMM3Options) (*MusicMM3, error) {
	if opt.Device == nil || opt.Swap == nil {
		return nil, fmt.Errorf("backend: MiniMax-Music3 needs the device and a swap slot")
	}
	tok, err := tokenizer.Load(filepath.Join(opt.Models, MusicMM3ID, "tokenizer"))
	if err != nil {
		return nil, fmt.Errorf("backend: MiniMax-Music3's tokenizer: %w", err)
	}
	b := &MusicMM3{opt: opt, tok: tok, rng: rand.New(rand.NewSource(rand.Int63()))}
	if err := opt.Swap.Load(b); err != nil {
		return nil, err
	}
	return b, nil
}

// SwapName is the vertical's name in the swap log.
func (b *MusicMM3) SwapName() string { return "music-mm3" }

// Load stages the pipeline; it submits nothing.
func (b *MusicMM3) Load() error {
	o := mm3.DefaultOptions()
	if b.opt.LMFP16 {
		o.Bank = mm3.BankFP16
	}
	p, err := mm3.Load(b.opt.Device.dev, filepath.Join(b.opt.Models, MusicMM3ID), o)
	if err != nil {
		return fmt.Errorf("backend: loading MiniMax-Music3: %w", err)
	}
	b.pipe = p
	return nil
}

// Unload frees the pipeline.
func (b *MusicMM3) Unload() {
	if b.pipe != nil {
		b.pipe.Destroy()
		b.pipe = nil
	}
}

// Close frees the device objects.
func (b *MusicMM3) Close() { b.Unload() }

// Models is the one model.
func (b *MusicMM3) Models() []api.Model {
	return []api.Model{{ID: MusicMM3ID, Object: "model", OwnedBy: "local"}}
}

// MusicInfo is what a request may ask for.
func (b *MusicMM3) MusicInfo() api.MusicInfo {
	return api.MusicInfo{MinSeconds: 1, MaxSeconds: mm3.MaxSeconds, FallbackSeconds: mm3FallbackSeconds,
		SampleRate: mm3.SampleRate, Formats: formats(), Steps: mm3.DiTSteps}
}

const mm3FallbackSeconds = 120

// IsMusicMM3 says whether a request's model names MiniMax-Music3 (N-d4):
// the id, or a name with "minimax" or "music3" as a word.
func IsMusicMM3(model string) bool {
	if strings.EqualFold(model, MusicMM3ID) {
		return true
	}
	for _, w := range strings.FieldsFunc(strings.ToLower(model), func(r rune) bool {
		return r == '-' || r == '_' || r == '/' || r == ' ' || r == '.'
	}) {
		if w == "minimax" || w == "music3" {
			return true
		}
	}
	return false
}

// refuseACE is the 400 for a field only ACE serves.
func refuseACE(what string) error {
	return unsupported("%s is ACE-Step's; MiniMax-Music3 is text to music only (use model %q or leave model unset)", what, pipeline.DiTModels["base"])
}

// mm3Caption writes the request's metas into the caption (N-d3).
func mm3Caption(req *api.MusicRequest) string {
	var m []string
	if req.BPM > 0 {
		m = append(m, fmt.Sprintf("bpm is %d.", req.BPM))
	}
	if ks := strings.Fields(req.KeyScale); len(ks) > 0 {
		if len(ks) > 1 {
			m = append(m, fmt.Sprintf("key is %s, and scale is %s.", ks[0], strings.ToLower(strings.Join(ks[1:], " "))))
		} else {
			m = append(m, fmt.Sprintf("key is %s.", ks[0]))
		}
	}
	if ts := strings.TrimSpace(req.TimeSignature); ts != "" {
		if !strings.Contains(ts, "/") {
			ts += "/4"
		}
		m = append(m, fmt.Sprintf("The time signature is %s.", ts))
	}
	if l := strings.TrimSpace(req.Language); l != "" && l != "unknown" {
		if name, ok := languageNames[l]; ok {
			l = name
		}
		m = append(m, fmt.Sprintf("Sung in %s.", l))
	}
	caption := strings.TrimSpace(req.Caption)
	if len(m) == 0 {
		return caption
	}
	metas := strings.Join(m, " ")
	// A caption that has its own Basic Attributes line takes the request's
	// metas there, in place of any bpm or key it already states.
	if loc := basicAttrs.FindStringIndex(caption); loc != nil {
		end := strings.IndexByte(caption[loc[1]:], '\n')
		if end < 0 {
			end = len(caption) - loc[1]
		}
		rest := caption[loc[1] : loc[1]+end]
		if req.BPM > 0 {
			rest = statedBPM.ReplaceAllString(rest, "")
		}
		if req.KeyScale != "" {
			rest = statedKey.ReplaceAllString(rest, "")
		}
		return caption[:loc[1]] + metas + " " + strings.TrimSpace(rest) + caption[loc[1]+end:]
	}
	return "Global Metadata\nBasic Attributes: " + metas + "\n" + caption
}

var (
	basicAttrs = regexp.MustCompile(`(?m)^Basic Attributes:\s*`)
	statedBPM  = regexp.MustCompile(`bpm is \d+\.\s*`)
	statedKey  = regexp.MustCompile(`key is [^,]+, and scale is \w+\.\s*`)
)

var languageNames = map[string]string{"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
	"fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "yue": "Cantonese"}

func mm3Lyrics(req *api.MusicRequest) string {
	if strings.TrimSpace(req.Lyrics) == "" {
		return "[Instrumental]"
	}
	return req.Lyrics
}

// PlanMusic validates a request: refusals, the format, the seed, and the
// length the prompt leaves room for.
func (b *MusicMM3) PlanMusic(req *api.MusicRequest) (*api.MusicPlan, error) {
	switch {
	case req.Analysis:
		return nil, refuseACE("full_analysis_only (captioning)")
	case req.Thinking && req.ThinkingSet:
		return nil, refuseACE("thinking")
	case req.Sample:
		return nil, refuseACE("sample mode")
	case req.Task != "" && req.Task != pipeline.Text2Music:
		return nil, refuseACE("task " + req.Task)
	case req.SourceAudio != nil || req.ReferenceAudio != nil:
		return nil, refuseACE("audio in (src_audio, reference_audio)")
	case req.Shift != 0 || req.Timesteps != nil || req.InferenceSteps != 0 || req.GuidanceScale != nil ||
		req.CFGStart != nil || req.CFGEnd != nil:
		return nil, refuseACE("the DiT schedule (shift, timesteps, inference_steps, guidance)")
	case req.LMTemperature != nil || req.LMCFGScale != nil || req.LMTopP != nil:
		return nil, refuseACE("the 5 Hz LM's sampling")
	case req.LMSeed != nil:
		return nil, refuseACE("lm_seed")
	case req.Score && req.ScoreSet:
		return nil, unsupported("auto_score is not served for MiniMax-Music3 yet (research/music-minimax.md N-d6); score with model %q", pipeline.DiTModels["base"])
	case strings.TrimSpace(req.Caption) == "":
		return nil, unsupported("MiniMax-Music3 needs a caption (it works best long and structured: research/music-minimax.md N0)")
	}
	if req.BPM != 0 && (req.BPM < 30 || req.BPM > 300) {
		return nil, unsupported("bpm %d is outside 30-300", req.BPM)
	}
	if ts := strings.TrimSpace(req.TimeSignature); ts != "" && !validTimesigs[ts] {
		return nil, unsupported("time signature %q: 2, 3, 4 or 6 (or 2/4, 3/4, 4/4, 6/8)", req.TimeSignature)
	}
	if req.Duration != 0 && (req.Duration < 1 || req.Duration > mm3.MaxSeconds) {
		return nil, unsupported("duration %g s is outside 1-%d s", req.Duration, mm3.MaxSeconds)
	}
	format := req.Format
	if format == "" {
		format = "mp3"
	}
	if _, ok := pipeline.Formats[format]; !ok {
		return nil, unsupported("audio format %q: %s", req.Format, strings.Join(formats(), ", "))
	}
	ids, _, err := mm3.Prompt(b.tok, mm3Caption(req), mm3Lyrics(req))
	if err != nil {
		return nil, unsupported("%v", err)
	}
	room := float64(mm3.DefaultOptions().MaxLen-len(ids)-1) / mm3.FrameRate
	secs := req.Duration
	if secs == 0 {
		secs = mm3FallbackSeconds
	}
	if room < 1 {
		return nil, unsupported("the prompt is %d tokens, which leaves the cache no room for a song", len(ids))
	}
	if secs > room {
		if req.Duration != 0 {
			return nil, unsupported("the prompt is %d tokens, which leaves room for %.0f s of song, not %g; shorten the caption or lyrics, or the song",
				len(ids), math.Floor(room), req.Duration)
		}
		secs = math.Floor(room)
	}
	p := &api.MusicPlan{Format: format, Model: MusicMM3ID, Steps: mm3.DiTSteps, Seconds: secs}
	b.mu.Lock()
	p.Seed = int64(b.rng.Uint32())
	b.mu.Unlock()
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	p.Estimate = mm3Estimate(len(ids), secs).total()
	return p, nil
}

// mm3Cost is a request's expected wall time by stage, from N3-N7's
// measurements with the device to itself (int8 LMs): the prefill at
// ~0.75 ms a token of the pair, 75 ms a frame (25 a second), 6.7 s a
// window (30 CFG forwards; a window every 4 s of song past the first 8)
// and 0.22 s of vocoder a window, then ffmpeg's encode.
type mm3Cost struct {
	prefill, frames, dit, vocoder, write time.Duration
}

func (c mm3Cost) total() time.Duration { return c.prefill + c.frames + c.dit + c.vocoder + c.write }

func mm3Estimate(tokens int, seconds float64) mm3Cost {
	sec := func(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }
	windows := float64(len(mm3.WindowStarts(int(seconds * mm3.FrameRate))))
	return mm3Cost{
		prefill: sec(0.00075*float64(tokens) + 0.1),
		frames:  sec(0.075 * seconds * mm3.FrameRate),
		dit:     sec(6.7 * windows),
		vocoder: sec(0.22 * windows),
		write:   sec(0.004 * seconds),
	}
}

// GenerateMusic runs a planned request and writes the audio to dst.
func (b *MusicMM3) GenerateMusic(ctx context.Context, req *api.MusicRequest, mp *api.MusicPlan, dst string, progress func(api.MusicProgress)) error {
	caption, lyrics := mm3Caption(req), mm3Lyrics(req)
	ids, _, err := mm3.Prompt(b.tok, caption, lyrics)
	if err != nil {
		return err
	}
	c := mm3Estimate(len(ids), mp.Seconds)
	frac := func(stage string, done, total int) float64 {
		t := c.total().Seconds()
		part := func(d time.Duration) float64 {
			if total <= 0 {
				return 0
			}
			return d.Seconds() * float64(min(done, total)) / float64(total)
		}
		switch stage {
		case "frames":
			return (c.prefill.Seconds() + part(c.frames)) / t
		case "windows":
			return (c.prefill+c.frames).Seconds()/t + part(c.dit+c.vocoder)/t
		}
		return 0
	}
	if b.opt.Swap.Resident() != Swappable(b) {
		progress(api.MusicProgress{Stage: "load"})
	}
	release, err := b.opt.Swap.Hold(ctx, b)
	if err != nil {
		return err
	}
	defer release()
	var wav [2][]float32
	err = b.opt.Device.Do(func(*vk.Device) error {
		b.pipe.Between = func() error {
			if err := ctx.Err(); err != nil {
				return err
			}
			b.opt.Device.yield()
			return nil
		}
		defer func() { b.pipe.Between = nil }()
		var err error
		wav, _, err = b.pipe.Generate(mm3.Request{Caption: caption, Lyrics: lyrics, Seconds: mp.Seconds, Seed: uint64(mp.Seed)},
			func(stage string, done, total int) {
				if stage == "planned" {
					// The song's length is known: redo the estimate around
					// it, keeping what the frames took.
					secs := float64(done) / mm3.FrameRate
					spent := c.prefill + c.frames
					c = mm3Estimate(len(ids), secs)
					c.prefill, c.frames = spent, 0
					progress(api.MusicProgress{Stage: "planned", Fraction: frac("windows", 0, 1), Seconds: secs,
						Estimate: c.total(), Plan: &api.MusicMetadata{Caption: caption, Lyrics: lyrics, BPM: req.BPM,
							KeyScale: req.KeyScale, TimeSignature: req.TimeSignature, Duration: secs,
							Language: cmpStr(req.Language, "unknown")}})
					return
				}
				progress(api.MusicProgress{Stage: stage, Fraction: frac(stage, done, total)})
			})
		return err
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return fmt.Errorf("mm3: %w", err)
	}
	secs := float64(len(wav[0])) / mm3.SampleRate
	progress(api.MusicProgress{Stage: "write", Fraction: 1 - c.write.Seconds()/c.total().Seconds(), Seconds: secs})
	inter := make([]float32, 2*len(wav[0]))
	for i := range wav[0] {
		inter[2*i], inter[2*i+1] = wav[0][i], wav[1][i]
	}
	return pipeline.WriteRate(dst, inter, mm3.SampleRate)
}
