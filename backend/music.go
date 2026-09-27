package backend

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/api"
	"strix-halo-vulkan/vk"
)

// MusicOptions is what cmd/serve's flags come to.
type MusicOptions struct {
	// Models is the checkpoint root: acestep-v15-xl-turbo/, Ace-Step1.5/vae,
	// Qwen3-Embedding-0.6B/ and acestep-5Hz-lm-4B/ under it.
	Models string
	// Device is required.
	Device *Device
	// NoLM leaves the 5 Hz LM out (8 GB of weights and 1.8 GB of KV
	// cache): the DiT-only path, and a thinking request is a 400.
	NoLM bool
	// ID is the model id this backend answers to.
	ID string
}

const defaultMusicModelID = "acestep-v15-xl-turbo"

// Music is the ACE-Step 1.5 adapter: an api.MusicBackend over ace/pipeline
// (MUSIC.md A9).
//
// **Everything is resident** (A-o4): the DiT, its encoders, the VAE, the
// text encoder and the LM with its cache, ~21 GB. Staging them is ~15 s and
// a 60 s song is 26 s, so staging per request would be most of it.
//
// **The device is shared a step at a time.** A request holds it only
// between yields: after every LM step (~50 ms) and every DiT forward
// (0.1-2.4 s by length), so speech waits for one of those, not the song.
type Music struct {
	opt  MusicOptions
	id   string
	pipe *pipeline.Pipeline

	mu  sync.Mutex // rng
	rng *rand.Rand
}

// NewMusic stages the models on the device.
func NewMusic(opt MusicOptions) (*Music, error) {
	if opt.ID == "" {
		opt.ID = defaultMusicModelID
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the music pipeline needs a device; there is no host path for it")
	}
	b := &Music{opt: opt, id: opt.ID, rng: rand.New(rand.NewSource(rand.Int63()))}
	err := opt.Device.Do(func(d *vk.Device) error {
		p, err := pipeline.New(d, pipeline.DefaultDirs(opt.Models), plan.MaxSeconds)
		if err != nil {
			return err
		}
		if !opt.NoLM {
			if err := p.LoadLM(d, pipeline.LMDir(opt.Models), lm.DefaultOptions()); err != nil {
				p.Destroy()
				return err
			}
		}
		b.pipe = p
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("backend: loading ACE-Step from %s: %w", opt.Models, err)
	}
	return b, nil
}

// Models reports the one model this backend serves.
func (b *Music) Models() []api.Model {
	return []api.Model{{ID: b.id, Object: "model", OwnedBy: "local"}}
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
		Thinking: b.pipe.LM != nil, Steps: len(plan.Schedule(plan.DefaultShift, nil))}
}

// Close frees the device objects. The job queue must be closed first.
func (b *Music) Close() {
	b.opt.Device.Do(func(*vk.Device) error {
		b.pipe.Destroy()
		return nil
	})
}

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
	if req.Thinking && b.pipe.LM == nil {
		return nil, unsupported("thinking needs the 5 Hz LM, which this server did not load (-music-lm)")
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
	p := &api.MusicPlan{Format: format, Steps: len(plan.Schedule(cmpOr(req.Shift, plan.DefaultShift), req.Timesteps))}
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
	case req.Duration > 0:
		p.Seconds = float64(musicLatents(r, req.Thinking)*plan.Hop) / plan.SampleRate
	case !req.Thinking:
		p.Seconds = plan.FallbackSeconds
	}
	if req.Thinking {
		// The codes phase's prompt is the CoT's plus the CoT, and the codes
		// follow it in the same cache.
		secs := cmpOr(p.Seconds, plan.MaxSeconds)
		need, err := b.pipe.LM.Positions(r.Caption, r.Lyrics, secs)
		if err != nil {
			return nil, err
		}
		if need > b.pipe.LM.MaxLen() {
			return nil, unsupported("the LM would need up to %d positions for %.0f s of codes after these lyrics, and holds %d; shorten the lyrics or the song",
				need, secs, b.pipe.LM.MaxLen())
		}
	}
	p.Estimate = musicEstimate(r, req.Thinking, cmpOr(p.Seconds, 120)).total()
	return p, nil
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
// measurements with the device to itself: an LM step is 50 ms (A7), a CoT
// ~150 of them; a DiT forward is 2.65e-4·L + 7.16e-9·L² s over L = 12.5 a
// second tokens (A3's 0.86 s at 3,000 and 2.39 s at 7,500) plus 35 ms of
// host; the VAE is 10.4 ms a second of audio (A6); encoding the file is
// ffmpeg's.
type musicCost struct {
	think, codes, dit, vae, write time.Duration
}

func (c musicCost) total() time.Duration { return c.think + c.codes + c.dit + c.vae + c.write }

func musicEstimate(r *plan.Request, thinking bool, seconds float64) musicCost {
	sec := func(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }
	var c musicCost
	if thinking {
		if !lm.UserMeta(r).HasAll() {
			c.think = sec(150 * 0.050)
		}
		c.codes = sec((seconds*lm.CodesPerSecond + 1) * 0.050)
	}
	L := seconds * 12.5
	c.dit = sec(8 * (2.65e-4*L + 7.16e-9*L*L + 0.035))
	c.vae = sec(0.0104*seconds + 0.1)
	c.write = sec(0.004 * seconds)
	return c
}

// GenerateMusic runs a planned request and writes the audio to dst.
func (b *Music) GenerateMusic(ctx context.Context, req *api.MusicRequest, mp *api.MusicPlan, dst string, progress func(api.MusicProgress)) error {
	r := musicRequest(req)
	secs := cmpOr(mp.Seconds, 120)
	c := musicEstimate(r, req.Thinking, secs)
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
		case "think":
			return part(c.think, done, 150) / t
		case "codes":
			return (c.think.Seconds() + part(c.codes, done, total)) / t
		case "dit":
			return (c.think+c.codes).Seconds()/t + part(c.dit, done, total)/t
		case "vae":
			return (c.think + c.codes + c.dit).Seconds() / t
		}
		return 0
	}
	var res *pipeline.Result
	err := b.opt.Device.Do(func(*vk.Device) error {
		var err error
		res, err = b.pipe.Generate(r, pipeline.Options{
			Seed: uint64(mp.Seed), Think: req.Thinking, LMSeed: uint64(mp.LMSeed), Sampling: sampling,
			Shift: req.Shift, Timesteps: req.Timesteps,
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
			Planned: func(d *plan.Request, seconds float64) {
				// The plan fixes the length, so the estimate is redone
				// around it; what is spent is kept.
				spent := c.think
				c = musicEstimate(d, req.Thinking, seconds)
				c.think = spent
				progress(api.MusicProgress{Stage: "planned", Fraction: frac("codes", 0, 1), Seconds: seconds,
					Estimate: c.total(), Plan: &api.MusicMetadata{Caption: d.Caption, BPM: d.BPM,
						KeyScale: d.KeyScale, TimeSignature: d.TimeSignature, Duration: seconds,
						Language: cmpStr(d.Language, "unknown")}})
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
	progress(api.MusicProgress{Stage: "write", Fraction: 1 - c.write.Seconds()/c.total().Seconds()})
	return pipeline.Write(dst, res.Audio)
}

func cmpStr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
