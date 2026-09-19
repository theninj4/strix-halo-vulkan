package backend

import (
	"context"
	"errors"
	"fmt"
	"log"
	"slices"
	"sort"
	"strconv"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/g2p"
	"strix-halo-vulkan/kokoro"
	"strix-halo-vulkan/vk"
)

// TTSOptions is what cmd/serve's flags come to.
type TTSOptions struct {
	// Model is the converted Kokoro-82M checkpoint directory.
	Model string
	// Lexicon is the misaki lexicon directory (reference/convert_misaki.py).
	// Empty loads no front end, and the endpoint then takes phonemes only.
	Lexicon string
	// Espeak enables the espeak-ng fallback for words outside the lexicon.
	// Without it those words are dropped, which is a wrong utterance rather
	// than a slow one, so the default in cmd/serve is on and a failure to
	// open it is a warning and not a fatal error (SPEECH.md T5).
	Espeak bool
	// British switches both the lexicon and the fallback to en-GB.
	British bool
	// Voice is what a request that names none gets.
	Voice string
	// Device, when set, runs the model on the GPU. See Speak for what that
	// costs per request.
	Device *Device
	// Noise seeds the vocoder's excitation. Zero leaves it off, which is
	// what the reference dump was taken with and the only setting under
	// which two runs agree bit for bit.
	Noise int64
	// MaxFrames is the longest utterance the device arenas are staged for,
	// in 25 ms alignment frames. Zero takes defaultMaxFrames. It costs
	// memory and nothing else -- a request's cost follows its own length,
	// not this -- and an utterance past it restages rather than failing.
	MaxFrames int
	// ID is the model id this backend answers to in /v1/models. Empty takes
	// defaultTTSModelID.
	ID string
}

const defaultTTSModelID = "kokoro-82m"

// defaultMaxFrames is the ceiling the device attachment is staged for: 1000
// alignment frames, 25 seconds of speech.
//
// The number is a memory budget and not a latency one. Staging is ~110 ms
// whatever it is, a request costs what its own utterance costs, and the
// arenas are about 150 MB plus 0.7 MB a frame -- so 25 s is 850 MB on a part
// with 128 GB, and covers an utterance far longer than anything a
// /v1/audio/speech caller sends in one request.
const defaultMaxFrames = 1000

// TTS is the kokoro adapter: an api.SpeechBackend over the model cmd/tts
// drives.
//
// One mutex guards the whole thing. The model is not reentrant -- the
// vocoder's arenas, the GPU attachment and espeak's process are all shared
// state -- so utterances are serialised, and on the GPU path they would be
// anyway (see Device).
type TTS struct {
	opt    TTSOptions
	id     string
	model  *kokoro.Model
	lex    *g2p.Lexicon
	espeak *g2p.Espeak
	voices []string

	mu sync.Mutex
	// staged is the frame count the device attachment is sized for, or 0
	// when nothing is attached. It is a ceiling: see synthesize.
	staged int
	// dec and pred are the style vectors the attachment is conditioned on,
	// so a run of requests in one voice at one length re-conditions once.
	dec, pred []float32
}

// NewTTS loads the checkpoint and, if asked for, the grapheme-to-phoneme
// front end, and stages the model onto the device when one was configured.
//
// Staging happens here rather than on the first request because it is the
// only thing about a request that is not the utterance: ~110 ms of pipelines
// and weights, sized for MaxFrames and reused by every clip after it
// (SPEECH.md T9). A device that cannot hold it should say so at startup and
// not on somebody's first sentence.
func NewTTS(opt TTSOptions) (*TTS, error) {
	if opt.ID == "" {
		opt.ID = defaultTTSModelID
	}
	if opt.Voice == "" {
		opt.Voice = "af_heart"
	}
	model, err := kokoro.Load(opt.Model)
	if err != nil {
		return nil, fmt.Errorf("backend: loading %s: %w "+
			"(the checkpoint ships as a pickle; run reference/convert_kokoro.py first)", opt.Model, err)
	}
	if opt.Noise != 0 {
		model.SetExcitationNoise(opt.Noise)
	}
	t := &TTS{opt: opt, id: opt.ID, model: model}
	for name := range model.Voices {
		t.voices = append(t.voices, name)
	}
	sort.Strings(t.voices)
	// A blend is a voice here too, so the default is checked the way a
	// request's is: the spelling first, then every name in it.
	if err := model.CheckVoice(opt.Voice); err != nil {
		return nil, fmt.Errorf("backend: %w (in %s)", err, opt.Model)
	}

	if opt.Lexicon != "" {
		if t.lex, err = g2p.Load(opt.Lexicon, opt.British); err != nil {
			return nil, fmt.Errorf("backend: loading the lexicon from %s: %w "+
				"(run reference/convert_misaki.py to write it)", opt.Lexicon, err)
		}
		if opt.Espeak {
			if e, err := g2p.Open(opt.British); err == nil {
				t.espeak = e
				t.lex.Fallback = e
			} else {
				log.Printf("backend: no espeak fallback (%v); "+
					"words outside the dictionary will be dropped", err)
			}
		}
	}

	if opt.Device != nil {
		// Any style will do to stage with -- every request re-conditions
		// anyway, because the style row is indexed by the phoneme count as
		// well as by the voice -- but the default voice's first row is the
		// one most likely to still be current when the first request lands.
		dec, pred, err := model.Style(opt.Voice, 1)
		if err != nil {
			return nil, fmt.Errorf("backend: %w", err)
		}
		if err := opt.Device.Do(func(dev *vk.Device) error {
			return t.attach(dev, t.maxFrames(), dec, pred)
		}); err != nil {
			return nil, fmt.Errorf("backend: staging kokoro on the device: %w", err)
		}
	}
	return t, nil
}

// Models reports the one model this backend serves.
func (t *TTS) Models() []api.Model {
	return []api.Model{{ID: t.id, Object: "model", OwnedBy: "local"}}
}

// Voices are the voice pack names in the checkpoint, sorted.
//
// A request may also name a mixture of them, which is every comma-joined
// subset and not a list anything could enumerate, so what this reports is the
// alphabet a blend is spelled in rather than everything Speak accepts.
func (t *TTS) Voices() []string {
	out := make([]string, len(t.voices))
	copy(out, t.voices)
	return out
}

// Close releases the espeak process and any device attachment. The
// checkpoint itself is a mapping the runtime reclaims.
func (t *TTS) Close() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.opt.Device != nil {
		_ = t.opt.Device.Do(func(*vk.Device) error { t.detach(); return nil })
	}
	if t.espeak != nil {
		t.espeak.Close()
	}
}

// Speak synthesises one utterance.
//
// The context is checked on the way in and not again: the model has no
// cancellation point, so a client that hangs up mid-utterance still costs the
// synthesis.
//
// The request's phonemes win over its text: the checkpoint's vocabulary is
// 178 IPA symbols and reaching them from English is a separate model's worth
// of rules, so a caller that has already done that says so and this does not
// guess (SPEECH.md T5).
func (t *TTS) Speak(ctx context.Context, req *api.SpeechRequest) (*audio.Clip, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	voice := req.Voice
	if voice == "" {
		voice = t.opt.Voice
	}
	// The voice may name a mixture -- "af_bella,af_sky" is upstream's
	// spelling for the equal mean of two packs, and ":weight" is this
	// server's for an unequal one -- so the error has to name the component
	// that was not found rather than the whole string.
	if err := t.model.CheckVoice(voice); err != nil {
		var unknown *kokoro.UnknownVoiceError
		if errors.As(err, &unknown) {
			where := ""
			if unknown.Blend != "" {
				where = " in the blend " + strconv.Quote(unknown.Blend)
			}
			return nil, fmt.Errorf("no voice %q%s; this checkpoint has %d, and GET /v1/models lists them: %w",
				unknown.Name, where, len(t.voices), api.ErrUnsupported)
		}
		return nil, fmt.Errorf("%w: %w", err, api.ErrUnsupported)
	}

	phonemes := req.Phonemes
	if phonemes == "" {
		if t.lex == nil {
			return nil, fmt.Errorf(
				"this server has no grapheme-to-phoneme front end loaded; send IPA in \"phonemes\": %w",
				api.ErrUnsupported)
		}
		var unknown int
		phonemes, unknown = t.lex.Phonemize(req.Input)
		if unknown > 0 {
			// Not an error: misaki drops what it cannot pronounce too, and
			// the utterance is still the one the caller asked for minus
			// those tokens. It is logged because a rise in this number is
			// how a missing espeak shows up.
			log.Printf("backend: tts: %d token(s) in %q could not be pronounced and were dropped",
				unknown, req.Input)
		}
	}

	ids, dropped := t.model.Config.Phonemes(phonemes)
	if len(ids) <= 2 {
		return nil, fmt.Errorf("nothing to speak: %q is %d phonemes in this vocabulary: %w",
			phonemes, len(ids), api.ErrUnsupported)
	}
	// The style row is indexed by the phoneme count the caller supplied,
	// minus whatever fell outside the vocabulary, which is how KPipeline
	// indexes it and is not the token count.
	decStyle, predStyle, err := t.model.Style(voice, len([]rune(phonemes))-dropped)
	if err != nil {
		return nil, err
	}
	speed := float32(req.Speed)
	if speed == 0 {
		speed = 1
	}

	samples, err := t.synthesize(ids, decStyle, predStyle, speed)
	if err != nil {
		return nil, err
	}
	return &audio.Clip{Rate: t.model.Config.SamplingRate, Samples: samples}, nil
}

// synthesize runs the model, on the device when one was configured.
//
// **The attachment is staged once and outlives the request.** Kokoro's arenas
// are sized for a frame count, and until T9 that count was the utterance's --
// which the durations decide, so the phoneme side had to run on the *host*
// first just to learn how big to build the device. A request paid a 400 ms
// host prosody pass and an 85 ms staging pass to save 25 ms of device time,
// and the endpoint was 550 ms for an utterance the device does in 48.
//
// Now the count is a ceiling (kokoro.Model.AttachGPU): the arenas are built
// for MaxFrames at startup and every shorter utterance is a prefix of them,
// so a request is the model and nothing else. What is left here is the two
// things that do change per request -- the voice, which is a host-side
// projection into arenas that already exist, and an utterance longer than the
// ceiling, which restages once and keeps the larger arenas.
func (t *TTS) synthesize(ids []int, decStyle, predStyle []float32, speed float32) ([]float32, error) {
	if t.opt.Device == nil {
		p, err := t.model.Prosody(ids, predStyle, speed)
		if err != nil {
			return nil, err
		}
		samples, _, err := t.model.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle)
		return samples, err
	}

	var samples []float32
	err := t.opt.Device.Do(func(dev *vk.Device) error {
		if err := t.attach(dev, t.maxFrames(), decStyle, predStyle); err != nil {
			return err
		}
		p, err := t.model.Prosody(ids, predStyle, speed)
		if err != nil {
			// An utterance past the ceiling is the one case that restages,
			// and the durations have already said how long it is -- so the
			// new arenas are built for it with room to spare, and the next
			// long request finds them already there.
			var over *kokoro.FramesOverflowError
			if !errors.As(err, &over) {
				return err
			}
			want := roundUpFrames(over.Frames)
			log.Printf("backend: tts: %d frames against arenas for %d; restaging for %d",
				over.Frames, over.Staged, want)
			t.detach()
			if err := t.attach(dev, want, decStyle, predStyle); err != nil {
				return err
			}
			if p, err = t.model.Prosody(ids, predStyle, speed); err != nil {
				return err
			}
		}
		samples, _, err = t.model.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle)
		return err
	})
	return samples, err
}

// maxFrames is the ceiling to stage for, which never shrinks: a server that
// has seen one long utterance keeps the arenas for the next.
func (t *TTS) maxFrames() int {
	n := t.opt.MaxFrames
	if n <= 0 {
		n = defaultMaxFrames
	}
	return max(n, t.staged)
}

// frameBucket is what a restaging rounds the offending utterance up to, so a
// paragraph a few frames longer than the last one does not stage again.
const frameBucket = 256

func roundUpFrames(n int) int { return (n + frameBucket) / frameBucket * frameBucket }

// attach stages the model for `frames` if nothing is staged, and conditions
// whatever is staged on this utterance's voice.
//
// It must be called with the device held. Both halves are cheap when there is
// nothing to do: an attachment that is already large enough is left alone,
// and a voice identical to the last one is not projected again -- which is
// the common case for a server answering a run of requests in one voice at
// one length.
func (t *TTS) attach(dev *vk.Device, frames int, decStyle, predStyle []float32) error {
	if t.staged == 0 {
		start := time.Now()
		if err := t.model.AttachGPU(dev, frames, decStyle, predStyle, kokoro.DefaultConvKernel); err != nil {
			return err
		}
		t.staged = frames
		t.dec, t.pred = decStyle, predStyle
		log.Printf("backend: tts: staged for %d frames (%.0f s of speech) in %v",
			frames, float64(frames*t.model.Config.SamplesPerFrame())/float64(t.model.Config.SamplingRate),
			time.Since(start).Round(time.Millisecond))
		return nil
	}
	if slices.Equal(t.dec, decStyle) && slices.Equal(t.pred, predStyle) {
		return nil
	}
	// The style row is indexed by the phoneme count as well as the voice
	// (see Speak), so this changes with the length of the utterance and not
	// only with the name in the request.
	if err := t.model.SetVoice(decStyle, predStyle); err != nil {
		return err
	}
	t.dec, t.pred = decStyle, predStyle
	return nil
}

// detach tears down the device attachment. It must be called with the device
// held: destroying pipelines and buffers is device work like any other.
func (t *TTS) detach() {
	if t.staged != 0 {
		t.model.DetachGPU()
		t.staged, t.dec, t.pred = 0, nil, nil
	}
}
