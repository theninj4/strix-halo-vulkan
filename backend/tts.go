package backend

import (
	"context"
	"fmt"
	"log"
	"sort"
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
	// ID is the model id this backend answers to in /v1/models. Empty takes
	// defaultTTSModelID.
	ID string
}

const defaultTTSModelID = "kokoro-82m"

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
	// when nothing is attached. See Speak.
	staged int
}

// NewTTS loads the checkpoint and, if asked for, the grapheme-to-phoneme
// front end. It does not touch the device: the arenas are sized per
// utterance, so there is nothing to stage until there is something to say.
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
	if _, ok := model.Voices[opt.Voice]; !ok {
		return nil, fmt.Errorf("backend: no voice %q in %s", opt.Voice, opt.Model)
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
	return t, nil
}

// Models reports the one model this backend serves.
func (t *TTS) Models() []api.Model {
	return []api.Model{{ID: t.id, Object: "model", OwnedBy: "local"}}
}

// Voices are the voice pack names in the checkpoint, sorted.
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
	if _, ok := t.model.Voices[voice]; !ok {
		return nil, fmt.Errorf("no voice %q; this checkpoint has %d: %w",
			voice, len(t.voices), api.ErrUnsupported)
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
// **The device path stages per request, and that is a known cost rather than
// an oversight.** Kokoro's arenas are sized for one utterance's frame count
// -- the durations decide it, so it is not known before the phoneme side has
// run -- which is why AttachGPU takes `frames` and why cmd/tts computes the
// prosody twice. The sequence here is cmd/tts's, unchanged, because it is the
// one T4 and T6 measured: host prosody to settle the length, stage, then the
// whole utterance on the device.
//
// Two things would remove the staging from the request and both are
// measurements this session could not run:
//
//   - **Bucketed residency.** Size the arenas for a frame count above the
//     utterance's and zero-pad the alignment up to it, trimming the tail by
//     Prosody.Samples. The phoneme side already accepts any count up to the
//     one it was built for (GPUPhonemes.sharedGraph); whether the generator's
//     stages tolerate the padding is the open question.
//   - **Skipping the second prosody pass.** The vocoder reads `asr`, `f0` and
//     `energy` from the host (SPEECH.md T7), so the host prosody that settled
//     the length could feed the device vocoder directly and the phoneme side
//     would not need staging at all.
func (t *TTS) synthesize(ids []int, decStyle, predStyle []float32, speed float32) ([]float32, error) {
	if t.opt.Device == nil {
		p, err := t.model.Prosody(ids, predStyle, speed)
		if err != nil {
			return nil, err
		}
		samples, _, err := t.model.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle)
		return samples, err
	}

	// The whole utterance runs inside one Do: staging, the dispatches and the
	// teardown are all device work, and the queue is not externally
	// synchronised (see Device).
	var samples []float32
	err := t.opt.Device.Do(func(dev *vk.Device) error {
		// The frame count the arenas are sized for comes from the durations,
		// so the prosody has to run on the host before the device can be
		// staged -- and the previous request's attachment has to go first, or
		// this pass would run against arenas built for another utterance.
		t.detach()
		p, err := t.model.Prosody(ids, predStyle, speed)
		if err != nil {
			return err
		}
		start := time.Now()
		if err := t.model.AttachGPU(dev, p.Frames, decStyle, predStyle, kokoro.DefaultConvKernel); err != nil {
			return err
		}
		t.staged = p.Frames
		staging := time.Since(start)

		// Now the whole utterance on the device, the phoneme side included.
		if p, err = t.model.Prosody(ids, predStyle, speed); err != nil {
			return err
		}
		if samples, _, err = t.model.Vocoder.Apply(p.ASR, p.F0, p.Energy, decStyle); err != nil {
			return err
		}
		log.Printf("backend: tts: %d frames, staged in %v", p.Frames, staging.Round(time.Millisecond))
		return nil
	})
	return samples, err
}

// detach tears down the device attachment. It must be called with the device
// held: destroying pipelines and buffers is device work like any other.
func (t *TTS) detach() {
	if t.staged != 0 {
		t.model.DetachGPU()
		t.staged = 0
	}
}
