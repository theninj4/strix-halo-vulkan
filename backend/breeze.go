package backend

// The instructed speech adapter (research/tts-instruct-vertical.md B6):
// Breeze-TTS-2 behind api.StreamingSpeechBackend. It answers
// /v1/audio/speech requests that carry OpenAI's `instructions` (a description
// of the voice), name it or gpt-4o-mini-tts, or name a custom voice, and it
// streams: a frame is 80 ms of audio, decoded two at a time.
//
// **The device lock is taken a pass at a time**, not for the utterance:
// one encoder segment, one backbone step, one depth pass, one codec chunk.
// An utterance is ~17 device passes a frame for seconds on end, and holding
// the lock across it would put Rune's classifications (latency first,
// rune-always-resident) behind a whole paragraph. The FIFO lock hands the
// device to every waiter between two passes instead.

import (
	"context"
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"strings"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/breeze"
	"strix-halo-vulkan/vk"
)

// BreezeOptions is what cmd/serve's flags come to.
type BreezeOptions struct {
	// Model is the Breeze-TTS-2 checkpoint directory.
	Model string
	// Device runs it; it has no CPU path.
	Device *Device
	// ID is the model id in /v1/models; empty takes "breeze-tts-2".
	ID string
	// VoiceDir holds custom voices and their consents (B8); empty serves
	// none.
	VoiceDir string
}

// Breeze is the instructed speech backend.
type Breeze struct {
	opt BreezeOptions
	m   *breeze.Model
	// mu serialises utterances: the model holds one stream's state (two
	// KV slots, the codec's carried context).
	mu sync.Mutex

	// The custom voices (breeze_voices.go).
	store *voiceStore
}

// NewBreeze stages the model.
func NewBreeze(opt BreezeOptions) (*Breeze, error) {
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: breeze runs on the device only")
	}
	if opt.ID == "" {
		opt.ID = "breeze-tts-2"
	}
	b := &Breeze{opt: opt}
	err := opt.Device.Do(func(dev *vk.Device) error {
		m, err := breeze.LoadModel(dev, opt.Model, breeze.DefaultModelOptions())
		b.m = m
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("backend: staging breeze: %w", err)
	}
	if opt.VoiceDir != "" {
		if b.store, err = openVoiceStore(opt.VoiceDir, opt.Model); err != nil {
			return nil, err
		}
	}
	return b, nil
}

// Close releases the model.
func (b *Breeze) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	_ = b.opt.Device.Do(func(*vk.Device) error { b.m.Destroy(); return nil })
}

// Models is the one id this backend answers to.
func (b *Breeze) Models() []api.Model {
	return []api.Model{{ID: b.opt.ID, Object: "model", OwnedBy: "local"}}
}

// Voices are the custom voices' ids: the model has no named voices of its
// own. Without instructions or a custom voice it draws a speaker each
// request; a built-in name (alloy, a kokoro pack) is accepted and does not
// change that.
func (b *Breeze) Voices() []string {
	if b.store == nil {
		return nil
	}
	var out []string
	for _, v := range b.store.list() {
		out = append(out, v.ID)
	}
	return out
}

// Speak is SpeakStream into one clip.
func (b *Breeze) Speak(ctx context.Context, req *api.SpeechRequest) (*audio.Clip, error) {
	clip := &audio.Clip{}
	err := b.SpeakStream(ctx, req, func(rate int) error { clip.Rate = rate; return nil },
		func(s []float32) error { clip.Samples = append(clip.Samples, s...); return nil })
	if err != nil {
		return nil, err
	}
	return clip, nil
}

// SpeakStream synthesises req, chunk by chunk.
func (b *Breeze) SpeakStream(ctx context.Context, req *api.SpeechRequest, start func(int) error, emit func([]float32) error) error {
	if req.Phonemes != "" {
		return fmt.Errorf("%s reads text, not phonemes: %w", b.opt.ID, api.ErrUnsupported)
	}
	if req.Speed != 0 && req.Speed != 1 {
		return fmt.Errorf("%s has no speed control; describe the pace in instructions (\"speaking slowly\"): %w",
			b.opt.ID, api.ErrUnsupported)
	}
	r := &breeze.Request{Text: req.Input, Instruction: req.Instructions}
	if req.CustomVoice || strings.HasPrefix(req.Voice, "voice_") {
		var v *storedVoice
		if b.store != nil {
			v = b.store.get(req.Voice)
		}
		if v == nil {
			return fmt.Errorf("no custom voice %q (GET /v1/audio/voices): %w", req.Voice, api.ErrUnsupported)
		}
		r.RefText, r.RefFrames = v.Transcript, v.Codes
	}
	var seed uint64
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		var s [8]byte
		_, _ = crand.Read(s[:])
		seed = binary.LittleEndian.Uint64(s[:])
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	started := false
	_, err := b.m.Generate(ctx, r,
		breeze.GenOptions{
			Seed: seed,
			Lock: func(f func() error) error { return b.opt.Device.Do(func(*vk.Device) error { return f() }) },
		},
		func(s []float32) error {
			if !started {
				started = true
				if err := start(breeze.SampleRate); err != nil {
					return err
				}
			}
			return emit(s)
		})
	return err
}
