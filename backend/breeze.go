package backend

// The instructed speech adapter (research/tts-instruct-vertical.md B6):
// Breeze-TTS-2 behind api.StreamingSpeechBackend. It answers
// /v1/audio/speech requests that carry OpenAI's `instructions` (a description
// of the voice), name it or gpt-4o-mini-tts, or name a custom voice, and it
// streams: a frame is 80 ms of audio, decoded two at a time.
//
// **The device lock is taken a submit at a time**, not for the utterance:
// one encoder segment, one prefill pass, a quarter of a frame (<= ~11 ms),
// one codec chunk. Holding it across an utterance would put Rune's
// classifications (latency first, rune-always-resident) behind a whole
// paragraph. The FIFO lock hands the device to every waiter between two
// submits instead.
//
// **Two utterances run at once** and share every frame's passes (breeze's
// scheduler, B7); a third waits for a slot.

import (
	"context"
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"log"
	"strings"
	"unicode"

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

	// Hear is the speech-to-text model that finds where an utterance's
	// words end, so the breath the model may close on can be cut (the
	// closing breath, breeze/tail.go); nil keeps every ending.
	Hear api.TranscriptionBackend

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
	if err == nil {
		b.m.Lock = func(f func() error) error { return opt.Device.Do(func(*vk.Device) error { return f() }) }
	}
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
	started := false
	o := breeze.GenOptions{Seed: seed}
	if b.Hear != nil {
		o.Ending = b.ending(ctx, req.Input)
	}
	_, err := b.m.Generate(ctx, r, o,
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

// ending is where the speech in wav (the utterance's last seconds) ends:
// the end of the last heard word that is the text's last word, when words
// follow it (the closing breath, heard as "and"); len(wav) otherwise.
func (b *Breeze) ending(ctx context.Context, text string) func([]float32) (int, error) {
	return func(wav []float32) (int, error) {
		c, err := audio.Resample(&audio.Clip{Samples: wav, Rate: breeze.SampleRate}, 16000)
		if err != nil {
			return 0, err
		}
		resp, err := b.Hear.Transcribe(ctx, c, &api.TranscriptionRequest{Language: "en"})
		if err != nil {
			if ctx.Err() != nil {
				return 0, err
			}
			// The audio has streamed: keep its ending rather than fail it.
			log.Printf("breeze: the ending check: %v", err)
			return len(wav), nil
		}
		if sec, ok := speechEnd(resp.Words, text); ok {
			return min(len(wav), int(sec*breeze.SampleRate)), nil
		}
		return len(wav), nil
	}
}

// speechEnd is the end of the last heard word that is text's last word,
// when other words follow it.
func speechEnd(heard []api.TranscriptionWord, text string) (float64, bool) {
	ws := strings.FieldsFunc(text, notWord)
	if len(ws) == 0 {
		return 0, false
	}
	last := strings.ToLower(ws[len(ws)-1])
	for i := len(heard) - 2; i >= 0; i-- {
		if w := strings.FieldsFunc(heard[i].Word, notWord); len(w) == 1 && strings.ToLower(w[0]) == last {
			return heard[i].End, true
		}
	}
	return 0, false
}

func notWord(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\'' }
