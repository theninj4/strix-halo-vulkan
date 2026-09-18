package api

import (
	"encoding/binary"
	"errors"
	"math"
	"net/http"
	"strconv"
	"time"
)

// OpenAI Speech Request
// https://platform.openai.com/docs/api-reference/audio/createSpeech

// SpeechRequest is the OpenAI Speech Request
type SpeechRequest struct {
	host           string
	Input          string  `json:"input"`
	Model          string  `json:"model"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format"` // mp3, wav, opus, flac, pcm
	Speed          float64 `json:"speed"`           // 0.5 - 2.0
	Stream         bool    `json:"stream"`          // false
	// Phonemes is an extension, not OpenAI's: IPA to speak directly,
	// bypassing grapheme-to-phoneme. Kokoro's vocabulary is 178 IPA symbols
	// and the front end that reaches them is a separate model's worth of
	// rules (SPEECH.md T5), so a caller that already has phonemes -- a test
	// against the reference dump, a caller with its own lexicon -- says so
	// here and the server does not guess. It wins over Input when both are
	// set.
	Phonemes string `json:"phonemes,omitempty"`
}

// SpeechResponse is the OpenAI Speech Response
type SpeechResponse []byte

// speedRange is what the endpoint accepts. OpenAI's own range is 0.25 to 4;
// kokoro reads it as a duration divisor, so the extremes are intelligible
// rather than merely legal.
const (
	minSpeed = 0.25
	maxSpeed = 4.0
)

// handleSpeech synthesises an utterance and returns it as one body.
//
// It does not stream. OpenAI's streaming speech is chunked audio as the model
// produces it, and kokoro does not produce it that way: the durations for the
// whole utterance are decided before a single sample exists (SPEECH.md T2),
// so there is nothing to send early. A request with stream:true is refused
// rather than answered with a whole body pretending to be a stream.
func (s *Server) handleSpeech(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Speech == nil {
		notLoaded(ctx, w, "text to speech", "-tts")
		return
	}
	var req SpeechRequest
	if !decodeJSON(ctx, w, r, &req) {
		return
	}
	if req.Input == "" && req.Phonemes == "" {
		badRequest(ctx, w, "input is empty")
		return
	}
	if req.Stream {
		badRequest(ctx, w, "streaming speech is not implemented; the utterance's durations are decided before any samples exist")
		return
	}
	if req.Speed == 0 {
		req.Speed = 1
	}
	if req.Speed < minSpeed || req.Speed > maxSpeed {
		badRequest(ctx, w, "speed is "+strconv.FormatFloat(req.Speed, 'g', -1, 64)+
			", outside ["+strconv.FormatFloat(minSpeed, 'g', -1, 64)+", "+
			strconv.FormatFloat(maxSpeed, 'g', -1, 64)+"]")
		return
	}
	format := req.ResponseFormat
	if format == "" {
		format = "wav"
	}
	switch format {
	case "wav", "pcm":
	default:
		badRequest(ctx, w, "response_format "+strconv.Quote(format)+
			" is not supported; this server encodes wav and pcm (16-bit signed, little endian, mono)")
		return
	}

	// The synthesis is timed on its own, not just as part of the request:
	// what it costs per second of audio produced is the number that says
	// whether this endpoint got faster, and the request's total duration
	// includes encoding and the write to the client.
	start := time.Now()
	clip, err := s.Speech.Speak(r.Context(), &req)
	if err != nil {
		backendError(ctx, w, "speech", err)
		return
	}
	if clip != nil {
		took := time.Since(start)
		audioLen := clip.Duration()
		// Faster than real time is the point, so report the ratio that
		// way round: 20x means a second of speech took 50 ms to make.
		speed := 0.0
		if took > 0 {
			speed = audioLen / took.Seconds()
		}
		logf(ctx, "speech: %d characters -> %.2fs of audio at %d Hz, voice %q, in %v (%.1fx real time)",
			len(req.Input)+len(req.Phonemes), audioLen, clip.Rate, req.Voice,
			took.Round(time.Millisecond), speed)
	}
	if clip == nil || len(clip.Samples) == 0 {
		serverError(ctx, w, "speech", errEmptyAudio)
		return
	}

	var body []byte
	switch format {
	case "wav":
		if body, err = clip.EncodeWAV(); err != nil {
			serverError(ctx, w, "speech", err)
			return
		}
		w.Header().Set("Content-Type", "audio/wav")
	case "pcm":
		body = encodePCM16(clip.Samples)
		w.Header().Set("Content-Type", "audio/pcm")
	}
	// The rate is in the WAV header but not in raw PCM, and a client that
	// asked for PCM still has to know what to play it at.
	w.Header().Set("X-Sample-Rate", strconv.Itoa(clip.Rate))
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// encodePCM16 is audio.Clip.EncodeWAV's sample conversion without the
// container, scaled by 32767 and clamped for the same reason: a full-scale
// +1.0 that 32768 would wrap to the most negative sample stays at the top of
// the range instead of inverting.
func encodePCM16(samples []float32) []byte {
	out := make([]byte, 0, 2*len(samples))
	for _, s := range samples {
		v := math.Round(float64(s) * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(v)))
	}
	return out
}

// errEmptyAudio is its own value so the handler's two failure paths -- a
// backend error, and a backend that returned nothing without saying why --
// are distinguishable in the log.
var errEmptyAudio = errors.New("the speech backend returned no samples")
