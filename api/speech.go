package api

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// OpenAI Speech Request
// https://platform.openai.com/docs/api-reference/audio/createSpeech

// SpeechRequest is the OpenAI Speech Request
type SpeechRequest struct {
	host  string
	Input string `json:"input"`
	Model string `json:"model"`
	// Voice is a voice pack name, or a mixture of them: "af_bella,af_sky"
	// is the equal mean of two packs, which is the spelling and the meaning
	// hexgrad's own pipeline gives it, and "af_bella:3,af_sky:1" weights the
	// mix, which is an extension it has no spelling for. Weights are
	// normalised by their sum. GET /v1/models lists the names.
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format"` // mp3, wav, opus, flac, pcm
	Speed          float64 `json:"speed"`           // 0.5 - 2.0
	Stream         bool    `json:"stream"`          // false
	// Instructions describe the voice and its delivery ("An enthusiastic
	// radio DJ"): OpenAI's field, for gpt-4o-mini-tts and not tts-1. Here it
	// is answered by the instructed model (InstructedSpeech), and a request
	// that carries it for kokoro is refused rather than read as nothing.
	Instructions string `json:"instructions,omitempty"`
	// StreamFormat is OpenAI's: "audio" streams the body as the samples are
	// made (chunked), "sse" as speech.audio.delta events of base64 audio
	// and a closing speech.audio.done. Empty is one whole body.
	StreamFormat string `json:"stream_format,omitempty"`
	// CustomVoice is set when `voice` was OpenAI's custom-voice object,
	// {"id": "voice_..."}; Voice then holds the id.
	CustomVoice bool `json:"-"`
	// Seed is an extension, not OpenAI's: the instructed model samples, and
	// without a reference its speaker is a draw, so a seed is how a caller
	// gets the same reading twice. Kokoro is deterministic and ignores it.
	Seed *uint64 `json:"seed,omitempty"`
	// Phonemes is an extension, not OpenAI's: IPA to speak directly,
	// bypassing grapheme-to-phoneme. Kokoro's vocabulary is 178 IPA symbols
	// and the front end that reaches them is a separate model's worth of
	// rules (SPEECH.md T5), so a caller that already has phonemes -- a test
	// against the reference dump, a caller with its own lexicon -- says so
	// here and the server does not guess. It wins over Input when both are
	// set.
	Phonemes string `json:"phonemes,omitempty"`
}

// UnmarshalJSON takes `voice` as OpenAI does: a name, or a custom voice as
// {"id": "voice_..."}.
func (r *SpeechRequest) UnmarshalJSON(b []byte) error {
	type plain SpeechRequest
	aux := struct {
		*plain
		Voice json.RawMessage `json:"voice"`
	}{plain: (*plain)(r)}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	if len(aux.Voice) == 0 || string(aux.Voice) == "null" {
		return nil
	}
	if aux.Voice[0] == '"' {
		return json.Unmarshal(aux.Voice, &r.Voice)
	}
	var obj struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(aux.Voice, &obj); err != nil || obj.ID == "" {
		return errors.New(`"voice" is a name or {"id": "voice_..."}`)
	}
	r.Voice, r.CustomVoice = obj.ID, true
	return nil
}

// StreamingSpeechBackend is a speech model that makes its audio as it goes,
// so the endpoint can send it as it goes: the instructed model's frames are
// 80 ms of audio each, decoded a couple at a time.
type StreamingSpeechBackend interface {
	SpeechBackend
	// SpeakStream synthesises req, handing emit mono samples at the
	// returned rate, chunk by chunk. rate is known before the first chunk.
	SpeakStream(ctx context.Context, req *SpeechRequest, start func(rate int) error, emit func([]float32) error) error
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

// speechFor picks the model a request is for. The instructed model answers
// to its own ids, to OpenAI's instruction-taking gpt-4o-mini-tts, to a
// custom voice, and to any request that carries instructions and names no
// other model; kokoro to everything else, as before. An instruction for a
// model that cannot take one is refused, as OpenAI refuses it for tts-1.
func (s *Server) speechFor(req *SpeechRequest) (SpeechBackend, error) {
	instructed := s.InstructedSpeech != nil && (owns(s.InstructedSpeech, req.Model) ||
		strings.HasPrefix(req.Model, "gpt-4o-mini-tts"))
	kokoro := s.Speech != nil && req.Model != "" && owns(s.Speech, req.Model)
	switch {
	case instructed:
		return s.InstructedSpeech, nil
	case kokoro && (req.Instructions != "" || req.CustomVoice):
		return nil, errors.New("model " + strconv.Quote(req.Model) + " takes neither instructions nor a custom voice; " +
			"ask for the instructed model (GET /v1/models) or drop them")
	case !kokoro && (req.Instructions != "" || req.CustomVoice):
		if s.InstructedSpeech == nil {
			return nil, errors.New("instructions and custom voices need the instructed speech model, " +
				"which this server was not started with (-tts-instruct)")
		}
		return s.InstructedSpeech, nil
	case s.Speech != nil:
		return s.Speech, nil
	case s.InstructedSpeech != nil:
		return s.InstructedSpeech, nil
	}
	return nil, nil
}

// handleSpeech synthesises an utterance: one body, or streamed as it is made
// when stream_format asks and the model can.
func (s *Server) handleSpeech(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Speech == nil && s.InstructedSpeech == nil {
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
	if v := s.voiceNamed(req.Voice); v != nil && !req.CustomVoice {
		req.Voice, req.CustomVoice = v.ID, true
	}
	backend, err := s.speechFor(&req)
	if err != nil {
		badRequest(ctx, w, err.Error())
		return
	}
	stream, streaming := backend.(StreamingSpeechBackend)
	switch req.StreamFormat {
	case "", "audio", "sse":
	default:
		badRequest(ctx, w, "stream_format "+strconv.Quote(req.StreamFormat)+" is not one of audio, sse")
		return
	}
	if req.Stream && req.StreamFormat == "" {
		req.StreamFormat = "audio"
	}
	if req.StreamFormat != "" && !streaming {
		badRequest(ctx, w, "streaming speech is not implemented for this model; the utterance's durations are decided before any samples exist")
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
	case "mp3", "opus", "aac", "flac":
		// OpenAI's speech endpoint defaults to mp3, so a client written
		// against it asks for a compressed format without anyone
		// choosing to -- which makes this the likeliest 400 on this
		// endpoint, and worth answering with the fix rather than just
		// the refusal.
		badRequest(ctx, w, "response_format "+strconv.Quote(format)+
			" is not supported; this server has no audio encoder and writes wav and pcm "+
			"(16-bit signed, little endian, mono). "+
			strconv.Quote(format)+" is OpenAI's default rather than something you asked for, "+
			"so send \"response_format\": \"wav\", or omit the field, which is the same thing")
		return
	default:
		badRequest(ctx, w, "response_format "+strconv.Quote(format)+
			" is not supported; this server encodes wav and pcm (16-bit signed, little endian, mono)")
		return
	}
	if req.StreamFormat != "" {
		s.streamSpeech(ctx, w, stream, &req, format)
		return
	}

	// The synthesis is timed on its own, not just as part of the request:
	// what it costs per second of audio produced is the number that says
	// whether this endpoint got faster, and the request's total duration
	// includes encoding and the write to the client.
	start := time.Now()
	clip, err := backend.Speak(r.Context(), &req)
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

// streamSpeech sends the audio as the model makes it. "audio" is the
// response_format's bytes over a chunked body -- a WAV header first, its
// sizes the streaming placeholder 0xffffffff, since the length is not known
// until the end -- and "sse" the same bytes base64'd in speech.audio.delta
// events, then speech.audio.done. Once the first byte is out a failure can
// only end the stream: the status is already 200.
func (s *Server) streamSpeech(ctx context.Context, w http.ResponseWriter, b StreamingSpeechBackend, req *SpeechRequest, format string) {
	start := time.Now()
	var ev *sse
	rc := http.NewResponseController(w)
	samples := 0
	rate := 0
	write := func(p []byte) error {
		if ev != nil {
			return ev.send("", speechDelta{Type: "speech.audio.delta", Audio: base64.StdEncoding.EncodeToString(p)})
		}
		if _, err := w.Write(p); err != nil {
			return err
		}
		return rc.Flush()
	}
	begin := func(r int) error {
		rate = r
		if req.StreamFormat == "sse" {
			ev = newSSE(w)
		} else {
			w.Header().Set("Content-Type", map[string]string{"wav": "audio/wav", "pcm": "audio/pcm"}[format])
			w.Header().Set("X-Sample-Rate", strconv.Itoa(r))
			w.WriteHeader(http.StatusOK)
		}
		if format == "wav" {
			return write(streamingWAVHeader(r))
		}
		return nil
	}
	emit := func(chunk []float32) error {
		samples += len(chunk)
		return write(encodePCM16(chunk))
	}
	err := b.SpeakStream(ctx, req, begin, emit)
	if err != nil {
		if rate == 0 {
			backendError(ctx, w, "speech", err)
			return
		}
		logf(ctx, "speech: stream ended early: %v", err)
		return
	}
	if ev != nil {
		_ = ev.send("", speechDone{Type: "speech.audio.done"})
	}
	if rate > 0 {
		audioLen := float64(samples) / float64(rate)
		took := time.Since(start)
		logf(ctx, "speech: %d characters -> %.2fs of audio at %d Hz, streamed (%s, %s), in %v (%.1fx real time)",
			len(req.Input), audioLen, rate, req.StreamFormat, format, took.Round(time.Millisecond), audioLen/took.Seconds())
	}
}

// speechDelta and speechDone are OpenAI's streamed speech events.
type speechDelta struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

type speechDone struct {
	Type string `json:"type"`
}

// streamingWAVHeader is a 16-bit mono WAV header whose RIFF and data sizes
// are 0xffffffff: the length is not known when it is sent, and that is the
// value players and ffmpeg read as "until the end of the stream".
func streamingWAVHeader(rate int) []byte {
	b := make([]byte, 0, 44)
	put32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	put16 := func(v uint16) { b = binary.LittleEndian.AppendUint16(b, v) }
	b = append(b, "RIFF"...)
	put32(0xffffffff)
	b = append(b, "WAVEfmt "...)
	put32(16)
	put16(1)
	put16(1)
	put32(uint32(rate))
	put32(uint32(rate * 2))
	put16(2)
	put16(16)
	b = append(b, "data"...)
	put32(0xffffffff)
	return b
}
