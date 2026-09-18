package api

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/audio"
)

// TranscriptionRequest represents the request structure for audio transcription.
type TranscriptionRequest struct {
	host             string
	Model            string  `json:"model"`
	ChunkingStrategy string  `json:"chunking_strategy"` // "auto"
	Language         string  `json:"language"`          // "en"
	Prompt           string  `json:"prompt"`
	ResponseFormat   string  `json:"response_format"` // "json"
	Data             []byte  `json:"file"`
	Temperature      float64 `json:"temperature"`
	Stream           bool    `json:"stream"`
	// Granularities is `timestamp_granularities`: "segment", "word", or
	// both. OpenAI's own clients send the field with a `[]` on the end --
	// it is a repeated multipart field -- and a JSON client writes it
	// without, so both spellings are read and Wants merges them.
	Granularities      StringList `json:"timestamp_granularities,omitempty"`
	GranularitiesArray StringList `json:"timestamp_granularities[],omitempty"`
}

// Wants reports whether the request asked for this timestamp granularity.
// The default is segments, which is what OpenAI's verbose_json returns when
// nothing is named.
func (r *TranscriptionRequest) Wants(kind string) bool {
	all := append(append(StringList{}, r.Granularities...), r.GranularitiesArray...)
	if len(all) == 0 {
		return kind == "segment"
	}
	for _, g := range all {
		if g == kind {
			return true
		}
	}
	return false
}

// TranscriptionResponse is the non-streaming response, and it is OpenAI's
// `verbose_json` object with the Whisper-specific fields left out rather than
// invented.
//
// `avg_logprob`, `compression_ratio`, `no_speech_prob`, `seek` and
// `temperature` are properties of Whisper's decoder, and a TDT transducer has
// none of them: it emits a token at a frame or it emits a blank. A number in
// those fields would be a number a client could filter on, so there is none.
type TranscriptionResponse struct {
	Task     string                 `json:"task,omitempty"`
	Language string                 `json:"language,omitempty"`
	Text     string                 `json:"text"`
	Duration float64                `json:"duration,omitempty"`
	Words    []TranscriptionWord    `json:"words,omitempty"`
	Segments []TranscriptionSegment `json:"segments,omitempty"`
}

// TranscriptionWord is one word and when it was said, in seconds.
type TranscriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// TranscriptionSegment is a sentence of them.
type TranscriptionSegment struct {
	Text  string  `json:"text"`
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// TranscriptionStreamResponse represents the streaming response.
type TranscriptionStreamResponse struct {
	Type  string `json:"type"`            // "transcript.text.delta" or "transcript.text.done"
	Delta string `json:"delta,omitempty"` // Only present when Type is "transcript.text.delta"
	Text  string `json:"text,omitempty"`  // Only present when Type is "transcript.text.done"
}

// defaultMaxUpload bounds an audio upload. 128 MB is a little over two hours
// of the 16 kHz mono 16-bit PCM this model reads, which is far past what a
// single request should carry -- the encoder's attention is quadratic in the
// clip (SPEECH.md S9) -- and is here to stop a body, not to size one.
const defaultMaxUpload = 128 << 20

// handleTranscription takes an audio file and returns text.
//
// Both encodings of the request are accepted: multipart/form-data, which is
// what OpenAI's clients send, and a JSON body whose "file" is base64, which
// is what curl and a test can write by hand.
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Transcription == nil {
		notLoaded(ctx, w, "speech to text", "-stt")
		return
	}
	// The body is already capped by Server.limitBody.
	var req TranscriptionRequest
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		if !parseMultipart(w, r, &req) {
			return
		}
	} else if !decodeJSON(ctx, w, r, &req) {
		return
	}
	if len(req.Data) == 0 {
		badRequest(ctx, w, "no audio: send multipart/form-data with a 'file' part, or JSON with a base64 'file'")
		return
	}
	if req.Stream {
		badRequest(ctx, w, "streaming transcription is not implemented")
		return
	}

	clip, err := audio.DecodeWAV(req.Data)
	if err != nil {
		badRequest(ctx, w, "this server decodes 16-bit PCM WAV only: "+err.Error())
		return
	}

	start := time.Now()
	resp, err := s.Transcription.Transcribe(r.Context(), clip, &req)
	if err != nil {
		backendError(ctx, w, "transcription", err)
		return
	}
	// Same shape as the speech line: the clip's length against the time it
	// took to hear it, which is what a change to this path moves.
	took := time.Since(start)
	speed := 0.0
	if took > 0 {
		speed = clip.Duration() / took.Seconds()
	}
	logf(ctx, "transcription: %.2fs of audio at %d Hz -> %d characters, %d segments, in %v (%.1fx real time)",
		clip.Duration(), clip.Rate, len(resp.Text), len(resp.Segments),
		took.Round(time.Millisecond), speed)

	switch req.ResponseFormat {
	case "", "json":
		// OpenAI's `json` is the text and nothing else, so the timings the
		// backend computed are dropped rather than sent to a client that
		// asked for the short shape.
		writeJSON(w, http.StatusOK, TranscriptionResponse{Text: resp.Text})
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, resp.Text)
	case "verbose_json":
		out := *resp
		out.Task = "transcribe"
		// The language is the request's, or absent. This checkpoint is
		// multilingual and does not report which one it heard, and
		// "english" written in by a server that did not detect it is a
		// field a client would go on to trust.
		out.Language = req.Language
		if !req.Wants("word") {
			out.Words = nil
		}
		if !req.Wants("segment") {
			out.Segments = nil
		}
		writeJSON(w, http.StatusOK, out)
	case "srt":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, encodeSRT(resp.Segments))
	case "vtt":
		w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, encodeVTT(resp.Segments))
	default:
		badRequest(ctx, w, "response_format "+strconv.Quote(req.ResponseFormat)+
			" is not supported; this server writes json, verbose_json, text, srt and vtt")
	}
}

// encodeSRT and encodeVTT are the same segments in the two subtitle
// containers. They differ in three things -- the header, the separator
// inside a timestamp, and whether the cues are numbered -- which is why they
// are one formatter and two callers.
func encodeSRT(segments []TranscriptionSegment) string {
	var b strings.Builder
	for i, seg := range segments {
		fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n\n",
			i+1, timecode(seg.Start, ","), timecode(seg.End, ","), seg.Text)
	}
	return b.String()
}

func encodeVTT(segments []TranscriptionSegment) string {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, seg := range segments {
		fmt.Fprintf(&b, "%s --> %s\n%s\n\n",
			timecode(seg.Start, "."), timecode(seg.End, "."), seg.Text)
	}
	return b.String()
}

// timecode is hours:minutes:seconds and milliseconds, with the separator the
// container wants in front of the milliseconds.
func timecode(t float64, sep string) string {
	if t < 0 {
		t = 0
	}
	ms := int64(math.Round(t * 1000))
	return fmt.Sprintf("%02d:%02d:%02d%s%03d",
		ms/3600000, ms/60000%60, ms/1000%60, sep, ms%1000)
}

// parseMultipart reads OpenAI's multipart form into req. It answers the
// client itself on a malformed body and reports whether to carry on.
func parseMultipart(w http.ResponseWriter, r *http.Request, req *TranscriptionRequest) bool {
	ctx := r.Context()
	// ParseMultipartForm's argument is how much it keeps in memory; the rest
	// spills to a temporary file, and Server.limitBody is what actually
	// bounds the upload.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if tooLarge(ctx, w, err) {
			return false
		}
		badRequest(ctx, w, "malformed multipart body: "+err.Error())
		return false
	}
	defer func() { _ = r.MultipartForm.RemoveAll() }()

	f, _, err := r.FormFile("file")
	if err != nil {
		badRequest(ctx, w, "no 'file' part in the form: "+err.Error())
		return false
	}
	defer f.Close()
	if req.Data, err = io.ReadAll(f); err != nil {
		if tooLarge(ctx, w, err) {
			return false
		}
		badRequest(ctx, w, "reading the 'file' part: "+err.Error())
		return false
	}

	req.Model = r.FormValue("model")
	req.Language = r.FormValue("language")
	req.Prompt = r.FormValue("prompt")
	req.ResponseFormat = r.FormValue("response_format")
	req.ChunkingStrategy = r.FormValue("chunking_strategy")
	// A repeated multipart field, which FormValue would give only the first
	// of.
	req.Granularities = r.Form["timestamp_granularities"]
	req.GranularitiesArray = r.Form["timestamp_granularities[]"]
	if v := r.FormValue("temperature"); v != "" {
		if req.Temperature, err = strconv.ParseFloat(v, 64); err != nil {
			badRequest(ctx, w, "temperature is not a number: "+v)
			return false
		}
	}
	if v := r.FormValue("stream"); v != "" {
		if req.Stream, err = strconv.ParseBool(v); err != nil {
			badRequest(ctx, w, "stream is not a boolean: "+v)
			return false
		}
	}
	return true
}
