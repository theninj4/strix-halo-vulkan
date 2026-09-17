package api

import (
	"io"
	"net/http"
	"strconv"
	"strings"

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
}

// TranscriptionResponse represents the non-streaming response.
type TranscriptionResponse struct {
	Text string `json:"text"`
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
	if s.Transcription == nil {
		notLoaded(w, "speech to text", "-stt")
		return
	}
	// The body is already capped by Server.limitBody.
	var req TranscriptionRequest
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		if !parseMultipart(w, r, &req) {
			return
		}
	} else if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Data) == 0 {
		badRequest(w, "no audio: send multipart/form-data with a 'file' part, or JSON with a base64 'file'")
		return
	}
	if req.Stream {
		badRequest(w, "streaming transcription is not implemented")
		return
	}

	clip, err := audio.DecodeWAV(req.Data)
	if err != nil {
		badRequest(w, "this server decodes 16-bit PCM WAV only: "+err.Error())
		return
	}

	resp, err := s.Transcription.Transcribe(r.Context(), clip, &req)
	if err != nil {
		backendError(w, "transcription", err)
		return
	}

	switch req.ResponseFormat {
	case "", "json":
		writeJSON(w, http.StatusOK, resp)
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, resp.Text)
	default:
		// verbose_json, srt and vtt all want per-segment timings. The
		// transducer has them -- every emission carries the frame it was
		// made at, in 80 ms units -- so this is a shape to fill in and not a
		// capability that is missing.
		badRequest(w, "response_format "+strconv.Quote(req.ResponseFormat)+
			" is not supported; this server writes json and text")
	}
}

// parseMultipart reads OpenAI's multipart form into req. It answers the
// client itself on a malformed body and reports whether to carry on.
func parseMultipart(w http.ResponseWriter, r *http.Request, req *TranscriptionRequest) bool {
	// ParseMultipartForm's argument is how much it keeps in memory; the rest
	// spills to a temporary file, and Server.limitBody is what actually
	// bounds the upload.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if tooLarge(w, err) {
			return false
		}
		badRequest(w, "malformed multipart body: "+err.Error())
		return false
	}
	defer func() { _ = r.MultipartForm.RemoveAll() }()

	f, _, err := r.FormFile("file")
	if err != nil {
		badRequest(w, "no 'file' part in the form: "+err.Error())
		return false
	}
	defer f.Close()
	if req.Data, err = io.ReadAll(f); err != nil {
		if tooLarge(w, err) {
			return false
		}
		badRequest(w, "reading the 'file' part: "+err.Error())
		return false
	}

	req.Model = r.FormValue("model")
	req.Language = r.FormValue("language")
	req.Prompt = r.FormValue("prompt")
	req.ResponseFormat = r.FormValue("response_format")
	req.ChunkingStrategy = r.FormValue("chunking_strategy")
	if v := r.FormValue("temperature"); v != "" {
		if req.Temperature, err = strconv.ParseFloat(v, 64); err != nil {
			badRequest(w, "temperature is not a number: "+v)
			return false
		}
	}
	if v := r.FormValue("stream"); v != "" {
		if req.Stream, err = strconv.ParseBool(v); err != nil {
			badRequest(w, "stream is not a boolean: "+v)
			return false
		}
	}
	return true
}
