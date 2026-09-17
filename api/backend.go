package api

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strconv"

	"strix-halo-vulkan/audio"
)

// Backend is what every adapter has in common: the model ids it answers to.
// They are what GET /v1/models lists, and what a request's "model" field is
// matched against -- leniently, because a client configured for "tts-1" is
// still asking for the only voice model this process has.
type Backend interface {
	Models() []Model
}

// SpeechBackend synthesises speech. It returns samples rather than a file
// because the container is the HTTP layer's business: the same utterance is a
// WAV or raw PCM depending on the request's response_format, and a model
// should not know which was asked for.
type SpeechBackend interface {
	Backend
	// Speak turns text, or IPA phonemes, into mono PCM. The request carries
	// the voice and the speed; a backend that cannot honour either returns
	// an error rather than substituting one silently.
	Speak(ctx context.Context, req *SpeechRequest) (*audio.Clip, error)
	// Voices are the voice names Speak accepts, sorted. GET /v1/models
	// reports them so a client can populate a menu without a second call.
	Voices() []string
}

// TranscriptionBackend turns audio into text.
//
// The clip is decoded by the HTTP layer and handed over already in the
// float32 form every model in this repository reads, so a backend never
// parses a container. It does still check the sample rate, because resampling
// is a signal-processing decision and silently doing it at the door would
// hide a client sending 44.1 kHz into a 16 kHz model.
type TranscriptionBackend interface {
	Backend
	Transcribe(ctx context.Context, clip *audio.Clip, req *TranscriptionRequest) (*TranscriptionResponse, error)
}

// CompletionBackend generates text.
//
// **The signature is a streaming one**, and the buffered response is written
// in terms of it rather than the other way round. That is not a preference:
// a token on this part costs 12 ms (LLM.md L8d), so a 500-token answer is six
// seconds, and an interface that returned the whole thing would make the
// streamed endpoint impossible to build on top of it while the buffered one
// is trivial to build on this. emit is called once per token; an error from
// it -- a client that hung up -- stops the loop and comes back from Complete.
//
// What crosses this boundary is model-shaped and not HTTP-shaped: the backend
// says what it generated and what stopped it, and the handler decides what a
// `chat.completion` object looks like. Rendering the conversation into the
// checkpoint's own chat template is the backend's, because the template is
// part of the checkpoint.
type CompletionBackend interface {
	Backend
	Complete(ctx context.Context, req *CompletionRequest, emit func(Delta) error) (*CompletionResult, error)
}

// CompletionResult is how a generation ended.
//
// The text is not in it. The handler has already seen every token through
// emit, and a backend that returned the whole answer as well would be a
// second copy of it that could disagree with the first.
type CompletionResult struct {
	// FinishReason is "stop" -- the model emitted an end-of-generation
	// token, or ran into a stop sequence -- "length", or "tool_calls".
	FinishReason string
	// StopSequence is the stop sequence that ended the generation, when one
	// did. OpenAI's envelopes fold that into "stop" and Anthropic's reports
	// it, which is the only reason it is carried separately.
	StopSequence string
	// ToolCalls are the calls the generation made, whole. They are not
	// streamed: a call is not a call until it has closed, and its arguments
	// are not JSON until they have been typed against the tool's schema.
	ToolCalls []*ToolCall
	Usage     Usage
}

// EmbeddingBackend is a placeholder: it lets a process advertise a model it
// has loaded through GET /v1/models, and carries no method yet, because
// there is no embedding model in this repository to give it one (GOALS.md).
type EmbeddingBackend interface{ Backend }

// ImageBackend is a placeholder; see EmbeddingBackend. `zimage/pipeline` is
// the right shape for one -- resident weights, Generate(prompt, seed,
// progress) -- but its width and height are fixed at construction, so
// `aspect_ratio` is a residency question rather than a parameter and the
// interface should not pretend otherwise.
type ImageBackend interface{ Backend }

// ErrUnsupported is what a backend returns when a request is well formed but
// asks for something this model cannot do -- an unknown voice, a sample rate
// it does not take. It becomes a 400 rather than a 500, because the client
// can fix it.
var ErrUnsupported = errors.New("unsupported request")

// errorBody is OpenAI's error envelope, which is what every client of this
// API already knows how to read.
type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

type errorResponse struct {
	Error errorBody `json:"error"`
}

// writeError answers with OpenAI's error shape. typ is the coarse class the
// client switches on ("invalid_request_error", "server_error").
func writeError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: errorBody{Message: msg, Type: typ}})
}

// badRequest is the client's fault; serverError is ours, and is the one that
// gets logged, because nobody sees the response body of a 500 at 3am.
func badRequest(w http.ResponseWriter, msg string) {
	writeError(w, http.StatusBadRequest, "invalid_request_error", msg)
}

func serverError(w http.ResponseWriter, where string, err error) {
	log.Printf("api: %s: %v", where, err)
	writeError(w, http.StatusInternalServerError, "server_error", err.Error())
}

// backendError maps a backend's error onto a status. ErrUnsupported is the
// client's fault; anything else is a run that went wrong.
func backendError(w http.ResponseWriter, where string, err error) {
	if errors.Is(err, ErrUnsupported) {
		badRequest(w, err.Error())
		return
	}
	if errors.Is(err, context.Canceled) {
		// The client hung up mid-run. There is nobody to answer.
		log.Printf("api: %s: client cancelled", where)
		return
	}
	serverError(w, where, err)
}

// notLoaded is the 501 an endpoint gives when the process was started without
// the model behind it. The message names the flag, because the answer to
// "why did this 501" is almost always "it was not asked for on the command
// line".
func notLoaded(w http.ResponseWriter, what, flag string) {
	writeError(w, http.StatusNotImplemented, "not_implemented",
		what+" is not loaded; start the server with "+flag)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// tooLarge answers 413 when err is the body cap being hit, and reports
// whether it did. A body over the limit is its own failure and not a
// malformed one: the client can retry with a shorter clip.
func tooLarge(w http.ResponseWriter, err error) bool {
	var maxErr *http.MaxBytesError
	if !errors.As(err, &maxErr) {
		return false
	}
	writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error",
		"the request body is larger than this server's limit of "+
			strconv.FormatInt(maxErr.Limit, 10)+" bytes")
	return true
}

// decodeJSON reads a JSON request body into v, answering the client itself if
// it cannot. It reports whether the handler should carry on.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		if tooLarge(w, err) {
			return false
		}
		badRequest(w, "malformed JSON body: "+err.Error())
		return false
	}
	return true
}
