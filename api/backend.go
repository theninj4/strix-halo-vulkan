package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"log"
	"math"
	"net/http"
	"strconv"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/util"
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

// EmbeddingBackend turns texts into vectors.
//
// The whole request crosses the boundary rather than just the texts, because
// three of its fields are the model's business and not the envelope's: the
// instruction a query is prefixed with, the output width an MRL model can be
// truncated to, and -- through the text count -- how the backend batches. The
// encoding format is *not* the backend's: base64 or JSON numbers is a
// transport question, and a model that knew about it would be answering two
// questions at once.
//
// Vectors come back in the request's order, one per input, already
// normalised: every consumer of an embedding either takes a cosine or a dot
// product, and those are the same number only if the vectors are unit length.
type EmbeddingBackend interface {
	Backend
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResult, error)
}

// EmbeddingResult is a batch of vectors and what they cost.
type EmbeddingResult struct {
	// Vectors are in the request's input order, one per input.
	Vectors [][]float32
	// Usage counts the tokens actually run, which is after truncation and
	// after any instruction prefix -- what the model read, as everywhere
	// else in this API. CompletionTokens is always zero.
	Usage Usage
}

// errVectorCount is the one thing the handler checks about a backend's
// answer, because getting it wrong misaligns a client's corpus silently.
var errVectorCount = errors.New("the backend returned a different number of vectors than there were inputs")

// ImageBackend generates images.
//
// What crosses this boundary is an `image.Image` and not a file, for the same
// reason a SpeechBackend returns samples: PNG or JPEG, base64 or raw, is what
// the request asked for and is none of the model's business.
//
// The geometry is split in two because the two halves are different kinds of
// decision. **A size is a request parameter** -- the transformer states its
// run length per upload and the VAE re-records its graph from the latent it
// is handed, so a smaller image is the same graph with smaller numbers in it.
// **A ceiling is residency**, fixed when the arenas were allocated. So a
// backend answers Geometry() once and the handler validates against it,
// rather than every request discovering the limit by failing.
type ImageBackend interface {
	Backend
	// Generate renders one image. The request's geometry is already resolved
	// -- `size`, `aspect_ratio` and the defaults are the HTTP layer's
	// business -- so what arrives here is pixels.
	Generate(ctx context.Context, req *ImageRequest) (*ImageResult, error)
	// Geometry is what this process was started for: the size a request that
	// names none gets, the largest it will accept, and whether it can send
	// in-progress previews.
	Geometry() ImageGeometry
}

// ImageRequest is one image to render, with everything resolved.
type ImageRequest struct {
	Prompt string
	// Width and Height are in pixels; zero takes the backend's default.
	Width, Height int
	// Steps is the denoising schedule's length; zero takes the backend's.
	Steps int
	// Seed is the initial latent's. Nil draws one, and the result says which
	// was drawn, so an image a caller likes can be asked for again.
	Seed *int64
	// Transparent asks for an RGBA image with a transparent background, i.e.
	// OpenAI's `background: "transparent"`.
	//
	// It is a request field rather than a post-process because in this model
	// it is not one: Qwen-Image-2.1's VAE emits four channels whatever the
	// prompt says, and what makes the background actually transparent is
	// *asking for it in the prompt*. So a backend that sets this rewrites the
	// prompt and keeps the alpha plane, and one that leaves it false
	// composites over white. A backend whose model has no alpha refuses it.
	Transparent bool

	// PartialImages is how many in-progress frames the caller wants before
	// the finished one, and Partial is where they go. Zero, or a nil Partial,
	// asks for none -- which is the only thing a backend without a preview
	// decoder can honour, and why ImageGeometry reports whether it has one.
	//
	// **Which steps they come from is the backend's decision, not the HTTP
	// layer's.** The step count is the backend's (a request may not have
	// named one), the relationship between a step and how finished the
	// picture looks is the scheduler's, and neither is visible from a
	// handler. What the handler decides is how many.
	PartialImages int
	// Partial is called with each in-progress frame, in order, before
	// Generate returns. An error from it -- a client that hung up mid-stream
	// is the one that happens -- stops any *further* frames and is what
	// Generate returns.
	//
	// It does not stop the run. A denoising step is a submit-and-fence with
	// no cancellation point in it, so a request that is abandoned halfway
	// still costs the device the whole image; what it stops costing is the
	// encoding and the writing. That is the same bargain the non-streaming
	// path already makes, and it is written down in API.md rather than fixed.
	Partial func(ImagePartial) error

	// Init holds the reference images an edit is conditioned on, in the
	// order the client sent them, and a non-empty Init is what makes this
	// request an edit rather than a generation. They are image.Images at
	// whatever size the client sent: resizing them is the *backend's*,
	// because a resample is arithmetic on pixels and the HTTP layer has no
	// business doing arithmetic on pixels -- and in this model the resampler
	// is part of the model, exact to the 8-bit level.
	//
	// **An edit is the same call as a generation and not a second one**,
	// which is the whole reason it is a field here rather than a method on
	// the interface: partial frames, seeds, sizes and step counts all mean
	// exactly what they already meant, and the streaming path did not have
	// to be written twice.
	//
	// It is a *list*, and that is the model's doing rather than OpenAI's.
	// Under SDEdit an edit had one input and a `strength` saying how much of
	// it to keep. Qwen-Image-2.1 edits by conditional generation: the
	// references become rows of the transformer's prefix, up to ten of them,
	// every step runs, and there is no strength knob to have. A backend
	// reports how many it will take as ImageGeometry.MaxRefs.
	Init []image.Image
}

// ImagePartial is one in-progress frame of an image being generated.
//
// It is a decoded picture rather than a latent for the same reason
// ImageResult is: what a preview decoder is for is that the HTTP layer never
// has to know what a latent is.
type ImagePartial struct {
	// Index counts the partials actually sent, from zero. It is OpenAI's
	// partial_image_index and it is not the step number -- a client asking
	// for three frames of an eight-step schedule gets 0, 1, 2.
	Index int
	// Step is the denoising step the frame came from, and Steps how many
	// there are. Neither is in OpenAI's envelope; they are here because a
	// client watching a preview arrive wants to know how much is left, and
	// the alternative is guessing from the index.
	Step, Steps   int
	Image         image.Image
	Width, Height int
}

// ImageResult is one rendered image and the parameters that produced it --
// including the ones the request left to the server, which is the only way a
// client can reproduce an image it did not fully specify.
type ImageResult struct {
	Image         image.Image
	Width, Height int
	Steps         int
	Seed          int64
}

// ImageGeometry is what an image backend will accept. It is reported on the
// model object in GET /v1/models for the same reason the voice list is: the
// alternative is a client discovering the limit from a 400.
//
// **The ceiling is an area and not a box**, which is a correction: under
// Z-Image it was a pair of side limits, on the argument that the VAE's
// blocked fp16 copies are padded per axis so 128x512 costs more arena than
// the 256x256 it has the same area as. That argument does not survive the
// Qwen-Image-2.1 decoder, and it was measured rather than reasoned about:
// `qimage/vae`'s TestArenaShape plans the graph for a dozen shapes and finds
// the arena is **exactly 3060 bytes a pixel** for every one of them, 256x4096
// and 1024x1024 included. The transformer's row count is the pixel count over
// 256 by construction. So residency is a function of the *area* alone, and a
// side limit taxes every non-square request for nothing: 16:9 inside a
// 1024x1024 box is 1024x576, 56% of the pixels a square request gets from the
// same arenas.
type ImageGeometry struct {
	// Width and Height are what a request that names no size gets.
	Width, Height int
	// MaxPixels is the area budget: width*height may not exceed it, and the
	// shape is otherwise free. Zero means unbounded.
	MaxPixels int
	// Multiple is what both sides must be a multiple of.
	Multiple int
	// Steps is the default denoising schedule's length.
	Steps int
	// Previews reports whether this backend can send in-progress frames, i.e.
	// whether a preview decoder is loaded. A client reads it to know whether
	// `stream: true` will be answered or refused.
	Previews bool
	// MaxPartials bounds `partial_images`. It is OpenAI's 3 and it is a
	// policy rather than a limit of the model: each frame is a decode, and
	// what makes three reasonable is that it is 5% of the image at 1024x1024.
	MaxPartials int
	// Edits reports whether ImageRequest.Init will be accepted, i.e. whether
	// the vision tower and the VAE's *encoder* are resident. A client reads
	// it to know whether /v1/images/edits will be answered or refused, for
	// the same reason it reads Previews.
	Edits bool
	// MaxRefs is how many reference images one edit may carry. It is
	// residency and not policy -- every reference adds its latent rows to
	// the transformer's prefix and its own share of the prefix KV cache, so
	// the number is fixed when the server starts. Zero when Edits is false.
	MaxRefs int
}

// MaxSquare is the largest square inside the pixel budget, snapped down to
// Multiple. It is what `max_size` reports, because a client that wants one
// number rather than a rule wants this one -- and because a size the server
// is certain to accept is the useful thing to echo back into a request.
func (g ImageGeometry) MaxSquare() (int, int) {
	m := g.Multiple
	if m < 1 {
		m = 1
	}
	side := int(math.Sqrt(float64(g.MaxPixels))) / m * m
	if side < m {
		side = m
	}
	return side, side
}

// MarshalJSON writes the geometry the way a client reads it: the sizes
// spelled as OpenAI spells a size, so one can be echoed straight back into a
// request's `size` without being reassembled first.
//
// `max_pixels` is the rule and `max_size` is the largest square that obeys
// it. Both are reported because a client asking for a square only needs the
// second, and one asking for 16:9 cannot work it out from the second alone.
func (g ImageGeometry) MarshalJSON() ([]byte, error) {
	maxW, maxH := g.MaxSquare()
	return json.Marshal(struct {
		DefaultSize  string `json:"default_size"`
		MaxSize      string `json:"max_size"`
		MaxPixels    int    `json:"max_pixels"`
		SizeMultiple int    `json:"size_multiple"`
		DefaultSteps int    `json:"default_steps"`
		Previews     bool   `json:"previews"`
		MaxPartials  int    `json:"max_partial_images,omitempty"`
		Edits        bool   `json:"edits"`
		MaxRefs      int    `json:"max_reference_images,omitempty"`
	}{
		DefaultSize:  formatSize(g.Width, g.Height),
		MaxSize:      formatSize(maxW, maxH),
		MaxPixels:    g.MaxPixels,
		SizeMultiple: g.Multiple,
		DefaultSteps: g.Steps,
		Previews:     g.Previews,
		MaxPartials:  g.MaxPartials,
		Edits:        g.Edits,
		MaxRefs:      g.MaxRefs,
	})
}

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

// logf is how this package logs anything about a request in flight. It
// carries the access log's request id, without which a line like
// "api: chat completion: context deadline exceeded" cannot be attributed to
// one of the several requests that were in flight when it was written.
func logf(ctx context.Context, format string, args ...any) {
	if id := util.RequestID(ctx); id != "" {
		log.Printf("[%s] api: %s", id, fmt.Sprintf(format, args...))
		return
	}
	log.Printf("api: %s", fmt.Sprintf(format, args...))
}

// badRequest is the client's fault; serverError is ours, and is the one that
// gets logged, because nobody sees the response body of a 500 at 3am.
//
// A 400 is logged too. It is the client's mistake rather than the server's,
// but it is still a request that did not do what whoever sent it wanted, and
// the reason is in the response body where only they can see it.
func badRequest(ctx context.Context, w http.ResponseWriter, msg string) {
	logf(ctx, "400: %s", msg)
	writeError(w, http.StatusBadRequest, "invalid_request_error", msg)
}

func serverError(ctx context.Context, w http.ResponseWriter, where string, err error) {
	logf(ctx, "%s: %v", where, err)
	writeError(w, http.StatusInternalServerError, "server_error", err.Error())
}

// backendError maps a backend's error onto a status. ErrUnsupported is the
// client's fault; anything else is a run that went wrong.
func backendError(ctx context.Context, w http.ResponseWriter, where string, err error) {
	if errors.Is(err, ErrUnsupported) {
		badRequest(ctx, w, err.Error())
		return
	}
	if errors.Is(err, context.Canceled) {
		// The client hung up mid-run. There is nobody to answer.
		logf(ctx, "%s: client cancelled", where)
		return
	}
	serverError(ctx, w, where, err)
}

// notLoaded is the 501 an endpoint gives when the process was started without
// the model behind it. The message names the flag, because the answer to
// "why did this 501" is almost always "it was not asked for on the command
// line".
func notLoaded(ctx context.Context, w http.ResponseWriter, what, flag string) {
	logf(ctx, "501: %s is not loaded (needs %s)", what, flag)
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
func tooLarge(ctx context.Context, w http.ResponseWriter, err error) bool {
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
func decodeJSON(ctx context.Context, w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		if tooLarge(ctx, w, err) {
			return false
		}
		badRequest(ctx, w, "malformed JSON body: "+err.Error())
		return false
	}
	return true
}
