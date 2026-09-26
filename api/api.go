// Package api provides the HTTP API for the AI backend.
//
// It is the routing layer and nothing else. Every endpoint is answered by a
// backend interface this package *defines* and does not implement -- see
// backend.go -- so `api` imports no checkpoint loader, no model and no
// Vulkan, and its tests run in milliseconds against fakes. `backend/` holds
// the adapters that own the weights and the device; `cmd/serve` parses the
// flags and wires the two together.
//
// A missing backend is not a missing route. Every endpoint is always served;
// one whose backend is nil answers 501. That is what makes `serve -tts`
// without `-stt` the same binary with a smaller resident set rather than a
// different build, and it is what lets a client discover what this instance
// can do from GET /v1/models instead of from a connection error.
package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"strix-halo-vulkan/util"
)

// Server is one instance of the API: a bearer token and whichever backends
// the process loaded. The zero value serves every route, asks for no token
// and answers 501 to everything, which is what a test that only exercises the
// middleware wants.
type Server struct {
	// Token is the bearer token every /v1 request must carry. Empty means
	// the API is open, which is for a loopback-only development run; it is
	// never the default in cmd/serve.
	Token string

	// MaxUploadBytes bounds a request body that carries a file -- an audio
	// upload, an image to edit. Zero takes defaultMaxUpload.
	MaxUploadBytes int64

	// LogBodies prints each request's and response's text in the access
	// log. It is off by default because this server's bodies are
	// conversations: a chat endpoint with body logging on writes every
	// prompt and every answer to disk as a side effect of having a log.
	// With it off the log still carries the method, the path, the status,
	// the duration and the sizes.
	LogBodies bool

	// Presets are the virtual models a chat request's `model` may name
	// (presets.go). Empty serves the checkpoint's own defaults to every
	// request, whatever model it names.
	Presets []Preset

	// The backends, any of which may be nil.
	Speech        SpeechBackend
	Transcription TranscriptionBackend
	Completion    CompletionBackend
	Embedding     EmbeddingBackend
	Image         ImageBackend
	SystemOne     SystemOneBackend
	// Videos is the video job queue over its backend (videos.go).
	Videos *VideoJobs
}

// Handler builds the mux. It is a fresh *http.ServeMux rather than
// http.DefaultServeMux so that two Servers -- a test's and the process's, or
// two tests' -- do not collide on the global one.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// No request / response, it's just headers. This is registered outside
	// the authorization middleware on purpose: browsers never send
	// Authorization on a preflight.
	mux.Handle("OPTIONS /v1/", s.log(http.HandlerFunc(util.CORSPreflight)))

	// Output: ModelsResponse{}
	mux.Handle("GET /v1/models", s.route(s.handleModels))

	// The OG completion endpoint
	// Input: CompletionRequest{}, Output: CompletionResponse{}
	mux.Handle("POST /v1/chat/completions", s.route(s.handleChatCompletions))
	// OpenAI's "Responses" API
	// Input: ResponsesRequest{}, Output: ResponsesResponse{}
	mux.Handle("POST /v1/responses", s.route(s.handleResponses))
	// Anthropic's "Messages" API
	// Input: MessagesRequest{}, Output: MessagesResponse{}
	mux.Handle("POST /v1/messages", s.route(s.handleMessages))

	// Input: EmbeddingRequest{}, Output: EmbeddingResponse{}
	mux.Handle("POST /v1/embeddings", s.route(s.handleEmbeddings))

	// TypeSafe's System One: typed questions about a text, answered with
	// calibrated probabilities (systemone.go, CLASSIFICATION.md).
	mux.Handle("POST /v1/systemone", s.route(s.handleSystemOne))

	// Input: SpeechRequest{}, Output: SpeechResponse{}
	mux.Handle("POST /v1/audio/speech", s.route(s.handleSpeech))

	// Input: TranscriptionRequest{}, Output: TranscriptionResponse{}
	mux.Handle("POST /v1/audio/transcriptions", s.route(s.handleTranscription))

	// Input: ImageGenerationRequest{}, Output: ImageGenerationResponse{}
	mux.Handle("POST /v1/images/generations", s.route(s.handleImageGeneration))
	// Input: ImageEditRequest{}, Output: ImageGenerationResponse{}
	mux.Handle("POST /v1/images/edits", s.route(s.handleImageEdit))

	// OpenAI's asynchronous video API (videos.go, VIDEO.md M9).
	// Input: VideoCreateRequest{}, Output: VideoJob{}
	mux.Handle("POST /v1/videos", s.route(s.handleVideoCreate))
	mux.Handle("GET /v1/videos", s.route(s.handleVideoList))
	mux.Handle("GET /v1/videos/{id}", s.route(s.handleVideoGet))
	mux.Handle("GET /v1/videos/{id}/content", s.route(s.handleVideoContent))
	mux.Handle("DELETE /v1/videos/{id}", s.route(s.handleVideoDelete))

	return mux
}

// route is the middleware stack every endpoint carries, outermost first.
//
// The body limit is outside the logger on purpose: the logger reads small
// bodies so it can print them, and it must never be handed an unbounded one.
func (s *Server) route(h http.HandlerFunc) http.Handler {
	return s.limitBody(s.log(s.authorize(addHeaders(priority(h)))))
}

type priorityKey struct{}

// priority carries a request's X-Priority header to the backend, which reads
// it with Priority. It is a header as well as a body field (`service_tier`)
// because the caller that most needs it — a voice assistant's OpenAI
// integration — builds its own bodies and will not add a field, but can be
// pointed through a proxy or given extra headers (CONCURRENCY.md).
func priority(h http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if p := r.Header.Get("X-Priority"); p != "" {
			r = r.WithContext(context.WithValue(r.Context(), priorityKey{}, p))
		}
		h.ServeHTTP(w, r)
	}
}

// Priority is the request's X-Priority header, or "". The backend decides
// what the values mean.
func Priority(ctx context.Context) string {
	p, _ := ctx.Value(priorityKey{}).(string)
	return p
}

// log is the access log, with or without the bodies.
func (s *Server) log(h http.Handler) http.HandlerFunc {
	return util.LogRequestFunc(s.LogBodies)(h)
}

// limitBody caps what a request may carry. Past the cap the read fails with
// an *http.MaxBytesError, which the handlers turn into a 413.
func (s *Server) limitBody(fs http.Handler) http.HandlerFunc {
	limit := s.MaxUploadBytes
	if limit <= 0 {
		limit = defaultMaxUpload
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, limit)
		}
		fs.ServeHTTP(w, r)
	}
}

// HTTPServer wraps Handler in an http.Server with timeouts chosen for what
// this API actually carries.
//
// ReadHeaderTimeout is short because a stalled header is never legitimate.
// ReadTimeout is minutes rather than seconds because a transcription request
// is an audio upload. **WriteTimeout is zero**, i.e. unbounded: a response
// here is a model run, and a deadline on the write side would cut off a long
// synthesis or a streamed completion mid-token. What bounds a slow client
// instead is IdleTimeout between requests.
func (s *Server) HTTPServer(addr string) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 15 * time.Second,
		ReadTimeout:       10 * time.Minute,
		WriteTimeout:      0,
		IdleTimeout:       2 * time.Minute,
	}
}

// authorize gates every /v1 route on the bearer token.
//
// A rejection is logged with *why* it was rejected. The access log already
// records that a request got a 401, but "the header was absent", "it was some
// other scheme" and "the token was wrong" are three different bugs in the
// caller and the status alone does not separate them -- which is the whole
// reason a 401 is hard to chase from the other end of the wire.
func (s *Server) authorize(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.Token == "" || r.Header.Get("Authorization") == "Bearer "+s.Token {
			fs.ServeHTTP(w, r)
			return
		}

		reason := unauthorizedReason(r.Header.Get("Authorization"))
		logf(r.Context(), "401 %s %s: %s", r.Method, r.URL.Path, reason)
		// The reason goes to the client too, in the same envelope every
		// other error here uses. It names what is wrong with the header
		// and never echoes the token that was offered.
		writeError(w, http.StatusUnauthorized, "invalid_request_error", reason)
	}
}

// unauthorizedReason classifies a failed Authorization header. It is written
// from the header alone and never compares against the configured token
// beyond "it did not match", so no part of the real credential can reach a
// log line or a response body through it.
func unauthorizedReason(header string) string {
	switch {
	case header == "":
		return "no Authorization header; pass \"Authorization: Bearer <token>\""
	case strings.EqualFold(header, "Bearer"), strings.EqualFold(header, "Bearer "):
		return "Authorization header carries no token after \"Bearer\""
	case !strings.HasPrefix(header, "Bearer "):
		scheme, _, ok := strings.Cut(header, " ")
		if !ok || scheme == "" {
			return "Authorization header is not \"Bearer <token>\""
		}
		return fmt.Sprintf("Authorization uses the %q scheme; this API takes \"Bearer <token>\"", scheme)
	default:
		return "bearer token does not match the server's -token"
	}
}

func addHeaders(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Prevent clickjacking attacks
		w.Header().Set("X-Frame-Options", "DENY")

		// Prevent MIME-type sniffing
		w.Header().Set("X-Content-Type-Options", "nosniff")

		// Enable strict HTTPS
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

		// Control permitted sources for content
		w.Header().Set("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none';")

		// Prevent information leakage
		w.Header().Set("Referrer-Policy", "no-referrer")

		// Cross-origin isolation
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		w.Header().Set("Cross-Origin-Resource-Policy", "same-site")
		w.Header().Set("X-Permitted-Cross-Domain-Policies", "none")

		// Set CORS headers (reflects the request Origin when allow-listed)
		util.SetCORSHeaders(w, r)

		// Remove server header
		w.Header().Del("Server")

		// Opt out of robots
		w.Header().
			Set("X-Robots-Tag", "noindex, nofollow, noarchive, nositelinkssearchbox, nosnippet, notranslate, noimageindex")

		// Disable caching
		w.Header().Set("Cache-Control", "no-cache, must-revalidate, max-age=0")

		fs.ServeHTTP(w, r)
	}
}
