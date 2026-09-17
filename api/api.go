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
	"net/http"
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

	// The backends, any of which may be nil.
	Speech        SpeechBackend
	Transcription TranscriptionBackend
	Completion    CompletionBackend
	Embedding     EmbeddingBackend
	Image         ImageBackend
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

	// Input: SpeechRequest{}, Output: SpeechResponse{}
	mux.Handle("POST /v1/audio/speech", s.route(s.handleSpeech))

	// Input: TranscriptionRequest{}, Output: TranscriptionResponse{}
	mux.Handle("POST /v1/audio/transcriptions", s.route(s.handleTranscription))

	// Input: ImageGenerationRequest{}, Output: ImageGenerationResponse{}
	mux.Handle("POST /v1/images/generations", s.route(s.handleImageGeneration))
	// Input: ImageEditRequest{}, Output: ImageGenerationResponse{}
	mux.Handle("POST /v1/images/edits", s.route(s.handleImageEdit))

	return mux
}

// route is the middleware stack every endpoint carries, outermost first.
//
// The body limit is outside the logger on purpose: the logger reads small
// bodies so it can print them, and it must never be handed an unbounded one.
func (s *Server) route(h http.HandlerFunc) http.Handler {
	return s.limitBody(s.log(s.authorize(addHeaders(h))))
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

func (s *Server) authorize(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.Token == "" || r.Header.Get("Authorization") == "Bearer "+s.Token {
			fs.ServeHTTP(w, r)
			return
		}

		w.WriteHeader(http.StatusUnauthorized)
		// A write that fails here is a client that hung up, which is not
		// this process's problem: log nothing and return.
		_, _ = w.Write([]byte("Unauthorized"))
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
