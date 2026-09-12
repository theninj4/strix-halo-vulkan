// Package api provides the HTTP API for the AI backend.
package api

import (
	"fmt"
	"net/http"
	"strix-halo-vulkan/util"
)

// Serve sets up the HTTP handlers for the AI backend API.
func Serve() {
	// No request / response, it's just headers.
	http.Handle("OPTIONS /v1/", util.LogRequest(http.HandlerFunc(util.CORSPreflight)))

	// Output: ModelsResponse{}
	http.Handle("GET /v1/models", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))

	// The OG completion endpoint
	// Input: CompletionRequest{}, Output: CompletionResponse{}
	http.Handle("POST /v1/chat/completions", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))
	// OpenAI's "Responses" API
	// Input: ResponsesRequest{}, Output: ResponsesResponse{}
	http.Handle("POST /v1/responses", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))
	// Anthropic's "Messages" API
	// Input: MessagesRequest{}, Output: MessagesResponse{}
	http.Handle("POST /v1/messages", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))

	// Input: EmbeddingRequest{}, Output: EmbeddingResponse{}
	http.Handle("POST /v1/embeddings", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))

	// Input: SpeechRequest{}, Output: SpeechResponse{}
	http.Handle("POST /v1/audio/speech", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))

	// Input: TranscriptionRequest{}, Output: TranscriptionResponse{}
	http.Handle(
		"POST /v1/audio/transcriptions",
		util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))),
	)

	// Input: ImageGenerationRequest{}, Output: ImageGenerationResponse{}
	http.Handle("POST /v1/images/generations", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))
	// // Input: ImageEditRequest{}, Output: ImageGenerationResponse{}
	http.Handle("POST /v1/images/edits", util.LogRequest(authorize(addHeaders(http.HandlerFunc(NotImplemented)))))
}

func authorize(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer womblesofwimbledon" {
			fs.ServeHTTP(w, r)
			return
		}

		w.WriteHeader(http.StatusUnauthorized)
		_, err := w.Write([]byte("Unauthorized"))
		util.Assert(err, "Failed to write response")
		fmt.Printf("Auth failed with %s", r.Header.Get("Authorization"))
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

func NotImplemented(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
	_, err := w.Write([]byte("Not Implemented"))
	util.Assert(err, "Failed to write response")
}
