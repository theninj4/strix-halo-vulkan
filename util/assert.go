// Package util provides utility functions for the web server.
package util

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
)

const maxLogBodySize = 1024

// maxBufferedBody is the largest request body LogRequest will read into
// memory so that it can be printed. Past it the body streams to the handler
// untouched and the log says how big it was: a logger is not a reason to hold
// a copy of an upload, and on the transcription endpoint it is the difference
// between a bounded upload and an unbounded one.
const maxBufferedBody = 1 << 20

// responseRecorder wraps http.ResponseWriter to capture the status code and response body.
//
// A binary body -- a synthesised WAV, raw PCM, a PNG -- is counted and not
// kept: logging it would print a kilobyte of noise per request and hold a
// second copy of every response in memory while it did.
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
	body       bytes.Buffer
	size       int
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	r.size += len(b)
	if textual(r.Header().Get("Content-Type")) && r.body.Len() <= maxLogBodySize {
		r.body.Write(b)
	}
	return r.ResponseWriter.Write(b)
}

func (r *responseRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// errorReader replays a read failure to whoever reads next.
type errorReader struct{ err error }

func (e errorReader) Read([]byte) (int, error) { return 0, e.err }

func truncate(b []byte) string {
	if len(b) > maxLogBodySize {
		return string(b[:maxLogBodySize]) + "... (truncated)"
	}
	return string(b)
}

// mediaType is a Content-Type without its parameters, lowercased -- so that
// a multipart form's boundary does not end up in the log line summarising it.
func mediaType(contentType string) string {
	if i := strings.IndexByte(contentType, ';'); i >= 0 {
		contentType = contentType[:i]
	}
	return strings.TrimSpace(strings.ToLower(contentType))
}

// textual reports whether a body of this content type is worth printing.
// An empty type is treated as textual: that is what a handler that wrote
// nothing, or wrote plain text without saying so, has.
func textual(contentType string) bool {
	contentType = mediaType(contentType)
	switch {
	case contentType == "":
		return true
	case strings.HasPrefix(contentType, "text/"):
		return true
	case strings.HasPrefix(contentType, "application/json"):
		return true
	case contentType == "application/x-www-form-urlencoded":
		return true
	case strings.HasSuffix(contentType, "+json"):
		return true
	}
	return false
}

// bodyFor is what the log prints for one body: the body itself when it is
// text, and its size and type when it is not.
func bodyFor(contentType string, b []byte, size int) string {
	if textual(contentType) {
		return truncate(b)
	}
	if t := mediaType(contentType); t != "" {
		return fmt.Sprintf("<%d bytes of %s>", size, t)
	}
	return fmt.Sprintf("<%d bytes>", size)
}

func formatHeaders(h http.Header) string {
	keys := make([]string, 0, len(h))
	for k := range h {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		v := strings.Join(h[k], ", ")
		// The bearer token is a credential, and a log file is not the place
		// for it. The scheme is kept because "was it sent at all, and as
		// what" is the question a 401 in this log is read to answer.
		if strings.EqualFold(k, "Authorization") {
			v = redactCredential(v)
		}
		parts = append(parts, fmt.Sprintf("%s: %s", k, v))
	}
	return strings.Join(parts, "; ")
}

// redactCredential keeps an Authorization header's scheme and replaces its
// value, so "Bearer <redacted>" still distinguishes a missing header from a
// wrong token.
func redactCredential(v string) string {
	if v == "" {
		return v
	}
	if scheme, _, ok := strings.Cut(v, " "); ok {
		return scheme + " <redacted>"
	}
	return "<redacted>"
}

// Assert checks if an error is not nil and logs a message if it is.
func Assert(e error, msg string, args ...interface{}) {
	if e != nil {
		// Log the error message with the provided arguments
		// fmt.Printf("Error: %s\n", fmt.Sprintf(msg, args...))
		panic(e)
	}
}

// AssertOk checks if a condition is true and logs a message if it is not.
func AssertOk(ok bool, msg string, args ...interface{}) {
	if !ok {
		// Log the error message with the provided arguments
		// fmt.Printf("Error: %s\n", fmt.Sprintf(msg, args...))
		panic(msg)
	}
}

// LogRequest logs the HTTP request method, URL path, duration, and the
// payloads worth printing.
//
// **It buffers only a small textual request body.** An audio upload or a
// multipart form is left to stream to the handler untouched: reading it here
// would hold the whole thing in memory before whatever limit the handler set
// could apply, which on the transcription endpoint is the difference between
// a bounded upload and an unbounded one.
func LogRequest(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Read and buffer the request body, when it is small and textual, so
		// downstream handlers can still consume it.
		var reqBody []byte
		buffered := false
		ct := r.Header.Get("Content-Type")
		if r.Body != nil && textual(ct) && r.ContentLength >= 0 && r.ContentLength <= maxBufferedBody {
			var err error
			reqBody, err = io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				// The read failed -- a cap was hit, the connection broke.
				// The handler is given the same failure rather than a
				// truncated body, which it would parse as a malformed one
				// and answer the wrong thing about.
				r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(reqBody), errorReader{err}))
			} else {
				r.Body = io.NopCloser(bytes.NewReader(reqBody))
			}
			buffered = true
		}

		rec := &responseRecorder{ResponseWriter: w, statusCode: http.StatusOK}
		start := time.Now()
		fs.ServeHTTP(rec, r)
		duration := time.Since(start)

		req := fmt.Sprintf("<%d bytes not read>", r.ContentLength)
		switch {
		case buffered:
			req = truncate(reqBody)
		case r.ContentLength == 0:
			req = ""
		case !textual(ct):
			req = bodyFor(ct, nil, int(r.ContentLength))
		}

		log.Printf("%s %s %d %v\n  headers: %s\n  req body: %s\n  res body: %s",
			r.Method, r.URL.Path, rec.statusCode, duration,
			formatHeaders(r.Header), req,
			bodyFor(rec.Header().Get("Content-Type"), rec.body.Bytes(), rec.size))
	}
}
