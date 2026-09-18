// Package util provides utility functions for the web server.
package util

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
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
	// keep is whether this response's text is wanted in the log at all. It
	// is off for a server carrying conversations, and off for an event
	// stream whatever the caller asked, because a fragment of the first
	// kilobyte of a stream is noise rather than a record.
	keep bool

	// first is when the first byte of the response went out. On a
	// streaming endpoint that is the number worth optimising -- the total
	// only says how long the whole generation took, while this says how
	// long the caller sat looking at nothing.
	first time.Time
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
	if r.first.IsZero() {
		r.first = time.Now()
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	if r.first.IsZero() {
		r.first = time.Now()
	}
	r.size += len(b)
	if r.keep && r.body.Len() <= maxLogBodySize {
		r.body.Write(b)
	}
	return r.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController reach the real writer, so a handler
// that sets a deadline or flushes through the controller is not defeated by
// this wrapper.
func (r *responseRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (r *responseRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// timings renders how long the request took, and -- when the response was
// streamed rather than written in one go -- how long the caller waited for
// its first byte. The two are the same number on a buffered response, so the
// second is only printed when it says something the first does not.
func timings(rec *responseRecorder, start time.Time, total time.Duration) string {
	if rec.first.IsZero() {
		return fmt.Sprintf("in %v (nothing written)", total.Round(time.Microsecond))
	}
	ttfb := rec.first.Sub(start)
	if total-ttfb < time.Millisecond {
		return fmt.Sprintf("in %v", total.Round(time.Microsecond))
	}
	return fmt.Sprintf("in %v (ttfb %v, then %v streaming)",
		total.Round(time.Microsecond), ttfb.Round(time.Microsecond),
		(total - ttfb).Round(time.Microsecond))
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
// resBodyFor is the response's line: its text when the text is wanted and
// keepable, and its size and media type otherwise. An event stream is always
// the latter -- the first kilobyte of a stream is a fragment of a frame.
func resBodyFor(rec *responseRecorder, bodies bool) string {
	ct := rec.Header().Get("Content-Type")
	if !bodies || mediaType(ct) == "text/event-stream" {
		return bodyFor(ct, nil, rec.size)
	}
	return bodyFor(ct, rec.body.Bytes(), rec.size)
}

func bodyFor(contentType string, b []byte, size int) string {
	if textual(contentType) && b != nil {
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

// requestCounter numbers requests within one process run. A counter rather
// than a random id because it is short, it sorts, and "how many requests has
// this server taken" is then readable off any single line.
var requestCounter atomic.Uint64

type requestIDKey struct{}

// RequestID returns the id the access log gave this request, or "" outside
// one. Every line a handler logs about a request should carry it: the access
// log prints several lines per request and a busy server interleaves them, so
// the id is the only thing that says which lines belong together.
func RequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey{}).(string)
	return id
}

// safeRequestID accepts a client's own id only if it is short and printable.
// It ends up in a log line and in a response header, and a header is whatever
// the caller decided to send: an id carrying a newline would otherwise let a
// client write its own lines into this server's journal.
func safeRequestID(id string) string {
	if len(id) == 0 || len(id) > 64 {
		return ""
	}
	for _, c := range id {
		if c < ' ' || c > '~' {
			return ""
		}
	}
	return id
}

// clientAddr is the caller's IP without the ephemeral port, which changes
// every connection and so is noise in a log that is read by eye.
func clientAddr(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// LogRequest logs the HTTP request method, URL path, duration, and the
// payloads worth printing.
//
// **It buffers only a small textual request body.** An audio upload or a
// multipart form is left to stream to the handler untouched: reading it here
// would hold the whole thing in memory before whatever limit the handler set
// could apply, which on the transcription endpoint is the difference between
// a bounded upload and an unbounded one.
func LogRequest(fs http.Handler) http.HandlerFunc { return LogRequestFunc(true)(fs) }

// LogRequestFunc is LogRequest with the bodies made optional.
//
// **A chat server's bodies are its users' conversations.** Logging them is
// the right default for a tool being driven by hand -- every other command
// here is one -- and the wrong default for a process that stays up and serves
// other people, where it puts every prompt and every answer on disk as a side
// effect of having a log at all. So `cmd/serve` turns them off and takes a
// flag to turn them back on, and what is printed instead is the size and the
// media type, which is what an operator actually reads.
func LogRequestFunc(bodies bool) func(http.Handler) http.HandlerFunc {
	return func(fs http.Handler) http.HandlerFunc {
		return logRequest(fs, bodies)
	}
}

func logRequest(fs http.Handler, bodies bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// One id per request, echoed to the client in X-Request-Id so a
		// caller holding a failed response can name the exact line to
		// look at here. An id the client supplied wins, which is what
		// makes a trace survive a proxy in front of this.
		id := safeRequestID(r.Header.Get("X-Request-Id"))
		if id == "" {
			id = "r" + strconv.FormatUint(requestCounter.Add(1), 10)
		}
		r = r.WithContext(context.WithValue(r.Context(), requestIDKey{}, id))
		w.Header().Set("X-Request-Id", id)

		// Read and buffer the request body, when it is small and textual, so
		// downstream handlers can still consume it.
		var reqBody []byte
		buffered := false
		ct := r.Header.Get("Content-Type")
		if bodies && r.Body != nil && textual(ct) && r.ContentLength >= 0 && r.ContentLength <= maxBufferedBody {
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
		rec.keep = bodies
		start := time.Now()
		fs.ServeHTTP(rec, r)
		duration := time.Since(start)

		req := fmt.Sprintf("<%d bytes not read>", r.ContentLength)
		switch {
		case buffered:
			req = truncate(reqBody)
		case r.ContentLength == 0:
			req = ""
		case r.ContentLength < 0:
			// A chunked body of unknown length, which was streamed past
			// this logger: the default line above says so.
		case !bodies || !textual(ct):
			req = bodyFor(ct, nil, int(r.ContentLength))
		}

		// The id leads every line, including the continuations, so that
		// `grep -F "[r42]"` pulls one whole request out of an
		// interleaved log rather than just its first line.
		tag := "[" + id + "]"
		log.Printf("%s %s %s %d %s from %s\n  %s headers: %s\n  %s req body: %s\n  %s res body: %s (%d bytes)",
			tag, r.Method, r.URL.Path, rec.statusCode, timings(rec, start, duration), clientAddr(r),
			tag, formatHeaders(r.Header), tag, req,
			tag, resBodyFor(rec, bodies), rec.size)
	}
}
