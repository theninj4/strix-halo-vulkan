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

// responseRecorder wraps http.ResponseWriter to capture the status code and response body.
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
	body       bytes.Buffer
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	r.body.Write(b)
	return r.ResponseWriter.Write(b)
}

func (r *responseRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func truncate(b []byte) string {
	if len(b) > maxLogBodySize {
		return string(b[:maxLogBodySize]) + "... (truncated)"
	}
	return string(b)
}

func formatHeaders(h http.Header) string {
	keys := make([]string, 0, len(h))
	for k := range h {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s: %s", k, strings.Join(h[k], ", ")))
	}
	return strings.Join(parts, "; ")
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

// LogRequest logs the HTTP request method, URL path, duration, and full request/response payloads.
func LogRequest(fs http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Read and buffer the request body so downstream handlers can still consume it
		var reqBody []byte
		if r.Body != nil {
			reqBody, _ = io.ReadAll(r.Body)
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(reqBody))
		}

		rec := &responseRecorder{ResponseWriter: w, statusCode: http.StatusOK}
		start := time.Now()
		fs.ServeHTTP(rec, r)
		duration := time.Since(start)

		log.Printf("%s %s %d %v\n  headers: %s\n  req body: %s\n  res body: %s",
			r.Method, r.URL.Path, rec.statusCode, duration,
			formatHeaders(r.Header), truncate(reqBody), truncate(rec.body.Bytes()))
	}
}
