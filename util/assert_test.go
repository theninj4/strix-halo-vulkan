package util

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestAssertNoError(_ *testing.T) {
	// This should not panic
	Assert(nil, "This should not cause panic")
}

func TestAssertWithError(t *testing.T) {
	// Set up a deferred function to recover from panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected assert to panic with error, but it did not")
		}
	}()

	// This should panic
	Assert(errors.New("test error"), "Test error occurred")

	// If we get here, it means the function did not panic
	t.Error("Expected assert to panic but it did not")
}

func TestLogRequest(t *testing.T) {
	// Create a test HTTP handler that echoes the request body and writes a response
	testHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("response:" + string(body)))
	})

	// Create a buffer to capture log output
	var logOutput bytes.Buffer
	log.SetOutput(&logOutput)
	defer log.SetOutput(os.Stderr)

	wrappedHandler := LogRequest(testHandler)

	reqBody := `{"key":"value"}`
	req, err := http.NewRequest("POST", "/test-path", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	rr := httptest.NewRecorder()
	wrappedHandler.ServeHTTP(rr, req)

	logString := logOutput.String()

	// Verify method, path, status code, timing
	if !strings.Contains(logString, "POST") {
		t.Errorf("Log output doesn't contain request method: %s", logString)
	}
	if !strings.Contains(logString, "/test-path") {
		t.Errorf("Log output doesn't contain request path: %s", logString)
	}
	if !strings.Contains(logString, "200") {
		t.Errorf("Log output doesn't contain status code: %s", logString)
	}
	if !regexp.MustCompile(`\d+(\.\d+)?(µs|ns|ms|s)`).MatchString(logString) {
		t.Errorf("Log output doesn't contain timing information: %s", logString)
	}

	// Verify request and response payloads are logged
	if !strings.Contains(logString, reqBody) {
		t.Errorf("Log output doesn't contain request body: %s", logString)
	}
	if !strings.Contains(logString, "response:"+reqBody) {
		t.Errorf("Log output doesn't contain response body: %s", logString)
	}
}

func TestLogRequestTruncatesLargeBody(t *testing.T) {
	largeBody := strings.Repeat("x", maxLogBodySize+100)
	testHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})

	var logOutput bytes.Buffer
	log.SetOutput(&logOutput)
	defer log.SetOutput(os.Stderr)

	wrappedHandler := LogRequest(testHandler)
	req, _ := http.NewRequest("POST", "/big", strings.NewReader(largeBody))
	rr := httptest.NewRecorder()
	wrappedHandler.ServeHTTP(rr, req)

	logString := logOutput.String()
	if !strings.Contains(logString, "... (truncated)") {
		t.Errorf("Large request body was not truncated: %s", logString)
	}
}

// The bearer token is a credential and a log file is not the place for it.
func TestLogRequestRedactsAuthorization(t *testing.T) {
	var logOutput bytes.Buffer
	log.SetOutput(&logOutput)
	defer log.SetOutput(os.Stderr)

	handler := LogRequest(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	req, _ := http.NewRequest("GET", "/v1/models", http.NoBody)
	req.Header.Set("Authorization", "Bearer womblesofwimbledon")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	got := logOutput.String()
	if strings.Contains(got, "womblesofwimbledon") {
		t.Errorf("the token is in the log: %s", got)
	}
	// The scheme stays, so a 401 in this log still says what was sent.
	if !strings.Contains(got, "Bearer <redacted>") {
		t.Errorf("log output %q, want the scheme kept", got)
	}
}

// A synthesised WAV is a kilobyte of noise in a log and a second copy of
// every response in memory. It is counted instead.
func TestLogRequestSummarisesBinaryBodies(t *testing.T) {
	var logOutput bytes.Buffer
	log.SetOutput(&logOutput)
	defer log.SetOutput(os.Stderr)

	wav := append([]byte("RIFF"), bytes.Repeat([]byte{0x7f, 0x00}, 5000)...)
	handler := LogRequest(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(wav)
	}))
	req, _ := http.NewRequest("POST", "/v1/audio/speech", strings.NewReader(`{"input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	got := logOutput.String()
	if !strings.Contains(got, fmt.Sprintf("<%d bytes of audio/wav>", len(wav))) {
		t.Errorf("log output %q, want the response summarised", got)
	}
	// The request was small and textual, so it is still printed in full.
	if !strings.Contains(got, `{"input":"hi"}`) {
		t.Errorf("log output %q, want the JSON request body", got)
	}
	// And the client still got the whole body.
	if rr.Body.Len() != len(wav) {
		t.Errorf("%d bytes reached the client, want %d", rr.Body.Len(), len(wav))
	}
}

// A body past maxBufferedBody streams to the handler rather than being held
// in memory so it can be printed.
func TestLogRequestDoesNotBufferHugeBodies(t *testing.T) {
	var logOutput bytes.Buffer
	log.SetOutput(&logOutput)
	defer log.SetOutput(os.Stderr)

	body := strings.Repeat("x", maxBufferedBody+1)
	var seen int
	handler := LogRequest(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ := io.ReadAll(r.Body)
		seen = len(got)
		w.WriteHeader(http.StatusOK)
	}))
	req, _ := http.NewRequest("POST", "/big", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if seen != len(body) {
		t.Errorf("the handler read %d bytes of %d", seen, len(body))
	}
	if got := logOutput.String(); !strings.Contains(got, "not read") {
		t.Errorf("log output %q, want it to say the body was not read", got)
	}
}
