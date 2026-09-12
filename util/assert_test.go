package util

import (
	"bytes"
	"errors"
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
