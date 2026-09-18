package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A 401 must say which of the three client mistakes it was, and must never
// echo the credential it was offered.
func TestUnauthorizedReason(t *testing.T) {
	tests := []struct{ header, want string }{
		{"", "no Authorization header"},
		{"Bearer", "no token"},
		{"Bearer ", "no token"},
		{"Basic dXNlcjpwdw==", `"Basic" scheme`},
		{"tokenwithnoscheme", `not "Bearer <token>"`},
		{"Bearer wrongtoken", "does not match"},
	}
	for _, tt := range tests {
		got := unauthorizedReason(tt.header)
		if !strings.Contains(got, tt.want) {
			t.Errorf("unauthorizedReason(%q) = %q, want it to contain %q", tt.header, got, tt.want)
		}
		if strings.Contains(got, "wrongtoken") || strings.Contains(got, "dXNlcjpwdw==") {
			t.Errorf("unauthorizedReason(%q) = %q, which echoes the credential", tt.header, got)
		}
	}
}

func TestAuthorizeProxy(t *testing.T) {
	tests := []struct {
		name           string
		authHeader     string
		expectedBody   string
		expectedStatus int
	}{
		{
			name:           "Authorized request",
			authHeader:     "Bearer womblesofwimbledon",
			expectedStatus: http.StatusOK,
			expectedBody:   "Success",
		},
		{
			name:           "Unauthorized request - no header",
			authHeader:     "",
			expectedStatus: http.StatusUnauthorized,
			expectedBody:   "no Authorization header",
		},
		{
			name:           "Unauthorized request - wrong token",
			authHeader:     "Bearer wrongtoken",
			expectedStatus: http.StatusUnauthorized,
			expectedBody:   "does not match",
		},
		{
			name:           "Unauthorized request - wrong scheme",
			authHeader:     "Basic dXNlcjpwdw==",
			expectedStatus: http.StatusUnauthorized,
			// The envelope is JSON, so the quotes around the scheme
			// arrive backslash-escaped; match the part that does not
			// depend on the encoding.
			expectedBody: "scheme; this API takes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test handler that writes a success message
			testHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("Success"))
			})

			// Wrap the test handler with our authorize middleware
			s := &Server{Token: "womblesofwimbledon"}
			handler := s.authorize(testHandler)

			// Create a test request
			req := httptest.NewRequest("GET", "/test", http.NoBody)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			// Create a test response recorder
			rec := httptest.NewRecorder()

			// Call the handler
			handler.ServeHTTP(rec, req)

			// Check the status code
			if rec.Code != tt.expectedStatus {
				t.Errorf("Expected status code %d, got %d", tt.expectedStatus, rec.Code)
			}

			// The body is OpenAI's error envelope on a rejection, so
			// what matters is that it names the reason, not that it is
			// one exact string.
			if !strings.Contains(rec.Body.String(), tt.expectedBody) {
				t.Errorf("Expected body to contain %q, got %q", tt.expectedBody, rec.Body.String())
			}
		})
	}
}
