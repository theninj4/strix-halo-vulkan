package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

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
			expectedBody:   "Unauthorized",
		},
		{
			name:           "Unauthorized request - wrong token",
			authHeader:     "Bearer wrongtoken",
			expectedStatus: http.StatusUnauthorized,
			expectedBody:   "Unauthorized",
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

			// Check the response body
			if rec.Body.String() != tt.expectedBody {
				t.Errorf("Expected body %q, got %q", tt.expectedBody, rec.Body.String())
			}
		})
	}
}
