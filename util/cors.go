package util

import (
	"net/http"
	"net/url"
)

// prodOrigin is the origin the production frontend is served from.
const prodOrigin = "https://x.e12b.com"

// allowedOrigin reports whether the given Origin header value may make
// cross-origin requests to the backend. The production frontend is served from
// prodOrigin; local development runs on an arbitrary loopback port (Vite), so
// any localhost/127.0.0.1/::1 origin is permitted too.
func allowedOrigin(origin string) bool {
	if origin == prodOrigin {
		return true
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	switch u.Hostname() {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	return false
}

// SetCORSHeaders writes the CORS response headers, reflecting the request's
// Origin back when it is allow-listed. A previous value of
// "x.e12b.com" (without a scheme) is not a valid Origin and is never matched by
// a browser, so requests were blocked. Shared by the API and asset handlers.
func SetCORSHeaders(w http.ResponseWriter, r *http.Request) {
	if origin := r.Header.Get("Origin"); allowedOrigin(origin) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Add("Vary", "Origin")
	}
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Allow-Credentials", "false")
	w.Header().Set("Access-Control-Max-Age", "86400")
}

// CORSPreflight answers CORS preflight (OPTIONS) requests. Browsers never send
// the Authorization header on a preflight, so this must run before the
// authorization middleware and simply returns the CORS headers with an empty
// 204 response.
func CORSPreflight(w http.ResponseWriter, r *http.Request) {
	SetCORSHeaders(w, r)
	w.WriteHeader(http.StatusNoContent)
}
