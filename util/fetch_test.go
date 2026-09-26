package util

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFetchImage(t *testing.T) {
	body := []byte("\x89PNG not really")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/a.png":
			w.Write(body)
		case "/moved":
			http.Redirect(w, r, "/a.png", http.StatusFound)
		case "/big":
			w.Write(make([]byte, MaxImageBytes+1))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	ctx := context.Background()
	for _, url := range []string{
		srv.URL + "/a.png",
		srv.URL + "/moved",
		"data:image/png;base64," + base64.StdEncoding.EncodeToString(body),
		"data:image/png;base64," + base64.RawStdEncoding.EncodeToString(body),
	} {
		got, err := FetchImage(ctx, url)
		if err != nil || string(got) != string(body) {
			t.Errorf("%.40s: %q, %v", url, got, err)
		}
	}
	for _, url := range []string{
		srv.URL + "/missing",
		srv.URL + "/big",
		"file:///etc/passwd",
		"ftp://example.com/a.png",
		"a.png",
		"data:image/png,notbase64",
		"data:image/png;base64,",
	} {
		_, err := FetchImage(ctx, url)
		if !errors.Is(err, ErrImageURL) {
			t.Errorf("%.40s: %v, want ErrImageURL", url, err)
		}
	}
	// A signed URL's query stays out of the error.
	_, err := FetchImage(ctx, srv.URL+"/missing?sig=secret")
	if err == nil || strings.Contains(err.Error(), "secret") {
		t.Errorf("error %v", err)
	}
}
