package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type fakeSystemOne struct{ got []byte }

func (f *fakeSystemOne) Models() []Model { return []Model{{ID: "kev-latest"}} }

func (f *fakeSystemOne) SystemOne(_ context.Context, body []byte) ([]byte, error) {
	f.got = body
	if strings.Contains(string(body), "bad") {
		return nil, fmt.Errorf("%w: questions.a.type must be one of ...", ErrUnprocessable)
	}
	return []byte(`{"answers":{}}`), nil
}

func TestSystemOne(t *testing.T) {
	post := func(s *Server, body string, hdr map[string]string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body))
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		w := httptest.NewRecorder()
		s.Handler().ServeHTTP(w, req)
		return w
	}

	if w := post(&Server{}, `{}`, nil); w.Code != http.StatusNotImplemented || w.Header().Get("X-Typesafe-Request-Id") == "" {
		t.Errorf("unloaded: %d, request id %q", w.Code, w.Header().Get("X-Typesafe-Request-Id"))
	}

	f := &fakeSystemOne{}
	s := &Server{SystemOne: f}
	w := post(s, `{"state":"x"}`, map[string]string{"X-Typesafe-Request-Id": "abc"})
	if w.Code != http.StatusOK || w.Body.String() != `{"answers":{}}` || string(f.got) != `{"state":"x"}` {
		t.Errorf("ok: %d %q, backend saw %q", w.Code, w.Body.String(), f.got)
	}
	if got := w.Header().Get("X-Typesafe-Request-Id"); got != "abc" {
		t.Errorf("the caller's request id was not echoed: %q", got)
	}
	if w := post(s, `bad`, nil); w.Code != http.StatusUnprocessableEntity {
		t.Errorf("invalid: %d %s", w.Code, w.Body.String())
	}
}
