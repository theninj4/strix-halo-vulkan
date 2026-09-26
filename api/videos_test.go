package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"
	"time"
)

// fakeVideo plans every request at the shape it asked for and writes a few
// bytes as its "mp4". A test holds a job in GenerateVideo by closing over
// gate, and sees it start on started.
type fakeVideo struct {
	mu      sync.Mutex
	reqs    []*VideoRequest
	gate    chan struct{} // nil runs straight through
	started chan string
	fail    error
}

func (f *fakeVideo) Models() []Model {
	return []Model{{ID: "minimax-h3", Object: "model", OwnedBy: "local"}}
}

func (f *fakeVideo) VideoGeometry() VideoGeometry {
	return VideoGeometry{DefaultSize: "864x480", DefaultSeconds: 5, MinSeconds: 5, MaxSeconds: 15,
		MaxShortEdge: 768, DefaultSteps: 20, FPS: 24, SampleRate: 32000}
}

func (f *fakeVideo) PlanVideo(req *VideoRequest) (*VideoPlan, error) {
	if req.Seconds > 15 {
		return nil, fmt.Errorf("too long: %w", ErrUnsupported)
	}
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	seed := int64(7)
	if req.Seed != nil {
		seed = *req.Seed
	}
	return &VideoPlan{Width: 864, Height: 480, Frames: 124, Seconds: 124.0 / 24, Steps: 20, Seed: seed,
		Estimate: 13 * time.Minute}, nil
}

func (f *fakeVideo) GenerateVideo(ctx context.Context, req *VideoRequest, plan *VideoPlan, dst string, progress func(VideoProgress)) error {
	if f.started != nil {
		f.started <- req.Prompt
	}
	progress(VideoProgress{Stage: "step", Fraction: 0.5})
	if f.gate != nil {
		select {
		case <-f.gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if f.fail != nil {
		return f.fail
	}
	return os.WriteFile(dst, []byte("not really an mp4"), 0o644)
}

func newVideoServer(t *testing.T, f *fakeVideo, opt VideoJobsOptions) *Server {
	t.Helper()
	if opt.Dir == "" {
		opt.Dir = t.TempDir()
	}
	q, err := NewVideoJobs(f, opt)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(q.Close)
	return &Server{Videos: q}
}

func decodeJob(t *testing.T, rec *httptest.ResponseRecorder) VideoJob {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var j VideoJob
	if err := json.Unmarshal(rec.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	return j
}

// waitStatus polls a job until it reaches status.
func waitStatus(t *testing.T, s *Server, id, status string) VideoJob {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		j := decodeJob(t, do(t, s, httptest.NewRequest("GET", "/v1/videos/"+id, nil)))
		if j.Status == status {
			return j
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %s is %s, never %s", id, j.Status, status)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestVideoLifecycle: create, poll to completed, fetch the content, list,
// delete, and the 404 after.
func TestVideoLifecycle(t *testing.T) {
	f := &fakeVideo{}
	s := newVideoServer(t, f, VideoJobsOptions{})
	j := decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{
		"model": "sora-2", "prompt": "a lighthouse", "seconds": "8", "size": "1280x720",
	})))
	if j.Object != "video" || j.Model != "minimax-h3" || j.Size != "864x480" || j.Seconds != "5.17" ||
		j.Seed != 7 || j.Steps != 20 || j.Estimated != 780 || j.CompletedAt != nil {
		t.Errorf("created job %+v", j)
	}
	r := f.reqs[0]
	if r.AspectW != 1280 || r.AspectH != 720 || r.ShortEdge != 720 || r.Seconds != 8 || r.Prompt != "a lighthouse" {
		t.Errorf("request %+v", r)
	}
	done := waitStatus(t, s, j.ID, VideoCompleted)
	if done.Progress != 100 || done.CompletedAt == nil || done.ExpiresAt == nil || *done.ExpiresAt-*done.CompletedAt != 86400 {
		t.Errorf("completed job %+v", done)
	}
	rec := do(t, s, httptest.NewRequest("GET", "/v1/videos/"+j.ID+"/content", nil))
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "video/mp4" || rec.Body.String() != "not really an mp4" {
		t.Errorf("content: %d %q %q", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	rec = do(t, s, httptest.NewRequest("GET", "/v1/videos", nil))
	var list VideoList
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil || len(list.Data) != 1 || list.Data[0].ID != j.ID {
		t.Errorf("list: %v %s", err, rec.Body)
	}
	rec = do(t, s, httptest.NewRequest("DELETE", "/v1/videos/"+j.ID, nil))
	if rec.Code != http.StatusOK || !bytes.Contains(rec.Body.Bytes(), []byte(`"deleted":true`)) {
		t.Errorf("delete: %d %s", rec.Code, rec.Body)
	}
	if _, err := os.Stat(s.Videos.Dir() + "/" + j.ID + ".mp4"); !os.IsNotExist(err) {
		t.Errorf("the file outlived its job: %v", err)
	}
	for _, path := range []string{"/v1/videos/" + j.ID, "/v1/videos/" + j.ID + "/content"} {
		if rec := do(t, s, httptest.NewRequest("GET", path, nil)); rec.Code != http.StatusNotFound {
			t.Errorf("%s after delete: %d", path, rec.Code)
		}
	}
}

// TestVideoSGLangEnvelope: the H3 request scripts' shape reaches the backend
// as the same request.
func TestVideoSGLangEnvelope(t *testing.T) {
	f := &fakeVideo{}
	s := newVideoServer(t, f, VideoJobsOptions{})
	decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{
		"task": "t2va", "prompt": "p", "conditions": []any{},
		"target": map[string]any{"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10},
		"seed":   0, "num_inference_steps": 50,
	})))
	r := f.reqs[0]
	if r.AspectW != 16 || r.AspectH != 9 || r.ShortEdge != 768 || r.Seconds != 10 || r.Steps != 50 || r.Seed == nil || *r.Seed != 0 {
		t.Errorf("request %+v", r)
	}
}

// TestVideoMultipart: OpenAI's SDK sends a form.
func TestVideoMultipart(t *testing.T) {
	f := &fakeVideo{}
	s := newVideoServer(t, f, VideoJobsOptions{})
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	mw.WriteField("prompt", "a fox")
	mw.WriteField("seconds", "12")
	mw.WriteField("size", "720x1280")
	mw.WriteField("seed", "42")
	mw.Close()
	req := httptest.NewRequest("POST", "/v1/videos", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	j := decodeJob(t, do(t, s, req))
	r := f.reqs[0]
	if r.Prompt != "a fox" || r.Seconds != 12 || r.AspectW != 720 || r.AspectH != 1280 || r.ShortEdge != 720 || j.Seed != 42 {
		t.Errorf("request %+v, job %+v", r, j)
	}
}

func TestVideoRejects(t *testing.T) {
	s := newVideoServer(t, &fakeVideo{}, VideoJobsOptions{})
	for name, body := range map[string]map[string]any{
		"no prompt":     {"seconds": "8"},
		"keyframes":     {"prompt": "p", "task": "fl2va"},
		"conditions":    {"prompt": "p", "conditions": []any{map[string]any{"type": "image"}}},
		"bad size":      {"prompt": "p", "size": "wide"},
		"bad aspect":    {"prompt": "p", "target": map[string]any{"aspect_ratio": "16x9"}},
		"bad seconds":   {"prompt": "p", "seconds": "eight"},
		"backend says":  {"prompt": "p", "seconds": 20},
		"unknown task":  {"prompt": "p", "task": "v2v"},
		"negative step": {"prompt": "p", "steps": -1},
	} {
		if rec := do(t, s, jsonRequest("POST", "/v1/videos", body)); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: %d %s", name, rec.Code, rec.Body)
		}
	}
	if rec := do(t, s, httptest.NewRequest("GET", "/v1/videos/video_nope", nil)); rec.Code != http.StatusNotFound {
		t.Errorf("unknown id: %d", rec.Code)
	}
	if rec := do(t, &Server{}, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": "p"})); rec.Code != http.StatusNotImplemented {
		t.Errorf("not loaded: %d", rec.Code)
	}
}

// TestVideoQueueAndCancel: one job runs at a time; the content of an
// unfinished one is a 409; a full queue is a 429; DELETE cancels the running
// job and the next one starts.
func TestVideoQueueAndCancel(t *testing.T) {
	f := &fakeVideo{gate: make(chan struct{}), started: make(chan string, 4)}
	s := newVideoServer(t, f, VideoJobsOptions{MaxQueued: 1})
	post := func(p string) *httptest.ResponseRecorder {
		return do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": p}))
	}
	a := decodeJob(t, post("a"))
	if got := <-f.started; got != "a" {
		t.Fatalf("started %q", got)
	}
	running := waitStatus(t, s, a.ID, VideoInProgress)
	if running.Progress != 50 || running.Stage != "step" {
		t.Errorf("running job %+v", running)
	}
	b := decodeJob(t, post("b"))
	if rec := post("c"); rec.Code != http.StatusTooManyRequests {
		t.Errorf("past the queue: %d", rec.Code)
	}
	if rec := do(t, s, httptest.NewRequest("GET", "/v1/videos/"+b.ID+"/content", nil)); rec.Code != http.StatusConflict {
		t.Errorf("queued content: %d", rec.Code)
	}
	if rec := do(t, s, httptest.NewRequest("DELETE", "/v1/videos/"+a.ID, nil)); rec.Code != http.StatusOK {
		t.Fatalf("delete running: %d", rec.Code)
	}
	if got := <-f.started; got != "b" {
		t.Fatalf("after the cancel, started %q", got)
	}
	close(f.gate)
	waitStatus(t, s, b.ID, VideoCompleted)
}

// TestVideoFailureAndShutdown: a backend error is a failed job with the
// message; Close fails what is still queued.
func TestVideoFailureAndShutdown(t *testing.T) {
	f := &fakeVideo{fail: errors.New("the device went away")}
	s := newVideoServer(t, f, VideoJobsOptions{})
	j := decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": "p"})))
	failed := waitStatus(t, s, j.ID, VideoFailed)
	if failed.Error == nil || failed.Error.Code != "generation_failed" || failed.Error.Message != "the device went away" {
		t.Errorf("failed job %+v", failed)
	}

	g := &fakeVideo{gate: make(chan struct{}), started: make(chan string, 4)}
	s = newVideoServer(t, g, VideoJobsOptions{})
	a := decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": "a"})))
	<-g.started
	b := decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": "b"})))
	s.Videos.Close()
	for _, id := range []string{a.ID, b.ID} {
		j, _ := s.Videos.Get(id)
		if j.Status != VideoFailed || j.Error.Code != "server_shutdown" {
			t.Errorf("%s after Close: %+v", id, j)
		}
	}
}

// TestVideoExpiry: a finished job and its file go at their expiry.
func TestVideoExpiry(t *testing.T) {
	s := newVideoServer(t, &fakeVideo{}, VideoJobsOptions{TTL: time.Hour})
	j := decodeJob(t, do(t, s, jsonRequest("POST", "/v1/videos", map[string]any{"prompt": "p"})))
	waitStatus(t, s, j.ID, VideoCompleted)
	s.Videos.expire(time.Now().Add(59 * time.Minute))
	if _, ok := s.Videos.Get(j.ID); !ok {
		t.Fatal("expired early")
	}
	s.Videos.expire(time.Now().Add(61 * time.Minute))
	if _, ok := s.Videos.Get(j.ID); ok {
		t.Fatal("never expired")
	}
	if _, err := os.Stat(s.Videos.Dir() + "/" + j.ID + ".mp4"); !os.IsNotExist(err) {
		t.Errorf("the file outlived its job: %v", err)
	}
}

func TestVideoModelGeometry(t *testing.T) {
	s := newVideoServer(t, &fakeVideo{}, VideoJobsOptions{})
	rec := do(t, s, httptest.NewRequest("GET", "/v1/models", nil))
	var resp ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil || len(resp.Data) != 1 ||
		resp.Data[0].Video == nil || resp.Data[0].Video.DefaultSize != "864x480" {
		t.Errorf("models: %v %s", err, rec.Body)
	}
}
