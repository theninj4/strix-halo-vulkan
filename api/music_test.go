package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeMusic plans every request, reports a plan partway, and writes a few
// bytes as its audio. gate holds a job in GenerateMusic.
type fakeMusic struct {
	mu      sync.Mutex
	reqs    []*MusicRequest
	gate    chan struct{}
	started chan string
}

func (f *fakeMusic) Models() []Model {
	return []Model{{ID: "acestep-v15-xl-turbo", Object: "model", OwnedBy: "local"}}
}

func (f *fakeMusic) MusicInfo() MusicInfo {
	return MusicInfo{MinSeconds: 10, MaxSeconds: 600, FallbackSeconds: 120, SampleRate: 48000,
		Formats: []string{"flac", "mp3", "opus", "wav"}, Thinking: true, Steps: 8}
}

func (f *fakeMusic) PlanMusic(req *MusicRequest) (*MusicPlan, error) {
	if req.Duration > 600 {
		return nil, fmt.Errorf("too long: %w", ErrUnsupported)
	}
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	p := &MusicPlan{Seed: 7, LMSeed: 8, Format: "mp3", Steps: 8, Seconds: req.Duration, Estimate: 26 * time.Second}
	if req.Format != "" {
		p.Format = req.Format
	}
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	return p, nil
}

func (f *fakeMusic) GenerateMusic(ctx context.Context, req *MusicRequest, p *MusicPlan, dst string, progress func(MusicProgress)) error {
	if f.started != nil {
		f.started <- req.Caption
	}
	progress(MusicProgress{Stage: "planned", Fraction: 0.3, Seconds: 95, Estimate: 40 * time.Second,
		Plan: &MusicMetadata{Caption: "rewritten", BPM: 128, KeyScale: "F# minor", TimeSignature: "4", Duration: 95, Language: "unknown"}})
	if f.gate != nil {
		select {
		case <-f.gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return os.WriteFile(dst, []byte("not really an mp3"), 0o644)
}

func newMusicServer(t *testing.T, f *fakeMusic) *Server {
	t.Helper()
	q, err := NewMusicJobs(f, MusicJobsOptions{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(q.Close)
	return &Server{Music: q}
}

func decodeMusic(t *testing.T, rec *httptest.ResponseRecorder) MusicJob {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var j MusicJob
	if err := json.Unmarshal(rec.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	return j
}

func waitMusic(t *testing.T, s *Server, id, status string) MusicJob {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		j := decodeMusic(t, do(t, s, httptest.NewRequest("GET", "/v1/music/"+id, nil)))
		if j.Status == status {
			return j
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %s is %s, never %s", id, j.Status, status)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestMusicLifecycle: create with no duration (the LM chooses), poll to
// completed with the plan filled in, fetch, list, delete, 404.
func TestMusicLifecycle(t *testing.T) {
	f := &fakeMusic{}
	s := newMusicServer(t, f)
	j := decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{
		"prompt": "lo-fi hip-hop", "lyrics": "[Instrumental]",
	})))
	if j.Object != "music" || j.Model != "acestep-v15-xl-turbo" || !j.Thinking || j.Seconds != nil ||
		j.Metadata != nil || j.Format != "mp3" || j.Seed != 7 || j.LMSeed != 8 || j.Estimated != 26 {
		t.Errorf("created job %+v", j)
	}
	done := waitMusic(t, s, j.ID, MusicCompleted)
	if done.Progress != 100 || done.Seconds == nil || *done.Seconds != 95 || done.Metadata == nil ||
		done.Metadata.BPM != 128 || done.Metadata.Caption != "rewritten" || done.Estimated != 40 {
		t.Errorf("completed job %+v (metadata %+v)", done, done.Metadata)
	}
	rec := do(t, s, httptest.NewRequest("GET", "/v1/music/"+j.ID+"/content", nil))
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "audio/mpeg" || rec.Body.String() != "not really an mp3" {
		t.Errorf("content: %d %q %q", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	var list MusicList
	rec = do(t, s, httptest.NewRequest("GET", "/v1/music", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil || len(list.Data) != 1 || list.Data[0].ID != j.ID {
		t.Errorf("list: %s", rec.Body)
	}
	if rec := do(t, s, httptest.NewRequest("DELETE", "/v1/music/"+j.ID, nil)); rec.Code != http.StatusOK {
		t.Errorf("delete: %d %s", rec.Code, rec.Body)
	}
	if rec := do(t, s, httptest.NewRequest("GET", "/v1/music/"+j.ID, nil)); rec.Code != http.StatusNotFound {
		t.Errorf("after delete: %d", rec.Code)
	}
}

// TestMusicUpstreamFields: release_task's names, aliases and nested metas,
// as JSON and as a form.
func TestMusicUpstreamFields(t *testing.T) {
	f := &fakeMusic{}
	s := newMusicServer(t, f)
	decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{
		"caption": "synthwave", "lyrics": "[Verse]\nla", "thinking": false,
		"metas":          map[string]any{"bpm": 120, "keyScale": "C major", "timesignature": 4, "duration": "30"},
		"vocal_language": "en", "audio_format": "flac", "seed": 42, "lm_seed": 9, "use_random_seed": false,
		"shift": 2, "timesteps": "0.97,0.76,0.5,0", "lm_temperature": 0.7, "lm_cfg_scale": 2.5, "lm_top_p": 1,
		"inference_steps": 8, "guidance_scale": 7, "task_type": "text2music", "batch_size": 1,
	})))
	form := url.Values{"prompt": {"jazz"}, "audio_duration": {"45.5"}, "key_scale": {"Bb major"},
		"time_signature": {"3"}, "thinking": {"true"}, "timesteps": {"[1, 0.5]"}}
	req := httptest.NewRequest("POST", "/v1/music", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	decodeMusic(t, do(t, s, req))

	a, b := f.reqs[0], f.reqs[1]
	if a.Caption != "synthwave" || a.Thinking || a.BPM != 120 || a.KeyScale != "C major" || a.TimeSignature != "4" ||
		a.Duration != 30 || a.Language != "en" || a.Format != "flac" || a.Seed == nil || *a.Seed != 42 ||
		a.LMSeed == nil || *a.LMSeed != 9 || a.Shift != 2 || len(a.Timesteps) != 4 || a.LMTemperature == nil ||
		*a.LMTemperature != 0.7 || *a.LMCFGScale != 2.5 || *a.LMTopP != 1 {
		t.Errorf("JSON request %+v", a)
	}
	if b.Caption != "jazz" || !b.Thinking || b.Duration != 45.5 || b.KeyScale != "Bb major" || b.TimeSignature != "3" ||
		len(b.Timesteps) != 2 || b.Seed != nil {
		t.Errorf("form request %+v", b)
	}
}

// TestMusicRefusals: what release_task offers and this server does not run
// is a 400 naming the field; the default value of such a field is fine.
func TestMusicRefusals(t *testing.T) {
	s := newMusicServer(t, &fakeMusic{})
	for _, c := range []struct {
		body map[string]any
		want string
	}{
		{map[string]any{"caption": "x", "task_type": "cover"}, "task_type"},
		{map[string]any{"caption": "x", "batch_size": 2}, "batch_size"},
		{map[string]any{"caption": "x", "sample_query": "a love song"}, "sample_query"},
		{map[string]any{"caption": "x", "use_format": true}, "use_format"},
		{map[string]any{"caption": "x", "audio_code_string": "<|audio_code_1|>"}, "audio_code_string"},
		{map[string]any{"caption": "x", "src_audio_path": "/tmp/a.wav"}, "src_audio_path"},
		{map[string]any{"caption": "x", "lm_top_k": 40}, "lm_top_k"},
		{map[string]any{"caption": "x", "use_cot_caption": false}, "use_cot_caption"},
		{map[string]any{"caption": "x", "infer_method": "sde"}, "infer_method"},
		{map[string]any{"lyrics": " "}, "caption"},
		{map[string]any{"caption": "x", "duration": 900}, "too long"},
	} {
		rec := do(t, s, jsonRequest("POST", "/v1/music", c.body))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%v: %d %s", c.body, rec.Code, rec.Body)
		}
	}
	decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{
		"caption": "x", "task_type": "text2music", "batch_size": 1, "lm_top_k": 0, "use_cot_caption": true,
		"lm_negative_prompt": "NO USER INPUT", "infer_method": "ode", "lm_repetition_penalty": 1.0,
	})))
}

// TestMusicCancel: DELETE cancels a running job, and content before
// completion is a 409.
func TestMusicCancel(t *testing.T) {
	f := &fakeMusic{gate: make(chan struct{}), started: make(chan string, 1)}
	s := newMusicServer(t, f)
	j := decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{"caption": "held"})))
	<-f.started
	running := waitMusic(t, s, j.ID, MusicInProgress)
	if running.Stage != "planned" || running.Progress != 30 {
		t.Errorf("running job %+v", running)
	}
	if rec := do(t, s, httptest.NewRequest("GET", "/v1/music/"+j.ID+"/content", nil)); rec.Code != http.StatusConflict {
		t.Errorf("content while running: %d", rec.Code)
	}
	if rec := do(t, s, httptest.NewRequest("DELETE", "/v1/music/"+j.ID, nil)); rec.Code != http.StatusOK {
		t.Fatalf("delete: %d", rec.Code)
	}
	if rec := do(t, s, httptest.NewRequest("GET", "/v1/music/"+j.ID, nil)); rec.Code != http.StatusNotFound {
		t.Errorf("after delete: %d", rec.Code)
	}
}

// TestMusicNotLoaded: without -music every route is a 501 naming the flag.
func TestMusicNotLoaded(t *testing.T) {
	s := &Server{}
	rec := do(t, s, jsonRequest("POST", "/v1/music", map[string]any{"caption": "x"}))
	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "-music") {
		t.Errorf("%d %s", rec.Code, rec.Body)
	}
}

// TestMusicSample: sample mode's fields (MUSIC.md A12) -- a query under any
// of upstream's names, or sample_mode alone -- reach the backend, the job
// reports them, and a caption or lyrics beside them is a 400.
func TestMusicSample(t *testing.T) {
	f := &fakeMusic{}
	s := newMusicServer(t, f)
	j := decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{"sample_query": "a love song"})))
	if !j.Sample || j.SampleQuery != "a love song" {
		t.Errorf("job %+v", j)
	}
	decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{"description": "sea shanty", "duration": 60})))
	decodeMusic(t, do(t, s, jsonRequest("POST", "/v1/music", map[string]any{"sample_mode": true})))
	if len(f.reqs) != 3 {
		t.Fatalf("%d requests", len(f.reqs))
	}
	a, b, c := f.reqs[0], f.reqs[1], f.reqs[2]
	if !a.Sample || a.SampleMode || a.SampleQuery != "a love song" || !a.Thinking {
		t.Errorf("query request %+v", a)
	}
	if !b.Sample || b.SampleQuery != "sea shanty" || b.Duration != 60 {
		t.Errorf("description request %+v", b)
	}
	if !c.Sample || !c.SampleMode || c.SampleQuery != "" {
		t.Errorf("sample_mode request %+v", c)
	}
	for _, body := range []map[string]any{
		{"sample_query": "x", "lyrics": "[Verse]\nla"},
		{"sample_mode": true, "prompt": "jazz"},
		{"sample_mode": "maybe"},
	} {
		rec := do(t, s, jsonRequest("POST", "/v1/music", body))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%v: %d %s", body, rec.Code, rec.Body)
		}
	}
}
