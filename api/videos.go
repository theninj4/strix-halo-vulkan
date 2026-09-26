package api

// OpenAI's asynchronous video API (VIDEO.md M9), which is also what SGLang
// serves MiniMax-H3 behind:
//
//	POST   /v1/videos               -> a job, queued
//	GET    /v1/videos               -> the jobs, newest first
//	GET    /v1/videos/{id}          -> one job's status and progress
//	GET    /v1/videos/{id}/content  -> the mp4, once completed
//	DELETE /v1/videos/{id}          -> cancel it if running, and forget it
//
// **The queue is here and not in the backend.** A video is ~14 minutes at
// the served 480p and ~2 hours at the trained 768p, so no client holds a
// connection open for one; the job, its progress and its file are transport
// state, the same for any model behind the door. The backend is synchronous
// (VideoBackend.GenerateVideo, one mp4 to a path it is handed), and this file
// runs one job at a time through it: the model's stagings hold 50 GB at their
// peak, and two at once would not fit, never mind go faster.
//
// Jobs live in memory. A restart forgets them, and their files with them:
// what a client holds is an id that 404s, which is what an expired job looks
// like too.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// VideoBackend generates one video at a time, synchronously.
type VideoBackend interface {
	Backend
	// PlanVideo validates a request and resolves it -- canvas, frames,
	// steps, seed -- without touching the device. It is called when the job
	// is submitted, so a request that cannot run is a 400 then, and not a
	// failed job twenty minutes later. An error wrapping ErrUnsupported is
	// the client's.
	PlanVideo(req *VideoRequest) (*VideoPlan, error)
	// GenerateVideo runs a planned request and writes an mp4 to dst.
	// progress is called as it moves; ctx cancels it (DELETE, shutdown).
	GenerateVideo(ctx context.Context, req *VideoRequest, plan *VideoPlan, dst string, progress func(VideoProgress)) error
	// VideoGeometry is what the backend accepts, for GET /v1/models.
	VideoGeometry() VideoGeometry
}

// VideoRequest is a video request after the HTTP layer has parsed it, from
// either envelope: OpenAI's (`size`, `seconds`) or SGLang's H3 one
// (`target.short_edge`, `target.aspect_ratio`, `target.duration_seconds`).
type VideoRequest struct {
	Model  string
	Prompt string
	// AspectW:AspectH and ShortEdge are the canvas as H3 asks for one. An
	// OpenAI `size` becomes both: its ratio, and its shorter side. Zero takes
	// the backend's default.
	AspectW, AspectH float64
	ShortEdge        int
	// Seconds is the duration; zero takes the backend's default.
	Seconds float64
	// Steps is the sampling schedule's length; zero takes the default.
	Steps int
	// Seed is the noise seed; nil draws one, and the plan reports it.
	Seed *int64
}

// VideoPlan is what a request resolves to.
type VideoPlan struct {
	Width, Height int
	Frames        int
	Seconds       float64 // Frames at the frame rate: what the clip will actually run
	Steps         int
	Seed          int64
	PromptTokens  int
	// Estimate is the backend's guess at the run's wall time, from measured
	// rates. It is reported so a client can tell a four-minute job from a
	// two-hour one before it starts.
	Estimate time.Duration
}

// VideoProgress is a running job's position: a stage name and the fraction
// of the whole run done.
type VideoProgress struct {
	Stage    string
	Fraction float64
}

// VideoGeometry is reported on the model object in GET /v1/models.
type VideoGeometry struct {
	DefaultSize    string  `json:"default_size"`
	DefaultSeconds float64 `json:"default_seconds"`
	MinSeconds     float64 `json:"min_seconds"`
	MaxSeconds     float64 `json:"max_seconds"`
	MaxShortEdge   int     `json:"max_short_edge"`
	DefaultSteps   int     `json:"default_steps"`
	FPS            int     `json:"fps"`
	SampleRate     int     `json:"audio_sample_rate"`
}

// Video job statuses, as OpenAI names them.
const (
	VideoQueued     = "queued"
	VideoInProgress = "in_progress"
	VideoCompleted  = "completed"
	VideoFailed     = "failed"
)

// VideoJob is OpenAI's video object, plus the extensions a client of this
// model needs: the seed and step count it ran with, the frame count, the
// stage it is in, and an estimate of its wall time.
type VideoJob struct {
	ID          string      `json:"id"`
	Object      string      `json:"object"` // "video"
	Model       string      `json:"model"`
	Status      string      `json:"status"`
	Progress    int         `json:"progress"` // 0-100
	CreatedAt   int64       `json:"created_at"`
	CompletedAt *int64      `json:"completed_at"`
	ExpiresAt   *int64      `json:"expires_at"`
	Prompt      string      `json:"prompt"`
	Size        string      `json:"size"`
	Seconds     string      `json:"seconds"`
	Quality     string      `json:"quality"`
	Error       *VideoError `json:"error"`
	Remixed     *string     `json:"remixed_from_video_id"`

	Seed      int64   `json:"seed"`
	Steps     int     `json:"steps"`
	Frames    int     `json:"frames"`
	Stage     string  `json:"stage,omitempty"`
	Estimated float64 `json:"estimated_seconds"`
}

// VideoError is why a job failed.
type VideoError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// videoJob is a job and what only the queue sees.
type videoJob struct {
	VideoJob
	req    *VideoRequest
	plan   *VideoPlan
	path   string
	cancel context.CancelFunc // set while it runs
	gone   bool               // deleted; the worker skips it or discards its result
}

// VideoJobs is the queue and the store, and what Server.Videos is.
type VideoJobs struct {
	backend VideoBackend
	dir     string
	ownDir  bool
	ttl     time.Duration

	mu    sync.Mutex
	jobs  map[string]*videoJob
	queue chan *videoJob

	ctx    context.Context
	stop   context.CancelFunc
	done   chan struct{}
	closed sync.Once
}

// VideoJobsOptions configures NewVideoJobs.
type VideoJobsOptions struct {
	// Dir is where finished mp4s are kept until they expire. Empty makes a
	// temporary directory, removed on Close.
	Dir string
	// TTL is how long a finished job and its file are kept (0: 24 h).
	TTL time.Duration
	// MaxQueued bounds the jobs waiting behind the running one (0: 16). A
	// full queue is a 429: at 14 minutes a job, sixteen is most of a day.
	MaxQueued int
}

// NewVideoJobs starts the worker over b.
func NewVideoJobs(b VideoBackend, opt VideoJobsOptions) (*VideoJobs, error) {
	if opt.TTL <= 0 {
		opt.TTL = 24 * time.Hour
	}
	if opt.MaxQueued <= 0 {
		opt.MaxQueued = 16
	}
	q := &VideoJobs{backend: b, dir: opt.Dir, ttl: opt.TTL, jobs: map[string]*videoJob{},
		queue: make(chan *videoJob, opt.MaxQueued), done: make(chan struct{})}
	if q.dir == "" {
		d, err := os.MkdirTemp("", "videos-")
		if err != nil {
			return nil, err
		}
		q.dir, q.ownDir = d, true
	} else {
		if err := os.MkdirAll(q.dir, 0o755); err != nil {
			return nil, err
		}
		// A previous process's files: its jobs are gone, so nothing can
		// ever ask for them.
		old, _ := filepath.Glob(filepath.Join(q.dir, "video_*.mp4"))
		for _, f := range old {
			os.Remove(f)
		}
	}
	q.ctx, q.stop = context.WithCancel(context.Background())
	go q.work()
	return q, nil
}

// Close cancels the running job, fails the queued ones, and waits for the
// worker. It must run before the backend is closed.
func (q *VideoJobs) Close() {
	q.closed.Do(func() {
		q.stop()
		<-q.done
		if q.ownDir {
			os.RemoveAll(q.dir)
		}
	})
}

// Dir is where the finished files are.
func (q *VideoJobs) Dir() string { return q.dir }

func newVideoID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return "video_" + hex.EncodeToString(b[:])
}

// ErrQueueFull is a submit past MaxQueued.
var ErrQueueFull = errors.New("the video queue is full")

// Submit plans a request and queues it.
func (q *VideoJobs) Submit(req *VideoRequest) (VideoJob, error) {
	plan, err := q.backend.PlanVideo(req)
	if err != nil {
		return VideoJob{}, err
	}
	model := req.Model
	if ms := q.backend.Models(); len(ms) > 0 {
		model = ms[0].ID
	}
	j := &videoJob{req: req, plan: plan, VideoJob: VideoJob{
		ID: newVideoID(), Object: "video", Model: model, Status: VideoQueued,
		CreatedAt: time.Now().Unix(), Prompt: req.Prompt,
		Size:    formatSize(plan.Width, plan.Height),
		Seconds: strconv.FormatFloat(plan.Seconds, 'f', 2, 64), Quality: "standard",
		Seed: plan.Seed, Steps: plan.Steps, Frames: plan.Frames,
		Estimated: plan.Estimate.Round(time.Second).Seconds(),
	}}
	j.path = filepath.Join(q.dir, j.ID+".mp4")
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.ctx.Err() != nil {
		return VideoJob{}, fmt.Errorf("the server is shutting down")
	}
	select {
	case q.queue <- j:
	default:
		return VideoJob{}, ErrQueueFull
	}
	q.jobs[j.ID] = j
	return j.VideoJob, nil
}

// Get is one job's current state.
func (q *VideoJobs) Get(id string) (VideoJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return VideoJob{}, false
	}
	return j.VideoJob, true
}

// List is every job, oldest first.
func (q *VideoJobs) List() []VideoJob {
	q.mu.Lock()
	out := make([]VideoJob, 0, len(q.jobs))
	for _, j := range q.jobs {
		out = append(out, j.VideoJob)
	}
	q.mu.Unlock()
	sort.Slice(out, func(a, b int) bool {
		if out[a].CreatedAt != out[b].CreatedAt {
			return out[a].CreatedAt < out[b].CreatedAt
		}
		return out[a].ID < out[b].ID
	})
	return out
}

// content is a completed job's file.
func (q *VideoJobs) content(id string) (path string, job VideoJob, ok bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return "", VideoJob{}, false
	}
	return j.path, j.VideoJob, true
}

// Delete forgets a job: a queued one never runs, a running one is
// cancelled, and a finished one's file is removed.
func (q *VideoJobs) Delete(id string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return false
	}
	delete(q.jobs, id)
	j.gone = true
	if j.cancel != nil {
		j.cancel()
	}
	if j.Status == VideoCompleted || j.Status == VideoFailed {
		os.Remove(j.path)
	}
	return true
}

// work runs the queue one job at a time until Close, and expires finished
// jobs as it goes.
func (q *VideoJobs) work() {
	defer close(q.done)
	tick := time.NewTicker(time.Minute)
	defer tick.Stop()
	for {
		select {
		case <-q.ctx.Done():
			q.drain()
			return
		case <-tick.C:
			q.expire(time.Now())
		case j := <-q.queue:
			q.run(j)
		}
	}
}

// drain fails whatever is still queued at shutdown.
func (q *VideoJobs) drain() {
	for {
		select {
		case j := <-q.queue:
			q.mu.Lock()
			q.finish(j, &VideoError{Code: "server_shutdown", Message: "the server shut down before this job ran"})
			q.mu.Unlock()
		default:
			return
		}
	}
}

func (q *VideoJobs) expire(now time.Time) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for id, j := range q.jobs {
		if j.ExpiresAt != nil && now.Unix() >= *j.ExpiresAt {
			delete(q.jobs, id)
			os.Remove(j.path)
		}
	}
}

// finish marks a job done, failed when e is set. q.mu is held.
func (q *VideoJobs) finish(j *videoJob, e *VideoError) {
	now := time.Now()
	done, exp := now.Unix(), now.Add(q.ttl).Unix()
	j.CompletedAt, j.ExpiresAt, j.Stage = &done, &exp, ""
	j.cancel = nil
	if e != nil {
		j.Status, j.Error = VideoFailed, e
		os.Remove(j.path)
		return
	}
	j.Status, j.Progress = VideoCompleted, 100
}

func (q *VideoJobs) run(j *videoJob) {
	ctx, cancel := context.WithCancel(q.ctx)
	defer cancel()
	q.mu.Lock()
	if j.gone {
		q.mu.Unlock()
		return
	}
	j.Status, j.cancel = VideoInProgress, cancel
	q.mu.Unlock()

	start := time.Now()
	err := q.backend.GenerateVideo(ctx, j.req, j.plan, j.path, func(p VideoProgress) {
		q.mu.Lock()
		defer q.mu.Unlock()
		j.Stage = p.Stage
		// 100 is for a job whose file is there to fetch.
		j.Progress = min(99, max(j.Progress, int(p.Fraction*100)))
	})

	q.mu.Lock()
	defer q.mu.Unlock()
	switch {
	case j.gone:
		os.Remove(j.path)
		logf(ctx, "video %s: deleted after %v", j.ID, time.Since(start).Round(time.Second))
	case err != nil && q.ctx.Err() != nil:
		q.finish(j, &VideoError{Code: "server_shutdown", Message: "the server shut down while this job ran"})
	case err != nil:
		code := "generation_failed"
		if errors.Is(err, ErrUnsupported) {
			code = "invalid_request"
		}
		logf(ctx, "video %s: failed after %v: %v", j.ID, time.Since(start).Round(time.Second), err)
		q.finish(j, &VideoError{Code: code, Message: err.Error()})
	default:
		logf(ctx, "video %s: %s, %s s, %d steps in %v", j.ID, j.Size, j.Seconds, j.Steps, time.Since(start).Round(time.Second))
		q.finish(j, nil)
	}
}

// VideoCreateRequest is POST /v1/videos's body: OpenAI's fields, SGLang's H3
// fields, and this server's extensions (steps, seed).
type VideoCreateRequest struct {
	Model  string `json:"model,omitempty"`
	Prompt string `json:"prompt"`
	// Size is OpenAI's "WIDTHxHEIGHT". It is read as an aspect ratio and a
	// short edge, and the canvas is the model's for that pair, snapped to
	// its 32-pixel grid: 1280x720 runs as 1280x704, and the job says so.
	Size string `json:"size,omitempty"`
	// Seconds is OpenAI's duration, which it sends as a string ("8").
	Seconds flexFloat `json:"seconds,omitempty"`
	// InputReference is OpenAI's keyframe; it arrives as a multipart file.
	// Task and Conditions are SGLang's: "t2va", or "fl2va" with keyframe
	// conditions. Keyframes are VIDEO.md M10 and not served yet.
	Task       string            `json:"task,omitempty"`
	Conditions []json.RawMessage `json:"conditions,omitempty"`
	Target     *struct {
		ShortEdge   int       `json:"short_edge,omitempty"`
		AspectRatio string    `json:"aspect_ratio,omitempty"`
		Duration    flexFloat `json:"duration_seconds,omitempty"`
	} `json:"target,omitempty"`
	// AspectRatio and ShortEdge are the same as Target's, at the top level.
	AspectRatio string `json:"aspect_ratio,omitempty"`
	ShortEdge   int    `json:"short_edge,omitempty"`
	Steps       int    `json:"steps,omitempty"`
	NumSteps    int    `json:"num_inference_steps,omitempty"`
	Seed        *int64 `json:"seed,omitempty"`
}

// flexFloat is a number sent as a number or as a string.
type flexFloat float64

func (f *flexFloat) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), `"`)
	if s == "" || s == "null" {
		*f = 0
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fmt.Errorf("%q is not a number", s)
	}
	*f = flexFloat(v)
	return nil
}

// parseAspect reads "16:9"; "" and "auto" are zero (the default).
func parseAspect(s string) (w, h float64, err error) {
	if s == "" || s == "auto" {
		return 0, 0, nil
	}
	a, b, ok := strings.Cut(s, ":")
	if ok {
		w, err = strconv.ParseFloat(a, 64)
		if err == nil {
			h, err = strconv.ParseFloat(b, 64)
		}
	}
	if !ok || err != nil || w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("aspect_ratio %q is not W:H", s)
	}
	return w, h, nil
}

// toRequest turns the envelope into a VideoRequest.
func (c *VideoCreateRequest) toRequest() (*VideoRequest, error) {
	if strings.TrimSpace(c.Prompt) == "" {
		return nil, errors.New("prompt is required")
	}
	switch c.Task {
	case "", "t2va":
	case "fl2va", "i2va", "ref2va":
		return nil, fmt.Errorf("task %q (keyframes and references) is not served yet; only t2va is", c.Task)
	default:
		return nil, fmt.Errorf("unknown task %q", c.Task)
	}
	if len(c.Conditions) > 0 {
		return nil, errors.New("conditions (keyframes) are not served yet; only t2va is")
	}
	r := &VideoRequest{Model: c.Model, Prompt: c.Prompt, Seconds: float64(c.Seconds), Seed: c.Seed,
		ShortEdge: c.ShortEdge, Steps: c.Steps}
	if r.Steps == 0 {
		r.Steps = c.NumSteps
	}
	aspect := c.AspectRatio
	if c.Target != nil {
		if c.Target.ShortEdge != 0 {
			r.ShortEdge = c.Target.ShortEdge
		}
		if c.Target.AspectRatio != "" {
			aspect = c.Target.AspectRatio
		}
		if c.Target.Duration != 0 {
			r.Seconds = float64(c.Target.Duration)
		}
	}
	var err error
	if r.AspectW, r.AspectH, err = parseAspect(aspect); err != nil {
		return nil, err
	}
	if c.Size != "" && c.Size != "auto" {
		w, h, err := parseSize(c.Size)
		if err != nil {
			return nil, err
		}
		r.AspectW, r.AspectH, r.ShortEdge = float64(w), float64(h), min(w, h)
	}
	if r.Seconds < 0 || r.Steps < 0 || r.ShortEdge < 0 {
		return nil, errors.New("seconds, steps and short_edge must not be negative")
	}
	return r, nil
}

// readVideoCreate reads the body as JSON or, as OpenAI's SDK sends it, as a
// multipart form.
func readVideoCreate(ctx context.Context, w http.ResponseWriter, r *http.Request) (*VideoCreateRequest, bool) {
	var c VideoCreateRequest
	mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if mt != "multipart/form-data" && mt != "application/x-www-form-urlencoded" {
		return &c, decodeJSON(ctx, w, r, &c)
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil && !errors.Is(err, http.ErrNotMultipart) {
		if tooLarge(ctx, w, err) {
			return nil, false
		}
		badRequest(ctx, w, "malformed form: "+err.Error())
		return nil, false
	}
	if r.MultipartForm != nil && len(r.MultipartForm.File["input_reference"]) > 0 {
		badRequest(ctx, w, "input_reference (a keyframe) is not served yet; only t2va is")
		return nil, false
	}
	c.Model, c.Prompt, c.Size = r.FormValue("model"), r.FormValue("prompt"), r.FormValue("size")
	c.Task, c.AspectRatio = r.FormValue("task"), r.FormValue("aspect_ratio")
	for _, f := range []struct {
		name string
		dst  any
	}{{"seconds", &c.Seconds}, {"short_edge", &c.ShortEdge}, {"steps", &c.Steps},
		{"num_inference_steps", &c.NumSteps}, {"seed", &c.Seed}} {
		v := r.FormValue(f.name)
		if v == "" {
			continue
		}
		if err := json.Unmarshal([]byte(v), f.dst); err != nil {
			badRequest(ctx, w, fmt.Sprintf("%s: %v", f.name, err))
			return nil, false
		}
	}
	return &c, true
}

func (s *Server) handleVideoCreate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Videos == nil {
		notLoaded(ctx, w, "the video model", "-video")
		return
	}
	c, ok := readVideoCreate(ctx, w, r)
	if !ok {
		return
	}
	req, err := c.toRequest()
	if err != nil {
		badRequest(ctx, w, err.Error())
		return
	}
	job, err := s.Videos.Submit(req)
	switch {
	case errors.Is(err, ErrQueueFull):
		logf(ctx, "429: %v", err)
		writeError(w, http.StatusTooManyRequests, "rate_limit_error", err.Error())
	case err != nil:
		backendError(ctx, w, "video", err)
	default:
		logf(ctx, "video %s queued: %s, %s s, %d steps, seed %d, ~%v", job.ID, job.Size, job.Seconds,
			job.Steps, job.Seed, time.Duration(job.Estimated)*time.Second)
		writeJSON(w, http.StatusOK, job)
	}
}

func (s *Server) videoJob(w http.ResponseWriter, r *http.Request) (VideoJob, bool) {
	ctx := r.Context()
	if s.Videos == nil {
		notLoaded(ctx, w, "the video model", "-video")
		return VideoJob{}, false
	}
	job, ok := s.Videos.Get(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no video "+r.PathValue("id"))
	}
	return job, ok
}

func (s *Server) handleVideoGet(w http.ResponseWriter, r *http.Request) {
	if job, ok := s.videoJob(w, r); ok {
		writeJSON(w, http.StatusOK, job)
	}
}

// VideoList is GET /v1/videos's page.
type VideoList struct {
	Object  string     `json:"object"` // "list"
	Data    []VideoJob `json:"data"`
	FirstID *string    `json:"first_id"`
	LastID  *string    `json:"last_id"`
	HasMore bool       `json:"has_more"`
}

func (s *Server) handleVideoList(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Videos == nil {
		notLoaded(ctx, w, "the video model", "-video")
		return
	}
	qv := r.URL.Query()
	limit := 20
	if v := qv.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 100 {
			badRequest(ctx, w, "limit must be 1-100")
			return
		}
		limit = n
	}
	jobs := s.Videos.List()
	switch qv.Get("order") {
	case "", "desc":
		for i, j := 0, len(jobs)-1; i < j; i, j = i+1, j-1 {
			jobs[i], jobs[j] = jobs[j], jobs[i]
		}
	case "asc":
	default:
		badRequest(ctx, w, "order must be asc or desc")
		return
	}
	if after := qv.Get("after"); after != "" {
		for i := range jobs {
			if jobs[i].ID == after {
				jobs = jobs[i+1:]
				break
			}
		}
	}
	out := VideoList{Object: "list", Data: jobs}
	if len(jobs) > limit {
		out.Data, out.HasMore = jobs[:limit], true
	}
	if n := len(out.Data); n > 0 {
		out.FirstID, out.LastID = &out.Data[0].ID, &out.Data[n-1].ID
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleVideoContent(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Videos == nil {
		notLoaded(ctx, w, "the video model", "-video")
		return
	}
	if v := r.URL.Query().Get("variant"); v != "" && v != "video" {
		badRequest(ctx, w, fmt.Sprintf("variant %q is not served; only video", v))
		return
	}
	path, job, ok := s.Videos.content(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no video "+r.PathValue("id"))
		return
	}
	if job.Status != VideoCompleted {
		writeError(w, http.StatusConflict, "invalid_request_error",
			fmt.Sprintf("video %s is %s, not completed", job.ID, job.Status))
		return
	}
	f, err := os.Open(path)
	if err != nil {
		// Deleted or expired between the lookup and the open.
		writeError(w, http.StatusNotFound, "invalid_request_error", "no video "+job.ID)
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		serverError(ctx, w, "video content", err)
		return
	}
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Disposition", `inline; filename="`+job.ID+`.mp4"`)
	http.ServeContent(w, r, "", st.ModTime(), f)
}

func (s *Server) handleVideoDelete(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Videos == nil {
		notLoaded(ctx, w, "the video model", "-video")
		return
	}
	id := r.PathValue("id")
	if !s.Videos.Delete(id) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no video "+id)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "video.deleted", "deleted": true})
}
