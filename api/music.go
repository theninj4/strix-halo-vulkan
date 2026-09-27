package api

// Music generation as asynchronous jobs (MUSIC.md A9):
//
//	POST   /v1/music               -> a job, queued
//	GET    /v1/music               -> the jobs, newest first
//	GET    /v1/music/{id}          -> one job's status, progress and plan
//	GET    /v1/music/{id}/content  -> the audio, once completed
//	DELETE /v1/music/{id}          -> cancel it if running, and forget it
//
// OpenAI has no music endpoint, so this is the shape of the one it has for
// a generator that runs longer than a connection should wait: /v1/videos
// (videos.go), job for job. A song is tens of seconds rather than minutes,
// but most of that is the 5 Hz LM planning it one token at a time, and the
// plan (the metadata it chose, the song's length) is worth showing a client
// before the audio exists.
//
// The request reads ACE-Step's own API (`POST /release_task`, docs/en/API.md
// upstream): its field names and their aliases, flat or in a nested
// `metas`/`metadata`/`user_metadata` object, as JSON or a form. What that API
// offers and this server does not run -- other tasks, reference audio,
// batches, format mode -- is a 400 naming the field, never ignored.
//
// Sample mode (`sample_query`, or `sample_mode` alone for a song of the
// LM's choosing; MUSIC.md A12) has the LM write the caption, metas and
// lyrics from a description first. A caption or lyrics sent with it would
// be discarded, as upstream discards them, so they are a 400 instead.
//
// Jobs live in memory, as the video ones do: a restart forgets them and
// their files.

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

// MusicBackend generates one song at a time, synchronously.
type MusicBackend interface {
	Backend
	// PlanMusic validates a request and resolves what can be resolved
	// before it runs -- seeds, format, the length when it was given -- with
	// no device work. An error wrapping ErrUnsupported is the client's.
	PlanMusic(req *MusicRequest) (*MusicPlan, error)
	// GenerateMusic runs a planned request and writes the audio to dst.
	// progress is called as it moves; ctx cancels it (DELETE, shutdown).
	GenerateMusic(ctx context.Context, req *MusicRequest, plan *MusicPlan, dst string, progress func(MusicProgress)) error
	// MusicInfo is what the backend accepts, for GET /v1/models.
	MusicInfo() MusicInfo
}

// MusicRequest is a music request after the HTTP layer has parsed it.
type MusicRequest struct {
	Model   string
	Caption string
	Lyrics  string
	// Duration is the song's length in seconds; zero lets the LM choose it
	// (thinking) or takes upstream's 120 s (not).
	Duration float64
	// The metas: zero values are unset, and the LM fills them.
	BPM           int
	KeyScale      string
	TimeSignature string
	Language      string
	// Thinking runs the 5 Hz LM before the DiT: upstream's default path.
	Thinking bool
	// Sample has the LM write the song from SampleQuery first (sample mode):
	// upstream's sample_mode flag, or a query given. SampleMode is the flag
	// itself, which also keeps phase 1 from filling a meta the song lacks.
	Sample, SampleMode bool
	SampleQuery        string
	// Seed is the DiT's noise seed and LMSeed the LM's sampling seed; nil
	// draws one, and the plan reports it.
	Seed, LMSeed *int64
	// Format is the container: mp3, wav, flac or opus ("" is mp3).
	Format string
	// Shift and Timesteps are the DiT's schedule (zero and nil: shift 3).
	Shift     float64
	Timesteps []float64
	// The LM's sampling, nil for upstream's (0.85, 2.0, 0.9).
	LMTemperature, LMCFGScale, LMTopP *float64
}

// MusicPlan is what a request resolves to at submit time.
type MusicPlan struct {
	Seed, LMSeed int64
	Format       string
	// Seconds is the song's length when it is known before the LM runs,
	// and zero when the LM is to choose it.
	Seconds  float64
	Steps    int
	Estimate time.Duration
}

// MusicProgress is a running job's position. Plan is set once the LM has
// planned the song (or at once, without thinking), with the length and the
// estimate that follow from it.
type MusicProgress struct {
	Stage    string
	Fraction float64
	Plan     *MusicMetadata
	Seconds  float64
	Estimate time.Duration
}

// MusicMetadata is what the DiT was asked for: the request's metas where
// it gave them and the LM's where it did not, as upstream reports `metas`.
type MusicMetadata struct {
	Caption string `json:"caption"`
	// Lyrics and Genres are what sample mode wrote (empty otherwise: the
	// lyrics are the request's own).
	Lyrics        string  `json:"lyrics,omitempty"`
	Genres        string  `json:"genres,omitempty"`
	BPM           int     `json:"bpm,omitempty"`
	KeyScale      string  `json:"keyscale,omitempty"`
	TimeSignature string  `json:"timesignature,omitempty"`
	Duration      float64 `json:"duration"`
	Language      string  `json:"vocal_language"`
}

// MusicInfo is reported on the model object in GET /v1/models.
type MusicInfo struct {
	MinSeconds      float64  `json:"min_seconds"`
	MaxSeconds      float64  `json:"max_seconds"`
	FallbackSeconds float64  `json:"fallback_seconds"` // no duration and no thinking
	SampleRate      int      `json:"sample_rate"`
	Formats         []string `json:"formats"`
	Thinking        bool     `json:"thinking"` // the LM is loaded
	Steps           int      `json:"steps"`
}

// Music job statuses: the video ones.
const (
	MusicQueued     = VideoQueued
	MusicInProgress = VideoInProgress
	MusicCompleted  = VideoCompleted
	MusicFailed     = VideoFailed
)

// MusicJob is a music job as a client sees it.
type MusicJob struct {
	ID          string      `json:"id"`
	Object      string      `json:"object"` // "music"
	Model       string      `json:"model"`
	Status      string      `json:"status"`
	Progress    int         `json:"progress"` // 0-100
	CreatedAt   int64       `json:"created_at"`
	CompletedAt *int64      `json:"completed_at"`
	ExpiresAt   *int64      `json:"expires_at"`
	Error       *VideoError `json:"error"`

	Caption  string `json:"caption"`
	Lyrics   string `json:"lyrics"`
	Thinking bool   `json:"thinking"`
	// SampleQuery is sample mode's description; Sample says it ran (a
	// query, or sample_mode with none).
	Sample      bool   `json:"sample_mode,omitempty"`
	SampleQuery string `json:"sample_query,omitempty"`
	Format      string `json:"format"`
	// Seconds is the song's length: null until known, which with thinking
	// and no duration is when the LM has planned it.
	Seconds *float64 `json:"seconds"`
	// Metadata is what the DiT is asked for, once planned.
	Metadata  *MusicMetadata `json:"metadata"`
	Seed      int64          `json:"seed"`
	LMSeed    int64          `json:"lm_seed"`
	Steps     int            `json:"steps"`
	Stage     string         `json:"stage,omitempty"`
	Estimated float64        `json:"estimated_seconds"`
}

type musicJob struct {
	MusicJob
	req    *MusicRequest
	plan   *MusicPlan
	path   string
	cancel context.CancelFunc
	gone   bool
}

// MusicJobs is the queue and the store, and what Server.Music is.
type MusicJobs struct {
	backend MusicBackend
	dir     string
	ownDir  bool
	ttl     time.Duration

	mu    sync.Mutex
	jobs  map[string]*musicJob
	queue chan *musicJob

	ctx    context.Context
	stop   context.CancelFunc
	done   chan struct{}
	closed sync.Once
}

// MusicJobsOptions configures NewMusicJobs.
type MusicJobsOptions struct {
	// Dir is where finished songs are kept until they expire. Empty makes a
	// temporary directory, removed on Close.
	Dir string
	// TTL is how long a finished job and its file are kept (0: 24 h).
	TTL time.Duration
	// MaxQueued bounds the jobs waiting behind the running one (0: 32).
	MaxQueued int
}

// NewMusicJobs starts the worker over b.
func NewMusicJobs(b MusicBackend, opt MusicJobsOptions) (*MusicJobs, error) {
	if opt.TTL <= 0 {
		opt.TTL = 24 * time.Hour
	}
	if opt.MaxQueued <= 0 {
		opt.MaxQueued = 32
	}
	q := &MusicJobs{backend: b, dir: opt.Dir, ttl: opt.TTL, jobs: map[string]*musicJob{},
		queue: make(chan *musicJob, opt.MaxQueued), done: make(chan struct{})}
	if q.dir == "" {
		d, err := os.MkdirTemp("", "music-")
		if err != nil {
			return nil, err
		}
		q.dir, q.ownDir = d, true
	} else {
		if err := os.MkdirAll(q.dir, 0o755); err != nil {
			return nil, err
		}
		old, _ := filepath.Glob(filepath.Join(q.dir, "music_*"))
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
func (q *MusicJobs) Close() {
	q.closed.Do(func() {
		q.stop()
		<-q.done
		if q.ownDir {
			os.RemoveAll(q.dir)
		}
	})
}

// Dir is where the finished files are.
func (q *MusicJobs) Dir() string { return q.dir }

func newMusicID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return "music_" + hex.EncodeToString(b[:])
}

// ErrMusicQueueFull is a submit past MaxQueued.
var ErrMusicQueueFull = errors.New("the music queue is full")

// Submit plans a request and queues it.
func (q *MusicJobs) Submit(req *MusicRequest) (MusicJob, error) {
	plan, err := q.backend.PlanMusic(req)
	if err != nil {
		return MusicJob{}, err
	}
	model := req.Model
	if ms := q.backend.Models(); len(ms) > 0 {
		model = ms[0].ID
	}
	j := &musicJob{req: req, plan: plan, MusicJob: MusicJob{
		ID: newMusicID(), Object: "music", Model: model, Status: MusicQueued,
		CreatedAt: time.Now().Unix(), Caption: req.Caption, Lyrics: req.Lyrics, Thinking: req.Thinking,
		Sample: req.Sample, SampleQuery: req.SampleQuery,
		Format: plan.Format, Seed: plan.Seed, LMSeed: plan.LMSeed, Steps: plan.Steps,
		Estimated: plan.Estimate.Round(time.Second).Seconds(),
	}}
	if plan.Seconds > 0 {
		s := plan.Seconds
		j.Seconds = &s
	}
	j.path = filepath.Join(q.dir, j.ID+"."+plan.Format)
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.ctx.Err() != nil {
		return MusicJob{}, fmt.Errorf("the server is shutting down")
	}
	select {
	case q.queue <- j:
	default:
		return MusicJob{}, ErrMusicQueueFull
	}
	q.jobs[j.ID] = j
	return j.MusicJob, nil
}

// Get is one job's current state.
func (q *MusicJobs) Get(id string) (MusicJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return MusicJob{}, false
	}
	return j.MusicJob, true
}

// List is every job, oldest first.
func (q *MusicJobs) List() []MusicJob {
	q.mu.Lock()
	out := make([]MusicJob, 0, len(q.jobs))
	for _, j := range q.jobs {
		out = append(out, j.MusicJob)
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

func (q *MusicJobs) content(id string) (string, MusicJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	j, ok := q.jobs[id]
	if !ok {
		return "", MusicJob{}, false
	}
	return j.path, j.MusicJob, true
}

// Delete forgets a job: a queued one never runs, a running one is
// cancelled, and a finished one's file is removed.
func (q *MusicJobs) Delete(id string) bool {
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
	if j.Status == MusicCompleted || j.Status == MusicFailed {
		os.Remove(j.path)
	}
	return true
}

func (q *MusicJobs) work() {
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

func (q *MusicJobs) drain() {
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

func (q *MusicJobs) expire(now time.Time) {
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
func (q *MusicJobs) finish(j *musicJob, e *VideoError) {
	now := time.Now()
	done, exp := now.Unix(), now.Add(q.ttl).Unix()
	j.CompletedAt, j.ExpiresAt, j.Stage = &done, &exp, ""
	j.cancel = nil
	if e != nil {
		j.Status, j.Error = MusicFailed, e
		os.Remove(j.path)
		return
	}
	j.Status, j.Progress = MusicCompleted, 100
}

func (q *MusicJobs) run(j *musicJob) {
	ctx, cancel := context.WithCancel(q.ctx)
	defer cancel()
	q.mu.Lock()
	if j.gone {
		q.mu.Unlock()
		return
	}
	j.Status, j.cancel = MusicInProgress, cancel
	q.mu.Unlock()

	start := time.Now()
	err := q.backend.GenerateMusic(ctx, j.req, j.plan, j.path, func(p MusicProgress) {
		q.mu.Lock()
		defer q.mu.Unlock()
		j.Stage = p.Stage
		j.Progress = min(99, max(j.Progress, int(p.Fraction*100)))
		if p.Plan != nil {
			m := *p.Plan
			j.Metadata = &m
		}
		if p.Seconds > 0 {
			s := p.Seconds
			j.Seconds = &s
		}
		if p.Estimate > 0 {
			j.Estimated = p.Estimate.Round(time.Second).Seconds()
		}
	})

	q.mu.Lock()
	defer q.mu.Unlock()
	switch {
	case j.gone:
		os.Remove(j.path)
		logf(ctx, "music %s: deleted after %v", j.ID, time.Since(start).Round(time.Millisecond))
	case err != nil && q.ctx.Err() != nil:
		q.finish(j, &VideoError{Code: "server_shutdown", Message: "the server shut down while this job ran"})
	case err != nil:
		code := "generation_failed"
		if errors.Is(err, ErrUnsupported) {
			code = "invalid_request"
		}
		logf(ctx, "music %s: failed after %v: %v", j.ID, time.Since(start).Round(time.Millisecond), err)
		q.finish(j, &VideoError{Code: code, Message: err.Error()})
	default:
		secs := 0.0
		if j.Seconds != nil {
			secs = *j.Seconds
		}
		logf(ctx, "music %s: %.1f s of %s in %v", j.ID, secs, j.Format, time.Since(start).Round(time.Millisecond))
		q.finish(j, nil)
	}
}

// ---- The create request: ACE-Step's release_task fields.

// musicFields is a create request's fields, flat, as raw JSON values: a
// JSON body's top level with any nested metas object folded in beneath it,
// or a form's values (a value that is not JSON is a string).
type musicFields map[string]json.RawMessage

// musicNested are the objects upstream reads metas out of.
var musicNested = []string{"metas", "metadata", "user_metadata"}

func (f musicFields) get(names ...string) (json.RawMessage, string, bool) {
	for _, n := range names {
		if v, ok := f[n]; ok && string(v) != "null" {
			return v, n, true
		}
	}
	return nil, "", false
}

func (f musicFields) str(names ...string) (string, error) {
	v, n, ok := f.get(names...)
	if !ok {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		// A number or a boolean where a string was meant
		// ("time_signature": 4, or a form's caption that parses as one).
		if t := strings.TrimSpace(string(v)); t != "" && t[0] != '{' && t[0] != '[' {
			return t, nil
		}
		return "", fmt.Errorf("%s: want a string", n)
	}
	return s, nil
}

func (f musicFields) num(names ...string) (float64, bool, error) {
	v, n, ok := f.get(names...)
	if !ok {
		return 0, false, nil
	}
	var x flexFloat
	if err := json.Unmarshal(v, &x); err != nil {
		return 0, false, fmt.Errorf("%s: %v", n, err)
	}
	return float64(x), true, nil
}

func (f musicFields) boolean(names ...string) (bool, bool, error) {
	v, n, ok := f.get(names...)
	if !ok {
		return false, false, nil
	}
	var b bool
	if err := json.Unmarshal(v, &b); err != nil {
		var s string
		if json.Unmarshal(v, &s) == nil {
			if b, err = strconv.ParseBool(s); err == nil {
				return b, true, nil
			}
		}
		return false, false, fmt.Errorf("%s: want true or false", n)
	}
	return b, true, nil
}

// readMusicFields reads a JSON body or a form into fields.
func readMusicFields(ctx context.Context, w http.ResponseWriter, r *http.Request) (musicFields, bool) {
	mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	f := musicFields{}
	if mt != "multipart/form-data" && mt != "application/x-www-form-urlencoded" {
		var top map[string]json.RawMessage
		if !decodeJSON(ctx, w, r, &top) {
			return nil, false
		}
		for _, n := range musicNested {
			var inner map[string]json.RawMessage
			if v, ok := top[n]; ok && json.Unmarshal(v, &inner) == nil {
				for k, v := range inner {
					f[k] = v
				}
			}
		}
		for k, v := range top {
			f[k] = v
		}
		return f, true
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil && !errors.Is(err, http.ErrNotMultipart) {
		if tooLarge(ctx, w, err) {
			return nil, false
		}
		badRequest(ctx, w, "malformed form: "+err.Error())
		return nil, false
	}
	if r.MultipartForm != nil {
		for name := range r.MultipartForm.File {
			badRequest(ctx, w, fmt.Sprintf("file %q: this server takes no audio files; text2music is served", name))
			return nil, false
		}
	}
	for k, vs := range r.Form {
		if len(vs) == 0 {
			continue
		}
		v := vs[len(vs)-1]
		// A form value is JSON when it parses as a scalar (120, true) or is
		// the timesteps list; anything else is its text.
		raw := json.RawMessage(v)
		isList := v != "" && (v[0] == '{' || v[0] == '[')
		if !json.Valid(raw) || (isList && k != "timesteps") {
			raw, _ = json.Marshal(v)
		}
		f[k] = raw
	}
	return f, true
}

// musicRefused are release_task fields this server does not run, with why.
// A field set to its upstream default is fine; anything else is a 400.
var musicRefused = []struct {
	names []string
	ok    func(v json.RawMessage) bool
	why   string
}{
	{[]string{"task_type", "taskType"}, isOneOf("text2music"), "only text2music is served (cover, repaint, lego, extract and complete are not)"},
	{[]string{"batch_size", "batchSize"}, isOneOf("1"), "one song a job; submit one job per song"},
	{[]string{"use_format", "useFormat", "format"}, isOneOf("false"), "format mode is not served"},
	{[]string{"audio_code_string", "audioCodeString"}, isOneOf(""), "audio codes in are not served; thinking writes them"},
	{[]string{"reference_audio_path", "src_audio_path", "reference_audio", "src_audio"}, isOneOf(""), "no reference or source audio: text2music is served"},
	{[]string{"infer_method", "inferMethod"}, isOneOf("ode"), "only the ODE (Euler) sampler is served"},
	{[]string{"lm_top_k", "lmTopK"}, isOneOf("0"), "top-k is not served (upstream's default is off)"},
	{[]string{"lm_repetition_penalty", "lmRepetitionPenalty"}, isOneOf("1"), "a repetition penalty is not served (upstream's default is 1)"},
	{[]string{"lm_negative_prompt", "lmNegativePrompt"}, isOneOf("", "NO USER INPUT"), "a negative prompt is not served"},
	{[]string{"use_cot_caption", "cot_caption", "cot-caption"}, isOneOf("true"), "the LM always rewrites the caption, as upstream's default does"},
	{[]string{"use_cot_language", "cot_language", "cot-language"}, isOneOf("true"), "the LM always writes the language, as upstream's default does"},
	{[]string{"use_cot_metas"}, isOneOf("true"), "the LM always fills the missing metas, as upstream's default does"},
	{[]string{"constrained_decoding", "constrainedDecoding", "constrained"}, isOneOf("true"), "the LM always decodes under its FSM"},
	{[]string{"use_adg"}, isOneOf("false"), "ADG is for the base model; this is turbo"},
	{[]string{"lora", "lora_path", "lm_model_path"}, isOneOf(""), "the models are fixed at startup"},
}

// isOneOf matches a raw field against defaults, comparing numbers as
// numbers, booleans and strings as their text.
func isOneOf(vals ...string) func(json.RawMessage) bool {
	return func(v json.RawMessage) bool {
		var s string
		if json.Unmarshal(v, &s) != nil {
			s = string(v)
		}
		for _, want := range vals {
			if s == want {
				return true
			}
			a, e1 := strconv.ParseFloat(s, 64)
			b, e2 := strconv.ParseFloat(want, 64)
			if e1 == nil && e2 == nil && a == b {
				return true
			}
		}
		return false
	}
}

// toMusicRequest reads the fields into a request.
func (f musicFields) toMusicRequest() (*MusicRequest, error) {
	for _, r := range musicRefused {
		if v, n, ok := f.get(r.names...); ok && !r.ok(v) {
			return nil, fmt.Errorf("%s: %s", n, r.why)
		}
	}
	var err error
	r := &MusicRequest{Thinking: true}
	if r.Model, err = f.str("model"); err != nil {
		return nil, err
	}
	if r.Caption, err = f.str("caption", "prompt"); err != nil {
		return nil, err
	}
	if r.Lyrics, err = f.str("lyrics"); err != nil {
		return nil, err
	}
	if r.SampleQuery, err = f.str("sample_query", "sampleQuery", "description", "desc"); err != nil {
		return nil, err
	}
	if r.SampleMode, _, err = f.boolean("sample_mode", "sampleMode"); err != nil {
		return nil, err
	}
	r.Sample = r.SampleMode || strings.TrimSpace(r.SampleQuery) != ""
	switch {
	case r.Sample && (strings.TrimSpace(r.Caption) != "" || strings.TrimSpace(r.Lyrics) != ""):
		return nil, errors.New("sample mode writes the caption and lyrics itself: send sample_query without them, or them without it")
	case !r.Sample && strings.TrimSpace(r.Caption) == "" && strings.TrimSpace(r.Lyrics) == "":
		return nil, errors.New("caption (or prompt) is required, or lyrics, or a sample_query")
	}
	if b, ok, err := f.boolean("thinking"); err != nil {
		return nil, err
	} else if ok {
		r.Thinking = b
	}
	if r.Duration, _, err = f.num("audio_duration", "duration", "target_duration", "audioDuration"); err != nil {
		return nil, err
	}
	if r.Duration < 0 {
		r.Duration = 0 // upstream's -1: unset
	}
	bpm, _, err := f.num("bpm")
	if err != nil {
		return nil, err
	}
	if bpm < 0 || bpm != float64(int(bpm)) {
		return nil, fmt.Errorf("bpm %v: want a whole number", bpm)
	}
	r.BPM = int(bpm)
	if r.KeyScale, err = f.str("key_scale", "keyscale", "keyScale"); err != nil {
		return nil, err
	}
	if r.TimeSignature, err = f.str("time_signature", "timesignature", "timeSignature"); err != nil {
		return nil, err
	}
	if r.Language, err = f.str("vocal_language", "vocalLanguage", "language"); err != nil {
		return nil, err
	}
	if r.Format, err = f.str("audio_format", "audioFormat", "response_format"); err != nil {
		return nil, err
	}
	// Upstream's seed is used only with use_random_seed false; a seed sent
	// without that flag is taken as meant, and -1 is random.
	random, _, err := f.boolean("use_random_seed", "useRandomSeed")
	if err != nil {
		return nil, err
	}
	for _, s := range []struct {
		names []string
		dst   **int64
	}{{[]string{"seed"}, &r.Seed}, {[]string{"lm_seed", "lmSeed"}, &r.LMSeed}} {
		v, ok, err := f.num(s.names...)
		if err != nil {
			return nil, err
		}
		if ok && v >= 0 && !random {
			x := int64(v)
			*s.dst = &x
		}
	}
	if r.Shift, _, err = f.num("shift"); err != nil {
		return nil, err
	}
	if v, n, ok := f.get("timesteps"); ok {
		var list []float64
		var s string
		switch {
		case json.Unmarshal(v, &list) == nil:
		case json.Unmarshal(v, &s) == nil:
			for _, p := range strings.Split(s, ",") {
				if p = strings.TrimSpace(p); p == "" {
					continue
				}
				x, err := strconv.ParseFloat(p, 64)
				if err != nil {
					return nil, fmt.Errorf("%s: %q is not a number", n, p)
				}
				list = append(list, x)
			}
		default:
			return nil, fmt.Errorf("%s: a list of numbers, or them comma-separated", n)
		}
		for _, t := range list {
			if t < 0 || t > 1 {
				return nil, fmt.Errorf("%s: %v is outside [0, 1]", n, t)
			}
		}
		r.Timesteps = list
	}
	for _, s := range []struct {
		names []string
		dst   **float64
	}{{[]string{"lm_temperature", "lmTemperature"}, &r.LMTemperature},
		{[]string{"lm_cfg_scale", "lmCfgScale"}, &r.LMCFGScale},
		{[]string{"lm_top_p", "lmTopP"}, &r.LMTopP}} {
		v, ok, err := f.num(s.names...)
		if err != nil {
			return nil, err
		}
		if ok {
			x := v
			*s.dst = &x
		}
	}
	// inference_steps and guidance_scale are accepted and do nothing: the
	// turbo model runs its fixed 8-step table without CFG, which is what
	// upstream does with them too (MUSIC.md A1).
	for _, n := range []string{"inference_steps", "guidance_scale"} {
		if _, _, err := f.num(n); err != nil {
			return nil, err
		}
	}
	return r, nil
}

func (s *Server) handleMusicCreate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Music == nil {
		notLoaded(ctx, w, "the music model", "-music")
		return
	}
	f, ok := readMusicFields(ctx, w, r)
	if !ok {
		return
	}
	req, err := f.toMusicRequest()
	if err != nil {
		badRequest(ctx, w, err.Error())
		return
	}
	job, err := s.Music.Submit(req)
	switch {
	case errors.Is(err, ErrMusicQueueFull):
		logf(ctx, "429: %v", err)
		writeError(w, http.StatusTooManyRequests, "rate_limit_error", err.Error())
	case err != nil:
		backendError(ctx, w, "music", err)
	default:
		logf(ctx, "music %s queued: thinking %v, sample %v, %s, seed %d, ~%v", job.ID, job.Thinking, job.Sample,
			job.Format, job.Seed, time.Duration(job.Estimated)*time.Second)
		writeJSON(w, http.StatusOK, job)
	}
}

func (s *Server) handleMusicGet(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Music == nil {
		notLoaded(ctx, w, "the music model", "-music")
		return
	}
	job, ok := s.Music.Get(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no music job "+r.PathValue("id"))
		return
	}
	writeJSON(w, http.StatusOK, job)
}

// MusicList is GET /v1/music's page.
type MusicList struct {
	Object  string     `json:"object"` // "list"
	Data    []MusicJob `json:"data"`
	FirstID *string    `json:"first_id"`
	LastID  *string    `json:"last_id"`
	HasMore bool       `json:"has_more"`
}

func (s *Server) handleMusicList(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Music == nil {
		notLoaded(ctx, w, "the music model", "-music")
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
	jobs := s.Music.List()
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
	out := MusicList{Object: "list", Data: jobs}
	if len(jobs) > limit {
		out.Data, out.HasMore = jobs[:limit], true
	}
	if n := len(out.Data); n > 0 {
		out.FirstID, out.LastID = &out.Data[0].ID, &out.Data[n-1].ID
	}
	writeJSON(w, http.StatusOK, out)
}

// musicTypes are the content types of the formats served.
var musicTypes = map[string]string{
	"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "opus": "audio/ogg",
}

func (s *Server) handleMusicContent(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Music == nil {
		notLoaded(ctx, w, "the music model", "-music")
		return
	}
	path, job, ok := s.Music.content(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no music job "+r.PathValue("id"))
		return
	}
	if job.Status != MusicCompleted {
		writeError(w, http.StatusConflict, "invalid_request_error",
			fmt.Sprintf("music %s is %s, not completed", job.ID, job.Status))
		return
	}
	f, err := os.Open(path)
	if err != nil {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no music job "+job.ID)
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		serverError(ctx, w, "music content", err)
		return
	}
	ct := musicTypes[job.Format]
	if ct == "" {
		ct = "application/octet-stream"
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Content-Disposition", `inline; filename="`+job.ID+"."+job.Format+`"`)
	http.ServeContent(w, r, "", st.ModTime(), f)
}

func (s *Server) handleMusicDelete(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Music == nil {
		notLoaded(ctx, w, "the music model", "-music")
		return
	}
	id := r.PathValue("id")
	if !s.Music.Delete(id) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "no music job "+id)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "music.deleted", "deleted": true})
}
