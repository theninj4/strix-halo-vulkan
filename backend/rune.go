package backend

// Rune v3, the decision model that replaces Kev (research/rune-vertical.md
// R6): one staged model behind two endpoints. /v1/decisions is surogate's
// decisions v1 as Rune is served upstream; /v1/systemone is TypeSafe's
// System One, Kev's endpoint, translated onto the same questions
// (decide/systemone.go), so Kev's clients keep working.
//
// Requests are packed passes run by one worker under the device lock,
// several to a pass (R8). Images (R10) go through the vision tower on the
// request's own goroutine first.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/decide"
	"strix-halo-vulkan/gemma4"
	"strix-halo-vulkan/util"
	"strix-halo-vulkan/vk"
)

// RuneOptions is what cmd/serve's -rune flags come to.
type RuneOptions struct {
	Model  string // the checkpoint directory
	Device *Device
	// Rows is the longest pass: the shared state once plus every question's
	// suffix. Zero takes 8192.
	Rows int
	// Temperature is the calibration temperature every answer is read at
	// (decisions v1's --decision-temperature). Zero takes 2, Rune's card.
	Temperature float64
	// BatchTokens is the most rows concurrent requests share one pass up
	// to (R8): a pass reads every expert whatever its length, so a row is
	// cheapest at ~1024 (0.36 ms) and dearest alone. A request over it runs
	// alone. Zero takes 1024.
	BatchTokens int
	// Vision stages the vision tower (~1.3 GB) so requests may carry
	// `images` (R10); without it they answer 400 vision_disabled.
	Vision bool
}

// runeModelIDs are the names a request may carry: Rune's own, and Kev's and
// the TypeSafe SDK's defaults, so a System One client configured for Kev
// works unchanged.
var runeModelIDs = []string{"rune", "rune-26b-a4b", "kev-latest", "jev-latest"}

// Rune serves decisions and System One requests.
type Rune struct {
	opt      RuneOptions
	mu       sync.Mutex // guards gpu and jobs against Close
	gpu      *gemma4.GPU
	codebook []string

	jobs    chan runeJob
	stopped chan struct{}
	// passes and batched count the passes and the requests they answered.
	passes, batched int
}

// runeJob is one planned request waiting for the device.
type runeJob struct {
	plan  *gemma4.Plan
	start time.Time
	done  chan runeResult
}

type runeResult struct {
	ro  gemma4.Readout
	ms  float64 // the request's wall time from queueing to its answer
	err error
}

// NewRune stages Rune: ~26 GB of int8 weights, quantised from bf16 at load.
func NewRune(opt RuneOptions) (*Rune, error) {
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: rune needs the device (it has no CPU path)")
	}
	if opt.Rows <= 0 {
		opt.Rows = 8192
	}
	if opt.Temperature == 0 {
		opt.Temperature = 2
	}
	if opt.BatchTokens <= 0 {
		opt.BatchTokens = 1024
	}
	if !(opt.Temperature > 0) {
		return nil, fmt.Errorf("backend: the decision temperature must be greater than zero")
	}
	b := &Rune{opt: opt}
	err := opt.Device.Do(func(dev *vk.Device) error {
		g, err := gemma4.Load(dev, opt.Model, gemma4.Options{Rows: opt.Rows, Vision: opt.Vision})
		b.gpu = g
		return err
	})
	if err != nil {
		return nil, err
	}
	b.codebook = b.gpu.Tok.Codebook()
	b.jobs = make(chan runeJob, 1024)
	b.stopped = make(chan struct{})
	go b.work()
	return b, nil
}

// Close releases the device objects.
func (b *Rune) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.jobs != nil {
		close(b.jobs)
		<-b.stopped
		b.jobs = nil
	}
	if b.gpu != nil {
		_ = b.opt.Device.Do(func(*vk.Device) error { b.gpu.Destroy(); return nil })
		b.gpu = nil
	}
}

// Models is what GET /v1/models reports.
func (b *Rune) Models() []api.Model {
	out := make([]api.Model, len(runeModelIDs))
	for i, id := range runeModelIDs {
		out[i] = api.Model{ID: id, Object: "model", OwnedBy: "invergent-ai"}
	}
	return out
}

// answer plans a request on the caller's goroutine (tokenising, the
// refusals) and queues it for the worker.
func (b *Rune) answer(ctx context.Context, req *decide.Request) ([]decide.Answer, gemma4.Usage, float64, error) {
	start := time.Now()
	b.mu.Lock()
	g := b.gpu
	b.mu.Unlock()
	if g == nil {
		return nil, gemma4.Usage{}, 0, fmt.Errorf("backend: the decision model is closed")
	}
	pics, err := b.pictures(ctx, g, req)
	if err != nil {
		return nil, gemma4.Usage{}, 0, err
	}
	plan, err := g.PlanImages(req, b.codebook, pics)
	if err != nil {
		return nil, gemma4.Usage{}, 0, err
	}
	// The tower runs here, on the request's goroutine under the device
	// lock, so the worker's passes stay text only; a picture seen in the
	// last few requests is not run again (gemma4's feature cache).
	var todo []*gemma4.Picture
	for _, p := range pics {
		if !g.Cached(p) {
			todo = append(todo, p)
		}
	}
	if len(todo) > 0 {
		if err := b.opt.Device.Do(func(*vk.Device) error { return g.EncodePictures(todo) }); err != nil {
			return nil, gemma4.Usage{}, 0, err
		}
	}
	job := runeJob{plan: plan, start: start, done: make(chan runeResult, 1)}
	b.mu.Lock()
	if b.jobs == nil {
		b.mu.Unlock()
		return nil, gemma4.Usage{}, 0, fmt.Errorf("backend: the decision model is closed")
	}
	b.jobs <- job
	b.mu.Unlock()
	var res runeResult
	select {
	case res = <-job.done:
	case <-ctx.Done():
		// The pass runs anyway; its answer goes to a buffered channel.
		return nil, gemma4.Usage{}, 0, ctx.Err()
	}
	if res.err != nil {
		return nil, res.ro.Usage, res.ms, res.err
	}
	answers := make([]decide.Answer, len(req.Questions))
	for i := range req.Questions {
		a, err := decide.Resolve(&req.Questions[i], res.ro.Logits[i], b.opt.Temperature)
		if err != nil {
			return nil, res.ro.Usage, res.ms, err
		}
		answers[i] = a
	}
	return answers, res.ro.Usage, res.ms, nil
}

// pictures reads a request's images (data or http(s) URLs, fetched as
// OpenAI's API fetches image URLs) and patchifies them on the host.
func (b *Rune) pictures(ctx context.Context, g *gemma4.GPU, req *decide.Request) ([]*gemma4.Picture, error) {
	if len(req.Images) == 0 {
		return nil, nil
	}
	if g.Vis == nil {
		return nil, &decide.Error{Message: "this server was started without vision; images are not accepted",
			Param: "images", Code: "vision_disabled"}
	}
	pics := make([]*gemma4.Picture, len(req.Images))
	for i, u := range req.Images {
		raw, err := util.FetchImage(ctx, u)
		if err != nil {
			return nil, &decide.Error{Message: fmt.Sprintf("images[%d]: %v", i, err), Param: "images", Code: "invalid_decisions_request"}
		}
		if pics[i], err = g.Vis.Cfg.NewPicture(raw); err != nil {
			return nil, &decide.Error{Message: fmt.Sprintf("images[%d]: %v", i, err), Param: "images", Code: "invalid_decisions_request"}
		}
	}
	return pics, nil
}

// work is the one goroutine that runs the model (R8). It takes a request,
// then everything else already waiting that fits beside it within
// BatchTokens rows, and answers them in one pass: they share its weight
// reads. A lone request waits for nothing; a request over the budget runs
// alone; one that does not fit waits for the next pass, first in line.
func (b *Rune) work() {
	defer close(b.stopped)
	var held *runeJob
	for {
		var first runeJob
		if held != nil {
			first, held = *held, nil
		} else {
			j, ok := <-b.jobs
			if !ok {
				return
			}
			first = j
		}
		batch := []runeJob{first}
		segs := []gemma4.Segment{first.plan.Segment()}
		rows := first.plan.Rows()
	gather:
		for rows < b.opt.BatchTokens {
			select {
			case j, ok := <-b.jobs:
				if !ok {
					break gather
				}
				next := append(segs, j.plan.Segment())
				if rows+j.plan.Rows() > b.opt.BatchTokens || !b.gpu.Fits(next) {
					held = &j
					break gather
				}
				batch, segs, rows = append(batch, j), next, rows+j.plan.Rows()
			default:
				break gather
			}
		}
		plans := make([]*gemma4.Plan, len(batch))
		for i, j := range batch {
			plans[i] = j.plan
		}
		var ros []gemma4.Readout
		err := b.opt.Device.Do(func(*vk.Device) error {
			var err error
			ros, err = b.gpu.RunPlans(plans)
			return err
		})
		b.passes++
		b.batched += len(batch)
		for i, j := range batch {
			res := runeResult{ms: float64(time.Since(j.start).Microseconds()) / 1000, err: err}
			if err == nil {
				res.ro = ros[i]
			}
			j.done <- res
		}
	}
}

// Batching is how many passes ran and how many requests they answered.
func (b *Rune) Batching() (passes, requests int) { return b.passes, b.batched }

// Decisions answers one /v1/decisions body.
func (b *Rune) Decisions(ctx context.Context, body []byte) ([]byte, error) {
	req, err := decide.Parse(body)
	if err != nil {
		return nil, err
	}
	if !slices.Contains(runeModelIDs, req.Model) {
		return nil, fmt.Errorf("%w: %q (this server has %v)", api.ErrUnknownModel, req.Model, runeModelIDs)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	answers, use, _, err := b.answer(ctx, req)
	if err != nil {
		return nil, err
	}
	var id [8]byte
	_, _ = rand.Read(id[:])
	return decide.Response("dec-"+hex.EncodeToString(id[:]), req.Model, "strix-halo-vulkan", answers,
		use.InputTokens, use.OutputTokens), nil
}

// SystemOne answers one /v1/systemone body, Kev's contract, from the same
// model.
func (b *Rune) SystemOne(ctx context.Context, body []byte) ([]byte, error) {
	req, err := decide.ParseSystemOne(body)
	if err != nil {
		return nil, runeUnprocessable(err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	answers, use, ms, err := b.answer(ctx, req)
	if err != nil {
		return nil, runeUnprocessable(err)
	}
	out := len(b.gpu.Tok.Encode(decide.SystemOneAnswers(answers, true)))
	return decide.SystemOneResponse(req.Model, answers, use.InputTokens, out, ms), nil
}

// runeUnprocessable is System One's 422 for a request the protocol refuses,
// as Kev answered them; anything else passes through as the server's.
func runeUnprocessable(err error) error {
	var se *decide.SystemOneError
	var de *decide.Error
	if errors.As(err, &se) || errors.As(err, &de) {
		return fmt.Errorf("%w: %v", api.ErrUnprocessable, err)
	}
	return err
}
