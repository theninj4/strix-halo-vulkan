package ocr

// Element-level recognition (OCR.md O5, batched in O11): an image and a
// task prompt in, the text out. This is the HF checkpoint alone, and the unit
// the chat door (O6) and the page pipeline (O8) both call.
//
// A request is the host side (Process, BuildPrompt), the tower (GPUTower)
// and the language model (LM): the image rows the tower returns replace the
// prompt's placeholder rows, the prompt is prefilled with its last row's
// logits, and generation is greedy (OCR.md decision 6) until </s> or the
// cap.
//
// Requests are batched (OCR.md decision 10, O11): one goroutine owns the
// device state and runs passes whose rows are every admitted request's next
// rows, a slot each. A decoding request adds its last token, a new one its
// prompt (in chunks of what a pass has left), and each pass returns the
// logits of every request whose rows it finished. A step of 16 rows costs
// about two of one. The KV cache is a shared page pool: when it runs dry, the
// youngest request gives its pages back and later prefills its prompt and
// its tokens so far again (vLLM's recompute preemption), so the oldest always
// progresses.
//
// A request's numbers depend a little on its company: a pass of up to three
// rows runs GEMVs and a larger one GEMMs, which sum in another order. Greedy
// text can flip at a near-tie between a request run alone and the same one
// run in a crowd; RecognizeAll submits a page's regions in one go, so a page
// run alone is reproducible.
//
// What is *not* here, because the server PaddleOCR talks to does not do it
// either: Spotting's 2x Lanczos upscale of small crops
// (pre_process_for_spotting), the per-label pixel bounds, and the
// post-processing of the text (OTSL to HTML, formula delimiters, repetition
// truncation). Those are PaddleX's pipeline, O8.

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/vk"
)

// DefaultMaxTokens is PaddleX's PADDLEOCR_VL_MAX_NEW_TOKENS.
const DefaultMaxTokens = 8192

// Options size an engine.
type Options struct {
	// MaxPatches is the tower's budget (0: the checkpoint's max_pixels,
	// 5,120). A request above it is refused, not resized.
	MaxPatches int
	// LM sizes the language model: MaxLen bounds one request's prompt and
	// generation, Slots how many run at once, CachePositions the page pool
	// they share, Rows a pass.
	LM LMOptions
}

// DefaultOptions fits the checkpoint's largest image and PaddleX's cap, 32
// requests at once over 64k cached positions (1.2 GB).
func DefaultOptions() Options {
	return Options{LM: LMOptions{MaxLen: 10240, Slots: 32, Rows: 2048, CachePositions: 65536}}
}

// Stats count the scheduler's work since the engine loaded.
type Stats struct {
	Towers, Passes      int
	Rows, LogitRows     int // over every pass
	Tower, Pass, Sample time.Duration
	Preemptions         int
}

// Stats is a snapshot of the counters.
func (e *Engine) Stats() Stats {
	e.statsMu.Lock()
	defer e.statsMu.Unlock()
	return e.stats
}

func (e *Engine) count(f func(*Stats)) {
	e.statsMu.Lock()
	f(&e.stats)
	e.statsMu.Unlock()
}

// ErrClosed is a request's error when the engine is destroyed under it.
var ErrClosed = errors.New("ocr: the engine is closed")

// Engine is PaddleOCR-VL resident on a device.
type Engine struct {
	Tok   *Tokenizer
	tower *GPUTower
	lm    *LM
	o     Options

	// Do, if set, runs each device pass: a tower, and every pass of the
	// language model. A server sets it to its device lock so another
	// model's work interleaves with a long generation step by step, rather
	// than waiting out a page (backend/ocr.go). Nil runs them directly. Set
	// it before the first request.
	Do func(func() error) error

	start sync.Once
	in    chan []*job
	quit  chan struct{}
	ended chan struct{}

	statsMu sync.Mutex
	stats   Stats
	// greedyAdmit admits on a free slot alone, so the pool runs dry and
	// preemption runs: for its gate.
	greedyAdmit bool

	// The scheduler's own, touched by run alone.
	seq     int
	waiting []*job // by seq
	active  []*job
	slots   []int // free
}

func (e *Engine) do(fn func() error) error {
	if e.Do != nil {
		return e.Do(fn)
	}
	return fn()
}

// Load stages the tokenizer, the tower and the language model from dir.
func Load(dev *vk.Device, dir string, o Options) (*Engine, error) {
	if o.MaxPatches <= 0 {
		o.MaxPatches = MaxPatches
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	tw, err := LoadTower(dir, -1)
	if err != nil {
		return nil, err
	}
	gt, err := NewGPUTower(dev, tw, o.MaxPatches)
	if err != nil {
		return nil, err
	}
	lm, err := LoadLM(dev, dir, o.LM)
	if err != nil {
		gt.Destroy()
		return nil, err
	}
	o.LM = lm.o
	e := &Engine{Tok: tok, tower: gt, lm: lm, o: o,
		in: make(chan []*job), quit: make(chan struct{}), ended: make(chan struct{})}
	for s := o.LM.Slots - 1; s >= 0; s-- {
		e.slots = append(e.slots, s)
	}
	return e, nil
}

// Destroy fails the requests in flight and releases the device state.
func (e *Engine) Destroy() {
	started := true
	e.start.Do(func() { started = false })
	close(e.quit)
	if started {
		<-e.ended
	}
	e.tower.Destroy()
	e.lm.Destroy()
}

// DeviceBytes is what the engine holds on the device.
func (e *Engine) DeviceBytes() int {
	return e.tower.WeightBytes() + e.tower.ActivationBytes() + e.lm.DeviceBytes()
}

// Request is one element: an image and the text after it (one of Tasks'
// values, or any prompt).
type Request struct {
	Image  *pixels.RGB
	Prompt string
	// MinPixels and MaxPixels bound the resize (0: the checkpoint's).
	MinPixels, MaxPixels int
	// MaxTokens caps generation (0: DefaultMaxTokens), and is lowered to
	// what one slot holds after the prompt.
	MaxTokens int
	// RepetitionPenalty is vLLM's (and HF's): every token already in the
	// prompt or the output has its logit divided by it when positive and
	// multiplied by it when not. 0 or 1 is off. PaddleX sends it only when
	// its caller sets one.
	RepetitionPenalty float64
	// OnToken, if set, sees each generated id as it is chosen (</s> too),
	// on the caller's goroutine, so a slow one delays nobody else. An error
	// from it stops the generation and is Recognize's.
	OnToken func(id int32) error
}

// Result is the generation and where the time went.
type Result struct {
	IDs  []int32 // generated, </s> included when it ended there
	Text string  // decoded, specials skipped
	// Finish is "stop" (</s>) or "length" (the cap).
	Finish       string
	PromptTokens int
	GridH, GridW int // the patch grid

	// Process is the host side; Wait the queue before a slot; Tower the
	// image; Prefill from the tower to the first token and Decode from
	// there to the end, both in passes shared with other requests.
	Process, Wait, Tower, Prefill, Decode time.Duration
	// Preempted counts the times the request gave its cache back.
	Preempted int
}

// job is a request in the scheduler.
type job struct {
	req    Request
	ctx    context.Context
	im     *Image // the tower's input, until it has run
	p      *Prompt
	rows   []Row // the prompt's, image rows filled in by the tower
	maxTok int
	res    *Result

	seq    int
	slot   int // -1 when not holding one
	filled int // positions in the cache
	take   int // rows in the pass being built
	ids    []int32
	seen   []bool // the repetition penalty's

	tokens chan int32 // to the caller, never full
	done   chan error // one send

	arrived, admitted, towered, first time.Time
}

// target is how many positions must be cached for the next token's logits:
// the prompt and every token so far.
func (j *job) target() int { return len(j.rows) + len(j.ids) }

// row is the job's i-th sequence row: a prompt row, or generated token
// i-len(prompt), whose rope runs on from the prompt's Next on all axes.
func (j *job) row(i int) Row {
	if i < len(j.rows) {
		r := j.rows[i]
		r.Slot = j.slot
		return r
	}
	k := i - len(j.rows)
	pos := j.p.Next + int32(k)
	return Row{ID: j.ids[k], Slot: j.slot, Pos: i, Rope: [3]int32{pos, pos, pos}}
}

// prepare is a request's host side.
func (e *Engine) prepare(req Request) (*job, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("ocr: no image")
	}
	minPx, maxPx := req.MinPixels, req.MaxPixels
	if minPx <= 0 {
		minPx = DefaultMinPixels
	}
	if maxPx <= 0 {
		maxPx = DefaultMaxPixels
	}
	if minPx > maxPx {
		return nil, fmt.Errorf("ocr: min_pixels %d over max_pixels %d", minPx, maxPx)
	}
	t0 := time.Now()
	im, err := Process(req.Image, minPx, maxPx)
	if err != nil {
		return nil, err
	}
	if n := im.GridH * im.GridW; n > e.o.MaxPatches {
		return nil, fmt.Errorf("ocr: a %dx%d image is %d patches; this engine is staged for %d (max_pixels %d)",
			req.Image.W, req.Image.H, n, e.o.MaxPatches, e.o.MaxPatches*PatchSize*PatchSize)
	}
	p, err := e.Tok.BuildPrompt(req.Prompt, im.MergedH(), im.MergedW())
	if err != nil {
		return nil, err
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = DefaultMaxTokens
	}
	if room := e.o.LM.MaxLen - len(p.IDs); maxTok > room {
		if room <= 0 {
			return nil, fmt.Errorf("ocr: a %d-token prompt; the cache holds %d", len(p.IDs), e.o.LM.MaxLen)
		}
		maxTok = room
	}
	j := &job{req: req, im: im, p: p, maxTok: maxTok, slot: -1,
		res:    &Result{PromptTokens: len(p.IDs), GridH: im.GridH, GridW: im.GridW, Process: time.Since(t0)},
		tokens: make(chan int32, maxTok+1), done: make(chan error, 1)}
	j.rows = make([]Row, len(p.IDs))
	for i, id := range p.IDs {
		j.rows[i] = Row{ID: id, Pos: i, Rope: p.Pos[i]}
	}
	if pen := req.RepetitionPenalty; pen > 0 && pen != 1 {
		j.seen = make([]bool, e.lm.Vocab())
		for _, id := range p.IDs {
			j.seen[id] = true
		}
	}
	return j, nil
}

// Recognize runs one request to the end.
func (e *Engine) Recognize(ctx context.Context, req Request) (*Result, error) {
	res, err := e.RecognizeAll(ctx, []Request{req})
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

// RecognizeAll runs requests together, queued in their order in one go, and
// returns their results in that order; the first error cancels the rest.
func (e *Engine) RecognizeAll(ctx context.Context, reqs []Request) ([]*Result, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make([]*job, len(reqs))
	errs := make([]error, len(reqs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := range reqs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			jobs[i], errs[i] = e.prepare(reqs[i])
		}(i)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	now := time.Now()
	for _, j := range jobs {
		j.ctx, j.arrived = ctx, now
	}
	e.start.Do(func() { go e.run() })
	select {
	case e.in <- jobs:
	case <-e.quit:
		return nil, ErrClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	out := make([]*Result, len(jobs))
	for i, j := range jobs {
		wg.Add(1)
		go func(i int, j *job) {
			defer wg.Done()
			errs[i] = e.wait(j, cancel)
			if errs[i] != nil {
				cancel()
			}
			out[i] = j.res
		}(i, j)
	}
	wg.Wait()
	// The first error in order that is not the cancellation it caused.
	var first error
	for _, err := range errs {
		if err != nil && (first == nil || errors.Is(first, context.Canceled) && !errors.Is(err, context.Canceled)) {
			first = err
		}
	}
	if first != nil {
		return nil, first
	}
	return out, nil
}

// wait delivers a job's tokens to OnToken and returns its end.
func (e *Engine) wait(j *job, cancel func()) error {
	var cbErr error
	deliver := func(id int32) {
		if cbErr == nil && j.req.OnToken != nil {
			if cbErr = j.req.OnToken(id); cbErr != nil {
				cancel()
			}
		}
	}
	for {
		select {
		case id := <-j.tokens:
			deliver(id)
		case err := <-j.done:
			for {
				select {
				case id := <-j.tokens:
					deliver(id)
					continue
				default:
				}
				break
			}
			if cbErr != nil {
				return cbErr
			}
			if err != nil {
				return err
			}
			j.res.IDs = j.ids
			j.res.Text = e.Tok.Decode(j.ids, true)
			return nil
		}
	}
}

// run is the scheduler: it owns the tower, the language model and every
// job from queueing to its end.
func (e *Engine) run() {
	defer close(e.ended)
	for {
		if len(e.waiting) == 0 && len(e.active) == 0 {
			select {
			case js := <-e.in:
				e.enqueue(js)
			case <-e.quit:
				return
			}
		}
	drain:
		for {
			select {
			case js := <-e.in:
				e.enqueue(js)
			case <-e.quit:
				e.failAll(ErrClosed)
				return
			default:
				break drain
			}
		}
		e.reap()
		e.admit()
		if len(e.active) > 0 {
			if err := e.step(); err != nil {
				e.failAll(err)
			}
		}
	}
}

func (e *Engine) enqueue(js []*job) {
	for _, j := range js {
		j.seq = e.seq
		e.seq++
		e.waiting = append(e.waiting, j)
	}
}

// requeue puts a preempted job back in the queue by its age.
func (e *Engine) requeue(j *job) {
	i := sort.Search(len(e.waiting), func(i int) bool { return e.waiting[i].seq > j.seq })
	e.waiting = append(e.waiting, nil)
	copy(e.waiting[i+1:], e.waiting[i:])
	e.waiting[i] = j
}

// finish ends a job: its slot and pages go back, and its caller hears.
func (e *Engine) finish(j *job, err error) {
	if j.slot >= 0 {
		e.lm.Release(j.slot)
		e.slots = append(e.slots, j.slot)
		j.slot = -1
		for i, a := range e.active {
			if a == j {
				e.active = append(e.active[:i], e.active[i+1:]...)
				break
			}
		}
	}
	if err == nil {
		now := time.Now()
		j.res.Decode = now.Sub(j.first)
	}
	j.done <- err
}

func (e *Engine) failAll(err error) {
	for _, j := range append(append([]*job(nil), e.active...), e.waiting...) {
		e.finish(j, err)
	}
	e.waiting = nil
}

// reap ends the jobs whose callers have gone.
func (e *Engine) reap() {
	kept := e.waiting[:0]
	for _, j := range e.waiting {
		if err := j.ctx.Err(); err != nil {
			e.finish(j, err)
		} else {
			kept = append(kept, j)
		}
	}
	e.waiting = kept
	for _, j := range append([]*job(nil), e.active...) {
		if err := j.ctx.Err(); err != nil {
			e.finish(j, err)
		}
	}
}

// admit gives waiting jobs free slots, oldest first, while the pool has
// their cache and a page for each job already running, and runs the tower of
// each that has not had it. (Several crops in one tower pass cost what they
// cost one at a time: the tower is compute, not dispatch latency. O11.)
func (e *Engine) admit() {
	promised := len(e.active)
	for len(e.waiting) > 0 && len(e.slots) > 0 {
		j := e.waiting[0]
		need := (j.target() + PageSize - 1) / PageSize
		if len(e.active) > 0 && e.lm.FreePages() < promised+need && !e.greedyAdmit {
			return
		}
		e.waiting = e.waiting[1:]
		if j.admitted.IsZero() {
			j.admitted = time.Now()
			j.res.Wait = j.admitted.Sub(j.arrived)
		}
		if j.towered.IsZero() {
			if err := e.runTower(j); err != nil {
				e.finish(j, err)
				continue
			}
		}
		promised += need
		j.slot = e.slots[len(e.slots)-1]
		e.slots = e.slots[:len(e.slots)-1]
		e.active = append(e.active, j)
	}
}

func (e *Engine) runTower(j *job) error {
	t0 := time.Now()
	err := e.do(func() error {
		vis, err := e.tower.Forward(j.ctx, j.im)
		if err != nil {
			return err
		}
		if vis.Rows != j.p.ImageLen {
			return fmt.Errorf("ocr: the tower gave %d rows for %d image tokens", vis.Rows, j.p.ImageLen)
		}
		for k := 0; k < j.p.ImageLen; k++ {
			j.rows[j.p.ImageAt+k].Embed = vis.Row(k)
		}
		return nil
	})
	if err != nil {
		return err
	}
	j.towered = time.Now()
	j.res.Tower = j.towered.Sub(t0)
	j.im = nil
	e.count(func(s *Stats) { s.Towers++; s.Tower += j.res.Tower })
	return nil
}

// preempt takes a job's slot and cache back and queues it again; it keeps
// its tokens and prefills them with its prompt when it returns.
func (e *Engine) preempt(j *job) {
	e.lm.Release(j.slot)
	e.slots = append(e.slots, j.slot)
	j.slot, j.filled, j.take = -1, 0, 0
	for i, a := range e.active {
		if a == j {
			e.active = append(e.active[:i], e.active[i+1:]...)
			break
		}
	}
	j.res.Preempted++
	e.count(func(s *Stats) { s.Preemptions++ })
	e.requeue(j)
}

// reserve finds cache for j's rows in this pass, preempting the youngest
// job younger than j that has no rows in it yet; false if j must wait.
func (e *Engine) reserve(j *job, n int) bool {
	for {
		err := e.lm.Reserve(j.slot, n)
		if err == nil {
			return true
		}
		if !errors.Is(err, ErrCacheFull) {
			panic(err) // n is inside the slot by maxTok's clamp
		}
		var victim *job
		for _, a := range e.active {
			if a != j && a.take == 0 && a.seq > j.seq && (victim == nil || a.seq > victim.seq) {
				victim = a
			}
		}
		if victim == nil {
			return false
		}
		e.preempt(victim)
	}
}

// step runs one pass: a row for each decoding job, then prompt rows for the
// prefilling ones, oldest first, up to the pass's rows; then it samples every
// job whose rows the pass finished.
func (e *Engine) step() error {
	budget := e.o.LM.Rows
	jobs := append([]*job(nil), e.active...)
	sort.Slice(jobs, func(a, b int) bool { return jobs[a].seq < jobs[b].seq })
	for _, j := range jobs {
		j.take = 0
	}
	used := 0
	var body, tail []Row
	var ends []*job
	plan := func(j *job) {
		if j.slot < 0 { // preempted while planning
			return
		}
		n := min(j.target()-j.filled, budget-used)
		if n <= 0 || !e.reserve(j, j.filled+n) {
			return
		}
		j.take = n
		used += n
		for i := j.filled; i < j.filled+n-1; i++ {
			body = append(body, j.row(i))
		}
		last := j.row(j.filled + n - 1)
		if j.filled+n == j.target() {
			tail = append(tail, last)
			ends = append(ends, j)
		} else {
			body = append(body, last)
		}
	}
	for _, j := range jobs {
		if j.target()-j.filled == 1 {
			plan(j)
		}
	}
	for _, j := range jobs {
		if j.target()-j.filled > 1 {
			plan(j)
		}
	}
	if used == 0 {
		return nil
	}
	var logits [][]float32
	t0 := time.Now()
	err := e.do(func() error {
		var err error
		logits, _, err = e.lm.Pass(append(body, tail...), len(tail))
		return err
	})
	if err != nil {
		return err
	}
	for _, j := range jobs {
		j.filled += j.take
		j.take = 0
	}
	t1 := time.Now()

	// Sample in parallel (a vocabulary scan a row), then book in order.
	ids := make([]int32, len(ends))
	var wg sync.WaitGroup
	for i, j := range ends {
		wg.Add(1)
		go func(i int, j *job) {
			defer wg.Done()
			lg := logits[i]
			if j.seen != nil {
				pen := float32(j.req.RepetitionPenalty)
				for v, s := range j.seen {
					if !s {
						continue
					}
					if lg[v] > 0 {
						lg[v] /= pen
					} else {
						lg[v] *= pen
					}
				}
			}
			ids[i] = int32(argmaxF32(lg))
		}(i, j)
	}
	wg.Wait()
	now := time.Now()
	e.count(func(s *Stats) {
		s.Passes++
		s.Rows += used
		s.LogitRows += len(ends)
		s.Pass += t1.Sub(t0)
		s.Sample += now.Sub(t1)
	})
	eos := e.Tok.EOS()
	for i, j := range ends {
		id := ids[i]
		if len(j.ids) == 0 {
			j.first = now
			j.res.Prefill = now.Sub(j.towered)
		}
		j.ids = append(j.ids, id)
		if j.seen != nil {
			j.seen[id] = true
		}
		j.tokens <- id
		switch {
		case id == eos:
			j.res.Finish = "stop"
			e.finish(j, nil)
		case len(j.ids) >= j.maxTok:
			j.res.Finish = "length"
			e.finish(j, nil)
		}
	}
	return nil
}

// argmaxF32 is the first index of the largest value: greedy's tie rule,
// torch.argmax's.
func argmaxF32(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
