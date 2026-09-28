package backend

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/embed"
	"strix-halo-vulkan/vk"
)

// EmbedOptions is what cmd/serve's flags come to.
type EmbedOptions struct {
	// Model is the Qwen3-Embedding-0.6B checkpoint directory.
	Model string
	// Device runs the model on the GPU. Without it this is the fp32 CPU
	// reference, which is two seconds a text rather than twelve
	// milliseconds -- correct, and not a thing to serve.
	Device *Device
	// MaxTokens is the longest run the arenas are sized for, and therefore
	// where an input is truncated. Zero takes defaultEmbedTokens.
	MaxTokens int
	// ID is the model id this backend answers to in /v1/models. Empty takes
	// defaultEmbedModelID.
	ID string
	// BatchTokens is the most rows one pass runs (E7), across however many
	// inputs and requests it takes them from. Zero takes
	// defaultEmbedBatchTokens; it is never below MaxTokens.
	BatchTokens int
}

const (
	defaultEmbedModelID = "qwen3-embedding-0.6b"
	// defaultEmbedTokens is the arena size, not the model's limit: the
	// checkpoint handles 32k positions, and the arenas are what a run costs
	// up front. 512 covers a query and most documents; a corpus of long
	// documents wants -embed-tokens raised and the memory that comes with
	// it.
	defaultEmbedTokens = 512
	// defaultEmbedBatchTokens is where a pass stops growing. Past ~450 rows a
	// pass is bound by the matrix cores (0.056 ms a row at 900 and at 1800),
	// so a bigger one buys no throughput and makes the query that arrives
	// behind it wait longer: at 1024 that wait is about 55 ms.
	defaultEmbedBatchTokens = 1024
)

// Embed is the Qwen3-Embedding adapter: an api.EmbeddingBackend over the
// model cmd/embed drives.
//
// On the device, requests do not run: their inputs join a queue that one
// worker drains a pass at a time (E7). A pass takes inputs round-robin across
// the waiting requests up to BatchTokens rows, so a lone query that arrives
// behind a 128-text indexing job rides the next pass instead of waiting out
// the job, and every pass is one Device.Do, so the other verticals get the
// device between passes. The CPU path runs a request at a time under mu,
// because it is only there to be correct.
type Embed struct {
	opt EmbedOptions
	id  string

	mu    sync.Mutex       // the model; held by a pass and by Close
	onGPU bool             // set at load and never changed: which path a request takes
	tok   *embed.Tokenizer // likewise
	gpu   *embed.GPU
	cpu   *embed.Model

	qmu     sync.Mutex // the queue, and every embedJob field below done
	queue   []*embedJob
	wake    chan struct{}
	quit    chan struct{}
	stopped chan struct{}
	passes  int // passes run, for the tests
}

// embedJob is one request's inputs on their way through the passes.
type embedJob struct {
	ctx  context.Context
	seqs [][]int32
	next int // the first input no pass has taken yet
	vecs [][]float32
	left int // inputs not yet back
	err  error
	done chan struct{}
}

// finish closes the job once. The caller holds qmu.
func (j *embedJob) finish(err error) {
	if j.left < 0 {
		return
	}
	j.err, j.left = err, -1
	close(j.done)
}

// NewEmbed loads the checkpoint. On the device that is 0.88 GB of fp16 banks
// staged in about 1.3 s; on the host it is 2.4 GB of fp32 and about 2 s.
func NewEmbed(opt EmbedOptions) (*Embed, error) {
	if opt.ID == "" {
		opt.ID = defaultEmbedModelID
	}
	if opt.MaxTokens <= 0 {
		opt.MaxTokens = defaultEmbedTokens
	}
	b := &Embed{opt: opt, id: opt.ID}
	if opt.Device == nil {
		m, err := embed.Load(opt.Model)
		if err != nil {
			return nil, err
		}
		b.cpu, b.tok = m, m.Tok
		return b, nil
	}
	if opt.BatchTokens <= 0 {
		opt.BatchTokens = defaultEmbedBatchTokens
	}
	opt.BatchTokens = max(opt.BatchTokens, opt.MaxTokens)
	b.opt = opt
	err := opt.Device.Do(func(dev *vk.Device) error {
		// The arenas are twice the pass: a pass is BatchTokens rows, but
		// each input's attention region is rounded up to a 64-row key block,
		// and 32 queries of 14 tokens want 2048 rows of plane for 448 of
		// stream. The rows are cheap (~80 KB each); the passes they allow are
		// 1.1 ms a query against 1.8 at half the width.
		g, err := embed.NewGPU(dev, opt.Model, 2*opt.BatchTokens)
		if err != nil {
			return err
		}
		b.gpu, b.tok = g, g.Tok
		return nil
	})
	if err != nil {
		return nil, err
	}
	b.onGPU = true
	b.wake = make(chan struct{}, 1)
	b.quit = make(chan struct{})
	b.stopped = make(chan struct{})
	go b.worker(b.quit)
	return b, nil
}

// Close releases the device objects.
func (b *Embed) Close() {
	b.qmu.Lock()
	quit := b.quit
	b.quit = nil
	b.qmu.Unlock()
	if quit != nil {
		close(quit)
		<-b.stopped
		b.qmu.Lock()
		for _, j := range b.queue {
			j.finish(fmt.Errorf("backend: the embedding model is closed"))
		}
		b.queue = nil
		b.qmu.Unlock()
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.gpu != nil {
		_ = b.opt.Device.Do(func(*vk.Device) error { b.gpu.Destroy(); return nil })
		b.gpu = nil
	}
	b.cpu = nil
}

// Models is what GET /v1/models reports.
func (b *Embed) Models() []api.Model {
	return []api.Model{{ID: b.id, Object: "model", OwnedBy: "qwen"}}
}

// MaxTokens is where an input is truncated.
func (b *Embed) MaxTokens() int { return b.opt.MaxTokens }

// Embed embeds every input in the request, in order.
//
// On the device the inputs are tokenized here and queued for the worker,
// which runs them in passes shared with whatever else is waiting (see Embed).
// A text's vector is the one it would get alone: RunBatch keeps the texts of a
// pass from seeing each other, and TestGPUBatchMatchesSingle holds that to the
// bit.
func (b *Embed) Embed(ctx context.Context, req *api.EmbeddingRequest) (*api.EmbeddingResult, error) {
	if req.Dimensions > embed.Dim {
		return nil, fmt.Errorf("%w: dimensions is %d; this model produces %d",
			api.ErrUnsupported, req.Dimensions, embed.Dim)
	}
	res := &api.EmbeddingResult{Vectors: make([][]float32, len(req.Input))}
	seqs := make([][]int32, len(req.Input))
	for i, text := range req.Input {
		// The instruction is prepended here rather than by the caller
		// because it is part of the *text* the model reads, so it has to be
		// inside the token count the response reports.
		if req.Instruct != "" {
			task := req.Instruct
			if task == "default" {
				task = embed.DefaultTask
			}
			text = embed.Instruct(task, text)
		}
		ids, err := b.encode(text)
		if err != nil {
			return nil, err
		}
		seqs[i] = ids
		res.Usage.PromptTokens += len(ids)
	}
	res.Usage.TotalTokens = res.Usage.PromptTokens

	// Not b.mu to ask whether the model is closed: a pass holds it on the
	// device, and a request that waited for it would join the queue only
	// once the passes ahead of it had drained -- a lone query behind a
	// 128-text job came back with the job, at 300 ms. queued and serial
	// each answer "closed" under their own lock.
	var vecs [][]float32
	var err error
	if b.onGPU {
		vecs, err = b.queued(ctx, seqs)
	} else {
		vecs, err = b.serial(ctx, seqs)
	}
	if err != nil {
		return nil, err
	}
	for i, v := range vecs {
		res.Vectors[i] = embed.Truncate(v, req.Dimensions)
	}
	return res, nil
}

// serial is the CPU path: one input at a time, cancellation between them.
func (b *Embed) serial(ctx context.Context, seqs [][]int32) ([][]float32, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cpu == nil {
		return nil, fmt.Errorf("backend: the embedding model is closed")
	}
	out := make([][]float32, len(seqs))
	for i, ids := range seqs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, err := b.cpu.EmbedIDs(ids)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// queued hands the inputs to the worker and waits for the last of them. A
// client that hangs up takes its untaken inputs out of the queue; the ones
// already in a pass finish with it, since a pass is a submitted command
// buffer with nothing to abandon inside it.
func (b *Embed) queued(ctx context.Context, seqs [][]int32) ([][]float32, error) {
	if len(seqs) == 0 {
		return nil, nil
	}
	j := &embedJob{ctx: ctx, seqs: seqs, vecs: make([][]float32, len(seqs)), left: len(seqs), done: make(chan struct{})}
	b.qmu.Lock()
	if b.quit == nil {
		b.qmu.Unlock()
		return nil, fmt.Errorf("backend: the embedding model is closed")
	}
	b.queue = append(b.queue, j)
	b.qmu.Unlock()
	select {
	case b.wake <- struct{}{}:
	default:
	}
	select {
	case <-j.done:
	case <-ctx.Done():
		b.qmu.Lock()
		j.finish(ctx.Err())
		b.dropFinished()
		b.qmu.Unlock()
		<-j.done
	}
	b.qmu.Lock()
	defer b.qmu.Unlock()
	return j.vecs, j.err
}

// dropFinished removes jobs that are done from the queue. The caller holds
// qmu.
func (b *Embed) dropFinished() {
	b.queue = slices.DeleteFunc(b.queue, func(j *embedJob) bool { return j.left < 0 })
}

// worker runs passes while anything is queued, until quit closes. It is
// handed quit rather than reading b.quit, which Close clears.
func (b *Embed) worker(quit <-chan struct{}) {
	defer close(b.stopped)
	for {
		select {
		case <-quit:
			return
		case <-b.wake:
		}
		for b.pass() {
			select {
			case <-quit:
				return
			default:
			}
		}
	}
}

// embedTake is one input a pass takes: job queue[job], input i of it.
type embedTake struct{ job, i int }

// pickEmbed chooses a pass: round-robin over the jobs in arrival order, one
// input from each per round, while the pass stays within budget rows and fits
// (the encoder's own test, which also counts the attention planes). A job
// whose next input does not fit is passed over for the rest of this pass, so
// a long document does not stop the short inputs behind it from joining.
// next[k] is job k's first untaken input.
func pickEmbed(lens [][]int, next []int, budget int, fits func([]int) bool) []embedTake {
	var pick []embedTake
	var picked []int
	rows := 0
	taken := make([]int, len(lens))
	full := make([]bool, len(lens))
	for {
		progress := false
		for k := range lens {
			i := next[k] + taken[k]
			if full[k] || i >= len(lens[k]) {
				continue
			}
			n := lens[k][i]
			if rows+n > budget || !fits(append(picked[:len(picked):len(picked)], n)) {
				full[k] = true
				continue
			}
			pick = append(pick, embedTake{job: k, i: i})
			picked = append(picked, n)
			rows += n
			taken[k]++
			progress = true
		}
		if !progress {
			// Something always goes: an input over the budget runs alone.
			if len(pick) == 0 {
				for k := range lens {
					if next[k] < len(lens[k]) {
						return []embedTake{{job: k, i: next[k]}}
					}
				}
			}
			return pick
		}
	}
}

// pass runs one pass and reports whether it found anything to run.
func (b *Embed) pass() bool {
	b.qmu.Lock()
	for _, j := range b.queue {
		if j.ctx.Err() != nil {
			j.finish(j.ctx.Err())
		}
	}
	b.dropFinished()
	lens := make([][]int, len(b.queue))
	next := make([]int, len(b.queue))
	for k, j := range b.queue {
		lens[k] = make([]int, len(j.seqs))
		for i, s := range j.seqs {
			lens[k][i] = len(s)
		}
		next[k] = j.next
	}
	var take []embedTake
	if len(b.queue) > 0 {
		take = pickEmbed(lens, next, b.opt.BatchTokens, b.gpu.Fits)
	}
	jobs := slices.Clone(b.queue)
	seqs := make([][]int32, len(take))
	for n, t := range take {
		seqs[n] = jobs[t.job].seqs[t.i]
		jobs[t.job].next = max(jobs[t.job].next, t.i+1)
	}
	b.qmu.Unlock()
	if len(take) == 0 {
		return false
	}

	var vecs [][]float32
	b.mu.Lock()
	err := fmt.Errorf("backend: the embedding model is closed")
	if b.gpu != nil {
		err = b.opt.Device.Do(func(*vk.Device) error {
			var e error
			vecs, e = b.gpu.EmbedBatch(seqs)
			return e
		})
	}
	b.mu.Unlock()

	b.qmu.Lock()
	defer b.qmu.Unlock()
	b.passes++
	for n, t := range take {
		j := jobs[t.job]
		if j.left < 0 {
			continue
		}
		if err != nil {
			j.finish(err)
			continue
		}
		j.vecs[t.i] = vecs[n]
		if j.left--; j.left == 0 {
			j.finish(nil)
		}
	}
	b.dropFinished()
	return true
}

// encode tokenizes and truncates. Truncation keeps the *front* of the text
// and re-appends the end-of-text token, which is what HF's
// `truncation=True` does and what last-token pooling requires: the pooled row
// is the appended token's, so a truncation that dropped it would pool a
// different token than every other input did.
func (b *Embed) encode(text string) ([]int32, error) {
	tok := b.cpuTokenizer()
	return tok.EncodeLimit(text, b.opt.MaxTokens)
}

// cpuTokenizer is the tokenizer, kept apart from the model at load: requests
// tokenize without the model lock a pass holds, and Close drops the model.
func (b *Embed) cpuTokenizer() *embed.Tokenizer { return b.tok }
