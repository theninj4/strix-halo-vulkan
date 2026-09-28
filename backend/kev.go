package backend

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/kev"
	"strix-halo-vulkan/vk"
)

// KevOptions is what cmd/serve's -kev flags come to.
type KevOptions struct {
	// Model is Kev's checkpoint directory: the adapter, the converted head
	// (reference/convert_kev_head.py) and the tokenizer.
	Model string
	// Base is the Qwen3.5 base it was trained on, at the revision head.json
	// names.
	Base string
	// Device runs the model. There is no CPU path: the oracle is Kev's own
	// Python (reference/dump_kev.py), and 4B parameters in fp32 on the host
	// is not a thing to serve.
	Device *Device
	// MaxTokens is the longest packed request (the state once, plus every
	// question's branch) one pass holds. Zero takes defaultKevTokens.
	MaxTokens int
	// FP16 stages the weights as halves instead of int8 (K7.1's control):
	// 1.27x slower a request, 7.1 GB against 3.8.
	FP16 bool
	// CacheStates and CacheTokens size the prefix cache (K7.2): how many
	// states a repeated text is answered from, and the longest one kept.
	// Negative CacheStates turns it off; zero takes kev's defaults (4, 4096).
	CacheStates, CacheTokens int
	// MaxBatch is the most requests one pass answers (K7.5); zero takes 8,
	// the model's state slots.
	MaxBatch int
	// ChunkTokens is the chunk a long request's state runs in (K9), rounded
	// up to a multiple of 64; zero takes 512, negative never chunks.
	ChunkTokens int
	// BatchTokens is the most rows requests share a pass up to (K9): past
	// it a pass is compute-bound, so a bigger one only makes its short
	// requests wait for its long ones. A request over it runs alone. Zero
	// takes defaultKevBatchTokens.
	BatchTokens int
}

const (
	defaultKevTokens      = 8192
	defaultKevBatchTokens = 1024
)

// kevModelIDs are the names a request may carry. kev-latest is Kev's own;
// jev-latest is the TypeSafe SDK's default, so an unconfigured client works,
// as it does against Kev's server.
var kevModelIDs = []string{"kev-latest", "jev-latest"}

// Kev is the classification adapter: an api.SystemOneBackend over kev.GPU.
// One mutex guards the pass, as for every single-arena model here.
type Kev struct {
	opt KevOptions
	enc *kev.Encoder

	mu  sync.Mutex // guards gpu and jobs against Close
	gpu *kev.GPU

	jobs    chan kevJob
	stopped chan struct{}
	stream  *kevStream // the request running in chunks, owned by work
	// batches and batched count the passes and the requests they answered.
	batches, batched int
}

// NewKev loads Kev-4B with the LoRA merged in: 3.8 GB of int8 weights (7.1 GB
// with FP16), staged in about ten seconds.
func NewKev(opt KevOptions) (*Kev, error) {
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: kev needs the device (it has no CPU path)")
	}
	if opt.MaxTokens <= 0 {
		opt.MaxTokens = defaultKevTokens
	}
	if opt.MaxBatch <= 0 {
		opt.MaxBatch = 8
	}
	if opt.BatchTokens <= 0 {
		opt.BatchTokens = defaultKevBatchTokens
	}
	enc, err := kev.LoadEncoder(opt.Model)
	if err != nil {
		return nil, err
	}
	b := &Kev{opt: opt, enc: enc}
	err = opt.Device.Do(func(dev *vk.Device) error {
		o := kev.DefaultOptions()
		o.MaxTokens = opt.MaxTokens
		if opt.FP16 {
			o.Bank = kev.BankFP16
		}
		if opt.CacheStates != 0 {
			o.CacheStates = max(opt.CacheStates, 0)
		}
		if opt.CacheTokens > 0 {
			o.CacheTokens = opt.CacheTokens
		}
		o.Slots = opt.MaxBatch
		g, err := kev.LoadWith(dev, opt.Model, opt.Base, o)
		if err == nil {
			switch {
			case opt.ChunkTokens < 0:
				g.ChunkRows = 0
			case opt.ChunkTokens > 0:
				g.ChunkRows = (opt.ChunkTokens + 63) &^ 63
			}
		}
		b.gpu = g
		return err
	})
	if err != nil {
		return nil, err
	}
	b.jobs = make(chan kevJob, 1024)
	b.stopped = make(chan struct{})
	go b.work()
	return b, nil
}

// Close releases the device objects.
func (b *Kev) Close() {
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
func (b *Kev) Models() []api.Model {
	out := make([]api.Model, len(kevModelIDs))
	for i, id := range kevModelIDs {
		out[i] = api.Model{ID: id, Object: "model", OwnedBy: "jaredpalmer"}
	}
	return out
}

// Cache is the prefix cache's slots and the longest state it keeps.
func (b *Kev) Cache() (slots, tokens int) {
	_, _, _, slots, tokens = b.gpu.CacheStats()
	return slots, tokens
}

// MaxTokens is the pass size.
func (b *Kev) MaxTokens() int { return b.gpu.Rows() }

// Bank is the weight width staged.
func (b *Kev) Bank() string { return b.gpu.Bank.String() }

// SystemOne answers one request body with Kev's response body.
func (b *Kev) SystemOne(ctx context.Context, body []byte) ([]byte, error) {
	req, err := kev.ParseRequest(body)
	if err != nil {
		return nil, unprocessable(err)
	}
	rec, meta := kev.ToRecord(req)
	enc, err := b.enc.Encode(rec, kev.MaxState, kev.MaxRow)
	if err != nil {
		return nil, unprocessable(err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	job := kevJob{enc: enc, done: make(chan kevResult, 1)}
	b.mu.Lock()
	if b.gpu == nil {
		b.mu.Unlock()
		return nil, fmt.Errorf("backend: the classification model is closed")
	}
	b.jobs <- job
	b.mu.Unlock()
	var res kevResult
	select {
	case res = <-job.done:
	case <-ctx.Done():
		// The pass runs anyway; its answer goes to a buffered channel nobody reads.
		return nil, ctx.Err()
	}
	if res.err != nil {
		return nil, unprocessable(res.err)
	}

	answers := kev.ToAnswers(res.probs, meta)
	out, err := b.enc.Tok.Encode(string(kev.AnswersJSON(answers, true)))
	if err != nil {
		return nil, err
	}
	return kev.ResponseJSON(req.Model, answers, len(enc.IDs), len(out), res.ms), nil
}

// kevJob is one encoded request waiting for the device.
type kevJob struct {
	enc  *kev.Encoding
	done chan kevResult
}

type kevResult struct {
	probs [][]float64
	ms    float64 // the pass's wall time: latency_ms, the batch's model time
	err   error
}

// work is the one goroutine that runs the model (K7.5, K9). Whenever the
// device frees up it takes everything queued into its waiting list and runs
// one pass chosen by pickKev: the cheapest requests first, packed together up
// to -kev-batch requests and -kev-batch-tokens rows, so a burst of short
// requests does not wait behind a long text's pass, and requests that do not
// all fit one pass still share passes. A lone request pays nothing for this.
//
// A request over the budget whose state is long runs as the model's stream
// (kev.GPU.BeginStream): its state a chunk a pass, -kev-chunk tokens, so a
// short request that arrives meanwhile waits for one chunk and not the whole
// text. The stream's chunks go when nothing cheaper is waiting, or when the
// stream has waited kevMaxWait; cheaper requests ride along within the
// budget. One stream runs at a time, and a second long request waits for it.
//
// A batch that fails is retried a request at a time, so one bad request (a
// question longer than a pass) fails alone.
func (b *Kev) work() {
	defer close(b.stopped)
	var waiting []kevWaiting
	open := true
	for open || len(waiting) > 0 || b.stream != nil {
		if len(waiting) == 0 && b.stream == nil {
			job, ok := <-b.jobs
			if !ok {
				break
			}
			waiting = append(waiting, kevWaiting{job: job, since: time.Now()})
		}
	drain:
		for open {
			select {
			case j, ok := <-b.jobs:
				if !ok {
					open = false
					break drain
				}
				waiting = append(waiting, kevWaiting{job: j, since: time.Now()})
			default:
				break drain
			}
		}
		for i := range waiting {
			waiting[i].cost = b.gpu.PassCost(waiting[i].job.enc)
		}
		now := time.Now()
		budget := b.opt.BatchTokens
		fits := func(withStream bool, sub []int) func([]int) bool {
			return func(idx []int) bool {
				encs := make([]*kev.Encoding, len(idx))
				for i, k := range idx {
					encs[i] = waiting[sub[k]].job.enc
				}
				return b.gpu.FitsStep(withStream, encs)
			}
		}
		// small is the waiting requests within the budget that fit a pass
		// beside the stream.
		small := func() []int {
			var sub []int
			for k, w := range waiting {
				if w.cost <= budget && fits(false, []int{k})([]int{0}) {
					sub = append(sub, k)
				}
			}
			return sub
		}
		pickFrom := func(sub []int, lead int, withStream bool) []int {
			if len(sub) == 0 {
				return nil
			}
			w := make([]kevWaiting, len(sub))
			for i, k := range sub {
				w[i] = waiting[k]
			}
			n := b.opt.MaxBatch
			if b.stream != nil {
				n-- // the stream keeps a state slot
			}
			var out []int
			for _, i := range pickKev(w, now, n, budget, kevMaxWait, lead, fits(withStream, sub)) {
				out = append(out, sub[i])
			}
			return out
		}

		var pick []int
		withStream := false
		if b.stream == nil {
			all := make([]int, len(waiting))
			for i := range all {
				all[i] = i
			}
			pick = pickFrom(all, -1, false)
			head := waiting[pick[0]]
			if len(pick) == 1 && head.cost > budget && b.gpu.Chunks(head.job.enc) {
				if err := b.gpu.BeginStream(head.job.enc); err == nil {
					b.stream = &kevStream{job: head.job, moved: now}
					waiting = append(waiting[:pick[0]], waiting[pick[0]+1:]...)
					withStream = true
					pick = pickFrom(small(), b.gpu.StreamRows(), true)
				}
			}
		} else {
			if now.Sub(b.stream.moved) < kevMaxWait {
				pick = pickFrom(small(), -1, false)
			}
			if len(pick) == 0 {
				withStream = true
				pick = pickFrom(small(), b.gpu.StreamRows(), true)
			}
		}

		batch := make([]kevJob, len(pick))
		taken := make([]bool, len(waiting))
		for i, k := range pick {
			batch[i], taken[k] = waiting[k].job, true
		}
		rest := waiting[:0]
		for k, w := range waiting {
			if !taken[k] {
				rest = append(rest, w)
			}
		}
		waiting = rest
		if b.stream != nil {
			b.step(withStream, batch)
		} else {
			b.run(batch)
		}
	}
}

// kevMaxWait is how long a request may be passed over for cheaper ones
// before it goes first, so a long text is not starved by a stream of short
// ones.
const kevMaxWait = time.Second

// kevWaiting is a queued request, when it arrived, and its pass cost in rows.
type kevWaiting struct {
	job   kevJob
	since time.Time
	cost  int
}

// kevStream is the request running as the model's stream (K9): when its
// last chunk ran, and the wall time of the passes it was in.
type kevStream struct {
	job   kevJob
	moved time.Time
	ms    float64
}

// pickKev chooses the next pass from the waiting requests and returns their
// indices. With lead < 0 the first is the oldest request if it has waited
// maxWait, else the cheapest (the oldest among equals), and it runs even
// alone and over budget. With lead >= 0 the pass already holds lead rows
// (the stream's step) and nothing leads. Then the others join, cheapest
// first, while the pass stays within maxN requests and budget rows and fits
// (one pass, as the planner lays it out). A request that does not fit is
// skipped, not the end of the search.
func pickKev(w []kevWaiting, now time.Time, maxN, budget int, maxWait time.Duration, lead int, fits func([]int) bool) []int {
	order := make([]int, len(w))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return w[order[a]].cost < w[order[b]].cost })
	var pick []int
	rows := lead
	if lead < 0 {
		head := order[0]
		oldest := 0
		for i := range w {
			if w[i].since.Before(w[oldest].since) {
				oldest = i
			}
		}
		if now.Sub(w[oldest].since) >= maxWait {
			head = oldest
		}
		pick, rows = []int{head}, w[head].cost
	}
	for _, k := range order {
		if len(pick) >= maxN {
			break
		}
		if slices.Contains(pick, k) || rows+w[k].cost > budget {
			continue
		}
		if !fits(append(pick[:len(pick):len(pick)], k)) {
			continue
		}
		pick, rows = append(pick, k), rows+w[k].cost
	}
	return pick
}

func (b *Kev) run(batch []kevJob) {
	encs := make([]*kev.Encoding, len(batch))
	for i, j := range batch {
		encs[i] = j.enc
	}
	start := time.Now()
	var probs [][][]float64
	err := b.opt.Device.Do(func(*vk.Device) error {
		var e error
		probs, _, e = b.gpu.ProbsBatch(encs)
		return e
	})
	ms := float64(time.Since(start).Microseconds()) / 1000
	if err != nil && len(batch) > 1 {
		for _, j := range batch {
			b.run([]kevJob{j})
		}
		return
	}
	b.batches++
	b.batched += len(batch)
	for i, j := range batch {
		if err != nil {
			j.done <- kevResult{err: err}
			continue
		}
		j.done <- kevResult{probs: probs[i], ms: ms}
	}
}

// step runs one pass beside the stream (K9): its next chunk when withStream,
// and batch. The stream's latency_ms is the wall time of the passes it was
// in. A failed step drops the stream, which fails, and retries batch a
// request at a time.
func (b *Kev) step(withStream bool, batch []kevJob) {
	encs := make([]*kev.Encoding, len(batch))
	for i, j := range batch {
		encs[i] = j.enc
	}
	start := time.Now()
	var sp [][]float64
	var probs [][][]float64
	err := b.opt.Device.Do(func(*vk.Device) error {
		var e error
		sp, probs, e = b.gpu.StepProbs(withStream, encs)
		return e
	})
	ms := float64(time.Since(start).Microseconds()) / 1000
	st := b.stream
	if withStream {
		st.ms += ms
		st.moved = time.Now()
	}
	if err != nil {
		b.gpu.AbortStream()
		b.stream = nil
		st.job.done <- kevResult{err: err}
		for _, j := range batch {
			b.run([]kevJob{j})
		}
		return
	}
	if len(batch) > 0 {
		b.batches++
		b.batched += len(batch)
	}
	for i, j := range batch {
		j.done <- kevResult{probs: probs[i], ms: ms}
	}
	if sp != nil {
		b.stream = nil
		b.batches++
		b.batched++
		st.job.done <- kevResult{probs: sp, ms: st.ms}
	}
}

func unprocessable(err error) error {
	var ve *kev.ValidationError
	var oe *kev.OverflowError
	if errors.As(err, &ve) || errors.As(err, &oe) {
		return fmt.Errorf("%w: %v", api.ErrUnprocessable, err)
	}
	return err
}
