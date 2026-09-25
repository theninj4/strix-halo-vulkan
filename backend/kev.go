package backend

import (
	"context"
	"errors"
	"fmt"
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
}

const defaultKevTokens = 8192

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

// work is the one goroutine that runs the model (K7.5). It takes whatever is
// queued when the device frees up -- up to MaxBatch requests, never waiting
// for more -- and runs them as one pass: a lone request pays nothing for the
// batching, and a burst shares its passes. A batch that fails is retried a
// request at a time, so one bad request (a question longer than a pass)
// fails alone.
func (b *Kev) work() {
	defer close(b.stopped)
	for job := range b.jobs {
		batch := []kevJob{job}
	drain:
		for len(batch) < b.opt.MaxBatch {
			select {
			case j, ok := <-b.jobs:
				if !ok {
					break drain
				}
				batch = append(batch, j)
			default:
				break drain
			}
		}
		b.run(batch)
	}
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

func unprocessable(err error) error {
	var ve *kev.ValidationError
	var oe *kev.OverflowError
	if errors.As(err, &ve) || errors.As(err, &oe) {
		return fmt.Errorf("%w: %v", api.ErrUnprocessable, err)
	}
	return err
}
