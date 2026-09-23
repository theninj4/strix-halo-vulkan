package backend

// The language model's scheduler (CONCURRENCY.md C2).
//
// One graph, several conversations. The graph holds `Slots` sequences
// (llm.GraphOpts.Slots, C1) and runs one pass at a time, so concurrency here
// is interleaving: a request holds a slot for its whole life, and its work
// reaches the device as **units** — one prefill chunk or one decode step —
// that a single goroutine picks from every live request's queue in turn.
// Nobody waits for another conversation's whole generation any more; they
// wait for one unit of it.
//
// Which unit runs next is the whole policy:
//
//  1. **Interactive before background, strictly.** While an interactive
//     request holds a slot, no background unit starts — including in the
//     millisecond between two of its decode steps, when it is sampling and
//     has nothing queued. Without that rule a background chunk slips into
//     every gap and the voice reply runs at half rate or worse.
//  2. **Within a class, whoever has had the least device time.** A prefill
//     chunk is worth hundreds of decode steps, so counting units would hand
//     a long prompt the device; counting time shares it.
//  3. **A background prefill runs in chunks of PreemptChunk**, because a
//     unit cannot be interrupted once it is on the device (the activation
//     arenas are shared), and an interactive request can arrive at any
//     moment. The served -llm-batch 8192 is ~7 s a chunk; the quantum is
//     what bounds a voice command's wait.
//
// Slots are handed out on the same order, and a slot is reserved for the
// interactive class when there is more than one, so that background agents
// can never occupy every slot and lock a voice command out.

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/vk"
)

// llmClass is a request's priority.
type llmClass int

const (
	classInteractive llmClass = iota
	classBackground
)

func (c llmClass) String() string {
	if c == classInteractive {
		return "interactive"
	}
	return "background"
}

// parseClass reads a class name, the way the flag and the header spell it.
func parseClass(s string) (llmClass, bool) {
	switch s {
	case "interactive", "priority":
		return classInteractive, true
	case "background", "flex", "default", "auto":
		return classBackground, true
	}
	return 0, false
}

// llmSlot is one of the graph's sequence slots, as the scheduler sees it.
type llmSlot struct {
	// held is the token sequence the slot's state is for, exactly as far as
	// the device has run it: what makes a second turn of the same
	// conversation a continuation. nil when the slot holds nothing usable.
	held []int32
	// job is the request holding the slot, or nil when it is free.
	job *llmJob
	// used is when it was last released, for least-recently-used.
	used time.Time
}

// llmJob is one request's claim on the scheduler.
type llmJob struct {
	ctx   context.Context
	class llmClass
	slot  int
	// reused is how many leading tokens of the prompt the slot already held.
	reused int
	// arrival breaks ties, oldest first.
	arrival uint64
	// served is the device time its units have taken, which is what rule 2
	// shares out.
	served time.Duration
	// prompt is what the slot is chosen against: the slot already holding
	// the longest prefix of it wins.
	prompt  []int32
	granted chan struct{}
}

// llmUnit is a pending run of tokens for one job: a prefill, which the
// scheduler cuts into chunks, or a decode step.
type llmUnit struct {
	job   *llmJob
	ids   []int32
	fresh bool
	// busy is the device time spent on it, and logits/err its result once
	// done is closed.
	busy   time.Duration
	logits []float32
	err    error
	done   chan struct{}
}

type llmSched struct {
	g     *llm.Graph
	dev   *Device
	batch int
	// preempt is the background prefill chunk; see rule 3.
	preempt int
	// reserve is how many slots only the interactive class may take.
	reserve int

	mu      sync.Mutex
	wake    *sync.Cond
	slots   []llmSlot
	waiting []*llmJob
	pending []*llmUnit
	// running is the unit on the device, which a cancelled caller must not
	// take out of the queue from under the loop.
	running *llmUnit
	arrival uint64
	closed  bool
	stopped chan struct{}
}

func newLLMSched(g *llm.Graph, dev *Device, batch, preempt, reserve int) *llmSched {
	s := &llmSched{
		g: g, dev: dev, batch: batch, preempt: preempt, reserve: reserve,
		slots: make([]llmSlot, g.Slots()), stopped: make(chan struct{}),
	}
	s.wake = sync.NewCond(&s.mu)
	go s.loop()
	return s
}

// acquire waits for a slot and claims it. The job it returns must be
// released, whatever happens after.
func (s *llmSched) acquire(ctx context.Context, class llmClass, ids []int32) (*llmJob, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, fmt.Errorf("the language model is closed")
	}
	s.arrival++
	j := &llmJob{ctx: ctx, class: class, slot: -1, arrival: s.arrival, prompt: ids,
		granted: make(chan struct{})}
	s.waiting = append(s.waiting, j)
	s.grant()
	s.mu.Unlock()

	select {
	case <-j.granted:
		return j, nil
	case <-ctx.Done():
		s.mu.Lock()
		defer s.mu.Unlock()
		if j.slot >= 0 {
			// Granted in the meantime; give it straight back.
			s.releaseLocked(j)
			return nil, ctx.Err()
		}
		s.dropWaiter(j)
		return nil, ctx.Err()
	}
}

// grant hands free slots to waiting jobs, best first.
func (s *llmSched) grant() {
	for {
		w := s.bestWaiter()
		if w == nil {
			return
		}
		slot := s.freeSlot(w.class, w.prompt)
		if slot < 0 {
			return
		}
		s.claim(w, slot)
	}
}

// bestWaiter is the waiting job a free slot goes to: interactive first, then
// the oldest.
func (s *llmSched) bestWaiter() *llmJob {
	var best *llmJob
	for _, w := range s.waiting {
		if best == nil || w.class < best.class || (w.class == best.class && w.arrival < best.arrival) {
			best = w
		}
	}
	return best
}

// freeSlot picks the slot for a job of this class: the one already holding
// the longest usable prefix of its prompt, else the least recently used.
// It is -1 when there is none the class may take.
func (s *llmSched) freeSlot(class llmClass, prompt []int32) int {
	free, bgBusy := 0, 0
	for i := range s.slots {
		switch {
		case s.slots[i].job == nil:
			free++
		case s.slots[i].job.class == classBackground:
			bgBusy++
		}
	}
	if free == 0 {
		return -1
	}
	if class == classBackground && len(s.slots) > s.reserve && bgBusy >= len(s.slots)-s.reserve {
		return -1
	}
	best, bestReuse := -1, -1
	for i := range s.slots {
		sl := &s.slots[i]
		if sl.job != nil {
			continue
		}
		reuse := usablePrefix(sl.held, prompt)
		if best < 0 || reuse > bestReuse || (reuse == bestReuse && sl.used.Before(s.slots[best].used)) {
			best, bestReuse = i, reuse
		}
	}
	return best
}

// usablePrefix is how much of a prompt a slot's held sequence saves. The
// reuse is all-or-nothing — a DeltaNet state has no inverse, so the held
// sequence must be a prefix of the prompt in full — and the prompt must be
// longer, because the logits of a finished pass are not kept.
func usablePrefix(held, prompt []int32) int {
	if n := commonPrefix(held, prompt); n > 0 && n == len(held) && n < len(prompt) {
		return n
	}
	return 0
}

func (s *llmSched) claim(j *llmJob, slot int) {
	s.dropWaiter(j)
	j.slot = slot
	j.reused = usablePrefix(s.slots[slot].held, j.prompt)
	// A newcomer starts level with the least-served job of its class rather
	// than at zero, or it would hold the device until it had caught up with
	// a conversation that has been running for a minute.
	first := true
	for i := range s.slots {
		if o := s.slots[i].job; o != nil && o.class == j.class && (first || o.served < j.served) {
			j.served, first = o.served, false
		}
	}
	s.slots[slot].job = j
	close(j.granted)
}

func (s *llmSched) dropWaiter(j *llmJob) {
	for i, w := range s.waiting {
		if w == j {
			s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
			return
		}
	}
}

// release gives the job's slot back. What the slot holds stays, so the
// conversation's next turn can continue it.
func (s *llmSched) release(j *llmJob) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseLocked(j)
}

func (s *llmSched) releaseLocked(j *llmJob) {
	if j.slot < 0 || s.slots[j.slot].job != j {
		return
	}
	s.slots[j.slot].job = nil
	s.slots[j.slot].used = time.Now()
	s.grant()
	s.wake.Broadcast()
}

// run queues tokens on the job's slot and waits for them: a whole prefill
// (fresh starts the slot's sequence over) or one decode step. It returns the
// last token's logits, and how long the unit sat behind other jobs' work.
func (s *llmSched) run(j *llmJob, ids []int32, fresh bool) ([]float32, time.Duration, error) {
	start := time.Now()
	u := &llmUnit{job: j, ids: ids, fresh: fresh, done: make(chan struct{})}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, 0, fmt.Errorf("the language model is closed")
	}
	s.pending = append(s.pending, u)
	s.wake.Broadcast()
	s.mu.Unlock()
	select {
	case <-u.done:
	case <-j.ctx.Done():
		// A unit still in the queue is withdrawn — it may be waiting behind
		// an interactive request for seconds — and one on the device is
		// waited out, because the loop is writing its result.
		s.mu.Lock()
		if s.running != u {
			s.dropUnit(u)
			select {
			case <-u.done:
			default:
				u.err = j.ctx.Err()
				close(u.done)
			}
		}
		s.mu.Unlock()
		<-u.done
	}
	return u.logits, time.Since(start) - u.busy, u.err
}

// close stops the loop, failing whatever is queued.
func (s *llmSched) close() {
	s.mu.Lock()
	s.closed = true
	s.wake.Broadcast()
	s.mu.Unlock()
	<-s.stopped
}

// loop is the one goroutine that touches the graph.
func (s *llmSched) loop() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		var u *llmUnit
		for {
			if s.closed {
				for _, p := range s.pending {
					p.err = errors.New("the language model is closed")
					close(p.done)
				}
				s.pending = nil
				s.mu.Unlock()
				return
			}
			if u = s.pick(); u != nil {
				break
			}
			s.wake.Wait()
		}
		n := min(len(u.ids), s.batch)
		if u.job.class == classBackground && s.preempt > 0 && len(u.ids) > 1 {
			n = min(n, s.preempt)
		}
		chunk, next := u.ids[:n], u.ids[n:min(len(u.ids), 2*n)]
		slot, fresh := u.job.slot, u.fresh
		s.running = u
		s.mu.Unlock()

		var logits []float32
		t0 := time.Now()
		ran := false
		err := u.job.ctx.Err()
		if err == nil {
			ran = true
			err = s.dev.Do(func(*vk.Device) error {
				if err := s.g.UseSlot(slot); err != nil {
					return err
				}
				if len(next) > 0 {
					// The next chunk's n-gram pages, faulted in while this
					// one runs (P17).
					s.g.PrefetchPLE(chunk, next)
				}
				var err error
				if fresh {
					logits, _, err = s.g.Forward(chunk)
				} else {
					logits, _, err = s.g.Extend(chunk)
				}
				return err
			})
		}
		took := time.Since(t0)

		s.mu.Lock()
		s.running = nil
		sl := &s.slots[slot]
		switch {
		case err != nil && ran:
			// Whatever the slot held is no longer something this can
			// describe; the next request on it starts over.
			sl.held = nil
		case err != nil:
			// Cancelled before the chunk ran: the device is where the held
			// sequence says it is.
		case fresh:
			sl.held = append(sl.held[:0], chunk...)
		default:
			sl.held = append(sl.held, chunk...)
		}
		u.busy += took
		u.job.served += took
		u.ids, u.fresh = u.ids[n:], false
		if err != nil || len(u.ids) == 0 {
			u.logits, u.err = logits, err
			s.dropUnit(u)
			close(u.done)
		}
		s.mu.Unlock()
	}
}

// pick is the scheduling policy: the next unit to run, or nil to wait. See
// the file comment for the three rules.
func (s *llmSched) pick() *llmUnit {
	interactive := false
	for i := range s.slots {
		if j := s.slots[i].job; j != nil && j.class == classInteractive {
			interactive = true
		}
	}
	var best *llmUnit
	for _, u := range s.pending {
		if interactive && u.job.class == classBackground {
			continue
		}
		if best == nil || u.job.served < best.job.served ||
			(u.job.served == best.job.served && u.job.arrival < best.job.arrival) {
			best = u
		}
	}
	if best != nil || !interactive {
		return best
	}
	// Rule 1: an interactive job holds the device between its own units.
	// Only when every interactive slot-holder is gone does background run.
	return nil
}

func (s *llmSched) dropUnit(u *llmUnit) {
	for i, p := range s.pending {
		if p == u {
			s.pending = append(s.pending[:i], s.pending[i+1:]...)
			return
		}
	}
}
