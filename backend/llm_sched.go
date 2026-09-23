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
//
// **Decode steps of different conversations share a pass** (C5). When the
// unit picked is a decode step, every other pending decode step of the same
// class rides with it, a row each (llm.Graph.DecodeRows): three rows cost
// ~1.8 steps, so three streams get ~19 tok/s each instead of ~11. After a
// batched step every conversation samples its token and submits the next
// step within a millisecond or so, so the loop waits up to `coalesce` for
// the ones still sampling rather than running the first one back alone.
// Background rows do not ride in an interactive pass: a rider costs the
// voice stream a third of its rate (C5a), and rule 1 already says who waits.
//
// **Each slot also keeps a checkpoint** (C4): its carried state at the end of
// everything before the last user turn of the last request it ran. The next
// voice command on the same system prompt, or the next turn of a
// conversation whose re-rendered history no longer matches what was
// generated, restores it and prefills only what follows. The prefill chunk
// is cut at that boundary so the checkpoint can be taken there.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"slices"
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
	// ckpt is the slot's checkpoint, or nil. It is usable only while held
	// still starts with its tokens: that is what says the cache cells below
	// it are still the ones it was taken over (llm.Graph.Restore).
	ckpt *llm.Checkpoint
}

// reuse is how much of a prompt the slot saves, and whether that is by
// restoring its checkpoint rather than continuing what it holds.
func (sl *llmSlot) reuse(prompt []int32) (int, bool) {
	n := usablePrefix(sl.held, prompt)
	if c := sl.ckpt; c != nil {
		b := c.Past()
		if b > n && b < len(prompt) && commonPrefix(sl.held, c.Ids()) == b && commonPrefix(prompt, c.Ids()) == b {
			return b, true
		}
	}
	return n, false
}

// llmJob is one request's claim on the scheduler.
type llmJob struct {
	ctx   context.Context
	class llmClass
	slot  int
	// reused is how many leading tokens of the prompt the slot already held,
	// and restore whether that is its checkpoint rather than its sequence.
	reused  int
	restore bool
	// mark is where to take the slot's checkpoint during the prefill, as a
	// position in the prompt, or 0 for nowhere.
	mark int
	// arrival breaks ties, oldest first.
	arrival uint64
	// served is the device time its units have taken, which is what rule 2
	// shares out.
	served time.Duration
	// prompt is what the slot is chosen against: the slot already holding
	// the longest prefix of it wins.
	prompt  []int32
	granted chan struct{}
	// stepping is set by its first decode step: from then on, a moment with
	// no unit queued is a moment it is sampling, and a batched step is worth
	// waiting a little for it.
	stepping bool
	// progress is what the periodic progress line reads (llm_progress.go).
	progress jobProgress
}

// llmUnit is a pending run of tokens for one job: a prefill, which the
// scheduler cuts into chunks, or a decode step.
type llmUnit struct {
	job   *llmJob
	ids   []int32
	fresh bool
	// restore takes the slot back to its checkpoint before the first chunk.
	restore bool
	// step is a decode step, which may share a pass with other slots' (C5).
	step bool
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
	// take out of the queue from under the loop, and inflight the units of a
	// batched step on it.
	running  *llmUnit
	inflight []*llmUnit
	// coalesce is how long a decode step waits for the other decoding
	// conversations of its class to submit theirs, measured from the end of
	// the last step (C5). lastStep is that end.
	coalesce time.Duration
	lastStep time.Time
	// noBatch is LLMOptions.NoBatchDecode.
	noBatch bool
	arrival uint64
	closed  bool
	stopped chan struct{}
	// restores counts checkpoint restores, and batches and batchRows the
	// batched decode passes and the rows they carried, for a test that has
	// to know one happened rather than infer it from a matching answer.
	restores, batches, batchRows int
}

func newLLMSched(g *llm.Graph, dev *Device, batch, preempt, reserve int, noBatch bool, progress time.Duration) *llmSched {
	s := &llmSched{
		g: g, dev: dev, batch: batch, preempt: preempt, reserve: reserve, noBatch: noBatch,
		slots: make([]llmSlot, g.Slots()), stopped: make(chan struct{}),
		coalesce: 5 * time.Millisecond,
	}
	s.wake = sync.NewCond(&s.mu)
	go s.loop()
	if progress > 0 {
		go s.reportProgress(progress)
	}
	return s
}

// acquire waits for a slot and claims it. The job it returns must be
// released, whatever happens after.
func (s *llmSched) acquire(ctx context.Context, class llmClass, ids []int32, mark int) (*llmJob, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, fmt.Errorf("the language model is closed")
	}
	s.arrival++
	j := &llmJob{ctx: ctx, class: class, slot: -1, arrival: s.arrival, prompt: ids, mark: mark,
		granted: make(chan struct{}), progress: jobProgress{enter: time.Now()}}
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

// freeSlot picks the slot for a job of this class: the one that saves the
// most of its prompt (a held prefix or a checkpoint), else the least recently
// used.
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
		reuse, _ := sl.reuse(prompt)
		if best < 0 || reuse > bestReuse || (reuse == bestReuse && s.cheaper(sl, &s.slots[best])) {
			best, bestReuse = i, reuse
		}
	}
	return best
}

// cheaper is whether a is the better slot to overwrite than b, when neither
// saves the new job anything: the one with less checkpointed, then the least
// recently used. Without the first rule, background agents arriving after a
// voice command take its slot by LRU and the next command re-reads the whole
// system prompt (C4).
func (s *llmSched) cheaper(a, b *llmSlot) bool {
	if ca, cb := ckptLen(a), ckptLen(b); ca != cb {
		return ca < cb
	}
	return a.used.Before(b.used)
}

func ckptLen(sl *llmSlot) int {
	if sl.ckpt == nil {
		return 0
	}
	return sl.ckpt.Past()
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
	j.reused, j.restore = s.slots[slot].reuse(j.prompt)
	// A newcomer starts level with the least-served job of its class rather
	// than at zero, or it would hold the device until it had caught up with
	// a conversation that has been running for a minute.
	first := true
	for i := range s.slots {
		if o := s.slots[i].job; o != nil && o.class == j.class && (first || o.served < j.served) {
			j.served, first = o.served, false
		}
	}
	p := &j.progress
	p.granted, p.prefilled = time.Now(), j.reused
	p.lastAt, p.lastPrefilled, p.lastServed = p.granted, j.reused, j.served
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
	return s.submit(&llmUnit{job: j, ids: ids, fresh: fresh, done: make(chan struct{})})
}

// step is run for one decode token, which the loop may batch with other
// conversations' steps.
func (s *llmSched) step(j *llmJob, id int32) ([]float32, time.Duration, error) {
	return s.submit(&llmUnit{job: j, ids: []int32{id}, step: true, done: make(chan struct{})})
}

func (s *llmSched) submit(u *llmUnit) ([]float32, time.Duration, error) {
	start := time.Now()
	j := u.job
	s.mu.Lock()
	if u.step && !j.stepping {
		j.stepping = true
		j.progress.decoding = time.Now()
	}
	u.restore, j.restore = j.restore, false
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
		if s.running != u && !slices.Contains(s.inflight, u) {
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
		var steps []*llmUnit
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
				if !u.step {
					break
				}
				steps = s.steps(u)
				wait := s.coalesceWait(u, len(steps))
				if wait <= 0 {
					break
				}
				time.AfterFunc(wait, func() {
					s.mu.Lock()
					s.wake.Broadcast()
					s.mu.Unlock()
				})
			}
			s.wake.Wait()
		}
		if len(steps) > 1 {
			s.runSteps(steps)
			continue
		}
		n := min(len(u.ids), s.batch)
		if u.job.class == classBackground && s.preempt > 0 && len(u.ids) > 1 {
			n = min(n, s.preempt)
		}
		slot, fresh, restore := u.job.slot, u.fresh, u.restore
		sl := &s.slots[slot]
		var ck *llm.Checkpoint
		pos := len(sl.held)
		switch {
		case fresh:
			pos = 0
		case restore:
			ck = sl.ckpt
			pos = ck.Past()
		}
		// The chunk that reaches the checkpoint's boundary ends there.
		mark := u.job.mark
		if mark > pos && mark < pos+n {
			n = mark - pos
		}
		take := mark > 0 && pos+n == mark
		chunk, next := u.ids[:n], u.ids[n:min(len(u.ids), 2*n)]
		keep := sl.ckpt
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
				if restore {
					if err := s.g.Restore(ck); err != nil {
						return err
					}
				}
				var err error
				if fresh {
					logits, _, err = s.g.Forward(chunk)
				} else {
					logits, _, err = s.g.Extend(chunk)
				}
				if err == nil && take {
					// A checkpoint that cannot be taken costs the next
					// request its prefix, not this one its answer.
					if keep, err = s.g.Checkpoint(keep); err != nil {
						log.Printf("llm: slot %d: no checkpoint at %d: %v", slot, mark, err)
						keep, err = nil, nil
					}
				}
				return err
			})
		}
		took := time.Since(t0)

		s.mu.Lock()
		s.running = nil
		switch {
		case err != nil && ran:
			// Whatever the slot held is no longer something this can
			// describe; the next request on it starts over.
			sl.held, sl.ckpt = nil, nil
		case err != nil:
			// Cancelled before the chunk ran: the device is where the held
			// sequence says it is.
		case fresh:
			sl.held = append(sl.held[:0], chunk...)
		case restore:
			sl.held = append(sl.held[:ck.Past()], chunk...)
			s.restores++
		default:
			sl.held = append(sl.held, chunk...)
		}
		if err == nil && take {
			sl.ckpt = keep
		}
		if u.step {
			s.lastStep = time.Now()
		}
		u.busy += took
		u.job.served += took
		if err == nil {
			if u.step {
				u.job.progress.steps++
			} else {
				u.job.progress.prefilled += n
			}
		}
		u.ids, u.fresh, u.restore = u.ids[n:], false, false
		if err != nil || len(u.ids) == 0 {
			u.logits, u.err = logits, err
			s.dropUnit(u)
			close(u.done)
		}
		s.mu.Unlock()
	}
}

// steps is the batch a picked decode step runs in: it and every other
// pending decode step of its class, up to one a slot (C5).
func (s *llmSched) steps(u *llmUnit) []*llmUnit {
	b := []*llmUnit{u}
	if s.noBatch {
		return b
	}
	for _, p := range s.pending {
		if p != u && p.step && p.job.class == u.job.class {
			b = append(b, p)
		}
	}
	return b
}

// coalesceWait is how much longer a batch of n steps should wait for the
// conversations of its class that are decoding but sampling right now, or 0
// to run it as it is.
func (s *llmSched) coalesceWait(u *llmUnit, n int) time.Duration {
	if s.noBatch {
		return 0
	}
	peers := 0
	for i := range s.slots {
		if j := s.slots[i].job; j != nil && j.class == u.job.class && j.stepping {
			peers++
		}
	}
	if n >= peers {
		return 0
	}
	return time.Until(s.lastStep.Add(s.coalesce))
}

// runSteps runs a batch of decode steps on different slots as one pass. It is
// called with the lock held and returns with it released.
func (s *llmSched) runSteps(batch []*llmUnit) {
	// A conversation that hung up while queued is not run.
	live := batch[:0]
	for _, u := range batch {
		if err := u.job.ctx.Err(); err != nil {
			u.err = err
			s.dropUnit(u)
			close(u.done)
			continue
		}
		live = append(live, u)
	}
	batch = live
	if len(batch) == 0 {
		s.mu.Unlock()
		return
	}
	slots := make([]int, len(batch))
	ids := make([]int32, len(batch))
	for i, u := range batch {
		slots[i], ids[i] = u.job.slot, u.ids[0]
	}
	s.inflight = batch
	s.batches++
	s.batchRows += len(batch)
	s.mu.Unlock()

	var logits []float32
	t0 := time.Now()
	err := s.dev.Do(func(*vk.Device) error {
		var err error
		logits, err = s.g.DecodeRows(slots, ids)
		return err
	})
	took := time.Since(t0)

	s.mu.Lock()
	s.inflight = nil
	vocab := 0
	if err == nil {
		vocab = len(logits) / len(batch)
	}
	for i, u := range batch {
		sl := &s.slots[u.job.slot]
		if err != nil {
			// Which rows' states moved is not something this can say.
			sl.held, sl.ckpt = nil, nil
		} else {
			sl.held = append(sl.held, u.ids[0])
			u.logits = logits[i*vocab : (i+1)*vocab]
			u.job.progress.steps++
		}
		// Each row had the whole pass's latency, and a share of its cost.
		u.busy += took
		u.job.served += took / time.Duration(len(batch))
		u.err = err
		u.ids = nil
		s.dropUnit(u)
		close(u.done)
	}
	s.lastStep = time.Now()
	s.mu.Unlock()
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
