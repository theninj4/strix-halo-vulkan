package backend

import (
	"context"
	"log"
	"runtime/debug"
	"sync"
	"time"
)

// Swappable is a vertical that takes its turn in a Swap: image, music and
// video, the three that hold 20-31 GB each and are asked for rarely.
type Swappable interface {
	// SwapName is the vertical's name, for the log.
	SwapName() string
	// Load stages the weights. It runs without the device lock: staging is
	// allocation, pipeline creation and mapped writes, and none of it
	// submits (TestStagingSkipsTheQueue), so speech keeps the device
	// through a load.
	Load() error
	// Unload frees them. Nothing is running on them when it is called.
	Unload()
}

// Swap is one residency slot that the heavy verticals take turns in: a
// request makes its vertical resident, evicting whichever other one is,
// and runs with the slot held, so at most one of them is ever staged and
// only one of their requests runs at a time. After Idle with no request
// the slot is emptied, so at rest they hold nothing.
//
// The price is a load on a switch: ~28 s for image, ~16 s for music
// (measured by TestStagingSkipsTheQueue); video stages per request anyway
// and has nothing to load.
type Swap struct {
	idle time.Duration
	sem  chan struct{} // held by a running request, a switch, or the idle unload

	mu    sync.Mutex // cur, gen, timer
	cur   Swappable
	gen   int // bumped by every Hold, so a stale idle timer does nothing
	timer *time.Timer
}

// NewSwap is an empty slot. idle 0 never unloads on idle.
func NewSwap(idle time.Duration) *Swap {
	return &Swap{idle: idle, sem: make(chan struct{}, 1)}
}

// Hold waits for the slot, makes v resident, and returns the release, which
// the caller runs when its request is done. A waiting caller gives up when
// ctx does: an image request queued behind a video is a client that may
// hang up. A failed load leaves the slot empty.
func (s *Swap) Hold(ctx context.Context, v Swappable) (release func(), err error) {
	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	s.mu.Lock()
	s.gen++
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	cur := s.cur
	s.mu.Unlock()

	if cur != v {
		if cur != nil {
			s.evict(cur, "for "+v.SwapName())
		}
		start := time.Now()
		if err := v.Load(); err != nil {
			freeMemory()
			<-s.sem
			return nil, err
		}
		freeMemory()
		s.mu.Lock()
		s.cur = v
		s.mu.Unlock()
		log.Printf("swap: %s loaded in %v", v.SwapName(), time.Since(start).Round(time.Millisecond))
	}
	var once sync.Once
	return func() { once.Do(s.release) }, nil
}

// Load makes v resident and returns: cmd/serve stages each vertical once
// at startup, so a bad checkpoint is reported then and not at the first
// request.
func (s *Swap) Load(v Swappable) error {
	release, err := s.Hold(context.Background(), v)
	if err != nil {
		return err
	}
	release()
	return nil
}

func (s *Swap) release() {
	s.mu.Lock()
	if s.idle > 0 && s.cur != nil {
		gen := s.gen
		s.timer = time.AfterFunc(s.idle, func() { s.expire(gen) })
	}
	s.mu.Unlock()
	<-s.sem
}

// expire is the idle timer: it empties the slot unless a request has come
// since it was armed. It does not wait for the slot -- whoever holds it
// re-arms the timer on release.
func (s *Swap) expire(gen int) {
	select {
	case s.sem <- struct{}{}:
	default:
		return
	}
	defer func() { <-s.sem }()
	s.mu.Lock()
	cur := s.cur
	if gen != s.gen || cur == nil {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	s.evict(cur, "idle "+s.idle.String())
}

// evict unloads v, which the caller has checked is resident, with the slot
// held.
func (s *Swap) evict(v Swappable, why string) {
	start := time.Now()
	v.Unload()
	s.mu.Lock()
	s.cur = nil
	s.mu.Unlock()
	freeMemory()
	log.Printf("swap: %s unloaded (%s) in %v", v.SwapName(), why, time.Since(start).Round(time.Millisecond))
}

// Resident is the vertical staged now, or nil.
func (s *Swap) Resident() Swappable {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cur
}

// Close waits for the running request and unloads whatever is resident.
// The job queues must be closed first, or a queued job would take the slot
// back.
func (s *Swap) Close() {
	s.sem <- struct{}{}
	defer func() { <-s.sem }()
	s.mu.Lock()
	s.gen++
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	cur := s.cur
	s.mu.Unlock()
	if cur != nil {
		s.evict(cur, "shutdown")
	}
}

// freeMemory hands a load's or an unload's garbage back to the kernel. A
// staging leaves gigabytes of collected host buffers in the heap, and the
// server's large live heap sets a GC target that would otherwise keep them
// (VIDEO.md M11a): unloading 20 GB of device memory is no use if the heap
// then grows into the room it left.
func freeMemory() {
	debug.FreeOSMemory()
}
