package backend

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeVertical records its loads and unloads into a shared log.
type fakeVertical struct {
	name string
	log  *eventLog
	fail error
}

type eventLog struct {
	mu sync.Mutex
	ev []string
}

func (l *eventLog) add(s string) { l.mu.Lock(); l.ev = append(l.ev, s); l.mu.Unlock() }
func (l *eventLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Join(l.ev, " ")
}

func (f *fakeVertical) SwapName() string { return f.name }
func (f *fakeVertical) Load() error {
	if f.fail != nil {
		return f.fail
	}
	f.log.add("+" + f.name)
	return nil
}
func (f *fakeVertical) Unload() { f.log.add("-" + f.name) }

func hold(t *testing.T, s *Swap, v Swappable) {
	t.Helper()
	release, err := s.Hold(context.Background(), v)
	if err != nil {
		t.Fatal(err)
	}
	release()
}

// TestSwapTakesTurns: a request for another vertical unloads the resident
// one before loading its own, and a request for the resident one loads
// nothing.
func TestSwapTakesTurns(t *testing.T) {
	l := &eventLog{}
	img, music, video := &fakeVertical{"image", l, nil}, &fakeVertical{"music", l, nil}, &fakeVertical{"video", l, nil}
	s := NewSwap(0)
	hold(t, s, img)
	hold(t, s, img)
	hold(t, s, music)
	hold(t, s, video)
	hold(t, s, img)
	s.Close()
	if got, want := l.String(), "+image -image +music -music +video -video +image -image"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if s.Resident() != nil {
		t.Fatal("closed slot still resident")
	}
}

// TestSwapIdle: the slot empties after the idle time, and a request inside
// it restarts the clock.
func TestSwapIdle(t *testing.T) {
	l := &eventLog{}
	img := &fakeVertical{"image", l, nil}
	s := NewSwap(80 * time.Millisecond)
	hold(t, s, img)
	for range 4 { // 4 x 40 ms of requests, each inside the idle time
		time.Sleep(40 * time.Millisecond)
		hold(t, s, img)
	}
	if s.Resident() == nil {
		t.Fatalf("unloaded while in use: %s", l)
	}
	time.Sleep(200 * time.Millisecond)
	if s.Resident() != nil || l.String() != "+image -image" {
		t.Fatalf("not unloaded after idle: %s", l)
	}
	hold(t, s, img)
	if l.String() != "+image -image +image" {
		t.Fatalf("not reloaded: %s", l)
	}
	s.Close()
}

// TestSwapIdleWaitsForTheRequest: the timer never unloads under a running
// request, however long it runs.
func TestSwapIdleWaitsForTheRequest(t *testing.T) {
	l := &eventLog{}
	img := &fakeVertical{"image", l, nil}
	s := NewSwap(30 * time.Millisecond)
	hold(t, s, img)
	release, err := s.Hold(context.Background(), img)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if s.Resident() == nil {
		t.Fatal("unloaded under a running request")
	}
	release()
	time.Sleep(100 * time.Millisecond)
	if s.Resident() != nil {
		t.Fatalf("not unloaded after the request: %s", l)
	}
}

// TestSwapFailedLoad leaves the slot empty and usable.
func TestSwapFailedLoad(t *testing.T) {
	l := &eventLog{}
	img, bad := &fakeVertical{"image", l, nil}, &fakeVertical{"music", l, errors.New("no checkpoint")}
	s := NewSwap(0)
	hold(t, s, img)
	if _, err := s.Hold(context.Background(), bad); err == nil {
		t.Fatal("a failed load held the slot")
	}
	if s.Resident() != nil {
		t.Fatal("slot not empty after a failed load")
	}
	hold(t, s, img)
	if got, want := l.String(), "+image -image +image"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestSwapWaiterGivesUp: a request queued behind a running one leaves when
// its context does.
func TestSwapWaiterGivesUp(t *testing.T) {
	l := &eventLog{}
	img, music := &fakeVertical{"image", l, nil}, &fakeVertical{"music", l, nil}
	s := NewSwap(0)
	release, err := s.Hold(context.Background(), img)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := s.Hold(ctx, music); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	release()
	if l.String() != "+image" {
		t.Fatalf("the abandoned request loaded: %s", l)
	}
}
