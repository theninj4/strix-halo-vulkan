package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Server-Sent Events, which all three chat envelopes are streamed over.
//
// The three disagree about almost everything -- OpenAI's chat endpoint sends
// unnamed frames and ends with a `[DONE]` sentinel, its Responses API names
// every frame and numbers them, Anthropic's names them and does not number
// them -- so what is shared is only this: the headers, the framing, and a
// flush after every frame. A streaming endpoint that does not flush is a
// buffered endpoint with extra syntax.

// sse is one streamed response. Creating it commits the status code, which is
// why every check a handler wants to answer with a status has to happen
// first.
type sse struct {
	w   http.ResponseWriter
	rc  *http.ResponseController
	seq int

	// mu serialises the writes, because the heartbeat writes from a
	// goroutine of its own while the generation writes from the handler's.
	// last is when a byte last went out, and stopped is set once the
	// handler is done with the stream: nothing may be written to a
	// ResponseWriter after its handler returns.
	mu      sync.Mutex
	last    time.Time
	stopped bool
}

func newSSE(w http.ResponseWriter) *sse {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Connection", "keep-alive")
	// Neither a proxy nor this process may hold a frame back: the whole
	// point of the endpoint is that a token is visible when it exists.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	return &sse{w: w, rc: http.NewResponseController(w), last: time.Now()}
}

// keepAlive writes an SSE comment (`: keep-alive`) whenever the stream has
// been silent for every, until the returned stop is called. The caller defers
// stop; once it returns, the heartbeat has written its last byte.
//
// A stream can be silent for minutes while the request is still being worked
// on: waiting for a slot, prefilling 200k tokens, and above all while a tool
// call is being written, since the decoder in `llm` holds a call back until it
// closes (a 9k-token `write` is 5-10 minutes at a background slot's rate).
// Clients and proxies read that silence as a dead connection -- opencode cut
// such streams at 6 and 10 minutes and regenerated the whole turn. A comment
// is the one frame every SSE parser is required to ignore, so it keeps the
// connection visibly alive without being an event in any of the three
// envelopes: it carries no data, takes no sequence number in the Responses
// API, and cannot be mistaken for a chunk.
func (s *sse) keepAlive(every time.Duration) (stop func()) {
	done := make(chan struct{})
	go func() {
		t := time.NewTimer(every)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
			}
			s.mu.Lock()
			idle := time.Since(s.last)
			if s.stopped {
				s.mu.Unlock()
				return
			}
			if idle >= every {
				// A failed write is a client that has gone; the
				// generation notices through the request's context, so
				// there is nothing to do about it here.
				if _, err := fmt.Fprint(s.w, ": keep-alive\n\n"); err == nil {
					_ = s.rc.Flush()
				}
				s.last, idle = time.Now(), 0
			}
			s.mu.Unlock()
			t.Reset(every - idle)
		}
	}()
	return func() {
		s.mu.Lock()
		s.stopped = true
		s.mu.Unlock()
		close(done)
	}
}

// send writes one frame. An empty name writes no `event:` line, which is what
// OpenAI's chat endpoint expects and what the other two never want.
func (s *sse) send(event string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if event != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\n", event); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	s.seq++
	s.last = time.Now()
	return s.rc.Flush()
}

// done writes OpenAI's end-of-stream sentinel, which is not JSON and not an
// event: it is the literal `data: [DONE]`.
func (s *sse) done() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := fmt.Fprint(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	return s.rc.Flush()
}

// next is the sequence number of the frame about to be sent. The Responses
// API numbers its events and clients use the numbers to spot a gap.
func (s *sse) next() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seq
}

// defaultKeepAlive is how long a chat stream may go without a byte before a
// heartbeat is written (Server.KeepAlive). Ten seconds is well inside every
// idle timeout seen in practice -- the shortest common ones are 30 s at a
// proxy and 60 s at a load balancer -- and costs 14 bytes a beat.
const defaultKeepAlive = 10 * time.Second

// keepAliveEvery is the server's heartbeat interval.
func (s *Server) keepAliveEvery() time.Duration {
	if s.KeepAlive > 0 {
		return s.KeepAlive
	}
	return defaultKeepAlive
}
