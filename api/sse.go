package api

import (
	"encoding/json"
	"fmt"
	"net/http"
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
}

func newSSE(w http.ResponseWriter) *sse {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Connection", "keep-alive")
	// Neither a proxy nor this process may hold a frame back: the whole
	// point of the endpoint is that a token is visible when it exists.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	return &sse{w: w, rc: http.NewResponseController(w)}
}

// send writes one frame. An empty name writes no `event:` line, which is what
// OpenAI's chat endpoint expects and what the other two never want.
func (s *sse) send(event string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if event != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\n", event); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	s.seq++
	return s.rc.Flush()
}

// done writes OpenAI's end-of-stream sentinel, which is not JSON and not an
// event: it is the literal `data: [DONE]`.
func (s *sse) done() error {
	if _, err := fmt.Fprint(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	return s.rc.Flush()
}

// next is the sequence number of the frame about to be sent. The Responses
// API numbers its events and clients use the numbers to spot a gap.
func (s *sse) next() int { return s.seq }
