package api

import (
	"crypto/rand"
	"encoding/hex"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// POST /v1/chat/completions, buffered and streamed (LLM.md L9a).
//
// The two paths are one generation. `Complete` hands a token to a callback as
// it is produced and the two handlers differ only in what they do with it:
// the streamed one writes an SSE frame, the buffered one appends to a string
// builder. That is deliberate -- a server whose streaming path is a separate
// implementation is a server whose two answers to the same prompt eventually
// differ, and the difference always shows up as a whitespace or a marker that
// one path strips and the other does not.
//
// What this endpoint refuses is in the request validation below, and it is
// all the same refusal: it answers for what it did, not for what the field
// asked for. A `min_p` accepted and ignored is worse than a 400, because the
// client never learns that the knob it turned does nothing.

// handleChatCompletions is the OpenAI chat endpoint.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if s.Completion == nil {
		notLoaded(w, "the language model", "-llm")
		return
	}
	var req CompletionRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if !validCompletion(w, &req) {
		return
	}

	id := "chatcmpl-" + randomID()
	model := modelID(s.Completion, req.Model)
	created := time.Now().Unix()
	if req.Stream {
		s.streamCompletion(w, r, &req, id, model, created)
		return
	}

	var content, reasoning strings.Builder
	res, err := s.Completion.Complete(r.Context(), &req, func(d Delta) error {
		content.WriteString(d.Content)
		reasoning.WriteString(d.ReasoningContent)
		return nil
	})
	if err != nil {
		backendError(w, "chat completion", err)
		return
	}
	msg := &Delta{
		Role:             "assistant",
		ReasoningContent: reasoning.String(),
		Content:          content.String(),
		ToolCalls:        res.ToolCalls,
	}
	usage := res.Usage
	writeJSON(w, http.StatusOK, CompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []Choice{{Index: 0, Message: msg, FinishReason: &res.FinishReason}},
		Usage:   &usage,
	})
}

// streamCompletion writes the generation as Server-Sent Events.
//
// **The status is committed before the first token**, which is what makes the
// error handling here different from every other handler in this package: by
// the time a run fails, 200 has been sent and there is no status left to
// change. So a failure becomes a final SSE frame carrying the error envelope
// -- the shape a client already parses -- and the log gets the rest.
func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request,
	req *CompletionRequest, id, model string, created int64,
) {
	stream := newSSE(w)
	frame := func(v any) error { return stream.send("", v) }
	chunk := func(d *Delta, finish *string) Chunk {
		return Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []Choice{{Index: 0, Delta: d, FinishReason: finish}},
		}
	}

	// The role arrives on its own, before any content: it is what tells a
	// client which message the deltas that follow belong to.
	if err := frame(chunk(&Delta{Role: "assistant"}, nil)); err != nil {
		log.Printf("api: chat completion: %v", err)
		return
	}

	res, err := s.Completion.Complete(r.Context(), req, func(d Delta) error {
		return frame(chunk(&d, nil))
	})
	if err != nil {
		if r.Context().Err() != nil {
			// The client hung up mid-generation. There is nobody to tell.
			log.Printf("api: chat completion: client cancelled")
			return
		}
		log.Printf("api: chat completion: %v", err)
		_ = frame(errorResponse{Error: errorBody{Message: err.Error(), Type: "server_error"}})
		return
	}

	last := &Delta{}
	if len(res.ToolCalls) > 0 {
		last.ToolCalls = res.ToolCalls
	}
	if err := frame(chunk(last, &res.FinishReason)); err != nil {
		log.Printf("api: chat completion: %v", err)
		return
	}
	// OpenAI's usage chunk: an extra frame with no choices, and only when
	// the client asked for it.
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		usage := res.Usage
		if err := frame(Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []Choice{}, Usage: &usage,
		}); err != nil {
			log.Printf("api: chat completion: %v", err)
			return
		}
	}
	_ = stream.done()
}

// validCompletion answers the client itself on a request this server will not
// serve, and reports whether the handler should carry on.
func validCompletion(w http.ResponseWriter, req *CompletionRequest) bool {
	if len(req.Messages) == 0 {
		badRequest(w, "messages is empty")
		return false
	}
	for i, m := range req.Messages {
		if kind := m.Content.NonText(); kind != "" {
			badRequest(w, "message "+strconv.Itoa(i)+" carries a "+strconv.Quote(kind)+
				" content block; this server has no vision model and would answer about the text alone")
			return false
		}
	}
	if req.N > 1 {
		badRequest(w, "n is "+strconv.Itoa(req.N)+
			"; this server generates one completion per request, and n of them would be n runs of a model sized to saturate the device")
		return false
	}
	if len(req.Stop) > maxStopSequences {
		badRequest(w, strconv.Itoa(len(req.Stop))+" stop sequences; this server takes at most "+
			strconv.Itoa(maxStopSequences))
		return false
	}
	for _, seq := range req.Stop {
		if seq == "" {
			badRequest(w, "a stop sequence is empty, which every generation matches immediately")
			return false
		}
	}
	if req.MinP != nil || req.RepeatPenalty != nil {
		badRequest(w, "min_p and repeat_penalty are not implemented; this server's sampler is temperature, top_k and top_p")
		return false
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type != "" && req.ResponseFormat.Type != "text" {
		badRequest(w, "response_format "+strconv.Quote(req.ResponseFormat.Type)+
			" is not implemented; there is no constrained decoding in this server yet")
		return false
	}
	return true
}

// maxStopSequences bounds `stop`. Every one of them is a string search per
// token and a tail held back from the stream, and four is what OpenAI's own
// API takes.
const maxStopSequences = 4

// modelID is what the response reports it ran. It is the backend's own id
// rather than the one the request asked for: a client configured for "gpt-4"
// against this server got this model, and saying otherwise would make a
// transcript unreadable a week later.
func modelID(b Backend, _ string) string {
	if ms := b.Models(); len(ms) > 0 {
		return ms[0].ID
	}
	return "unknown"
}

// randomID is the completion's id. It is random rather than sequential
// because it ends up in client-side logs beside other servers' ids and
// nothing should be inferable from it.
func randomID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand does not fail on this platform; if it ever does, a
		// timestamp is still unique enough for a log line.
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(b[:])
}
