package api

// POST /v1/decisions (research/rune-vertical.md R6): surogate's decisions v1,
// OpenRouter's field names, as Rune is served. `/api/alpha/decisions`
// (OpenRouter's path) and `/api/v1/decisions` are the same endpoint.
//
// As with /v1/systemone the body crosses into the backend as bytes: its key
// order, number spellings and repeated keys decide the prompt, so it is
// parsed by `decide.Parse` with Python's reading of JSON, not here. What
// stays here is the transport and v1's error envelope, which carries a
// `param` and a `code` (`invalid_decisions_request`, `vision_disabled`, ...).

import (
	"context"
	"errors"
	"io"
	"net/http"

	"strix-halo-vulkan/decide"
)

// DecisionsBackend answers decisions requests.
type DecisionsBackend interface {
	Backend
	// Decisions parses and answers one request body, returning the response
	// body. A *decide.Error is the client's (a 400); ErrUnknownModel a 404;
	// anything else is the server's.
	Decisions(ctx context.Context, body []byte) ([]byte, error)
}

// ErrUnknownModel is a decisions request naming a model this server does not
// serve.
var ErrUnknownModel = errors.New("unknown model")

type decisionsError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

func (s *Server) handleDecisions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Decisions == nil {
		notLoaded(ctx, w, "the decision model", "-rune")
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		if tooLarge(ctx, w, err) {
			return
		}
		badRequest(ctx, w, "reading the body: "+err.Error())
		return
	}
	out, err := s.Decisions.Decisions(ctx, body)
	var de *decide.Error
	switch {
	case err == nil:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(out)
	case errors.As(err, &de):
		logf(ctx, "400: %s", de.Message)
		writeJSON(w, http.StatusBadRequest, map[string]decisionsError{"error": {
			Message: de.Message, Type: "invalid_request_error", Param: de.Param, Code: de.Code}})
	case errors.Is(err, ErrUnknownModel):
		logf(ctx, "404: %v", err)
		writeJSON(w, http.StatusNotFound, map[string]decisionsError{"error": {
			Message: err.Error(), Type: "invalid_request_error", Param: "model", Code: "model_not_found"}})
	case errors.Is(err, context.Canceled):
		logf(ctx, "decisions: client cancelled")
	default:
		serverError(ctx, w, "decisions", err)
	}
}
