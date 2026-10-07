package api

// POST /v1/decisions is OpenAI's Decisions API, translated onto surogate's
// decisions v1 in decide/openai.go so Rune sees the prompt it was trained on
// (research/rune-vertical.md).
//
// As with /v1/systemone the body crosses into the backend as bytes and is
// parsed by the decide package. What stays here is the transport and
// OpenAI's error envelope, whose `param` and `code` are null when unset.

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
	// body. A *decide.Error is the client's (a 400); anything else is the
	// server's. Any model name is answered.
	Decisions(ctx context.Context, body []byte) ([]byte, error)
}

type decisionsError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// nullable is "" as JSON null, as OpenAI writes an absent param or code.
func nullable(s string) *string {
	if s == "" {
		return nil
	}
	return &s
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
			Message: de.Message, Type: "invalid_request_error", Param: nullable(de.Param), Code: nullable(de.Code)}})
	case errors.Is(err, context.Canceled):
		logf(ctx, "decisions: client cancelled")
	default:
		serverError(ctx, w, "decisions", err)
	}
}
