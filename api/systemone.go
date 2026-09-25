package api

// POST /v1/systemone (CLASSIFICATION.md K6): TypeSafe's System One contract,
// as Kev serves it. One text (the state) and typed questions in, one
// calibrated probability distribution per question out, in one prefill.
//
// Unlike every other endpoint here the envelope is not OpenAI's, and it is
// not decoded in this package. The request is model-shaped all the way down:
// an object state's key order and a number's int-ness decide the text the
// model reads, and a choice's criteria order decides its option slots, so the
// body crosses the boundary as bytes and the backend parses it with the
// rules the model was trained under. What stays here is the transport: the
// body cap, auth, the request id TypeSafe's clients read, and the status.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"net/http"

	"strix-halo-vulkan/util"
)

// SystemOneBackend answers System One requests.
type SystemOneBackend interface {
	Backend
	// SystemOne parses and answers one request body, returning the response
	// body. An error wrapping ErrUnprocessable is the client's (a 422, as
	// TypeSafe and Kev answer it); anything else is the server's.
	SystemOne(ctx context.Context, body []byte) ([]byte, error)
}

// ErrUnprocessable is a well-formed request the model cannot answer as asked:
// a missing field, an unknown question type, too many options, a branch
// longer than a row.
var ErrUnprocessable = errors.New("unprocessable request")

func (s *Server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.Header.Get("X-Typesafe-Request-Id")
	if id == "" {
		id = util.RequestID(ctx)
	}
	if id == "" {
		var b [16]byte
		_, _ = rand.Read(b[:])
		id = hex.EncodeToString(b[:])
	}
	w.Header().Set("X-Typesafe-Request-Id", id)

	if s.SystemOne == nil {
		notLoaded(ctx, w, "the classification model", "-kev")
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
	out, err := s.SystemOne.SystemOne(ctx, body)
	if err != nil {
		if errors.Is(err, ErrUnprocessable) {
			logf(ctx, "422: %v", err)
			writeError(w, http.StatusUnprocessableEntity, "invalid_request_error", err.Error())
			return
		}
		if errors.Is(err, context.Canceled) {
			logf(ctx, "systemone: client cancelled")
			return
		}
		serverError(ctx, w, "systemone", err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}
