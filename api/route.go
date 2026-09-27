package api

import "context"

// CompletionRoute answers the chat door with several models: the one whose
// own ids include the request's `model`, and the first otherwise (OCR.md
// decision 2: with -llm and -ocr in one process, "PaddleOCR-VL-1.6-0.9B"
// reaches the OCR model and every other name the language model, as before).
//
// The first backend is the default for a reason a client can see: GET
// /v1/models lists its ids first, and a client that picks the first listed
// model -- PaddleOCR's own pipeline does, unless told a name -- gets it.
type CompletionRoute []CompletionBackend

// Models is every backend's ids, in order.
func (r CompletionRoute) Models() []Model {
	var out []Model
	for _, b := range r {
		out = append(out, b.Models()...)
	}
	return out
}

// Complete runs the request on the backend that owns its model.
func (r CompletionRoute) Complete(ctx context.Context, req *CompletionRequest, emit func(Delta) error) (*CompletionResult, error) {
	return r.For(req.Model).Complete(ctx, req, emit)
}

// For is the backend a request naming model reaches.
func (r CompletionRoute) For(model string) CompletionBackend {
	for _, b := range r {
		if owns(b, model) {
			return b
		}
	}
	return r[0]
}
