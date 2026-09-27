package api

import (
	"encoding/json"
	"net/http"
	"testing"
)

// namedLLM is a fakeLLM answering to another id: the second model of a
// routed chat door.
type namedLLM struct {
	fakeLLM
	id string
}

func (n *namedLLM) Models() []Model { return []Model{{ID: n.id, Object: "model", OwnedBy: "local"}} }

// TestCompletionRoute sends the OCR model's name to the OCR backend and every
// other name to the first, reports the model that ran, lists the first
// first (PaddleOCR picks data[0] when not told a name), and carries vLLM's
// extras through to the backend.
func TestCompletionRoute(t *testing.T) {
	llm := &fakeLLM{pieces: []Delta{{Content: "hi"}}}
	ocr := &namedLLM{fakeLLM: fakeLLM{pieces: []Delta{{Content: "text"}}}, id: "PaddleOCR-VL-1.6-0.9B"}
	s := &Server{Completion: CompletionRoute{llm, ocr}}

	for _, c := range []struct {
		model, want string
		got         *fakeLLM
	}{
		{"PaddleOCR-VL-1.6-0.9B", "PaddleOCR-VL-1.6-0.9B", &ocr.fakeLLM},
		{"gpt-4", "qwen3.8-flash-next", llm},
		{"qwen3.8-flash-next", "qwen3.8-flash-next", llm},
	} {
		llm.req, ocr.req = nil, nil
		body := chatBody(user("x"))
		body["model"] = c.model
		body["repetition_penalty"] = 1.05
		body["skip_special_tokens"] = false
		body["mm_processor_kwargs"] = map[string]any{"min_pixels": 112896, "max_pixels": 2822400}
		rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", c.model, rec.Code, rec.Body)
		}
		var resp CompletionResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Model != c.want {
			t.Errorf("%s: the response says %q ran, want %q", c.model, resp.Model, c.want)
		}
		if c.got.req == nil {
			t.Fatalf("%s: routed to the wrong backend", c.model)
		}
		r := c.got.req
		if r.RepetitionPenalty == nil || *r.RepetitionPenalty != 1.05 || r.SkipSpecialTokens == nil || *r.SkipSpecialTokens ||
			r.MMProcessorKwargs == nil || r.MMProcessorKwargs.MinPixels != 112896 || r.MMProcessorKwargs.MaxPixels != 2822400 {
			t.Errorf("%s: vLLM's extras did not arrive: %+v", c.model, r)
		}
	}

	rec := do(t, s, jsonRequest("GET", "/v1/models", nil))
	var models ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &models); err != nil {
		t.Fatal(err)
	}
	if len(models.Data) != 2 || models.Data[0].ID != "qwen3.8-flash-next" || models.Data[1].ID != "PaddleOCR-VL-1.6-0.9B" {
		t.Errorf("models: %+v", models.Data)
	}
}
