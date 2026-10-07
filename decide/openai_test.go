package decide

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestOpenAIRendersAsV1 is the translation's gate: an OpenAI-shape request
// shows the model exactly what the v1 request it stands for does, state
// text, system prompts, labels and branches.
func TestOpenAIRendersAsV1(t *testing.T) {
	oa, err := ParseOpenAI([]byte(`{"model": "gpt-6-luna",
		"input": [{"role": "user", "content": [
			{"type": "input_text", "text": "I was charged twice."},
			{"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"},
			{"type": "input_text", "text": "Order 1042."}]}],
		"questions": [
			{"type": "predicate", "name": "refund", "instructions": "Does the customer want a refund?"},
			{"type": "choice", "name": "department", "instructions": "Which department?",
			 "choices": [{"value": "billing", "description": "Payments and refunds."}, {"value": "technical"}]},
			{"type": "score", "instructions": "How urgent?",
			 "levels": [{"label": "Low"}, {"label": "High", "description": "Money is at stake."}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	v1, err := Parse([]byte(`{"model": "rune", "state": "I was charged twice.\n\nOrder 1042.",
		"images": ["data:image/png;base64,AAAA"],
		"questions": {
			"refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
			"department": {"type": "choice", "instructions": "Which department?",
			 "criteria": {"billing": "Payments and refunds.", "technical": "technical"}},
			"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Money is at stake."]}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if oa.StateText != v1.StateText || !reflect.DeepEqual(oa.Images, v1.Images) {
		t.Fatalf("state %q images %v, want %q %v", oa.StateText, oa.Images, v1.StateText, v1.Images)
	}
	for i := range v1.Questions {
		got, _ := Render(&oa.Questions[i], nil)
		want, _ := Render(&v1.Questions[i], nil)
		if !reflect.DeepEqual(got, want) || oa.Questions[i].Type != v1.Questions[i].Type {
			t.Errorf("question %d renders %+v, want %+v", i, got, want)
		}
	}
	if *oa.Names[0] != "refund" || oa.Names[2] != nil {
		t.Errorf("names %v", oa.Names)
	}
}

// TestOpenAIResponse pins the answer objects' shape against the guide's
// examples: typed choice values, score levels by index and label, a null
// name, and usage with nothing generated.
func TestOpenAIResponse(t *testing.T) {
	r, err := ParseOpenAI([]byte(`{"model": "gpt-6-luna", "input": "x", "questions": [
		{"type": "predicate", "name": "p", "instructions": "?"},
		{"type": "choice", "name": "c", "instructions": "?", "choices": [{"value": true}, {"value": "true"}]},
		{"type": "score", "instructions": "?", "levels": [{"label": "None"}, {"label": "Some"}, {"label": "Lots"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	logits := [][]float32{{0, 2}, {1, 0}, {0, 1, 0}}
	answers := make([]Answer, len(logits))
	for i := range logits {
		if answers[i], err = Resolve(&r.Questions[i], logits[i], 1); err != nil {
			t.Fatal(err)
		}
	}
	var got struct {
		Model   string
		Answers []map[string]any
		Usage   map[string]any
	}
	if err := json.Unmarshal(OpenAIResponse(r, "rune", answers, 42), &got); err != nil {
		t.Fatal(err)
	}
	p, c, s := got.Answers[0], got.Answers[1], got.Answers[2]
	if p["type"] != "predicate" || p["name"] != "p" || p["probability"] != answers[0].Value {
		t.Errorf("predicate %v", p)
	}
	if c["type"] != "choice" || c["choice"] != true || c["confidence"] != answers[1].Confidence {
		t.Errorf("choice %v", c)
	}
	probs := c["probabilities"].([]any)
	if probs[0].(map[string]any)["value"] != true || probs[1].(map[string]any)["value"] != "true" {
		t.Errorf("choice probabilities %v", probs)
	}
	if v, ok := s["name"]; !ok || v != nil || s["type"] != "score" || s["score"] != 1.0 {
		t.Errorf("score %v", s)
	}
	level := s["probabilities"].([]any)[2].(map[string]any)
	if level["value"] != 2.0 || level["label"] != "Lots" {
		t.Errorf("score level %v", level)
	}
	if got.Model != "rune" || got.Usage["input_tokens"] != 42.0 || got.Usage["output_tokens"] != 0.0 || got.Usage["total_tokens"] != 42.0 {
		t.Errorf("model %q usage %v", got.Model, got.Usage)
	}
}

func TestOpenAIRefusals(t *testing.T) {
	for _, c := range []struct{ body, param, msg string }{
		{`{"input": "x", "questions": [{"type": "predicate", "instructions": "?"}]}`, "model", "model"},
		{`{"model": "m", "questions": [{"type": "predicate", "instructions": "?"}]}`, "input", "input"},
		{`{"model": "m", "input": "x", "questions": []}`, "questions", "at least one"},
		{`{"model": "m", "input": "x", "questions": [{"type": "noul", "instructions": "?"}]}`, "questions[0].type", "predicate"},
		{`{"model": "m", "input": "x", "questions": [{"type": "predicate"}]}`, "questions[0].instructions", "instructions"},
		{`{"model": "m", "input": "x", "questions": [{"type": "choice", "instructions": "?", "choices": [{"value": "a"}, {"value": "a"}]}]}`,
			"questions[0].choices[1].value", "twice"},
		{`{"model": "m", "input": [{"role": "system", "content": "x"}], "questions": [{"type": "predicate", "instructions": "?"}]}`,
			"input[0].role", "user"},
		{`{"model": "m", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}],
			"questions": [{"type": "predicate", "instructions": "?"}]}`, "input[0].content[0].image_url", "data URL"},
	} {
		_, err := ParseOpenAI([]byte(c.body))
		var de *Error
		if !errors.As(err, &de) || de.Param != c.param || de.Code != "" || !strings.Contains(de.Message, c.msg) {
			t.Errorf("%s: got %v", c.body, err)
		}
	}
}
