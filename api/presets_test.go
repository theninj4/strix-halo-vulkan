package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestPresetsParseTheShippedFile reads a copy of the root models.ini, which
// spells its keys both ways (`top_p` in two sections, `top-p` in the others).
func TestPresetsParseTheShippedFile(t *testing.T) {
	ps, err := LoadPresets("testdata/models.ini")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, p := range ps {
		names = append(names, p.Name)
	}
	if strings.Join(names, ",") != "chatting,instruct,thinking,coding" {
		t.Fatalf("sections %v", names)
	}
	for _, p := range ps {
		if p.Temperature == nil || p.TopP == nil || p.TopK == nil || p.MinP == nil ||
			p.RepeatPenalty == nil || p.PresencePenalty == nil {
			t.Errorf("[%s] a sampling key was dropped: %+v", p.Name, p)
		}
	}
	chat, coding := ps[0], ps[3]
	if *chat.TopP != 0.80 || *chat.PresencePenalty != 1.5 || *coding.TopP != 0.95 {
		t.Errorf("values: chatting top_p %v presence %v, coding top_p %v",
			*chat.TopP, *chat.PresencePenalty, *coding.TopP)
	}
	think := func(p Preset) Thinking {
		th, err := (&CompletionRequest{ChatTemplateKwargs: p.Kwargs}).Thinking()
		if err != nil {
			t.Fatal(err)
		}
		return th
	}
	if th := think(chat); !th.Off || th.DropHistory {
		t.Errorf("chatting thinks: %+v", th)
	}
	if th := think(coding); th.Off || th.Effort != "xhigh" {
		t.Errorf("coding: %+v", th)
	}
}

func TestPresetsRefuse(t *testing.T) {
	for name, ini := range map[string]string{
		"an unknown key":         "[a]\nctx-size = 4096\n",
		"a key outside sections": "temperature = 1\n",
		"a duplicate section":    "[a]\n[a]\n",
		"a bad number":           "[a]\ntop_k = lots\n",
		"reasoning against the kwargs": "[a]\nchat-template-kwargs = {\"enable_thinking\":true}\n" +
			"reasoning = off\n",
		"an unknown kwarg": "[a]\nchat-template-kwargs = {\"add_vision_id\":true}\n",
	} {
		if _, err := ParsePresets(strings.NewReader(ini)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestPresetsAreDefaults: the preset fills what the request left out and
// nothing else, and the response names the preset.
func TestPresetsAreDefaults(t *testing.T) {
	ps, err := LoadPresets("testdata/models.ini")
	if err != nil {
		t.Fatal(err)
	}
	b := &fakeLLM{pieces: []Delta{{Content: "hi"}}}
	s := &Server{Completion: b, Presets: ps}
	body := chatBody(user("hi"))
	body["model"] = "chatting"
	body["temperature"] = 0
	body["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if *b.req.Temperature != 0 {
		t.Errorf("the preset overrode the request's temperature: %v", *b.req.Temperature)
	}
	if b.req.TopP == nil || *b.req.TopP != 0.8 || b.req.PresencePenalty == nil || *b.req.PresencePenalty != 1.5 {
		t.Errorf("the preset's defaults did not arrive: top_p %v presence %v", b.req.TopP, b.req.PresencePenalty)
	}
	if th, _ := b.req.Thinking(); th.Off {
		t.Errorf("the request asked to think and the preset turned it off")
	}
	var got CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "chatting" {
		t.Errorf("model %q, want the preset's name", got.Model)
	}

	// A name that is not a preset gets the checkpoint's defaults.
	body = chatBody(user("hi"))
	do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if b.req.TopP != nil || b.req.ChatTemplateKwargs != nil {
		t.Errorf("gpt-4 picked up a preset: %+v", b.req)
	}

	rec = do(t, s, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	var models ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &models); err != nil {
		t.Fatal(err)
	}
	if len(models.Data) != 5 || models.Data[1].ID != "chatting" {
		t.Errorf("models %+v", models.Data)
	}
}

// TestPresetThinkingOffWins: the top-level reasoning_effort a client sends to
// every model does not make a non-thinking preset think, in either API's
// spelling of it.
func TestPresetThinkingOffWins(t *testing.T) {
	ps, err := LoadPresets("testdata/models.ini")
	if err != nil {
		t.Fatal(err)
	}
	b := &fakeLLM{pieces: []Delta{{Content: "hi"}}}
	s := &Server{Completion: b, Presets: ps}
	// The top-level reasoning_effort a client sends to every model does
	// not make a non-thinking preset think; each API's spelling of it.
	for _, c := range []struct {
		path string
		body map[string]any
	}{
		{"/v1/chat/completions", map[string]any{"model": "instruct", "messages": []any{user("hi")},
			"reasoning_effort": "medium"}},
		{"/v1/responses", map[string]any{"model": "chatting", "input": "hi",
			"reasoning": map[string]any{"effort": "high"}}},
	} {
		rec := do(t, s, jsonRequest("POST", c.path, c.body))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", c.path, rec.Code, rec.Body)
		}
		if th, _ := b.req.Thinking(); !th.Off {
			t.Errorf("%s: a non-thinking preset thinks: %+v", c.path, th)
		}
	}
	// A thinking preset still takes the client's effort.
	body := chatBody(user("hi"))
	body["model"] = "coding"
	body["reasoning_effort"] = "low"
	do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if th, _ := b.req.Thinking(); th.Off || th.Effort != "low" {
		t.Errorf("coding with reasoning_effort low: %+v", th)
	}
}

func TestThinkingResolution(t *testing.T) {
	kw := func(s string) map[string]json.RawMessage {
		var m map[string]json.RawMessage
		if err := json.Unmarshal([]byte(s), &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	for _, c := range []struct {
		req  CompletionRequest
		want Thinking
	}{
		{CompletionRequest{}, Thinking{}},
		{CompletionRequest{ReasoningEffort: "none"}, Thinking{Off: true}},
		{CompletionRequest{ChatTemplateKwargs: kw(`{"reasoning_effort":"none"}`)}, Thinking{Off: true}},
		{CompletionRequest{ChatTemplateKwargs: kw(`{"enable_thinking":true,"reasoning_effort":"none"}`)}, Thinking{}},
		{CompletionRequest{ChatTemplateKwargs: kw(`{"enable_thinking":false,"reasoning_effort":"low"}`)}, Thinking{Off: true}},
		{CompletionRequest{ReasoningEffort: "low", ChatTemplateKwargs: kw(`{"enable_thinking":false}`)},
			Thinking{Effort: "low"}},
		{CompletionRequest{ChatTemplateKwargs: kw(`{"preserve_thinking":false}`)}, Thinking{DropHistory: true}},
	} {
		got, err := c.req.Thinking()
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("%+v %s: got %+v, want %+v", c.req.ReasoningEffort, mustJSON(c.req.ChatTemplateKwargs), got, c.want)
		}
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
