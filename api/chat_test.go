package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeLLM is the third of this package's fakes, and the reason the chat
// endpoint can be tested without 84 GB of weights and a minute of staging.
// pieces are handed to emit one at a time, which is what a real backend does
// a token at a time.
type fakeLLM struct {
	req    *CompletionRequest
	pieces []Delta
	res    *CompletionResult
	err    error
	// failAt emits this many pieces before returning err, which is how a
	// failure part way through a stream is tested.
	failAt int
}

func (f *fakeLLM) Models() []Model {
	return []Model{{ID: "qwen3.8-flash-next", Object: "model", OwnedBy: "local"}}
}

func (f *fakeLLM) Complete(_ context.Context, req *CompletionRequest,
	emit func(Delta) error,
) (*CompletionResult, error) {
	f.req = req
	for i, p := range f.pieces {
		if f.err != nil && i == f.failAt {
			return nil, f.err
		}
		if err := emit(p); err != nil {
			return nil, err
		}
	}
	if f.err != nil {
		return nil, f.err
	}
	if f.res != nil {
		return f.res, nil
	}
	return &CompletionResult{
		FinishReason: "stop",
		Usage:        Usage{PromptTokens: 7, CompletionTokens: len(f.pieces), TotalTokens: 7 + len(f.pieces)},
	}, nil
}

func chatBody(msgs ...map[string]any) map[string]any {
	return map[string]any{"model": "gpt-4", "messages": msgs}
}

func user(text string) map[string]any {
	return map[string]any{"role": "user", "content": text}
}

// frames splits an SSE body into its data payloads, which is all this
// endpoint ever sends: no event names, no ids, no retry.
func frames(t *testing.T, body string) []string {
	t.Helper()
	var out []string
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		data, ok := strings.CutPrefix(block, "data: ")
		if !ok {
			t.Fatalf("frame is not a data line: %q", block)
		}
		out = append(out, data)
	}
	return out
}

func TestChatNotLoaded(t *testing.T) {
	rec := do(t, &Server{}, jsonRequest("POST", "/v1/chat/completions", chatBody(user("hi"))))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status %d, want 501: %s", rec.Code, rec.Body)
	}
	// The message names the flag, because that is the answer to "why did
	// this 501" in almost every case.
	if !strings.Contains(rec.Body.String(), "-llm") {
		t.Errorf("the 501 does not name the flag: %s", rec.Body)
	}
}

func TestChatBuffered(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{
		{ReasoningContent: "Two and"}, {ReasoningContent: " two."},
		{Content: "It is "}, {Content: "4."},
	}}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", chatBody(user("2+2?"))))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Object != "chat.completion" || !strings.HasPrefix(got.ID, "chatcmpl-") {
		t.Errorf("object %q, id %q", got.Object, got.ID)
	}
	// The model reported is the one that ran, not the one the request asked
	// for: this server has one and the request said "gpt-4".
	if got.Model != "qwen3.8-flash-next" {
		t.Errorf("model %q", got.Model)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("%d choices", len(got.Choices))
	}
	c := got.Choices[0]
	if c.Message == nil || c.Delta != nil {
		t.Fatalf("a buffered choice carries %+v / %+v", c.Message, c.Delta)
	}
	if c.Message.Role != "assistant" || c.Message.Content != "It is 4." {
		t.Errorf("message %+v", c.Message)
	}
	if c.Message.ReasoningContent != "Two and two." {
		t.Errorf("reasoning %q", c.Message.ReasoningContent)
	}
	if c.FinishReason == nil || *c.FinishReason != "stop" {
		t.Errorf("finish_reason %v", c.FinishReason)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 11 {
		t.Errorf("usage %+v", got.Usage)
	}
}

func TestChatStreams(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{Content: "It is "}, {Content: "4."}}}
	s := &Server{Completion: b}
	body := chatBody(user("2+2?"))
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type %q", ct)
	}

	got := frames(t, rec.Body.String())
	// role, two deltas, the finish, the usage, [DONE].
	if len(got) != 6 {
		t.Fatalf("%d frames: %q", len(got), got)
	}
	if got[len(got)-1] != "[DONE]" {
		t.Errorf("the stream does not end with [DONE]: %q", got[len(got)-1])
	}

	var text strings.Builder
	for i, frame := range got[:len(got)-1] {
		var c Chunk
		if err := json.Unmarshal([]byte(frame), &c); err != nil {
			t.Fatalf("frame %d: %v: %s", i, err, frame)
		}
		if c.Object != "chat.completion.chunk" {
			t.Errorf("frame %d object %q", i, c.Object)
		}
		switch i {
		case 0:
			if c.Choices[0].Delta.Role != "assistant" {
				t.Errorf("the first frame does not carry the role: %s", frame)
			}
		case 1, 2:
			if c.Choices[0].FinishReason != nil {
				t.Errorf("frame %d finished early: %s", i, frame)
			}
			text.WriteString(c.Choices[0].Delta.Content)
		case 3:
			if c.Choices[0].FinishReason == nil || *c.Choices[0].FinishReason != "stop" {
				t.Errorf("the last choice does not finish: %s", frame)
			}
		case 4:
			// The usage frame carries no choice, which is what OpenAI's
			// include_usage does.
			if len(c.Choices) != 0 || c.Usage == nil || c.Usage.CompletionTokens != 2 {
				t.Errorf("usage frame %s", frame)
			}
		}
	}
	if text.String() != "It is 4." {
		t.Errorf("the deltas assemble to %q", text.String())
	}
}

// TestChatStreamsWithoutUsage: the usage frame is only sent when it was asked
// for, because a client that does not know the field would see a chunk with
// an empty choices array and nothing to do with it.
func TestChatStreamsWithoutUsage(t *testing.T) {
	s := &Server{Completion: &fakeLLM{pieces: []Delta{{Content: "4."}}}}
	body := chatBody(user("2+2?"))
	body["stream"] = true
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	got := frames(t, rec.Body.String())
	if len(got) != 4 { // role, one delta, the finish, [DONE]
		t.Fatalf("%d frames: %q", len(got), got)
	}
	for _, frame := range got[:3] {
		if strings.Contains(frame, `"usage"`) {
			t.Errorf("a usage field was sent unasked: %s", frame)
		}
	}
}

// TestChatStreamFailureIsAFrame: once the first byte is written the status is
// committed, so a run that fails has nowhere to put a 500 but the stream.
func TestChatStreamFailureIsAFrame(t *testing.T) {
	s := &Server{Completion: &fakeLLM{
		pieces: []Delta{{Content: "It is "}, {Content: "4."}},
		err:    context.DeadlineExceeded, failAt: 1,
	}}
	body := chatBody(user("2+2?"))
	body["stream"] = true
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	got := frames(t, rec.Body.String())
	var e errorResponse
	if err := json.Unmarshal([]byte(got[len(got)-1]), &e); err != nil {
		t.Fatalf("the last frame is not an error envelope: %q", got[len(got)-1])
	}
	if e.Error.Type != "server_error" {
		t.Errorf("error frame %+v", e.Error)
	}
}

func TestChatToolCalls(t *testing.T) {
	s := &Server{Completion: &fakeLLM{
		pieces: []Delta{{Content: "Looking it up."}},
		res: &CompletionResult{
			FinishReason: "tool_calls",
			ToolCalls: []*ToolCall{{
				ID: "call_1", Type: "function",
				Function: ToolCallReq{Name: "get_weather", Arguments: `{"city":"Wimbledon"}`},
			}},
		},
	}}
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", chatBody(user("weather?"))))
	var got CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	calls := got.Choices[0].Message.ToolCalls
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("tool calls %+v", calls)
	}
	if *got.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason %q", *got.Choices[0].FinishReason)
	}
}

// TestChatRequestReachesTheBackend covers the one decode this endpoint cannot
// get wrong quietly: `temperature: 0` is greedy and not "unset", and a plain
// float field cannot tell the two apart.
func TestChatRequestReachesTheBackend(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{Content: "4."}}}
	s := &Server{Completion: b}
	body := chatBody(
		map[string]any{"role": "system", "content": "Be terse."},
		user("2+2?"),
	)
	body["temperature"] = 0
	body["max_completion_tokens"] = 32
	body["reasoning_effort"] = "low"
	do(t, s, jsonRequest("POST", "/v1/chat/completions", body))
	if b.req == nil {
		t.Fatal("the backend was not called")
	}
	if b.req.Temperature == nil || *b.req.Temperature != 0 {
		t.Errorf("temperature %v, want a set zero", b.req.Temperature)
	}
	if b.req.TopP != nil {
		t.Errorf("top_p %v, want unset", b.req.TopP)
	}
	if b.req.Budget() != 32 {
		t.Errorf("budget %d", b.req.Budget())
	}
	if b.req.ReasoningEffort != "low" {
		t.Errorf("reasoning_effort %q", b.req.ReasoningEffort)
	}
	if len(b.req.Messages) != 2 || b.req.Messages[0].Content.Text() != "Be terse." {
		t.Errorf("messages %+v", b.req.Messages)
	}
}

// TestChatRefusals is the endpoint's own list: a request it will not serve is
// answered, not approximated.
func TestChatRefusals(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
		want string
	}{
		{"no messages", chatBody(), "messages"},
		{"n above one", withField(chatBody(user("hi")), "n", 2), "one completion"},
		{"too many stop sequences", withField(chatBody(user("hi")), "stop",
			[]string{"a", "b", "c", "d", "e"}), "at most 4"},
		{"an empty stop sequence", withField(chatBody(user("hi")), "stop",
			[]string{""}), "empty"},
		{"min_p above one", withField(chatBody(user("hi")), "min_p", 1.5), "min_p"},
		{"repeat_penalty of zero", withField(chatBody(user("hi")), "repeat_penalty", 0), "repeat_penalty"},
		{"an unknown template kwarg", withField(chatBody(user("hi")), "chat_template_kwargs",
			map[string]any{"add_vision_id": true}), "add_vision_id"},
		{"json mode", withField(chatBody(user("hi")), "response_format",
			map[string]any{"type": "json_object"}), "response_format"},
		{"an image block", chatBody(map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "what is this?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:,"}},
		}}), "vision"},
	}
	for _, c := range cases {
		s := &Server{Completion: &fakeLLM{pieces: []Delta{{Content: "4."}}}}
		rec := do(t, s, jsonRequest("POST", "/v1/chat/completions", c.body))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400: %s", c.name, rec.Code, rec.Body)
			continue
		}
		if !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%s: the 400 does not mention %q: %s", c.name, c.want, rec.Body)
		}
	}
}

func withField(body map[string]any, key string, v any) map[string]any {
	body[key] = v
	return body
}

// TestChatStopAsAString covers OpenAI's other spelling of `stop`: a bare
// string rather than an array. Both have to decode to the same thing, or a
// client using the shorthand would get no stop sequence and no error either.
func TestChatStopAsAString(t *testing.T) {
	for _, c := range []struct {
		name string
		stop any
		want []string
	}{
		{"a bare string", "\n\n", []string{"\n\n"}},
		{"an array", []string{"\n\n", "END"}, []string{"\n\n", "END"}},
		{"null", nil, nil},
	} {
		b := &fakeLLM{pieces: []Delta{{Content: "4."}}}
		s := &Server{Completion: b}
		rec := do(t, s, jsonRequest("POST", "/v1/chat/completions",
			withField(chatBody(user("hi")), "stop", c.stop)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status %d: %s", c.name, rec.Code, rec.Body)
			continue
		}
		if len(b.req.Stop) != len(c.want) {
			t.Errorf("%s: stop %q, want %q", c.name, b.req.Stop, c.want)
			continue
		}
		for i := range c.want {
			if b.req.Stop[i] != c.want[i] {
				t.Errorf("%s: stop %q, want %q", c.name, b.req.Stop, c.want)
			}
		}
	}
}

// TestChatStopSequenceFinishesAsStop: OpenAI folds a stop sequence into
// "stop", so a client cannot tell it from an end-of-generation token -- which
// is OpenAI's own behaviour and not an omission here. Anthropic's envelope
// reports which, and TestMessagesStopSequence checks that.
func TestChatStopSequenceFinishesAsStop(t *testing.T) {
	s := &Server{Completion: &fakeLLM{
		pieces: []Delta{{Content: "one line"}},
		res: &CompletionResult{
			FinishReason: "stop", StopSequence: "\n\n",
			Usage: Usage{PromptTokens: 7, CompletionTokens: 1, TotalTokens: 8},
		},
	}}
	rec := do(t, s, jsonRequest("POST", "/v1/chat/completions",
		withField(chatBody(user("hi")), "stop", []string{"\n\n"})))
	var got CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if *got.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason %q", *got.Choices[0].FinishReason)
	}
	if got.Choices[0].Message.Content != "one line" {
		t.Errorf("content %q -- the stop sequence should not be in it", got.Choices[0].Message.Content)
	}
}

// TestChatModelsListsTheLanguageModel: a client discovers what this process
// loaded from GET /v1/models, and the language model has to be in it.
func TestChatModelsListsTheLanguageModel(t *testing.T) {
	s := &Server{Completion: &fakeLLM{}, Speech: &fakeSpeech{}}
	rec := do(t, s, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	var got ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 2 || got.Data[0].ID != "qwen3.8-flash-next" {
		t.Fatalf("models %+v", got.Data)
	}
	// The voices belong to the speech model and not to this one.
	if len(got.Data[0].Voices) != 0 {
		t.Errorf("the language model has voices: %v", got.Data[0].Voices)
	}
}
