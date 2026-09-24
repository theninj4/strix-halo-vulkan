package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// The other two envelopes. They are the same generation as
// /v1/chat/completions, so what is tested here is the translation in both
// directions and nothing about the model.

// events splits a named SSE body into (name, payload) pairs. Both of these
// endpoints name every frame, unlike the chat one.
func events(t *testing.T, body string) [][2]string {
	t.Helper()
	var out [][2]string
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		name, rest, ok := strings.Cut(block, "\n")
		if !ok {
			t.Fatalf("frame has no data line: %q", block)
		}
		name = strings.TrimPrefix(name, "event: ")
		data := strings.TrimPrefix(rest, "data: ")
		out = append(out, [2]string{name, data})
	}
	return out
}

func names(evs [][2]string) []string {
	out := make([]string, len(evs))
	for i, e := range evs {
		out[i] = e[0]
	}
	return out
}

// --- /v1/responses ---

func TestResponsesBuffered(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{
		{ReasoningContent: "Two and two."}, {Content: "It is "}, {Content: "4."},
	}}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/responses", map[string]any{
		"model": "gpt-5", "input": "2+2?", "instructions": "Be terse.",
		"max_output_tokens": 32, "reasoning": map[string]any{"effort": "low"},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Object != "response" || got.Status != "completed" || !strings.HasPrefix(got.ID, "resp_") {
		t.Errorf("%+v", got)
	}
	if len(got.Output) != 2 {
		t.Fatalf("%d output items: %+v", len(got.Output), got.Output)
	}
	// The reasoning is an item of its own, before the message it led to.
	if got.Output[0].Type != "reasoning" || got.Output[0].Summary[0].Text != "Two and two." {
		t.Errorf("reasoning item %+v", got.Output[0])
	}
	msg := got.Output[1]
	if msg.Type != "message" || msg.Role != "assistant" || msg.Content[0].Text != "It is 4." {
		t.Errorf("message item %+v", msg)
	}
	if got.Usage == nil || got.Usage.OutputTokens != 3 {
		t.Errorf("usage %+v", got.Usage)
	}
	// The instructions became the leading system turn, and the effort and
	// the budget crossed over under their other names.
	if len(b.req.Messages) != 2 || b.req.Messages[0].Role != "system" {
		t.Fatalf("messages %+v", b.req.Messages)
	}
	if b.req.Messages[0].Content.Text() != "Be terse." || b.req.Messages[1].Content.Text() != "2+2?" {
		t.Errorf("messages %+v", b.req.Messages)
	}
	if b.req.Budget() != 32 || b.req.ReasoningEffort != "low" {
		t.Errorf("budget %d, effort %q", b.req.Budget(), b.req.ReasoningEffort)
	}
}

// TestResponsesInputItems is the shape an agent loop replays: a question, the
// call the model made, what the tool returned, and the answer.
func TestResponsesInputItems(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{Content: "18 degrees."}}}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/responses", map[string]any{
		"input": []any{
			map[string]any{"role": "user", "content": "Weather?"},
			map[string]any{"type": "reasoning", "summary": []any{
				map[string]any{"type": "summary_text", "text": "Call the tool."},
			}},
			map[string]any{
				"type": "function_call", "call_id": "call_1",
				"name": "get_weather", "arguments": `{"city":"Wimbledon"}`,
			},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "18 °C"},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "And in words?"},
			}},
		},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	got := b.req.Messages
	if len(got) != 4 {
		t.Fatalf("%d messages: %+v", len(got), got)
	}
	if got[1].Role != "assistant" || got[1].ReasoningContent != "Call the tool." {
		t.Errorf("the call's message %+v", got[1])
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool calls %+v", got[1].ToolCalls)
	}
	if got[2].Role != "tool" || got[2].Content.Text() != "18 °C" {
		t.Errorf("the tool result %+v", got[2])
	}
	if got[3].Content.Text() != "And in words?" {
		t.Errorf("the last turn %+v", got[3])
	}
}

func TestResponsesStreams(t *testing.T) {
	s := &Server{Completion: &fakeLLM{pieces: []Delta{
		{ReasoningContent: "Two and two."}, {Content: "It is "}, {Content: "4."},
	}}}
	rec := do(t, s, jsonRequest("POST", "/v1/responses", map[string]any{
		"input": "2+2?", "stream": true,
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	evs := events(t, rec.Body.String())
	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
		"response.output_item.done",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done", "response.output_item.done",
		"response.completed",
	}
	if got := names(evs); len(got) != len(want) {
		t.Fatalf("%d events, want %d:\n  %v", len(got), len(want), got)
	} else {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("event %d is %q, want %q\n  %v", i, got[i], want[i], got)
			}
		}
	}
	// The sequence numbers are the client's way of spotting a gap, so they
	// have to be consecutive from zero.
	var text strings.Builder
	for i, e := range evs {
		var ev responsesEvent
		if err := json.Unmarshal([]byte(e[1]), &ev); err != nil {
			t.Fatalf("event %d: %v", i, err)
		}
		if ev.SequenceNumber != i {
			t.Errorf("event %d has sequence_number %d", i, ev.SequenceNumber)
		}
		if ev.Type != e[0] {
			t.Errorf("event %d is named %q and typed %q", i, e[0], ev.Type)
		}
		if e[0] == "response.output_text.delta" {
			text.WriteString(ev.Delta)
		}
		if e[0] == "response.completed" {
			if ev.Response == nil || ev.Response.Status != "completed" || len(ev.Response.Output) != 2 {
				t.Errorf("the final response is %+v", ev.Response)
			}
		}
	}
	if text.String() != "It is 4." {
		t.Errorf("the deltas assemble to %q", text.String())
	}
}

func TestResponsesRefusals(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
		want string
	}{
		{"no input", map[string]any{"model": "x"}, "input is empty"},
		{"an image by file id", map[string]any{"input": []any{map[string]any{
			"role": "user", "content": []any{map[string]any{"type": "input_image", "file_id": "file-1"}},
		}}}, "keeps no files"},
		{"a file part", map[string]any{"input": []any{map[string]any{
			"role": "user", "content": []any{map[string]any{"type": "input_file", "file_id": "file-1"}},
		}}}, "input_file"},
		{"an unknown item", map[string]any{"input": []any{
			map[string]any{"type": "web_search_call"},
		}}, "web_search_call"},
		{"a json schema", map[string]any{
			"input": "hi",
			"text":  map[string]any{"format": map[string]any{"type": "json_schema"}},
		}, "constrained decoding"},
	}
	for _, c := range cases {
		s := &Server{Completion: &fakeLLM{pieces: []Delta{{Content: "4."}}}}
		rec := do(t, s, jsonRequest("POST", "/v1/responses", c.body))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400: %s", c.name, rec.Code, rec.Body)
			continue
		}
		if !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%s: the 400 does not mention %q: %s", c.name, c.want, rec.Body)
		}
	}
}

// --- /v1/messages ---

func TestMessagesBuffered(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{ReasoningContent: "Two and two."}, {Content: "It is 4."}}}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/messages", map[string]any{
		"model": "claude-3", "max_tokens": 64, "system": "Be terse.",
		"messages": []any{map[string]any{"role": "user", "content": "2+2?"}},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Type != "message" || got.Role != "assistant" || !strings.HasPrefix(got.ID, "msg_") {
		t.Errorf("%+v", got)
	}
	if len(got.Content) != 2 {
		t.Fatalf("%d blocks: %+v", len(got.Content), got.Content)
	}
	if got.Content[0].Type != "thinking" || got.Content[0].Thinking != "Two and two." {
		t.Errorf("thinking block %+v", got.Content[0])
	}
	if got.Content[1].Type != "text" || got.Content[1].Text != "It is 4." {
		t.Errorf("text block %+v", got.Content[1])
	}
	if got.StopReason == nil || *got.StopReason != "end_turn" {
		t.Errorf("stop_reason %v", got.StopReason)
	}
	if got.Usage.InputTokens != 7 || got.Usage.OutputTokens != 2 {
		t.Errorf("usage %+v", got.Usage)
	}
	if b.req.Budget() != 64 || b.req.Messages[0].Role != "system" {
		t.Errorf("request %+v", b.req.Messages)
	}
}

// TestMessagesBlocks is Anthropic's own round trip: an assistant turn with a
// tool_use block, and the user turn carrying its result, which becomes a tool
// message and not a user one.
func TestMessagesBlocks(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{Content: "18 degrees."}}}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/messages", map[string]any{
		"max_tokens": 64,
		"messages": []any{
			map[string]any{"role": "user", "content": "Weather?"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "thinking", "thinking": "Call the tool."},
				map[string]any{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
					"input": map[string]any{"city": "Wimbledon"}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18 °C"},
				map[string]any{"type": "text", "text": "And in words?"},
			}},
		},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	got := b.req.Messages
	if len(got) != 4 {
		t.Fatalf("%d messages: %+v", len(got), got)
	}
	if got[1].Role != "assistant" || got[1].ReasoningContent != "Call the tool." {
		t.Errorf("the assistant turn %+v", got[1])
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Function.Arguments != `{"city":"Wimbledon"}` {
		t.Errorf("tool calls %+v", got[1].ToolCalls[0])
	}
	// The tool result is its own message and comes before the user's text.
	if got[2].Role != "tool" || got[2].Content.Text() != "18 °C" {
		t.Errorf("the tool result %+v", got[2])
	}
	if got[3].Role != "user" || got[3].Content.Text() != "And in words?" {
		t.Errorf("the last turn %+v", got[3])
	}
}

func TestMessagesStreams(t *testing.T) {
	s := &Server{Completion: &fakeLLM{
		pieces: []Delta{{ReasoningContent: "Two and two."}, {Content: "It is "}, {Content: "4."}},
		res: &CompletionResult{
			FinishReason: "tool_calls",
			ToolCalls: []*ToolCall{{ID: "call_1", Type: "function", Function: ToolCallReq{
				Name: "get_weather", Arguments: `{"city":"Wimbledon"}`,
			}}},
			Usage: Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
		},
	}}
	rec := do(t, s, jsonRequest("POST", "/v1/messages", map[string]any{
		"max_tokens": 64, "stream": true,
		"messages": []any{map[string]any{"role": "user", "content": "2+2?"}},
	}))
	evs := events(t, rec.Body.String())
	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_stop", // thinking
		"content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // text
		"content_block_start", "content_block_delta", "content_block_stop", // tool_use
		"message_delta", "message_stop",
	}
	got := names(evs)
	if len(got) != len(want) {
		t.Fatalf("%d events, want %d:\n  %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event %d is %q, want %q\n  %v", i, got[i], want[i], got)
		}
	}
	// The block indices go up by one and never repeat across kinds.
	wantIndex := []int{0, 0, 0, 1, 1, 1, 1, 2, 2, 2}
	for i, e := range evs[1 : len(evs)-2] {
		var ev messagesEvent
		if err := json.Unmarshal([]byte(e[1]), &ev); err != nil {
			t.Fatal(err)
		}
		if ev.Index != wantIndex[i] {
			t.Errorf("event %d (%s) has index %d, want %d", i+1, e[0], ev.Index, wantIndex[i])
		}
	}
	var final messagesEvent
	if err := json.Unmarshal([]byte(evs[len(evs)-2][1]), &final); err != nil {
		t.Fatal(err)
	}
	if final.Delta == nil || final.Delta.StopReason == nil || *final.Delta.StopReason != "tool_use" {
		t.Errorf("message_delta %+v", final.Delta)
	}
	if final.Usage == nil || final.Usage.OutputTokens != 3 {
		t.Errorf("usage %+v", final.Usage)
	}
}

func TestMessagesRefusals(t *testing.T) {
	msg := []any{map[string]any{"role": "user", "content": "hi"}}
	cases := []struct {
		name string
		body map[string]any
		want string
	}{
		{"no max_tokens", map[string]any{"messages": msg}, "max_tokens"},
		{"a thinking budget", map[string]any{
			"messages": msg, "max_tokens": 16,
			"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024},
		}, "budget_tokens"},
		{"an image with no source type", map[string]any{"max_tokens": 16, "messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "image", "source": map[string]any{"data": ""}},
			}},
		}}, "base64 and url"},
		{"a document block", map[string]any{"max_tokens": 16, "messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "document", "source": map[string]any{"type": "base64"}},
			}},
		}}, "document"},
	}
	for _, c := range cases {
		s := &Server{Completion: &fakeLLM{pieces: []Delta{{Content: "4."}}}}
		rec := do(t, s, jsonRequest("POST", "/v1/messages", c.body))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400: %s", c.name, rec.Code, rec.Body)
			continue
		}
		if !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%s: the 400 does not mention %q: %s", c.name, c.want, rec.Body)
		}
	}
}

// TestMessagesStopSequence: Anthropic's envelope is the one that says *which*
// sequence stopped the generation, so the backend carries it separately from
// the finish reason the other two fold it into.
func TestMessagesStopSequence(t *testing.T) {
	b := &fakeLLM{
		pieces: []Delta{{Content: "one line"}},
		res: &CompletionResult{
			FinishReason: "stop", StopSequence: "\n\n",
			Usage: Usage{PromptTokens: 7, CompletionTokens: 1, TotalTokens: 8},
		},
	}
	s := &Server{Completion: b}
	rec := do(t, s, jsonRequest("POST", "/v1/messages", map[string]any{
		"max_tokens": 16, "stop_sequences": []string{"\n\n"},
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	// The request's sequences reached the backend under the other name.
	if len(b.req.Stop) != 1 || b.req.Stop[0] != "\n\n" {
		t.Errorf("stop %q", b.req.Stop)
	}
	var got MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.StopReason == nil || *got.StopReason != "stop_sequence" {
		t.Errorf("stop_reason %v", got.StopReason)
	}
	if got.StopSequence == nil || *got.StopSequence != "\n\n" {
		t.Errorf("stop_sequence %v", got.StopSequence)
	}
}

// TestEnvelopesNeedTheModel: all three answer 501 naming the flag when the
// process was started without it, rather than 404.
func TestEnvelopesNeedTheModel(t *testing.T) {
	for _, path := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages"} {
		rec := do(t, &Server{}, jsonRequest("POST", path, map[string]any{}))
		if rec.Code != http.StatusNotImplemented {
			t.Errorf("%s: status %d, want 501", path, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), "-llm") {
			t.Errorf("%s: the 501 does not name the flag: %s", path, rec.Body)
		}
	}
}

// TestMessagesThinkingDisabled: Anthropic's switch reaches the backend as the
// effort the template understands.
func TestMessagesThinkingDisabled(t *testing.T) {
	b := &fakeLLM{pieces: []Delta{{Content: "4."}}}
	s := &Server{Completion: b}
	do(t, s, jsonRequest("POST", "/v1/messages", map[string]any{
		"max_tokens": 16, "thinking": map[string]any{"type": "disabled"},
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}))
	if b.req.ReasoningEffort != "none" {
		t.Errorf("reasoning_effort %q", b.req.ReasoningEffort)
	}
}

// TestImagesReachTheBackend: each door hands an image to the backend as the
// one form it reads, an image_url block holding a data: URL, in its place
// between the texts (LLM-VISION.md V8).
func TestImagesReachTheBackend(t *testing.T) {
	const png = "data:image/png;base64,iVBORw0KGgo="
	cases := []struct {
		name, path string
		body       map[string]any
	}{
		{"chat", "/v1/chat/completions", chatBody(map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "what is "},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": png}},
			map[string]any{"type": "text", "text": " this?"},
		}})},
		{"responses", "/v1/responses", map[string]any{"input": []any{map[string]any{
			"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "what is "},
				map[string]any{"type": "input_image", "image_url": png},
				map[string]any{"type": "input_text", "text": " this?"},
			}}}}},
		{"messages", "/v1/messages", map[string]any{"max_tokens": 16, "messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "what is "},
				map[string]any{"type": "image", "source": map[string]any{
					"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
				map[string]any{"type": "text", "text": " this?"},
			}}}}},
	}
	for _, c := range cases {
		b := &fakeLLM{pieces: []Delta{{Content: "a barn"}}}
		rec := do(t, &Server{Completion: b}, jsonRequest("POST", c.path, c.body))
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status %d: %s", c.name, rec.Code, rec.Body)
			continue
		}
		msgs := b.req.Messages
		got := msgs[len(msgs)-1].Content
		if len(got) != 3 || got[0].Text != "what is " || got[1].Type != "image_url" ||
			got[1].ImageURL == nil || got[1].ImageURL.URL != png || got[2].Text != " this?" {
			t.Errorf("%s: the backend saw %+v", c.name, got)
		}
	}
}
