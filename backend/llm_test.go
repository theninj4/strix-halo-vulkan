package backend

import (
	"encoding/json"
	"strings"
	"testing"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm"
)

// The parts of the language-model adapter that are not the device: the
// translation from an HTTP request into the conversation the template reads,
// the prefix comparison that decides whether a turn continues the last one,
// and the byte-to-rune buffering between the tokenizer and the stream. Each
// runs in microseconds and needs no checkpoint.

// TestCommonPrefix is what decides whether a second turn is a continuation.
func TestCommonPrefix(t *testing.T) {
	cases := []struct {
		a, b []int32
		want int
	}{
		{nil, []int32{1, 2}, 0},
		{[]int32{1, 2}, nil, 0},
		{[]int32{1, 2, 3}, []int32{1, 2, 3, 4, 5}, 3},
		{[]int32{1, 2, 3}, []int32{1, 2, 3}, 3},
		{[]int32{1, 2, 3}, []int32{1, 9, 3}, 1},
		{[]int32{1, 2, 3}, []int32{9, 2, 3}, 0},
		{[]int32{1, 2, 3, 4}, []int32{1, 2}, 2},
	}
	for _, c := range cases {
		if got := commonPrefix(c.a, c.b); got != c.want {
			t.Errorf("commonPrefix(%v, %v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// TestUTF8Stream: the model's tokens are bytes, and a multi-byte rune is
// routinely split across two of them. A delta cut there is invalid UTF-8 that
// JSON replaces with U+FFFD, so the tail is held back until it is whole.
func TestUTF8Stream(t *testing.T) {
	const want = "a café 🍵 end"
	for _, size := range []int{1, 2, 3, 5} {
		var u utf8Stream
		var got strings.Builder
		for i := 0; i < len(want); i += size {
			end := min(i+size, len(want))
			got.WriteString(u.write(want[i:end]))
		}
		if got.String() != want {
			t.Errorf("in %d-byte pieces: %q, want %q", size, got.String(), want)
		}
		if len(u.tail) != 0 {
			t.Errorf("in %d-byte pieces: %q held back at the end", size, u.tail)
		}
	}
	// Nothing partial is ever emitted: every piece is whole runes.
	var u utf8Stream
	for _, b := range []byte("🍵") {
		if out := u.write(string([]byte{b})); out != "" && out != "🍵" {
			t.Errorf("emitted %q, which is not whole runes", out)
		}
	}
}

// TestChatRequestTranslation is the one place the API's vocabulary and the
// model's meet: roles, the reasoning effort, and a tool call replayed with its
// arguments in the order they arrived.
func TestChatRequestTranslation(t *testing.T) {
	req := &api.CompletionRequest{
		ReasoningEffort: "LOW",
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolDefinition{Name: "get_weather"},
		}},
		Messages: []api.Message{
			{Role: "system", Content: api.MessageContent{{Type: "text", Text: "Be terse."}}},
			{Role: "user", Content: api.MessageContent{{Type: "text", Text: "Weather?"}}},
			{
				Role:             "assistant",
				ReasoningContent: "Call it.",
				ToolCalls: []*api.ToolCall{{
					Function: api.ToolCallReq{
						Name:      "get_weather",
						Arguments: `{"z":1,"a":"two"}`,
					},
				}},
			},
			{Role: "tool", Content: api.MessageContent{{Type: "text", Text: "18 C"}}},
		},
	}
	msgs, opt, err := chatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if opt.Effort != "low" || opt.NoThinking {
		t.Errorf("effort %q, no thinking %v", opt.Effort, opt.NoThinking)
	}
	if len(opt.Tools) != 1 {
		t.Fatalf("%d tools", len(opt.Tools))
	}
	if len(msgs) != 4 || msgs[0].Role != "system" || msgs[3].Role != "tool" {
		t.Fatalf("messages %+v", msgs)
	}
	if msgs[2].Reasoning != "Call it." {
		t.Errorf("the assistant turn's reasoning is %q", msgs[2].Reasoning)
	}
	// The order is the point: the template writes one block per argument in
	// the order the object carried them, and a map would have lost it.
	args := msgs[2].ToolCalls[0].Args
	if len(args) != 2 || args[0].Name != "z" || args[1].Name != "a" {
		t.Fatalf("arguments %+v", args)
	}
	// And the whole thing renders, which is the only test that matters for a
	// translation into a template.
	if _, err := llm.RenderChat(msgs, opt); err != nil {
		t.Errorf("rendering: %v", err)
	}
}

// TestChatRequestEffortNone: OpenAI's two spellings for "do not think" become
// the template's, which enforces it in the prompt rather than asking.
func TestChatRequestEffortNone(t *testing.T) {
	for _, effort := range []string{"none", "minimal", "None"} {
		_, opt, err := chatRequest(&api.CompletionRequest{
			ReasoningEffort: effort,
			Messages:        []api.Message{{Role: "user", Content: api.MessageContent{{Text: "hi"}}}},
		})
		if err != nil {
			t.Fatalf("%s: %v", effort, err)
		}
		if !opt.NoThinking || opt.Effort != "" {
			t.Errorf("%s: no thinking %v, effort %q", effort, opt.NoThinking, opt.Effort)
		}
	}
}

// TestChatRequestToolChoice: "none" leaves the tool block out of the prompt
// altogether, and anything that names a function is refused, because forcing
// a call is constrained decoding and there is none here.
func TestChatRequestToolChoice(t *testing.T) {
	tools := []api.Tool{{Type: "function", Function: api.ToolDefinition{Name: "f"}}}
	msgs := []api.Message{{Role: "user", Content: api.MessageContent{{Text: "hi"}}}}

	for _, choice := range []string{``, `null`, `"auto"`} {
		_, opt, err := chatRequest(&api.CompletionRequest{
			Tools: tools, Messages: msgs, ToolChoice: json.RawMessage(choice),
		})
		if err != nil {
			t.Fatalf("%s: %v", choice, err)
		}
		if len(opt.Tools) != 1 {
			t.Errorf("%s: %d tools", choice, len(opt.Tools))
		}
	}

	_, opt, err := chatRequest(&api.CompletionRequest{
		Tools: tools, Messages: msgs, ToolChoice: json.RawMessage(`"none"`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(opt.Tools) != 0 {
		t.Errorf(`tool_choice "none" left %d tools in the prompt`, len(opt.Tools))
	}

	for _, choice := range []string{`"required"`, `{"type":"function","function":{"name":"f"}}`} {
		if _, _, err := chatRequest(&api.CompletionRequest{
			Tools: tools, Messages: msgs, ToolChoice: json.RawMessage(choice),
		}); err == nil {
			t.Errorf("%s was accepted", choice)
		}
	}
}

// TestChatRequestKeepsTheClientsToolJSON: the template writes the tool object
// the client sent, so the bytes have to survive the round trip rather than
// being re-encoded from a struct whose field order is its own.
func TestChatRequestKeepsTheClientsToolJSON(t *testing.T) {
	raw := `{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}`
	var tool api.Tool
	if err := json.Unmarshal([]byte(raw), &tool); err != nil {
		t.Fatal(err)
	}
	_, opt, err := chatRequest(&api.CompletionRequest{
		Tools:    []api.Tool{tool},
		Messages: []api.Message{{Role: "user", Content: api.MessageContent{{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(opt.Tools[0].JSON); got != raw {
		t.Errorf("the tool reached the template as\n  %s\nand the client sent\n  %s", got, raw)
	}
}
