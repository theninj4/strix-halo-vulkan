package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The oracle is Jinja rendering the checkpoint's own template, case for case:
//
//	go run ./cmd/llm -model …-00001-of-00004.gguf -chat-template \
//	    > reference/out/chat/template.jinja
//	.venv/bin/python reference/dump_chat_template.py
const chatRefDir = "../reference/out/chat"

// The reference's shape mirrors llm's types rather than being them, so that
// the exported API is not bent into a test fixture's JSON.
type chatRef struct {
	Cases []struct {
		Name     string `json:"name"`
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
			ToolCalls []struct {
				Name string `json:"name"`
				Args []struct {
					Name  string          `json:"name"`
					Value json.RawMessage `json:"value"`
				} `json:"args"`
			} `json:"tool_calls"`
		} `json:"messages"`
		Tools []json.RawMessage `json:"tools"`
		Opts  struct {
			Effort             string `json:"effort"`
			NoThinking         bool   `json:"no_thinking"`
			DropThinking       bool   `json:"drop_thinking"`
			NoGenerationPrompt bool   `json:"no_generation_prompt"`
		} `json:"opts"`
		Rendered string `json:"rendered"`
	} `json:"cases"`
}

func loadChatRef(t *testing.T) *chatRef {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(chatRefDir, "cases.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_chat_template.py", chatRefDir, err)
	}
	var r chatRef
	if err := json.Unmarshal(buf, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// TestRenderChat is the stage's gate: the transcription in chat.go against
// Jinja's own rendering of the template it was transcribed from, character
// for character. A prompt one space or one newline off is a prompt the model
// was never trained on, and nothing downstream would report it.
func TestRenderChat(t *testing.T) {
	ref := loadChatRef(t)
	if len(ref.Cases) == 0 {
		t.Fatal("the reference dump has no cases")
	}
	for _, c := range ref.Cases {
		msgs := make([]ChatMessage, 0, len(c.Messages))
		for _, m := range c.Messages {
			msg := ChatMessage{Role: m.Role, Content: m.Content, Reasoning: m.Reasoning}
			for _, call := range m.ToolCalls {
				out := ChatToolCall{Name: call.Name}
				for _, a := range call.Args {
					out.Args = append(out.Args, ChatArg{Name: a.Name, Value: a.Value})
				}
				msg.ToolCalls = append(msg.ToolCalls, out)
			}
			msgs = append(msgs, msg)
		}
		opt := ChatOpts{
			Effort:             c.Opts.Effort,
			NoThinking:         c.Opts.NoThinking,
			DropThinking:       c.Opts.DropThinking,
			NoGenerationPrompt: c.Opts.NoGenerationPrompt,
		}
		for _, raw := range c.Tools {
			opt.Tools = append(opt.Tools, ChatTool{JSON: raw})
		}
		got, err := RenderChat(msgs, opt)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if got != c.Rendered {
			t.Errorf("%s: %s", c.Name, firstDiff(got, c.Rendered))
			continue
		}
		t.Logf("%-24s %6d chars", c.Name, len(got))
	}
}

// firstDiff reports where two renderings part company, with the line they
// part on, because a 2500-character prompt diffed whole is unreadable.
func firstDiff(got, want string) string {
	n := min(len(got), len(want))
	i := 0
	for ; i < n && got[i] == want[i]; i++ {
	}
	if i == n && len(got) == len(want) {
		return "identical"
	}
	window := func(s string) string {
		return fmt.Sprintf("%q", s[max(i-40, 0):min(i+40, len(s))])
	}
	return fmt.Sprintf("byte %d (line %d) of %d, want %d\n  got  ...%s...\n  want ...%s...",
		i, 1+strings.Count(got[:i], "\n"), len(got), len(want), window(got), window(want))
}

// TestRenderChatRefuses covers the template's own raise_exception calls: the
// cases where it stops rather than render something the model has not seen.
func TestRenderChatRefuses(t *testing.T) {
	cases := []struct {
		name string
		msgs []ChatMessage
		opt  ChatOpts
	}{
		{"no messages", nil, ChatOpts{}},
		{"system in the middle", []ChatMessage{
			{Role: "user", Content: "hi"}, {Role: "system", Content: "be terse"},
		}, ChatOpts{}},
		{"unknown role", []ChatMessage{{Role: "narrator", Content: "hi"}}, ChatOpts{}},
		{"unknown effort", []ChatMessage{{Role: "user", Content: "hi"}}, ChatOpts{Effort: "ultra"}},
		{"nameless tool call", []ChatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", ToolCalls: []ChatToolCall{{}}},
		}, ChatOpts{}},
	}
	for _, c := range cases {
		if _, err := RenderChat(c.msgs, c.opt); err == nil {
			t.Errorf("%s: rendered without an error", c.name)
		}
	}
}

// TestParseToolArguments checks the one property a map would lose: OpenAI
// sends arguments as a JSON string, the template writes one block per
// argument in the object's order, and replaying a call has to reproduce it.
func TestParseToolArguments(t *testing.T) {
	args, err := ParseToolArguments(`{"z": 1, "a": "two", "m": {"k": [1,2]}}`)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"z", "a", "m"}
	if len(args) != len(want) {
		t.Fatalf("%d arguments, want %d", len(args), len(want))
	}
	for i, w := range want {
		if args[i].Name != w {
			t.Errorf("argument %d is %q, want %q", i, args[i].Name, w)
		}
	}
	if got := string(args[2].Value); got != `{"k": [1,2]}` {
		t.Errorf("nested value %q", got)
	}
	if _, err := ParseToolArguments(`[1,2]`); err == nil {
		t.Error("an array was accepted as an argument object")
	}
	if args, err := ParseToolArguments("  "); err != nil || args != nil {
		t.Errorf("empty arguments: %v, %v", args, err)
	}
}

// TestPyJSON is the detail that decides whether the tool block is the model's
// own: Python's json.dumps writes a space after every separator and escapes
// neither `<` nor `&`, where Go's encoder does the opposite of both.
func TestPyJSON(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{"a":1,"b":[1,2],"c":{"d":true}}`, `{"a": 1, "b": [1, 2], "c": {"d": true}}`},
		{`{"a":"<b> & </b>"}`, `{"a": "<b> & </b>"}`},
		{`{"a":"café °C"}`, `{"a": "café °C"}`},
		{`{"a":"tab\there\nand"}`, `{"a": "tab\there\nand"}`},
		{`{"a":1.50,"b":1e3,"c":null}`, `{"a": 1.50, "b": 1e3, "c": null}`},
		{`[]`, `[]`},
		{`{}`, `{}`},
	}
	for _, c := range cases {
		got, err := pyJSON(json.RawMessage(c.in))
		if err != nil {
			t.Errorf("%s: %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("%s\n  got  %s\n  want %s", c.in, got, c.want)
		}
	}
}
