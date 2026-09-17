package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// feed runs a generation through the decoder one piece at a time, which is
// the only way it is ever used: the pieces are tokens, and every marker in
// the format can straddle two of them.
func feed(d *ChatDecoder, pieces ...string) (reasoning, content string, calls []ChatCall, err error) {
	var r, c strings.Builder
	for _, p := range pieces {
		rr, cc := d.Write(p)
		r.WriteString(rr)
		c.WriteString(cc)
	}
	rr, cc, calls, err := d.Close()
	r.WriteString(rr)
	c.WriteString(cc)
	return r.String(), c.String(), calls, err
}

// TestChatDecoderSplitsThinking is the endpoint's `reasoning_content`: the
// generation prompt leaves `<think>` open, so everything up to `</think>` is
// the model's working and everything after it is the answer.
func TestChatDecoderSplitsThinking(t *testing.T) {
	cases := []struct {
		name     string
		thinking bool
		pieces   []string
		wantR    string
		wantC    string
	}{
		{"whole", true, []string{"Two and two.</think>\n\nIt is 4."}, "Two and two.", "It is 4."},
		{
			"marker split across tokens", true,
			[]string{"Two and", " two.<", "/th", "ink>", "\n\n", "It is ", "4."},
			"Two and two.", "It is 4.",
		},
		{"no close", true, []string{"Still thinking"}, "Still thinking", ""},
		{"not thinking", false, []string{"It is 4."}, "", "It is 4."},
		{"empty reasoning", true, []string{"</think>\n\nIt is 4."}, "", "It is 4."},
		{
			"a less-than that is not a marker", true,
			[]string{"if a <", "b then", "</think>", "\n\n", "1 <", " 2"},
			"if a <b then", "1 < 2",
		},
		{"trailing whitespace is the answer's", true, []string{"x</think>\n\nline\n\n"}, "x", "line\n\n"},
	}
	for _, c := range cases {
		d := NewChatDecoder(nil, c.thinking)
		r, got, calls, err := feed(d, c.pieces...)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if len(calls) != 0 {
			t.Errorf("%s: %d tool calls", c.name, len(calls))
		}
		if r != c.wantR {
			t.Errorf("%s: reasoning %q, want %q", c.name, r, c.wantR)
		}
		if got != c.wantC {
			t.Errorf("%s: content %q, want %q", c.name, got, c.wantC)
		}
	}
}

// TestChatDecoderHoldsBackAPartialMarker is the property that makes streaming
// safe: a token ending in `<` might be the beginning of a tool call, and a
// client that was already sent it would have to be told to take it back.
func TestChatDecoderHoldsBackAPartialMarker(t *testing.T) {
	d := NewChatDecoder(nil, false)
	if _, got := d.Write("done. <tool_c"); got != "done. " {
		t.Fatalf("emitted %q; the partial marker should have been held back", got)
	}
	if _, got := d.Write("all>\n<function=f>\n</function>\n</tool_call>"); got != "" {
		t.Fatalf("emitted %q after the marker completed", got)
	}
	_, content, calls, err := d.Close()
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Errorf("content %q after the call", content)
	}
	if len(calls) != 1 || calls[0].Name != "f" {
		t.Fatalf("calls %+v", calls)
	}
}

// TestChatDecoderToolCalls checks the inverse of the template's own rule: a
// string argument was written raw and everything else was written as JSON, so
// rebuilding `arguments` needs the schema and not a guess.
func TestChatDecoderToolCalls(t *testing.T) {
	tools := []ChatTool{{JSON: json.RawMessage(`{
		"type": "function",
		"function": {
			"name": "get_weather",
			"parameters": {"type": "object", "properties": {
				"city":  {"type": "string"},
				"days":  {"type": "integer"},
				"exact": {"type": "boolean"},
				"where": {"type": "object"},
				"note":  {"type": ["string", "null"]}
			}}
		}
	}`)}}
	gen := "I will look it up." +
		"\n\n<tool_call>\n<function=get_weather>\n" +
		"<parameter=city>\nWimbledon\n</parameter>\n" +
		"<parameter=days>\n2\n</parameter>\n" +
		"<parameter=exact>\ntrue\n</parameter>\n" +
		"<parameter=where>\n{\"lat\": 51.4, \"lon\": -0.2}\n</parameter>\n" +
		"<parameter=note>\n123\n</parameter>\n" +
		"</function>\n</tool_call>"

	d := NewChatDecoder(tools, false)
	_, content, calls, err := feed(d, gen)
	if err != nil {
		t.Fatal(err)
	}
	if content != "I will look it up." {
		t.Errorf("content %q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("%d calls, want 1", len(calls))
	}
	// `days` is an integer so `2` is the number; `note` is a string so the
	// same digits are the text "123", which is the whole point of reading
	// the schema.
	want := `{"city":"Wimbledon","days":2,"exact":true,"where":{"lat": 51.4, "lon": -0.2},"note":"123"}`
	if calls[0].Name != "get_weather" || calls[0].Arguments != want {
		t.Errorf("call %+v\n  want arguments %s", calls[0], want)
	}
	if !json.Valid([]byte(calls[0].Arguments)) {
		t.Errorf("arguments are not JSON: %s", calls[0].Arguments)
	}
}

// TestChatDecoderTwoCalls covers a multi-line string argument -- the format
// has no quoting, so a value ends at its closing tag and nowhere else -- and
// a second call in the same generation.
func TestChatDecoderTwoCalls(t *testing.T) {
	tools := []ChatTool{{JSON: json.RawMessage(
		`{"name": "search", "parameters": {"properties": {"query": {"type": "string"}}}}`)}}
	gen := "<tool_call>\n<function=search>\n<parameter=query>\nline one\nline two\n</parameter>\n" +
		"</function>\n</tool_call>\n" +
		"<tool_call>\n<function=search>\n<parameter=query>\nagain\n</parameter>\n" +
		"</function>\n</tool_call>"
	_, _, calls, err := feed(NewChatDecoder(tools, false), gen)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("%d calls, want 2: %+v", len(calls), calls)
	}
	if got := calls[0].Arguments; got != `{"query":"line one\nline two"}` {
		t.Errorf("first call arguments %s", got)
	}
	if got := calls[1].Arguments; got != `{"query":"again"}` {
		t.Errorf("second call arguments %s", got)
	}
}

// TestChatDecoderDropsATruncatedCall: a generation that hit the token limit
// mid-call has no complete call in it, and half an argument reported as a
// whole one is worse than none.
func TestChatDecoderDropsATruncatedCall(t *testing.T) {
	gen := "<tool_call>\n<function=search>\n<parameter=query>\nline on"
	_, content, calls, err := feed(NewChatDecoder(nil, false), gen)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 0 {
		t.Errorf("%d calls from a truncated block: %+v", len(calls), calls)
	}
	if content != "" {
		t.Errorf("content %q", content)
	}
}

// TestChatDecoderUnknownTool: a call to a function the request never declared
// has no schema, so a value that parses as JSON is taken at face value and
// anything else is a string.
func TestChatDecoderUnknownTool(t *testing.T) {
	gen := "<tool_call>\n<function=mystery>\n<parameter=a>\n7\n</parameter>\n" +
		"<parameter=b>\nseven\n</parameter>\n</function>\n</tool_call>"
	_, _, calls, err := feed(NewChatDecoder(nil, false), gen)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Arguments != `{"a":7,"b":"seven"}` {
		t.Fatalf("calls %+v", calls)
	}
}

// TestChatDecoderStopSequences: a stop sequence is another marker, so it gets
// the same holdback -- its first character must not reach the client and then
// need taking back.
func TestChatDecoderStopSequences(t *testing.T) {
	cases := []struct {
		name   string
		stop   []string
		pieces []string
		want   string
		wantAt string
	}{
		{
			"whole", []string{"END"},
			[]string{"the answer ENDand more"}, "the answer ", "END",
		},
		{
			"split across tokens", []string{"END"},
			[]string{"the ans", "wer E", "N", "D more"}, "the answer ", "END",
		},
		{
			"a blank line", []string{"\n\n"},
			[]string{"one line\n", "\nthe rest"}, "one line", "\n\n",
		},
		{
			"never matched", []string{"END"},
			[]string{"the answer"}, "the answer", "",
		},
		{
			"the earliest of several", []string{"ZZZ", "END"},
			[]string{"the answer END then ZZZ"}, "the answer ", "END",
		},
		{
			"a partial match is still the answer", []string{"END"},
			[]string{"the answer EN"}, "the answer EN", "",
		},
	}
	for _, c := range cases {
		d := NewChatDecoder(nil, false)
		d.StopAt(c.stop)
		_, content, calls, err := feed(d, c.pieces...)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if content != c.want {
			t.Errorf("%s: content %q, want %q", c.name, content, c.want)
		}
		if d.Stopped() != c.wantAt {
			t.Errorf("%s: stopped at %q, want %q", c.name, d.Stopped(), c.wantAt)
		}
		if len(calls) != 0 {
			t.Errorf("%s: %d calls", c.name, len(calls))
		}
	}
}

// TestChatDecoderStopIsNotWatchedInReasoning: the working is not the answer,
// and a stop of "\n\n" would end most generations before a word of the answer
// existed.
func TestChatDecoderStopIsNotWatchedInReasoning(t *testing.T) {
	d := NewChatDecoder(nil, true)
	d.StopAt([]string{"\n\n"})
	reasoning, content, _, err := feed(d, "first thought\n\nsecond thought</think>\n\nThe answer\n\nmore")
	if err != nil {
		t.Fatal(err)
	}
	if reasoning != "first thought\n\nsecond thought" {
		t.Errorf("reasoning %q", reasoning)
	}
	if content != "The answer" {
		t.Errorf("content %q", content)
	}
	if d.Stopped() != "\n\n" {
		t.Errorf("stopped at %q", d.Stopped())
	}
}
