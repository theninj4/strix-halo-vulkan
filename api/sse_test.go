package api

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The heartbeat. A streamed generation can be silent for minutes while it is
// still being worked on -- queued for a slot, prefilling, or writing a tool
// call the decoder holds back until it closes -- and a client reads that
// silence as a dead connection. So these run over a real connection and time
// every line as it arrives, which a ResponseRecorder cannot do.

// quietLLM is a generation with the two silences a real one has: wait before
// the first piece (the queue and the prefill), and quiet after the last (a
// tool call being written, which only arrives in the result).
type quietLLM struct {
	wait, quiet time.Duration
	pieces      []Delta
	res         *CompletionResult
}

func (f *quietLLM) Models() []Model {
	return []Model{{ID: "qwen3.8-flash-next", Object: "model", OwnedBy: "local"}}
}

func (f *quietLLM) Complete(ctx context.Context, _ *CompletionRequest,
	emit func(Delta) error,
) (*CompletionResult, error) {
	pause := func(d time.Duration) error {
		select {
		case <-time.After(d):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if err := pause(f.wait); err != nil {
		return nil, err
	}
	for _, p := range f.pieces {
		if err := emit(p); err != nil {
			return nil, err
		}
	}
	if err := pause(f.quiet); err != nil {
		return nil, err
	}
	return f.res, nil
}

// writeCall is a large `write` call, the shape of every turn opencode lost.
func writeCall() *CompletionResult {
	args, _ := json.Marshal(map[string]string{
		"filePath": "/src/main.go",
		"content":  strings.Repeat("func f() {}\n", 700),
	})
	return &CompletionResult{
		FinishReason: "tool_calls",
		ToolCalls: []*ToolCall{{
			ID: "call_0000beef", Type: "function", Index: 0,
			Function: ToolCallReq{Name: "write", Arguments: string(args)},
		}},
		Usage: Usage{PromptTokens: 7, CompletionTokens: 9000, TotalTokens: 9007},
	}
}

// timedLine is one line of the response and when it arrived.
type timedLine struct {
	at   time.Time
	text string
}

// streamTimed posts body to path and reads the response line by line, timing each.
func streamTimed(t *testing.T, s *Server, path string, body any) (lines []timedLine, start time.Time) {
	t.Helper()
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	req := jsonRequest("POST", srv.URL+path, body)
	req.RequestURI = ""
	start = time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		lines = append(lines, timedLine{time.Now(), sc.Text()})
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return lines, start
}

// longestGap is the longest the client went without a line, from the request
// to the last line.
func longestGap(lines []timedLine, start time.Time) time.Duration {
	var gap time.Duration
	prev := start
	for _, l := range lines {
		gap = max(gap, l.at.Sub(prev))
		prev = l.at
	}
	return gap
}

// sseFrames drops the heartbeats and the blank separators, leaving what a client
// parses. A heartbeat must leave it exactly as it would have been.
func sseFrames(lines []timedLine) (out []string, beats int) {
	for _, l := range lines {
		switch {
		case l.text == ": keep-alive":
			beats++
		case l.text == "":
		default:
			out = append(out, l.text)
		}
	}
	return out, beats
}

// TestStreamKeepAlive runs each envelope through a silent queue wait and a
// silent tool call twelve heartbeats long, and checks that the client never
// waits more than a beat (and some slack for the scheduler) for a byte, and
// that with the heartbeats taken out the stream is frame for frame the one an
// unhurried generation produces.
func TestStreamKeepAlive(t *testing.T) {
	const beat = 40 * time.Millisecond
	cases := []struct {
		path string
		body map[string]any
	}{
		{"/v1/chat/completions", map[string]any{
			"model": "gpt-4", "stream": true,
			"messages": []map[string]any{{"role": "user", "content": "write it"}},
		}},
		{"/v1/responses", map[string]any{"model": "gpt-5", "stream": true, "input": "write it"}},
		{"/v1/messages", map[string]any{
			"model": "claude", "stream": true, "max_tokens": 32,
			"messages": []map[string]any{{"role": "user", "content": "write it"}},
		}},
	}
	pieces := []Delta{{ReasoningContent: "The file."}, {Content: "Writing it."}}
	for _, c := range cases {
		t.Run(strings.TrimPrefix(c.path, "/v1/"), func(t *testing.T) {
			slow := &Server{KeepAlive: beat, Completion: &quietLLM{
				wait: 6 * beat, quiet: 12 * beat, pieces: pieces, res: writeCall(),
			}}
			lines, start := streamTimed(t, slow, c.path, c.body)
			if gap := longestGap(lines, start); gap > 3*beat {
				t.Errorf("the client went %v without a byte; the heartbeat is every %v", gap, beat)
			}
			got, beats := sseFrames(lines)
			// Roughly one a beat across eighteen beats of silence.
			if beats < 12 {
				t.Errorf("%d heartbeats across %v of silence", beats, 18*beat)
			}

			fast := &Server{KeepAlive: time.Hour, Completion: &quietLLM{pieces: pieces, res: writeCall()}}
			ref, _ := streamTimed(t, fast, c.path, c.body)
			want, n := sseFrames(ref)
			if n != 0 {
				t.Errorf("%d heartbeats in a stream that was never silent", n)
			}
			// The ids and timestamps differ between two runs, so the frames
			// are compared with those masked.
			if !equalFrames(got, want) {
				t.Errorf("the heartbeat changed the stream:\n got %q\nwant %q", got, want)
			}
		})
	}
}

// TestStreamToolCallFinalChunk pins the chat stream's last chunk after a long
// silent tool call: the call whole, on index 0, with arguments that parse, and
// finish_reason tool_calls, followed by the sentinel.
func TestStreamToolCallFinalChunk(t *testing.T) {
	const beat = 20 * time.Millisecond
	s := &Server{KeepAlive: beat, Completion: &quietLLM{quiet: 10 * beat, res: writeCall()}}
	lines, _ := streamTimed(t, s, "/v1/chat/completions", map[string]any{
		"model": "gpt-4", "stream": true,
		"messages": []map[string]any{{"role": "user", "content": "write it"}},
	})
	got, beats := sseFrames(lines)
	if beats == 0 {
		t.Error("no heartbeat during the call")
	}
	if len(got) < 3 || got[len(got)-1] != "data: [DONE]" {
		t.Fatalf("frames %q", got)
	}
	var last Chunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(got[len(got)-2], "data: ")), &last); err != nil {
		t.Fatal(err)
	}
	if len(last.Choices) != 1 || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("last chunk %+v", last)
	}
	calls := last.Choices[0].Delta.ToolCalls
	if len(calls) != 1 || calls[0].Index != 0 || calls[0].ID != "call_0000beef" || calls[0].Function.Name != "write" {
		t.Fatalf("calls %+v", calls)
	}
	var args struct{ FilePath, Content string }
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments do not parse: %v", err)
	}
	if args.FilePath != "/src/main.go" || len(args.Content) != 700*len("func f() {}\n") {
		t.Errorf("arguments %q, %d bytes of content", args.FilePath, len(args.Content))
	}
}

// equalFrames compares two streams with the fields that differ run to run --
// ids, creation times -- masked.
func equalFrames(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if maskVolatile(a[i]) != maskVolatile(b[i]) {
			return false
		}
	}
	return true
}

func maskVolatile(line string) string {
	data, ok := strings.CutPrefix(line, "data: ")
	if !ok {
		return line
	}
	var v any
	if json.Unmarshal([]byte(data), &v) != nil {
		return line
	}
	var walk func(any) any
	walk = func(v any) any {
		switch x := v.(type) {
		case map[string]any:
			for k, e := range x {
				switch k {
				case "id", "created", "created_at", "item_id":
					x[k] = "*"
				default:
					x[k] = walk(e)
				}
			}
		case []any:
			for i, e := range x {
				x[i] = walk(e)
			}
		}
		return v
	}
	b, _ := json.Marshal(walk(v))
	return "data: " + string(b)
}
