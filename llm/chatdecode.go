package llm

// Reading a completion back (LLM.md L9a).
//
// What the model emits is one stream of text, and what a chat client expects
// is three fields: the reasoning, the answer, and any tool calls. The
// boundaries between them are the markers the prompt itself established --
// the generation prompt opens `<think>`, so the first thing the model writes
// is inside it, and the tool block told it to wrap a call in
// `<tool_call>`.
//
// The splitting has to happen **as the tokens arrive**, which is the whole
// difficulty: `</think>` can straddle two tokens, and a `<` that turns out to
// begin `<tool_call>` must not already have been sent to the client as an
// answer. So a decoder holds back any tail that could still be the start of a
// marker, and releases it as soon as the next piece settles the question.
// That is the only buffering here: everything else is forwarded the moment it
// is known.
//
// Tool calls are **not** streamed. A call is only a call once it has closed,
// its arguments have to be typed against the schema before they are JSON at
// all, and a client that was sent half a call would have to be told to
// retract it. So they are buffered and delivered whole, which is what
// `finish_reason: "tool_calls"` means.

import (
	"encoding/json"
	"strings"
)

// ChatCall is one call the model made, with its arguments rebuilt as the
// JSON object an OpenAI client reads.
type ChatCall struct {
	Name      string
	Arguments string
}

// decoder states: inside the reasoning block, in the answer, or buffering a
// tool call to the end of the generation.
const (
	stateThink = iota
	stateContent
	stateCall
	stateStopped
)

// ChatDecoder splits a generation into reasoning, content and tool calls.
// The zero value is not usable; NewChatDecoder builds one.
type ChatDecoder struct {
	// params is every tool's parameter types, by function name, so that a
	// value written between `<parameter=…>` tags can be turned back into
	// the JSON it was rendered from.
	params map[string]map[string]string

	state int
	// pend is the tail held back because it could still be the beginning of
	// a marker.
	pend string
	// call accumulates the tool block from the first `<tool_call>` to the
	// end of the generation.
	call strings.Builder
	// lead is set until the first non-blank content, so that the `\n\n` the
	// template puts after `</think>` does not open the answer.
	lead bool

	// stop is the request's stop sequences and stopped is the one that
	// matched, if any. They are watched in the answer and not in the
	// reasoning: a stop of "\n\n" would end a generation inside its own
	// working, before a word of the answer existed.
	stop    []string
	stopped string
}

// NewChatDecoder builds a decoder for a request's tools. thinking says
// whether the generation prompt left the `<think>` block open, which is what
// decides where the stream starts.
func NewChatDecoder(tools []ChatTool, thinking bool) *ChatDecoder {
	d := &ChatDecoder{params: toolParams(tools), state: stateContent, lead: true}
	if thinking {
		d.state = stateThink
	}
	return d
}

// StopAt sets the sequences that end the answer. It must be called before the
// first Write; empty strings are ignored.
//
// A stop sequence is another marker, which is why it costs nothing here but a
// line: the holdback that keeps `</think>` and `<tool_call>` from being
// forwarded half-written is the same one that keeps the first character of a
// stop sequence from being forwarded and then needing to be taken back.
func (d *ChatDecoder) StopAt(seqs []string) {
	d.stop = d.stop[:0]
	for _, s := range seqs {
		if s != "" {
			d.stop = append(d.stop, s)
		}
	}
}

// Stopped is the stop sequence that ended the generation, or the empty string.
// A caller polls it after each Write and stops feeding tokens when it is set.
func (d *ChatDecoder) Stopped() string { return d.stopped }

// Write feeds the next piece of generated text and returns what has become
// safe to emit: the reasoning and the answer, either of which may be empty,
// and both of which are non-empty when this piece crossed `</think>`.
func (d *ChatDecoder) Write(s string) (reasoning, content string) {
	d.pend += s
	for {
		switch d.state {
		case stateThink:
			if i := strings.Index(d.pend, thinkClose); i >= 0 {
				reasoning += d.pend[:i]
				d.pend = d.pend[i+len(thinkClose):]
				d.state = stateContent
				continue
			}
			held := partialSuffix(d.pend, thinkClose)
			reasoning += d.pend[:len(d.pend)-held]
			d.pend = d.pend[len(d.pend)-held:]
			return reasoning, content
		case stateContent:
			// The blank line the template puts between `</think>` and the
			// answer is dropped before anything is matched against it. It
			// belongs to the format, and a stop sequence of "\n\n" would
			// otherwise fire on it and end every generation with an empty
			// answer.
			if d.lead {
				if trimmed := strings.TrimLeft(d.pend, " \t\r\n"); trimmed != "" {
					d.pend = trimmed
				} else {
					d.pend = ""
					return reasoning, content
				}
			}
			// Whichever marker comes first ends the answer: a stop sequence
			// ends the generation, a `<tool_call>` ends the prose and begins
			// a call.
			at, stop := -1, ""
			if i := strings.Index(d.pend, callOpen); i >= 0 {
				at = i
			}
			for _, seq := range d.stop {
				if i := strings.Index(d.pend, seq); i >= 0 && (at < 0 || i < at) {
					at, stop = i, seq
				}
			}
			switch {
			case at >= 0 && stop != "":
				// The stop sequence is not part of the answer, and nothing
				// after it is either.
				content += d.answer(d.pend[:at])
				d.pend = ""
				d.stopped = stop
				d.state = stateStopped
				return reasoning, content
			case at >= 0:
				// The blank line before a call is the format's separator
				// and not the answer's, which is why the answer's own
				// trailing whitespace is never forwarded until something
				// follows it.
				content += d.answer(strings.TrimRight(d.pend[:at], " \t\r\n"))
				d.call.WriteString(d.pend[at:])
				d.pend = ""
				d.state = stateCall
				return reasoning, content
			}
			held := max(partialSuffix(d.pend, callOpen), trailingSpace(d.pend))
			for _, seq := range d.stop {
				held = max(held, partialSuffix(d.pend, seq))
			}
			content += d.answer(d.pend[:len(d.pend)-held])
			d.pend = d.pend[len(d.pend)-held:]
			return reasoning, content
		case stateStopped:
			// Past a stop sequence there is nothing to report, and the
			// caller has already been told to stop feeding tokens.
			d.pend = ""
			return reasoning, content
		default: // stateCall
			d.call.WriteString(d.pend)
			d.pend = ""
			return reasoning, content
		}
	}
}

// Close flushes whatever was held back and parses the tool calls. The text it
// returns is text the generation ended in the middle of a marker on -- a
// truncated `</thin`, which was never a marker at all and is part of the
// answer.
func (d *ChatDecoder) Close() (reasoning, content string, calls []ChatCall, err error) {
	switch d.state {
	case stateThink:
		reasoning = d.pend
	case stateContent:
		content = d.answer(d.pend)
	}
	d.pend = ""
	if d.call.Len() > 0 {
		if calls, err = parseToolCalls(d.call.String(), d.params); err != nil {
			return reasoning, content, nil, err
		}
	}
	return reasoning, content, calls, nil
}

// answer trims the blank line the template puts between `</think>` and the
// answer, and only there: once any non-blank text has been emitted the
// stream is passed through untouched, because whitespace inside an answer is
// the answer's.
func (d *ChatDecoder) answer(s string) string {
	if !d.lead {
		return s
	}
	if t := strings.TrimLeft(s, " \t\r\n"); t != "" {
		d.lead = false
		return t
	}
	return ""
}

// partialSuffix is how many bytes at the end of s could still be the
// beginning of marker. It is the reason a `<` at the end of a token is not
// forwarded until the next one arrives.
func partialSuffix(s, marker string) int {
	n := len(marker) - 1
	if n > len(s) {
		n = len(s)
	}
	for ; n > 0; n-- {
		if strings.HasSuffix(s, marker[:n]) {
			return n
		}
	}
	return 0
}

// trailingSpace is how many bytes of whitespace s ends with. They are held
// back for the same reason a partial marker is: if a `<tool_call>` follows,
// they were the separator in front of it rather than part of the answer, and
// a streamed answer has to agree with the one a buffered request returns.
func trailingSpace(s string) int {
	return len(s) - len(strings.TrimRight(s, " \t\r\n"))
}

// parseToolCalls reads the format the tool block asked for:
//
//	<tool_call>
//	<function=name>
//	<parameter=key>
//	value
//	</parameter>
//	</function>
//	</tool_call>
//
// An unterminated call at the end -- a generation that hit the token limit
// mid-call -- is dropped rather than half-reported: the client would have no
// way to tell a truncated argument from a short one.
func parseToolCalls(s string, params map[string]map[string]string) ([]ChatCall, error) {
	var out []ChatCall
	for {
		i := strings.Index(s, callOpen)
		if i < 0 {
			return out, nil
		}
		s = s[i+len(callOpen):]
		end := strings.Index(s, callClose)
		if end < 0 {
			return out, nil // truncated; see above
		}
		body := s[:end]
		s = s[end+len(callClose):]
		call, ok, err := parseOneCall(body, params)
		if err != nil {
			return nil, err
		}
		if ok {
			out = append(out, call)
		}
	}
}

// parseOneCall reads the inside of one `<tool_call>` block. It reports
// whether there was a call in it at all, so that an empty or malformed block
// is skipped rather than turned into a call with no name.
func parseOneCall(body string, params map[string]map[string]string) (ChatCall, bool, error) {
	open := strings.Index(body, "<function=")
	if open < 0 {
		return ChatCall{}, false, nil
	}
	rest := body[open+len("<function="):]
	gt := strings.Index(rest, ">")
	if gt < 0 {
		return ChatCall{}, false, nil
	}
	name := strings.TrimSpace(rest[:gt])
	if name == "" {
		return ChatCall{}, false, nil
	}
	rest = rest[gt+1:]
	if i := strings.Index(rest, "</function>"); i >= 0 {
		rest = rest[:i]
	}

	types := params[name]
	var args strings.Builder
	args.WriteByte('{')
	n := 0
	for {
		p := strings.Index(rest, "<parameter=")
		if p < 0 {
			break
		}
		rest = rest[p+len("<parameter="):]
		gt := strings.Index(rest, ">")
		if gt < 0 {
			break
		}
		key := strings.TrimSpace(rest[:gt])
		rest = rest[gt+1:]
		shut := strings.Index(rest, "</parameter>")
		if shut < 0 {
			break
		}
		// The template writes `\n` after the opening tag and before the
		// closing one; they are the format's and not the value's.
		value := strings.TrimSuffix(strings.TrimPrefix(rest[:shut], "\n"), "\n")
		rest = rest[shut+len("</parameter>"):]

		if n > 0 {
			args.WriteByte(',')
		}
		n++
		k, err := json.Marshal(key)
		if err != nil {
			return ChatCall{}, false, err
		}
		args.Write(k)
		args.WriteByte(':')
		args.WriteString(argJSON(value, types[key]))
	}
	args.WriteByte('}')
	return ChatCall{Name: name, Arguments: args.String()}, true, nil
}

// argJSON turns one rendered parameter back into JSON, which needs the
// schema: the template wrote a string value raw and everything else as JSON,
// so `123` is the number 123 for an integer parameter and the text "123" for
// a string one. Without a type -- a call to a function the request did not
// declare -- valid JSON is taken at face value and anything else is a string,
// which is the reading that loses the least.
func argJSON(value, typ string) string {
	if typ != "string" {
		trimmed := strings.TrimSpace(value)
		if json.Valid([]byte(trimmed)) && trimmed != "" {
			return trimmed
		}
	}
	b, err := json.Marshal(value)
	if err != nil { // impossible for a string
		return `""`
	}
	return string(b)
}

// toolParams reads every tool's parameter types out of its JSON schema. Both
// shapes are accepted -- OpenAI's `{"type":"function","function":{…}}` and
// the bare function object -- because both are sent in practice and the
// template passes either through untouched.
func toolParams(tools []ChatTool) map[string]map[string]string {
	out := map[string]map[string]string{}
	for _, t := range tools {
		var fn struct {
			Name     string `json:"name"`
			Function *struct {
				Name       string `json:"name"`
				Parameters struct {
					Properties map[string]struct {
						Type json.RawMessage `json:"type"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"function"`
			Parameters struct {
				Properties map[string]struct {
					Type json.RawMessage `json:"type"`
				} `json:"properties"`
			} `json:"parameters"`
		}
		if err := json.Unmarshal(t.JSON, &fn); err != nil {
			continue
		}
		name, props := fn.Name, fn.Parameters.Properties
		if fn.Function != nil {
			name, props = fn.Function.Name, fn.Function.Parameters.Properties
		}
		if name == "" {
			continue
		}
		types := map[string]string{}
		for k, v := range props {
			types[k] = schemaType(v.Type)
		}
		out[name] = types
	}
	return out
}

// schemaType reads a JSON Schema `type`, which is a string or a list of
// them. A union counts as a string only if every arm is one, because that is
// the only case where the template certainly wrote the value raw.
func schemaType(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err != nil || len(list) == 0 {
		return ""
	}
	for _, t := range list {
		if t != "string" && t != "null" {
			return ""
		}
	}
	return "string"
}
