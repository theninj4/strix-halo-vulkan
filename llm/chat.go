package llm

// The chat template, transcribed (LLM.md L9a).
//
// A chat request is a list of messages and this model reads one string, so
// something has to render the conversation the way the checkpoint was
// trained to see it. That rendering is not ours to invent: it is
// `tokenizer.chat_template` in the GGUF's own metadata, 180 lines of Jinja,
// and a server that improvises it produces a prompt the model has never been
// shown -- which does not fail, it just answers slightly wrong, everywhere,
// with nothing to point at.
//
// So this file is a transcription rather than a design, in the same spirit as
// `zimage/tokenizer.ChatPrompt` but for the whole of it: the merged system
// turn, the reasoning-effort instructions, the tool block's wire format, the
// `<think>` block on an assistant turn, and the generation prompt. The
// acceptance criterion is Jinja's own output, case for case, character for
// character -- `reference/dump_chat_template.py` renders the template out of
// the checkpoint with the same environment transformers uses, and
// TestRenderChat diffs against it.
//
// An image content part renders as the template renders it (LLM-VISION.md
// V7): `<|vision_start|><|image_pad|><|vision_end|>` in place, optionally
// "Picture N: " before it, one pad that ExpandImages then widens to the
// image's grid. A video part is still the caller's to refuse.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ChatMessage is one turn of a conversation, flattened to the text the
// template reads. `Content` is the message's text blocks joined; a
// non-text block is the caller's to refuse before it gets here.
type ChatMessage struct {
	// Role is "system", "developer", "user", "assistant" or "tool".
	Role string
	// Content is the turn's text.
	Content string
	// Parts is the turn as the template's content list, text and images in
	// order. When it is set Content is not read. A system turn may not hold
	// an image, as the template says.
	Parts []ChatPart
	// Reasoning is an assistant turn's `<think>` block, which this
	// checkpoint keeps in the history rather than dropping. A client that
	// replays a previous completion's `reasoning_content` puts it here.
	Reasoning string
	// ToolCalls are an assistant turn's calls, in the order they were made.
	ToolCalls []ChatToolCall
}

// ChatPart is one item of a content list: its text, or an image.
type ChatPart struct {
	Text  string
	Image bool
}

// ImageMarkup is what the template writes for an image part, before the
// processor widens its one pad to the image's tokens.
const ImageMarkup = "<|vision_start|><|image_pad|><|vision_end|>"

// content is the template's `render_content(message.content,
// do_vision_count, is_system_content)`, untrimmed. `counting` is
// do_vision_count: whether an image advances *seen, the conversation's image
// counter, whose value is what "Picture N: " prints either way.
func (m ChatMessage) content(counting bool, seen *int, addID, system bool) (string, error) {
	if m.Parts == nil {
		return m.Content, nil
	}
	var b strings.Builder
	for _, p := range m.Parts {
		if !p.Image {
			b.WriteString(p.Text)
			continue
		}
		if system {
			return "", fmt.Errorf("llm: a system message cannot contain images")
		}
		if counting {
			*seen++
		}
		if addID {
			fmt.Fprintf(&b, "Picture %d: ", *seen)
		}
		b.WriteString(ImageMarkup)
	}
	return b.String(), nil
}

// ChatToolCall is one call: the function's name and its arguments **in
// order**, because the template writes one `<parameter=…>` block per
// argument in the order the object carried them and a Go map has none.
type ChatToolCall struct {
	Name string
	Args []ChatArg
}

// ChatArg is one argument. Value is the JSON that arrived: a JSON string is
// written raw between the parameter tags, anything else is written as JSON,
// which is the rule the template states and the one ParseToolCalls inverts.
type ChatArg struct {
	Name  string
	Value json.RawMessage
}

// ChatTool is one entry of the request's `tools` array, as it arrived. It is
// carried as raw JSON and re-encoded rather than modelled, because the
// template writes `tool | tojson` -- whatever the client sent, including the
// fields this server has no opinion about.
type ChatTool struct {
	JSON json.RawMessage
}

// ChatOpts are the template's own switches. Every zero value is the
// template's default, so a caller that wants what the model card describes
// passes nothing.
type ChatOpts struct {
	// Tools is the request's tool array. Non-empty replaces the system turn
	// with the tool block, which carries the system text at its end.
	Tools []ChatTool
	// Effort is the reasoning effort: "", "low", "medium", "high" or
	// "xhigh". Empty and "high" are both "xhigh", which is the template's
	// default; "medium" is the one effort with no instructions of its own.
	Effort string
	// NoThinking is `enable_thinking=false`: no reasoning instructions, and
	// a generation prompt that closes the `<think>` block before the model
	// can open it.
	NoThinking bool
	// DropThinking is `preserve_thinking=false`: an assistant turn at or
	// before the last user message is rendered without its reasoning. It is
	// off by default because the template's default is to keep it.
	DropThinking bool
	// NoGenerationPrompt leaves off the trailing `<|im_start|>assistant`,
	// which is what a caller scoring an existing conversation wants.
	NoGenerationPrompt bool
	// AddVisionID is the template's `add_vision_id`: "Picture N: " before
	// each image, numbered across the conversation.
	AddVisionID bool
}

// The reasoning instructions, one per effort. "medium" has none, which is
// not an omission here: the template sets the string empty for it.
const (
	xhighInstructions = "Reasoning effort is set to xhigh. Please think carefully through the task, " +
		"validate key assumptions, consider plausible alternatives, and prioritize correctness, " +
		"consistency, and clarity in the final answer."
	lowInstructions = "Reasoning effort is set to low. Keep your thinking brief and focused, " +
		"moving directly to the conclusion without unnecessary elaboration."
)

// The tool block's preamble and its rules, verbatim from the template. They
// are what the model was trained to emit calls against, so ParseToolCalls
// reads back exactly the format this text asks for.
const (
	toolsPreamble = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
	toolsRules    = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n" +
		"multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block " +
		"must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the " +
		"function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current " +
		"knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

// The markers the rendering and the reading of a completion share.
const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
	callOpen   = "<tool_call>"
	callClose  = "</tool_call>"
)

// RenderChat renders a conversation the way the checkpoint's template does.
//
// The errors are the template's own `raise_exception` calls, which are the
// cases where it would rather stop than render something the model has not
// seen: no messages at all, a system turn that is not at the beginning, an
// unknown role, an effort that is not one of the three.
func RenderChat(msgs []ChatMessage, opt ChatOpts) (string, error) {
	if len(msgs) == 0 {
		return "", fmt.Errorf("llm: no messages provided")
	}

	// The leading run of system and developer turns is merged into one, and
	// only that run: a system message further down is an error rather than a
	// turn, because this template has nowhere to put it.
	numSys := 0
	var merged strings.Builder
	for i, m := range msgs {
		if numSys != i || (m.Role != "system" && m.Role != "developer") {
			break
		}
		raw, err := m.content(false, new(int), opt.AddVisionID, true)
		if err != nil {
			return "", err
		}
		if c := strings.TrimSpace(raw); c != "" {
			if merged.Len() > 0 {
				merged.WriteByte('\n')
			}
			merged.WriteString(c)
		}
		numSys++
	}
	system := merged.String()

	instructions, err := reasoningInstructions(opt)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	switch {
	case len(opt.Tools) > 0:
		b.WriteString("<|im_start|>system\n")
		if instructions != "" {
			b.WriteString(instructions + "\n\n")
		}
		b.WriteString(toolsPreamble)
		for i, t := range opt.Tools {
			js, err := pyJSON(t.JSON)
			if err != nil {
				return "", fmt.Errorf("llm: tool %d: %w", i, err)
			}
			b.WriteString("\n" + js)
		}
		b.WriteString("\n</tools>")
		b.WriteString(toolsRules)
		if system != "" {
			b.WriteString("\n\n" + system)
		}
		b.WriteString("<|im_end|>\n")
	case system != "":
		b.WriteString("<|im_start|>system\n")
		if instructions != "" {
			b.WriteString(instructions + "\n\n")
		}
		b.WriteString(system + "<|im_end|>\n")
	case instructions != "":
		b.WriteString("<|im_start|>system\n" + instructions + "<|im_end|>\n")
	}

	lastQuery, err := lastQueryIndex(msgs, opt.AddVisionID)
	if err != nil {
		return "", err
	}

	images := 0
	for i, m := range msgs {
		if i < numSys {
			continue
		}
		raw, err := m.content(true, &images, opt.AddVisionID, false)
		if err != nil {
			return "", err
		}
		content := strings.TrimSpace(raw)
		switch m.Role {
		case "system", "developer":
			return "", fmt.Errorf("llm: a system message must be at the beginning of the conversation")
		case "user":
			b.WriteString("<|im_start|>user\n" + content + "<|im_end|>\n")
		case "assistant":
			if !opt.DropThinking || i > lastQuery {
				b.WriteString("<|im_start|>assistant\n<think>\n" +
					strings.TrimSpace(m.Reasoning) + "\n</think>\n\n" + content)
			} else {
				b.WriteString("<|im_start|>assistant\n" + content)
			}
			for j, call := range m.ToolCalls {
				if call.Name == "" {
					return "", fmt.Errorf("llm: tool call %d is missing a function name", j)
				}
				switch {
				case j > 0:
					b.WriteString("\n<tool_call>\n<function=" + call.Name + ">\n")
				case content != "":
					b.WriteString("\n\n<tool_call>\n<function=" + call.Name + ">\n")
				default:
					b.WriteString("<tool_call>\n<function=" + call.Name + ">\n")
				}
				for _, a := range call.Args {
					v, err := renderArg(a.Value)
					if err != nil {
						return "", fmt.Errorf("llm: tool call %q, argument %q: %w", call.Name, a.Name, err)
					}
					b.WriteString("<parameter=" + a.Name + ">\n" + v + "\n</parameter>\n")
				}
				b.WriteString("</function>\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			// A run of tool results is one user turn: the opener is written
			// at the first of them and the closer at the last, so several
			// results of one assistant turn arrive together.
			if i > 0 && msgs[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(msgs)-1 || msgs[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		default:
			return "", fmt.Errorf("llm: unexpected message role %q", m.Role)
		}
	}

	if !opt.NoGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
		if opt.NoThinking {
			// The block is opened and closed before the model writes a
			// token, which is how "no thinking" is enforced by the prompt
			// rather than hoped for.
			b.WriteString("<think>\n\n</think>\n\n")
		} else {
			b.WriteString("<think>\n")
		}
	}
	return b.String(), nil
}

// reasoningInstructions resolves the effort to the sentence the system turn
// carries. `high` is an alias for `xhigh` -- the template's own aliasing, not
// a kindness to OpenAI's vocabulary -- and `medium` resolves to no sentence
// at all.
func reasoningInstructions(opt ChatOpts) (string, error) {
	if opt.NoThinking {
		return "", nil
	}
	effort := opt.Effort
	if effort == "" {
		effort = "xhigh"
	}
	if effort == "high" {
		effort = "xhigh"
	}
	switch effort {
	case "xhigh":
		return xhighInstructions, nil
	case "low":
		return lowInstructions, nil
	case "medium":
		return "", nil
	default:
		return "", fmt.Errorf("llm: unexpected reasoning effort %q; supported are xhigh (default), medium and low", effort)
	}
}

// lastQueryIndex is the last user turn that is a question rather than a tool
// result, which is what decides whether an assistant turn keeps its
// reasoning under DropThinking: everything after the user's last real message
// is this answer's own working, and everything before it is history.
//
// The template's scan renders without counting images, so a "Picture N: "
// here always reads 0, which only matters in that it is not a tool response.
func lastQueryIndex(msgs []ChatMessage, addID bool) (int, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		raw, err := msgs[i].content(false, new(int), addID, false)
		if err != nil {
			return 0, err
		}
		c := strings.TrimSpace(raw)
		if strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>") {
			continue
		}
		return i, nil
	}
	return len(msgs) - 1, nil
}

// renderArg writes one argument the way the template does: a JSON string
// unquoted, so a multi-line value spans lines between its tags, and anything
// else as JSON.
func renderArg(v json.RawMessage) (string, error) {
	if len(v) == 0 {
		return "", nil
	}
	if v[0] == '"' {
		var s string
		if err := json.Unmarshal(v, &s); err != nil {
			return "", err
		}
		return s, nil
	}
	return pyJSON(v)
}

// ParseToolArguments turns OpenAI's `arguments` -- a JSON *string* holding a
// JSON object -- into the ordered arguments the template writes.
//
// The order is the point. `json.Unmarshal` into a map loses it, and the
// template emits one `<parameter=…>` block per argument in the object's own
// order, so a replayed assistant turn would otherwise be rendered differently
// from the one the model produced.
func ParseToolArguments(s string) ([]ChatArg, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("llm: tool arguments are not a JSON object")
	}
	var out []ChatArg
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name, ok := key.(string)
		if !ok {
			return nil, fmt.Errorf("llm: tool argument name is not a string")
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, err
		}
		out = append(out, ChatArg{Name: name, Value: raw})
	}
	return out, nil
}
