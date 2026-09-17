package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"

	"strix-halo-vulkan/util"
)

// POST /v1/messages -- Anthropic's Messages API over the same backend
// (LLM.md L9a).
//
// The third envelope, and the one that is least like the other two. A message
// is a *list of blocks* rather than a string with fields beside it: the
// reasoning is a `thinking` block, a call is a `tool_use` block, and its
// result comes back as a `tool_result` block inside a user turn. That is
// closer to what this checkpoint's template actually renders than OpenAI's
// shape is, so the translation in here is mostly a matter of putting the
// blocks in the right order.
//
// One thing is missing rather than translated: a `thinking` block from
// Anthropic carries a `signature`, which is a cryptographic receipt from
// their inference stack. This server has nothing to sign with and does not
// invent one. A client that verifies signatures will not accept these blocks,
// and it should not.

type MessagesRequest struct {
	Thinking      *AnthropicThinking `json:"thinking,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	Model         string             `json:"model"`
	System        json.RawMessage    `json:"system,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
	Messages      []AnthropicMessage `json:"messages"`
	Tools         []AnthropicTool    `json:"tools,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Stream        bool               `json:"stream"`
}

// AnthropicMessage is one turn. Content is a string or a list of blocks, and
// both are read.
type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// AnthropicBlock is every block type at once -- text, thinking, tool_use and
// tool_result -- for the same reason InputItem is: which one it is is decided
// by `type`, and one struct that reads all four is less code than four that
// each read one.
type AnthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type AnthropicTool struct {
	InputSchema util.JSONSchemaType `json:"input_schema"`
	Type        string              `json:"type,omitempty"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
}

// AnthropicThinking is the `thinking` object: `enabled` with a token budget,
// or `disabled`. The budget is not honoured -- this server's reasoning stops
// where the model stops it, not at a count -- so a request that names one is
// refused rather than quietly served without it.
type AnthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type MessagesResponse struct {
	StopReason   *string          `json:"stop_reason"`
	StopSequence *string          `json:"stop_sequence"`
	ID           string           `json:"id"`
	Type         string           `json:"type"`
	Role         string           `json:"role"`
	Model        string           `json:"model"`
	Content      []AnthropicBlock `json:"content"`
	Usage        AnthropicUsage   `json:"usage"`
}

// messagesEvent is one frame of a streamed message. As with the Responses
// API, the payload differs per event name and this carries the union.
type messagesEvent struct {
	Message      *MessagesResponse `json:"message,omitempty"`
	ContentBlock *AnthropicBlock   `json:"content_block,omitempty"`
	Delta        *messagesDelta    `json:"delta,omitempty"`
	Usage        *AnthropicUsage   `json:"usage,omitempty"`
	Type         string            `json:"type"`
	Index        int               `json:"index,omitempty"`
}

// messagesDelta is both deltas the protocol has: the one inside a content
// block, and the one on the message that carries the stop reason.
type messagesDelta struct {
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
	Type         string  `json:"type,omitempty"`
	Text         string  `json:"text,omitempty"`
	Thinking     string  `json:"thinking,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
}

// handleMessages is the Anthropic endpoint.
func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	if s.Completion == nil {
		notLoaded(w, "the language model", "-llm")
		return
	}
	var req MessagesRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.MaxTokens <= 0 {
		badRequest(w, "max_tokens is required and must be positive")
		return
	}
	if req.Thinking != nil && req.Thinking.BudgetTokens > 0 {
		badRequest(w, "thinking.budget_tokens is not implemented; this model's reasoning ends where it ends, "+
			"and a budget that is not enforced is worse than one that is refused")
		return
	}
	comp, err := completionFromMessages(&req)
	if err != nil {
		badRequest(w, err.Error())
		return
	}
	if !validCompletion(w, comp) {
		return
	}

	id := "msg_" + randomID()
	model := modelID(s.Completion, req.Model)
	if req.Stream {
		s.streamMessages(w, r, comp, id, model)
		return
	}

	var content, reasoning strings.Builder
	res, err := s.Completion.Complete(r.Context(), comp, func(d Delta) error {
		content.WriteString(d.Content)
		reasoning.WriteString(d.ReasoningContent)
		return nil
	})
	if err != nil {
		backendError(w, "messages", err)
		return
	}
	stop, seq := stopReason(res)
	writeJSON(w, http.StatusOK, MessagesResponse{
		ID: id, Type: "message", Role: "assistant", Model: model,
		Content:      messageBlocks(reasoning.String(), content.String(), res.ToolCalls),
		StopReason:   &stop,
		StopSequence: seq,
		Usage: AnthropicUsage{
			InputTokens: res.Usage.PromptTokens, OutputTokens: res.Usage.CompletionTokens,
		},
	})
}

// messageBlocks is the response's content: the thinking, then the text, then
// one tool_use per call, each left out when it is empty.
func messageBlocks(reasoning, content string, calls []*ToolCall) []AnthropicBlock {
	out := []AnthropicBlock{}
	if reasoning != "" {
		out = append(out, AnthropicBlock{Type: "thinking", Thinking: reasoning})
	}
	if content != "" {
		out = append(out, AnthropicBlock{Type: "text", Text: content})
	}
	for _, c := range calls {
		out = append(out, AnthropicBlock{
			Type: "tool_use", ID: c.ID, Name: c.Function.Name,
			Input: json.RawMessage(c.Function.Arguments),
		})
	}
	return out
}

// stopReason translates the backend's word into Anthropic's, and names the
// stop sequence when one was what ended it.
//
// This is the one place the three envelopes genuinely disagree about an
// outcome rather than a shape: OpenAI folds a stop sequence into "stop", and
// Anthropic reports `stop_sequence` and says which.
func stopReason(res *CompletionResult) (string, *string) {
	switch {
	case res.StopSequence != "":
		seq := res.StopSequence
		return "stop_sequence", &seq
	case res.FinishReason == "length":
		return "max_tokens", nil
	case res.FinishReason == "tool_calls":
		return "tool_use", nil
	default:
		return "end_turn", nil
	}
}

// streamMessages writes the generation as Anthropic's events.
//
// The protocol is a state machine over *content blocks*: one is started, its
// deltas arrive, it is stopped, and the next one begins. So a generation that
// reasons and then answers opens two blocks, and the switch between them
// happens at the first content token -- which is exactly where the decoder in
// `llm` puts the boundary.
func (s *Server) streamMessages(w http.ResponseWriter, r *http.Request,
	comp *CompletionRequest, id, model string,
) {
	stream := newSSE(w)
	fail := func(err error) {
		if r.Context().Err() != nil {
			log.Printf("api: messages: client cancelled")
			return
		}
		log.Printf("api: messages: %v", err)
		_ = stream.send("error", map[string]any{
			"type":  "error",
			"error": errorBody{Message: err.Error(), Type: "server_error"},
		})
	}
	send := func(typ string, e messagesEvent) error {
		e.Type = typ
		return stream.send(typ, e)
	}

	if err := send("message_start", messagesEvent{Message: &MessagesResponse{
		ID: id, Type: "message", Role: "assistant", Model: model,
		Content: []AnthropicBlock{},
	}}); err != nil {
		fail(err)
		return
	}

	// index is the content block being written; open says whether there is
	// one, and thinking says which kind it is.
	index, open, thinking := 0, false, false
	closeBlock := func() error {
		if !open {
			return nil
		}
		open = false
		if err := send("content_block_stop", messagesEvent{Index: index}); err != nil {
			return err
		}
		index++
		return nil
	}

	res, err := s.Completion.Complete(r.Context(), comp, func(d Delta) error {
		if d.ReasoningContent != "" {
			if !open {
				open, thinking = true, true
				if err := send("content_block_start", messagesEvent{
					Index: index, ContentBlock: &AnthropicBlock{Type: "thinking"},
				}); err != nil {
					return err
				}
			}
			if err := send("content_block_delta", messagesEvent{
				Index: index, Delta: &messagesDelta{Type: "thinking_delta", Thinking: d.ReasoningContent},
			}); err != nil {
				return err
			}
		}
		if d.Content == "" {
			return nil
		}
		if open && thinking {
			if err := closeBlock(); err != nil {
				return err
			}
		}
		if !open {
			open, thinking = true, false
			if err := send("content_block_start", messagesEvent{
				Index: index, ContentBlock: &AnthropicBlock{Type: "text"},
			}); err != nil {
				return err
			}
		}
		return send("content_block_delta", messagesEvent{
			Index: index, Delta: &messagesDelta{Type: "text_delta", Text: d.Content},
		})
	})
	if err != nil {
		fail(err)
		return
	}
	if err := closeBlock(); err != nil {
		fail(err)
		return
	}
	// A call's arguments are known whole, so its block is started, given one
	// `input_json_delta` and stopped. A client assembling partial JSON gets
	// a single piece that happens to be the whole of it.
	for _, c := range res.ToolCalls {
		if err := send("content_block_start", messagesEvent{
			Index: index,
			ContentBlock: &AnthropicBlock{
				Type: "tool_use", ID: c.ID, Name: c.Function.Name, Input: json.RawMessage(`{}`),
			},
		}); err != nil {
			fail(err)
			return
		}
		if err := send("content_block_delta", messagesEvent{
			Index: index,
			Delta: &messagesDelta{Type: "input_json_delta", PartialJSON: c.Function.Arguments},
		}); err != nil {
			fail(err)
			return
		}
		open = true
		if err := closeBlock(); err != nil {
			fail(err)
			return
		}
	}

	stop, seq := stopReason(res)
	if err := send("message_delta", messagesEvent{
		Delta: &messagesDelta{StopReason: &stop, StopSequence: seq},
		Usage: &AnthropicUsage{
			InputTokens: res.Usage.PromptTokens, OutputTokens: res.Usage.CompletionTokens,
		},
	}); err != nil {
		fail(err)
		return
	}
	_ = send("message_stop", messagesEvent{})
}

// completionFromMessages translates the Anthropic request into the one the
// backend reads.
func completionFromMessages(req *MessagesRequest) (*CompletionRequest, error) {
	out := &CompletionRequest{
		Model: req.Model, Stream: req.Stream, MaxTokens: req.MaxTokens,
		Temperature: req.Temperature, TopP: req.TopP, TopK: req.TopK,
		Stop: StringList(req.StopSequences), ToolChoice: req.ToolChoice,
	}
	if req.Thinking != nil && req.Thinking.Type == "disabled" {
		// OpenAI's spelling of the same instruction, which is what the
		// backend reads. The template enforces it in the prompt.
		out.ReasoningEffort = "none"
	}
	for _, t := range req.Tools {
		tool, err := toolFromAnthropic(t)
		if err != nil {
			return nil, err
		}
		out.Tools = append(out.Tools, tool)
	}
	if len(req.System) > 0 {
		system, err := blocksText(req.System)
		if err != nil {
			return nil, fmt.Errorf("system: %v", err)
		}
		if system != "" {
			out.Messages = append(out.Messages, Message{
				Role: "system", Content: MessageContent{{Type: "text", Text: system}},
			})
		}
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages is empty")
	}
	for i, m := range req.Messages {
		msgs, err := messageFromAnthropic(m)
		if err != nil {
			return nil, fmt.Errorf("message %d: %v", i, err)
		}
		out.Messages = append(out.Messages, msgs...)
	}
	return out, nil
}

// messageFromAnthropic turns one turn into the one or more the other shape
// needs.
//
// One turn can become several: a user turn carrying tool results is, in
// OpenAI's vocabulary and in this checkpoint's template, one `tool` message
// per result followed by whatever the user also said.
func messageFromAnthropic(m AnthropicMessage) ([]Message, error) {
	raw := json.RawMessage(strings.TrimSpace(string(m.Content)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return []Message{{Role: m.Role, Content: MessageContent{{Type: "text", Text: s}}}}, nil
	}
	var blocks []AnthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}

	var out []Message
	msg := Message{Role: m.Role}
	var text strings.Builder
	for _, b := range blocks {
		switch b.Type {
		case "text", "":
			text.WriteString(b.Text)
		case "thinking":
			msg.ReasoningContent += b.Thinking
		case "redacted_thinking":
			// Nothing to replay: the text is encrypted and not ours.
		case "tool_use":
			args := string(b.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, &ToolCall{
				ID: b.ID, Type: "function",
				Function: ToolCallReq{Name: b.Name, Arguments: args},
			})
		case "tool_result":
			result, err := blocksText(b.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, Message{
				Role: "tool", ToolCallID: b.ToolUseID,
				Content: MessageContent{{Type: "text", Text: result}},
			})
		default:
			return nil, fmt.Errorf("a %s block cannot be read by this server, which has no vision model",
				strconv.Quote(b.Type))
		}
	}
	if text.Len() > 0 || len(msg.ToolCalls) > 0 || msg.ReasoningContent != "" {
		msg.Content = MessageContent{{Type: "text", Text: text.String()}}
		out = append(out, msg)
	}
	return out, nil
}

// blocksText flattens a string or a list of blocks to its text, which is what
// `system` and a tool result both are.
func blocksText(raw json.RawMessage) (string, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		err := json.Unmarshal(raw, &s)
		return s, err
	}
	var blocks []AnthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", err
	}
	var b strings.Builder
	for _, block := range blocks {
		switch block.Type {
		case "text", "":
			b.WriteString(block.Text)
		default:
			return "", fmt.Errorf("a %s block cannot be read by this server, which has no vision model",
				strconv.Quote(block.Type))
		}
	}
	return b.String(), nil
}

// toolFromAnthropic turns `input_schema` into `parameters`, which is the only
// difference between the two tool shapes that reaches the prompt.
func toolFromAnthropic(t AnthropicTool) (Tool, error) {
	if t.Type != "" && !strings.HasPrefix(t.Type, "custom") {
		return Tool{}, fmt.Errorf("tool type %s is not implemented; this server has custom tools only",
			strconv.Quote(t.Type))
	}
	if t.Name == "" {
		return Tool{}, fmt.Errorf("a tool has no name")
	}
	out := Tool{Type: "function", Function: ToolDefinition{
		Name: t.Name, Description: t.Description, Parameters: t.InputSchema,
	}}
	raw, err := json.Marshal(struct {
		Type     string `json:"type"`
		Function struct {
			Name        string              `json:"name"`
			Description string              `json:"description,omitempty"`
			Parameters  util.JSONSchemaType `json:"parameters"`
		} `json:"function"`
	}{
		Type: "function",
		Function: struct {
			Name        string              `json:"name"`
			Description string              `json:"description,omitempty"`
			Parameters  util.JSONSchemaType `json:"parameters"`
		}{t.Name, t.Description, t.InputSchema},
	})
	if err != nil {
		return Tool{}, err
	}
	out.SetRaw(raw)
	return out, nil
}
