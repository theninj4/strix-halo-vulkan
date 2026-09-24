package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/util"
)

// POST /v1/responses -- OpenAI's Responses API over the same backend
// (LLM.md L9a).
//
// This endpoint adds no capability. It is the same generation as
// /v1/chat/completions in a different envelope, and it exists because a
// client written against the newer API should not have to be rewritten to
// talk to this server. So the whole file is translation: an `input` -- a
// string, or a list of items that are messages, calls and call outputs -- to
// the same []Message the chat endpoint builds, and a generation back out as
// the Responses object's `output` array.
//
// The one place the two envelopes genuinely differ is reasoning. A chat
// completion carries it in a field beside the content; a response carries it
// as an *item of its own*, before the message it led to. That is the better
// shape and it is the one this file writes.

type ResponsesRequest struct {
	Reasoning       *ResponsesReasoning `json:"reasoning,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	TopP            *float64            `json:"top_p,omitempty"`
	Model           string              `json:"model"`
	Instructions    string              `json:"instructions,omitempty"`
	Input           json.RawMessage     `json:"input"`
	ToolChoice      json.RawMessage     `json:"tool_choice,omitempty"`
	Text            json.RawMessage     `json:"text,omitempty"`
	ServiceTier     string              `json:"service_tier,omitempty"`
	Tools           []ResponsesTool     `json:"tools,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Stream          bool                `json:"stream"`
}

// InputItem is one entry of an `input` array. It is every item shape at once,
// because which one it is is decided by the fields that are present: a
// message has a role, a call has a name and arguments, a call output has an
// output.
type InputItem struct {
	Type      string             `json:"type,omitempty"`
	Role      string             `json:"role,omitempty"`
	Content   json.RawMessage    `json:"content,omitempty"`
	Name      string             `json:"name,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Output    json.RawMessage    `json:"output,omitempty"`
	Summary   []InputContentPart `json:"summary,omitempty"`
}

type InputContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	// input_image support
	ImageURL string `json:"image_url,omitempty"`
}

type ResponsesTool struct {
	Parameters  util.JSONSchemaType `json:"parameters"`
	Type        string              `json:"type"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Strict      bool                `json:"strict,omitempty"`
}

type ResponsesReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// --- Response types ---

type ResponsesResponse struct {
	Usage           *ResponsesUsage       `json:"usage,omitempty"`
	Temperature     *float64              `json:"temperature,omitempty"`
	TopP            *float64              `json:"top_p,omitempty"`
	ID              string                `json:"id"`
	Object          string                `json:"object"`
	Status          string                `json:"status"`
	Model           string                `json:"model"`
	Output          []ResponsesOutputItem `json:"output"`
	Tools           []ResponsesTool       `json:"tools,omitempty"`
	CreatedAt       int64                 `json:"created_at"`
	MaxOutputTokens int                   `json:"max_output_tokens,omitempty"`
}

type ResponsesOutputItem struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id"`
	Status    string                 `json:"status,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Content   []ResponsesContentPart `json:"content,omitempty"`
	Summary   []ResponsesContentPart `json:"summary,omitempty"`
	Name      string                 `json:"name,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
}

type ResponsesContentPart struct {
	Type        string   `json:"type"`
	Text        string   `json:"text"`
	Annotations []string `json:"annotations,omitempty"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// responsesEvent is one frame of a streamed response. The Responses API names
// and numbers every event and the payload differs per name, so this carries
// the union of the fields the events below use rather than one struct each.
type responsesEvent struct {
	Response       *ResponsesResponse    `json:"response,omitempty"`
	Item           *ResponsesOutputItem  `json:"item,omitempty"`
	Part           *ResponsesContentPart `json:"part,omitempty"`
	Type           string                `json:"type"`
	ItemID         string                `json:"item_id,omitempty"`
	Delta          string                `json:"delta,omitempty"`
	Text           string                `json:"text,omitempty"`
	Arguments      string                `json:"arguments,omitempty"`
	SequenceNumber int                   `json:"sequence_number"`
	OutputIndex    int                   `json:"output_index"`
	ContentIndex   int                   `json:"content_index,omitempty"`
	SummaryIndex   int                   `json:"summary_index,omitempty"`
}

// handleResponses is the Responses endpoint.
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Completion == nil {
		notLoaded(ctx, w, "the language model", "-llm")
		return
	}
	var req ResponsesRequest
	if !decodeJSON(ctx, w, r, &req) {
		return
	}
	comp, err := completionFromResponses(&req)
	if err != nil {
		badRequest(ctx, w, err.Error())
		return
	}
	model := s.completionModel(comp)
	if !validCompletion(ctx, w, comp) {
		return
	}

	id := "resp_" + randomID()
	created := time.Now().Unix()
	base := func() *ResponsesResponse {
		return &ResponsesResponse{
			ID: id, Object: "response", CreatedAt: created, Model: model,
			Status: "in_progress", Output: []ResponsesOutputItem{},
			Temperature: comp.Temperature, TopP: comp.TopP,
			MaxOutputTokens: req.MaxOutputTokens, Tools: req.Tools,
		}
	}
	if req.Stream {
		s.streamResponses(w, r, comp, base, id)
		return
	}

	var content, reasoning strings.Builder
	res, err := s.Completion.Complete(r.Context(), comp, func(d Delta) error {
		content.WriteString(d.Content)
		reasoning.WriteString(d.ReasoningContent)
		return nil
	})
	if err != nil {
		backendError(ctx, w, "responses", err)
		return
	}
	out := base()
	out.Status = "completed"
	out.Output = responseOutput(id, reasoning.String(), content.String(), res.ToolCalls)
	out.Usage = &ResponsesUsage{
		InputTokens:  res.Usage.PromptTokens,
		OutputTokens: res.Usage.CompletionTokens,
		TotalTokens:  res.Usage.TotalTokens,
	}
	writeJSON(w, http.StatusOK, out)
}

// responseOutput is the `output` array: the reasoning, then the message, then
// one item per call. Each is left out when it is empty, which is how a
// response that is only a tool call has no message in it.
func responseOutput(id, reasoning, content string, calls []*ToolCall) []ResponsesOutputItem {
	out := []ResponsesOutputItem{}
	if reasoning != "" {
		out = append(out, ResponsesOutputItem{
			Type: "reasoning", ID: "rs_" + id, Status: "completed",
			Summary: []ResponsesContentPart{{Type: "summary_text", Text: reasoning}},
		})
	}
	if content != "" {
		out = append(out, ResponsesOutputItem{
			Type: "message", ID: "msg_" + id, Status: "completed", Role: "assistant",
			Content: []ResponsesContentPart{{Type: "output_text", Text: content, Annotations: []string{}}},
		})
	}
	for _, c := range calls {
		out = append(out, ResponsesOutputItem{
			Type: "function_call", ID: "fc_" + c.ID, Status: "completed",
			CallID: c.ID, Name: c.Function.Name, Arguments: c.Function.Arguments,
		})
	}
	return out
}

// streamResponses writes the generation as the Responses API's named events.
//
// The taxonomy is deeper than the chat endpoint's -- an item is opened, its
// content part is opened, the text is delivered as deltas, and all three are
// closed again -- and this emits the subset that carries a text answer, a
// reasoning summary and function calls. What it does not emit, it does not
// emit at all rather than approximately: there is no `response.incomplete`
// here because nothing in this server produces one.
func (s *Server) streamResponses(w http.ResponseWriter, r *http.Request,
	comp *CompletionRequest, base func() *ResponsesResponse, id string,
) {
	ctx := r.Context()
	stream := newSSE(w)
	fail := func(err error) {
		if r.Context().Err() != nil {
			logf(ctx, "responses: client cancelled")
			return
		}
		logf(ctx, "responses: %v", err)
		_ = stream.send("error", errorBody{Message: err.Error(), Type: "server_error"})
	}
	send := func(typ string, e responsesEvent) error {
		e.Type, e.SequenceNumber = typ, stream.next()
		return stream.send(typ, e)
	}

	created := base()
	if err := send("response.created", responsesEvent{Response: created}); err != nil {
		fail(err)
		return
	}
	if err := send("response.in_progress", responsesEvent{Response: created}); err != nil {
		fail(err)
		return
	}

	// The two items this generation can open, opened lazily: a response with
	// no reasoning should not carry an empty reasoning item, and one that is
	// only a tool call should not carry an empty message.
	var (
		index                  int
		inReasoning, inMessage bool
		reasoning, content     strings.Builder
		reasoningID            = "rs_" + id
		messageID              = "msg_" + id
	)
	closeReasoning := func() error {
		if !inReasoning {
			return nil
		}
		inReasoning = false
		if err := send("response.reasoning_summary_text.done", responsesEvent{
			ItemID: reasoningID, OutputIndex: index, Text: reasoning.String(),
		}); err != nil {
			return err
		}
		item := ResponsesOutputItem{
			Type: "reasoning", ID: reasoningID, Status: "completed",
			Summary: []ResponsesContentPart{{Type: "summary_text", Text: reasoning.String()}},
		}
		if err := send("response.output_item.done", responsesEvent{
			OutputIndex: index, Item: &item,
		}); err != nil {
			return err
		}
		index++
		return nil
	}

	res, err := s.Completion.Complete(r.Context(), comp, func(d Delta) error {
		if d.ReasoningContent != "" {
			if !inReasoning {
				inReasoning = true
				item := ResponsesOutputItem{Type: "reasoning", ID: reasoningID, Status: "in_progress"}
				if err := send("response.output_item.added", responsesEvent{
					OutputIndex: index, Item: &item,
				}); err != nil {
					return err
				}
				if err := send("response.reasoning_summary_part.added", responsesEvent{
					ItemID: reasoningID, OutputIndex: index,
					Part: &ResponsesContentPart{Type: "summary_text"},
				}); err != nil {
					return err
				}
			}
			reasoning.WriteString(d.ReasoningContent)
			if err := send("response.reasoning_summary_text.delta", responsesEvent{
				ItemID: reasoningID, OutputIndex: index, Delta: d.ReasoningContent,
			}); err != nil {
				return err
			}
		}
		if d.Content == "" {
			return nil
		}
		if err := closeReasoning(); err != nil {
			return err
		}
		if !inMessage {
			inMessage = true
			item := ResponsesOutputItem{
				Type: "message", ID: messageID, Status: "in_progress", Role: "assistant",
				Content: []ResponsesContentPart{},
			}
			if err := send("response.output_item.added", responsesEvent{
				OutputIndex: index, Item: &item,
			}); err != nil {
				return err
			}
			if err := send("response.content_part.added", responsesEvent{
				ItemID: messageID, OutputIndex: index,
				Part: &ResponsesContentPart{Type: "output_text", Annotations: []string{}},
			}); err != nil {
				return err
			}
		}
		content.WriteString(d.Content)
		return send("response.output_text.delta", responsesEvent{
			ItemID: messageID, OutputIndex: index, Delta: d.Content,
		})
	})
	if err != nil {
		fail(err)
		return
	}
	if err := closeReasoning(); err != nil {
		fail(err)
		return
	}
	if inMessage {
		part := ResponsesContentPart{Type: "output_text", Text: content.String(), Annotations: []string{}}
		if err := send("response.output_text.done", responsesEvent{
			ItemID: messageID, OutputIndex: index, Text: content.String(),
		}); err != nil {
			fail(err)
			return
		}
		if err := send("response.content_part.done", responsesEvent{
			ItemID: messageID, OutputIndex: index, Part: &part,
		}); err != nil {
			fail(err)
			return
		}
		item := ResponsesOutputItem{
			Type: "message", ID: messageID, Status: "completed", Role: "assistant",
			Content: []ResponsesContentPart{part},
		}
		if err := send("response.output_item.done", responsesEvent{
			OutputIndex: index, Item: &item,
		}); err != nil {
			fail(err)
			return
		}
		index++
	}
	// A call's arguments are known whole -- they are not JSON until the call
	// has closed -- so the delta and the done carry the same string. The
	// delta is sent anyway, because a client that only listens for deltas
	// should still see the arguments.
	for _, c := range res.ToolCalls {
		item := ResponsesOutputItem{
			Type: "function_call", ID: "fc_" + c.ID, Status: "in_progress",
			CallID: c.ID, Name: c.Function.Name,
		}
		if err := send("response.output_item.added", responsesEvent{
			OutputIndex: index, Item: &item,
		}); err != nil {
			fail(err)
			return
		}
		if err := send("response.function_call_arguments.delta", responsesEvent{
			ItemID: item.ID, OutputIndex: index, Delta: c.Function.Arguments,
		}); err != nil {
			fail(err)
			return
		}
		if err := send("response.function_call_arguments.done", responsesEvent{
			ItemID: item.ID, OutputIndex: index, Arguments: c.Function.Arguments,
		}); err != nil {
			fail(err)
			return
		}
		item.Status = "completed"
		item.Arguments = c.Function.Arguments
		if err := send("response.output_item.done", responsesEvent{
			OutputIndex: index, Item: &item,
		}); err != nil {
			fail(err)
			return
		}
		index++
	}

	final := base()
	final.Status = "completed"
	final.Output = responseOutput(id, reasoning.String(), content.String(), res.ToolCalls)
	final.Usage = &ResponsesUsage{
		InputTokens:  res.Usage.PromptTokens,
		OutputTokens: res.Usage.CompletionTokens,
		TotalTokens:  res.Usage.TotalTokens,
	}
	if err := send("response.completed", responsesEvent{Response: final}); err != nil {
		fail(err)
	}
}

// completionFromResponses translates the Responses request into the one the
// backend reads. Everything it refuses is a shape this server would otherwise
// have to guess at.
func completionFromResponses(req *ResponsesRequest) (*CompletionRequest, error) {
	out := &CompletionRequest{
		Model: req.Model, Stream: req.Stream,
		Temperature: req.Temperature, TopP: req.TopP,
		MaxCompletionTokens: req.MaxOutputTokens,
		ToolChoice:          req.ToolChoice,
		ServiceTier:         req.ServiceTier,
	}
	if req.Reasoning != nil {
		out.ReasoningEffort = req.Reasoning.Effort
	}
	if len(req.Text) > 0 {
		var text struct {
			Format struct {
				Type string `json:"type"`
			} `json:"format"`
		}
		if err := json.Unmarshal(req.Text, &text); err == nil &&
			text.Format.Type != "" && text.Format.Type != "text" {
			return nil, fmt.Errorf("text.format %s is not implemented; "+
				"there is no constrained decoding in this server yet", strconv.Quote(text.Format.Type))
		}
	}
	for _, t := range req.Tools {
		tool, err := toolFromResponses(t)
		if err != nil {
			return nil, err
		}
		out.Tools = append(out.Tools, tool)
	}
	// `instructions` is the Responses API's system message, and it is
	// replaced rather than appended by each request -- which is exactly a
	// leading system turn.
	if req.Instructions != "" {
		out.Messages = append(out.Messages, Message{
			Role: "system", Content: MessageContent{{Type: "text", Text: req.Instructions}},
		})
	}
	msgs, err := messagesFromInput(req.Input)
	if err != nil {
		return nil, err
	}
	out.Messages = append(out.Messages, msgs...)
	return out, nil
}

// toolFromResponses turns the Responses API's flat tool into the nested one
// the chat template writes, and hands over the bytes it built so that the
// prompt carries the conventional field order rather than a Go struct's.
func toolFromResponses(t ResponsesTool) (Tool, error) {
	if t.Type != "" && t.Type != "function" {
		return Tool{}, fmt.Errorf("tool type %s is not implemented; this server has function tools only",
			strconv.Quote(t.Type))
	}
	if t.Name == "" {
		return Tool{}, fmt.Errorf("a tool has no name")
	}
	out := Tool{Type: "function", Function: ToolDefinition{
		Name: t.Name, Description: t.Description, Parameters: t.Parameters, Strict: t.Strict,
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
		}{t.Name, t.Description, t.Parameters},
	})
	if err != nil {
		return Tool{}, err
	}
	out.SetRaw(raw)
	return out, nil
}

// messagesFromInput reads the Responses API's `input`: a bare string, which
// is one user turn, or a list of items.
//
// The item types that carry a conversation are all handled -- a message, a
// function call, a call's output, and the reasoning item that precedes an
// assistant turn -- because an agent loop replays exactly those, and a server
// that dropped the calls would answer as though the tools had never run.
func messagesFromInput(raw json.RawMessage) ([]Message, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("input is empty")
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return []Message{{Role: "user", Content: MessageContent{{Type: "text", Text: s}}}}, nil
	}
	var items []InputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input is neither a string nor a list of items: %v", err)
	}

	var out []Message
	// A reasoning item comes *before* the assistant message it belongs to, so
	// it is held until that message arrives.
	var pending string
	for i, it := range items {
		switch {
		case it.Type == "reasoning":
			var b strings.Builder
			for _, p := range it.Summary {
				b.WriteString(p.Text)
			}
			pending = b.String()
		case it.Type == "function_call":
			out = append(out, Message{Role: "assistant", ReasoningContent: pending, ToolCalls: []*ToolCall{{
				ID: it.CallID, Type: "function",
				Function: ToolCallReq{Name: it.Name, Arguments: it.Arguments},
			}}})
			pending = ""
		case it.Type == "function_call_output":
			text, err := inputText(it.Output)
			if err != nil {
				return nil, fmt.Errorf("input item %d: %v", i, err)
			}
			out = append(out, Message{
				Role: "tool", ToolCallID: it.CallID,
				Content: MessageContent{{Type: "text", Text: text}},
			})
		case it.Role != "":
			content, err := inputContent(it.Content)
			if err != nil {
				return nil, fmt.Errorf("input item %d: %v", i, err)
			}
			m := Message{Role: it.Role, Content: content}
			if it.Role == "assistant" {
				m.ReasoningContent = pending
				pending = ""
			}
			out = append(out, m)
		default:
			return nil, fmt.Errorf("input item %d has type %s, which this server does not read",
				i, strconv.Quote(it.Type))
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("input holds no messages")
	}
	return out, nil
}

// inputContent is a message item's content as text and image parts in
// order: `input_image` becomes the image_url block the backend reads, from its
// `image_url` (a data: URL; the backend fetches nothing). A `file_id` image is
// refused, because this server keeps no files.
func inputContent(raw json.RawMessage) (MessageContent, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || raw[0] != '[' {
		text, err := inputText(raw)
		if err != nil {
			return nil, err
		}
		return MessageContent{{Type: "text", Text: text}}, nil
	}
	var parts []InputContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, err
	}
	var out MessageContent
	for _, p := range parts {
		switch p.Type {
		case "", "input_text", "output_text", "text", "summary_text", "refusal":
			if n := len(out); n > 0 && out[n-1].Type == "text" {
				out[n-1].Text += p.Text
			} else {
				out = append(out, Content{Type: "text", Text: p.Text})
			}
		case "input_image":
			if p.ImageURL == "" {
				return nil, fmt.Errorf("an input_image with no image_url; this server keeps no files, " +
					"so send the image as a data: URL")
			}
			out = append(out, Content{Type: "image_url", ImageURL: &struct {
				URL string `json:"url"`
			}{URL: p.ImageURL}})
		default:
			return nil, fmt.Errorf("a %s content part cannot be read by this server, which reads text and images",
				strconv.Quote(p.Type))
		}
	}
	if len(out) == 0 {
		out = MessageContent{{Type: "text"}}
	}
	return out, nil
}

// inputText flattens an item's content, which the Responses API allows as a
// string or as a list of parts. A part that is neither text nor an image is
// refused: answering about the text alone would look like an answer.
func inputText(raw json.RawMessage) (string, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		err := json.Unmarshal(raw, &s)
		return s, err
	}
	var parts []InputContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", err
	}
	var b strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "", "input_text", "output_text", "text", "summary_text", "refusal":
			b.WriteString(p.Text)
		default:
			return "", fmt.Errorf("a %s content part cannot be read here: this server takes images in message items only",
				strconv.Quote(p.Type))
		}
	}
	return b.String(), nil
}
