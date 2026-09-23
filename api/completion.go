package api

import (
	"bytes"
	"encoding/json"
	"strings"

	"strix-halo-vulkan/util"
)

// OpenAI Chat Completion CompletionRequest
// https://platform.openai.com/docs/api-reference/chat/create
//
// The sampling fields are pointers where the zero value is a legitimate
// setting: `temperature: 0` is greedy decoding and not "unset", and a server
// that cannot tell them apart either refuses to sample or refuses to be
// deterministic. Unset takes the checkpoint's own recommendation, which this
// one carries in its metadata (`general.sampling.*`).
type CompletionRequest struct {
	ResponseFormat  *ResponseFormat `json:"response_format,omitempty"`
	StreamOptions   *StreamOptions  `json:"stream_options,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	host            string
	Model           string          `json:"model"`
	Tools           []Tool          `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
	Messages        []Message       `json:"messages"`
	Stop            StringList      `json:"stop,omitempty"`
	MinP            *float64        `json:"min_p,omitempty"`
	RepeatPenalty   *float64        `json:"repeat_penalty,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	TopK            *int            `json:"top_k,omitempty"`
	Seed            *int64          `json:"seed,omitempty"`
	MaxTokens       int             `json:"max_tokens,omitempty"`
	// MaxCompletionTokens is OpenAI's newer spelling of MaxTokens and wins
	// when both are sent, which is what their own clients do during the
	// migration.
	MaxCompletionTokens int  `json:"max_completion_tokens,omitempty"`
	N                   int  `json:"n,omitempty"`
	Stream              bool `json:"stream"`
	// ServiceTier is OpenAI's field, read as this server's priority class
	// (CONCURRENCY.md): "priority" is interactive — a voice command, served
	// ahead of everything else — and "flex" is background. Anything else
	// takes the server's default. The X-Priority header overrides it.
	ServiceTier string `json:"service_tier,omitempty"`
}

// Budget is how many tokens the completion may run to, under either of
// OpenAI's two spellings, or zero for the server's own default.
func (r *CompletionRequest) Budget() int {
	if r.MaxCompletionTokens > 0 {
		return r.MaxCompletionTokens
	}
	return r.MaxTokens
}

// StreamOptions is OpenAI's `stream_options`. The only member that means
// anything here is the usage one.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// StringList is a field OpenAI allows as either a single string or an array
// of them -- `stop` is the one this API carries. Both decode to a slice.
type StringList []string

func (l *StringList) UnmarshalJSON(b []byte) error {
	trimmed := bytes.TrimSpace(b)
	switch {
	case len(trimmed) == 0 || string(trimmed) == "null":
		*l = nil
		return nil
	case trimmed[0] == '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		*l = StringList{s}
		return nil
	default:
		var out []string
		if err := json.Unmarshal(trimmed, &out); err != nil {
			return err
		}
		*l = out
		return nil
	}
}

// Usage is the token accounting a completion reports. The prompt count is the
// whole rendered conversation, template included, because that is what the
// model read and what the time was spent on.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse is one non-streamed completion: OpenAI's
// `chat.completion` object.
type CompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"` // "chat.completion"
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
	Created int64    `json:"created"`
}

// Chunk is one Server-Sent Event of a streamed completion: OpenAI's
// `chat.completion.chunk`.
type Chunk struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"` // "chat.completion.chunk"
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
	Created int64    `json:"created"`
}

// Choice is one of a response's completions -- this server only ever returns
// the first, `n > 1` being refused rather than served serially.
//
// `Message` carries a whole completion and `Delta` carries a piece of one, so
// exactly one of them is set: a chunk with both would be a response claiming
// to be streamed and buffered at once. FinishReason is a pointer because the
// wire value in every chunk before the last is `null`, and a client
// distinguishing "still going" from "stopped" reads exactly that.
type Choice struct {
	Message      *Delta  `json:"message,omitempty"`
	Delta        *Delta  `json:"delta,omitempty"`
	FinishReason *string `json:"finish_reason"`
	Index        int     `json:"index"`
}

// Delta is a message, or the piece of one a chunk carries.
//
// ReasoningContent is not OpenAI's field but it is the one every client that
// talks to a reasoning model already reads. It is separate from Content
// because this checkpoint's template makes it separate: the generation prompt
// opens a `<think>` block, and what the model writes inside it is working and
// not an answer.
type Delta struct {
	Role             string      `json:"role,omitempty"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	Content          string      `json:"content,omitempty"`
	ToolCalls        []*ToolCall `json:"tool_calls,omitempty"`
}

type ResponseFormat struct {
	JSONSchema *util.JSONSchema `json:"json_schema,omitempty"`
	Type       string           `json:"type"`
}

// Tool is one entry of the request's `tools` array.
//
// It keeps **the bytes it arrived as**. The chat template writes a tool with
// `tool | tojson` -- the client's own object, field order and all -- and a
// round trip through this struct would not reproduce it: Go sorts a
// `map[string]any`'s keys, writes the struct's fields in declaration order
// rather than the client's, and adds every field the client left out. None of
// that is wrong JSON and all of it is a different prompt.
type Tool struct {
	raw      json.RawMessage
	Type     string         `json:"type"`
	Function ToolDefinition `json:"function"`
}

// UnmarshalJSON decodes the typed view and keeps the original beside it.
func (t *Tool) UnmarshalJSON(b []byte) error {
	type plain Tool // no UnmarshalJSON of its own, so this does not recurse
	var v plain
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*t = Tool(v)
	t.raw = append(json.RawMessage(nil), b...)
	return nil
}

// Raw is the tool as the client wrote it, or -- for a tool this server built
// itself, translating another envelope's shape -- the typed view encoded.
func (t Tool) Raw() json.RawMessage {
	if len(t.raw) > 0 {
		return t.raw
	}
	b, err := json.Marshal(struct {
		Type     string         `json:"type"`
		Function ToolDefinition `json:"function"`
	}{t.Type, t.Function})
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return b
}

// SetRaw is how a handler translating another envelope hands over the object
// it built, so that the template writes that rather than this struct's field
// order.
func (t *Tool) SetRaw(b json.RawMessage) { t.raw = append(json.RawMessage(nil), b...) }

type ToolDefinition struct {
	Parameters  util.JSONSchemaType `json:"parameters"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Strict      bool                `json:"strict,omitempty"`
}

// ToolCall represents a tool call in the response
type ToolCall struct {
	Function ToolCallReq `json:"function"`
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Index    int         `json:"index"`
}

// ToolCallReq represents a tool call request
type ToolCallReq struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Content represents different types of content in a message
type Content struct {
	Type string `json:"type"`
	// For text content
	Text string `json:"text,omitempty"`
	// For image content
	ImageURL *struct {
		URL string `json:"url"` // base64
	} `json:"image_url,omitempty"`
	// For error content
	Error string `json:"error,omitempty"`
}

// MessageContent holds a message's content, which the OpenAI API allows as either
// a plain string or an array of content blocks. It always marshals as an array.
type MessageContent []Content

// Message represents a message from or to the LLM
type Message struct {
	Role       string         `json:"role"`                   // can be 'user', 'assistant', 'thinking', 'error' or 'tool'
	Content    MessageContent `json:"content,omitempty"`      // List of complex content (or a plain string when received)
	ToolCallID string         `json:"tool_call_id,omitempty"` // ID of the tool call
	ToolCalls  []*ToolCall    `json:"tool_calls,omitempty"`   // List of tool calls
	// ReasoningContent is an assistant turn's `<think>` block, replayed. This
	// checkpoint's template keeps the reasoning of previous turns in the
	// history, so a client that drops it is sending a different conversation
	// from the one it was shown.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// UnmarshalJSON accepts both shapes OpenAI's schema allows for a message's
// content: a bare string, which is what most clients send, and an array of
// content blocks, which is what a multimodal message is. Both land as an
// array here, so nothing downstream has to know which arrived.
//
// A null content -- which an assistant message carrying only tool calls has
// -- decodes to no blocks rather than an error.
func (c *MessageContent) UnmarshalJSON(b []byte) error {
	trimmed := bytes.TrimSpace(b)
	switch {
	case len(trimmed) == 0 || string(trimmed) == "null":
		*c = nil
		return nil
	case trimmed[0] == '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		*c = MessageContent{{Type: "text", Text: s}}
		return nil
	default:
		var parts []Content
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return err
		}
		*c = parts
		return nil
	}
}

// Text is the message's text blocks joined, which is what a text-only model
// is given. Blocks of any other type are skipped rather than rendered, so an
// image in a conversation does not arrive as a caption the model would read
// as words.
func (c MessageContent) Text() string {
	var b strings.Builder
	for _, part := range c {
		if part.Type == "" || part.Type == "text" {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}

// NonText is the first content block that is not text, or the empty string.
// It is what the chat handler refuses on: there is no vision tower in this
// repository, so an image block would be dropped silently and answered about
// as though it had never been sent.
func (c MessageContent) NonText() string {
	for _, part := range c {
		if part.Type != "" && part.Type != "text" {
			return part.Type
		}
	}
	return ""
}
