package api

import (
	"bytes"
	"encoding/json"
	"strings"

	"strix-halo-vulkan/util"
)

// OpenAI Chat Completion CompletionRequest
// https://platform.openai.com/docs/api-reference/chat/create
type CompletionRequest struct {
	ResponseFormat  *ResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	host            string
	Model           string          `json:"model"`
	Tools           []Tool          `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
	Messages        []Message       `json:"messages"`
	MinP            float64         `json:"min_p,omitempty"`
	RepeatPenalty   float64         `json:"repeat_penalty,omitempty"`
	Temperature     float64         `json:"temperature,omitempty"`
	MaxTokens       int             `json:"max_tokens,omitempty"`
	TopK            int             `json:"top_k,omitempty"`
	TopP            float64         `json:"top_p,omitempty"`
	Stream          bool            `json:"stream"`
}

// Usage represents token usage statistics from vLLM
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse represents the full response from the LLM API
type CompletionResponse struct {
	Messages []Message `json:"messages"`
}

// Chunk represents a single chunk from a streaming completion response
type Chunk struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Created int64    `json:"created"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	FinishReason string `json:"finish_reason"`
	Delta        Delta  `json:"delta"`
	Message      Delta  `json:"message"`
	Index        int    `json:"index"`
}

// Delta represents the incremental content in a streaming response
type Delta struct {
	ReasoningContent string      `json:"reasoning_content"`
	Content          string      `json:"content"`
	ToolCalls        []*ToolCall `json:"tool_calls"`
	IsThinking       bool        `json:"is_thinking"`
}

type ResponseFormat struct {
	JSONSchema *util.JSONSchema `json:"json_schema,omitempty"`
	Type       string           `json:"type"`
}

type Tool struct {
	Type     string         `json:"type"`
	Function ToolDefinition `json:"function"`
}

type ToolDefinition struct {
	Parameters  util.JSONSchemaType `json:"parameters"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Strict      bool                `json:"strict"`
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
