package api

import (
	"encoding/json"
	"strix-halo-vulkan/util"
)

type ResponsesRequest struct {
	Model           string              `json:"model"`
	Input           json.RawMessage     `json:"input"`
	Instructions    string              `json:"instructions,omitempty"`
	Temperature     float64             `json:"temperature,omitempty"`
	TopP            float64             `json:"top_p,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Tools           []ResponsesTool     `json:"tools,omitempty"`
	ToolChoice      json.RawMessage     `json:"tool_choice,omitempty"`
	Stream          bool                `json:"stream"`
	Reasoning       *ResponsesReasoning `json:"reasoning,omitempty"`
	Text            json.RawMessage     `json:"text,omitempty"`
}

type InputItem struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type InputContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	// input_image support
	ImageURL string `json:"image_url,omitempty"`
}

type ResponsesTool struct {
	Type        string              `json:"type"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Parameters  util.JSONSchemaType `json:"parameters"`
	Strict      bool                `json:"strict,omitempty"`
}

type ResponsesReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// --- Response types ---

type ResponsesResponse struct {
	ID              string                `json:"id"`
	Object          string                `json:"object"`
	CreatedAt       int64                 `json:"created_at"`
	Status          string                `json:"status"`
	Model           string                `json:"model"`
	Output          []ResponsesOutputItem `json:"output"`
	Usage           *ResponsesUsage       `json:"usage,omitempty"`
	Temperature     float64               `json:"temperature,omitempty"`
	TopP            float64               `json:"top_p,omitempty"`
	MaxOutputTokens int                   `json:"max_output_tokens,omitempty"`
	Tools           []ResponsesTool       `json:"tools,omitempty"`
}

type ResponsesOutputItem struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id"`
	Status    string                 `json:"status,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Content   []ResponsesContentPart `json:"content,omitempty"`
	Name      string                 `json:"name,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
}

type ResponsesContentPart struct {
	Type        string   `json:"type"`
	Text        string   `json:"text"`
	Annotations []string `json:"annotations"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
