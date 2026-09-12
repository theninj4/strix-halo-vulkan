package api

// EmbeddingRequest is an OpenAI Embedding CompletionRequest
// https://platform.openai.com/docs/api-reference/embeddings/create
type EmbeddingRequest struct {
	host           string
	Input          string `json:"input"`
	Model          string `json:"model"`
	EncodingFormat string `json:"encoding_format"`
	Dimensions     int    `json:"dimensions"`
}

// EmbeddingResponse is an OpenAI Embedding CompletionResponse
type EmbeddingResponse struct {
	Object string          `json:"object"` // "list"
	Data   []EmbeddingItem `json:"data"`
}

// EmbeddingItem is an OpenAI Embedding CompletionResponse item
type EmbeddingItem struct {
	Object    string    `json:"object"` // "embedding"
	Embedding []float32 `json:"embedding"`
}
