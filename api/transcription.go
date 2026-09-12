package api

// TranscriptionRequest represents the request structure for audio transcription.
type TranscriptionRequest struct {
	host             string
	Model            string  `json:"model"`
	ChunkingStrategy string  `json:"chunking_strategy"` // "auto"
	Language         string  `json:"language"`          // "en"
	Prompt           string  `json:"prompt"`
	ResponseFormat   string  `json:"response_format"` // "json"
	Data             []byte  `json:"file"`
	Temperature      float64 `json:"temperature"`
	Stream           bool    `json:"stream"`
}

// TranscriptionResponse represents the non-streaming response.
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// TranscriptionStreamResponse represents the streaming response.
type TranscriptionStreamResponse struct {
	Type  string `json:"type"`            // "transcript.text.delta" or "transcript.text.done"
	Delta string `json:"delta,omitempty"` // Only present when Type is "transcript.text.delta"
	Text  string `json:"text,omitempty"`  // Only present when Type is "transcript.text.done"
}
