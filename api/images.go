package api

type ImageGenerationRequest struct {
	host        string
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Model       string `json:"model,omitempty"`
	Prompt      string `json:"prompt"`
	Stream      bool   `json:"stream,omitempty"`
}

type ImageEditRequest struct {
	host        string
	AspectRatio string   `json:"aspect_ratio,omitempty"`
	Model       string   `json:"model,omitempty"`
	Image       []string `json:"image"`
	Prompt      string   `json:"prompt"`
	Stream      bool     `json:"stream,omitempty"`
}
