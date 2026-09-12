package api

// OpenAI Speech Request
// https://platform.openai.com/docs/api-reference/audio/createSpeech

// SpeechRequest is the OpenAI Speech Request
type SpeechRequest struct {
	host           string
	Input          string  `json:"input"`
	Model          string  `json:"model"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format"` // mp3, wav, opus, flac, pcm
	Speed          float64 `json:"speed"`           // 0.5 - 2.0
	Stream         bool    `json:"stream"`          // false
}

// SpeechResponse is the OpenAI Speech Response
type SpeechResponse []byte
