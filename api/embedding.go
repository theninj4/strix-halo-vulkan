package api

// POST /v1/embeddings (EMBEDDING.md E8).

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"net/http"
)

// EmbeddingRequest is OpenAI's embedding request.
// https://platform.openai.com/docs/api-reference/embeddings/create
type EmbeddingRequest struct {
	// Input is one text or several. OpenAI also allows pre-tokenized input
	// (arrays of ids); this does not, because the ids would be another
	// tokenizer's and there is no way to tell.
	Input StringList `json:"input"`
	Model string     `json:"model"`
	// EncodingFormat is "float" (the default) or "base64" -- little-endian
	// float32, which is what OpenAI's clients decode.
	EncodingFormat string `json:"encoding_format"`
	// Dimensions truncates the vector and renormalises it (MRL). Zero is the
	// model's full width.
	Dimensions int `json:"dimensions"`

	// Instruct is **not** OpenAI's. It is the one thing this model needs
	// that the envelope has no room for: Qwen3-Embedding is instruction
	// aware, and a *query* is supposed to be prefixed with a one-sentence
	// description of the retrieval task while a *document* is not
	// (EMBEDDING.md E0). Set it on the query side of a search and leave it
	// off when embedding the corpus; "default" asks for the model card's own
	// instruction. Leaving it off entirely is what a client that does not
	// know about it does, and that is the documents' behaviour, which is the
	// safe half.
	Instruct string `json:"instruct,omitempty"`
}

// EmbeddingResponse is OpenAI's embedding response.
type EmbeddingResponse struct {
	Object string          `json:"object"` // "list"
	Data   []EmbeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  *EmbeddingUsage `json:"usage,omitempty"`
}

// EmbeddingItem is one vector. Embedding is a []float32 or, when the request
// asked for base64, a string.
type EmbeddingItem struct {
	Object    string `json:"object"` // "embedding"
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

// EmbeddingUsage is the token accounting. There is no completion half: an
// embedding reads and does not write.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// handleEmbeddings is POST /v1/embeddings.
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if s.Embedding == nil {
		notLoaded(w, "the embedding model", "-embed")
		return
	}
	var req EmbeddingRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Input) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "input is required")
		return
	}
	for _, text := range req.Input {
		if text == "" {
			writeError(w, http.StatusBadRequest, "invalid_request_error",
				"input must not contain an empty string")
			return
		}
	}
	switch req.EncodingFormat {
	case "", "float", "base64":
	default:
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"encoding_format must be \"float\" or \"base64\"")
		return
	}
	if req.Dimensions < 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"dimensions must be positive")
		return
	}

	res, err := s.Embedding.Embed(r.Context(), &req)
	if err != nil {
		backendError(w, "embedding", err)
		return
	}
	if len(res.Vectors) != len(req.Input) {
		// A backend that returned a different number of vectors than there
		// were inputs would silently misalign a client's corpus, so it is an
		// error here rather than a response.
		backendError(w, "embedding", errVectorCount)
		return
	}

	resp := EmbeddingResponse{
		Object: "list",
		Model:  modelID(s.Embedding, req.Model),
		Data:   make([]EmbeddingItem, len(res.Vectors)),
	}
	for i, v := range res.Vectors {
		item := EmbeddingItem{Object: "embedding", Index: i}
		if req.EncodingFormat == "base64" {
			item.Embedding = base64Vector(v)
		} else {
			item.Embedding = v
		}
		resp.Data[i] = item
	}
	if res.Usage.PromptTokens > 0 {
		resp.Usage = &EmbeddingUsage{
			PromptTokens: res.Usage.PromptTokens,
			TotalTokens:  res.Usage.TotalTokens,
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// base64Vector is OpenAI's compact encoding: the float32s little-endian, then
// standard base64. It is half the bytes of the JSON array and exactly what
// the official clients decode when they asked for it.
func base64Vector(v []float32) string {
	buf := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}
