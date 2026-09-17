package api

import (
	"net/http"
	"sort"
)

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// Model represents a single model in the list of models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`   // "model"
	OwnedBy string `json:"owned_by"` // "organization_owner"
	// Voices is this model's voice names, for a speech model. It is not part
	// of OpenAI's model object -- clients ignore fields they do not know --
	// and it is here because the alternative is a second round trip to find
	// out what /v1/audio/speech will accept.
	Voices []string `json:"voices,omitempty"`
	// Image is the geometry an image model will accept, and is here for the
	// same reason: the sizes /v1/images/generations takes are decided by what
	// this process staged, so a client that has to guess will guess wrong on
	// a server started at something other than 1024x1024.
	Image *ImageGeometry `json:"image,omitempty"`
}

// backends returns every loaded backend, in the order GET /v1/models lists
// them. A nil field is a model this process was not started with.
func (s *Server) backends() []Backend {
	var out []Backend
	if s.Completion != nil {
		out = append(out, s.Completion)
	}
	if s.Embedding != nil {
		out = append(out, s.Embedding)
	}
	if s.Speech != nil {
		out = append(out, s.Speech)
	}
	if s.Transcription != nil {
		out = append(out, s.Transcription)
	}
	if s.Image != nil {
		out = append(out, s.Image)
	}
	return out
}

// handleModels lists what this process actually loaded, which is the only
// honest answer: a server started without -tts does not have a speech model,
// and saying so here is cheaper for a client than a 501 later.
func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	resp := ModelsResponse{Object: "list", Data: []Model{}}
	seen := map[string]bool{}
	for _, b := range s.backends() {
		for _, m := range b.Models() {
			if seen[m.ID] {
				continue
			}
			seen[m.ID] = true
			if m.Object == "" {
				m.Object = "model"
			}
			resp.Data = append(resp.Data, m)
		}
	}
	if s.Speech != nil {
		voices := s.Speech.Voices()
		sort.Strings(voices)
		for i := range resp.Data {
			if owns(s.Speech, resp.Data[i].ID) {
				resp.Data[i].Voices = voices
			}
		}
	}
	if s.Image != nil {
		geo := s.Image.Geometry()
		for i := range resp.Data {
			if owns(s.Image, resp.Data[i].ID) {
				g := geo
				resp.Data[i].Image = &g
			}
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// owns reports whether id is one of this backend's own models, so that the
// voice list and the image geometry land on the models they describe and not
// on whatever else the process loaded beside them.
func owns(b Backend, id string) bool {
	for _, m := range b.Models() {
		if m.ID == id {
			return true
		}
	}
	return false
}
