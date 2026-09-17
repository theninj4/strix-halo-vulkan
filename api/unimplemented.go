package api

import "net/http"

// The endpoints whose models are not wired yet.
//
// They are registered rather than left off the mux so that a client gets a
// 501 with a message saying what is missing, instead of a 404 that looks like
// a misspelled path. Each one becomes a handler when its backend arrives:
// chat completions, responses and messages when `llm`'s generation loop does
// (LLM.md L7), embeddings when there is an embedding model at all
// (GOALS.md), and images when `zimage`'s pipeline is wired in. The nil check
// each one will then carry -- "loaded, or not asked for on the command line"
// -- is the one handleSpeech and handleTranscription already have.

func (s *Server) handleChatCompletions(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/chat/completions", "the language model")
}

func (s *Server) handleResponses(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/responses", "the language model")
}

func (s *Server) handleMessages(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/messages", "the language model")
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/embeddings", "an embedding model")
}

func (s *Server) handleImageGeneration(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/images/generations", "the image model")
}

func (s *Server) handleImageEdit(w http.ResponseWriter, _ *http.Request) {
	notImplementedYet(w, "/v1/images/edits", "the image model")
}

// notImplementedYet is the other 501: not "this process was started without
// the model" but "no flag would have loaded it", because the route is not
// written yet. The two are worth distinguishing, because only one of them is
// fixed by a restart.
func notImplementedYet(w http.ResponseWriter, route, what string) {
	writeError(w, http.StatusNotImplemented, "not_implemented",
		route+" is not implemented on this server yet: "+what+" is not wired up")
}
