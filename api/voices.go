package api

// Custom voices (research/tts-instruct-vertical.md B8), OpenAI's flow
// (developers.openai.com/api/docs/guides/custom-voices):
//
//	GET  /v1/audio/consent_phrases   the phrase a speaker reads to consent
//	POST /v1/audio/voice_consents    name, language, recording -> {id: "cons_..."}
//	POST /v1/audio/voices            name, audio_sample, consent -> {id: "voice_..."}
//	POST /v1/audio/speech            voice: {"id": "voice_..."}
//
// plus two extensions OpenAI's guide leaves to its dashboard: GET
// /v1/audio/voices lists them and DELETE /v1/audio/voices/{id} removes one.
//
// Designed voices are an extension: POST /v1/audio/voices with a JSON body
// {name, instructions, input, seed} has the model speak input in the voice
// instructions describe and keeps the result as a voice (a clone of its own
// output), since without a reference the model draws a new speaker every
// request and a seed holds one only for the same text. The same fields on
// /v1/audio/speech give the same clip, so a caller auditions seeds there
// and keeps the one it likes. No consent is asked: no one's voice is
// recorded. GET /v1/audio/voices/{id}/sample returns its clip.
//
// A voice's name is an address too (an extension): `voice: "narrator"`
// reaches the voice named narrator, so names are unique, and may not be a
// built-in voice's.
//
// What the server checks, and what it cannot:
//
//   - the consent recording is transcribed (the speech-to-text model, so
//     -stt is required) and must be the phrase, word for word, allowing one
//     word for the recogniser's errors;
//   - the voice sample is transcribed the same way, because the model clones
//     from the reference audio *and its exact transcript*, as OpenAI also
//     transcribes it itself; a `transcript` field (an extension) supplies it
//     instead, which is how a language the recogniser does not read gets in;
//   - **that the consent and the sample are the same speaker is not
//     checked**: OpenAI rejects a mismatch, and nothing here can tell two
//     speakers apart. The consent recording is kept with the voice.

import (
	"context"
	crand "crypto/rand"
	"encoding/binary"
	"errors"
	"io"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"strix-halo-vulkan/audio"
)

// ConsentPhrase is what a speaker reads, in English: OpenAI's phrase with
// this server in place of OpenAI.
const ConsentPhrase = "I am the owner of this voice and I consent to this server using this voice to create a synthetic voice model."

// VoiceConsent is a consent recording, accepted.
type VoiceConsent struct {
	ID        string `json:"id"`
	Object    string `json:"object"` // "audio.voice_consent"
	Name      string `json:"name"`
	Language  string `json:"language"`
	CreatedAt int64  `json:"created_at"`
}

// Voice is a custom voice.
type Voice struct {
	ID         string  `json:"id"`
	Object     string  `json:"object"` // "audio.voice"
	Name       string  `json:"name"`
	Consent    string  `json:"consent"`
	CreatedAt  int64   `json:"created_at"`
	Transcript string  `json:"transcript"`
	Seconds    float64 `json:"seconds"`
	// Instructions and Seed are a designed voice's: what made it.
	Instructions string  `json:"instructions,omitempty"`
	Seed         *uint64 `json:"seed,omitempty"`
}

// VoiceBackend stores consents and voices for a speech model that clones.
type VoiceBackend interface {
	// CreateConsent stores a consent recording the HTTP layer has checked.
	CreateConsent(ctx context.Context, name, language string, rec *audio.Clip) (*VoiceConsent, error)
	// CreateVoice encodes a sample against a stored consent.
	CreateVoice(ctx context.Context, name string, sample *audio.Clip, transcript, consent string) (*Voice, error)
	// DesignVoice speaks input in the voice instructions describe, under
	// seed, and keeps it as a voice.
	DesignVoice(ctx context.Context, name, instructions, input string, seed uint64) (*Voice, error)
	// VoiceSample is a designed voice's clip as WAV; ErrNotFound for one
	// that has none.
	VoiceSample(id string) ([]byte, error)
	// ListVoices is every stored voice, oldest first.
	ListVoices() []Voice
	// DeleteVoice removes one; ErrNotFound if there is none.
	DeleteVoice(id string) error
}

// ErrNotFound is a backend's "no such object", a 404.
var ErrNotFound = errNotFound{}

type errNotFound struct{}

func (errNotFound) Error() string { return "not found" }

// maxVoiceSeconds is OpenAI's bound on a sample; minVoiceSeconds is the
// guide's "at least five seconds of speech".
const (
	maxVoiceSeconds = 30
	minVoiceSeconds = 5
)

func (s *Server) handleConsentPhrases(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]string{
		{"object": "audio.consent_phrase", "language": "en", "phrase": ConsentPhrase},
	}})
}

// voiceUpload reads a multipart form's audio part as a mono clip.
func voiceUpload(w http.ResponseWriter, r *http.Request, part string) (*audio.Clip, bool) {
	ctx := r.Context()
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if !tooLarge(ctx, w, err) {
			badRequest(ctx, w, "send multipart/form-data: "+err.Error())
		}
		return nil, false
	}
	f, _, err := r.FormFile(part)
	if err != nil {
		badRequest(ctx, w, "no '"+part+"' part in the form: "+err.Error())
		return nil, false
	}
	defer f.Close()
	b, err := io.ReadAll(f)
	if err != nil {
		badRequest(ctx, w, "reading '"+part+"': "+err.Error())
		return nil, false
	}
	clip, err := audio.DecodeWAV(b)
	if err != nil {
		badRequest(ctx, w, "this server decodes 16-bit PCM WAV only: "+err.Error()+
			"; convert with `ffmpeg -i in -ac 1 -c:a pcm_s16le out.wav`")
		return nil, false
	}
	return clip, true
}

// transcribe runs the speech-to-text model on a clip at its own rate.
func (s *Server) transcribe(ctx context.Context, clip *audio.Clip) (string, error) {
	c16, err := audio.Resample(clip, 16000)
	if err != nil {
		return "", err
	}
	resp, err := s.Transcription.Transcribe(ctx, c16, &TranscriptionRequest{Language: "en"})
	if err != nil {
		return "", err
	}
	return resp.Text, nil
}

var nonWord = regexp.MustCompile(`[^a-z0-9' ]+`)

func words(s string) []string {
	return strings.Fields(nonWord.ReplaceAllString(strings.ToLower(s), " "))
}

// wordDistance is the edit distance between two word sequences.
func wordDistance(a, b []string) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			c := prev[j-1]
			if a[i-1] != b[j-1] {
				c++
			}
			cur[j] = min(c, prev[j]+1, cur[j-1]+1)
		}
		prev = cur
	}
	return prev[len(b)]
}

func (s *Server) handleVoiceConsent(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Voices == nil {
		notLoaded(ctx, w, "custom voices", "-tts-instruct")
		return
	}
	if s.Transcription == nil {
		notLoaded(ctx, w, "the consent check (speech to text)", "-stt")
		return
	}
	clip, ok := voiceUpload(w, r, "recording")
	if !ok {
		return
	}
	name, lang := r.FormValue("name"), r.FormValue("language")
	if name == "" {
		badRequest(ctx, w, "name is required")
		return
	}
	if lang == "" {
		lang = "en"
	}
	if lang != "en" {
		badRequest(ctx, w, "consent can be recorded in en only: GET /v1/audio/consent_phrases")
		return
	}
	heard, err := s.transcribe(ctx, clip)
	if err != nil {
		backendError(ctx, w, "voice consent", err)
		return
	}
	if d := wordDistance(words(heard), words(ConsentPhrase)); d > 1 {
		badRequest(ctx, w, "the recording is not the consent phrase (heard "+strconv.Quote(heard)+
			"); read it exactly: "+strconv.Quote(ConsentPhrase))
		return
	}
	c, err := s.Voices.CreateConsent(ctx, name, lang, clip)
	if err != nil {
		backendError(ctx, w, "voice consent", err)
		return
	}
	writeJSON(w, http.StatusOK, c)
}

// voiceNamed is the stored voice called name, ignoring case; nil if none.
func (s *Server) voiceNamed(name string) *Voice {
	if s.Voices == nil || name == "" {
		return nil
	}
	for _, v := range s.Voices.ListVoices() {
		if strings.EqualFold(v.Name, name) {
			return &v
		}
	}
	return nil
}

// checkVoiceName refuses a name that cannot be an address: taken, a
// built-in voice's, an id's, or a kokoro mix's spelling.
func (s *Server) checkVoiceName(name string) string {
	switch {
	case name == "":
		return "name is required"
	case strings.ContainsAny(name, ",:") || strings.HasPrefix(name, "voice_"):
		return "a voice's name is how requests reach it, so it may not contain ',' or ':' or start with voice_"
	case s.voiceNamed(name) != nil:
		return "there is already a voice named " + strconv.Quote(name) + " (" + s.voiceNamed(name).ID +
			"); DELETE /v1/audio/voices/{id} it first"
	}
	for _, b := range []SpeechBackend{s.Speech, s.InstructedSpeech} {
		if b == nil {
			continue
		}
		for _, v := range b.Voices() {
			if strings.EqualFold(v, name) {
				return strconv.Quote(name) + " is a built-in voice's name"
			}
		}
	}
	return ""
}

func (s *Server) handleVoiceCreate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Voices == nil {
		notLoaded(ctx, w, "custom voices", "-tts-instruct")
		return
	}
	if mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); mt == "application/json" {
		s.handleVoiceDesign(w, r)
		return
	}
	clip, ok := voiceUpload(w, r, "audio_sample")
	if !ok {
		return
	}
	name, consent, transcript := r.FormValue("name"), r.FormValue("consent"), r.FormValue("transcript")
	if name == "" || consent == "" {
		badRequest(ctx, w, "name and consent are required")
		return
	}
	if msg := s.checkVoiceName(name); msg != "" {
		badRequest(ctx, w, msg)
		return
	}
	if sec := clip.Duration(); sec > maxVoiceSeconds || sec < minVoiceSeconds {
		badRequest(ctx, w, "the sample is "+strconv.FormatFloat(sec, 'f', 1, 64)+" s; it must be 5 to 30 s of speech")
		return
	}
	if transcript == "" {
		if s.Transcription == nil {
			badRequest(ctx, w, "this server has no speech-to-text model to transcribe the sample; send its exact text in 'transcript'")
			return
		}
		var err error
		if transcript, err = s.transcribe(ctx, clip); err != nil {
			backendError(ctx, w, "voice", err)
			return
		}
	}
	if len(strings.Fields(transcript)) < 5 {
		badRequest(ctx, w, "the sample's transcript is "+strconv.Quote(transcript)+"; a voice needs several sentences of speech")
		return
	}
	v, err := s.Voices.CreateVoice(ctx, name, clip, transcript, consent)
	if err != nil {
		backendError(ctx, w, "voice", err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

// VoiceDesignRequest is a designed voice (an extension): the fields of the
// /v1/audio/speech request that auditioned it.
type VoiceDesignRequest struct {
	Name         string  `json:"name"`
	Instructions string  `json:"instructions"`
	Input        string  `json:"input"`
	Seed         *uint64 `json:"seed,omitempty"`
}

func (s *Server) handleVoiceDesign(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var req VoiceDesignRequest
	if !decodeJSON(ctx, w, r, &req) {
		return
	}
	if msg := s.checkVoiceName(req.Name); msg != "" {
		badRequest(ctx, w, msg)
		return
	}
	if req.Instructions == "" {
		badRequest(ctx, w, "instructions describe the voice to design; they are required")
		return
	}
	if len(strings.Fields(req.Input)) < 10 {
		badRequest(ctx, w, "input is the line the voice is kept from; give it a sentence or two (5-20 s of speech)")
		return
	}
	var seed uint64
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		var b [8]byte
		_, _ = crand.Read(b[:])
		seed = binary.LittleEndian.Uint64(b[:])
	}
	v, err := s.Voices.DesignVoice(ctx, req.Name, req.Instructions, req.Input, seed)
	if err != nil {
		backendError(ctx, w, "voice", err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func (s *Server) handleVoiceSample(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Voices == nil {
		notLoaded(ctx, w, "custom voices", "-tts-instruct")
		return
	}
	id := r.PathValue("id")
	if v := s.voiceNamed(id); v != nil {
		id = v.ID
	}
	b, err := s.Voices.VoiceSample(id)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "invalid_request_error",
				"no sample for "+strconv.Quote(id)+": only designed voices keep their clip")
			return
		}
		backendError(ctx, w, "voice", err)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	_, _ = w.Write(b)
}

func (s *Server) handleVoiceList(w http.ResponseWriter, r *http.Request) {
	if s.Voices == nil {
		notLoaded(r.Context(), w, "custom voices", "-tts-instruct")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": s.Voices.ListVoices()})
}

func (s *Server) handleVoiceDelete(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if s.Voices == nil {
		notLoaded(ctx, w, "custom voices", "-tts-instruct")
		return
	}
	id := r.PathValue("id")
	if v := s.voiceNamed(id); v != nil {
		id = v.ID
	}
	if err := s.Voices.DeleteVoice(id); err != nil {
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "invalid_request_error", "no voice "+strconv.Quote(id))
			return
		}
		backendError(ctx, w, "voice", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "audio.voice", "deleted": true})
}
