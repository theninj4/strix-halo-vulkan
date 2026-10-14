package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"strix-halo-vulkan/audio"
)

// fakeInstructed is the instructed model: it records the request it saw.
type fakeInstructed struct{ fakeSpeech }

func (f *fakeInstructed) Models() []Model {
	return []Model{{ID: "breeze-tts-2", Object: "model", OwnedBy: "local"}}
}
func (f *fakeInstructed) Voices() []string { return nil }
func (f *fakeInstructed) SpeakStream(ctx context.Context, req *SpeechRequest, start func(int) error, emit func([]float32) error) error {
	c, err := f.Speak(ctx, req)
	if err != nil {
		return err
	}
	if err := start(c.Rate); err != nil {
		return err
	}
	return emit(c.Samples)
}

// fakeVoices keeps voices in memory; DesignVoice records what it was asked.
type fakeVoices struct {
	voices []Voice
	design *VoiceDesignRequest
}

func (f *fakeVoices) CreateConsent(context.Context, string, string, *audio.Clip) (*VoiceConsent, error) {
	return nil, ErrUnsupported
}
func (f *fakeVoices) CreateVoice(context.Context, string, *audio.Clip, string, string) (*Voice, error) {
	return nil, ErrUnsupported
}
func (f *fakeVoices) DesignVoice(_ context.Context, name, instructions, input string, seed uint64) (*Voice, error) {
	f.design = &VoiceDesignRequest{Name: name, Instructions: instructions, Input: input, Seed: &seed}
	v := Voice{ID: "voice_" + strings.ToLower(name), Object: "audio.voice", Name: name, Transcript: input,
		Instructions: instructions, Seed: &seed}
	f.voices = append(f.voices, v)
	return &v, nil
}
func (f *fakeVoices) VoiceSample(id string) ([]byte, error) { return nil, ErrNotFound }
func (f *fakeVoices) ListVoices() []Voice                  { return f.voices }
func (f *fakeVoices) DeleteVoice(id string) error {
	for i, v := range f.voices {
		if v.ID == id {
			f.voices = append(f.voices[:i], f.voices[i+1:]...)
			return nil
		}
	}
	return ErrNotFound
}

// A designed voice is kept under its name, and the name then reaches it as
// a custom voice on the instructed model, past kokoro, the default.
func TestDesignedVoiceByName(t *testing.T) {
	kokoro, breeze, voices := &fakeSpeech{}, &fakeInstructed{}, &fakeVoices{}
	s := &Server{Speech: kokoro, InstructedSpeech: breeze, Voices: voices}
	line := "Pull up a chair and let me tell you about the winter the river froze."
	seed := uint64(7)
	rec := do(t, s, jsonRequest("POST", "/v1/audio/voices", VoiceDesignRequest{
		Name: "Storyteller", Instructions: "a gravelly old man", Input: line, Seed: &seed}))
	if rec.Code != http.StatusOK {
		t.Fatalf("design: %d %s", rec.Code, rec.Body)
	}
	var v Voice
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil || v.ID != "voice_storyteller" {
		t.Fatalf("design answered %s (%v)", rec.Body, err)
	}
	if d := voices.design; d.Input != line || *d.Seed != 7 {
		t.Errorf("the backend was asked %+v", d)
	}

	rec = do(t, s, jsonRequest("POST", "/v1/audio/speech", SpeechRequest{Input: "hello", Voice: "storyteller"}))
	if rec.Code != http.StatusOK {
		t.Fatalf("speech: %d %s", rec.Code, rec.Body)
	}
	if kokoro.req != nil || breeze.req == nil || !breeze.req.CustomVoice || breeze.req.Voice != "voice_storyteller" {
		t.Errorf("kokoro saw %+v, the instructed model %+v", kokoro.req, breeze.req)
	}

	for _, c := range []struct{ name, why string }{
		{"STORYTELLER", "already a voice"},
		{"af_heart", "built-in"},
		{"a,b", "may not contain"},
		{"voice_x", "may not contain"},
	} {
		rec := do(t, s, jsonRequest("POST", "/v1/audio/voices", VoiceDesignRequest{
			Name: c.name, Instructions: "x", Input: line}))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), c.why) {
			t.Errorf("name %q: %d %s", c.name, rec.Code, rec.Body)
		}
	}
	rec = do(t, s, jsonRequest("POST", "/v1/audio/voices", VoiceDesignRequest{Name: "n", Instructions: "x", Input: "too short"}))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("a short line: %d %s", rec.Code, rec.Body)
	}

	if rec := do(t, s, httptestRequest("DELETE", "/v1/audio/voices/storyteller")); rec.Code != http.StatusOK || len(voices.voices) != 0 {
		t.Errorf("delete by name: %d %s", rec.Code, rec.Body)
	}
}

func httptestRequest(method, path string) *http.Request {
	r, _ := http.NewRequest(method, path, nil)
	return r
}
