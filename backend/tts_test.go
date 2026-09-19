package backend

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strix-halo-vulkan/api"
)

// The speech adapter's voice handling, which is the half of it that does not
// need a device: the checkpoint is 360 MB of mapping and loads in ~150 ms, and
// the utterances here are short.
const ttsModelDir = "../models/Kokoro-82M"

func loadTTS(t *testing.T) *TTS {
	t.Helper()
	if _, err := os.Stat(filepath.Join(ttsModelDir, "model.safetensors")); err != nil {
		t.Skipf("no converted checkpoint in %s (%v); run reference/convert_kokoro.py", ttsModelDir, err)
	}
	// No lexicon: every request below sends phonemes, so the front end is
	// not what is under test and espeak is not on the path.
	b, err := NewTTS(TTSOptions{Model: ttsModelDir})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Close)
	return b
}

// The dump's sentence, so the utterance is one the rest of the tests use.
const testPhonemes = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ."

// TestSpeakBlend is T8's request: the three-way mix that used to be a 400.
//
// A blend is not a crossfade of two utterances -- half the style vector
// conditions the *predictor*, so the durations move too -- which is why this
// checks the length as well as that samples came back at all.
func TestSpeakBlend(t *testing.T) {
	b := loadTTS(t)
	for _, voice := range []string{
		"af_heart",
		"af_alloy,af_bella,af_heart",
		"af_bella:3,af_sky:1",
		" af_bella , af_sky ",
	} {
		clip, err := b.Speak(context.Background(), &api.SpeechRequest{
			Phonemes: testPhonemes, Voice: voice,
		})
		if err != nil {
			t.Fatalf("%q: %v", voice, err)
		}
		if len(clip.Samples) == 0 {
			t.Fatalf("%q: no samples", voice)
		}
		if d := clip.Duration(); d < 2 || d > 5 {
			t.Errorf("%q: %.3fs of audio for a 48-phoneme sentence", voice, d)
		}
	}
}

// TestSpeakUnknownVoiceNamesTheComponent is the message that opened T8. The
// error has to be api.ErrUnsupported so the endpoint answers 400 rather than
// 500, and it has to name the component rather than the whole string.
func TestSpeakUnknownVoiceNamesTheComponent(t *testing.T) {
	b := loadTTS(t)
	_, err := b.Speak(context.Background(), &api.SpeechRequest{
		Phonemes: testPhonemes, Voice: "af_bella,af_nobody,af_sky",
	})
	if err == nil {
		t.Fatal("a blend naming a voice that does not exist was spoken")
	}
	if !errors.Is(err, api.ErrUnsupported) {
		t.Errorf("error %v is not api.ErrUnsupported, so the endpoint would answer 500", err)
	}
	if !strings.Contains(err.Error(), `"af_nobody"`) {
		t.Errorf("error does not name the component that is missing: %v", err)
	}
	if !strings.Contains(err.Error(), "af_bella,af_nobody,af_sky") {
		t.Errorf("error does not carry the blend it came from: %v", err)
	}
}

// TestNewTTSRejectsAnUnknownDefault: a typo in -voice is a refusal to start,
// not a 400 on the first request, and that has to go on holding for a blend.
func TestNewTTSRejectsAnUnknownDefault(t *testing.T) {
	if _, err := os.Stat(filepath.Join(ttsModelDir, "model.safetensors")); err != nil {
		t.Skipf("no converted checkpoint in %s (%v)", ttsModelDir, err)
	}
	for _, voice := range []string{"af_nobody", "af_heart,af_nobody", "af_heart:"} {
		if _, err := NewTTS(TTSOptions{Model: ttsModelDir, Voice: voice}); err == nil {
			t.Errorf("-voice %q was accepted at startup", voice)
		}
	}
}
