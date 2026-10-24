package backend

import (
	"errors"
	"strings"
	"testing"

	"strix-halo-vulkan/api"
)

func TestIsMusicMM3(t *testing.T) {
	for model, want := range map[string]bool{
		"MiniMax-Music3": true, "minimax-music3": true, "MiniMaxAI/MiniMax-Music3": true, "minimax": true, "music3": true,
		"": false, "acestep-v15-xl-base": false, "turbo": false, "music": false, "minimaxer": false, "music30": false,
	} {
		if got := IsMusicMM3(model); got != want {
			t.Errorf("IsMusicMM3(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestMusicMM3Caption(t *testing.T) {
	for _, c := range []struct {
		req  api.MusicRequest
		want string
	}{
		{api.MusicRequest{Caption: "A ballad."}, "A ballad."},
		{api.MusicRequest{Caption: "A ballad.", BPM: 92, KeyScale: "E minor"},
			"Global Metadata\nBasic Attributes: bpm is 92. key is E, and scale is minor.\nA ballad."},
		{api.MusicRequest{Caption: "x", TimeSignature: "3", Language: "en"},
			"Global Metadata\nBasic Attributes: The time signature is 3/4. Sung in English.\nx"},
		// A caption with its own Basic Attributes takes the request's metas
		// there, in place of the ones it states.
		{api.MusicRequest{Caption: "Global Metadata\nBasic Attributes: bpm is 76. key is D, and scale is major. Indie Folk.\nMore.",
			BPM: 100, KeyScale: "A minor"},
			"Global Metadata\nBasic Attributes: bpm is 100. key is A, and scale is minor. Indie Folk.\nMore."},
		{api.MusicRequest{Caption: "Basic Attributes: bpm is 76. Folk.", KeyScale: "C major"},
			"Basic Attributes: key is C, and scale is major. bpm is 76. Folk."},
	} {
		if got := mm3Caption(&c.req); got != c.want {
			t.Errorf("mm3Caption(%+v):\n got %q\nwant %q", c.req, got, c.want)
		}
	}
}

// TestMusicMM3Refusals: everything ACE's is a 400 that names ACE, before
// any device work (a backend with no pipeline and no tokenizer would
// panic past the refusals).
func TestMusicMM3Refusals(t *testing.T) {
	b := &MusicMM3{}
	f := 1.0
	var seed int64 = 3
	for name, req := range map[string]api.MusicRequest{
		"thinking": {Caption: "x", Thinking: true, ThinkingSet: true}, "sample": {Caption: "x", Sample: true},
		"analysis": {Analysis: true}, "cover": {Caption: "x", Task: "cover"},
		"reference": {Caption: "x", ReferenceAudio: []byte{1}}, "shift": {Caption: "x", Shift: 3},
		"steps": {Caption: "x", InferenceSteps: 8}, "guidance": {Caption: "x", GuidanceScale: &f},
		"lm": {Caption: "x", LMTemperature: &f}, "lm seed": {Caption: "x", LMSeed: &seed},
		"score": {Caption: "x", Score: true, ScoreSet: true}, "no caption": {Caption: "  "},
	} {
		_, err := b.PlanMusic(&req)
		if !errors.Is(err, api.ErrUnsupported) {
			t.Errorf("%s: %v, want a 400", name, err)
			continue
		}
		if name != "score" && name != "no caption" && !strings.Contains(err.Error(), "ACE-Step's") {
			t.Errorf("%s: %v does not name ACE", name, err)
		}
	}
}

func TestMusicRouter(t *testing.T) {
	ace, mm := &Music{}, &MusicMM3{}
	r := &MusicRouter{ACE: ace, MM3: mm}
	for model, want := range map[string]api.MusicBackend{"": ace, "turbo": ace, "MiniMax-Music3": mm, "gpt-music": ace} {
		if got := r.pick(model); got != want {
			t.Errorf("pick(%q) = %T", model, got)
		}
	}
	if got := (&MusicRouter{MM3: mm}).pick(""); got != api.MusicBackend(mm) {
		t.Errorf("MiniMax alone: pick = %T", got)
	}
}
