package wyoming

import (
	"reflect"
	"testing"
)

// The checkpoint's own voice list, which is what backend.TTS.Voices reports.
// It is spelled out rather than read from models/ so that this test runs on a
// machine with no checkpoint, and because what is being checked is the naming
// convention and not the directory listing.
var kokoroVoices = []string{
	"af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
	"af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
	"am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
	"am_onyx", "am_puck", "am_santa",
	"bf_alice", "bf_emma", "bf_isabella", "bf_lily",
	"bm_daniel", "bm_fable", "bm_george", "bm_lewis",
	"ef_dora", "em_alex", "em_santa", "ff_siwis",
	"hf_alpha", "hf_beta", "hm_omega", "hm_psi",
	"if_sara", "im_nicola",
	"jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
	"pf_dora", "pm_alex", "pm_santa",
	"zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
	"zm_yunjian", "zm_yunxia", "zm_yunxi", "zm_yunyang",
}

func TestVoicesEnglishOnly(t *testing.T) {
	got := Voices(kokoroVoices, false)
	if len(got) != 28 {
		t.Errorf("%d voices, want the 28 English packs", len(got))
	}
	for _, v := range got {
		if len(v.Languages) != 1 || !englishTags[v.Languages[0]] {
			t.Errorf("%s: languages = %v, want one English tag", v.Name, v.Languages)
		}
		if !v.Installed {
			t.Errorf("%s: not installed", v.Name)
		}
		if v.Speakers != nil {
			t.Errorf("%s: speakers = %v, want none -- a pack is one speaker", v.Name, v.Speakers)
		}
		if v.Attribution.Name == "" || v.Attribution.URL == "" {
			t.Errorf("%s: attribution %+v, which the reference client will not decode",
				v.Name, v.Attribution)
		}
	}
	if want := []string{"en-GB", "en-US"}; !reflect.DeepEqual(Languages(got), want) {
		t.Errorf("languages = %v, want %v", Languages(got), want)
	}
}

func TestVoicesAll(t *testing.T) {
	got := Voices(kokoroVoices, true)
	if len(got) != len(kokoroVoices) {
		t.Errorf("%d voices, want all %d", len(got), len(kokoroVoices))
	}
	want := []string{"en-GB", "en-US", "es", "fr-fr", "hi", "it", "ja", "pt-br", "zh"}
	if !reflect.DeepEqual(Languages(got), want) {
		t.Errorf("languages = %v, want %v", Languages(got), want)
	}
	// Sorted by name, so a client's menu does not reshuffle across restarts.
	for i := 1; i < len(got); i++ {
		if got[i-1].Name >= got[i].Name {
			t.Fatalf("not sorted at %d: %q then %q", i, got[i-1].Name, got[i].Name)
		}
	}
}

func TestVoicesDescriptions(t *testing.T) {
	// Home Assistant shows the description in its voice menu and sends the
	// name back as the id, so the description has to be readable and the
	// pair has to stay distinguishable: three packs are called "santa".
	desc := map[string]string{}
	for _, v := range Voices(kokoroVoices, true) {
		desc[v.Name] = *v.Description
	}
	for name, want := range map[string]string{
		"af_heart":    "Heart (American English, female)",
		"bm_george":   "George (British English, male)",
		"am_santa":    "Santa (American English, male)",
		"em_santa":    "Santa (Spanish, male)",
		"pm_santa":    "Santa (Brazilian Portuguese, male)",
		"zf_xiaoni":   "Xiaoni (Mandarin Chinese, female)",
		"ff_siwis":    "Siwis (French, female)",
		"jf_alpha":    "Alpha (Japanese, female)",
		"if_sara":     "Sara (Italian, female)",
		"hm_omega":    "Omega (Hindi, male)",
		"af_bella":    "Bella (American English, female)",
		"bf_isabella": "Isabella (British English, female)",
	} {
		if got := desc[name]; got != want {
			t.Errorf("%s: %q, want %q", name, got, want)
		}
	}
	seen := map[string]string{}
	for name, d := range desc {
		if other, dup := seen[d]; dup {
			t.Errorf("%s and %s are both %q", name, other, d)
		}
		seen[d] = name
	}
}

func TestVoicesUnknownPrefix(t *testing.T) {
	// A name outside kokoro's convention is described with no language, and
	// is therefore never offered with -wyoming-all-voices off.
	got := Voices([]string{"af_heart", "custom", "xx_thing"}, true)
	if len(got) != 3 {
		t.Fatalf("%d voices, want 3", len(got))
	}
	for _, v := range got {
		if v.Name == "af_heart" {
			continue
		}
		if len(v.Languages) != 0 {
			t.Errorf("%s: languages = %v, want none", v.Name, v.Languages)
		}
		if *v.Description != v.Name {
			t.Errorf("%s: description = %q, want the name back", v.Name, *v.Description)
		}
	}
	if english := Voices([]string{"custom"}, false); len(english) != 0 {
		t.Errorf("a nameless-language pack was advertised: %+v", english)
	}
}

func TestASRLanguages(t *testing.T) {
	// The 25 of parakeet-tdt-0.6b-v3's release, each a distinct two-letter
	// tag. Home Assistant uses them to decide which pipeline languages may
	// select this engine, so a duplicate or a typo silently removes one.
	if len(asrLanguages) != 25 {
		t.Errorf("%d languages, want 25", len(asrLanguages))
	}
	seen := map[string]bool{}
	for _, tag := range asrLanguages {
		if len(tag) != 2 {
			t.Errorf("%q is not a two-letter tag", tag)
		}
		if seen[tag] {
			t.Errorf("%q twice", tag)
		}
		seen[tag] = true
	}
	if !seen["en"] {
		t.Error("english is missing")
	}
}
