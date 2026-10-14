package backend

import (
	"testing"

	"strix-halo-vulkan/api"
)

// TestSpeechEnd: the closing breath heard as a word is cut after the text's
// last word; an ending heard as the text, or not heard as it, is kept.
func TestSpeechEnd(t *testing.T) {
	words := func(ws ...string) []api.TranscriptionWord {
		var out []api.TranscriptionWord
		for i, w := range ws {
			out = append(out, api.TranscriptionWord{Word: w, Start: float64(i) / 2, End: float64(i+1) / 2})
		}
		return out
	}
	for _, c := range []struct {
		text  string
		heard []api.TranscriptionWord
		end   float64
		ok    bool
	}{
		{"from the west.", words("from", "the", "west,", "and"), 1.5, true},
		{"from the west.", words("west", "said"), 0.5, true},
		{"from the west.", words("the", "west."), 0, false},
		{"They might hear us.", words("might", "hear", "it."), 0, false},
		{"See you next week!", words("next", "Week", "uh", "hm"), 1, true},
		{"It's the West's", words("the", "west's", "and"), 1, true},
		{"breakfast show", words("the", "brick."), 0, false},
		{"", words("and"), 0, false},
	} {
		end, ok := speechEnd(c.heard, c.text)
		if end != c.end || ok != c.ok {
			t.Errorf("%q heard as %v: %v %v, want %v %v", c.text, c.heard, end, ok, c.end, c.ok)
		}
	}
}
