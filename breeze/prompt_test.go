package breeze

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"strix-halo-vulkan/gemma4"
)

// modelDir is the checkpoint the tests read: BREEZE_MODEL, or models/Breeze-TTS-2.
func modelDir() string {
	if d := os.Getenv("BREEZE_MODEL"); d != "" {
		return d
	}
	return "../models/Breeze-TTS-2"
}

func loadTokenizer(t *testing.T) *gemma4.Tokenizer {
	t.Helper()
	tok, err := gemma4.LoadTokenizer(modelDir())
	if os.IsNotExist(err) {
		t.Skipf("no checkpoint at %s", modelDir())
	}
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestPromptIDs is the B1 gate: every branch of every case in
// reference/dump_breeze_tokens.py's corpus gives upstream's ids, text mask
// and segment lengths.
func TestPromptIDs(t *testing.T) {
	tok := loadTokenizer(t)
	raw, err := os.ReadFile("testdata/prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	type branch struct {
		IDs      []int32 `json:"ids"`
		TextMask []int   `json:"text_mask"`
		Segments []int   `json:"segments"`
	}
	var f struct {
		Cases []struct {
			Template    string  `json:"template"`
			Text        string  `json:"text"`
			Instruction *string `json:"instruction"`
			RefText     *string `json:"ref_text"`
			RefFrames   int     `json:"ref_frames"`
			Cond        branch  `json:"cond"`
			Uncond      *branch `json:"uncond"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	check := func(name string, got *Branch, want *branch) {
		t.Helper()
		mask := make([]int, len(got.TextMask))
		for i, m := range got.TextMask {
			if m {
				mask[i] = 1
			}
		}
		if !slices.Equal(got.IDs, want.IDs) || !slices.Equal(mask, want.TextMask) || !slices.Equal(got.Segments, want.Segments) {
			t.Errorf("%s:\n got %v %v\nwant %v %v", name, got.IDs, got.Segments, want.IDs, want.Segments)
		}
	}
	for i, c := range f.Cases {
		r := &Request{Text: c.Text, RefFrames: make([][16]int32, c.RefFrames)}
		if c.Instruction != nil {
			r.Instruction = *c.Instruction
		}
		if c.RefText != nil {
			r.RefText = *c.RefText
		}
		p, err := BuildPrompt(tok, r)
		if err != nil {
			t.Fatal(err)
		}
		if p.Template.String() != c.Template {
			t.Errorf("case %d: template %s, want %s", i, p.Template, c.Template)
			continue
		}
		check(c.Template+" cond "+c.Text, &p.Cond, &c.Cond)
		if (p.Uncond == nil) != (c.Uncond == nil) {
			t.Errorf("case %d: negative branch present %v, want %v", i, p.Uncond != nil, c.Uncond != nil)
		} else if p.Uncond != nil {
			check(c.Template+" uncond "+c.Text, p.Uncond, c.Uncond)
		}
	}
	t.Logf("%d cases", len(f.Cases))
}
