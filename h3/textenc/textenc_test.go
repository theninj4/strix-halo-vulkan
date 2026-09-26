package textenc

import (
	"encoding/json"
	"os"
	"testing"

	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference is reference/dump_h3_tokens.py (HF's tokenizer over the H3
// repository's tokenizer/, no weights). Regenerate with:
//
//	.venv/bin/python reference/dump_h3_tokens.py
const (
	tokensRef = "../../reference/out/h3tokens/tokens.json"
	modelDir  = "../../models/MiniMax-H3"
)

// TestTokens is the presentation gate: the Go tokenizer reproduces HF's ids
// on plain text, CJK, Context-IR's section and shot markup, `<d>` dialogue
// with Arabic, and the README's full prompt.
func TestTokens(t *testing.T) {
	buf, err := os.ReadFile(tokensRef)
	if err != nil {
		t.Skipf("no reference at %s (%v); run reference/dump_h3_tokens.py", tokensRef, err)
	}
	var ref map[string]struct {
		Prompt string  `json:"prompt"`
		IDs    []int32 `json:"ids"`
	}
	if err := json.Unmarshal(buf, &ref); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(modelDir + "/tokenizer")
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	for label, c := range ref {
		got, err := EncodePrompt(tok, c.Prompt)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		if len(got) != len(c.IDs) {
			t.Errorf("%s: %d tokens, want %d", label, len(got), len(c.IDs))
			continue
		}
		for i := range got {
			if got[i] != c.IDs[i] {
				t.Errorf("%s: token %d is %d, want %d", label, i, got[i], c.IDs[i])
				break
			}
		}
	}
}

// TestConfig pins the conditioner's shape: Layers of its layers must exist,
// and its width must be the transformer's text_dim.
func TestConfig(t *testing.T) {
	cfg, err := LoadConfig(modelDir + "/text_encoder")
	if err != nil {
		t.Skipf("no text encoder config (%v)", err)
	}
	if cfg.NumLayers <= Layers {
		t.Errorf("%d layers; hidden_states[%d] needs more", cfg.NumLayers, Layers)
	}
	if cfg.HiddenSize != Width {
		t.Errorf("hidden size %d, want %d", cfg.HiddenSize, Width)
	}
}
