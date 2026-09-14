package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The reference comes from reference/dump_tokenizer.py, which tokenizes the
// same corpus with HF's fast Qwen2 tokenizer out of the checkpoint's own
// tokenizer/ directory. Regenerate with:
//
//	.venv/bin/python reference/dump_tokenizer.py
const (
	refDir  = "../../reference/out/tokenizer"
	tokDir  = "../../models/Z-Image-Turbo/tokenizer"
	caseFmt = "%s: %s"
)

type refCase struct {
	Name      string  `json:"name"`
	Text      string  `json:"text"`
	IDs       []int32 `json:"ids"`
	Rendered  string  `json:"rendered"`
	PromptIDs []int32 `json:"prompt_ids"`
}

type refCases struct {
	VocabSize int       `json:"vocab_size"`
	Cases     []refCase `json:"cases"`
	NFC       refCase   `json:"nfc"`
}

func loadCases(t *testing.T) *refCases {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(refDir, "cases.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_tokenizer.py", refDir, err)
	}
	var c refCases
	if err := json.Unmarshal(buf, &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

func load(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := Load(tokDir)
	if err != nil {
		t.Skipf("no tokenizer in %s (%v)", tokDir, err)
	}
	return tok
}

func diff(got, want []int32) string {
	if len(got) != len(want) {
		return "length " + itoa(len(got)) + ", want " + itoa(len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return "id " + itoa(i) + " is " + itoa(int(got[i])) + ", want " + itoa(int(want[i]))
		}
	}
	return ""
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b []byte
	for v > 0 {
		b = append([]byte{byte('0' + v%10)}, b...)
		v /= 10
	}
	if neg {
		return "-" + string(b)
	}
	return string(b)
}

// TestEncode is the stage's main check: every case, raw and through the chat
// template, id for id.
func TestEncode(t *testing.T) {
	tok, cases := load(t), loadCases(t)
	for _, c := range cases.Cases {
		got, err := tok.Encode(c.Text)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if d := diff(got, c.IDs); d != "" {
			t.Errorf("%s: %s\n  text %q\n  got  %v\n  want %v", c.Name, d, c.Text, got, c.IDs)
			continue
		}
		t.Logf("%-22s %3d ids", c.Name, len(got))
	}
}

// TestChatTemplate checks the hand-transcribed template against the one
// Jinja rendered, and then the ids it produces. ChatPrompt is the only piece
// of this package not read out of tokenizer.json, so it is the piece most
// able to drift.
func TestChatTemplate(t *testing.T) {
	tok, cases := load(t), loadCases(t)
	for _, c := range cases.Cases {
		if got := ChatPrompt(c.Text); got != c.Rendered {
			t.Errorf("%s: template rendered\n  got  %q\n  want %q", c.Name, got, c.Rendered)
			continue
		}
		got, err := tok.EncodePrompt(c.Text)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if d := diff(got, c.PromptIDs); d != "" {
			t.Errorf("%s: %s\n  got  %v\n  want %v", c.Name, d, got, c.PromptIDs)
		}
	}
}

// TestSpecialTokens checks that an added token written out in the text
// becomes one id rather than being taken apart by BPE -- which is the whole
// mechanism the chat template rides on.
func TestSpecialTokens(t *testing.T) {
	tok := load(t)
	ids, err := tok.EncodePrompt("a cat")
	if err != nil {
		t.Fatal(err)
	}
	imStart, ok := tok.ID("<|im_start|>")
	if !ok {
		t.Fatal("no <|im_start|> in the vocabulary")
	}
	imEnd, _ := tok.ID("<|im_end|>")
	if ids[0] != imStart {
		t.Errorf("first id is %d, want <|im_start|> = %d", ids[0], imStart)
	}
	var seenEnd bool
	for _, id := range ids {
		seenEnd = seenEnd || id == imEnd
	}
	if !seenEnd {
		t.Errorf("no <|im_end|> in %v", ids)
	}
}

// TestRoundTrip decodes every case back and requires the original text. The
// byte-level alphabet makes this exact, so any loss is a mapping bug.
func TestRoundTrip(t *testing.T) {
	tok, cases := load(t), loadCases(t)
	for _, c := range cases.Cases {
		ids, err := tok.Encode(c.Text)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		back, err := tok.Decode(ids)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if back != c.Text {
			t.Errorf("%s: round trip gave %q, want %q", c.Name, back, c.Text)
		}
	}
}

// TestNFCIsTheKnownGap measures the one divergence this package documents
// rather than fixes: the reference normalises to NFC and this does not. The
// test asserts the *shape* of the gap -- decomposed input differs, and the
// composed spelling of the same string agrees -- so that the day someone
// adds normalisation, this is what tells them it worked.
func TestNFCIsTheKnownGap(t *testing.T) {
	tok, cases := load(t), loadCases(t)
	c := cases.NFC
	if !strings.Contains(c.Text, "́") {
		t.Fatalf("the NFC case %q holds no combining mark; the dump is not testing what it says", c.Text)
	}
	got, err := tok.Encode(c.Text)
	if err != nil {
		t.Fatal(err)
	}
	if diff(got, c.IDs) == "" {
		t.Errorf("decomposed %q tokenized identically to the NFC reference; "+
			"either normalisation was added (update this test) or the dump is stale", c.Text)
	}
	composed := strings.ReplaceAll(c.Text, "é", "é")
	gotComposed, err := tok.Encode(composed)
	if err != nil {
		t.Fatal(err)
	}
	if d := diff(gotComposed, c.IDs); d != "" {
		t.Errorf("the composed spelling %q should match the NFC reference: %s", composed, d)
	}
	t.Logf("decomposed %v vs NFC %v", got, c.IDs)
}

// TestValidationDetectsErrors is the negative control: each break is a
// plausible mistake in this package, and the corpus above has to catch it.
// Without it these tests would prove only that the tokenizer runs.
func TestValidationDetectsErrors(t *testing.T) {
	cases := loadCases(t)
	breaks := []struct {
		name  string
		apply func(tok *Tokenizer)
	}{
		{"no merges", func(tok *Tokenizer) { tok.ranks = map[string]int{} }},
		{"merge order reversed", func(tok *Tokenizer) {
			for k, v := range tok.ranks {
				tok.ranks[k] = -v
			}
		}},
		{"byte alphabet identity", func(tok *Tokenizer) {
			for b := 0; b < 256; b++ {
				tok.byteToRune[b] = rune(b)
			}
		}},
		{"special tokens ignored", func(tok *Tokenizer) { tok.specials = nil }},
	}
	for _, b := range breaks {
		tok := load(t)
		b.apply(tok)
		caught := 0
		for _, c := range cases.Cases {
			got, err := tok.Encode(c.Text)
			if err != nil || diff(got, c.IDs) != "" {
				caught++
			}
		}
		if caught == 0 {
			t.Errorf("%s: broke the tokenizer and every one of the %d cases still passed",
				b.name, len(cases.Cases))
			continue
		}
		t.Logf("%-24s caught by %d of %d cases", b.name, caught, len(cases.Cases))
	}
}
