package parakeet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Tokenizer decodes parakeet's token ids into text.
//
// It only decodes. An ASR model never encodes: text goes in nowhere, and the
// prediction network is fed its own previous *ids*, so the merge table, the
// `Precompiled` normaliser and the byte-fallback machinery in tokenizer.json
// are all encode-side and none of them is read here. What is left is a
// vocabulary lookup and one rule.
//
// That rule is Metaspace: SentencePiece writes a word boundary as U+2581
// ("▁") glued to the front of the token that follows it, so decoding is a
// concatenation, a replacement of ▁ by a space, and — because
// `prepend_scheme` is "always" — the removal of the single leading space that
// the first word's marker produces.
//
// Note this is *not* the byte-level BPE zimage/tokenizer implements for
// Qwen2: there is no byte-to-rune alphabet in the way, and the vocabulary
// holds the text itself.
type Tokenizer struct {
	tokens  []string // id -> piece
	special map[int]bool
}

// LoadTokenizer reads a checkpoint's tokenizer.json.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var raw struct {
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		Decoder struct {
			Type          string `json:"type"`
			Replacement   string `json:"replacement"`
			PrependScheme string `json:"prepend_scheme"`
		} `json:"decoder"`
		Model struct {
			Type  string         `json:"type"`
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("parakeet: parsing tokenizer.json: %w", err)
	}
	// The decoder is the half of the file this depends on, so a checkpoint
	// that changes it should fail here rather than silently produce text with
	// no spaces in it.
	if raw.Decoder.Type != "Metaspace" || raw.Decoder.Replacement != metaspace ||
		raw.Decoder.PrependScheme != "always" {
		return nil, fmt.Errorf("parakeet: tokenizer.json has a %q decoder (%q, %q), this reads Metaspace/▁/always",
			raw.Decoder.Type, raw.Decoder.Replacement, raw.Decoder.PrependScheme)
	}

	t := &Tokenizer{special: map[int]bool{}}
	for piece, id := range raw.Model.Vocab {
		for id >= len(t.tokens) {
			t.tokens = append(t.tokens, "")
		}
		t.tokens[id] = piece
	}
	for _, a := range raw.AddedTokens {
		for a.ID >= len(t.tokens) {
			t.tokens = append(t.tokens, "")
		}
		t.tokens[a.ID] = a.Content
		if a.Special {
			t.special[a.ID] = true
		}
	}
	if len(t.tokens) == 0 {
		return nil, fmt.Errorf("parakeet: tokenizer.json has an empty vocabulary")
	}
	return t, nil
}

const metaspace = "▁"

// Size is the number of ids the vocabulary covers, which is the model's
// `vocab_size`: the blank has an entry of its own at 8192, added as
// `<blank>` and flagged special, so Decode drops it like any other special
// token and a raw emission stream can be handed straight to it.
func (t *Tokenizer) Size() int { return len(t.tokens) }

// Piece returns the raw token for an id, U+2581 markers and all.
func (t *Tokenizer) Piece(id int) (string, error) {
	if id < 0 || id >= len(t.tokens) {
		return "", fmt.Errorf("parakeet: token id %d outside a vocabulary of %d", id, len(t.tokens))
	}
	return t.tokens[id], nil
}

// Decode turns emitted ids into text.
//
// `<unk>`, `<pad>` and `<blank>` are dropped, matching transformers'
// `skip_special_tokens`;
// the `<|pnc|>`-style control tokens are *not* flagged special in
// tokenizer.json and so decode literally, which is what the reference does
// and therefore what the comparison has to reproduce.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		if t.special[id] {
			continue
		}
		piece, err := t.Piece(id)
		if err != nil {
			return "", err
		}
		b.WriteString(piece)
	}
	out := strings.ReplaceAll(b.String(), metaspace, " ")
	// prepend_scheme "always" puts a marker before the first word; exactly
	// one leading space comes back off, so text that legitimately began with
	// whitespace is not silently trimmed.
	return strings.TrimPrefix(out, " "), nil
}
