// Package tokenizer implements Qwen2's byte-level BPE tokenizer, which is
// what Z-Image's text encoder is fed.
//
// It is written against `tokenizer.json` -- the vocabulary, the merge table,
// the added tokens and the pre-tokenizer regex all come out of that file, so
// there is nothing transcribed by hand that could drift from the checkpoint.
// The one piece that is transcribed is the chat template (see ChatPrompt):
// rendering Jinja is not worth a dependency for a template with one
// variable in it.
//
// Two deliberate gaps, both measured in the tests rather than assumed:
//
//   - `tokenizer.json` asks for NFC normalisation and this does not
//     normalise. The repository has no third-party Go modules and NFC needs
//     the Unicode composition tables; prompts that are already NFC -- which
//     is everything a text editor or a browser produces -- are unaffected.
//     TestNFCIsTheKnownGap pins where the two diverge.
//   - Truncation is the caller's: the pipeline pads to 512 and masks the
//     padding straight back out, so the padding cannot change a hidden
//     state (the attention is causal and the padding is on the right), but
//     the 512-token *limit* is real and Encode does not enforce it.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Tokenizer converts text to token ids and back.
type Tokenizer struct {
	vocab  map[string]int32
	tokens []string       // id -> token, over the byte-level alphabet
	ranks  map[string]int // "a\x00b" -> merge priority, lower merges first

	specials  []string // added tokens, longest first
	specialID map[string]int32

	byteToRune [256]rune
	runeToByte map[rune]byte

	// marks selects the qwen35 pre-tokenizer over the qwen2 one; see split().
	marks bool
}

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Normalizer struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	Model struct {
		Type         string           `json:"type"`
		IgnoreMerges bool             `json:"ignore_merges"`
		ByteFallback bool             `json:"byte_fallback"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       [][2]string      `json:"merges"`
	} `json:"model"`
}

// Load reads a tokenizer directory -- the one next to the checkpoint, e.g.
// models/Z-Image-Turbo/tokenizer.
func Load(dir string) (*Tokenizer, error) {
	buf, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var raw tokenizerJSON
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("tokenizer: parsing tokenizer.json: %w", err)
	}
	// These three would each change the algorithm below, so they are checked
	// rather than trusted: this checkpoint has plain BPE with no unknown
	// token and no byte fallback, because byte-level input cannot miss.
	if raw.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: model is %q, want BPE", raw.Model.Type)
	}
	if raw.Model.IgnoreMerges || raw.Model.ByteFallback {
		return nil, fmt.Errorf("tokenizer: ignore_merges/byte_fallback are set and unimplemented")
	}

	t := &Tokenizer{
		vocab:      raw.Model.Vocab,
		ranks:      make(map[string]int, len(raw.Model.Merges)),
		specialID:  make(map[string]int32, len(raw.AddedTokens)),
		runeToByte: make(map[rune]byte, 256),
	}
	for i, m := range raw.Model.Merges {
		t.ranks[m[0]+"\x00"+m[1]] = i
	}
	for _, a := range raw.AddedTokens {
		t.specials = append(t.specials, a.Content)
		t.specialID[a.Content] = a.ID
	}
	// Longest first, so "<|im_start|>" is not shadowed by a shorter added
	// token that happens to be a prefix of it.
	sort.Slice(t.specials, func(i, j int) bool { return len(t.specials[i]) > len(t.specials[j]) })

	max := int32(-1)
	for _, id := range t.vocab {
		if id > max {
			max = id
		}
	}
	for _, id := range t.specialID {
		if id > max {
			max = id
		}
	}
	t.tokens = make([]string, max+1)
	for tok, id := range t.vocab {
		t.tokens[id] = tok
	}
	for tok, id := range t.specialID {
		t.tokens[id] = tok
	}

	t.buildByteAlphabet()
	return t, nil
}

// buildByteAlphabet is GPT-2's bytes_to_unicode: the 188 bytes that are
// already printable and not space map to themselves, and the other 68 map to
// U+0100 upward. It is what makes every byte sequence representable in a
// vocabulary of ordinary strings, which is why byte fallback is unnecessary.
func (t *Tokenizer) buildByteAlphabet() {
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	next := rune(256)
	for b := 0; b < 256; b++ {
		if printable(b) {
			t.byteToRune[b] = rune(b)
		} else {
			t.byteToRune[b] = next
			next++
		}
		t.runeToByte[t.byteToRune[b]] = byte(b)
	}
}

// Size is the number of ids, vocabulary plus added tokens. It is not the
// model's vocab_size, which is padded upward.
func (t *Tokenizer) Size() int { return len(t.tokens) }

// ID returns the id of a token string, e.g. an added token like "<|im_end|>".
func (t *Tokenizer) ID(token string) (int32, bool) {
	if id, ok := t.specialID[token]; ok {
		return id, true
	}
	id, ok := t.vocab[token]
	return id, ok
}

// ChatPrompt wraps a prompt the way ZImagePipeline._encode_prompt does:
// tokenizer.apply_chat_template of a single user message, with
// add_generation_prompt=True and enable_thinking=True. Qwen3's template
// emits no <think> block in that combination, so the whole of it is this
// one line; the test checks that against the rendered template.
func ChatPrompt(prompt string) string {
	return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// EncodePrompt is Encode of ChatPrompt, which is what the text encoder is
// actually given.
func (t *Tokenizer) EncodePrompt(prompt string) ([]int32, error) {
	return t.Encode(ChatPrompt(prompt))
}

// Encode turns text into token ids. Added tokens written out in the text --
// which is how the chat template delivers them -- are matched as whole
// tokens, matching the HF default of split_special_tokens=False.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	var ids []int32
	for len(text) > 0 {
		at, tok := t.nextSpecial(text)
		if at < 0 {
			more, err := t.encodeOrdinary(text)
			if err != nil {
				return nil, err
			}
			return append(ids, more...), nil
		}
		if at > 0 {
			more, err := t.encodeOrdinary(text[:at])
			if err != nil {
				return nil, err
			}
			ids = append(ids, more...)
		}
		ids = append(ids, t.specialID[tok])
		text = text[at+len(tok):]
	}
	return ids, nil
}

// nextSpecial finds the earliest added token in text, longest match at that
// position, or -1.
func (t *Tokenizer) nextSpecial(text string) (int, string) {
	best, bestTok := -1, ""
	for _, s := range t.specials {
		at := strings.Index(text, s)
		if at < 0 || (best >= 0 && at >= best) {
			continue
		}
		best, bestTok = at, s
	}
	return best, bestTok
}

// encodeOrdinary runs the pre-tokenizer, the byte-level mapping and BPE.
func (t *Tokenizer) encodeOrdinary(text string) ([]int32, error) {
	var ids []int32
	for _, piece := range split(text, t.marks) {
		var mapped strings.Builder
		for i := 0; i < len(piece); i++ {
			mapped.WriteRune(t.byteToRune[piece[i]])
		}
		for _, tok := range t.bpe(mapped.String()) {
			id, ok := t.vocab[tok]
			if !ok {
				return nil, fmt.Errorf("tokenizer: %q is not in the vocabulary", tok)
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// bpe merges the symbols of one pre-token. Each round finds the lowest-rank
// adjacent pair anywhere in the word and merges every non-overlapping
// occurrence of it, which is the reference implementation's loop.
func (t *Tokenizer) bpe(word string) []string {
	syms := make([]string, 0, len(word))
	for _, r := range word {
		syms = append(syms, string(r))
	}
	for len(syms) > 1 {
		bestRank, bestAt := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			rank, ok := t.ranks[syms[i]+"\x00"+syms[i+1]]
			if !ok || (bestAt >= 0 && rank >= bestRank) {
				continue
			}
			bestRank, bestAt = rank, i
		}
		if bestAt < 0 {
			break
		}
		a, b := syms[bestAt], syms[bestAt+1]
		out := syms[:0:0]
		for i := 0; i < len(syms); {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				out = append(out, a+b)
				i += 2
				continue
			}
			out = append(out, syms[i])
			i++
		}
		syms = out
	}
	return syms
}

// Decode maps ids back to text. It exists for tests and for reading a dump;
// nothing in the pipeline decodes.
func (t *Tokenizer) Decode(ids []int32) (string, error) {
	var out []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.tokens) || t.tokens[id] == "" {
			return "", fmt.Errorf("tokenizer: id %d is not in the vocabulary", id)
		}
		tok := t.tokens[id]
		if _, special := t.specialID[tok]; special {
			out = append(out, tok...)
			continue
		}
		for _, r := range tok {
			b, ok := t.runeToByte[r]
			if !ok {
				return "", fmt.Errorf("tokenizer: token %q holds %q, which is not byte-level", tok, r)
			}
			out = append(out, b)
		}
	}
	return string(out), nil
}
