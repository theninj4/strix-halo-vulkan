package ocr

// ERNIE-4.5's tokenizer as PaddleOCR-VL-1.6 ships it (OCR.md O1): a
// Llama-style BPE over characters with byte fallback, read from
// tokenizer.json.
//
// It is not the byte-level BPE `zimage/tokenizer` implements for Qwen, and
// the differences are each a place a port that assumed the familiar one
// would be silently wrong:
//
//   - **The only normalisation is ' ' -> '▁'.** No prefix space is added
//     (the file has no Prepend step, whatever `legacy: true` suggests), and
//     there is no pre-tokeniser: a whole text segment is one BPE word.
//   - **Characters outside the vocabulary become their UTF-8 bytes**, as the
//     tokens <0x00>..<0xFF>, and decoding reassembles a run of them, one
//     U+FFFD a byte if the run is not valid UTF-8.
//   - **Added tokens are split out before anything else**, leftmost-longest,
//     on the raw text. There are 1,041 of them, and they are not only the
//     specials: **the digits 0-9 are added tokens**, so every digit is its
//     own token; so are the OTSL table tags (<fcel>, <nl>, ...) and the
//     1,001 <|LOC_N|> coordinates spotting answers in.
//   - **Only 22 of the added tokens are special.** skip_special_tokens drops
//     those; <|LOC_N|> and the OTSL tags survive it, which is what the
//     table and spotting post-processing read.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer encodes prompts and decodes generations.
type Tokenizer struct {
	vocab   map[string]int32
	pieces  []string // id -> token string, added tokens included
	special []bool   // id -> an added token with special: true
	merges  map[[2]int32]merge
	byteID  [256]int32 // <0xNN>
	unk     int32

	// added tokens by their first byte, longest first
	added map[byte][]addedToken
}

type merge struct{ rank, id int32 }

type addedToken struct {
	content string
	id      int32
}

// LoadTokenizer reads dir/tokenizer.json.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var f struct {
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		Normalizer struct {
			Type        string `json:"type"`
			Normalizers []struct {
				Type    string            `json:"type"`
				Pattern map[string]string `json:"pattern"`
				Content string            `json:"content"`
			} `json:"normalizers"`
		} `json:"normalizer"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		Model        struct {
			Type         string           `json:"type"`
			UnkToken     string           `json:"unk_token"`
			ByteFallback bool             `json:"byte_fallback"`
			Vocab        map[string]int32 `json:"vocab"`
			Merges       [][2]string      `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("ocr: tokenizer.json: %w", err)
	}
	// Hold the file to the scheme this port implements rather than trusting
	// that a different checkpoint's tokenizer is the same kind.
	n := f.Normalizer.Normalizers
	if f.Model.Type != "BPE" || !f.Model.ByteFallback || f.Normalizer.Type != "Sequence" || len(n) != 1 ||
		n[0].Type != "Replace" || n[0].Pattern["String"] != " " || n[0].Content != "▁" ||
		(len(f.PreTokenizer) > 0 && string(f.PreTokenizer) != "null") {
		return nil, fmt.Errorf("ocr: tokenizer.json is not the byte-fallback BPE this port reads")
	}

	t := &Tokenizer{vocab: f.Model.Vocab, merges: make(map[[2]int32]merge, len(f.Model.Merges)), added: map[byte][]addedToken{}}
	size := int32(0)
	for _, id := range f.Model.Vocab {
		size = max(size, id+1)
	}
	for _, a := range f.AddedTokens {
		size = max(size, a.ID+1)
	}
	t.pieces = make([]string, size)
	t.special = make([]bool, size)
	for s, id := range f.Model.Vocab {
		t.pieces[id] = s
	}
	for _, a := range f.AddedTokens {
		t.pieces[a.ID] = a.Content
		t.special[a.ID] = a.Special
		if a.Content != "" {
			t.added[a.Content[0]] = append(t.added[a.Content[0]], addedToken{a.Content, a.ID})
		}
	}
	for _, l := range t.added {
		sort.SliceStable(l, func(i, j int) bool { return len(l[i].content) > len(l[j].content) })
	}
	for b := 0; b < 256; b++ {
		id, ok := t.vocab[fmt.Sprintf("<0x%02X>", b)]
		if !ok {
			return nil, fmt.Errorf("ocr: tokenizer has no byte token <0x%02X>", b)
		}
		t.byteID[b] = id
	}
	unk, ok := t.vocab[f.Model.UnkToken]
	if !ok {
		return nil, fmt.Errorf("ocr: tokenizer has no unk token %q", f.Model.UnkToken)
	}
	t.unk = unk
	for rank, m := range f.Model.Merges {
		a, okA := t.vocab[m[0]]
		b, okB := t.vocab[m[1]]
		id, okC := t.vocab[m[0]+m[1]]
		if !okA || !okB || !okC {
			return nil, fmt.Errorf("ocr: merge %q %q is outside the vocabulary", m[0], m[1])
		}
		if _, dup := t.merges[[2]int32{a, b}]; !dup {
			t.merges[[2]int32{a, b}] = merge{int32(rank), id}
		}
	}
	return t, nil
}

// VocabSize is one past the largest id.
func (t *Tokenizer) VocabSize() int { return len(t.pieces) }

// ID is a token's id, for the template's markers; ok is false if absent.
func (t *Tokenizer) ID(token string) (int32, bool) {
	for _, a := range t.added[firstByte(token)] {
		if a.content == token {
			return a.id, true
		}
	}
	id, ok := t.vocab[token]
	return id, ok
}

func firstByte(s string) byte {
	if s == "" {
		return 0
	}
	return s[0]
}

// Special reports whether id is one of the 22 special added tokens.
func (t *Tokenizer) Special(id int32) bool {
	return id >= 0 && int(id) < len(t.special) && t.special[id]
}

// Encode is `tokenizer.encode(text, add_special_tokens=False)`.
func (t *Tokenizer) Encode(text string) []int32 {
	var out []int32
	start := 0
	for i := 0; i < len(text); {
		if a, ok := t.addedAt(text, i); ok {
			out = t.bpe(out, text[start:i])
			out = append(out, a.id)
			i += len(a.content)
			start = i
			continue
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}
	return t.bpe(out, text[start:])
}

// addedAt is the longest added token starting at byte i.
func (t *Tokenizer) addedAt(text string, i int) (addedToken, bool) {
	for _, a := range t.added[text[i]] {
		if strings.HasPrefix(text[i:], a.content) {
			return a, true
		}
	}
	return addedToken{}, false
}

// bpe appends the ids of one segment with no added token in it.
func (t *Tokenizer) bpe(out []int32, seg string) []int32 {
	if seg == "" {
		return out
	}
	seg = strings.ReplaceAll(seg, " ", "▁")
	var sym []int32
	for _, r := range seg {
		if id, ok := t.vocab[string(r)]; ok {
			sym = append(sym, id)
			continue
		}
		var buf [4]byte
		for _, b := range buf[:utf8.EncodeRune(buf[:], r)] {
			sym = append(sym, t.byteID[b])
		}
	}
	// Merge the lowest-ranked adjacent pair, leftmost on a tie, until none
	// is in the table. Quadratic, and the prompts are a few dozen symbols.
	for len(sym) > 1 {
		best, at := merge{rank: -1}, -1
		for i := 0; i+1 < len(sym); i++ {
			m, ok := t.merges[[2]int32{sym[i], sym[i+1]}]
			if ok && (at < 0 || m.rank < best.rank) {
				best, at = m, i
			}
		}
		if at < 0 {
			break
		}
		sym[at] = best.id
		sym = append(sym[:at+1], sym[at+2:]...)
	}
	return append(out, sym...)
}

// Decode is `tokenizer.decode(ids, skip_special_tokens=skip)`: '▁' back to
// a space, byte-fallback runs reassembled, specials dropped when skip is set.
// An id outside the vocabulary decodes to nothing, as HF's does.
func (t *Tokenizer) Decode(ids []int32, skip bool) string {
	var sb strings.Builder
	var run []byte
	flush := func() {
		if len(run) == 0 {
			return
		}
		if utf8.Valid(run) {
			sb.Write(run)
		} else {
			for range run {
				sb.WriteString("�")
			}
		}
		run = run[:0]
	}
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.pieces) || (skip && t.special[id]) {
			continue
		}
		p := t.pieces[id]
		if b, ok := byteToken(p); ok {
			run = append(run, b)
			continue
		}
		flush()
		sb.WriteString(strings.ReplaceAll(p, "▁", " "))
	}
	flush()
	return sb.String()
}

// byteToken parses "<0xNN>".
func byteToken(p string) (byte, bool) {
	if len(p) != 6 || p[:3] != "<0x" || p[5] != '>' {
		return 0, false
	}
	var v byte
	for _, c := range p[3:5] {
		switch {
		case c >= '0' && c <= '9':
			v = v<<4 | byte(c-'0')
		case c >= 'A' && c <= 'F':
			v = v<<4 | byte(c-'A'+10)
		default:
			return 0, false
		}
	}
	return v, true
}
