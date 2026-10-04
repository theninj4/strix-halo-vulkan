// Package gemma4 runs Gemma 4's 26B-A4B text tower, the backbone of Rune v3,
// the decision model that replaced Kev (`research/rune-vertical.md`).
package gemma4

// Gemma 4's tokenizer, read from tokenizer.json (R2).
//
// It is the Llama-style scheme `ocr/tokenizer.go` reads for ERNIE (a BPE
// over characters with byte fallback, ' ' normalised to '▁', added tokens
// split out first), with two differences that each matter here:
//
//   - **The file has a pre-tokenizer, and it does nothing.** It splits on
//     " ", but it runs after the normaliser has already turned every space
//     into '▁', so a whole segment between added tokens is still one BPE
//     word. A decisions state is one segment, often thousands of
//     characters, so the merge loop is a heap over a linked list
//     (O(n log n)), not ocr's rescan per merge (quadratic, fine for a
//     few dozen symbols).
//   - **Merges cross the state/question boundary.** The user turn is one
//     word, so "...\n\nQUESTION:" may merge across the join, and a
//     question's tokens are not the state's tokens plus its own. The shared
//     prefix is therefore found on tokens, as surogate does.

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer encodes prompts and decodes ids.
type Tokenizer struct {
	vocab   map[string]int32
	pieces  []string // id -> token string, added tokens included
	special []bool
	merges  map[[2]int32]merge
	byteID  [256]int32 // <0xNN>
	unk     int32

	added map[byte][]addedToken // by first byte, longest first
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
	type replace struct {
		Type    string            `json:"type"`
		Pattern map[string]string `json:"pattern"`
		Content string            `json:"content"`
	}
	var f struct {
		AddedTokens []struct {
			ID         int32  `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
			Lstrip     bool   `json:"lstrip"`
			Rstrip     bool   `json:"rstrip"`
		} `json:"added_tokens"`
		Normalizer   replace `json:"normalizer"`
		PreTokenizer struct {
			Type     string            `json:"type"`
			Pattern  map[string]string `json:"pattern"`
			Behavior string            `json:"behavior"`
		} `json:"pre_tokenizer"`
		Model struct {
			Type         string           `json:"type"`
			UnkToken     string           `json:"unk_token"`
			ByteFallback bool             `json:"byte_fallback"`
			IgnoreMerges bool             `json:"ignore_merges"`
			Vocab        map[string]int32 `json:"vocab"`
			Merges       [][2]string      `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("gemma4: tokenizer.json: %w", err)
	}
	// Hold the file to the scheme this port implements rather than trusting
	// that another checkpoint's tokenizer is the same kind.
	n, p := f.Normalizer, f.PreTokenizer
	if f.Model.Type != "BPE" || !f.Model.ByteFallback || f.Model.IgnoreMerges ||
		n.Type != "Replace" || n.Pattern["String"] != " " || n.Content != "▁" ||
		p.Type != "Split" || p.Pattern["String"] != " " {
		return nil, fmt.Errorf("gemma4: tokenizer.json is not the byte-fallback BPE this port reads")
	}
	for _, a := range f.AddedTokens {
		if a.Normalized || a.Lstrip || a.Rstrip {
			return nil, fmt.Errorf("gemma4: added token %q is normalized or stripped, which this port does not do", a.Content)
		}
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
	for b := range 256 {
		id, ok := t.vocab[fmt.Sprintf("<0x%02X>", b)]
		if !ok {
			return nil, fmt.Errorf("gemma4: tokenizer has no byte token <0x%02X>", b)
		}
		t.byteID[b] = id
	}
	unk, ok := t.vocab[f.Model.UnkToken]
	if !ok {
		return nil, fmt.Errorf("gemma4: tokenizer has no unk token %q", f.Model.UnkToken)
	}
	t.unk = unk
	for rank, m := range f.Model.Merges {
		a, okA := t.vocab[m[0]]
		b, okB := t.vocab[m[1]]
		id, okC := t.vocab[m[0]+m[1]]
		if !okA || !okB || !okC {
			return nil, fmt.Errorf("gemma4: merge %q %q is outside the vocabulary", m[0], m[1])
		}
		if _, dup := t.merges[[2]int32{a, b}]; !dup {
			t.merges[[2]int32{a, b}] = merge{int32(rank), id}
		}
	}
	return t, nil
}

// VocabSize is one past the largest id.
func (t *Tokenizer) VocabSize() int { return len(t.pieces) }

// ID is a token's id; ok is false if absent.
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

// candidate is a mergeable adjacent pair: the symbol at pos and its right
// neighbour, which held ids a and b when it was pushed.
type candidate struct {
	rank, id int32
	pos      int32
	a, b     int32
}

type candidates []candidate

func (h candidates) Len() int { return len(h) }
func (h candidates) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos // leftmost on a tie, as HF's Word::merge_all
}
func (h candidates) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candidates) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *candidates) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// bpe appends the ids of one segment with no added token in it: merge the
// lowest-ranked adjacent pair, leftmost on a tie, until none is in the
// table. A symbol keeps its original position as its key, so "leftmost" is
// the same order the quadratic loop uses.
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
	n := len(sym)
	next := make([]int32, n) // -1 past the end
	prev := make([]int32, n) // -1 before the start
	for i := range n {
		next[i], prev[i] = int32(i+1), int32(i-1)
	}
	next[n-1] = -1
	h := &candidates{}
	push := func(i int32) {
		if i < 0 || next[i] < 0 {
			return
		}
		j := next[i]
		if m, ok := t.merges[[2]int32{sym[i], sym[j]}]; ok {
			heap.Push(h, candidate{m.rank, m.id, i, sym[i], sym[j]})
		}
	}
	for i := range n - 1 {
		push(int32(i))
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(candidate)
		j := next[c.pos]
		if sym[c.pos] != c.a || j < 0 || sym[j] != c.b {
			continue // stale: one side has merged since
		}
		sym[c.pos] = c.id
		sym[j] = -1
		next[c.pos] = next[j]
		if next[j] >= 0 {
			prev[next[j]] = c.pos
		}
		push(prev[c.pos])
		push(c.pos)
	}
	for i := int32(0); i >= 0; i = next[i] {
		out = append(out, sym[i])
	}
	return out
}

// Decode is `tokenizer.decode(ids, skip_special_tokens=skip)`: '▁' back to
// a space, byte-fallback runs reassembled (one U+FFFD a byte if a run is not
// UTF-8), specials dropped when skip is set.
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
