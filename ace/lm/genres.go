package lm

// The `genres:` value's vocabulary (MUSIC.md A12): upstream's
// _get_allowed_genres_tokens over acestep/genres_vocab.txt, which sample
// mode reaches on every run (genres are not skipped there) and the thinking
// path only if the model writes "genres:" after its caption.
//
// Upstream holds the 178k lower-cased genres in a character trie. Here they
// are one sorted list: the genres under a prefix are a contiguous run of
// it, so a trie node is a binary search and its children are the distinct
// runes after the prefix in that run. The caption-matched trie upstream
// prefers is never built (_extract_caption_genres has no caller), so the
// full vocabulary is always the one used.

import (
	"bufio"
	"os"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// GenresFile is where upstream keeps the vocabulary, under the models root.
const GenresFile = "ACE-Step-1.5-src/acestep/genres_vocab.txt"

type genreVocab struct {
	list []string // sorted, unique, lower-cased
	// byFirst maps a lower-cased first character (and a space-prefixed
	// token's first non-space one) to the tokens that start with it:
	// _precompute_char_token_mapping's _char_to_tokens.
	byFirst map[string][]int32
	// text is a token's decoded text lower-cased and right-stripped, or
	// " " when it is all whitespace; absent when it decodes to nothing.
	text map[int32]string
}

// loadGenres reads the vocabulary as _load_genres_vocab does and builds the
// token tables over the whole vocabulary.
func loadGenres(v *fsmVocab, path string) (*genreVocab, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	seen := map[string]bool{}
	g := &genreVocab{byFirst: map[string][]int32{}, text: map[int32]string{}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := pyStrip(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if l := pyLower(line); !seen[l] {
			seen[l] = true
			g.list = append(g.list, l)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	sort.Strings(g.list) // byte order is code-point order
	for id := int32(0); int(id) < v.vocab; id++ {
		s := v.decode(id)
		if s == "" {
			continue
		}
		low := pyLower(s)
		if pyStrip(low) != "" {
			g.text[id] = strings.TrimRightFunc(low, isPySpace)
		} else {
			g.text[id] = " "
		}
		first, _ := utf8.DecodeRuneInString(s)
		g.byFirst[pyLower(string(first))] = append(g.byFirst[pyLower(string(first))], id)
		if st := strings.TrimLeftFunc(s, isPySpace); st != "" && st != s {
			r, _ := utf8.DecodeRuneInString(st)
			g.byFirst[pyLower(string(r))] = append(g.byFirst[pyLower(string(r))], id)
		}
	}
	return g, nil
}

// has reports a trie node for prefix: some genre starts with it.
func (g *genreVocab) has(prefix string) bool {
	i := sort.SearchStrings(g.list, prefix)
	return i < len(g.list) && strings.HasPrefix(g.list[i], prefix)
}

// children is a node's next characters and whether the prefix is itself a
// genre (its `_end`).
func (g *genreVocab) children(prefix string) (next map[string]bool, complete bool) {
	next = map[string]bool{}
	i := sort.SearchStrings(g.list, prefix)
	if i < len(g.list) && g.list[i] == prefix {
		complete = true
		i++
	}
	for i < len(g.list) && strings.HasPrefix(g.list[i], prefix) {
		r, _ := utf8.DecodeRuneInString(g.list[i][len(prefix):])
		next[string(r)] = true
		q := prefix + string(r)
		i += sort.Search(len(g.list)-i, func(j int) bool {
			s := g.list[i+j]
			return s >= q && !strings.HasPrefix(s, q)
		})
	}
	return next, complete
}

// allowed is _get_allowed_genres_tokens for the value so far; nil means
// no continuation, and the caller forces the newline.
func (g *genreVocab) allowed(acc string, newline int32) []int32 {
	prefix := pyStrip(pyLower(acc))
	if !g.has(prefix) {
		return []int32{newline}
	}
	next, complete := g.children(prefix)
	var out []int32
	seen := map[int32]bool{}
	for c := range next {
		for _, id := range g.byFirst[c] {
			if seen[id] {
				continue
			}
			seen[id] = true
			t := g.text[id]
			if t == "" || pyStrip(t) == "" {
				if next[" "] || next[","] {
					out = append(out, id)
				}
				continue
			}
			if g.has(prefix + t) {
				out = append(out, id)
			}
		}
	}
	if complete && !slices.Contains(out, newline) {
		out = append(out, newline)
	}
	return out
}

// pyLower is str.lower(): Go's per-rune mapping plus Python's two
// context-free differences, the dotted capital I and a final sigma.
func pyLower(s string) string {
	if isASCII(s) {
		return strings.ToLower(s)
	}
	rs := []rune(s)
	var b strings.Builder
	for i, r := range rs {
		switch {
		case r == 'İ':
			b.WriteString("i̇")
		case r == 'Σ' && finalSigma(rs, i):
			b.WriteRune('ς')
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// finalSigma is Unicode's Final_Sigma: a cased letter before (skipping
// case-ignorables) and none after.
func finalSigma(rs []rune, i int) bool {
	cased := func(r rune) bool { return unicode.IsUpper(r) || unicode.IsLower(r) || unicode.IsTitle(r) }
	ignorable := func(r rune) bool {
		return unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf, unicode.Lm, unicode.Sk) || r == '\'' || r == '.' || r == ':'
	}
	before := false
	for j := i - 1; j >= 0; j-- {
		if ignorable(rs[j]) {
			continue
		}
		before = cased(rs[j])
		break
	}
	if !before {
		return false
	}
	for j := i + 1; j < len(rs); j++ {
		if ignorable(rs[j]) {
			continue
		}
		return !cased(rs[j])
	}
	return true
}
