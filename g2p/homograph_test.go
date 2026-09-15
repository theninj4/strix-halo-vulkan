package g2p

import (
	"bufio"
	"os"
	"sort"
	"strings"
	"testing"
)

// TestHomographTagger measures what a dozen rules are worth against spacy's
// neural tagger, over every occurrence of a tag-ambiguous word in the survey
// corpus.
//
// The metric is the *pronunciation*, not the tag: two tags that select the
// same entry are equally right, and most of them do — a noun/verb homograph
// tagged NN and one tagged JJ both fall through to DEFAULT. That is why the
// no-tagger baseline is already high and why the number that matters is the
// gap the rules close, which the test prints per word.
func TestHomographTagger(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	f, err := os.Open("../reference/out/g2p/homographs.txt")
	if err != nil {
		t.Skipf("no homograph table (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	type score struct{ base, ruled, n int }
	per := map[string]*score{}
	var total, base, ruled int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 5 {
			continue
		}
		word, tag, prev, next := cols[0], cols[1], cols[3], cols[4]
		want, ok := l.tagVariant(word, tag)
		if !ok {
			continue
		}
		total++
		key := strings.ToLower(word)
		s := per[key]
		if s == nil {
			s = &score{}
			per[key] = s
		}
		s.n++
		if got, _ := l.tagVariant(word, ""); got == want {
			base++
			s.base++
		}
		if got, _ := l.tagVariant(word, l.HomographTag(prev, word, next)); got == want {
			ruled++
			s.ruled++
		}
	}
	if total == 0 {
		t.Skip("no ambiguous occurrences")
	}
	t.Logf("%d occurrences of %d words: no tagger %d (%.1f%%), rules %d (%.1f%%)",
		total, len(per), base, 100*float64(base)/float64(total),
		ruled, 100*float64(ruled)/float64(total))

	words := make([]string, 0, len(per))
	for w := range per {
		words = append(words, w)
	}
	sort.Slice(words, func(a, b int) bool {
		return per[words[a]].n-per[words[a]].ruled > per[words[b]].n-per[words[b]].ruled
	})
	for _, w := range words {
		s := per[w]
		if s.n == s.ruled {
			break
		}
		t.Logf("  %-14s %4d occurrences, no tagger %3d, rules %3d", w, s.n, s.base, s.ruled)
	}
	if ruled < base {
		t.Errorf("the rules are worse than no tagger at all: %d against %d", ruled, base)
	}
}
