package g2p

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// TestPhonemize measures the whole pipeline — tokenizer, subtokenizer, merge
// loop, dictionary and numbers — against misaki's output for the reference
// corpus.
//
// It reports rather than asserts on most of it, on purpose. Everything below
// the tokenizer is held to string equality by its own test; what this one
// measures is the part that is an *approximation*: spacy's tagger is replaced
// by a dozen rules, and there is no espeak fallback, so a sentence can differ
// for a reason that is known and quantified rather than for a bug. The number
// is what says whether the next piece of T5 is the tagger or the fallback.
func TestPhonemize(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	if e, err := Open(m.British); err == nil {
		defer e.Close()
		l.Fallback = e
	} else {
		t.Logf("no espeak fallback: %v", err)
	}

	var exact, total int
	for _, s := range m.Sentences {
		total++
		got, unknown := l.Phonemize(s.Text)
		if got == s.Phonemes {
			exact++
			continue
		}
		t.Logf("%-20s %d unknown\n     go %s\n  misaki %s", s.Kind, unknown, got, s.Phonemes)
	}
	t.Logf("%d of %d sentences exact", exact, total)
	if l.Fallback != nil && exact != total {
		// With the dictionary, the numbers, the tokenizer, the homograph
		// rules and the fallback all in place, the designed corpus is exact —
		// every branch of misaki's English G2P fires in it and every one is
		// reproduced. A regression here is a bug, not an approximation.
		t.Errorf("%d of %d corpus sentences differ", total-exact, total)
	}
}

// TestPhonemizeTokens measures agreement token by token rather than sentence by
// sentence, which is the number that says how much of a difference any one
// sentence's difference is.
func TestPhonemizeTokens(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	if e, err := Open(m.British); err == nil {
		defer e.Close()
		l.Fallback = e
	}
	var same, total int
	for _, s := range m.Sentences {
		got, _ := l.Phonemize(s.Text)
		g, w := strings.Fields(got), strings.Fields(s.Phonemes)
		for i := 0; i < len(g) && i < len(w); i++ {
			total++
			if g[i] == w[i] {
				same++
			}
		}
		total += abs(len(g) - len(w))
	}
	t.Logf("%d of %d whitespace-separated phoneme words agree (%.1f%%)",
		same, total, 100*float64(same)/float64(total))
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// TestPhonemizeWild is the generalisation measurement: 400 sentences lifted
// out of the repository's own prose, which nobody chose and which nothing was
// fitted to.
//
// The curated corpus is a designed test — every branch fires and the port was
// aimed at it — so passing it says the pieces work, not that they hold up.
// This says how the tokenizer and the homograph rules do on whatever real
// English happens to contain, and it is a hard case: this prose is a tenth
// acronyms, with hyphenation, section numbers and units all over it.
func TestPhonemizeWild(t *testing.T) {
	f, err := os.Open("../reference/out/g2p/wild.txt")
	if err != nil {
		t.Skipf("no wild corpus (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	if e, err := Open(m.British); err == nil {
		defer e.Close()
		l.Fallback = e
	} else {
		t.Logf("no espeak fallback: %v", err)
	}

	var exact, total, same, words int
	var shown int
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		text, want, ok := strings.Cut(sc.Text(), "\t")
		if !ok {
			continue
		}
		total++
		got, _ := l.Phonemize(text)
		if got == want {
			exact++
		} else if shown < 5 {
			shown++
			t.Logf("differs:\n    text %s\n      go %s\n  misaki %s", text, got, want)
		}
		g, w := strings.Fields(got), strings.Fields(want)
		for i := 0; i < len(g) && i < len(w); i++ {
			words++
			if g[i] == w[i] {
				same++
			}
		}
		words += abs(len(g) - len(w))
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	sentencePct := 100 * float64(exact) / float64(total)
	wordPct := 100 * float64(same) / float64(words)
	t.Logf("%d sentences: %d exact (%.1f%%); %d of %d phoneme words agree (%.1f%%)",
		total, exact, sentencePct, same, words, wordPct)
	// Floors, not targets. They are set a little below what was measured when
	// this was written (69.0% and 92.4%) so that a real regression fails and
	// ordinary drift in the corpus does not.
	if sentencePct < 65 {
		t.Errorf("sentence agreement fell to %.1f%%", sentencePct)
	}
	if wordPct < 90 {
		t.Errorf("word agreement fell to %.1f%%", wordPct)
	}
}
