package g2p

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// TestFromEspeak holds the rewrite to misaki on the third and fourth columns
// of the espeak table — phonemizer's IPA in, kokoro's phonemes out.
//
// It is a separate test from the binding on purpose: this half is pure string
// rewriting and runs on any machine, while the half that calls espeak needs
// the library. When the two are tested together a mismatch says only "the
// fallback is wrong"; apart, it says which of the three layers moved.
func TestFromEspeak(t *testing.T) {
	f, err := os.Open("../reference/out/g2p/espeak.txt")
	if err != nil {
		t.Skipf("no espeak table (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	var checked, wrong int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 4 {
			continue
		}
		word, phonemizer, want := cols[0], cols[2], cols[3]
		checked++
		if got := FromEspeak(phonemizer, false); got != want {
			wrong++
			t.Errorf("FromEspeak(%q) = %q, misaki gives %q  [%s]", phonemizer, got, want, word)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d words, %d wrong", checked, wrong)
}

// TestEspeak exercises the binding itself against the table's second column —
// espeak's own output, before phonemizer or misaki touch it.
//
// It skips where the library is absent, which is the same thing the engine
// does: the fallback is optional and its absence costs 8.8% of tokens rather
// than breaking anything.
func TestEspeak(t *testing.T) {
	e, err := Open(false)
	if err != nil {
		t.Skipf("no espeak: %v", err)
	}
	defer e.Close()
	t.Logf("library %s, data %s", e.Library, e.Data)

	f, err := os.Open("../reference/out/g2p/espeak.txt")
	if err != nil {
		t.Skipf("no espeak table (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	var checked, rawWrong, wrong int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 4 {
			continue
		}
		word, raw, want := cols[0], cols[1], cols[3]
		checked++
		// The tie is U+0361 in the table's raw column and `^` here, which is
		// phonemizer's substitution and the one difference between them.
		if got := e.Raw(word); got != strings.ReplaceAll(raw, "͡", "^") {
			rawWrong++
			t.Errorf("Raw(%q) = %q, espeak gives %q", word, got, raw)
		}
		got, rating, ok := e.Phonemes(word)
		if !ok || got != want || rating != RatingEspeak {
			wrong++
			t.Errorf("Phonemes(%q) = %q/%d/%v, misaki gives %q", word, got, rating, ok, want)
		}
	}
	t.Logf("%d words: %d raw disagreements, %d after the rewrite", checked, rawWrong, wrong)
}
