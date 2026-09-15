package g2p

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// TestSubtokenize holds the scanner to misaki's regex on every word shape the
// dump covers: hyphenation, camel case, initialisms, comma-grouped digits,
// internal apostrophes, accented letters and the degenerate inputs.
//
// The order of the alternatives is the behaviour, so this is string equality
// on the whole split, not a spot check — and it is the only way to be sure,
// because two of the alternatives are lookaheads RE2 cannot express and the
// port is a scanner rather than a translation.
func TestSubtokenize(t *testing.T) {
	f, err := os.Open("../reference/out/g2p/subtokens.txt")
	if err != nil {
		t.Skipf("no subtoken table (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	var checked, wrong int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		word, want, ok := strings.Cut(sc.Text(), "\t")
		if !ok {
			continue
		}
		checked++
		got := strings.Join(Subtokenize(word), "|")
		if got != want {
			wrong++
			t.Errorf("Subtokenize(%q) = %q, misaki gives %q", word, got, want)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d words, %d wrong", checked, wrong)
}
