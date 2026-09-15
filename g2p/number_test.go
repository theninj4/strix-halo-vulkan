package g2p

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestNumbers holds the three number spellings to `num2words`, exhaustively
// over 0..10000 and at every scale boundary above it.
//
// Exhaustive rather than sampled because the join rules change at 100 and at
// every power of a thousand, and because the whole point of the table is that
// a rule I got subtly wrong shows up as a specific integer rather than as a
// vague suspicion. `reference/out/g2p/numbers.txt` is a tab-separated
// value/cardinal/ordinal/year table; the rows whose value parses as a float
// rather than an integer are the decimal cases.
func TestNumbers(t *testing.T) {
	f, err := os.Open("../reference/out/g2p/numbers.txt")
	if err != nil {
		t.Skipf("no number table (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	var ints, decimals, wrong int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 4 {
			continue
		}
		value, cardinal, ordinal, year := cols[0], cols[1], cols[2], cols[3]
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			decimals++
			if got := Decimal(value); got != cardinal {
				wrong++
				t.Errorf("Decimal(%s) = %q, num2words gives %q", value, got, cardinal)
			}
			continue
		}
		ints++
		if got := Cardinal(n); got != cardinal {
			wrong++
			if wrong < 20 {
				t.Errorf("Cardinal(%d) = %q, num2words gives %q", n, got, cardinal)
			}
		}
		if ordinal != "" {
			if got := Ordinal(n); got != ordinal {
				wrong++
				if wrong < 20 {
					t.Errorf("Ordinal(%d) = %q, num2words gives %q", n, got, ordinal)
				}
			}
		}
		if year != "" {
			if got := Year(n); got != year {
				wrong++
				if wrong < 20 {
					t.Errorf("Year(%d) = %q, num2words gives %q", n, got, year)
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d integers (cardinal, ordinal and year) and %d decimals, %d wrong",
		ints, decimals, wrong)
}

// TestNumWords covers the split misaki applies to a spelled-out number, which
// is what makes the separators mostly invisible — and "and" the one that is
// not.
func TestNumWords(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"one thousand, two hundred and thirty-four",
			[]string{"one", "thousand", "two", "hundred", "and", "thirty", "four"}},
		{"nineteen oh-one", []string{"nineteen", "oh", "one"}},
		{"minus five", []string{"minus", "five"}},
		{"twenty-first", []string{"twenty", "first"}},
	} {
		got := numWords(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("numWords(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestNumberPath holds `Number` to misaki on the corpus's 21 number tokens,
// which between them cover every branch: cardinals, a comma-grouped million, a
// four-digit year that is *not* one because a currency preceded it, ordinals,
// decimals and all three currency symbols.
//
// Like the word path, the dump records `get_number` in isolation along with
// the three inputs it takes from the tokenizer rather than from the text — the
// currency symbol, whether the token heads its group, and the preprocessor's
// flags — so this too is a pure-function test.
func TestNumberPath(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	var checked, wrong int
	for _, s := range m.Sentences {
		for _, tk := range s.Tokens {
			if tk.NumPS == nil {
				continue
			}
			checked++
			got, rating, ok := l.Number(tk.Word, tk.Currency, tk.IsHead, tk.NumFlags)
			if !ok {
				wrong++
				t.Errorf("%s: %q -> not resolved, misaki gives %q", s.Kind, tk.Word, *tk.NumPS)
				continue
			}
			if Americanize(got) != *tk.NumPS {
				wrong++
				t.Errorf("%s: %q cur=%q head=%v flags=%q -> %q, misaki gives %q",
					s.Kind, tk.Word, tk.Currency, tk.IsHead, tk.NumFlags,
					Americanize(got), *tk.NumPS)
			}
			if tk.NumRating != nil && rating != *tk.NumRating {
				wrong++
				t.Errorf("%s: %q rating %d, misaki says %d", s.Kind, tk.Word, rating, *tk.NumRating)
			}
		}
	}
	t.Logf("%d number tokens, %d wrong", checked, wrong)
}

// TestIsNumber covers the gate in front of the number path, which decides
// whether a token is a number at all — and which treats a leading minus as a
// sign only on the head of a group, because elsewhere it is a hyphen.
func TestIsNumber(t *testing.T) {
	for _, c := range []struct {
		word string
		head bool
		want bool
	}{
		{"1024", true, true},
		{"1,024,000", true, true},
		{"3.50", true, true},
		{"21st", true, true},
		{"1990s", true, true},
		{"-5", true, true},
		{"-5", false, false},
		{"hello", true, false},
		{"10:30", true, false},
		{"3a", true, false},
	} {
		if got := IsNumber(c.word, c.head); got != c.want {
			t.Errorf("IsNumber(%q, %v) = %v, want %v", c.word, c.head, got, c.want)
		}
	}
}

// TestNumberCases is the wide version of TestNumberPath: every interesting
// digit string against every currency, both values of is_head and the five
// flag combinations, 1760 cases in all.
//
// The corpus reaches 21 number tokens and `get_number` has eight branches
// whose conditions turn on the digit count, a leading zero, a decimal point,
// whether the token heads its group and whether a currency preceded it. This
// is where a port goes quietly wrong — "2024" is a year and "$2024" is not,
// "0.5" and ".5" and "3.50" take three different paths — so the cross product
// is dumped rather than reasoned about.
func TestNumberCases(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	f, err := os.Open("../reference/out/g2p/number_cases.txt")
	if err != nil {
		t.Skipf("no number cases (%v); run reference/dump_g2p.py", err)
	}
	defer f.Close()

	var checked, wrong int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 6 {
			continue
		}
		word, currency, head, flags, want := cols[0], cols[1], cols[2] == "1", cols[3], cols[4]
		wantRating, _ := strconv.Atoi(cols[5])
		checked++
		got, rating, ok := l.Number(word, currency, head, flags)
		if !ok {
			wrong++
			if wrong < 15 {
				t.Errorf("Number(%q, %q, %v, %q) -> not resolved, misaki gives %q",
					word, currency, head, flags, want)
			}
			continue
		}
		if Americanize(got) != want || rating != wantRating {
			wrong++
			if wrong < 15 {
				t.Errorf("Number(%q, %q, %v, %q) = %q/%d, misaki gives %q/%d",
					word, currency, head, flags, Americanize(got), rating, want, wantRating)
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d get_number cases, %d wrong", checked, wrong)
}
