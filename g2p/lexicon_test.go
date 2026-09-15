package g2p

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// The oracle comes from reference/dump_g2p.py, which runs misaki's own
// en.G2P over a 24-sentence corpus and records, per token, what
// `Lexicon.get_word` alone made of it and the two context values it saw.
// Regenerate both of these, in this order:
//
//	.venv/bin/python reference/convert_misaki.py
//	.venv/bin/python reference/dump_g2p.py
const (
	refPath = "../reference/out/g2p/manifest.json"
	lexDir  = "../models/misaki"
)

type refToken struct {
	Text        string   `json:"text"`
	Tag         string   `json:"tag"`
	Whitespace  string   `json:"whitespace"`
	Phonemes    *string  `json:"phonemes"`
	Rating      *int     `json:"rating"`
	Word        string   `json:"word"`
	Stress      *float64 `json:"stress"`
	FutureVowel *bool    `json:"future_vowel"`
	FutureTo    bool     `json:"future_to"`
	WordPS      *string  `json:"word_ps"`
	WordRating  *int     `json:"word_rating"`
	Currency    string   `json:"currency"`
	IsHead      bool     `json:"is_head"`
	NumFlags    string   `json:"num_flags"`
	NumPS       *string  `json:"num_ps"`
	NumRating   *int     `json:"num_rating"`
}

type refSentence struct {
	Kind     string     `json:"kind"`
	Text     string     `json:"text"`
	Phonemes string     `json:"phonemes"`
	Tokens   []refToken `json:"tokens"`
}

type refManifest struct {
	British   bool          `json:"british"`
	Sentences []refSentence `json:"sentences"`
	Survey    struct {
		Tokens      int                `json:"tokens"`
		Paths       map[string]int     `json:"paths"`
		PathsPct    map[string]float64 `json:"paths_pct"`
		ResolvedPct float64            `json:"resolved_pct"`
		Tagger      struct {
			NeedsATagger    int      `json:"needs_a_tagger"`
			NeedsATaggerPct float64  `json:"needs_a_tagger_pct"`
			ContextOnly     int      `json:"context_only"`
			DistinctWords   []string `json:"distinct_words"`
		} `json:"tagger"`
	} `json:"survey"`
}

func loadRef(t *testing.T) *refManifest {
	t.Helper()
	buf, err := os.ReadFile(refPath)
	if err != nil {
		t.Skipf("no reference dump at %s (%v); run reference/dump_g2p.py", refPath, err)
	}
	var m refManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadLexicon(t *testing.T, british bool) *Lexicon {
	t.Helper()
	name := "us_gold.txt"
	if british {
		name = "gb_gold.txt"
	}
	if _, err := os.Stat(filepath.Join(lexDir, name)); err != nil {
		t.Skipf("no lexicon in %s (%v); run reference/convert_misaki.py", lexDir, err)
	}
	t0 := time.Now()
	l, err := Load(lexDir, british)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded %d gold and %d silver entries in %.0f ms",
		len(l.gold), len(l.silver), float64(time.Since(t0).Microseconds())/1000)
	return l
}

// TestWordPath holds the port to misaki token by token, on exactly the
// function it implements.
//
// The dump records `get_word`'s output separately from the token's final
// phonemes, along with the tag, the capitalisation stress and the context the
// lookup was given — so this is a pure-function test with no tokenizer, no
// number expansion and no espeak in it. Every token misaki resolved this way
// must come out identical; a token it resolved some other way is counted and
// reported rather than asserted on, because that is the part T5 has not
// built yet.
func TestWordPath(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)

	var checked, wrong, other int
	for _, s := range m.Sentences {
		for _, tk := range s.Tokens {
			if tk.WordPS == nil {
				other++
				continue
			}
			ctx := &Context{FutureVowel: tk.FutureVowel, FutureTo: tk.FutureTo}
			got, _, ok := l.Word(tk.Word, tk.Tag, tk.Stress, ctx)
			checked++
			if !ok {
				wrong++
				t.Errorf("%s: %q [%s] -> not resolved, misaki gives %q",
					s.Kind, tk.Word, tk.Tag, *tk.WordPS)
				continue
			}
			if Americanize(got) != *tk.WordPS {
				wrong++
				t.Errorf("%s: %q [%s] fv=%v stress=%v -> %q, misaki gives %q",
					s.Kind, tk.Word, tk.Tag, fmtBool(tk.FutureVowel), fmtF64(tk.Stress),
					Americanize(got), *tk.WordPS)
			}
		}
	}
	t.Logf("%d tokens on the word path, %d wrong; %d tokens took another path "+
		"(numbers, punctuation, the tokenizer or espeak)", checked, wrong, other)
}

// TestRating checks that the port agrees with misaki about *which* dictionary
// answered, not only about the answer.
//
// It is a separate test because it fails differently: a wrong rating with the
// right phonemes means the gold/silver order or the proper-noun path is off,
// and that is invisible in the output until a word appears that the two
// dictionaries disagree about.
func TestRating(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	var checked, wrong int
	for _, s := range m.Sentences {
		for _, tk := range s.Tokens {
			if tk.WordPS == nil || tk.WordRating == nil {
				continue
			}
			ctx := &Context{FutureVowel: tk.FutureVowel, FutureTo: tk.FutureTo}
			_, rating, ok := l.Word(tk.Word, tk.Tag, tk.Stress, ctx)
			if !ok {
				continue
			}
			checked++
			if rating != *tk.WordRating {
				wrong++
				t.Errorf("%s: %q rating %d, misaki says %d", s.Kind, tk.Word, rating, *tk.WordRating)
			}
		}
	}
	t.Logf("%d ratings checked, %d wrong", checked, wrong)
}

// TestContextNext checks the reverse walk that produces a lookup's context,
// against the contexts the dump recorded.
//
// This is the half of the tri-state that is not in the dictionary: given a
// token's phonemes, what does the token to its *left* see. It is checked on
// the dump's own final phonemes so that a disagreement here cannot be an
// artifact of a lookup this package got wrong.
func TestContextNext(t *testing.T) {
	m := loadRef(t)
	for _, s := range m.Sentences {
		ctx := Context{}
		want := make([]Context, len(s.Tokens))
		for i := len(s.Tokens) - 1; i >= 0; i-- {
			want[i] = ctx
			tk := s.Tokens[i]
			ps := ""
			if tk.Phonemes != nil {
				ps = *tk.Phonemes
			}
			ctx = ctx.Next(ps, tk.Text, tk.Tag)
		}
		for i, tk := range s.Tokens {
			if fmtBool(want[i].FutureVowel) != fmtBool(tk.FutureVowel) {
				t.Errorf("%s: before %q future_vowel %s, misaki says %s",
					s.Kind, tk.Text, fmtBool(want[i].FutureVowel), fmtBool(tk.FutureVowel))
			}
			if want[i].FutureTo != tk.FutureTo {
				t.Errorf("%s: before %q future_to %v, misaki says %v",
					s.Kind, tk.Text, want[i].FutureTo, tk.FutureTo)
			}
		}
	}
}

// TestApplyStress covers the rewrite rules on their own, including the two
// branches the corpus does not reach.
//
// The interesting one is restress: a stress mark belongs immediately before
// its vowel, so promoting an unmarked word does not prepend a mark, it
// inserts one — `strɪŋ` takes it before the ɪ, not before the s.
func TestApplyStress(t *testing.T) {
	for _, c := range []struct {
		ps     string
		stress *float64
		want   string
	}{
		{"hˈɛlO", nil, "hˈɛlO"},
		{"hˈɛlO", f64(-2), "hɛlO"},
		{"hˈɛlO", f64(-1), "hˌɛlO"},
		{"hˈɛlO", f64(-0.5), "hˌɛlO"},
		{"hˌɛlO", f64(1), "hˈɛlO"},
		{"strɪŋ", f64(0.5), "strˌɪŋ"},
		{"strɪŋ", f64(2), "strˈɪŋ"},
		{"ŋ", f64(2), "ŋ"}, // no vowel: nothing to mark
		{"hˈɛlO", f64(2), "hˈɛlO"},
	} {
		if got := applyStress(c.ps, c.stress); got != c.want {
			t.Errorf("applyStress(%q, %s) = %q, want %q", c.ps, fmtF64(c.stress), got, c.want)
		}
	}
}

// TestNNP covers the spelling-out path, which is what every acronym and every
// unknown proper noun in the corpus goes through.
func TestNNP(t *testing.T) {
	m := loadRef(t)
	l := loadLexicon(t, m.British)
	for _, c := range []struct{ word, want string }{
		{"GPU", "ʤˌipˌijˈu"},
		{"CPU", "sˌipˌijˈu"},
		{"GLSL", "ʤˌiˌɛlˌɛsˈɛl"},
	} {
		got, rating, ok := l.NNP(c.word)
		if !ok || got != c.want {
			t.Errorf("NNP(%q) = %q,%d,%v, want %q", c.word, got, rating, ok, c.want)
		}
	}
}

// TestSurvey re-states the measurement the dump made, so that the numbers
// SPEECH.md quotes are in the test output rather than only in a document.
func TestSurvey(t *testing.T) {
	m := loadRef(t)
	s := m.Survey
	if s.Tokens == 0 {
		t.Skip("no survey in the dump")
	}
	t.Logf("%d tokens of running English:", s.Tokens)
	for _, k := range []string{"gold", "silver", "stems", "stem's", "stemed", "stemes", "steming", "unresolved"} {
		if n, ok := s.Paths[k]; ok {
			t.Logf("  %-12s %7d  %5.2f%%", k, n, s.PathsPct[k])
		}
	}
	t.Logf("  resolved by the dictionary and the three suffix rules: %.2f%%", s.ResolvedPct)
	t.Logf("  a part-of-speech tagger decides %d tokens (%.2f%%) over %d distinct words; "+
		"%d more look tag-conditioned and are decided by context",
		s.Tagger.NeedsATagger, s.Tagger.NeedsATaggerPct, len(s.Tagger.DistinctWords), s.Tagger.ContextOnly)
}

func fmtBool(b *bool) string {
	if b == nil {
		return "none"
	}
	return strconv.FormatBool(*b)
}

func fmtF64(f *float64) string {
	if f == nil {
		return "none"
	}
	return strconv.FormatFloat(*f, 'g', -1, 64)
}
