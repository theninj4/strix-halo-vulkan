package g2p

import (
	"strings"
	"unicode"
)

// e2m is misaki's `EspeakFallback.E2M`: espeak's IPA rewritten into kokoro's
// 178-symbol alphabet.
//
// The order is load-bearing and is misaki's — longest key first, and within a
// length, the order the table is written in. `e^ɪ` has to run before `e`, or
// every diphthong would lose its first half; `ə^l` before `l`; `ʲo` and `ʲə`
// before the bare `ʲ` that is deleted.
//
// `^` is the tie character phonemizer asks espeak for. It marks the two halves
// of one phoneme, which is exactly what makes these rules expressible: without
// it, `eɪ` in `e͡ɪ` and `eɪ` across a syllable boundary would look the same.
// It is deleted at the very end, after every rule that needed it.
var e2m = []struct{ from, to string }{
	{"ʔˌn̩", "ʔn"},
	{"ʔn̩", "ʔn"},
	{"a^ɪ", "I"},
	{"a^ʊ", "W"},
	{"d^ʒ", "ʤ"},
	{"e^ɪ", "A"},
	{"t^ʃ", "ʧ"},
	{"ɔ^ɪ", "Y"},
	{"ə^l", "ᵊl"},
	{"ʲo", "jo"},
	{"ʲə", "jə"},
	{"e", "A"},
	{"ʲ", ""},
	{"ɚ", "əɹ"},
	{"r", "ɹ"},
	{"x", "k"},
	{"ç", "k"},
	{"ɐ", "ə"},
	{"ɬ", "l"},
	{"̃", ""},
}

// FromEspeak is misaki's `EspeakFallback.__call__` minus the call: given
// espeak's IPA with `^` ties, it produces the phonemes kokoro reads.
//
// The American rules at the end are where the two Englishes diverge and they
// are not symmetrical. `ɜː` becomes `ɜɹ` — the rhotic vowel espeak spells as a
// length mark — and then *every* remaining length mark is dropped, because
// kokoro's American vocabulary has no `ː` in it at all. The British build
// keeps `ː` and rewrites three diphthongs instead.
func FromEspeak(ps string, british bool) string {
	ps = strings.TrimSpace(ps)
	for _, r := range e2m {
		ps = strings.ReplaceAll(ps, r.from, r.to)
	}
	// A syllabic consonant — `n̩` in "button" — becomes a schwa in front of it.
	ps = syllabic(ps)
	if british {
		ps = strings.NewReplacer(
			"e^ə", "ɛː",
			"iə", "ɪə",
			"ə^ʊ", "Q",
		).Replace(ps)
	} else {
		ps = strings.ReplaceAll(ps, "o^ʊ", "O")
		ps = strings.ReplaceAll(ps, "ɜːɹ", "ɜɹ")
		ps = strings.ReplaceAll(ps, "ɜː", "ɜɹ")
		ps = strings.ReplaceAll(ps, "ɪə", "iə")
		ps = strings.ReplaceAll(ps, "ː", "")
	}
	// espeak below 1.52 spells the open-o with a plain `o`.
	ps = strings.ReplaceAll(ps, "o", "ɔ")
	ps = Americanize(ps)
	return strings.ReplaceAll(ps, "^", "")
}

// syllabic is `re.sub(r'(\S)̩', r'ᵊ\1', ps)` followed by deleting any
// combining ring that is left: the mark says the consonant before it *is* the
// syllable, and kokoro spells that as a schwa in front of the consonant.
func syllabic(ps string) string {
	rs := []rune(ps)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		if i+1 < len(rs) && rs[i+1] == '̩' && !unicode.IsSpace(rs[i]) {
			b.WriteRune('ᵊ')
			b.WriteRune(rs[i])
			i++
			continue
		}
		if rs[i] == '̩' {
			continue
		}
		b.WriteRune(rs[i])
	}
	return b.String()
}
