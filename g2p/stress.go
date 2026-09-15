// Package g2p turns English text into the IPA phonemes Kokoro-82M reads
// (SPEECH.md T5).
//
// The oracle is misaki's `en.G2P`, which is what the model was trained on, and
// the port follows its structure rather than improving on it: a pronunciation
// looked up in a 183-thousand-entry dictionary, three suffix rules for what
// the dictionary misses, a handful of special cases for the function words
// English stresses differently depending on what follows them, and a set of
// rules for moving stress marks around. `reference/dump_g2p.py` records what
// misaki makes of a 24-sentence corpus, token by token, and `lexicon_test.go`
// holds this package to it.
//
// What is *not* here yet, and what each is worth, measured over 54 k tokens of
// real running English in `reference/out/g2p/manifest.json`:
//
//   - numbers, currency and ordinals, which need a `num2words` of their own;
//   - an espeak fallback for the 8.8% of tokens the dictionary misses (a
//     figure inflated by the survey corpus being full of acronyms);
//   - a part-of-speech tagger, which decides **1.74%** of tokens over 69
//     distinct words. misaki runs spacy for this; the survey separates those
//     from the 1.83% that only *look* tag-conditioned and are really decided
//     by whether a vowel follows.
package g2p

import (
	"sort"
	"strings"
)

// The two stress marks, in misaki's order.
const (
	secondaryStress = 'ˌ'
	primaryStress   = 'ˈ'
)

// vowels is misaki's US_VOCAB vowel set, including the five symbols kokoro
// spells diphthongs with — A, I, O, W, Y are `eɪ`, `aɪ`, `oʊ`, `aʊ` and `ɔɪ`
// written as one character each, which is what makes the vocabulary 178
// symbols instead of 200.
var vowels = runeSet("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")

// consonants and nonQuotePuncts are what decide the context a lookup sees:
// the next *spoken* symbol is a vowel, a consonant, or the end of a phrase.
var (
	consonants     = runeSet("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
	nonQuotePuncts = runeSet(";:,.!?—…")
	diphthongs     = runeSet("AIOQWYʤʧ")
)

func runeSet(s string) map[rune]bool {
	m := make(map[rune]bool, len(s))
	for _, r := range s {
		m[r] = true
	}
	return m
}

func hasAny(s string, set map[rune]bool) bool {
	for _, r := range s {
		if set[r] {
			return true
		}
	}
	return false
}

// stressWeight is misaki's, and counts a diphthong as two — which is what
// makes it a syllable count in a vocabulary where a diphthong is one rune.
func stressWeight(ps string) int {
	n := 0
	for _, r := range ps {
		if diphthongs[r] {
			n += 2
		} else {
			n++
		}
	}
	return n
}

// applyStress is misaki's `apply_stress`, which is the only part of this
// package that rewrites a pronunciation rather than looking one up.
//
// The argument is a level rather than a flag, and the branches are ordered:
// below -1 strips every mark, -1 demotes primary to secondary, 0 and 0.5
// promote an unmarked word to secondary, 1 and above promote secondary to
// primary. `nil` means "leave it alone", which is what an all-lowercase word
// gets — capitalisation is the whole of what raises the level, 0.5 for a
// capitalised word and 2 for an all-caps one.
//
// The two promoting branches do not simply prepend a mark: a stress mark sits
// immediately before the vowel it belongs to, so restress moves it there.
func applyStress(ps string, stress *float64) string {
	if stress == nil {
		return ps
	}
	s := *stress
	hasPrimary := strings.ContainsRune(ps, primaryStress)
	hasSecondary := strings.ContainsRune(ps, secondaryStress)
	unmarked := !hasPrimary && !hasSecondary
	switch {
	case s < -1:
		return strings.NewReplacer(string(primaryStress), "", string(secondaryStress), "").Replace(ps)
	case s == -1, (s == 0 || s == -0.5) && hasPrimary:
		// Sequential in misaki — remove every secondary, then demote the
		// primary — which one pass reproduces because neither output is the
		// other's input.
		return strings.NewReplacer(
			string(secondaryStress), "",
			string(primaryStress), string(secondaryStress)).Replace(ps)
	case (s == 0 || s == 0.5 || s == 1) && unmarked:
		if !hasAny(ps, vowels) {
			return ps
		}
		return restress(string(secondaryStress) + ps)
	case s >= 1 && !hasPrimary && hasSecondary:
		return strings.ReplaceAll(ps, string(secondaryStress), string(primaryStress))
	case s > 1 && unmarked:
		if !hasAny(ps, vowels) {
			return ps
		}
		return restress(string(primaryStress) + ps)
	}
	return ps
}

// restress moves every stress mark to just before the next vowel.
//
// misaki does it by giving each mark the fractional position `j - 0.5`, where
// j is the index of that vowel, and re-sorting — so a mark slides rightwards
// past any consonants between it and the syllable it marks, and two marks
// cannot cross. The sort is stable, which is what keeps a mark behind an
// earlier one that landed on the same vowel.
func restress(ps string) string {
	rs := []rune(ps)
	type placed struct {
		at float64
		r  rune
	}
	out := make([]placed, len(rs))
	for i, r := range rs {
		out[i] = placed{float64(i), r}
		if r != primaryStress && r != secondaryStress {
			continue
		}
		for j := i; j < len(rs); j++ {
			if vowels[rs[j]] {
				out[i].at = float64(j) - 0.5
				break
			}
		}
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].at < out[b].at })
	var b strings.Builder
	for _, p := range out {
		b.WriteRune(p.r)
	}
	return b.String()
}

// Americanize is the substitution misaki makes once, at the very end of a
// whole utterance: the alveolar tap and the glottal stop are spelled `T` and
// `t` in kokoro's vocabulary rather than `ɾ` and `ʔ`.
//
// It is not folded into the lookup because it must not happen before the
// suffix rules run — `_ed` branches on whether a stem ends in `t`, and a tap
// rewritten early would change which branch it takes.
func Americanize(ps string) string {
	return strings.NewReplacer("ɾ", "T", "ʔ", "t").Replace(ps)
}

// Context is what a lookup knows about the token to its right, and it is the
// finding that keeps a neural tagger out of this package.
//
// FutureVowel is nil where the next spoken symbol is a phrase boundary or a
// punctuation mark, true where it is a vowel and false where it is a
// consonant. A nil FutureVowel selects a lexicon entry's `None` key — which
// looks like a part-of-speech key and is not one: `that`, `this`, `by`, `has`,
// `be`, `would`, `there` and the rest of the function words carry a stressed
// form for the end of a phrase and an unstressed one for the middle of it, and
// 1.83% of running-text tokens are decided this way.
type Context struct {
	FutureVowel *bool
	FutureTo    bool
}

// Next is misaki's `token_context`: the context a token's left neighbour will
// see, given this token's phonemes and text.
//
// The scan stops at the first symbol that is a vowel, a consonant or a
// non-quote punctuation mark, and quotes are skipped rather than treated as a
// boundary — so `she said "hello"` gives the same context as `she said hello`.
// A token with no phonemes at all leaves the context unchanged, which is what
// carries it across an elided subtoken.
func (c Context) Next(phonemes, text, tag string) Context {
	out := Context{FutureVowel: c.FutureVowel}
	for _, r := range phonemes {
		if !vowels[r] && !consonants[r] && !nonQuotePuncts[r] {
			continue
		}
		if nonQuotePuncts[r] {
			out.FutureVowel = nil
		} else {
			v := vowels[r]
			out.FutureVowel = &v
		}
		break
	}
	out.FutureTo = text == "to" || text == "To" || (text == "TO" && (tag == "TO" || tag == "IN"))
	return out
}

// f64 makes a stress level addressable, since nil is one of its values.
func f64(v float64) *float64 { return &v }
