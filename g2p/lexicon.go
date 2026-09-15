package g2p

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

// Ratings are misaki's confidence in a pronunciation, and they are worth
// keeping because they say which mechanism produced it: 4 is the hand-checked
// gold dictionary, 3 the generated silver one or a proper noun spelled out
// letter by letter, and 2 the espeak fallback (which this package does not
// have yet).
const (
	RatingGold   = 4
	RatingSilver = 3
	RatingEspeak = 2
)

// entry is one dictionary line: either a single pronunciation or one per
// part-of-speech tag.
//
// A tag mapping to the empty string is misaki's JSON `null`, and it does not
// mean "no pronunciation" — it means *this* tag has none, so the lookup falls
// through to the proper-noun path rather than to DEFAULT. 67 gold entries are
// like that, and collapsing them onto DEFAULT would mispronounce every one.
type entry struct {
	plain string
	byTag map[string]string
}

func (e entry) conditioned() bool { return e.byTag != nil }

// Lexicon is misaki's English dictionary and the rules that reach past it.
//
// Two dictionaries, not one, and the order matters: `gold` is hand-checked and
// `silver` generated, so a word in both takes gold's pronunciation and gold's
// rating. Together they resolve 91% of the tokens in real running text.
type Lexicon struct {
	British bool

	gold   map[string]entry
	silver map[string]entry

	// capStresses is misaki's (0.5, 2): the stress level a capitalised word
	// gets and the one an all-caps word gets. Nothing else raises it.
	capStresses [2]float64

	// Fallback, when set, reads a word the dictionary cannot answer for.
	// Nil is a working configuration and was the only one before T5d: an
	// unknown word is then dropped and counted rather than guessed at.
	Fallback *Espeak
}

// symbols and addSymbols are the punctuation misaki pronounces rather than
// passes through. `.` and `/` are only spoken when spacy tags them ADD, which
// is what it does inside a URL or an email address.
var (
	symbols    = map[string]string{"%": "percent", "&": "and", "+": "plus", "@": "at"}
	addSymbols = map[string]string{".": "dot", "/": "slash"}
	usTaus     = runeSet("AIOWYiuæɑəɛɪɹʊʌ")
	vsRe       = regexp.MustCompile(`(?i)^vs\.?$`)
)

// doubledIng is misaki's `([bcdgklmnprstvxz])\1ing$|cking$`, written out
// because RE2 has no backreferences: "stopping" and "trafficking" drop four
// characters to reach their stem, not three.
func doubledIng(word string) bool {
	rs := []rune(word)
	if len(rs) < 5 || string(rs[len(rs)-3:]) != "ing" {
		return false
	}
	a, b := rs[len(rs)-5], rs[len(rs)-4]
	if a == 'c' && b == 'k' {
		return true
	}
	return a == b && strings.ContainsRune("bcdgklmnprstvxz", a)
}

// lexiconOrds is misaki's LEXICON_ORDS: the apostrophe, the hyphen and the
// two ASCII alphabets. A word containing anything else is not looked up at
// all — which is why `café` goes to the fallback, as misaki's own TODO says.
func inLexiconOrds(r rune) bool {
	return r == '\'' || r == '-' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')
}

// Load reads the two dictionaries `reference/convert_misaki.py` exports.
//
// dir is the directory holding `us_gold.txt` and `us_silver.txt` (or the `gb_`
// pair). Both are plain text, one entry per line, so this is a scan rather
// than a JSON parse: 183 thousand entries in about 60 ms.
func Load(dir string, british bool) (*Lexicon, error) {
	prefix := "us"
	if british {
		prefix = "gb"
	}
	l := &Lexicon{British: british, capStresses: [2]float64{0.5, 2}}
	var err error
	if l.gold, err = readDict(filepath.Join(dir, prefix+"_gold.txt")); err != nil {
		return nil, err
	}
	if l.silver, err = readDict(filepath.Join(dir, prefix+"_silver.txt")); err != nil {
		return nil, err
	}
	l.gold = growDictionary(l.gold)
	l.silver = growDictionary(l.silver)
	return l, nil
}

func readDict(path string) (map[string]entry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("g2p: %w (run reference/convert_misaki.py)", err)
	}
	defer f.Close()
	out := make(map[string]entry, 1<<17)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		word, value, ok := strings.Cut(line, "\t")
		if !ok || word == "" {
			continue
		}
		if !strings.Contains(value, "=") {
			out[word] = entry{plain: value}
			continue
		}
		byTag := map[string]string{}
		for _, part := range strings.Split(value, "|") {
			tag, ps, _ := strings.Cut(part, "=")
			byTag[tag] = ps
		}
		out[word] = entry{byTag: byTag}
	}
	return out, sc.Err()
}

// growDictionary is misaki's: a lowercase entry also answers for its
// capitalised form and vice versa, with the originals winning any collision.
//
// It is what makes `Fox` find `fox` without a separate case-folding pass in
// every lookup — and it has to run after both files are read, because the
// added forms must not shadow a real entry.
func growDictionary(d map[string]entry) map[string]entry {
	out := make(map[string]entry, len(d)*2)
	for k, v := range d {
		if len([]rune(k)) < 2 {
			continue
		}
		lower := strings.ToLower(k)
		if k == lower {
			if c := capitalize(k); k != c {
				out[c] = v
			}
		} else if k == capitalize(lower) {
			out[lower] = v
		}
	}
	for k, v := range d {
		out[k] = v
	}
	return out
}

// capitalize is Python's str.capitalize: first rune upper, the rest lower.
func capitalize(s string) string {
	rs := []rune(strings.ToLower(s))
	if len(rs) == 0 {
		return s
	}
	rs[0] = unicode.ToUpper(rs[0])
	return string(rs)
}

func isAlpha(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func allInLexiconOrds(s string) bool {
	for _, r := range s {
		if !inLexiconOrds(r) {
			return false
		}
	}
	return true
}

// IsKnown is misaki's `is_known`: whether the dictionaries can be expected to
// answer for this word, including the acronym case — an all-caps word whose
// lowercase form is in gold, or one whose tail is all capitals, is spelled out
// letter by letter rather than looked up.
func (l *Lexicon) IsKnown(word, tag string) bool {
	if _, ok := l.gold[word]; ok {
		return true
	}
	if _, ok := symbols[word]; ok {
		return true
	}
	if _, ok := l.silver[word]; ok {
		return true
	}
	if !isAlpha(word) || !allInLexiconOrds(word) {
		return false
	}
	if len([]rune(word)) == 1 {
		return true
	}
	if word == strings.ToUpper(word) {
		if _, ok := l.gold[strings.ToLower(word)]; ok {
			return true
		}
	}
	tail := string([]rune(word)[1:])
	return tail == strings.ToUpper(tail)
}

// parentTag collapses a Penn Treebank tag onto the coarse one the dictionary
// keys its 790 conditioned entries by.
func parentTag(tag string) string {
	switch {
	case tag == "":
		return ""
	case strings.HasPrefix(tag, "VB"):
		return "VERB"
	case strings.HasPrefix(tag, "NN"):
		return "NOUN"
	case strings.HasPrefix(tag, "ADV"), strings.HasPrefix(tag, "RB"):
		return "ADV"
	case strings.HasPrefix(tag, "ADJ"), strings.HasPrefix(tag, "JJ"):
		return "ADJ"
	}
	return tag
}

// lookup is misaki's, and the tri-state in the middle is the whole of why this
// package does not need a tagger for most words: when the context has no vowel
// information — a phrase boundary — an entry's `None` key wins over any tag.
func (l *Lexicon) lookup(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	isNNP := false
	if _, inGold := l.gold[word]; word == strings.ToUpper(word) && !inGold {
		word = strings.ToLower(word)
		isNNP = tag == "NNP"
	}
	e, ok := l.gold[word]
	rating := RatingGold
	if !ok && !isNNP {
		e, ok = l.silver[word]
		rating = RatingSilver
	}
	ps := ""
	if ok {
		if !e.conditioned() {
			ps = e.plain
		} else {
			key := tag
			if ctx != nil && ctx.FutureVowel == nil {
				if _, has := e.byTag["None"]; has {
					key = "None"
				} else if _, has := e.byTag[key]; !has {
					key = parentTag(key)
				}
			} else if _, has := e.byTag[key]; !has {
				key = parentTag(key)
			}
			if v, has := e.byTag[key]; has {
				ps = v // may be "" — misaki's null, which falls through below
			} else {
				ps = e.byTag["DEFAULT"]
			}
		}
	}
	if ps == "" || (isNNP && !strings.ContainsRune(ps, primaryStress)) {
		if nnp, r, ok := l.NNP(word); ok {
			return nnp, r, true
		}
		if ps == "" {
			return "", 0, false
		}
	}
	return applyStress(ps, stress), rating, true
}

// NNP spells a word out letter by letter, which is what an acronym and an
// unknown proper noun both get: each letter's own pronunciation, every stress
// demoted to secondary, and the last one promoted back to primary.
//
// `GPU` is `ʤˌipˌijˈu` — three letters, stress on the last, which is how
// English says an initialism.
func (l *Lexicon) NNP(word string) (string, int, bool) {
	var b strings.Builder
	for _, r := range word {
		if !unicode.IsLetter(r) {
			continue
		}
		e, ok := l.gold[strings.ToUpper(string(r))]
		if !ok || e.conditioned() {
			return "", 0, false
		}
		b.WriteString(e.plain)
	}
	if b.Len() == 0 {
		return "", 0, false
	}
	ps := applyStress(b.String(), f64(0))
	if i := strings.LastIndex(ps, string(secondaryStress)); i >= 0 {
		ps = ps[:i] + string(primaryStress) + ps[i+len(string(secondaryStress)):]
	}
	return ps, RatingSilver, true
}

// specialCase is misaki's `get_special_case`: the function words whose
// pronunciation is decided by what follows them rather than by the dictionary.
//
// Every one of these is a real English alternation. `the` is `ði` before a
// vowel and `ðə` otherwise; `to` is `tʊ` before a vowel, `tə` before a
// consonant and the dictionary's `tu` at a boundary; `a` is `ɐ` as an article
// and `ˈA` as the letter. Getting them wrong is not subtle — it is the
// difference between speech and a spelling bee.
func (l *Lexicon) specialCase(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	fv := func() *bool {
		if ctx == nil {
			return nil
		}
		return ctx.FutureVowel
	}
	switch {
	case tag == "ADD" && addSymbols[word] != "":
		return l.lookup(addSymbols[word], "", f64(-0.5), ctx)
	case symbols[word] != "":
		return l.lookup(symbols[word], "", nil, ctx)
	case strings.Contains(strings.Trim(word, "."), ".") &&
		isAlpha(strings.ReplaceAll(word, ".", "")) && longestPart(word, ".") < 3:
		return l.NNP(word)
	case word == "a" || word == "A":
		if tag == "DT" {
			return "ɐ", RatingGold, true
		}
		return "ˈA", RatingGold, true
	case word == "am" || word == "Am" || word == "AM":
		if strings.HasPrefix(tag, "NN") {
			return l.NNP(word)
		}
		if fv() == nil || word != "am" || (stress != nil && *stress > 0) {
			return l.gold["am"].plain, RatingGold, true
		}
		return "ɐm", RatingGold, true
	case word == "an" || word == "An" || word == "AN":
		if word == "AN" && strings.HasPrefix(tag, "NN") {
			return l.NNP(word)
		}
		return "ɐn", RatingGold, true
	case word == "I" && tag == "PRP":
		return string(secondaryStress) + "I", RatingGold, true
	case (word == "by" || word == "By" || word == "BY") && parentTag(tag) == "ADV":
		return "bˈI", RatingGold, true
	case word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")):
		switch v := fv(); {
		case v == nil:
			return l.gold["to"].plain, RatingGold, true
		case *v:
			return "tʊ", RatingGold, true
		default:
			return "tə", RatingGold, true
		}
	case word == "in" || word == "In" || (word == "IN" && tag != "NNP"):
		mark := ""
		if fv() == nil || tag != "IN" {
			mark = string(primaryStress)
		}
		return mark + "ɪn", RatingGold, true
	case word == "the" || word == "The" || (word == "THE" && tag == "DT"):
		if v := fv(); v != nil && *v {
			return "ði", RatingGold, true
		}
		return "ðə", RatingGold, true
	case tag == "IN" && vsRe.MatchString(word):
		return l.lookup("versus", "", nil, ctx)
	case word == "used" || word == "Used" || word == "USED":
		if (tag == "VBD" || tag == "JJ") && ctx != nil && ctx.FutureTo {
			return l.gold["used"].byTag["VBD"], RatingGold, true
		}
		return l.gold["used"].byTag["DEFAULT"], RatingGold, true
	}
	return "", 0, false
}

func longestPart(s, sep string) int {
	n := 0
	for _, p := range strings.Split(s, sep) {
		if len([]rune(p)) > n {
			n = len([]rune(p))
		}
	}
	return n
}

// The three suffix rules. They are phonological, not orthographic: `-s` is
// /s/ after a voiceless consonant, /ᵻz/ after a sibilant and /z/ otherwise,
// and `-ed` is /t/, /ᵻd/ or /d/ on the same principle. Between them they
// cover 2.7% of running-text tokens the dictionary does not list.
func (l *Lexicon) suffixS(stem string) (string, bool) {
	if stem == "" {
		return "", false
	}
	last := lastRune(stem)
	switch {
	case strings.ContainsRune("ptkfθ", last):
		return stem + "s", true
	case strings.ContainsRune("szʃʒʧʤ", last):
		if l.British {
			return stem + "ɪz", true
		}
		return stem + "ᵻz", true
	}
	return stem + "z", true
}

func (l *Lexicon) suffixEd(stem string) (string, bool) {
	if stem == "" {
		return "", false
	}
	rs := []rune(stem)
	last := rs[len(rs)-1]
	switch {
	case strings.ContainsRune("pkfθʃsʧ", last):
		return stem + "t", true
	case last == 'd':
		if l.British {
			return stem + "ɪd", true
		}
		return stem + "ᵻd", true
	case last != 't':
		return stem + "d", true
	case l.British || len(rs) < 2:
		return stem + "ɪd", true
	case usTaus[rs[len(rs)-2]]:
		// A tap between two vowels: "waited" is `wAɾᵻd`, not `wAtᵻd`.
		return string(rs[:len(rs)-1]) + "ɾᵻd", true
	}
	return stem + "ᵻd", true
}

func (l *Lexicon) suffixIng(stem string) (string, bool) {
	if stem == "" {
		return "", false
	}
	rs := []rune(stem)
	last := rs[len(rs)-1]
	if l.British {
		if last == 'ə' || last == 'ː' {
			return "", false
		}
	} else if len(rs) > 1 && last == 't' && usTaus[rs[len(rs)-2]] {
		return string(rs[:len(rs)-1]) + "ɾɪŋ", true
	}
	return stem + "ɪŋ", true
}

func lastRune(s string) rune {
	rs := []rune(s)
	return rs[len(rs)-1]
}

func (l *Lexicon) stemS(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	rs := []rune(word)
	if len(rs) < 3 || !strings.HasSuffix(word, "s") {
		return "", 0, false
	}
	var stem string
	switch {
	case !strings.HasSuffix(word, "ss") && l.IsKnown(string(rs[:len(rs)-1]), tag):
		stem = string(rs[:len(rs)-1])
	case (strings.HasSuffix(word, "'s") ||
		(len(rs) > 4 && strings.HasSuffix(word, "es") && !strings.HasSuffix(word, "ies"))) &&
		l.IsKnown(string(rs[:len(rs)-2]), tag):
		stem = string(rs[:len(rs)-2])
	case len(rs) > 4 && strings.HasSuffix(word, "ies") && l.IsKnown(string(rs[:len(rs)-3])+"y", tag):
		stem = string(rs[:len(rs)-3]) + "y"
	default:
		return "", 0, false
	}
	ps, rating, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", 0, false
	}
	out, ok := l.suffixS(ps)
	return out, rating, ok
}

func (l *Lexicon) stemEd(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	rs := []rune(word)
	if len(rs) < 4 || !strings.HasSuffix(word, "d") {
		return "", 0, false
	}
	var stem string
	switch {
	case !strings.HasSuffix(word, "dd") && l.IsKnown(string(rs[:len(rs)-1]), tag):
		stem = string(rs[:len(rs)-1])
	case len(rs) > 4 && strings.HasSuffix(word, "ed") && !strings.HasSuffix(word, "eed") &&
		l.IsKnown(string(rs[:len(rs)-2]), tag):
		stem = string(rs[:len(rs)-2])
	default:
		return "", 0, false
	}
	ps, rating, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", 0, false
	}
	out, ok := l.suffixEd(ps)
	return out, rating, ok
}

func (l *Lexicon) stemIng(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	rs := []rune(word)
	if len(rs) < 5 || !strings.HasSuffix(word, "ing") {
		return "", 0, false
	}
	var stem string
	switch {
	case len(rs) > 5 && l.IsKnown(string(rs[:len(rs)-3]), tag):
		stem = string(rs[:len(rs)-3])
	case l.IsKnown(string(rs[:len(rs)-3])+"e", tag):
		stem = string(rs[:len(rs)-3]) + "e"
	case len(rs) > 5 && doubledIng(word) && l.IsKnown(string(rs[:len(rs)-4]), tag):
		stem = string(rs[:len(rs)-4])
	default:
		return "", 0, false
	}
	ps, rating, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", 0, false
	}
	out, ok := l.suffixIng(ps)
	return out, rating, ok
}

// Word is misaki's `get_word`: everything between a token and its
// pronunciation that is not a number, a tokenizer or an espeak fallback.
//
// The order is the finding. Special cases first, because `the` and `to` are in
// the dictionary and the dictionary is wrong about them in context. Then a
// retry in lowercase for a capitalised word the dictionary only knows in
// lower case — guarded so that a genuine proper noun is *not* folded down and
// gets spelled out instead. Then the dictionary, then the possessives, then
// the three suffix rules in the order a word is most likely to have taken.
//
// stress is nil for an all-lowercase word, 0.5 for a capitalised one and 2 for
// an all-caps one; see applyStress.
func (l *Lexicon) Word(word, tag string, stress *float64, ctx *Context) (string, int, bool) {
	if ps, rating, ok := l.specialCase(word, tag, stress, ctx); ok {
		return ps, rating, true
	}
	wl := strings.ToLower(word)
	if l.shouldLower(word, wl, tag, stress, ctx) {
		word = wl
	}
	if l.IsKnown(word, tag) {
		return l.lookup(word, tag, stress, ctx)
	}
	rs := []rune(word)
	if strings.HasSuffix(word, "s'") && l.IsKnown(string(rs[:len(rs)-2])+"'s", tag) {
		return l.lookup(string(rs[:len(rs)-2])+"'s", tag, stress, ctx)
	}
	if strings.HasSuffix(word, "'") && l.IsKnown(string(rs[:len(rs)-1]), tag) {
		return l.lookup(string(rs[:len(rs)-1]), tag, stress, ctx)
	}
	if ps, rating, ok := l.stemS(word, tag, stress, ctx); ok {
		return ps, rating, true
	}
	if ps, rating, ok := l.stemEd(word, tag, stress, ctx); ok {
		return ps, rating, true
	}
	ingStress := stress
	if ingStress == nil {
		ingStress = f64(0.5)
	}
	if ps, rating, ok := l.stemIng(word, tag, ingStress, ctx); ok {
		return ps, rating, true
	}
	return "", 0, false
}

// shouldLower is misaki's one long condition, kept in one piece because every
// clause of it is load-bearing: a capitalised word is folded to lower case
// only when it is alphabetic, is not in either dictionary as written, is
// either all-caps or capitalised-only, and its lowercase form *is* resolvable.
// A short NNP is exempt, which is what keeps `Smith` out of the dictionary's
// `smith` and sends it to the spelling path instead.
func (l *Lexicon) shouldLower(word, wl, tag string, stress *float64, ctx *Context) bool {
	rs := []rune(word)
	if len(rs) <= 1 || !isAlpha(strings.ReplaceAll(word, "'", "")) || word == wl {
		return false
	}
	if tag == "NNP" && len(rs) <= 7 {
		return false
	}
	if _, ok := l.gold[word]; ok {
		return false
	}
	if _, ok := l.silver[word]; ok {
		return false
	}
	tail := string(rs[1:])
	if word != strings.ToUpper(word) && tail != strings.ToLower(tail) {
		return false
	}
	if _, ok := l.gold[wl]; ok {
		return true
	}
	if _, ok := l.silver[wl]; ok {
		return true
	}
	for _, fn := range []func(string, string, *float64, *Context) (string, int, bool){
		l.stemS, l.stemEd, l.stemIng,
	} {
		if _, _, ok := fn(wl, tag, stress, ctx); ok {
			return true
		}
	}
	return false
}

// CapStress is the stress level a token's capitalisation implies, which is the
// only input to applyStress that comes from the spelling rather than the
// dictionary.
func (l *Lexicon) CapStress(word string) *float64 {
	if word == strings.ToLower(word) {
		return nil
	}
	if word == strings.ToUpper(word) {
		return f64(l.capStresses[1])
	}
	return f64(l.capStresses[0])
}
