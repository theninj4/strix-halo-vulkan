package g2p

import (
	"sort"
	"strings"
	"unicode"
)

// stressIdx is one piece of a run, for resolve_tokens' half-demotion.
type stressIdx struct {
	primary bool
	weight  int
	i       int
}

// Token is misaki's `MToken`: a piece of the input with whatever the pipeline
// has worked out about it so far.
type Token struct {
	Text       string
	Tag        string
	Whitespace string

	Phonemes string
	Resolved bool // Phonemes is meaningful, even when it is empty
	Rating   int

	IsHead   bool
	Currency string
	NumFlags string

	// Punct is set by the tokenizer for a piece that stands beside a word
	// rather than inside one. The distinction cannot be recovered later: the
	// hyphens in `well-known` and the period at the end of a sentence are the
	// same characters, and only the tokenizer knows which is which.
	Punct bool
	// Prespace is misaki's: a space goes in front of this piece when the run
	// it belongs to was two words that happened to be adjacent — `12%` is
	// "twelve percent", not "twelvepercent".
	Prespace bool
}

// puncts is misaki's PUNCTS: the marks that survive into the phoneme string,
// because kokoro's vocabulary contains them and its prosody predictor was
// trained to pause at them.
var (
	puncts        = runeSet(";:,.!?—…\"“”")
	subtokenJunks = runeSet("',-._‘’/")
)

// Tokenize splits text the way misaki's pipeline does, minus spacy.
//
// misaki runs spacy for two things and only one of them is hard. The splitting
// is rules — peel the punctuation off each whitespace-delimited chunk, then
// `Subtokenize` what is left — and the *tagging* is a neural model, which T5a
// priced: a part-of-speech tag decides 1.74% of tokens over 69 distinct words.
// So the tags come from `tagOf`, which is a dozen rules, and `TestPhonemize`
// reports what that costs rather than hiding it.
//
// Two things about the peeling matter more than they look. A trailing period
// is punctuation at the end of a sentence and part of the word in `Dr.` — and
// the dictionary already knows which, so it is asked rather than a list being
// maintained here. And every subtoken of a chunk inherits the chunk's tag,
// which is what spacy's own tokens do: the hyphens inside `well-known` are not
// punctuation, they are inside a word, and treating them as punctuation would
// split the run that `resolveStress` has to see whole.
func (l *Lexicon) Tokenize(text string) []*Token {
	var out []*Token
	text = strings.TrimLeft(text, " \t\n\r")
	rs := []rune(text)
	sentenceStart := true
	for i := 0; i < len(rs); {
		j := i
		for j < len(rs) && !unicode.IsSpace(rs[j]) {
			j++
		}
		chunk := string(rs[i:j])
		k := j
		for k < len(rs) && unicode.IsSpace(rs[k]) {
			k++
		}
		space := " "
		if k == len(rs) {
			space = ""
		}
		i = k
		if chunk == "" {
			continue
		}

		var pieces []*Token
		cr := []rune(chunk)
		for len(cr) > 1 && prefixPunct[cr[0]] {
			pieces = append(pieces, &Token{Text: string(cr[0]), Tag: ".", IsHead: true, Punct: true})
			cr = cr[1:]
		}
		var suffixes []*Token
		for len(cr) > 1 && suffixPunct[cr[len(cr)-1]] && !l.isAbbreviation(string(cr)) {
			suffixes = append([]*Token{{Text: string(cr[len(cr)-1]), Tag: ".", IsHead: true, Punct: true}}, suffixes...)
			cr = cr[:len(cr)-1]
		}
		core := string(cr)
		if core != "" {
			tag := tagOf(core, sentenceStart && len(pieces) == 0)
			subs := Subtokenize(core)
			// A piece of a chunk is punctuation only when the whole chunk was:
			// otherwise it is a hyphen or an apostrophe *inside* a word, and
			// breaking the run there would split what the merge loop and the
			// stress rule both have to see whole.
			wholePunct := !isWordy(core)
			for _, sub := range subs {
				st := tag
				if IsNumber(sub, true) {
					st = "CD"
				}
				pieces = append(pieces, &Token{
					Text: sub, Tag: st, IsHead: true, Punct: wholePunct,
				})
			}
		}
		pieces = append(pieces, suffixes...)
		if len(pieces) == 0 {
			continue
		}
		pieces[len(pieces)-1].Whitespace = space
		out = append(out, pieces...)
		for _, p := range pieces {
			if isSentenceEnd(p.Text) {
				sentenceStart = true
			} else if isWordy(p.Text) {
				sentenceStart = false
			}
		}
	}
	l.retag(out)
	return out
}

// specialCaseWords are the function words `Lexicon.specialCase` decides
// before the dictionary is consulted. Their tags mean particular things there
// — `a` is an article only when DT, `by` is stressed only when ADV — so the
// homograph rules, which know nothing about those meanings, are kept away.
var specialCaseWords = wordSet(`a an the to in i am by used vs`)

// retag is the second pass that resolves homographs, and it is second because
// it needs both neighbours: a noun/verb pair is decided by what is in front of
// it, and `that` by whether a noun is.
//
// It only touches words the dictionary actually pronounces two ways, which is
// 671 of 90201 entries — everywhere else the tag changes nothing and guessing
// one would be free to be wrong.
func (l *Lexicon) retag(tokens []*Token) {
	for i, tk := range tokens {
		if specialCaseWords[strings.ToLower(tk.Text)] {
			continue
		}
		if _, ok := l.tagVariant(tk.Text, ""); !ok {
			continue
		}
		prev, next := "", ""
		if i > 0 {
			prev = tokens[i-1].Text
		}
		if i+1 < len(tokens) {
			next = tokens[i+1].Text
		}
		if tag := l.HomographTag(prev, tk.Text, next); tag != "" {
			tk.Tag = tag
		}
	}
}

// prefixPunct and suffixPunct are what spacy peels off the ends of a chunk.
// The apostrophe is deliberately not a suffix: `users'` is a possessive the
// dictionary path knows how to strip, and peeling it here would hide that.
var (
	prefixPunct = runeSet("\"“‘'([{«¿¡")
	suffixPunct = runeSet(".,;:!?…\"”)]}»")
)

// isAbbreviation decides whether a trailing period belongs to the word.
//
// The dictionary is the authority — it carries `Dr.`, `Mr.`, `etc.` and the
// rest as entries in their own right — plus one pattern it does not cover: a
// run of single letters separated by periods, which is how an initialism like
// `U.S.A.` is written.
func (l *Lexicon) isAbbreviation(chunk string) bool {
	if _, ok := l.gold[chunk]; ok {
		return true
	}
	if !strings.HasSuffix(chunk, ".") {
		return false
	}
	parts := strings.Split(strings.TrimSuffix(chunk, "."), ".")
	if len(parts) < 2 {
		return false
	}
	for _, p := range parts {
		if len([]rune(p)) != 1 || !unicode.IsLetter([]rune(p)[0]) {
			return false
		}
	}
	return true
}

func isWordy(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func isSentenceEnd(s string) bool {
	return s == "." || s == "!" || s == "?" || s == "…"
}

// tagOf is the rule-based stand-in for spacy's tagger.
//
// It only has to produce the tags the lookup actually branches on, which after
// T5a's survey is a short list: CD for a number, DT for the articles, PRP for
// `I`, and NNP for a proper noun — the last because `get_word` exempts a short
// NNP from being folded to lower case, which is what sends `Smith` to the
// spelling path instead of to the dictionary's `smith`.
//
// Everything else is left blank, which the lookup treats as "no tag" and
// resolves through a conditioned entry's DEFAULT. That is right far more often
// than it looks: half the apparently tag-conditioned hits in running text are
// decided by whether a vowel follows, not by a tag at all.
func tagOf(s string, sentenceInitial bool) string {
	switch {
	case s == "":
		return ""
	case IsNumber(s, true):
		return "CD"
	case s == "I":
		return "PRP"
	}
	lower := strings.ToLower(s)
	switch lower {
	case "a", "an", "the", "that", "this", "these", "those", "all", "no", "every":
		return "DT"
	case "to":
		return "TO"
	case "in", "of", "by", "for", "with", "at", "on", "from", "as", "into",
		"than", "through", "over", "under", "about", "since", "without", "versus", "vs", "vs.":
		return "IN"
	case "hundred", "thousand", "million", "billion", "trillion", "dozen":
		// spacy tags these CD, which is what carries a currency symbol past a
		// number onto the scale word: "$3.5 million" is millions of dollars.
		return "CD"
	}
	if !isWordy(s) {
		if s == "-" || s == "–" || s == "—" {
			return ":"
		}
		return "."
	}
	// A capitalised word that does not open a sentence, or an initialism.
	if !sentenceInitial && s != lower && len([]rune(s)) > 1 {
		return "NNP"
	}
	if s == strings.ToUpper(s) && len([]rune(s)) > 1 {
		return "NNP"
	}
	return ""
}

// Retokenize is misaki's: it assigns the punctuation and currency tokens their
// phonemes, and groups the rest into the runs a dictionary lookup is tried
// over — consecutive subtokens with no whitespace between them.
//
// The currency rule is the subtle one and it reaches forwards: a symbol arms a
// flag, and the flag is spent on the *last* number token of the run that
// follows, so `$3.5 million` puts the currency on `million` rather than on
// `3.5` and comes out "three point five million dollars".
func Retokenize(tokens []*Token) [][]*Token {
	var groups [][]*Token
	currency := ""
	openQuote := true
	for i, tk := range tokens {
		switch {
		case currencies[tk.Text] != [2]string{}:
			currency = tk.Text
			tk.Phonemes, tk.Resolved, tk.Rating = "", true, RatingGold
		case symbols[tk.Text] != "":
			// `%`, `&`, `@` and `+` are spoken, not punctuation, and the
			// lookup's special cases are where they are turned into words.
		case tk.Tag == ":" && (tk.Text == "-" || tk.Text == "–"):
			tk.Phonemes, tk.Resolved, tk.Rating = "—", true, RatingSilver
		case tk.Punct:
			tk.Phonemes, tk.Resolved, tk.Rating = punctPhonemes(tk.Text, &openQuote), true, RatingGold
		case currency != "":
			if tk.Tag != "CD" {
				currency = ""
			} else if i+1 == len(tokens) || tokens[i+1].Tag != "CD" {
				tk.Currency = currency
				currency = ""
			}
		}
		switch {
		case tk.Resolved:
			groups = append(groups, []*Token{tk})
		case len(groups) > 0 && len(groups[len(groups)-1]) > 0 &&
			!groups[len(groups)-1][0].Resolved &&
			groups[len(groups)-1][len(groups[len(groups)-1])-1].Whitespace == "":
			tk.IsHead = false
			last := groups[len(groups)-1]
			groups[len(groups)-1] = append(last, tk)
		default:
			groups = append(groups, []*Token{tk})
		}
	}
	return groups
}

// punctPhonemes keeps only the marks kokoro's vocabulary carries, and turns a
// straight double quote into the curly one the model was trained on —
// alternating, since an opening quote and a closing one are different symbols
// and the text does not say which is which.
func punctPhonemes(text string, openQuote *bool) string {
	var b strings.Builder
	for _, r := range text {
		switch r {
		case '"':
			if *openQuote {
				b.WriteRune('“')
			} else {
				b.WriteRune('”')
			}
			*openQuote = !*openQuote
		case '(':
			b.WriteRune('(')
		case ')':
			b.WriteRune(')')
		default:
			if puncts[r] {
				b.WriteRune(r)
			}
		}
	}
	return b.String()
}

// mergeText is `merge_tokens`' text and tag: the pieces glued back together
// with their whitespace, tagged by whichever piece carries the most letters —
// with a capital counting double, so `Smith's` is tagged by `Smith`.
func mergeText(tokens []*Token) (string, string, string, string) {
	var b strings.Builder
	for i, tk := range tokens {
		b.WriteString(tk.Text)
		if i != len(tokens)-1 {
			b.WriteString(tk.Whitespace)
		}
	}
	best, weight := "", -1
	currency, flags := "", ""
	for _, tk := range tokens {
		w := 0
		for _, r := range tk.Text {
			if unicode.IsUpper(r) {
				w += 2
			} else {
				w++
			}
		}
		if w > weight {
			best, weight = tk.Tag, w
		}
		if tk.Currency != "" && tk.Currency > currency {
			currency = tk.Currency
		}
		flags += tk.NumFlags
	}
	return b.String(), best, currency, flags
}

// resolveGroup is misaki's inner loop: try the whole run against the
// dictionary, and on a miss give up a piece from the left and try again, so
// the *longest suffix* that resolves wins and the search then continues on
// what is left of the prefix.
//
// It is what puts `o'c`+`lock` back together — Subtokenize cuts it in the
// middle of a syllable and this is what makes that harmless.
//
// When a piece is neither in the dictionary nor punctuation, the *whole* run
// goes to espeak instead and the piecewise results are thrown away — misaki's
// rule, and the right one: a word the dictionary half-knows is worse read in
// halves than read whole by something that knows letter-to-sound rules.
func (l *Lexicon) resolveGroup(g []*Token, ctx Context) Context {
	left, right := 0, len(g)
	fallback := false
	for left < right {
		resolved := false
		for _, tk := range g[left:right] {
			if tk.Resolved {
				resolved = true
				break
			}
		}
		var ps string
		var rating int
		var ok bool
		var text, tag string
		if !resolved {
			var currency, flags string
			text, tag, currency, flags = mergeText(g[left:right])
			ps, rating, ok = l.Resolve(text, tag, currency, g[left].IsHead, flags, &ctx)
		}
		if ok {
			g[left].Phonemes, g[left].Resolved, g[left].Rating = ps, true, rating
			for _, x := range g[left+1 : right] {
				x.Phonemes, x.Resolved, x.Rating = "", true, rating
			}
			ctx = ctx.Next(ps, text, tag)
			right, left = left, 0
			continue
		}
		if left+1 < right {
			left++
			continue
		}
		right--
		tk := g[right]
		if !tk.Resolved {
			junk := true
			for _, r := range tk.Text {
				if !subtokenJunks[r] {
					junk = false
					break
				}
			}
			if junk {
				tk.Phonemes, tk.Resolved, tk.Rating = "", true, RatingSilver
			} else if l.Fallback != nil {
				fallback = true
				break
			}
		}
		left = 0
	}
	if fallback {
		text, tag, _, _ := mergeText(g)
		ps, rating, ok := l.Fallback.Phonemes(normalizeWord(text))
		for i, tk := range g {
			tk.Phonemes, tk.Resolved, tk.Rating = "", ok, rating
			if i == 0 {
				tk.Phonemes = ps
			}
		}
		if ok {
			ctx = ctx.Next(ps, text, tag)
		}
		return ctx
	}
	resolveStress(g)
	return ctx
}

// Resolve is the whole of a token's pronunciation: the word path, and the
// number path behind it.
func (l *Lexicon) Resolve(word, tag, currency string, isHead bool, numFlags string,
	ctx *Context) (string, int, bool) {
	word = normalizeWord(word)
	stress := l.CapStress(word)
	if ps, rating, ok := l.Word(word, tag, stress, ctx); ok {
		return applyStress(l.AppendCurrency(ps, currency), nil), rating, true
	}
	if IsNumber(word, isHead) {
		return l.Number(word, currency, isHead, numFlags)
	}
	if !allInLexiconOrds(word) {
		return "", 0, false
	}
	return "", 0, false
}

// normalizeWord is `Lexicon.__call__`'s preamble: curly apostrophes become
// straight ones so the dictionary's `don't` is found, and everything else is
// left alone.
func normalizeWord(word string) string {
	return strings.NewReplacer("‘", "'", "’", "'").Replace(word)
}

// resolveStress is misaki's `resolve_tokens`: inside a run that was resolved
// piece by piece rather than as a whole, demote the stress of the quieter
// half, so `twenty-one-year-old` does not come out with four primary stresses.
//
// It does nothing when the run had a space or a slash in it, or mixed letters
// with digits — those are separate words that happened to be adjacent, not one
// word cut up.
func resolveStress(g []*Token) {
	if len(g) < 2 {
		return
	}
	var b strings.Builder
	for i, tk := range g {
		b.WriteString(tk.Text)
		if i != len(g)-1 {
			b.WriteString(tk.Whitespace)
		}
	}
	text := b.String()
	kinds := map[int]bool{}
	for _, r := range text {
		if subtokenJunks[r] {
			continue
		}
		switch {
		case unicode.IsLetter(r):
			kinds[0] = true
		case isASCIIDigit(r):
			kinds[1] = true
		default:
			kinds[2] = true
		}
	}
	prespace := strings.ContainsAny(text, " /") || len(kinds) > 1
	if prespace {
		for _, tk := range g[1:] {
			if tk.Resolved {
				tk.Prespace = true
			}
		}
		return
	}
	var indices []stressIdx
	for i, tk := range g {
		if tk.Phonemes == "" {
			continue
		}
		indices = append(indices, stressIdx{
			strings.ContainsRune(tk.Phonemes, primaryStress), stressWeight(tk.Phonemes), i,
		})
	}
	if len(indices) == 2 && len([]rune(g[indices[0].i].Text)) == 1 {
		i := indices[1].i
		g[i].Phonemes = applyStress(g[i].Phonemes, f64(-0.5))
		return
	}
	n := 0
	for _, v := range indices {
		if v.primary {
			n++
		}
	}
	if len(indices) < 2 || n <= (len(indices)+1)/2 {
		return
	}
	// Sorted as misaki sorts the tuples: unstressed first, then by weight,
	// then by position — and the quieter half is demoted.
	sort.Slice(indices, func(a, b int) bool {
		if indices[a].primary != indices[b].primary {
			return !indices[a].primary
		}
		if indices[a].weight != indices[b].weight {
			return indices[a].weight < indices[b].weight
		}
		return indices[a].i < indices[b].i
	})
	for _, v := range indices[:len(indices)/2] {
		g[v.i].Phonemes = applyStress(g[v.i].Phonemes, f64(-0.5))
	}
}

// Phonemize turns English text into the phoneme string `cmd/tts` takes.
//
// The tokens are walked backwards, because a lookup's context is what follows
// it: whether the next spoken symbol is a vowel decides `the` against `thee`,
// `to` against `tuh`, and the stressed or unstressed form of every function
// word in the language.
//
// A token the dictionary cannot answer for is left out entirely, and the count
// is returned: there is no espeak fallback yet (T5d), so an unknown word is
// silence rather than a guess.
func (l *Lexicon) Phonemize(text string) (string, int) {
	tokens := l.Tokenize(text)
	groups := Retokenize(tokens)
	ctx := Context{}
	for i := len(groups) - 1; i >= 0; i-- {
		g := groups[i]
		if len(g) == 1 && g[0].Resolved {
			ctx = ctx.Next(g[0].Phonemes, g[0].Text, g[0].Tag)
			continue
		}
		ctx = l.resolveGroup(g, ctx)
	}
	var b strings.Builder
	unknown := 0
	for _, tk := range tokens {
		if !tk.Resolved {
			unknown++
		} else {
			out := Americanize(tk.Phonemes)
			if tk.Prespace && out != "" && !endsWithSpace(b.String()) {
				b.WriteString(" ")
			}
			b.WriteString(out)
		}
		b.WriteString(tk.Whitespace)
	}
	return strings.TrimSpace(collapseSpaces(b.String())), unknown
}

func endsWithSpace(s string) bool {
	return s != "" && s[len(s)-1] == ' '
}

func collapseSpaces(s string) string {
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return s
}
