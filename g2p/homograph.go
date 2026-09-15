package g2p

import "strings"

// tagVariant is the pronunciation a tag-conditioned entry gives for one tag,
// ignoring the context — which is what isolates a tagger's contribution from
// the `None` key's, since that one is chosen by whether a vowel follows.
//
// It returns false for a word the tag cannot change, which is most of them:
// 89411 of 90201 gold entries carry a single pronunciation.
func (l *Lexicon) tagVariant(word, tag string) (string, bool) {
	e, ok := l.gold[word]
	if !ok {
		e, ok = l.gold[strings.ToLower(word)]
	}
	if !ok || !e.conditioned() {
		return "", false
	}
	key := tag
	if _, has := e.byTag[key]; !has {
		key = parentTag(key)
	}
	if v, has := e.byTag[key]; has {
		return v, true
	}
	return e.byTag["DEFAULT"], true
}

// The words that decide a homograph, by what they do to the word after them.
var (
	// A determiner or a possessive in front of a noun/verb homograph makes it
	// a noun: "a record", "the present", "my object".
	determiners = wordSet(`a an the this that these those my your his her its
		our their no any some every each another other such what which whose
		one two three four five six seven eight nine ten`)
	// A modal, an infinitive marker or a do-support auxiliary makes it a bare
	// verb — which for these entries is the same as no tag at all.
	bareVerbCue = wordSet(`to will would shall should can could may might must
		do does did let please not n't dont don't cannot help`)
	// A perfect or passive auxiliary makes it a past participle: "have read",
	// "was recorded", "is present".
	perfectCue = wordSet(`have has had having been be am is are was were get
		gets got getting`)
	// A subject in front of it makes it a tensed verb: "he read", "they live".
	subjectCue = wordSet(`i you he she it we they who`)
	// A preposition or a conjunction in front makes it a noun again: "of
	// record", "in present", "and object".
	nounCue = wordSet(`of in on at for with from into over under about by as
		than through without and or but nor`)
	// The left context that makes `that` a determiner rather than a relative
	// pronoun. Wider than nounCue on purpose: it is a measured set, not a
	// syntactic category.
	thatFunctionWords = wordSet(`of and so on in for with at from to but or nor
		as than what which when where while because if since that this the a an`)
)

// isPunctWord reports whether a token is punctuation rather than a word, which
// for the homograph rules means a boundary is on the left.
func isPunctWord(s string) bool {
	return s != "" && !isWordy(s)
}

func wordSet(s string) map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields(s) {
		m[w] = true
	}
	return m
}

// HomographTag is the rule-based stand-in for spacy's tagger, and it exists
// for one job: choosing between the two pronunciations of the 671 gold entries
// that have them (SPEECH.md T5d).
//
// It looks at one word either side, which is most of what a noun/verb
// homograph needs in English — the distinction is syntactic and local. A
// determiner in front makes a noun, a modal makes a bare verb, a perfect
// auxiliary makes a participle, a subject makes a tensed verb. Everything else
// gets no tag, which resolves through the entry's DEFAULT and is right far
// more often than a most-frequent-tag table would be, because DEFAULT *is*
// the frequent form.
//
// `TestHomographTagger` measures it against spacy over 1962 real occurrences
// and prints what each word costs. The number to beat is the no-tagger
// baseline, not perfection.
func (l *Lexicon) HomographTag(prev, word, next string) string {
	p := strings.ToLower(prev)
	w := strings.ToLower(word)

	// `that` is the most common of them by far — 445 of 1962 occurrences —
	// and the only one whose distinction is not noun-against-verb: spacy
	// calls it DT when it determines a noun phrase and WDT or IN when it
	// opens a relative or a complement clause, and only DT is stressed.
	//
	// The signal is entirely on the left. A relative `that` follows the noun
	// it modifies ("the kernel that", "things that"); a determiner follows a
	// boundary or a function word ("of that", "and that", ". That"). Measured
	// over those 445: **83.1% against a 67.0% baseline** of never saying DT.
	//
	// The obvious refinement — a determiner cannot be followed by a verb — is
	// *wrong here* and costs 8.5 points, because "that is" is overwhelmingly a
	// determiner in running prose. It was tried and measured rather than
	// reasoned about, which is the only way that would have been found.
	if w == "that" {
		if prev == "" || isPunctWord(prev) || thatFunctionWords[p] {
			return "DT"
		}
		return "IN"
	}

	switch {
	case determiners[p]:
		return "NN"
	case perfectCue[p]:
		return "VBN"
	case bareVerbCue[p]:
		return "VB"
	case subjectCue[p]:
		return "VBP"
	case nounCue[p]:
		return "NN"
	}
	// A content word in front means a subject, and a subject means a tensed
	// verb: "the colonel read the record". It is restricted to the handful of
	// entries that carry a VBD key — `read` and three others — because
	// applied generally it is a disaster: "a memory fragment" would become the
	// verb, and `fragment` alone is 79 of the corpus's 1962 occurrences.
	if l.hasTagKey(word, "VBD") && prev != "" && !isPunctWord(prev) {
		return "VBD"
	}
	return ""
}

// hasTagKey reports whether a conditioned entry is keyed by a particular tag,
// which is how a rule asks whether it is even relevant to this word.
func (l *Lexicon) hasTagKey(word, tag string) bool {
	e, ok := l.gold[word]
	if !ok {
		e, ok = l.gold[strings.ToLower(word)]
	}
	if !ok || !e.conditioned() {
		return false
	}
	_, has := e.byTag[tag]
	return has
}
