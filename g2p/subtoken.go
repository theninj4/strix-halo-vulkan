package g2p

import "unicode"

// Subtokenize is misaki's `subtokenize`, the one part of its tokenizer that is
// its own rather than spacy's: a word is cut into the pieces the dictionary
// might answer for, and the main loop then tries progressively shorter merges
// of them.
//
// Upstream it is a single nine-alternative regex with two lookaheads, which
// RE2 cannot express, so this is a scanner that tries the same alternatives in
// the same order at each position. The order is the whole behaviour — an
// earlier alternative wins even when a later one would match more — and it is
// pinned by `reference/out/g2p/subtokens.txt` rather than by reading:
//
//	well-known        well | - | known
//	camelCase         camel | Case
//	XMLHttpRequest    XMLHttp | Request
//	macOS             mac | OS
//	XYz               X | Yz
//	1,024,000         1,024,000
//	o'clock           o'c | lock
//
// The last two are the ones worth knowing about. A comma-grouped number stays
// whole because the digit alternative runs before the single-character one.
// And `o'clock` splits in the middle of a syllable, because the word
// alternative takes letters greedily and then apostrophe-letter pairs — which
// is upstream's behaviour, reproduced rather than fixed: the merge loop puts
// it back together when the dictionary answers for the whole word, and a port
// that "improved" this would take a different path through that loop.
func Subtokenize(word string) []string {
	rs := []rune(word)
	var out []string
	emit := func(from, to int) int {
		out = append(out, string(rs[from:to]))
		return to
	}
	at := func(i int) rune {
		if i < 0 || i >= len(rs) {
			return 0
		}
		return rs[i]
	}
	isQuote := func(r rune) bool { return r == '\'' || r == '‘' || r == '’' }

	for i := 0; i < len(rs); {
		// 1. Leading apostrophes, at the start of the word only.
		if i == 0 && isQuote(rs[0]) {
			j := 0
			for j < len(rs) && isQuote(rs[j]) {
				j++
			}
			i = emit(0, j)
			continue
		}
		// 2. An uppercase letter followed by an uppercase and then a
		//    lowercase, which is where an initialism ends and a word begins.
		if unicode.IsUpper(rs[i]) && unicode.IsUpper(at(i+1)) && unicode.IsLower(at(i+2)) {
			i = emit(i, i+1)
			continue
		}
		// 3. A number: runs of digits with a single comma or point between.
		if j := scanNumber(rs, i); j > i {
			i = emit(i, j)
			continue
		}
		// 4. A run of hyphens or underscores.
		if rs[i] == '-' || rs[i] == '_' {
			j := i
			for j < len(rs) && (rs[j] == '-' || rs[j] == '_') {
				j++
			}
			i = emit(i, j)
			continue
		}
		// 5. Two or more apostrophes.
		if isQuote(rs[i]) && isQuote(at(i+1)) {
			j := i
			for j < len(rs) && isQuote(rs[j]) {
				j++
			}
			i = emit(i, j)
			continue
		}
		// 6. Lazily, letters ending at a lowercase that is followed by an
		//    uppercase — the camel-case cut. Lazy, so the *shortest* such
		//    prefix wins: `macOS` gives `mac`, not `macO`.
		if j := scanCamel(rs, i, isQuote); j > i {
			i = emit(i, j)
			continue
		}
		// 7. A word: letters, then any number of apostrophe-letter pairs.
		if unicode.IsLetter(rs[i]) {
			j := i
			for j < len(rs) && unicode.IsLetter(rs[j]) {
				j++
			}
			for j+1 < len(rs) && isQuote(rs[j]) && unicode.IsLetter(rs[j+1]) {
				j += 2
			}
			i = emit(i, j)
			continue
		}
		// 9. Trailing apostrophes. (8 would match them one at a time, and the
		//    two are indistinguishable at the end of a word.)
		// 8. Anything else, one character at a time.
		i = emit(i, i+1)
	}
	return out
}

// scanNumber is `(?:^-)?(?:\d?[,.]?\d)+`: a leading minus only at the start of
// the word, then runs of digits that may carry a single separator between any
// two of them. It is what keeps `1,024,000` and `3.50` whole.
func scanNumber(rs []rune, i int) int {
	j := i
	if i == 0 && j < len(rs) && rs[j] == '-' {
		j++
	}
	start := j
	for {
		n := digitGroup(rs, j)
		if n == j {
			break
		}
		j = n
	}
	if j == start {
		return i // a minus on its own is not a number
	}
	return j
}

// digitGroup is one repetition of `\d?[,.]?\d`, with the two optional parts
// tried in the order a backtracking engine would: greedy first, then giving
// each of them up in turn.
//
// Writing the loop without this was the one bug the table caught. Taking the
// leading digit greedily and then demanding another leaves `3.50` as `3.5`
// and `-5` as `-` and `5`, because the last digit of a run has to be matched
// by a repetition whose optional parts are both empty.
func digitGroup(rs []rune, j int) int {
	for _, take := range [4][2]bool{{true, true}, {true, false}, {false, true}, {false, false}} {
		k := j
		if take[0] {
			if k >= len(rs) || !isASCIIDigit(rs[k]) {
				continue
			}
			k++
		}
		if take[1] {
			if k >= len(rs) || (rs[k] != ',' && rs[k] != '.') {
				continue
			}
			k++
		}
		if k >= len(rs) || !isASCIIDigit(rs[k]) {
			continue
		}
		return k + 1
	}
	return j
}

func isASCIIDigit(r rune) bool { return r >= '0' && r <= '9' }

// scanCamel is `\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})`, lazily: the shortest
// run of letters (optionally with apostrophe-letter pairs) that ends on a
// lowercase immediately before an uppercase.
func scanCamel(rs []rune, i int, isQuote func(rune) bool) int {
	for j := i; j < len(rs); j++ {
		r := rs[j]
		if unicode.IsLower(r) && j+1 < len(rs) && unicode.IsUpper(rs[j+1]) {
			return j + 1
		}
		if unicode.IsLetter(r) {
			continue
		}
		// An apostrophe may continue the run only if a letter follows it.
		if isQuote(r) && j+1 < len(rs) && unicode.IsLetter(rs[j+1]) {
			j++
			continue
		}
		return i
	}
	return i
}
