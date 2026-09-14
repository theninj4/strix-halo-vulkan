package tokenizer

import "unicode"

// The pre-tokenizer. `tokenizer.json` states it as a regex with `Isolated`
// behaviour:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Go's regexp is RE2 and has no lookahead, so `\s+(?!\S)` cannot be written
// with it and the whole thing is hand-rolled here. The alternatives are
// tried in source order at each position and the first that matches wins,
// which is what a backtracking engine does and is *not* what a leftmost-
// longest engine would do -- the order is load-bearing, so each alternative
// below keeps its number.
//
// Every character matches at least one alternative (a letter alt 2, a digit
// alt 3, whitespace alt 7, anything else alt 4), so the pieces tile the
// input and `Isolated` never has a gap to hand back.
func split(text string) []string {
	r := []rune(text)
	var out []string
	for i := 0; i < len(r); {
		n := 0
		for _, alt := range alternatives {
			if n = alt(r, i); n > 0 {
				break
			}
		}
		if n == 0 {
			// Unreachable: alternative 4 takes everything the others leave.
			n = 1
		}
		out = append(out, string(r[i:i+n]))
		i += n
	}
	return out
}

// alternatives are the regex's branches, in the order it writes them. A
// backtracking engine takes the first that matches at a position, not the
// longest, so this order is the definition and not an optimisation.
var alternatives = []func(r []rune, i int) int{
	matchContraction,   // 1
	matchWord,          // 2
	matchNumber,        // 3
	matchSymbols,       // 4
	matchNewlines,      // 5
	matchTrailingSpace, // 6
	matchSpace,         // 7
}

// matchNumber is `\p{N}`: one digit at a time, so "2024" is four pieces.
func matchNumber(r []rune, i int) int {
	if isNumber(r[i]) {
		return 1
	}
	return 0
}

// matchSpace is the final `\s+`, reached only when the run holds no line
// break and is a single character before a non-space.
func matchSpace(r []rune, i int) int {
	if !unicode.IsSpace(r[i]) {
		return 0
	}
	return spaceRun(r, i) - i
}

// contractions are the suffixes alternative 1 splits off, matched
// case-insensitively after a literal apostrophe.
var contractions = []string{"s", "t", "re", "ve", "m", "ll", "d"}

func matchContraction(r []rune, i int) int {
	if r[i] != '\'' {
		return 0
	}
	for _, c := range contractions {
		if len(r)-i-1 < len(c) {
			continue
		}
		ok := true
		for j, want := range c {
			if unicode.ToLower(r[i+1+j]) != want {
				ok = false
				break
			}
		}
		if ok {
			return 1 + len(c)
		}
	}
	return 0
}

// matchWord is `[^\r\n\p{L}\p{N}]?\p{L}+`. The optional leading character is
// greedy, so " cat" is one piece and not two -- which is where the leading
// space in most of this vocabulary's tokens comes from.
func matchWord(r []rune, i int) int {
	lead := 0
	if c := r[i]; c != '\r' && c != '\n' && !unicode.IsLetter(c) && !isNumber(c) {
		lead = 1
	}
	for _, skip := range []int{lead, 0} {
		j := i + skip
		for j < len(r) && unicode.IsLetter(r[j]) {
			j++
		}
		if j > i+skip {
			return j - i
		}
		if skip == 0 {
			break
		}
	}
	return 0
}

// matchSymbols is ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
func matchSymbols(r []rune, i int) int {
	space := 0
	if r[i] == ' ' {
		space = 1
	}
	for _, skip := range []int{space, 0} {
		j := i + skip
		for j < len(r) && !unicode.IsSpace(r[j]) && !unicode.IsLetter(r[j]) && !isNumber(r[j]) {
			j++
		}
		if j > i+skip {
			for j < len(r) && (r[j] == '\r' || r[j] == '\n') {
				j++
			}
			return j - i
		}
		if skip == 0 {
			break
		}
	}
	return 0
}

// matchNewlines is `\s*[\r\n]+`. Both quantifiers are greedy, so `\s*` backs
// off to the *last* line break in the whitespace run and `[\r\n]+` then takes
// only it: "  \n  " matches "  \n" and leaves the two trailing spaces.
func matchNewlines(r []rune, i int) int {
	if !unicode.IsSpace(r[i]) {
		return 0
	}
	end := spaceRun(r, i)
	for j := end - 1; j >= i; j-- {
		if r[j] == '\r' || r[j] == '\n' {
			return j + 1 - i
		}
	}
	return 0
}

// matchTrailingSpace is `\s+(?!\S)`: the run, minus its last character
// unless the run ends the input. That last character is what alternative 2
// or 4 then attaches to the token it precedes.
func matchTrailingSpace(r []rune, i int) int {
	if !unicode.IsSpace(r[i]) {
		return 0
	}
	end := spaceRun(r, i)
	if end == len(r) {
		return end - i
	}
	if end-i >= 2 {
		return end - i - 1
	}
	return 0
}

func spaceRun(r []rune, i int) int {
	for i < len(r) && unicode.IsSpace(r[i]) {
		i++
	}
	return i
}

// isNumber is `\p{N}`, the Unicode Number category rather than ASCII 0-9.
func isNumber(c rune) bool { return unicode.IsNumber(c) }
