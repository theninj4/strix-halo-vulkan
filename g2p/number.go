package g2p

import (
	"strconv"
	"strings"
)

// The English number names, in `num2words`' own grouping: the twenty forms
// that are single words, the tens, and the scales.
//
// misaki spells a number by calling num2words and then splitting the result on
// anything that is not a lowercase letter, so what has to be reproduced is the
// word sequence — and the separators too, because one of them is the word
// "and", which a token's flags can keep.
var (
	lowNumWords = []string{
		"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
		"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
		"sixteen", "seventeen", "eighteen", "nineteen", "twenty",
	}
	tensNumWords = []string{
		"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
		"eighty", "ninety",
	}
	// Short-scale names, as `set_high_numwords` builds them: 10^6 up.
	scaleNames = []string{
		"", "thousand", "million", "billion", "trillion", "quadrillion",
		"quintillion", "sextillion", "septillion", "octillion", "nonillion",
		"decillion",
	}
	// The irregular ordinals; everything else takes -th, with a final y
	// becoming -ie first.
	ordinals = map[string]string{
		"one": "first", "two": "second", "three": "third", "four": "fourth",
		"five": "fifth", "six": "sixth", "seven": "seventh", "eight": "eighth",
		"nine": "ninth", "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth",
	}
)

// Cardinal is `num2words(n)`: the ordinary spelling of an integer.
//
// The joins are the whole of it, and they are not uniform. Inside a group of
// three, hundreds and a remainder below 100 are joined by " and " and tens and
// units by a hyphen; between groups the join is ", " unless what follows is
// itself below 100, in which case it is " and " again. So 1001 is "one
// thousand and one" and 1100 is "one thousand, one hundred" — the rule is
// about the *value* on the right, not about the position.
func Cardinal(n int64) string {
	if n < 0 {
		return "minus " + Cardinal(-n)
	}
	if n < 1000 {
		return underThousand(n)
	}
	// Groups of three, most significant first, with their scale.
	var groups []int64
	for v := n; v > 0; v /= 1000 {
		groups = append(groups, v%1000)
	}
	var out string
	for i := len(groups) - 1; i >= 0; i-- {
		g := groups[i]
		if g == 0 {
			continue
		}
		piece := underThousand(g)
		if s := scaleNames[i]; s != "" {
			piece += " " + s
		}
		if out == "" {
			out = piece
			continue
		}
		// The value of everything from this group down, which is what
		// num2words' merge rule tests — the join is about what is on the
		// right, not about which group it is.
		if rest := n % pow1000(int64(i+1)); rest < 100 {
			out += " and " + piece
		} else {
			out += ", " + piece
		}
	}
	return out
}

func pow1000(k int64) int64 {
	v := int64(1)
	for ; k > 0; k-- {
		v *= 1000
	}
	return v
}

// underThousand spells 0..999.
func underThousand(n int64) string {
	if n >= 100 {
		out := lowNumWords[n/100] + " hundred"
		if r := n % 100; r > 0 {
			out += " and " + underHundred(r)
		}
		return out
	}
	return underHundred(n)
}

func underHundred(n int64) string {
	if n <= 20 {
		return lowNumWords[n]
	}
	out := tensNumWords[n/10]
	if u := n % 10; u > 0 {
		out += "-" + lowNumWords[u]
	}
	return out
}

// Ordinal is `num2words(n, to='ordinal')`: the cardinal with its last word
// made ordinal.
//
// The last *word* is found after splitting on spaces and then on hyphens, so
// "twenty-four" becomes "twenty-fourth" and only the tail changes. A final `y`
// becomes `ie` — "twenty" is "twentieth" — and everything the irregular table
// does not cover takes a plain -th.
func Ordinal(n int64) string {
	words := strings.Split(Cardinal(n), " ")
	last := words[len(words)-1]
	parts := strings.Split(last, "-")
	w := strings.ToLower(parts[len(parts)-1])
	if o, ok := ordinals[w]; ok {
		w = o
	} else {
		if strings.HasSuffix(w, "y") {
			w = w[:len(w)-1] + "ie"
		}
		w += "th"
	}
	parts[len(parts)-1] = w
	words[len(words)-1] = strings.Join(parts, "-")
	return strings.Join(words, " ")
}

// Year is `num2words(n, to='year')`: the way a year is said rather than
// counted — "nineteen ninety-nine", not "one thousand nine hundred and
// ninety-nine".
//
// Three cases fall back to the cardinal, and the middle one is the
// non-obvious: a year whose century is a round multiple of ten *and* whose
// last two digits are under ten is said as a number, which is why 2001 is
// "two thousand and one" and 1901 is "nineteen oh-one".
func Year(n int64) string {
	suffix := ""
	if n < 0 {
		n, suffix = -n, " BC"
	}
	high, low := n/100, n%100
	if high == 0 || (high%10 == 0 && low < 10) || high >= 100 {
		return Cardinal(n) + suffix
	}
	var lowText string
	switch {
	case low == 0:
		lowText = "hundred"
	case low < 10:
		lowText = "oh-" + Cardinal(low)
	default:
		lowText = Cardinal(low)
	}
	return Cardinal(high) + " " + lowText + suffix
}

// Decimal is `num2words(float)`: the integer part as a cardinal, then "point",
// then the fraction **digit by digit**.
//
// Spelling the fraction as a number would be wrong in a way that matters here:
// 1.25 is "one point two five", not "one point twenty-five", and the two
// differ in syllable count as well as in words.
func Decimal(s string) string {
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	intPart, frac, hasPoint := strings.Cut(s, ".")
	if intPart == "" {
		intPart = "0"
	}
	n, err := strconv.ParseInt(intPart, 10, 64)
	if err != nil {
		return ""
	}
	out := Cardinal(n)
	// num2words drops a trailing zero fraction: 7.0 is "seven".
	frac = strings.TrimRight(frac, "0")
	if hasPoint && frac != "" {
		var b strings.Builder
		b.WriteString(out)
		b.WriteString(" point")
		for _, d := range frac {
			b.WriteString(" ")
			b.WriteString(lowNumWords[d-'0'])
		}
		out = b.String()
	}
	if neg {
		out = "minus " + out
	}
	return out
}

// numWords splits a spelled-out number the way misaki does: on any run of
// characters that is not a lowercase letter, so hyphens, commas and spaces all
// separate and nothing else survives.
func numWords(s string) []string {
	out := []string{}
	cur := strings.Builder{}
	for _, r := range s {
		if r >= 'a' && r <= 'z' {
			cur.WriteRune(r)
			continue
		}
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}
