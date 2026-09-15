package g2p

import (
	"strconv"
	"strings"
)

// currencies is misaki's: the symbol, its major unit and its minor one.
var currencies = map[string][2]string{
	"$": {"dollar", "cent"},
	"£": {"pound", "pence"},
	"€": {"euro", "cent"},
}

var ordinalSuffixes = map[string]bool{"st": true, "nd": true, "rd": true, "th": true}

func isDigit(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// numSuffix is misaki's `[a-z']+$`: the letters trailing a number, which carry
// the ordinal marker and the inflections — `21st`, `1990s`, `1000's`.
func numSuffix(word string) string {
	rs := []rune(word)
	i := len(rs)
	for i > 0 {
		r := rs[i-1]
		if (r >= 'a' && r <= 'z') || r == '\'' {
			i--
			continue
		}
		break
	}
	if i == len(rs) {
		return ""
	}
	return string(rs[i:])
}

// IsNumber is misaki's `is_number`: whether the number path should be tried at
// all, after an inflection or an ordinal marker is taken off the end.
//
// The leading minus is only a minus on the *head* of a group — elsewhere it is
// a hyphen, which is the difference between "minus five" and the second half
// of "twenty-five".
func IsNumber(word string, isHead bool) bool {
	if !strings.ContainsFunc(word, func(r rune) bool { return r >= '0' && r <= '9' }) {
		return false
	}
	for _, s := range []string{"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"} {
		if strings.HasSuffix(word, s) {
			word = word[:len(word)-len(s)]
			break
		}
	}
	for i, c := range word {
		switch {
		case c >= '0' && c <= '9', c == ',', c == '.':
		case isHead && i == 0 && c == '-':
		default:
			return false
		}
	}
	return true
}

// isWholeCurrency is misaki's `is_currency`: whether a decimal can be read as
// an amount at all. Three or more digits after the point is not a price, it is
// a measurement — unless they are all zeros.
func isWholeCurrency(word string) bool {
	if !strings.Contains(word, ".") {
		return true
	}
	if strings.Count(word, ".") > 1 {
		return false
	}
	cents := word[strings.Index(word, ".")+1:]
	if len(cents) < 3 {
		return true
	}
	return strings.Trim(cents, "0") == ""
}

// AppendCurrency names the unit after a number that is not itself an amount —
// "three million dollars", where the amount branch below never ran because the
// number was not a plain one.
func (l *Lexicon) AppendCurrency(ps, currency string) string {
	c, ok := currencies[currency]
	if !ok {
		return ps
	}
	unit, _, ok := l.stemS(c[0]+"s", "", nil, nil)
	if !ok {
		return ps
	}
	return ps + " " + unit
}

// piece is one spelled-out fragment of a number, with the rating of whichever
// dictionary supplied it.
type piece struct {
	ps     string
	rating int
}

// Number is misaki's `get_number`: a digit string spelled out, looked up word
// by word, and glued back together.
//
// Nothing here spells a number the same way twice, and that is the point —
// English says the same digits differently depending on what they are. The
// branches, in the order they are tried:
//
//   - an explicit ordinal marker: `21st` is "twenty-first".
//   - a bare four-digit number that is not an amount: a **year**, so 2024 is
//     "twenty twenty-four" and not "two thousand and twenty-four".
//   - not the head of its group: a bare run of digits, said digit by digit if
//     it starts with a zero or is longer than three, and as a telephone-style
//     "seven-oh-nine" if it is exactly three.
//   - an amount, when a currency symbol preceded it: the two halves become
//     major and minor units, and a zero half is dropped rather than said.
//   - otherwise the ordinary cardinal, ordinal or decimal.
//
// numFlags are misaki's, set by the preprocessor: `&` keeps the word "and",
// `n` glues it onto the previous piece as a schwa, and `a` says a leading
// "one" as "a".
func (l *Lexicon) Number(word, currency string, isHead bool, numFlags string) (string, int, bool) {
	suffix := numSuffix(word)
	word = word[:len(word)-len(suffix)]

	var result []piece
	appendWord := func(w string, stress *float64) {
		ps, rating, ok := l.lookup(w, "", stress, nil)
		if !ok {
			// Every word num2words can emit is in the dictionary; a miss here
			// is a bug rather than an out-of-vocabulary word, and dropping the
			// piece would silently shorten the number.
			return
		}
		result = append(result, piece{ps, rating})
	}
	// extendNum spells one fragment and looks up each of its words. `escape`
	// means the fragment is already words rather than digits.
	extendNum := func(num string, first, escape bool) {
		text := num
		if !escape {
			n, err := strconv.ParseInt(num, 10, 64)
			if err != nil {
				return
			}
			text = Cardinal(n)
		}
		splits := numWords(text)
		for i, w := range splits {
			if w != "and" || strings.Contains(numFlags, "&") {
				if first && i == 0 && len(splits) > 1 && w == "one" && strings.Contains(numFlags, "a") {
					result = append(result, piece{"ə", RatingGold})
					continue
				}
				var stress *float64
				if w == "point" {
					stress = f64(-2)
				}
				appendWord(w, stress)
			} else if w == "and" && strings.Contains(numFlags, "n") && len(result) > 0 {
				result[len(result)-1].ps += "ən"
			}
		}
	}
	digitByDigit := func(num string) {
		for _, d := range num {
			extendNum(string(d), false, false)
		}
	}

	if strings.HasPrefix(word, "-") {
		appendWord("minus", nil)
		word = word[1:]
	}
	_, isKnownCurrency := currencies[currency]
	plain := strings.ReplaceAll(word, ",", "")

	switch {
	case isDigit(word) && ordinalSuffixes[suffix]:
		n, err := strconv.ParseInt(word, 10, 64)
		if err != nil {
			return "", 0, false
		}
		extendNum(Ordinal(n), true, true)

	case len(result) == 0 && len([]rune(word)) == 4 && !isKnownCurrency && isDigit(word):
		n, _ := strconv.ParseInt(word, 10, 64)
		extendNum(Year(n), true, true)

	case !isHead && !strings.Contains(word, "."):
		switch {
		case plain == "" || plain[0] == '0' || len(plain) > 3:
			digitByDigit(plain)
		case len(plain) == 3 && !strings.HasSuffix(plain, "00"):
			extendNum(plain[:1], true, false)
			if plain[1] == '0' {
				// "709" is "seven oh nine": the middle zero is the letter,
				// stress stripped, not the number.
				appendWord("O", f64(-2))
				extendNum(plain[2:3], false, false)
			} else {
				extendNum(plain[1:], false, false)
			}
		default:
			extendNum(plain, true, false)
		}

	case strings.Count(word, ".") > 1 || !isHead:
		first := true
		for _, num := range strings.Split(plain, ".") {
			switch {
			case num == "":
			case num[0] == '0' || (len(num) != 2 && strings.Trim(num[1:], "0") != ""):
				digitByDigit(num)
			default:
				extendNum(num, first, false)
			}
			first = false
		}

	case isKnownCurrency && isWholeCurrency(word):
		units := currencies[currency]
		parts := strings.Split(plain, ".")
		type amount struct {
			n    int64
			unit string
		}
		var pairs []amount
		for i, p := range parts {
			if i >= len(units) {
				break
			}
			var n int64
			if p != "" {
				n, _ = strconv.ParseInt(p, 10, 64)
			}
			pairs = append(pairs, amount{n, units[i]})
		}
		if len(pairs) > 1 {
			// "$3.00" is three dollars, not three dollars and zero cents; and
			// "$0.99" is ninety-nine cents.
			if pairs[1].n == 0 {
				pairs = pairs[:1]
			} else if pairs[0].n == 0 {
				pairs = pairs[1:]
			}
		}
		for i, p := range pairs {
			if i > 0 {
				appendWord("and", nil)
			}
			extendNum(strconv.FormatInt(p.n, 10), i == 0, false)
			if abs64(p.n) != 1 && p.unit != "pence" {
				if ps, rating, ok := l.stemS(p.unit+"s", "", nil, nil); ok {
					result = append(result, piece{ps, rating})
				}
			} else {
				appendWord(p.unit, nil)
			}
		}

	default:
		var text string
		switch {
		case isDigit(word):
			n, _ := strconv.ParseInt(word, 10, 64)
			text = Cardinal(n)
		case !strings.Contains(word, "."):
			n, err := strconv.ParseInt(plain, 10, 64)
			if err != nil {
				return "", 0, false
			}
			if ordinalSuffixes[suffix] {
				text = Ordinal(n)
			} else {
				text = Cardinal(n)
			}
		case strings.HasPrefix(plain, "."):
			// A bare fraction: "point one two five", each digit its own word.
			var b strings.Builder
			b.WriteString("point")
			for _, d := range plain[1:] {
				b.WriteString(" ")
				b.WriteString(lowNumWords[d-'0'])
			}
			text = b.String()
		default:
			text = Decimal(plain)
		}
		extendNum(text, true, true)
	}

	if len(result) == 0 {
		return "", 0, false
	}
	parts := make([]string, len(result))
	rating := RatingGold
	for i, p := range result {
		parts[i] = p.ps
		if p.rating < rating {
			rating = p.rating
		}
	}
	out := strings.Join(parts, " ")
	switch suffix {
	case "s", "'s":
		out, _ = l.suffixS(out)
	case "ed", "'d":
		out, _ = l.suffixEd(out)
	case "ing":
		out, _ = l.suffixIng(out)
	}
	return out, rating, true
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}
