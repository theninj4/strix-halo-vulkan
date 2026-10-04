package decide

// JSON the way Python holds it, because the model was trained on Python's
// rendering of it.
//
// Decisions v1 renders the state, and any non-string instructions or option
// description, with `json.dumps(x, ensure_ascii=False)` over what
// `json.loads` returned. Three things Go's own decoder throws away decide
// that text:
//
//   - **Key order.** A Python dict keeps insertion order and a Go map does
//     not. A repeated key keeps its first position and takes the last value
//     (Python's dict assignment).
//   - **Int against float.** `json.loads` makes `2` an int and `2.0` a float,
//     and `json.dumps` writes them back as `2` and `2.0`; `1E5` comes back as
//     `100000.0`. So a number is kept as its literal and written by the rule
//     its literal implies: an int as its exact decimal (any width), a float
//     as Python's repr.
//   - **Escapes.** `ensure_ascii=False` writes every rune as itself except
//     `"`, `\` and the C0 controls; Go escapes `<`, `>`, `&`, U+2028 and
//     U+2029 as well.
//
// This is a copy of `kev/value.go`'s decoder, which goes when Kev does
// (`research/rune-vertical.md` R9), plus the duplicate-key record v1 needs:
// a question named twice, or an option key sent twice, is refused rather
// than resolved.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// Kind is which of JSON's six shapes a Value holds.
type Kind int

const (
	Null Kind = iota
	Bool
	Number
	String
	Array
	Object
)

// Value is a decoded JSON value with its object keys in document order and
// its numbers as their literal text.
type Value struct {
	Kind Kind
	B    bool
	Num  string // the literal, e.g. "2", "2.0", "1e21"
	S    string
	Arr  []Value
	Keys []string // object keys, first-occurrence order
	Vals []Value  // object values, parallel to Keys
	Dup  string   // an object's first repeated key, "" if none
}

// Get returns an object's value for key, and whether it is there.
func (v Value) Get(key string) (Value, bool) {
	for i, k := range v.Keys {
		if k == key {
			return v.Vals[i], true
		}
	}
	return Value{}, false
}

// ParseValue decodes one JSON document.
func ParseValue(b []byte) (Value, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return Value{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Value{}, fmt.Errorf("decide: trailing data after a JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (Value, error) {
	t, err := dec.Token()
	if err != nil {
		return Value{}, err
	}
	switch t := t.(type) {
	case nil:
		return Value{Kind: Null}, nil
	case bool:
		return Value{Kind: Bool, B: t}, nil
	case json.Number:
		return Value{Kind: Number, Num: string(t)}, nil
	case string:
		return Value{Kind: String, S: t}, nil
	case json.Delim:
		switch t {
		case '[':
			v := Value{Kind: Array, Arr: []Value{}}
			for dec.More() {
				x, err := decodeValue(dec)
				if err != nil {
					return Value{}, err
				}
				v.Arr = append(v.Arr, x)
			}
			_, err := dec.Token() // ']'
			return v, err
		case '{':
			v := Value{Kind: Object, Keys: []string{}, Vals: []Value{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return Value{}, err
				}
				k := kt.(string)
				x, err := decodeValue(dec)
				if err != nil {
					return Value{}, err
				}
				replaced := false
				for i, have := range v.Keys {
					if have == k {
						v.Vals[i], replaced = x, true
						if v.Dup == "" {
							v.Dup = k
						}
						break
					}
				}
				if !replaced {
					v.Keys = append(v.Keys, k)
					v.Vals = append(v.Vals, x)
				}
			}
			_, err := dec.Token() // '}'
			return v, err
		}
	}
	return Value{}, fmt.Errorf("decide: unexpected JSON token %v", t)
}

// isFloatLiteral is whether json.loads makes this literal a float: it has a
// fraction or an exponent. Everything else is an int.
func isFloatLiteral(num string) bool { return strings.ContainsAny(num, ".eE") }

// Dumps is `json.dumps(v, ensure_ascii=False)`.
func Dumps(v Value) string {
	var b strings.Builder
	dumps(&b, v, false)
	return b.String()
}

// dumps writes v; legend is v1's legend rule, under which an integer
// outside [-2^63, 2^64-1] is written as the nearest double.
func dumps(b *strings.Builder, v Value, legend bool) {
	switch v.Kind {
	case Null:
		b.WriteString("null")
	case Bool:
		if v.B {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case Number:
		b.WriteString(pyNumber(v.Num, legend))
	case String:
		pyString(b, v.S)
	case Array:
		b.WriteByte('[')
		for i, x := range v.Arr {
			if i > 0 {
				b.WriteString(", ")
			}
			dumps(b, x, legend)
		}
		b.WriteByte(']')
	case Object:
		b.WriteByte('{')
		for i, k := range v.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			pyString(b, k)
			b.WriteString(": ")
			dumps(b, v.Vals[i], legend)
		}
		b.WriteByte('}')
	}
}

var (
	minInt64  = big.NewInt(math.MinInt64)
	maxUint64 = new(big.Int).SetUint64(math.MaxUint64)
)

// pyNumber is json.dumps of what json.loads made of the literal num.
func pyNumber(num string, legend bool) string {
	if !isFloatLiteral(num) {
		n, ok := new(big.Int).SetString(num, 10)
		if !ok {
			return num
		}
		if legend && (n.Cmp(minInt64) < 0 || n.Cmp(maxUint64) > 0) {
			f, _ := new(big.Float).SetInt(n).Float64()
			return pyFloatJSON(f)
		}
		return n.String()
	}
	f, _ := strconv.ParseFloat(num, 64) // out of range gives ±Inf, as Python's float() does
	return pyFloatJSON(f)
}

// pyFloatJSON is json.dumps of a float: repr, except the non-finite values,
// which json.dumps spells as JavaScript does.
func pyFloatJSON(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}
	return PyFloatRepr(f)
}

// PyFloatRepr is Python's repr() of a finite float: the shortest digits that
// round trip, fixed notation when the decimal point falls in (-4, 16],
// scientific with a signed exponent of at least two digits otherwise, and
// ".0" on an integral fixed value. repr(1e16) is "1e+16"; repr(1e15) is
// "1000000000000000.0"; repr(1e-05) is "1e-05"; repr(0.0001) is "0.0001".
func PyFloatRepr(f float64) string {
	sign := ""
	if math.Signbit(f) {
		sign, f = "-", -f
	}
	if f == 0 {
		return sign + "0.0"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // "d.ddde±XX"
	mant, expStr, _ := strings.Cut(e, "e")
	digits := strings.Replace(mant, ".", "", 1)
	exp, _ := strconv.Atoi(expStr)
	decpt := exp + 1 // digits are 0.d1d2... x 10^decpt
	if decpt > -4 && decpt <= 16 {
		switch {
		case decpt <= 0:
			return sign + "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			return sign + digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			return sign + digits[:decpt] + "." + digits[decpt:]
		}
	}
	m := digits[:1]
	if len(digits) > 1 {
		m += "." + digits[1:]
	}
	es := "+"
	if exp < 0 {
		es, exp = "-", -exp
	}
	return fmt.Sprintf("%s%se%s%02d", sign, m, es, exp)
}

// pyString writes one JSON string with Python's escaping: the five short
// escapes, `\u00XX` for the remaining C0 controls, and every other rune --
// DEL, U+2028 and `<` included -- as itself, which is what ensure_ascii=False
// means.
func pyString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}
