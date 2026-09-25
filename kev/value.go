package kev

// JSON the way Python holds it, because the model reads Python's rendering.
//
// Every free-form field of a System One request (the state, a question's
// instructions, an option's description, a score level) may be a string,
// an object, an array, a number, a bool or null. Kev turns it into the text
// the model reads with `api.render`, which is Python's `str()` over what
// `json.loads` returned. Two things Go's own decoder throws away decide
// that text:
//
//   - **Key order.** A Python dict keeps insertion order and a Go map does
//     not. An object state renders as its keys in the order the caller wrote
//     them, and a choice's options are its criteria in that order, so the
//     order is the model's input and also the answer's slot order.
//   - **Int against float.** `json.loads` gives `2` an int and `2.0` a
//     float, and `str()` renders them as `2` and `2.0`. A float64 cannot
//     tell them apart, so a number is kept as its literal and rendered by
//     the rule its literal implies.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode"
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

// Len is the number of elements of an array or keys of an object.
func (v Value) Len() int {
	if v.Kind == Array {
		return len(v.Arr)
	}
	return len(v.Keys)
}

// UnmarshalJSON decodes one value.
func (v *Value) UnmarshalJSON(b []byte) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	got, err := decodeValue(dec)
	if err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return fmt.Errorf("kev: trailing data after a JSON value")
	}
	*v = got
	return nil
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
				// A repeated key is Python's dict assignment: the last value
				// wins and the key keeps its first position.
				replaced := false
				for i, have := range v.Keys {
					if have == k {
						v.Vals[i], replaced = x, true
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
	return Value{}, fmt.Errorf("kev: unexpected JSON token %v", t)
}

// isFloatLiteral is whether json.loads makes this literal a float: it has a
// fraction or an exponent. Everything else is an int.
func isFloatLiteral(num string) bool { return strings.ContainsAny(num, ".eE") }

// pyStr is Python's str() of what json.loads returns for a scalar. It is
// only called on scalars.
func (v Value) pyStr() string {
	switch v.Kind {
	case Null:
		return "None"
	case Bool:
		if v.B {
			return "True"
		}
		return "False"
	case String:
		return v.S
	case Number:
		if !isFloatLiteral(v.Num) {
			// Python ints are arbitrary precision; str() is the canonical
			// decimal, which for a JSON int literal is itself except "-0".
			n, ok := new(big.Int).SetString(v.Num, 10)
			if !ok {
				return v.Num
			}
			return n.String()
		}
		f, err := strconv.ParseFloat(v.Num, 64)
		if err != nil && !math.IsInf(f, 0) {
			return v.Num
		}
		return PyFloatRepr(f)
	}
	return ""
}

// PyFloatRepr is Python's repr() of a float: the shortest digits that round
// trip, fixed notation when the decimal point falls in (-4, 16], scientific
// with a signed exponent of at least two digits otherwise, and ".0" on an
// integral fixed value. repr(1e16) is "1e+16"; repr(1e15) is
// "1000000000000000.0"; repr(1e-05) is "1e-05"; repr(0.0001) is "0.0001".
func PyFloatRepr(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
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

// Render is Kev's `api.render`: the text the model reads for a JSON value.
//
//	None                -> ""
//	str/int/float/bool  -> str(v)
//	list                -> "\n".join(f"{pad}- {render(x, indent+1).lstrip()}" ...)
//	dict                -> "\n".join(f"{pad}{k}:\n{render(x, indent+1)}" if x is a dict/list
//	                                 else f"{pad}{k}: {render(x)}" ...)
//
// with pad two spaces per level. Note what that does and does not do: a
// scalar inside a dict is rendered at indent 0 (it has no pad of its own),
// a list item is lstripped of *all* leading whitespace, including a string
// item's own, and an empty list or dict is "".
func Render(v Value) string { return render(v, 0) }

func render(v Value, indent int) string {
	pad := strings.Repeat("  ", indent)
	switch v.Kind {
	case Null:
		return ""
	case Array:
		parts := make([]string, len(v.Arr))
		for i, x := range v.Arr {
			parts[i] = pad + "- " + pyLstrip(render(x, indent+1))
		}
		return strings.Join(parts, "\n")
	case Object:
		parts := make([]string, len(v.Keys))
		for i, k := range v.Keys {
			x := v.Vals[i]
			if x.Kind == Object || x.Kind == Array {
				parts[i] = pad + k + ":\n" + render(x, indent+1)
			} else {
				parts[i] = pad + k + ": " + render(x, 0)
			}
		}
		return strings.Join(parts, "\n")
	}
	return v.pyStr()
}

// pyLstrip is Python's str.lstrip() with no argument: it strips every
// character for which str.isspace() is true. That is Unicode White_Space
// plus the four information separators U+001C..U+001F, which Go's
// unicode.IsSpace does not count.
func pyLstrip(s string) string {
	return strings.TrimLeftFunc(s, pyIsSpace)
}

func pyIsSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}
