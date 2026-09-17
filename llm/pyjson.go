package llm

// Python's json.dumps, because the chat template's `tojson` is Python's.
//
// This is the kind of detail that decides whether a transcription is exact or
// merely close. The template writes a tool's schema with `tool | tojson`, and
// transformers binds that filter to `json.dumps(x, ensure_ascii=False)` --
// whose separators are `", "` and `": "`, with the spaces. Go's
// `encoding/json` writes neither space, escapes `<`, `>` and `&` as `<`
// and friends, and escapes U+2028 and U+2029. Four differences, every one of
// them a different prompt from the one the model was trained on, and none of
// them visible in a diff read quickly.
//
// Numbers are re-emitted **as they arrived** rather than re-formatted, which
// sidesteps the fifth difference: Go's shortest-round-trip float formatting
// and Python's `repr` do not always agree, and neither of them has any
// business rewriting a literal the client typed.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// pyJSON re-encodes raw the way Python's json.dumps(ensure_ascii=False)
// would, preserving object key order.
func pyJSON(raw json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var b strings.Builder
	if err := pyValue(dec, &b); err != nil {
		return "", err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return "", fmt.Errorf("llm: trailing content after a JSON value")
	}
	return b.String(), nil
}

// pyValue writes the next value from dec. It is one token unless that token
// opens an object or an array, in which case it is that whole structure.
func pyValue(dec *json.Decoder, b *strings.Builder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	return pyToken(dec, b, tok)
}

func pyToken(dec *json.Decoder, b *strings.Builder, tok json.Token) error {
	switch v := tok.(type) {
	case json.Delim:
		switch v {
		case '{':
			b.WriteByte('{')
			for i := 0; dec.More(); i++ {
				key, err := dec.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok {
					return fmt.Errorf("llm: object key is not a string")
				}
				if i > 0 {
					b.WriteString(", ")
				}
				pyString(b, name)
				b.WriteString(": ")
				if err := pyValue(dec, b); err != nil {
					return err
				}
			}
			if _, err := dec.Token(); err != nil { // the closing brace
				return err
			}
			b.WriteByte('}')
		case '[':
			b.WriteByte('[')
			for i := 0; dec.More(); i++ {
				if i > 0 {
					b.WriteString(", ")
				}
				if err := pyValue(dec, b); err != nil {
					return err
				}
			}
			if _, err := dec.Token(); err != nil { // the closing bracket
				return err
			}
			b.WriteByte(']')
		default:
			return fmt.Errorf("llm: unexpected %q", v)
		}
	case string:
		pyString(b, v)
	case json.Number:
		b.WriteString(v.String())
	case bool:
		// Python writes them lower case, which is JSON's spelling too.
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	default:
		return fmt.Errorf("llm: unexpected token %T", tok)
	}
	return nil
}

// pyString writes one JSON string with Python's escaping: the five
// short escapes, `\uXXXX` for the remaining control characters, and every
// other rune -- including `<`, `&` and U+2028 -- written out as itself,
// which is what ensure_ascii=False means.
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
			switch {
			case r < 0x20:
				fmt.Fprintf(b, `\u%04x`, r)
			case r == utf8.RuneError:
				// A byte that is not UTF-8 at all. Python would have
				// refused the input; the closest thing to that here is the
				// replacement character the range loop already produced.
				b.WriteRune(r)
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}
