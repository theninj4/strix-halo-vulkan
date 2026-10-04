package decide

// POST /v1/systemone on a decisions model: TypeSafe's System One contract,
// as Kev served it, translated onto decisions v1 so clients written against
// Kev (the TypeSafe SDK among them) keep working when Rune replaces it
// (`research/rune-vertical.md` R-o1, the user's call on 2026-10-03).
//
// The two contracts ask the same three kinds of question (noul, choice,
// score) about one state. What differs, and how the translation settles it:
//
//   - **Request rules.** System One is looser: any state (null included),
//     optional instructions, a noul's criteria absent, null or with either
//     side missing. Each is mapped onto the nearest v1 request: the state
//     is rendered with json.dumps whatever it is, missing instructions are
//     "", and a noul's missing side takes v1's default description (the
//     key itself). Refusals are 422, as Kev answered them.
//   - **The prompt is v1's, not Kev's.** Kev rendered the state with its own
//     YAML-ish `render` and a noul as [no, yes]; Rune was trained on v1's
//     prompt, so that is what it sees. The answer's meaning is unchanged:
//     noul is p(true), choice the most probable key, score the expected
//     level.
//   - **Numbers.** System One reports every number rounded to four
//     decimals (Kev's `round(x, 4)`, which keeps a 255-option distribution's
//     rounded sum within TypeSafe's |sum - 1| < 0.02), computed from the
//     unrounded distribution. The confidences are v1's: TypeSafe's published
//     `score_confidence` replaces Kev's approximation of it, and choice
//     confidence is the same formula.
//   - **The envelope.** `{"model", "answers", "usage", "latency_ms"}`, with
//     the answers' keys in Kev's order (a score's legend before its
//     probabilities, confidence last) and a score's legend values as the
//     rendered level texts.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// SystemOneDefaultModel is what a request that names no model gets.
const SystemOneDefaultModel = "kev-latest"

// SystemOneError is a request System One refuses with a 422.
type SystemOneError struct{ Msg string }

func (e *SystemOneError) Error() string { return e.Msg }

func unprocessable(format string, a ...any) error {
	return &SystemOneError{Msg: fmt.Sprintf(format, a...)}
}

// ParseSystemOne reads a System One body into the decisions request that
// answers it. The request's Model is the name the caller sent, for the
// response; the backend serves it with whatever model it has.
func ParseSystemOne(body []byte) (*Request, error) {
	root, err := ParseValue(body)
	if err != nil {
		return nil, unprocessable("body is not JSON: %v", err)
	}
	if root.Kind != Object {
		return nil, unprocessable("body must be a JSON object")
	}
	r := &Request{Model: SystemOneDefaultModel}
	state, ok := root.Get("state")
	if !ok {
		return nil, unprocessable("state is required")
	}
	r.State = state
	r.StateText = StatePrefix + Dumps(state) + "\n\n"
	if m, ok := root.Get("model"); ok {
		if m.Kind != String {
			return nil, unprocessable("model must be a string")
		}
		r.Model = m.S
	}
	// Images (research/rune-vertical.md R10): decisions v1's extension,
	// same forms and meaning, ahead of the text in every question's turn.
	// Not in TypeSafe's contract; a client that does not send it is
	// unaffected.
	if r.Images, err = parseImages(root); err != nil {
		return nil, err
	}
	qs, ok := root.Get("questions")
	if !ok || qs.Kind != Object {
		return nil, unprocessable("questions must be an object of id -> question")
	}
	if len(qs.Keys) == 0 {
		return nil, unprocessable("questions must not be empty")
	}
	for i, id := range qs.Keys {
		q, err := systemOneQuestion(id, qs.Vals[i])
		if err != nil {
			return nil, err
		}
		r.Questions = append(r.Questions, q)
	}
	return r, nil
}

func systemOneQuestion(id string, v Value) (Question, error) {
	q := Question{Name: id}
	if v.Kind != Object {
		return q, unprocessable("questions.%s must be an object", id)
	}
	t, ok := v.Get("type")
	switch {
	case ok && t.Kind == String && t.S == "noul":
		q.Type = Noul
	case ok && t.Kind == String && t.S == "choice":
		q.Type = Choice
	case ok && t.Kind == String && t.S == "score":
		q.Type = Score
	default:
		return q, unprocessable("questions.%s.type must be one of \"noul\", \"choice\", \"score\"", id)
	}
	if ins, ok := v.Get("instructions"); ok && ins.Kind != Null {
		q.Instructions = text(ins)
	}
	add := func(key string, v Value) {
		q.Keys = append(q.Keys, key)
		q.Texts = append(q.Texts, text(v))
		q.Values = append(q.Values, v)
	}
	c, has := v.Get("criteria")
	switch q.Type {
	case Noul:
		if has && c.Kind != Null && c.Kind != Object {
			return q, unprocessable("questions.%s.criteria must be an object with optional \"true\" and \"false\"", id)
		}
		for _, side := range []string{"false", "true"} {
			d, ok := c.Get(side)
			if !ok || d.Kind == Null {
				d = Value{Kind: String, S: side}
			}
			add(side, d)
		}
	case Choice:
		if !has || c.Kind != Object {
			return q, unprocessable("questions.%s.criteria must be an object of option -> description", id)
		}
		if n := len(c.Keys); n < MinOptions || n > MaxOptions {
			return q, unprocessable("questions.%s.criteria must have 1..%d options, got %d", id, MaxOptions, n)
		}
		for i, k := range c.Keys {
			d := c.Vals[i]
			if d.Kind == Null || (d.Kind == String && d.S == "") {
				d = Value{Kind: String, S: k}
			}
			add(k, d)
		}
	case Score:
		if !has || c.Kind != Array {
			return q, unprocessable("questions.%s.criteria must be an array of levels, lowest first", id)
		}
		if n := len(c.Arr); n < MinOptions || n > MaxOptions {
			return q, unprocessable("questions.%s.criteria must have 1..%d levels, got %d", id, MaxOptions, n)
		}
		for i, l := range c.Arr {
			add(strconv.Itoa(i), l)
		}
	}
	return q, nil
}

// roundProb is Python's `round(float(x), 4)`: correctly rounded from the
// exact binary value, which strconv's 'f' formatting also is.
func roundProb(x float64) float64 {
	r, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	return r
}

// SystemOneAnswers writes the answers object. With python set it is
// `json.dumps(answers)` (", " and ": ", ASCII-only), the string whose tokens
// `usage.output_tokens` counts; without it, compact UTF-8 for the body.
func SystemOneAnswers(answers []Answer, python bool) string {
	w := &s1Writer{python: python}
	w.b.WriteByte('{')
	for i := range answers {
		a := &answers[i]
		q := a.Q
		w.sep(i)
		w.key(q.Name)
		w.b.WriteByte('{')
		w.key("type")
		w.str(q.Type.String())
		w.sep(1)
		switch q.Type {
		case Noul:
			w.key("noul")
			w.float(roundProb(a.Value))
		case Choice:
			w.key("choice")
			w.str(q.Keys[a.Choice])
			w.sep(1)
			w.key("confidence")
			w.float(roundProb(a.Confidence))
			w.sep(1)
			w.key("probabilities")
			w.dist(q.Keys, a.Probabilities)
		case Score:
			w.key("score")
			w.float(roundProb(a.Value))
			w.sep(1)
			w.key("legend")
			w.b.WriteByte('{')
			for j, k := range q.Keys {
				w.sep(j)
				w.key(k)
				w.str(q.Texts[j])
			}
			w.b.WriteByte('}')
			w.sep(1)
			w.key("probabilities")
			w.dist(q.Keys, a.Probabilities)
			w.sep(1)
			w.key("confidence")
			w.float(roundProb(a.Confidence))
		}
		w.b.WriteByte('}')
	}
	w.b.WriteByte('}')
	return w.b.String()
}

// SystemOneResponse is the /v1/systemone body.
func SystemOneResponse(model string, answers []Answer, inputTokens, outputTokens int, latencyMS float64) []byte {
	w := &s1Writer{}
	w.b.WriteString(`{"model":`)
	w.str(model)
	w.b.WriteString(`,"answers":`)
	w.b.WriteString(SystemOneAnswers(answers, false))
	fmt.Fprintf(&w.b, `,"usage":{"input_tokens":%d,"output_tokens":%d},"latency_ms":%s}`,
		inputTokens, outputTokens, PyFloatRepr(math.Round(latencyMS*10)/10))
	return []byte(w.b.String())
}

type s1Writer struct {
	b      strings.Builder
	python bool
}

func (w *s1Writer) sep(i int) {
	if i == 0 {
		return
	}
	if w.python {
		w.b.WriteString(", ")
	} else {
		w.b.WriteByte(',')
	}
}

func (w *s1Writer) key(k string) {
	w.str(k)
	if w.python {
		w.b.WriteString(": ")
	} else {
		w.b.WriteByte(':')
	}
}

func (w *s1Writer) float(f float64) { w.b.WriteString(PyFloatRepr(f)) }

func (w *s1Writer) dist(keys []string, p []float64) {
	w.b.WriteByte('{')
	for j, k := range keys {
		w.sep(j)
		w.key(k)
		w.float(roundProb(p[j]))
	}
	w.b.WriteByte('}')
}

// str writes a JSON string: in python mode as json.dumps with ensure_ascii
// (every non-ASCII code point as \uXXXX, astral ones as a surrogate pair),
// otherwise UTF-8 with only what JSON requires escaped.
func (w *s1Writer) str(s string) {
	if !w.python {
		pyString(&w.b, s)
		return
	}
	w.b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			w.b.WriteString(`\"`)
		case '\\':
			w.b.WriteString(`\\`)
		case '\n':
			w.b.WriteString(`\n`)
		case '\r':
			w.b.WriteString(`\r`)
		case '\t':
			w.b.WriteString(`\t`)
		case '\b':
			w.b.WriteString(`\b`)
		case '\f':
			w.b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20:
				fmt.Fprintf(&w.b, `\u%04x`, r)
			case r < 0x80:
				w.b.WriteRune(r)
			case r > 0xffff:
				r -= 0x10000
				fmt.Fprintf(&w.b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
			default:
				fmt.Fprintf(&w.b, `\u%04x`, r)
			}
		}
	}
	w.b.WriteByte('"')
}
