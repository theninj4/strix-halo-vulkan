package kev

// TypeSafe's System One contract, as Kev serves it (kev/api.py at 9fdf054).
//
//	Noul   -> 2 options [no, yes];               answer = p(yes)
//	Choice -> options "name" or "name: desc";    answer = argmax, probabilities by name, confidence
//	Score  -> options = the rendered levels;     answer = expected level, legend, probabilities by index
//
// Everything here is host-side and exact: it decides the tokens the model
// sees and the numbers the caller gets, and a slip in it is a wrong answer
// with no numerical signature. So it is a port line for line, and it is
// checked against Kev's own output on the K1 fixtures rather than against
// a reading of it.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// MaxOptions is the most options a choice or levels a score may have:
// 2^8 - 1, TypeSafe's limit, which Kev enforces too.
const MaxOptions = 255

// DefaultModel is what a request that names no model gets.
const DefaultModel = "kev-latest"

// Question types. "noul" is TypeSafe's name for a yes/no question, and it is
// the name on the wire in both directions; it is not a typo for "bool".
const (
	TypeNoul   = "noul"
	TypeChoice = "choice"
	TypeScore  = "score"
)

// Request is a parsed POST /v1/systemone body.
type Request struct {
	State     Value
	Model     string
	Questions []RequestQuestion // in the order the caller wrote them
}

// RequestQuestion is one entry of `questions`. ID is its key, which the
// model never sees.
type RequestQuestion struct {
	ID           string
	Type         string
	Instructions Value
	Criteria     Value // Null for a noul that gives none
}

// ValidationError is a request Kev would refuse with a 422.
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

func invalid(format string, args ...any) error {
	return &ValidationError{Msg: fmt.Sprintf(format, args...)}
}

// ParseRequest decodes and validates a request body with the rules Kev's
// pydantic models apply: state is required (null is allowed); questions is a
// non-empty object; a choice has 1..255 named criteria; a score has 1..255
// levels; a noul's criteria are absent, null or an object. Unknown fields
// are ignored.
func ParseRequest(body []byte) (*Request, error) {
	var root Value
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, invalid("body is not JSON: %v", err)
	}
	if root.Kind != Object {
		return nil, invalid("body must be a JSON object")
	}
	req := &Request{Model: DefaultModel}
	state, ok := root.Get("state")
	if !ok {
		return nil, invalid("state is required")
	}
	req.State = state
	if m, ok := root.Get("model"); ok {
		if m.Kind != String {
			return nil, invalid("model must be a string")
		}
		req.Model = m.S
	}
	qs, ok := root.Get("questions")
	if !ok || qs.Kind != Object {
		return nil, invalid("questions must be an object of id -> question")
	}
	if qs.Len() == 0 {
		return nil, invalid("questions must not be empty")
	}
	for i, id := range qs.Keys {
		q, err := parseQuestion(id, qs.Vals[i])
		if err != nil {
			return nil, err
		}
		req.Questions = append(req.Questions, q)
	}
	return req, nil
}

func parseQuestion(id string, v Value) (RequestQuestion, error) {
	q := RequestQuestion{ID: id}
	if v.Kind != Object {
		return q, invalid("questions.%s must be an object", id)
	}
	t, ok := v.Get("type")
	if !ok || t.Kind != String || (t.S != TypeNoul && t.S != TypeChoice && t.S != TypeScore) {
		return q, invalid("questions.%s.type must be one of %q, %q, %q", id, TypeNoul, TypeChoice, TypeScore)
	}
	q.Type = t.S
	q.Instructions, _ = v.Get("instructions")
	c, has := v.Get("criteria")
	switch q.Type {
	case TypeNoul:
		if has && c.Kind != Null && c.Kind != Object {
			return q, invalid("questions.%s.criteria must be an object with optional \"true\" and \"false\"", id)
		}
	case TypeChoice:
		if !has || c.Kind != Object {
			return q, invalid("questions.%s.criteria must be an object of option -> description", id)
		}
		if n := c.Len(); n < 1 || n > MaxOptions {
			return q, invalid("questions.%s.criteria must have 1..%d options, got %d", id, MaxOptions, n)
		}
	case TypeScore:
		if !has || c.Kind != Array {
			return q, invalid("questions.%s.criteria must be an array of levels, lowest first", id)
		}
		if n := c.Len(); n < 1 || n > MaxOptions {
			return q, invalid("questions.%s.criteria must have 1..%d levels, got %d", id, MaxOptions, n)
		}
	}
	q.Criteria = c
	return q, nil
}

// Record is Kev's internal record: the rendered state and, per question, the
// rendered instructions and option texts. It is exactly what the encoder
// tokenises.
type Record struct {
	State     string
	Questions []Question
}

// Question is one rendered question.
type Question struct {
	Instr   string
	Options []string
}

// Meta is what maps a question's probabilities back to its answer.
type Meta struct {
	ID     string
	Type   string
	Keys   []string // the keys probabilities are reported under, in option order
	Legend []string // score only: the rendered levels
}

// optionText is `name` for a null or empty-string description and
// `name: <render(desc)>` otherwise. Only the empty *string* counts as empty:
// an empty list renders "" but still gets the colon, as in Kev.
func optionText(name string, desc Value) string {
	if desc.Kind == Null || (desc.Kind == String && desc.S == "") {
		return name
	}
	return name + ": " + Render(desc)
}

// ToRecord is `api.to_record`.
func ToRecord(req *Request) (Record, []Meta) {
	rec := Record{State: Render(req.State)}
	meta := make([]Meta, 0, len(req.Questions))
	for _, q := range req.Questions {
		m := Meta{ID: q.ID, Type: q.Type}
		var opts []string
		switch q.Type {
		case TypeNoul:
			var no, yes Value
			if q.Criteria.Kind == Object {
				no, _ = q.Criteria.Get("false")
				yes, _ = q.Criteria.Get("true")
			}
			// The order is [no, yes], and the answer is the second slot.
			opts = []string{optionText("no", no), optionText("yes", yes)}
			m.Keys = []string{"false", "true"}
		case TypeChoice:
			for i, k := range q.Criteria.Keys {
				opts = append(opts, optionText(k, q.Criteria.Vals[i]))
			}
			m.Keys = append([]string(nil), q.Criteria.Keys...)
		case TypeScore:
			for i, x := range q.Criteria.Arr {
				opts = append(opts, Render(x))
				m.Keys = append(m.Keys, strconv.Itoa(i))
			}
			m.Legend = opts
		}
		rec.Questions = append(rec.Questions, Question{Instr: Render(q.Instructions), Options: opts})
		meta = append(meta, m)
	}
	return rec, meta
}

// Answer is one question's answer. Which fields are set depends on Type.
type Answer struct {
	ID            string
	Type          string
	Noul          float64   // noul: p(yes)
	Choice        string    // choice: the most likely option
	Score         float64   // score: the expected level index
	Confidence    float64   // choice, score
	Keys          []string  // choice: option names; score: level indices
	Probabilities []float64 // choice, score, parallel to Keys
	Legend        []string  // score
}

// ChoiceConfidence is `(p_max - 1/K) / (1 - 1/K)`, and 1 for a single option.
// It measures how far the leading answer stands above uniform; it is not an
// estimate of being right.
func ChoiceConfidence(p []float64) float64 {
	k := float64(len(p))
	if len(p) == 1 {
		return 1
	}
	return (p[argmax(p)] - 1/k) / (1 - 1/k)
}

// ScoreConfidence is Kev's approximation of TypeSafe's unpublished formula:
// 1 - E|level - mode| / (L - 1).
func ScoreConfidence(p []float64) float64 {
	if len(p) == 1 {
		return 1
	}
	mode := argmax(p)
	var s float64
	for i, pi := range p {
		s += pi * math.Abs(float64(i-mode))
	}
	return 1 - s/float64(len(p)-1)
}

// argmax is the first index of the largest value, which is what Python's
// max(range(n), key=...) returns on a tie.
func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

// RoundProb is `round(float(x), 4)`: correctly rounded from the exact binary
// value, which is what both Python's round and strconv's 'f' formatting do.
// Four decimals keep a 255-option distribution's rounded sum within
// TypeSafe's |sum - 1| < 0.02.
func RoundProb(x float64) float64 {
	r, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	return r
}

// ToAnswers is `api.to_answers`: the unrounded probabilities decide the
// choice, the score and both confidences, and only what is reported is
// rounded.
func ToAnswers(probs [][]float64, meta []Meta) []Answer {
	out := make([]Answer, len(meta))
	for i, m := range meta {
		p := probs[i]
		a := Answer{ID: m.ID, Type: m.Type}
		switch m.Type {
		case TypeNoul:
			a.Noul = RoundProb(p[1])
		case TypeChoice:
			a.Choice = m.Keys[argmax(p)]
			a.Confidence = RoundProb(ChoiceConfidence(p))
			a.Keys = m.Keys
			for _, v := range p {
				a.Probabilities = append(a.Probabilities, RoundProb(v))
			}
		case TypeScore:
			var s float64
			for j, v := range p {
				s += float64(j) * v
			}
			a.Score = RoundProb(s)
			a.Legend = m.Legend
			a.Keys = m.Keys
			for _, v := range p {
				a.Probabilities = append(a.Probabilities, RoundProb(v))
			}
			a.Confidence = RoundProb(ScoreConfidence(p))
		}
		out[i] = a
	}
	return out
}

// AnswersJSON writes the answers object with its keys in Kev's order. With
// python set it is byte-identical to Python's `json.dumps(answers)` (", " and
// ": " separators, ASCII-only with \u escapes, floats by repr), which is the
// string `usage.output_tokens` counts; without it, it is compact UTF-8 JSON
// for the response body.
func AnswersJSON(answers []Answer, python bool) []byte {
	w := &jsonWriter{python: python}
	w.open('{')
	for i, a := range answers {
		w.sep(i)
		w.key(a.ID)
		w.open('{')
		w.key("type")
		w.str(a.Type)
		switch a.Type {
		case TypeNoul:
			w.sep(1)
			w.key("noul")
			w.float(a.Noul)
		case TypeChoice:
			w.sep(1)
			w.key("choice")
			w.str(a.Choice)
			w.sep(1)
			w.key("confidence")
			w.float(a.Confidence)
			w.sep(1)
			w.key("probabilities")
			w.dist(a.Keys, a.Probabilities)
		case TypeScore:
			w.sep(1)
			w.key("score")
			w.float(a.Score)
			w.sep(1)
			w.key("legend")
			w.open('{')
			for j, l := range a.Legend {
				w.sep(j)
				w.key(a.Keys[j])
				w.str(l)
			}
			w.close('}')
			w.sep(1)
			w.key("probabilities")
			w.dist(a.Keys, a.Probabilities)
			w.sep(1)
			w.key("confidence")
			w.float(a.Confidence)
		}
		w.close('}')
	}
	w.close('}')
	return w.b.Bytes()
}

type jsonWriter struct {
	b      bytes.Buffer
	python bool
}

func (w *jsonWriter) open(c byte)  { w.b.WriteByte(c) }
func (w *jsonWriter) close(c byte) { w.b.WriteByte(c) }

func (w *jsonWriter) sep(i int) {
	if i == 0 {
		return
	}
	if w.python {
		w.b.WriteString(", ")
	} else {
		w.b.WriteByte(',')
	}
}

func (w *jsonWriter) key(k string) {
	w.str(k)
	if w.python {
		w.b.WriteString(": ")
	} else {
		w.b.WriteByte(':')
	}
}

func (w *jsonWriter) float(f float64) { w.b.WriteString(PyFloatRepr(f)) }

func (w *jsonWriter) dist(keys []string, p []float64) {
	w.open('{')
	for j, k := range keys {
		w.sep(j)
		w.key(k)
		w.float(p[j])
	}
	w.close('}')
}

// str writes a JSON string. In python mode it escapes as json.dumps does
// with ensure_ascii: \" \\ \n \r \t \b \f, other controls and every
// non-ASCII code point as \uXXXX (astral ones as a surrogate pair). Outside
// python mode it escapes only what JSON requires and writes UTF-8.
func (w *jsonWriter) str(s string) {
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
			case r < 0x80 || !w.python:
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

// jsonString is a compact JSON string literal, for the response envelope.
func jsonString(s string) string {
	w := &jsonWriter{}
	w.str(s)
	return w.b.String()
}

// ResponseJSON is the /v1/systemone body: model, answers, usage, latency_ms.
func ResponseJSON(model string, answers []Answer, inputTokens, outputTokens int, latencyMS float64) []byte {
	var b strings.Builder
	b.WriteString(`{"model":`)
	b.WriteString(jsonString(model))
	b.WriteString(`,"answers":`)
	b.Write(AnswersJSON(answers, false))
	fmt.Fprintf(&b, `,"usage":{"input_tokens":%d,"output_tokens":%d},"latency_ms":%s}`,
		inputTokens, outputTokens, PyFloatRepr(math.Round(latencyMS*10)/10))
	return []byte(b.String())
}
