package decide

// POST /v1/decisions in OpenAI's shape (developers.openai.com/api/docs/guides/
// decisions, and the API reference's "Create a decision"), translated onto
// decisions v1 so Rune answers it with the prompt it was trained on.
// OpenRouter's paths keep v1 itself.
//
// The two shapes ask the same three kinds of question. How the translation
// settles what differs:
//
//   - **Input.** `input` is a string or user messages of `input_text` and
//     `input_image` parts. The text is v1's state, a JSON string: the
//     string as sent, or the messages' text parts joined by a blank line.
//     The images are v1's `images`, in order, ahead of the text in every
//     question's user turn (where an image sat between two text parts is
//     not kept). Images must be data URLs, at most 128 of them, as OpenAI
//     requires; `detail` is read and ignored.
//   - **Questions.** An array, answered in its order; `name` is optional
//     and echoed (null when absent). `predicate` is v1's noul with its
//     default options, `choice` is v1's choice, `score` v1's score. An
//     option shows the model its `description`, or its value or label when
//     there is none, as v1 shows a criterion's description. A choice value
//     is a string or a boolean, and the two are distinct, so `"true"` and
//     `true` are two options; one sent twice is refused.
//   - **Answers.** `{"model", "answers", "usage"}`. A predicate answers
//     p(true), a choice its most probable value with the value-typed
//     distribution, a score the expected level with the per-level
//     distribution (`value` the level index, `label` its label). The
//     probabilities and confidences are v1's, at the same calibration
//     temperature. Nothing is generated, so output_tokens is 0; Rune never
//     refuses, so there is no `refusal` answer.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// MaxOpenAIImages is OpenAI's cap on images across one request.
const MaxOpenAIImages = 128

// OpenAIRequest is a parsed OpenAI-shape request: the v1 request it
// translates to, and what its answers echo that v1 has no place for.
type OpenAIRequest struct {
	Request
	Names  []*string  // each question's name, nil when it has none
	Values [][]Value  // a choice's option values as sent (string or boolean)
	Labels [][]string // a score's level labels
}

func openAIInvalid(param, format string, a ...any) error {
	return &Error{Message: fmt.Sprintf(format, a...), Param: param} // OpenAI's code is null here
}

// ParseOpenAI reads an OpenAI-shape request body.
func ParseOpenAI(body []byte) (*OpenAIRequest, error) {
	root, err := ParseValue(body)
	if err != nil {
		return nil, openAIInvalid("", "We could not parse the JSON body of your request.")
	}
	if root.Kind != Object {
		return nil, openAIInvalid("", "The request body must be a JSON object.")
	}
	r := &OpenAIRequest{}
	m, ok := root.Get("model")
	if !ok {
		return nil, openAIInvalid("model", "Missing required parameter: 'model'.")
	}
	if m.Kind != String {
		return nil, openAIInvalid("model", "Invalid type for 'model': expected a string.")
	}
	r.Model = m.S

	in, ok := root.Get("input")
	if !ok {
		return nil, openAIInvalid("input", "Missing required parameter: 'input'.")
	}
	text, err := r.parseInput(in)
	if err != nil {
		return nil, err
	}
	r.State = Value{Kind: String, S: text}
	r.StateText = StatePrefix + Dumps(r.State) + "\n\n"

	qs, ok := root.Get("questions")
	if !ok {
		return nil, openAIInvalid("questions", "Missing required parameter: 'questions'.")
	}
	if qs.Kind != Array {
		return nil, openAIInvalid("questions", "Invalid type for 'questions': expected an array of questions.")
	}
	if len(qs.Arr) == 0 {
		return nil, openAIInvalid("questions", "'questions' must contain at least one question.")
	}
	for i, spec := range qs.Arr {
		if err := r.parseQuestion(i, spec); err != nil {
			return nil, err
		}
	}

	if sid, ok := root.Get("safety_identifier"); ok && sid.Kind != Null {
		if sid.Kind != String {
			return nil, openAIInvalid("safety_identifier", "Invalid type for 'safety_identifier': expected a string.")
		}
		if len(sid.S) > 128 {
			return nil, openAIInvalid("safety_identifier", "'safety_identifier' is longer than 128 characters.")
		}
	}
	return r, nil
}

// parseInput collects the input's text, returned, and its images, into
// r.Images.
func (r *OpenAIRequest) parseInput(in Value) (string, error) {
	switch in.Kind {
	case String:
		return in.S, nil
	case Array:
	default:
		return "", openAIInvalid("input", "Invalid type for 'input': expected a string or an array of user messages.")
	}
	var texts []string
	for i, msg := range in.Arr {
		param := fmt.Sprintf("input[%d]", i)
		if msg.Kind != Object {
			return "", openAIInvalid(param, "Invalid type for '%s': expected a message object.", param)
		}
		if t, ok := msg.Get("type"); ok && (t.Kind != String || t.S != "message") {
			return "", openAIInvalid(param+".type", "Invalid value for '%s.type': only 'message' is supported.", param)
		}
		if role, ok := msg.Get("role"); !ok || role.Kind != String || role.S != "user" {
			return "", openAIInvalid(param+".role", "Invalid value for '%s.role': only 'user' messages are supported.", param)
		}
		content, ok := msg.Get("content")
		if !ok {
			return "", openAIInvalid(param+".content", "Missing required parameter: '%s.content'.", param)
		}
		if content.Kind == String {
			texts = append(texts, content.S)
			continue
		}
		if content.Kind != Array {
			return "", openAIInvalid(param+".content", "Invalid type for '%s.content': expected a string or an array of parts.", param)
		}
		for j, part := range content.Arr {
			pp := fmt.Sprintf("%s.content[%d]", param, j)
			t, _ := part.Get("type")
			if part.Kind != Object || t.Kind != String {
				return "", openAIInvalid(pp, "Invalid type for '%s': expected an input_text or input_image part.", pp)
			}
			switch t.S {
			case "input_text":
				s, ok := part.Get("text")
				if !ok || s.Kind != String {
					return "", openAIInvalid(pp+".text", "Missing required parameter: '%s.text'.", pp)
				}
				texts = append(texts, s.S)
			case "input_image":
				u, ok := part.Get("image_url")
				if !ok || u.Kind != String {
					return "", openAIInvalid(pp+".image_url", "Missing required parameter: '%s.image_url'.", pp)
				}
				if !strings.HasPrefix(u.S, "data:") {
					return "", openAIInvalid(pp+".image_url",
						"Invalid value for '%s.image_url': images must be inline base64 data URLs; hosted URLs and file IDs are not supported.", pp)
				}
				if d, ok := part.Get("detail"); ok && d.Kind != Null {
					switch d.S {
					case "low", "high", "auto", "original":
					default:
						return "", openAIInvalid(pp+".detail", "Invalid value for '%s.detail': expected 'low', 'high', 'auto' or 'original'.", pp)
					}
				}
				r.Images = append(r.Images, u.S)
				if len(r.Images) > MaxOpenAIImages {
					return "", openAIInvalid("input", "Too many images: at most %d are allowed across the request.", MaxOpenAIImages)
				}
			default:
				return "", openAIInvalid(pp+".type", "Invalid value for '%s.type': only 'input_text' and 'input_image' are supported.", pp)
			}
		}
	}
	return strings.Join(texts, "\n\n"), nil
}

// optionText is what an option shows the model: its description, or
// fallback when it has none.
func optionText(spec Value, param, fallback string) (string, error) {
	d, ok := spec.Get("description")
	if !ok || d.Kind == Null {
		return fallback, nil
	}
	if d.Kind != String {
		return "", openAIInvalid(param+".description", "Invalid type for '%s.description': expected a string.", param)
	}
	if d.S == "" {
		return fallback, nil
	}
	return d.S, nil
}

func (r *OpenAIRequest) parseQuestion(i int, spec Value) error {
	param := fmt.Sprintf("questions[%d]", i)
	if spec.Kind != Object {
		return openAIInvalid(param, "Invalid type for '%s': expected a question object.", param)
	}
	var name *string
	if n, ok := spec.Get("name"); ok && n.Kind != Null {
		if n.Kind != String {
			return openAIInvalid(param+".name", "Invalid type for '%s.name': expected a string.", param)
		}
		name = &n.S
	}
	ins, ok := spec.Get("instructions")
	if !ok {
		return openAIInvalid(param+".instructions", "Missing required parameter: '%s.instructions'.", param)
	}
	if ins.Kind != String {
		return openAIInvalid(param+".instructions", "Invalid type for '%s.instructions': expected a string.", param)
	}
	q := Question{Instructions: ins.S}
	if name != nil {
		q.Name = *name
	}
	var values []Value
	var labels []string
	add := func(key, text string, v Value) {
		q.Keys = append(q.Keys, key)
		q.Texts = append(q.Texts, text)
		q.Values = append(q.Values, v)
	}
	t, _ := spec.Get("type")
	if t.Kind != String {
		return openAIInvalid(param+".type", "Missing required parameter: '%s.type'.", param)
	}
	switch t.S {
	case "predicate":
		q.Type = Noul
		add("false", "false", Value{Kind: String, S: "false"})
		add("true", "true", Value{Kind: String, S: "true"})
	case "choice":
		q.Type = Choice
		cs, ok := spec.Get("choices")
		if !ok || cs.Kind != Array {
			return openAIInvalid(param+".choices", "Missing required parameter: '%s.choices'.", param)
		}
		seen := map[string]bool{}
		for j, c := range cs.Arr {
			cp := fmt.Sprintf("%s.choices[%d]", param, j)
			v, _ := c.Get("value")
			if c.Kind != Object || (v.Kind != String && v.Kind != Bool) {
				return openAIInvalid(cp+".value", "Invalid type for '%s.value': expected a string or a boolean.", cp)
			}
			key := Dumps(v) // typed: "true" and true differ
			if seen[key] {
				return openAIInvalid(cp+".value", "Choice value %s appears twice in '%s.choices'.", key, param)
			}
			seen[key] = true
			fallback := v.S
			if v.Kind == Bool {
				fallback = key
			}
			text, err := optionText(c, cp, fallback)
			if err != nil {
				return err
			}
			add(key, text, Value{Kind: String, S: text})
			values = append(values, v)
		}
	case "score":
		q.Type = Score
		ls, ok := spec.Get("levels")
		if !ok || ls.Kind != Array {
			return openAIInvalid(param+".levels", "Missing required parameter: '%s.levels'.", param)
		}
		for j, l := range ls.Arr {
			lp := fmt.Sprintf("%s.levels[%d]", param, j)
			label, _ := l.Get("label")
			if l.Kind != Object || label.Kind != String {
				return openAIInvalid(lp+".label", "Missing required parameter: '%s.label'.", lp)
			}
			text, err := optionText(l, lp, label.S)
			if err != nil {
				return err
			}
			add(fmt.Sprint(j), text, Value{Kind: String, S: text})
			labels = append(labels, label.S)
		}
	default:
		return openAIInvalid(param+".type", "Invalid value for '%s.type': expected 'predicate', 'choice' or 'score'.", param)
	}
	if n := len(q.Keys); n < MinOptions || n > MaxOptions {
		return openAIInvalid(param, "'%s' has %d options; between %d and %d are supported.", param, n, MinOptions, MaxOptions)
	}
	r.Questions = append(r.Questions, q)
	r.Names = append(r.Names, name)
	r.Values = append(r.Values, values)
	r.Labels = append(r.Labels, labels)
	return nil
}

// openAIValue is a choice value as sent: a JSON string or boolean.
func openAIValue(v Value) any {
	if v.Kind == Bool {
		return v.B
	}
	return v.S
}

type openAIProb struct {
	Value       any     `json:"value"`
	Label       *string `json:"label,omitempty"`
	Probability float64 `json:"probability"`
}

type openAIAnswer struct {
	Type          string       `json:"type"`
	Name          *string      `json:"name"`
	Probability   *float64     `json:"probability,omitempty"`
	Choice        any          `json:"choice,omitempty"`
	Score         *float64     `json:"score,omitempty"`
	Probabilities []openAIProb `json:"probabilities,omitempty"`
	Confidence    *float64     `json:"confidence,omitempty"`
}

// OpenAIResponse is the OpenAI-shape body: the answers in question order
// and usage, with nothing generated (output_tokens 0).
func OpenAIResponse(r *OpenAIRequest, model string, answers []Answer, inputTokens int) []byte {
	out := struct {
		Model   string         `json:"model"`
		Answers []openAIAnswer `json:"answers"`
		Usage   any            `json:"usage"`
	}{Model: model, Answers: make([]openAIAnswer, len(answers))}
	for i := range answers {
		a := &answers[i]
		o := openAIAnswer{Name: r.Names[i]}
		switch a.Q.Type {
		case Noul:
			o.Type, o.Probability = "predicate", &a.Value
		case Choice:
			o.Type, o.Confidence = "choice", &a.Confidence
			o.Choice = openAIValue(r.Values[i][a.Choice])
			for j, p := range a.Probabilities {
				o.Probabilities = append(o.Probabilities, openAIProb{Value: openAIValue(r.Values[i][j]), Probability: p})
			}
		case Score:
			o.Type, o.Score, o.Confidence = "score", &a.Value, &a.Confidence
			for j, p := range a.Probabilities {
				o.Probabilities = append(o.Probabilities, openAIProb{Value: j, Label: &r.Labels[i][j], Probability: p})
			}
		}
		out.Answers[i] = o
	}
	type details struct {
		CacheWrite int `json:"cache_write_tokens"`
		Cached     int `json:"cached_tokens"`
	}
	out.Usage = struct {
		InputTokens         int            `json:"input_tokens"`
		InputTokensDetails  details        `json:"input_tokens_details"`
		OutputTokens        int            `json:"output_tokens"`
		OutputTokensDetails map[string]int `json:"output_tokens_details"`
		TotalTokens         int            `json:"total_tokens"`
	}{InputTokens: inputTokens, OutputTokensDetails: map[string]int{"reasoning_tokens": 0}, TotalTokens: inputTokens}
	b, _ := json.Marshal(out)
	return b
}
