package decide

import (
	"errors"
	"strings"
	"testing"
)

// TestSystemOne holds the System One translation (R-o1): the loose request
// becomes the v1 question the model reads, and the answers come back in
// Kev's envelope, keys in Kev's order, rounded to four places.
func TestSystemOne(t *testing.T) {
	body := `{"state": {"ticket": "Order 8812 arrived crushed. Refund please — café"},
	  "questions": {
	    "refund": {"type": "noul", "instructions": "Is a refund requested?", "criteria": {"true": "A refund is requested"}},
	    "tone":   {"type": "choice", "criteria": {"calm": "", "annoyed": null, "furious": "Very angry"}},
	    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", {"level": 2}]}}}`
	r, err := ParseSystemOne([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if r.Model != "kev-latest" {
		t.Errorf("default model %q", r.Model)
	}
	want := []string{
		"QUESTION:\nIs a refund requested?\nOPTIONS:\nA: false\nB: A refund is requested\nAnswer with one option letter only.",
		"QUESTION:\n\nOPTIONS:\nA: calm\nB: annoyed\nC: Very angry\nAnswer with one option letter only.",
		"QUESTION:\nHow urgent?\nOPTIONS:\nA: low\nB: {\"level\": 2}\nAnswer with one option letter only.",
	}
	logits := [][]float32{{-1.5, 2.25}, {0.5, 3, 1}, {2, 0.125}}
	var answers []Answer
	for i := range r.Questions {
		rd, err := Render(&r.Questions[i], nil)
		if err != nil {
			t.Fatal(err)
		}
		if rd.Branch != want[i] {
			t.Errorf("%s: branch\n got %q\nwant %q", r.Questions[i].Name, rd.Branch, want[i])
		}
		a, err := Resolve(&r.Questions[i], logits[i], 1)
		if err != nil {
			t.Fatal(err)
		}
		answers = append(answers, a)
	}
	if !strings.HasPrefix(r.StateText, `SHARED STATE (JSON string):`+"\n"+`{"ticket": "Order 8812 arrived crushed. Refund please — café"}`) {
		t.Errorf("state text %q", r.StateText)
	}
	got := string(SystemOneResponse(r.Model, answers, 120, 30, 53.26))
	exp := `{"model":"kev-latest","answers":{` +
		`"refund":{"type":"noul","noul":0.977},` +
		`"tone":{"type":"choice","choice":"annoyed","confidence":0.7321,"probabilities":{"calm":0.0674,"annoyed":0.8214,"furious":0.1112}},` +
		`"urgency":{"type":"score","score":0.133,"legend":{"0":"low","1":"{\"level\": 2}"},"probabilities":{"0":0.867,"1":0.133},"confidence":0.7341}},` +
		`"usage":{"input_tokens":120,"output_tokens":30},"latency_ms":53.3}`
	if got != exp {
		t.Errorf("response\n got %s\nwant %s", got, exp)
	}
	py := SystemOneAnswers(answers[:1], true)
	if py != `{"refund": {"type": "noul", "noul": 0.977}}` {
		t.Errorf("python answers %s", py)
	}

	for _, bad := range []string{
		`{"questions": {"q": {"type": "noul"}}}`,
		`{"state": "s", "questions": {}}`,
		`{"state": "s", "questions": {"q": {"type": "bool"}}}`,
		`{"state": "s", "questions": {"q": {"type": "choice", "criteria": ["a"]}}}`,
		`{"state": "s", "model": 3, "questions": {"q": {"type": "noul"}}}`,
	} {
		var e *SystemOneError
		if _, err := ParseSystemOne([]byte(bad)); !errors.As(err, &e) {
			t.Errorf("%s: %v, want a 422", bad, err)
		}
	}
}
