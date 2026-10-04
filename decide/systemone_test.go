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

// TestSystemOneImages holds System One's `images` (R10) to decisions v1's:
// the same forms accepted in order, the same refusals, and none when absent.
func TestSystemOneImages(t *testing.T) {
	q := `"questions": {"x": {"type": "noul", "instructions": "?"}}`
	r, err := ParseSystemOne([]byte(`{"state": "s", "images": ["data:image/png;base64,AA==", {"url": "https://example.com/a.png"}], ` + q + `}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Images) != 2 || r.Images[0] != "data:image/png;base64,AA==" || r.Images[1] != "https://example.com/a.png" {
		t.Errorf("images %q", r.Images)
	}
	for _, body := range []string{`{"state": "s", ` + q + `}`, `{"state": "s", "images": null, ` + q + `}`} {
		if r, err := ParseSystemOne([]byte(body)); err != nil || len(r.Images) != 0 {
			t.Errorf("%s: images %q, err %v", body, r.Images, err)
		}
	}
	for _, bad := range []string{`"images": "data:image/png;base64,AA=="`, `"images": [""]`, `"images": ["ftp://x/a.png"]`, `"images": [3]`} {
		_, err := ParseSystemOne([]byte(`{"state": "s", ` + bad + `, ` + q + `}`))
		var de *Error
		if !errors.As(err, &de) || de.Param != "images" {
			t.Errorf("%s: want a refusal naming images, got %v", bad, err)
		}
	}
	// The decisions parser takes the same forms.
	d, err := Parse([]byte(`{"model": "rune", "state": "s", "images": [{"url": "http://h/b.jpg"}], ` + q + `}`))
	if err != nil || len(d.Images) != 1 || d.Images[0] != "http://h/b.jpg" {
		t.Errorf("decisions images %q, err %v", d.Images, err)
	}
}
