package decide

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
)

type goldenFile struct {
	Version     string
	Temperature float64
	Cases       []struct {
		Name      string
		Body      string
		StateText string `json:"state_text"`
		Questions []struct {
			Name   string
			System string
			Labels []string
			Branch string
			Logits []float64
			Answer json.RawMessage
		}
	}
}

// TestGolden is the R1 gate: surogate's decisions v1 golden file, from an
// independent Python implementation of the protocol. Every text byte for
// byte, every number bit for bit (the answers are compared as Python's
// json.dumps text of what each side parsed, and repr round-trips).
func TestGolden(t *testing.T) {
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Version != "v1" || g.Temperature != 1 || len(g.Cases) != 14 {
		t.Fatalf("golden file: version %q, T %v, %d cases", g.Version, g.Temperature, len(g.Cases))
	}
	codebook := Codebook(nil)
	for _, c := range g.Cases {
		t.Run(c.Name, func(t *testing.T) {
			r, err := Parse([]byte(c.Body))
			if err != nil {
				t.Fatal(err)
			}
			if r.StateText != c.StateText {
				t.Errorf("state text\n got %q\nwant %q", r.StateText, c.StateText)
			}
			if len(r.Questions) != len(c.Questions) {
				t.Fatalf("%d questions, want %d", len(r.Questions), len(c.Questions))
			}
			for i, want := range c.Questions {
				q := &r.Questions[i]
				if q.Name != want.Name {
					t.Errorf("question %d is %q, want %q", i, q.Name, want.Name)
				}
				rd, err := Render(q, codebook)
				if err != nil {
					t.Fatal(err)
				}
				if rd.System != want.System {
					t.Errorf("%s: system %q", q.Name, rd.System)
				}
				if strings.Join(rd.Labels, ",") != strings.Join(want.Labels, ",") {
					t.Errorf("%s: labels %v, want %v", q.Name, rd.Labels, want.Labels)
				}
				if rd.Branch != want.Branch {
					t.Errorf("%s: branch\n got %q\nwant %q", q.Name, rd.Branch, want.Branch)
				}
				logits := make([]float32, len(want.Logits))
				for j, z := range want.Logits {
					logits[j] = float32(z)
					if float64(logits[j]) != z {
						t.Fatalf("golden logit %v is not float32-exact", z)
					}
				}
				a, err := Resolve(q, logits, 1)
				if err != nil {
					t.Fatal(err)
				}
				var b strings.Builder
				a.AppendJSON(&b)
				got, err := ParseValue([]byte(b.String()))
				if err != nil {
					t.Fatalf("%s: our answer is not JSON: %v\n%s", q.Name, err, b.String())
				}
				exp, err := ParseValue(want.Answer)
				if err != nil {
					t.Fatal(err)
				}
				if Dumps(got) != Dumps(exp) {
					t.Errorf("%s: answer\n got %s\nwant %s", q.Name, Dumps(got), Dumps(exp))
				}
			}
		})
	}
}

// TestTemperature checks the tempered readout against the formula
// p_i = exp((z_i - max z)/T) / sum, and that the choice does not move.
func TestTemperature(t *testing.T) {
	r, err := Parse([]byte(`{"model": "rune", "state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y", "c": "z"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	q := &r.Questions[0]
	logits := []float32{1.5, 4.25, -2}
	a, err := Resolve(q, logits, 2)
	if err != nil {
		t.Fatal(err)
	}
	w := []float64{exp((1.5 - 4.25) / 2), 1, exp((-2 - 4.25) / 2)}
	s := w[0] + w[1] + w[2]
	for i := range w {
		if a.Probabilities[i] != w[i]/s {
			t.Errorf("p[%d] = %v, want %v", i, a.Probabilities[i], w[i]/s)
		}
	}
	if a.Choice != 1 {
		t.Errorf("choice %d, want 1", a.Choice)
	}
	if _, err := Resolve(q, []float32{1, float32(math.NaN()), 0}, 1); !errors.Is(err, ErrNonFinite) {
		t.Errorf("NaN logit: %v", err)
	}
}

// TestRefusals covers the 400s v1 documents.
func TestRefusals(t *testing.T) {
	for _, c := range []struct{ name, body, param string }{
		{"not json", `{`, ""},
		{"no model", `{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}`, "model"},
		{"no state", `{"model": "m", "questions": {"q": {"type": "noul", "instructions": "i"}}}`, "state"},
		{"number state", `{"model": "m", "state": 3, "questions": {"q": {"type": "noul", "instructions": "i"}}}`, "state"},
		{"no questions", `{"model": "m", "state": "s", "questions": {}}`, "questions"},
		{"named twice", `{"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}, "q": {"type": "noul", "instructions": "j"}}}`, "questions"},
		{"key twice", `{"model": "m", "state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "x", "a": "y"}}}}`, "questions.q.criteria"},
		{"unknown type", `{"model": "m", "state": "s", "questions": {"q": {"type": "pick", "instructions": "i"}}}`, "questions.q.type"},
		{"no instructions", `{"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}}`, "questions.q.instructions"},
		{"null instructions", `{"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": null}}}`, "questions.q.instructions"},
		{"half a noul", `{"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"true": "y"}}}}`, "questions.q.criteria"},
		{"empty choice", `{"model": "m", "state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {}}}}`, "questions.q.criteria"},
		{"string thinking", `{"model": "m", "state": "s", "thinking": "true", "questions": {"q": {"type": "noul", "instructions": "i"}}}`, "thinking"},
		{"both extensions", `{"model": "m", "state": "s", "thinking": true, "order_averaging": true, "questions": {"q": {"type": "noul", "instructions": "i"}}}`, "order_averaging"},
		{"NaN state", `{"model": "m", "state": [NaN], "questions": {"q": {"type": "noul", "instructions": "i"}}}`, ""},
	} {
		_, err := Parse([]byte(c.body))
		var e *Error
		if !errors.As(err, &e) {
			t.Errorf("%s: %v, want a refusal", c.name, err)
			continue
		}
		if e.Param != c.param {
			t.Errorf("%s: param %q, want %q (%s)", c.name, e.Param, c.param, e.Message)
		}
	}
	// What is accepted: an omitted noul's criteria, a null choice
	// description, a single-option choice, unknown fields.
	r, err := Parse([]byte(`{"model": "m", "state": {}, "user": "u", "questions": {"n": {"type": "noul", "instructions": "i"}, "c": {"type": "choice", "instructions": "i", "criteria": {"only": null}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(r.Questions[0].Texts, ","); got != "false,true" {
		t.Errorf("default noul texts %q", got)
	}
	if got := r.Questions[1].Texts[0]; got != "only" {
		t.Errorf("null description rendered %q", got)
	}
}
