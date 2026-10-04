package gemma4

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/decide"
)

// tokenizerDir is the tokenizer the tests read: GEMMA4_TOKENIZER, or
// google/gemma-4-26B-A4B-it's files (Rune's are the same template; R0
// re-runs reference/dump_gemma4_tokens.py against Rune's own).
func tokenizerDir() string {
	if d := os.Getenv("GEMMA4_TOKENIZER"); d != "" {
		return d
	}
	return "../models/gemma-4-tokenizer"
}

func loadTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer(tokenizerDir())
	if os.IsNotExist(err) {
		t.Skipf("no tokenizer at %s (see reference/dump_gemma4_tokens.py)", tokenizerDir())
	}
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestPromptTokens is the R2 gate: for every decisions v1 golden question
// and a corpus of awkward states, the transcribed chat template renders
// HF's text and Encode gives HF's ids, every label is one token in context,
// and the codebook is the reference's.
func TestPromptTokens(t *testing.T) {
	tok := loadTokenizer(t)
	raw, err := os.ReadFile("testdata/prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Codebook []string
		Prompts  []struct {
			Name, System, User, Text string
			IDs                      []int32
			Labels                   []string
		}
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	for _, p := range f.Prompts {
		if p.System != "" {
			if got := DecisionPrompt(p.System, p.User); got != p.Text {
				t.Errorf("%s: rendered prompt differs from apply_chat_template", p.Name)
			}
		}
		start := time.Now()
		got := tok.Encode(p.Text)
		took := time.Since(start)
		if !slices.Equal(got, p.IDs) {
			i := 0
			for i < min(len(got), len(p.IDs)) && got[i] == p.IDs[i] {
				i++
			}
			t.Errorf("%s: %d ids, want %d; first difference at %d", p.Name, len(got), len(p.IDs), i)
			continue
		}
		if len(p.IDs) > 4000 {
			t.Logf("%s: %d chars, %d ids in %v", p.Name, len(p.Text), len(p.IDs), took)
		}
		if dec := tok.Decode(got, false); strings.HasPrefix(p.Name, "raw/") && dec != p.Text {
			t.Errorf("%s: decode does not round-trip", p.Name)
		}
		for _, l := range p.Labels {
			more := tok.Encode(p.Text + l)
			if len(more) != len(got)+1 || !slices.Equal(more[:len(got)], got) {
				t.Errorf("%s: label %q is not one token in context", p.Name, l)
			}
		}
	}
	cb := decide.Codebook(func(code string) (int, bool) {
		ids := tok.Encode(code)
		return int(firstOr(ids)), len(ids) == 1 && tok.Decode(ids, false) == code
	})
	if !slices.Equal(cb, f.Codebook) {
		t.Errorf("codebook: %d codes, want %d", len(cb), len(f.Codebook))
	}
}

func firstOr(ids []int32) int32 {
	if len(ids) == 0 {
		return -1
	}
	return ids[0]
}

// TestPlan holds the shared-prefix rule on real prompts: the prefix is
// common to every prompt, ends before "QUESTION:", leaves every branch at
// least one token, and prefix + branch is each prompt's own tokens.
func TestPlan(t *testing.T) {
	tok := loadTokenizer(t)
	cb := tok.Codebook()
	for _, body := range []string{
		`{"model": "rune", "state": "Customer wrote: the package arrived late and damaged, I want my money back.", "questions": {"sentiment": {"type": "choice", "instructions": "What is the customer's sentiment?", "criteria": {"positive": "+", "neutral": "0", "negative": "-"}}, "refund": {"type": "noul", "instructions": "Refund?"}}}`,
		`{"model": "rune", "state": {"a": [1, 2]}, "order_averaging": true, "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"x": "ex", "y": "why"}}, "s": {"type": "score", "instructions": "Rate", "criteria": ["lo", "hi"]}}}`,
		`{"model": "rune", "state": "short", "questions": {"only": {"type": "noul", "instructions": "Is it short?"}}}`,
	} {
		req, err := decide.Parse([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		pl, err := tok.plan(req, cb)
		if err != nil {
			t.Fatal(err)
		}
		wantReads := len(req.Questions)
		if req.OrderAveraging {
			wantReads++ // the choice question's mirror; the score has none
		}
		if len(pl.reads) != wantReads {
			t.Errorf("%d readings, want %d", len(pl.reads), wantReads)
		}
		prefix := tok.Decode(pl.prompts[0].ids[:pl.pref], false)
		if strings.Contains(prefix, "QUESTION:") || !strings.Contains(prefix, "SHARED STATE") {
			t.Errorf("prefix %q", prefix)
		}
		for k, p := range pl.prompts {
			if len(pl.branches[k]) == 0 || !slices.Equal(append(append([]int32(nil), p.ids[:pl.pref]...), pl.branches[k]...), p.ids) {
				t.Errorf("reading %d: prefix + branch is not its prompt", k)
			}
			if len(p.labels) == 0 {
				t.Errorf("reading %d: no labels", k)
			}
		}
		t.Logf("%d readings, prefix %d tokens, branches %v tokens", len(pl.reads), pl.pref, func() []int {
			var n []int
			for _, b := range pl.branches {
				n = append(n, len(b))
			}
			return n
		}())
	}
}
