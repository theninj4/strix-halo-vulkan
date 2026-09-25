package kev

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const (
	refDir    = "../reference/out/kev"
	fixtures  = "../reference/kev_fixtures.json"
	kevDir    = "../models/kev-4b"
	baseDir   = "../models/Qwen3.5-4B-Base"
	dumpUsage = "run reference/dump_kev.py"
)

// kevRecord is reference/out/kev/<name>/record.json, the parts the host side
// is checked against.
type kevRecord struct {
	Record struct {
		State     string `json:"state"`
		Questions []struct {
			Instr   string   `json:"instr"`
			Options []string `json:"options"`
		} `json:"questions"`
	} `json:"record"`
	Meta []struct {
		ID     string            `json:"id"`
		Type   string            `json:"type"`
		Keys   []string          `json:"keys"`
		Legend map[string]string `json:"legend"`
	} `json:"meta"`
	Enc struct {
		IDs       []int32 `json:"ids"`
		Seg       []int32 `json:"seg"`
		Pos       []int32 `json:"pos"`
		Opt       []int32 `json:"opt"`
		DecideIdx []int   `json:"decide_idx"`
		OptIdx    [][]int `json:"opt_idx"`
	} `json:"enc"`
	StateTruncated bool `json:"state_truncated"`
	Rows           []struct {
		IDs    []int32 `json:"ids"`
		Pos    []int32 `json:"pos"`
		Decide int     `json:"decide"`
		Opts   []int   `json:"opts"`
	} `json:"rows"`
	StateLen     int         `json:"state_len"`
	Probs        [][]float64 `json:"probs"`
	RowLogitsRaw [][]float64 `json:"row_logits_raw"`
	RowProbs     [][]float64 `json:"row_probs"`
	Temperature  float64     `json:"temperature"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	AnswersJSON string `json:"answers_json"`
}

type fixture struct {
	Name    string          `json:"name"`
	Request json.RawMessage `json:"request"`
	Ref     *kevRecord
}

func loadFixtures(t *testing.T) []fixture {
	t.Helper()
	buf, err := os.ReadFile(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	var fx []fixture
	if err := json.Unmarshal(buf, &fx); err != nil {
		t.Fatal(err)
	}
	for i := range fx {
		b, err := os.ReadFile(filepath.Join(refDir, fx[i].Name, "record.json"))
		if err != nil {
			t.Skipf("no reference dump for %s (%v); %s", fx[i].Name, err, dumpUsage)
		}
		fx[i].Ref = new(kevRecord)
		if err := json.Unmarshal(b, fx[i].Ref); err != nil {
			t.Fatal(err)
		}
	}
	return fx
}

func loadEncoder(t *testing.T) *Encoder {
	t.Helper()
	if _, err := os.Stat(filepath.Join(kevDir, "tokenizer.json")); err != nil {
		t.Skipf("no checkpoint at %s", kevDir)
	}
	e, err := LoadEncoder(kevDir)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// TestDelimiterIDs pins the delimiter roles to the ids Kev's dump read out of
// the same tokenizer. The roles are the trap: <decide> is fim_suffix.
func TestDelimiterIDs(t *testing.T) {
	e := loadEncoder(t)
	buf, err := os.ReadFile(filepath.Join(refDir, "tokens.json"))
	if err != nil {
		t.Skipf("no tokens.json (%v); run reference/dump_kev_tokens.py", err)
	}
	var ref struct {
		Special map[string]int32 `json:"special"`
	}
	if err := json.Unmarshal(buf, &ref); err != nil {
		t.Fatal(err)
	}
	got := []int32{e.State, e.Q, e.Open, e.Close, e.Decide}
	for i, name := range delimiterTokens {
		if got[i] != ref.Special[name] {
			t.Errorf("%s: %d, Kev's tokenizer says %d", name, got[i], ref.Special[name])
		}
	}
}

// TestUserTokensMatchKev runs Kev's user_tokens over the adversarial strings
// of reference/dump_kev_tokens.py. The one string that is not NFC is the
// documented gap and is required to *differ*, so the day it stops differing
// this test says the gap is closed.
func TestUserTokensMatchKev(t *testing.T) {
	e := loadEncoder(t)
	buf, err := os.ReadFile(filepath.Join(refDir, "tokens.json"))
	if err != nil {
		t.Skipf("no tokens.json (%v); run reference/dump_kev_tokens.py", err)
	}
	var ref struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
			NFC  bool    `json:"nfc"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(buf, &ref); err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		got, err := e.UserTokens(c.Text)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		same := slices.Equal(got, c.IDs)
		if c.NFC && !same {
			t.Errorf("%q:\n got %v\nwant %v", c.Text, got, c.IDs)
		}
		if !c.NFC && same {
			t.Errorf("%q is not NFC and still tokenised the same: the NFC gap may be closed", c.Text)
		}
	}
}

// TestRecordMatchesKev checks rendering and the question mapping: the text
// the model reads, byte for byte, and the keys answers are reported under.
func TestRecordMatchesKev(t *testing.T) {
	for _, fx := range loadFixtures(t) {
		req, err := ParseRequest(fx.Request)
		if err != nil {
			t.Fatalf("%s: %v", fx.Name, err)
		}
		rec, meta := ToRecord(req)
		want := fx.Ref.Record
		if rec.State != want.State {
			t.Errorf("%s: state\n got %q\nwant %q", fx.Name, rec.State, want.State)
		}
		if len(rec.Questions) != len(want.Questions) {
			t.Fatalf("%s: %d questions, want %d", fx.Name, len(rec.Questions), len(want.Questions))
		}
		for k, q := range rec.Questions {
			if q.Instr != want.Questions[k].Instr {
				t.Errorf("%s q%d: instr %q, want %q", fx.Name, k, q.Instr, want.Questions[k].Instr)
			}
			if !slices.Equal(q.Options, want.Questions[k].Options) {
				t.Errorf("%s q%d: options\n got %q\nwant %q", fx.Name, k, q.Options, want.Questions[k].Options)
			}
			m, wm := meta[k], fx.Ref.Meta[k]
			if m.ID != wm.ID || m.Type != wm.Type || !slices.Equal(m.Keys, wm.Keys) {
				t.Errorf("%s q%d: meta %+v, want %+v", fx.Name, k, m, wm)
			}
			for j, l := range m.Legend {
				if wm.Legend[m.Keys[j]] != l {
					t.Errorf("%s q%d: legend[%d] %q, want %q", fx.Name, k, j, l, wm.Legend[m.Keys[j]])
				}
			}
		}
	}
}

// TestEncodingMatchesKev is K2's gate: the packed ids, segments, positions,
// option markers and readout indices, identical to Kev's encode().
func TestEncodingMatchesKev(t *testing.T) {
	e := loadEncoder(t)
	for _, fx := range loadFixtures(t) {
		req, err := ParseRequest(fx.Request)
		if err != nil {
			t.Fatal(err)
		}
		rec, _ := ToRecord(req)
		enc, err := e.Encode(rec, MaxState, MaxRow)
		if err != nil {
			t.Fatalf("%s: %v", fx.Name, err)
		}
		w := fx.Ref.Enc
		for _, c := range []struct {
			name      string
			got, want []int32
		}{{"ids", enc.IDs, w.IDs}, {"seg", enc.Seg, w.Seg}, {"pos", enc.Pos, w.Pos}, {"opt", enc.Opt, w.Opt}} {
			if !slices.Equal(c.got, c.want) {
				t.Errorf("%s: %s differ\n got %v\nwant %v", fx.Name, c.name, c.got, c.want)
			}
		}
		if !slices.Equal(enc.Decide, w.DecideIdx) {
			t.Errorf("%s: decide %v, want %v", fx.Name, enc.Decide, w.DecideIdx)
		}
		for k := range w.OptIdx {
			if !slices.Equal(enc.Opts[k], w.OptIdx[k]) {
				t.Errorf("%s q%d: opts %v, want %v", fx.Name, k, enc.Opts[k], w.OptIdx[k])
			}
		}
		if enc.StateTruncated != fx.Ref.StateTruncated || enc.StateLen != fx.Ref.StateLen {
			t.Errorf("%s: state %d truncated=%v, want %d %v", fx.Name, enc.StateLen, enc.StateTruncated, fx.Ref.StateLen, fx.Ref.StateTruncated)
		}
		if len(enc.IDs) != fx.Ref.Usage.InputTokens {
			t.Errorf("%s: input_tokens %d, want %d", fx.Name, len(enc.IDs), fx.Ref.Usage.InputTokens)
		}
	}
}

// TestAnswersMatchKev feeds Kev's own probabilities through ToAnswers: the
// serialised answers must be json.dumps's bytes, and output_tokens its count.
func TestAnswersMatchKev(t *testing.T) {
	e := loadEncoder(t)
	for _, fx := range loadFixtures(t) {
		req, err := ParseRequest(fx.Request)
		if err != nil {
			t.Fatal(err)
		}
		_, meta := ToRecord(req)
		got := string(AnswersJSON(ToAnswers(fx.Ref.Probs, meta), true))
		if got != fx.Ref.AnswersJSON {
			t.Errorf("%s:\n got %s\nwant %s", fx.Name, got, fx.Ref.AnswersJSON)
		}
		ids, err := e.Tok.Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != fx.Ref.Usage.OutputTokens {
			t.Errorf("%s: output_tokens %d, want %d", fx.Name, len(ids), fx.Ref.Usage.OutputTokens)
		}
		var any map[string]any
		if err := json.Unmarshal(AnswersJSON(ToAnswers(fx.Ref.Probs, meta), false), &any); err != nil {
			t.Errorf("%s: compact answers are not JSON: %v", fx.Name, err)
		}
	}
}

func TestPyFloatRepr(t *testing.T) {
	for _, c := range []struct {
		f    float64
		want string
	}{
		{0, "0.0"}, {1, "1.0"}, {-2, "-2.0"}, {0.1, "0.1"}, {59.9, "59.9"}, {0.0001, "0.0001"},
		{0.00001, "1e-05"}, {1e15, "1000000000000000.0"}, {1e16, "1e+16"}, {1e21, "1e+21"},
		{123456789.125, "123456789.125"}, {1.5e-7, "1.5e-07"}, {2.5e300, "2.5e+300"}, {0.5093, "0.5093"},
		{1.0 / 3, "0.3333333333333333"},
	} {
		if got := PyFloatRepr(c.f); got != c.want {
			t.Errorf("repr(%v) = %q, want %q", c.f, got, c.want)
		}
	}
}

func TestRenderScalars(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{`"x"`, "x"}, {`null`, ""}, {`true`, "True"}, {`false`, "False"}, {`2`, "2"}, {`2.0`, "2.0"},
		{`-0`, "0"}, {`1E2`, "100.0"}, {`123456789012345678901234567890`, "123456789012345678901234567890"},
		{`[]`, ""}, {`{}`, ""}, {`["  padded", "\u001ftab"]`, "- padded\n- tab"},
		{`{"a": {"b": [1, {"c": null}]}}`, "a:\n  b:\n    - 1\n    - c: "},
	} {
		var v Value
		if err := json.Unmarshal([]byte(c.in), &v); err != nil {
			t.Fatal(err)
		}
		if got := Render(v); got != c.want {
			t.Errorf("render(%s) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestParseRequestRefuses(t *testing.T) {
	for _, body := range []string{
		`[]`, `{"questions": {"a": {"type": "noul"}}}`, `{"state": "x"}`, `{"state": "x", "questions": {}}`,
		`{"state": "x", "questions": {"a": {"type": "bool"}}}`, `{"state": "x", "questions": {"a": {"type": "choice"}}}`,
		`{"state": "x", "questions": {"a": {"type": "choice", "criteria": {}}}}`,
		`{"state": "x", "questions": {"a": {"type": "score", "criteria": {"a": 1}}}}`,
		`{"state": "x", "questions": {"a": {"type": "noul", "criteria": ["yes"]}}}`,
		`{"state": "x", "model": 3, "questions": {"a": {"type": "noul"}}}`,
	} {
		_, err := ParseRequest([]byte(body))
		var ve *ValidationError
		if !errors.As(err, &ve) {
			t.Errorf("%s: accepted (%v)", body, err)
		}
	}
	req, err := ParseRequest([]byte(`{"state": null, "questions": {"a": {"type": "noul", "criteria": null, "extra": 1}}}`))
	if err != nil || req.Model != DefaultModel {
		t.Errorf("a minimal request: %v %+v", err, req)
	}
}

// TestKeyOrderAndDuplicates: options are the criteria in the caller's order,
// and a repeated key is Python's dict assignment.
func TestKeyOrderAndDuplicates(t *testing.T) {
	req, err := ParseRequest([]byte(`{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"z": null, "a": "A", "z": "Z"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	rec, meta := ToRecord(req)
	if !slices.Equal(rec.Questions[0].Options, []string{"z: Z", "a: A"}) || !slices.Equal(meta[0].Keys, []string{"z", "a"}) {
		t.Errorf("options %q keys %q", rec.Questions[0].Options, meta[0].Keys)
	}
}
