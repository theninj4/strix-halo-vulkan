package lm

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The references: reference/dump_ace_lm.py (upstream's LLMHandler on the
// thinking path, fp32 on the CPU; a recorded sampled run, since upstream's
// LM sampling is unseeded) and reference/dump_ace_yaml.py (yaml.dump through
// _format_metadata_as_cot, fuzzed).
const (
	lmRef   = "../../reference/out/acelm"
	yamlRef = "../../reference/out/aceyaml"
	lmDir   = "../../models/acestep-5Hz-lm-4B"
)

type lmManifest struct {
	Cases map[string]struct {
		Request map[string]any `json:"request"`
		Phases  []struct {
			Rows  int    `json:"rows"`
			Text  string `json:"text"`
			Steps []struct {
				FSM        string  `json:"fsm"`
				FSMAllowed int     `json:"fsm_allowed"`
				FSMIDs     []int32 `json:"fsm_ids"`
				Allowed    int     `json:"allowed"`
				AllowedIDs []int32 `json:"allowed_ids"`
				Token      int32   `json:"token"`
			} `json:"steps"`
		} `json:"phases"`
		Prompts [][]string `json:"prompts"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int  `json:"shape"`
		Dtype string `json:"dtype"`
	} `json:"tensors"`
}

func loadLMManifest(t testing.TB) *lmManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(lmRef, "manifest.json"))
	if err != nil {
		t.Skipf("no LM reference (%v); run reference/dump_ace_lm.py", err)
	}
	var m lmManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func readRef(t testing.TB, m *lmManifest, name string) ([]int, []byte) {
	t.Helper()
	info, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("no tensor %s in the reference", name)
	}
	buf, err := os.ReadFile(filepath.Join(lmRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	return info.Shape, buf
}

func refInt32(t testing.TB, m *lmManifest, name string) ([]int, []int32) {
	shape, buf := readRef(t, m, name)
	out := make([]int32, len(buf)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return shape, out
}

func refFloat32(t testing.TB, m *lmManifest, name string) ([]int, []float32) {
	shape, buf := readRef(t, m, name)
	out := make([]float32, len(buf)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return shape, out
}

// promptRows is a phase's prompt as the model reads each row: left pads
// (mask 0) dropped, which RoPE makes a no-op (MUSIC.md § A7 — the oracle).
func promptRows(t testing.TB, m *lmManifest, label string, phase int) [][]int32 {
	shape, ids := refInt32(t, m, label+"_p"+string(rune('0'+phase))+"_prompt_ids")
	_, mask := refInt32(t, m, label+"_p"+string(rune('0'+phase))+"_prompt_mask")
	rows := make([][]int32, shape[0])
	for r := range rows {
		for j := 0; j < shape[1]; j++ {
			if mask[r*shape[1]+j] != 0 {
				rows[r] = append(rows[r], ids[r*shape[1]+j])
			}
		}
	}
	return rows
}

func loadTokenizer(t testing.TB) *tokenizer.Tokenizer {
	t.Helper()
	tk, err := tokenizer.Load(lmDir)
	if err != nil {
		t.Skipf("no LM checkpoint: %v", err)
	}
	return tk
}

// TestYAML checks the phase-2 CoT against yaml.dump on 3,100 metadata dicts:
// folding at width 80, quoting, implicit-type strings, unicode.
func TestYAML(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(yamlRef, "cots.json"))
	if err != nil {
		t.Skipf("no YAML reference (%v); run reference/dump_ace_yaml.py", err)
	}
	var cases []struct {
		Meta Meta   `json:"meta"`
		CoT  string `json:"cot"`
	}
	if err := json.Unmarshal(buf, &cases); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i, c := range cases {
		if got := c.Meta.CoT(); got != c.CoT {
			bad++
			if bad <= 5 {
				t.Errorf("case %d %q:\n got %q\nwant %q", i, c.Meta, got, c.CoT)
			}
		}
	}
	t.Logf("%d of %d CoTs exact", len(cases)-bad, len(cases))
}

// TestPrompts rebuilds every prompt the oracle's LM read -- phase 1's, and
// phase 2's two rows, whose CoT is phase 1's output parsed and re-dumped
// (or the request's own metas) -- and checks the text and the token ids.
func TestPrompts(t *testing.T) {
	m := loadLMManifest(t)
	tk := loadTokenizer(t)
	for label, c := range m.Cases {
		caption, _ := c.Request["caption"].(string)
		lyrics, _ := c.Request["lyrics"].(string)
		var want [][]string
		var got [][]string
		phase2 := 0
		if len(c.Phases) == 2 {
			// Phase 1 ran: its prompt, and its sampled text parsed.
			want = append(want, c.Prompts[0])
			got = append(got, []string{Phase1Prompt(caption, lyrics)})
			phase2 = 1
		}
		var meta Meta
		if phase2 == 1 {
			meta = ParseCoT(c.Phases[0].Text)
		} else {
			meta = UserMeta(requestOf(c.Request))
		}
		want = append(want, c.Prompts[phase2])
		got = append(got, []string{Phase2Prompt(caption, lyrics, meta.CoT()), Phase2Uncond()})

		for ph := range want {
			rows := promptRows(t, m, label, ph)
			for r := range want[ph] {
				// The dumped text is the decoded padded row; drop the pads.
				w := want[ph][r]
				for len(w) >= len("<|endoftext|>") && w[:len("<|endoftext|>")] == "<|endoftext|>" {
					w = w[len("<|endoftext|>"):]
				}
				if got[ph][r] != w {
					t.Errorf("%s phase %d row %d text:\n got %q\nwant %q", label, ph, r, got[ph][r], w)
					continue
				}
				ids, err := tk.Encode(got[ph][r])
				if err != nil {
					t.Fatal(err)
				}
				if at := equalIDs(ids, rows[r]); at >= 0 {
					t.Errorf("%s phase %d row %d: ids differ at %d (%d vs %d)", label, ph, r, at, len(ids), len(rows[r]))
					continue
				}
				t.Logf("%s phase %d row %d: %d ids exact", label, ph, r, len(ids))
			}
		}
	}
}

// requestOf is a dumped request as a plan.Request.
func requestOf(q map[string]any) *plan.Request {
	r := &plan.Request{}
	r.Caption, _ = q["caption"].(string)
	r.Lyrics, _ = q["lyrics"].(string)
	if v, ok := q["bpm"].(float64); ok {
		r.BPM = int(v)
	}
	r.KeyScale, _ = q["keyscale"].(string)
	r.TimeSignature, _ = q["timesignature"].(string)
	r.Duration, _ = q["duration"].(float64)
	r.Language, _ = q["vocal_language"].(string)
	return r
}
