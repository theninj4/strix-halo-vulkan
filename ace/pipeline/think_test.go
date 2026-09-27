package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/plan"
)

// lmRef is reference/dump_ace_lm.py: upstream's thinking path, recorded up
// to the DiT's door.
const lmRef = "../../reference/out/acelm"

// TestThinkingInputs is A8's gate on everything between the LM and the DiT,
// from the oracle's own CoT and codes: the DiT request rebuilt from the CoT
// (caption, metas, language) must encode to the caption and lyric states
// upstream handed the DiT, and the codes must detokenize to its hints.
func TestThinkingInputs(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(lmRef, "manifest.json"))
	if err != nil {
		t.Skipf("no LM reference (%v)", err)
	}
	var m struct {
		Cases map[string]struct {
			Request map[string]any `json:"request"`
			Phases  []struct {
				Rows int    `json:"rows"`
				Text string `json:"text"`
			} `json:"phases"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	p := pipeline(t)
	for _, label := range []string{"given_duration", "all_metas"} {
		c := m.Cases[label]
		q := c.Request
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

		cotRan := len(c.Phases) == 2
		meta := lm.UserMeta(r)
		if cotRan {
			meta = lm.ParseCoT(c.Phases[0].Text)
		}
		d := DiTRequest(r, meta, cotRan)

		raw, err := os.ReadFile(filepath.Join(lmRef, label+"_p"+string(rune('0'+len(c.Phases)-1))+"_tokens.bin"))
		if err != nil {
			t.Fatal(err)
		}
		var codes []int
		for i := 0; i+4 <= len(raw); i += 4 {
			tok := int(int32(binary.LittleEndian.Uint32(raw[i:])))
			if tok == lm.ImEndID {
				break
			}
			codes = append(codes, tok-lm.CodeBase)
		}
		T := max(plan.MinLatents, lm.CodesPerSecond*len(codes))

		text, lyric, err := p.Condition(d)
		if err != nil {
			t.Fatal(err)
		}
		want := readF32(t, filepath.Join(lmRef, label+"_dit_text_hidden_states.bin"))
		if len(want) != len(text.Data) {
			t.Fatalf("%s: caption states %v, want %d rows (DiT caption %q, metas %q)", label, text, len(want)/1024,
				d.Caption, d.Metas())
		}
		_, rms, _ := gap(text.Data, want)
		wantL := readF32(t, filepath.Join(lmRef, label+"_dit_lyric_hidden_states.bin"))
		if len(wantL) != len(lyric.Data) {
			t.Fatalf("%s: lyric rows %v, want %d", label, lyric, len(wantL)/1024)
		}
		for i := range wantL {
			if lyric.Data[i] != wantL[i] {
				t.Fatalf("%s: lyric embed %d differs (language %q)", label, i, d.Language)
			}
		}
		hints, err := p.Hints(codes, T)
		if err != nil {
			t.Fatal(err)
		}
		wantH := readF32(t, filepath.Join(lmRef, label+"_dit_precomputed_lm_hints_25Hz.bin"))
		if len(wantH) != len(hints) {
			t.Fatalf("%s: %d hint values, want %d", label, len(hints), len(wantH))
		}
		_, hrms, _ := gap(hints, wantH)
		t.Logf("%-14s %d codes -> T %d; caption %v rms %.2e; lyrics exact; hints rms %.2e", label, len(codes), T,
			text, rms, hrms)
		if rms > 5e-3 || hrms > 1e-3 {
			t.Errorf("%s: caption rms %.2e, hints rms %.2e", label, rms, hrms)
		}
	}
}
