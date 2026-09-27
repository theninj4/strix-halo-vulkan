package ocr

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/zimage/qwen"
)

// imageTokenID is <|IMAGE_PLACEHOLDER|>.
const imageTokenID = 100295

// promptRows is a record's prompt as LM rows in slot 0: text rows by token
// id (so the host's embedding lookup is exercised), image rows from the
// dump's inputs_embeds.
func promptRows(t *testing.T, rec record, embeds []float32) []Row {
	t.Helper()
	rows := make([]Row, len(rec.InputIDs))
	for i, id := range rec.InputIDs {
		rows[i] = Row{ID: id, Pos: i,
			Rope: [3]int32{rec.PositionIDs[0][i], rec.PositionIDs[1][i], rec.PositionIDs[2][i]}}
		if id == imageTokenID {
			rows[i].Embed = embeds[i*lmHidden : (i+1)*lmHidden]
		}
	}
	return rows
}

// genRow is generated token j's row: cache position L+j, and on every rope
// axis that plus the record's rope delta.
func genRow(rec record, j int, id int32) Row {
	p := int32(len(rec.InputIDs)+j) + rec.RopeDelta
	return Row{ID: id, Pos: len(rec.InputIDs) + j, Rope: [3]int32{p, p, p}}
}

// logitStats is the relative error, KL(ref || got) and whether the argmaxes
// agree, for one row of logits.
func logitStats(got, want []float32) (rel, kl float64, agree bool) {
	rel = relErr(got, want)
	lse := func(v []float32) float64 {
		m := math.Inf(-1)
		for _, x := range v {
			m = max(m, float64(x))
		}
		var s float64
		for _, x := range v {
			s += math.Exp(float64(x) - m)
		}
		return m + math.Log(s)
	}
	lg, lw := lse(got), lse(want)
	for i := range want {
		pw := float64(want[i]) - lw
		kl += math.Exp(pw) * (pw - (float64(got[i]) - lg))
	}
	return rel, kl, argmax(got) == argmax(want)
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// The bounds, measured (OCR.md O4) over all six cases: worst layer rel
// 7.5e-4, worst logits rel 6.3e-3 (a teacher-forced step of the chart),
// worst KL 5.3e-6. The 1-D rope control moves the prompt logits by rel 0.24,
// KL 6.7e-3 (TestLMRopeControl).
const (
	lmLayerTol = 5e-3 // any layer's residual, any case
	lmLogitTol = 2e-2 // rel, prompt and teacher-forced logits
	lmKLTol    = 1e-4 // per step
)

// TestLM holds the device ERNIE to the fp32 reference on every case: each
// layer's output over the prompt, the prompt's logits, the teacher-forced
// decode steps, and then greedy generation token for token (on the five
// element cases; the page has an fp16 near-tie, OCR.md O0, and is reported).
func TestLM(t *testing.T) {
	recs := records(t)
	dev, done := newTestDevice(t)
	defer done()
	start := time.Now()
	lm, err := LoadLM(dev, modelDir, DefaultLMOptions())
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer lm.Destroy()
	t.Logf("staged in %v", time.Since(start).Round(time.Millisecond))

	sort.Slice(recs, func(i, j int) bool { return recs[i].Name < recs[j].Name })
	for _, rec := range recs {
		t.Run(rec.Name, func(t *testing.T) {
			names := []string{"lm.embed", "logits.prompt", "logits.tf"}
			for i := 0; i < lmLayers; i++ {
				names = append(names, fmt.Sprintf("lm.layer%d", i))
			}
			ref := refTensors(t, rec.Name, names)
			rows := promptRows(t, rec, ref["lm.embed"])

			worst := 0.0
			lm.Tap = func(name string, got *qwen.Mat) {
				want := ref[name]
				e := relErr(got.Data, want)
				worst = max(worst, e)
				if e > lmLayerTol {
					t.Errorf("%s: rel %.2e over %.0e", name, e, lmLayerTol)
				}
			}
			if _, _, err := lm.Pass(rows, 1); err != nil {
				t.Fatal(err)
			}
			lm.Tap = nil
			wall := time.Now()
			lg, gpu, err := lm.Pass(rows, 1)
			if err != nil {
				t.Fatal(err)
			}
			pre := time.Since(wall)
			rel, kl, agree := logitStats(lg[0], ref["logits.prompt"])
			t.Logf("prefill %d rows: %v (%v on the device); worst layer rel %.2e; prompt logits rel %.2e KL %.1e agree %v",
				len(rows), pre.Round(time.Millisecond), gpu.Round(time.Millisecond), worst, rel, kl, agree)
			if rel > lmLogitTol || kl > lmKLTol || !agree {
				t.Errorf("prompt logits: rel %.2e KL %.1e agree %v", rel, kl, agree)
			}

			// Teacher-forced: step j feeds generated[j-1] and is held to row j.
			tf := ref["logits.tf"]
			steps := len(tf) / lm.Vocab()
			var worstRel, worstKL float64
			for j := 1; j < steps; j++ {
				lg, _, err := lm.Pass([]Row{genRow(rec, j-1, rec.Generated[j-1])}, 1)
				if err != nil {
					t.Fatal(err)
				}
				rel, kl, agree := logitStats(lg[0], tf[j*lm.Vocab():(j+1)*lm.Vocab()])
				worstRel, worstKL = max(worstRel, rel), max(worstKL, kl)
				if rel > lmLogitTol || kl > lmKLTol || !agree {
					t.Errorf("tf step %d: rel %.2e KL %.1e agree %v", j, rel, kl, agree)
				}
			}
			t.Logf("teacher-forced %d steps: worst rel %.2e, worst KL %.1e", steps-1, worstRel, worstKL)

			// Greedy from the prompt, against the reference's generation.
			if _, _, err := lm.Pass(rows[:len(rows)-1], 0); err != nil {
				t.Fatal(err)
			}
			next := rows[len(rows)-1]
			var got []int32
			wall = time.Now()
			for len(got) < len(rec.Generated) {
				lg, _, err := lm.Pass([]Row{next}, 1)
				if err != nil {
					t.Fatal(err)
				}
				id := int32(argmax(lg[0]))
				got = append(got, id)
				if id == EOS {
					break
				}
				next = genRow(rec, len(got)-1, id)
			}
			dec := time.Since(wall)
			first := -1
			for i := range got {
				if i >= len(rec.Generated) || got[i] != rec.Generated[i] {
					first = i
					break
				}
			}
			if first < 0 && len(got) != len(rec.Generated) {
				first = len(got)
			}
			t.Logf("greedy %d tokens in %v (%.2f ms a token); first divergence %d of %d",
				len(got), dec.Round(time.Millisecond), float64(dec.Microseconds())/1e3/float64(len(got)), first, len(rec.Generated))
			if first >= 0 && rec.Name != "page" {
				t.Errorf("greedy diverges at token %d of %d", first, len(rec.Generated))
			}
		})
	}
}

// TestLMRopeControl gives every row a plain 1-D rope at its cache index,
// which is what a Qwen3 port does and what this model must not: the image
// rows' (p, p+r, p+c) and the text's resumption at p + max(mh, mw) both go.
// The gate above has to be able to tell.
func TestLMRopeControl(t *testing.T) {
	var rec record
	for _, r := range records(t) {
		if r.Name == "chart" {
			rec = r
		}
	}
	if rec.Name == "" {
		t.Skip("no chart case in the reference")
	}
	dev, done := newTestDevice(t)
	defer done()
	lm, err := LoadLM(dev, modelDir, LMOptions{MaxLen: 1024, Slots: 1, Rows: 512})
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	defer lm.Destroy()
	ref := refTensors(t, rec.Name, []string{"lm.embed", "logits.prompt"})
	rows := promptRows(t, rec, ref["lm.embed"])
	for i := range rows {
		p := int32(rows[i].Pos)
		rows[i].Rope = [3]int32{p, p, p}
	}
	lg, _, err := lm.Pass(rows, 1)
	if err != nil {
		t.Fatal(err)
	}
	rel, kl, agree := logitStats(lg[0], ref["logits.prompt"])
	t.Logf("1-D rope: prompt logits rel %.2e KL %.1e agree %v", rel, kl, agree)
	if rel <= lmLogitTol*5 && kl <= lmKLTol*5 {
		t.Errorf("the control moved the logits by only rel %.2e KL %.1e", rel, kl)
	}
}
