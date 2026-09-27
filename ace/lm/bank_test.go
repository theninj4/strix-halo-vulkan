package lm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

// tfRef is reference/dump_ace_lm_tf.py's output: fp32 logits at every
// teacher-forced step, and what bf16 (upstream's CUDA dtype) drifts from them.
const tfRef = "../../reference/out/acelmtf"

type tfCase struct {
	Rows, Lo, Hi, Steps int
	BF16KL              [][]float64 `json:"bf16_kl"`
	BF16CFGKL           []float64   `json:"bf16_cfg_kl"`
	BF16CFGArgmax       []bool      `json:"bf16_cfg_argmax"`
	BF16FSMKL           []*float64  `json:"bf16_fsm_kl"`
	BF16FSMArgmax       []*bool     `json:"bf16_fsm_argmax"`
}

// TestBankAgainstBF16 is A10's price: the first 101 teacher-forced steps of
// every sampled phase, the bank's KL from fp32 at temperature 0.85 beside
// upstream bf16's on the same steps (MUSIC.md A10).
//
//   - Phase 1 is over what the FSM allows at each step, as the oracle
//     recorded it; a step where it forces one token chooses nothing and is
//     left out (bf16's worst full-vocabulary steps are those).
//   - Phase 2 is over the codes, each CFG row and after CFG.
//
// The gate pools each kind of phase over its cases (a phase-1 case's worst
// step is one sensitive position): the bank's mean KL no worse than bf16's,
// and its worst step within 2x of bf16's worst. That is a quantisation
// inside what upstream itself serves. int8 measures 0.9x and 1.4x on the
// text steps and 0.02x and 0.06x on the codes.
func TestBankAgainstBF16(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(tfRef, "manifest.json"))
	if err != nil {
		t.Skipf("no teacher-forced reference (%v); run reference/dump_ace_lm_tf.py", err)
	}
	var tf struct{ Cases map[string]tfCase }
	if err := json.Unmarshal(buf, &tf); err != nil {
		t.Fatal(err)
	}
	m := loadLMManifest(t)
	g := loadGPU(t)
	names := make([]string, 0, len(tf.Cases))
	for k := range tf.Cases {
		names = append(names, k)
	}
	sort.Strings(names)
	pool := map[string]*[2][]float64{"text": {}, "codes": {}}
	for _, name := range names {
		if only := os.Getenv("ACE_LM_TF_CASE"); only != "" && only != name {
			continue
		}
		c := tf.Cases[name]
		var ph int
		if _, err := fmt.Sscanf(name[len(name)-3:], "_p%d", &ph); err != nil {
			t.Fatal(name, err)
		}
		label := name[:len(name)-3]
		rows := promptRows(t, m, label, ph)
		_, toks := refInt32(t, m, fmt.Sprintf("%s_p%d_tokens", label, ph))
		fsm := m.Cases[label].Phases[ph].Steps
		raw, err := os.ReadFile(filepath.Join(tfRef, name+"_logits.bin"))
		if err != nil {
			t.Fatal(err)
		}
		w := c.Hi - c.Lo
		wantAt := func(step, row int) []float32 {
			off := (step*c.Rows + row) * w * 4
			out := make([]float32, w)
			for i := range out {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off+4*i:]))
			}
			return out
		}

		var pre []Tok
		for s, r := range rows {
			for i, id := range r[:len(r)-1] {
				pre = append(pre, Tok{ID: id, Slot: s, Pos: i})
			}
		}
		if _, _, err := g.Pass(pre, Range{}); err != nil {
			t.Fatal(err)
		}
		step := make([]Tok, len(rows))
		for s, r := range rows {
			step[s] = Tok{ID: r[len(r)-1], Slot: s, Pos: len(r) - 1}
		}
		var ours, theirs []float64
		worstStep := 0
		counted, agree, bfAgree := 0, 0, 0
		var tDec time.Duration
		for i := 0; i < c.Steps; i++ {
			t0 := time.Now()
			out, _, err := g.Pass(step, Range{Lo: c.Lo, Hi: c.Hi})
			if err != nil {
				t.Fatal(err)
			}
			tDec += time.Since(t0)
			note := func(kl float64) {
				if len(ours) == 0 || kl > ours[worstStep] {
					worstStep = len(ours)
				}
				ours = append(ours, kl)
			}
			if c.Rows == 2 {
				gc := cfg(out[0], out[1], 2)
				wc := cfg(wantAt(i, 0), wantAt(i, 1), 2)
				st := compareLogits(gc, wc, 0, w)
				note(st.kl)
				theirs = append(theirs, c.BF16CFGKL[i])
				for r := range rows {
					note(compareLogits(out[r], wantAt(i, r), 0, w).kl)
					theirs = append(theirs, c.BF16KL[i][r])
				}
				counted++
				if st.argmax {
					agree++
				}
				if c.BF16CFGArgmax[i] {
					bfAgree++
				}
			} else if c.BF16FSMKL[i] != nil {
				got, want := out[0], wantAt(i, 0)
				if ids := fsm[i].FSMIDs; len(ids) > 0 {
					got, want = gather(got, ids, c.Lo), gather(want, ids, c.Lo)
				}
				st := compareLogits(got, want, 0, len(got))
				note(st.kl)
				theirs = append(theirs, *c.BF16FSMKL[i])
				counted++
				if st.argmax {
					agree++
				}
				if *c.BF16FSMArgmax[i] {
					bfAgree++
				}
			}
			if i+1 < c.Steps {
				for s := range step {
					step[s].ID = toks[i]
					step[s].Pos++
				}
			}
		}
		om, ox := meanMax(ours)
		bm, bx := meanMax(theirs)
		t.Logf("%-18s %s: KL mean %.2e max %.2e, argmax %d/%d | bf16: KL mean %.2e max %.2e, argmax %d/%d | %.1f ms a step",
			name, g.o.Bank, om, ox, agree, counted, bm, bx, bfAgree, counted,
			float64(tDec.Microseconds())/1e3/float64(c.Steps))
		if ox > bx {
			t.Logf("%s: the bank's worst is entry %d", name, worstStep)
		}
		kind := pool["text"]
		if c.Rows == 2 {
			kind = pool["codes"]
		}
		kind[0], kind[1] = append(kind[0], ours...), append(kind[1], theirs...)
	}
	for _, k := range []string{"text", "codes"} {
		if len(pool[k][0]) == 0 {
			continue
		}
		om, ox := meanMax(pool[k][0])
		bm, bx := meanMax(pool[k][1])
		t.Logf("%s, pooled: KL mean %.2e (%.2fx bf16's) max %.2e (%.2fx)", k, om, om/bm, ox, ox/bx)
		if !(om <= bm && ox <= 2*bx) { // NaN fails too
			t.Errorf("%s: the bank's KL (mean %.2e, max %.2e) is past bf16's (%.2e, 2 x %.2e)", k, om, ox, bm, bx)
		}
	}
}

// gather is x's entries at vocabulary ids, x starting at id lo.
func gather(x []float32, ids []int32, lo int) []float32 {
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = x[int(id)-lo]
	}
	return out
}

func meanMax(x []float64) (mean, mx float64) {
	for _, v := range x {
		mean += v
		mx = nanMax(mx, v)
	}
	return mean / float64(len(x)), mx
}

// TestSlabLadder times each projection's split-K rungs at decode (one row,
// and two for CFG), one projection at a time with the rest at the table's,
// as whole steps at a fixed position (MUSIC.md A10). ACE_LM_LADDER=1 builds
// every rung; without it the test skips.
func TestSlabLadder(t *testing.T) {
	if os.Getenv("ACE_LM_LADDER") == "" {
		t.Skip("ACE_LM_LADDER=1 to run")
	}
	g := loadGPU(t)
	var pre []Tok
	for s := 0; s < 2; s++ {
		for i := 0; i < 128; i++ {
			pre = append(pre, Tok{ID: int32(1000 + i), Slot: s, Pos: i})
		}
	}
	if _, _, err := g.Pass(pre, Range{}); err != nil {
		t.Fatal(err)
	}
	stepMs := func(rows int) float64 {
		step := make([]Tok, rows)
		for s := range step {
			step[s] = Tok{ID: 2000, Slot: s, Pos: 128}
		}
		best := math.Inf(1)
		for trial := 0; trial < 3; trial++ {
			var tot time.Duration
			for i := 0; i < 10; i++ {
				_, d, err := g.Pass(step, Range{Lo: CodeBase, Hi: CodeBase + NumCodes})
				if err != nil {
					t.Fatal(err)
				}
				tot += d
			}
			best = math.Min(best, float64(tot.Microseconds())/1e4)
		}
		return best
	}
	for rows := 1; rows <= 2; rows++ {
		t.Logf("rows %d, the table's: %.2f ms a step", rows, stepMs(rows))
		for p := pQ; p < nProj; p++ {
			kt := projShape[p][1] / tile
			line := fmt.Sprintf("rows %d %-4s (K %d):", rows, projNames[p], projShape[p][1])
			for _, s := range []int{1, 2, 4, 8, 16, 20, 32, 40} {
				if kt%s != 0 || (kt/s)%4 != 0 {
					continue
				}
				g.slabOverride[p] = s
				line += fmt.Sprintf(" k%d %.2f", s, stepMs(rows))
			}
			g.slabOverride[p] = 0
			t.Log(line)
		}
	}
}
