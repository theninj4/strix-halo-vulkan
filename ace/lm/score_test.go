package lm

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// The reference: reference/dump_ace_score.py (MUSIC.md A14b), upstream's
// calculate_pmi_score_per_condition in fp32 on the CPU, and --bf16 the
// same run the way upstream serves it on CUDA.
const (
	scoreRef     = "../../reference/out/acescore"
	scoreBF16Ref = "../../reference/out/acescore_bf16"
)

var pmiLabels = []string{"pmi_match", "pmi_mismatch"}

type scoreManifest struct {
	Cases map[string]struct {
		CodesFrom string         `json:"codes_from"`
		Caption   string         `json:"caption"`
		Lyrics    string         `json:"lyrics"`
		Metadata  map[string]any `json:"metadata"`
		Calls     []struct {
			Unconditional bool    `json:"unconditional"`
			Target        string  `json:"target"`
			PromptLen     int     `json:"prompt_len"`
			TargetLen     int     `json:"target_len"`
			MeanLogp      float64 `json:"mean_logp"`
		} `json:"calls"`
		Scores map[string]float64 `json:"scores"`
		Reward float64            `json:"reward"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
	} `json:"tensors"`
}

func loadScoreManifest(t testing.TB, dir string) *scoreManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no score reference (%v); run reference/dump_ace_score.py", err)
	}
	var m scoreManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func scoreRaw(t testing.TB, dir, name string) []uint32 {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]uint32, len(buf)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(buf[4*i:])
	}
	return out
}

func scoreInts(t testing.TB, dir, name string) []int32 {
	raw := scoreRaw(t, dir, name)
	out := make([]int32, len(raw))
	for i, v := range raw {
		out[i] = int32(v)
	}
	return out
}

func scoreFloats(t testing.TB, dir, name string) []float32 {
	raw := scoreRaw(t, dir, name)
	out := make([]float32, len(raw))
	for i, v := range raw {
		out[i] = math.Float32frombits(v)
	}
	return out
}

// pmiConditions are a case's conditions as the oracle passed them.
func pmiConditions(t testing.TB, m *scoreManifest, label string) (Conditions, []int) {
	c := m.Cases[label]
	meta := Meta{}
	for k, v := range c.Metadata {
		if k == "caption" {
			continue
		}
		switch x := v.(type) {
		case float64:
			meta[k] = strconv.Itoa(int(x))
		case string:
			meta[k] = x
		default:
			t.Fatalf("%s: metadata %s is %T", label, k, v)
		}
	}
	lm := loadLMManifest(t)
	_, codes32 := refInt32(t, lm, c.CodesFrom+"_codes")
	codes := make([]int, len(codes32))
	for i, v := range codes32 {
		codes[i] = int(v)
	}
	return Conditions{Caption: c.Caption, Lyrics: c.Lyrics, Meta: meta}, codes
}

// scoreCall is one of the oracle's teacher-forced forwards, as ours runs it.
type scoreCall struct {
	slot   int
	prompt []int32
	target []int32
}

// scoreCalls are ScoreTargets in upstream's call order (each PMI target
// twice, conditional then unconditional), checked against the oracle's
// targets and ids.
func scoreCalls(t *testing.T, p *Planner, m *scoreManifest, label string, c Conditions, codes []int) []scoreCall {
	cond, uncond, err := p.ScorePrompts(codes)
	if err != nil {
		t.Fatal(err)
	}
	keys, texts, pmi := ScoreTargets(c)
	var calls []scoreCall
	for i := range keys {
		target, err := p.Tk.Encode(texts[i])
		if err != nil {
			t.Fatal(err)
		}
		calls = append(calls, scoreCall{0, cond, target})
		if pmi[i] {
			calls = append(calls, scoreCall{1, uncond, target})
		}
	}
	want := m.Cases[label].Calls
	if len(calls) != len(want) {
		t.Fatalf("%s: %d calls, upstream made %d", label, len(calls), len(want))
	}
	for i, call := range calls {
		k := 0
		for ; k < len(keys) && texts[k] != want[i].Target; k++ {
		}
		if k == len(keys) {
			t.Errorf("%s: call %d's target %q is none of ours", label, i, want[i].Target)
		}
		ids := scoreInts(t, scoreRef, label+"_call"+strconv.Itoa(i)+"_ids")
		got := append(append([]int32{}, call.prompt...), call.target...)
		if at := equalIDs(got, ids); at >= 0 {
			t.Errorf("%s: call %d's ids differ at %d of %d", label, i, at, len(ids))
		}
		if len(call.prompt) != want[i].PromptLen || len(call.target) != want[i].TargetLen {
			t.Errorf("%s: call %d is %d+%d ids, upstream's %d+%d", label, i, len(call.prompt), len(call.target),
				want[i].PromptLen, want[i].TargetLen)
		}
	}
	return calls
}

// TestScoreTargets: the targets (yaml.dump of one field, the lyric block)
// and every forward's ids against upstream's, prompt and target.
func TestScoreTargets(t *testing.T) {
	m := loadScoreManifest(t, scoreRef)
	p, err := NewPlanner(nil, loadTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, label := range pmiLabels {
		c, codes := pmiConditions(t, m, label)
		calls := scoreCalls(t, p, m, label, c, codes)
		t.Logf("%s: %d calls, ids exact", label, len(calls))
	}
	if r := Reward(m.Cases["pmi_match"].Scores); math.Abs(r-m.Cases["pmi_match"].Reward) > 1e-12 {
		t.Errorf("reward %v from upstream's scores, upstream %v", r, m.Cases["pmi_match"].Reward)
	}
}

// TestScore is A14b's LM gate. Each teacher-forced forward's mean log p of
// its target against fp32's: on the fp16 bank (ACE_LM_BANK=fp16, the
// implementation) within 1e-2 a forward, every one under bf16's 5e-3-2.5e-1;
// on the served int8 bank priced as A10 priced it, pooled -- the mean over a
// case's forwards no further from fp32 than upstream bf16's (a forward alone
// can be: int8's worst is 6e-2 to bf16's 2e-2, a lyric target's). Then Score
// end to end, each condition and the reward within bf16's drift of fp32 (or
// 0.01, a recall's step being 1/len).
func TestScore(t *testing.T) {
	m := loadScoreManifest(t, scoreRef)
	mb := loadScoreManifest(t, scoreBF16Ref)
	g := loadGPU(t)
	p, err := NewPlanner(g, loadTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	fp16 := g.o.Bank == BankFP16
	for _, label := range pmiLabels {
		c, codes := pmiConditions(t, m, label)
		calls := scoreCalls(t, p, m, label, c, codes)
		var pooled, pooledBF float64
		cond, uncond, err := p.ScorePrompts(codes)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.prefill([][]int32{cond, uncond}); err != nil {
			t.Fatal(err)
		}
		for i, call := range calls {
			name := label + "_call" + strconv.Itoa(i)
			lp, rank, took, err := p.Teach(call.slot, call.prompt, call.target)
			if err != nil {
				t.Fatal(err)
			}
			want := scoreFloats(t, scoreRef, name+"_logp")
			bf := scoreFloats(t, scoreBF16Ref, name+"_logp")
			top := scoreInts(t, scoreRef, name+"_top")
			var worst, worstBF, sum, sumW, sumB float64
			ranks := 0
			for j := range want {
				worst = math.Max(worst, math.Abs(lp[j]-float64(want[j])))
				worstBF = math.Max(worstBF, math.Abs(float64(bf[j])-float64(want[j])))
				sum += lp[j]
				sumW += float64(want[j])
				sumB += float64(bf[j])
				wr := scoreTopK + 1
				for k := 0; k < scoreTopK; k++ {
					if top[j*scoreTopK+k] == call.target[j] {
						wr = k + 1
						break
					}
				}
				if min(rank[j], scoreTopK+1) != wr {
					ranks++
				}
			}
			n := float64(len(want))
			dm, dmB := math.Abs(sum-sumW)/n, math.Abs(sumB-sumW)/n
			t.Logf("%s slot %d: %d tokens in %v; log p worst %.2e (bf16 %.2e), mean %.2e (bf16 %.2e), %d ranks moved",
				name, call.slot, len(want), took, worst, worstBF, dm, dmB, ranks)
			if fp16 && dm > 1e-2 {
				t.Errorf("%s: mean log p off fp32 by %.2e", name, dm)
			}
			pooled, pooledBF = pooled+dm/float64(len(calls)), pooledBF+dmB/float64(len(calls))
		}
		t.Logf("%s: mean log p off fp32 by %.2e a forward (%s bank), bf16 by %.2e", label, pooled, g.o.Bank, pooledBF)
		if pooled > pooledBF {
			t.Errorf("%s: %s bank's log p %.2e off fp32 a forward, bf16's %.2e", label, g.o.Bank, pooled, pooledBF)
		}

		sc, err := p.Score(codes, c, ScoreScale)
		if err != nil {
			t.Fatal(err)
		}
		w, b := m.Cases[label], mb.Cases[label]
		t.Logf("%s: %v; reward %.4f (fp32 %.4f, bf16 %.4f)", label, sc.Stats, sc.Reward, w.Reward, b.Reward)
		check := func(k string, got, want, bf float64) {
			d, db := math.Abs(got-want), math.Abs(bf-want)
			t.Logf("  %-14s %.4f  fp32 %.4f  bf16 %.4f", k, got, want, bf)
			if d > math.Max(db, 0.01) {
				t.Errorf("%s %s: %.4f, fp32 %.4f (bf16 is %.4f off)", label, k, got, want, db)
			}
		}
		for k, v := range w.Scores {
			got, ok := sc.Conditions[k]
			if !ok {
				t.Errorf("%s: no %s score", label, k)
				continue
			}
			check(k, got, v, b.Scores[k])
		}
		check("reward", sc.Reward, w.Reward, b.Reward)
	}
}
