package lm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

const fsmRef = "../../reference/out/acefsm"

// replayFSM steps an FSM through tokens, handing it at each step scores
// whose global argmax is top and whose best language candidate is lang (the
// only two things upstream's processor reads from the logits), and checks
// the allowed set against want.
func replayFSM(t *testing.T, name string, f *FSM, vocab int, steps []fsmStep) {
	t.Helper()
	scores := make([]float32, vocab)
	for i, s := range steps {
		for j := range scores {
			scores[j] = 0
		}
		scores[s.Lang] = 1
		scores[s.Top] = 2
		if f.state.String() != s.State {
			t.Fatalf("%s step %d: state %s, want %s", name, i, f.state, s.State)
		}
		f.Apply(scores)
		n, sum := 0, 0
		var ids []int32
		for j, v := range scores {
			if !math.IsInf(float64(v), -1) {
				n++
				sum += j
				if len(ids) <= 64 {
					ids = append(ids, int32(j))
				}
			}
		}
		if n != s.N || (s.Sum != 0 && sum != s.Sum) {
			t.Fatalf("%s step %d (%s): %d allowed (sum %d), want %d (sum %d); ids %v want %v",
				name, i, s.State, n, sum, s.N, s.Sum, head(ids), head(s.IDs))
		}
		if s.IDs != nil && equalIDs(ids, s.IDs) >= 0 {
			t.Fatalf("%s step %d (%s): ids %v, want %v", name, i, s.State, ids, s.IDs)
		}
		if math.IsInf(float64(scores[s.Tok]), -1) {
			t.Fatalf("%s step %d: the oracle's token %d is masked", name, i, s.Tok)
		}
		f.Update(s.Tok)
	}
	if !f.Done() {
		t.Errorf("%s: not done after %d steps (state %s)", name, len(steps), f.state)
	}
}

func head(ids []int32) []int32 {
	if len(ids) > 12 {
		return ids[:12]
	}
	return ids
}

type fsmStep struct {
	State string  `json:"state"`
	N     int     `json:"n"`
	Sum   int     `json:"sum"`
	Top   int32   `json:"top"`
	Lang  int32   `json:"lang"`
	Tok   int32   `json:"tok"`
	IDs   []int32 `json:"ids"`
}

// TestFSM replays reference/dump_ace_fsm.py: upstream's processor driven
// through scripted CoTs (free, user metas injected, extremes, fields out of
// order, invalid values drifting into the free caption path).
func TestFSM(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(fsmRef, "fsm.json"))
	if err != nil {
		t.Skipf("no FSM reference (%v); run reference/dump_ace_fsm.py", err)
	}
	var ref struct {
		Vocab     int `json:"vocab"`
		Scenarios []struct {
			Name  string         `json:"name"`
			User  map[string]any `json:"user"`
			Steps []fsmStep      `json:"steps"`
		} `json:"scenarios"`
	}
	if err := json.Unmarshal(buf, &ref); err != nil {
		t.Fatal(err)
	}
	tk := loadTokenizer(t)
	v, err := newFSMVocab(tk)
	if err != nil {
		t.Fatal(err)
	}
	if v.vocab != ref.Vocab {
		t.Fatalf("vocab %d, want %d", v.vocab, ref.Vocab)
	}
	for _, sc := range ref.Scenarios {
		user := Meta{}
		for k, val := range sc.User {
			switch x := val.(type) {
			case float64:
				user[k] = formatInt(x)
			case string:
				user[k] = x
			}
		}
		replayFSM(t, sc.Name, newFSM(v, user), v.vocab, sc.Steps)
		t.Logf("%s: %d steps exact", sc.Name, len(sc.Steps))
	}
}

func formatInt(x float64) string { return itoa(int(x)) }

// TestFSMTrace replays the real LM run's phase 1 (reference/dump_ace_lm.py),
// taking the sampled token as the argmax at each step.
func TestFSMTrace(t *testing.T) {
	m := loadLMManifest(t)
	tk := loadTokenizer(t)
	v, err := newFSMVocab(tk)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cases["given_duration"]
	user := UserMeta(requestOf(c.Request))
	var steps []fsmStep
	for _, s := range c.Phases[0].Steps {
		steps = append(steps, fsmStep{State: s.FSM, N: s.FSMAllowed, Top: s.Token, Lang: s.Token, Tok: s.Token, IDs: s.FSMIDs})
	}
	replayFSM(t, "given_duration", newFSM(v, user), v.vocab, steps)
	t.Logf("given_duration phase 1: %d steps exact", len(steps))
}

func itoa(i int) string { return strconv.Itoa(i) }

// TestTopP applies the FSM and top-p to the oracle's own logits at every step
// it dumped them, and wants exactly the survivors upstream recorded: phase 1
// under the FSM, phase 2 after CFG over the codes.
func TestTopP(t *testing.T) {
	m := loadLMManifest(t)
	tk := loadTokenizer(t)
	v, err := newFSMVocab(tk)
	if err != nil {
		t.Fatal(err)
	}
	vocab := v.vocab
	for _, label := range []string{"given_duration", "all_metas"} {
		c := m.Cases[label]
		for ph, p := range c.Phases {
			var f *FSM
			if p.Rows == 1 {
				f = newFSM(v, UserMeta(requestOf(c.Request)))
			}
			checked := 0
			for i, s := range p.Steps {
				name := fmt.Sprintf("%s_p%d_logits%d", label, ph, i)
				_, has := m.Tensors[name]
				if f != nil && has {
					_, logits := refFloat32(t, m, name)
					f.Apply(logits)
					edge := topPCum(logits, DefaultSampling.TopP)
					checkSurvivors(t, name, logits, 0, s.Allowed, s.AllowedIDs, edge)
					checked++
				} else if f != nil {
					scores := make([]float32, vocab)
					scores[s.Token] = 1
					f.Apply(scores)
				}
				if f != nil {
					f.Update(s.Token)
				}
				if f == nil && has {
					_, logits := refFloat32(t, m, name)
					cs := codeScores(logits[ImEndID:CodeBase+NumCodes], logits[vocab+ImEndID:vocab+CodeBase+NumCodes],
						DefaultSampling.CFG, i, len(p.Steps)-1)
					edge := topPCum(cs, DefaultSampling.TopP)
					checkSurvivors(t, name, cs, CodeBase, s.Allowed, s.AllowedIDs, edge)
					checked++
				}
			}
			t.Logf("%s phase %d: %d steps' survivors exact", label, ph, checked)
		}
	}
}

// checkSurvivors compares the finite scores (index + base, with a code
// phase's EOS at NumCodes) with the oracle's count and ids. One token more
// than torch is allowed when the last one kept sits within 2e-5 of p (its
// fp32 reduction; see topP).
func checkSurvivors(t *testing.T, name string, scores []float32, base int32, n int, ids []int32, edge float64) {
	t.Helper()
	var got []int32
	for i, s := range scores {
		if !math.IsInf(float64(s), -1) {
			id := int32(i) + base
			if base != 0 && i == NumCodes {
				id = ImEndID
			}
			got = append(got, id)
		}
	}
	if len(got) == n+1 && math.Abs(edge-DefaultSampling.TopP) < 2e-5 {
		t.Logf("%s: %d survivors against torch's %d, the last at cumulative %.7f (fp32 boundary)", name, len(got), n, edge)
		return
	}
	if len(got) != n || (ids != nil && equalIDs(got, ids) >= 0) {
		t.Errorf("%s: %d survivors %v, want %d %v", name, len(got), head(got), n, head(ids))
	}
}
