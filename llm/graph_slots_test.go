package llm

// CONCURRENCY.md C1: one graph holding two conversations.
//
//	go test ./llm/ -v -run TestGraphSlotsAreSequences   # four layers, ~9 GB

import (
	"fmt"
	"testing"
)

// TestGraphSlotsAreSequences is C1's gate: two sequences interleaved a token at
// a time through one graph's two slots produce **bit for bit** the logits each
// one produces run alone. The prompts differ in length and content, the decode
// goes through the recorded step (P1c) in both slots, and each prefill lands
// between the other sequence's steps, so any state the slots share — a cache
// region, a recurrent state, a ring, the position, the recorded buffer's
// offsets — shows up as a difference.
//
// The controls are the reason to believe it. Each re-runs the interleave with
// one block's slot switch sabotaged, so that block keeps using slot 0 while the
// graph thinks it is on slot 1; every one of them must move sequence A's
// logits. Without them a gate that passes might only be saying that the
// interleave never reached the state it claims to separate.
func TestGraphSlotsAreSequences(t *testing.T) {
	const steps = 24
	g, _, ids := graphFixture(t, GraphOpts{Layers: 4, Slots: 2})
	if g.Slots() != 2 {
		t.Fatalf("staged %d slots", g.Slots())
	}
	a := ids
	b := append([]int32(nil), ids[len(ids)/3:]...)
	b[0], b[1] = b[1], b[0]

	argmax := func(l []float32) int32 {
		id := int32(0)
		for j := range l {
			if l[j] > l[id] {
				id = int32(j)
			}
		}
		return id
	}
	clone := func(l []float32) []float32 { return append([]float32(nil), l...) }

	// solo runs one sequence alone on a slot, greedily, and returns the
	// tokens it chose and every row of logits: the prefill's and each step's.
	solo := func(slot int, prompt []int32) ([]int32, [][]float32) {
		if err := g.UseSlot(slot); err != nil {
			t.Fatal(err)
		}
		logits, _, err := g.Forward(prompt)
		if err != nil {
			t.Fatal(err)
		}
		rows := [][]float32{clone(logits)}
		var toks []int32
		for i := 0; i < steps; i++ {
			id := argmax(logits)
			toks = append(toks, id)
			if logits, _, err = g.Extend([]int32{id}); err != nil {
				t.Fatal(err)
			}
			rows = append(rows, clone(logits))
		}
		return toks, rows
	}

	// A first run over freshly allocated arenas is not bit-repeatable against
	// later ones (see TestPrerecordedDecodeIsTheRecordedDecode), and slot 1's
	// region is fresh until something runs in it. So each slot is warmed up.
	solo(0, a)
	solo(1, b)
	aToks, aWant := solo(0, a)
	bToks, bWant := solo(1, b)
	if !g.Prerecorded() {
		t.Fatal("the solo runs never captured a decode step, so the gate would not cover the replay")
	}

	// interleave runs both, teacher-forced on the tokens their solo runs
	// chose, alternating slots after every pass. sabotage, when set, runs
	// after each switch to slot 1 and is how a control breaks one block.
	interleave := func(sabotage func()) (aGot, bGot [][]float32) {
		use := func(s int) {
			if err := g.UseSlot(s); err != nil {
				t.Fatal(err)
			}
			if s == 1 && sabotage != nil {
				sabotage()
			}
		}
		run := func(s int, f func() ([]float32, []float32, error)) []float32 {
			use(s)
			l, _, err := f()
			if err != nil {
				t.Fatal(err)
			}
			return clone(l)
		}
		aGot = append(aGot, run(0, func() ([]float32, []float32, error) { return g.Forward(a) }))
		bGot = append(bGot, run(1, func() ([]float32, []float32, error) { return g.Forward(b) }))
		for i := 0; i < steps; i++ {
			aGot = append(aGot, run(0, func() ([]float32, []float32, error) { return g.Extend(aToks[i : i+1]) }))
			bGot = append(bGot, run(1, func() ([]float32, []float32, error) { return g.Extend(bToks[i : i+1]) }))
		}
		// A sabotaged run leaves a block pointing at the wrong slot; put
		// every block back where the graph thinks it is.
		if err := g.UseSlot(0); err != nil {
			t.Fatal(err)
		}
		for _, f := range []func(int) error{g.attn.UseSlot, g.dn.UseSlot, g.ple.UseSlot} {
			if err := f(0); err != nil {
				t.Fatal(err)
			}
		}
		return aGot, bGot
	}

	// firstDiff is the first (row, logit) where two runs part, or "".
	firstDiff := func(got, want [][]float32) string {
		for r := range want {
			for j := range want[r] {
				if got[r][j] != want[r][j] {
					return fmt.Sprintf("row %d logit %d: %g against %g", r, j, got[r][j], want[r][j])
				}
			}
		}
		return ""
	}

	aGot, bGot := interleave(nil)
	if d := firstDiff(aGot, aWant); d != "" {
		t.Fatalf("sequence A interleaved is not sequence A alone: %s", d)
	}
	if d := firstDiff(bGot, bWant); d != "" {
		t.Fatalf("sequence B interleaved is not sequence B alone: %s", d)
	}
	t.Logf("two sequences (%d and %d prompt tokens, %d steps each) interleaved through two slots: "+
		"%d + %d logit rows bit-identical to the solo runs", len(a), len(b), steps, len(aGot), len(bGot))

	for _, c := range []struct {
		name string
		f    func()
	}{
		{"attention cache", func() { _ = g.attn.UseSlot(0) }},
		{"deltanet state", func() { _ = g.dn.UseSlot(0) }},
		{"ple ring", func() { _ = g.ple.UseSlot(0) }},
	} {
		aGot, _ := interleave(c.f)
		d := firstDiff(aGot, aWant)
		if d == "" {
			t.Errorf("control: sharing the %s between the slots left sequence A unchanged, "+
				"so the gate cannot see that block's separation", c.name)
			continue
		}
		t.Logf("control, %s shared: sequence A moves at %s", c.name, d)
	}
}
