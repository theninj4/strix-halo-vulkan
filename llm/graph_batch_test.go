package llm

// CONCURRENCY.md C5: several sequences, one token each, one pass.
//
//	go test ./llm/ -v -run TestGraphDecodeRowsIsEachSlot   # four layers, ~9 GB

import (
	"fmt"
	"math"
	"testing"
)

// TestGraphDecodeRowsIsEachSlot is C5's gate. Three sequences of different
// prompts and lengths (700, 300 and 500 tokens) decode together, a batched pass per step, and each
// row's logits are compared with the same sequence stepped alone in its slot.
// The same is done with two of the three, on slots 0 and 2.
//
// **It is bit equality.** An R-row decode GEMV multiplies one weight read
// into R accumulators and sums each row in the order the one-row kernel
// does, and every per-sequence kernel runs a row as a one-token pass, so
// nothing is reassociated. If a future kernel makes that untrue, it is the
// decode-schedule tolerance in TestGraphIsAChunkSplit (1e-3 rms) that this
// should relax to, knowingly.
//
// Each control overwrites one block's per-row position table after
// DecodeRows has written it: every row reads row 0's position. Each must move
// the logits, or the gate cannot see that block's per-row position.
func TestGraphDecodeRowsIsEachSlot(t *testing.T) {
	const steps = 8
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 2000 {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: 1024, NKV: 2048, Layers: 4, Slots: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	prompts := [][]int32{all[:700], all[800:1100], all[1200:1700]}

	argmax := func(l []float32) int32 {
		id := int32(0)
		for j := range l {
			if l[j] > l[id] {
				id = int32(j)
			}
		}
		return id
	}
	use := func(s int) {
		if err := g.UseSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	prefill := func(s int) []float32 {
		use(s)
		l, _, err := g.Forward(prompts[s])
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...)
	}

	// Solo: each slot prefilled and stepped greedily on its own. Twice, the
	// first time only to warm the arenas (TestPrerecordedDecodeIsTheRecordedDecode).
	toks := make([][]int32, 3)
	want := make([][][]float32, 3)
	for pass := 0; pass < 2; pass++ {
		for s := range 3 {
			l := prefill(s)
			toks[s], want[s] = nil, nil
			for range steps {
				id := argmax(l)
				toks[s] = append(toks[s], id)
				var err error
				if l, _, err = g.Extend([]int32{id}); err != nil {
					t.Fatal(err)
				}
				want[s] = append(want[s], append([]float32(nil), l...))
			}
		}
	}
	vocab := len(want[0][0])

	// batched runs the named slots together, teacher-forced on the solo
	// tokens, and returns the worst difference from the solo logits.
	batched := func(slots []int) (maxAbs, rms float64, identical bool) {
		for _, s := range slots {
			prefill(s)
		}
		use(0)
		identical = true
		var sum float64
		var n int
		for i := range steps {
			in := make([]int32, len(slots))
			for r, s := range slots {
				in[r] = toks[s][i]
			}
			l, err := g.DecodeRows(slots, in)
			if err != nil {
				t.Fatal(err)
			}
			for r, s := range slots {
				row := l[r*vocab : (r+1)*vocab]
				for j, v := range row {
					d := math.Abs(float64(v - want[s][i][j]))
					if d != 0 {
						identical = false
					}
					maxAbs = math.Max(maxAbs, d)
					sum += d * d
					n++
				}
			}
		}
		for _, s := range slots {
			use(s)
			if g.Past() != len(prompts[s])+steps {
				t.Fatalf("slot %d is at %d after the batch, want %d", s, g.Past(), len(prompts[s])+steps)
			}
		}
		use(0)
		return maxAbs, math.Sqrt(sum / float64(n)), identical
	}

	for _, slots := range [][]int{{0, 1, 2}, {0, 2}} {
		maxAbs, rms, same := batched(slots)
		name := fmt.Sprintf("slots %v", slots)
		if !same {
			t.Errorf("%s: batched rows differ from the solo steps by %.3e rms (max %.3e)", name, rms, maxAbs)
			continue
		}
		t.Logf("%s, %d steps: every row bit-identical to its solo step", name, steps)
	}

	// The controls: one block's rows all read row 0's position.
	for _, c := range []struct {
		name string
		f    func(g *Graph)
	}{
		{"attention", func(g *Graph) { g.attn.abuf.WriteUint32At(2, []uint32{uint32(g.attn.batch[0].past)}) }},
		{"deltanet", func(g *Graph) { g.dn.abuf.WriteUint32At(2, []uint32{uint32(g.dn.batch[0].past)}) }},
		{"ple", func(g *Graph) { g.ple.abuf.WriteUint32At(2, []uint32{uint32(g.ple.batch[0].past)}) }},
	} {
		batchSabotage = c.f
		_, rms, _ := batched([]int{0, 1, 2})
		batchSabotage = nil
		if rms <= 1e-3 {
			t.Errorf("control: row 1 reading row 0's position in the %s block left the logits within tolerance "+
				"(%.3e rms), so the gate cannot see that block's per-row position", c.name, rms)
			continue
		}
		t.Logf("control, %s row 1 at row 0's position: %.3e rms", c.name, rms)
	}
}
