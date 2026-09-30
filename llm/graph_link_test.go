package llm

import (
	"testing"
)

// TestGraphHCLinkBitExact is P22b's gate: the graph with the sublayers'
// ports linked to the hyper-connection block (HCLink — the combine reading
// a sublayer's output where it lies, the decode up GEMV writing the next
// sublayer's A operand directly) produces the same bits as the graph that
// moves every activation across the boundary with `llm_move.comp`.
//
// Bit-exact and not to a tolerance, because nothing arithmetic changes: the
// combine reads the same floats from another buffer, and the GEMV narrows
// the same fp32 value the move narrowed. A prefill pass exercises the
// output side alone (the padded GEMM still writes `mixed`), and one-, two-
// and three-row passes exercise both sides wherever the plan's up rung is
// the GEMV — which is the K-quant bank (`LLM_DENSE_BANK` at the shipped
// plan); on the Q8 bank the rung has no GEMV and the test says so.
func TestGraphHCLinkBitExact(t *testing.T) {
	const layers, nTok = 4, 16
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < nTok {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	ids := all[:nTok]

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: nTok, Layers: layers})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	// The passes: a prefill of eight, then one, two, three and one rows.
	cuts := []int{8, 9, 11, 14, 15}
	arm := func(moves bool) ([][]float32, int) {
		g.hcMoves = moves
		// A captured decode step is the other arm's dispatches.
		g.dropPrerecorded()
		if err := g.Reset(); err != nil {
			t.Fatal(err)
		}
		moved := g.move.moves
		var out [][]float32
		from := 0
		for _, to := range cuts {
			l, _, err := g.Forward(ids[from:to])
			if err != nil {
				t.Fatalf("moves=%v rows [%d,%d): %v", moves, from, to, err)
			}
			out = append(out, append([]float32(nil), l...))
			from = to
		}
		return out, g.move.moves - moved
	}
	// Run 1 of a freshly staged graph diverges from runs 2+ at ~1.6e-5
	// (P1c finding 4), so the first arm is thrown away.
	arm(true)
	moved, nMoved := arm(true)
	linked, nLinked := arm(false)
	if err := g.hc.Resize(1); err != nil {
		t.Fatal(err)
	}
	gemv := g.hc.up == HCUpGemv
	t.Logf("%d layers, %d moves a graph with the moves, %d linked; the one-row up rung is %q (input side linked: %v)",
		layers, nMoved, nLinked, g.hc.up, gemv)
	if nLinked >= nMoved {
		t.Errorf("linking removed no moves: %d against %d", nLinked, nMoved)
	}
	from := 0
	for i, to := range cuts {
		r, err := compare(linked[i], moved[i])
		if err != nil {
			t.Fatal(err)
		}
		if r.maxAbs != 0 {
			t.Errorf("rows [%d,%d): the linked graph differs from the moved one: %v", from, to, r)
		}
		from = to
	}
}
