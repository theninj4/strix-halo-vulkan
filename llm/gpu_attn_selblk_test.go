package llm

import (
	"testing"
)

// P14-1: the selection over block scores, and the gate that says it is the same
// selection.
//
// `llm_attn_select.comp` radix-selects over `llm_attn_expand.comp`'s per-cell
// tensor; `llm_attn_selblk.comp` selects over the block scores that tensor was
// expanded *from*, with a per-block weight. The claim is an equality and not a
// tolerance — the same threshold, the same ties, the same bitmask — because all
// `ratio` cells of a pooled block carry one score and every cell the causal
// mask drops carries -inf, so the two histograms count the same multiset.
//
// The chain this closes: `TestAttnGPUSelectionIsTheCPUs` holds the cell select
// to the CPU `topK` over the device's own expanded scores, and this holds the
// block select to the cell select. Nothing in the shipped path computes the
// expanded tensor any more, which is 1.14 GB of arena at 128 000 cells and a
// 2048-row batch — and what a prefill at depth wants that arena for is a wider
// ubatch.
//
// **The chunk schedules are the point.** At 4096 tokens in a 4096-cell cache
// the whole cache is whole blocks, so `nBid >= nBlocks` and the *incomplete
// tail* — the spare block that collects every cell past the last whole one, at
// llama.cpp's 1e9 "always visible" bias — never exists. A run in chunks of
// seven reaches a cell count that is not a multiple of `ratio` at every step,
// which is where the weight of the tail block and the causal boundary inside a
// partly visible block are both live. Both are the places a weighted histogram
// could be wrong while a dense one was right.
func TestAttnGPUSelectBlocksIsTheCellSelect(t *testing.T) {
	schedules := [][]int{
		{4096},
		{4095, 1},
		{1000, 1000, 1000, 1096},
		append(even(4088, 7), 8),
	}

	// Two constructions, because which selection runs is an arena decision
	// taken at construction and not a rung: the cell select reads a tensor the
	// block select does not allocate.
	run := func(expand bool) ([][]uint32, [][]float32, int, bool) {
		var sels [][]uint32
		var outs [][]float32
		var width int
		var sparse bool
		func() {
			if expand {
				t.Setenv("LLM_ATTN_EXPAND_CELLS", "1")
			} else {
				t.Setenv("LLM_ATTN_EXPAND_CELLS", "")
			}
			g, tr, c, nTok, done := attnGPU4k(t)
			defer done()
			if g.ExpandsCells() != expand {
				t.Fatalf("asked for expandCells %v and got %v", expand, g.ExpandsCells())
			}
			src, err := tr.Get("hc_mixed-3", 0)
			if err != nil {
				t.Fatal(err)
			}
			width, sparse = g.selWidth(), g.Sparse()
			// P8's split is a different kernel with its own rounding and is
			// not the subject; the selection is what is being compared.
			g.SetSplits(1)
			defer g.SetSplits(0)
			for _, chunks := range schedules {
				g.Reset()
				var sel []uint32
				past := 0
				for _, n := range chunks {
					if err := g.SetPast(past); err != nil {
						t.Fatal(err)
					}
					if err := g.Upload(src.Vals[past*c.NEmbd:(past+n)*c.NEmbd], n); err != nil {
						t.Fatal(err)
					}
					if err := g.Run(0); err != nil {
						t.Fatalf("chunk of %d at cell %d: %v", n, past, err)
					}
					// Only the rows the chunk actually ran: the bitmask is
					// allocated for every arena row and the pad rows hold
					// whatever the widest previous chunk left.
					sel = append(sel, append([]uint32(nil), g.Selection()[:n*g.selWords()]...)...)
					past += n
				}
				sels = append(sels, sel)
				outs = append(outs, append([]float32(nil), g.Out()...))
			}
			_ = nTok
		}()
		return sels, outs, width, sparse
	}

	wantSel, wantOut, width, sparse := run(true)
	if !sparse {
		t.Skip("the selection does not bite at this cache size")
	}
	gotSel, gotOut, _, _ := run(false)

	bits := 0
	for s := range schedules {
		if len(gotSel[s]) != len(wantSel[s]) {
			t.Fatalf("schedule %v: %d selection words against %d",
				schedules[s], len(gotSel[s]), len(wantSel[s]))
		}
		for i := range wantSel[s] {
			if gotSel[s][i] != wantSel[s][i] {
				t.Fatalf("schedule %v, word %d: %#08x from the block select, "+
					"%#08x from the cell select",
					schedules[s], i, gotSel[s][i], wantSel[s][i])
			}
			bits += bitsSet(wantSel[s][i])
		}
		// And the layer's own output, which is the bitmask's only consumer: the
		// same mask over the same keys is the same arithmetic, so this is an
		// equality too.
		if len(gotOut[s]) != len(wantOut[s]) {
			t.Fatalf("schedule %v: %d outputs against %d", schedules[s], len(gotOut[s]), len(wantOut[s]))
		}
		for i := range wantOut[s] {
			if gotOut[s][i] != wantOut[s][i] {
				t.Fatalf("schedule %v, output %d: %v from the block select, %v from the cell select",
					schedules[s], i, gotOut[s][i], wantOut[s][i])
			}
		}
	}
	t.Logf("%d selected cells over %d chunk schedules, and the layer's output, "+
		"identical to the last bit: the block select at width %d is the cell select",
		bits, len(schedules), width)
}

// TestAttnGPUSelectBlocksSkipsTheExpansion is the other half of P14-1 and it is
// about the *cost*, which is the reason the equality above was worth having: a
// shipped pass must not dispatch `expand` and must not allocate the tensor it
// wrote.
//
// The arena is the number that matters. `rows * nKV` floats is 1.14 GB at 2048
// rows over 139 000 cells and 4.6 GB at the 8192 rows a prefill at depth wants,
// so this is not a tidy-up — it is the arena a wider ubatch at depth is taken
// out of.
func TestAttnGPUSelectBlocksSkipsTheExpansion(t *testing.T) {
	g, _, _, nTok, done := attnGPU4k(t)
	defer done()
	if g.ExpandsCells() {
		t.Fatal("the shipped path expands to cells")
	}
	if g.Cells() != nil {
		t.Error("Cells() returns a tensor nothing computed")
	}
	if err := g.Upload(make([]float32, nTok*g.cfg.NEmbd), nTok); err != nil {
		t.Fatal(err)
	}
	_, kinds, err := g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	var sel, expand int
	for _, k := range kinds {
		switch k {
		case "select":
			sel++
		case "expand":
			expand++
		}
	}
	if expand != 0 {
		t.Errorf("%d expand dispatches in a shipped pass", expand)
	}
	if sel != 1 {
		t.Errorf("%d select dispatches, want 1", sel)
	}

	// And the arena, against what the expansion would have cost at this shape.
	saved := float64(g.arenaRows) * float64(g.NKV()) * 4 / 1e6
	t.Logf("%d dispatches, no expansion: %.1f MB of arena at %d rows over %d cells "+
		"that a shipped pass no longer allocates (%.2f GB at 2048 rows over 139k)",
		len(kinds), saved, g.arenaRows, g.NKV(), 2048*139000*4/1e9)
}
