package llm

import (
	"fmt"
	"sort"
	"testing"
	"time"
)

// L4b: the QSA selection on the device, at the only length where it exists.
//
// L4a settled what the selection *is* on the CPU — a radix select, reproduced
// pass for pass and set for set on the 2045 rows of the 4k dump where it bites
// — and left the kernel. This file is the kernel: `llm_attn_select.comp`
// writing a per-cell bitmask, and `llm_attn_wmma.comp` reading it beside the
// causal mask.
//
// Everything here needs the 4096-token trace and a layer staged for a
// 4096-cell cache, which is ~0.7 GB of device memory and a minute of setup, so
// it is a separate fixture from the 7-token one the rest of the GPU tests use.
// At 256 cells the selection asks for 2051 of them and is the identity;
// `AttnGPU.sparse` is false there and the dense arm runs, which is why none of
// L2f's numbers move.

// attnGPU4k stages layer 3 for the 4k fixture and uploads llama.cpp's own
// input for it. It returns the trace beside the device so that the reference's
// selection and its attention output are both to hand.
func attnGPU4k(t *testing.T) (*AttnGPU, *Trace, AttnConfig, int, func()) {
	t.Helper()
	c, w, nTok, nKV, _, tr, _ := qsaFixtures4k(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	start := time.Now()
	g, err := NewAttnGPU(dev, c, nTok, nKV, []AttnWeights{w}, denseQ8Test)
	if err != nil {
		done()
		t.Fatal(err)
	}
	if err := g.Upload(in.Vals, nTok); err != nil {
		g.Destroy()
		done()
		t.Fatal(err)
	}
	t.Logf("staged in %v: %.1f MB of weights, %.1f MB of arenas, %d tokens, %d cells, %d blocks, width %d, sparse %v",
		time.Since(start).Round(time.Millisecond),
		float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6,
		nTok, g.NKV(), g.NBlocks(), g.selWidth(), g.Sparse())
	return g, tr, c, nTok, func() { g.Destroy(); done() }
}

// TestAttnGPUSelectionEngagesAt4k is the precondition, asserted rather than
// assumed: the layer decides for itself whether to run the selection, from the
// cache it was built for, and at 4096 cells against a width of 2051 it has to
// say yes. At 256 cells — every other GPU test in this package — it has to say
// no, because there the selection names every cell and running it would be the
// identity.
func TestAttnGPUSelectionEngagesAt4k(t *testing.T) {
	g, _, _, _, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Fatalf("width %d of %d cells and the layer still runs dense", g.selWidth(), g.NKV())
	}
	d, kinds, err := g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d dispatches: %v", len(d), kinds)
	if len(d) != 7 || kinds[4] != "select" {
		t.Errorf("the graph is %v; the selection should be the fifth of seven dispatches", kinds)
	}
}

// TestAttnGPUSelectionIsTheCPUs is the kernel's own gate, and it is exact.
//
// The device and `topK` in attn.go are the same algorithm over the same
// numbers — the kernel reads the cell scores the device itself wrote, and the
// comparison feeds those same scores to the CPU — so there is no tolerance
// here at all. Every bit of the bitmask has to be the bit the CPU selection
// would set, on all 4096 rows.
//
// That is a stronger statement than "it agrees with llama.cpp", and it is the
// one worth making about a kernel: the tie fill is the only place the two
// could differ, because the reference's own is `atomicAdd` order and ours is
// ascending cell index (L4a-2 measured that the reference's order is not even
// reproducible against itself). A bitmask has no order, so ours is a function
// of the scores alone, and this test says so.
func TestAttnGPUSelectionIsTheCPUs(t *testing.T) {
	g, _, _, nTok, done := attnGPU4k(t)
	defer done()

	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	cells := g.Cells()
	mask := g.Selection()
	nKV, width := g.NKV(), g.selWidth()

	var mismatched int
	var firstErr string
	parallel(nTok, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			// topK emits the strictly-above cells first and appends the tie
			// fill behind them, so its list is ascending within each group
			// and not overall; a bitmask has only the one order. Sorting is
			// the comparison, not a weakening of it — the *set* is what both
			// sides are claiming, and L4a-2 measured that the reference has
			// no reproducible order to compare against anyway.
			want, _ := topK(cells[i*nKV:(i+1)*nKV], width)
			sort.Slice(want, func(a, b int) bool { return want[a] < want[b] })
			got := g.SelectedCells(mask, i)
			bad := len(got) != len(want)
			if !bad {
				for j := range got {
					if got[j] != want[j] {
						bad = true
						break
					}
				}
			}
			if bad {
				mismatched++
				if firstErr == "" {
					firstErr = fmt.Sprintf("token %d: the kernel selects %d cells, topK selects %d", i, len(got), len(want))
				}
			}
		}
	})
	if mismatched != 0 {
		t.Errorf("%d of %d rows differ from the CPU selection over the device's own scores — %s",
			mismatched, nTok, firstErr)
	}
	t.Logf("all %d rows: the kernel's bitmask is topK's selection, cell for cell", nTok)

	// And the width itself: past the biting point every row holds exactly
	// `width` cells, which is what says the tie fill ran and did not overfill.
	for _, i := range []int{0, width - 1, width, width + 1, nTok / 2, nTok - 1} {
		n := len(g.SelectedCells(mask, i))
		if n != width {
			t.Errorf("token %d has %d cells selected, want %d", i, n, width)
		}
	}
}

// TestAttnGPUSelectionAgainstTheReference is the other half: ours is the
// reference's selection, from *our* scores rather than from llama.cpp's.
//
// A top-k is discontinuous, so this cannot be zero and L4a-4 already priced
// it on the CPU — 138 of 6 298 621 visible cells, 0.0022%, over 30 of 4096
// tokens. The device computes the scores in its own kernels rather than in
// Go, so the number here is not the same number; what has to hold is that it
// stays the same *size*, because a kernel that had the selection structurally
// wrong would move whole blocks rather than single cells.
func TestAttnGPUSelectionAgainstTheReference(t *testing.T) {
	g, _, _, nTok, done := attnGPU4k(t)
	defer done()
	_, _, _, _, width, _, topk := qsaFixtures4k(t)

	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	mask := g.Selection()

	var differing, visible, rowsWithDiff, worstN, worstRow int
	worstRow = -1
	for i := 0; i < nTok; i++ {
		ours := map[int32]bool{}
		for _, cell := range g.SelectedCells(mask, i) {
			if int(cell) <= i {
				ours[cell] = true
			}
		}
		ref := map[int32]bool{}
		for _, cell := range topk.Ints[i*width : (i+1)*width] {
			if int(cell) <= i {
				ref[cell] = true
			}
		}
		visible += len(ref)
		n := 0
		for cell := range ref {
			if !ours[cell] {
				n++
			}
		}
		for cell := range ours {
			if !ref[cell] {
				n++
			}
		}
		differing += n
		if n > 0 {
			rowsWithDiff++
		}
		if n > worstN {
			worstN, worstRow = n, i
		}
	}
	t.Logf("the kernel's selection against llama.cpp's, visible cells only: %d of %d differ (%.4f%%), over %d of %d tokens; worst row %d with %d",
		differing, visible, 100*float64(differing)/float64(visible), rowsWithDiff, nTok, worstRow, worstN)
	if frac := float64(differing) / float64(visible); frac > 0.01 {
		t.Errorf("%.2f%% of the selected cells differ, which is more than the tie fill and our score error can explain", 100*frac)
	}
}

// TestAttnGPULayer4k is L4b's gate: the whole layer on the device at a context
// where the selection actually removes cells the causal mask allows, against
// llama.cpp's own `attn_output-3`.
//
// The tolerance is L4a-5's and not L2f's. At 4096 columns every quantised
// matmul in the reference accumulates in **fp16**, so the reference sits
// ~5.4e-03 rms from the f32 model on a tensor whose own rms is 1.21 and no
// model of the operands closes it: the 2e-3 this same comparison uses at 7
// tokens is a fact about a 7-token dump, not about the kernel.
func TestAttnGPULayer4k(t *testing.T) {
	g, tr, _, _, done := attnGPU4k(t)
	defer done()

	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		{"attn_gated-3", g.Context(), 2e-2},
		{"attn_output-3", g.Out(), 2e-2},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-14s at 4096 tokens vs llama.cpp  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against llama.cpp, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestAttnGPUSelectionChangesTheOutput is the negative control, and the whole
// point of running the fixture at 4 k.
//
// A kernel that read the bitmask and ignored it, or read it from the wrong
// row, or selected every cell, would pass every tolerance above by being a
// *dense* causal attention — which at 7 tokens is the right answer and at 4096
// is a different model. So this runs the same layer with the selection turned
// off and demands the two disagree, and demands the sparse one be the nearer
// of the two to llama.cpp. Only the second half is the real claim: the first
// merely says the bit is read.
func TestAttnGPUSelectionChangesTheOutput(t *testing.T) {
	g, tr, _, _, done := attnGPU4k(t)
	defer done()

	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	sparse := g.Out()

	g.SetSparse(false)
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	dense := g.Out()
	g.SetSparse(true)

	between, err := compare(sparse, dense)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("sparse against dense: %v", between)
	if between.rms < 1e-3 {
		t.Errorf("the selection moves the output by rms %.3e — the bitmask is not being read", between.rms)
	}

	want, err := tr.Get("attn_output-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	rs, err := compare(sparse, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	rd, err := compare(dense, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("against llama.cpp: sparse rms %.3e, dense rms %.3e, %.1fx", rs.rms, rd.rms, rd.rms/rs.rms)
	if rs.rms >= rd.rms {
		t.Errorf("running the selection is no nearer llama.cpp (%.3e) than ignoring it (%.3e) — the reference is sparse here, so this kernel is not reproducing it",
			rs.rms, rd.rms)
	}
}

// TestAttnGPUIndexer4k prices the one thing the device does not reproduce
// about the reference, and explains why the selection disagrees 17x more here
// than it did on the CPU.
//
// L4a-6 found that the indexer's two BF16 weights meet a **bf16** activation
// above the 8-column threshold — `y_non_contig` converts the f32 activation
// and a bf16 x bf16 kernel runs, eight mantissa bits — and modelling that was
// worth 2061x on `indexer_k_raw`. The device cannot model it: its indexer
// projections are two column ranges of the layer's one fused fp16 weight
// (L2f-3), which is the whole point of fusing them, and fp16 is 1.02x *worse*
// than f32 against a bf16 reference. So the device's score is further from
// llama.cpp's than the CPU's modelled one is, and the selection — being
// discontinuous — is where that shows.
//
// This test measures both halves rather than asserting a bound on either: how
// far the device's indexer chain sits from the reference, and what it costs
// downstream, which TestAttnGPUSelectionAgainstTheReference has already
// bounded at 0.0381% of visible cells and TestAttnGPULayer4k at 9.9e-04 rms
// on the layer's output. The trade is one weight instead of six.
func TestAttnGPUIndexer4k(t *testing.T) {
	g, tr, c, _, done := attnGPU4k(t)
	defer done()

	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		{"indexer_k_raw-3", g.Column(g.ColIK(), c.IdxDim), 5e-3},
		{"indexer_k-3", g.IdxK(), 5e-2},
		{"indexer_q-3", g.IdxQ(), 5e-2},
		{"indexer_score-3", g.Score(), 5e-1},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-18s at 4096 tokens vs llama.cpp  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestAttnGPUCacheSizeDoesNotChangeTheAnswer is the gate on the one regime
// every other test in this file misses, and the one the served model is
// always in: a cache **larger** than the context it holds.
//
// The 4k fixture stages 4096 tokens into a 4096-cell cache, so `nKV` and the
// live cell count are the same number and nothing distinguishes "the cache is
// this big" from "the context is this deep". The server stages 131 072 cells
// and answers a turn at a few thousand, and three kernels used to walk the
// cache rather than the context there — llm_attn_score.comp scoring all
// `nKV/ratio` pooled blocks, llm_attn_select.comp running four radix passes
// and an emit over all `nKV` cells, llm_attn_wmma.comp reading every key block
// because the selection was only a mask. Cutting all three against the live
// cells is only sound if it changes nothing, and *nothing* here means bit for
// bit: a dead cell is an -inf the causal mask wrote, an -inf is the smallest
// key `f2ui` has, and a dead pooled block holds cell 0 pooled `ratio` times
// however many of them there are.
//
// So this runs the fixture twice, once in its own cache and once in a cache
// three times too big, and demands the same tensors out. It is the negative
// control on the depth work the way TestAttnGPUSelectionChangesTheOutput is
// the negative control on the selection: a kernel that read a stale score, or
// emitted a bitmask word it had not written, or skipped a key block that had
// a cell in it, would show up here and nowhere else.
func TestAttnGPUCacheSizeDoesNotChangeTheAnswer(t *testing.T) {
	c, w, nTok, nKV, _, tr, _ := qsaFixtures4k(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	run := func(cells int) (out, ctx []float32, sel [][]int32) {
		t.Helper()
		dev, done := newTestDevice(t)
		defer done()
		g, err := NewAttnGPU(dev, c, nTok, cells, []AttnWeights{w}, denseQ8Test)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		if !g.Sparse() {
			t.Fatalf("%d cells: the layer runs dense, so this proves nothing", cells)
		}
		if err := g.Upload(in.Vals, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		mask := g.Selection()
		sel = make([][]int32, nTok)
		for i := range sel {
			sel[i] = g.SelectedCells(mask, i)
		}
		return append([]float32(nil), g.Out()...), append([]float32(nil), g.Context()...), sel
	}

	wantOut, wantCtx, wantSel := run(nKV)
	gotOut, gotCtx, gotSel := run(3 * nKV)

	// The selection first, because it is the cause: a difference in the
	// bitmask explains a difference in the output and not the other way
	// round, and it names the token it happened on.
	var rows int
	for i := range wantSel {
		if len(gotSel[i]) != len(wantSel[i]) {
			t.Fatalf("token %d selects %d cells in a %d-cell cache and %d in a %d-cell one",
				i, len(wantSel[i]), nKV, len(gotSel[i]), 3*nKV)
		}
		for j := range wantSel[i] {
			if gotSel[i][j] != wantSel[i][j] {
				rows++
				break
			}
		}
	}
	if rows != 0 {
		t.Errorf("%d of %d rows select different cells in a cache three times too big", rows, nTok)
	}

	for _, tc := range []struct {
		name      string
		want, got []float32
	}{
		{"attn_gated-3", wantCtx, gotCtx},
		{"attn_output-3", wantOut, gotOut},
	} {
		if len(tc.got) != len(tc.want) {
			t.Fatalf("%s is %d values in one cache and %d in the other", tc.name, len(tc.want), len(tc.got))
		}
		for i := range tc.want {
			if tc.got[i] != tc.want[i] {
				t.Fatalf("%s value %d: %g in a %d-cell cache, %g in a %d-cell one — "+
					"the answer depends on how much room the cache has",
					tc.name, i, tc.want[i], nKV, tc.got[i], 3*nKV)
			}
		}
		t.Logf("%-14s identical to the last bit across a %dx cache", tc.name, 3)
	}
}
