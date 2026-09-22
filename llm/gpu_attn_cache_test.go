package llm

import (
	"fmt"
	"math"
	"testing"
)

// L7a: the KV cache, and the one claim that makes decode possible at all.
//
// Everything above this file runs a full-attention layer as a *fresh
// sequence*: token t is cell t, its position is t, and the key and value
// planes are the batch's own padded token rows. Decode is the opposite — one
// token at a time, at cell `past`, attending over every cell before it — and
// the only way that is the same model is if splitting a prompt into chunks
// changes nothing.
//
// So that is the gate, and it is an equality rather than a tolerance. A
// chunked run recomputes nothing: the key and value cells a previous chunk
// wrote are read back as they were written, the indexer's pooled blocks are
// carried over, and the flash-attention key loop walks the same absolute
// blocks in the same order. Every dispatch is therefore the *same arithmetic
// on the same values*, and the output has to come back bit for bit. A
// tolerance here would pass a cache that was subtly off by a position.
//
// The fixture is the 4 k one, because at 256 cells a chunk boundary can only
// fall in one place and the selection never bites. At 4096 tokens the
// schedules below put boundaries inside a pooled indexer block, inside a
// 16-row fragment tile, and — the decode case — at every single token.

// runChunks runs the layer over `in` in the given chunk sizes, starting from a
// fresh sequence, and returns the layer's output for every token.
func runChunks(t *testing.T, g *AttnGPU, in []float32, nEmbd int, chunks []int) []float32 {
	t.Helper()
	g.Reset()
	out := make([]float32, 0, len(in))
	past := 0
	for _, n := range chunks {
		if err := g.SetPast(past); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[past*nEmbd:(past+n)*nEmbd], n); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("chunk of %d at cell %d: %v", n, past, err)
		}
		out = append(out, g.Out()...)
		past += n
	}
	return out
}

// even cuts n into chunks of size k, with whatever is left over last.
func even(n, k int) []int {
	var out []int
	for i := 0; i < n; i += k {
		out = append(out, minInt(k, n-i))
	}
	return out
}

// TestAttnGPUCacheIsAChunkSplit is L7a's gate.
func TestAttnGPUCacheIsAChunkSplit(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals

	// Both runs on one pinned plan. The rungs are chosen from the *run's*
	// length by default, and a rung is a different tiling — so leaving that
	// on would be asking whether two different kernels agree bit for bit,
	// which is not the question this test is about.
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	// And the key split off, for exactly the same reason (P8): it is chosen
	// from the run's length too, so a 4096-token chunk would run the unsplit
	// kernel where a 7-token one ran the split, and the two fold their
	// partial softmaxes together in a different order. The split path has its
	// own chunk gate below — this one is about the cache.
	g.SetSplits(1)
	// And P14-2's gather off, for a reason that is *not* the same: the split
	// could have been made chunk-invariant and was (it strides absolute key
	// blocks); a gathered list cannot be, because it is the union of a query
	// tile's sixteen rows and a chunk that ends inside the tile has fewer rows
	// to union. See AttnGPU.SetGather — it is pinned off with the rest of the
	// reassociating kernels under Graph.PinSchedule, and
	// TestAttnGPUGatherIsTheBlockKernel is the tolerance that replaces this
	// equality for it.
	g.SetGather(false)
	defer g.AutoGather()
	t.Logf("plan pinned at %s / %s / %s, splits and gather off, sparse %v",
		attn, gemm, outGemm, g.Sparse())

	want := runChunks(t, g, in, c.NEmbd, []int{nTok})

	for _, tc := range []struct {
		name   string
		chunks []int
	}{
		{"512 at a time", even(nTok, 512)},
		{"64 at a time, inside a fragment tile", even(nTok, 64)},
		{"7 at a time, crossing every block boundary", even(nTok, 7)},
		{"a prompt and then one token at a time", append([]int{nTok - 33}, even(33, 1)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := runChunks(t, g, in, c.NEmbd, tc.chunks)
			if len(got) != len(want) {
				t.Fatalf("%d values over %d chunks, want %d", len(got), len(tc.chunks), len(want))
			}
			bad, first := 0, -1
			for i := range got {
				if got[i] != want[i] {
					bad++
					if first < 0 {
						first = i
					}
				}
			}
			if bad != 0 {
				t.Errorf("%d of %d values differ, first at token %d component %d: %v against %v",
					bad, len(got), first/c.NEmbd, first%c.NEmbd, got[first], want[first])
				return
			}
			t.Logf("%d chunks, %d tokens: identical to the last place", len(tc.chunks), nTok)
		})
	}
}

// TestAttnGPUCacheKeepsPooledBlocks is the negative control for the one thing
// the chunk split above cannot see.
//
// A pooled indexer block is final as soon as its `ratio` cells exist, so a
// continuing run rebuilds only the blocks its own tokens completed and reads
// every earlier one out of the table. If the table were *not* carried over —
// if a chunk rebuilt only its own blocks and left the rest of the table as
// whatever the previous layer or the previous sequence wrote — the equality
// above would still hold whenever each chunk happened to complete every block
// it needed. This asserts the range directly: it is [past/ratio,
// (past+T)/ratio) on a continuing run and the whole table on a fresh one.
func TestAttnGPUCacheKeepsPooledBlocks(t *testing.T) {
	g, _, c, _, done := attnGPU4k(t)
	defer done()
	for _, tc := range []struct{ past, tokens, lo, hi int }{
		{0, 4096, 0, g.NBlocks()}, // fresh: the whole table, dead blocks and all
		{0, 7, 0, g.NBlocks()},    // fresh and short: still the whole table
		{4000, 96, 1000, 1024},    // the dead blocks become real
		{4000, 1, 1000, 1000},     // one token, no block completed
		{4003, 1, 1000, 1001},     // one token, and it finishes block 1000
		{7, 1, 1, 2},              // a chunk boundary inside block 1
	} {
		name := fmt.Sprintf("%d tokens at cell %d", tc.tokens, tc.past)
		if err := g.SetPast(tc.past); err != nil {
			t.Fatal(err)
		}
		if err := g.Resize(tc.tokens); err != nil {
			t.Fatal(err)
		}
		lo, hi := g.blockRange()
		if lo != tc.lo || hi != tc.hi {
			t.Errorf("%s: blocks [%d, %d), want [%d, %d)", name, lo, hi, tc.lo, tc.hi)
			continue
		}
		// Every block the run must own: complete by the end of it, and not
		// complete before it started.
		for b := 0; b < g.NBlocks(); b++ {
			done := (b+1)*c.Ratio <= tc.past+tc.tokens
			was := (b+1)*c.Ratio <= tc.past
			need := done && !was && tc.past != 0
			if got := b >= lo && b < hi; tc.past != 0 && got != need {
				t.Errorf("%s: block %d dispatched %v, completed-by-this-run %v", name, b, got, need)
			}
		}
	}
}

// TestAttnGPUSplitMatchesUnsplit is P8's gate: cutting the key axis across
// workgroups does not change the answer.
//
// It is a **tolerance and not an equality**, and that is the honest shape for
// it. The split runs an online softmax per slice and folds the slices
// together afterwards, so the rescales happen in a different order from the
// single walk's and fp32 addition is not associative. What must hold is that
// it is the same arithmetic: the same selected cells, and an output that
// differs only where the last places of a different summation order put it.
//
// The schedule is the decode one — a prefill and then one token at a time —
// because that is the only regime the split path runs in. Both arms take the
// same schedule, so the pooled indexer table, the scores and the selection
// are identical by construction and the attention kernel is the only thing
// that differs between them.
func TestAttnGPUSplitMatchesUnsplit(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	// 64 decode steps over a cache already 4032 cells deep, which is past the
	// 2051-cell width, so the selection bites on every one of them.
	chunks := append([]int{nTok - 64}, even(64, 1)...)
	from := (nTok - 64) * c.NEmbd

	g.SetSplits(1)
	want := runChunks(t, g, in, c.NEmbd, chunks)
	defer g.SetSplits(0)

	// Every slice count, because the shape of the disagreement is the
	// evidence. A reordering error grows with how many partial sums are
	// folded together and stays far below the arithmetic's own noise; a
	// *bug* — a dropped slice, a mis-scaled one — does not care how many
	// there are, or grows in proportion to what it drops.
	for _, splits := range []int{2, 4, 8, attnMaxSplits} {
		g.SetSplits(splits)
		got := runChunks(t, g, in, c.NEmbd, chunks)
		if len(got) != len(want) {
			t.Fatalf("%d values, want %d", len(got), len(want))
		}
		var maxAbs, sum2, ref2 float64
		for i := from; i < len(got); i++ {
			d := math.Abs(float64(got[i]) - float64(want[i]))
			maxAbs = math.Max(maxAbs, d)
			sum2 += d * d
			ref2 += float64(want[i]) * float64(want[i])
		}
		rms := math.Sqrt(sum2 / ref2)
		t.Logf("%2d splits: rms %.3g, max abs %.3g", splits, rms, maxAbs)
		// The bar is the arithmetic's own noise, not zero. The dense bank
		// this layer reads is itself 5.4e-3 rms from the f32 model
		// (TestAttnGPULayer4k), and llama.cpp's own output is compared at
		// 2e-2 — so a reordering that lands two orders of magnitude under
		// the quantisation is not a difference this model can carry.
		if rms > 1e-3 {
			t.Errorf("%d splits: rms %.3g against the unsplit kernel", splits, rms)
		}
	}

	// And the bar itself, measured rather than asserted: what a decode step
	// costs against **llama.cpp's own output** on each arm. This is the
	// comparison that decides whether the split is wrong, because neither
	// kernel is the truth — the layer already sits ~1e-3 rms from the
	// reference (TestAttnGPULayer4k), most of it the fp16 indexer the device
	// cannot model (TestAttnGPUIndexer4k). A split that moved the output less
	// than that is inside the noise the layer already has.
	want4k, err := tr.Get("attn_output-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	last := (nTok - 1) * c.NEmbd
	ref := want4k.Vals[last : last+c.NEmbd]
	for _, splits := range []int{1, attnMaxSplits} {
		g.SetSplits(splits)
		out := runChunks(t, g, in, c.NEmbd, []int{nTok - 1, 1})
		r, err := compare(out[last:last+c.NEmbd], ref)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("decode step at cell %d, %2d splits, against llama.cpp: %v", nTok-1, splits, r)
		if r.rms > 5e-3 {
			t.Errorf("%d splits: rms %.3e against llama.cpp", splits, r.rms)
		}
	}
}

// TestAttnGPUSplitSelectsTheSameCells is the other half of the gate above:
// the split changes the *order* the selected cells are folded in and nothing
// about which cells they are.
//
// The selection is written by llm_attn_select.comp, which the split does not
// touch — so this is a control, and it is here because a split that silently
// dropped a slice's blocks would show up as a small rms above and as nothing
// at all here, which is exactly the pair of symptoms that would send the next
// reader to the wrong kernel.
func TestAttnGPUSplitSelectsTheSameCells(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals
	sel := func(splits int) []uint32 {
		g.SetSplits(splits)
		g.Reset()
		if err := g.SetPast(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[:(nTok-1)*c.NEmbd], nTok-1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		if err := g.SetPast(nTok - 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[(nTok-1)*c.NEmbd:], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]uint32(nil), g.Selection()...)
	}
	off, on := sel(1), sel(attnMaxSplits)
	g.SetSplits(0)
	if len(off) != len(on) {
		t.Fatalf("%d words against %d", len(on), len(off))
	}
	bits := 0
	for i := range off {
		if off[i] != on[i] {
			t.Fatalf("selection word %d: %#08x with the split, %#08x without", i, on[i], off[i])
		}
		bits += bitsSet(off[i])
	}
	t.Logf("%d selected cells, identical with the split and without", bits)
}

func bitsSet(x uint32) int {
	n := 0
	for ; x != 0; x &= x - 1 {
		n++
	}
	return n
}

// TestAttnGPUCellStripeDoesNotChangeTheAnswer is P9's gate, and unlike P8's
// it is an **equality**.
//
// The indexer's score and its expansion to cells were one dispatch of one
// workgroup a token, because the expansion reads every block's score and the
// barrier that enforced that also pinned the scoring. P9 makes them two
// dispatches striped over the grid's y. Nothing in either sums across the
// stripe — every output element is one read, one add and one store — so
// which lane writes it is not observable, and the tensors have to come back
// bit for bit however many ways the grid is cut. If they do not, the stripe
// is racing or dropping elements, and both of those are invisible to a
// tolerance on the layer's output: a dropped cell score is one -inf among
// 2051 selected cells, which moves the answer by almost nothing and moves the
// *selection* by a cell.
func TestAttnGPUCellStripeDoesNotChangeTheAnswer(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4kExpanded(t)
	defer done()
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	g.SetSplits(1) // P8's arm is not the subject here
	defer g.SetSplits(0)

	run := func(stripes string) (score, cell []float32, sel []uint32) {
		t.Setenv("LLM_ATTN_CELL_SPLITS", stripes)
		g.Reset()
		if err := g.SetPast(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[:(nTok-1)*c.NEmbd], nTok-1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		if err := g.SetPast(nTok - 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[(nTok-1)*c.NEmbd:], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.Score()...), append([]float32(nil), g.Cells()...),
			append([]uint32(nil), g.Selection()...)
	}

	wantScore, wantCell, wantSel := run("1")
	for _, stripes := range []string{"2", "7", "16", "64"} {
		gotScore, gotCell, gotSel := run(stripes)
		for _, tc := range []struct {
			name      string
			got, want []float32
		}{
			{"indexer score", gotScore, wantScore},
			{"expanded cells", gotCell, wantCell},
		} {
			if len(tc.got) != len(tc.want) {
				t.Fatalf("%s stripes, %s: %d values against %d", stripes, tc.name, len(tc.got), len(tc.want))
			}
			for i := range tc.got {
				a, b := tc.got[i], tc.want[i]
				// -inf is most of the cell row and -inf == -inf, so this is
				// an ordinary comparison and not a bit one.
				if a != b {
					t.Fatalf("%s stripes, %s: element %d is %v against %v", stripes, tc.name, i, a, b)
				}
			}
		}
		for i := range gotSel {
			if gotSel[i] != wantSel[i] {
				t.Fatalf("%s stripes: selection word %d is %#08x against %#08x",
					stripes, i, gotSel[i], wantSel[i])
			}
		}
		t.Logf("%2s stripes: score, cells and selection identical to the last bit", stripes)
	}
}

// TestAttnGPUSelectWidthDoesNotChangeTheAnswer is P10's gate, and like P9's
// it is an **equality**.
//
// The selection is one workgroup a token and stays that way; what changed is
// how wide that workgroup is — 256 lanes to 1024, so that sixteen waves
// rather than four are resident on the one compute unit it lands on and the
// five streams it makes of the row at depth have something to hide behind.
//
// Nothing about the answer may move. The radix histogram is built with LDS
// atomics, which commute; the bucket search is a suffix sum whose result is a
// count and not an order; the emit's running `taken` is computed identically
// on every lane by construction. The one thing width *could* have broken is
// the bucket vote — lanes past the radix have no bucket, and a pass whose
// `desired` has fallen to zero would elect the highest *lane* rather than the
// highest bucket — so this asserts the selected set bit for bit at both
// widths, which is the only place that failure would show.
func TestAttnGPUSelectWidthDoesNotChangeTheAnswer(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals

	run := func(wg string) ([]uint32, []float32) {
		t.Setenv("LLM_ATTN_SELECT_WG", wg)
		g.Reset()
		// A prefill and then one decode step, so both the wide-batch path
		// and the one-token path are covered at the same width.
		if err := g.SetPast(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[:(nTok-1)*c.NEmbd], nTok-1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		wide := append([]uint32(nil), g.Selection()...)
		if err := g.SetPast(nTok - 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[(nTok-1)*c.NEmbd:], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append(wide, g.Selection()...), append([]float32(nil), g.Out()...)
	}

	wantSel, wantOut := run("256")
	gotSel, gotOut := run("")
	if len(gotSel) != len(wantSel) {
		t.Fatalf("%d selection words against %d", len(gotSel), len(wantSel))
	}
	bits := 0
	for i := range wantSel {
		if gotSel[i] != wantSel[i] {
			t.Fatalf("selection word %d: %#08x at 1024 lanes, %#08x at 256", i, gotSel[i], wantSel[i])
		}
		bits += bitsSet(wantSel[i])
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("layer output %d: %v at 1024 lanes, %v at 256", i, gotOut[i], wantOut[i])
		}
	}
	t.Logf("%d selected cells over a prefill and a decode step, and the layer's "+
		"output, identical to the last bit at both widths", bits)
}
