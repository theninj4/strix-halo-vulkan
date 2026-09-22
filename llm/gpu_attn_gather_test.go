package llm

import (
	"math"
	"testing"
)

// P14-2: the per-cell gather, and the two gates it needs.
//
// The first is an **equality on the set**, which is the part that can be exact:
// the gathered list is the union of a query tile's rows' selections restricted
// to what the tile can causally see, and the per-row mask says which of the
// union each row selected. Both are computable from the bitmask the selection
// wrote, so the kernel is checked against that and not against a tolerance.
//
// The second is a **tolerance on the answer**, which is the part that cannot be
// exact. The gathered kernel folds the same terms in a different order than the
// block kernel does — see AttnGPU.SetGather — so the two differ in the last
// places, and what has to hold is that the difference is rounding and not a
// missing cell: a gather that dropped a selected cell would move the layer's
// output by far more than the two kernels' distance from the reference.

// TestAttnGPUGatherSelectsTheSameCells is the exact half. Every tile, every row:
// the list is ascending and holds exactly the union, and the mask is exactly the
// selection AND the row's causal extent.
func TestAttnGPUGatherSelectsTheSameCells(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	g.SetGather(true)
	defer g.AutoGather()
	// The token-tile layout is what Gathered reads; P17's per-token one is
	// TestAttnGPUHeadRowsIsTheGather's.
	g.SetHeadRows(false)
	defer g.AutoHeadRows()

	// Two shapes: the whole prompt from a cold cache, and a second chunk on top
	// of it, so the causal extent the gather folds in is exercised at a
	// non-zero `past` as well as at zero.
	for _, chunks := range [][]int{{nTok}, {nTok - 512, 512}} {
		g.Reset()
		past := 0
		for _, n := range chunks {
			if err := g.SetPast(past); err != nil {
				t.Fatal(err)
			}
			if err := g.Upload(src.Vals[past*c.NEmbd:(past+n)*c.NEmbd], n); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(0); err != nil {
				t.Fatal(err)
			}

			mask := g.Selection()
			words := g.selWords()
			tiles := (n + gathBM - 1) / gathBM
			totalCells, totalBits := 0, 0
			for tile := 0; tile < tiles; tile++ {
				q0 := tile * gathBM
				// What the union has to be: the OR of this tile's real rows,
				// cut at the last cell any of them can see.
				last := minInt(past+n, past+q0+gathBM)
				want := map[int32]bool{}
				for r := 0; r < gathBM && q0+r < n; r++ {
					row := mask[(q0+r)*words:]
					for cell := 0; cell < last; cell++ {
						if row[cell>>5]&(1<<uint(cell&31)) != 0 {
							want[int32(cell)] = true
						}
					}
				}
				cells, rowMask := g.Gathered(tile)
				if len(cells) != len(want) {
					t.Fatalf("chunk %v, tile %d at cell %d: %d gathered cells, the union is %d",
						chunks, tile, past+q0, len(cells), len(want))
				}
				for i, cell := range cells {
					if !want[cell] {
						t.Fatalf("chunk %v, tile %d: gathered cell %d is %d, which no row selected",
							chunks, tile, i, cell)
					}
					if i > 0 && cells[i-1] >= cell {
						t.Fatalf("chunk %v, tile %d: the list is not ascending at %d (%d after %d)",
							chunks, tile, i, cell, cells[i-1])
					}
				}
				// And the mask: the selection AND this row's own extent, which
				// is what lets the attention loop have no causal test in it.
				for r := 0; r < gathBM; r++ {
					for i, cell := range cells {
						sel := false
						if q0+r < n {
							row := mask[(q0+r)*words:]
							sel = row[cell>>5]&(1<<uint(cell&31)) != 0 &&
								int(cell) <= past+q0+r
						}
						if rowMask[r][i] != sel {
							t.Fatalf("chunk %v, tile %d row %d, cell %d: mask says %v, "+
								"the selection and the causal extent say %v",
								chunks, tile, r, cell, rowMask[r][i], sel)
						}
						if sel {
							totalBits++
						}
					}
				}
				totalCells += len(cells)
			}
			t.Logf("chunk of %d at cell %d: %d tiles, %d gathered cells, %d live (row, cell) pairs, "+
				"%.2f cells a tile against %d the block kernel would read",
				n, past, tiles, totalCells, totalBits,
				float64(totalCells)/float64(maxInt(tiles, 1)), g.NKV())
			past += n
		}
	}
}

// TestAttnGPUGatherIsTheBlockKernel is the tolerance half, and it is the same
// argument TestAttnGPUSplitMatchesDense makes for P8: the two kernels are the
// same arithmetic over the same set in a different order, so they are held to
// the **reference** rather than to each other, and their distance from each
// other has to be small beside their distance from it.
func TestAttnGPUGatherIsTheBlockKernel(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := tr.Get("attn_output-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	// One pinned plan for both, and the split off: the question is the gather.
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	g.SetSplits(1)
	defer g.SetSplits(0)

	run := func(on bool) []float32 {
		g.SetGather(on)
		if g.Gathers() != on {
			t.Fatalf("asked for gather %v and got %v", on, g.Gathers())
		}
		g.Reset()
		if err := g.SetPast(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(src.Vals[:nTok*c.NEmbd], nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.Out()...)
	}
	blocks := run(false)
	gathered := run(true)
	g.AutoGather()

	rms := func(a, b []float32) (r, worst float64) {
		var acc float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			acc += d * d
			if math.Abs(d) > worst {
				worst = math.Abs(d)
			}
		}
		return math.Sqrt(acc / float64(maxInt(len(a), 1))), worst
	}
	pair, pairWorst := rms(gathered, blocks)
	vsRefB, _ := rms(blocks, ref.Vals)
	vsRefG, _ := rms(gathered, ref.Vals)

	// **NaN first, because every comparison below is false against one** and a
	// kernel that reads LDS a barrier did not cover produces exactly that. This
	// test reported PASS on a NaN once.
	if math.IsNaN(pair) || math.IsNaN(vsRefG) {
		t.Fatalf("the gathered kernel produced NaN (pair %v, vs reference %v) — "+
			"something is read before it is written", pair, vsRefG)
	}

	// The bar: the two kernels must be nearer each other than either is to the
	// reference. A gather that dropped a selected cell — or gathered one no row
	// selected — would fail this by orders, because the layer's output is a
	// softmax over the cells it read.
	if pair >= vsRefB || pair >= vsRefG {
		t.Errorf("the two kernels differ by %.3e rms, which is not small beside "+
			"their %.3e / %.3e from llama.cpp — the gather is not the same set",
			pair, vsRefB, vsRefG)
	}
	// And the gathered kernel must not be *further* from the reference than the
	// one it replaces by more than the rounding between them.
	if vsRefG > vsRefB+pair {
		t.Errorf("the gathered kernel is %.3e from llama.cpp against the block "+
			"kernel's %.3e, further than the %.3e between them", vsRefG, vsRefB, pair)
	}
	t.Logf("gathered vs blocks %.3e rms (worst %.3e); vs llama.cpp %.3e gathered, "+
		"%.3e blocks — the same set, folded in a different order",
		pair, pairWorst, vsRefG, vsRefB)
}

// TestAttnGPUGatherRungsAgree is the gate the gathered ladder needs for exactly
// the reason P13's did: a rung changes the *shape* of what the kernel stages, and
// a shape that is wrong is still a perfectly well-formed attention over the wrong
// cells — which every tolerance against llama.cpp passes by being roughly a dense
// causal attention.
//
// The shape here is the mask slice. A gathered chunk stages `BW = BN/32` words of
// the compacted mask a row, and the mask pass writes `ceil(count/32)` of them
// rounded up to the widest rung's BW; a 64-cell rung whose last chunk asked for a
// second word the pass had not written would read the previous pass's bits, and
// the consuming loop has no `col >= cnt` test any more because the mask is what
// replaced it. That is a one-chunk-in-two-hundred error at one rung, and this is
// what makes it visible.
func TestAttnGPUGatherRungsAgree(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	g.SetSplits(1)
	defer g.SetSplits(0)
	g.SetGather(true)
	defer g.AutoGather()

	run := func(rung string) []float32 {
		t.Setenv("LLM_ATTN_GATHER_KERNEL", rung)
		g.Reset()
		if err := g.SetPast(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(src.Vals[:nTok*c.NEmbd], nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("rung %s: %v", rung, err)
		}
		return append([]float32(nil), g.Out()...)
	}

	shipped := string(g.gathRung())
	want := run(shipped)
	for _, rung := range []string{"qt1_kt1", "qt1_kt2", "qt1_kt4"} {
		got := run(rung)
		var acc, worst float64
		for i := range got {
			d := float64(got[i]) - float64(want[i])
			acc += d * d
			if math.Abs(d) > worst {
				worst = math.Abs(d)
			}
		}
		rms := math.Sqrt(acc / float64(maxInt(len(got), 1)))
		// The rungs tile the same arithmetic differently, so this is a rounding
		// bar and not zero — but it is an order under the 9.9e-04 the layer
		// already sits from llama.cpp, which is what a wrong cell set would blow
		// through.
		if rms > 1e-4 {
			t.Errorf("gathered rung %s is %.3e rms from %s (worst %.3e) — that is a "+
				"different set of cells, not a different order", rung, rms, shipped, worst)
			continue
		}
		t.Logf("gathered rung %-8s %.3e rms, worst %.3e", rung, rms, worst)
	}
}

// TestAttnGPUGatherIsDispatched is the shape gate: a prefill wide enough to take
// the gather dispatches the two compaction passes and the gathered rung, a
// decode step takes neither, and P13's block list is not built beside it.
func TestAttnGPUGatherIsDispatched(t *testing.T) {
	g, _, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	in := make([]float32, nTok*c.NEmbd)

	kindsFor := func(rows int) map[string]int {
		if err := g.Upload(in[:rows*c.NEmbd], rows); err != nil {
			t.Fatal(err)
		}
		_, kinds, err := g.graph(0)
		if err != nil {
			t.Fatal(err)
		}
		out := map[string]int{}
		for _, k := range kinds {
			out[k]++
		}
		return out
	}

	wide := kindsFor(nTok)
	for _, k := range []string{"attn.gather", "attn.gathmask", "attn"} {
		if wide[k] != 1 {
			t.Errorf("a %d-row pass dispatches %d %q, want 1", nTok, wide[k], k)
		}
	}
	if wide["attn.split"] != 0 || wide["blocks"] != 0 {
		t.Errorf("a gathered pass also dispatches %d splits and %d block lists",
			wide["attn.split"], wide["blocks"])
	}

	// A decode step is one query tile: its 24 single-wave workgroups all fit at
	// once, so what it costs is one wave's serial walk (P8) — which a gather in
	// front of it lengthens rather than cuts, on top of two dispatches with one
	// tile to amortise them over.
	one := kindsFor(1)
	if one["attn.gather"] != 0 || one["attn.gathmask"] != 0 {
		t.Errorf("a one-row pass dispatches %d gathers and %d masks",
			one["attn.gather"], one["attn.gathmask"])
	}

	// And the arena, which is what P14-1's freed expansion paid for.
	t.Logf("%d query tiles x %d uints = %.1f MB of gather arena at %d rows over %d cells "+
		"(the expansion this replaces was %.1f MB)",
		g.gathTiles, g.gathStride(), float64(g.gathTiles*g.gathStride()*4)/1e6,
		g.arenaRows, g.NKV(), float64(g.arenaRows*g.NKV()*4)/1e6)
}

// TestAttnGPUGatherSplitIsTheGather is P15's gate: the gathered kernel with its
// axis cut across workgroups is the gathered kernel.
//
// It is the same argument TestAttnGPUGatherIsTheBlockKernel makes one level up,
// and for the same reason — a split folds the online softmax's partial maxima
// in a different order, so the two are the same arithmetic over the same set
// and not the same bits. The bar is the arithmetic's own noise, exactly as
// TestAttnGPUSplitMatchesUnsplit sets it for the block kernel.
//
// **It has to be run as decode steps and not as a prompt**, because
// `attnSplits` refuses any batch wider than `attnSplitMaxRows` — a wide batch
// fills the grid on the query axis alone and the split buys nothing there. A
// version of this test that uploaded the whole fixture at once passed
// instantly with a 0.000e+00 rms on every arm, which is what a split that never
// engaged looks like. So the shape below is the cache 4032 cells deep and then
// 64 single-token steps, and `from` reads only those.
//
// The slice counts are swept rather than sampled because the failure this is
// really watching for is a *boundary* one: a slice whose first chunk starts
// past `nGath` has to fall through to the epilogue and write the combine's
// identity, and a count that divides the chunk count evenly would never
// produce such a slice.
func TestAttnGPUGatherSplitIsTheGather(t *testing.T) {
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
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	// 64 decode steps over a cache already past the 2051-cell selection width,
	// so the selection bites on every one of them and the gather has something
	// to compact.
	chunks := append([]int{nTok - 64}, even(64, 1)...)
	from := (nTok - 64) * c.NEmbd

	g.SetGather(true)
	defer g.AutoGather()
	defer g.SetSplits(0)

	g.SetSplits(1)
	want := runChunks(t, g, in, c.NEmbd, chunks)

	for _, splits := range []int{2, 3, 5, 8, 16, attnMaxSplits} {
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
		// NaN first: every comparison below is false against one, and a slice
		// that read LDS a barrier did not cover produces exactly that. The
		// sibling test reported PASS on a NaN once.
		if math.IsNaN(rms) {
			t.Fatalf("%d slices produced NaN — something is read before it is written", splits)
		}
		// And zero is a failure too, here: it means the split did not engage,
		// which is how the first draft of this test passed without testing
		// anything.
		if rms == 0 {
			t.Fatalf("%d slices came back bit-identical to the unsplit gather, "+
				"which means the split never engaged — check attnSplits against "+
				"the row count", splits)
		}
		t.Logf("%2d slices: rms %.3g, max abs %.3g", splits, rms, maxAbs)
		if rms > 1e-3 {
			t.Errorf("%d slices: rms %.3g against the unsplit gather", splits, rms)
		}
	}
}

// TestAttnGPUHeadRowsIsTheGather is P17's gate: the gathered attention with a
// token's query heads on the fragment's rows (`-DHROWS`) against the token-row
// gathered kernel it replaces, at both of its shapes — a prompt (unsplit, one
// workgroup a token and kv head) and decode steps (split, over the same
// per-token list).
//
// The two read the same set: every row of a heads fragment selected exactly the
// cells of its token's list, where a token fragment masked the tile's union down
// to them. So the bar is the rounding between two orders of the same online
// softmax, and it is set by the gathered kernel's own distance from the block
// kernel — well under llama.cpp's. **Zero is a failure**, because it is what a
// build that never engaged looks like; the split's sibling test cannot see this
// arm at all, since its noise is the fp16 store of the unsplit output.
func TestAttnGPUHeadRowsIsTheGather(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	if !g.Sparse() {
		t.Skip("the selection does not bite at this cache size")
	}
	if !g.hrowsFits() {
		t.Skip("this head geometry does not fit a sixteen-row fragment")
	}
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := tr.Get("attn_output-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	g.SetGather(true)
	defer g.AutoGather()
	defer g.AutoHeadRows()
	defer g.SetSplits(0)

	cmp := func(a, b []float32, from int) (rms, worst float64) {
		var s2, r2 float64
		for i := from; i < len(a); i++ {
			d := float64(a[i]) - float64(b[i])
			s2 += d * d
			r2 += float64(b[i]) * float64(b[i])
			worst = math.Max(worst, math.Abs(d))
		}
		return math.Sqrt(s2 / math.Max(r2, 1e-30)), worst
	}

	// The prompt, whole: the unsplit build.
	prompt := []int{nTok}
	g.SetSplits(1)
	g.SetHeadRows(false)
	tokRows := runChunks(t, g, in, c.NEmbd, prompt)
	g.SetHeadRows(true)
	if !g.headRows() {
		t.Fatal("asked for heads on the rows and the pass refused it")
	}
	headRows := runChunks(t, g, in, c.NEmbd, prompt)
	pair, worst := cmp(headRows, tokRows, 0)
	vsRef, _ := cmp(headRows, ref.Vals[:len(headRows)], 0)
	vsRefTok, _ := cmp(tokRows, ref.Vals[:len(tokRows)], 0)
	if math.IsNaN(pair) || math.IsNaN(vsRef) {
		t.Fatalf("heads on the rows produced NaN (pair %v, vs reference %v)", pair, vsRef)
	}
	if pair == 0 {
		t.Fatal("heads on the rows came back bit-identical to the token rows — the build never engaged")
	}
	t.Logf("prompt: heads vs tokens %.3e rel rms (worst %.3e); vs llama.cpp %.3e heads, %.3e tokens",
		pair, worst, vsRef, vsRefTok)
	if pair > 1e-3 || vsRef > vsRefTok+pair {
		t.Errorf("prompt: heads on the rows is %.3e from the token rows and %.3e from llama.cpp (tokens %.3e)",
			pair, vsRef, vsRefTok)
	}

	// Decode steps over a cache past the selection's width: the split build.
	//
	// **Here the two are the same bits, and that is the assertion.** At one row
	// the token fragment's union *is* the token's selection, so both kernels
	// fold the same cells in the same chunks in the same order, and a matrix
	// product's rows are independent — which row of the fragment a head lands
	// in does not change its sums. So engagement cannot be read off the answer
	// at decode; it is read off the dispatch list instead, below.
	chunks := append([]int{nTok - 64}, even(64, 1)...)
	from := (nTok - 64) * c.NEmbd
	for _, splits := range []int{1, 3, attnDefaultSplits} {
		g.SetSplits(splits)
		g.SetHeadRows(false)
		want := runChunks(t, g, in, c.NEmbd, chunks)
		g.SetHeadRows(true)
		got := runChunks(t, g, in, c.NEmbd, chunks)
		rms, worst := cmp(got, want, from)
		if math.IsNaN(rms) {
			t.Fatalf("%d slices: heads on the rows produced NaN", splits)
		}
		t.Logf("decode, %2d slices: heads vs tokens %.3e rel rms, worst %.3e", splits, rms, worst)
		if rms != 0 {
			t.Errorf("%d slices: a one-row pass is %.3e from the token rows, want the same bits", splits, rms)
		}
	}

	// Engagement at one row: the heads build reads the per-token list and has
	// no mask pass behind it.
	g.SetSplits(attnDefaultSplits)
	g.SetHeadRows(true)
	if err := g.Upload(in[:c.NEmbd], 1); err != nil {
		t.Fatal(err)
	}
	_, kinds, err := g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	n := map[string]int{}
	for _, k := range kinds {
		n[k]++
	}
	if n["attn.gather"] != 1 || n["attn.gathmask"] != 0 || n["attn.split"] != 1 {
		t.Errorf("a one-row heads-on-rows pass dispatches %d gathers, %d masks, %d splits; want 1, 0, 1",
			n["attn.gather"], n["attn.gathmask"], n["attn.split"])
	}
}
