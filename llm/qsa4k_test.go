package llm

import (
	"math"
	"os"
	"testing"
)

// L4: the QSA selection, at the only context length where it exists.
//
// Everything in L2 and L3 was checked against a 7-token dump, and at 7 tokens
// this block is a no-op: the reference asks for `top_k + ratio - 1` = 2051
// cells out of a 256-cell cache, so it names every one of them and running
// the attention with the selection is bit-identical to running it without
// (TestAttnTopKCannotBiteHere). A second dump at 4096 tokens in a 4096-cell
// cache is where the width finally binds — 2051 of 4096 — and where the two
// questions L2e left open can be asked at all: what the selection *is*, and
// whether ours is the reference's.
//
// The fixture is a second trace directory rather than a replacement, because
// the 7-token one is still the only place a great many things are legible:
// the empty blocks that pool cell zero, the spare block, the f32 vector path
// that L3a-5 found the reference takes below 8 output columns. The two are
// complementary and both are kept.
const (
	traceDir4k  = "../reference/out/llm4k"
	regenHint4k = `run:
  L=/home/kube/repos/llama.cpp
  H=$(mktemp -d) && (cd $L && git archive cff184438 include ggml/include | tar -x -C $H)
  gcc -O2 -o /tmp/eval_dump reference/eval_dump.c -I$H/include -I$H/ggml/include \
      -L$L/build/bin -lllama -lggml-base -Wl,-rpath,$L/build/bin
  mkdir -p reference/out/llm4k
  head -c 40000 models/wikitext-2-raw/wiki.test.raw | head -n -1 > reference/out/llm4k/prompt.txt
  /tmp/eval_dump -m $M -o reference/out/llm4k -f reference/out/llm4k/prompt.txt \
      -nt 4096 -c 4096 -ub 4096 \
      -n '^(model\.input_embed|ple_embd|hc_init)$|-(0|3)$|^ple_(gate|gated_value|conv_out)-1$'
  (8.0 GB, ~4 minutes; the headers are pinned to the *build* in $L/build/bin,
   which is cff184438 even when the checkout has moved past it)`
)

// fixtures4k is the 4096-token trace beside the 7-token one.
func fixtures4k(t *testing.T) (*Model, *Trace) {
	t.Helper()
	if _, err := os.Stat(modelDir); err != nil {
		t.Skipf("checkpoint absent: %v", err)
	}
	if _, err := os.Stat(traceDir4k); err != nil {
		t.Skipf("4k reference trace absent (%v); %s", err, regenHint4k)
	}
	tr, err := OpenTrace(traceDir4k)
	if err != nil {
		t.Skipf("4k reference trace unusable (%v); %s", err, regenHint4k)
	}
	m, err := Open(modelDir)
	if err != nil {
		t.Fatalf("open checkpoint: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m, tr
}

// qsaFixtures4k opens layer 3 at 4096 tokens and returns the reference's own
// selection beside it. nKV is read off the dump, not assumed: it is the
// padded cell count and every part of the indexer is cut against it.
func qsaFixtures4k(t *testing.T) (c AttnConfig, w AttnWeights, nTok, nKV, width int, tr *Trace, topk *Dump) {
	t.Helper()
	m, tr := fixtures4k(t)
	c, ok, err := m.AttnConfig(attnLayer)
	if err != nil || !ok {
		t.Fatalf("layer %d attention config: %v (present %v)", attnLayer, err, ok)
	}
	if w, err = m.AttnWeights(attnLayer); err != nil {
		t.Fatal(err)
	}
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok = len(ids)
	cells, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	nKV = int(cells.NE[0])
	if topk, err = tr.Get("indexer_top_k-3", 0); err != nil {
		t.Fatal(err)
	}
	if topk.Ints == nil {
		t.Fatalf("indexer_top_k-3 is ggml type %d, want an integer tensor", topk.GType)
	}
	width = int(topk.NE[0])
	return c, w, nTok, nKV, width, tr, topk
}

// TestQSASelectionBitesAt4k is the precondition every other test in this file
// rests on, asserted rather than assumed: at 4096 cells the width is 2051, so
// the selection removes cells the causal mask allows.
//
// It also locates exactly where that starts. A token at position i has i+1
// visible cells, so the first token whose selection can drop anything is
// i = width, and the reference's own output says so.
func TestQSASelectionBitesAt4k(t *testing.T) {
	_, _, nTok, nKV, width, _, topk := qsaFixtures4k(t)
	if width >= nKV {
		t.Fatalf("width %d of %d cells: the selection still cannot bite, so this dump is too short", width, nKV)
	}
	t.Logf("%d tokens, %d cells, top-k width %d — %d cells a token are unreachable",
		nTok, nKV, width, nKV-width)

	first := -1
	var worstDropped int
	for i := 0; i < nTok; i++ {
		sel := map[int32]bool{}
		for _, cell := range topk.Ints[i*width : (i+1)*width] {
			sel[cell] = true
		}
		dropped := 0
		for j := 0; j <= i; j++ {
			if !sel[int32(j)] {
				dropped++
			}
		}
		if dropped > 0 && first < 0 {
			first = i
		}
		if dropped > worstDropped {
			worstDropped = dropped
		}
		if dropped > 0 && i+1 <= width {
			t.Errorf("token %d has %d visible cells and drops %d, but the width is %d", i, i+1, dropped, width)
		}
	}
	if first < 0 {
		t.Fatal("no token drops a visible cell: the selection is still a no-op")
	}
	t.Logf("the selection first drops a visible cell at token %d, and drops at most %d of them", first, worstDropped)
	if first != width {
		t.Errorf("the selection starts biting at token %d, want %d (= the width)", first, width)
	}
	if want := nTok - width; worstDropped != want {
		t.Errorf("the last token drops %d visible cells, want %d", worstDropped, want)
	}
}

// TestQSASelectionIsARadixSelect is the L4 gate on the selection itself, and
// it is deliberately fed llama.cpp's *own* per-cell scores rather than ours:
// what is under test is the selection rule, not the arithmetic in front of it.
//
// The rule is `topk_radix_select.comp`, and it is not a sort. It finds the key
// of the width-th largest value by four 8-bit radix passes, emits every cell
// **strictly above** it, and then fills the remainder from the cells **equal**
// to it — in `atomicAdd` order, which the algorithm does not fix.
//
// So there are two claims here and they are of different strengths. The
// algorithmic one holds on every row: every cell we call strictly-above is in
// the reference's selection, and every cell the reference has that we do not
// is one of ours at the threshold. The stronger one holds on the 2045 rows
// where the selection actually bites: taking the tied cells in **ascending
// cell index** reproduces the reference's set exactly, all 2045 of them, with
// no cells to spare. That is not guaranteed by the shader — it is what a wave
// resolving an atomic in lane order does over a tie group of `ratio`
// consecutive cells — so it is asserted as a measured property rather than
// relied on, and TestQSASelectionIsStableAcrossRuns is the other half of the
// evidence for it.
func TestQSASelectionIsARadixSelect(t *testing.T) {
	_, _, nTok, nKV, width, tr, topk := qsaFixtures4k(t)
	cells, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}

	fill := map[int]int{}
	for i := 0; i < nTok; i++ {
		row := cells.Vals[i*nKV : (i+1)*nKV]
		sel, above := topK(row, width)
		if len(sel) != width {
			t.Fatalf("token %d: our selection is %d cells, the reference's is %d", i, len(sel), width)
		}
		thr := topKThreshold(row, width)

		ref := map[int32]bool{}
		for _, cell := range topk.Ints[i*width : (i+1)*width] {
			if ref[cell] {
				t.Fatalf("token %d: the reference names cell %d twice", i, cell)
			}
			ref[cell] = true
		}
		for _, cell := range sel[:above] {
			if !ref[cell] {
				t.Fatalf("token %d: cell %d is strictly above the threshold and the reference does not select it",
					i, cell)
			}
		}
		ours := map[int32]bool{}
		for _, cell := range sel[:above] {
			ours[cell] = true
		}
		for cell := range ref {
			if ours[cell] {
				continue
			}
			if f2ui(row[cell]) != thr {
				t.Fatalf("token %d: the reference selects cell %d, which is neither above the threshold nor at it (%g vs key %#x)",
					i, cell, row[cell], thr)
			}
		}
		if i+1 > width {
			fill[width-above]++
			// Where the tie group is one block, our ascending-index fill is
			// the reference's choice — set for set, with nothing over.
			if len(ref) != width {
				t.Fatalf("token %d: the reference names %d distinct cells of %d", i, len(ref), width)
			}
			for _, cell := range sel {
				if !ref[cell] {
					t.Fatalf("token %d: our ascending-index tie fill takes cell %d, which the reference does not",
						i, cell)
				}
			}
		}
	}

	// The fill count is the size of the ambiguity, and it is not incidental:
	// all `ratio` cells of a block carry one block score, so the tie group at
	// the threshold *is* a block and the fill takes 1, 2 or 3 of its 4 cells.
	//
	// Only the rows where the selection bites are counted. Below that the
	// threshold is -inf, the tie group is every masked cell in the row, and
	// the fill is hundreds of cells that the causal mask drops again — real,
	// but unobservable, which is exactly why L2e could not test any of this.
	t.Logf("tie fill over the %d tokens where the selection bites: %v", nTok-width, fill)
	worst := 0
	for n := range fill {
		if n > worst {
			worst = n
		}
	}
	// The fill is at most `ratio`, because the tie group is one block. A fill
	// of exactly 4 is the benign case — the threshold landed on the last cell
	// of a block and the whole block goes in; 1, 2 or 3 is the case where a
	// block is split and the reference's choice of which cells is arbitrary.
	if worst > 4 {
		t.Errorf("the tie fill reaches %d cells; a block is 4, so the tie group is larger than one block", worst)
	}
}

// TestQSASelectionIsStableAcrossRuns is the evidence behind the previous
// test's stronger half, and it is a property of this GPU rather than of the
// algorithm — which is exactly why it is measured instead of assumed.
//
// A second dump of the same prompt gives **byte-identical** per-cell scores,
// a top-k whose *order* differs on all 4096 rows, and a top-k whose *set*
// differs on 1967 of them — all of it among the -inf cells the causal mask
// drops. Restricted to the cells attention can actually read, the two runs
// agree on every row and every cell. So the reference's selection is
// reproducible in practice while its order is not reproducible at all, and a
// kernel of ours is free to emit whatever order is cheapest.
//
// It skips when the second dump is absent, since it is a second four-minute
// model load for one bit of information:
//
//	/tmp/eval_dump -m $M -o reference/out/llm4k_rerun -f reference/out/llm4k/prompt.txt \
//	    -nt 4096 -c 4096 -ub 4096 -n '^indexer_(top_k|score_tokens)-3$'
func TestQSASelectionIsStableAcrossRuns(t *testing.T) {
	_, _, nTok, nKV, width, tr, topk := qsaFixtures4k(t)
	const rerunDir = "../reference/out/llm4k_rerun"
	if _, err := os.Stat(rerunDir); err != nil {
		t.Skipf("no second dump to compare against (%v)", err)
	}
	re, err := OpenTrace(rerunDir)
	if err != nil {
		t.Skipf("second dump unusable: %v", err)
	}
	again, err := re.Get("indexer_top_k-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if int(again.NE[0]) != width || int(again.NE[1]) != nTok {
		t.Fatalf("the second dump is %v, the first is [%d, %d]", again.NE[:2], width, nTok)
	}

	// The scores first: if those moved, nothing below means anything.
	cells, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	cells2, err := re.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range cells.Vals {
		if cells.Vals[i] != cells2.Vals[i] {
			t.Fatalf("the per-cell scores are not reproducible: cell %d of token %d is %g then %g",
				i%nKV, i/nKV, cells.Vals[i], cells2.Vals[i])
		}
	}

	orderDiff, setDiff, visDiff := 0, 0, 0
	for i := 0; i < nTok; i++ {
		a := topk.Ints[i*width : (i+1)*width]
		b := again.Ints[i*width : (i+1)*width]
		for j := range a {
			if a[j] != b[j] {
				orderDiff++
				break
			}
		}
		sa, sb := map[int32]bool{}, map[int32]bool{}
		va, vb := map[int32]bool{}, map[int32]bool{}
		for j := range a {
			sa[a[j]], sb[b[j]] = true, true
			if int(a[j]) <= i {
				va[a[j]] = true
			}
			if int(b[j]) <= i {
				vb[b[j]] = true
			}
		}
		if len(sa) != len(sb) {
			setDiff++
		} else {
			for c := range sa {
				if !sb[c] {
					setDiff++
					break
				}
			}
		}
		for c := range va {
			if !vb[c] {
				visDiff++
			}
		}
		for c := range vb {
			if !va[c] {
				visDiff++
			}
		}
	}
	t.Logf("two runs of the same prompt: scores byte-identical, order differs on %d of %d rows, set on %d, visible cells on %d",
		orderDiff, nTok, setDiff, visDiff)
	if visDiff != 0 {
		t.Errorf("%d visible cells differ between runs; the selection is not reproducible after all", visDiff)
	}
	if orderDiff == 0 {
		t.Errorf("the order is identical across runs, so `indexer_top_k` may be a sort after all — re-read topk_radix_select.comp")
	}
}

// TestQSASelectionIsBlockGranular is the finding vLLM supplied during L3 and
// this dump can finally check: the selection returns **whole blocks**, plus
// the incomplete causal tail, plus whatever the tie fill splits off one block.
//
// It matters because a cell-level top-k that split blocks freely would pass
// every test writable at 7 tokens and be a different model. Here it is read
// straight off llama.cpp's own output: for every token past the point where
// the selection bites, the selected-and-visible cells are `(i+1) mod ratio`
// tail cells, exactly 512 = top_k/ratio whole blocks, and a remainder of
// `(width - tail) mod ratio` cells from one further block — which is the
// lowest-scoring block in the selection.
func TestQSASelectionIsBlockGranular(t *testing.T) {
	_, _, nTok, nKV, width, tr, topk := qsaFixtures4k(t)
	const ratio = 4
	score, err := tr.Get("indexer_score-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	nBlocks := int(score.NE[0])
	if nBlocks != nKV/ratio {
		t.Fatalf("%d blocks over %d cells at ratio %d", nBlocks, nKV, ratio)
	}

	partials := map[int]int{}
	for i := width; i < nTok; i++ {
		count := map[int]int{}
		for _, cell := range topk.Ints[i*width : (i+1)*width] {
			if int(cell) <= i {
				count[int(cell)/ratio]++
			}
		}
		tailStart := (i + 1) / ratio * ratio
		tailBlk := tailStart / ratio
		tail := i + 1 - tailStart

		whole, partial, partialBlk := 0, 0, -1
		for b, n := range count {
			switch {
			case b == tailBlk && tail > 0:
				if n != tail {
					t.Fatalf("token %d: the tail block %d contributes %d cells, want %d", i, b, n, tail)
				}
			case n == ratio:
				whole++
			default:
				partial += n
				partialBlk = b
			}
		}
		if want := (width - tail) / ratio; whole != want {
			t.Fatalf("token %d: %d whole blocks, want %d (tail %d)", i, whole, want, tail)
		}
		if want := (width - tail) % ratio; partial != want {
			t.Fatalf("token %d: %d cells split off a block, want %d (tail %d)", i, partial, want, tail)
		}
		partials[partial]++

		// The split block is the cheapest one in the selection: it is the tie
		// group the radix select's threshold landed on.
		if partialBlk < 0 {
			continue
		}
		lowest := float32(math.Inf(1))
		for b := range count {
			if b == tailBlk && tail > 0 {
				continue
			}
			if s := score.Vals[i*nBlocks+b]; s < lowest {
				lowest = s
			}
		}
		if s := score.Vals[i*nBlocks+partialBlk]; s != lowest {
			t.Fatalf("token %d: the split block %d scores %g, but the selection holds one at %g",
				i, partialBlk, s, lowest)
		}
	}
	t.Logf("cells split off a block, over tokens %d..%d: %v — always fewer than ratio=%d",
		width, nTok-1, partials, ratio)
	t.Logf("the selection is %d/%d = %d whole blocks plus the tail plus at most %d cells",
		2048, ratio, 2048/ratio, ratio-1)
}

// TestQSAIndexer4k runs our indexer over llama.cpp's own block input at 4096
// tokens and compares every tensor it names — which is the other thing this
// dump is for. L3a-5 found that `ggml_vk_mul_mat` takes the f32 *vector* path
// at up to 8 output columns and the fp16 coopmat GEMM above it, so every
// tolerance in L2 and L3 was measured on a path a real ubatch does not use.
// This is the same block, the same code, 4096 columns instead of 7.
//
// It runs the indexer alone rather than the whole layer: the layer's own
// projections are 139 G multiply-adds over 126 MB of weights a token, which is
// a CPU reference for a 7-token fixture and a coffee break for a 4096-token
// one. The indexer's two weights are 6.6 MB together.
func TestQSAIndexer4k(t *testing.T) {
	c, w, nTok, nKV, _, tr, _ := qsaFixtures4k(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(in.Vals) / c.NEmbd; got != nTok {
		t.Fatalf("hc_mixed-3 is %d tokens, the fixture has %d", got, nTok)
	}
	c.Act = RefQ8
	got := &AttnTrace{}
	got.indexer(c, w, in.Vals, nTok, nKV)

	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		// The tolerances are the 4096-token ones and they are 20-50x the
		// 7-token file's, which is the point of this test rather than a
		// disappointment: the reference is on a different arithmetic here.
		// `indexer_k_raw` is the tightest because it is the only one where
		// the bf16 model is the whole story; everything after it carries the
		// fp16 indexer cache (L2e-2) and then a norm and a rotation on top.
		{"indexer_k_raw-3", got.IdxKRaw, 5e-6},       // 4.45e-07; 9.18e-04 without the bf16 activation
		{"indexer_k_pooled-3", got.IdxKPooled, 5e-5}, // 6.93e-06
		{"indexer_k-3", got.IdxK, 1e-4},              // 2.40e-05
		{"indexer_q-3", got.IdxQ, 1e-4},              // 1.32e-05
		{"indexer_score-3", got.IdxScore, 5e-3},      // 7.93e-04, on values to 149.4 — 5e-6 relative
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-22s at %d tokens  %v", tc.name, nTok, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// The per-cell score carries infinities, so the mask is compared as a
	// mask and only the finite cells are subtracted. The mask has to agree
	// exactly: a cell either exists for this query or it does not.
	want, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.IdxScoreCell) != len(want.Vals) {
		t.Fatalf("indexer_score_tokens-3 is %d values, reference has %d", len(got.IdxScoreCell), len(want.Vals))
	}
	var finite, worst float64
	for i := range want.Vals {
		g, wv := float64(got.IdxScoreCell[i]), float64(want.Vals[i])
		if math.IsInf(wv, -1) != math.IsInf(g, -1) {
			t.Fatalf("cell %d of token %d: got %g, reference %g — the mask disagrees", i%nKV, i/nKV, g, wv)
		}
		if math.IsInf(wv, -1) {
			continue
		}
		finite++
		if d := math.Abs(g - wv); d > worst {
			worst = d
		}
	}
	t.Logf("indexer_score_tokens-3 %.0f finite cells of %d, worst |diff| %.3e", finite, len(want.Vals), worst)
	// The 1e9 "always visible" marker dominates the finite values, so the
	// bound is on the score riding on top of it.
	if worst > 1e-1 {
		t.Errorf("indexer_score_tokens-3: worst finite disagreement %.3e", worst)
	}
}

// TestQSASelectionSurvivesOurScores is the question the gate above does not
// answer: our score is not llama.cpp's to the last bit, and a top-k is a
// *discontinuous* function of its input. Two blocks whose scores differ by
// less than our error can swap across the threshold, and then our attention
// reads a cell theirs does not.
//
// So this runs the whole selection off our own numbers and counts the
// disagreement in cells, which is the only unit that matters: a cell either
// enters the softmax or it does not. The count is logged rather than asserted
// to zero, because it cannot be zero — but it is bounded, and the bound is
// what a later kernel has to stay inside.
func TestQSASelectionSurvivesOurScores(t *testing.T) {
	c, w, nTok, nKV, width, tr, topk := qsaFixtures4k(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	c.Act = RefQ8
	got := &AttnTrace{}
	got.indexer(c, w, in.Vals, nTok, nKV)

	// Only cells the causal mask already allows can matter: everything the
	// selection names beyond token i is masked either way.
	var differing, visible, rowsWithDiff int
	worstRow, worstN := -1, 0
	for i := 0; i < nTok; i++ {
		ours := map[int32]bool{}
		for _, cell := range got.TopK[i*width : (i+1)*width] {
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
	t.Logf("our selection against the reference's, visible cells only: %d of %d differ (%.4f%%), over %d of %d tokens; worst row %d with %d",
		differing, visible, 100*float64(differing)/float64(visible), rowsWithDiff, nTok, worstRow, worstN)

	// The tie fill alone accounts for up to ratio-1 cells a row either way,
	// and the reference's half of it is not reproducible at all — so the
	// floor here is not zero. What would not be acceptable is whole blocks
	// swapping in bulk.
	if frac := float64(differing) / float64(visible); frac > 0.01 {
		t.Errorf("%.2f%% of the selected cells differ, which is more than the tie fill can explain", 100*frac)
	}
}
