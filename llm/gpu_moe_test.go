package llm

import (
	"fmt"
	"testing"
	"time"
)

// L5b: the MoE block on the device, against llama.cpp's own activations at
// 4096 tokens.
//
// The fixture is L4a's 4 k dump, for the reason L5a gave and one more. The
// routing is a **discrete** function of the logits, so a short prompt says
// nothing about whether our top-10 of 512 is the reference's; and the
// *schedule* this kernel is built around only exists at length — at a 512-row
// ubatch L5a-4 measured 274 of 512 experts touched with one taking 95% of the
// tokens, and a kernel that scheduled a uniform split would be correct and
// useless.
//
// One staged layer is 1.57 GB of quantised bank, so these tests share a single
// device fixture per run and the whole block runs at once.

// moeGPU4k stages layer 3's FFN half and uploads llama.cpp's own input for it:
// `hc_mixed-3` written the **second** time in the layer, which is the FFN
// mixer's output and not the attention mixer's.
func moeGPU4k(t *testing.T) (*MoEGPU, *Trace, MoEConfig, MoEWeights, []float32, int, func()) {
	t.Helper()
	c, w, in, nTok, tr := moeFixtures4k(t)
	dev, done := newTestDevice(t)
	start := time.Now()
	g, err := NewMoEGPU(dev, c, nTok, []MoEWeights{w})
	if err != nil {
		done()
		t.Fatal(err)
	}
	if err := g.Upload(in, nTok); err != nil {
		g.Destroy()
		done()
		t.Fatal(err)
	}
	t.Logf("staged in %v: %.2f GB of weights, %.1f MB of arenas, %d tokens, %d experts, %d used",
		time.Since(start).Round(time.Millisecond),
		float64(g.WeightBytes())/1e9, float64(g.ActivationBytes())/1e6,
		nTok, c.NExpert, c.NExpertUsed)
	return g, tr, c, w, in, nTok, func() { g.Destroy(); done() }
}

// TestMoEGPUGraph is the cheap structural guard: nine dispatches, in the order
// the block depends on, against the roughly forty lines llama.cpp's graph
// spends on the same work.
func TestMoEGPUGraph(t *testing.T) {
	g, _, _, _, _, _, done := moeGPU4k(t)
	defer done()
	d, kinds, err := g.graph(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d dispatches: %v", len(d), kinds)
	want := []string{"router", "route", "perm.up", "perm.down", "up", "down", "shexp.up", "shexp.down", "combine"}
	if len(kinds) != len(want) {
		t.Fatalf("the graph is %v, want %v", kinds, want)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Errorf("dispatch %d is %q, want %q", i, kinds[i], want[i])
		}
	}
	up, down, boundUp, boundDown := g.Tiles()
	t.Logf("schedule: %d tile records against a bound of %d (up), %d against %d (down)",
		up, boundUp, down, boundDown)
}

// TestMoEGPURouting is the half that can be checked on every token, and the
// half where being wrong is not a tolerance but a different set of experts.
//
// The one deviation from the reference is here and is measured rather than
// waved at: `shared_expert_gate` is a 513th column of the router's matrix, so
// it runs on the fp16 matrix cores where the reference — one output column,
// below L3a-5's 8-column threshold at any prompt length — runs it in f32.
func TestMoEGPURouting(t *testing.T) {
	g, tr, c, _, _, nTok, done := moeGPU4k(t)
	defer done()
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

	nE, used := c.NExpert, c.NExpertUsed
	logits := g.Logits()
	gotLogits := make([]float32, nTok*nE)
	gotGate := make([]float32, nTok)
	for i := 0; i < nTok; i++ {
		copy(gotLogits[i*nE:(i+1)*nE], logits[i*(nE+1):i*(nE+1)+nE])
		gotGate[i] = logits[i*(nE+1)+nE]
	}
	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		// fp16 operands, f32 accumulator — L5a-1's numeric, and the CPU
		// reference lands at 3.6e-05 on the same tensor.
		{"ffn_moe_logits-3", gotLogits, 5e-4},
		// The deviation. The reference computes this one in f32.
		{"shared_expert_gate-3", gotGate, 5e-3},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals[:len(tc.got)])
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-24s %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// The selection, as a set. A top-k is discontinuous, so an rms on the
	// weights would hide a swapped expert entirely: the tenth and eleventh
	// probabilities differ by very little and the weight attached to either
	// is almost the same number.
	want, err := tr.Get("ffn_moe_topk-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if want.Ints == nil {
		t.Fatalf("ffn_moe_topk-3 is ggml type %d, want an integer tensor", want.GType)
	}
	got := g.TopK()
	var setDiff, seqDiff, rowsSet int
	for i := 0; i < nTok; i++ {
		ref := map[int32]bool{}
		for k := 0; k < used; k++ {
			ref[want.Ints[i*used+k]] = true
		}
		bad := 0
		for k := 0; k < used; k++ {
			if !ref[got[i*used+k]] {
				bad++
				setDiff++
			}
			if got[i*used+k] != want.Ints[i*used+k] {
				seqDiff++
			}
		}
		if bad > 0 {
			rowsSet++
		}
	}
	t.Logf("selection over %d tokens: %d of %d slots differ as a set (%d rows), %d differ in order",
		nTok, setDiff, nTok*used, rowsSet, seqDiff)
	if setDiff != 0 {
		t.Errorf("%d slots of %d name an expert the reference did not", setDiff, nTok*used)
	}

	// And the normalised weights, which are only meaningful once the set is.
	wn, err := tr.Get("ffn_moe_weights_norm-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	gotW := g.Weights()
	flat := make([]float32, nTok*used)
	for i := 0; i < nTok; i++ {
		copy(flat[i*used:(i+1)*used], gotW[i*(used+1):i*(used+1)+used])
	}
	r, err := compare(flat, wn.Vals[:len(flat)])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%-24s %v", "ffn_moe_weights_norm-3", r)
	if r.rms > 1e-4 {
		t.Errorf("ffn_moe_weights_norm-3: rms %.3e (%v)", r.rms, r)
	}
}

// TestMoEGPUPermutation is the gate on the schedule, and it is exact.
//
// Nothing downstream can be right if the counting sort is wrong, and nothing
// about the counting sort is a tolerance: it is a permutation of
// `tokens * used` entries, every entry has to land in the range of the expert
// that chose it, and the histogram has to be the routing's. The last of those
// is also L5a-4's distribution re-measured on the device, which is the number
// the kernel's schedule was designed against.
func TestMoEGPUPermutation(t *testing.T) {
	g, _, c, _, _, nTok, done := moeGPU4k(t)
	defer done()
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	used, slots := c.NExpertUsed, c.NExpertUsed+1

	// The histogram, against the selection the device itself emitted.
	topk := g.TopK()
	want := make([]uint32, c.NExpert)
	for _, e := range topk {
		want[e]++
	}
	counts := g.Counts()
	touched, maxBucket, maxAt := 0, uint32(0), 0
	for e := range counts {
		if counts[e] != want[e] {
			t.Fatalf("expert %d has %d rows, the selection gives it %d", e, counts[e], want[e])
		}
		if counts[e] > 0 {
			touched++
		}
		if counts[e] > maxBucket {
			maxBucket, maxAt = counts[e], e
		}
	}
	t.Logf("routing at %d tokens: %d of %d experts touched, widest bucket expert %d with %d rows (%.0f%% of the ubatch)",
		nTok, touched, c.NExpert, maxAt, maxBucket, 100*float64(maxBucket)/float64(nTok))

	// The permuted row space: the shared expert's group at the front, then
	// every routed expert's range padded up to the schedule's alignment. Each
	// expert's real rows must be a bijection onto the (token, slot) pairs that
	// chose it, and every padding row must be the sentinel — a token one past
	// the batch, whose activation row is zero. Nothing here is a tolerance.
	perm := g.Perm()
	offsets := g.Offsets()
	sentinel := uint32(nTok * slots)
	if got := len(perm); got != g.Rows() {
		t.Fatalf("permutation is %d entries, want %d", got, g.Rows())
	}
	for i := 0; i < nTok; i++ {
		if perm[i] != uint32(i*slots+used) {
			t.Fatalf("shared row %d is %d, want %d", i, perm[i], i*slots+used)
		}
	}
	for i := nTok; i < g.Reserve(); i++ {
		if perm[i] != sentinel {
			t.Fatalf("the shared group's padding row %d is %d, want the sentinel %d", i, perm[i], sentinel)
		}
	}
	seen := make([]bool, nTok*slots)
	var padRows int
	for e := 0; e < c.NExpert; e++ {
		off := int(offsets[e])
		cnt := int(counts[e])
		for r := off; r < off+cnt; r++ {
			p := int(perm[r])
			if p%slots >= used {
				t.Fatalf("routed row %d is %d, which is not a (token, routed slot)", r, p)
			}
			if seen[p] {
				t.Fatalf("permutation entry %d repeats (token %d, slot %d)", r, p/slots, p%slots)
			}
			seen[p] = true
			if got := int(topk[(p/slots)*used+p%slots]); got != e {
				t.Fatalf("row %d sits in expert %d's range but token %d slot %d chose expert %d",
					r, e, p/slots, p%slots, got)
			}
		}
		// The slack up to the alignment.
		end := off
		if cnt > 0 {
			end = off + ((cnt + 15) / 16 * 16)
		}
		for r := off + cnt; r < end && r < len(perm); r++ {
			if perm[r] != sentinel {
				t.Fatalf("expert %d's padding row %d is %d, want the sentinel %d", e, r, perm[r], sentinel)
			}
			padRows++
		}
	}
	n := 0
	for _, s := range seen {
		if s {
			n++
		}
	}
	if n != nTok*used {
		t.Fatalf("the buckets hold %d rows, the selection has %d", n, nTok*used)
	}
	t.Logf("the permutation is a bijection over all %d (token, slot) pairs, with %d shared rows in front and every expert's slack filled with the sentinel",
		n, nTok)

	// And the inverse, which is what the combine reads.
	inv := g.InvPerm()
	for tk := 0; tk < nTok*slots; tk++ {
		if tk%slots == used {
			if int(inv[tk]) != tk/slots {
				t.Fatalf("the shared slot of token %d inverts to row %d, want %d", tk/slots, inv[tk], tk/slots)
			}
			continue
		}
		if int(perm[inv[tk]]) != tk {
			t.Fatalf("(token %d, slot %d) inverts to row %d, which names %d",
				tk/slots, tk%slots, inv[tk], perm[inv[tk]])
		}
	}
	t.Logf("the inverse permutation round-trips on all %d slots", nTok*slots)
}

// TestMoEGPUBlock is the gate: every tensor of both halves, at 4096 tokens,
// on the device.
//
// The tolerances are L4a-5's and not L2's. Every matmul here is a quantised
// weight against an activation the reference quantises to int8, and at 4096
// columns it accumulates all of them in **fp16** — so ~3e-03 rms on a tensor
// whose own rms is order 1 is the oracle's precision, not ours, and L5a-3
// measured that modelling its arithmetic makes the fit *worse* by 1.4-1.8x.
// The f32 accumulator this kernel uses is the nearer of the two to the model.
func TestMoEGPUBlock(t *testing.T) {
	g, tr, c, _, _, nTok, done := moeGPU4k(t)
	defer done()
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	used, ff, embd := c.NExpertUsed, c.FFNExpert, c.NEmbd

	swiglu := g.Unpermute(g.Swiglu(), ff)
	weighted := g.Weighted()
	moeW := make([]float32, nTok*used*embd)
	gated := make([]float32, nTok*embd)
	for i := 0; i < nTok; i++ {
		copy(moeW[i*used*embd:], weighted[i*(used+1)*embd:i*(used+1)*embd+used*embd])
		copy(gated[i*embd:], weighted[(i*(used+1)+used)*embd:(i*(used+1)+used+1)*embd])
	}

	for _, tc := range []struct {
		name  string
		got   []float32
		width int
		tol   float64
	}{
		{"ffn_moe_swiglu-3", swiglu, used * ff, 2e-2},
		{"ffn_moe_weighted-3", moeW, used * embd, 5e-3},
		{"ffn_swiglu-3", g.ShSwiglu(), ff, 2e-2},
		{"ffn_shexp_gated-3", gated, embd, 2e-2},
		{"ffn_out-3", g.Out(), embd, 1e-2},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals[:len(tc.got)])
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-24s over %d tokens  %v", tc.name, nTok, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestMoEGPULadderAgrees is the ladder's correctness half: the row block a
// rung compiles in changes how many tiles the schedule holds, how much of
// each is padding and — for the up projection — how many scattered rows one
// workgroup gathers, but it must not change a number.
//
// It is also the guard on the one structural thing that could silently break:
// the two modes are cut to **different** tile lists, and a tile list is only
// consistent with the permutation that outlives it. Running both permutation
// passes before either GEMM is what makes that true, and a plan whose rungs
// differ is the only thing that tests it.
func TestMoEGPULadderAgrees(t *testing.T) {
	g, _, _, _, _, nTok, done := moeGPU4k(t)
	defer done()

	ref := map[string][]float32{}
	for _, plan := range [][2]MoEKernel{
		{MoEM2, MoEM2}, {MoEM1, MoEM4}, {MoEM4, MoEM1}, {MoEM1, MoEM1}, {MoEM4, MoEM4},
		{MoEW4M1, MoEW4M1}, {MoEW2M1, MoEM1}, {MoEM2, MoEW4M1},
	} {
		if err := g.SetPlan(plan[0], plan[1]); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		out := g.Out()
		up, down, _, _ := g.Tiles()
		name := fmt.Sprintf("%s/%s", plan[0], plan[1])
		if ref["out"] == nil {
			ref["out"] = out
			t.Logf("%-8s %d up tiles, %d down tiles — the reference run", name, up, down)
			continue
		}
		r, err := compare(out, ref["out"])
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		t.Logf("%-8s %d up tiles, %d down tiles, against m2/m2: %v", name, up, down, r)
		// The rungs execute the same arithmetic in the same order per row —
		// only the tiling changes — so this is exact, and anything else means
		// a tile read a row that was not its own.
		if r.maxAbs != 0 {
			t.Errorf("%s disagrees with m2/m2 by %.3e at %d over %d tokens",
				name, r.maxAbs, r.at, nTok)
		}
	}
}

// TestMoEGPUPaddingIsInert is the negative control on the choice the whole
// epilogue rests on.
//
// A cooperative-matrix store covers sixteen rows and cannot be masked, so
// rather than staging the accumulators in LDS and copying them out under a
// bound — 8-16 KB on a kernel that is short of occupancy, not of LDS — the
// permutation pads every expert's row range up to the schedule's alignment
// and fills the slack with a sentinel naming a token one past the batch. The
// bet is that those rows compute something nothing reads.
//
// Nothing in a tensor comparison tests that bet: the padding rows are
// invisible, and a kernel whose padding overlapped a *real* row would be
// wrong only where the overlap happened to matter. So this poisons the
// sentinel's activation row with a large value and demands the block's output
// not move by a single bit. It is the same shape of control as
// `TestDeltaNetGPUWindow`'s zeroing: the thing being tested is a layout, and a
// layout that is load-bearing has to be shown to be.
func TestMoEGPUPaddingIsInert(t *testing.T) {
	g, _, _, _, _, nTok, done := moeGPU4k(t)
	defer done()
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	clean := g.Out()

	// Every rung, because the alignment moves with the plan and so does how
	// many padding rows there are.
	for _, plan := range [][2]MoEKernel{{MoEM1, MoEM1}, {MoEM2, MoEM2}, {MoEM4, MoEM4}, {MoEW4M1, MoEW4M1}} {
		if err := g.SetPlan(plan[0], plan[1]); err != nil {
			t.Fatal(err)
		}
		g.poisonPad(1000)
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		_, exec, real := g.Schedule(plan[0])
		r, err := compare(g.Out(), clean)
		if err != nil {
			t.Fatalf("%s: %v", plan[0], err)
		}
		t.Logf("%s: %d padded rows over %d real (%.2fx), sentinel row poisoned with 1000: %v",
			plan[0], exec, real, float64(exec)/float64(real), r)
		if r.maxAbs != 0 {
			t.Errorf("%s: poisoning the padding moved %d tokens' output by up to %.3e",
				plan[0], nTok, r.maxAbs)
		}
	}
}
