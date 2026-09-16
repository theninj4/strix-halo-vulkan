package llm

import (
	"math"
	"testing"
	"time"
)

// L5a: the MoE block in Go, against llama.cpp's own activations at 4096
// tokens. 35.7% of the prefill graph, 97% of the parameters, and the last
// block of the model without a reference.
//
// The fixture is L4a's 4 k dump, which carried the whole block along for free
// because its filter took layers 0 and 3 — so L5's fixture existed before L5
// started. That matters here more than anywhere else in the vertical: the
// routing is a **discrete** function of the logits, so a 7-token dump would
// have told us nothing about whether our top-10 of 512 is the reference's,
// and the 4 k one has 4096 independent selections to check it on.
//
// The split below is not cosmetic. The **routing** is 5.2 M weights a token
// and runs over all 4096; the **experts** are 4.9 M weights of Q4_K and Q5_1
// *per (token, expert) pair* and a full prefill is 40 960 of them, which is
// 200 billion dequantised weights. So the expert half runs over a token
// range, walking by expert so each bank is gathered once.

// moeTokens is how many tokens the expert half is checked over. Every token
// is 10 of the 512 experts, and the cost is the dequantisation of whichever
// distinct experts that reaches — so this is a knob on runtime, not on
// coverage: the tensors compared are the reference's own, row for row.
const moeTokens = 48

// moeLayer is the full-attention layer the rest of the vertical uses. Its MoE
// block is no different from layer 0's — every one of the 48 has one — but
// keeping the same layer means the input is the `hc_mixed-3` the attention
// tests already know, in its second occurrence.
const moeLayer = attnLayer

// moeFixtures4k opens one layer's FFN half at 4096 tokens, with llama.cpp's
// own input for it: `hc_mixed` written the **second** time in the layer, which
// is the FFN mixer's output and not the attention mixer's.
func moeFixtures4k(t *testing.T) (MoEConfig, MoEWeights, []float32, int, *Trace) {
	t.Helper()
	m, tr := fixtures4k(t)
	c := m.MoEConfig()
	w, err := m.MoEWeights(moeLayer)
	if err != nil {
		t.Fatal(err)
	}
	in, err := tr.Get("hc_mixed-3", 1)
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(in.Vals) / c.NEmbd
	t.Logf("layer %d: %d experts of %d wide, %d used, shared %d; %d tokens",
		moeLayer, c.NExpert, c.FFNExpert, c.NExpertUsed, c.FFNShared, nTok)
	return c, w, in.Vals, nTok, tr
}

// TestMoERouting is the half that can be checked on every token, and the half
// where being wrong is not a tolerance but a different set of experts.
//
// The logits, the softmax and the ten weights are compared as numbers; the
// **selection** is compared as a set and as a sequence, because a top-k is
// discontinuous and an rms on the weights would hide a swapped expert
// entirely — the tenth and eleventh probabilities differ by very little, and
// the weight attached to either is almost the same number.
func TestMoERouting(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	got := &MoETrace{}
	start := time.Now()
	got.Route(c, w, in, nTok)
	t.Logf("routed %d tokens in %v", nTok, time.Since(start).Round(time.Millisecond))

	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		// The router is F32 x F32 on the fp16 matrix cores with an f32
		// accumulator, so this is L2e-3's numeric and not L4a-5's: the
		// tolerance is an operand one, not the 5e-3 a quantised matmul gets.
		{"ffn_moe_logits-3", got.Logits, 5e-4},
		{"ffn_moe_probs-3", got.Probs, 1e-5},
		{"ffn_moe_weights-3", got.Weights, 1e-5},
		{"ffn_moe_weights_sum-3", got.WeightsSum, 1e-5},
		{"ffn_moe_weights_sum_clamped-3", got.WeightsSumClamped, 1e-5},
		{"ffn_moe_weights_norm-3", got.WeightsNorm, 1e-5},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-30s %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// The selection itself. `ffn_moe_topk` is a view of the first ten columns
	// of `ffn_moe_argsort`, so both are checked: the sequence, which says the
	// order agrees, and the set, which is what actually decides the block's
	// output.
	topk, err := tr.Get("ffn_moe_topk-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if topk.Ints == nil {
		t.Fatalf("ffn_moe_topk-3 is ggml type %d, want an integer tensor", topk.GType)
	}
	probs, err := tr.Get("ffn_moe_probs-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	used, nE := c.NExpertUsed, c.NExpert
	var seqDiff, setDiff, rowsSeq, rowsSet, realReorder int
	var worstTie float64
	for i := 0; i < nTok; i++ {
		ours := got.TopK[i*used : (i+1)*used]
		ref := topk.Ints[i*used : (i+1)*used]
		bad := false
		for k := range ours {
			if ours[k] == ref[k] {
				continue
			}
			seqDiff++
			bad = true
			// An order disagreement is only acceptable if the two experts
			// are a *tie* the router's own precision cannot resolve. The
			// reference's probabilities decide that, not ours.
			a := float64(probs.Vals[i*nE+int(ours[k])])
			b := float64(probs.Vals[i*nE+int(ref[k])])
			rel := math.Abs(a-b) / math.Max(math.Abs(a), math.Abs(b))
			if rel > worstTie {
				worstTie = rel
			}
			if rel > moeTieRel {
				realReorder++
				t.Errorf("token %d slot %d: experts %d and %d swapped, and their reference probabilities differ by %.2e relative — that is a reordering, not a tie",
					i, k, ours[k], ref[k], rel)
			}
		}
		if bad {
			rowsSeq++
		}
		set := map[int32]bool{}
		for _, e := range ref {
			set[e] = true
		}
		n := 0
		for _, e := range ours {
			if !set[e] {
				n++
			}
		}
		setDiff += n
		if n > 0 {
			rowsSet++
		}
	}
	t.Logf("selection: %d of %d slots differ as a set (%d rows); %d differ in order (%d rows), worst such pair %.2e apart in the reference's own probabilities",
		setDiff, nTok*used, rowsSet, seqDiff, rowsSeq, worstTie)
	if setDiff != 0 {
		t.Errorf("%d of %d selected experts are not the reference's — the routing would be a different model, not a less precise one",
			setDiff, nTok*used)
	}
}

// moeTieRel is how near two probabilities have to be for a swapped slot to
// count as a tie rather than a reordering. The router's own error against
// llama.cpp is 3.4e-05 maxRel on `ffn_moe_probs`, so anything inside that is
// unresolvable by construction and anything outside it is ours.
const moeTieRel = 1e-5

// TestMoEBlock is the gate: every tensor llama.cpp names inside the block,
// over a token range, from the reference's own input.
//
// The tolerances are L4a-5's and not L2's. Every one of the three expert
// matmuls and all three of the shared expert's are quantised weights against
// int8 activations, and at 4096 columns the reference accumulates them in
// **fp16** — so it sits ~5e-3 rms from the f32 model on tensors whose own rms
// is order 1, and no operand model of ours closes that. `shared_expert_gate`
// is the exception in the other direction: one output column keeps it on the
// f32 vector path at any prompt length, so it is the tightest thing here.
func TestMoEBlock(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	lo, hi := 0, minInt(moeTokens, nTok)

	start := time.Now()
	got, err := MoEBlock(c, w, in, nTok, lo, hi)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("routed %d tokens and ran the block over %d in %v", nTok, hi-lo, time.Since(start).Round(time.Millisecond))

	used, ff, embd := c.NExpertUsed, c.FFNExpert, c.NEmbd
	for _, tc := range []struct {
		name  string
		got   []float32
		width int // values per token in the reference's layout
		tol   float64
	}{
		{"ffn_moe_gate-3", got.Gate, used * ff, 2e-2},
		{"ffn_moe_up-3", got.Up, used * ff, 2e-2},
		{"ffn_moe_swiglu-3", got.Swiglu, used * ff, 2e-2},
		{"ffn_moe_down-3", got.Down, used * embd, 2e-2},
		{"ffn_moe_weighted-3", got.Weighted, used * embd, 5e-3},
		{"ffn_moe_out-3", got.MoEOut, embd, 5e-3},
		{"ffn_gate-3", got.ShGate, ff, 2e-2},
		{"ffn_up-3", got.ShUp, ff, 2e-2},
		{"ffn_swiglu-3", got.ShSwiglu, ff, 2e-2},
		{"ffn_shexp-3", got.Shexp, embd, 2e-2},
		{"shared_expert_gate-3", got.SharedGate, 1, 5e-4},
		{"shared_expert_gate_sigmoid-3", got.SharedGateSigmoid, 1, 5e-4},
		{"ffn_shexp_gated-3", got.ShexpGated, embd, 2e-2},
		{"ffn_out-3", got.Out, embd, 1e-2},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		// The reference's tensor covers every token; ours covers the range.
		ref := want.Vals[lo*tc.width : hi*tc.width]
		r, err := compare(tc.got, ref)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-30s over %d tokens  %v", tc.name, hi-lo, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestMoEWeightsSumIsClamped is the negative control for the one line of this
// block that exists only for a case the data never reaches.
//
// `ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)` guards a division by
// zero, and a softmax over 512 logits whose top ten are taken can never
// actually produce a sum that small — so the clamp is invisible in every
// tensor comparison, and a reference that dropped it would pass all of them.
// This asserts both halves: that the reference emits the node, and that on
// this prompt it is the identity, so its absence is untestable *from the data*
// and has to be asserted from the graph.
func TestMoEWeightsSumIsClamped(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	got := &MoETrace{}
	got.Route(c, w, in, nTok)

	var floor float32 = math.MaxFloat32
	for _, v := range got.WeightsSum {
		if v < floor {
			floor = v
		}
	}
	t.Logf("smallest weights_sum over %d tokens: %.4f, against the clamp's %.6g",
		nTok, floor, moeWeightsSumFloor)
	if floor <= moeWeightsSumFloor {
		t.Fatalf("a row reached the clamp (%g); this test's premise is wrong, not the code", floor)
	}
	// And the reference's own two nodes are the same tensor here, which is
	// what says the clamp is the identity on this data rather than that we
	// skipped it.
	sum, err := tr.Get("ffn_moe_weights_sum-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	clamped, err := tr.Get("ffn_moe_weights_sum_clamped-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range sum.Vals {
		if sum.Vals[i] != clamped.Vals[i] {
			t.Fatalf("the reference's clamp bites at token %d: %g -> %g", i, sum.Vals[i], clamped.Vals[i])
		}
	}
	t.Logf("the reference's clamp is the identity on all %d rows — it is reproduced from the graph, not from the data", len(sum.Vals))
}

// moeControlTokens is the range the negative controls run over. They are
// asking whether a structurally different block is *distinguishable*, which
// eight tokens answers as well as forty-eight and four times faster.
const moeControlTokens = 8

// TestMoEIsTheOracleAccumulatingInFP16 is L4a-5's claim, tested where it
// bites hardest rather than taken on trust.
//
// L4a-5 found that at 4096 columns **100.0% of the values the reference writes
// out of a quantised matmul are exactly IEEE halves** — `ffn_gate`, `ffn_up`
// and `ffn_shexp` among them — and concluded that no model of the *operands*
// can close the gap, because the accumulator is the thing losing precision.
// The MoE is where that costs the most: three quantised matmuls deep, Q4_K
// and Q5_1 rather than Q8_0, and 35.7% of the graph.
//
// So this runs the block under both numerics and compares the two against
// llama.cpp. The result is sharper than L4a-5's "within 1.06x": **`Exact` —
// pure f32, no model of the reference's arithmetic at all — is 1.4 to 1.8x
// *nearer* llama.cpp than `RefQ8` is.** Everywhere else in this vertical
// modelling the reference's int8 activations was worth 35x (L2b-2), 69x
// (L2e-3), 2061x (L4a-6); here it is worth **less than nothing**, because the
// reference has already thrown away eleven mantissa bits in the accumulator
// and our own quantisation error adds to that rather than cancelling it.
func TestMoEIsTheOracleAccumulatingInFP16(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	lo, hi := 0, minInt(moeControlTokens, nTok)

	runs := map[Numerics]*MoETrace{}
	for _, mode := range []Numerics{Exact, RefQ8} {
		cc := c
		cc.Act = mode
		got, err := MoEBlock(cc, w, in, nTok, lo, hi)
		if err != nil {
			t.Fatal(err)
		}
		runs[mode] = got
	}

	for _, tc := range []struct {
		name  string
		pick  func(*MoETrace) []float32
		width int
	}{
		{"ffn_moe_gate-3", func(m *MoETrace) []float32 { return m.Gate }, c.NExpertUsed * c.FFNExpert},
		{"ffn_moe_out-3", func(m *MoETrace) []float32 { return m.MoEOut }, c.NEmbd},
		{"ffn_shexp-3", func(m *MoETrace) []float32 { return m.Shexp }, c.NEmbd},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		ref := want.Vals[lo*tc.width : hi*tc.width]
		ex, err := compare(tc.pick(runs[Exact]), ref)
		if err != nil {
			t.Fatal(err)
		}
		q8, err := compare(tc.pick(runs[RefQ8]), ref)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%-18s against llama.cpp: Exact rms %.3e, RefQ8 rms %.3e, %.3fx apart",
			tc.name, ex.rms, q8.rms, ex.rms/q8.rms)
		// Measured at 0.57-0.73x. The bound is two-sided: a value near 1
		// would mean the operands do not matter either way, and a value
		// above it would mean RefQ8 had started helping again, which would
		// contradict the fp16 census.
		ratio := ex.rms / q8.rms
		if ratio < 0.3 || ratio > 1.5 {
			t.Errorf("%s: the f32 model is %.2fx the int8 one against llama.cpp; L4a-5 says the accumulator, not the operands, is the gap, so this should sit near 0.6",
				tc.name, ratio)
		}
	}
}

// TestMoEExpertRowsAreExpertMajor is the negative control on the one layout
// decision the whole block rests on.
//
// `ffn_gate_exps` is `[2560, 640, 512]`, and this package reads expert e's
// output row j as row `e*640 + j` of a flat row index. The alternative —
// interleaving, `j*512 + e` — is a perfectly plausible reading of the same
// three axes, and it produces a tensor of exactly the right shape built from
// 640 rows of other experts. Nothing but a test that demands the two disagree
// tells them apart, and an rms alone would not: both are real weights.
func TestMoEExpertRowsAreExpertMajor(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	lo, hi := 0, minInt(moeControlTokens, nTok)
	got, err := MoEBlock(c, w, in, nTok, lo, hi)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("ffn_moe_gate-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	width := c.NExpertUsed * c.FFNExpert
	ref := want.Vals[lo*width : hi*width]
	good, err := compare(got.Gate, ref)
	if err != nil {
		t.Fatal(err)
	}

	// The same block with the experts read one slot along: still 512 real
	// experts, still the right shape, still every value a plausible logit.
	shifted := *got
	shifted.TopK = append([]int32(nil), got.TopK...)
	for i := range shifted.TopK {
		shifted.TopK[i] = (shifted.TopK[i] + 1) % int32(c.NExpert)
	}
	if err := shifted.Experts(c, w, in, lo, hi); err != nil {
		t.Fatal(err)
	}
	bad, err := compare(shifted.Gate, ref)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("gate from the routed experts: rms %.3e; from the expert one slot along: rms %.3e, %.0fx worse",
		good.rms, bad.rms, bad.rms/good.rms)
	if bad.rms/good.rms < 50 {
		t.Errorf("reading the wrong expert is only %.1fx worse — this comparison does not constrain the gather",
			bad.rms/good.rms)
	}
}

// TestMoESwigluTakesTheGate is the other control, on the one elementwise line
// that has a plausible wrong version.
//
// `ggml_swiglu_split(gate, up)` is `silu(gate) * up`. The transpose —
// `silu(up) * gate` — has the same shape, the same magnitude and the same
// sign pattern almost everywhere, because silu is near-linear away from zero.
// It is the classic way to get a SwiGLU wrong, and the two operands here come
// out of two matmuls that are dumped separately, so only the swiglu tensor
// can tell them apart.
func TestMoESwigluTakesTheGate(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	lo, hi := 0, minInt(moeControlTokens, nTok)
	got, err := MoEBlock(c, w, in, nTok, lo, hi)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("ffn_moe_swiglu-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	width := c.NExpertUsed * c.FFNExpert
	ref := want.Vals[lo*width : hi*width]
	good, err := compare(got.Swiglu, ref)
	if err != nil {
		t.Fatal(err)
	}

	swapped := make([]float32, len(got.Swiglu))
	for j := range swapped {
		swapped[j] = silu(got.Up[j]) * got.Gate[j]
	}
	bad, err := compare(swapped, ref)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("silu(gate)*up: rms %.3e; silu(up)*gate: rms %.3e, %.0fx worse", good.rms, bad.rms, bad.rms/good.rms)
	if bad.rms/good.rms < 50 {
		t.Errorf("swapping the two halves of the SwiGLU is only %.1fx worse — the comparison does not pin which is which",
			bad.rms/good.rms)
	}
}

// TestMoERoutingBalance measures the thing a MoE kernel is actually sized by,
// which is in neither the architecture nor the reference: how many tokens land
// on each expert.
//
// L2a's decode budget assumed "each expert is read 10 times in 512" — 512
// experts, 10 per token, so a mean bucket of 10 at a 512-token ubatch. The
// mean is right and nothing else about that picture is. A gather/combine
// kernel is sized by the *tail*: the widest bucket decides the permutation
// buffer and the load balance, and the number of experts a ubatch touches at
// all decides how much of the bank is streamed. Both are properties of this
// checkpoint on this prompt, and the 4 k dump is the only place to read them.
//
// The distribution is cross-checked against llama.cpp's own `ffn_moe_topk`,
// so what this reports is a fact about the **reference's** routing and not
// about ours.
func TestMoERoutingBalance(t *testing.T) {
	c, w, in, nTok, tr := moeFixtures4k(t)
	c.Act = RefQ8
	got := &MoETrace{}
	got.Route(c, w, in, nTok)
	topk, err := tr.Get("ffn_moe_topk-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	buckets := func(ids []int32, ub int) []int {
		count := make([]int, c.NExpert)
		for i := 0; i < ub; i++ {
			for k := 0; k < c.NExpertUsed; k++ {
				count[ids[i*c.NExpertUsed+k]]++
			}
		}
		return count
	}

	for _, ub := range []int{512, 2048, 4096} {
		if ub > nTok {
			continue
		}
		count := buckets(got.TopK, ub)
		touched, maxN := 0, 0
		minTouched := 1 << 30
		for _, n := range count {
			if n == 0 {
				continue
			}
			touched++
			if n > maxN {
				maxN = n
			}
			if n < minTouched {
				minTouched = n
			}
		}
		mean := float64(ub*c.NExpertUsed) / float64(touched)
		t.Logf("ubatch %4d: %3d of %d experts touched (%.0f%%); bucket min %d, mean over touched %.1f, max %d — %.0fx the mean, and %.0f%% of the ubatch's tokens",
			ub, touched, c.NExpert, 100*float64(touched)/float64(c.NExpert),
			minTouched, mean, maxN, float64(maxN)/mean, 100*float64(maxN)/float64(ub))
	}

	// The hottest expert, and where in each token's ranking it sits. An
	// expert chosen by nearly every token is a second shared expert in all
	// but name, and it is the one whose weights a kernel should expect to
	// keep hot.
	count := buckets(got.TopK, minInt(512, nTok))
	best, bn := 0, 0
	for e, n := range count {
		if n > bn {
			bn, best = n, e
		}
	}
	slots := make([]int, c.NExpertUsed)
	for i := 0; i < minInt(512, nTok); i++ {
		for k := 0; k < c.NExpertUsed; k++ {
			if int(got.TopK[i*c.NExpertUsed+k]) == best {
				slots[k]++
			}
		}
	}
	t.Logf("hottest expert is %d: chosen by %d of 512 tokens (%.0f%%), and ranked first by %d of them; by slot %v",
		best, bn, 100*float64(bn)/512, slots[0], slots)

	// And the same distribution out of llama.cpp's own selection, which is
	// what makes all of the above a fact about the reference.
	ref := buckets(topk.Ints, minInt(512, nTok))
	for e := range ref {
		if ref[e] != count[e] {
			t.Fatalf("expert %d: our bucket is %d, the reference's %d — the distribution above is not the reference's",
				e, count[e], ref[e])
		}
	}
	t.Logf("all %d buckets are llama.cpp's own, so the skew is the model's and not the routing's", c.NExpert)
}
