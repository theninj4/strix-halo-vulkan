package llm

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// The GPU half of L2e, checked the way L2c and L2d were: against the CPU
// reference in attn.go first, because that is what the kernels are a port of,
// and against llama.cpp's own tensors behind it, because that is what the CPU
// reference was built against.
//
// Layer 3 is the only full-attention layer in the four the trace covers — the
// interval is 4 — and `hc_mixed-3` is llama.cpp's own value for its input, so
// every comparison below is that stage's own and not an accumulation.

// attnGPU stages layer 3 and runs it over the trace's own input.
func attnGPU(t *testing.T) (*AttnGPU, *Trace, AttnConfig, AttnWeights, []float32, int, int, func()) {
	t.Helper()
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	g, err := NewAttnGPU(dev, c, nTok, nKV, []AttnWeights{w}, denseQ8Test)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("staged: %.1f MB of weights, %.1f MB of arenas, %d cells, %d blocks",
		float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6, g.NKV(), g.NBlocks())
	return g, tr, c, w, in.Vals, nTok, nKV, func() { g.Destroy(); done() }
}

// TestAttnGPUPushBlockFits is the cheap structural guard. The push block is a
// contract with four shaders and vk.DispatchMultiTimed records a sequence into
// one command buffer only if every pipeline declares the same range, so the
// one number that has to hold is that the whole vertical's block still fits
// the device's limit — which the attention group nearly doubled.
func TestAttnGPUPushBlockFits(t *testing.T) {
	size := int(unsafe.Sizeof(push{}))
	t.Logf("push block: %d uints, %d bytes", size/4, size)
	if size%4 != 0 {
		t.Errorf("push block is %d bytes, which is not whole uints", size)
	}
	if size > 256 {
		t.Errorf("push block is %d bytes, past the 256 this device reports", size)
	}
}

// TestAttnGPUFusedProjection is the gate on the layout the whole layer rests
// on: six of llama.cpp's matrices packed as column ranges of one.
//
// Nothing downstream can be right if the ranges are wrong, and nothing
// downstream says so clearly — a query read at the key's columns is a
// perfectly plausible tensor. So each range is compared against the tensor
// llama.cpp names, where the trace has one, and against the CPU reference's
// own matvec where it does not.
func TestAttnGPUFusedProjection(t *testing.T) {
	g, tr, c, w, in, nTok, _, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

	// The reference, exactly: fp32 activations against fp32 weights, which is
	// what the kernel computes in fp16 operands with fp32 accumulators.
	proj := func(weight []float32, n int) []float32 {
		out := make([]float32, nTok*n)
		for i := 0; i < nTok; i++ {
			matvec(out[i*n:(i+1)*n], weight, in[i*c.NEmbd:(i+1)*c.NEmbd], n, c.NEmbd)
		}
		return out
	}
	for _, tc := range []struct {
		name  string
		col   int
		width int
		want  []float32
		tol   float64
	}{
		{"attn_q+gate", 0, c.QWidth(), proj(w.Q, c.QWidth()), 2e-3},
		{"attn_k", g.ColK(), c.KVWidth(), proj(w.K, c.KVWidth()), 2e-3},
		{"attn_v", g.ColV(), c.KVWidth(), proj(w.V, c.KVWidth()), 2e-3},
		{"indexer_q_raw", g.ColIQ(), c.IdxHeads * c.IdxDim, proj(w.IdxQ, c.IdxHeads*c.IdxDim), bankTol(2e-3, 5e-3)},
		{"indexer_k_raw", g.ColIK(), c.IdxDim, proj(w.IdxK, c.IdxDim), bankTol(2e-3, 5e-3)},
	} {
		r, err := compare(g.Column(tc.col, tc.width), tc.want)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-14s col %5d  vs CPU  %v", tc.name, tc.col, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// And the one of the six the trace names directly. It is dumped before the
	// fp16 cache write, so this is the projection alone.
	want, err := tr.Get("indexer_k_raw-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Column(g.ColIK(), c.IdxDim), want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%-14s           vs llama.cpp  %v", "indexer_k_raw", r)
	if r.rms > bankTol(2e-3, 5e-3) {
		t.Errorf("indexer_k_raw-3: rms %.3e over %.0e (%v)", r.rms, bankTol(2e-3, 5e-3), r)
	}
}

// TestAttnGPUIndexer is the gate on the QSA indexer: the pooled key, the
// query, the score and the per-cell expansion, all on the device.
//
// The pooled key is where the reference's fp16 cache shows, and this path
// reproduces it rather than modelling it — the cache is fp16 here because the
// arena is, so `float16_t()` on the way out of the projection is the same
// arithmetic and not an approximation of it. The tolerances say so: the pooled
// key lands where `RefQ8` does on the CPU, not where `Exact` does.
func TestAttnGPUIndexer(t *testing.T) {
	// The per-cell score is the reference's `indexer_score_tokens` and since
	// P14-1 a shipped pass does not compute it — the selection reads the block
	// scores with a per-block weight instead. This test is about that tensor,
	// so it runs the arm that still writes it.
	t.Setenv("LLM_ATTN_EXPAND_CELLS", "1")
	g, tr, c, w, in, nTok, nKV, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

	ref := c
	ref.Act = RefQ8
	cpu := AttnLayer(ref, w, in, nTok, nKV)

	for _, tc := range []struct {
		name      string
		got, want []float32
		tol, rtol float64
	}{
		{"indexer_k_pooled-3", g.IdxK(), cpu.IdxK, bankTol(1e-3, 5e-3), bankTol(1e-3, 5e-3)},
		{"indexer_q-3", g.IdxQ(), cpu.IdxQ, bankTol(1e-3, 5e-3), bankTol(1e-3, 5e-3)},
		{"indexer_score-3", g.Score(), cpu.IdxScore, 5e-2, 5e-2},
	} {
		r, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatalf("%s against the CPU reference: %v", tc.name, err)
		}
		t.Logf("%-20s vs CPU RefQ8  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against the CPU reference, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
		// indexer_k_pooled is dumped before the norm and the rotary, so the
		// device tensor — which carries both — is compared to llama.cpp under
		// the name of the tensor it actually is.
		name := tc.name
		if name == "indexer_k_pooled-3" {
			name = "indexer_k-3"
		}
		want, err := tr.Get(name, 0)
		if err != nil {
			t.Fatal(err)
		}
		if r, err = compare(tc.got, want.Vals); err != nil {
			t.Fatalf("%s against llama.cpp: %v", name, err)
		}
		t.Logf("%-20s vs llama.cpp  %v", name, r)
		if r.rms > tc.rtol {
			t.Errorf("%s: rms %.3e against llama.cpp, over %.1e (%v)", name, r.rms, tc.rtol, r)
		}
	}

	// The per-cell score carries infinities — every masked cell is one — so it
	// is compared as a mask plus the finite values, not by subtraction. The
	// mask has to agree exactly: a cell the reference calls invisible and we
	// do not is a cell attention would read.
	want, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	got := g.Cells()
	if len(got) != len(want.Vals) {
		t.Fatalf("indexer_score_tokens-3 is %d values, reference has %d", len(got), len(want.Vals))
	}
	var finite, worst float64
	for i := range want.Vals {
		gv, wv := float64(got[i]), float64(want.Vals[i])
		if math.IsInf(wv, -1) != math.IsInf(gv, -1) {
			t.Fatalf("cell %d of token %d: got %g, reference %g — the mask disagrees",
				i%nKV, i/nKV, gv, wv)
		}
		if math.IsInf(wv, -1) {
			continue
		}
		finite++
		if d := math.Abs(gv - wv); d > worst {
			worst = d
		}
	}
	t.Logf("indexer_score_tokens-3 %.0f finite cells of %d, worst |diff| %.3e", finite, len(got), worst)
	if worst > 1e-1 {
		t.Errorf("indexer_score_tokens-3: worst finite disagreement %.3e", worst)
	}
}

// TestAttnGPUEmptyBlocksPoolCellZero is the negative control on the one part
// of the indexer that cannot be inferred from the graph.
//
// A block exists only if all `ratio` of its cells are occupied, and llama.cpp
// leaves `blk_cells` zero-filled — so at 7 tokens in a 256-cell cache there is
// exactly one real block and 63 that pool cell 0 four times. Their pooled key
// is therefore *token 0's key*, four times over and divided by four, and every
// one of the 63 is identical. A kernel that pooled the cells it found, or
// clamped to the last real one, or skipped the empty blocks, would produce a
// perfectly plausible tensor and fail nothing else — the score it feeds is
// masked to -inf at every one of those blocks.
func TestAttnGPUEmptyBlocksPoolCellZero(t *testing.T) {
	g, _, c, _, in, nTok, _, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	nBid := nTok / c.Ratio
	if nBid >= g.NBlocks() {
		t.Skipf("%d tokens fill all %d blocks; this control needs empty ones", nTok, g.NBlocks())
	}
	k := g.IdxK()
	first := k[nBid*c.IdxDim : (nBid+1)*c.IdxDim]
	for b := nBid + 1; b < g.NBlocks(); b++ {
		row := k[b*c.IdxDim : (b+1)*c.IdxDim]
		for i := range row {
			if row[i] != first[i] {
				t.Fatalf("block %d component %d is %g, block %d has %g — the empty blocks are not all cell 0",
					b, i, row[i], nBid, first[i])
			}
		}
	}
	t.Logf("%d blocks of %d do not exist and all %d are byte-identical",
		g.NBlocks()-nBid, g.NBlocks(), g.NBlocks()-nBid)
}

// TestAttnGPULayer is the L2f gate: the whole layer, against the CPU reference
// and against llama.cpp.
//
// Q, K and V are read back out of the fragment tiling so that a disagreement
// in the output can be located in the norm or the rotary rather than
// attributed to the kernel that consumed them — which is the thing a flash
// kernel makes hardest to see from its output alone.
func TestAttnGPULayer(t *testing.T) {
	g, tr, c, w, in, nTok, nKV, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

	exact := c
	exact.Act = Exact
	cpu := AttnLayer(exact, w, in, nTok, nKV)

	for _, tc := range []struct {
		name      string
		got, want []float32
		tol       float64
	}{
		{"q (normed, rotated, scaled)", g.Q(), scaled(cpu.Q, attnLog2Scale(c)), 2e-3},
		{"k (normed, rotated)", g.K(), cpu.K, 2e-3},
		{"v", g.V(), cpu.V, 2e-3},
		{"attn_gated", g.Context(), cpu.Gated, 2e-3},
		{"attn_output", g.Out(), cpu.Out, 5e-3},
	} {
		r, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-28s vs CPU Exact  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against the CPU reference, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// And against the oracle. These are the tolerances L2e measured for the
	// CPU reference under `RefQ8`, which is the side computing in int8: the
	// device concedes nothing extra, so it is held to the same numbers.
	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		{"attn_gated-3", g.Context(), 5e-4},
		{"attn_output-3", g.Out(), 2e-3},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-28s vs llama.cpp  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against llama.cpp, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}

// TestAttnGPUGateIsFused is the negative control for the kernel's one
// structural claim: that the epilogue applies the *gate* and not merely the
// softmax divide.
//
// The ungated attention output — `attn_pregate` — is a perfectly plausible
// tensor of the right shape and a comparable magnitude, and it is what a
// kernel that forgot the sigmoid would write. Nothing but a test that demands
// the two disagree tells them apart, and the margin is what says the gate is
// really doing something: sigmoid is bounded by 1, so the gated tensor is
// strictly smaller and no tolerance covers the gap.
func TestAttnGPUGateIsFused(t *testing.T) {
	g, tr, _, _, in, nTok, _, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	pre, err := tr.Get("attn_pregate-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Context(), pre.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("context against the *ungated* output: %v", r)
	if r.rms < 1e-2 {
		t.Errorf("the context agrees with attn_pregate to rms %.3e — the gate is not being applied", r.rms)
	}
}

// TestAttnGPUIsNearerTheModelThanTheReference is the counterpart of the
// hyper-connection block's test of the same name, and it is what says the
// remaining gap to llama.cpp is the *reference's* arithmetic and not ours.
//
// The layer has three Q8_0 projections in front of it and one behind, and
// llama.cpp evaluates every one of them over int8 activations (L2b-2). The
// device evaluates them in fp16 operands with fp32 accumulators — which L2e-3
// showed is the same arithmetic the backend's own matrix cores use wherever it
// is not quantising — so against the f32 model the kernel should be well
// inside where the oracle sits. If it ever stops being, the cause is ours.
func TestAttnGPUIsNearerTheModelThanTheReference(t *testing.T) {
	g, tr, c, w, in, nTok, nKV, done := attnGPU(t)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	exact := c
	exact.Act = Exact
	model := AttnLayer(exact, w, in, nTok, nKV)

	for _, tc := range []struct {
		name string
		got  []float32
		want []float32
		min  float64
	}{
		{"attn_gated-3", g.Context(), model.Gated, 5},
		{"attn_output-3", g.Out(), model.Out, 5},
	} {
		ours, err := compare(tc.got, tc.want)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		theirs, err := compare(ref.Vals, tc.want)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%-14s against the f32 model: ours rms %.3e, llama.cpp rms %.3e, %.1fx",
			tc.name, ours.rms, theirs.rms, theirs.rms/ours.rms)
		if theirs.rms/ours.rms < tc.min {
			t.Errorf("%s: the kernel is only %.1fx nearer the model than llama.cpp; fp16 operands should be far better than int8 ones",
				tc.name, theirs.rms/ours.rms)
		}
	}
}

// attnLog2Scale is the factor the pack folds into the query so the attention
// kernel's exponential can be exp2: 1/sqrt(headDim) * log2(e). The CPU
// reference carries it on the score instead, which is why comparing the two
// queries needs it here.
func attnLog2Scale(c AttnConfig) float32 {
	return float32(math.Log2(math.E) / math.Sqrt(float64(c.HeadDim)))
}

func scaled(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// TestAttnGPUGemvAgrees is L8e-1's gate: llm_gemv.comp against llm_gemm.comp
// MODE 2 on both of this layer's projections, at the one token the GEMV
// exists for. It is TestDeltaNetGPUGemvAgrees on the other block, and the two
// claims are the same two.
//
// The **fused projection** has a tail — the indexer's q_proj and k_proj are
// the model's only BF16 weights and stay halves at `gateOff`, numbered from
// `lowRank/16` — so this is also the check that the GEMV derives the same
// split the GEMM does, and a column read out of the wrong plane is not a
// tolerance but somebody else's indexer query. The **output projection** has
// none: `attn_output` is Q8_0 all the way through.
//
// It is a tolerance and not an equality at every rung, including KSLABS = 1.
// The *weights* are the same halves — L8b-2's argument, and the reason this
// kernel multiplies in fp16 rather than converting — but a cooperative-matrix
// accumulator sums sixteen k an instruction in an order the extension does not
// define where a lane sums them serially, so the two differ by f32 round-off
// over a 2560- or 6144-long chain. What makes that a measurement rather than
// an alibi is that it does not move with the rung.
func TestAttnGPUGemvAgrees(t *testing.T) {
	g, _, c, _, in, nTok, _, done := attnGPU(t)
	defer done()
	if nTok < 1 {
		t.Skip("the trace has no tokens")
	}
	run := func(qkv, out GEMVKernel) ([]float32, []float32) {
		t.Helper()
		if err := g.SetPlan(DefaultAttnKernel(), GEMMKernelFor(1), OutGEMMKernelFor(1)); err != nil {
			t.Fatal(err)
		}
		g.Reset()
		if err := g.Upload(in[:c.NEmbd], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...)
	}

	refQKV, refOut := run(GEMVOff, GEMVOff)
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.NEmbd) {
			continue
		}
		gotQKV, _ := run(k, GEMVOff)
		r, err := compare(gotQKV, refQKV)
		if err != nil {
			t.Fatalf("qkv %s: %v", k, err)
		}
		t.Logf("qkv %-4s against the GEMM over %d columns: %v", k, len(refQKV), r)
		if r.rms > 1e-4 {
			t.Errorf("qkv %s: rms %.3e against the GEMM (%v)", k, r.rms, r)
		}
		// And the indexer's columns on their own. Folded into the
		// whole-matrix rms above they are 4.6% of the columns and would hide.
		tail := g.colIQ()
		rt, err := compare(gotQKV[tail:], refQKV[tail:])
		if err != nil {
			t.Fatalf("qkv %s indexer columns: %v", k, err)
		}
		t.Logf("    the indexer's two (%d rows from %d): %v", len(refQKV)-tail, tail, rt)
		if rt.rms > 1e-4 {
			t.Errorf("qkv %s: the indexer's columns are rms %.3e from the GEMM's (%v)", k, rt.rms, rt)
		}
	}
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.GateWidth()) {
			continue
		}
		_, gotOut := run(GEMVOff, k)
		r, err := compare(gotOut, refOut)
		if err != nil {
			t.Fatalf("out %s: %v", k, err)
		}
		t.Logf("out %-4s against the GEMM over %d columns: %v", k, len(refOut), r)
		if r.rms > 1e-4 {
			t.Errorf("out %s: rms %.3e against the GEMM (%v)", k, r.rms, r)
		}
	}

	// The negative controls: a rung that cannot cut K into whole four-tile
	// steps, and a batch the one-token kernel does not read.
	for _, k := range GEMVKernels() {
		if GEMVFits(k, c.NEmbd) {
			continue
		}
		if err := g.SetGemv(k, GEMVOff); err == nil {
			t.Errorf("qkv %s was accepted for K = %d, which it does not divide into whole steps", k, c.NEmbd)
		}
	}
	if nTok > 1 {
		if err := g.Resize(nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(attnQKVGemv, attnOutGemv); err == nil {
			t.Errorf("the GEMV rungs were accepted for a batch of %d rows", nTok)
		}
	}
}

// TestAttnGPUQ4IsTheSim is L8c-7's first gate, and it is the same shape as
// the lm head's and the gated DeltaNet's: **the bank against the simulation
// that chose it**, value for value, over a whole layer rather than one
// matmul.
//
// L8c-3's +4.24% is a perplexity measured through `sim.go`, which stages a
// candidate format's floats and lets the fp16 kernels multiply them. This
// bank stores that format instead, through the same encoder — so the halves
// the matrix cores see are the same halves, the fragment order is the same,
// and the two arms have to agree exactly. They are compared at the fused
// projection, where a wrong address would show first, and at the layer's
// output, where the pack, the indexer, the score, the flash attention and the
// second projection have all run on top of it.
//
// The indexer's two BF16 projections are on the plane with everything else —
// the fp16 tail left with D13's payoff (P2) — so the simulation's arm
// quantises all six sources.
//
// It is `rtn` rather than `imatrix` so the test needs nothing but the
// checkpoint; the calibrated arm is the same code path with `qw` non-nil, and
// TestBankQKIsTheSim covers that at the encoder.
//
// **Both widths run it**, and the fifth bit is the reason this block has one:
// P3 measured `full_attn` at `q5_k` as 14.1 pp/GB against the best buildable
// four-bit arm's 4.6, so P3a's `qh` plane lands here first. The plane is
// addressed off the same tile index as the nibbles it belongs to, which is
// exactly the kind of arithmetic that is right in the GEMM and wrong in the
// GEMV, so the gate is the whole layer on both.
func TestAttnGPUQ4IsTheSim(t *testing.T) {
	for _, spec := range []string{"q4_k/32", "q5_k/32"} {
		t.Run(spec, func(t *testing.T) { attnBankIsTheSim(t, spec) })
	}
}

func attnBankIsTheSim(t *testing.T, spec string) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	sim, err := ParseQuantSim(spec)
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"
	bank, err := BankForSim(sim)
	if err != nil {
		t.Fatal(err)
	}

	dev, done := newTestDevice(t)
	defer done()

	run := func(bank DenseBank, s QuantSim, ws AttnWeights) ([]float32, []float32, int) {
		t.Helper()
		g, err := NewAttnGPUBank(dev, c, nTok, nKV, []AttnWeights{ws}, bank, s)
		if err != nil {
			t.Fatalf("attn (%s): %v", bank, err)
		}
		defer g.Destroy()
		if err := g.Upload(in.Vals, nTok); err != nil {
			t.Fatalf("upload (%s): %v", bank, err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("run (%s): %v", bank, err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...), g.WeightBytes()
	}

	// The simulation's arm: every source of the fused matrix and the output
	// projection round-tripped through the format, each under its own name,
	// before the fp16 bank ever sees the weights.
	simW := w
	simW.Q = append([]float32(nil), w.Q...)
	simW.K = append([]float32(nil), w.K...)
	simW.V = append([]float32(nil), w.V...)
	simW.O = append([]float32(nil), w.O...)
	simW.IdxQ = append([]float32(nil), w.IdxQ...)
	simW.IdxK = append([]float32(nil), w.IdxK...)
	for _, x := range [][]float32{simW.Q, simW.K, simW.V, simW.IdxQ, simW.IdxK} {
		if err := sim.Apply(x, c.NEmbd); err != nil {
			t.Fatal(err)
		}
	}
	if err := sim.Apply(simW.O, c.GateWidth()); err != nil {
		t.Fatal(err)
	}

	simQKV, simOut, simBytes := run(BankFP16, QuantSim{}, simW)
	q4QKV, q4Out, q4Bytes := run(bank, sim, w)
	t.Logf("%s bank %.1f MB against the simulation's %.1f MB of halves",
		bank, float64(q4Bytes)/1e6, float64(simBytes)/1e6)
	if q4Bytes >= simBytes {
		t.Fatalf("the %s bank is %d bytes, no smaller than %d", bank, q4Bytes, simBytes)
	}
	for _, tc := range []struct {
		what     string
		got, ref []float32
	}{
		{"the fused projection", q4QKV, simQKV},
		{"the layer's output", q4Out, simOut},
	} {
		if len(tc.got) != len(tc.ref) {
			t.Fatalf("%s: %d values against %d", tc.what, len(tc.got), len(tc.ref))
		}
		for i := range tc.ref {
			if tc.got[i] != tc.ref[i] {
				t.Fatalf("%s [%d]: bank %.9g, simulation %.9g", tc.what, i, tc.got[i], tc.ref[i])
			}
		}
		t.Logf("%s: %d values identical", tc.what, len(tc.ref))
	}
}

// TestAttnGPUQ4BankSize states what L8c-7 stages against what it replaces,
// and where the 4.500 bits stop being exact: the plane covers qkvN rows, of
// which the last few are the column block's padding — staged nibbles nothing
// reads — so the matrix as staged is a little over 4.5 bits and the test says
// how much rather than asserting a round number.
func TestAttnGPUQ4BankSize(t *testing.T) {
	_, _, c, _, _, _ := attnFixtures(t)
	real := c.QWidth() + 2*c.KVWidth() + c.IdxHeads*c.IdxDim + c.IdxDim
	n, k := roundUpInt(real, attnBN), c.NEmbd
	pad := n - real

	q4 := q8Align(qkBytes(4, n, k)) + q8Align(qkBytes(4, c.NEmbd, c.GateWidth()))
	q8 := q8Align(q8Bytes(n, k)) + q8Align(q8Bytes(c.NEmbd, c.GateWidth()))
	half := n*k*2 + c.NEmbd*c.GateWidth()*2
	weights := real*k + c.NEmbd*c.GateWidth()

	t.Logf("a layer: q4_k %.2f MB, q8 %.2f MB, halves %.2f MB — %.3f, %.3f and %.3f bits a weight",
		float64(q4)/1e6, float64(q8)/1e6, float64(half)/1e6,
		float64(q4)*8/float64(weights), float64(q8)*8/float64(weights), float64(half)*8/float64(weights))
	t.Logf("12 layers: %.2f GB against %.2f GB and %.2f GB",
		float64(12*q4)/1e9, float64(12*q8)/1e9, float64(12*half)/1e9)
	if q4 >= q8 || q8 >= half {
		t.Fatalf("the three banks are %d, %d, %d bytes and are not in order", q4, q8, half)
	}
	// Subtract the pad rows' nibbles and state the remainder rather than
	// demanding 4.500 — the record plane covers the pad rows too.
	body := qkBytes(4, n, k) - pad*k/2 + qkBytes(4, c.NEmbd, c.GateWidth())
	if bits := float64(body) * 8 / float64(weights); bits < 4.5 || bits > 4.53 {
		t.Fatalf("the staged matrix is %.4f bits a weight, want 4.500 plus the padding's records", bits)
	}
}

// TestAttnGPUQ4Gemv prices this layer's decode kernel against the GEMM on the
// 4.5-bit bank, and it is the one path TestAttnGPUQ4IsTheSim cannot reach:
// that test runs the trace's seven tokens, so both arms are `llm_gemm.comp`,
// and the split-K GEMV only exists at one row.
//
// It is **not** an equality and cannot be, for the reason the lm head's gives:
// a K-quant group is affine, so the GEMV forms `d*sc*l - dmin*m` in a register
// and multiplies it by the fp16 activation in f32 where the GEMM rounds that
// same value to a half on its way into LDS — one *fewer* rounding per weight.
// What has to hold exactly is the addressing, and the fused projection is
// where a wrong address shows — so the indexer's columns are compared on
// their own as well as folded in: they are 4.6% of the width and would hide.
//
// **Both widths**, because P3a's plane is read here with an index the GEMV
// derives differently from the GEMM — `col*2 + u` inside a tile rather than a
// lane's share of the whole run — and a plane byte off by one is a wrong
// weight, not a slow one.
func TestAttnGPUQ4Gemv(t *testing.T) {
	for _, spec := range []string{"q4_k/32", "q5_k/32"} {
		t.Run(spec, func(t *testing.T) { attnBankGemv(t, spec) })
	}
}

func attnBankGemv(t *testing.T, spec string) {
	_, tr, c, w, _, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	sim, err := ParseQuantSim(spec)
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"
	bank, err := BankForSim(sim)
	if err != nil {
		t.Fatal(err)
	}

	dev, done := newTestDevice(t)
	defer done()

	g, err := NewAttnGPUBank(dev, c, 1, nKV, []AttnWeights{w}, bank, sim)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	run := func(qkv, out GEMVKernel) ([]float32, []float32) {
		t.Helper()
		g.Reset()
		if err := g.Upload(in.Vals[:c.NEmbd], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...)
	}
	refQKV, refOut := run(GEMVOff, GEMVOff)
	tail := g.colIQ()
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.NEmbd) {
			continue
		}
		gotQKV, _ := run(k, GEMVOff)
		for _, tc := range []struct {
			what     string
			got, ref []float32
		}{
			{"qkv", gotQKV, refQKV},
			{"qkv's indexer columns", gotQKV[tail:], refQKV[tail:]},
		} {
			if len(tc.ref) == 0 {
				continue
			}
			r, err := compare(tc.got, tc.ref)
			if err != nil {
				t.Fatalf("%s %s: %v", tc.what, k, err)
			}
			t.Logf("%-21s %-4s against the GEMM: %v", tc.what, k, r)
			if r.rms > 1e-3 {
				t.Errorf("%s %s is rms %.3e from the GEMM, want one rounding's worth (%v)",
					tc.what, k, r.rms, r)
			}
		}
	}
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.GateWidth()) {
			continue
		}
		_, gotOut := run(GEMVOff, k)
		r, err := compare(gotOut, refOut)
		if err != nil {
			t.Fatalf("out %s: %v", k, err)
		}
		t.Logf("out                   %-4s against the GEMM: %v", k, r)
		if r.rms > 1e-3 {
			t.Errorf("out %s is rms %.3e from the GEMM (%v)", k, r.rms, r)
		}
	}
}

// TestAttnGPUGemvTwoRows is P5b's gate on the full-attention layer, and the
// same assertion TestDeltaNetGPUGemvTwoRows makes on the gated DeltaNet: the
// decode GEMV carrying two rows of A computes what the GEMM computes, row for
// row. The kernel is shared, so what this adds over that test is *this*
// block's wiring — its partial arena, its two grids and its sum's row axis.
func TestAttnGPUGemvTwoRows(t *testing.T) {
	g, _, c, _, in, nTok, _, done := attnGPU(t)
	defer done()
	if nTok < 2 {
		t.Skip("the trace has fewer than two tokens")
	}
	const rows = 2
	if rows > GEMVMaxRows {
		t.Skipf("GEMVMaxRows is %d", GEMVMaxRows)
	}
	run := func(qkv, out GEMVKernel) ([]float32, []float32) {
		t.Helper()
		if err := g.SetPlan(DefaultAttnKernel(), GEMMKernelFor(rows), OutGEMMKernelFor(rows)); err != nil {
			t.Fatal(err)
		}
		g.Reset()
		if err := g.Upload(in[:rows*c.NEmbd], rows); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...)
	}

	// The row-1 control; see rowMoved and the DeltaNet's twin.
	second := func(tok int, qkv, out GEMVKernel) []float32 {
		t.Helper()
		buf := make([]float32, rows*c.NEmbd)
		copy(buf, in[:c.NEmbd])
		copy(buf[c.NEmbd:], in[tok*c.NEmbd:(tok+1)*c.NEmbd])
		if err := g.SetPlan(DefaultAttnKernel(), GEMMKernelFor(rows), OutGEMMKernelFor(rows)); err != nil {
			t.Fatal(err)
		}
		g.Reset()
		if err := g.Upload(buf, rows); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()[g.qkvN():]...)
	}

	refQKV, refOut := run(GEMVOff, GEMVOff)
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.NEmbd) {
			continue
		}
		if nTok > 2 {
			rowMoved(t, fmt.Sprintf("qkv %s row 1", k), second(1, k, GEMVOff), second(2, k, GEMVOff))
		}
		gotQKV, _ := run(k, GEMVOff)
		for _, arm := range []struct {
			name string
			got  []float32
			want []float32
		}{
			{"both rows", gotQKV, refQKV},
			{"row 1", gotQKV[g.qkvN():], refQKV[g.qkvN():]},
		} {
			r, err := compare(arm.got, arm.want)
			if err != nil {
				t.Fatalf("qkv %s %s: %v", k, arm.name, err)
			}
			t.Logf("qkv %-4s at %d rows, %-9s over %d values: %v", k, rows, arm.name, len(arm.want), r)
			if r.rms > 1e-4 {
				t.Errorf("qkv %s at %d rows, %s: rms %.3e against the GEMM (%v)", k, rows, arm.name, r.rms, r)
			}
		}
	}
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.GateWidth()) {
			continue
		}
		_, gotOut := run(GEMVOff, k)
		for _, arm := range []struct {
			name string
			got  []float32
			want []float32
		}{
			{"both rows", gotOut, refOut},
			{"row 1", gotOut[c.NEmbd:], refOut[c.NEmbd:]},
		} {
			r, err := compare(arm.got, arm.want)
			if err != nil {
				t.Fatalf("out %s %s: %v", k, arm.name, err)
			}
			t.Logf("out %-4s at %d rows, %-9s over %d values: %v", k, rows, arm.name, len(arm.want), r)
			if r.rms > 1e-4 {
				t.Errorf("out %s at %d rows, %s: rms %.3e against the GEMM (%v)", k, rows, arm.name, r.rms, r)
			}
		}
	}
}
