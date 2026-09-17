package llm

import (
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
		{"indexer_q_raw", g.ColIQ(), c.IdxHeads * c.IdxDim, proj(w.IdxQ, c.IdxHeads*c.IdxDim), 2e-3},
		{"indexer_k_raw", g.ColIK(), c.IdxDim, proj(w.IdxK, c.IdxDim), 2e-3},
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
	if r.rms > 2e-3 {
		t.Errorf("indexer_k_raw-3: rms %.3e over 2e-3 (%v)", r.rms, r)
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
		{"indexer_k_pooled-3", g.IdxK(), cpu.IdxK, 1e-3, 1e-3},
		{"indexer_q-3", g.IdxQ(), cpu.IdxQ, 1e-3, 1e-3},
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
		// And the BF16 tail on its own. Folded into the whole-matrix rms
		// above it is 4.6% of the columns and would hide.
		tail := g.qkvQuantRows()
		rt, err := compare(gotQKV[tail:], refQKV[tail:])
		if err != nil {
			t.Fatalf("qkv %s tail: %v", k, err)
		}
		t.Logf("    tail (the indexer's two, %d halved rows from %d): %v", len(refQKV)-tail, tail, rt)
		if rt.rms > 1e-4 {
			t.Errorf("qkv %s: the fp16 tail is rms %.3e from the GEMM's (%v)", k, rt.rms, rt)
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
// Both arrangements of the indexer are checked, because the tail is the one
// structural choice this stage makes: with `qsa_indexer` out of the plan its
// two BF16 projections stay halves and the GEMM takes its tail branch, and
// with it in there is no tail at all.
//
// It is `rtn` rather than `imatrix` so the test needs nothing but the
// checkpoint; the calibrated arm is the same code path with `qw` non-nil, and
// TestBankQ4KIsTheSim covers that at the encoder.
func TestAttnGPUQ4IsTheSim(t *testing.T) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	sim, err := ParseQuantSim("q4_k/32")
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"

	dev, done := newTestDevice(t)
	defer done()

	run := func(bank DenseBank, idxQ bool) ([]float32, []float32, int) {
		t.Helper()
		g, err := NewAttnGPUBank(dev, c, nTok, nKV, []AttnWeights{w}, bank, sim, idxQ)
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

	// The simulation's arm. A plan that covers `full_attn` alone quantises
	// the query, the key and the value and leaves the indexer's two BF16
	// matrices as the checkpoint's own floats — which on the fp16 bank is
	// exactly what staging them unquantised means — so the same fixture
	// serves both arrangements as long as the halves it stages are the
	// format's. That is what simQuantise does here, in place, before the
	// block ever sees the weights.
	simW := w
	simW.Q = append([]float32(nil), w.Q...)
	simW.K = append([]float32(nil), w.K...)
	simW.V = append([]float32(nil), w.V...)
	simW.O = append([]float32(nil), w.O...)
	for _, x := range [][]float32{simW.Q, simW.K, simW.V} {
		if err := sim.Apply(x, c.NEmbd); err != nil {
			t.Fatal(err)
		}
	}
	if err := sim.Apply(simW.O, c.GateWidth()); err != nil {
		t.Fatal(err)
	}

	simQKV, simOut, simBytes := func() ([]float32, []float32, int) {
		g, err := NewAttnGPUBank(dev, c, nTok, nKV, []AttnWeights{simW}, BankFP16, QuantSim{}, false)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		if err := g.Upload(in.Vals, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...), g.WeightBytes()
	}()

	q4QKV, q4Out, q4Bytes := run(BankQ4K, false)
	t.Logf("bank %.1f MB against the simulation's %.1f MB of halves",
		float64(q4Bytes)/1e6, float64(simBytes)/1e6)
	if q4Bytes >= simBytes {
		t.Fatalf("the q4_k bank is %d bytes, no smaller than %d", q4Bytes, simBytes)
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

	// And the same with the indexer on the plane, against a simulation that
	// quantises its two matrices too. Only the fused projection's indexer
	// columns can differ, so both tensors are compared again rather than
	// only the ones that moved.
	idxW := simW
	idxW.IdxQ = append([]float32(nil), w.IdxQ...)
	idxW.IdxK = append([]float32(nil), w.IdxK...)
	for _, x := range [][]float32{idxW.IdxQ, idxW.IdxK} {
		if err := sim.Apply(x, c.NEmbd); err != nil {
			t.Fatal(err)
		}
	}
	idxSimQKV, idxSimOut, _ := func() ([]float32, []float32, int) {
		g, err := NewAttnGPUBank(dev, c, nTok, nKV, []AttnWeights{idxW}, BankFP16, QuantSim{}, false)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		if err := g.Upload(in.Vals, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Out()...), g.WeightBytes()
	}()
	idxQKV, idxOut, idxBytes := run(BankQ4K, true)
	t.Logf("with the indexer on the plane: %.1f MB, against %.1f MB with it in the fp16 tail",
		float64(idxBytes)/1e6, float64(q4Bytes)/1e6)
	if idxBytes >= q4Bytes {
		t.Fatalf("staging the indexer at 4.5 bits is %d bytes, no smaller than %d", idxBytes, q4Bytes)
	}
	for _, tc := range []struct {
		what     string
		got, ref []float32
	}{
		{"the fused projection, indexer included", idxQKV, idxSimQKV},
		{"the layer's output, indexer included", idxOut, idxSimOut},
	} {
		for i := range tc.ref {
			if tc.got[i] != tc.ref[i] {
				t.Fatalf("%s [%d]: bank %.9g, simulation %.9g", tc.what, i, tc.got[i], tc.ref[i])
			}
		}
		t.Logf("%s: %d values identical", tc.what, len(tc.ref))
	}
}

// TestAttnGPUQ4BankSize states what L8c-7 stages against what it replaces,
// and where the 4.500 bits stop being exact.
//
// The fused projection's tail is the reason: the indexer's two projections
// are BF16 in the checkpoint and stay halves unless `qsa_indexer` is in the
// plan, and the quantised plane still covers all qkvN rows because the kernel
// derives the record plane's base from gemmN * gemmK. So the matrix as staged
// is a little over 4.5 bits and the test says how much rather than asserting
// a round number.
func TestAttnGPUQ4BankSize(t *testing.T) {
	_, _, c, _, _, _ := attnFixtures(t)
	n, k := roundUpInt(c.QWidth()+2*c.KVWidth()+c.IdxHeads*c.IdxDim+c.IdxDim, attnBN), c.NEmbd
	tail := n - (c.QWidth() + 2*c.KVWidth())

	q4 := q8Align(q4kBytes(n, k)) + tail*k*2 + q8Align(q4kBytes(c.NEmbd, c.GateWidth()))
	q4idx := q8Align(q4kBytes(n, k)) + q8Align(q4kBytes(c.NEmbd, c.GateWidth()))
	q8 := q8Align(q8Bytes(n, k)) + tail*k*2 + q8Align(q8Bytes(c.NEmbd, c.GateWidth()))
	half := n*k*2 + c.NEmbd*c.GateWidth()*2
	weights := (c.QWidth()+2*c.KVWidth())*k + c.NEmbd*c.GateWidth() + tail*k

	t.Logf("a layer: q4_k %.2f MB (%.2f with the indexer on the plane), q8 %.2f MB, halves %.2f MB — %.3f, %.3f, %.3f and %.3f bits a weight",
		float64(q4)/1e6, float64(q4idx)/1e6, float64(q8)/1e6, float64(half)/1e6,
		float64(q4)*8/float64(weights), float64(q4idx)*8/float64(weights),
		float64(q8)*8/float64(weights), float64(half)*8/float64(weights))
	t.Logf("12 layers: %.2f GB (%.2f) against %.2f GB and %.2f GB",
		float64(12*q4)/1e9, float64(12*q4idx)/1e9, float64(12*q8)/1e9, float64(12*half)/1e9)
	if q4idx >= q4 || q4 >= q8 || q8 >= half {
		t.Fatalf("the four banks are %d, %d, %d, %d bytes and are not in order", q4idx, q4, q8, half)
	}
	// The quantised half alone is the format's own width to the bit. The
	// record plane covers the tail's rows too, so subtract only the nibbles
	// and state the remainder rather than demanding 4.500.
	body := q4kBytes(n, k) - tail*k/2 + q4kBytes(c.NEmbd, c.GateWidth())
	bodyW := (c.QWidth()+2*c.KVWidth())*k + c.NEmbd*c.GateWidth()
	if bits := float64(body) * 8 / float64(bodyW); bits < 4.5 || bits > 4.53 {
		t.Fatalf("the quantised half is %.4f bits a weight, want 4.500 plus the tail's records", bits)
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
// where a wrong address shows: with the indexer in the fp16 tail the GEMV has
// to derive the same split the GEMM does, and a column read out of the wrong
// plane is not a tolerance but somebody else's indexer query. So the tail's
// columns are compared on their own as well as folded in — they are 4.6% of
// the width and would hide.
func TestAttnGPUQ4Gemv(t *testing.T) {
	_, tr, c, w, _, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	sim, err := ParseQuantSim("q4_k/32")
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"

	dev, done := newTestDevice(t)
	defer done()

	for _, idxQ := range []bool{false, true} {
		g, err := NewAttnGPUBank(dev, c, 1, nKV, []AttnWeights{w}, BankQ4K, sim, idxQ)
		if err != nil {
			t.Fatal(err)
		}
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
		tail := g.qkvQuantRows()
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
				{"qkv's fp16 tail", gotQKV[tail:], refQKV[tail:]},
			} {
				if len(tc.ref) == 0 {
					continue
				}
				r, err := compare(tc.got, tc.ref)
				if err != nil {
					t.Fatalf("%s %s: %v", tc.what, k, err)
				}
				t.Logf("indexer quantised %-5v %-14s %-4s against the GEMM: %v", idxQ, tc.what, k, r)
				if r.rms > 1e-3 {
					t.Errorf("indexer quantised %v: %s %s is rms %.3e from the GEMM, want one rounding's worth (%v)",
						idxQ, tc.what, k, r.rms, r)
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
			t.Logf("indexer quantised %-5v out            %-4s against the GEMM: %v", idxQ, k, r)
			if r.rms > 1e-3 {
				t.Errorf("indexer quantised %v: out %s is rms %.3e from the GEMM (%v)", idxQ, k, r.rms, r)
			}
		}
		g.Destroy()
	}
}
