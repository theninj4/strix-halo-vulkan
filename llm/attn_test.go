package llm

import (
	"math"
	"testing"
)

// attnFixtures opens the checkpoint, the trace and layer 3 — the first
// full-attention layer, and the only one of the four dumped layers that has
// any of this in it (the interval is 4, so layers 0-2 are linear).
func attnFixtures(t *testing.T) (*Model, *Trace, AttnConfig, AttnWeights, int, int) {
	t.Helper()
	m, tr := fixtures(t)
	c, ok, err := m.AttnConfig(attnLayer)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("layer %d has no attention tensors", attnLayer)
	}
	w, err := m.AttnWeights(attnLayer)
	if err != nil {
		t.Fatal(err)
	}
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	// The cell count is the reference's, not the prompt's: flash attention
	// pads it to 256, and every part of the indexer — the block grid, the
	// bias, the top-k width — is cut against that number rather than against
	// the 7 tokens. Reading it off the dump is the only honest way to get it.
	d, err := tr.Get("indexer_score_tokens-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	return m, tr, c, w, len(ids), int(d.NE[0])
}

const attnLayer = 3

// TestAttnConfig checks the layer's shape against the checkpoint, including
// the two arrays that are per layer rather than global: the rope sections and
// the compress ratios, of which only the full-attention layers' are non-zero.
func TestAttnConfig(t *testing.T) {
	m, _ := fixtures(t)
	for _, layer := range []int{0, 1, 2} {
		if _, ok, _ := m.AttnConfig(layer); ok {
			t.Errorf("layer %d reports attention tensors, but the interval is 4", layer)
		}
	}
	c, ok, err := m.AttnConfig(attnLayer)
	if err != nil || !ok {
		t.Fatalf("layer %d: %v (present %v)", attnLayer, err, ok)
	}
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"heads", c.NHead, 24},
		{"kv heads", c.NHeadKV, 2},
		{"head dim", c.HeadDim, 256},
		{"rope dims", c.RopeDims, 64},
		{"indexer heads", c.IdxHeads, 4},
		{"indexer dim", c.IdxDim, 128},
		{"top k", c.TopK, 2048},
		{"compress ratio", c.Ratio, 4},
		{"sections sum", c.Sections[0] + c.Sections[1] + c.Sections[2] + c.Sections[3], c.RopeDims / 2},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.RopeBase != 1e7 {
		t.Errorf("rope base = %g, want 1e7", c.RopeBase)
	}
}

// TestRoPEMultiIsNeoXOnText states the claim the rotary rests on.
//
// The model's rotary is interleaved M-RoPE with sections [11, 11, 10, 0], and
// none of that is visible on a text batch: llama.cpp puts the token position
// in the t, h and w slots and zero in e, and no sector of [11, 11, 10] ever
// selects e — so every pair rotates by the same angle it would under plain
// NeoX rope. If that ever stops being true, this test says so rather than a
// tensor comparison failing three stages downstream.
func TestRoPEMultiIsNeoXOnText(t *testing.T) {
	c := AttnConfig{RopeDims: 64, RopeBase: 1e7, Sections: [4]int{11, 11, 10, 0}}
	const heads, dim, nTok = 3, 128, 5
	x := make([]float32, nTok*heads*dim)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.37))
	}
	want := append([]float32(nil), x...)
	pos := []int32{0, 1, 2, 17, 4095}

	RoPEMulti(c, x, pos, heads, dim, nTok)

	// Plain NeoX: pair i with i+n_rot/2, angle pos * base^(-2i/n_rot).
	half := c.RopeDims / 2
	scale := math.Pow(float64(c.RopeBase), -2/float64(c.RopeDims))
	for tk := 0; tk < nTok; tk++ {
		theta := float64(pos[tk])
		for i := 0; i < half; i++ {
			cos, sin := float32(math.Cos(theta)), float32(math.Sin(theta))
			for h := 0; h < heads; h++ {
				row := want[(tk*heads+h)*dim:]
				x0, x1 := row[i], row[i+half]
				row[i] = x0*cos - x1*sin
				row[i+half] = x0*sin + x1*cos
			}
			theta *= scale
		}
	}
	r, err := compare(x, want)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("imrope against NeoX on text positions: %v", r)
	if r.maxAbs != 0 {
		t.Errorf("interleaved M-RoPE is not NeoX on a text batch: %v", r)
	}
}

// TestRoPEMultiRotatesOnlyNRot is the other half of the rotary's contract:
// 64 of the 256 head dims rotate and the other 192 are copied untouched,
// which is the thing a transcription that reads n_rot as the head dim gets
// wrong without producing anything obviously broken.
func TestRoPEMultiRotatesOnlyNRot(t *testing.T) {
	c := AttnConfig{RopeDims: 64, RopeBase: 1e7, Sections: [4]int{11, 11, 10, 0}}
	const dim = 256
	x := make([]float32, dim)
	for i := range x {
		x[i] = float32(i + 1)
	}
	want := append([]float32(nil), x...)
	RoPEMulti(c, x, []int32{3}, 1, dim, 1)
	for i := c.RopeDims; i < dim; i++ {
		if x[i] != want[i] {
			t.Fatalf("dim %d changed (%g -> %g); only the first %d rotate", i, want[i], x[i], c.RopeDims)
		}
	}
	same := 0
	for i := 0; i < c.RopeDims; i++ {
		if x[i] == want[i] {
			same++
		}
	}
	if same > 2 {
		t.Errorf("%d of the first %d dims are unchanged; the rotation is not being applied", same, c.RopeDims)
	}
}

// TestAttnIndexer is the gate on the QSA indexer: every tensor llama.cpp
// names, driven from its own block input (`hc_mixed-3`).
//
// The tolerances are measured, and the three numerics the reference brings to
// this block are what they measure — see TestAttnNumericsAreTheReferences.
func TestAttnIndexer(t *testing.T) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	c.Act = RefQ8
	got := AttnLayer(c, w, in.Vals, nTok, nKV)

	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		{"indexer_k_raw-3", got.IdxKRaw, 1e-5}, // BF16 weights, so no quantisation either side
		{"indexer_k_pooled-3", got.IdxKPooled, 1e-6},
		{"indexer_k-3", got.IdxK, 1e-6},
		{"indexer_q-3", got.IdxQ, 1e-5},
		{"indexer_score-3", got.IdxScore, 1e-3}, // on values to 145.8, i.e. 4e-7 relative
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-22s %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// The per-cell score carries infinities — every masked cell is one — so
	// it is compared as a mask plus the finite values, not by subtraction.
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
			t.Fatalf("cell %d of token %d: got %g, reference %g — the mask disagrees",
				i%nKV, i/nKV, g, wv)
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

// TestAttnTopKCannotBiteHere is the gate's honest limit at this prompt
// length, stated rather than papered over — and one thing about the oracle
// that had to be measured to be believed.
//
// The reference asks for `top_k + ratio - 1` = 2051 cells and the cache has
// 256, so the selection names every cell and *cannot* remove anything. The
// test therefore checks what is actually checkable: that the width is the
// reference's, that both selections cover every visible cell, and that
// running the attention with the selection gives bit-identical output to
// running it without — which is the property the whole sparse path rests on
// at short context.
//
// And the order is *not* checked, because llama.cpp's is not score order.
// At token 4 its cells 4-6 carry the 1e9 "always visible" marker and cells
// 0-3 carry 117.7, yet `indexer_top_k` comes back as the identity
// permutation. The Vulkan backend's TOP_K is a selection, not a sort, and
// with k = n_kv its output is unobservable downstream — every cell is
// unmasked either way. L4's longer context is where that stops being true
// and where the order becomes testable.
func TestAttnTopKCannotBiteHere(t *testing.T) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	c.Act = RefQ8
	got := AttnLayer(c, w, in.Vals, nTok, nKV)
	want, err := tr.Get("indexer_top_k-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if want.Ints == nil {
		t.Fatalf("indexer_top_k-3 is ggml type %d, want an integer tensor", want.GType)
	}
	width := int(want.NE[0])
	if w := len(got.TopK) / nTok; w != width {
		t.Fatalf("our top-k is %d wide, the reference's is %d", w, width)
	}
	if width < nKV {
		t.Fatalf("the selection bites at %d cells of %d — this test is only valid while it cannot", width, nKV)
	}
	t.Logf("top-k width %d of %d cells: every cell is named, so the selection is a no-op here", width, nKV)

	for i := 0; i < nTok; i++ {
		ours, theirs := map[int32]bool{}, map[int32]bool{}
		for j := 0; j < width; j++ {
			ours[got.TopK[i*width+j]] = true
			theirs[want.Ints[i*width+j]] = true
		}
		for j := 0; j <= i; j++ {
			if !ours[int32(j)] {
				t.Errorf("token %d: our selection misses visible cell %d", i, j)
			}
			if !theirs[int32(j)] {
				t.Errorf("token %d: the reference's selection misses visible cell %d", i, j)
			}
		}
	}

	// The property that makes the above safe: with every cell selected, the
	// sparse path and the dense one are the same computation.
	dense := attention(c, f16Copy(got.Q), got.K, got.V, nTok, nil)
	r, err := compare(got.Pregate, dense)
	if err != nil {
		t.Fatal(err)
	}
	if r.maxAbs != 0 {
		t.Errorf("the selection changed the attention output (%v), but it names every cell", r)
	}
}

// TestAttnLayer is the L2 gate for the layer itself: the fused query/gate
// split, the rotary, the attention and the output projection, against
// llama.cpp's own tensors.
func TestAttnLayer(t *testing.T) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []struct {
		name string
		act  Numerics
		tol  map[string]float64
	}{
		// Exact is the mathematical model: the gap to it is the reference's
		// own arithmetic, not ours. RefQ8 models all three things the backend
		// does — int8 activations under a Q8_0 weight, an fp16 KV cache, and
		// an fp16 matmul — and what is left after that is the flash-attention
		// kernel's own accumulation order, which is not reproducible from
		// outside it.
		{"Exact", Exact, map[string]float64{
			"gate_reshaped-3": 5e-3, "gate_sigmoid-3": 1e-3,
			"attn_pregate-3": 5e-3, "attn_gated-3": 1e-3, "attn_output-3": 5e-3,
		}},
		{"RefQ8", RefQ8, map[string]float64{
			"gate_reshaped-3": 1e-5, "gate_sigmoid-3": 1e-6,
			"attn_pregate-3": 5e-4, "attn_gated-3": 1e-4, "attn_output-3": 5e-4,
		}},
	} {
		cfg := c
		cfg.Act = mode.act
		got := AttnLayer(cfg, w, in.Vals, nTok, nKV)
		for _, tc := range []struct {
			name string
			got  []float32
		}{
			{"gate_reshaped-3", got.Gate},
			{"gate_sigmoid-3", got.GateSigmoid},
			{"attn_pregate-3", got.Pregate},
			{"attn_gated-3", got.Gated},
			{"attn_output-3", got.Out},
		} {
			want, err := tr.Get(tc.name, 0)
			if err != nil {
				t.Fatal(err)
			}
			r, err := compare(tc.got, want.Vals)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			t.Logf("%-6s %-18s %v", mode.name, tc.name, r)
			if tol := mode.tol[tc.name]; r.rms > tol {
				t.Errorf("%s %s: rms %.3e over %.1e (%v)", mode.name, tc.name, r.rms, tol, r)
			}
		}
	}
}

// TestAttnNumericsAreTheReferences states the two numerics this stage
// discovered, as checked claims rather than comments — the same discipline
// L2b used for the int8 activations it found.
//
// Both are things the reference's *graph* does not say and only a measurement
// finds: the indexer's key cache is fp16, and a matmul between two F32
// tensors is evaluated on fp16 matrix cores.
func TestAttnNumericsAreTheReferences(t *testing.T) {
	_, tr, c, w, nTok, nKV := attnFixtures(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	rms := func(cfg AttnConfig, name string, pick func(*AttnTrace) []float32) float64 {
		got := AttnLayer(cfg, w, in.Vals, nTok, nKV)
		want, err := tr.Get(name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(pick(got), want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		return r.rms
	}
	exact, ref := c, c
	exact.Act, ref.Act = Exact, RefQ8

	pooled := [2]float64{
		rms(exact, "indexer_k_pooled-3", func(a *AttnTrace) []float32 { return a.IdxKPooled }),
		rms(ref, "indexer_k_pooled-3", func(a *AttnTrace) []float32 { return a.IdxKPooled }),
	}
	score := [2]float64{
		rms(exact, "indexer_score-3", func(a *AttnTrace) []float32 { return a.IdxScore }),
		rms(ref, "indexer_score-3", func(a *AttnTrace) []float32 { return a.IdxScore }),
	}
	t.Logf("indexer_k_pooled: f32 %.3e, fp16 cache %.3e — %.0fx", pooled[0], pooled[1], pooled[0]/pooled[1])
	t.Logf("indexer_score:    f32 %.3e, fp16 matmul %.3e — %.0fx", score[0], score[1], score[0]/score[1])
	if pooled[0]/pooled[1] < 100 {
		t.Errorf("modelling the fp16 indexer cache is only worth %.0fx; it no longer explains the gap", pooled[0]/pooled[1])
	}
	if score[0]/score[1] < 20 {
		t.Errorf("modelling the fp16 matmul is only worth %.0fx; it no longer explains the gap", score[0]/score[1])
	}
}

// TestAttnLayerChain is the deepest end-to-end check L2 can make: the whole
// attention half of a full-attention layer, from the previous layer's output
// to the residual the FFN half reads.
//
//	l_last-2  ->  hc_mix (attn)  ->  attention + indexer  ->  hc_combine  ->  hc_combine-3
//
// Nothing in the middle is llama.cpp's — the mixer's output, the query, the
// selection and the attention are all ours — so this is four stages composing
// rather than four stages checked in isolation. What it cannot reach is
// `l_last-3`, which needs the MoE half (L5).
func TestAttnLayerChain(t *testing.T) {
	m, tr, c, w, nTok, nKV := attnFixtures(t)
	c.Act = RefQ8

	in, err := tr.Get("l_last-2", 0)
	if err != nil {
		t.Fatal(err)
	}
	res := append([]float32(nil), in.Vals...)

	hw, err := m.HCWeights(attnLayer, "attn")
	if err != nil {
		t.Fatal(err)
	}
	hc := m.Config.HCConfig()
	hc.Act = RefQ8
	mixed, inject, _, _ := HCMix(hc, hw, res, nTok)

	got := AttnLayer(c, w, mixed, nTok, nKV)
	HCCombine(hc, res, got.Out, inject, nTok)

	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		{"hc_mixed-3", mixed, 2e-3},
		{"attn_output-3", got.Out, 5e-4},
		{"hc_combine-3", res, 1e-4},
	} {
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-16s %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}
}
