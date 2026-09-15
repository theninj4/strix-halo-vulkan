package llm

import (
	"fmt"
	"testing"
)

// Layer 0 is the first linear-attention layer and the first layer of the
// model: its input is `hc_mixed-0`, which L2b already checks, and its
// recurrent state and convolution history are both zero. Layers 1 and 2 are
// the other two the dump carries, and they run with the same code.
const dnLayer = 0

func dnFixtures(t *testing.T, layer int) (*Trace, DeltaNetConfig, DeltaNetWeights, int) {
	t.Helper()
	m, tr := fixtures(t)
	c, ok, err := m.DeltaNetConfig(layer)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("layer %d has no ssm tensors", layer)
	}
	w, err := m.DeltaNetWeights(layer)
	if err != nil {
		t.Fatal(err)
	}
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	// RefQ8 is the oracle's arithmetic (L2b-3); L2Max is the oracle's
	// *formula*, which is the build's rather than the model's — see
	// TestDeltaNetL2NormIsTheBuilds.
	c.Act = RefQ8
	c.QKNorm = L2Max
	return tr, c, w, len(ids)
}

// TestDeltaNetConfig checks the shape the layer runs at against the
// checkpoint's own `ssm.*` keys, and that the two layer families partition:
// layer 3 is full attention and must report no linear-attention tensors.
func TestDeltaNetConfig(t *testing.T) {
	m, _ := fixtures(t)
	c, ok, err := m.DeltaNetConfig(dnLayer)
	if err != nil || !ok {
		t.Fatalf("layer %d: %v (present %v)", dnLayer, err, ok)
	}
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"head dim", c.HeadDim, 128},
		{"key heads", c.NHeadK, 16},
		{"value heads", c.NHeadV, 48},
		{"conv kernel", c.Conv, 4},
		{"inner", c.Inner, 6144},
		{"conv width", c.ConvWidth(), 10240},
		{"qk width", c.QKWidth(), 2048},
		{"v width", c.VWidth(), 6144},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	// 3.1 MB a layer, 113 MB over the 36 linear layers, and constant in the
	// context length — which is the architecture's whole claim.
	if got := c.StateSize() * 4; got != 3145728 {
		t.Errorf("state is %d bytes, want 3145728", got)
	}
	if _, ok, _ := m.DeltaNetConfig(attnLayer); ok {
		t.Errorf("layer %d reports ssm tensors, but it is a full-attention layer", attnLayer)
	}
}

// TestDeltaNetLayer is L3's gate: every tensor llama.cpp names inside layer
// 0's linear attention, against llama.cpp's own value for it.
//
// The input is the reference's `hc_mixed-0` rather than our own, so a
// disagreement here is this layer's and not an accumulation of L2's.
func TestDeltaNetLayer(t *testing.T) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	got := DeltaNetLayer(c, w, in.Vals, nTok, nil)

	for _, tc := range []struct {
		name string
		got  []float32
		tol  float64
	}{
		// The projections, and the two scalars per head they feed.
		{"linear_attn_qkv_mixed", got.QKVMixed, 2e-4},
		{"z", got.Z, 2e-4},
		{"alpha", got.Alpha, 5e-4},
		{"a_softplus", got.ASoftplus, 5e-4},
		{"gate", got.Gate, 5e-3},
		{"beta", got.Beta, 5e-4},
		{"beta_sigmoid", got.BetaSig, 1e-4},
		// The convolution and its three views.
		{"conv_output_raw", got.ConvRaw, 2e-4},
		{"conv_output_silu", got.ConvSilu, 2e-4},
		{"q_conv", got.Q, 2e-4},
		{"k_conv", got.K, 2e-4},
		{"v_conv_predelta", got.V, 2e-4},
		{"q_conv_predelta", got.QNorm, 1e-6},
		{"k_conv_predelta", got.KNorm, 1e-6},
		// The recurrence, and the block's output.
		{"attn_output", got.Out, 1e-7},
		{"new_state", got.NewState, 1e-6},
		{"final_output", got.Final, 1e-6},
		{"linear_attn_out", got.Result, 1e-6},
	} {
		want, err := tr.Get(fmt.Sprintf("%s-%d", tc.name, dnLayer), 0)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		r, err := compare(tc.got, want.Vals)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		t.Logf("%-22s %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.0e", tc.name, r.rms, tc.tol)
		}
	}

	// `state_predelta` is the input state, and it is zero: layer 0 of a fresh
	// sequence has nothing to remember. Saying so is what makes every number
	// above a check of the recurrence rather than of the fixture.
	pre, err := tr.Get(fmt.Sprintf("state_predelta-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range pre.Vals {
		if v != 0 {
			t.Fatalf("state_predelta[%d] = %g, want a zero state", i, v)
		}
	}
}

// TestDeltaNetLayers runs the other two linear layers the dump carries.
// Layers 1 and 2 read a residual the PLE block has written into and the
// hyper-connection block has mixed, so they are the check that the layer is
// not accidentally fitted to layer 0's inputs.
func TestDeltaNetLayers(t *testing.T) {
	for _, layer := range []int{1, 2} {
		tr, c, w, nTok := dnFixtures(t, layer)
		in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", layer), 0)
		if err != nil {
			t.Fatal(err)
		}
		got := DeltaNetLayer(c, w, in.Vals, nTok, nil)
		for _, tc := range []struct {
			name string
			got  []float32
			tol  float64
		}{
			{"attn_output", got.Out, 1e-6},
			{"new_state", got.NewState, 1e-5},
			{"final_output", got.Final, 1e-5},
			{"linear_attn_out", got.Result, 1e-4},
		} {
			want, err := tr.Get(fmt.Sprintf("%s-%d", tc.name, layer), 0)
			if err != nil {
				t.Fatal(err)
			}
			r, err := compare(tc.got, want.Vals)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("layer %d %-16s %v", layer, tc.name, r)
			if r.rms > tc.tol {
				t.Errorf("layer %d %s: rms %.3e over %.0e", layer, tc.name, r.rms, tc.tol)
			}
		}
	}
}

// TestDeltaNetGateNumerics asks the question L2e-3 left open for every other
// F32 matmul in the model: `ssm_alpha` and `ssm_beta` are F32 tensors read by
// an F32 activation, and the reference evaluated the indexer's F32 x F32
// score on the fp16 matrix cores. Does it do the same to a [2560, 48]
// projection?
//
// The test measures rather than assumes, and the answer picks F32MM's
// default. `alpha` is the tensor to read: `gate` is the same number through
// an exponential and `beta_sigmoid` through a sigmoid, both of which
// compress the difference.
func TestDeltaNetGateNumerics(t *testing.T) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get(fmt.Sprintf("alpha-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	var best Numerics
	bestRMS := 0.0
	for _, mode := range []struct {
		name string
		n    Numerics
	}{{"f32", Exact}, {"fp16 operands", RefQ8}} {
		c.F32MM = mode.n
		got := DeltaNetLayer(c, w, in.Vals, nTok, nil)
		r, err := compare(got.Alpha, want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("alpha under %-14s %v", mode.name, r)
		if bestRMS == 0 || r.rms < bestRMS {
			best, bestRMS = mode.n, r.rms
		}
	}
	if best != Exact {
		t.Errorf("the F32 projections fit better in fp16 (rms %.3e); F32MM's default should say so", bestRMS)
	}
}

// TestDeltaNetHeadMapIsModulo is the negative control on the one choice in
// this layer that a reading of the architecture does not settle.
//
// 48 value heads read 16 key heads. `ggml_repeat` tiles and so maps h to
// h % 16; `ggml_mul_mat`'s own broadcast divides and so maps h to h / 3. Both
// are how ggml broadcasts something, both produce a plausible tensor, and
// they agree on 16 of the 48 heads. The fused op does modulo. This runs the
// same recurrence with divide and shows where that lands.
func TestDeltaNetHeadMapIsModulo(t *testing.T) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get(fmt.Sprintf("attn_output-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	got := DeltaNetLayer(c, w, in.Vals, nTok, nil)
	good, err := compare(got.Out, want.Vals)
	if err != nil {
		t.Fatal(err)
	}

	// Same q, k, v, gate and beta; the other map, from a zero state.
	div := make([]int, c.NHeadV)
	for h := range div {
		div[h] = h / (c.NHeadV / c.NHeadK)
	}
	for i := range got.Out {
		got.Out[i] = 0
	}
	deltaRule(c, got, NewDeltaNetState(c), nTok, div)
	bad, err := compare(got.Out, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("modulo: %v", good)
	t.Logf("divide: %v", bad)
	if bad.rms < 20*good.rms {
		t.Errorf("the divide map is only %.1fx worse than the modulo one, so this control proves nothing",
			bad.rms/good.rms)
	}
}

// TestDeltaNetChunkBoundary is the second half of L3's gate: the state has to
// be exact across a split, because at L7 every decoded token is its own
// batch and at L6 a long prompt is several ubatches.
//
// Three tokens then four must produce, to the last bit, what seven produce in
// one pass — both the output and the recurrent state. Two things carry across
// the boundary and only one of them is the delta rule's: the convolution's
// three-column history is the other, and a kernel that forgets it is wrong in
// a way only this test sees.
func TestDeltaNetChunkBoundary(t *testing.T) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	whole := DeltaNetLayer(c, w, in.Vals, nTok, nil)

	const split = 3
	st := NewDeltaNetState(c)
	a := DeltaNetLayer(c, w, in.Vals, split, st)
	b := DeltaNetLayer(c, w, in.Vals[split*c.NEmbd:], nTok-split, st)

	joined := append(append([]float32(nil), a.Result...), b.Result...)
	r, err := compare(joined, whole.Result)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("linear_attn_out, %d+%d against %d: %v", split, nTok-split, nTok, r)
	if r.maxAbs != 0 {
		t.Errorf("a split batch is not bit-identical: %v", r)
	}
	if r, err = compare(b.NewState, whole.NewState); err != nil {
		t.Fatal(err)
	}
	t.Logf("new_state, %d+%d against %d: %v", split, nTok-split, nTok, r)
	if r.maxAbs != 0 {
		t.Errorf("the recurrent state is not bit-identical across the split: %v", r)
	}

	// And the control: the second chunk run *without* the first chunk's
	// state is a different answer, so the test above is not passing because
	// the state is ignored.
	cold := DeltaNetLayer(c, w, in.Vals[split*c.NEmbd:], nTok-split, nil)
	if r, err = compare(cold.Result, b.Result); err != nil {
		t.Fatal(err)
	}
	t.Logf("second chunk without the carried state: %v", r)
	if r.maxAbs == 0 {
		t.Error("carrying the state changes nothing, so the split test proves nothing")
	}
}

// TestDeltaNetL2NormIsTheBuilds is the one place in this vertical where the
// oracle is not merely imprecise but has since been called wrong.
//
// `build_gdn_l2_norm` divided by max(|x|, eps) until llama.cpp commit
// 5fdfa6282, "models : fix GDN normalization from `max` to `rsqrt`" (#28068),
// and the build behind this repo's trace — cff184438, which is also L1's
// `PPL 4.0340` and `tg128 25.15` — predates it. The two spellings differ by
// about 5e-07 relative at |x| near 1, which is nothing to the model and 15x
// the noise floor of this comparison, so the residual is explained rather
// than tolerated. If llama.cpp is ever rebuilt past that commit, this test
// fails and says exactly why.
func TestDeltaNetL2NormIsTheBuilds(t *testing.T) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get(fmt.Sprintf("k_conv_predelta-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	rms := map[GDNNorm]float64{}
	for _, mode := range []struct {
		name string
		n    GDNNorm
	}{{"max  (ggml_l2_norm, pre-#28068)", L2Max}, {"rsqrt (the model, post-#28068)", L2Rsqrt}} {
		c.QKNorm = mode.n
		got := DeltaNetLayer(c, w, in.Vals, nTok, nil)
		r, err := compare(got.KNorm, want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("k_conv_predelta under %-32s %v", mode.name, r)
		rms[mode.n] = r.rms
	}
	if rms[L2Max] > rms[L2Rsqrt]/5 {
		t.Errorf("the trace no longer prefers the max form (%.3e against %.3e): "+
			"llama.cpp has been rebuilt past 5fdfa6282 and the trace is stale",
			rms[L2Max], rms[L2Rsqrt])
	}
}
