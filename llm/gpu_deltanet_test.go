package llm

import (
	"fmt"
	"math"
	"testing"
)

// The GPU half of L3, checked the way L2c, L2d and L2f were: against the CPU
// reference in deltanet.go first, because that is what the kernels are a port
// of, and against llama.cpp's own tensors behind it, because that is what the
// CPU reference was built against.
//
// Layers 0, 1 and 2 are the linear-attention layers the trace covers — the
// full-attention interval is 4, so layer 3 is L2f's — and `hc_mixed-N` is
// llama.cpp's own value for each layer's input, so every comparison is that
// layer's own rather than an accumulation of L2's.

// dnGPU stages one layer and returns it over the trace's own input.
func dnGPU(t *testing.T, layer int) (*DeltaNetGPU, *Trace, DeltaNetConfig, DeltaNetWeights, []float32, int, func()) {
	t.Helper()
	tr, c, w, nTok := dnFixtures(t, layer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", layer), 0)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	g, err := NewDeltaNetGPU(dev, c, nTok, []DeltaNetWeights{w}, denseQ8Test)
	if err != nil {
		done()
		t.Fatal(err)
	}
	g.KeepSilu(true)
	t.Logf("staged: %.1f MB of weights, %.1f MB of arenas, %d tokens",
		float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6, nTok)
	return g, tr, c, w, in.Vals, nTok, func() { g.Destroy(); done() }
}

// TestDeltaNetGPUFusedProjection is the gate on the layout the whole layer
// rests on: four of llama.cpp's matrices packed as column ranges of one.
//
// Nothing downstream can be right if the ranges are wrong and nothing
// downstream says so clearly — the value half read at the key's columns is a
// perfectly plausible tensor, and alpha read one column late is 47 heads of
// somebody else's decay. So each range is compared against the reference's
// own matvec over the same input.
func TestDeltaNetGPUFusedProjection(t *testing.T) {
	g, tr, c, w, in, nTok, done := dnGPU(t, dnLayer)
	defer done()

	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

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
		{"attn_qkv", g.ColQ(), c.ConvWidth(), proj(w.QKV, c.ConvWidth()), 2e-3},
		{"attn_gate", g.ColZ(), c.Inner, proj(w.Z, c.Inner), 2e-3},
		{"ssm_alpha", g.ColAlpha(), c.NHeadV, proj(w.Alpha, c.NHeadV), bankTol(2e-3, 5e-3)},
		{"ssm_beta", g.ColBeta(), c.NHeadV, proj(w.Beta, c.NHeadV), bankTol(2e-3, 5e-3)},
	} {
		r, err := compare(g.Column(tc.col, tc.width), tc.want)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%-10s col %5d  vs CPU f32  %v", tc.name, tc.col, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
	}

	// And the one of the four the trace names directly.
	want, err := tr.Get(fmt.Sprintf("linear_attn_qkv_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Column(g.ColQ(), c.ConvWidth()), want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	// A looser bound than the CPU comparison above, and for a reason worth
	// stating: our kernel and llama.cpp's are two *different* approximations
	// of the same f32 matmul — fp16 operands with f32 accumulators on one
	// side, Q8_0 weights against int8-per-block activations on the other — so
	// this row carries both errors and neither is the other's.
	t.Logf("%-10s            vs llama.cpp %v", "attn_qkv", r)
	if r.rms > 8e-3 {
		t.Errorf("linear_attn_qkv_mixed: rms %.3e (%v)", r.rms, r)
	}
}

// TestDeltaNetGPULayer is L3b's gate: every tensor of the layer, against the
// CPU reference L3a checked against llama.cpp.
//
// The tolerances are fp16's and not f32's, and deliberately so. The kernels
// evaluate five matmuls on the fp16 matrix cores where the reference — at
// *this* prompt length — used the f32 vector path (L3a-5), so the projections
// come in around 1e-4 relative and everything downstream of them inherits it.
// What the test is really asserting is that nothing beyond that is happening:
// the recurrence, the convolution's history, the head map and the two norm
// spellings are all exactly the reference's.
func TestDeltaNetGPULayer(t *testing.T) {
	for _, layer := range []int{0, 1, 2} {
		t.Run(fmt.Sprintf("layer%d", layer), func(t *testing.T) {
			g, tr, c, w, in, nTok, done := dnGPU(t, layer)
			defer done()
			if err := g.Upload(in, nTok); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(0); err != nil {
				t.Fatal(err)
			}
			// The CPU reference from the same input, with the oracle's own
			// numerics on the Q8_0 projections and the fp16 path on the two
			// F32 ones — which is what the fused weight puts them on.
			cpu := c
			cpu.F32MM = RefQ8
			ref := DeltaNetLayer(cpu, w, in, nTok, nil)
			qk, vw := c.QKWidth(), c.VWidth()

			for _, tc := range []struct {
				name string
				got  []float32
				cpu  []float32
				tol  float64
			}{
				{"conv_output_silu", g.ConvSilu(), ref.ConvSilu, 2e-3},
				{"q_conv_predelta", g.NormColumn(g.ColQ(), qk), ref.QNorm, 5e-4},
				{"k_conv_predelta", g.NormColumn(g.ColK(), qk), ref.KNorm, 5e-4},
				{"v_conv_predelta", g.NormColumn(g.ColV(), vw), ref.V, 2e-3},
				{"gate", g.Gate(), ref.Gate, bankTol(5e-2, 8e-2)},
				{"beta_sigmoid", g.BetaSig(), ref.BetaSig, bankTol(5e-4, 2e-3)},
				{"attn_output", g.Out(), ref.Out, 2e-4},
				{"final_output", g.Final(), ref.Final, 2e-3},
				{"linear_attn_out", g.Result(), ref.Result, 2e-3},
			} {
				r, err := compare(tc.got, tc.cpu)
				if err != nil {
					t.Errorf("%s: %v", tc.name, err)
					continue
				}
				t.Logf("%-17s vs CPU  %v", tc.name, r)
				if r.rms > tc.tol {
					t.Errorf("%s: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
				}
			}

			// And the same tensors against llama.cpp itself, which is where
			// the CPU reference's own residual joins ours.
			for _, tc := range []struct {
				name string
				got  []float32
				tol  float64
			}{
				{"conv_output_silu", g.ConvSilu(), 2e-3},
				{"q_conv_predelta", g.NormColumn(g.ColQ(), qk), 5e-4},
				{"k_conv_predelta", g.NormColumn(g.ColK(), qk), 5e-4},
				{"attn_output", g.Out(), 2e-4},
				{"linear_attn_out", g.Result(), 2e-3},
			} {
				want, err := tr.Get(fmt.Sprintf("%s-%d", tc.name, layer), 0)
				if err != nil {
					t.Errorf("%s: %v", tc.name, err)
					continue
				}
				r, err := compare(tc.got, want.Vals)
				if err != nil {
					t.Errorf("%s: %v", tc.name, err)
					continue
				}
				t.Logf("%-17s vs llama.cpp %v", tc.name, r)
				if r.rms > tc.tol {
					t.Errorf("%s vs llama.cpp: rms %.3e over %.1e (%v)", tc.name, r.rms, tc.tol, r)
				}
			}
		})
	}
}

// TestDeltaNetGPUGateIsTheFp16Path prices the one deviation the fused
// projection introduces, rather than leaving it inside a tolerance.
//
// `ssm_alpha` and `ssm_beta` are F32 weights read by an F32 activation, and at
// the dump's 7 tokens llama.cpp evaluates them on the f32 *vector* path —
// `ggml_vk_mul_mat` takes it up to `mul_mat_vec_max_cols = 8` output columns
// and the fp16 coopmat GEMM above it (L3a-5). Our fused weight has 16512
// columns, so ours is always the second, and so is llama.cpp's at any real
// ubatch. The deviation is therefore from *this dump*, not from the reference.
//
// It matters more than its magnitude suggests, because `gate` is a **log**
// decay: an absolute error on alpha is multiplied by |ssm_a|, which reaches 23
// in this layer, before exp() turns it into a per-token forgetting factor. So
// the number this test reports is the worst ratio between the two decays, and
// what it asserts is that our kernel is an fp16 matmul of the same quality as
// the reference's would be — not that it reproduces one particular fp16
// evaluation order, which no two implementations share.
func TestDeltaNetGPUGateIsTheFp16Path(t *testing.T) {
	// The fp16 bank explicitly, whatever the environment says: this test
	// prices the fp16 *path*, and on a quantised bank alpha and beta go
	// through int8 as well (P2), which is a different question — the corpus's
	// (D13: +0.01%), not this test's.
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in0, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
	if err != nil {
		t.Fatal(err)
	}
	in := in0.Vals
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewDeltaNetGPU(dev, c, nTok, []DeltaNetWeights{w}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}

	exact, fp16 := c, c
	exact.F32MM = Exact
	fp16.F32MM = RefQ8
	refExact := DeltaNetLayer(exact, w, in, nTok, nil)
	refFp16 := DeltaNetLayer(fp16, w, in, nTok, nil)

	// What the f32 path costs, measured on the CPU where both are available:
	// the two references differ only in whether alpha and beta go through
	// fp16 operands.
	refGap, err := compare(refFp16.Gate, refExact.Gate)
	if err != nil {
		t.Fatal(err)
	}
	rExact, err := compare(g.Gate(), refExact.Gate)
	if err != nil {
		t.Fatal(err)
	}
	rFp16, err := compare(g.Gate(), refFp16.Gate)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("CPU fp16 vs CPU f32        %v", refGap)
	t.Logf("GPU      vs CPU f32        %v", rExact)
	t.Logf("GPU      vs CPU fp16       %v", rFp16)

	// The consequence, in the units the recurrence uses. exp(gate) is the
	// factor the whole state is multiplied by once per token, so the worst
	// ratio is what a 512-token prefill would compound.
	var worst float64
	for i := range refExact.Gate {
		a, b := float64(refExact.Gate[i]), float64(g.Gate()[i])
		if d := math.Abs(math.Exp(b-a) - 1); d > worst {
			worst = d
		}
	}
	t.Logf("worst per-token decay ratio: %.3f%%", 100*worst)

	// Three evaluations of the same matmul, two of them in fp16: the two fp16
	// ones must be the same distance from f32 to within a small factor, or
	// ours is doing something the reference's would not.
	if rExact.rms > 4*refGap.rms {
		t.Errorf("the GPU is %.3e rms from the f32 model where a CPU fp16 evaluation is %.3e; "+
			"that is more than an fp16 matmul should cost", rExact.rms, refGap.rms)
	}
	if refGap.rms == 0 {
		t.Errorf("the two CPU references agree exactly, so F32MM is not selecting anything")
	}
}

// TestDeltaNetGPUScanLadderAgrees is the negative control on the one thing
// the scan's ladder could get wrong.
//
// LPC decomposes a [128, 128] state over lanes and workgroups eleven
// different ways — 2 registers a lane and 6144 workgroups at one end, 64 and
// 48 at the other, with a cross-lane reduction whose cluster width changes on
// every rung. Every one of them has to compute the same recurrence, and a
// wrong cluster width or a wrong column map produces a plausible tensor
// rather than a broken one. So every rung is run from the same state and
// compared against the rung the default names.
func TestDeltaNetGPUScanLadderAgrees(t *testing.T) {
	g, _, c, _, in, nTok, done := dnGPU(t, dnLayer)
	defer done()

	var want []float32
	for _, k := range DNKernels() {
		if err := g.SetPlan(k, GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("%s: %v", k, err)
		}
		got := g.Out()
		if want == nil {
			want = got
			t.Logf("%-6s is the reference rung", k)
			continue
		}
		r, err := compare(got, want)
		if err != nil {
			t.Fatalf("%s: %v", k, err)
		}
		regs, wgs, reread := DNShape(k, c)
		arm := "LDS"
		if DNOperandsInRegisters(k) {
			arm = "reg"
		}
		t.Logf("%-6s %3d regs/lane, %4d workgroups, q/k %s read %3dx  %v",
			k, regs, wgs, arm, reread, r)
		// The reductions run in different orders, so this is f32 round-off
		// and not bit-identity — but it is round-off on a 512-long chain of
		// them, so the bound is still tight.
		if r.rms > 1e-6 {
			t.Errorf("%s: rms %.3e against the first rung (%v)", k, r.rms, r)
		}
	}
}

// TestDeltaNetGPUStateCarries is L3a-6's test on the device: 3 + 4 tokens
// have to reproduce 7, output and state both.
//
// Two things cross the boundary and only one of them is obvious. The
// [128, 128, 48] recurrent state is the one a reading of the architecture
// predicts; the convolution's three-column window is the one that is easy to
// forget, and a kernel that carries the first and drops the second is wrong
// in a way nothing else here sees — at decode, where every token is its own
// batch, it would be wrong on every token. The control is the same run with
// the state reset between the halves.
func TestDeltaNetGPUStateCarries(t *testing.T) {
	g, _, c, _, in, nTok, done := dnGPU(t, dnLayer)
	defer done()
	if nTok < 4 {
		t.Skipf("the trace has %d tokens, which does not split", nTok)
	}
	split := nTok / 2

	if err := g.Reset(0); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPast(0); err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(in, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	whole := append([]float32(nil), g.Result()...)
	wholeState, err := g.State(0)
	if err != nil {
		t.Fatal(err)
	}

	// The same prompt in two batches, with only what DeltaNetState holds
	// carried between them.
	if err := g.Reset(0); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPast(0); err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(in[:split*c.NEmbd], split); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	// The window moves on the device now (L7b): the layer's last dispatch
	// leaves its Conv-1 rows in the layer's own ring, and the only thing the
	// caller says is where the next run starts.
	if err := g.SetPast(split); err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(in[split*c.NEmbd:], nTok-split); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Result(), whole[split*c.NEmbd:])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d + %d against %d, output  %v", split, nTok-split, nTok, r)
	if r.rms != 0 {
		t.Errorf("the second batch is %.3e from the whole run, want bit-identical (%v)", r.rms, r)
	}
	splitState, err := g.State(0)
	if err != nil {
		t.Fatal(err)
	}
	rs, err := compare(splitState.S, wholeState.S)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d + %d against %d, state   %v", split, nTok-split, nTok, rs)
	if rs.rms != 0 {
		t.Errorf("the carried state is %.3e from the whole run's (%v)", rs.rms, rs)
	}

	// The control: without the convolution's window, the second batch is
	// wrong — and wrong in a way that only the first three tokens of it show,
	// which is why it needs a number rather than an eyeball.
	if err := g.Reset(0); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPast(0); err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(in[:split*c.NEmbd], split); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	st, err := g.State(0)
	if err != nil {
		t.Fatal(err)
	}
	st.Conv = make([]float32, len(st.Conv)) // the recurrent state, no window
	if err := g.SetPast(split); err != nil {
		t.Fatal(err)
	}
	if err := g.SetState(0, st); err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(in[split*c.NEmbd:], nTok-split); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0); err != nil {
		t.Fatal(err)
	}
	rc, err := compare(g.Result(), whole[split*c.NEmbd:])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("control, window dropped     %v", rc)
	if rc.rms < 1e-3 {
		t.Errorf("dropping the convolution's window moves the output by only %.3e, "+
			"so this test would not have caught a kernel that forgot it", rc.rms)
	}
}

// TestDeltaNetGPUGemvAgrees is L8d-4's gate: llm_gemv.comp against
// llm_gemm.comp MODE 2 on both of this layer's projections, at the one token
// the GEMV exists for.
//
// It is two claims and they need different bounds. The **fused projection**
// is the one with a tail — alpha and beta are the checkpoint's only F32
// matrices here and stay halves at `gateOff`, numbered from `lowRank/16` —
// so this is also the check that the GEMV derives the same split the GEMM
// does, and a column read out of the wrong plane is not a tolerance but 47
// heads of somebody else's decay. The **output projection** has no tail.
//
// It is a tolerance and not an equality at every rung, including KSLABS = 1.
// The *weights* are the same halves — L8b-2's argument, and the reason this
// kernel multiplies in fp16 rather than converting — but a cooperative-matrix
// accumulator sums sixteen k an instruction in an order the extension does
// not define, where a lane sums them serially, so the two differ by f32
// round-off over a 2560- or 6144-long chain. What makes that a *measurement*
// rather than an alibi is that it does not move with the rung: every split
// lands within 0.2% of the same rms, where a mis-read scale plane or a tail
// taken from the wrong row would be a different number at every one.
func TestDeltaNetGPUGemvAgrees(t *testing.T) {
	g, _, c, _, in, nTok, done := dnGPU(t, dnLayer)
	defer done()
	if nTok < 1 {
		t.Skip("the trace has no tokens")
	}
	scan := DefaultDNKernel()
	run := func(qkv, out GEMVKernel) ([]float32, []float32) {
		t.Helper()
		if err := g.SetPlan(scan, GEMMKernelFor(1), OutGEMMKernelFor(1)); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[:c.NEmbd], 1); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Result()...)
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
		// And alpha and beta's columns on their own. Folded into the
		// whole-matrix rms above they are 0.6% of the columns and would
		// hide completely.
		tail := g.ColAlpha()
		rt, err := compare(gotQKV[tail:], refQKV[tail:])
		if err != nil {
			t.Fatalf("qkv %s alpha/beta: %v", k, err)
		}
		t.Logf("    alpha and beta (%d rows from %d): %v", len(refQKV)-tail, tail, rt)
		if rt.rms > 1e-4 {
			t.Errorf("qkv %s: alpha and beta are rms %.3e from the GEMM's (%v)", k, rt.rms, rt)
		}
	}
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.Inner) {
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

	// The negative control: the rungs that cannot cut K into whole four-tile
	// steps have to be refused rather than run short.
	for _, k := range GEMVKernels() {
		if GEMVFits(k, c.NEmbd) {
			continue
		}
		if err := g.SetGemv(k, GEMVOff); err == nil {
			t.Errorf("qkv %s was accepted for K = %d, which it does not divide into whole steps", k, c.NEmbd)
		}
	}
}

// TestDeltaNetGPUQ4IsTheSim is L8c-5's first gate, and it is the same shape
// as the lm head's: **the bank against the simulation that chose it**, value
// for value, over a whole layer rather than one matmul.
//
// L8c-3's +4.24% is a perplexity measured through `sim.go`, which stages a
// candidate format's floats and lets the fp16 kernels multiply them. This
// bank stores that format instead, through the same encoder — so the halves
// the matrix cores see are the same halves, the fragment order is the same,
// and the two arms have to agree exactly. They are compared at the fused
// projection, where a wrong address would show first, and at the layer's
// output, where the convolution, the recurrence and the second projection
// have all run on top of it.
//
// It is `rtn` rather than `imatrix` so the test needs nothing but the
// checkpoint; the calibrated arm is the same code path with `qw` non-nil, and
// TestBankQ4KIsTheSim covers that at the encoder.
//
// Both widths, since P3a: the `qh` plane is the same code in every block and
// a plane index that is right here and wrong in one other is exactly what a
// per-block gate catches.
func TestDeltaNetGPUQ4IsTheSim(t *testing.T) {
	for _, spec := range []string{"q4_k/32", "q5_k/32"} {
		t.Run(spec, func(t *testing.T) { dnBankIsTheSim(t, spec) })
	}
}

func dnBankIsTheSim(t *testing.T, spec string) {
	tr, c, w, nTok := dnFixtures(t, dnLayer)
	in, err := tr.Get(fmt.Sprintf("hc_mixed-%d", dnLayer), 0)
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

	run := func(bank DenseBank) ([]float32, []float32, int) {
		t.Helper()
		g, err := NewDeltaNetGPUBank(dev, c, nTok, []DeltaNetWeights{w}, bank, sim)
		if err != nil {
			t.Fatalf("deltanet (%s): %v", bank, err)
		}
		defer g.Destroy()
		if err := g.Upload(in.Vals, nTok); err != nil {
			t.Fatalf("upload (%s): %v", bank, err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("run (%s): %v", bank, err)
		}
		return append([]float32(nil), g.Column(g.ColQ(), g.ColBeta()+c.NHeadV)...),
			append([]float32(nil), g.Result()...), g.WeightBytes()
	}

	simQKV, simOut, simBytes := run(BankFP16)
	q4QKV, q4Out, q4Bytes := run(bank)

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
}

// TestDeltaNetGPUQ4BankSize states what L8c-5 stages against what it
// replaces, and where the 4.500 bits stop being exact: the plane covers qkvN
// rows, of which the last few are the column block's padding — staged
// nibbles nothing reads — so the matrix as staged is a little over 4.5 bits
// and the test says how much rather than asserting a round number.
func TestDeltaNetGPUQ4BankSize(t *testing.T) {
	_, c, _, _ := dnFixtures(t, dnLayer)
	real := c.ConvWidth() + c.Inner + 2*c.NHeadV
	n, k := roundUpInt(real, dnBN), c.NEmbd
	pad := n - real

	q4 := q8Align(qkBytes(4, n, k)) + q8Align(qkBytes(4, c.NEmbd, c.Inner))
	q8 := q8Align(q8Bytes(n, k)) + q8Align(q8Bytes(c.NEmbd, c.Inner))
	half := n*k*2 + c.NEmbd*c.Inner*2
	weights := real*k + c.NEmbd*c.Inner

	t.Logf("a layer: q4_k %.2f MB, q8 %.2f MB, halves %.2f MB — %.3f, %.3f and %.3f bits a weight",
		float64(q4)/1e6, float64(q8)/1e6, float64(half)/1e6,
		float64(q4)*8/float64(weights), float64(q8)*8/float64(weights),
		float64(half)*8/float64(weights))
	t.Logf("36 layers: %.2f GB against %.2f GB and %.2f GB",
		float64(36*q4)/1e9, float64(36*q8)/1e9, float64(36*half)/1e9)
	if q4 >= q8 || q8 >= half {
		t.Fatalf("the three banks are %d, %d, %d bytes and are not in order", q4, q8, half)
	}
	// Subtract the pad rows' nibbles and state the remainder rather than
	// demanding 4.500 — the record plane covers the pad rows too.
	body := qkBytes(4, n, k) - pad*k/2 + qkBytes(4, c.NEmbd, c.Inner)
	if bits := float64(body) * 8 / float64(weights); bits < 4.5 || bits > 4.52 {
		t.Fatalf("the staged matrix is %.4f bits a weight, want 4.500 plus the padding's records", bits)
	}
}

// TestDeltaNetGPUGemvTwoRows is P5b's gate on this block: the decode GEMV
// carrying **two** rows of A computes what the GEMM computes.
//
// D15 refused the rung above one token until P5b, on the grounds that a
// sixteen-row fragment holding one row is the reason the GEMV wins — which is
// a fact about the GEMM's shape and not about a dot product's. The R-row arm
// reads the weight once and multiplies it into two accumulators, so the
// tolerance is the same one the one-row test uses and for the same reason:
// the two paths associate a 2560- or 6144-long sum differently (PinSchedule),
// they do not read different weights.
//
// The negative control is in the assertion rather than in a second arm: row 1
// is a *different* token, so a kernel that ignored `ROWS` and wrote row 0's
// answer twice would fail the comparison against the GEMM's second row.
func TestDeltaNetGPUGemvTwoRows(t *testing.T) {
	g, _, c, _, in, nTok, done := dnGPU(t, dnLayer)
	defer done()
	if nTok < 2 {
		t.Skip("the trace has fewer than two tokens")
	}
	const rows = 2
	if rows > GEMVMaxRows {
		t.Skipf("GEMVMaxRows is %d", GEMVMaxRows)
	}
	scan := DefaultDNKernel()
	run := func(qkv, out GEMVKernel) ([]float32, []float32) {
		t.Helper()
		if err := g.SetPlan(scan, GEMMKernelFor(rows), OutGEMMKernelFor(rows)); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(0); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[:rows*c.NEmbd], rows); err != nil {
			t.Fatal(err)
		}
		if err := g.SetGemv(qkv, out); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), g.QKV()...), append([]float32(nil), g.Result()...)
	}

	// The row-1 control: a second token that is a *different* token, so a
	// rung that does not write row 1 is caught rather than flattered. See
	// rowMoved.
	second := func(tok int, qkv, out GEMVKernel) []float32 {
		t.Helper()
		buf := make([]float32, rows*c.NEmbd)
		copy(buf, in[:c.NEmbd])
		copy(buf[c.NEmbd:], in[tok*c.NEmbd:(tok+1)*c.NEmbd])
		if err := g.SetPlan(scan, GEMMKernelFor(rows), OutGEMMKernelFor(rows)); err != nil {
			t.Fatal(err)
		}
		if err := g.Reset(0); err != nil {
			t.Fatal(err)
		}
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
		r, err := compare(gotQKV, refQKV)
		if err != nil {
			t.Fatalf("qkv %s: %v", k, err)
		}
		t.Logf("qkv %-4s at %d rows against the GEMM over %d values: %v", k, rows, len(refQKV), r)
		if r.rms > 1e-4 {
			t.Errorf("qkv %s at %d rows: rms %.3e against the GEMM (%v)", k, rows, r.rms, r)
		}
		// The second row on its own. Folded into the whole-matrix rms it is
		// half the values, but a kernel that wrote row 0 twice would still
		// show as a large rms only if the two tokens differ a lot — so the
		// row is compared against its own reference explicitly.
		w := g.qkvN()
		r1, err := compare(gotQKV[w:], refQKV[w:])
		if err != nil {
			t.Fatalf("qkv %s row 1: %v", k, err)
		}
		if r1.rms > 1e-4 {
			t.Errorf("qkv %s row 1: rms %.3e against the GEMM (%v)", k, r1.rms, r1)
		}
	}
	for _, k := range GEMVKernels() {
		if !GEMVFits(k, c.Inner) {
			continue
		}
		_, gotOut := run(GEMVOff, k)
		r, err := compare(gotOut, refOut)
		if err != nil {
			t.Fatalf("out %s: %v", k, err)
		}
		t.Logf("out %-4s at %d rows against the GEMM over %d values: %v", k, rows, len(refOut), r)
		if r.rms > 1e-4 {
			t.Errorf("out %s at %d rows: rms %.3e against the GEMM (%v)", k, rows, r.rms, r)
		}
		r1, err := compare(gotOut[c.NEmbd:], refOut[c.NEmbd:])
		if err != nil {
			t.Fatalf("out %s row 1: %v", k, err)
		}
		if r1.rms > 1e-4 {
			t.Errorf("out %s row 1: rms %.3e against the GEMM (%v)", k, r1.rms, r1)
		}
	}
}
