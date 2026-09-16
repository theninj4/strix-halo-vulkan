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
		{"ssm_alpha", g.ColAlpha(), c.NHeadV, proj(w.Alpha, c.NHeadV), 2e-3},
		{"ssm_beta", g.ColBeta(), c.NHeadV, proj(w.Beta, c.NHeadV), 2e-3},
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
				{"gate", g.Gate(), ref.Gate, 5e-2},
				{"beta_sigmoid", g.BetaSig(), ref.BetaSig, 5e-4},
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
	g, _, c, w, in, nTok, done := dnGPU(t, dnLayer)
	defer done()
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
