package dit

import (
	"fmt"
	"math"
	"testing"
)

// fp16GEMMRelTol is the bound the whole-block graph is held to. It is a
// different number from fp16RelTol for a structural reason rather than a
// tuning one: there, only attention's operands were narrowed and q, k and v
// arrived in fp32 from the host, while here all seven projections narrow
// *both* operands and each one's output is the next one's input.
//
// Measured against the diffusers fp32 reference at 320 tokens, error
// normalised by the tensor's RMS as everywhere else in this project:
//
//	modulation, attention_norm1 (fp32 paths)   1e-6 to 2e-6
//	q, v (the projections themselves)          1.2e-3 to 1.4e-3
//	q_normed, q_roped, k_roped                 1.6e-3 to 3.2e-3
//	attn_in (the fp16 narrowing itself)        7.6e-3
//	attn_ctx, attn_out, feed_forward           4.3e-3 to 1.5e-2
//	ffn_norm2, attention_norm2                 3.1e-2, 3.7e-2
//	out (the block)                            1.7e-2
//
// The bound is 8e-2, a little over twice the worst of them, which is the same
// rule stage 2 and stage 3c set theirs by.
//
// The two outliers are the RMS normalisation rather than the arithmetic, and
// it is worth being precise about why. attention_norm2's worst element is
// 0.0108 off a value of 5.757 -- 1.9e-3 of itself, about four fp16 quanta at
// that magnitude -- but the tensor's RMS is 0.295, because an RMS norm's
// output inherits the learned weight's dynamic range and most of this one's
// elements are twenty times smaller than its largest. Normalising by the RMS
// is still the right default (these activations cross zero constantly, so a
// per-element relative error is dominated by the values nearest zero), but on
// a tensor with that spread it reports a number twenty times the per-element
// one.
//
// What pins the other side is TestGPUBlockDetectsErrors, whose three
// breakages land at 1160x, 2828x and 15828x this bound, and
// TestGPUBlockKernelsAgree: all five GEMM builds -- three weight layouts, two
// tilings -- produce the same output to every digit logged, so the residue
// here follows the arithmetic and not the kernel.
const fp16GEMMRelTol = 8e-2

// TestGPUBlockAgainstDiffusers walks the whole block graph stage by stage
// against reference/dump_dit_block.py's dump, which is the rule PIPELINE.md
// sets for every stage: an end-to-end check says the block is wrong, this
// says which of its twenty-six dispatches made it wrong.
//
// Each stage re-runs the graph's prefix (GPUBlock.RunTo) rather than reading
// the arena after Apply, because most of these tensors do not survive to the
// end of the graph: the two block-level norms share one scratch tensor and
// four of the five norms run in place.
func TestGPUBlockAgainstDiffusers(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	m := f.m
	x := loadRef(t, m, "x")
	adaln := loadRef(t, m, "adaln_input")

	g, err := NewGPUBlock(dev, f.blk, f.rope, m.Seq, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	// The unfused graph: stage 4b's fused passes collapse seven dispatches
	// into three, and four of the tensors below are the fp32 intermediates
	// they stop materialising. TestGPUBlockFusedMatchesUnfused is what ties
	// the fused path back to this walk.
	g.Fused = false
	t.Logf("plan: %v", g.Plan())
	t.Logf("arenas: %d MB activations, %d MB weights, %d dispatches",
		g.ActivationBytes()>>20, g.WeightBytes()>>20, len(g.Labels()))

	dim := g.Dim()
	// stage is one comparison: run the graph up to a labelled dispatch, then
	// read the tensor it produced.
	stage := func(label, ref string, read func() *Mat, tol float64) {
		t.Helper()
		if err := g.RunTo(x, adaln.Data, label); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		compareTol(t, ref, read(), loadRef(t, m, ref), tol)
	}
	f32 := func(off func() uint32, cols int) func() *Mat {
		return func() *Mat { return g.Read(off(), cols) }
	}

	// The modulation, first: every later stage is scaled or gated by it, and
	// the checkpoint does not label its four chunks (TestValidationDetectsErrors
	// is the CPU side of the same hazard).
	if err := g.RunTo(x, adaln.Data, "adaln"); err != nil {
		t.Fatal(err)
	}
	mod := g.Mod()
	for i, name := range []string{"scale_msa", "gate_msa", "scale_mlp", "gate_mlp"} {
		got := &Mat{Rows: 1, Cols: dim, Data: mod[i*dim : (i+1)*dim]}
		compare(t, name, got, loadRef(t, m, name))
	}

	// Attention. The fp32 stages are held to the project-wide 2e-4; anything
	// downstream of a matrix-core GEMM to fp16GEMMRelTol.
	stage("rmsnorm x", "attention_norm1", f32(g.TensorNorm1, dim), relTol)
	stage("attn in", "attn_in", func() *Mat { return g.ReadF16(g.TensorA(), dim, g.LDA()) }, fp16GEMMRelTol)
	stage("gemm q", "q", f32(g.TensorQ, dim), fp16GEMMRelTol)
	stage("rmsnorm q", "q_normed", f32(g.TensorQ, dim), fp16GEMMRelTol)
	stage("rope q", "q_roped", f32(g.TensorQ, dim), fp16GEMMRelTol)
	stage("rope k", "k_roped", f32(g.TensorK, dim), fp16GEMMRelTol)
	stage("gemm v", "v", f32(g.TensorV, dim), fp16GEMMRelTol)
	stage("attention", "attn_ctx", f32(g.TensorCtx, dim), fp16GEMMRelTol)
	stage("gemm o", "attn_out", f32(g.TensorAttn, dim), fp16GEMMRelTol)
	stage("rmsnorm attn", "attention_norm2", f32(g.TensorAttn, dim), fp16GEMMRelTol)

	// Feed forward. "rmsnorm ffn" is the first thing downstream of the msa
	// residual, so it is also the check that the gated add landed.
	stage("rmsnorm ffn", "ffn_norm1", f32(g.TensorNorm1, dim), fp16GEMMRelTol)
	// w2's output is the one tensor in the graph that is not on the model's
	// own scale: the SwiGLU pass divides by GPUStack.FFScale so that the
	// product fits fp16, and nothing downstream undoes it because the next
	// thing that happens to it is an RMS norm. Here the reference is on the
	// model's scale, so the comparison undoes it -- and the stage after this
	// one, "rmsnorm ff", is then where the claim that the scale is invisible
	// is actually checked, since it is compared unscaled.
	stage("gemm w2", "feed_forward", scaled(f32(g.TensorFF, dim), 1/g.FFScale), fp16GEMMRelTol)
	stage("rmsnorm ff", "ffn_norm2", f32(g.TensorFF, dim), fp16GEMMRelTol)

	// And the block itself, through Apply rather than RunTo, so that what is
	// compared is the entry point and not the walk.
	out, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "out", out, loadRef(t, m, "out"), fp16GEMMRelTol)
}

// TestGPUBlockFusedMatchesUnfused is what lets stage 4b fuse at all.
//
// The stagewise walk needs the fp32 intermediates between a norm and its
// consumer, and fusing exists precisely to stop writing them. So the fused
// path is checked against the unfused one, which the walk has already checked
// against diffusers — and checked at the two points where the fusions' own
// output is still visible, before the fp16 GEMMs downstream have had a chance
// to amplify a difference into something that has to be argued about:
//
//	attn in    the fp16 A operand, which is norm+scale+narrow's whole output
//	attention  the context, which is everything the q/k fusion feeds
//
// The first is **bit-identical** — that fusion removes a round trip and
// changes nothing else. The second differs by 5.1e-4 of the tensor's RMS,
// which is the q/k fusion keeping the norm and the rotation in fp32 registers
// where the unfused path rounds to fp32 in the arena between them, and it is
// the only arithmetic difference between the two graphs. The block's output
// then differs by 1.6e-3, because an fp16 operand one ulp apart is a
// different GEMM input. The bound is 5e-3, which is where the measurement
// puts it and an order of magnitude below the reference comparison's.
func TestGPUBlockFusedMatchesUnfused(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")

	g, err := NewGPUBlock(dev, f.blk, f.rope, f.m.Seq, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	// The fp16 FFN is the one stage-4b change that is *not* arithmetic
	// preserving, so it is held out here and measured on its own in
	// TestGPUBlockFP16FFN.
	g.FP16FFN = false

	// Both paths up to a label, then the tensor that label produced.
	at := func(label string, read func() *Mat) (*Mat, *Mat) {
		t.Helper()
		g.Fused = false
		if err := g.RunTo(x, adaln.Data, label); err != nil {
			t.Fatal(err)
		}
		unfused := read()
		g.Fused = true
		if err := g.RunTo(x, adaln.Data, label); err != nil {
			t.Fatal(err)
		}
		return read(), unfused
	}

	const fuseTol = 5e-3
	a, aRef := at("attn in", func() *Mat { return g.ReadF16(g.TensorA(), g.Dim(), g.LDA()) })
	compareTol(t, "attn in", a, aRef, fuseTol)
	ctx, ctxRef := at("attention", func() *Mat { return g.Read(g.TensorCtx(), g.Dim()) })
	compareTol(t, "attn ctx", ctx, ctxRef, fuseTol)

	g.Fused = false
	unfused, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	nUnfused := len(g.Labels())
	g.Fused = true
	fused, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d dispatches fused against %d unfused", len(g.Labels()), nUnfused)
	compareTol(t, "block out", fused, unfused, fuseTol)
	compareTol(t, "fused vs diffusers", fused, loadRef(t, f.m, "out"), fp16GEMMRelTol)
}

// TestGPUBlockFP16FFN measures what §2.6's narrowing costs, rather than
// asserting it is free.
//
// The gate and up projections write fp16 instead of fp32, so SwiGLU's two
// inputs are rounded before it multiplies them. The question is not whether
// that changes the output -- it must -- but whether it changes it by more
// than the fp16 GEMMs upstream already do, and whether the values fit at all:
// stage 2 found this model's VAE could not hold its intermediates in fp16 at
// 1.16e7 against a 65504 limit, and TestFFNIntermediatesFitFP16 is why the
// same question is settled here before the kernel is built rather than after.
func TestGPUBlockFP16FFN(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")
	ref := loadRef(t, f.m, "out")

	g, err := NewGPUBlock(dev, f.blk, f.rope, f.m.Seq, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	g.FP16FFN = false
	fp32FFN, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	g.FP16FFN = true
	fp16FFN, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}

	base := compareTol(t, "fp32 ffn vs diffusers", fp32FFN, ref, fp16GEMMRelTol)
	got := compareTol(t, "fp16 ffn vs diffusers", fp16FFN, ref, fp16GEMMRelTol)
	maxAbs, rms, rel, worst := deviation(fp16FFN, fp32FFN)
	t.Logf("the narrowing alone: rel %.3g (max abs %.3g at %d, %.2g of that element), rms %.4g",
		rel, maxAbs, worst, maxAbs/math.Abs(float64(fp32FFN.Data[worst])), rms)
	if got > 2*base {
		t.Errorf("narrowing the FFN doubles the block's error against the reference: %.3g against %.3g",
			got, base)
	}
}

// TestGPUBlockMatchesCPU checks the graph against zimage/dit's own CPU
// implementation as well as against diffusers. The two references answer
// different questions: diffusers says the architecture is right, and the CPU
// block -- which is the same arithmetic in the same order as the graph, in
// fp32 -- says the *difference* is narrowing and not a wiring mistake, since
// a bug that both implementations share would pass the first check and a
// tolerance that is really hiding one would fail this.
func TestGPUBlockMatchesCPU(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")

	want, err := f.blk.Apply(x, adaln.Data, f.rope)
	if err != nil {
		t.Fatal(err)
	}

	g, err := NewGPUBlock(dev, f.blk, f.rope, f.m.Seq, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	got, err := g.Apply(x, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "block vs cpu", got, want, fp16GEMMRelTol)
}

// TestGPUBlockKernelsAgree runs every GEMM build in the ladder over the whole
// block and holds each to the same bound.
//
// The point is not the timing -- cmd/ditblock does that -- but that the five
// builds are five *layouts and tilings of the same arithmetic*: they read the
// weight out of three different arrangements of memory, so a packer that
// transposes an index or a kernel that walks tiles in the wrong order shows
// up here as one row disagreeing with the other four rather than as a slow
// image later.
func TestGPUBlockKernelsAgree(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")
	ref := loadRef(t, f.m, "out")

	for _, k := range GEMMKernels() {
		g, err := NewGPUBlock(dev, f.blk, f.rope, f.m.Seq, UniformGEMMPlan(k))
		if err != nil {
			t.Fatalf("%s: %v", k, err)
		}
		out, err := g.Apply(x, adaln.Data)
		if err != nil {
			g.Destroy()
			t.Fatalf("%s: %v", k, err)
		}
		compareTol(t, "out "+string(k), out, ref, fp16GEMMRelTol)
		g.Destroy()
	}
}

// TestGPUBlockDetectsErrors is the negative control, and without it the test
// above proves only that the graph runs.
//
// The three breakages are the ones this stage actually invites. adaLN's four
// chunks are unlabelled in the checkpoint, so scale and gate are
// interchangeable to anyone reading the tensor names. A graph of twenty-six
// dispatches can lose one and still produce a plausible tensor. And the new
// machinery here is the *weight layout*: three arrangements of the same
// numbers, chosen per kernel, where staging the wrong one is a silent
// mismatch rather than an error.
func TestGPUBlockDetectsErrors(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")
	ref := loadRef(t, f.m, "out")

	for _, c := range []struct {
		name string
		ctl  blockControls
		plan GEMMPlan
	}{
		{"modulation chunks swapped", blockControls{swapModulation: true}, nil},
		{"attention norm dropped", blockControls{dropNorm2: true}, nil},
		// The tiled kernel, handed a weight staged in the natural [N, K+pad]
		// layout. Both are the same 14.7 M numbers; only the order differs.
		{"weight staged in the wrong layout", blockControls{wrongBLayout: true}, UniformGEMMPlan(GEMMReg64HKA4Tiled)},
	} {
		g, err := newGPUBlock(dev, f.blk, f.rope, f.m.Seq, c.plan, c.ctl)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		out, err := g.Apply(x, adaln.Data)
		g.Destroy()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		_, rms, rel, _ := deviation(out, ref)
		if rel <= fp16GEMMRelTol {
			t.Errorf("%s: rel %.3g is inside the %.0e bound -- the check cannot see this break",
				c.name, rel, fp16GEMMRelTol)
			continue
		}
		t.Logf("%-34s rel %.3g = %.0fx the bound (rms %.4g)", c.name, rel, rel/fp16GEMMRelTol, rms)
	}
}

// TestGPUBlockFFScale checks the SwiGLU headroom, which is the one thing in
// the graph that changes a tensor's value on purpose.
//
// The claim it rests on is that w2's output is read by an RMS norm and by
// nothing else, so a positive scale on the SwiGLU output cannot reach the
// block's output. That is exactly the kind of claim that is true when it is
// written and false three commits later, and it is load-bearing: without the
// scale a real prompt overflows fp16 in a few elements of a few million, one
// infinity makes a whole row of w2's output a NaN, and eight denoising steps
// later the image is black.
//
// So: the scale moves the intermediate by the factor it says and does not move
// the block's output. Three decades of it, because what would break the
// invariance is the norm's epsilon, and that can only matter once the scale is
// small enough for a row's mean square to approach it -- at 1/256 the mean
// square is 65536x smaller and the bound still holds by four orders.
//
// The test also reports the headroom the default buys on this fixture, which
// is the number the default was chosen against.
func TestGPUBlockFFScale(t *testing.T) {
	f := loadFixture(t)
	defer f.close()
	dev, done := newTestDevice(t)
	defer done()

	x := loadRef(t, f.m, "x")
	adaln := loadRef(t, f.m, "adaln_input")

	g, err := NewGPUBlock(dev, f.blk, f.rope, f.m.Seq, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	// swiglu is w2's fp16 A operand, which is what the scale is protecting.
	swiglu := func() *Mat { return g.ReadF16(g.TensorHFFN(), g.FFN(), g.LDAFFN()) }
	run := func(scale float64) (ff, out *Mat, peak float64) {
		t.Helper()
		g.FFScale = scale
		if err := g.RunTo(x, adaln.Data, "gemm w2"); err != nil {
			t.Fatal(err)
		}
		ff = g.Read(g.TensorFF(), g.Dim())
		for _, v := range swiglu().Data {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		out, err = g.Apply(x, adaln.Data)
		if err != nil {
			t.Fatal(err)
		}
		return ff, out, peak
	}
	defer func() { g.FFScale = defaultFFScale }()

	baseFF, baseOut, peak := run(1)
	t.Logf("SwiGLU peak unscaled %.4g, at the default 1/%g %.4g -- %.0fx under fp16's 65504",
		peak, 1/defaultFFScale, peak*defaultFFScale, 65504/(peak*defaultFFScale))

	// 1e-3 is where the fp16 rounding of a rescaled operand lands; the
	// reference comparison's own bound is 8e-2, so this is two orders tighter
	// than the level at which the scale could hide.
	const scaleTol = 1e-3
	for _, s := range []float64{1.0 / 4, defaultFFScale, 1.0 / 256} {
		ff, out, _ := run(s)
		want := baseFF.Clone()
		for i := range want.Data {
			want.Data[i] *= float32(s)
		}
		compareTol(t, fmt.Sprintf("feed_forward x%g", s), ff, want, scaleTol)
		compareTol(t, fmt.Sprintf("out x%g", s), out, baseOut, scaleTol)
	}
}

// scaled multiplies a read tensor by a constant, for the one stage whose
// device tensor is deliberately not on the model's scale.
func scaled(read func() *Mat, by float64) func() *Mat {
	return func() *Mat {
		m := read()
		for i := range m.Data {
			m.Data[i] *= float32(by)
		}
		return m
	}
}
