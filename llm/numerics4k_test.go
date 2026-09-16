package llm

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// L4, the half that is not about QSA: what the reference's arithmetic becomes
// at a real ubatch.
//
// L3a-5 found the threshold — `ggml_vk_mul_mat` sends a matmul to the f32
// *vector* path at up to `mul_mat_vec_max_cols` = 8 output columns and to the
// coopmat GEMM above it — and drew the consequence that every tolerance in
// L2b, L2d, L2e and L3a was measured on a path no real prefill takes. The 4k
// dump is the first chance to see what is on the other side, and there are two
// answers rather than one. Both are about the *oracle* rather than about us.

// TestMatMulsAccumulateInF16AtAWideUbatch is the larger of the two, and it
// needs no arithmetic to state: at 4096 columns **every value the reference
// writes out of a quantised matmul is exactly representable as an IEEE half**.
//
// That is an fp16 accumulator, not an fp16 operand. `ggml_vk_mul_mat_q_f16`
// picks `.f16acc` over `.f32acc` whenever the precision is
// `GGML_PREC_DEFAULT`, the device has fp16 and its coopmat supports an fp16
// accumulator — all true here — and a K = 2560 dot product summed in halves
// carries about sqrt(K/16) roundings of 2^-11, which is the 0.45% relative
// disagreement the next test measures.
//
// The test is a sampled census rather than a claim about one tensor, because
// the conclusion is graph-wide: it holds for the DeltaNet's fused projection,
// the full-attention layer's, the MoE router, the shared expert and the output
// projection alike — every matmul in the model that has a quantised weight.
// The exceptions are as informative as the rule and are asserted too: the
// [10240, 4] `hc_inject` and the router keep an f32 accumulator, and the
// indexer's BF16 weights take the BF16 x BF16 kernel, which also does not.
func TestMatMulsAccumulateInF16AtAWideUbatch(t *testing.T) {
	_, tr := fixtures4k(t)

	// The f32/BF16-weighted matmuls, which are not on the fp16 path. `hc_inject`
	// is F32 and 4 columns wide, `shared_expert_gate` is F32 and one column
	// wide, `ffn_moe_logits` is the F32 router, and `indexer_k_raw` is BF16.
	notF16 := map[string]bool{
		"hc_inject-0": true, "hc_inject-3": true,
		"shared_expert_gate-0": true, "shared_expert_gate-3": true,
		"ffn_moe_logits-0": true, "ffn_moe_logits-3": true,
		"indexer_k_raw-3": true,
	}

	seen := 0
	for _, path := range tr.Entries {
		name := strings.TrimSuffix(filepath.Base(path), ".bin")
		if len(name) > 5 && name[4] == '_' {
			name = name[5:]
		}
		d, err := ReadDump(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if d.Op != "MUL_MAT" || len(d.Vals) == 0 {
			continue
		}
		seen++
		n, exact := 0, 0
		for i, v := range d.Vals {
			if i%97 != 0 { // a sample: these tensors are up to 200 MB
				continue
			}
			n++
			if f16Round(v) == v {
				exact++
			}
		}
		frac := float64(exact) / float64(n)
		t.Logf("%-22s %-18v %5.1f%% of %d sampled values are exactly fp16", name, d.NE[:2], 100*frac, n)
		switch {
		case notF16[name]:
			if frac > 0.9 {
				t.Errorf("%s: %.1f%% fp16-exact — it was expected to keep an f32 accumulator", name, 100*frac)
			}
		case frac < 0.999:
			t.Errorf("%s: only %.1f%% of its values are fp16-exact, so it is not accumulating in halves", name, 100*frac)
		}
	}
	if seen < 10 {
		t.Fatalf("only %d MUL_MAT tensors in the trace; the filter must have changed", seen)
	}
}

// TestF16AccumulatorSetsTheTolerance is the consequence, priced. No model of
// the *operands* moves it: exact f32, fp16 operands and the reference's own
// int8 activations all land within 6% of each other and all of them sit ~0.45%
// away from the reference, because the gap is the reference's own accumulator
// noise and not an approximation on our side.
//
// So the honest reading is the one L2c-3, L2f-6 and L3b-6 reached at 7 tokens,
// an order of magnitude larger and from a different cause: at a real ubatch
// **the reference is the side losing precision**, and a kernel of ours that
// accumulates in f32 cannot get closer to it than this. Every tolerance for a
// quantised projection at prefill is ~5e-3 rms, not the 1e-6 the 7-token dump
// suggested, and chasing the difference would mean reproducing a tile schedule
// rather than an arithmetic.
func TestF16AccumulatorSetsTheTolerance(t *testing.T) {
	c, w, _, _, _, tr, _ := qsaFixtures4k(t)
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("Qcur_full-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	// A prefix is enough for an rms over 12288 x 256 values, and the whole
	// tensor is 201 MB.
	const probe = 256

	run := func(name string, act Numerics, f16w bool) float64 {
		wt := w.Q
		if f16w {
			wt = f16Copy(w.Q)
		}
		out := make([]float32, probe*c.QWidth())
		parallel(probe, func(lo, hi int) {
			buf := make([]float32, c.NEmbd)
			for i := lo; i < hi; i++ {
				a := in.Vals[i*c.NEmbd : (i+1)*c.NEmbd]
				switch {
				case act == RefQ8:
					a = quantAct(a, buf, RefQ8)
				case f16w:
					for j, v := range a {
						buf[j] = f16Round(v)
					}
					a = buf
				}
				matvec(out[i*c.QWidth():(i+1)*c.QWidth()], wt, a, c.QWidth(), c.NEmbd)
			}
		})
		r, err := compare(out, want.Vals[:probe*c.QWidth()])
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Qcur_full-3, %-28s %v", name, r)
		return r.rms
	}
	exact := run("exact f32 operands", Exact, false)
	int8 := run("the reference's int8 activations", RefQ8, false)
	half := run("fp16 operands", Exact, true)

	lo, hi := exact, exact
	for _, v := range []float64{int8, half} {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	t.Logf("three operand models span %.3e..%.3e — %.2fx, against a %.3e gap to the reference",
		lo, hi, hi/lo, exact)
	if hi/lo > 1.2 {
		t.Errorf("the operand model matters by %.2fx; the fp16 accumulator was supposed to dominate it", hi/lo)
	}
	if exact > 1e-2 {
		t.Errorf("exact f32 is %.3e from the reference, which is past what an fp16 accumulator explains", exact)
	}
}

// TestIndexerProjectionIsBF16AtAWideUbatch is the smaller finding, and the one
// that changes our code rather than only our expectations.
//
// The indexer's two weights are the only BF16 tensors in this model. Above the
// 8-column threshold a BF16 src0 against an F32 src1 sets `y_non_contig`,
// which converts the **activation** to BF16 and runs a BF16 x BF16 kernel — so
// at any real ubatch these projections meet an activation with eight mantissa
// bits. Modelling it is worth 2061x, and the control that matters is that it
// is bf16 and *not* fp16: rounding to halves here is worse than not rounding
// at all, which is what makes this a different numeric from L2e-3's rather
// than the same one seen again.
func TestIndexerProjectionIsBF16AtAWideUbatch(t *testing.T) {
	c, w, nTok, _, _, tr, _ := qsaFixtures4k(t)
	if nTok <= vecMaxCols {
		t.Fatalf("%d tokens is on the vector side of the threshold", nTok)
	}
	in, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("indexer_k_raw-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	run := func(name string, round func([]float32) []float32) float64 {
		src := in.Vals
		if round != nil {
			src = round(src)
		}
		out := make([]float32, nTok*c.IdxDim)
		parallel(nTok, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				matvec(out[i*c.IdxDim:(i+1)*c.IdxDim], w.IdxK, src[i*c.NEmbd:(i+1)*c.NEmbd], c.IdxDim, c.NEmbd)
			}
		})
		r, err := compare(out, want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("indexer_k_raw-3, %-24s %v", name, r)
		return r.rms
	}
	f32 := run("f32 activation", nil)
	bf := run("bf16 activation", bf16Copy)
	f16 := run("fp16 activation", f16Copy)

	t.Logf("bf16 is %.0fx nearer the reference than f32, and fp16 is %.2fx *further* than f32",
		f32/bf, f16/f32)
	if bf > f32/100 {
		t.Errorf("bf16 gains only %.1fx over f32; the BF16 x BF16 kernel was supposed to dominate", f32/bf)
	}
	if f16 < f32 {
		t.Errorf("fp16 (%.3e) beats f32 (%.3e): this looks like L2e-3's fp16 path, not a BF16 one", f16, f32)
	}

	// And the other half of the rule: the checkpoint's BF16 weight is already
	// on that grid, so only the activation moves.
	off := 0
	for _, v := range w.IdxK {
		if bf16Round(v) != v {
			off++
		}
	}
	if off != 0 {
		t.Errorf("%d of %d indexer key weights are not exactly bf16", off, len(w.IdxK))
	}
}

// TestHalfPrecisionRoundTrips is the control the census above needs, and it
// needs no fixture: `f16Round` and `f16` are the two halves of one
// conversion, so every one of the 65 536 halves must widen to a float32 that
// rounds straight back to itself.
//
// It exists because the census found two bugs behind one symptom. 11 047 of
// `ffn_shexp-3`'s 10.5 M values looked not-quite-fp16 where every other
// quantised matmul in the graph was exactly fp16, and all of them were the
// smallest magnitudes and all of them negative: `f16Round` built a negative
// subnormal as `-0.0 + q*2^-24`, which is positive. Fixing that exposed the
// second — `f16` widened every subnormal half to **half** its value, an
// exponent field of 113+e where the arithmetic gives 114+e. Neither is
// reachable through a normal half, which is why five stages of tensor
// comparison never saw them.
func TestHalfPrecisionRoundTrips(t *testing.T) {
	subnormals, checked := 0, 0
	for u := 0; u < 1<<16; u++ {
		v := f16(uint16(u))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			continue
		}
		checked++
		if u&0x7c00 == 0 && u&0x03ff != 0 {
			subnormals++
		}
		if got := f16Round(v); got != v {
			t.Fatalf("half %#04x widens to %g, which rounds back to %g", u, v, got)
		}
	}
	// The smallest positive half is 2^-24 and the largest subnormal is
	// 1023*2^-24, stated as values rather than as bit patterns so that an
	// exponent off by one cannot pass.
	if got, want := f16(0x0001), float32(1)/(1<<24); got != want {
		t.Errorf("the smallest subnormal half is %g, want %g", got, want)
	}
	if got, want := f16(0x03ff), float32(1023)/(1<<24); got != want {
		t.Errorf("the largest subnormal half is %g, want %g", got, want)
	}
	if got := f16Round(-f16(0x0001)); got >= 0 {
		t.Errorf("f16Round of a negative subnormal is %g, which is not negative", got)
	}
	t.Logf("%d finite halves round-trip, %d of them subnormal", checked, subnormals)

	// bf16 is the same conversion one field narrower, and it has no
	// separate subnormal path — the top 16 bits of the float32 are the
	// value — so the round trip is a fixed-point check on the rounding.
	for u := 0; u < 1<<16; u++ {
		v := math.Float32frombits(uint32(u) << 16)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			continue
		}
		if got := bf16Round(v); got != v {
			t.Fatalf("bfloat %#04x widens to %g, which rounds back to %g", u, v, got)
		}
	}
}
