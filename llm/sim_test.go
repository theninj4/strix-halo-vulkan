package llm

// L8c-1's instrument checked against a case whose answer is already known.
//
//	go test ./llm/ -v -run TestQuantSim
//
// A width simulation is only worth a perplexity run if the halves it stages
// are the ones a real kernel would multiply, and that claim has a free test:
// **at q8sym/32 over a Q8_0 tensor the simulation must be an identity.** D13
// is why — ggml picks `d = amax/127`, so re-deriving `(d, q)` from the
// dequantised floats returns the checkpoint's own pair, and `fp16(q*d)` is
// then exactly what `tileB` would have written without a simulation at all.
// If the round trip, the scale convention, the rounding rule or the fp16
// store were wrong anywhere, this would not come back bit for bit.

import (
	"math"
	"sort"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// q8_0Tensors are real dense weights the checkpoint ships as Q8_0, one from
// each of the three families L8a stages that way, plus the head.
var q8_0Tensors = []struct{ name, family string }{
	{"blk.0.attn_qkv.weight", "deltanet"},
	{"blk.3.attn_q.weight", "full_attn"},
	{"blk.0.hc_attn_down.weight", "hyper_conn"},
	{"output.weight", "lm_head"},
}

func TestQuantSimIsIdentityAtQ8(t *testing.T) {
	m, _ := fixtures(t)
	q, err := ParseQuantSim("q8sym/32")
	if err != nil {
		t.Fatal(err)
	}
	if q.Group != q8Group {
		t.Fatalf("q8sym/32 has group %d, the bank's is %d", q.Group, q8Group)
	}
	for _, tc := range q8_0Tensors {
		t.Run(tc.name, func(t *testing.T) {
			tn, err := m.Set.Get(tc.name)
			if err != nil {
				t.Skipf("no %s: %v", tc.name, err)
			}
			if got := simFamily(tc.name); got != tc.family {
				t.Fatalf("simFamily(%q) is %q, want %q", tc.name, got, tc.family)
			}
			want, err := tn.Dequantize(nil)
			if err != nil {
				t.Fatal(err)
			}
			// The reference is what tileB would stage with no simulation:
			// the dequantised f32 narrowed to a half. The simulation has to
			// land on the same halves, not on the same floats, because the
			// half is what the kernel multiplies.
			got := append([]float32(nil), want...)
			if err := q.Apply(got, int(tn.Dims[0])); err != nil {
				t.Fatal(err)
			}
			bad, first := 0, -1
			for i := range got {
				if safetensors.F32ToF16(got[i]) != safetensors.F32ToF16(want[i]) {
					bad++
					if first < 0 {
						first = i
					}
				}
			}
			if bad != 0 {
				t.Errorf("%s: %d of %d halves differ, first at %d: %v against %v",
					tc.name, bad, len(got), first, got[first], want[first])
				return
			}
			t.Logf("%s: %d values, q8sym/32 is an identity on the halves", tc.name, len(got))
		})
	}
}

// TestQuantSimLevels is the two things a format has to be: no more levels per
// group than its bits allow, and monotonically less error as the bits go up.
func TestQuantSimLevels(t *testing.T) {
	m, _ := fixtures(t)
	const name = "blk.0.attn_qkv.weight"
	tn, err := m.Set.Get(name)
	if err != nil {
		t.Skipf("no %s: %v", name, err)
	}
	ref, err := tn.Dequantize(nil)
	if err != nil {
		t.Fatal(err)
	}
	k := int(tn.Dims[0])
	// One matrix of 10240 x 2560 is 26 M values; a slab is enough and keeps
	// the test a second rather than a minute.
	if rows := 256; len(ref) > rows*k {
		ref = ref[:rows*k]
	}

	var prev float64
	for _, spec := range []string{"q2sym/32", "q3sym/32", "q4sym/32", "q4_0/32", "q5sym/32", "q6sym/32", "q8sym/32"} {
		q, err := ParseQuantSim(spec)
		if err != nil {
			t.Fatal(err)
		}
		got := append([]float32(nil), ref...)
		if err := q.Apply(got, k); err != nil {
			t.Fatal(err)
		}
		// Distinct levels in one group, which must not exceed 2^bits.
		levels := map[float32]bool{}
		for _, v := range got[:q.Group] {
			levels[v] = true
		}
		if n, max := len(levels), 1<<q.Bits; n > max {
			t.Errorf("%s: %d distinct values in a group of %d, %d bits allow %d", spec, n, q.Group, q.Bits, max)
		}
		var se, sr float64
		for i := range got {
			d := float64(got[i] - ref[i])
			se += d * d
			sr += float64(ref[i]) * float64(ref[i])
		}
		rel := math.Sqrt(se / sr)
		t.Logf("%-9s %5.3f bits/w   rel rms %.4e   %2d levels used in group 0",
			spec, q.BitsPerWeight(), rel, len(levels))
		// q4_0 sits beside q4sym at the same bit count rather than below it,
		// so it is the one comparison that is not part of the ladder.
		if spec == "q4_0/32" {
			continue
		}
		if prev != 0 && rel > prev {
			t.Errorf("%s: rel rms %.4e is worse than the format below it (%.4e)", spec, rel, prev)
		}
		prev = rel
	}
}

// TestQuantPlan is the per-family grammar: what a real bank is described by.
func TestQuantPlan(t *testing.T) {
	p, err := ParseQuantPlan("deltanet=q4_0/32,hyper_conn=q6sym/32", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		q8   bool
		want string
	}{
		{"blk.0.attn_qkv.weight", true, "q4_0/32"},
		{"blk.0.ssm_alpha.weight", false, "q4_0/32"},
		{"blk.0.hc_attn_down.weight", true, "q6sym/32"},
		{"blk.3.attn_q.weight", true, ""},        // not in the plan
		{"output.weight", true, ""},              // not in the plan
		{"blk.0.ssm_norm.weight", true, ""},      // never in any plan
		{"blk.0.ffn_gate_inp.weight", false, ""}, // the router, D3 keeps it wide
	} {
		q, ok := p.For(tc.name, tc.q8)
		got := ""
		if ok {
			got = q.String()
		}
		if got != tc.want {
			t.Errorf("For(%q) is %q, want %q", tc.name, got, tc.want)
		}
	}

	// Src is the other axis, and it is L8a-2's split.
	p, err = ParseQuantPlan("q8sym/32", "", "tail", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.For("blk.0.attn_qkv.weight", true); ok {
		t.Error("src=tail covered a Q8_0 tensor")
	}
	if _, ok := p.For("blk.0.ssm_alpha.weight", false); !ok {
		t.Error("src=tail missed ssm_alpha, which is the tail")
	}

	for _, bad := range []string{"deltanet=q4_0/32,nope=q4_0/32", "deltanet", "deltanet=q9sym/32"} {
		if _, err := ParseQuantPlan(bad, "", "", ""); err == nil {
			t.Errorf("%q parsed, want an error", bad)
		}
	}
	if _, err := ParseQuantPlan("deltanet=q4_0/32", "lm_head", "", ""); err == nil {
		t.Error("a per-family spec plus a family filter parsed, want an error")
	}
}

// TestSimFamilyCoversEveryDenseMatmul is the regression guard for the bug the
// per-family screen found: `lm_head` read as **exactly** unchanged at 4 bits
// because the head does not reach the device through Model.F32 and the hook
// was not repeated in gpu_head.go. A family the simulation never touches
// screens as "tolerates 4 bits", which is the most expensive wrong answer
// available here.
//
// The tally is the general guard — a run prints what it reached — and this is
// the static half: every family the whitelist names has to match at least one
// tensor of the real checkpoint, so a rename cannot quietly empty one.
func TestSimFamilyCoversEveryDenseMatmul(t *testing.T) {
	m, _ := fixtures(t)
	seen := map[string]int{}
	for _, tn := range m.Set.Tensors {
		if f := simFamily(tn.Name); f != "" {
			seen[f]++
		}
	}
	for _, f := range SimFamilies() {
		if seen[f] == 0 {
			t.Errorf("family %q matches no tensor in the checkpoint", f)
			continue
		}
		t.Logf("%-12s %4d tensors", f, seen[f])
	}
	// And the exclusions, which are as load-bearing as the inclusions: the
	// router's ties decide which ten of 512 experts run (L5a), and a norm is
	// not read a row at a time.
	for _, name := range []string{
		"blk.0.ffn_gate_inp.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_a",
		"blk.0.ssm_dt.bias", "blk.0.ssm_conv1d.weight", "blk.3.attn_q_norm.weight",
		"token_embd.weight", "blk.1.per_layer_token_embd.weight",
	} {
		if f := simFamily(name); f != "" {
			t.Errorf("simFamily(%q) is %q, want it excluded", name, f)
		}
	}
}

// TestQuantSimOffIsNothing is the default path: no environment, no change.
func TestQuantSimOff(t *testing.T) {
	q, err := ParseQuantSim("")
	if err != nil {
		t.Fatal(err)
	}
	if !q.Off() || q.String() != "off" {
		t.Fatalf("the empty spec is %v, want off", q)
	}
	x := []float32{1, -2.5, 3e-7, 0, 65504}
	want := append([]float32(nil), x...)
	if err := q.Apply(x, len(x)); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if x[i] != want[i] {
			t.Fatalf("off changed %v to %v at %d", want[i], x[i], i)
		}
	}
	for _, bad := range []string{"q4sym", "q9sym/32", "nope/32", "q4sym/12", "q4sym/0"} {
		if _, err := ParseQuantSim(bad); err == nil {
			t.Errorf("%q parsed, want an error", bad)
		}
	}
	// A group that does not divide a row has to be refused, not rounded: the
	// hyper-connection up projection's k is 320, so q4sym/128 is a real case
	// and not a hypothetical.
	q, err = ParseQuantSim("q4sym/128")
	if err != nil {
		t.Fatal(err)
	}
	if err := q.Apply(make([]float32, 320), 320); err == nil {
		t.Error("q4sym/128 accepted a row of 320, want an error")
	}
}

// TestImatrixQuant is L8c-2's instrument: the published importance matrix,
// and the two things ggml's make_qx_quants does at once.
//
//	go test ./llm/ -v -run TestImatrixQuant
//
// `rtn` is ggml's uncalibrated path and is what every L8c-1 rung was measured
// with. `search` is make_qx_quants with rmse_type 1 — a least-squares scale
// and a nineteen-rung sweep of the initial one. `imatrix` is the same search
// with unsloth's published column importances folded into the weight, which
// is exactly quantize_row_q4_0_impl. Keeping the middle arm is the whole
// point: without it an improvement cannot be attributed to calibration.
//
// The assertion is on the objective each arm actually optimises. `imatrix`
// minimises the importance-weighted error, so that is what it has to win on;
// it is under no obligation to win on the unweighted one, and reporting both
// is how that stays honest.
//
// **And winning that objective turns out not to be winning.** L8c-2 measured
// the same three arms through the whole model and calibration is *worse* —
// 22.4% against round-to-nearest's 18.5% at 4 bits — because what this test
// reports is the residual, and what costs the model is the systematic gain
// (TestQuantSimGain). So this stays a check that the arms are what they claim
// to be, and is not evidence that the calibrated one is better.
func TestImatrixQuant(t *testing.T) {
	m, _ := fixtures(t)
	im, err := DefaultImatrix()
	if err != nil {
		t.Skipf("no imatrix: %v", err)
	}
	t.Logf("imatrix: %d chunks of %q", im.Chunks, im.Datasets)

	for _, name := range []string{
		"blk.0.attn_qkv.weight", "blk.0.hc_attn_down.weight", "blk.3.attn_q.weight",
	} {
		t.Run(name, func(t *testing.T) {
			tn, err := m.Set.Get(name)
			if err != nil {
				t.Skipf("no %s: %v", name, err)
			}
			k := int(tn.Dims[0])
			qw, err := im.Columns(name)
			if err != nil {
				t.Fatal(err)
			}
			if qw == nil {
				t.Fatalf("no imatrix entry for %s", name)
			}
			if len(qw) != k {
				t.Fatalf("%s: %d importance columns for a row of %d", name, len(qw), k)
			}
			for i, v := range qw {
				if v < 0 || v != v {
					t.Fatalf("%s: importance[%d] is %v", name, i, v)
				}
			}
			ref, err := tn.Dequantize(nil)
			if err != nil {
				t.Fatal(err)
			}
			if rows := 512; len(ref) > rows*k {
				ref = ref[:rows*k]
			}

			// Both errors, for each arm. The weighted one uses the same
			// column importances the imatrix arm optimises against.
			errs := func(got []float32) (plain, weighted float64) {
				var se, sr, we, wr float64
				for i := range got {
					d := float64(got[i] - ref[i])
					w := float64(qw[i%k])
					se += d * d
					sr += float64(ref[i]) * float64(ref[i])
					we += w * d * d
					wr += w * float64(ref[i]) * float64(ref[i])
				}
				return math.Sqrt(se / sr), math.Sqrt(we / wr)
			}

			var rtnW, imatW float64
			for _, mode := range []string{"rtn", "search", "imatrix"} {
				p, err := ParseQuantPlan("deltanet=q4_0/32,hyper_conn=q4_0/32,full_attn=q4_0/32", "", "", mode)
				if err != nil {
					t.Fatal(err)
				}
				q, ok := p.For(name, true)
				if !ok {
					t.Fatalf("the plan does not cover %s", name)
				}
				got := append([]float32(nil), ref...)
				var w []float32
				if mode == "imatrix" {
					w = qw
				}
				if err := q.ApplyWeighted(got, k, w); err != nil {
					t.Fatal(err)
				}
				plain, weighted := errs(got)
				t.Logf("%-8s plain rel rms %.4e   imatrix-weighted %.4e", mode, plain, weighted)
				switch mode {
				case "rtn":
					rtnW = weighted
				case "imatrix":
					imatW = weighted
				}
			}
			if imatW >= rtnW {
				t.Errorf("imatrix weighted error %.4e is not below rtn's %.4e", imatW, rtnW)
			} else {
				t.Logf("calibration is %.3fx on the objective it optimises", rtnW/imatW)
			}
		})
	}
}

// TestQuantSimGain is the diagnostic L8c-2 needed when the calibrated arms
// came back *worse* end to end than round-to-nearest.
//
//	go test ./llm/ -v -run TestQuantSimGain
//
// A quantiser's error has two parts and they do not cost the same. The
// **noise** is whatever is left after the best scalar fit, and it averages
// out across a wide sum. The **gain** is that scalar — `sum(w_hat*w) /
// sum(w*w)`, the least-squares coefficient relating the reconstruction to the
// original — and a gain below one is a *systematic* shrinkage of the whole
// matrix, which does not average out and which compounds through 48 layers on
// L6b-3's x1.085.
//
// make_qx_quants minimises a per-group weighted squared error with the levels
// clamped, and minimising squared error under clamping biases the scale down:
// the optimum trades a little bias for a lot of variance. That is the right
// trade for one group in isolation and the wrong one for a residual stream
// read 97 times a pass.
//
// **L8c-3 runs the same diagnostic over the asymmetric form**, because that
// is where the mechanism makes a prediction rather than a description: a
// symmetric group has one free parameter and it *is* the gain, so a squared
// error fit has nowhere to put a bias but into the gain. A K-quant group has
// two — a scale and a min — so the fit can absorb an offset without shrinking
// the scale, and if the mechanism is right the asymmetric gain should sit
// nearer 1 on exactly the tensor (`hc_attn_up`) where the symmetric one does
// not. The formats are swept beside each other for that comparison.
func TestQuantSimGain(t *testing.T) {
	m, _ := fixtures(t)
	im, err := DefaultImatrix()
	if err != nil {
		t.Skipf("no imatrix: %v", err)
	}
	for _, name := range []string{
		"blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight",
		"blk.0.attn_qkv.weight", "output.weight",
	} {
		t.Run(name, func(t *testing.T) {
			tn, err := m.Set.Get(name)
			if err != nil {
				t.Skipf("no %s: %v", name, err)
			}
			k := int(tn.Dims[0])
			ref, err := tn.Dequantize(nil)
			if err != nil {
				t.Fatal(err)
			}
			if rows := 256; len(ref) > rows*k {
				ref = ref[:rows*k]
			}
			cols, err := im.Columns(name)
			if err != nil {
				t.Fatal(err)
			}
			for _, spec := range []string{"q4_0/32", "q4_k/32"} {
				for _, mode := range []string{"rtn", "search", "imatrix"} {
					q, err := ParseQuantSim(spec)
					if err != nil {
						t.Fatal(err)
					}
					q.Mode = mode
					var qw []float32
					if mode == "imatrix" {
						if cols == nil {
							t.Logf("%-7s %-8s no imatrix entry, skipped", spec, mode)
							continue
						}
						qw = cols
					}
					got := append([]float32(nil), ref...)
					if err := q.ApplyWeighted(got, k, qw); err != nil {
						t.Fatal(err)
					}
					// gain is the least-squares scalar; resid is what is left
					// after removing it, which is the part that averages out.
					var wx, ww, se float64
					for i := range got {
						wx += float64(got[i]) * float64(ref[i])
						ww += float64(ref[i]) * float64(ref[i])
						d := float64(got[i] - ref[i])
						se += d * d
					}
					gain := wx / ww
					var rs float64
					for i := range got {
						d := float64(got[i]) - gain*float64(ref[i])
						rs += d * d
					}
					t.Logf("%-7s %-8s gain %.6f  (%+.3f%%)   rel rms %.4e   residual after gain %.4e",
						spec, mode, gain, 100*(gain-1), math.Sqrt(se/ww), math.Sqrt(rs/ww))
				}
			}
		})
	}
}

// TestImatrixSkew is the measurement that explains L8c-2, and it is about the
// calibration data rather than about any quantiser.
//
//	go test ./llm/ -v -run TestImatrixSkew
//
// An importance-weighted scale is a least-squares fit over the columns of one
// scale group. How well-conditioned that fit is depends on how evenly the
// importance is spread inside the group — and the **participation ratio**
// `(sum w)^2 / (n * sum w^2)` is exactly that: 1 when all 32 columns matter
// equally, 1/32 when one carries everything. Multiply it by 32 and it is the
// effective number of columns the scale is being fitted to.
//
// The hyper-connection block's *up* projection reads the low-rank space — the
// down projection's output, which is a gate — and its energy is concentrated
// beyond anything else in the model. That is why calibration hurts it most.
func TestImatrixSkew(t *testing.T) {
	_, _ = fixtures(t)
	im, err := DefaultImatrix()
	if err != nil {
		t.Skip(err)
	}
	for _, name := range []string{
		"blk.0.hc_attn_up.weight", "blk.0.hc_attn_down.weight",
		"blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.3.attn_q.weight",
	} {
		c, err := im.Columns(name)
		if err != nil || c == nil {
			t.Logf("%-28s no entry", name)
			continue
		}
		k := len(c)
		s := append([]float32(nil), c...)
		sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
		var sum float64
		for _, v := range c {
			sum += float64(v)
		}
		// Top-1% share, and the mean within-group participation ratio:
		// (sum w)^2 / (n * sum w^2), which is 1 when a group's 32 columns are
		// equally important and 1/32 when one column carries everything.
		top := 0.0
		for i := k - k/100; i < k; i++ {
			top += float64(s[i])
		}
		var pr float64
		groups := k / 32
		for g := 0; g < groups; g++ {
			var a, b float64
			for j := g * 32; j < (g+1)*32; j++ {
				a += float64(c[j])
				b += float64(c[j]) * float64(c[j])
			}
			if b > 0 {
				pr += a * a / (32 * b)
			}
		}
		pr /= float64(groups)
		t.Logf("%-28s k=%5d  max/median %8.1f  top1%% share %5.1f%%  mean group participation %.3f",
			name, k, float64(s[k-1])/math.Max(float64(s[k/2]), 1e-30), 100*top/sum, pr)
	}
}

// TestQuantSimAsym is the structural half of L8c-3: the K-quant arm is the
// format it says it is, including on the one row width ggml's own quantiser
// refuses.
//
//	go test ./llm/ -v -run TestQuantSimAsym
//
// The arithmetic half is TestQuantSimMatchesGGML, which says the port is
// bit-identical to ggml where ggml will run at all. This is the part ggml
// cannot check, because `quantize_row_q4_K_ref` asserts `k % 256 == 0` and
// `hc_attn_up` — L8c-2's outlier, and the tensor the whole asymmetric
// question is about — is 320 wide. There the super-block is the whole row,
// ten groups rather than eight, which is 4.475 bits a weight and not 4.500.
func TestQuantSimAsym(t *testing.T) {
	q4, err := ParseQuantSim("q4_k/32")
	if err != nil {
		t.Fatal(err)
	}
	q5, err := ParseQuantSim("q5_k/32")
	if err != nil {
		t.Fatal(err)
	}
	if got := q4.BitsPerWeight(); got != 4.5 {
		t.Errorf("q4_k/32 is %.4f bits, want 4.5 — the same as q4_0/32, which is the point", got)
	}
	if got := q5.BitsPerWeight(); got != 5.5 {
		t.Errorf("q5_k/32 is %.4f bits, want 5.5", got)
	}
	for _, tc := range []struct {
		k    int
		sub  int
		bits float64
	}{
		{2560, 8, 4.500}, {6144, 8, 4.500}, {10240, 8, 4.500}, {320, 10, 4.475},
	} {
		sub, err := q4.superBlocks(tc.k)
		if err != nil {
			t.Fatalf("k=%d: %v", tc.k, err)
		}
		if sub != tc.sub {
			t.Errorf("k=%d: %d groups a super-block, want %d", tc.k, sub, tc.sub)
		}
		if got := q4.bitsPerWeightSuper(sub); math.Abs(got-tc.bits) > 1e-9 {
			t.Errorf("k=%d: %.4f bits, want %.4f", tc.k, got, tc.bits)
		}
	}
	// A width that is neither eight super-blocks nor one short row has no
	// honest reading, and guessing would make the bits column a fiction.
	if _, err := q4.superBlocks(544); err == nil {
		t.Error("k=544 gave a super-block, want an error")
	}
	if _, err := q4.superBlocks(300); err == nil {
		t.Error("k=300 is not a whole number of groups, want an error")
	}

	// And on real weights, including the 320-wide one. An asymmetric group
	// has 2^bits reachable levels where the symmetric form reaches 15 or 16,
	// and the min means they need not straddle zero.
	m, _ := fixtures(t)
	for _, name := range []string{"blk.0.hc_attn_up.weight", "blk.0.hc_attn_down.weight", "blk.0.attn_qkv.weight"} {
		t.Run(name, func(t *testing.T) {
			tn, err := m.Set.Get(name)
			if err != nil {
				t.Skipf("no %s: %v", name, err)
			}
			k := int(tn.Dims[0])
			ref, err := tn.Dequantize(nil)
			if err != nil {
				t.Fatal(err)
			}
			if rows := 256; len(ref) > rows*k {
				ref = ref[:rows*k]
			}
			for _, spec := range []string{"q4_0/32", "q4_k/32", "q5sym/32", "q5_k/32"} {
				q, err := ParseQuantSim(spec)
				if err != nil {
					t.Fatal(err)
				}
				got := append([]float32(nil), ref...)
				if err := q.Apply(got, k); err != nil {
					t.Fatal(err)
				}
				levels := map[float32]bool{}
				for _, v := range got[:q.Group] {
					levels[v] = true
				}
				if n, max := len(levels), 1<<q.Bits; n > max {
					t.Errorf("%s: %d distinct values in a group of %d, %d bits allow %d",
						spec, n, q.Group, q.Bits, max)
				}
				var se, sr float64
				for i := range got {
					d := float64(got[i] - ref[i])
					se += d * d
					sr += float64(ref[i]) * float64(ref[i])
				}
				sub, _ := q.superBlocks(k)
				bits := q.BitsPerWeight()
				if q.Asym {
					bits = q.bitsPerWeightSuper(sub)
				}
				t.Logf("%-9s %5.3f bits/w   rel rms %.4e   %2d levels used in group 0",
					spec, bits, math.Sqrt(se/sr), len(levels))
			}
		})
	}
}
