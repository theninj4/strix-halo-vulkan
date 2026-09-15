package llm

import "testing"

// pleFixtures opens the checkpoint, the trace and the block's constants.
func pleFixtures(t *testing.T) (*Model, *Trace, PLEConfig, []int32) {
	t.Helper()
	m, tr := fixtures(t)
	c, ok, err := m.PLEConfig()
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("the checkpoint states no PLE module, but qwen4exp has one at layer 1")
	}
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	return m, tr, c, ids
}

// TestPLEConfig checks the constants the hash is made of against the
// checkpoint, because every one of them is a silent wrong answer if
// transcribed: a wrong multiplier or vocabulary gathers a real row of a
// 320 M-row table that simply belongs to another n-gram.
func TestPLEConfig(t *testing.T) {
	_, _, c, _ := pleFixtures(t)
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"ngram", c.NGram, 3},
		{"heads per ngram", c.PerGram, 8},
		{"heads", c.NHeads, 16},
		{"head dim", c.HeadDim, 160},
		{"conv kernel", c.Conv, 4},
		{"layers", len(c.Layers), 1},
		{"ple layer", c.Layers[0], 1},
		{"multipliers", len(c.Mult), 3},
		{"offsets", len(c.Offsets), 16},
		{"vocabs", len(c.Vocabs), 16},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.NHeads*c.HeadDim != c.NEmbd {
		t.Errorf("%d heads of %d is not n_embd %d", c.NHeads, c.HeadDim, c.NEmbd)
	}
	// The sixteen ranges tile the table without overlapping, which is what
	// makes one mixed value into sixteen independent lookups.
	for h := 1; h < c.NHeads; h++ {
		if c.Offsets[h] != c.Offsets[h-1]+c.Vocabs[h-1] {
			t.Errorf("head %d starts at %d, but head %d ends at %d",
				h, c.Offsets[h], h-1, c.Offsets[h-1]+c.Vocabs[h-1])
		}
	}
	t.Logf("eos %d, image %d, multipliers %v", c.EOS, c.Image, c.Mult)
	t.Logf("head 0: offset %d vocab %d; head 15: offset %d vocab %d",
		c.Offsets[0], c.Vocabs[0], c.Offsets[15], c.Vocabs[15])
}

// TestPLEGather is the gate on the hash. `ple_embd` is the gathered n-gram
// embedding, and it is a *lookup*: if the sixteen row indices are right the
// 2560 values are bit-exact, and if any one of them is wrong the row is a
// plausible embedding from somewhere else in the table. There is no tolerance
// to hide in either direction.
func TestPLEGather(t *testing.T) {
	m, tr, c, ids := pleFixtures(t)
	rows := PLERows(c, ids)
	t.Logf("token %d rows: %v", 0, rows[:c.NHeads])
	for i, r := range rows {
		if h := i % c.NHeads; uint32(r) < c.Offsets[h] || uint32(r) >= c.Offsets[h]+c.Vocabs[h] {
			t.Fatalf("token %d head %d hashed to row %d, outside its range [%d, %d)",
				i/c.NHeads, h, r, c.Offsets[h], c.Offsets[h]+c.Vocabs[h])
		}
	}
	got, err := m.PLEGather(rows, c.NHeads, c.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("ple_embd", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(got, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ple_embd: %v", r)
	if r.maxAbs != 0 {
		t.Errorf("ple_embd is not bit-exact: %v — the hash or the gather disagrees with llama.cpp", r)
	}
}

// TestPLERowsCutOnEOS states the window rule as a checked claim rather than a
// comment. Running off the start of a sequence reads as EOS, and so does an
// EOS in the window — so a token at position 0 and the same token after an EOS
// must hash identically, while the same token with real predecessors must not.
func TestPLERowsCutOnEOS(t *testing.T) {
	_, _, c, ids := pleFixtures(t)
	tok := ids[2]
	first := PLERows(c, []int32{tok})
	afterEOS := PLERows(c, []int32{c.EOS, tok})[c.NHeads:]
	withCtx := PLERows(c, []int32{ids[0], ids[1], tok})[2*c.NHeads:]

	for h := 0; h < c.NHeads; h++ {
		if first[h] != afterEOS[h] {
			t.Errorf("head %d: at position 0 the row is %d, after an EOS %d; a missing predecessor should read as EOS",
				h, first[h], afterEOS[h])
		}
	}
	// The trigram heads (the second group) are the ones two predecessors
	// reach; the bigram heads only see one.
	same := 0
	for h := c.PerGram; h < c.NHeads; h++ {
		if first[h] == withCtx[h] {
			same++
		}
	}
	if same == c.PerGram {
		t.Errorf("every trigram head hashes the same with and without context; the window is not being read")
	}
}

// TestPLEBlock is the L2 gate for the block: all three of its dumped tensors,
// driven from llama.cpp's own residual (`l_last-0`) and its own gathered
// embedding, so a disagreement is this block's.
func TestPLEBlock(t *testing.T) {
	m, tr, c, ids := pleFixtures(t)
	nTok := len(ids)
	layer := c.Layers[0]
	w, err := m.PLEWeights(layer)
	if err != nil {
		t.Fatal(err)
	}
	embd, err := tr.Get("ple_embd", 0)
	if err != nil {
		t.Fatal(err)
	}
	in, err := tr.Get("l_last-0", 0)
	if err != nil {
		t.Fatal(err)
	}

	// Both numerics, as in TestHCMix: the key and value projections are Q8_0,
	// so under Exact the disagreement is the reference's int8 activations and
	// under RefQ8 it is not.
	for _, mode := range []struct {
		name string
		act  Numerics
		tol  map[string]float64
	}{
		// Measured, not chosen. Exact is 1.4e-05 / 1.4e-06 / 6.6e-06 — the
		// reference's int8 activations, since the only difference at that
		// setting is what llama.cpp does to the gathered embedding before it
		// multiplies a Q8_0 weight. RefQ8 is 7.7e-08 / 2.1e-09 / 1.3e-08,
		// which is f32 round-off and nothing else.
		{"Exact", Exact, map[string]float64{"ple_gate-1": 5e-5, "ple_gated_value-1": 5e-6, "ple_conv_out-1": 2e-5}},
		{"RefQ8", RefQ8, map[string]float64{"ple_gate-1": 1e-6, "ple_gated_value-1": 1e-7, "ple_conv_out-1": 1e-7}},
	} {
		cfg := c
		cfg.Act = mode.act
		res := append([]float32(nil), in.Vals...)
		gate, gated, conv := PLEBlock(cfg, w, res, embd.Vals, nTok)
		for _, tc := range []struct {
			name string
			got  []float32
		}{
			{"ple_gate-1", gate},
			{"ple_gated_value-1", gated},
			{"ple_conv_out-1", conv},
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

// TestPLEClosesTheLayer1Gap is the check the GPU test could not make: layer
// 1's attn mixer reads `l_last-0` *after* this block has added into it, which
// is why that one mixer is excluded from the trace comparison in gpu_test.go.
// Running the block should reproduce the residual that `hc_norm-1` normalises.
func TestPLEClosesTheLayer1Gap(t *testing.T) {
	m, tr, c, ids := pleFixtures(t)
	nTok := len(ids)
	c.Act = RefQ8
	w, err := m.PLEWeights(c.Layers[0])
	if err != nil {
		t.Fatal(err)
	}
	embd, err := tr.Get("ple_embd", 0)
	if err != nil {
		t.Fatal(err)
	}
	in, err := tr.Get("l_last-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	res := append([]float32(nil), in.Vals...)
	PLEBlock(c, w, res, embd.Vals, nTok)

	// The mixer that reads it, run on our own residual, against llama.cpp's
	// normalised tensor. Anything wrong with the PLE block shows here
	// amplified, because hc_norm divides by the rms of what this produced.
	hw, err := m.HCWeights(1, "attn")
	if err != nil {
		t.Fatal(err)
	}
	hc := m.Config.HCConfig()
	hc.Act = RefQ8
	_, _, xn, _ := HCMix(hc, hw, res, nTok)
	want, err := tr.Get("hc_norm-1", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(xn, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hc_norm-1 through the PLE block: %v", r)
	if r.rms > 1e-4 {
		t.Errorf("hc_norm-1: rms %.3e over 1e-4 (%v)", r.rms, r)
	}
}

// TestPLEGPU is the GPU block against both references: the CPU one it was
// ported from, where the only difference is the fp16 key/value projection,
// and llama.cpp's own tensors behind it.
//
// The gate is the sensitive one and deliberately so — it is a dot product of
// two 2560-wide normalised vectors put through a square root, so an fp16
// operand shows there before it shows anywhere else in the block.
func TestPLEGPU(t *testing.T) {
	m, tr, c, ids := pleFixtures(t)
	nTok := len(ids)
	w, err := m.PLEWeights(c.Layers[0])
	if err != nil {
		t.Fatal(err)
	}
	embd, err := tr.Get("ple_embd", 0)
	if err != nil {
		t.Fatal(err)
	}
	in, err := tr.Get("l_last-0", 0)
	if err != nil {
		t.Fatal(err)
	}

	dev, done := newTestDevice(t)
	defer done()
	g, err := NewPLEGPU(dev, c, nTok, w, PLEOpts{ConvOut: true})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged: %.1f MB of weights, %.1f MB of arenas",
		float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6)

	if err := g.Upload(in.Vals, embd.Vals, nTok); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}

	cfg := c
	cfg.Act = Exact
	cpuRes := append([]float32(nil), in.Vals...)
	cpuGate, cpuGated, cpuConv := PLEBlock(cfg, w, cpuRes, embd.Vals, nTok)

	for _, tc := range []struct {
		name      string
		got, cpu  []float32
		tol, rtol float64
	}{
		// Against the f32 model these are what fp16 operands cost; against
		// llama.cpp they also carry its int8 activations, which L2b measured
		// as the larger term everywhere a Q8_0 weight is involved.
		{"ple_gate-1", g.Gate(), cpuGate, 2e-5, 5e-5},
		{"ple_gated_value-1", g.Gated(), cpuGated, 2e-6, 5e-6},
		{"ple_conv_out-1", g.ConvOut(), cpuConv, 1e-5, 2e-5},
	} {
		r, err := compare(tc.got, tc.cpu)
		if err != nil {
			t.Fatalf("%s against the CPU reference: %v", tc.name, err)
		}
		t.Logf("%-18s vs CPU Exact  %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against the CPU reference, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
		want, err := tr.Get(tc.name, 0)
		if err != nil {
			t.Fatal(err)
		}
		if r, err = compare(tc.got, want.Vals); err != nil {
			t.Fatalf("%s against llama.cpp: %v", tc.name, err)
		}
		t.Logf("%-18s vs llama.cpp  %v", tc.name, r)
		if r.rms > tc.rtol {
			t.Errorf("%s: rms %.3e against llama.cpp, over %.1e (%v)", tc.name, r.rms, tc.rtol, r)
		}
	}

	// And the residual the block leaves behind, which is what layer 1's
	// hyper-connection mixer reads and the only output that matters.
	r, err := compare(g.Res(), cpuRes)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%-18s vs CPU Exact  %v", "residual", r)
	if r.rms > 1e-5 {
		t.Errorf("residual: rms %.3e over 1e-4 (%v)", r.rms, r)
	}
}

// TestPLEConvIsDilatedAndOrdered is the negative control for the two things
// about this convolution that a transcription gets wrong silently.
//
// llama.cpp's own comment says `ggml_conv_1d_dw` is unreliable and spells the
// operation out as a sum of shifted copies; the two ways to spell it wrong are
// to lose the dilation (taps at 1, 2, 3 back instead of 3, 6, 9) and to
// reverse the tap order (which is the difference between a convolution and a
// correlation). Both produce a perfectly plausible tensor of the right shape
// and the right magnitude, so only a test that demands they disagree tells
// them from the real thing.
func TestPLEConvIsDilatedAndOrdered(t *testing.T) {
	m, tr, c, ids := pleFixtures(t)
	nTok := len(ids)
	c.Act = RefQ8
	w, err := m.PLEWeights(c.Layers[0])
	if err != nil {
		t.Fatal(err)
	}
	embd, err := tr.Get("ple_embd", 0)
	if err != nil {
		t.Fatal(err)
	}
	in, err := tr.Get("l_last-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("ple_conv_out-1", 0)
	if err != nil {
		t.Fatal(err)
	}

	run := func(cfg PLEConfig, ww PLEWeights) float64 {
		res := append([]float32(nil), in.Vals...)
		_, _, conv := PLEBlock(cfg, ww, res, embd.Vals, nTok)
		r, err := compare(conv, want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		return r.rms
	}

	undilated := c
	undilated.NGram = 1 // the conv's dilation is the n-gram size, and only the conv reads it here
	reversed := w
	reversed.Conv1d = append([]float32(nil), w.Conv1d...)
	for ch := 0; ch < c.Wide(); ch++ {
		tap := reversed.Conv1d[ch*c.Conv : (ch+1)*c.Conv]
		for i, j := 0, len(tap)-1; i < j; i, j = i+1, j-1 {
			tap[i], tap[j] = tap[j], tap[i]
		}
	}

	good := run(c, w)
	for _, tc := range []struct {
		name string
		rms  float64
	}{
		{"dilation 1", run(undilated, w)},
		{"taps reversed", run(c, reversed)},
	} {
		t.Logf("%-14s rms %.3e against the correct %.3e, %.0fx", tc.name, tc.rms, good, tc.rms/good)
		if tc.rms < 100*good {
			t.Errorf("%s is only %.1fx worse than the real convolution; the test cannot tell them apart",
				tc.name, tc.rms/good)
		}
	}
}
