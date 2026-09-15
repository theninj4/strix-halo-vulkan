package llm

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// The fixtures: a mapped checkpoint and llama.cpp's own trace of a forward
// pass over it. Both are large and neither is in git, so every test here
// skips rather than fails when they are absent — `go test ./...` on a machine
// without the 111 GB checkpoint is expected to pass.
const (
	modelDir  = "../models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
	traceDir  = "../reference/out/llm"
	regenHint = `run:
  L=/home/kube/repos/llama.cpp
  gcc -O2 -o /tmp/eval_dump reference/eval_dump.c -I$L/include -I$L/ggml/include \
      -L$L/build/bin -lllama -lggml-base -Wl,-rpath,$L/build/bin
  /tmp/eval_dump -m $M -o reference/out/llm -c 64 -p 'The capital of France is Paris.' -n '<regex>'`
)

func fixtures(t *testing.T) (*Model, *Trace) {
	t.Helper()
	if _, err := os.Stat(modelDir); err != nil {
		t.Skipf("checkpoint absent: %v", err)
	}
	if _, err := os.Stat(traceDir); err != nil {
		t.Skipf("reference trace absent (%v); %s", err, regenHint)
	}
	tr, err := OpenTrace(traceDir)
	if err != nil {
		t.Skipf("reference trace unusable (%v); %s", err, regenHint)
	}
	m, err := Open(modelDir)
	if err != nil {
		t.Fatalf("open checkpoint: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m, tr
}

// cmp reports the worst absolute and relative disagreement between a computed
// tensor and llama.cpp's, and the reference's own scale so a tolerance can be
// read rather than guessed.
type cmpResult struct {
	maxAbs, maxRel, rms, refMax float64
	at                          int
}

func compare(got, want []float32) (cmpResult, error) {
	if len(got) != len(want) {
		return cmpResult{}, fmt.Errorf("length %d, reference has %d", len(got), len(want))
	}
	var r cmpResult
	var ss float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			return r, fmt.Errorf("value %d is %v", i, g)
		}
		d := math.Abs(g - w)
		ss += d * d
		if a := math.Abs(w); a > r.refMax {
			r.refMax = a
		}
		if d > r.maxAbs {
			r.maxAbs, r.at = d, i
		}
		// Relative to the reference's magnitude, floored so that a value that
		// is zero in both does not divide by nothing.
		if den := math.Abs(w); den > 1e-3 {
			if rel := d / den; rel > r.maxRel {
				r.maxRel = rel
			}
		}
	}
	r.rms = math.Sqrt(ss / float64(len(got)))
	return r, nil
}

func (r cmpResult) String() string {
	return fmt.Sprintf("maxAbs %.3e at %d, maxRel %.3e, rms %.3e, reference |max| %.3f",
		r.maxAbs, r.at, r.maxRel, r.rms, r.refMax)
}

// TestConfigMatchesCheckpoint is the cheap guard on everything below: the
// shapes the CPU reference runs at come from the checkpoint, so if they are
// read wrong every other comparison fails for the wrong reason.
func TestConfigMatchesCheckpoint(t *testing.T) {
	m, _ := fixtures(t)
	c := m.Config
	for _, tc := range []struct {
		name string
		got  int
		want int
	}{
		{"arch layers", c.NLayer, 48},
		{"n_embd", c.NEmbd, 2560},
		{"heads", c.NHead, 24},
		{"kv heads", c.NHeadKV, 2},
		{"hc", c.HC, 4},
		{"hc low rank", c.HCLowRank, 320},
		{"experts", c.NExpert, 512},
		{"experts used", c.NExpertUsed, 10},
		{"expert ffn", c.FFNExpert, 640},
		{"rope dims", c.RopeDims, 64},
		{"vocab", c.VocabSize, 248320},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.Arch != "qwen4exp" {
		t.Errorf("arch = %q, want qwen4exp", c.Arch)
	}
	if c.RMSEps <= 0 || c.RMSEps > 1e-4 {
		t.Errorf("rms eps = %g, want a small positive", c.RMSEps)
	}
}

// TestHCInit checks the wide residual's starting value against llama.cpp's
// `hc_init`: hc identical copies of the embedding. It is also the check that
// our token_embd gather agrees with the reference's, since `hc_init` is the
// first thing built out of it.
func TestHCInit(t *testing.T) {
	m, tr := fixtures(t)
	ids, prompt, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt %q, %d tokens %v", prompt, len(ids), ids)

	embd, err := m.Embeddings(ids)
	if err != nil {
		t.Fatal(err)
	}
	wantEmbd, err := tr.Get("model.input_embed", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(embd, wantEmbd.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("token_embd gather: %v", r)
	// The gather is exact: both sides dequantise the same Q8_0 rows, and
	// gguf/dequant.go is bit-exact against ggml (L1).
	if r.maxAbs != 0 {
		t.Errorf("token_embd gather is not bit-exact: %v", r)
	}

	got := HCInit(m.Config.HCConfig(), embd, len(ids))
	want, err := tr.Get("hc_init", 0)
	if err != nil {
		t.Fatal(err)
	}
	if r, err = compare(got, want.Vals); err != nil {
		t.Fatal(err)
	}
	t.Logf("hc_init: %v", r)
	if r.maxAbs != 0 {
		t.Errorf("hc_init disagrees: %v", r)
	}
}

// TestHCMix is the L2 gate for the block L2a promoted to the front of the
// queue. It runs the first mixer of layer 0 — the one whose input is
// `hc_init`, so nothing upstream can be wrong — and checks all four of its
// outputs against llama.cpp's own tensors.
func TestHCMix(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(ids)
	cfg := m.Config.HCConfig()

	embd, err := m.Embeddings(ids)
	if err != nil {
		t.Fatal(err)
	}
	res := HCInit(cfg, embd, nTok)

	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}

	// Both numerics are run, and the tolerances differ by two orders of
	// magnitude, because the oracle is not exact: llama.cpp's Vulkan backend
	// evaluates a Q8_0 matmul as an integer dot product over int8-quantised
	// activations. Exact is the mathematical model; RefQ8 is what reproduces
	// the reference, and only under RefQ8 does a tight bound mean anything.
	//
	// The two tensors whose weights are F32 in the checkpoint — hc_norm and
	// hc_inject — are unaffected and hold the tight bound either way. That
	// asymmetry is what localised the difference: a formula error would not
	// spare exactly the F32-weighted half of the block.
	for _, mode := range []struct {
		name string
		act  Numerics
		tol  map[string]float64
	}{
		{"Exact", Exact, map[string]float64{
			"hc_norm-0": 1e-5, "hc_inject-0": 1e-4, "hc_gate-0": 0.1, "hc_mixed-0": 0.5,
		}},
		{"RefQ8", RefQ8, map[string]float64{
			"hc_norm-0": 1e-5, "hc_inject-0": 1e-4, "hc_gate-0": 5e-4, "hc_mixed-0": 2e-3,
		}},
	} {
		c := cfg
		c.Act = mode.act
		mixed, inject, xn, gate := HCMix(c, w, res, nTok)
		for _, tc := range []struct {
			name string
			got  []float32
		}{
			{"hc_norm-0", xn},
			{"hc_gate-0", gate},
			{"hc_mixed-0", mixed},
			{"hc_inject-0", inject},
		} {
			want, err := tr.Get(tc.name, 0)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			r, err := compare(tc.got, want.Vals)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			t.Logf("%-6s %-12s %v", mode.name, tc.name, r)
			if tol := mode.tol[tc.name]; r.maxAbs > tol {
				t.Errorf("%s %s: maxAbs %.3e over tolerance %.1e (%v)", mode.name, tc.name, r.maxAbs, tol, r)
			}
		}
	}
}

// TestHCMixRefQ8IsTheCloserModel states the measurement the tolerances above
// rest on, so that it is a checked claim and not a comment: quantising the
// activations the way llama.cpp does moves the gate two orders of magnitude
// closer to it.
func TestHCMixRefQ8IsTheCloserModel(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(ids)
	cfg := m.Config.HCConfig()
	embd, err := m.Embeddings(ids)
	if err != nil {
		t.Fatal(err)
	}
	res := HCInit(cfg, embd, nTok)
	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("hc_gate-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	rms := map[Numerics]float64{}
	for _, act := range []Numerics{Exact, RefQ8} {
		c := cfg
		c.Act = act
		_, _, _, gate := HCMix(c, w, res, nTok)
		r, err := compare(gate, want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		rms[act] = r.rms
		t.Logf("act=%d gate %v", act, r)
	}
	if ratio := rms[Exact] / rms[RefQ8]; ratio < 100 {
		t.Errorf("RefQ8 is only %.1fx closer than Exact; the int8-activation model no longer explains the gap", ratio)
	}
}

// TestHCCombine closes the loop: the block's output scattered back into every
// stream. `linear_attn_out-0` is llama.cpp's own value for the block output,
// so this checks the combine alone rather than the whole layer — which is the
// point, since DeltaNet does not exist yet.
func TestHCCombine(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(ids)
	cfg := m.Config.HCConfig()

	embd, err := m.Embeddings(ids)
	if err != nil {
		t.Fatal(err)
	}
	res := HCInit(cfg, embd, nTok)

	// The inject the combine uses is the one the *attn* mixer produced, and
	// the block output is the reference's, so any disagreement here is the
	// combine's own.
	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	c := cfg
	c.Act = RefQ8
	_, inject, _, _ := HCMix(c, w, res, nTok)

	blockOut, err := tr.Get("linear_attn_out-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	HCCombine(cfg, res, blockOut.Vals, inject, nTok)

	want, err := tr.Get("hc_combine-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(res, want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hc_combine-0: %v", r)
	if r.maxAbs > 2e-4 {
		t.Errorf("hc_combine-0: maxAbs %.3e over tolerance 2e-4 (%v)", r.maxAbs, r)
	}
}

// TestHCBlockAcrossLayers is the wider gate: every hyper-connection mixer and
// combine in the four layers the trace covers, each driven from llama.cpp's
// own value for its input so that a failure is that block's and not an
// upstream one.
//
// It is what catches a weight-indexing mistake, which the layer-0 tests
// cannot: `blk.N.hc_attn_*` and `blk.N.hc_ffn_*` are eight different tensor
// sets here, and HCMix would happily run any of them.
//
// Layer 1's attn mixer is absent on purpose. It is the PLE layer, so its
// input residual is the n-gram block's output rather than the previous
// layer's, and that tensor is not dumped — it belongs to L2's next step.
func TestHCBlockAcrossLayers(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(ids)
	cfg := m.Config.HCConfig()
	cfg.Act = RefQ8

	embd, err := m.Embeddings(ids)
	if err != nil {
		t.Fatal(err)
	}
	// resIn names where a mixer's input residual comes from in the trace.
	// "" means hc_init, which only layer 0 uses.
	get := func(name string, nth int) []float32 {
		t.Helper()
		d, err := tr.Get(name, nth)
		if err != nil {
			t.Fatalf("%s#%d: %v", name, nth, err)
		}
		return d.Vals
	}

	type mixCase struct {
		layer int
		side  string
		nth   int // which write of hc_norm-N etc. this mixer is
		in    []float32
	}
	cases := []mixCase{
		{0, "attn", 0, HCInit(cfg, embd, nTok)},
		{0, "ffn", 1, get("hc_combine-0", 0)},
		{1, "ffn", 1, get("hc_combine-1", 0)},
		{2, "attn", 0, get("l_last-1", 0)},
		{2, "ffn", 1, get("hc_combine-2", 0)},
		{3, "attn", 0, get("l_last-2", 0)},
		{3, "ffn", 1, get("hc_combine-3", 0)},
	}
	for _, tc := range cases {
		w, err := m.HCWeights(tc.layer, tc.side)
		if err != nil {
			t.Fatalf("blk.%d.hc_%s: %v", tc.layer, tc.side, err)
		}
		if testing.Verbose() {
			// Which numerics fits this particular dispatch better, reported
			// per mixer: the backend does not take the same path for all of
			// them, and the gate tolerance below has to cover the worse case.
			ref, err := tr.Get(fmt.Sprintf("hc_gate-%d", tc.layer), tc.nth)
			if err != nil {
				t.Fatal(err)
			}
			for _, act := range []Numerics{Exact, RefQ8} {
				c2 := cfg
				c2.Act = act
				_, _, _, g2 := HCMix(c2, w, tc.in, nTok)
				r2, err := compare(g2, ref.Vals)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("  blk.%d.hc_%-4s gate act=%d rms %.3e maxAbs %.3e", tc.layer, tc.side, act, r2.rms, r2.maxAbs)
			}
		}
		mixed, inject, xn, gate := HCMix(cfg, w, tc.in, nTok)
		// The two tensors downstream of a Q8_0 matmul are bounded on rms
		// rather than on maxAbs, and loosely. Four of the seven mixers here
		// reproduce llama.cpp's gate to 6e-08 rms — f32 round-off over 71680
		// values through two matmuls, a SiLU and a sigmoid, which is what
		// proves the formula, the weight indexing and the layout. The other
		// three sit at 6e-06 to 3.4e-04 because an int8 grid turns a
		// last-bit difference in accumulation order into a whole quantisation
		// step: one flipped element of `lo`, which is only 320 values in ten
		// blocks, moves every one of the 10240 gate outputs. The bound is set
		// above the worst of those and the spread is logged.
		for _, o := range []struct {
			name   string
			got    []float32
			maxAbs float64 // 0 means "bound the rms instead"
			rms    float64
		}{
			{"hc_norm", xn, 1e-5, 0},
			{"hc_inject", inject, 3e-4, 0},
			{"hc_gate", gate, 0, 1e-3},
			{"hc_mixed", mixed, 0, 2e-3},
		} {
			ref := fmt.Sprintf("%s-%d", o.name, tc.layer)
			r, err := compare(o.got, get(ref, tc.nth))
			if err != nil {
				t.Fatalf("%s#%d: %v", ref, tc.nth, err)
			}
			t.Logf("blk.%d.hc_%-4s %-10s %v", tc.layer, tc.side, o.name, r)
			switch {
			case o.maxAbs > 0 && r.maxAbs > o.maxAbs:
				t.Errorf("blk.%d.hc_%s %s: maxAbs %.3e over %.1e", tc.layer, tc.side, o.name, r.maxAbs, o.maxAbs)
			case o.rms > 0 && r.rms > o.rms:
				t.Errorf("blk.%d.hc_%s %s: rms %.3e over %.1e", tc.layer, tc.side, o.name, r.rms, o.rms)
			}
		}
	}

	// The combines, likewise: one per block, taking the reference's own
	// block output and inject so that only the scatter is under test.
	type combCase struct {
		layer    int
		resIn    []float32
		blockOut string
		injNth   int
		want     string
		wantNth  int
	}
	combs := []combCase{
		{0, HCInit(cfg, embd, nTok), "linear_attn_out-0", 0, "hc_combine-0", 0},
		{0, get("hc_combine-0", 0), "ffn_out-0", 1, "l_last-0", 0},
		{1, get("hc_combine-1", 0), "ffn_out-1", 1, "l_last-1", 0},
		{2, get("l_last-1", 0), "linear_attn_out-2", 0, "hc_combine-2", 0},
		{2, get("hc_combine-2", 0), "ffn_out-2", 1, "l_last-2", 0},
		{3, get("l_last-2", 0), "attn_output-3", 0, "hc_combine-3", 0},
		{3, get("hc_combine-3", 0), "ffn_out-3", 1, "l_last-3", 0},
	}
	for _, tc := range combs {
		res := append([]float32(nil), tc.resIn...)
		inject := get(fmt.Sprintf("hc_inject-%d", tc.layer), tc.injNth)
		HCCombine(cfg, res, get(tc.blockOut, 0), inject, nTok)
		r, err := compare(res, get(tc.want, tc.wantNth))
		if err != nil {
			t.Fatalf("%s: %v", tc.want, err)
		}
		t.Logf("combine -> %-14s %v", tc.want, r)
		if r.maxAbs > 1e-5 {
			t.Errorf("%s: maxAbs %.3e over 1e-5 (%v)", tc.want, r.maxAbs, r)
		}
	}
}
