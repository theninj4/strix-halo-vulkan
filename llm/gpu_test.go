package llm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

// denseQ8Test is which dense bank the GPU tests stage. L8's is the default,
// because it is what the graph runs; `LLM_DENSE_FP16=1` puts them back on the
// halves, which is how the two are compared against the same dump.
var denseQ8Test = DenseQ8()

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("llm-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// mixerInput returns llama.cpp's own value for the residual a mixer reads, so
// that a disagreement is that mixer's and not an accumulation of every block
// before it. The chain is the model's: layer 0's attn mixer reads `hc_init`,
// every other attn mixer reads the previous layer's output `l_last`, and a
// layer's ffn mixer reads its own attn combine. The two combines are dumped
// under different names — `hc_combine-L` is the first and `l_last-L` the
// second — which is why this is a switch and not an index.
//
// Layer 1's attn mixer is the exception, and the reason is the whole of L2d:
// the PLE n-gram block runs between `l_last-0` and it, adding into the
// residual, so the tensor that mixer reads is not in the trace at all. It is
// built here by running the block — which is also the end-to-end check that
// the two stages compose.
func mixerInput(t *testing.T, m *Model, tr *Trace, layer int, side string) []float32 {
	t.Helper()
	get := func(name string, nth int) []float32 {
		d, err := tr.Get(name, nth)
		if err != nil {
			t.Fatal(err)
		}
		return d.Vals
	}
	switch {
	case layer == 0 && side == "attn":
		return get("hc_init", 0)
	case layer == 1 && side == "attn":
		c, ok, err := m.PLEConfig()
		if err != nil || !ok {
			t.Fatalf("PLE config: %v (present %v)", err, ok)
		}
		c.Act = RefQ8
		w, err := m.PLEWeights(c.Layers[0])
		if err != nil {
			t.Fatal(err)
		}
		res := append([]float32(nil), get("l_last-0", 0)...)
		PLEBlock(c, w, res, get("ple_embd", 0), len(res)/c.Wide())
		return res
	case side == "attn":
		return get(fmt.Sprintf("l_last-%d", layer-1), 0)
	default:
		return get(fmt.Sprintf("hc_combine-%d", layer), 0)
	}
}

// gpuFixture stages one mixer's weights on the device, with the gate arena
// on, and runs it over llama.cpp's own input.
func gpuFixture(t *testing.T, layer int, side string) (*HCGPU, *Trace, int) {
	t.Helper()
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	w, err := m.HCWeights(layer, side)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewHCGPU(dev, m.Config.HCConfig(), len(ids), []HCWeights{w}, HCOpts{Gate: true, Q8: denseQ8Test})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	in := mixerInput(t, m, tr, layer, side)
	if err := g.Upload(in, len(ids)); err != nil {
		t.Fatal(err)
	}
	return g, tr, len(ids)
}

// fp16 tolerances for this block, as rms bounds, and measured rather than
// chosen — the same discipline L2b settled on for the CPU reference
// (research/l2b-hyper-connections.md): an int8 or fp16 grid makes the error
// heavy-tailed, so one flipped rounding in `lo`'s 320 values moves all 10240
// gate outputs and maxAbs says nothing about the kernel.
//
// Two references, and the difference between them is the point. Against the
// CPU reference in Exact mode the only difference is the fp16 operands, so
// these bounds are what narrowing xn, the weights and lo to eleven mantissa
// bits costs. Against llama.cpp the *reference* is the one carrying error:
// its Vulkan backend evaluates these Q8_0 matmuls over int8 activations, and
// L2b measured that at 3.1e-03 rms on the gate — an order of magnitude past
// anything below, and the reason the trace bounds are the looser pair.
// The worst of the eight mixers of layers 0-3 is quoted against each.
const (
	tolXn     = 5e-4 // worst 2.74e-04, and maxRel is 4.9e-04 — fp16's own step
	tolGate   = 3e-4 // worst 1.24e-04, on a gate in [0, 1]
	tolMixed  = 3e-4 // worst 1.15e-04
	tolInject = 3e-3 // worst 1.13e-03, on values to |71.7|

	tolLlamaXn = 1e-3 // worst 2.74e-04: this tensor's weights are F32 either side
	tolLlamaG  = 1e-2 // worst 4.67e-03, against 8e-05 from the CPU reference
	tolLlamaM  = 2e-2 // worst 6.53e-03
)

// bankTol widens a trace bound where the default bank's *deliberate* width
// change moved it: P2 put the fp16 tail's tensors — `inject`, `ssm_alpha`,
// `ssm_beta` and the indexer's two projections — on the quantised plane at
// D13's measured price (+0.01% of corpus perplexity for the int8 three,
// L8c-1), so the staged model now differs from the reference by more than
// the kernels' own rounding on exactly those tensors. The fp16 bank
// (LLM_DENSE_FP16=1) keeps the tight bound, which is what makes these tests
// still a kernel gate rather than a width gate. A structural bug — a wrong
// plane, a wrong record — is orders of magnitude past either bound.
func bankTol(fp16, quantised float64) float64 {
	if denseQ8Test {
		return quantised
	}
	return fp16
}

// TestHCGPUMixer is the L2 gate for the fused kernel: every tensor of the
// first mixer of layer 0 — whose input is `hc_init`, so nothing upstream can
// be wrong — against both the CPU reference it was ported from and
// llama.cpp's own activations.
func TestHCGPUMixer(t *testing.T) {
	g, tr, nTok := gpuFixture(t, 0, "attn")
	m, _ := fixtures(t)
	cfg := m.Config.HCConfig()
	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	in := mixerInput(t, m, tr, 0, "attn")
	wantMixed, wantInject, wantXn, wantGate := HCMix(cfg, w, in, nTok)

	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		got  []float32
		cpu  []float32
		tol  float64
		ref  string
		rtol float64
	}{
		{"hc_norm-0", g.Xn(), wantXn, tolXn, "hc_norm-0", tolLlamaXn},
		{"hc_inject-0", g.Inject(), wantInject, bankTol(tolInject, 4e-2), "hc_inject-0", bankTol(tolInject, 4e-2)},
		{"hc_gate-0", g.Gate(), wantGate, tolGate, "hc_gate-0", tolLlamaG},
		{"hc_mixed-0", g.Mixed(), wantMixed, tolMixed, "hc_mixed-0", tolLlamaM},
	} {
		r, err := compare(tc.got, tc.cpu)
		if err != nil {
			t.Fatalf("%s against the CPU reference: %v", tc.name, err)
		}
		t.Logf("%-12s vs CPU Exact   %v", tc.name, r)
		if r.rms > tc.tol {
			t.Errorf("%s: rms %.3e against the CPU reference, over %.1e (%v)", tc.name, r.rms, tc.tol, r)
		}
		want, err := tr.Get(tc.ref, 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err = compare(tc.got, want.Vals)
		if err != nil {
			t.Fatalf("%s against llama.cpp: %v", tc.name, err)
		}
		t.Logf("%-12s vs llama.cpp   %v", tc.name, r)
		if r.rms > tc.rtol {
			t.Errorf("%s: rms %.3e against llama.cpp, over %.1e (%v)", tc.name, r.rms, tc.rtol, r)
		}
	}
}

// TestHCGPUEveryMixer runs all eight mixers of the four dumped layers, each
// from llama.cpp's own residual. The weights differ per mixer and so does the
// input, so this is the check that the staging is indexed right — a wrong
// bank offset produces a perfectly plausible tensor, which is the hazard
// zimage/qwen's shiftBank control exists for.
func TestHCGPUEveryMixer(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	nTok := len(ids)
	cfg := m.Config.HCConfig()
	dev, done := newTestDevice(t)
	defer done()

	type mixerRef struct {
		layer int
		side  string
		nth   int // which occurrence of the layer's tensors this mixer is
	}
	var refs []mixerRef
	var ws []HCWeights
	for layer := 0; layer < 4; layer++ {
		for nth, side := range []string{"attn", "ffn"} {
			w, err := m.HCWeights(layer, side)
			if err != nil {
				t.Fatal(err)
			}
			ws = append(ws, w)
			refs = append(refs, mixerRef{layer, side, nth})
		}
	}
	g, err := NewHCGPU(dev, cfg, nTok, ws, HCOpts{Gate: true, Q8: denseQ8Test})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("%d mixers staged: %.1f MB of weights, %.1f MB of arenas",
		g.Mixers(), float64(g.WeightBytes())/1e6, float64(g.ActivationBytes())/1e6)

	for i, ref := range refs {
		in := mixerInput(t, m, tr, ref.layer, ref.side)
		if err := g.Upload(in, nTok); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(i, false); err != nil {
			t.Fatal(err)
		}
		// The CPU reference is run on the same input, so each tensor has
		// two bounds: a tight one against the mathematical model, which is
		// the fp16 operands alone, and a loose one against llama.cpp, which
		// carries the reference's own int8 activations.
		cpuMixed, cpuInject, cpuXn, cpuGate := HCMix(cfg, ws[i], in, nTok)
		for _, tc := range []struct {
			name      string
			got, cpu  []float32
			tol, rtol float64
		}{
			{"hc_norm", g.Xn(), cpuXn, tolXn, tolLlamaXn},
			{"hc_gate", g.Gate(), cpuGate, tolGate, tolLlamaG},
			{"hc_mixed", g.Mixed(), cpuMixed, tolMixed, tolLlamaM},
			{"hc_inject", g.Inject(), cpuInject, bankTol(tolInject, 4e-2), bankTol(tolInject, 4e-2)},
		} {
			r, err := compare(tc.got, tc.cpu)
			if err != nil {
				t.Fatalf("blk.%d.hc_%s %s against the CPU reference: %v", ref.layer, ref.side, tc.name, err)
			}
			if r.rms > tc.tol {
				t.Errorf("blk.%d.hc_%s %s: rms %.3e against the CPU reference, over %.1e (%v)",
					ref.layer, ref.side, tc.name, r.rms, tc.tol, r)
			}
			cpuRMS := r.rms
			want, err := tr.Get(fmt.Sprintf("%s-%d", tc.name, ref.layer), ref.nth)
			if err != nil {
				t.Fatal(err)
			}
			if r, err = compare(tc.got, want.Vals); err != nil {
				t.Fatalf("blk.%d.hc_%s %s: %v", ref.layer, ref.side, tc.name, err)
			}
			t.Logf("blk.%d.hc_%-4s %-10s cpu rms %.3e | llama.cpp %v",
				ref.layer, ref.side, tc.name, cpuRMS, r)
			if r.rms > tc.rtol {
				t.Errorf("blk.%d.hc_%s %s: rms %.3e against llama.cpp, over %.1e (%v)",
					ref.layer, ref.side, tc.name, r.rms, tc.rtol, r)
			}
		}
	}
}

// TestHCGPUCombine closes the block on the device: the mix, then the scatter
// back into every stream, against `hc_combine-0`. The block output is
// llama.cpp's own (`linear_attn_out-0`), since DeltaNet does not exist yet,
// so a disagreement here is the combine's or the inject's.
func TestHCGPUCombine(t *testing.T) {
	g, tr, _ := gpuFixture(t, 0, "attn")
	out, err := tr.Get("linear_attn_out-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.UploadBlockOut(out.Vals); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0, true); err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("hc_combine-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Res(), want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hc_combine-0: %v", r)
	// The residual is |37.9| at this point and the scatter weight is a
	// sigmoid of inject, so this inherits inject's fp16 error through a
	// saturating nonlinearity and little else.
	if r.rms > 2e-3 {
		t.Errorf("hc_combine-0: rms %.3e over 2e-3 (%v)", r.rms, r)
	}
}

// TestHCGPULadderAgrees runs every rung of both ladders over the same input
// and requires them to produce the same tensors. The rungs differ only in how
// many token rows a workgroup carries, so a disagreement is a padding or an
// indexing bug — which at a 7-token prompt is exactly what a BM of 64 would
// expose.
func TestHCGPULadderAgrees(t *testing.T) {
	g, _, _ := gpuFixture(t, 0, "attn")
	if err := g.SetPlan(HCDownM4, HCUpM4); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}
	baseMixed, baseInject := g.Mixed(), g.Inject()

	for _, down := range DownKernels() {
		for _, up := range UpKernels() {
			if err := g.SetPlan(down, up); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(0, false); err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct {
				name string
				got  []float32
				want []float32
			}{
				{"mixed", g.Mixed(), baseMixed},
				{"inject", g.Inject(), baseInject},
			} {
				r, err := compare(tc.got, tc.want)
				if err != nil {
					t.Fatalf("%s/%s %s: %v", down, up, tc.name, err)
				}
				if r.maxAbs != 0 {
					t.Errorf("%s/%s %s disagrees with down_m4/up_m4: %v", down, up, tc.name, r)
				}
			}
		}
	}
}

// TestHCGPUIsNearerTheModelThanTheReference states the measurement the
// tolerances above rest on, so that it is a checked claim and not a comment.
//
// The fused kernel's operands are fp16 and llama.cpp's are int8, and this is
// what that is worth: against the same f32 model, on the same input, the
// kernel is more than an order of magnitude closer than the oracle it is
// being validated against. Which is why every bound here is an rms against
// the CPU reference, with the trace as the looser second opinion — L2b's
// conclusion, now measured from the other side.
func TestHCGPUIsNearerTheModelThanTheReference(t *testing.T) {
	g, tr, nTok := gpuFixture(t, 0, "attn")
	m, _ := fixtures(t)
	cfg := m.Config.HCConfig()
	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	in := mixerInput(t, m, tr, 0, "attn")
	_, _, _, model := HCMix(cfg, w, in, nTok)
	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}
	ours, err := compare(g.Gate(), model)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := tr.Get("hc_gate-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	theirs, err := compare(ref.Vals, model)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("gate against the f32 model: fused kernel rms %.3e, llama.cpp rms %.3e, %.0fx",
		ours.rms, theirs.rms, theirs.rms/ours.rms)
	if theirs.rms/ours.rms < 10 {
		t.Errorf("the fused kernel is only %.1fx nearer the model than llama.cpp; fp16 operands should be far better than int8 ones",
			theirs.rms/ours.rms)
	}
}

// TestHCGPUUnpermutedUpIsWrong is the negative control for the one layout
// decision the fusion rests on. The up projection's rows are permuted at
// upload so that the four streams of a feature land in one workgroup, which
// is what lets the kernel collapse the gate instead of writing it; staging
// the checkpoint's own row order instead is not a crash and not a NaN, it is
// a plausible tensor built from the wrong four columns. Nothing but a test
// that demands it disagree distinguishes the two.
func TestHCGPUUnpermutedUpIsWrong(t *testing.T) {
	m, tr := fixtures(t)
	ids, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	w, err := m.HCWeights(0, "attn")
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewHCGPU(dev, m.Config.HCConfig(), len(ids), []HCWeights{w}, HCOpts{UnpermutedUp: true, Q8: denseQ8Test})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	in := mixerInput(t, m, tr, 0, "attn")
	if err := g.Upload(in, len(ids)); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}
	want, err := tr.Get("hc_mixed-0", 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := compare(g.Mixed(), want.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("unpermuted up: %v", r)
	if r.rms < 10*tolLlamaM {
		t.Errorf("staging the up projection unpermuted still lands within %.3e rms of llama.cpp, so the permutation is not doing what the kernel says it does", r.rms)
	}
}

// TestHCGPUGemvAgrees is the L7d gate for the decode rungs: at one token the
// split-K GEMV has to produce what the GEMM rung produces, for every split.
//
// It cannot be a bit-for-bit comparison, and that is the whole reason this is
// a separate test rather than another arm of TestHCGPULadderAgrees. The GEMM
// rungs agree with each other exactly because they differ only in how many
// token rows a workgroup carries — the reduction over K is the same sequence
// of cooperative-matrix steps in all three. The GEMV reduces K in a different
// order and in a different place: `KSLABS` partial sums, each accumulated by
// one lane over sixteen-wide strides and closed by two shuffles, then summed
// slab by slab in the second dispatch. Every product is the same fp32 product
// of the same two fp16 operands; only the association changes. So the bound
// below is fp32 round-off over a 10240-long dot product, and the test asserts
// it is that small rather than asserting zero.
//
// `lo` is the tensor to bound, because it is the one the up projection reads
// and it carries a silu; `inject` is the same accumulator's other sixteen
// columns, written fp32 without an epilogue.
func TestHCGPUGemvAgrees(t *testing.T) {
	g, _, _ := gpuFixture(t, 0, "attn")
	m, tr := fixtures(t)
	in := mixerInput(t, m, tr, 0, "attn")
	// One token, which is what the decode rungs are for. The row is
	// llama.cpp's own, so the magnitudes are the model's.
	if err := g.Upload(in[:g.cfg.Wide()], 1); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPlan(HCDownM1, HCUpM1); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}
	baseLo, baseInject, baseMixed := g.Lo(), g.Inject(), g.Mixed()

	// The scale the bounds are relative to, so a change in the fixture is
	// visible rather than silently absorbed.
	var loMax float64
	for _, v := range baseLo {
		loMax = max(loMax, math.Abs(float64(v)))
	}
	for _, down := range GemvKernels() {
		if err := g.SetPlan(down, HCUpM1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0, false); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name   string
			got    []float32
			want   []float32
			maxAbs float64
		}{
			// `lo` is fp16 in the arena, so its own step is the floor: the
			// bound is one ulp at the magnitudes this mixer reaches.
			{"lo", g.Lo(), baseLo, 1e-2},
			{"inject", g.Inject(), baseInject, 1e-3},
			{"mixed", g.Mixed(), baseMixed, 1e-3},
		} {
			r, err := compare(tc.got, tc.want)
			if err != nil {
				t.Fatalf("%s %s: %v", down, tc.name, err)
			}
			t.Logf("%-14s %-7s %v", down, tc.name, r)
			if r.maxAbs > tc.maxAbs {
				t.Errorf("%s %s disagrees with down_m1: maxAbs %.3e over %.1e (|lo| to %.1f)",
					down, tc.name, r.maxAbs, tc.maxAbs, loMax)
			}
		}
	}
}

// TestHCGPUGemvTwoRows is P5b's gate on this block: the decode down rungs
// carrying two rows of A compute what the GEMM computes, in all three of the
// epilogue's outputs.
//
// The block is the one whose epilogue is not a plain row store — the gate's
// branch is narrowed into fp16 at `ldaLo` and `inject`'s tile is written f32
// at its own stride — so what this adds over the DeltaNet's and the attention
// layer's two-row tests is that the *second row* of each lands where the
// combine reads it. A kernel that wrote both rows at row 0's offset would
// pass a comparison of row 0 and fail this one.
func TestHCGPUGemvTwoRows(t *testing.T) {
	const rows = 2
	if rows > GEMVMaxRows {
		t.Skipf("GEMVMaxRows is %d", GEMVMaxRows)
	}
	g, _, nTok := gpuFixture(t, 0, "attn")
	if nTok < rows {
		t.Skipf("the fixture is %d tokens", nTok)
	}
	m, tr := fixtures(t)
	in := mixerInput(t, m, tr, 0, "attn")
	if err := g.Upload(in[:rows*g.cfg.Wide()], rows); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPlan(HCDownM1, HCUpM1); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(0, false); err != nil {
		t.Fatal(err)
	}
	baseLo, baseInject, baseMixed := append([]float32(nil), g.Lo()...),
		append([]float32(nil), g.Inject()...), append([]float32(nil), g.Mixed()...)

	// **The control that matters, and the one a weaker test misses.**
	// Comparing the GEMV's second row against the GEMM's passes trivially
	// when the GEMV does not write it at all: the arena still holds what the
	// GEMM put there. So the second token is *changed* and the second output
	// row has to move. (This caught a real one — the `.spv` are generated and
	// gitignored, so a kernel edit that is not followed by `go generate`
	// leaves the old module embedded and every row-against-row comparison
	// passes on stale bytes.)
	moved := func(down HCKernel) {
		t.Helper()
		row1 := func(second int) []float32 {
			buf := make([]float32, 2*g.cfg.Wide())
			copy(buf, in[:g.cfg.Wide()])
			copy(buf[g.cfg.Wide():], in[second*g.cfg.Wide():(second+1)*g.cfg.Wide()])
			if err := g.Upload(buf, rows); err != nil {
				t.Fatal(err)
			}
			if err := g.SetPlan(down, HCUpM1); err != nil {
				t.Fatal(err)
			}
			if err := g.Run(0, false); err != nil {
				t.Fatal(err)
			}
			inj := g.Inject()
			return append([]float32(nil), inj[len(inj)/rows:]...)
		}
		a, b := row1(1), row1(2)
		same := true
		for i := range a {
			if a[i] != b[i] {
				same = false
				break
			}
		}
		if same {
			t.Errorf("%s: the second output row did not move when the second token did — "+
				"the rung is not writing row 1", down)
		}
	}

	for _, down := range GemvKernels() {
		moved(down)
		// Back to the pair the comparison below is against.
		if err := g.Upload(in[:rows*g.cfg.Wide()], rows); err != nil {
			t.Fatal(err)
		}
		if err := g.SetPlan(down, HCUpM1); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0, false); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name   string
			got    []float32
			want   []float32
			maxAbs float64
		}{
			{"lo", g.Lo(), baseLo, 1e-2},
			{"inject", g.Inject(), baseInject, 1e-3},
			{"mixed", g.Mixed(), baseMixed, 1e-3},
		} {
			// Whole tensor, then the second row alone — half the values, and
			// the half a row-blind kernel would get wrong.
			for _, half := range []struct {
				what string
				got  []float32
				want []float32
			}{
				{"both rows", tc.got, tc.want},
				{"row 1", tc.got[len(tc.got)/rows:], tc.want[len(tc.want)/rows:]},
			} {
				r, err := compare(half.got, half.want)
				if err != nil {
					t.Fatalf("%s %s %s: %v", down, tc.name, half.what, err)
				}
				t.Logf("%-14s %-7s %-9s %v", down, tc.name, half.what, r)
				if r.maxAbs > tc.maxAbs {
					t.Errorf("%s %s (%s) disagrees with down_m1 at %d rows: maxAbs %.3e over %.1e",
						down, tc.name, half.what, rows, r.maxAbs, tc.maxAbs)
				}
			}
		}
	}
}

// TestHCGPUGemvRefusesABatch is D15's refusal, at the bound P5b moved it to.
//
// The rung is no longer "one token": it carries up to `GEMVMaxRows` rows of
// A, because a verification pass is two rows and a dot product extends to
// them for the price of the second row's operand. What has not changed is
// that the bound is *enforced* — a batch past it would read one row and
// silently drop the rest, which is exactly what D15 exists to prevent.
func TestHCGPUGemvRefusesABatch(t *testing.T) {
	g, _, nTok := gpuFixture(t, 0, "attn")
	if nTok <= GEMVMaxRows {
		t.Skipf("the fixture is %d tokens and the rung carries %d", nTok, GEMVMaxRows)
	}
	if err := g.SetPlan(HCDownGemv160, HCUpM1); err == nil {
		t.Fatalf("SetPlan took the decode rung for a %d-token run, past the %d it carries",
			nTok, GEMVMaxRows)
	}
	// And it is taken at the bound, which is the half of the assertion P5b
	// adds: a refusal that refused everything would pass the line above.
	if err := g.Resize(GEMVMaxRows); err != nil {
		t.Fatal(err)
	}
	if err := g.SetPlan(HCDownGemv160, HCUpM1); err != nil {
		t.Fatalf("SetPlan refused the decode rung at %d rows, which is the bound: %v", GEMVMaxRows, err)
	}
}

// TestHCGPUQ8IsTheHalves is the L8b gate, and it is an equality rather than a
// tolerance for the reason L8a's was: the int8 bank and the fp16 bank hold
// **the same halves**.
//
// Both of this block's projections are Q8_0 in the checkpoint, so re-deriving
// (d, q) from the dequantised floats returns the checkpoint's own pair
// (TestBankQ8RoundTrip) and `float(q) * float(d)` rounded to fp16 is what
// tileB wrote. The GEMM arm puts those halves in LDS instead of reading them
// from global, and the GEMV arm multiplies in fp16 so that the rounding the
// LDS store gives the GEMM for free happens in a register too — which it
// would not if the product were formed in f32 and converted, because RADV
// folds that pair away (D10). So every rung of both ladders has to agree with
// the fp16 bank to the last bit — `inject` included, now that it sits on the
// plane with everything else (P2).
//
// The weights are synthetic and that is deliberate: the claim is a property
// of the format, not of this checkpoint's values, and a [10240, 320] pair out
// of the real model is 13 MB a mixer to make a point about rounding.
func TestHCGPUQ8IsTheHalves(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	// The checkpoint's own widths, because the GEMV rungs split 640 k-tiles
	// up to 160 ways and a narrower K has no such ladder. The weights are
	// synthetic and that is the point: this is a property of the format.
	cfg := HCConfig{NEmbd: 2560, HC: 4, LowRank: 320, Eps: 1e-6}
	wide := cfg.Wide()
	rng := rand.New(rand.NewSource(11))

	// Down, Up and Inject are genuine Q8_0 values — a scale per 32 and a
	// level at 127 in every group — because that is what makes the round
	// trip exact. The real `inject` is F32 and its int8 staging is a real
	// re-quantisation (P2); values on the int8 grid keep this an *equality*
	// test of the addressing, while the accuracy of quantising the real
	// inject is the corpus's question, not this test's.
	down, _, _ := q8Source(rng, cfg.LowRank, wide)
	up, _, _ := q8Source(rng, wide, cfg.LowRank)
	inject, _, _ := q8Source(rng, cfg.HC, wide)
	w := HCWeights{
		Norm:   make([]float32, wide),
		Down:   down,
		Up:     up,
		Inject: inject,
	}
	for i := range w.Norm {
		w.Norm[i] = 1 + float32(rng.NormFloat64())*0.1
	}

	const nTok = 5
	res := make([]float32, nTok*wide)
	for i := range res {
		res[i] = float32(rng.NormFloat64())
	}
	blockOut := make([]float32, nTok*cfg.NEmbd)
	for i := range blockOut {
		blockOut[i] = float32(rng.NormFloat64())
	}

	stage := func(q8 bool) *HCGPU {
		t.Helper()
		g, err := NewHCGPU(dev, cfg, nTok, []HCWeights{w}, HCOpts{Gate: true, Q8: q8})
		if err != nil {
			t.Fatalf("hc (q8=%v): %v", q8, err)
		}
		return g
	}
	// Destroyed before the device is, which is why this is a defer and not a
	// t.Cleanup: cleanups run after the deferred device teardown above.
	half, q8 := stage(false), stage(true)
	defer half.Destroy()
	defer q8.Destroy()
	t.Logf("banks: %.2f MB of int8 and scales against %.2f MB of halves",
		float64(q8.WeightBytes())/1e6, float64(half.WeightBytes())/1e6)
	if q8.WeightBytes() >= half.WeightBytes() {
		t.Fatalf("the q8 bank is %d bytes, no smaller than the fp16 one", q8.WeightBytes())
	}

	type out struct {
		xn, lo, inject, mixed, gate, combined []float32
	}
	exec := func(g *HCGPU, dk, uk HCKernel, rows int) out {
		t.Helper()
		if err := g.Upload(res[:rows*wide], rows); err != nil {
			t.Fatalf("upload: %v", err)
		}
		if err := g.SetPlan(dk, uk); err != nil {
			t.Fatalf("plan %s/%s: %v", dk, uk, err)
		}
		if err := g.Run(0, false); err != nil {
			t.Fatalf("run %s/%s: %v", dk, uk, err)
		}
		o := out{xn: g.Xn(), lo: g.Lo(), inject: g.Inject(), mixed: g.Mixed(), gate: g.Gate()}
		if err := g.UploadBlockOut(blockOut[:rows*cfg.NEmbd]); err != nil {
			t.Fatalf("block out: %v", err)
		}
		if err := g.RunCombine(0); err != nil {
			t.Fatalf("combine: %v", err)
		}
		o.combined = g.Res()
		return o
	}

	equal := func(what string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: %d values against %d", what, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s[%d]: q8 %.9g, fp16 %.9g", what, i, a[i], b[i])
			}
		}
	}
	compareRuns := func(label string, a, b out, tensors ...string) {
		t.Helper()
		got := map[string][2][]float32{
			"hc_norm":    {a.xn, b.xn},
			"lo":         {a.lo, b.lo},
			"hc_inject":  {a.inject, b.inject},
			"hc_gate":    {a.gate, b.gate},
			"hc_mixed":   {a.mixed, b.mixed},
			"hc_combine": {a.combined, b.combined},
		}
		for _, name := range tensors {
			equal(label+" "+name, got[name][0], got[name][1])
		}
	}

	// Every rung of both GEMM ladders at a length where the row blocks
	// differ, then the decode pair at one token.
	all := []string{"hc_norm", "lo", "hc_inject", "hc_gate", "hc_mixed", "hc_combine"}
	for _, dk := range DownKernels() {
		for _, uk := range UpKernels() {
			compareRuns(fmt.Sprintf("%s/%s", dk, uk),
				exec(q8, dk, uk, nTok), exec(half, dk, uk, nTok), all...)
		}
	}
	for _, dk := range GemvKernels() {
		compareRuns(string(dk), exec(q8, dk, HCUpM1, 1), exec(half, dk, HCUpM1, 1),
			"lo", "hc_inject", "hc_mixed", "hc_combine")
	}
	t.Logf("%d GEMM pairs and %d split-K rungs, every tensor identical",
		len(DownKernels())*len(UpKernels()), len(GemvKernels()))
}

// TestHCGPUQ8BankSize states what L8b stages against what it replaces, at the
// checkpoint's own widths. There is no fp16 tail any more (P2): the whole
// fused N — `inject` and the pad rows included — is int8 plus scales, so what
// a token reads is exactly what is staged.
func TestHCGPUQ8BankSize(t *testing.T) {
	cfg := HCConfig{NEmbd: 2560, HC: 4, LowRank: 320}
	g := &HCGPU{cfg: cfg}
	wide, n := cfg.Wide(), g.gemmN()
	half := (n*wide + wide*cfg.LowRank) * 2
	q8 := q8Align(q8Bytes(n, wide)) + q8Align(q8Bytes(wide, cfg.LowRank))
	t.Logf("staged and read %.2f MB against %.2f of halves", float64(q8)/1e6, float64(half)/1e6)
	if ratio := float64(half) / float64(q8); ratio < 1.85 || ratio > 1.91 {
		t.Fatalf("a token reads %.3fx less, want ~1.88", ratio)
	}
}

// TestHCGPUQ4IsTheSim is L8c-6's first gate, and it is the same shape as the
// lm head's and the gated DeltaNet's: **the bank against the simulation that
// chose it**, value for value, over a whole mixer.
//
// L8c-3's +4.24% is a perplexity measured through `sim.go`, which stages a
// candidate format's floats and lets the fp16 kernels multiply them. This
// bank stores that format instead, through the same encoder, so the halves
// the matrix cores see are the same halves and the two arms have to agree
// exactly.
//
// Two things in this block are not the DeltaNet's, and both are checked here.
// **The up projection's k is 320**, so its super-block is the whole row —
// ten groups — and a wrong record packing would show as a wrong scale on one
// group in ten rather than as garbage. And **the fp16 tail is no longer
// free**: the 32 low-rank rows between the split and `inject` are read out of
// the tail plane, so if `stageQ4` staged the *unquantised* rows there the
// bank would be more accurate than the format it claims to be — which is a
// difference `hc_mixed` shows and `hc_inject` does not, because inject's four
// rows are F32 in the checkpoint and stay halves on both arms (D13).
//
// It is `rtn` rather than `imatrix` so the test needs nothing but a device;
// the calibrated arm is the same code path with `qw` non-nil, and
// TestBankQ4KIsTheSim covers that at the encoder.
//
// Both widths, since P3a. This block is the one that stages **two** record
// packings — the down projection's eight groups and the 320-wide up
// projection's ten — so it is where the fifth bit's plane has to survive a
// super-block that is the whole row.
func TestHCGPUQ4IsTheSim(t *testing.T) {
	for _, spec := range []string{"q4_k/32", "q5_k/32"} {
		t.Run(spec, func(t *testing.T) { hcBankIsTheSim(t, spec) })
	}
}

func hcBankIsTheSim(t *testing.T, spec string) {
	dev, done := newTestDevice(t)
	defer done()

	cfg := HCConfig{NEmbd: 2560, HC: 4, LowRank: 320, Eps: 1e-6}
	wide := cfg.Wide()
	rng := rand.New(rand.NewSource(23))
	sim, err := ParseQuantSim(spec)
	if err != nil {
		t.Fatal(err)
	}
	sim.Mode = "rtn"
	bank, err := BankForSim(sim)
	if err != nil {
		t.Fatal(err)
	}

	w := HCWeights{
		Norm:   make([]float32, wide),
		Down:   make([]float32, cfg.LowRank*wide),
		Up:     make([]float32, wide*cfg.LowRank),
		Inject: make([]float32, cfg.HC*wide),
		Name:   "blk.0.hc_attn_",
	}
	for i := range w.Norm {
		w.Norm[i] = 1 + float32(rng.NormFloat64())*0.1
	}
	for _, x := range [][]float32{w.Down, w.Up, w.Inject} {
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * 0.05
		}
	}

	const nTok = 5
	res := make([]float32, nTok*wide)
	for i := range res {
		res[i] = float32(rng.NormFloat64())
	}
	blockOut := make([]float32, nTok*cfg.NEmbd)
	for i := range blockOut {
		blockOut[i] = float32(rng.NormFloat64())
	}

	stage := func(bank DenseBank) *HCGPU {
		t.Helper()
		g, err := NewHCGPU(dev, cfg, nTok, []HCWeights{w},
			HCOpts{Gate: true, Bank: bank, Q8: bank == BankQ8, Sim: sim})
		if err != nil {
			t.Fatalf("hc (%s): %v", bank, err)
		}
		return g
	}
	simulated, q4 := stage(BankFP16), stage(bank)
	defer simulated.Destroy()
	defer q4.Destroy()
	t.Logf("bank %.2f MB against the simulation's %.2f MB of halves",
		float64(q4.WeightBytes())/1e6, float64(simulated.WeightBytes())/1e6)
	if q4.WeightBytes() >= simulated.WeightBytes() {
		t.Fatalf("the q4_k bank is %d bytes, no smaller than %d",
			q4.WeightBytes(), simulated.WeightBytes())
	}

	type out struct {
		xn, lo, inject, mixed, gate, combined []float32
	}
	exec := func(g *HCGPU, dk, uk HCKernel, rows int) out {
		t.Helper()
		if err := g.Upload(res[:rows*wide], rows); err != nil {
			t.Fatalf("upload: %v", err)
		}
		if err := g.SetPlan(dk, uk); err != nil {
			t.Fatalf("plan %s/%s: %v", dk, uk, err)
		}
		if err := g.Run(0, false); err != nil {
			t.Fatalf("run %s/%s: %v", dk, uk, err)
		}
		o := out{xn: g.Xn(), lo: g.Lo(), inject: g.Inject(), mixed: g.Mixed(), gate: g.Gate()}
		if err := g.UploadBlockOut(blockOut[:rows*cfg.NEmbd]); err != nil {
			t.Fatalf("block out: %v", err)
		}
		if err := g.RunCombine(0); err != nil {
			t.Fatalf("combine: %v", err)
		}
		o.combined = g.Res()
		return o
	}

	equal := func(what string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: %d values against %d", what, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s[%d]: bank %.9g, simulation %.9g", what, i, a[i], b[i])
			}
		}
	}
	n := 0
	for _, dk := range DownKernels() {
		for _, uk := range UpKernels() {
			a, b := exec(q4, dk, uk, nTok), exec(simulated, dk, uk, nTok)
			label := fmt.Sprintf("%s/%s ", dk, uk)
			equal(label+"hc_norm", a.xn, b.xn)
			equal(label+"lo", a.lo, b.lo)
			equal(label+"hc_inject", a.inject, b.inject)
			equal(label+"hc_gate", a.gate, b.gate)
			equal(label+"hc_mixed", a.mixed, b.mixed)
			equal(label+"hc_combine", a.combined, b.combined)
			n += len(a.xn) + len(a.lo) + len(a.inject) + len(a.gate) + len(a.mixed) + len(a.combined)
		}
	}
	t.Logf("%d GEMM pairs, %d values identical to the simulation",
		len(DownKernels())*len(UpKernels()), n)

	// The split-K GEMV at one token is **not** bit-exact against the GEMM on
	// this bank and cannot be (L8c-4): a K-quant group is affine, so there is
	// no pair of exact halves to multiply and the register path carries one
	// fewer rounding than the GEMM's LDS store. What it has to be is the same
	// arithmetic to within that one rounding, over a 10240-long dot product.
	ref := exec(simulated, HCDownM1, HCUpM1, 1)
	for _, dk := range GemvKernels() {
		got := exec(q4, dk, HCUpM1, 1)
		var num, den float64
		for i := range ref.mixed {
			d := float64(got.mixed[i] - ref.mixed[i])
			num += d * d
			den += float64(ref.mixed[i]) * float64(ref.mixed[i])
		}
		rms := math.Sqrt(num / math.Max(den, 1e-30))
		if rms > 1e-3 {
			t.Fatalf("%s: hc_mixed is %.3e rms off the simulation's GEMM", dk, rms)
		}
		t.Logf("%-14s hc_mixed %.3e rms against the GEMM, one rounding apart", dk, rms)
	}
}

// TestHCGPUQ4BankSize states what L8c-6 stages against what it replaces, at
// the checkpoint's own widths, and says where the 4.500 bits stop being
// exact.
//
// Two places, and each is a decision rather than an approximation. The
// **plane covers the whole fused N** — `inject` and the twelve pad rows
// included, now that the fp16 tail is gone (P2) — so the pad rows are staged
// nibbles nothing reads. And the **up projection's record is twenty bytes
// over 320 weights** where ggml's is sixteen over 256: the simulation quotes
// that row at 4.475 bits because nothing there has to be addressable, and a
// word-aligned record makes it exactly 4.500.
func TestHCGPUQ4BankSize(t *testing.T) {
	cfg := HCConfig{NEmbd: 2560, HC: 4, LowRank: 320}
	g := &HCGPU{cfg: cfg}
	wide, n := cfg.Wide(), g.gemmN()

	half := (n*wide + wide*cfg.LowRank) * 2
	q8 := q8Align(q8Bytes(n, wide)) + q8Align(q8Bytes(wide, cfg.LowRank))
	q4 := q8Align(qkBytes(4, n, wide)) + q8Align(qkBytes(4, wide, cfg.LowRank))
	weights := cfg.LowRank*wide + wide*cfg.LowRank + cfg.HC*wide

	t.Logf("a mixer: q4_k %.2f MB, q8 %.2f MB, halves %.2f MB — %.3f, %.3f and %.3f bits a weight",
		float64(q4)/1e6, float64(q8)/1e6, float64(half)/1e6,
		float64(q4)*8/float64(weights), float64(q8)*8/float64(weights),
		float64(half)*8/float64(weights))
	t.Logf("97 mixers: %.2f GB against %.2f GB and %.2f GB",
		float64(97*q4)/1e9, float64(97*q8)/1e9, float64(97*half)/1e9)
	if q4 >= q8 || q8 >= half {
		t.Fatalf("the three banks are %d, %d, %d bytes and are not in order", q4, q8, half)
	}

	// The up projection alone is the 320-wide family, and it is exactly
	// 4.500 bits: nibbles plus one twenty-byte record per row.
	up := qkBytes(4, wide, cfg.LowRank)
	if bits := float64(up) * 8 / float64(wide*cfg.LowRank); bits != 4.5 {
		t.Fatalf("the up projection is %.4f bits a weight, want 4.500", bits)
	}
	if rec := q4kRecordBytes(hcUpSub); rec != 20 {
		t.Fatalf("a ten-group record is %d bytes, want 20", rec)
	}

	t.Logf("read %.2f MB a token against the fp16 bank's %.2f", float64(q4)/1e6, float64(half)/1e6)
	if ratio := float64(half) / float64(q4); ratio < 3.4 || ratio > 3.6 {
		t.Fatalf("a token reads %.3fx less than on halves, want ~3.5", ratio)
	}
}

// rowMoved is P5b's real control for an R-row kernel, and it is the one a
// comparison against the GEMM does not provide.
//
// Comparing the GEMV's second output row against the GEMM's passes trivially
// when the GEMV never writes that row: the arena still holds what the GEMM
// put there on the reference pass. So the *input* row is changed and the
// output row has to move. It caught a real one — the `.spv` in this repo are
// generated and gitignored, so a kernel edit not followed by `go generate`
// leaves the previous module embedded, and every row-against-row comparison
// then passes on stale bytes while the measurements read as a free speed-up.
func rowMoved(t *testing.T, what string, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: %d values against %d", what, len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			return
		}
	}
	t.Errorf("%s: the output row did not move when its token did — the rung is not writing it", what)
}
