package llm

import (
	"fmt"
	"math"
	"testing"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

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
	g, err := NewHCGPU(dev, m.Config.HCConfig(), len(ids), []HCWeights{w}, HCOpts{Gate: true})
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
		{"hc_inject-0", g.Inject(), wantInject, tolInject, "hc_inject-0", tolInject},
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
	g, err := NewHCGPU(dev, cfg, nTok, ws, HCOpts{Gate: true})
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
			{"hc_inject", g.Inject(), cpuInject, tolInject, tolInject},
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
	g, err := NewHCGPU(dev, m.Config.HCConfig(), len(ids), []HCWeights{w}, HCOpts{UnpermutedUp: true})
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

// TestHCGPUGemvRefusesABatch is the control for the one thing the decode
// rungs cannot do. The GEMV writes a single output row and reads a single
// activation row: there is no M in it at all, so a two-token run through it
// would not be slow, it would silently drop a token. The plan has to refuse.
func TestHCGPUGemvRefusesABatch(t *testing.T) {
	g, _, nTok := gpuFixture(t, 0, "attn")
	if nTok < 2 {
		t.Skip("the fixture is one token")
	}
	if err := g.SetPlan(HCDownGemv160, HCUpM1); err == nil {
		t.Fatalf("SetPlan took the decode rung for a %d-token run", nTok)
	}
}
