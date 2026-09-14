package qwen

import (
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qwen-test")
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

// fp16RelTol is what the matrix-core path is held to, against the same fp32
// reference the CPU implementation matches at 2e-5. It is measured, not
// chosen: the projections narrow both operands to fp16 and accumulate in
// fp32, so a stage's error is set by the weight's 11-bit mantissa and grows
// down the 35-layer chain. TestGPUEncoder logs the whole ladder.
const fp16RelTol = 6e-3

func openSet(t *testing.T) (*safetensors.Set, *Config) {
	t.Helper()
	if _, err := os.Stat(encoder); err != nil {
		t.Skipf("no text encoder checkpoint at %s", encoder)
	}
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	return set, cfg
}

func newEncoderFor(t *testing.T, dev *vk.Device, layers int, m *manifest, plan GEMMPlan, ctl controls, bankBytes int) *GPUEncoder {
	t.Helper()
	set, cfg := openSet(t)
	defer set.Close()
	g, err := newEncoder(dev, set, cfg, layers, m.Seq, plan, ctl, bankBytes)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestGPULayer walks one layer's graph dispatch by dispatch against the
// reference dump. It is the test that says which dispatch is wrong when the
// encoder's output is; TestGPUEncoder only says that something is.
func TestGPULayer(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	g := newEncoderFor(t, dev, 1, m, nil, controls{}, maxBankBytes)
	defer g.Destroy()

	c := g.cfg
	qWidth := c.NumHeads * c.HeadDim
	kvWidth := c.NumKVHeads * c.HeadDim
	t.Logf("plan %v, attention %s, %d dispatches per layer", g.plan[ProjQ], g.Attention(), len(g.Labels()))

	// The A operand is fp16 and lives in the other arena, so the two norm
	// stages are read back through ReadF16 at their own leading dimension.
	type stage struct {
		label string
		ref   string
		read  func() *Mat
	}
	stages := []stage{
		{"attn in", "input_layernorm", func() *Mat { return g.ReadF16(g.hA, c.HiddenSize, g.ldaDim) }},
		{"gemm q", "q", func() *Mat { return g.Read(g.aQ, qWidth) }},
		{"gemm k", "k", func() *Mat { return g.Read(g.aK, kvWidth) }},
		{"gemm v", "v", func() *Mat { return g.Read(g.aV, kvWidth) }},
		{"rmsnorm q", "q_normed", func() *Mat { return g.Read(g.aQ, qWidth) }},
		{"rmsnorm k", "k_normed", func() *Mat { return g.Read(g.aK, kvWidth) }},
		{"rope q", "q_roped", func() *Mat { return g.Read(g.aQ, qWidth) }},
		{"rope k", "k_roped", func() *Mat { return g.Read(g.aK, kvWidth) }},
		{"attention", "attn_ctx", func() *Mat { return g.Read(g.aCtx, qWidth) }},
		{"gemm o", "attn_out", func() *Mat { return g.Read(g.aAttn, c.HiddenSize) }},
		{"resid attn", "resid1", func() *Mat { return g.Read(g.aX, c.HiddenSize) }},
		{"ffn in", "post_attention_layernorm", func() *Mat { return g.ReadF16(g.hA, c.HiddenSize, g.ldaDim) }},
		{"gemm down", "mlp", func() *Mat { return g.Read(g.aFF, c.HiddenSize) }},
		{"resid ffn", "layer_out", func() *Mat { return g.Read(g.aX, c.HiddenSize) }},
	}
	for _, s := range stages {
		if err := g.RunTo(m.IDs, 0, s.label); err != nil {
			t.Fatalf("%s: %v", s.label, err)
		}
		compareTol(t, s.label, s.read(), loadRef(t, m, s.ref), fp16RelTol)
	}
}

// TestGPUEncoder runs what the pipeline runs -- 35 of the 36 layers -- and
// checks the hidden state it hands the DiT, for two prompts. This is stage
// 5c's acceptance check.
func TestGPUEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 7.06 GB of fp16 weights")
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	set, cfg := openSet(t)
	g, err := NewGPUEncoder(dev, set, cfg, cfg.EncoderLayers(), 512, nil)
	set.Close()
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	bytes, byLayer := g.Banks()
	t.Logf("%d layers, %d banks %v MB, activations %d MB",
		g.Layers(), len(bytes), mb(bytes), g.ActivationBytes()>>20)
	t.Logf("last layer is in bank %d", byLayer[len(byLayer)-1])

	got, err := g.Forward(m.IDs)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "final", got, loadRef(t, m, "final"), fp16RelTol)
	compareTol(t, "pipeline_out", got, loadRef(t, m, "pipeline_out"), fp16RelTol)

	got2, err := g.Forward(m.IDs2)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "final2", got2, loadRef(t, m, "final2"), fp16RelTol)
}

func mb(b []int) []int {
	out := make([]int, len(b))
	for i, v := range b {
		out[i] = v >> 20
	}
	return out
}

// TestGPUKernelsAgree runs every rung of the GEMM ladder over one layer and
// requires them to agree with the reference. A rung that is fast and wrong is
// the failure mode a ladder invites, and the rungs differ in how much of the
// padded row block they cover -- 24 tokens become 32 rows under a 16-row tile
// and 128 under the DiT's -- which is exactly where a bounds bug would sit.
//
// It is also what exercises SetPlan: one staging of the weights, seven plans
// over it.
func TestGPUKernelsAgree(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	want := loadRef(t, m, "layer_out")
	g := newEncoderFor(t, dev, 1, m, nil, controls{}, maxBankBytes)
	defer g.Destroy()
	for _, k := range GEMMKernels() {
		if err := g.SetPlan(UniformGEMMPlan(k)); err != nil {
			t.Fatal(err)
		}
		got, err := g.Forward(m.IDs)
		if err != nil {
			t.Fatalf("%s: %v", k, err)
		}
		compareTol(t, string(k), got, want, fp16RelTol)
	}
}

// TestGPUBanksAreIdentical is stage 4c's check on this model: the weight
// arena split across storage buffers has to produce the same numbers as one
// buffer, exactly rather than within a tolerance, because nothing about the
// arithmetic changes -- only which descriptor set a dispatch names.
func TestGPUBanksAreIdentical(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	one := newEncoderFor(t, dev, 4, m, nil, controls{}, maxBankBytes)
	defer one.Destroy()
	wantBanks, _ := one.Banks()
	want, err := one.Forward(m.IDs)
	if err != nil {
		t.Fatal(err)
	}
	if len(wantBanks) != 1 {
		t.Fatalf("four layers should be one bank, got %d", len(wantBanks))
	}

	// A bank that holds one layer, so four layers are four banks.
	split := newEncoderFor(t, dev, 4, m, nil, controls{}, 300<<20)
	defer split.Destroy()
	gotBanks, byLayer := split.Banks()
	if len(gotBanks) != 4 {
		t.Fatalf("a 300 MB bank should hold one 202 MB layer, got %d banks for 4 layers", len(gotBanks))
	}
	got, err := split.Forward(m.IDs)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("bank split changed element %d: %g against %g", i, got.Data[i], want.Data[i])
		}
	}
	t.Logf("4 banks %v MB, one layer each %v: bit-identical to one bank", mb(gotBanks), byLayer)
}

// TestGPUValidationDetectsErrors is the negative control. Each break is a
// mistake this port actually invites, and three of the five are only
// reachable from inside the graph -- which shader a dispatch names, what span
// a norm is given, which bank a layer reads.
func TestGPUValidationDetectsErrors(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()
	want := loadRef(t, m, "layer_out")
	wantCtx := loadRef(t, m, "attn_ctx")

	for _, c := range []struct {
		name  string
		ctl   controls
		layer int
		stage string
		ref   *Mat
	}{
		{"rope paired adjacent not by halves", controls{ropeAdjacent: true}, 1, "", want},
		{"attention not causal", controls{noCausal: true}, 1, "attention", wantCtx},
		{"qk norm over full width not per head", controls{qkNormWide: true}, 1, "", want},
		{"weight staged row-major for a tiled kernel", controls{wrongBLayout: true}, 1, "", want},
		// Four layers in four banks, each layer sent to the next one: layer i
		// then reads layer i+1's weights, at the same offsets, and produces a
		// perfectly plausible tensor.
		{"layer reads the next bank", controls{shiftBank: true}, 4, "", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			bank := maxBankBytes
			if c.ctl.shiftBank {
				bank = 300 << 20
			}
			ref := c.ref
			if ref == nil {
				good := newEncoderFor(t, dev, c.layer, m, nil, controls{}, bank)
				out, err := good.Forward(m.IDs)
				good.Destroy()
				if err != nil {
					t.Fatal(err)
				}
				ref = out
			}
			g := newEncoderFor(t, dev, c.layer, m, nil, c.ctl, bank)
			defer g.Destroy()
			var got *Mat
			if c.stage != "" {
				if err := g.RunTo(m.IDs, 0, c.stage); err != nil {
					t.Fatal(err)
				}
				got = g.Read(g.aCtx, g.cfg.NumHeads*g.cfg.HeadDim)
			} else {
				out, err := g.Forward(m.IDs)
				if err != nil {
					t.Fatal(err)
				}
				got = out
			}
			assertCaughtTol(t, got, ref, fp16RelTol)
		})
	}
}

// assertCaughtTol fails if a deliberately broken tensor slips under a bound.
func assertCaughtTol(t *testing.T, got, want *Mat, tol float64) {
	t.Helper()
	maxAbs, rms, rel, _ := deviation(got, want)
	if rel <= tol {
		t.Errorf("perturbation NOT caught: max abs %.6g, rms %.6g, rel %.3g <= %.0e", maxAbs, rms, rel, tol)
		return
	}
	t.Logf("caught: rel %.3g (%.0fx the %.0e bound)", rel, rel/tol, tol)
}

// compareTol is compare against an explicit bound, for the matrix-core path,
// which narrows its operands to fp16 and does not carry fp32's drift.
func compareTol(t *testing.T, name string, got, want *Mat, tol float64) float64 {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > tol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, tol)
		return rel
	}
	t.Logf("%-26s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
	return rel
}
