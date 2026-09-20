package textenc

import (
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-textenc-test")
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

// The fp16 bounds are measured, and the story behind them matters (all
// numbers 2026-09-20, TestGPULadder and the torch probes beside it):
//
// This checkpoint's residual stream grows a ~13k massive-activation channel
// at layer 6 (and again at 16), carries it flat for 18 layers, and *cancels*
// it in layers 34-35 — the two layers Z-Image's hidden_states[-2] never ran,
// which is why zimage/qwen's fp16 path could hold 6e-3 and this one cannot.
// Every reduced-precision pass through the FFN injects ~5e-4 of bulk
// rounding variance into that channel (keeping just the spike elements exact
// measurably does NOT help), and when the channel collapses the carried
// error surfaces: rel 0.024 on a real prompt, 0.21 elementwise on the
// all-template empty prompt whose rows are mostly cancellation residue.
//
// Priced against the model's own deployment: the *official* bf16 pipeline
// deviates from fp32 by rel 2.0 (max abs 204) through the same mechanism —
// two orders of magnitude beyond this path. If Q3/Q6 ever show prompt
// adherence suffering, the known lever is an fp32-activation GEMM arm for
// the encoder, and at prompt-sized M (~40 rows) the encoder is
// weight-bandwidth-bound, so that arm would cost little; it is not built
// because nothing yet shows it is needed.
const (
	gpuTol      = 3e-2 // en/cjk: measured 0.024-0.027
	gpuEmptyTol = 0.25 // empty prompt: measured 0.214, residue-dominated
)

// TestGPUEncoder is Q1's GPU acceptance gate: zimage/qwen's GPUEncoder,
// unchanged, over the Qwen3-VL config — all 36 layers, no final norm — for
// the three dumped prompts. ~15 GB of fp16 banks, staged one layer at a
// time.
func TestGPUEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 15 GB of fp16 banks")
	}
	m := loadManifest(t)
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
	defer set.Close()

	dev, done := newTestDevice(t)
	defer done()
	g, err := qwen.NewGPUEncoder(dev, set, cfg, cfg.NumLayers, 512, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	for _, label := range []string{"en", "cjk", "empty"} {
		p := m.Prompts[label]
		out, err := g.Forward(p.IDs)
		if err != nil {
			t.Fatal(err)
		}
		tol := gpuTol
		if label == "empty" {
			tol = gpuEmptyTol
		}
		if label == "en" {
			compareAt(t, "en_last_prenorm", out, loadRef(t, m, "en_last_prenorm"), tol)
		}
		embeds, err := Drop(out, m.DropIdx)
		if err != nil {
			t.Fatal(err)
		}
		compareAt(t, label+"_prompt_embeds", embeds, loadRef(t, m, label+"_prompt_embeds"), tol)
	}
}
