package textenc

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference is reference/dump_h3_textenc.py: hidden_states[50] of the
// Qwen3-VL-32B conditioner, from the pipeline's own bf16 path ("_bf16") and
// from the same weights with fp32 arithmetic ("_fp32"). Regenerate with:
//
//	.venv/bin/python reference/dump_h3_textenc.py   (~70 GB RSS)
const (
	encRef  = "../../reference/out/h3textenc"
	encoder = modelDir + "/text_encoder"
)

const strixHaloDeviceID = 0x1586

type encManifest struct {
	Layer   int `json:"layer"`
	Prompts map[string]struct {
		IDs []int32 `json:"ids"`
	} `json:"prompts"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadEncRef(t *testing.T) *encManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(encRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_textenc.py", encRef, err)
	}
	var m encManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func encMat(t *testing.T, m *encManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(encRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := &qwen.Mat{Rows: meta.Shape[0], Cols: meta.Shape[1], Data: make([]float32, meta.Count)}
	for i := range out.Data {
		out.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// gap is max abs error over the reference's absmax, the rel figure Q1's
// bounds are stated in.
func gap(got, want *qwen.Mat) (rel, maxAbs, rms float64) {
	var ref, sq float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want.Data[i])))
		sq += d * d
	}
	return maxAbs / ref, maxAbs, math.Sqrt(sq / float64(len(want.Data)))
}

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("h3-textenc-test")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// gpuTol bounds the fp16 path against the fp32 oracle, relative to absmax.
// Measured 2026-09-26: 6.5e-5 (en), 2.4e-6 (cjk), 1.1e-4 (the 537-token
// README prompt), where the official bf16 pipeline is at 1.2–1.4e-2 against
// the same oracle. Unlike Qwen-Image's 8 B encoder (Q1: 0.024), nothing here
// cancels the massive-activation channel before layer 50 reads it.
const gpuTol = 5e-4

// TestGPUEncoder is M2's gate: zimage/qwen's GPU encoder, unchanged, over
// the 32 B config with Layers of its 64 layers, for every dumped prompt,
// against the fp32 oracle. It also reports where the official bf16 pipeline
// sits against the same oracle, which is the scale the bound is priced on.
// ~50 GB of fp16 banks.
func TestGPUEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 50 GB of fp16 banks")
	}
	m := loadEncRef(t)
	if m.Layer != Layers {
		t.Fatalf("reference reads hidden_states[%d], this package %d", m.Layer, Layers)
	}
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder (%v)", err)
	}
	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	dev, done := newTestDevice(t)
	defer done()

	start := time.Now()
	g, err := qwen.NewGPUEncoder(dev, set, cfg, Layers, 1024, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged %d layers in %v", Layers, time.Since(start).Round(time.Millisecond))
	for _, label := range []string{"en", "cjk", "readme"} {
		start := time.Now()
		out, err := g.Forward(m.Prompts[label].IDs)
		if err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		want := encMat(t, m, label+"_fp32")
		rel, maxAbs, rms := gap(out, want)
		brel, babs, _ := gap(encMat(t, m, label+"_bf16"), want)
		t.Logf("%-7s %4d tokens in %v: fp16 rel %.3g (max abs %.4g, rms %.3g); official bf16 rel %.3g (max abs %.4g)",
			label, out.Rows, took.Round(time.Millisecond), rel, maxAbs, rms, brel, babs)
		if rel > gpuTol || math.IsNaN(rel) {
			t.Errorf("%s: rel %.3g > %.0e", label, rel, gpuTol)
		}
	}
}
