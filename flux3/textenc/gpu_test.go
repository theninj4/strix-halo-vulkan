package textenc

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	if testing.Short() {
		t.Skip("a GPU gate: run without -short, and one package at a time (-p 1) -- packages in parallel share the ring, and the fp32 decoder's batches then pass amdgpu's 2 s watchdog")
	}
	inst, err := vk.NewInstance("flux3-textenc-test")
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
		t.Skipf("no subgroup size control: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// tapGap is the rms error over the reference's rms, per tap (each a
// 2560-wide column block), over the real rows or the pad rows.
func tapGap(got, want *qwen.Mat, from, to, tap, h int) float64 {
	var sq, ref float64
	for r := from; r < to; r++ {
		g, w := got.Row(r)[tap*h:(tap+1)*h], want.Row(r)[tap*h:(tap+1)*h]
		for i := range w {
			d := float64(g[i]) - float64(w[i])
			sq += d * d
			ref += float64(w[i]) * float64(w[i])
		}
	}
	return math.Sqrt(sq / ref)
}

// gpuTapTol bounds each tap's rms error against the fp32 dump, real rows and
// pad rows separately. X5 measured 3.7e-4 – 7.2e-4 (real) and 3.1e-4 –
// 1.1e-3 (pad, growing with depth) over the three prompts; 5e-3 keeps 4x.
const gpuTapTol = 5e-3

// TestGPUEncode is the device encoder against the dumped fp32 embeddings of
// the three prompts, tap by tap, real rows and pad rows apart, and the
// PAD_LAST control: the plain causal build lets the pad row see itself.
func TestGPUEncode(t *testing.T) {
	m := loadManifest(t)
	if _, err := os.Stat(filepath.Join(modDir, "config.json")); err != nil {
		t.Skipf("no text encoder at %s", modDir)
	}
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPU(dev, modDir, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	h := g.cfg.HiddenSize
	for _, name := range []string{"short", "long", "cjk"} {
		c := m.Prompts[name]
		ids, err := g.IDs(c.Prompt)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != c.Real {
			t.Fatalf("%s: %d ids, the reference has %d", name, len(ids), c.Real)
		}
		start := time.Now()
		got, err := g.Encode(ids)
		if err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		want := loadRef(t, m, name+"_prompt_embeds")
		if got.Rows != want.Rows {
			t.Fatalf("%s: %d rows, want %d", name, got.Rows, want.Rows)
		}
		line := fmt.Sprintf("%-5s %v  real/pad rms rel by tap:", name, took.Round(time.Millisecond))
		for j := range Taps {
			re, pa := tapGap(got, want, 0, c.Real, j, h), tapGap(got, want, c.Real, got.Rows, j, h)
			line += fmt.Sprintf(" %.1e/%.1e", re, pa)
			if !(re <= gpuTapTol && pa <= gpuTapTol) {
				t.Errorf("%s tap %d: real %.3g, pad %.3g > %.0e", name, Taps[j], re, pa, gpuTapTol)
			}
		}
		t.Log(line)
	}
}

// TestGPUPadControl runs the short prompt with the plain causal attention,
// where the pad row also attends to its own key: the pad rows must move
// past the bound the real build meets, and the real rows must not move.
func TestGPUPadControl(t *testing.T) {
	m := loadManifest(t)
	if _, err := os.Stat(filepath.Join(modDir, "config.json")); err != nil {
		t.Skipf("no text encoder at %s", modDir)
	}
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPU(dev, modDir, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	// The same encoder with the causal build put back.
	set, err := safetensors.OpenSet(modDir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	plain, err := qwen.NewGPUEncoder(dev, set, g.cfg, Taps[len(Taps)-1], 129, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Destroy()
	c := m.Prompts["short"]
	want := loadRef(t, m, "short_prompt_embeds")
	ids, err := g.IDs(c.Prompt)
	if err != nil {
		t.Fatal(err)
	}
	padLast := g.Enc
	g.Enc = plain
	got, err := g.Encode(ids)
	g.Enc = padLast // each encoder destroyed once, by its own defer
	if err != nil {
		t.Fatal(err)
	}
	h, last := g.cfg.HiddenSize, len(Taps)-1
	re, pa := tapGap(got, want, 0, c.Real, last, h), tapGap(got, want, c.Real, got.Rows, last, h)
	if pa <= gpuTapTol || re > gpuTapTol {
		t.Errorf("control: pad rows at %.3g, real rows at %.3g; want pad > %.0e >= real", pa, re, gpuTapTol)
		return
	}
	t.Logf("control: pad row sees itself: tap 32 pad rms rel %.3g (%.0fx the bound), real %.2g", pa, pa/gpuTapTol, re)
}
