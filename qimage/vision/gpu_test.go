package vision

import (
	"fmt"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-vision-test")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// fp16Tol is the bound this port is gated at, and it is measured rather than
// chosen: TestFP16Ladder runs the CPU oracle with every linear taking fp16
// operands and accumulating in fp32 — what a matrix core does — and lands at
// rel 0.135 on the last hidden state and 0.14 on the merged rows. This path
// does the same arithmetic on the device, so that is what it should cost;
// 0.25 leaves room for the summation order differing too, and is still an
// order of magnitude inside the *structural* mistakes qimage/vision's own
// controls produce (rel 0.76 to 8.1).
//
// The number to read beside it: the checkpoint's own bf16 pipeline deviates
// from fp32 by rel 2.0 on the text embedding this tower ultimately feeds.
const fp16Tol = 0.25

// TestGPUTower is the device port's gate: the same card through the CPU
// oracle and the device graph, stage by stage where the dump has a stage and
// end to end where it does not.
func TestGPUTower(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole tower twice")
	}
	cfg, model, m := load(t, 0)
	card := loadRef(t, m, "card_vision_rgb")
	pixels, gridH, gridW, err := cfg.Patchify(card.Data, m.Size, m.Size)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Forward(pixels, gridH, gridW, nil)
	if err != nil {
		t.Fatal(err)
	}

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGPU(dev, model, gridH, gridW)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("%dx%d grid: %d MB weights, %d MB activations",
		gridH, gridW, g.WeightBytes()>>20, g.ActivationBytes()>>20)

	got, err := g.Forward(pixels)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name      string
		got, want *qwen.Mat
	}{
		{"last_hidden", got.Last, want.Last},
		{"merged", got.Merged, want.Merged},
	} {
		compareAt(t, c.name, c.got, c.want, fp16Tol)
	}
	for i := range want.Deepstack {
		compareAt(t, fmt.Sprintf("deepstack%d", i), got.Deepstack[i], want.Deepstack[i], fp16Tol)
	}
	// Against the dump as well, so the device path is measured against the
	// reference and not only against the port it was debugged on.
	compareAt(t, "merged_vs_dump", got.Merged, loadRef(t, m, "vis_merged"), fp16Tol)
}

// compareAt is compare with an explicit bound.
func compareAt(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %dx%d, want %dx%d", name, got.Rows, got.Cols, want.Rows, want.Cols)
	}
	rel := relGap(got, want)
	if rel > tol {
		t.Errorf("%-16s %dx%d: rel %.3g > %.2g", name, got.Rows, got.Cols, rel, tol)
		return
	}
	t.Logf("%-16s %dx%d  rel %.3g  (bound %.2g)", name, got.Rows, got.Cols, rel, tol)
}

// TestGPUTowerTiming is what an edit's prefix costs per reference image at a
// served condition size, and the number that says whether the scalar
// attention kernel is worth replacing.
func TestGPUTowerTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("stages the whole tower")
	}
	cfg, model, _ := load(t, 0)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	const side = 64 // 64x64 patches = a 1024x1024 condition image
	g, err := NewGPU(dev, model, side, side)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	pixels := qwen.NewMat(side*side, cfg.PatchElems())
	for i := range pixels.Data {
		pixels.Data[i] = float32(i%255)/127.5 - 1
	}
	t.Logf("%dx%d patches: %d MB weights, %d MB activations",
		side, side, g.WeightBytes()>>20, g.ActivationBytes()>>20)
	for run := 0; run < 2; run++ {
		start := time.Now()
		if _, err := g.Forward(pixels); err != nil {
			t.Fatal(err)
		}
		t.Logf("1024x1024 condition: %v", time.Since(start).Round(time.Millisecond))
	}
}
