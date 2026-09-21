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

// The bounds this port is gated at, and they are measured rather than chosen:
// TestFP16Ladder runs the CPU oracle with every linear taking fp16 operands
// and accumulating in fp32 — what a matrix core does — and reports what the
// device should therefore cost. This path does the same arithmetic on the
// device, and lands within a third of the prediction on both cards.
//
// **There are two of them, and the second is the finding.** The ladder was
// first run on the square card alone: 0.135 on the last hidden state, 0.14 on
// the merged rows, and 0.25 was the bound. Run on the *wide* card it gives
// **0.297 and 0.438** — three times as much from the same weights and the
// same code, on a picture 288 patches long instead of 256. That is not a
// second reading of one number, it is the fp16 face of what Q8.2 already
// measured in fp32: the dumped fp32 tensors are themselves rel 8.9e-4 from
// float64 on this card against 1.0e-4 on the square one, and a float64 run
// found the *reference* to be the less accurate side. Some pictures amplify
// rounding an order of magnitude harder than others, and a tower gated on one
// of them would either be too loose for the other or reject it.
//
// So each card is gated at its own measurement, and the device agrees with
// the prediction on both: square, ladder 0.14 against device 0.127; wide,
// ladder 0.438 against device 0.562.
//
// The number to read beside them: the checkpoint's own bf16 pipeline deviates
// from fp32 by rel 2.0 on the text embedding this tower ultimately feeds, and
// a *structural* mistake in this tower moves these tensors by 0.76 to 8.1
// (qimage/vision's four controls).
const (
	fp16Tol     = 0.25 // the square card: ladder 0.14, device 0.127
	fp16WideTol = 0.7  // the wide card: ladder 0.438, device 0.562
)

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
	// Staged for twice the card's patches, and run at the card's grid: a
	// served edit sizes this tower once and then hands it whatever grid
	// `calculate_dimensions` produced for the request, so the budget and the
	// run have to be allowed to differ *in the gate* and not only in the
	// code that serves it.
	budget := 2 * gridH * gridW
	g, err := NewGPU(dev, model, budget)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("%d-patch budget, %dx%d grid: %d MB weights, %d MB activations",
		budget, gridH, gridW, g.WeightBytes()>>20, g.ActivationBytes()>>20)

	got, err := g.Forward(t.Context(), pixels, gridH, gridW)
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

	// A *second, differently shaped* grid on the same staged tower — 12x24
	// against 16x16, so the rows, the merged rows, the transposed keys'
	// stride and the deepstack packing all change while nothing is restaged.
	// This is the whole of what the budget buys, and the single-grid port
	// could not have run it at all.
	if len(m.MultiGridTHW) == 2 && len(m.MultiSizes) == 2 {
		card2 := loadRef(t, m, "card2_vision_rgb")
		sz := m.MultiSizes[1]
		px2, gh2, gw2, err := cfg.Patchify(card2.Data, sz[0], sz[1])
		if err != nil {
			t.Fatal(err)
		}
		want2, err := model.Forward(px2, gh2, gw2, nil)
		if err != nil {
			t.Fatal(err)
		}
		got2, err := g.Forward(t.Context(), px2, gh2, gw2)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("second grid %dx%d (%d patches) on the same tower", gh2, gw2, gh2*gw2)
		compareAt(t, "card2_last_hidden", got2.Last, want2.Last, fp16WideTol)
		compareAt(t, "card2_merged", got2.Merged, want2.Merged, fp16WideTol)
		compareAt(t, "card2_merged_vs_dump", got2.Merged, loadRef(t, m, "vis2_merged"), fp16WideTol)
		// And the first grid again afterwards, which is what catches a run
		// leaving state behind: the same numbers as above or nothing.
		again, err := g.Forward(t.Context(), pixels, gridH, gridW)
		if err != nil {
			t.Fatal(err)
		}
		compareAt(t, "first grid, re-run", again.Merged, got.Merged, 0)
	}

	// The budget is a refusal, not a silent resize.
	if _, err := g.Forward(t.Context(), qwen.NewMat(budget+4, cfg.PatchElems()), 2, (budget+4)/2); err == nil {
		t.Error("a grid past the staged budget was accepted")
	}
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
	g, err := NewGPU(dev, model, side*side)
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
		if _, err := g.Forward(t.Context(), pixels, side, side); err != nil {
			t.Fatal(err)
		}
		t.Logf("1024x1024 condition: %v", time.Since(start).Round(time.Millisecond))
	}
}
