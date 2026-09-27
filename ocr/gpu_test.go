package ocr

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("ocr-test")
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
	// Subgroup size control as the server and cmd/ocr open it: with it the
	// tower's attention takes its wave32 build, without it wave64.
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// refTensors reads every tensor of a case's dump whose name is in names and
// is present (the page dumps only blocks 0, 1, 13 and 26).
func refTensors(t *testing.T, name string, names []string) map[string][]float32 {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "tensors.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string][]float32{}
	for _, n := range names {
		ten, err := f.Get(n)
		if err != nil {
			continue
		}
		if out[n], err = ten.F32(nil); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// The bounds, measured (OCR.md O3): the worst of the six cases is the page
// at proj 2.8e-3, post_ln 1.8e-3 and embed 7.6e-5, and the line's last block
// at 7.1e-3 (a residual stream before its norm). The raster-order control
// moves proj by ~1 (TestGPUTowerRasterControl).
var gpuTol = map[string]float64{"vis.embed": 3e-4, "vis.post_ln": 6e-3, "proj": 1e-2}

const gpuBlockTol = 2e-2

// TestGPUTower runs all six cases through the device tower and projector and
// holds every dumped stage to the fp32 reference.
func TestGPUTower(t *testing.T) {
	recs := records(t)
	tw, err := LoadTower(modelDir, -1)
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	dev, done := newTestDevice(t)
	defer done()
	gt, err := NewGPUTower(dev, tw, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer gt.Destroy()
	t.Logf("device: %.2f GB weights, %.2f GB activations", float64(gt.WeightBytes())/1e9, float64(gt.ActivationBytes())/1e9)

	names := []string{"vis.embed", "vis.post_ln", "proj"}
	for i := 0; i < VisionDepth; i++ {
		names = append(names, fmt.Sprintf("vis.block%d", i))
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].Name < recs[j].Name })
	for _, rec := range recs {
		t.Run(rec.Name, func(t *testing.T) {
			ref := refTensors(t, rec.Name, append([]string{"pixels"}, names...))
			im := &Image{Patches: ref["pixels"], GridH: rec.GridTHW[1], GridW: rec.GridTHW[2]}
			errs := map[string]float64{}
			check := func(name string, got *qwen.Mat) {
				want, ok := ref[name]
				if !ok {
					return
				}
				if len(want) != len(got.Data) {
					t.Fatalf("%s: %d values, reference has %d", name, len(got.Data), len(want))
				}
				errs[name] = relErr(got.Data, want)
			}
			gt.Tap = check
			if _, err := gt.Forward(context.Background(), im); err != nil {
				t.Fatal(err)
			}
			gt.Tap = nil
			start := time.Now()
			out, err := gt.Forward(context.Background(), im)
			if err != nil {
				t.Fatal(err)
			}
			wall := time.Since(start)
			check("proj", out)
			for _, n := range names {
				e, ok := errs[n]
				if !ok {
					continue
				}
				t.Logf("%-14s rel %.2e", n, e)
				tol, ok := gpuTol[n]
				if !ok {
					tol = gpuBlockTol
				}
				if e > tol {
					t.Errorf("%s: rel %.2e over %.0e", n, e, tol)
				}
			}
			if _, ok := errs["proj"]; !ok {
				t.Error("no proj in the reference")
			}
			t.Logf("%dx%d patches: %v untapped", im.GridW, im.GridH, wall.Round(time.Millisecond))
		})
	}
}

// TestGPUTowerRasterControl feeds the patches in raster order, which is what
// the checkpoint's own processor emits and what the merger must not see: the
// gate above has to be able to tell.
func TestGPUTowerRasterControl(t *testing.T) {
	var rec record
	for _, r := range records(t) {
		if r.Name == "table" {
			rec = r
		}
	}
	if rec.Name == "" {
		t.Skip("no table case in the reference")
	}
	tw, err := LoadTower(modelDir, -1)
	if err != nil {
		t.Skipf("no checkpoint (%v)", err)
	}
	dev, done := newTestDevice(t)
	defer done()
	gt, err := NewGPUTower(dev, tw, rec.GridTHW[1]*rec.GridTHW[2])
	if err != nil {
		t.Fatal(err)
	}
	defer gt.Destroy()
	ref := refTensors(t, rec.Name, []string{"pixels", "proj"})
	gt.rasterControl = true
	out, err := gt.Forward(context.Background(), &Image{Patches: ref["pixels"], GridH: rec.GridTHW[1], GridW: rec.GridTHW[2]})
	if err != nil {
		t.Fatal(err)
	}
	e := relErr(out.Data, ref["proj"])
	t.Logf("raster order into the merger: proj rel %.2e", e)
	if e <= gpuTol["proj"]*10 {
		t.Errorf("the control moved proj by only %.2e", e)
	}
}
