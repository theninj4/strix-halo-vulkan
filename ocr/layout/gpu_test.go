package layout

import (
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("layout-test")
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
		t.Skipf("no subgroup size query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// The device's bounds on the regions: fp16 operands move scores by up to
// 7.3e-4 and boxes by 0.02 px over the five cases (the ladder predicted
// 1e-3 and 0.1 px); every region, label and order is HF's.
const (
	gpuScoreTol = 3e-3
	gpuBoxTol   = 0.5
)

// TestGPU runs every case through the device trunk and the host head, and
// holds the trunk's stages to HF's fp32 and the regions to HF's.
func TestGPU(t *testing.T) {
	m := load(t)
	dev, done := newTestDevice(t)
	defer done()
	start := time.Now()
	g, err := NewGPU(dev, m)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged in %v, %.2f GB on the device", time.Since(start).Round(time.Millisecond), float64(g.DeviceBytes())/1e9)
	for _, rec := range records(t) {
		t.Run(rec.Name, func(t *testing.T) {
			ref := refTensors(t, rec.Name)
			trunk := true
			worst := 0.0
			out, _, err := g.Forward(ref["pixels"], func(name string, got []float32) {
				if name == "enc.class" {
					trunk = false
				}
				if !trunk {
					return
				}
				e := relErr(got, ref[name])
				worst = max(worst, e)
				t.Logf("%-14s rel %.2e", name, e)
			})
			if err != nil {
				t.Fatal(err)
			}
			_, tm, err := g.Forward(ref["pixels"], nil)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("worst trunk stage %.2e; upload %v, segment A %v, AIFI %v, segment B %v, readback %v, head %v",
				worst, tm.Upload.Round(time.Millisecond), tm.SegA.Round(time.Millisecond), tm.AIFI.Round(time.Millisecond),
				tm.SegB.Round(time.Millisecond), tm.Readback.Round(time.Millisecond), tm.Head.Round(time.Millisecond))
			checkDetections(t, rec, out.Detect(rec.Size[1], rec.Size[0], rec.Threshold), gpuScoreTol, gpuBoxTol)
		})
	}
}
