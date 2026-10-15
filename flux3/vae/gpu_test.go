package vae

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	if testing.Short() {
		t.Skip("a GPU gate: run without -short, and one package at a time (-p 1) -- packages in parallel share the ring, and the fp32 decoder's batches then pass amdgpu's 2 s watchdog")
	}
	inst, err := vk.NewInstance("flux3-vae-test")
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

// pictureGap is a decode's distance from the fp32 dump (max abs, rms rel)
// and its 8-bit picture's from the run's PNG (levels that differ, the worst,
// PSNR).
func pictureGap(t *testing.T, img, want *Tensor) (maxAbs, rmsRel float64, diff, worst int, psnr float64) {
	t.Helper()
	var sq, ref float64
	for i, v := range img.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("decode produced %v at %d", v, i)
		}
		d := float64(v) - float64(want.Data[i])
		maxAbs = math.Max(maxAbs, math.Abs(d))
		sq += d * d
		ref += float64(want.Data[i]) * float64(want.Data[i])
	}
	raw, err := os.ReadFile(filepath.Join(runDir, "image.bin"))
	if err != nil {
		t.Fatal(err)
	}
	var lsq float64
	for i, g := range RGB8(img) {
		w := int(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		if e := abs(int(g) - w); e > 0 {
			diff++
			worst = max(worst, e)
			lsq += float64(e * e)
		}
	}
	n := float64(img.H * img.W * 3)
	psnr = math.Inf(1)
	if lsq > 0 {
		psnr = 10 * math.Log10(255*255/(lsq/n))
	}
	return maxAbs, math.Sqrt(sq / ref), diff, worst, psnr
}

// gpuDecodeMinPSNR bounds a device decode's 8-bit picture against the fp32
// run's PNG.
const gpuDecodeMinPSNR = 45

// TestGPUDecode decodes the t2i run's 1024² latent on the device, on the
// fp16 matrix-core path (the default) and on the fp32 scalar one, against
// the dump's fp32 decode and the run's PNG, and times each.
func TestGPUDecode(t *testing.T) {
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	z, want := d.tensor(t, "dec_in"), d.tensor(t, "dec_out")
	for _, arm := range []struct {
		name string
		opt  Options
	}{
		{"fp16 cores", Options{}},
		{"fp32 scalar", Options{Attn: AttnScalar, Conv: ConvScalar}},
	} {
		g, err := NewGPUDecoderOpts(dev, dec, z.H, z.W, arm.opt)
		if err != nil {
			t.Fatal(err)
		}
		var img *Tensor
		var best time.Duration
		for rep := 0; rep < 2; rep++ {
			start := time.Now()
			if img, err = g.Apply(z); err != nil {
				g.Destroy()
				t.Fatal(err)
			}
			if el := time.Since(start); rep == 0 || el < best {
				best = el
			}
		}
		g.Destroy()
		maxAbs, rms, diff, worst, psnr := pictureGap(t, img, want)
		msg := "%-11s %v: max abs %.3g, rms rel %.3g; 8-bit %d levels differ (worst %d), %.1f dB"
		if psnr < gpuDecodeMinPSNR {
			t.Errorf(msg+" < %d dB", arm.name, best.Round(time.Millisecond), maxAbs, rms, diff, worst, psnr, gpuDecodeMinPSNR)
			continue
		}
		t.Logf(msg, arm.name, best.Round(time.Millisecond), maxAbs, rms, diff, worst, psnr)
	}
}
