package vae

import (
	"os"
	"testing"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

// newTestDevice opens the Strix Halo iGPU, or skips.
func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("vae-test")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// TestGPUDecoderAgainstDiffusers runs the Vulkan graph against the same
// reference the CPU implementation is held to. The CPU path already matched,
// so any failure here is the shaders and not the architecture -- which is
// the whole reason stage 2a came first.
func TestGPUDecoderAgainstDiffusers(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	gpu, err := NewGPUDecoder(dev, cpu, latent.H, latent.W)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Destroy()

	n, err := gpu.Dispatches(latent.H, latent.W)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("graph: %d dispatches, %.1f MB weights, %.1f MB activations",
		n, float64(gpu.wbuf.Size())/1e6, float64(gpu.abuf.Size())/1e6)

	got, err := gpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "gpu image", got, loadRef(t, m, "image"))
}

// TestGPUMatchesCPU compares the two implementations directly, which is the
// check that keeps meaning something once the GPU path moves to fp16 and the
// diffusers tolerance has to widen.
func TestGPUMatchesCPU(t *testing.T) {
	if _, err := os.Stat(vaeDir); err != nil {
		t.Skipf("no VAE checkpoint at %s", vaeDir)
	}
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	defer done()

	cpu, err := LoadDecoder(vaeDir, FluxConfig())
	if err != nil {
		t.Fatal(err)
	}
	latent := loadRef(t, m, "latent")

	gpu, err := NewGPUDecoder(dev, cpu, latent.H, latent.W)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Destroy()

	gotGPU, err := gpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	gotCPU, err := cpu.Apply(latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "gpu vs cpu", gotGPU, gotCPU)
}
