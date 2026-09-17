// Package backend holds the model adapters the HTTP API is served from: one
// per vertical, each owning its weights, its device residency and the lock
// that serialises requests against them.
//
// It exists so that `api` can stay a routing layer with no model in it and no
// Vulkan behind it, and so that the adapters are testable and reusable
// outside `cmd/serve`. The direction of the dependency is
// `cmd/serve -> backend -> api`, never the other way: `api` declares the
// interfaces (api.SpeechBackend, api.TranscriptionBackend) and the types here
// satisfy them.
package backend

import (
	"fmt"
	"sync"

	"strix-halo-vulkan/vk"
)

// strixHaloDeviceID is this machine's iGPU (gfx1151 / RADV STRIX_HALO), which
// every command in this repository prefers when it is present.
const strixHaloDeviceID = 0x1586

// Device is the Vulkan device plus the lock that makes it safe to serve from.
//
// **The lock is not a performance choice, it is a correctness one.** A
// vk.Device holds a single queue, and nothing in `vk` is externally
// synchronised: two requests recording and submitting at the same time is
// undefined behaviour and not merely contention. Every run -- and every
// staging pass, which also records and submits -- goes through Do, so the
// server is concurrent at the HTTP layer and serial at the device.
//
// That is also the right shape for the hardware. The models here are sized to
// saturate the iGPU on their own; two of them interleaved would not go
// faster, they would just share a 236 GB/s bus.
type Device struct {
	inst *vk.Instance
	dev  *vk.Device
	mu   sync.Mutex
}

// OpenDevice picks this machine's iGPU, or the first device with a compute
// queue, and enables the features every model in this repository is built
// against: fp16, cooperative matrix, and subgroup size control where the
// driver reports it.
func OpenDevice(appName string) (*Device, error) {
	inst, err := vk.NewInstance(appName)
	if err != nil {
		return nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil {
		inst.Destroy()
		return nil, err
	}
	if len(devices) == 0 {
		inst.Destroy()
		return nil, fmt.Errorf("backend: no Vulkan devices")
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
		return nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		return nil, err
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		return nil, err
	}
	return &Device{inst: inst, dev: dev}, nil
}

// Do runs fn with the device held. It is the only way a caller outside this
// package touches the device.
func (d *Device) Do(fn func(*vk.Device) error) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return fn(d.dev)
}

// Name is the physical device's name, for the startup banner.
func (d *Device) Name() string { return d.dev.Physical().Name }

// Close destroys the device and the instance. Every model staged on it must
// already have been destroyed.
func (d *Device) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.dev.Destroy()
	d.inst.Destroy()
}
