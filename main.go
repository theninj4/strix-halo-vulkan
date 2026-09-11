// Command strix-halo-vulkan runs a minimal Vulkan compute shader (doubling
// an array of floats) on the local GPU, picking the Strix Halo iGPU
// (gfx1151 / RADV STRIX_HALO, deviceID 0x1586) if present.
package main

import (
	"fmt"
	"log"

	"strix-halo-vulkan/shaders"
)

const strixHaloDeviceID = 0x1586

const (
	elementCount = 256
	localSizeX   = 64 // must match shaders/double.comp's local_size_x
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	instance, err := NewInstance("strix-halo-vulkan")
	if err != nil {
		return err
	}
	defer instance.Destroy()

	devices, err := instance.PhysicalDevices()
	if err != nil {
		return err
	}

	phys := pickDevice(devices)
	fmt.Printf("using device: %s (vendor 0x%04x, device 0x%04x)\n", phys.Name, phys.VendorID, phys.DeviceID)

	queueFamily, err := phys.ComputeQueueFamily()
	if err != nil {
		return err
	}

	device, err := NewDevice(phys, queueFamily)
	if err != nil {
		return err
	}
	defer device.Destroy()

	buf, err := device.NewBuffer(elementCount * 4)
	if err != nil {
		return err
	}
	defer buf.Destroy()

	input := make([]float32, elementCount)
	for i := range input {
		input[i] = float32(i)
	}
	buf.WriteFloat32(input)

	shader, err := device.NewShaderModule(shaders.Double)
	if err != nil {
		return err
	}
	defer shader.Destroy()

	pipe, err := device.NewComputePipeline(shader, buf)
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	groups := uint32((elementCount + localSizeX - 1) / localSizeX)
	if err := pipe.Dispatch(groups); err != nil {
		return err
	}

	out := buf.ReadFloat32(elementCount)
	for i, v := range out {
		want := input[i] * 2
		if v != want {
			return fmt.Errorf("mismatch at %d: got %v want %v", i, v, want)
		}
	}
	fmt.Printf("verified: %d elements doubled correctly (e.g. %v -> %v)\n", elementCount, input[1], out[1])
	return nil
}

// pickDevice prefers the Strix Halo iGPU by deviceID, falling back to the
// first device the instance reports.
func pickDevice(devices []PhysicalDevice) *PhysicalDevice {
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			return &devices[i]
		}
	}
	return &devices[0]
}
