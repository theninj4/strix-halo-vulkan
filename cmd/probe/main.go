// Command probe dispatches a single SPIR-V compute shader once, so the RADV
// debug environment variables can be pointed at it: `RADV_DEBUG=asm` prints
// the generated RDNA disassembly and `RADV_DEBUG=shaderstats` the VGPR/SGPR,
// LDS and spill counts. That is IDEAS.md §6.1's "dump and read the ISA", and
// it is how the B-operand layout effect in shaders/gemm_wmma.comp was found:
// the instruction mix, not the timings, showed that a row-major
// cooperative-matrix B fragment is gathered by sixteen scalar 16-bit loads
// where a column-major one is two buffer_load_b128.
//
//	RADV_DEBUG=shaderstats go run ./cmd/probe shaders/gemm_wmma_reg64.spv
//	RADV_DEBUG=asm         go run ./cmd/probe shaders/gemm_wmma_reg64.spv
//
// An optional second argument pins the wave size the shader is compiled for
// (IDEAS.md §6.2), which is the only way to see what a variant costs at
// wave32 — the register file is per lane, so the VGPR count RADV prints is a
// different number at each size and it is the number that decides how far the
// hoisting ladder of §2.7 can go:
//
//	RADV_DEBUG=shaderstats go run ./cmd/probe shaders/gemm_wmma_reg64_w32.spv 32
//
// The shader is dispatched with one workgroup, zeroed push constants, and a
// single small buffer bound at every binding it declares, so it is only
// useful for compilation-time questions — nothing it prints depends on the
// shader computing anything meaningful. Kernels that need real buffers or a
// real grid belong in the bench suite, not here.
package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

// probeBindings is bound-above the binding count of any shader here; extra
// descriptors in the set are harmless, a missing one is not.
const probeBindings = 8

// probePushConstants is likewise bound-above the push-constant block of any
// shader here (the largest is the VAE's 88 bytes, vae_common.glsl). A layout
// range smaller than the block the shader declares is invalid, so err large.
const probePushConstants = 128

func main() {
	if len(os.Args) != 2 && len(os.Args) != 3 {
		log.Fatalf("usage: %s <shader.spv> [subgroup-size]", os.Args[0])
	}
	subgroupSize := 0
	if len(os.Args) == 3 {
		n, err := strconv.Atoi(os.Args[2])
		if err != nil {
			log.Fatalf("subgroup size %q: %v", os.Args[2], err)
		}
		subgroupSize = n
	}
	if err := run(os.Args[1], uint32(subgroupSize)); err != nil {
		log.Fatal(err)
	}
}

func run(path string, subgroupSize uint32) error {
	spirv, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	instance, err := vk.NewInstance("strix-halo-probe")
	if err != nil {
		return err
	}
	defer instance.Destroy()

	devices, err := instance.PhysicalDevices()
	if err != nil {
		return err
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
		}
	}
	fmt.Printf("device: %s\n", phys.Name)

	queueFamily, err := phys.ComputeQueueFamily()
	if err != nil {
		return err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		return err
	}
	if subgroupSize != 0 && !sgs.Supported {
		return fmt.Errorf("device does not support a required subgroup size")
	}

	dev, err := vk.NewDevice(phys, queueFamily, vk.DeviceFeatures{
		Float16: true, Int8: true, IntegerDotProduct: true, CoopMatrix: true,
		SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		return err
	}
	defer dev.Destroy()

	mod, err := dev.NewShaderModule(spirv)
	if err != nil {
		return err
	}
	defer mod.Destroy()

	bufs := make([]*vk.Buffer, probeBindings)
	for i := range bufs {
		b, err := dev.NewBuffer(4096)
		if err != nil {
			return err
		}
		defer b.Destroy()
		bufs[i] = b
	}

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: probePushConstants, RequiredSubgroupSize: subgroupSize,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	_, err = pipe.DispatchTimed(1, 1, 1, 1, make([]byte, probePushConstants))
	return err
}
