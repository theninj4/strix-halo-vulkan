package bench

import (
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const bandwidthLocalSize = 256

// RunBandwidth measures copy.comp — the achievable GB/s ceiling other ops
// are judged against.
func RunBandwidth(dev *vk.Device, sizes []int, warmup, iters uint32) ([]Result, error) {
	shaderMod, err := dev.NewShaderModule(shaders.Copy)
	if err != nil {
		return nil, err
	}
	defer shaderMod.Destroy()

	var results []Result
	for _, n := range sizes {
		res, err := runCopyCase(dev, shaderMod, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("bandwidth copy size=%d: %w", n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func runCopyCase(dev *vk.Device, shaderMod *vk.ShaderModule, n int, warmup, iters uint32) (Result, error) {
	src, err := dev.NewBuffer(n * 4)
	if err != nil {
		return Result{}, err
	}
	defer src.Destroy()
	dst, err := dev.NewBuffer(n * 4)
	if err != nil {
		return Result{}, err
	}
	defer dst.Destroy()

	data := randomFloats(n)
	src.WriteFloat32(data)

	pipe, err := dev.NewPipeline(shaderMod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{src, dst},
		PushConstantSize: 4,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(n)).Bytes()
	groups := groupsFor(n, bandwidthLocalSize)

	if _, err := pipe.DispatchTimed(groups, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	out := dst.ReadFloat32(n)
	for i := range data {
		if out[i] != data[i] {
			return Result{}, fmt.Errorf("mismatch at %d: got %v want %v", i, out[i], data[i])
		}
	}

	ns, err := TimeDispatch(pipe, groups, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	bytesMoved := float64(n) * 4 * 2 // one read + one write, fp32
	return Result{
		Op: "bandwidth", Variant: "copy", Size: n,
		NsPerIter: ns,
		GBPS:      bytesMoved / (ns / 1e9) / 1e9,
	}, nil
}
