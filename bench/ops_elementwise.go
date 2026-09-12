package bench

import (
	"encoding/binary"
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const elementwiseLocalSize = 256

// opRelu matches shaders/elementwise.comp's op==2 branch (relu) — a
// representative bandwidth-bound activation.
const opRelu = 2

// RunElementwise measures elementwise.comp (relu) in fp32 and fp16.
func RunElementwise(dev *vk.Device, sizes []int, warmup, iters uint32) ([]Result, error) {
	f32Mod, err := dev.NewShaderModule(shaders.ElementwiseF32)
	if err != nil {
		return nil, err
	}
	defer f32Mod.Destroy()
	f16Mod, err := dev.NewShaderModule(shaders.ElementwiseF16)
	if err != nil {
		return nil, err
	}
	defer f16Mod.Destroy()

	var results []Result
	for _, n := range sizes {
		r, err := runElementwiseF32(dev, f32Mod, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("elementwise fp32 size=%d: %w", n, err)
		}
		results = append(results, r)

		r, err = runElementwiseF16(dev, f16Mod, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("elementwise fp16 size=%d: %w", n, err)
		}
		results = append(results, r)
	}
	return results, nil
}

func elementwisePushConstants(n int) []byte {
	return newPC().U32(uint32(n)).U32(opRelu).F32(0).Bytes()
}

func runElementwiseF32(dev *vk.Device, mod *vk.ShaderModule, n int, warmup, iters uint32) (Result, error) {
	buf, err := dev.NewBuffer(n * 4)
	if err != nil {
		return Result{}, err
	}
	defer buf.Destroy()

	data := randomFloats(n)
	buf.WriteFloat32(data)

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{buf}, PushConstantSize: 12})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := elementwisePushConstants(n)
	groups := groupsFor(n, elementwiseLocalSize)

	if _, err := pipe.DispatchTimed(groups, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	out := buf.ReadFloat32(n)
	for i, v := range data {
		want := v
		if want < 0 {
			want = 0
		}
		if out[i] != want {
			return Result{}, fmt.Errorf("mismatch at %d: got %v want %v", i, out[i], want)
		}
	}

	// Reset input (relu is destructive) before the timed run.
	buf.WriteFloat32(data)
	ns, clocks, err := TimeDispatch(pipe, groups, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	bytesMoved := float64(n) * 4 * 2 // one read + one write, in place
	return Result{Op: "elementwise", Variant: "relu", WeightFormat: "fp32", Size: n,
		NsPerIter: ns, GBPS: bytesMoved / (ns / 1e9) / 1e9, Clocks: clocks}, nil
}

func runElementwiseF16(dev *vk.Device, mod *vk.ShaderModule, n int, warmup, iters uint32) (Result, error) {
	buf, err := dev.NewBuffer(n * 2)
	if err != nil {
		return Result{}, err
	}
	defer buf.Destroy()

	data := randomFloats(n)
	buf.WriteBytes(float32SliceToFloat16Bytes(data))

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{buf}, PushConstantSize: 12})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := elementwisePushConstants(n)
	groups := groupsFor(n, elementwiseLocalSize)

	if _, err := pipe.DispatchTimed(groups, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	outBytes := buf.ReadBytes(n * 2)
	for i, v := range float16RoundTrip(data) {
		bits := binary.LittleEndian.Uint16(outBytes[i*2:])
		got := float16ToFloat32(bits)
		want := v
		if want < 0 {
			want = 0
		}
		if got != want {
			return Result{}, fmt.Errorf("mismatch at %d: got %v want %v", i, got, want)
		}
	}

	buf.WriteBytes(float32SliceToFloat16Bytes(data))
	ns, clocks, err := TimeDispatch(pipe, groups, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	bytesMoved := float64(n) * 2 * 2 // one read + one write, in place, fp16
	return Result{Op: "elementwise", Variant: "relu", WeightFormat: "fp16", Size: n,
		NsPerIter: ns, GBPS: bytesMoved / (ns / 1e9) / 1e9, Clocks: clocks}, nil
}
