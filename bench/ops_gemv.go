package bench

import (
	"fmt"
	"os"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// gemvVariant is one reduction strategy (naive serial loop vs. one-subgroup-
// per-row) across all four weight-format shader binaries.
type gemvVariant struct {
	name      string
	shaderF32 []byte
	shaderF16 []byte
	shaderQ8  []byte
	shaderQ4  []byte
	groupsX   func(M int) uint32
}

// RunGEMV measures y = W*x — the dominant op in autoregressive LLM decode —
// across naive/subgroup reduction strategies and fp32/fp16/q8/q4 weights,
// plus a W8A8 (int8 weights AND activations, via packed dot-product
// instructions) variant if the device supports it.
func RunGEMV(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, blocks []int, warmup, iters uint32) ([]Result, error) {
	variants := []gemvVariant{
		{
			name: "naive", shaderF32: shaders.GEMVNaiveF32, shaderF16: shaders.GEMVNaiveF16,
			shaderQ8: shaders.GEMVNaiveQ8, shaderQ4: shaders.GEMVNaiveQ4,
			groupsX: func(M int) uint32 { return groupsFor(M, 64) },
		},
		{
			name: "subgroup", shaderF32: shaders.GEMVSubgroupF32, shaderF16: shaders.GEMVSubgroupF16,
			shaderQ8: shaders.GEMVSubgroupQ8, shaderQ4: shaders.GEMVSubgroupQ4,
			groupsX: func(M int) uint32 { return uint32(M) },
		},
	}

	var results []Result
	for _, v := range variants {
		f32Mod, err := dev.NewShaderModule(v.shaderF32)
		if err != nil {
			return nil, err
		}
		defer f32Mod.Destroy()
		f16Mod, err := dev.NewShaderModule(v.shaderF16)
		if err != nil {
			return nil, err
		}
		defer f16Mod.Destroy()
		q8Mod, err := dev.NewShaderModule(v.shaderQ8)
		if err != nil {
			return nil, err
		}
		defer q8Mod.Destroy()
		q4Mod, err := dev.NewShaderModule(v.shaderQ4)
		if err != nil {
			return nil, err
		}
		defer q4Mod.Destroy()

		for _, n := range sizes {
			M, N := n, n

			r, err := runGEMVFloat(dev, f32Mod, v.name, "fp32", v.groupsX(M), M, N, warmup, iters, false)
			if err != nil {
				return nil, fmt.Errorf("gemv %s fp32 size=%d: %w", v.name, n, err)
			}
			results = append(results, r)

			r, err = runGEMVFloat(dev, f16Mod, v.name, "fp16", v.groupsX(M), M, N, warmup, iters, true)
			if err != nil {
				return nil, fmt.Errorf("gemv %s fp16 size=%d: %w", v.name, n, err)
			}
			results = append(results, r)

			for _, block := range blocks {
				if N%block != 0 {
					continue
				}
				r, err = runGEMVQ8(dev, q8Mod, v.name, v.groupsX(M), M, N, block, warmup, iters)
				if err != nil {
					return nil, fmt.Errorf("gemv %s q8 block=%d size=%d: %w", v.name, block, n, err)
				}
				results = append(results, r)

				r, err = runGEMVQ4(dev, q4Mod, v.name, v.groupsX(M), M, N, block, warmup, iters)
				if err != nil {
					return nil, fmt.Errorf("gemv %s q4 block=%d size=%d: %w", v.name, block, n, err)
				}
				results = append(results, r)
			}
		}
	}

	feat, err := phys.SupportedFeatures()
	if err != nil {
		return nil, err
	}
	if !feat.IntegerDotProduct {
		fmt.Fprintln(os.Stderr, "gemv w8a8: shaderIntegerDotProduct not supported, skipping")
	} else {
		w8a8, err := runGEMVW8A8(dev, sizes, blocks, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, w8a8...)
	}
	return results, nil
}

func gemvActivation(dev *vk.Device, N int) (*vk.Buffer, []float32, error) {
	x, err := dev.NewBuffer(N * 4)
	if err != nil {
		return nil, nil, err
	}
	xData := randomFloats(N)
	x.WriteFloat32(xData)
	return x, xData, nil
}

func cpuGEMV(w, x []float32, M, N int) []float32 {
	y := make([]float32, M)
	for m := 0; m < M; m++ {
		var acc float32
		base := m * N
		for n := 0; n < N; n++ {
			acc += w[base+n] * x[n]
		}
		y[m] = acc
	}
	return y
}

func compareVec(got, want []float32, relTol float32) error {
	for i := range want {
		if !approxEqual(got[i], want[i], relTol) {
			return fmt.Errorf("mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
	return nil
}

func runGEMVFloat(dev *vk.Device, mod *vk.ShaderModule, variant, weightFormat string, groupsX uint32, M, N int, warmup, iters uint32, isF16 bool) (Result, error) {
	wData := randomFloats(M * N)

	elemSize := 4
	if isF16 {
		elemSize = 2
	}
	wBuf, err := dev.NewBuffer(M * N * elemSize)
	if err != nil {
		return Result{}, err
	}
	defer wBuf.Destroy()
	if isF16 {
		wBuf.WriteBytes(float32SliceToFloat16Bytes(wData))
	} else {
		wBuf.WriteFloat32(wData)
	}

	scalesBuf, err := dev.NewBuffer(4) // dummy, unused by the f32/f16 shader path
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()

	x, xData, err := gemvActivation(dev, N)
	if err != nil {
		return Result{}, err
	}
	defer x.Destroy()
	y, err := dev.NewBuffer(M * 4)
	if err != nil {
		return Result{}, err
	}
	defer y.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{wBuf, scalesBuf, x, y},
		PushConstantSize: 12,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(M)).U32(uint32(N)).U32(0).Bytes()

	if _, err := pipe.DispatchTimed(groupsX, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	got := y.ReadFloat32(M)

	refW := wData
	tol := float32(1e-4)
	if isF16 {
		refW = float16RoundTrip(wData)
		tol = 5e-3
	}
	want := cpuGEMV(refW, xData, M, N)
	if err := compareVec(got, want, tol); err != nil {
		return Result{}, fmt.Errorf("correctness check failed: %w", err)
	}

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	weightBytes := float64(M * N * elemSize)
	flops := float64(2 * M * N)
	return Result{
		Op: "gemv", Variant: variant, WeightFormat: weightFormat, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      weightBytes / (ns / 1e9) / 1e9,
	}, nil
}

func runGEMVQ8(dev *vk.Device, mod *vk.ShaderModule, variant string, groupsX uint32, M, N, block int, warmup, iters uint32) (Result, error) {
	wData := randomFloats(M * N)
	q, scales := quantizeQ8(wData, M, N, block)

	wBuf, err := dev.NewBuffer(len(q))
	if err != nil {
		return Result{}, err
	}
	defer wBuf.Destroy()
	wBuf.WriteBytes(int8SliceToBytes(q))

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	x, xData, err := gemvActivation(dev, N)
	if err != nil {
		return Result{}, err
	}
	defer x.Destroy()
	y, err := dev.NewBuffer(M * 4)
	if err != nil {
		return Result{}, err
	}
	defer y.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{wBuf, scalesBuf, x, y},
		PushConstantSize: 12,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(M)).U32(uint32(N)).U32(uint32(block)).Bytes()

	if _, err := pipe.DispatchTimed(groupsX, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	got := y.ReadFloat32(M)

	refW := dequantizeQ8(q, scales, M, N, block)
	want := cpuGEMV(refW, xData, M, N)
	if err := compareVec(got, want, 1e-3); err != nil {
		return Result{}, fmt.Errorf("correctness check failed: %w", err)
	}

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	weightBytes := float64(len(q)) + float64(len(scales)*2)
	flops := float64(2 * M * N)
	return Result{
		Op: "gemv", Variant: variant, WeightFormat: "q8", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      weightBytes / (ns / 1e9) / 1e9,
	}, nil
}

func runGEMVQ4(dev *vk.Device, mod *vk.ShaderModule, variant string, groupsX uint32, M, N, block int, warmup, iters uint32) (Result, error) {
	wData := randomFloats(M * N)
	packed, scales := quantizeQ4(wData, M, N, block)

	wBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return Result{}, err
	}
	defer wBuf.Destroy()
	wBuf.WriteBytes(packed)

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	x, xData, err := gemvActivation(dev, N)
	if err != nil {
		return Result{}, err
	}
	defer x.Destroy()
	y, err := dev.NewBuffer(M * 4)
	if err != nil {
		return Result{}, err
	}
	defer y.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{wBuf, scalesBuf, x, y},
		PushConstantSize: 12,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(M)).U32(uint32(N)).U32(uint32(block)).Bytes()

	if _, err := pipe.DispatchTimed(groupsX, 1, 1, 1, pc); err != nil {
		return Result{}, err
	}
	got := y.ReadFloat32(M)

	refW := dequantizeQ4(packed, scales, M, N, block)
	want := cpuGEMV(refW, xData, M, N)
	if err := compareVec(got, want, 1e-3); err != nil {
		return Result{}, fmt.Errorf("correctness check failed: %w", err)
	}

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	weightBytes := float64(len(packed)) + float64(len(scales)*2)
	flops := float64(2 * M * N)
	return Result{
		Op: "gemv", Variant: variant, WeightFormat: "q4", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      weightBytes / (ns / 1e9) / 1e9,
	}, nil
}
