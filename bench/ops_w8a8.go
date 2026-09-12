package bench

import (
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// runGEMVW8A8 measures y = W*x with both W and x quantized to int8,
// reduced via VK_KHR_shader_integer_dot_product's packed-int8 dot
// instruction (see shaders/gemv_w8a8.comp) instead of the
// dequantize-then-float-multiply PRECISION_Q8 path RunGEMV already covers.
// Subgroup reduction only, to compare directly against subgroup q8/q4 —
// the variants that actually win on this hardware (see TODO.md).
func runGEMVW8A8(dev *vk.Device, sizes []int, blocks []int, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(shaders.GEMVW8A8)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	var results []Result
	for _, block := range blocks {
		if err := verifyGEMVW8A8(dev, mod, block); err != nil {
			return nil, fmt.Errorf("gemv w8a8 block=%d correctness check: %w", block, err)
		}
		for _, n := range sizes {
			M, N := n, n
			if N%block != 0 {
				continue
			}
			res, err := timeGEMVW8A8(dev, mod, M, N, block, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemv w8a8 block=%d size=%d: %w", block, n, err)
			}
			results = append(results, res)
		}
	}
	return results, nil
}

// w8a8PushConstants packs {M,N,block,xScale} matching gemv_w8a8.comp.
func gemvW8A8PushConstants(M, N, block int, xScale float32) []byte {
	return newPC().U32(uint32(M)).U32(uint32(N)).U32(uint32(block)).F32(xScale).Bytes()
}

func buildGEMVW8A8Buffers(dev *vk.Device, M, N, block int) (wBuf, scalesBuf, xBuf, yBuf *vk.Buffer, wData, xData []float32, xScale float32, err error) {
	wData = randomFloats(M * N)
	packedW, wScales := quantizeQ8(wData, M, N, block)

	wBuf, err = dev.NewBuffer(len(packedW))
	if err != nil {
		return
	}
	wBuf.WriteBytes(int8SliceToBytes(packedW))

	scalesBuf, err = dev.NewBuffer(len(wScales) * 2)
	if err != nil {
		return
	}
	scalesBuf.WriteBytes(float16SliceToBytes(wScales))

	xData = randomFloats(N)
	packedX, xScales := quantizeQ8(xData, 1, N, N) // one scale for the whole vector
	xScale = float16ToFloat32(xScales[0])

	xBuf, err = dev.NewBuffer(len(packedX))
	if err != nil {
		return
	}
	xBuf.WriteBytes(int8SliceToBytes(packedX))

	yBuf, err = dev.NewBuffer(M * 4)
	return
}

func verifyGEMVW8A8(dev *vk.Device, mod *vk.ShaderModule, block int) error {
	n := gemmCorrectnessSize
	if n%block != 0 {
		n = block * 2
	}
	M, N := n, n

	wBuf, scalesBuf, xBuf, yBuf, wData, xData, xScale, err := buildGEMVW8A8Buffers(dev, M, N, block)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	defer scalesBuf.Destroy()
	defer xBuf.Destroy()
	defer yBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{wBuf, scalesBuf, xBuf, yBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := gemvW8A8PushConstants(M, N, block, xScale)
	if _, err := pipe.DispatchTimed(uint32(M), 1, 1, 1, pc); err != nil {
		return err
	}
	got := yBuf.ReadFloat32(M)

	packedW, wScales := quantizeQ8(wData, M, N, block)
	refW := dequantizeQ8(packedW, wScales, M, N, block)
	packedX, xScales := quantizeQ8(xData, 1, N, N)
	refX := dequantizeQ8(packedX, xScales, 1, N, N)
	want := cpuGEMV(refW, refX, M, N)
	return compareVec(got, want, 2e-2)
}

func timeGEMVW8A8(dev *vk.Device, mod *vk.ShaderModule, M, N, block int, warmup, iters uint32) (Result, error) {
	wBuf, scalesBuf, xBuf, yBuf, _, _, xScale, err := buildGEMVW8A8Buffers(dev, M, N, block)
	if err != nil {
		return Result{}, err
	}
	defer wBuf.Destroy()
	defer scalesBuf.Destroy()
	defer xBuf.Destroy()
	defer yBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{wBuf, scalesBuf, xBuf, yBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemvW8A8PushConstants(M, N, block, xScale)
	groupsX := uint32(M)

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	weightBytes := float64(M*N) + float64((M*N/block)*2)
	flops := float64(2 * M * N)
	return Result{
		Op: "gemv", Variant: "subgroup", WeightFormat: "w8a8", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      weightBytes / (ns / 1e9) / 1e9,
	}, nil
}

// runGEMMW8A8 measures C = A*B with both A and B quantized to int8, reduced
// via dotPacked4x8EXT. B is stored transposed (NxK) so each output
// column's K values are contiguous — see gemm_w8a8.comp. Naive (one thread
// per output element) only: this isn't meant to compete with the coopmat
// int8 path (a different hardware unit, already covered by
// gemm_coopmat_int8.comp), just to measure the packed-dot-instruction
// approach against the naive dequant-and-multiply baselines.
func runGEMMW8A8(dev *vk.Device, sizes []int, blocks []int, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(shaders.GEMMW8A8)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	var results []Result
	for _, block := range blocks {
		if err := verifyGEMMW8A8(dev, mod, block); err != nil {
			return nil, fmt.Errorf("gemm w8a8 block=%d correctness check: %w", block, err)
		}
		for _, n := range sizes {
			if n%block != 0 || n%4 != 0 {
				continue
			}
			res, err := timeGEMMW8A8(dev, mod, n, n, n, block, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemm w8a8 block=%d size=%d: %w", block, n, err)
			}
			results = append(results, res)
		}
	}
	return results, nil
}

func buildGEMMW8A8Buffers(dev *vk.Device, M, N, K, block int) (aBuf, aScalesBuf, bBuf, bScalesBuf, cBuf *vk.Buffer, aData, bTData []float32, err error) {
	aData = randomFloats(M * K)
	packedA, aScales := quantizeQ8(aData, M, K, K) // one scale per row of A

	aBuf, err = dev.NewBuffer(len(packedA))
	if err != nil {
		return
	}
	aBuf.WriteBytes(int8SliceToBytes(packedA))

	aScalesBuf, err = dev.NewBuffer(len(aScales) * 2)
	if err != nil {
		return
	}
	aScalesBuf.WriteBytes(float16SliceToBytes(aScales))

	bTData = randomFloats(N * K) // B transposed: N rows x K cols
	packedB, bScales := quantizeQ8(bTData, N, K, block)

	bBuf, err = dev.NewBuffer(len(packedB))
	if err != nil {
		return
	}
	bBuf.WriteBytes(int8SliceToBytes(packedB))

	bScalesBuf, err = dev.NewBuffer(len(bScales) * 2)
	if err != nil {
		return
	}
	bScalesBuf.WriteBytes(float16SliceToBytes(bScales))

	cBuf, err = dev.NewBuffer(M * N * 4)
	return
}

// transpose returns the cols x rows transpose of an rows x cols row-major
// matrix.
func transpose(m []float32, rows, cols int) []float32 {
	out := make([]float32, len(m))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = m[r*cols+c]
		}
	}
	return out
}

func verifyGEMMW8A8(dev *vk.Device, mod *vk.ShaderModule, block int) error {
	n := gemmCorrectnessSize
	if n%block != 0 {
		n = block * 2
	}
	M, N, K := n, n, n

	aBuf, aScalesBuf, bBuf, bScalesBuf, cBuf, aData, bTData, err := buildGEMMW8A8Buffers(dev, M, N, K, block)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	defer aScalesBuf.Destroy()
	defer bBuf.Destroy()
	defer bScalesBuf.Destroy()
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, aScalesBuf, bBuf, bScalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, block)
	groupsX, groupsY := groupsFor(N, 16), groupsFor(M, 16)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(M * N)

	packedA, aScales := quantizeQ8(aData, M, K, K)
	refA := dequantizeQ8(packedA, aScales, M, K, K)
	packedB, bScales := quantizeQ8(bTData, N, K, block)
	refBT := dequantizeQ8(packedB, bScales, N, K, block)
	refB := transpose(refBT, N, K)
	want := cpuGEMM(refA, refB, M, N, K)
	return compareMat(got, want, 3e-2)
}

func timeGEMMW8A8(dev *vk.Device, mod *vk.ShaderModule, M, N, K, block int, warmup, iters uint32) (Result, error) {
	aBuf, aScalesBuf, bBuf, bScalesBuf, cBuf, _, _, err := buildGEMMW8A8Buffers(dev, M, N, K, block)
	if err != nil {
		return Result{}, err
	}
	defer aBuf.Destroy()
	defer aScalesBuf.Destroy()
	defer bBuf.Destroy()
	defer bScalesBuf.Destroy()
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, aScalesBuf, bBuf, bScalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, block)
	groupsX, groupsY := groupsFor(N, 16), groupsFor(M, 16)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: "naive", WeightFormat: "w8a8", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}
