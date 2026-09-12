package bench

import (
	"fmt"
	"os"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// correctnessSize is small enough that an O(size^3) CPU reference GEMM runs
// in milliseconds; correctness is checked once per variant at this size,
// not at every swept (much larger) size — see the plan's Phase 2 note.
const gemmCorrectnessSize = 64

func cpuGEMM(a, b []float32, M, N, K int) []float32 {
	c := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc float32
			for k := 0; k < K; k++ {
				acc += a[m*K+k] * b[k*N+n]
			}
			c[m*N+n] = acc
		}
	}
	return c
}

func compareMat(got, want []float32, relTol float32) error {
	for i := range want {
		if !approxEqual(got[i], want[i], relTol) {
			return fmt.Errorf("mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
	return nil
}

// gemmPushConstants packs {M,N,K,block} matching every gemm_*.comp shader's
// push_constant layout.
func gemmPushConstants(M, N, K, block int) []byte {
	return newPC().U32(uint32(M)).U32(uint32(N)).U32(uint32(K)).U32(uint32(block)).Bytes()
}

// RunGEMM measures C = A*B across naive/shared-memory-tiled/cooperative-
// matrix strategies and fp32/fp16/q8 weights. phys is needed to discover
// which MxNxK shapes VK_KHR_cooperative_matrix actually supports on this
// device; coopmat cases are skipped (not failed) if the feature or a
// matching shape isn't available.
func RunGEMM(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, blocks []int, warmup, iters uint32) ([]Result, error) {
	var results []Result

	naiveF32, err := runGEMMPlainVariant(dev, "naive", "fp32", shaders.GEMMNaiveF32, 16, false, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, naiveF32...)

	naiveF16, err := runGEMMPlainVariant(dev, "naive", "fp16", shaders.GEMMNaiveF16, 16, true, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, naiveF16...)

	tiledF32, err := runGEMMPlainVariant(dev, "tiled", "fp32", shaders.GEMMTiledF32, 16, false, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, tiledF32...)

	tiledF16, err := runGEMMPlainVariant(dev, "tiled", "fp16", shaders.GEMMTiledF16, 16, true, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, tiledF16...)

	for _, block := range blocks {
		q8, err := runGEMMQ8Variant(dev, "naive", shaders.GEMMNaiveQ8, 16, block, sizes, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, q8...)

		q4, err := runGEMMQ4Variant(dev, "naive", shaders.GEMMNaiveQ4, 16, block, sizes, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, q4...)

		tiledQ8, err := runGEMMQ8Variant(dev, "tiled", shaders.GEMMTiledQ8, 16, block, sizes, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, tiledQ8...)

		tiledQ4, err := runGEMMQ4Variant(dev, "tiled", shaders.GEMMTiledQ4, 16, block, sizes, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, tiledQ4...)
	}

	coopFP16, err := runGEMMCoopMatFP16(dev, phys, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, coopFP16...)

	coopInt8, err := runGEMMCoopMatInt8(dev, phys, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, coopInt8...)

	wmma, err := runGEMMWMMA(dev, phys, sizes, warmup, iters)
	if err != nil {
		return nil, err
	}
	results = append(results, wmma...)

	for _, block := range blocks {
		twoPass, err := runGEMMCoopMatQ4TwoPass(dev, phys, sizes, block, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, twoPass...)

		fused, err := runGEMMCoopMatQ4Fused(dev, phys, sizes, block, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, fused...)
	}

	feat, err := phys.SupportedFeatures()
	if err != nil {
		return nil, err
	}
	if !feat.IntegerDotProduct {
		fmt.Fprintln(os.Stderr, "gemm w8a8: shaderIntegerDotProduct not supported, skipping")
	} else {
		w8a8, err := runGEMMW8A8(dev, sizes, blocks, warmup, iters)
		if err != nil {
			return nil, err
		}
		results = append(results, w8a8...)
	}

	return results, nil
}

func runGEMMPlainVariant(dev *vk.Device, variant, weightFormat string, spirv []byte, localSize int, isF16 bool, sizes []int, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(spirv)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	if err := verifyGEMMPlain(dev, mod, isF16); err != nil {
		return nil, fmt.Errorf("gemm %s %s correctness check: %w", variant, weightFormat, err)
	}

	var results []Result
	for _, n := range sizes {
		res, err := timeGEMMPlain(dev, mod, variant, weightFormat, localSize, isF16, n, n, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm %s %s size=%d: %w", variant, weightFormat, n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func verifyGEMMPlain(dev *vk.Device, mod *vk.ShaderModule, isF16 bool) error {
	n := gemmCorrectnessSize
	res, aData, bData, err := dispatchGEMMPlainOnce(dev, mod, isF16, n, n, n)
	if err != nil {
		return err
	}
	refA, refB := aData, bData
	tol := float32(1e-3)
	if isF16 {
		refA, refB = float16RoundTrip(aData), float16RoundTrip(bData)
		tol = 1e-2
	}
	want := cpuGEMM(refA, refB, n, n, n)
	return compareMat(res, want, tol)
}

func dispatchGEMMPlainOnce(dev *vk.Device, mod *vk.ShaderModule, isF16 bool, M, N, K int) (c, aData, bData []float32, err error) {
	elemSize := 4
	if isF16 {
		elemSize = 2
	}
	aData = randomFloats(M * K)
	bData = randomFloats(K * N)

	aBuf, err := dev.NewBuffer(M * K * elemSize)
	if err != nil {
		return nil, nil, nil, err
	}
	defer aBuf.Destroy()
	bBuf, err := dev.NewBuffer(K * N * elemSize)
	if err != nil {
		return nil, nil, nil, err
	}
	defer bBuf.Destroy()
	if isF16 {
		aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))
		bBuf.WriteBytes(float32SliceToFloat16Bytes(bData))
	} else {
		aBuf.WriteFloat32(aData)
		bBuf.WriteFloat32(bData)
	}
	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return nil, nil, nil, err
	}
	defer scalesBuf.Destroy()
	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return nil, nil, nil, err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return nil, nil, nil, err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := groupsFor(N, 16), groupsFor(M, 16)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return nil, nil, nil, err
	}
	return cBuf.ReadFloat32(M * N), aData, bData, nil
}

func timeGEMMPlain(dev *vk.Device, mod *vk.ShaderModule, variant, weightFormat string, localSize int, isF16 bool, M, N, K int, warmup, iters uint32) (Result, error) {
	elemSize := 4
	if isF16 {
		elemSize = 2
	}
	aData := randomFloats(M * K)
	bData := randomFloats(K * N)

	aBuf, err := dev.NewBuffer(M * K * elemSize)
	if err != nil {
		return Result{}, err
	}
	defer aBuf.Destroy()
	bBuf, err := dev.NewBuffer(K * N * elemSize)
	if err != nil {
		return Result{}, err
	}
	defer bBuf.Destroy()
	if isF16 {
		aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))
		bBuf.WriteBytes(float32SliceToFloat16Bytes(bData))
	} else {
		aBuf.WriteFloat32(aData)
		bBuf.WriteFloat32(bData)
	}
	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return Result{}, err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := groupsFor(N, localSize), groupsFor(M, localSize)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: variant, WeightFormat: weightFormat, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}

func runGEMMQ8Variant(dev *vk.Device, variant string, spirv []byte, localSize, block int, sizes []int, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(spirv)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	if err := verifyGEMMQ8(dev, mod, block); err != nil {
		return nil, fmt.Errorf("gemm %s q8 block=%d correctness check: %w", variant, block, err)
	}

	var results []Result
	for _, n := range sizes {
		if n%block != 0 {
			continue
		}
		res, err := timeGEMMQ8(dev, mod, variant, localSize, n, n, n, block, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm %s q8 block=%d size=%d: %w", variant, block, n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func verifyGEMMQ8(dev *vk.Device, mod *vk.ShaderModule, block int) error {
	n := gemmCorrectnessSize
	if n%block != 0 {
		n = block * 2
	}
	aData := randomFloats(n * n)
	bData := randomFloats(n * n)
	q, scales := quantizeQ8(bData, n, n, block)

	aBuf, err := dev.NewBuffer(n * n * 4)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	aBuf.WriteFloat32(aData)

	bBuf, err := dev.NewBuffer(len(q))
	if err != nil {
		return err
	}
	defer bBuf.Destroy()
	bBuf.WriteBytes(int8SliceToBytes(q))

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	cBuf, err := dev.NewBuffer(n * n * 4)
	if err != nil {
		return err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(n, n, n, block)
	groupsX, groupsY := groupsFor(n, 16), groupsFor(n, 16)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(n * n)

	refB := dequantizeQ8(q, scales, n, n, block)
	want := cpuGEMM(aData, refB, n, n, n)
	return compareMat(got, want, 1e-2)
}

func timeGEMMQ8(dev *vk.Device, mod *vk.ShaderModule, variant string, localSize, M, N, K, block int, warmup, iters uint32) (Result, error) {
	aData := randomFloats(M * K)
	bData := randomFloats(K * N)
	q, scales := quantizeQ8(bData, K, N, block)

	aBuf, err := dev.NewBuffer(M * K * 4)
	if err != nil {
		return Result{}, err
	}
	defer aBuf.Destroy()
	aBuf.WriteFloat32(aData)

	bBuf, err := dev.NewBuffer(len(q))
	if err != nil {
		return Result{}, err
	}
	defer bBuf.Destroy()
	bBuf.WriteBytes(int8SliceToBytes(q))

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return Result{}, err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, block)
	groupsX, groupsY := groupsFor(N, localSize), groupsFor(M, localSize)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: variant, WeightFormat: "q8", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}

func runGEMMQ4Variant(dev *vk.Device, variant string, spirv []byte, localSize, block int, sizes []int, warmup, iters uint32) ([]Result, error) {
	mod, err := dev.NewShaderModule(spirv)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	if err := verifyGEMMQ4(dev, mod, block); err != nil {
		return nil, fmt.Errorf("gemm %s q4 block=%d correctness check: %w", variant, block, err)
	}

	var results []Result
	for _, n := range sizes {
		if n%block != 0 {
			continue
		}
		res, err := timeGEMMQ4(dev, mod, variant, localSize, n, n, n, block, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm %s q4 block=%d size=%d: %w", variant, block, n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func verifyGEMMQ4(dev *vk.Device, mod *vk.ShaderModule, block int) error {
	n := gemmCorrectnessSize
	if n%block != 0 {
		n = block * 2
	}
	aData := randomFloats(n * n)
	bData := randomFloats(n * n)
	packed, scales := quantizeQ4(bData, n, n, block)

	aBuf, err := dev.NewBuffer(n * n * 4)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	aBuf.WriteFloat32(aData)

	bBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return err
	}
	defer bBuf.Destroy()
	bBuf.WriteBytes(packed)

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	cBuf, err := dev.NewBuffer(n * n * 4)
	if err != nil {
		return err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(n, n, n, block)
	groupsX, groupsY := groupsFor(n, 16), groupsFor(n, 16)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(n * n)

	refB := dequantizeQ4(packed, scales, n, n, block)
	want := cpuGEMM(aData, refB, n, n, n)
	return compareMat(got, want, 1e-2)
}

func timeGEMMQ4(dev *vk.Device, mod *vk.ShaderModule, variant string, localSize, M, N, K, block int, warmup, iters uint32) (Result, error) {
	aData := randomFloats(M * K)
	bData := randomFloats(K * N)
	packed, scales := quantizeQ4(bData, K, N, block)

	aBuf, err := dev.NewBuffer(M * K * 4)
	if err != nil {
		return Result{}, err
	}
	defer aBuf.Destroy()
	aBuf.WriteFloat32(aData)

	bBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return Result{}, err
	}
	defer bBuf.Destroy()
	bBuf.WriteBytes(packed)

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return Result{}, err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return Result{}, err
	}
	defer cBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemmPushConstants(M, N, K, block)
	groupsX, groupsY := groupsFor(N, localSize), groupsFor(M, localSize)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: variant, WeightFormat: "q4", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}

// findCoopMatShape picks the first subgroup-scope shape matching aType/cType.
func findCoopMatShape(phys *vk.PhysicalDevice, aType, cType vk.ComponentType) (vk.CoopMatShape, bool, error) {
	feat, err := phys.SupportedFeatures()
	if err != nil {
		return vk.CoopMatShape{}, false, err
	}
	if !feat.CoopMatrix {
		return vk.CoopMatShape{}, false, nil
	}
	shapes, err := phys.CooperativeMatrixShapes()
	if err != nil {
		return vk.CoopMatShape{}, false, err
	}
	for _, s := range shapes {
		if s.Scope == vk.ScopeSubgroup && s.AType == aType && s.BType == aType && s.CType == cType && s.ResultType == cType {
			return s, true, nil
		}
	}
	return vk.CoopMatShape{}, false, nil
}

func runGEMMCoopMatFP16(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok {
		fmt.Fprintln(os.Stderr, "gemm coopmat fp16: no matching cooperative-matrix shape reported, skipping")
		return nil, nil
	}

	mod, err := dev.NewShaderModule(shaders.GEMMCoopMatFP16)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	specConsts := []vk.SpecConstant{
		{ID: 0, Value: uint32(shape.M)}, {ID: 1, Value: uint32(shape.N)}, {ID: 2, Value: uint32(shape.K)},
	}

	if err := verifyCoopMatFP16(dev, mod, shape, specConsts); err != nil {
		return nil, fmt.Errorf("gemm coopmat fp16 correctness check: %w", err)
	}

	var results []Result
	for _, n := range sizes {
		if n%shape.M != 0 || n%shape.N != 0 || n%shape.K != 0 {
			continue
		}
		res, err := timeCoopMatFP16(dev, mod, shape, specConsts, n, n, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm coopmat fp16 size=%d: %w", n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func buildCoopMatFP16Pipeline(dev *vk.Device, mod *vk.ShaderModule, specConsts []vk.SpecConstant, M, N, K int) (pipe *vk.ComputePipeline, aBuf, bBuf, cBuf *vk.Buffer, aData, bData []float32, err error) {
	aData = randomFloats(M * K)
	bData = randomFloats(K * N)

	aBuf, err = dev.NewBuffer(M * K * 2)
	if err != nil {
		return
	}
	bBuf, err = dev.NewBuffer(K * N * 2)
	if err != nil {
		return
	}
	aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))
	bBuf.WriteBytes(float32SliceToFloat16Bytes(bData))

	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return
	}
	cBuf, err = dev.NewBuffer(M * N * 4)
	if err != nil {
		return
	}

	pipe, err = dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
		SpecConstants:    specConsts,
	})
	return
}

func verifyCoopMatFP16(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant) error {
	M, N, K := shape.M*2, shape.N*2, shape.K*2
	pipe, aBuf, bBuf, cBuf, aData, bData, err := buildCoopMatFP16Pipeline(dev, mod, specConsts, M, N, K)
	if err != nil {
		return err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(M * N)

	refA, refB := float16RoundTrip(aData), float16RoundTrip(bData)
	want := cpuGEMM(refA, refB, M, N, K)
	return compareMat(got, want, 5e-2)
}

func timeCoopMatFP16(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant, M, N, K int, warmup, iters uint32) (Result, error) {
	pipe, aBuf, bBuf, cBuf, _, _, err := buildCoopMatFP16Pipeline(dev, mod, specConsts, M, N, K)
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: "coopmat", WeightFormat: "fp16", Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}

func runGEMMCoopMatInt8(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentSInt8, vk.ComponentSInt32)
	if err != nil {
		return nil, err
	}
	if !ok {
		fmt.Fprintln(os.Stderr, "gemm coopmat int8: no matching cooperative-matrix shape reported, skipping")
		return nil, nil
	}

	mod, err := dev.NewShaderModule(shaders.GEMMCoopMatInt8)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	specConsts := []vk.SpecConstant{
		{ID: 0, Value: uint32(shape.M)}, {ID: 1, Value: uint32(shape.N)}, {ID: 2, Value: uint32(shape.K)},
	}

	if err := verifyCoopMatInt8(dev, mod, shape, specConsts); err != nil {
		return nil, fmt.Errorf("gemm coopmat int8 correctness check: %w", err)
	}

	var results []Result
	for _, n := range sizes {
		if n%shape.M != 0 || n%shape.N != 0 || n%shape.K != 0 {
			continue
		}
		res, err := timeCoopMatInt8(dev, mod, shape, specConsts, n, n, n, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm coopmat int8 size=%d: %w", n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func randomInt8s(n int, maxAbs int8) []int8 {
	out := make([]int8, n)
	data := randomFloats(n)
	for i, v := range data {
		out[i] = int8(v * float32(maxAbs))
	}
	return out
}

func cpuGEMMInt8(a, b []int8, M, N, K int) []int32 {
	c := make([]int32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc int32
			for k := 0; k < K; k++ {
				acc += int32(a[m*K+k]) * int32(b[k*N+n])
			}
			c[m*N+n] = acc
		}
	}
	return c
}

func buildCoopMatInt8Pipeline(dev *vk.Device, mod *vk.ShaderModule, specConsts []vk.SpecConstant, M, N, K int) (pipe *vk.ComputePipeline, aBuf, bBuf, cBuf *vk.Buffer, aData, bData []int8, err error) {
	aData = randomInt8s(M*K, 8)
	bData = randomInt8s(K*N, 8)

	aBuf, err = dev.NewBuffer(M * K)
	if err != nil {
		return
	}
	bBuf, err = dev.NewBuffer(K * N)
	if err != nil {
		return
	}
	aBuf.WriteBytes(int8SliceToBytes(aData))
	bBuf.WriteBytes(int8SliceToBytes(bData))

	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return
	}
	cBuf, err = dev.NewBuffer(M * N * 4) // int32 output, 4 bytes/elem
	if err != nil {
		return
	}

	pipe, err = dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
		SpecConstants:    specConsts,
	})
	return
}

func readInt32Buffer(buf *vk.Buffer, n int) []int32 {
	raw := buf.ReadBytes(n * 4)
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		out[i] = int32(raw[i*4]) | int32(raw[i*4+1])<<8 | int32(raw[i*4+2])<<16 | int32(raw[i*4+3])<<24
	}
	return out
}

func verifyCoopMatInt8(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant) error {
	M, N, K := shape.M*2, shape.N*2, shape.K*2
	pipe, aBuf, bBuf, cBuf, aData, bData, err := buildCoopMatInt8Pipeline(dev, mod, specConsts, M, N, K)
	if err != nil {
		return err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)
	if _, err := pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := readInt32Buffer(cBuf, M*N)
	want := cpuGEMMInt8(aData, bData, M, N, K)
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("mismatch at %d: got %d want %d", i, got[i], want[i])
		}
	}
	return nil
}

func timeCoopMatInt8(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant, M, N, K int, warmup, iters uint32) (Result, error) {
	pipe, aBuf, bBuf, cBuf, _, _, err := buildCoopMatInt8Pipeline(dev, mod, specConsts, M, N, K)
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: "coopmat", WeightFormat: "q8", Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}
