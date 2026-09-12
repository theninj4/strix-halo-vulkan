package bench

import (
	"encoding/binary"
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// runGEMMCoopMatQ4TwoPass measures the "dequantize Q4 weights to fp16 once,
// then reuse them for every subsequent cooperative-matrix matmul" strategy.
// A real inference engine loads weights once and reuses them across every
// forward pass, so the dequant cost is reported as its own one-time
// measurement (op="dequant") separate from steady-state matmul throughput
// (op="gemm", variant="coopmat_dequant"), which should run at essentially
// plain-fp16-coopmat speed once the weights are sitting in fp16.
func runGEMMCoopMatQ4TwoPass(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, block int, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil // already reported as skipped by runGEMMCoopMatFP16
	}

	dequantMod, err := dev.NewShaderModule(shaders.DequantQ4ToF16)
	if err != nil {
		return nil, err
	}
	defer dequantMod.Destroy()
	coopMod, err := dev.NewShaderModule(shaders.GEMMCoopMatFP16)
	if err != nil {
		return nil, err
	}
	defer coopMod.Destroy()

	specConsts := []vk.SpecConstant{
		{ID: 0, Value: uint32(shape.M)}, {ID: 1, Value: uint32(shape.N)}, {ID: 2, Value: uint32(shape.K)},
	}

	if err := verifyDequantQ4(dev, dequantMod, block); err != nil {
		return nil, fmt.Errorf("dequant q4->fp16 block=%d correctness check: %w", block, err)
	}
	if err := verifyCoopMatQ4TwoPass(dev, dequantMod, coopMod, shape, specConsts, block); err != nil {
		return nil, fmt.Errorf("gemm coopmat_dequant q4 block=%d correctness check: %w", block, err)
	}

	var results []Result
	for _, n := range sizes {
		if n%block != 0 || n%shape.M != 0 || n%shape.N != 0 || n%shape.K != 0 {
			continue
		}
		M, N, K := n, n, n

		bData := randomFloats(K * N)
		packed, scales := quantizeQ4(bData, K, N, block)

		packedBuf, err := dev.NewBuffer(len(packed))
		if err != nil {
			return nil, err
		}
		packedBuf.WriteBytes(packed)

		scalesBuf, err := dev.NewBuffer(len(scales) * 2)
		if err != nil {
			return nil, err
		}
		scalesBuf.WriteBytes(float16SliceToBytes(scales))

		bF16Buf, err := dev.NewBuffer(K * N * 2)
		if err != nil {
			return nil, err
		}

		dequantPipe, err := dev.NewPipeline(dequantMod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{packedBuf, scalesBuf, bF16Buf},
			PushConstantSize: 12,
		})
		if err != nil {
			return nil, err
		}

		dequantPC := newPC().U32(uint32(K)).U32(uint32(N)).U32(uint32(block)).Bytes()
		dequantGroups := groupsFor(K*N, 256)

		dequantNs, err := TimeDispatch(dequantPipe, dequantGroups, 1, 1, warmup, iters, dequantPC)
		if err != nil {
			return nil, err
		}

		dequantBytes := float64(len(packed)) + float64(len(scales)*2) + float64(K*N*2)
		results = append(results, Result{
			Op: "dequant", Variant: "q4_to_fp16", WeightFormat: "q4", BlockSize: block, Size: n,
			NsPerIter: dequantNs,
			GBPS:      dequantBytes / (dequantNs / 1e9) / 1e9,
		})

		dequantPipe.Destroy()
		packedBuf.Destroy()
		scalesBuf.Destroy()

		// bF16Buf now holds the dequantized fp16 weights; run the plain
		// fp16 coopmat matmul against it as-is, no re-upload — exactly
		// what a real engine does on every forward pass after load time.
		aData := randomFloats(M * K)
		aBuf, err := dev.NewBuffer(M * K * 2)
		if err != nil {
			return nil, err
		}
		aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))

		dummyScales, err := dev.NewBuffer(4)
		if err != nil {
			return nil, err
		}
		cBuf, err := dev.NewBuffer(M * N * 4)
		if err != nil {
			return nil, err
		}

		coopPipe, err := dev.NewPipeline(coopMod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{aBuf, bF16Buf, dummyScales, cBuf},
			PushConstantSize: 16,
			SpecConstants:    specConsts,
		})
		if err != nil {
			return nil, err
		}

		pc := gemmPushConstants(M, N, K, 0)
		groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)

		ns, err := TimeDispatch(coopPipe, groupsX, groupsY, 1, warmup, iters, pc)
		if err != nil {
			return nil, err
		}

		flops := float64(2 * M * N * K)
		results = append(results, Result{
			Op: "gemm", Variant: "coopmat_dequant", WeightFormat: "q4", BlockSize: block, Size: n,
			NsPerIter: ns,
			GFLOPS:    flops / (ns / 1e9) / 1e9,
		})

		coopPipe.Destroy()
		aBuf.Destroy()
		dummyScales.Destroy()
		cBuf.Destroy()
		bF16Buf.Destroy()
	}
	return results, nil
}

func verifyDequantQ4(dev *vk.Device, mod *vk.ShaderModule, block int) error {
	rows, cols := block*2, block*2
	bData := randomFloats(rows * cols)
	packed, scales := quantizeQ4(bData, rows, cols, block)

	packedBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return err
	}
	defer packedBuf.Destroy()
	packedBuf.WriteBytes(packed)

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	outBuf, err := dev.NewBuffer(rows * cols * 2)
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{packedBuf, scalesBuf, outBuf},
		PushConstantSize: 12,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := newPC().U32(uint32(rows)).U32(uint32(cols)).U32(uint32(block)).Bytes()
	groups := groupsFor(rows*cols, 256)
	if _, err := pipe.DispatchTimed(groups, 1, 1, 1, pc); err != nil {
		return err
	}

	outBytes := outBuf.ReadBytes(rows * cols * 2)
	// The shader stores the dequantized value as float16_t (one more
	// rounding step beyond the plain float32 arithmetic dequantizeQ4 does),
	// so round the reference through fp16 too before comparing.
	want := float16RoundTrip(dequantizeQ4(packed, scales, rows, cols, block))
	for i, w := range want {
		bits := binary.LittleEndian.Uint16(outBytes[i*2:])
		got := float16ToFloat32(bits)
		if got != w {
			return fmt.Errorf("mismatch at %d: got %v want %v", i, got, w)
		}
	}
	return nil
}

// verifyCoopMatQ4TwoPass checks the two-pass dequant-then-coopmat path once,
// at a size small enough that the O(size^3) CPU reference GEMM is cheap —
// this must NOT run inside the size sweep (an earlier version did, and a
// single N=4096 CPU reference GEMM in pure Go takes minutes: 4096^3*2 ≈
// 137 billion scalar float ops, repeated per block size).
func verifyCoopMatQ4TwoPass(dev *vk.Device, dequantMod, coopMod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant, block int) error {
	n := block
	if n%shape.M != 0 || n%shape.N != 0 || n%shape.K != 0 {
		return fmt.Errorf("verification size %d not compatible with coopmat shape %+v", n, shape)
	}
	M, N, K := n, n, n

	bData := randomFloats(K * N)
	packed, scales := quantizeQ4(bData, K, N, block)

	packedBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return err
	}
	defer packedBuf.Destroy()
	packedBuf.WriteBytes(packed)

	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return err
	}
	defer scalesBuf.Destroy()
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	bF16Buf, err := dev.NewBuffer(K * N * 2)
	if err != nil {
		return err
	}
	defer bF16Buf.Destroy()

	dequantPipe, err := dev.NewPipeline(dequantMod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{packedBuf, scalesBuf, bF16Buf},
		PushConstantSize: 12,
	})
	if err != nil {
		return err
	}
	defer dequantPipe.Destroy()

	dequantPC := newPC().U32(uint32(K)).U32(uint32(N)).U32(uint32(block)).Bytes()
	if _, err := dequantPipe.DispatchTimed(groupsFor(K*N, 256), 1, 1, 1, dequantPC); err != nil {
		return err
	}

	aData := randomFloats(M * K)
	aBuf, err := dev.NewBuffer(M * K * 2)
	if err != nil {
		return err
	}
	defer aBuf.Destroy()
	aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))

	dummyScales, err := dev.NewBuffer(4)
	if err != nil {
		return err
	}
	defer dummyScales.Destroy()
	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return err
	}
	defer cBuf.Destroy()

	coopPipe, err := dev.NewPipeline(coopMod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bF16Buf, dummyScales, cBuf},
		PushConstantSize: 16,
		SpecConstants:    specConsts,
	})
	if err != nil {
		return err
	}
	defer coopPipe.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)
	if _, err := coopPipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(M * N)

	refA := float16RoundTrip(aData)
	refB := float16RoundTrip(dequantizeQ4(packed, scales, K, N, block))
	want := cpuGEMM(refA, refB, M, N, K)
	return compareMat(got, want, 5e-2)
}

// q4CoopMatBuild bundles everything one fused Q4-coopmat dispatch needs, so
// the verify/time helpers below can share one builder.
type q4CoopMatBuild struct {
	pipe                        *vk.ComputePipeline
	aBuf, bBuf, scalesBuf, cBuf *vk.Buffer
	aData                       []float32
	packed                      []uint8
	scales                      []uint16
}

func (b *q4CoopMatBuild) destroy() {
	b.pipe.Destroy()
	b.aBuf.Destroy()
	b.bBuf.Destroy()
	b.scalesBuf.Destroy()
	b.cBuf.Destroy()
}

func buildCoopMatQ4Pipeline(dev *vk.Device, mod *vk.ShaderModule, specConsts []vk.SpecConstant, M, N, K, block int) (*q4CoopMatBuild, error) {
	aData := randomFloats(M * K)
	bData := randomFloats(K * N)
	packed, scales := quantizeQ4(bData, K, N, block)

	aBuf, err := dev.NewBuffer(M * K * 2)
	if err != nil {
		return nil, err
	}
	bBuf, err := dev.NewBuffer(len(packed))
	if err != nil {
		return nil, err
	}
	scalesBuf, err := dev.NewBuffer(len(scales) * 2)
	if err != nil {
		return nil, err
	}
	cBuf, err := dev.NewBuffer(M * N * 4)
	if err != nil {
		return nil, err
	}

	aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))
	bBuf.WriteBytes(packed)
	scalesBuf.WriteBytes(float16SliceToBytes(scales))

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
		SpecConstants:    specConsts,
	})
	if err != nil {
		return nil, err
	}

	return &q4CoopMatBuild{pipe: pipe, aBuf: aBuf, bBuf: bBuf, scalesBuf: scalesBuf, cBuf: cBuf, aData: aData, packed: packed, scales: scales}, nil
}

// runGEMMCoopMatQ4Fused measures the single-pass alternative to the
// two-pass strategy above: dequantize each K-tile of Q4 weights straight
// into shared memory and feed the cooperative-matrix accelerator from
// there (shaders/gemm_coopmat_q4.comp) — no separate dequant pass or
// fp16-sized scratch buffer, at the cost of redoing the dequant work on
// every dispatch instead of once.
func runGEMMCoopMatQ4Fused(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, block int, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}

	mod, err := dev.NewShaderModule(shaders.GEMMCoopMatQ4)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	specConsts := []vk.SpecConstant{
		{ID: 0, Value: uint32(shape.M)}, {ID: 1, Value: uint32(shape.N)}, {ID: 2, Value: uint32(shape.K)},
	}

	if err := verifyCoopMatQ4Fused(dev, mod, shape, specConsts, block); err != nil {
		return nil, fmt.Errorf("gemm coopmat fused q4 block=%d correctness check: %w", block, err)
	}

	var results []Result
	for _, n := range sizes {
		if n%block != 0 || n%shape.M != 0 || n%shape.N != 0 || n%shape.K != 0 {
			continue
		}
		res, err := timeCoopMatQ4Fused(dev, mod, shape, specConsts, n, n, n, block, warmup, iters)
		if err != nil {
			return nil, fmt.Errorf("gemm coopmat fused q4 block=%d size=%d: %w", block, n, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func verifyCoopMatQ4Fused(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant, block int) error {
	n := block // block is always a multiple of 16, so this satisfies both the block and tile-shape divisibility constraints
	b, err := buildCoopMatQ4Pipeline(dev, mod, specConsts, n, n, n, block)
	if err != nil {
		return err
	}
	defer b.destroy()

	pc := gemmPushConstants(n, n, n, block)
	groupsX, groupsY := uint32(n/shape.N), uint32(n/shape.M)
	if _, err := b.pipe.DispatchTimed(groupsX, groupsY, 1, 1, pc); err != nil {
		return err
	}
	got := b.cBuf.ReadFloat32(n * n)

	refA := float16RoundTrip(b.aData)
	refB := float16RoundTrip(dequantizeQ4(b.packed, b.scales, n, n, block))
	want := cpuGEMM(refA, refB, n, n, n)
	return compareMat(got, want, 5e-2)
}

func timeCoopMatQ4Fused(dev *vk.Device, mod *vk.ShaderModule, shape vk.CoopMatShape, specConsts []vk.SpecConstant, M, N, K, block int, warmup, iters uint32) (Result, error) {
	b, err := buildCoopMatQ4Pipeline(dev, mod, specConsts, M, N, K, block)
	if err != nil {
		return Result{}, err
	}
	defer b.destroy()

	pc := gemmPushConstants(M, N, K, block)
	groupsX, groupsY := uint32(N/shape.N), uint32(M/shape.M)

	ns, err := TimeDispatch(b.pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: "coopmat_fused", WeightFormat: "q4", BlockSize: block, Size: N,
		NsPerIter: ns,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}
