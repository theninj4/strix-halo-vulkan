package bench

import (
	"fmt"
	"os"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// wmmaVariant is one tile geometry of shaders/gemm_wmma.comp. The geometry is
// baked into the SPIR-V (it sizes register and `shared` arrays), so the Go
// side has to be told it rather than reading it back: bm/bn/bk give the
// dispatch grid and reject shapes the variant can't tile, and colMajorB says
// whether B must be uploaded as [N,K].
type wmmaVariant struct {
	name       string
	spirv      []byte
	bm, bn, bk int  // workgroup tile: M, N, and K-slab depth per iteration
	waves      int  // waves per workgroup
	colMajorB  bool // B stored [N,K] and loaded column-major (IDEAS §2.3)
}

// intensity is the kernel's arithmetic intensity in FLOP per byte of A and B
// read from global memory: 2*BM*BN*BK flops per (BM+BN)*BK*2 bytes, which
// reduces to BM*BN/(BM+BN). This is the number IDEAS §2.1 argues is the
// binding constraint — gemm_coopmat_fp16 supplies 8, and saturating the
// measured 55.5 TFLOP/s at 236 GB/s would need ~235 — so it is reported with
// each measurement rather than left to be recomputed from the tile shape.
//
// For the single-wave variants BM x BN is that wave's own accumulator grid;
// the reuse is in registers rather than shared memory, but the global traffic,
// and so this figure, is the same. For the multi-wave *non*-LDS (wg*)
// variants it is an upper bound rather than a fact: the four waves issue
// overlapping fragment loads and only reach it if L0/L1 dedups them, which is
// exactly what those rows are there to measure.
func (v wmmaVariant) intensity() float64 {
	return float64(v.bm*v.bn) / float64(v.bm+v.bn)
}

// wmmaVariants is an ablation ordered so it reads off the results table.
// Arithmetic intensity climbs 16 -> 32 -> 43 -> 64 -> 85 FLOP/byte against
// gemm_coopmat_fp16's 8; reg64 vs reg64_bt isolates the B layout at constant
// intensity, lds128 vs lds128_db isolates double buffering, and lds128_db vs
// lds128_db_bt isolates the B layout again inside the LDS path, where it
// changes the shared-memory read codegen rather than the global one.
var wmmaVariants = []wmmaVariant{
	{name: "wmma_reg32", spirv: shaders.GEMMWMMAReg32, bm: 32, bn: 32, bk: 16, waves: 1},
	{name: "wmma_reg64", spirv: shaders.GEMMWMMAReg64, bm: 64, bn: 64, bk: 16, waves: 1},
	{name: "wmma_reg64_bt", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true},
	{name: "wmma_reg64x128", spirv: shaders.GEMMWMMAReg64x128, bm: 64, bn: 128, bk: 16, waves: 1},
	{name: "wmma_reg64_bt_k64", spirv: shaders.GEMMWMMAReg64BTK64, bm: 64, bn: 64, bk: 64, waves: 1, colMajorB: true},
	{name: "wmma_wg128", spirv: shaders.GEMMWMMAWG128, bm: 128, bn: 128, bk: 16, waves: 4},
	{name: "wmma_wg128x256", spirv: shaders.GEMMWMMAWG128x256, bm: 128, bn: 256, bk: 16, waves: 4},
	{name: "wmma_lds128", spirv: shaders.GEMMWMMALDS128, bm: 128, bn: 128, bk: 16, waves: 4},
	{name: "wmma_lds128_db", spirv: shaders.GEMMWMMALDS128DB, bm: 128, bn: 128, bk: 16, waves: 4},
	{name: "wmma_lds128_db_bt", spirv: shaders.GEMMWMMALDS128DBBT, bm: 128, bn: 128, bk: 16, waves: 4, colMajorB: true},
	{name: "wmma_lds128k32_db_bt", spirv: shaders.GEMMWMMALDS128K32DBBT, bm: 128, bn: 128, bk: 32, waves: 4, colMajorB: true},
	{name: "wmma_lds256x128_db_bt", spirv: shaders.GEMMWMMALDS256x128DBBT, bm: 256, bn: 128, bk: 16, waves: 4, colMajorB: true},
}

// runGEMMWMMA measures the register-blocked cooperative-matrix GEMM
// (IDEAS.md §2.1) against the same fp16 inputs and fp32 output as
// runGEMMCoopMatFP16, so the two are directly comparable row for row.
func runGEMMWMMA(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, warmup, iters uint32) ([]Result, error) {
	shape, ok, err := findCoopMatShape(phys, vk.ComponentFloat16, vk.ComponentFloat32)
	if err != nil {
		return nil, err
	}
	if !ok {
		fmt.Fprintln(os.Stderr, "gemm wmma: no matching cooperative-matrix shape reported, skipping")
		return nil, nil
	}
	// The shader hard-codes the 16x16x16 shape because it is the only one
	// this device reports; refuse to produce numbers on a device where that
	// stopped being true rather than silently computing the wrong thing.
	if shape.M != 16 || shape.N != 16 || shape.K != 16 {
		fmt.Fprintf(os.Stderr, "gemm wmma: device reports %dx%dx%d, shader is built for 16x16x16, skipping\n",
			shape.M, shape.N, shape.K)
		return nil, nil
	}

	var results []Result
	for _, v := range wmmaVariants {
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()

		if err := verifyGEMMWMMA(dev, mod, v); err != nil {
			return nil, fmt.Errorf("gemm %s correctness check: %w", v.name, err)
		}
		for _, n := range sizes {
			if n%v.bm != 0 || n%v.bn != 0 || n%v.bk != 0 {
				continue
			}
			res, err := timeGEMMWMMA(dev, mod, v, n, n, n, warmup, iters)
			if err != nil {
				return nil, fmt.Errorf("gemm %s size=%d: %w", v.name, n, err)
			}
			results = append(results, res)
		}
	}
	return results, nil
}

// buildWMMAPipeline allocates one case's buffers with the same 4-binding
// layout every gemm_*.comp uses (A, B, unused scales, C) and the same
// {M,N,K,block} push constants. bData is always the logical [K,N] matrix, so
// the CPU reference is written once regardless of layout; the _bt variants
// upload its [N,K] transpose, which is both what RDNA's WMMA B operand wants
// and how a real Linear weight is already stored.
func buildWMMAPipeline(dev *vk.Device, mod *vk.ShaderModule, v wmmaVariant, M, N, K int) (pipe *vk.ComputePipeline, aBuf, bBuf, cBuf *vk.Buffer, aData, bData []float32, err error) {
	aData = randomFloats(M * K)
	bData = randomFloats(K * N)

	if aBuf, err = dev.NewBuffer(M * K * 2); err != nil {
		return
	}
	if bBuf, err = dev.NewBuffer(K * N * 2); err != nil {
		return
	}
	aBuf.WriteBytes(float32SliceToFloat16Bytes(aData))
	bUpload := bData
	if v.colMajorB {
		bUpload = transpose(bData, K, N)
	}
	bBuf.WriteBytes(float32SliceToFloat16Bytes(bUpload))

	scalesBuf, err := dev.NewBuffer(4)
	if err != nil {
		return
	}
	if cBuf, err = dev.NewBuffer(M * N * 4); err != nil {
		return
	}

	pipe, err = dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{aBuf, bBuf, scalesBuf, cBuf},
		PushConstantSize: 16,
	})
	return
}

// verifyGEMMWMMA checks one variant at the smallest shape that still
// exercises what could go wrong: 2x2 workgroups (so the tile origins and the
// wave-to-tile mapping have to be right, not just the degenerate
// single-workgroup case) over 4 K-slabs (so the accumulator carries across
// iterations and, for the LDS variants, both barriers matter).
func verifyGEMMWMMA(dev *vk.Device, mod *vk.ShaderModule, v wmmaVariant) error {
	M, N, K := v.bm*2, v.bn*2, v.bk*4
	pipe, aBuf, bBuf, cBuf, aData, bData, err := buildWMMAPipeline(dev, mod, v, M, N, K)
	if err != nil {
		return err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	if _, err := pipe.DispatchTimed(uint32(N/v.bn), uint32(M/v.bm), 1, 1, pc); err != nil {
		return err
	}
	got := cBuf.ReadFloat32(M * N)

	refA, refB := float16RoundTrip(aData), float16RoundTrip(bData)
	want := cpuGEMM(refA, refB, M, N, K)
	return compareMat(got, want, 5e-2)
}

func timeGEMMWMMA(dev *vk.Device, mod *vk.ShaderModule, v wmmaVariant, M, N, K int, warmup, iters uint32) (Result, error) {
	pipe, aBuf, bBuf, cBuf, _, _, err := buildWMMAPipeline(dev, mod, v, M, N, K)
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()
	defer aBuf.Destroy()
	defer bBuf.Destroy()
	defer cBuf.Destroy()

	pc := gemmPushConstants(M, N, K, 0)
	groupsX, groupsY := uint32(N/v.bn), uint32(M/v.bm)

	ns, clocks, err := TimeDispatch(pipe, groupsX, groupsY, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	// Workgroup count matters here in a way it doesn't for the other GEMMs:
	// §0.1 found WMMA needs 2 waves per CU (80 waves on this part) to reach
	// full rate, and a 256x128 tile at N=256 launches two workgroups. Recording
	// it makes an under-occupied row identifiable as one.
	waves := int(groupsX*groupsY) * v.waves
	flops := float64(2 * M * N * K)
	return Result{
		Op: "gemm", Variant: v.name, WeightFormat: "fp16", Size: N,
		Detail:    fmt.Sprintf("tile=%dx%dx%d waves=%d ai=%.0f", v.bm, v.bn, v.bk, waves, v.intensity()),
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}
