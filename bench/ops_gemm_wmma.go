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
	// padB pads B's leading dimension by this many halves: B's rows are
	// stored ldb = (K or N) + padB apart while its extents stay powers of
	// two. This is a run-time knob (the shader takes ldb as a push constant),
	// so a padded case reuses its unpadded case's SPIR-V and differs in
	// nothing else — which is what makes the pair a clean test of IDEAS
	// §2.3's channel/bank-aliasing hypothesis. Must be a multiple of 8 to
	// keep each row 16-byte aligned, or coopMatLoad drops back to scalar
	// 16-bit loads and the comparison measures alignment instead of aliasing.
	padB int
	// padA is the same knob for A, whose fragment loads are K-strided in
	// every variant here — so unlike padB it applies to the kernels that
	// already win, not only to the transposed-B ones.
	padA int
}

// lda is A's leading dimension in elements: A is always [M, K+padA].
func (v wmmaVariant) lda(K int) int { return K + v.padA }

// ldb is B's leading dimension in elements: B is [N, K+padB] for the
// column-major variants and [K, N+padB] for the row-major ones.
func (v wmmaVariant) ldb(N, K int) int {
	if v.colMajorB {
		return K + v.padB
	}
	return N + v.padB
}

// bRows is how many rows of B are stored, i.e. the other extent from ldb.
func (v wmmaVariant) bRows(N, K int) int {
	if v.colMajorB {
		return N
	}
	return K
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

// wmmaVariants is an ablation ordered so it reads off the results table. It
// has two halves. The first varies *geometry*: arithmetic intensity climbs
// 16 -> 32 -> 43 -> 64 -> 85 FLOP/byte against gemm_coopmat_fp16's 8;
// reg64 vs reg64_bt isolates the B layout at constant intensity, lds128 vs
// lds128_db isolates double buffering, and lds128_db vs lds128_db_bt
// isolates the B layout again inside the LDS path, where it changes the
// shared-memory read codegen rather than the global one.
//
// The second half varies *stride* at fixed geometry — same SPIR-V, one
// different push-constant word — because that turned out to matter as much
// as the tiling did (IDEAS §2.3): a sweep of B's leading dimension against
// the 2 KB channel-interleave period, the same pad applied to A on the three
// kernels that win, and two controls where the mechanism predicted no
// effect — it was right about one of them and wrong about the other.
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

	// IDEAS §2.3's last hypothesis: the column-major B load is slow because
	// its 16 in-flight addresses are a power-of-two stride apart and so
	// alias onto one DRAM channel and bank. Each of these reuses the SPIR-V
	// of the unpadded row above it and changes only ldb, so any difference
	// is the stride and nothing else.
	//
	// At K=4096 the unpadded row stride is 8192 B, a multiple of every
	// plausible channel-interleave period. The pads sweep the stride in
	// bytes from 8208 to 12288: 16 B (pad 8, the smallest that keeps 16-byte
	// row alignment), then 128 B, 256 B, 512 B and 1 KB, then 2 KB and 4 KB
	// — the last two land back on a multiple of 2 KB, the likeliest
	// channel-interleave rotation, so if the effect is periodic in the
	// stride they should regress to the unpadded row's speed rather than
	// continuing the trend. That is the difference between "aliasing" and
	// "any pad at all helps".
	{name: "wmma_reg64_bt_pad8", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 8},
	{name: "wmma_reg64_bt_pad64", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 64},
	{name: "wmma_reg64_bt_pad128", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 128},
	{name: "wmma_reg64_bt_pad256", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 256},
	{name: "wmma_reg64_bt_pad512", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 512},
	{name: "wmma_reg64_bt_pad1024", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 1024},
	{name: "wmma_reg64_bt_pad2048", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padB: 2048},
	// Control: the same pad on the row-major winner, whose B fragments are
	// gathered element by element rather than by a 16-address strided load.
	// If padding is about the strided gather, this row should not move — and
	// it doesn't (23538 vs 24530 at N=4096, inside this kernel's own 23.5-24.9
	// spread across runs), which is also what says padding costs nothing.
	{name: "wmma_reg64_pad64", spirv: shaders.GEMMWMMAReg64, bm: 64, bn: 64, bk: 16, waves: 1, padB: 64},
	// Second control, and the one that came out wrong: the LDS path's global
	// B reads are contiguous 128-bit staging loads rather than coopMatLoad
	// gathers, so this was predicted not to move. It moves by 1.55x (13286 ->
	// 20597 at N=4096), because a staging load is contiguous only *within* a
	// slab row and consecutive rows are ldb*2 bytes apart — so the wave's 64
	// addresses are the same aliased pattern. The stride across concurrent
	// loads is what matters, not which instruction issues them.
	{name: "wmma_lds128_db_bt_pad64", spirv: shaders.GEMMWMMALDS128DBBT, bm: 128, bn: 128, bk: 16, waves: 4, colMajorB: true, padB: 64},

	// With the aliasing removed, re-test the one mechanism the *unpadded*
	// sweep refuted: a column-major fragment consumes only 32 B of each
	// 128 B line, so a deeper K-slab should pay off — it didn't before, but
	// before, aliasing was the larger term and could have masked it. It
	// still doesn't above MALL size (14026 vs 14028 at N=4096), and still
	// does below it (19212 vs 17661 at N=1024).
	{name: "wmma_reg64_bt_k64_pad128", spirv: shaders.GEMMWMMAReg64BTK64, bm: 64, bn: 64, bk: 64, waves: 1, colMajorB: true, padB: 128},

	// A's fragment loads are K-strided in every variant, so the aliasing
	// found on B should be costing the *winning* kernels too. These pad A
	// alone, and then both operands, on the row-major winner and on the best
	// transposed-B case. It is: 1.35x on `reg64` at N=2048, and 0-5% at
	// N=4096 — kernel-dependent and at the edge of the run-to-run spread.
	{name: "wmma_reg64_pada128", spirv: shaders.GEMMWMMAReg64, bm: 64, bn: 64, bk: 16, waves: 1, padA: 128},
	{name: "wmma_reg64x128_pada128", spirv: shaders.GEMMWMMAReg64x128, bm: 64, bn: 128, bk: 16, waves: 1, padA: 128},
	{name: "wmma_wg128x256_pada128", spirv: shaders.GEMMWMMAWG128x256, bm: 128, bn: 256, bk: 16, waves: 4, padA: 128},
	{name: "wmma_reg64_bt_padab128", spirv: shaders.GEMMWMMAReg64BT, bm: 64, bn: 64, bk: 16, waves: 1, colMajorB: true, padA: 128, padB: 128},
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
// layout every gemm_*.comp uses (A, B, unused scales, C) and this kernel's
// {M,N,K,block,ldb,lda} push constants — six words where the other
// gemm_*.comp kernels push four. bData is always the logical [K,N] matrix, so
// the CPU reference is written once regardless of layout; the _bt variants
// upload its [N,K] transpose, which is both what RDNA's WMMA B operand wants
// and how a real Linear weight is already stored.
func buildWMMAPipeline(dev *vk.Device, mod *vk.ShaderModule, v wmmaVariant, M, N, K int) (pipe *vk.ComputePipeline, aBuf, bBuf, cBuf *vk.Buffer, aData, bData []float32, err error) {
	aData = randomFloats(M * K)
	bData = randomFloats(K * N)

	lda, ldb, bRows := v.lda(K), v.ldb(N, K), v.bRows(N, K)
	if aBuf, err = dev.NewBuffer(M * lda * 2); err != nil {
		return
	}
	if bBuf, err = dev.NewBuffer(bRows * ldb * 2); err != nil {
		return
	}
	aUpload := aData
	if v.padA != 0 {
		aUpload = padRows(aUpload, M, K, lda)
	}
	aBuf.WriteBytes(float32SliceToFloat16Bytes(aUpload))
	bUpload := bData
	if v.colMajorB {
		bUpload = transpose(bData, K, N)
	}
	if v.padB != 0 {
		bUpload = padRows(bUpload, bRows, len(bUpload)/bRows, ldb)
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
		PushConstantSize: wmmaPushConstantSize,
	})
	return
}

// padRows re-lays a rows x cols matrix at a wider row stride, leaving the pad
// elements zero. The kernel never reads them — they exist only to move where
// the rows it does read land in the memory system (IDEAS §2.3).
func padRows(m []float32, rows, cols, stride int) []float32 {
	out := make([]float32, rows*stride)
	for r := 0; r < rows; r++ {
		copy(out[r*stride:r*stride+cols], m[r*cols:(r+1)*cols])
	}
	return out
}

// wmmaPushConstantSize is the byte size of that block, and must match the
// GLSL exactly: a pipeline layout range shorter than the block the shader
// declares leaves every read past it out of range, which a driver is free to
// satisfy with garbage rather than diagnose.
const wmmaPushConstantSize = 24

// wmmaPushConstants is gemmPushConstants plus the two leading dimensions,
// which the wmma kernels take separately from K and N so either stride can
// be padded without changing the matrix extents.
func wmmaPushConstants(M, N, K, ldb, lda int) []byte {
	pc := newPC().U32(uint32(M)).U32(uint32(N)).U32(uint32(K)).U32(0).
		U32(uint32(ldb)).U32(uint32(lda)).Bytes()
	if len(pc) != wmmaPushConstantSize {
		// Keeping these two in sync by hand is exactly how a stride ends up
		// being read from outside the declared range, which the driver
		// answers with garbage rather than an error.
		panic(fmt.Sprintf("wmma push constants: built %d bytes, layout declares %d",
			len(pc), wmmaPushConstantSize))
	}
	return pc
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

	pc := wmmaPushConstants(M, N, K, v.ldb(N, K), v.lda(K))
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

	pc := wmmaPushConstants(M, N, K, v.ldb(N, K), v.lda(K))
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
		Detail: fmt.Sprintf("tile=%dx%dx%d waves=%d ai=%.0f strideA=%dB strideB=%dB",
			v.bm, v.bn, v.bk, waves, v.intensity(), v.lda(K)*2, v.ldb(N, K)*2),
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
	}, nil
}
