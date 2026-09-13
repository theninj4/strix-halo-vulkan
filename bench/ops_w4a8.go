package bench

import (
	"fmt"
	"math/bits"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// w4a8Variant is one load width of shaders/gemv_w4a8.comp. VEC=1 reads one
// uint (8 weights) per lane per step, VEC=4 one uvec4 (32 weights) — the
// same kernel otherwise, so the pair isolates load width (IDEAS §1.3) from
// everything else the W4A8 path changes.
type w4a8Variant struct {
	name           string
	spirv          []byte
	weightsPerLoad int
	// waveSize pins the pipeline's subgroup size (IDEAS §6.2); zero takes the
	// default of 64. The kernel reduces a whole row with subgroupAdd, so a
	// binary built with -DWAVE=32 must be run at 32 and nothing else.
	waveSize uint32
}

var w4a8Variants = []w4a8Variant{
	{name: "subgroup", spirv: shaders.GEMVW4A8, weightsPerLoad: 8},
	{name: "subgroup_vec4", spirv: shaders.GEMVW4A8Vec4, weightsPerLoad: 32},
	// IDEAS §6.2 on the kernel that is already at 89% of the DRAM bus, so the
	// prediction is "no change" — which is worth a row because the wave size
	// is the one knob here that changes the *access pattern*: 32 lanes
	// sweeping a row means each lane's stride doubles, and §5.1b's coverage
	// law prices strides. A move either way is informative; a flat pair is the
	// falsification test the law asks for.
	{name: "subgroup_w32", spirv: shaders.GEMVW4A8W32, weightsPerLoad: 8, waveSize: 32},
	{name: "subgroup_vec4_w32", spirv: shaders.GEMVW4A8Vec4W32, weightsPerLoad: 32, waveSize: 32},
}

// runGEMVW4A8 measures y = W*x with 4-bit weights and int8 activations, both
// fed to the packed-int8 dot instruction (shaders/gemv_w4a8.comp). This is
// IDEAS.md §1.1: same bytes as the PRECISION_Q4 path, same arithmetic as
// the W8A8 path.
func runGEMVW4A8(dev *vk.Device, phys *vk.PhysicalDevice, sizes []int, blocks []int, warmup, iters uint32) ([]Result, error) {
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		return nil, err
	}
	variants := filterWaveVariants(w4a8Variants, sgs, func(v w4a8Variant) (string, uint32) {
		return "gemv " + v.name + " w4a8", v.waveSize
	})

	var results []Result
	for _, v := range variants {
		mod, err := dev.NewShaderModule(v.spirv)
		if err != nil {
			return nil, err
		}
		defer mod.Destroy()

		for _, block := range blocks {
			if !w4a8BlockOK(block, v.weightsPerLoad) {
				continue
			}
			if err := verifyGEMVW4A8(dev, mod, v.waveSize, block); err != nil {
				return nil, fmt.Errorf("gemv %s w4a8 block=%d correctness check: %w", v.name, block, err)
			}
			for _, n := range sizes {
				M, N := n, n
				if N%block != 0 || N%v.weightsPerLoad != 0 {
					continue
				}
				res, err := timeGEMVW4A8(dev, mod, v, M, N, block, warmup, iters)
				if err != nil {
					return nil, fmt.Errorf("gemv %s w4a8 block=%d size=%d: %w", v.name, block, n, err)
				}
				results = append(results, res)
			}
		}
	}
	return results, nil
}

// w4a8BlockOK reports whether a quantization block size is usable by the
// kernel: it must be a power of two (the shader shifts by log2(block)
// instead of dividing, IDEAS §1.2) and at least one load wide, so that a
// load's worth of weights never straddles two scales.
func w4a8BlockOK(block, weightsPerLoad int) bool {
	return block >= weightsPerLoad && block&(block-1) == 0
}

func log2u(v int) uint32 {
	return uint32(bits.TrailingZeros(uint(v)))
}

// gemvW4A8PushConstants packs {M,N,logBlock,xScale} matching gemv_w4a8.comp.
func gemvW4A8PushConstants(M, N, block int, xScale float32) []byte {
	return newPC().U32(uint32(M)).U32(uint32(N)).U32(log2u(block)).F32(xScale).Bytes()
}

// w4a8Buffers is one W4A8 GEMV's device-side state. The activation-side
// buffers (x and its per-block sums) are what a real engine would produce
// per token from the previous layer's output.
type w4a8Buffers struct {
	w, scales, x, y, xSums *vk.Buffer
	xScale                 float32
}

func (b *w4a8Buffers) destroy() {
	for _, buf := range []*vk.Buffer{b.w, b.scales, b.x, b.y, b.xSums} {
		if buf != nil {
			buf.Destroy()
		}
	}
}

func (b *w4a8Buffers) list() []*vk.Buffer {
	return []*vk.Buffer{b.w, b.scales, b.x, b.y, b.xSums}
}

func buildGEMVW4A8Buffers(dev *vk.Device, M, N, block int) (bufs w4a8Buffers, xData []float32, packedW []uint8, wScales []uint16, err error) {
	wData := randomFloats(M * N)
	packedW, wScales = quantizeQ4(wData, M, N, block)
	words := repackQ4ToW4A8(packedW, M, N)

	if bufs.w, err = dev.NewBuffer(len(words) * 4); err != nil {
		return
	}
	bufs.w.WriteBytes(uint32SliceToBytes(words))

	if bufs.scales, err = dev.NewBuffer(len(wScales) * 2); err != nil {
		return
	}
	bufs.scales.WriteBytes(float16SliceToBytes(wScales))

	xData = randomFloats(N)
	packedX, xScales := quantizeQ8(xData, 1, N, N) // one scale for the whole vector
	bufs.xScale = float16ToFloat32(xScales[0])

	if bufs.x, err = dev.NewBuffer(len(packedX)); err != nil {
		return
	}
	bufs.x.WriteBytes(int8SliceToBytes(packedX))

	sums := int8BlockSums(packedX, block)
	if bufs.xSums, err = dev.NewBuffer(len(sums) * 4); err != nil {
		return
	}
	bufs.xSums.WriteBytes(int32SliceToBytes(sums))

	bufs.y, err = dev.NewBuffer(M * 4)
	return
}

func verifyGEMVW4A8(dev *vk.Device, mod *vk.ShaderModule, waveSize uint32, block int) error {
	n := gemmCorrectnessSize
	if n%block != 0 {
		n = block * 2
	}
	M, N := n, n

	bufs, xData, packedW, wScales, err := buildGEMVW4A8Buffers(dev, M, N, block)
	defer bufs.destroy()
	if err != nil {
		return err
	}

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              bufs.list(),
		PushConstantSize:     16,
		RequiredSubgroupSize: waveSize,
	})
	if err != nil {
		return err
	}
	defer pipe.Destroy()

	pc := gemvW4A8PushConstants(M, N, block, bufs.xScale)
	if _, err := pipe.DispatchTimed(uint32(M), 1, 1, 1, pc); err != nil {
		return err
	}
	got := bufs.y.ReadFloat32(M)

	// The reference is built from the *pre-repack* Q4 arrays, so it checks
	// repackQ4ToW4A8's permutation as well as the kernel.
	refW := dequantizeQ4(packedW, wScales, M, N, block)
	packedX, xScales := quantizeQ8(xData, 1, N, N)
	refX := dequantizeQ8(packedX, xScales, 1, N, N)
	want := cpuGEMV(refW, refX, M, N)
	return compareVec(got, want, 2e-2)
}

func timeGEMVW4A8(dev *vk.Device, mod *vk.ShaderModule, v w4a8Variant, M, N, block int, warmup, iters uint32) (Result, error) {
	bufs, _, _, _, err := buildGEMVW4A8Buffers(dev, M, N, block)
	defer bufs.destroy()
	if err != nil {
		return Result{}, err
	}

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              bufs.list(),
		PushConstantSize:     16,
		RequiredSubgroupSize: v.waveSize,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	pc := gemvW4A8PushConstants(M, N, block, bufs.xScale)
	groupsX := uint32(M)

	ns, clocks, err := TimeDispatch(pipe, groupsX, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	weightBytes := float64(M*N)/2 + float64((M*N/block)*2)
	flops := float64(2 * M * N)
	return Result{
		Op: "gemv", Variant: v.name, WeightFormat: "w4a8", BlockSize: block, Size: N,
		NsPerIter: ns,
		Clocks:    clocks,
		GFLOPS:    flops / (ns / 1e9) / 1e9,
		GBPS:      weightBytes / (ns / 1e9) / 1e9,
	}, nil
}
