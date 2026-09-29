package bench

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	aluPeakLocalSize = 64
	// aluPeakInputFloats is how many operand elements the input buffer
	// holds. The scalar modes read lanes at lid, lid+64, lid+128 and
	// lid+192; the coopmat modes read a full 16x16 tile. 256 covers both.
	aluPeakInputFloats = 256
	// aluPeakProbeReps is the rep count used to estimate per-rep cost
	// before choosing the real one. Large enough that the ~30us
	// per-dispatch floor is a small fraction of the probe.
	aluPeakProbeReps = 20000
	// aluPeakTargetNs is how long one calibrated dispatch should take:
	// long enough for launch overhead to be negligible, short enough that
	// TimeDispatch still batches several iterations inside its budget.
	aluPeakTargetNs = 20_000_000
	aluPeakMaxReps  = 4_000_000
)

// peakWaveCounts is the swept dispatch size, in workgroups — and since each
// workgroup is exactly one 64-lane subgroup, in waves. 40 is this chip's CU
// count, so the sweep runs from one wave per CU up to 32, which is where
// instruction issue rather than occupancy should be the limit.
var peakWaveCounts = []int{40, 80, 160, 320, 640, 1280}

// peakMode is one instruction-issue ceiling to measure.
type peakMode struct {
	name  string
	spirv []byte
	// operands builds the input buffer's contents. Values are chosen so
	// that accumulators stay comfortably inside their type's range even at
	// aluPeakMaxReps: the loop only ever adds, so a large operand product
	// times millions of reps would overflow (fp16 especially) and measure
	// infinity arithmetic instead of the intended instruction.
	operands func() []byte
	// opsPerRepPerChain is the arithmetic operations one rep of one
	// accumulator chain performs per *wave* — the scalar modes multiply
	// their per-lane count by the 64 lanes, while a coopMatMulAdd is a
	// single whole-subgroup operation.
	opsPerRepPerChain float64
	// chains is the ACC value baked into this variant's SPIR-V.
	chains int
	// outElemsPerGroup/outElemSize size the sink buffer that stops the
	// work from being dead-code-eliminated.
	outElemsPerGroup int
	outElemSize      int
	// coopMat, when set, is the A/C component-type pair whose native shape
	// this variant needs bound as specialization constants.
	coopMat *coopMatTypes
	// wave pins the subgroup size the variant was built for (-DWAVE); 0
	// leaves the driver's default, which is 64 here (KERNELS.md G0).
	wave uint32
}

type coopMatTypes struct {
	aType vk.ComponentType
	cType vk.ComponentType
}

// aluPeakOperands returns the fp32 operand buffer, also used by the clock
// warmer (which runs the same fp32 FMA kernel).
func aluPeakOperands() []float32 {
	out := make([]float32, aluPeakInputFloats)
	for i := range out {
		out[i] = 0.5
	}
	return out
}

func peakModes() []peakMode {
	return []peakMode{
		{
			name: "fma_fp32", spirv: shaders.ALUPeakFMA,
			operands:          func() []byte { return float32SliceToBytes(aluPeakOperands()) },
			opsPerRepPerChain: 2 * aluPeakLocalSize, // one fp32 fma per lane
			chains:            8,
			outElemsPerGroup:  aluPeakLocalSize, outElemSize: 4,
		},
		{
			name: "fma_pk_fp16", spirv: shaders.ALUPeakPackedFMA,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.001)) },
			opsPerRepPerChain: 4 * aluPeakLocalSize, // f16vec2: two fp16 fma per lane
			chains:            8,
			outElemsPerGroup:  aluPeakLocalSize, outElemSize: 4,
		},
		{
			name: "dot4_int8", spirv: shaders.ALUPeakDot4,
			operands:          func() []byte { return packedOnesBytes(aluPeakInputFloats) },
			opsPerRepPerChain: 8 * aluPeakLocalSize, // v_dot4_i32_i8: 4 multiplies + 4 adds per lane
			chains:            8,
			outElemsPerGroup:  aluPeakLocalSize, outElemSize: 4,
		},
		{
			name: "wmma_fp16", spirv: shaders.ALUPeakWMMAFP16,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16, // one 16x16x16 multiply-accumulate per subgroup
			chains:            4,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
		},
		// KERNELS.md G0: the same WMMA loop at 8 and 16 chains, and at
		// wave32, to tell the hardware's rate from the 4-chain probe's.
		{
			name: "wmma_fp16_acc8", spirv: shaders.ALUPeakWMMAFP16Acc8,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16,
			chains:            8,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
		},
		{
			name: "wmma_fp16_acc16", spirv: shaders.ALUPeakWMMAFP16Acc16,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16,
			chains:            16,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
		},
		{
			name: "wmma_fp16_w32", spirv: shaders.ALUPeakWMMAFP16W32,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16,
			chains:            4,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
			wave:    32,
		},
		{
			name: "wmma_fp16_w32_acc8", spirv: shaders.ALUPeakWMMAFP16W32Acc8,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16,
			chains:            8,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
			wave:    32,
		},
		{
			name: "wmma_int8", spirv: shaders.ALUPeakWMMAInt8,
			operands:          func() []byte { return onesInt8Bytes(aluPeakInputFloats) },
			opsPerRepPerChain: 2 * 16 * 16 * 16,
			chains:            4,
			outElemsPerGroup:  16 * 16, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentSInt8, cType: vk.ComponentSInt32},
		},
	}
}

// RunPeak measures the hardware's instruction-issue ceilings with kernels
// that touch no memory inside their loop (see shaders/alu_peak.comp). These
// are the denominators every other kernel's GFLOP/s should be read against:
// without them, "4200 GFLOP/s" can only be compared to a spec-sheet figure
// extrapolated from a different chip.
func RunPeak(dev *vk.Device, phys *vk.PhysicalDevice, warmup, iters uint32) ([]Result, error) {
	var results []Result
	modes := peakModes()
	if loaded, err := loadedPeakModes(); err != nil {
		return nil, err
	} else if len(loaded) > 0 {
		modes = loaded
	}
	for _, mode := range modes {
		var specConsts []vk.SpecConstant
		if mode.coopMat != nil {
			shape, ok, err := findCoopMatShape(phys, mode.coopMat.aType, mode.coopMat.cType)
			if err != nil {
				return nil, err
			}
			if !ok {
				fmt.Fprintf(os.Stderr, "peak %s: no matching cooperative-matrix shape reported, skipping\n", mode.name)
				continue
			}
			if shape.M != 16 || shape.N != 16 || shape.K != 16 {
				// opsPerRepPerChain above is hard-coded for 16x16x16; the
				// shape query has only ever reported that on this device,
				// but don't silently mis-scale if it changes.
				fmt.Fprintf(os.Stderr, "peak %s: shape %dx%dx%d is not the assumed 16x16x16, skipping\n",
					mode.name, shape.M, shape.N, shape.K)
				continue
			}
			specConsts = []vk.SpecConstant{
				{ID: 0, Value: uint32(shape.M)}, {ID: 1, Value: uint32(shape.N)}, {ID: 2, Value: uint32(shape.K)},
			}
		}

		mod, err := dev.NewShaderModule(mode.spirv)
		if err != nil {
			return nil, err
		}
		for _, waves := range peakWaveCounts {
			res, err := runPeakCase(dev, mod, mode, specConsts, waves, warmup, iters)
			if err != nil {
				mod.Destroy()
				return nil, fmt.Errorf("peak %s waves=%d: %w", mode.name, waves, err)
			}
			results = append(results, res)
		}
		mod.Destroy()
	}
	return results, nil
}

// loadedPeakModes is the screening hook (KERNELS.md G-o8): PEAK_SPV names
// one or more WMMA fp16 builds on disk, comma-separated, and the peak family
// runs those instead of its own table, so a kernel-shaped probe such as
// shaders/wmma_issue_probe.comp is timed by the same calibrated loop and
// read against the same 480 FLOP/clk/CU. The build's geometry comes with
// it: PEAK_MMAS is the 16x16x16 MMAs one subgroup issues per rep,
// PEAK_SUBGROUPS the subgroups a workgroup (each stores its own tile), and
// PEAK_WAVE pins the subgroup size (0 leaves the driver's 64).
func loadedPeakModes() ([]peakMode, error) {
	paths := os.Getenv("PEAK_SPV")
	if paths == "" {
		return nil, nil
	}
	envInt := func(name string, def int) (int, error) {
		v := os.Getenv(name)
		if v == "" {
			return def, nil
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return 0, fmt.Errorf("%s=%q: %w", name, v, err)
		}
		return n, nil
	}
	mmas, err := envInt("PEAK_MMAS", 1)
	if err != nil {
		return nil, err
	}
	subgroups, err := envInt("PEAK_SUBGROUPS", 1)
	if err != nil {
		return nil, err
	}
	wave, err := envInt("PEAK_WAVE", 0)
	if err != nil {
		return nil, err
	}
	var modes []peakMode
	for _, p := range strings.Split(paths, ",") {
		spirv, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		modes = append(modes, peakMode{
			name: strings.TrimSuffix(filepath.Base(p), ".spv"), spirv: spirv,
			operands:          func() []byte { return float32SliceToFloat16Bytes(constFloats(aluPeakInputFloats, 0.05)) },
			opsPerRepPerChain: 2 * 16 * 16 * 16 * float64(mmas) * float64(subgroups),
			chains:            1,
			outElemsPerGroup:  16 * 16 * subgroups, outElemSize: 4,
			coopMat: &coopMatTypes{aType: vk.ComponentFloat16, cType: vk.ComponentFloat32},
			wave:    uint32(wave),
		})
	}
	return modes, nil
}

func runPeakCase(dev *vk.Device, mod *vk.ShaderModule, mode peakMode, specConsts []vk.SpecConstant, waves int, warmup, iters uint32) (Result, error) {
	in, err := dev.NewBuffer(aluPeakInputFloats * 4) // 4 bytes/element is the widest any mode needs
	if err != nil {
		return Result{}, err
	}
	defer in.Destroy()
	in.WriteBytes(mode.operands())

	out, err := dev.NewBuffer(waves * mode.outElemsPerGroup * mode.outElemSize)
	if err != nil {
		return Result{}, err
	}
	defer out.Destroy()

	pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:              []*vk.Buffer{in, out},
		PushConstantSize:     4,
		SpecConstants:        specConsts,
		RequiredSubgroupSize: mode.wave,
	})
	if err != nil {
		return Result{}, err
	}
	defer pipe.Destroy()

	groups := uint32(waves)
	reps, err := calibratePeakReps(pipe, groups)
	if err != nil {
		return Result{}, err
	}

	pc := newPC().U32(reps).Bytes()
	ns, clocks, err := TimeDispatch(pipe, groups, 1, 1, warmup, iters, pc)
	if err != nil {
		return Result{}, err
	}

	ops := float64(waves) * float64(mode.chains) * float64(reps) * mode.opsPerRepPerChain
	return Result{
		Op: "peak", Variant: mode.name, Size: waves,
		NsPerIter: ns,
		GFLOPS:    ops / (ns / 1e9) / 1e9,
		Clocks:    clocks,
	}, nil
}

// calibratePeakReps estimates the per-rep cost, then picks a rep count
// putting one dispatch near aluPeakTargetNs. Going through TimeDispatch
// rather than DispatchTimed means the probe itself runs clock-warmed, so
// the estimate isn't taken at the idle clock and then overshot.
func calibratePeakReps(pipe *vk.ComputePipeline, groups uint32) (uint32, error) {
	probePC := newPC().U32(aluPeakProbeReps).Bytes()
	probeNs, _, err := TimeDispatch(pipe, groups, 1, 1, 1, 1, probePC)
	if err != nil {
		return 0, err
	}
	nsPerRep := probeNs / aluPeakProbeReps
	if nsPerRep <= 0 {
		return aluPeakProbeReps, nil
	}
	reps := aluPeakTargetNs / nsPerRep
	if reps < aluPeakProbeReps {
		reps = aluPeakProbeReps
	}
	if reps > aluPeakMaxReps {
		reps = aluPeakMaxReps
	}
	return uint32(reps), nil
}

// PrintPeakSummary reports, per instruction-issue ceiling, the best rate
// measured and what it implies per clock per CU — the form that can be
// compared against published RDNA3 figures (512 FLOP/clk/CU for fp16 WMMA,
// 1024 for int8) to check whether RDNA3.5 matches them.
func PrintPeakSummary(w io.Writer, results []Result, cus int) {
	best := map[string]Result{}
	var order []string
	for _, r := range results {
		if r.Op != "peak" {
			continue
		}
		cur, seen := best[r.Variant]
		if !seen {
			order = append(order, r.Variant)
		}
		if !seen || r.GFLOPS > cur.GFLOPS {
			best[r.Variant] = r
		}
	}
	if len(order) == 0 {
		return
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintf(tw, "CEILING\tBEST\tAT WAVES\tSCLK\tOPS/CLK/CU (%d CUs)\n", cus)
	for _, name := range order {
		r := best[name]
		perClkPerCU := "-"
		if r.Clocks.SclkMHz > 0 && cus > 0 {
			perClkPerCU = fmt.Sprintf("%.0f", r.GFLOPS*1e9/(r.Clocks.SclkMHz*1e6)/float64(cus))
		}
		sclk := "-"
		if r.Clocks.SclkMHz > 0 {
			sclk = fmt.Sprintf("%.0f MHz", r.Clocks.SclkMHz)
		}
		fmt.Fprintf(tw, "%s\t%.0f GOP/s\t%d\t%s\t%s\n", name, r.GFLOPS, r.Size, sclk, perClkPerCU)
	}
	tw.Flush()
}

func constFloats(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// packedOnesBytes builds uint32 words whose four int8 lanes are all 1, so
// each dotPacked4x8 contributes exactly 4 to its accumulator.
func packedOnesBytes(n int) []byte {
	out := make([]byte, n*4)
	for i := range out {
		out[i] = 1
	}
	return out
}

func onesInt8Bytes(n int) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = 1
	}
	return out
}
