package vae

import (
	"fmt"

	"strix-halo-vulkan/shaders"
)

// The two kernel choices Q9's ledger left open, and the one it did not —
// IMAGE.md Q9b.
//
// The ledger priced two ports off `TestGPUDecodeProfile`: conv3x3 at 69.7% of
// a 1024² decode and 3.2 TFLOP/s, and the mid block's four projections at
// 19.9% and 35 GFLOP/s. Both were priced as **z-image's matrix-core kernels,
// reused** — its implicit-GEMM convolution took the same operator from 3.18 s
// to 258 ms, and `dit_gemm` replaced the same naive linear.
//
// **That route is refused here, and the refusal is measured**
// (`TestConvFP16Ladder`). Both of z-image's kernels narrow their operands to
// fp16, and in *this* decoder a narrowed operand anywhere costs the decoded
// image **max abs 0.09** against the fp32 path's own 7.3e-4 — a hundred
// times the port's whole distance from the dump, and 23 of 255 8-bit levels.
// It is not a range problem and no exclusion rule fixes it: **one** narrowed
// convolution, conv_in, which is 0.006% of the decode's arithmetic, costs
// 0.0885 by itself. The mechanism is the tail norm. This decoder's residual
// stream runs at absmax 1e4–2.6e5 and `norm_out` divides a per-pixel L2 out
// of it, so a 5e-4 *relative* perturbation upstream arrives at a quiet pixel
// as an absolute one that pixel's own norm cannot absorb — measured at
// norm_out as rel 1.13 against a mean of 6.7e-5. z-image's decoder peaks at
// 497 and has a group norm; the inheritance was not a small extrapolation.
//
// So the arithmetic stays fp32, and what these kernels change is only the
// *shape* — which is where both of them were actually losing:
//
//   - the convolution is bound by activation loads, one per OC_BLOCK
//     multiply-adds, and OC_BLOCK had been 8 since the day it beat 1;
//   - the projections are bound by weight loads, because the naive kernel
//     gives one thread one output element and re-reads a whole [OC, K] weight
//     per row of A.
//
// Both replacements compute every accumulator in exactly the same order as
// the kernel they replace, so **the output is bit-identical** and every gate
// in vae_test.go and gpu_test.go holds unchanged. `TestGPUKernelScreen`
// asserts that rather than assuming it: a screen that only timed the arms
// could pick one that computes the wrong thing quickly.

// ConvKernel names a build of shaders/vae_conv2d.comp by its register block.
// The name is the output-channel block, which is what sets both the grid and
// the arithmetic intensity; where two arms share one, the input-channel block
// is in the name too.
type ConvKernel string

const (
	// ConvOC8 is the inherited kernel, unchanged: z-image's build, which
	// qimage ran uncontested through Q5g and Q9's first pass. It stays
	// selectable as the arm every other one is measured against.
	ConvOC8      ConvKernel = "oc8"
	ConvOC8IC8   ConvKernel = "oc8ic8"
	ConvOC16     ConvKernel = "oc16"
	ConvOC16IC8  ConvKernel = "oc16ic8"
	ConvOC32     ConvKernel = "oc32"
	ConvOC32IC32 ConvKernel = "oc32ic32"
	ConvOC48     ConvKernel = "oc48"
	ConvOC48IC16 ConvKernel = "oc48ic16"
	ConvOC64     ConvKernel = "oc64"
)

type convVariant struct {
	name    ConvKernel
	spirv   []byte
	ocBlock int
}

// The arms map both axes: OC_BLOCK at a fixed IC_BLOCK and back, because the
// first sweep moved them together and could not say which one paid.
var convVariants = []convVariant{
	{ConvOC8, shaders.VAEConv2D, 8},             // ic32, the inherited build
	{ConvOC8IC8, shaders.VAEConv2DOC8IC8, 8},    // ic8
	{ConvOC16, shaders.VAEConv2DOC16, 16},       // ic32
	{ConvOC16IC8, shaders.VAEConv2DOC16IC8, 16}, // ic8
	{ConvOC32, shaders.VAEConv2DOC32, 32},       // ic16
	{ConvOC32IC32, shaders.VAEConv2DOC32IC32, 32},
	{ConvOC48, shaders.VAEConv2DOC48, 48}, // ic8
	{ConvOC48IC16, shaders.VAEConv2DOC48IC16, 48},
	{ConvOC64, shaders.VAEConv2DOC64, 64}, // ic8
}

// DefaultConvKernel is TestGPUKernelScreen's winner at 1024²: conv3x3 goes
// **5.12 s -> 3.38 s, 3.3 -> 4.9 TFLOP/s**, for the same bits.
//
// The map is worth keeping, because it is not the one the argument above
// predicts. Raising OC_BLOCK alone buys *nothing* — at IC_BLOCK 32, 8 -> 16
// is 5.12 s -> 5.18 — and raising it with the input block left at 32 is
// actively worse: oc32ic32 is 6.85 s, the slowest arm in the screen. What
// pays is the **LDS footprint**, and OC_BLOCK only pays once the staged
// filter slab is small enough to keep waves resident: at OC_BLOCK 16, ic32 ->
// ic8 alone is 5.18 -> 3.83. So the winner is the arm with the largest
// register block over the smallest slab, and the ceiling it stops at (4.9 of
// the ~22.9 TFLOP/s fp32 FMA peak) is the one term neither block moves — the
// inner loop still spends one shared read per multiply-add. Falling below
// that needs a pixel block, which is priced in IMAGE.md and not taken here.
//
// 48 also happens to divide every output-channel count in the decoder
// (1152, 576, 288, 144) and all but one in the encoder, so the masked tail
// the store carries is unused; 64 divides two of them and measures behind.
const DefaultConvKernel = ConvOC48

// MidKernel names the kernel the mid block's four projections run on.
type MidKernel string

const (
	// MidLinear is shaders/vae_linear.comp, one thread per output element.
	// It stays selectable as the arm the tiled kernel is measured against,
	// and as the fallback shape: it needs nothing of the device.
	MidLinear MidKernel = "linear"
	// MidGEMM is shaders/qvae_gemm_f32.comp, a 64x64 tile with a 4x4 register
	// tile per thread.
	MidGEMM MidKernel = "gemm"
)

type midVariant struct {
	name  MidKernel
	spirv []byte
	// tiled is whether the grid is a tile grid rather than one thread per
	// output element. It is the only thing the host has to know.
	tiled bool
	bm    int
	bn    int
}

var midVariants = []midVariant{
	{MidLinear, shaders.VAELinear, false, 0, 0},
	{MidGEMM, shaders.QVAEGemmF32, true, 64, 64},
}

// DefaultMidKernel is TestGPUKernelScreen's winner at 1024², and it is the
// larger of the two by a long way: the four projections go **1.08–1.70 s to
// 9 ms, 0.04 -> 5.0 TFLOP/s, a factor of ~150**, for the same bits. The
// naive kernel was not slightly wrong about its shape; it was reading the
// whole weight once per row of A.
const DefaultMidKernel = MidGEMM

// Options names the graph's two screened kernels. The zero value is each
// screen's winner, which is what every caller outside the screen wants.
type Options struct {
	Conv ConvKernel
	Mid  MidKernel
}

// ConvKernels and MidKernels list the arms, in the order they are screened.
func ConvKernels() []ConvKernel {
	out := make([]ConvKernel, 0, len(convVariants))
	for _, v := range convVariants {
		out = append(out, v.name)
	}
	return out
}

func MidKernels() []MidKernel {
	out := make([]MidKernel, 0, len(midVariants))
	for _, v := range midVariants {
		out = append(out, v.name)
	}
	return out
}

// chooseKernels resolves Options onto the engine.
func (e *engine) chooseKernels(opt Options) error {
	conv := opt.Conv
	if conv == "" {
		conv = DefaultConvKernel
	}
	found := false
	for _, v := range convVariants {
		if v.name == conv {
			e.conv, found = v, true
		}
	}
	if !found {
		return fmt.Errorf("qvae: no conv kernel %q (have %v)", conv, ConvKernels())
	}
	mid := opt.Mid
	if mid == "" {
		mid = DefaultMidKernel
	}
	found = false
	for _, v := range midVariants {
		if v.name == mid {
			e.mid, found = v, true
		}
	}
	if !found {
		return fmt.Errorf("qvae: no mid kernel %q (have %v)", mid, MidKernels())
	}
	return nil
}

// Kernels names what a graph is running, for a screen's output.
func (e *engine) Kernels() (ConvKernel, MidKernel) { return e.conv.name, e.mid.name }

// kernelSet is a graph's base pipelines with the two chosen kernels folded
// in. The base set is a package-level map shared by every decoder on the
// process, so it is copied rather than written to.
//
// "conv2d8" is the ocBlock-8 build, always present: a convolution with fewer
// output channels than the chosen block would have most of its accumulators
// masked off at the store and its arithmetic wasted, and conv_out has three.
func (e *engine) kernelSet(base map[string][]byte) map[string][]byte {
	set := make(map[string][]byte, len(base)+2)
	for name, spirv := range base {
		set[name] = spirv
	}
	set["conv2d"] = e.conv.spirv
	set["conv2d8"] = shaders.VAEConv2D
	set["linear"] = e.mid.spirv
	return set
}

// convPipe picks the build for one convolution, and the output-channel block
// that sets its grid.
func (e *engine) convPipe(outC int) (string, int) {
	if outC < e.conv.ocBlock {
		return "conv2d8", 8
	}
	return "conv2d", e.conv.ocBlock
}
