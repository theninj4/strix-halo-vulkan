package dit

import (
	"fmt"
	"math"
	"strings"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// GPUStack runs a stack of Z-Image DiT blocks on the device: per block, the
// adaLN modulation, the four RMS norms, the seven projections, per-head q/k
// norms, rotary embedding, matrix-core attention, SwiGLU and both gated
// residuals, as one sequence of 18 dispatches over shared arenas.
//
// One block is the unit of *validation* -- it is what the diffusers reference
// dumps stage by stage (reference/dump_dit_block.py) -- and GPUBlock is this
// type with a single block in it. The stack is the unit of *residency*, which
// is stage 4c's whole problem: the DiT's 34 blocks are 12.0 GB of fp16
// weights and a single Vulkan storage buffer on this device addresses 4.29 GB
// (research/stage-2-vae-decoder.md), so the projection weights live in
// several *banks* and a block names the bank it reads from. Nothing about the
// kernels changes; what changes is that a GEMM pipeline is built per bank,
// since the buffer a pipeline reads is in its descriptor set and not in its
// push constants.
//
// Everything else is shared across the stack and sized once: the two
// activation arenas, the fp32 weight arena (the norms, the rotary table and
// the adaLN projections of every block, 0.5 GB for all 34), and every
// dispatch that is not a GEMM. A block is then a row of offsets --
// blockWeights -- and the graph is the same code with a different one.
//
// Everything the matrix cores touch is fp16 with fp32 accumulators, which is
// not a choice: it is the only operand type this device's cooperative
// matrices take. What that costs is measured in gpublock_test.go against the
// same fp32 reference the CPU implementation is held to.
type GPUStack struct {
	dev *vk.Device

	// Four arenas, in the binding order every DiT shader declares:
	// 0 fp32 weights, 1 fp32 activations, 2 fp16 activations, 3 fp16 weights.
	// The split is by *lifetime and type*, not by tensor: an offset says
	// which tensor, so one pipeline layout serves the whole graph.
	//
	// Binding 3 is the only one that is not one buffer: banks[b] is bound to
	// the GEMM pipelines in gemms[b], and a block's bank field says which.
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	banks []*vk.Buffer

	pipes map[string]*vk.ComputePipeline
	// gemms[bank][kernel] is the GEMM build for that kernel reading that
	// bank. With one bank this is the same set of pipelines pipes would hold.
	gemms []map[GEMMKernel]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	// ctl is empty in every real use; see blockControls.
	ctl blockControls

	// Fused selects the stage-4b tail: one pass each for norm+scale+narrow,
	// norm+gate+residual, and the q/k norm + RoPE + fragment pack, instead of
	// the seven dispatches they replace. On by default; off is the graph
	// stage 4a validated, kept because each of those dispatches is a tensor
	// the diffusers dump can be compared against and because it is what the
	// fused path is checked against.
	//
	// It is *arithmetic-preserving*: the A operand it produces is bit-identical
	// to the unfused one, and what moves is only where a float is rounded.
	Fused bool

	// FP16FFN narrows the gate and up projections' output to fp16 (§2.6),
	// which halves both their store and SwiGLU's load -- 320 MB of the block's
	// traffic at 4096 tokens.
	//
	// A separate knob from Fused, because unlike the fusions it *is* a
	// precision trade: it rounds an intermediate that used to be fp32. What
	// that costs is measured in TestGPUBlockFP16FFN -- 1.8e-4 per element, and
	// the block's total error against diffusers goes 1.7e-2 to 1.8e-2, i.e.
	// it disappears next to what the fp16 GEMMs already cost. It applies only
	// where the values fit: w1 and w3 peak at 250 and 508 against fp16's
	// 65504, while w2's output reaches 6e5 and stays fp32.
	FP16FFN bool

	// FuseLayout removes the block's two pure-layout dispatches by making the
	// kernel that produces each tensor store it in the shape its consumer
	// wants (PIPELINE.md stage 10).
	//
	// `pack v` and `narrow ctx` computed nothing. The first read v's fp32 C
	// and wrote the same numbers back as fp16 fragment tiles; the second read
	// the fp32 context and wrote the same numbers back as the output
	// projection's fp16 A operand. Both are now epilogues -- dit_gemm.comp's
	// C_PACK and dit_attention_wmma.comp's OUT_F16 -- and the round trips they
	// cost are gone: at 4096 tokens, 94 MB of traffic per block per step for v
	// and the same again for the context.
	//
	// Neither changes an arithmetic operation, and the context epilogue
	// removes a rounding: v comes out bit-identical, while 84 halves in 1.2 M
	// of the context differ by one ulp because the old path rounded to fp32
	// and then to fp16 and this one rounds once. Every one of those 84 is an
	// exact fp16 tie, which is what TestGPUBlockFusedLayout asserts.
	//
	// Off is the graph stages 4-9 ran, and is also what the stagewise
	// validation needs: with the epilogues on, `v` and `attn_ctx` are never
	// materialised as fp32, and those are two of the tensors the diffusers
	// dump is walked against.
	FuseLayout bool

	// FFScale is the constant the SwiGLU output is multiplied by before it is
	// stored as w2's fp16 A operand, and it is a range fix rather than a
	// precision one.
	//
	// The gate and up projections peak around 276 and 256 on a real prompt,
	// so their product reaches 7.1e4 against fp16's 65504 and a few elements
	// in a few million overflow to infinity. One infinity anywhere in a row
	// of an A operand makes that whole row of the GEMM's output a NaN, and
	// the NaN is then in the residual stream, so a denoising loop turns the
	// image black two steps later. Random inputs never reach it -- stage 4b
	// measured the two halves at 250 and 508 and did not multiply them -- and
	// it is the first thing a real prompt does.
	//
	// It costs nothing to undo because it never has to be undone: w2's output
	// is read by an RMS norm and by nothing else, and an RMS norm is invariant
	// to a positive scale on its input. So the scale lives in one shader and
	// no consumer knows about it. 1/16 is a power of two -- exact in the
	// exponent, no mantissa lost -- and leaves 15x of headroom over the
	// measured peak.
	FFScale float64

	attn    wmmaVariant
	plan    GEMMPlan
	kernels map[GEMMKernel]gemmVariant
	// cf16 is the fp16-C build of the FFN's projections, empty when the
	// planned kernel has no companion; see cf16For.
	cf16 GEMMKernel
	// cpack is the fragment-tile-C build of the v projection, empty when the
	// planned kernel has no companion; see cpackFor.
	cpack GEMMKernel

	dim, ffn       int
	heads, headDim int
	// tokens is the sequence length the arenas were built for and tokPad its
	// rounding up to a workgroup tile. rows is the length of the run in
	// progress, which may be shorter: the refiners run on the caption stream
	// and the layers on the unified sequence, and both come out of one stack.
	tokens, tokPad int
	rows           int
	ldaDim, ldaFFN int
	adaIn          int
	normEps, qkEps float64

	// The rotary table, shared by every block in the fp32 weight arena.
	wCos, wSin uint32

	// One row per block, in the order the stack runs them.
	w []blockWeights

	// fp32 activation arena.
	aX, aH, aQ, aK, aV, aCtx, aAttn uint32
	aGate, aUp, aFF                 uint32
	aMod, aAdaIn                    uint32
	actElems                        int

	// fp16 activation arena.
	hA, hQ, hK, hV, hCtx, hFFN uint32
	hGate, hUp                 uint32
	hElems                     int

	// The head's slots (stage 9, gpuhead.go). They are in the stack's arenas
	// rather than in the head's own, because the patch embedder writes the
	// residual stream and the final layer reads it: the two are the same
	// tensor, so they have to be the same buffer, and a Vulkan binding is one
	// buffer. patchDim is zero on a config that does not describe a patch,
	// which is what the block tests build, and then none of this is allocated.
	patchDim                  int
	aAdaSiLU, aFinalMod, aOut uint32
	hPatch, hPatchLDA         uint32
}

// blockWeights is where one block's weights sit in the arenas. Everything a
// block does not share with its neighbours is here, which is why a 34-block
// stack costs one of these per block and one of everything else.
type blockWeights struct {
	// fp32 weight arena.
	norm1, norm2, ffn1, ffn2 uint32
	normQ, normK             uint32
	adaLN, adaLNBias         uint32

	// bank indexes GPUStack.banks and bOff is the offset of each projection
	// inside it, in halves.
	bank int
	bOff map[Proj]uint32

	// modulated is false for the two context_refiner blocks, which diffusers
	// builds with modulation=False: no adaLN projection, no scale on either
	// branch input, and both residuals ungated. It is not a variant of the
	// block so much as the block with four of its inputs missing, so the
	// graph drops five dispatches' worth of work rather than branching.
	modulated bool

	// prefix is the checkpoint name this came from, for error messages and
	// for Blocks().
	prefix string
}

// Proj names one of the block's seven projections. They differ in shape --
// three of them are the only shapes the whole DiT has -- and results/shapes.csv
// says the best kernel is not the same for all three, so the plan is per
// projection rather than per block.
type Proj string

const (
	ProjQ  Proj = "q"
	ProjK  Proj = "k"
	ProjV  Proj = "v"
	ProjO  Proj = "o"
	ProjW1 Proj = "w1"
	ProjW3 Proj = "w3"
	ProjW2 Proj = "w2"
)

// projOrder is every projection, in the order the graph issues them.
var projOrder = []Proj{ProjQ, ProjK, ProjV, ProjO, ProjW1, ProjW3, ProjW2}

// GEMMKernel names one build of shaders/dit_gemm.comp.
type GEMMKernel string

const (
	// GEMMReg64HKA4 is §2.7's crown kernel: one wave, a 64x64 accumulator
	// grid, a four-tile K-slab with A's loads hoisted, and the weight in its
	// natural [N, K] layout. Fastest on dit.qkv and dit.ff.w2 at M=4096.
	GEMMReg64HKA4 GEMMKernel = "reg64_hka4"
	// GEMMReg64HKA4Tiled is the same kernel reading a weight pre-packed as
	// 16x16 fragment tiles (stage 3c's finding, applied to B).
	GEMMReg64HKA4Tiled GEMMKernel = "reg64_hka4_bt16"
	// GEMMReg64Tiled is the tiled weight *without* the hoisted A slab, which
	// is the control for "does the tiling make the hoist unnecessary".
	GEMMReg64Tiled GEMMKernel = "reg64_bt16"
	// GEMMWG128x256 is four waves on a 128x256 workgroup tile with the weight
	// stored [K, N]. Fastest on dit.ff.w13 at M=4096, by 1.66x.
	GEMMWG128x256 GEMMKernel = "wg128x256"
	// GEMMWG128x256Tiled is that geometry against the tiled weight.
	GEMMWG128x256Tiled GEMMKernel = "wg128x256_bt16"
	// The swizzle arms (§2.4): the same geometry and layout, with the grid
	// walked in bands of SWZ columns instead of row by row, so that the
	// workgroups resident at one moment share B slabs rather than streaming
	// all of B past a single A slab.
	GEMMWG128x256SWZ4       GEMMKernel = "wg128x256_swz4"
	GEMMWG128x256SWZ8       GEMMKernel = "wg128x256_swz8"
	GEMMWG128x256TiledSWZ2  GEMMKernel = "wg128x256_bt16_swz2"
	GEMMWG128x256TiledSWZ4  GEMMKernel = "wg128x256_bt16_swz4"
	GEMMWG128x256TiledSWZ8  GEMMKernel = "wg128x256_bt16_swz8"
	GEMMWG128x256TiledSWZ16 GEMMKernel = "wg128x256_bt16_swz16"
)

// gemmVariant is one build's geometry. As in bench/ops_gemm_wmma.go and
// wmmaVariant above, the Go side has to be told it: BM, BN and the B layout
// size register arrays and address weights, so they are compiled in and
// cannot be read back out of the SPIR-V.
type gemmVariant struct {
	name  GEMMKernel
	spirv []byte
	bm    int // workgroup tile rows = WAVES_M * WM * 16
	bn    int // workgroup tile columns = WAVES_N * WN * 16
	waves int // waves per workgroup; > 1 pins the subgroup size
	// layout is how B has to be stored for this build: 0 is [N, K+pad],
	// 1 is [K, N], 2 is 16x16 fragment tiles.
	layout int
}

var gemmVariants = []gemmVariant{
	{name: GEMMReg64HKA4, spirv: shaders.DiTGEMMReg64HKA4, bm: 64, bn: 64, waves: 1, layout: 0},
	{name: GEMMReg64HKA4Tiled, spirv: shaders.DiTGEMMReg64HKA4Tiled, bm: 64, bn: 64, waves: 1, layout: 2},
	{name: GEMMReg64Tiled, spirv: shaders.DiTGEMMReg64Tiled, bm: 64, bn: 64, waves: 1, layout: 2},
	{name: GEMMWG128x256, spirv: shaders.DiTGEMMWG128x256, bm: 128, bn: 256, waves: 4, layout: 1},
	{name: GEMMWG128x256Tiled, spirv: shaders.DiTGEMMWG128x256Tiled, bm: 128, bn: 256, waves: 4, layout: 2},
	{name: GEMMWG128x256SWZ4, spirv: shaders.DiTGEMMWG128x256SWZ4, bm: 128, bn: 256, waves: 4, layout: 1},
	{name: GEMMWG128x256SWZ8, spirv: shaders.DiTGEMMWG128x256SWZ8, bm: 128, bn: 256, waves: 4, layout: 1},
	{name: GEMMWG128x256TiledSWZ2, spirv: shaders.DiTGEMMWG128x256TiledSWZ2, bm: 128, bn: 256, waves: 4, layout: 2},
	{name: GEMMWG128x256TiledSWZ4, spirv: shaders.DiTGEMMWG128x256TiledSWZ4, bm: 128, bn: 256, waves: 4, layout: 2},
	{name: GEMMWG128x256TiledSWZ8, spirv: shaders.DiTGEMMWG128x256TiledSWZ8, bm: 128, bn: 256, waves: 4, layout: 2},
	{name: GEMMWG128x256TiledSWZ16, spirv: shaders.DiTGEMMWG128x256TiledSWZ16, bm: 128, bn: 256, waves: 4, layout: 2},
}

// cf16For names the fp16-C companion build of a kernel, for the two
// projections whose consumer reads their output elementwise and narrows it
// anyway -- the FFN's gate and up, feeding SwiGLU (§2.6). Only the default
// kernel has one, and a plan naming any other keeps the fp32 C: this is a
// property of the *consumer*, not of the tiling, and `dit.ff.w2`'s output
// reaches 6e5 against fp16's 65504, so it is deliberately not a ladder rung
// that something could select by accident.
var cf16For = map[GEMMKernel]gemmVariant{
	GEMMWG128x256TiledSWZ8: {
		name: "wg128x256_bt16_swz8_cf16", spirv: shaders.DiTGEMMWG128x256TiledSWZ8CF16,
		bm: 128, bn: 256, waves: 4, layout: 2,
	},
}

// cpackFor names the fragment-tile-C companion build of a kernel, for the one
// projection whose consumer does not read a matrix at all: v, which attention
// reads as 16x16 tiles, per head, each transposed (dit_pack_f16.comp mode 1).
// Same argument as cf16For and the same shape of table -- a property of the
// *consumer*, so it is reached through a companion and not through a plan --
// and the same restriction: only the default kernel has one, because a build
// with a different tile geometry would need its own, and nothing else in the
// block wants a packed C.
var cpackFor = map[GEMMKernel]gemmVariant{
	GEMMWG128x256TiledSWZ8: {
		name: "wg128x256_bt16_swz8_cpack", spirv: shaders.DiTGEMMWG128x256TiledSWZ8CPack,
		bm: 128, bn: 256, waves: 4, layout: 2,
	},
}

// GEMMPlan chooses a kernel per projection. The weights are staged in the
// layout the chosen kernel reads, so the plan is fixed at construction: a
// different plan is a different upload, not a different push constant.
type GEMMPlan map[Proj]GEMMKernel

// DefaultGEMMPlan is the fastest kernel for each of the model's three shapes
// at 4096 tokens, measured by cmd/ditblock on this device
// (research/stage-4-dit-graph.md), not read off results/shapes.csv: that
// table was measured before a weight could be stored as fragment tiles or the
// grid walked in bands, and both change the answer.
//
//	qkv/o   N=3840  K=3840   42.2 TFLOP/s (76% of the ceiling)
//	ff.w13  N=10240 K=3840   40.2
//	ff.w2   N=3840  K=10240  41.3
//
// One kernel now wins all three, which is the point of the entry rather than
// an accident of it: the shapes disagreed only while the losing ones were
// bound by something other than their arithmetic. `wg128x256_bt16_swz8` is
// four waves on a 128x256 tile, a weight stored as 16x16 fragment tiles
// (§2.8), and the grid walked in bands of eight columns (§2.4).
func DefaultGEMMPlan() GEMMPlan { return UniformGEMMPlan(GEMMWG128x256TiledSWZ8) }

// UniformGEMMPlan runs every projection on one kernel, which is what the
// ladder in cmd/ditblock sweeps.
func UniformGEMMPlan(k GEMMKernel) GEMMPlan {
	p := GEMMPlan{}
	for _, r := range projOrder {
		p[r] = k
	}
	return p
}

// GEMMKernels lists every build, in ladder order.
func GEMMKernels() []GEMMKernel {
	out := make([]GEMMKernel, 0, len(gemmVariants))
	for _, v := range gemmVariants {
		out = append(out, v.name)
	}
	return out
}

// gemmPad is the leading-dimension pad, in halves, applied to A and to the
// [N, K] weight layout. §2.3: a K-strided fragment load issues 16 addresses
// one row stride apart inside a single instruction, and every extent in this
// model is a multiple of 256, so without the pad all 16 land in one DRAM
// channel. 128 halves is 256 B, the value the ablation peaked at.
const gemmPad = 128

// defaultFFScale is GPUStack.FFScale's value; see the field.
const defaultFFScale = 1.0 / 16

// blockControls are the deliberate breakages the negative control switches
// on. They live here rather than in the test because what they break is the
// *graph* -- which modulation vector a dispatch is handed, whether a dispatch
// is issued at all, which layout a weight was staged in -- and none of that is
// reachable from outside Apply.
type blockControls struct {
	// swapModulation hands the scale sites the gate vector and vice versa,
	// which is the mistake the checkpoint invites: adaLN's four chunks are
	// unlabelled and only their order says which is which.
	swapModulation bool
	// dropNorm2 omits the RMS norm on the attention output, i.e. loses one
	// dispatch out of twenty-six.
	dropNorm2 bool
	// wrongBLayout stages every weight in the [N, K+pad] layout whatever the
	// kernel reading it expects, so a tiled kernel reads a natural-layout
	// weight and a row-major one reads a column-major.
	wrongBLayout bool
	// shiftBank sends every block's projections to the next bank round-robin,
	// at the same offsets. It is stage 4c's own breakage: with one block per
	// bank it makes block i read block i+1's weights, which is exactly the
	// mistake a per-block descriptor set invites and which produces a
	// perfectly plausible tensor.
	shiftBank bool
}

// maxBankBytes is the largest storage buffer this device will allocate and
// address: maxStorageBufferRange is 4294967295 and maxMemoryAllocationSize
// 0xfffffffc, so 4.29 GB either way (research/stage-2-vae-decoder.md). The
// DiT's 34 blocks are 12.0 GB of fp16 projection weights, so they occupy
// three banks; a block is never split across two, since a projection's
// offset is a single uint32 into one bound buffer.
const maxBankBytes = 0xfffffffc

// blockSpec names one block of the stack: where it is in the checkpoint,
// whether it carries adaLN modulation, and -- for the single-block case the
// tests build from a fixture -- an already-loaded copy of it.
type blockSpec struct {
	prefix    string
	modulated bool
	blk       *Block
}

// load returns the block, reading it out of the checkpoint if it was not
// handed over already.
func (s blockSpec) load(set *safetensors.Set, cfg *Config) (*Block, error) {
	if s.blk != nil {
		return s.blk, nil
	}
	if set == nil {
		return nil, fmt.Errorf("dit: block %s has no checkpoint to load from", s.prefix)
	}
	return LoadBlock(set, s.prefix, cfg)
}

// StackBlocks is the DiT's 34 blocks in the order the transformer runs them:
// the two noise refiners over the image tokens, the two context refiners over
// the caption, and then the 30 layers over the two concatenated. The context
// refiners are the unmodulated ones -- diffusers builds them with
// modulation=False -- which is the only structural difference in the list.
//
// The order matters to a stack only through the weights it stages; which of
// them runs over which sequence is the caller's business, and Apply takes the
// selection.
func StackBlocks(cfg *Config) []blockSpec {
	out := make([]blockSpec, 0, 2*cfg.NRefiner+cfg.NLayers)
	for i := 0; i < cfg.NRefiner; i++ {
		out = append(out, specFor(fmt.Sprintf("noise_refiner.%d", i)))
	}
	for i := 0; i < cfg.NRefiner; i++ {
		out = append(out, specFor(fmt.Sprintf("context_refiner.%d", i)))
	}
	for i := 0; i < cfg.NLayers; i++ {
		out = append(out, specFor(fmt.Sprintf("layers.%d", i)))
	}
	return out
}

// specFor names one block by its checkpoint prefix, deciding from the prefix
// whether it is modulated. The context refiners are the DiT's only
// unmodulated blocks, and the rule lives here so that a stack assembled by
// hand -- a test's three layers, say -- gets the same answer as StackBlocks.
func specFor(prefix string) blockSpec {
	return blockSpec{prefix: prefix, modulated: !strings.HasPrefix(prefix, "context_refiner")}
}

// NewGPUStack builds the whole DiT -- every block StackBlocks names -- for a
// fixed maximum sequence length, reading the weights out of the checkpoint
// one block at a time. plan may be nil, which takes DefaultGEMMPlan.
func NewGPUStack(dev *vk.Device, set *safetensors.Set, cfg *Config, rope *RoPE, tokens int, plan GEMMPlan) (*GPUStack, error) {
	return newStack(dev, set, cfg, StackBlocks(cfg), rope, tokens, plan, blockControls{}, maxBankBytes)
}

// newStack is the constructor both a stack and a single block go through. set
// may be nil when every spec carries its own block, which is how the tests
// build one out of a fixture.
func newStack(dev *vk.Device, set *safetensors.Set, cfg *Config, specs []blockSpec, rope *RoPE, tokens int, plan GEMMPlan, ctl blockControls, bankBytes int) (*GPUStack, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("dit: a stack needs at least one block")
	}
	if plan == nil {
		plan = DefaultGEMMPlan()
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("dit: this device has no 16x16x16 fp16 cooperative matrix; the block graph needs one")
	}
	g := &GPUStack{
		dev:        dev,
		ctl:        ctl,
		pipes:      make(map[string]*vk.ComputePipeline),
		kernels:    make(map[GEMMKernel]gemmVariant),
		plan:       plan,
		tokens:     tokens,
		rows:       tokens,
		Fused:      true,
		FP16FFN:    true,
		FuseLayout: true,
		FFScale:    defaultFFScale,
	}
	// The shapes come from the first block, which is also what proves the
	// checkpoint is the model the config describes.
	first, err := specs[0].load(set, cfg)
	if err != nil {
		return nil, err
	}
	g.dim, g.ffn = first.Dim, first.FFN.W1.Out
	g.heads, g.headDim = first.Attn.Heads, first.Attn.HeadDim
	g.normEps, g.qkEps = first.AttnNorm1.Eps, first.Attn.NormQ.Eps
	if first.AdaLN != nil {
		g.adaIn = first.AdaLN.In
	} else if specs[0].modulated {
		return nil, fmt.Errorf("dit: block %s has no adaLN modulation", specs[0].prefix)
	}
	if g.headDim != wmmaHeadDim {
		return nil, fmt.Errorf("dit: head dim %d, but the attention kernel is built for %d", g.headDim, wmmaHeadDim)
	}
	// The tile is 16 and the largest workgroup tile is 128 rows, so the token
	// count is padded to 128 -- which is also what the attention kernel's key
	// blocks want, so one number serves both.
	g.tokPad = (tokens + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)
	// cfg is nil when a caller supplies its own block (NewGPUBlock does), and
	// then there is no patch to size the head's slots from -- which is right:
	// a single block has no head.
	if cfg != nil && len(cfg.PatchSize) > 0 && cfg.InChan > 0 {
		g.patchDim = cfg.PatchSize[0] * cfg.PatchSize[0] * cfg.InChan
	}
	g.ldaDim = g.dim + gemmPad
	g.ldaFFN = g.ffn + gemmPad

	if err := g.layoutWeights(specs, bankBytes); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.allocActivations(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stageWeights(set, cfg, specs, rope); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// planBanks assigns every block its offsets and its bank, and returns the
// size of each arena. It allocates nothing and reads no tensor: the sizes
// follow from the config, so the whole 12 GB layout is decided -- and can be
// checked -- before the first weight is touched.
func (g *GPUStack) planBanks(specs []blockSpec, bankBytes int) (total32 int, bankElems []int, err error) {
	// fp32: the rotary table once, then per block the five norms and -- only
	// if it is modulated -- the adaLN projection and its bias. The table is
	// cos and sin over headDim/2 pairs per token, since RoPE rotates adjacent
	// components as one complex number.
	ropeElems := 2 * g.tokens * (g.headDim / 2)
	total32 = ropeElems
	perBlock32 := func(modulated bool) int {
		n := 4*g.dim + 2*g.headDim
		if modulated {
			n += 4*g.dim*g.adaIn + 4*g.dim
		}
		return n
	}
	// fp16: the seven projections, in the layout their kernel reads.
	shapes := g.projShapes()
	perProj := make(map[Proj]int, len(projOrder))
	perBlock16 := 0
	for _, r := range projOrder {
		v, ok := variantFor(g.plan[r])
		if !ok {
			return 0, nil, fmt.Errorf("dit: no GEMM kernel %q (have %v)", g.plan[r], GEMMKernels())
		}
		n := bElems(shapes[r][0], shapes[r][1], v.layout)
		if g.ctl.wrongBLayout {
			// The control stages a weight in a layout its kernel does not
			// expect, and the [N, K+pad] one is the largest of the three, so
			// the bank is sized for it -- but only when the control is on:
			// the pad is 0.35 GB across the real stack's 34 blocks.
			n = max(n, bElems(shapes[r][0], shapes[r][1], 0))
		}
		perProj[r] = n
		perBlock16 += n
	}
	if perBlock16*2 > bankBytes {
		return 0, nil, fmt.Errorf("dit: one block's weights are %d MB and a bank holds %d MB",
			(perBlock16*2)>>20, bankBytes>>20)
	}

	g.w = make([]blockWeights, len(specs))
	cur := -1
	for i, s := range specs {
		if s.modulated && g.adaIn == 0 {
			return 0, nil, fmt.Errorf("dit: block %s is modulated but the stack has no adaLN shape", s.prefix)
		}
		w := blockWeights{modulated: s.modulated, prefix: s.prefix, bOff: make(map[Proj]uint32, len(projOrder))}
		w.norm1 = uint32(total32)
		w.norm2 = w.norm1 + uint32(g.dim)
		w.ffn1 = w.norm2 + uint32(g.dim)
		w.ffn2 = w.ffn1 + uint32(g.dim)
		w.normQ = w.ffn2 + uint32(g.dim)
		w.normK = w.normQ + uint32(g.headDim)
		if s.modulated {
			w.adaLN = w.normK + uint32(g.headDim)
			w.adaLNBias = w.adaLN + uint32(4*g.dim*g.adaIn)
		}
		total32 += perBlock32(s.modulated)

		// A new bank whenever this block would not fit the current one. They
		// come out uneven -- 12, 12, 10 for the real model -- and that is
		// fine: what has to hold is that a block is in exactly one.
		if cur < 0 || (bankElems[cur]+perBlock16)*2 > bankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
		w.bank = cur
		for _, r := range projOrder {
			w.bOff[r] = uint32(bankElems[cur])
			bankElems[cur] += perProj[r]
		}
		g.w[i] = w
	}
	return total32, bankElems, nil
}

// layoutWeights plans the arenas and allocates them.
func (g *GPUStack) layoutWeights(specs []blockSpec, bankBytes int) error {
	total32, bankElems, err := g.planBanks(specs, bankBytes)
	if err != nil {
		return err
	}
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("dit: fp32 weight arena (%d MB): %w", (total32*4)>>20, err)
	}
	g.wCos, g.wSin = 0, uint32(g.tokens*(g.headDim/2))
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("dit: fp16 weight bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}
	return nil
}

// projShapes is [out, in] for each of the seven projections, from the two
// widths the block has.
func (g *GPUStack) projShapes() map[Proj][2]int {
	return map[Proj][2]int{
		ProjQ: {g.dim, g.dim}, ProjK: {g.dim, g.dim}, ProjV: {g.dim, g.dim}, ProjO: {g.dim, g.dim},
		ProjW1: {g.ffn, g.dim}, ProjW3: {g.ffn, g.dim}, ProjW2: {g.dim, g.ffn},
	}
}

// stageWeights fills the arenas one block at a time: fp32 for the things that
// are consumed elementwise (the five norms, the rotary table, the adaLN
// projection) and fp16 for the seven matrices the cooperative matrices read.
//
// One block at a time is the point. LoadBlock materialises a block as fp32 on
// the host, which is 724 MB; all 34 at once would be 24.6 GB, and none of it
// is wanted once it has been narrowed into its bank.
func (g *GPUStack) stageWeights(set *safetensors.Set, cfg *Config, specs []blockSpec, rope *RoPE) error {
	if len(rope.Cos) != g.tokens*g.headDim/2 || len(rope.Sin) != len(rope.Cos) {
		return fmt.Errorf("dit: rotary table is %d long, want tokens*headDim/2 = %d",
			len(rope.Cos), g.tokens*g.headDim/2)
	}
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin)

	shapes := g.projShapes()
	for i, s := range specs {
		blk, err := s.load(set, cfg)
		if err != nil {
			return err
		}
		w := &g.w[i]
		if blk.Dim != g.dim || blk.FFN.W1.Out != g.ffn {
			return fmt.Errorf("dit: block %s is [%d %d], want [%d %d]",
				s.prefix, blk.Dim, blk.FFN.W1.Out, g.dim, g.ffn)
		}
		g.wbuf.WriteFloat32At(int(w.norm1), blk.AttnNorm1.Weight)
		g.wbuf.WriteFloat32At(int(w.norm2), blk.AttnNorm2.Weight)
		g.wbuf.WriteFloat32At(int(w.ffn1), blk.FFNNorm1.Weight)
		g.wbuf.WriteFloat32At(int(w.ffn2), blk.FFNNorm2.Weight)
		g.wbuf.WriteFloat32At(int(w.normQ), blk.Attn.NormQ.Weight)
		g.wbuf.WriteFloat32At(int(w.normK), blk.Attn.NormK.Weight)
		switch {
		case w.modulated && (blk.AdaLN == nil || blk.AdaLN.Bias == nil):
			return fmt.Errorf("dit: block %s has no adaLN weight and bias", s.prefix)
		case w.modulated:
			if blk.AdaLN.Out != 4*g.dim || blk.AdaLN.In != g.adaIn {
				return fmt.Errorf("dit: block %s adaLN is [%d %d], want [%d %d]",
					s.prefix, blk.AdaLN.Out, blk.AdaLN.In, 4*g.dim, g.adaIn)
			}
			g.wbuf.WriteFloat32At(int(w.adaLN), blk.AdaLN.Weight)
			g.wbuf.WriteFloat32At(int(w.adaLNBias), blk.AdaLN.Bias)
		case blk.AdaLN != nil:
			return fmt.Errorf("dit: block %s is listed unmodulated but carries adaLN weights", s.prefix)
		}

		lins := map[Proj]*Linear{
			ProjQ: blk.Attn.Q, ProjK: blk.Attn.K, ProjV: blk.Attn.V, ProjO: blk.Attn.Out,
			ProjW1: blk.FFN.W1, ProjW3: blk.FFN.W3, ProjW2: blk.FFN.W2,
		}
		bank := g.banks[w.bank]
		for _, r := range projOrder {
			lin := lins[r]
			if lin.Bias != nil {
				return fmt.Errorf("dit: %s %s has a bias; the projection GEMM has no bias path", s.prefix, r)
			}
			if lin.Out != shapes[r][0] || lin.In != shapes[r][1] {
				return fmt.Errorf("dit: %s %s is [%d %d], want %v", s.prefix, r, lin.Out, lin.In, shapes[r])
			}
			v, _ := variantFor(g.plan[r])
			layout := v.layout
			if g.ctl.wrongBLayout {
				layout = 0
			}
			buf := make([]uint16, bElems(lin.Out, lin.In, layout))
			packB(buf, lin.Weight, lin.Out, lin.In, layout)
			bank.WriteUint16At(int(w.bOff[r]), buf)
		}
	}
	return nil
}

// variantFor looks a kernel up in the ladder.
func variantFor(k GEMMKernel) (gemmVariant, bool) {
	for _, v := range gemmVariants {
		if v.name == k {
			return v, true
		}
	}
	return gemmVariant{}, false
}

// bElems is how many halves a weight [n, k] occupies in the given B layout.
func bElems(n, k, layout int) int {
	switch layout {
	case 0:
		return n * (k + gemmPad)
	case 1:
		return k * (n + gemmPad)
	default:
		return n * k
	}
}

// bLD is the leading dimension the shader is told, in halves. Both stored
// layouts are padded by gemmPad: §2.3 measured the pad worthless on a
// row-major B at N=4096, but on this model's dit.ff.w13 -- N=10240, so a
// 20 KB unpadded stride, five whole channel rotations -- it is a reproducible
// 1.035x, and it costs 0.25% of the arena. The tiled layout has no leading
// dimension at all (its addressing is (tile index) * 256), so it is passed
// zero, which nothing reads.
func bLD(n, k, layout int) int {
	switch layout {
	case 0:
		return k + gemmPad
	case 1:
		return n + gemmPad
	default:
		return 0
	}
}

// packB narrows a [n, k] row-major PyTorch weight into one of the three B
// layouts. This runs once per block at upload, which is what makes layout 2
// interesting at all: the retiling that costs attention a pass per step costs
// a weight nothing per step.
// The rows are independent in all three layouts, and there are 6.0e9 elements
// to narrow for the whole 34-block stack, so the output rows are split into
// chunks across the cores. 64 rows is enough that the scheduling is free and
// small enough that 3840 of them still fill 32 workers.
const packChunk = 64

func packB(dst []uint16, w []float32, n, k, layout int) {
	const tile = coopMatTile
	chunks := (n + packChunk - 1) / packChunk
	rows := func(c int, fn func(i int)) {
		for i := c * packChunk; i < min((c+1)*packChunk, n); i++ {
			fn(i)
		}
	}
	switch layout {
	case 0:
		ld := k + gemmPad
		parallelFor(chunks, func(c int) {
			rows(c, func(i int) {
				row, out := w[i*k:(i+1)*k], dst[i*ld:]
				for j, v := range row {
					out[j] = safetensors.F32ToF16(v)
				}
			})
		})
	case 1:
		// Transposed to [K, N]. Read sequentially, write strided: the other
		// way round is a strided read per output row, and this runs once.
		ld := n + gemmPad
		parallelFor(chunks, func(c int) {
			rows(c, func(i int) {
				row := w[i*k : (i+1)*k]
				for j, v := range row {
					dst[j*ld+i] = safetensors.F32ToF16(v)
				}
			})
		})
	default:
		// Fragment tiles: tile (nt, kt) is 256 contiguous halves holding
		// element (k, n) at (n%16)*16 + k%16, tiles ordered kt-fastest so
		// that one n-tile's whole K row is contiguous -- which is the order
		// the kernel walks and what makes a fragment load cover 512 B.
		kt := k / tile
		parallelFor(chunks, func(c int) {
			rows(c, func(i int) {
				row := w[i*k : (i+1)*k]
				base := (i / tile) * kt * tile * tile
				lane := (i % tile) * tile
				for j, v := range row {
					dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
				}
			})
		})
	}
}

// allocActivations lays out the two activation arenas, shared by every block
// in the stack. Every tensor is sized for the *padded* token count of the
// longest run the stack will take: the GEMM has no bounds check, so it writes
// whole 64- or 128-row tiles.
//
// What is in the rows past the sequence stopped mattering when runs became
// variable-length (blockGraph): the first run finds the zeros below, a later
// shorter one finds whatever a longer one left, and neither is read. A GEMM's
// rows are independent, and attention masks its key tail against pc.tokens
// rather than trusting the pad.
func (g *GPUStack) allocActivations() error {
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	rows := g.tokPad
	g.aX = alloc(rows * g.dim)
	g.aH = alloc(rows * g.dim)
	g.aQ = alloc(rows * g.dim)
	g.aK = alloc(rows * g.dim)
	g.aV = alloc(rows * g.dim)
	g.aCtx = alloc(rows * g.dim)
	g.aAttn = alloc(rows * g.dim)
	g.aGate = alloc(rows * g.ffn)
	g.aUp = alloc(rows * g.ffn)
	g.aFF = alloc(rows * g.dim)
	g.aMod = alloc(4 * g.dim)
	g.aAdaIn = alloc(g.adaIn)
	if g.patchDim > 0 {
		// The final layer's adaLN reads SiLU(t) where every block's reads t
		// itself -- diffusers applies the non-linearity in the final layer
		// and not in the blocks -- so the two forms are both resident and
		// SetAdaLN writes both.
		g.aAdaSiLU = alloc(g.adaIn)
		g.aFinalMod = alloc(g.dim)
		g.aOut = alloc(rows * g.patchDim)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	plane := g.heads * g.tokPad * g.headDim
	g.hA = halloc(rows * g.ldaDim)
	g.hQ = halloc(plane)
	g.hK = halloc(plane)
	g.hV = halloc(plane)
	g.hCtx = halloc(rows * g.ldaDim)
	g.hFFN = halloc(rows * g.ldaFFN)
	g.hGate = halloc(rows * g.ffn)
	g.hUp = halloc(rows * g.ffn)
	if g.patchDim > 0 {
		// The patch embedder's A operand: the patchified latent, its bias
		// column and its pad-token column, narrowed. headEmbedK is the padded
		// reduction extent and gemmPad the §2.3 stride pad.
		g.hPatchLDA = uint32(headEmbedK(g.patchDim) + gemmPad)
		g.hPatch = halloc(rows * int(g.hPatchLDA))
	}

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("dit: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("dit: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once, and never written again outside the sequence: the pad rows
	// and the pad columns of every A operand, and the pad tokens the
	// attention kernel's tail reads, all come from here. Stage 4a leaned on
	// that; a variable-length run no longer can (see allocActivations), so it
	// is now the initial state rather than an invariant.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	return nil
}

// build compiles every pipeline in the graph over all four arenas. They are
// bound to every pipeline, used or not, so that one descriptor layout and one
// push-constant size serve the whole sequence -- which is what
// vk.DispatchMultiTimed needs to record it into a single command buffer.
func (g *GPUStack) build() error {
	// Binding 3 is the fp16 weight bank. Only the GEMM builds read it, so
	// every other pipeline is built once over bank 0 and shared by the whole
	// stack; the GEMMs are built once per bank.
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[0]}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	simple := map[string][]byte{
		"adaln":     shaders.DiTAdaLN,
		"rmsnorm":   shaders.DiTRMSNorm,
		"rope":      shaders.DiTRoPE,
		"pack":      shaders.DiTPackF16,
		"scale":     shaders.DiTScaleF16,
		"swiglu":    shaders.DiTSwiGLUF16,
		"gateadd":   shaders.DiTGateAdd,
		"normscale": shaders.DiTNormScaleF16,
		"normgate":  shaders.DiTNormGateAdd,
		"qkpack":    shaders.DiTQKPack,
		"swiglu16":  shaders.DiTSwiGLUF16In16,
	}
	for name, spirv := range simple {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}

	// The attention kernel: the stage-3c default, or the first variant this
	// device can pin if the default's wave size is unavailable.
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("dit: subgroup size control: %w", err)
	}
	pinnable := func(wave uint32) bool {
		return wave == 0 || (feat.SubgroupSizeControl && sgs.Supported &&
			wave >= sgs.MinSubgroupSize && wave <= sgs.MaxSubgroupSize)
	}
	for _, v := range wmmaVariants {
		if v.name == DefaultKernel && pinnable(v.wave) {
			g.attn = v
			break
		}
	}
	if g.attn.spirv == nil {
		for _, v := range wmmaVariants {
			if pinnable(v.wave) {
				g.attn = v
				break
			}
		}
	}
	if g.attn.spirv == nil {
		return fmt.Errorf("dit: no runnable attention kernel")
	}
	if err := g.pipeline("attention", g.attn.spirv, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
	}); err != nil {
		return err
	}
	// Its fp16-context companion, where the chosen variant has one. Both are
	// built, because FuseLayout is a switch on a live graph and the unfused
	// arm is the oracle the fused one is checked against.
	if g.attn.of16 != nil {
		if err := g.pipeline("attention16", g.attn.of16, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
		}); err != nil {
			return err
		}
	}

	// The GEMM builds the plan names, plus the fp16-C companion if the FFN's
	// projections are on a kernel that has one (same geometry, same weight
	// layout, a different store).
	//
	// A multi-wave build maps gl_SubgroupID onto its wave grid, so its
	// subgroup size is pinned rather than assumed: at wave32 the workgroup
	// would hold twice the waves the tiling expects and half of them would
	// write outside it.
	for _, r := range projOrder {
		k := g.plan[r]
		if _, done := g.kernels[k]; done {
			continue
		}
		v, ok := variantFor(k)
		if !ok {
			return fmt.Errorf("dit: no GEMM kernel %q (have %v)", k, GEMMKernels())
		}
		g.kernels[k] = v
	}
	if c, ok := cf16For[g.plan[ProjW1]]; ok && g.plan[ProjW3] == g.plan[ProjW1] {
		g.kernels[c.name] = c
		g.cf16 = c.name
	}
	if c, ok := cpackFor[g.plan[ProjV]]; ok {
		g.kernels[c.name] = c
		g.cpack = c.name
	}
	// One set per bank. The buffer a GEMM reads its weight out of is in the
	// pipeline's descriptor set, not in its push constants, so a second bank
	// is a second pipeline -- which is the whole cost of splitting the arena:
	// three sets of two builds for the 34-block stack, and nothing at all in
	// the dispatch path, since vk.DispatchMultiTimed binds each dispatch's own
	// descriptor set anyway.
	g.gemms = make([]map[GEMMKernel]*vk.ComputePipeline, len(g.banks))
	for b := range g.banks {
		g.gemms[b] = make(map[GEMMKernel]*vk.ComputePipeline, len(g.kernels))
		bankBufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}
		for k, v := range g.kernels {
			spec := vk.PipelineSpec{Buffers: bankBufs, PushConstantSize: pcSize}
			if v.waves > 1 {
				if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
					return fmt.Errorf("dit: kernel %q needs a pinned 64-wide subgroup, which this device cannot do", k)
				}
				spec.RequiredSubgroupSize = 64
			}
			mod, err := g.dev.NewShaderModule(v.spirv)
			if err != nil {
				return fmt.Errorf("dit: shader %s: %w", k, err)
			}
			g.mods = append(g.mods, mod)
			pipe, err := g.dev.NewPipeline(mod, spec)
			if err != nil {
				return fmt.Errorf("dit: pipeline %s on bank %d: %w", k, b, err)
			}
			g.gemms[b][k] = pipe
		}
	}
	return nil
}

func (g *GPUStack) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("dit: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("dit: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// Destroy releases every Vulkan object.
func (g *GPUStack) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, set := range g.gemms {
		for _, p := range set {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range append([]*vk.Buffer{g.hbuf, g.abuf, g.wbuf}, g.banks...) {
		if b != nil {
			b.Destroy()
		}
	}
}

// perSubmit is how many dispatches go into one command buffer. A whole graph
// in one buffer can outlive the driver's reset watchdog, which is how the VAE
// decoder first failed at 1024x1024 (research/stage-2-vae-decoder.md), and a
// 34-block stack is 612 dispatches and 1.7 s of work -- well past it.
const perSubmit = 8

// submit runs a recorded dispatch sequence in watchdog-sized batches.
func submit(d []vk.MultiDispatch) error {
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("dit: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

// selection is the block indices to run: sel, or every block in order when
// sel is nil.
func (g *GPUStack) selection(sel []int) ([]int, error) {
	if sel == nil {
		sel = make([]int, len(g.w))
		for i := range sel {
			sel[i] = i
		}
		return sel, nil
	}
	for _, i := range sel {
		if i < 0 || i >= len(g.w) {
			return nil, fmt.Errorf("dit: block %d out of range, the stack has %d", i, len(g.w))
		}
	}
	return sel, nil
}

// Apply runs the selected blocks in order over x [rows, dim] with the
// timestep embedding adaln, and returns the residual stream they leave.
//
// The residual stream is one tensor for the whole run: a block reads and
// writes g.aX, so the stack costs no copy between blocks and the only thing
// that changes from one to the next is which weights its dispatches name.
// sel may be nil, which is every block in order.
func (g *GPUStack) Apply(x *Mat, adaln []float32, sel []int) (*Mat, error) {
	if err := g.upload(x, adaln); err != nil {
		return nil, err
	}
	if err := g.Run(sel); err != nil {
		return nil, err
	}
	return g.Read(g.aX, g.dim), nil
}

// Run is Apply without the upload or the read-back: it runs the selected
// blocks over whatever the residual stream already holds and leaves the
// result there.
//
// It is what a denoising step is made of. Apply's read-back is separate from
// it because the pipeline wants the result once per image rather than once
// per phase -- and, since stage 9, because it does not want it at all: the
// tail runs on the device (gpuhead.go) and what crosses the bus per step is
// the [tokens, 64] latent rather than the [tokens, 3840] stream.
//
// How much that read-back costs depends on something outside this file. A
// harness that has allocated little gets the device-local host-visible heap,
// where a host read runs at 0.18 GB/s and 63 MB is 344 ms (stage 3c); the
// pipeline, with 20.5 GB of weights resident, is past that heap's ~8 GB
// budget and reads the same arena at 15 GB/s, where 63 MB is 6.5 ms. See
// cmd/bus and research/stage-9-head-and-tail.md.
func (g *GPUStack) Run(sel []int) error {
	sel, err := g.selection(sel)
	if err != nil {
		return err
	}
	for _, i := range sel {
		d, _, err := g.blockGraph(i)
		if err != nil {
			return err
		}
		if err := submit(d); err != nil {
			return fmt.Errorf("dit: block %d (%s): %w", i, g.w[i].prefix, err)
		}
	}
	return nil
}

// Profile runs the same graph one dispatch at a time and times each on the
// GPU. Wall clock around Apply is not a measurement of the stack: it also
// carries the host write of x and the read-back of the result, and this
// arena's reads run at 0.2 GB/s (research/stage-3-dit-attention.md).
//
// The stages of every selected block are concatenated, so a 34-block profile
// is 34 copies of the same 18 labels; Stage.Block says which block a stage
// belongs to.
func (g *GPUStack) Profile(x *Mat, adaln []float32, sel []int) ([]Stage, *Mat, error) {
	sel, err := g.selection(sel)
	if err != nil {
		return nil, nil, err
	}
	if err := g.upload(x, adaln); err != nil {
		return nil, nil, err
	}
	var stages []Stage
	for _, b := range sel {
		d, kinds, err := g.blockGraph(b)
		if err != nil {
			return nil, nil, err
		}
		for i := range d {
			dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true)
			if err != nil {
				return stages, nil, fmt.Errorf("dit: block %d dispatch %d (%s): %w", b, i, kinds[i], err)
			}
			stages = append(stages, Stage{Index: len(stages), Block: b, Kind: kinds[i], GPU: dur})
		}
	}
	return stages, g.Read(g.aX, g.dim), nil
}

// SetRoPE replaces the rotary table, which is the one thing in the weight
// arena that belongs to a *run* rather than to a block.
//
// The transformer's three phases are three different sequences -- the noise
// refiners over the image tokens, the context refiners over the caption, the
// layers over the two concatenated -- and each has its own positional ids. The
// table is 4 MB against the stack's 12 GB, so it is rewritten between phases
// rather than held three times over.
func (g *GPUStack) SetRoPE(rope *RoPE) error {
	if len(rope.Cos) > g.tokens*g.headDim/2 || len(rope.Sin) != len(rope.Cos) {
		return fmt.Errorf("dit: rotary table is %d long, want at most tokens*headDim/2 = %d",
			len(rope.Cos), g.tokens*g.headDim/2)
	}
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin)
	return nil
}

// Blocks is the checkpoint prefix of every block in the stack, in run order.
func (g *GPUStack) Blocks() []string {
	out := make([]string, len(g.w))
	for i, w := range g.w {
		out[i] = w.prefix
	}
	return out
}

// Banks reports how the fp16 projection weights were split: one byte count
// per storage buffer, and one bank index per block.
func (g *GPUStack) Banks() (bytes []int, byBlock []int) {
	for _, b := range g.banks {
		bytes = append(bytes, b.Size())
	}
	byBlock = make([]int, len(g.w))
	for i, w := range g.w {
		byBlock[i] = w.bank
	}
	return bytes, byBlock
}

// Read copies one tensor out of the fp32 activation arena, by the offset the
// Tensor* constants name. It is how the stagewise validation reaches the
// intermediates; nothing in the pipeline reads a block's output back.
func (g *GPUStack) Read(off uint32, cols int) *Mat {
	out := NewMat(g.rows, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), g.rows*cols))
	return out
}

// Labels lists one block's dispatches in order, which is both what Profile
// reports and what RunTo accepts. Every block in the stack has the same
// labels except that an unmodulated one has no "adaln".
func (g *GPUStack) Labels(block int) []string {
	_, kinds, err := g.blockGraph(block)
	if err != nil {
		return nil
	}
	return kinds
}

// RunTo runs the graph up to and including the dispatch with the given label,
// and leaves the arenas as that dispatch left them.
//
// It is what makes the validation stagewise. Most of the block's
// intermediates are overwritten before the graph ends -- the two block norms
// share one scratch tensor, and four of the five norms run in place -- so a
// test that only ran Apply could compare nine of the reference's twenty-one
// tensors. Stopping the graph early instead costs one re-run of a prefix per
// stage, which at 320 tokens is a few milliseconds, and it is also the first
// thing to reach for when a block is wrong and it is not obvious which
// dispatch made it so.
func (g *GPUStack) RunTo(x *Mat, adaln []float32, block int, label string) error {
	if err := g.upload(x, adaln); err != nil {
		return err
	}
	d, kinds, err := g.blockGraph(block)
	if err != nil {
		return err
	}
	end := -1
	for i, k := range kinds {
		if k == label {
			end = i + 1
			break
		}
	}
	if end < 0 {
		return fmt.Errorf("dit: no dispatch labelled %q (have %v)", label, kinds)
	}
	return submit(d[:end])
}

// ReadF16 copies a [tokens, cols] tensor out of the fp16 activation arena,
// widening it. The row stride is the operand's leading dimension, not cols:
// A's rows are padded (§2.3).
func (g *GPUStack) ReadF16(off uint32, cols, lda int) *Mat {
	raw := g.hbuf.ReadUint16At(int(off), (g.rows-1)*lda+cols)
	out := NewMat(g.rows, cols)
	for r := 0; r < g.rows; r++ {
		row, src := out.Row(r), raw[r*lda:]
		for c := 0; c < cols; c++ {
			row[c] = safetensors.F16ToF32(src[c])
		}
	}
	return out
}

// The stage-boundary tensors, for the validation walk. These are the points
// the diffusers reference dumps, which is why the graph keeps them as
// separate tensors rather than fusing the norms into their consumers: a
// fused block tells you the output is wrong, these tell you which dispatch
// made it wrong.
func (g *GPUStack) TensorX() uint32     { return g.aX }
func (g *GPUStack) TensorNorm1() uint32 { return g.aH }
func (g *GPUStack) TensorQ() uint32     { return g.aQ }
func (g *GPUStack) TensorK() uint32     { return g.aK }
func (g *GPUStack) TensorV() uint32     { return g.aV }
func (g *GPUStack) TensorCtx() uint32   { return g.aCtx }
func (g *GPUStack) TensorAttn() uint32  { return g.aAttn }
func (g *GPUStack) TensorFF() uint32    { return g.aFF }
func (g *GPUStack) TensorGate() uint32  { return g.aGate }

// TensorHV is v's fragment-tile plane in the fp16 arena and TensorHCtx the
// output projection's fp16 A operand. They are what stage 10's epilogues
// write in place of the fp32 tensors above, so they are the points the fused
// and unfused graphs are compared at.
func (g *GPUStack) TensorHV() uint32   { return g.hV }
func (g *GPUStack) TensorHCtx() uint32 { return g.hCtx }

// PlaneElems is how many halves one of the packed q/k/v tensors holds -- all
// heads, every token tile the arena was built for. It is the extent
// ReadF16Flat needs to read one back, since a tiled plane has no row stride to
// hand ReadF16.
func (g *GPUStack) PlaneElems() int { return g.heads * g.tokPad * g.headDim }

// ReadF16Flat copies n halves out of the fp16 activation arena exactly as
// they lie, widening them. ReadF16 covers the operands that are [rows, lda];
// this covers the ones whose layout is 16x16 fragment tiles, where there is
// no row stride to state.
func (g *GPUStack) ReadF16Flat(off uint32, n int) []float32 {
	raw := g.hbuf.ReadUint16At(int(off), n)
	out := make([]float32, n)
	for i, h := range raw {
		out[i] = safetensors.F16ToF32(h)
	}
	return out
}

// TensorA is the fp16 A operand the projections read, and LDA its row
// stride: it holds the modulated block input, then the modulated FFN input.
func (g *GPUStack) TensorA() uint32 { return g.hA }
func (g *GPUStack) LDA() int        { return g.ldaDim }

// TensorHFFN is w2's fp16 A operand -- the SwiGLU output -- and LDAFFN its row
// stride. It is the one intermediate whose *range* has to be watched rather
// than its precision, which is what FFScale is for and what
// TestGPUBlockFFScale reads it to measure.
func (g *GPUStack) TensorHFFN() uint32 { return g.hFFN }
func (g *GPUStack) LDAFFN() int        { return g.ldaFFN }

// Mod reads the four modulation vectors back, already transformed: scale_msa,
// gate_msa, scale_mlp, gate_mlp, concatenated.
func (g *GPUStack) Mod() []float32 { return g.abuf.ReadFloat32At(int(g.aMod), 4*g.dim) }

// Dim, FFN and Tokens describe the shapes a caller needs to read tensors;
// Tokens is the length the arenas were built for, which is the longest run
// they will take. Len is how many blocks the stack holds.
func (g *GPUStack) Dim() int    { return g.dim }
func (g *GPUStack) FFN() int    { return g.ffn }
func (g *GPUStack) Tokens() int { return g.tokens }
func (g *GPUStack) Len() int    { return len(g.w) }

// Plan is the kernel chosen for each projection.
func (g *GPUStack) Plan() GEMMPlan { return g.plan }

// FLOPs is one block's multiply-add count at a sequence length, counting the
// seven projections and both attention matmuls. adaLN and the elementwise
// passes are below the noise and are left out, as they are in the model-shape
// budget (§3.4).
func (g *GPUStack) FLOPs(rows int) float64 {
	t := float64(rows)
	dim, ffn := float64(g.dim), float64(g.ffn)
	proj := 2 * t * dim * dim * 4 // q, k, v, o
	ff := 2 * t * dim * ffn * 3   // w1, w3, w2
	attn := 4 * t * t * dim       // scores and context
	return proj + ff + attn
}

// upload writes the run's two inputs into the activation arena: the residual
// stream, which every dispatch after the first reads and the last two write,
// and the timestep embedding every modulated block projects its own
// modulation from.
//
// x may be shorter than the length the arenas were built for -- the refiners
// run on the caption stream and the layers on the unified sequence -- and its
// row count is what the rest of the run uses.
func (g *GPUStack) upload(x *Mat, adaln []float32) error {
	if x.Cols != g.dim {
		return fmt.Errorf("dit: x is %s, want [rows %d]", x, g.dim)
	}
	if x.Rows <= 0 || x.Rows > g.tokens {
		return fmt.Errorf("dit: x has %d rows; the stack was built for at most %d", x.Rows, g.tokens)
	}
	if g.adaIn > 0 && len(adaln) != g.adaIn {
		return fmt.Errorf("dit: adaln is %d wide, want %d", len(adaln), g.adaIn)
	}
	g.rows = x.Rows
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	if g.adaIn > 0 {
		g.writeAdaLN(adaln)
	}
	return nil
}

// writeAdaLN puts the timestep embedding in the arena in both the forms the
// model asks for: as it is, which is what every block's adaLN projects, and
// through a SiLU, which is what the final layer's does (head.go's Final).
// The asymmetry is diffusers' and is worth one extra kilobyte per step rather
// than a shader that has to know which caller it has.
func (g *GPUStack) writeAdaLN(v []float32) {
	g.abuf.WriteFloat32At(int(g.aAdaIn), v)
	if g.patchDim > 0 {
		act := make([]float32, len(v))
		for i, x := range v {
			act[i] = x / (1 + float32(math.Exp(float64(-x))))
		}
		g.abuf.WriteFloat32At(int(g.aAdaSiLU), act)
	}
}

// Upload writes x into the residual stream starting at row `at`, and makes the
// run `rows` long.
//
// It is what the pipeline's three phases need and Apply cannot express: the
// caption stream is refined once per image at the front of the arena, and
// then written back *behind* the image stream every step, because the layers
// run over the two concatenated. Nothing copies -- the concatenation is the
// arena's own layout -- and the rows past `at+x.Rows` keep whatever they held,
// which is why `rows` is stated rather than inferred.
func (g *GPUStack) Upload(x *Mat, at, rows int) error {
	if x.Cols != g.dim {
		return fmt.Errorf("dit: x is %s, want [rows %d]", x, g.dim)
	}
	if at < 0 || x.Rows <= 0 || at+x.Rows > rows {
		return fmt.Errorf("dit: %s written at row %d does not fit in a %d-row run", x, at, rows)
	}
	if rows > g.tokens {
		return fmt.Errorf("dit: a %d-row run; the stack was built for at most %d", rows, g.tokens)
	}
	g.rows = rows
	g.abuf.WriteFloat32At(int(g.aX)+at*g.dim, x.Data)
	return nil
}

// SetAdaLN writes the timestep embedding every modulated block projects its
// own modulation from. It changes once per denoising step and nothing else in
// the arena does, which is why it is separate from the stream.
func (g *GPUStack) SetAdaLN(v []float32) error {
	if len(v) != g.adaIn {
		return fmt.Errorf("dit: adaln is %d wide, want %d", len(v), g.adaIn)
	}
	g.writeAdaLN(v)
	return nil
}

// Rows is the length of the run the arena currently holds.
func (g *GPUStack) Rows() int { return g.rows }

// blockGraph builds one block's dispatch sequence, with a label per dispatch.
// Apply, Profile and RunTo share it so that what the profiler times is what
// Apply runs, and the stack is the concatenation of one of these per block.
// It touches no memory, so asking for the labels costs nothing and disturbs
// nothing.
//
// The run's length is g.rows, which may be shorter than the length the arenas
// were built for: the rows past it hold whatever the last longer run left
// there, and nothing reads them. A GEMM covers whole tiles so it computes
// them, and attention masks its key tail against pc.tokens rather than
// trusting the pad to be zero, which is what makes a short run safe.
func (g *GPUStack) blockGraph(i int) ([]vk.MultiDispatch, []string, error) {
	w := &g.w[i]
	if w.bank >= len(g.gemms) {
		return nil, nil, fmt.Errorf("dit: block %d names bank %d of %d", i, w.bank, len(g.gemms))
	}
	bank := w.bank
	if g.ctl.shiftBank {
		bank = (bank + 1) % len(g.gemms)
	}
	tokPad := (g.rows + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)
	base := pushConstants{
		Tokens: uint32(g.rows), Dim: uint32(g.dim),
		Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// rmsnorm over a whole row, in or out of place.
	norm := func(kind string, in, out, w uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = in, out, w
		pc.Span = uint32(g.dim)
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("rmsnorm", kind, uint32(g.rows), 1, pc)
	}
	// Narrow an fp32 tensor into the fp16 arena, optionally scaling by a
	// modulation vector first.
	narrow := func(kind string, in, out uint32, width, lda int, scale uint32, scaled bool) {
		pc := base
		pc.InOff, pc.OutOff = in, out
		pc.Dim = uint32(width)
		pc.LDA = uint32(lda)
		pc.Aux0 = scale
		if scaled {
			pc.Aux2 = 1
		}
		add("scale", kind, uint32(g.rows), 1, pc)
	}
	// One projection: C[tokPad, n] = A[tokPad, k] * B[n, k]. kernel overrides
	// the plan's choice, which is how the FFN reaches the fp16-C companion and
	// v the fragment-tile one; packed says the store is the latter, whose
	// destination geometry is per-head planes of g.tokPad tokens rather than a
	// row stride.
	gemmTo := func(r Proj, kernel GEMMKernel, aOff, cOff uint32, n, k, lda int, packed bool) error {
		if kernel == "" {
			kernel = g.plan[r]
		}
		v, ok := g.kernels[kernel]
		if !ok {
			return fmt.Errorf("dit: projection %s has no pipeline", r)
		}
		pipe, ok := g.gemms[bank][kernel]
		if !ok {
			return fmt.Errorf("dit: kernel %s is not built for bank %d", kernel, bank)
		}
		if n%v.bn != 0 || tokPad%v.bm != 0 {
			return fmt.Errorf("dit: %s tile %dx%d does not divide [%d %d]", r, v.bm, v.bn, tokPad, n)
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, w.bOff[r]
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(tokPad), uint32(n), uint32(k)
		pc.LDA, pc.LDB = uint32(lda), uint32(bLD(n, k, v.layout))
		if packed {
			// The plane stride the packed tiles are addressed by, as in
			// qkPack below: the arena's token count, not the run's.
			pc.Aux1 = uint32(g.tokPad)
		}
		d = append(d, vk.MultiDispatch{
			Pipeline: pipe, GroupsX: uint32(n / v.bn), GroupsY: uint32(tokPad / v.bm), PushConstants: pc.bytes(),
		})
		kinds = append(kinds, "gemm "+string(r))
		return nil
	}
	gemm := func(r Proj, kernel GEMMKernel, aOff, cOff uint32, n, k, lda int) error {
		return gemmTo(r, kernel, aOff, cOff, n, k, lda, false)
	}
	// The gated residual: x += gate * y.
	gate := func(kind string, y, gateOff uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.Aux0 = y, g.aX, gateOff
		if w.modulated {
			pc.Aux2 = 1
		}
		add("gateadd", kind, uint32(g.rows), 1, pc)
	}
	// The three fused passes (stage 4b). Each replaces the two or three
	// dispatches above it and nothing else: same arithmetic, one round trip.
	normScale := func(kind string, in, out, wOff, scale uint32, lda int) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = in, out, wOff, scale
		if w.modulated {
			pc.Aux2 = 1
		}
		pc.LDA = uint32(lda)
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("normscale", kind, uint32(g.rows), 1, pc)
	}
	normGate := func(kind string, y, wOff, gateOff uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = y, g.aX, wOff, gateOff
		if w.modulated {
			pc.Aux2 = 1
		}
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("normgate", kind, uint32(g.rows), 1, pc)
	}
	qkPack := func(kind string, in, out, wOff uint32, scale float32) {
		pc := base
		pc.InOff, pc.OutOff = in, out
		pc.WOff, pc.Aux0, pc.Aux2 = g.wCos, g.wSin, wOff
		// The plane stride is the arena's token count, not the run's: the
		// packed q/k/v planes are laid out for the longest run the stack was
		// built for, and a short run writes a prefix of each.
		pc.Aux1 = uint32(g.tokPad)
		pc.Eps = math.Float32bits(float32(g.qkEps))
		pc.Scale = math.Float32bits(scale)
		add("qkpack", kind, groups(tokPad, coopMatTile), uint32(g.heads), pc)
	}

	// ---- Modulation. One workgroup per output row of a [4*dim, adaIn]
	// projection, with the 1+x and tanh already applied.
	//
	// An unmodulated block -- the two context refiners -- has none of this:
	// no projection to run and no vectors, so the four sites that would read
	// them are told not to (pc.aux2 = 0) and this dispatch is not issued.
	var scaleMSA, gateMSA, scaleMLP, gateMLP uint32
	if w.modulated {
		pcMod := base
		pcMod.InOff, pcMod.OutOff, pcMod.WOff, pcMod.Aux0 = g.aAdaIn, g.aMod, w.adaLN, w.adaLNBias
		pcMod.GemmN, pcMod.GemmK = uint32(4*g.dim), uint32(g.adaIn)
		add("adaln", "adaln", uint32(4*g.dim), 1, pcMod)

		scaleMSA, gateMSA = g.aMod, g.aMod+uint32(g.dim)
		scaleMLP, gateMLP = g.aMod+uint32(2*g.dim), g.aMod+uint32(3*g.dim)
		if g.ctl.swapModulation {
			scaleMSA, gateMSA = gateMSA, scaleMSA
			scaleMLP, gateMLP = gateMLP, scaleMLP
		}
	}

	// ---- Attention.
	const log2e = 1.4426950408889634
	scale := float32(1 / math.Sqrt(float64(g.headDim)))
	if g.Fused {
		normScale("attn in", g.aX, g.hA, w.norm1, scaleMSA, g.ldaDim)
	} else {
		norm("rmsnorm x", g.aX, g.aH, w.norm1)
		narrow("attn in", g.aH, g.hA, g.dim, g.ldaDim, scaleMSA, w.modulated)
	}
	// v's destination is the fragment-tile arena when the projection can
	// write it there itself (stage 10), and the fp32 C the pack reads
	// otherwise. q and k are not the same case: their fragments carry a norm
	// and a rotation between the GEMM and the pack, so something has to read
	// them back whatever the store does.
	vOut, vKernel, vPacked := g.aV, GEMMKernel(""), false
	if g.FuseLayout && g.cpack != "" {
		vOut, vKernel, vPacked = g.hV, g.cpack, true
	}
	for _, s := range []struct {
		r      Proj
		kernel GEMMKernel
		out    uint32
		packed bool
	}{{ProjQ, "", g.aQ, false}, {ProjK, "", g.aK, false}, {ProjV, vKernel, vOut, vPacked}} {
		if err := gemmTo(s.r, s.kernel, g.hA, s.out, g.dim, g.dim, g.ldaDim, s.packed); err != nil {
			return nil, nil, err
		}
	}
	// q and k: the per-head norm, the rotation, the softmax scale and the
	// fragment pack. Fused these are one pass each; unfused they are the
	// three separately validated dispatches stage 4a used, and the q/k norms
	// are *per head* -- the weight is headDim wide and a row holds every
	// head, so one span is one head.
	//
	// q carries 1/sqrt(headDim) and log2(e) with it either way, so the
	// attention kernel's exponential is exp2, one instruction on this ISA.
	if g.Fused {
		qkPack("qkpack q", g.aQ, g.hQ, w.normQ, scale*log2e)
		qkPack("qkpack k", g.aK, g.hK, w.normK, 1)
	} else {
		for i, s := range []struct{ off, w uint32 }{{g.aQ, w.normQ}, {g.aK, w.normK}} {
			pc := base
			pc.InOff, pc.OutOff, pc.WOff = s.off, s.off, s.w
			pc.Span = uint32(g.headDim)
			pc.Eps = math.Float32bits(float32(g.qkEps))
			add("rmsnorm", []string{"rmsnorm q", "rmsnorm k"}[i], uint32(g.rows*g.heads), 1, pc)
		}
		for i, off := range []uint32{g.aQ, g.aK} {
			pc := base
			pc.InOff, pc.OutOff = off, off
			pc.WOff, pc.Aux0 = g.wCos, g.wSin
			add("rope", []string{"rope q", "rope k"}[i], groups(g.rows*g.heads*g.headDim/2, 256), 1, pc)
		}
		for i, s := range []struct {
			src, dst uint32
			scale    float32
		}{{g.aQ, g.hQ, scale * log2e}, {g.aK, g.hK, 1}} {
			pc := base
			pc.InOff, pc.OutOff = s.src, s.dst
			pc.Aux0, pc.Aux1 = 0, uint32(g.tokPad)
			pc.Scale = math.Float32bits(s.scale)
			add("pack", []string{"pack q", "pack k"}[i], groups(tokPad, coopMatTile), uint32(g.heads), pc)
		}
	}
	// v is transposed into its tiles (mode 1) and has no norm or rotation, so
	// it is the plain pack -- unless the projection already wrote it there.
	if !vPacked {
		pcV := base
		pcV.InOff, pcV.OutOff = g.aV, g.hV
		pcV.Aux0, pcV.Aux1 = 1, uint32(g.tokPad)
		pcV.Scale = math.Float32bits(1)
		add("pack", "pack v", groups(tokPad, coopMatTile), uint32(g.heads), pcV)
	}
	// The context goes straight out as the output projection's fp16 A operand
	// where the kernel has that epilogue, and as fp32 followed by a narrowing
	// pass where it does not.
	attnPipe, ctxPacked := "attention", false
	if g.FuseLayout && g.attn.of16 != nil {
		attnPipe, ctxPacked = "attention16", true
	}
	pcAttn := base
	pcAttn.InOff, pcAttn.OutOff = g.hQ, g.aCtx
	pcAttn.KOff, pcAttn.VOff = g.hK, g.hV
	pcAttn.Aux1 = uint32(g.tokPad)
	if ctxPacked {
		pcAttn.OutOff, pcAttn.LDA = g.hCtx, uint32(g.ldaDim)
	}
	add(attnPipe, "attention", groups(g.rows, g.attn.rows()), uint32(g.heads), pcAttn)

	if !ctxPacked {
		narrow("narrow ctx", g.aCtx, g.hCtx, g.dim, g.ldaDim, 0, false)
	}
	if err := gemm(ProjO, "", g.hCtx, g.aAttn, g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	if g.Fused && !g.ctl.dropNorm2 {
		normGate("gate msa", g.aAttn, w.norm2, gateMSA)
	} else {
		if !g.ctl.dropNorm2 {
			norm("rmsnorm attn", g.aAttn, g.aAttn, w.norm2)
		}
		gate("gate msa", g.aAttn, gateMSA)
	}

	// ---- Feed forward.
	if g.Fused {
		normScale("ffn in", g.aX, g.hA, w.ffn1, scaleMLP, g.ldaDim)
	} else {
		norm("rmsnorm ffn", g.aX, g.aH, w.ffn1)
		narrow("ffn in", g.aH, g.hA, g.dim, g.ldaDim, scaleMLP, w.modulated)
	}
	// The gate and up projections, and SwiGLU. Fused, they write fp16 C
	// straight into the fp16 arena and SwiGLU reads it there (§2.6): the same
	// two 4096x10240 tensors at half the bytes, on both the write and the read.
	gluKernel, gluPipe := GEMMKernel(""), "swiglu"
	gateOut, upOut := g.aGate, g.aUp
	if g.FP16FFN && g.cf16 != "" {
		gluKernel, gluPipe = g.cf16, "swiglu16"
		gateOut, upOut = g.hGate, g.hUp
	}
	for _, s := range []struct {
		r   Proj
		out uint32
	}{{ProjW1, gateOut}, {ProjW3, upOut}} {
		if err := gemm(s.r, gluKernel, g.hA, s.out, g.ffn, g.dim, g.ldaDim); err != nil {
			return nil, nil, err
		}
	}
	pcGLU := base
	pcGLU.InOff, pcGLU.KOff, pcGLU.OutOff = gateOut, upOut, g.hFFN
	pcGLU.Dim = uint32(g.ffn)
	pcGLU.LDA = uint32(g.ldaFFN)
	pcGLU.Scale = math.Float32bits(float32(g.FFScale))
	add(gluPipe, "swiglu", uint32(g.rows), 1, pcGLU)
	if err := gemm(ProjW2, "", g.hFFN, g.aFF, g.dim, g.ffn, g.ldaFFN); err != nil {
		return nil, nil, err
	}
	if g.Fused {
		normGate("gate mlp", g.aFF, w.ffn2, gateMLP)
	} else {
		norm("rmsnorm ff", g.aFF, g.aFF, w.ffn2)
		gate("gate mlp", g.aFF, gateMLP)
	}

	return d, kinds, nil
}

// ActivationBytes is what the shared arenas cost on the device and
// WeightBytes what the whole stack's weights cost, fp32 arena and every fp16
// bank together.
func (g *GPUStack) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }
func (g *GPUStack) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

// Elapsed sums a profile.
func Elapsed(stages []Stage) time.Duration {
	var t time.Duration
	for _, s := range stages {
		t += s.GPU
	}
	return t
}
