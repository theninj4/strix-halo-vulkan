package dit

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// GPUBlock runs one whole Z-Image DiT block on the device: adaLN modulation,
// the four RMS norms, the seven projections, per-head q/k norms, rotary
// embedding, matrix-core attention, SwiGLU and both gated residuals, as one
// sequence of 26 dispatches over four arenas.
//
// The unit is deliberately the *block* and not the layer stack. A block is
// what the diffusers reference dumps stage by stage (reference/dump_dit_block.py),
// so it is the largest thing that can still be validated against something
// other than itself; 34 of them with their weights streamed is stage 4c's
// problem and needs nothing from this file but a loop.
//
// Everything the matrix cores touch is fp16 with fp32 accumulators, which is
// not a choice: it is the only operand type this device's cooperative
// matrices take. What that costs is measured in gpublock_test.go against the
// same fp32 reference the CPU implementation is held to.
type GPUBlock struct {
	dev *vk.Device

	// Four arenas, in the binding order every DiT shader declares:
	// 0 fp32 weights, 1 fp32 activations, 2 fp16 activations, 3 fp16 weights.
	// The split is by *lifetime and type*, not by tensor: an offset says
	// which tensor, so one pipeline layout serves the whole graph.
	wbuf *vk.Buffer
	abuf *vk.Buffer
	hbuf *vk.Buffer
	w16  *vk.Buffer

	pipes map[string]*vk.ComputePipeline
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

	attn    wmmaVariant
	plan    GEMMPlan
	kernels map[GEMMKernel]gemmVariant
	// cf16 is the fp16-C build of the FFN's projections, empty when the
	// planned kernel has no companion; see cf16For.
	cf16 GEMMKernel

	dim, ffn       int
	heads, headDim int
	tokens, tokPad int
	ldaDim, ldaFFN int
	adaIn          int
	normEps, qkEps float64

	// fp32 weight arena.
	wNorm1, wNorm2, wFFN1, wFFN2 uint32
	wNormQ, wNormK               uint32
	wCos, wSin                   uint32
	wAdaLN, wAdaLNBias           uint32

	// fp16 weight arena, one offset per projection.
	bOff map[Proj]uint32

	// fp32 activation arena.
	aX, aH, aQ, aK, aV, aCtx, aAttn uint32
	aGate, aUp, aFF                 uint32
	aMod, aAdaIn                    uint32
	actElems                        int

	// fp16 activation arena.
	hA, hQ, hK, hV, hCtx, hFFN uint32
	hGate, hUp                 uint32
	hElems                     int
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
}

// NewGPUBlock uploads one block's weights and builds its graph for a fixed
// sequence length. plan may be nil, which takes DefaultGEMMPlan.
func NewGPUBlock(dev *vk.Device, blk *Block, rope *RoPE, tokens int, plan GEMMPlan) (*GPUBlock, error) {
	return newGPUBlock(dev, blk, rope, tokens, plan, blockControls{})
}

func newGPUBlock(dev *vk.Device, blk *Block, rope *RoPE, tokens int, plan GEMMPlan, ctl blockControls) (*GPUBlock, error) {
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
	if blk.AdaLN == nil {
		return nil, fmt.Errorf("dit: block has no adaLN modulation")
	}
	g := &GPUBlock{
		dev:     dev,
		ctl:     ctl,
		pipes:   make(map[string]*vk.ComputePipeline),
		kernels: make(map[GEMMKernel]gemmVariant),
		plan:    plan,
		bOff:    make(map[Proj]uint32),
		dim:     blk.Dim,
		ffn:     blk.FFN.W1.Out,
		heads:   blk.Attn.Heads,
		headDim: blk.Attn.HeadDim,
		tokens:  tokens,
		adaIn:   blk.AdaLN.In,
		normEps: blk.AttnNorm1.Eps,
		qkEps:   blk.Attn.NormQ.Eps,
		Fused:   true,
		FP16FFN: true,
	}
	if g.headDim != wmmaHeadDim {
		return nil, fmt.Errorf("dit: head dim %d, but the attention kernel is built for %d", g.headDim, wmmaHeadDim)
	}
	// The tile is 16 and the largest workgroup tile is 128 rows, so the token
	// count is padded to 128 -- which is also what the attention kernel's key
	// blocks want, so one number serves both.
	g.tokPad = (tokens + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)
	g.ldaDim = g.dim + gemmPad
	g.ldaFFN = g.ffn + gemmPad

	if err := g.stageWeights(blk, rope); err != nil {
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
	return g, nil
}

// stageWeights fills the two weight arenas: fp32 for the things that are
// consumed elementwise (the five norms, the rotary table, the adaLN
// projection) and fp16 for the seven matrices the cooperative matrices read.
func (g *GPUBlock) stageWeights(blk *Block, rope *RoPE) error {
	var w []float32
	put := func(v []float32) uint32 {
		off := uint32(len(w))
		w = append(w, v...)
		return off
	}
	g.wNorm1 = put(blk.AttnNorm1.Weight)
	g.wNorm2 = put(blk.AttnNorm2.Weight)
	g.wFFN1 = put(blk.FFNNorm1.Weight)
	g.wFFN2 = put(blk.FFNNorm2.Weight)
	g.wNormQ = put(blk.Attn.NormQ.Weight)
	g.wNormK = put(blk.Attn.NormK.Weight)
	g.wCos = put(rope.Cos)
	g.wSin = put(rope.Sin)
	g.wAdaLN = put(blk.AdaLN.Weight)
	if blk.AdaLN.Bias == nil {
		return fmt.Errorf("dit: adaLN has no bias")
	}
	g.wAdaLNBias = put(blk.AdaLN.Bias)

	var err error
	if g.wbuf, err = g.dev.NewBuffer(len(w) * 4); err != nil {
		return fmt.Errorf("dit: fp32 weight arena: %w", err)
	}
	g.wbuf.WriteFloat32(w)

	// fp16 weights. Each projection is staged in the layout its kernel reads,
	// so a plan that names one kernel per shape pays for one copy of each.
	type staged struct {
		lin    *Linear
		layout int
		elems  int
	}
	plans := make(map[Proj]staged, len(projOrder))
	lins := map[Proj]*Linear{
		ProjQ: blk.Attn.Q, ProjK: blk.Attn.K, ProjV: blk.Attn.V, ProjO: blk.Attn.Out,
		ProjW1: blk.FFN.W1, ProjW3: blk.FFN.W3, ProjW2: blk.FFN.W2,
	}
	total := 0
	for _, r := range projOrder {
		lin := lins[r]
		if lin.Bias != nil {
			return fmt.Errorf("dit: %s has a bias; the projection GEMM has no bias path", r)
		}
		v, ok := variantFor(g.plan[r])
		if !ok {
			return fmt.Errorf("dit: no GEMM kernel %q (have %v)", g.plan[r], GEMMKernels())
		}
		layout := v.layout
		if g.ctl.wrongBLayout {
			layout = 0
		}
		// The buffer is sized for what the kernel will *read*, so a control
		// that stages the wrong layout still cannot run off the end of it.
		n := max(bElems(lin.Out, lin.In, v.layout), bElems(lin.Out, lin.In, layout))
		plans[r] = staged{lin: lin, layout: layout, elems: n}
		g.bOff[r] = uint32(total)
		total += n
	}
	if g.w16, err = g.dev.NewBuffer(total * 2); err != nil {
		return fmt.Errorf("dit: fp16 weight arena (%d MB): %w", (total*2)>>20, err)
	}
	for _, r := range projOrder {
		s := plans[r]
		buf := make([]uint16, s.elems)
		packB(buf, s.lin.Weight, s.lin.Out, s.lin.In, s.layout)
		g.w16.WriteUint16At(int(g.bOff[r]), buf)
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
func packB(dst []uint16, w []float32, n, k, layout int) {
	const tile = coopMatTile
	switch layout {
	case 0:
		ld := k + gemmPad
		for i := 0; i < n; i++ {
			row, out := w[i*k:(i+1)*k], dst[i*ld:]
			for j, v := range row {
				out[j] = safetensors.F32ToF16(v)
			}
		}
	case 1:
		// Transposed to [K, N]. Read sequentially, write strided: the other
		// way round is a strided read per output row, and this runs once.
		ld := n + gemmPad
		for i := 0; i < n; i++ {
			row := w[i*k : (i+1)*k]
			for j, v := range row {
				dst[j*ld+i] = safetensors.F32ToF16(v)
			}
		}
	default:
		// Fragment tiles: tile (nt, kt) is 256 contiguous halves holding
		// element (k, n) at (n%16)*16 + k%16, tiles ordered kt-fastest so
		// that one n-tile's whole K row is contiguous -- which is the order
		// the kernel walks and what makes a fragment load cover 512 B.
		kt := k / tile
		for i := 0; i < n; i++ {
			row := w[i*k : (i+1)*k]
			base := (i / tile) * kt * tile * tile
			lane := (i % tile) * tile
			for j, v := range row {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	}
}

// allocActivations lays out the two activation arenas. Every tensor is sized
// for the *padded* token count: the GEMM has no bounds check, so it writes
// whole 64- or 128-row tiles, and the rows past the sequence hold the
// products of the zeros the fp16 arena is initialised with.
func (g *GPUBlock) allocActivations() error {
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

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("dit: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("dit: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once, and never written again outside the sequence: the pad rows
	// and the pad columns of every A operand, and the pad tokens the
	// attention kernel's tail reads, all come from here.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	return nil
}

// build compiles every pipeline in the graph over all four arenas. They are
// bound to every pipeline, used or not, so that one descriptor layout and one
// push-constant size serve the whole sequence -- which is what
// vk.DispatchMultiTimed needs to record it into a single command buffer.
func (g *GPUBlock) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.w16}
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

	// One GEMM pipeline per distinct kernel in the plan. A multi-wave build
	// maps gl_SubgroupID onto its wave grid, so its subgroup size is pinned
	// rather than assumed: at wave32 the workgroup would hold twice the waves
	// the tiling expects and half of them would write outside it.
	for _, r := range projOrder {
		k := g.plan[r]
		if _, done := g.kernels[k]; done {
			continue
		}
		v, ok := variantFor(k)
		if !ok {
			return fmt.Errorf("dit: no GEMM kernel %q (have %v)", k, GEMMKernels())
		}
		spec := vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}
		if v.waves > 1 {
			if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
				return fmt.Errorf("dit: kernel %q needs a pinned 64-wide subgroup, which this device cannot do", k)
			}
			spec.RequiredSubgroupSize = 64
		}
		if err := g.pipeline(string(k), v.spirv, spec); err != nil {
			return err
		}
		g.kernels[k] = v
	}
	// The fp16-C companion, if the FFN's projections are on a kernel that has
	// one. Same geometry, same weight layout, a different store.
	if c, ok := cf16For[g.plan[ProjW1]]; ok && g.plan[ProjW3] == g.plan[ProjW1] {
		spec := vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}
		if c.waves > 1 {
			spec.RequiredSubgroupSize = 64
		}
		if err := g.pipeline(string(c.name), c.spirv, spec); err != nil {
			return err
		}
		g.kernels[c.name] = c
		g.cf16 = c.name
	}
	return nil
}

func (g *GPUBlock) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
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
func (g *GPUBlock) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.w16, g.hbuf, g.abuf, g.wbuf} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Apply runs the block over x [tokens, dim] with the timestep embedding
// adaln, and returns the new residual stream.
func (g *GPUBlock) Apply(x *Mat, adaln []float32) (*Mat, error) {
	if err := g.upload(x, adaln); err != nil {
		return nil, err
	}
	d, _, err := g.graph()
	if err != nil {
		return nil, err
	}
	// Batched submits, as in GPUAttention.Apply: a whole graph in one command
	// buffer can outlive the driver's reset watchdog, which is how the VAE
	// decoder first failed at 1024x1024 (PIPELINE.md).
	const perSubmit = 8
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("dit: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return g.Read(g.aX, g.dim), nil
}

// Profile runs the same graph one dispatch at a time and times each on the
// GPU. Wall clock around Apply is not a measurement of the block: it also
// carries the host write of x and the read-back of the result, and this
// arena's reads run at 0.2 GB/s (research/stage-3-dit-attention.md).
func (g *GPUBlock) Profile(x *Mat, adaln []float32) ([]Stage, *Mat, error) {
	if err := g.upload(x, adaln); err != nil {
		return nil, nil, err
	}
	d, kinds, err := g.graph()
	if err != nil {
		return nil, nil, err
	}
	stages := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true)
		if err != nil {
			return stages, nil, fmt.Errorf("dit: dispatch %d (%s): %w", i, kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: kinds[i], GPU: dur})
	}
	return stages, g.Read(g.aX, g.dim), nil
}

// Read copies one tensor out of the fp32 activation arena, by the offset the
// Tensor* constants name. It is how the stagewise validation reaches the
// intermediates; nothing in the pipeline reads a block's output back.
func (g *GPUBlock) Read(off uint32, cols int) *Mat {
	out := NewMat(g.tokens, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), g.tokens*cols))
	return out
}

// Labels lists the graph's dispatches in order, which is both what Profile
// reports and what RunTo accepts.
func (g *GPUBlock) Labels() []string {
	_, kinds, err := g.graph()
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
func (g *GPUBlock) RunTo(x *Mat, adaln []float32, label string) error {
	if err := g.upload(x, adaln); err != nil {
		return err
	}
	d, kinds, err := g.graph()
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
	const perSubmit = 8
	for i := 0; i < end; i += perSubmit {
		j := min(i+perSubmit, end)
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("dit: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

// ReadF16 copies a [tokens, cols] tensor out of the fp16 activation arena,
// widening it. The row stride is the operand's leading dimension, not cols:
// A's rows are padded (§2.3).
func (g *GPUBlock) ReadF16(off uint32, cols, lda int) *Mat {
	raw := g.hbuf.ReadUint16At(int(off), (g.tokens-1)*lda+cols)
	out := NewMat(g.tokens, cols)
	for r := 0; r < g.tokens; r++ {
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
func (g *GPUBlock) TensorX() uint32     { return g.aX }
func (g *GPUBlock) TensorNorm1() uint32 { return g.aH }
func (g *GPUBlock) TensorQ() uint32     { return g.aQ }
func (g *GPUBlock) TensorK() uint32     { return g.aK }
func (g *GPUBlock) TensorV() uint32     { return g.aV }
func (g *GPUBlock) TensorCtx() uint32   { return g.aCtx }
func (g *GPUBlock) TensorAttn() uint32  { return g.aAttn }
func (g *GPUBlock) TensorFF() uint32    { return g.aFF }
func (g *GPUBlock) TensorGate() uint32  { return g.aGate }

// TensorA is the fp16 A operand the projections read, and LDA its row
// stride: it holds the modulated block input, then the modulated FFN input.
func (g *GPUBlock) TensorA() uint32 { return g.hA }
func (g *GPUBlock) LDA() int        { return g.ldaDim }

// Mod reads the four modulation vectors back, already transformed: scale_msa,
// gate_msa, scale_mlp, gate_mlp, concatenated.
func (g *GPUBlock) Mod() []float32 { return g.abuf.ReadFloat32At(int(g.aMod), 4*g.dim) }

// Dim, FFN and Tokens describe the shapes a caller needs to read tensors.
func (g *GPUBlock) Dim() int    { return g.dim }
func (g *GPUBlock) FFN() int    { return g.ffn }
func (g *GPUBlock) Tokens() int { return g.tokens }

// Plan is the kernel chosen for each projection.
func (g *GPUBlock) Plan() GEMMPlan { return g.plan }

// FLOPs is the block's multiply-add count at this sequence length, counting
// the seven projections and both attention matmuls. adaLN and the
// elementwise passes are below the noise and are left out, as they are in
// the model-shape budget (§3.4).
func (g *GPUBlock) FLOPs() float64 {
	t := float64(g.tokens)
	dim, ffn := float64(g.dim), float64(g.ffn)
	proj := 2 * t * dim * dim * 4 // q, k, v, o
	ff := 2 * t * dim * ffn * 3   // w1, w3, w2
	attn := 4 * t * t * dim       // scores and context
	return proj + ff + attn
}

// upload writes the block's two inputs into the activation arena: the
// residual stream, which every dispatch after the first reads and the last
// two write, and the timestep embedding the modulation is projected from.
func (g *GPUBlock) upload(x *Mat, adaln []float32) error {
	if x.Rows != g.tokens || x.Cols != g.dim {
		return fmt.Errorf("dit: x is %s, want [%d %d]", x, g.tokens, g.dim)
	}
	if len(adaln) != g.adaIn {
		return fmt.Errorf("dit: adaln is %d wide, want %d", len(adaln), g.adaIn)
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	g.abuf.WriteFloat32At(int(g.aAdaIn), adaln)
	return nil
}

// graph builds the dispatch sequence, with a label per dispatch. Apply,
// Profile and RunTo share it so that what the profiler times is what Apply
// runs. It touches no memory, so asking for the labels costs nothing and
// disturbs nothing.
func (g *GPUBlock) graph() ([]vk.MultiDispatch, []string, error) {
	base := pushConstants{
		Tokens: uint32(g.tokens), Dim: uint32(g.dim),
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
		add("rmsnorm", kind, uint32(g.tokens), 1, pc)
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
		add("scale", kind, uint32(g.tokens), 1, pc)
	}
	// One projection: C[tokPad, n] = A[tokPad, k] * B[n, k]. kernel overrides
	// the plan's choice, which is how the FFN reaches the fp16-C companion.
	gemm := func(r Proj, kernel GEMMKernel, aOff, cOff uint32, n, k, lda int) error {
		if kernel == "" {
			kernel = g.plan[r]
		}
		v, ok := g.kernels[kernel]
		if !ok {
			return fmt.Errorf("dit: projection %s has no pipeline", r)
		}
		if n%v.bn != 0 || g.tokPad%v.bm != 0 {
			return fmt.Errorf("dit: %s tile %dx%d does not divide [%d %d]", r, v.bm, v.bn, g.tokPad, n)
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, g.bOff[r]
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(g.tokPad), uint32(n), uint32(k)
		pc.LDA, pc.LDB = uint32(lda), uint32(bLD(n, k, v.layout))
		add(string(v.name), "gemm "+string(r), uint32(n/v.bn), uint32(g.tokPad/v.bm), pc)
		return nil
	}
	// The gated residual: x += gate * y.
	gate := func(kind string, y, gateOff uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.Aux0 = y, g.aX, gateOff
		add("gateadd", kind, uint32(g.tokens), 1, pc)
	}
	// The three fused passes (stage 4b). Each replaces the two or three
	// dispatches above it and nothing else: same arithmetic, one round trip.
	normScale := func(kind string, in, out, w, scale uint32, lda int) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0, pc.Aux2 = in, out, w, scale, 1
		pc.LDA = uint32(lda)
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("normscale", kind, uint32(g.tokens), 1, pc)
	}
	normGate := func(kind string, y, w, gateOff uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = y, g.aX, w, gateOff
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("normgate", kind, uint32(g.tokens), 1, pc)
	}
	qkPack := func(kind string, in, out, w uint32, scale float32) {
		pc := base
		pc.InOff, pc.OutOff = in, out
		pc.WOff, pc.Aux0, pc.Aux2 = g.wCos, g.wSin, w
		pc.Aux1 = uint32(g.tokPad)
		pc.Eps = math.Float32bits(float32(g.qkEps))
		pc.Scale = math.Float32bits(scale)
		add("qkpack", kind, groups(g.tokPad, coopMatTile), uint32(g.heads), pc)
	}

	// ---- Modulation. One workgroup per output row of a [4*dim, adaIn]
	// projection, with the 1+x and tanh already applied.
	pcMod := base
	pcMod.InOff, pcMod.OutOff, pcMod.WOff, pcMod.Aux0 = g.aAdaIn, g.aMod, g.wAdaLN, g.wAdaLNBias
	pcMod.GemmN, pcMod.GemmK = uint32(4*g.dim), uint32(g.adaIn)
	add("adaln", "adaln", uint32(4*g.dim), 1, pcMod)

	scaleMSA, gateMSA := g.aMod, g.aMod+uint32(g.dim)
	scaleMLP, gateMLP := g.aMod+uint32(2*g.dim), g.aMod+uint32(3*g.dim)
	if g.ctl.swapModulation {
		scaleMSA, gateMSA = gateMSA, scaleMSA
		scaleMLP, gateMLP = gateMLP, scaleMLP
	}

	// ---- Attention.
	const log2e = 1.4426950408889634
	scale := float32(1 / math.Sqrt(float64(g.headDim)))
	if g.Fused {
		normScale("attn in", g.aX, g.hA, g.wNorm1, scaleMSA, g.ldaDim)
	} else {
		norm("rmsnorm x", g.aX, g.aH, g.wNorm1)
		narrow("attn in", g.aH, g.hA, g.dim, g.ldaDim, scaleMSA, true)
	}
	for _, s := range []struct {
		r   Proj
		out uint32
	}{{ProjQ, g.aQ}, {ProjK, g.aK}, {ProjV, g.aV}} {
		if err := gemm(s.r, "", g.hA, s.out, g.dim, g.dim, g.ldaDim); err != nil {
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
		qkPack("qkpack q", g.aQ, g.hQ, g.wNormQ, scale*log2e)
		qkPack("qkpack k", g.aK, g.hK, g.wNormK, 1)
	} else {
		for i, s := range []struct{ off, w uint32 }{{g.aQ, g.wNormQ}, {g.aK, g.wNormK}} {
			pc := base
			pc.InOff, pc.OutOff, pc.WOff = s.off, s.off, s.w
			pc.Span = uint32(g.headDim)
			pc.Eps = math.Float32bits(float32(g.qkEps))
			add("rmsnorm", []string{"rmsnorm q", "rmsnorm k"}[i], uint32(g.tokens*g.heads), 1, pc)
		}
		for i, off := range []uint32{g.aQ, g.aK} {
			pc := base
			pc.InOff, pc.OutOff = off, off
			pc.WOff, pc.Aux0 = g.wCos, g.wSin
			add("rope", []string{"rope q", "rope k"}[i], groups(g.tokens*g.heads*g.headDim/2, 256), 1, pc)
		}
		for i, s := range []struct {
			src, dst uint32
			scale    float32
		}{{g.aQ, g.hQ, scale * log2e}, {g.aK, g.hK, 1}} {
			pc := base
			pc.InOff, pc.OutOff = s.src, s.dst
			pc.Aux0, pc.Aux1 = 0, uint32(g.tokPad)
			pc.Scale = math.Float32bits(s.scale)
			add("pack", []string{"pack q", "pack k"}[i], groups(g.tokPad, coopMatTile), uint32(g.heads), pc)
		}
	}
	// v is transposed into its tiles (mode 1) and has no norm or rotation, so
	// it is the plain pack in both paths.
	pcV := base
	pcV.InOff, pcV.OutOff = g.aV, g.hV
	pcV.Aux0, pcV.Aux1 = 1, uint32(g.tokPad)
	pcV.Scale = math.Float32bits(1)
	add("pack", "pack v", groups(g.tokPad, coopMatTile), uint32(g.heads), pcV)
	pcAttn := base
	pcAttn.InOff, pcAttn.OutOff = g.hQ, g.aCtx
	pcAttn.KOff, pcAttn.VOff = g.hK, g.hV
	pcAttn.Aux1 = uint32(g.tokPad)
	add("attention", "attention", groups(g.tokens, g.attn.rows()), uint32(g.heads), pcAttn)

	narrow("narrow ctx", g.aCtx, g.hCtx, g.dim, g.ldaDim, 0, false)
	if err := gemm(ProjO, "", g.hCtx, g.aAttn, g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	if g.Fused && !g.ctl.dropNorm2 {
		normGate("gate msa", g.aAttn, g.wNorm2, gateMSA)
	} else {
		if !g.ctl.dropNorm2 {
			norm("rmsnorm attn", g.aAttn, g.aAttn, g.wNorm2)
		}
		gate("gate msa", g.aAttn, gateMSA)
	}

	// ---- Feed forward.
	if g.Fused {
		normScale("ffn in", g.aX, g.hA, g.wFFN1, scaleMLP, g.ldaDim)
	} else {
		norm("rmsnorm ffn", g.aX, g.aH, g.wFFN1)
		narrow("ffn in", g.aH, g.hA, g.dim, g.ldaDim, scaleMLP, true)
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
	add(gluPipe, "swiglu", uint32(g.tokens), 1, pcGLU)
	if err := gemm(ProjW2, "", g.hFFN, g.aFF, g.dim, g.ffn, g.ldaFFN); err != nil {
		return nil, nil, err
	}
	if g.Fused {
		normGate("gate mlp", g.aFF, g.wFFN2, gateMLP)
	} else {
		norm("rmsnorm ff", g.aFF, g.aFF, g.wFFN2)
		gate("gate mlp", g.aFF, gateMLP)
	}

	return d, kinds, nil
}

// ActivationBytes and WeightBytes are what one block costs on the device.
func (g *GPUBlock) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }
func (g *GPUBlock) WeightBytes() int     { return g.wbuf.Size() + g.w16.Size() }

// Elapsed sums a profile.
func Elapsed(stages []Stage) time.Duration {
	var t time.Duration
	for _, s := range stages {
		t += s.GPU
	}
	return t
}
