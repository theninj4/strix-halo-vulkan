// Vulkan text encoder: PIPELINE.md stage 5c.
//
// The graph is the DiT's (zimage/dit/gpustack.go) with three things changed,
// and the reason it is a separate type rather than a parameter of that one is
// that all three are structural: attention is causal and grouped, the rotary
// convention is NeoX, and -- the one that decides the kernels -- the whole
// model sits on the *other* half of the roofline. At T tokens a projection
// here is T flop per byte of weight against this device's 235 crossover
// (research/3.4-model-shapes.md), so a prompt of 8-512 tokens is memory-bound
// where the DiT at 4096 is compute-bound, and the tile that wins there fills
// 24 of its 128 rows here.
//
// What is shared is everything that did not have to change: the four-arena
// binding layout and the push-constant block of shaders/dit_common.glsl, the
// projection GEMM, the RMS-norm-into-fp16 pass, SwiGLU, the fragment pack and
// the residual add. The GEMM builds this uses are narrow-M rungs of the same
// shader (shaders/shaders.go, stage 5c), and the attention kernel is the same
// file with CAUSAL and GQA compiled in.
package qwen

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// pushConstants mirrors the PC block in shaders/dit_common.glsl.
//
// It is a copy of zimage/dit's rather than a shared type on purpose and
// against the usual instinct: the layout is a contract with a *shader*, and
// this package reuses the DiT's compiled GEMM, so the copy is the thing that
// has to stay identical. Sharing the Go struct would make that look like one
// decision when it is two -- a change to the DiT's block would then silently
// be a change to the encoder's.
type pushConstants struct {
	InOff, OutOff, WOff uint32
	Tokens, Dim, Heads  uint32
	HeadDim, Span       uint32
	KOff, VOff, KStride uint32
	Eps, Scale          uint32
	Aux0, Aux1, Aux2    uint32
	BOff                uint32
	GemmM, GemmN, GemmK uint32
	LDA, LDB            uint32
}

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// coopMatTile is the cooperative-matrix extent, 16x16x16, which is the only
// shape this device reports.
const coopMatTile = 16

// gemmPad is the leading-dimension pad in halves, §2.3's 256 B, applied to
// every A operand here for the same reason it is in the DiT: a K-strided
// fragment load issues 16 addresses one row stride apart, and 2560, 4096 and
// 9728 are all multiples of 256.
const gemmPad = 128

// kernelLayout is the B layout every rung of this ladder reads: 16x16
// fragment tiles (§2.8). A weight is packed once at upload, so the coverage a
// fragment load gets is free here in a way it is not for an activation --
// which is why there is no row-major rung to choose between.
const kernelLayout = 2

// maxBankBytes is the largest storage buffer this device addresses. The
// encoder's 35 layers are 7.06 GB of fp16 weights, so they occupy two banks;
// stage 4c's machinery, unchanged, because a projection's offset is a single
// uint32 into one bound buffer.
const maxBankBytes = 0xfffffffc

// Proj names one of the seven projections a layer runs.
type Proj string

const (
	ProjQ    Proj = "q"
	ProjK    Proj = "k"
	ProjV    Proj = "v"
	ProjO    Proj = "o"
	ProjGate Proj = "gate"
	ProjUp   Proj = "up"
	ProjDown Proj = "down"
)

// projOrder is every projection, in the order the graph issues them.
var projOrder = []Proj{ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown}

// GEMMKernel names one build of shaders/dit_gemm.comp.
type GEMMKernel string

const (
	// The narrow-M rungs, which exist for this model: the grid has to be deep
	// enough to fill 40 CUs at a sequence of a few dozen rows.
	GEMMReg16x64   GEMMKernel = "reg16x64_bt16"
	GEMMReg16x256  GEMMKernel = "reg16x256_bt16"
	GEMMReg32x64   GEMMKernel = "reg32x64_bt16"
	GEMMReg32x128  GEMMKernel = "reg32x128_bt16"
	GEMMReg16x128  GEMMKernel = "reg16x128_bt16"
	GEMMReg16x64K8 GEMMKernel = "reg16x64_bt16_k8"
	GEMMWG16x256   GEMMKernel = "wg16x256_bt16"
	// The DiT's winner at 4096 tokens, kept as the ladder's control: it is
	// what "use the kernel we already have" would pick.
	GEMMWG128x256 GEMMKernel = "wg128x256_bt16_swz8"
	// The single-wave 64x64 tile, the other shape the DiT measured.
	GEMMReg64 GEMMKernel = "reg64_bt16"
)

// gemmVariant is one build's geometry, which the Go side has to be told
// because BM, BN and the B layout are compiled into the SPIR-V.
type gemmVariant struct {
	name   GEMMKernel
	spirv  []byte
	bm, bn int
	waves  int
	layout int // 2 everywhere here: the fragment-tiled weight (§2.8)
}

var gemmVariants = []gemmVariant{
	{name: GEMMReg16x64, spirv: shaders.DiTGEMMReg16x64Tiled, bm: 16, bn: 64, waves: 1, layout: 2},
	{name: GEMMReg16x256, spirv: shaders.DiTGEMMReg16x256Tiled, bm: 16, bn: 256, waves: 1, layout: 2},
	{name: GEMMReg32x64, spirv: shaders.DiTGEMMReg32x64Tiled, bm: 32, bn: 64, waves: 1, layout: 2},
	{name: GEMMReg32x128, spirv: shaders.DiTGEMMReg32x128Tiled, bm: 32, bn: 128, waves: 1, layout: 2},
	{name: GEMMReg16x128, spirv: shaders.DiTGEMMReg16x128Tiled, bm: 16, bn: 128, waves: 1, layout: 2},
	{name: GEMMReg16x64K8, spirv: shaders.DiTGEMMReg16x64TiledK8, bm: 16, bn: 64, waves: 1, layout: 2},
	{name: GEMMWG16x256, spirv: shaders.DiTGEMMWG16x256Tiled, bm: 16, bn: 256, waves: 2, layout: 2},
	{name: GEMMReg64, spirv: shaders.DiTGEMMReg64Tiled, bm: 64, bn: 64, waves: 1, layout: 2},
	{name: GEMMWG128x256, spirv: shaders.DiTGEMMWG128x256TiledSWZ8, bm: 128, bn: 256, waves: 4, layout: 2},
}

// GEMMPlan chooses a kernel per projection. As in the DiT the weights are
// staged in the layout the chosen kernel reads, so a different plan is a
// different upload rather than a different push constant -- though here every
// rung reads the same fragment-tiled layout, so the plan only moves the tile.
type GEMMPlan map[Proj]GEMMKernel

// DefaultGEMMPlan is the best *single* kernel over the whole prompt range,
// measured by `cmd/textenc -gpu -ladder` on this device
// (research/stage-5-text-encoder.md). It is never more than 1.11x off the
// rung that wins at a given length, where the DiT's own default is 1.24x off
// at a short prompt and the narrowest rung is 1.92x off at a long one.
//
// PlanFor picks the winner per length, which is what an encoder with
// AutoPlan set does; this is what it falls back to.
func DefaultGEMMPlan() GEMMPlan { return UniformGEMMPlan(GEMMReg64) }

// PlanFor is the measured schedule: which tile wins at a given sequence
// length. Unlike the DiT, where one kernel won all three shapes once the
// weight and the launch order were right, this model's winner moves with T
// -- because T is what its arithmetic intensity *is*, so the same weights are
// read under three different amounts of work.
//
//	tokens   winner                 runner-up
//	   16    reg16x64    1.00x      reg32x128   1.00x
//	   24    reg16x64    1.00x      reg32x128   1.00x
//	   64    reg32x128   1.00x      reg16x64    1.01x
//	  128    reg64       1.00x      wg128x256   1.12x
//	  256    wg128x256   1.00x      reg64       1.05x
//	  512    wg128x256   1.00x      reg64       1.11x
//
// The boundaries are placed between measured points rather than on them, and
// the cost of getting one wrong is small on the near side and 1.2-1.5x on the
// far side, which is why they sit closer to the shorter length.
func PlanFor(tokens int) GEMMPlan {
	switch {
	case tokens <= 96:
		return UniformGEMMPlan(GEMMReg32x128)
	case tokens <= 192:
		return UniformGEMMPlan(GEMMReg64)
	default:
		return UniformGEMMPlan(GEMMWG128x256)
	}
}

// UniformGEMMPlan runs every projection on one kernel, which is what the
// ladder sweeps.
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

func variantFor(k GEMMKernel) (gemmVariant, bool) {
	for _, v := range gemmVariants {
		if v.name == k {
			return v, true
		}
	}
	return gemmVariant{}, false
}

// attnVariant is one build of the causal grouped-query attention kernel.
type attnVariant struct {
	name  string
	spirv []byte
	qt    int
	ktil  int
	wave  uint32
}

func (v attnVariant) rows() int     { return v.qt * coopMatTile }
func (v attnVariant) keyBlock() int { return v.ktil * coopMatTile }

// attnVariants is the short ladder stage 3c's table already narrows: the
// encoder's attention is 1-4% of its work at any prompt length it will see,
// so this exists to be correct and to not be a surprise, not to be tuned.
var attnVariants = []attnVariant{
	{name: "qt1_kt4_w32", spirv: shaders.QwenAttentionQT1KT4W32, qt: 1, ktil: 4, wave: 32},
	{name: "qt1_kt4", spirv: shaders.QwenAttentionQT1KT4, qt: 1, ktil: 4},
	{name: "qt2_kt4", spirv: shaders.QwenAttentionQT2KT4, qt: 2, ktil: 4},
}

// controls are the deliberate breakages the negative control switches on.
// They live here rather than in the test because each one breaks the *graph*
// -- which shader a dispatch names, what span a norm is given, which bank a
// layer reads -- and none of it is reachable from outside Run.
type controls struct {
	// ropeAdjacent rotates with the DiT's convention instead of NeoX, by
	// naming the other shader. Both are built and both produce a plausible
	// tensor, which is the whole hazard.
	ropeAdjacent bool
	// noCausal uses the grouped-query build without the causal mask, i.e.
	// lets every token see the whole prompt.
	noCausal bool
	// qkNormWide norms q and k over the full width instead of per head.
	qkNormWide bool
	// wrongBLayout stages every weight row-major while the kernel reads
	// fragment tiles.
	wrongBLayout bool
	// shiftBank sends every layer's projections to the next bank at the same
	// offsets, so layer i reads layer i+1's weights -- the mistake a
	// per-bank descriptor set invites, and one that produces a perfectly
	// plausible tensor.
	shiftBank bool
}

// layerWeights is where one layer's weights sit in the arenas.
type layerWeights struct {
	// fp32 arena: the four norms.
	attnNorm, ffnNorm uint32
	qNorm, kNorm      uint32
	// fp16 bank and the offset of each projection inside it, in halves.
	bank int
	bOff map[Proj]uint32
}

// GPUEncoder runs Qwen3 layers on the device. One instance holds every layer
// the pipeline needs -- 7.06 GB across two storage buffers -- so a prompt is
// a dispatch sequence and not a load.
type GPUEncoder struct {
	dev *vk.Device
	cfg *Config

	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	banks []*vk.Buffer

	pipes map[string]*vk.ComputePipeline
	gemms []map[GEMMKernel]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	// AutoPlan re-plans every run for its own length, using PlanFor. It is on
	// by default and it is free: every rung reads the same staged weight, so
	// this changes which pipeline a dispatch names and nothing else. Set it
	// false to hold a plan across lengths, which is what the ladder does.
	AutoPlan bool

	plan    GEMMPlan
	kernels map[GEMMKernel]gemmVariant
	attn    attnVariant
	ctl     controls

	// tokens is the longest run the arenas were built for and arenaRows its
	// rounding up to the largest tile *any* kernel covers, so that every plan
	// fits the arenas that were allocated once. rows is the length of the run
	// in progress and align the current plan's tile, which is what a run is
	// actually padded to -- the whole point of a narrow-M rung is that it
	// pads 24 tokens to 32 rows and not to 128.
	tokens, arenaRows, rows int
	align                   int
	// stagedLayout is how the weights were written; kernelLayout is what
	// every rung reads. They differ only under the wrongBLayout control,
	// which is the whole point of it.
	stagedLayout         int
	ldaDim, ldaQ, ldaFFN int

	embedRows int
	embed     []float32

	wCos, wSin uint32
	w          []layerWeights

	// fp32 activation arena.
	aX, aQ, aK, aV, aCtx, aAttn uint32
	aGate, aUp, aFF             uint32
	actElems                    int

	// fp16 activation arena.
	hA, hQ, hK, hV, hCtx, hFFN uint32
	hElems                     int
}

// NewGPUEncoder builds the encoder: layers decoder layers out of set, sized
// for at most maxTokens. plan may be nil for DefaultGEMMPlan.
//
// set is read during construction only -- the embedding table is copied to
// the host as fp32 (1.56 GB) and every projection is narrowed into a bank --
// so the caller may close it afterwards, as the DiT's stack allows.
func NewGPUEncoder(dev *vk.Device, set *safetensors.Set, cfg *Config, layers, maxTokens int, plan GEMMPlan) (*GPUEncoder, error) {
	return newEncoder(dev, set, cfg, layers, maxTokens, plan, controls{}, maxBankBytes)
}

func newEncoder(dev *vk.Device, set *safetensors.Set, cfg *Config, layers, maxTokens int, plan GEMMPlan, ctl controls, bankBytes int) (*GPUEncoder, error) {
	explicitPlan := plan
	if layers <= 0 || layers > cfg.NumLayers {
		return nil, fmt.Errorf("qwen: asked for %d of %d layers", layers, cfg.NumLayers)
	}
	if maxTokens <= 0 {
		return nil, fmt.Errorf("qwen: maxTokens is %d", maxTokens)
	}
	if plan == nil {
		plan = DefaultGEMMPlan()
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("qwen: this device has no 16x16x16 fp16 cooperative matrix; the encoder graph needs one")
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("qwen: %d heads do not divide into %d kv heads", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim != headDim {
		return nil, fmt.Errorf("qwen: head dim %d, but the kernels are built for %d", cfg.HeadDim, headDim)
	}
	g := &GPUEncoder{
		dev:     dev,
		cfg:     cfg,
		ctl:     ctl,
		plan:    plan,
		pipes:   make(map[string]*vk.ComputePipeline),
		kernels: make(map[GEMMKernel]gemmVariant),
		tokens:  maxTokens,
		rows:    maxTokens,
		ldaDim:  cfg.HiddenSize + gemmPad,
		ldaQ:    cfg.NumHeads*cfg.HeadDim + gemmPad,
		ldaFFN:  cfg.IntermediateSize + gemmPad,
	}
	if err := g.chooseAttention(); err != nil {
		g.Destroy()
		return nil, err
	}
	// The run length is padded to the largest tile any dispatch covers: the
	// GEMM's BM, and the attention kernel's key block, which reads whole
	// blocks of the packed planes. One number serves both.
	//
	// Every rung is built and every rung reads the same fragment-tiled
	// weight, so the plan can be changed after construction (SetPlan) without
	// restaging 7 GB -- which is what makes the ladder in cmd/textenc one
	// load instead of seven. The arenas are therefore sized for the widest
	// tile in the table rather than for the plan.
	g.stagedLayout = 2
	if ctl.wrongBLayout {
		g.stagedLayout = 0
	}
	arenaAlign := g.attn.keyBlock()
	for _, v := range gemmVariants {
		if v.layout == kernelLayout {
			g.kernels[v.name] = v
			arenaAlign = max(arenaAlign, v.bm)
		}
	}
	if err := g.SetPlan(plan); err != nil {
		g.Destroy()
		return nil, err
	}
	// A caller that named a plan meant it; only the default schedules.
	g.AutoPlan = explicitPlan == nil
	g.arenaRows = roundUp(maxTokens, arenaAlign)

	if err := g.layoutWeights(layers, bankBytes); err != nil {
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
	if err := g.stageWeights(set, layers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// headDim is compiled into the pack and attention shaders.
const headDim = 128

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// canWMMA reports whether this device can run the matrix-core kernels at all.
// A device reporting some other cooperative-matrix shape would run these
// shaders *wrong* rather than slowly, so the answer is "no" rather than "try".
func canWMMA(dev *vk.Device) (bool, error) {
	feat := dev.Features()
	if !feat.Float16 || !feat.CoopMatrix {
		return false, nil
	}
	shapes, err := dev.Physical().CooperativeMatrixShapes()
	if err != nil {
		return false, fmt.Errorf("qwen: cooperative-matrix shapes: %w", err)
	}
	for _, sh := range shapes {
		if sh.Scope == vk.ScopeSubgroup && sh.M == coopMatTile && sh.N == coopMatTile && sh.K == coopMatTile &&
			sh.AType == vk.ComponentFloat16 && sh.BType == vk.ComponentFloat16 &&
			sh.CType == vk.ComponentFloat32 && sh.ResultType == vk.ComponentFloat32 {
			return true, nil
		}
	}
	return false, nil
}

// chooseAttention picks the first variant whose wave size this device can
// pin, in ladder order.
func (g *GPUEncoder) chooseAttention() error {
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("qwen: subgroup size control: %w", err)
	}
	for _, v := range attnVariants {
		if v.wave == 0 || (feat.SubgroupSizeControl && sgs.Supported &&
			v.wave >= sgs.MinSubgroupSize && v.wave <= sgs.MaxSubgroupSize) {
			g.attn = v
			return nil
		}
	}
	return fmt.Errorf("qwen: no runnable attention kernel")
}

// projShapes is [out, in] for each projection.
func (g *GPUEncoder) projShapes() map[Proj][2]int {
	c := g.cfg
	q := c.NumHeads * c.HeadDim
	kv := c.NumKVHeads * c.HeadDim
	return map[Proj][2]int{
		ProjQ: {q, c.HiddenSize}, ProjK: {kv, c.HiddenSize}, ProjV: {kv, c.HiddenSize},
		ProjO:    {c.HiddenSize, q},
		ProjGate: {c.IntermediateSize, c.HiddenSize}, ProjUp: {c.IntermediateSize, c.HiddenSize},
		ProjDown: {c.HiddenSize, c.IntermediateSize},
	}
}

// bElems is how many halves a weight [n, k] occupies in a B layout.
func bElems(n, k, layout int) int {
	if layout == 2 {
		return n * k
	}
	return n * (k + gemmPad)
}

func bLD(n, k, layout int) int {
	if layout == 2 {
		return 0
	}
	return k + gemmPad
}

// layoutWeights plans the arenas and allocates them. Nothing is read here:
// the sizes follow from the config, so the whole 7 GB layout is decided --
// and can be checked -- before the first weight is touched.
func (g *GPUEncoder) layoutWeights(layers, bankBytes int) error {
	c := g.cfg
	// fp32: the rotary table once, then four norm vectors per layer.
	ropeElems := 2 * g.tokens * (c.HeadDim / 2)
	total32 := ropeElems
	perLayer32 := 2*c.HiddenSize + 2*c.HeadDim

	shapes := g.projShapes()
	perProj := make(map[Proj]int, len(projOrder))
	perLayer16 := 0
	for _, r := range projOrder {
		// Sized for the larger of the two layouts, which differ only when
		// the wrongBLayout control is on: the natural [n, k+pad] carries the
		// §2.3 pad and the fragment tiling does not.
		n := max(bElems(shapes[r][0], shapes[r][1], g.stagedLayout),
			bElems(shapes[r][0], shapes[r][1], kernelLayout))
		perProj[r] = n
		perLayer16 += n
	}
	if perLayer16*2 > bankBytes {
		return fmt.Errorf("qwen: one layer's weights are %d MB and a bank holds %d MB",
			(perLayer16*2)>>20, bankBytes>>20)
	}

	g.w = make([]layerWeights, layers)
	var bankElems []int
	cur := -1
	for i := range g.w {
		w := layerWeights{bOff: make(map[Proj]uint32, len(projOrder))}
		w.attnNorm = uint32(total32)
		w.ffnNorm = w.attnNorm + uint32(c.HiddenSize)
		w.qNorm = w.ffnNorm + uint32(c.HiddenSize)
		w.kNorm = w.qNorm + uint32(c.HeadDim)
		total32 += perLayer32

		if cur < 0 || (bankElems[cur]+perLayer16)*2 > bankBytes {
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

	var err error
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("qwen: fp32 weight arena (%d MB): %w", (total32*4)>>20, err)
	}
	g.wCos, g.wSin = 0, uint32(g.tokens*(c.HeadDim/2))
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("qwen: fp16 weight bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}
	return nil
}

// allocActivations lays out the two activation arenas, shared by every layer.
// Every tensor is sized for the padded token count of the longest run: the
// GEMM has no bounds check and writes whole tiles.
func (g *GPUEncoder) allocActivations() error {
	c := g.cfg
	q := c.NumHeads * c.HeadDim
	kv := c.NumKVHeads * c.HeadDim
	rows := g.arenaRows

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aX = alloc(rows * c.HiddenSize)
	g.aQ = alloc(rows * q)
	g.aK = alloc(rows * kv)
	g.aV = alloc(rows * kv)
	g.aCtx = alloc(rows * q)
	g.aAttn = alloc(rows * c.HiddenSize)
	g.aGate = alloc(rows * c.IntermediateSize)
	g.aUp = alloc(rows * c.IntermediateSize)
	g.aFF = alloc(rows * c.HiddenSize)

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hA = halloc(rows * g.ldaDim)
	g.hQ = halloc(c.NumHeads * rows * c.HeadDim)
	g.hK = halloc(c.NumKVHeads * rows * c.HeadDim)
	g.hV = halloc(c.NumKVHeads * rows * c.HeadDim)
	g.hCtx = halloc(rows * g.ldaQ)
	g.hFFN = halloc(rows * g.ldaFFN)

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("qwen: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("qwen: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the pad rows and pad columns of every A operand come from
	// here, and a short run leaves them as a longer one wrote them.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	return nil
}

// build compiles every pipeline over all four arenas. They are bound to every
// pipeline, used or not, so that one descriptor layout and one push-constant
// size serve the whole sequence, which is what vk.DispatchMultiTimed needs to
// record it into a single command buffer.
func (g *GPUEncoder) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[0]}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	simple := map[string][]byte{
		"rmsnorm": shaders.DiTRMSNorm,
		"normf16": shaders.DiTNormScaleF16,
		"scale":   shaders.DiTScaleF16,
		"rope":    shaders.QwenRoPE,
		"ropedit": shaders.DiTRoPE,
		"pack":    shaders.DiTPackF16,
		"swiglu":  shaders.DiTSwiGLUF16,
		"add":     shaders.DiTGateAdd,
	}
	for name, spirv := range simple {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	if err := g.pipeline("attention", g.attn.spirv, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
	}); err != nil {
		return err
	}
	if g.ctl.noCausal {
		if err := g.pipeline("attention", shaders.QwenAttentionQT1KT4NoCausal, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize,
		}); err != nil {
			return err
		}
		g.attn = attnVariant{name: "qt1_kt4_nocausal", qt: 1, ktil: 4}
	}

	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("qwen: subgroup size control: %w", err)
	}
	// One set of GEMM pipelines per bank: the buffer a GEMM reads its weight
	// out of is in the pipeline's descriptor set, not in its push constants.
	g.gemms = make([]map[GEMMKernel]*vk.ComputePipeline, len(g.banks))
	for b := range g.banks {
		g.gemms[b] = make(map[GEMMKernel]*vk.ComputePipeline, len(g.kernels))
		bankBufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}
		for k, v := range g.kernels {
			spec := vk.PipelineSpec{Buffers: bankBufs, PushConstantSize: pcSize}
			if v.waves > 1 {
				if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
					return fmt.Errorf("qwen: kernel %q needs a pinned 64-wide subgroup", k)
				}
				spec.RequiredSubgroupSize = 64
			}
			mod, err := g.dev.NewShaderModule(v.spirv)
			if err != nil {
				return fmt.Errorf("qwen: shader %s: %w", k, err)
			}
			g.mods = append(g.mods, mod)
			pipe, err := g.dev.NewPipeline(mod, spec)
			if err != nil {
				return fmt.Errorf("qwen: pipeline %s on bank %d: %w", k, b, err)
			}
			g.gemms[b][k] = pipe
		}
	}
	return nil
}

func (g *GPUEncoder) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("qwen: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("qwen: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stageWeights fills the arenas one layer at a time: fp32 for the four norms,
// fp16 for the seven matrices the cooperative matrices read.
//
// One layer at a time is the point. LoadLayer materialises a layer as fp32 on
// the host, 404 MB; all 35 at once would be 14.1 GB and none of it is wanted
// once it has been narrowed into its bank.
func (g *GPUEncoder) stageWeights(set *safetensors.Set, layers int) error {
	c := g.cfg
	rows, embed, err := LoadEmbedding(set, c)
	if err != nil {
		return err
	}
	g.embedRows, g.embed = rows, embed

	shapes := g.projShapes()
	for i := range g.w {
		layer, err := LoadLayer(set, i, c)
		if err != nil {
			return err
		}
		w := &g.w[i]
		g.wbuf.WriteFloat32At(int(w.attnNorm), layer.AttnNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.ffnNorm), layer.FFNNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.qNorm), layer.QNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.kNorm), layer.KNorm.Weight)

		lins := map[Proj]*Linear{
			ProjQ: layer.Q, ProjK: layer.K, ProjV: layer.V, ProjO: layer.O,
			ProjGate: layer.Gate, ProjUp: layer.Up, ProjDown: layer.Down,
		}
		bank := g.banks[w.bank]
		for _, r := range projOrder {
			lin := lins[r]
			if lin.Out != shapes[r][0] || lin.In != shapes[r][1] {
				return fmt.Errorf("qwen: layer %d %s is [%d %d], want %v", i, r, lin.Out, lin.In, shapes[r])
			}
			buf := make([]uint16, bElems(lin.Out, lin.In, g.stagedLayout))
			packB(buf, lin.Weight, lin.Out, lin.In, g.stagedLayout)
			bank.WriteUint16At(int(w.bOff[r]), buf)
		}
	}
	return nil
}

// packChunk is how many output rows one worker narrows at a time.
const packChunk = 64

// packB narrows a [n, k] row-major PyTorch weight into a B layout. Layout 2
// is the fragment tiling (§2.8): tile (nt, kt) is 256 contiguous halves
// holding element (k, n) at (n%16)*16 + k%16, tiles ordered kt-fastest.
// Layout 0 is the natural [n, k+pad] and exists only for the negative
// control.
func packB(dst []uint16, w []float32, n, k, layout int) {
	const tile = coopMatTile
	chunks := (n + packChunk - 1) / packChunk
	rows := func(c int, fn func(i int)) {
		for i := c * packChunk; i < min((c+1)*packChunk, n); i++ {
			fn(i)
		}
	}
	if layout == 2 {
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
		return
	}
	ld := k + gemmPad
	parallelFor(chunks, func(c int) {
		rows(c, func(i int) {
			row, out := w[i*k:(i+1)*k], dst[i*ld:]
			for j, v := range row {
				out[j] = safetensors.F32ToF16(v)
			}
		})
	})
}

// Destroy releases every Vulkan object.
func (g *GPUEncoder) Destroy() {
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

// perSubmit is how many dispatches go into one command buffer: a whole graph
// in one buffer can outlive the driver's reset watchdog (stage 2).
const perSubmit = 8

func submit(d []vk.MultiDispatch) error {
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("qwen: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

// Stage is one timed dispatch of a profile.
type Stage struct {
	Index int
	Layer int
	Kind  string
	GPU   time.Duration
}

// Elapsed sums a profile.
func Elapsed(stages []Stage) time.Duration {
	var total time.Duration
	for _, s := range stages {
		total += s.GPU
	}
	return total
}

// Forward runs every loaded layer over a token sequence and returns the
// hidden state -- `hidden_states[-2]` when the encoder holds
// cfg.EncoderLayers() layers, which is what the DiT's cap_embedder consumes.
func (g *GPUEncoder) Forward(ids []int32) (*Mat, error) {
	if err := g.RunIDs(ids); err != nil {
		return nil, err
	}
	return g.Read(g.aX, g.cfg.HiddenSize), nil
}

// RunIDs is Forward without the read-back: it uploads a token sequence and
// runs every layer, leaving the result in the residual stream for the caller
// to read how it likes. A caller that wants one row of the output rather than
// all of it (ReadRow) saves the whole tensor's trip back through a 0.2 GB/s
// mapping, which at a few hundred tokens is larger than the forward pass.
func (g *GPUEncoder) RunIDs(ids []int32) error {
	if err := g.upload(ids); err != nil {
		return err
	}
	return g.Run()
}

// Run is Forward without the upload or the read-back: it runs every layer
// over whatever the residual stream already holds.
func (g *GPUEncoder) Run() error { return g.RunHooked(nil) }

// RunHooked is Run with a seam between the layers: after is called with each
// layer's index once that layer's dispatches have completed and before the
// next layer starts, so it may read the residual stream (Read) or change it
// (AddRows).
//
// It is the device's counterpart of Model.ForwardEmbeds' `after`, and it
// exists for the same caller: Qwen-Image-2.1's edit path adds a vision
// tower's deepstack features at the image-pad slots after layers 0, 1 and 2
// (qimage/textenc). The hook takes no tensor because reading this arena costs
// 0.2 GB/s and almost every hook wants to write rather than read.
func (g *GPUEncoder) RunHooked(after func(layer int) error) error {
	for i := range g.w {
		d, _, err := g.layerGraph(i)
		if err != nil {
			return err
		}
		if err := submit(d); err != nil {
			return fmt.Errorf("qwen: layer %d: %w", i, err)
		}
		if after != nil {
			if err := after(i); err != nil {
				return fmt.Errorf("qwen: after layer %d: %w", i, err)
			}
		}
	}
	return nil
}

// Profile runs the same graph one dispatch at a time and times each on the
// GPU. Wall clock around Forward is not a measurement of the encoder: it also
// carries the read-back, and this arena's reads run at 0.2 GB/s
// (research/stage-3-dit-attention.md).
func (g *GPUEncoder) Profile(ids []int32) ([]Stage, *Mat, error) {
	if err := g.upload(ids); err != nil {
		return nil, nil, err
	}
	var stages []Stage
	for i := range g.w {
		d, kinds, err := g.layerGraph(i)
		if err != nil {
			return nil, nil, err
		}
		for j := range d {
			dur, err := vk.DispatchMultiTimed(d[j:j+1], 1, 1, true)
			if err != nil {
				return stages, nil, fmt.Errorf("qwen: layer %d dispatch %d (%s): %w", i, j, kinds[j], err)
			}
			stages = append(stages, Stage{Index: len(stages), Layer: i, Kind: kinds[j], GPU: dur})
		}
	}
	return stages, g.Read(g.aX, g.cfg.HiddenSize), nil
}

// RunTo runs the graph up to and including the dispatch with the given label
// in the given layer, and leaves the arenas as that dispatch left them.
//
// It is what makes the validation stagewise: most intermediates are
// overwritten before the graph ends, so a test that only ran Forward could
// compare the output and nothing else.
func (g *GPUEncoder) RunTo(ids []int32, layer int, label string) error {
	if layer < 0 || layer >= len(g.w) {
		return fmt.Errorf("qwen: layer %d out of range, the encoder holds %d", layer, len(g.w))
	}
	if err := g.upload(ids); err != nil {
		return err
	}
	for i := 0; i < layer; i++ {
		d, _, err := g.layerGraph(i)
		if err != nil {
			return err
		}
		if err := submit(d); err != nil {
			return err
		}
	}
	d, kinds, err := g.layerGraph(layer)
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
		return fmt.Errorf("qwen: no dispatch labelled %q (have %v)", label, kinds)
	}
	return submit(d[:end])
}

// Labels lists one layer's dispatches in order, which is both what Profile
// reports and what RunTo accepts.
func (g *GPUEncoder) Labels() []string {
	_, kinds, err := g.layerGraph(0)
	if err != nil {
		return nil
	}
	return kinds
}

// upload gathers the token embeddings on the host and writes them into the
// residual stream. The gather is the one thing the GPU does not do: it is
// [tokens, 2560] out of a 1.56 GB table that nothing else reads, so keeping
// the table on the device would cost 778 MB of fp16 and a shader to save a
// 240 KB write.
func (g *GPUEncoder) upload(ids []int32) error {
	if len(ids) == 0 || len(ids) > g.tokens {
		return fmt.Errorf("qwen: %d tokens; the encoder was built for 1 to %d", len(ids), g.tokens)
	}
	x, err := g.Embeddings(ids)
	if err != nil {
		return err
	}
	if err := g.UploadEmbeds(x); err != nil {
		return err
	}
	g.setRoPE(len(ids))
	return nil
}

// Embeddings gathers a token sequence's rows out of the resident embedding
// table, as the CPU model's method of the same name does.
//
// It is exported for the caller that has to *edit* the embeddings before they
// run: an edit scatters the vision tower's merged rows over the image-pad
// slots, and what the transformer then sees is not a function of the ids
// alone. upload() below is this gather followed by UploadEmbeds and the plain
// rotary table.
func (g *GPUEncoder) Embeddings(ids []int32) (*Mat, error) {
	c := g.cfg
	out := NewMat(len(ids), c.HiddenSize)
	for i, id := range ids {
		if id < 0 || int(id) >= g.embedRows {
			return nil, fmt.Errorf("qwen: token id %d is outside the %d embedding rows", id, g.embedRows)
		}
		copy(out.Row(i), g.embed[int(id)*c.HiddenSize:(int(id)+1)*c.HiddenSize])
	}
	return out, nil
}

// UploadEmbeds writes caller-supplied embeddings into the residual stream and
// fixes the run's length.
//
// It deliberately does *not* touch the rotary table, which is the one thing
// upload() also does: an edit's positions are three-dimensional and not
// 0..T-1, so the table is the caller's (SetRoPE) and the two seams are
// separate. A caller that sets one and forgets the other runs a t2i table
// over an edit sequence -- qimage/textenc runs exactly that as a negative
// control, and it is caught at four orders of magnitude.
func (g *GPUEncoder) UploadEmbeds(x *Mat) error {
	if x.Cols != g.cfg.HiddenSize {
		return fmt.Errorf("qwen: embeddings are %d wide, the model is %d", x.Cols, g.cfg.HiddenSize)
	}
	if x.Rows <= 0 || x.Rows > g.tokens {
		return fmt.Errorf("qwen: %d rows; the encoder was built for 1 to %d", x.Rows, g.tokens)
	}
	g.rows = x.Rows
	if g.AutoPlan {
		if err := g.SetPlan(PlanFor(x.Rows)); err != nil {
			return err
		}
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	return nil
}

// SetRoPE uploads a rotary table the caller built, for the run UploadEmbeds
// has already fixed the length of. The table is the distinct half -- [rows,
// HeadDim/2] cos and sin -- which is what NewRoPE produces and what the
// shader indexes twice.
func (g *GPUEncoder) SetRoPE(r *RoPE) error {
	if r.HeadDim != g.cfg.HeadDim {
		return fmt.Errorf("qwen: a rotary table of head dim %d for a model of %d", r.HeadDim, g.cfg.HeadDim)
	}
	need := g.rows * (g.cfg.HeadDim / 2)
	if len(r.Cos) < need || len(r.Sin) < need {
		return fmt.Errorf("qwen: rotary table holds %d positions, the run is %d rows",
			len(r.Cos)/(g.cfg.HeadDim/2), g.rows)
	}
	g.wbuf.WriteFloat32At(int(g.wCos), r.Cos[:need])
	g.wbuf.WriteFloat32At(int(g.wSin), r.Sin[:need])
	return nil
}

// AddRows adds m into the residual stream at row `at` -- x[at:at+m.Rows] += m.
//
// The staging tensor is the attention branch's output, which is [rows,
// hidden] and dead between layers, which is the only place this is called
// from (RunHooked). One dispatch per call, so a caller with several
// contiguous runs to add should make one call per run and not one per row.
func (g *GPUEncoder) AddRows(at int, m *Mat) error {
	c := g.cfg
	if m.Cols != c.HiddenSize {
		return fmt.Errorf("qwen: adding %d-wide rows to a %d-wide stream", m.Cols, c.HiddenSize)
	}
	if at < 0 || at+m.Rows > g.rows {
		return fmt.Errorf("qwen: adding %d rows at %d, the run is %d rows", m.Rows, at, g.rows)
	}
	if m.Rows == 0 {
		return nil
	}
	g.abuf.WriteFloat32At(int(g.aAttn), m.Data)
	pc := pushConstants{
		Tokens: uint32(m.Rows), Dim: uint32(c.HiddenSize),
		Heads: uint32(c.NumHeads), HeadDim: uint32(c.HeadDim),
		InOff: g.aAttn, OutOff: g.aX + uint32(at*c.HiddenSize),
	}
	return submit([]vk.MultiDispatch{{
		Pipeline: g.pipes["add"], GroupsX: uint32(m.Rows), GroupsY: 1, PushConstants: pc.bytes(),
	}})
}

// setRoPE rewrites the rotary table for a run's length. It belongs to the
// *run* rather than to a layer -- every layer shares it -- and it is 4 KB per
// token, so it is rewritten rather than held for the longest run.
func (g *GPUEncoder) setRoPE(tokens int) {
	rope := NewRoPE(g.cfg.HeadDim, tokens, g.cfg.RopeTheta)
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin)
}

// Read copies one tensor out of the fp32 activation arena, by the offset the
// Tensor* accessors name.
func (g *GPUEncoder) Read(off uint32, cols int) *Mat {
	out := NewMat(g.rows, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), g.rows*cols))
	return out
}

// ReadRow copies one row out of the fp32 activation arena. It exists for the
// case where the whole tensor is not wanted: this arena reads at 0.2 GB/s
// (research/stage-3-dit-attention.md), so an embedding model that only needs
// the last token's row pays 4 KB rather than the run's whole hidden state.
func (g *GPUEncoder) ReadRow(off uint32, row, cols int) []float32 {
	// Offsets into this arena are in float32s, not bytes, as Read's are.
	return g.abuf.ReadFloat32At(int(off)+row*cols, cols)
}

// ReadF16 copies a [tokens, cols] tensor out of the fp16 arena, widening it.
// The row stride is the operand's leading dimension, not cols.
func (g *GPUEncoder) ReadF16(off uint32, cols, lda int) *Mat {
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

// The stage-boundary tensors, for the validation walk.
func (g *GPUEncoder) TensorX() uint32    { return g.aX }
func (g *GPUEncoder) TensorQ() uint32    { return g.aQ }
func (g *GPUEncoder) TensorK() uint32    { return g.aK }
func (g *GPUEncoder) TensorV() uint32    { return g.aV }
func (g *GPUEncoder) TensorCtx() uint32  { return g.aCtx }
func (g *GPUEncoder) TensorAttn() uint32 { return g.aAttn }
func (g *GPUEncoder) TensorGate() uint32 { return g.aGate }
func (g *GPUEncoder) TensorFF() uint32   { return g.aFF }

// TensorA is the fp16 A operand the projections read and LDA its row stride.
func (g *GPUEncoder) TensorA() uint32 { return g.hA }
func (g *GPUEncoder) LDA() int        { return g.ldaDim }

// Layers is how many decoder layers the encoder holds and Tokens the longest
// run its arenas were built for.
func (g *GPUEncoder) Layers() int    { return len(g.w) }
func (g *GPUEncoder) Tokens() int    { return g.tokens }
func (g *GPUEncoder) Plan() GEMMPlan { return g.plan }

// SetPlan changes which kernel each projection runs on. Every rung reads the
// same fragment-tiled weight, so this moves a pipeline and a tile and
// restages nothing -- the ladder is one 7 GB load, not one per rung.
func (g *GPUEncoder) SetPlan(plan GEMMPlan) error {
	if plan == nil {
		plan = DefaultGEMMPlan()
	}
	align := g.attn.keyBlock()
	for _, r := range projOrder {
		v, ok := variantFor(plan[r])
		if !ok {
			return fmt.Errorf("qwen: no GEMM kernel %q (have %v)", plan[r], GEMMKernels())
		}
		if v.layout != kernelLayout {
			return fmt.Errorf("qwen: kernel %q reads B layout %d, but every rung here reads %d",
				v.name, v.layout, kernelLayout)
		}
		align = max(align, v.bm)
	}
	g.plan, g.align = plan, align
	return nil
}
func (g *GPUEncoder) Attention() string { return g.attn.name }

// Banks reports how the fp16 weights were split: one byte count per storage
// buffer, and one bank index per layer.
func (g *GPUEncoder) Banks() (bytes []int, byLayer []int) {
	for _, b := range g.banks {
		bytes = append(bytes, b.Size())
	}
	byLayer = make([]int, len(g.w))
	for i, w := range g.w {
		byLayer[i] = w.bank
	}
	return bytes, byLayer
}

// ActivationBytes is what the shared arenas cost on the device and
// WeightBytes what every layer's weights cost, fp32 arena and fp16 banks.
func (g *GPUEncoder) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }
func (g *GPUEncoder) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

// FLOPs is one layer's multiply-add count at a sequence length: the seven
// projections and both attention matmuls, the second of which is causal and
// so does half the work.
func (g *GPUEncoder) FLOPs(rows int) float64 {
	c := g.cfg
	t := float64(rows)
	hidden, ffn := float64(c.HiddenSize), float64(c.IntermediateSize)
	q := float64(c.NumHeads * c.HeadDim)
	kv := float64(c.NumKVHeads * c.HeadDim)
	proj := 2 * t * hidden * (q + 2*kv + q)
	ff := 2 * t * hidden * ffn * 3
	attn := 2 * t * t * q // scores and context, halved by the causal mask
	return proj + ff + attn
}

// WeightReadBytes is what one layer's projections cost to read once, which is
// the floor this model is actually against: at T tokens its intensity is T
// flop per byte and the crossover is 235.
func (g *GPUEncoder) WeightReadBytes() float64 {
	shapes := g.projShapes()
	var n float64
	for _, r := range projOrder {
		n += 2 * float64(shapes[r][0]) * float64(shapes[r][1])
	}
	return n
}

// layerGraph builds one layer's dispatch sequence, with a label per dispatch.
// Forward, Profile and RunTo share it so that what the profiler times is what
// Forward runs, and the encoder is the concatenation of one of these per
// layer. It touches no memory, so asking for the labels costs nothing.
//
// Twenty-one dispatches, against the DiT block's eighteen. The difference is
// entirely that this is the *unfused* shape: the q and k norms, the rotation
// and the fragment pack are four dispatches here where stage 4b's `qkpack`
// makes them one, because each of the four is a tensor reference/dump_qwen.py
// dumps and this is the port being validated rather than tuned. Fusing them
// is the same lever stage 4b already measured, waiting for a profile that
// says it is worth taking.
func (g *GPUEncoder) layerGraph(i int) ([]vk.MultiDispatch, []string, error) {
	w := &g.w[i]
	bank := w.bank
	if g.ctl.shiftBank {
		bank = (bank + 1) % len(g.gemms)
	}
	c := g.cfg
	qWidth := c.NumHeads * c.HeadDim
	kvWidth := c.NumKVHeads * c.HeadDim
	rep := c.NumHeads / c.NumKVHeads
	tokPad := roundUp(g.rows, g.align)

	base := pushConstants{
		Tokens: uint32(g.rows), Dim: uint32(c.HiddenSize),
		Heads: uint32(c.NumHeads), HeadDim: uint32(c.HeadDim),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// RMS norm over the whole row, narrowed into the fp16 arena as a GEMM A
	// operand. Qwen3 has no modulation, so the scale site of the DiT's
	// shader is switched off (aux2 = 0) and this is the norm alone.
	normF16 := func(kind string, in, out, wOff uint32, width, lda int) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = in, out, wOff
		pc.Dim = uint32(width)
		pc.LDA = uint32(lda)
		pc.Eps = math.Float32bits(float32(c.RMSEps))
		add("normf16", kind, uint32(g.rows), 1, pc)
	}
	// The per-head q/k norms: the weight is headDim wide and a row holds
	// every head, so one span is one head. The control widens the span to the
	// whole row, which is the plausible mistake.
	headNorm := func(kind string, off, wOff uint32, width int) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = off, off, wOff
		pc.Dim = uint32(width)
		pc.Span = uint32(c.HeadDim)
		if g.ctl.qkNormWide {
			pc.Span = uint32(width)
		}
		pc.Eps = math.Float32bits(float32(c.RMSEps))
		add("rmsnorm", kind, uint32(g.rows*width)/pc.Span, 1, pc)
	}
	rope := func(kind string, off uint32, width, heads int) {
		pc := base
		pc.InOff, pc.OutOff = off, off
		pc.Dim, pc.Heads = uint32(width), uint32(heads)
		pc.WOff, pc.Aux0 = g.wCos, g.wSin
		pipe := "rope"
		if g.ctl.ropeAdjacent {
			pipe = "ropedit"
		}
		add(pipe, kind, groups(g.rows*heads*c.HeadDim/2, 256), 1, pc)
	}
	// The fragment pack: one plane per head, tiles of 16 tokens. mode 1 is
	// the transpose v needs as the B operand of p.v.
	pack := func(kind string, in, out uint32, width, heads, mode int, scale float32) {
		pc := base
		pc.InOff, pc.OutOff = in, out
		pc.Dim = uint32(width)
		pc.Aux0, pc.Aux1 = uint32(mode), uint32(tokPad)
		pc.Scale = math.Float32bits(scale)
		add("pack", kind, groups(tokPad, coopMatTile), uint32(heads), pc)
	}
	gemm := func(r Proj, kind string, aOff, cOff uint32, n, k, lda int) error {
		kernel := g.plan[r]
		v, ok := g.kernels[kernel]
		if !ok {
			return fmt.Errorf("qwen: projection %s has no pipeline", r)
		}
		pipe, ok := g.gemms[bank][kernel]
		if !ok {
			return fmt.Errorf("qwen: kernel %s is not built for bank %d", kernel, bank)
		}
		if n%v.bn != 0 || tokPad%v.bm != 0 {
			return fmt.Errorf("qwen: %s tile %dx%d does not divide [%d %d]", r, v.bm, v.bn, tokPad, n)
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, w.bOff[r]
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(tokPad), uint32(n), uint32(k)
		pc.LDA, pc.LDB = uint32(lda), uint32(bLD(n, k, v.layout))
		d = append(d, vk.MultiDispatch{
			Pipeline: pipe, GroupsX: uint32(n / v.bn), GroupsY: uint32(tokPad / v.bm), PushConstants: pc.bytes(),
		})
		kinds = append(kinds, kind)
		return nil
	}
	// The residual, ungated: Qwen3 has no adaLN, so the DiT's gate site is
	// switched off and this is x += y.
	residual := func(kind string, y uint32) {
		pc := base
		pc.InOff, pc.OutOff = y, g.aX
		add("add", kind, uint32(g.rows), 1, pc)
	}

	// ---- Attention.
	const log2e = 1.4426950408889634
	scale := float32(1/math.Sqrt(float64(c.HeadDim))) * log2e
	normF16("attn in", g.aX, g.hA, w.attnNorm, c.HiddenSize, g.ldaDim)
	for _, s := range []struct {
		r    Proj
		kind string
		out  uint32
		n    int
	}{{ProjQ, "gemm q", g.aQ, qWidth}, {ProjK, "gemm k", g.aK, kvWidth}, {ProjV, "gemm v", g.aV, kvWidth}} {
		if err := gemm(s.r, s.kind, g.hA, s.out, s.n, c.HiddenSize, g.ldaDim); err != nil {
			return nil, nil, err
		}
	}
	headNorm("rmsnorm q", g.aQ, w.qNorm, qWidth)
	headNorm("rmsnorm k", g.aK, w.kNorm, kvWidth)
	rope("rope q", g.aQ, qWidth, c.NumHeads)
	rope("rope k", g.aK, kvWidth, c.NumKVHeads)
	// q carries 1/sqrt(headDim) and log2(e), so the kernel's exponential is
	// exp2 -- one instruction on this ISA.
	pack("pack q", g.aQ, g.hQ, qWidth, c.NumHeads, 0, scale)
	pack("pack k", g.aK, g.hK, kvWidth, c.NumKVHeads, 0, 1)
	pack("pack v", g.aV, g.hV, kvWidth, c.NumKVHeads, 1, 1)

	pcAttn := base
	pcAttn.InOff, pcAttn.OutOff = g.hQ, g.aCtx
	pcAttn.KOff, pcAttn.VOff = g.hK, g.hV
	pcAttn.Dim = uint32(qWidth)
	pcAttn.Aux1, pcAttn.Aux2 = uint32(tokPad), uint32(rep)
	add("attention", "attention", groups(g.rows, g.attn.rows()), uint32(c.NumHeads), pcAttn)

	pcNarrow := base
	pcNarrow.InOff, pcNarrow.OutOff = g.aCtx, g.hCtx
	pcNarrow.Dim = uint32(qWidth)
	pcNarrow.LDA = uint32(g.ldaQ)
	add("scale", "narrow ctx", uint32(g.rows), 1, pcNarrow)

	if err := gemm(ProjO, "gemm o", g.hCtx, g.aAttn, c.HiddenSize, qWidth, g.ldaQ); err != nil {
		return nil, nil, err
	}
	residual("resid attn", g.aAttn)

	// ---- Feed forward.
	normF16("ffn in", g.aX, g.hA, w.ffnNorm, c.HiddenSize, g.ldaDim)
	for _, s := range []struct {
		r    Proj
		kind string
		out  uint32
	}{{ProjGate, "gemm gate", g.aGate}, {ProjUp, "gemm up", g.aUp}} {
		if err := gemm(s.r, s.kind, g.hA, s.out, c.IntermediateSize, c.HiddenSize, g.ldaDim); err != nil {
			return nil, nil, err
		}
	}
	pcGLU := base
	pcGLU.InOff, pcGLU.KOff, pcGLU.OutOff = g.aGate, g.aUp, g.hFFN
	pcGLU.Dim = uint32(c.IntermediateSize)
	pcGLU.LDA = uint32(g.ldaFFN)
	// The SwiGLU shader scales its output on the way into fp16 and its
	// consumer is expected to undo it. Here nothing can: this tensor's GEMM
	// feeds a residual add, not an RMS norm, so the scale has to be 1 and this
	// model pays the fp16 range it costs. (The DiT's is not 1 -- see
	// GPUStack.FFScale.) It is stated rather than left at the push constant's
	// zero, which would multiply the whole feed-forward by nothing.
	pcGLU.Scale = math.Float32bits(1)
	add("swiglu", "swiglu", uint32(g.rows), 1, pcGLU)

	if err := gemm(ProjDown, "gemm down", g.hFFN, g.aFF, c.HiddenSize, c.IntermediateSize, g.ldaFFN); err != nil {
		return nil, nil, err
	}
	residual("resid ffn", g.aFF)
	return d, kinds, nil
}
