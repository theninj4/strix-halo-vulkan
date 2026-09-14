// Vulkan FastConformer encoder: SPEECH.md stage S6.
//
// The encoder is 91.5% of this model's CPU time and ~180 GFLOP for eleven
// seconds of audio, and almost all of it is the GEMM ladder this repository
// already has -- two feed forwards, four attention projections, two pointwise
// convolutions and the relative-position projection, every one of them
// [T, 1024] against a [1024, N] weight. So the graph here is
// zimage/qwen/gpu.go's with the encoder's own arithmetic in the gaps, and it
// is a separate type for the same reason that one is: what differs is
// structural.
//
//	LayerNorm, not RMS norm, and five of them per layer, each with a weight
//	  *and* a bias. There is no RMS norm anywhere in this model.
//	Two score matrices, not one. Transformer-XL relative-position attention
//	  adds (q + bias_v).rel_k^T to (q + bias_u).k^T, folded onto the [T, T]
//	  grid by a shift that is a diagonal read (parakeet_relshift.comp).
//	A convolution branch -- a GLU over channels, a 9-tap depthwise
//	  convolution over time, a folded BatchNorm -- which is the one operator
//	  family in the encoder that is neither a GEMM nor elementwise.
//	M is the clip length. 11 s is 138 frames and 30 s is 375, against a DiT
//	  tuned at M=4096, so the tile is chosen per run (PlanFor) exactly as
//	  stage 5c found it had to be for a prompt.
//
// What is shared is everything that did not have to change: the four-arena
// binding layout and push-constant block of shaders/dit_common.glsl, the
// projection GEMM, the fragment pack, and the matrix-core attention kernel,
// which takes the position term as an additive bias (its REL_BIAS build).
//
// Everything the matrix cores touch is fp16 with fp32 accumulators, which is
// not a choice -- it is the only operand type this device implements -- and
// what it costs was measured on the CPU first: SPEECH.md's fp16 survey
// narrows every operand of the reference and gets the same transcript with
// the same decode trace, at 4.3% relative drift at the encoder output. So the
// bound this port is held to is the transcript, and the per-stage tensors
// locate a fault rather than certify its absence.
package parakeet

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
// A copy of zimage/dit's and zimage/qwen's rather than a shared type, for the
// reason qwen states: the layout is a contract with a *shader*, and this
// package dispatches the DiT's compiled GEMM, so the copy is the thing that
// has to stay identical. One push-constant size across every pipeline is what
// lets vk.DispatchMultiTimed record a whole layer into one command buffer.
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

// pushConstantSize is the block's size, which every pipeline in this package
// declares -- including the transducer tail's, over a completely separate set
// of buffers. One size across every pipeline is what lets
// vk.DispatchMultiTimed record a mixed sequence into one command buffer.
var pushConstantSize = int(unsafe.Sizeof(pushConstants{}))

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

const (
	// coopMatTile is the cooperative-matrix extent: 16x16x16 is the only
	// shape this device reports.
	coopMatTile = 16
	// gemmPad is the leading-dimension pad in halves (§2.3's 256 B), applied
	// to every A operand: a K-strided fragment load issues 16 addresses one
	// row stride apart, and 1024, 2048 and 4096 are all multiples of 256.
	gemmPad = 128
	// kernelLayout is the B layout every projection rung reads: 16x16
	// fragment tiles (§2.8), which is free on a weight because it is packed
	// once at upload.
	kernelLayout = 2
	// noW is dit_common.glsl's NO_W, i.e. "this pass has no weight vector".
	noW = 0xffffffff
	// arenaAlign is what the *arenas* are sized to: the widest BM in the
	// ladder, so that any plan fits the buffers allocated once.
	arenaAlign = 128
	// planeAlign is what the attention geometry is padded to, which is a
	// different number from the GEMM's tile and has to be. The score kernel
	// reads whole key blocks, so the packed planes and the position-bias
	// plane need roundUp(keys, keyBlock) columns to exist; the GEMM's M only
	// needs to reach the tile it covers. Keeping them apart is what lets an
	// 11 s clip run its feed forwards at M=160 instead of M=256 -- 1.6x the
	// arithmetic, all of it multiplying padding.
	planeAlign = 64
	// log2e turns the score kernel's exp2 into exp: q carries it (through
	// the pack) and so must the position bias, since the two are summed
	// before the softmax.
	log2e = 1.4426950408889634
)

// GEMMKernel names one build of shaders/dit_gemm.comp.
type GEMMKernel string

const (
	// The wave32 narrow-M rungs, which are what a clip-length M wants:
	// results/shapes.csv measures the 32x32 tile at wave32 reaching 23.1
	// TFLOP/s at M=138 against 12.7 for the wave64 rungs (§6.2, and
	// research/3.4-model-shapes.md).
	GEMMReg32x32W32 GEMMKernel = "reg32x32_bt16_w32"
	GEMMReg32x64W32 GEMMKernel = "reg32x64_bt16_w32"
	GEMMReg16x64W32 GEMMKernel = "reg16x64_bt16_w32"
	// The wave64 rungs the text encoder measured, kept as the ladder's
	// control: they are what "use the kernel we already have" would pick.
	GEMMReg16x64  GEMMKernel = "reg16x64_bt16"
	GEMMReg32x64  GEMMKernel = "reg32x64_bt16"
	GEMMReg32x128 GEMMKernel = "reg32x128_bt16"
	GEMMReg64     GEMMKernel = "reg64_bt16"
	GEMMWG128x256 GEMMKernel = "wg128x256_bt16_swz8"
	// The position term's rungs. Its B operand is rel_k -- an activation,
	// different every clip -- so it cannot be staged as fragment tiles and
	// reads the natural [N, ldb] layout instead (B_LAYOUT=0). These are the
	// only two builds of it, and they are not interchangeable with the rest:
	// a plan may not name them and they may not be named by one.
	gemmActReg32x64 GEMMKernel = "reg32x64_hka4"
	gemmActReg64    GEMMKernel = "reg64_hka4"
)

// gemmVariant is one build's geometry, which the Go side has to be told: BM,
// BN, the B layout and the wave size are compiled into the SPIR-V and cannot
// be read back out of it.
type gemmVariant struct {
	name   GEMMKernel
	spirv  []byte
	bm, bn int
	layout int
	wave   uint32 // 0 takes the driver's default, which is 64 here
}

// gemmVariants is the projection ladder: every rung reads the fragment-tiled
// weight, so a plan can be changed after construction without restaging the
// bank, which is what makes the ladder in cmd/asr one load instead of eight.
var gemmVariants = []gemmVariant{
	{name: GEMMReg32x32W32, spirv: shaders.DiTGEMMReg32x32TiledW32, bm: 32, bn: 32, layout: 2, wave: 32},
	{name: GEMMReg32x64W32, spirv: shaders.DiTGEMMReg32x64TiledW32, bm: 32, bn: 64, layout: 2, wave: 32},
	{name: GEMMReg16x64W32, spirv: shaders.DiTGEMMReg16x64TiledW32, bm: 16, bn: 64, layout: 2, wave: 32},
	{name: GEMMReg16x64, spirv: shaders.DiTGEMMReg16x64Tiled, bm: 16, bn: 64, layout: 2},
	{name: GEMMReg32x64, spirv: shaders.DiTGEMMReg32x64Tiled, bm: 32, bn: 64, layout: 2},
	{name: GEMMReg32x128, spirv: shaders.DiTGEMMReg32x128Tiled, bm: 32, bn: 128, layout: 2},
	{name: GEMMReg64, spirv: shaders.DiTGEMMReg64Tiled, bm: 64, bn: 64, layout: 2},
	{name: GEMMWG128x256, spirv: shaders.DiTGEMMWG128x256TiledSWZ8, bm: 128, bn: 256, layout: 2},
}

// actGEMMVariants are the rungs whose B operand comes out of the fp16
// *activation* arena. They are built over a different descriptor set -- same
// shader, the arena bound where the weight bank is -- so they are their own
// table rather than a flag on the one above.
var actGEMMVariants = []gemmVariant{
	{name: gemmActReg32x64, spirv: shaders.DiTGEMMReg32x64HKA4, bm: 32, bn: 64, layout: 0},
	{name: gemmActReg64, spirv: shaders.DiTGEMMReg64HKA4, bm: 64, bn: 64, layout: 0},
}

// Proj names one of the eleven matrices a layer multiplies by.
type Proj string

const (
	ProjFF1A Proj = "ff1.linear1"
	ProjFF1B Proj = "ff1.linear2"
	ProjQ    Proj = "q"
	ProjK    Proj = "k"
	ProjV    Proj = "v"
	ProjO    Proj = "o"
	ProjRelK Proj = "rel_k"
	ProjPW1  Proj = "conv.pw1"
	ProjPW2  Proj = "conv.pw2"
	ProjFF2A Proj = "ff2.linear1"
	ProjFF2B Proj = "ff2.linear2"
)

// projOrder is every projection, in the order the graph issues them.
var projOrder = []Proj{ProjFF1A, ProjFF1B, ProjQ, ProjK, ProjV, ProjRelK, ProjO, ProjPW1, ProjPW2, ProjFF2A, ProjFF2B}

// GEMMPlan chooses a kernel per projection.
type GEMMPlan map[Proj]GEMMKernel

// UniformGEMMPlan runs every projection on one kernel, which is what the
// ladder sweeps.
func UniformGEMMPlan(k GEMMKernel) GEMMPlan {
	p := GEMMPlan{}
	for _, r := range projOrder {
		p[r] = k
	}
	return p
}

// GEMMKernels lists every projection build, in ladder order.
func GEMMKernels() []GEMMKernel {
	out := make([]GEMMKernel, 0, len(gemmVariants))
	for _, v := range gemmVariants {
		out = append(out, v.name)
	}
	return out
}

// DefaultGEMMPlan is the rung that wins at the length a clip usually is: the
// wave32 32x32 tile, measured over the whole graph by TestGPUGEMMLadder.
func DefaultGEMMPlan() GEMMPlan { return UniformGEMMPlan(GEMMReg32x32W32) }

// PlanFor is the measured schedule: which tile wins at a given frame count.
//
// TestGPUGEMMLadder times the whole encoder once per rung at four lengths, so
// these are the graph's numbers and not results/shapes.csv's -- the two agree
// on the winner and not on the margin, because that table measures one shape
// with its operands hot and this streams 1.15 GB of weights past the cores
// once per clip.
//
//	frames   winner                    runner-up
//	   138   reg32x32_w32  1.00x       reg32x64_w32  1.04x
//	   384   reg32x32_w32  1.00x       reg32x64_w32  1.03x
//	   768   reg32x64_w32  1.00x       reg32x32_w32  1.01x
//	  1024   reg32x64_w32  1.00x       reg64         1.05x
//
// The result the table is really about is the wave size. Every wave32 rung
// beats its wave64 twin at every length -- 1.28x at 138 frames on the same
// tile, 1.15x at 1024 -- which is §6.2's finding holding on a whole model,
// and the DiT's own winner (wg128x256, four waves on a 128-row tile) is 1.83x
// off the pace at 138 frames because 86 of those rows would be padding.
//
// The boundary sits between measured points, and the cost of getting it wrong
// is 1-6% either side, so it is placed where the two curves cross rather than
// on a length anything was measured at.
func PlanFor(frames int) GEMMPlan {
	if frames <= 512 {
		return UniformGEMMPlan(GEMMReg32x32W32)
	}
	return UniformGEMMPlan(GEMMReg32x64W32)
}

// controls are the deliberate breakages the negative control switches on.
// They live here rather than in the test because each one breaks the *graph*
// -- which shader a dispatch names, what a pass is told its bounds are -- and
// none of it is reachable from outside Run.
type controls struct {
	// noMean normalises without subtracting the mean, i.e. makes every
	// LayerNorm in the encoder an RMS norm. It is the mistake the norm kernel
	// exists to not make, and it produces a plausible tensor.
	noMean bool
	// shiftSlice reads the position scores down the middle T columns of the
	// 2T-1 wide matrix instead of down the diagonal -- the thing the shift is
	// most often confused with, which agrees with it on exactly one row
	// (TestRelShiftIsNotASlice).
	shiftSlice bool
	// noPadZero leaves the frames past the clip's valid length in the GLU
	// output, so the depthwise convolution's 9-tap window walks padding
	// backwards into the last four real frames.
	noPadZero bool
	// noBiasU drops the content bias from q's pack, which is half of what
	// makes this attention Transformer-XL's rather than a plain one.
	noBiasU bool
	// subNoMask leaves the frames past the valid length in every stage of the
	// subsampling stack, so the padding at the end of the clip walks forward
	// through three stride-2 convolutions into the encoder's last frames.
	subNoMask bool
	// subFlatOrder stages the subsampling linear's weight without the column
	// permutation the channel-last feature map needs, i.e. reads the flatten
	// with the frequency axis slower instead of the channel axis. It is the
	// stage's sharpest failure mode because its output is an ordinary tensor.
	subFlatOrder bool
}

// layerWeights is where one layer's weights sit in the arenas.
type layerWeights struct {
	// fp32 arena: the five norms (weight and bias), attention's two biases,
	// the depthwise filters and the folded BatchNorm affine.
	normFF1, normAttn, normConv, normFF2, normOut [2]uint32
	biasU, biasV                                  uint32
	dw, bnScale, bnShift                          uint32
	// fp16 bank: the offset of each projection, in halves.
	bOff map[Proj]uint32
}

// GPUEncoder runs the FastConformer stack on the device. One instance holds
// every layer -- 1.21 GB of fp16 weights for all 24 -- so a clip is a
// dispatch sequence and not a load.
type GPUEncoder struct {
	dev *vk.Device
	cfg EncoderConfig

	// The subsampling stack, which S7 moved onto the device
	// (parakeet/gpusub.go). The host implementation is kept -- it is the CPU
	// reference the stage is validated against, and HostSubsampling selects
	// it, which is what makes the two comparable in one process.
	sub        *Subsampling
	scaleInput bool

	// HostSubsampling runs the five convolutions and the linear on the CPU,
	// as S6 did. It is the ladder's control and it is 7x the whole rest of
	// the encoder, so it is off.
	HostSubsampling bool
	subChans        int
	maxGeom         subGeom // the longest clip the arenas were sized for
	geom            subGeom // the clip in the arenas
	subPlan         SubPlan
	subW            subWeights
	subPending      bool
	sMel, sV0       uint32
	sV1, sV2        uint32
	hS1, hS2, hFlat uint32
	ldaSub, ldaFlat int

	wbuf *vk.Buffer // fp32 weights
	abuf *vk.Buffer // fp32 activations
	hbuf *vk.Buffer // fp16 activations
	bank *vk.Buffer // fp16 weights

	pipes    map[string]*vk.ComputePipeline
	gemms    map[GEMMKernel]*vk.ComputePipeline
	actGEMMs map[GEMMKernel]*vk.ComputePipeline
	mods     []*vk.ShaderModule

	// AutoPlan re-plans every run for its own length, using PlanFor. It is on
	// unless the caller named a plan, and it is free: every rung reads the
	// same staged weight, so this changes which pipeline a dispatch names and
	// nothing else.
	AutoPlan bool

	plan    GEMMPlan
	kernels map[GEMMKernel]gemmVariant
	attn    attnVariant
	actGEMM gemmVariant
	ctl     controls

	// frames is the longest clip the arenas were built for and rowsPad its
	// rounding up to arenaAlign; rows is the run in progress, rowsRun its
	// padding up to the plan's tile, planeRows its padding up to the
	// attention key block, and valid how many of its frames are not padding.
	frames, rowsPad     int
	rows, rowsRun       int
	planeRows           int
	valid               int
	nPad, nRun          int // the position term's 2T-1 extent, padded
	planBM              int // the widest BM the current plan names
	dim, ffn            int
	heads, headDim      int
	convK               int
	ldaDim, ldaFFN      int
	normEps             float64
	w                   []layerWeights
	aX, aQ, aK, aV      uint32
	aBranch, aFF, aPW1  uint32
	aGLU, aRelK         uint32
	aBD, aBias          uint32
	actElems            int
	hA, hFF, hPW, hQV   uint32
	hPos, hRelK         uint32
	hQ, hK, hV, hCtx    uint32
	hElems              int
	posRows             int // how many rows of hPos the current run wrote
	uploadedPositionsAt int
}

// attnVariant is one build of the score kernel.
type attnVariant struct {
	name  string
	spirv []byte
	qt    int
	ktil  int
	wave  uint32
}

func (v attnVariant) rows() int     { return v.qt * coopMatTile }
func (v attnVariant) keyBlock() int { return v.ktil * coopMatTile }

// attnVariants is the short ladder stage 3c already narrowed. At 138 frames
// attention is 2% of the encoder's arithmetic, so this exists to be correct
// and unsurprising rather than to be tuned; the wave32 rung is first because
// it is the measured winner at every length the DiT swept.
var attnVariants = []attnVariant{
	{name: "qt1_kt4_w32", spirv: shaders.ParakeetAttentionQT1KT4W32, qt: 1, ktil: 4, wave: 32},
	{name: "qt1_kt4", spirv: shaders.ParakeetAttentionQT1KT4, qt: 1, ktil: 4},
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// NewGPUEncoder stages an already-loaded encoder onto the device, sized for
// clips of at most maxFrames *encoder* frames -- which is one eightieth of
// the audio's samples, not its mel frames. plan may be nil, which takes
// PlanFor per run.
//
// The weights are read from the host-side Encoder rather than from the
// checkpoint, because this is the same model object the CPU reference runs:
// a test that stages from one and compares against the other is then
// comparing two implementations and not two loads.
func NewGPUEncoder(dev *vk.Device, enc *Encoder, maxFrames int, plan GEMMPlan) (*GPUEncoder, error) {
	return newGPUEncoder(dev, enc, maxFrames, plan, controls{})
}

func newGPUEncoder(dev *vk.Device, enc *Encoder, maxFrames int, plan GEMMPlan, ctl controls) (*GPUEncoder, error) {
	if maxFrames <= 0 {
		return nil, fmt.Errorf("parakeet: maxFrames is %d", maxFrames)
	}
	if len(enc.Layers) == 0 {
		return nil, fmt.Errorf("parakeet: the encoder has no layers")
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("parakeet: this device has no 16x16x16 fp16 cooperative matrix; the encoder graph needs one")
	}
	cfg := enc.Config
	g := &GPUEncoder{
		dev:        dev,
		cfg:        cfg,
		ctl:        ctl,
		pipes:      make(map[string]*vk.ComputePipeline),
		gemms:      make(map[GEMMKernel]*vk.ComputePipeline),
		actGEMMs:   make(map[GEMMKernel]*vk.ComputePipeline),
		kernels:    make(map[GEMMKernel]gemmVariant),
		frames:     maxFrames,
		dim:        cfg.HiddenSize,
		ffn:        cfg.IntermediateSize,
		heads:      cfg.NumAttentionHeads,
		headDim:    cfg.HeadDim(),
		convK:      enc.Layers[0].Conv.Kernel,
		normEps:    enc.Layers[0].NormFF1.Eps,
		sub:        enc.Subsampling,
		scaleInput: cfg.ScaleInput,
	}
	if g.headDim != headDim {
		return nil, fmt.Errorf("parakeet: head dim %d, but the kernels are built for %d", g.headDim, headDim)
	}
	// The subsampling stack's extents follow from the encoder frame count: a
	// stride-2 convolution with kernel 3 and padding 1 takes n to ceil(n/2),
	// so a clip of at most maxFrames encoder frames is at most 8*maxFrames
	// mel frames and the three intermediate tensors are at most 4, 2 and 1
	// times maxFrames deep.
	g.subChans = cfg.SubsamplingChannels
	g.maxGeom = subGeometry(8*maxFrames, cfg.NumMelBins, 8*maxFrames)
	if err := g.checkSubsampling(enc.Subsampling); err != nil {
		return nil, err
	}
	g.ldaDim = g.dim + gemmPad
	g.ldaFFN = g.ffn + gemmPad
	g.rowsPad = roundUp(maxFrames, arenaAlign)
	g.nPad = roundUp(2*maxFrames-1, arenaAlign)
	g.rows, g.rowsRun, g.valid = maxFrames, g.rowsPad, maxFrames
	g.planeRows, g.nRun = g.rowsPad, g.nPad

	if err := g.chooseAttention(); err != nil {
		g.Destroy()
		return nil, err
	}
	for _, v := range gemmVariants {
		g.kernels[v.name] = v
	}
	g.actGEMM = actGEMMVariants[0]
	explicit := plan
	if plan == nil {
		plan = PlanFor(maxFrames)
	}
	if err := g.SetPlan(plan); err != nil {
		g.Destroy()
		return nil, err
	}
	g.AutoPlan = explicit == nil
	if err := g.SetSubPlan(DefaultSubPlan()); err != nil {
		g.Destroy()
		return nil, err
	}

	if err := g.layoutWeights(len(enc.Layers)); err != nil {
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
	if err := g.stageWeights(enc); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// headDim is compiled into the pack and score shaders.
const headDim = 128

// canWMMA reports whether this device can run the matrix-core kernels at all.
// A device reporting some other shape would run these shaders *wrong* rather
// than slowly, so the answer is "no" rather than "try".
func canWMMA(dev *vk.Device) (bool, error) {
	feat := dev.Features()
	if !feat.Float16 || !feat.CoopMatrix {
		return false, nil
	}
	shapes, err := dev.Physical().CooperativeMatrixShapes()
	if err != nil {
		return false, fmt.Errorf("parakeet: cooperative-matrix shapes: %w", err)
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

// canPin reports whether a pipeline on this device may require a wave size.
func (g *GPUEncoder) canPin(wave uint32) (bool, error) { return canPinWave(g.dev, wave) }

// canPinWave is canPin for a caller that holds only the device, which the
// transducer tail does (parakeet/gpudecode.go).
func canPinWave(dev *vk.Device, wave uint32) (bool, error) {
	if wave == 0 {
		return true, nil
	}
	feat := dev.Features()
	if !feat.SubgroupSizeControl {
		return false, nil
	}
	sgs, err := dev.Physical().SubgroupSizeControl()
	if err != nil {
		return false, fmt.Errorf("parakeet: subgroup size control: %w", err)
	}
	return sgs.Supported && wave >= sgs.MinSubgroupSize && wave <= sgs.MaxSubgroupSize, nil
}

// chooseAttention picks the first score kernel whose wave size this device
// can pin, in ladder order.
func (g *GPUEncoder) chooseAttention() error {
	for _, v := range attnVariants {
		ok, err := g.canPin(v.wave)
		if err != nil {
			return err
		}
		if ok {
			g.attn = v
			return nil
		}
	}
	return fmt.Errorf("parakeet: no runnable attention kernel")
}

// SetPlan changes which tile each projection runs on. Every rung is built and
// every rung reads the same staged weight, so this is a change of pipeline
// and not of upload.
func (g *GPUEncoder) SetPlan(plan GEMMPlan) error {
	next := GEMMPlan{}
	g.planBM = 0
	for _, r := range projOrder {
		k, ok := plan[r]
		if !ok {
			return fmt.Errorf("parakeet: plan names no kernel for %s", r)
		}
		v, ok := g.kernels[k]
		if !ok {
			return fmt.Errorf("parakeet: no such projection kernel %q", k)
		}
		if v.layout != kernelLayout {
			return fmt.Errorf("parakeet: kernel %q reads B layout %d, the weights are staged as %d", k, v.layout, kernelLayout)
		}
		next[r] = k
		g.planBM = max(g.planBM, v.bm)
	}
	g.plan = next
	g.AutoPlan = false
	g.setExtents()
	return nil
}

// setExtents recomputes the run's three padded extents. They move with the
// plan as well as with the clip, which is why this is not part of Upload.
func (g *GPUEncoder) setExtents() {
	if g.planBM == 0 || g.rows == 0 {
		return
	}
	g.rowsRun = roundUp(g.rows, g.planBM)
	g.planeRows = roundUp(g.rows, planeAlign)
	// The position projection's M is 2T-1 rows of embeddings, and the
	// position scores' N is the same extent, so it is padded to whichever of
	// the two tiles covering it is wider.
	g.nRun = roundUp(2*g.rows-1, max(g.planBM, g.actGEMM.bn))
}

// Plan is the kernel currently chosen for each projection.
func (g *GPUEncoder) Plan() GEMMPlan { return g.plan }

// projShapes is [out, in] for each projection.
func (g *GPUEncoder) projShapes() map[Proj][2]int {
	d, f := g.dim, g.ffn
	return map[Proj][2]int{
		ProjFF1A: {f, d}, ProjFF1B: {d, f},
		ProjQ: {d, d}, ProjK: {d, d}, ProjV: {d, d}, ProjO: {d, d}, ProjRelK: {d, d},
		ProjPW1: {2 * d, d}, ProjPW2: {d, d},
		ProjFF2A: {f, d}, ProjFF2B: {d, f},
	}
}

// layoutWeights plans the two weight arenas and allocates them. Nothing is
// read here: the sizes follow from the config, so the whole 1.2 GB layout is
// decided -- and can be checked -- before the first weight is touched.
func (g *GPUEncoder) layoutWeights(layers int) error {
	shapes := g.projShapes()
	perLayer16 := 0
	for _, r := range projOrder {
		perLayer16 += shapes[r][0] * shapes[r][1]
	}
	// fp32, per layer: five norms with a weight and a bias, the two attention
	// biases, the depthwise filters and the folded BatchNorm affine.
	perLayer32 := 10*g.dim + 2*g.dim + g.dim*g.convK + 2*g.dim

	total16 := perLayer16 * layers
	total32 := perLayer32 * layers
	if total16*2 > maxBankBytes {
		return fmt.Errorf("parakeet: %d layers are %d MB of fp16 weights and a buffer holds %d MB",
			layers, (total16*2)>>20, maxBankBytes>>20)
	}

	g.w = make([]layerWeights, layers)
	var off16, off32 uint32
	take32 := func(n int) uint32 { o := off32; off32 += uint32(n); return o }
	for i := range g.w {
		w := &g.w[i]
		w.bOff = make(map[Proj]uint32, len(projOrder))
		for _, r := range projOrder {
			w.bOff[r] = off16
			off16 += uint32(shapes[r][0] * shapes[r][1])
		}
		for _, n := range []*[2]uint32{&w.normFF1, &w.normAttn, &w.normConv, &w.normFF2, &w.normOut} {
			n[0], n[1] = take32(g.dim), take32(g.dim)
		}
		w.biasU, w.biasV = take32(g.dim), take32(g.dim)
		w.dw = take32(g.dim * g.convK)
		w.bnScale, w.bnShift = take32(g.dim), take32(g.dim)
	}
	if int(off16) != total16 || int(off32) != total32 {
		return fmt.Errorf("parakeet: weight layout came to %d/%d, planned %d/%d", off16, off32, total16, total32)
	}
	// The subsampling stack is not per layer, so it claims what is left of
	// both arenas after the 24 layers have been laid out.
	off32, off16 = g.layoutSub(off32, off16)

	var err error
	if g.wbuf, err = g.dev.NewBuffer(int(off32) * 4); err != nil {
		return fmt.Errorf("parakeet: fp32 weight arena: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(int(off16) * 2); err != nil {
		return fmt.Errorf("parakeet: fp16 weight bank (%d MB): %w", (int(off16)*2)>>20, err)
	}
	return nil
}

// maxBankBytes is the largest storage buffer this device addresses
// (research/stage-2-vae-decoder.md). The whole encoder is 1.21 GB, so unlike
// the DiT's 12 GB it needs one bank and not three.
const maxBankBytes = 0xfffffffc

// allocActivations lays out the two activation arenas. Every tensor is sized
// for the padded frame count of the longest clip the encoder will take: the
// GEMM has no bounds check, so it writes whole tiles.
func (g *GPUEncoder) allocActivations() error {
	var off32 uint32
	take32 := func(n int) uint32 { o := off32; off32 += uint32(n); return o }
	rows, dim := g.rowsPad, g.dim
	g.aX = take32(rows * dim)
	g.aQ, g.aK, g.aV = take32(rows*dim), take32(rows*dim), take32(rows*dim)
	// One tensor for every branch output -- the two feed forwards, attention
	// and the convolution -- because each is consumed by the residual add
	// that immediately follows it. Keeping them apart would cost 3 MB and buy
	// nothing: RunTo is what the stagewise validation reads them with, and it
	// stops the graph before the next writer runs.
	g.aBranch = take32(rows * dim)
	g.aFF = take32(rows * g.ffn)
	g.aPW1 = take32(rows * 2 * dim)
	g.aGLU = take32(rows * dim)
	g.aRelK = take32(g.nPad * dim)
	g.aBD = take32(g.heads * rows * g.nPad)
	g.aBias = take32(g.heads * rows * rows)

	var off16 uint32
	take16 := func(n int) uint32 { o := off16; off16 += uint32(n); return o }
	g.hA = take16(rows * g.ldaDim)
	g.hFF = take16(rows * g.ldaFFN)
	g.hPW = take16(rows * g.ldaDim)
	g.hQV = take16(rows * g.ldaDim)
	g.hPos = take16(g.nPad * g.ldaDim)
	g.hRelK = take16(g.nPad * g.ldaDim)
	// The packed q/k/v planes: heads * rowsPad * headDim halves each, which
	// is rows*dim however the tiles are cut.
	g.hQ, g.hK, g.hV = take16(rows*dim), take16(rows*dim), take16(rows*dim)
	g.hCtx = take16(rows * g.ldaDim)

	// The subsampling stack's five intermediates, claimed after the layers'
	// so that a graph without them is byte for byte the graph S6 ran.
	off32, off16 = g.allocSub(off32, off16)
	g.actElems, g.hElems = int(off32), int(off16)

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("parakeet: fp32 activation arena: %w", err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("parakeet: fp16 activation arena: %w", err)
	}
	return nil
}

// build compiles every pipeline in the graph. All four arenas are bound to
// every pipeline, used or not, so that one descriptor layout and one
// push-constant size serve the whole sequence -- which is what
// vk.DispatchMultiTimed needs to record it into a single command buffer.
func (g *GPUEncoder) build() error {
	arenas := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}

	norm16 := shaders.ParakeetLayerNormF16
	if g.ctl.noMean {
		norm16 = shaders.ParakeetLayerNormF16NoMean
	}
	shift := shaders.ParakeetRelShift
	if g.ctl.shiftSlice {
		shift = shaders.ParakeetRelShiftSlice
	}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"layernorm", shaders.ParakeetLayerNorm},
		{"layernorm16", norm16},
		{"silu16", shaders.ParakeetSiLUF16},
		{"residual", shaders.ParakeetResidual},
		{"glu", shaders.ParakeetGLU},
		{"dwconv16", shaders.ParakeetDWConvF16},
		{"narrow16", shaders.ParakeetNarrowF16},
		{"relshift", shift},
		{"pack", shaders.DiTPackF16},
		{"packbias", shaders.ParakeetPackBias},
		{"subconv0", shaders.ParakeetSubConv0},
		{"subdw", shaders.ParakeetSubDW},
		{"subbias", shaders.ParakeetSubBias},
		{"subflatten", shaders.ParakeetSubFlattenF16},
	} {
		if err := g.pipeline(s.name, s.spirv, spec); err != nil {
			return err
		}
	}
	attnSpec := spec
	attnSpec.RequiredSubgroupSize = g.attn.wave
	if err := g.pipeline("attention", g.attn.spirv, attnSpec); err != nil {
		return err
	}

	for _, v := range gemmVariants {
		ok, err := g.canPin(v.wave)
		if err != nil {
			return err
		}
		if !ok {
			// A rung whose wave size this device will not pin is dropped
			// whole rather than run at the driver's default, which would be
			// a different kernel wearing the same name.
			delete(g.kernels, v.name)
			continue
		}
		gspec := spec
		gspec.RequiredSubgroupSize = v.wave
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("parakeet: gemm %s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, gspec)
		if err != nil {
			return fmt.Errorf("parakeet: gemm %s: %w", v.name, err)
		}
		g.gemms[v.name] = pipe
	}
	if _, ok := g.kernels[g.plan[ProjQ]]; !ok {
		// The plan named a rung this device dropped; fall back to the widest
		// wave64 tile, which every device that reaches here can run.
		if err := g.SetPlan(UniformGEMMPlan(GEMMReg32x64)); err != nil {
			return err
		}
	}
	for _, r := range subProjOrder {
		if _, ok := g.kernels[g.subPlan[r]]; !ok {
			if err := g.SetSubPlan(UniformSubPlan(GEMMReg32x64)); err != nil {
				return err
			}
			break
		}
	}

	// The position term's GEMM reads its B operand out of the fp16
	// *activation* arena, so its descriptor set binds that buffer where the
	// others bind the weight bank. Same shader, same push constants, one more
	// descriptor set.
	actSpec := spec
	actSpec.Buffers = []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.hbuf}
	for _, v := range actGEMMVariants {
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("parakeet: act gemm %s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, actSpec)
		if err != nil {
			return fmt.Errorf("parakeet: act gemm %s: %w", v.name, err)
		}
		g.actGEMMs[v.name] = pipe
	}
	return nil
}

func (g *GPUEncoder) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("parakeet: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("parakeet: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stageWeights fills the arenas: fp32 for everything consumed elementwise
// (the norms, the two attention biases, the depthwise filters and the folded
// BatchNorm), fp16 fragment tiles for the eleven matrices the cooperative
// matrices read.
func (g *GPUEncoder) stageWeights(enc *Encoder) error {
	shapes := g.projShapes()
	for i, layer := range enc.Layers {
		w := &g.w[i]
		for _, n := range []struct {
			off [2]uint32
			ln  *LayerNorm
		}{
			{w.normFF1, layer.NormFF1}, {w.normAttn, layer.NormAttn},
			{w.normConv, layer.NormConv}, {w.normFF2, layer.NormFF2},
			{w.normOut, layer.NormOut},
		} {
			if len(n.ln.Weight) != g.dim || len(n.ln.Bias) != g.dim {
				return fmt.Errorf("parakeet: layer %d has a norm of width %d, want %d", i, len(n.ln.Weight), g.dim)
			}
			g.wbuf.WriteFloat32At(int(n.off[0]), n.ln.Weight)
			g.wbuf.WriteFloat32At(int(n.off[1]), n.ln.Bias)
		}
		g.wbuf.WriteFloat32At(int(w.biasU), layer.Attn.BiasU)
		g.wbuf.WriteFloat32At(int(w.biasV), layer.Attn.BiasV)
		if layer.Conv.Kernel != g.convK {
			return fmt.Errorf("parakeet: layer %d has conv kernel %d, layer 0 has %d", i, layer.Conv.Kernel, g.convK)
		}
		g.wbuf.WriteFloat32At(int(w.dw), layer.Conv.DW)
		g.wbuf.WriteFloat32At(int(w.bnScale), layer.Conv.BNScale)
		g.wbuf.WriteFloat32At(int(w.bnShift), layer.Conv.BNShift)

		for _, r := range projOrder {
			lin, err := layerProj(layer, r)
			if err != nil {
				return err
			}
			n, k := shapes[r][0], shapes[r][1]
			if lin.Out != n || lin.In != k {
				return fmt.Errorf("parakeet: layer %d %s is [%d %d], want [%d %d]", i, r, lin.Out, lin.In, n, k)
			}
			if lin.Bias != nil {
				return fmt.Errorf("parakeet: layer %d %s has a bias, and the GEMM has nowhere to put one", i, r)
			}
			dst := make([]uint16, n*k)
			packB(dst, lin.Weight, n, k)
			g.bank.WriteUint16At(int(w.bOff[r]), dst)
		}
	}
	return g.stageSub(enc.Subsampling)
}

// layerProj is the matrix a projection names inside a layer.
func layerProj(l *EncoderLayer, r Proj) (*Linear, error) {
	switch r {
	case ProjFF1A:
		return l.FF1.Linear1, nil
	case ProjFF1B:
		return l.FF1.Linear2, nil
	case ProjQ:
		return l.Attn.Q, nil
	case ProjK:
		return l.Attn.K, nil
	case ProjV:
		return l.Attn.V, nil
	case ProjO:
		return l.Attn.O, nil
	case ProjRelK:
		return l.Attn.RelK, nil
	case ProjPW1:
		return l.Conv.PW1, nil
	case ProjPW2:
		return l.Conv.PW2, nil
	case ProjFF2A:
		return l.FF2.Linear1, nil
	case ProjFF2B:
		return l.FF2.Linear2, nil
	}
	return nil, fmt.Errorf("parakeet: no projection %q", r)
}

// packB narrows a [n, k] row-major PyTorch weight into the fragment-tile B
// layout (§2.8): tile (nt, kt) is 256 contiguous halves holding element
// (k, n) at (n%16)*16 + k%16, tiles ordered kt-fastest, which is the order
// the kernel walks and what makes a fragment load cover 512 B.
//
// This runs once per layer at upload, which is what makes the layout free:
// the retiling that would cost an activation a pass per clip costs a weight
// nothing per clip.
func packB(dst []uint16, w []float32, n, k int) {
	const tile = coopMatTile
	const chunk = 64
	chunks := (n + chunk - 1) / chunk
	kt := k / tile
	parallelFor(chunks, func(c int) {
		for i := c * chunk; i < min((c+1)*chunk, n); i++ {
			row := w[i*k : (i+1)*k]
			base := (i / tile) * kt * tile * tile
			lane := (i % tile) * tile
			for j, v := range row {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

// Destroy releases every Vulkan object.
func (g *GPUEncoder) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, p := range g.gemms {
		p.Destroy()
	}
	for _, p := range g.actGEMMs {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.gemms, g.actGEMMs, g.mods = nil, nil, nil, nil
	g.wbuf, g.abuf, g.hbuf, g.bank = nil, nil, nil, nil
}

// perSubmit is how many dispatches go into one command buffer. The DiT keeps
// this at 8 because one of its blocks is 50 ms of work and a whole graph in
// one buffer outlives the driver's reset watchdog; a layer here is ~0.3 ms
// and 40 dispatches, so the constraint is the other way round -- a submit per
// handful of dispatches would charge 960 fence waits to an encoder that has
// 7 ms of work in it.
const perSubmit = 240

// submit runs a recorded dispatch sequence in watchdog-sized batches.
func submit(d []vk.MultiDispatch) error {
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("parakeet: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

// Apply runs the whole encoder over x, the subsampled and scaled frames, and
// returns the hidden states. valid is how many of x's rows are not padding.
func (g *GPUEncoder) Apply(x *Mat, valid int) (*Mat, error) {
	if err := g.Upload(x, valid); err != nil {
		return nil, err
	}
	if err := g.Run(); err != nil {
		return nil, err
	}
	return g.Read(g.aX, g.dim), nil
}

// ApplyMel is the whole encoder from the front end's output: the subsampling
// stack and the 24 layers, in one dispatch sequence, with nothing crossing
// the bus but the mel spectrogram in and the hidden states out. Its signature
// is Encoder.Apply's, so a caller switches between the two by switching which
// one it calls.
//
// HostSubsampling runs the convolutions on the CPU instead, which is what S6
// did and what the stage is measured against.
func (g *GPUEncoder) ApplyMel(mel *Mat, validMel int) (*Mat, int, error) {
	if g.HostSubsampling {
		x, valid, err := g.sub.Apply(mel, validMel)
		if err != nil {
			return nil, 0, err
		}
		if g.scaleInput {
			s := float32(math.Sqrt(float64(g.dim)))
			for i := range x.Data {
				x.Data[i] *= s
			}
		}
		h, err := g.Apply(x, valid)
		if err != nil {
			return nil, 0, err
		}
		return h, valid, nil
	}
	if err := g.UploadMel(mel, validMel); err != nil {
		return nil, 0, err
	}
	if err := g.Run(); err != nil {
		return nil, 0, err
	}
	return g.Read(g.aX, g.dim), g.valid, nil
}

// RunMel is ApplyMel without the read-back: it leaves the hidden states in the
// residual stream and returns only how many frames are valid.
//
// It is what the resident pipeline uses (GPUDecoder.Attach). Reading [T, 1024]
// back out of the arena is 2.4 ms at 552 KB -- device-local host-visible
// memory reads at 0.2 GB/s on this part -- which is half again what the whole
// decode loop costs, and nothing on the device needs it back.
func (g *GPUEncoder) RunMel(mel *Mat, validMel int) (int, error) {
	if err := g.UploadMel(mel, validMel); err != nil {
		return 0, err
	}
	if err := g.Run(); err != nil {
		return 0, err
	}
	return g.valid, nil
}

// UploadMel writes the clip's log-mel spectrogram into the arena and settles
// the run's geometry from it -- the encoder frame count, which the caller no
// longer computes, the plan, and the position embeddings.
//
// It leaves the subsampling stack pending rather than dispatching it, so that
// Run and Profile see one graph: the stack's ten dispatches in front of the
// 960 the layers are.
func (g *GPUEncoder) UploadMel(mel *Mat, validMel int) error {
	if mel.Cols != g.maxGeom.melBins {
		return fmt.Errorf("parakeet: the mel spectrogram is %s, want [rows %d]", mel, g.maxGeom.melBins)
	}
	if mel.Rows <= 0 || mel.Rows > g.maxGeom.melRows {
		return fmt.Errorf("parakeet: %d mel frames; the encoder was built for at most %d", mel.Rows, g.maxGeom.melRows)
	}
	if validMel < 0 || validMel > mel.Rows {
		return fmt.Errorf("parakeet: %d valid frames of %d", validMel, mel.Rows)
	}
	geom := subGeometry(mel.Rows, mel.Cols, validMel)
	if geom.t[2] > g.frames {
		return fmt.Errorf("parakeet: %d mel frames are %d encoder frames; the encoder was built for at most %d",
			mel.Rows, geom.t[2], g.frames)
	}
	g.geom = geom
	g.rows, g.valid = geom.t[2], geom.v[2]
	if g.AutoPlan {
		auto := g.AutoPlan
		if err := g.SetPlan(PlanFor(g.rows)); err != nil {
			return err
		}
		g.AutoPlan = auto
	}
	g.setExtents()
	g.abuf.WriteFloat32At(int(g.sMel), mel.Data)
	g.subPending = true
	return g.writePositions()
}

// Upload writes the run's input into the residual stream and computes the
// clip-length things that go with it: the padded extents, the plan, and the
// sinusoidal position embeddings the position term projects.
func (g *GPUEncoder) Upload(x *Mat, valid int) error {
	if x.Cols != g.dim {
		return fmt.Errorf("parakeet: x is %s, want [rows %d]", x, g.dim)
	}
	if x.Rows <= 0 || x.Rows > g.frames {
		return fmt.Errorf("parakeet: x has %d rows; the encoder was built for at most %d", x.Rows, g.frames)
	}
	if valid < 0 || valid > x.Rows {
		return fmt.Errorf("parakeet: %d valid frames of %d", valid, x.Rows)
	}
	g.rows, g.valid = x.Rows, valid
	g.subPending = false
	if g.AutoPlan {
		auto := g.AutoPlan
		if err := g.SetPlan(PlanFor(x.Rows)); err != nil {
			return err
		}
		g.AutoPlan = auto
	}
	g.setExtents()
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	return g.writePositions()
}

// writePositions narrows the [2T-1, dim] relative-position embedding into the
// fp16 arena as the position projection's A operand.
//
// It is computed on the host in float64 and rounded once, as the CPU
// reference does: torch holds the angle in float32, and one ulp at an offset
// of 137 frames is already 1e-5 (SPEECH.md's accuracy table). The rows past
// 2T-1 are written as zeros, because the GEMM covers whole tiles and the
// product of those rows is what the position term's B operand will be.
func (g *GPUEncoder) writePositions() error {
	p := 2*g.rows - 1
	buf := make([]uint16, g.nRun*g.ldaDim)
	dim := g.dim
	parallelFor(g.nRun, func(r int) {
		if r >= p {
			return
		}
		pos := float64(g.rows - 1 - r)
		row := buf[r*g.ldaDim:]
		for i := 0; i < dim/2; i++ {
			freq := pos / math.Pow(10000, float64(2*i)/float64(dim))
			row[2*i] = safetensors.F32ToF16(float32(math.Sin(freq)))
			row[2*i+1] = safetensors.F32ToF16(float32(math.Cos(freq)))
		}
	})
	g.hbuf.WriteUint16At(int(g.hPos), buf)
	g.posRows = g.nRun
	return nil
}

// Run runs every layer over whatever the residual stream already holds.
//
// The whole stack is recorded first and submitted in watchdog-sized batches,
// rather than a submit per layer. A layer is 40 dispatches and ~0.6 ms, so a
// submit per layer is 24 fence waits for work that does not need them: the
// measured cost of that was 3 ms on a 14 ms encoder, i.e. a fifth of the
// stack spent waiting for the host.
func (g *GPUEncoder) Run() error {
	var d []vk.MultiDispatch
	if g.subPending {
		sub, _, err := g.subGraph()
		if err != nil {
			return err
		}
		d = append(d, sub...)
	}
	for i := range g.w {
		layer, _, err := g.layerGraph(i)
		if err != nil {
			return err
		}
		d = append(d, layer...)
	}
	return submit(d)
}

// Stage is one dispatch's GPU time, as Profile reports it.
type Stage struct {
	Index int
	Layer int
	Kind  string
	GPU   time.Duration
}

// Profile runs the layer stack one dispatch at a time and times each on the
// GPU, from the subsampled input. Wall clock around Apply is not a
// measurement of the encoder: it also carries the host write of x and the
// read-back of the result.
func (g *GPUEncoder) Profile(x *Mat, valid int) ([]Stage, *Mat, error) {
	if err := g.Upload(x, valid); err != nil {
		return nil, nil, err
	}
	return g.profileLayers(nil)
}

// ProfileMel is Profile over the whole encoder, subsampling stack included.
// The stack's dispatches are reported at layer -1, which is what distinguishes
// them from the 24 layers' in a caller that totals by kind.
func (g *GPUEncoder) ProfileMel(mel *Mat, validMel int) ([]Stage, *Mat, error) {
	if err := g.UploadMel(mel, validMel); err != nil {
		return nil, nil, err
	}
	d, kinds, err := g.subGraph()
	if err != nil {
		return nil, nil, err
	}
	stages, err := g.timeEach(nil, -1, d, kinds)
	if err != nil {
		return stages, nil, err
	}
	return g.profileLayers(stages)
}

func (g *GPUEncoder) profileLayers(stages []Stage) ([]Stage, *Mat, error) {
	for l := range g.w {
		d, kinds, err := g.layerGraph(l)
		if err != nil {
			return nil, nil, err
		}
		if stages, err = g.timeEach(stages, l, d, kinds); err != nil {
			return stages, nil, err
		}
	}
	return stages, g.Read(g.aX, g.dim), nil
}

func (g *GPUEncoder) timeEach(stages []Stage, layer int, d []vk.MultiDispatch, kinds []string) ([]Stage, error) {
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true)
		if err != nil {
			return stages, fmt.Errorf("parakeet: layer %d dispatch %d (%s): %w", layer, i, kinds[i], err)
		}
		stages = append(stages, Stage{Index: len(stages), Layer: layer, Kind: kinds[i], GPU: dur})
	}
	return stages, nil
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

// RunTo runs the graph from the clip's input up to and including the dispatch
// with the given label in the given layer, and leaves the arenas as that
// dispatch left them.
//
// It is what makes the validation stagewise: most of a layer's intermediates
// are overwritten before it ends -- the four branch outputs share one tensor
// and every norm writes the same fp16 operand -- so a test that only ran
// Apply could compare a handful of the reference's tensors. Stopping early
// costs one re-run of the prefix per stage.
//
// It takes the input rather than reusing what is in the arena, and that is
// not an interface convenience: the residual stream is updated *in place*, so
// running a prefix twice over the arena a previous prefix left would add
// every residual in it twice. Nothing about that failure looks like a
// dispatch being wrong.
func (g *GPUEncoder) RunTo(x *Mat, valid, layer int, label string) error {
	if layer < 0 || layer >= len(g.w) {
		return fmt.Errorf("parakeet: layer %d out of range, the encoder has %d", layer, len(g.w))
	}
	if err := g.Upload(x, valid); err != nil {
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
		return fmt.Errorf("parakeet: no dispatch labelled %q (have %v)", label, kinds)
	}
	return submit(d[:end])
}

// Read copies one tensor out of the fp32 activation arena, rows deep and cols
// wide, by the offset the Tensor* accessors name.
func (g *GPUEncoder) Read(off uint32, cols int) *Mat {
	out := NewMat(g.rows, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), g.rows*cols))
	return out
}

// ReadRows is Read over a stated number of rows, for the tensors whose row
// count is not the clip's -- the position projection's 2T-1.
func (g *GPUEncoder) ReadRows(off uint32, rows, cols int) *Mat {
	out := NewMat(rows, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), rows*cols))
	return out
}

// ReadF16 copies a [rows, cols] tensor out of the fp16 activation arena,
// widening it. The row stride is the operand's leading dimension, not cols:
// an A operand's rows are padded (§2.3).
func (g *GPUEncoder) ReadF16(off uint32, rows, cols, lda int) *Mat {
	raw := g.hbuf.ReadUint16At(int(off), (rows-1)*lda+cols)
	out := NewMat(rows, cols)
	for r := 0; r < rows; r++ {
		row, src := out.Row(r), raw[r*lda:]
		for c := 0; c < cols; c++ {
			row[c] = safetensors.F16ToF32(src[c])
		}
	}
	return out
}

// The stage-boundary tensors, for the validation walk.
func (g *GPUEncoder) TensorX() uint32      { return g.aX }
func (g *GPUEncoder) TensorQ() uint32      { return g.aQ }
func (g *GPUEncoder) TensorK() uint32      { return g.aK }
func (g *GPUEncoder) TensorV() uint32      { return g.aV }
func (g *GPUEncoder) TensorBranch() uint32 { return g.aBranch }
func (g *GPUEncoder) TensorFF() uint32     { return g.aFF }
func (g *GPUEncoder) TensorPW1() uint32    { return g.aPW1 }
func (g *GPUEncoder) TensorGLU() uint32    { return g.aGLU }
func (g *GPUEncoder) TensorRelK() uint32   { return g.aRelK }
func (g *GPUEncoder) TensorBD() uint32     { return g.aBD }
func (g *GPUEncoder) TensorBias() uint32   { return g.aBias }

// The fp16 operands, which is where four of the five norms and both branch
// activations land.
func (g *GPUEncoder) TensorA() uint32   { return g.hA }
func (g *GPUEncoder) TensorFFA() uint32 { return g.hFF }
func (g *GPUEncoder) TensorPWA() uint32 { return g.hPW }
func (g *GPUEncoder) TensorCtx() uint32 { return g.hCtx }
func (g *GPUEncoder) LDA() int          { return g.ldaDim }
func (g *GPUEncoder) LDAFFN() int       { return g.ldaFFN }

// Rows is the length of the run the arenas currently hold, PosRows the
// position term's 2T-1 and RowsPadded the tile-aligned frame count every
// GEMM actually covers.
func (g *GPUEncoder) Rows() int         { return g.rows }
func (g *GPUEncoder) Valid() int        { return g.valid }
func (g *GPUEncoder) PosRows() int      { return 2*g.rows - 1 }
func (g *GPUEncoder) RowsPadded() int   { return g.rowsRun }
func (g *GPUEncoder) PlaneRows() int    { return g.planeRows }
func (g *GPUEncoder) PosPadded() int    { return g.nRun }
func (g *GPUEncoder) Layers() int       { return len(g.w) }
func (g *GPUEncoder) Dim() int          { return g.dim }
func (g *GPUEncoder) Heads() int        { return g.heads }
func (g *GPUEncoder) MaxFrames() int    { return g.frames }
func (g *GPUEncoder) Attention() string { return g.attn.name }

// WeightBytes is what the staged weights cost on the device and
// ActivationBytes what the shared arenas do.
func (g *GPUEncoder) WeightBytes() int     { return g.wbuf.Size() + g.bank.Size() }
func (g *GPUEncoder) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// FLOPs is the encoder's multiply-add count at a frame count, counting the
// eleven projections and both attention matmuls -- the same accounting
// §3.4 uses, which leaves the elementwise passes out as noise.
func (g *GPUEncoder) FLOPs(rows int) float64 {
	t, p := float64(rows), float64(2*rows-1)
	dim, ffn := float64(g.dim), float64(g.ffn)
	perLayer := 2 * (2 * t * dim * ffn * 2) // two feed forwards, two matrices each
	perLayer += 2 * t * dim * dim * 4       // q, k, v, o
	perLayer += 2 * p * dim * dim           // the position projection
	perLayer += 2 * t * dim * (2*dim + dim) // the two pointwise convolutions
	perLayer += 2 * t * t * dim * 2         // content scores and context
	perLayer += 2 * t * p * dim             // the position scores
	return perLayer * float64(len(g.w))
}
