package llm

// The gated DeltaNet on the device: LLM.md L3b, and the GPU half of the
// reference L3a built in deltanet.go.
//
// Thirty-six of the 48 layers are this one, and it is the only block in the
// model with a loop-carried dependency as long as the prompt. L2a priced the
// nameable lines of it at **2.7% of llama.cpp's 512-token prefill graph**,
// spread over four projections, a convolution, two L2 norms, the recurrence
// and a gated norm:
//
//	MUL_MAT q8_0 m=10240 k=2560   36   the fused q/k/v projection
//	MUL_MAT q8_0 m=6144  k=2560   36   the output gate z
//	MUL_MAT f32  m=48    k=2560   72   ssm_alpha and ssm_beta
//	SSM_CONV_SILU                 36    6.8 ms   depthwise k=4, then SiLU
//	L2_NORM                       72    4.5 ms   q and k, per head of 128
//	SOFTPLUS, MUL, SIGMOID, CONT  ~180           the two per-head scalars
//	GATED_DELTA_NET               36   15.8 ms   the recurrence
//	RMS_NORM, MUL, SIGMOID, MUL   ~144           the gated output norm
//	MUL_MAT q8_0 m=2560  k=6144   36   the output projection
//
// **Five dispatches replace those**, and every fusion is the argument L2a
// made for `inject`, L2d for the PLE block's value and L2f for the attention
// layer's six matrices: two matmuls that read the same activation are one
// matmul with more columns.
//
//	qkv    xn -> q, k, v, z, alpha, beta          (the plain GEMM arm)
//	conv   depthwise conv + SiLU + L2 norm + the two per-head scalars
//	scan   the delta rule, state in registers across the token loop
//	norm   the gated RMS norm, into the output matmul's A operand
//	out    the output projection                  (the plain GEMM again)
//
// **Four of llama.cpp's matrices are one.** `attn_qkv` [2560, 10240],
// `attn_gate` [2560, 6144] and the two [2560, 48] F32 gate projections all
// read the same block input, so they are one [16512, 2560] weight. The two
// F32 ones are the interesting half: they are L2a's "tiny-N F32 matmul"
// scandal in miniature — 72 dispatches a graph for 0.25 MB of weights — and
// here they are 96 more columns on a matrix that already has 16384.
//
// That fusion puts them on the fp16 matrix cores, which is a **deviation from
// the 7-token dump and not from the reference**: L3a-5 found that
// `ggml_vk_mul_mat` takes the f32 vector path only up to
// `mul_mat_vec_max_cols = 8` output columns, so at any real ubatch llama.cpp
// evaluates alpha and beta in fp16 too. The dump has 7 tokens and therefore
// does not, which is why `TestDeltaNetGPUGateIsTheFp16Path` measures the gap
// rather than hiding it.
//
// What is **not** here is the chunked form. L3a-2 found llama.cpp running a
// sequential scan and declining its own `build_delta_net_chunking`, and vLLM
// doing the opposite; this file builds the scan, measures its whole ladder,
// and research/l3b-deltanet-gpu.md prices the chunk against what the scan
// turns out to cost.

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// dnBN is the compiled column block of the plain GEMM arm, which both of this
// layer's projections run on. Same kernel as the attention layer's.
const dnBN = attnBN

// The decode rungs of the two projections, from results/l8d_dn.csv (L8d-4),
// and they are different numbers for the reason D12 gives. A slab is
// `(gemmK/16/KSLABS) * 256` bytes on the int8 bank. The fused projection
// reduces over nEmbd = 2560 — 160 k-tiles, so 1 and 2 slabs are 10 and 5 whole
// multiples of 4 KB and measure 276 and 309 us against 4, 8, 20 and 40's
// 240/196/225/207. The output projection reduces over Inner = 6144, 384
// tiles, where 1, 4 and 8 slabs are whole multiples (32.8, 25.8 and 30.7 us)
// and 32 is 0.75 of one: **22.0**.
const dnQKVGemv = GEMVK8
const dnOutGemv = GEMVK32

// dnWave is the wave the scan is pinned to, and the width every rung's
// LPC divides.
const dnWave = 64

// DNKernel names one build of shaders/llm_dn_scan.comp, by its (LPC, WAVES).
//
// LPC is how many lanes own one column of the [128, 128] state and it moves
// three things at once: a lane holds 128/LPC state elements in registers, a
// wave owns 64/LPC columns, and every workgroup of a head re-reads that
// head's whole q and k row once per token — so the re-read factor is
// 128/(COLS*WAVES). The top rung is llama.cpp's own shape, and it is the
// **worst** one here: the ladder is a bandwidth-to-ALU crossover that turns at
// LPC 8, and its ends are 6.3x apart on identical arithmetic
// (research/l3b-deltanet-gpu.md). The `…s` suffix forces the LDS arm and
// `…p` a one-deep software pipeline, both of which are ablations rather than
// candidates — the second is worth nothing anywhere.
type DNKernel string

const (
	DNScanL64   DNKernel = "l64"
	DNScanL32   DNKernel = "l32"
	DNScanL16   DNKernel = "l16"
	DNScanL8    DNKernel = "l8"
	DNScanL4    DNKernel = "l4"
	DNScanL2    DNKernel = "l2"
	DNScanL1    DNKernel = "l1"
	DNScanL16W4 DNKernel = "l16w4"
	DNScanL8W4  DNKernel = "l8w4"
	DNScanL4W4  DNKernel = "l4w4"
	DNScanL2W4  DNKernel = "l2w4"
	// The LDS arm of four of those shapes, which is what prices QKREG.
	DNScanL64S DNKernel = "l64s"
	DNScanL16S DNKernel = "l16s"
	DNScanL8S  DNKernel = "l8s"
	DNScanL4S  DNKernel = "l4s"
	// And four with the loads of token t+1 issued before the arithmetic of
	// token t, which is what an SSBO the kernel also writes to costs.
	DNScanL64P DNKernel = "l64p"
	DNScanL16P DNKernel = "l16p"
	DNScanL8P  DNKernel = "l8p"
	DNScanL4P  DNKernel = "l4p"
)

type dnVariant struct {
	name       DNKernel
	spirv      []byte
	lpc, waves int
	// state is how many state elements a lane holds; qkreg says whether the
	// 2*state operand elements sit beside them or go through LDS.
	state    int
	qkreg    bool
	prefetch bool
}

var dnVariants = []dnVariant{
	{DNScanL64, shaders.LLMDNScanL64, 64, 1, 2, true, false},
	{DNScanL32, shaders.LLMDNScanL32, 32, 1, 4, true, false},
	{DNScanL16, shaders.LLMDNScanL16, 16, 1, 8, true, false},
	{DNScanL8, shaders.LLMDNScanL8, 8, 1, 16, true, false},
	{DNScanL4, shaders.LLMDNScanL4, 4, 1, 32, true, false},
	{DNScanL2, shaders.LLMDNScanL2, 2, 1, 64, false, false},
	{DNScanL1, shaders.LLMDNScanL1, 1, 1, 128, false, false},
	{DNScanL16W4, shaders.LLMDNScanL16W4, 16, 4, 8, true, false},
	{DNScanL8W4, shaders.LLMDNScanL8W4, 8, 4, 16, true, false},
	{DNScanL4W4, shaders.LLMDNScanL4W4, 4, 4, 32, true, false},
	{DNScanL2W4, shaders.LLMDNScanL2W4, 2, 4, 64, false, false},
	{DNScanL64S, shaders.LLMDNScanL64S, 64, 1, 2, false, false},
	{DNScanL16S, shaders.LLMDNScanL16S, 16, 1, 8, false, false},
	{DNScanL8S, shaders.LLMDNScanL8S, 8, 1, 16, false, false},
	{DNScanL4S, shaders.LLMDNScanL4S, 4, 1, 32, false, false},
	{DNScanL64P, shaders.LLMDNScanL64P, 64, 1, 2, true, true},
	{DNScanL16P, shaders.LLMDNScanL16P, 16, 1, 8, true, true},
	{DNScanL8P, shaders.LLMDNScanL8P, 8, 1, 16, true, true},
	{DNScanL4P, shaders.LLMDNScanL4P, 4, 1, 32, true, true},
}

// DNKernels lists the rungs, the reference's shape first.
func DNKernels() []DNKernel {
	out := make([]DNKernel, len(dnVariants))
	for i, v := range dnVariants {
		out[i] = v.name
	}
	return out
}

// DefaultDNKernel is the measured winner of that ladder: 16 state elements a
// lane, 768 workgroups, q and k re-read 16 times, and **413 us a layer at 512
// tokens against llama.cpp's 438**. It wins or ties at every length from 64 to
// 2048, with l8w4 inside the 1-3% run-to-run spread.
// See research/l3b-deltanet-gpu.md and results/l3b_dn.csv.
func DefaultDNKernel() DNKernel { return DNScanL8 }

// cols is how many state columns one workgroup of this rung owns, and so
// 128/cols is both its workgroup count per head and its q/k re-read factor.
func (v dnVariant) cols() int { return dnWave / v.lpc * v.waves }

func dnVariantFor(k DNKernel) (dnVariant, bool) {
	for _, v := range dnVariants {
		if v.name == k {
			return v, true
		}
	}
	return dnVariant{}, false
}

// dnLayerWeights is where one layer's staged weights sit.
type dnLayerWeights struct {
	// qkv and out are halves into the fp16 bank, or **bytes** into L8's int8
	// one — the arm of llm_gemm.comp that reads it takes bOff as a byte
	// offset, because a matrix there is n*k bytes of tiles followed by its
	// scale plane and the kernel derives the second from the first.
	qkv, out  uint32
	conv      uint32 // fp32 arena: [convWidth][kern], channel-major
	gamma     uint32 // fp32 arena: ssm_norm, [headDim], shared by all 48 heads
	a, dtBias uint32 // fp32 arena: [nHeadV] each
	state     uint32 // fp32 activation arena: [nHeadV][headDim][headDim]
	win       uint32 // fp32 activation arena: [Conv-1][qkvN], a ring by position
}

// DeltaNetGPU runs linear-attention layers on the device. It holds however
// many layers it was staged with — at L3b that is the three the trace covers,
// at L6 it will be all 36 — plus one set of activation arenas sized for the
// longest prompt, and one recurrent state per layer.
type DeltaNetGPU struct {
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
	dev *vk.Device
	cfg DeltaNetConfig

	wbuf, abuf, hbuf, wbank *vk.Buffer
	pipes                   map[string]*vk.ComputePipeline
	mods                    []*vk.ShaderModule

	// bank is which width this block's two projections are staged in: the
	// halves every block ran on before L8, L8a's int8-plus-scales, or
	// L8c-5's 4.5-bit K-quant. 36 of the 48 layers are this block's, and its
	// two projections are 4.18 GB of a decode token's 9.67 as halves — the
	// largest single thing L8 narrows, at every width.
	bank DenseBank
	// sim is the format the 4.5-bit bank quantises to (`q4_k/32` and a
	// mode), and is unread on the other two. It is a QuantSim because that
	// is what L8c-3 measured the format with, and one encoder serves both
	// (quantk.go).
	sim QuantSim
	// imLayer maps a staged layer to its `blk.N.` prefix, which is what the
	// published imatrix is keyed by: the fused projection's two quantised
	// sources are different tensors with different importance rows, and the
	// simulation this bank has to reproduce looked each of them up by name.
	imLayer []int

	layers []dnLayerWeights

	scan          DNKernel
	gemm, outGemm GEMMKernel
	// The decode rungs of the same two projections: llm_gemv.comp at one
	// token, GEMVOff at every other length (L8d-4).
	qkvGemv, outGemv GEMVKernel
	// pinGemv holds both projections on llm_gemm.comp whatever the batch, so
	// that a prompt run in chunks is the prompt run whole to the last place
	// (Graph.PinSchedule). The GEMV is the second rung in this vertical that
	// is not bit-exact against its siblings, for L7d's reason: a split sum is
	// a different association of the same products.
	pinGemv bool
	// autoPlan re-chooses both GEMM rungs per run. SetPlan turns it off,
	// because a caller that named a rung meant it.
	autoPlan bool
	// keepSilu writes the convolution's un-normalised SiLU output, which is
	// `conv_output_silu` and is otherwise never materialised.
	keepSilu bool

	tokens, arenaRows, rows int
	lda, ldCtx              int

	// fp32 weight arena: conv taps, the head norm and the two per-head
	// vectors, per layer.
	wElems int

	// fp32 activations. aQKV is the fused projection's output; the
	// convolution's Conv-1 history rows are a ring **of their own, per
	// layer**, at aWin (L7b) rather than in front of it, because that arena
	// is shared by every staged layer and a carried window is not.
	aQKV                          uint32
	aWin                          uint32
	aNorm, aSilu, aOut            uint32
	aGate, aBeta, aState, aResult uint32
	aPart                         uint32
	actElems                      int
	// past is how many tokens of this sequence are behind the run, which is
	// what turns the ring's absolute positions into slots. Zero is a fresh
	// sequence, where a tap reaching before token zero contributes nothing.
	past int
	// fp16 activations.
	hXn, hCtx uint32
	hElems    int
}

// quant is whether the bank is one of the two the kernel unpacks — which is
// what every line that is about "int8 or halves" rather than about a
// particular width asks.
func (g *DeltaNetGPU) quant() bool { return g.bank != BankFP16 }

// qkvN is the fused projection's output width: the convolution's 10240
// channels, the output gate's 6144, and the two per-head scalars, padded up
// to the GEMM's column block. Four of llama.cpp's matrices, one of ours.
func (g *DeltaNetGPU) qkvN() int {
	c := g.cfg
	return roundUpInt(c.ConvWidth()+c.Inner+2*c.NHeadV, dnBN)
}

// The column each of the fused projection's outputs starts at. The shaders
// derive the same numbers from heads, kvHeads and headDim, so these are the
// host's half of that contract and the tests check them.
func (g *DeltaNetGPU) colQ() int     { return 0 }
func (g *DeltaNetGPU) colK() int     { return g.cfg.QKWidth() }
func (g *DeltaNetGPU) colV() int     { return 2 * g.cfg.QKWidth() }
func (g *DeltaNetGPU) colZ() int     { return g.cfg.ConvWidth() }
func (g *DeltaNetGPU) colAlpha() int { return g.cfg.ConvWidth() + g.cfg.Inner }
func (g *DeltaNetGPU) colBeta() int  { return g.colAlpha() + g.cfg.NHeadV }

// ColQ, ColK, ColV, ColZ, ColAlpha and ColBeta are those columns, for a test
// that wants to slice one tensor out of the fused output.
func (g *DeltaNetGPU) ColQ() int     { return g.colQ() }
func (g *DeltaNetGPU) ColK() int     { return g.colK() }
func (g *DeltaNetGPU) ColV() int     { return g.colV() }
func (g *DeltaNetGPU) ColZ() int     { return g.colZ() }
func (g *DeltaNetGPU) ColAlpha() int { return g.colAlpha() }
func (g *DeltaNetGPU) ColBeta() int  { return g.colBeta() }

// NewDeltaNetGPU stages layers onto the device and builds every pipeline.
//
// maxTokens is the longest prompt the arenas are built for. Every staged
// layer gets its own recurrent state, zeroed here, because the state is the
// layer's and not the batch's.
func NewDeltaNetGPU(dev *vk.Device, cfg DeltaNetConfig, maxTokens int, layers []DeltaNetWeights, q8 bool) (*DeltaNetGPU, error) {
	return NewDeltaNetGPUBank(dev, cfg, maxTokens, layers, bankOf(q8), QuantSim{})
}

// NewDeltaNetGPUBank is the same, with the bank named rather than implied —
// and with the format, for the one bank that has one to choose (L8c-5).
func NewDeltaNetGPUBank(dev *vk.Device, cfg DeltaNetConfig, maxTokens int,
	layers []DeltaNetWeights, bank DenseBank, sim QuantSim) (*DeltaNetGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: %d tokens", maxTokens)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("llm: no linear-attention layers to stage")
	}
	if cfg.HeadDim != 128 {
		return nil, fmt.Errorf("llm: the scan is built for headDim 128, this checkpoint says %d", cfg.HeadDim)
	}
	if cfg.NHeadV%cfg.NHeadK != 0 {
		return nil, fmt.Errorf("llm: %d value heads do not group over %d key heads", cfg.NHeadV, cfg.NHeadK)
	}
	if cfg.NHeadV*cfg.HeadDim != cfg.Inner {
		return nil, fmt.Errorf("llm: %d value heads of %d do not make inner %d", cfg.NHeadV, cfg.HeadDim, cfg.Inner)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	if bank == BankQ4K && sim.Off() {
		return nil, fmt.Errorf("llm: a q4_k deltanet bank needs a format to quantise to")
	}
	g := &DeltaNetGPU{
		dev: dev, cfg: cfg, bank: bank, sim: sim,
		pipes:    make(map[string]*vk.ComputePipeline),
		tokens:   maxTokens,
		rows:     maxTokens,
		lda:      cfg.NEmbd + gemmPad,
		ldCtx:    cfg.Inner + gemmPad,
		scan:     DefaultDNKernel(),
		gemm:     GEMMKernelFor(maxTokens),
		outGemm:  OutGEMMKernelFor(maxTokens),
		autoPlan: true,
	}
	g.imLayer = make([]int, len(layers))
	for i, w := range layers {
		g.imLayer[i] = w.Layer
	}
	align := coopMatTile
	for _, v := range gemmBuildsFor(bank) {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)

	if err := g.alloc(len(layers)); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(layers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// alloc lays out the four arenas. Nothing is read here — every size follows
// from the config — so the layout can be checked before a weight is touched.
func (g *DeltaNetGPU) alloc(nLayers int) error {
	c := g.cfg
	rows := g.arenaRows
	cw := c.ConvWidth()

	perLayerW := cw*c.Conv + c.HeadDim + 2*c.NHeadV
	g.wElems = nLayers * perLayerW
	var err error
	if g.wbuf, err = g.dev.NewBuffer(g.wElems * 4); err != nil {
		return fmt.Errorf("llm: deltanet fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	// The convolution's history: Conv-1 rows a layer, a ring addressed by
	// position, which is the same 120 KB window the CPU reference carries
	// beside the [128, 128, 48] state. A kernel that carries the state and
	// forgets it is wrong in a way nothing but a batch-split test sees
	// (L3a-6) — and a *graph* that carries one shared window across 36 layers
	// is wrong in a way nothing but a chunk-split test sees (L7b-2), which is
	// why it is per layer here and was not before.
	hist := c.Conv - 1
	// The sequence position, and it must be the arena's dword 0: the kernels
	// read it as `actu[0]` (SEQ_PAST in llm_common.glsl, P1c). SetPast writes
	// it.
	if seq := alloc(1); seq != 0 {
		return fmt.Errorf("llm: the sequence slot is at %d, and SEQ_PAST is actu[0]", seq)
	}
	g.aQKV = alloc(rows * g.qkvN())
	g.aWin = alloc(nLayers * hist * g.qkvN())
	g.aNorm = alloc(rows * cw)
	g.aSilu = alloc(rows * cw)
	g.aOut = alloc(rows * c.Inner)
	g.aGate = alloc(rows * c.NHeadV)
	g.aBeta = alloc(rows * c.NHeadV)
	g.aState = alloc(nLayers * c.StateSize())
	g.aResult = alloc(rows * c.NEmbd)
	// The decode GEMV's partial sums, f32 [KSLABS][gemmN] (L8d-4). The wider
	// of the two projections is the fused one, so 32 x 16512 x 4 = 2.1 MB —
	// allocated whichever rung runs, because the arena plan is fixed at
	// construction and the rung is not.
	g.aPart = alloc(gemvMaxSlabs * g.qkvN())
	if g.abuf, err = newArena(g.dev, g.actElems*4); err != nil {
		return fmt.Errorf("llm: deltanet fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	g.writeSeq()

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hXn = halloc(rows * g.lda)
	g.hCtx = halloc(rows * g.ldCtx)
	if g.hbuf, err = newArena(g.dev, g.hElems*2); err != nil {
		return fmt.Errorf("llm: deltanet fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the A operands' pad columns, every short run's pad rows,
	// the convolution's history and every layer's recurrent state all start
	// here, and none of these kernels bounds-check.
	g.hbuf.Zero()
	g.abuf.Zero()

	// A layer's two matrices, one after the other. In the fp16 bank an offset
	// is a half and a matrix is n*k of them; in L8's it is a byte and a
	// matrix is n*k bytes of int8 tiles plus a scale plane, aligned so that
	// every base is a whole word for the kernel's `uint` view. There is no
	// fp16 tail: alpha and beta are quantised onto the plane with the rest of
	// the family (D13's payoff, P2).
	qkvBank, outBank := g.qkvN()*c.NEmbd*2, c.NEmbd*c.Inner*2
	unit := 2
	if g.quant() {
		if g.bank == BankQ4K {
			if err := q4kFits(g.qkvN(), c.NEmbd); err != nil {
				return err
			}
			if err := q4kFits(c.NEmbd, c.Inner); err != nil {
				return err
			}
			qkvBank, outBank = q8Align(q4kBytes(g.qkvN(), c.NEmbd)), q8Align(q4kBytes(c.NEmbd, c.Inner))
		} else {
			qkvBank, outBank = q8Align(q8Bytes(g.qkvN(), c.NEmbd)), q8Align(q8Bytes(c.NEmbd, c.Inner))
		}
		unit = 1
	}
	perBank := qkvBank + outBank
	if g.wbank, err = g.dev.NewBuffer(nLayers * perBank); err != nil {
		return fmt.Errorf("llm: deltanet weight bank (%d MB): %w", (nLayers*perBank)>>20, err)
	}
	g.layers = make([]dnLayerWeights, nLayers)
	for i := range g.layers {
		base := uint32(i * perLayerW)
		g.layers[i] = dnLayerWeights{
			qkv:    uint32(i * perBank / unit),
			out:    uint32((i*perBank + qkvBank) / unit),
			conv:   base,
			gamma:  base + uint32(cw*c.Conv),
			a:      base + uint32(cw*c.Conv+c.HeadDim),
			dtBias: base + uint32(cw*c.Conv+c.HeadDim+c.NHeadV),
			state:  g.aState + uint32(i*c.StateSize()),
			win:    g.aWin + uint32(i*(c.Conv-1)*g.qkvN()),
		}
	}
	return nil
}

func (g *DeltaNetGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.wbank, g.abuf}
	// Only the two projections read the bank, and only their Q8 build names
	// the sixth buffer; the conv, the scan and the norm are unchanged.
	gemmBufs := bufs
	if g.quant() {
		gemmBufs = append(append([]*vk.Buffer{}, bufs...), g.wbank)
	}
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"conv": shaders.LLMDNConv,
		"norm": shaders.LLMDNNorm,
		"hist": shaders.LLMSeqHist,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < dnWave {
		return fmt.Errorf("llm: the scan and GEMM rungs need a pinned %d-wide subgroup", dnWave)
	}
	for _, v := range dnVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
		}); err != nil {
			return err
		}
	}
	// Both builds, when the bank is quantised: the two projections read the
	// narrow plane and the fp16 tail dispatch reads halves out of the same
	// buffer, so the block needs a pipeline of each. On the fp16 bank there
	// is no tail and the second set is never built.
	for _, v := range gemmVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
		}); err != nil {
			return err
		}
	}
	if g.quant() {
		for _, v := range gemmBuildsFor(g.bank) {
			if err := g.pipeline(bankPipe(g.bank, v.name), v.spirv, vk.PipelineSpec{
				Buffers: gemmBufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
			}); err != nil {
				return err
			}
		}
	}
	// The decode GEMV, every rung of both banks plus the reduce (L8d-4). They
	// are built whatever the bank is, because the reduce reads neither and
	// the fp16 partials arm is what a block on the fp16 bank runs.
	for name, spirv := range gemvSPIRV {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{
			Buffers: gemmBufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (g *DeltaNetGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader dn.%s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline dn.%s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage packs each layer's weights.
//
// The fused projection is the four matrices laid end to end as output rows of
// one [qkvN, nEmbd] matrix, in the order the shaders derive: the
// convolution's 10240 channels (q, then k, then v), the gate z, then alpha
// and beta. The pad rows the column block adds are left zero and their
// outputs are never read.
func (g *DeltaNetGPU) stage(layers []DeltaNetWeights) error {
	c := g.cfg
	cw := c.ConvWidth()
	for i, w := range layers {
		for _, t := range []struct {
			name string
			off  uint32
			src  []float32
			want int
		}{
			{"ssm_conv1d", g.layers[i].conv, w.Conv1d, cw * c.Conv},
			{"ssm_norm", g.layers[i].gamma, w.Norm, c.HeadDim},
			{"ssm_a", g.layers[i].a, w.A, c.NHeadV},
			{"ssm_dt.bias", g.layers[i].dtBias, w.DTBias, c.NHeadV},
		} {
			if len(t.src) != t.want {
				return fmt.Errorf("llm: layer %d %s is %d values, want %d", i, t.name, len(t.src), t.want)
			}
			g.wbuf.WriteFloat32At(int(t.off), t.src)
		}

		for _, t := range []struct {
			name string
			src  []float32
			n    int
		}{
			{"attn_qkv", w.QKV, cw},
			{"attn_gate", w.Z, c.Inner},
			{"ssm_alpha", w.Alpha, c.NHeadV},
			{"ssm_beta", w.Beta, c.NHeadV},
		} {
			if len(t.src) != t.n*c.NEmbd {
				return fmt.Errorf("llm: layer %d %s is %d values, want %d", i, t.name, len(t.src), t.n*c.NEmbd)
			}
		}
		if len(w.Out) != c.NEmbd*c.Inner {
			return fmt.Errorf("llm: layer %d ssm_out is %d values, want %d", i, len(w.Out), c.NEmbd*c.Inner)
		}

		baseZ, baseA, baseB := g.colZ(), g.colAlpha(), g.colBeta()
		if g.bank == BankQ4K {
			// L8c-5's bank, with the fp16 tail gone (D13's payoff, P2):
			// alpha and beta go on the plane at the family's width, which is
			// what the simulation behind L8c-3's uniform number always did —
			// the bank finally is the simulation with no scope carve-out.
			//
			// **Each source of the fused matrix is quantised under its own
			// name.** The published matrix has a row per tensor, `attn_qkv`
			// and `attn_gate` are different tensors, and the simulation
			// L8c-3's number came out of looked each of them up separately —
			// so a bank that calibrated the fused thing with one row would
			// not be the format that was measured, however little the two
			// rows differ.
			nsbQKV := c.NEmbd / (q4kSuper * 32)
			q4 := make([]byte, g.qkvN()*c.NEmbd/2)
			rec := make([]byte, g.qkvN()/coopMatTile*nsbQKV*coopMatTile*q4kRecord)
			for _, t := range []struct {
				name string
				src  []float32
				n    int
				base int
			}{
				{"attn_qkv.weight", w.QKV, cw, 0},
				{"attn_gate.weight", w.Z, c.Inner, baseZ},
				{"ssm_alpha.weight", w.Alpha, c.NHeadV, baseA},
				{"ssm_beta.weight", w.Beta, c.NHeadV, baseB},
			} {
				name := fmt.Sprintf("blk.%d.%s", g.imLayer[i], t.name)
				q, qw, err := bankImatrix(name, g.sim)
				if err != nil {
					return err
				}
				base := t.base
				if err := tileBQ4K(q4, rec, t.src, t.n, c.NEmbd,
					func(r int) int { return base + r }, q, qw); err != nil {
					return fmt.Errorf("llm: layer %d %s: %w", i, name, err)
				}
			}
			g.wbank.WriteBytesAt(int(g.layers[i].qkv), q4)
			g.wbank.WriteBytesAt(int(g.layers[i].qkv)+len(q4), rec)

			name := fmt.Sprintf("blk.%d.ssm_out.weight", g.imLayer[i])
			q, qw, err := bankImatrix(name, g.sim)
			if err != nil {
				return err
			}
			nsbOut := c.Inner / (q4kSuper * 32)
			oq := make([]byte, c.NEmbd*c.Inner/2)
			orec := make([]byte, c.NEmbd/coopMatTile*nsbOut*coopMatTile*q4kRecord)
			if err := tileBQ4K(oq, orec, w.Out, c.NEmbd, c.Inner,
				func(r int) int { return r }, q, qw); err != nil {
				return fmt.Errorf("llm: layer %d %s: %w", i, name, err)
			}
			g.wbank.WriteBytesAt(int(g.layers[i].out), oq)
			g.wbank.WriteBytesAt(int(g.layers[i].out)+len(oq), orec)
			continue
		}
		if g.quant() {
			// The conv and gate rows are Q8_0 and round-trip exactly; alpha
			// and beta are F32, so int8 is a real re-quantisation of them —
			// priced at L8c-1 as +0.01% over the whole corpus, which is what
			// retired the fp16 tail and its second plane (D13's payoff, P2).
			qs := make([]byte, g.qkvN()*c.NEmbd)
			sc := make([]uint16, g.qkvN()*c.NEmbd/q8Group)
			tileBQ8(qs, sc, w.QKV, cw, c.NEmbd, func(r int) int { return r })
			tileBQ8(qs, sc, w.Z, c.Inner, c.NEmbd, func(r int) int { return baseZ + r })
			tileBQ8(qs, sc, w.Alpha, c.NHeadV, c.NEmbd, func(r int) int { return baseA + r })
			tileBQ8(qs, sc, w.Beta, c.NHeadV, c.NEmbd, func(r int) int { return baseB + r })
			g.wbank.WriteBytesAt(int(g.layers[i].qkv), qs)
			g.wbank.WriteUint16At((int(g.layers[i].qkv)+len(qs))/2, sc)

			oq := make([]byte, c.NEmbd*c.Inner)
			os := make([]uint16, c.NEmbd*c.Inner/q8Group)
			tileBQ8(oq, os, w.Out, c.NEmbd, c.Inner, func(r int) int { return r })
			g.wbank.WriteBytesAt(int(g.layers[i].out), oq)
			g.wbank.WriteUint16At((int(g.layers[i].out)+len(oq))/2, os)
			continue
		}

		// **The simulation of this bank, on the fp16 arm.** A format handed
		// to the halves is L8c-1's round trip done explicitly rather than
		// through the environment, and it is what lets one process hold the
		// simulated q4_k and the real one and compare a layer's output value
		// for value. It covers exactly what the q4_k bank quantises — the
		// two fused sources and the output projection, not the tail — so the
		// two arms are the same format applied to the same weights.
		qkvSrc, zSrc, outSrc := w.QKV, w.Z, w.Out
		alphaSrc, betaSrc := w.Alpha, w.Beta
		if g.bank == BankFP16 && !g.sim.Off() {
			for _, t := range []struct {
				name string
				src  *[]float32
				k    int
			}{
				{"attn_qkv.weight", &qkvSrc, c.NEmbd},
				{"attn_gate.weight", &zSrc, c.NEmbd},
				{"ssm_alpha.weight", &alphaSrc, c.NEmbd},
				{"ssm_beta.weight", &betaSrc, c.NEmbd},
				{"ssm_out.weight", &outSrc, c.Inner},
			} {
				name := fmt.Sprintf("blk.%d.%s", g.imLayer[i], t.name)
				q, qw, err := bankImatrix(name, g.sim)
				if err != nil {
					return err
				}
				x := append([]float32(nil), *t.src...)
				if err := q.ApplyWeighted(x, t.k, qw); err != nil {
					return fmt.Errorf("llm: layer %d %s: %w", i, name, err)
				}
				*t.src = x
			}
		}

		qkv := make([]uint16, g.qkvN()*c.NEmbd)
		tileB(qkv, qkvSrc, cw, c.NEmbd, func(r int) int { return r })
		tileB(qkv, zSrc, c.Inner, c.NEmbd, func(r int) int { return baseZ + r })
		tileB(qkv, alphaSrc, c.NHeadV, c.NEmbd, func(r int) int { return baseA + r })
		tileB(qkv, betaSrc, c.NHeadV, c.NEmbd, func(r int) int { return baseB + r })
		g.wbank.WriteUint16At(int(g.layers[i].qkv), qkv)

		out := make([]uint16, c.NEmbd*c.Inner)
		tileB(out, outSrc, c.NEmbd, c.Inner, func(r int) int { return r })
		g.wbank.WriteUint16At(int(g.layers[i].out), out)
	}
	return nil
}

// SetPlan chooses the three rungs: the scan's, and one row block for each of
// the two projections. Every rung is built and they all read the same staged
// weight, so this moves a pipeline and restages nothing.
func (g *DeltaNetGPU) SetPlan(scan DNKernel, gemm, outGemm GEMMKernel) error {
	if _, ok := dnVariantFor(scan); !ok {
		return fmt.Errorf("llm: no scan kernel %q (have %v)", scan, DNKernels())
	}
	for _, k := range []GEMMKernel{gemm, outGemm} {
		if _, ok := gemmVariantFor(k); !ok {
			return fmt.Errorf("llm: no GEMM kernel %q (have %v)", k, GEMMKernels())
		}
	}
	g.scan, g.gemm, g.outGemm = scan, gemm, outGemm
	g.autoPlan = false
	return nil
}

// Bank reports which width the two projections are staged in, and Format the
// plan that chose it — "off" on the two banks that are the checkpoint's own.
func (g *DeltaNetGPU) Bank() DenseBank  { return g.bank }
func (g *DeltaNetGPU) Format() QuantSim { return g.sim }

// Plan reports the rungs in use.
func (g *DeltaNetGPU) Plan() (DNKernel, GEMMKernel, GEMMKernel) {
	return g.scan, g.gemm, g.outGemm
}

// SetGemv chooses the two projections' decode rungs, or GEMVOff to leave them
// on llm_gemm.comp. It is separate from SetPlan because the GEMV is a
// different kernel over the same staged weight and its ladder does not cross
// the GEMM's — and because the two projections do not want the same split:
// K is 2560 on one and 6144 on the other, so D12's 4 KB rotation lands on
// different rungs.
func (g *DeltaNetGPU) SetGemv(qkv, out GEMVKernel) error {
	for _, t := range []struct {
		k     GEMVKernel
		gemmK int
		what  string
	}{{qkv, g.cfg.NEmbd, "qkv"}, {out, g.cfg.Inner, "out"}} {
		if t.k == GEMVOff {
			continue
		}
		if gemvSlabs(t.k) == 0 {
			return fmt.Errorf("llm: no GEMV kernel %q (have %v)", t.k, GEMVKernels())
		}
		if !GEMVFits(t.k, t.gemmK) {
			return fmt.Errorf("llm: GEMV rung %q does not cut K = %d of %s into whole four-tile steps",
				t.k, t.gemmK, t.what)
		}
	}
	if (qkv != GEMVOff || out != GEMVOff) && g.rows != 1 {
		return fmt.Errorf("llm: the GEMV rungs read one token's row; this batch is %d", g.rows)
	}
	g.qkvGemv, g.outGemv = qkv, out
	g.autoPlan = false
	return nil
}

// Gemv reports the decode rungs in use.
func (g *DeltaNetGPU) Gemv() (GEMVKernel, GEMVKernel) { return g.qkvGemv, g.outGemv }

// PinGemv holds the two projections on llm_gemm.comp whatever the batch, and
// releases them to the measured schedule when off. See Graph.PinSchedule.
func (g *DeltaNetGPU) PinGemv(on bool) {
	g.pinGemv = on
	if on {
		g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
	} else if g.autoPlan {
		g.qkvGemv, g.outGemv = DNGemvFor(g.rows)
	}
}

// DNGemvFor is the decode plan: llm_gemv.comp at one token and the GEMM at
// every other length, where its A operand is a full fragment and its grid is
// no longer `gemmN/64` workgroups for one row.
func DNGemvFor(tokens int) (GEMVKernel, GEMVKernel) {
	if tokens != 1 || !DecodeGEMV() {
		return GEMVOff, GEMVOff
	}
	return dnQKVGemv, dnOutGemv
}

// KeepSilu makes the convolution also write its un-normalised SiLU output,
// which is `conv_output_silu` and which the fused kernel otherwise never
// materialises — the same NO_W switch the hyper-connection block's gate has.
// It costs a second 21 MB write at 512 tokens, so it is off by default and on
// in the tests.
func (g *DeltaNetGPU) KeepSilu(on bool) { g.keepSilu = on }

// Layers is how many layers are staged and Tokens the longest prompt the
// arenas were built for.
func (g *DeltaNetGPU) Layers() int { return len(g.layers) }
func (g *DeltaNetGPU) Tokens() int { return g.tokens }

// WeightBytes is what the staged layers cost on the device and
// ActivationBytes what the shared arenas cost.
func (g *DeltaNetGPU) WeightBytes() int { return g.wbuf.Size() + g.wbank.Size() }

// Buffers is how many device allocations the layer holds (L6a). All 36
// layers' fused projections are one bank of 4.18 GB, which clears this
// device's 4 GiB - 4 by 2.8%.
func (g *DeltaNetGPU) Buffers() int         { return 4 }
func (g *DeltaNetGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Upload writes the layer's input: the hyper-connection block's output
// `hc_mixed`, [T][nEmbd], narrowed into the fused projection's A layout.
//
// The rows between the prompt and the GEMM's row block are zeroed rather than
// left, so a short run never reads what a longer one wrote — the kernels do
// not bounds-check and the pad rows' products have to be products of zeros.
func (g *DeltaNetGPU) Upload(xn []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(xn) != nTok*c.NEmbd {
		return fmt.Errorf("llm: input is %d values, want %d", len(xn), nTok*c.NEmbd)
	}
	g.rows = nTok
	if g.autoPlan {
		g.gemm, g.outGemm = GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)
		if g.pinGemv {
			g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
		} else {
			g.qkvGemv, g.outGemv = DNGemvFor(nTok)
		}
	}
	slab := make([]uint16, nTok*g.lda)
	narrowRows(slab, xn, nTok, c.NEmbd, g.lda)
	g.hbuf.WriteUint16At(int(g.hXn), slab)
	if pad := g.arenaRows - nTok; pad > 0 {
		g.hbuf.ZeroUint16At(int(g.hXn)+nTok*g.lda, pad*g.lda)
		g.hbuf.ZeroUint16At(int(g.hCtx)+nTok*g.ldCtx, pad*g.ldCtx)
	}
	return nil
}

// Past is how many tokens of this sequence are behind the run, SetPast moves
// it, and Reset clears one layer.
//
// The position is the block's, not the layer's: every staged layer runs over
// the same tokens, so one number serves all of them and the graph sets it
// once a run.
func (g *DeltaNetGPU) Past() int { return g.past }

// SetPast places the next run's first token at position n.
func (g *DeltaNetGPU) SetPast(n int) error {
	if n < 0 {
		return fmt.Errorf("llm: position %d", n)
	}
	g.past = n
	g.writeSeq()
	return nil
}

// writeSeq puts the position where the kernels read it: dword 0 of the fp32
// arena (SEQ_PAST in llm_common.glsl), a host write instead of a push
// constant so the recorded decode step is byte-identical every token (P1c).
func (g *DeltaNetGPU) writeSeq() { g.abuf.WriteUint32At(0, []uint32{uint32(g.past)}) }

// Reset zeroes one layer's recurrent state and the convolution's window,
// which is what a fresh sequence starts from.
//
// The window would not strictly need clearing — a tap that reaches before
// position zero contributes nothing whatever the ring holds — but this is
// 198 KB a layer against a 3.1 MB state, and `State` below is easier to
// believe when the two agree about what an empty sequence looks like.
func (g *DeltaNetGPU) Reset(layer int) error {
	if layer < 0 || layer >= len(g.layers) {
		return fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	g.abuf.ZeroFloat32At(int(g.layers[layer].state), c.StateSize())
	g.abuf.ZeroFloat32At(int(g.layers[layer].win), (c.Conv-1)*g.qkvN())
	return nil
}

// SetState writes a layer's carried state — the recurrence's matrices and the
// convolution's window — in the CPU reference's layouts, so the two
// implementations can be started from the same place.
func (g *DeltaNetGPU) SetState(layer int, st *DeltaNetState) error {
	if layer < 0 || layer >= len(g.layers) {
		return fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	if st == nil {
		return g.Reset(layer)
	}
	if len(st.S) != c.StateSize() || len(st.Conv) != c.ConvWidth()*(c.Conv-1) {
		return fmt.Errorf("llm: state is %d + %d values, want %d + %d",
			len(st.S), len(st.Conv), c.StateSize(), c.ConvWidth()*(c.Conv-1))
	}
	g.abuf.WriteFloat32At(int(g.layers[layer].state), st.S)
	// DeltaNetState.Conv is [channel][position] with the token axis fastest,
	// which is what ggml_ssm_conv slides a window over; the ring is
	// [position mod hist][channel]. Column 0 is the oldest, and the positions
	// it stands for are the `hist` before the run Past names — so a caller
	// seeds the window by setting the position first.
	hist := c.Conv - 1
	rows := make([]float32, hist*g.qkvN())
	for ch := 0; ch < c.ConvWidth(); ch++ {
		for p := 0; p < hist; p++ {
			pos := g.past - hist + p
			if pos < 0 {
				continue
			}
			rows[(pos%hist)*g.qkvN()+ch] = st.Conv[ch*hist+p]
		}
	}
	g.abuf.WriteFloat32At(int(g.layers[layer].win), rows)
	return nil
}

// State reads a layer's carried state back in the same layouts, which is what
// makes a chunk-boundary comparison against the CPU reference possible.
func (g *DeltaNetGPU) State(layer int) (*DeltaNetState, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	hist := c.Conv - 1
	st := &DeltaNetState{
		S:    g.abuf.ReadFloat32At(int(g.layers[layer].state), c.StateSize()),
		Conv: make([]float32, c.ConvWidth()*hist),
	}
	// The window after a run is the last hist positions of the sequence so
	// far — this batch's tail unless the batch is shorter than it, in which
	// case the ring still holds what an earlier one left.
	rows := g.abuf.ReadFloat32At(int(g.layers[layer].win), hist*g.qkvN())
	end := g.past + g.rows
	for ch := 0; ch < c.ConvWidth(); ch++ {
		for p := 0; p < hist; p++ {
			pos := end - hist + p
			if pos < 0 {
				continue
			}
			st.Conv[ch*hist+p] = rows[(pos%hist)*g.qkvN()+ch]
		}
	}
	return st, nil
}

// graph builds one layer's dispatch sequence, with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes.
func (g *DeltaNetGPU) graph(layer int) ([]vk.MultiDispatch, []string, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	w := g.layers[layer]
	sv, _ := dnVariantFor(g.scan)
	gv, _ := gemmVariantFor(g.gemm)
	ov, _ := gemmVariantFor(g.outGemm)

	norm := uint32(0)
	if c.QKNorm == L2Max {
		norm = 1
	}
	base := push{
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd),
		LDA: uint32(g.lda), LDCtx: uint32(g.ldCtx), Eps: math.Float32bits(c.Eps),
		QKVOff: g.aQKV, NormOff: g.aNorm, OutOff: g.aOut, CtxOff: g.hCtx,
		ConvOff: w.conv, GammaOff: w.gamma, Kern: uint32(c.Conv),
		ConvOutOff: noW,
		SSMGateOff: g.aGate, SSMBetaOff: g.aBeta, SSMStateOff: w.state,
		SSMAOff: w.a, SSMDTOff: w.dtBias, SSMNorm: norm,
		Heads: uint32(c.NHeadV), KVHeads: uint32(c.NHeadK), HeadDim: uint32(c.HeadDim),
		GemmN: uint32(g.qkvN()),
		// SEQ_HIST: a field this layer does not otherwise use, because the
		// push block is full at 64 uints (llm_common.glsl). The position is
		// not here any more: SEQ_PAST is dword 0 of the arena (P1c), written
		// by SetPast, so `lowRank` stays zero and the dispatch is
		// byte-identical every decode step.
		InjOff: w.win,
	}
	if g.keepSilu {
		base.ConvOutOff = g.aSilu
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// 1. The one fused projection: four of llama.cpp's matrices, one matmul.
	//    Alpha and beta are on the quantised plane with the rest of the
	//    family (D13's payoff, P2), so no field states a split.
	qkv := base
	qkv.XnOff, qkv.OutOff, qkv.BOff = g.hXn, g.aQKV, w.qkv
	qkv.GemmM, qkv.GemmK = uint32(roundUpInt(g.rows, gv.bm)), uint32(c.NEmbd)
	gy := uint32(roundUpInt(g.rows, gv.bm) / gv.bm)
	if ks := gemvSlabs(g.qkvGemv); ks > 0 {
		// The decode kernel: one row, so the parallelism comes from K and not
		// from a sixteen-row fragment of which fifteen rows are padding
		// (L8d-4). The partials ride `resOff`, which this block does not use.
		qkv.ResOff = g.aPart
		add(gemvBankPipe(g.qkvGemv, g.bank), "qkv", uint32(ks), uint32(g.qkvN()/16), qkv)
		if ks > 1 {
			add(gemvSumPipe(g.qkvGemv), "qkv.sum", uint32(roundUpInt(g.qkvN(), dnWave)/dnWave), 1, qkv)
		}
	} else if g.quant() {
		add(bankPipe(g.bank, g.gemm), "qkv", uint32(g.qkvN()/dnBN), gy, qkv)
	} else {
		add(string(g.gemm), "qkv", uint32(g.qkvN()/dnBN), gy, qkv)
	}

	// 2. The convolution, its SiLU, the two L2 norms and the two per-head
	//    scalars. One plane per head, plus one for the scalars.
	planes := uint32(2*c.NHeadK + c.NHeadV + 1)
	add("conv", "conv", planes, uint32(g.rows), base)

	// 3. The delta rule.
	add(string(g.scan), "scan", uint32(c.NHeadV), uint32(c.HeadDim/sv.cols()), base)

	// 4. The gated output norm, into the output projection's A operand.
	add("norm", "norm", uint32(c.NHeadV), uint32(g.rows), base)

	// 5. The output projection.
	out := base
	out.XnOff, out.LDA = g.hCtx, uint32(g.ldCtx)
	out.OutOff, out.BOff = g.aResult, w.out
	out.GemmM, out.GemmN, out.GemmK = uint32(roundUpInt(g.rows, ov.bm)), uint32(c.NEmbd), uint32(c.Inner)
	outPipe := string(g.outGemm)
	if g.quant() {
		outPipe = bankPipe(g.bank, g.outGemm)
	}
	if ks := gemvSlabs(g.outGemv); ks > 0 {
		out.ResOff = g.aPart
		add(gemvBankPipe(g.outGemv, g.bank), "out", uint32(ks), uint32(c.NEmbd/16), out)
		if ks > 1 {
			add(gemvSumPipe(g.outGemv), "out.sum", uint32(roundUpInt(c.NEmbd, dnWave)/dnWave), 1, out)
		}
	} else {
		add(outPipe, "out", uint32(c.NEmbd/dnBN), uint32(roundUpInt(g.rows, ov.bm)/ov.bm), out)
	}

	// 6. The convolution's window, for whatever runs next: the last Conv-1
	//    rows of the projection into this layer's ring. Three rows of 16512
	//    floats — 198 KB — and the only dispatch here a single-shot prefill
	//    does not need.
	hist := c.Conv - 1
	win := base
	win.LoOff = g.aQKV // SEQ_SRC
	win.GemmM, win.GemmK = uint32(hist), uint32(g.qkvN())
	add("hist", "hist", uint32((g.qkvN()+255)/256), uint32(minInt(g.rows, hist)), win)
	return d, kinds, nil
}

// Run executes one layer over whatever Upload left in the arenas, advancing
// that layer's recurrent state.
func (g *DeltaNetGPU) Run(layer int) error {
	d, kinds, err := g.graph(layer)
	if err != nil {
		return err
	}
	// One command buffer for the whole block, not one a dispatch: a submit
	// and a fence wait is ~150 us here and a decode step is a batch of one,
	// so what the sequence costs is how many times it is handed over
	// (LLM.md L7c). And when the graph is recording a whole pass, not even
	// one a block (L7d).
	if g.rec.add(ownDN, kinds, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: deltanet, %d dispatches (%v): %w", len(d), kinds, err)
	}
	return nil
}

// Profile times each dispatch on the GPU over iters back-to-back repetitions.
//
// **The scan is not idempotent**, which is the one place this differs from
// the other two blocks' profilers: it reads the recurrent state and writes it
// back, so a repetition starts from the state the last one left. That does
// not change what it costs — the arithmetic, the traffic and the dependency
// chain are identical whatever the state holds — but it does mean a Profile
// leaves a state a Run would not, so the correctness tests drive Run and the
// state is reset around a profile.
func (g *DeltaNetGPU) Profile(layer, iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	d, kinds, err := g.graph(layer)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: deltanet dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across every staged layer, and is
// the number to quote rather than Profile's — the same argument HCGPU and
// AttnGPU make: one layer's weights are 116 MB and cannot sit in the 32 MiB
// MALL however often a dispatch is repeated, but the activations would stay
// warm in a way a real graph running 36 different layers does not.
func (g *DeltaNetGPU) ProfileSweep(iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	_, kinds, err := g.graph(0)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(kinds))
	for k := range kinds {
		var total time.Duration
		for l := range g.layers {
			d, _, err := g.graph(l)
			if err != nil {
				return nil, err
			}
			dur, err := vk.DispatchMultiTimed(d[k:k+1], 1, uint32(iters), true)
			if err != nil {
				return out, fmt.Errorf("llm: %s sweep, layer %d: %w", kinds[k], l, err)
			}
			total += dur
		}
		out = append(out, Stage{Kind: kinds[k], GPU: total / time.Duration(len(g.layers)*iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the layout deltanet.go's CPU reference
// uses.

// Result is the layer's output, `linear_attn_out`, [T][nEmbd].
func (g *DeltaNetGPU) Result() []float32 {
	return g.abuf.ReadFloat32At(int(g.aResult), g.rows*g.cfg.NEmbd)
}

// QKV is the fused projection's whole output row, [T][qkvN] —
// `linear_attn_qkv_mixed` and three more tensors. Column slices one of them
// out, which is how a disagreement gets located in a projection instead of in
// what consumed it.
func (g *DeltaNetGPU) QKV() []float32 {
	return g.abuf.ReadFloat32At(int(g.aQKV), g.rows*g.qkvN())
}

// Column slices one output of the fused projection out of its row.
func (g *DeltaNetGPU) Column(col, width int) []float32 {
	raw := g.QKV()
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*g.qkvN()+col:])
	}
	return out
}

// ConvSilu is the convolution's un-normalised output, [T][convWidth], and is
// only written when KeepSilu is on.
func (g *DeltaNetGPU) ConvSilu() []float32 {
	return g.abuf.ReadFloat32At(int(g.aSilu), g.rows*g.cfg.ConvWidth())
}

// Norm is the convolution's output with q and k L2-normalised per head and v
// left alone, [T][convWidth]: `q_conv_predelta`, `k_conv_predelta` and
// `v_conv_predelta` laid end to end. NormColumn slices one out.
func (g *DeltaNetGPU) Norm() []float32 {
	return g.abuf.ReadFloat32At(int(g.aNorm), g.rows*g.cfg.ConvWidth())
}

// NormColumn slices one of the three out of that tensor.
func (g *DeltaNetGPU) NormColumn(col, width int) []float32 {
	raw := g.Norm()
	cw := g.cfg.ConvWidth()
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*cw+col:])
	}
	return out
}

// Gate is the log decay `gate`, [T][nHeadV], and BetaSig is `beta_sigmoid`.
func (g *DeltaNetGPU) Gate() []float32 {
	return g.abuf.ReadFloat32At(int(g.aGate), g.rows*g.cfg.NHeadV)
}

// BetaSig is the write strength after its sigmoid, [T][nHeadV].
func (g *DeltaNetGPU) BetaSig() []float32 {
	return g.abuf.ReadFloat32At(int(g.aBeta), g.rows*g.cfg.NHeadV)
}

// Out is the recurrence's own output, `attn_output`, [T][nHeadV][headDim].
func (g *DeltaNetGPU) Out() []float32 {
	return g.abuf.ReadFloat32At(int(g.aOut), g.rows*g.cfg.Inner)
}

// Final is the gated output norm, `final_output`, read back out of the fp16 A
// operand it was narrowed into — so it carries fp16's step and not the
// reference's f32.
func (g *DeltaNetGPU) Final() []float32 {
	c := g.cfg
	raw := g.hbuf.ReadUint16At(int(g.hCtx), (g.rows-1)*g.ldCtx+c.Inner)
	out := make([]float32, g.rows*c.Inner)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.ldCtx : t*g.ldCtx+c.Inner] {
			out[t*c.Inner+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *DeltaNetGPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.hbuf, g.abuf, g.wbuf, g.wbank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// DNShape reports a scan rung's three coupled properties: how many state
// elements a lane holds in registers, how many workgroups the whole layer
// dispatches, and how many times each head's q and k row is re-read per
// token. They move together — that is the ladder — so a benchmark that
// reports the time without them is reporting half the measurement.
func DNShape(k DNKernel, c DeltaNetConfig) (regsPerLane, workgroups, reread int) {
	v, ok := dnVariantFor(k)
	if !ok {
		return 0, 0, 0
	}
	regs := v.state
	switch {
	case v.qkreg && v.prefetch:
		regs *= 5 // the state, and two tokens' worth of q and k
	case v.qkreg:
		regs *= 3 // the state, plus this lane's q and k components
	}
	return regs, c.NHeadV * c.HeadDim / v.cols(), c.HeadDim / v.cols()
}

// DNOperandsInRegisters says whether a rung keeps q and k beside the state or
// stages them in LDS — the second half of what its name means.
func DNOperandsInRegisters(k DNKernel) bool {
	v, _ := dnVariantFor(k)
	return v.qkreg
}

// DNPrefetches says whether a rung issues the next token's operands before
// the current token's arithmetic.
func DNPrefetches(k DNKernel) bool {
	v, _ := dnVariantFor(k)
	return v.prefetch
}

// InPort is the layer's input as the fused projection's A operand wants it:
// fp16 [T][lda], `hc_mixed` narrowed. The row count is the arena's, not the
// run's, because the GEMM rungs have no bounds check and the rows between the
// prompt and the row block have to carry the products of zeros.
func (g *DeltaNetGPU) InPort() Port {
	return Port{Buf: g.hbuf, Off: g.hXn, Stride: g.lda, Width: g.cfg.NEmbd,
		Rows: g.arenaRows, Half: true}
}

// OutPort is the layer's output, `linear_attn_out`: fp32 [T][nEmbd].
func (g *DeltaNetGPU) OutPort() Port {
	return Port{Buf: g.abuf, Off: g.aResult, Stride: g.cfg.NEmbd, Width: g.cfg.NEmbd}
}

// Resize sets the length of the run and clears what a Move will not reach.
//
// The context operand's pad rows are the part that matters: `llm_dn_norm`
// writes only the run's tokens into `hCtx`, so a shorter run after a longer
// one would leave the output projection reading the longer one's tail.
func (g *DeltaNetGPU) Resize(nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	g.rows = nTok
	if g.autoPlan {
		g.gemm, g.outGemm = GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)
		if g.pinGemv {
			g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
		} else {
			g.qkvGemv, g.outGemv = DNGemvFor(nTok)
		}
	}
	if pad := g.arenaRows - nTok; pad > 0 {
		g.hbuf.ZeroUint16At(int(g.hCtx)+nTok*g.ldCtx, pad*g.ldCtx)
	}
	return nil
}
