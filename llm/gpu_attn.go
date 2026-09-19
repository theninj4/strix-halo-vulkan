package llm

// The full-attention layer on the device: LLM.md L2f, and the GPU half of the
// reference L2e built in attn.go.
//
// Twelve of the 48 layers are this one. L2a priced the nameable lines of them
// at **5.64% of llama.cpp's 512-token prefill graph in 228 dispatches** — 19 a
// layer — spread over five projections, four RMS norms, four ropes, the flash
// attention, the top-k and the gate:
//
//	MUL_MAT q8_0 m=12288 k=2560   12   20384 us   query and gate
//	MUL_MAT q8_0 m=512   k=2560   24    2780      key, value
//	MUL_MAT bf16 m=512   k=2560   12    3491      indexer query
//	MUL_MAT bf16 m=128   k=2560   12    1536      indexer key
//	MUL_MAT q8_0 m=2560  k=6144   12    8559      output (12 of that line's 48)
//	RMS_NORM_MUL (256,24) (256,2) 24    2197      query, key
//	RMS_NORM_MUL (128,4)  (128,T) 24     358      indexer query, indexer key
//	ROPE                          48    1135      all four of them
//	MUL_MAT f32 m=512 n=2048 k=128 12     460      the indexer score
//	RELU                          12     269      its rectifier
//	FLASH_ATTN_EXT                12   25180
//	TOP_K, TOPK_QSA GET_ROWS      24    2401
//
// Five dispatches replace those nineteen, and every one of the fusions is the
// same argument L2a made for `inject` and L2d for the PLE block's value: two
// matmuls that read the same activation are one matmul with more columns.
//
//	qkv    xn -> query, gate, key, value, indexer query, indexer key
//	pack   per-head norm + IMRoPE + the fragment tiling, for q, k and v
//	idx    the indexer's pooled key and its query: pool, norm, rope, fp16
//	score  the rectified block score, its bias, the cells and the mask
//	attn   causal GQA on the matrix cores, with the output gate in its epilogue
//	out    the output projection                     (the plain GEMM again)
//
// Two tensors never exist, which is where the time goes. The **gate** is
// [T, 6144] — llama.cpp materialises sigmoid(gate), multiplies, and copies the
// product to make it contiguous, three lines and two round trips — and here it
// is one read inside the attention kernel's epilogue, which already holds the
// output tile for the softmax divide. And the **fused projection's six
// outputs** are one [13952, 2560] weight instead of five matrices, which at
// 512 tokens turns five dispatches reading 71 MB between them into one
// dispatch reading 71 MB once.
//
// What is *not* here is the selection. `top_k + ratio - 1` is 2051 against a
// 256-cell cache, so the indexer names every cell and the sparse path is
// bit-identical to the dense causal one (research/l2e-attention.md); the score
// is computed and checked, and L4 is where it starts to bite.

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// attnBN is the compiled column block of the plain GEMM arm, which both
// projections here run on.
const attnBN = 64

// attnWave is the wave every rung here is pinned to, and the width the
// decode GEMV's reduce divides.
const attnWave = 64

// The decode rungs of this layer's two projections, from results/l8e_attn.csv
// (L8e-1). Same kernel and the same two reduction extents as the gated
// DeltaNet's pair — nEmbd = 2560 for the fused projection and 6144 for the
// output one — and D12 still says the rung is per *matrix* and not per K, so
// the ladder was re-measured here rather than copied from dnQKVGemv.
const attnQKVGemv = GEMVK8
const attnOutGemv = GEMVK32

// AttnKernel names one build of shaders/llm_attn_wmma.comp, by its tile.
type AttnKernel string

const (
	AttnQT1KT2 AttnKernel = "qt1_kt2"
	AttnQT1KT4 AttnKernel = "qt1_kt4"
	AttnQT2KT4 AttnKernel = "qt2_kt4"
)

type attnVariant struct {
	name       AttnKernel
	spirv      []byte
	qt, ktil   int
	rows, keys int
}

var attnVariants = []attnVariant{
	{AttnQT1KT2, shaders.LLMAttnQT1KT2, 1, 2, 16, 32},
	{AttnQT1KT4, shaders.LLMAttnQT1KT4, 1, 4, 16, 64},
	{AttnQT2KT4, shaders.LLMAttnQT2KT4, 2, 4, 32, 64},
}

// AttnKernels lists the rungs, narrowest first.
func AttnKernels() []AttnKernel { return []AttnKernel{AttnQT1KT2, AttnQT1KT4, AttnQT2KT4} }

// DefaultAttnKernel is the measured winner, and the ladder turned over from
// where every other attention kernel in this repo sits.
//
// At 512 tokens, per layer: qt1_kt2 **167 us**, qt1_kt4 219, qt2_kt4 241 —
// 19.3, 14.7 and 13.3 TFLOP/s, monotonic in how much a wave holds live. Both
// of the DiT's and the text encoder's ladders end at qt1_kt4 or wider; this
// one ends one rung narrower because headDim is 256 and not 128, so a wave
// already carries 16 query fragments and 16 output accumulators across the key
// loop before KTIL adds a score accumulator per key tile. Arithmetic intensity
// is inert here exactly as §3.3 found it — the register file decides, and at
// this head dim it decides sooner.
func DefaultAttnKernel() AttnKernel { return AttnQT1KT2 }

func attnVariantFor(k AttnKernel) (attnVariant, bool) {
	for _, v := range attnVariants {
		if v.name == k {
			return v, true
		}
	}
	return attnVariant{}, false
}

// GEMMKernel names one build of the plain arm of llm_gemm.comp, by its row
// block. Both of this layer's projections read a B far larger than the MALL —
// the fused QKV weight is 71.4 MB and the output projection 31.5 — so the rung
// is chosen against the weight, exactly as the PLE block's is: a workgroup
// carrying BM token rows reads all of B M/BM times.
type GEMMKernel string

const (
	GEMMM2 GEMMKernel = "gemm_m2"
	GEMMM4 GEMMKernel = "gemm_m4"
	GEMMM8 GEMMKernel = "gemm_m8"
)

type gemmVariant struct {
	name  GEMMKernel
	spirv []byte
	bm    int
}

var gemmVariants = []gemmVariant{
	{GEMMM2, shaders.LLMGEMMPlainM2, 32},
	{GEMMM4, shaders.LLMGEMMPlainM4, 64},
	{GEMMM8, shaders.LLMGEMMPlainM8, 128},
}

// The same three rungs over L8's dense bank: int8 tiles and an fp16 scale
// plane instead of halves, unpacked into LDS per K-step. They are a separate
// table rather than a field on the one above because a *build* of the kernel
// is what differs, and the two are never mixed inside one block — a bank is
// staged one way or the other and the rung is chosen afterwards.
var gemmQ8Variants = []gemmVariant{
	{GEMMM2, shaders.LLMGEMMQ8M2, 32},
	{GEMMM4, shaders.LLMGEMMQ8M4, 64},
	{GEMMM8, shaders.LLMGEMMQ8M8, 128},
}

// The same three rungs again over L8c-4's 4.5-bit bank: nibble tiles and a
// sixteen-byte ggml record per (column, super-block).
var gemmQ4Variants = []gemmVariant{
	{GEMMM2, shaders.LLMGEMMQ4M2, 32},
	{GEMMM4, shaders.LLMGEMMQ4M4, 64},
	{GEMMM8, shaders.LLMGEMMQ4M8, 128},
}

// gemmQ5Variants is P3a's fifth bit on the same three rungs: the same tiles
// and the same record with a `qh` plane between them.
var gemmQ5Variants = []gemmVariant{
	{GEMMM2, shaders.LLMGEMMQ5M2, 32},
	{GEMMM4, shaders.LLMGEMMQ5M4, 64},
	{GEMMM8, shaders.LLMGEMMQ5M8, 128},
}

// gemmBuildsFor is the table a block builds its pipelines from, by bank.
func gemmBuildsFor(b DenseBank) []gemmVariant {
	switch b {
	case BankQ8:
		return gemmQ8Variants
	case BankQ4K:
		return gemmQ4Variants
	case BankQ5K:
		return gemmQ5Variants
	}
	return gemmVariants
}

// GEMMKernels lists the rungs, narrowest first.
func GEMMKernels() []GEMMKernel { return []GEMMKernel{GEMMM2, GEMMM4, GEMMM8} }

// GEMMKernelFor is the rung for the *fused projection*, whose B is 71.4 MB —
// 2.2x the MALL — so it is chosen against the weight exactly as L2d's is: a
// workgroup carrying BM token rows reads all of B M/BM times, and that is the
// whole shape of this dispatch. Measured, us per layer, BM 32 / 64 / 128:
//
//	T =   64    497 /  413 /  452      64 rows do not fill a BM of 128
//	T =  128    917 /  529 /  452
//	T =  512   3118 / 2003 / 1520
//	T = 2048  19060 / 9570 / 6027      a 3.2x spread on identical arithmetic
func GEMMKernelFor(tokens int) GEMMKernel {
	if tokens <= 64 {
		return GEMMM4
	}
	return GEMMM8
}

// OutGEMMKernelFor is the rung for the *output projection*, and it is a
// different one at every length — which is the point of having two.
//
// Its B is 31.5 MB, just **inside** the 32 MiB MALL where the fused
// projection's is 2.2x past it, and the two ladders separate on exactly that.
// Measured, us per layer, BM 32 / 64 / 128:
//
//	T =   64    138 /  170 /  286      the widest rung is 2.1x the narrowest
//	T =  128    194 /  205 /  285
//	T =  256    265 /  306 /  316
//	T =  512    507 /  458 /  519      the middle rung, and only here
//	T = 1024   1035 /  965 /  858
//	T = 2048   2142 / 1807 / 1623      now the widest, 1.32x the narrowest
//
// So the schedule inverts across the sweep: at 64 tokens BM=128 is the worst
// rung by 2.1x and at 2048 it is the best by 1.32x, on the same kernel and the
// same arithmetic. While B fits the MALL, reading it fewer times buys nothing
// and the occupancy a narrow row block leaves is the whole story; once M is
// large enough that the *activation* traffic dominates, the wide rung wins
// back. The fused projection never has the first regime because its B never
// fits, which is why one schedule could not serve both.
func OutGEMMKernelFor(tokens int) GEMMKernel {
	switch {
	case tokens <= 256:
		return GEMMM2
	case tokens <= 512:
		return GEMMM4
	default:
		return GEMMM8
	}
}

func gemmVariantFor(k GEMMKernel) (gemmVariant, bool) {
	for _, v := range gemmVariants {
		if v.name == k {
			return v, true
		}
	}
	return gemmVariant{}, false
}

// attnLayerWeights is where one layer's staged weights sit.
type attnLayerWeights struct {
	// qkv and out are halves into the fp16 bank, or **bytes** into a
	// quantised one.
	qkv, out         uint32
	gQ, gK, gIQ, gIK uint32 // fp32 arena: the four norm gammas
}

// AttnGPU runs full-attention layers on the device. It holds however many
// layers it was staged with — at L2f that is the one the trace covers, at L6
// it will be all twelve — plus one set of activation arenas sized for the
// longest prompt.
type AttnGPU struct {
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
	dev *vk.Device
	cfg AttnConfig

	wbuf, abuf, hbuf, wbank *vk.Buffer
	pipes                   map[string]*vk.ComputePipeline
	mods                    []*vk.ShaderModule

	// bank is which width the two projections are staged in, and sim the
	// format the 4.5-bit one encodes with (L8c-7). Twelve of the 48 layers
	// are this block's, and with the indexer they read 1.27 GB of a decode
	// token as halves, 0.674 on L8a's int8 and 0.347 here. The indexer's two
	// BF16 projections go on the quantised plane with the rest of the fused
	// matrix — the fp16 tail left with D13's payoff (P2), priced by L8c-7 at
	// +0.005% where the selection bites.
	bank DenseBank
	sim  QuantSim

	layers []attnLayerWeights

	attn          AttnKernel
	gemm, outGemm GEMMKernel
	// The decode rungs of the same two projections: llm_gemv.comp at one
	// token, GEMVOff at every other length (L8e-1, on L8d-4's kernel).
	qkvGemv, outGemv GEMVKernel
	// pinGemv holds both projections on llm_gemm.comp whatever the batch, so
	// that a prompt run in chunks is the prompt run whole to the last place
	// (Graph.PinSchedule). A split sum is a different association of the same
	// products, which is the one thing a rung here does not preserve.
	pinGemv bool
	// autoPlan re-chooses both GEMM rungs per run. SetPlan turns it off,
	// because a caller that named a rung meant it.
	autoPlan bool

	tokens, arenaRows, rows int
	nKV                     int
	lda, ldCtx              int
	// past is how many cells the cache already holds: token t of the next run
	// is cell past+t and its position is past+t (L7a). Zero is a fresh
	// sequence, which is every run before L7 and every test above L4.
	past int
	// layer is the one the last graph was built for, so that the read-back
	// accessors below reach into that layer's cache rather than layer 0's.
	layer int

	// fp32 weight arena: four gammas a layer, then the rotary table.
	wElems int
	wRope  uint32

	// fp32 activations.
	aQKV, aScore, aCell, aOut uint32
	// The QSA selection, a bitmask of nKV bits a token, read as uints out of
	// the same fp32 arena through binding 4. L4b.
	aSel uint32
	// The decode GEMV's partial sums, f32 [KSLABS][gemmN] (L8e-1).
	aPart    uint32
	actElems int
	// sparse is whether the selection is dispatched and the attention kernel
	// reads it. It is decided by the cache the layer was built for, because
	// `top_k + ratio - 1` cells out of fewer than that many is the identity —
	// see selWidth. SetSparse overrides it, which is how the kernel is priced
	// at a length where it does not bite and how the dense control runs at one
	// where it does.
	sparse bool
	// fp16 activations. hXn, hQ, hCtx and hIdxQ are this batch's and are
	// overwritten by the next one.
	hXn, hQ, hCtx, hIdxQ uint32
	// The KV cache, in the same fp16 arena and **one plane per staged
	// layer**: the key and the value as nKV cells of fragment tiles, the
	// indexer's raw key per cell, and the pooled key per block (L7a). These
	// are the only tensors in this vertical that outlive the run that wrote
	// them, which is why they are strided by layer where everything else is
	// shared.
	hK, hV, hIdxRaw, hIdxK             uint32
	kvStride, idxRawStride, idxKStride int
	hElems                             int
}

// qkvN is the fused projection's output width: the query and its gate, the key,
// the value and both of the indexer's operands, padded up to the GEMM's column
// block. Six of llama.cpp's matrices, one of ours.
func (g *AttnGPU) qkvN() int {
	c := g.cfg
	n := c.QWidth() + 2*c.KVWidth() + c.IdxHeads*c.IdxDim + c.IdxDim
	return roundUpInt(n, attnBN)
}

// The column each of the fused projection's six outputs starts at. The shaders
// derive the same numbers from heads, kvHeads, headDim and the indexer's two,
// so these are the host's half of that contract and the tests check them.
func (g *AttnGPU) colK() int  { return g.cfg.QWidth() }
func (g *AttnGPU) colV() int  { return g.colK() + g.cfg.KVWidth() }
func (g *AttnGPU) colIQ() int { return g.colV() + g.cfg.KVWidth() }
func (g *AttnGPU) colIK() int { return g.colIQ() + g.cfg.IdxHeads*g.cfg.IdxDim }

// quant is whether the two projections read a narrow bank at all.
func (g *AttnGPU) quant() bool { return g.bank != BankFP16 }

// Bank is the width the two projections are staged in and Format the encoder
// the 4.5-bit one used, for a header line and a CSV.
func (g *AttnGPU) Bank() DenseBank  { return g.bank }
func (g *AttnGPU) Format() QuantSim { return g.sim }

// selWidth is what the selection asks for: whole blocks plus the incomplete
// tail, `top_k + ratio - 1`, and never more cells than the cache has. At 2051
// against a 256-cell cache it names every cell, which is why L2 could not test
// the selection at all and why `sparse` is false there.
func (g *AttnGPU) selWidth() int { return minInt(g.cfg.TopK+g.cfg.Ratio-1, g.nKV) }

// selWords is the bitmask's row stride: one bit a cell, 32 cells a uint. At a
// 4096-cell cache that is 128 uints a token — 2 MB for the whole prompt,
// against the 32 MB the reference's `width` int32 indices would take.
func (g *AttnGPU) selWords() int { return (g.nKV + 31) / 32 }

// Sparse reports whether the layer runs the selection. SetSparse forces it
// either way: on where it is the identity, to price the kernel against
// llama.cpp's TOP_K, and off where it bites, which is the negative control
// that says the selection changes the attention output at all.
func (g *AttnGPU) Sparse() bool     { return g.sparse }
func (g *AttnGPU) SetSparse(v bool) { g.sparse = v }

// Past is how many cells the cache holds before the next run, SetPast moves
// it, and Reset starts a fresh sequence.
//
// Nothing is cleared: a cell past `past + nTok` is masked out of every score
// and every softmax, so what a longer previous run left there is unreadable
// rather than merely unread. What a fresh sequence does need is the whole
// pooled block table rebuilt, and `blockRange` is where that is said.
func (g *AttnGPU) Past() int { return g.past }
func (g *AttnGPU) Reset()    { g.past = 0; g.writeSeq() }

// SetPast places the next run's first token at cell n.
func (g *AttnGPU) SetPast(n int) error {
	if n < 0 || n >= g.nKV {
		return fmt.Errorf("llm: cell %d of a %d-cell cache", n, g.nKV)
	}
	g.past = n
	g.writeSeq()
	return nil
}

// writeSeq puts the position where the kernels read it: dword 0 of the fp32
// arena (SEQ_PAST in llm_common.glsl). A host write to a mapped buffer, so
// it costs nothing a push constant did not — and unlike a push constant it
// is not baked into a recorded command buffer, which is what lets P1c record
// the decode step once.
func (g *AttnGPU) writeSeq() { g.abuf.WriteUint32At(0, []uint32{uint32(g.past)}) }

// blockRange is the half-open range of pooled indexer blocks a run rebuilds,
// and llm_attn_idx.comp derives the same two numbers from the push block.
//
// A cell's contents never change, so a block is final as soon as all `ratio`
// of its cells exist: a continuing run touches only the blocks its own tokens
// completed. A fresh sequence rebuilds the whole table, because the blocks
// past the last complete one pool cell 0 `ratio` times and have to be written
// once before they can be left alone.
func (g *AttnGPU) blockRange() (int, int) {
	if g.past == 0 {
		return 0, g.NBlocks()
	}
	return g.past / g.cfg.Ratio, (g.past + g.rows) / g.cfg.Ratio
}

// NBlocks is the indexer's block count: the cache's cell count over the
// compress ratio, which at 7 tokens in a 256-cell cache is 64 — one real block
// and 63 that do not exist.
func (g *AttnGPU) NBlocks() int { return (g.nKV + g.cfg.Ratio - 1) / g.cfg.Ratio }

// NewAttnGPU stages layers onto the device and builds every pipeline.
//
// maxTokens is the longest prompt the arenas are built for and nKV the cache's
// cell count, which is the reference's padded number rather than the prompt's:
// the indexer's block grid, its bias and the rotary table are all cut against
// it (research/l2e-attention.md).
func NewAttnGPU(dev *vk.Device, cfg AttnConfig, maxTokens, nKV int, layers []AttnWeights, q8 bool) (*AttnGPU, error) {
	return NewAttnGPUBank(dev, cfg, maxTokens, nKV, layers, bankOf(q8), QuantSim{})
}

// NewAttnGPUBank is the same with the bank named rather than implied and the
// format the 4.5-bit one encodes with. The indexer's two BF16 projections go
// on a quantised bank's plane with the rest of the fused matrix (P2).
func NewAttnGPUBank(dev *vk.Device, cfg AttnConfig, maxTokens, nKV int, layers []AttnWeights,
	bank DenseBank, sim QuantSim) (*AttnGPU, error) {
	if maxTokens <= 0 || nKV < maxTokens {
		return nil, fmt.Errorf("llm: %d tokens in a %d-cell cache", maxTokens, nKV)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("llm: no attention layers to stage")
	}
	if cfg.HeadDim != 256 {
		return nil, fmt.Errorf("llm: the attention kernel is built for headDim 256, this checkpoint says %d", cfg.HeadDim)
	}
	if cfg.IdxDim != 128 || cfg.IdxHeads != 4 {
		return nil, fmt.Errorf("llm: the indexer kernels are built for 4 heads of 128, this checkpoint says %d of %d",
			cfg.IdxHeads, cfg.IdxDim)
	}
	if cfg.Ratio <= 0 {
		return nil, fmt.Errorf("llm: layer has compress ratio %d, so it is not a full-attention layer", cfg.Ratio)
	}
	if cfg.NHead%cfg.NHeadKV != 0 {
		return nil, fmt.Errorf("llm: %d query heads do not group over %d kv heads", cfg.NHead, cfg.NHeadKV)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	if _, ok := qkBank(bank); ok && sim.Off() {
		return nil, fmt.Errorf("llm: a %s attention bank needs a format to quantise to", bank)
	}
	g := &AttnGPU{
		dev: dev, cfg: cfg, bank: bank, sim: sim,
		pipes:    make(map[string]*vk.ComputePipeline),
		tokens:   maxTokens,
		rows:     maxTokens,
		nKV:      nKV,
		lda:      cfg.NEmbd + gemmPad,
		ldCtx:    cfg.GateWidth() + gemmPad,
		attn:     DefaultAttnKernel(),
		gemm:     GEMMKernelFor(maxTokens),
		outGemm:  OutGEMMKernelFor(maxTokens),
		autoPlan: true,
	}
	// The plane is padded up to the widest tile any rung covers, because the
	// attention kernel reads whole key blocks and the pack writes whole token
	// tiles; both would otherwise read whatever a previous, longer run left.
	align := coopMatTile
	for _, v := range attnVariants {
		align = maxInt(align, maxInt(v.rows, v.keys))
	}
	for _, v := range gemmBuildsFor(bank) {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)
	// The selection is the identity wherever it asks for at least as many
	// cells as the cache holds, so below that it is not run and the attention
	// kernel takes its dense arm. That is a fact about the cache rather than
	// about the prompt, and the cache is fixed here.
	g.sparse = cfg.TopK > 0 && g.selWidth() < nKV

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
func (g *AttnGPU) alloc(nLayers int) error {
	c := g.cfg
	rows := g.arenaRows
	rotHalf := c.RopeDims / 2

	// fp32 weights: four gammas a layer, then one rotary table for all of them.
	perLayer := 2*c.HeadDim + 2*c.IdxDim
	g.wElems = nLayers*perLayer + 2*g.nKV*rotHalf
	g.wRope = uint32(nLayers * perLayer)
	var err error
	if g.wbuf, err = g.dev.NewBuffer(g.wElems * 4); err != nil {
		return fmt.Errorf("llm: attention fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	// The sequence position, and it must be the arena's dword 0: the shaders
	// read it as `actu[0]` (SEQ_PAST in llm_common.glsl), because P1c moved it
	// out of the push block so that a decode step's command buffer is
	// byte-identical every token. SetPast writes it.
	if seq := alloc(1); seq != 0 {
		return fmt.Errorf("llm: the sequence slot is at %d, and SEQ_PAST is actu[0]", seq)
	}
	g.aQKV = alloc(rows * g.qkvN())
	g.aScore = alloc(rows * g.NBlocks())
	g.aCell = alloc(rows * g.nKV)
	// The output gets its own arena rather than being written back over the
	// projection's. It is 5 MB at 512 tokens against the 27 the fused
	// projection holds, and aliasing them would cost the layer's own inputs —
	// which at L6 the residual add still wants — to save that.
	g.aOut = alloc(rows * c.NEmbd)
	// The bitmask is allocated for every arena row, not just the prompt's,
	// because the attention kernel's pad rows index it — they read it as dense
	// rather than branching, and an unallocated row would be a read past the
	// buffer.
	g.aSel = alloc(rows * g.selWords())
	// The decode GEMV's partial sums, f32 [KSLABS][gemmN] (L8e-1). The wider
	// of the two projections is the fused one, so 40 x 13952 x 4 = 2.2 MB —
	// allocated whichever rung runs, because the arena plan is fixed at
	// construction and the rung is not.
	g.aPart = alloc(GEMVMaxRows * gemvMaxSlabs * g.qkvN())
	if g.abuf, err = newArena(g.dev, g.actElems*4); err != nil {
		return fmt.Errorf("llm: attention fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	g.writeSeq()

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	plane := rows * c.HeadDim
	g.hXn = halloc(rows * g.lda)
	g.hQ = halloc(c.NHead * plane)
	g.hCtx = halloc(rows * g.ldCtx)
	g.hIdxQ = halloc(rows * c.IdxHeads * c.IdxDim)

	// The cache, one plane a layer. A query plane is padded token *rows* of
	// one batch; a key plane is the layer's nKV *cells* and outlives it, so
	// the two geometries part company here and the shaders are told both.
	//
	// It is 2.31 KB a cell a layer — 1 KB of key, 1 KB of value, 256 B of the
	// indexer's raw key and 64 B of its pooled block — so twelve layers are
	// 27.7 KB a cell: 57 MB at 2048 cells, 906 MB at 32768. The one structural
	// limit is that it shares a buffer with the arenas above and
	// `maxStorageBufferRange` is 4 GiB - 4, which caps this at about 148k
	// cells; past that the cache wants L6a's array-of-buffers, one a layer.
	g.kvStride = c.NHeadKV * g.nKV * c.HeadDim
	g.idxRawStride = g.nKV * c.IdxDim
	g.idxKStride = g.NBlocks() * c.IdxDim
	g.hK = halloc(nLayers * g.kvStride)
	g.hV = halloc(nLayers * g.kvStride)
	g.hIdxRaw = halloc(nLayers * g.idxRawStride)
	g.hIdxK = halloc(nLayers * g.idxKStride)
	if g.hbuf, err = newArena(g.dev, g.hElems*2); err != nil {
		return fmt.Errorf("llm: attention fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: every A operand's pad columns and every short run's pad
	// rows come from here, and none of these kernels bounds-check.
	g.hbuf.Zero()
	g.abuf.Zero()

	// A layer's two matrices, one after the other. An offset into a
	// quantised bank is a **byte**; there is no fp16 tail — the indexer's
	// two BF16 projections are quantised onto the plane with the rest of the
	// fused matrix (D13's payoff, P2).
	qkvBank, outBank := g.qkvN()*c.NEmbd*2, c.NEmbd*c.GateWidth()*2
	unit := 2
	if g.quant() {
		if bits, ok := qkBank(g.bank); ok {
			if err := qkFits(g.qkvN(), c.NEmbd); err != nil {
				return err
			}
			if err := qkFits(c.NEmbd, c.GateWidth()); err != nil {
				return err
			}
			qkvBank = q8Align(qkBytes(bits, g.qkvN(), c.NEmbd))
			outBank = q8Align(qkBytes(bits, c.NEmbd, c.GateWidth()))
		} else {
			qkvBank, outBank = q8Align(q8Bytes(g.qkvN(), c.NEmbd)), q8Align(q8Bytes(c.NEmbd, c.GateWidth()))
		}
		unit = 1
	}
	perBank := qkvBank + outBank
	if g.wbank, err = g.dev.NewBuffer(nLayers * perBank); err != nil {
		return fmt.Errorf("llm: attention weight bank (%d MB): %w", (nLayers*perBank)>>20, err)
	}
	g.layers = make([]attnLayerWeights, nLayers)
	for i := range g.layers {
		base := uint32(i * perLayer)
		g.layers[i] = attnLayerWeights{
			qkv: uint32(i * perBank / unit),
			out: uint32((i*perBank + qkvBank) / unit),
			gQ:  base,
			gK:  base + uint32(c.HeadDim),
			gIQ: base + uint32(2*c.HeadDim),
			gIK: base + uint32(2*c.HeadDim+c.IdxDim),
		}
	}
	return nil
}

func (g *AttnGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.wbank, g.abuf}
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"pack":  shaders.LLMAttnPack,
		"idx":   shaders.LLMAttnIdx,
		"score": shaders.LLMAttnScore,
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
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
		return fmt.Errorf("llm: the attention and GEMM rungs need a pinned 64-wide subgroup")
	}
	// The selection's bucket search is a subgroup suffix sum over 256 radix
	// buckets held four waves wide, so it wants the wave pinned for the same
	// reason the matrix-core rungs do.
	if err := g.pipeline("select", shaders.LLMAttnSelect, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
	}); err != nil {
		return err
	}
	for _, v := range attnVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// Both builds when the bank is quantised: the two projections read the
	// narrow plane and the indexer's fp16 tail reads halves out of the same
	// buffer, so the block needs a pipeline of each. On the fp16 bank there
	// is no tail and the second set is never built.
	for _, v := range gemmVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// Only the two projections read the bank, and only their quantised build
	// names the sixth buffer; the pack, the indexer, the score and the
	// attention kernel are unchanged.
	gemmBufs := bufs
	if g.quant() {
		gemmBufs = append(append([]*vk.Buffer{}, bufs...), g.wbank)
		for _, v := range gemmBuildsFor(g.bank) {
			if err := g.pipeline(bankPipe(g.bank, v.name), v.spirv, vk.PipelineSpec{
				Buffers: gemmBufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
			}); err != nil {
				return err
			}
		}
	}
	// The decode GEMV, every rung of both banks plus the reduce (L8e-1). They
	// are built whatever the bank is, because the reduce reads neither and
	// the fp16 partials arm is what a block on the fp16 bank runs.
	for name, spirv := range gemvSPIRV {
		// One module, one pipeline per row count (P5b), as the gated
		// DeltaNet's build says.
		for rows := 1; rows <= GEMVMaxRows; rows++ {
			if err := g.pipeline(gemvRowName(name, rows), spirv, vk.PipelineSpec{
				Buffers: gemmBufs, PushConstantSize: pcSize, RequiredSubgroupSize: attnWave,
				SpecConstants: gemvSpec(rows),
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *AttnGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader attn.%s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline attn.%s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage packs each layer's weights and builds the rotary table.
//
// The fused projection is the six matrices laid end to end as output rows of
// one [qkvN, nEmbd] matrix, in the order the shaders derive: query and gate
// interleaved per head (which is the checkpoint's own order), then key, value,
// indexer query, indexer key. The pad rows the column block adds are left
// zero, and their outputs are never read.
func (g *AttnGPU) stage(layers []AttnWeights) error {
	c := g.cfg
	for i, w := range layers {
		for _, t := range []struct {
			name string
			off  uint32
			src  []float32
			want int
		}{
			{"attn_q_norm", g.layers[i].gQ, w.QNorm, c.HeadDim},
			{"attn_k_norm", g.layers[i].gK, w.KNorm, c.HeadDim},
			{"indexer.q_norm", g.layers[i].gIQ, w.IdxQNorm, c.IdxDim},
			{"indexer.k_norm", g.layers[i].gIK, w.IdxKNorm, c.IdxDim},
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
			{"attn_q", w.Q, c.QWidth()},
			{"attn_k", w.K, c.KVWidth()},
			{"attn_v", w.V, c.KVWidth()},
			{"indexer.q_proj", w.IdxQ, c.IdxHeads * c.IdxDim},
			{"indexer.k_proj", w.IdxK, c.IdxDim},
		} {
			if len(t.src) != t.n*c.NEmbd {
				return fmt.Errorf("llm: layer %d %s is %d values, want %d", i, t.name, len(t.src), t.n*c.NEmbd)
			}
		}
		if len(w.O) != c.NEmbd*c.GateWidth() {
			return fmt.Errorf("llm: layer %d attn_output is %d values, want %d", i, len(w.O), c.NEmbd*c.GateWidth())
		}

		base, baseV := g.colK(), g.colV()
		baseIQ, baseIK := g.colIQ(), g.colIK()
		if _, ok := qkBank(g.bank); ok {
			if err := g.stageQ4K(i, w); err != nil {
				return err
			}
			continue
		}
		if g.quant() {
			// The indexer's two BF16 projections go on the plane as int8 —
			// a real re-quantisation where q, k and v round-trip exactly,
			// bounded by L8c-7's +0.005% at half these bits (P2).
			qs := make([]byte, g.qkvN()*c.NEmbd)
			sc := make([]uint16, g.qkvN()*c.NEmbd/q8Group)
			tileBQ8(qs, sc, w.Q, c.QWidth(), c.NEmbd, func(r int) int { return r })
			tileBQ8(qs, sc, w.K, c.KVWidth(), c.NEmbd, func(r int) int { return base + r })
			tileBQ8(qs, sc, w.V, c.KVWidth(), c.NEmbd, func(r int) int { return baseV + r })
			tileBQ8(qs, sc, w.IdxQ, c.IdxHeads*c.IdxDim, c.NEmbd, func(r int) int { return baseIQ + r })
			tileBQ8(qs, sc, w.IdxK, c.IdxDim, c.NEmbd, func(r int) int { return baseIK + r })
			g.wbank.WriteBytesAt(int(g.layers[i].qkv), qs)
			g.wbank.WriteUint16At((int(g.layers[i].qkv)+len(qs))/2, sc)

			oq := make([]byte, c.NEmbd*c.GateWidth())
			os := make([]uint16, c.NEmbd*c.GateWidth()/q8Group)
			tileBQ8(oq, os, w.O, c.NEmbd, c.GateWidth(), func(r int) int { return r })
			g.wbank.WriteBytesAt(int(g.layers[i].out), oq)
			g.wbank.WriteUint16At((int(g.layers[i].out)+len(oq))/2, os)
			continue
		}

		qkv := make([]uint16, g.qkvN()*c.NEmbd)
		tileB(qkv, w.Q, c.QWidth(), c.NEmbd, func(r int) int { return r })
		tileB(qkv, w.K, c.KVWidth(), c.NEmbd, func(r int) int { return base + r })
		tileB(qkv, w.V, c.KVWidth(), c.NEmbd, func(r int) int { return baseV + r })
		tileB(qkv, w.IdxQ, c.IdxHeads*c.IdxDim, c.NEmbd, func(r int) int { return baseIQ + r })
		tileB(qkv, w.IdxK, c.IdxDim, c.NEmbd, func(r int) int { return baseIK + r })
		g.wbank.WriteUint16At(int(g.layers[i].qkv), qkv)

		out := make([]uint16, c.NEmbd*c.GateWidth())
		tileB(out, w.O, c.NEmbd, c.GateWidth(), func(r int) int { return r })
		g.wbank.WriteUint16At(int(g.layers[i].out), out)
	}
	g.wbuf.WriteFloat32At(int(g.wRope), ropeTable(g.cfg, g.nKV))
	return nil
}

// stageQ4K packs one layer onto L8c-4's 4.5-bit bank: LLM.md L8c-7.
//
// **Each source of the fused matrix is quantised under its own name**, for
// the reason L8c-5 gives — the published imatrix has a row per tensor, and
// `attn_q`, `attn_k` and `attn_v` are three of them, so a bank that
// calibrated the fused thing with one row would not be the format the
// simulation measured. `attn_q` is the query *and its gate* interleaved per
// head, which is the checkpoint's own tensor and its own imatrix row.
//
// The indexer's two BF16 projections go in beside them at the same width —
// L8c-7 priced that at +0.005% where the selection bites, which is what
// retired the fp16 tail (D13's payoff, P2).
func (g *AttnGPU) stageQ4K(i int, w AttnWeights) error {
	c := g.cfg
	bits, _ := qkBank(g.bank)
	qkv := newQKStage(bits, g.qkvN(), c.NEmbd)
	for _, t := range []struct {
		name string
		src  []float32
		n    int
		base int
	}{
		{"attn_q.weight", w.Q, c.QWidth(), 0},
		{"attn_k.weight", w.K, c.KVWidth(), g.colK()},
		{"attn_v.weight", w.V, c.KVWidth(), g.colV()},
		{"indexer.q_proj.weight", w.IdxQ, c.IdxHeads * c.IdxDim, g.colIQ()},
		{"indexer.k_proj.weight", w.IdxK, c.IdxDim, g.colIK()},
	} {
		name := fmt.Sprintf("blk.%d.%s", w.Layer, t.name)
		q, qw, err := bankImatrix(name, g.sim)
		if err != nil {
			return err
		}
		base := t.base
		if err := qkv.Fill(t.src, t.n, c.NEmbd,
			func(r int) int { return base + r }, q, qw); err != nil {
			return fmt.Errorf("llm: layer %d %s: %w", i, name, err)
		}
	}
	qkv.WriteTo(g.wbank, int(g.layers[i].qkv))

	name := fmt.Sprintf("blk.%d.attn_output.weight", w.Layer)
	q, qw, err := bankImatrix(name, g.sim)
	if err != nil {
		return err
	}
	out := newQKStage(bits, c.NEmbd, c.GateWidth())
	if err := out.Fill(w.O, c.NEmbd, c.GateWidth(),
		func(r int) int { return r }, q, qw); err != nil {
		return fmt.Errorf("llm: layer %d %s: %w", i, name, err)
	}
	out.WriteTo(g.wbank, int(g.layers[i].out))
	return nil
}

// ropeTable is the rotary's cosines for every cache position followed by its
// sines: [nKV][rot/2] then [nKV][rot/2].
//
// The angles are built here, in float64, rather than from a `pow` per lane,
// for the reason the shader's header gives — the CPU reference walks
// `theta *= base^(-2/rot)` in float64 and a float32 `pow` does not reproduce
// it, which at a long context is an absolute angle error and not a relative
// one. Position is the only input, so one table serves the query, the key and
// the indexer's pooled blocks, whose position is the first cell they cover.
//
// It is the interleaved M-RoPE's angle and not plain NeoX's as a matter of
// derivation rather than of coincidence: this calls the same RoPEMulti the CPU
// reference does, on a basis vector, and reads the rotation back off it. On a
// text batch the two agree bit for bit (`TestRoPEMultiIsNeoXOnText`), and the
// day an image batch makes them differ this table follows without an edit.
func ropeTable(c AttnConfig, nKV int) []float32 {
	rotHalf := c.RopeDims / 2
	out := make([]float32, 2*nKV*rotHalf)
	// A unit vector in dim i and zero in dim i+rotHalf rotates to
	// (cos, sin), so one pass over a [1][rot] probe per position reads the
	// whole row of angles off the reference implementation itself.
	probe := make([]float32, c.RopeDims)
	for pos := 0; pos < nKV; pos++ {
		for i := range probe {
			probe[i] = 0
		}
		for i := 0; i < rotHalf; i++ {
			probe[i] = 1
		}
		RoPEMulti(c, probe, []int32{int32(pos)}, 1, c.RopeDims, 1)
		for i := 0; i < rotHalf; i++ {
			out[pos*rotHalf+i] = probe[i]                     // cos
			out[nKV*rotHalf+pos*rotHalf+i] = probe[i+rotHalf] // sin
		}
	}
	return out
}

// SetPlan chooses the three rungs: the attention tile, and one row block for
// each of the two projections. Every rung is built and they all read the same
// staged weight, so this moves a pipeline and restages nothing.
func (g *AttnGPU) SetPlan(attn AttnKernel, gemm, outGemm GEMMKernel) error {
	if _, ok := attnVariantFor(attn); !ok {
		return fmt.Errorf("llm: no attention kernel %q (have %v)", attn, AttnKernels())
	}
	for _, k := range []GEMMKernel{gemm, outGemm} {
		if _, ok := gemmVariantFor(k); !ok {
			return fmt.Errorf("llm: no GEMM kernel %q (have %v)", k, GEMMKernels())
		}
	}
	g.attn, g.gemm, g.outGemm = attn, gemm, outGemm
	g.autoPlan = false
	return nil
}

// Plan reports the rungs in use.
func (g *AttnGPU) Plan() (AttnKernel, GEMMKernel, GEMMKernel) {
	return g.attn, g.gemm, g.outGemm
}

// SetGemv chooses the two projections' decode rungs, or GEMVOff to leave them
// on llm_gemm.comp. It is separate from SetPlan for the reason DeltaNetGPU's
// is: the GEMV is a different kernel over the same staged weight, its ladder
// does not cross the GEMM's, and the two projections do not want the same
// split because K is 2560 on one and 6144 on the other.
func (g *AttnGPU) SetGemv(qkv, out GEMVKernel) error {
	for _, t := range []struct {
		k     GEMVKernel
		gemmK int
		what  string
	}{{qkv, g.cfg.NEmbd, "qkv"}, {out, g.cfg.GateWidth(), "out"}} {
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
	if (qkv != GEMVOff || out != GEMVOff) && (g.rows < 1 || g.rows > GEMVMaxRows) {
		return fmt.Errorf("llm: the GEMV rungs carry at most %d rows of A (P5b); this batch is %d",
			GEMVMaxRows, g.rows)
	}
	g.qkvGemv, g.outGemv = qkv, out
	g.autoPlan = false
	return nil
}

// Gemv reports the decode rungs in use.
func (g *AttnGPU) Gemv() (GEMVKernel, GEMVKernel) { return g.qkvGemv, g.outGemv }

// PinGemv holds the two projections on llm_gemm.comp whatever the batch, and
// releases them to the measured schedule when off. See Graph.PinSchedule.
func (g *AttnGPU) PinGemv(on bool) {
	g.pinGemv = on
	if on {
		g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
	} else if g.autoPlan {
		g.qkvGemv, g.outGemv = AttnGemvFor(g.rows)
	}
}

// AttnGemvFor is the decode plan: llm_gemv.comp at **up to GEMVMaxRows
// tokens** and the GEMM at every other length. See DNGemvFor for why the
// bound moved from one to two at P5b.
func AttnGemvFor(tokens int) (GEMVKernel, GEMVKernel) {
	if tokens < 1 || tokens > GEMVMaxRows || !DecodeGEMV() {
		return GEMVOff, GEMVOff
	}
	return attnQKVGemv, attnOutGemv
}

// Layers is how many layers are staged, Tokens the longest prompt the arenas
// were built for and NKV the cache's cell count.
func (g *AttnGPU) Layers() int { return len(g.layers) }
func (g *AttnGPU) Tokens() int { return g.tokens }
func (g *AttnGPU) NKV() int    { return g.nKV }

// WeightBytes is what the staged layers cost on the device and
// ActivationBytes what the shared arenas cost.
func (g *AttnGPU) WeightBytes() int { return g.wbuf.Size() + g.wbank.Size() }

// Buffers is how many device allocations the layer holds (L6a).
func (g *AttnGPU) Buffers() int         { return 4 }
func (g *AttnGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Upload writes the layer's input: the hyper-connection block's output
// `hc_mixed`, [T][nEmbd], narrowed into the fused projection's A layout.
func (g *AttnGPU) Upload(xn []float32, nTok int) error {
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
			g.qkvGemv, g.outGemv = AttnGemvFor(nTok)
		}
	}
	slab := make([]uint16, nTok*g.lda)
	narrowRows(slab, xn, nTok, c.NEmbd, g.lda)
	g.hbuf.WriteUint16At(int(g.hXn), slab)
	return nil
}

// graph builds one layer's dispatch sequence, with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes.
func (g *AttnGPU) graph(layer int) ([]vk.MultiDispatch, []string, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	if g.past < 0 || g.past+g.rows > g.nKV {
		return nil, nil, fmt.Errorf("llm: %d tokens at cell %d of a %d-cell cache", g.rows, g.past, g.nKV)
	}
	g.layer = layer
	c := g.cfg
	w := g.layers[layer]
	av, _ := attnVariantFor(g.attn)
	gv, _ := gemmVariantFor(g.gemm)
	ov, _ := gemmVariantFor(g.outGemm)
	// The packed planes are padded to the widest tile, so a short run never
	// reads what a longer one left behind.
	plane := roundUpInt(g.rows, maxInt(av.rows, av.keys))

	base := push{
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd),
		LDA: uint32(g.lda), Eps: math.Float32bits(c.Eps),
		QKVOff: g.aQKV, QOff: g.hQ, CtxOff: g.hCtx, IdxQOff: g.hIdxQ,
		// The cache is this layer's, not the block's: four planes, each
		// strided by the layer that filled it (L7a).
		KOff:    g.hK + uint32(layer*g.kvStride),
		VOff:    g.hV + uint32(layer*g.kvStride),
		IdxKOff: g.hIdxK + uint32(layer*g.idxKStride),
		// ATTN_IDXRAW: a field this layer does not otherwise use, because the
		// push block is full at 64 uints. llm_common.glsl carries the mapping.
		// The position is *not* here any more: SEQ_PAST is dword 0 of the
		// arena (P1c), written by SetPast, so `lowRank` stays zero and the
		// dispatch is byte-identical every decode step.
		LoOff:    g.hIdxRaw + uint32(layer*g.idxRawStride),
		ScoreOff: g.aScore, CellOff: g.aCell, RopeOff: g.wRope,
		GammaOff: w.gQ, GammaKOff: w.gK, GammaIQOff: w.gIQ, GammaIKOff: w.gIK,
		Heads: uint32(c.NHead), KVHeads: uint32(c.NHeadKV), HeadDim: uint32(c.HeadDim),
		NKV: uint32(g.nKV), Plane: uint32(plane), LDCtx: uint32(g.ldCtx),
		IdxHeads: uint32(c.IdxHeads), IdxDim: uint32(c.IdxDim),
		Ratio: uint32(c.Ratio), RotDims: uint32(c.RopeDims),
		AttnScale: math.Float32bits(float32(math.Log2(math.E) / math.Sqrt(float64(c.HeadDim)))),
		GemmN:     uint32(g.qkvN()),
		SelOff:    noW, SelWidth: uint32(g.selWidth()),
	}
	if g.sparse {
		base.SelOff = g.aSel
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// 1. The one fused projection: six of llama.cpp's matrices, one matmul.
	qkv := base
	qkv.XnOff, qkv.OutOff, qkv.BOff = g.hXn, g.aQKV, w.qkv
	qkv.GemmM, qkv.GemmK = uint32(roundUpInt(g.rows, gv.bm)), uint32(c.NEmbd)
	qkvPipe := string(g.gemm)
	if g.quant() {
		qkvPipe = bankPipe(g.bank, g.gemm)
	}
	if ks := gemvSlabs(g.qkvGemv); ks > 0 {
		// The decode kernel: one row, so the parallelism comes from K and not
		// from a sixteen-row fragment of which fifteen rows are padding
		// (L8e-1). The partials ride `resOff`, which this block does not use.
		qkv.ResOff = g.aPart
		add(gemvBankPipeRows(g.qkvGemv, g.bank, g.rows), "qkv", uint32(ks), uint32(g.qkvN()/coopMatTile), qkv)
		if ks > 1 {
			add(gemvSumPipeRows(g.qkvGemv, g.rows), "qkv.sum",
				uint32(roundUpInt(g.qkvN(), attnWave)/attnWave), uint32(g.rows), qkv)
		}
	} else {
		add(qkvPipe, "qkv", uint32(g.qkvN()/attnBN), uint32(roundUpInt(g.rows, gv.bm)/gv.bm), qkv)
	}

	// 2. Norm, rotary and the fragment tiling for q, k and v together, plus
	//    the one plane that is not a head: the indexer's raw key into its own
	//    cell cache, which the pooling behind this dispatch reads.
	add("pack", "pack", uint32(plane/coopMatTile), uint32(c.NHead+2*c.NHeadKV+1), base)

	// 3. The indexer's two operands: one pooled key per **newly completed**
	//    block, one query per (token, head). A block whose cells all existed
	//    before this batch is already in the table and is not dispatched.
	lo, hi := g.blockRange()
	units := maxInt(hi-lo, g.rows)
	add("idx", "idx", uint32(units), uint32(1+c.IdxHeads), base)

	// 4. Its score, the bias, the cells and the causal mask.
	add("score", "score", uint32(g.rows), 1, base)

	// 5. The selection: one workgroup a token, four radix passes over the
	//    cells, a bitmask out. Absent below 2051 cells, where it would name
	//    every cell and the attention kernel's dense arm is the same
	//    computation.
	if g.sparse {
		add("select", "select", uint32(g.rows), 1, base)
	}

	// 6. Causal GQA with the output gate in the epilogue, reading the
	//    selection beside the causal mask.
	add(string(g.attn), "attn", uint32(roundUpInt(g.rows, av.rows)/av.rows), uint32(c.NHead), base)

	// 7. The output projection, off the gated context.
	out := base
	out.XnOff, out.LDA = g.hCtx, uint32(g.ldCtx)
	out.OutOff, out.BOff = g.aOut, w.out
	out.GemmM, out.GemmN, out.GemmK = uint32(roundUpInt(g.rows, ov.bm)), uint32(c.NEmbd), uint32(c.GateWidth())
	outPipe := string(g.outGemm)
	if g.quant() {
		outPipe = bankPipe(g.bank, g.outGemm)
	}
	if ks := gemvSlabs(g.outGemv); ks > 0 {
		out.ResOff = g.aPart
		add(gemvBankPipeRows(g.outGemv, g.bank, g.rows), "out", uint32(ks), uint32(c.NEmbd/coopMatTile), out)
		if ks > 1 {
			add(gemvSumPipeRows(g.outGemv, g.rows), "out.sum",
				uint32(roundUpInt(c.NEmbd, attnWave)/attnWave), uint32(g.rows), out)
		}
	} else {
		add(outPipe, "out", uint32(c.NEmbd/attnBN), uint32(roundUpInt(g.rows, ov.bm)/ov.bm), out)
	}
	return d, kinds, nil
}

// Run executes one layer over whatever Upload left in the arenas.
func (g *AttnGPU) Run(layer int) error {
	d, kinds, err := g.graph(layer)
	if err != nil {
		return err
	}
	// One command buffer for the whole block, not one a dispatch: a submit
	// and a fence wait is ~150 us here and a decode step is a batch of one,
	// so what the sequence costs is how many times it is handed over
	// (LLM.md L7c). And when the graph is recording a whole pass, not even
	// one a block (L7d).
	if g.rec.add(ownAttn, kinds, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: attention, %d dispatches (%v): %w", len(d), kinds, err)
	}
	return nil
}

// Profile times each dispatch on the GPU over iters back-to-back repetitions.
// Wall clock around a run is not a measurement of the layer: it carries the
// upload and the read-back, and this arena reads at 0.2 GB/s.
//
// Every dispatch here is idempotent — each reads arenas an earlier one wrote
// and writes one no earlier one reads — so a profile leaves the same tensors a
// Run does. It is still Run that the correctness tests drive, so that what
// they check is the sequence and not a repetition of it.
func (g *AttnGPU) Profile(layer, iters int) ([]Stage, error) {
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
			return out, fmt.Errorf("llm: attention dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across every staged layer, and is
// the number to quote rather than Profile's.
//
// The difference is the cache — the same argument HCGPU.ProfileSweep makes,
// from the other side. One layer's *weights* are 103 MB and cannot sit in the
// 32 MiB MALL however often a dispatch is repeated, so those reads are already
// cold; it is the activations that a repetition would keep warm and a real
// graph running twelve different layers would not.
func (g *AttnGPU) ProfileSweep(iters int) ([]Stage, error) {
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

// The tensors a run leaves behind, in the layout attn.go's CPU reference uses.

// Out is the layer's output, `attn_output`, [T][nEmbd].
func (g *AttnGPU) Out() []float32 {
	return g.abuf.ReadFloat32At(int(g.aOut), g.rows*g.cfg.NEmbd)
}

// QKV is the fused projection's whole output row, [T][qkvN]. Column returns
// one of its six tensors by the column it starts at, which is how a
// disagreement gets located in a projection instead of in what consumed it.
func (g *AttnGPU) QKV() []float32 {
	return g.abuf.ReadFloat32At(int(g.aQKV), g.rows*g.qkvN())
}

// Column slices one output of the fused projection out of its row: the key at
// ColK, the indexer's raw key at ColIK, and so on.
func (g *AttnGPU) Column(col, width int) []float32 {
	raw := g.QKV()
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*g.qkvN()+col:])
	}
	return out
}

// ColK, ColV, ColIQ and ColIK are where the fused projection's later tensors
// start. The shaders derive the same numbers from the head counts, so these
// are the host's half of that contract.
func (g *AttnGPU) ColK() int  { return g.colK() }
func (g *AttnGPU) ColV() int  { return g.colV() }
func (g *AttnGPU) ColIQ() int { return g.colIQ() }
func (g *AttnGPU) ColIK() int { return g.colIK() }

// Score is the indexer's rectified per-block score, [T][nBlocks].
func (g *AttnGPU) Score() []float32 {
	return g.abuf.ReadFloat32At(int(g.aScore), g.rows*g.NBlocks())
}

// Selection is the QSA bitmask a run left behind: one bit a cell, 32 cells a
// uint, [T][selWords]. SelectedCells unpacks one token's row into the
// ascending cell list `topK` returns, so the two can be compared set for set.
func (g *AttnGPU) Selection() []uint32 {
	return g.abuf.ReadUint32At(int(g.aSel), g.rows*g.selWords())
}

// SelectedCells unpacks token t's row of the bitmask.
func (g *AttnGPU) SelectedCells(mask []uint32, t int) []int32 {
	out := make([]int32, 0, g.selWidth())
	row := mask[t*g.selWords() : (t+1)*g.selWords()]
	for j := 0; j < g.nKV; j++ {
		if row[j>>5]&(1<<uint(j&31)) != 0 {
			out = append(out, int32(j))
		}
	}
	return out
}

// Cells is the same score biased, expanded over the cache and causally
// masked, [T][nKV] — `indexer_score_tokens`, infinities and all.
func (g *AttnGPU) Cells() []float32 {
	return g.abuf.ReadFloat32At(int(g.aCell), g.rows*g.nKV)
}

// IdxK is the pooled, normed and rotated indexer key of the layer the last
// run was for, [nBlocks][idxDim], widened out of the fp16 arena.
func (g *AttnGPU) IdxK() []float32 {
	return g.readF16(g.hIdxK+uint32(g.layer*g.idxKStride), g.NBlocks()*g.cfg.IdxDim)
}

// IdxRaw is the indexer's raw key cache, [nKV][idxDim] — the halves the
// pooling above averages, which is where the reference's fp16 round trip
// happens.
func (g *AttnGPU) IdxRaw() []float32 {
	return g.readF16(g.hIdxRaw+uint32(g.layer*g.idxRawStride), g.nKV*g.cfg.IdxDim)
}

// IdxQ is the indexer's query, [T][idxHeads][idxDim].
func (g *AttnGPU) IdxQ() []float32 {
	return g.readF16(g.hIdxQ, g.rows*g.cfg.IdxHeads*g.cfg.IdxDim)
}

// Context is the gated attention output, [T][nHead*headDim] — `attn_gated`,
// read out of the output projection's A layout, whose row stride is not the
// width.
func (g *AttnGPU) Context() []float32 {
	width := g.cfg.GateWidth()
	raw := g.hbuf.ReadUint16At(int(g.hCtx), (g.rows-1)*g.ldCtx+width)
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.ldCtx : t*g.ldCtx+width] {
			out[t*width+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Q and K are the packed query and key planes, un-tiled back into the CPU
// reference's [T][heads][headDim]. They exist so that a disagreement in the
// attention output can be located in the norm or the rotary instead of being
// attributed to the kernel that consumed them.
func (g *AttnGPU) Q() []float32 {
	av, _ := attnVariantFor(g.attn)
	return g.unpack(g.hQ, g.cfg.NHead, roundUpInt(g.rows, maxInt(av.rows, av.keys)), 0, false)
}

// K and V are the *cache's* rows for this run's tokens: cells past..past+T of
// the layer the last run was for, which on a fresh sequence is 0..T and is
// what every test above L4 compares.
func (g *AttnGPU) K() []float32 {
	return g.unpack(g.hK+uint32(g.layer*g.kvStride), g.cfg.NHeadKV, g.nKV, g.past, false)
}

// V is the same, out of the transposed tiling the value is stored in.
func (g *AttnGPU) V() []float32 {
	return g.unpack(g.hV+uint32(g.layer*g.kvStride), g.cfg.NHeadKV, g.nKV, g.past, true)
}

// unpack reads g.rows rows out of a packed plane of `plane` rows, starting at
// row `first`, and un-tiles them into the CPU reference's [T][heads][headDim].
func (g *AttnGPU) unpack(off uint32, heads, plane, first int, transposed bool) []float32 {
	const tile = coopMatTile
	c := g.cfg
	hdt := c.HeadDim / tile
	raw := g.hbuf.ReadUint16At(int(off), heads*plane*c.HeadDim)
	out := make([]float32, g.rows*heads*c.HeadDim)
	for h := 0; h < heads; h++ {
		for t := 0; t < g.rows; t++ {
			r := first + t
			for d := 0; d < c.HeadDim; d++ {
				idx := (h*(plane/tile)+r/tile)*(hdt*tile*tile) + (d/tile)*(tile*tile)
				if transposed {
					idx += (d%tile)*tile + r%tile
				} else {
					idx += (r%tile)*tile + d%tile
				}
				out[(t*heads+h)*c.HeadDim+d] = safetensors.F16ToF32(raw[idx])
			}
		}
	}
	return out
}

func (g *AttnGPU) readF16(off uint32, n int) []float32 {
	raw := g.hbuf.ReadUint16At(int(off), n)
	out := make([]float32, n)
	for i, h := range raw {
		out[i] = safetensors.F16ToF32(h)
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *AttnGPU) Destroy() {
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

// InPort is the layer's input as the fused projection's A operand wants it:
// fp16 [T][lda], `hc_mixed` narrowed.
func (g *AttnGPU) InPort() Port {
	return Port{Buf: g.hbuf, Off: g.hXn, Stride: g.lda, Width: g.cfg.NEmbd,
		Rows: g.arenaRows, Half: true}
}

// OutPort is the layer's output, `attn_output`: fp32 [T][nEmbd].
func (g *AttnGPU) OutPort() Port {
	return Port{Buf: g.abuf, Off: g.aOut, Stride: g.cfg.NEmbd, Width: g.cfg.NEmbd}
}

// Resize sets the length of the run without writing the input.
func (g *AttnGPU) Resize(nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	g.rows = nTok
	if g.autoPlan {
		g.gemm, g.outGemm = GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)
		if g.pinGemv {
			g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
		} else {
			g.qkvGemv, g.outGemv = AttnGemvFor(nTok)
		}
	}
	return nil
}
