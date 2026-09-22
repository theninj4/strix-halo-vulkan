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
	"math/bits"
	"os"
	"strconv"
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
	AttnQT2KT2 AttnKernel = "qt2_kt2"
	AttnQT1KT1 AttnKernel = "qt1_kt1"
)

type attnVariant struct {
	name       AttnKernel
	spirv      []byte
	qt, ktil   int
	rows, keys int
	// split is the same rung built with -DSPLITK: the key axis cut across
	// workgroups, writing partials for llm_attn_combine.comp instead of the
	// gated context (P8). One a rung, because the rung is a knob.
	split []byte
}

var attnVariants = []attnVariant{
	{AttnQT1KT2, shaders.LLMAttnQT1KT2, 1, 2, 16, 32, shaders.LLMAttnQT1KT2Split},
	{AttnQT1KT4, shaders.LLMAttnQT1KT4, 1, 4, 16, 64, shaders.LLMAttnQT1KT4Split},
	{AttnQT2KT4, shaders.LLMAttnQT2KT4, 2, 4, 32, 64, shaders.LLMAttnQT2KT4Split},
	// P13's probe rung. It is the one shape the ladder never had, and at
	// depth it is the interesting one: a 32-row query tile over a 32-cell key
	// block loads the same K and V for **twice** the queries, so K/V traffic
	// a query falls 1.46x at 128 000 cells while the work a query rises 1.37x
	// (the union over 32 adjacent queries is wider than over 16 — P11-7's
	// table, 23.9% against 17.4%). Which way that trades is exactly the
	// question of whether this kernel is bound by its arithmetic or by its
	// cache reads, and nothing else in the ladder separates the two.
	{AttnQT2KT2, shaders.LLMAttnQT2KT2, 2, 2, 32, 32, shaders.LLMAttnQT2KT2Split},
	// The **narrow** rung, and at depth it is the one with a number behind
	// it. P11-7 measured what the QSA selection lets a tile skip and found
	// "the whole remaining headroom is the 1.40x between a 32-cell block and
	// a 16-cell one" — at 128 000 cells a 16x16 tile keeps 11.7% of the pairs
	// live against 17.4% for the shipped 16x32, which is **1.49x fewer cells
	// a query**. P11-7 then tried to get that by branching on a per-k-tile
	// flag *inside* the unrolled loop and lost 1.18x, and drew the rule: a
	// branch inside a cooperative-matrix loop costs more than the work it
	// removes, so change the loop's granularity instead. This is that change
	// of granularity — a build, not a branch.
	{AttnQT1KT1, shaders.LLMAttnQT1KT1, 1, 1, 16, 16, shaders.LLMAttnQT1KT1Split},
}

// splitPipe names the split build of a rung in the pipeline map.
func splitPipe(k AttnKernel) string { return string(k) + ".split" }

// The split decode attention's constants (P8).
//
// `attnDefaultSplits` is set by the grid the device wants: 24 heads times 16
// slices is 384 single-wave workgroups over 40 compute units, which is where
// a CU first holds enough waves to hide a `coopMatLoad`'s latency — the whole
// finding P8 rests on is that at 24 workgroups it holds one and cannot.
// `attnSplitBlocks` stops a shallow cache from paying a combine dispatch to
// parallelise four blocks.
//
// **16 is a plateau and it was measured, not guessed.** At 65 536 cells on
// the 4-layer probe, `attn.attn.split` runs 0.318 / 0.288 / 0.291 / 0.283 ms
// a token at 8 / 16 / 32 / 64 slices while the combine behind it climbs
// 0.004 / 0.006 / 0.010 / 0.018 — so past 16 the walk stops getting shorter
// and only the fold gets longer. The arena bound is left at twice the
// default so the sweep can be repeated without restaging; it is 25 MB.
const (
	// attnMaxSplits is the *arena* bound — what the partial buffer is sized
	// for and what any pin or override is clamped to. attnDefaultSplits is
	// what a run actually takes, and the two are separate so the count can be
	// swept with LLM_ATTN_SPLITS without restaging.
	attnMaxSplits     = 32
	attnDefaultSplits = 16
	attnSplitBlocks   = 4
	// attnCellSplits is how many ways the indexer's score and its expansion
	// to cells are striped (P9). Unlike the attention's slice count this one
	// is free of every constraint — both kernels write one value per output
	// element and reassociate nothing, so the grid is not observable in the
	// answer — and it is a constant only so that a decode step's command
	// buffer stays byte-identical for P1c's prerecording.
	attnCellSplits = 16
	// attnSplitMaxRows is the widest batch the split path will take, and so
	// the row count the partial arena is sized for. It is the widest rung's
	// query tile: past that the query axis alone already fills the grid and
	// the split buys nothing.
	attnSplitMaxRows = 32
)

// AttnKernels lists the rungs, narrowest first.
func AttnKernels() []AttnKernel {
	return []AttnKernel{AttnQT1KT2, AttnQT1KT4, AttnQT2KT4, AttnQT2KT2, AttnQT1KT1}
}

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
func DefaultAttnKernel() AttnKernel { return AttnKernelFor(false, 0) }

// AttnKernelFor is the rung, and **the selection decides it** (P13).
//
// L2f chose qt1_kt2 on 512-token prefills from cell zero, where the key loop
// is dense and a 32-cell block is two cooperative-matrix tiles of reuse
// against one. That is still the right rung for a dense causal attention and
// it is what the `sparse == false` arm returns.
//
// Where the selection is live it is the wrong one, and P11-7 said so with a
// number before anything was built: what the kernel can skip is a (query
// tile, key block) pair, so the cost is the *union* of sixteen adjacent
// queries' selections at the block's width — 17.4% of the axis at a 32-cell
// block and 11.7% at a 16-cell one at 128 000 cells, which is **1.49x fewer
// cells a query**. P11-7 tried to collect that by branching on a per-k-tile
// flag inside the unrolled loop and lost 1.18x instead, and drew the rule:
// change the loop's granularity, not what happens inside it. A narrower rung
// is that change, and it is a build rather than a branch.
//
// Measured, 4 layers, `-pp 2048`, attention block in ms (two passes of the
// control agreeing to 0.2%):
//
//	rung       depth 0    64 000   128 000
//	qt1_kt4      16.9      280.1     597.4
//	qt1_kt2      16.0      186.3     379.3
//	qt1_kt1      16.3      127.2     250.1     1.46x / 1.52x
//
// The ladder is monotone in the block width at depth and flat at depth zero,
// which is the shape the live-pair table predicts and not the shape L2f's
// reuse argument predicts — because at depth this kernel is not short of
// reuse, it is short of cells it is allowed to skip.
//
// **And it is prefill's rung, not decode's**, which is the same split P8
// drew and for the same reason. At decode the kernel is 24 single-wave
// workgroups that all fit on the device at once, so the dispatch costs one
// wave's *serial walk* and neither its traffic nor its work
// (research/p8-decode-attention-split.md) -- and halving the key block
// doubles the walk. Measured at 48 layers and 128 000 cells: prefill
// 471.7 -> 563.6 tok/s and decode 22.94 -> 22.27, which is the trade in both
// directions on one change. So the narrow rung runs where the query axis
// fills the grid and the wide one runs where it does not, which is exactly
// the `attnSplitMaxRows` boundary the split already turns on.
//
// LLM_ATTN_KERNEL pins a rung for measurement.
func AttnKernelFor(sparse bool, rows int) AttnKernel {
	if k := AttnKernel(os.Getenv("LLM_ATTN_KERNEL")); k != "" {
		if _, ok := attnVariantFor(k); ok {
			return k
		}
	}
	if sparse && rows > attnSplitMaxRows {
		return AttnQT1KT1
	}
	return AttnQT1KT2
}

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
	// The split decode attention's partials (P8), ATTN_SPLIT in the shaders:
	// [heads][splits][rows][headDim] of unnormalised context and the (row
	// max, row sum) pair per slice-row behind it.
	aSplit uint32
	// pinSplits fixes the slice count where a test needs two run lengths on
	// one kernel; 0 leaves attnSplits to decide.
	pinSplits int
	// P13's live-block list: per query tile, a count and then the ascending
	// indices of the key blocks the selection leaves anything in.
	aBlk     uint32
	blkTiles int
	// P14-2's gather: per query tile, the ascending cells its rows select
	// between them and a per-row bitmask over those positions.
	aGath     uint32
	gathTiles int
	// pinGather fixes whether the gather runs: 0 leaves gather() to decide, 1
	// forces it on, -1 forces it off. It is what PinSchedule turns off, for
	// the reason SetGather gives.
	pinGather int
	// expandCells is whether the indexer's score is expanded to one f32 a
	// cache cell (P9's llm_attn_expand.comp) and the selection runs over that
	// tensor, or whether the selection runs over the **block** scores with a
	// per-block weight and the expansion is not computed at all (P14-1).
	//
	// It is off by default and it is decided at construction, not per pass,
	// because it is an arena: the expanded tensor is `rows * nKV` floats —
	// 1.14 GB at 128 000 cells and a 2048-row batch, and 4.6 GB at the 8192
	// rows a prefill at depth wants. LLM_ATTN_EXPAND_CELLS=1 is the control
	// arm, and it is also what Cells() reads.
	expandCells bool
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
func (g *AttnGPU) Sparse() bool { return g.sparse }

// SetSparse takes the rung with it unless a plan has been pinned, because
// which rung is right is a fact about whether the selection is live (P13) --
// so a dense control arm must run the dense ladder's winner or it is two
// changes at once.
func (g *AttnGPU) SetSparse(v bool) {
	g.sparse = v
	if g.autoPlan {
		g.attn = AttnKernelFor(v, g.rows)
	}
}

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
func (g *AttnGPU) blockRange() (int, int) { return g.blockRangeAt(g.past, g.rows) }

// blockRangeAt is that range for a run that has not been staged yet, which is
// what the rollback's snapshot needs: it has to know what the *next* pass will
// overwrite before the pass sets `past` and `rows` (P5c).
func (g *AttnGPU) blockRangeAt(past, rows int) (int, int) {
	if past == 0 {
		return 0, g.NBlocks()
	}
	return past / g.cfg.Ratio, (past + rows) / g.cfg.Ratio
}

// SnapshotBlocks copies the pooled indexer rows the next run would rewrite
// into host memory, and RestoreBlocks puts them back. Together they are this
// block's whole rollback (P5c).
//
// **Everything else here rewinds for free and this one does not.** A KV cell
// past `past + nTok` is masked out of every score, so a rejected token's key
// is unreadable rather than stale; the indexer's raw per-cell key is
// overwritten when the position is re-run. The pooled table is different
// because a block is *final* — it is written the one time a run completes its
// `ratio` cells and never again — so a pass that completes a block out of a
// token that is then rejected leaves a row nothing will rebuild until the
// block's cells are all re-written, and the selection reads it in between.
//
// It is small: a run's range is at most `rows/ratio + 1` blocks of
// `idxDim` halves in each of the staged layers, which is a few kilobytes.
// `past == 0` is refused rather than snapshotted, because a fresh sequence
// rebuilds the whole table and is not a pass anybody rewinds.
func (g *AttnGPU) SnapshotBlocks(past, rows int) ([]uint16, error) {
	if past == 0 {
		return nil, fmt.Errorf("llm: a fresh sequence rebuilds the whole pooled table; it is not a rewindable pass")
	}
	lo, hi := g.blockRangeAt(past, rows)
	if hi <= lo {
		return []uint16{}, nil
	}
	n := (hi - lo) * g.cfg.IdxDim
	out := make([]uint16, 0, len(g.layers)*n)
	for l := range g.layers {
		off := int(g.hIdxK) + l*g.idxKStride + lo*g.cfg.IdxDim
		out = append(out, g.hbuf.ReadUint16At(off, n)...)
	}
	return out, nil
}

// RestoreBlocks writes a snapshot back. The range is recomputed from the same
// two numbers the snapshot was taken with, so a caller hands back the pass's
// own `past` and row count rather than whatever the block is holding now.
func (g *AttnGPU) RestoreBlocks(past, rows int, snap []uint16) error {
	lo, hi := g.blockRangeAt(past, rows)
	if hi <= lo {
		if len(snap) != 0 {
			return fmt.Errorf("llm: a %d-value pooled snapshot against an empty range", len(snap))
		}
		return nil
	}
	n := (hi - lo) * g.cfg.IdxDim
	if len(snap) != len(g.layers)*n {
		return fmt.Errorf("llm: a %d-value pooled snapshot against %d blocks of %d layers",
			len(snap), hi-lo, len(g.layers))
	}
	for l := range g.layers {
		off := int(g.hIdxK) + l*g.idxKStride + lo*g.cfg.IdxDim
		g.hbuf.WriteUint16At(off, snap[l*n:(l+1)*n])
	}
	return nil
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
		pipes:  make(map[string]*vk.ComputePipeline),
		tokens: maxTokens,
		rows:   maxTokens,
		nKV:    nKV,
		lda:    cfg.NEmbd + gemmPad,
		ldCtx:  cfg.GateWidth() + gemmPad,
		attn:   DefaultAttnKernel(),
		gemm:   GEMMKernelFor(maxTokens),
		// P14-1: the expansion to cells is the control arm and the debug read,
		// not the shipped path. It is an arena decision, so it is taken here
		// and not per pass.
		expandCells: os.Getenv("LLM_ATTN_EXPAND_CELLS") == "1",
		outGemm:     OutGEMMKernelFor(maxTokens),
		autoPlan:    true,
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
	// And the rung follows it (P13): the narrow key block is only worth
	// anything where there is something to skip, and `sparse` is exactly
	// where there is.
	g.attn = AttnKernelFor(g.sparse, maxTokens)

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
	// The expanded per-cell score, and **only when something reads it**
	// (P14-1). It is the largest arena in the block by an order — `rows * nKV`
	// floats, 1.14 GB at a 2048-row batch over 139k cells — and the selection
	// no longer needs it.
	if g.expandCells {
		g.aCell = alloc(rows * g.nKV)
	} else {
		g.aCell = noW
	}
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
	// P13's live-block list, one run per query tile: a count and then at most
	// one index per key block of the cache. It is sized for the *narrowest*
	// tile and the *narrowest* block any rung uses -- the rung is a knob and
	// the arena plan is fixed at construction -- so at 139k cells and a 2048-
	// row batch it is 128 tiles x 4345 uints, 2.2 MB beside the 1.1 GB the
	// expanded cell scores already take.
	g.blkTiles = (rows + blkListBM - 1) / blkListBM
	g.aBlk = alloc(g.blkTiles * g.blkStride())
	// P14-2's gather arena, and it is what the expansion's 1.14 GB paid for.
	// Per query tile: a count, at most `BM * selWidth` cell indices, and BM
	// rows of bitmask over them — 197 KB a tile at 139k cells, so 25 MB at a
	// 2048-row batch and 101 MB at 8192. Sized for the *narrowest* tile any
	// rung uses, because the rung is a knob and the arena plan is fixed here.
	if g.sparse {
		g.gathTiles = (rows + gathBM - 1) / gathBM
		g.aGath = alloc(g.gathTiles * g.gathStride())
	} else {
		g.aGath = noW
	}
	// The decode GEMV's partial sums, f32 [KSLABS][gemmN] (L8e-1). The wider
	// of the two projections is the fused one, so 40 x 13952 x 4 = 2.2 MB —
	// allocated whichever rung runs, because the arena plan is fixed at
	// construction and the rung is not.
	g.aPart = alloc(GEMVMaxRows * gemvMaxSlabs * g.qkvN())
	// The split attention's partials, sized for the widest grid it will ever
	// dispatch rather than for the one this run needs — the arena plan is
	// fixed at construction and the slice count is chosen per pass. 24 heads
	// x 16 slices x 32 rows x 256 dims is 12.6 MB beside a cache that is
	// 708 MB at 131k cells.
	splitRows := minInt(attnSplitMaxRows, rows)
	g.aSplit = alloc(c.NHead * attnMaxSplits * splitRows * (c.HeadDim + 2))
	if err := checkBufferRange("attention fp32 arena", g.actElems*4,
		"reduce the batch (-llm-batch / -pp), which is what these arenas are cut against"); err != nil {
		return err
	}
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
	// **The cap, asserted rather than commented** (P14). At 27.7 KB a cell over
	// twelve layers the paragraph above runs out of descriptor range at about
	// 148k cells, and the driver clamps the range instead of refusing it — so a
	// cache past the cap reads as zeros and the model produces plausible, wrong
	// numbers *faster* than the real thing, which is the one failure a
	// benchmark cannot see. It cost this vertical a day's sweep once.
	if err := checkBufferRange("attention fp16 arena", g.hElems*2,
		fmt.Sprintf("this cache is %d cells over %d layers at %.2f KB a cell a layer, "+
			"so the cap is about %d cells — past that the KV planes want one buffer a layer (L6a)",
			g.nKV, nLayers, float64(g.kvStride+g.idxRawStride+g.idxKStride)*2/float64(g.nKV)/1024,
			maxBufferRange/(2*(g.hElems/maxInt(g.nKV, 1))))); err != nil {
		return err
	}
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
	if err := g.pipeline("expand", shaders.LLMAttnExpand, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize,
	}); err != nil {
		return err
	}
	for name, spirv := range map[string][]byte{
		"select":  shaders.LLMAttnSelect,
		selectW1k: shaders.LLMAttnSelectW1024,
		// P14-1's build of the same select over block scores. Both are always
		// compiled — the choice is an arena decision taken at construction and
		// the control arm has to be reachable from an env var.
		selBlk:    shaders.LLMAttnSelBlk,
		selBlkW1k: shaders.LLMAttnSelBlkW1024,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// P14-4's score on the matrix cores: a cooperative matrix wants the wave
	// pinned for the reason every rung here does.
	if err := g.pipeline(scoreWMMA, shaders.LLMAttnScoreWMMA, vk.PipelineSpec{
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
		if err := g.pipeline(splitPipe(v.name), v.split, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// P13's compaction, one build per (query tile, key block) shape, because
	// both extents are compiled into the kernel whose loop will read it. It
	// is a ballot and a prefix over waves, so the wave is pinned for the
	// reason the selection's bucket search is.
	for name, spirv := range blkListPipes {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// P14-2's gather: the two compaction passes and the attention rungs that
	// read what they write. Wave-pinned for the reason every kernel here with a
	// ballot or a cooperative matrix in it is.
	for name, spirv := range gathPipes {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	// The combine is one lane a head dim and reads no matrix core, so it
	// takes the device's own subgroup size rather than pinning one.
	if err := g.pipeline("combine", shaders.LLMAttnCombine, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize,
	}); err != nil {
		return err
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
		g.attn = AttnKernelFor(g.sparse, nTok)
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
// attnSplits is how many ways this pass cuts the key axis, and 1 for the
// unsplit kernel.
//
// **The question it answers is how long one wave's walk is**, not how much
// work the dispatch has. The grid is (query blocks, heads), every workgroup is
// a single wave, and 24 of them fit on this device's 40 compute units at once
// — so a decode step's attention costs one wave walking every key block of the
// cache, and nothing about the other 23 changes that. Measured: cutting the
// grid from 24 workgroups to 2, which is a twelfth of the work and a twelfth
// of the traffic, moved the dispatch 1.00x at every depth. Only cutting the
// walk moves it.
//
// So the split is taken on the one thing that says the walk is the cost: a
// batch no wider than one query tile, which is the decode regime. Past that
// the query axis already fills the grid — at 512 rows it dispatches 768
// workgroups — and the unsplit kernel is what runs.
//
// **The count is a constant, and every other candidate is a bug.** It must not
// depend on `SEQ_PAST`, because a decode step's command buffer is recorded
// once and replayed byte for byte every token (P1c) and a count that moved
// with the depth would stale it. And it must not depend on `nKV` — which is
// the tempting one, since a deep arena is where the split pays — because
// `TestAttnGPUCacheSizeDoesNotChangeTheAnswer` is a gate that the cache's
// *size* changes nothing about the answer, and a slice count read off `nKV`
// would change the order the partial softmaxes are folded together and so
// change the last places of the result. That is the whole class of bug P7
// closed, re-entered through the rounding rather than through the cost. So
// the number is fixed here, the slices' *contents* are the live depth, and a
// slice with no block in it falls through to its epilogue and writes the
// combine's identity.
//
// LLM_ATTN_SPLITS overrides for measurement: 1 is the unsplit kernel, which
// is the control arm this was measured against.
// P13's live-block list: the compaction between the selection and the
// attention kernel, and the arena it writes.
//
// `llm_attn_wmma.comp` skips a key block the selection left empty, and has
// since P7; what it could not do is stop walking the axis to find out. The
// walk is what makes a prefill's attention grow with depth even though the
// *live* count does not -- 515 blocks at 32 000 cells, 518 at 64 000, 696 at
// 128 000 against an axis of 1000, 2000 and 4000 -- and it is paid once per
// head, on sixteen scattered cache lines a block. This turns the question
// into an answer computed once for all twenty-four heads.
const (
	// The narrowest query tile and the narrowest key block any rung uses.
	// The arena is sized for those because the rung is a knob and the arena
	// plan is fixed at construction: every other rung indexes fewer tiles of
	// fewer entries into the same allocation.
	blkListBM = 16
	blkListBN = 16
)

var blkListPipes = map[string][]byte{
	"blocks.bm16_bn32": shaders.LLMAttnBlocksBM16BN32,
	"blocks.bm16_bn64": shaders.LLMAttnBlocksBM16BN64,
	"blocks.bm32_bn64": shaders.LLMAttnBlocksBM32BN64,
	"blocks.bm32_bn32": shaders.LLMAttnBlocksBM32BN32,
	"blocks.bm16_bn16": shaders.LLMAttnBlocksBM16BN16,
}

// blkListPipe names the build for a rung: the list is the OR of the bitmask
// down BM query rows over BN key cells, and both are compiled into the
// kernel that reads it, so the producer has to be compiled the same way.
func blkListPipe(av attnVariant) string {
	return fmt.Sprintf("blocks.bm%d_bn%d", av.rows, av.keys)
}

// blkStride is how many uints one query tile's run holds: the count, then at
// most one index per key block of the cache.
func (g *AttnGPU) blkStride() int { return 1 + (g.nKV+blkListBN-1)/blkListBN }

// blockList is whether this pass builds one, and it is **off by default,
// because it is measured inert at prefill** (P13-3).
//
// The hypothesis was that the axis walk is the depth slope: at 128 000 cells
// the kernel visits 4000 key blocks to do real work in 696, staging sixteen
// scattered mask words at each, once per head. The compaction deletes 83% of
// those visits and costs a thousandth of a millisecond a token -- and
// `attn.attn` does not move, at any depth. The same sixteen lines are read by
// all twenty-four heads of a query tile, a tile's whole mask is 128 KB, and
// the MALL serves it.
//
// That is P11-4's answer a third time and it is the engine's rather than this
// kernel's: *latency and redundant reads that matter at decode do not matter
// at prefill, because prefill has occupancy* -- 3072 single-wave workgroups
// against a decode step's 24.
//
// It is kept rather than reverted for the two places that are not prefill:
// the split decode kernel walks the same axis with no occupancy to hide
// behind, and a per-cell gather would be built on this compaction.
// LLM_ATTN_BLOCK_LIST=1 turns it on; it is bit-identical either way, which
// TestAttnGPUBlockListDoesNotChangeTheAnswer asserts as an equality.
func (g *AttnGPU) blockList() bool { return os.Getenv("LLM_ATTN_BLOCK_LIST") == "1" }

// ---- P14-2: the per-cell gather, and it is the last large term in a prefill
// at depth.
//
// P13 closed the arithmetic of 900 tok/s at 128k and named one route to it. The
// selection skips cells; the kernel skips **blocks**; and at 128 000 cells a
// sixteen-row query tile's live 16-cell blocks hold 24 580 cells to do work on
// the 10 058 its rows actually selected. P13's block list deleted 83% of the
// *visits* and measured 1.00x, because the visits were never the cost — the
// cells were. This deletes the cells: a compaction one granularity finer, and a
// dense attention over what comes out.
//
// The three things that make it affordable, none of which P13 had:
//
//   - **The value plane is cell-major** (P14-2's other half). A selected run is
//     `ratio` consecutive cells, so a run is one 128-byte line per head-dim
//     group in both planes. Transposed, the value's run was four halves out of
//     each of sixteen rows — the whole tile read for a quarter of it.
//   - **The staging is a head-dim group, not a chunk.** 1 KB of LDS, not 32, so
//     the single-wave workgroups stay twenty-deep on a compute unit. Occupancy
//     is the one thing a prefill cannot trade (P13-3).
//   - **The causal test is in the mask.** The gather ANDs each row's own extent
//     into the bits it writes, so the loop has one predicate where the block
//     kernel had three.
//
// It is **prefill's and not decode's**, on P8's rule: a decode step is one query
// tile, its 24 single-wave workgroups all fit at once, and what it costs is one
// wave's serial walk — which a gather in front of it lengthens rather than cuts,
// on top of two dispatches it cannot amortise over 128 tiles.
const (
	// The narrowest query tile any gathered rung uses. The arena is sized for
	// it because the rung is a knob and the arena plan is fixed at
	// construction.
	gathBM = 16
	// The row count at or below which the gather is not taken: the decode
	// regime, where the split kernel owns the axis.
	gathMinRows = attnSplitMaxRows + 1
	// Head-dim groups staged per barrier, and it is a measured rung — see
	// gathGroups. 2 is 1.00x and 4 is 1.34x against, because the barriers were
	// never the cost.
	gathDefaultGroups = 1
	// Query heads a workgroup — see gathHeadsPerWG.
	gathDefaultHeads = 1
)

var gathPipes = map[string][]byte{
	"gather":          shaders.LLMAttnGatherBM16,
	"gathmask":        shaders.LLMAttnGathMaskBM16,
	"gath.qt1_kt1":    shaders.LLMAttnGathQT1KT1,
	"gath.qt1_kt2":    shaders.LLMAttnGathQT1KT2,
	"gath.qt1_kt4":    shaders.LLMAttnGathQT1KT4,
	"gath.qt1_kt1_g2": shaders.LLMAttnGathQT1KT1G2,
	"gath.qt1_kt2_g2": shaders.LLMAttnGathQT1KT2G2,
	"gath.qt1_kt4_g2": shaders.LLMAttnGathQT1KT4G2,
	"gath.qt1_kt1_g4": shaders.LLMAttnGathQT1KT1G4,
	"gath.qt1_kt2_g4": shaders.LLMAttnGathQT1KT2G4,
	"gath.qt1_kt4_g4": shaders.LLMAttnGathQT1KT4G4,
	"gath.qt1_kt1_h2": shaders.LLMAttnGathQT1KT1H2,
	"gath.qt1_kt2_h2": shaders.LLMAttnGathQT1KT2H2,
	"gath.qt1_kt4_h2": shaders.LLMAttnGathQT1KT4H2,
	"gath.qt1_kt1_h4": shaders.LLMAttnGathQT1KT1H4,
	"gath.qt1_kt2_h4": shaders.LLMAttnGathQT1KT2H4,
	"gath.qt1_kt4_h4": shaders.LLMAttnGathQT1KT4H4,
}

// gathMax is the tight bound on one query tile's union: each of its gathBM rows
// names at most `selWidth` cells, and never more than the cache holds. Rounded
// up to 64 so the mask is a whole number of words and the list a whole number of
// key chunks at every rung — `gathMax` in llm_common.glsl, and the two have to
// agree to the word.
func (g *AttnGPU) gathMax() int {
	return (minInt(g.nKV, gathBM*g.selWidth()) + 63) &^ 63
}

// gathStride is how many uints one query tile holds: the count, the cell
// indices, and gathBM rows of bitmask over them.
func (g *AttnGPU) gathStride() int {
	mx := g.gathMax()
	return 1 + mx + gathBM*(mx/32)
}

// gathPipe names the gathered build for a rung and a staging width, or "" where
// there is none. `grp` of 1 leaves the suffix off, so the shipped name is the
// one the ladder started with.
func gathPipe(av attnVariant, grp, hpw int) string {
	if av.qt != 1 || av.ktil > 4 {
		return ""
	}
	base := fmt.Sprintf("gath.qt%d_kt%d", av.qt, av.ktil)
	switch {
	case hpw > 1 && grp > 1:
		return "" // not a build; the two knobs are swept one at a time
	case hpw > 1:
		return fmt.Sprintf("%s_h%d", base, hpw)
	case grp > 1:
		return fmt.Sprintf("%s_g%d", base, grp)
	}
	return base
}

// gathHeadsPerWG is how many query heads share one workgroup, and so one staging
// of the gathered key and value. It has to divide the head count — every wave of
// the workgroup reaches every barrier — so a count that does not is refused back
// to 1 rather than deadlocking.
//
// LLM_ATTN_GATHER_HEADS is the ladder.
func (g *AttnGPU) gathHeadsPerWG() int {
	n := gathDefaultHeads
	if v, err := strconv.Atoi(os.Getenv("LLM_ATTN_GATHER_HEADS")); err == nil && v >= 1 {
		n = v
	}
	if n <= 1 || g.cfg.NHead%n != 0 {
		return 1
	}
	return n
}

// gathGroups is how many head-dim groups the gathered kernel stages per barrier.
// LLM_ATTN_GATHER_GRP is the ladder.
func (g *AttnGPU) gathGroups() int {
	if n, err := strconv.Atoi(os.Getenv("LLM_ATTN_GATHER_GRP")); err == nil && n >= 1 {
		return n
	}
	return gathDefaultGroups
}

// gathRung is which rung a gathered pass runs, and it is **not the block
// kernel's**. P13 took the narrow 16-cell block because at depth this kernel was
// short of cells it was allowed to skip; the gathered axis has nothing left to
// skip, so the block width is decided by reuse again — a 64-cell chunk loads the
// same query fragments for four key tiles — which is the ladder L2f measured
// before depth was in the picture.
func (g *AttnGPU) gathRung() AttnKernel {
	if v := os.Getenv("LLM_ATTN_GATHER_KERNEL"); v != "" {
		return AttnKernel(v)
	}
	return AttnQT1KT2
}

// gather is whether this pass builds the gathered list and runs the kernel over
// it. LLM_ATTN_GATHER=0 is the control arm — the block kernel P13 shipped.
func (g *AttnGPU) gather() bool {
	if g.pinGather != 0 {
		return g.pinGather > 0
	}
	if v := os.Getenv("LLM_ATTN_GATHER"); v != "" {
		return v == "1"
	}
	return g.sparse && g.rows >= gathMinRows
}

// SetGather pins the gather on (1), off (-1) or back to the default (0).
//
// **It exists because the gather is the first kernel in this vertical that is
// not chunk-invariant, and that cannot be fixed.** L7a's gate is that a token
// at cell `pos` comes back bit for bit whether it arrived alone or seventh of
// seven, and every ladder here holds it: a rung tiles the same arithmetic
// differently, P7's block skip drops blocks that contribute nothing, and P8's
// split strides *absolute* key blocks precisely so a slice does not depend on
// how the prompt was cut.
//
// A gathered list cannot have that property. It is the union of a query tile's
// sixteen rows, and a chunk that ends inside the tile has fewer rows to union —
// so the same absolute tile gathers a different list, the list is partitioned
// into different key chunks, and the online softmax folds its rescales in a
// different order. The *set* each row attends over is identical either way, and
// so is every nonzero term in its accumulators; what moves is the grouping, and
// fp32 addition is not associative.
//
// So it goes off with the other five reassociating kernels under
// `Graph.PinSchedule`, and the chunk-equality gates pin it. What is left in the
// shipped path is narrower than it sounds: a prompt prefilled twice at the same
// ubatch is bit-identical, and decode never takes this kernel at all — only
// changing the ubatch moves the last places. `TestAttnGPUGatherIsTheBlockKernel`
// is the tolerance that replaces the equality.
func (g *AttnGPU) SetGather(on bool) {
	if on {
		g.pinGather = 1
	} else {
		g.pinGather = -1
	}
}

// AutoGather gives the default back.
func (g *AttnGPU) AutoGather() { g.pinGather = 0 }

// Gathers reports whether the pass this instance would plan now takes the
// gather.
func (g *AttnGPU) Gathers() bool { return g.gather() }

func (g *AttnGPU) attnSplits(av attnVariant) int {
	if g.rows > av.rows || g.rows > attnSplitMaxRows {
		return 1
	}
	if g.pinSplits > 0 {
		return minInt(g.pinSplits, attnMaxSplits)
	}
	if n, err := strconv.Atoi(os.Getenv("LLM_ATTN_SPLITS")); err == nil && n >= 1 {
		return minInt(n, attnMaxSplits)
	}
	return attnDefaultSplits
}

// cellSplits is how many ways the indexer's score and its expansion to cells
// are striped, and 1 puts both back on the single workgroup a token they ran
// on before P9.
//
// Nothing about the answer depends on it — every output element is one read,
// one add and one store, and which lane does it is not observable — so unlike
// attnSplits this needs no argument about rounding and no place in
// PinSchedule. It is a constant only because a decode step's command buffer
// is recorded once and replayed (P1c). LLM_ATTN_CELL_SPLITS is the control
// arm.
func (g *AttnGPU) cellSplits() uint32 {
	if n, err := strconv.Atoi(os.Getenv("LLM_ATTN_CELL_SPLITS")); err == nil && n >= 1 {
		return uint32(n)
	}
	return attnCellSplits
}

// selectW1k names the sixteen-wave build of the selection in the pipeline map,
// and selBlk/selBlkW1k the two builds of P14-1's block-score select.
const (
	selectW1k = "select.w1024"
	selBlk    = "selblk"
	selBlkW1k = "selblk.w1024"
	scoreWMMA = "score.wmma"
)

// scorePipe is which build of the indexer's score this pass runs, and what the
// grid over it is: the scalar kernel is one workgroup a **token**, the matrix-core
// one is a workgroup a **token tile** (P14-4).
//
// LLM_ATTN_SCORE=scalar is the control arm — the kernel L2e measured, which is
// also the one every tolerance against llama.cpp was originally set on.
func (g *AttnGPU) scorePipe() (pipe string, gx uint32) {
	if os.Getenv("LLM_ATTN_SCORE") == "scalar" {
		return "score", uint32(g.rows)
	}
	return scoreWMMA, uint32(roundUpInt(g.rows, coopMatTile) / coopMatTile)
}

// selectPipe is which build of llm_attn_select.comp this run uses (P10).
//
// The selection is **one workgroup a token and there is no second one to
// give it**: four radix passes over a row are a reduction and not a stripe,
// so unlike the score beside it (P9) it cannot be spread over the grid
// without a global barrier between every pass — which at twelve layers is
// ~96 extra dispatches a step and costs more than it saves. What is left is
// the width of that one workgroup: at 65 536 cells the row does not fit
// `SEL_LDS`, so all five passes stream it out of DRAM, and four waves on one
// compute unit have nothing to hide that behind.
//
// LLM_ATTN_SELECT_WG=256 is the narrow control arm.
// **Which of the two selections** is P14-1's and is not a rung: the block
// select reads a tensor the cell select's expansion writes, so the arena plan
// decides it and `expandCells` carries that decision from construction.
func (g *AttnGPU) selectPipe() string {
	narrow := os.Getenv("LLM_ATTN_SELECT_WG") == "256"
	if g.expandCells {
		if narrow {
			return "select"
		}
		return selectW1k
	}
	if narrow {
		return selBlk
	}
	return selBlkW1k
}

// SetSplits pins how many ways the decode attention cuts its key axis; 1 is
// the unsplit kernel and 0 gives the default back.
//
// It is separate from SetPlan for the reason SetGemv is: it is not a rung of
// the same ladder but a different kernel with a dispatch behind it. It exists
// for the same reason SetPlan does, though — the split is chosen from the
// *run's* length, so a test that compares two run lengths bit for bit has to
// pin it or it is comparing two kernels.
func (g *AttnGPU) SetSplits(n int) { g.pinSplits = n }

// Splits reports the pin, or 0 for the default.
func (g *AttnGPU) Splits() int { return g.pinSplits }

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

	// 4. Its score, then the bias, the cells and the causal mask — two
	//    dispatches (P9), because the expansion reads every block's score and
	//    the barrier that used to enforce that also pinned the scoring to one
	//    workgroup a token. Both are striped `attnCellSplits` ways over the
	//    grid's y. Neither sums anything across the stripe, so the tensors
	//    they write are the same bits however the grid is cut.
	cells := g.cellSplits()
	scorer, scoreX := g.scorePipe()
	add(scorer, "score", scoreX, cells, base)
	// The expansion, and **only when the cell select is the one running**
	// (P14-1): the block select reads the scores directly, with a per-block
	// weight, and gets the same bitmask without this tensor existing.
	if g.expandCells {
		add("expand", "expand", uint32(g.rows), cells, base)
	}

	// 5. The selection: one workgroup a token, four radix passes over the
	//    cells, a bitmask out. Absent below 2051 cells, where it would name
	//    every cell and the attention kernel's dense arm is the same
	//    computation.
	if g.sparse {
		add(g.selectPipe(), "select", uint32(g.rows), 1, base)
	}

	qBlocks := uint32(roundUpInt(g.rows, av.rows) / av.rows)
	sp := g.attnSplits(av)

	// 5a. P14-2's gather, when this pass takes it: the union of each query
	//     tile's rows' selections compacted to an ascending cell list, then the
	//     per-row mask over it. Two dispatches because the mask reads the list,
	//     and both over the *gathered* rung's tile rather than `av`'s — the
	//     gathered kernel is its own build and chooses its own block width, for
	//     the reason gathRung says.
	//
	//     It is exclusive with the split (decode) and with P13's block list
	//     (the coarser compaction it replaces).
	gath, hpw := "", 1
	if g.sparse && sp == 1 && g.gather() {
		gv, ok := attnVariantFor(g.gathRung())
		if ok {
			hpw = g.gathHeadsPerWG()
			gath = gathPipe(gv, g.gathGroups(), hpw)
			if _, ok := g.pipes[gath]; !ok {
				gath, hpw = "", 1
			} else {
				av = gv
				qBlocks = uint32(roundUpInt(g.rows, av.rows) / av.rows)
			}
		}
	}
	if gath != "" {
		gb := base
		gb.ResOff = g.aGath
		gTiles := uint32(roundUpInt(g.rows, gathBM) / gathBM)
		add("gather", "attn.gather", gTiles, 1, gb)
		add("gathmask", "attn.gathmask", gTiles, 1, gb)
	}

	// 5b. The live-block list (P13), when the kernel behind it is the unsplit
	//     one. The split build strides *absolute* key blocks so that a slice
	//     does not depend on how the prompt was chunked (L7a's gate), and a
	//     stride over a compacted list would not have that property — so the
	//     two are exclusive, which costs nothing because the split is decode
	//     and the list is depth.
	blkPipe, haveBlk := "", false
	if g.sparse && sp == 1 && gath == "" && g.blockList() {
		blkPipe = blkListPipe(av)
		_, haveBlk = g.pipes[blkPipe]
	}
	if haveBlk {
		bl := base
		bl.ResOff = g.aBlk
		add(blkPipe, "blocks", qBlocks, 1, bl)
	}

	// 6. Causal GQA with the output gate in the epilogue, reading the
	//    selection beside the causal mask — or, at decode over a cache deep
	//    enough to be worth it, the same kernel with its key axis cut across
	//    workgroups and a combine behind it (P8).
	if gath != "" {
		at := base
		at.ResOff = g.aGath
		// The grid's y is **head groups**, not heads: one workgroup of `hpw`
		// waves covers `hpw` heads and stages the gather once for all of them.
		add(gath, "attn", qBlocks, uint32(c.NHead/hpw), at)
	} else if sp > 1 {
		split := base
		split.ResOff, split.InjOff = g.aSplit, uint32(sp)
		add(splitPipe(g.attn), "attn.split", qBlocks*uint32(sp), uint32(c.NHead), split)
		// Over the whole query tile and not the prompt: the combine writes
		// the context's pad rows too, as the unsplit epilogue does.
		add("combine", "attn.combine", uint32(c.NHead), qBlocks*uint32(av.rows), split)
	} else {
		at := base
		// NO_W and not zero when there is no list: zero is a valid arena
		// offset, and the kernel decides on this field.
		at.ResOff = noW
		if haveBlk {
			at.ResOff = g.aBlk
		}
		add(string(g.attn), "attn", qBlocks, uint32(c.NHead), at)
	}

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

// Gathered reads one query tile's gathered list and its per-row mask out of the
// arena (P14-2): the cells the tile's sixteen rows selected between them, and
// which of them each row can actually see.
//
// `mask[r][i]` is whether row r attends over `cells[i]`. It is the AND of the
// selection and the causal extent, because the gather folds the causal test in
// so the attention loop has none.
func (g *AttnGPU) Gathered(tile int) (cells []int32, mask [][]bool) {
	if !g.sparse || g.aGath == noW {
		return nil, nil
	}
	mx := g.gathMax()
	raw := g.abuf.ReadUint32At(int(g.aGath)+tile*g.gathStride(), g.gathStride())
	cnt := int(raw[0])
	if cnt > mx {
		return nil, nil
	}
	cells = make([]int32, cnt)
	for i := range cells {
		cells[i] = int32(raw[1+i])
	}
	words := mx / 32
	mask = make([][]bool, gathBM)
	for r := range mask {
		row := raw[1+mx+r*words:]
		mask[r] = make([]bool, cnt)
		for i := 0; i < cnt; i++ {
			mask[r][i] = row[i>>5]&(1<<uint(i&31)) != 0
		}
	}
	return cells, mask
}

// GathTiles is how many query tiles the last pass gathered for.
func (g *AttnGPU) GathTiles() int { return roundUpInt(g.rows, gathBM) / gathBM }

// SelSkip is what the attention kernel's block skip can skip, measured on the
// selection a pass actually left behind (P11).
//
// `llm_attn_wmma.comp` walks the key axis in blocks of `bn` cells and tests a
// whole query tile of `bm` rows at once: a block is skipped only when **no**
// row of the tile has a cell selected in it, because a cooperative-matrix
// fragment is 16 rows and there is no finer granularity to mask at. So the
// saving is not the selection's density — 2051 cells of `nKV`, which at 64k
// is 3.2% — it is the density of the **union** over `bm` adjacent queries,
// and how those two differ is a fact about the text rather than about the
// kernel.
//
// It returns the fraction of (query tile, key block) pairs that survive, and
// the same for a tile of one row, which is the floor a per-query gather would
// reach. Both count only blocks inside the tile's causal extent, since the
// kernel never walks past that.
func (g *AttnGPU) SelSkip(mask []uint32, bm, bn int) (tile, perRow float64) {
	words := g.selWords()
	var live, total, rowLive, rowTotal int
	for q0 := 0; q0 < g.rows; q0 += bm {
		rows := minInt(bm, g.rows-q0)
		// The kernel's own bound: `min(n, past + q0 + bm)`.
		last := minInt(g.nKV, g.past+q0+bm)
		nb := (last + bn - 1) / bn
		for b := 0; b < nb; b++ {
			any := false
			for r := 0; r < rows; r++ {
				row := mask[(q0+r)*words:]
				hit := false
				for c := b * bn; c < minInt((b+1)*bn, g.nKV); c++ {
					if row[c>>5]&(1<<uint(c&31)) != 0 {
						hit = true
						break
					}
				}
				if hit {
					rowLive++
					any = true
				}
				rowTotal++
			}
			if any {
				live++
			}
			total++
		}
	}
	if total == 0 || rowTotal == 0 {
		return 0, 0
	}
	return float64(live) / float64(total), float64(rowLive) / float64(rowTotal)
}

// SelUnion prices the gather: how many cells a query tile of `bm` rows would
// have to read if the attention ran over the **union** of its rows'
// selections, against how many the block-granular kernel reads at key block
// `bn`, and against the per-row floor.
//
// The three numbers are the whole of the gather's arithmetic (P13's "what 900
// tok/s at 128k still needs"). `blocks` is what the shipped kernel visits —
// live blocks times bn — `union` is what a per-cell gather would visit, and
// `row` is the mean of one query's own selection, which a 16-row tile cannot
// reach because the fragment is sixteen rows wide.
func (g *AttnGPU) SelUnion(mask []uint32, bm, bn int) (blocks, union, row float64) {
	words := g.selWords()
	var nBlk, nUni, nRow, tiles, rows int
	for q0 := 0; q0 < g.rows; q0 += bm {
		nr := minInt(bm, g.rows-q0)
		last := minInt(g.nKV, g.past+q0+bm)
		u := make([]uint32, (last+31)/32)
		for r := 0; r < nr; r++ {
			src := mask[(q0+r)*words:]
			c := 0
			for w := range u {
				u[w] |= src[w]
				c += bits.OnesCount32(src[w])
			}
			nRow += c
			rows++
		}
		for b := 0; b*bn < last; b++ {
			any := false
			for c := b * bn; c < minInt((b+1)*bn, last) && !any; c++ {
				any = u[c>>5]&(1<<uint(c&31)) != 0
			}
			if any {
				nBlk += bn
			}
		}
		for w := range u {
			nUni += bits.OnesCount32(u[w])
		}
		tiles++
	}
	if tiles == 0 {
		return 0, 0, 0
	}
	return float64(nBlk) / float64(tiles), float64(nUni) / float64(tiles),
		float64(nRow) / float64(maxInt(rows, 1))
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
//
// **It needs LLM_ATTN_EXPAND_CELLS=1**, because since P14-1 nothing in a
// shipped pass computes this tensor: the selection reads the block scores with
// a per-block weight and the arena is not allocated. It returns nil otherwise
// rather than reading 1.14 GB of somebody else's arena.
func (g *AttnGPU) Cells() []float32 {
	if !g.expandCells {
		return nil
	}
	return g.abuf.ReadFloat32At(int(g.aCell), g.rows*g.nKV)
}

// ExpandsCells reports whether this instance computes the per-cell score
// tensor Cells() reads.
func (g *AttnGPU) ExpandsCells() bool { return g.expandCells }

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

// V is the same. **It is no longer transposed** (P14-2): the value plane is
// cell-major like the key, read as a RowMajor B operand rather than a
// ColumnMajor one, so a selected run of `ratio` cells is one line and a gather
// can read it.
func (g *AttnGPU) V() []float32 {
	return g.unpack(g.hV+uint32(g.layer*g.kvStride), g.cfg.NHeadKV, g.nKV, g.past, false)
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
		g.attn = AttnKernelFor(g.sparse, nTok)
		g.gemm, g.outGemm = GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)
		if g.pinGemv {
			g.qkvGemv, g.outGemv = GEMVOff, GEMVOff
		} else {
			g.qkvGemv, g.outGemv = AttnGemvFor(nTok)
		}
	}
	return nil
}
