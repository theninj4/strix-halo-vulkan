package llm

// The hyper-connection block on the device: LLM.md L2's kernel, against the
// CPU reference in hc.go and llama.cpp's own tensors behind it.
//
// L2a made this the vertical's first kernel rather than its last. Per-op
// attribution of llama.cpp's prefill graph puts 30.6% in elementwise glue,
// 12.2% in tiny-N F32 matmul and ~7% in this block's low-rank pair — over
// half the graph, against 1.6% for the gated DeltaNet that three quarters of
// the layers are made of. Almost none of it is arithmetic, so what this file
// is about is how few dispatches and how little DRAM traffic the block can be
// expressed in:
//
//	norm      res -> xn, fp16, in the GEMM's A layout        RMS_NORM + MUL + CONT
//	down      xn -> silu(lo/hc) and inject, one matmul       MUL_MAT x2 + SCALE + SILU
//	up        lo -> sigmoid -> collapsed into mixed          MUL_MAT + SIGMOID + MUL + 3 ADD
//	combine   res += out * 2*sigmoid(inject/hc)              SCALE + SIGMOID + REPEAT + MUL + ADD
//
// Four dispatches against about sixteen, and two tensors that never exist:
// `inject` is four more output columns on the down projection (the
// reference's single worst line — 95 dispatches a graph at 31.8 GFLOP/s,
// 10.3% of prefill) and the [10240, T] gate is consumed inside the up
// projection's epilogue instead of being written to DRAM, which at 512 tokens
// is 21 MB — the whole MALL — per mixer.
//
// The arena and staging machinery is zimage/qwen's, unchanged: four buffers
// bound to every pipeline, tensors addressed by offset, weights packed once
// into the §2.8 fragment tiling, M padded up to the tile because the kernels
// have no bounds checks.

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// coopMatTile is the cooperative-matrix extent, 16x16x16, which is the only
// shape this device reports.
const coopMatTile = 16

// gemmPad is the leading-dimension pad in halves, §2.3's 256 B, on every A
// operand: a K-strided fragment load issues 16 addresses one row stride
// apart, and both 10240 and 320 are multiples of 256.
const gemmPad = 128

// push mirrors the PC block of shaders/llm_common.glsl. One size for every
// pipeline in the vertical, because vk.DispatchMultiTimed records a sequence
// into a single command buffer only if every layout declares the same range —
// so the PLE block's fields sit here too, unused by the hyper-connection
// kernels and zero in their dispatches.
type push struct {
	ResOff, XnOff, LoOff, InjOff uint32
	OutOff, GammaOff, BOff       uint32
	GateOff                      uint32
	Tokens, NEmbd, HC, LowRank   uint32
	LDA, LDALo                   uint32
	GemmM, GemmN, GemmK          uint32
	Eps                          uint32

	KVOff, GatedOff, NormOff        uint32
	ConvOff, ConvOutOff             uint32
	GammaQOff, GammaCOff, Kern, Dil uint32

	// The full-attention layer and the QSA indexer (gpu_attn.go). Named one
	// field per tensor rather than folded onto the spares above, for the
	// reason llm_common.glsl gives: this block has four norm gammas and five
	// activation tensors that exist nowhere else in the vertical, and reading
	// the indexer's key norm through a field called GammaOff is a class of
	// wrong no tolerance catches. 51 uints is 204 bytes against the device's
	// 256.
	QKVOff, QOff, KOff, VOff, CtxOff  uint32
	IdxKOff, IdxQOff                  uint32
	ScoreOff, CellOff, RopeOff        uint32
	GammaKOff, GammaIQOff, GammaIKOff uint32
	Heads, KVHeads, HeadDim           uint32
	NKV, Plane, LDCtx                 uint32
	IdxHeads, IdxDim                  uint32
	SelOff, SelWidth                  uint32
	Ratio, RotDims                    uint32
	AttnScale                         uint32

	// The gated DeltaNet (gpu_deltanet.go). Six fields, because everything
	// else the layer needs is already above under the same meaning — see the
	// note in llm_common.glsl.
	SSMGateOff, SSMBetaOff, SSMStateOff uint32
	SSMAOff, SSMDTOff, SSMNorm          uint32

	// The MoE block (gpu_moe.go). Five fields, and with them the block is
	// **full**: 64 uints is 256 bytes, which is this device's whole
	// push-constant range, and TestAttnGPUPushBlockFits is what guards it.
	// Everything else the MoE needs is said by a field above that already
	// means it, and llm_common.glsl carries the mapping — `xnOff` the block
	// input, `qkvOff` the one fused projection's output (here the router's
	// 512 logits with the shared expert's gate as a 513th column), `ctxOff`
	// the fp16 A operand of the output projection (here `silu(gate)*up`),
	// `gatedOff` "value * gate" (here the down projection's output times its
	// routing weight) and `outOff` the block's output.
	MoEPermOff, MoETileOff, MoEWeightOff uint32
	MoEBOff2, MoEUsed                    uint32
}

func (p push) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*push)(unsafe.Pointer(&out[0])) = p
	return out
}

// noW marks an optional tensor as absent, as NO_W does in the shader.
const noW = 0xffffffff

// HCKernel names one build of shaders/llm_gemm.comp. The two projections
// have different N and different epilogues, so they have separate ladders;
// within each, the rungs differ only in how many token rows a workgroup
// carries, which is what decides occupancy at a short prompt and reuse at a
// long one.
type HCKernel string

const (
	HCDownM1 HCKernel = "down_m1"
	HCDownM2 HCKernel = "down_m2"
	HCDownM4 HCKernel = "down_m4"
	HCDownM8 HCKernel = "down_m8"
	HCUpM1   HCKernel = "up_m1"
	HCUpM2   HCKernel = "up_m2"
	HCUpM4   HCKernel = "up_m4"
	HCUpM8   HCKernel = "up_m8"
	// The decode rungs: the same weight as a split-K GEMV, two dispatches,
	// one token only (L7d). The suffix is how many ways K is split, which is
	// compiled into the SPIR-V pair.
	HCDownGemv8   HCKernel = "down_gemv8"
	HCDownGemv16  HCKernel = "down_gemv16"
	HCDownGemv32  HCKernel = "down_gemv32"
	HCDownGemv40  HCKernel = "down_gemv40"
	HCDownGemv80  HCKernel = "down_gemv80"
	HCDownGemv160 HCKernel = "down_gemv160"
)

// hcVariant is one build's geometry, which the host has to be told because BM
// and BN are compiled into the SPIR-V.
//
// mode 2 is the split-K GEMV, which has no tile at all: `slabs` is how many
// ways it cuts K and `reduce` the second dispatch's SPIR-V, because the pair
// is compiled from one slab count and only makes sense together.
// q8 is the same build over L8's dense bank, where one exists. The reduction
// of a GEMV pair has none: it reads partial sums and touches no weight, so
// the fp16 SPIR-V serves both banks.
type hcVariant struct {
	name   HCKernel
	spirv  []byte
	q8     []byte
	q4     []byte
	mode   int
	bm, bn int
	reduce []byte
	slabs  int
}

// spirvFor is a rung's build for a bank. Every rung has all three.
func (v hcVariant) spirvFor(b DenseBank) []byte {
	switch b {
	case BankQ8:
		return v.q8
	case BankQ4K:
		return v.q4
	}
	return v.spirv
}

var hcVariants = []hcVariant{
	{name: HCDownM1, spirv: shaders.LLMHCDownM1, q8: shaders.LLMHCDownQ8M1, q4: shaders.LLMHCDownQ4M1, mode: 0, bm: 16, bn: 48},
	{name: HCDownM2, spirv: shaders.LLMHCDownM2, q8: shaders.LLMHCDownQ8M2, q4: shaders.LLMHCDownQ4M2, mode: 0, bm: 32, bn: 48},
	{name: HCDownM4, spirv: shaders.LLMHCDownM4, q8: shaders.LLMHCDownQ8M4, q4: shaders.LLMHCDownQ4M4, mode: 0, bm: 64, bn: 48},
	{name: HCDownM8, spirv: shaders.LLMHCDownM8, q8: shaders.LLMHCDownQ8M8, q4: shaders.LLMHCDownQ4M8, mode: 0, bm: 128, bn: 48},
	{name: HCUpM1, spirv: shaders.LLMHCUpM1, q8: shaders.LLMHCUpQ8M1, q4: shaders.LLMHCUpQ4M1, mode: 1, bm: 16, bn: 64},
	{name: HCUpM2, spirv: shaders.LLMHCUpM2, q8: shaders.LLMHCUpQ8M2, q4: shaders.LLMHCUpQ4M2, mode: 1, bm: 32, bn: 64},
	{name: HCUpM4, spirv: shaders.LLMHCUpM4, q8: shaders.LLMHCUpQ8M4, q4: shaders.LLMHCUpQ4M4, mode: 1, bm: 64, bn: 64},
	{name: HCUpM8, spirv: shaders.LLMHCUpM8, q8: shaders.LLMHCUpQ8M8, q4: shaders.LLMHCUpQ4M8, mode: 1, bm: 128, bn: 64},
	{name: HCDownGemv8, spirv: shaders.LLMHCGemvS8, q8: shaders.LLMHCGemvQ8S8, q4: shaders.LLMHCGemvQ4S8,
		mode: 2, reduce: shaders.LLMHCGemvR8, slabs: 8},
	{name: HCDownGemv16, spirv: shaders.LLMHCGemvS16, q8: shaders.LLMHCGemvQ8S16, q4: shaders.LLMHCGemvQ4S16,
		mode: 2, reduce: shaders.LLMHCGemvR16, slabs: 16},
	{name: HCDownGemv32, spirv: shaders.LLMHCGemvS32, q8: shaders.LLMHCGemvQ8S32, q4: shaders.LLMHCGemvQ4S32,
		mode: 2, reduce: shaders.LLMHCGemvR32, slabs: 32},
	{name: HCDownGemv40, spirv: shaders.LLMHCGemvS40, q8: shaders.LLMHCGemvQ8S40, q4: shaders.LLMHCGemvQ4S40,
		mode: 2, reduce: shaders.LLMHCGemvR40, slabs: 40},
	{name: HCDownGemv80, spirv: shaders.LLMHCGemvS80, q8: shaders.LLMHCGemvQ8S80, q4: shaders.LLMHCGemvQ4S80,
		mode: 2, reduce: shaders.LLMHCGemvR80, slabs: 80},
	{name: HCDownGemv160, spirv: shaders.LLMHCGemvS160, q8: shaders.LLMHCGemvQ8S160, q4: shaders.LLMHCGemvQ4S160,
		mode: 2, reduce: shaders.LLMHCGemvR160, slabs: 160},
}

// hcMaxSlabs is the widest split any rung asks for, and so how many rows of
// gemmN the partial-sum scratch has to hold.
const hcMaxSlabs = 160

// DownKernels and UpKernels list the rungs of each ladder, narrowest first.
// The GEMV rungs are not in DownKernels because they only answer at one token;
// DownKernelsAt is the ladder for a given length.
func DownKernels() []HCKernel { return []HCKernel{HCDownM1, HCDownM2, HCDownM4, HCDownM8} }
func UpKernels() []HCKernel   { return []HCKernel{HCUpM1, HCUpM2, HCUpM4, HCUpM8} }

// GemvKernels lists the decode rungs.
func GemvKernels() []HCKernel {
	return []HCKernel{HCDownGemv8, HCDownGemv16, HCDownGemv32, HCDownGemv40, HCDownGemv80, HCDownGemv160}
}

// DownKernelsAt is every rung that can run at a given token count.
func DownKernelsAt(tokens int) []HCKernel {
	if tokens == 1 {
		return append(DownKernels(), GemvKernels()...)
	}
	return DownKernels()
}

// DefaultPlan is the best single pair over the whole prompt range on the
// bank the block was built with, measured by `cmd/llm -hc -ladder`
// (results/l8b_hc.csv). It is never more than 1.05x off the rung that wins at
// a given length.
func DefaultPlan(q8 bool) (HCKernel, HCKernel) { return DefaultPlanBank(bankOf(q8)) }

// DefaultPlanBank is the same by bank, which is the three-valued spelling
// L8c-6 needs.
func DefaultPlanBank(b DenseBank) (HCKernel, HCKernel) {
	switch b {
	case BankQ8, BankQ4K:
		return HCDownM2, HCUpM4
	}
	return HCDownM2, HCUpM2
}

// PlanFor is the measured schedule: which pair wins at a given prompt length,
// **on which bank** — because L8b moved both ladders and did not move them the
// same way. Microseconds a mixer at each projection's best rung
// (results/l8b_hc.csv, results/l8b_hc_fp16.csv):
//
//	         down                     up
//	T      fp16         Q8          fp16        Q8
//	   1   30.0 m160   12.8 gemv32   40.0 m2    16.1 m1
//	  64  234.7 m1    204.2 m1       39.5 m2    30.9 m4
//	 512  254.5 m2    253.2 m1      142.0 m4   152.8 m4
//	2048  457.3 m4    525.6 m4      638.2 m2   694.5 m4
//
// Two things move. **The GEMV ladder slides one rung along §5.1b's 4 KB
// rotation**, which is D12 rather than a surprise: a slab is now
// `(gemmK/16/KSLABS) * 256` bytes and not 512, so the split whose stride
// misses the rotation is 32 and no longer 160 — 5120 bytes against 2048 —
// and 160 is second by 6%. **And the up projection wants a wider row block on
// the Q8 bank than it ever did on halves**, because its unpack costs 256/WM
// element conversions per cooperative-matrix step: up_m4 wins from 64 tokens
// up, where the fp16 arm's own ladder is on m2 at every length but 128 and
// 512. That is also why L8b cut the collapse's scratch to one m-tile — at
// [BM][BN] a rung wide enough to amortise the unpack would not have fitted
// beside it.
//
// The boundaries sit between measured points. The worst a rung named here is
// off the one that wins at its own length is **1.12x**, at 128 tokens on the
// Q8 bank, where up_m8 beats up_m4 by 6 us a mixer; everywhere else it is
// within 1.05x.
func PlanFor(tokens int, q8 bool) (HCKernel, HCKernel) { return PlanForBank(tokens, bankOf(q8)) }

// PlanForBank is the same by bank, and L8c-6 adds the third column.
// Microseconds a *block* at each length's best pair, 24 mixers staged
// (results/l8c_hc_ladder.csv, results/l8c_hc_ladder_q8.csv):
//
//	T        q8                q4_k             q4_k / q8
//	   1    36.9 gemv32/m1     24.6 gemv32/m1     1.50x
//	  64   255.1 m1/up_m4     202.4 m1/up_m4      1.26x
//	 128   289.3 m1/up_m2     241.9 m1/up_m8      1.20x
//	 512   536.9 m2/up_m4     517.1 m1/up_m8      1.04x
//	1024  1629.3 m2/up_m4    1601.1 m2/up_m8      1.02x
//	2048  3303.0 m4/up_m4    3373.3 m4/up_m4      0.98x
//
// **Two things do not move and one does.** The GEMV rung stays at 32, where
// D12 predicted it would slide: a slab is `(gemmK/16/KSLABS) * 128` bytes
// here, half the int8 arm's, so **no** rung is a whole multiple of 4 KB and
// L7d's rule does not decide this ladder at all — yet the spread is still
// 1.65x, with 32 (2560 bytes) fastest and 40 (2048) second-slowest. What the
// two quantised banks share is that the winner is 5/8 of a 4 KB multiple on
// both, which is why nothing moved; it is written down rather than promoted
// to a rule. And the down ladder's row block is the Q8 one shifted a rung
// *narrower* at 512, because what a wide BM buys is reuse of an unpack and
// this bank's unpack has already halved the bytes it feeds on.
//
// **What moves is `up`, and it moves to m8** at every length between 128 and
// 1536 — the projection whose k is 320 and whose unpack is now a nibble, a
// six-bit pair and an affine term per weight. L8b's rule ("the Q8 arm wants a
// wider row block than the fp16 arm ever did, because the unpack costs
// 256/WM conversions a matrix step") applies once more and in the same
// direction.
//
// **And D14's hazard fires at the top of the range**: at ubatch 2048 the
// narrower bank is **0.98x**, the second time in this vertical a narrower
// bank is slower (L8b-5 was the first). Each weight is read 32 times out of
// the 32 MiB MALL there, so only the unpack's ALU is left and this one has
// more of it. Whether that survives into the graph is a different question
// and `-graph` answers it.
func PlanForBank(tokens int, b DenseBank) (HCKernel, HCKernel) {
	if b == BankQ4K {
		switch {
		case tokens == 1:
			return hcQ4DecodePlan()
		case tokens <= 64:
			return HCDownM1, HCUpM4
		case tokens <= 768:
			return HCDownM1, HCUpM8
		case tokens <= 1536:
			return HCDownM2, HCUpM8
		default:
			return HCDownM4, HCUpM4
		}
	}
	q8 := b == BankQ8
	if !q8 {
		switch {
		case tokens == 1:
			return HCDownGemv160, HCUpM2
		case tokens <= 384:
			return HCDownM1, HCUpM2
		case tokens <= 768:
			return HCDownM2, HCUpM4
		case tokens <= 1536:
			return HCDownM2, HCUpM2
		default:
			return HCDownM4, HCUpM2
		}
	}
	switch {
	case tokens == 1:
		// Decode. The GEMM rung's grid is seven workgroups here whatever its
		// BM; the GEMV's is 3360, and 32 is the split whose slab stride
		// misses §5.1b's 4 KB rotation now that a slab is bytes (L8b-2).
		return HCDownGemv32, HCUpM1
	case tokens <= 384:
		return HCDownM1, HCUpM4
	case tokens <= 1536:
		return HCDownM2, HCUpM4
	default:
		return HCDownM4, HCUpM4
	}
}

func hcVariantFor(k HCKernel) (hcVariant, bool) {
	for _, v := range hcVariants {
		if v.name == k {
			return v, true
		}
	}
	return hcVariant{}, false
}

// hcMixer is where one mixer's weights sit in the arenas.
//
// On L8's bank `down` and `up` are **byte** offsets of int8 tiles with an
// fp16 scale plane behind each, and `downTail` is the fp16 remainder of the
// fused down projection — the column block that holds `inject`, in halves.
// On the fp16 bank they are half offsets and there is no tail (bank.go).
type hcMixer struct {
	gamma    uint32 // fp32 arena
	down     uint32 // the fused [lowRank + hc, wide] matrix
	downTail uint32 // halves: its last column block, or noW
	up       uint32 // the permuted [wide, lowRank] matrix
}

// HCGPU runs hyper-connection mixers on the device. It holds however many
// mixers it was built with — at L2 that is the four dumped layers' worth, at
// L6 it will be all 97 — plus one set of activation arenas sized for the
// longest run.
type HCGPU struct {
	dev *vk.Device
	cfg HCConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	mixers []hcMixer

	down, up HCKernel
	// autoPlan re-plans every run for its own length, using PlanFor. It is
	// free — every rung reads the same staged weight, so it changes which
	// pipeline a dispatch names and nothing else — and SetPlan turns it off,
	// because a caller that named a pair meant it.
	autoPlan bool

	// tokens is the longest run the arenas were built for, arenaRows its
	// rounding up to the widest tile any rung covers, and rows the length of
	// the run in progress.
	tokens, arenaRows, rows int
	lda, ldaLo              int

	// fp32 arena.
	aRes, aInject, aMixed, aOut, aGate uint32
	aPart                              uint32
	actElems                           int
	// fp16 arena.
	hXn, hLo uint32
	hElems   int

	// gate is whether the up projection's [wide, T] gate has somewhere to be
	// written. It is validation machinery — the whole point of the kernel is
	// that the tensor does not exist — so it costs an arena only when asked
	// for.
	gate bool
	// q8 is whether the bank is quantised at all — L8's int8-plus-scales or
	// L8c-6's nibbles, rather than the fp16 tiling both replace (bank.go). It
	// changes what stage writes, which pipeline a dispatch names and two
	// fields of its push block; on the int8 bank and nothing else, every
	// tensor it produces is the same to the last place.
	q8 bool
	// dbank is which of the three it is, and sim the format the 4.5-bit one
	// encodes with. Everything the two quantised banks share is said through
	// `q8`; everything that differs is said here.
	dbank DenseBank
	sim   QuantSim
	// names is one tensor prefix a mixer — `blk.7.hc_ffn_` or `output_hc_` —
	// because the 4.5-bit bank looks each matrix's importance row up under
	// its own name, the way L8c-5's fused projection does.
	names []string
	// ctl is the construction-time options, kept because one of them changes
	// how the weights were staged and a reader of a wrong tensor should be
	// able to ask.
	ctl HCOpts
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
}

// hcQ4DecodePlan is the pair the 4.5-bit bank runs at one token, and it is
// named rather than inlined because the GEMV rung is the one number in this
// file D12 says has to be **re-measured** whenever a slab's stride changes.
// L8c-6 is the third bank this ladder has been run on and the first where
// the winner did not move; the next width has to run it again anyway.
func hcQ4DecodePlan() (HCKernel, HCKernel) { return HCDownGemv32, HCUpM1 }

// hcUpSub is the super-block the up rungs were compiled for: `-DQ4K_SUB=10`,
// because the up projection's k is the low rank and 320 is ten groups of 32,
// not a multiple of ggml's 256 (LLM.md L8c-6). It is compiled in for the
// reason BM and BN are — a mismatch would decode a record with the wrong
// scheme rather than run slowly — so the host checks the checkpoint agrees.
const hcUpSub = 10

// downBN is the down ladder's column block. Its N is lowRank + one fragment
// tile = 336 = 21 tiles, and 21 is 3 x 7, so 48 is the widest block that
// divides it — which is why that ladder moves BM alone.
const downBN = 48

// q8Split is where the fused down projection stops being int8 and becomes
// halves, and it is the one number L8b turned on.
//
// `inject` is four **F32** rows at lowRank = 320 of a matrix whose fused N is
// 336, so re-quantising them is L8c's decision and not this stage's (D13).
// L8a's tail takes the rows that are not int8 out of the main plane and
// stages them as a fp16 fragment-tiled matrix of their own — but it requires
// a whole column block to fall on one side of the split, and 320 is not a
// multiple of the down ladder's BN of 48. (48 is forced: 336 is 21 tiles and
// 21 is 3 x 7.)
//
// So the split is the column block that *contains* the first non-Q8 row,
// 288 here, and the 32 low-rank rows between 288 and inject are staged twice
// — once in the int8 plane, where they are never read, and once in the tail,
// where they are. That costs 0.33 MB a mixer against the 3.2 the block saves,
// and it buys the branch staying per workgroup: a compare inside the k-loop
// was 1.27x on a matrix with no tail at all (L8a-3).
//
// Both kernels derive it, from `pc.lowRank` and their own BN — llm_gemm.comp
// MODE 0 from the BN it was built with, llm_hc_gemv.comp from DOWN_BN, which
// build() checks against this one.
func (g *HCGPU) q8Split() int { return g.cfg.LowRank / downBN * downBN }

// q8TailRows is the rest of the fused N: the block the split leaves over.
func (g *HCGPU) q8TailRows() int { return g.gemmN() - g.q8Split() }

// HCQ8Split is that split for a caller with no block staged — the benchmark,
// pricing what the down projection reads off DRAM.
func HCQ8Split(c HCConfig) int { return (&HCGPU{cfg: c}).q8Split() }

// injStride is the inject tensor's row stride: the down projection's last
// fragment tile, not hc. The kernel stores a whole 16-wide tile of which the
// first hc columns mean anything.
func (g *HCGPU) injStride() int { return coopMatTile }

// gemmN is the fused down projection's output width: lowRank plus the tile
// that carries inject, padded up to the down ladder's BN.
func (g *HCGPU) gemmN() int {
	return roundUpInt(g.cfg.LowRank+g.injStride(), downBN)
}

// HCOpts are the choices that change what the block allocates, which is why
// they are made at construction and not per run.
type HCOpts struct {
	// Gate makes the up projection write the 10240-wide gate as well as the
	// collapsed output, into an arena allocated for it. It exists so the
	// fused kernel can be checked against `hc_gate-N` tensor for tensor; the
	// graph the model runs leaves it off, which is the point of the fusion —
	// at 512 tokens that tensor is 21 MB a mixer, the whole MALL.
	Gate bool
	// UnpermutedUp stages the up projection in the checkpoint's own row
	// order while the kernel goes on reading the permuted one. It is a
	// negative control, in the spirit of zimage/qwen's: the collapse depends
	// on a *layout*, and a layout that is load-bearing has to be shown to be
	// — an unpermuted stage produces a perfectly plausible tensor, so nothing
	// but a test that demands it disagree can tell the two apart.
	UnpermutedUp bool
	// Q8 stages the two projections as L8's int8 bank rather than as halves
	// (LLM.md L8b). It is an option and not the only way because the fp16
	// bank is the control the two are compared against, and
	// `LLM_DENSE_FP16=1` is how a whole graph is put back on it.
	Q8 bool
	// Bank overrides Q8 with the whole three-valued choice, and Sim is the
	// format BankQ4K encodes with (LLM.md L8c-6). A zero Bank means "whatever
	// Q8 says", which is what every caller from before L8c-6 means.
	Bank DenseBank
	Sim  QuantSim
}

// bank is that choice resolved: the plan's if it names this family, and the
// two-valued one otherwise.
func (o HCOpts) bank() DenseBank {
	if o.Bank == BankQ4K {
		return BankQ4K
	}
	return bankOf(o.Q8)
}

// HCBankOpts is the bank half of HCOpts as `LLM_DENSE_BANK` and
// `LLM_DENSE_FP16` between them decide it. It is exported because a benchmark
// that staged a different bank from the graph's would be measuring a kernel
// nothing runs, and the two lines that decide it should exist once.
func HCBankOpts() HCOpts {
	o := HCOpts{Q8: DenseQ8()}
	if q, ok := DenseBankPlan().For("blk.0.hc_attn_down.weight", true); ok {
		o.Bank, o.Sim = BankQ4K, q
	}
	return o
}

// NewHCGPU stages mixers onto the device and builds every pipeline the block
// needs. The weights are copied into the arenas here, so the caller may drop
// them afterwards.
func NewHCGPU(dev *vk.Device, cfg HCConfig, maxTokens int, mixers []HCWeights, opts HCOpts) (*HCGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: maxTokens is %d", maxTokens)
	}
	if len(mixers) == 0 {
		return nil, fmt.Errorf("llm: no mixers to stage")
	}
	// HC_STREAMS is compiled into the collapse and cut into the up
	// projection's weight permutation, so a checkpoint with another stream
	// count would be answered wrongly rather than slowly.
	if cfg.HC != 4 {
		return nil, fmt.Errorf("llm: the fused kernel is built for hc=4, this checkpoint says %d", cfg.HC)
	}
	if cfg.NEmbd%coopMatTile != 0 || cfg.LowRank%coopMatTile != 0 {
		return nil, fmt.Errorf("llm: n_embd %d and low rank %d must be multiples of %d",
			cfg.NEmbd, cfg.LowRank, coopMatTile)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	g := &HCGPU{
		dev: dev, cfg: cfg,
		pipes:  make(map[string]*vk.ComputePipeline),
		tokens: maxTokens, rows: maxTokens,
		gate: opts.Gate, ctl: opts,
		dbank: opts.bank(), sim: opts.Sim,
		autoPlan: true,
		lda:      cfg.Wide() + gemmPad,
		ldaLo:    cfg.LowRank + gemmPad,
	}
	g.q8 = g.dbank != BankFP16
	if g.dbank == BankQ4K {
		if opts.Sim.Off() {
			return nil, fmt.Errorf("llm: the 4.5-bit bank needs the format to stage in")
		}
		if err := q4kFits(g.gemmN(), cfg.Wide()); err != nil {
			return nil, fmt.Errorf("llm: hc down projection: %w", err)
		}
		// The up projection is the 320-wide one, and the whole reason this
		// bank needed a second record packing (L8c-6).
		if err := q4kFits(cfg.Wide(), cfg.LowRank); err != nil {
			return nil, fmt.Errorf("llm: hc up projection: %w", err)
		}
		sub, _, _, err := q4kShape(cfg.LowRank)
		if err != nil {
			return nil, err
		}
		if sub != hcUpSub {
			return nil, fmt.Errorf("llm: the up rungs are built for a %d-group super-block, "+
				"a low rank of %d wants %d", hcUpSub, cfg.LowRank, sub)
		}
		// The *down* rungs and `llm_hc_gemv.comp`'s Q4 arm are the plain
		// build, so the fused projection's k has to be ggml's own eight
		// groups; both derive `nsb` as `gemmK >> 8` and index a group with
		// `& 7`, which a 320-wide k would answer wrongly rather than slowly.
		if sub, _, _, err = q4kShape(cfg.Wide()); err != nil {
			return nil, err
		}
		if sub != q4kSuper {
			return nil, fmt.Errorf("llm: the down rungs are built for a %d-group super-block, "+
				"a width of %d wants %d", q4kSuper, cfg.Wide(), sub)
		}
	}
	align := coopMatTile
	for _, v := range hcVariants {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)
	g.down, g.up = DefaultPlanBank(g.dbank)

	if err := g.alloc(len(mixers)); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(mixers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// Bank is which width this block's two projections were staged in.
func (g *HCGPU) Bank() DenseBank { return g.dbank }

// WantGate reports whether this block was built to materialise the gate.
func (g *HCGPU) WantGate() bool { return g.gate }

// alloc lays out the four arenas. Nothing is read here — every size follows
// from the config — so the whole layout can be checked before a weight is
// touched.
func (g *HCGPU) alloc(nMixers int) error {
	c := g.cfg
	rows := g.arenaRows

	total32 := nMixers * c.Wide()
	var err error
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("llm: fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aRes = alloc(rows * c.Wide())
	g.aMixed = alloc(rows * c.NEmbd)
	g.aOut = alloc(rows * c.NEmbd)
	g.aInject = alloc(rows * g.injStride())
	// The GEMV's partial sums, [hcMaxSlabs][gemmN] fp32: 43 KB, and the one
	// tensor in this block that exists because of a grid rather than a
	// formula. It is allocated whatever the run length, because which rung a
	// graph names is decided per run and the arena is not.
	g.aPart = alloc(hcMaxSlabs * g.gemmN())
	g.aGate = noW
	if g.gate {
		g.aGate = alloc(rows * c.Wide())
	}
	if g.abuf, err = newArena(g.dev, g.actElems*4); err != nil {
		return fmt.Errorf("llm: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hXn = halloc(rows * g.lda)
	g.hLo = halloc(rows * g.ldaLo)
	if g.hbuf, err = newArena(g.dev, g.hElems*2); err != nil {
		return fmt.Errorf("llm: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once. The pad columns of every A operand and the pad rows of a
	// short run come from here, and the kernels have no bounds checks.
	g.hbuf.Zero()
	g.abuf.Zero()

	// The two projections, one after the other. In the fp16 bank an offset is
	// a half and a matrix is n*k of them; in L8's it is a byte, a matrix is
	// n*k bytes of int8 tiles plus a scale plane, and the fused down
	// projection carries a fp16 tail between the two (q8Split). Every piece
	// is aligned so that each base is a whole word for the kernel's `uint`
	// view and a whole half for the plane's.
	downBank, upBank, tailBank := g.gemmN()*c.Wide()*2, c.Wide()*c.LowRank*2, 0
	unit := 2
	if g.q8 {
		if g.dbank == BankQ4K {
			// Both planes of both projections, at 4.5 bits. The down
			// projection's plane is still the full fused N — the kernel
			// derives the record plane's base from gemmN * gemmK and gemmN is
			// the stride of the output it writes — so the columns past the
			// split are staged and never read: 0.08 MB a mixer of nibbles
			// against the 1.6 the width saves.
			downBank = q8Align(q4kBytes(g.gemmN(), c.Wide()))
			upBank = q8Align(q4kBytes(c.Wide(), c.LowRank))
		} else {
			downBank = q8Align(q8Bytes(g.gemmN(), c.Wide()))
			upBank = q8Align(q8Bytes(c.Wide(), c.LowRank))
		}
		tailBank = g.q8TailRows() * c.Wide() * 2
		unit = 1
	}
	perMixer := downBank + tailBank + upBank
	if g.bank, err = g.dev.NewBuffer(nMixers * perMixer); err != nil {
		return fmt.Errorf("llm: weight bank (%d MB): %w", (nMixers*perMixer)>>20, err)
	}
	g.mixers = make([]hcMixer, nMixers)
	for i := range g.mixers {
		g.mixers[i] = hcMixer{
			gamma: uint32(i * c.Wide()),
			down:  uint32(i * perMixer / unit),
			// The tail is halves wherever it sits, because the build that
			// reads it is the fp16 one.
			downTail: uint32((i*perMixer + downBank) / 2),
			up:       uint32((i*perMixer + downBank + tailBank) / unit),
		}
		if !g.q8 {
			g.mixers[i].downTail = noW
		}
	}
	return nil
}

// build compiles every pipeline over all four arenas, bound whether the
// shader declares them or not, so one descriptor layout and one push-constant
// size serve the whole sequence.
// gemvDownBN is the column block llm_hc_gemv.comp's Q8 arm compiles in as
// DOWN_BN. The two kernels read one staged bank and have to round `lowRank`
// down to the same boundary, and a mismatch would read the wrong plane rather
// than run slowly, so it is checked rather than trusted.
const gemvDownBN = 48

func (g *HCGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf}
	// The Q8 builds name a sixth buffer: the bank again, as raw words, for
	// the tiles the scale plane at binding 3 belongs to (llm_common.glsl).
	q8bufs := append(append([]*vk.Buffer{}, bufs...), g.bank)
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"norm":    shaders.LLMHCNorm,
		"combine": shaders.LLMHCCombine,
		// P1a: the two of them in one dispatch, for the 94 boundaries of the
		// 96 where a combine is immediately followed by the next mixer's
		// norm. Both spellings stay built: the graph still needs a standalone
		// combine where the PLE block or the final row move comes between,
		// and a standalone norm opens the pass.
		"cn": shaders.LLMHCCN,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	// Every rung is built, so a ladder is one staging and a different
	// pipeline per dispatch rather than a reload.
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
		return fmt.Errorf("llm: the GEMM rungs need a pinned 64-wide subgroup")
	}
	if downBN != gemvDownBN {
		return fmt.Errorf("llm: the down ladder blocks %d columns and the GEMV's Q8 arm was built for %d",
			downBN, gemvDownBN)
	}
	if g.q8 && g.q8Split()%downBN != 0 {
		return fmt.Errorf("llm: the Q8 split is column %d, not a whole %d-column block",
			g.q8Split(), downBN)
	}
	for _, v := range hcVariants {
		if v.mode == 0 && v.bn != downBN {
			return fmt.Errorf("llm: down rung %q has BN %d, but the fused N is padded to %d",
				v.name, v.bn, downBN)
		}
		if v.mode == 1 && v.bn%(coopMatTile*g.cfg.HC) != 0 {
			return fmt.Errorf("llm: up rung %q has BN %d, which is not whole feature blocks", v.name, v.bn)
		}
		if v.mode == 2 {
			// A slab is a whole number of fragment tiles and a step is four
			// of them, so K has to divide evenly twice over; the kernel has
			// no remainder arm and would read the next n-tile's weights.
			if ktiles := g.cfg.Wide() / coopMatTile; ktiles%(v.slabs*4) != 0 {
				return fmt.Errorf("llm: gemv rung %q splits %d tiles %d ways in steps of four",
					v.name, ktiles, v.slabs)
			}
			if v.slabs > hcMaxSlabs {
				return fmt.Errorf("llm: gemv rung %q wants %d slabs, the scratch holds %d",
					v.name, v.slabs, hcMaxSlabs)
			}
			if err := g.pipeline(string(v.name)+"_reduce", v.reduce, vk.PipelineSpec{
				Buffers: bufs, PushConstantSize: pcSize,
			}); err != nil {
				return err
			}
		}
		spirv, pipeBufs := v.spirv, bufs
		if g.q8 {
			// The reduction of a GEMV pair stays on the fp16 build above: it
			// reads partial sums out of the arena and no weight at all. Both
			// quantised banks bind the same sixth buffer — the bank again, as
			// raw words — so only the SPIR-V differs (llm_common.glsl).
			spirv, pipeBufs = v.spirvFor(g.dbank), q8bufs
		}
		if err := g.pipeline(string(v.name), spirv, vk.PipelineSpec{
			Buffers: pipeBufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (g *HCGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage fills the arenas: gamma fp32, and the two packed fp16 matrices.
func (g *HCGPU) stage(mixers []HCWeights) error {
	c := g.cfg
	wide, lr, n := c.Wide(), c.LowRank, g.gemmN()
	g.names = make([]string, len(mixers))
	for i, w := range mixers {
		g.names[i] = w.Name
		if len(w.Norm) != wide {
			return fmt.Errorf("llm: mixer %d norm is %d, want %d", i, len(w.Norm), wide)
		}
		if len(w.Down) != lr*wide {
			return fmt.Errorf("llm: mixer %d down is %d, want %d", i, len(w.Down), lr*wide)
		}
		if len(w.Up) != wide*lr {
			return fmt.Errorf("llm: mixer %d up is %d, want %d", i, len(w.Up), wide*lr)
		}
		if w.Inject != nil && len(w.Inject) != c.HC*wide {
			return fmt.Errorf("llm: mixer %d inject is %d, want %d", i, len(w.Inject), c.HC*wide)
		}
		g.wbuf.WriteFloat32At(int(g.mixers[i].gamma), w.Norm)

		if g.dbank == BankQ4K {
			if err := g.stageQ4(i, w); err != nil {
				return err
			}
			continue
		}
		if g.q8 {
			if err := g.stageQ8(i, w); err != nil {
				return err
			}
			continue
		}

		// **The simulation of the 4.5-bit bank, on the fp16 arm** (L8c-5's
		// precedent). A format handed to the halves is L8c-1's round trip
		// done explicitly rather than through the environment, and it is what
		// lets one process hold the simulated q4_k and the real one and
		// compare a mixer's output value for value. It covers exactly what
		// `stageQ4` quantises — both projections whole, `inject` not at all —
		// so the two arms are the same format on the same weights.
		downSrc, upSrc := w.Down, w.Up
		if !g.sim.Off() {
			for _, t := range []struct {
				suffix string
				src    *[]float32
				k      int
			}{
				{"down.weight", &downSrc, wide},
				{"up.weight", &upSrc, lr},
			} {
				q, qw, err := g.hcImatrix(i, t.suffix)
				if err != nil {
					return err
				}
				x := append([]float32(nil), *t.src...)
				if err := q.ApplyWeighted(x, t.k, qw); err != nil {
					return fmt.Errorf("llm: mixer %d %s%s: %w", i, g.names[i], t.suffix, err)
				}
				*t.src = x
			}
		}

		down := make([]uint16, n*wide)
		packDownB(down, downSrc, w.Inject, lr, c.HC, wide)
		g.bank.WriteUint16At(int(g.mixers[i].down), down)

		up := make([]uint16, wide*lr)
		if g.ctl.UnpermutedUp {
			tileB(up, upSrc, wide, lr, func(o int) int { return o })
		} else {
			packUpB(up, upSrc, wide, lr, c.NEmbd)
		}
		g.bank.WriteUint16At(int(g.mixers[i].up), up)
	}
	return nil
}

// stageQ8 writes one mixer onto L8's bank: the two projections as the int8
// the checkpoint already ships them as, and the down projection's last column
// block as halves behind it.
//
// Both matrices are Q8_0 in the checkpoint, so every value here is the one
// the fp16 bank holds — `float(q) * float(d)` is exact and rounding it to
// fp16 is what tileB wrote (L8a-1). The tail is not a re-quantisation either:
// its 32 low-rank rows are the same halves in both banks and `inject` is F32
// in the checkpoint and fp16 in both.
func (g *HCGPU) stageQ8(i int, w HCWeights) error {
	c := g.cfg
	wide, lr, n := c.Wide(), c.LowRank, g.gemmN()
	split, tailRows := g.q8Split(), g.q8TailRows()

	// The int8 plane is the full fused N rather than the split, because the
	// kernel derives the scale plane's offset from gemmN * gemmK and gemmN is
	// the stride of the output it writes. The columns past the split are
	// staged and never read: 0.16 MB a mixer for an arithmetic the push block
	// has no room to state.
	qs := make([]byte, n*wide)
	sc := make([]uint16, n*wide/q8Group)
	tileBQ8(qs, sc, w.Down, lr, wide, func(r int) int { return r })
	g.bank.WriteBytesAt(int(g.mixers[i].down), qs)
	g.bank.WriteUint16At((int(g.mixers[i].down)+len(qs))/2, sc)

	// The tail: the low-rank rows the split left over, then inject, in a
	// fragment tiling of its own numbered from the split.
	tail := make([]uint16, tailRows*wide)
	tileB(tail, w.Down[split*wide:lr*wide], lr-split, wide, func(r int) int { return r })
	if w.Inject != nil {
		tileB(tail, w.Inject, c.HC, wide, func(r int) int { return lr - split + r })
	}
	g.bank.WriteUint16At(int(g.mixers[i].downTail), tail)

	uq := make([]byte, wide*lr)
	us := make([]uint16, wide*lr/q8Group)
	row := upRow(wide, c.NEmbd)
	if g.ctl.UnpermutedUp {
		row = func(o int) int { return o }
	}
	tileBQ8(uq, us, w.Up, wide, lr, row)
	g.bank.WriteBytesAt(int(g.mixers[i].up), uq)
	g.bank.WriteUint16At((int(g.mixers[i].up)+len(uq))/2, us)
	return nil
}

// hcImatrix is one of a mixer's matrices as the 4.5-bit bank encodes it: the
// format, and the published importance row for that tensor or nil.
//
// **Each matrix is calibrated under its own name**, which is what the
// simulation L8c-3's number came out of did — `hc_attn_up` and `hc_ffn_up`
// are different tensors with very different importance (L8c-2's outlier is
// one of them), and a bank that calibrated a mixer with one row would not be
// the format that was measured.
func (g *HCGPU) hcImatrix(i int, suffix string) (QuantSim, []float32, error) {
	if g.names[i] == "" {
		if quantCalibrated(g.sim.Mode) {
			return g.sim, nil, fmt.Errorf("llm: mixer %d has no tensor name to calibrate %s against", i, suffix)
		}
		return g.sim, nil, nil
	}
	return bankImatrix(g.names[i]+suffix, g.sim)
}

// stageQ4 writes one mixer onto L8c-4's 4.5-bit bank: LLM.md L8c-6.
//
// Three things make it more than stageQ8 with a different tiler.
//
// **The up projection is the 320-wide family**, so its super-block is the
// whole row — ten groups of 32 — and its record is `packScaleMin12`'s twenty
// bytes rather than ggml's sixteen. Nothing here says so: `tileBQ4K` reads it
// off k, exactly as the simulation does, which is what keeps the two the same
// format by construction rather than by agreement.
//
// **The fp16 tail is no longer free.** On L8's bank the 32 low-rank rows
// between the split and `inject` were Q8_0, so the halves in the tail and the
// bytes in the main plane were the same numbers and it did not matter which
// the kernel read. At 4.5 bits it does: the simulation quantises all 320 rows
// of `hc_*_down`, so the tail has to carry the *quantised* rows or the bank
// would be 10% of a matrix more accurate than the format it claims to be.
// They go through the same encoder and are then rounded to halves, which is
// what `sim.go` does one step later.
//
// **`inject` stays exactly where D13 left it**: four F32 rows of the
// checkpoint, staged as halves, never quantised — and the split is still the
// column block that contains them, because the ladder's BN has not moved.
func (g *HCGPU) stageQ4(i int, w HCWeights) error {
	c := g.cfg
	wide, lr, n := c.Wide(), c.LowRank, g.gemmN()
	split, tailRows := g.q8Split(), g.q8TailRows()

	q, qw, err := g.hcImatrix(i, "down.weight")
	if err != nil {
		return err
	}
	dq := make([]byte, n*wide/2)
	drec := make([]byte, q4kRecPlane(n, wide))
	if err := tileBQ4K(dq, drec, w.Down, lr, wide, func(r int) int { return r }, q, qw); err != nil {
		return fmt.Errorf("llm: mixer %d %sdown.weight: %w", i, g.names[i], err)
	}
	g.bank.WriteBytesAt(int(g.mixers[i].down), dq)
	g.bank.WriteBytesAt(int(g.mixers[i].down)+len(dq), drec)

	// The tail, through the same format: the rows the split left over
	// quantised and then narrowed, `inject` narrowed alone.
	tailSrc := append([]float32(nil), w.Down[split*wide:lr*wide]...)
	if err := q.ApplyWeighted(tailSrc, wide, qw); err != nil {
		return fmt.Errorf("llm: mixer %d %sdown.weight tail: %w", i, g.names[i], err)
	}
	tail := make([]uint16, tailRows*wide)
	tileB(tail, tailSrc, lr-split, wide, func(r int) int { return r })
	if w.Inject != nil {
		tileB(tail, w.Inject, c.HC, wide, func(r int) int { return lr - split + r })
	}
	g.bank.WriteUint16At(int(g.mixers[i].downTail), tail)

	if q, qw, err = g.hcImatrix(i, "up.weight"); err != nil {
		return err
	}
	row := upRow(wide, c.NEmbd)
	if g.ctl.UnpermutedUp {
		row = func(o int) int { return o }
	}
	uq := make([]byte, wide*lr/2)
	urec := make([]byte, q4kRecPlane(wide, lr))
	if err := tileBQ4K(uq, urec, w.Up, wide, lr, row, q, qw); err != nil {
		return fmt.Errorf("llm: mixer %d %sup.weight: %w", i, g.names[i], err)
	}
	g.bank.WriteBytesAt(int(g.mixers[i].up), uq)
	g.bank.WriteBytesAt(int(g.mixers[i].up)+len(uq), urec)
	return nil
}

// packChunk is how many output rows one worker narrows at a time.
const packChunk = 64

// tileB writes one [n, k] row-major matrix into the §2.8 fragment tiling:
// tile (nt, kt) is 256 contiguous halves holding element (k, n) at
// (n%16)*16 + k%16, tiles ordered kt-fastest. row(i) says which *packed* row
// output row i of the source occupies, which is how the up projection's
// permutation is expressed without a second copy of the matrix.
func tileB(dst []uint16, src []float32, n, k int, row func(int) int) {
	const tile = coopMatTile
	kt := k / tile
	parallelFor((n+packChunk-1)/packChunk, func(ch int) {
		for i := ch * packChunk; i < minInt((ch+1)*packChunk, n); i++ {
			r := row(i)
			base := (r / tile) * kt * tile * tile
			lane := (r % tile) * tile
			for j, v := range src[i*k : (i+1)*k] {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

// packDownB fuses the two projections that read the same xn into one weight.
//
// This is L2a's finding 3 made into a layout: `w_inject` is a [wide, hc] F32
// matrix applied to the *same* normalised activation `w_down` reads, and
// llama.cpp runs it as its own MUL_MAT at 31.8 GFLOP/s for 10.3% of prefill.
// Here it is rows lowRank..lowRank+hc of a matrix that already has lowRank of
// them, and the kernel tells them apart by which fragment tile a column falls
// in. The rows between hc and the tile, and any tile the ladder's BN pads on
// the end, are left zero.
func packDownB(dst []uint16, down, inject []float32, lowRank, hc, wide int) {
	tileB(dst, down, lowRank, wide, func(i int) int { return i })
	if inject != nil {
		tileB(dst, inject, hc, wide, func(i int) int { return lowRank + i })
	}
}

// packUpB permutes the up projection's output rows so that the four streams
// of one feature land in the same workgroup, which is what lets the kernel
// collapse the gate instead of writing it.
//
// Output row o of the checkpoint is stream c = o/nEmbd, feature i = o%nEmbd.
// It is packed at (i/16)*64 + c*16 + i%16, so a 64-column block is 16
// features x 4 streams and the kernel's epilogue reads column c*16 + r of the
// block for stream c of feature (block*16 + r).
func packUpB(dst []uint16, up []float32, wide, lowRank, nEmbd int) {
	tileB(dst, up, wide, lowRank, upRow(wide, nEmbd))
}

// upRow is that permutation on its own, because L8's bank applies it to the
// same rows through a different tiler (stageQ8) and a second copy of it would
// be a layout stated twice.
func upRow(wide, nEmbd int) func(int) int {
	const tile = coopMatTile
	hc := wide / nEmbd
	return func(o int) int {
		c, i := o/nEmbd, o%nEmbd
		return (i/tile)*(tile*hc) + c*tile + i%tile
	}
}

// SetPlan chooses which rung each projection runs on. Every rung is built and
// they all read the same staged weight, so this moves a pipeline and a tile
// and restages nothing.
func (g *HCGPU) SetPlan(down, up HCKernel) error {
	dv, ok := hcVariantFor(down)
	if !ok || (dv.mode != 0 && dv.mode != 2) {
		return fmt.Errorf("llm: %q is not a down-projection kernel (have %v)", down, DownKernelsAt(g.rows))
	}
	if dv.mode == 2 && g.rows != 1 {
		return fmt.Errorf("llm: %q is the decode rung and this run is %d tokens", down, g.rows)
	}
	uv, ok := hcVariantFor(up)
	if !ok || uv.mode != 1 {
		return fmt.Errorf("llm: %q is not an up-projection kernel (have %v)", up, UpKernels())
	}
	g.down, g.up = down, up
	g.autoPlan = false
	return nil
}

// AutoPlan puts the block back on the measured schedule, undoing a SetPlan.
func (g *HCGPU) AutoPlan() {
	g.autoPlan = true
	g.down, g.up = PlanForBank(g.rows, g.dbank)
}

// Plan reports the rungs in use.
func (g *HCGPU) Plan() (down, up HCKernel) { return g.down, g.up }

// Mixers is how many mixers are staged and Tokens the longest run the arenas
// were built for.
func (g *HCGPU) Mixers() int { return len(g.mixers) }
func (g *HCGPU) Tokens() int { return g.tokens }

// WeightBytes is what the staged mixers cost on the device and
// ActivationBytes what the shared arenas cost.
func (g *HCGPU) WeightBytes() int { return g.wbuf.Size() + g.bank.Size() }

// Buffers is how many device allocations the block holds (L6a).
func (g *HCGPU) Buffers() int         { return 4 }
func (g *HCGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Upload writes the wide residual for a run: [T][hc*nEmbd], which is ggml's
// [nEmbd, hc, T] read the same way round.
func (g *HCGPU) Upload(res []float32, nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(res) != nTok*g.cfg.Wide() {
		return fmt.Errorf("llm: residual is %d values, want %d", len(res), nTok*g.cfg.Wide())
	}
	g.rows = nTok
	if g.autoPlan {
		g.down, g.up = PlanForBank(nTok, g.dbank)
	}
	g.abuf.WriteFloat32At(int(g.aRes), res)
	return nil
}

// UploadInit writes `hc_init` — the embedding repeated into every stream —
// straight into the residual arena.
//
// It is `Upload(HCInit(cfg, embd, nTok), nTok)` without the tensor in
// between. HCInit's result is hc times the embedding, which at ubatch 2048 is
// **84 MB** of Go allocation that exists only to be copied into a mapped
// buffer and dropped; here the four streams are written where they belong and
// the host never holds the wide tensor at all.
func (g *HCGPU) UploadInit(embd []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(embd) != nTok*c.NEmbd {
		return fmt.Errorf("llm: embedding is %d values, want %d", len(embd), nTok*c.NEmbd)
	}
	if err := g.Resize(nTok); err != nil {
		return err
	}
	for t := 0; t < nTok; t++ {
		row := embd[t*c.NEmbd : (t+1)*c.NEmbd]
		for ch := 0; ch < c.HC; ch++ {
			g.abuf.WriteFloat32At(int(g.aRes)+(t*c.HC+ch)*c.NEmbd, row)
		}
	}
	return nil
}

// UploadBlockOut writes the block output the combine scatters back.
func (g *HCGPU) UploadBlockOut(out []float32) error {
	if len(out) != g.rows*g.cfg.NEmbd {
		return fmt.Errorf("llm: block output is %d values, want %d", len(out), g.rows*g.cfg.NEmbd)
	}
	g.abuf.WriteFloat32At(int(g.aOut), out)
	return nil
}

// graph builds one mixer's dispatch sequence, with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes. combine appends the scatter, which needs a block output.
//
// `prev` is P1a's fusion: a mixer >= 0 whose *combine* this graph opens with,
// in the one dispatch that also does this mixer's norm (`llm_hc_cn.comp`).
// -1 is the standalone norm, which is what a bench, a test and the two
// boundaries in the graph that something else sits across all want.
func (g *HCGPU) graph(mixer, prev int, combine bool) ([]vk.MultiDispatch, []string, error) {
	if mixer < 0 || mixer >= len(g.mixers) {
		return nil, nil, fmt.Errorf("llm: mixer %d of %d", mixer, len(g.mixers))
	}
	if prev >= len(g.mixers) {
		return nil, nil, fmt.Errorf("llm: closing mixer %d of %d", prev, len(g.mixers))
	}
	c := g.cfg
	m := g.mixers[mixer]
	dv, _ := hcVariantFor(g.down)
	uv, _ := hcVariantFor(g.up)

	base := push{
		ResOff: g.aRes, XnOff: g.hXn, LoOff: g.hLo, InjOff: g.aInject,
		OutOff: g.aMixed, GammaOff: m.gamma, GateOff: noW,
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd), HC: uint32(c.HC),
		LowRank: uint32(c.LowRank),
		LDA:     uint32(g.lda), LDALo: uint32(g.ldaLo),
		Eps: math.Float32bits(c.Eps),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	if prev >= 0 {
		// The closing mixer's scatter and the opening mixer's norm read and
		// write the same 2560 values per (token, stream) back to back, so
		// they are one workgroup's work rather than two dispatches'. The
		// scatter weights are in the arena and not in the bank — nothing here
		// names `prev` — so what the fusion needs from the push block is the
		// block output and the inject row stride the combine would have set.
		cn := base
		cn.OutOff = g.aOut
		cn.GemmN = uint32(g.gemmN())
		add("cn", "cn", uint32(g.rows), uint32(c.HC), cn)
	} else {
		add("norm", "norm", uint32(g.rows), uint32(c.HC), base)
	}

	down := base
	down.BOff = m.down
	down.GemmN, down.GemmK = uint32(g.gemmN()), uint32(c.Wide())
	// The split, in the one field the down projection does not otherwise use:
	// `gateOff` is where the fp16 tail is, or NO_W on the bank that has none.
	// Where it *begins* is not pushed — both kernels round `lowRank` down to
	// the ladder's BN (q8Split) — because 64 uints is this device's whole
	// push range and the block has been full since L5b.
	down.GateOff = m.downTail
	if dv.mode == 2 {
		if g.rows != 1 {
			return nil, nil, fmt.Errorf("llm: %q is the decode rung and this run is %d tokens", g.down, g.rows)
		}
		// `gammaOff` carries the partial sums: the GEMV reads no norm, and
		// the block is out of push fields (llm_common.glsl).
		down.GammaOff = g.aPart
		down.GemmM = 1
		add(string(g.down), "down", uint32(dv.slabs), uint32(g.gemmN()/coopMatTile), down)
		add(string(g.down)+"_reduce", "down_reduce", uint32(roundUpInt(g.gemmN(), 64)/64), 1, down)
	} else {
		down.GemmM = uint32(roundUpInt(g.rows, dv.bm))
		add(string(g.down), "down", uint32(g.gemmN()/dv.bn), uint32(roundUpInt(g.rows, dv.bm)/dv.bm), down)
	}

	up := base
	up.BOff = m.up
	up.GemmM, up.GemmN, up.GemmK = uint32(roundUpInt(g.rows, uv.bm)), uint32(c.Wide()), uint32(c.LowRank)
	if g.gate {
		up.GateOff = g.aGate
	}
	add(string(g.up), "up", uint32(c.Wide()/uv.bn), uint32(roundUpInt(g.rows, uv.bm)/uv.bm), up)

	if combine {
		cb := base
		cb.OutOff = g.aOut
		cb.GemmN = uint32(g.gemmN())
		add("combine", "combine", uint32(g.rows), uint32(c.HC), cb)
	}
	return d, kinds, nil
}

// RunCombine executes the scatter alone, over whatever `inject` the mixer's
// own Run left in the arena and whatever block output UploadBlockOut wrote.
//
// It exists because the graph is not `Run(mixer, true)`. A layer is mix, then
// a block, then combine: the mix produces the block's input *and* the scatter
// weights, the block runs on another set of arenas, and only then does the
// residual move. `Run(mixer, true)` would recompute the mix on the way to the
// combine, which is the same answer at 1.4 ms a mixer and 97 mixers a graph.
//
// The scatter reads nothing that names the mixer — the weights are in
// `aInject` and not in the bank — so the argument is here for the error
// message and for the reader, who should be able to see which mix a combine
// closes.
func (g *HCGPU) RunCombine(mixer int) error {
	if mixer < 0 || mixer >= len(g.mixers) {
		return fmt.Errorf("llm: mixer %d of %d", mixer, len(g.mixers))
	}
	c := g.cfg
	pc := push{
		ResOff: g.aRes, InjOff: g.aInject, OutOff: g.aOut,
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd), HC: uint32(c.HC),
		LowRank: uint32(c.LowRank), GemmN: uint32(g.gemmN()),
	}
	d := []vk.MultiDispatch{{
		Pipeline: g.pipes["combine"],
		GroupsX:  uint32(g.rows), GroupsY: uint32(c.HC),
		PushConstants: pc.bytes(),
	}}
	if g.rec.add(ownHC, []string{"combine"}, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: mixer %d combine: %w", mixer, err)
	}
	return nil
}

// perSubmit is how many dispatches go into one command buffer.
const perSubmit = 8

// Run executes one mixer over whatever Upload left in the residual.
func (g *HCGPU) Run(mixer int, combine bool) error {
	d, kinds, err := g.graph(mixer, -1, combine)
	if err != nil {
		return err
	}
	if g.rec.add(ownHC, kinds, d) {
		return nil
	}
	for i := 0; i < len(d); i += perSubmit {
		j := minInt(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("llm: mixer %d dispatch %d-%d: %w", mixer, i, j-1, err)
		}
	}
	return nil
}

// RunCombineMix closes mixer `prev` and opens mixer `mixer` in one sequence,
// with the scatter and the norm fused into a single dispatch (P1a).
//
// It is `RunCombine(prev)` followed by `Run(mixer, false)` and it is the
// same arithmetic to the last place — the combine is elementwise and the
// norm's sum keeps its 256-thread partition — but it is four dispatches
// where the pair is five, and it reads the wide residual once where the pair
// reads it twice. The graph uses it at every boundary where nothing else
// touches `res` in between, which is 94 of the model's 96 combines: the PLE
// block moves the residual out and back at its own layer, and the final
// mixer is reached through a row move.
func (g *HCGPU) RunCombineMix(prev, mixer int) error {
	if prev < 0 {
		return fmt.Errorf("llm: RunCombineMix needs a mixer to close, not %d", prev)
	}
	d, kinds, err := g.graph(mixer, prev, false)
	if err != nil {
		return err
	}
	if g.rec.add(ownHC, kinds, d) {
		return nil
	}
	for i := 0; i < len(d); i += perSubmit {
		j := minInt(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("llm: mixer %d after %d, dispatch %d-%d: %w", mixer, prev, i, j-1, err)
		}
	}
	return nil
}

// Stage is one timed dispatch of a profile.
type Stage struct {
	Kind string
	GPU  time.Duration
}

// Elapsed sums a profile.
func Elapsed(st []Stage) time.Duration {
	var total time.Duration
	for _, s := range st {
		total += s.GPU
	}
	return total
}

// Profile runs the same graph one dispatch at a time, each timed on the GPU
// over `iters` back-to-back repetitions. Wall clock around a run is not a
// measurement of the block: it carries the upload and the read-back, and this
// arena reads at 0.2 GB/s.
func (g *HCGPU) Profile(mixer int, combine bool, iters int) ([]Stage, error) {
	d, kinds, err := g.graph(mixer, -1, combine)
	if err != nil {
		return nil, err
	}
	if iters <= 0 {
		iters = 1
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across *every* staged mixer, and
// is the number to quote rather than Profile's.
//
// The difference is the cache. One mixer is 13.4 MB of fp16 weights, which
// sits inside the 32 MiB MALL, so repeating one dispatch measures a kernel
// reading L3 — where the graph this is a model of reads 97 different mixers
// and every one of them is cold. Sweeping the staged set puts that back, as
// long as the caller stages enough of them to overflow the MALL.
func (g *HCGPU) ProfileSweep(combine bool, iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	_, kinds, err := g.graph(0, -1, combine)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(kinds))
	for k := range kinds {
		var d []vk.MultiDispatch
		for m := range g.mixers {
			dm, _, err := g.graph(m, -1, combine)
			if err != nil {
				return nil, err
			}
			d = append(d, dm[k])
		}
		var total time.Duration
		for i := 0; i < len(d); i += perSubmit {
			j := minInt(i+perSubmit, len(d))
			dur, err := vk.DispatchMultiTimed(d[i:j], 1, uint32(iters), true)
			if err != nil {
				return out, fmt.Errorf("llm: %s sweep %d-%d: %w", kinds[k], i, j-1, err)
			}
			total += dur
		}
		out = append(out, Stage{Kind: kinds[k], GPU: total / time.Duration(len(d)*iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the layout hc.go's CPU reference uses,
// so a comparison is a call to compare and not a reshape.

// Mixed is the block's input, [T][nEmbd].
func (g *HCGPU) Mixed() []float32 { return g.abuf.ReadFloat32At(int(g.aMixed), g.rows*g.cfg.NEmbd) }

// Res is the wide residual, [T][hc*nEmbd] — what the combine updated.
func (g *HCGPU) Res() []float32 { return g.abuf.ReadFloat32At(int(g.aRes), g.rows*g.cfg.Wide()) }

// Inject is the scatter weights, [T][hc], gathered out of the tile-strided
// rows the down projection writes.
func (g *HCGPU) Inject() []float32 {
	raw := g.abuf.ReadFloat32At(int(g.aInject), g.rows*g.injStride())
	out := make([]float32, g.rows*g.cfg.HC)
	for t := 0; t < g.rows; t++ {
		copy(out[t*g.cfg.HC:(t+1)*g.cfg.HC], raw[t*g.injStride():])
	}
	return out
}

// Gate is the [T][hc*nEmbd] gate, and is empty unless the block was built to
// write it.
func (g *HCGPU) Gate() []float32 {
	if !g.gate {
		return nil
	}
	return g.abuf.ReadFloat32At(int(g.aGate), g.rows*g.cfg.Wide())
}

// Xn is the normalised activation, widened out of the fp16 arena. The row
// stride is the A operand's leading dimension, not the width.
func (g *HCGPU) Xn() []float32 {
	wide := g.cfg.Wide()
	raw := g.hbuf.ReadUint16At(int(g.hXn), (g.rows-1)*g.lda+wide)
	out := make([]float32, g.rows*wide)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.lda : t*g.lda+wide] {
			out[t*wide+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Lo is silu(down·xn / hc), the up projection's A operand, widened out of the
// fp16 arena.
func (g *HCGPU) Lo() []float32 {
	lr := g.cfg.LowRank
	raw := g.hbuf.ReadUint16At(int(g.hLo), (g.rows-1)*g.ldaLo+lr)
	out := make([]float32, g.rows*lr)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.ldaLo : t*g.ldaLo+lr] {
			out[t*lr+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *HCGPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.hbuf, g.abuf, g.wbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// canWMMA reports whether this device has the cooperative-matrix shape the
// GEMM rungs are built for. A device reporting some other shape would run
// them *wrong* rather than slowly, so the answer is "no" rather than "try".
func canWMMA(dev *vk.Device) (bool, error) {
	feat := dev.Features()
	if !feat.Float16 || !feat.CoopMatrix {
		return false, nil
	}
	shapes, err := dev.Physical().CooperativeMatrixShapes()
	if err != nil {
		return false, fmt.Errorf("llm: cooperative-matrix shapes: %w", err)
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

func roundUpInt(n, m int) int { return (n + m - 1) / m * m }

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// The graph's view of this block's arena (LLM.md L6c). Three tensors cross
// the boundary: the mixer's output goes to a sublayer, the sublayer's output
// comes back for the combine, and the wide residual goes to the PLE block and
// returns. Everything else the block holds stays inside it.

// MixedPort is `hc_mixed` as the mixer leaves it: fp32 [T][nEmbd].
func (g *HCGPU) MixedPort() Port {
	return Port{Buf: g.abuf, Off: g.aMixed, Stride: g.cfg.NEmbd, Width: g.cfg.NEmbd}
}

// MixedRowPort is that same tensor from row t on, for a caller that reads it
// a slab at a time rather than whole. The head's logit arena is 0.99 MB a
// row, so a perplexity run over 2048 tokens takes the mixer's output in
// pieces the head is built for (L8c).
func (g *HCGPU) MixedRowPort(t int) Port {
	n := g.cfg.NEmbd
	return Port{Buf: g.abuf, Off: g.aMixed + uint32(t*n), Stride: n, Width: n}
}

// BlockOutPort is where the combine reads a sublayer's output: fp32
// [T][nEmbd]. The combine has no row block — one workgroup is one (token,
// stream) — so nothing past the run's tokens needs writing.
func (g *HCGPU) BlockOutPort() Port {
	return Port{Buf: g.abuf, Off: g.aOut, Stride: g.cfg.NEmbd, Width: g.cfg.NEmbd, Rows: g.rows}
}

// ResPort is the wide residual: fp32 [T][hc*nEmbd], the one tensor here that
// another block both reads and writes.
func (g *HCGPU) ResPort() Port {
	return Port{Buf: g.abuf, Off: g.aRes, Stride: g.cfg.Wide(), Width: g.cfg.Wide(), Rows: g.rows}
}

// ResRowPort is one token's row of that residual, which is what the final
// mixer reads: `inp_out_ids` keeps the last token and drops the rest.
func (g *HCGPU) ResRowPort(t int) Port {
	wide := g.cfg.Wide()
	return Port{Buf: g.abuf, Off: g.aRes + uint32(t*wide), Stride: wide, Width: wide, Rows: 1}
}

// Resize sets the length of the run without writing the residual, for a
// caller that is about to fill it with a Move rather than an Upload.
func (g *HCGPU) Resize(nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	g.rows = nTok
	if g.autoPlan {
		g.down, g.up = PlanForBank(nTok, g.dbank)
	}
	return nil
}
