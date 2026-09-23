package llm

// The dense bank, in the width the checkpoint already ships: LLM.md L8.
//
// Every GEMM in this vertical but the MoE's reads a B operand the host has
// dequantised to halves and packed into §2.8 fragment tiles, and L7c-5 priced
// what that costs. A decode step reads 8.07 GB of dense weight where the
// checkpoint holds 3.67 — the dense half is 83% of a token's bytes and 76% of
// the checkpoint's own budget — so the *first* 1.66x of phase 2 needs no
// re-quantisation at all, only not expanding what is already there.
//
// So a dense weight is staged as int8 in the same fragment tiling, with one
// fp16 scale per 32 elements of a row, and the kernel unpacks its slab into
// LDS per K-step (shaders/llm_gemm.comp, -DQ8B). 8.5 bits a weight against
// 16, and for a Q8_0 tensor it is **the same numbers**: ggml picks
// d = amax/127, so the largest |q| in a block is 127, and re-deriving (d, q)
// from the dequantised floats returns exactly the pair the checkpoint stored.
// float(q) * float(d) is exact in f32 and rounding that to fp16 is what
// tileB wrote, so the halves the kernel multiplies are bit-identical.
// TestBankQ8IsTheHalves is the check, and it is an equality rather than a
// tolerance.
//
// Three families in this model are not Q8_0 and would be a real
// re-quantisation: the F32 routers (left on the fp16 arm — they are the MoE's
// and their ties decide which experts run), the two BF16 indexer projections
// and the three small F32 gate matrices fused into other matrices' rows. Each
// one is named where it is staged.

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// DenseBank is which width a dense weight is staged in — the one axis phase 2
// is about, and now the only one with three values.
//
//	BankFP16  the halves every GEMM here read before L8: 16 bits a weight,
//	          and the bank a simulation still runs on (sim.go).
//	BankQ8    the checkpoint's own int8 with an fp16 scale per 32 (L8a,
//	          L8b): 8.5 bits, and **bit-identical arithmetic**.
//	BankQ4K   ggml's asymmetric K-quant, 4.500 bits (L8c-4): the first bank
//	          here that changes what the model computes, at +4.24% of
//	          perplexity for 4.264 GB a token against 6.334.
//	BankQ5K   the same format with P3a's `qh` plane, 5.500 bits: the same
//	          tiles and the same record plus one bit a weight, which P3
//	          priced at 14.1 pp/GB on `full_attn` against 4.6 for the best
//	          arm the four-bit kernel can stage.
type DenseBank int

const (
	BankFP16 DenseBank = iota
	BankQ8
	BankQ4K
	BankQ5K
)

// qkBank reports whether a bank is one of the two K-quant widths, and which
// level width it stages at. Everything but the bytes and the build key is
// shared between them, so this is the predicate most callers want.
func qkBank(b DenseBank) (bits int, ok bool) {
	switch b {
	case BankQ4K:
		return 4, true
	case BankQ5K:
		return 5, true
	}
	return 0, false
}

// bankOfSim is which bank a plan's format stages on, for the constructors
// that take a `QuantSim` from `DenseBankPlan` and have to say where it lands.
func bankOfSim(q QuantSim) (DenseBank, error) {
	bits, err := qkBits(q)
	if err != nil {
		return BankFP16, err
	}
	if bits == 5 {
		return BankQ5K, nil
	}
	return BankQ4K, nil
}

// String names a bank the way every CSV and header line in this vertical
// does.
func (b DenseBank) String() string {
	switch b {
	case BankQ8:
		return "q8"
	case BankQ4K:
		return "q4_k"
	case BankQ5K:
		return "q5_k"
	}
	return "fp16"
}

// BankFor is bankOf for a caller outside the package — the benchmarks, which
// take the same two-valued choice from `LLM_DENSE_FP16` that a graph does.
func BankFor(q8 bool) DenseBank { return bankOf(q8) }

// BankForSim is bankOfSim for a caller outside the package: which of the two
// K-quant banks a plan's format stages on. The benchmarks need it for the
// same reason the graph does — a bench that staged a different width from the
// one the graph stages would be measuring a kernel nothing runs.
func BankForSim(q QuantSim) (DenseBank, error) { return bankOfSim(q) }

// KQuant reports whether a bank is one of the two K-quant widths, for the
// callers outside the package that have to say "this family is on the narrow
// bank" without saying which of the two.
func (b DenseBank) KQuant() bool { _, ok := qkBank(b); return ok }

// bankOf is the two-valued spelling every constructor took before L8c-4.
func bankOf(q8 bool) DenseBank {
	if q8 {
		return BankQ8
	}
	return BankFP16
}

// q8Group is how many elements of a row share one fp16 scale. It is ggml's
// Q8_0 block, which is what makes the round trip exact — and it is two whole
// sixteen-wide k-tiles, which is what makes the scale constant across a
// cooperative-matrix tile and the unpack one multiply per four bytes.
const q8Group = 32

// q8Bytes is the staged size of an [n, k] matrix: the int8 tiles, then the
// scale plane. The kernel derives the plane's offset from bOff + n*k rather
// than from a push field, because 64 uints is this device's whole push range
// and the block has been full since L5b.
func q8Bytes(n, k int) int { return n*k + (n*k/q8Group)*2 }

// tileBQ8 writes one [n, k] row-major matrix into the §2.8 fragment tiling as
// int8 plus scales, with tileB's row remapping.
//
// The tiles are exactly tileB's layout with a byte where it puts a half: tile
// (nt, kt) is 256 contiguous bytes holding element (k, n) at (n%16)*16 + k%16,
// tiles ordered kt-fastest. The plane is k-major inside an n-tile — scale
// (nt, group g, n%16) at (nt*kgroups + g)*16 + n%16 — so the sixteen scales a
// tile needs are one contiguous 32-byte run (D8).
func tileBQ8(dstQ []byte, dstS []uint16, src []float32, n, k int, row func(int) int) {
	const tile = coopMatTile
	kt := k / tile
	kg := k / q8Group
	parallelFor((n+packChunk-1)/packChunk, func(ch int) {
		for i := ch * packChunk; i < minInt((ch+1)*packChunk, n); i++ {
			r := row(i)
			base := (r / tile) * kt * tile * tile
			lane := (r % tile) * tile
			sbase := (r/tile)*kg*tile + r%tile
			x := src[i*k : (i+1)*k]
			for g := 0; g < kg; g++ {
				blk := x[g*q8Group : (g+1)*q8Group]
				amax := float32(0)
				for _, v := range blk {
					if v < 0 {
						v = -v
					}
					if v > amax {
						amax = v
					}
				}
				d := amax / 127
				id := float32(0)
				if d != 0 {
					id = 1 / d
				}
				dstS[sbase+g*tile] = safetensors.F32ToF16(d)
				for j, v := range blk {
					c := g*q8Group + j
					dstQ[base+(c/tile)*tile*tile+lane+c%tile] = byte(roundI8(v * id))
				}
			}
		}
	})
}

// roundI8 is ggml's roundf, clamped. Half away from zero, which is what
// quantize_row_q8_0 does and what makes the round trip an identity.
func roundI8(v float32) int8 {
	if v >= 0 {
		v += 0.5
		if v > 127 {
			return 127
		}
		return int8(int32(v))
	}
	v -= 0.5
	if v < -127 {
		return -127
	}
	return int8(int32(v))
}

// q8Align rounds a matrix's staged size up so that the next one starts on a
// whole word: the kernel reads the tiles through a `uint` view, and a base
// that is not four-aligned would read across the boundary.
func q8Align(n int) int { return (n + 15) &^ 15 }

// BankLevelBytes and BankPlaneBytes split a staged [n, k] matrix into its two
// planes, because the benchmarks price them separately: a fused matrix stages
// the *whole* N in both planes and reads only part of the first, so the
// arithmetic a dispatch's GB/s comes out of is not `q8Bytes` or `q4kBytes`.
//
//	fp16    2 bytes a weight, no second plane
//	q8      1 byte a weight, an fp16 scale per 32
//	q4_k    a nibble a weight, a sixteen- or twenty-byte record per
//	        (n-tile, super-block, row)
//	q5_k    the same, plus P3a's one-bit plane — which counts with the
//	        record rather than with the levels, because it is addressed off
//	        the same derived base and not off bOff
func BankLevelBytes(b DenseBank, n, k int) int {
	switch b {
	case BankQ8:
		return n * k
	case BankQ4K, BankQ5K:
		return n * k / 2
	}
	return n * k * 2
}

// BankPlaneBytes is everything but the levels, and zero on the fp16 bank.
func BankPlaneBytes(b DenseBank, n, k int) int {
	switch b {
	case BankQ8:
		return n * k / q8Group * 2
	case BankQ4K, BankQ5K:
		bits, _ := qkBank(b)
		return qkBytes(bits, n, k) - n*k/2
	}
	return 0
}

// bankPipe names a bank's build of a GEMM rung inside a block that holds
// more than one. A block on a quantised bank still needs the fp16 arm for
// whatever rows of a fused matrix the checkpoint does not ship quantised, so
// the builds cannot share a key.
func bankPipe(b DenseBank, k GEMMKernel) string {
	switch b {
	case BankQ8:
		return "q8_" + string(k)
	case BankQ4K:
		return "q4_" + string(k)
	case BankQ5K:
		return "q5_" + string(k)
	}
	return string(k)
}

// GEMVKernel names one build of shaders/llm_gemv.comp, by how many ways it
// splits K — the vertical's plain projection **at one token** (LLM.md L8d).
//
// `GEMVOff` is llm_gemm.comp MODE 2, which is the right kernel at every other
// length and is half the bus at this one: a sixteen-row fragment holding one
// row, an LDS slab per K-step behind a barrier, and a grid of `gemmN/64`
// workgroups. The rest are the split-K GEMV, in two dispatches — or one at
// `GEMVK1`, where there are no partials to sum.
//
// **The rung is per projection**, because D12's 4 KB rotation is a fact about
// a slab's stride and a slab is `(gemmK/16/KSLABS) * 256` bytes on the int8
// bank: a K of 2560 and a K of 6144 do not want the same split.
type GEMVKernel string

const (
	GEMVOff GEMVKernel = "gemm"
	GEMVK1  GEMVKernel = "k1"
	GEMVK2  GEMVKernel = "k2"
	GEMVK4  GEMVKernel = "k4"
	GEMVK8  GEMVKernel = "k8"
	GEMVK16 GEMVKernel = "k16"
	GEMVK32 GEMVKernel = "k32"
	GEMVK20 GEMVKernel = "k20"
	GEMVK40 GEMVKernel = "k40"
)

// GEMVKernels lists the split-K rungs, narrowest split first.
func GEMVKernels() []GEMVKernel {
	return []GEMVKernel{GEMVK1, GEMVK2, GEMVK4, GEMVK8, GEMVK16, GEMVK20, GEMVK32, GEMVK40}
}

// gemvSlabs is a rung's split, and zero for the GEMM.
func gemvSlabs(k GEMVKernel) int {
	switch k {
	case GEMVK1:
		return 1
	case GEMVK2:
		return 2
	case GEMVK4:
		return 4
	case GEMVK8:
		return 8
	case GEMVK16:
		return 16
	case GEMVK32:
		return 32
	case GEMVK20:
		return 20
	case GEMVK40:
		return 40
	}
	return 0
}

// gemvMaxSlabs sizes a block's partial-sum arena: f32 [KSLABS][gemmN].
const gemvMaxSlabs = 40

// GEMVFits reports whether a rung can run a given reduction extent. A wave
// covers four k-tiles a step, so a slab has to be a whole number of steps —
// which is what keeps a ladder from naming a rung that cannot exist: at
// K = 2560 the k-tiles are 160 and only 1, 2, 4, 8 and 10 divide it into whole
// steps, where at K = 6144 they are 384 and every rung does.
func GEMVFits(k GEMVKernel, gemmK int) bool {
	s := gemvSlabs(k)
	if s == 0 {
		return false
	}
	kt := gemmK / 16
	return kt%s == 0 && (kt/s)%4 == 0
}

// GEMVMaxRows is how many rows of A a decode GEMV dispatch may carry: P5b's
// `MAXROWS`, and the bound every block's partial arena is sized by.
//
// **It is three because three sequences decode at once** (CONCURRENCY.md
// C5): a batched pass is a row a sequence through the same weights. It was
// two, for P5a's depth-one verification pass. Raising it is a `MAXROWS` in
// four shaders and this constant, and it costs an accumulator per row at
// every rung: C5a measured a two-row pass at 41.2 ms against 38.9 with
// MAXROWS 2, and one row unchanged. A new count lands with its rungs in
// TestGraphIsAChunkSplit (the decode-schedule subtests at 2 and 3).
const GEMVMaxRows = 3

// gemvBankPipe names the pipeline for a rung: the partials over one bank or
// another, and the sum. `rows` is P5b's specialization — one module, one
// pipeline per row count, because the constant folds at pipeline build.
func gemvBankPipe(k GEMVKernel, b DenseBank) string { return gemvBankPipeRows(k, b, 1) }

func gemvBankPipeRows(k GEMVKernel, b DenseBank, rows int) string {
	var base string
	switch b {
	case BankQ8:
		base = fmt.Sprintf("gemv_q8_k%d", gemvSlabs(k))
	case BankQ4K:
		base = fmt.Sprintf("gemv_q4_k%d", gemvSlabs(k))
	case BankQ5K:
		base = fmt.Sprintf("gemv_q5_k%d", gemvSlabs(k))
	default:
		base = fmt.Sprintf("gemv_k%d", gemvSlabs(k))
	}
	return gemvRowName(base, rows)
}

func gemvSumPipe(k GEMVKernel) string { return gemvSumPipeRows(k, 1) }

func gemvSumPipeRows(k GEMVKernel, rows int) string {
	return gemvRowName(fmt.Sprintf("gemv_sum_k%d", gemvSlabs(k)), rows)
}

// gemvRowName suffixes a pipeline name with its row count. One row keeps the
// bare name, so every CSV and every test written before P5b still names the
// same pipeline.
func gemvRowName(base string, rows int) string {
	if rows <= 1 {
		return base
	}
	return fmt.Sprintf("%s_r%d", base, rows)
}

// gemvSpec is the specialization a row count needs: constant 0 is `ROWS` in
// llm_gemv.comp, llm_hc_gemv.comp and llm_moe_router.comp.
func gemvSpec(rows int) []vk.SpecConstant {
	if rows <= 1 {
		return nil
	}
	return []vk.SpecConstant{{ID: 0, Value: uint32(rows)}}
}

// gemvSPIRV is every rung of both banks plus the reduce, which each block that
// has a decode path builds.
var gemvSPIRV = map[string][]byte{
	"gemv_k20":     shaders.LLMGEMVK20,
	"gemv_q8_k20":  shaders.LLMGEMVQ8K20,
	"gemv_sum_k20": shaders.LLMGEMVSumK20,
	"gemv_k40":     shaders.LLMGEMVK40,
	"gemv_q8_k40":  shaders.LLMGEMVQ8K40,
	"gemv_sum_k40": shaders.LLMGEMVSumK40,
	"gemv_k1":      shaders.LLMGEMVK1,
	"gemv_k2":      shaders.LLMGEMVK2,
	"gemv_k4":      shaders.LLMGEMVK4,
	"gemv_k8":      shaders.LLMGEMVK8,
	"gemv_k16":     shaders.LLMGEMVK16,
	"gemv_k32":     shaders.LLMGEMVK32,
	"gemv_q8_k1":   shaders.LLMGEMVQ8K1,
	"gemv_q8_k2":   shaders.LLMGEMVQ8K2,
	"gemv_q8_k4":   shaders.LLMGEMVQ8K4,
	"gemv_q8_k8":   shaders.LLMGEMVQ8K8,
	"gemv_q8_k16":  shaders.LLMGEMVQ8K16,
	"gemv_q8_k32":  shaders.LLMGEMVQ8K32,
	"gemv_sum_k2":  shaders.LLMGEMVSumK2,
	"gemv_sum_k4":  shaders.LLMGEMVSumK4,
	"gemv_sum_k8":  shaders.LLMGEMVSumK8,
	"gemv_sum_k16": shaders.LLMGEMVSumK16,
	"gemv_sum_k32": shaders.LLMGEMVSumK32,
	"gemv_q4_k1":   shaders.LLMGEMVQ4K1,
	"gemv_q4_k2":   shaders.LLMGEMVQ4K2,
	"gemv_q4_k4":   shaders.LLMGEMVQ4K4,
	"gemv_q4_k8":   shaders.LLMGEMVQ4K8,
	"gemv_q4_k16":  shaders.LLMGEMVQ4K16,
	"gemv_q4_k20":  shaders.LLMGEMVQ4K20,
	"gemv_q4_k32":  shaders.LLMGEMVQ4K32,
	"gemv_q4_k40":  shaders.LLMGEMVQ4K40,
	"gemv_q5_k1":   shaders.LLMGEMVQ5K1,
	"gemv_q5_k2":   shaders.LLMGEMVQ5K2,
	"gemv_q5_k4":   shaders.LLMGEMVQ5K4,
	"gemv_q5_k8":   shaders.LLMGEMVQ5K8,
	"gemv_q5_k16":  shaders.LLMGEMVQ5K16,
	"gemv_q5_k20":  shaders.LLMGEMVQ5K20,
	"gemv_q5_k32":  shaders.LLMGEMVQ5K32,
	"gemv_q5_k40":  shaders.LLMGEMVQ5K40,
}
