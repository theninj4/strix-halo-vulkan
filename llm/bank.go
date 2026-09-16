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
	"strix-halo-vulkan/safetensors"
)

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

// q8Pipe names the Q8 build of a GEMM rung inside a block that holds both.
// A block on L8's bank still needs the fp16 arm for whatever rows of a fused
// matrix the checkpoint does not ship as Q8_0, so the two cannot share a key.
func q8Pipe(k GEMMKernel) string { return "q8_" + string(k) }
