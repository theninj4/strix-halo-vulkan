package llm

// The dense bank at 4.5 bits: LLM.md L8c-4.
//
// L8a and L8b staged every dense weight in the width the checkpoint already
// ships — int8 with an fp16 scale per 32 — which is bit-identical arithmetic
// and took a decode token from 9.67 GB to 6.05. That is where the
// *checkpoint's* width runs out. L8c then measured what a narrower one costs,
// and the answer turned twice: symmetric Q4 at 4.5 bits is +18.5% of
// perplexity, and ggml's **asymmetric** K-quant at the same 4.500 bits is
// +7.04% uncalibrated and **+4.24% with unsloth's published imatrix**
// (L8c-3). So this is the bank that measurement recommends: 4.264 GB a token
// against 6.334 and a 56.7 tok/s ceiling against 38.2.
//
// # The layout
//
// Everything about it is §2.8's fragment tiling with a *nibble* where L8a
// puts a byte, plus a second plane that is ggml's own super-block record.
//
//	tiles    tile (nt, kt) is 128 contiguous bytes, kt fastest inside an
//	         n-tile, holding element (k, n) of the matrix at nibble
//	         (n%16)*16 + k%16 — byte (n%16)*8 + (k%16)/2, low nibble for
//	         even k. So a `uint` is eight consecutive k of one output
//	         column, where L8a's is four.
//
//	records  one per (n-tile, super-block, row): sixteen bytes, `d` and
//	         `dmin` as halves and then the twelve bytes ggml's
//	         `get_scale_min_k4` reads eight 6-bit (scale, min) pairs out of.
//	         Indexed ((nt*nsb + sb)*16 + n%16), so the sixteen records a
//	         tile column needs are 256 contiguous bytes — the same k-major
//	         arrangement D8 chose for L8a's scale plane, for the same
//	         reason.
//
// 4 bits of level, 12 bits a group of 32 and 32 bits a super-block of 256 is
// **4.500 bits a weight**, which is the number every L8c-3 rung is quoted at.
//
// # What it does not cover yet
//
// A super-block is ggml's eight groups, so `k % 256 == 0` — every dense
// matrix in this model but one family. `hc_{attn,ffn}_up` and `output_hc_up`
// read the low-rank space and are 320 wide, which the *simulation* handles
// with a ten-group super-block (sim.go's superBlocks) and which
// `get_scale_min_k4`'s packing does not: the scheme is four low pairs and
// four high ones and it is not a length. That family is L8b's, it is the one
// the imatrix result turns on, and it gets its own packing when its block
// gets this bank.

import (
	"encoding/binary"
	"fmt"
)

// q4kSuper is the groups in a super-block: ggml's QK_K/32.
const q4kSuper = 8

// q4kRecord is one super-block's parameters for one row: d, dmin and the
// twelve packed 6-bit pairs.
const q4kRecord = 16

// q4kBytes is the staged size of an [n, k] matrix: the nibble tiles, then the
// record plane. As with L8a's, the kernel derives the plane's offset from
// bOff + n*k/2 rather than from a push field — 64 uints is this device's
// whole push range and the block has been full since L5b.
func q4kBytes(n, k int) int {
	return n*k/2 + (n/coopMatTile)*(k/(q4kSuper*32))*coopMatTile*q4kRecord
}

// q4kFits reports whether a matrix can be staged in this bank at all.
func q4kFits(n, k int) error {
	if n%coopMatTile != 0 {
		return fmt.Errorf("llm: a q4_k bank wants whole %d-row tiles, not %d rows", coopMatTile, n)
	}
	if k%(q4kSuper*32) != 0 {
		return fmt.Errorf("llm: a q4_k bank wants k a multiple of %d, not %d — "+
			"the 320-wide hyper-connection family needs its own packing", q4kSuper*32, k)
	}
	return nil
}

// tileBQ4K writes one [n, k] row-major matrix into the bank, quantising it
// through the same encoder `sim.go` measured the format with.
//
// `src` is n rows of k floats, `row` is tileB's row remapping, `q` the format
// (`q4_k/32`) and `qw` the importance of each of the k input columns
// — the imatrix row for this tensor, or nil for round-to-nearest.
//
// dstQ and dstR are the destination's two planes, and their length is what
// says how many rows the destination has. They are the *whole* matrix's or a
// slab's, as long as the slab is whole sixteen-row tiles of the destination,
// which is what lets the lm head stage 4096 rows at a time without a 2.54 GB
// copy of itself ever existing — and they may be wider than `n`, which is
// what lets four of llama.cpp's matrices be staged into one fused plane.
func tileBQ4K(dstQ, dstR []byte, src []float32, n, k int, row func(int) int, q QuantSim, qw []float32) error {
	if k%(q4kSuper*32) != 0 {
		return fmt.Errorf("llm: a q4_k bank wants k a multiple of %d, not %d — "+
			"the 320-wide hyper-connection family needs its own packing", q4kSuper*32, k)
	}
	// **The destination's rows are the plane's, not the source's.** The lm
	// head stages one matrix into a plane of its own size, but the gated
	// DeltaNet's fused projection is four of llama.cpp's matrices written
	// into one — `row` sends each source row to its column of the fused
	// output — so the plane a call fills is wider than the rows it carries.
	// The buffer says how wide: dstQ is dstN*k/2 bytes and nothing else can
	// be, which is the same check one level up.
	dstN := len(dstQ) * 2 / k
	if err := q4kFits(dstN, k); err != nil {
		return err
	}
	if !q.Asym || q.Group != 32 {
		return fmt.Errorf("llm: %s is not an asymmetric K-quant on groups of 32", q)
	}
	// **Four bits, because a tile is nibbles.** L8c-3's mixed plan puts the
	// attention layer and the hyper-connection block at `q5_k` for +2.70%
	// against the uniform plan's +4.24%, and a fifth bit is a plane of its
	// own the way ggml's `qh` is — a third stream through the unpack, and a
	// format whose tiles are no longer one byte per two elements. The
	// uniform plan is what this bank is built for; the fifth bit is the
	// decision the mixed one costs.
	if q.Bits != 4 {
		return fmt.Errorf("llm: %s is %d bits and a tile of this bank is nibbles", q, q.Bits)
	}
	if qw != nil && len(qw) != k {
		return fmt.Errorf("llm: %d importance columns for a row of %d", len(qw), k)
	}
	const tile = coopMatTile
	kt := k / tile
	nsb := k / (q4kSuper * 32)
	if len(dstQ) != dstN*k/2 {
		return fmt.Errorf("llm: q4_k tile plane is %d bytes, want %d", len(dstQ), dstN*k/2)
	}
	if len(dstR) != (dstN/tile)*nsb*tile*q4kRecord {
		return fmt.Errorf("llm: q4_k record plane is %d bytes, want %d",
			len(dstR), (dstN/tile)*nsb*tile*q4kRecord)
	}
	if n > dstN {
		return fmt.Errorf("llm: %d source rows into a %d-row plane", n, dstN)
	}
	if len(src) < n*k {
		return fmt.Errorf("llm: %d source values for %d rows of %d", len(src), n, k)
	}
	if err := q4kPackFits(q4kSuper); err != nil {
		return err
	}
	parallelFor((n+packChunk-1)/packChunk, func(ch int) {
		e := newAsymEnc(q, q4kSuper)
		for i := ch * packChunk; i < minInt((ch+1)*packChunk, n); i++ {
			r := row(i)
			// The tiles of one n-tile are contiguous across kt, and a tile is
			// 128 bytes, so this row's nibbles start at the n-tile's base and
			// are indexed by (kt, n%16, k%16) from there.
			base := (r / tile) * kt * (tile * tile / 2)
			lane := (r % tile) * (tile / 2)
			rbase := ((r/tile)*nsb*tile + r%tile) * q4kRecord
			x := src[i*k : (i+1)*k]
			for s := 0; s < nsb; s++ {
				blk := x[s*q4kSuper*32 : (s+1)*q4kSuper*32]
				var cols []float32
				if qw != nil {
					cols = qw[s*q4kSuper*32 : (s+1)*q4kSuper*32]
				}
				e.Encode(blk, cols)
				rec := dstR[rbase+s*tile*q4kRecord:]
				binary.LittleEndian.PutUint16(rec[0:], e.D)
				binary.LittleEndian.PutUint16(rec[2:], e.DMin)
				packScaleMinK4(rec[4:16], e.LS, e.LM)
				for j := 0; j < q4kSuper*32; j += 2 {
					c := s*q4kSuper*32 + j
					dstQ[base+(c/tile)*(tile*tile/2)+lane+(c%tile)/2] =
						e.Q[j] | e.Q[j+1]<<4
				}
			}
		}
	})
	return nil
}

// q4kDequant reads one element back out of a staged bank exactly as the
// kernels do — the record's pair, `get_scale_min_k4`, and d*sc*l - dmin*m in
// f32. It is what the tests compare against the simulation.
func q4kDequant(bankQ, bankR []byte, n, k, dstRow, col int) float32 {
	const tile = coopMatTile
	kt := k / tile
	nsb := k / (q4kSuper * 32)
	base := (dstRow/tile)*kt*(tile*tile/2) + (dstRow%tile)*(tile/2)
	b := bankQ[base+(col/tile)*(tile*tile/2)+(col%tile)/2]
	l := int(b & 0xF)
	if col%2 == 1 {
		l = int(b >> 4)
	}
	sb := col / (q4kSuper * 32)
	rec := bankR[((dstRow/tile)*nsb*tile+sb*tile+dstRow%tile)*q4kRecord:]
	d := f16(binary.LittleEndian.Uint16(rec[0:]))
	dmin := f16(binary.LittleEndian.Uint16(rec[2:]))
	sc, mn := unpackScaleMinK4(rec[4:16], (col%(q4kSuper*32))/32)
	return d*float32(sc)*float32(l) - dmin*float32(mn)
}

// bankImatrix is one tensor's importance row and the format to encode it
// with: the plan's, or round-to-nearest where the published matrix has no
// entry for the tensor.
//
// The fallback is ggml's own — `quantize_row_q4_K_impl` returns
// `quantize_row_q4_K_ref` on a null `quant_weights`, not the *search* with
// unit weights, which is a third arm and one L8c-3 measured as worse than
// either — and it is what `sim.go`'s `ApplyTo` does. A bank that did anything
// else at an uncovered tensor would not be the format the simulation
// measured, which is the one property this whole stage rests on.
func bankImatrix(name string, q QuantSim) (QuantSim, []float32, error) {
	qw, err := imatrixCols(name, q.Mode)
	if err != nil {
		return q, nil, err
	}
	if qw == nil && (q.Mode == "imatrix" || q.Mode == "imatrix+gain") {
		q.Mode = "rtn"
	}
	return q, qw, nil
}
