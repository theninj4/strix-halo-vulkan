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
//	records  one per (n-tile, super-block, row): `d` and `dmin` as halves
//	         and then the 6-bit (scale, min) pair of every group, which is
//	         **sixteen bytes at eight groups** — ggml's own
//	         `get_scale_min_k4` record — and **twenty at ten** (see below).
//	         Indexed ((nt*nsb + sb)*16 + n%16), so the sixteen records a
//	         tile column needs are one contiguous run — the same k-major
//	         arrangement D8 chose for L8a's scale plane, for the same
//	         reason.
//
// 4 bits of level, 12 bits a group of 32 and 32 bits a super-block of 256 is
// **4.500 bits a weight**, which is the number every L8c-3 rung is quoted at.
//
// # The 320-wide family, and the one place this is not ggml
//
// A super-block is ggml's eight groups wherever `k % 256 == 0`, which is
// every dense matrix in this model but one family. `hc_{attn,ffn}_up` and
// `output_hc_up` read the low-rank space and are 320 wide, so their
// super-block is the **whole row** — ten groups — exactly as the simulation
// L8c-3 measured them with (`asymSubBlocks`). What does not carry over is the
// record: `get_scale_min_k4` is four low pairs and four high ones and it is
// not a length, so a ten-group record uses `packScaleMin12` instead and is
// twenty bytes rather than sixteen. The levels, the scales and the mins are
// the simulation's to the bit; only where the bits sit differs.

import (
	"fmt"
	"sync"
)

// q4kSuper is the groups in a super-block: ggml's QK_K/32.
const q4kSuper = 8

// q4kRecord is one super-block's parameters for one row: d, dmin and the
// twelve packed 6-bit pairs.
const q4kRecord = 16

// q4kShape is a row's division into super-blocks and the record size that
// division implies: eight groups into sixteen bytes, or ten into twenty.
func q4kShape(k int) (sub, nsb, rec int, err error) {
	if sub, err = asymSubBlocks(k, 32); err != nil {
		return 0, 0, 0, err
	}
	if err = q4kPackFits(sub); err != nil {
		return 0, 0, 0, err
	}
	return sub, k / (sub * 32), q4kRecordBytes(sub), nil
}

// q4kBytes is the staged size of an [n, k] matrix: the nibble tiles, then the
// record plane. As with L8a's, the kernel derives the plane's offset from
// bOff + n*k/2 rather than from a push field — 64 uints is this device's
// whole push range and the block has been full since L5b.
func q4kBytes(n, k int) int {
	_, nsb, rec, err := q4kShape(k)
	if err != nil {
		return 0
	}
	return n*k/2 + (n/coopMatTile)*nsb*coopMatTile*rec
}

// q4kRecPlane is the record plane's size for an [n, k] matrix, which the two
// planes' callers allocate separately because they write them separately.
func q4kRecPlane(n, k int) int {
	_, nsb, rec, err := q4kShape(k)
	if err != nil {
		return 0
	}
	return (n / coopMatTile) * nsb * coopMatTile * rec
}

// q4kFits reports whether a matrix can be staged in this bank at all.
func q4kFits(n, k int) error {
	if n%coopMatTile != 0 {
		return fmt.Errorf("llm: a q4_k bank wants whole %d-row tiles, not %d rows", coopMatTile, n)
	}
	_, _, _, err := q4kShape(k)
	return err
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
	sub, nsb, recBytes, err := q4kShape(k)
	if err != nil {
		return err
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
	if len(dstQ) != dstN*k/2 {
		return fmt.Errorf("llm: q4_k tile plane is %d bytes, want %d", len(dstQ), dstN*k/2)
	}
	if len(dstR) != (dstN/tile)*nsb*tile*recBytes {
		return fmt.Errorf("llm: q4_k record plane is %d bytes, want %d",
			len(dstR), (dstN/tile)*nsb*tile*recBytes)
	}
	if n > dstN {
		return fmt.Errorf("llm: %d source rows into a %d-row plane", n, dstN)
	}
	if len(src) < n*k {
		return fmt.Errorf("llm: %d source values for %d rows of %d", len(src), n, k)
	}
	var perr error
	var perrOnce sync.Once
	parallelFor((n+packChunk-1)/packChunk, func(ch int) {
		e := newAsymEnc(q, sub)
		for i := ch * packChunk; i < minInt((ch+1)*packChunk, n); i++ {
			r := row(i)
			// The tiles of one n-tile are contiguous across kt, and a tile is
			// 128 bytes, so this row's nibbles start at the n-tile's base and
			// are indexed by (kt, n%16, k%16) from there.
			base := (r / tile) * kt * (tile * tile / 2)
			lane := (r % tile) * (tile / 2)
			rbase := ((r/tile)*nsb*tile + r%tile) * recBytes
			x := src[i*k : (i+1)*k]
			for s := 0; s < nsb; s++ {
				blk := x[s*sub*32 : (s+1)*sub*32]
				var cols []float32
				if qw != nil {
					cols = qw[s*sub*32 : (s+1)*sub*32]
				}
				e.Encode(blk, cols)
				if err := packQ4KRecord(dstR[rbase+s*tile*recBytes:], e); err != nil {
					perrOnce.Do(func() { perr = err })
					return
				}
				for j := 0; j < sub*32; j += 2 {
					c := s*sub*32 + j
					dstQ[base+(c/tile)*(tile*tile/2)+lane+(c%tile)/2] =
						e.Q[j] | e.Q[j+1]<<4
				}
			}
		}
	})
	return perr
}

// q4kDequant reads one element back out of a staged bank exactly as the
// kernels do — the record's pair, `get_scale_min_k4`, and d*sc*l - dmin*m in
// f32. It is what the tests compare against the simulation.
func q4kDequant(bankQ, bankR []byte, n, k, dstRow, col int) float32 {
	const tile = coopMatTile
	kt := k / tile
	sub, nsb, recBytes, err := q4kShape(k)
	if err != nil {
		panic(err)
	}
	base := (dstRow/tile)*kt*(tile*tile/2) + (dstRow%tile)*(tile/2)
	b := bankQ[base+(col/tile)*(tile*tile/2)+(col%tile)/2]
	l := int(b & 0xF)
	if col%2 == 1 {
		l = int(b >> 4)
	}
	sb := col / (sub * 32)
	rec := bankR[((dstRow/tile)*nsb*tile+sb*tile+dstRow%tile)*recBytes:]
	d, dmin, sc, mn := unpackQ4KRecord(rec, sub, (col%(sub*32))/32)
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
