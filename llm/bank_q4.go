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
// # The fifth bit: P3a's `qh` plane
//
// P3 measured what a fifth bit is worth and it is the best remaining trade on
// the board by three times — `full_attn` at `q5_k` recovers 1.1 points of
// perplexity for 0.077 GB a token, **14.1 pp/GB**, where the best arm the
// four-bit kernel can stage is 4.6. ggml's own `Q5_K` is `Q4_K` plus a
// `qh` bit-plane and *the same record*, so the format here is the same three
// numbers plus one plane:
//
//	high     tile (nt, kt) is 32 contiguous bytes, one per 128-byte nibble
//	         tile and in the same order, and **byte i of it is the top bit of
//	         each of the eight levels in word i of the nibble tile** — bit e
//	         for the level at k offset e. So the high byte's index inside a
//	         tile *is* the nibble word's index inside that tile, which is
//	         what makes the unpack a second load at the same address
//	         arithmetic rather than a second addressing scheme: the GEMM's
//	         `u` and the GEMV's `col*2 + u` index both planes unchanged.
//
// It sits **between** the tiles and the records — bOff, then n*k/2, then
// n*k/8 — so the record plane's base stays derivable from `gemmN * gemmK`
// and no push field is needed for either (64 uints is this device's whole
// push range and the block has been full since L5b).
//
// 5 bits of level, 12 bits a group of 32 and 32 bits a super-block is
// **5.500 bits a weight**, ggml's own Q5_K number.
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

	"strix-halo-vulkan/vk"
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

// qkBits is the level width a plan's format stages at, and the one check
// that says which of the two banks a matrix lands on: four bits is L8c-4's
// nibble tiles alone, five is those tiles plus P3a's `qh` plane.
func qkBits(q QuantSim) (int, error) {
	if !q.Asym || q.Group != 32 {
		return 0, fmt.Errorf("llm: %s is not an asymmetric K-quant on groups of 32", q)
	}
	if q.Bits != 4 && q.Bits != 5 {
		return 0, fmt.Errorf("llm: %s is %d bits and this bank is nibbles or nibbles plus a plane", q, q.Bits)
	}
	return q.Bits, nil
}

// qkBytes is the staged size of an [n, k] matrix: the nibble tiles, the
// fifth-bit plane where there is one, then the record plane. As with L8a's,
// the kernel derives both following planes from bOff + n*k/2 rather than from
// a push field — 64 uints is this device's whole push range and the block has
// been full since L5b.
func qkBytes(bits, n, k int) int {
	return n*k/2 + qkHighPlane(bits, n, k) + qkRecPlane(n, k)
}

// qkHighPlane is P3a's plane alone: one bit a weight, and nothing at four
// bits. It is one 32-byte tile per 128-byte nibble tile, in the same order,
// so it is exactly a quarter of the level plane.
func qkHighPlane(bits, n, k int) int {
	if bits != 5 {
		return 0
	}
	return n * k / 8
}

// qkRecPlane is the record plane's size for an [n, k] matrix, which the
// planes' callers allocate separately because they write them separately.
// ggml's Q5_K carries the same record as its Q4_K, so this does not move with
// the width.
func qkRecPlane(n, k int) int {
	_, nsb, rec, err := q4kShape(k)
	if err != nil {
		return 0
	}
	return (n / coopMatTile) * nsb * coopMatTile * rec
}

// qkFits reports whether a matrix can be staged in this bank at all. It is
// the same answer at both widths: the tiling is the four-bit one and the
// fifth bit rides on it.
func qkFits(n, k int) error {
	if n%coopMatTile != 0 {
		return fmt.Errorf("llm: a K-quant bank wants whole %d-row tiles, not %d rows", coopMatTile, n)
	}
	_, _, _, err := q4kShape(k)
	return err
}

// tileBQK writes one [n, k] row-major matrix into the bank, quantising it
// through the same encoder `sim.go` measured the format with.
//
// `src` is n rows of k floats, `row` is tileB's row remapping, `q` the format
// (`q4_k/32` or `q5_k/32`) and `qw` the importance of each of the k input
// columns — the imatrix row for this tensor, or nil for round-to-nearest.
//
// dstQ, dstH and dstR are the destination's planes, and the first one's
// length is what says how many rows the destination has. `dstH` is nil at
// four bits and `qkHighPlane` bytes at five. They are the *whole* matrix's or
// a slab's, as long as the slab is whole sixteen-row tiles of the
// destination, which is what lets the lm head stage 4096 rows at a time
// without a 2.54 GB copy of itself ever existing — and they may be wider than
// `n`, which is what lets four of llama.cpp's matrices be staged into one
// fused plane.
func tileBQK(dstQ, dstH, dstR []byte, src []float32, n, k int, row func(int) int, q QuantSim, qw []float32) error {
	sub, nsb, recBytes, err := q4kShape(k)
	if err != nil {
		return err
	}
	bits, err := qkBits(q)
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
	if err := qkFits(dstN, k); err != nil {
		return err
	}
	if qw != nil && len(qw) != k {
		return fmt.Errorf("llm: %d importance columns for a row of %d", len(qw), k)
	}
	const tile = coopMatTile
	kt := k / tile
	if len(dstQ) != dstN*k/2 {
		return fmt.Errorf("llm: K-quant tile plane is %d bytes, want %d", len(dstQ), dstN*k/2)
	}
	if want := qkHighPlane(bits, dstN, k); len(dstH) != want {
		return fmt.Errorf("llm: %d-bit high plane is %d bytes, want %d", bits, len(dstH), want)
	}
	if len(dstR) != (dstN/tile)*nsb*tile*recBytes {
		return fmt.Errorf("llm: K-quant record plane is %d bytes, want %d",
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
			// are indexed by (kt, n%16, k%16) from there. The high plane is
			// the same arithmetic at a quarter of the stride — a tile is 32
			// bytes there, and the byte index inside it is the nibble *word*
			// index inside the 128.
			base := (r / tile) * kt * (tile * tile / 2)
			lane := (r % tile) * (tile / 2)
			hbase := (r / tile) * kt * (tile * tile / 8)
			hlane := (r % tile) * (tile / 8)
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
						e.Q[j]&0xF | (e.Q[j+1]&0xF)<<4
					if bits != 5 {
						continue
					}
					// Two levels a byte of nibbles is two bits a byte of
					// plane, at bit (k%8) — and eight k of one column share
					// one byte, so the first pair of a byte writes it and the
					// other three or-in. A super-block starts on a multiple
					// of 32, so `c%8 == 0` is exactly a byte's first pair.
					h := (e.Q[j]>>4)&1 | ((e.Q[j+1]>>4)&1)<<1
					hi := hbase + (c/tile)*(tile*tile/8) + hlane + (c%tile)/8
					if c%8 == 0 {
						dstH[hi] = h << (c % 8)
					} else {
						dstH[hi] |= h << (c % 8)
					}
				}
			}
		}
	})
	return perr
}

// qkDequant reads one element back out of a staged bank exactly as the
// kernels do — the nibble, P3a's top bit where there is one, the record's
// pair, `get_scale_min_k4`, and d*sc*l - dmin*m in f32. It is what the tests
// compare against the simulation.
func qkDequant(bankQ, bankH, bankR []byte, bits, n, k, dstRow, col int) float32 {
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
	if bits == 5 {
		hbase := (dstRow/tile)*kt*(tile*tile/8) + (dstRow%tile)*(tile/8)
		hb := bankH[hbase+(col/tile)*(tile*tile/8)+(col%tile)/8]
		l |= int(hb>>uint(col%8)&1) << 4
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

// qkStage is one matrix's staged planes, in the order the bank writes them
// and the order every kernel derives them from `bOff`: nibble tiles, P3a's
// fifth-bit plane — empty at four bits — then the records.
//
// It exists so that "the planes are contiguous, in this order" is stated once
// instead of at each of the six blocks that stage a dense matrix. Before P3a
// there were two planes and each site wrote them itself; with three, one of
// which is sometimes absent, the ordering is the sort of fact that goes wrong
// in one place and is found by a perplexity run rather than by a test.
type qkStage struct {
	Bits           int
	Lvl, High, Rec []byte
}

// newQKStage allocates the planes of an [n, k] matrix at one of the two
// widths. n is the *destination plane's* rows, which is wider than the source
// wherever several of llama.cpp's matrices are fused into one.
func newQKStage(bits, n, k int) qkStage {
	return qkStage{
		Bits: bits,
		Lvl:  make([]byte, n*k/2),
		High: make([]byte, qkHighPlane(bits, n, k)),
		Rec:  make([]byte, qkRecPlane(n, k)),
	}
}

// Fill quantises one source matrix into the planes. It may be called more
// than once on the same stage with different `row` remappings, which is how a
// fused plane is written from the several matrices that share it.
func (s qkStage) Fill(src []float32, n, k int, row func(int) int, q QuantSim, qw []float32) error {
	return tileBQK(s.Lvl, s.High, s.Rec, src, n, k, row, q, qw)
}

// Len is the staged size, which has to equal `qkBytes` for the same shape.
func (s qkStage) Len() int { return len(s.Lvl) + len(s.High) + len(s.Rec) }

// WriteTo copies the planes into the bank at `off`, in that order.
func (s qkStage) WriteTo(b *vk.Buffer, off int) {
	b.WriteBytesAt(off, s.Lvl)
	if len(s.High) > 0 {
		b.WriteBytesAt(off+len(s.Lvl), s.High)
	}
	b.WriteBytesAt(off+len(s.Lvl)+len(s.High), s.Rec)
}

// Rows is the stage restricted to the first n rows of an [n, k] fill, which
// is how the lm head stages one slab at a time of a matrix that is 2.54 GB as
// floats and never exists whole.
func (s qkStage) Rows(n, k int) qkStage {
	return qkStage{
		Bits: s.Bits,
		Lvl:  s.Lvl[:n*k/2],
		High: s.High[:qkHighPlane(s.Bits, n, k)],
		Rec:  s.Rec[:qkRecPlane(n, k)],
	}
}
