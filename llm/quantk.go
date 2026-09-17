package llm

// ggml's K-quant super-block, as a thing that can be *stored*: LLM.md L8c-4.
//
// L8c-3 measured the asymmetric form through `sim.go`, which round-trips a
// weight to floats and lets the existing fp16 kernels multiply them. That
// answered the accuracy question and nothing else — a simulation stages
// halves, so it says nothing about bytes or about tok/s. The bank does, and
// the bank needs the *levels* the simulation threw away.
//
// So the per-super-block arithmetic lives here rather than inside
// `applyAsym`, and both callers go through it:
//
//	sim.go     encodes, then writes back d*sc*l - dmin*m as floats
//	bank_q4.go encodes, then packs (d, dmin, the 6-bit pairs, the nibbles)
//
// which makes "the bank is the format the simulation measured" true by
// construction rather than by a tolerance. TestBankQ4KIsTheSim is the check,
// and it is an equality.

import (
	"encoding/binary"
	"fmt"

	"strix-halo-vulkan/safetensors"
)

// asymEnc encodes one super-block at a time, reusing its scratch. One per
// goroutine — `parallelFor` over rows is what every caller does.
type asymEnc struct {
	sim   QuantSim
	sub   int // groups in a super-block
	group int // elements in a group
	nmax  int // the top level, 15 for q4_k and 31 for q5_k

	// The search's two constants, which ggml sets per format.
	rmin       float32
	nstep      int
	calibrated bool

	// scratch
	scales, mins, sw, w []float32
	lv, laux            []uint8

	// The result of the last Encode.
	//
	// D, DMin are the halves as stored; DF, DMinF the floats they read back
	// as, which is what the levels were chosen against (D10's neighbour).
	D, DMin   uint16
	DF, DMinF float32
	LS, LM    []int   // the 6-bit scale and min index per group
	Q         []uint8 // the levels, sub*group of them
}

// newAsymEnc builds the encoder for one format and one super-block length.
func newAsymEnc(q QuantSim, sub int) *asymEnc {
	e := &asymEnc{
		sim: q, sub: sub, group: q.Group, nmax: 1<<q.Bits - 1,
		rmin: -1, nstep: 20,
		calibrated: quantCalibrated(q.Mode),
	}
	// The two constants that steer make_qkx2_quants' sweep: Q4_K's is
	// (-1, 0.1, 20) and Q5_K's (-0.5, 0.1, 15). The calibrated path uses
	// (-0.9, 0.05, 36) for both, and reads them at the call site below.
	if q.Bits == 5 {
		e.rmin, e.nstep = -0.5, 15
	}
	e.scales = make([]float32, sub)
	e.mins = make([]float32, sub)
	e.sw = make([]float32, sub)
	e.LS = make([]int, sub)
	e.LM = make([]int, sub)
	e.w = make([]float32, e.group)
	e.lv = make([]uint8, e.group)
	e.laux = make([]uint8, e.group)
	e.Q = make([]uint8, sub*e.group)
	return e
}

// Encode fits one super-block: `blk` is sub*group contiguous elements of a
// row, and `qw` is the importance of each of them or nil.
//
// It is `quantize_row_q4_K_ref` on the uncalibrated arm and
// `quantize_row_q4_K_impl` on the other two, with the one departure L8c-3
// names: the super-block length is the caller's, because this model has a
// 320-wide dense family and ggml's K-quants want k % 256 == 0.
func (e *asymEnc) Encode(blk []float32, qw []float32) {
	var d, dmin float32
	if e.calibrated {
		// sigma2 is over the **super-block**, where the symmetric arm's is
		// over the whole row: ggml computes it per block of QK_K in
		// quantize_row_q4_K_impl, and its accumulator is a float.
		var sum2 float32
		for _, v := range blk {
			sum2 += v * v
		}
		sigma2 := 2 * sum2 / float32(len(blk))
		avx := sqrt32(sigma2)
		for j := 0; j < e.sub; j++ {
			g := blk[j*e.group : (j+1)*e.group]
			var sumw float32
			for i, v := range g {
				if qw != nil {
					e.w[i] = qw[j*e.group+i] * sqrt32(sigma2+v*v)
				} else {
					e.w[i] = avx + abs32(v)
				}
				sumw += e.w[i]
			}
			e.sw[j] = sumw
			e.scales[j], e.mins[j] = makeQkxQuants(g, e.nmax, e.w, e.lv, e.laux, -0.9, 0.05, 36)
		}
		d = makeQpQuants(e.scales, 63, e.sw, e.LS)
		dmin = makeQpQuants(e.mins, 63, e.sw, e.LM)
	} else {
		var maxScale, maxMin float32
		for j := 0; j < e.sub; j++ {
			g := blk[j*e.group : (j+1)*e.group]
			var sum2 float32
			for _, v := range g {
				sum2 += v * v
			}
			avx := sqrt32(sum2 / float32(e.group))
			for i, v := range g {
				e.w[i] = avx + abs32(v)
			}
			e.scales[j], e.mins[j] = makeQkxQuants(g, e.nmax, e.w, e.lv, e.laux, e.rmin, 0.1, e.nstep)
			if e.scales[j] > maxScale {
				maxScale = e.scales[j]
			}
			if e.mins[j] > maxMin {
				maxMin = e.mins[j]
			}
		}
		var invScale, invMin float32
		if maxScale > 0 {
			invScale = 63 / maxScale
		}
		if maxMin > 0 {
			invMin = 63 / maxMin
		}
		for j := 0; j < e.sub; j++ {
			e.LS[j] = clamp63(nearestInt(invScale * e.scales[j]))
			e.LM[j] = clamp63(nearestInt(invMin * e.mins[j]))
		}
		d, dmin = maxScale/63, maxMin/63
	}
	// **The pair is stored as halves and read back**, so the levels below are
	// chosen against the stored value and not against the one the search
	// returned — the same reason the symmetric arm rounds its scale before
	// quantising (D10's neighbour).
	e.D, e.DMin = safetensors.F32ToF16(d), safetensors.F32ToF16(dmin)
	e.DF, e.DMinF = safetensors.F16ToF32(e.D), safetensors.F16ToF32(e.DMin)
	for j := 0; j < e.sub; j++ {
		g := blk[j*e.group : (j+1)*e.group]
		q := e.Q[j*e.group : (j+1)*e.group]
		ds, dm := e.DF*float32(e.LS[j]), e.DMinF*float32(e.LM[j])
		if ds == 0 {
			// ggml's `if (!d) continue;` leaves the levels alone and
			// dequantises them through a zero scale, so every element of the
			// group comes back as -dm. Writing zero here instead would be a
			// different format.
			for i := range q {
				q[i] = 0
			}
			continue
		}
		for i, v := range g {
			l := nearestInt((v + dm) / ds)
			if l < 0 {
				l = 0
			}
			if l > float32(e.nmax) {
				l = float32(e.nmax)
			}
			q[i] = uint8(l)
		}
	}
}

// Dequant is what a kernel reading this super-block forms: d*sc*l - dmin*m,
// in f32, for element i of the super-block.
func (e *asymEnc) Dequant(i int) float32 {
	j := i / e.group
	return e.DF*float32(e.LS[j])*float32(e.Q[i]) - e.DMinF*float32(e.LM[j])
}

// quantCalibrated reports whether a mode does ggml's scale search — which is
// the same predicate on both forms, and is what `search` and `imatrix` share.
func quantCalibrated(mode string) bool {
	return mode == "search" || mode == "imatrix" ||
		mode == "search+gain" || mode == "imatrix+gain"
}

// packScaleMinK4 is the inverse of the `get_scale_min_k4` every ggml kernel
// reads a K-quant's group parameters through, and of the copy of it in
// llm_moe_gemv.comp: `sub` six-bit pairs into `12*sub/8` bytes, with the two
// high bits of each of the last four pairs stolen from the first four's spare
// ones.
//
// It is ggml's packing verbatim at sub = 8 and nothing at any other length —
// the scheme *is* eight groups, four low and four high — so a super-block
// that is not eight groups is a different packing wearing the same name.
// That is the 320-wide hyper-connection family, and L8c-6 gives it
// `packScaleMin12` rather than a tenth pair this scheme has nowhere to put.
//
// The lengths are the caller's to check, once, before it starts a parallel
// loop: `q4kSuper` is a constant and an error returned from inside
// `parallelFor` would be a data race over a fact that cannot vary.
// `q4kPackFits` is that check.
func packScaleMinK4(dst []byte, ls, lm []int) {
	for i := range dst {
		dst[i] = 0
	}
	for j := 0; j < 8; j++ {
		s, m := byte(ls[j]), byte(lm[j])
		if j < 4 {
			dst[j] = s
			dst[j+4] = m
		} else {
			dst[j+4] = (s & 0xF) | ((m & 0xF) << 4)
			dst[j-4] |= (s >> 4) << 6
			dst[j] |= (m >> 4) << 6
		}
	}
}

// q4kPackFits is the record packings' precondition, stated where a caller can
// act on it: eight groups in ggml's twelve bytes, or ten in L8c-6's fifteen.
func q4kPackFits(sub int) error {
	if sub != asymSuperBlocks && sub != 10 {
		return fmt.Errorf("llm: a record is eight groups (get_scale_min_k4) or ten (L8c-6), not %d", sub)
	}
	return nil
}

// unpackScaleMinK4 is `get_scale_min_k4` itself, for the test that reads the
// bank back the way the shader does.
func unpackScaleMinK4(src []byte, j int) (sc, mn int) {
	if j < 4 {
		return int(src[j] & 63), int(src[j+4] & 63)
	}
	a := src[j+4]
	return int(a&0xF) | int(src[j-4]>>6)<<4, int(a>>4) | int(src[j]>>6)<<4
}

// q4kRecordBytes is one super-block's record for one output column, in bytes:
// the fp16 (d, dmin) pair and then the packed six-bit (scale, min) pairs,
// rounded up to a whole word because every kernel reads the plane through a
// `uint` view.
//
//	sub =  8   4 + 12 = 16 bytes, ggml's own record  (4.500 bits a weight)
//	sub = 10   4 + 15 = 19, rounded to 20            (4.500 bits a weight)
//
// The rounding is the only place the bank is wider than the format L8c-3
// measured: the simulation's 320-wide row is quoted at 4.475 bits because
// nothing there has to be addressable. One byte of padding a record is
// 0.025 bits a weight, and it is stated rather than rounded off.
func q4kRecordBytes(sub int) int { return (4 + (12*sub+7)/8 + 3) &^ 3 }

// packScaleMin12 is the ten-group super-block's packing, and it exists
// because `get_scale_min_k4` is not a length (LLM.md L8c-6).
//
// ggml's scheme steals the two high bits of each of the first four pairs to
// carry the four high ones: it *is* eight groups, four low and four high, and
// there is no ten-group spelling of it. The hyper-connection block's up
// projection reads the low-rank space and is 320 wide, so its super-block is
// the whole row — ten groups — and it gets the obvious packing instead: the
// twelve bits of group j at bit 12*j of a little-endian bit stream, scale in
// the low six and min in the high six.
//
// 120 bits into four words, with the top eight unused. The decode is two
// shifts and a mask (`q4kScaleMin12` in shaders/llm_q4k.glsl), where ggml's
// is a branch on j — so the departure costs nothing to read and is confined
// to the one family that forces it.
func packScaleMin12(dst []byte, ls, lm []int) {
	for i := range dst {
		dst[i] = 0
	}
	for j := range ls {
		v := uint32(ls[j]&63) | uint32(lm[j]&63)<<6
		b := 12 * j
		for i := 0; i < 12; i++ {
			if v>>uint(i)&1 != 0 {
				dst[(b+i)/8] |= 1 << uint((b+i)%8)
			}
		}
	}
}

// unpackScaleMin12 is its inverse, for the test that reads a staged bank back
// the way the shader does.
func unpackScaleMin12(src []byte, j int) (sc, mn int) {
	var v uint32
	b := 12 * j
	for i := 0; i < 12; i++ {
		if src[(b+i)/8]>>uint((b+i)%8)&1 != 0 {
			v |= 1 << uint(i)
		}
	}
	return int(v & 63), int(v >> 6 & 63)
}

// packQ4KRecord writes one column's record: the pair, then whichever packing
// its super-block length has.
func packQ4KRecord(dst []byte, e *asymEnc) error {
	binary.LittleEndian.PutUint16(dst[0:], e.D)
	binary.LittleEndian.PutUint16(dst[2:], e.DMin)
	switch e.sub {
	case asymSuperBlocks:
		packScaleMinK4(dst[4:16], e.LS, e.LM)
	case 10:
		packScaleMin12(dst[4:19], e.LS, e.LM)
	default:
		return fmt.Errorf("llm: no record packing for a %d-group super-block", e.sub)
	}
	return nil
}

// unpackQ4KRecord is the read side, and the shaders' `q4kScaleMin`/
// `q4kScaleMin12` are its two spellings.
func unpackQ4KRecord(src []byte, sub, j int) (d, dmin float32, sc, mn int) {
	d = f16(binary.LittleEndian.Uint16(src[0:]))
	dmin = f16(binary.LittleEndian.Uint16(src[2:]))
	switch sub {
	case asymSuperBlocks:
		sc, mn = unpackScaleMinK4(src[4:16], j)
	default:
		sc, mn = unpackScaleMin12(src[4:19], j)
	}
	return
}
