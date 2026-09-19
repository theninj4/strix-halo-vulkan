package gguf

import (
	"encoding/binary"
	"fmt"

	"strix-halo-vulkan/safetensors"
)

// The five quantised formats UD-Q4_K_XL actually uses, plus the three
// unquantised ones its norms, routers and indexer are in. LLM.md's inventory:
//
//	Q4_K    44.4 GB   the experts' gate and up projections
//	Q5_1    27.1 GB   ffn_down_exps on 43 of 48 layers
//	Q8_0     ~9 GB    every dense tensor, and the 5 remaining down projections
//	Q5_K     1.2 GB
//	IQ4_NL  28.8 GB   the n-gram table, which is gathered rather than streamed
//	F32/BF16 0.3 GB   the routers and the QSA indexer
//
// **Q5_0 and Q6_K are not in that checkpoint and are here for the draft
// head** (P5a): `mtp-…-Q4_K_M.gguf` is a `Q4_K_M` mix rather than unsloth's
// `UD-Q4_K_XL`, so `blk.48` ships its three `hc_*_up` matrices as Q5_0 and
// `hc_ffn_down`, `attn_v` and its own copy of `output` as Q6_K, where the
// trunk ships all of them as Q8_0. Q5_0 is Q5_1's record less the min, with
// the levels centred by subtracting 16 instead; Q6_K is a 256-element
// super-block of sixteen int8 group scales over six-bit levels, centred the
// same way.
//
// These are transcriptions of ggml-quants.c's `dequantize_row_*`, block
// layout included, and they are written to be obviously that rather than to
// be fast: the GPU path dequantises in a shader, and what CPU-side dequant
// is for is checking that path and the loader against llama.cpp. Anything
// clever here would be checking the wrong thing.
//
// A block's scale is fp16, so every path starts by widening one; the
// arithmetic after that is float32, again matching ggml.

// IQ4NLValues is the codebook a nibble of IQ4_NL indexes, for a caller that
// has to *choose* levels rather than read them — P4c's transcode of
// `ffn_down_exps` is the one that does. It is a copy: the table is the
// format's definition and nothing may edit it in place.
func IQ4NLValues() [16]int8 { return kvaluesIQ4NL }

// kvaluesIQ4NL is IQ4_NL's non-linear codebook — the 16 int8 levels a nibble
// indexes, which is what makes it "NL" and not Q4_0.
var kvaluesIQ4NL = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// Dequantize converts n elements of packed data to float32, appending into
// dst so a caller can stage a whole tensor into one buffer.
//
// n must be a whole number of blocks, which is how ggml stores a row: the
// blocking is along Dims[0] only, so a row is always block-aligned and any
// span of whole rows is too.
func Dequantize(typ Type, data []byte, n int64, dst []float32) ([]float32, error) {
	info, ok := types[typ]
	if !ok {
		return nil, fmt.Errorf("gguf: cannot dequantize unknown ggml type %d", uint32(typ))
	}
	if n%int64(info.elems) != 0 {
		return nil, fmt.Errorf("gguf: %d elements is not a whole number of %s blocks of %d", n, info.name, info.elems)
	}
	nb := n / int64(info.elems)
	if want := nb * int64(info.nbytes); int64(len(data)) < want {
		return nil, fmt.Errorf("gguf: %d elements of %s need %d bytes, got %d", n, info.name, want, len(data))
	}

	switch typ {
	case F32:
		for i := int64(0); i < n; i++ {
			dst = append(dst, f32At(data, i))
		}
	case F16:
		for i := int64(0); i < n; i++ {
			dst = append(dst, safetensors.F16ToF32(binary.LittleEndian.Uint16(data[i*2:])))
		}
	case BF16:
		for i := int64(0); i < n; i++ {
			dst = append(dst, bf16ToF32(binary.LittleEndian.Uint16(data[i*2:])))
		}
	case Q8_0:
		dst = dequantQ8_0(data, nb, dst)
	case Q4_1:
		dst = dequantQ4_1(data, nb, dst)
	case Q5_0:
		dst = dequantQ5_0(data, nb, dst)
	case Q5_1:
		dst = dequantQ5_1(data, nb, dst)
	case IQ4_NL:
		dst = dequantIQ4NL(data, nb, dst)
	case Q4_K:
		dst = dequantQ4_K(data, nb, dst)
	case Q5_K:
		dst = dequantQ5_K(data, nb, dst)
	case Q6_K:
		dst = dequantQ6_K(data, nb, dst)
	default:
		return nil, fmt.Errorf("gguf: dequantizing %s is not implemented", info.name)
	}
	return dst, nil
}

// Dequantizable reports whether Dequantize handles a type. The loader uses it
// to refuse a checkpoint at open time rather than halfway through a layer.
func Dequantizable(t Type) bool {
	switch t {
	case F32, F16, BF16, Q8_0, Q4_1, Q5_0, Q5_1, IQ4_NL, Q4_K, Q5_K, Q6_K:
		return true
	}
	return false
}

// Dequantize converts a whole tensor to float32.
func (t *Tensor) Dequantize(dst []float32) ([]float32, error) {
	return Dequantize(t.Type, t.Data, t.Elems(), dst)
}

// DequantizeRow converts one row — the unit a MoE gather and every
// tensor-for-tensor check against llama-eval-callback work in.
func (t *Tensor) DequantizeRow(i int64, dst []float32) ([]float32, error) {
	if i < 0 || i >= t.Rows() {
		return nil, fmt.Errorf("gguf: tensor %q has %d rows, asked for %d", t.Name, t.Rows(), i)
	}
	return Dequantize(t.Type, t.Row(i), t.Dims[0], dst)
}

// dequantQ8_0 is 32 elements as one fp16 scale and 32 int8s: 34 bytes, and
// the format every dense tensor in UD-Q4_K_XL is in.
func dequantQ8_0(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*34:]
		d := f16At(blk, 0)
		for j := 0; j < 32; j++ {
			dst = append(dst, float32(int8(blk[2+j]))*d)
		}
	}
	return dst
}

// dequantQ4_1 is 32 elements as fp16 scale, fp16 min and 16 nibble pairs: 20
// bytes, 5.0 bits a weight. It is Q5_1 without the plane of fifth bits, and
// the same 0..15 / 16..31 split of the pairs — which is why it is the
// cheapest thing `ffn_down_exps` can be narrowed to (P4c).
//
// Nothing in UD-Q4_K_XL ships in it. It is here because P4c transcodes into
// it, and a transcode needs an independent reader to be checked against.
func dequantQ4_1(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*20:]
		d := f16At(blk, 0)
		m := f16At(blk, 2)
		lo := len(dst)
		dst = append(dst, make([]float32, 32)...)
		out := dst[lo:]
		for j := 0; j < 16; j++ {
			out[j] = float32(blk[4+j]&0x0F)*d + m
			out[j+16] = float32(blk[4+j]>>4)*d + m
		}
	}
	return dst
}

// dequantQ5_0 is 32 elements as one fp16 scale, a 32-bit plane of fifth bits
// and 16 nibble pairs: 22 bytes, 5.5 bits a weight. It is Q5_1's record
// without the min, and symmetric — the five-bit level is centred by
// subtracting 16 rather than by an offset the fit chose.
//
// The plane is read the same way, and the same way round: the fifth bit of
// element j is bit j of qh for the low half and bit j+16 for the high one,
// which `qh >> (j+12)` against a mask of 0x10 selects without a shift back.
func dequantQ5_0(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*22:]
		d := f16At(blk, 0)
		qh := binary.LittleEndian.Uint32(blk[2:])
		lo := len(dst)
		dst = append(dst, make([]float32, 32)...)
		out := dst[lo:]
		for j := 0; j < 16; j++ {
			xh0 := byte((qh >> j << 4) & 0x10)
			xh1 := byte((qh >> (j + 12)) & 0x10)
			x0 := int32(blk[6+j]&0x0F|xh0) - 16
			x1 := int32(blk[6+j]>>4|xh1) - 16
			out[j] = float32(x0) * d
			out[j+16] = float32(x1) * d
		}
	}
	return dst
}

// dequantQ5_1 is 32 elements as fp16 scale, fp16 min, a 32-bit plane of
// fifth bits and 16 nibble pairs: 24 bytes. The fifth bit of element j is
// bit j of qh, and the nibble pairs are split 0..15 / 16..31 rather than
// adjacent — which is why the high bit for the upper half is taken from
// `qh >> (j+12)` against a mask of 0x10.
func dequantQ5_1(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*24:]
		d := f16At(blk, 0)
		m := f16At(blk, 2)
		qh := binary.LittleEndian.Uint32(blk[4:])
		lo := len(dst)
		dst = append(dst, make([]float32, 32)...)
		out := dst[lo:]
		for j := 0; j < 16; j++ {
			xh0 := byte((qh >> j << 4) & 0x10)
			xh1 := byte((qh >> (j + 12)) & 0x10)
			x0 := blk[8+j]&0x0F | xh0
			x1 := blk[8+j]>>4 | xh1
			out[j] = float32(x0)*d + m
			out[j+16] = float32(x1)*d + m
		}
	}
	return dst
}

// dequantIQ4NL is 32 elements as one fp16 scale and 16 nibble pairs indexing
// the non-linear codebook: 18 bytes, the same size as Q4_0 for a lower error.
// The n-gram table is 28.80 GB of it.
func dequantIQ4NL(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*18:]
		d := f16At(blk, 0)
		lo := len(dst)
		dst = append(dst, make([]float32, 32)...)
		out := dst[lo:]
		for j := 0; j < 16; j++ {
			out[j] = d * float32(kvaluesIQ4NL[blk[2+j]&0xF])
			out[j+16] = d * float32(kvaluesIQ4NL[blk[2+j]>>4])
		}
	}
	return dst
}

// scaleMinK4 is ggml's get_scale_min_k4: the 6-bit scale and 6-bit min of
// sub-block j, packed into 12 bytes for 8 sub-blocks. The first four are
// plain 6-bit fields; the last four take their low nibble from bytes 8..11
// and their top two bits from the spare bits of bytes 0..7.
func scaleMinK4(j int, q []byte) (sc, m byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return q[j+4]&0xF | q[j-4]>>6<<4, q[j+4]>>4 | q[j]>>6<<4
}

// dequantQ4_K is a 256-element super-block: fp16 scale, fp16 min, the 12
// packed 6-bit scale/min pairs, and 128 bytes of nibbles. 144 bytes, 4.5
// bits a weight, and 44.4 GB of the expert bank.
//
// The nibble order is the format's, not the obvious one: within each 64
// elements the low nibbles of all 32 bytes come first and then the high
// nibbles, so a pair of sub-blocks shares 32 bytes rather than interleaving.
func dequantQ4_K(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*144:]
		d := f16At(blk, 0)
		dmin := f16At(blk, 2)
		scales := blk[4:16]
		q := blk[16:144]
		is := 0
		for j := 0; j < 256; j += 64 {
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for l := 0; l < 32; l++ {
				dst = append(dst, d1*float32(q[l]&0xF)-m1)
			}
			for l := 0; l < 32; l++ {
				dst = append(dst, d2*float32(q[l]>>4)-m2)
			}
			q = q[32:]
			is += 2
		}
	}
	return dst
}

// dequantQ5_K is Q4_K plus a 32-byte plane of fifth bits: 176 bytes for 256
// elements. The plane is read two bits at a time per byte, u1 and u2 walking
// up by 2 for each 64 elements, so one byte of qh serves all eight
// sub-blocks at the same position.
func dequantQ5_K(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*176:]
		d := f16At(blk, 0)
		dmin := f16At(blk, 2)
		scales := blk[4:16]
		qh := blk[16:48]
		ql := blk[48:176]
		is := 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < 256; j += 64 {
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for l := 0; l < 32; l++ {
				v := ql[l] & 0xF
				if qh[l]&u1 != 0 {
					v += 16
				}
				dst = append(dst, d1*float32(v)-m1)
			}
			for l := 0; l < 32; l++ {
				v := ql[l] >> 4
				if qh[l]&u2 != 0 {
					v += 16
				}
				dst = append(dst, d2*float32(v)-m2)
			}
			ql = ql[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return dst
}

// dequantQ6_K is a 256-element super-block: 128 bytes of low nibbles, 64 of
// the pairs of high bits, sixteen **signed int8** group scales and one fp16
// super-scale — 210 bytes, 6.5625 bits a weight.
//
// Two things separate it from the K-quants above and both are in the loop.
// Its scales are plain int8 rather than `get_scale_min_k4`'s packed six-bit
// pairs, and there is **no min**: a level is centred by subtracting 32, so
// the format is symmetric where Q4_K and Q5_K are not. The 128-element pass
// interleaves four quarters that are 32 apart in the output and take their
// scales 2 apart in the group, which is the layout the reference walks with
// four pointers and this walks with offsets.
func dequantQ6_K(data []byte, nb int64, dst []float32) []float32 {
	for b := int64(0); b < nb; b++ {
		blk := data[b*210:]
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		d := f16At(blk, 208)
		lo := len(dst)
		dst = append(dst, make([]float32, 256)...)
		out := dst[lo:]
		for n := 0; n < 256; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				h := qh[n/4+l]
				q1 := int32(ql[n/2+l]&0xF|((h>>0&3)<<4)) - 32
				q2 := int32(ql[n/2+l+32]&0xF|((h>>2&3)<<4)) - 32
				q3 := int32(ql[n/2+l]>>4|((h>>4&3)<<4)) - 32
				q4 := int32(ql[n/2+l+32]>>4|((h>>6&3)<<4)) - 32
				g := sc[n/16+is:]
				out[n+l] = d * float32(int8(g[0])) * float32(q1)
				out[n+l+32] = d * float32(int8(g[2])) * float32(q2)
				out[n+l+64] = d * float32(int8(g[4])) * float32(q3)
				out[n+l+96] = d * float32(int8(g[6])) * float32(q4)
			}
		}
	}
	return dst
}

func f16At(b []byte, off int) float32 {
	return safetensors.F16ToF32(binary.LittleEndian.Uint16(b[off:]))
}

func f32At(b []byte, i int64) float32 {
	return f32FromBits(binary.LittleEndian.Uint32(b[i*4:]))
}
