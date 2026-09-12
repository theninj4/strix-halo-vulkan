package bench

import (
	"encoding/binary"
	"math"
)

// float32ToFloat16 converts f to IEEE754 binary16 bits, rounding the
// mantissa to nearest-even (matching what the GPU's float16_t cast does —
// truncating instead would silently disagree with the shader by up to a
// couple of ULPs and make correctness checks flag phantom mismatches).
// Values too small to represent as a normal half are flushed to zero
// rather than represented as half-subnormals — fine for this benchmark's
// data ranges (weights/scales drawn from [-1,1], nowhere near that threshold).
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	if exp <= 0 {
		return sign
	}
	if exp >= 0x1F {
		return sign | 0x7C00
	}

	halfMant := mant >> 13
	remainder := mant & 0x1FFF
	const roundBit = uint32(0x1000) // half of 0x2000 (2^13), the discarded-bits midpoint
	if remainder > roundBit || (remainder == roundBit && halfMant&1 == 1) {
		halfMant++
		if halfMant == 0x400 { // mantissa overflowed into the exponent
			halfMant = 0
			exp++
			if exp >= 0x1F {
				return sign | 0x7C00
			}
		}
	}
	return sign | uint16(exp<<10) | uint16(halfMant)
}

// float16ToFloat32 converts IEEE754 binary16 bits back to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	case exp == 0x1F:
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	}
	exp = exp - 15 + 127
	return math.Float32frombits(sign | (uint32(exp) << 23) | (mant << 13))
}

// float32SliceToBytes reinterprets fp32 values as their little-endian byte
// representation, for buffers a shader reads as plain `float`.
func float32SliceToBytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

func float16SliceToBytes(v []uint16) []byte {
	out := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(out[i*2:], x)
	}
	return out
}

func int8SliceToBytes(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// float16RoundTrip quantizes data to fp16 and back, matching exactly what a
// shader reading a float16_t buffer will see — so a CPU reference computed
// against this round-tripped data can be compared to the GPU's fp16-path
// output without fp16-rounding itself being mistaken for a bug.
func float16RoundTrip(data []float32) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float16ToFloat32(float32ToFloat16(v))
	}
	return out
}

func float32SliceToFloat16Bytes(data []float32) []byte {
	bits := make([]uint16, len(data))
	for i, v := range data {
		bits[i] = float32ToFloat16(v)
	}
	return float16SliceToBytes(bits)
}

// quantizeQ8 quantizes data (rows x cols, row-major) into per-block
// symmetric int8 weights (one float16 scale per block of `block` contiguous
// elements along a row): q[i] = round(data[i]/scale), scale = max(|block|)/127.
// Matches shaders/gemv_naive.comp and gemm_naive.comp's PRECISION_Q8 path.
func quantizeQ8(data []float32, rows, cols, block int) (q []int8, scales []uint16) {
	blocksPerRow := cols / block
	q = make([]int8, len(data))
	scales = make([]uint16, rows*blocksPerRow)
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocksPerRow; bi++ {
			start := r*cols + bi*block
			var maxAbs float32
			for i := 0; i < block; i++ {
				if a := absF32(data[start+i]); a > maxAbs {
					maxAbs = a
				}
			}
			scale := maxAbs / 127.0
			if scale == 0 {
				scale = 1.0
			}
			scaleBits := float32ToFloat16(scale)
			scales[r*blocksPerRow+bi] = scaleBits
			scaleF := float16ToFloat32(scaleBits)
			for i := 0; i < block; i++ {
				qv := clampInt(int(math.Round(float64(data[start+i]/scaleF))), -127, 127)
				q[start+i] = int8(qv)
			}
		}
	}
	return q, scales
}

// dequantizeQ8 is the exact inverse quantizeQ8's shader-side reader
// performs, used to build a CPU reference for correctness checks.
func dequantizeQ8(q []int8, scales []uint16, rows, cols, block int) []float32 {
	blocksPerRow := cols / block
	out := make([]float32, len(q))
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocksPerRow; bi++ {
			scale := float16ToFloat32(scales[r*blocksPerRow+bi])
			start := r*cols + bi*block
			for i := 0; i < block; i++ {
				out[start+i] = float32(q[start+i]) * scale
			}
		}
	}
	return out
}

// quantizeQ4 quantizes data into per-block symmetric int4 weights (range
// [-8,7]) packed two-per-byte (low nibble = even index, high nibble = odd
// index) plus one float16 scale per block. Matches gemv_naive.comp's
// PRECISION_Q4 path. block must be even.
func quantizeQ4(data []float32, rows, cols, block int) (packed []uint8, scales []uint16) {
	blocksPerRow := cols / block
	packed = make([]uint8, len(data)/2)
	scales = make([]uint16, rows*blocksPerRow)
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocksPerRow; bi++ {
			start := r*cols + bi*block
			var maxAbs float32
			for i := 0; i < block; i++ {
				if a := absF32(data[start+i]); a > maxAbs {
					maxAbs = a
				}
			}
			scale := maxAbs / 7.0
			if scale == 0 {
				scale = 1.0
			}
			scaleBits := float32ToFloat16(scale)
			scales[r*blocksPerRow+bi] = scaleBits
			scaleF := float16ToFloat32(scaleBits)
			for i := 0; i < block; i += 2 {
				idx0, idx1 := start+i, start+i+1
				q0 := clampInt(int(math.Round(float64(data[idx0]/scaleF))), -8, 7) + 8
				q1 := clampInt(int(math.Round(float64(data[idx1]/scaleF))), -8, 7) + 8
				packed[idx0/2] = uint8(q0) | uint8(q1<<4)
			}
		}
	}
	return packed, scales
}

// dequantizeQ4 is the exact inverse of quantizeQ4's shader-side reader.
func dequantizeQ4(packed []uint8, scales []uint16, rows, cols, block int) []float32 {
	blocksPerRow := cols / block
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocksPerRow; bi++ {
			scale := float16ToFloat32(scales[r*blocksPerRow+bi])
			start := r*cols + bi*block
			for i := 0; i < block; i++ {
				idx := start + i
				b := packed[idx/2]
				var nibble uint8
				if idx%2 == 0 {
					nibble = b & 0xF
				} else {
					nibble = b >> 4
				}
				out[idx] = float32(int(nibble)-8) * scale
			}
		}
	}
	return out
}
