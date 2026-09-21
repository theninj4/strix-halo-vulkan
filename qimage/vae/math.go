package vae

import "math"

// exp32 is math.Exp at float32 width. SiLU is evaluated elementwise over
// tens of millions of values, and the float64 conversion is the whole cost,
// but correctness first: this matches PyTorch, which also computes sigmoid
// in the tensor's own precision via a float32 expf.
func exp32(v float32) float32 { return float32(math.Exp(float64(v))) }

// ceilDiv and floorDiv are integer division that rounds the way the name
// says at *both* signs, which Go's own truncating division does not. The
// strided convolution's column bounds need them: the numerator is
// `pad - kw`, which is negative for every tap right of the padding.
func ceilDiv(a, b int) int {
	if a <= 0 {
		return -((-a) / b)
	}
	return (a + b - 1) / b
}

func floorDiv(a, b int) int {
	if a >= 0 {
		return a / b
	}
	return -ceilDiv(-a, b)
}
