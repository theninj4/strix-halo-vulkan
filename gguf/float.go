package gguf

import "math"

// f32FromBits and bf16ToF32 are the two widenings the dequant paths need
// that safetensors does not already export: a float32 from its bits, and
// bfloat16 (the QSA indexer's format) from its top-16-bits-of-a-float32
// representation.
func f32FromBits(b uint32) float32 { return math.Float32frombits(b) }

func bf16ToF32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
