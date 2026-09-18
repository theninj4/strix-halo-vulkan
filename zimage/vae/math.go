package vae

import "math"

func sqrt64(v float64) float64 { return math.Sqrt(v) }

// exp32 is math.Exp at float32 width. SiLU is evaluated elementwise over
// tens of millions of values, and the float64 conversion is the whole cost,
// but correctness first: this matches PyTorch, which also computes sigmoid
// in the tensor's own precision via a float32 expf.
func exp32(v float32) float32 { return float32(math.Exp(float64(v))) }

// tanh32 is math.Tanh at float32 width, for taef1's input clamp. Same
// argument as exp32: PyTorch evaluates it in the tensor's own precision.
func tanh32(v float32) float32 { return float32(math.Tanh(float64(v))) }
