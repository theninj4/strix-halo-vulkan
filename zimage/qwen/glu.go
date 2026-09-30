package qwen

// InterleaveGLU lays a SwiGLU's gate and up projections ([n, k] each) out
// as one [2n, k] weight in groups of 64 rows -- 32 gate rows, then the up
// rows of the same 32 outputs -- which is what dit_gemm.comp's C_SWIGLU
// epilogue is built on (KERNELS.md G3, research §2.14): a 64-column block
// of the fused GEMM then holds 32 gate columns and the up columns of the
// same 32 outputs, so a wave's four accumulator tiles are gate 0-1 and up
// 0-1 of one layout. dst holds 2*n*k.
func InterleaveGLU(dst, gate, up []float32, n, k int) {
	if n%32 != 0 {
		panic("qwen: InterleaveGLU wants n a multiple of 32")
	}
	parallelFor(n/32, func(g int) {
		copy(dst[(64*g)*k:(64*g+32)*k], gate[32*g*k:(32*g+32)*k])
		copy(dst[(64*g+32)*k:(64*g+64)*k], up[32*g*k:(32*g+32)*k])
	})
}
