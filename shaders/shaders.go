// Package shaders embeds pre-compiled SPIR-V binaries.
//
// Regenerate with `go generate ./...` after editing any .comp source
// (requires glslc, part of the Vulkan SDK / shaderc). Precision/tile-size
// variants of the same .comp source are generated via glslc -D flags rather
// than duplicating GLSL, and embedded under distinct names below.
package shaders

//go:generate glslc --target-env=vulkan1.2 -O -o double.spv double.comp

import _ "embed"

//go:embed double.spv
var Double []byte

// Bandwidth baseline.

//go:generate glslc --target-env=vulkan1.2 -O -o copy.spv copy.comp

//go:embed copy.spv
var Copy []byte

// Per-dispatch overhead floor: a shader that does nothing.

//go:generate glslc --target-env=vulkan1.2 -O -o empty.spv empty.comp

//go:embed empty.spv
var Empty []byte

// ALU / matrix-core peak microbenchmarks. Operands are register-resident
// before the loop, so these measure pure instruction-issue throughput with
// no memory traffic — the denominator for reading every other kernel's
// GFLOP/s as a fraction of what the hardware can actually issue. ACC (the
// number of independent accumulator chains) is baked per variant; the
// scalar paths get 8, the coopmat paths 4 (each accumulator is a full
// 16x16 fp32/int32 tile and so far more register-hungry).

//go:generate glslc --target-env=vulkan1.2 -O -DMODE_FMA -DACC=8 -o alu_peak_fma.spv alu_peak.comp
//go:generate glslc --target-env=vulkan1.2 -O -DMODE_PKFMA -DACC=8 -o alu_peak_pkfma.spv alu_peak.comp
//go:generate glslc --target-env=vulkan1.2 -O -DMODE_DOT4 -DACC=8 -o alu_peak_dot4.spv alu_peak.comp
//go:generate glslc --target-env=vulkan1.2 -O -DMODE_WMMA_F16 -DACC=4 -o alu_peak_wmma_f16.spv alu_peak.comp
//go:generate glslc --target-env=vulkan1.2 -O -DMODE_WMMA_I8 -DACC=4 -o alu_peak_wmma_i8.spv alu_peak.comp

//go:embed alu_peak_fma.spv
var ALUPeakFMA []byte

//go:embed alu_peak_pkfma.spv
var ALUPeakPackedFMA []byte

//go:embed alu_peak_dot4.spv
var ALUPeakDot4 []byte

//go:embed alu_peak_wmma_f16.spv
var ALUPeakWMMAFP16 []byte

//go:embed alu_peak_wmma_i8.spv
var ALUPeakWMMAInt8 []byte

// Elementwise / activation.

//go:generate glslc --target-env=vulkan1.2 -O -o elementwise_f32.spv elementwise.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_F16 -o elementwise_f16.spv elementwise.comp

//go:embed elementwise_f32.spv
var ElementwiseF32 []byte

//go:embed elementwise_f16.spv
var ElementwiseF16 []byte

// GEMV: y = W*x. Naive (one thread per row) and subgroup (one subgroup per
// row, subgroupAdd reduction) flavours, each in fp32/fp16/Q8/Q4 weights.

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_naive_f32.spv gemv_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_F16 -o gemv_naive_f16.spv gemv_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q8 -o gemv_naive_q8.spv gemv_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q4 -o gemv_naive_q4.spv gemv_naive.comp

//go:embed gemv_naive_f32.spv
var GEMVNaiveF32 []byte

//go:embed gemv_naive_f16.spv
var GEMVNaiveF16 []byte

//go:embed gemv_naive_q8.spv
var GEMVNaiveQ8 []byte

//go:embed gemv_naive_q4.spv
var GEMVNaiveQ4 []byte

// GEMV W8A8: both weights and activations int8, reduced via
// VK_KHR_shader_integer_dot_product's packed dot instruction instead of
// dequant-and-multiply. Subgroup reduction only (see gemv_w8a8.comp).

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_w8a8.spv gemv_w8a8.comp

//go:embed gemv_w8a8.spv
var GEMVW8A8 []byte

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_subgroup_f32.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_F16 -o gemv_subgroup_f16.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q8 -o gemv_subgroup_q8.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q4 -o gemv_subgroup_q4.spv gemv_subgroup.comp

//go:embed gemv_subgroup_f32.spv
var GEMVSubgroupF32 []byte

//go:embed gemv_subgroup_f16.spv
var GEMVSubgroupF16 []byte

//go:embed gemv_subgroup_q8.spv
var GEMVSubgroupQ8 []byte

//go:embed gemv_subgroup_q4.spv
var GEMVSubgroupQ4 []byte

// GEMM: C = A*B. Naive (fp32/fp16/Q8 weights), shared-memory tiled
// (fp32/fp16), and cooperative-matrix (fp16 and int8, via the RDNA3.5
// matrix-multiply accelerator).

//go:generate glslc --target-env=vulkan1.2 -O -o gemm_naive_f32.spv gemm_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_F16 -o gemm_naive_f16.spv gemm_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q8 -o gemm_naive_q8.spv gemm_naive.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q4 -o gemm_naive_q4.spv gemm_naive.comp

//go:embed gemm_naive_f32.spv
var GEMMNaiveF32 []byte

//go:embed gemm_naive_f16.spv
var GEMMNaiveF16 []byte

//go:embed gemm_naive_q8.spv
var GEMMNaiveQ8 []byte

//go:embed gemm_naive_q4.spv
var GEMMNaiveQ4 []byte

// Dequantizes a Q4-packed buffer into plain fp16 — the "dequant once, reuse
// across every subsequent matmul" half of the two-pass Q4+coopmat strategy.

//go:generate glslc --target-env=vulkan1.2 -O -o dequant_q4_to_f16.spv dequant_q4_to_f16.comp

//go:embed dequant_q4_to_f16.spv
var DequantQ4ToF16 []byte

//go:generate glslc --target-env=vulkan1.2 -O -DTILE=16 -o gemm_tiled_f32.spv gemm_tiled.comp
//go:generate glslc --target-env=vulkan1.2 -O -DTILE=16 -DPRECISION_F16 -o gemm_tiled_f16.spv gemm_tiled.comp
//go:generate glslc --target-env=vulkan1.2 -O -DTILE=16 -DPRECISION_Q8 -o gemm_tiled_q8.spv gemm_tiled.comp
//go:generate glslc --target-env=vulkan1.2 -O -DTILE=16 -DPRECISION_Q4 -o gemm_tiled_q4.spv gemm_tiled.comp

//go:embed gemm_tiled_f32.spv
var GEMMTiledF32 []byte

//go:embed gemm_tiled_f16.spv
var GEMMTiledF16 []byte

//go:embed gemm_tiled_q8.spv
var GEMMTiledQ8 []byte

//go:embed gemm_tiled_q4.spv
var GEMMTiledQ4 []byte

// GEMM W8A8: both A and B int8, reduced via dotPacked4x8EXT. Naive (one
// thread per output element) only — see gemm_w8a8.comp for why this isn't
// pitted against the coopmat variants (different hardware path, already
// covered by gemm_coopmat_int8.comp).

//go:generate glslc --target-env=vulkan1.2 -O -o gemm_w8a8.spv gemm_w8a8.comp

//go:embed gemm_w8a8.spv
var GEMMW8A8 []byte

//go:generate glslc --target-env=vulkan1.2 -O -o gemm_coopmat_fp16.spv gemm_coopmat_fp16.comp
//go:generate glslc --target-env=vulkan1.2 -O -o gemm_coopmat_int8.spv gemm_coopmat_int8.comp
//go:generate glslc --target-env=vulkan1.2 -O -o gemm_coopmat_q4.spv gemm_coopmat_q4.comp

//go:embed gemm_coopmat_fp16.spv
var GEMMCoopMatFP16 []byte

//go:embed gemm_coopmat_int8.spv
var GEMMCoopMatInt8 []byte

//go:embed gemm_coopmat_q4.spv
var GEMMCoopMatQ4 []byte

// Reductions: RMSNorm and softmax, each with a shared-memory-tree and a
// subgroup-op reduction flavour.

//go:generate glslc --target-env=vulkan1.2 -O -o rmsnorm_shared.spv rmsnorm_shared.comp
//go:generate glslc --target-env=vulkan1.2 -O -o rmsnorm_subgroup.spv rmsnorm_subgroup.comp

//go:embed rmsnorm_shared.spv
var RMSNormShared []byte

//go:embed rmsnorm_subgroup.spv
var RMSNormSubgroup []byte

//go:generate glslc --target-env=vulkan1.2 -O -o softmax_shared.spv softmax_shared.comp
//go:generate glslc --target-env=vulkan1.2 -O -o softmax_subgroup.spv softmax_subgroup.comp

//go:embed softmax_shared.spv
var SoftmaxShared []byte

//go:embed softmax_subgroup.spv
var SoftmaxSubgroup []byte
