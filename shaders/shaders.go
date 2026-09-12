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

// Strided read bandwidth (IDEAS §5.1b): the same fixed set of bytes read
// with the rows spaced an arbitrary stride apart, in five request shapes.
// LANES_PER_ROW is how many consecutive 16-byte chunks of one row go to
// consecutive lanes, so one 64-lane load instruction spans 64/LANES_PER_ROW
// distinct rows: 64 is a plain contiguous sweep, 1 is a 64-address gather of
// 16 bytes each, and 2 is the shape a 16x16 fp16 coopMatLoad issues. The
// stride is a push constant, so each variant sweeps it without recompiling.

//go:generate glslc --target-env=vulkan1.2 -O -DLANES_PER_ROW=64 -o strided_read_lpr64.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DLANES_PER_ROW=16 -o strided_read_lpr16.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DLANES_PER_ROW=4 -o strided_read_lpr4.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DLANES_PER_ROW=2 -o strided_read_lpr2.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DLANES_PER_ROW=1 -o strided_read_lpr1.spv strided_read.comp

//go:embed strided_read_lpr64.spv
var StridedReadLPR64 []byte

//go:embed strided_read_lpr16.spv
var StridedReadLPR16 []byte

//go:embed strided_read_lpr4.spv
var StridedReadLPR4 []byte

//go:embed strided_read_lpr2.spv
var StridedReadLPR2 []byte

//go:embed strided_read_lpr1.spv
var StridedReadLPR1 []byte

// The same five shapes with the traversal swapped (IDEAS §5.1b's remaining
// question): CROSS_WAVE makes consecutive 64-lane requests advance by a row
// group instead of by a column block, so the loads in flight together are a
// stride apart rather than contiguous with one another. Same byte set, same
// instruction count, same checksum — only which addresses are outstanding
// at the same moment changes. This is the GEMV/W4A8 pattern, and it is the
// one thing the five variants above cannot express, because each of them
// varies addresses only *inside* a request.

//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=64 -o strided_read_xw_lpr64.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=16 -o strided_read_xw_lpr16.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=4 -o strided_read_xw_lpr4.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=2 -o strided_read_xw_lpr2.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=1 -o strided_read_xw_lpr1.spv strided_read.comp

//go:embed strided_read_xw_lpr64.spv
var StridedReadXWLPR64 []byte

//go:embed strided_read_xw_lpr16.spv
var StridedReadXWLPR16 []byte

//go:embed strided_read_xw_lpr4.spv
var StridedReadXWLPR4 []byte

//go:embed strided_read_xw_lpr2.spv
var StridedReadXWLPR2 []byte

//go:embed strided_read_xw_lpr1.spv
var StridedReadXWLPR1 []byte

// The cross-wave traversal again, with each request issuing LOADS_PER_WAVE
// loads along its own row instead of retiring after one. At the top of the
// ladder (LOADS_PER_WAVE = chunksPerRow/LANES_PER_ROW) one wave streams a
// whole row and consecutive waves take consecutive rows, which is the shape
// every GEMV kernel here has — so this is what says whether the traversal
// penalty is something a real kernel inherits or something only a
// one-load-per-wave dispatch can produce. Contiguous requests only
// (LANES_PER_ROW=64): the gather shapes have their own, separate penalty
// and would confound the ladder.

//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=64 -DLOADS_PER_WAVE=2 -o strided_read_xw_l2.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=64 -DLOADS_PER_WAVE=4 -o strided_read_xw_l4.spv strided_read.comp
//go:generate glslc --target-env=vulkan1.2 -O -DCROSS_WAVE=1 -DLANES_PER_ROW=64 -DLOADS_PER_WAVE=8 -o strided_read_xw_l8.spv strided_read.comp

//go:embed strided_read_xw_l2.spv
var StridedReadXWL2 []byte

//go:embed strided_read_xw_l4.spv
var StridedReadXWL4 []byte

//go:embed strided_read_xw_l8.spv
var StridedReadXWL8 []byte

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

// GEMV W4A8: 4-bit weights fed to the same packed-int8 dot instruction,
// against int8 activations. Q4's bytes with W8A8's arithmetic — see
// gemv_w4a8.comp. VEC=1 loads one uint (8 weights) per lane per step,
// VEC=4 one uvec4 (32 weights), which is the load-width half of IDEAS §1.3.

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_w4a8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=4 -o gemv_w4a8_v4.spv gemv_w4a8.comp

//go:embed gemv_w4a8.spv
var GEMVW4A8 []byte

//go:embed gemv_w4a8_v4.spv
var GEMVW4A8Vec4 []byte

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

// GEMM WMMA, register-blocked: the same cooperative-matrix hardware as
// gemm_coopmat_fp16.comp, restructured for arithmetic intensity (IDEAS §2.1).
// The variants form an ablation over four independent axes, so each effect is
// attributable rather than bundled:
//
//   - register blocking: WM x WN 16x16 accumulators per wave instead of one,
//     raising arithmetic intensity from 8 to 16..85 FLOP/byte
//   - four waves per workgroup with no LDS (wg*): the waves share A rows and
//     B columns through the L0/L1 caches rather than through shared memory,
//     which costs nothing but relies on the cache to dedup the overlap
//   - LDS staging (lds*): four waves share each staged K-slab explicitly
//   - double buffering (*_db): one barrier per K-step instead of two, with
//     the next slab's global loads issued underneath the current slab's MMAs
//   - B stored [N,K] and loaded column-major (*_bt), IDEAS §2.3: this is the
//     layout RDNA's WMMA B operand actually wants — without it every B
//     fragment is gathered by sixteen scalar 16-bit loads rather than two
//     buffer_load_b128 — and it is also how a real Linear weight is stored
//
// Tile geometry is -D rather than specialization constants because it sizes
// register and `shared` arrays; see the shader header.
//
// Operand *strides* are deliberately not an axis here: the kernel takes both
// leading dimensions as push constants, so the stride-padding cases that
// IDEAS §2.3 turned into a 1.13x on the best kernel (and a 1.6x on the
// transposed-B one) reuse these same binaries at a different `ldb`/`lda`.
// See the `pad*` rows in bench/ops_gemm_wmma.go.

//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -o gemm_wmma_reg32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -o gemm_wmma_reg64.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DB_COLMAJOR=1 -o gemm_wmma_reg64_bt.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=8 -o gemm_wmma_reg64x128.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=4 -o gemm_wmma_reg64_bt_k64.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -o gemm_wmma_wg128.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -o gemm_wmma_wg128x256.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DUSE_LDS=1 -o gemm_wmma_lds128.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DUSE_LDS=1 -DDOUBLE_BUFFER=1 -o gemm_wmma_lds128_db.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DUSE_LDS=1 -DDOUBLE_BUFFER=1 -DB_COLMAJOR=1 -o gemm_wmma_lds128_db_bt.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DUSE_LDS=1 -DDOUBLE_BUFFER=1 -DB_COLMAJOR=1 -DBK_TILES=2 -o gemm_wmma_lds128k32_db_bt.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=8 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DUSE_LDS=1 -DDOUBLE_BUFFER=1 -DB_COLMAJOR=1 -o gemm_wmma_lds256x128_db_bt.spv gemm_wmma.comp

//go:embed gemm_wmma_reg32.spv
var GEMMWMMAReg32 []byte

//go:embed gemm_wmma_reg64.spv
var GEMMWMMAReg64 []byte

//go:embed gemm_wmma_reg64_bt.spv
var GEMMWMMAReg64BT []byte

//go:embed gemm_wmma_reg64x128.spv
var GEMMWMMAReg64x128 []byte

//go:embed gemm_wmma_reg64_bt_k64.spv
var GEMMWMMAReg64BTK64 []byte

//go:embed gemm_wmma_wg128.spv
var GEMMWMMAWG128 []byte

//go:embed gemm_wmma_wg128x256.spv
var GEMMWMMAWG128x256 []byte

//go:embed gemm_wmma_lds128.spv
var GEMMWMMALDS128 []byte

//go:embed gemm_wmma_lds128_db.spv
var GEMMWMMALDS128DB []byte

//go:embed gemm_wmma_lds128_db_bt.spv
var GEMMWMMALDS128DBBT []byte

//go:embed gemm_wmma_lds128k32_db_bt.spv
var GEMMWMMALDS128K32DBBT []byte

//go:embed gemm_wmma_lds256x128_db_bt.spv
var GEMMWMMALDS256x128DBBT []byte

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
