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
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -o gemv_w8a8_w32.spv gemv_w8a8.comp

//go:embed gemv_w8a8.spv
var GEMVW8A8 []byte

//go:embed gemv_w8a8_w32.spv
var GEMVW8A8W32 []byte

// GEMV W4A8: 4-bit weights fed to the same packed-int8 dot instruction,
// against int8 activations. Q4's bytes with W8A8's arithmetic — see
// gemv_w4a8.comp. VEC=1 loads one uint (8 weights) per lane per step,
// VEC=4 one uvec4 (32 weights), which is the load-width half of IDEAS §1.3.

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_w4a8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=4 -o gemv_w4a8_v4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -o gemv_w4a8_w32.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=4 -DWAVE=32 -o gemv_w4a8_v4_w32.spv gemv_w4a8.comp

//go:embed gemv_w4a8.spv
var GEMVW4A8 []byte

//go:embed gemv_w4a8_v4.spv
var GEMVW4A8Vec4 []byte

// The wave32 arm of IDEAS §6.2. Decode is bandwidth-bound — W4A8 already
// reads at 89% of the DRAM bus — so the expected answer here is "no change",
// and that is worth a measurement precisely because it is the family where
// the wave size changes the *access pattern*: half the lanes per row means
// each lane strides twice as far, and §5.1b's coverage law is about exactly
// that. The pair is the falsification test the law asks for.

//go:embed gemv_w4a8_w32.spv
var GEMVW4A8W32 []byte

//go:embed gemv_w4a8_v4_w32.spv
var GEMVW4A8Vec4W32 []byte

// The four cells that complete IDEAS §6.2's follow-up into a 2x2x2 over
// (wave size) x (load width) x (rows per workgroup). §6.2 measured only the
// ROWS=1 face of it and found the wave32 VEC=4 decode GEMV reading 96% of
// the DRAM bus against wave64's 89%, reproducibly and unexplained; the wave
// size changes three things at once there, and these arms separate them.
//
//   - VEC=8 doubles the bytes one lane asks for per step, so wave32 at VEC=8
//     issues the same 1024 B per row per request round that wave64 does at
//     VEC=4. If the win is a function of that round size, this cell inherits
//     wave64's number, not wave32's.
//   - ROWS=2 puts two subgroups in a workgroup, which gives a wave32
//     pipeline back the 64 threads per workgroup and half the grid width
//     that wave64 has, while leaving 32 lanes sweeping each row. If the win
//     is about workgroup count or scheduling rather than the per-row sweep,
//     it disappears here.
//
// Both knobs are orthogonal to the wave size, so the four wave64 cells are
// the controls that say whether either does anything on its own.

//go:generate glslc --target-env=vulkan1.2 -O -DVEC=8 -o gemv_w4a8_v8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=8 -DWAVE=32 -o gemv_w4a8_v8_w32.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=16 -o gemv_w4a8_v16.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=16 -DWAVE=32 -o gemv_w4a8_v16_w32.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=4 -DROWS=2 -o gemv_w4a8_v4_r2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=4 -DWAVE=32 -DROWS=2 -o gemv_w4a8_v4_w32_r2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=8 -DROWS=2 -o gemv_w4a8_v8_r2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DVEC=8 -DWAVE=32 -DROWS=2 -o gemv_w4a8_v8_w32_r2.spv gemv_w4a8.comp

//go:embed gemv_w4a8_v8.spv
var GEMVW4A8Vec8 []byte

//go:embed gemv_w4a8_v8_w32.spv
var GEMVW4A8Vec8W32 []byte

// VEC=16 is the arm the VEC=8 result asked for: at wave64 it puts 4096 B of
// one row in a single lane-step, which is a whole weight row at N=8192 — the
// shape the target models actually decode at, and the one N where VEC=8 is
// still two steps short of the bus.

//go:embed gemv_w4a8_v16.spv
var GEMVW4A8Vec16 []byte

//go:embed gemv_w4a8_v16_w32.spv
var GEMVW4A8Vec16W32 []byte

//go:embed gemv_w4a8_v4_r2.spv
var GEMVW4A8Vec4Rows2 []byte

//go:embed gemv_w4a8_v4_w32_r2.spv
var GEMVW4A8Vec4W32Rows2 []byte

//go:embed gemv_w4a8_v8_r2.spv
var GEMVW4A8Vec8Rows2 []byte

//go:embed gemv_w4a8_v8_w32_r2.spv
var GEMVW4A8Vec8W32Rows2 []byte

// The grouped (MoE decode) arm of the same kernel: -DGROUPED=1 adds a table
// of routed (token, expert) pairs, a weight-bank row stride and per-row
// activations, so one dispatch covers every pair a decode step routes. This
// is the kernel IDEAS §3.5 called the honest decode measurement and did not
// have — it measured decode with the grouped *GEMM* at M=1, where 15 of every
// 16 rows of each cooperative-matrix tile, and of A's traffic, are padding.
//
// The load widths are the same four, because §1.7's rule (one lane-step
// covers a whole weight row, VEC = K/(8*WAVE)) predicts a different winner
// for each of the two expert shapes: VEC=4 covers a 4-bit K=640 row and VEC=8
// a K=2560 one, both at wave64. The two wave32 arms are the control that says
// whether the wave size does anything once C is held fixed, which §1.7 says
// it does not.
//
// GROUPED=0 is the default and is textually the file these binaries were
// already built from, so every non-grouped .spv above is byte-identical
// across this change (`cmp`-verified).

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -o gemv_w4a8_g.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -o gemv_w4a8_g_v4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=8 -o gemv_w4a8_g_v8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=16 -o gemv_w4a8_g_v16.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DWAVE=32 -o gemv_w4a8_g_v4_w32.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=8 -DWAVE=32 -o gemv_w4a8_g_v8_w32.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g.spv
var GEMVW4A8Grouped []byte

//go:embed gemv_w4a8_g_v4.spv
var GEMVW4A8GroupedVec4 []byte

//go:embed gemv_w4a8_g_v8.spv
var GEMVW4A8GroupedVec8 []byte

//go:embed gemv_w4a8_g_v16.spv
var GEMVW4A8GroupedVec16 []byte

//go:embed gemv_w4a8_g_v4_w32.spv
var GEMVW4A8GroupedVec4W32 []byte

//go:embed gemv_w4a8_g_v8_w32.spv
var GEMVW4A8GroupedVec8W32 []byte

// The M-blocked grouped builds (IDEAS §1.8 finding 2, the "not built" fix):
// -DMROWS=n gives one workgroup n activation rows against one weight row, so
// an expert's weights are read once per n routed pairs instead of once per
// pair. The sweep is 2/4/8 at the width that won the unblocked arm at both
// expert shapes (VEC=4), plus one VEC=8 cell to say whether the load width
// and the M block interact — at MROWS=8 a step holds eight slots' activation
// uvec4s live at once, so this axis spends registers where the load width
// spends them too.

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=2 -o gemv_w4a8_g_v4_m2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=4 -o gemv_w4a8_g_v4_m4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=8 -o gemv_w4a8_g_v4_m8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=8 -DMROWS=2 -o gemv_w4a8_g_v8_m2.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g_v4_m2.spv
var GEMVW4A8GroupedVec4M2 []byte

//go:embed gemv_w4a8_g_v4_m4.spv
var GEMVW4A8GroupedVec4M4 []byte

//go:embed gemv_w4a8_g_v4_m8.spv
var GEMVW4A8GroupedVec4M8 []byte

//go:embed gemv_w4a8_g_v8_m2.spv
var GEMVW4A8GroupedVec8M2 []byte

// The output-row blocks (IDEAS §1.8 finding 9, the one open cell of the decode
// path: the down projection reads 198 GB/s where gate_up reads 235 for
// identical bytes per expert, with 4x the workgroups and 4x the output
// writes). Two arms that divide the workgroup count the same way and differ in
// everything else:
//
//   -DROWS=n  puts n subgroups in a workgroup, one output row each. The wave
//             count, the loads, the reductions and the stores are all exactly
//             what they were; only the number of workgroups changes.
//   -DNROWS=n gives one subgroup n consecutive weight rows against one
//             activation row, so the wave count falls with the workgroup
//             count, the activation loads are shared n ways, the n results
//             merge into one wide store, and each active lane carries n rows'
//             dots against one fixed per-wave cost. (It does not fill down's
//             44 idle lanes: the load loop is still strided by the wave size,
//             so the same 20 lanes do n times the work.)
//
// Subtracting them is the experiment. Both are built at VEC=4, the width that
// won the unblocked arm at both expert shapes (§1.8 finding 7: the spread
// across widths is 3-5% here).

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DROWS=2 -o gemv_w4a8_g_v4_r2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DROWS=4 -o gemv_w4a8_g_v4_r4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DROWS=8 -o gemv_w4a8_g_v4_r8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DNROWS=2 -o gemv_w4a8_g_v4_n2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DNROWS=4 -o gemv_w4a8_g_v4_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DNROWS=8 -o gemv_w4a8_g_v4_n8.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g_v4_r2.spv
var GEMVW4A8GroupedVec4Rows2 []byte

//go:embed gemv_w4a8_g_v4_r4.spv
var GEMVW4A8GroupedVec4Rows4 []byte

//go:embed gemv_w4a8_g_v4_r8.spv
var GEMVW4A8GroupedVec4Rows8 []byte

//go:embed gemv_w4a8_g_v4_n2.spv
var GEMVW4A8GroupedVec4N2 []byte

//go:embed gemv_w4a8_g_v4_n4.spv
var GEMVW4A8GroupedVec4N4 []byte

//go:embed gemv_w4a8_g_v4_n8.spv
var GEMVW4A8GroupedVec4N8 []byte

// The (VEC, NROWS) grid and the wave32 arm — the two probes IDEAS §1.10 left
// beside the corner below. It swept the row block at VEC=4 only, and its
// finding 4 says the two do not simply trade: the one build in the whole grid
// whose lanes are fully occupied (VEC=1 at down's K=640, 80 loads over 64
// lanes) is 1.05x *slower* than VEC=4, while NROWS=4 leaves 44 lanes idle and
// is 1.17x faster. So the row block is swept across the width here, at the
// width that reaches the bus (4), the one that fills the lanes (1) and the two
// that empty them further (8, 16). The wave32 build is the other half: §1.7
// only ever measured the wave size with one row per wave, and a 32-lane wave
// has half the idle lanes to amortize the same fixed cost over.

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=1 -DNROWS=4 -o gemv_w4a8_g_v1_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=8 -DNROWS=4 -o gemv_w4a8_g_v8_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=16 -DNROWS=4 -o gemv_w4a8_g_v16_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DNROWS=4 -DWAVE=32 -o gemv_w4a8_g_v4_n4_w32.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g_v1_n4.spv
var GEMVW4A8GroupedVec1N4 []byte

//go:embed gemv_w4a8_g_v8_n4.spv
var GEMVW4A8GroupedVec8N4 []byte

//go:embed gemv_w4a8_g_v16_n4.spv
var GEMVW4A8GroupedVec16N4 []byte

//go:embed gemv_w4a8_g_v4_n4_w32.spv
var GEMVW4A8GroupedVec4N4W32 []byte

// The corner: both blocks in one build (-DMROWS=m -DNROWS=n). The M block is a
// throughput lever (1.39-1.66x at 256 sequences in flight, 0.39x at one) and
// the N block a latency one (1.11-1.94x on down at every batch), and at a
// serving batch they were measured to pay for the same thing — the MALL
// serving re-read requests — so the cell where 256 sequences actually sit is
// the one neither single-axis sweep reaches.
//
// The widths are the ones the two engine rules name at t=256. §1.9's rule for
// M is "the block nearest the mean rows per expert, erring low", which is 4-8
// at 5.05 rows; §1.10's rule for N is NROWS >= 8*VEC*WAVE/K, which is 4 at
// down's K=640 and 1 at gate_up's K=2560. Both shapes run every build, so the
// rules are being tested and not just applied. NROWS=8 is not built against an
// M block: it is past the register knee on its own, and a cell is an
// accumulator plus a partial, so the corner's register cost is the product.

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=2 -DNROWS=2 -o gemv_w4a8_g_v4_m2_n2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=4 -DNROWS=2 -o gemv_w4a8_g_v4_m4_n2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=8 -DNROWS=2 -o gemv_w4a8_g_v4_m8_n2.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=2 -DNROWS=4 -o gemv_w4a8_g_v4_m2_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=4 -DNROWS=4 -o gemv_w4a8_g_v4_m4_n4.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g_v4_m2_n2.spv
var GEMVW4A8GroupedVec4M2N2 []byte

//go:embed gemv_w4a8_g_v4_m4_n2.spv
var GEMVW4A8GroupedVec4M4N2 []byte

//go:embed gemv_w4a8_g_v4_m8_n2.spv
var GEMVW4A8GroupedVec4M8N2 []byte

//go:embed gemv_w4a8_g_v4_m2_n4.spv
var GEMVW4A8GroupedVec4M2N4 []byte

//go:embed gemv_w4a8_g_v4_m4_n4.spv
var GEMVW4A8GroupedVec4M4N4 []byte

// The three corner cells §1.11 left unbuilt, and the reason each was left.
// (8, 4) was ruled out on registers before they were measured: §1.11 finding 6
// then read 83 VGPRs at (4, 4) and found no scratch anywhere in the family, so
// the ceiling it was avoiding is not there. (2, 8) and (4, 8) were ruled out
// by the same argument one axis over — NROWS=8 was called past the register
// knee on its own — and the row-block table has NROWS=8 as the *best*
// single-axis block on `down` at t=64 and t=256, which makes it the block an M
// block should be composed with rather than the one to stop before.
//
// They are the fixed-width half of the selection arm below: an engine that
// sizes its block from the routing histogram can only pick from what is built,
// and at t=256 routing hands `down` 5.05 rows per expert with a tail out to 12.

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=8 -DNROWS=4 -o gemv_w4a8_g_v4_m8_n4.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=2 -DNROWS=8 -o gemv_w4a8_g_v4_m2_n8.spv gemv_w4a8.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DVEC=4 -DMROWS=4 -DNROWS=8 -o gemv_w4a8_g_v4_m4_n8.spv gemv_w4a8.comp

//go:embed gemv_w4a8_g_v4_m8_n4.spv
var GEMVW4A8GroupedVec4M8N4 []byte

//go:embed gemv_w4a8_g_v4_m2_n8.spv
var GEMVW4A8GroupedVec4M2N8 []byte

//go:embed gemv_w4a8_g_v4_m4_n8.spv
var GEMVW4A8GroupedVec4M4N8 []byte

//go:generate glslc --target-env=vulkan1.2 -O -o gemv_subgroup_f32.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_F16 -o gemv_subgroup_f16.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q8 -o gemv_subgroup_q8.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DPRECISION_Q4 -o gemv_subgroup_q4.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -o gemv_subgroup_f32_w32.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DPRECISION_F16 -o gemv_subgroup_f16_w32.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DPRECISION_Q8 -o gemv_subgroup_q8_w32.spv gemv_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DPRECISION_Q4 -o gemv_subgroup_q4_w32.spv gemv_subgroup.comp

//go:embed gemv_subgroup_f32.spv
var GEMVSubgroupF32 []byte

//go:embed gemv_subgroup_f16.spv
var GEMVSubgroupF16 []byte

//go:embed gemv_subgroup_q8.spv
var GEMVSubgroupQ8 []byte

//go:embed gemv_subgroup_q4.spv
var GEMVSubgroupQ4 []byte

//go:embed gemv_subgroup_f32_w32.spv
var GEMVSubgroupF32W32 []byte

//go:embed gemv_subgroup_f16_w32.spv
var GEMVSubgroupF16W32 []byte

//go:embed gemv_subgroup_q8_w32.spv
var GEMVSubgroupQ8W32 []byte

//go:embed gemv_subgroup_q4_w32.spv
var GEMVSubgroupQ4W32 []byte

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
//   - hoisted K-tile loads (*_hka/_hkb/_hkab), IDEAS §5.1b mechanism 3: a
//     whole K-slab of one operand's fragment loads is issued before the slab's
//     first MMA, so the bytes of a single row a wave has outstanding go from
//     32 (one fragment) to 32*BK_TILES. The byte set, the MMA count, the tile
//     and the accumulator grid are unchanged; only the number of loads in
//     flight moves. Pairs with the existing *_k64 rows, which deepen the slab
//     *without* hoisting and measured as nothing, twice.
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

// The hoisted-K ladder (IDEAS §5.1b mechanism 3, aimed at §2.3's residue).
// BK_TILES is the rung: a 16x16 fp16 fragment holds 32 B of each row it
// touches, so 1/2/4/8 K-tiles in flight hold 32/64/128/256 B — and 256 B is
// exactly the gcd a +256 B pad leaves against the 4 KB channel rotation, i.e.
// the rung at which the coverage law predicts the penalty is gone.
//
// Hoisting is per operand because the register file, not the idea, sets how
// far the ladder goes. On wave64 an accumulator costs ~8 VGPRs and a fragment
// ~4, and RADV_DEBUG=shaderstats (via cmd/probe) prices every rung:
//
//   WM=WN=4  reg64      144 VGPR   hkab2  192   hka4/hkb4  252   hkab4  spills
//                                                            (140 VGPR, 3 KB scratch)
//   WM=WN=2  reg32_bt    48 VGPR   hkab2   84   hkab4      144   hkab8  spills (38)
//                                                            hka8/hkb8  192
//
// So the ladder is run twice and each arm stops where the registers do: to
// rung 4 at WM=WN=4 (AI 32, the shape §2.1 and §2.3 measured), where past
// rung 2 only one operand fits at a time — which is a bonus, since hka4 vs
// hkb4 attributes the effect to an operand rather than to the pair — and to
// rung 8 at WM=WN=2 (AI 16, four accumulators), single-operand at the top
// rung. The AI-16 arm is also where §2.1 measured the kernel pinned at 765
// GB/s — literally bandwidth-bound — which is where a bandwidth-coverage
// effect should show up most clearly if it is real.
//
// The transposed-B arm is the one under test: both its operands are
// K-contiguous gathers, so hoisting deepens both runs. The row-major arm is
// the control — its A loads deepen identically, but a row-major B fragment is
// gathered element by element along N, so hoisting B cannot deepen anything.

//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DB_COLMAJOR=1 -o gemm_wmma_reg32_bt.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=2 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg32_bt_hkab2.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg32_bt_hkab4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=8 -DHOIST_A=1 -o gemm_wmma_reg32_bt_hka8.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=8 -DHOIST_B=1 -o gemm_wmma_reg32_bt_hkb8.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg32_hkab4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=2 -DWN=2 -DBK_TILES=8 -DHOIST_A=1 -o gemm_wmma_reg32_hka8.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=2 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg64_bt_hkab2.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=4 -DHOIST_B=1 -o gemm_wmma_reg64_bt_hkb4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=4 -DHOIST_A=1 -o gemm_wmma_reg64_bt_hka4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWM=4 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -o gemm_wmma_reg64_hka4.spv gemm_wmma.comp

// The same ladder at wave32 (IDEAS §6.2). §2.7 ended on the 256-VGPR wave64
// budget — hoisting both operands at BK_TILES=4 spills — and predicted wave32
// would buy headroom back, on the grounds that RDNA's WMMA 16x16x16 is
// natively a wave32 shape. **RADV_DEBUG=shaderstats says the opposite, before
// any of these is run**: a 16x16 fragment spread over 32 lanes puts twice as
// many elements in each lane as over 64, so every variant costs *more* VGPRs
// per lane at wave32, not fewer:
//
//   variant             w64 VGPR / sg per SIMD   w32 VGPR / sg per SIMD
//   reg32_bt              48 / 32                  72 / 16
//   reg32_bt_hkab4       144 / 10                 168 /  9
//   reg32_bt_hka8        192 /  8                 192 /  8
//   reg64_bt             144 / 10                 192 /  8
//   reg64_bt_hkab2       192 /  8                 256 /  5
//   reg64_bt_hka4        252 /  6                 256 /  5, 62 spilled
//   reg64_bt_hkab4       spills (140 VGPR, 3 KB)  256 /  5, 324 spilled, 12 KB
//
// So wave32 does not extend the ladder; it shortens it — the rung that is the
// suite's best kernel at wave64 spills at wave32. What is left worth measuring
// is the other half of §6.2's hypothesis, which the register file does not
// speak to: shorter dependent-instruction latency and finer scheduling
// granularity, against half the lanes per instruction issued. These binaries
// are built so that question gets a number instead of an argument.
//
// -DWAVE=32 only changes `local_size_x`; the host must pin the pipeline to the
// same size with VK_EXT_subgroup_size_control, which bench/ops_gemm_wmma.go's
// `waveSize` field does. The wave64 binaries above are byte-identical to what
// they were before WAVE became a -D (verified with cmp).

//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=4 -DWN=4 -DB_COLMAJOR=1 -o gemm_wmma_reg64_bt_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=2 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg64_bt_hkab2_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=4 -DWN=4 -DB_COLMAJOR=1 -DBK_TILES=4 -DHOIST_A=1 -o gemm_wmma_reg64_bt_hka4_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=2 -DWN=2 -DB_COLMAJOR=1 -o gemm_wmma_reg32_bt_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_reg32_bt_hkab4_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=2 -DWN=2 -DB_COLMAJOR=1 -DBK_TILES=8 -DHOIST_A=1 -o gemm_wmma_reg32_bt_hka8_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -DWM=4 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -o gemm_wmma_reg64_hka4_w32.spv gemm_wmma.comp

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

//go:embed gemm_wmma_reg32_bt.spv
var GEMMWMMAReg32BT []byte

//go:embed gemm_wmma_reg32_bt_hkab2.spv
var GEMMWMMAReg32BTHKAB2 []byte

//go:embed gemm_wmma_reg32_bt_hkab4.spv
var GEMMWMMAReg32BTHKAB4 []byte

//go:embed gemm_wmma_reg32_bt_hka8.spv
var GEMMWMMAReg32BTHKA8 []byte

//go:embed gemm_wmma_reg32_bt_hkb8.spv
var GEMMWMMAReg32BTHKB8 []byte

//go:embed gemm_wmma_reg32_hkab4.spv
var GEMMWMMAReg32HKAB4 []byte

//go:embed gemm_wmma_reg32_hka8.spv
var GEMMWMMAReg32HKA8 []byte

//go:embed gemm_wmma_reg64_bt_hkab2.spv
var GEMMWMMAReg64BTHKAB2 []byte

//go:embed gemm_wmma_reg64_bt_hkb4.spv
var GEMMWMMAReg64BTHKB4 []byte

//go:embed gemm_wmma_reg64_bt_hka4.spv
var GEMMWMMAReg64BTHKA4 []byte

//go:embed gemm_wmma_reg64_hka4.spv
var GEMMWMMAReg64HKA4 []byte

//go:embed gemm_wmma_reg64_bt_w32.spv
var GEMMWMMAReg64BTW32 []byte

//go:embed gemm_wmma_reg64_bt_hkab2_w32.spv
var GEMMWMMAReg64BTHKAB2W32 []byte

//go:embed gemm_wmma_reg64_bt_hka4_w32.spv
var GEMMWMMAReg64BTHKA4W32 []byte

//go:embed gemm_wmma_reg32_bt_w32.spv
var GEMMWMMAReg32BTW32 []byte

//go:embed gemm_wmma_reg32_bt_hkab4_w32.spv
var GEMMWMMAReg32BTHKAB4W32 []byte

//go:embed gemm_wmma_reg32_bt_hka8_w32.spv
var GEMMWMMAReg32BTHKA8W32 []byte

//go:embed gemm_wmma_reg64_hka4_w32.spv
var GEMMWMMAReg64HKA4W32 []byte

// IDEAS §3.5, the grouped/MoE GEMM. Same source, same inner loop, same
// operand layout as the winners above; the only change is -DGROUPED=1, which
// replaces "derive the tile origin from gl_WorkGroupID" with "read it out of
// a table". That is what lets one dispatch cover every expert's tiles at
// once, against the 512 dispatches an expert-at-a-time loop needs, and it is
// why the two arms are directly comparable: they run the same binary.
//
// (Adding GROUPED shifted the SPIR-V ids in the non-grouped binaries — the
// preprocessor sees one more `#if` — but not a single instruction:
// `spirv-dis` output with ids renumbered is identical, and every file is the
// same size to the byte.)
//
// The geometries are the two §3.4 crowned, plus the ones only an MoE shape
// motivates. At prefill an expert sees ~40 rows, so the M tile is most of the
// question:
//
//   BM=64  40 rows -> one tile of 64      62% useful
//   BM=32  40 rows -> two tiles of 32     62% useful
//   BM=16  40 rows -> three tiles of 48   83% useful
//
// so BM=16 is the padding arm — it buys back a third of the waste and pays
// for it in arithmetic intensity (10.7-12.8 FLOP/byte against 16 and 32),
// which is the trade this family exists to price. All five hoist at
// BK_TILES=4, §2.7's rung, because that lever is about the memory system and
// nothing here changes the memory system.

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DB_COLMAJOR=1 -DWM=4 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -o gemm_wmma_moe_reg64_bt_hka4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DB_COLMAJOR=1 -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_moe_reg32_bt_hkab4.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DB_COLMAJOR=1 -DWAVE=32 -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_moe_reg32_bt_hkab4_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DB_COLMAJOR=1 -DWAVE=32 -DWM=1 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_moe_reg16x32_bt_hkab4_w32.spv gemm_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DB_COLMAJOR=1 -DWAVE=32 -DWM=1 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DHOIST_B=1 -o gemm_wmma_moe_reg16x64_bt_hkab4_w32.spv gemm_wmma.comp

//go:embed gemm_wmma_moe_reg64_bt_hka4.spv
var GEMMWMMAMoEReg64BTHKA4 []byte

//go:embed gemm_wmma_moe_reg32_bt_hkab4.spv
var GEMMWMMAMoEReg32BTHKAB4 []byte

//go:embed gemm_wmma_moe_reg32_bt_hkab4_w32.spv
var GEMMWMMAMoEReg32BTHKAB4W32 []byte

//go:embed gemm_wmma_moe_reg16x32_bt_hkab4_w32.spv
var GEMMWMMAMoEReg16x32BTHKAB4W32 []byte

//go:embed gemm_wmma_moe_reg16x64_bt_hkab4_w32.spv
var GEMMWMMAMoEReg16x64BTHKAB4W32 []byte

// IDEAS §2.2, the Q4 grouped GEMM: the same grouped schedule over the same
// tile table, with the expert bank held at 4 bits per weight and dequantized
// into LDS a K-slab at a time (shaders/gemm_wmma_q4.comp). §3.5 measured the
// fp16 grouped kernel at 93% of the ceiling its format implies and that
// ceiling is entirely weight bytes, so this is the only lever the MoE prefill
// has left.
//
// The geometry arm is the same five tiles §3.5 ran, so the two formats are
// comparable row for row — and so that §3.5's finding 2 (tile padding costs
// nothing, because FLOPs were free) can be re-tested where FLOPs are not
// expected to be free any more.
//
// The other three vary one thing each against the first of those:
//
//	qb128     one fp16 scale per 128 nibbles instead of 32, which takes the
//	          scales from 6.2% of the bank's bytes to 1.6%
//	ldspad0   no LDS row pad, so the B slab's rows are 128 B = one full
//	          rotation of the 32 LDS banks apart
//	db        the next K-slab's global loads issued before this slab's MMAs

//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -o gemm_wmma_q4_moe_reg64.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=2 -DWN=2 -o gemm_wmma_q4_moe_reg32.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWAVE=32 -DWM=2 -DWN=2 -o gemm_wmma_q4_moe_reg32_w32.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWAVE=32 -DWM=1 -DWN=2 -o gemm_wmma_q4_moe_reg16x32_w32.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWAVE=32 -DWM=1 -DWN=4 -o gemm_wmma_q4_moe_reg16x64_w32.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DQBLOCK=128 -o gemm_wmma_q4_moe_reg64_qb128.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DLDS_PAD=0 -o gemm_wmma_q4_moe_reg64_ldspad0.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DDOUBLE_BUFFER=1 -o gemm_wmma_q4_moe_reg64_db.spv gemm_wmma_q4.comp

//go:embed gemm_wmma_q4_moe_reg64.spv
var GEMMWMMAQ4MoEReg64 []byte

//go:embed gemm_wmma_q4_moe_reg32.spv
var GEMMWMMAQ4MoEReg32 []byte

//go:embed gemm_wmma_q4_moe_reg32_w32.spv
var GEMMWMMAQ4MoEReg32W32 []byte

//go:embed gemm_wmma_q4_moe_reg16x32_w32.spv
var GEMMWMMAQ4MoEReg16x32W32 []byte

//go:embed gemm_wmma_q4_moe_reg16x64_w32.spv
var GEMMWMMAQ4MoEReg16x64W32 []byte

//go:embed gemm_wmma_q4_moe_reg64_qb128.spv
var GEMMWMMAQ4MoEReg64QB128 []byte

//go:embed gemm_wmma_q4_moe_reg64_ldspad0.spv
var GEMMWMMAQ4MoEReg64LDSPad0 []byte

//go:embed gemm_wmma_q4_moe_reg64_db.spv
var GEMMWMMAQ4MoEReg64DB []byte

// The gather and combine passes the grouped GEMM cannot do for itself
// (IDEAS §3.5). TOPK is baked in because it sizes the combine's unrolled
// accumulation; 10 is qwen3.8-flash-next's num_experts_per_tok.

//go:generate glslc --target-env=vulkan1.2 -O -DMODE=0 -o moe_gather.spv moe_route.comp
//go:generate glslc --target-env=vulkan1.2 -O -DMODE=1 -DTOPK=10 -o moe_combine.spv moe_route.comp

//go:embed moe_gather.spv
var MoEGather []byte

//go:embed moe_combine.spv
var MoECombine []byte

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
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -o rmsnorm_subgroup_w32.spv rmsnorm_subgroup.comp

//go:embed rmsnorm_shared.spv
var RMSNormShared []byte

//go:embed rmsnorm_subgroup.spv
var RMSNormSubgroup []byte

//go:embed rmsnorm_subgroup_w32.spv
var RMSNormSubgroupW32 []byte

//go:generate glslc --target-env=vulkan1.2 -O -o softmax_shared.spv softmax_shared.comp
//go:generate glslc --target-env=vulkan1.2 -O -o softmax_subgroup.spv softmax_subgroup.comp
//go:generate glslc --target-env=vulkan1.2 -O -DWAVE=32 -o softmax_subgroup_w32.spv softmax_subgroup.comp

//go:embed softmax_shared.spv
var SoftmaxShared []byte

//go:embed softmax_subgroup.spv
var SoftmaxSubgroup []byte

// IDEAS §3.7 has the subgroup reductions measuring 3.3x *slower* than the
// shared-memory tree, which is backwards and unexplained. §6.2's wave32 arm is
// one of the two candidate causes worth ruling in or out cheaply: a
// 64-thread workgroup sweeping a row with subgroupAdd against a 32-thread one.

//go:embed softmax_subgroup_w32.spv
var SoftmaxSubgroupW32 []byte

// VAE decoder (PIPELINE.md stage 2b). Every one of these is built over the
// same four bindings and declares the same 88-byte push-constant block,
// defined in vae_common.glsl, so a whole decode can be recorded into one
// command buffer by vk.DispatchMultiTimed. All fp32: this is the correctness
// port, with zimage/vae's CPU implementation as its oracle.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_conv2d.spv vae_conv2d.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_groupnorm.spv vae_groupnorm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_silu.spv vae_silu.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_add.spv vae_add.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_upsample2x.spv vae_upsample2x.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_nchw_to_rows.spv vae_nchw_to_rows.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_rows_to_nchw_add.spv vae_rows_to_nchw_add.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_linear.spv vae_linear.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_attention.spv vae_attention.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_transpose.spv vae_transpose.comp

//go:embed vae_conv2d.spv
var VAEConv2D []byte

//go:embed vae_groupnorm.spv
var VAEGroupNorm []byte

//go:embed vae_silu.spv
var VAESiLU []byte

//go:embed vae_add.spv
var VAEAdd []byte

//go:embed vae_upsample2x.spv
var VAEUpsample2x []byte

//go:embed vae_nchw_to_rows.spv
var VAENCHWToRows []byte

//go:embed vae_rows_to_nchw_add.spv
var VAERowsToNCHWAdd []byte

//go:embed vae_linear.spv
var VAELinear []byte

//go:embed vae_attention.spv
var VAEAttention []byte

//go:embed vae_transpose.spv
var VAETranspose []byte

// The mid block on the matrix cores (PIPELINE.md stage 7). The scalar
// attention above is 26% of a 1024x1024 decode in one dispatch and its four
// projections another 10% at 62 GFLOP/s, so both move to fp16 operands and
// cooperative-matrix multiplies. The pack is what makes a fragment load
// cover its 512 bytes (§5.1b); TRANSPOSE=1 is the v operand, whose matmul
// reduces over the row rather than the component.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_narrow_f16.spv vae_narrow_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_pack_f16.spv vae_pack_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DTRANSPOSE=1 -o vae_pack_f16_t.spv vae_pack_f16.comp

//go:embed vae_narrow_f16.spv
var VAENarrowF16 []byte

//go:embed vae_pack_f16.spv
var VAEPackF16 []byte

//go:embed vae_pack_f16_t.spv
var VAEPackF16T []byte

// The attention ladder. QT query tiles per workgroup and KTIL keys tiles per
// block are the two knobs stage 3c's ablation ended on; what is new here is
// that the workgroup, not the wave, owns the head, so WAVES is fixed at 4 by
// the 512-wide head and only the arms that keep the register file under 256
// are built -- QT=2 with KTIL=4 spills 44 VGPRs into 11 KB of scratch, and so
// does KTIL=8 at wave32, where an accumulator tile costs 8 registers a lane
// rather than 4. The wave32 arms are §6.2's, and they win.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=2 -o vae_attn_wmma_qt1_kt2.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -o vae_attn_wmma_qt1_kt4.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=8 -o vae_attn_wmma_qt1_kt8.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=2 -o vae_attn_wmma_qt2_kt2.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=2 -DWAVE=32 -o vae_attn_wmma_qt1_kt2_w32.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DWAVE=32 -o vae_attn_wmma_qt1_kt4_w32.spv vae_attention_wmma.comp

//go:embed vae_attn_wmma_qt1_kt2.spv
var VAEAttentionWMMAQT1KT2 []byte

//go:embed vae_attn_wmma_qt1_kt4.spv
var VAEAttentionWMMAQT1KT4 []byte

//go:embed vae_attn_wmma_qt1_kt8.spv
var VAEAttentionWMMAQT1KT8 []byte

//go:embed vae_attn_wmma_qt2_kt2.spv
var VAEAttentionWMMAQT2KT2 []byte

//go:embed vae_attn_wmma_qt1_kt2_w32.spv
var VAEAttentionWMMAQT1KT2W32 []byte

//go:embed vae_attn_wmma_qt1_kt4_w32.spv
var VAEAttentionWMMAQT1KT4W32 []byte

// The two negative controls (zimage/vae/gpu_test.go). NO_CROSS_WAVE drops
// the three partial score matrices this kernel's structure exists to sum;
// NO_RESCALE drops the online-softmax correction. Built and dispatchable,
// deliberately out of the ladder.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DNO_CROSS_WAVE=1 -o vae_attn_wmma_nocross.spv vae_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DNO_RESCALE=1 -o vae_attn_wmma_norescale.spv vae_attention_wmma.comp

//go:embed vae_attn_wmma_nocross.spv
var VAEAttentionWMMANoCrossWave []byte

//go:embed vae_attn_wmma_norescale.spv
var VAEAttentionWMMANoRescale []byte

// conv2d on the matrix cores (PIPELINE.md stage 8). Stage 7's profiler left
// conv3x3 at 85% of a 1024x1024 decode, 3.0-3.2 TFLOP/s against a 55.5
// TFLOP/s ceiling, and it is the one operator in this pipeline whose implicit
// GEMM had never been written. The pack is what makes that GEMM's B operand
// a 512 B contiguous fragment load at an *unaligned* pixel offset, which is
// what a +-1 tap shift needs; vae_pack_conv.comp is the argument.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o vae_pack_conv.spv vae_pack_conv.comp

//go:embed vae_pack_conv.spv
var VAEPackConv []byte

// The conv ladder. BM x BN is the workgroup's output tile -- output channels
// by pixels -- and the two knobs that matter are different ones from the
// DiT's: BM sets how many times the activation is streamed (once per
// ceil(OC/BM)), BN how much of the filter slab a workgroup amortises. The
// wave32 arms are §6.2's, which has won four times.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -o vae_conv_wmma_64x64.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=2 -o vae_conv_wmma_64x64_k2.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_N=2 -o vae_conv_wmma_64x128.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=2 -o vae_conv_wmma_128x64.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -o vae_conv_wmma_128x128.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=4 -o vae_conv_wmma_256x64.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVE=32 -o vae_conv_wmma_64x64_w32.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVE=32 -o vae_conv_wmma_128x64_w32.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DWAVE=32 -o vae_conv_wmma_128x128_w32.spv vae_conv_wmma.comp

//go:embed vae_conv_wmma_64x64.spv
var VAEConvWMMA64x64 []byte

//go:embed vae_conv_wmma_64x64_k2.spv
var VAEConvWMMA64x64K2 []byte

//go:embed vae_conv_wmma_64x128.spv
var VAEConvWMMA64x128 []byte

//go:embed vae_conv_wmma_128x64.spv
var VAEConvWMMA128x64 []byte

//go:embed vae_conv_wmma_128x128.spv
var VAEConvWMMA128x128 []byte

//go:embed vae_conv_wmma_256x64.spv
var VAEConvWMMA256x64 []byte

//go:embed vae_conv_wmma_64x64_w32.spv
var VAEConvWMMA64x64W32 []byte

//go:embed vae_conv_wmma_128x64_w32.spv
var VAEConvWMMA128x64W32 []byte

//go:embed vae_conv_wmma_128x128_w32.spv
var VAEConvWMMA128x128W32 []byte

// The conv path's two negative controls (zimage/vae/gpu_conv_test.go).
// NO_TAP_SHIFT drops the horizontal tap offset, which is the one thing the
// blocked layout exists to make free; PAD_CLAMP replicates the edge pixel
// instead of zeroing the border, which is the padding bug a conv port
// actually has. Built and dispatchable, deliberately out of the ladder.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DWAVES_M=2 -DWAVES_N=2 -DNO_TAP_SHIFT=1 -o vae_conv_wmma_notapshift.spv vae_conv_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DPAD_CLAMP=1 -o vae_pack_conv_clamp.spv vae_pack_conv.comp

//go:embed vae_conv_wmma_notapshift.spv
var VAEConvWMMANoTapShift []byte

//go:embed vae_pack_conv_clamp.spv
var VAEPackConvClamp []byte

// Z-Image DiT (PIPELINE.md stage 3). Same two-arena binding convention as
// the VAE shaders, defined in dit_common.glsl. fp32; the CPU implementation
// in zimage/dit is the oracle.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_rmsnorm.spv dit_rmsnorm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_rope.spv dit_rope.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_transpose_k.spv dit_transpose_k.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_attention.spv dit_attention.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_attention_flash.spv dit_attention_flash.comp

//go:embed dit_rmsnorm.spv
var DiTRMSNorm []byte

//go:embed dit_rope.spv
var DiTRoPE []byte

//go:embed dit_transpose_k.spv
var DiTTransposeK []byte

//go:embed dit_attention.spv
var DiTAttention []byte

//go:embed dit_attention_flash.spv
var DiTAttentionFlash []byte

// Attention on the matrix cores (PIPELINE.md stage 3c, IDEAS §3.3). The
// ladder varies the two knobs that decided the GEMM ablation and nothing
// else: QT, the query tiles one wave accumulates, which is the whole of the
// kernel's arithmetic intensity at 16*QT FLOP/byte; and KTIL, the key tiles
// per block, which moves the softmax bookkeeping and the live score
// accumulators without moving intensity at all. Both size register and
// `shared` arrays, so they are -D and not specialization constants, exactly as
// in gemm_wmma.comp.
//
// The wave32 arms are IDEAS §6.2 applied here: the register file is per lane,
// so the same accumulator grid costs twice the VGPRs at wave32, and the GEMM
// found the crossover at M=1024 -- which the DiT straddles, 320 tokens for the
// caption stream and 4096 for a 1024x1024 latent. A variant built with -DWAVE
// must be paired with RequiredSubgroupSize on the host or the workgroup is no
// longer one wave and the LDS scratch is shared by two.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_pack_f16.spv dit_pack_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -o dit_attn_wmma_qt1_kt4.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=8 -o dit_attn_wmma_qt1_kt8.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=2 -o dit_attn_wmma_qt2_kt2.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=4 -o dit_attn_wmma_qt2_kt4.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=8 -o dit_attn_wmma_qt2_kt8.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=4 -DKTIL=4 -o dit_attn_wmma_qt4_kt4.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=4 -DNO_TAIL_MASK=1 -o dit_attn_wmma_qt2_kt4_nomask.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DWAVE=32 -o dit_attn_wmma_qt1_kt4_w32.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=4 -DWAVE=32 -o dit_attn_wmma_qt2_kt4_w32.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=2 -DWAVE=32 -o dit_attn_wmma_qt1_kt2_w32.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=8 -DWAVE=32 -o dit_attn_wmma_qt1_kt8_w32.spv dit_attention_wmma.comp

//go:embed dit_pack_f16.spv
var DiTPackF16 []byte

//go:embed dit_attn_wmma_qt1_kt4.spv
var DiTAttentionWMMAQT1KT4 []byte

//go:embed dit_attn_wmma_qt1_kt8.spv
var DiTAttentionWMMAQT1KT8 []byte

//go:embed dit_attn_wmma_qt2_kt2.spv
var DiTAttentionWMMAQT2KT2 []byte

//go:embed dit_attn_wmma_qt2_kt4.spv
var DiTAttentionWMMAQT2KT4 []byte

//go:embed dit_attn_wmma_qt2_kt8.spv
var DiTAttentionWMMAQT2KT8 []byte

//go:embed dit_attn_wmma_qt4_kt4.spv
var DiTAttentionWMMAQT4KT4 []byte

// The negative control: qt2_kt4 with the tail mask compiled out, and nothing
// else changed. zimage/dit/gpu_test.go asserts it fails at a sequence length
// that is not a multiple of the key block, which is what makes the masked
// kernel's agreement evidence rather than coincidence.

//go:embed dit_attn_wmma_qt2_kt4_nomask.spv
var DiTAttentionWMMAQT2KT4NoMask []byte

//go:embed dit_attn_wmma_qt1_kt4_w32.spv
var DiTAttentionWMMAQT1KT4W32 []byte

//go:embed dit_attn_wmma_qt2_kt4_w32.spv
var DiTAttentionWMMAQT2KT4W32 []byte

//go:embed dit_attn_wmma_qt1_kt2_w32.spv
var DiTAttentionWMMAQT1KT2W32 []byte

//go:embed dit_attn_wmma_qt1_kt8_w32.spv
var DiTAttentionWMMAQT1KT8W32 []byte

// PIPELINE.md stage 4, the DiT graph: the rest of the block, so that a whole
// transformer layer is a dispatch sequence rather than a kernel with host
// code around it. Four of the five are elementwise passes that exist to keep
// the GEMMs fed -- the modulation vector, the two narrowings into the fp16
// arena, the gated residual -- and the fifth is the GEMM itself.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_adaln.spv dit_adaln.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_scale_f16.spv dit_scale_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_swiglu_f16.spv dit_swiglu_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DIN_F16=1 -o dit_swiglu_f16_in16.spv dit_swiglu_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_gate_add.spv dit_gate_add.comp

// The fused tail (PIPELINE.md stage 4b). Stage 4a's graph kept every norm as
// its own dispatch so that each one is a tensor the diffusers dump can be
// compared against; once that has been done, the fp32 intermediate between a
// norm and its consumer is a DRAM round trip bought for nothing. These three
// are the same arithmetic in one pass each, and `GPUBlock` keeps both paths so
// the fused output is checked against the unfused one rather than only
// against the reference.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_norm_scale_f16.spv dit_norm_scale_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_norm_gate_add.spv dit_norm_gate_add.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_qk_pack.spv dit_qk_pack.comp

//go:embed dit_norm_scale_f16.spv
var DiTNormScaleF16 []byte

// The transformer's tail (PIPELINE.md stage 9). A LayerNorm -- the mean is
// subtracted, which nothing else in this model does -- times the final
// layer's adaLN scale, narrowed into the fp16 arena as a GEMM A operand. The
// NO_MEAN build is the control: it is an RMS norm, which is the mistake.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o dit_final_norm.spv dit_final_norm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNO_MEAN=1 -o dit_final_norm_nomean.spv dit_final_norm.comp

//go:embed dit_final_norm.spv
var DiTFinalNorm []byte

//go:embed dit_final_norm_nomean.spv
var DiTFinalNormNoMean []byte

//go:embed dit_norm_gate_add.spv
var DiTNormGateAdd []byte

//go:embed dit_qk_pack.spv
var DiTQKPack []byte

//go:embed dit_adaln.spv
var DiTAdaLN []byte

//go:embed dit_scale_f16.spv
var DiTScaleF16 []byte

//go:embed dit_swiglu_f16.spv
var DiTSwiGLUF16 []byte

//go:embed dit_swiglu_f16_in16.spv
var DiTSwiGLUF16In16 []byte

//go:embed dit_gate_add.spv
var DiTGateAdd []byte

// The projection GEMM's ladder. Two knobs, both read off results/shapes.csv
// rather than swept blind:
//
//   - the tile geometry, where the model's three shapes disagreed: the
//     single-wave 64x64 register-blocked tile with a hoisted K-slab (§2.7)
//     wins dit.qkv and dit.ff.w2 at M=4096, and the four-wave 128x256 tile
//     wins dit.ff.w13 by 1.66x.
//   - the B layout, which is the new question. A weight is uploaded once, so
//     storing it as 16x16 fragment tiles costs nothing per step and buys the
//     coverage (§5.1b) that stage 3c measured 30x of on attention's
//     activations. Each geometry is built against the layout it won with and
//     against the tiled one, which is the comparison.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=0 -o dit_gemm_reg64_hka4.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg64_hka4_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=4 -DB_LAYOUT=2 -o dit_gemm_reg64_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=1 -o dit_gemm_wg128x256.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -o dit_gemm_wg128x256_bt16.spv dit_gemm.comp

// The swizzle arms (IDEAS §2.4). Same kernel, same layouts, only the mapping
// from gl_WorkGroupID to a tile of C: bands of SWZ columns walked top to
// bottom instead of the grid's own row-major order.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=1 -DSWZ=4 -o dit_gemm_wg128x256_swz4.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=1 -DSWZ=8 -o dit_gemm_wg128x256_swz8.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=2 -o dit_gemm_wg128x256_bt16_swz2.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=4 -o dit_gemm_wg128x256_bt16_swz4.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=8 -o dit_gemm_wg128x256_bt16_swz8.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=16 -o dit_gemm_wg128x256_bt16_swz16.spv dit_gemm.comp

//go:embed dit_gemm_wg128x256_swz4.spv
var DiTGEMMWG128x256SWZ4 []byte

//go:embed dit_gemm_wg128x256_swz8.spv
var DiTGEMMWG128x256SWZ8 []byte

//go:embed dit_gemm_wg128x256_bt16_swz2.spv
var DiTGEMMWG128x256TiledSWZ2 []byte

//go:embed dit_gemm_wg128x256_bt16_swz4.spv
var DiTGEMMWG128x256TiledSWZ4 []byte

//go:embed dit_gemm_wg128x256_bt16_swz8.spv
var DiTGEMMWG128x256TiledSWZ8 []byte

//go:embed dit_gemm_wg128x256_bt16_swz16.spv
var DiTGEMMWG128x256TiledSWZ16 []byte

// The fp16-C companion of the default kernel (§2.6). Not a ladder rung: it is
// only correct where the consumer reads fp16 and the values fit, which in this
// block is the FFN's gate and up projections and nothing else, so GPUBlock
// reaches it through a companion table rather than through a plan.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=8 -DC_F16=1 -o dit_gemm_wg128x256_bt16_swz8_cf16.spv dit_gemm.comp

//go:embed dit_gemm_wg128x256_bt16_swz8_cf16.spv
var DiTGEMMWG128x256TiledSWZ8CF16 []byte

//go:embed dit_gemm_reg64_hka4.spv
var DiTGEMMReg64HKA4 []byte

//go:embed dit_gemm_reg64_hka4_bt16.spv
var DiTGEMMReg64HKA4Tiled []byte

//go:embed dit_gemm_reg64_bt16.spv
var DiTGEMMReg64Tiled []byte

//go:embed dit_gemm_wg128x256.spv
var DiTGEMMWG128x256 []byte

//go:embed dit_gemm_wg128x256_bt16.spv
var DiTGEMMWG128x256Tiled []byte

// PIPELINE.md stage 5c, the text encoder. Three things it needs and the DiT
// did not, all of them builds of shaders the DiT already has:
//
//   - **Narrow-M rungs of the projection GEMM.** The DiT runs at M=4096 and
//     its winning tile is 128 rows deep; a prompt is 8-512 tokens, so that
//     tile would fill 24 of its 128 rows at a short prompt *and* leave the
//     whole q projection to 16 workgroups, on a part with 40 CUs. These are
//     the same kernel at 16 and 32 rows, where the grid is deep enough to
//     fill the machine. All are against the fragment-tiled weight (§2.8),
//     which is free on a weight and is what makes a fragment load covered.
//   - **A causal, grouped-query build of the attention kernel.** Two
//     compile-time flags on dit_attention_wmma.comp, so the DiT's own builds
//     take neither: their SPIR-V comes out instruction for instruction the
//     same afterwards, with only the SSA ids shifted by the two dead
//     assignments in the `#else` arms (checked with `spirv-dis` and the ids
//     normalised).
//   - **NeoX rotary**, which is its own file (qwen_rope.comp) because the
//     convention is a property of the checkpoint and not a knob.
//
// The GEMM keeps its `dit_` prefix because it is the DiT's kernel; what the
// encoder adds is geometry, not a shader.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg16x64_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=16 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg16x256_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg32x64_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=8 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg32x128_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=8 -DWAVES_N=2 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_wg16x256_bt16.spv dit_gemm.comp

//go:embed dit_gemm_reg16x64_bt16.spv
var DiTGEMMReg16x64Tiled []byte

//go:embed dit_gemm_reg16x256_bt16.spv
var DiTGEMMReg16x256Tiled []byte

//go:embed dit_gemm_reg32x64_bt16.spv
var DiTGEMMReg32x64Tiled []byte

//go:embed dit_gemm_reg32x128_bt16.spv
var DiTGEMMReg32x128Tiled []byte

//go:embed dit_gemm_wg16x256_bt16.spv
var DiTGEMMWG16x256Tiled []byte

// Two more rungs, aimed at the one thing the first sweep left unexplained:
// at 24 tokens the best rung reaches 124 GB/s of a 236 GB/s bus while doing
// nothing but streaming a weight it never reuses, which is the signature of
// too few loads in flight rather than of a tile being the wrong shape. Both
// of these buy outstanding requests per wave without changing the tile's
// memory traffic -- wider in N (eight B fragments per K step instead of
// four) and deeper in K (an eight-tile slab instead of four).

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=8 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg16x128_bt16.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=4 -DBK_TILES=8 -DHOIST_A=1 -DB_LAYOUT=2 -o dit_gemm_reg16x64_bt16_k8.spv dit_gemm.comp

//go:embed dit_gemm_reg16x128_bt16.spv
var DiTGEMMReg16x128Tiled []byte

//go:embed dit_gemm_reg16x64_bt16_k8.spv
var DiTGEMMReg16x64TiledK8 []byte

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DCAUSAL=1 -DGQA=1 -o qwen_attn_wmma_qt1_kt4.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=4 -DCAUSAL=1 -DGQA=1 -o qwen_attn_wmma_qt2_kt4.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DWAVE=32 -DCAUSAL=1 -DGQA=1 -o qwen_attn_wmma_qt1_kt4_w32.spv dit_attention_wmma.comp

//go:embed qwen_attn_wmma_qt1_kt4.spv
var QwenAttentionQT1KT4 []byte

//go:embed qwen_attn_wmma_qt2_kt4.spv
var QwenAttentionQT2KT4 []byte

//go:embed qwen_attn_wmma_qt1_kt4_w32.spv
var QwenAttentionQT1KT4W32 []byte

// The same kernel with the causal mask compiled out, i.e. every token seeing
// the whole prompt. Built and dispatchable but deliberately left out of the
// ladder: its only caller is the negative control in zimage/qwen/gpu_test.go,
// which is the counterpart of dit_attn_wmma_qt2_kt4_nomask.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DGQA=1 -o qwen_attn_wmma_qt1_kt4_nocausal.spv dit_attention_wmma.comp

//go:embed qwen_attn_wmma_qt1_kt4_nocausal.spv
var QwenAttentionQT1KT4NoCausal []byte

//go:generate glslc --target-env=vulkan1.2 -O -I. -o qwen_rope.spv qwen_rope.comp

//go:embed qwen_rope.spv
var QwenRoPE []byte
