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

// Weight-bank gather probe (LLM.md L0a): a fixed number of bytes read as
// slabs drawn from a bank of tens of gigabytes, to find out whether the DRAM
// bus survives a working set 20x larger than anything else here measures.
// Nothing about the arithmetic matters; the address range does.

//go:generate glslc --target-env=vulkan1.2 -O -o bank_gather.spv bank_gather.comp

//go:embed bank_gather.spv
var BankGather []byte

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

// LLM.md L0c: the scale plane's layout, the mechanism test §2.2 finding 4
// asked for. Two layouts that make one staging step's BN scales contiguous
// (k-major, and tile-blocked), each at both scale-block sizes, so the 1.24x
// QBLOCK=128 win can be read against the layout that should remove it.
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DSCALE_LAYOUT=1 -o gemm_wmma_q4_moe_reg64_smk.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DSCALE_LAYOUT=1 -DQBLOCK=128 -o gemm_wmma_q4_moe_reg64_smk_qb128.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DSCALE_LAYOUT=2 -o gemm_wmma_q4_moe_reg64_smt.spv gemm_wmma_q4.comp
//go:generate glslc --target-env=vulkan1.2 -O -DGROUPED=1 -DBK_TILES=4 -DHOIST_A=1 -DWM=4 -DWN=4 -DSCALE_LAYOUT=2 -DQBLOCK=128 -o gemm_wmma_q4_moe_reg64_smt_qb128.spv gemm_wmma_q4.comp

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

//go:embed gemm_wmma_q4_moe_reg64_smk.spv
var GEMMWMMAQ4MoEReg64SMK []byte

//go:embed gemm_wmma_q4_moe_reg64_smk_qb128.spv
var GEMMWMMAQ4MoEReg64SMKQB128 []byte

//go:embed gemm_wmma_q4_moe_reg64_smt.spv
var GEMMWMMAQ4MoEReg64SMT []byte

//go:embed gemm_wmma_q4_moe_reg64_smt_qb128.spv
var GEMMWMMAQ4MoEReg64SMTQB128 []byte

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

// The fp16-context build of the winning variant (PIPELINE.md stage 10). Same
// tiling, same wave size; the epilogue writes the output projection's A
// operand directly instead of an fp32 context a second dispatch then narrows.
// Like the fp16-C GEMM it is a companion rather than a ladder rung: nothing
// sweeps it, GPUStack reaches it through a table.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DWAVE=32 -DOUT_F16=1 -o dit_attn_wmma_qt1_kt4_w32_of16.spv dit_attention_wmma.comp

//go:embed dit_pack_f16.spv
var DiTPackF16 []byte

//go:embed dit_attn_wmma_qt1_kt4_w32_of16.spv
var DiTAttentionWMMAQT1KT4W32OutF16 []byte

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

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=8 -DWAVES_M=2 -DWAVES_N=2 -DB_LAYOUT=2 -DSWZ=8 -DC_PACK=1 -o dit_gemm_wg128x256_bt16_swz8_cpack.spv dit_gemm.comp

//go:embed dit_gemm_wg128x256_bt16_swz8_cf16.spv
var DiTGEMMWG128x256TiledSWZ8CF16 []byte

// The fragment-tile-C companion (PIPELINE.md stage 10), for the one
// projection whose consumer reads fragment tiles: v, feeding attention.

//go:embed dit_gemm_wg128x256_bt16_swz8_cpack.spv
var DiTGEMMWG128x256TiledSWZ8CPack []byte

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

// SPEECH.md S6, the parakeet encoder. The FastConformer is the same GEMM
// ladder the DiT and the text encoder run -- two feed forwards, four
// projections, two pointwise convolutions and the relative-position
// projection, all [T, 1024] against [1024, N] -- so what this section adds is
// the five things the ladder does not have, and one geometry it does.
//
//   - **A LayerNorm with a mean and an affine** (parakeet_layernorm.comp).
//     Five per layer, and the only norm the model has.
//   - **The Transformer-XL score bias**: the position term, its shift
//     (parakeet_relshift.comp) and the additive bias that folds it into the
//     score kernel (dit_attention_wmma.comp's REL_BIAS build).
//   - **The convolution branch**: a GLU over the channel axis
//     (parakeet_glu.comp) and a depthwise convolution over time with the
//     folded BatchNorm and silu (parakeet_dwconv_f16.comp).
//   - **silu on its own** (parakeet_silu_f16.comp), which is the conformer's
//     feed forward where the DiT's is gated.
//   - **A scalar-weighted residual** (parakeet_residual.comp): the macaron
//     half-step is 0.5, not a per-channel gate.
//
// and the geometry is the **wave32 narrow-M GEMM**. An 11 s clip is 138
// frames, where results/shapes.csv measures the 32x32 wave32 tile at 23.1
// TFLOP/s against 12.7 for the wave64 rungs that win at M=1024 -- nearly 2x,
// on the same weights, for the same reason the attention kernel took 1.40x
// from the same lever (§6.2). dit_gemm.comp already takes -DWAVE, so these
// are builds and not a kernel.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_layernorm.spv parakeet_layernorm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DOUT_F16=1 -o parakeet_layernorm_f16.spv parakeet_layernorm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DOUT_F16=1 -DNO_MEAN=1 -o parakeet_layernorm_f16_nomean.spv parakeet_layernorm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_silu_f16.spv parakeet_silu_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_residual.spv parakeet_residual.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_glu.spv parakeet_glu.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_dwconv_f16.spv parakeet_dwconv_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_narrow_f16.spv parakeet_narrow_f16.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_relshift.spv parakeet_relshift.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DSHIFT_SLICE=1 -o parakeet_relshift_slice.spv parakeet_relshift.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DBIAS=1 -o parakeet_pack_bias.spv dit_pack_f16.comp

//go:embed parakeet_layernorm.spv
var ParakeetLayerNorm []byte

//go:embed parakeet_layernorm_f16.spv
var ParakeetLayerNormF16 []byte

//go:embed parakeet_layernorm_f16_nomean.spv
var ParakeetLayerNormF16NoMean []byte

//go:embed parakeet_silu_f16.spv
var ParakeetSiLUF16 []byte

//go:embed parakeet_residual.spv
var ParakeetResidual []byte

//go:embed parakeet_glu.spv
var ParakeetGLU []byte

//go:embed parakeet_dwconv_f16.spv
var ParakeetDWConvF16 []byte

//go:embed parakeet_narrow_f16.spv
var ParakeetNarrowF16 []byte

//go:embed parakeet_relshift.spv
var ParakeetRelShift []byte

//go:embed parakeet_relshift_slice.spv
var ParakeetRelShiftSlice []byte

//go:embed parakeet_pack_bias.spv
var ParakeetPackBias []byte

// SPEECH.md S7, the subsampling stack. Four kernels and three GEMMs replace
// the 92 ms the host was spending on 2.8 GFLOP -- 30 GFLOP/s, on a part that
// had just done 177 GFLOP in 13.8 ms.
//
// What the ladder did not have is a stride-2 conv2d over a *single-channel*
// input and the depthwise form of it. What it did have, once the feature map
// is held channel-last, is everything else: the two 1x1 convolutions are
// [P, 256] x [256, 256] GEMMs over positions and the linear is the same
// [T, 4096] x [4096, 1024] it always was. The layout is the whole design and
// it is argued in parakeet_sub_conv0.comp.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_sub_conv0.spv parakeet_sub_conv0.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_sub_dw.spv parakeet_sub_dw.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_sub_bias.spv parakeet_sub_bias.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_sub_flatten_f16.spv parakeet_sub_flatten_f16.comp

//go:embed parakeet_sub_conv0.spv
var ParakeetSubConv0 []byte

//go:embed parakeet_sub_dw.spv
var ParakeetSubDW []byte

//go:embed parakeet_sub_bias.spv
var ParakeetSubBias []byte

//go:embed parakeet_sub_flatten_f16.spv
var ParakeetSubFlattenF16 []byte

// SPEECH.md S8, the transducer tail: the encoder projector, the prediction
// network, the joint and the TDT greedy loop.
//
// This is the first thing in the engine that is *latency* bound rather than
// throughput bound. Greedy decoding is sequential by construction -- the
// prediction network advances on the token it just emitted -- so the whole
// tail runs at M = 1, once per emitted symbol, and what it costs is round
// trips and weight streaming rather than arithmetic: 24 MFLOP per emission
// against 18 MB of weights read to do it.
//
// So there is no new GEMM here. The four projections (both LSTM layers, the
// prediction projector and the 8198-wide head) run on the same dit_gemm.comp
// rungs everything else does, with M padded from 1 up to the narrowest tile
// in the ladder -- which costs arithmetic the part has spare and no bandwidth
// at all, since a GEMM reads its weight once whatever M is. What the four
// kernels below add is the state machine around them.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_lstm_in.spv parakeet_lstm_in.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_lstm_gate.spv parakeet_lstm_gate.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_joint_sum.spv parakeet_joint_sum.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o parakeet_argmax.spv parakeet_argmax.comp

//go:embed parakeet_lstm_in.spv
var ParakeetLSTMIn []byte

//go:embed parakeet_lstm_gate.spv
var ParakeetLSTMGate []byte

//go:embed parakeet_joint_sum.spv
var ParakeetJointSum []byte

//go:embed parakeet_argmax.spv
var ParakeetArgmax []byte

// The score kernel with the position bias compiled in, at the two wave sizes.
// Same ladder position as everywhere else -- QT=1, KTIL=4 -- since stage 3c's
// table says the geometry is decided by the register file and the wave size
// and not by the sequence, and an 11 s clip's 138 frames make attention 2% of
// the encoder's arithmetic either way.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DREL_BIAS=1 -DOUT_F16=1 -DWAVE=32 -o parakeet_attn_wmma_qt1_kt4_w32.spv dit_attention_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -DREL_BIAS=1 -DOUT_F16=1 -o parakeet_attn_wmma_qt1_kt4.spv dit_attention_wmma.comp

//go:embed parakeet_attn_wmma_qt1_kt4_w32.spv
var ParakeetAttentionQT1KT4W32 []byte

//go:embed parakeet_attn_wmma_qt1_kt4.spv
var ParakeetAttentionQT1KT4 []byte

// The wave32 rungs of the projection GEMM, at the two narrow tiles
// results/shapes.csv puts at the top for a clip-length M.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DWAVE=32 -o dit_gemm_reg32x32_bt16_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DWAVE=32 -o dit_gemm_reg32x64_bt16_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=1 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DWAVE=32 -o dit_gemm_reg16x64_bt16_w32.spv dit_gemm.comp

//go:embed dit_gemm_reg32x32_bt16_w32.spv
var DiTGEMMReg32x32TiledW32 []byte

//go:embed dit_gemm_reg32x64_bt16_w32.spv
var DiTGEMMReg32x64TiledW32 []byte

//go:embed dit_gemm_reg16x64_bt16_w32.spv
var DiTGEMMReg16x64TiledW32 []byte

// The position term's own GEMM is the one in the encoder whose B operand is
// an activation rather than a weight -- rel_k, [2T-1, 1024], different every
// clip -- so it cannot be staged as fragment tiles and reads the natural
// [N, ldb] layout instead. That is B_LAYOUT=0, i.e. dit_gemm_reg64_hka4.spv,
// which the DiT already builds; what is new is a narrow-M rung of it, since
// this GEMM's M is the clip length like every other here.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=0 -o dit_gemm_reg32x64_hka4.spv dit_gemm.comp

//go:embed dit_gemm_reg32x64_hka4.spv
var DiTGEMMReg32x64HKA4 []byte

// Kokoro's vocoder (SPEECH.md T4). The generator is eight AdaIN residual
// blocks and 97% of the vocoder's 160 GFLOP, and every convolution in them is
// a k-tap filter over a channel-last [T, C] activation at C = 128 or 256 with
// a dilation of 1, 3 or 5. A_CONV=1 makes those the same GEMM as everything
// else: the im2col is an addressing rule on the A operand and the padding is
// a zero border in the arena, so a convolution costs one dispatch and no
// packing pass. See dit_gemm.comp's A_CONV block.
//
// Four rungs, the same ladder shape the parakeet encoder sweeps, because the
// vocoder's M is 2600 or 15601 and its N is 128 or 256 -- a much taller and
// narrower shape than anything measured so far.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=1 -DWAVE=32 -o kokoro_conv_reg32x32_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=1 -DWAVE=32 -o kokoro_conv_reg32x64_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=4 -DB_LAYOUT=2 -DA_CONV=1 -o kokoro_conv_reg64.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=8 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=1 -o kokoro_conv_reg32x128.spv dit_gemm.comp

//go:embed kokoro_conv_reg32x32_w32.spv
var KokoroConvReg32x32W32 []byte

//go:embed kokoro_conv_reg32x64_w32.spv
var KokoroConvReg32x64W32 []byte

//go:embed kokoro_conv_reg64.spv
var KokoroConvReg64 []byte

//go:embed kokoro_conv_reg32x128.spv
var KokoroConvReg32x128 []byte

// The AdaIN residual block's scalar passes. AdaIN normalises each channel
// over *time*, so its reduction runs down a column of a channel-last tensor —
// two dispatches (partials, then the per-channel affine) rather than one,
// which is what makes it parallel over a 15601-frame tensor. Applying it,
// the Snake activation and the narrow into the convolution's fp16 arena are
// then one fused pass.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_stats.spv kokoro_stats.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_affine.spv kokoro_affine.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_act.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DSNAKE=1 -o kokoro_act_snake.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLEAKY=1 -DNO_AFFINE=1 -o kokoro_act_leaky.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_upadd.spv kokoro_upadd.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DPAD=1 -o kokoro_upadd_pad.spv kokoro_upadd.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_residual.spv kokoro_residual.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DCOPY=1 -o kokoro_copy.spv kokoro_residual.comp

//go:embed kokoro_stats.spv
var KokoroStats []byte

//go:embed kokoro_affine.spv
var KokoroAffine []byte

//go:embed kokoro_act.spv
var KokoroAct []byte

//go:embed kokoro_act_snake.spv
var KokoroActSnake []byte

//go:embed kokoro_act_leaky.spv
var KokoroActLeaky []byte

//go:embed kokoro_upadd.spv
var KokoroUpAdd []byte

//go:embed kokoro_upadd_pad.spv
var KokoroUpAddPad []byte

//go:embed kokoro_residual.spv
var KokoroResidual []byte

//go:embed kokoro_copy.spv
var KokoroCopy []byte

// The decoder — upstream's `Decoder`, the four AdaIN residual blocks between
// the length regulator and the generator (SPEECH.md T4c). It is 54% of the
// vocoder's time on 5.5% of its arithmetic, and the only reason it was still
// on the host is its channel counts: 514, 1090, 1024 and 512, two of which
// are not powers of two, so A_CONV=1's shift and mask do not apply.
//
// A_CONV=2 is the same convolution with the tap index carried across the K
// loop instead of divided out, over a row stride the host pads to a multiple
// of BK. Four rungs again, because the shape is the opposite of the
// generator's — M of 130 or 260 against N of 512 or 1024, where the generator
// is M in the thousands by N of 128.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=2 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=2 -DWAVE=32 -o kokoro_dconv_reg32x32_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=4 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=2 -DWAVE=32 -o kokoro_dconv_reg32x64_w32.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=4 -DWN=4 -DBK_TILES=4 -DB_LAYOUT=2 -DA_CONV=2 -o kokoro_dconv_reg64.spv dit_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DWM=2 -DWN=8 -DBK_TILES=4 -DHOIST_A=1 -DB_LAYOUT=2 -DA_CONV=2 -o kokoro_dconv_reg32x128.spv dit_gemm.comp

//go:embed kokoro_dconv_reg32x32_w32.spv
var KokoroDConvReg32x32W32 []byte

//go:embed kokoro_dconv_reg32x64_w32.spv
var KokoroDConvReg32x64W32 []byte

//go:embed kokoro_dconv_reg64.spv
var KokoroDConvReg64 []byte

//go:embed kokoro_dconv_reg32x128.spv
var KokoroDConvReg32x128 []byte

// The decoder's scalar passes. NPOT swaps the shift-and-mask index
// decomposition for a 2-D grid, which costs nothing and works at any channel
// count; the three activation builds are the three places a [T, C] fp32
// tensor is narrowed into a convolution's A operand — after a normalisation,
// before a shortcut, and before the one shortcut that upsamples.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -o kokoro_dstats.spv kokoro_stats.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -DLEAKY=1 -o kokoro_dact.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -DNO_AFFINE=1 -o kokoro_dnarrow.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -DNO_AFFINE=1 -DREPEAT=1 -o kokoro_dnarrow_rep.spv kokoro_act.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_shortcut.spv kokoro_shortcut.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_pool.spv kokoro_pool.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_proj.spv kokoro_proj.comp

//go:embed kokoro_dstats.spv
var KokoroDStats []byte

//go:embed kokoro_dact.spv
var KokoroDAct []byte

//go:embed kokoro_dnarrow.spv
var KokoroDNarrow []byte

//go:embed kokoro_dnarrow_rep.spv
var KokoroDNarrowRepeat []byte

//go:embed kokoro_shortcut.spv
var KokoroShortcut []byte

//go:embed kokoro_pool.spv
var KokoroPool []byte

// The prosody predictor's two 1-wide projections (SPEECH.md T6b). N = 1 is
// the one output shape the ladder cannot express, a fragment tile being 16
// columns wide, so it is a scalar reduction instead — one workgroup a frame.

//go:embed kokoro_proj.spv
var KokoroProj []byte

// The six bidirectional LSTMs of kokoro's phoneme side (SPEECH.md T6c), which
// after T6a and T6b are 96% of it. One dispatch a timestep, both directions,
// with the input projections lifted out of the loop and done once as a GEMM —
// so what is left in the sequential part is `W_hh`, [H, 4H], half a megabyte
// that stays MALL-resident for every step.

// Two knobs, and they are not equally important. CELLS is how many cells a
// workgroup owns — 256/CELLS bands a direction, so 2*256/CELLS workgroups on
// a step — and UNROLL is how many taps of the reduction one thread has in
// flight.
//
// A step is 0.5 MB of weights against 0.26 MFLOP, which looks like a
// bandwidth problem and is not: spreading it from two workgroups to sixteen
// moved it by nothing at all, and deepening the unroll from 4 to 32 moved it
// from 12.0 microseconds to 3.7 (TestGPULSTMLadder). What the kernel is short
// of is *outstanding loads*, and a recurrence has only 4H threads to issue
// them from. The `u4` build is kept so that finding stays measurable.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DCELLS=64 -DUNROLL=4 -o kokoro_lstm_c64u4.spv kokoro_lstm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DCELLS=128 -DUNROLL=32 -o kokoro_lstm_c128.spv kokoro_lstm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DCELLS=64 -DUNROLL=32 -o kokoro_lstm_c64.spv kokoro_lstm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DCELLS=32 -DUNROLL=32 -o kokoro_lstm_c32.spv kokoro_lstm.comp

// The rest of the phoneme side (SPEECH.md T6d): the length regulator as a
// gather, and the leaky rectifier that feeds the text encoder's convolutions
// — an NPOT build of the narrow every other stage already uses.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_gather.spv kokoro_gather.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -DNO_AFFINE=1 -DLEAKY=1 -o kokoro_dleaky.spv kokoro_act.comp

//go:embed kokoro_gather.spv
var KokoroGather []byte

//go:embed kokoro_dleaky.spv
var KokoroDLeaky []byte

//go:embed kokoro_lstm_c64u4.spv
var KokoroLSTMC64U4 []byte

//go:embed kokoro_lstm_c128.spv
var KokoroLSTMC128 []byte

//go:embed kokoro_lstm_c64.spv
var KokoroLSTMC64 []byte

//go:embed kokoro_lstm_c32.spv
var KokoroLSTMC32 []byte

// The vocoder's tail: `conv_post`, the two nonlinearities, and the inverse
// transform (SPEECH.md T4c). Moving these onto the device is worth more than
// the arithmetic in them — 78000 samples come back where 8 MB of [15601, 128]
// used to, and a device-local host-visible buffer reads at 210 MB/s.
//
// conv_post's N is 22, which is not a multiple of the 16-wide fragment tile;
// the host pads it to 32 with zero weight columns and the epilogue reads the
// 22 that mean anything.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_post.spv kokoro_post.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_istft.spv kokoro_istft.comp

//go:embed kokoro_post.spv
var KokoroPost []byte

//go:embed kokoro_istft.spv
var KokoroISTFT []byte

// PL-BERT, the phoneme encoder (SPEECH.md T6). Twelve layers sharing one
// weight group, 768 wide over fifty tokens — 6.7 GFLOP that the CPU
// reference takes 104 ms over, which is 47% of the phoneme side and 40% of a
// whole utterance.
//
// Almost all of it runs on kernels that already exist: the projections and
// the feed-forward are `dit_gemm.comp`'s narrow-M rungs, the post-norms are
// parakeet's LayerNorm (a mean *and* an affine, which is what this model's
// 1e-12 norm needs), and the residual adds are parakeet's. What is new is the
// attention, which at 7.7 MFLOP a layer is not worth a matrix-core kernel,
// and the feed-forward's activation.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o kokoro_bert_attn.spv kokoro_bert_attn.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNPOT=1 -DNO_AFFINE=1 -DGELU=1 -o kokoro_gelu.spv kokoro_act.comp

//go:embed kokoro_bert_attn.spv
var KokoroBertAttn []byte

//go:embed kokoro_gelu.spv
var KokoroGELU []byte

// The hyper-connection block (LLM.md L2), qwen3.8-flash-next's replacement
// for the residual stream — and, by L2a's per-op attribution of llama.cpp's
// own graph, over half of its prefill: 30.6% elementwise glue, 12.2% tiny-N
// F32 matmul and ~7% for the low-rank pair, against 1.6% for the gated
// DeltaNet that three quarters of the layers are made of.
//
// Four dispatches where the reference has about sixteen. The norm writes the
// fp16 A operand both matmuls read; the down projection carries `inject`'s
// four output columns (the reference's worst single line, 10.3% of prefill at
// 31.8 GFLOP/s) and applies silu on the accumulator; the up projection's
// epilogue is the sigmoid and the 4-branch collapse, so the [10240, T] gate —
// 21 MB at 512 tokens, the whole MALL — is never written; and the combine is
// one pass over the residual instead of five.
//
// Both GEMM rungs are M-blocked ladders over the same shader. The down
// projection's N is 336 = 21 tiles, so its BN is 48 (WN=3) and the ladder
// moves BM alone; the up projection's collapse needs the four streams of a
// feature in one workgroup, so its BN is 64 (WN=4) and the host permutes the
// weight's rows to match.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_hc_norm.spv llm_hc_norm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_hc_combine.spv llm_hc_combine.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_hc_cn.spv llm_hc_cn.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=1 -DWN=3 -o llm_hc_down_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=2 -DWN=3 -o llm_hc_down_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=4 -DWN=3 -o llm_hc_down_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=1 -DWN=4 -o llm_hc_up_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=2 -DWN=4 -o llm_hc_up_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=4 -DWN=4 -o llm_hc_up_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=8 -DWN=3 -o llm_hc_down_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=8 -DWN=4 -o llm_hc_up_m8.spv llm_gemm.comp

// The same six rungs over L8's dense bank (LLM.md L8b). The hyper-connection
// block is the last dense family staged as halves — 1.31 GB a token, 13.3% of
// a decode step — and it is last because neither of its modes is the plain
// arm: MODE 0 carries `inject` in its last tile, four F32 rows that do not
// begin on a column block, and MODE 1 collapses the gate through LDS the
// unpack now sits beside. Both are the same two flags the plain arm takes.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=1 -DWN=3 -DQ8B -DDENSE_Q8 -o llm_hc_down_q8_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=2 -DWN=3 -DQ8B -DDENSE_Q8 -o llm_hc_down_q8_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=4 -DWN=3 -DQ8B -DDENSE_Q8 -o llm_hc_down_q8_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=1 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_hc_up_q8_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=2 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_hc_up_q8_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=4 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_hc_up_q8_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=8 -DWN=3 -DQ8B -DDENSE_Q8 -o llm_hc_down_q8_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=8 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_hc_up_q8_m8.spv llm_gemm.comp

// The same eight rungs over L8c-4's 4.5-bit bank (LLM.md L8c-6), and the one
// build in this file that is not ggml's K-quant. MODE 0's k is 10240, so a
// super-block is ggml's eight groups of 32 and the record is its sixteen
// bytes; **MODE 1's k is 320** — the low-rank space — so its super-block is
// the whole row, ten groups, and `-DQ4K_SUB=10` picks the twenty-byte record
// and the twelve-bit decode `get_scale_min_k4` has no spelling for.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=1 -DWN=3 -DQ4B -DDENSE_Q4 -o llm_hc_down_q4_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=1 -DWN=3 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_down_q5_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=2 -DWN=3 -DQ4B -DDENSE_Q4 -o llm_hc_down_q4_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=2 -DWN=3 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_down_q5_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=4 -DWN=3 -DQ4B -DDENSE_Q4 -o llm_hc_down_q4_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=4 -DWN=3 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_down_q5_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=8 -DWN=3 -DQ4B -DDENSE_Q4 -o llm_hc_down_q4_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DWM=8 -DWN=3 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_down_q5_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=1 -DWN=4 -DQ4B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q4_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=1 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q5_m1.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=2 -DWN=4 -DQ4B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q4_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=2 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q5_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=4 -DWN=4 -DQ4B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q4_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=4 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q5_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=8 -DWN=4 -DQ4B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q4_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DWM=8 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -DQ4K_SUB=10 -o llm_hc_up_q5_m8.spv llm_gemm.comp

// The down projection again, at one token: LLM.md L7d. `llm_gemm.comp` blocks
// the output columns, so a fused N of 336 is seven workgroups on a 40-CU
// device and the projection reads 29 GB/s of a 242 GB/s bus (L7c-6). At M=1
// the parallelism has to come from K, so this is the split-K GEMV over the
// same staged weight — one dispatch of 21 x KSLABS waves writing partial
// sums, one that reduces them and carries the GEMM's own epilogue. KSLABS is
// compiled in for the reason BM and BN are: the host sizes the scratch and
// the grid from it.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -o llm_hc_gemv_s8.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=8 -o llm_hc_gemv_r8.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -o llm_hc_gemv_s16.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=16 -o llm_hc_gemv_r16.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -o llm_hc_gemv_s32.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=32 -o llm_hc_gemv_r32.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -o llm_hc_gemv_s40.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=40 -o llm_hc_gemv_r40.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=80 -o llm_hc_gemv_s80.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=80 -o llm_hc_gemv_r80.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=160 -o llm_hc_gemv_s160.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=160 -o llm_hc_gemv_r160.spv llm_hc_gemv.comp

// And the same six splits over L8's bank (L8b). Only the first dispatch of
// each pair has an arm: the reduction reads partial sums and touches no
// weight, so a Q8 build of it would be the same SPIR-V.
//
// **The rung to pick is not the one L7d picked**, and that is D12 rather than
// a surprise: a slab is now `(gemmK/16/KSLABS) * 256` bytes, half what it
// was, so the whole ladder slides one rung along the 4 KB rotation and the
// two splits whose stride is not a whole multiple of 4 KB are 16 and 32
// rather than 32 and 160. Measured by `-hc -tokens 1 -ladder`.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s8.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s16.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s32.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s40.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=80 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s80.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=160 -DQ8B -DDENSE_Q8 -o llm_hc_gemv_q8_s160.spv llm_hc_gemv.comp

// And the same six over the 4.5-bit bank (L8c-6). A slab is
// `(gemmK/16/KSLABS) * 128` bytes here — half the Q8 arm's again — so D12
// says the ladder slides one more rung along §5.1b's 4 KB rotation and the
// rung to pick is measured rather than carried over.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s8.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s8.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s16.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s16.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s32.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s32.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s40.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s40.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=80 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s80.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=80 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s80.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=160 -DQ4B -DDENSE_Q4 -o llm_hc_gemv_q4_s160.spv llm_hc_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=160 -DQ4B -DQ5B -DDENSE_Q4 -o llm_hc_gemv_q5_s160.spv llm_hc_gemv.comp

// The PLE n-gram block (LLM.md L2), which runs once, at layer 1. It is 0.1% of
// a prefill graph — its key projection is one dispatch of the 37 that share
// that shape — so none of this is tuned; what it has to be is on the device,
// because the alternative is a 21 MB round trip of the residual to the host in
// the middle of the stack.
//
// Three dispatches against llama.cpp's ~25: the key and value projections are
// one fused [hc*nEmbd + nEmbd, nEmbd] weight on the plain arm of llm_gemm,
// everything between the projections and the convolution is one workgroup per
// (token, stream), and the dilated depthwise convolution carries its SiLU and
// the residual add.

// The key/value projection's tile is chosen against the *weight*, not the
// activation: its fused B is 65.5 MB of fp16, twice the 32 MiB MALL, so a
// workgroup that carries BM token rows reads all of it M/BM times. At 512
// tokens BM=32 is 1.05 GB of weight traffic and BM=128 is 262 MB. That is the
// whole shape of this dispatch, and the ladder exists to show it.

// MODE=2 is llm_gemm.comp with no epilogue at all, and it is the vertical's
// ordinary projection kernel rather than this block's: the PLE key/value pair
// was its first caller, the full-attention layer's fused QKV and its output
// projection are the next two. The rungs differ only in BM.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=2 -DWN=4 -o llm_gemm_plain_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=4 -DWN=4 -o llm_gemm_plain_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=8 -DWN=4 -o llm_gemm_plain_m8.spv llm_gemm.comp

// The same rungs over L8's dense bank: int8 tiles and an fp16 scale plane
// instead of halves, unpacked a slab at a time into LDS. -DDENSE_Q8 is what
// names the sixth buffer, -DQ8B what changes the kernel; they are two flags
// because the first is a fact about the *descriptor set* and the second about
// the loop, and llm_common.glsl is shared with kernels that want neither.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=2 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_gemm_q8_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=4 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_gemm_q8_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=8 -DWN=4 -DQ8B -DDENSE_Q8 -o llm_gemm_q8_m8.spv llm_gemm.comp

// And the same rungs again over L8c-4's: nibble tiles and a sixteen-byte
// ggml record per (column, super-block), 4.500 bits a weight against the
// checkpoint's 8.5. -DDENSE_Q4 names the same sixth buffer -DDENSE_Q8 does,
// because a nibble bank and a byte bank are both "the dense bank as raw
// words"; -DQ4B is the unpack.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=2 -DWN=4 -DQ4B -DDENSE_Q4 -o llm_gemm_q4_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=2 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemm_q5_m2.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=4 -DWN=4 -DQ4B -DDENSE_Q4 -o llm_gemm_q4_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=4 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemm_q5_m4.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=8 -DWN=4 -DQ4B -DDENSE_Q4 -o llm_gemm_q4_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -DWM=8 -DWN=4 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemm_q5_m8.spv llm_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_ple_gate.spv llm_ple_gate.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_ple_conv.spv llm_ple_conv.comp

//go:embed llm_hc_norm.spv
var LLMHCNorm []byte

//go:embed llm_hc_combine.spv
var LLMHCCombine []byte

//go:embed llm_hc_cn.spv
var LLMHCCN []byte

//go:embed llm_hc_down_m1.spv
var LLMHCDownM1 []byte

//go:embed llm_hc_down_m2.spv
var LLMHCDownM2 []byte

//go:embed llm_hc_down_m4.spv
var LLMHCDownM4 []byte

//go:embed llm_hc_up_m1.spv
var LLMHCUpM1 []byte

//go:embed llm_hc_up_m2.spv
var LLMHCUpM2 []byte

//go:embed llm_hc_up_m4.spv
var LLMHCUpM4 []byte

// The widest rung of each ladder, and L8b's reason for it: the Q8 arm's
// unpack costs 256/WM element conversions per cooperative-matrix step, so it
// wants more token rows a workgroup than the fp16 arm ever did.

//go:embed llm_hc_down_m8.spv
var LLMHCDownM8 []byte

//go:embed llm_hc_up_m8.spv
var LLMHCUpM8 []byte

// The same rungs over L8's dense bank (L8b).

//go:embed llm_hc_down_q8_m1.spv
var LLMHCDownQ8M1 []byte

//go:embed llm_hc_down_q8_m2.spv
var LLMHCDownQ8M2 []byte

//go:embed llm_hc_down_q8_m4.spv
var LLMHCDownQ8M4 []byte

//go:embed llm_hc_up_q8_m1.spv
var LLMHCUpQ8M1 []byte

//go:embed llm_hc_up_q8_m2.spv
var LLMHCUpQ8M2 []byte

//go:embed llm_hc_up_q8_m4.spv
var LLMHCUpQ8M4 []byte

//go:embed llm_hc_down_q8_m8.spv
var LLMHCDownQ8M8 []byte

//go:embed llm_hc_up_q8_m8.spv
var LLMHCUpQ8M8 []byte

// The same rungs over L8c-4's 4.5-bit bank (L8c-6).

//go:embed llm_hc_down_q4_m1.spv
var LLMHCDownQ4M1 []byte

//go:embed llm_hc_down_q5_m1.spv
var LLMHCDownQ5M1 []byte

//go:embed llm_hc_down_q4_m2.spv
var LLMHCDownQ4M2 []byte

//go:embed llm_hc_down_q5_m2.spv
var LLMHCDownQ5M2 []byte

//go:embed llm_hc_down_q4_m4.spv
var LLMHCDownQ4M4 []byte

//go:embed llm_hc_down_q5_m4.spv
var LLMHCDownQ5M4 []byte

//go:embed llm_hc_down_q4_m8.spv
var LLMHCDownQ4M8 []byte

//go:embed llm_hc_down_q5_m8.spv
var LLMHCDownQ5M8 []byte

//go:embed llm_hc_up_q4_m1.spv
var LLMHCUpQ4M1 []byte

//go:embed llm_hc_up_q5_m1.spv
var LLMHCUpQ5M1 []byte

//go:embed llm_hc_up_q4_m2.spv
var LLMHCUpQ4M2 []byte

//go:embed llm_hc_up_q5_m2.spv
var LLMHCUpQ5M2 []byte

//go:embed llm_hc_up_q4_m4.spv
var LLMHCUpQ4M4 []byte

//go:embed llm_hc_up_q5_m4.spv
var LLMHCUpQ5M4 []byte

//go:embed llm_hc_up_q4_m8.spv
var LLMHCUpQ4M8 []byte

//go:embed llm_hc_up_q5_m8.spv
var LLMHCUpQ5M8 []byte

// LLMHCGemvS* is the split-K down projection at one token and LLMHCGemvR* the
// reduction and epilogue that closes it (L7d). They come in pairs: the slab
// count is compiled into both.

//go:embed llm_hc_gemv_s8.spv
var LLMHCGemvS8 []byte

//go:embed llm_hc_gemv_r8.spv
var LLMHCGemvR8 []byte

//go:embed llm_hc_gemv_s16.spv
var LLMHCGemvS16 []byte

//go:embed llm_hc_gemv_r16.spv
var LLMHCGemvR16 []byte

//go:embed llm_hc_gemv_s32.spv
var LLMHCGemvS32 []byte

//go:embed llm_hc_gemv_r32.spv
var LLMHCGemvR32 []byte

//go:embed llm_hc_gemv_s40.spv
var LLMHCGemvS40 []byte

//go:embed llm_hc_gemv_r40.spv
var LLMHCGemvR40 []byte

//go:embed llm_hc_gemv_s80.spv
var LLMHCGemvS80 []byte

//go:embed llm_hc_gemv_r80.spv
var LLMHCGemvR80 []byte

//go:embed llm_hc_gemv_s160.spv
var LLMHCGemvS160 []byte

//go:embed llm_hc_gemv_r160.spv
var LLMHCGemvR160 []byte

// LLMHCGemvQ8S* is the first dispatch of each pair over L8's bank (L8b). The
// reduction is shared with the fp16 build: it reads partial sums.

//go:embed llm_hc_gemv_q8_s8.spv
var LLMHCGemvQ8S8 []byte

//go:embed llm_hc_gemv_q8_s16.spv
var LLMHCGemvQ8S16 []byte

//go:embed llm_hc_gemv_q8_s32.spv
var LLMHCGemvQ8S32 []byte

//go:embed llm_hc_gemv_q8_s40.spv
var LLMHCGemvQ8S40 []byte

//go:embed llm_hc_gemv_q8_s80.spv
var LLMHCGemvQ8S80 []byte

//go:embed llm_hc_gemv_q8_s160.spv
var LLMHCGemvQ8S160 []byte

// LLMHCGemvQ4S* is the same over the 4.5-bit bank (L8c-6).

//go:embed llm_hc_gemv_q4_s8.spv
var LLMHCGemvQ4S8 []byte

//go:embed llm_hc_gemv_q5_s8.spv
var LLMHCGemvQ5S8 []byte

//go:embed llm_hc_gemv_q4_s16.spv
var LLMHCGemvQ4S16 []byte

//go:embed llm_hc_gemv_q5_s16.spv
var LLMHCGemvQ5S16 []byte

//go:embed llm_hc_gemv_q4_s32.spv
var LLMHCGemvQ4S32 []byte

//go:embed llm_hc_gemv_q5_s32.spv
var LLMHCGemvQ5S32 []byte

//go:embed llm_hc_gemv_q4_s40.spv
var LLMHCGemvQ4S40 []byte

//go:embed llm_hc_gemv_q5_s40.spv
var LLMHCGemvQ5S40 []byte

//go:embed llm_hc_gemv_q4_s80.spv
var LLMHCGemvQ4S80 []byte

//go:embed llm_hc_gemv_q5_s80.spv
var LLMHCGemvQ5S80 []byte

//go:embed llm_hc_gemv_q4_s160.spv
var LLMHCGemvQ4S160 []byte

//go:embed llm_hc_gemv_q5_s160.spv
var LLMHCGemvQ5S160 []byte

//go:embed llm_gemm_plain_m2.spv
var LLMGEMMPlainM2 []byte

//go:embed llm_gemm_plain_m4.spv
var LLMGEMMPlainM4 []byte

//go:embed llm_gemm_plain_m8.spv
var LLMGEMMPlainM8 []byte

//go:embed llm_gemm_q8_m2.spv
var LLMGEMMQ8M2 []byte

//go:embed llm_gemm_q8_m4.spv
var LLMGEMMQ8M4 []byte

//go:embed llm_gemm_q8_m8.spv
var LLMGEMMQ8M8 []byte

//go:embed llm_gemm_q4_m2.spv
var LLMGEMMQ4M2 []byte

//go:embed llm_gemm_q5_m2.spv
var LLMGEMMQ5M2 []byte

//go:embed llm_gemm_q4_m4.spv
var LLMGEMMQ4M4 []byte

//go:embed llm_gemm_q5_m4.spv
var LLMGEMMQ5M4 []byte

//go:embed llm_gemm_q4_m8.spv
var LLMGEMMQ4M8 []byte

//go:embed llm_gemm_q5_m8.spv
var LLMGEMMQ5M8 []byte

// The **decode** projection: llm_gemv.comp, LLM.md L8d. llm_gemm.comp MODE 2
// at one token is a sixteen-row fragment holding one row, an LDS slab per
// K-step and a grid of gemmN/64 workgroups; this is the split-K GEMV over the
// same fragment tiling and the same L8a bank, in two dispatches — or one at
// KSLABS=1, where MODE 0 stores the row itself. Every rung of both banks is
// built because D12's 4 KB rotation picks a different one per K.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -o llm_gemv_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -o llm_gemv_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -o llm_gemv_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -o llm_gemv_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -o llm_gemv_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -o llm_gemv_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=2 -o llm_gemv_sum_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=4 -o llm_gemv_sum_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=8 -o llm_gemv_sum_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=16 -o llm_gemv_sum_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=32 -o llm_gemv_sum_k32.spv llm_gemv.comp

//go:embed llm_gemv_k1.spv
var LLMGEMVK1 []byte

//go:embed llm_gemv_k2.spv
var LLMGEMVK2 []byte

//go:embed llm_gemv_k4.spv
var LLMGEMVK4 []byte

//go:embed llm_gemv_k8.spv
var LLMGEMVK8 []byte

//go:embed llm_gemv_k16.spv
var LLMGEMVK16 []byte

//go:embed llm_gemv_k32.spv
var LLMGEMVK32 []byte

//go:embed llm_gemv_q8_k1.spv
var LLMGEMVQ8K1 []byte

//go:embed llm_gemv_q8_k2.spv
var LLMGEMVQ8K2 []byte

//go:embed llm_gemv_q8_k4.spv
var LLMGEMVQ8K4 []byte

//go:embed llm_gemv_q8_k8.spv
var LLMGEMVQ8K8 []byte

//go:embed llm_gemv_q8_k16.spv
var LLMGEMVQ8K16 []byte

//go:embed llm_gemv_q8_k32.spv
var LLMGEMVQ8K32 []byte

//go:embed llm_gemv_q4_k1.spv
var LLMGEMVQ4K1 []byte

//go:embed llm_gemv_q5_k1.spv
var LLMGEMVQ5K1 []byte

//go:embed llm_gemv_q4_k2.spv
var LLMGEMVQ4K2 []byte

//go:embed llm_gemv_q5_k2.spv
var LLMGEMVQ5K2 []byte

//go:embed llm_gemv_q4_k4.spv
var LLMGEMVQ4K4 []byte

//go:embed llm_gemv_q5_k4.spv
var LLMGEMVQ5K4 []byte

//go:embed llm_gemv_q4_k8.spv
var LLMGEMVQ4K8 []byte

//go:embed llm_gemv_q5_k8.spv
var LLMGEMVQ5K8 []byte

//go:embed llm_gemv_q4_k16.spv
var LLMGEMVQ4K16 []byte

//go:embed llm_gemv_q5_k16.spv
var LLMGEMVQ5K16 []byte

//go:embed llm_gemv_q4_k32.spv
var LLMGEMVQ4K32 []byte

//go:embed llm_gemv_q5_k32.spv
var LLMGEMVQ5K32 []byte

//go:embed llm_gemv_sum_k2.spv
var LLMGEMVSumK2 []byte

//go:embed llm_gemv_sum_k4.spv
var LLMGEMVSumK4 []byte

//go:embed llm_gemv_sum_k8.spv
var LLMGEMVSumK8 []byte

//go:embed llm_gemv_sum_k16.spv
var LLMGEMVSumK16 []byte

//go:embed llm_gemv_sum_k32.spv
var LLMGEMVSumK32 []byte

//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=20 -o llm_gemv_k20.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=20 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k20.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=20 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k20.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=20 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k20.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=20 -o llm_gemv_sum_k20.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -o llm_gemv_k40.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ8B -DDENSE_Q8 -o llm_gemv_q8_k40.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ4B -DDENSE_Q4 -o llm_gemv_q4_k40.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -DQ4B -DQ5B -DDENSE_Q4 -o llm_gemv_q5_k40.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=40 -o llm_gemv_sum_k40.spv llm_gemv.comp

//go:embed llm_gemv_k20.spv
var LLMGEMVK20 []byte

//go:embed llm_gemv_q8_k20.spv
var LLMGEMVQ8K20 []byte

//go:embed llm_gemv_q4_k20.spv
var LLMGEMVQ4K20 []byte

//go:embed llm_gemv_q5_k20.spv
var LLMGEMVQ5K20 []byte

//go:embed llm_gemv_sum_k20.spv
var LLMGEMVSumK20 []byte

//go:embed llm_gemv_k40.spv
var LLMGEMVK40 []byte

//go:embed llm_gemv_q8_k40.spv
var LLMGEMVQ8K40 []byte

//go:embed llm_gemv_q4_k40.spv
var LLMGEMVQ4K40 []byte

//go:embed llm_gemv_q5_k40.spv
var LLMGEMVQ5K40 []byte

//go:embed llm_gemv_sum_k40.spv
var LLMGEMVSumK40 []byte

//go:embed llm_ple_gate.spv
var LLMPLEGate []byte

//go:embed llm_ple_conv.spv
var LLMPLEConv []byte

// LLMSeqHist stores a run's last rows into a convolution's ring, so that the
// next run can reach behind itself (L7b). One kernel for the PLE block and
// the gated DeltaNet both, because the shape is the same; it is a pure write
// — the older slots of the ring are already what they should be — which is
// what lets a one-token decode step cost one row and no read.
//
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_seq_hist.spv llm_seq_hist.comp
//go:embed llm_seq_hist.spv
var LLMSeqHist []byte

// The full-attention layer and the QSA indexer (LLM.md L2f): 12 of the 48
// layers, and 5.64% of llama.cpp's prefill graph in 19 dispatches a layer.
//
// Five dispatches replace them, and the saving is all in what never gets
// written. One fused projection produces the query, its gate, the key, the
// value and both of the indexer's operands, because all six read the same
// normalised residual — the same argument that put `inject` on the hyper-
// connection block's down projection (L2a) and the value on the PLE block's
// key. One staging pass does the per-head norm, the interleaved M-RoPE and the
// fragment tiling for all three of q, k and v. The indexer's pool, norm and
// rotate are one kernel over two addressings, and its score, bias, expansion
// and mask are another. And the output gate — a SIGMOID, a MUL and a CONT in
// the reference, over a [T, 6144] tensor — is folded into the attention
// kernel's epilogue, which already holds the output tile for the softmax
// divide.
//
// HEAD_DIM is 256 here against every other attention kernel in the repo at
// 128, which doubles both the query fragments and the output accumulators a
// wave holds across the key loop. That is why the ladder starts at QT=1: the
// register file is what decides it (§2.7, §6.2), not arithmetic intensity,
// which the DiT's ablation found inert.

//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_attn_pack.spv llm_attn_pack.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_attn_idx.spv llm_attn_idx.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_attn_score.spv llm_attn_score.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_attn_select.spv llm_attn_select.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=2 -o llm_attn_qt1_kt2.spv llm_attn_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=1 -DKTIL=4 -o llm_attn_qt1_kt4.spv llm_attn_wmma.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DQT=2 -DKTIL=4 -o llm_attn_qt2_kt4.spv llm_attn_wmma.comp

//go:embed llm_attn_pack.spv
var LLMAttnPack []byte

//go:embed llm_attn_idx.spv
var LLMAttnIdx []byte

//go:embed llm_attn_score.spv
var LLMAttnScore []byte

// LLMAttnSelect is L4b's radix select: llama.cpp's topk_radix_select.comp
// ported pass for pass, writing a per-cell bitmask instead of an index list
// and so deleting the GET_ROWS the reference spends turning one into the
// other. It only runs where the width binds — past 2051 cells — because below
// that it provably names every cell.
//
//go:embed llm_attn_select.spv
var LLMAttnSelect []byte

//go:embed llm_attn_qt1_kt2.spv
var LLMAttnQT1KT2 []byte

//go:embed llm_attn_qt1_kt4.spv
var LLMAttnQT1KT4 []byte

//go:embed llm_attn_qt2_kt4.spv
var LLMAttnQT2KT4 []byte

// The gated DeltaNet (LLM.md L3b): 36 of the 48 layers, and the only kernel
// in this model with a loop-carried dependency as long as the prompt.
//
// Three shaders. `llm_dn_conv` is everything between the fused input
// projection and the recurrence — the depthwise causal convolution, its SiLU,
// the per-head L2 normalisation of q and k, and the two per-head scalars the
// decay and the write strength come from — in one dispatch over a (plane,
// token) grid, against the eight lines of llama.cpp's graph that do the same.
// `llm_dn_norm` is the gated RMS norm behind it, narrowed straight into the
// output matmul's A operand. The input and output projections themselves run
// on the plain arm of llm_gemm.comp, which is already built above.
//
// `llm_dn_scan` is the delta rule, and it is a **ladder rather than a
// kernel**. LPC is how many lanes own one column of the [128, 128] state, so
// it sets the register count per lane (128/LPC), the workgroup count per head
// (128/(COLS*WAVES)) and the number of times each head's q and k row is
// re-read per token, all at once. llama.cpp's own kernel is the top rung —
// one column per workgroup, 6144 workgroups, a 128-fold re-read. Nineteen
// builds over three dials, and their ends are **6.3x apart on identical
// arithmetic**: the measured winner is LPC 8, which is 413 us a layer against
// the reference's 438 (research/l3b-deltanet-gpu.md).

//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_dn_conv.spv llm_dn_conv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_dn_norm.spv llm_dn_norm.comp
// The second dial is QKREG: whether q and k live in registers beside the
// state, as llama.cpp's kernel keeps them, or are staged once per token in
// LDS. Registers are the default wherever 128/LPC of each still fit, and the
// `…s` rungs are the same shapes with the LDS arm forced, which is what
// prices the choice. LPC 2 and 1 have no register arm: 128 state elements a
// lane plus 256 more would not fit a register file with 256.

//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=64 -DWAVES=1 -o llm_dn_scan_l64.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=32 -DWAVES=1 -o llm_dn_scan_l32.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=16 -DWAVES=1 -o llm_dn_scan_l16.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=8 -DWAVES=1 -o llm_dn_scan_l8.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=4 -DWAVES=1 -o llm_dn_scan_l4.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=2 -DWAVES=1 -o llm_dn_scan_l2.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=1 -DWAVES=1 -o llm_dn_scan_l1.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=16 -DWAVES=4 -o llm_dn_scan_l16w4.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=8 -DWAVES=4 -o llm_dn_scan_l8w4.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=4 -DWAVES=4 -o llm_dn_scan_l4w4.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=2 -DWAVES=4 -o llm_dn_scan_l2w4.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=64 -DWAVES=1 -DQKREG=0 -o llm_dn_scan_l64s.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=16 -DWAVES=1 -DQKREG=0 -o llm_dn_scan_l16s.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=8 -DWAVES=1 -DQKREG=0 -o llm_dn_scan_l8s.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=4 -DWAVES=1 -DQKREG=0 -o llm_dn_scan_l4s.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=64 -DWAVES=1 -DPREFETCH=1 -o llm_dn_scan_l64p.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=16 -DWAVES=1 -DPREFETCH=1 -o llm_dn_scan_l16p.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=8 -DWAVES=1 -DPREFETCH=1 -o llm_dn_scan_l8p.spv llm_dn_scan.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DLPC=4 -DWAVES=1 -DPREFETCH=1 -o llm_dn_scan_l4p.spv llm_dn_scan.comp

//go:embed llm_dn_conv.spv
var LLMDNConv []byte

//go:embed llm_dn_norm.spv
var LLMDNNorm []byte

//go:embed llm_dn_scan_l64.spv
var LLMDNScanL64 []byte

//go:embed llm_dn_scan_l32.spv
var LLMDNScanL32 []byte

//go:embed llm_dn_scan_l16.spv
var LLMDNScanL16 []byte

//go:embed llm_dn_scan_l8.spv
var LLMDNScanL8 []byte

//go:embed llm_dn_scan_l4.spv
var LLMDNScanL4 []byte

//go:embed llm_dn_scan_l2.spv
var LLMDNScanL2 []byte

//go:embed llm_dn_scan_l1.spv
var LLMDNScanL1 []byte

//go:embed llm_dn_scan_l16w4.spv
var LLMDNScanL16W4 []byte

//go:embed llm_dn_scan_l8w4.spv
var LLMDNScanL8W4 []byte

//go:embed llm_dn_scan_l4w4.spv
var LLMDNScanL4W4 []byte

//go:embed llm_dn_scan_l2w4.spv
var LLMDNScanL2W4 []byte

//go:embed llm_dn_scan_l64s.spv
var LLMDNScanL64S []byte

//go:embed llm_dn_scan_l16s.spv
var LLMDNScanL16S []byte

//go:embed llm_dn_scan_l8s.spv
var LLMDNScanL8S []byte

//go:embed llm_dn_scan_l4s.spv
var LLMDNScanL4S []byte

//go:embed llm_dn_scan_l64p.spv
var LLMDNScanL64P []byte

//go:embed llm_dn_scan_l16p.spv
var LLMDNScanL16P []byte

//go:embed llm_dn_scan_l8p.spv
var LLMDNScanL8P []byte

//go:embed llm_dn_scan_l4p.spv
var LLMDNScanL4P []byte

// The MoE block — LLM.md L5b — which is 35.7% of llama.cpp's prefill graph,
// 97% of this checkpoint's parameters and the last block of the model to get
// a kernel.
//
// Four shaders. `llm_moe_route` is the router's tail — the softmax over 512,
// the ten workgroup argmaxes that reproduce `ggml_argsort`'s DESC comparator
// without materialising the other 502 ranks, the normalised weights and the
// shared expert's sigmoid gate — one workgroup a token, which is
// `llm_attn_select.comp`'s shape. `llm_moe_perm` is a counting sort over the
// experts and the tile schedule it implies, in one workgroup, because
// L5a-4's routing is a 26x load imbalance that cannot be covered by a static
// grid. `llm_moe_combine` sums a token's eleven contributions, which are
// contiguous by the permutation's own indexing.
//
// `llm_moe_gemm` is the block: a grouped cooperative-matrix GEMM that reads
// the checkpoint's **own quantised blocks** rather than a dequantised bank,
// because one layer's three expert tensors are 5.03 GB at fp16 and 241 GB
// across the model against 1.57 and 75 as they ship. It is a cross of two
// modes (gate/up with `silu(gate)*up` fused onto the accumulators, and down
// with the routing weight and the scatter fused onto the store) against the
// four formats the bank ships in (Q4_K and Q5_K for gate/up, Q5_1 or Q8_0
// for down, Q8_0 for the shared expert), and a row-block ladder over each,
// because L5a-4's distribution is what decides whether a wide tile is reuse
// or padding.
//
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_route.spv llm_moe_route.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_perm.spv llm_moe_perm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_combine.spv llm_moe_combine.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_route.spv llm_moe_route.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_perm.spv llm_moe_perm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -o llm_moe_combine.spv llm_moe_combine.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q4k_m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q4k_m2.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=4 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q4k_m4.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=1 -DWAVES=2 -DNBANK=48 -o llm_moe_up_q4k_w2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=1 -DWAVES=4 -DNBANK=48 -o llm_moe_up_q4k_w4m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q5k_m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q5k_m2.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=4 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q5k_m4.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=1 -DWAVES=2 -DNBANK=48 -o llm_moe_up_q5k_w2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=1 -DWAVES=4 -DNBANK=48 -o llm_moe_up_q5k_w4m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q80_m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q80_m2.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=4 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q80_m4.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=1 -DWAVES=2 -DNBANK=48 -o llm_moe_up_q80_w2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=1 -DWAVES=4 -DNBANK=48 -o llm_moe_up_q80_w4m1.spv llm_moe_gemm.comp

// The **narrow-N** rungs of the up mode: LLM.md L7d. Every rung above blocks
// the output columns at 64, so at one token the routed pair is 10 experts x
// (640/64) = 100 workgroups and the shared expert is **ten**, on a 40-CU
// device — and the kernel is unpack-bound, so what it is short of is
// workgroups and not rows (L7c-6). These cut BN to 16 and 32 instead, which
// multiplies the grid by four and two and divides the slab each workgroup
// unpacks by the same, for the same total unpack. They are up-mode only: the
// down mode's N is 2560 and its grid is already 400.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=1 -DWN=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q4k_n1m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DWM=1 -DWN=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q4k_n2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=1 -DWN=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q5k_n1m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DWM=1 -DWN=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q5k_n2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=1 -DWN=1 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q80_n1m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DWM=1 -DWN=2 -DWAVES=1 -DNBANK=48 -o llm_moe_up_q80_n2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DWM=1 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q51_m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DWM=2 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q51_m2.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DWM=4 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q51_m4.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DWM=1 -DWAVES=2 -DNBANK=48 -o llm_moe_down_q51_w2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DWM=1 -DWAVES=4 -DNBANK=48 -o llm_moe_down_q51_w4m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DWM=1 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q80_m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DWM=2 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q80_m2.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DWM=4 -DWAVES=1 -DNBANK=48 -o llm_moe_down_q80_m4.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DWM=1 -DWAVES=2 -DNBANK=48 -o llm_moe_down_q80_w2m1.spv llm_moe_gemm.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DWM=1 -DWAVES=4 -DNBANK=48 -o llm_moe_down_q80_w4m1.spv llm_moe_gemm.comp

//go:embed llm_moe_up_q4k_m1.spv
var LLMMoEUpQ4KM1 []byte

//go:embed llm_moe_up_q4k_m2.spv
var LLMMoEUpQ4KM2 []byte

//go:embed llm_moe_up_q4k_m4.spv
var LLMMoEUpQ4KM4 []byte

//go:embed llm_moe_up_q4k_w2m1.spv
var LLMMoEUpQ4KW2M1 []byte

//go:embed llm_moe_up_q4k_w4m1.spv
var LLMMoEUpQ4KW4M1 []byte

//go:embed llm_moe_up_q5k_m1.spv
var LLMMoEUpQ5KM1 []byte

//go:embed llm_moe_up_q5k_m2.spv
var LLMMoEUpQ5KM2 []byte

//go:embed llm_moe_up_q5k_m4.spv
var LLMMoEUpQ5KM4 []byte

//go:embed llm_moe_up_q5k_w2m1.spv
var LLMMoEUpQ5KW2M1 []byte

//go:embed llm_moe_up_q5k_w4m1.spv
var LLMMoEUpQ5KW4M1 []byte

//go:embed llm_moe_up_q80_m1.spv
var LLMMoEUpQ80M1 []byte

//go:embed llm_moe_up_q80_m2.spv
var LLMMoEUpQ80M2 []byte

//go:embed llm_moe_up_q80_m4.spv
var LLMMoEUpQ80M4 []byte

//go:embed llm_moe_up_q80_w2m1.spv
var LLMMoEUpQ80W2M1 []byte

//go:embed llm_moe_up_q80_w4m1.spv
var LLMMoEUpQ80W4M1 []byte

//go:embed llm_moe_up_q4k_n1m1.spv
var LLMMoEUpQ4KN1M1 []byte

//go:embed llm_moe_up_q4k_n2m1.spv
var LLMMoEUpQ4KN2M1 []byte

//go:embed llm_moe_up_q5k_n1m1.spv
var LLMMoEUpQ5KN1M1 []byte

//go:embed llm_moe_up_q5k_n2m1.spv
var LLMMoEUpQ5KN2M1 []byte

//go:embed llm_moe_up_q80_n1m1.spv
var LLMMoEUpQ80N1M1 []byte

//go:embed llm_moe_up_q80_n2m1.spv
var LLMMoEUpQ80N2M1 []byte

//go:embed llm_moe_down_q51_m1.spv
var LLMMoEDownQ51M1 []byte

//go:embed llm_moe_down_q51_m2.spv
var LLMMoEDownQ51M2 []byte

//go:embed llm_moe_down_q51_m4.spv
var LLMMoEDownQ51M4 []byte

//go:embed llm_moe_down_q51_w2m1.spv
var LLMMoEDownQ51W2M1 []byte

//go:embed llm_moe_down_q51_w4m1.spv
var LLMMoEDownQ51W4M1 []byte

//go:embed llm_moe_down_q80_m1.spv
var LLMMoEDownQ80M1 []byte

//go:embed llm_moe_down_q80_m2.spv
var LLMMoEDownQ80M2 []byte

//go:embed llm_moe_down_q80_m4.spv
var LLMMoEDownQ80M4 []byte

//go:embed llm_moe_down_q80_w2m1.spv
var LLMMoEDownQ80W2M1 []byte

//go:embed llm_moe_down_q80_w4m1.spv
var LLMMoEDownQ80W4M1 []byte

// The **decode** kernel of the same grouped GEMM: llm_moe_gemv.comp, LLM.md
// L8d. Every rung above is a cooperative-matrix GEMM whose workgroup unpacks
// a BN x BK slab of the checkpoint's own blocks into LDS per K-step and
// multiplies it by BM rows of A — right at prefill, where the unpack is
// amortised over sixteen to sixty-four rows, and wrong at one token, where
// the row block is fifteen sixteenths padding and the loop is a barrier and a
// dependent load per K-step. These have no LDS slab, no barrier in the K
// loop and no fragment: LPR lanes share one output column and walk its row of
// the bank in stride, straight into registers.
//
// Two axes. **LPR** is lanes a column, which is the shape of a load — LPR
// consecutive lanes read LPR consecutive dwords — and it wants to divide the
// row's payload-dword count: 320 for Q4_K/Q5_K at K = 2560, 640 for Q8_0
// there, 160 for Q8_0 at 640 and 80 for Q5_1. **WAVES** buys one thing, the A
// vector staged in LDS once for four waves instead of once for one, and
// changes no arithmetic. AKMAX is the LDS that stages it: nEmbd for the up
// mode, the expert width for the down mode.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=16 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v16.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=32 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v32.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=64 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v64.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=16 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v16w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=32 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v32w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=0 -DLPR=64 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q4k_v64w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=16 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v16.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=32 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v32.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=64 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v64.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=16 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v16w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=32 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v32w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=1 -DLPR=64 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q5k_v64w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=16 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v16.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=32 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v32.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=64 -DWAVES=1 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v64.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=16 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v16w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=32 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v32w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DQFMT=3 -DLPR=64 -DWAVES=4 -DAKMAX=2560 -DNBANK=48 -o llm_moe_gv_up_q80_v64w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=16 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v16.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=32 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v32.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=64 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v64.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=16 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v16w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=32 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v32w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=2 -DLPR=64 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q51_v64w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=16 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v16.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=32 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v32.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=64 -DWAVES=1 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v64.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=16 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v16w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=32 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v32w4.spv llm_moe_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DQFMT=3 -DLPR=64 -DWAVES=4 -DAKMAX=640 -DNBANK=48 -o llm_moe_gv_down_q80_v64w4.spv llm_moe_gemv.comp

//go:embed llm_moe_gv_up_q4k_v16.spv
var LLMMoEGVUpQ4KV16 []byte

//go:embed llm_moe_gv_up_q4k_v32.spv
var LLMMoEGVUpQ4KV32 []byte

//go:embed llm_moe_gv_up_q4k_v64.spv
var LLMMoEGVUpQ4KV64 []byte

//go:embed llm_moe_gv_up_q4k_v16w4.spv
var LLMMoEGVUpQ4KV16W4 []byte

//go:embed llm_moe_gv_up_q4k_v32w4.spv
var LLMMoEGVUpQ4KV32W4 []byte

//go:embed llm_moe_gv_up_q4k_v64w4.spv
var LLMMoEGVUpQ4KV64W4 []byte

//go:embed llm_moe_gv_up_q5k_v16.spv
var LLMMoEGVUpQ5KV16 []byte

//go:embed llm_moe_gv_up_q5k_v32.spv
var LLMMoEGVUpQ5KV32 []byte

//go:embed llm_moe_gv_up_q5k_v64.spv
var LLMMoEGVUpQ5KV64 []byte

//go:embed llm_moe_gv_up_q5k_v16w4.spv
var LLMMoEGVUpQ5KV16W4 []byte

//go:embed llm_moe_gv_up_q5k_v32w4.spv
var LLMMoEGVUpQ5KV32W4 []byte

//go:embed llm_moe_gv_up_q5k_v64w4.spv
var LLMMoEGVUpQ5KV64W4 []byte

//go:embed llm_moe_gv_up_q80_v16.spv
var LLMMoEGVUpQ80V16 []byte

//go:embed llm_moe_gv_up_q80_v32.spv
var LLMMoEGVUpQ80V32 []byte

//go:embed llm_moe_gv_up_q80_v64.spv
var LLMMoEGVUpQ80V64 []byte

//go:embed llm_moe_gv_up_q80_v16w4.spv
var LLMMoEGVUpQ80V16W4 []byte

//go:embed llm_moe_gv_up_q80_v32w4.spv
var LLMMoEGVUpQ80V32W4 []byte

//go:embed llm_moe_gv_up_q80_v64w4.spv
var LLMMoEGVUpQ80V64W4 []byte

//go:embed llm_moe_gv_down_q51_v16.spv
var LLMMoEGVDownQ51V16 []byte

//go:embed llm_moe_gv_down_q51_v32.spv
var LLMMoEGVDownQ51V32 []byte

//go:embed llm_moe_gv_down_q51_v64.spv
var LLMMoEGVDownQ51V64 []byte

//go:embed llm_moe_gv_down_q51_v16w4.spv
var LLMMoEGVDownQ51V16W4 []byte

//go:embed llm_moe_gv_down_q51_v32w4.spv
var LLMMoEGVDownQ51V32W4 []byte

//go:embed llm_moe_gv_down_q51_v64w4.spv
var LLMMoEGVDownQ51V64W4 []byte

//go:embed llm_moe_gv_down_q80_v16.spv
var LLMMoEGVDownQ80V16 []byte

//go:embed llm_moe_gv_down_q80_v32.spv
var LLMMoEGVDownQ80V32 []byte

//go:embed llm_moe_gv_down_q80_v64.spv
var LLMMoEGVDownQ80V64 []byte

//go:embed llm_moe_gv_down_q80_v16w4.spv
var LLMMoEGVDownQ80V16W4 []byte

//go:embed llm_moe_gv_down_q80_v32w4.spv
var LLMMoEGVDownQ80V32W4 []byte

//go:embed llm_moe_gv_down_q80_v64w4.spv
var LLMMoEGVDownQ80V64W4 []byte

// The **decode** router: llm_moe_router.comp, LLM.md L8d. Split-K over the
// same fragment tiling `llm_gemm.comp` MODE 2 reads, in two dispatches — the
// partials and their sum — because at one token 513 output columns are nine
// workgroups and the parallelism has to come from K. KSLABS has to divide 40
// (160 k-tiles in whole four-tile steps) and wants to miss D12's 4 KB
// rotation, so the ladder is 8/10/20/40 and the rule predicts 8 and 40.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -o llm_moe_router_k8.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=8 -o llm_moe_router_k8_r.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=10 -o llm_moe_router_k10.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=10 -o llm_moe_router_k10_r.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=20 -o llm_moe_router_k20.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=20 -o llm_moe_router_k20_r.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=40 -o llm_moe_router_k40.spv llm_moe_router.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=40 -o llm_moe_router_k40_r.spv llm_moe_router.comp

//go:embed llm_moe_router_k8.spv
var LLMMoERouterK8 []byte

//go:embed llm_moe_router_k8_r.spv
var LLMMoERouterK8R []byte

//go:embed llm_moe_router_k10.spv
var LLMMoERouterK10 []byte

//go:embed llm_moe_router_k10_r.spv
var LLMMoERouterK10R []byte

//go:embed llm_moe_router_k20.spv
var LLMMoERouterK20 []byte

//go:embed llm_moe_router_k20_r.spv
var LLMMoERouterK20R []byte

//go:embed llm_moe_router_k40.spv
var LLMMoERouterK40 []byte

//go:embed llm_moe_router_k40_r.spv
var LLMMoERouterK40R []byte

//go:embed llm_moe_route.spv
var LLMMoERoute []byte

//go:embed llm_moe_perm.spv
var LLMMoEPerm []byte

//go:embed llm_moe_combine.spv
var LLMMoECombine []byte

// The move between two blocks' activation arenas (LLM.md L6c). Three
// bindings of its own rather than llm_common.glsl's five, because its two
// buffers belong to *different* blocks; the host builds one pipeline per
// (source, destination) pair.

//go:generate glslc --target-env=vulkan1.2 -O -o llm_move.spv llm_move.comp

//go:embed llm_move.spv
var LLMMove []byte
