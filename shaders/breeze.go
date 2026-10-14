package shaders

import _ "embed"

// Breeze-TTS-2 (breeze/, research/tts-instruct-vertical.md B3/B4): ace/lm's
// prep and split-key attention rebuilt at the backbone's heads (Qwen3-1.7B:
// 16 q, 8 kv, q/k norm) and the depth decoder's (Llama: 8 q, 2 kv, no q/k
// norm). Both stacks rope with NeoX pairing from a host-built table, so the
// depth decoder's llama3 frequency scaling costs the kernels nothing.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNQ=16u -DNKV=8u -o breeze_bb_attn.spv ace_lm_attn.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNQ=16u -DNKV=8u -DCOMBINE=1 -o breeze_bb_attn_combine.spv ace_lm_attn.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNQ=8u -DNKV=2u -o breeze_dd_attn.spv ace_lm_attn.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DNQ=8u -DNKV=2u -DCOMBINE=1 -o breeze_dd_attn_combine.spv ace_lm_attn.comp

//go:embed breeze_bb_attn.spv
var BreezeBBAttn []byte

//go:embed breeze_bb_attn_combine.spv
var BreezeBBAttnCombine []byte

//go:embed breeze_dd_attn.spv
var BreezeDDAttn []byte

//go:embed breeze_dd_attn_combine.spv
var BreezeDDAttnCombine []byte

// The text encoder's prep and windowed bidirectional attention (B2).
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -o breeze_enc_prep.spv breeze_enc.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -o breeze_enc_attn.spv breeze_enc.comp

//go:embed breeze_enc_prep.spv
var BreezeEncPrep []byte

//go:embed breeze_enc_attn.spv
var BreezeEncAttn []byte

// The codec decoder's kernels between its GEMMs (B5), one build a mode.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -o breeze_codec_narrow.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -o breeze_codec_bias.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -o breeze_codec_ln.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=3 -o breeze_codec_gelu.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=4 -o breeze_codec_resid.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=5 -o breeze_codec_dwconv.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=6 -o breeze_codec_copy32.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=7 -o breeze_codec_copy16.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=8 -o breeze_codec_out.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=9 -o breeze_codec_prep.spv breeze_codec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=10 -o breeze_codec_attn.spv breeze_codec.comp

//go:embed breeze_codec_narrow.spv
var BreezeCodecNarrow []byte

//go:embed breeze_codec_bias.spv
var BreezeCodecBias []byte

//go:embed breeze_codec_ln.spv
var BreezeCodecLN []byte

//go:embed breeze_codec_gelu.spv
var BreezeCodecGELU []byte

//go:embed breeze_codec_resid.spv
var BreezeCodecResid []byte

//go:embed breeze_codec_dwconv.spv
var BreezeCodecDWConv []byte

//go:embed breeze_codec_copy32.spv
var BreezeCodecCopy32 []byte

//go:embed breeze_codec_copy16.spv
var BreezeCodecCopy16 []byte

//go:embed breeze_codec_out.spv
var BreezeCodecOut []byte

//go:embed breeze_codec_prep.spv
var BreezeCodecPrep []byte

//go:embed breeze_codec_attn.spv
var BreezeCodecAttn []byte

// B7's fused steps: PREP over a fused q|k|v (one build a stack's heads),
// SwiGLU over a fused gate|up, and the residual join folded into the next
// norm.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DNQ=16u -DNKV=8u -o breeze_dec_prep_bb.spv breeze_dec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DNQ=8u -DNKV=2u -DQK_NORM=0 -o breeze_dec_prep_dd.spv breeze_dec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -o breeze_dec_swiglu.spv breeze_dec.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -o breeze_dec_addnorm.spv breeze_dec.comp

//go:embed breeze_dec_prep_bb.spv
var BreezeDecPrepBB []byte

//go:embed breeze_dec_prep_dd.spv
var BreezeDecPrepDD []byte

//go:embed breeze_dec_swiglu.spv
var BreezeDecSwiGLU []byte

//go:embed breeze_dec_addnorm.spv
var BreezeDecAddNorm []byte

// B7's device frame: the depth decoder's input gather, the backbone's summed
// frame embedding, a hidden-state copy, and the sampler (top-k by radix
// select), so a frame records once and replays.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -o breeze_frame_gather.spv breeze_frame.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -o breeze_frame_embed.spv breeze_frame.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=2 -o breeze_frame_copyh.spv breeze_frame.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=3 -o breeze_frame_sample.spv breeze_frame.comp

//go:embed breeze_frame_gather.spv
var BreezeFrameGather []byte

//go:embed breeze_frame_embed.spv
var BreezeFrameEmbed []byte

//go:embed breeze_frame_copyh.spv
var BreezeFrameCopyH []byte

//go:embed breeze_frame_sample.spv
var BreezeFrameSample []byte

// B7's shared frames: llm_gemv at MAXROWS 8, so two streams' CFG pairs (a
// depth pass 0 of eight rows) stay on the GEMV and every row comes out as it
// does alone.
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -DMAXROWS=8 -o breeze_gemv_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=1 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k1.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -DMAXROWS=8 -o breeze_gemv_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=2 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=2 -DMAXROWS=8 -o breeze_gemv_sum_k2.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -DMAXROWS=8 -o breeze_gemv_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=4 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=4 -DMAXROWS=8 -o breeze_gemv_sum_k4.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DMAXROWS=8 -o breeze_gemv_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=8 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=8 -DMAXROWS=8 -o breeze_gemv_sum_k8.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DMAXROWS=8 -o breeze_gemv_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=16 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=16 -DMAXROWS=8 -o breeze_gemv_sum_k16.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DMAXROWS=8 -o breeze_gemv_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=0 -DKSLABS=32 -DMAXROWS=8 -DQ8B -DDENSE_Q8 -o breeze_gemv_q8_k32.spv llm_gemv.comp
//go:generate glslc --target-env=vulkan1.2 -O -I. -DMODE=1 -DKSLABS=32 -DMAXROWS=8 -o breeze_gemv_sum_k32.spv llm_gemv.comp

//go:embed breeze_gemv_k1.spv
var BreezeGEMVK1 []byte

//go:embed breeze_gemv_q8_k1.spv
var BreezeGEMVQ8K1 []byte

//go:embed breeze_gemv_k2.spv
var BreezeGEMVK2 []byte

//go:embed breeze_gemv_q8_k2.spv
var BreezeGEMVQ8K2 []byte

//go:embed breeze_gemv_sum_k2.spv
var BreezeGEMVSumK2 []byte

//go:embed breeze_gemv_k4.spv
var BreezeGEMVK4 []byte

//go:embed breeze_gemv_q8_k4.spv
var BreezeGEMVQ8K4 []byte

//go:embed breeze_gemv_sum_k4.spv
var BreezeGEMVSumK4 []byte

//go:embed breeze_gemv_k8.spv
var BreezeGEMVK8 []byte

//go:embed breeze_gemv_q8_k8.spv
var BreezeGEMVQ8K8 []byte

//go:embed breeze_gemv_sum_k8.spv
var BreezeGEMVSumK8 []byte

//go:embed breeze_gemv_k16.spv
var BreezeGEMVK16 []byte

//go:embed breeze_gemv_q8_k16.spv
var BreezeGEMVQ8K16 []byte

//go:embed breeze_gemv_sum_k16.spv
var BreezeGEMVSumK16 []byte

//go:embed breeze_gemv_k32.spv
var BreezeGEMVK32 []byte

//go:embed breeze_gemv_q8_k32.spv
var BreezeGEMVQ8K32 []byte

//go:embed breeze_gemv_sum_k32.spv
var BreezeGEMVSumK32 []byte
