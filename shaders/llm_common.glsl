// Shared binding and push-constant layout for the qwen3.8-flash-next
// kernels (LLM.md L2).
//
// Same four-arena arrangement the DiT and the text encoder use
// (dit_common.glsl): one fp32 weight arena, one fp32 activation arena, one
// fp16 activation arena and one fp16 weight bank, with "which tensor"
// expressed as an offset so that a whole block is a dispatch sequence over
// one descriptor set. vk.DispatchMultiTimed records a sequence into a single
// command buffer only if every pipeline in it was built over the same buffers
// with the same push-constant size, so the block below is one size for every
// kernel in this vertical.
//
// It is a separate file from dit_common.glsl rather than a reuse of it
// because the fields are a contract with a *shader*, and a hyper-connection
// block has no heads, no head dim and no rotary table, while it does have a
// stream count, a low rank and three activation tensors that do not exist
// anywhere else. Naming one of those `span` or `aux1` is how a graph gets
// silently wrong (the reason zimage/qwen keeps its own copy of the DiT's).

layout(binding = 0) readonly buffer Weights { float wbuf[]; };
layout(binding = 1) buffer Act { float act[]; };
layout(binding = 2) buffer HAct { float16_t hact[]; };
layout(binding = 3) readonly buffer W16 { float16_t w16[]; };

layout(push_constant) uniform PC {
    // The wide residual, fp32 in the activation arena: [T][hc*nEmbd], which
    // is ggml's [nEmbd, hc, T] read the same way round (ne[0] is fastest).
    uint resOff;
    // The normalised activation, fp16: [T][lda], stream c at column c*nEmbd.
    // Both matmuls of the block read it, which is the fusion.
    uint xnOff;
    // The low-rank gate input after silu(x/hc), fp16: [T][ldaLo].
    uint loOff;
    // The scatter weights, fp32: [T][gemmN - lowRank]. The down projection's
    // last tile writes them, so the row stride is the tile and not hc.
    uint injOff;
    // The block's fp32 output: `mixed` out of the mix, `res` out of the
    // combine, whose other input is the block output at outOff.
    uint outOff;
    // Per-stream RMSNorm gamma, fp32: [hc*nEmbd].
    uint gammaOff;
    // The packed fp16 weight, in the bank.
    uint bOff;
    // Validation only: where to write the 10240-wide gate the fused kernel
    // otherwise never materialises, or NO_W to leave it unwritten.
    uint gateOff;

    uint tokens;
    uint nEmbd;
    uint hc;
    uint lowRank;
    uint lda;       // xn row stride, halves
    uint ldaLo;     // lo row stride, halves
    uint gemmM;     // rows of A and C, padded up to the tile
    uint gemmN;     // columns of B and C
    uint gemmK;     // the reduction extent
    uint eps;       // float bits

    // The PLE block (llm_ple_gate.comp, llm_ple_conv.comp). It runs once, at
    // layer 1, so nothing here is on a hot path — but it reads the same wide
    // residual and writes back into it, so it belongs in the same contract
    // rather than in arenas of its own.
    uint kvOff;      // f32 key and value from one fused projection, [T][gemmN]
    uint gatedOff;   // f32 value * gate, broadcast over the streams
    uint normOff;    // f32 grouped-normed gated value: the convolution's input
    uint convOff;    // f32 depthwise weights, [hc*nEmbd][kern], channel-major
    uint convOutOff; // f32 silu(conv), or NO_W — the residual add does not need it
    uint gammaQOff;  // f32 query norm, over the residual
    uint gammaCOff;  // f32 conv norm, over the gated value
    uint kern;       // conv taps
    uint dil;        // conv dilation, which is the n-gram size
} pc;

const uint NO_W = 0xffffffffu;
