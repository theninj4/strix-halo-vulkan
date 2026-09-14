// Shared binding and push-constant layout for the VAE decoder shaders.
//
// Every decoder pipeline is built over the *same* four buffers and declares
// the *same* push-constant size, because vk.DispatchMultiTimed records a
// whole graph into one command buffer and requires that. A shader declares
// only the bindings it reads; the descriptors it does not name are inert.
// It is also the right shape for an engine: weights are one arena written
// once, activations are one arena the graph ping-pongs inside, and "which
// tensor" is an offset rather than a descriptor rebind.
//
// The decoder is fp32 except in the mid block, where stage 7 put the four
// projections and the attention on the matrix cores: those read two further
// bindings, an fp16 activation arena and an fp16 weight arena, and the
// shaders that do not use them simply leave them undeclared. Everything else
// is still the fp32 correctness port that had the CPU implementation in
// zimage/vae as its oracle.

layout(binding = 0) readonly buffer Weights { float wbuf[]; };
layout(binding = 1) buffer Act { float act[]; };

layout(push_constant) uniform PC {
    uint inOff;    // element offset of the input tensor in act
    uint outOff;   // element offset of the output tensor in act
    uint C;        // input channels
    uint H;        // input height
    uint W;        // input width
    uint OC;       // output channels
    uint KH;       // kernel height
    uint KW;       // kernel width
    uint pad;      // symmetric zero padding
    uint wOff;     // element offset of the weight tensor in wbuf
    uint bOff;     // element offset of the bias tensor in wbuf, or NO_BIAS
    uint groups;   // group count for group norm
    uint resOff;   // element offset of the residual/second operand in act
    uint aux0;
    uint aux1;
    uint aux2;
    // The GEMM block, mirroring shaders/dit_common.glsl field for field at
    // the same byte offsets. The attention projections run on
    // shaders/dit_gemm.comp *unmodified* -- it is a tuned kernel with three B
    // layouts and a grid swizzle behind it, and porting it onto a second
    // push-constant block would fork it -- so the two blocks have to agree
    // wherever that shader reads: inOff and outOff, which already coincide at
    // 0 and 1, and these six. Their size has to agree too, since
    // vk.DispatchMultiTimed records one push-constant size across a whole
    // command buffer, which is what fixes this block at 22 words.
    uint gemmB;    // B, in the fp16 weight arena
    uint gemmM;    // rows of A and C, padded up to the workgroup tile
    uint gemmN;    // columns of B and C; also C's row stride
    uint gemmK;    // the reduction extent
    uint lda;      // A's row stride in halves
    uint ldb;      // B's row stride in halves; unused when B is tiled
} pc;

const uint NO_BIAS = 0xffffffffu;
