// Shared binding and push-constant layout for the Z-Image DiT shaders.
//
// Same arrangement as the VAE decoder's (vae_common.glsl) and for the same
// reason: one weight arena, one activation arena, "which tensor" expressed
// as an offset, and one push-constant size across every pipeline so
// vk.DispatchMultiTimed can record a graph into one command buffer.
//
// fp32 throughout for the scalar kernels, whose correctness port had
// zimage/dit's CPU implementation as its oracle. The matrix-core kernels add a
// third binding -- the fp16 fragment-tile arena of dit_pack_f16.comp -- which
// the shaders that do not use it simply leave undeclared.

layout(binding = 0) readonly buffer Weights { float wbuf[]; };
layout(binding = 1) buffer Act { float act[]; };

layout(push_constant) uniform PC {
    uint inOff;     // input tensor
    uint outOff;    // output tensor
    uint wOff;      // weight tensor, or NO_W
    uint tokens;    // sequence length
    uint dim;       // full feature width
    uint heads;
    uint headDim;
    uint span;      // rmsnorm width: dim for the block norms, headDim for q/k
    uint kOff;      // attention: transposed keys
    uint vOff;      // attention: values
    uint kStride;   // attention: padded token stride of the transposed keys
    uint eps;       // float bits
    uint scale;     // float bits
    uint aux0;      // rope: sin table; pack: 0 natural / 1 transposed tiles
    uint aux1;      // WMMA path: padded token count, i.e. the per-head plane
    uint aux2;
    // The GEMM block (dit_gemm.comp). C = A*B with A the activations in the
    // fp16 arena at inOff, B the weights in the fp16 weight arena at bOff and
    // C fp32 in the activation arena at outOff, so the three offsets a
    // projection needs are already here and only its extents and strides are
    // not. Kept as their own fields rather than folded onto `dim`/`span`
    // because a GEMM's four numbers are M, N, K and a leading dimension per
    // operand, and reusing an unrelated name for one of them is how a graph
    // with seven projections in it gets silently wrong.
    uint bOff;      // GEMM: B, in the fp16 weight arena
    uint gemmM;     // GEMM: rows of A and C, padded up to the tile
    uint gemmN;     // GEMM: columns of B and C; also C's row stride
    uint gemmK;     // GEMM: the reduction extent
    uint lda;       // GEMM: A's row stride in halves, K + pad (IDEAS §2.3)
    uint ldb;       // GEMM: B's row stride in halves; unused when B is tiled
} pc;

const uint NO_W = 0xffffffffu;
