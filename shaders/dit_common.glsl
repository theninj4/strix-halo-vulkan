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
} pc;

const uint NO_W = 0xffffffffu;
