// Shared binding and push-constant layout for the Z-Image DiT shaders.
//
// Same arrangement as the VAE decoder's (vae_common.glsl) and for the same
// reason: one weight arena, one activation arena, "which tensor" expressed
// as an offset, and one push-constant size across every pipeline so
// vk.DispatchMultiTimed can record a graph into one command buffer.
//
// fp32 throughout. Stage 3 is a correctness port with zimage/dit's CPU
// implementation as its oracle.

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
    uint aux0;
    uint aux1;
    uint aux2;
} pc;

const uint NO_W = 0xffffffffu;
