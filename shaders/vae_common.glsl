// Shared binding and push-constant layout for the VAE decoder shaders.
//
// Every decoder pipeline is built over the *same* buffers -- one weight
// arena and one activation arena, since Q9b retired the two fp16 ones -- and
// declares the *same* push-constant size, because vk.DispatchMultiTimed
// records a whole graph into one command buffer and requires that. A shader
// declares only the bindings it reads; the descriptors it does not name are
// inert.
// It is also the right shape for an engine: weights are one arena written
// once, activations are one arena the graph ping-pongs inside, and "which
// tensor" is an offset rather than a descriptor rebind.
//
// The decoder is fp32 end to end, and that is a measured decision rather
// than an unfinished one: IMAGE.md Q9b's TestConvFP16Ladder shows a single
// narrowed operand anywhere in this graph moving the decoded image by max
// abs 0.09, against the fp32 path's own 7.3e-4, because the tail norm
// divides a per-pixel L2 out of a residual stream at absmax 2.6e5. Z-Image's
// mid block did put its projections and attention on the matrix cores over
// two further bindings, an fp16 activation arena and an fp16 weight arena;
// those shaders are gone and no shader here declares those bindings. The
// oracle is the CPU implementation in qimage/vae.

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
    // the same byte offsets. It exists because the mid block's projections
    // once ran on shaders/dit_gemm.comp *unmodified*, which meant the two
    // push-constant blocks had to agree wherever that shader reads: inOff and
    // outOff, which already coincide at 0 and 1, and these six.
    //
    // No VAE shader reads them today -- qvae_gemm_f32.comp took the
    // projections in fp32 (IMAGE.md Q9b) and uses the fields above -- but
    // they stay, because vk.DispatchMultiTimed records one push-constant size
    // across a whole command buffer and that is what fixes this block at 22
    // words. Shrinking it is a change to every VAE pipeline and to the Go
    // struct that mirrors it, for six words of a block that already fits
    // inside Vulkan's guaranteed 128.
    uint gemmB;    // B, in the fp16 weight arena
    uint gemmM;    // rows of A and C, padded up to the workgroup tile
    uint gemmN;    // columns of B and C; also C's row stride
    uint gemmK;    // the reduction extent
    uint lda;      // A's row stride in halves
    uint ldb;      // B's row stride in halves; unused when B is tiled
} pc;

const uint NO_BIAS = 0xffffffffu;
