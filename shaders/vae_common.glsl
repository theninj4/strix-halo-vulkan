// Shared binding and push-constant layout for the VAE decoder shaders.
//
// Every decoder pipeline declares the *same* two buffers and the *same*
// push-constant size, because vk.DispatchMultiTimed records a whole graph
// into one command buffer and requires that. It is also the right shape for
// an engine: weights are one arena written once, activations are one arena
// the graph ping-pongs inside, and "which tensor" is an offset rather than a
// descriptor rebind.
//
// All arithmetic is fp32 here. Stage 2b is a correctness port with the CPU
// implementation in zimage/vae as its oracle; moving the weights to fp16 and
// the matmuls onto the WMMA path is a separate change with its own
// measurement, and doing both at once would leave a numerical discrepancy
// with two possible causes.

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
} pc;

const uint NO_BIAS = 0xffffffffu;
