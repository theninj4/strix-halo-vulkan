package vae

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
)

// conv2d on the matrix cores -- PIPELINE.md stage 8.
//
// Stage 7 ended with the decode 85% one kernel: shaders/vae_conv2d.comp, 3.02
// s of a 3.62 s decode at 3.0-3.2 TFLOP/s against a 55.5 TFLOP/s fp16
// ceiling. It is the only operator in this pipeline that was never a GEMM
// with the wrong kernel in front of it -- its implicit GEMM had not been
// written -- and this file writes it.
//
// The shape of the answer was measured before any of it: stage 2 found the
// decoder's activations peak at absmax 497, 132x inside fp16, while the sums
// that reach 1.16e7 are the mid block's attention *scores*. So conv's
// operands narrow safely and its accumulators stay fp32, which is what a
// matrix core does natively. TestConvInputsFitFP16 is that measurement as an
// assertion rather than a memory.
//
// Two pieces, and the layout is the interesting one:
//
//   - shaders/vae_pack_conv.comp writes the activation as
//     [ceil(C/16)][H+2][W+2][16] -- 16 channels contiguous per pixel, with a
//     one-pixel zero border. That makes a B fragment (16 channels x 16
//     pixels) a 512 B contiguous column-major load *at any pixel offset*,
//     which is what a +-1 tap shift needs and what no tiling of the pixel
//     axis can give.
//   - shaders/vae_conv_wmma.comp is the implicit GEMM: C[OC, HW] = A[OC,
//     taps*C] * B[taps*C, HW], with B read through addresses rather than
//     materialised (im2col at 1024x1024 would be 4.7 GB) and A the filter,
//     packed once at load into the same 16x16 fragment tiles every WMMA
//     kernel in this engine has wanted since stage 3c.
//
// Every conv in the decoder goes through it, including the two shapes a
// tiled kernel would normally refuse: conv_out has three output channels and
// the 16x16 reference latent makes W as small as 16. The kernel's masked
// epilogue is what buys that, and it is worth the branch precisely because
// the tests run at those sizes -- a fast path the reference latent cannot
// reach is a fast path nothing checks.

// ConvKernel names a build of shaders/vae_conv_wmma.comp, or stage 2b's
// scalar fp32 convolution.
type ConvKernel string

const (
	// ConvScalar is vae_conv2d.comp, the register-blocked fp32 kernel of
	// stage 2b (research/stage-2-vae-decoder.md, IDEAS §2.1). It narrows
	// nothing, so it stays the oracle the matrix-core path is measured
	// against.
	ConvScalar ConvKernel = "scalar"

	Conv64x64      ConvKernel = "64x64"
	Conv64x64K2    ConvKernel = "64x64_k2"
	Conv64x128     ConvKernel = "64x128"
	Conv128x64     ConvKernel = "128x64"
	Conv128x128    ConvKernel = "128x128"
	Conv256x64     ConvKernel = "256x64"
	Conv64x64W32   ConvKernel = "64x64_w32"
	Conv128x64W32  ConvKernel = "128x64_w32"
	Conv128x128W32 ConvKernel = "128x128_w32"

	// The negative controls. Both produce a wrong answer by construction:
	// ConvNoTapShift drops the horizontal tap offset, and ConvPadClamp
	// replicates the edge pixel instead of zeroing the border. They are
	// selectable but are not in the ladder, because a ladder is a measurement
	// and a build that computes the wrong answer has nothing to contribute
	// to one.
	ConvNoTapShift ConvKernel = "no_tap_shift"
	ConvPadClamp   ConvKernel = "pad_clamp"
)

// DefaultConvKernel is the ladder's winner at a 1024x1024 image: 258 ms and
// 38.3 TFLOP/s, against the 3.18 s and 3.1 TFLOP/s of the kernel it replaces.
// It ties with Conv128x128W32 to within the noise at that size and is taken
// over it for the small ones, where a 64-wide tile leaves fewer waves idle on
// a 16-pixel row. See research/stage-8-vae-conv.md.
const DefaultConvKernel = Conv128x64W32

// convVariant is what the host has to know that the SPIR-V does not carry:
// the workgroup's output tile, because it sets the grid, and the wave size,
// because the kernel maps gl_SubgroupID onto a wave grid and a workgroup
// holding the wrong number of waves would write outside its tile.
type convVariant struct {
	name   ConvKernel
	spirv  []byte
	bm, bn int
	waves  int
	wave   uint32
	// pack overrides the packing pass, which only a control does.
	pack []byte
}

// packSpirv is the pass that feeds this build.
func (v convVariant) packSpirv() []byte {
	if v.pack != nil {
		return v.pack
	}
	return shaders.VAEPackConv
}

var convVariants = []convVariant{
	{name: Conv64x64, spirv: shaders.VAEConvWMMA64x64, bm: 64, bn: 64, waves: 1, wave: 64},
	{name: Conv64x64K2, spirv: shaders.VAEConvWMMA64x64K2, bm: 64, bn: 64, waves: 1, wave: 64},
	{name: Conv64x128, spirv: shaders.VAEConvWMMA64x128, bm: 64, bn: 128, waves: 2, wave: 64},
	{name: Conv128x64, spirv: shaders.VAEConvWMMA128x64, bm: 128, bn: 64, waves: 2, wave: 64},
	{name: Conv128x128, spirv: shaders.VAEConvWMMA128x128, bm: 128, bn: 128, waves: 4, wave: 64},
	{name: Conv256x64, spirv: shaders.VAEConvWMMA256x64, bm: 256, bn: 64, waves: 4, wave: 64},
	{name: Conv64x64W32, spirv: shaders.VAEConvWMMA64x64W32, bm: 64, bn: 64, waves: 1, wave: 32},
	{name: Conv128x64W32, spirv: shaders.VAEConvWMMA128x64W32, bm: 128, bn: 64, waves: 2, wave: 32},
	{name: Conv128x128W32, spirv: shaders.VAEConvWMMA128x128W32, bm: 128, bn: 128, waves: 4, wave: 32},
}

var convControls = []convVariant{
	{name: ConvNoTapShift, spirv: shaders.VAEConvWMMANoTapShift, bm: 128, bn: 128, waves: 4, wave: 64},
	{name: ConvPadClamp, spirv: shaders.VAEConvWMMA128x128, bm: 128, bn: 128, waves: 4, wave: 64,
		pack: shaders.VAEPackConvClamp},
}

// ConvKernels lists every build in the ladder, in the order they are
// measured. The controls are not in it.
func ConvKernels() []ConvKernel {
	out := []ConvKernel{ConvScalar}
	for _, v := range convVariants {
		out = append(out, v.name)
	}
	return out
}

func convVariantFor(k ConvKernel) (convVariant, bool) {
	for _, v := range convVariants {
		if v.name == k {
			return v, true
		}
	}
	for _, v := range convControls {
		if v.name == k {
			return v, true
		}
	}
	return convVariant{}, false
}

// convBorder is the one-pixel zero frame the packed layout carries on every
// side. It is what makes the nine taps branchless: every address the kernel
// forms holds either a real activation or a zero, so the convolution's
// padding is in the data rather than in the inner loop.
const convBorder = 1

func ctiles(c int) int { return (c + coopMatTile - 1) / coopMatTile }

// convPadW is the packed layout's row width, and the floor is the whole of
// what it says. The kernel clamps every pixel address to the last whole
// fragment of its channel plane so that nothing it forms can leave the
// buffer; that fragment is the tail of the bottom border row and must
// therefore be sixteen zeros. At a row narrower than sixteen it would not be
// -- it would reach back into the last real row -- which is a bug the
// 1024x1024 image cannot have and the 96x96 one the tests decode does.
func convPadW(w int) int { return max(w+2*convBorder, coopMatTile) }

// packedConvElems is the size, in halves, of a tensor's blocked fp16 form.
func packedConvElems(c, h, w int) int {
	return ctiles(c) * coopMatTile * (h + 2*convBorder) * convPadW(w)
}

// packConvA narrows one filter into the A-operand fragment tiles
// vae_conv_wmma.comp reads: tile (mt, kt) is 256 contiguous halves holding
// element (oc, k) at (oc%16)*16 + k%16, tiles ordered kt-fastest, with
// k = tap*ceil(InC/16)*16 + ic.
//
// The k axis is tap-major so that a k-tile is 16 input channels at *one*
// tap, which is what lets the kernel hold the tap loop outside and form one
// pixel address per tap rather than per k-tile. Both the channel axis and
// the output-channel axis are padded to 16 with zeros, so an InC of 16
// (conv_in) and an OutC of 3 (conv_out) are a padded K and a masked store
// rather than two special cases.
func packConvA(dst []uint16, w []float32, oc, ic, taps int) {
	ct := ctiles(ic)
	kt := ct * taps
	for o := 0; o < oc; o++ {
		mt, lane := o/coopMatTile, (o%coopMatTile)*coopMatTile
		for t := 0; t < taps; t++ {
			for c := 0; c < ic; c++ {
				k := t*ct*coopMatTile + c
				dst[(mt*kt+k/coopMatTile)*coopMatTile*coopMatTile+lane+k%coopMatTile] =
					safetensors.F32ToF16(w[(o*ic+c)*taps+t])
			}
		}
	}
}

// convWeightElems is how many halves packConvA writes for one filter.
func convWeightElems(c *Conv2D) int {
	return ctiles(c.OutC) * ctiles(c.InC) * c.KH * c.KW * coopMatTile * coopMatTile
}

// expandBias repeats each output channel's bias across a 16-wide row, so
// that one row-major accumulator load of stride 16 broadcasts it over a
// fragment's sixteen pixels. That is the whole of the bias handling: the
// kernel *starts* from it instead of adding it, which costs no dispatch, no
// epilogue and nothing in the inner loop. The 16x it costs is 32 KB on the
// widest conv in this decoder.
func expandBias(bias []float32, oc int) []float32 {
	out := make([]float32, ctiles(oc)*coopMatTile*coopMatTile)
	for o := 0; o < oc; o++ {
		for i := 0; i < coopMatTile; i++ {
			out[o*coopMatTile+i] = bias[o]
		}
	}
	return out
}

// stageConvWeights appends every conv filter to the fp16 weight arena in its
// fragment-tile layout. This is 84 M parameters -- the whole decoder -- so
// unlike stage 7's four projections it is worth saying what it costs: 168 MB
// of fp16 beside the 336 MB of fp32 the scalar path still reads for the
// convs it keeps. Nothing is freed, because the fp32 copy is what
// TestGPUConvMatchesScalar compares against and what a device without matrix
// cores runs.
func (g *GPUDecoder) stageConvWeights(data []uint16) []uint16 {
	for _, cr := range g.convs {
		off := uint32(len(data))
		data = append(data, make([]uint16, convWeightElems(cr.conv))...)
		packConvA(data[off:], cr.conv.Weight, cr.conv.OutC, cr.conv.InC, cr.conv.KH*cr.conv.KW)
		g.w16[cr.name+".weight"] = off
	}
	return data
}

// packConv records the fp32 -> blocked-fp16 pass in front of one convolution
// and returns the packed tensor's offset in the fp16 arena.
//
// It is a read of the activation and a write of half of it -- 7 ms at the
// largest shape in the decode against the 30 the convolution it feeds should
// take -- and it is the price of the layout. The cheaper form is to have the
// producer write it (every conv in a resnet is fed by a SiLU), which is a
// fusion this stage leaves on the table rather than one it needs.
func (b *builder) packConv(x tensor) uint32 {
	off := b.har.alloc(packedConvElems(x.C, x.H, x.W))
	b.label(fmt.Sprintf("packconv %dx%dx%d", x.C, x.H, x.W), 0)
	b.add("pack_conv", uint32(ctiles(x.C)*(x.H+2*convBorder)), pushConstants{
		InOff: x.off, OutOff: off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		Aux1: uint32(convPadW(x.W)),
	})
	return off
}

// convCores records one convolution on the matrix cores: the pack, then the
// implicit GEMM.
//
// The grid is (H * ceil(W/BN), ceil(OC/BM)). A pixel tile never straddles two
// image rows -- a tap shift would wrap the wrong pixels in -- so the x axis
// carries the row as well as the tile, and the shader is told how many tiles
// a row holds rather than dividing by a constant it does not have.
func (b *builder) convCores(name string, c *Conv2D, x tensor) tensor {
	v := b.g.conv
	out := tensor{off: b.ar.alloc(c.OutC * x.H * x.W), C: c.OutC, H: x.H, W: x.W}
	packed := b.packConv(x)
	tilesPerRow := (x.W + v.bn - 1) / v.bn
	b.label(convLabel(c, x), convFlops(c, x))
	b.addXY("conv_wmma", uint32(x.H*tilesPerRow), uint32((c.OutC+v.bm-1)/v.bm), pushConstants{
		InOff: packed, OutOff: out.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		OC: uint32(c.OutC), KH: uint32(c.KH), KW: uint32(c.KW), Pad: uint32(c.Pad),
		BOff:  b.convBiasOff(name),
		GemmB: b.b16Off(name + ".weight"),
		Aux0:  uint32(tilesPerRow), Aux1: uint32(convPadW(x.W)),
	})
	b.har.release(packed, packedConvElems(x.C, x.H, x.W))
	return out
}

// convBiasOff resolves the expanded bias, or NO_BIAS. A conv without one is
// not a case this decoder has, but there is no push-constant value that means
// "not set" (stage 6), so it is said rather than assumed.
func (b *builder) convBiasOff(name string) uint32 {
	if off, ok := b.g.weights.off[name+".bias16"]; ok {
		return off
	}
	return noBias
}

func convLabel(c *Conv2D, x tensor) string {
	return fmt.Sprintf("conv%dx%d %d->%d @%dx%d", c.KH, c.KW, x.C, c.OutC, x.H, x.W)
}

func convFlops(c *Conv2D, x tensor) float64 {
	return 2 * float64(c.OutC) * float64(x.H) * float64(x.W) * float64(x.C) *
		float64(c.KH) * float64(c.KW)
}
