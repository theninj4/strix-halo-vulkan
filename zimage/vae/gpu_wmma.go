package vae

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// The mid block on the matrix cores -- PIPELINE.md stage 7.
//
// Stage 2b's decoder is fp32 everywhere and its attention block is the two
// slowest things in a 1024x1024 decode: the softmax over 16384 pixels is
// **1.42 s in one dispatch, 26%** of the image, and the four 512x512
// projections another 544 ms at **62 GFLOP/s**, which is 0.1% of what this
// part's matrix cores do. Both are GEMMs in the end, and the engine already
// has a tuned one.
//
// So this file is mostly plumbing around two reuses:
//
//   - the projections run on shaders/dit_gemm.comp *unmodified*, which is why
//     vae_common.glsl's push-constant block mirrors dit_common.glsl's GEMM
//     fields at the same offsets. Forking a kernel that carries §2.1, §2.3,
//     §2.4 and §2.7 behind it to rename six fields would be the wrong trade.
//   - the weights are staged as 16x16 fragment tiles, which stage 4a measured
//     at 1.46x on the DiT's projections and which costs nothing here: a VAE
//     weight is packed once at load and read every decode.
//
// The attention kernel is the one thing that could not be reused --
// shaders/vae_attention_wmma.comp says why -- and the fp16 activations it
// wants are produced by two small passes, a row-major narrowing for the GEMM
// operands and a fragment-tile pack for q, k and v.
//
// The fp32 path is kept and is what the fp16 one is tested against
// (gpu_test.go). It is also the fallback: a device without fp16 storage,
// cooperative matrices or subgroup-size control runs the whole decoder
// exactly as stage 2b left it.

// AttnKernel names a build of shaders/vae_attention_wmma.comp, or the fp32
// kernel of stage 2b.
type AttnKernel string

const (
	// AttnScalar is vae_attention.comp, the two-pass fp32 kernel. It stays
	// the oracle: it never narrows anything, so a disagreement between it and
	// a matrix-core build is the matrix-core build's.
	AttnScalar AttnKernel = "scalar"

	AttnQT1KT2    AttnKernel = "qt1_kt2"
	AttnQT1KT4    AttnKernel = "qt1_kt4"
	AttnQT1KT8    AttnKernel = "qt1_kt8"
	AttnQT2KT2    AttnKernel = "qt2_kt2"
	AttnQT1KT2W32 AttnKernel = "qt1_kt2_w32"
	AttnQT1KT4W32 AttnKernel = "qt1_kt4_w32"

	// The negative controls. Both produce a wrong answer by construction and
	// exist so that the tolerance the right answer is held to has been shown
	// to catch something.
	AttnNoCrossWave AttnKernel = "no_cross_wave"
	AttnNoRescale   AttnKernel = "no_rescale"
)

// DefaultAttnKernel is the ladder's winner at a 1024x1024 image: 15.1 ms and
// 36.4 TFLOP/s, against 23.6 for the same tiling at wave64. See
// research/stage-7-vae-mid-block.md.
const DefaultAttnKernel = AttnQT1KT4W32

// attnVariant is what the host has to know about a build that the SPIR-V does
// not carry: the two tile knobs, because they set the grid and the padding,
// and the wave size, because a multi-wave workgroup has to have it pinned --
// at wave32 the workgroup would hold twice the waves the kernel maps onto its
// component axis and half of them would write outside it.
type attnVariant struct {
	name  AttnKernel
	spirv []byte
	qt    int // query tiles per workgroup
	ktil  int // key tiles per block
	wave  uint32
}

var attnVariants = []attnVariant{
	{AttnQT1KT4, shaders.VAEAttentionWMMAQT1KT4, 1, 4, 64},
	{AttnQT1KT2, shaders.VAEAttentionWMMAQT1KT2, 1, 2, 64},
	{AttnQT1KT8, shaders.VAEAttentionWMMAQT1KT8, 1, 8, 64},
	{AttnQT2KT2, shaders.VAEAttentionWMMAQT2KT2, 2, 2, 64},
	{AttnQT1KT2W32, shaders.VAEAttentionWMMAQT1KT2W32, 1, 2, 32},
	{AttnQT1KT4W32, shaders.VAEAttentionWMMAQT1KT4W32, 1, 4, 32},
	{AttnNoCrossWave, shaders.VAEAttentionWMMANoCrossWave, 1, 4, 64},
	{AttnNoRescale, shaders.VAEAttentionWMMANoRescale, 1, 4, 64},
}

// AttnKernels lists every build in the ladder, in the order they are measured.
func AttnKernels() []AttnKernel {
	out := []AttnKernel{AttnScalar}
	for _, v := range attnVariants {
		out = append(out, v.name)
	}
	return out
}

func attnVariantFor(k AttnKernel) (attnVariant, bool) {
	for _, v := range attnVariants {
		if v.name == k {
			return v, true
		}
	}
	return attnVariant{}, false
}

// GEMMKernel names a build of shaders/dit_gemm.comp. Only the fragment-tile
// arms are offered: the weight is packed once at load, so the layout that
// costs the DiT a pass per step costs the VAE nothing, and stage 4a measured
// every other arm behind it.
type GEMMKernel string

const (
	GEMMWG128x256 GEMMKernel = "wg128x256_bt16_swz8"
	GEMMReg64     GEMMKernel = "reg64_bt16"
	GEMMReg64HKA4 GEMMKernel = "reg64_hka4_bt16"
	GEMMReg16x256 GEMMKernel = "reg16x256_bt16"
)

// DefaultGEMMKernel is the ladder's winner, by 4% over three others that tie.
// See research/stage-7-vae-mid-block.md.
const DefaultGEMMKernel = GEMMReg64HKA4

type gemmVariant struct {
	name   GEMMKernel
	spirv  []byte
	bm, bn int
	waves  int
}

var gemmVariants = []gemmVariant{
	{GEMMWG128x256, shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256, 4},
	{GEMMReg64, shaders.DiTGEMMReg64Tiled, 64, 64, 1},
	{GEMMReg64HKA4, shaders.DiTGEMMReg64HKA4Tiled, 64, 64, 1},
	{GEMMReg16x256, shaders.DiTGEMMReg16x256Tiled, 16, 256, 1},
}

// GEMMKernels lists every projection kernel, in the order they are measured.
func GEMMKernels() []GEMMKernel {
	out := make([]GEMMKernel, 0, len(gemmVariants))
	for _, v := range gemmVariants {
		out = append(out, v.name)
	}
	return out
}

func gemmVariantFor(k GEMMKernel) (gemmVariant, bool) {
	for _, v := range gemmVariants {
		if v.name == k {
			return v, true
		}
	}
	return gemmVariant{}, false
}

const (
	coopMatTile = 16
	// gemmRowAlign is the largest workgroup tile any offered GEMM build has
	// in the M direction. dit_gemm.comp has no bounds check -- it writes whole
	// tiles -- so every row count it sees is padded to this, and the
	// attention kernel's key blocks (at most 8 tiles) divide it.
	gemmRowAlign = 128
	// gemmPad is the pad on A's row stride, in halves (§2.3): a fragment load
	// of A issues 16 addresses one row apart, and an unpadded 512-column row
	// is 1 KB, an exact quarter of the 4 KB channel rotation.
	gemmPad = 128
)

func padRows(rows int) int { return (rows + gemmRowAlign - 1) &^ (gemmRowAlign - 1) }

// hasMatrixCores reports whether a device can run the fp16 path at all. The
// decoder falls back to stage 2b's kernels rather than failing, because every
// stage of this pipeline still has an fp32 answer and a device that cannot do
// coopmat should still decode.
func hasMatrixCores(dev *vk.Device) bool {
	f := dev.Features()
	if !f.Float16 || !f.CoopMatrix || !f.SubgroupSizeControl {
		return false
	}
	sgs, err := dev.Physical().SubgroupSizeControl()
	return err == nil && sgs.Supported && sgs.MinSubgroupSize <= 32 && sgs.MaxSubgroupSize >= 64
}

// packTiledB narrows a [n, k] row-major PyTorch weight into the 16x16
// fragment tiles dit_gemm.comp's B_LAYOUT=2 reads: tile (nt, kt) is 256
// contiguous halves holding element (k, n) at (n%16)*16 + k%16, tiles ordered
// kt-fastest so one n-tile's whole K row is contiguous. This is zimage/dit's
// packB, layout 2, without the chunking -- the VAE has four 512x512 weights
// where the DiT has 6.0e9 elements.
func packTiledB(dst []uint16, w []float32, n, k int) {
	kt := k / coopMatTile
	for i := 0; i < n; i++ {
		row := w[i*k : (i+1)*k]
		base := (i / coopMatTile) * kt * coopMatTile * coopMatTile
		lane := (i % coopMatTile) * coopMatTile
		for j, v := range row {
			dst[base+(j/coopMatTile)*coopMatTile*coopMatTile+lane+j%coopMatTile] = safetensors.F32ToF16(v)
		}
	}
}

// stageProjections packs the mid block's four projection weights into the
// fp16 weight arena. Nothing else in the decoder is narrowed: these are the
// only tensors a matrix-core kernel reads.
func (g *GPUDecoder) stageProjections(data []uint16, a *Attention) []uint16 {
	put := func(name string, l *Linear) {
		off := uint32(len(data))
		data = append(data, make([]uint16, l.Out*l.In)...)
		packTiledB(data[off:], l.Weight, l.Out, l.In)
		g.w16[name] = off
	}
	put("mid.attn.q", a.Q)
	put("mid.attn.k", a.K)
	put("mid.attn.v", a.V)
	put("mid.attn.out", a.Out)
	return data
}

// narrow records the fp32 -> fp16 pass that turns an activation into a GEMM A
// operand.
func (b *builder) narrow(in tensor, rows, rowsPad, lda int) uint32 {
	off := b.har.alloc(rowsPad * lda)
	b.label(fmt.Sprintf("narrow %dx%d", rows, in.C), 0)
	b.add("narrow", uint32(rowsPad), pushConstants{
		InOff: in.off, OutOff: off,
		C:    uint32(in.C),
		Aux0: uint32(rows),
		LDA:  uint32(lda),
	})
	return off
}

// proj records one projection: C = A * B with B the fragment-tiled fp16
// weight. The bias is not here -- dit_gemm.comp has none, and every one of
// these outputs is read next by a pass that can add it for free.
// The output carries the *padded* row count as its shape: the GEMM writes
// whole tiles, so the rows past the image are part of the allocation and have
// to be part of what the arena reclaims.
func (b *builder) proj(name string, l *Linear, aOff uint32, rows, rowsPad, lda int) tensor {
	v := b.dec.gemm
	out := tensor{off: b.ar.alloc(rowsPad * l.Out), C: l.Out, H: rowsPad, W: 1}
	b.label(fmt.Sprintf("gemm %d->%d x%d", l.In, l.Out, rows),
		2*float64(rowsPad)*float64(l.Out)*float64(l.In))
	b.addXY("gemm", uint32(l.Out/v.bn), uint32(rowsPad/v.bm), pushConstants{
		InOff: aOff, OutOff: out.off,
		GemmB: b.b16Off(name), GemmM: uint32(rowsPad),
		GemmN: uint32(l.Out), GemmK: uint32(l.In),
		LDA: uint32(lda),
	})
	return out
}

// pack records the fragment-tile pack of one attention operand, adding the
// projection's bias and folding in a scale.
func (b *builder) pack(pipe, kind string, in tensor, off uint32, rows, rowsPad int, bias uint32, scale float32) {
	b.label(kind, 0)
	b.add(pipe, uint32(rowsPad/coopMatTile), pushConstants{
		InOff: in.off, OutOff: off,
		Aux0: uint32(rows), Aux1: math.Float32bits(scale), Aux2: bias,
	})
}

// attentionWMMA is the mid block with its projections on dit_gemm.comp and
// its softmax on vae_attention_wmma.comp. Same graph as builder.attention,
// dispatch for dispatch, except that the scalar kernel's K transpose is
// subsumed by the pack (a tile of k is stored the same way as a tile of q,
// because both matmuls reduce over the component axis) and two narrowings
// appear, which is what fp16 operands cost.
func (b *builder) attentionWMMA(name string, a *Attention, x tensor) tensor {
	rows := x.H * x.W
	dim := a.Q.Out
	rowsPad := padRows(rows)
	lda := dim + gemmPad

	h := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.add("groupnorm", uint32(a.GroupNorm.Groups), pushConstants{
		InOff: x.off, OutOff: h.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		WOff: b.wOff(name + ".norm.weight"), BOff: b.wOff(name + ".norm.bias"),
		Groups: uint32(a.GroupNorm.Groups), Aux0: math.Float32bits(float32(a.GroupNorm.Eps)),
	})

	seq := tensor{off: b.ar.alloc(rows * x.C), C: x.C, H: x.H, W: x.W}
	b.add("to_rows", groups(rows*x.C, 256), pushConstants{
		InOff: h.off, OutOff: seq.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
	})

	hA := b.narrow(seq, rows, rowsPad, lda)
	q := b.proj(name+".q", a.Q, hA, rows, rowsPad, lda)
	k := b.proj(name+".k", a.K, hA, rows, rowsPad, lda)
	v := b.proj(name+".v", a.V, hA, rows, rowsPad, lda)
	b.har.release(hA, rowsPad*lda)

	// The packed planes are addressed in tiles and padded to the workgroup
	// tile, so the key loop can read whole blocks past the last real row: a
	// pad row is zeroed, and a zero key scores zero, which the kernel then
	// masks out of the denominator anyway.
	plane := rowsPad * dim
	hQ, hK, hV := b.har.alloc(plane), b.har.alloc(plane), b.har.alloc(plane)
	// log2(e) folds into q with the softmax scale so the kernel's exponential
	// is exp2, which is one instruction on this ISA.
	const log2e = 1.4426950408889634
	b.pack("pack", "pack q", q, hQ, rows, rowsPad, b.bOff(name+".q.bias"), float32(a.Scale*log2e))
	b.pack("pack", "pack k", k, hK, rows, rowsPad, b.bOff(name+".k.bias"), 1)
	b.pack("packt", "pack v", v, hV, rows, rowsPad, b.bOff(name+".v.bias"), 1)

	ctx := tensor{off: b.ar.alloc(rowsPad * dim), C: dim, H: rowsPad, W: 1}
	b.label(fmt.Sprintf("attention %d rows x %d", rows, dim),
		2*2*float64(rows)*float64(rows)*float64(dim))
	b.add("attn_wmma", uint32((rows+b.dec.attn.qt*coopMatTile-1)/(b.dec.attn.qt*coopMatTile)), pushConstants{
		InOff: hQ, OutOff: ctx.off, ResOff: hK, Aux2: hV,
		C:    uint32(dim),
		Aux0: uint32(rows),
	})
	b.har.release(hQ, plane)
	b.har.release(hK, plane)
	b.har.release(hV, plane)

	hCtx := b.narrow(ctx, rows, rowsPad, lda)
	outRows := b.proj(name+".out", a.Out, hCtx, rows, rowsPad, lda)
	b.har.release(hCtx, rowsPad*lda)

	res := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.add("to_nchw", groups(x.elems(), 256), pushConstants{
		InOff: outRows.off, OutOff: res.off, ResOff: x.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		Aux0: b.bOff(name + ".out.bias"),
	})
	b.release(h, seq, q, k, v, ctx, outRows, x)
	return res
}

// b16Off resolves a projection weight in the fp16 arena.
func (b *builder) b16Off(name string) uint32 {
	off, ok := b.e.w16[name]
	if !ok && b.err == nil {
		b.err = fmt.Errorf("vae: no fp16 weight %q", name)
	}
	return off
}
