package dit

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// The transformer's head and tail on the device -- PIPELINE.md stage 9.
//
// After stage 8 the image was 94% denoising, and the largest single item left
// anywhere in the pipeline was not a kernel: it was the *boundary*. The patch
// embedder and the final layer ran on the host, so every denoising step wrote
// a [4128, 3840] fp32 residual stream across the bus and read it back --
// 63 MB each way for a latent that is 1 MB -- and spent 58 ms of CPU on two
// GEMMs the device does in one.
//
// Both GEMMs are shapes the block's own kernel already covers, so this file
// adds no matrix-core code. What it adds is two ideas:
//
//   - **The bias and the pad token are extra columns of K.** dit_gemm.comp
//     has no bias, and rather than fork it or add an elementwise pass, the
//     patch embedder's A operand carries two more columns: one holding 1 for
//     a real token, and one holding 1 for a padded token. B's matching rows
//     are the embedder's bias and the learned `x_pad_token`. So `y = Wx + b`
//     for the image rows and `y = x_pad_token` for the pad rows come out of
//     the same GEMM, with no branch, no second dispatch and no host write of
//     the padding. It costs K = 64 -> 128, which is 2 GFLOP on a 1.8 s step.
//   - **The final layer's bias goes where the host already is.** Its output
//     is 64 values per token and is read back to be unpatchified, so the CPU
//     adds the bias while it is copying -- which is stage 7's rule (put a
//     bias in the pass that is already reading the tensor) reaching across
//     the device boundary.
//
// The head borrows the stack's two activation arenas: the patch embedder
// *writes* the residual stream and the final layer *reads* it, so they are
// the same tensor and therefore the same buffer. Its own two buffers hold
// only weights -- 4 MB of fp32 adaLN projection and 1.5 MB of fp16 GEMM
// operands, against the stack's 12.54 GB.
//
// The CPU implementation in head.go is untouched and is both the oracle
// (gpuhead_test.go) and the fallback.

// headEmbedK is the patch embedder's reduction extent: the patch itself, the
// bias column, the pad-token column, and then rounded up to the 64-wide K
// slab every offered GEMM build steps by (BK_TILES=4).
func headEmbedK(patchDim int) int { return (patchDim + 2 + 63) &^ 63 }

// headEmbedKernel and headFinalKernel are the builds the two shapes need.
// Neither is a tuning choice -- they are the only rungs whose tiles divide
// the shapes. The embedder is [tokens, K] x [K, 3840] and the tail is
// [tokens, 3840] x [3840, 64], so BN has to divide 3840 and then 64, and BM
// has to divide a token count that is only guaranteed to be a multiple of
// SeqMultiOf = 32. A 16-row tile is what satisfies that at both ends.
var (
	headEmbedVariant = gemmVariant{
		name: "reg16x256_bt16", spirv: shaders.DiTGEMMReg16x256Tiled,
		bm: 16, bn: 256, waves: 1, layout: 2,
	}
	headFinalVariant = gemmVariant{
		name: "reg16x64_bt16", spirv: shaders.DiTGEMMReg16x64Tiled,
		bm: 16, bn: 64, waves: 1, layout: 2,
	}
)

// GPUHead runs the patch embedder and the final layer on the device, into and
// out of a GPUStack's residual stream.
type GPUHead struct {
	dev   *vk.Device
	stack *GPUStack
	head  *Head

	wbuf *vk.Buffer // fp32: the final layer's adaLN projection and its bias
	w16  *vk.Buffer // fp16: the two GEMM weights as 16x16 fragment tiles

	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	wAda, wAdaBias uint32
	bEmbed, bFinal uint32
	embedK         int

	// noMean selects the control build of the final norm. Zero in every real
	// use; gpuhead_test.go says what it is for.
	noMean bool
}

// NewGPUHead stages the head's weights and builds its four pipelines against
// an existing stack.
func NewGPUHead(g *GPUStack, h *Head) (*GPUHead, error) {
	return newGPUHead(g, h, false)
}

func newGPUHead(g *GPUStack, h *Head, noMean bool) (*GPUHead, error) {
	if g.patchDim == 0 {
		return nil, fmt.Errorf("dit: the stack was built for a config with no patch size")
	}
	if h.PatchDim() != g.patchDim {
		return nil, fmt.Errorf("dit: head's patch is %d values, the stack's arena is sized for %d",
			h.PatchDim(), g.patchDim)
	}
	if h.Dim != g.dim {
		return nil, fmt.Errorf("dit: head is %d wide, the stack is %d", h.Dim, g.dim)
	}
	if h.FinalAda.In != g.adaIn {
		return nil, fmt.Errorf("dit: final adaLN takes %d, the stack's timestep embedding is %d",
			h.FinalAda.In, g.adaIn)
	}
	if h.FinalAda.Bias == nil || h.XEmbed.Bias == nil || h.FinalLin.Bias == nil {
		return nil, fmt.Errorf("dit: the head's three linears must all have biases")
	}
	p := &GPUHead{
		dev: g.dev, stack: g, head: h,
		pipes:  make(map[string]*vk.ComputePipeline),
		embedK: headEmbedK(g.patchDim),
		noMean: noMean,
	}
	if err := p.stageWeights(); err != nil {
		p.Destroy()
		return nil, err
	}
	if err := p.build(); err != nil {
		p.Destroy()
		return nil, err
	}
	return p, nil
}

// stageWeights narrows the two GEMM operands into fragment tiles and copies
// the final adaLN projection as it is.
func (p *GPUHead) stageWeights() error {
	h, g := p.head, p.stack
	var f32 []float32
	put32 := func(v []float32) uint32 {
		off := uint32(len(f32))
		f32 = append(f32, v...)
		return off
	}
	p.wAda = put32(h.FinalAda.Weight)
	p.wAdaBias = put32(h.FinalAda.Bias)

	// The embedder's B is [dim, embedK]: the weight, then the bias column,
	// then the pad-token column, then zeros to the slab. Built as one
	// row-major matrix so the stack's own packB does the tiling.
	wide := make([]float32, g.dim*p.embedK)
	for o := 0; o < g.dim; o++ {
		copy(wide[o*p.embedK:], h.XEmbed.Weight[o*h.XEmbed.In:(o+1)*h.XEmbed.In])
		wide[o*p.embedK+h.XEmbed.In] = h.XEmbed.Bias[o]
		wide[o*p.embedK+h.XEmbed.In+1] = h.XPad[o]
	}

	nEmbed := bElems(g.dim, p.embedK, headEmbedVariant.layout)
	nFinal := bElems(h.FinalLin.Out, h.FinalLin.In, headFinalVariant.layout)
	f16 := make([]uint16, nEmbed+nFinal)
	p.bEmbed, p.bFinal = 0, uint32(nEmbed)
	packB(f16[:nEmbed], wide, g.dim, p.embedK, headEmbedVariant.layout)
	packB(f16[nEmbed:], h.FinalLin.Weight, h.FinalLin.Out, h.FinalLin.In, headFinalVariant.layout)

	var err error
	if p.wbuf, err = p.dev.NewBuffer(len(f32) * 4); err != nil {
		return fmt.Errorf("dit: head fp32 weights (%d MB): %w", (len(f32)*4)>>20, err)
	}
	p.wbuf.WriteFloat32(f32)
	if p.w16, err = p.dev.NewBuffer(len(f16) * 2); err != nil {
		return fmt.Errorf("dit: head fp16 weights (%d MB): %w", (len(f16)*2)>>20, err)
	}
	p.w16.WriteUint16At(0, f16)
	return nil
}

// build compiles the four pipelines. Bindings 1 and 2 are the *stack's*
// arenas and 0 and 3 are the head's own, which is the whole trick: a dispatch
// reads its weights from here and the residual stream from there.
func (p *GPUHead) build() error {
	g := p.stack
	bufs := []*vk.Buffer{p.wbuf, g.abuf, g.hbuf, p.w16}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	norm := shaders.DiTFinalNorm
	if p.noMean {
		norm = shaders.DiTFinalNormNoMean
	}
	set := map[string][]byte{
		"adaln":     shaders.DiTAdaLN,
		"finalnorm": norm,
		"embed":     headEmbedVariant.spirv,
		"final":     headFinalVariant.spirv,
	}
	for name, spirv := range set {
		mod, err := p.dev.NewShaderModule(spirv)
		if err != nil {
			return fmt.Errorf("dit: head shader %s: %w", name, err)
		}
		p.mods = append(p.mods, mod)
		pipe, err := p.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("dit: head pipeline %s: %w", name, err)
		}
		p.pipes[name] = pipe
	}
	return nil
}

// Destroy releases the head's own Vulkan objects. The stack's are not its.
func (p *GPUHead) Destroy() {
	for _, pipe := range p.pipes {
		pipe.Destroy()
	}
	for _, m := range p.mods {
		m.Destroy()
	}
	if p.wbuf != nil {
		p.wbuf.Destroy()
	}
	if p.w16 != nil {
		p.w16.Destroy()
	}
}

// Embed runs the patch embedder over patches -- Head.Patchify's output, which
// stays on the host because it is index arithmetic on a megabyte -- and
// leaves the padded image stream in the stack's residual stream.
//
// total is the padded length, and the rows between patches.Rows and it come
// out as the learned pad token because of the second K column, not because
// anything writes them.
func (p *GPUHead) Embed(patches *Mat, total int) error {
	g := p.stack
	if patches.Cols != g.patchDim {
		return fmt.Errorf("dit: patches are %s, want %d columns", patches, g.patchDim)
	}
	if total%headEmbedVariant.bm != 0 {
		return fmt.Errorf("dit: a %d-row image stream is not a multiple of the %d-row tile",
			total, headEmbedVariant.bm)
	}
	if total > g.tokens || patches.Rows > total {
		return fmt.Errorf("dit: %d image tokens padded to %d, in a stack built for %d",
			patches.Rows, total, g.tokens)
	}

	// The A operand, narrowed on the host: the patch, a 1 in the bias column
	// for a real token, a 1 in the pad column for a padded one. A host write
	// into this arena runs at 11.5 GB/s and this is 2 MB.
	lda := int(g.hPatchLDA)
	a := make([]uint16, total*lda)
	one := safetensors.F32ToF16(1)
	for r := 0; r < total; r++ {
		row := a[r*lda:]
		if r < patches.Rows {
			for c, v := range patches.Row(r) {
				row[c] = safetensors.F32ToF16(v)
			}
			row[g.patchDim] = one
		} else {
			row[g.patchDim+1] = one
		}
	}
	g.hbuf.WriteUint16At(int(g.hPatch), a)
	g.rows = total

	pc := pushConstants{
		InOff: g.hPatch, OutOff: g.aX, BOff: p.bEmbed,
		GemmM: uint32(total), GemmN: uint32(g.dim), GemmK: uint32(p.embedK),
		LDA: uint32(lda),
	}
	v := headEmbedVariant
	return submit([]vk.MultiDispatch{{
		Pipeline: p.pipes["embed"],
		GroupsX:  uint32(g.dim / v.bn), GroupsY: uint32(total / v.bm),
		PushConstants: pc.bytes(),
	}})
}

// Final runs the transformer's tail over the first `rows` of the residual
// stream and returns [rows, patchDim] with the projection's bias added.
//
// Three dispatches: the final layer's own adaLN projection (the block kernel,
// unmodified -- its chunk 0 is `1 + v`, which is exactly the scale this layer
// wants), the LayerNorm-scale-narrow pass, and the GEMM. Then one read of
// rows*64 floats, which is the whole of what crosses the bus per step.
func (p *GPUHead) Final(rows int) (*Mat, error) {
	g, h := p.stack, p.head
	if rows <= 0 || rows > g.rows {
		return nil, fmt.Errorf("dit: %d rows of a %d-row run", rows, g.rows)
	}
	if rows%headFinalVariant.bm != 0 {
		return nil, fmt.Errorf("dit: %d rows is not a multiple of the %d-row tile",
			rows, headFinalVariant.bm)
	}
	v := headFinalVariant
	pcAda := pushConstants{
		InOff: g.aAdaSiLU, OutOff: g.aFinalMod, WOff: p.wAda, Aux0: p.wAdaBias,
		Dim: uint32(g.dim), GemmN: uint32(g.dim), GemmK: uint32(g.adaIn),
	}
	pcNorm := pushConstants{
		InOff: g.aX, OutOff: g.hA, Aux0: g.aFinalMod,
		Dim: uint32(g.dim), Tokens: uint32(rows),
		Eps: math.Float32bits(float32(h.FinalEps)), LDA: uint32(g.ldaDim),
	}
	pcGemm := pushConstants{
		InOff: g.hA, OutOff: g.aOut, BOff: p.bFinal,
		GemmM: uint32(rows), GemmN: uint32(g.patchDim), GemmK: uint32(g.dim),
		LDA: uint32(g.ldaDim),
	}
	d := []vk.MultiDispatch{
		{Pipeline: p.pipes["adaln"], GroupsX: uint32(g.dim), GroupsY: 1, PushConstants: pcAda.bytes()},
		{Pipeline: p.pipes["finalnorm"], GroupsX: uint32(rows), GroupsY: 1, PushConstants: pcNorm.bytes()},
		{Pipeline: p.pipes["final"], GroupsX: uint32(g.patchDim / v.bn), GroupsY: uint32(rows / v.bm),
			PushConstants: pcGemm.bytes()},
	}
	if err := submit(d); err != nil {
		return nil, err
	}
	out := NewMat(rows, g.patchDim)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.aOut), rows*g.patchDim))
	// The projection's bias, added by the pass that was going to touch every
	// one of these values anyway.
	for r := 0; r < rows; r++ {
		row := out.Row(r)
		for c, b := range h.FinalLin.Bias {
			row[c] += b
		}
	}
	return out, nil
}
