package dit

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU runs the Qwen-Image-2.1 DiT on the device: the joint-sequence graph of
// Q2's CPU model as ~26 dispatches per block over shared arenas, in the two
// row regimes the sampler uses — the step-0 prefill over the whole joint
// sequence, and the cached decode that recomputes only the target rows
// against each block's stored prefix K/V.
//
// It is zimage/dit's GPUStack re-plumbed for what 2.1 changes, reusing that
// stage's shaders and their validated semantics wherever the two models
// agree (the GEMM and attention kernels, the packs, SwiGLU, the gated
// residual) and mapping the differences onto existing kernels rather than
// new ones:
//
//   - the block's two pre-norms are LayerNorms, not RMS norms — which is
//     dit_final_norm.comp exactly (mean subtracted, non-affine, times a
//     precomputed scale vector, narrowed to the fp16 A operand), so every
//     norm+modulate+narrow in this graph is that one shader;
//   - the shared modulation is two rows of host arithmetic per step, not a
//     per-block GEMM. The host uploads ten vectors — (1+scale), tanh(gate)
//     twice over, and the tail's (1+scale), for the sampled-t row and the
//     t=0 row — and the per-token row *selection* (prefix reads t=0, target
//     reads t) is expressed as two dispatch ranges of the same pipeline,
//     since both groups are contiguous;
//   - the prefill's block-causal mask is "every row attends to a contiguous
//     key prefix": its own index for a text token, its image block's last
//     index for an image token, the whole sequence for the target. The WMMA
//     kernel runs bidirectionally over the whole sequence — already right
//     for the target's thousands of rows — and the prefix is then repaired
//     segment by segment, *last segment first*, because an image segment's
//     pass necessarily recomputes every row before it. Text segments take
//     dit_attn_causal.comp, whose online softmax carries no cap on the key
//     range; image segments take the same WMMA kernel with a smaller key
//     count. For t2i the whole of that is one causal pass over the prompt;
//   - the prefix KV cache is each block's post-RoPE k and v prefix rows,
//     saved at the prefill and copied back under the fresh target rows each
//     cached step before the fragment pack rebuilds the planes. It lives in
//     **fp16 banks of its own** (dit_kvcache.comp) rather than in the
//     activation arena: a single 1024² condition image's prefix is 4.34 GB
//     of fp32 k and v across the 32 blocks, past this device's 4 GiB
//     single-buffer limit. The narrowing is free, because the pack narrows
//     those rows to halves either way;
//   - SwiGLU's output is scaled by ffScale (a power of two) into fp16 and
//     the inverse rides in the uploaded MLP gate vectors: 2.1's down
//     projection feeds a gated residual, not a norm, so unlike z-image the
//     scale must be undone — and multiplying tanh(gate) by 16 on the host
//     undoes it exactly.
//
// Everything the matrix cores touch is fp16 with fp32 accumulators; what
// that costs against the fp32 CPU oracle is measured in gpu_test.go, not
// assumed.
type GPU struct {
	dev  *vk.Device
	head *Model // 0-block CPU model: txt_in, timestep, modulation, tail scale

	wbuf  *vk.Buffer // fp32: rope table, per-block q/k norm weights
	abuf  *vk.Buffer // fp32 activations
	hbuf  *vk.Buffer // fp16 activations and fragment planes
	banks []*vk.Buffer

	pipes map[string]*vk.ComputePipeline
	gemms []map[gemmKernel]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	attn attnVariant

	dim, ffn, heads, headDim      int
	maxTokens, maxPrefix, maxText int
	tokPad                        int // padded arena rows
	ldaDim, ldaFFN, ldaLat        int
	eps                           float64

	wCos, wSin uint32
	blocks     []gpuBlockW

	// fp32 arena offsets.
	aX, aQ, aK, aV, aCtx, aAttn uint32
	aGate, aUp, aFF             uint32
	aMod, aLat, aOut, aTxt      uint32
	actElems                    int

	// The prefix KV cache, fp16, in banks of its own (dit_kvcache.comp).
	kvBanks []*vk.Buffer
	kvPipes []*vk.ComputePipeline

	// noPrefixRepair drops the prefill's segment passes, leaving every
	// prefix row attending bidirectionally over the whole joint sequence.
	// It is a negative control and nothing else — the mask is the
	// difference between an edit and a sequence of unrelated pixels, and it
	// is invisible in every shape the graph checks.
	noPrefixRepair bool

	// fp16 arena offsets.
	hA, hQ, hK, hV, hCtx, hFFN, hLat uint32
	hElems                           int

	// Head projections, staged like a block's.
	imgInOff, projOutOff uint32
	outChannels          int

	// Per-image state, set by BeginImage.
	lay      *Layout
	txtP     *qwen.Mat
	p, s     int // prefix rows, joint rows
	runs     []prefixRun
	condRows int
}

// prefixRun is one of the prefix's segments as the graph addresses it: `n`
// rows landing at joint-sequence row `row`, sourced from row `src` of either
// the projected text (aTxt) or the packed condition latents (aLat, past the
// target's rows).
type prefixRun struct {
	row, n, src int
	text        bool
}

// gpuBlockW is one block's arena offsets.
type gpuBlockW struct {
	normQ, normK uint32
	bank         int
	bOff         map[proj]uint32
	// The prefix KV cache: k at kvOff of bank kvBank, v maxPrefix*dim halves
	// after it.
	kvBank int
	kvOff  uint32
}

type proj string

const (
	projQ  proj = "q"
	projK  proj = "k"
	projV  proj = "v"
	projO  proj = "o"
	projW1 proj = "w1" // img_mlp.gate_layer, the silu'd half
	projW3 proj = "w3" // img_mlp.proj, the linear half
	projW2 proj = "w2" // img_mlp.out
)

var projOrder = []proj{projQ, projK, projV, projO, projW1, projW3, projW2}

// pushConstants mirrors shaders/dit_common.glsl, as zimage/dit's does.
type pushConstants struct {
	InOff, OutOff, WOff uint32
	Tokens, Dim, Heads  uint32
	HeadDim, Span       uint32
	KOff, VOff, KStride uint32
	Eps, Scale          uint32
	Aux0, Aux1, Aux2    uint32
	BOff                uint32
	GemmM, GemmN, GemmK uint32
	LDA, LDB            uint32
}

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// gemmKernel names a dit_gemm.comp build. Two are enough here: the crown
// kernel for every big projection (its 256-wide tile divides 4096 and
// 12288), and the 64x64 one for proj_out's N=64. Both read fragment-tiled
// weights, so packB below has one layout.
type gemmKernel string

const (
	gemmSWZ8  gemmKernel = "wg128x256_bt16_swz8"
	gemmReg64 gemmKernel = "reg64_bt16"
)

type gemmVariant struct {
	name   gemmKernel
	spirv  []byte
	bm, bn int
	waves  int
}

var gemmVariants = map[gemmKernel]gemmVariant{
	gemmSWZ8:  {name: gemmSWZ8, spirv: shaders.DiTGEMMWG128x256TiledSWZ8, bm: 128, bn: 256, waves: 4},
	gemmReg64: {name: gemmReg64, spirv: shaders.DiTGEMMReg64Tiled, bm: 64, bn: 64, waves: 1},
}

// attnVariant is the attention build, as in zimage/dit.
type attnVariant struct {
	name  string
	spirv []byte
	qt    int
	wave  uint32
}

func (v attnVariant) rows() int { return v.qt * coopMatTile }

const (
	coopMatTile    = 16
	wmmaTokenAlign = 128
	gemmPad        = 128
	log2e          = 1.4426950408889634
	// ffScale scales SwiGLU's output into fp16 and its inverse is folded
	// into the uploaded MLP gate vectors — see the type comment. A power of
	// two, so the round trip moves no mantissa bit.
	ffScale = 1.0 / 16
)

// NewGPU stages the transformer for sequences up to maxTokens (joint rows)
// with prefixes up to maxPrefix rows, of which at most maxText are text —
// loading blocks one at a time, ~14 GB of fp16 banks for the 32 blocks plus
// the prefix KV cache's own.
//
// The two prefix numbers are one number for t2i, where the prefix *is* the
// prompt. They part company for an edit: a condition image contributes
// thousands of latent rows to the prefix and none to the text, and the two
// sit in different places (the cache is sized by the first, the projected
// text staging by the second).
func NewGPU(dev *vk.Device, dir string, maxTokens, maxPrefix, maxText int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	head, err := Load(dir, 0)
	if err != nil {
		return nil, err
	}
	if maxText > maxPrefix {
		return nil, fmt.Errorf("dit: maxText %d exceeds maxPrefix %d", maxText, maxPrefix)
	}
	g := &GPU{
		dev: dev, head: head,
		pipes: make(map[string]*vk.ComputePipeline),
		dim:   cfg.Dim(), ffn: cfg.Dim() * cfg.MLPRatio,
		heads: cfg.NumHeads, headDim: cfg.HeadDim,
		maxTokens: maxTokens, maxPrefix: maxPrefix, maxText: maxText,
		eps:         cfg.Eps,
		outChannels: cfg.OutChannels,
	}
	g.tokPad = (maxTokens+wmmaTokenAlign-1)&^(wmmaTokenAlign-1) + wmmaTokenAlign
	g.ldaDim = g.dim + gemmPad
	g.ldaFFN = g.ffn + gemmPad
	g.ldaLat = cfg.InChannels + gemmPad

	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	if err := g.layoutAndStage(set, cfg); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.allocActivations(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// maxBankBytes is the device's storage-buffer ceiling, as in zimage/dit.
const maxBankBytes = 0xfffffffc

// packB narrows a [n, k] row-major weight into 16x16 fragment tiles,
// kt-fastest — dit_pack_f16.comp's layout, zimage/dit's packB layout 2.
func packB(dst []uint16, w []float32, n, k int) {
	const tile = coopMatTile
	kt := k / tile
	parallelHeads((n+63)/64, func(c int) {
		for i := c * 64; i < min((c+1)*64, n); i++ {
			row := w[i*k : (i+1)*k]
			base := (i / tile) * kt * tile * tile
			lane := (i % tile) * tile
			for j, v := range row {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

func (g *GPU) projShape(r proj) [2]int {
	switch r {
	case projW1, projW3:
		return [2]int{g.ffn, g.dim}
	case projW2:
		return [2]int{g.dim, g.ffn}
	default:
		return [2]int{g.dim, g.dim}
	}
}

// layoutAndStage plans the fp32/fp16 weight arenas and fills them, one block
// at a time.
func (g *GPU) layoutAndStage(set *safetensors.Set, cfg *Config) error {
	// fp32: the rotary table (sized for maxTokens) and per-block q/k norms.
	ropeElems := g.maxTokens * (g.headDim / 2)
	g.wCos, g.wSin = 0, uint32(ropeElems)
	total32 := 2 * ropeElems

	perBlock16 := 0
	for _, r := range projOrder {
		sh := g.projShape(r)
		perBlock16 += sh[0] * sh[1]
	}
	headElems := g.dim*cfg.InChannels + g.outChannels*g.dim

	g.blocks = make([]gpuBlockW, cfg.NumLayers)
	var bankElems []int
	cur := -1
	newBank := func(need int) {
		if cur < 0 || (bankElems[cur]+need)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
	}
	// The head's two projections first, in bank 0.
	newBank(headElems + perBlock16)
	g.imgInOff = uint32(bankElems[cur])
	bankElems[cur] += g.dim * cfg.InChannels
	g.projOutOff = uint32(bankElems[cur])
	bankElems[cur] += g.outChannels * g.dim

	for i := range g.blocks {
		w := &g.blocks[i]
		w.normQ = uint32(total32)
		w.normK = w.normQ + uint32(g.headDim)
		total32 += 2 * g.headDim
		newBank(perBlock16)
		w.bank = cur
		w.bOff = make(map[proj]uint32, len(projOrder))
		for _, r := range projOrder {
			sh := g.projShape(r)
			w.bOff[r] = uint32(bankElems[cur])
			bankElems[cur] += sh[0] * sh[1]
		}
	}

	var err error
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("dit: fp32 weight arena: %w", err)
	}
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("dit: fp16 bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}

	// Stage the head projections.
	stage := func(bank int, off uint32, w []float32, n, k int) {
		buf := make([]uint16, n*k)
		packB(buf, w, n, k)
		g.banks[bank].WriteUint16At(int(off), buf)
	}
	stage(0, g.imgInOff, g.head.ImgIn.Weight, g.dim, cfg.InChannels)
	stage(0, g.projOutOff, g.head.ProjOut.Weight, g.outChannels, g.dim)

	// Stage the blocks, one at a time — a block is 872 MB as fp32 on the
	// host and none of it is wanted once narrowed.
	for i := range g.blocks {
		blk, err := LoadBlock(set, i, cfg)
		if err != nil {
			return err
		}
		w := &g.blocks[i]
		g.wbuf.WriteFloat32At(int(w.normQ), blk.QNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.normK), blk.KNorm.Weight)
		lins := map[proj]*qwen.Linear{
			projQ: blk.Q, projK: blk.K, projV: blk.V, projO: blk.O,
			projW1: blk.GateL, projW3: blk.Proj, projW2: blk.Out,
		}
		for _, r := range projOrder {
			lin := lins[r]
			sh := g.projShape(r)
			if lin.Out != sh[0] || lin.In != sh[1] {
				return fmt.Errorf("dit: block %d %s is [%d %d], want %v", i, r, lin.Out, lin.In, sh)
			}
			stage(w.bank, w.bOff[r], lin.Weight, lin.Out, lin.In)
		}
	}
	return nil
}

func (g *GPU) allocActivations() error {
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	rows := g.tokPad
	g.aX = alloc(rows * g.dim)
	g.aQ = alloc(rows * g.dim)
	g.aK = alloc(rows * g.dim)
	g.aV = alloc(rows * g.dim)
	g.aCtx = alloc(rows * g.dim)
	g.aAttn = alloc(rows * g.dim)
	g.aGate = alloc(rows * g.ffn)
	g.aUp = alloc(rows * g.ffn)
	g.aFF = alloc(rows * g.dim)
	g.aMod = alloc(10 * g.dim)
	g.aLat = alloc(rows * 64)
	g.aOut = alloc(rows * g.outChannels)
	// The projected text rows, staged here rather than written straight into
	// the residual stream: the img_in GEMMs that fill the prefix's *image*
	// rows write whole 128-row tiles and so overrun into whatever follows
	// them, and a host write cannot be ordered after a dispatch that has not
	// been submitted yet. One copy dispatch per text run does order.
	g.aTxt = alloc(g.maxText * g.dim)

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	plane := g.heads * g.tokPad * g.headDim
	g.hA = halloc(rows * g.ldaDim)
	g.hQ = halloc(plane)
	g.hK = halloc(plane)
	g.hV = halloc(plane)
	g.hCtx = halloc(rows * g.ldaDim)
	g.hFFN = halloc(rows * g.ldaFFN)
	g.hLat = halloc(rows * g.ldaLat)

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("dit: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("dit: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zero the fp16 arena once: pad rows and columns of every A operand and
	// the attention tail read from here.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	return g.allocCache()
}

// allocCache lays out the prefix KV cache: per block, k and v over maxPrefix
// rows as halves. At a t2i prompt that is 40 MB in total; at one 1024²
// condition image it is 2.2 GB, which is why it is banked like the weights
// rather than allocated inside the activation arena — one storage buffer
// stops at 4 GiB on this device and the fp32 form of the same cache is 4.34.
func (g *GPU) allocCache() error {
	perBlock := 2 * g.maxPrefix * g.dim
	if perBlock*2 > maxBankBytes {
		return fmt.Errorf("dit: one block's KV cache for a %d-row prefix is %d MB, past the %d MB buffer limit",
			g.maxPrefix, (perBlock*2)>>20, maxBankBytes>>20)
	}
	var bankElems []int
	cur := -1
	for i := range g.blocks {
		if cur < 0 || (bankElems[cur]+perBlock)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
		g.blocks[i].kvBank = cur
		g.blocks[i].kvOff = uint32(bankElems[cur])
		bankElems[cur] += perBlock
	}
	for i, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("dit: KV cache bank %d (%d MB): %w", i, (n*2)>>20, err)
		}
		g.kvBanks = append(g.kvBanks, b)
	}
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("dit: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("dit: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

func (g *GPU) build() error {
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	for name, spirv := range map[string][]byte{
		"final_norm": shaders.DiTFinalNorm,
		"gate":       shaders.DiTGateAdd,
		"scale":      shaders.DiTScaleF16,
		"rmsnorm":    shaders.DiTRMSNorm,
		"rope":       shaders.DiTRoPE,
		"qkpack":     shaders.DiTQKPack,
		"pack":       shaders.DiTPackF16,
		"swiglu":     shaders.DiTSwiGLUF16,
		"causal":     shaders.DiTAttnCausal,
		"copy":       shaders.DiTCopy,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}

	// The attention kernel: wave32 where the size can be pinned, wave64
	// otherwise — zimage/dit's measured ladder, first and second place.
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("dit: subgroup size control: %w", err)
	}
	g.attn = attnVariant{name: "attention", spirv: shaders.DiTAttentionWMMAQT1KT4W32, qt: 1, wave: 32}
	tailMax := shaders.DiTAttentionWMMAQT1KT4W32TailMax
	if !feat.SubgroupSizeControl || !sgs.Supported || 32 < sgs.MinSubgroupSize || 32 > sgs.MaxSubgroupSize {
		g.attn = attnVariant{name: "attention", spirv: shaders.DiTAttentionWMMAQT1KT4, qt: 1}
		tailMax = shaders.DiTAttentionWMMAQT1KT4TailMax
	}
	if err := g.pipeline(g.attn.name, g.attn.spirv, vk.PipelineSpec{
		Buffers: base, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
	}); err != nil {
		return err
	}
	// The same kernel with the tail mask reaching the row max, for the edit
	// prefill's image-segment repair: it runs with the key count cut to a
	// condition block's end, inside a plane whose later rows are real and
	// large, and the plain build lets those set the scale every real weight
	// is then measured against before P is narrowed to fp16.
	if err := g.pipeline("prefix_attn", tailMax, vk.PipelineSpec{
		Buffers: base, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
	}); err != nil {
		return err
	}

	// One KV-cache pipeline per cache bank: which buffer a dispatch reads or
	// writes is in its descriptor set, not in its push constants.
	for b := range g.kvBanks {
		mod, err := g.dev.NewShaderModule(shaders.DiTKVCache)
		if err != nil {
			return fmt.Errorf("dit: kvcache shader: %w", err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.kvBanks[b]},
			PushConstantSize: pcSize,
		})
		if err != nil {
			return fmt.Errorf("dit: kvcache pipeline on bank %d: %w", b, err)
		}
		g.kvPipes = append(g.kvPipes, pipe)
	}

	// GEMM pipelines, per bank and kernel.
	for b := range g.banks {
		m := make(map[gemmKernel]*vk.ComputePipeline, len(gemmVariants))
		for name, v := range gemmVariants {
			mod, err := g.dev.NewShaderModule(v.spirv)
			if err != nil {
				return fmt.Errorf("dit: gemm %s: %w", name, err)
			}
			g.mods = append(g.mods, mod)
			pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
				Buffers:          []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]},
				PushConstantSize: pcSize,
			})
			if err != nil {
				return fmt.Errorf("dit: gemm %s bank %d: %w", name, b, err)
			}
			m[name] = pipe
		}
		g.gemms = append(g.gemms, m)
	}
	return nil
}

// Destroy releases every Vulkan object.
func (g *GPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, p := range g.kvPipes {
		p.Destroy()
	}
	for _, m := range g.gemms {
		for _, p := range m {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.hbuf, g.abuf, g.wbuf} {
		if b != nil {
			b.Destroy()
		}
	}
	for _, b := range g.banks {
		b.Destroy()
	}
	for _, b := range g.kvBanks {
		b.Destroy()
	}
}

// BeginImage prepares one request: the text is projected through txt_in on
// the host (a few dozen rows of fp32 GEMV), the condition images' latents are
// uploaded, the rotary table is built for the layout and uploaded, and the
// joint geometry is fixed until the next call.
//
// cond is the packed [rows, 64] latents of every condition image, in the
// order their runs appear in the prompt and concatenated — nil or empty for
// t2i, where the prefix is the prompt and nothing else. The DiT takes them
// through the *same* img_in projection as the target's noise, which is the
// whole of what makes an edit a conditional generation rather than a
// different model: the condition rows are latents sitting earlier in the same
// sequence, modulated from t = 0 and therefore step-independent.
func (g *GPU) BeginImage(txt *qwen.Mat, lay *Layout, cond *qwen.Mat) error {
	s := lay.Seq()
	if s > g.maxTokens {
		return fmt.Errorf("dit: layout is %d tokens, staged for %d", s, g.maxTokens)
	}
	if lay.PrefixLen > g.maxPrefix {
		return fmt.Errorf("dit: prefix is %d rows, staged for %d", lay.PrefixLen, g.maxPrefix)
	}
	// The prefix's runs, resolved into where each one's rows come from. This
	// is Model.Assemble's walk: txt is the *whole* VLM embedding, image-pad
	// rows included, and a slot's row is dropped in favour of four latent
	// rows — so a text run's source index is its VLM index and not a
	// compacted one.
	target := lay.ImgShapes[len(lay.ImgShapes)-1]
	vlmLen := len(lay.VLMMask) - target[0]*target[1]*target[2]/4
	if txt.Rows != vlmLen {
		return fmt.Errorf("dit: %d text rows for a layout with %d VLM tokens", txt.Rows, vlmLen)
	}
	if vlmLen > g.maxText {
		return fmt.Errorf("dit: %d VLM rows, staged for %d", vlmLen, g.maxText)
	}
	var runs []prefixRun
	condRows, row, i := 0, 0, 0
	for i < vlmLen {
		if lay.VLMMask[i] {
			n := 0
			for ; i < vlmLen && lay.VLMMask[i]; i++ {
				n += 4
			}
			runs = append(runs, prefixRun{row: row, n: n, src: condRows})
			condRows, row = condRows+n, row+n
			continue
		}
		src, n := i, 0
		for ; i < vlmLen && !lay.VLMMask[i]; i++ {
			n++
		}
		runs = append(runs, prefixRun{row: row, n: n, src: src, text: true})
		row += n
	}
	if row != lay.PrefixLen {
		return fmt.Errorf("dit: the VLM mask lays out %d prefix rows, the layout says %d", row, lay.PrefixLen)
	}
	if condRows == 0 {
		if cond != nil && cond.Rows != 0 {
			return fmt.Errorf("dit: %d condition rows for a layout with none", cond.Rows)
		}
	} else if cond == nil || cond.Rows != condRows || cond.Cols != 64 {
		return fmt.Errorf("dit: condition latents are %v, want [%d 64]", cond, condRows)
	}
	txtP, err := g.head.TxtProject(txt)
	if err != nil {
		return err
	}
	rope, err := NewRope(lay, g.head.Cfg.AxesDims, g.headDim, 10000)
	if err != nil {
		return err
	}
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos.Data)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin.Data)
	g.abuf.WriteFloat32At(int(g.aTxt), txtP.Data)
	// The condition latents live past the target's in aLat, so one narrowing
	// dispatch covers both and the img_in GEMMs differ only in their offsets.
	if condRows > 0 {
		g.abuf.WriteFloat32At(int(g.aLat)+(s-lay.PrefixLen)*64, cond.Data)
	}
	g.lay, g.txtP, g.runs, g.condRows = lay, txtP, runs, condRows
	g.p, g.s = lay.PrefixLen, s
	return nil
}

// modOffsets are the ten uploaded vectors' offsets: row 0 the sampled
// timestep, row 1 the t=0 row; within a row scale1, gate1, scale2, gate2,
// tail scale.
func (g *GPU) modOff(row, idx int) uint32 {
	return g.aMod + uint32((row*5+idx)*g.dim)
}

// uploadModulation computes the shared modulation for the step on the host —
// two rows of small GEMV against weights already resident as fp32 — and
// uploads the vectors the dispatch sites read: 1+scale, tanh(gate) (the MLP
// gate carrying ffScale's inverse), and the tail's 1+scale.
func (g *GPU) uploadModulation(t float64) error {
	temb, err := g.head.TimeEmbed(t)
	if err != nil {
		return err
	}
	smod := temb.Clone()
	for i := range smod.Data {
		smod.Data[i] = silu(smod.Data[i])
	}
	mod, err := g.head.Modulation.Apply(smod)
	if err != nil {
		return err
	}
	tail, err := g.head.NormOut.Apply(smod)
	if err != nil {
		return err
	}
	buf := make([]float32, 10*g.dim)
	for row := 0; row < 2; row++ {
		m := mod.Row(row)
		for i := 0; i < g.dim; i++ {
			buf[(row*5+0)*g.dim+i] = 1 + m[i]
			buf[(row*5+1)*g.dim+i] = float32(math.Tanh(float64(m[g.dim+i])))
			buf[(row*5+2)*g.dim+i] = 1 + m[2*g.dim+i]
			buf[(row*5+3)*g.dim+i] = float32(math.Tanh(float64(m[3*g.dim+i]))) / ffScale
			buf[(row*5+4)*g.dim+i] = 1 + tail.Row(row)[i]
		}
	}
	g.abuf.WriteFloat32At(int(g.aMod), buf)
	return nil
}

// Step runs one denoising step. latents is the packed target [T, 64], t the
// pipeline's timestep/1000; prefill selects the step-0 regime, which runs
// the whole joint sequence and extracts the prefix KV cache. The result is
// the model's prediction for the target rows, [T, out_channels].
func (g *GPU) Step(latents *qwen.Mat, t float64, prefill bool) (*qwen.Mat, error) {
	d, _, err := g.stepGraph(latents, t, prefill)
	if err != nil {
		return nil, err
	}
	const perSubmit = 8
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("dit: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	tTok := g.s - g.p
	out := qwen.NewMat(tTok, g.outChannels)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.aOut), tTok*g.outChannels))
	return out, nil
}

// ReadRows reads a tensor slice out of the fp32 arena — the bring-up
// instrument, not a data path.
func (g *GPU) ReadRows(off uint32, rows, cols int) *qwen.Mat {
	m := qwen.NewMat(rows, cols)
	copy(m.Data, g.abuf.ReadFloat32At(int(off), rows*cols))
	return m
}

// WeightBytes is what the transformer holds on the device: the fp32 arena
// (the rope table and the per-block q/k norms) plus the fp16 banks the
// blocks' projections are staged into.
func (g *GPU) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

// ActivationBytes is the fp32 arena plus the fp16 one. The prefix KV cache
// is separate (CacheBytes): it is sized by the prefix rather than by the
// sequence, and at an edit it is the larger of the two.
func (g *GPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// CacheBytes is the prefix KV cache's banks.
func (g *GPU) CacheBytes() int {
	n := 0
	for _, b := range g.kvBanks {
		n += b.Size()
	}
	return n
}

// SetNoPrefixRepair switches the prefill's block-causal repair off. It is
// the negative control for the mask (qimage/dit's TestGPUEditOracle) and has
// no other use.
func (g *GPU) SetNoPrefixRepair(v bool) { g.noPrefixRepair = v }

// TensorX and friends expose arena offsets for ReadRows.
func (g *GPU) TensorX() uint32    { return g.aX }
func (g *GPU) TensorCtx() uint32  { return g.aCtx }
func (g *GPU) TensorAttn() uint32 { return g.aAttn }

// stepGraph builds the step's dispatch list. Everything before the blocks —
// the latent upload and its narrowing, the text rows at a prefill — happens
// here too, so a Step is one function.
func (g *GPU) stepGraph(latents *qwen.Mat, t float64, prefill bool) ([]vk.MultiDispatch, []string, error) {
	if g.lay == nil {
		return nil, nil, fmt.Errorf("dit: Step before BeginImage")
	}
	p, s := g.p, g.s
	tTok := s - p
	if latents.Rows != tTok || latents.Cols != 64 {
		return nil, nil, fmt.Errorf("dit: latents are %s, want [%d 64]", latents, tTok)
	}
	if err := g.uploadModulation(t); err != nil {
		return nil, nil, err
	}
	g.abuf.WriteFloat32At(int(g.aLat), latents.Data)

	// rows is the graph's local row count and xBase where the target's rows
	// start inside aX: at a prefill the graph spans the joint sequence and
	// the target sits at row p; at a cached step the graph is target-local.
	rows := tTok
	if prefill {
		rows = s
	}
	planePad := (rows + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)
	imgRow := 0
	if prefill {
		imgRow = p
	}

	base := pushConstants{
		Tokens: uint32(rows), Dim: uint32(g.dim),
		Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
	}
	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	// gemm issues C[MPad, n] = A * B over the fp16 A at aOff.
	gemm := func(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, mRows, n, k, lda int) error {
		v := gemmVariants[kernel]
		mPad := (mRows + v.bm - 1) &^ (v.bm - 1)
		if n%v.bn != 0 {
			return fmt.Errorf("dit: %s tile %d does not divide N=%d", kind, v.bn, n)
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, bOff
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(mPad), uint32(n), uint32(k)
		pc.LDA = uint32(lda)
		d = append(d, vk.MultiDispatch{
			Pipeline: g.gemms[bank][kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(mPad / v.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, kind)
		return nil
	}
	// normScale is dit_final_norm over a row range: LayerNorm times the
	// modulation row's vector, narrowed into hA. At a prefill the prefix
	// range reads the t=0 row and the target range the sampled row; a
	// cached step has only the target range.
	normScale := func(kind string, vecIdx int) {
		emit := func(start, n int, modRow int) {
			if n == 0 {
				return
			}
			pc := base
			pc.InOff = g.aX + uint32(start*g.dim)
			pc.OutOff = g.hA + uint32(start*g.ldaDim)
			pc.LDA = uint32(g.ldaDim)
			pc.Aux0 = g.modOff(modRow, vecIdx)
			pc.Eps = math.Float32bits(float32(g.eps))
			add("final_norm", kind, uint32(n), 1, pc)
		}
		if prefill {
			emit(0, p, 1)
			emit(p, rows-p, 0)
		} else {
			emit(0, rows, 0)
		}
	}
	// gateAdd is the gated residual over the same split ranges.
	gateAdd := func(kind string, y uint32, vecIdx int) {
		emit := func(start, n, modRow int) {
			if n == 0 {
				return
			}
			pc := base
			pc.Tokens = uint32(n)
			pc.InOff = y + uint32(start*g.dim)
			pc.OutOff = g.aX + uint32(start*g.dim)
			pc.Aux0 = g.modOff(modRow, vecIdx)
			pc.Aux2 = 1
			add("gate", kind, uint32(n), 1, pc)
		}
		if prefill {
			emit(0, p, 1)
			emit(p, rows-p, 0)
		} else {
			emit(0, rows, 0)
		}
	}

	// ---- The head: narrow the latents, project them into the residual
	// stream. At a prefill that is every image block — the condition images
	// as well as the target — through the one img_in projection, and the
	// condition rows are narrowed in the same dispatch because they sit
	// straight after the target's in aLat.
	latRows := tTok
	if prefill {
		latRows += g.condRows
	}
	pcLat := base
	pcLat.Tokens, pcLat.Dim = uint32(latRows), 64
	pcLat.InOff, pcLat.OutOff, pcLat.LDA = g.aLat, g.hLat, uint32(g.ldaLat)
	add("scale", "narrow latents", uint32(latRows), 1, pcLat)
	if prefill {
		// Ascending row order, because a GEMM writes whole 128-row tiles and
		// so runs past a short image block into what follows it; the next
		// block's own GEMM, and finally the text copies below, write over it.
		for _, r := range g.runs {
			if r.text {
				continue
			}
			if err := gemm(gemmSWZ8, 0, "gemm img_in cond", g.hLat+uint32((tTok+r.src)*g.ldaLat),
				g.aX+uint32(r.row*g.dim), g.imgInOff, r.n, g.dim, 64, g.ldaLat); err != nil {
				return nil, nil, err
			}
		}
	}
	if err := gemm(gemmSWZ8, 0, "gemm img_in", g.hLat, g.aX+uint32(imgRow*g.dim), g.imgInOff, tTok, g.dim, 64, g.ldaLat); err != nil {
		return nil, nil, err
	}
	if prefill {
		for _, r := range g.runs {
			if !r.text {
				continue
			}
			pc := base
			pc.Tokens = uint32(r.n)
			pc.InOff = g.aTxt + uint32(r.src*g.dim)
			pc.OutOff = g.aX + uint32(r.row*g.dim)
			add("copy", "text rows", uint32((r.n*g.dim+255)/256), 1, pc)
		}
	}

	// ---- The blocks.
	scale := float32(1 / math.Sqrt(float64(g.headDim)))
	for i := range g.blocks {
		w := &g.blocks[i]
		cacheK := w.kvOff
		cacheV := cacheK + uint32(g.maxPrefix*g.dim)
		kvPipe := g.kvPipes[w.kvBank]

		normScale("attn in", 0)
		for _, pr := range []struct {
			r   proj
			out uint32
		}{{projQ, g.aQ}, {projK, g.aK}, {projV, g.aV}} {
			cOff := pr.out
			m := rows
			if !prefill && pr.r != projQ {
				// Fresh k/v land under the cached prefix at their global rows.
				cOff += uint32(p * g.dim)
			}
			if err := gemm(gemmSWZ8, w.bank, "gemm "+string(pr.r), g.hA, cOff, w.bOff[pr.r], m, g.dim, g.dim, g.ldaDim); err != nil {
				return nil, nil, err
			}
		}

		// q: norm + rope + scale + pack, fused. The rope table is indexed by
		// the dispatch-local row, so a cached step points it at the target's
		// global rows.
		ropeOff := uint32(0)
		if !prefill {
			ropeOff = uint32(p * (g.headDim / 2))
		}
		pcQ := base
		pcQ.InOff, pcQ.OutOff = g.aQ, g.hQ
		pcQ.WOff, pcQ.Aux0, pcQ.Aux2 = g.wCos+ropeOff, g.wSin+ropeOff, w.normQ
		pcQ.Aux1 = uint32(g.tokPad)
		pcQ.Eps = math.Float32bits(float32(g.eps))
		pcQ.Scale = math.Float32bits(scale * log2e)
		add("qkpack", "qkpack q", uint32(planePad/coopMatTile), uint32(g.heads), pcQ)

		// k: the unfused trio, because the cache wants the fp32 post-RoPE
		// rows. Both k and the pack work on global rows.
		kBase, kvRows := uint32(0), rows
		if !prefill {
			kBase, kvRows = uint32(p*g.dim), s
		}
		pcNK := base
		pcNK.Tokens = uint32(kvRows - int(kBase)/g.dim)
		pcNK.InOff, pcNK.OutOff, pcNK.WOff = g.aK+kBase, g.aK+kBase, w.normK
		pcNK.Span = uint32(g.headDim)
		pcNK.Eps = math.Float32bits(float32(g.eps))
		add("rmsnorm", "rmsnorm k", uint32(int(pcNK.Tokens)*g.heads), 1, pcNK)
		pcRK := base
		pcRK.Tokens = pcNK.Tokens
		pcRK.InOff = g.aK + kBase
		pcRK.WOff, pcRK.Aux0 = g.wCos+ropeOff, g.wSin+ropeOff
		add("rope", "rope k", uint32((int(pcRK.Tokens)*g.heads*g.headDim/2+255)/256), 1, pcRK)

		// The prefix KV cache, in its own fp16 banks: saved at the prefill,
		// restored under the fresh target rows at every cached step. The
		// dispatch names its bank through the pipeline, so this is the one
		// site in the graph that does not read `pipes`.
		kv := func(kind string, mode, in, out uint32) {
			pc := base
			pc.Tokens = uint32(p)
			pc.InOff, pc.OutOff, pc.Aux0 = in, out, mode
			d = append(d, vk.MultiDispatch{
				Pipeline: kvPipe, GroupsX: uint32((p*g.dim + 255) / 256), GroupsY: 1,
				PushConstants: pc.bytes(),
			})
			kinds = append(kinds, kind)
		}
		if prefill {
			kv("save k", 0, g.aK, cacheK)
			kv("save v", 0, g.aV, cacheV)
		} else {
			kv("restore k", 1, cacheK, g.aK)
			kv("restore v", 1, cacheV, g.aV)
		}

		// Pack k and v over the whole key range.
		kvPad := (kvRows + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)
		for _, pk := range []struct {
			kind     string
			src, dst uint32
			mode     uint32
		}{{"pack k", g.aK, g.hK, 0}, {"pack v", g.aV, g.hV, 1}} {
			pc := base
			pc.Tokens = uint32(kvRows)
			pc.InOff, pc.OutOff = pk.src, pk.dst
			pc.Aux0, pc.Aux1 = pk.mode, uint32(g.tokPad)
			pc.Scale = math.Float32bits(1)
			add("pack", pk.kind, uint32(kvPad/coopMatTile), uint32(g.heads), pc)
		}

		// Attention: bidirectional over the key range; query rows are the
		// graph's rows. The prefill then recomputes the text rows causally
		// over the same planes and overwrites their context.
		pcA := base
		pcA.Tokens = uint32(kvRows)
		pcA.InOff, pcA.OutOff = g.hQ, g.aCtx
		pcA.KOff, pcA.VOff = g.hK, g.hV
		pcA.Aux1 = uint32(g.tokPad)
		add(g.attn.name, "attention", uint32((rows+g.attn.rows()-1)/g.attn.rows()), uint32(g.heads), pcA)
		// The prefill's mask is block-causal, and the pass above computed the
		// prefix's rows bidirectionally over the whole sequence — right for
		// the target's thousands of rows and wrong for every prefix row. So
		// the prefix is repaired segment by segment, **last segment first**:
		// an image segment's pass necessarily recomputes every row before it
		// too (the matrix-core kernel's query block starts at row 0), and
		// running backwards means each row's final writer is its own
		// segment's pass. Cost, at one 1024² reference: about a quarter again
		// of the prefill's attention, once per image.
		if prefill && !g.noPrefixRepair {
			for i := len(g.lay.Segments) - 1; i >= 0; i-- {
				seg := g.lay.Segments[i]
				pcS := base
				pcS.InOff, pcS.OutOff = g.hQ, g.aCtx
				pcS.KOff, pcS.VOff = g.hK, g.hV
				pcS.Aux1 = uint32(g.tokPad)
				if seg.Text {
					// Causal over its own rows, keys [0, row]: tens of rows,
					// and a triangle at a query offset the WMMA kernel's
					// block causality cannot express.
					pcS.Tokens = uint32(seg.End - seg.Start)
					pcS.Aux2 = uint32(seg.Start)
					add("causal", "causal text", uint32(seg.End-seg.Start), uint32(g.heads), pcS)
					continue
				}
				// An image block is internally bidirectional and sees
				// everything before it: the same kernel, with the key count
				// cut to the block's end.
				pcS.Tokens = uint32(seg.End)
				add("prefix_attn", "prefix image", uint32((seg.End+g.attn.rows()-1)/g.attn.rows()), uint32(g.heads), pcS)
			}
		}

		pcN := base
		pcN.InOff, pcN.OutOff, pcN.LDA = g.aCtx, g.hCtx, uint32(g.ldaDim)
		add("scale", "narrow ctx", uint32(rows), 1, pcN)
		if err := gemm(gemmSWZ8, w.bank, "gemm o", g.hCtx, g.aAttn, w.bOff[projO], rows, g.dim, g.dim, g.ldaDim); err != nil {
			return nil, nil, err
		}
		gateAdd("gate msa", g.aAttn, 1)

		normScale("ffn in", 2)
		for _, pr := range []struct {
			r   proj
			out uint32
		}{{projW1, g.aGate}, {projW3, g.aUp}} {
			if err := gemm(gemmSWZ8, w.bank, "gemm "+string(pr.r), g.hA, pr.out, w.bOff[pr.r], rows, g.ffn, g.dim, g.ldaDim); err != nil {
				return nil, nil, err
			}
		}
		pcGLU := base
		pcGLU.InOff, pcGLU.KOff, pcGLU.OutOff = g.aGate, g.aUp, g.hFFN
		pcGLU.Dim = uint32(g.ffn)
		pcGLU.LDA = uint32(g.ldaFFN)
		pcGLU.Scale = math.Float32bits(float32(ffScale))
		add("swiglu", "swiglu", uint32(rows), 1, pcGLU)
		if err := gemm(gemmSWZ8, w.bank, "gemm w2", g.hFFN, g.aFF, w.bOff[projW2], rows, g.dim, g.ffn, g.ldaFFN); err != nil {
			return nil, nil, err
		}
		gateAdd("gate mlp", g.aFF, 3)
	}

	// ---- The tail, target rows only: their prediction is all the sampler
	// keeps, in either regime.
	xTarget := g.aX
	if prefill {
		xTarget += uint32(p * g.dim)
	}
	pcT := base
	pcT.Tokens = uint32(tTok)
	pcT.InOff = xTarget
	pcT.OutOff = g.hA
	pcT.LDA = uint32(g.ldaDim)
	pcT.Aux0 = g.modOff(0, 4)
	pcT.Eps = math.Float32bits(float32(g.eps))
	add("final_norm", "tail norm", uint32(tTok), 1, pcT)
	if err := gemm(gemmReg64, 0, "gemm proj_out", g.hA, g.aOut, g.projOutOff, tTok, g.outChannels, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	return d, kinds, nil
}
