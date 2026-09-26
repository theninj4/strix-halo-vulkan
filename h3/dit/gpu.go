package dit

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU runs MiniMax-H3's transformer on the device (VIDEO.md M7), over the
// image DiT's validated kernels — the fp16 fragment-tiled GEMM, the WMMA
// flash attention, the fragment packs, SwiGLU and the gated residual — plus
// three of its own (shaders/h3_*.comp): the modulated RMS pre-norm, the q/k
// pack with H3's 96-channel rotate-half RoPE, and a bias copy.
//
// What is different from qimage/dit's graph, and why:
//
//   - **Row chunks.** The attention is 7168 wide against a 5376 residual and
//     the FFN is 2×14336, so one row costs ~300 KB of fp32 activations. The
//     image DiT's single fp32 arena would stop at ~14k rows, below the served
//     480p shape (16k), and one storage buffer stops at 4 GiB. So every
//     row-local stage — the norms, the projections, the FFN — runs over row
//     chunks of `chunk` rows through one shared scratch, and only two things
//     span the sequence: the fp32 residual X (21.5 KB a row) and the fp16 q,
//     k and v planes (43 KB a row), which is what attention needs whole. A
//     block is two passes over the chunks: q/k/v for every chunk (the planes
//     have to be complete before any query reads them), then attention, the
//     output projection and the FFN chunk by chunk. The fp16 arena holds the
//     planes, which caps a staging near 90k rows (a 10 s trained-canvas
//     clip is 74k).
//   - **The residual is fp32 everywhere.** M3 measured it at 3.1e4 after two
//     blocks, half of fp16's range.
//   - **Per-run modulation.** Every row reads the AdaLN row of its (timestep,
//     modality), and the packed sequence's rows come in at most four runs
//     that share one ([text | keyframes | audio | video]). Each modulated
//     stage is one dispatch per run, pointed at that run's uploaded vectors:
//     a = norm.weight·(1+scale) and b = shift for the norm, the gate for the
//     residual. The vectors come from the host's AdaLN tables (Tables), which
//     are the transformer's 13 B of projections applied once per request.
//   - **The refiner runs here too**, once per request over the text rows: a
//     plain pre-norm block is this block with a = norm.weight, b = 0, a gate
//     of ones and an identity rotary table. Its final norm, and everything
//     else that is a few rows of arithmetic (the timestep MLP, the output
//     norm's modulation), is on the host in fp32.
type GPU struct {
	dev  *vk.Device
	cfg  *Config
	host *Model // fp32 pieces the host runs: time MLP, refiner norm, norm_out.linear, head biases

	wbuf, abuf, hbuf *vk.Buffer
	banks            []*vk.Buffer
	pipes            map[string]*vk.ComputePipeline
	gemms            []map[gemmKernel]*vk.ComputePipeline
	mods             []*vk.ShaderModule
	attn             AttnVariant // a fixed build, when attnFixed; else attnFor picks
	attnFixed        bool
	attnPipes        map[AttnVariant]*vk.ComputePipeline

	H, inner, ffn, heads, headDim, ropeHalf int
	maxRows, maxText, chunk, planeRows      int
	ldaH, ldaInner, ldaFFN, ldaText         int
	ldaVid, ldaAud                          int
	vidK, audK                              int // input projections' K, padded to the tile

	blocks, refiner []blockW
	norms           [][2][]float32 // every block's norm1/norm2 weights, for the host's a = w·(1+scale)
	textIn          headW          // context_embedder
	vidIn, audIn    headW          // proj_in, audio_proj_in
	vidOut, audOut  headW          // proj_out, audio_proj_out, N padded to the GEMM tile

	// wbuf (fp32): rope tables, identity tables, per-block q/k norm weights.
	wCos, wSin, wCosID, wSinID uint32
	// abuf (fp32): the residual, the chunk scratch, the per-forward
	// modulation vectors, the refiner's vectors, head outputs, biases.
	aX, aS                  uint32
	aMod, aModRef, aModTail uint32
	aOutV, aOutA            uint32
	aOnes, aZeros           uint32
	aBias                   map[string]uint32
	actElems                int
	// hbuf (fp16): the q/k/v planes, the chunk A operands, the input rows.
	hQ, hK, hV, hA, hCtx, hFFN uint32
	hVid, hAud, hText          uint32
	hElems                     int

	// noKeyTail skips zeroKeyTail: TestGPUForward's negative control, and
	// nothing else.
	noKeyTail bool

	// Per-request state (Begin).
	lay   *plan.Layout
	text  *qwen.Mat // refined text rows, fp32
	temb  *qwen.Mat // the request's distinct timesteps' embeddings
	tabs  []*Table  // per block, over temb
	tvals []float32 // the distinct timesteps temb rows embed
	mod   *qwen.Mat // norm_out.linear(silu(temb)): [shift | scale] per timestep
}

type headW struct {
	bank     int
	off      uint32
	n, k     int // n padded to the GEMM tile
	realN    int
	biasName string
}

type blockW struct {
	bank         int
	off          map[proj]uint32
	normQ, normK uint32
}

type proj int

const (
	projQ proj = iota
	projK
	projV
	projO
	projGate // the silu'd half of ff.net.0.proj, rows [ffn, 2·ffn)
	projUp   // the linear half, rows [0, ffn)
	projDown
)

var projOrder = []proj{projQ, projK, projV, projO, projGate, projUp, projDown}

// pushConstants mirrors shaders/dit_common.glsl.
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

type gemmKernel int

const (
	gemmBig   gemmKernel = iota // 128x256 tiles, the image DiT's measured winner
	gemmSmall                   // 64x64, for the narrow heads
)

var gemmVariants = map[gemmKernel]struct {
	spirv  []byte
	bm, bn int
}{
	gemmBig:   {shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256},
	gemmSmall: {shaders.DiTGEMMReg64Tiled, 64, 64},
}

const (
	tile     = 16
	gemmPad  = 128
	rowAlign = 128 // chunk and plane granularity: the big GEMM's M tile

	log2e = 1.4426950408889634
	// ffScale scales SwiGLU's fp16 output down, and its inverse rides in the
	// uploaded MLP gate, as in qimage/dit. Here it is load-bearing: M4's
	// oracle has the down projection's input at 1.27e5 in block 39 (and
	// 7.5e4 in 45), past fp16's 65504; at 1/16 the peak is 7.9e3. Every other
	// fp16 operand stays under 1.7e3 (k at block 48); the residual they feed
	// reaches 8.5e6, which is why it never leaves fp32.
	ffScale = 1.0 / 16
	// maxBankBytes is one storage buffer's range on this device.
	maxBankBytes = 0xfffffffc
)

// NewGPU stages the transformer for sequences of up to maxRows rows, of
// which at most maxText are text, running row-local stages chunk rows at a
// time. ~40 GB of fp16 banks, staged a block at a time.
func NewGPU(dev *vk.Device, dir string, maxRows, maxText, chunk int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if chunk <= 0 || chunk%rowAlign != 0 {
		return nil, fmt.Errorf("dit: chunk %d is not a positive multiple of %d", chunk, rowAlign)
	}
	host, err := Load(dir, 0)
	if err != nil {
		return nil, err
	}
	g := &GPU{
		dev: dev, cfg: cfg, host: host,
		pipes: map[string]*vk.ComputePipeline{},
		H:     cfg.Hidden, inner: cfg.Inner(), ffn: cfg.FFN, heads: cfg.Heads, headDim: cfg.HeadDim,
		ropeHalf: cfg.RopeWidth() / 2,
		maxRows:  maxRows, maxText: maxText, chunk: chunk,
		aBias: map[string]uint32{},
	}
	g.planeRows = roundUp(maxRows, rowAlign)
	g.ldaH, g.ldaInner, g.ldaFFN = g.H+gemmPad, g.inner+gemmPad, g.ffn+gemmPad
	g.ldaText = cfg.TextDim + gemmPad
	g.vidK, g.audK = roundUp(cfg.Patch(), tile), roundUp(cfg.AudioChannels, tile)
	g.ldaVid, g.ldaAud = g.vidK+gemmPad, g.audK+gemmPad

	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	for _, step := range []func(*safetensors.Set) error{g.stage, g.allocActivations, g.build} {
		if err := step(set); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// packB narrows a [n, k] row-major weight into 16x16 fragment tiles,
// kt-fastest: dit_gemm.comp's B layout (qimage/dit's packB).
func packB(dst []uint16, w []float32, n, k int) {
	kt := k / tile
	parallel((n+63)/64, func(c int) {
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

func (g *GPU) projShape(p proj) [2]int {
	switch p {
	case projQ, projK, projV:
		return [2]int{g.inner, g.H}
	case projO:
		return [2]int{g.H, g.inner}
	case projGate, projUp:
		return [2]int{g.ffn, g.H}
	default:
		return [2]int{g.H, g.ffn}
	}
}

// stage lays the fp16 banks out and fills them, and fills the fp32 weight
// arena: rope tables are per request, so here it is the identity tables and
// the per-block q/k norm weights.
func (g *GPU) stage(set *safetensors.Set) error {
	perBlock := 0
	for _, p := range projOrder {
		sh := g.projShape(p)
		perBlock += sh[0] * sh[1]
	}
	var bankElems []int
	cur := -1
	place := func(n int) (int, uint32) {
		if cur < 0 || (bankElems[cur]+n)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
		off := bankElems[cur]
		bankElems[cur] += n
		return cur, uint32(off)
	}
	head := func(n, k, realN int, bias string) headW {
		b, off := place(n * k)
		return headW{bank: b, off: off, n: n, k: k, realN: realN, biasName: bias}
	}
	c := g.cfg
	g.textIn = head(g.H, c.TextDim, g.H, "context_embedder")
	g.vidIn = head(g.H, g.vidK, g.H, "proj_in")
	g.audIn = head(g.H, g.audK, g.H, "audio_proj_in")
	g.vidOut = head(roundUp(c.Patch(), gemmVariants[gemmSmall].bn), g.H, c.Patch(), "proj_out")
	g.audOut = head(roundUp(c.AudioChannels, gemmVariants[gemmSmall].bn), g.H, c.AudioChannels, "audio_proj_out")

	w32 := 0
	g.wCos, g.wSin = 0, uint32(g.maxRows*g.ropeHalf)
	g.wCosID, g.wSinID = uint32(2*g.maxRows*g.ropeHalf), uint32((2*g.maxRows+g.maxText)*g.ropeHalf)
	w32 = (2*g.maxRows + 2*g.maxText) * g.ropeHalf
	newBlock := func() blockW {
		w := blockW{off: map[proj]uint32{}, normQ: uint32(w32), normK: uint32(w32 + g.headDim)}
		w32 += 2 * g.headDim
		// A block's seven projections share a bank, so a GEMM's pipeline is
		// picked by the block alone.
		b, off := place(perBlock)
		w.bank = b
		for _, p := range projOrder {
			sh := g.projShape(p)
			w.off[p] = off
			off += uint32(sh[0] * sh[1])
		}
		return w
	}
	g.refiner = make([]blockW, c.RefinerLayers)
	for i := range g.refiner {
		g.refiner[i] = newBlock()
	}
	g.blocks = make([]blockW, c.Layers)
	for i := range g.blocks {
		g.blocks[i] = newBlock()
	}

	var err error
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return fmt.Errorf("dit: fp32 weight arena: %w", err)
	}
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("dit: fp16 bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}
	ones := make([]float32, g.maxText*g.ropeHalf)
	for i := range ones {
		ones[i] = 1
	}
	g.wbuf.WriteFloat32At(int(g.wCosID), ones)
	g.wbuf.ZeroFloat32At(int(g.wSinID), g.maxText*g.ropeHalf)

	stage := func(bank int, off uint32, w []float32, n, k, realN, realK int) {
		src := w
		if n != realN || k != realK {
			src = make([]float32, n*k)
			for i := 0; i < realN; i++ {
				copy(src[i*k:i*k+realK], w[i*realK:(i+1)*realK])
			}
		}
		buf := make([]uint16, n*k)
		packB(buf, src, n, k)
		g.banks[bank].WriteUint16At(int(off), buf)
	}
	h := g.host
	for _, hd := range []struct {
		w    headW
		lin  *Linear
		inK  int
		name string
	}{
		{g.textIn, h.TextIn, c.TextDim, "context_embedder"},
		{g.vidIn, h.ProjIn, c.Patch(), "proj_in"},
		{g.audIn, h.AudioProjIn, c.AudioChannels, "audio_proj_in"},
		{g.vidOut, h.ProjOut, g.H, "proj_out"},
		{g.audOut, h.AudioProjOut, g.H, "audio_proj_out"},
	} {
		stage(hd.w.bank, hd.w.off, hd.lin.Weight, hd.w.n, hd.w.k, hd.lin.Out, hd.inK)
	}

	stageBlock := func(w blockW, p string) error {
		l := &loader{set: set}
		wq := l.f32(p+"attn.to_q.weight", g.inner, g.H)
		wk := l.f32(p+"attn.to_k.weight", g.inner, g.H)
		wv := l.f32(p+"attn.to_v.weight", g.inner, g.H)
		wo := l.f32(p+"attn.to_out.0.weight", g.H, g.inner)
		up := l.f32(p+"ff.net.0.proj.weight", 2*g.ffn, g.H)
		down := l.f32(p+"ff.net.2.weight", g.H, g.ffn)
		nq := l.f32(p+"attn.norm_q.weight", g.headDim)
		nk := l.f32(p+"attn.norm_k.weight", g.headDim)
		if l.err != nil {
			return l.err
		}
		// SwiGLU is value·silu(gate) with the projection's rows [value | gate].
		lins := map[proj][]float32{
			projQ: wq, projK: wk, projV: wv, projO: wo,
			projUp: up[:g.ffn*g.H], projGate: up[g.ffn*g.H:], projDown: down,
		}
		for _, pr := range projOrder {
			sh := g.projShape(pr)
			stage(w.bank, w.off[pr], lins[pr], sh[0], sh[1], sh[0], sh[1])
		}
		g.wbuf.WriteFloat32At(int(w.normQ), nq)
		g.wbuf.WriteFloat32At(int(w.normK), nk)
		return nil
	}
	g.norms = make([][2][]float32, c.Layers)
	for i := range g.norms {
		l := &loader{set: set}
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		g.norms[i] = [2][]float32{l.f32(p+"norm1.weight", g.H), l.f32(p+"norm2.weight", g.H)}
		if l.err != nil {
			return l.err
		}
	}
	for i, w := range g.refiner {
		if err := stageBlock(w, fmt.Sprintf("token_refiner.refiner_blocks.%d.", i)); err != nil {
			return err
		}
	}
	for i, w := range g.blocks {
		if err := stageBlock(w, fmt.Sprintf("transformer_blocks.%d.", i)); err != nil {
			return err
		}
	}
	return nil
}

func (g *GPU) allocActivations(*safetensors.Set) error {
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += roundUp(n, 64)
		return off
	}
	rows := g.planeRows + rowAlign
	g.aX = alloc(rows * g.H)
	// The chunk scratch: q, k, v in the first pass; the attention output,
	// the FFN's two halves and its output in the second. Its span is also
	// where the input projections land before their bias copy.
	g.aS = alloc(g.chunk * max(3*g.inner, g.H+2*g.ffn+g.H))
	nRuns := 8 // runs a forward's modulation may split into
	g.aMod = alloc(g.cfg.Layers * nRuns * 6 * g.H)
	g.aModRef = alloc(g.cfg.RefinerLayers * 6 * g.H)
	g.aModTail = alloc(nRuns * 2 * g.H)
	g.aOutV = alloc(rows * g.vidOut.n)
	g.aOutA = alloc(rows * g.audOut.n)
	g.aOnes = alloc(g.H)
	g.aZeros = alloc(g.H)
	for _, name := range []string{"context_embedder", "proj_in", "audio_proj_in"} {
		g.aBias[name] = alloc(g.H)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += roundUp(n, 64)
		return off
	}
	plane := g.heads * g.planeRows * g.headDim
	g.hQ, g.hK, g.hV = halloc(plane), halloc(plane), halloc(plane)
	g.hA = halloc(g.chunk * g.ldaH)
	g.hCtx = halloc(g.chunk * g.ldaInner)
	g.hFFN = halloc(g.chunk * g.ldaFFN)
	g.hVid = halloc(rows * g.ldaVid)
	g.hAud = halloc(rows * g.ldaAud)
	g.hText = halloc(roundUp(g.maxText, rowAlign) * g.ldaText)

	// A buffer past the storage-buffer range is created without complaint
	// and clamped where it is bound, so a kernel reading its tail gets zeros
	// and the run comes back fast and wrong. Refuse here instead.
	for _, a := range []struct {
		name  string
		bytes int
	}{{"fp32 activation", g.actElems * 4}, {"fp16 activation", g.hElems * 2}} {
		if a.bytes > maxBankBytes {
			return fmt.Errorf("dit: the %s arena for %d rows (chunk %d) is %d MB, past the %d MB storage-buffer range",
				a.name, g.maxRows, g.chunk, a.bytes>>20, maxBankBytes>>20)
		}
	}
	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("dit: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("dit: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Pad rows and columns of every A operand, and the planes' tails, read
	// as zeros.
	g.hbuf.ZeroUint16At(0, g.hElems)
	ones := make([]float32, g.H)
	for i := range ones {
		ones[i] = 1
	}
	g.abuf.WriteFloat32At(int(g.aOnes), ones)
	g.abuf.ZeroFloat32At(int(g.aZeros), g.H)
	h := g.host
	g.abuf.WriteFloat32At(int(g.aBias["context_embedder"]), h.TextIn.Bias)
	g.abuf.WriteFloat32At(int(g.aBias["proj_in"]), h.ProjIn.Bias)
	g.abuf.WriteFloat32At(int(g.aBias["audio_proj_in"]), h.AudioProjIn.Bias)
	return nil
}

func (g *GPU) build(*safetensors.Set) error {
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	newPipe := func(spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
		mod, err := g.dev.NewShaderModule(spirv)
		if err != nil {
			return nil, err
		}
		g.mods = append(g.mods, mod)
		return g.dev.NewPipeline(mod, spec)
	}
	for name, spirv := range map[string][]byte{
		"norm":   shaders.H3NormMod,
		"qkpack": shaders.H3QKPackTPW8,
		"bias":   shaders.H3BiasCopy,
		"pack":   shaders.DiTPackF16TPW8,
		"gate":   shaders.DiTGateAdd,
		"swiglu": shaders.DiTSwiGLUF16,
		"narrow": shaders.DiTScaleF16,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("dit: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	// The attention, writing its context straight into the output
	// projection's fp16 A operand: wave32 where the size can be pinned.
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return err
	}
	if !g.dev.Features().SubgroupSizeControl || !sgs.Supported || 32 < sgs.MinSubgroupSize || 32 > sgs.MaxSubgroupSize {
		return fmt.Errorf("dit: the attention build needs a pinned wave32")
	}
	g.attnPipes = map[AttnVariant]*vk.ComputePipeline{}
	for v, spirv := range attnBuilds {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize, RequiredSubgroupSize: 32})
		if err != nil {
			return fmt.Errorf("dit: attention pipeline %v: %w", v, err)
		}
		g.attnPipes[v] = p
	}

	for b := range g.banks {
		m := map[gemmKernel]*vk.ComputePipeline{}
		for k, v := range gemmVariants {
			p, err := newPipe(v.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}, PushConstantSize: pcSize})
			if err != nil {
				return fmt.Errorf("dit: gemm bank %d: %w", b, err)
			}
			m[k] = p
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
	for _, p := range g.attnPipes {
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
	for _, b := range append([]*vk.Buffer{g.hbuf, g.abuf, g.wbuf}, g.banks...) {
		if b != nil {
			b.Destroy()
		}
	}
}

// WeightBytes and ActivationBytes are what the transformer holds on the
// device.
func (g *GPU) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

func (g *GPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Begin prepares one request: the layout's rotary tables, the AdaLN tables
// for every timestep the request will use (tabs, one per block, built by
// Tables over the distinct timesteps tvals), and the text rows — cond, the
// encoder's [text, 5120] conditioning, through context_embedder and the
// refiner on the device and the refiner's final norm on the host.
func (g *GPU) Begin(lay *plan.Layout, cond *qwen.Mat, tvals []float32, tabs []*Table) error {
	if len(lay.Pos) > g.maxRows {
		return fmt.Errorf("dit: layout is %d rows, staged for %d", len(lay.Pos), g.maxRows)
	}
	if len(lay.Text) > g.maxText || cond.Rows != len(lay.Text) || cond.Cols != g.cfg.TextDim {
		return fmt.Errorf("dit: conditioning %v for %d text rows (staged for %d)", cond, len(lay.Text), g.maxText)
	}
	if len(tabs) != len(g.blocks) {
		return fmt.Errorf("dit: %d AdaLN tables for %d blocks", len(tabs), len(g.blocks))
	}
	temb, err := g.host.TimeEmbed(tvals)
	if err != nil {
		return err
	}
	mod, err := g.host.NormOutLinear.Apply(siluMat(temb))
	if err != nil {
		return err
	}
	cos, sin := lay.Rope(plan.InvFreq())
	half, width := g.ropeHalf, 2*g.ropeHalf
	c, s := make([]float32, len(lay.Pos)*half), make([]float32, len(lay.Pos)*half)
	for r := range lay.Pos {
		copy(c[r*half:(r+1)*half], cos[r*width:r*width+half])
		copy(s[r*half:(r+1)*half], sin[r*width:r*width+half])
	}
	g.wbuf.WriteFloat32At(int(g.wCos), c)
	g.wbuf.WriteFloat32At(int(g.wSin), s)
	g.lay, g.temb, g.tabs, g.tvals, g.mod = lay, temb, tabs, tvals, mod
	// Both attention lengths this request runs — the refiner's text rows and
	// the whole sequence — get a zero tail.
	g.zeroKeyTail(cond.Rows)
	g.text, err = g.refine(cond)
	g.zeroKeyTail(len(lay.Pos))
	return err
}

// zeroKeyTail zeroes the k and v plane rows between the last tile a pack of
// `keys` rows writes and the end of the attention's last key block. The
// kernel reads whole key blocks and masks the rows past its key count out
// of P but **not out of the row max**, so a stale row from an earlier,
// longer sequence (or from the whole sequence, when the refiner runs over
// the text rows after it) can set the max, underflow every real weight to
// zero and turn the row into 0/0. The second request of a test at two
// shapes did exactly that to all 537 text rows.
func (g *GPU) zeroKeyTail(keys int) {
	if g.noKeyTail {
		return
	}
	from := roundUp(keys, tile)
	// The longest key block of any build, so switching builds needs no rezero.
	to := min(roundUp(keys, maxKeyBlock), g.planeRows)
	if from >= to {
		return
	}
	for _, pl := range []uint32{g.hK, g.hV} {
		for h := 0; h < g.heads; h++ {
			base := int(pl) + h*g.planeRows*g.headDim
			g.hbuf.ZeroUint16At(base+from*g.headDim, (to-from)*g.headDim)
		}
	}
}

// run is a stretch of rows sharing one modulation row.
type run struct{ start, n, key int }

func runsOf(keys []int32, rows []int32) []run {
	var out []run
	for _, r := range rows {
		k := int(keys[r])
		if n := len(out); n > 0 && out[n-1].key == k && out[n-1].start+out[n-1].n == int(r) {
			out[n-1].n++
			continue
		}
		out = append(out, run{start: int(r), n: 1, key: k})
	}
	return out
}

// graph accumulates one submission's dispatches.
type graph struct {
	g      *GPU
	d      []vk.MultiDispatch
	kinds  []string
	flops  []float64
	base   pushConstants
	noBase bool
}

func (g *GPU) newGraph() *graph {
	return &graph{g: g, base: pushConstants{Heads: uint32(g.heads), HeadDim: uint32(g.headDim)}}
}

func (gr *graph) add(pipe, kind string, gx, gy uint32, pc pushConstants) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, 0)
}

// gemm issues C[mPad, n] fp32 at cOff (row stride n) = A[m, k] fp16 at aOff
// (row stride lda) × B, the fragment-tiled weight at bOff of bank.
func (gr *graph) gemm(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, m, n, k, lda int) error {
	v := gemmVariants[kernel]
	if n%v.bn != 0 || k%tile != 0 {
		return fmt.Errorf("dit: %s: N=%d K=%d do not tile by %d/%d", kind, n, k, v.bn, tile)
	}
	mPad := roundUp(m, v.bm)
	pc := gr.base
	pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(mPad), uint32(n), uint32(k), uint32(lda)
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.gemms[bank][kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(mPad / v.bm), PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, 2*float64(mPad)*float64(n)*float64(k))
	return nil
}

// submitBudget is how much estimated device time one submission may hold.
// amdgpu kills a submission that holds the ring for 2 s (research/p0-ring-
// watchdog.md), and a count of dispatches is no bound: one block's five
// attention chunks at 38k rows are 2.15 s. Each dispatch is priced at its
// FLOPs over submitRate — below what any matrix-core kernel here measures —
// plus a flat cost for the elementwise passes, and a submission closes
// before it would pass the budget. A dispatch priced over the budget on its
// own still goes, alone; NewGPU's chunk is what keeps those under 2 s.
const (
	submitBudget = 800 * time.Millisecond
	submitRate   = 12e12
	submitFlat   = 2 * time.Millisecond
)

func (gr *graph) submit() (time.Duration, error) {
	var total time.Duration
	for i := 0; i < len(gr.d); {
		j, est := i, time.Duration(0)
		for j < len(gr.d) {
			c := submitFlat + time.Duration(gr.flops[j]/submitRate*1e9)
			if j > i && est+c > submitBudget {
				break
			}
			est += c
			j++
		}
		t, err := vk.DispatchMultiTimed(gr.d[i:j], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("dit: dispatch %d-%d (%s): %w", i, j-1, gr.kinds[i], err)
		}
		total += t
		i = j
	}
	gr.d, gr.kinds, gr.flops = gr.d[:0], gr.kinds[:0], gr.flops[:0]
	return total, nil
}

// modVecs names where a run's six modulation vectors live in the fp32
// arena: a1, b1, g1 for attention, a2, b2, g2 for the FFN.
type modVecs [6]uint32

// blockPass records one block over rows [0, rows) of the residual with the
// given rotary tables, query/key planes spanning all of them, and the runs'
// vectors. It is the transformer block and, with the refiner's vectors and
// identity tables, the refiner block.
func (gr *graph) blockPass(w blockW, rows int, cos, sin uint32, runs []run, vecs map[int]modVecs) error {
	g := gr.g
	eps := math.Float32bits(float32(g.cfg.NormEps))
	qkEps := math.Float32bits(float32(g.cfg.QKNormEps))
	aQ := g.aS
	aK := aQ + uint32(g.chunk*g.inner)
	aV := aK + uint32(g.chunk*g.inner)
	aAttn := g.aS
	aGate := aAttn + uint32(g.chunk*g.H)
	aUp := aGate + uint32(g.chunk*g.ffn)
	aFF := aUp + uint32(g.chunk*g.ffn)
	planeTile := uint32(g.headDim / tile * tile * tile) // halves per 16-row tile of one head

	// norm is the modulated pre-norm over chunk rows [r0, r1) into hA.
	norm := func(kind string, r0, r1, which int) {
		for _, ru := range runs {
			lo, hi := max(ru.start, r0), min(ru.start+ru.n, r1)
			if lo >= hi {
				continue
			}
			v := vecs[ru.key]
			pc := gr.base
			pc.Tokens, pc.Dim, pc.LDA = uint32(hi-lo), uint32(g.H), uint32(g.ldaH)
			pc.InOff = g.aX + uint32(lo*g.H)
			pc.OutOff = g.hA + uint32((lo-r0)*g.ldaH)
			pc.Aux0, pc.Aux1, pc.Eps = v[3*which], v[3*which+1], eps
			gr.add("norm", kind, uint32(hi-lo), 1, pc)
		}
	}
	gate := func(kind string, r0, r1, which int, y uint32) {
		for _, ru := range runs {
			lo, hi := max(ru.start, r0), min(ru.start+ru.n, r1)
			if lo >= hi {
				continue
			}
			pc := gr.base
			pc.Tokens, pc.Dim = uint32(hi-lo), uint32(g.H)
			pc.InOff = y + uint32((lo-r0)*g.H)
			pc.OutOff = g.aX + uint32(lo*g.H)
			pc.Aux0, pc.Aux2 = vecs[ru.key][3*which+2], 1
			gr.add("gate", kind, uint32(hi-lo), 1, pc)
		}
	}

	// Pass 1: q, k, v for every chunk into the planes.
	for r0 := 0; r0 < rows; r0 += g.chunk {
		r1 := min(r0+g.chunk, rows)
		n := r1 - r0
		norm("attn in", r0, r1, 0)
		for _, pr := range []struct {
			p   proj
			out uint32
		}{{projQ, aQ}, {projK, aK}, {projV, aV}} {
			if err := gr.gemm(gemmBig, w.bank, "gemm qkv", g.hA, pr.out, w.off[pr.p], n, g.inner, g.H, g.ldaH); err != nil {
				return err
			}
		}
		tiles := roundUp(n, tile) / tile
		tileOff := uint32(r0/tile) * planeTile
		for _, qk := range []struct {
			kind     string
			src, dst uint32
			norm     uint32
			scale    float32
		}{
			{"qkpack q", aQ, g.hQ, w.normQ, float32(1/math.Sqrt(float64(g.headDim))) * log2e},
			{"qkpack k", aK, g.hK, w.normK, 1},
		} {
			pc := gr.base
			pc.Tokens, pc.Dim = uint32(n), uint32(g.inner)
			pc.InOff, pc.OutOff = qk.src, qk.dst+tileOff
			pc.WOff, pc.Aux0 = cos+uint32(r0*g.ropeHalf), sin+uint32(r0*g.ropeHalf)
			pc.Aux1, pc.Aux2 = uint32(g.planeRows), qk.norm
			pc.Span, pc.Eps, pc.Scale = uint32(tiles), qkEps, math.Float32bits(qk.scale)
			gr.add("qkpack", qk.kind, uint32((tiles+7)/8), uint32(g.heads), pc)
		}
		pc := gr.base
		pc.Tokens, pc.Dim = uint32(n), uint32(g.inner)
		pc.InOff, pc.OutOff = aV, g.hV+tileOff
		pc.Aux0, pc.Aux1, pc.Aux2 = 1, uint32(g.planeRows), uint32(tiles)
		pc.Scale = math.Float32bits(1)
		gr.add("pack", "pack v", uint32((tiles+7)/8), uint32(g.heads), pc)
	}

	// Pass 2: attention and everything after it, chunk by chunk.
	for r0 := 0; r0 < rows; r0 += g.chunk {
		r1 := min(r0+g.chunk, rows)
		n := r1 - r0
		pc := gr.base
		pc.Tokens = uint32(rows)
		pc.InOff = g.hQ + uint32(r0/tile)*planeTile
		pc.KOff, pc.VOff = g.hK, g.hV
		pc.OutOff, pc.LDA = g.hCtx, uint32(g.ldaInner)
		pc.Aux1 = uint32(g.planeRows)
		gr.attention(rows, n, pc)
		if err := gr.gemm(gemmBig, w.bank, "gemm o", g.hCtx, aAttn, w.off[projO], n, g.H, g.inner, g.ldaInner); err != nil {
			return err
		}
		gate("gate attn", r0, r1, 0, aAttn)
		norm("ffn in", r0, r1, 1)
		if err := gr.gemm(gemmBig, w.bank, "gemm gate", g.hA, aGate, w.off[projGate], n, g.ffn, g.H, g.ldaH); err != nil {
			return err
		}
		if err := gr.gemm(gemmBig, w.bank, "gemm up", g.hA, aUp, w.off[projUp], n, g.ffn, g.H, g.ldaH); err != nil {
			return err
		}
		pcG := gr.base
		pcG.Tokens, pcG.Dim, pcG.LDA = uint32(n), uint32(g.ffn), uint32(g.ldaFFN)
		pcG.InOff, pcG.KOff, pcG.OutOff = aGate, aUp, g.hFFN
		pcG.Scale = math.Float32bits(ffScale)
		gr.add("swiglu", "swiglu", uint32(n), 1, pcG)
		if err := gr.gemm(gemmBig, w.bank, "gemm down", g.hFFN, aFF, w.off[projDown], n, g.H, g.ffn, g.ldaFFN); err != nil {
			return err
		}
		gate("gate ffn", r0, r1, 1, aFF)
	}
	return nil
}

// project records an input projection over rows of an fp16 A operand into
// the residual rows starting at dst: the GEMM into the chunk scratch, then
// the bias copy.
func (gr *graph) project(hw headW, aOff uint32, lda, rows, dst int) error {
	g := gr.g
	for r0 := 0; r0 < rows; r0 += g.chunk {
		n := min(g.chunk, rows-r0)
		if err := gr.gemm(gemmBig, hw.bank, "gemm "+hw.biasName, aOff+uint32(r0*lda), g.aS, hw.off, n, hw.n, hw.k, lda); err != nil {
			return err
		}
		pc := gr.base
		pc.Tokens, pc.Dim, pc.LDA = uint32(n), uint32(g.H), uint32(hw.n)
		pc.InOff, pc.OutOff = g.aS, g.aX+uint32((dst+r0)*g.H)
		pc.Aux0, pc.Aux2 = g.aBias[hw.biasName], 1
		gr.add("bias", "bias "+hw.biasName, uint32(n), 1, pc)
	}
	return nil
}

// narrowRows writes rows of an fp32 host matrix as an fp16 A operand with
// row stride lda at hOff: the input projections' operands, built on the host
// because they are small and change every step.
func (g *GPU) narrowRows(hOff uint32, m *qwen.Mat, lda int) {
	buf := make([]uint16, m.Rows*lda)
	for r := 0; r < m.Rows; r++ {
		for c, v := range m.Row(r) {
			buf[r*lda+c] = safetensors.F32ToF16(v)
		}
	}
	g.hbuf.WriteUint16At(int(hOff), buf)
}

// refine runs context_embedder and the two refiner blocks over the text
// rows on the device, reads them back and applies the refiner's final norm
// on the host. The refiner has no AdaLN and no rotary embedding: its vectors
// are the norm weights, zero shifts and unit gates, and its tables are the
// identity.
func (g *GPU) refine(cond *qwen.Mat) (*qwen.Mat, error) {
	n := cond.Rows
	g.narrowRows(g.hText, cond, g.ldaText)
	gr := g.newGraph()
	if err := gr.project(g.textIn, g.hText, g.ldaText, n, 0); err != nil {
		return nil, err
	}
	vecs := map[int]modVecs{}
	for i := range g.refiner {
		b := g.host.Refiner[i]
		base := g.aModRef + uint32(i*6*g.H)
		ffGate := make([]float32, g.H)
		for j := range ffGate {
			ffGate[j] = 1 / ffScale
		}
		for j, v := range [][]float32{b.Norm1.Weight, nil, nil, b.Norm2.Weight, nil, ffGate} {
			if v != nil {
				g.abuf.WriteFloat32At(int(base)+j*g.H, v)
			}
		}
		vecs[i] = modVecs{base, g.aZeros, g.aOnes, base + uint32(3*g.H), g.aZeros, base + uint32(5*g.H)}
	}
	for i, w := range g.refiner {
		if err := gr.blockPass(w, n, g.wCosID, g.wSinID, []run{{0, n, i}}, map[int]modVecs{i: vecs[i]}); err != nil {
			return nil, err
		}
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	x := qwen.NewMat(n, g.H)
	copy(x.Data, g.abuf.ReadFloat32At(int(g.aX), n*g.H))
	return g.host.RefinerNorm.Apply(x)
}

// uploadStep writes forward f's modulation vectors: for every block and
// every run of the layout, a1 = norm1·(1+scale_msa), b1 = shift_msa,
// g1 = gate_msa, a2 = norm2·(1+scale_mlp), b2 = shift_mlp and
// g2 = gate_mlp/ffScale; and the tail's a = norm_out·(1+scale), b = shift per
// timestep run. rowT indexes tvals.
func (g *GPU) uploadStep(rowT []int32) ([]run, []run, error) {
	lay := g.lay
	rowMod := RowMod(rowT, lay.Tags)
	all := make([]int32, len(lay.Pos))
	for i := range all {
		all[i] = int32(i)
	}
	runs := runsOf(rowMod, all)
	keys := map[int]int{}
	for _, r := range runs {
		if _, ok := keys[r.key]; !ok {
			keys[r.key] = len(keys)
		}
	}
	if len(keys) > 8 {
		return nil, nil, fmt.Errorf("dit: %d modulation rows in one forward, room for 8", len(keys))
	}
	H := g.H
	buf := make([]float32, len(g.blocks)*8*6*H)
	for b, tab := range g.tabs {
		n1, n2 := g.norms[b][0], g.norms[b][1]
		for key, slot := range keys {
			o := (b*8 + slot) * 6 * H
			sm1, sc1, g1 := tab.Vec(key, 0), tab.Vec(key, 1), tab.Vec(key, 2)
			sm2, sc2, g2 := tab.Vec(key, 3), tab.Vec(key, 4), tab.Vec(key, 5)
			for i := 0; i < H; i++ {
				buf[o+i] = n1[i] * (1 + sc1[i])
				buf[o+H+i] = sm1[i]
				buf[o+2*H+i] = g1[i]
				buf[o+3*H+i] = n2[i] * (1 + sc2[i])
				buf[o+4*H+i] = sm2[i]
				buf[o+5*H+i] = g2[i] / ffScale
			}
		}
	}
	g.abuf.WriteFloat32At(int(g.aMod), buf)

	// The tail: modulated by timestep alone, over the video and audio rows.
	heads := append(append([]int32(nil), lay.Audio...), lay.Video...)
	tail := runsOf(rowT, heads)
	tkeys := map[int]int{}
	for _, r := range tail {
		if _, ok := tkeys[r.key]; !ok {
			tkeys[r.key] = len(tkeys)
		}
	}
	tbuf := make([]float32, 8*2*H)
	nw := g.host.NormOut.Weight
	for key, slot := range tkeys {
		m := g.mod.Row(key)
		shift, scale := m[:H], m[H:]
		for i := 0; i < H; i++ {
			tbuf[slot*2*H+i] = nw[i] * (1 + scale[i])
			tbuf[slot*2*H+H+i] = shift[i]
		}
	}
	g.abuf.WriteFloat32At(int(g.aModTail), tbuf)
	for i := range runs {
		runs[i].key = keys[runs[i].key]
	}
	for i := range tail {
		tail[i].key = tkeys[tail[i].key]
	}
	return runs, tail, nil
}

// vecsFor is where forward-local modulation slot k of block b lives.
func (g *GPU) vecsFor(b int, slots int) map[int]modVecs {
	out := map[int]modVecs{}
	for k := 0; k < slots; k++ {
		o := g.aMod + uint32((b*8+k)*6*g.H)
		var v modVecs
		for j := range v {
			v[j] = o + uint32(j*g.H)
		}
		out[k] = v
	}
	return out
}

// Step runs one forward: video are the layout's video rows (conditioning
// rows first) [len(Video), 96], audio its audio rows [len(Audio), 32], rowT
// every row's index into the tvals Begin was given. It returns the velocity
// of every video and audio row, in the same order, and the device time.
func (g *GPU) Step(video, audio *qwen.Mat, rowT []int32) (v, a *qwen.Mat, took time.Duration, err error) {
	gr, err := g.stepGraph(video, audio, rowT, -1)
	if err != nil {
		return nil, nil, 0, err
	}
	if took, err = gr.submit(); err != nil {
		return nil, nil, 0, err
	}
	v, a = g.readHeads()
	return v, a, took, nil
}

// Blocks runs blocks [from, to) over x, the packed residual [rows, hidden],
// and returns the residual after them: the teacher-forced instrument M7's
// gates run block by block against the oracle.
func (g *GPU) Blocks(x *qwen.Mat, rowT []int32, from, to int) (*qwen.Mat, error) {
	if x.Rows != len(g.lay.Pos) || x.Cols != g.H {
		return nil, fmt.Errorf("dit: residual %v for a %d-row layout", x, len(g.lay.Pos))
	}
	runs, _, err := g.uploadStep(rowT)
	if err != nil {
		return nil, err
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	gr := g.newGraph()
	for b := from; b < to; b++ {
		if err := gr.blockPass(g.blocks[b], x.Rows, g.wCos, g.wSin, runs, g.vecsFor(b, 8)); err != nil {
			return nil, err
		}
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	out := qwen.NewMat(x.Rows, g.H)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.aX), x.Rows*g.H))
	return out, nil
}

// stepGraph records a forward. lastBlock < 0 runs all of them.
func (g *GPU) stepGraph(video, audio *qwen.Mat, rowT []int32, lastBlock int) (*graph, error) {
	lay := g.lay
	if lay == nil {
		return nil, fmt.Errorf("dit: Step before Begin")
	}
	if video.Rows != len(lay.Video) || video.Cols != g.cfg.Patch() || audio.Rows != len(lay.Audio) || audio.Cols != g.cfg.AudioChannels {
		return nil, fmt.Errorf("dit: video %v / audio %v for a layout of %d / %d rows", video, audio, len(lay.Video), len(lay.Audio))
	}
	runs, tail, err := g.uploadStep(rowT)
	if err != nil {
		return nil, err
	}
	rows := len(lay.Pos)
	g.narrowRows(g.hVid, video, g.ldaVid)
	g.narrowRows(g.hAud, audio, g.ldaAud)
	// The text rows go straight into the residual: nothing is dispatched
	// before this submission, so nothing can overwrite them.
	for i, r := range lay.Text {
		g.abuf.WriteFloat32At(int(g.aX)+int(r)*g.H, g.text.Row(i))
	}
	gr := g.newGraph()
	// The input projections, a contiguous run of rows at a time.
	for _, seg := range segments(lay.Audio) {
		if err := gr.project(g.audIn, g.hAud+uint32(seg.src*g.ldaAud), g.ldaAud, seg.n, seg.row); err != nil {
			return nil, err
		}
	}
	for _, seg := range segments(lay.Video) {
		if err := gr.project(g.vidIn, g.hVid+uint32(seg.src*g.ldaVid), g.ldaVid, seg.n, seg.row); err != nil {
			return nil, err
		}
	}
	last := len(g.blocks)
	if lastBlock >= 0 {
		last = lastBlock
	}
	for b := 0; b < last; b++ {
		if err := gr.blockPass(g.blocks[b], rows, g.wCos, g.wSin, runs, g.vecsFor(b, 8)); err != nil {
			return nil, err
		}
	}
	// The tail: the output norm over the audio and video rows and the two
	// heads, into aOutA/aOutV in the layout's order.
	eps := math.Float32bits(float32(g.cfg.FinalNormEps))
	for _, hd := range []struct {
		rows []int32
		w    headW
		out  uint32
	}{{lay.Audio, g.audOut, g.aOutA}, {lay.Video, g.vidOut, g.aOutV}} {
		for _, seg := range segments(hd.rows) {
			for c0 := 0; c0 < seg.n; c0 += g.chunk {
				n := min(g.chunk, seg.n-c0)
				r0, r1 := seg.row+c0, seg.row+c0+n
				for _, ru := range tail {
					lo, hi := max(ru.start, r0), min(ru.start+ru.n, r1)
					if lo >= hi {
						continue
					}
					pc := gr.base
					pc.Tokens, pc.Dim, pc.LDA = uint32(hi-lo), uint32(g.H), uint32(g.ldaH)
					pc.InOff = g.aX + uint32(lo*g.H)
					pc.OutOff = g.hA + uint32((lo-r0)*g.ldaH)
					pc.Aux0 = g.aModTail + uint32(ru.key*2*g.H)
					pc.Aux1 = pc.Aux0 + uint32(g.H)
					pc.Eps = eps
					gr.add("norm", "tail norm", uint32(hi-lo), 1, pc)
				}
				if err := gr.gemm(gemmSmall, hd.w.bank, "gemm "+hd.w.biasName, g.hA, hd.out+uint32((seg.src+c0)*hd.w.n), hd.w.off, n, hd.w.n, g.H, g.ldaH); err != nil {
					return nil, err
				}
			}
		}
	}
	return gr, nil
}

// segment is a contiguous run of sequence rows [row, row+n) that is rows
// [src, src+n) of a modality's list.
type segment struct{ row, src, n int }

func segments(rows []int32) []segment {
	var out []segment
	for i, r := range rows {
		if n := len(out); n > 0 && out[n-1].row+out[n-1].n == int(r) {
			out[n-1].n++
			continue
		}
		out = append(out, segment{row: int(r), src: i, n: 1})
	}
	return out
}

// readHeads reads the velocity rows back and adds the heads' fp32 biases.
func (g *GPU) readHeads() (v, a *qwen.Mat) {
	read := func(off uint32, rows int, hw headW, bias []float32) *qwen.Mat {
		raw := g.abuf.ReadFloat32At(int(off), rows*hw.n)
		m := qwen.NewMat(rows, hw.realN)
		for r := 0; r < rows; r++ {
			row := m.Row(r)
			for c := range row {
				row[c] = raw[r*hw.n+c] + bias[c]
			}
		}
		return m
	}
	return read(g.aOutV, len(g.lay.Video), g.vidOut, g.host.ProjOut.Bias),
		read(g.aOutA, len(g.lay.Audio), g.audOut, g.host.AudioProjOut.Bias)
}

// Stage is one dispatch's measurement.
type Stage struct {
	Kind  string
	GPU   time.Duration
	Flops float64 // 2× multiply-accumulates; zero for elementwise passes
}

// Profile runs one forward's graph a dispatch at a time, timing each on the
// device. The total is above Step's (a fence a dispatch); the distribution is
// what it is for.
func (g *GPU) Profile(video, audio *qwen.Mat, rowT []int32) ([]Stage, error) {
	gr, err := g.stepGraph(video, audio, rowT, -1)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(gr.d))
	for i := range gr.d {
		d, err := vk.DispatchMultiTimed(gr.d[i:i+1], 1, 1, true)
		if err != nil {
			return out, fmt.Errorf("dit: dispatch %d (%s): %w", i, gr.kinds[i], err)
		}
		out = append(out, Stage{Kind: gr.kinds[i], GPU: d, Flops: gr.flops[i]})
	}
	return out, nil
}

// AttnVariant is one build of the WMMA attention with fp16 context out: QT
// query tiles a wave, KTIL key tiles a block, wave32.
type AttnVariant struct{ QT, KTIL int }

// attnFor is the build for a key count. TestGPUAttentionScreen, block 49's
// planes after a real forward (2026-09-26): at 15,936 keys QT1 KTIL4 is
// 26.3 TFLOP/s and QT2 KTIL4 24.5; at 38,247 keys QT1 is 19.6 and QT2 22.4,
// +14%. Both write bit-identical context. QT4 spills (5.5) and KTIL8 loses
// at both. The switch sits between the two measured points.
func (g *GPU) attnFor(keys int) AttnVariant {
	if g.attnFixed {
		return g.attn
	}
	if keys >= 24576 {
		return AttnVariant{2, 4}
	}
	return AttnVariant{1, 4}
}

// attention records one query chunk's attention of n rows over keys keys.
func (gr *graph) attention(keys, n int, pc pushConstants) {
	v := gr.g.attnFor(keys)
	rowsPer := v.QT * tile
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.attnPipes[v], GroupsX: uint32((n + rowsPer - 1) / rowsPer),
		GroupsY: uint32(gr.g.heads), PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, "attention")
	gr.flops = append(gr.flops, 4*float64(n)*float64(keys)*float64(gr.g.headDim)*float64(gr.g.heads))
}

var attnBuilds = map[AttnVariant][]byte{
	{1, 4}: shaders.DiTAttentionWMMAQT1KT4W32OutF16,
	{1, 8}: shaders.H3AttnQT1KT8,
	{2, 4}: shaders.H3AttnQT2KT4,
	{2, 8}: shaders.H3AttnQT2KT8,
	{4, 4}: shaders.H3AttnQT4KT4,
}

// maxKeyBlock is the longest key block any build reads past its count.
const maxKeyBlock = 8 * tile

// AttnVariants lists the builds a screen chooses between.
func AttnVariants() []AttnVariant {
	return []AttnVariant{{1, 4}, {1, 8}, {2, 4}, {2, 8}, {4, 4}}
}

// SetAttention fixes the attention build every block records, overriding
// attnFor. Every build reads and writes the same planes, so a screen is one
// staging.
func (g *GPU) SetAttention(v AttnVariant) error {
	if _, ok := g.attnPipes[v]; !ok {
		return fmt.Errorf("dit: no attention build %v", v)
	}
	g.attn, g.attnFixed = v, true
	return nil
}

// TimeAttention runs the attention of one block over the planes as they
// stand (after a Step, the last block's), for every chunk of the current
// layout, and returns the device time and the context of the last chunk:
// the screen's instrument.
func (g *GPU) TimeAttention() (time.Duration, []uint16, error) {
	rows := len(g.lay.Pos)
	gr := g.newGraph()
	planeTile := uint32(g.headDim / tile * tile * tile)
	last := 0
	for r0 := 0; r0 < rows; r0 += g.chunk {
		n := min(g.chunk, rows-r0)
		pc := gr.base
		pc.Tokens = uint32(rows)
		pc.InOff = g.hQ + uint32(r0/tile)*planeTile
		pc.KOff, pc.VOff = g.hK, g.hV
		pc.OutOff, pc.LDA = g.hCtx, uint32(g.ldaInner)
		pc.Aux1 = uint32(g.planeRows)
		gr.attention(rows, n, pc)
		last = n
	}
	took, err := gr.submit()
	if err != nil {
		return 0, nil, err
	}
	return took, g.hbuf.ReadUint16At(int(g.hCtx), last*g.ldaInner), nil
}
