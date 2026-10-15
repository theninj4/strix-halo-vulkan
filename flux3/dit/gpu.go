package dit

import (
	"fmt"
	"math"
	"runtime"
	"time"
	"unsafe"

	h3dit "strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU runs FLUX 3 Image Klein's transformer on the device (X5,
// research/flux3-vertical.md), over the kernels the image and video DiTs
// validated: the fp16 fragment-tiled LDS GEMM with its SwiGLU epilogue, the
// fused q/k pack (per-head RMS norm, adjacent-pair RoPE, softmax scale), the
// WMMA flash attention with fp16 context out, and the gated residual. Its
// shape is h3/dit's graph — row chunks through one scratch, an fp32
// residual, per-run modulation — with what FLUX 3 changes:
//
//   - **The block is parallel.** One modulated LayerNorm feeds q, k, v and
//     the SwiGLU; the residual is x + gate·(attn + ff). So a chunk's first
//     pass runs the FFN to completion and adds gate·ff (its q/k/v already in
//     the planes, its residual rows are no longer read), and the second pass
//     adds gate·attn once every key is in. Two fp32 adds instead of one sum.
//   - **Modes.** A mode block runs over its mode's rows only, with its own
//     weights and attention over those rows alone: the same block pass over
//     a row range of the joint [text, image, cond] residual, its planes
//     starting at row 0. A single block is the pass over every row with
//     three runs of modulation.
//   - **The pre-norm is a LayerNorm** (flux3_norm_mod: h3_norm_mod with the
//     mean subtracted), a = 1+scale, b = shift.
//   - **The key tail is zeroed in the graph.** The planes are rewritten at
//     three lengths every forward (each mode's, then the joint one), so a
//     pack of n keys leaves the last key block's rows past n holding a
//     longer pass's keys. The attention reads whole key blocks; zero
//     dispatches clear [n↑16, n↑128) of k and v before each attention, as
//     h3/dit's zeroKeyTail does from the host for its two fixed lengths.
//   - **The conditioning is the host's, in float64.** Every modulation is a
//     single-row GEMV, and X2 measured a float32 sum there as 1e-4 of
//     norm_out; the host's Model.Modulations is the CPU oracle's own code.
//   - **txt_in is the host's, in fp32, once a request.** The text embedding
//     reaches 1.1e4 and its projection 1.1e5 (X0): the fp16 operand would
//     round the massive channel, and the projection does not depend on the
//     step, so it is computed once at Begin and copied in each forward.
//
// The residual never leaves fp32 (text rows reach 1.1e7). Everything the
// matrix cores touch is fp16 with fp32 accumulators; gpu_test.go measures
// what that costs against the fp32 CPU model.
type GPU struct {
	// Between, when set, is called between two submissions, when nothing of
	// this model's is on the device (h3/dit's hook): a server lets other work
	// run inside a forward, and an error abandons the forward there.
	Between func() error

	dev  *vk.Device
	cfg  *Config
	host *Model // no blocks: embedders, modulations, norm_out — the host's fp32 pieces

	wbuf, abuf, hbuf *vk.Buffer
	banks            []*vk.Buffer
	pipes            map[string]*vk.ComputePipeline
	gemms            []map[gemmKernel]*vk.ComputePipeline
	attnPipes        map[attnVariant]*vk.ComputePipeline
	mods             []*vk.ShaderModule
	attnT            bool // the device's element order is h3_attn_t.comp's
	attnProbed       bool

	dim, ffn, heads, headDim, half int
	maxRows, maxText, chunk        int
	planeRows                      int
	ldaDim, ldaFFN, ldaIn          int

	mode   [3][]blockW // per Mode
	single []blockW
	imgIn  headW
	condIn headW
	out    headW // proj_out

	// wbuf (fp32): the rotary table, the q/k norm weights.
	wCos, wSin uint32
	// abuf (fp32): the residual, the chunk scratch, the projected text, the
	// modulation vectors, the velocity.
	aX, aS, aTxt, aMod, aOut uint32
	actElems                 int
	// hbuf (fp16): the planes, the chunk A operands, the latents as the
	// embedders' A operands.
	hQ, hK, hV, hA, hCtx, hFFN, hImg, hCond uint32
	hElems                                  int

	// noKeyTail drops the key-tail zeroing: a negative control
	// (TestGPUControls) and nothing else.
	noKeyTail bool

	// Per request (Begin): each mode's rows of the joint sequence.
	rows   [3]segRows
	begun  bool
	nTotal int
}

// segRows is a mode's rows [start, start+n) of the joint sequence.
type segRows struct{ start, n int }

type headW struct {
	bank int
	off  uint32
	n, k int
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
	// projGateUp is ff.linear_in (gate | up) interleaved in 64-row groups
	// (qwen.InterleaveGLU), the C_SWIGLU epilogue's operand order.
	projGateUp
	projDown
)

var projOrder = []proj{projQ, projK, projV, projO, projGateUp, projDown}

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
	gemmBig    gemmKernel = iota // 128x256, LDS-staged wave32 (KERNELS.md G2)
	gemmBigGLU                   // the same, SwiGLU epilogue (G3)
	gemmSmall                    // 64x64, proj_out's N = 128
)

var gemmVariants = map[gemmKernel]struct {
	spirv  []byte
	bm, bn int
	wave   uint32
}{
	gemmBig:    {shaders.DiTGEMMWG128x256LDSW32, 128, 256, 32},
	gemmBigGLU: {shaders.DiTGEMMWG128x256LDSW32SwiGLU, 128, 256, 32},
	gemmSmall:  {shaders.DiTGEMMReg64Tiled, 64, 64, 0},
}

// attnVariant is an attention build: T is the transposed kernel staging
// K/V through LDS for eight waves (KERNELS.md G4b, h3/dit's default where
// the element order allows), otherwise the plain QT1 KTIL4 build.
type attnVariant struct {
	qt int
	t  bool
}

var attnBuilds = map[attnVariant][]byte{
	{qt: 1}:          shaders.DiTAttentionWMMAQT1KT4W32OutF16,
	{qt: 8, t: true}: shaders.H3AttnTKVLDSW8KT4,
}

const (
	tile     = 16
	gemmPad  = 128
	rowAlign = 128 // chunk and plane granularity: the big GEMM's M tile
	// maxKeyBlock is the longest key block any attention build reads past
	// its count; the tail zeroing covers it.
	maxKeyBlock = 8 * tile
	log2e       = 1.4426950408889634
	// ffScale scales SwiGLU's fp16 output down and its inverse rides in the
	// uploaded FFN gate, as in qimage/dit and h3/dit.
	ffScale      = 1.0 / 16
	maxBankBytes = 0xfffffffc
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// NewGPU stages the transformer, fp16, for joint sequences of up to maxRows
// rows of which at most maxText are text, running the row-local stages
// chunk rows at a time (a multiple of 128). ~11 GB of banks, staged a block
// at a time.
func NewGPU(dev *vk.Device, dir string, maxRows, maxText, chunk int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if chunk <= 0 || chunk%rowAlign != 0 {
		return nil, fmt.Errorf("dit: chunk %d is not a positive multiple of %d", chunk, rowAlign)
	}
	host, err := Load(dir, Parts{})
	if err != nil {
		return nil, err
	}
	g := &GPU{
		dev: dev, cfg: cfg, host: host,
		pipes: map[string]*vk.ComputePipeline{},
		dim:   cfg.Dim(), ffn: cfg.MLP(), heads: cfg.NumHeads, headDim: cfg.HeadDim,
		maxRows: maxRows, maxText: maxText, chunk: chunk,
	}
	g.half = g.headDim / 2
	g.planeRows = roundUp(maxRows, rowAlign)
	g.ldaDim, g.ldaFFN = g.dim+gemmPad, g.ffn+gemmPad
	g.ldaIn = max(cfg.InChannels, cfg.CondInChannels) + gemmPad
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	for _, step := range []func(*safetensors.Set) error{g.stage, g.alloc, g.build} {
		if err := step(set); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

// packB narrows a [n, k] row-major weight into 16x16 fragment tiles,
// kt-fastest (dit_pack_f16.comp's layout).
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
	case projGateUp:
		return [2]int{2 * g.ffn, g.dim}
	case projDown:
		return [2]int{g.dim, g.ffn}
	}
	return [2]int{g.dim, g.dim}
}

// stage lays out the banks and fills them a block at a time.
func (g *GPU) stage(set *safetensors.Set) error {
	cfg := g.cfg
	ropeElems := g.planeRows * g.half
	g.wCos, g.wSin = 0, uint32(ropeElems)
	total32 := 2 * ropeElems

	var bankElems []int
	place := func(n int) (int, uint32) {
		if len(bankElems) == 0 || (bankElems[len(bankElems)-1]+n)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
		}
		b := len(bankElems) - 1
		off := bankElems[b]
		bankElems[b] += n
		return b, uint32(off)
	}
	head := func(n, k int) headW {
		b, off := place(n * k)
		return headW{bank: b, off: off, n: n, k: k}
	}
	g.imgIn = head(g.dim, cfg.InChannels)
	g.condIn = head(g.dim, cfg.CondInChannels)
	g.out = head(cfg.InChannels, g.dim)
	perBlock := 0
	for _, p := range projOrder {
		sh := g.projShape(p)
		perBlock += sh[0] * sh[1]
	}
	newBlock := func() blockW {
		w := blockW{off: map[proj]uint32{}}
		w.bank, _ = place(0)
		if (bankElems[w.bank]+perBlock)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			w.bank = len(bankElems) - 1
		}
		for _, p := range projOrder {
			sh := g.projShape(p)
			w.off[p] = uint32(bankElems[w.bank])
			bankElems[w.bank] += sh[0] * sh[1]
		}
		w.normQ = uint32(total32)
		w.normK = w.normQ + uint32(g.headDim)
		total32 += 2 * g.headDim
		return w
	}
	var prefixes []string
	var ws []*blockW
	for _, md := range Modes {
		g.mode[md] = make([]blockW, cfg.NumLayers)
		for i := range g.mode[md] {
			g.mode[md][i] = newBlock()
			prefixes = append(prefixes, fmt.Sprintf("%s.%d", md.blockPrefix(), i))
			ws = append(ws, &g.mode[md][i])
		}
	}
	g.single = make([]blockW, cfg.NumSingle)
	for i := range g.single {
		g.single[i] = newBlock()
		prefixes = append(prefixes, fmt.Sprintf("single_transformer_blocks.%d", i))
		ws = append(ws, &g.single[i])
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
	put := func(bank int, off uint32, w []float32, n, k int) {
		buf := make([]uint16, n*k)
		packB(buf, w, n, k)
		g.banks[bank].WriteUint16At(int(off), buf)
	}
	h := g.host
	put(g.imgIn.bank, g.imgIn.off, h.ImgIn.Weight, g.imgIn.n, g.imgIn.k)
	put(g.condIn.bank, g.condIn.off, h.CondIn.Weight, g.condIn.n, g.condIn.k)
	put(g.out.bank, g.out.off, h.ProjOut.Weight, g.out.n, g.out.k)

	// A block is ~1 GB of fp32 on the host and none of it is wanted once
	// narrowed; collected per block so the dead ones never pile up beside a
	// server's live heap (VIDEO.md M11a).
	l := &loader{set: set}
	for i, prefix := range prefixes {
		runtime.GC()
		blk := l.block(prefix, cfg)
		if l.err != nil {
			return l.err
		}
		w := ws[i]
		g.wbuf.WriteFloat32At(int(w.normQ), blk.NormQ.Weight)
		g.wbuf.WriteFloat32At(int(w.normK), blk.NormK.Weight)
		gu := make([]float32, 2*g.ffn*g.dim)
		qwen.InterleaveGLU(gu, blk.FFIn.Weight[:g.ffn*g.dim], blk.FFIn.Weight[g.ffn*g.dim:], g.ffn, g.dim)
		for p, wt := range map[proj][]float32{
			projQ: blk.Q.Weight, projK: blk.K.Weight, projV: blk.V.Weight, projO: blk.Out.Weight,
			projGateUp: gu, projDown: blk.FFOut.Weight,
		} {
			sh := g.projShape(p)
			put(w.bank, w.off[p], wt, sh[0], sh[1])
		}
	}
	return nil
}

// alloc lays out and allocates the activation arenas, refusing a layout past
// the storage-buffer range (a clamped binding reads zeros, silently).
func (g *GPU) alloc(*safetensors.Set) error {
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += roundUp(n, 64)
		return off
	}
	rows := g.planeRows + rowAlign // the embedders' last tile runs past the sequence
	g.aX = alloc(rows * g.dim)
	// The chunk scratch: q, k, v and the FFN's output in the first pass; the
	// attention output (over q) in the second.
	g.aS = alloc(g.chunk * 4 * g.dim)
	g.aTxt = alloc(g.maxText * g.dim)
	g.aMod = alloc(modVectors * g.dim)
	g.aOut = alloc(rows * g.out.n)

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += roundUp(n, 64)
		return off
	}
	plane := g.heads * g.planeRows * g.headDim
	g.hQ, g.hK, g.hV = halloc(plane), halloc(plane), halloc(plane)
	g.hA = halloc(g.chunk * g.ldaDim)
	g.hCtx = halloc(g.chunk * g.ldaDim)
	g.hFFN = halloc(g.chunk * g.ldaFFN)
	g.hImg = halloc(rows * g.ldaIn)
	g.hCond = halloc(rows * g.ldaIn)

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
	// Pad rows and columns of every A operand read as zeros.
	g.hbuf.ZeroUint16At(0, g.hElems)
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
		"norm":   shaders.Flux3NormMod,
		"qkpack": shaders.DiTQKPackTPW8HX,
		"pack":   shaders.DiTPackF16TPW8,
		"gate":   shaders.DiTGateAddHoist,
		"zero":   shaders.ZeroF16,
		"copy":   shaders.DiTCopy,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("dit: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return err
	}
	if !g.dev.Features().SubgroupSizeControl || !sgs.Supported || 32 < sgs.MinSubgroupSize || 32 > sgs.MaxSubgroupSize {
		return fmt.Errorf("dit: the attention and GEMM builds need a pinned wave32")
	}
	g.attnPipes = map[attnVariant]*vk.ComputePipeline{}
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
			p, err := newPipe(v.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]},
				PushConstantSize: pcSize, RequiredSubgroupSize: v.wave})
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

// Begin prepares one request: the text embedding [nText, 20480] through
// txt_in on the host, the rotary table of the joint [text, image, cond]
// sequence from each mode's ids ([rows, 4], image h/w already scaled), and
// the reference tokens [nCond, 128] (nil for t2i) as the cond embedder's
// operand. Forward then needs only the image tokens and the two times.
func (g *GPU) Begin(txt, txtIDs, imgIDs, cond, condIDs *qwen.Mat) error {
	g.begun = false
	if !g.attnProbed {
		ok, err := h3dit.TransposedAttentionOK(g.dev)
		if err != nil {
			return err
		}
		g.attnT, g.attnProbed = ok, true
	}
	if txt == nil || txt.Rows != txtIDs.Rows || imgIDs == nil {
		return fmt.Errorf("dit: Begin needs the text embedding with its ids and the image ids")
	}
	if txt.Rows > g.maxText {
		return fmt.Errorf("dit: %d text rows, staged for %d", txt.Rows, g.maxText)
	}
	nCond := 0
	if cond != nil {
		if condIDs == nil || cond.Rows != condIDs.Rows || cond.Cols != g.condIn.k {
			return fmt.Errorf("dit: reference tokens %v with ids %v", cond, condIDs)
		}
		nCond = cond.Rows
	}
	total := txt.Rows + imgIDs.Rows + nCond
	if total > g.maxRows {
		return fmt.Errorf("dit: %d joint rows, staged for %d", total, g.maxRows)
	}
	g.rows[Text] = segRows{0, txt.Rows}
	g.rows[Image] = segRows{txt.Rows, imgIDs.Rows}
	g.rows[Cond] = segRows{txt.Rows + imgIDs.Rows, nCond}
	g.nTotal = total

	p, err := g.host.TxtIn.Apply(txt)
	if err != nil {
		return err
	}
	g.abuf.WriteFloat32At(int(g.aTxt), p.Data)
	ids := []*qwen.Mat{txtIDs, imgIDs}
	if nCond > 0 {
		ids = append(ids, condIDs)
		g.narrowRows(g.hCond, cond, g.ldaIn)
	}
	var ropes []*Rope
	for _, m := range ids {
		r, err := NewRope(m, g.cfg.AxesDimsRope, g.cfg.RopeTheta)
		if err != nil {
			return err
		}
		ropes = append(ropes, r)
	}
	rope := Concat(ropes...)
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos.Data)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin.Data)
	g.begun = true
	return nil
}

// narrowRows writes an fp32 host matrix as an fp16 A operand of row stride
// lda at hOff.
func (g *GPU) narrowRows(hOff uint32, m *qwen.Mat, lda int) {
	buf := make([]uint16, m.Rows*lda)
	for r := 0; r < m.Rows; r++ {
		for c, v := range m.Row(r) {
			buf[r*lda+c] = safetensors.F32ToF16(v)
		}
	}
	g.hbuf.WriteUint16At(int(hOff), buf)
}

// The modulation vectors' slots in aMod: per mode and stage, a = 1+scale,
// b = shift, g = gate and g/ffScale (the FFN's add, undoing ffScale), then
// the output layer's a and b.
const (
	vecsPerSet = 4
	modVectors = 3*2*vecsPerSet + 2
)

// modVecs is where one run's four vectors live.
type modVecs [vecsPerSet]uint32

func (g *GPU) vecs(md Mode, single bool) modVecs {
	set := int(md) * 2
	if single {
		set++
	}
	var v modVecs
	for j := range v {
		v[j] = g.aMod + uint32((set*vecsPerSet+j)*g.dim)
	}
	return v
}

func (g *GPU) tailVecs() (a, b uint32) {
	o := g.aMod + uint32(3*2*vecsPerSet*g.dim)
	return o, o + uint32(g.dim)
}

// putMod writes one Mod's four vectors at v.
func (g *GPU) putMod(v modVecs, m Mod) {
	a := make([]float32, g.dim)
	gs := make([]float32, g.dim)
	for i := range a {
		a[i] = 1 + m.Scale[i]
		gs[i] = m.Gate[i] / ffScale
	}
	g.abuf.WriteFloat32At(int(v[0]), a)
	g.abuf.WriteFloat32At(int(v[1]), m.Shift)
	g.abuf.WriteFloat32At(int(v[2]), m.Gate)
	g.abuf.WriteFloat32At(int(v[3]), gs)
}

// uploadMods computes a forward's modulations on the host (float64 GEMVs,
// the CPU oracle's code) and writes every vector the graph reads.
func (g *GPU) uploadMods(t, tNext float64) error {
	mods, err := g.host.Modulations(t, tNext)
	if err != nil {
		return err
	}
	for _, md := range Modes {
		g.putMod(g.vecs(md, false), mods.Early[md])
		g.putMod(g.vecs(md, true), mods.Single[md])
	}
	act := make([]float32, len(mods.ImageTemb))
	for i, v := range mods.ImageTemb {
		act[i] = silu(v)
	}
	e, err := gemv(g.host.NormOut, act)
	if err != nil {
		return err
	}
	scale, shift := e[:g.dim], e[g.dim:] // scale first, unlike the blocks
	a := make([]float32, g.dim)
	for i := range a {
		a[i] = 1 + scale[i]
	}
	ta, tb := g.tailVecs()
	g.abuf.WriteFloat32At(int(ta), a)
	g.abuf.WriteFloat32At(int(tb), shift)
	return nil
}

// graph accumulates one submission's dispatches.
type graph struct {
	g     *GPU
	d     []vk.MultiDispatch
	kinds []string
	flops []float64
	base  pushConstants
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
// (row stride lda) × B, the fragment-tiled weight at bOff of bank; tweak
// sets what an epilogue build reads beyond the GEMM's own constants.
func (gr *graph) gemm(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, m, n, k, lda int, tweak func(*pushConstants)) error {
	v := gemmVariants[kernel]
	if n%v.bn != 0 || k%(2*tile) != 0 {
		return fmt.Errorf("dit: %s: N=%d K=%d do not tile by %d/%d", kind, n, k, v.bn, 2*tile)
	}
	mPad := roundUp(m, v.bm)
	pc := gr.base
	pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(mPad), uint32(n), uint32(k), uint32(lda)
	if tweak != nil {
		tweak(&pc)
	}
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.gemms[bank][kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(mPad / v.bm), PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, 2*float64(mPad)*float64(n)*float64(k))
	return nil
}

// The submission budget, h3/dit's: amdgpu kills a submission that holds the
// ring for 2 s, so dispatches are priced at their FLOPs over submitRate plus
// a flat cost, and a submission closes before it would pass the budget.
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
		if i > 0 && gr.g.Between != nil {
			if err := gr.g.Between(); err != nil {
				return total, err
			}
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

// run is a stretch of a pass's rows (pass-local) sharing one modulation.
type run struct {
	start, n int
	v        modVecs
}

// blockPass records one block over residual rows [r0, r0+rows), with
// attention over exactly those rows (their planes start at plane row 0) and
// each run's modulation.
func (gr *graph) blockPass(w blockW, r0, rows int, runs []run) error {
	g := gr.g
	eps := math.Float32bits(float32(g.cfg.Eps))
	aQ := g.aS
	aK := aQ + uint32(g.chunk*g.dim)
	aV := aK + uint32(g.chunk*g.dim)
	aFF := aV + uint32(g.chunk*g.dim)
	aAttn := aQ
	planeTile := uint32(g.headDim / tile * tile * tile) // halves a 16-row tile of one head

	// forRuns calls fn for each run's part of chunk rows [c0, c1).
	forRuns := func(c0, c1 int, fn func(lo, hi int, v modVecs)) {
		for _, ru := range runs {
			if lo, hi := max(ru.start, c0), min(ru.start+ru.n, c1); lo < hi {
				fn(lo, hi, ru.v)
			}
		}
	}
	gate := func(kind string, c0, c1 int, y uint32, which int) {
		forRuns(c0, c1, func(lo, hi int, v modVecs) {
			pc := gr.base
			pc.Tokens, pc.Dim = uint32(hi-lo), uint32(g.dim)
			pc.InOff = y + uint32((lo-c0)*g.dim)
			pc.OutOff = g.aX + uint32((r0+lo)*g.dim)
			pc.Aux0, pc.Aux2 = v[which], 1
			gr.add("gate", kind, uint32(hi-lo), 1, pc)
		})
	}

	// Pass 1, a chunk at a time: the norm, q/k/v into the planes, and the
	// whole FFN, whose gated add the residual takes at once.
	for c0 := 0; c0 < rows; c0 += g.chunk {
		c1 := min(c0+g.chunk, rows)
		n := c1 - c0
		tiles := roundUp(n, tile) / tile
		tileOff := uint32(c0/tile) * planeTile
		forRuns(c0, c1, func(lo, hi int, v modVecs) {
			pc := gr.base
			pc.Tokens, pc.Dim, pc.LDA = uint32(hi-lo), uint32(g.dim), uint32(g.ldaDim)
			pc.InOff = g.aX + uint32((r0+lo)*g.dim)
			pc.OutOff = g.hA + uint32((lo-c0)*g.ldaDim)
			pc.Aux0, pc.Aux1, pc.Eps = v[0], v[1], eps
			gr.add("norm", "norm", uint32(hi-lo), 1, pc)
		})
		for _, pr := range []struct {
			p   proj
			out uint32
		}{{projQ, aQ}, {projK, aK}, {projV, aV}} {
			if err := gr.gemm(gemmBig, w.bank, "gemm qkv", g.hA, pr.out, w.off[pr.p], n, g.dim, g.dim, g.ldaDim, nil); err != nil {
				return err
			}
		}
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
			pc.Tokens, pc.Dim = uint32(n), uint32(g.dim)
			pc.InOff, pc.OutOff = qk.src, qk.dst+tileOff
			pc.WOff = g.wCos + uint32((r0+c0)*g.half)
			pc.Aux0 = g.wSin + uint32((r0+c0)*g.half)
			pc.Aux1, pc.Aux2 = uint32(g.planeRows), qk.norm
			pc.Span, pc.Eps, pc.Scale = uint32(tiles), eps, math.Float32bits(qk.scale)
			// HEAD_X: a head across, a token tile down (KERNELS.md G6).
			gr.add("qkpack", qk.kind, uint32(g.heads), uint32((tiles+7)/8), pc)
		}
		pc := gr.base
		pc.Tokens, pc.Dim = uint32(n), uint32(g.dim)
		pc.InOff, pc.OutOff = aV, g.hV+tileOff
		pc.Aux0, pc.Aux1, pc.Aux2 = 1, uint32(g.planeRows), uint32(tiles)
		pc.Scale = math.Float32bits(1)
		gr.add("pack", "pack v", uint32((tiles+7)/8), uint32(g.heads), pc)

		if err := gr.gemm(gemmBigGLU, w.bank, "gemm gateup", g.hA, g.hFFN, w.off[projGateUp], n, 2*g.ffn, g.dim, g.ldaDim,
			func(pc *pushConstants) { pc.Aux0, pc.Scale = uint32(g.ldaFFN), math.Float32bits(ffScale) }); err != nil {
			return err
		}
		if err := gr.gemm(gemmBig, w.bank, "gemm down", g.hFFN, aFF, w.off[projDown], n, g.dim, g.ffn, g.ldaFFN, nil); err != nil {
			return err
		}
		gate("gate ff", c0, c1, aFF, 3)
	}

	// The key tail: k and v rows from the last pack's tile to the end of the
	// last key block an attention build reads, which a longer pass of this
	// forward (or the last one) left full.
	if from, to := roundUp(rows, tile), min(roundUp(rows, maxKeyBlock), g.planeRows); from < to && !g.noKeyTail {
		for _, pl := range []uint32{g.hK, g.hV} {
			pc := gr.base
			pc.Tokens, pc.Dim, pc.LDA = uint32(g.heads), uint32((to-from)*g.headDim), uint32(g.planeRows*g.headDim)
			pc.OutOff = pl + uint32(from*g.headDim)
			gr.add("zero", "zero key tail", uint32(g.heads), 1, pc)
		}
	}

	// Pass 2: attention over every key, the output projection, its add.
	v := attnVariant{qt: 1}
	if g.attnT {
		v = attnVariant{qt: 8, t: true}
	}
	for c0 := 0; c0 < rows; c0 += g.chunk {
		c1 := min(c0+g.chunk, rows)
		n := c1 - c0
		pc := gr.base
		pc.Tokens = uint32(rows)
		pc.InOff = g.hQ + uint32(c0/tile)*planeTile
		pc.KOff, pc.VOff = g.hK, g.hV
		pc.OutOff, pc.LDA = g.hCtx, uint32(g.ldaDim)
		pc.Aux1 = uint32(g.planeRows)
		per := v.qt * tile
		gr.d = append(gr.d, vk.MultiDispatch{Pipeline: g.attnPipes[v], GroupsX: uint32((n + per - 1) / per),
			GroupsY: uint32(g.heads), PushConstants: pc.bytes()})
		gr.kinds = append(gr.kinds, "attention")
		gr.flops = append(gr.flops, 4*float64(n)*float64(rows)*float64(g.headDim)*float64(g.heads))
		if err := gr.gemm(gemmBig, w.bank, "gemm o", g.hCtx, aAttn, w.off[projO], n, g.dim, g.dim, g.ldaDim, nil); err != nil {
			return err
		}
		gate("gate attn", c0, c1, aAttn, 2)
	}
	return nil
}

// embed records the residual's first state: the projected text copied in,
// then the image and reference tokens through their embedders. Ascending
// rows, because a GEMM writes whole 128-row tiles and runs past its rows
// into the next mode's, which that mode's own write then covers.
func (gr *graph) embed() error {
	g := gr.g
	pc := gr.base
	pc.Tokens, pc.Dim = uint32(g.rows[Text].n), uint32(g.dim)
	pc.InOff, pc.OutOff = g.aTxt, g.aX
	gr.add("copy", "text rows", uint32((g.rows[Text].n*g.dim+255)/256), 1, pc)
	for _, e := range []struct {
		md  Mode
		w   headW
		src uint32
	}{{Image, g.imgIn, g.hImg}, {Cond, g.condIn, g.hCond}} {
		s := g.rows[e.md]
		if s.n == 0 {
			continue
		}
		if err := gr.gemm(gemmBig, e.w.bank, "gemm "+e.md.String()+" in", e.src, g.aX+uint32(s.start*g.dim), e.w.off,
			s.n, e.w.n, e.w.k, g.ldaIn, nil); err != nil {
			return err
		}
	}
	return nil
}

// tail records norm_out and proj_out over the image rows into aOut.
func (gr *graph) tail() error {
	g := gr.g
	s := g.rows[Image]
	ta, tb := g.tailVecs()
	for c0 := 0; c0 < s.n; c0 += g.chunk {
		n := min(g.chunk, s.n-c0)
		pc := gr.base
		pc.Tokens, pc.Dim, pc.LDA = uint32(n), uint32(g.dim), uint32(g.ldaDim)
		pc.InOff = g.aX + uint32((s.start+c0)*g.dim)
		pc.OutOff = g.hA
		pc.Aux0, pc.Aux1, pc.Eps = ta, tb, math.Float32bits(float32(g.cfg.Eps))
		gr.add("norm", "norm_out", uint32(n), 1, pc)
		if err := gr.gemm(gemmSmall, g.out.bank, "gemm proj_out", g.hA, g.aOut+uint32(c0*g.out.n), g.out.off,
			n, g.out.n, g.dim, g.ldaDim, nil); err != nil {
			return err
		}
	}
	return nil
}

// forwardGraph uploads a forward's inputs and records its graph.
func (g *GPU) forwardGraph(img *qwen.Mat, t, tNext float64) (*graph, error) {
	if !g.begun {
		return nil, fmt.Errorf("dit: Forward before Begin")
	}
	if img.Rows != g.rows[Image].n || img.Cols != g.imgIn.k {
		return nil, fmt.Errorf("dit: image tokens %v for a request of %d", img, g.rows[Image].n)
	}
	if err := g.uploadMods(t, tNext); err != nil {
		return nil, err
	}
	g.narrowRows(g.hImg, img, g.ldaIn)
	gr := g.newGraph()
	if err := gr.embed(); err != nil {
		return nil, err
	}
	for _, md := range Modes {
		s := g.rows[md]
		if s.n == 0 {
			continue
		}
		for _, w := range g.mode[md] {
			if err := gr.blockPass(w, s.start, s.n, []run{{0, s.n, g.vecs(md, false)}}); err != nil {
				return nil, err
			}
		}
	}
	var runs []run
	for _, md := range Modes {
		if s := g.rows[md]; s.n > 0 {
			runs = append(runs, run{s.start, s.n, g.vecs(md, true)})
		}
	}
	for _, w := range g.single {
		if err := gr.blockPass(w, 0, g.nTotal, runs); err != nil {
			return nil, err
		}
	}
	return gr, gr.tail()
}

// Forward returns the velocity of the image tokens [nImage, 128] at time t
// in a step that jumps to tNext, and the device time.
func (g *GPU) Forward(img *qwen.Mat, t, tNext float64) (*qwen.Mat, time.Duration, error) {
	gr, err := g.forwardGraph(img, t, tNext)
	if err != nil {
		return nil, 0, err
	}
	took, err := gr.submit()
	if err != nil {
		return nil, took, err
	}
	n := g.rows[Image].n
	out := qwen.NewMat(n, g.out.n)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.aOut), n*g.out.n))
	return out, took, nil
}

// Stage is one dispatch's measurement.
type Stage struct {
	Kind  string
	GPU   time.Duration
	Flops float64 // 2x multiply-accumulates; zero for elementwise passes
}

// Profile runs one forward a dispatch at a time, timing each on the device.
// The total is above Forward's (a fence a dispatch); the distribution is
// what it is for.
func (g *GPU) Profile(img *qwen.Mat, t, tNext float64) ([]Stage, error) {
	gr, err := g.forwardGraph(img, t, tNext)
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

// runBlock runs one staged block over x with the given segments' modulation
// and rotary table — the per-block instrument the gates feed exact dumped
// inputs to. It overwrites the request's rotary table and modulation.
func (g *GPU) runBlock(w blockW, x *qwen.Mat, segs []Segment, rope *Rope) (*qwen.Mat, error) {
	if x.Rows > g.maxRows || len(segs) > 3 || rope.Cos.Rows != x.Rows {
		return nil, fmt.Errorf("dit: block over %v with %d segments, rope %v", x, len(segs), rope.Cos)
	}
	if !g.attnProbed {
		ok, err := h3dit.TransposedAttentionOK(g.dev)
		if err != nil {
			return nil, err
		}
		g.attnT, g.attnProbed = ok, true
	}
	g.wbuf.WriteFloat32At(int(g.wCos), rope.Cos.Data)
	g.wbuf.WriteFloat32At(int(g.wSin), rope.Sin.Data)
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	var runs []run
	start := 0
	for i, s := range segs {
		v := g.vecs(Mode(i), false)
		g.putMod(v, s.Mod)
		runs = append(runs, run{start, s.Rows, v})
		start += s.Rows
	}
	if start != x.Rows {
		return nil, fmt.Errorf("dit: segments cover %d of %d rows", start, x.Rows)
	}
	gr := g.newGraph()
	if err := gr.blockPass(w, 0, x.Rows, runs); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	out := qwen.NewMat(x.Rows, g.dim)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.aX), x.Rows*g.dim))
	return out, nil
}

// newControlPipe builds a pipeline a control swaps in for one of the graph's
// own: "h3_norm_mod" is the RMS pre-norm (no mean subtracted).
func newControlPipe(g *GPU, name string) (*vk.ComputePipeline, error) {
	spirv := map[string][]byte{"h3_norm_mod": shaders.H3NormMod}[name]
	if spirv == nil {
		return nil, fmt.Errorf("dit: no control build %q", name)
	}
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, err
	}
	g.mods = append(g.mods, mod)
	p, err := g.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf},
		PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))})
	if err == nil {
		g.pipes["control "+name] = p
	}
	return p, err
}
