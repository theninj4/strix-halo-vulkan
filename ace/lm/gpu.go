package lm

// The 5 Hz LM on the device (MUSIC.md A7a/A7b): a dense Qwen3-4B with a KV
// cache, for prefill and decode.
//
// Nothing in the repo decoded a dense Qwen3 before this: zimage/qwen is
// prefill-only, Kev is Qwen3.5 (GDN + attention) and the LLM is MoE. The
// pieces all exist, though, and they read the same weight:
//
//   - the projections are staged once, fp16 in the §2.8 fragment tiling,
//     which is both zimage/qwen's layout 2 (the DiT GEMM rungs, for a pass of
//     many rows) and llm_gemv.comp's fp16 bank (split-K GEMV at 1-3 rows, the
//     decode path: a step is one row, or two for CFG);
//   - the norms, SwiGLU and residual adds are the DiT's scalar kernels;
//   - what is new is ace_lm_prep.comp (q/k norm, NeoX rope, the KV-cache
//     write) and ace_lm_attn.comp (causal GQA over the cache, split over the
//     keys, and its combine).
//
// A pass is any set of rows, each a token of some slot (KV cache) at some
// position. Attention reads keys only from the cache, which prep writes
// before it runs, so a prompt can be prefilled in several passes, the CFG
// pair's two prompts can share one, and a decode step is a pass of one row
// per slot. The logits come from a pass of at most 3 rows (the head is a
// GEMV); a prefill leaves its last token for the first decode step.
//
// The residual is fp32; every GEMM/GEMV operand is fp16 with fp32 sums.

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The model's shape, which the shaders compile in.
const (
	hidden  = 2560
	nq      = 32
	nkv     = 8
	hd      = 128
	ffn     = 9728
	qWidth  = nq * hd
	kvWidth = nkv * hd

	tile         = 16
	gemmPad      = 128 // A-operand row pad in halves (zimage/qwen's §2.3)
	rowAlign     = 128 // the arena's rows: the widest GEMM tile's BM
	maxBankBytes = 0xfffffffc
	pushBytes    = 256 // the LLM's 64-uint block; the dit_common kernels read its first 88 bytes

	// MaxRows is llm_gemv's MAXROWS: a pass of at most this many rows runs
	// its projections as GEMVs and can return logits.
	MaxRows = 3
	// keyBlock is how many keys ace_lm_attn scores at once (one per lane).
	keyBlock = 64
	// partBudget caps (row, head, chunk) partials a pass writes.
	partBudget = 1 << 16
	partStride = 2 + hd

	// ffScale narrows the SwiGLU product into fp16 and the residual add
	// undoes it. Qwen3's massive activations put layer 16's product at
	// 6.39e4 on a newline of the phase-2 prompt -- 2.4% under fp16's
	// 65,504 -- and layer 6's at 1.2e4; the residual reaches 1.1e6 (fp32).
	// 1/16 is H3's factor (VIDEO.md), 16x of headroom over the peak seen.
	ffScale = 1.0 / 16
)

// push mirrors dit_common.glsl's block.
type push struct {
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

func (p push) bytes() []byte {
	out := make([]byte, pushBytes)
	*(*push)(unsafe.Pointer(&out[0])) = p
	return out
}

// gemvPush is the fields of llm_common.glsl's block llm_gemv.comp reads, at
// their indices.
type gemvPush struct {
	resOff, xnOff, outOff, bOff uint32
	lda, gemmN, gemmK           uint32
}

func (p gemvPush) bytes() []byte {
	var u [pushBytes / 4]uint32
	u[0], u[1], u[4], u[6] = p.resOff, p.xnOff, p.outOff, p.bOff
	u[7] = 0xffffffff // gateOff: NO_W
	u[12], u[15], u[16] = p.lda, p.gemmN, p.gemmK
	out := make([]byte, pushBytes)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), pushBytes))
	return out
}

// proj is one of a layer's seven projections.
type proj int

const (
	pQ proj = iota
	pK
	pV
	pO
	pGate
	pUp
	pDown
	nProj
)

var projNames = [nProj]string{"q", "k", "v", "o", "gate", "up", "down"}

// projShape is [out, in].
var projShape = [nProj][2]int{
	pQ: {qWidth, hidden}, pK: {kvWidth, hidden}, pV: {kvWidth, hidden}, pO: {hidden, qWidth},
	pGate: {ffn, hidden}, pUp: {ffn, hidden}, pDown: {hidden, ffn},
}

// gemvSlabs is each projection's split-K: enough workgroups to fill the
// device at one row (MUSIC.md A10 measures the ladder).
var gemvSlabs = [nProj]int{pQ: 8, pK: 20, pV: 20, pO: 16, pGate: 4, pUp: 4, pDown: 8}

var gemvSPIRV = map[int][]byte{
	1: shaders.LLMGEMVK1, 2: shaders.LLMGEMVK2, 4: shaders.LLMGEMVK4, 8: shaders.LLMGEMVK8,
	16: shaders.LLMGEMVK16, 20: shaders.LLMGEMVK20, 32: shaders.LLMGEMVK32, 40: shaders.LLMGEMVK40,
}

var gemvSumSPIRV = map[int][]byte{
	2: shaders.LLMGEMVSumK2, 4: shaders.LLMGEMVSumK4, 8: shaders.LLMGEMVSumK8,
	16: shaders.LLMGEMVSumK16, 20: shaders.LLMGEMVSumK20, 32: shaders.LLMGEMVSumK32, 40: shaders.LLMGEMVSumK40,
}

// gemmKernel is a DiT GEMM rung over the fragment-tiled weight.
type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	waves  int
}

// gemmKernels are the rungs zimage/qwen's PlanFor picks between for a
// 2560-wide Qwen3, and gemmFor its schedule (as Kev uses it).
var gemmKernels = []gemmKernel{
	{"reg32x128", shaders.DiTGEMMReg32x128Tiled, 32, 128, 1},
	{"reg64", shaders.DiTGEMMReg64Tiled, 64, 64, 1},
	{"wg128x256", shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256, 4},
}

func gemmFor(rows int) gemmKernel {
	switch {
	case rows <= 96:
		return gemmKernels[0]
	case rows <= 192:
		return gemmKernels[1]
	default:
		return gemmKernels[2]
	}
}

type layerWeights struct {
	bank int
	off  [nProj]uint32 // halves, in the bank
	// fp32 arena
	attnNorm, ffnNorm, qkNorm uint32
	// fp16 arena: this layer's caches, [slot][pos][8][128]
	kc, vc uint32
}

// Options size the device state.
type Options struct {
	// MaxLen is the positions a slot holds: prompt, CoT and codes. A 10-minute
	// song is 3,000 codes after a prompt of up to ~2,500 tokens.
	MaxLen int
	// Slots is how many sequences hold a cache at once: 2 for CFG.
	Slots int
	// Rows is the longest pass; a longer prefill runs as several.
	Rows int
}

// DefaultOptions fits a 10-minute song with a long lyric sheet, CFG pair.
func DefaultOptions() Options { return Options{MaxLen: 6144, Slots: 2, Rows: 512} }

// GPU is the 5 Hz LM, resident.
type GPU struct {
	Cfg *qwen.Config
	o   Options

	dev   *vk.Device
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	banks []*vk.Buffer
	head  int // the bank holding the tied head, [vocabPad][2560] tiled
	pipes map[string]*vk.ComputePipeline
	// per bank: GEMM rungs, and GEMV builds by "k<slabs>_r<rows>"
	gemms []map[string]*vk.ComputePipeline
	gemvs []map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	embed    []uint16 // bf16 [vocab][2560], gathered on the host
	vocab    int
	vocabPad int

	w         []layerWeights
	finalNorm uint32
	rope      uint32 // cos [MaxLen][64], then sin

	arenaRows int
	// fp32 arena
	aX, aQ, aK, aV, aO, aGate, aUp, aFF, aMeta, aPart, aRes, aLogits uint32
	aUnscale                                                         uint32 // [2560] of 1/ffScale, the residual add's gate
	actElems                                                         int
	// fp16 arena
	hA, hQ, hCtx, hFFN uint32
	hElems             int
	ldaDim, ldaQ, ldaF int

	// LayersPerSubmit is how many layers go in one command buffer.
	LayersPerSubmit int
}

// Load stages the LM from its checkpoint directory.
func Load(dev *vk.Device, dir string, o Options) (*GPU, error) {
	cfg, err := qwen.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.HiddenSize != hidden || cfg.NumHeads != nq || cfg.NumKVHeads != nkv || cfg.HeadDim != hd ||
		cfg.IntermediateSize != ffn {
		return nil, fmt.Errorf("lm: %+v is not the shape the kernels are built for", *cfg)
	}
	if o.MaxLen <= 0 || o.Slots <= 0 || o.Rows <= 0 {
		return nil, fmt.Errorf("lm: options %+v", o)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	if set.Has("model.embed_tokens.weight") {
		cfg.Prefix = "model."
	}
	g := &GPU{
		Cfg: cfg, o: o, dev: dev,
		pipes:           map[string]*vk.ComputePipeline{},
		ldaDim:          hidden + gemmPad,
		ldaQ:            qWidth + gemmPad,
		ldaF:            ffn + gemmPad,
		arenaRows:       roundUp(o.Rows, rowAlign),
		LayersPerSubmit: 6,
	}
	if err := g.layout(set); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(set); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

func (g *GPU) layout(set *safetensors.Set) error {
	c := g.Cfg
	emb, err := set.Get(c.Prefix + "embed_tokens.weight")
	if err != nil {
		return err
	}
	if len(emb.Shape) != 2 || emb.Shape[1] != hidden || emb.DType != safetensors.BF16 {
		return fmt.Errorf("lm: embed_tokens is %v %s, want [vocab 2560] bf16", emb.Shape, emb.DType)
	}
	g.vocab = emb.Shape[0]
	g.vocabPad = roundUp(g.vocab, tile)

	// fp16 banks: the layers in order, a new bank when one is full, and the
	// head in a bank of its own.
	perLayer := 0
	for _, s := range projShape {
		perLayer += s[0] * s[1]
	}
	var bankElems []int
	g.w = make([]layerWeights, c.NumLayers)
	for i := range g.w {
		if len(bankElems) == 0 || (bankElems[len(bankElems)-1]+perLayer)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
		}
		b := len(bankElems) - 1
		g.w[i].bank = b
		for p, s := range projShape {
			g.w[i].off[p] = uint32(bankElems[b])
			bankElems[b] += s[0] * s[1]
		}
	}
	g.head = len(bankElems)
	bankElems = append(bankElems, g.vocabPad*hidden)
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("lm: fp16 bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}

	// fp32 weights: the norms and the rope table.
	w32 := 0
	for i := range g.w {
		g.w[i].attnNorm = uint32(w32)
		g.w[i].ffnNorm = uint32(w32 + hidden)
		g.w[i].qkNorm = uint32(w32 + 2*hidden)
		w32 += 2*hidden + 2*hd
	}
	g.finalNorm = uint32(w32)
	w32 += hidden
	g.rope = uint32(w32)
	w32 += 2 * g.o.MaxLen * hd / 2
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return err
	}

	// fp32 activations, host-cached: the host reads the logits every step.
	rows := g.arenaRows
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aX = alloc(rows * hidden)
	g.aQ = alloc(rows * qWidth)
	g.aK = alloc(rows * kvWidth)
	g.aV = alloc(rows * kvWidth)
	g.aO = alloc(rows * hidden)
	g.aGate = alloc(rows * ffn)
	g.aUp = alloc(rows * ffn)
	g.aFF = alloc(rows * hidden)
	g.aMeta = alloc(rows * 2)
	g.aPart = alloc(partBudget * partStride)
	maxSlabs := 0
	for _, s := range gemvSlabs {
		maxSlabs = max(maxSlabs, s)
	}
	g.aRes = alloc(MaxRows * maxSlabs * ffn)
	g.aLogits = alloc(MaxRows * g.vocabPad)
	g.aUnscale = alloc(hidden)
	if g.abuf, err = g.dev.NewHostCachedBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("lm: fp32 arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	unscale := make([]float32, hidden)
	for i := range unscale {
		unscale[i] = 1 / ffScale
	}
	g.abuf.WriteFloat32At(int(g.aUnscale), unscale)

	// fp16 activations and the KV cache.
	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hA = halloc(rows * g.ldaDim)
	g.hQ = halloc(rows * qWidth)
	g.hCtx = halloc(rows * g.ldaQ)
	g.hFFN = halloc(rows * g.ldaF)
	cache := g.o.Slots * g.o.MaxLen * kvWidth
	for i := range g.w {
		g.w[i].kc = halloc(cache)
		g.w[i].vc = halloc(cache)
	}
	if g.hElems*2 > maxBankBytes {
		return fmt.Errorf("lm: the fp16 arena is %d MB, past one binding's 4 GB; lower MaxLen or Slots", (g.hElems*2)>>20)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("lm: fp16 arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the GEMMs read pad rows and pad columns.
	g.hbuf.Zero()
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, fmt.Errorf("lm: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return nil, fmt.Errorf("lm: pipeline %s: %w", name, err)
	}
	return pipe, nil
}

func (g *GPU) build() error {
	pc := uint32(pushBytes)
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[0]}
	for _, s := range []struct {
		name  string
		spirv []byte
		wave  uint32
	}{
		{"normf16", shaders.DiTNormScaleF16, 0},
		{"swiglu", shaders.DiTSwiGLUF16, 0},
		{"add", shaders.DiTGateAdd, 0},
		{"prep", shaders.ACELMPrep, 0},
		{"attn", shaders.ACELMAttn, 64},
		{"combine", shaders.ACELMAttnCombine, 0},
	} {
		p, err := g.pipeline(s.name, s.spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc, RequiredSubgroupSize: s.wave})
		if err != nil {
			return err
		}
		g.pipes[s.name] = p
	}
	for slabs, spirv := range gemvSumSPIRV {
		for r := 1; r <= MaxRows; r++ {
			p, err := g.pipeline("gemv sum", spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc,
				SpecConstants: rowSpec(r), RequiredSubgroupSize: 64})
			if err != nil {
				return err
			}
			g.pipes[fmt.Sprintf("sum_k%d_r%d", slabs, r)] = p
		}
	}
	used := map[int]bool{1: true}
	for _, s := range gemvSlabs {
		used[s] = true
	}
	g.gemms = make([]map[string]*vk.ComputePipeline, len(g.banks))
	g.gemvs = make([]map[string]*vk.ComputePipeline, len(g.banks))
	for b, bank := range g.banks {
		g.gemms[b] = map[string]*vk.ComputePipeline{}
		g.gemvs[b] = map[string]*vk.ComputePipeline{}
		bb := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank}
		if b != g.head {
			for _, k := range gemmKernels {
				spec := vk.PipelineSpec{Buffers: bb, PushConstantSize: pc}
				if k.waves > 1 {
					spec.RequiredSubgroupSize = 64
				}
				p, err := g.pipeline("gemm "+k.name, k.spirv, spec)
				if err != nil {
					return err
				}
				g.gemms[b][k.name] = p
			}
		}
		for slabs := range used {
			if b == g.head && slabs != 1 {
				continue
			}
			for r := 1; r <= MaxRows; r++ {
				p, err := g.pipeline("gemv", gemvSPIRV[slabs], vk.PipelineSpec{Buffers: bb, PushConstantSize: pc,
					SpecConstants: rowSpec(r), RequiredSubgroupSize: 64})
				if err != nil {
					return err
				}
				g.gemvs[b][fmt.Sprintf("k%d_r%d", slabs, r)] = p
			}
		}
	}
	return nil
}

// rowSpec is llm_gemv's ROWS specialisation.
func rowSpec(r int) []vk.SpecConstant { return []vk.SpecConstant{{ID: 0, Value: uint32(r)}} }

// tileB writes a [n, k] row-major matrix into the §2.8 fragment tiling:
// tile (nt, kt) is 256 contiguous halves holding (k, n) at (n%16)*16 + k%16,
// tiles kt-fastest. row(i) gives source row i's values as fp32.
func tileB(dst []uint16, n, k int, row func(i int, buf []float32)) {
	kt := k / tile
	workers := runtime.GOMAXPROCS(0)
	chunk := (n + workers - 1) / workers
	done := make(chan struct{}, workers)
	for w := 0; w < workers; w++ {
		go func(lo, hi int) {
			buf := make([]float32, k)
			for i := lo; i < hi; i++ {
				row(i, buf)
				base := (i / tile) * kt * tile * tile
				lane := (i % tile) * tile
				for j, v := range buf {
					dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
				}
			}
			done <- struct{}{}
		}(w*chunk, min(n, (w+1)*chunk))
	}
	for w := 0; w < workers; w++ {
		<-done
	}
}

func (g *GPU) stage(set *safetensors.Set) error {
	c := g.Cfg
	emb, err := set.Get(c.Prefix + "embed_tokens.weight")
	if err != nil {
		return err
	}
	g.embed = make([]uint16, g.vocab*hidden)
	for i := range g.embed {
		g.embed[i] = binary.LittleEndian.Uint16(emb.Data[2*i:])
	}
	// The tied head: the embedding, tiled; the pad rows stay zero.
	head := make([]uint16, g.vocabPad*hidden)
	tileB(head, g.vocab, hidden, func(i int, buf []float32) {
		for j := range buf {
			buf[j] = bf16(g.embed[i*hidden+j])
		}
	})
	g.banks[g.head].WriteUint16At(0, head)
	head = nil

	norm, err := set.Get(c.Prefix + "norm.weight")
	if err != nil {
		return err
	}
	fn, err := norm.F32(nil)
	if err != nil {
		return err
	}
	g.wbuf.WriteFloat32At(int(g.finalNorm), fn)

	rope := qwen.NewRoPE(hd, g.o.MaxLen, c.RopeTheta)
	g.wbuf.WriteFloat32At(int(g.rope), rope.Cos)
	g.wbuf.WriteFloat32At(int(g.rope)+g.o.MaxLen*hd/2, rope.Sin)

	for i := range g.w {
		runtime.GC()
		layer, err := qwen.LoadLayer(set, i, c)
		if err != nil {
			return err
		}
		w := &g.w[i]
		g.wbuf.WriteFloat32At(int(w.attnNorm), layer.AttnNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.ffnNorm), layer.FFNNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.qkNorm), layer.QNorm.Weight)
		g.wbuf.WriteFloat32At(int(w.qkNorm)+hd, layer.KNorm.Weight)
		lins := [nProj]*qwen.Linear{layer.Q, layer.K, layer.V, layer.O, layer.Gate, layer.Up, layer.Down}
		for p, lin := range lins {
			n, k := projShape[p][0], projShape[p][1]
			if lin.Out != n || lin.In != k {
				return fmt.Errorf("lm: layer %d %s is [%d %d]", i, projNames[p], lin.Out, lin.In)
			}
			buf := make([]uint16, n*k)
			tileB(buf, n, k, func(r int, dst []float32) { copy(dst, lin.Weight[r*k:(r+1)*k]) })
			g.banks[w.bank].WriteUint16At(int(w.off[p]), buf)
		}
	}
	return nil
}

func bf16(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// Destroy releases every Vulkan object.
func (g *GPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, set := range append(g.gemms, g.gemvs...) {
		for _, p := range set {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range append([]*vk.Buffer{g.wbuf, g.abuf, g.hbuf}, g.banks...) {
		if b != nil {
			b.Destroy()
		}
	}
}

// Tok is one row of a pass: a token id, the slot whose cache it extends
// and its position there.
type Tok struct {
	ID        int32
	Slot, Pos int
}

// Range is a span of vocabulary rows to compute logits for; the head reads
// only those. Phase 1 wants the text tokens, phase 2 the codes and EOS.
type Range struct{ Lo, Hi int }

// Pass runs every layer over toks. With a non-empty head range it also
// returns each row's logits over it (at most MaxRows rows).
func (g *GPU) Pass(toks []Tok, head Range) ([][]float32, time.Duration, error) {
	n := len(toks)
	if n == 0 || n > g.o.Rows {
		return nil, 0, fmt.Errorf("lm: a pass of %d rows; 1 to %d", n, g.o.Rows)
	}
	logits := head.Hi > head.Lo
	if logits && n > MaxRows {
		return nil, 0, fmt.Errorf("lm: logits for %d rows; at most %d", n, MaxRows)
	}
	lo, hi := head.Lo/tile*tile, roundUp(head.Hi, tile)
	if logits && (lo < 0 || hi > g.vocabPad) {
		return nil, 0, fmt.Errorf("lm: head range %v outside the vocabulary", head)
	}
	x := make([]float32, n*hidden)
	meta := make([]uint32, 2*n)
	maxKeys := 0
	for r, t := range toks {
		if t.ID < 0 || int(t.ID) >= g.vocab {
			return nil, 0, fmt.Errorf("lm: token %d outside the vocabulary", t.ID)
		}
		if t.Slot < 0 || t.Slot >= g.o.Slots || t.Pos < 0 || t.Pos >= g.o.MaxLen {
			return nil, 0, fmt.Errorf("lm: row %d at slot %d position %d; the cache is %d x %d", r, t.Slot, t.Pos, g.o.Slots, g.o.MaxLen)
		}
		e := g.embed[int(t.ID)*hidden : (int(t.ID)+1)*hidden]
		for j, v := range e {
			x[r*hidden+j] = bf16(v)
		}
		meta[2*r], meta[2*r+1] = uint32(t.Slot), uint32(t.Pos)
		maxKeys = max(maxKeys, t.Pos+1)
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.WriteUint32At(int(g.aMeta), meta)

	// Chunks of keys: 64 for a decode step, longer when many rows would
	// overrun the partials.
	chunks := (maxKeys + keyBlock - 1) / keyBlock
	if lim := partBudget / (n * nq); chunks > lim {
		chunks = max(lim, 1)
	}
	chunkLen := roundUp((maxKeys+chunks-1)/chunks, keyBlock)
	chunks = (maxKeys + chunkLen - 1) / chunkLen

	var total time.Duration
	per := max(g.LayersPerSubmit, 1)
	for i := 0; i < len(g.w); i += per {
		var d []vk.MultiDispatch
		for j := i; j < min(i+per, len(g.w)); j++ {
			d = append(d, g.layerGraph(j, n, chunks, chunkLen)...)
		}
		if logits && i+per >= len(g.w) {
			d = append(d, g.headGraph(n, lo, hi)...)
		}
		dur, err := vk.DispatchMultiTimed(d, 1, 1, true)
		if err != nil {
			return nil, total, fmt.Errorf("lm: layers %d..: %w", i, err)
		}
		total += dur
	}
	if !logits {
		return nil, total, nil
	}
	out := make([][]float32, n)
	w := hi - lo
	for r := range out {
		row := g.abuf.ReadFloat32At(int(g.aLogits)+r*w, w)
		out[r] = row[head.Lo-lo : head.Hi-lo]
	}
	return out, total, nil
}

// gemv is one projection at n ≤ MaxRows rows: split-K partials and their
// sum, or one dispatch at a single slab.
func (g *GPU) gemv(bank int, slabs int, aOff uint32, lda int, cOff, bOff uint32, nOut, k, n int) []vk.MultiDispatch {
	kt := k / tile
	if kt%slabs != 0 || (kt/slabs)%4 != 0 {
		panic(fmt.Sprintf("lm: %d slabs do not split K %d", slabs, k))
	}
	p := gemvPush{resOff: g.aRes, xnOff: aOff, outOff: cOff, bOff: bOff, lda: uint32(lda), gemmN: uint32(nOut), gemmK: uint32(k)}
	d := []vk.MultiDispatch{{
		Pipeline: g.gemvs[bank][fmt.Sprintf("k%d_r%d", slabs, n)],
		GroupsX:  uint32(slabs), GroupsY: uint32(nOut / tile), PushConstants: p.bytes(),
	}}
	if slabs > 1 {
		d = append(d, vk.MultiDispatch{
			Pipeline: g.pipes[fmt.Sprintf("sum_k%d_r%d", slabs, n)],
			GroupsX:  groups(nOut, 64), GroupsY: uint32(n), PushConstants: p.bytes(),
		})
	}
	return d
}

func (g *GPU) layerGraph(i, n, chunks, chunkLen int) []vk.MultiDispatch {
	w := &g.w[i]
	eps := math.Float32bits(float32(g.Cfg.RMSEps))
	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	norm := func(wOff uint32) {
		add("normf16", uint32(n), 1, push{InOff: g.aX, OutOff: g.hA, WOff: wOff, Tokens: uint32(n), Dim: hidden,
			LDA: uint32(g.ldaDim), Eps: eps})
	}
	tokPad := 0
	var gk gemmKernel
	if n > MaxRows {
		gk = gemmFor(n)
		tokPad = roundUp(n, gk.bm)
	}
	proj := func(p proj, aOff uint32, lda int, cOff uint32) {
		nOut, k := projShape[p][0], projShape[p][1]
		if n <= MaxRows {
			d = append(d, g.gemv(w.bank, gemvSlabs[p], aOff, lda, cOff, w.off[p], nOut, k, n)...)
			return
		}
		pc := push{InOff: aOff, OutOff: cOff, BOff: w.off[p], GemmM: uint32(tokPad), GemmN: uint32(nOut),
			GemmK: uint32(k), LDA: uint32(lda)}
		d = append(d, vk.MultiDispatch{Pipeline: g.gemms[w.bank][gk.name], GroupsX: uint32(nOut / gk.bn),
			GroupsY: uint32(tokPad / gk.bm), PushConstants: pc.bytes()})
	}

	// Attention.
	norm(w.attnNorm)
	proj(pQ, g.hA, g.ldaDim, g.aQ)
	proj(pK, g.hA, g.ldaDim, g.aK)
	proj(pV, g.hA, g.ldaDim, g.aV)
	add("prep", uint32(n), nq+2*nkv, push{InOff: g.aQ, KOff: g.aK, VOff: g.aV, OutOff: g.hQ, WOff: w.qkNorm,
		Aux0: g.aMeta, Aux1: w.kc, Aux2: w.vc, Span: uint32(g.o.MaxLen), BOff: g.rope, KStride: uint32(g.o.MaxLen),
		Eps: eps, Scale: math.Float32bits(float32(1 / math.Sqrt(hd))), Tokens: uint32(n)})
	add("attn", uint32(n), uint32(nq*chunks), push{InOff: g.hQ, OutOff: g.aPart, KOff: w.kc, VOff: w.vc,
		Aux0: g.aMeta, Aux1: uint32(chunkLen), Aux2: uint32(chunks), Span: uint32(g.o.MaxLen), Tokens: uint32(n)})
	add("combine", uint32(n), nq, push{InOff: g.aPart, OutOff: g.hCtx, Dim: uint32(g.ldaQ), Aux0: g.aMeta,
		Aux1: uint32(chunkLen), Aux2: uint32(chunks), Tokens: uint32(n)})
	proj(pO, g.hCtx, g.ldaQ, g.aO)
	add("add", uint32(n), 1, push{InOff: g.aO, OutOff: g.aX, Tokens: uint32(n), Dim: hidden})

	// Feed forward.
	norm(w.ffnNorm)
	proj(pGate, g.hA, g.ldaDim, g.aGate)
	proj(pUp, g.hA, g.ldaDim, g.aUp)
	add("swiglu", uint32(n), 1, push{InOff: g.aGate, KOff: g.aUp, OutOff: g.hFFN, Tokens: uint32(n), Dim: ffn,
		LDA: uint32(g.ldaF), Scale: math.Float32bits(ffScale)})
	proj(pDown, g.hFFN, g.ldaF, g.aFF)
	add("add", uint32(n), 1, push{InOff: g.aFF, OutOff: g.aX, Tokens: uint32(n), Dim: hidden, Aux0: g.aUnscale, Aux2: 1})
	return d
}

// headGraph is the final norm and the tied head over vocabulary rows
// [lo, hi), into the logits at a row stride of hi-lo.
func (g *GPU) headGraph(n, lo, hi int) []vk.MultiDispatch {
	eps := math.Float32bits(float32(g.Cfg.RMSEps))
	d := []vk.MultiDispatch{{Pipeline: g.pipes["normf16"], GroupsX: uint32(n), GroupsY: 1,
		PushConstants: push{InOff: g.aX, OutOff: g.hA, WOff: g.finalNorm, Tokens: uint32(n), Dim: hidden,
			LDA: uint32(g.ldaDim), Eps: eps}.bytes()}}
	return append(d, g.gemv(g.head, 1, g.hA, g.ldaDim, g.aLogits, uint32(lo*hidden), hi-lo, hidden, n)...)
}
