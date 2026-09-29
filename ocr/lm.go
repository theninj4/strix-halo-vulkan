package ocr

// ERNIE-4.5-0.3B on the device (OCR.md O4): the language half of
// PaddleOCR-VL, with a KV cache, for prefill and decode.
//
// It is ace/lm's decoder (MUSIC.md A7) at a quarter of the width, and the
// design carries over whole: the projections are staged once in fp16 in the
// §2.8 fragment tiling, which the DiT GEMM rungs read for a pass of many rows
// and llm_gemv's split-K GEMV reads at 1-3 rows; the norms, SwiGLU and
// residual adds are the DiT's kernels; ace_lm_prep and ace_lm_attn are
// rebuilt at this model's heads (ocr_lm_*.spv). A pass is any set of rows,
// each a token of some slot at some cache position, so a prompt can be
// prefilled in several passes and a page's regions decode as slots of one
// step (OCR.md decision 10).
//
// What is this model's own, each a place ace/lm's code would give plausible
// numbers:
//
//  1. **No q/k norm** (the prep's QK_NORM=0 build).
//  2. **Rope positions are three numbers a row, apart from the cache
//     position.** An image at p gives token (r, c) the position
//     (p, p+r, p+c) and the text after it resumes at p + max(mh, mw), so a
//     row carries its cache index (for the causal mask and the cache cell)
//     and its (t, h, w) (for the rope), and the prep reads pair i's position
//     from t, h or w in chunked [16, 24, 24] sections (MROPE=1).
//  3. **An untied head.** lm_head differs from embed_tokens by up to 0.20
//     (OCR.md decision 5); both are loaded.
//  4. **A row may be an embedding rather than a token**: the image rows are
//     the projector's output, spliced in on the host.
//
// The KV cache is paged (OCR.md O11): every layer's keys and values are a
// pool of 64-position pages, and each slot a table of them, so 32 slots cost
// what the positions they hold cost rather than 32 full-length planes. A
// page is the attention's key block, so a block of keys is one page.
//
// No narrowing factor on the SwiGLU product: OCR.md O0's audit put ERNIE's
// largest activation at 2,725, 24x under fp16's ceiling, where ace/lm's
// Qwen3 reached 6.4e4 and needed 1/16.
//
// The residual is fp32; every GEMM/GEMV operand is fp16 with fp32 sums.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// ERNIE's shape (config.json), which the shaders compile in.
const (
	lmHidden  = TextHidden
	lmNQ      = 16
	lmNKV     = 2
	lmHD      = 128
	lmFFN     = 3072
	lmLayers  = 18
	lmQWidth  = lmNQ * lmHD
	lmKVWidth = lmNKV * lmHD
	lmRMSEps  = 1e-5
	lmTheta   = 500000.0
	// EOS is </s>.
	EOS = 2

	tile         = 16
	gemmPad      = 128 // A-operand row pad in halves (zimage/qwen's §2.3)
	rowAlign     = 128 // the arena's rows: the widest GEMM tile's BM
	maxBankBytes = 0xfffffffc
	pushBytes    = 256 // the LLM's 64-uint block; the dit_common kernels read its first 88 bytes

	// gemvMaxRows is the OCR builds' MAXROWS (ocr_gemv_k*.spv): the most
	// rows a GEMV pass takes.
	gemvMaxRows = 16
	// PageSize is the positions a KV cache page holds: ace_lm_attn's key
	// block, which the PAGED build reads one page at a time.
	PageSize = 64
	// maxSlots bounds the logit rows of a pass to the smallest GEMM rung's
	// reach (reg32x128, up to 96 rows).
	maxSlots = 96
	// keyBlock is how many keys ace_lm_attn scores at once (one per lane).
	keyBlock = 64
	// partBudget caps (row, head, chunk) partials a pass writes.
	partBudget = 1 << 16
	partStride = 2 + lmHD
)

// lmPush mirrors dit_common.glsl's block.
type lmPush struct {
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

func (p lmPush) bytes() []byte {
	out := make([]byte, pushBytes)
	*(*lmPush)(unsafe.Pointer(&out[0])) = p
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

var projNames = [nProj]string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
	"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}

// projShape is [out, in].
var projShape = [nProj][2]int{
	pQ: {lmQWidth, lmHidden}, pK: {lmKVWidth, lmHidden}, pV: {lmKVWidth, lmHidden}, pO: {lmHidden, lmQWidth},
	pGate: {lmFFN, lmHidden}, pUp: {lmFFN, lmHidden}, pDown: {lmHidden, lmFFN},
}

// gemvRows is where a pass stops running its projections and its head as
// GEMVs and runs them as GEMMs (OCR.md O11); at most gemvMaxRows.
var gemvRows = gemvMaxRows

// gemvSlabs is each projection's split-K at 1-3 rows: K/16/slabs must be a
// multiple of 4. Measured (TestLMStepLadder, OCR.md O4), a step at depth 200
// on the device: ace/lm's ladder scaled to these widths (8/16/16/16/4/4/8)
// 5.64 ms, all 1 4.56, all 16 4.31, this 4.06. The ladder is not monotone:
// gate and up at 8 slabs cost 0.7 ms over 16, and every table in between
// the two ends is slower than both.
var gemvSlabs = [nProj]int{pQ: 16, pK: 16, pV: 16, pO: 1, pGate: 16, pUp: 16, pDown: 16}

// gemvSPIRV are the slab counts with 16-row builds; O4's ladder also tried
// the LLM's 2, 4 and 8, three rows at most.
var gemvSPIRV = map[int][]byte{1: shaders.OCRGEMVK1, 16: shaders.OCRGEMVK16}

var gemvSumSPIRV = map[int][]byte{16: shaders.LLMGEMVSumK16}

// gemmKernel is a DiT GEMM rung over the fragment-tiled weight.
type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	waves  int
	wave   uint32 // a pinned subgroup size; 0 is 64 for a multi-wave build
}

// gemmKernels are ace/lm's rungs, all of which divide ERNIE's N (256 and up).
var gemmKernels = []gemmKernel{
	{"reg32x128", shaders.DiTGEMMReg32x128Tiled, 32, 128, 1, 0},
	{"reg64", shaders.DiTGEMMReg64Tiled, 64, 64, 1, 0},
	// The LDS-staged wave32 build (KERNELS.md G2, research §2.9).
	{"wg128x256", shaders.DiTGEMMWG128x256LDSW32, 128, 256, 8, 32},
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

type lmLayer struct {
	off               [nProj]uint32 // in the layer bank, halves
	attnNorm, ffnNorm uint32        // fp32 arena
	kc, vc            uint32        // fp16 arena: this layer's caches, [page][64][2][128]
}

// LMOptions size the device state.
type LMOptions struct {
	// MaxLen is the positions a slot holds: prompt and generated text.
	MaxLen int
	// Slots is how many sequences hold a cache at once.
	Slots int
	// Rows is the longest pass; a longer prefill runs as several.
	Rows int
	// CachePositions is the page pool the slots share (0: Slots x MaxLen,
	// every slot full length at once). At least MaxLen.
	CachePositions int
}

// DefaultLMOptions fits one full-bound crop (1,280 image tokens and the
// template) and PaddleX's 8,192 new tokens, in one slot.
func DefaultLMOptions() LMOptions { return LMOptions{MaxLen: 10240, Slots: 1, Rows: 2048} }

// ErrCacheFull is Reserve's (and Pass's) when the page pool is spent.
var ErrCacheFull = errors.New("ocr: the KV cache has no free page")

// LM is ERNIE, resident.
type LM struct {
	o LMOptions

	dev   *vk.Device
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	bank  *vk.Buffer // the 18 layers' projections
	headB *vk.Buffer // lm_head, [vocab][1024] tiled
	pipes map[string]*vk.ComputePipeline
	// per bank (0 layers, 1 head): GEMM rungs, and GEMV builds by "k<slabs>_r<rows>"
	gemms [2]map[string]*vk.ComputePipeline
	gemvs [2]map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	embed []uint16 // bf16 [vocab][1024], gathered on the host
	vocab int

	w         []lmLayer
	finalNorm uint32
	rope      uint32 // cos [MaxLen][64], then sin

	// The page pool: free pages, and each slot's table of the pages that
	// hold its positions 0, 64, 128, ...
	slotPages int // a table's length, MaxLen in pages
	free      []int32
	tables    [][]int32
	ptHost    []uint32
	logitsBuf [][]float32
	maxLogits int

	arenaRows int
	// fp32 arena
	aX, aQ, aK, aV, aO, aGate, aUp, aFF, aMeta, aRope, aPart, aRes, aLogits, aPT uint32
	actElems                                                                     int
	// fp16 arena
	hA, hQ, hCtx, hFFN uint32
	hElems             int
	ldaDim, ldaQ, ldaF int

	// LayersPerSubmit is how many layers of a prefill go in one command
	// buffer; a decode step is always one.
	LayersPerSubmit int
	// Tap, if set, sees the residual after each layer ("lm.layer{i}", the
	// pass's rows), one submit a layer. For gates.
	Tap func(name string, x *qwen.Mat)
}

// DeviceBytes is what the language model holds on the device: weights,
// activations and the KV cache.
func (g *LM) DeviceBytes() int {
	return g.wbuf.Size() + g.abuf.Size() + g.hbuf.Size() + g.bank.Size() + g.headB.Size()
}

// Vocab is the head's width.
func (g *LM) Vocab() int { return g.vocab }

// LoadLM stages ERNIE from the checkpoint directory.
func LoadLM(dev *vk.Device, dir string, o LMOptions) (*LM, error) {
	if o.CachePositions == 0 {
		o.CachePositions = o.Slots * roundUp(o.MaxLen, PageSize)
	}
	if o.MaxLen <= 0 || o.Slots <= 0 || o.Slots > maxSlots || o.Rows <= 0 || o.CachePositions < o.MaxLen {
		return nil, fmt.Errorf("ocr: LM options %+v", o)
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g := &LM{
		o: o, dev: dev,
		pipes:           map[string]*vk.ComputePipeline{},
		ldaDim:          lmHidden + gemmPad,
		ldaQ:            lmQWidth + gemmPad,
		ldaF:            lmFFN + gemmPad,
		arenaRows:       roundUp(o.Rows, rowAlign),
		LayersPerSubmit: 6,
	}
	if err := g.layout(f); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(f); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

func (g *LM) layout(f *safetensors.File) error {
	emb, err := f.Get("model.embed_tokens.weight")
	if err != nil {
		return err
	}
	if len(emb.Shape) != 2 || emb.Shape[1] != lmHidden || emb.DType != safetensors.BF16 {
		return fmt.Errorf("ocr: embed_tokens is %v %s, want [vocab 1024] bf16", emb.Shape, emb.DType)
	}
	g.vocab = emb.Shape[0]
	if g.vocab%gemmKernels[0].bn != 0 {
		return fmt.Errorf("ocr: a vocabulary of %d is not a multiple of %d", g.vocab, gemmKernels[0].bn)
	}

	perLayer := 0
	for _, s := range projShape {
		perLayer += s[0] * s[1] * 2
	}
	g.w = make([]lmLayer, lmLayers)
	bytes := 0
	for i := range g.w {
		for p, s := range projShape {
			g.w[i].off[p] = uint32(bytes / 2)
			bytes += s[0] * s[1] * 2
		}
	}
	if g.bank, err = g.dev.NewBuffer(bytes); err != nil {
		return fmt.Errorf("ocr: layer bank (%d MB): %w", bytes>>20, err)
	}
	if g.headB, err = g.dev.NewBuffer(g.vocab * lmHidden * 2); err != nil {
		return fmt.Errorf("ocr: head bank: %w", err)
	}

	// fp32 weights: the norms and the rope table.
	w32 := 0
	for i := range g.w {
		g.w[i].attnNorm = uint32(w32)
		g.w[i].ffnNorm = uint32(w32 + lmHidden)
		w32 += 2 * lmHidden
	}
	g.finalNorm = uint32(w32)
	w32 += lmHidden
	g.rope = uint32(w32)
	w32 += 2 * g.o.MaxLen * lmHD / 2
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
	g.aX = alloc(rows * lmHidden)
	g.aQ = alloc(rows * lmQWidth)
	g.aK = alloc(rows * lmKVWidth)
	g.aV = alloc(rows * lmKVWidth)
	g.aO = alloc(rows * lmHidden)
	g.aGate = alloc(rows * lmFFN)
	g.aUp = alloc(rows * lmFFN)
	g.aFF = alloc(rows * lmHidden)
	g.aMeta = alloc(rows * 2)
	g.aRope = alloc(rows * 3)
	g.aPart = alloc(partBudget * partStride)
	g.aRes = alloc(gemvMaxRows * 16 * lmFFN) // 16: the widest split-K rung
	// Logits for a row a slot; the head's GEMM writes its padding rows too.
	g.maxLogits = max(gemvMaxRows, g.o.Slots)
	g.aLogits = alloc(roundUp(g.maxLogits, gemmKernels[0].bm) * g.vocab)
	g.slotPages = (g.o.MaxLen + PageSize - 1) / PageSize
	g.aPT = alloc(g.o.Slots * g.slotPages)
	if g.abuf, err = g.dev.NewHostCachedBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("ocr: fp32 arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	// fp16 activations and the KV cache.
	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hA = halloc(rows * g.ldaDim)
	g.hQ = halloc(rows * lmQWidth)
	g.hCtx = halloc(rows * g.ldaQ)
	g.hFFN = halloc(rows * g.ldaF)
	pages := (g.o.CachePositions + PageSize - 1) / PageSize
	for p := pages - 1; p >= 0; p-- {
		g.free = append(g.free, int32(p))
	}
	g.tables = make([][]int32, g.o.Slots)
	g.ptHost = make([]uint32, g.o.Slots*g.slotPages)
	cache := pages * PageSize * lmKVWidth
	for i := range g.w {
		g.w[i].kc = halloc(cache)
		g.w[i].vc = halloc(cache)
	}
	if g.hElems*2 > maxBankBytes {
		return fmt.Errorf("ocr: the fp16 arena is %d MB, past one binding's 4 GB; lower MaxLen or Slots", (g.hElems*2)>>20)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("ocr: fp16 arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the GEMMs read pad rows and pad columns.
	g.hbuf.Zero()
	return nil
}

func (g *LM) pipeline(name string, spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, fmt.Errorf("ocr: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return nil, fmt.Errorf("ocr: pipeline %s: %w", name, err)
	}
	return pipe, nil
}

func (g *LM) build() error {
	pc := uint32(pushBytes)
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	for _, s := range []struct {
		name  string
		spirv []byte
		wave  uint32
	}{
		{"normf16", shaders.DiTNormScaleF16, 0},
		{"swiglu", shaders.DiTSwiGLUF16, 0},
		{"add", shaders.DiTGateAdd, 0},
		{"prep", shaders.OCRLMPrep, 0},
		{"attn", shaders.OCRLMAttn, 64},
		{"combine", shaders.OCRLMAttnCombine, 0},
	} {
		p, err := g.pipeline(s.name, s.spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc, RequiredSubgroupSize: s.wave})
		if err != nil {
			return err
		}
		g.pipes[s.name] = p
	}
	for slabs, spirv := range gemvSumSPIRV {
		for r := 1; r <= gemvMaxRows; r++ {
			p, err := g.pipeline("gemv sum", spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc,
				SpecConstants: rowSpec(r), RequiredSubgroupSize: 64})
			if err != nil {
				return err
			}
			g.pipes[fmt.Sprintf("sum_k%d_r%d", slabs, r)] = p
		}
	}
	for b, bank := range []*vk.Buffer{g.bank, g.headB} {
		g.gemms[b] = map[string]*vk.ComputePipeline{}
		g.gemvs[b] = map[string]*vk.ComputePipeline{}
		bb := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank}
		used := map[int]bool{1: true}
		if b == 1 {
			// The head over more rows than a GEMV takes: the smallest rung.
			k := gemmKernels[0]
			p, err := g.pipeline("gemm head "+k.name, k.spirv, vk.PipelineSpec{Buffers: bb, PushConstantSize: pc})
			if err != nil {
				return err
			}
			g.gemms[b][k.name] = p
		}
		if b == 0 {
			for _, k := range gemmKernels {
				spec := vk.PipelineSpec{Buffers: bb, PushConstantSize: pc}
				if k.wave != 0 {
					spec.RequiredSubgroupSize = k.wave
				} else if k.waves > 1 {
					spec.RequiredSubgroupSize = 64
				}
				p, err := g.pipeline("gemm "+k.name, k.spirv, spec)
				if err != nil {
					return err
				}
				g.gemms[b][k.name] = p
			}
			for _, s := range gemvSlabs {
				used[s] = true
			}
		}
		for slabs := range used {
			for r := 1; r <= gemvMaxRows; r++ {
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

func (g *LM) stage(f *safetensors.File) error {
	get := func(name string, want int) ([]float32, error) {
		t, err := f.Get(name)
		if err != nil {
			return nil, err
		}
		v, err := t.F32(nil)
		if err != nil {
			return nil, err
		}
		if len(v) != want {
			return nil, fmt.Errorf("ocr: %s has %d values, want %d", name, len(v), want)
		}
		return v, nil
	}
	emb, err := f.Get("model.embed_tokens.weight")
	if err != nil {
		return err
	}
	g.embed = make([]uint16, g.vocab*lmHidden)
	for i := range g.embed {
		g.embed[i] = binary.LittleEndian.Uint16(emb.Data[2*i:])
	}
	// The untied head (OCR.md decision 5).
	head, err := get("lm_head.weight", g.vocab*lmHidden)
	if err != nil {
		return err
	}
	buf := make([]uint16, g.vocab*lmHidden)
	tileB(buf, g.vocab, lmHidden, func(i int, dst []float32) { copy(dst, head[i*lmHidden:(i+1)*lmHidden]) })
	g.headB.WriteUint16At(0, buf)

	fn, err := get("model.norm.weight", lmHidden)
	if err != nil {
		return err
	}
	g.wbuf.WriteFloat32At(int(g.finalNorm), fn)
	rope := qwen.NewRoPE(lmHD, g.o.MaxLen, lmTheta)
	g.wbuf.WriteFloat32At(int(g.rope), rope.Cos)
	g.wbuf.WriteFloat32At(int(g.rope)+g.o.MaxLen*lmHD/2, rope.Sin)

	for i := range g.w {
		p := fmt.Sprintf("model.layers.%d.", i)
		w := &g.w[i]
		for _, nm := range []struct {
			name string
			off  uint32
		}{{"input_layernorm", w.attnNorm}, {"post_attention_layernorm", w.ffnNorm}} {
			v, err := get(p+nm.name+".weight", lmHidden)
			if err != nil {
				return err
			}
			g.wbuf.WriteFloat32At(int(nm.off), v)
		}
		for pr := range projShape {
			n, k := projShape[pr][0], projShape[pr][1]
			v, err := get(p+projNames[pr]+".weight", n*k)
			if err != nil {
				return err
			}
			buf := make([]uint16, n*k)
			tileB(buf, n, k, func(r int, dst []float32) { copy(dst, v[r*k:(r+1)*k]) })
			g.bank.WriteUint16At(int(w.off[pr]), buf)
		}
	}
	return nil
}

func bf16(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// Destroy releases every Vulkan object.
func (g *LM) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, set := range append(g.gemms[:], g.gemvs[:]...) {
		for _, p := range set {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.headB} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Row is one row of a pass: a token (or an image row's embedding), the slot
// whose cache it extends, its cache position there, and its rope position
// (t, h, w).
type Row struct {
	ID    int32
	Embed []float32 // [1024]; when set, ID is ignored
	Slot  int
	Pos   int
	Rope  [3]int32
}

// Reserve gives slot pages for its positions [0, n), taking them from the
// pool; ErrCacheFull, with nothing taken, when the pool has too few.
func (g *LM) Reserve(slot, n int) error {
	if slot < 0 || slot >= g.o.Slots || n > g.o.MaxLen {
		return fmt.Errorf("ocr: %d positions in slot %d; the cache is %d x %d", n, slot, g.o.Slots, g.o.MaxLen)
	}
	need := (n+PageSize-1)/PageSize - len(g.tables[slot])
	if need <= 0 {
		return nil
	}
	if need > len(g.free) {
		return ErrCacheFull
	}
	for ; need > 0; need-- {
		p := g.free[len(g.free)-1]
		g.free = g.free[:len(g.free)-1]
		g.tables[slot] = append(g.tables[slot], p)
	}
	return nil
}

// Release returns slot's pages to the pool; its positions are forgotten.
func (g *LM) Release(slot int) {
	for i := len(g.tables[slot]) - 1; i >= 0; i-- {
		g.free = append(g.free, g.tables[slot][i])
	}
	g.tables[slot] = g.tables[slot][:0]
}

// FreePages is how many pages the pool has left.
func (g *LM) FreePages() int { return len(g.free) }

// Pages is the pool's size.
func (g *LM) Pages() int { return (g.o.CachePositions + PageSize - 1) / PageSize }

// Pass runs every layer over rows and returns the logits of the last
// `logits` rows (at most one a slot, or gemvMaxRows if more), over the whole
// vocabulary. The rows may come in any order: every row's keys are in the
// cache before any row attends. A row's slot gets pages for its position as
// Reserve would (ErrCacheFull if the pool is spent). The logits are the
// LM's own buffers, valid until the next pass.
func (g *LM) Pass(rows []Row, logits int) ([][]float32, time.Duration, error) {
	n := len(rows)
	if n == 0 || n > g.o.Rows {
		return nil, 0, fmt.Errorf("ocr: a pass of %d rows; 1 to %d", n, g.o.Rows)
	}
	if logits < 0 || logits > min(n, g.maxLogits) {
		return nil, 0, fmt.Errorf("ocr: logits for %d of %d rows; at most %d", logits, n, g.maxLogits)
	}
	x := make([]float32, n*lmHidden)
	meta := make([]uint32, 2*n)
	rp := make([]uint32, 3*n)
	maxKeys := 0
	for r, t := range rows {
		dst := x[r*lmHidden : (r+1)*lmHidden]
		if t.Embed != nil {
			if len(t.Embed) != lmHidden {
				return nil, 0, fmt.Errorf("ocr: row %d's embedding has %d values", r, len(t.Embed))
			}
			copy(dst, t.Embed)
		} else {
			if t.ID < 0 || int(t.ID) >= g.vocab {
				return nil, 0, fmt.Errorf("ocr: token %d outside the vocabulary", t.ID)
			}
			for j, v := range g.embed[int(t.ID)*lmHidden : (int(t.ID)+1)*lmHidden] {
				dst[j] = bf16(v)
			}
		}
		if t.Slot < 0 || t.Slot >= g.o.Slots || t.Pos < 0 || t.Pos >= g.o.MaxLen {
			return nil, 0, fmt.Errorf("ocr: row %d at slot %d position %d; the cache is %d x %d", r, t.Slot, t.Pos, g.o.Slots, g.o.MaxLen)
		}
		for a, p := range t.Rope {
			if p < 0 || int(p) >= g.o.MaxLen {
				return nil, 0, fmt.Errorf("ocr: row %d's rope position %v past the table's %d", r, t.Rope, g.o.MaxLen)
			}
			rp[3*r+a] = uint32(p)
		}
		if err := g.Reserve(t.Slot, t.Pos+1); err != nil {
			return nil, 0, err
		}
		meta[2*r], meta[2*r+1] = uint32(t.Slot), uint32(t.Pos)
		maxKeys = max(maxKeys, t.Pos+1)
	}
	for s, tb := range g.tables {
		for i, p := range tb {
			g.ptHost[s*g.slotPages+i] = uint32(p)
		}
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.WriteUint32At(int(g.aMeta), meta)
	g.abuf.WriteUint32At(int(g.aRope), rp)
	g.abuf.WriteUint32At(int(g.aPT), g.ptHost)

	// Chunks of keys: 64 for a decode step, longer when many rows would
	// overrun the partials.
	chunks := (maxKeys + keyBlock - 1) / keyBlock
	if lim := partBudget / (n * lmNQ); chunks > lim {
		chunks = max(lim, 1)
	}
	chunkLen := roundUp((maxKeys+chunks-1)/chunks, keyBlock)
	chunks = (maxKeys + chunkLen - 1) / chunkLen

	var total time.Duration
	per := max(g.LayersPerSubmit, 1)
	if n <= gemvRows {
		per = len(g.w)
	}
	if g.Tap != nil {
		per = 1
	}
	for i := 0; i < len(g.w); i += per {
		var d []vk.MultiDispatch
		for j := i; j < min(i+per, len(g.w)); j++ {
			d = append(d, g.layerGraph(j, n, chunks, chunkLen)...)
		}
		if logits > 0 && i+per >= len(g.w) {
			d = append(d, g.headGraph(n, logits)...)
		}
		dur, err := vk.DispatchMultiTimed(d, 1, 1, true)
		if err != nil {
			return nil, total, fmt.Errorf("ocr: layers %d..: %w", i, err)
		}
		total += dur
		if g.Tap != nil {
			m := qwen.NewMat(n, lmHidden)
			copy(m.Data, g.abuf.ReadFloat32At(int(g.aX), n*lmHidden))
			g.Tap(fmt.Sprintf("lm.layer%d", i), m)
		}
	}
	if logits == 0 {
		return nil, total, nil
	}
	for len(g.logitsBuf) < logits {
		g.logitsBuf = append(g.logitsBuf, make([]float32, g.vocab))
	}
	out := g.logitsBuf[:logits]
	for r := range out {
		g.abuf.ReadFloat32Into(int(g.aLogits)+r*g.vocab, out[r])
	}
	return out, total, nil
}

// gemv is one projection at n ≤ gemvRows rows: split-K partials and
// their sum, or one dispatch at a single slab.
func (g *LM) gemv(bank int, slabs int, aOff uint32, lda int, cOff, bOff uint32, nOut, k, n int) []vk.MultiDispatch {
	kt := k / tile
	if kt%slabs != 0 || (kt/slabs)%4 != 0 {
		panic(fmt.Sprintf("ocr: %d slabs do not split K %d", slabs, k))
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

func (g *LM) layerGraph(i, n, chunks, chunkLen int) []vk.MultiDispatch {
	w := &g.w[i]
	eps := math.Float32bits(float32(lmRMSEps))
	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p lmPush) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	norm := func(wOff uint32) {
		add("normf16", uint32(n), 1, lmPush{InOff: g.aX, OutOff: g.hA, WOff: wOff, Tokens: uint32(n), Dim: lmHidden,
			LDA: uint32(g.ldaDim), Eps: eps})
	}
	var gk gemmKernel
	tokPad := 0
	if n > gemvRows {
		gk = gemmFor(n)
		tokPad = roundUp(n, gk.bm)
	}
	proj := func(p proj, aOff uint32, lda int, cOff uint32) {
		nOut, k := projShape[p][0], projShape[p][1]
		if n <= gemvRows {
			d = append(d, g.gemv(0, gemvSlabs[p], aOff, lda, cOff, w.off[p], nOut, k, n)...)
			return
		}
		pc := lmPush{InOff: aOff, OutOff: cOff, BOff: w.off[p], GemmM: uint32(tokPad), GemmN: uint32(nOut),
			GemmK: uint32(k), LDA: uint32(lda)}
		d = append(d, vk.MultiDispatch{Pipeline: g.gemms[0][gk.name], GroupsX: uint32(nOut / gk.bn),
			GroupsY: uint32(tokPad / gk.bm), PushConstants: pc.bytes()})
	}

	// Attention.
	norm(w.attnNorm)
	proj(pQ, g.hA, g.ldaDim, g.aQ)
	proj(pK, g.hA, g.ldaDim, g.aK)
	proj(pV, g.hA, g.ldaDim, g.aV)
	add("prep", uint32(n), lmNQ+2*lmNKV, lmPush{InOff: g.aQ, KOff: g.aK, VOff: g.aV, OutOff: g.hQ,
		Aux0: g.aMeta, Aux1: w.kc, Aux2: w.vc, Heads: g.aPT, Span: uint32(g.slotPages), BOff: g.rope, KStride: uint32(g.o.MaxLen),
		Dim: g.aRope, Scale: math.Float32bits(float32(1 / math.Sqrt(lmHD))), Tokens: uint32(n)})
	add("attn", uint32(n), uint32(lmNQ*chunks), lmPush{InOff: g.hQ, OutOff: g.aPart, KOff: w.kc, VOff: w.vc,
		Aux0: g.aMeta, Aux1: uint32(chunkLen), Aux2: uint32(chunks), Heads: g.aPT, Span: uint32(g.slotPages), Tokens: uint32(n)})
	add("combine", uint32(n), lmNQ, lmPush{InOff: g.aPart, OutOff: g.hCtx, Dim: uint32(g.ldaQ), Aux0: g.aMeta,
		Aux1: uint32(chunkLen), Aux2: uint32(chunks), Tokens: uint32(n)})
	proj(pO, g.hCtx, g.ldaQ, g.aO)
	add("add", uint32(n), 1, lmPush{InOff: g.aO, OutOff: g.aX, Tokens: uint32(n), Dim: lmHidden})

	// Feed forward.
	norm(w.ffnNorm)
	proj(pGate, g.hA, g.ldaDim, g.aGate)
	proj(pUp, g.hA, g.ldaDim, g.aUp)
	add("swiglu", uint32(n), 1, lmPush{InOff: g.aGate, KOff: g.aUp, OutOff: g.hFFN, Tokens: uint32(n), Dim: lmFFN,
		LDA: uint32(g.ldaF), Scale: math.Float32bits(1)})
	proj(pDown, g.hFFN, g.ldaF, g.aFF)
	add("add", uint32(n), 1, lmPush{InOff: g.aFF, OutOff: g.aX, Tokens: uint32(n), Dim: lmHidden})
	return d
}

// headGraph is the final norm and the head over the pass's last `rows` rows,
// into the logits at a row stride of the vocabulary.
func (g *LM) headGraph(n, rows int) []vk.MultiDispatch {
	eps := math.Float32bits(float32(lmRMSEps))
	d := []vk.MultiDispatch{{Pipeline: g.pipes["normf16"], GroupsX: uint32(rows), GroupsY: 1,
		PushConstants: lmPush{InOff: g.aX + uint32((n-rows)*lmHidden), OutOff: g.hA, WOff: g.finalNorm,
			Tokens: uint32(rows), Dim: lmHidden, LDA: uint32(g.ldaDim), Eps: eps}.bytes()}}
	if rows <= gemvRows {
		return append(d, g.gemv(1, 1, g.hA, g.ldaDim, g.aLogits, 0, g.vocab, lmHidden, rows)...)
	}
	k := gemmKernels[0]
	tokPad := roundUp(rows, k.bm)
	pc := lmPush{InOff: g.hA, OutOff: g.aLogits, GemmM: uint32(tokPad), GemmN: uint32(g.vocab), GemmK: lmHidden,
		LDA: uint32(g.ldaDim)}
	return append(d, vk.MultiDispatch{Pipeline: g.gemms[1][k.name], GroupsX: uint32(g.vocab / k.bn),
		GroupsY: uint32(tokPad / k.bm), PushConstants: pc.bytes()})
}
