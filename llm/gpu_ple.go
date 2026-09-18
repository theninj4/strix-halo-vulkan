package llm

// The PLE n-gram block on the device: three dispatches for what llama.cpp
// spends about twenty-five on.
//
// Nothing here is a performance problem and none of it is tuned. The block
// runs **once**, at layer 1: its key projection is one dispatch of the 37 that
// share `MUL_MAT q8_0 m=10240 n=512 k=2560` in L2a's table, i.e. 0.11% of a
// prefill graph. What it has to be is *on the device*, because the alternative
// is reading the 10240-wide residual back to the host in the middle of the
// stack — 21 MB at 512 tokens, out of an arena that reads at 0.2 GB/s.
//
// What is not on the device is the gather itself, and that is D2 rather than
// an omission: `per_layer_token_embd` is 28.80 GB of IQ4_NL, a quarter of the
// whole checkpoint, and a token reads sixteen rows of 160 values out of it —
// 1.41 KB. It stays in the host mapping, the sixteen rows are hashed and
// gathered on the CPU (PLERows, Model.PLEGather), and what crosses to the
// device is the [T][2560] result. llama.cpp does exactly this, with
// `TENSOR_READ_LAZY` and a host-built row index.
//
// Arenas are this type's own. At L6, where the whole stack is resident, they
// merge with HCGPU's — which is why both share one binding contract and one
// push-constant block (shaders/llm_common.glsl) even though neither uses all
// of it.

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// pleKVBN is the compiled column block of the fused key/value projection,
// which is the plain arm of shaders/llm_gemm.comp. The row block is a ladder,
// because this dispatch is bound by the *weight*: B is 65.5 MB of fp16,
// twice the 32 MiB MALL, so a workgroup carrying BM token rows reads all of
// it M/BM times — 1.05 GB at 512 tokens with BM=32, and 262 MB with BM=128.
const pleKVBN = 64

// PLEKernel names one build of the fused projection, by its row block.
type PLEKernel string

const (
	PLEKVM2 PLEKernel = "kv_m2"
	PLEKVM4 PLEKernel = "kv_m4"
	PLEKVM8 PLEKernel = "kv_m8"
)

type pleVariant struct {
	name  PLEKernel
	spirv []byte
	bm    int
}

var pleVariants = []pleVariant{
	{PLEKVM2, shaders.LLMGEMMPlainM2, 32},
	{PLEKVM4, shaders.LLMGEMMPlainM4, 64},
	{PLEKVM8, shaders.LLMGEMMPlainM8, 128},
}

// PLEKernels lists the rungs, narrowest first.
func PLEKernels() []PLEKernel { return []PLEKernel{PLEKVM2, PLEKVM4, PLEKVM8} }

// PLEKernelFor is the measured schedule (research/l2d-ple.md). The rung is
// chosen against the *weight*: at 512 tokens BM=32/64/128 cost 2643/1611/1382
// us and at 2048 they cost 16995/7848/5177 — a 3.3x spread, which is the
// weight read M/BM times and nothing else. Below ~128 tokens there are not
// enough rows to fill a BM of 128 and the ladder turns over.
func PLEKernelFor(tokens int) PLEKernel {
	if tokens <= 128 {
		return PLEKVM4
	}
	return PLEKVM8
}

// DefaultPLEKernel is what a block runs before Upload has said how long the
// prompt is.
func DefaultPLEKernel() PLEKernel { return PLEKVM8 }

func pleVariantFor(k PLEKernel) (pleVariant, bool) {
	for _, v := range pleVariants {
		if v.name == k {
			return v, true
		}
	}
	return pleVariant{}, false
}

// PLEOpts are the choices that change what the block allocates.
type PLEOpts struct {
	// ConvOut makes the convolution write silu(conv) as well as adding it
	// into the residual, so it can be checked against `ple_conv_out-N`. The
	// residual add does not need the tensor; the trace comparison does.
	ConvOut bool
}

// PLEGPU runs the n-gram block for one layer.
type PLEGPU struct {
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
	dev *vk.Device
	cfg PLEConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	tokens, arenaRows, rows int
	lda                     int
	kv                      PLEKernel
	// autoKernel re-chooses the rung per run, using PLEKernelFor. SetKernel
	// turns it off, because a caller that named a rung meant it.
	autoKernel bool

	// fp32 weights: the three norms and the depthwise kernel.
	wNormKey, wNormQuery, wNormConv, wConv uint32
	// fp32 activations.
	aRes, aKV, aGate, aGated, aNorm, aConvOut uint32
	// aHist is the convolution's ring: (Conv-1)*NGram rows of the normed
	// gated value, addressed by **position** modulo its length, and the only
	// tensor in this block that outlives a run (L7b).
	aHist    uint32
	actElems int
	// past is how many tokens of this sequence are already behind the run,
	// which is how far back the convolution may reach. Zero is a fresh
	// sequence, where anything before token 0 is zero.
	past int
	// fp16 activations: the gathered n-gram embedding, the GEMM's A operand.
	hEmb   uint32
	hElems int

	convOut bool
}

// kvN is the fused projection's output width: the wide key and the n_embd
// value in one matrix, because they read the same gathered embedding — the
// same argument that puts `inject` on the hyper-connection block's down
// projection.
func (g *PLEGPU) kvN() int { return roundUpInt(g.cfg.Wide()+g.cfg.NEmbd, pleKVBN) }

// NewPLEGPU stages one layer's n-gram block onto the device.
func NewPLEGPU(dev *vk.Device, cfg PLEConfig, maxTokens int, w PLEWeights, opts PLEOpts) (*PLEGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: maxTokens is %d", maxTokens)
	}
	if cfg.NEmbd%coopMatTile != 0 || cfg.EmbdWidth() != cfg.NEmbd {
		return nil, fmt.Errorf("llm: the fused key/value projection assumes %d heads of %d make n_embd %d",
			cfg.NHeads, cfg.HeadDim, cfg.NEmbd)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	g := &PLEGPU{
		dev: dev, cfg: cfg,
		pipes:      make(map[string]*vk.ComputePipeline),
		tokens:     maxTokens,
		rows:       maxTokens,
		lda:        cfg.NEmbd + gemmPad,
		convOut:    opts.ConvOut,
		kv:         DefaultPLEKernel(),
		autoKernel: true,
	}
	align := 1
	for _, v := range pleVariants {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)
	if err := g.alloc(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(w); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func (g *PLEGPU) alloc() error {
	c := g.cfg
	wide, rows := c.Wide(), g.arenaRows

	total32 := 3*wide + wide*c.Conv
	var err error
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("llm: PLE fp32 weight arena: %w", err)
	}
	g.wNormKey = 0
	g.wNormQuery = uint32(wide)
	g.wNormConv = uint32(2 * wide)
	g.wConv = uint32(3 * wide)

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aRes = alloc(rows * wide)
	g.aKV = alloc(rows * g.kvN())
	g.aGated = alloc(rows * wide)
	g.aNorm = alloc(rows * wide)
	g.aGate = alloc(rows * c.HC)
	// 9 rows of 10240 floats — 368 KB — against the 168 MB a whole context's
	// worth of `aNorm` would be, which is the other way to let a tap reach
	// behind the run.
	g.aHist = alloc(c.ConvHist() * wide)
	g.aConvOut = noW
	if g.convOut {
		g.aConvOut = alloc(rows * wide)
	}
	if g.abuf, err = newArena(g.dev, g.actElems*4); err != nil {
		return fmt.Errorf("llm: PLE fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	g.hEmb = 0
	g.hElems = rows * g.lda
	if g.hbuf, err = newArena(g.dev, g.hElems*2); err != nil {
		return fmt.Errorf("llm: PLE fp16 activation arena: %w", err)
	}
	// Zeroed once: the A operand's pad columns and a short run's pad rows.
	g.hbuf.Zero()
	g.abuf.Zero()

	if g.bank, err = g.dev.NewBuffer(g.kvN() * c.NEmbd * 2); err != nil {
		return fmt.Errorf("llm: PLE fp16 weight bank (%d MB): %w", (g.kvN()*c.NEmbd*2)>>20, err)
	}
	return nil
}

func (g *PLEGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf}
	pcSize := uint32(unsafe.Sizeof(push{}))
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
		return fmt.Errorf("llm: the GEMM rung needs a pinned 64-wide subgroup")
	}
	for name, spirv := range map[string][]byte{
		"gate": shaders.LLMPLEGate,
		"conv": shaders.LLMPLEConv,
		"hist": shaders.LLMSeqHist,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	for _, v := range pleVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (g *PLEGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader ple.%s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline ple.%s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage packs the block's weights: the three norms and the depthwise kernel
// as fp32, and the key and value projections as one fragment-tiled matrix.
func (g *PLEGPU) stage(w PLEWeights) error {
	c := g.cfg
	wide := c.Wide()
	for _, t := range []struct {
		name string
		off  uint32
		src  []float32
		want int
	}{
		{"norm_key", g.wNormKey, w.NormKey, wide},
		{"norm_query", g.wNormQuery, w.NormQuery, wide},
		{"norm_conv", g.wNormConv, w.NormConv, wide},
		{"conv1d", g.wConv, w.Conv1d, wide * c.Conv},
	} {
		if len(t.src) != t.want {
			return fmt.Errorf("llm: ple_%s is %d values, want %d", t.name, len(t.src), t.want)
		}
		g.wbuf.WriteFloat32At(int(t.off), t.src)
	}
	if len(w.Key) != wide*c.NEmbd {
		return fmt.Errorf("llm: ple_key is %d values, want %d", len(w.Key), wide*c.NEmbd)
	}
	if len(w.Value) != c.NEmbd*c.NEmbd {
		return fmt.Errorf("llm: ple_value is %d values, want %d", len(w.Value), c.NEmbd*c.NEmbd)
	}
	// Key rows first, then value's: the gate kernel reads the value at column
	// `wide` of the same row, which is what makes this one dispatch.
	buf := make([]uint16, g.kvN()*c.NEmbd)
	tileB(buf, w.Key, wide, c.NEmbd, func(i int) int { return i })
	tileB(buf, w.Value, c.NEmbd, c.NEmbd, func(i int) int { return wide + i })
	g.bank.WriteUint16At(0, buf)
	return nil
}

// SetKernel chooses the fused projection's rung. Every rung is built and they
// all read the same staged weight, so this moves a pipeline and nothing else.
func (g *PLEGPU) SetKernel(k PLEKernel) error {
	if _, ok := pleVariantFor(k); !ok {
		return fmt.Errorf("llm: no PLE kernel %q (have %v)", k, PLEKernels())
	}
	g.kv = k
	g.autoKernel = false
	return nil
}

// Kernel reports the rung in use.
func (g *PLEGPU) Kernel() PLEKernel { return g.kv }

// Upload writes a run's inputs: the wide residual this block adds into, and
// the gathered n-gram embedding, [T][NHeads*HeadDim], which the host produced
// from the mmap'd table.
func (g *PLEGPU) Upload(res, embd []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(res) != nTok*c.Wide() {
		return fmt.Errorf("llm: residual is %d values, want %d", len(res), nTok*c.Wide())
	}
	if len(embd) != nTok*c.EmbdWidth() {
		return fmt.Errorf("llm: embedding is %d values, want %d", len(embd), nTok*c.EmbdWidth())
	}
	g.rows = nTok
	if g.autoKernel {
		g.kv = PLEKernelFor(nTok)
	}
	g.abuf.WriteFloat32At(int(g.aRes), res)
	slab := make([]uint16, nTok*g.lda)
	narrowRows(slab, embd, nTok, c.EmbdWidth(), g.lda)
	g.hbuf.WriteUint16At(int(g.hEmb), slab)
	return nil
}

// graph is the block's three dispatches, with a label each.
func (g *PLEGPU) graph() ([]vk.MultiDispatch, []string) {
	c := g.cfg
	wide := c.Wide()
	base := push{
		ResOff: g.aRes, XnOff: g.hEmb, GammaOff: g.wNormKey, GateOff: g.aGate,
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd), HC: uint32(c.HC),
		LDA: uint32(g.lda), Eps: math.Float32bits(c.Eps),
		KVOff: g.aKV, GatedOff: g.aGated, NormOff: g.aNorm,
		ConvOff: g.wConv, ConvOutOff: g.aConvOut,
		GammaQOff: g.wNormQuery, GammaCOff: g.wNormConv,
		Kern: uint32(c.Conv), Dil: uint32(c.NGram),
		GemmN: uint32(g.kvN()),
		// SEQ_HIST and SEQ_PAST: two fields this block does not otherwise
		// use, because the push block is full at 64 uints (llm_common.glsl).
		InjOff:  g.aHist,
		LowRank: uint32(g.past),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	v, _ := pleVariantFor(g.kv)
	kv := base
	kv.OutOff = g.aKV
	kv.GemmM, kv.GemmK = uint32(roundUpInt(g.rows, v.bm)), uint32(c.NEmbd)
	add(string(g.kv), "kv", uint32(g.kvN()/pleKVBN), uint32(roundUpInt(g.rows, v.bm)/v.bm), kv)

	add("gate", "gate", uint32(c.HC), uint32(g.rows), base)
	add("conv", "conv", uint32((wide+255)/256), uint32(g.rows), base)
	// The ring, for whatever runs next. It is the only dispatch here that a
	// single-shot prefill does not need, and it is nine rows.
	hist := base
	hist.LoOff = g.aNorm // SEQ_SRC
	hist.GemmM, hist.GemmK = uint32(c.ConvHist()), uint32(wide)
	add("hist", "hist", uint32((wide+255)/256), uint32(minInt(g.rows, c.ConvHist())), hist)
	return d, kinds
}

// Past is how many tokens of this sequence precede the next run, SetPast moves
// it, and Reset starts a fresh sequence.
//
// Nothing is cleared on a reset: the convolution masks every tap that reaches
// before position zero, so the ring's contents are unreachable rather than
// merely stale.
func (g *PLEGPU) Past() int { return g.past }
func (g *PLEGPU) Reset()    { g.past = 0 }

// SetPast places the next run's first token at position n.
func (g *PLEGPU) SetPast(n int) error {
	if n < 0 {
		return fmt.Errorf("llm: position %d", n)
	}
	g.past = n
	return nil
}

// Run executes the block over whatever Upload left in the arenas. The
// residual is updated in place.
func (g *PLEGPU) Run() error {
	d, kinds := g.graph()
	// One command buffer for the whole block, not one a dispatch: a submit
	// and a fence wait is ~150 us here and a decode step is a batch of one,
	// so what the sequence costs is how many times it is handed over
	// (LLM.md L7c). And when the graph is recording a whole pass, not even
	// one a block (L7d).
	if g.rec.add(ownPLE, kinds, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: ple, %d dispatches (%v): %w", len(d), kinds, err)
	}
	return nil
}

// Profile times each dispatch on the GPU over iters back-to-back runs. The
// residual is read and written by the last one, so a repeated dispatch is not
// idempotent — this measures cost, not a result.
func (g *PLEGPU) Profile(iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	d, kinds := g.graph()
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: ple dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the CPU reference's layout.

// Res is the wide residual after the block added into it, [T][hc*nEmbd].
func (g *PLEGPU) Res() []float32 { return g.abuf.ReadFloat32At(int(g.aRes), g.rows*g.cfg.Wide()) }

// Gate is the per-stream scalar gate, [T][hc].
func (g *PLEGPU) Gate() []float32 { return g.abuf.ReadFloat32At(int(g.aGate), g.rows*g.cfg.HC) }

// Gated is the value vector scaled by it, [T][hc*nEmbd].
func (g *PLEGPU) Gated() []float32 { return g.abuf.ReadFloat32At(int(g.aGated), g.rows*g.cfg.Wide()) }

// ConvOut is silu(conv), and is empty unless the block was built to write it.
func (g *PLEGPU) ConvOut() []float32 {
	if !g.convOut {
		return nil
	}
	return g.abuf.ReadFloat32At(int(g.aConvOut), g.rows*g.cfg.Wide())
}

// Key and Value are the fused projection's two halves, for a failure that
// needs locating upstream of the gate.
func (g *PLEGPU) Key() []float32   { return g.kvSlice(0, g.cfg.Wide()) }
func (g *PLEGPU) Value() []float32 { return g.kvSlice(g.cfg.Wide(), g.cfg.NEmbd) }

func (g *PLEGPU) kvSlice(col, width int) []float32 {
	raw := g.abuf.ReadFloat32At(int(g.aKV), g.rows*g.kvN())
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*g.kvN()+col:])
	}
	return out
}

// WeightBytes is what the block costs on the device and ActivationBytes what
// its arenas cost.
func (g *PLEGPU) WeightBytes() int { return g.wbuf.Size() + g.bank.Size() }

// Buffers is how many device allocations the block holds. L6a counts them
// across the whole model: `maxStorageBufferRange` is 4 GiB - 4 here, so the
// number is a residency fact and not bookkeeping.
func (g *PLEGPU) Buffers() int         { return 4 }
func (g *PLEGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Destroy releases every Vulkan object.
func (g *PLEGPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.hbuf, g.abuf, g.wbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// ResPort is the wide residual this block adds into: fp32 [T][hc*nEmbd], read
// and written in place. It is the only tensor of the PLE block that crosses a
// boundary — the gathered n-gram embedding comes from the host, because the
// 28.80 GB table it is gathered from never reaches the device (D2).
func (g *PLEGPU) ResPort() Port {
	return Port{Buf: g.abuf, Off: g.aRes, Stride: g.cfg.Wide(), Width: g.cfg.Wide(), Rows: g.rows}
}

// UploadEmbd writes the gathered n-gram embedding alone, for a caller that is
// filling the residual with a Move rather than an Upload.
func (g *PLEGPU) UploadEmbd(embd []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(embd) != nTok*c.EmbdWidth() {
		return fmt.Errorf("llm: embedding is %d values, want %d", len(embd), nTok*c.EmbdWidth())
	}
	g.rows = nTok
	if g.autoKernel {
		g.kv = PLEKernelFor(nTok)
	}
	slab := make([]uint16, nTok*g.lda)
	narrowRows(slab, embd, nTok, c.EmbdWidth(), g.lda)
	g.hbuf.WriteUint16At(int(g.hEmb), slab)
	return nil
}
