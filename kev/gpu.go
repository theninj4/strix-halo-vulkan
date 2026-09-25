package kev

// Kev-4B on the device (CLASSIFICATION.md K4/K5): Qwen3.5-4B-Base with the
// LoRA folded in, run over one packed pass of segments -- the state, then
// every question's branch -- and read back only at the readout rows.
//
// The arena and GEMM machinery is zimage/qwen's, unchanged: four buffers
// bound to every pipeline (fp32 weights, fp32 activations, fp16 activations,
// fp16 weight bank), tensors addressed by offset, weights packed once into the
// 16x16 fragment tiling, M padded up to the tile. Every projection's N is
// padded to 256 so any rung of dit_gemm's ladder divides it, and each layer's
// input projections are fused into one GEMM:
//
//	GDN layer   [qkv 8192 | z 4096 | a 32 | b 32 | pad]   = 12544 x 2560
//	attention   [q+gate 8192 | k 1024 | v 1024]           = 10240 x 2560
//
// **The norms are zero-centred**: HF's Qwen3_5RMSNorm multiplies by (1 + w),
// so the loader stores 1 + w and the plain RMS-norm kernels apply it. The
// GDN's gated output norm is the exception (plain w), and so is nothing else.

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// push mirrors dit_common.glsl's PC block; each kev kernel documents which
// fields it reads.
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

// pushBytes is the push range every pipeline here declares and every
// dispatch pushes: the LLM's 64-uint block, because the int8 GEMM is
// llm_gemm.comp's and one command buffer takes one range. The dit_common
// kernels read the first 88 bytes of it.
const pushBytes = 256

func (p push) bytes() []byte {
	out := make([]byte, pushBytes)
	*(*push)(unsafe.Pointer(&out[0])) = p
	return out
}

// llmPush is the fields of llm_common.glsl's block that llm_gemm.comp's plain
// arm (MODE 2) reads, at their indices in the 64-uint block.
type llmPush struct {
	xnOff, outOff, bOff uint32 // A (fp16 arena, halves), C (fp32 arena), B (bank, bytes)
	lda, ldaLo          uint32 // ldaLo: the GLU epilogue's output row stride
	gemmM, gemmN, gemmK uint32
}

func (p llmPush) bytes() []byte {
	var u [pushBytes / 4]uint32
	u[1], u[4], u[6] = p.xnOff, p.outOff, p.bOff
	u[12], u[13], u[14], u[15], u[16] = p.lda, p.ldaLo, p.gemmM, p.gemmN, p.gemmK
	u[7] = 0xffffffff // gateOff: NO_W
	out := make([]byte, pushBytes)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), pushBytes))
	return out
}

// Bank is the width the projections are staged in.
type Bank int

const (
	// BankQ8 is int8 in the fragment tiling with one fp16 scale per 32
	// elements of a row -- the LLM's L8 bank, read by llm_gemm.comp -DQ8B --
	// quantised here from the merged fp32 weights. 8.5 bits a weight.
	BankQ8 Bank = iota
	// BankFP16 is halves, what K4-K6 ran: the control.
	BankFP16
)

func (b Bank) String() string {
	if b == BankFP16 {
		return "fp16"
	}
	return "q8"
}

// q8Group is how many elements of a row share one scale.
const q8Group = 32

// The int8 GEMM's rungs: the LLM's llm_gemm.comp -DQ8B plain arm at BM = 32,
// 64, 128; BN is 64 in all three. K7.1 tried three kernels of its own for
// short passes (split-K, several waves sharing one unpacked slab, and a
// register-pipelined slab) and each lost to these at every length measured;
// CLASSIFICATION.md has the ladder.
var q8Kernels = []gemmKernel{
	{"q8m2", shaders.LLMGEMMQ8M2, 32, 64, 1, false},
	{"q8m4", shaders.LLMGEMMQ8M4, 64, 64, 1, false},
	{"q8m8", shaders.LLMGEMMQ8M8, 128, 64, 1, false},
}

// gluKernels are kev_gemm_q8_glu at the q8m2 and q8m4 rungs' tiles; the
// schedule picks the one matching q8For's BM.
var gluKernels = []gemmKernel{
	{"glum2", shaders.KevGEMMQ8GLUM2, 32, 64, 1, true},
	{"glum4", shaders.KevGEMMQ8GLUM4, 64, 64, 1, true},
	{"glum8", shaders.KevGEMMQ8GLUM8, 128, 64, 1, true},
	// The same kernel with a plain fp32 store, for every other projection.
	{"rbm2", shaders.KevGEMMQ8RBM2, 32, 64, 1, true},
	{"rbm4", shaders.KevGEMMQ8RBM4, 64, 64, 1, true},
	{"rbm8", shaders.KevGEMMQ8RBM8, 128, 64, 1, true},
}

// grid is a GEMM's workgroup counts for n columns over tokPad rows.
func (k gemmKernel) grid(n, tokPad int) (uint32, uint32) {
	if k.rowFast {
		return uint32(tokPad / k.bm), uint32(n / k.bn)
	}
	return uint32(n / k.bn), uint32(tokPad / k.bm)
}

// q8For picks the int8 rung for one projection by pass length and output
// width (K7.6). The row-block-fastest grid (the rb rungs) wins on the wide
// input projection up to a few hundred rows, and the LLM's llm_gemm order
// wins on the 2560-wide ones (out, down) everywhere and on the wide one at
// long passes. TestGPUProjLadder, kernel ms summed over the layers:
//
//	               101 tokens        494 tokens        2,446 tokens
//	in proj        rbm4 8.7 (q8m4 8.8)  rbm4 27.9 (q8m4 31.7)  q8m8 148.4 (rbm8 155.3)
//	out proj       q8m2 4.1 (rbm2 4.4)  q8m4 10.0 (rbm4 10.1)  q8m4 45.9 (rbm8 51.9)
//	down           q8m2 8.6 (rbm2 8.6)  q8m4 20.8 (rbm4 21.0)  q8m4 117.0 (rbm8 140.8)
//
// (gate+up has its own table, at gluFor.) The boundaries sit between the
// measured lengths.
func q8For(rows, n int) gemmKernel {
	if n <= 4096 { // out proj, down
		if rows <= 128 {
			return q8Kernels[0]
		}
		return q8Kernels[1]
	}
	if rows <= 1024 { // in proj
		return gluKernels[4]
	}
	return q8Kernels[2]
}

// gluFor picks the gate+up+SwiGLU rung: glum2 16.9 ms at 101 tokens (glum4
// 17.5), glum8 40.2 at 494 (glum4 42.7) and 199.9 at 2,446 (glum4 245.4).
func gluFor(rows int) gemmKernel {
	if rows <= 128 {
		return gluKernels[0]
	}
	return gluKernels[2]
}

const (
	tile    = 16  // cooperative-matrix extent
	gemmPad = 128 // A-operand row pad in halves (zimage/qwen's §2.3)
	nAlign  = 256 // every projection's N is padded to this
	// rowAlign is the arena's row rounding: the widest GEMM tile's BM.
	rowAlign = 128
	// maxBankBytes is the largest storage buffer this device addresses.
	maxBankBytes = 0xfffffffc
	// ropeRows is the rotary table's length: a row is at most MaxRow tokens,
	// so no position reaches it.
	ropeRows = MaxRow
)

// Fused input projection widths.
const (
	gdnN  = 12544 // 8192 + 4096 + 32 + 32 = 12352, padded to 256
	attnN = 10240
)

type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	waves  int
	// rowFast is kev_gemm_q8_glu's grid: x = row block, y = column tile.
	rowFast bool
}

// The rungs zimage/qwen's PlanFor chooses between for a 2560-wide Qwen, which
// this is.
var gemmKernels = []gemmKernel{
	{"reg32x128", shaders.DiTGEMMReg32x128Tiled, 32, 128, 1, false},
	{"reg64", shaders.DiTGEMMReg64Tiled, 64, 64, 1, false},
	{"wg128x256", shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256, 4, false},
}

// gemmFor is zimage/qwen's measured schedule for hidden 2560.
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

// proj is one projection's place in a bank.
type proj struct {
	off  uint32 // halves on the fp16 bank, bytes on the int8 one
	n, k int
}

type layerWeights struct {
	attn bool
	bank int
	// fp16 bank
	in, out, gate, up, down proj
	// int8 bank: gate and up interleaved a 32-row group at a time, for the
	// GLU epilogue (K7.6); gate and up are unused there
	glu proj
	// fp32 arena
	inNorm, postNorm uint32
	conv, gates      uint32 // GDN: taps [8192][4]; -exp(A_log) [32] then dt_bias [32]
	gnorm            uint32 // GDN: the gated norm's plain weight [128]
	qkNorm           uint32 // attention: q_norm (1+w) [256] then k_norm (1+w) [256]
	// act arena, per layer
	state, tail uint32 // GDN: S [32][128][128]; conv tail [3][8192]
	// fp16 arena, per layer (K7.3)
	kc, vc uint32 // attention: key and value caches, fp16 [cells][4][256], in halves
}

// GPU is a loaded Kev-4B.
type GPU struct {
	Bank Bank
	Cfg  *Config
	Head *Head
	Norm []float32 // the final norm, (1 + w)

	dev   *vk.Device
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	banks []*vk.Buffer
	pipes map[string]*vk.ComputePipeline
	gemms []map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	embed []uint16 // bf16 [vocab][hidden], gathered on the host

	// GEMM names a rung to use at every length instead of the schedule, for
	// the ladder; empty is the schedule. GLU does the same for the gate+up
	// GEMM (glum2, glum4, glum8).
	GEMM, GLU string
	// LayersPerSubmit is how many layers go into one command buffer (see
	// run). Load sets 8.
	LayersPerSubmit int
	// PassRows caps a pass below the arena, so a test can make a small
	// request take several passes. Zero is the arena.
	PassRows int
	// Attention is "wmma" (kev_attn_wmma, the default) or "naive" (K4's
	// kev_attention, the control). Both read the same fp16 cache.
	Attention string
	// ScanLPC is how many lanes share a column of the GDN state in the scan
	// (1, 2, 4, 8 or 16; K7.4). Load sets the ladder's winner.
	ScanLPC int
	// Poison fills every per-pass scratch tensor with NaN before a pass, so
	// that a kernel reading something this pass never wrote shows up as a NaN
	// at a readout. A debugging mode (TestGPUReadsOnlyWhatItWrote).
	Poison bool

	cache *prefixCache
	slots int // requests one pass can hold: state slots in the GDN block (K7.5)

	rows  int // arena rows: the longest pass
	cells int // attention cache cells
	w     []layerWeights
	rope  uint32

	// act arena
	aX, aP, aG, aO, aY, aMeta, aSeg uint32
	actElems                        int
	// Every GDN layer's S and conv tail, contiguous so the prefix cache
	// moves them in one copy.
	gdnBlock  uint32
	gdnFloats int
	// fp16 arena
	hA, hQ uint32 // o_proj's and every GEMM's A operand; fp16 q for the attention
	hMLP   uint32 // down's A operand when the GLU epilogue writes it (int8 bank)
	hElems int
	lda    int // the widest A operand's row stride
}

// Options are what a load decides up front.
type Options struct {
	// MaxTokens is the longest pass: the arena's rows. A request longer than
	// this runs as several passes. Zero takes 8192.
	MaxTokens int
	// Bank is the weight width.
	Bank Bank
	// CacheStates is how many states the prefix cache keeps; zero is none.
	// Load sets 4, Kev's default.
	CacheStates int
	// CacheTokens is the longest state it keeps. A slot is 52.7 MB of
	// recurrent state plus 32 KB a token of KV (8 attention layers, K and V,
	// 1024 halves each). Load sets 4096: 181 MB a slot, 0.7 GB for four.
	CacheTokens int
	// Slots is how many requests one pass can hold (K7.5): each needs its
	// own GDN state, 52.7 MB of the fp32 arena. Load sets 8.
	Slots int
}

// DefaultOptions is int8 weights, passes of 8192 tokens, and four cached
// states of up to 4096 tokens (Kev's serving table measures a 2,200-token
// text).
func DefaultOptions() Options {
	return Options{MaxTokens: 8192, Bank: BankQ8, CacheStates: 4, CacheTokens: 4096, Slots: 8}
}

// Load builds the model from Kev's checkpoint directory (adapter, head,
// tokenizer) and the base it names, for passes of at most maxTokens packed
// tokens, with DefaultOptions otherwise.
func Load(dev *vk.Device, kevDir, baseDir string, maxTokens int) (*GPU, error) {
	return LoadBank(dev, kevDir, baseDir, maxTokens, BankQ8)
}

// LoadBank is Load with the weight width named.
func LoadBank(dev *vk.Device, kevDir, baseDir string, maxTokens int, bank Bank) (*GPU, error) {
	o := DefaultOptions()
	o.MaxTokens, o.Bank = maxTokens, bank
	return LoadWith(dev, kevDir, baseDir, o)
}

// LoadWith is Load with every option named.
func LoadWith(dev *vk.Device, kevDir, baseDir string, o Options) (*GPU, error) {
	if o.MaxTokens <= 0 {
		o.MaxTokens = 8192
	}
	maxTokens, bank := o.MaxTokens, o.Bank
	o.Slots = max(o.Slots, 1)
	cfg, err := LoadConfig(baseDir)
	if err != nil {
		return nil, err
	}
	head, err := LoadHead(kevDir)
	if err != nil {
		return nil, err
	}
	if head.Dim <= 0 || len(head.QW) != head.Dim*cfg.Hidden {
		return nil, fmt.Errorf("kev: head is %d x %d, want %d x %d", len(head.QW)/max(head.Dim, 1), head.Dim, head.Dim, cfg.Hidden)
	}
	ad, err := OpenAdapter(kevDir)
	if err != nil {
		return nil, err
	}
	defer ad.Close()
	set, err := safetensors.OpenSet(baseDir)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	g := &GPU{Bank: bank, Cfg: cfg, Head: head, dev: dev, pipes: map[string]*vk.ComputePipeline{}, LayersPerSubmit: 8, Attention: "wmma", ScanLPC: 8, slots: o.Slots}
	g.rows = roundUp(maxTokens, rowAlign)
	// A pass of branches continues a state of up to MaxState cells, each
	// branch starting on a 64-aligned cell; the planner splits a pass whose
	// padding would run past this.
	g.cells = MaxState + 2*g.rows
	g.lda = cfg.Intermediate + gemmPad
	if err := g.layout(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if g.cache, err = newPrefixCache(g, o.CacheStates, min(o.CacheTokens, MaxState)); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(set, ad); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// Rows is the longest pass this instance runs.
func (g *GPU) Rows() int { return g.rows }

// layout plans every arena and allocates them.
func (g *GPU) layout() error {
	c := g.Cfg
	R := g.rows
	// fp32 weights: rope table, final norm, then per layer.
	w32 := 0
	alloc32 := func(n int) uint32 { off := uint32(w32); w32 += (n + 63) &^ 63; return off }
	g.rope = alloc32(2 * ropeRows * 32)

	// act arena.
	alloc := func(n int) uint32 { off := uint32(g.actElems); g.actElems += (n + 63) &^ 63; return off }
	g.aX = alloc(R * c.Hidden)
	g.aP = alloc(R * 2 * c.Intermediate) // >= gdnN and attnN, and holds gate and up side by side
	g.aG = alloc(R * (2048 + 2048 + 4096 + 32 + 32))
	g.aO = alloc(R * 4096)
	g.aY = alloc(R * c.Hidden)
	g.aMeta = alloc(R * 8) // 8 words a row (K7.5)
	g.aSeg = alloc(R * 4)  // 4 words a segment, and a segment is at least a row

	// Banks are counted in bytes. A matrix is n*k halves on the fp16 bank,
	// and n*k bytes plus an fp16 scale per 32 on the int8 one, rounded up to
	// sixteen so the next matrix's words are aligned.
	size := func(n, k int) int {
		if g.Bank == BankQ8 {
			return (n*k + n*k/q8Group*2 + 15) &^ 15
		}
		return n * k * 2
	}
	bankBytes := []int{0}
	put := func(n, k int) proj {
		at := bankBytes[len(bankBytes)-1]
		off := at
		if g.Bank == BankFP16 {
			off /= 2
		}
		bankBytes[len(bankBytes)-1] += size(n, k)
		return proj{off: uint32(off), n: n, k: k}
	}
	g.w = make([]layerWeights, c.Layers)
	const gdnLayer = 32*128*128 + 3*8192 // S, then the conv tail
	nGDN := 0
	for i := 0; i < c.Layers; i++ {
		if !c.Attention(i) {
			nGDN++
		}
	}
	g.gdnFloats = nGDN * gdnLayer // one slot; slot i is at gdnBlock + i*gdnFloats
	g.gdnBlock = alloc(g.gdnFloats * g.slots)
	gdnAt := g.gdnBlock
	for i := range g.w {
		lw := layerWeights{attn: c.Attention(i)}
		// A layer's projections all go in one bank: start a new one if the
		// whole layer would not fit.
		inN := gdnN
		if lw.attn {
			inN = attnN
		}
		layerBytes := size(inN, c.Hidden) + size(c.Hidden, 4096) + 3*size(c.Intermediate, c.Hidden)
		if g.Bank == BankQ8 {
			layerBytes = size(inN, c.Hidden) + size(c.Hidden, 4096) + size(2*c.Intermediate, c.Hidden) + size(c.Hidden, c.Intermediate)
		}
		if bankBytes[len(bankBytes)-1]+layerBytes > maxBankBytes {
			bankBytes = append(bankBytes, 0)
		}
		lw.bank = len(bankBytes) - 1
		lw.in = put(inN, c.Hidden)
		lw.out = put(c.Hidden, 4096)
		if g.Bank == BankQ8 {
			lw.glu = put(2*c.Intermediate, c.Hidden)
		} else {
			lw.gate = put(c.Intermediate, c.Hidden)
			lw.up = put(c.Intermediate, c.Hidden)
		}
		lw.down = put(c.Hidden, c.Intermediate)

		lw.inNorm = alloc32(c.Hidden)
		lw.postNorm = alloc32(c.Hidden)
		if lw.attn {
			lw.qkNorm = alloc32(2 * c.HeadDim)
		} else {
			lw.conv = alloc32(8192 * 4)
			lw.gates = alloc32(64)
			lw.gnorm = alloc32(128)
			lw.state = gdnAt
			lw.tail = gdnAt + 32*128*128
			gdnAt += gdnLayer
		}
		g.w[i] = lw
	}
	g.hA = 0
	g.hElems = R * g.lda
	halloc := func(n int) uint32 { off := uint32(g.hElems); g.hElems += (n + 63) &^ 63; return off }
	g.hQ = halloc(R * 16 * 256)
	if g.Bank == BankQ8 {
		g.hMLP = halloc(R * g.lda)
	}
	for i := range g.w {
		if g.w[i].attn {
			// Padded by a key block: the attention loads whole blocks of 64.
			g.w[i].kc = halloc((g.cells + 64) * kvRow)
			g.w[i].vc = halloc((g.cells + 64) * kvRow)
		}
	}
	if g.hElems*2 > maxBankBytes {
		return fmt.Errorf("kev: the fp16 arena is %d MB, over the 4 GiB a buffer can address", (g.hElems*2)>>20)
	}

	if g.actElems*4 > maxBankBytes {
		return fmt.Errorf("kev: the activation arena is %d MB at %d rows, over the 4 GiB a buffer can address", (g.actElems*4)>>20, R)
	}
	var err error
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return err
	}
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("kev: activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	// Zeroed once, like the fp16 arena: a GEMM's pad rows (past a pass's n,
	// up to its tile) read whatever is here, and fresh device memory can hold
	// NaN bit patterns.
	g.abuf.Zero()
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return err
	}
	g.hbuf.ZeroUint16At(0, g.hElems)
	for _, n := range bankBytes {
		b, err := g.dev.NewBuffer(n)
		if err != nil {
			return fmt.Errorf("kev: weight bank (%d MB): %w", n>>20, err)
		}
		g.banks = append(g.banks, b)
	}
	return nil
}

func (g *GPU) build() error {
	pc := uint32(pushBytes)
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[0]}
	for name, spirv := range map[string][]byte{
		"normf16":  shaders.DiTNormScaleF16,
		"swiglu":   shaders.DiTSwiGLUF16,
		"add":      shaders.DiTGateAdd,
		"gdnprep":  shaders.KevGDNPrep,
		"gdnnorm":  shaders.KevGDNNorm,
		"attnprep": shaders.KevAttnPrep,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc}); err != nil {
			return err
		}
	}
	if err := g.pipeline("attention", shaders.KevAttention, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc, RequiredSubgroupSize: 64}); err != nil {
		return err
	}
	for lpc, spirv := range shaders.KevGDNScan {
		if err := g.pipeline(fmt.Sprintf("gdnscan%d", lpc), spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc, RequiredSubgroupSize: 64}); err != nil {
			return err
		}
	}
	if err := g.pipeline("attnwmma", shaders.KevAttnWMMA, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc, RequiredSubgroupSize: 64}); err != nil {
		return err
	}
	g.gemms = make([]map[string]*vk.ComputePipeline, len(g.banks))
	for b, bank := range g.banks {
		g.gemms[b] = map[string]*vk.ComputePipeline{}
		kernels := gemmKernels
		if g.Bank == BankQ8 {
			kernels = append(append([]gemmKernel(nil), q8Kernels...), gluKernels...)
		}
		for _, k := range kernels {
			spec := vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank}, PushConstantSize: pc}
			if k.waves > 1 {
				spec.RequiredSubgroupSize = 64
			}
			if g.Bank == BankQ8 {
				// llm_common.glsl's bindings: the bank is read as halves at 3
				// (the scale plane) and as words at 5 (the tiles); 4 is
				// declared and unused by this arm.
				spec.Buffers = []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank, g.abuf, bank}
				spec.RequiredSubgroupSize = 64
			}
			mod, err := g.dev.NewShaderModule(k.spirv)
			if err != nil {
				return err
			}
			g.mods = append(g.mods, mod)
			pipe, err := g.dev.NewPipeline(mod, spec)
			if err != nil {
				return fmt.Errorf("kev: gemm %s on bank %d: %w", k.name, b, err)
			}
			g.gemms[b][k.name] = pipe
		}
	}
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("kev: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("kev: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// GEMMRungs lists the rungs this instance's bank can run, for the ladder.
func (g *GPU) GEMMRungs() []string {
	ks := gemmKernels
	if g.Bank == BankQ8 {
		ks = append(append([]gemmKernel(nil), q8Kernels...), gluKernels[3:]...)
	}
	var out []string
	for _, k := range ks {
		out = append(out, k.name)
	}
	return out
}

// Destroy releases every device object.
func (g *GPU) Destroy() {
	g.cache.destroy()
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, set := range g.gemms {
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

// ---- staging

func (g *GPU) stage(set *safetensors.Set, ad *Adapter) error {
	c := g.Cfg
	const pre = "model.language_model."
	get := func(name string) ([]float32, []int, error) {
		t, err := set.Get(pre + name)
		if err != nil {
			return nil, nil, err
		}
		v, err := t.F32(nil)
		return v, t.Shape, err
	}
	onePlus := func(w []float32) []float32 {
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = 1 + v
		}
		return out
	}

	// Embeddings stay bf16 on the host; a pass gathers its rows.
	et, err := set.Get(pre + "embed_tokens.weight")
	if err != nil {
		return err
	}
	if et.DType != "BF16" || len(et.Shape) != 2 || et.Shape[1] != c.Hidden {
		return fmt.Errorf("kev: embed_tokens is %s %v", et.DType, et.Shape)
	}
	g.embed = make([]uint16, et.Elems())
	copy(g.embed, unsafe.Slice((*uint16)(unsafe.Pointer(&et.Data[0])), et.Elems()))

	norm, _, err := get("norm.weight")
	if err != nil {
		return err
	}
	g.Norm = onePlus(norm)
	g.wbuf.WriteFloat32At(int(g.rope), ropeTable(c))

	merged := 0
	for i := range g.w {
		lw := &g.w[i]
		L := fmt.Sprintf("layers.%d.", i)
		in, _, err := get(L + "input_layernorm.weight")
		if err != nil {
			return err
		}
		post, _, err := get(L + "post_attention_layernorm.weight")
		if err != nil {
			return err
		}
		g.wbuf.WriteFloat32At(int(lw.inNorm), onePlus(in))
		g.wbuf.WriteFloat32At(int(lw.postNorm), onePlus(post))

		// weight loads one projection, merges its adapter and returns it.
		weight := func(name string, out, k int) ([]float32, error) {
			w, shape, err := get(L + name + ".weight")
			if err != nil {
				return nil, err
			}
			if len(shape) != 2 || shape[0] != out || shape[1] != k {
				return nil, fmt.Errorf("kev: %s%s is %v, want [%d %d]", L, name, shape, out, k)
			}
			if err := ad.Merge(L+name, w, out, k); err != nil {
				return nil, err
			}
			merged++
			return w, nil
		}
		// fuse stacks projections by rows into a padded [n, k].
		fuse := func(n, k int, parts ...[]float32) []float32 {
			out := make([]float32, n*k)
			at := 0
			for _, p := range parts {
				copy(out[at:], p)
				at += len(p)
			}
			return out
		}

		var inW []float32
		if lw.attn {
			q, err := weight("self_attn.q_proj", 2*c.Heads*c.HeadDim, c.Hidden)
			if err != nil {
				return err
			}
			k, err := weight("self_attn.k_proj", c.KVHeads*c.HeadDim, c.Hidden)
			if err != nil {
				return err
			}
			v, err := weight("self_attn.v_proj", c.KVHeads*c.HeadDim, c.Hidden)
			if err != nil {
				return err
			}
			inW = fuse(attnN, c.Hidden, q, k, v)
			o, err := weight("self_attn.o_proj", c.Hidden, c.Heads*c.HeadDim)
			if err != nil {
				return err
			}
			if err := g.packInto(lw.bank, lw.out, o); err != nil {
				return err
			}
			qn, _, err := get(L + "self_attn.q_norm.weight")
			if err != nil {
				return err
			}
			kn, _, err := get(L + "self_attn.k_norm.weight")
			if err != nil {
				return err
			}
			g.wbuf.WriteFloat32At(int(lw.qkNorm), append(onePlus(qn), onePlus(kn)...))
		} else {
			qkv, err := weight("linear_attn.in_proj_qkv", 8192, c.Hidden)
			if err != nil {
				return err
			}
			z, err := weight("linear_attn.in_proj_z", 4096, c.Hidden)
			if err != nil {
				return err
			}
			a, err := weight("linear_attn.in_proj_a", 32, c.Hidden)
			if err != nil {
				return err
			}
			b, err := weight("linear_attn.in_proj_b", 32, c.Hidden)
			if err != nil {
				return err
			}
			inW = fuse(gdnN, c.Hidden, qkv, z, a, b)
			o, err := weight("linear_attn.out_proj", c.Hidden, 4096)
			if err != nil {
				return err
			}
			if err := g.packInto(lw.bank, lw.out, o); err != nil {
				return err
			}
			conv, shape, err := get(L + "linear_attn.conv1d.weight")
			if err != nil {
				return err
			}
			if len(conv) != 8192*4 {
				return fmt.Errorf("kev: %sconv1d is %v", L, shape)
			}
			g.wbuf.WriteFloat32At(int(lw.conv), conv)
			alog, _, err := get(L + "linear_attn.A_log")
			if err != nil {
				return err
			}
			dt, _, err := get(L + "linear_attn.dt_bias")
			if err != nil {
				return err
			}
			gates := make([]float32, 64)
			for h := 0; h < 32; h++ {
				// HF: -self.A_log.float().exp(), in fp32.
				gates[h] = -float32(math.Exp(float64(alog[h])))
				gates[32+h] = dt[h]
			}
			g.wbuf.WriteFloat32At(int(lw.gates), gates)
			gn, _, err := get(L + "linear_attn.norm.weight")
			if err != nil {
				return err
			}
			g.wbuf.WriteFloat32At(int(lw.gnorm), gn)
		}
		if err := g.packInto(lw.bank, lw.in, inW); err != nil {
			return err
		}
		gate, err := weight("mlp.gate_proj", c.Intermediate, c.Hidden)
		if err != nil {
			return err
		}
		up, err := weight("mlp.up_proj", c.Intermediate, c.Hidden)
		if err != nil {
			return err
		}
		if g.Bank == BankQ8 {
			if err := g.packInto(lw.bank, lw.glu, interleaveGLU(gate, up, c.Intermediate, c.Hidden)); err != nil {
				return err
			}
		} else {
			if err := g.packInto(lw.bank, lw.gate, gate); err != nil {
				return err
			}
			if err := g.packInto(lw.bank, lw.up, up); err != nil {
				return err
			}
		}
		down, err := weight("mlp.down_proj", c.Hidden, c.Intermediate)
		if err != nil {
			return err
		}
		if err := g.packInto(lw.bank, lw.down, down); err != nil {
			return err
		}
	}
	if merged*2 != ad.Count() {
		return fmt.Errorf("kev: merged %d projections, the adapter has %d tensors", merged, ad.Count())
	}
	return nil
}

// packInto narrows a [n, k] weight into its bank in the fragment tiling:
// tile (nt, kt) is 256 contiguous halves holding (k, n) at (n%16)*16 + k%16,
// tiles kt-fastest. It refuses a value fp16 cannot hold.
func (g *GPU) packInto(bank int, p proj, w []float32) error {
	if len(w) != p.n*p.k {
		return fmt.Errorf("kev: packing %d values into [%d %d]", len(w), p.n, p.k)
	}
	if g.Bank == BankQ8 {
		return g.packQ8(bank, p, w)
	}
	dst := make([]uint16, p.n*p.k)
	kt := p.k / tile
	var mu sync.Mutex
	var bad error
	parallelRows(p.n, func(i int) {
		row := w[i*p.k : (i+1)*p.k]
		base := (i / tile) * kt * tile * tile
		lane := (i % tile) * tile
		for j, v := range row {
			if math.Abs(float64(v)) > 65504 {
				mu.Lock()
				bad = fmt.Errorf("kev: weight %v overflows fp16", v)
				mu.Unlock()
			}
			dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
		}
	})
	if bad != nil {
		return bad
	}
	g.banks[bank].WriteUint16At(int(p.off), dst)
	return nil
}

// packQ8 stages a [n, k] weight as int8 in the fragment tiling plus its
// scale plane -- llm.tileBQ8's layout: tile (nt, kt) is 256 bytes holding
// (k, n) at (n%16)*16 + k%16, tiles kt-fastest; scale (nt, group g, n%16) at
// (nt*kgroups + g)*16 + n%16, after the tiles.
//
// One thing differs from the LLM's, and it is deliberate: there a Q8_0
// checkpoint is re-derived exactly, so q is taken against the fp32 amax/127.
// Here the weights are bf16 plus a LoRA delta and this *is* the quantisation,
// so q is rounded against the scale the kernel will actually multiply by --
// the fp16 one -- which is the smaller error of the two.
func (g *GPU) packQ8(bank int, p proj, w []float32) error {
	n, k := p.n, p.k
	if k%q8Group != 0 || n%tile != 0 {
		return fmt.Errorf("kev: [%d %d] does not tile for the int8 bank", n, k)
	}
	kt, kg := k/tile, k/q8Group
	q := make([]byte, n*k)
	sc := make([]uint16, n*k/q8Group)
	parallelRows(n, func(r int) {
		base := (r / tile) * kt * tile * tile
		lane := (r % tile) * tile
		sbase := (r/tile)*kg*tile + r%tile
		x := w[r*k : (r+1)*k]
		for gi := 0; gi < kg; gi++ {
			blk := x[gi*q8Group : (gi+1)*q8Group]
			var amax float32
			for _, v := range blk {
				amax = max(amax, float32(math.Abs(float64(v))))
			}
			dh := safetensors.F32ToF16(amax / 127)
			d := safetensors.F16ToF32(dh)
			sc[sbase+gi*tile] = dh
			for j, v := range blk {
				var qi int32
				if d != 0 {
					qi = int32(math.Round(float64(v / d)))
				}
				qi = min(max(qi, -127), 127)
				c := gi*q8Group + j
				q[base+(c/tile)*tile*tile+lane+c%tile] = byte(int8(qi))
			}
		}
	})
	g.banks[bank].WriteBytesAt(int(p.off), q)
	g.banks[bank].WriteUint16At((int(p.off)+len(q))/2, sc)
	return nil
}

// gluGroup is how many gate rows, then as many up rows, one 64-column group
// of the fused projection holds: kev_gemm_q8_glu's BN / 2.
const gluGroup = 32

// interleaveGLU stacks gate and up ([n, k] each) into [2n, k] in groups:
// gate rows [32g, 32g+32) then up rows [32g, 32g+32), so a 64-column tile of
// the fused GEMM holds both halves of the same 32 outputs.
func interleaveGLU(gate, up []float32, n, k int) []float32 {
	out := make([]float32, 2*n*k)
	for g0 := 0; g0 < n; g0 += gluGroup {
		dst := 2 * g0 * k
		copy(out[dst:dst+gluGroup*k], gate[g0*k:(g0+gluGroup)*k])
		copy(out[dst+gluGroup*k:dst+2*gluGroup*k], up[g0*k:(g0+gluGroup)*k])
	}
	return out
}

// ropeTable is HF's cos and sin for the 64 rotary dims, [pos][32] each: fp32
// inv_freq = 1 / theta^(2i/64), times the fp32 position, then cos and sin of
// that fp32 product.
func ropeTable(c *Config) []float32 {
	dims := int(float64(c.HeadDim) * c.RotaryFactor)
	half := dims / 2
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(1 / math.Pow(c.RopeTheta, float64(2*i)/float64(dims)))
	}
	out := make([]float32, 2*ropeRows*half)
	for p := 0; p < ropeRows; p++ {
		for i := 0; i < half; i++ {
			f := float64(inv[i] * float32(p))
			out[p*half+i] = float32(math.Cos(f))
			out[ropeRows*half+p*half+i] = float32(math.Sin(f))
		}
	}
	return out
}

// ---- passes

// preq is one request's place in a pass (K7.5: a pass may hold several).
type preq struct {
	enc       *Encoding
	slot      int  // its state slot: S and conv tail of every GDN layer
	sLo       int  // its state's first attention cell
	withState bool // its state's rows are in this pass (else they are resident)
	stateRow  int  // the pass row of its state's first token, when they are
}

// prow names one pass row: which request of the pass, which index of its
// encoding.
type prow struct{ q, e int }

// pass is one forward over the device. Its rows are segments laid end to end:
// requests' states and branches. A request whose state is not in the pass
// runs against its slot and its cells, which an earlier pass or the prefix
// cache left there.
type pass struct {
	reqs     []*preq
	src      []prow
	meta     []uint32 // 8 words a row, as kev_gdn_prep.comp documents
	states   [][3]int // state segments: first row, length, slot
	branches [][3]int // branch segments, the same
	readouts []prow
	cursor   int // the next free attention cell
	n        int
}

// keyAlign is the attention's key block (kev_attn_wmma's BN). Every state's
// and every branch's keys start on a multiple of it, so a key block never
// mixes two segments, and a row's attention is the same bits wherever its
// segment sits in the cache (K7.3).
const keyAlign = 64

// row appends one pass row and its metadata.
func (p *pass) row(q, e, cell int, branch bool, segStart, sLo, sHi, stateEnd int) {
	r := p.reqs[q]
	p.src = append(p.src, prow{q, e})
	b := uint32(0)
	if branch {
		b = 1
	}
	p.meta = append(p.meta, uint32(r.enc.Pos[e]), b, uint32(segStart), uint32(cell),
		uint32(sLo), uint32(sHi), uint32(r.slot), uint32(stateEnd))
}

// addState appends request q's state rows at its cells.
func (p *pass) addState(q int) {
	r := p.reqs[q]
	ls := r.enc.StateLen
	r.stateRow = len(p.src)
	p.states = append(p.states, [3]int{r.stateRow, ls, r.slot})
	for e := 0; e < ls; e++ {
		// A state row: its own segment from the state's first cell, no
		// separate state range, and its state's end for the conv tail.
		p.row(q, e, r.sLo+e, false, r.stateRow, 0, 0, r.stateRow+ls)
	}
	p.cursor = max(p.cursor, r.sLo+ls)
}

// fitsBranch is whether a branch of `length` rows fits.
func (p *pass) fitsBranch(length, rows, cells int) bool {
	return len(p.src)+length <= rows && roundUp(p.cursor, keyAlign)+length <= cells
}

// addBranch appends request q's branch k at the next aligned cell.
func (p *pass) addBranch(q, k int) {
	r := p.reqs[q]
	lo, hi := r.enc.Branch(k)
	at := roundUp(p.cursor, keyAlign)
	first := len(p.src)
	p.branches = append(p.branches, [3]int{first, hi - lo, r.slot})
	end := 0
	if r.withState {
		end = r.stateRow + r.enc.StateLen
	}
	for e := lo; e < hi; e++ {
		p.row(q, e, at+e-lo, true, first, r.sLo, r.sLo+r.enc.StateLen, end)
	}
	p.cursor = at + hi - lo
	p.readouts = append(p.readouts, prow{q, r.enc.Decide[k]})
	for _, o := range r.enc.Opts[k] {
		p.readouts = append(p.readouts, prow{q, o})
	}
}

// plan cuts one request into passes of at most g.passRows() rows, in slot 0
// at cell 0: the state plus as many branches as fit (unless the state is
// resident), then branches alone. A branch longer than a pass is an error, as
// it would be a 422 at the door.
func (g *GPU) plan(enc *Encoding, resident bool) ([]*pass, error) {
	limit := g.passRows()
	if !resident && enc.StateLen > limit {
		return nil, &OverflowError{Msg: fmt.Sprintf("a %d-token state does not fit a pass of %d tokens (-kev-tokens)", enc.StateLen, limit)}
	}
	fresh := func(withState bool) *pass {
		p := &pass{reqs: []*preq{{enc: enc, withState: withState}}, cursor: enc.StateLen}
		if withState {
			p.addState(0)
		}
		return p
	}
	var out []*pass
	p := fresh(!resident)
	for k := range enc.Decide {
		lo, hi := enc.Branch(k)
		if !p.fitsBranch(hi-lo, limit, g.cells) {
			if len(p.src) > 0 {
				p.n = len(p.src)
				out = append(out, p)
				p = fresh(false)
			}
			if !p.fitsBranch(hi-lo, limit, g.cells) {
				return nil, &OverflowError{Msg: fmt.Sprintf("question %d is %d tokens, longer than a pass of %d tokens (-kev-tokens)", k, hi-lo, limit)}
			}
		}
		p.addBranch(0, k)
	}
	p.n = len(p.src)
	if p.n > 0 {
		out = append(out, p)
	}
	return out, nil
}

// planBatch lays several requests out in one pass, request i in slot i with
// its state on the next aligned cells, or reports false if they do not fit
// its rows, cells or slots.
func (g *GPU) planBatch(encs []*Encoding, resident []bool) (*pass, bool) {
	if len(encs) > g.slots {
		return nil, false
	}
	p := &pass{}
	for i, enc := range encs {
		r := &preq{enc: enc, slot: i, sLo: roundUp(p.cursor, keyAlign), withState: !resident[i]}
		p.reqs = append(p.reqs, r)
		if r.sLo+enc.StateLen > g.cells {
			return nil, false
		}
		if r.withState {
			p.addState(i)
		} else {
			p.cursor = r.sLo + enc.StateLen
		}
	}
	for i, enc := range encs {
		for k := range enc.Decide {
			lo, hi := enc.Branch(k)
			if !p.fitsBranch(hi-lo, g.passRows(), g.cells) {
				return nil, false
			}
			p.addBranch(i, k)
		}
	}
	p.n = len(p.src)
	return p, p.n > 0 && p.n <= g.passRows()
}

// passRows is the most rows a pass takes: the arena, or PassRows when a test
// asks for less.
func (g *GPU) passRows() int {
	if g.PassRows > 0 {
		return min(g.PassRows, g.rows)
	}
	return g.rows
}

// Pass is what one request's forward reads back: the final-normed hidden
// state at every readout row, in its encoding's index space.
type Pass struct {
	Hidden   map[int][]float32
	GPU      time.Duration // summed dispatch time, copies included; in a batch, the batch's
	Passes   int           // device passes the request took
	CacheHit bool          // the state came from the prefix cache
	Batch    int           // requests in the pass that answered it (K7.5)
}

// upload writes a pass's embeddings, token metadata and segment tables.
func (g *GPU) upload(p *pass) error {
	c := g.Cfg
	if p.n == 0 || p.n > g.rows {
		return fmt.Errorf("kev: a pass of %d tokens; this instance holds %d", p.n, g.rows)
	}
	if g.Poison {
		g.poison()
	}
	x := make([]float32, p.n*c.Hidden)
	for i, pr := range p.src {
		id := p.reqs[pr.q].enc.IDs[pr.e]
		if id < 0 || int(id)*c.Hidden >= len(g.embed) {
			return fmt.Errorf("kev: token id %d is outside the embedding table", id)
		}
		row := g.embed[int(id)*c.Hidden : (int(id)+1)*c.Hidden]
		for j, b := range row {
			x[i*c.Hidden+j] = math.Float32frombits(uint32(b) << 16)
		}
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.WriteUint32At(int(g.aMeta), p.meta)
	tab := make([]uint32, 0, 4*(len(p.states)+len(p.branches)))
	for _, sg := range append(append([][3]int(nil), p.states...), p.branches...) {
		tab = append(tab, uint32(sg[0]), uint32(sg[1]), uint32(sg[2]), 0)
	}
	g.abuf.WriteUint32At(int(g.aSeg), tab)
	// A state shorter than the conv window leaves tail rows the prep kernel
	// never writes; they are the zeros before a sequence's start.
	for _, r := range p.reqs {
		if r.withState && r.enc.StateLen < 3 {
			for i := range g.w {
				if !g.w[i].attn {
					g.abuf.ZeroFloat32At(int(g.w[i].tail)+r.slot*g.gdnFloats, 3*8192)
				}
			}
		}
	}
	return nil
}

// poison fills the per-pass scratch -- not the residual stream, the metadata
// or the carried state (KV cells, S, tails) -- with NaN.
func (g *GPU) poison() {
	R := g.rows
	nan := math.Float32frombits(0x7fc00000)
	fill := func(off uint32, n int) {
		v := make([]float32, n)
		for i := range v {
			v[i] = nan
		}
		g.abuf.WriteFloat32At(int(off), v)
	}
	fill(g.aX, R*g.Cfg.Hidden)
	fill(g.aP, R*2*g.Cfg.Intermediate)
	fill(g.aG, R*(2048+2048+4096+32+32))
	fill(g.aO, R*4096)
	fill(g.aY, R*g.Cfg.Hidden)
	h := make([]uint16, R*g.lda)
	for i := range h {
		h[i] = 0x7e00 // fp16 NaN
	}
	g.hbuf.WriteUint16At(int(g.hA), h)
	g.hbuf.WriteUint16At(int(g.hQ), h[:R*16*256])
	if g.hMLP != 0 {
		g.hbuf.WriteUint16At(int(g.hMLP), h)
	}
}

// layerGraph is one layer's dispatches over a pass, with labels.
func (g *GPU) layerGraph(i int, p *pass) ([]vk.MultiDispatch, []string) {
	c := g.Cfg
	lw := &g.w[i]
	n := p.n
	R := uint32(g.rows)
	eps := math.Float32bits(float32(c.RMSEps))
	// Each projection picks its own rung, and pads the pass to its own tile.
	rung := func(pr proj) gemmKernel {
		k := gemmFor(n)
		if g.Bank == BankQ8 {
			k = q8For(n, pr.n)
		}
		for _, r := range append(append(gemmKernels[:len(gemmKernels):len(gemmKernels)], q8Kernels...), gluKernels[3:]...) {
			if r.name == g.GEMM {
				k = r
			}
		}
		return k
	}
	base := push{Tokens: uint32(n), Span: R, Eps: eps, Aux1: g.aMeta}

	var d []vk.MultiDispatch
	var labels []string
	add := func(pipe, label string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		labels = append(labels, label)
	}
	normF16 := func(label string, wOff uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = g.aX, g.hA, wOff
		pc.Dim, pc.LDA = uint32(c.Hidden), uint32(c.Hidden+gemmPad)
		add("normf16", label, uint32(n), 1, pc)
	}
	gemmFrom := func(label string, pr proj, aOff, cOff uint32, lda int) {
		k := rung(pr)
		tokPad := roundUp(n, k.bm)
		if g.Bank == BankQ8 {
			pc := llmPush{xnOff: aOff, outOff: cOff, bOff: pr.off, lda: uint32(lda),
				gemmM: uint32(tokPad), gemmN: uint32(pr.n), gemmK: uint32(pr.k)}
			gx, gy := k.grid(pr.n, tokPad)
			d = append(d, vk.MultiDispatch{Pipeline: g.gemms[lw.bank][k.name], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
			labels = append(labels, label)
			return
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, pr.off
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(tokPad), uint32(pr.n), uint32(pr.k)
		pc.LDA = uint32(lda)
		d = append(d, vk.MultiDispatch{Pipeline: g.gemms[lw.bank][k.name], GroupsX: uint32(pr.n / k.bn), GroupsY: uint32(tokPad / k.bm), PushConstants: pc.bytes()})
		labels = append(labels, label)
	}
	gemm := func(label string, pr proj, cOff uint32, lda int) { gemmFrom(label, pr, g.hA, cOff, lda) }
	residual := func(label string) {
		pc := base
		pc.InOff, pc.OutOff, pc.Dim = g.aY, g.aX, uint32(c.Hidden)
		add("add", label, uint32(n), 1, pc)
	}

	normF16("in norm", lw.inNorm)
	gemm("in proj", lw.in, g.aP, c.Hidden+gemmPad)
	if lw.attn {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = g.aP, g.aG, lw.qkNorm
		pc.KOff, pc.VOff = lw.kc, lw.vc
		pc.Aux0, pc.Aux2, pc.BOff = uint32(lw.in.n), g.rope, g.hQ
		pc.Span = ropeRows
		add("attnprep", "attn prep", uint32(n), 1, pc)

		pc = base
		pc.InOff, pc.OutOff = g.aG, g.hA
		pc.KOff, pc.VOff = lw.kc, lw.vc
		pc.GemmK, pc.Aux0 = g.aP, uint32(lw.in.n)
		pc.LDA = uint32(4096 + gemmPad)
		if g.Attention == "naive" {
			add("attention", "attention", uint32(n), 4, pc)
		} else {
			pc.InOff = g.hQ
			add("attnwmma", "attention", groups(n, 16), 16, pc)
		}
	} else {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.KOff = g.aP, g.aG, lw.conv, lw.gates
		pc.Aux0, pc.VOff, pc.Dim = uint32(lw.in.n), lw.tail, uint32(g.gdnFloats)
		add("gdnprep", "gdn prep", uint32(n), 1, pc)

		pc = base
		pc.InOff, pc.OutOff, pc.KOff, pc.Aux1 = g.aG, g.aO, lw.state, g.aSeg
		pc.Dim = uint32(g.gdnFloats) // the slot stride
		scan := fmt.Sprintf("gdnscan%d", g.ScanLPC)
		wgs := uint32(32 * 128 * g.ScanLPC / 64) // 32 heads, 128 columns, 64/LPC columns a workgroup
		if len(p.states) > 0 {
			pc.Aux0, pc.Aux2 = 0, 2 // every state in the pass, from zero, keeping its final S in its slot
			add(scan, "gdn scan state", wgs, uint32(len(p.states)), pc)
		}
		if len(p.branches) > 0 {
			pc.Aux0, pc.Aux2 = uint32(len(p.states)), 1 // every branch, from its request's S
			add(scan, "gdn scan branches", wgs, uint32(len(p.branches)), pc)
		}

		pc = base
		pc.InOff, pc.KOff, pc.WOff, pc.OutOff = g.aO, g.aP, lw.gnorm, g.hA
		pc.Aux0, pc.LDA = uint32(lw.in.n), uint32(4096+gemmPad)
		add("gdnnorm", "gdn norm", uint32(n), 1, pc)
	}
	gemm("out proj", lw.out, g.aY, 4096+gemmPad)
	residual("mixer residual")

	normF16("post norm", lw.postNorm)
	if g.Bank == BankQ8 {
		// gate and up in one GEMM, SwiGLU in its epilogue, fp16 straight
		// into down's A operand (K7.6). It reads hA and so writes a second
		// slab, hMLP, which down then reads.
		glu := gluFor(n)
		if g.GLU != "" {
			for _, r := range gluKernels[:3] {
				if r.name == g.GLU {
					glu = r
				}
			}
		}
		gp := roundUp(n, glu.bm)
		pc := llmPush{xnOff: g.hA, outOff: g.hMLP, bOff: lw.glu.off, lda: uint32(c.Hidden + gemmPad), ldaLo: uint32(g.lda),
			gemmM: uint32(gp), gemmN: uint32(lw.glu.n), gemmK: uint32(lw.glu.k)}
		gx, gy := glu.grid(lw.glu.n, gp)
		d = append(d, vk.MultiDispatch{Pipeline: g.gemms[lw.bank][glu.name], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		labels = append(labels, "gate+up+swiglu")
		gemmFrom("down", lw.down, g.hMLP, g.aY, g.lda)
	} else {
		gemm("gate", lw.gate, g.aP, c.Hidden+gemmPad)
		gemm("up", lw.up, g.aP+R*uint32(c.Intermediate), c.Hidden+gemmPad)
		pc := base
		pc.InOff, pc.KOff, pc.OutOff = g.aP, g.aP+R*uint32(c.Intermediate), g.hA
		pc.Dim, pc.LDA, pc.Scale = uint32(c.Intermediate), uint32(g.lda), math.Float32bits(1)
		add("swiglu", "swiglu", uint32(n), 1, pc)
		gemm("down", lw.down, g.aY, g.lda)
	}
	residual("mlp residual")
	return d, labels
}

// run executes the first `layers` layers of one pass.
func (g *GPU) run(p *pass, layers int) (time.Duration, error) {
	if err := g.upload(p); err != nil {
		return 0, err
	}
	var total time.Duration
	// Layers are submitted LayersPerSubmit at a time: a submit is a fence
	// wait (worth ~2 ms of a 54 ms request at 8). The cap is there because a
	// long state's pass is seconds, and one command buffer that long can
	// outlive the driver's reset watchdog (zimage/qwen's perSubmit).
	per := max(g.LayersPerSubmit, 1)
	if p.n > 2048 {
		per = 1
	}
	for i := 0; i < layers; i += per {
		var d []vk.MultiDispatch
		var labels []string
		for j := i; j < min(i+per, layers); j++ {
			dj, lj := g.layerGraph(j, p)
			d, labels = append(d, dj...), append(labels, lj...)
		}
		dur, err := vk.DispatchMultiTimed(d, 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("kev: layers %d.. (%s): %w", i, strings.Join(labels, ","), err)
		}
		total += dur
	}
	return total, nil
}

// readRows reads pass rows back, final-normed when every layer ran, into one
// map per request of the pass, keyed by encoding index.
func (g *GPU) readRows(p *pass, rows []prow, final bool, into []map[int][]float32) {
	at := make(map[prow]int, len(p.src))
	for r, pr := range p.src {
		at[pr] = r
	}
	for _, pr := range rows {
		h := g.abuf.ReadFloat32At(int(g.aX)+at[pr]*g.Cfg.Hidden, g.Cfg.Hidden)
		if final {
			h = g.finalNorm(h)
		}
		into[pr.q][pr.e] = h
	}
}

// Forward runs every layer over one packed encoding and returns the
// final-normed hidden state at its readout rows (every <decide> and </opt>).
// The state comes from the prefix cache when it is there, and goes into it
// when it fits; a request longer than a pass runs as several.
func (g *GPU) Forward(enc *Encoding) (*Pass, error) {
	out := &Pass{Hidden: map[int][]float32{}, Batch: 1}
	key := enc.IDs[:enc.StateLen]
	resident := false
	if slot := g.cache.lookup(key); slot >= 0 {
		dur, err := g.cache.restore(g, slot, 0, 0, enc.StateLen)
		if err != nil {
			return nil, err
		}
		out.GPU += dur
		out.CacheHit, resident = true, true
	}
	passes, err := g.plan(enc, resident)
	if err != nil {
		return nil, err
	}
	for _, p := range passes {
		dur, err := g.run(p, len(g.w))
		if err != nil {
			return nil, err
		}
		out.GPU += dur
		out.Passes++
		g.readRows(p, p.readouts, true, []map[int][]float32{out.Hidden})
		if len(p.states) > 0 && g.cache.fits(enc.StateLen) {
			dur, err := g.cache.save(g, key, 0, 0)
			if err != nil {
				return nil, err
			}
			out.GPU += dur
		}
	}
	return out, nil
}

// ForwardBatch answers several requests. When they fit one pass together --
// rows, attention cells, and a state slot each -- they run as one (K7.5):
// each request's state in its own slot, a cached state restored into it,
// every branch reading its own request's state and nothing else. When they do
// not, each runs as Forward runs it. Every Pass of a batch carries the
// batch's GPU time.
func (g *GPU) ForwardBatch(encs []*Encoding) ([]*Pass, error) {
	if len(encs) == 1 {
		p, err := g.Forward(encs[0])
		return []*Pass{p}, err
	}
	resident := make([]bool, len(encs))
	slot := make([]int, len(encs))
	for i, enc := range encs {
		slot[i] = g.cache.peek(enc.IDs[:enc.StateLen])
		resident[i] = slot[i] >= 0
	}
	p, ok := g.planBatch(encs, resident)
	if !ok {
		out := make([]*Pass, len(encs))
		for i, enc := range encs {
			var err error
			if out[i], err = g.Forward(enc); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	var total time.Duration
	for i, r := range p.reqs {
		g.cache.count(resident[i])
		if resident[i] {
			dur, err := g.cache.restore(g, slot[i], r.slot, r.sLo, r.enc.StateLen)
			if err != nil {
				return nil, err
			}
			total += dur
		}
	}
	dur, err := g.run(p, len(g.w))
	if err != nil {
		return nil, err
	}
	total += dur
	hidden := make([]map[int][]float32, len(encs))
	for i := range hidden {
		hidden[i] = map[int][]float32{}
	}
	g.readRows(p, p.readouts, true, hidden)
	for _, r := range p.reqs {
		if r.withState && g.cache.fits(r.enc.StateLen) {
			dur, err := g.cache.save(g, r.enc.IDs[:r.enc.StateLen], r.slot, r.sLo)
			if err != nil {
				return nil, err
			}
			total += dur
		}
	}
	out := make([]*Pass, len(encs))
	for i := range encs {
		out[i] = &Pass{Hidden: hidden[i], GPU: total, Passes: 1, CacheHit: resident[i], Batch: len(encs)}
	}
	return out, nil
}

// ForwardRows runs the first `layers` layers of a request that fits one pass,
// with no cache, and returns the given rows of the residual stream --
// final-normed only when every layer ran. It is the stagewise seam the K4
// gate uses.
func (g *GPU) ForwardRows(enc *Encoding, rows []int, layers int) (*Pass, error) {
	p := &pass{reqs: []*preq{{enc: enc, withState: true}}}
	p.addState(0)
	for k := range enc.Decide {
		lo, hi := enc.Branch(k)
		if !p.fitsBranch(hi-lo, g.rows, g.cells) {
			return nil, fmt.Errorf("kev: question %d does not fit one pass", k)
		}
		p.addBranch(0, k)
	}
	p.n = len(p.src)
	dur, err := g.run(p, layers)
	if err != nil {
		return nil, err
	}
	out := &Pass{Hidden: map[int][]float32{}, GPU: dur, Passes: 1, Batch: 1}
	var want []prow
	for _, r := range rows {
		want = append(want, prow{0, r})
	}
	g.readRows(p, want, layers == len(g.w), []map[int][]float32{out.Hidden})
	return out, nil
}

// finalNorm is HF's Qwen3_5RMSNorm on one row, in fp32: x * rsqrt(mean(x^2)
// + eps) * (1 + w).
func (g *GPU) finalNorm(x []float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1 / math.Sqrt(float64(ss/float32(len(x))+float32(g.Cfg.RMSEps))))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * g.Norm[i]
	}
	return out
}

// headProbs runs the pointer head over one request's readouts.
func (g *GPU) headProbs(enc *Encoding, p *Pass) [][]float64 {
	out := make([][]float64, len(enc.Decide))
	for k, d := range enc.Decide {
		opts := make([][]float32, len(enc.Opts[k]))
		for j, o := range enc.Opts[k] {
			opts[j] = p.Hidden[o]
		}
		out[k] = Softmax(g.Head.Logits(p.Hidden[d], opts, g.Head.Temperature))
	}
	return out
}

// Probs runs a request and the pointer head: per question, the calibrated
// distribution over its options.
func (g *GPU) Probs(enc *Encoding) ([][]float64, *Pass, error) {
	p, err := g.Forward(enc)
	if err != nil {
		return nil, nil, err
	}
	return g.headProbs(enc, p), p, nil
}

// ProbsBatch is Probs for several requests, in one pass when they fit.
func (g *GPU) ProbsBatch(encs []*Encoding) ([][][]float64, []*Pass, error) {
	ps, err := g.ForwardBatch(encs)
	if err != nil {
		return nil, nil, err
	}
	out := make([][][]float64, len(encs))
	for i, enc := range encs {
		out[i] = g.headProbs(enc, ps[i])
	}
	return out, ps, nil
}

// Profile runs a request that fits one pass one dispatch at a time, without
// the cache, and sums the GPU time by label over the layers. It is a
// measurement of the kernels, not of a request: every dispatch pays its own
// submit.
func (g *GPU) Profile(enc *Encoding) (map[string]time.Duration, error) {
	ps, err := g.plan(enc, false)
	if err != nil {
		return nil, err
	}
	if len(ps) != 1 {
		return nil, fmt.Errorf("kev: Profile takes a request that fits one pass")
	}
	p := ps[0]
	if err := g.upload(p); err != nil {
		return nil, err
	}
	out := map[string]time.Duration{}
	for i := range g.w {
		d, labels := g.layerGraph(i, p)
		for j := range d {
			dur, err := vk.DispatchMultiTimed(d[j:j+1], 1, 1, true)
			if err != nil {
				return nil, fmt.Errorf("kev: layer %d %s: %w", i, labels[j], err)
			}
			out[labels[j]] += dur
		}
	}
	return out, nil
}
