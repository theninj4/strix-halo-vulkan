// Package mm3 is MiniMax-Music3 on the device (research/music-minimax.md,
// N-stages): caption + lyrics to a stereo song.
package mm3

// The global LM (N1): a dense Qwen3-8B with a KV cache, two rows (the CFG
// pair) for the prompt and every audio frame.
//
// It is ace/lm's 5 Hz LM (MUSIC.md A7a/A7b, A10) at another width: the same
// heads (32 query, 8 KV, 128 wide), so ace_lm_prep and ace_lm_attn run as
// built, and every other kernel takes its widths in the push block. What
// differs from ace/lm:
//
//   - hidden 4096 and FFN 12288, so the split-K tables are new (K's tile
//     count is 256 or 768: a slab count must leave a multiple of 4 tiles);
//   - the head is untied (lm_head), and only the span the frame loop reads
//     is staged: end-of-audio (151670) through the last of the 16,384
//     semantic codes, 16,389 rows of the 200,000 (134 MB, not 1.6 GB);
//   - a row's input is a token's embedding (the prompt) or a vector the
//     caller gives (a frame's feedback, the sum of its codes' embeddings);
//   - Pass returns each row's final-normed hidden, which the local LM and
//     the condition encoder read.
//
// The residual is fp32; every GEMM/GEMV operand is fp16 with fp32 sums.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The model's shape, which the shaders compile in (heads) or the checks
// pin (widths).
const (
	hidden  = 4096
	nq      = 32
	nkv     = 8
	hd      = 128
	ffn     = 12288
	qWidth  = nq * hd
	kvWidth = nkv * hd

	// HeadLo and HeadHi are the vocabulary rows the head is staged for:
	// AudioEnd, four ids upstream masks, then the semantic codes.
	HeadLo = AudioEnd
	HeadHi = CodeOffset + SemanticCodes

	// AudioEnd stops the frame loop; CodeOffset is semantic code 0's id.
	AudioEnd      = 151670
	CodeOffset    = 151675
	SemanticCodes = 16384

	tile         = 16
	gemmPad      = 128 // A-operand row pad in halves (zimage/qwen's §2.3)
	rowAlign     = 128 // the arena's rows: the widest GEMM tile's BM
	maxBankBytes = 0xfffffffc
	pushBytes    = 256 // the LLM's 64-uint block; the dit_common kernels read its first 88 bytes

	// MaxRows is the most rows a pass runs as GEMVs and returns logits for.
	MaxRows = 3
	// keyBlock is how many keys ace_lm_attn scores at once (one per lane).
	keyBlock = 64
	// partBudget caps (row, head, chunk) partials a pass writes.
	partBudget = 1 << 16
	partStride = 2 + hd

	// ffScale narrows the SwiGLU product into fp16 and the residual add
	// undoes it, as in ace/lm (Qwen3's massive activations); N1's gate
	// checks it holds at 8B.
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
	lda, gemmM, gemmN, gemmK    uint32
}

func (p gemvPush) bytes() []byte {
	var u [pushBytes / 4]uint32
	u[0], u[1], u[4], u[6] = p.resOff, p.xnOff, p.outOff, p.bOff
	u[7] = 0xffffffff // gateOff: NO_W
	u[12], u[14], u[15], u[16] = p.lda, p.gemmM, p.gemmN, p.gemmK
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

// gemvSlabs is each projection's split-K on the fp16 bank (the control):
// ace/lm's table (MUSIC.md A10) moved to the nearest rung that divides this
// model's K (40 does not divide 256 tiles). gemvSlabsQ8 is the int8 bank's,
// from N3's ladder: the int8 GEMV's k8 and k16 rungs are slow at two rows
// (k8 everywhere 63.1 ms a step, k16 70.5, k32 45.3; k1-k4 are near k32),
// so every projection takes k32.
var (
	gemvSlabs   = [nProj]int{pQ: 8, pK: 16, pV: 16, pO: 16, pGate: 4, pUp: 4, pDown: 8}
	gemvSlabsQ8 = [nProj]int{pQ: 32, pK: 32, pV: 32, pO: 32, pGate: 32, pUp: 32, pDown: 32}
)

var gemvSPIRV = map[int][]byte{
	1: shaders.LLMGEMVK1, 2: shaders.LLMGEMVK2, 4: shaders.LLMGEMVK4, 8: shaders.LLMGEMVK8,
	16: shaders.LLMGEMVK16, 20: shaders.LLMGEMVK20, 32: shaders.LLMGEMVK32, 40: shaders.LLMGEMVK40,
}

var gemvQ8SPIRV = map[int][]byte{
	1: shaders.LLMGEMVQ8K1, 2: shaders.LLMGEMVQ8K2, 4: shaders.LLMGEMVQ8K4, 8: shaders.LLMGEMVQ8K8,
	16: shaders.LLMGEMVQ8K16, 20: shaders.LLMGEMVQ8K20, 32: shaders.LLMGEMVQ8K32, 40: shaders.LLMGEMVQ8K40,
}

var gemvSumSPIRV = map[int][]byte{
	2: shaders.LLMGEMVSumK2, 4: shaders.LLMGEMVSumK4, 8: shaders.LLMGEMVSumK8,
	16: shaders.LLMGEMVSumK16, 20: shaders.LLMGEMVSumK20, 32: shaders.LLMGEMVSumK32, 40: shaders.LLMGEMVSumK40,
}

// gemmKernel is a GEMM rung over the fragment-tiled weight.
type gemmKernel struct {
	name  string
	spirv []byte
	bm    int
	bn    int
	waves int
	wave  uint32 // a pinned subgroup size; 0 is 64 for a multi-wave build
}

// gemmKernels are ace/lm's fp16 rungs and gemmFor its schedule.
var gemmKernels = []gemmKernel{
	{"reg32x128", shaders.DiTGEMMReg32x128Tiled, 32, 128, 1, 0},
	{"reg64", shaders.DiTGEMMReg64Tiled, 64, 64, 1, 0},
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

// q8Kernels are the int8 bank's rungs for a pass of many rows (Kev's
// llm_gemm.comp -DQ8B arm, as ace/lm runs it). BN is 64.
var q8Kernels = []gemmKernel{
	{"q8m2", shaders.LLMGEMMQ8M2, 32, 64, 1, 0},
	{"q8m4", shaders.LLMGEMMQ8M4, 64, 64, 1, 0},
	{"q8m8", shaders.LLMGEMMQ8M8, 128, 64, 1, 0},
}

func q8For(rows int) gemmKernel {
	if rows <= 128 {
		return q8Kernels[0]
	}
	return q8Kernels[1]
}

// Bank is the width the layers' projections are staged in. The head stays
// fp16 in either.
type Bank int

const (
	// BankQ8 is int8 in the fragment tiling with an fp16 scale per 32
	// elements of a row (ace/lm's A10 bank): 7.4 GB of layers against 13.9.
	BankQ8 Bank = iota
	// BankFP16 is halves: the control.
	BankFP16
)

func (b Bank) String() string {
	if b == BankFP16 {
		return "fp16"
	}
	return "q8"
}

const q8Group = 32

func (b Bank) projBytes(n, k int) int {
	if b == BankQ8 {
		return (n*k + n*k/q8Group*2 + 15) &^ 15
	}
	return n * k * 2
}

type layerWeights struct {
	bank int
	off  [nProj]uint32 // in the bank: halves (fp16) or bytes (int8)
	// fp32 arena
	attnNorm, ffnNorm, qkNorm uint32
	// fp16 arena: this layer's caches, [slot][pos][8][128]
	kc, vc uint32
}

// Options size the device state.
type Options struct {
	// MaxLen is the positions a slot holds: the prompt (≤ 5,000 tokens
	// upstream) and the frames (≤ 9,000).
	MaxLen int
	// Slots is how many sequences hold a cache: 2 for the CFG pair.
	Slots int
	// Rows is the longest pass; a longer prefill runs as several.
	Rows int
	// Bank is the layers' weight width: int8 (the zero value) or fp16.
	Bank Bank
	// allSlabs builds every split-K rung, for the ladder (TestSlabLadder).
	allSlabs bool
}

// DefaultOptions is a CFG pair's cache at 13,312 positions, 3.9 GB: a
// 6-minute song (9,000 frames) after a prompt of up to 4,311 tokens. The
// model was trained to 10,240 positions (max_position_embeddings); upstream
// allows 5,000 + 9,000 and so runs past it too. 14,000 would pass one
// binding's 4 GB.
func DefaultOptions() Options { return Options{MaxLen: 13312, Slots: 2, Rows: 512} }

// LM is the global LM, resident.
type LM struct {
	Cfg *qwen.Config
	o   Options

	dev   *vk.Device
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	banks []*vk.Buffer
	head  int // the bank holding the head span, [headRows][4096] tiled
	pipes map[string]*vk.ComputePipeline
	gemms []map[string]*vk.ComputePipeline
	gemvs []map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	embed    []uint16 // bf16 [vocab][4096], gathered on the host
	vocab    int
	headRows int // HeadHi-HeadLo padded to a tile

	w         []layerWeights
	finalNorm uint32
	rope      uint32 // cos [MaxLen][64], then sin

	arenaRows int
	// fp32 arena
	aX, aQ, aK, aV, aO, aGate, aUp, aFF, aMeta, aPart, aRes, aLogits uint32
	aUnscale                                                         uint32
	actElems                                                         int
	// fp16 arena
	hA, hQ, hCtx, hFFN uint32
	hElems             int
	ldaDim, ldaQ, ldaF int

	// LayersPerSubmit is how many layers of a prefill go in one command
	// buffer; a decode step is always one.
	LayersPerSubmit int
	// slabOverride replaces a projection's split-K (the ladder); 0 keeps
	// the table's. Only rungs built at load (allSlabs) can be chosen.
	slabOverride [nProj]int
}

func (g *LM) slabs(p proj) int {
	if s := g.slabOverride[p]; s != 0 {
		return s
	}
	if g.o.Bank == BankQ8 {
		return gemvSlabsQ8[p]
	}
	return gemvSlabs[p]
}

// LoadConfig reads the LM's config.json. transformers 5 writes the rope
// base under rope_parameters, which qwen.Config does not read.
func LoadConfig(dir string) (*qwen.Config, error) {
	cfg, err := qwen.LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.RopeTheta == 0 {
		buf, err := os.ReadFile(filepath.Join(dir, "config.json"))
		if err != nil {
			return nil, err
		}
		var c struct {
			Rope struct {
				Theta float64 `json:"rope_theta"`
				Type  string  `json:"rope_type"`
			} `json:"rope_parameters"`
		}
		if err := json.Unmarshal(buf, &c); err != nil {
			return nil, err
		}
		if c.Rope.Type != "" && c.Rope.Type != "default" {
			return nil, fmt.Errorf("mm3: rope_type %q; only the default is built", c.Rope.Type)
		}
		cfg.RopeTheta = c.Rope.Theta
	}
	if cfg.RopeTheta == 0 {
		return nil, fmt.Errorf("mm3: %s/config.json has no rope_theta", dir)
	}
	return cfg, nil
}

// LoadLM stages the global LM from its checkpoint directory
// (models/MiniMax-Music3/language_model).
func LoadLM(dev *vk.Device, dir string, o Options) (*LM, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.HiddenSize != hidden || cfg.NumHeads != nq || cfg.NumKVHeads != nkv || cfg.HeadDim != hd ||
		cfg.IntermediateSize != ffn {
		return nil, fmt.Errorf("mm3: %+v is not the shape the kernels are built for", *cfg)
	}
	if o.MaxLen <= 0 || o.Slots <= 0 || o.Rows <= 0 {
		return nil, fmt.Errorf("mm3: LM options %+v", o)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	g := &LM{
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

func (g *LM) layout(set *safetensors.Set) error {
	c := g.Cfg
	emb, err := set.Get(c.Prefix + "embed_tokens.weight")
	if err != nil {
		return err
	}
	if len(emb.Shape) != 2 || emb.Shape[1] != hidden || emb.DType != safetensors.BF16 {
		return fmt.Errorf("mm3: embed_tokens is %v %s, want [vocab 4096] bf16", emb.Shape, emb.DType)
	}
	g.vocab = emb.Shape[0]
	if g.vocab < HeadHi {
		return fmt.Errorf("mm3: a vocabulary of %d has no semantic codes", g.vocab)
	}
	g.headRows = roundUp(HeadHi-HeadLo, tile)

	bk := g.o.Bank
	perLayer := 0
	for _, s := range projShape {
		perLayer += bk.projBytes(s[0], s[1])
	}
	unit := 2 // what an offset counts: halves on fp16, bytes on int8
	if bk == BankQ8 {
		unit = 1
	}
	var bankBytes []int
	g.w = make([]layerWeights, c.NumLayers)
	for i := range g.w {
		if len(bankBytes) == 0 || bankBytes[len(bankBytes)-1]+perLayer > maxBankBytes {
			bankBytes = append(bankBytes, 0)
		}
		b := len(bankBytes) - 1
		g.w[i].bank = b
		for p, s := range projShape {
			g.w[i].off[p] = uint32(bankBytes[b] / unit)
			bankBytes[b] += bk.projBytes(s[0], s[1])
		}
	}
	g.head = len(bankBytes)
	bankBytes = append(bankBytes, g.headRows*hidden*2)
	for i, n := range bankBytes {
		b, err := g.dev.NewBuffer(n)
		if err != nil {
			return fmt.Errorf("mm3: bank %d (%d MB): %w", i, n>>20, err)
		}
		g.banks = append(g.banks, b)
	}

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

	// fp32 activations, host-cached: the host reads the logits every frame.
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
	g.aRes = alloc(MaxRows * 40 * ffn) // 40: the widest split-K rung
	g.aLogits = alloc(MaxRows * g.headRows)
	g.aUnscale = alloc(hidden)
	if g.abuf, err = g.dev.NewHostCachedBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("mm3: fp32 arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	unscale := make([]float32, hidden)
	for i := range unscale {
		unscale[i] = 1 / ffScale
	}
	g.abuf.WriteFloat32At(int(g.aUnscale), unscale)

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
		return fmt.Errorf("mm3: the fp16 arena is %d MB, past one binding's 4 GB; lower MaxLen or Slots", (g.hElems*2)>>20)
	}
	if g.hbuf, err = g.dev.NewHostCachedBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("mm3: fp16 arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the GEMMs read pad rows and pad columns.
	g.hbuf.Zero()
	return nil
}

func (g *LM) pipeline(name string, spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, fmt.Errorf("mm3: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return nil, fmt.Errorf("mm3: pipeline %s: %w", name, err)
	}
	return pipe, nil
}

func (g *LM) build() error {
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
	g.gemms = make([]map[string]*vk.ComputePipeline, len(g.banks))
	g.gemvs = make([]map[string]*vk.ComputePipeline, len(g.banks))
	for b, bank := range g.banks {
		g.gemms[b] = map[string]*vk.ComputePipeline{}
		g.gemvs[b] = map[string]*vk.ComputePipeline{}
		bb := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank}
		q8 := b != g.head && g.o.Bank == BankQ8
		gemms, gemvs, table := gemmKernels, gemvSPIRV, gemvSlabs
		if q8 {
			// llm_common.glsl's bindings: the scale plane is read as halves
			// at 3 and the tiles as words at 5; 4 is declared and unused.
			bb = []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, bank, g.abuf, bank}
			gemms, gemvs, table = q8Kernels, gemvQ8SPIRV, gemvSlabsQ8
		}
		if b != g.head {
			for _, k := range gemms {
				spec := vk.PipelineSpec{Buffers: bb, PushConstantSize: pc}
				if k.wave != 0 && !q8 {
					spec.RequiredSubgroupSize = k.wave
				} else if k.waves > 1 || q8 {
					spec.RequiredSubgroupSize = 64
				}
				p, err := g.pipeline("gemm "+k.name, k.spirv, spec)
				if err != nil {
					return err
				}
				g.gemms[b][k.name] = p
			}
		}
		used := map[int]bool{1: true}
		if b != g.head {
			for _, s := range table {
				used[s] = true
			}
			if g.o.allSlabs {
				for s := range gemvs {
					used[s] = true
				}
			}
		}
		for slabs := range used {
			for r := 1; r <= MaxRows; r++ {
				p, err := g.pipeline("gemv", gemvs[slabs], vk.PipelineSpec{Buffers: bb, PushConstantSize: pc,
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

func (g *LM) stage(set *safetensors.Set) error {
	c := g.Cfg
	emb, err := set.Get(c.Prefix + "embed_tokens.weight")
	if err != nil {
		return err
	}
	g.embed = make([]uint16, g.vocab*hidden)
	for i := range g.embed {
		g.embed[i] = binary.LittleEndian.Uint16(emb.Data[2*i:])
	}

	// The head's span, tiled; the pad rows stay zero.
	lm, err := set.Get("lm_head.weight")
	if err != nil {
		return err
	}
	if len(lm.Shape) != 2 || lm.Shape[0] != g.vocab || lm.Shape[1] != hidden || lm.DType != safetensors.BF16 {
		return fmt.Errorf("mm3: lm_head is %v %s, want [%d 4096] bf16", lm.Shape, lm.DType, g.vocab)
	}
	head := make([]uint16, g.headRows*hidden)
	tileB(head, HeadHi-HeadLo, hidden, func(i int, buf []float32) {
		row := lm.Data[2*(HeadLo+i)*hidden:]
		for j := range buf {
			buf[j] = bf16(binary.LittleEndian.Uint16(row[2*j:]))
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
				return fmt.Errorf("mm3: layer %d %s is [%d %d]", i, projNames[p], lin.Out, lin.In)
			}
			if g.o.Bank == BankQ8 {
				q, sc := tileBQ8(lin.Weight, n, k)
				g.banks[w.bank].WriteBytesAt(int(w.off[p]), q)
				g.banks[w.bank].WriteUint16At((int(w.off[p])+len(q))/2, sc)
				continue
			}
			buf := make([]uint16, n*k)
			tileB(buf, n, k, func(r int, dst []float32) { copy(dst, lin.Weight[r*k:(r+1)*k]) })
			g.banks[w.bank].WriteUint16At(int(w.off[p]), buf)
		}
	}
	return nil
}

// tileBQ8 quantises a [n, k] row-major matrix into the int8 bank's layout
// (ace/lm's): tile (nt, kt) is 256 bytes holding (k, n) at (n%16)*16 + k%16,
// tiles kt-fastest; then the scale plane, scale (nt, group g, n%16) at
// (nt*kgroups + g)*16 + n%16. q is rounded against the fp16 scale the kernel
// multiplies by.
func tileBQ8(w []float32, n, k int) ([]byte, []uint16) {
	kt, kg := k/tile, k/q8Group
	q := make([]byte, n*k)
	sc := make([]uint16, n*kg)
	workers := runtime.GOMAXPROCS(0)
	chunk := (n + workers - 1) / workers
	done := make(chan struct{}, workers)
	for wk := 0; wk < workers; wk++ {
		go func(lo, hi int) {
			for r := lo; r < hi; r++ {
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
			}
			done <- struct{}{}
		}(wk*chunk, min(n, (wk+1)*chunk))
	}
	for wk := 0; wk < workers; wk++ {
		<-done
	}
	return q, sc
}

func bf16(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// Destroy releases every Vulkan object.
func (g *LM) Destroy() {
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

// Row is one row of a pass: the slot whose cache it extends, its position
// there, and its input: Vec when set (a frame's feedback, 4096 values),
// else token ID's embedding.
type Row struct {
	ID        int32
	Vec       []float32
	Slot, Pos int
}

// Embed returns token id's embedding row, as fp32.
func (g *LM) Embed(id int) []float32 {
	out := make([]float32, hidden)
	for j, v := range g.embed[id*hidden : (id+1)*hidden] {
		out[j] = bf16(v)
	}
	return out
}

// Out is what a pass of at most MaxRows rows returns per row: the head's
// logits over [HeadLo, HeadHi) and the final-normed hidden (fp16 on the
// device, widened).
type Out struct {
	Logits [][]float32
	Hidden [][]float32
}

// Pass runs every layer over rows. With head set (at most MaxRows rows) it
// also runs the final norm and the head, and returns each row's logits and
// hidden.
func (g *LM) Pass(rows []Row, head bool) (*Out, time.Duration, error) {
	n := len(rows)
	if head && n > MaxRows {
		return nil, 0, fmt.Errorf("mm3: logits for %d rows; at most %d", n, MaxRows)
	}
	var tail []vk.MultiDispatch
	if head {
		tail = g.headGraph(n)
	}
	total, err := g.run(rows, tail)
	if err != nil || !head {
		return nil, total, err
	}
	out := &Out{Logits: make([][]float32, n), Hidden: make([][]float32, n)}
	for r := range rows {
		out.Logits[r] = g.abuf.ReadFloat32At(int(g.aLogits)+r*g.headRows, HeadHi-HeadLo)
		h := g.hbuf.ReadUint16At(int(g.hA)+r*g.ldaDim, hidden)
		out.Hidden[r] = make([]float32, hidden)
		for j, v := range h {
			out.Hidden[r][j] = safetensors.F16ToF32(v)
		}
	}
	return out, total, nil
}

// run uploads rows and runs every layer over them, with tail recorded after
// the last layer in the same submit.
func (g *LM) run(rows []Row, tail []vk.MultiDispatch) (time.Duration, error) {
	n := len(rows)
	if n == 0 || n > g.o.Rows {
		return 0, fmt.Errorf("mm3: a pass of %d rows; 1 to %d", n, g.o.Rows)
	}
	x := make([]float32, n*hidden)
	meta := make([]uint32, 2*n)
	maxKeys := 0
	for r, t := range rows {
		if t.Slot < 0 || t.Slot >= g.o.Slots || t.Pos < 0 || t.Pos >= g.o.MaxLen {
			return 0, fmt.Errorf("mm3: row %d at slot %d position %d; the cache is %d x %d", r, t.Slot, t.Pos, g.o.Slots, g.o.MaxLen)
		}
		dst := x[r*hidden : (r+1)*hidden]
		switch {
		case t.Vec != nil:
			if len(t.Vec) != hidden {
				return 0, fmt.Errorf("mm3: row %d's input is %d wide", r, len(t.Vec))
			}
			copy(dst, t.Vec)
		case t.ID >= 0 && int(t.ID) < g.vocab:
			for j, v := range g.embed[int(t.ID)*hidden : (int(t.ID)+1)*hidden] {
				dst[j] = bf16(v)
			}
		default:
			return 0, fmt.Errorf("mm3: token %d outside the vocabulary", t.ID)
		}
		meta[2*r], meta[2*r+1] = uint32(t.Slot), uint32(t.Pos)
		maxKeys = max(maxKeys, t.Pos+1)
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.WriteUint32At(int(g.aMeta), meta)

	chunks := (maxKeys + keyBlock - 1) / keyBlock
	if lim := partBudget / (n * nq); chunks > lim {
		chunks = max(lim, 1)
	}
	chunkLen := roundUp((maxKeys+chunks-1)/chunks, keyBlock)
	chunks = (maxKeys + chunkLen - 1) / chunkLen

	var total time.Duration
	per := max(g.LayersPerSubmit, 1)
	if n <= MaxRows {
		per = len(g.w) // a decode step is one submit (ace/lm's TestSubmitSplit)
	}
	for i := 0; i < len(g.w); i += per {
		var d []vk.MultiDispatch
		for j := i; j < min(i+per, len(g.w)); j++ {
			d = append(d, g.layerGraph(j, n, chunks, chunkLen)...)
		}
		if i+per >= len(g.w) {
			d = append(d, tail...)
		}
		dur, err := vk.DispatchMultiTimed(d, 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("mm3: layers %d..: %w", i, err)
		}
		total += dur
	}
	return total, nil
}

// gemv is one projection at n ≤ MaxRows rows: split-K partials and their
// sum, or one dispatch at a single slab.
func (g *LM) gemv(bank int, slabs int, aOff uint32, lda int, cOff, bOff uint32, nOut, k, n int) []vk.MultiDispatch {
	kt := k / tile
	if kt%slabs != 0 || (kt/slabs)%4 != 0 {
		panic(fmt.Sprintf("mm3: %d slabs do not split K %d", slabs, k))
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
	eps := math.Float32bits(float32(g.Cfg.RMSEps))
	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	norm := func(wOff uint32) {
		add("normf16", uint32(n), 1, push{InOff: g.aX, OutOff: g.hA, WOff: wOff, Tokens: uint32(n), Dim: hidden,
			LDA: uint32(g.ldaDim), Eps: eps})
	}
	q8 := g.o.Bank == BankQ8
	tokPad := 0
	var gk gemmKernel
	if n > MaxRows {
		gk = gemmFor(n)
		if q8 {
			gk = q8For(n)
		}
		tokPad = roundUp(n, gk.bm)
	}
	proj := func(p proj, aOff uint32, lda int, cOff uint32) {
		nOut, k := projShape[p][0], projShape[p][1]
		if n <= MaxRows {
			d = append(d, g.gemv(w.bank, g.slabs(p), aOff, lda, cOff, w.off[p], nOut, k, n)...)
			return
		}
		if q8 {
			pc := gemvPush{xnOff: aOff, outOff: cOff, bOff: w.off[p], lda: uint32(lda),
				gemmM: uint32(tokPad), gemmN: uint32(nOut), gemmK: uint32(k)}
			d = append(d, vk.MultiDispatch{Pipeline: g.gemms[w.bank][gk.name], GroupsX: uint32(nOut / gk.bn),
				GroupsY: uint32(tokPad / gk.bm), PushConstants: pc.bytes()})
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

// headGraph is the final norm (into hA, which Pass reads back as the
// hidden) and the head over its whole span, into the logits at a row
// stride of headRows.
func (g *LM) headGraph(n int) []vk.MultiDispatch {
	eps := math.Float32bits(float32(g.Cfg.RMSEps))
	d := []vk.MultiDispatch{{Pipeline: g.pipes["normf16"], GroupsX: uint32(n), GroupsY: 1,
		PushConstants: push{InOff: g.aX, OutOff: g.hA, WOff: g.finalNorm, Tokens: uint32(n), Dim: hidden,
			LDA: uint32(g.ldaDim), Eps: eps}.bytes()}}
	return append(d, g.gemv(g.head, 1, g.hA, g.ldaDim, g.aLogits, 0, g.headRows, hidden, n)...)
}
