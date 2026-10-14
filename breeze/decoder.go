package breeze

// The backbone and the depth decoder on the device (B3, B4).
//
// Both are ace/lm's decoder (MUSIC.md A7) at other widths: projections
// staged once in the §2.8 fragment tiling -- int8 by default, priced
// against upstream's bf16 (B7) -- which llm_gemm's rungs read for a pass of
// many rows and llm_gemv's split-K GEMV reads at 1-3 rows; ace_lm_attn
// rebuilt at each stack's heads. A layer is four projections (q|k|v and
// gate|up fused) and breeze_dec.comp's prep, SwiGLU and residual-join-plus-
// next-norm between them: decode is dispatch latency, so B7 counted them
// (22 -> 13 a layer). A pass is any set of rows, each a position of some
// slot's KV cache, so a prompt prefills in passes and the CFG pair's two
// rows run as one.
//
// The two stacks share one set of buffers -- one weight arena, one fp32 and
// one fp16 activation arena, the banks -- so a frame can be recorded as one
// submit: the backbone's normed hidden state is the depth decoder's first
// input without leaving the device (not yet done: B7's handoff).
//
// What is this model's own:
//
//  1. **A row is an embedding, not a token.** The backbone's prompt is the
//     text encoder's projected states and summed audio-code embeddings, and
//     a decode step's input is the sum of the last frame's 16 code rows; the
//     depth decoder's rows are the backbone's hidden state and one code
//     row each, through an input projector (2048 -> 1024).
//  2. **The depth decoder's heads are per position**: codebook k's logits
//     are W[k] (1024 x 2051) of the row at position k+1.
//  3. **Two ropes**: the backbone's plain theta 1e6 (backbone_config; the
//     top-level llama3 rope_scaling is the unused BreezeRotaryEmbedding's),
//     the depth decoder's llama3-scaled theta 5e5 over 16 positions.
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
)

const (
	tile         = 16
	gemmPad      = 128 // A-operand row pad in halves (zimage/qwen's §2.3)
	rowAlign     = 128 // the arena's rows: the widest GEMM tile's BM
	maxBankBytes = 0xfffffffc
	pushBytes    = 256

	hd = 128 // both stacks' head dim

	// MaxRows is llm_gemv's MAXROWS: a pass of at most this many rows runs
	// its projections as GEMVs.
	MaxRows = 3
	// keyBlock is how many keys ace_lm_attn scores at once (one per lane).
	keyBlock   = 64
	partBudget = 1 << 16
	partStride = 2 + hd

	// Vocab is a codebook's logits: 2048 codes and three reserved ids (2050
	// is the pad); the backbone's head has one more, EOS.
	Vocab       = 2051
	CodebookEOS = Vocab // the backbone's EOS logit
	Codes       = 2048
	Codebooks   = 16

	bbHidden = 2048
	ddHidden = 1024

	// ffScale narrows the SwiGLU product into fp16 and the residual add
	// undoes it (ace/lm's; an exact power of two, so it costs no bits).
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

// gemvPush is the fields of llm_common.glsl's block llm_gemv.comp reads.
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

type proj int

// A layer's projections, fused where they share an input (B7): q|k|v as
// one, gate|up as one (gate's rows, then up's).
const (
	pQKV proj = iota
	pO
	pGU
	pDown
	nProj
)

// projParts are the checkpoint's tensors each projection stacks.
var projParts = [nProj][]string{
	pQKV:  {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"},
	pO:    {"self_attn.o_proj"},
	pGU:   {"mlp.gate_proj", "mlp.up_proj"},
	pDown: {"mlp.down_proj"},
}

var gemvSPIRV = map[int][]byte{
	1: shaders.LLMGEMVK1, 2: shaders.LLMGEMVK2, 4: shaders.LLMGEMVK4, 8: shaders.LLMGEMVK8,
	16: shaders.LLMGEMVK16, 32: shaders.LLMGEMVK32,
}

var gemvSumSPIRV = map[int][]byte{
	2: shaders.LLMGEMVSumK2, 4: shaders.LLMGEMVSumK4, 8: shaders.LLMGEMVSumK8,
	16: shaders.LLMGEMVSumK16, 32: shaders.LLMGEMVSumK32,
}

// gemmKernel is a DiT GEMM rung over the fragment-tiled weight.
type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	waves  int
	wave   uint32
}

var gemmKernels = []gemmKernel{
	{"reg32x128", shaders.DiTGEMMReg32x128Tiled, 32, 128, 1, 0},
	{"reg64", shaders.DiTGEMMReg64Tiled, 64, 64, 1, 0},
	{"wg128x256", shaders.DiTGEMMWG128x256LDSW32, 128, 256, 8, 32},
}

// gemmFor picks a rung whose BN divides n.
func gemmFor(rows, n int) gemmKernel {
	k := gemmKernels[2]
	switch {
	case rows <= 96:
		k = gemmKernels[0]
	case rows <= 192:
		k = gemmKernels[1]
	}
	for n%k.bn != 0 {
		k = gemmKernels[1] // 64 divides every width here
	}
	return k
}

// shape is one stack's widths.
type shape struct {
	name     string
	prefix   string // the layers' tensor prefix, "...layers."
	hidden   int
	nq, nkv  int
	ffn      int
	layers   int
	eps      float64
	qkNorm   bool
	maxLen   int // positions a slot holds
	slots    int
	slabs    [nProj]int // split-K at 1-3 rows
	prepF    []byte     // the q|k|v prep over the fused projection (breeze_dec.comp)
	attn     []byte
	combine  []byte
	ropeCos  func(maxLen int) (cos, sin []float32)
	normName string // the final norm's tensor
}

func (s *shape) projShape(p proj) (out, in int) {
	q, kv := s.nq*hd, s.nkv*hd
	switch p {
	case pQKV:
		return q + 2*kv, s.hidden
	case pO:
		return s.hidden, q
	case pGU:
		return 2 * s.ffn, s.hidden
	}
	return s.hidden, s.ffn
}

type layerWeights struct {
	off                       [nProj]uint32 // in the layer bank: halves (fp16) or bytes (int8)
	attnNorm, ffnNorm, qkNorm uint32        // weight arena
	kc, vc                    uint32        // fp16 arena: [slot][pos][nkv][128]
}

// stack is one transformer's state in the shared buffers.
type stack struct {
	shape
	w         []layerWeights
	finalNorm uint32
	rope      uint32
	rows      int // the longest pass
	// fp32 arena
	aX, aQKV, aO, aGU, aFF, aMeta, aPart, aRes uint32
	// fp16 arena
	hA, hQ, hCtx, hFFN     uint32
	ldaDim, ldaQ, ldaF     int
	pPrep, pAttn, pCombine *vk.ComputePipeline
}

// Options size the device state.
type Options struct {
	// MaxLen is the positions a backbone slot holds: the prompt (reference
	// frames included) and the frames generated. Upstream's is 2048.
	MaxLen int
	// Rows is the longest backbone pass; a longer prefill runs as several.
	Rows int
	// Bank is the layers' weight width: int8 (the zero value) or fp16. The
	// heads and the depth decoder's input projector stay fp16 either way.
	Bank Bank
}

// Bank is the width the stacks' projections are staged in (B7).
type Bank int

const (
	// BankQ8 is int8 in the fragment tiling with an fp16 scale per 32
	// elements of a row: the LLM's L8a bank as ace/lm stages it (MUSIC.md
	// A10), read by llm_gemv.comp and llm_gemm.comp -DQ8B.
	BankQ8 Bank = iota
	// BankFP16 is halves, what B3-B5 gated: the control.
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

// projBytes is a [n, k] projection's staged size, rounded so the next one
// starts on 16 bytes.
func (b Bank) projBytes(n, k int) int {
	if b == BankQ8 {
		return (n*k + n*k/q8Group*2 + 15) &^ 15
	}
	return n * k * 2
}

var gemvQ8SPIRV = map[int][]byte{
	1: shaders.LLMGEMVQ8K1, 2: shaders.LLMGEMVQ8K2, 4: shaders.LLMGEMVQ8K4, 8: shaders.LLMGEMVQ8K8,
	16: shaders.LLMGEMVQ8K16, 32: shaders.LLMGEMVQ8K32,
}

// q8Kernels are the int8 bank's GEMM rungs for a pass of many rows (BN 64).
var q8Kernels = []gemmKernel{
	{"q8m2", shaders.LLMGEMMQ8M2, 32, 64, 1, 0},
	{"q8m4", shaders.LLMGEMMQ8M4, 64, 64, 1, 0},
}

// The two banks: the stacks' projections, and the fp16 heads.
const (
	layerBank = 0
	headBank  = 1
)

// DefaultOptions is upstream's bound: 2048 positions, two slots for CFG.
func DefaultOptions() Options { return Options{MaxLen: 2048, Rows: 512} }

// Decoder is the backbone, the depth decoder and their heads, resident.
type Decoder struct {
	o Options

	dev   *vk.Device
	wbuf  *vk.Buffer    // fp32: norms, rope tables
	abuf  *vk.Buffer    // fp32 activations, host-cached
	hbuf  *vk.Buffer    // fp16 activations and the KV caches
	banks [2]*vk.Buffer // the projections (int8 or fp16), and the fp16 heads
	pipes map[string]*vk.ComputePipeline
	// per bank: GEMM rungs by name and GEMVs by "k<slabs>_r<rows>"
	gemms, gemvs [2]map[string]*vk.ComputePipeline
	mods         []*vk.ShaderModule

	bb, dd *stack

	// Embed is the shared code embedding, bf16 [16*2051][2048], on the host.
	Embed []uint16

	// Heads, in the bank: the backbone's lm_head [2052->2064][2048], the
	// depth decoder's input projector [1024][2048] and its 15 codebook heads
	// [15][2051->2064][1024] (transposed from the checkpoint's [in][out]).
	lmHead, projIn, ddHeads uint32
	headPad                 int
	aLogits                 uint32 // [rows][headPad]
	hIn                     uint32 // fp16 depth decoder input rows, [rows][2048+pad]
	ldaIn                   int

	actElems, hElems, w32 int
}

// Load stages the decoder half of the checkpoint in dir.
func Load(dev *vk.Device, dir string, o Options) (*Decoder, error) {
	if o.MaxLen <= 0 || o.Rows <= 0 {
		return nil, fmt.Errorf("breeze: options %+v", o)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	d := &Decoder{o: o, dev: dev, pipes: map[string]*vk.ComputePipeline{}}
	d.bb = &stack{shape: shape{
		name: "backbone", prefix: "backbone_model.layers.", hidden: bbHidden, nq: 16, nkv: 8, ffn: 6144,
		layers: 28, eps: 1e-6, qkNorm: true, maxLen: o.MaxLen, slots: 2,
		// TestSlabLadder at two rows on the int8 bank (B7): every projection
		// is fastest at 32 slabs, 204-230 GB/s, where the first table's
		// k4/k8 read 77-117.
		slabs: [nProj]int{pQKV: 32, pO: 32, pGU: 32, pDown: 32},
		prepF: shaders.BreezeDecPrepBB,
		attn:  shaders.BreezeBBAttn, combine: shaders.BreezeBBAttnCombine,
		ropeCos:  func(n int) ([]float32, []float32) { return ropeTable(n, 1e6, nil) },
		normName: "backbone_model.norm.weight",
	}, rows: o.Rows}
	d.dd = &stack{shape: shape{
		name: "depth", prefix: "depth_decoder.model.layers.", hidden: ddHidden, nq: 8, nkv: 2, ffn: 8192,
		layers: 12, eps: 1e-5, maxLen: Codebooks, slots: 2,
		// The ladder again: 16 slabs except the down projection (K 8192),
		// whose 4 reads 158 GB/s to 16's 96.
		slabs: [nProj]int{pQKV: 16, pO: 16, pGU: 16, pDown: 4},
		prepF: shaders.BreezeDecPrepDD,
		attn:  shaders.BreezeDDAttn, combine: shaders.BreezeDDAttnCombine,
		ropeCos:  func(n int) ([]float32, []float32) { return ropeTable(n, 5e5, llama3Depth) },
		normName: "depth_decoder.model.norm.weight",
	}, rows: Codebooks * 2}
	if err := d.layout(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.build(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.stage(set); err != nil {
		d.Destroy()
		return nil, err
	}
	return d, nil
}

// llama3Depth is the depth decoder's rope_scaling: factor 32, original 16
// positions, low/high frequency factors 1/512 and 1/128.
var llama3Depth = &llama3{factor: 32, low: 0.001953125, high: 0.0078125, orig: 16}

type llama3 struct{ factor, low, high, orig float64 }

// ropeTable is cos [n][64] and sin [n][64] for NeoX pairing, with HF's
// llama3 inv_freq adjustment when l is set (_compute_llama3_parameters),
// computed in fp32 as torch does.
func ropeTable(n int, theta float64, l *llama3) (cos, sin []float32) {
	inv := make([]float32, hd/2)
	for i := range inv {
		inv[i] = float32(1 / math.Pow(theta, float64(2*i)/hd))
	}
	if l != nil {
		lowWave, highWave := l.orig/l.low, l.orig/l.high
		for i, f := range inv {
			wave := 2 * math.Pi / float64(f)
			switch {
			case wave > lowWave:
				inv[i] = f / float32(l.factor)
			case wave >= highWave:
				smooth := (l.orig/wave - l.low) / (l.high - l.low)
				inv[i] = float32((1-smooth)*float64(f)/l.factor + smooth*float64(f))
			}
		}
	}
	cos, sin = make([]float32, n*hd/2), make([]float32, n*hd/2)
	for p := 0; p < n; p++ {
		for i, f := range inv {
			a := float32(p) * f
			cos[p*hd/2+i] = float32(math.Cos(float64(a)))
			sin[p*hd/2+i] = float32(math.Sin(float64(a)))
		}
	}
	return cos, sin
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

func (d *Decoder) alloc(n int) uint32 {
	off := uint32(d.actElems)
	d.actElems += (n + 63) &^ 63
	return off
}

func (d *Decoder) halloc(n int) uint32 {
	off := uint32(d.hElems)
	d.hElems += (n + 63) &^ 63
	return off
}

func (d *Decoder) walloc(n int) uint32 {
	off := uint32(d.w32)
	d.w32 += (n + 63) &^ 63
	return off
}

func (d *Decoder) layout() error {
	d.headPad = roundUp(Vocab+1, 128) // a GEMM rung's BN divides it
	bk := d.o.Bank
	unit := 2 // what a layer-bank offset counts: halves on fp16, bytes on int8
	if bk == BankQ8 {
		unit = 1
	}
	lbytes := 0
	bank := 0 // the head bank, halves
	balloc := func(n int) uint32 {
		off := uint32(bank)
		bank += roundUp(n, 64)
		return off
	}
	for _, s := range []*stack{d.bb, d.dd} {
		s.w = make([]layerWeights, s.layers)
		for i := range s.w {
			for p := proj(0); p < nProj; p++ {
				n, k := s.projShape(p)
				s.w[i].off[p] = uint32(lbytes / unit)
				lbytes += bk.projBytes(n, k)
			}
			s.w[i].attnNorm = d.walloc(s.hidden)
			s.w[i].ffnNorm = d.walloc(s.hidden)
			s.w[i].qkNorm = d.walloc(2 * hd)
		}
		s.finalNorm = d.walloc(s.hidden)
		s.rope = d.walloc(s.maxLen * hd)
	}
	d.lmHead = balloc(d.headPad * bbHidden)
	d.projIn = balloc(ddHidden * bbHidden)
	d.ddHeads = balloc((Codebooks - 1) * d.headPad * ddHidden)
	if lbytes > maxBankBytes {
		return fmt.Errorf("breeze: the layer bank is %d MB, past one binding", lbytes>>20)
	}
	var err error
	if d.banks[layerBank], err = d.dev.NewBuffer(lbytes); err != nil {
		return fmt.Errorf("breeze: layer bank (%d MB): %w", lbytes>>20, err)
	}
	if d.banks[headBank], err = d.dev.NewBuffer(bank * 2); err != nil {
		return fmt.Errorf("breeze: head bank (%d MB): %w", bank*2>>20, err)
	}
	if d.wbuf, err = d.dev.NewBuffer(d.w32 * 4); err != nil {
		return err
	}

	for _, s := range []*stack{d.bb, d.dd} {
		rows := roundUp(s.rows, rowAlign)
		q, kv := s.nq*hd, s.nkv*hd
		s.ldaDim, s.ldaQ, s.ldaF = s.hidden+gemmPad, q+gemmPad, s.ffn+gemmPad
		s.aX = d.alloc(rows * s.hidden)
		s.aQKV = d.alloc(rows * (q + 2*kv))
		s.aO = d.alloc(rows * s.hidden)
		s.aGU = d.alloc(rows * 2 * s.ffn)
		s.aFF = d.alloc(rows * s.hidden)
		s.aMeta = d.alloc(rows * 2)
		s.aPart = d.alloc(partBudget * partStride)
		s.aRes = d.alloc(MaxRows * 32 * max(s.ffn, s.hidden))
		s.hA = d.halloc(rows * s.ldaDim)
		s.hQ = d.halloc(rows * q)
		s.hCtx = d.halloc(rows * s.ldaQ)
		s.hFFN = d.halloc(rows * s.ldaF)
		cache := s.slots * s.maxLen * kv
		for i := range s.w {
			s.w[i].kc = d.halloc(cache)
			s.w[i].vc = d.halloc(cache)
		}
	}
	d.aLogits = d.alloc(roundUp(max(d.bb.rows, d.dd.rows), rowAlign) * d.headPad)
	d.ldaIn = bbHidden + gemmPad
	d.hIn = d.halloc(roundUp(d.dd.rows, rowAlign) * d.ldaIn)
	if d.hElems*2 > maxBankBytes {
		return fmt.Errorf("breeze: the fp16 arena is %d MB, past one binding", d.hElems*2>>20)
	}
	if d.abuf, err = d.dev.NewHostCachedBuffer(d.actElems * 4); err != nil {
		return fmt.Errorf("breeze: fp32 arena (%d MB): %w", d.actElems*4>>20, err)
	}
	if d.hbuf, err = d.dev.NewBuffer(d.hElems * 2); err != nil {
		return fmt.Errorf("breeze: fp16 arena (%d MB): %w", d.hElems*2>>20, err)
	}
	d.hbuf.Zero() // the GEMMs read pad rows and pad columns
	return nil
}

func (d *Decoder) pipeline(name string, spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
	mod, err := d.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, fmt.Errorf("breeze: shader %s: %w", name, err)
	}
	d.mods = append(d.mods, mod)
	p, err := d.dev.NewPipeline(mod, spec)
	if err != nil {
		return nil, fmt.Errorf("breeze: pipeline %s: %w", name, err)
	}
	return p, nil
}

func (d *Decoder) build() error {
	pc := uint32(pushBytes)
	bufs := []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.banks[headBank]}
	add := func(key, name string, spirv []byte, spec vk.PipelineSpec) error {
		spec.Buffers, spec.PushConstantSize = bufs, pc
		p, err := d.pipeline(name, spirv, spec)
		if err != nil {
			return err
		}
		d.pipes[key] = p
		return nil
	}
	for _, s := range []struct {
		name  string
		spirv []byte
		wave  uint32
	}{
		{"normf16", shaders.DiTNormScaleF16, 0},
		{"swiglu", shaders.BreezeDecSwiGLU, 0},
		{"addnorm", shaders.BreezeDecAddNorm, 0},
	} {
		if err := add(s.name, s.name, s.spirv, vk.PipelineSpec{RequiredSubgroupSize: s.wave}); err != nil {
			return err
		}
	}
	for _, s := range []*stack{d.bb, d.dd} {
		var err error
		spec := vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc}
		if s.pPrep, err = d.pipeline(s.name+" prep", s.prepF, spec); err != nil {
			return err
		}
		if s.pCombine, err = d.pipeline(s.name+" combine", s.combine, spec); err != nil {
			return err
		}
		spec.RequiredSubgroupSize = 64
		if s.pAttn, err = d.pipeline(s.name+" attn", s.attn, spec); err != nil {
			return err
		}
	}
	for b, bank := range d.banks {
		d.gemms[b], d.gemvs[b] = map[string]*vk.ComputePipeline{}, map[string]*vk.ComputePipeline{}
		bb := []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, bank}
		q8 := b == layerBank && d.o.Bank == BankQ8
		gemvs, gemms := gemvSPIRV, gemmKernels
		if q8 {
			// llm_common.glsl's bindings: the scale plane is read as halves
			// at 3 and the tiles as words at 5; 4 is declared and unused.
			bb = []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, bank, d.abuf, bank}
			gemvs, gemms = gemvQ8SPIRV, q8Kernels
		}
		for slabs, spirv := range gemvs {
			for r := 1; r <= MaxRows; r++ {
				p, err := d.pipeline("gemv", spirv, vk.PipelineSpec{Buffers: bb, PushConstantSize: pc,
					SpecConstants: rowSpec(r), RequiredSubgroupSize: 64})
				if err != nil {
					return err
				}
				d.gemvs[b][fmt.Sprintf("k%d_r%d", slabs, r)] = p
			}
		}
		for _, k := range gemms {
			spec := vk.PipelineSpec{Buffers: bb, PushConstantSize: pc}
			if k.wave != 0 && !q8 {
				spec.RequiredSubgroupSize = k.wave
			} else if k.waves > 1 || q8 {
				spec.RequiredSubgroupSize = 64
			}
			p, err := d.pipeline("gemm "+k.name, k.spirv, spec)
			if err != nil {
				return err
			}
			d.gemms[b][k.name] = p
		}
	}
	for slabs, spirv := range gemvSumSPIRV {
		for r := 1; r <= MaxRows; r++ {
			if err := add(fmt.Sprintf("sum_k%d_r%d", slabs, r), "gemv sum", spirv,
				vk.PipelineSpec{SpecConstants: rowSpec(r), RequiredSubgroupSize: 64}); err != nil {
				return err
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

func bf16(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// get reads a bf16 tensor of the given shape.
func get(set *safetensors.Set, name string, shape ...int) (*safetensors.Tensor, error) {
	t, err := set.Get(name)
	if err != nil {
		return nil, err
	}
	if t.DType != safetensors.BF16 || len(t.Shape) != len(shape) {
		return nil, fmt.Errorf("breeze: %s is %v %s, want %v bf16", name, t.Shape, t.DType, shape)
	}
	for i := range shape {
		if t.Shape[i] != shape[i] {
			return nil, fmt.Errorf("breeze: %s is %v, want %v", name, t.Shape, shape)
		}
	}
	return t, nil
}

func bfAt(t *safetensors.Tensor, i int) float32 {
	return bf16(binary.LittleEndian.Uint16(t.Data[2*i:]))
}

// stageMatrix tiles rows [n][k] of t (row i at element offset i*k) into the
// fp16 head bank at off, with rows n..pad left zero.
func (d *Decoder) stageMatrix(off uint32, t *safetensors.Tensor, n, k, pad int) {
	buf := make([]uint16, pad*k)
	tileB(buf, n, k, func(i int, dst []float32) {
		for j := range dst {
			dst[j] = bfAt(t, i*k+j)
		}
	})
	d.banks[headBank].WriteUint16At(int(off), buf)
}

// stageProj stages a layer projection, the parts' rows stacked, in the
// layer bank's width.
func (d *Decoder) stageProj(off uint32, parts []*safetensors.Tensor, n, k int) {
	w := make([]float32, 0, n*k)
	for _, t := range parts {
		for i := 0; i < t.Shape[0]*k; i++ {
			w = append(w, bfAt(t, i))
		}
	}
	if d.o.Bank == BankQ8 {
		q, sc := tileBQ8(w, n, k)
		d.banks[layerBank].WriteBytesAt(int(off), q)
		d.banks[layerBank].WriteUint16At((int(off)+len(q))/2, sc)
		return
	}
	buf := make([]uint16, n*k)
	tileB(buf, n, k, func(i int, dst []float32) { copy(dst, w[i*k:(i+1)*k]) })
	d.banks[layerBank].WriteUint16At(int(off), buf)
}

// tileBQ8 quantises a [n, k] row-major matrix into the int8 bank's layout
// (ace/lm's, Kev's packQ8): tile (nt, kt) is 256 bytes holding (k, n) at
// (n%16)*16 + k%16, tiles kt-fastest; then the scale plane, scale (nt, group
// g, n%16) at (nt*kgroups + g)*16 + n%16. q is rounded against the fp16
// scale the kernel multiplies by.
func tileBQ8(w []float32, n, k int) ([]byte, []uint16) {
	kt, kg := k/tile, k/q8Group
	q := make([]byte, n*k)
	sc := make([]uint16, n*kg)
	parallel(n, func(r int) {
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
			dv := safetensors.F16ToF32(dh)
			sc[sbase+gi*tile] = dh
			for j, v := range blk {
				var qi int32
				if dv != 0 {
					qi = int32(math.Round(float64(v / dv)))
				}
				qi = min(max(qi, -127), 127)
				c := gi*q8Group + j
				q[base+(c/tile)*tile*tile+lane+c%tile] = byte(int8(qi))
			}
		}
	})
	return q, sc
}

func (d *Decoder) stageVec(off uint32, set *safetensors.Set, name string, n int) error {
	t, err := get(set, name, n)
	if err != nil {
		return err
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = bfAt(t, i)
	}
	d.wbuf.WriteFloat32At(int(off), v)
	return nil
}

func (d *Decoder) stage(set *safetensors.Set) error {
	for _, s := range []*stack{d.bb, d.dd} {
		cos, sin := s.ropeCos(s.maxLen)
		d.wbuf.WriteFloat32At(int(s.rope), cos)
		d.wbuf.WriteFloat32At(int(s.rope)+s.maxLen*hd/2, sin)
		if err := d.stageVec(s.finalNorm, set, s.normName, s.hidden); err != nil {
			return err
		}
		for i := range s.w {
			w := &s.w[i]
			pre := fmt.Sprintf("%s%d.", s.prefix, i)
			if err := d.stageVec(w.attnNorm, set, pre+"input_layernorm.weight", s.hidden); err != nil {
				return err
			}
			if err := d.stageVec(w.ffnNorm, set, pre+"post_attention_layernorm.weight", s.hidden); err != nil {
				return err
			}
			if s.qkNorm {
				if err := d.stageVec(w.qkNorm, set, pre+"self_attn.q_norm.weight", hd); err != nil {
					return err
				}
				if err := d.stageVec(w.qkNorm+hd, set, pre+"self_attn.k_norm.weight", hd); err != nil {
					return err
				}
			}
			for p := proj(0); p < nProj; p++ {
				n, k := s.projShape(p)
				var parts []*safetensors.Tensor
				rows := 0
				for _, name := range projParts[p] {
					t, err := set.Get(pre + name + ".weight")
					if err != nil {
						return err
					}
					if t.DType != safetensors.BF16 || len(t.Shape) != 2 || t.Shape[1] != k {
						return fmt.Errorf("breeze: %s%s is %v %s", pre, name, t.Shape, t.DType)
					}
					parts = append(parts, t)
					rows += t.Shape[0]
				}
				if rows != n {
					return fmt.Errorf("breeze: %s%v stack to %d rows, want %d", pre, projParts[p], rows, n)
				}
				d.stageProj(w.off[p], parts, n, k)
			}
		}
		runtime.GC()
	}
	t, err := get(set, "lm_head.weight", Vocab+1, bbHidden)
	if err != nil {
		return err
	}
	d.stageMatrix(d.lmHead, t, Vocab+1, bbHidden, d.headPad)
	if t, err = get(set, "depth_decoder.model.inputs_embeds_projector.weight", ddHidden, bbHidden); err != nil {
		return err
	}
	d.stageMatrix(d.projIn, t, ddHidden, bbHidden, ddHidden)
	// codebooks_head is [15][in][out]: h @ W[k], so tile its transpose.
	if t, err = get(set, "depth_decoder.codebooks_head.weight", Codebooks-1, ddHidden, Vocab); err != nil {
		return err
	}
	for k := 0; k < Codebooks-1; k++ {
		buf := make([]uint16, d.headPad*ddHidden)
		base := k * ddHidden * Vocab
		tileB(buf, Vocab, ddHidden, func(i int, dst []float32) {
			for j := range dst {
				dst[j] = bfAt(t, base+j*Vocab+i)
			}
		})
		d.banks[headBank].WriteUint16At(int(d.ddHeads)+k*d.headPad*ddHidden, buf)
	}
	if t, err = get(set, "depth_decoder.model.embed_tokens.weight", Codebooks*Vocab, bbHidden); err != nil {
		return err
	}
	d.Embed = make([]uint16, Codebooks*Vocab*bbHidden)
	for i := range d.Embed {
		d.Embed[i] = binary.LittleEndian.Uint16(t.Data[2*i:])
	}
	return nil
}

// Destroy releases every Vulkan object.
func (d *Decoder) Destroy() {
	for _, p := range d.pipes {
		p.Destroy()
	}
	for _, set := range append(d.gemms[:], d.gemvs[:]...) {
		for _, p := range set {
			p.Destroy()
		}
	}
	for _, s := range []*stack{d.bb, d.dd} {
		if s == nil {
			continue
		}
		for _, p := range []*vk.ComputePipeline{s.pPrep, s.pAttn, s.pCombine} {
			if p != nil {
				p.Destroy()
			}
		}
	}
	for _, m := range d.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.banks[0], d.banks[1]} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Row is one row of a pass: the slot whose cache it extends and its position.
type Row struct{ Slot, Pos int }

// proj is one [nOut, k] product of n rows: A fp16 at aOff (row stride lda),
// B tiled at bOff, C fp32 at cOff (row stride nOut). GEMV at ≤ MaxRows rows
// (split-K over slabs), a GEMM rung above.
func (d *Decoder) proj(bank, slabs int, aOff uint32, lda int, cOff, bOff uint32, nOut, k, n int) []vk.MultiDispatch {
	if n > MaxRows {
		if bank == layerBank && d.o.Bank == BankQ8 {
			gk := q8Kernels[0]
			if n > 128 {
				gk = q8Kernels[1]
			}
			tokPad := roundUp(n, gk.bm)
			pc := gemvPush{xnOff: aOff, outOff: cOff, bOff: bOff, lda: uint32(lda),
				gemmM: uint32(tokPad), gemmN: uint32(nOut), gemmK: uint32(k)}
			return []vk.MultiDispatch{{Pipeline: d.gemms[bank][gk.name], GroupsX: uint32(nOut / gk.bn),
				GroupsY: uint32(tokPad / gk.bm), PushConstants: pc.bytes()}}
		}
		gk := gemmFor(n, nOut)
		tokPad := roundUp(n, gk.bm)
		return []vk.MultiDispatch{{Pipeline: d.gemms[bank][gk.name], GroupsX: uint32(nOut / gk.bn),
			GroupsY: uint32(tokPad / gk.bm), PushConstants: push{InOff: aOff, OutOff: cOff, BOff: bOff,
				GemmM: uint32(tokPad), GemmN: uint32(nOut), GemmK: uint32(k), LDA: uint32(lda)}.bytes()}}
	}
	kt := k / tile
	if kt%slabs != 0 || (kt/slabs)%4 != 0 {
		panic(fmt.Sprintf("breeze: %d slabs do not split K %d", slabs, k))
	}
	p := gemvPush{resOff: d.bb.aRes, xnOff: aOff, outOff: cOff, bOff: bOff, lda: uint32(lda),
		gemmN: uint32(nOut), gemmK: uint32(k)}
	out := []vk.MultiDispatch{{Pipeline: d.gemvs[bank][fmt.Sprintf("k%d_r%d", slabs, n)],
		GroupsX: uint32(slabs), GroupsY: uint32(nOut / tile), PushConstants: p.bytes()}}
	if slabs > 1 {
		out = append(out, vk.MultiDispatch{Pipeline: d.pipes[fmt.Sprintf("sum_k%d_r%d", slabs, n)],
			GroupsX: groups(nOut, 64), GroupsY: uint32(n), PushConstants: p.bytes()})
	}
	return out
}

// keyChunks splits a pass's keys for ace_lm_attn: 64 at a time for a decode
// step, longer when many rows would overrun the partials.
func keyChunks(n, nq, maxKeys int) (chunks, chunkLen int) {
	chunks = (maxKeys + keyBlock - 1) / keyBlock
	if lim := partBudget / (n * nq); chunks > lim {
		chunks = max(lim, 1)
	}
	chunkLen = roundUp((maxKeys+chunks-1)/chunks, keyBlock)
	return (maxKeys + chunkLen - 1) / chunkLen, chunkLen
}

// layerGraph is layer i of s over n rows whose metadata is at s.aMeta, from
// the layer's normed input in s.hA to the next layer's (or, after the last,
// the final norm's) in s.hA.
func (d *Decoder) layerGraph(s *stack, i, n, chunks, chunkLen int) []vk.MultiDispatch {
	w := &s.w[i]
	eps := math.Float32bits(float32(s.eps))
	var out []vk.MultiDispatch
	add := func(p *vk.ComputePipeline, gx, gy uint32, pc push) {
		out = append(out, vk.MultiDispatch{Pipeline: p, GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	}
	pj := func(p proj, aOff uint32, lda int, cOff uint32) {
		nOut, k := s.projShape(p)
		out = append(out, d.proj(layerBank, s.slabs[p], aOff, lda, cOff, w.off[p], nOut, k, n)...)
	}
	// addNorm is the residual join and the next norm in one: x += y * scale,
	// then rms(x) * w into s.hA.
	addNorm := func(y uint32, scale float32, next uint32) {
		add(d.pipes["addnorm"], uint32(n), 1, push{InOff: y, OutOff: s.aX, WOff: next, BOff: s.hA,
			LDA: uint32(s.ldaDim), Tokens: uint32(n), Dim: uint32(s.hidden), Eps: eps, Scale: math.Float32bits(scale)})
	}
	qkv, _ := s.projShape(pQKV)
	pj(pQKV, s.hA, s.ldaDim, s.aQKV)
	add(s.pPrep, uint32(n), uint32(s.nq+2*s.nkv), push{InOff: s.aQKV, Dim: uint32(qkv), OutOff: s.hQ,
		WOff: w.qkNorm, Aux0: s.aMeta, Aux1: w.kc, Aux2: w.vc, Span: uint32(s.maxLen), BOff: s.rope,
		KStride: uint32(s.maxLen), Eps: eps, Scale: math.Float32bits(float32(1 / math.Sqrt(hd))), Tokens: uint32(n)})
	add(s.pAttn, uint32(n), uint32(s.nq*chunks), push{InOff: s.hQ, OutOff: s.aPart, KOff: w.kc, VOff: w.vc,
		Aux0: s.aMeta, Aux1: uint32(chunkLen), Aux2: uint32(chunks), Span: uint32(s.maxLen), Tokens: uint32(n)})
	add(s.pCombine, uint32(n), uint32(s.nq), push{InOff: s.aPart, OutOff: s.hCtx, Dim: uint32(s.ldaQ),
		Aux0: s.aMeta, Aux1: uint32(chunkLen), Aux2: uint32(chunks), Tokens: uint32(n)})
	pj(pO, s.hCtx, s.ldaQ, s.aO)
	addNorm(s.aO, 1, w.ffnNorm)
	pj(pGU, s.hA, s.ldaDim, s.aGU)
	add(d.pipes["swiglu"], groups(n*s.ffn, 256), 1, push{InOff: s.aGU, OutOff: s.hFFN, Tokens: uint32(n),
		Dim: uint32(s.ffn), LDA: uint32(s.ldaF), Scale: math.Float32bits(ffScale)})
	pj(pDown, s.hFFN, s.ldaF, s.aFF)
	next := s.finalNorm
	if i+1 < len(s.w) {
		next = s.w[i+1].attnNorm
	}
	addNorm(s.aFF, 1/ffScale, next)
	return out
}

// stackGraph writes rows' metadata and returns every layer of s over them,
// then the final norm into s.hA (fp16, row stride ldaDim).
func (d *Decoder) stackGraph(s *stack, rows []Row) ([]vk.MultiDispatch, error) {
	n := len(rows)
	if n == 0 || n > s.rows {
		return nil, fmt.Errorf("breeze: a %s pass of %d rows; 1 to %d", s.name, n, s.rows)
	}
	meta := make([]uint32, 2*n)
	maxKeys := 0
	for r, row := range rows {
		if row.Slot < 0 || row.Slot >= s.slots || row.Pos < 0 || row.Pos >= s.maxLen {
			return nil, fmt.Errorf("breeze: %s row %d at slot %d position %d; the cache is %d x %d",
				s.name, r, row.Slot, row.Pos, s.slots, s.maxLen)
		}
		meta[2*r], meta[2*r+1] = uint32(row.Slot), uint32(row.Pos)
		maxKeys = max(maxKeys, row.Pos+1)
	}
	d.abuf.WriteUint32At(int(s.aMeta), meta)
	chunks, chunkLen := keyChunks(n, s.nq, maxKeys)
	g := []vk.MultiDispatch{{Pipeline: d.pipes["normf16"], GroupsX: uint32(n), GroupsY: 1,
		PushConstants: push{InOff: s.aX, OutOff: s.hA, WOff: s.w[0].attnNorm, Tokens: uint32(n),
			Dim: uint32(s.hidden), LDA: uint32(s.ldaDim), Eps: math.Float32bits(float32(s.eps))}.bytes()}}
	for i := range s.w {
		g = append(g, d.layerGraph(s, i, n, chunks, chunkLen)...)
	}
	return g, nil
}

// readRows reads n fp16 rows of width w at off (row stride lda) as fp32.
func (d *Decoder) readRows(off uint32, lda, w, n int) [][]float32 {
	out := make([][]float32, n)
	for r := range out {
		h := d.hbuf.ReadUint16At(int(off)+r*lda, w)
		out[r] = make([]float32, w)
		for j, v := range h {
			out[r][j] = safetensors.F16ToF32(v)
		}
	}
	return out
}

// readLogits reads n rows of the logits arena, each its first w values.
func (d *Decoder) readLogits(n, w int) [][]float32 {
	out := make([][]float32, n)
	for r := range out {
		out[r] = d.abuf.ReadFloat32At(int(d.aLogits)+r*d.headPad, w)
	}
	return out
}

// Backbone runs the backbone over rows, x their input embeddings ([n][2048]
// fp32). It returns each row's normed hidden state (the fp16 the head and
// the depth decoder read) and, for the last `logits` rows, the head's 2052
// logits (index 2051 is EOS).
func (d *Decoder) Backbone(x []float32, rows []Row, logits int) (hidden, lg [][]float32, took time.Duration, err error) {
	n := len(rows)
	if len(x) != n*bbHidden || logits > n {
		return nil, nil, 0, fmt.Errorf("breeze: backbone input of %d values for %d rows, %d logits", len(x), n, logits)
	}
	g, err := d.stackGraph(d.bb, rows)
	if err != nil {
		return nil, nil, 0, err
	}
	d.abuf.WriteFloat32At(int(d.bb.aX), x)
	if logits > 0 {
		g = append(g, d.proj(headBank, 1, d.bb.hA+uint32((n-logits)*d.bb.ldaDim), d.bb.ldaDim, d.aLogits, d.lmHead,
			d.headPad, bbHidden, logits)...)
	}
	if took, err = vk.DispatchMultiTimed(g, 1, 1, true); err != nil {
		return nil, nil, took, err
	}
	hidden = d.readRows(d.bb.hA, d.bb.ldaDim, bbHidden, n)
	if logits > 0 {
		lg = d.readLogits(logits, Vocab+1)
	}
	return hidden, lg, took, nil
}

// Depth runs the depth decoder over rows, in its input rows ([n][2048] fp32,
// narrowed to fp16 as the projector's operand). heads[r] is the codebook
// (0-14, for codebooks 1-15) whose logits row r returns, or -1 for none.
func (d *Decoder) Depth(in []float32, rows []Row, heads []int) (lg [][]float32, took time.Duration, err error) {
	n := len(rows)
	if len(in) != n*bbHidden || len(heads) != n {
		return nil, 0, fmt.Errorf("breeze: depth input of %d values for %d rows", len(in), n)
	}
	h := make([]uint16, n*d.ldaIn)
	for r := 0; r < n; r++ {
		for j := 0; j < bbHidden; j++ {
			h[r*d.ldaIn+j] = safetensors.F32ToF16(in[r*bbHidden+j])
		}
	}
	d.hbuf.WriteUint16At(int(d.hIn), h)
	g, err := d.stackGraph(d.dd, rows)
	if err != nil {
		return nil, 0, err
	}
	pre := d.proj(headBank, 8, d.hIn, d.ldaIn, d.dd.aX, d.projIn, ddHidden, bbHidden, n)
	g = append(pre, g...)
	// Consecutive rows on one codebook (the CFG pair) share a GEMV.
	for r := 0; r < n; {
		k := heads[r]
		e := r + 1
		for e < n && heads[e] == k && e-r < MaxRows {
			e++
		}
		if k >= 0 {
			g = append(g, d.proj(headBank, 1, d.dd.hA+uint32(r*d.dd.ldaDim), d.dd.ldaDim, d.aLogits+uint32(r*d.headPad),
				d.ddHeads+uint32(k*d.headPad*ddHidden), d.headPad, ddHidden, e-r)...)
		}
		r = e
	}
	if took, err = vk.DispatchMultiTimed(g, 1, 1, true); err != nil {
		return nil, took, err
	}
	all := d.readLogits(n, Vocab)
	lg = make([][]float32, n)
	for r, k := range heads {
		if k >= 0 {
			lg[r] = all[r]
		}
	}
	return lg, took, nil
}

// frameEmbed is the backbone's input for a frame: the sum of its 16 code
// rows (BreezeBackboneModelEmbeddings), in fp32 over the bf16 table.
func (d *Decoder) frameEmbed(dst []float32, f [16]int32) {
	clear(dst)
	for k, c := range f {
		row := d.Embed[(k*Vocab+int(c))*bbHidden:]
		for j := range dst {
			dst[j] += bf16(row[j])
		}
	}
}

// codeEmbed is code c of codebook k's row.
func (d *Decoder) codeEmbed(dst []float32, k int, c int32) {
	row := d.Embed[(k*Vocab+int(c))*bbHidden:]
	for j := range dst {
		dst[j] = bf16(row[j])
	}
}
