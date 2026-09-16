package llm

// The gated DeltaNet on the device: LLM.md L3b, and the GPU half of the
// reference L3a built in deltanet.go.
//
// Thirty-six of the 48 layers are this one, and it is the only block in the
// model with a loop-carried dependency as long as the prompt. L2a priced the
// nameable lines of it at **2.7% of llama.cpp's 512-token prefill graph**,
// spread over four projections, a convolution, two L2 norms, the recurrence
// and a gated norm:
//
//	MUL_MAT q8_0 m=10240 k=2560   36   the fused q/k/v projection
//	MUL_MAT q8_0 m=6144  k=2560   36   the output gate z
//	MUL_MAT f32  m=48    k=2560   72   ssm_alpha and ssm_beta
//	SSM_CONV_SILU                 36    6.8 ms   depthwise k=4, then SiLU
//	L2_NORM                       72    4.5 ms   q and k, per head of 128
//	SOFTPLUS, MUL, SIGMOID, CONT  ~180           the two per-head scalars
//	GATED_DELTA_NET               36   15.8 ms   the recurrence
//	RMS_NORM, MUL, SIGMOID, MUL   ~144           the gated output norm
//	MUL_MAT q8_0 m=2560  k=6144   36   the output projection
//
// **Five dispatches replace those**, and every fusion is the argument L2a
// made for `inject`, L2d for the PLE block's value and L2f for the attention
// layer's six matrices: two matmuls that read the same activation are one
// matmul with more columns.
//
//	qkv    xn -> q, k, v, z, alpha, beta          (the plain GEMM arm)
//	conv   depthwise conv + SiLU + L2 norm + the two per-head scalars
//	scan   the delta rule, state in registers across the token loop
//	norm   the gated RMS norm, into the output matmul's A operand
//	out    the output projection                  (the plain GEMM again)
//
// **Four of llama.cpp's matrices are one.** `attn_qkv` [2560, 10240],
// `attn_gate` [2560, 6144] and the two [2560, 48] F32 gate projections all
// read the same block input, so they are one [16512, 2560] weight. The two
// F32 ones are the interesting half: they are L2a's "tiny-N F32 matmul"
// scandal in miniature — 72 dispatches a graph for 0.25 MB of weights — and
// here they are 96 more columns on a matrix that already has 16384.
//
// That fusion puts them on the fp16 matrix cores, which is a **deviation from
// the 7-token dump and not from the reference**: L3a-5 found that
// `ggml_vk_mul_mat` takes the f32 vector path only up to
// `mul_mat_vec_max_cols = 8` output columns, so at any real ubatch llama.cpp
// evaluates alpha and beta in fp16 too. The dump has 7 tokens and therefore
// does not, which is why `TestDeltaNetGPUGateIsTheFp16Path` measures the gap
// rather than hiding it.
//
// What is **not** here is the chunked form. L3a-2 found llama.cpp running a
// sequential scan and declining its own `build_delta_net_chunking`, and vLLM
// doing the opposite; this file builds the scan, measures its whole ladder,
// and research/l3b-deltanet-gpu.md prices the chunk against what the scan
// turns out to cost.

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// dnBN is the compiled column block of the plain GEMM arm, which both of this
// layer's projections run on. Same kernel as the attention layer's.
const dnBN = attnBN

// dnWave is the wave the scan is pinned to, and the width every rung's
// LPC divides.
const dnWave = 64

// DNKernel names one build of shaders/llm_dn_scan.comp, by its (LPC, WAVES).
//
// LPC is how many lanes own one column of the [128, 128] state and it moves
// three things at once: a lane holds 128/LPC state elements in registers, a
// wave owns 64/LPC columns, and every workgroup of a head re-reads that
// head's whole q and k row once per token — so the re-read factor is
// 128/(COLS*WAVES). The top rung is llama.cpp's own shape, and it is the
// **worst** one here: the ladder is a bandwidth-to-ALU crossover that turns at
// LPC 8, and its ends are 6.3x apart on identical arithmetic
// (research/l3b-deltanet-gpu.md). The `…s` suffix forces the LDS arm and
// `…p` a one-deep software pipeline, both of which are ablations rather than
// candidates — the second is worth nothing anywhere.
type DNKernel string

const (
	DNScanL64   DNKernel = "l64"
	DNScanL32   DNKernel = "l32"
	DNScanL16   DNKernel = "l16"
	DNScanL8    DNKernel = "l8"
	DNScanL4    DNKernel = "l4"
	DNScanL2    DNKernel = "l2"
	DNScanL1    DNKernel = "l1"
	DNScanL16W4 DNKernel = "l16w4"
	DNScanL8W4  DNKernel = "l8w4"
	DNScanL4W4  DNKernel = "l4w4"
	DNScanL2W4  DNKernel = "l2w4"
	// The LDS arm of four of those shapes, which is what prices QKREG.
	DNScanL64S DNKernel = "l64s"
	DNScanL16S DNKernel = "l16s"
	DNScanL8S  DNKernel = "l8s"
	DNScanL4S  DNKernel = "l4s"
	// And four with the loads of token t+1 issued before the arithmetic of
	// token t, which is what an SSBO the kernel also writes to costs.
	DNScanL64P DNKernel = "l64p"
	DNScanL16P DNKernel = "l16p"
	DNScanL8P  DNKernel = "l8p"
	DNScanL4P  DNKernel = "l4p"
)

type dnVariant struct {
	name       DNKernel
	spirv      []byte
	lpc, waves int
	// state is how many state elements a lane holds; qkreg says whether the
	// 2*state operand elements sit beside them or go through LDS.
	state    int
	qkreg    bool
	prefetch bool
}

var dnVariants = []dnVariant{
	{DNScanL64, shaders.LLMDNScanL64, 64, 1, 2, true, false},
	{DNScanL32, shaders.LLMDNScanL32, 32, 1, 4, true, false},
	{DNScanL16, shaders.LLMDNScanL16, 16, 1, 8, true, false},
	{DNScanL8, shaders.LLMDNScanL8, 8, 1, 16, true, false},
	{DNScanL4, shaders.LLMDNScanL4, 4, 1, 32, true, false},
	{DNScanL2, shaders.LLMDNScanL2, 2, 1, 64, false, false},
	{DNScanL1, shaders.LLMDNScanL1, 1, 1, 128, false, false},
	{DNScanL16W4, shaders.LLMDNScanL16W4, 16, 4, 8, true, false},
	{DNScanL8W4, shaders.LLMDNScanL8W4, 8, 4, 16, true, false},
	{DNScanL4W4, shaders.LLMDNScanL4W4, 4, 4, 32, true, false},
	{DNScanL2W4, shaders.LLMDNScanL2W4, 2, 4, 64, false, false},
	{DNScanL64S, shaders.LLMDNScanL64S, 64, 1, 2, false, false},
	{DNScanL16S, shaders.LLMDNScanL16S, 16, 1, 8, false, false},
	{DNScanL8S, shaders.LLMDNScanL8S, 8, 1, 16, false, false},
	{DNScanL4S, shaders.LLMDNScanL4S, 4, 1, 32, false, false},
	{DNScanL64P, shaders.LLMDNScanL64P, 64, 1, 2, true, true},
	{DNScanL16P, shaders.LLMDNScanL16P, 16, 1, 8, true, true},
	{DNScanL8P, shaders.LLMDNScanL8P, 8, 1, 16, true, true},
	{DNScanL4P, shaders.LLMDNScanL4P, 4, 1, 32, true, true},
}

// DNKernels lists the rungs, the reference's shape first.
func DNKernels() []DNKernel {
	out := make([]DNKernel, len(dnVariants))
	for i, v := range dnVariants {
		out[i] = v.name
	}
	return out
}

// DefaultDNKernel is the measured winner of that ladder: 16 state elements a
// lane, 768 workgroups, q and k re-read 16 times, and **413 us a layer at 512
// tokens against llama.cpp's 438**. It wins or ties at every length from 64 to
// 2048, with l8w4 inside the 1-3% run-to-run spread.
// See research/l3b-deltanet-gpu.md and results/l3b_dn.csv.
func DefaultDNKernel() DNKernel { return DNScanL8 }

// cols is how many state columns one workgroup of this rung owns, and so
// 128/cols is both its workgroup count per head and its q/k re-read factor.
func (v dnVariant) cols() int { return dnWave / v.lpc * v.waves }

func dnVariantFor(k DNKernel) (dnVariant, bool) {
	for _, v := range dnVariants {
		if v.name == k {
			return v, true
		}
	}
	return dnVariant{}, false
}

// dnLayerWeights is where one layer's staged weights sit.
type dnLayerWeights struct {
	qkv, out  uint32 // fp16 bank
	conv      uint32 // fp32 arena: [convWidth][kern], channel-major
	gamma     uint32 // fp32 arena: ssm_norm, [headDim], shared by all 48 heads
	a, dtBias uint32 // fp32 arena: [nHeadV] each
	state     uint32 // fp32 activation arena: [nHeadV][headDim][headDim]
}

// DeltaNetGPU runs linear-attention layers on the device. It holds however
// many layers it was staged with — at L3b that is the three the trace covers,
// at L6 it will be all 36 — plus one set of activation arenas sized for the
// longest prompt, and one recurrent state per layer.
type DeltaNetGPU struct {
	dev *vk.Device
	cfg DeltaNetConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	layers []dnLayerWeights

	scan          DNKernel
	gemm, outGemm GEMMKernel
	// autoPlan re-chooses both GEMM rungs per run. SetPlan turns it off,
	// because a caller that named a rung meant it.
	autoPlan bool
	// keepSilu writes the convolution's un-normalised SiLU output, which is
	// `conv_output_silu` and is otherwise never materialised.
	keepSilu bool

	tokens, arenaRows, rows int
	lda, ldCtx              int

	// fp32 weight arena: conv taps, the head norm and the two per-head
	// vectors, per layer.
	wElems int

	// fp32 activations. aQKV is token **zero** of the fused projection's
	// output; the convolution's Conv-1 history rows sit in front of it.
	aQKVBase, aQKV                uint32
	aNorm, aSilu, aOut            uint32
	aGate, aBeta, aState, aResult uint32
	actElems                      int
	// fp16 activations.
	hXn, hCtx uint32
	hElems    int
}

// qkvN is the fused projection's output width: the convolution's 10240
// channels, the output gate's 6144, and the two per-head scalars, padded up
// to the GEMM's column block. Four of llama.cpp's matrices, one of ours.
func (g *DeltaNetGPU) qkvN() int {
	c := g.cfg
	return roundUpInt(c.ConvWidth()+c.Inner+2*c.NHeadV, dnBN)
}

// The column each of the fused projection's outputs starts at. The shaders
// derive the same numbers from heads, kvHeads and headDim, so these are the
// host's half of that contract and the tests check them.
func (g *DeltaNetGPU) colQ() int     { return 0 }
func (g *DeltaNetGPU) colK() int     { return g.cfg.QKWidth() }
func (g *DeltaNetGPU) colV() int     { return 2 * g.cfg.QKWidth() }
func (g *DeltaNetGPU) colZ() int     { return g.cfg.ConvWidth() }
func (g *DeltaNetGPU) colAlpha() int { return g.cfg.ConvWidth() + g.cfg.Inner }
func (g *DeltaNetGPU) colBeta() int  { return g.colAlpha() + g.cfg.NHeadV }

// ColQ, ColK, ColV, ColZ, ColAlpha and ColBeta are those columns, for a test
// that wants to slice one tensor out of the fused output.
func (g *DeltaNetGPU) ColQ() int     { return g.colQ() }
func (g *DeltaNetGPU) ColK() int     { return g.colK() }
func (g *DeltaNetGPU) ColV() int     { return g.colV() }
func (g *DeltaNetGPU) ColZ() int     { return g.colZ() }
func (g *DeltaNetGPU) ColAlpha() int { return g.colAlpha() }
func (g *DeltaNetGPU) ColBeta() int  { return g.colBeta() }

// NewDeltaNetGPU stages layers onto the device and builds every pipeline.
//
// maxTokens is the longest prompt the arenas are built for. Every staged
// layer gets its own recurrent state, zeroed here, because the state is the
// layer's and not the batch's.
func NewDeltaNetGPU(dev *vk.Device, cfg DeltaNetConfig, maxTokens int, layers []DeltaNetWeights) (*DeltaNetGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: %d tokens", maxTokens)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("llm: no linear-attention layers to stage")
	}
	if cfg.HeadDim != 128 {
		return nil, fmt.Errorf("llm: the scan is built for headDim 128, this checkpoint says %d", cfg.HeadDim)
	}
	if cfg.NHeadV%cfg.NHeadK != 0 {
		return nil, fmt.Errorf("llm: %d value heads do not group over %d key heads", cfg.NHeadV, cfg.NHeadK)
	}
	if cfg.NHeadV*cfg.HeadDim != cfg.Inner {
		return nil, fmt.Errorf("llm: %d value heads of %d do not make inner %d", cfg.NHeadV, cfg.HeadDim, cfg.Inner)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	g := &DeltaNetGPU{
		dev: dev, cfg: cfg,
		pipes:    make(map[string]*vk.ComputePipeline),
		tokens:   maxTokens,
		rows:     maxTokens,
		lda:      cfg.NEmbd + gemmPad,
		ldCtx:    cfg.Inner + gemmPad,
		scan:     DefaultDNKernel(),
		gemm:     GEMMKernelFor(maxTokens),
		outGemm:  OutGEMMKernelFor(maxTokens),
		autoPlan: true,
	}
	align := coopMatTile
	for _, v := range gemmVariants {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)

	if err := g.alloc(len(layers)); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(layers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// alloc lays out the four arenas. Nothing is read here — every size follows
// from the config — so the layout can be checked before a weight is touched.
func (g *DeltaNetGPU) alloc(nLayers int) error {
	c := g.cfg
	rows := g.arenaRows
	cw := c.ConvWidth()

	perLayerW := cw*c.Conv + c.HeadDim + 2*c.NHeadV
	g.wElems = nLayers * perLayerW
	var err error
	if g.wbuf, err = g.dev.NewBuffer(g.wElems * 4); err != nil {
		return fmt.Errorf("llm: deltanet fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	// The convolution's history is Conv-1 rows of *negative* token index in
	// front of the projection's own output, so a tap that reaches before
	// token zero is an ordinary read at a wrapped offset and a fresh sequence
	// is those rows zeroed. It is the same 120 KB window the CPU reference
	// carries beside the [128, 128, 48] state, and a kernel that forgets it
	// is wrong in a way nothing but a batch-split test sees (L3a-6).
	hist := c.Conv - 1
	g.aQKVBase = alloc((rows + hist) * g.qkvN())
	g.aQKV = g.aQKVBase + uint32(hist*g.qkvN())
	g.aNorm = alloc(rows * cw)
	g.aSilu = alloc(rows * cw)
	g.aOut = alloc(rows * c.Inner)
	g.aGate = alloc(rows * c.NHeadV)
	g.aBeta = alloc(rows * c.NHeadV)
	g.aState = alloc(nLayers * c.StateSize())
	g.aResult = alloc(rows * c.NEmbd)
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("llm: deltanet fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hXn = halloc(rows * g.lda)
	g.hCtx = halloc(rows * g.ldCtx)
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("llm: deltanet fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once: the A operands' pad columns, every short run's pad rows,
	// the convolution's history and every layer's recurrent state all start
	// here, and none of these kernels bounds-check.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	g.abuf.WriteFloat32(make([]float32, g.actElems))

	perBank := g.qkvN()*c.NEmbd + c.NEmbd*c.Inner
	if g.bank, err = g.dev.NewBuffer(nLayers * perBank * 2); err != nil {
		return fmt.Errorf("llm: deltanet fp16 weight bank (%d MB): %w", (nLayers*perBank*2)>>20, err)
	}
	g.layers = make([]dnLayerWeights, nLayers)
	for i := range g.layers {
		base := uint32(i * perLayerW)
		g.layers[i] = dnLayerWeights{
			qkv:    uint32(i * perBank),
			out:    uint32(i*perBank + g.qkvN()*c.NEmbd),
			conv:   base,
			gamma:  base + uint32(cw*c.Conv),
			a:      base + uint32(cw*c.Conv+c.HeadDim),
			dtBias: base + uint32(cw*c.Conv+c.HeadDim+c.NHeadV),
			state:  g.aState + uint32(i*c.StateSize()),
		}
	}
	return nil
}

func (g *DeltaNetGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf}
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"conv": shaders.LLMDNConv,
		"norm": shaders.LLMDNNorm,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < dnWave {
		return fmt.Errorf("llm: the scan and GEMM rungs need a pinned %d-wide subgroup", dnWave)
	}
	for _, v := range dnVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
		}); err != nil {
			return err
		}
	}
	for _, v := range gemmVariants {
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: dnWave,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (g *DeltaNetGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader dn.%s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline dn.%s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage packs each layer's weights.
//
// The fused projection is the four matrices laid end to end as output rows of
// one [qkvN, nEmbd] matrix, in the order the shaders derive: the
// convolution's 10240 channels (q, then k, then v), the gate z, then alpha
// and beta. The pad rows the column block adds are left zero and their
// outputs are never read.
func (g *DeltaNetGPU) stage(layers []DeltaNetWeights) error {
	c := g.cfg
	cw := c.ConvWidth()
	for i, w := range layers {
		for _, t := range []struct {
			name string
			off  uint32
			src  []float32
			want int
		}{
			{"ssm_conv1d", g.layers[i].conv, w.Conv1d, cw * c.Conv},
			{"ssm_norm", g.layers[i].gamma, w.Norm, c.HeadDim},
			{"ssm_a", g.layers[i].a, w.A, c.NHeadV},
			{"ssm_dt.bias", g.layers[i].dtBias, w.DTBias, c.NHeadV},
		} {
			if len(t.src) != t.want {
				return fmt.Errorf("llm: layer %d %s is %d values, want %d", i, t.name, len(t.src), t.want)
			}
			g.wbuf.WriteFloat32At(int(t.off), t.src)
		}

		for _, t := range []struct {
			name string
			src  []float32
			n    int
		}{
			{"attn_qkv", w.QKV, cw},
			{"attn_gate", w.Z, c.Inner},
			{"ssm_alpha", w.Alpha, c.NHeadV},
			{"ssm_beta", w.Beta, c.NHeadV},
		} {
			if len(t.src) != t.n*c.NEmbd {
				return fmt.Errorf("llm: layer %d %s is %d values, want %d", i, t.name, len(t.src), t.n*c.NEmbd)
			}
		}
		if len(w.Out) != c.NEmbd*c.Inner {
			return fmt.Errorf("llm: layer %d ssm_out is %d values, want %d", i, len(w.Out), c.NEmbd*c.Inner)
		}

		qkv := make([]uint16, g.qkvN()*c.NEmbd)
		tileB(qkv, w.QKV, cw, c.NEmbd, func(r int) int { return r })
		baseZ := g.colZ()
		tileB(qkv, w.Z, c.Inner, c.NEmbd, func(r int) int { return baseZ + r })
		baseA := g.colAlpha()
		tileB(qkv, w.Alpha, c.NHeadV, c.NEmbd, func(r int) int { return baseA + r })
		baseB := g.colBeta()
		tileB(qkv, w.Beta, c.NHeadV, c.NEmbd, func(r int) int { return baseB + r })
		g.bank.WriteUint16At(int(g.layers[i].qkv), qkv)

		out := make([]uint16, c.NEmbd*c.Inner)
		tileB(out, w.Out, c.NEmbd, c.Inner, func(r int) int { return r })
		g.bank.WriteUint16At(int(g.layers[i].out), out)
	}
	return nil
}

// SetPlan chooses the three rungs: the scan's, and one row block for each of
// the two projections. Every rung is built and they all read the same staged
// weight, so this moves a pipeline and restages nothing.
func (g *DeltaNetGPU) SetPlan(scan DNKernel, gemm, outGemm GEMMKernel) error {
	if _, ok := dnVariantFor(scan); !ok {
		return fmt.Errorf("llm: no scan kernel %q (have %v)", scan, DNKernels())
	}
	for _, k := range []GEMMKernel{gemm, outGemm} {
		if _, ok := gemmVariantFor(k); !ok {
			return fmt.Errorf("llm: no GEMM kernel %q (have %v)", k, GEMMKernels())
		}
	}
	g.scan, g.gemm, g.outGemm = scan, gemm, outGemm
	g.autoPlan = false
	return nil
}

// Plan reports the rungs in use.
func (g *DeltaNetGPU) Plan() (DNKernel, GEMMKernel, GEMMKernel) {
	return g.scan, g.gemm, g.outGemm
}

// KeepSilu makes the convolution also write its un-normalised SiLU output,
// which is `conv_output_silu` and which the fused kernel otherwise never
// materialises — the same NO_W switch the hyper-connection block's gate has.
// It costs a second 21 MB write at 512 tokens, so it is off by default and on
// in the tests.
func (g *DeltaNetGPU) KeepSilu(on bool) { g.keepSilu = on }

// Layers is how many layers are staged and Tokens the longest prompt the
// arenas were built for.
func (g *DeltaNetGPU) Layers() int { return len(g.layers) }
func (g *DeltaNetGPU) Tokens() int { return g.tokens }

// WeightBytes is what the staged layers cost on the device and
// ActivationBytes what the shared arenas cost.
func (g *DeltaNetGPU) WeightBytes() int { return g.wbuf.Size() + g.bank.Size() }

// Buffers is how many device allocations the layer holds (L6a). All 36
// layers' fused projections are one bank of 4.18 GB, which clears this
// device's 4 GiB - 4 by 2.8%.
func (g *DeltaNetGPU) Buffers() int         { return 4 }
func (g *DeltaNetGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Upload writes the layer's input: the hyper-connection block's output
// `hc_mixed`, [T][nEmbd], narrowed into the fused projection's A layout.
//
// The rows between the prompt and the GEMM's row block are zeroed rather than
// left, so a short run never reads what a longer one wrote — the kernels do
// not bounds-check and the pad rows' products have to be products of zeros.
func (g *DeltaNetGPU) Upload(xn []float32, nTok int) error {
	c := g.cfg
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(xn) != nTok*c.NEmbd {
		return fmt.Errorf("llm: input is %d values, want %d", len(xn), nTok*c.NEmbd)
	}
	g.rows = nTok
	if g.autoPlan {
		g.gemm, g.outGemm = GEMMKernelFor(nTok), OutGEMMKernelFor(nTok)
	}
	row := make([]uint16, c.NEmbd)
	for t := 0; t < nTok; t++ {
		for i, v := range xn[t*c.NEmbd : (t+1)*c.NEmbd] {
			row[i] = safetensors.F32ToF16(v)
		}
		g.hbuf.WriteUint16At(int(g.hXn)+t*g.lda, row)
	}
	if pad := g.arenaRows - nTok; pad > 0 {
		g.hbuf.WriteUint16At(int(g.hXn)+nTok*g.lda, make([]uint16, pad*g.lda))
		g.hbuf.WriteUint16At(int(g.hCtx)+nTok*g.ldCtx, make([]uint16, pad*g.ldCtx))
	}
	return nil
}

// Reset zeroes one layer's recurrent state and the convolution's history,
// which is what a fresh sequence starts from. The history is shared by every
// layer in an arena this size, so it is reset with each of them.
func (g *DeltaNetGPU) Reset(layer int) error {
	if layer < 0 || layer >= len(g.layers) {
		return fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	g.abuf.WriteFloat32At(int(g.layers[layer].state), make([]float32, c.StateSize()))
	g.abuf.WriteFloat32At(int(g.aQKVBase), make([]float32, (c.Conv-1)*g.qkvN()))
	return nil
}

// SetState writes a layer's carried state — the recurrence's matrices and the
// convolution's window — in the CPU reference's layouts, so the two
// implementations can be started from the same place.
func (g *DeltaNetGPU) SetState(layer int, st *DeltaNetState) error {
	if layer < 0 || layer >= len(g.layers) {
		return fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	if st == nil {
		return g.Reset(layer)
	}
	if len(st.S) != c.StateSize() || len(st.Conv) != c.ConvWidth()*(c.Conv-1) {
		return fmt.Errorf("llm: state is %d + %d values, want %d + %d",
			len(st.S), len(st.Conv), c.StateSize(), c.ConvWidth()*(c.Conv-1))
	}
	g.abuf.WriteFloat32At(int(g.layers[layer].state), st.S)
	// DeltaNetState.Conv is [channel][position] with the token axis fastest,
	// which is what ggml_ssm_conv slides a window over; the arena is
	// [position][channel] because it is the projection's own output extended
	// backwards. Column 0 is the oldest either way.
	hist := c.Conv - 1
	rows := make([]float32, hist*g.qkvN())
	for ch := 0; ch < c.ConvWidth(); ch++ {
		for p := 0; p < hist; p++ {
			rows[p*g.qkvN()+ch] = st.Conv[ch*hist+p]
		}
	}
	g.abuf.WriteFloat32At(int(g.aQKVBase), rows)
	return nil
}

// State reads a layer's carried state back in the same layouts, which is what
// makes a chunk-boundary comparison against the CPU reference possible.
func (g *DeltaNetGPU) State(layer int) (*DeltaNetState, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	hist := c.Conv - 1
	st := &DeltaNetState{
		S:    g.abuf.ReadFloat32At(int(g.layers[layer].state), c.StateSize()),
		Conv: make([]float32, c.ConvWidth()*hist),
	}
	// The window after a run is the last hist rows of what the convolution
	// read, which is this batch's tail unless the batch is shorter than it.
	rows := g.abuf.ReadFloat32At(int(g.aQKVBase), (hist+g.rows)*g.qkvN())
	for ch := 0; ch < c.ConvWidth(); ch++ {
		for p := 0; p < hist; p++ {
			st.Conv[ch*hist+p] = rows[(g.rows+p)*g.qkvN()+ch]
		}
	}
	return st, nil
}

// Carry advances the convolution's window in place, so that the next Upload
// and Run continue the sequence. The recurrent state carries by itself — the
// scan reads and writes the same arena — but the window is Conv-1 rows of the
// *projection's* output and has to move to the front of it.
func (g *DeltaNetGPU) Carry() {
	c := g.cfg
	hist := c.Conv - 1
	rows := g.abuf.ReadFloat32At(int(g.aQKV)+(g.rows-hist)*g.qkvN(), hist*g.qkvN())
	g.abuf.WriteFloat32At(int(g.aQKVBase), rows)
}

// graph builds one layer's dispatch sequence, with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes.
func (g *DeltaNetGPU) graph(layer int) ([]vk.MultiDispatch, []string, error) {
	if layer < 0 || layer >= len(g.layers) {
		return nil, nil, fmt.Errorf("llm: layer %d of %d", layer, len(g.layers))
	}
	c := g.cfg
	w := g.layers[layer]
	sv, _ := dnVariantFor(g.scan)
	gv, _ := gemmVariantFor(g.gemm)
	ov, _ := gemmVariantFor(g.outGemm)

	norm := uint32(0)
	if c.QKNorm == L2Max {
		norm = 1
	}
	base := push{
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd),
		LDA: uint32(g.lda), LDCtx: uint32(g.ldCtx), Eps: math.Float32bits(c.Eps),
		QKVOff: g.aQKV, NormOff: g.aNorm, OutOff: g.aOut, CtxOff: g.hCtx,
		ConvOff: w.conv, GammaOff: w.gamma, Kern: uint32(c.Conv),
		ConvOutOff: noW,
		SSMGateOff: g.aGate, SSMBetaOff: g.aBeta, SSMStateOff: w.state,
		SSMAOff: w.a, SSMDTOff: w.dtBias, SSMNorm: norm,
		Heads: uint32(c.NHeadV), KVHeads: uint32(c.NHeadK), HeadDim: uint32(c.HeadDim),
		GemmN: uint32(g.qkvN()),
	}
	if g.keepSilu {
		base.ConvOutOff = g.aSilu
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// 1. The one fused projection: four of llama.cpp's matrices, one matmul.
	qkv := base
	qkv.XnOff, qkv.OutOff, qkv.BOff = g.hXn, g.aQKV, w.qkv
	qkv.GemmM, qkv.GemmK = uint32(roundUpInt(g.rows, gv.bm)), uint32(c.NEmbd)
	add(string(g.gemm), "qkv", uint32(g.qkvN()/dnBN), uint32(roundUpInt(g.rows, gv.bm)/gv.bm), qkv)

	// 2. The convolution, its SiLU, the two L2 norms and the two per-head
	//    scalars. One plane per head, plus one for the scalars.
	planes := uint32(2*c.NHeadK + c.NHeadV + 1)
	add("conv", "conv", planes, uint32(g.rows), base)

	// 3. The delta rule.
	add(string(g.scan), "scan", uint32(c.NHeadV), uint32(c.HeadDim/sv.cols()), base)

	// 4. The gated output norm, into the output projection's A operand.
	add("norm", "norm", uint32(c.NHeadV), uint32(g.rows), base)

	// 5. The output projection.
	out := base
	out.XnOff, out.LDA = g.hCtx, uint32(g.ldCtx)
	out.OutOff, out.BOff = g.aResult, w.out
	out.GemmM, out.GemmN, out.GemmK = uint32(roundUpInt(g.rows, ov.bm)), uint32(c.NEmbd), uint32(c.Inner)
	add(string(g.outGemm), "out", uint32(c.NEmbd/dnBN), uint32(roundUpInt(g.rows, ov.bm)/ov.bm), out)
	return d, kinds, nil
}

// Run executes one layer over whatever Upload left in the arenas, advancing
// that layer's recurrent state.
func (g *DeltaNetGPU) Run(layer int) error {
	d, kinds, err := g.graph(layer)
	if err != nil {
		return err
	}
	for i := range d {
		if _, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true); err != nil {
			return fmt.Errorf("llm: deltanet dispatch %d (%s): %w", i, kinds[i], err)
		}
	}
	return nil
}

// Profile times each dispatch on the GPU over iters back-to-back repetitions.
//
// **The scan is not idempotent**, which is the one place this differs from
// the other two blocks' profilers: it reads the recurrent state and writes it
// back, so a repetition starts from the state the last one left. That does
// not change what it costs — the arithmetic, the traffic and the dependency
// chain are identical whatever the state holds — but it does mean a Profile
// leaves a state a Run would not, so the correctness tests drive Run and the
// state is reset around a profile.
func (g *DeltaNetGPU) Profile(layer, iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	d, kinds, err := g.graph(layer)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: deltanet dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across every staged layer, and is
// the number to quote rather than Profile's — the same argument HCGPU and
// AttnGPU make: one layer's weights are 116 MB and cannot sit in the 32 MiB
// MALL however often a dispatch is repeated, but the activations would stay
// warm in a way a real graph running 36 different layers does not.
func (g *DeltaNetGPU) ProfileSweep(iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	_, kinds, err := g.graph(0)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(kinds))
	for k := range kinds {
		var total time.Duration
		for l := range g.layers {
			d, _, err := g.graph(l)
			if err != nil {
				return nil, err
			}
			dur, err := vk.DispatchMultiTimed(d[k:k+1], 1, uint32(iters), true)
			if err != nil {
				return out, fmt.Errorf("llm: %s sweep, layer %d: %w", kinds[k], l, err)
			}
			total += dur
		}
		out = append(out, Stage{Kind: kinds[k], GPU: total / time.Duration(len(g.layers)*iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the layout deltanet.go's CPU reference
// uses.

// Result is the layer's output, `linear_attn_out`, [T][nEmbd].
func (g *DeltaNetGPU) Result() []float32 {
	return g.abuf.ReadFloat32At(int(g.aResult), g.rows*g.cfg.NEmbd)
}

// QKV is the fused projection's whole output row, [T][qkvN] —
// `linear_attn_qkv_mixed` and three more tensors. Column slices one of them
// out, which is how a disagreement gets located in a projection instead of in
// what consumed it.
func (g *DeltaNetGPU) QKV() []float32 {
	return g.abuf.ReadFloat32At(int(g.aQKV), g.rows*g.qkvN())
}

// Column slices one output of the fused projection out of its row.
func (g *DeltaNetGPU) Column(col, width int) []float32 {
	raw := g.QKV()
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*g.qkvN()+col:])
	}
	return out
}

// ConvSilu is the convolution's un-normalised output, [T][convWidth], and is
// only written when KeepSilu is on.
func (g *DeltaNetGPU) ConvSilu() []float32 {
	return g.abuf.ReadFloat32At(int(g.aSilu), g.rows*g.cfg.ConvWidth())
}

// Norm is the convolution's output with q and k L2-normalised per head and v
// left alone, [T][convWidth]: `q_conv_predelta`, `k_conv_predelta` and
// `v_conv_predelta` laid end to end. NormColumn slices one out.
func (g *DeltaNetGPU) Norm() []float32 {
	return g.abuf.ReadFloat32At(int(g.aNorm), g.rows*g.cfg.ConvWidth())
}

// NormColumn slices one of the three out of that tensor.
func (g *DeltaNetGPU) NormColumn(col, width int) []float32 {
	raw := g.Norm()
	cw := g.cfg.ConvWidth()
	out := make([]float32, g.rows*width)
	for t := 0; t < g.rows; t++ {
		copy(out[t*width:(t+1)*width], raw[t*cw+col:])
	}
	return out
}

// Gate is the log decay `gate`, [T][nHeadV], and BetaSig is `beta_sigmoid`.
func (g *DeltaNetGPU) Gate() []float32 {
	return g.abuf.ReadFloat32At(int(g.aGate), g.rows*g.cfg.NHeadV)
}

// BetaSig is the write strength after its sigmoid, [T][nHeadV].
func (g *DeltaNetGPU) BetaSig() []float32 {
	return g.abuf.ReadFloat32At(int(g.aBeta), g.rows*g.cfg.NHeadV)
}

// Out is the recurrence's own output, `attn_output`, [T][nHeadV][headDim].
func (g *DeltaNetGPU) Out() []float32 {
	return g.abuf.ReadFloat32At(int(g.aOut), g.rows*g.cfg.Inner)
}

// Final is the gated output norm, `final_output`, read back out of the fp16 A
// operand it was narrowed into — so it carries fp16's step and not the
// reference's f32.
func (g *DeltaNetGPU) Final() []float32 {
	c := g.cfg
	raw := g.hbuf.ReadUint16At(int(g.hCtx), (g.rows-1)*g.ldCtx+c.Inner)
	out := make([]float32, g.rows*c.Inner)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.ldCtx : t*g.ldCtx+c.Inner] {
			out[t*c.Inner+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *DeltaNetGPU) Destroy() {
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

// DNShape reports a scan rung's three coupled properties: how many state
// elements a lane holds in registers, how many workgroups the whole layer
// dispatches, and how many times each head's q and k row is re-read per
// token. They move together — that is the ladder — so a benchmark that
// reports the time without them is reporting half the measurement.
func DNShape(k DNKernel, c DeltaNetConfig) (regsPerLane, workgroups, reread int) {
	v, ok := dnVariantFor(k)
	if !ok {
		return 0, 0, 0
	}
	regs := v.state
	switch {
	case v.qkreg && v.prefetch:
		regs *= 5 // the state, and two tokens' worth of q and k
	case v.qkreg:
		regs *= 3 // the state, plus this lane's q and k components
	}
	return regs, c.NHeadV * c.HeadDim / v.cols(), c.HeadDim / v.cols()
}

// DNOperandsInRegisters says whether a rung keeps q and k beside the state or
// stages them in LDS — the second half of what its name means.
func DNOperandsInRegisters(k DNKernel) bool {
	v, _ := dnVariantFor(k)
	return v.qkreg
}

// DNPrefetches says whether a rung issues the next token's operands before
// the current token's arithmetic.
func DNPrefetches(k DNKernel) bool {
	v, _ := dnVariantFor(k)
	return v.prefetch
}
