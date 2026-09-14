package dit

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// pushConstants mirrors the PC block in shaders/dit_common.glsl.
type pushConstants struct {
	InOff, OutOff, WOff uint32
	Tokens, Dim, Heads  uint32
	HeadDim, Span       uint32
	KOff, VOff, KStride uint32
	Eps, Scale          uint32
	Aux0, Aux1, Aux2    uint32
}

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// Kernel names one attention implementation. The scalar pair are kept
// alongside the matrix-core ladder on purpose: `KernelSimple` is small enough
// to read in one sitting, so it is the oracle the other two are checked
// against, and a disagreement between them is a bug in a kernel rather than a
// widening of everyone's tolerance.
type Kernel string

const (
	// KernelSimple is one query row per workgroup: shaders/dit_attention.comp.
	KernelSimple Kernel = "simple"
	// KernelFlash is the scalar tiled kernel with an online softmax:
	// shaders/dit_attention_flash.comp.
	KernelFlash Kernel = "flash"
)

// wmmaVariant is one tile geometry of shaders/dit_attention_wmma.comp. The
// geometry is baked into the SPIR-V -- it sizes register and `shared` arrays --
// so as in bench/ops_gemm_wmma.go the Go side has to be told it rather than
// reading it back: qt gives the dispatch grid, and wave must match the -DWAVE
// the variant was built with or the workgroup stops being a single wave and
// two waves share the same LDS scratch.
type wmmaVariant struct {
	name  Kernel
	spirv []byte
	qt    int    // query tiles per wave (16 rows each)
	ktil  int    // key tiles per block
	wave  uint32 // pinned subgroup size; 0 takes the driver's default (64 here)
}

// rows is how many query rows one workgroup covers.
func (v wmmaVariant) rows() int { return v.qt * coopMatTile }

// Intensity is the variant's arithmetic intensity in FLOP per byte of k and v
// read from global memory. Per key block a wave reads ktil*headDim/16
// fragments of each of k and v and issues 2*qt*ktil*headDim/16 multiplies, so
// ktil cancels and only qt moves the ratio -- which is why the ladder varies
// qt for intensity and ktil for register pressure, and why there is no third
// knob. The figure to beat is the ~28 FLOP/byte that saturates the MALL, and
// the scalar flash kernel's 2.67 (research/stage-3-dit-attention.md).
func (v wmmaVariant) Intensity() float64 { return float64(coopMatTile * v.qt) }

// coopMatTile is the cooperative-matrix extent: 16x16x16 is the only shape
// this device reports, so it is a constant rather than a parameter, and
// NewGPUAttention checks that it is still true before building any variant.
const coopMatTile = 16

// wmmaVariants is the stage-3c ladder, ordered so it reads off the results
// table: intensity 16 -> 32 -> 64 FLOP/byte at fixed ktil, then ktil varied at
// fixed intensity, then the wave32 arms (IDEAS §6.2). qt=4 was included
// expecting it to spill -- 32 accumulators plus 8 score tiles is past the
// wave64 register budget §2.7 ended on -- and it does, 361 VGPRs into scratch
// for 2.9x the time of qt=1. The measured table is in
// research/stage-3-dit-attention.md; the short version is that intensity does
// nothing here and the ladder is decided by spilling and by wave size.
var wmmaVariants = []wmmaVariant{
	{name: "wmma_qt1_kt4", spirv: shaders.DiTAttentionWMMAQT1KT4, qt: 1, ktil: 4},
	{name: "wmma_qt2_kt4", spirv: shaders.DiTAttentionWMMAQT2KT4, qt: 2, ktil: 4},
	{name: "wmma_qt4_kt4", spirv: shaders.DiTAttentionWMMAQT4KT4, qt: 4, ktil: 4},
	{name: "wmma_qt2_kt2", spirv: shaders.DiTAttentionWMMAQT2KT2, qt: 2, ktil: 2},
	{name: "wmma_qt2_kt8", spirv: shaders.DiTAttentionWMMAQT2KT8, qt: 2, ktil: 8},
	{name: "wmma_qt1_kt8", spirv: shaders.DiTAttentionWMMAQT1KT8, qt: 1, ktil: 8},
	{name: "wmma_qt1_kt4_w32", spirv: shaders.DiTAttentionWMMAQT1KT4W32, qt: 1, ktil: 4, wave: 32},
	{name: "wmma_qt2_kt4_w32", spirv: shaders.DiTAttentionWMMAQT2KT4W32, qt: 2, ktil: 4, wave: 32},
	{name: "wmma_qt1_kt2_w32", spirv: shaders.DiTAttentionWMMAQT1KT2W32, qt: 1, ktil: 2, wave: 32},
	{name: "wmma_qt1_kt8_w32", spirv: shaders.DiTAttentionWMMAQT1KT8W32, qt: 1, ktil: 8, wave: 32},
}

// wmmaControl is the same kernel as wmma_qt2_kt4 with the tail mask compiled
// out (-DNO_TAIL_MASK). It is built and dispatchable but deliberately left out
// of wmmaVariants, so it never appears in Kernels() and no benchmark picks it
// up: its only caller is the negative control in gpu_test.go.
var wmmaControl = wmmaVariant{
	name:  "wmma_qt2_kt4_nomask",
	spirv: shaders.DiTAttentionWMMAQT2KT4NoMask,
	qt:    2, ktil: 4,
}

// controls are the other two deliberate breakages the negative control
// switches on. They live here rather than in the test because what they break
// is the dispatch -- which layout v is packed in, and whether q carries the
// log2(e) that makes the kernel's exp2 the right base -- and a test cannot
// reach either from outside Apply.
type controls struct {
	// vNatural packs v in the same per-head layout as q and k instead of
	// transposed, so p.v reduces over the wrong axis.
	vNatural bool
	// noLog2E drops log2(e) from q's scale, leaving the kernel computing
	// 2^s where it should compute e^s.
	noLog2E bool
}

// DefaultKernel is the variant Apply uses unless told otherwise. It is the
// measured winner at 4096 tokens, which is the size that matters -- a 1024x1024
// latent is 4096 tokens and the caption stream's 320 is a rounding error beside
// it -- and it wins at 1024 and 2048 too: 6.64 ms against the runners-up's 9.1
// and the scalar flash kernel's 203.
//
// That the winner is the *lowest*-intensity geometry, at wave32, is the result
// of this stage. Arithmetic intensity was the whole story for the GEMM (§2.1)
// and it does nothing here: qt1, qt2 and qt4 span 16 to 64 FLOP/byte and land
// within 1.06x of each other at wave64. What moves the kernel is the register
// file -- qt4 spills and halves -- and the wave size, worth a clean 1.40x at
// the same tiling. K and V are read by every query block of the same head, so
// the caches supply the reuse that register blocking exists to provide, which
// is the same reason LDS staging lost in the GEMM.
const DefaultKernel Kernel = "wmma_qt1_kt4_w32"

// GPUAttention runs the DiT's attention stack -- q/k RMS norms, rotary
// embedding and multi-head softmax attention -- on a Vulkan device.
//
// It takes q, k and v already projected. The projections are GEMMs, and
// wiring them to the register-blocked WMMA kernels is stage 4's job; keeping
// them out here means a failure in this code is a failure of the attention
// kernel and nothing else.
type GPUAttention struct {
	dev  *vk.Device
	wbuf *vk.Buffer
	abuf *vk.Buffer
	// hbuf is the fp16 arena the matrix-core kernels read: q and k packed
	// per head, v packed per head and transposed. Nil when the device cannot
	// run them, which is also what makes Kernels() shorter.
	hbuf *vk.Buffer

	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule
	wmma  map[Kernel]wmmaVariant

	// Kernel selects the implementation. Defaults to DefaultKernel where the
	// device can run it and KernelFlash where it cannot.
	Kernel Kernel

	// ctl is empty in every real use; see controls.
	ctl controls

	cfg     *Config
	heads   int
	headDim int
	tokens  int
	kStride int

	// Weight arena offsets.
	normQOff, normKOff uint32
	cosOff, sinOff     uint32

	// Activation arena offsets, all fixed for a given token count.
	qOff, kOff, vOff, kTOff, outOff uint32
	actElems                        int

	// fp16 arena: three planes of 16x16 fragment tiles, one per operand.
	qhOff, khOff, vtOff uint32
	tokPad              int
	hElems              int
}

// NewGPUAttention uploads the norm weights and the rotary table, and builds
// the pipelines for a fixed sequence length.
func NewGPUAttention(dev *vk.Device, attn *Attention, rope *RoPE, tokens int) (*GPUAttention, error) {
	g := &GPUAttention{
		dev:     dev,
		pipes:   make(map[string]*vk.ComputePipeline),
		wmma:    make(map[Kernel]wmmaVariant),
		heads:   attn.Heads,
		headDim: attn.HeadDim,
		tokens:  tokens,
	}
	dim := attn.Heads * attn.HeadDim

	// Pad the transposed key stride off a multiple of the 4 KB channel
	// rotation (IDEAS §5.1b). 64 floats is 256 B, which lands the gcd inside
	// the [128, 256] window the rule wants.
	g.kStride = tokens + 64

	var w []float32
	put := func(v []float32) uint32 {
		off := uint32(len(w))
		w = append(w, v...)
		return off
	}
	g.normQOff = put(attn.NormQ.Weight)
	g.normKOff = put(attn.NormK.Weight)
	g.cosOff = put(rope.Cos)
	g.sinOff = put(rope.Sin)

	// Activation arena: q, k, v, the transposed keys, and the output.
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.qOff = alloc(tokens * dim)
	g.kOff = alloc(tokens * dim)
	g.vOff = alloc(tokens * dim)
	g.kTOff = alloc(attn.Heads * attn.HeadDim * g.kStride)
	g.outOff = alloc(tokens * dim)

	var err error
	if g.wbuf, err = dev.NewBuffer(len(w) * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("dit: weight buffer: %w", err)
	}
	g.wbuf.WriteFloat32(w)
	if g.abuf, err = dev.NewBuffer(g.actElems * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("dit: activation buffer (%d MB): %w", (g.actElems*4)>>20, err)
	}

	for name, spirv := range map[string][]byte{
		"rmsnorm":            shaders.DiTRMSNorm,
		"rope":               shaders.DiTRoPE,
		"transpose":          shaders.DiTTransposeK,
		string(KernelSimple): shaders.DiTAttention,
		string(KernelFlash):  shaders.DiTAttentionFlash,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{g.wbuf, g.abuf},
			PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
		}); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	g.Kernel = KernelFlash

	if err := g.buildWMMA(); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// pipeline compiles one shader module and its pipeline into g.
func (g *GPUAttention) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
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

// buildWMMA allocates the fp16 arena and builds the matrix-core ladder, or
// does nothing at all if this device cannot run it. Not being able to is not
// an error: the scalar kernels still work, and a device that reports a
// cooperative-matrix shape other than 16x16x16 fp16/fp32 would run these
// shaders wrong rather than slowly, so it is skipped rather than trusted.
func (g *GPUAttention) buildWMMA() error {
	feat := g.dev.Features()
	if !feat.Float16 || !feat.CoopMatrix {
		return nil
	}
	shapes, err := g.dev.Physical().CooperativeMatrixShapes()
	if err != nil {
		return fmt.Errorf("dit: cooperative-matrix shapes: %w", err)
	}
	ok := false
	for _, sh := range shapes {
		if sh.Scope == vk.ScopeSubgroup && sh.M == coopMatTile && sh.N == coopMatTile && sh.K == coopMatTile &&
			sh.AType == vk.ComponentFloat16 && sh.BType == vk.ComponentFloat16 &&
			sh.CType == vk.ComponentFloat32 && sh.ResultType == vk.ComponentFloat32 {
			ok = true
			break
		}
	}
	if !ok {
		return nil
	}
	// HEAD_DIM is compiled into the shader, as the accumulator grid it sizes
	// has to be known to the front end. Z-Image is 3840/30 = 128 everywhere.
	if g.headDim != wmmaHeadDim {
		return nil
	}

	// fp16 arena. All three operands are stored as 16x16 fragment tiles --
	// see shaders/dit_pack_f16.comp -- so the three planes are the same size
	// and differ only in what one tile holds.
	//
	// tokPad rounds the sequence up so a workgroup's last query tile and a key
	// block's tail read inside the arena rather than off the end; the padding
	// is zeroed once here, so a pad key scores zero and the kernel's tail mask
	// has something finite to work with.
	g.tokPad = (g.tokens + wmmaTokenAlign - 1) &^ (wmmaTokenAlign - 1)

	plane := g.heads * g.tokPad * g.headDim
	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.qhOff = halloc(plane)
	g.khOff = halloc(plane)
	g.vtOff = halloc(plane)

	buf, err := g.dev.NewBuffer(g.hElems * 2)
	if err != nil {
		return fmt.Errorf("dit: fp16 arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	g.hbuf = buf
	// Zero it once: the kernel reads the pad rows and columns, and fresh
	// device memory is not specified to be zero.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))

	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	if err := g.pipeline("pack", shaders.DiTPackF16, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: pcSize,
	}); err != nil {
		return err
	}

	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("dit: subgroup size control: %w", err)
	}
	for _, v := range append(append([]wmmaVariant{}, wmmaVariants...), wmmaControl) {
		// A wave32 variant is only correct if the size can actually be
		// pinned: without it the driver would pack two waves into the
		// 32-thread workgroup the SPIR-V declares.
		if v.wave != 0 && (!feat.SubgroupSizeControl || !sgs.Supported ||
			v.wave < sgs.MinSubgroupSize || v.wave > sgs.MaxSubgroupSize) {
			continue
		}
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: v.wave,
		}); err != nil {
			return err
		}
		g.wmma[v.name] = v
	}
	if _, ok := g.wmma[DefaultKernel]; ok {
		g.Kernel = DefaultKernel
	}
	return nil
}

// wmmaHeadDim is the head dimension compiled into
// shaders/dit_attention_wmma.comp (-DHEAD_DIM), and wmmaTokenAlign the
// sequence padding that covers every variant's query block and key block.
const (
	wmmaHeadDim    = 128
	wmmaTokenAlign = 128
)

// Kernels lists the implementations this device can run, scalar first.
func (g *GPUAttention) Kernels() []Kernel {
	out := []Kernel{KernelSimple, KernelFlash}
	for _, v := range wmmaVariants {
		if _, ok := g.wmma[v.name]; ok {
			out = append(out, v.name)
		}
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *GPUAttention) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	if g.hbuf != nil {
		g.hbuf.Destroy()
	}
	if g.abuf != nil {
		g.abuf.Destroy()
	}
	if g.wbuf != nil {
		g.wbuf.Destroy()
	}
}

func groups(n, local int) uint32 { return uint32((n + local - 1) / local) }

// flashQueryBlock must match QB in shaders/dit_attention_flash.comp.
const flashQueryBlock = 8

// Apply runs the attention stack over already-projected q, k and v, each
// [tokens, heads*headDim]. normAndRope selects whether the q/k norms and the
// rotary embedding run, so the pieces can be checked separately.
func (g *GPUAttention) Apply(q, k, v *Mat, normAndRope bool) (*Mat, error) {
	d, _, err := g.graph(q, k, v, normAndRope)
	if err != nil {
		return nil, err
	}
	// Batch the submits: a whole graph in one command buffer can exceed the
	// driver's reset watchdog, which is how the VAE decoder first failed at
	// 1024x1024 (PIPELINE.md).
	const perSubmit = 8
	for i := 0; i < len(d); i += perSubmit {
		j := min(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("dit: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return g.result(), nil
}

// Stage is one dispatch's identity and cost, as returned by Profile.
type Stage struct {
	Index int
	Kind  string
	GPU   time.Duration
}

// Profile runs the same graph Apply does, one dispatch at a time, and times
// each on the GPU. Wall-clock around Apply is not a measurement of the kernel:
// at 4096 tokens it also carries 189 MB of host writes for q, k and v and 63
// MB of read-back, which is how every tile geometry in the ladder came to look
// identical at 347 ms.
func (g *GPUAttention) Profile(q, k, v *Mat, normAndRope bool) ([]Stage, *Mat, error) {
	d, kinds, err := g.graph(q, k, v, normAndRope)
	if err != nil {
		return nil, nil, err
	}
	stages := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true)
		if err != nil {
			return stages, nil, fmt.Errorf("dit: dispatch %d (%s): %w", i, kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: kinds[i], GPU: dur})
	}
	return stages, g.result(), nil
}

// result copies the output tensor out of the activation arena.
func (g *GPUAttention) result() *Mat {
	dim := g.heads * g.headDim
	out := NewMat(g.tokens, dim)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.outOff), g.tokens*dim))
	return out
}

// graph uploads q, k and v and builds the dispatch sequence for the selected
// kernel, with a label per dispatch. Apply and Profile share it so that what
// the profiler times is what Apply runs.
func (g *GPUAttention) graph(q, k, v *Mat, normAndRope bool) ([]vk.MultiDispatch, []string, error) {
	dim := g.heads * g.headDim
	if q.Rows != g.tokens || q.Cols != dim {
		return nil, nil, fmt.Errorf("dit: q is %s, want [%d %d]", q, g.tokens, dim)
	}
	for name, m := range map[string]*Mat{"k": k, "v": v} {
		if m.Rows != q.Rows || m.Cols != q.Cols {
			return nil, nil, fmt.Errorf("dit: %s is %s, want %s", name, m, q)
		}
	}
	variant, isWMMA := g.wmma[g.Kernel]
	if !isWMMA && g.Kernel != KernelSimple && g.Kernel != KernelFlash {
		return nil, nil, fmt.Errorf("dit: no attention kernel %q (have %v)", g.Kernel, g.Kernels())
	}

	g.abuf.WriteFloat32At(int(g.qOff), q.Data)
	g.abuf.WriteFloat32At(int(g.kOff), k.Data)
	g.abuf.WriteFloat32At(int(g.vOff), v.Data)

	base := pushConstants{
		Tokens: uint32(g.tokens), Dim: uint32(dim),
		Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	if normAndRope {
		// The q/k norms are per head, so the span is headDim and a row holds
		// `heads` spans.
		for i, s := range []struct {
			off  uint32
			wOff uint32
		}{{g.qOff, g.normQOff}, {g.kOff, g.normKOff}} {
			pc := base
			pc.InOff, pc.OutOff, pc.WOff = s.off, s.off, s.wOff
			pc.Span = uint32(g.headDim)
			pc.Eps = math.Float32bits(float32(qkNormEps))
			add("rmsnorm", []string{"rmsnorm q", "rmsnorm k"}[i], uint32(g.tokens*g.heads), 1, pc)
		}
		for i, off := range []uint32{g.qOff, g.kOff} {
			pc := base
			pc.InOff, pc.OutOff = off, off
			pc.WOff, pc.Aux0 = g.cosOff, g.sinOff
			add("rope", []string{"rope q", "rope k"}[i], groups(g.tokens*g.heads*g.headDim/2, 256), 1, pc)
		}
	}

	scale := float32(1 / math.Sqrt(float64(g.headDim)))
	if isWMMA {
		// Pack into the fp16 arena as fragment tiles. q carries the softmax
		// scale and log2(e) with it, so the kernel's exponential is exp2 --
		// one instruction on this ISA.
		const log2e = 1.4426950408889634
		for _, s := range []struct {
			name     string
			src, dst uint32
			mode     uint32
			scale    float32
		}{
			{"pack q", g.qOff, g.qhOff, 0, scale * log2e},
			{"pack k", g.kOff, g.khOff, 0, 1},
			{"pack v", g.vOff, g.vtOff, 1, 1},
		} {
			if s.dst == g.qhOff && g.ctl.noLog2E {
				s.scale = scale
			}
			if s.dst == g.vtOff && g.ctl.vNatural {
				s.mode = 0
			}
			pc := base
			pc.InOff, pc.OutOff = s.src, s.dst
			pc.Aux0, pc.Aux1 = s.mode, uint32(g.tokPad)
			pc.Scale = math.Float32bits(s.scale)
			// One workgroup per (token tile, head). The grid covers the
			// padding as well as the sequence, so the pad tiles are rewritten
			// as zeros rather than trusted to have stayed that way.
			add("pack", s.name, groups(g.tokPad, coopMatTile), uint32(g.heads), pc)
		}

		pc := base
		pc.InOff, pc.OutOff = g.qhOff, g.outOff
		pc.KOff, pc.VOff = g.khOff, g.vtOff
		pc.Aux1 = uint32(g.tokPad)
		add(string(variant.name), "attention", groups(g.tokens, variant.rows()), uint32(g.heads), pc)
	} else {
		// The scalar kernels read the keys transposed instead, in fp32.
		pcT := base
		pcT.InOff, pcT.OutOff = g.kOff, g.kTOff
		pcT.KStride = uint32(g.kStride)
		add("transpose", "transpose k", groups(g.tokens*dim, 256), 1, pcT)

		pc := base
		pc.InOff, pc.OutOff = g.qOff, g.outOff
		pc.KOff, pc.VOff, pc.KStride = g.kTOff, g.vOff, uint32(g.kStride)
		pc.Scale = math.Float32bits(scale)
		if g.Kernel == KernelFlash {
			// One workgroup per QB query rows; QB is compiled into the shader.
			add(string(KernelFlash), "attention", groups(g.tokens, flashQueryBlock), uint32(g.heads), pc)
		} else {
			add(string(KernelSimple), "attention", uint32(g.tokens), uint32(g.heads), pc)
		}
	}
	return d, kinds, nil
}

// Intensity is the named kernel's arithmetic intensity in FLOP per byte of k
// and v read from global memory, or 0 for the scalar kernels -- whose figure
// is a measurement rather than a property of the tiling, and lives in
// research/stage-3-dit-attention.md.
func (g *GPUAttention) Intensity(k Kernel) float64 {
	if v, ok := g.wmma[k]; ok {
		return v.Intensity()
	}
	return 0
}

// ActivationBytes is the size of the activation arena.
func (g *GPUAttention) ActivationBytes() int { return g.abuf.Size() }
