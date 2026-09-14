package dit

import (
	"fmt"
	"math"
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

	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	// Flash selects the tiled kernel over the one-query-per-workgroup one.
	// Both are kept: the simple one is far easier to reason about and is
	// what the tiled one is checked against.
	Flash bool

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
}

// NewGPUAttention uploads the norm weights and the rotary table, and builds
// the pipelines for a fixed sequence length.
func NewGPUAttention(dev *vk.Device, attn *Attention, rope *RoPE, tokens int) (*GPUAttention, error) {
	g := &GPUAttention{
		dev:     dev,
		Flash:   true,
		pipes:   make(map[string]*vk.ComputePipeline),
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
		"rmsnorm":   shaders.DiTRMSNorm,
		"rope":      shaders.DiTRoPE,
		"transpose": shaders.DiTTransposeK,
		"attention": shaders.DiTAttention,
		"flash":     shaders.DiTAttentionFlash,
	} {
		mod, err := dev.NewShaderModule(spirv)
		if err != nil {
			g.Destroy()
			return nil, fmt.Errorf("dit: shader %s: %w", name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{g.wbuf, g.abuf},
			PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
		})
		if err != nil {
			g.Destroy()
			return nil, fmt.Errorf("dit: pipeline %s: %w", name, err)
		}
		g.pipes[name] = pipe
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUAttention) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
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
	dim := g.heads * g.headDim
	if q.Rows != g.tokens || q.Cols != dim {
		return nil, fmt.Errorf("dit: q is %s, want [%d %d]", q, g.tokens, dim)
	}
	for name, m := range map[string]*Mat{"k": k, "v": v} {
		if m.Rows != q.Rows || m.Cols != q.Cols {
			return nil, fmt.Errorf("dit: %s is %s, want %s", name, m, q)
		}
	}

	g.abuf.WriteFloat32At(int(g.qOff), q.Data)
	g.abuf.WriteFloat32At(int(g.kOff), k.Data)
	g.abuf.WriteFloat32At(int(g.vOff), v.Data)

	base := pushConstants{
		Tokens: uint32(g.tokens), Dim: uint32(dim),
		Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
	}

	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	}

	if normAndRope {
		// The q/k norms are per head, so the span is headDim and a row holds
		// `heads` spans.
		for _, s := range []struct {
			off  uint32
			wOff uint32
		}{{g.qOff, g.normQOff}, {g.kOff, g.normKOff}} {
			pc := base
			pc.InOff, pc.OutOff, pc.WOff = s.off, s.off, s.wOff
			pc.Span = uint32(g.headDim)
			pc.Eps = math.Float32bits(float32(qkNormEps))
			add("rmsnorm", uint32(g.tokens*g.heads), 1, pc)
		}
		for _, off := range []uint32{g.qOff, g.kOff} {
			pc := base
			pc.InOff, pc.OutOff = off, off
			pc.WOff, pc.Aux0 = g.cosOff, g.sinOff
			add("rope", groups(g.tokens*g.heads*g.headDim/2, 256), 1, pc)
		}
	}

	pcT := base
	pcT.InOff, pcT.OutOff = g.kOff, g.kTOff
	pcT.KStride = uint32(g.kStride)
	add("transpose", groups(g.tokens*dim, 256), 1, pcT)

	pcA := base
	pcA.InOff, pcA.OutOff = g.qOff, g.outOff
	pcA.KOff, pcA.VOff, pcA.KStride = g.kTOff, g.vOff, uint32(g.kStride)
	pcA.Scale = math.Float32bits(float32(1 / math.Sqrt(float64(g.headDim))))
	if g.Flash {
		// One workgroup per QB query rows; QB is compiled into the shader.
		add("flash", groups(g.tokens, flashQueryBlock), uint32(g.heads), pcA)
	} else {
		add("attention", uint32(g.tokens), uint32(g.heads), pcA)
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

	out := NewMat(g.tokens, dim)
	copy(out.Data, g.abuf.ReadFloat32At(int(g.outOff), g.tokens*dim))
	return out, nil
}

// ActivationBytes is the size of the activation arena.
func (g *GPUAttention) ActivationBytes() int { return g.abuf.Size() }
