package gemma4

// Rune's text tower on the device (research/rune-vertical.md R4/R5).
//
// One request is one packed pass, Kev's arrangement (K5, K7.3): the shared
// state's rows, then every question's branch at the next 64-aligned cache
// cell, attending to the state and to itself. A layer is:
//
//	normf16      x -> rms(x) * w_in, fp16                     dit_norm_scale_f16
//	qkv          the fused projection, int8                   llm_gemm Q8 rungs
//	prep, attn   norms, rope, K/V cache, flash attention      gemma4_attn_prep, gemma4_attn
//	o            the output projection                        llm_gemm
//	post_attn    x += rms(o) * w; xn = rms(x) to here and to the MoE's input
//	gate+up      GELU-tanh in the epilogue, fp16              kev_gemm_q8_glu -DGELU
//	down                                                       llm_gemm
//	  -- submit --
//	MoE          llm.MoEGPU: router, route, perm, Q8_0 up (GELU), down, combine
//	  -- submit --
//	post_ffn     x = (x + rms(rms(d) w1 + rms(m) w2) w3) * layer_scalar
//
// The dense weights are int8 rows with an fp16 scale per 32 (Kev's K7.1
// bank, llm_gemm's Q8B layout), the experts GGUF Q8_0 (experts.go): the Q8
// target the user set. Every weight that multiplies an input column is
// folded into its matrix before quantising (the router's scale and
// hidden^-1/2, both pre-FFN norms), so one unweighted rms(x) feeds the whole
// FFN half.
//
// The embedding stays on the host as the checkpoint's bf16 (1.48 GB,
// mmap'd), looked up as HF does it: bf16(w * bf16(sqrt(hidden))). The final
// norm and the head run on the host too, and only for the readout rows and
// only over the option labels' rows of the tied embedding: the answer is a
// softmax over at most 255 logits, not 262,144.

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	tile       = 16
	gemmPad    = 128 // an A operand's row pad in halves (§2.3)
	q8Group    = 32
	rowAlign   = 128 // the widest GEMM tile's BM
	keyAlign   = 64  // gemma4_attn's key block: every segment starts on one
	sharedFF   = 64  // the stand-in shared expert's width (its down is zero)
	moeRowsMax = 128
)

// push mirrors dit_common.glsl's PC block (Kev's `push`), padded to the
// 256-byte range every pipeline declares.
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

// pushBytes is the range every pipeline declares: the LLM's 64-uint block,
// because the int8 GEMM is llm_gemm.comp's and one command buffer takes one
// range.
const pushBytes = 256

func (p push) bytes() []byte {
	out := make([]byte, pushBytes)
	*(*push)(unsafe.Pointer(&out[0])) = p
	return out
}

// llmPush is llm_gemm.comp's push block (and kev_gemm_q8_glu's).
type llmPush struct {
	xnOff, outOff, bOff uint32 // A (fp16 arena, halves), C (fp32 arena, or fp16 for the GLU), B (bank, bytes)
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

type gemmKernel struct {
	name    string
	spirv   []byte
	bm, bn  int
	rowFast bool // kev_gemm_q8_glu's grid: x = row block, y = column tile
}

var gemmKernels = []gemmKernel{
	{"q8m2", shaders.LLMGEMMQ8M2, 32, 64, false},
	{"q8m4", shaders.LLMGEMMQ8M4, 64, 64, false},
	{"q8m8", shaders.LLMGEMMQ8M8, 128, 64, false},
	{"rbm2", shaders.KevGEMMQ8RBM2, 32, 64, true},
	{"rbm4", shaders.KevGEMMQ8RBM4, 64, 64, true},
	{"rbm8", shaders.KevGEMMQ8RBM8, 128, 64, true},
	{"gegm2", shaders.Gemma4GEMMQ8GEGLUM2, 32, 64, true},
	{"gegm4", shaders.Gemma4GEMMQ8GEGLUM4, 64, 64, true},
	{"gegm8", shaders.Gemma4GEMMQ8GEGLUM8, 128, 64, true},
}

func kernel(name string) gemmKernel {
	for _, k := range gemmKernels {
		if k.name == name {
			return k
		}
	}
	panic("gemma4: no GEMM " + name)
}

func (k gemmKernel) grid(n, tokPad int) (uint32, uint32) {
	if k.rowFast {
		return uint32(tokPad / k.bm), uint32(n / k.bn)
	}
	return uint32(n / k.bn), uint32(tokPad / k.bm)
}

// gemmFor is the rung ladder measured on Rune's shapes (R8, `cmd/rune
// -ladder`, ms over the 30 layers):
//
//	rows        64     128    256    512    1024   2048
//	qkv  wide   q8m4   q8m4   rbm8   rbm8   rbm8   q8m4     39.8 at 1024 (Kev's schedule 45.5)
//	o, down     q8m2   q8m2   q8m4   rbm4   rbm4   q8m8
//
// Everything else is within 1-5% of Kev's K7.6 schedule, which it replaces.
func gemmFor(rows, n int) gemmKernel {
	if n <= 4096 { // o, down
		switch {
		case rows <= 192:
			return kernel("q8m2")
		case rows <= 384:
			return kernel("q8m4")
		case rows <= 1536:
			return kernel("rbm4")
		}
		return kernel("q8m8")
	}
	switch { // the fused qkv
	case rows <= 192:
		return kernel("q8m4")
	case rows <= 1536:
		return kernel("rbm8")
	}
	return kernel("q8m4")
}

// gegluFor: gegm2 at 64 rows, gegm4 at 128-256 (and level with gegm8 at
// 1024), gegm8 at 512 and 2048.
func gegluFor(rows int) gemmKernel {
	switch {
	case rows <= 96:
		return kernel("gegm2")
	case rows <= 384:
		return kernel("gegm4")
	}
	return kernel("gegm8")
}

// proj is one projection's place in the int8 bank.
type proj struct {
	off  uint32 // bytes
	n, k int
}

type layer struct {
	full               bool
	hd, nkv            int
	qkv, o, gu, down   proj
	inNorm, postAttn   uint32 // weights arena
	qkNorm             uint32 // q_norm [hd] then k_norm [hd]
	post1, post2, post uint32
	pre, pre2          uint32 // the pre-FFN norms, applied in post_attn (not folded: R5)
	scalar             float32
}

// Options are what a load decides up front.
type Options struct {
	// Rows is the longest pass: the arena's rows. Zero takes 4096.
	Rows int
	// Vision stages the vision tower too (R10, ~1.2 GB), for images.
	Vision bool
}

// GPU is a staged Rune.
type GPU struct {
	Cfg *Config
	Tok *Tokenizer

	dev   *vk.Device
	set   *safetensors.Set
	embed *safetensors.Tensor
	norm  []float32 // the final norm

	wbuf, abuf, hbuf, bank *vk.Buffer
	moe                    *llm.MoEGPU
	// Vis is the vision tower, when Options.Vision staged it.
	Vis     *Vision
	feats   featureCache
	imageID int32 // <|image|>, whose rows a pass takes from Segment.Soft
	pipes   map[string]*vk.ComputePipeline
	mods    []*vk.ShaderModule
	layers  []layer

	// GEMM forces a rung per projection ("qkv", "o", "gate+up", "down" to a
	// gemmKernels name), for the ladder; unset ones take the schedule.
	GEMM map[string]string
	// Attention is "gqa" (the default: the full layers' KV heads through LDS
	// once for their eight query heads, R8) or "wave" (gemma4_attn.comp
	// for them too, the control).
	Attention string
	// LayersPerSubmit is how many layers go into one command buffer; Load
	// sets 8.
	LayersPerSubmit int
	// Trace, when set, is called after every layer with the residual in the
	// arena (Residual), for the per-layer comparison against the oracle. It
	// costs a submit a layer.
	Trace func(layer int)

	rows, cells int
	lda, ldMLP  int
	ropeSlide   uint32 // cos [rows][128] then sin
	ropeFull    uint32 // cos [rows][256] then sin

	aX, aP, aY, aMeta    uint32
	hA, hQ, hK, hV, hMLP uint32
}

// Load stages models/rune-26b-a4b (or another Gemma 4 26B-A4B text tower).
func Load(dev *vk.Device, dir string, o Options) (*GPU, error) {
	c, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	if o.Rows == 0 {
		o.Rows = 4096
	}
	g := &GPU{Cfg: c, Tok: tok, dev: dev, set: set, pipes: map[string]*vk.ComputePipeline{}, LayersPerSubmit: 8}
	g.rows = roundUp(o.Rows, rowAlign)
	// Every branch starts on a key block, so the cache holds the pass plus
	// one block of alignment slack a segment; a pass of R rows has at most R
	// segments, and twice the rows is a generous bound for real requests.
	g.cells = 2 * g.rows
	if err := g.stage(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	g.imageID, _ = tok.ID(imageToken)
	if o.Vision {
		if g.Vis, err = LoadVision(dev, set, dir, c.Hidden); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

const lmPre = "model.language_model."

func (g *GPU) f32(name string) ([]float32, error) {
	t, err := g.set.Get(lmPre + name)
	if err != nil {
		return nil, err
	}
	return t.F32(nil)
}

// stage lays out the arenas and stages every weight.
func (g *GPU) stage() error {
	c := g.Cfg
	H, R := c.Hidden, g.rows
	var err error
	if g.embed, err = g.set.Get(lmPre + "embed_tokens.weight"); err != nil {
		return err
	}
	if g.norm, err = g.f32("norm.weight"); err != nil {
		return err
	}

	// ---- the weights arena: norms and the two rope tables
	var w []float32
	walloc := func(v []float32) uint32 {
		off := uint32(len(w))
		w = append(w, v...)
		for len(w)%64 != 0 {
			w = append(w, 0)
		}
		return off
	}
	g.ropeSlide = walloc(ropeTable(R, c.HeadDim, c.Rope["sliding_attention"].Theta, c.HeadDim/2))
	full := c.Rope["full_attention"]
	g.ropeFull = walloc(ropeTable(R, c.GlobalHeadDim, full.Theta, int(full.Partial*float64(c.GlobalHeadDim))/2))

	// ---- the int8 bank: per layer [qkv | o | gate+up | down]
	bankBytes := 0
	g.layers = make([]layer, c.Layers)
	for i := range g.layers {
		l := &g.layers[i]
		l.full = c.Full(i)
		l.hd, l.nkv = c.HeadDim, c.KVHeads
		if l.full {
			l.hd, l.nkv = c.GlobalHeadDim, c.GlobalKVHeads
		}
		qkvN := c.Heads*l.hd + l.nkv*l.hd
		if !l.full {
			qkvN += l.nkv * l.hd
		}
		place := func(n, k int) proj {
			p := proj{off: uint32(bankBytes), n: n, k: k}
			bankBytes += roundUp(n*k+n*k/q8Group*2, 256)
			return p
		}
		l.qkv = place(qkvN, H)
		l.o = place(H, c.Heads*l.hd)
		l.gu = place(2*c.Intermediate, H)
		l.down = place(H, c.Intermediate)
	}
	if bankBytes > 0xfffffffc {
		return fmt.Errorf("gemma4: the int8 bank is %d bytes, over one buffer's range", bankBytes)
	}
	if g.bank, err = g.dev.NewBuffer(bankBytes); err != nil {
		return fmt.Errorf("gemma4: int8 bank (%d MB): %w", bankBytes>>20, err)
	}

	moeLayers := make([]llm.MoEWeights, c.Layers)
	zeroQ8 := func(name string, in, out int) *gguf.Tensor {
		data := make([]byte, 0, in*out/q8Block*(2+q8Block))
		zero := make([]float32, in)
		for range out {
			data = quantQ8_0(data, zero)
		}
		return &gguf.Tensor{Name: name, Type: gguf.Q8_0, Dims: []int64{int64(in), int64(out)}, Data: data}
	}
	shGate, shUp, shDown := zeroQ8("shgate", H, sharedFF), zeroQ8("shup", H, sharedFF), zeroQ8("shdown", sharedFF, H)
	rootScale := float32(1 / math.Sqrt(float64(H)))
	t0 := time.Now()
	for i := range g.layers {
		l := &g.layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		get := func(name string) []float32 {
			if err != nil {
				return nil
			}
			var v []float32
			v, err = g.f32(p + name)
			return v
		}
		in, postAttn := get("input_layernorm.weight"), get("post_attention_layernorm.weight")
		qn, kn := get("self_attn.q_norm.weight"), get("self_attn.k_norm.weight")
		pre, pre2 := get("pre_feedforward_layernorm.weight"), get("pre_feedforward_layernorm_2.weight")
		p1, p2, pst := get("post_feedforward_layernorm_1.weight"), get("post_feedforward_layernorm_2.weight"), get("post_feedforward_layernorm.weight")
		scalar := get("layer_scalar")
		q, k := get("self_attn.q_proj.weight"), get("self_attn.k_proj.weight")
		var v []float32
		if !l.full {
			v = get("self_attn.v_proj.weight")
		}
		o := get("self_attn.o_proj.weight")
		gate, up, down := get("mlp.gate_proj.weight"), get("mlp.up_proj.weight"), get("mlp.down_proj.weight")
		router, rscale := get("router.proj.weight"), get("router.scale")
		if err != nil {
			return err
		}
		l.inNorm, l.postAttn = walloc(in), walloc(postAttn)
		l.qkNorm = walloc(append(append([]float32(nil), qn...), kn...))
		l.post1, l.post2, l.post = walloc(p1), walloc(p2), walloc(pst)
		l.pre, l.pre2 = walloc(pre), walloc(pre2)
		l.scalar = scalar[0]

		if err := g.packQ8(l.qkv, append(append(q, k...), v...)); err != nil {
			return err
		}
		if err := g.packQ8(l.o, o); err != nil {
			return err
		}
		if err := g.packQ8(l.gu, interleaveGLU(gate, up, c.Intermediate, H)); err != nil {
			return err
		}
		if err := g.packQ8(l.down, down); err != nil {
			return err
		}
		// The router reads rms(x) * scale * hidden^-1/2, and the MoE block's
		// input is rms(x) * w_pre2 (gemma4_post.comp), so its fp16 matrix
		// carries scale * hidden^-1/2 / w_pre2. fp16 keeps a per-element
		// exponent, so the column spread is harmless here, where it was not
		// in an int8 group; the bound below is fp16's.
		for e := range c.Experts {
			row := router[e*H : (e+1)*H]
			for j := range row {
				row[j] *= rscale[j] * rootScale / pre2[j]
				if math.Abs(float64(row[j])) > 60000 || math.IsNaN(float64(row[j])) {
					return fmt.Errorf("gemma4: layer %d router weight %v does not fit fp16 after the fold", i, row[j])
				}
			}
		}
		ex, err := LoadExperts(g.set, c, i)
		if err != nil {
			return err
		}
		if !plainQ8 {
			// The grouped GEMM's Q8_TILED layout (R10): bit-identical,
			// 1.15-1.4x on the experts.
			ex.Gate.Data = tileQ8(ex.Gate.Data, c.Experts*c.MoEInter, H)
			ex.Up.Data = tileQ8(ex.Up.Data, c.Experts*c.MoEInter, H)
			ex.Down.Data = tileQ8(ex.Down.Data, c.Experts*H, c.MoEInter)
		}
		moeLayers[i] = llm.MoEWeights{
			Router: router, SharedGate: make([]float32, H),
			Gate:       &llm.ExpertBank{T: ex.Gate, In: H, Out: c.MoEInter, NExp: c.Experts},
			Up:         &llm.ExpertBank{T: ex.Up, In: H, Out: c.MoEInter, NExp: c.Experts},
			Down:       &llm.ExpertBank{T: ex.Down, In: c.MoEInter, Out: H, NExp: c.Experts},
			GateShexpT: shGate, UpShexpT: shUp, DownShexpT: shDown,
		}
		runtime.GC() // a layer's fp32 experts are 2.5 GB of garbage
	}
	quantised := time.Since(t0)
	if g.wbuf, err = g.dev.NewBuffer(len(w) * 4); err != nil {
		return err
	}
	g.wbuf.WriteFloat32(w)

	cfg := llm.MoEConfig{NEmbd: H, NExpert: c.Experts, NExpertUsed: c.TopK, FFNExpert: c.MoEInter, FFNShared: sharedFF}
	opts := []llm.MoEOption{llm.WithMoEGELU(), llm.WithMoENoShared()}
	if !plainQ8 {
		opts = append(opts, llm.WithMoEQ8Tiled())
	}
	if g.moe, err = llm.NewMoEGPU(g.dev, cfg, R, moeLayers, opts...); err != nil {
		return fmt.Errorf("gemma4: experts: %w", err)
	}
	g.moe.PinGemv(true)
	moeLayers = nil
	runtime.GC()
	_ = quantised

	// ---- the activation arenas
	maxQKV := 0
	for _, l := range g.layers {
		maxQKV = max(maxQKV, l.qkv.n)
	}
	var acts int
	aalloc := func(n int) uint32 {
		off := uint32(acts)
		acts += roundUp(n, 64)
		return off
	}
	Rp := R + moeRowsMax // a padded row block past the last row
	g.aX = aalloc(Rp * H)
	g.aP = aalloc(Rp * maxQKV)
	g.aY = aalloc(Rp * H)
	g.aMeta = aalloc(Rp * 8)
	if g.abuf, err = g.dev.NewBuffer(acts * 4); err != nil {
		return err
	}
	g.lda = c.Heads*c.GlobalHeadDim + gemmPad
	g.ldMLP = c.Intermediate + gemmPad
	var halves int
	halloc := func(n int) uint32 {
		off := uint32(halves)
		halves += roundUp(n, 64)
		return off
	}
	kv := max(c.KVHeads*c.HeadDim, c.GlobalKVHeads*c.GlobalHeadDim)
	g.hA = halloc(Rp * g.lda)
	g.hQ = halloc(Rp * c.Heads * c.GlobalHeadDim)
	g.hK = halloc(g.cells * kv)
	g.hV = halloc(g.cells * kv)
	g.hMLP = halloc(Rp * g.ldMLP)
	if g.hbuf, err = g.dev.NewBuffer(halves * 2); err != nil {
		return err
	}
	g.hbuf.Zero()
	return nil
}

// ropeTable is HF's cos and sin, [rows][hd/2] each: fp32 inv_freq_j =
// 1/theta^(2j/hd) for j < rotated and 0 past it (Gemma's "proportional"
// RoPE), times the fp32 position, then cos and sin of that fp32 product.
func ropeTable(rows, hd int, theta float64, rotated int) []float32 {
	half := hd / 2
	inv := make([]float32, half)
	for j := range rotated {
		inv[j] = float32(1 / math.Pow(theta, float64(2*j)/float64(hd)))
	}
	out := make([]float32, 2*rows*half)
	for p := range rows {
		for j := range half {
			f := float64(inv[j] * float32(p))
			out[p*half+j] = float32(math.Cos(f))
			out[rows*half+p*half+j] = float32(math.Sin(f))
		}
	}
	return out
}

// packQ8 stages a [n, k] weight as int8 in the fragment tiling plus its
// scale plane, llm.tileBQ8's layout (Kev's K7.1 packQ8): tile (nt, kt) is 256
// bytes holding (k, n) at (n%16)*16 + k%16, tiles kt-fastest; scale (nt,
// group g, n%16) at (nt*kgroups + g)*16 + n%16, after the tiles. q is
// rounded against the fp16 scale the kernel multiplies by.
func (g *GPU) packQ8(p proj, w []float32) error {
	n, k := p.n, p.k
	if len(w) != n*k {
		return fmt.Errorf("gemma4: packing %d values into [%d %d]", len(w), n, k)
	}
	if k%q8Group != 0 || n%tile != 0 {
		return fmt.Errorf("gemma4: [%d %d] does not tile for the int8 bank", n, k)
	}
	kt, kg := k/tile, k/q8Group
	q := make([]byte, n*k)
	sc := make([]uint16, n*k/q8Group)
	parallelFor(n, func(r int) {
		base := (r / tile) * kt * tile * tile
		lane := (r % tile) * tile
		sbase := (r/tile)*kg*tile + r%tile
		x := w[r*k : (r+1)*k]
		for gi := range kg {
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
	g.bank.WriteBytesAt(int(p.off), q)
	g.bank.WriteUint16At((int(p.off)+len(q))/2, sc)
	return nil
}

// gluGroup is kev_gemm_q8_glu's BN / 2: a 64-column tile of the fused
// projection holds 32 gate rows then the same 32 up rows.
const gluGroup = 32

func interleaveGLU(gate, up []float32, n, k int) []float32 {
	out := make([]float32, 2*n*k)
	for g0 := 0; g0 < n; g0 += gluGroup {
		dst := 2 * g0 * k
		copy(out[dst:dst+gluGroup*k], gate[g0*k:(g0+gluGroup)*k])
		copy(out[dst+gluGroup*k:dst+2*gluGroup*k], up[g0*k:(g0+gluGroup)*k])
	}
	return out
}

func (g *GPU) build() error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	in, out := g.moe.InPort(), g.moe.OutPort()
	for _, p := range []struct {
		name  string
		spirv []byte
		bufs  []*vk.Buffer
		wave  uint32
	}{
		{"normf16", shaders.DiTNormScaleF16, base, 0},
		{"prep_slide", shaders.Gemma4AttnPrepSlide, base, 0},
		{"prep_full", shaders.Gemma4AttnPrepFull, base, 0},
		{"attn_slide", shaders.Gemma4AttnSlide, base, 64},
		{"attn_full", shaders.Gemma4AttnFull, base, 64},
		{"attn_gqa", shaders.Gemma4AttnGQA, base, 64},
		// The joins bind the MoE block's two arenas: they write its input
		// and read its output where they lie (llm's HCLink idea).
		{"post_attn", shaders.Gemma4PostAttn, []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, out.Buf, in.Buf}, 0},
		{"post_ffn", shaders.Gemma4PostFFN, []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, out.Buf, in.Buf}, 0},
	} {
		if err := g.pipeline(p.name, p.spirv, vk.PipelineSpec{Buffers: p.bufs, PushConstantSize: pc, RequiredSubgroupSize: p.wave}); err != nil {
			return err
		}
	}
	for _, k := range gemmKernels {
		// llm_common.glsl's bindings: the bank as halves at 3 (the scale
		// plane) and as words at 5 (the tiles); 4 is declared and unused.
		spec := vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf, g.bank},
			PushConstantSize: pc, RequiredSubgroupSize: 64}
		if err := g.pipeline(k.name, k.spirv, spec); err != nil {
			return err
		}
	}
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("gemma4: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("gemma4: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// Destroy releases every device object and the checkpoint's mapping.
func (g *GPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	if g.moe != nil {
		g.moe.Destroy()
	}
	if g.Vis != nil {
		g.Vis.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	if g.set != nil {
		g.set.Close()
	}
}

// ---- a pass

// plainQ8 (RUNE_Q8_TILED=0) stages the experts in GGUF's own Q8_0 layout
// and runs the plain builds: the control for the tiled ones.
var plainQ8 = os.Getenv("RUNE_Q8_TILED") == "0"

// causalImages drops the image tokens' bidirectional attention: the
// control for TestVisionOracle's mask, never served.
var causalImages = false

// Pass is one packed forward: the state's tokens, then each branch's.
type Pass struct {
	n        int
	meta     []uint32
	ids      []int32
	readouts []int             // the row each branch is read at: its last
	soft     map[int][]float32 // an image token's row: its input, in place of the embedding
}

// Segment is one request's share of a pass: its state, then its branches.
// Soft holds the state's images' features in order ([soft][hidden] each):
// the state's image tokens take their rows from it, one a token.
type Segment struct {
	State    []int32
	Branches [][]int32
	Soft     [][]float32
}

// NewPass lays out a state and its branches as Kev's planner does: the state
// at cells [0, len), positions [0, len); branch b at the next 64-aligned
// cell, positions continuing from the state's end.
func (g *GPU) NewPass(state []int32, branches [][]int32) (*Pass, error) {
	return g.NewBatch([]Segment{{State: state, Branches: branches}})
}

// NewBatch packs several requests into one pass (R8, Kev's K7.5): each
// request's state at the next 64-aligned cell, its branches after it, every
// request's positions from zero. A row's metadata names its own request's
// state cells, so a branch sees its state and itself and nothing of another
// request; the requests share the pass's weight reads, which for Rune's
// experts are most of a pass.
func (g *GPU) NewBatch(segs []Segment) (*Pass, error) {
	p := &Pass{}
	cursor := 0
	H := g.Cfg.Hidden
	for slot, sg := range segs {
		L := len(sg.State)
		if L == 0 {
			return nil, fmt.Errorf("gemma4: an empty state")
		}
		sLo := roundUp(cursor, keyAlign)
		stateRow := len(p.ids)
		end := uint32(stateRow + L) // nonzero: the state is in this pass, not resident
		// An image's tokens attend to each other both ways on the sliding
		// layers (HF's blockwise overlay; the full layers stay causal): a
		// row in a run of image tokens carries the cell past the run's end
		// in its kind word's upper bits, which gemma4_attn reads.
		img, soft := 0, 0
		for i, id := range sg.State {
			kind := uint32(0)
			if id == g.imageID && g.imageID != 0 {
				j := i
				for j < L && sg.State[j] == id {
					j++
				}
				if !causalImages {
					kind = uint32(sLo+j) << 1
				}
				if img >= len(sg.Soft) || soft >= len(sg.Soft[img])/H {
					return nil, fmt.Errorf("gemma4: more image tokens than image features")
				}
				if p.soft == nil {
					p.soft = map[int][]float32{}
				}
				p.soft[len(p.ids)] = sg.Soft[img][soft*H : (soft+1)*H]
				if soft++; soft == len(sg.Soft[img])/H {
					img, soft = img+1, 0
				}
			}
			p.ids = append(p.ids, id)
			p.meta = append(p.meta, uint32(i), kind, uint32(stateRow), uint32(sLo+i), 0, 0, uint32(slot), end)
		}
		if img != len(sg.Soft) || soft != 0 {
			return nil, fmt.Errorf("gemma4: %d images' features for %d images' tokens", len(sg.Soft), img)
		}
		cursor = sLo + L
		for _, b := range sg.Branches {
			if len(b) == 0 {
				return nil, fmt.Errorf("gemma4: an empty branch")
			}
			at := roundUp(cursor, keyAlign)
			first := len(p.ids)
			for i, id := range b {
				p.ids = append(p.ids, id)
				p.meta = append(p.meta, uint32(L+i), 1, uint32(first), uint32(at+i), uint32(sLo), uint32(sLo+L), uint32(slot), end)
			}
			p.readouts = append(p.readouts, len(p.ids)-1)
			cursor = at + len(b)
		}
	}
	p.n = len(p.ids)
	if p.n > g.rows || cursor > g.cells {
		return nil, fmt.Errorf("gemma4: %d rows over %d cells; the arena holds %d rows and %d cells", p.n, cursor, g.rows, g.cells)
	}
	return p, nil
}

// Fits is whether a set of segments fits one pass's rows and cells, so a scheduler
// can pack within the arena without building the pass.
func (g *GPU) Fits(segs []Segment) bool {
	rows, cursor := 0, 0
	for _, sg := range segs {
		cursor = roundUp(cursor, keyAlign) + len(sg.State)
		rows += len(sg.State)
		for _, b := range sg.Branches {
			cursor = roundUp(cursor, keyAlign) + len(b)
			rows += len(b)
		}
	}
	return rows <= g.rows && cursor <= g.cells
}

// Rows is the pass's length.
func (p *Pass) Rows() int { return p.n }

// bf16 rounds an fp32 to bf16 and back, round to nearest even.
func bf16(f float32) float32 {
	b := math.Float32bits(f)
	b += 0x7fff + (b>>16)&1
	return math.Float32frombits(b &^ 0xffff)
}

// Embed is the input row of one token: HF's Gemma4TextScaledWordEmbedding in
// bf16, the scale cast to bf16 first (53.0, not sqrt(2816)) and the product
// rounded to bf16.
func (g *GPU) Embed(id int32, dst []float32) {
	H := g.Cfg.Hidden
	scale := bf16(float32(math.Sqrt(float64(H))))
	raw := g.embed.Data[int(id)*H*2 : (int(id)+1)*H*2]
	for i := range H {
		w := math.Float32frombits((uint32(raw[2*i]) | uint32(raw[2*i+1])<<8) << 16)
		dst[i] = bf16(w * scale)
	}
}

// Forward runs the first `layers` layers (all of them when layers is zero)
// over a pass. It leaves every row's residual in the arena (Residual).
func (g *GPU) Forward(p *Pass, layers int) (time.Duration, error) {
	if layers == 0 {
		layers = g.Cfg.Layers
	}
	return g.ForwardFrom(p, nil, 0, layers)
}

// ForwardFrom runs layers [from, to) over a pass whose residual entering
// layer `from` is x ([rows][hidden]), or the embedding when x is nil. It is
// the teacher-forced form the per-layer gate uses: each layer fed the
// reference's own input, so a layer's error is its own.
func (g *GPU) ForwardFrom(p *Pass, x []float32, from, to int) (time.Duration, error) {
	c := g.Cfg
	H, n := c.Hidden, p.n
	if x != nil {
		g.abuf.WriteFloat32At(int(g.aX), x[:n*H])
		g.abuf.WriteUint32At(int(g.aMeta), p.meta)
		if err := g.moe.Resize(n); err != nil {
			return 0, err
		}
		return g.layerRange(n, from, to)
	}
	x = make([]float32, n*H)
	var wg sync.WaitGroup
	for t := 0; t < n; t += 64 {
		wg.Add(1)
		go func(t int) {
			defer wg.Done()
			for r := t; r < min(t+64, n); r++ {
				if f, ok := p.soft[r]; ok {
					copy(x[r*H:(r+1)*H], f)
					continue
				}
				g.Embed(p.ids[r], x[r*H:(r+1)*H])
			}
		}(t)
	}
	wg.Wait()
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.WriteUint32At(int(g.aMeta), p.meta)
	if err := g.moe.Resize(n); err != nil {
		return 0, err
	}
	return g.layerRange(n, from, to)
}

func (g *GPU) layerRange(n, from, to int) (time.Duration, error) {
	start := time.Now()
	var pending []vk.MultiDispatch
	var labels []string
	submit := func() error {
		if len(pending) == 0 {
			return nil
		}
		if _, err := vk.DispatchMultiTimed(pending, 1, 1, true); err != nil {
			return fmt.Errorf("gemma4: %s: %w", strings.Join(labels, ","), err)
		}
		pending, labels = pending[:0], labels[:0]
		return nil
	}
	// A layer is one dispatch sequence, the MoE block's included
	// (llm.MoEGPU.Dispatches), and layers go LayersPerSubmit to a command
	// buffer: a submit is a fence wait. A long pass submits a layer at a
	// time, so one command buffer cannot outlive the driver's reset
	// watchdog (Kev's and zimage/qwen's rule).
	per := max(g.LayersPerSubmit, 1)
	if n > 2048 {
		per = 1
	}
	for i := from; i < to; i++ {
		d, l := g.attnGraph(i, n)
		pending, labels = append(pending, d...), append(labels, l...)
		md, ml, err := g.moe.Dispatches(i)
		if err != nil {
			return 0, fmt.Errorf("gemma4: layer %d experts: %w", i, err)
		}
		pending, labels = append(pending, md...), append(labels, ml...)
		d, l = g.ffnJoin(i, n)
		pending, labels = append(pending, d...), append(labels, l...)
		if (i-from+1)%per == 0 && g.Trace == nil {
			if err := submit(); err != nil {
				return 0, err
			}
		}
		if g.Trace != nil {
			if err := submit(); err != nil {
				return 0, err
			}
			g.Trace(i)
		}
	}
	if err := submit(); err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

func (g *GPU) dispatch(d *[]vk.MultiDispatch, labels *[]string, pipe, label string, gx, gy uint32, pc []byte) {
	*d = append(*d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc})
	*labels = append(*labels, label)
}

func (g *GPU) gemm(d *[]vk.MultiDispatch, labels *[]string, label string, k gemmKernel, pr proj, aOff, cOff uint32, lda, ldaLo, n int) {
	if name, ok := g.GEMM[label]; ok {
		k = kernel(name)
	}
	tokPad := roundUp(n, k.bm)
	pc := llmPush{xnOff: aOff, outOff: cOff, bOff: pr.off, lda: uint32(lda), ldaLo: uint32(ldaLo),
		gemmM: uint32(tokPad), gemmN: uint32(pr.n), gemmK: uint32(pr.k)}
	gx, gy := k.grid(pr.n, tokPad)
	g.dispatch(d, labels, k.name, label, gx, gy, pc.bytes())
}

// attnGraph is a layer up to the MoE: the attention half, its join, and
// the dense MLP.
func (g *GPU) attnGraph(i, n int) ([]vk.MultiDispatch, []string) {
	c := g.Cfg
	l := &g.layers[i]
	H := c.Hidden
	eps := math.Float32bits(float32(c.Eps))
	var d []vk.MultiDispatch
	var labels []string

	g.dispatch(&d, &labels, "normf16", "in norm", uint32(n), 1,
		push{InOff: g.aX, OutOff: g.hA, WOff: l.inNorm, Dim: uint32(H), LDA: uint32(g.lda), Tokens: uint32(n), Eps: eps}.bytes())
	g.gemm(&d, &labels, "qkv", gemmFor(n, l.qkv.n), l.qkv, g.hA, g.aP, g.lda, 0, n)

	kind, rope := "slide", g.ropeSlide
	if l.full {
		kind, rope = "full", g.ropeFull
	}
	g.dispatch(&d, &labels, "prep_"+kind, "attn prep", uint32(n), 1,
		push{InOff: g.aP, Aux0: uint32(l.qkv.n), BOff: g.hQ, KOff: g.hK, VOff: g.hV, WOff: l.qkNorm,
			Aux1: g.aMeta, Aux2: rope, Span: uint32(g.rows), Eps: eps, Tokens: uint32(n)}.bytes())
	ap := push{InOff: g.hQ, KOff: g.hK, VOff: g.hV, OutOff: g.hA, LDA: uint32(g.lda), Aux1: g.aMeta, Tokens: uint32(n)}.bytes()
	if l.full && g.Attention != "wave" {
		g.dispatch(&d, &labels, "attn_gqa", "attention full", uint32((n+15)/16), uint32(2*l.nkv), ap)
	} else {
		g.dispatch(&d, &labels, "attn_"+kind, "attention "+kind, uint32((n+15)/16), uint32(c.Heads*l.hd/256), ap)
	}
	g.gemm(&d, &labels, "o", gemmFor(n, l.o.n), l.o, g.hA, g.aY, g.lda, 0, n)

	in := g.moe.InPort()
	pad := max(roundUp(n, rowAlign), in.Rows, n+1)
	g.dispatch(&d, &labels, "post_attn", "post attn", uint32(pad), 1,
		push{OutOff: g.aX, InOff: g.aY, WOff: l.postAttn, KOff: l.pre, VOff: l.pre2, Dim: uint32(H), Eps: eps, BOff: g.hA, LDA: uint32(g.lda),
			Aux1: in.Off, Aux2: uint32(in.Stride), Tokens: uint32(n), GemmM: uint32(pad)}.bytes())

	gk := gegluFor(n)
	g.gemm(&d, &labels, "gate+up", gk, l.gu, g.hA, g.hMLP, g.lda, g.ldMLP, n)
	g.gemm(&d, &labels, "down", gemmFor(n, l.down.n), l.down, g.hMLP, g.aY, g.ldMLP, 0, n)
	return d, labels
}

// ffnJoin is the layer's last dispatch: the dense MLP's and the experts'
// outputs through their norms into the residual.
func (g *GPU) ffnJoin(i, n int) ([]vk.MultiDispatch, []string) {
	c := g.Cfg
	l := &g.layers[i]
	var d []vk.MultiDispatch
	var labels []string
	g.dispatch(&d, &labels, "post_ffn", "post ffn", uint32(n), 1,
		push{OutOff: g.aX, InOff: g.aY, WOff: l.post1, KOff: l.post2, VOff: l.post, Aux0: g.moe.OutPort().Off,
			Dim: uint32(c.Hidden), Eps: math.Float32bits(float32(c.Eps)), Scale: math.Float32bits(l.scalar),
			Tokens: uint32(n)}.bytes())
	return d, labels
}

// Residual reads row r's residual after the last Forward.
func (g *GPU) Residual(r int) []float32 {
	H := g.Cfg.Hidden
	return g.abuf.ReadFloat32At(int(g.aX)+r*H, H)
}

// FinalNorm is the final RMS norm of one residual row, on the host.
func (g *GPU) FinalNorm(x []float32) []float32 {
	ss := 0.0
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+g.Cfg.Eps))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * g.norm[i]
	}
	return out
}

// LabelLogits is the tied head over the given token ids only, then Gemma's
// final softcap (logits/30 -> tanh -> *30). h is a final-normed row.
func (g *GPU) LabelLogits(h []float32, ids []int32) []float32 {
	H := g.Cfg.Hidden
	cap := g.Cfg.Softcap
	out := make([]float32, len(ids))
	for k, id := range ids {
		raw := g.embed.Data[int(id)*H*2 : (int(id)+1)*H*2]
		s := 0.0
		for i := range H {
			w := math.Float32frombits((uint32(raw[2*i]) | uint32(raw[2*i+1])<<8) << 16)
			s += float64(w) * float64(h[i])
		}
		out[k] = float32(math.Tanh(s/cap) * cap)
	}
	return out
}

// Profile times a pass dispatch by dispatch (R8): every layer's own
// dispatches with a timestamp after each, and the MoE block's through
// llm.MoEGPU.Profile. It runs the pass layer by layer, so its total is GPU
// time without the submits; Forward's wall time less this is the host and
// submit overhead. The map is GPU time by dispatch label, summed over the
// layers ("moe." prefixes the MoE block's).
func (g *GPU) Profile(p *Pass) (map[string]time.Duration, error) {
	if _, err := g.Forward(p, 0); err != nil { // the state every layer's inputs come from
		return nil, err
	}
	n := p.n
	H := g.Cfg.Hidden
	x := make([]float32, n*H)
	for r := range n {
		g.Embed(p.ids[r], x[r*H:(r+1)*H])
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	out := map[string]time.Duration{}
	for i := range g.Cfg.Layers {
		d, labels := g.attnGraph(i, n)
		_, marks, err := vk.DispatchMultiMarked(d, true)
		if err != nil {
			return nil, err
		}
		for k, l := range labels {
			out[l] += marks[k]
		}
		st, err := g.moe.Profile(i, 1)
		if err != nil {
			return nil, err
		}
		for _, s := range st {
			out["moe."+s.Kind] += s.GPU
		}
		d, labels = g.ffnJoin(i, n)
		_, marks, err = vk.DispatchMultiMarked(d, true)
		if err != nil {
			return nil, err
		}
		for k, l := range labels {
			out[l] += marks[k]
		}
	}
	return out, nil
}

// GEMMRungs is every rung a projection can run: the plain ones for qkv, o
// and down, the GELU-epilogue ones for gate+up.
func GEMMRungs(label string) []string {
	var out []string
	for _, k := range gemmKernels {
		if (label == "gate+up") == strings.HasPrefix(k.name, "gegm") {
			out = append(out, k.name)
		}
	}
	return out
}

// SyntheticPass is a pass of n rows over arbitrary in-vocabulary tokens, for
// timing: a state of n-1 and one branch of 1.
func (g *GPU) SyntheticPass(n int) (*Pass, error) {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(1000 + (i*7919)%200000)
	}
	return g.NewPass(ids[:n-1], [][]int32{ids[n-1:]})
}
