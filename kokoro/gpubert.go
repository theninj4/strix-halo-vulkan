package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// bertPad is the extra halves on an fp16 A operand's row stride, and it does
// two jobs at once.
//
// §2.3: a leading dimension that is a multiple of 4 KB aliases every row onto
// the same DRAM channel. The feed-forward's K is 2048, which in halves is
// exactly 4096 bytes and exactly that case.
//
// And stage 9's trick: **a bias is an extra column of K.** Column `width` of
// every A row holds a 1 and the rest of the pad holds zeros, so a B operand
// with the bias at that K index adds it inside the MMA — no epilogue, no
// broadcast pass, and no kernel that knows what a bias is. Six projections a
// layer, twelve layers, and none of them costs a dispatch.
//
// It is 64 rather than one tile because dit_gemm.comp's K loop steps by BK,
// so the padded K has to stay a multiple of it.
const bertPad = 64

// GPUAlbert runs PL-BERT — StyleTTS2's phoneme encoder — on the device.
//
// It is here because the profile said so rather than because the plan did.
// `SPEECH.md` carried "60% of the phoneme side is recurrences" from T2; timing
// the stages put **ALBERT at 104 ms of 222**, which is 47% of the phoneme side
// and 40% of a whole utterance, against 6.7 GFLOP of arithmetic — a transformer
// that this repository has had kernels for since stage 3.
//
// Two things about ALBERT make it cheap to stage. All twelve layers are *one*
// weight group applied twelve times, so 5.5 M parameters are uploaded once and
// the graph is the same block repeated; and the sequence is the token count,
// fifty here, so every arena is sized for `max_position_embeddings` and the
// object is built once for any utterance rather than per clip.
type GPUAlbert struct {
	dev *vk.Device

	cfg      PLBERTConfig
	maxTok   int
	tokens   int
	ldHidden int // fp16 row stride for the 768-wide operands
	ldFFN    int // and for the 2048-wide one

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline
	gemm                   *vk.ComputePipeline

	// ScalarAttn runs shaders/kokoro_bert_attn.comp instead of the
	// matrix-core pair. It is the oracle the WMMA path was debugged against
	// and what TestGPUAlbertScaling measures the port against; it is not a
	// fallback for short utterances, because there is no length at which it
	// wins. At the reference fifty tokens -- where the tiles are mostly
	// padding and the three packs are pure overhead -- it is still 2.0x
	// slower a layer, and at the model's 510-phoneme ceiling 56x (T10).
	// Set before Apply; the graph reads it per run.
	ScalarAttn bool

	// Arena offsets, fp32.
	aX, aQ, aK, aV     uint32
	aCtx, aTmp, aAttn  uint32
	aFFN               uint32
	hA, hB             uint32
	hQ, hK, hV         uint32 // q, k and v as 16x16 fp16 fragment tiles
	actElems, hElems   int
	wAttnG, wAttnB     uint32 // the two LayerNorm affines
	wOutG, wOutB       uint32
	bQ, bK, bV, bDense uint32 // fragment-tiled projections, biases included
	bFFN, bFFNOut      uint32
}

// The matrix-core attention's two compile-time facts, which the host has to
// match or the addressing is silently wrong.
//
// bertAttnHeadDim is -DHEAD_DIM: 768 over 12 heads. bertAttnQueryBlock is
// QT*16, the query rows one workgroup owns, and bertTokenAlign is what a
// sequence is padded to for the planes -- the key block is KTIL*16 = 64, and
// 128 covers that and every other variant of the kernel, which is what
// zimage/dit pads to for the same reason.
const (
	bertAttnHeadDim    = 64
	bertAttnQueryBlock = 16
	bertTokenAlign     = 128
	// log2e is folded into q by the pack so the kernel's softmax is exp2,
	// which is one instruction on this ISA.
	log2e = 1.4426950408889634
)

// bertVariant is the GEMM rung the projections run on.
//
// M is the token count — fifty for an ordinary utterance — which is the same
// short-M regime the decoder's ladder picked a 32-wide tile for, and for the
// same reason: two tile rows of work cannot feed a wider one.
var bertVariant = convVariant{name: ConvKernel("gemm_reg32x32_w32"),
	spirv: shaders.DiTGEMMReg32x32TiledW32, bm: 32, bn: 32, wave: 32}

// NewGPUAlbert stages the shared weight group and every arena.
//
// maxTokens bounds the sequence and therefore the arenas; at the model's own
// limit of 512 the whole set is 25 MB, so there is no reason to size it per
// utterance the way the vocoder's stages are.
func NewGPUAlbert(dev *vk.Device, a *ALBERT, maxTokens int) (*GPUAlbert, error) {
	if maxTokens <= 0 || maxTokens > a.Config.MaxPositionEmbed {
		return nil, fmt.Errorf("kokoro: albert over %d tokens, the model allows %d",
			maxTokens, a.Config.MaxPositionEmbed)
	}
	if a.Layer == nil {
		return nil, fmt.Errorf("kokoro: albert has no layer group")
	}
	// The matrix-core attention has its head width compiled in, and a
	// mismatch would not fail -- it would read the wrong columns of q and
	// produce a plausible tensor. So it is checked here, once, against the
	// checkpoint's own config.
	if a.Config.HeadDim() != bertAttnHeadDim {
		return nil, fmt.Errorf("kokoro: albert head width is %d, shaders are built for %d",
			a.Config.HeadDim(), bertAttnHeadDim)
	}
	g := &GPUAlbert{
		dev: dev, cfg: a.Config, maxTok: maxTokens,
		pipes: map[string]*vk.ComputePipeline{},
	}
	g.ldHidden = a.Config.HiddenSize + bertPad
	g.ldFFN = a.Config.IntermediateSize + bertPad
	if err := g.alloc(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(a.Layer); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func (g *GPUAlbert) alloc() error {
	d := g.cfg.HiddenSize
	ff := g.cfg.IntermediateSize
	// The GEMM writes roundUp(T, BM) rows, so every fp32 destination carries
	// the padding and its tail is the product of zeros.
	rows := roundUp(g.maxTok, framePad)

	var off uint32
	take := func(n int) uint32 { o := off; off += uint32(n); return o }
	g.aX = take(rows * d)
	g.aQ = take(rows * d)
	g.aK = take(rows * d)
	g.aV = take(rows * d)
	g.aCtx = take(rows * d)
	g.aTmp = take(rows * d)
	g.aAttn = take(rows * d)
	g.aFFN = take(rows * ff)
	g.actElems = int(off)

	var offH uint32
	takeH := func(n int) uint32 { o := offH; offH += uint32(n); return o }
	g.hA = takeH(rows * g.ldHidden)
	g.hB = takeH(rows * g.ldFFN)
	// One fragment-tile plane per head for each of q, k and v. Sized for the
	// padded ceiling, so a shorter utterance uses a prefix of every plane and
	// the stride it is addressed at is its own tokPad rather than this one.
	planes := roundUp(g.maxTok, bertTokenAlign) * g.cfg.NumAttentionHeads * bertAttnHeadDim
	g.hQ = takeH(planes)
	g.hK = takeH(planes)
	g.hV = takeH(planes)
	g.hElems = int(offH)

	// fp32 weights: two LayerNorm affines and six biases.
	offW := uint32(0)
	takeW := func(n int) uint32 { o := offW; offW += uint32(n); return o }
	g.wAttnG, g.wAttnB = takeW(d), takeW(d)
	g.wOutG, g.wOutB = takeW(d), takeW(d)

	// fp16 fragment tiles: four [d, d] projections, then [ff, d] and [d, ff].
	halves := 0
	g.bQ = uint32(halves)
	halves += d * (d + bertPad)
	g.bK = uint32(halves)
	halves += d * (d + bertPad)
	g.bV = uint32(halves)
	halves += d * (d + bertPad)
	g.bDense = uint32(halves)
	halves += d * (d + bertPad)
	g.bFFN = uint32(halves)
	halves += ff * (d + bertPad)
	g.bFFNOut = uint32(halves)
	halves += d * (ff + bertPad)

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("kokoro: albert fp32 arena: %w", err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: albert fp16 arena: %w", err)
	}
	if g.wbuf, err = g.dev.NewBuffer(int(offW) * 4); err != nil {
		return fmt.Errorf("kokoro: albert fp32 weights: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("kokoro: albert weight bank: %w", err)
	}
	return nil
}

func (g *GPUAlbert) build() error {
	arenas := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"narrow", shaders.KokoroDNarrow},
		{"gelu", shaders.KokoroGELU},
		{"attn", shaders.KokoroBertAttn},
		{"pack", shaders.KokoroBertPackHD64},
		{"residual", shaders.ParakeetResidual},
		{"layernorm", shaders.ParakeetLayerNorm},
	} {
		mod, err := g.dev.NewShaderModule(s.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", s.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", s.name, err)
		}
		g.pipes[s.name] = pipe
	}
	s := spec
	s.RequiredSubgroupSize = bertVariant.wave
	ok, err := canPinWave(g.dev, bertVariant.wave)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("kokoro: this device cannot pin a %d-wide subgroup", bertVariant.wave)
	}
	mod, err := g.dev.NewShaderModule(bertVariant.spirv)
	if err != nil {
		return fmt.Errorf("kokoro: shader %s: %w", bertVariant.name, err)
	}
	g.mods = append(g.mods, mod)
	if g.gemm, err = g.dev.NewPipeline(mod, s); err != nil {
		return fmt.Errorf("kokoro: pipeline %s: %w", bertVariant.name, err)
	}
	// The matrix-core attention, on the same pinned wave. It shares the GEMM's
	// spec because it has the same requirement and for the same reason: the
	// kernel is one wave per workgroup by construction, so a driver that
	// packed two into a 64-wide subgroup would have them share the LDS the
	// softmax bookkeeping lives in.
	amod, err := g.dev.NewShaderModule(shaders.KokoroBertAttnWMMAHD64W32)
	if err != nil {
		return fmt.Errorf("kokoro: shader bert attention (wmma): %w", err)
	}
	g.mods = append(g.mods, amod)
	if g.pipes["attnwmma"], err = g.dev.NewPipeline(amod, s); err != nil {
		return fmt.Errorf("kokoro: pipeline bert attention (wmma): %w", err)
	}
	return nil
}

// stage writes the one weight group.
//
// A `Linear`'s weight is [Out, In] — torch's order — which is exactly what
// packConvB takes at one tap, so the projections and the feed-forward are
// staged by the same packer the vocoder's convolutions use.
func (g *GPUAlbert) stage(l *ALBERTLayer) error {
	d, ff := g.cfg.HiddenSize, g.cfg.IntermediateSize
	if l.Q.In != d || l.Q.Out != d || l.FFN.Out != ff || l.FFNOut.In != ff {
		return fmt.Errorf("kokoro: albert layer is %d->%d/%d, the config says %d/%d",
			l.Q.In, l.Q.Out, l.FFN.Out, d, ff)
	}
	bank := make([]uint16, g.bank.Size()/2)
	for _, p := range []struct {
		off     uint32
		lin     *Linear
		out, in int
	}{
		{g.bQ, l.Q, d, d},
		{g.bK, l.K, d, d},
		{g.bV, l.V, d, d},
		{g.bDense, l.Dense, d, d},
		{g.bFFN, l.FFN, ff, d},
		{g.bFFNOut, l.FFNOut, d, ff},
	} {
		packLinearB(bank[p.off:], p.lin.Weight, p.lin.Bias, p.out, p.in, p.in+bertPad)
	}
	g.bank.WriteUint16At(0, bank)

	w32 := make([]float32, g.wbuf.Size()/4)
	copy(w32[g.wAttnG:], l.AttnNorm.Weight)
	copy(w32[g.wAttnB:], l.AttnNorm.Bias)
	copy(w32[g.wOutG:], l.OutNorm.Weight)
	copy(w32[g.wOutB:], l.OutNorm.Bias)
	g.wbuf.WriteFloat32At(0, w32)

	// The fp16 arena: zero everywhere except the bias column of every row,
	// which is a 1. The narrow passes only ever write columns [0, width), so
	// both survive for the life of the object.
	h := make([]uint16, g.hElems)
	one := safetensors.F32ToF16(1)
	rows := roundUp(g.maxTok, framePad)
	for r := 0; r < rows; r++ {
		h[int(g.hA)+r*g.ldHidden+d] = one
		h[int(g.hB)+r*g.ldFFN+ff] = one
	}
	g.hbuf.WriteUint16At(0, h)
	return nil
}

// packLinearB lays a [Out, In] projection out as the fragment tiles
// B_LAYOUT=2 reads, with the bias as one extra column of K.
//
// K index `in` is the bias and the columns past it are zero, which is what
// makes `y = Wx + b` come out of a biasless GEMM — the A operand carries a 1
// in exactly that column. Everything beyond is left zero so the padded K
// contributes nothing.
func packLinearB(dst []uint16, w, bias []float32, out, in, kPad int) {
	kt := kPad / coopMatTile
	parallelFor(out, func(o int) {
		base := (o / coopMatTile) * kt * coopMatTile * coopMatTile
		lane := (o % coopMatTile) * coopMatTile
		put := func(k int, v float32) {
			dst[base+(k/coopMatTile)*coopMatTile*coopMatTile+lane+k%coopMatTile] =
				safetensors.F32ToF16(v)
		}
		for i := 0; i < in; i++ {
			put(i, w[o*in+i])
		}
		if bias != nil {
			put(in, bias[o])
		}
	})
}

// graph is one layer, which is every layer: the weight group is shared, so
// the twelve iterations differ in nothing at all.
func (g *GPUAlbert) graph() ([]vk.MultiDispatch, []string) {
	d := uint32(g.cfg.HiddenSize)
	ff := uint32(g.cfg.IntermediateSize)
	t := uint32(g.tokens)
	m := uint32(roundUp(g.tokens, bertVariant.bm))

	// The fragment-tile planes' stride, derived here rather than held as a
	// field: the pack and the attention address the planes as
	// `head * tokPad * headDim`, so the two have to agree, and the only way
	// they cannot disagree is if neither of them is told.
	tokPad := roundUp(g.tokens, bertTokenAlign)

	var dis []vk.MultiDispatch
	var kinds []string
	add := func(pipe *vk.ComputePipeline, kind string, gx, gy uint32, pc pushConstants) {
		dis = append(dis, vk.MultiDispatch{Pipeline: pipe, GroupsX: gx, GroupsY: gy,
			PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	// A narrow into the fp16 A operand. kokoro_act's NO_AFFINE build, whose
	// 2-D grid is what makes it work at a width of 768.
	narrow := func(kind string, src uint32, width, lda int, dst uint32) {
		pc := pushConstants{Dim: uint32(width), InOff: src, OutOff: dst,
			Aux1: uint32(width), LDA: uint32(lda)}
		add(g.pipes["narrow"], kind, groups(width, 256), t, pc)
	}
	// One projection. The bias rides in the weight arena and is added by the
	// residual pass or the norm that follows, except where it does not — see
	// bias() below.
	// K is the padded width, because the bias is the column past the data.
	gemm := func(kind string, a uint32, lda int, bOff, out uint32, n, k int) {
		pc := pushConstants{InOff: a, OutOff: out, BOff: bOff,
			GemmM: m, GemmN: uint32(n), GemmK: uint32(k + bertPad), LDA: uint32(lda)}
		dis = append(dis, vk.MultiDispatch{Pipeline: g.gemm,
			GroupsX: uint32(n / bertVariant.bn), GroupsY: m / uint32(bertVariant.bm),
			PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	// Attention.
	narrow("narrow x", g.aX, int(d), g.ldHidden, g.hA)
	gemm("q", g.hA, g.ldHidden, g.bQ, g.aQ, int(d), int(d))
	gemm("k", g.hA, g.ldHidden, g.bK, g.aK, int(d), int(d))
	gemm("v", g.hA, g.ldHidden, g.bV, g.aV, int(d), int(d))
	heads := uint32(g.cfg.NumAttentionHeads)
	scale := float32(1 / math.Sqrt(float64(g.cfg.HeadDim())))
	if g.ScalarAttn {
		pcAttn := pushConstants{
			InOff: g.aQ, KOff: g.aK, VOff: g.aV, OutOff: g.aCtx,
			Tokens: t, Dim: d, Heads: heads,
			HeadDim: uint32(g.cfg.HeadDim()),
			Scale:   math.Float32bits(scale),
		}
		add(g.pipes["attn"], "attention", heads, t, pcAttn)
	} else {
		// Three packs and the kernel. Each pack turns one fp32 [T, 768]
		// tensor into 12 per-head planes of 16x16 fp16 fragment tiles, which
		// is the layout a coopMatLoad reads with full coverage --
		// dit_pack_f16.comp is where the 1.56x-against-an-order-of-magnitude
		// measurement that justifies the pass lives.
		//
		// The grid covers tokPad rather than t: the kernel's key block is 64
		// rows, so it reads whole tiles past the sequence, and a pad tile has
		// to hold a finite zero rather than whatever the last utterance left
		// there.
		pad := uint32(tokPad)
		for _, s := range []struct {
			kind     string
			src, dst uint32
			mode     uint32
			scale    float32
		}{
			// q carries the softmax scale and log2(e); v is packed
			// transposed, because p.v reduces over the token axis.
			{"pack q", g.aQ, g.hQ, 0, scale * float32(log2e)},
			{"pack k", g.aK, g.hK, 0, 1},
			{"pack v", g.aV, g.hV, 1, 1},
		} {
			add(g.pipes["pack"], s.kind, groups(tokPad, coopMatTile), heads,
				pushConstants{
					InOff: s.src, OutOff: s.dst,
					Tokens: t, Dim: d,
					Aux0: s.mode, Aux1: pad,
					Scale: math.Float32bits(s.scale),
				})
		}
		add(g.pipes["attnwmma"], "attention",
			groups(g.tokens, bertAttnQueryBlock), heads,
			pushConstants{
				InOff: g.hQ, KOff: g.hK, VOff: g.hV, OutOff: g.aCtx,
				Tokens: t, Dim: d, Heads: heads, Aux1: pad,
			})
	}
	narrow("narrow ctx", g.aCtx, int(d), g.ldHidden, g.hA)
	gemm("dense", g.hA, g.ldHidden, g.bDense, g.aTmp, int(d), int(d))
	add(g.pipes["residual"], "residual attn", t, 1, pushConstants{
		Tokens: t, Dim: d, InOff: g.aX, OutOff: g.aTmp, Scale: math.Float32bits(1)})
	add(g.pipes["layernorm"], "attn norm", t, 1, pushConstants{
		Tokens: t, Dim: d, InOff: g.aTmp, OutOff: g.aAttn,
		WOff: g.wAttnG, Aux0: g.wAttnB, Eps: math.Float32bits(float32(albertLayerEps))})

	// Feed-forward.
	narrow("narrow attn", g.aAttn, int(d), g.ldHidden, g.hA)
	gemm("ffn", g.hA, g.ldHidden, g.bFFN, g.aFFN, int(ff), int(d))
	pcGELU := pushConstants{Dim: ff, InOff: g.aFFN, OutOff: g.hB,
		Aux1: ff, LDA: uint32(g.ldFFN)}
	add(g.pipes["gelu"], "gelu", groups(int(ff), 256), t, pcGELU)
	gemm("ffn out", g.hB, g.ldFFN, g.bFFNOut, g.aTmp, int(d), int(ff))
	add(g.pipes["residual"], "residual ffn", t, 1, pushConstants{
		Tokens: t, Dim: d, InOff: g.aAttn, OutOff: g.aTmp, Scale: math.Float32bits(1)})
	add(g.pipes["layernorm"], "out norm", t, 1, pushConstants{
		Tokens: t, Dim: d, InOff: g.aTmp, OutOff: g.aX,
		WOff: g.wOutG, Aux0: g.wOutB, Eps: math.Float32bits(float32(albertLayerEps))})
	return dis, kinds
}

// Apply runs the whole encoder over the embedding stack's output, which the
// host computes: it is a table lookup, two adds and a 128->768 projection over
// fifty rows, and none of that is worth a dispatch.
func (g *GPUAlbert) Apply(hidden *Mat) (*Mat, error) {
	d := g.cfg.HiddenSize
	if hidden.Cols != d {
		return nil, fmt.Errorf("kokoro: albert takes %d channels, got %d", d, hidden.Cols)
	}
	if hidden.Rows > g.maxTok {
		return nil, fmt.Errorf("kokoro: %d tokens against an arena for %d", hidden.Rows, g.maxTok)
	}
	g.tokens = hidden.Rows
	g.abuf.WriteFloat32At(int(g.aX), hidden.Data)

	one, _ := g.graph()
	var dis []vk.MultiDispatch
	for i := 0; i < g.cfg.NumHiddenLayers; i++ {
		dis = append(dis, one...)
	}
	if err := submit(dis); err != nil {
		return nil, err
	}
	return &Mat{Rows: g.tokens, Cols: d,
		Data: g.abuf.ReadFloat32At(int(g.aX), g.tokens*d)}, nil
}

// Profile times every dispatch of one layer on the device.
func (g *GPUAlbert) Profile(tokens int) ([]Stage, error) {
	g.tokens = tokens
	dis, kinds := g.graph()
	out := make([]Stage, 0, len(dis))
	for i := range dis {
		t, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 4, true)
		if err != nil {
			return nil, err
		}
		out = append(out, Stage{Kind: kinds[i], Time: t.Seconds() / 4})
	}
	return out, nil
}

// Destroy releases every Vulkan object.
func (g *GPUAlbert) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	if g.gemm != nil {
		g.gemm.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.mods, g.gemm = nil, nil, nil
	g.wbuf, g.abuf, g.hbuf, g.bank = nil, nil, nil, nil
}
