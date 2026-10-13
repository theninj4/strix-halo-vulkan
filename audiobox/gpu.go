package audiobox

// The device graph (MUSIC.md A14c). A batch is up to maxWin windows; each
// window is framesPad rows of every activation arena.
//
// The feature extractor runs a window at a time over scratch arenas sized
// for one window: the first conv in fp32 (audiobox_conv0), its GroupNorm's
// per-channel statistics over the window's 31,999 frames (audiobox_colstats,
// two passes and Chan's combine), the norm and GELU narrowed to fp16
// (audiobox_act), then six strided convolutions as dit_gemm's implicit
// im2col (kokoro's A_CONV builds: the A row stride is stride·512, the tap
// stride one input row), each followed by the GELU narrow. The last conv's
// GELU stays fp32 for the LayerNorm, which narrows straight into the
// projection's operand.
//
// The encoder runs over the whole batch: every projection is one GEMM over
// all its rows with the bias as an extra K column (kokoro's PL-BERT trick:
// the A operand carries a 1 there), the post-LN layer as GPUAlbert has it,
// and per window the gated relative-position plane (audiobox_relbias), the
// three packs and dit_attention_wmma's REL_BIAS build at head width 64 with
// the window's valid frames as its key count. After the positional conv and
// after every layer, each window's mean over its valid frames is taken
// (audiobox_colstats), which is all the heads read.

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

type pushConstants struct {
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

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

const (
	tile = 16
	// biasPad is an fp16 operand's extra columns: the first holds the 1 the
	// bias column of B multiplies, and 64 keeps K a multiple of every BK.
	biasPad = 64
	// statChunk is audiobox_colstats' rows a partial.
	statChunk = 1024
	// log2e is folded into q and the bias plane: the kernel's softmax is
	// exp2.
	log2e = 1.4426950408889634
	// perSubmit bounds a command buffer, as kokoro's graphs do.
	perSubmit = 240
)

// gemmKernel is a dit_gemm build.
type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	wave   uint32 // pinned subgroup size, 0 for the driver's
}

var (
	// projKernel runs every projection: the LDS-staged build, the repo's
	// fastest at a batch's thousands of rows (KERNELS.md G2).
	projKernel = gemmKernel{"proj", shaders.DiTGEMMWG128x256LDSW32, 128, 256, 32}
	// convKernel is kokoro's A_CONV=1 build at 64x64.
	convKernel = gemmKernel{"conv", shaders.KokoroConvReg64, 64, 64, 0}
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// GPU is the encoder resident on a device.
type GPU struct {
	dev    *vk.Device
	maxWin int

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline

	// fp32 activations
	aWav, aF0, aF1, aStats, aGN uint32
	aX, aQ, aK, aV, aTmp, aFFN  uint32
	aBias, aPool                uint32
	// fp16 activations
	hF, hP, hA, hB, hQ, hK, hV uint32
	// fp32 weights
	wConv0, wGNG, wGNB, wFELNG, wFELNB uint32
	wPos, wPosB, wEncLNG, wEncLNB      uint32
	wTable                             uint32
	wLN                                [layers][4]uint32 // ln1 g, b, ln2 g, b
	wGrep                              [layers]uint32
	// fp16 fragment-tiled weights
	bConv [6]uint32
	bProj uint32
	bL    [layers][6]uint32 // q, k, v, o, fc1, fc2
}

// newGPU stages w for batches of up to maxWin windows.
func newGPU(dev *vk.Device, w *weights, maxWin int) (*GPU, error) {
	g := &GPU{dev: dev, maxWin: maxWin, pipes: map[string]*vk.ComputePipeline{}}
	for _, step := range []func() error{g.alloc, g.build, func() error { return g.stage(w) }} {
		if err := step(); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

func (g *GPU) alloc() error {
	R := g.maxWin * framesPad
	var a uint32
	take := func(n int) uint32 { o := a; a += uint32((n + 63) &^ 63); return o }
	g.aWav = take(g.maxWin * WindowSamples)
	g.aF0 = take(roundUp(convLens[0], 256) * convDim)
	g.aF1 = take(roundUp(convLens[1], 256) * convDim)
	g.aStats = take(roundUp(convLens[0], statChunk) / statChunk * 3 * convDim)
	g.aGN = take(2 * convDim)
	g.aX = take(R * hidden)
	g.aQ = take(R * hidden)
	g.aK = take(R * hidden)
	g.aV = take(R * hidden)
	g.aTmp = take(R * hidden)
	g.aFFN = take(R * ffn)
	g.aBias = take(g.maxWin * heads * framesPad * framesPad)
	g.aPool = take((layers + 1) * g.maxWin * 3 * hidden)
	acts := int(a)

	var h uint32
	takeH := func(n int) uint32 { o := h; h += uint32((n + 63) &^ 63); return o }
	// The conv operand: the first conv's frames, and slack for the tile
	// rows the A_CONV GEMM reads past the last output's taps.
	g.hF = takeH((convLens[0] + 256) * convDim)
	g.hP = takeH(R * (convDim + biasPad))
	g.hA = takeH(R * (hidden + biasPad))
	g.hB = takeH(R * (ffn + biasPad))
	plane := heads * framesPad * headDim
	g.hQ, g.hK, g.hV = takeH(plane), takeH(plane), takeH(plane)
	halves := int(h)

	var err error
	if g.abuf, err = g.dev.NewHostCachedBuffer(acts * 4); err != nil {
		return fmt.Errorf("audiobox: fp32 arena (%d MB): %w", acts>>18, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("audiobox: fp16 arena (%d MB): %w", halves>>19, err)
	}
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, wave uint32) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("audiobox: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	spec := vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank},
		PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})), RequiredSubgroupSize: wave}
	p, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("audiobox: pipeline %s: %w", name, err)
	}
	g.pipes[name] = p
	return nil
}

// build sizes the weight arenas (the pipelines bind them) and makes every
// pipeline.
func (g *GPU) build() error {
	var w uint32
	takeW := func(n int) uint32 { o := w; w += uint32((n + 63) &^ 63); return o }
	g.wConv0 = takeW(convDim * 10)
	g.wGNG, g.wGNB = takeW(convDim), takeW(convDim)
	g.wFELNG, g.wFELNB = takeW(convDim), takeW(convDim)
	g.wPos = takeW(hidden * hidden / posGroups * posKernel)
	g.wPosB = takeW(hidden)
	g.wEncLNG, g.wEncLNB = takeW(hidden), takeW(hidden)
	g.wTable = takeW(heads * (2*framesPad - 1))
	for l := range g.wLN {
		for i := range g.wLN[l] {
			g.wLN[l][i] = takeW(hidden)
		}
		g.wGrep[l] = takeW(8*headDim + 8 + heads)
	}
	var b uint32
	takeB := func(n, k int) uint32 { o := b; b += uint32(n * k); return o }
	for i := range g.bConv {
		g.bConv[i] = takeB(convDim, convLayers[i+1][0]*convDim)
	}
	g.bProj = takeB(hidden, convDim+biasPad)
	for l := range g.bL {
		for i := 0; i < 4; i++ {
			g.bL[l][i] = takeB(hidden, hidden+biasPad)
		}
		g.bL[l][4] = takeB(ffn, hidden+biasPad)
		g.bL[l][5] = takeB(hidden, ffn+biasPad)
	}
	var err error
	if g.wbuf, err = g.dev.NewBuffer(int(w) * 4); err != nil {
		return fmt.Errorf("audiobox: fp32 weights: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(int(b) * 2); err != nil {
		return fmt.Errorf("audiobox: fp16 weights: %w", err)
	}
	for _, s := range []struct {
		name  string
		spirv []byte
		wave  uint32
	}{
		{"conv0", shaders.AudioboxConv0, 0},
		{"colstats", shaders.AudioboxColStats, 0},
		{"act", shaders.AudioboxAct, 0},
		{"posconv", shaders.AudioboxPosConv, 0},
		{"relbias", shaders.AudioboxRelBias, 0},
		{"layernorm", shaders.ParakeetLayerNorm, 0},
		{"layernorm16", shaders.ParakeetLayerNormF16, 0},
		{"residual", shaders.ParakeetResidual, 0},
		{"pack", shaders.KokoroBertPackHD64, 0},
		{"attn", shaders.AudioboxAttnHD64W32, 32},
		{projKernel.name, projKernel.spirv, projKernel.wave},
		{convKernel.name, convKernel.spirv, convKernel.wave},
	} {
		if err := g.pipeline(s.name, s.spirv, s.wave); err != nil {
			return err
		}
	}
	return nil
}

// tileLinear lays a [out][in] weight out as the fragment tiles B_LAYOUT=2
// reads (tile (n/16, k/16) holds (k, n) at (n%16)·16 + k%16, tiles
// k-fastest) over a K of kPad, with the bias, when there is one, at K index
// in.
func tileLinear(dst []uint16, w, bias []float32, out, in, kPad int) {
	kt := kPad / tile
	parallel(out, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			base := (o / tile) * kt * tile * tile
			lane := (o % tile) * tile
			put := func(k int, v float32) {
				dst[base+(k/tile)*tile*tile+lane+k%tile] = safetensors.F32ToF16(v)
			}
			for i := 0; i < in; i++ {
				put(i, w[o*in+i])
			}
			if bias != nil {
				put(in, bias[o])
			}
		}
	})
}

func (g *GPU) stage(w *weights) error {
	w32 := make([]float32, g.wbuf.Size()/4)
	put := func(off uint32, v []float32) { copy(w32[off:], v) }
	put(g.wConv0, w.conv0)
	put(g.wGNG, w.gnG)
	put(g.wGNB, w.gnB)
	put(g.wFELNG, w.feLNG)
	put(g.wFELNB, w.feLNB)
	// pos_conv [768 c][48 j][128 k] → [16 g][48 j][128 k][48 c].
	cin := hidden / posGroups
	pos := w32[g.wPos:]
	for c := 0; c < hidden; c++ {
		grp, cc := c/cin, c%cin
		for j := 0; j < cin; j++ {
			for k := 0; k < posKernel; k++ {
				pos[((grp*cin+j)*posKernel+k)*cin+cc] = w.posW[(c*cin+j)*posKernel+k]
			}
		}
	}
	put(g.wPosB, w.posB)
	put(g.wEncLNG, w.encLNG)
	put(g.wEncLNB, w.encLNB)
	// The bias table over every offset a padded window has, in log2 units.
	n := framesPad
	for h := 0; h < heads; h++ {
		for d := -(n - 1); d <= n-1; d++ {
			w32[int(g.wTable)+h*(2*n-1)+d+n-1] = w.relBias[bucket(d)*heads+h] * log2e
		}
	}
	for l, L := range w.layer {
		put(g.wLN[l][0], L.ln1G)
		put(g.wLN[l][1], L.ln1B)
		put(g.wLN[l][2], L.ln2G)
		put(g.wLN[l][3], L.ln2B)
		put(g.wGrep[l], L.grepW)
		put(g.wGrep[l]+8*headDim, L.grepB)
		put(g.wGrep[l]+8*headDim+8, L.grepA)
	}
	g.wbuf.WriteFloat32At(0, w32)

	bank := make([]uint16, g.bank.Size()/2)
	for i := range g.bConv {
		// [512 o][512 c][k j] → K index j·512 + c, as A_CONV walks taps.
		k := convLayers[i+1][0]
		src := w.conv[i]
		re := make([]float32, len(src))
		for o := 0; o < convDim; o++ {
			for c := 0; c < convDim; c++ {
				for j := 0; j < k; j++ {
					re[o*k*convDim+j*convDim+c] = src[(o*convDim+c)*k+j]
				}
			}
		}
		tileLinear(bank[g.bConv[i]:], re, nil, convDim, k*convDim, k*convDim)
	}
	tileLinear(bank[g.bProj:], w.projW, w.projB, hidden, convDim, convDim+biasPad)
	for l, L := range w.layer {
		for i, p := range []struct {
			w, b    []float32
			out, in int
		}{{L.q, L.qB, hidden, hidden}, {L.k, L.kB, hidden, hidden}, {L.v, L.vB, hidden, hidden},
			{L.o, L.oB, hidden, hidden}, {L.fc1, L.fc1B, ffn, hidden}, {L.fc2, L.fc2B, hidden, ffn}} {
			tileLinear(bank[g.bL[l][i]:], p.w, p.b, p.out, p.in, p.in+biasPad)
		}
	}
	g.bank.WriteUint16At(0, bank)

	// The fp16 arena: zero, and a 1 in every operand row's bias column. The
	// passes that write these operands write only the columns before it.
	hz := make([]uint16, g.hbuf.Size()/2)
	one := safetensors.F32ToF16(1)
	for r := 0; r < g.maxWin*framesPad; r++ {
		hz[int(g.hP)+r*(convDim+biasPad)+convDim] = one
		hz[int(g.hA)+r*(hidden+biasPad)+hidden] = one
		hz[int(g.hB)+r*(ffn+biasPad)+ffn] = one
	}
	g.hbuf.WriteUint16At(0, hz)
	g.abuf.Zero()
	return nil
}

// Destroy releases every Vulkan object.
func (g *GPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.mods = nil, nil
	g.wbuf, g.abuf, g.hbuf, g.bank = nil, nil, nil, nil
}

// graph records a batch: windows wins (already in aWav), each with valid
// frames valid[b], through the extractor and stopAt encoder layers (layers
// for the whole encoder; −1 stops after the projection). It pools the
// hidden states as it goes.
type graph struct {
	g     *GPU
	d     []vk.MultiDispatch
	kinds []string
}

func (gr *graph) add(pipe, kind string, gx, gy uint32, pc pushConstants) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: gx, GroupsY: gy,
		PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
}

// gemm is C[m][n] = A[m][k]·B[n][k] on kernel k, m padded to its tile.
func (gr *graph) gemm(kern gemmKernel, kind string, a, lda, b, c uint32, m, n, k int, aux [3]uint32) {
	mp := roundUp(m, kern.bm)
	gr.add(kern.name, kind, uint32(n/kern.bn), uint32(mp/kern.bm), pushConstants{
		InOff: a, OutOff: c, BOff: b, GemmM: uint32(mp), GemmN: uint32(n), GemmK: uint32(k), LDA: lda,
		Aux0: aux[0], Aux1: aux[1], Aux2: aux[2]})
}

func f32bits(f float64) uint32 { return math.Float32bits(float32(f)) }

// extract records window b's feature extractor and its projection's
// operand rows.
func (gr *graph) extract(b int) {
	g := gr.g
	L := convLens
	gr.add("conv0", "conv0", groups(L[0]*convDim, 256), 1, pushConstants{
		InOff: g.aWav + uint32(b*WindowSamples), OutOff: g.aF0, WOff: g.wConv0, Tokens: uint32(L[0])})
	chunks := roundUp(L[0], statChunk) / statChunk
	gr.add("colstats", "groupnorm stats", convDim/64, uint32(chunks), pushConstants{
		InOff: g.aF0, OutOff: g.aStats, Tokens: uint32(L[0]), Dim: convDim, Aux1: statChunk})
	gr.add("colstats", "groupnorm combine", convDim/64, 1, pushConstants{
		InOff: g.aStats, OutOff: g.aGN, Dim: convDim, Aux0: uint32(chunks), Aux2: 1})
	gr.add("act", "groupnorm gelu", convDim/256, uint32(L[0]), pushConstants{
		InOff: g.aF0, OutOff: g.hF, Tokens: uint32(L[0]), Dim: convDim, LDA: convDim,
		Aux1: g.aGN, WOff: g.wGNG, Aux0: g.wGNB, Eps: f32bits(lnEps), Aux2: 3})
	for i := 1; i < 7; i++ {
		k, s := convLayers[i][0], convLayers[i][1]
		gr.gemm(convKernel, fmt.Sprintf("conv%d", i), g.hF, uint32(s*convDim), g.bConv[i-1], g.aF1,
			L[i], convDim, k*convDim, [3]uint32{convDim, convDim, 9})
		if i < 6 {
			gr.add("act", "gelu", convDim/256, uint32(L[i]), pushConstants{
				InOff: g.aF1, OutOff: g.hF, Tokens: uint32(L[i]), Dim: convDim, LDA: convDim, Aux2: 2})
		}
	}
	// The last conv's GELU in fp32, in place, then its LayerNorm into the
	// projection's operand. aux2 4 writes fp32 back to act.
	gr.add("act", "gelu", convDim/256, uint32(L[6]), pushConstants{
		InOff: g.aF1, OutOff: g.aF1, Tokens: uint32(L[6]), Dim: convDim, LDA: convDim, Aux2: 2 | 4})
	gr.add("layernorm16", "feature norm", uint32(L[6]), 1, pushConstants{
		InOff: g.aF1, OutOff: g.hP + uint32(b*framesPad*(convDim+biasPad)), Tokens: uint32(L[6]), Dim: convDim,
		LDA: convDim + biasPad, WOff: g.wFELNG, Aux0: g.wFELNB, Eps: f32bits(lnEps)})
}

// pool records window b's mean over its valid frames of the hidden state
// in aX, into pooled slot l.
func (gr *graph) pool(l, b int, valid []int) {
	g := gr.g
	gr.add("colstats", "pool", hidden/64, 1, pushConstants{
		InOff: g.aX + uint32(b*framesPad*hidden), OutOff: g.aPool + uint32((l*g.maxWin+b)*3*hidden),
		Tokens: uint32(valid[b]), Dim: hidden, Aux1: statChunk})
}

// encode records the projection, the positional conv and stopAt layers.
func (gr *graph) encode(valid []int, stopAt int) {
	g := gr.g
	W := len(valid)
	R := W * framesPad
	gr.gemm(projKernel, "proj", g.hP, convDim+biasPad, g.bProj, g.aX, R, hidden, convDim+biasPad, [3]uint32{})
	if stopAt < 0 {
		return
	}
	for b := 0; b < W; b++ {
		o := uint32(b * framesPad * hidden)
		gr.add("posconv", "pos conv", groups(Frames, 64), posGroups, pushConstants{
			InOff: g.aX + o, OutOff: g.aTmp + o, Tokens: Frames, Aux1: uint32(valid[b]),
			WOff: g.wPos, Aux0: g.wPosB})
	}
	ln := func(kind string, in, out, gam, bet uint32) {
		gr.add("layernorm", kind, uint32(R), 1, pushConstants{InOff: in, OutOff: out, Tokens: uint32(R),
			Dim: hidden, WOff: gam, Aux0: bet, Eps: f32bits(lnEps)})
	}
	ln("encoder norm", g.aTmp, g.aX, g.wEncLNG, g.wEncLNB)
	for b := 0; b < W; b++ {
		gr.pool(0, b, valid)
	}
	lda := uint32(hidden + biasPad)
	narrow := func(src uint32) {
		gr.add("act", "narrow", hidden/256, uint32(R), pushConstants{InOff: src, OutOff: g.hA, Tokens: uint32(R),
			Dim: hidden, LDA: lda})
	}
	qScale := float32(1 / math.Sqrt(headDim) * log2e)
	for l := 0; l < stopAt; l++ {
		bl := g.bL[l]
		narrow(g.aX)
		gr.gemm(projKernel, "q", g.hA, lda, bl[0], g.aQ, R, hidden, hidden+biasPad, [3]uint32{})
		gr.gemm(projKernel, "k", g.hA, lda, bl[1], g.aK, R, hidden, hidden+biasPad, [3]uint32{})
		gr.gemm(projKernel, "v", g.hA, lda, bl[2], g.aV, R, hidden, hidden+biasPad, [3]uint32{})
		for b := 0; b < W; b++ {
			o := uint32(b * framesPad * hidden)
			bias := g.aBias + uint32(b*heads*framesPad*framesPad)
			gr.add("relbias", "rel bias", framesPad, heads, pushConstants{InOff: g.aX + o, OutOff: bias,
				WOff: g.wGrep[l], Aux0: g.wTable, Aux1: framesPad, Tokens: Frames})
			for _, p := range []struct {
				kind     string
				src, dst uint32
				mode     uint32
				scale    float32
			}{{"pack q", g.aQ + o, g.hQ, 0, qScale}, {"pack k", g.aK + o, g.hK, 0, 1}, {"pack v", g.aV + o, g.hV, 1, 1}} {
				gr.add("pack", p.kind, groups(framesPad, tile), heads, pushConstants{InOff: p.src, OutOff: p.dst,
					Tokens: Frames, Dim: hidden, Aux0: p.mode, Aux1: framesPad, Scale: math.Float32bits(p.scale)})
			}
			gr.add("attn", "attention", groups(Frames, tile), heads, pushConstants{
				InOff: g.hQ, KOff: g.hK, VOff: g.hV, OutOff: g.hA + uint32(b*framesPad)*lda, LDA: lda,
				Tokens: uint32(valid[b]), KStride: Frames, Dim: hidden, Heads: heads, Aux0: bias, Aux1: framesPad})
		}
		gr.gemm(projKernel, "out", g.hA, lda, bl[3], g.aTmp, R, hidden, hidden+biasPad, [3]uint32{})
		gr.add("residual", "residual", uint32(R), 1, pushConstants{Tokens: uint32(R), Dim: hidden,
			InOff: g.aX, OutOff: g.aTmp, Scale: math.Float32bits(1)})
		ln("attn norm", g.aTmp, g.aX, g.wLN[l][0], g.wLN[l][1])
		narrow(g.aX)
		gr.gemm(projKernel, "fc1", g.hA, lda, bl[4], g.aFFN, R, ffn, hidden+biasPad, [3]uint32{})
		gr.add("act", "gelu", ffn/256, uint32(R), pushConstants{InOff: g.aFFN, OutOff: g.hB, Tokens: uint32(R),
			Dim: ffn, LDA: ffn + biasPad, Aux2: 2})
		gr.gemm(projKernel, "fc2", g.hB, ffn+biasPad, bl[5], g.aTmp, R, hidden, ffn+biasPad, [3]uint32{})
		gr.add("residual", "residual", uint32(R), 1, pushConstants{Tokens: uint32(R), Dim: hidden,
			InOff: g.aX, OutOff: g.aTmp, Scale: math.Float32bits(1)})
		ln("out norm", g.aTmp, g.aX, g.wLN[l][2], g.wLN[l][3])
		for b := 0; b < W; b++ {
			gr.pool(l+1, b, valid)
		}
	}
}

func (gr *graph) submit() (time.Duration, error) {
	var total time.Duration
	for d := gr.d; len(d) > 0; {
		n := min(len(d), perSubmit)
		t, err := vk.DispatchMultiTimed(d[:n], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("audiobox: %w", err)
		}
		total += t
		d = d[n:]
	}
	return total, nil
}

// run scores a batch of windows (each WindowSamples of 16 kHz audio, the
// tail zero-padded) whose valid frames are valid, through stopAt layers,
// and returns each window's pooled hidden states.
func (g *GPU) run(wins [][]float32, valid []int, stopAt int) ([][layers + 1][]float32, time.Duration, error) {
	W := len(wins)
	if W == 0 || W > g.maxWin || len(valid) != W {
		return nil, 0, fmt.Errorf("audiobox: a batch of %d windows; 1 to %d", W, g.maxWin)
	}
	for b, w := range wins {
		if len(w) != WindowSamples {
			return nil, 0, fmt.Errorf("audiobox: window %d is %d samples", b, len(w))
		}
		g.abuf.WriteFloat32At(int(g.aWav)+b*WindowSamples, w)
	}
	gr := &graph{g: g}
	for b := 0; b < W; b++ {
		gr.extract(b)
	}
	gr.encode(valid, stopAt)
	took, err := gr.submit()
	if err != nil {
		return nil, took, err
	}
	out := make([][layers + 1][]float32, W)
	for b := range out {
		for l := 0; l <= max(stopAt, -1) && l <= layers; l++ {
			out[b][l] = g.abuf.ReadFloat32At(int(g.aPool)+((l*g.maxWin+b)*3+1)*hidden, hidden)
		}
	}
	return out, took, nil
}
