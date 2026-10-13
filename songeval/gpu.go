package songeval

// The device graph (MUSIC.md A14d).
//
// MuQ's front end runs in chunks of chunkTok output tokens with a halo of
// halo tokens each side, so its activations (64 frequency rows x 512
// channels at 50 Hz) stay a few hundred MB whatever the song's length. A
// chunk computes every level over its local time range; outputs near the
// local edges read past it and are wrong, but they are inside the halo and
// never used. Positions outside the song are zeroed at every level, which
// is each convolution's padding. Levels: the mel (100 Hz, origin 4a − 12),
// block 0 (50 Hz, 2a − 6), block 1 (25 Hz, a − 3), for the chunk's first
// token a: the origins halve exactly, so a level's local index maps to the
// next's as the global ones do.
//
// The 512 -> 512 convolutions are dit_gemm's A_CONV=3 (one output frequency
// row a dispatch over channel-last fp16 with a zero border), the
// 1-channel ones and the epilogues the muq_* passes. Block 1's output goes
// straight into the Linear's operand, which writes the chunk's rows of the
// conformer's input.
//
// The conformer and SongEval's head run over the whole song: projections on
// the LDS-staged GEMM with the bias as an extra K column, attention on the
// WMMA flash kernels (16 heads of 64, then 8 of 128), and the head's
// time-mean and time-max on the host.

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
	tile      = 16
	biasPad   = 64
	chunkTok  = 512
	halo      = 3
	log2e     = 1.4426950408889634
	perSubmit = 240
	tokAlign  = 128 // GEMM rows and attention planes
	linK      = convDim * nMels / 4
)

// The chunk's local geometry at each level.
const (
	melLocal = 4*chunkTok + 8*halo // 100 Hz frames
	t1Local  = 2*chunkTok + 4*halo // block 0's 50 Hz times
	t2Local  = chunkTok + 2*halo   // block 1's 25 Hz times
	f1       = nMels / 2           // block 0's frequency rows
	f2       = nMels / 4
)

type gemmKernel struct {
	name   string
	spirv  []byte
	bm, bn int
	wave   uint32
}

var (
	projKernel = gemmKernel{"proj", shaders.DiTGEMMWG128x256LDSW32, 128, 256, 32}
	convKernel = gemmKernel{"conv2d", shaders.MuQConv2DReg64, 64, 64, 0}
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// The chunk's padded extents: GEMM rows (to the conv tile) and bordered
// time pitches, wide enough for the stride-2 reads of the next level's
// padded rows.
var (
	mpad1 = roundUp(t1Local, 64)
	mpad2 = roundUp(t2Local, 64)
	tp1   = max(mpad1, 2*mpad2) + 4
	tp2   = mpad2 + 4
)

// GPU is MuQ's first six layers and SongEval's head, staged.
type GPU struct {
	dev    *vk.Device
	maxTok int // the song's tokens at most, 25 a second

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline

	// fp32 activations: the chunk's
	aMel, aSC1, aY1, aY2a, aY2b, aY2c uint32
	// and the song's
	aX, aX2, aQ, aK, aV, aTmp, aFF, aCos, aSin uint32
	// fp16
	hMid1, hOut1, hMid2, hLin uint32
	hA, hA2, hB, hQ, hK, hV   uint32
	// fp32 weights
	wIn                        uint32 // conv1, b1, conv3, b3 of block 0
	wB0c2, wB1c1, wB1c2, wB1c3 uint32 // biases
	wLN                        [Layers][10]uint32
	wDW                        [Layers]uint32
	// fp16 tiled weights
	bB0c2, bB1c1, bB1c2, bB1c3, bLin uint32
	bL                               [Layers][9]uint32 // ffn1 1, 2; q k v o; pw1 pw2; ffn2 1
	ffn2Out                          []uint32          // ffn2 2, by layer
	bFfd1, bFfd2                     uint32
	bIn                              [4][3]uint32
	bOut                             [4]uint32
}

func newGPU(dev *vk.Device, w *weights, maxSeconds float64) (*GPU, error) {
	g := &GPU{dev: dev, pipes: map[string]*vk.ComputePipeline{}}
	g.maxTok = int(math.Ceil(maxSeconds*SampleRate/hop/4)) + 1
	for _, step := range []func() error{g.alloc, g.build, func() error { return g.stage(w) }} {
		if err := step(); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

func (g *GPU) alloc() error {
	R := roundUp(roundUp(g.maxTok, chunkTok), tokAlign)
	var a uint32
	take := func(n int) uint32 { o := a; a += uint32((n + 63) &^ 63); return o }
	g.aMel = take(nMels * melLocal)
	g.aSC1 = take(f1 * mpad1 * convDim)
	g.aY1 = take(f1 * mpad1 * convDim)
	g.aY2a = take(f2 * mpad2 * convDim)
	g.aY2b = take(f2 * mpad2 * convDim)
	g.aY2c = take(f2 * mpad2 * convDim)
	g.aX, g.aX2 = take(R*hidden), take(R*hidden)
	g.aQ, g.aK, g.aV, g.aTmp = take(R*hidden), take(R*hidden), take(R*hidden), take(R*hidden)
	g.aFF = take(R * ffn)
	g.aCos, g.aSin = take(R*headDim/2), take(R*headDim/2)
	acts := int(a)

	var h uint32
	takeH := func(n int) uint32 { o := h; h += uint32((n + 63) &^ 63); return o }
	// Bordered fp16 planes, with a row of slack: the stride-2 GEMM's padded
	// rows read past the last frequency row.
	g.hMid1 = takeH((f1+3)*tp1*convDim + 4*convDim)
	g.hOut1 = takeH((f1+3)*tp1*convDim + 4*convDim)
	g.hMid2 = takeH((f2+3)*tp2*convDim + 4*convDim)
	g.hLin = takeH(chunkTok * (linK + biasPad))
	g.hA = takeH(R * (hidden + biasPad))
	g.hA2 = takeH(R * (hidden + biasPad))
	g.hB = takeH(R * (ffn + biasPad))
	plane := R * hidden
	g.hQ, g.hK, g.hV = takeH(plane), takeH(plane), takeH(plane)
	halves := int(h)

	var err error
	if g.abuf, err = g.dev.NewHostCachedBuffer(acts * 4); err != nil {
		return fmt.Errorf("songeval: fp32 arena (%d MB): %w", acts>>18, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("songeval: fp16 arena (%d MB): %w", halves>>19, err)
	}
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, wave uint32) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("songeval: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	p, err := g.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank},
		PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})), RequiredSubgroupSize: wave})
	if err != nil {
		return fmt.Errorf("songeval: pipeline %s: %w", name, err)
	}
	g.pipes[name] = p
	return nil
}

func (g *GPU) build() error {
	var w uint32
	takeW := func(n int) uint32 { o := w; w += uint32((n + 63) &^ 63); return o }
	g.wIn = takeW(2 * convDim * 10)
	g.wB0c2, g.wB1c1, g.wB1c2, g.wB1c3 = takeW(convDim), takeW(convDim), takeW(convDim), takeW(convDim)
	for l := range g.wLN {
		for i := range g.wLN[l] {
			g.wLN[l][i] = takeW(hidden)
		}
		g.wDW[l] = takeW(hidden*dwKernel + 2*hidden)
	}
	var b uint32
	takeB := func(n, k int) uint32 { o := b; b += uint32(n * k); return o }
	K3 := 9 * convDim
	g.bB0c2, g.bB1c1, g.bB1c2, g.bB1c3 = takeB(convDim, K3), takeB(convDim, K3), takeB(convDim, K3), takeB(convDim, K3)
	g.bLin = takeB(hidden, linK+biasPad)
	kh, kf := hidden+biasPad, ffn+biasPad
	for l := range g.bL {
		g.bL[l][0], g.bL[l][1] = takeB(ffn, kh), takeB(hidden, kf)
		for i := 2; i < 6; i++ {
			g.bL[l][i] = takeB(hidden, kh)
		}
		g.bL[l][6], g.bL[l][7] = takeB(2*hidden, kh), takeB(hidden, kh)
		g.bL[l][8] = takeB(ffn, kh)
	}
	// ffn2: the second projection after the first.
	g.bFfd1, g.bFfd2 = takeB(ffn, kh), takeB(hidden, kf)
	for i := range g.bIn {
		for j := range g.bIn[i] {
			g.bIn[i][j] = takeB(hidden, kh)
		}
		g.bOut[i] = takeB(hidden, kh)
	}
	g.ffn2Out = make([]uint32, Layers)
	for l := range g.ffn2Out {
		g.ffn2Out[l] = takeB(hidden, kf)
	}
	var err error
	if g.wbuf, err = g.dev.NewBuffer(int(w) * 4); err != nil {
		return fmt.Errorf("songeval: fp32 weights: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(int(b) * 2); err != nil {
		return fmt.Errorf("songeval: fp16 weights (%d MB): %w", b>>19, err)
	}
	for _, s := range []struct {
		name  string
		spirv []byte
		wave  uint32
	}{
		{"convin", shaders.MuQConvIn, 0},
		{"res", shaders.MuQRes, 0},
		{"act", shaders.MuQAct, 0},
		{"rotary", shaders.MuQRotary, 0},
		{"dwconv", shaders.MuQDWConv, 0},
		{"layernorm", shaders.ParakeetLayerNorm, 0},
		{"layernorm16", shaders.ParakeetLayerNormF16, 0},
		{"residual", shaders.ParakeetResidual, 0},
		{"pack64", shaders.KokoroBertPackHD64, 0},
		{"attn64", shaders.MuQAttnHD64TailMaxW32, 32},
		{"pack128", shaders.DiTPackF16, 0},
		{"attn128", shaders.SongEvalAttnHD128TailMaxW32, 32},
		{projKernel.name, projKernel.spirv, projKernel.wave},
		{convKernel.name, convKernel.spirv, convKernel.wave},
	} {
		if err := g.pipeline(s.name, s.spirv, s.wave); err != nil {
			return err
		}
	}
	return nil
}

// tileLinear lays a [out][in] weight out as B_LAYOUT=2's fragment tiles over
// a K of kPad, the bias (when there is one) at K index in.
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

// tileConv lays a folded 3x3 conv [out][in][3][3] out for A_CONV=3: K index
// (3·df + dt)·in + c.
func tileConv(dst []uint16, c conv) {
	re := make([]float32, len(c.w))
	for o := 0; o < convDim; o++ {
		for i := 0; i < c.in; i++ {
			for t := 0; t < 9; t++ {
				re[o*9*c.in+t*c.in+i] = c.w[(o*c.in+i)*9+t]
			}
		}
	}
	tileLinear(dst, re, nil, convDim, 9*c.in, 9*c.in)
}

func (g *GPU) stage(w *weights) error {
	w32 := make([]float32, g.wbuf.Size()/4)
	put := func(off uint32, v []float32) { copy(w32[off:], v) }
	put(g.wIn, w.b0c1.w)
	put(g.wIn+4608, w.b0c1.b)
	put(g.wIn+5120, w.b0c3.w)
	put(g.wIn+9728, w.b0c3.b)
	put(g.wB0c2, w.b0c2.b)
	put(g.wB1c1, w.b1c1.b)
	put(g.wB1c2, w.b1c2.b)
	put(g.wB1c3, w.b1c3.b)
	for l, L := range w.layer {
		for i, v := range [][]float32{L.ffn1LNG, L.ffn1LNB, L.attnLNG, L.attnLNB, L.convLNG, L.convLNB,
			L.ffn2LNG, L.ffn2LNB, L.finalLNG, L.finalLNB} {
			put(g.wLN[l][i], v)
		}
		put(g.wDW[l], L.dw)
		put(g.wDW[l]+hidden*dwKernel, L.bnScale)
		put(g.wDW[l]+hidden*dwKernel+hidden, L.bnShift)
	}
	g.wbuf.WriteFloat32At(0, w32)

	bank := make([]uint16, g.bank.Size()/2)
	tileConv(bank[g.bB0c2:], w.b0c2)
	tileConv(bank[g.bB1c1:], w.b1c1)
	tileConv(bank[g.bB1c2:], w.b1c2)
	tileConv(bank[g.bB1c3:], w.b1c3)
	tileLinear(bank[g.bLin:], w.linW, w.linB, hidden, linK, linK+biasPad)
	kh, kf := hidden+biasPad, ffn+biasPad
	for l, L := range w.layer {
		tileLinear(bank[g.bL[l][0]:], L.ffn1W1, L.ffn1B1, ffn, hidden, kh)
		tileLinear(bank[g.bL[l][1]:], L.ffn1W2, L.ffn1B2, hidden, ffn, kf)
		tileLinear(bank[g.bL[l][2]:], L.q, L.qB, hidden, hidden, kh)
		tileLinear(bank[g.bL[l][3]:], L.k, L.kB, hidden, hidden, kh)
		tileLinear(bank[g.bL[l][4]:], L.v, L.vB, hidden, hidden, kh)
		tileLinear(bank[g.bL[l][5]:], L.o, L.oB, hidden, hidden, kh)
		tileLinear(bank[g.bL[l][6]:], L.pw1, nil, 2*hidden, hidden, kh)
		tileLinear(bank[g.bL[l][7]:], L.pw2, nil, hidden, hidden, kh)
		tileLinear(bank[g.bL[l][8]:], L.ffn2W1, L.ffn2B1, ffn, hidden, kh)
		tileLinear(bank[g.ffn2Out[l]:], L.ffn2W2, L.ffn2B2, hidden, ffn, kf)
	}
	tileLinear(bank[g.bFfd1:], w.ffdW1, w.ffdB1, ffn, hidden, kh)
	tileLinear(bank[g.bFfd2:], w.ffdW2, w.ffdB2, hidden, ffn, kf)
	for i := range g.bIn {
		for j := 0; j < 3; j++ {
			tileLinear(bank[g.bIn[i][j]:], w.inW[i][j*hidden*hidden:(j+1)*hidden*hidden],
				w.inB[i][j*hidden:(j+1)*hidden], hidden, hidden, kh)
		}
		tileLinear(bank[g.bOut[i]:], w.outW[i], w.outB[i], hidden, hidden, kh)
	}
	g.bank.WriteUint16At(0, bank)

	// fp16: zero (the borders of the conv planes stay so), and the 1 in
	// every operand row's bias column.
	hz := make([]uint16, g.hbuf.Size()/2)
	one := safetensors.F32ToF16(1)
	for r := 0; r < chunkTok; r++ {
		hz[int(g.hLin)+r*(linK+biasPad)+linK] = one
	}
	R := (int(g.hA2) - int(g.hA)) / (hidden + biasPad)
	for r := 0; r < R; r++ {
		hz[int(g.hA)+r*kh+hidden] = one
		hz[int(g.hA2)+r*kh+hidden] = one
		hz[int(g.hB)+r*kf+ffn] = one
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

func (gr *graph) gemm(kern gemmKernel, kind string, pc pushConstants, m, n, k int) {
	mp := roundUp(m, kern.bm)
	pc.GemmM, pc.GemmN, pc.GemmK = uint32(mp), uint32(n), uint32(k)
	gr.add(kern.name, kind, uint32(n/kern.bn), uint32(mp/kern.bm), pc)
}

func (gr *graph) submit() (time.Duration, error) {
	var total time.Duration
	for d := gr.d; len(d) > 0; {
		n := min(len(d), perSubmit)
		t, err := vk.DispatchMultiTimed(d[:n], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("songeval: %w", err)
		}
		total += t
		d = d[n:]
	}
	gr.d, gr.kinds = gr.d[:0], gr.kinds[:0]
	return total, nil
}

// chunk records the front end for the chunk whose first token is a, given
// the song's mel frames, block-0 times and tokens. The host has put the
// chunk's mel at aMel.
func (gr *graph) chunk(a, frames, t1, tokens int) {
	g := gr.g
	o1, o2 := 2*a-2*halo, a-halo
	gr.add("convin", "conv in", groups(t1Local*convDim, 256), f1, pushConstants{
		InOff: g.aMel, WOff: g.wIn, OutOff: g.hMid1, Span: uint32(tp1), KOff: g.aSC1, KStride: uint32(mpad1),
		Tokens: t1Local, Aux0: melLocal, Aux1: uint32(int32(o1)), Aux2: uint32(t1)})
	conv := func(kind string, in uint32, pitch, stride, F int, b, out uint32, m int) {
		for f := 0; f < F; f++ {
			gr.gemm(convKernel, kind, pushConstants{InOff: in + uint32(stride*f*pitch*convDim),
				LDA: uint32(stride * convDim), Aux0: convDim, Aux1: convDim, LDB: uint32(pitch * convDim),
				BOff: b, OutOff: out + uint32(f*m*convDim)}, m, convDim, 9*convDim)
		}
	}
	res := func(kind string, mode, F, T int, y, sc, bias, bias3, out uint32, m, pitch int, origin, total int, extra pushConstants) {
		pc := extra
		pc.InOff, pc.KOff, pc.WOff, pc.Aux0, pc.OutOff = y, sc, bias, bias3, out
		pc.HeadDim, pc.Heads, pc.Tokens, pc.KStride = uint32(mode), uint32(F), uint32(T), uint32(m)
		if mode < 2 {
			pc.Span, pc.Aux1, pc.Aux2 = uint32(pitch), uint32(int32(origin)), uint32(total)
		}
		gr.add("res", kind, groups(T*convDim, 256), uint32(F), pc)
	}
	conv("b0 conv2", g.hMid1, tp1, 1, f1, g.bB0c2, g.aY1, mpad1)
	res("b0 out", 1, f1, t1Local, g.aY1, g.aSC1, g.wB0c2, 0, g.hOut1, mpad1, tp1, o1, t1, pushConstants{})
	conv("b1 conv1", g.hOut1, tp1, 2, f2, g.bB1c1, g.aY2a, mpad2)
	conv("b1 conv3", g.hOut1, tp1, 2, f2, g.bB1c3, g.aY2c, mpad2)
	res("b1 mid", 0, f2, t2Local, g.aY2a, 0, g.wB1c1, 0, g.hMid2, mpad2, tp2, o2, tokens, pushConstants{})
	conv("b1 conv2", g.hMid2, tp2, 1, f2, g.bB1c2, g.aY2b, mpad2)
	n := min(chunkTok, tokens-a)
	res("b1 out", 2, f2, t2Local, g.aY2b, g.aY2c, g.wB1c2, g.wB1c3, g.hLin, mpad2, 0, 0, 0,
		pushConstants{Dim: halo, Span: uint32(n), LDA: linK + biasPad})
	gr.gemm(projKernel, "linear", pushConstants{InOff: g.hLin, LDA: linK + biasPad, BOff: g.bLin,
		OutOff: g.aX + uint32(a*hidden)}, chunkTok, hidden, linK+biasPad)
}

// The host's part of a chunk: its mel, zero outside the song.
func (g *GPU) putMel(mel []float32, frames, a int) {
	o0 := 4*a - 4*halo
	buf := make([]float32, nMels*melLocal)
	for m := 0; m < nMels; m++ {
		for j := 0; j < melLocal; j++ {
			if t := o0 + j; t >= 0 && t < frames {
				buf[m*melLocal+j] = mel[m*frames+t]
			}
		}
	}
	g.abuf.WriteFloat32At(int(g.aMel), buf)
}

// putRotary writes the rotary cos and sin tables for T tokens: torch's
// fp32 angle t·inv_freq, its cosine and sine.
func (g *GPU) putRotary(w *weights, T int) {
	c, s := make([]float32, T*headDim/2), make([]float32, T*headDim/2)
	for t := 0; t < T; t++ {
		for j, f := range w.invFreq {
			a := float32(t) * f
			c[t*headDim/2+j] = float32(math.Cos(float64(a)))
			s[t*headDim/2+j] = float32(math.Sin(float64(a)))
		}
	}
	g.abuf.WriteFloat32At(int(g.aCos), c)
	g.abuf.WriteFloat32At(int(g.aSin), s)
}

// encode records stop conformer layers over T tokens in aX; the hidden state
// ends in the returned offset.
func (gr *graph) encode(T, stop int) uint32 {
	g := gr.g
	R := roundUp(T, tokAlign)
	kh, kf := uint32(hidden+biasPad), uint32(ffn+biasPad)
	x, x2 := g.aX, g.aX2
	ln16 := func(kind string, gam, bet uint32) {
		gr.add("layernorm16", kind, uint32(R), 1, pushConstants{InOff: x, OutOff: g.hA, Tokens: uint32(R), Dim: hidden,
			LDA: kh, WOff: gam, Aux0: bet, Eps: math.Float32bits(lnEps)})
	}
	proj := func(kind string, a, lda, b, out uint32, n, k int) {
		gr.gemm(projKernel, kind, pushConstants{InOff: a, LDA: lda, BOff: b, OutOff: out}, R, n, k)
	}
	act := func(kind string, mode uint32, in, out uint32, dim int, lda uint32) {
		gr.add("act", kind, groups(dim, 256), uint32(R), pushConstants{InOff: in, OutOff: out, Tokens: uint32(R),
			Dim: uint32(dim), LDA: lda, Aux2: mode})
	}
	residual := func(kind string, branch uint32, scale float32) {
		gr.add("residual", kind, uint32(R), 1, pushConstants{Tokens: uint32(R), Dim: hidden, InOff: branch,
			OutOff: x, Scale: math.Float32bits(scale)})
	}
	ffnHalf := func(kind string, l, gi int, b1, b2 uint32) {
		ln16(kind+" norm", g.wLN[l][gi], g.wLN[l][gi+1])
		proj(kind+" 1", g.hA, kh, b1, g.aFF, ffn, hidden+biasPad)
		act(kind+" silu", 1, g.aFF, g.hB, ffn, kf)
		proj(kind+" 2", g.hB, kf, b2, g.aTmp, hidden, ffn+biasPad)
		residual(kind+" residual", g.aTmp, 0.5)
	}
	tp := uint32(roundUp(T, tokAlign))
	for l := 0; l < stop; l++ {
		bl := g.bL[l]
		ffnHalf("ffn1", l, 0, bl[0], bl[1])
		// Attention: rotary on the normed input for q and k, plain for v.
		gr.add("layernorm", "attn norm", uint32(R), 1, pushConstants{InOff: x, OutOff: g.aV, Tokens: uint32(R),
			Dim: hidden, WOff: g.wLN[l][2], Aux0: g.wLN[l][3], Eps: math.Float32bits(lnEps)})
		gr.add("rotary", "rotary", hidden/256, uint32(T), pushConstants{InOff: g.aV, OutOff: g.hA, Tokens: uint32(T),
			LDA: kh, Aux0: g.aCos, Aux1: g.aSin})
		act("narrow v", 0, g.aV, g.hA2, hidden, kh)
		proj("q", g.hA, kh, bl[2], g.aQ, hidden, hidden+biasPad)
		proj("k", g.hA, kh, bl[3], g.aK, hidden, hidden+biasPad)
		proj("v", g.hA2, kh, bl[4], g.aV, hidden, hidden+biasPad)
		gr.attention(64, heads, T, tp, g.aQ)
		act("narrow ctx", 0, g.aQ, g.hA, hidden, kh)
		proj("out", g.hA, kh, bl[5], g.aTmp, hidden, hidden+biasPad)
		residual("attn residual", g.aTmp, 1)
		// The convolution module.
		ln16("conv norm", g.wLN[l][4], g.wLN[l][5])
		proj("pw1", g.hA, kh, bl[6], g.aFF, 2*hidden, hidden+biasPad)
		act("glu", 3|4, g.aFF, g.aK, hidden, hidden)
		gr.add("dwconv", "dwconv", hidden/256, uint32(T), pushConstants{InOff: g.aK, OutOff: g.hA, Tokens: uint32(T),
			LDA: kh, WOff: g.wDW[l]})
		proj("pw2", g.hA, kh, bl[7], g.aTmp, hidden, hidden+biasPad)
		residual("conv residual", g.aTmp, 1)
		ffnHalf("ffn2", l, 6, bl[8], g.ffn2Out[l])
		gr.add("layernorm", "final norm", uint32(R), 1, pushConstants{InOff: x, OutOff: x2, Tokens: uint32(R),
			Dim: hidden, WOff: g.wLN[l][8], Aux0: g.wLN[l][9], Eps: math.Float32bits(lnEps)})
		x, x2 = x2, x
	}
	return x
}

// attention records q, k, v (fp32 at aQ, aK, aV) packed and attended, the
// context fp32 to out: hd 64 (MuQ's 16 heads) or 128 (SongEval's 8).
func (gr *graph) attention(hd, nh, T int, tp uint32, out uint32) {
	g := gr.g
	pack, attn := "pack64", "attn64"
	if hd == 128 {
		pack, attn = "pack128", "attn128"
	}
	scale := float32(1 / math.Sqrt(float64(hd)) * log2e)
	for _, p := range []struct {
		src, dst uint32
		mode     uint32
		scale    float32
	}{{g.aQ, g.hQ, 0, scale}, {g.aK, g.hK, 0, 1}, {g.aV, g.hV, 1, 1}} {
		gr.add(pack, "pack", groups(int(tp), tile), uint32(nh), pushConstants{InOff: p.src, OutOff: p.dst,
			Tokens: uint32(T), Dim: hidden, Aux0: p.mode, Aux1: tp, Scale: math.Float32bits(p.scale)})
	}
	gr.add(attn, "attention", groups(T, tile), uint32(nh), pushConstants{InOff: g.hQ, KOff: g.hK, VOff: g.hV,
		OutOff: out, Tokens: uint32(T), Dim: hidden, Heads: uint32(nh), Aux1: tp})
}

// head records SongEval's head over hidden (T tokens): the FFN into aX2's
// partner (ssl), then the four attention layers; it returns where the
// FFN's output and the last attention's output are.
func (gr *graph) head(hiddenOff uint32, T int) (ssl, last uint32) {
	g := gr.g
	R := roundUp(T, tokAlign)
	kh, kf := uint32(hidden+biasPad), uint32(ffn+biasPad)
	tp := uint32(roundUp(T, tokAlign))
	// ssl and the chain's two buffers are the ones hidden is not in.
	ssl, buf := g.aX2, g.aX
	if hiddenOff == g.aX2 {
		ssl, buf = g.aX, g.aX2
	}
	narrow := func(in uint32) {
		gr.add("act", "narrow", hidden/256, uint32(R), pushConstants{InOff: in, OutOff: g.hA, Tokens: uint32(R),
			Dim: hidden, LDA: kh})
	}
	proj := func(kind string, a, lda, b, out uint32, n, k int) {
		gr.gemm(projKernel, kind, pushConstants{InOff: a, LDA: lda, BOff: b, OutOff: out}, R, n, k)
	}
	narrow(hiddenOff)
	proj("ffd 1", g.hA, kh, g.bFfd1, g.aFF, ffn, hidden+biasPad)
	gr.add("act", "relu", ffn/256, uint32(R), pushConstants{InOff: g.aFF, OutOff: g.hB, Tokens: uint32(R),
		Dim: ffn, LDA: kf, Aux2: 2})
	proj("ffd 2", g.hB, kf, g.bFfd2, ssl, hidden, ffn+biasPad)
	cur := ssl
	for i := 0; i < 4; i++ {
		narrow(cur)
		proj("head q", g.hA, kh, g.bIn[i][0], g.aQ, hidden, hidden+biasPad)
		proj("head k", g.hA, kh, g.bIn[i][1], g.aK, hidden, hidden+biasPad)
		proj("head v", g.hA, kh, g.bIn[i][2], g.aV, hidden, hidden+biasPad)
		gr.attention(headDim2, headHeads, T, tp, g.aTmp)
		narrow(g.aTmp)
		proj("head out", g.hA, kh, g.bOut[i], buf, hidden, hidden+biasPad)
		cur = buf
	}
	return ssl, cur
}
