package layout

// The convolution trunk on the device (OCR.md O7): the backbone, the FPN and
// PAN, the mask prototypes and the decoder's input projections, which were
// 90% of the CPU forward's 2.2 s. The AIFI layer between the backbone and
// the FPN (625 tokens) and everything from the query selection on stay on
// the host for now (head, shared with the CPU path); O11 moves them.
//
// **fp16 operands, fp32 sums**, which the oracle measured before this was
// written (TestFP16Ladder): the largest activation anywhere is 204, the trunk
// moves by ~1e-3, and no region of the five cases moves at all.
//
// The layout:
//
//   - every map is fp32, channel-last, its row stride the channel count
//     rounded up to 64 (the GEMM tile), the pad channels zero. The input is
//     the exception, 3 channels at stride 3;
//   - a convolution is an im2col into the fp16 A operand (layout_ops MODE 0)
//     or, for a 1x1 stride-1 conv, the map narrowed straight into it; then
//     dit_gemm against the weight in the fragment tiling; then
//     qvit_bias_act for the folded batch norm's bias and the activation. The
//     weight is packed to match A: tap-major, real input channels innermost,
//     K rounded up to 64 with zero columns, N to 64 with zero rows;
//   - **a channel concatenation feeding a 1x1 conv is never built**: each
//     part is narrowed into A at its own column offset and the weight's
//     columns are placed to match (the HGNetV2 aggregation, the CSP inputs);
//   - **RepVGG is reparameterised**: each block's 1x1 branch is added into
//     its 3x3's centre tap at load, so a block is one conv (exact algebra;
//     the CPU oracle keeps the two branches);
//   - depthwise convs, the max-pool, the upsamples, the adds and the stem's
//     one concatenation that a 3x3 reads are layout_ops' other modes.
//
// The input is always 800x800, so the graph is static: it is recorded once
// at construction, every map has its own place in the arena (so a gate can
// read any stage after a run), and a Forward uploads the pixels, submits,
// reads four things back and runs the head.

import (
	"fmt"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	tile64   = 64  // the GEMM's M, N and K granularity
	tile16   = 16  // the fragment tiling's
	ldaPad   = 128 // A's row pad in halves (IDEAS §2.3)
	maxGroup = 65535
)

// pushConstants mirrors dit_common.glsl's block.
type pushConstants struct {
	InOff, OutOff, WOff         uint32
	Tokens, Dim, Heads, HeadDim uint32
	Span, KOff, VOff, KStride   uint32
	Eps, Scale                  uint32
	Aux0, Aux1, Aux2            uint32
	BOff, GemmM, GemmN, GemmK   uint32
	LDA, LDB                    uint32
}

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// gmap is a map in the activation arena.
type gmap struct {
	off        uint32
	H, W, C, S int // S is the row stride in floats
}

func (g gmap) rows() int { return g.H * g.W }

type op struct {
	pipe   string
	gx, gy uint32
	pc     pushConstants
}

// GPU is the trunk staged on a device.
type GPU struct {
	m   *Model
	dev *vk.Device

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	// Built before the buffers exist.
	w32      []float32
	halves   []uint16
	actElems int
	hElems   int
	ops      *[]op
	segA     []op
	segB     []op

	input, proj2, aifi, mf gmap
	levels, encLin         [numLevels]gmap
	values                 [6][numLevels]gmap
	taps                   map[string]gmap
	// Tap order, for a gate that walks them.
	tapNames []string
}

// NewGPU stages m's trunk on dev: 66 MB of fp16 weights and ~1 GB of maps.
func NewGPU(dev *vk.Device, m *Model) (*GPU, error) {
	g := &GPU{m: m, dev: dev, pipes: map[string]*vk.ComputePipeline{}, taps: map[string]gmap{}}
	g.build()
	if err := g.alloc(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.pipelines(); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
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
}

// DeviceBytes is what the trunk holds on the device.
func (g *GPU) DeviceBytes() int {
	return g.wbuf.Size() + g.abuf.Size() + g.hbuf.Size() + g.bank.Size()
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func (g *GPU) newMap(h, w, c int) gmap {
	s := roundUp(c, tile64)
	m := gmap{off: uint32(g.actElems), H: h, W: w, C: c, S: s}
	g.actElems += roundUp(roundUp(h*w, tile64)*s, 64)
	return m
}

func (g *GPU) weights(v []float32) uint32 {
	off := uint32(len(g.w32))
	g.w32 = append(g.w32, v...)
	return off
}

func (g *GPU) emit(pipe string, gx, gy uint32, pc pushConstants) {
	*g.ops = append(*g.ops, op{pipe: pipe, gx: gx, gy: gy, pc: pc})
}

// grid is a grid-stride kernel's workgroup count for n elements.
func grid(n int) uint32 { return uint32(min((n+255)/256, 4096)) }

func (g *GPU) tap(name string, m gmap) {
	g.taps[name] = m
	g.tapNames = append(g.tapNames, name)
}

// seg is one part of a 1x1 conv's input: a map at a column offset of A.
type seg struct {
	m   gmap
	col int
}

// packGEMM tiles an [n, kPad] weight into the bank and returns its offset.
func (g *GPU) packGEMM(w []float32, n, k int) uint32 {
	off := uint32(len(g.halves))
	dst := make([]uint16, n*k)
	kt := k / tile16
	for i := 0; i < n; i++ {
		base := (i / tile16) * kt * tile16 * tile16
		lane := (i % tile16) * tile16
		for j, v := range w[i*k : (i+1)*k] {
			dst[base+(j/tile16)*tile16*tile16+lane+j%tile16] = safetensors.F32ToF16(v)
		}
	}
	g.halves = append(g.halves, dst...)
	return off
}

// gemm records A x W^T into out, then the bias and the activation.
func (g *GPU) gemm(c *Conv, wp []float32, kPad int, out gmap, lda int) {
	n := out.S
	bank := g.packGEMM(wp, n, kPad)
	bias := make([]float32, n)
	copy(bias, c.B)
	boff := g.weights(bias)
	m := out.rows()
	mPad := roundUp(m, tile64)
	g.hElems = max(g.hElems, mPad*lda)
	g.emit("gemm", uint32(n/tile64), uint32(mPad/tile64), pushConstants{
		InOff: 0, OutOff: out.off, BOff: bank, GemmM: uint32(mPad), GemmN: uint32(n), GemmK: uint32(kPad), LDA: uint32(lda),
	})
	act := map[Act]uint32{ActNone: 0, ActReLU: 3, ActSiLU: 4, ActGELU: 2}[c.Act]
	for r := 0; r < m; r += maxGroup {
		rows := min(maxGroup, m-r)
		g.emit("biasact", uint32(rows), 1, pushConstants{
			OutOff: out.off + uint32(r*n), WOff: boff, Tokens: uint32(rows), Dim: uint32(n), Aux0: act,
		})
	}
}

// narrow records a map's rows into A at a column offset.
func (g *GPU) narrow(x gmap, col, lda int) {
	for r := 0; r < x.rows(); r += maxGroup {
		rows := min(maxGroup, x.rows()-r)
		g.emit("narrow", uint32(rows), 1, pushConstants{
			InOff: x.off + uint32(r*x.S), OutOff: uint32(r*lda + col), Tokens: uint32(rows), Dim: uint32(x.S), LDA: uint32(lda),
		})
	}
}

// conv1x1 records a 1x1 stride-1 conv over the concatenation of parts.
func (g *GPU) conv1x1(c *Conv, parts ...gmap) gmap {
	k := 0
	var segs []seg
	for _, p := range parts {
		segs = append(segs, seg{p, k})
		k += p.S
	}
	lda := k + ldaPad
	out := g.newMap(parts[0].H, parts[0].W, c.Out)
	wp := make([]float32, out.S*k)
	for o := 0; o < c.Out; o++ {
		in := 0
		for _, s := range segs {
			for ch := 0; ch < s.m.C; ch++ {
				wp[o*k+s.col+ch] = c.W[o*c.In+in+ch]
			}
			in += s.m.C
		}
		if in != c.In {
			panic(fmt.Sprintf("layout: a 1x1 conv of %d inputs given %d channels", c.In, in))
		}
	}
	for _, s := range segs {
		g.narrow(s.m, s.col, lda)
	}
	g.gemm(c, wp, k, out, lda)
	return out
}

// conv records a convolution; oh and ow override the output size (0 is the
// usual one), which is how the stem's F.pad(0, 1, 0, 1) is expressed.
func (g *GPU) conv(c *Conv, x gmap, oh, ow int) gmap {
	if c.Groups == c.In && c.Groups == c.Out && c.Groups > 1 {
		return g.dwconv(c, x)
	}
	if c.Groups != 1 {
		panic("layout: grouped conv")
	}
	if c.K == 1 && c.Stride == 1 && c.Pad == 0 && oh == 0 {
		return g.conv1x1(c, x)
	}
	if oh == 0 {
		oh = (x.H+2*c.Pad-c.K)/c.Stride + 1
		ow = (x.W+2*c.Pad-c.K)/c.Stride + 1
	}
	out := g.newMap(oh, ow, c.Out)
	kReal := c.K * c.K * c.In
	kPad := roundUp(kReal, tile64)
	lda := kPad + ldaPad
	wp := make([]float32, out.S*kPad)
	for o := 0; o < c.Out; o++ {
		for ci := 0; ci < c.In; ci++ {
			for ky := 0; ky < c.K; ky++ {
				for kx := 0; kx < c.K; kx++ {
					wp[o*kPad+(ky*c.K+kx)*c.In+ci] = c.W[((o*c.In+ci)*c.K+ky)*c.K+kx]
				}
			}
		}
	}
	g.emit("im2col", grid(out.rows()*kPad), 1, pushConstants{
		InOff: x.off, OutOff: 0, Tokens: uint32(out.rows()), Dim: uint32(kReal), GemmK: uint32(kPad), LDA: uint32(lda),
		Heads: uint32(c.In), HeadDim: uint32(x.S), Span: uint32(c.K), KOff: uint32(x.H), VOff: uint32(x.W),
		KStride: uint32(ow), Aux0: uint32(c.Stride), Aux1: uint32(c.Pad), Aux2: uint32(c.Pad),
	})
	g.gemm(c, wp, kPad, out, lda)
	return out
}

func actCode(a Act) uint32 { return map[Act]uint32{ActNone: 0, ActReLU: 3, ActSiLU: 4}[a] }

func (g *GPU) dwconv(c *Conv, x gmap) gmap {
	oh := (x.H+2*c.Pad-c.K)/c.Stride + 1
	ow := (x.W+2*c.Pad-c.K)/c.Stride + 1
	out := g.newMap(oh, ow, c.Out)
	if out.S != x.S {
		panic("layout: depthwise stride mismatch")
	}
	woff := g.weights(append(append([]float32(nil), c.W...), c.B...))
	g.emit("dwconv", grid(out.rows()*c.Out), 1, pushConstants{
		InOff: x.off, OutOff: out.off, WOff: woff, Tokens: uint32(out.rows()), Heads: uint32(c.Out),
		HeadDim: uint32(x.S), Span: uint32(c.K), KOff: uint32(x.H), VOff: uint32(x.W), KStride: uint32(ow),
		Aux0: uint32(c.Stride), Aux1: uint32(c.Pad), Scale: actCode(c.Act),
	})
	return out
}

func (g *GPU) addMaps(a, b gmap, act Act) gmap {
	out := g.newMap(a.H, a.W, a.C)
	g.emit("add", grid(a.rows()*a.C), 1, pushConstants{
		InOff: a.off, KOff: b.off, OutOff: out.off, Tokens: uint32(a.rows()), Heads: uint32(a.C), HeadDim: uint32(a.S), Scale: actCode(act),
	})
	return out
}

func (g *GPU) up(x gmap, bilinear bool) gmap {
	out := g.newMap(x.H*2, x.W*2, x.C)
	pipe := "upnearest"
	if bilinear {
		pipe = "upbilinear"
	}
	g.emit(pipe, grid(out.rows()*x.C), 1, pushConstants{
		InOff: x.off, OutOff: out.off, Tokens: uint32(out.rows()), Heads: uint32(x.C), HeadDim: uint32(x.S), KOff: uint32(x.H), VOff: uint32(x.W),
	})
	return out
}

// rep folds a RepVGG block into one 3x3 conv with SiLU.
func rep(r *repVGG) *Conv {
	c := *r.a
	c.W = append([]float32(nil), r.a.W...)
	c.B = make([]float32, c.Out)
	for o := 0; o < c.Out; o++ {
		c.B[o] = r.a.B[o] + r.b.B[o]
		for ci := 0; ci < c.In; ci++ {
			c.W[((o*c.In+ci)*3+1)*3+1] += r.b.W[o*c.In+ci]
		}
	}
	c.Act = ActSiLU
	return &c
}

func (g *GPU) csp(c *cspRep, parts ...gmap) gmap {
	a := g.conv1x1(c.conv1, parts...)
	for i := range c.bottlenecks {
		a = g.conv(rep(&c.bottlenecks[i]), a, 0, 0)
	}
	b := g.conv1x1(c.conv2, parts...)
	return g.addMaps(a, b, ActNone)
}

func (g *GPU) basic(b *basicLayer, x gmap) gmap {
	outs := []gmap{x}
	h := x
	for _, convs := range b.layers {
		for _, c := range convs {
			h = g.conv(c, h, 0, 0)
		}
		outs = append(outs, h)
	}
	y := g.conv1x1(b.squeeze, outs...)
	y = g.conv(b.excite, y, 0, 0)
	if b.residual {
		y = g.addMaps(y, x, ActNone)
	}
	return y
}

// build records both segments.
func (g *GPU) build() {
	m := g.m
	g.ops = &g.segA
	g.input = gmap{off: uint32(g.actElems), H: InputSize, W: InputSize, C: 3, S: 3}
	g.actElems += roundUp(3*InputSize*InputSize, 64)

	s1 := g.conv(m.stem[0], g.input, 0, 0)
	a := g.conv(m.stem[1], s1, s1.H, s1.W) // F.pad(0, 1, 0, 1), then k2 pad 0
	a = g.conv(m.stem[2], a, a.H, a.W)
	pool := g.newMap(s1.H, s1.W, s1.C)
	g.emit("maxpool", grid(s1.rows()*s1.C), 1, pushConstants{
		InOff: s1.off, OutOff: pool.off, Tokens: uint32(s1.rows()), Heads: uint32(s1.C), HeadDim: uint32(s1.S), KOff: uint32(s1.H), VOff: uint32(s1.W),
	})
	cat := g.newMap(s1.H, s1.W, pool.C+a.C)
	for _, p := range []struct {
		m   gmap
		off int
	}{{pool, 0}, {a, pool.C}} {
		g.emit("copych", grid(p.m.rows()*p.m.C), 1, pushConstants{
			InOff: p.m.off, OutOff: cat.off, Tokens: uint32(p.m.rows()), Heads: uint32(p.m.C), HeadDim: uint32(p.m.S),
			Aux0: uint32(p.off), Aux1: uint32(cat.S),
		})
	}
	h := g.conv(m.stem[3], cat, 0, 0)
	h = g.conv(m.stem[4], h, 0, 0)
	g.tap("bb.stem", h)
	var feats [4]gmap
	for s := range m.stages {
		st := &m.stages[s]
		if st.down != nil {
			h = g.conv(st.down, h, 0, 0)
		}
		for b := range st.blocks {
			h = g.basic(&st.blocks[b], h)
		}
		feats[s] = h
		g.tap(fmt.Sprintf("bb.stage%d", s+1), h)
	}
	var proj [3]gmap
	for i := range proj {
		proj[i] = g.conv(m.encProj[i], feats[i+1], 0, 0)
		g.tap(fmt.Sprintf("enc.proj%d", i), proj[i])
	}
	g.proj2 = proj[2]

	// Segment B: after the host's AIFI layer writes g.aifi.
	g.ops = &g.segB
	g.aifi = g.newMap(proj[2].H, proj[2].W, proj[2].C)
	fpn := []gmap{g.aifi}
	for i := 0; i < 2; i++ {
		lat := g.conv(m.lateral[i], fpn[len(fpn)-1], 0, 0)
		fpn[len(fpn)-1] = lat
		fpn = append(fpn, g.csp(&m.fpn[i], g.up(lat, false), proj[1-i]))
	}
	fpn[0], fpn[2] = fpn[2], fpn[0]
	pan := []gmap{fpn[0]}
	for i := 0; i < 2; i++ {
		down := g.conv(m.downsample[i], pan[len(pan)-1], 0, 0)
		pan = append(pan, g.csp(&m.pan[i], down, fpn[i+1]))
	}
	for i, p := range pan {
		g.tap(fmt.Sprintf("enc.pan%d", i), p)
	}
	var mf gmap
	for i, head := range m.scaleHeads {
		y := pan[i]
		for _, c := range head {
			y = g.conv(c, y, 0, 0)
			if i > 0 {
				y = g.up(y, true)
			}
		}
		if i == 0 {
			mf = y
		} else {
			mf = g.addMaps(mf, y, ActNone)
		}
	}
	mf = g.up(g.conv(m.maskOut, mf, 0, 0), true)
	mf = g.addMaps(mf, g.conv(m.maskLat, feats[0], 0, 0), ActNone)
	mf = g.conv(m.maskProto, g.conv(m.maskBase, mf, 0, 0), 0, 0)
	g.mf = mf
	g.tap("enc.mask_feat", mf)
	for l := range g.levels {
		g.levels[l] = g.conv(m.decProj[l], pan[l], 0, 0)
	}
	// The head's products over the whole memory: the encoder output's linear
	// and the six value projections, 6 GFLOP the host would spend 0.3 s on.
	for l := range g.levels {
		g.encLin[l] = g.conv1x1(linearConv(&m.encOutput), g.levels[l])
		for i := range m.layers {
			g.values[i][l] = g.conv1x1(linearConv(&m.layers[i].value), g.levels[l])
		}
	}
}

// linearConv is a Linear as the 1x1 conv the GEMM path takes.
func linearConv(l *Linear) *Conv {
	return &Conv{In: l.In, Out: l.Out, K: 1, Stride: 1, Groups: 1, W: l.W, B: l.B}
}

// rows reads the three levels of a per-level product as one [13125, 256].
func (g *GPU) rows(ms [numLevels]gmap) *Rows {
	out := newRows(0, dModel)
	for _, lv := range ms {
		raw := g.abuf.ReadFloat32At(int(lv.off), lv.rows()*lv.S)
		if lv.S == dModel {
			out.V = append(out.V, raw...)
		} else {
			for r := 0; r < lv.rows(); r++ {
				out.V = append(out.V, raw[r*lv.S:r*lv.S+dModel]...)
			}
		}
		out.N += lv.rows()
	}
	return out
}

func (g *GPU) alloc() error {
	var err error
	for _, a := range []struct {
		name  string
		bytes int
	}{{"fp32 activation", g.actElems * 4}, {"fp16 A operand", g.hElems * 2}, {"fp16 weight", len(g.halves) * 2}} {
		if a.bytes > 0xfffffffc {
			return fmt.Errorf("layout: the %s arena is %.2f GB, past one binding", a.name, float64(a.bytes)/1e9)
		}
	}
	if g.wbuf, err = g.dev.NewBuffer(len(g.w32) * 4); err != nil {
		return err
	}
	g.wbuf.WriteFloat32(g.w32)
	if g.abuf, err = g.dev.NewHostCachedBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("layout: activation arena (%d MB): %w", g.actElems*4>>20, err)
	}
	// Zeroed once: the maps' pad channels are read by 1x1 convs (against
	// zero weight columns) and are never written.
	g.abuf.Zero()
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return err
	}
	g.hbuf.Zero()
	if g.bank, err = g.dev.NewBuffer(len(g.halves) * 2); err != nil {
		return err
	}
	g.bank.WriteUint16At(0, g.halves)
	g.w32, g.halves = nil, nil
	return nil
}

func (g *GPU) pipelines() error {
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	for name, spirv := range map[string][]byte{
		"im2col": shaders.LayoutIm2col, "dwconv": shaders.LayoutDWConv, "maxpool": shaders.LayoutMaxPool,
		"upnearest": shaders.LayoutUpNearest, "upbilinear": shaders.LayoutUpBilinear, "add": shaders.LayoutAdd,
		"copych": shaders.LayoutCopyCh, "narrow": shaders.DiTScaleF16, "biasact": shaders.QViTBiasAct,
		"gemm": shaders.DiTGEMMReg64Tiled,
	} {
		bufs := base
		if name == "gemm" {
			bufs = append(append([]*vk.Buffer(nil), base...), g.bank)
		}
		mod, err := g.dev.NewShaderModule(spirv)
		if err != nil {
			return fmt.Errorf("layout: shader %s: %w", name, err)
		}
		g.mods = append(g.mods, mod)
		p, err := g.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: bufs, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))})
		if err != nil {
			return fmt.Errorf("layout: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	return nil
}

// opsPerSubmit bounds a command buffer, for the driver's watchdog.
const opsPerSubmit = 64

func (g *GPU) run(ops []op) (time.Duration, error) {
	var total time.Duration
	for i := 0; i < len(ops); i += opsPerSubmit {
		var ds []vk.MultiDispatch
		for _, o := range ops[i:min(i+opsPerSubmit, len(ops))] {
			ds = append(ds, vk.MultiDispatch{Pipeline: g.pipes[o.pipe], GroupsX: o.gx, GroupsY: max(o.gy, 1), PushConstants: o.pc.bytes()})
		}
		d, err := vk.DispatchMultiTimed(ds, 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("layout: dispatch %d: %w", i, err)
		}
		total += d
	}
	return total, nil
}

// read returns a map as [C, H, W].
func (g *GPU) read(m gmap) *Map {
	raw := g.abuf.ReadFloat32At(int(m.off), m.rows()*m.S)
	out := newMap(m.C, m.H, m.W)
	for c := 0; c < m.C; c++ {
		dst := out.Plane(c)
		for r := range dst {
			dst[r] = raw[r*m.S+c]
		}
	}
	return out
}

// Timings is where a GPU forward's time went.
type Timings struct {
	Upload, SegA, AIFI, SegB, Readback, Head time.Duration
}

// Forward runs the trunk on the device and the rest on the host. tap sees
// the trunk's stages (from the arena, after the run) and the head's.
func (g *GPU) Forward(pixels []float32, tap Tap) (*Output, *Timings, error) {
	if len(pixels) != 3*InputSize*InputSize {
		return nil, nil, fmt.Errorf("layout: %d values, want 3x%dx%d", len(pixels), InputSize, InputSize)
	}
	var tm Timings
	t0 := time.Now()
	nhwc := make([]float32, len(pixels))
	const n = InputSize * InputSize
	for c := 0; c < 3; c++ {
		for i := 0; i < n; i++ {
			nhwc[i*3+c] = pixels[c*n+i]
		}
	}
	g.abuf.WriteFloat32At(int(g.input.off), nhwc)
	tm.Upload = time.Since(t0)
	t0 = time.Now()
	if _, err := g.run(g.segA); err != nil {
		return nil, nil, err
	}
	tm.SegA = time.Since(t0)
	t0 = time.Now()
	a := g.m.aifiMap(g.read(g.proj2))
	rows := make([]float32, g.aifi.rows()*g.aifi.S)
	for c := 0; c < a.C; c++ {
		for r, v := range a.Plane(c) {
			rows[r*g.aifi.S+c] = v
		}
	}
	g.abuf.WriteFloat32At(int(g.aifi.off), rows)
	tm.AIFI = time.Since(t0)
	t0 = time.Now()
	if _, err := g.run(g.segB); err != nil {
		return nil, nil, err
	}
	tm.SegB = time.Since(t0)

	t0 = time.Now()
	if tap != nil {
		for _, name := range g.tapNames {
			if name == "enc.pan0" {
				tap("enc.aifi", a.D)
			}
			tap(name, g.read(g.taps[name]).D)
		}
	}
	memory := g.rows(g.levels)
	var shapes [numLevels][2]int
	for l, lv := range g.levels {
		shapes[l] = [2]int{lv.H, lv.W}
	}
	pre := &projected{encLin: g.rows(g.encLin)}
	for i := range pre.values {
		pre.values[i] = g.rows(g.values[i])
	}
	if tap != nil {
		tap("dec.memory", memory.V)
	}
	mf := g.read(g.mf)
	tm.Readback = time.Since(t0)
	t0 = time.Now()
	out, err := g.m.head(memory, shapes, mf, pre, tap)
	tm.Head = time.Since(t0)
	return out, &tm, err
}
