// Package vae is ACE-Step 1.5's audio VAE decoder on the device (MUSIC.md
// A5): diffusers' AutoencoderOobleck, 25 latents a second of 64 channels in,
// 48 kHz stereo out (hop 1920).
//
// The graph is conv1 (k7, 64 → 2048), five decoder blocks, a final Snake and
// conv2 (k7, 128 → 2, no bias). A block is a Snake, a ConvTranspose1d
// (kernel 2s, stride s, padding s/2; s = 10, 6, 4, 4, 2, channels halving
// from 2048 to 128) and three residual units (Snake, k7 conv at dilation
// 1/3/9, Snake, k1 conv, add). It is ~120 GFLOP a second of audio, 29 TFLOP
// for four minutes, so it runs on the device, on kokoro's machinery:
//
//   - **Every convolution is dit_gemm.comp's A_CONV GEMM** over a
//     channel-last fp16 activation: the im2col is an addressing rule, and the
//     padding is zero rows around the data. Every width here is a power of
//     two, which is what A_CONV=1 needs.
//   - **The transposed convolution is a 2-tap convolution with an s-times
//     wider output** (kokoro's upWeights): at kernel 2s every output frame
//     is reached by exactly two input frames, so a [T+1, s·C_out] GEMM read
//     as [(T+1)·s, C_out] from row s/2 is the upsampled signal.
//   - **Snake, the preceding bias and the narrowing are one pass**
//     (shaders/ace_vae.comp), which also lays the zero rows the next
//     convolution's taps and M padding read.
//   - **Weight norm is folded at load**: g·v/‖v‖ over every axis but the
//     first (the output channel of a Conv1d, the *input* channel of a
//     ConvTranspose1d).
//   - **The decode is tiled in time**: the last stage is 1920 × 128 fp32 a
//     latent, so a four-minute song does not fit one storage buffer. Tiles
//     are upstream's overlap-discard windows, and the oracle measures
//     upstream's own tiling against the untiled decode at ≤8e-6 (A-o6). The
//     decoder's receptive field is ~9 latents, and the margin here is 32.
package vae

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// Hop is the samples a latent decodes to; Rate is the sample rate.
const (
	Hop      = 1920
	Rate     = 48000
	Channels = 2
	Latent   = 64
)

var (
	strides  = []int{10, 6, 4, 4, 2}
	widths   = []int{2048, 1024, 512, 256, 128, 128} // before block i, and after the last
	dilation = []int{1, 3, 9}
)

const (
	tile = 16
	// border is the zero rows after (and before) the data in the fp16
	// arena: a k7 tap at dilation 9 reaches 27 rows, and a GEMM padded to
	// its 32-row M tile reads up to 31 rows more.
	border = 64
	// Margin is the overlap each side of a tile, in latents.
	Margin = 32
	noW    = 0xffffffff
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

// conv is one staged convolution: B at bOff of the bank, K = taps·in, N
// padded to the kernel's tile; bias in wbuf (or noW).
type conv struct {
	bOff        uint32
	bias        uint32
	in, n, taps int
	dil         int
}

// snake is a Snake's α then 1/(β + 1e-9) in wbuf, C each.
type snake uint32

type resUnit struct {
	s1, s2 snake
	c1, c2 conv
}

type block struct {
	s       snake
	up      conv // N = stride·out, K = 2·in
	stride  int
	in, out int
	res     [3]resUnit
}

// GPU holds the decoder on the device, for tiles of up to Window latents.
type GPU struct {
	// Between, when set, is called between tiles.
	Between func() error

	dev                    *vk.Device
	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule
	Window                 int

	conv1, conv2 conv
	blocks       []block
	final        snake

	aX, aC, aUp, aOut, aWav uint32
	hA                      uint32 // row 0 of the bordered fp16 arena
}

// convKernel is the GEMM rung: kokoro's measured winner for tall, narrow
// convolutions (32×64 tiles, wave32), and its 32×32 sibling for conv2's
// two output channels.
type convKernel struct {
	spirv  []byte
	bm, bn int
}

var (
	kConv   = convKernel{shaders.KokoroConvReg32x64W32, 32, 64}
	kNarrow = convKernel{shaders.KokoroConvReg32x32W32, 32, 32}
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func parallel(n int, fn func(i int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(w)
	for ; w > 0; w-- {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// packB lays a [n, k] weight (K = tap·in + c) out as dit_gemm.comp's
// B_LAYOUT=2 fragment tiles, kt fastest; rows past len(w)/k are zero.
func packB(dst []uint16, w []float32, n, k int) {
	kt := k / tile
	rows := len(w) / k
	parallel((n+63)/64, func(c int) {
		for i := c * 64; i < min((c+1)*64, rows); i++ {
			base := (i / tile) * kt * tile * tile
			lane := (i % tile) * tile
			for j, v := range w[i*k : (i+1)*k] {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

// weightNorm folds g·v/‖v‖, the norm over every axis but the first.
func weightNorm(g, v []float32) []float32 {
	per := len(v) / len(g)
	out := make([]float32, len(v))
	for i := range g {
		var s float64
		for _, x := range v[i*per : (i+1)*per] {
			s += float64(x) * float64(x)
		}
		f := float64(g[i]) / math.Sqrt(s)
		for j, x := range v[i*per : (i+1)*per] {
			out[i*per+j] = float32(float64(x) * f)
		}
	}
	return out
}

type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string, n int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err == nil && len(v) != n {
		err = fmt.Errorf("vae: %s has %d values, want %d", name, len(v), n)
	}
	l.err = err
	return v
}

// NewGPU stages the decoder from dir (the checkpoint's vae/) for tiles of
// up to window latents.
func NewGPU(dev *vk.Device, dir string, window int) (*GPU, error) {
	if window <= 2*Margin {
		return nil, fmt.Errorf("vae: a %d-latent window leaves no core past two %d-latent margins", window, Margin)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	g := &GPU{dev: dev, Window: window, pipes: map[string]*vk.ComputePipeline{}}
	if err := g.stage(set); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func (g *GPU) stage(set *safetensors.Set) error {
	l := &loader{set: set}
	var bank []uint16
	var w32 []float32
	put32 := func(v []float32) uint32 {
		off := uint32(len(w32))
		w32 = append(w32, v...)
		return off
	}
	// A convolution [out, in, taps] as B[o, j·in + c], N padded to bn.
	addConv := func(p string, out, in, taps, dil, bn int, bias bool) conv {
		v := l.f32(p+".weight_v", out*in*taps)
		gg := l.f32(p+".weight_g", out)
		c := conv{in: in, n: roundUp(out, bn), taps: taps, dil: dil, bias: noW}
		if l.err != nil {
			return c
		}
		w := weightNorm(gg, v)
		k := taps * in
		b := make([]float32, out*k)
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				for j := 0; j < taps; j++ {
					b[o*k+j*in+i] = w[(o*in+i)*taps+j]
				}
			}
		}
		c.bOff = uint32(len(bank))
		buf := make([]uint16, c.n*k)
		packB(buf, b, c.n, k)
		bank = append(bank, buf...)
		if bias {
			c.bias = put32(l.f32(p+".bias", out))
		}
		return c
	}
	// ConvTranspose1d [in, out, 2s] as the 2-tap GEMM: N = s·out, row
	// r·out + o; K = 2·in, tap 0 the earlier input frame (weight index
	// r + s), tap 1 the later (index r).
	addUp := func(p string, in, out, s int) conv {
		v := l.f32(p+".weight_v", in*out*2*s)
		gg := l.f32(p+".weight_g", in)
		c := conv{in: in, n: s * out, taps: 2, dil: 1}
		if l.err != nil {
			return c
		}
		w := weightNorm(gg, v)
		n, k := s*out, 2*in
		b := make([]float32, n*k)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				row := b[(r*out+o)*k:]
				for i := 0; i < in; i++ {
					row[i] = w[(i*out+o)*2*s+r+s]
					row[in+i] = w[(i*out+o)*2*s+r]
				}
			}
		}
		c.bOff = uint32(len(bank))
		buf := make([]uint16, n*k)
		packB(buf, b, n, k)
		bank = append(bank, buf...)
		c.bias = put32(l.f32(p+".bias", out))
		return c
	}
	addSnake := func(p string, ch int) snake {
		a := l.f32(p+".alpha", ch)
		b := l.f32(p+".beta", ch)
		if l.err != nil {
			return 0
		}
		v := make([]float32, 2*ch)
		for i := 0; i < ch; i++ {
			v[i] = float32(math.Exp(float64(a[i])))
			v[ch+i] = 1 / (float32(math.Exp(float64(b[i]))) + 1e-9)
		}
		return snake(put32(v))
	}

	g.conv1 = addConv("decoder.conv1", widths[0], Latent, 7, 1, kConv.bn, true)
	for i, s := range strides {
		p := fmt.Sprintf("decoder.block.%d.", i)
		b := block{stride: s, in: widths[i], out: widths[i+1]}
		b.s = addSnake(p+"snake1", b.in)
		b.up = addUp(p+"conv_t1", b.in, b.out, s)
		for r, d := range dilation {
			q := fmt.Sprintf("%sres_unit%d.", p, r+1)
			b.res[r] = resUnit{
				s1: addSnake(q+"snake1", b.out),
				c1: addConv(q+"conv1", b.out, b.out, 7, d, kConv.bn, true),
				s2: addSnake(q+"snake2", b.out),
				c2: addConv(q+"conv2", b.out, b.out, 1, 1, kConv.bn, true),
			}
		}
		g.blocks = append(g.blocks, b)
	}
	g.final = addSnake("decoder.snake1", widths[len(widths)-1])
	g.conv2 = addConv("decoder.conv2", Channels, widths[len(widths)-1], 7, 1, kNarrow.bn, false)
	if l.err != nil {
		return l.err
	}

	var err error
	if g.bank, err = g.dev.NewBuffer(len(bank) * 2); err != nil {
		return err
	}
	g.bank.WriteUint16At(0, bank)
	if g.wbuf, err = g.dev.NewBuffer(len(w32) * 4); err != nil {
		return err
	}
	g.wbuf.WriteFloat32At(0, w32)

	// Activations, sized by the widest stage of a full window.
	W := g.Window
	maxX, maxUp, rows := 0, 0, W
	for i, s := range strides {
		maxX = max(maxX, roundUp(rows, kConv.bm)*widths[i])
		maxUp = max(maxUp, roundUp(rows+1, kConv.bm)*s*widths[i+1])
		rows *= s
		maxX = max(maxX, roundUp(rows, kConv.bm)*widths[i+1])
	}
	var elems int
	alloc := func(n int) uint32 { off := uint32(elems); elems += roundUp(n, 64); return off }
	g.aX, g.aC, g.aUp = alloc(maxX), alloc(maxX), alloc(maxUp)
	g.aOut = alloc(roundUp(rows, kNarrow.bm) * kNarrow.bn)
	g.aWav = alloc(rows * Channels)
	if elems*4 > 0xfffffffc {
		return fmt.Errorf("vae: a %d-latent window needs a %d MB activation arena, past the storage-buffer range", W, elems*4>>20)
	}
	if g.abuf, err = g.dev.NewBuffer(elems * 4); err != nil {
		return err
	}
	// Row 0 sits one border in at the widest width, so a tap before the
	// first frame reads zeros at any width; after the data come the rows a
	// stage's taps and M padding read, which its Snake pass zeroes.
	g.hA = uint32(border * widths[0])
	tail := 0
	for i := range widths {
		r := W
		for _, s := range strides[:min(i, len(strides))] {
			r *= s
		}
		tail = max(tail, (r+1+border)*widths[i])
	}
	hElems := int(g.hA) + tail
	if hElems*2 > 0xfffffffc {
		return fmt.Errorf("vae: a %d-latent window needs a %d MB fp16 arena, past the storage-buffer range", W, hElems*2>>20)
	}
	if g.hbuf, err = g.dev.NewBuffer(hElems * 2); err != nil {
		return err
	}
	g.hbuf.ZeroUint16At(0, hElems)
	return nil
}

func (g *GPU) build() error {
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	newPipe := func(spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
		mod, err := g.dev.NewShaderModule(spirv)
		if err != nil {
			return nil, err
		}
		g.mods = append(g.mods, mod)
		return g.dev.NewPipeline(mod, spec)
	}
	for name, spirv := range map[string][]byte{
		"snake":       shaders.ACEVAESnake,
		"snake split": shaders.ACEVAESnakeSplit,
		"pick":        shaders.ACEVAEPick,
		"upadd":       shaders.KokoroUpAdd,
		"residual":    shaders.KokoroResidual,
		"copy":        shaders.KokoroCopy,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("vae: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	for name, k := range map[string]convKernel{"conv": kConv, "narrow": kNarrow} {
		p, err := newPipe(k.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank},
			PushConstantSize: pcSize, RequiredSubgroupSize: 32})
		if err != nil {
			return fmt.Errorf("vae: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
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
}

// Bytes is what the decoder holds on the device.
func (g *GPU) Bytes() int { return g.wbuf.Size() + g.abuf.Size() + g.hbuf.Size() + g.bank.Size() }

type graph struct {
	g     *GPU
	d     []vk.MultiDispatch
	kinds []string
}

func (gr *graph) add(pipe, kind string, gx uint32, pc pushConstants) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: gx, GroupsY: 1, PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
}

func groups(n int) uint32 { return uint32((n + 255) / 256) }

func log2(n int) uint32 { return uint32(bits.TrailingZeros(uint(n))) }

// snake records x [rows, C] (+ bias) → Snake → fp16 into the arena, with
// the zero rows after it.
func (gr *graph) snake(kind string, x uint32, rows, ch int, s snake, bias uint32) {
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.WOff, pc.KOff = x, gr.g.hA, uint32(s), bias
	pc.Tokens, pc.Dim, pc.Span, pc.Aux2 = uint32(rows), uint32(ch), border, log2(ch)
	gr.add("snake", kind, groups((rows+border)*ch), pc)
}

// conv records the A_CONV GEMM of c over rows of the arena (row stride
// c.in) into out, [roundUp(rows, bm), c.n] fp32.
func (gr *graph) conv(kind string, c conv, rows int, out uint32, k convKernel, pipe string) {
	pad := (c.taps - 1) * c.dil / 2
	if c.taps == 2 {
		pad = 1 // the transposed convolution reads frames q-1 and q
	}
	m := roundUp(rows, k.bm)
	var pc pushConstants
	pc.InOff = gr.g.hA - uint32(pad*c.in)
	pc.OutOff, pc.BOff = out, c.bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(m), uint32(c.n), uint32(c.taps*c.in), uint32(c.in)
	pc.Aux0, pc.Aux1, pc.Aux2 = uint32(c.in), uint32(c.dil*c.in), log2(c.in)
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: uint32(c.n / k.bn), GroupsY: uint32(m / k.bm),
		PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
}

// tileGraph records the decode of n latents already in the arena, leaving
// [n·1920, 2] at aWav.
func (g *GPU) tileGraph(n int) *graph {
	gr := &graph{g: g}
	rows := n
	gr.conv("conv1", g.conv1, rows, g.aC, kConv, "conv")
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, g.conv1.bias
	pc.Tokens, pc.Dim = uint32(rows), uint32(widths[0])
	gr.add("copy", "conv1 bias", groups(rows*widths[0]), pc)
	for bi, b := range g.blocks {
		gr.snake(fmt.Sprintf("b%d snake", bi), g.aX, rows, b.in, b.s, noW)
		gr.conv(fmt.Sprintf("b%d up", bi), b.up, rows+1, g.aUp, kConv, "conv")
		rows *= b.stride
		var pc pushConstants
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = g.aUp+uint32(b.stride/2*b.out), g.aX, b.up.bias, noW
		pc.Tokens, pc.Dim, pc.Aux2 = uint32(rows), uint32(b.out), log2(b.out)
		gr.add("upadd", fmt.Sprintf("b%d up bias", bi), groups(rows*b.out), pc)
		for ri, r := range b.res {
			k := fmt.Sprintf("b%d r%d", bi, ri)
			gr.snake(k+" snake1", g.aX, rows, b.out, r.s1, noW)
			gr.conv(k+" conv1", r.c1, rows, g.aC, kConv, "conv")
			gr.snake(k+" snake2", g.aC, rows, b.out, r.s2, r.c1.bias)
			gr.conv(k+" conv2", r.c2, rows, g.aC, kConv, "conv")
			var pc pushConstants
			pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, r.c2.bias
			pc.Tokens, pc.Dim = uint32(rows), uint32(b.out)
			gr.add("residual", k+" add", groups(rows*b.out), pc)
		}
	}
	last := widths[len(widths)-1]
	gr.snake("final snake", g.aX, rows, last, g.final, noW)
	gr.conv("conv2", g.conv2, rows, g.aOut, kNarrow, "narrow")
	pc = pushConstants{}
	pc.InOff, pc.OutOff, pc.Tokens, pc.Dim, pc.LDA = g.aOut, g.aWav, uint32(rows), Channels, uint32(g.conv2.n)
	gr.add("pick", "pick", groups(rows*Channels), pc)
	return gr
}

// upload writes latents rows [from, to) of z ([T, 64] time-major) into the
// arena as conv1's A operand, with zero rows after them.
func (g *GPU) upload(z []float32, from, to int) {
	n := to - from
	buf := make([]uint16, (n+border)*Latent)
	for i, v := range z[from*Latent : to*Latent] {
		buf[i] = safetensors.F32ToF16(v)
	}
	g.hbuf.WriteUint16At(int(g.hA), buf)
}

// Decode turns latents z ([T, 64], time-major) into interleaved stereo
// [T·1920, 2] at 48 kHz, in overlap-discard tiles of at most Window
// latents, and returns the device time.
func (g *GPU) Decode(z []float32) ([]float32, time.Duration, error) {
	if len(z)%Latent != 0 || len(z) == 0 {
		return nil, 0, fmt.Errorf("vae: %d values is not a whole number of %d-channel latents", len(z), Latent)
	}
	T := len(z) / Latent
	out := make([]float32, T*Hop*Channels)
	var took time.Duration
	core := g.Window - 2*Margin
	if T <= g.Window {
		core = T
	}
	for c0 := 0; c0 < T; c0 += core {
		c1 := min(c0+core, T)
		w0, w1 := max(0, c0-Margin), min(T, c1+Margin)
		if T <= g.Window {
			w0, w1 = 0, T
		}
		if c0 > 0 && g.Between != nil {
			if err := g.Between(); err != nil {
				return nil, took, err
			}
		}
		g.upload(z, w0, w1)
		gr := g.tileGraph(w1 - w0)
		t, err := vk.DispatchMultiTimed(gr.d, 1, 1, true)
		if err != nil {
			return nil, took, fmt.Errorf("vae: tile at latent %d: %w", c0, err)
		}
		took += t
		skip := (c0 - w0) * Hop * Channels
		n := (c1 - c0) * Hop * Channels
		copy(out[c0*Hop*Channels:], g.abuf.ReadFloat32At(int(g.aWav)+skip, n))
	}
	return out, took, nil
}

// Stage is one dispatch's measurement.
type Stage struct {
	Kind string
	GPU  time.Duration
}

// Profile runs one tile of n latents (zeros) a dispatch at a time.
func (g *GPU) Profile(n int) ([]Stage, error) {
	g.upload(make([]float32, n*Latent), 0, n)
	gr := g.tileGraph(n)
	var out []Stage
	for i := range gr.d {
		t, err := vk.DispatchMultiTimed(gr.d[i:i+1], 1, 1, true)
		if err != nil {
			return out, err
		}
		out = append(out, Stage{gr.kinds[i], t})
	}
	return out, nil
}

// ClampPeak is the first half of Normalize, generate_music_decode's: divide
// by the peak when it passes 1. A repaint splices its source in after it.
func ClampPeak(wav []float32) float32 {
	var peak float32
	for _, v := range wav {
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	if peak > 1 {
		for i := range wav {
			wav[i] /= peak
		}
		peak = 1
	}
	return peak
}

// Normalize is what upstream does to a decode before writing it: divide by
// the peak when it passes 1 (generate_music_decode), then scale the peak to
// db dBFS (audio_utils.normalize_audio, -1 by default), leaving
// near-silence (peak < 1e-6) alone.
func Normalize(wav []float32, db float64) {
	peak := ClampPeak(wav)
	if peak < 1e-6 {
		return
	}
	gain := float32(math.Pow(10, db/20)) / peak
	for i := range wav {
		wav[i] *= gain
	}
}
