package mm3

// The vocoder (N6, upstream's Flow-VAE decoder): a DAC-style decoder that
// turns a window's 128 latent channels into stereo at 44.1 kHz, the two
// channels decoded separately from 64 latent channels each.
//
// It is ACE-Step's Oobleck decoder (ace/vae, MUSIC.md A5) in structure --
// conv, four blocks of Snake + ConvTranspose1d (kernel 2s, stride s) + three
// residual units (Snake, k7 conv at dilation 1/3/9, Snake, k1 conv, add), a
// final Snake and conv -- so it runs on that host's kernels: every
// convolution is kokoro's A_CONV GEMM over a channel-last fp16 activation
// with zero rows for the padding, the transposed convolution is a 2-tap
// GEMM s times wider, and Snake + bias + narrowing is one pass
// (ace_vae.comp). Weight norm is folded at load. What differs:
//
//   - **The widths are 3·2^k** (1536 → 96), and A_CONV and the Snake pass
//     address by powers of two. Every width is padded to the next one
//     (2048, 1024, 512, 256, 128) with zero weights, zero biases and a
//     Snake of α = 0 there, which keeps a pad channel at zero through
//     every stage. It costs up to 1.8x the FLOPs of a ~110 GFLOP-a-second
//     decode, ~10 ms a second of audio.
//   - **Snake has α only**: x + sin²(αx)/(α + 1e-9). ace_vae.comp's form is
//     x + sin²(αx)·γ with γ given, so γ = 1/(α + 1e-9).
//   - **A k1 input projection** (dec_in_proj, 64 → 1024, a bias, no weight
//     norm) before conv_in: its bias rides a Snake pass at α = 0 (sin 0 is
//     0, so the pass is bias + narrowing).
//   - **The output is one channel and a tanh**: conv_out (96 → 1, a bias)
//     on the narrow 32-column rung, then bias and tanh on the host.
//   - **No tiling.** Upstream decodes a window's latents whole (≤ 689) and
//     crops; so does this, a window a call.

import (
	"fmt"
	"math"
	"math/bits"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	// Hop is the samples a latent decodes to; SampleRate the output rate.
	Hop        = 512
	SampleRate = 44100

	vLatent = LatentChannels / 2 // a channel's latents
	vProj   = 1024
	vBorder = 64 // zero rows around the fp16 arena's data (ace/vae's border)
	noW     = 0xffffffff
)

var (
	vStrides  = []int{8, 8, 4, 2}
	vWidths   = []int{1536, 768, 384, 192, 96} // before block i, and after the last
	vDilation = []int{1, 3, 9}
)

func pow2(n int) int { return 1 << bits.Len(uint(n-1)) }

type vConv struct {
	bOff        uint32
	bias        uint32
	in, n, taps int // in: the padded input width, n: the padded output width
	dil         int
}

type vSnake uint32 // α then γ = 1/(α + 1e-9), C each (padded) in wbuf

type vRes struct {
	s1, s2 vSnake
	c1, c2 vConv
}

type vBlock struct {
	s       vSnake
	up      vConv
	stride  int
	in, out int // padded
	res     [3]vRes
}

type convKernel struct {
	spirv  []byte
	bm, bn int
}

var (
	vkConv   = convKernel{shaders.KokoroConvReg32x64W32, 32, 64}
	vkNarrow = convKernel{shaders.KokoroConvReg32x32W32, 32, 32}
)

// Vocoder is the decoder, resident.
type Vocoder struct {
	dev                    *vk.Device
	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule
	window                 int

	proj, convIn, convOut vConv
	projSnake             vSnake // α = 0: bias + narrowing
	blocks                []vBlock
	final                 vSnake
	outBias               float32

	aX, aC, aUp, aOut, aWav uint32
	hA                      uint32
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

// LoadVocoder stages the vocoder (models/MiniMax-Music3/vocoder) for
// windows of up to window latents.
func LoadVocoder(dev *vk.Device, dir string, window int) (*Vocoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	g := &Vocoder{dev: dev, window: window, pipes: map[string]*vk.ComputePipeline{}}
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

func (g *Vocoder) stage(set *safetensors.Set) error {
	var lerr error
	f32 := func(name string, n int) []float32 {
		if lerr != nil {
			return nil
		}
		var v []float32
		v, lerr = tF32(set, name, n)
		return v
	}
	var bank []uint16
	var w32 []float32
	put32 := func(v []float32) uint32 {
		off := uint32(len(w32))
		w32 = append(w32, v...)
		return off
	}
	pad32 := func(v []float32, n int) []float32 {
		out := make([]float32, n)
		copy(out, v)
		return out
	}
	// A convolution [out, in, taps] (real widths) as B[o, j·inP + c], N
	// padded to outP and the kernel's tile; w is already weight-normed.
	addConvW := func(w, bias []float32, out, in, outP, inP, taps, dil, bn int) vConv {
		c := vConv{in: inP, n: roundUp(outP, bn), taps: taps, dil: dil, bias: noW}
		k := taps * inP
		b := make([]float32, c.n*k)
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				for j := 0; j < taps; j++ {
					b[o*k+j*inP+i] = w[(o*in+i)*taps+j]
				}
			}
		}
		c.bOff = uint32(len(bank))
		buf := make([]uint16, c.n*k)
		tileB(buf, c.n, k, func(r int, dst []float32) { copy(dst, b[r*k:(r+1)*k]) })
		bank = append(bank, buf...)
		if bias != nil {
			c.bias = put32(pad32(bias, outP))
		}
		return c
	}
	addConv := func(p string, out, in, taps, dil, bn int) vConv {
		v := f32(p+".weight_v", out*in*taps)
		gg := f32(p+".weight_g", out)
		bias := f32(p+".bias", out)
		if lerr != nil {
			return vConv{}
		}
		return addConvW(weightNorm(gg, v), bias, out, in, pow2(out), pow2(in), taps, dil, bn)
	}
	// ConvTranspose1d [in, out, 2s] as the 2-tap GEMM: N = s·outP, row
	// r·outP + o; K = 2·inP, tap 0 the earlier input frame (weight index
	// r + s), tap 1 the later (index r) -- ace/vae's addUp.
	addUp := func(p string, in, out, s int) vConv {
		v := f32(p+".weight_v", in*out*2*s)
		gg := f32(p+".weight_g", in)
		bias := f32(p+".bias", out)
		inP, outP := pow2(in), pow2(out)
		c := vConv{in: inP, n: s * outP, taps: 2, dil: 1}
		if lerr != nil {
			return c
		}
		w := weightNorm(gg, v)
		n, k := s*outP, 2*inP
		b := make([]float32, n*k)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				row := b[(r*outP+o)*k:]
				for i := 0; i < in; i++ {
					row[i] = w[(i*out+o)*2*s+r+s]
					row[inP+i] = w[(i*out+o)*2*s+r]
				}
			}
		}
		c.bOff = uint32(len(bank))
		buf := make([]uint16, n*k)
		tileB(buf, n, k, func(r int, dst []float32) { copy(dst, b[r*k:(r+1)*k]) })
		bank = append(bank, buf...)
		c.bias = put32(pad32(bias, outP))
		return c
	}
	addSnake := func(p string, ch int) vSnake {
		a := f32(p+".alpha", ch)
		if lerr != nil {
			return 0
		}
		cp := pow2(ch)
		v := make([]float32, 2*cp)
		for i := 0; i < ch; i++ {
			v[i] = a[i]
			v[cp+i] = 1 / (a[i] + 1e-9)
		}
		return vSnake(put32(v))
	}

	pw := f32("dec_in_proj.weight", vProj*vLatent)
	pb := f32("dec_in_proj.bias", vProj)
	if lerr != nil {
		return lerr
	}
	g.proj = addConvW(pw, nil, vProj, vLatent, vProj, vLatent, 1, 1, vkConv.bn)
	g.proj.bias = put32(pb)
	g.projSnake = vSnake(put32(make([]float32, 2*vProj)))
	g.convIn = addConv("conv_in", vWidths[0], vProj, 7, 1, vkConv.bn)
	for i, s := range vStrides {
		p := fmt.Sprintf("blocks.%d.", i)
		in, out := vWidths[i], vWidths[i+1]
		b := vBlock{stride: s, in: pow2(in), out: pow2(out)}
		b.s = addSnake(p+"snake1", in)
		b.up = addUp(p+"conv_t1", in, out, s)
		for r, d := range vDilation {
			q := fmt.Sprintf("%sres_unit%d.", p, r+1)
			b.res[r] = vRes{
				s1: addSnake(q+"snake1", out),
				c1: addConv(q+"conv1", out, out, 7, d, vkConv.bn),
				s2: addSnake(q+"snake2", out),
				c2: addConv(q+"conv2", out, out, 1, 1, vkConv.bn),
			}
		}
		g.blocks = append(g.blocks, b)
	}
	last := vWidths[len(vWidths)-1]
	g.final = addSnake("snake_out", last)
	ov := f32("conv_out.weight_v", last*7)
	og := f32("conv_out.weight_g", 1)
	ob := f32("conv_out.bias", 1)
	if lerr != nil {
		return lerr
	}
	g.convOut = addConvW(weightNorm(og, ov), nil, 1, last, 1, pow2(last), 7, 1, vkNarrow.bn)
	g.outBias = ob[0]

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
	W := g.window
	maxX := roundUp(W, vkConv.bm) * max(vProj, pow2(vWidths[0]))
	maxUp, rows := 0, W
	for i, s := range vStrides {
		in, out := pow2(vWidths[i]), pow2(vWidths[i+1])
		maxX = max(maxX, roundUp(rows, vkConv.bm)*in)
		maxUp = max(maxUp, roundUp(rows+1, vkConv.bm)*s*out)
		rows *= s
		maxX = max(maxX, roundUp(rows, vkConv.bm)*out)
	}
	var elems int
	alloc := func(n int) uint32 { off := uint32(elems); elems += roundUp(n, 64); return off }
	g.aX, g.aC, g.aUp = alloc(maxX), alloc(maxX), alloc(maxUp)
	g.aOut = alloc(roundUp(rows, vkNarrow.bm) * vkNarrow.bn)
	g.aWav = alloc(rows)
	if elems*4 > maxBankBytes {
		return fmt.Errorf("mm3: a %d-latent window needs a %d MB vocoder arena", W, elems*4>>20)
	}
	if g.abuf, err = g.dev.NewHostCachedBuffer(elems * 4); err != nil {
		return err
	}
	widest := max(vProj, pow2(vWidths[0]))
	g.hA = uint32(vBorder * widest)
	tail := (W + 1 + vBorder) * widest
	r := W
	for i, s := range vStrides {
		r *= s
		tail = max(tail, (r+1+vBorder)*pow2(vWidths[i+1]))
	}
	hElems := int(g.hA) + tail
	if hElems*2 > maxBankBytes {
		return fmt.Errorf("mm3: a %d-latent window needs a %d MB fp16 vocoder arena", W, hElems*2>>20)
	}
	if g.hbuf, err = g.dev.NewBuffer(hElems * 2); err != nil {
		return err
	}
	g.hbuf.ZeroUint16At(0, hElems)
	return nil
}

func (g *Vocoder) build() error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	pipe := func(name string, spirv []byte, spec vk.PipelineSpec) error {
		mod, err := g.dev.NewShaderModule(spirv)
		if err != nil {
			return fmt.Errorf("mm3: vocoder shader %s: %w", name, err)
		}
		g.mods = append(g.mods, mod)
		p, err := g.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("mm3: vocoder pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
		return nil
	}
	for name, spirv := range map[string][]byte{
		"snake":    shaders.ACEVAESnake,
		"pick":     shaders.ACEVAEPick,
		"upadd":    shaders.KokoroUpAdd,
		"residual": shaders.KokoroResidual,
		"copy":     shaders.KokoroCopy,
	} {
		if err := pipe(name, spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pc}); err != nil {
			return err
		}
	}
	for name, k := range map[string]convKernel{"conv": vkConv, "narrow": vkNarrow} {
		if err := pipe(name, k.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank},
			PushConstantSize: pc, RequiredSubgroupSize: 32}); err != nil {
			return err
		}
	}
	return nil
}

// Destroy releases every Vulkan object.
func (g *Vocoder) Destroy() {
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

func vGroups(n int) uint32 { return uint32((n + 255) / 256) }

func log2(n int) uint32 { return uint32(bits.TrailingZeros(uint(n))) }

type vGraph struct {
	g *Vocoder
	d []vk.MultiDispatch
}

func (gr *vGraph) add(pipe string, gx uint32, p push) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: gx, GroupsY: 1, PushConstants: p.bytes()})
}

// snake records x [rows, C] (+ bias) → Snake → fp16 into the arena, with
// the zero rows after it.
func (gr *vGraph) snake(x uint32, rows, ch int, s vSnake, bias uint32) {
	gr.add("snake", vGroups((rows+vBorder)*ch), push{InOff: x, OutOff: gr.g.hA, WOff: uint32(s), KOff: bias,
		Tokens: uint32(rows), Dim: uint32(ch), Span: vBorder, Aux2: log2(ch)})
}

// conv records the A_CONV GEMM of c over rows of the arena into out.
func (gr *vGraph) conv(c vConv, rows int, out uint32, k convKernel, pipe string) {
	pad := (c.taps - 1) * c.dil / 2
	if c.taps == 2 {
		pad = 1 // the transposed convolution reads frames q-1 and q
	}
	m := roundUp(rows, k.bm)
	p := push{InOff: gr.g.hA - uint32(pad*c.in), OutOff: out, BOff: c.bOff,
		GemmM: uint32(m), GemmN: uint32(c.n), GemmK: uint32(c.taps * c.in), LDA: uint32(c.in),
		Aux0: uint32(c.in), Aux1: uint32(c.dil * c.in), Aux2: log2(c.in)}
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: uint32(c.n / k.bn), GroupsY: uint32(m / k.bm),
		PushConstants: p.bytes()})
}

// graph records the decode of n latents already in the arena, leaving the
// pre-bias, pre-tanh waveform [n·512] at aWav.
func (g *Vocoder) graph(n int) []vk.MultiDispatch {
	gr := &vGraph{g: g}
	rows := n
	gr.conv(g.proj, rows, g.aC, vkConv, "conv")
	gr.snake(g.aC, rows, vProj, g.projSnake, g.proj.bias)
	gr.conv(g.convIn, rows, g.aC, vkConv, "conv")
	gr.add("copy", vGroups(rows*g.convIn.n), push{InOff: g.aC, OutOff: g.aX, WOff: g.convIn.bias, Tokens: uint32(rows),
		Dim: uint32(g.convIn.n)})
	for _, b := range g.blocks {
		gr.snake(g.aX, rows, b.in, b.s, noW)
		gr.conv(b.up, rows+1, g.aUp, vkConv, "conv")
		rows *= b.stride
		gr.add("upadd", vGroups(rows*b.out), push{InOff: g.aUp + uint32(b.stride/2*b.out), OutOff: g.aX, WOff: b.up.bias,
			Aux0: noW, Tokens: uint32(rows), Dim: uint32(b.out), Aux2: log2(b.out)})
		for _, r := range b.res {
			gr.snake(g.aX, rows, b.out, r.s1, noW)
			gr.conv(r.c1, rows, g.aC, vkConv, "conv")
			gr.snake(g.aC, rows, b.out, r.s2, r.c1.bias)
			gr.conv(r.c2, rows, g.aC, vkConv, "conv")
			gr.add("residual", vGroups(rows*b.out), push{InOff: g.aC, OutOff: g.aX, WOff: r.c2.bias, Tokens: uint32(rows),
				Dim: uint32(b.out)})
		}
	}
	last := pow2(vWidths[len(vWidths)-1])
	gr.snake(g.aX, rows, last, g.final, noW)
	gr.conv(g.convOut, rows, g.aOut, vkNarrow, "narrow")
	gr.add("pick", vGroups(rows), push{InOff: g.aOut, OutOff: g.aWav, Tokens: uint32(rows), Dim: 1, LDA: uint32(g.convOut.n)})
	return gr.d
}

// Decode turns one window's latents ([128][L], channel-major) into stereo,
// [2][L·512], as upstream's vocoder call does (no crop).
func (g *Vocoder) Decode(latents [][]float32) ([2][]float32, time.Duration, error) {
	var out [2][]float32
	if len(latents) != LatentChannels {
		return out, 0, fmt.Errorf("mm3: %d latent channels", len(latents))
	}
	L := len(latents[0])
	if L == 0 || L > g.window {
		return out, 0, fmt.Errorf("mm3: a %d-latent window; at most %d", L, g.window)
	}
	var took time.Duration
	for ch := 0; ch < 2; ch++ {
		buf := make([]uint16, (L+vBorder)*vLatent)
		for i := 0; i < L; i++ {
			for c := 0; c < vLatent; c++ {
				buf[i*vLatent+c] = safetensors.F32ToF16(latents[ch*vLatent+c][i])
			}
		}
		g.hbuf.WriteUint16At(int(g.hA), buf)
		t, err := vk.DispatchMultiTimed(g.graph(L), 1, 1, true)
		took += t
		if err != nil {
			return out, took, fmt.Errorf("mm3: vocoder channel %d: %w", ch, err)
		}
		w := g.abuf.ReadFloat32At(int(g.aWav), L*Hop)
		for i, v := range w {
			w[i] = float32(math.Tanh(float64(v + g.outBias)))
		}
		out[ch] = w
	}
	return out, took, nil
}
