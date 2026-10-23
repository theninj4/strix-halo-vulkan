package mm3

// The flow-matching DiT (N4): 36 pre-norm blocks at 2048 wide, 32 heads of
// 64, self-attention only, over a window's latents with the timestep's
// embedding prepended as a token. It is the H3 video VAE's ViT (h3/vae,
// VIDEO.md M5) in all but four things, so it runs on that host's kernels:
// the fragment-tiled LDS GEMM with the bias in a ones column, its fused
// SwiGLU epilogue (the silu'd half second, as here), the head-64 q/k and v
// packs and the transposed head-64 attention. What differs:
//
//   - **LayerNorm** with a weight and a bias (parakeet_layernorm.comp's
//     OUT_F16 build), not the VAE's modulated RMS norm.
//   - **No q/k norm**, and a 1-D rotate-half rope over 32 of a head's 64
//     channels at position = the token's index (the time token is 0):
//     mm3_qk_pack_hd64, h3_qk_pack.comp at QK_NORM=0.
//   - **No modulation.** The timestep only enters as token 0: the host
//     computes its embedding (Fourier features, then a SiLU MLP) in fp32 and
//     adds it into that row of the residual.
//   - **The two 1x1 convolutions fold into the projections.** The input is
//     [latents 128 | zeros 128 | condition 2048] through conv + residual,
//     then proj_in: proj_in·(conv + I) with the zero channels' columns
//     dropped is one [2048, 2176] weight. The output is proj_out then conv +
//     residual: (conv + I)·proj_out is one [128, 2048]. Both are formed on
//     the host in float64. The model has no norm before proj_out, so the
//     residual is narrowed into its A operand as it is (mm3_narrow.comp).
//
// A forward is the CFG pair, the condition and zeros, as two sequences side
// by side at a 16-row stride; every row-local stage is one dispatch over
// both and only the attention runs per sequence.

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const (
	tH       = 2048 // inner width
	tHeads   = 32
	tHD      = 64
	tFFN     = 8192
	tLayers  = 36
	tRope    = 32 // rotated channels a head
	tTheta   = 10000
	tFourier = 256
	tLNEps   = 1e-5

	// LatentChannels and CondChannels are the DiT's input widths.
	LatentChannels = 128
	CondChannels   = 2048

	tBK      = 32 // the LDS GEMM's K tile
	tBigBM   = 128
	tBigBN   = 256
	tHeadBN  = 128 // reg32x128, for proj_out's 128 columns
	tHeadBM  = 32
	tKeyPad  = 8 * tile // the longest key block the attention reads past its count
	tLog2e   = 1.4426950408889634
	tSubmitN = 6 // blocks a submission
	condTaps = 3 // the condition encoder's kernel
)

// DiT is the transformer, resident (4.9 GB of fp16).
type DiT struct {
	dev              *vk.Device
	wbuf, abuf, hbuf *vk.Buffer
	banks            []*vk.Buffer
	pipes            map[string]*vk.ComputePipeline
	gemms            []map[string]*vk.ComputePipeline
	mods             []*vk.ShaderModule
	attn             string // the attention build: transposed when the device's element order allows

	maxLatents      int // per sequence
	stride, maxRows int // a sequence's rows (16-aligned), both sequences' rows (128-aligned)
	kH, kF, kIn, kC int
	ldaH, ldaF      int
	ldaIn, ldaC     int
	cset            *safetensors.Set // the condition encoder's, while staging

	blocks []tBlock
	in     tHead // [2048, kIn]: the folded preprocess conv and proj_in
	out    tHead // [128, kH]: the folded proj_out and postprocess conv
	cond   tHead // [2048, kC]: the condition encoder's k3 conv, taps-major, its bias in column 3·4096

	// fp32 time MLP, on the host
	fourier            []float32 // [128]
	t1w, t1b, t2w, t2b []float32

	// wbuf (fp32): rope tables for maxRows rows, ones, the norms
	wCos, wSin, wOnes uint32
	// abuf (fp32, host-cached)
	aX, aQ, aK, aV, aO, aFF, aTime, aOut uint32
	// hbuf (fp16)
	hQ, hK, hV, hA, hCtx, hFFN, hIn, hCond uint32
	planeRows                              int
}

type tHead struct {
	bank int
	off  uint32
}

type tBlock struct {
	bank                     int
	q, k, v, o, gateUp, down uint32 // in the bank, halves
	n1w, n1b, n2w, n2b       uint32 // wbuf
}

// LoadDiT stages the transformer and the condition encoder's convolution
// (models/MiniMax-Music3/transformer and condition_encoder under root) for
// windows of up to maxLatents latents (689 for a full 200-frame window).
func LoadDiT(dev *vk.Device, root string, maxLatents int) (*DiT, error) {
	g := &DiT{dev: dev, pipes: map[string]*vk.ComputePipeline{}, maxLatents: maxLatents}
	dir := filepath.Join(root, "transformer")
	cset, err := safetensors.OpenSet(filepath.Join(root, "condition_encoder"))
	if err != nil {
		return nil, err
	}
	defer cset.Close()
	g.cset = cset
	g.stride = roundUp(maxLatents+1, tile)
	g.maxRows = roundUp(2*g.stride, tBigBM)
	g.kH, g.kF = roundUp(tH+1, tBK), roundUp(tFFN+1, tBK)
	g.kIn = roundUp(LatentChannels+CondChannels, tBK)
	g.kC = roundUp(condTaps*hidden+1, tBK)
	g.ldaH, g.ldaF, g.ldaIn, g.ldaC = g.kH+gemmPad, g.kF+gemmPad, g.kIn+gemmPad, g.kC+gemmPad
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	for _, step := range []func(*safetensors.Set) error{g.alloc, g.stage, g.build} {
		if err := step(set); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	g.cset = nil
	return g, nil
}

func (g *DiT) alloc(*safetensors.Set) error {
	rows := g.maxRows
	a := 0
	alloc := func(n int) uint32 {
		off := uint32(a)
		a += roundUp(n, 64)
		return off
	}
	g.aX = alloc(rows * tH)
	g.aQ = alloc(rows * tH)
	g.aK = alloc(rows * tH)
	g.aV = alloc(rows * tH)
	g.aO = alloc(rows * tH)
	g.aFF = alloc(rows * tH)
	g.aTime = alloc(tH)
	g.aOut = alloc(rows * LatentChannels)

	h := 0
	halloc := func(n int) uint32 {
		off := uint32(h)
		h += roundUp(n, 64)
		return off
	}
	g.planeRows = rows + tKeyPad
	plane := tHeads * g.planeRows * tHD
	g.hQ, g.hK, g.hV = halloc(plane), halloc(plane), halloc(plane)
	g.hA = halloc(rows * g.ldaH)
	g.hCtx = halloc(rows * g.ldaH)
	g.hFFN = halloc(rows * g.ldaF)
	g.hIn = halloc(rows * g.ldaIn)
	g.hCond = halloc(roundUp(WindowFrames, tBigBM) * g.ldaC)

	half := tRope / 2
	g.wCos, g.wSin = 0, uint32(rows*half)
	g.wOnes = uint32(2 * rows * half)
	w32 := 2*rows*half + tHD
	g.blocks = make([]tBlock, tLayers)
	for i := range g.blocks {
		b := &g.blocks[i]
		b.n1w, b.n1b, b.n2w, b.n2b = uint32(w32), uint32(w32+tH), uint32(w32+2*tH), uint32(w32+3*tH)
		w32 += 4 * tH
	}
	for _, x := range []struct {
		name  string
		bytes int
	}{{"fp32 activation", a * 4}, {"fp16 activation", h * 2}} {
		if x.bytes > maxBankBytes {
			return fmt.Errorf("mm3: the DiT's %s arena is %d MB, past one binding", x.name, x.bytes>>20)
		}
	}
	var err error
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return err
	}
	if g.abuf, err = g.dev.NewHostCachedBuffer(a * 4); err != nil {
		return err
	}
	if g.hbuf, err = g.dev.NewBuffer(h * 2); err != nil {
		return err
	}
	g.hbuf.ZeroUint16At(0, h)
	// The bias columns: a one just past the real width of every A operand
	// that feeds a projection with a bias (and harmlessly the ones that
	// don't, whose weight there is zero).
	one := safetensors.F32ToF16(1)
	for _, x := range []struct {
		off      uint32
		lda, col int
	}{{g.hA, g.ldaH, tH}, {g.hCtx, g.ldaH, tH}, {g.hFFN, g.ldaF, tFFN}} {
		buf := make([]uint16, rows*x.lda)
		for r := 0; r < rows; r++ {
			buf[r*x.lda+x.col] = one
		}
		g.hbuf.WriteUint16At(int(x.off), buf)
	}
	ones := make([]float32, tHD)
	for i := range ones {
		ones[i] = 1
	}
	g.wbuf.WriteFloat32At(int(g.wOnes), ones)

	// Rope: position = the row's index in its sequence, 16 angles a row.
	cos := make([]float32, rows*half)
	sin := make([]float32, rows*half)
	for r := 0; r < rows; r++ {
		p := float64(r % g.stride)
		for j := 0; j < half; j++ {
			inv := 1 / math.Pow(tTheta, float64(2*j)/tRope)
			// upstream: float32 inv_freq, float32 outer product
			ang := float64(float32(p) * float32(inv))
			cos[r*half+j] = float32(math.Cos(ang))
			sin[r*half+j] = float32(math.Sin(ang))
		}
	}
	g.wbuf.WriteFloat32At(int(g.wCos), cos)
	g.wbuf.WriteFloat32At(int(g.wSin), sin)
	return nil
}

func tF32(set *safetensors.Set, name string, n int) ([]float32, error) {
	t, err := set.Get(name)
	if err != nil {
		return nil, err
	}
	v, err := t.F32(nil)
	if err != nil {
		return nil, err
	}
	if len(v) != n {
		return nil, fmt.Errorf("mm3: %s has %d values, want %d", name, len(v), n)
	}
	return v, nil
}

// withBias lays a [n, realK] weight and its [n] bias out as [n, k]: the
// weight, the bias in column realK, zeros after.
func withBias(w, bias []float32, n, realK, k int) []float32 {
	out := make([]float32, n*k)
	for i := 0; i < n; i++ {
		copy(out[i*k:i*k+realK], w[i*realK:(i+1)*realK])
		if bias != nil {
			out[i*k+realK] = bias[i]
		}
	}
	return out
}

// parallelRows runs f(i) for i in [0, n) over the cores.
func parallelRows(n int, f func(i int)) {
	var wg sync.WaitGroup
	w := runtime.GOMAXPROCS(0)
	for c := 0; c < w; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			for i := c; i < n; i += w {
				f(i)
			}
		}(c)
	}
	wg.Wait()
}

func (g *DiT) stage(set *safetensors.Set) error {
	perBlock := 4*tH*g.kH + 2*tFFN*g.kH + tH*g.kF
	var bankElems []int
	place := func(n int) (int, uint32) {
		if len(bankElems) == 0 || (bankElems[len(bankElems)-1]+n)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
		}
		b := len(bankElems) - 1
		off := bankElems[b]
		bankElems[b] += n
		return b, uint32(off)
	}
	g.in.bank, g.in.off = place(tH * g.kIn)
	g.out.bank, g.out.off = place(LatentChannels * g.kH)
	g.cond.bank, g.cond.off = place(CondChannels * g.kC)
	for i := range g.blocks {
		b := &g.blocks[i]
		var off uint32
		b.bank, off = place(perBlock)
		b.q = off
		b.k = b.q + uint32(tH*g.kH)
		b.v = b.k + uint32(tH*g.kH)
		b.o = b.v + uint32(tH*g.kH)
		b.gateUp = b.o + uint32(tH*g.kH)
		b.down = b.gateUp + uint32(2*tFFN*g.kH)
	}
	for _, n := range bankElems {
		buf, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("mm3: DiT bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, buf)
	}
	put := func(bank int, off uint32, w []float32, n, k int) {
		buf := make([]uint16, n*k)
		tileB(buf, n, k, func(r int, dst []float32) { copy(dst, w[r*k:(r+1)*k]) })
		g.banks[bank].WriteUint16At(int(off), buf)
	}

	// proj_in · (preprocess_conv + I), the zero channels' columns dropped.
	cc := 2*LatentChannels + CondChannels
	pre, err := tF32(set, "preprocess_conv.weight", cc*cc)
	if err != nil {
		return err
	}
	pin, err := tF32(set, "proj_in.weight", tH*cc)
	if err != nil {
		return err
	}
	keep := make([]int, 0, LatentChannels+CondChannels) // source channel of each folded column
	for c := 0; c < cc; c++ {
		if c < LatentChannels || c >= 2*LatentChannels {
			keep = append(keep, c)
		}
	}
	in := make([]float32, tH*g.kIn)
	parallelRows(tH, func(o int) {
		row := pin[o*cc : (o+1)*cc]
		for j, c := range keep {
			s := float64(row[c]) // the identity
			for m := 0; m < cc; m++ {
				s += float64(row[m]) * float64(pre[m*cc+c])
			}
			in[o*g.kIn+j] = float32(s)
		}
	})
	put(g.in.bank, g.in.off, in, tH, g.kIn)

	// The condition encoder's Conv1d(4096, 2048, k3, pad 1) as a GEMM over
	// rows [x(f-1) | x(f) | x(f+1) | 1]: column k·4096 + c is weight[o, c, k].
	cw, err := tF32(g.cset, "proj.weight", CondChannels*hidden*condTaps)
	if err != nil {
		return err
	}
	cb, err := tF32(g.cset, "proj.bias", CondChannels)
	if err != nil {
		return err
	}
	cm := make([]float32, CondChannels*g.kC)
	parallelRows(CondChannels, func(o int) {
		for c := 0; c < hidden; c++ {
			for k := 0; k < condTaps; k++ {
				cm[o*g.kC+k*hidden+c] = cw[(o*hidden+c)*condTaps+k]
			}
		}
		cm[o*g.kC+condTaps*hidden] = cb[o]
	})
	put(g.cond.bank, g.cond.off, cm, CondChannels, g.kC)

	// (postprocess_conv + I) · proj_out.
	post, err := tF32(set, "postprocess_conv.weight", LatentChannels*LatentChannels)
	if err != nil {
		return err
	}
	pout, err := tF32(set, "proj_out.weight", LatentChannels*tH)
	if err != nil {
		return err
	}
	out := make([]float32, LatentChannels*g.kH)
	parallelRows(LatentChannels, func(o int) {
		for c := 0; c < tH; c++ {
			s := float64(pout[o*tH+c])
			for m := 0; m < LatentChannels; m++ {
				s += float64(post[o*LatentChannels+m]) * float64(pout[m*tH+c])
			}
			out[o*g.kH+c] = float32(s)
		}
	})
	put(g.out.bank, g.out.off, out, LatentChannels, g.kH)

	// The time MLP, fp32 on the host.
	if g.fourier, err = tF32(set, "time_proj.weight", tFourier/2); err != nil {
		return err
	}
	if g.t1w, err = tF32(set, "time_embed.linear_1.weight", tH*tFourier); err != nil {
		return err
	}
	if g.t1b, err = tF32(set, "time_embed.linear_1.bias", tH); err != nil {
		return err
	}
	if g.t2w, err = tF32(set, "time_embed.linear_2.weight", tH*tH); err != nil {
		return err
	}
	if g.t2b, err = tF32(set, "time_embed.linear_2.bias", tH); err != nil {
		return err
	}

	for i := range g.blocks {
		b := &g.blocks[i]
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		get := func(name string, n int) []float32 {
			if err != nil {
				return nil
			}
			var v []float32
			v, err = tF32(set, p+name, n)
			return v
		}
		wq, wk, wv, wo := get("attn.to_q.weight", tH*tH), get("attn.to_k.weight", tH*tH), get("attn.to_v.weight", tH*tH),
			get("attn.to_out.0.weight", tH*tH)
		ffw, ffb := get("ff_in.weight", 2*tFFN*tH), get("ff_in.bias", 2*tFFN)
		fow, fob := get("ff_out.weight", tH*tFFN), get("ff_out.bias", tH)
		n1w, n1b, n2w, n2b := get("norm1.weight", tH), get("norm1.bias", tH), get("norm2.weight", tH), get("norm2.bias", tH)
		if err != nil {
			return err
		}
		for _, x := range []struct {
			off uint32
			w   []float32
		}{{b.q, wq}, {b.k, wk}, {b.v, wv}, {b.o, wo}} {
			put(b.bank, x.off, withBias(x.w, nil, tH, tH, g.kH), tH, g.kH)
		}
		// ff_in's rows are [value | gate] (chunk(2), the gate silu'd);
		// interleaved for the SwiGLU epilogue, silu'd half first.
		fused := make([]float32, 2*tFFN*g.kH)
		qwen.InterleaveGLU(fused,
			withBias(ffw[tFFN*tH:], ffb[tFFN:], tFFN, tH, g.kH),
			withBias(ffw[:tFFN*tH], ffb[:tFFN], tFFN, tH, g.kH), tFFN, g.kH)
		put(b.bank, b.gateUp, fused, 2*tFFN, g.kH)
		put(b.bank, b.down, withBias(fow, fob, tH, tFFN, g.kF), tH, g.kF)
		g.wbuf.WriteFloat32At(int(b.n1w), n1w)
		g.wbuf.WriteFloat32At(int(b.n1b), n1b)
		g.wbuf.WriteFloat32At(int(b.n2w), n2w)
		g.wbuf.WriteFloat32At(int(b.n2b), n2b)
	}
	return nil
}

func (g *DiT) pipeline(name string, spirv []byte, spec vk.PipelineSpec) (*vk.ComputePipeline, error) {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return nil, fmt.Errorf("mm3: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	p, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return nil, fmt.Errorf("mm3: pipeline %s: %w", name, err)
	}
	return p, nil
}

func (g *DiT) build(*safetensors.Set) error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	for name, spirv := range map[string][]byte{
		"norm":   shaders.ParakeetLayerNormF16,
		"qkpack": shaders.MM3QKPackHD64,
		"pack":   shaders.H3VAEPackHD64,
		"add":    shaders.DiTGateAddHoist,
		"narrow": shaders.MM3Narrow,
	} {
		p, err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pc})
		if err != nil {
			return err
		}
		g.pipes[name] = p
	}
	ok, err := dit.TransposedAttentionOK(g.dev)
	if err != nil {
		return err
	}
	attn := map[string][]byte{"plain": shaders.H3VAEAttnHD64}
	g.attn = "plain"
	if ok {
		attn["transposed"] = shaders.H3VAEAttnTHD64QT2KT4
		g.attn = "transposed"
	}
	for name, spirv := range attn {
		p, err := g.pipeline("attention "+name, spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pc, RequiredSubgroupSize: 32})
		if err != nil {
			return err
		}
		g.pipes["attn "+name] = p
	}
	for b := range g.banks {
		bb := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}
		m := map[string]*vk.ComputePipeline{}
		for name, k := range map[string]struct {
			spirv []byte
			wave  uint32
		}{
			"big":  {shaders.DiTGEMMWG128x256LDSW32, 32},
			"glu":  {shaders.DiTGEMMWG128x256LDSW32SwiGLU, 32},
			"head": {shaders.DiTGEMMReg32x128Tiled, 0},
		} {
			p, err := g.pipeline("gemm "+name, k.spirv, vk.PipelineSpec{Buffers: bb, PushConstantSize: pc, RequiredSubgroupSize: k.wave})
			if err != nil {
				return err
			}
			m[name] = p
		}
		g.gemms = append(g.gemms, m)
	}
	return nil
}

// Destroy releases every Vulkan object.
func (g *DiT) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.gemms {
		for _, p := range m {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range append([]*vk.Buffer{g.hbuf, g.abuf, g.wbuf}, g.banks...) {
		if b != nil {
			b.Destroy()
		}
	}
}

// TimeEmbedding is the time token: Fourier features of t (2πtw, cos then
// sin), then linear_1, SiLU, linear_2, in fp32 as upstream's CPU run has it.
func (g *DiT) TimeEmbedding(t float32) []float32 {
	f := make([]float32, tFourier)
	for j, w := range g.fourier {
		a := float32(2*math.Pi) * t * w
		f[j] = float32(math.Cos(float64(a)))
		f[tFourier/2+j] = float32(math.Sin(float64(a)))
	}
	h := make([]float32, tH)
	for o := range h {
		s := g.t1b[o]
		for j, v := range f {
			s += g.t1w[o*tFourier+j] * v
		}
		h[o] = s / (1 + float32(math.Exp(float64(-s))))
	}
	out := make([]float32, tH)
	parallelRows(tH, func(o int) {
		s := g.t2b[o]
		row := g.t2w[o*tH : (o+1)*tH]
		for j, v := range h {
			s += row[j] * v
		}
		out[o] = s
	})
	return out
}

// Forward runs the DiT over one window's CFG pair: latents [128][L]
// (channel-major, as upstream holds them), the condition [L][2048] for the
// conditional sequence (the unconditional one's is zeros), at flow time t.
// It returns both predictions, [128][L] each.
func (g *DiT) Forward(latents [][]float32, cond []float32, t float32) (c, u [][]float32, took time.Duration, err error) {
	if len(latents) != LatentChannels {
		return nil, nil, 0, fmt.Errorf("mm3: %d latent channels", len(latents))
	}
	L := len(latents[0])
	if L == 0 || L > g.maxLatents || len(cond) != L*CondChannels {
		return nil, nil, 0, fmt.Errorf("mm3: a window of %d latents (at most %d) and %d condition values", L, g.maxLatents, len(cond))
	}
	// proj_in's A operand: row 1 + i of a sequence is latent i's channels
	// then its condition; the time rows and the pad rows are zero.
	buf := make([]uint16, 2*g.stride*g.ldaIn)
	parallelRows(L, func(i int) {
		for s := 0; s < 2; s++ {
			row := buf[(s*g.stride+1+i)*g.ldaIn:]
			for ch := 0; ch < LatentChannels; ch++ {
				row[ch] = safetensors.F32ToF16(latents[ch][i])
			}
			if s == 0 {
				for j, v := range cond[i*CondChannels : (i+1)*CondChannels] {
					row[LatentChannels+j] = safetensors.F32ToF16(v)
				}
			}
		}
	})
	g.hbuf.WriteUint16At(int(g.hIn), buf)
	g.abuf.WriteFloat32At(int(g.aTime), g.TimeEmbedding(t))

	var total time.Duration
	run := func(d []vk.MultiDispatch) error {
		dur, err := vk.DispatchMultiTimed(d, 1, 1, true)
		total += dur
		return err
	}
	if err := run(g.inGraph(L)); err != nil {
		return nil, nil, total, fmt.Errorf("mm3: DiT input: %w", err)
	}
	for i := 0; i < tLayers; i += tSubmitN {
		var d []vk.MultiDispatch
		for j := i; j < min(i+tSubmitN, tLayers); j++ {
			d = append(d, g.blockGraph(j, L)...)
		}
		if i+tSubmitN >= tLayers {
			d = append(d, g.outGraph()...)
		}
		if err := run(d); err != nil {
			return nil, nil, total, fmt.Errorf("mm3: DiT blocks %d..: %w", i, err)
		}
	}
	rows := g.abuf.ReadFloat32At(int(g.aOut), 2*g.stride*LatentChannels)
	pick := func(s int) [][]float32 {
		out := make([][]float32, LatentChannels)
		for ch := range out {
			out[ch] = make([]float32, L)
			for i := 0; i < L; i++ {
				out[ch][i] = rows[(s*g.stride+1+i)*LatentChannels+ch]
			}
		}
		return out
	}
	return pick(0), pick(1), total, nil
}

func (g *DiT) gemm(name string, bank int, aOff, cOff, bOff uint32, m, n, k, lda int, tweak func(*push)) vk.MultiDispatch {
	bm, bn := tBigBM, tBigBN
	if name == "head" {
		bm, bn = tHeadBM, tHeadBN
	}
	if n%bn != 0 || k%tile != 0 {
		panic(fmt.Sprintf("mm3: gemm %s N=%d K=%d", name, n, k))
	}
	mPad := roundUp(m, bm)
	p := push{InOff: aOff, OutOff: cOff, BOff: bOff, GemmM: uint32(mPad), GemmN: uint32(n), GemmK: uint32(k), LDA: uint32(lda),
		Heads: tHeads, HeadDim: tHD}
	if tweak != nil {
		tweak(&p)
	}
	return vk.MultiDispatch{Pipeline: g.gemms[bank][name], GroupsX: uint32(n / bn), GroupsY: uint32(mPad / bm), PushConstants: p.bytes()}
}

// inGraph is proj_in over both sequences into the residual, then the time
// token added into each sequence's row 0.
func (g *DiT) inGraph(L int) []vk.MultiDispatch {
	rows := 2 * g.stride
	d := []vk.MultiDispatch{g.gemm("big", g.in.bank, g.hIn, g.aX, g.in.off, rows, tH, g.kIn, g.ldaIn, nil)}
	for s := 0; s < 2; s++ {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes["add"], GroupsX: 1, GroupsY: 1,
			PushConstants: push{InOff: g.aTime, OutOff: g.aX + uint32(s*g.stride*tH), Tokens: 1, Dim: tH}.bytes()})
	}
	return d
}

func (g *DiT) blockGraph(i, L int) []vk.MultiDispatch {
	b := &g.blocks[i]
	rows := 2 * g.stride
	tiles := rows / tile
	eps := math.Float32bits(tLNEps)
	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		p.Heads, p.HeadDim = tHeads, tHD
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	norm := func(w, bias uint32) {
		add("norm", uint32(rows), 1, push{InOff: g.aX, OutOff: g.hA, WOff: w, Aux0: bias, Tokens: uint32(rows), Dim: tH,
			LDA: uint32(g.ldaH), Eps: eps})
	}
	norm(b.n1w, b.n1b)
	d = append(d,
		g.gemm("big", b.bank, g.hA, g.aQ, b.q, rows, tH, g.kH, g.ldaH, nil),
		g.gemm("big", b.bank, g.hA, g.aK, b.k, rows, tH, g.kH, g.ldaH, nil),
		g.gemm("big", b.bank, g.hA, g.aV, b.v, rows, tH, g.kH, g.ldaH, nil))
	for _, qk := range []struct {
		src, dst uint32
		scale    float32
	}{{g.aQ, g.hQ, float32(1/math.Sqrt(tHD)) * tLog2e}, {g.aK, g.hK, 1}} {
		add("qkpack", tHeads, uint32((tiles+7)/8), push{InOff: qk.src, OutOff: qk.dst, Tokens: uint32(rows), Dim: tH,
			WOff: g.wCos, Aux0: g.wSin, Aux1: uint32(g.planeRows), Aux2: g.wOnes, Span: uint32(tiles), Eps: eps,
			Scale: math.Float32bits(qk.scale)})
	}
	add("pack", uint32((tiles+7)/8), tHeads, push{InOff: g.aV, OutOff: g.hV, Tokens: uint32(rows), Dim: tH,
		Aux0: 1, Aux1: uint32(g.planeRows), Aux2: uint32(tiles), Scale: math.Float32bits(1)})
	rowsPer := 2 * tile // the transposed build's QT2
	if g.attn == "plain" {
		rowsPer = tile
	}
	for s := 0; s < 2; s++ {
		t0 := uint32(s*g.stride/tile) * uint32(tHD*tile)
		add("attn "+g.attn, uint32((L+1+rowsPer-1)/rowsPer), tHeads, push{Tokens: uint32(L + 1),
			InOff: g.hQ + t0, KOff: g.hK + t0, VOff: g.hV + t0,
			OutOff: g.hCtx + uint32(s*g.stride*g.ldaH), LDA: uint32(g.ldaH), Aux1: uint32(g.planeRows)})
	}
	d = append(d, g.gemm("big", b.bank, g.hCtx, g.aO, b.o, rows, tH, g.kH, g.ldaH, nil))
	add("add", uint32(rows), 1, push{InOff: g.aO, OutOff: g.aX, Tokens: uint32(rows), Dim: tH})
	norm(b.n2w, b.n2b)
	d = append(d,
		g.gemm("glu", b.bank, g.hA, g.hFFN, b.gateUp, rows, 2*tFFN, g.kH, g.ldaH, func(p *push) {
			p.Aux0, p.Scale = uint32(g.ldaF), math.Float32bits(1)
		}),
		g.gemm("big", b.bank, g.hFFN, g.aFF, b.down, rows, tH, g.kF, g.ldaF, nil))
	add("add", uint32(rows), 1, push{InOff: g.aFF, OutOff: g.aX, Tokens: uint32(rows), Dim: tH})
	return d
}

// outGraph narrows the residual into proj_out's A operand and runs the
// folded output projection into aOut, [rows][128].
func (g *DiT) outGraph() []vk.MultiDispatch {
	rows := 2 * g.stride
	return []vk.MultiDispatch{
		{Pipeline: g.pipes["narrow"], GroupsX: uint32(rows), GroupsY: 1,
			PushConstants: push{InOff: g.aX, OutOff: g.hA, Tokens: uint32(rows), Dim: tH, LDA: uint32(g.ldaH)}.bytes()},
		g.gemm("head", g.out.bank, g.hA, g.aOut, g.out.off, rows, LatentChannels, g.kH, g.ldaH, nil),
	}
}
