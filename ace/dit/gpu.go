package dit

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// GPU runs ACE-Step's three transformer stacks on the device: the lyric
// encoder (8 layers), the timbre encoder (4) and the DiT (32), over the
// image DiT's validated kernels (the fragment-tiled fp16 GEMM, the WMMA
// flash attention, the packs, SwiGLU, the gated residual) and H3's modulated
// norm and q/k pack, plus this vertical's three attention builds
// (shaders/ace_*: GQA with the tail out of the row max, the ±128 band, and
// cross-attention).
//
// What is different from h3/dit, and why:
//
//   - **No row chunks and no runs.** A 10-minute song is 7,500 tokens, and
//     every row of one forward shares one timestep, so a stage is one
//     dispatch over the whole sequence and the AdaLN vectors are one set a
//     layer a forward.
//   - **GQA everywhere**: 32 query heads over 8 kv heads in the DiT, 16 over
//     8 in the encoders; k and v are projected and packed at kv width.
//   - **Cross-attention with K/V computed once a request** (Begin). Every
//     layer's condition keys and values are packed into their own planes,
//     ~10 MB a layer, and the step reads them 8 times.
//   - **fp32 residual, fp16 operands, no FFN scale** (A-o1: the residual
//     reaches 5.7e5, every GEMM operand stays under 2.45e3).
//   - **The encoders are this block without modulation**: a = the norm
//     weight, b = 0, and a plain residual add.
type GPU struct {
	// Between, when set, is called between two submissions, when nothing of
	// this model's is on the device (h3/dit's hook, for a server).
	Between func() error

	dev *vk.Device
	cfg *Config
	// big is the kernel every big projection runs on: gemmBig, or what a
	// screen sets (TestGPUStepTiming's ACE_DIT_BIG).
	big  gemmKernel
	Host *Host
	// The GEMM's fp16 epilogues (KERNELS.md G3, research §2.14), both
	// bit-identical to the passes they replace: packV stores the v
	// projections as the attention's fragment tiles (no "pack v" pass);
	// fuseGLU runs gate|up as one GEMM whose epilogue is the SwiGLU, on the
	// LDS-staged build at every row count (its grid is twice a projection's,
	// and with no SwiGLU pass it beats the wave64 build's pair from 375
	// rows, 110 against 120 ms a forward, and ties it at 125). Off, each
	// runs its control: the pack, and the SwiGLU pass over the fused GEMM's
	// fp32 output. TestGPUBits compares them.
	packV, fuseGLU bool

	wbuf, abuf, hbuf *vk.Buffer
	banks            []*vk.Buffer
	pipes            map[string]*vk.ComputePipeline
	gemms            []map[gemmKernel]*vk.ComputePipeline
	mods             []*vk.ShaderModule

	dit, enc  dims
	maxTokens int
	rows      int // plane and scratch rows: every sequence here fits in it
	encRows   int // the cross planes' rows
	maxEnc    int // longest condition sequence

	ditL, lyricL, timbreL []layerW
	detokL, poolL         []layerW
	projIn, projOut       headW
	condEmb, textProj     headW
	lyricIn, timbreIn     headW
	detokIn, detokOut     headW
	tokIn                 headW // the tokenizer's two input projections, folded

	// wbuf (fp32): rope tables, identity tables, the detokenizer's tables
	// (position = row mod 5) and the tokenizer's pooler's (row mod 6), q/k
	// norm weights.
	wCos, wSin, wCosID, wSinID uint32
	wCosG, wSinG               uint32
	wCosP, wSinP               uint32
	// abuf (fp32): residual, scratch, per-step modulation, encoder norms.
	aX, aS, aMod, aTail, aZeros, aOnes uint32
	aProjInB, aLyricB, aTimbreB        uint32
	aDetokNorm, aPoolNorm              uint32
	actElems                           int
	// hbuf (fp16): planes, A operands, the cross planes.
	hQ, hK, hV, hA, hCtx, hFFN, hIn uint32
	hElems                          int

	// Per-request state (Begin).
	encLen int

	// detokChunk, when set, is how many codes Detokenize runs at a time:
	// the test's way to exercise the chunking at a short song.
	detokChunk int
}

type dims struct {
	H, heads, kv, ffn  int
	ldaH, ldaQ, ldaFFN int
}

func (d dims) qW() int  { return d.heads * headDim }
func (d dims) kvW() int { return d.kv * headDim }

type headW struct {
	bank int
	off  uint32
	n, k int
}

type proj int

const (
	pQ proj = iota
	pK
	pV
	pO
	// pGateUp is gate and up as one [2·ffn, H] projection, interleaved in
	// 64-row groups (qwen.InterleaveGLU): dit_gemm.comp's C_SWIGLU layout.
	pGateUp
	pDown
	pCQ // cross-attention: q from the residual, k/v from the condition
	pCK
	pCV
	pCO
)

type layerW struct {
	bank   int
	off    map[proj]uint32
	window bool
	// group > 0 makes self-attention block-diagonal over groups of that
	// many rows, with rope positions restarting every group (the
	// detokenizer's 5-row sequences).
	group int
	// wbuf: q/k norm weights; cross-attention's own pair on the DiT.
	normQ, normK, cnormQ, cnormK uint32
	// abuf: the encoders' input/post-attention norms, the DiT's cross norm.
	norm1, norm2, crossNorm uint32
	// hbuf: the DiT layer's cross-attention key and value planes.
	crossK, crossV uint32
}

// pushConstants mirrors shaders/dit_common.glsl.
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

type gemmKernel int

const (
	gemmBig        gemmKernel = iota // 128x256 tiles
	gemmSmall                        // 64x64, for the 128-wide output head
	gemmBigW64                       // the wave64 128x256 build, for short clips (bigW64Rows)
	gemmBigPack                      // gemmBig storing C as the attention's fragment tiles (G3)
	gemmBigW64Pack                   // the same epilogue on the wave64 build
	gemmBigGLU                       // gemmBig over pGateUp with the SwiGLU epilogue (G3)
)

// bigW64Rows is the row count up to which the big projections run on the
// wave64 build (KERNELS.md G8, 2026-09-29): at 375 rows (a 30 s clip) the
// LDS-staged build is 8% slower a forward (130 against 120 ms; its grid is
// 30 workgroups and the per-slab barrier shows), the two tie at 1500, and
// from 3000 rows the LDS build wins by 2.5-3%. The 64x64 rung is never the
// answer (131 ms at 375, 1.2-1.4x slower above).
const bigW64Rows = 1024

var gemmVariants = map[gemmKernel]struct {
	spirv  []byte
	bm, bn int
	bk     int    // the K tile: 32 for the LDS-staged build, 16 otherwise
	wave   uint32 // a pinned subgroup size, or 0 for the device's default
}{
	// The LDS-staged wave32 build (KERNELS.md G2, research §2.9), bit-identical
	// to the wave64 one it replaced.
	gemmBig:   {shaders.DiTGEMMWG128x256LDSW32, 128, 256, 32, 32},
	gemmSmall: {shaders.DiTGEMMReg64Tiled, 64, 64, 16, 0},
	// The wave64 build gemmBig replaced, which still wins up to bigW64Rows.
	gemmBigW64:     {shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256, 16, 0},
	gemmBigPack:    {shaders.DiTGEMMWG128x256LDSW32CPack, 128, 256, 32, 32},
	gemmBigW64Pack: {shaders.DiTGEMMWG128x256TiledSWZ8CPack, 128, 256, 16, 0},
	gemmBigGLU:     {shaders.DiTGEMMWG128x256LDSW32SwiGLU, 128, 256, 32, 32},
}

const (
	tile     = 16
	headDim  = 128
	gemmPad  = 128
	rowAlign = 128
	log2e    = 1.4426950408889634
	// maxKeyBlock is the key block every attention build reads whole.
	maxKeyBlock = 4 * tile
	// MaxLyricTokens and MaxTextTokens are upstream's truncations.
	MaxLyricTokens = 2048
	MaxTextTokens  = 256
	// TimbreRows is the silence reference's 750 frames plus the CLS row.
	TimbreRows   = 751
	maxBankBytes = 0xfffffffc
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// NewGPU stages the encoders and the DiT for songs of up to maxTokens DiT
// tokens (12.5 a second; 7,500 is upstream's 10-minute ceiling).
func NewGPU(dev *vk.Device, dir string, maxTokens int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	host, err := LoadHost(set, cfg)
	if err != nil {
		return nil, err
	}
	g := &GPU{dev: dev, cfg: cfg, Host: host, maxTokens: maxTokens, pipes: map[string]*vk.ComputePipeline{}, big: gemmBig, packV: true, fuseGLU: true}
	g.dit = dims{H: cfg.Hidden, heads: cfg.Heads, kv: cfg.KVHeads, ffn: cfg.FFN}
	g.enc = dims{H: cfg.EncHidden, heads: cfg.EncHeads, kv: cfg.EncKVHeads, ffn: cfg.EncFFN}
	for _, d := range []*dims{&g.dit, &g.enc} {
		d.ldaH, d.ldaQ, d.ldaFFN = d.H+gemmPad, d.qW()+gemmPad, d.ffn+gemmPad
	}
	g.maxEnc = MaxLyricTokens + 1 + MaxTextTokens
	g.encRows = roundUp(g.maxEnc, rowAlign)
	g.rows = roundUp(max(maxTokens, g.maxEnc, TimbreRows), rowAlign)
	for _, step := range []func(*safetensors.Set) error{g.stage, g.allocActivations, g.build} {
		if err := step(set); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

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

// packB narrows a [n, k] row-major weight into 16x16 fragment tiles,
// kt-fastest: dit_gemm.comp's B layout.
func packB(dst []uint16, w []float32, n, k int) {
	kt := k / tile
	parallel((n+63)/64, func(c int) {
		for i := c * 64; i < min((c+1)*64, n); i++ {
			row := w[i*k : (i+1)*k]
			base := (i / tile) * kt * tile * tile
			lane := (i % tile) * tile
			for j, v := range row {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

func (g *GPU) projShape(d dims, p proj) [2]int {
	switch p {
	case pQ, pCQ:
		return [2]int{d.qW(), d.H}
	case pK, pV, pCK, pCV:
		return [2]int{d.kvW(), d.H}
	case pO, pCO:
		return [2]int{d.H, d.qW()}
	case pGateUp:
		return [2]int{2 * d.ffn, d.H}
	default:
		return [2]int{d.H, d.ffn}
	}
}

var (
	selfProjs  = []proj{pQ, pK, pV, pO, pGateUp, pDown}
	crossProjs = []proj{pCQ, pCK, pCV, pCO}
	projNames  = map[proj]string{
		pQ: "self_attn.q_proj", pK: "self_attn.k_proj", pV: "self_attn.v_proj", pO: "self_attn.o_proj",
		pDown: "mlp.down_proj",
		pCQ:   "cross_attn.q_proj", pCK: "cross_attn.k_proj", pCV: "cross_attn.v_proj", pCO: "cross_attn.o_proj",
	}
)

// stage lays out and fills the fp16 banks and the fp32 weight arena.
func (g *GPU) stage(set *safetensors.Set) error {
	c := g.cfg
	var bankElems []int
	cur := -1
	place := func(n int) (int, uint32) {
		if cur < 0 || (bankElems[cur]+n)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
		off := bankElems[cur]
		bankElems[cur] += n
		return cur, uint32(off)
	}
	head := func(n, k int) headW {
		b, off := place(n * k)
		return headW{bank: b, off: off, n: n, k: k}
	}
	g.projIn = head(g.dit.H, c.PatchIn())
	g.projOut = head(c.PatchOut(), g.dit.H)
	g.condEmb = head(g.dit.H, g.enc.H)
	g.textProj = head(g.enc.H, c.TextDim)
	g.lyricIn = head(g.enc.H, c.TextDim)
	g.timbreIn = head(g.enc.H, c.TimbreDim)
	g.detokIn = head(g.enc.H, g.enc.H)
	g.detokOut = head(c.Latent, g.enc.H)
	g.tokIn = head(g.enc.H, c.Latent)

	ropeHalf := headDim / 2
	g.wCos, g.wSin = 0, uint32(g.rows*ropeHalf)
	g.wCosID, g.wSinID = uint32(2*g.rows*ropeHalf), uint32(3*g.rows*ropeHalf)
	g.wCosG, g.wSinG = uint32(4*g.rows*ropeHalf), uint32(5*g.rows*ropeHalf)
	g.wCosP, g.wSinP = uint32(6*g.rows*ropeHalf), uint32(7*g.rows*ropeHalf)
	w32 := 8 * g.rows * ropeHalf
	newLayer := func(d dims, i int, cross bool) layerW {
		w := layerW{off: map[proj]uint32{}, window: c.Windowed(i)}
		ps := selfProjs
		if cross {
			ps = append(append([]proj(nil), selfProjs...), crossProjs...)
		}
		n := 0
		for _, p := range ps {
			sh := g.projShape(d, p)
			n += sh[0] * sh[1]
		}
		b, off := place(n)
		w.bank = b
		for _, p := range ps {
			sh := g.projShape(d, p)
			w.off[p] = off
			off += uint32(sh[0] * sh[1])
		}
		w.normQ, w.normK = uint32(w32), uint32(w32+headDim)
		w32 += 2 * headDim
		if cross {
			w.cnormQ, w.cnormK = uint32(w32), uint32(w32+headDim)
			w32 += 2 * headDim
		}
		return w
	}
	for i := 0; i < c.LyricLayers; i++ {
		g.lyricL = append(g.lyricL, newLayer(g.enc, i, false))
	}
	for i := 0; i < c.TimbreLayers; i++ {
		g.timbreL = append(g.timbreL, newLayer(g.enc, i, false))
	}
	for i := 0; i < c.Layers; i++ {
		g.ditL = append(g.ditL, newLayer(g.dit, i, true))
	}
	for i := 0; i < c.DetokLayers; i++ {
		w := newLayer(g.enc, i, false)
		w.window, w.group = false, c.PoolWindow
		g.detokL = append(g.detokL, w)
	}
	for i := 0; i < c.DetokLayers; i++ {
		w := newLayer(g.enc, i, false)
		w.window, w.group = false, c.PoolWindow+1
		g.poolL = append(g.poolL, w)
	}

	var err error
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return fmt.Errorf("dit: fp32 weight arena: %w", err)
	}
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("dit: fp16 bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}

	// Rotary tables: Qwen3's inv_freq and angles, formed in fp32 as torch
	// forms them; and the identity pair cross-attention's q/k pack reads.
	cos, sin := make([]float32, g.rows*ropeHalf), make([]float32, g.rows*ropeHalf)
	ones := make([]float32, g.rows*ropeHalf)
	for j := 0; j < ropeHalf; j++ {
		e := float32(2*j) / float32(headDim)
		inv := 1 / float32(math.Pow(c.Theta, float64(e)))
		for p := 0; p < g.rows; p++ {
			a := float32(p) * inv
			cos[p*ropeHalf+j] = float32(math.Cos(float64(a)))
			sin[p*ropeHalf+j] = float32(math.Sin(float64(a)))
			ones[p*ropeHalf+j] = 1
		}
	}
	g.wbuf.WriteFloat32At(int(g.wCos), cos)
	g.wbuf.WriteFloat32At(int(g.wSin), sin)
	g.wbuf.WriteFloat32At(int(g.wCosID), ones)
	g.wbuf.ZeroFloat32At(int(g.wSinID), g.rows*ropeHalf)
	// Position row mod the group: every detokenizer group (5 rows) and
	// every pooler group (a CLS row and 5) starts at 0.
	for _, gt := range []struct {
		n        int
		cos, sin uint32
	}{{c.PoolWindow, g.wCosG, g.wSinG}, {c.PoolWindow + 1, g.wCosP, g.wSinP}} {
		cosG, sinG := make([]float32, g.rows*ropeHalf), make([]float32, g.rows*ropeHalf)
		for r := 0; r < g.rows; r++ {
			p := r % gt.n
			copy(cosG[r*ropeHalf:(r+1)*ropeHalf], cos[p*ropeHalf:(p+1)*ropeHalf])
			copy(sinG[r*ropeHalf:(r+1)*ropeHalf], sin[p*ropeHalf:(p+1)*ropeHalf])
		}
		g.wbuf.WriteFloat32At(int(gt.cos), cosG)
		g.wbuf.WriteFloat32At(int(gt.sin), sinG)
	}

	put := func(bank int, off uint32, w []float32, n, k int) {
		buf := make([]uint16, n*k)
		packB(buf, w, n, k)
		g.banks[bank].WriteUint16At(int(off), buf)
	}
	l := &loader{set: set}
	// proj_in is a Conv1d(192 → 2560, k2, s2): W[o, c, k] becomes
	// W'[o, k·192 + c], so a token's operand row is its two frames of
	// [context | x_t] side by side.
	{
		in, p := c.InChannels, c.Patch
		w := l.f32("decoder.proj_in.1.weight", g.dit.H, in, p)
		if l.err != nil {
			return l.err
		}
		r := make([]float32, len(w))
		for o := 0; o < g.dit.H; o++ {
			for ch := 0; ch < in; ch++ {
				for k := 0; k < p; k++ {
					r[o*p*in+k*in+ch] = w[(o*in+ch)*p+k]
				}
			}
		}
		put(g.projIn.bank, g.projIn.off, r, g.projIn.n, g.projIn.k)
	}
	// proj_out is a ConvTranspose1d(2560 → 64, k2, s2): W[c, o, k] becomes
	// W'[k·64 + o, c], so output column k·64 + o is frame 2i + k's channel o.
	{
		lat, p := c.Latent, c.Patch
		w := l.f32("decoder.proj_out.1.weight", g.dit.H, lat, p)
		if l.err != nil {
			return l.err
		}
		r := make([]float32, len(w))
		for ch := 0; ch < g.dit.H; ch++ {
			for o := 0; o < lat; o++ {
				for k := 0; k < p; k++ {
					r[(k*lat+o)*g.dit.H+ch] = w[(ch*lat+o)*p+k]
				}
			}
		}
		put(g.projOut.bank, g.projOut.off, r, g.projOut.n, g.projOut.k)
	}
	for _, hd := range []struct {
		w    headW
		name string
	}{
		{g.condEmb, "decoder.condition_embedder.weight"},
		{g.textProj, "encoder.text_projector.weight"},
		{g.lyricIn, "encoder.lyric_encoder.embed_tokens.weight"},
		{g.timbreIn, "encoder.timbre_encoder.embed_tokens.weight"},
		{g.detokIn, "detokenizer.embed_tokens.weight"},
		{g.detokOut, "detokenizer.proj_out.weight"},
	} {
		w := l.f32(hd.name, hd.w.n, hd.w.k)
		if l.err != nil {
			return l.err
		}
		put(hd.w.bank, hd.w.off, w, hd.w.n, hd.w.k)
	}
	// The tokenizer's audio_acoustic_proj (64 → 2048) and its pooler's
	// embed_tokens (2048 → 2048) have nothing between them, so they are
	// one [2048, 64] projection, folded in fp64 (the bias on the host).
	{
		H, L := g.enc.H, c.Latent
		wp := l.f32("tokenizer.audio_acoustic_proj.weight", H, L)
		we := l.f32("tokenizer.attention_pooler.embed_tokens.weight", H, H)
		if l.err != nil {
			return l.err
		}
		f := make([]float32, H*L)
		parallel(H, func(o int) {
			var acc [64]float64
			for j := 0; j < H; j++ {
				e := float64(we[o*H+j])
				for i := 0; i < L; i++ {
					acc[i] += e * float64(wp[j*L+i])
				}
			}
			for i := 0; i < L; i++ {
				f[o*L+i] = float32(acc[i])
			}
		})
		put(g.tokIn.bank, g.tokIn.off, f, H, L)
	}

	stageLayer := func(w layerW, d dims, p string, cross bool) error {
		ps := selfProjs
		if cross {
			ps = append(append([]proj(nil), selfProjs...), crossProjs...)
		}
		for _, pr := range ps {
			sh := g.projShape(d, pr)
			var wt []float32
			if pr == pGateUp {
				// SwiGLU is silu(gate)·up: the gate rows first in each group.
				gate := l.f32(p+"mlp.gate_proj.weight", d.ffn, d.H)
				up := l.f32(p+"mlp.up_proj.weight", d.ffn, d.H)
				if l.err != nil {
					return l.err
				}
				wt = make([]float32, 2*d.ffn*d.H)
				qwen.InterleaveGLU(wt, gate, up, d.ffn, d.H)
			} else {
				wt = l.f32(p+projNames[pr]+".weight", sh[0], sh[1])
			}
			if l.err != nil {
				return l.err
			}
			put(w.bank, w.off[pr], wt, sh[0], sh[1])
		}
		g.wbuf.WriteFloat32At(int(w.normQ), l.f32(p+"self_attn.q_norm.weight", headDim))
		g.wbuf.WriteFloat32At(int(w.normK), l.f32(p+"self_attn.k_norm.weight", headDim))
		if cross {
			g.wbuf.WriteFloat32At(int(w.cnormQ), l.f32(p+"cross_attn.q_norm.weight", headDim))
			g.wbuf.WriteFloat32At(int(w.cnormK), l.f32(p+"cross_attn.k_norm.weight", headDim))
		}
		return l.err
	}
	for i, w := range g.lyricL {
		if err := stageLayer(w, g.enc, fmt.Sprintf("encoder.lyric_encoder.layers.%d.", i), false); err != nil {
			return err
		}
	}
	for i, w := range g.timbreL {
		if err := stageLayer(w, g.enc, fmt.Sprintf("encoder.timbre_encoder.layers.%d.", i), false); err != nil {
			return err
		}
	}
	for i, w := range g.detokL {
		if err := stageLayer(w, g.enc, fmt.Sprintf("detokenizer.layers.%d.", i), false); err != nil {
			return err
		}
	}
	for i, w := range g.poolL {
		if err := stageLayer(w, g.enc, fmt.Sprintf("tokenizer.attention_pooler.layers.%d.", i), false); err != nil {
			return err
		}
	}
	for i, w := range g.ditL {
		runtime.GC()
		if err := stageLayer(w, g.dit, fmt.Sprintf("decoder.layers.%d.", i), true); err != nil {
			return err
		}
	}
	return nil
}

// allocActivations places every activation, refuses a layout past the
// storage-buffer range (a buffer past it is clamped silently where it is
// bound), allocates, and fills the fp32 constants.
func (g *GPU) allocActivations(set *safetensors.Set) error {
	c := g.cfg
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += roundUp(n, 64)
		return off
	}
	R := g.rows
	g.aX = alloc(R * max(g.dit.H, g.enc.H))
	g.aS = alloc(R * max(g.dit.qW()+2*g.dit.kvW(), 2*g.dit.H+2*g.dit.ffn))
	g.aMod = alloc(c.Layers * 6 * g.dit.H)
	g.aTail = alloc(2 * g.dit.H)
	g.aZeros = alloc(max(g.dit.H, g.enc.H))
	g.aOnes = alloc(max(g.dit.H, g.enc.H))
	g.aProjInB = alloc(g.dit.H)
	g.aLyricB = alloc(g.enc.H)
	g.aTimbreB = alloc(g.enc.H)
	for i := range g.lyricL {
		g.lyricL[i].norm1, g.lyricL[i].norm2 = alloc(g.enc.H), alloc(g.enc.H)
	}
	for i := range g.timbreL {
		g.timbreL[i].norm1, g.timbreL[i].norm2 = alloc(g.enc.H), alloc(g.enc.H)
	}
	for i := range g.detokL {
		g.detokL[i].norm1, g.detokL[i].norm2 = alloc(g.enc.H), alloc(g.enc.H)
	}
	g.aDetokNorm = alloc(g.enc.H)
	for i := range g.poolL {
		g.poolL[i].norm1, g.poolL[i].norm2 = alloc(g.enc.H), alloc(g.enc.H)
	}
	for i := range g.ditL {
		g.ditL[i].crossNorm = alloc(g.dit.H)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += roundUp(n, 64)
		return off
	}
	g.hQ = halloc(max(g.dit.heads, g.enc.heads) * R * headDim)
	g.hK = halloc(max(g.dit.kv, g.enc.kv) * R * headDim)
	g.hV = halloc(max(g.dit.kv, g.enc.kv) * R * headDim)
	g.hA = halloc(R * max(g.dit.ldaH, g.enc.ldaH))
	g.hCtx = halloc(R * max(g.dit.ldaQ, g.enc.ldaQ))
	g.hFFN = halloc(R * max(g.dit.ldaFFN, g.enc.ldaFFN))
	g.hIn = halloc(R * (max(c.PatchIn(), c.TextDim, c.TimbreDim, g.enc.H, g.dit.H) + gemmPad))
	for i := range g.ditL {
		g.ditL[i].crossK = halloc(g.dit.kv * g.encRows * headDim)
		g.ditL[i].crossV = halloc(g.dit.kv * g.encRows * headDim)
	}
	for _, a := range []struct {
		name  string
		bytes int
	}{{"fp32 activation", g.actElems * 4}, {"fp16 activation", g.hElems * 2}} {
		if a.bytes > maxBankBytes {
			return fmt.Errorf("dit: the %s arena for %d tokens is %d MB, past the %d MB storage-buffer range",
				a.name, g.maxTokens, a.bytes>>20, maxBankBytes>>20)
		}
	}
	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("dit: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("dit: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	g.hbuf.ZeroUint16At(0, g.hElems)
	g.abuf.ZeroFloat32At(0, g.actElems)
	ones := make([]float32, max(g.dit.H, g.enc.H))
	for i := range ones {
		ones[i] = 1
	}
	g.abuf.WriteFloat32At(int(g.aOnes), ones)

	l := &loader{set: set}
	g.abuf.WriteFloat32At(int(g.aProjInB), l.f32("decoder.proj_in.1.bias", g.dit.H))
	g.abuf.WriteFloat32At(int(g.aLyricB), l.f32("encoder.lyric_encoder.embed_tokens.bias", g.enc.H))
	g.abuf.WriteFloat32At(int(g.aTimbreB), l.f32("encoder.timbre_encoder.embed_tokens.bias", g.enc.H))
	for _, st := range []struct {
		ls []layerW
		p  string
	}{{g.lyricL, "encoder.lyric_encoder.layers.%d."}, {g.timbreL, "encoder.timbre_encoder.layers.%d."},
		{g.detokL, "detokenizer.layers.%d."}, {g.poolL, "tokenizer.attention_pooler.layers.%d."}} {
		for i, w := range st.ls {
			p := fmt.Sprintf(st.p, i)
			g.abuf.WriteFloat32At(int(w.norm1), l.f32(p+"input_layernorm.weight", g.enc.H))
			g.abuf.WriteFloat32At(int(w.norm2), l.f32(p+"post_attention_layernorm.weight", g.enc.H))
		}
	}
	g.abuf.WriteFloat32At(int(g.aDetokNorm), l.f32("detokenizer.norm.weight", g.enc.H))
	for i, w := range g.ditL {
		g.abuf.WriteFloat32At(int(w.crossNorm), l.f32(fmt.Sprintf("decoder.layers.%d.cross_attn_norm.weight", i), g.dit.H))
	}
	return l.err
}

func (g *GPU) build(*safetensors.Set) error {
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
		"norm":   shaders.H3NormMod,
		"qkpack": shaders.ACEQKPack,
		"bias":   shaders.H3BiasCopy,
		"pack":   shaders.DiTPackF16TPW8,
		"gate":   shaders.DiTGateAddHoist,
		// The SwiGLU pass over the fused gate|up GEMM's interleaved fp32
		// output (the epilogue's control), and the
		// zeroing of the A pad rows ahead of a GEMM that stores v's tiles.
		"swiglu": shaders.DiTSwiGLUF16IL32,
		"zero":   shaders.ZeroF16,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("dit: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return err
	}
	if !g.dev.Features().SubgroupSizeControl || !sgs.Supported || 32 < sgs.MinSubgroupSize || 32 > sgs.MaxSubgroupSize {
		return fmt.Errorf("dit: the attention builds need a pinned wave32")
	}
	for name, spirv := range map[string][]byte{
		"attn full":   shaders.ACEAttnFull,
		"attn window": shaders.ACEAttnWindow,
		"attn cross":  shaders.ACEAttnCross,
		"attn group":  shaders.ACEAttnGroup,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize, RequiredSubgroupSize: 32})
		if err != nil {
			return fmt.Errorf("dit: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	for b := range g.banks {
		m := map[gemmKernel]*vk.ComputePipeline{}
		for k, v := range gemmVariants {
			p, err := newPipe(v.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}, PushConstantSize: pcSize, RequiredSubgroupSize: v.wave})
			if err != nil {
				return fmt.Errorf("dit: gemm bank %d: %w", b, err)
			}
			m[k] = p
		}
		g.gemms = append(g.gemms, m)
	}
	return nil
}

// Destroy releases every Vulkan object.
func (g *GPU) Destroy() {
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

// WeightBytes and ActivationBytes are what the model holds on the device.
func (g *GPU) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

func (g *GPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// MaxTokens is the longest DiT sequence this staging takes.
func (g *GPU) MaxTokens() int { return g.maxTokens }

// ---- the graph ------------------------------------------------------------

type graph struct {
	g     *GPU
	d     []vk.MultiDispatch
	kinds []string
	flops []float64
	// bytes is what a streaming dispatch reads and writes (KERNELS.md G6):
	// its rate against the 236 GB/s a copy gets is its TFLOP/s.
	bytes []float64
}

func (g *GPU) newGraph() *graph { return &graph{g: g} }

// moved records the bytes the dispatch just added moves.
func (gr *graph) moved(b int) { gr.bytes[len(gr.bytes)-1] = float64(b) }

func (gr *graph) add(pipe, kind string, gx, gy uint32, pc pushConstants, flops float64) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, flops)
	gr.bytes = append(gr.bytes, 0)
}

// gemm issues C[mPad, n] fp32 at cOff (row stride n) = A[m, k] fp16 at aOff
// (row stride lda) × B, the fragment-tiled weight at bOff of bank.
func (gr *graph) gemm(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, m, n, k, lda int) error {
	return gr.gemmWith(kernel, bank, kind, aOff, cOff, bOff, m, n, k, lda, nil)
}

// gemmWith is gemm with the push constants an epilogue build reads beyond
// the GEMM's own (C_PACK's head width and plane rows, C_SWIGLU's output
// stride and scale) set by tweak.
func (gr *graph) gemmWith(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, m, n, k, lda int, tweak func(*pushConstants)) error {
	if m <= bigW64Rows {
		switch kernel {
		case gemmBig:
			kernel = gemmBigW64
		case gemmBigPack:
			kernel = gemmBigW64Pack
		}
	}
	v := gemmVariants[kernel]
	if n%v.bn != 0 || k%v.bk != 0 {
		return fmt.Errorf("dit: %s: N=%d K=%d do not tile by %d/%d", kind, n, k, v.bn, v.bk)
	}
	mPad := roundUp(m, v.bm)
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(mPad), uint32(n), uint32(k), uint32(lda)
	if tweak != nil {
		tweak(&pc)
	}
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.gemms[bank][kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(mPad / v.bm), PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, 2*float64(mPad)*float64(n)*float64(k))
	gr.bytes = append(gr.bytes, 0)
	return nil
}

// A submission closes before its estimated device time passes the budget
// (amdgpu's 2 s ring watchdog; h3/dit's pricing).
const (
	submitBudget = 800 * time.Millisecond
	submitRate   = 12e12
	submitFlat   = 2 * time.Millisecond
)

func (gr *graph) submit() (time.Duration, error) {
	var total time.Duration
	for i := 0; i < len(gr.d); {
		j, est := i, time.Duration(0)
		for j < len(gr.d) {
			c := submitFlat + time.Duration(gr.flops[j]/submitRate*1e9)
			if j > i && est+c > submitBudget {
				break
			}
			est += c
			j++
		}
		if i > 0 && gr.g.Between != nil {
			if err := gr.g.Between(); err != nil {
				return total, err
			}
		}
		t, err := vk.DispatchMultiTimed(gr.d[i:j], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("dit: dispatch %d-%d (%s): %w", i, j-1, gr.kinds[i], err)
		}
		total += t
		i = j
	}
	gr.d, gr.kinds, gr.flops, gr.bytes = gr.d[:0], gr.kinds[:0], gr.flops[:0], gr.bytes[:0]
	return total, nil
}

// norm records the modulated RMS pre-norm over rows [0, n) of the residual
// into hA: rms(x)·a + b.
func (gr *graph) norm(d dims, kind string, n int, a, b uint32) {
	g := gr.g
	var pc pushConstants
	pc.Tokens, pc.Dim, pc.LDA = uint32(n), uint32(d.H), uint32(d.ldaH)
	pc.InOff, pc.OutOff = g.aX, g.hA
	pc.Aux0, pc.Aux1 = a, b
	pc.Eps = math.Float32bits(float32(g.cfg.Eps))
	gr.add("norm", kind, uint32(n), 1, pc, 0)
	gr.moved(n * d.H * (4 + 2))
}

// residual adds y (row stride d.H) into the residual, times the vector at
// gate unless gate is noGate.
func (gr *graph) residual(d dims, kind string, n int, y, gate uint32) {
	var pc pushConstants
	pc.Tokens, pc.Dim = uint32(n), uint32(d.H)
	pc.InOff, pc.OutOff = y, gr.g.aX
	if gate != noGate {
		pc.Aux0, pc.Aux2 = gate, 1
	}
	gr.add("gate", kind, uint32(n), 1, pc, 0)
	gr.moved(n * d.H * (4 + 4 + 4))
}

const noGate = math.MaxUint32

// qkpack records the per-head norm, the rotation and the scale of heads
// heads of width 128 at src (row stride heads·128) into the planes at dst
// (planeRows rows a plane).
func (gr *graph) qkpack(kind string, n, heads int, src, dst uint32, planeRows int, cos, sin, norm uint32, scale float32) {
	tiles := roundUp(n, tile) / tile
	var pc pushConstants
	pc.Tokens, pc.Dim = uint32(n), uint32(heads*headDim)
	pc.InOff, pc.OutOff = src, dst
	pc.WOff, pc.Aux0 = cos, sin
	pc.Aux1, pc.Aux2 = uint32(planeRows), norm
	pc.Span, pc.Eps, pc.Scale = uint32(tiles), math.Float32bits(float32(gr.g.cfg.Eps)), math.Float32bits(scale)
	// A head across, a token tile down (the HEAD_X build, KERNELS.md G6): a
	// q row is 32 heads of 512 B, four turns of the DRAM channel rotation,
	// so one head of many tokens is an eighth of the channels.
	gr.add("qkpack", kind, uint32(heads), uint32((tiles+7)/8), pc, 0)
	gr.moved(n * heads * headDim * (4 + 2))
}

// projV records a v projection of n rows of the fp16 operand at aOff into
// the planes at dst (planeRows rows a plane) as the attention's transposed
// fragment tiles: the GEMM's own store (C_PACK) when packV is set and the
// big kernel has that build, else the GEMM into the fp32 scratch at tmp and
// the pack. The C_PACK GEMM's pad rows land in the plane as pad keys, which
// the pack leaves +0 (zeroKeyTail), so the A rows they are the products of
// are zeroed first.
func (gr *graph) projV(bank int, kind string, aOff, tmp, bOff, dst uint32, n, heads, k, lda, planeRows int) error {
	g := gr.g
	kernel, ok := map[gemmKernel]gemmKernel{gemmBig: gemmBigPack, gemmBigW64: gemmBigW64Pack}[g.big]
	if !g.packV || !ok {
		if err := gr.gemm(g.big, bank, "gemm "+kind, aOff, tmp, bOff, n, heads*headDim, k, lda); err != nil {
			return err
		}
		gr.vpack("pack "+kind, n, heads, tmp, dst, planeRows)
		return nil
	}
	if mPad := roundUp(n, gemmVariants[kernel].bm); mPad > n {
		var pc pushConstants
		pc.Tokens, pc.Dim, pc.LDA = uint32(mPad-n), uint32(lda), uint32(lda)
		pc.OutOff = aOff + uint32(n*lda)
		gr.add("zero", "zero pad", uint32(mPad-n), 1, pc, 0)
	}
	return gr.gemmWith(kernel, bank, "gemm "+kind, aOff, dst, bOff, n, heads*headDim, k, lda,
		func(pc *pushConstants) { pc.HeadDim, pc.Aux1 = headDim, uint32(planeRows) })
}

// vpack records v's narrowing into transposed fragment tiles.
func (gr *graph) vpack(kind string, n, heads int, src, dst uint32, planeRows int) {
	tiles := roundUp(n, tile) / tile
	var pc pushConstants
	pc.Tokens, pc.Dim = uint32(n), uint32(heads*headDim)
	pc.InOff, pc.OutOff = src, dst
	pc.Aux0, pc.Aux1, pc.Aux2 = 1, uint32(planeRows), uint32(tiles)
	pc.Scale = math.Float32bits(1)
	gr.add("pack", kind, uint32((tiles+7)/8), uint32(heads), pc, 0)
	gr.moved(n * heads * headDim * (4 + 2))
}

var qScale = float32(1/math.Sqrt(headDim)) * log2e

// modVecs is where one layer's six vectors live in the fp32 arena: a1, b1,
// g1 before attention, a2, b2, g2 before the MLP. The encoders' gates are
// noGate.
type modVecs [6]uint32

// layer records one transformer layer over rows [0, n) of the residual:
// self-attention (windowed on the even layers), cross-attention onto the
// request's condition planes when cross is set (the DiT), and the MLP.
func (gr *graph) layer(d dims, w layerW, n int, v modVecs, cross bool) error {
	g := gr.g
	R := g.rows
	aQ := g.aS
	aK := aQ + uint32(R*d.qW())
	aV := aK + uint32(R*d.kvW())
	aAttn := g.aS
	aGate := aAttn + uint32(R*d.H) // gate|up, fused: [R, 2·ffn]
	aFF := aGate + uint32(2*R*d.ffn)
	nf := float64(n)

	cos, sin := g.wCos, g.wSin
	switch w.group {
	case 0:
	case g.cfg.PoolWindow:
		cos, sin = g.wCosG, g.wSinG
	default:
		cos, sin = g.wCosP, g.wSinP
	}
	gr.norm(d, "attn in", n, v[0], v[1])
	for _, p := range []struct {
		p   proj
		out uint32
	}{{pQ, aQ}, {pK, aK}} {
		sh := g.projShape(d, p.p)
		if err := gr.gemm(g.big, w.bank, "gemm qkv", g.hA, p.out, w.off[p.p], n, sh[0], sh[1], d.ldaH); err != nil {
			return err
		}
	}
	if err := gr.projV(w.bank, "v", g.hA, aV, w.off[pV], g.hV, n, d.kv, d.H, d.ldaH, R); err != nil {
		return err
	}
	gr.qkpack("qkpack q", n, d.heads, aQ, g.hQ, R, cos, sin, w.normQ, qScale)
	gr.qkpack("qkpack k", n, d.kv, aK, g.hK, R, cos, sin, w.normK, 1)
	{
		var pc pushConstants
		pc.Tokens, pc.Heads = uint32(n), uint32(d.heads)
		pc.InOff, pc.KOff, pc.VOff = g.hQ, g.hK, g.hV
		pc.OutOff, pc.LDA = g.hCtx, uint32(d.ldaQ)
		pc.Aux1, pc.Aux2 = uint32(R), uint32(d.heads/d.kv)
		pipe, keys := "attn full", nf
		if w.window {
			pipe, pc.Span = "attn window", uint32(g.cfg.Window)
			keys = math.Min(nf, float64(2*g.cfg.Window+1))
		}
		if w.group > 0 {
			pipe, pc.Span = "attn group", uint32(w.group)
			keys = float64(w.group)
		}
		gr.add(pipe, "attention", uint32((n+tile-1)/tile), uint32(d.heads), pc, 4*nf*keys*headDim*float64(d.heads))
	}
	if err := gr.gemm(g.big, w.bank, "gemm o", g.hCtx, aAttn, w.off[pO], n, d.H, d.qW(), d.ldaQ); err != nil {
		return err
	}
	gr.residual(d, "gate attn", n, aAttn, v[2])

	if cross {
		gr.norm(d, "cross in", n, w.crossNorm, g.aZeros)
		if err := gr.gemm(g.big, w.bank, "gemm cross q", g.hA, aQ, w.off[pCQ], n, d.qW(), d.H, d.ldaH); err != nil {
			return err
		}
		gr.qkpack("qkpack cross q", n, d.heads, aQ, g.hQ, R, g.wCosID, g.wSinID, w.cnormQ, qScale)
		var pc pushConstants
		pc.Tokens, pc.KStride, pc.Heads = uint32(g.encLen), uint32(n), uint32(d.heads)
		pc.InOff, pc.KOff, pc.VOff = g.hQ, w.crossK, w.crossV
		pc.OutOff, pc.LDA = g.hCtx, uint32(d.ldaQ)
		pc.Aux1, pc.Aux2, pc.LDB = uint32(R), uint32(d.heads/d.kv), uint32(g.encRows)
		gr.add("attn cross", "cross attention", uint32((n+tile-1)/tile), uint32(d.heads), pc,
			4*nf*float64(g.encLen)*headDim*float64(d.heads))
		if err := gr.gemm(g.big, w.bank, "gemm cross o", g.hCtx, aAttn, w.off[pCO], n, d.H, d.qW(), d.ldaQ); err != nil {
			return err
		}
		gr.residual(d, "cross add", n, aAttn, noGate)
	}

	gr.norm(d, "ffn in", n, v[3], v[4])
	if g.fuseGLU && g.big == gemmBig {
		// gate|up as one GEMM over the interleaved weight, its epilogue the
		// SwiGLU into the down projection's fp16 A operand.
		if err := gr.gemmWith(gemmBigGLU, w.bank, "gemm gateup", g.hA, g.hFFN, w.off[pGateUp], n, 2*d.ffn, d.H, d.ldaH,
			func(pc *pushConstants) { pc.Aux0, pc.Scale = uint32(d.ldaFFN), math.Float32bits(1) }); err != nil {
			return err
		}
	} else {
		// The control, and the screens' other kernels: the same GEMM stored as
		// fp32 [n, 2·ffn], then the SwiGLU pass over its interleaved columns
		// -- the same values through the same expression.
		if err := gr.gemm(g.big, w.bank, "gemm gateup", g.hA, aGate, w.off[pGateUp], n, 2*d.ffn, d.H, d.ldaH); err != nil {
			return err
		}
		var pc pushConstants
		pc.Tokens, pc.Dim, pc.LDA = uint32(n), uint32(d.ffn), uint32(d.ldaFFN)
		pc.InOff, pc.OutOff = aGate, g.hFFN
		pc.Scale = math.Float32bits(1)
		gr.add("swiglu", "swiglu", uint32(n), 1, pc, 0)
		gr.moved(n * d.ffn * (8 + 2))
	}
	if err := gr.gemm(g.big, w.bank, "gemm down", g.hFFN, aFF, w.off[pDown], n, d.H, d.ffn, d.ldaFFN); err != nil {
		return err
	}
	gr.residual(d, "gate ffn", n, aFF, v[5])
	return nil
}

// zeroKeyTail zeroes k and v plane rows between the last tile a pack of
// keys rows writes and the end of the last key block the attention reads,
// in every plane at base (heads of them, planeRows rows each): a stale key
// from a longer sequence is masked, but a masked fp16 product still carries
// its sign into the sum (WMMA's -0), so the tail is made the same every time.
func (g *GPU) zeroKeyTail(bases []uint32, heads, planeRows, keys int) {
	from, to := roundUp(keys, tile), min(roundUp(keys, maxKeyBlock), planeRows)
	if from >= to {
		return
	}
	for _, b := range bases {
		for h := 0; h < heads; h++ {
			g.hbuf.ZeroUint16At(int(b)+(h*planeRows+from)*headDim, (to-from)*headDim)
		}
	}
}

// narrowRows writes an fp32 host matrix as an fp16 A operand with row
// stride lda at hOff.
func (g *GPU) narrowRows(hOff uint32, m *qwen.Mat, lda int) {
	buf := make([]uint16, m.Rows*lda)
	parallel(m.Rows, func(r int) {
		for c, v := range m.Row(r) {
			buf[r*lda+c] = safetensors.F32ToF16(v)
		}
	})
	g.hbuf.WriteUint16At(int(hOff), buf)
}

func (g *GPU) readX(rows, cols int) *qwen.Mat {
	m := qwen.NewMat(rows, cols)
	copy(m.Data, g.abuf.ReadFloat32At(int(g.aX), rows*cols))
	return m
}

// ---- the condition encoder (A2) --------------------------------------------

// Stack names one of the three stacks, for the per-layer instruments.
type Stack int

const (
	Lyric Stack = iota
	Timbre
	DiT
)

func (g *GPU) stack(s Stack) ([]layerW, dims) {
	switch s {
	case Lyric:
		return g.lyricL, g.enc
	case Timbre:
		return g.timbreL, g.enc
	default:
		return g.ditL, g.dit
	}
}

// encoderLayers records layers [from, to) of an encoder over n rows.
func (gr *graph) encoderLayers(ls []layerW, n, from, to int) error {
	g := gr.g
	g.zeroKeyTail([]uint32{g.hK, g.hV}, g.enc.kv, g.rows, n)
	for i := from; i < to; i++ {
		w := ls[i]
		v := modVecs{w.norm1, g.aZeros, noGate, w.norm2, g.aZeros, noGate}
		if err := gr.layer(g.enc, w, n, v, false); err != nil {
			return err
		}
	}
	return nil
}

// EncoderLayers runs layers [from, to) of the lyric or timbre encoder over
// x, their residual stream, and returns it after them: the teacher-forced
// instrument the A2 gates run layer by layer.
func (g *GPU) EncoderLayers(s Stack, x *qwen.Mat, from, to int) (*qwen.Mat, error) {
	ls, d := g.stack(s)
	if s == DiT || x.Cols != d.H || x.Rows > g.rows {
		return nil, fmt.Errorf("dit: encoder residual %v", x)
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	gr := g.newGraph()
	if err := gr.encoderLayers(ls, x.Rows, from, to); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	return g.readX(x.Rows, d.H), nil
}

// embedInto records an input projection of an fp16 operand (m rows at hIn,
// row stride k+pad) into the residual from row dst, plus its bias.
func (gr *graph) embedInto(hw headW, m, dst int, bias uint32) error {
	g := gr.g
	out := g.aX + uint32(dst*hw.n)
	if err := gr.gemm(g.big, hw.bank, "gemm embed", g.hIn, out, hw.off, m, hw.n, hw.k, hw.k+gemmPad); err != nil {
		return err
	}
	var pc pushConstants
	pc.Tokens, pc.Dim, pc.LDA = uint32(m), uint32(hw.n), uint32(hw.n)
	pc.InOff, pc.OutOff = out, out
	pc.Aux0, pc.Aux2 = bias, 1
	gr.add("bias", "bias", uint32(m), 1, pc, 0)
	gr.moved(m * hw.n * (4 + 4))
	return nil
}

// Encode runs the condition encoder: text is Qwen3-Embedding's last hidden
// state over the caption prompt [Lt, 1024], lyric the embedding table's rows
// for the lyric prompt [Ll, 1024], timbre the reference latents [750, 64]
// (silence_latent[:750] when there is none). It returns the packed
// cross-attention sequence [Ll + 1 + Lt, 2048]: lyric encoder, timbre CLS,
// projected text. The final norms run on the host in fp32.
func (g *GPU) Encode(text, lyric, timbre *qwen.Mat) (*qwen.Mat, error) {
	c := g.cfg
	if text.Cols != c.TextDim || lyric.Cols != c.TextDim || timbre.Cols != c.TimbreDim ||
		text.Rows > MaxTextTokens || lyric.Rows > MaxLyricTokens || timbre.Rows+1 > g.rows || lyric.Rows == 0 || text.Rows == 0 {
		return nil, fmt.Errorf("dit: encoder inputs text %v lyric %v timbre %v", text, lyric, timbre)
	}
	H := g.enc.H
	// The caption: text_projector alone, into the scratch.
	g.narrowRows(g.hIn, text, c.TextDim+gemmPad)
	gr := g.newGraph()
	if err := gr.gemm(g.big, g.textProj.bank, "gemm text", g.hIn, g.aS, g.textProj.off, text.Rows, H, c.TextDim, c.TextDim+gemmPad); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	txt := g.abuf.ReadFloat32At(int(g.aS), text.Rows*H)

	// The lyrics: embed_tokens, 8 layers, the final norm.
	g.narrowRows(g.hIn, lyric, c.TextDim+gemmPad)
	if err := gr.embedInto(g.lyricIn, lyric.Rows, 0, g.aLyricB); err != nil {
		return nil, err
	}
	if err := gr.encoderLayers(g.lyricL, lyric.Rows, 0, len(g.lyricL)); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	lyr, err := g.Host.LyricNorm.Apply(g.readX(lyric.Rows, H))
	if err != nil {
		return nil, err
	}

	// The timbre: CLS then embed_tokens(latents), 4 layers, the CLS row.
	n := timbre.Rows + 1
	g.narrowRows(g.hIn, timbre, c.TimbreDim+gemmPad)
	g.abuf.WriteFloat32At(int(g.aX), g.Host.TimbreCLS)
	if err := gr.embedInto(g.timbreIn, timbre.Rows, 1, g.aTimbreB); err != nil {
		return nil, err
	}
	if err := gr.encoderLayers(g.timbreL, n, 0, len(g.timbreL)); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	cls, err := g.Host.TimbreNorm.Apply(g.readX(1, H))
	if err != nil {
		return nil, err
	}

	out := qwen.NewMat(lyric.Rows+1+text.Rows, H)
	copy(out.Data, lyr.Data)
	copy(out.Row(lyric.Rows), cls.Data)
	copy(out.Data[(lyric.Rows+1)*H:], txt)
	return out, nil
}

// ---- the DiT (A3) ------------------------------------------------------------

// Begin prepares a request's cross-attention: enc, the packed condition
// sequence [L, 2048], through condition_embedder, then every layer's
// cross-attention keys (k_proj, k_norm, no rotation) and values into that
// layer's planes.
func (g *GPU) Begin(enc *qwen.Mat) error {
	if enc.Cols != g.enc.H || enc.Rows > g.maxEnc || enc.Rows == 0 {
		return fmt.Errorf("dit: condition sequence %v, staged for %d rows", enc, g.maxEnc)
	}
	L, H := enc.Rows, g.dit.H
	g.narrowRows(g.hIn, enc, g.enc.H+gemmPad)
	gr := g.newGraph()
	if err := gr.gemm(g.big, g.condEmb.bank, "gemm condition", g.hIn, g.aS, g.condEmb.off, L, H, g.enc.H, g.enc.H+gemmPad); err != nil {
		return err
	}
	if _, err := gr.submit(); err != nil {
		return err
	}
	ce := qwen.NewMat(L, H)
	copy(ce.Data, g.abuf.ReadFloat32At(int(g.aS), L*H))
	for r := 0; r < L; r++ {
		row := ce.Row(r)
		for i := range row {
			row[i] += g.Host.CondB[i]
		}
	}
	g.narrowRows(g.hIn, ce, H+gemmPad)
	aK := g.aS
	aV := aK + uint32(g.encRows*g.dit.kvW())
	kv := g.dit.kvW()
	for _, w := range g.ditL {
		g.zeroKeyTail([]uint32{w.crossK, w.crossV}, g.dit.kv, g.encRows, L)
		if err := gr.gemm(g.big, w.bank, "gemm cross k", g.hIn, aK, w.off[pCK], L, kv, H, H+gemmPad); err != nil {
			return err
		}
		if err := gr.projV(w.bank, "cross v", g.hIn, aV, w.off[pCV], w.crossV, L, g.dit.kv, H, H+gemmPad, g.encRows); err != nil {
			return err
		}
		gr.qkpack("qkpack cross k", L, g.dit.kv, aK, w.crossK, g.encRows, g.wCosID, g.wSinID, w.cnormK, 1)
	}
	if _, err := gr.submit(); err != nil {
		return err
	}
	g.encLen = L
	return nil
}

// uploadStep writes forward t's vectors: per layer a1 = self_attn_norm ·
// (1 + scale_msa), b1 = shift_msa, g1 = gate_msa, and the MLP's three the
// same way; and the output norm's a = norm_out · (1 + scale), b = shift.
func (g *GPU) uploadStep(t float32) error {
	temb, proj, err := g.Host.Timestep(t)
	if err != nil {
		return err
	}
	H := g.dit.H
	buf := make([]float32, len(g.ditL)*6*H)
	for l := range g.ditL {
		tab, n := g.Host.Tables[l], g.Host.Norms[l]
		o := buf[l*6*H : (l+1)*6*H]
		for i := 0; i < H; i++ {
			v := func(j int) float32 { return tab[j*H+i] + proj[j*H+i] }
			o[i] = n[0][i] * (1 + v(1))
			o[H+i] = v(0)
			o[2*H+i] = v(2)
			o[3*H+i] = n[1][i] * (1 + v(4))
			o[4*H+i] = v(3)
			o[5*H+i] = v(5)
		}
	}
	g.abuf.WriteFloat32At(int(g.aMod), buf)
	tail := make([]float32, 2*H)
	for i := 0; i < H; i++ {
		shift := g.Host.OutTable[i] + temb[i]
		scale := g.Host.OutTable[H+i] + temb[i]
		tail[i] = g.Host.NormOut[i] * (1 + scale)
		tail[H+i] = shift
	}
	g.abuf.WriteFloat32At(int(g.aTail), tail)
	return nil
}

func (g *GPU) vecs(l int) modVecs {
	var v modVecs
	o := g.aMod + uint32(l*6*g.dit.H)
	for j := range v {
		v[j] = o + uint32(j*g.dit.H)
	}
	return v
}

// Tokens is the DiT sequence length for T latent frames.
func (g *GPU) Tokens(frames int) int { return (frames + g.cfg.Patch - 1) / g.cfg.Patch }

// Step runs one forward: x is x_t [T, 64], ctx the context latents
// [T, 128] (source latents | chunk mask), t the timestep. It returns the
// velocity [T, 64] and the device time.
func (g *GPU) Step(x, ctx *qwen.Mat, t float32) (*qwen.Mat, time.Duration, error) {
	gr, n, err := g.stepGraph(x, ctx, t)
	if err != nil {
		return nil, 0, err
	}
	took, err := gr.submit()
	if err != nil {
		return nil, 0, err
	}
	return g.readVelocity(n, x.Rows), took, nil
}

func (g *GPU) stepGraph(x, ctx *qwen.Mat, t float32) (*graph, int, error) {
	c := g.cfg
	T := x.Rows
	n := g.Tokens(T)
	if g.encLen == 0 {
		return nil, 0, fmt.Errorf("dit: Step before Begin")
	}
	if x.Cols != c.Latent || ctx.Rows != T || ctx.Cols+x.Cols != c.InChannels || n > g.maxTokens {
		return nil, 0, fmt.Errorf("dit: x %v ctx %v for %d tokens", x, ctx, g.maxTokens)
	}
	if err := g.uploadStep(t); err != nil {
		return nil, 0, err
	}
	// The patch operand: token i is frames 2i and 2i+1 of [ctx | x], the
	// odd tail frame zero (upstream's F.pad).
	in := c.InChannels
	lda := c.PatchIn() + gemmPad
	buf := make([]uint16, n*lda)
	parallel(n, func(i int) {
		for k := 0; k < c.Patch; k++ {
			f := i*c.Patch + k
			if f >= T {
				break
			}
			row := buf[i*lda+k*in:]
			for j, v := range ctx.Row(f) {
				row[j] = safetensors.F32ToF16(v)
			}
			for j, v := range x.Row(f) {
				row[ctx.Cols+j] = safetensors.F32ToF16(v)
			}
		}
	})
	g.hbuf.WriteUint16At(int(g.hIn), buf)
	gr := g.newGraph()
	if err := gr.embedInto(g.projIn, n, 0, g.aProjInB); err != nil {
		return nil, 0, err
	}
	g.zeroKeyTail([]uint32{g.hK, g.hV}, g.dit.kv, g.rows, n)
	for l, w := range g.ditL {
		if err := gr.layer(g.dit, w, n, g.vecs(l), true); err != nil {
			return nil, 0, err
		}
	}
	gr.norm(g.dit, "norm out", n, g.aTail, g.aTail+uint32(g.dit.H))
	if err := gr.gemm(gemmSmall, g.projOut.bank, "gemm proj out", g.hA, g.aS, g.projOut.off, n, g.projOut.n, g.dit.H, g.dit.ldaH); err != nil {
		return nil, 0, err
	}
	return gr, n, nil
}

// readVelocity unpatches the head's [n, 2·64] output to [T, 64] with the
// bias.
func (g *GPU) readVelocity(n, T int) *qwen.Mat {
	lat := g.cfg.Latent
	raw := g.abuf.ReadFloat32At(int(g.aS), n*g.projOut.n)
	v := qwen.NewMat(T, lat)
	for f := 0; f < T; f++ {
		i, k := f/g.cfg.Patch, f%g.cfg.Patch
		row := v.Row(f)
		for o := range row {
			row[o] = raw[i*g.projOut.n+k*lat+o] + g.Host.ProjOutB[o]
		}
	}
	return v
}

// Layers runs DiT layers [from, to) at timestep t over x, the residual
// [n, 2560], and returns the residual after them: A3's teacher-forced
// instrument.
func (g *GPU) Layers(x *qwen.Mat, t float32, from, to int) (*qwen.Mat, error) {
	if x.Cols != g.dit.H || x.Rows > g.maxTokens || g.encLen == 0 {
		return nil, fmt.Errorf("dit: residual %v (Begin first)", x)
	}
	if err := g.uploadStep(t); err != nil {
		return nil, err
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	g.zeroKeyTail([]uint32{g.hK, g.hV}, g.dit.kv, g.rows, x.Rows)
	gr := g.newGraph()
	for l := from; l < to; l++ {
		if err := gr.layer(g.dit, g.ditL[l], x.Rows, g.vecs(l), true); err != nil {
			return nil, err
		}
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	return g.readX(x.Rows, g.dit.H), nil
}

// Stage is one dispatch's measurement.
type Stage struct {
	Kind  string
	GPU   time.Duration
	Flops float64
	Bytes float64 // read and written, for a streaming dispatch
}

// Profile runs one forward a dispatch at a time, timing each.
func (g *GPU) Profile(x, ctx *qwen.Mat, t float32) ([]Stage, error) {
	gr, _, err := g.stepGraph(x, ctx, t)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(gr.d))
	for i := range gr.d {
		d, err := vk.DispatchMultiTimed(gr.d[i:i+1], 1, 1, true)
		if err != nil {
			return out, fmt.Errorf("dit: dispatch %d (%s): %w", i, gr.kinds[i], err)
		}
		out = append(out, Stage{Kind: gr.kinds[i], GPU: d, Flops: gr.flops[i], Bytes: gr.bytes[i]})
	}
	return out, nil
}

// ---- the audio detokenizer (A4) ----------------------------------------------

// Detokenize turns 5 Hz audio codes into the 25 Hz LM hints the DiT reads
// as its source latents ([5·len(codes), 64]): upstream's
// _decode_audio_codes_to_latents. The FSQ codebook and project_out run on
// the host (six numbers a code); embed_tokens, the two encoder layers over
// every code's own 5 rows (the GROUP attention), the norm and proj_out on
// the device, chunk codes at a time so any song fits the planes.
func (g *GPU) Detokenize(codes []int32) (*qwen.Mat, error) {
	c := g.cfg
	h := g.Host
	H, P := g.enc.H, c.PoolWindow
	out := qwen.NewMat(len(codes)*P, c.Latent)
	chunk := g.rows / P
	if g.detokChunk > 0 {
		chunk = g.detokChunk
	}
	for c0 := 0; c0 < len(codes); c0 += chunk {
		cs := codes[c0:min(c0+chunk, len(codes))]
		q := qwen.NewMat(len(cs), H)
		for i, code := range cs {
			v, err := plan.FSQOutput(int(code), h.FSQW, h.FSQB)
			if err != nil {
				return nil, err
			}
			copy(q.Row(i), v)
		}
		g.narrowRows(g.hIn, q, H+gemmPad)
		gr := g.newGraph()
		if err := gr.gemm(g.big, g.detokIn.bank, "gemm detok embed", g.hIn, g.aS, g.detokIn.off, len(cs), H, H, H+gemmPad); err != nil {
			return nil, err
		}
		if _, err := gr.submit(); err != nil {
			return nil, err
		}
		e := g.abuf.ReadFloat32At(int(g.aS), len(cs)*H)
		n := len(cs) * P
		x := make([]float32, n*H)
		for i := range cs {
			for p := 0; p < P; p++ {
				row := x[(i*P+p)*H : (i*P+p+1)*H]
				for j := range row {
					row[j] = e[i*H+j] + h.DetokEmbedB[j] + h.DetokSpecial[p*H+j]
				}
			}
		}
		g.abuf.WriteFloat32At(int(g.aX), x)
		if err := gr.encoderLayers(g.detokL, n, 0, len(g.detokL)); err != nil {
			return nil, err
		}
		gr.norm(g.enc, "detok norm", n, g.aDetokNorm, g.aZeros)
		if err := gr.gemm(gemmSmall, g.detokOut.bank, "gemm detok out", g.hA, g.aS, g.detokOut.off, n, c.Latent, H, g.enc.ldaH); err != nil {
			return nil, err
		}
		if _, err := gr.submit(); err != nil {
			return nil, err
		}
		raw := g.abuf.ReadFloat32At(int(g.aS), n*c.Latent)
		dst := out.Data[c0*P*c.Latent:]
		for i, v := range raw {
			dst[i] = v + h.DetokProjB[i%c.Latent]
		}
	}
	return out, nil
}

// DetokLayers runs the detokenizer's layers [from, to) over x, its
// residual of 5-row groups: A4's teacher-forced instrument.
func (g *GPU) DetokLayers(x *qwen.Mat, from, to int) (*qwen.Mat, error) {
	if x.Cols != g.enc.H || x.Rows > g.rows || x.Rows%g.cfg.PoolWindow != 0 {
		return nil, fmt.Errorf("dit: detokenizer residual %v", x)
	}
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	gr := g.newGraph()
	if err := gr.encoderLayers(g.detokL, x.Rows, from, to); err != nil {
		return nil, err
	}
	if _, err := gr.submit(); err != nil {
		return nil, err
	}
	return g.readX(x.Rows, g.enc.H), nil
}

// ---- the audio tokenizer (A11c) ----------------------------------------------

// Tokenize turns 25 Hz latents [T, 64] (T a multiple of 5: upstream pads
// with the silence latent first) into 5 Hz audio codes, as upstream's
// tokenize does for a cover's source: audio_acoustic_proj and the pooler's
// embed_tokens (one folded projection), a CLS row before every 5 latents,
// the pooler's two layers over each 6-row group (the GROUP attention), the
// final norm on the CLS rows, and the FSQ on the host. It also returns the
// pooled rows (the FSQ's input, [T/5, 2048]) and each code's margin from a
// quantisation boundary (plan.FSQIndex), for the gates.
func (g *GPU) Tokenize(lat *qwen.Mat) (codes []int32, pooled *qwen.Mat, margins []float32, err error) {
	c := g.cfg
	h := g.Host
	H, P := g.enc.H, c.PoolWindow
	G := P + 1
	if lat.Cols != c.Latent || lat.Rows == 0 || lat.Rows%P != 0 {
		return nil, nil, nil, fmt.Errorf("dit: tokenizer input %v, want rows a multiple of %d", lat, P)
	}
	groups := lat.Rows / P
	pooled = qwen.NewMat(groups, H)
	codes = make([]int32, groups)
	margins = make([]float32, groups)
	chunk := g.rows / G
	for c0 := 0; c0 < groups; c0 += chunk {
		cs := min(chunk, groups-c0)
		in := &qwen.Mat{Rows: cs * P, Cols: c.Latent, Data: lat.Data[c0*P*c.Latent : (c0+cs)*P*c.Latent]}
		g.narrowRows(g.hIn, in, c.Latent+gemmPad)
		gr := g.newGraph()
		if err := gr.gemm(g.big, g.tokIn.bank, "gemm tok embed", g.hIn, g.aS, g.tokIn.off, cs*P, H, c.Latent, c.Latent+gemmPad); err != nil {
			return nil, nil, nil, err
		}
		if _, err := gr.submit(); err != nil {
			return nil, nil, nil, err
		}
		e := g.abuf.ReadFloat32At(int(g.aS), cs*P*H)
		n := cs * G
		x := make([]float32, n*H)
		parallel(cs, func(i int) {
			copy(x[i*G*H:], h.PoolSpecial)
			for p := 0; p < P; p++ {
				row := x[(i*G+1+p)*H : (i*G+2+p)*H]
				src := e[(i*P+p)*H:]
				for j := range row {
					row[j] = src[j] + h.TokB[j]
				}
			}
		})
		g.abuf.WriteFloat32At(int(g.aX), x)
		if err := gr.encoderLayers(g.poolL, n, 0, len(g.poolL)); err != nil {
			return nil, nil, nil, err
		}
		if _, err := gr.submit(); err != nil {
			return nil, nil, nil, err
		}
		all := g.abuf.ReadFloat32At(int(g.aX), n*H)
		cls := qwen.NewMat(cs, H)
		for i := 0; i < cs; i++ {
			copy(cls.Row(i), all[i*G*H:(i*G+1)*H])
		}
		normed, err := h.PoolNorm.Apply(cls)
		if err != nil {
			return nil, nil, nil, err
		}
		copy(pooled.Data[c0*H:], normed.Data)
		for i := 0; i < cs; i++ {
			row := normed.Row(i)
			var z [6]float32
			for d := range z {
				s := float64(h.FSQInB[d])
				for j, v := range h.FSQInW[d*H : (d+1)*H] {
					s += float64(v) * float64(row[j])
				}
				z[d] = float32(s)
			}
			idx, m := plan.FSQIndex(z)
			codes[c0+i], margins[c0+i] = int32(idx), m
		}
	}
	return codes, pooled, margins, nil
}
