package vae

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// GPU runs the decoder's ViT on the device, over the kernels M7 validated
// for the transformer (h3/dit/gpu.go): the fragment-tiled fp16 GEMM, the WMMA
// flash attention with fp16 context out, H3's rotate-half q/k pack, the RMS
// pre-norm, SwiGLU, the gated residual and the mean-subtracting final norm —
// three of them rebuilt at head 64 (shaders.H3VAE*).
//
// The shape of the work is what decides the graph. One decoder call is a
// tile-clip: 7 latent frames × 16 × 16 latents = 1,792 tokens, plus 4
// registers and a zero token, 1,797 rows of full attention. A 480p 5 s
// decode is 105 of them and a 768p one 196, all the same length and all
// independent. So they run **batched**: S sequences side by side at a row
// stride of 1,808 (a whole number of 16-row tiles), every row-local stage —
// the norms, all seven projections, SwiGLU, the gates — one dispatch over
// all S·1,808 rows, and only the attention per sequence (its planes offset
// to the sequence's first tile). The GEMMs then see M ≈ 14k at S = 8, where
// the image DiT's kernels are at their measured rate, instead of 1.8k.
//
// What differs from the transformer, and why:
//
//   - **Biases ride in the GEMM.** Every projection here has one and the
//     transformer's have none. Each A operand carries a column of ones just
//     past its real width and each weight its bias in that column (K padded
//     to the next tile), so a projection and its bias are one GEMM, in fp16
//     as the pipeline's own fp16 autocast has them. proj_in does the same
//     with one-hot columns for the four register tokens, so the whole
//     sequence's input — patches, registers, the zero token — is one GEMM
//     straight into the residual.
//   - **fp16 throughout, residual aside.** M5's oracle has every activation
//     of the stack under 832 (the FFN's down projection output; its input
//     peaks at 354), so no operand needs the transformer's 1/16 FFN scale.
//     The residual is fp32 because it is cheap to, not because it must be.
//   - **Stale keys are harmless here**, which they were not in M7. The
//     attention reads whole 64-key blocks and masks the keys past its count
//     out of P but not out of the row max, so a sequence's last block reads
//     the next sequence's first keys into its max. q and k are RMS-normed
//     per head with no weight, so every score is at most |q||k|/8 = 8 and a
//     foreign key can only move the max by a bounded amount, never
//     underflow a real weight. The planes are zeroed at allocation, and
//     every row the attention can read is written by a pack before it is.
type GPU struct {
	dev *vk.Device
	cfg *Config

	wbuf, abuf, hbuf *vk.Buffer
	banks            []*vk.Buffer
	pipes            map[string]*vk.ComputePipeline
	gemms            []map[gemmKernel]*vk.ComputePipeline
	attnPipe         *vk.ComputePipeline
	mods             []*vk.ShaderModule

	H, ffn, heads, headDim, ropeHalf int
	seqLen, stride, maxSeqs, maxRows int // tokens a sequence, its row stride, sequences a batch
	tileH, tileW                     int // latents a tile, for which the rope table is built
	kH, kF, kIn                      int // K of the projections off H, off the FFN, and proj_in
	ldaH, ldaF, ldaIn                int

	blocks  []blockW
	projIn  headW
	projOut headW
	outBias []float32 // proj_out.bias + proj_out.weight · norm_out.bias, fp32

	// wbuf (fp32): the rope tables for maxRows rows, the q/k norm's ones.
	wCos, wSin, wOnes uint32
	// abuf (fp32, host-cached: the host reads the output back): the
	// residual, the scratch, the per-block vectors, the output.
	aX, aS, aOut     uint32
	aZeros, aNormOut uint32
	actElems         int
	// hbuf (fp16): the q/k/v planes and the A operands.
	hQ, hK, hV, hA, hCtx, hFFN, hIn uint32
	planeRows, hElems               int
}

type headW struct {
	bank int
	off  uint32
	n, k int
}

// blockW is where a block's weights live: the fp16 projections in a bank,
// and its fp32 vectors in the activation arena, which is where the norm and
// gate kernels read them.
type blockW struct {
	bank                 int
	off                  map[proj]uint32
	norm1, norm2, s1, s2 uint32
}

type proj int

const (
	projQ proj = iota
	projK
	projV
	projO
	projGate // the silu'd half of ff.net.0.proj, rows [ffn, 2·ffn)
	projUp   // the linear half, rows [0, ffn)
	projDown
)

var projOrder = []proj{projQ, projK, projV, projO, projGate, projUp, projDown}

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

const gemmBig gemmKernel = 0 // 128x256 tiles, the image DiT's measured winner

var gemmVariants = map[gemmKernel]struct {
	spirv  []byte
	bm, bn int
}{
	gemmBig: {shaders.DiTGEMMWG128x256TiledSWZ8, 128, 256},
}

const (
	tile     = 16
	gemmPad  = 128
	rowAlign = 128 // the big GEMM's M tile
	log2e    = 1.4426950408889634
	// maxBankBytes is one storage buffer's range on this device.
	maxBankBytes = 0xfffffffc
	// maxKeyBlock is the longest key block the attention reads past its count.
	maxKeyBlock = 4 * tile
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// NewGPU stages the decoder (4.9 GB of fp16) for tiles of tileH × tileW
// latents, maxSeqs tile-clips a batch.
func NewGPU(dev *vk.Device, dir string, tileH, tileW, maxSeqs int) (*GPU, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if maxSeqs < 1 {
		return nil, fmt.Errorf("vae: %d sequences a batch", maxSeqs)
	}
	g := &GPU{
		dev: dev, cfg: cfg, pipes: map[string]*vk.ComputePipeline{},
		H: cfg.Hidden(), ffn: cfg.FFN(), heads: cfg.Heads, headDim: cfg.HeadDim, ropeHalf: cfg.RopeWidth() / 2,
		tileH: tileH, tileW: tileW, maxSeqs: maxSeqs,
	}
	g.seqLen = cfg.ClipTokens()*tileH*tileW + cfg.Registers + 1
	g.stride = roundUp(g.seqLen, tile)
	g.maxRows = roundUp(maxSeqs*g.stride, rowAlign)
	// One column past the real width for the bias, K padded to the tile.
	g.kH, g.kF = g.H+tile, g.ffn+tile
	g.kIn = roundUp(cfg.LatentChannels+1+cfg.Registers, tile)
	g.ldaH, g.ldaF, g.ldaIn = g.kH+gemmPad, g.kF+gemmPad, g.kIn+gemmPad

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
	return g, nil
}

// packB narrows a [n, k] row-major weight into 16x16 fragment tiles,
// kt-fastest: dit_gemm.comp's B layout.
func packB(dst []uint16, w []float32, n, k int) {
	kt := k / tile
	parallelFor((n+63)/64, func(c int) {
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

func (g *GPU) projShape(p proj) (n, k int) {
	switch p {
	case projQ, projK, projV, projO:
		return g.H, g.kH
	case projGate, projUp:
		return g.ffn, g.kH
	default:
		return g.H, g.kF
	}
}

// alloc sizes and creates every buffer, and refuses one past the storage-
// buffer range rather than let it clamp silently.
func (g *GPU) alloc(*safetensors.Set) error {
	c := g.cfg
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += roundUp(n, 64)
		return off
	}
	rows := g.maxRows
	g.aX = alloc(rows * g.H)
	// q, k, v in the first half of a block; the attention output, the FFN's
	// two halves and its output in the second.
	g.aS = alloc(rows * max(3*g.H, g.H+2*g.ffn+g.H))
	g.aOut = g.aS // the tail's proj_out lands in the scratch
	g.aZeros = alloc(g.H)
	g.aNormOut = alloc(g.H)
	g.blocks = make([]blockW, c.Layers)
	for i := range g.blocks {
		g.blocks[i] = blockW{norm1: alloc(g.H), norm2: alloc(g.H), s1: alloc(g.H), s2: alloc(g.H)}
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += roundUp(n, 64)
		return off
	}
	g.planeRows = rows + maxKeyBlock
	plane := g.heads * g.planeRows * g.headDim
	g.hQ, g.hK, g.hV = halloc(plane), halloc(plane), halloc(plane)
	g.hA = halloc(rows * g.ldaH)
	g.hCtx = halloc(rows * g.ldaH)
	g.hFFN = halloc(rows * g.ldaF)
	g.hIn = halloc(rows * g.ldaIn)

	w32 := 0
	g.wCos, g.wSin = 0, uint32(rows*g.ropeHalf)
	g.wOnes = uint32(2 * rows * g.ropeHalf)
	w32 = 2*rows*g.ropeHalf + g.headDim

	for _, a := range []struct {
		name  string
		bytes int
	}{{"fp32 activation", g.actElems * 4}, {"fp16 activation", g.hElems * 2}} {
		if a.bytes > maxBankBytes {
			return fmt.Errorf("vae: the %s arena for %d sequences of %d is %d MB, past the %d MB storage-buffer range",
				a.name, g.maxSeqs, g.seqLen, a.bytes>>20, maxBankBytes>>20)
		}
	}
	var err error
	if g.wbuf, err = g.dev.NewBuffer(w32 * 4); err != nil {
		return fmt.Errorf("vae: fp32 weight arena: %w", err)
	}
	if g.abuf, err = g.dev.NewHostCachedBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("vae: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("vae: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	g.hbuf.ZeroUint16At(0, g.hElems)
	// The bias columns: a one past the real width of every A operand, which
	// no kernel writes over (the norms, the attention and SwiGLU write only
	// their real widths).
	one := safetensors.F32ToF16(1)
	for _, a := range []struct {
		off      uint32
		lda, col int
	}{{g.hA, g.ldaH, g.H}, {g.hCtx, g.ldaH, g.H}, {g.hFFN, g.ldaF, g.ffn}} {
		buf := make([]uint16, rows*a.lda)
		for r := 0; r < rows; r++ {
			buf[r*a.lda+a.col] = one
		}
		g.hbuf.WriteUint16At(int(a.off), buf)
	}
	g.abuf.ZeroFloat32At(int(g.aZeros), g.H)
	ones := make([]float32, g.headDim)
	for i := range ones {
		ones[i] = 1
	}
	g.wbuf.WriteFloat32At(int(g.wOnes), ones)
	g.writeRope()
	return nil
}

// writeRope fills the rotary tables for maxRows rows: maxSeqs copies of one
// sequence's, whose tokens are the tile-clip's latents in (t, h, w) order,
// then the registers and the zero token at position 0 (cos 1, sin 0).
// `MiniMaxH3VideoRotaryPosEmbed` and the decoder's grids: each axis runs
// 2·(i + 0.5)/n − 1, the angle is 2π·x·inv_freq with inv_freq =
// θ^(−j/8) over 8 frequencies an axis, and the (t, h, w) angles are
// concatenated to 24 — half the rotated 48, which is all the pack reads.
func (g *GPU) writeRope() {
	c := g.cfg
	nf := g.ropeHalf / 3
	inv := make([]float32, nf)
	for j := range inv {
		x := float32(float64(j) * 6 / float64(c.RopeWidth()))
		inv[j] = float32(1 / float32(math.Pow(c.RopeTheta, float64(x))))
	}
	axis := func(i, n int) float32 {
		v := float32(float32(i)+0.5) / float32(n)
		return float32(2*v) - 1
	}
	twoPi := float32(2 * math.Pi)
	seqCos := make([]float32, g.stride*g.ropeHalf)
	seqSin := make([]float32, g.stride*g.ropeHalf)
	for r := 0; r < g.stride; r++ {
		pos := [3]float32{}
		if r < g.seqLen-c.Registers-1 {
			hw := g.tileH * g.tileW
			pos = [3]float32{axis(r/hw, c.ClipTokens()), axis(r%hw/g.tileW, g.tileH), axis(r%g.tileW, g.tileW)}
		}
		for a := 0; a < 3; a++ {
			for j := 0; j < nf; j++ {
				ang := float32(twoPi*pos[a]) * inv[j]
				seqCos[r*g.ropeHalf+a*nf+j] = float32(math.Cos(float64(ang)))
				seqSin[r*g.ropeHalf+a*nf+j] = float32(math.Sin(float64(ang)))
			}
		}
	}
	for s := 0; s < g.maxRows/g.stride+1; s++ {
		n := min(g.stride, g.maxRows-s*g.stride)
		if n <= 0 {
			break
		}
		g.wbuf.WriteFloat32At(int(g.wCos)+s*g.stride*g.ropeHalf, seqCos[:n*g.ropeHalf])
		g.wbuf.WriteFloat32At(int(g.wSin)+s*g.stride*g.ropeHalf, seqSin[:n*g.ropeHalf])
	}
}

// f32 reads a named tensor as float32, checking its element count.
func f32(set *safetensors.Set, name string, n int) ([]float32, error) {
	t, err := set.Get(name)
	if err != nil {
		return nil, err
	}
	v, err := t.F32(nil)
	if err != nil {
		return nil, err
	}
	if len(v) != n {
		return nil, fmt.Errorf("vae: %s has %d values, want %d", name, len(v), n)
	}
	return v, nil
}

// stage lays out and fills the fp16 banks (a block's seven projections
// share one, so a GEMM's pipeline is picked by the block), and the fp32
// vectors.
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
	perBlock := 0
	for _, p := range projOrder {
		n, k := g.projShape(p)
		perBlock += n * k
	}
	b, off := place(g.H * g.kIn)
	g.projIn = headW{bank: b, off: off, n: g.H, k: g.kIn}
	b, off = place(c.PatchOut() * g.H)
	g.projOut = headW{bank: b, off: off, n: c.PatchOut(), k: g.H}
	for i := range g.blocks {
		w := &g.blocks[i]
		w.bank, off = place(perBlock)
		w.off = map[proj]uint32{}
		for _, p := range projOrder {
			n, k := g.projShape(p)
			w.off[p] = off
			off += uint32(n * k)
		}
	}
	for _, n := range bankElems {
		buf, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("vae: fp16 bank %d (%d MB): %w", len(g.banks), (n*2)>>20, err)
		}
		g.banks = append(g.banks, buf)
	}
	put := func(bank int, off uint32, w []float32, n, k int) {
		buf := make([]uint16, n*k)
		packB(buf, w, n, k)
		g.banks[bank].WriteUint16At(int(off), buf)
	}

	// proj_in: the latent's channels, the bias column, then one column per
	// register token, whose weight column is the register itself.
	L, H := c.LatentChannels, g.H
	w, err := f32(set, "decoder.proj_in.weight", H*L)
	if err != nil {
		return err
	}
	bias, err := f32(set, "decoder.proj_in.bias", H)
	if err != nil {
		return err
	}
	regs, err := f32(set, "decoder.register_tokens", c.Registers*H)
	if err != nil {
		return err
	}
	in := withBias(w, bias, H, L, g.kIn)
	for r := 0; r < c.Registers; r++ {
		for i := 0; i < H; i++ {
			in[i*g.kIn+L+1+r] = regs[r*H+i]
		}
	}
	put(g.projIn.bank, g.projIn.off, in, H, g.kIn)

	// The tail: LayerNorm(x)·w + b, then proj_out. The norm's bias goes
	// through the projection on the host in float64, into proj_out's bias,
	// which the host adds when it reads the output back.
	nw, err := f32(set, "decoder.norm_out.weight", H)
	if err != nil {
		return err
	}
	nb, err := f32(set, "decoder.norm_out.bias", H)
	if err != nil {
		return err
	}
	po, err := f32(set, "decoder.proj_out.weight", c.PatchOut()*H)
	if err != nil {
		return err
	}
	pb, err := f32(set, "decoder.proj_out.bias", c.PatchOut())
	if err != nil {
		return err
	}
	g.outBias = make([]float32, c.PatchOut())
	for o := range g.outBias {
		s := float64(pb[o])
		for i := 0; i < H; i++ {
			s += float64(po[o*H+i]) * float64(nb[i])
		}
		g.outBias[o] = float32(s)
	}
	put(g.projOut.bank, g.projOut.off, po, c.PatchOut(), H)
	g.abuf.WriteFloat32At(int(g.aNormOut), nw)

	for i, bw := range g.blocks {
		p := fmt.Sprintf("decoder.transformer_blocks.%d.", i)
		get := func(name string, n int) []float32 {
			if err != nil {
				return nil
			}
			var v []float32
			v, err = f32(set, p+name, n)
			return v
		}
		wq, bq := get("attn.to_q.weight", H*H), get("attn.to_q.bias", H)
		wk, bk := get("attn.to_k.weight", H*H), get("attn.to_k.bias", H)
		wv, bv := get("attn.to_v.weight", H*H), get("attn.to_v.bias", H)
		wo, bo := get("attn.to_out.0.weight", H*H), get("attn.to_out.0.bias", H)
		up, bup := get("ff.net.0.proj.weight", 2*g.ffn*H), get("ff.net.0.proj.bias", 2*g.ffn)
		down, bdown := get("ff.net.2.weight", H*g.ffn), get("ff.net.2.bias", H)
		n1, n2 := get("norm1.weight", H), get("norm2.weight", H)
		s1, s2 := get("scale1", H), get("scale2", H)
		if err != nil {
			return err
		}
		// SwiGLU is value·silu(gate) with the projection's rows [value | gate].
		lin := map[proj][2][]float32{
			projQ: {wq, bq}, projK: {wk, bk}, projV: {wv, bv}, projO: {wo, bo},
			projUp: {up[:g.ffn*H], bup[:g.ffn]}, projGate: {up[g.ffn*H:], bup[g.ffn:]},
			projDown: {down, bdown},
		}
		for _, pr := range projOrder {
			n, k := g.projShape(pr)
			put(bw.bank, bw.off[pr], withBias(lin[pr][0], lin[pr][1], n, k-tile, k), n, k)
		}
		g.abuf.WriteFloat32At(int(bw.norm1), n1)
		g.abuf.WriteFloat32At(int(bw.norm2), n2)
		g.abuf.WriteFloat32At(int(bw.s1), s1)
		g.abuf.WriteFloat32At(int(bw.s2), s2)
	}
	return nil
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
		"norm":      shaders.H3NormMod,
		"qkpack":    shaders.H3VAEQKPackHD64,
		"pack":      shaders.H3VAEPackHD64,
		"gate":      shaders.DiTGateAdd,
		"swiglu":    shaders.DiTSwiGLUF16,
		"finalnorm": shaders.DiTFinalNorm,
	} {
		p, err := newPipe(spirv, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("vae: pipeline %s: %w", name, err)
		}
		g.pipes[name] = p
	}
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return err
	}
	if !g.dev.Features().SubgroupSizeControl || !sgs.Supported || 32 < sgs.MinSubgroupSize || 32 > sgs.MaxSubgroupSize {
		return fmt.Errorf("vae: the attention build needs a pinned wave32")
	}
	if g.attnPipe, err = newPipe(shaders.H3VAEAttnHD64, vk.PipelineSpec{Buffers: base, PushConstantSize: pcSize, RequiredSubgroupSize: 32}); err != nil {
		return fmt.Errorf("vae: attention pipeline: %w", err)
	}
	for b := range g.banks {
		m := map[gemmKernel]*vk.ComputePipeline{}
		for k, v := range gemmVariants {
			p, err := newPipe(v.spirv, vk.PipelineSpec{Buffers: []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]}, PushConstantSize: pcSize})
			if err != nil {
				return fmt.Errorf("vae: gemm bank %d: %w", b, err)
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
	if g.attnPipe != nil {
		g.attnPipe.Destroy()
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

// WeightBytes and ActivationBytes are what the decoder holds on the device.
func (g *GPU) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}

func (g *GPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// graph accumulates one batch's dispatches.
type graph struct {
	g     *GPU
	d     []vk.MultiDispatch
	kinds []string
	flops []float64
	base  pushConstants
}

func (g *GPU) newGraph() *graph {
	return &graph{g: g, base: pushConstants{Heads: uint32(g.heads), HeadDim: uint32(g.headDim)}}
}

func (gr *graph) add(pipe *vk.ComputePipeline, kind string, gx, gy uint32, pc pushConstants, flops float64) {
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: pipe, GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
	gr.flops = append(gr.flops, flops)
}

// gemm issues C[mPad, n] fp32 at cOff = A[m, k] fp16 at aOff (row stride
// lda) × B, the fragment-tiled weight at bOff of bank.
func (gr *graph) gemm(kernel gemmKernel, bank int, kind string, aOff, cOff, bOff uint32, m, n, k, lda int) error {
	v := gemmVariants[kernel]
	if n%v.bn != 0 || k%tile != 0 {
		return fmt.Errorf("vae: %s: N=%d K=%d do not tile by %d/%d", kind, n, k, v.bn, tile)
	}
	mPad := roundUp(m, v.bm)
	pc := gr.base
	pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(mPad), uint32(n), uint32(k), uint32(lda)
	gr.add(gr.g.gemms[bank][kernel], kind, uint32(n/v.bn), uint32(mPad/v.bm), pc, 2*float64(mPad)*float64(n)*float64(k))
	return nil
}

// submitBudget and the rest are M7's guard against the 2-second ring
// watchdog (research/p0-ring-watchdog.md): a submission closes before its
// estimated device time passes the budget.
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
		t, err := vk.DispatchMultiTimed(gr.d[i:j], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("vae: dispatch %d-%d (%s): %w", i, j-1, gr.kinds[i], err)
		}
		total += t
		i = j
	}
	gr.d, gr.kinds, gr.flops = gr.d[:0], gr.kinds[:0], gr.flops[:0]
	return total, nil
}

// blockPass records one block over seqs sequences.
func (gr *graph) blockPass(w blockW, seqs int) error {
	g := gr.g
	rows := seqs * g.stride
	eps := math.Float32bits(float32(g.cfg.NormEps))
	// The GEMMs write whole 128-row tiles, so the scratch's regions are
	// spaced by the padded row count and none spills into the next.
	pad := roundUp(rows, rowAlign)
	aQ := g.aS
	aK := aQ + uint32(pad*g.H)
	aV := aK + uint32(pad*g.H)
	aAttn := g.aS
	aGate := aAttn + uint32(pad*g.H)
	aUp := aGate + uint32(pad*g.ffn)
	aFF := aUp + uint32(pad*g.ffn)
	planeTile := uint32(g.headDim * tile) // halves a 16-row tile of one head

	norm := func(kind string, weight uint32) {
		pc := gr.base
		pc.Tokens, pc.Dim, pc.LDA = uint32(rows), uint32(g.H), uint32(g.ldaH)
		pc.InOff, pc.OutOff = g.aX, g.hA
		pc.Aux0, pc.Aux1, pc.Eps = weight, g.aZeros, eps
		gr.add(g.pipes["norm"], kind, uint32(rows), 1, pc, 0)
	}
	gate := func(kind string, y, vec uint32) {
		pc := gr.base
		pc.Tokens, pc.Dim = uint32(rows), uint32(g.H)
		pc.InOff, pc.OutOff = y, g.aX
		pc.Aux0, pc.Aux2 = vec, 1
		gr.add(g.pipes["gate"], kind, uint32(rows), 1, pc, 0)
	}

	norm("attn in", w.norm1)
	for _, pr := range []struct {
		p   proj
		out uint32
	}{{projQ, aQ}, {projK, aK}, {projV, aV}} {
		if err := gr.gemm(gemmBig, w.bank, "gemm qkv", g.hA, pr.out, w.off[pr.p], rows, g.H, g.kH, g.ldaH); err != nil {
			return err
		}
	}
	tiles := rows / tile
	for _, qk := range []struct {
		kind     string
		src, dst uint32
		scale    float32
	}{
		{"qkpack q", aQ, g.hQ, float32(1/math.Sqrt(float64(g.headDim))) * log2e},
		{"qkpack k", aK, g.hK, 1},
	} {
		pc := gr.base
		pc.Tokens, pc.Dim = uint32(rows), uint32(g.H)
		pc.InOff, pc.OutOff = qk.src, qk.dst
		pc.WOff, pc.Aux0 = g.wCos, g.wSin
		pc.Aux1, pc.Aux2 = uint32(g.planeRows), g.wOnes
		pc.Span, pc.Eps, pc.Scale = uint32(tiles), eps, math.Float32bits(qk.scale)
		gr.add(g.pipes["qkpack"], qk.kind, uint32((tiles+7)/8), uint32(g.heads), pc, 0)
	}
	pc := gr.base
	pc.Tokens, pc.Dim = uint32(rows), uint32(g.H)
	pc.InOff, pc.OutOff = aV, g.hV
	pc.Aux0, pc.Aux1, pc.Aux2 = 1, uint32(g.planeRows), uint32(tiles)
	pc.Scale = math.Float32bits(1)
	gr.add(g.pipes["pack"], "pack v", uint32((tiles+7)/8), uint32(g.heads), pc, 0)

	// The attention, a sequence at a time, each over its own keys.
	for s := 0; s < seqs; s++ {
		t0 := uint32(s*g.stride/tile) * planeTile
		pc := gr.base
		pc.Tokens = uint32(g.seqLen)
		pc.InOff, pc.KOff, pc.VOff = g.hQ+t0, g.hK+t0, g.hV+t0
		pc.OutOff, pc.LDA = g.hCtx+uint32(s*g.stride*g.ldaH), uint32(g.ldaH)
		pc.Aux1 = uint32(g.planeRows)
		gr.add(g.attnPipe, "attention", uint32((g.seqLen+tile-1)/tile), uint32(g.heads), pc,
			4*float64(g.seqLen)*float64(g.seqLen)*float64(g.headDim)*float64(g.heads))
	}
	if err := gr.gemm(gemmBig, w.bank, "gemm o", g.hCtx, aAttn, w.off[projO], rows, g.H, g.kH, g.ldaH); err != nil {
		return err
	}
	gate("gate attn", aAttn, w.s1)
	norm("ffn in", w.norm2)
	if err := gr.gemm(gemmBig, w.bank, "gemm gate", g.hA, aGate, w.off[projGate], rows, g.ffn, g.kH, g.ldaH); err != nil {
		return err
	}
	if err := gr.gemm(gemmBig, w.bank, "gemm up", g.hA, aUp, w.off[projUp], rows, g.ffn, g.kH, g.ldaH); err != nil {
		return err
	}
	pcG := gr.base
	pcG.Tokens, pcG.Dim, pcG.LDA = uint32(rows), uint32(g.ffn), uint32(g.ldaF)
	pcG.InOff, pcG.KOff, pcG.OutOff = aGate, aUp, g.hFFN
	pcG.Scale = math.Float32bits(1)
	gr.add(g.pipes["swiglu"], "swiglu", uint32(rows), 1, pcG, 0)
	if err := gr.gemm(gemmBig, w.bank, "gemm down", g.hFFN, aFF, w.off[projDown], rows, g.H, g.kF, g.ldaF); err != nil {
		return err
	}
	gate("gate ffn", aFF, w.s2)
	return nil
}

// input writes a batch's proj_in operand: each tile-clip's latents
// (post_quant_conv applied, [24, 7, th, tw]) as rows in (t, h, w) order with
// the bias column set, then its registers' one-hot rows; the zero token and
// the stride's pad rows are all zero.
func (g *GPU) input(clips []*Tensor) {
	c := g.cfg
	L := c.LatentChannels
	buf := make([]uint16, len(clips)*g.stride*g.ldaIn)
	one := safetensors.F32ToF16(1)
	parallelFor(len(clips), func(s int) {
		z := clips[s]
		hw := z.H * z.W
		base := s * g.stride * g.ldaIn
		for t := 0; t < z.T; t++ {
			for p := 0; p < hw; p++ {
				row := buf[base+(t*hw+p)*g.ldaIn:]
				for ch := 0; ch < L; ch++ {
					row[ch] = safetensors.F32ToF16(z.Data[(ch*z.T+t)*hw+p])
				}
				row[L] = one
			}
		}
		for r := 0; r < c.Registers; r++ {
			buf[base+(z.T*hw+r)*g.ldaIn+L+1+r] = one
		}
	})
	g.hbuf.WriteUint16At(int(g.hIn), buf)
}

// run records and submits the whole decoder over the batch's sequences: the
// input GEMM straight into the residual, the blocks, and the tail into
// aOut. last < 0 runs every block, and tail false stops after them.
func (g *GPU) run(seqs, first, last int, input, tail bool) (time.Duration, error) {
	gr := g.newGraph()
	rows := seqs * g.stride
	if input {
		if err := gr.gemm(gemmBig, g.projIn.bank, "gemm proj_in", g.hIn, g.aX, g.projIn.off, rows, g.H, g.kIn, g.ldaIn); err != nil {
			return 0, err
		}
	}
	if last < 0 {
		last = len(g.blocks)
	}
	for b := first; b < last; b++ {
		if err := gr.blockPass(g.blocks[b], seqs); err != nil {
			return 0, err
		}
	}
	if tail {
		pc := gr.base
		pc.Tokens, pc.Dim, pc.LDA = uint32(rows), uint32(g.H), uint32(g.ldaH)
		pc.InOff, pc.OutOff = g.aX, g.hA
		pc.Aux0, pc.Eps = g.aNormOut, math.Float32bits(float32(g.cfg.NormEps))
		gr.add(g.pipes["finalnorm"], "norm out", uint32(rows), 1, pc, 0)
		if err := gr.gemm(gemmBig, g.projOut.bank, "gemm proj_out", g.hA, g.aOut, g.projOut.off, rows, g.projOut.n, g.H, g.ldaH); err != nil {
			return 0, err
		}
	}
	return gr.submit()
}

// DecodeTiles runs the decoder over tile-clips — post_quant_conv'd latents
// [24, 7, th, tw] at the staged tile size, at most maxSeqs of them — and
// returns each one's pixels, [3, 28, 16·th, 16·tw], ImageNet-normalised.
func (g *GPU) DecodeTiles(clips []*Tensor) ([]*Tensor, time.Duration, error) {
	if len(clips) == 0 || len(clips) > g.maxSeqs {
		return nil, 0, fmt.Errorf("vae: %d tile-clips, staged for 1..%d", len(clips), g.maxSeqs)
	}
	c := g.cfg
	for _, z := range clips {
		if z.C != c.LatentChannels || z.T != c.ClipTokens() || z.H != g.tileH || z.W != g.tileW {
			return nil, 0, fmt.Errorf("vae: tile-clip %dx%dx%dx%d, staged for %dx%dx%dx%d",
				z.C, z.T, z.H, z.W, c.LatentChannels, c.ClipTokens(), g.tileH, g.tileW)
		}
	}
	g.input(clips)
	took, err := g.run(len(clips), 0, -1, true, true)
	if err != nil {
		return nil, 0, err
	}
	out := make([]*Tensor, len(clips))
	n := g.projOut.n
	pt, ps := c.Temporal(), c.Spatial()
	oc := c.OutChannels
	patches := c.ClipTokens() * g.tileH * g.tileW
	parallelFor(len(clips), func(s int) {
		raw := g.abuf.ReadFloat32At(int(g.aOut)+s*g.stride*n, patches*n)
		v := NewTensor(oc, c.ClipTokens()*pt, g.tileH*ps, g.tileW*ps)
		hw := g.tileH * g.tileW
		// proj_out's columns are (channel, t, y, x) of the token's patch.
		for tok := 0; tok < patches; tok++ {
			f, y, x := tok/hw, tok%hw/g.tileW, tok%g.tileW
			row := raw[tok*n : (tok+1)*n]
			i := 0
			for ch := 0; ch < oc; ch++ {
				for dt := 0; dt < pt; dt++ {
					for dy := 0; dy < ps; dy++ {
						dst := v.idx(ch, f*pt+dt, y*ps+dy, x*ps)
						for dx := 0; dx < ps; dx++ {
							v.Data[dst+dx] = row[i] + g.outBias[i]
							i++
						}
					}
				}
			}
		}
		out[s] = v
	})
	return out, took, nil
}

// Blocks runs blocks [from, to) over one sequence's residual [seqLen,
// hidden] and returns it after them: the teacher-forced instrument the gate
// uses against the oracle's per-block dumps.
func (g *GPU) Blocks(x []float32, from, to int) ([]float32, error) {
	if len(x) != g.seqLen*g.H {
		return nil, fmt.Errorf("vae: residual of %d values for %d×%d", len(x), g.seqLen, g.H)
	}
	g.abuf.WriteFloat32At(int(g.aX), x)
	g.abuf.ZeroFloat32At(int(g.aX)+len(x), (g.stride-g.seqLen)*g.H)
	if _, err := g.run(1, from, to, false, false); err != nil {
		return nil, err
	}
	return g.abuf.ReadFloat32At(int(g.aX), len(x)), nil
}

// Stats is the last decode's accounting.
type Stats struct {
	Calls, Batches int
	Device         time.Duration
}

// Decode decodes a denormalised latent [24, lf, lh, lw] into its video,
// [3, frames, 16·lh, 16·lw] ImageNet-normalised — diffusers' `vae.decode`,
// with its default tiling. The staged tile size must be the plan's.
func (g *GPU) Decode(z *Tensor, pqc *PostQuantConv) (*Tensor, Stats, error) {
	var st Stats
	c := g.cfg
	p, err := c.NewPlan(z.T, z.H, z.W)
	if err != nil {
		return nil, st, err
	}
	if p.TileH != g.tileH || p.TileW != g.tileW {
		return nil, st, fmt.Errorf("vae: a %dx%d latent tiles by %dx%d, staged for %dx%d", z.H, z.W, p.TileH, p.TileW, g.tileH, g.tileW)
	}
	zq := pqc.Apply(z)
	type item struct{ clip, tile int }
	var items []item
	for ci := 0; ci < p.Clips; ci++ {
		for ti := 0; ti < p.Tiles(); ti++ {
			items = append(items, item{ci, ti})
		}
	}
	tiles := make([][]*Tensor, p.Clips)
	stitched := make([]*Tensor, p.Clips)
	nx := len(p.XStarts)
	for b := 0; b < len(items); b += g.maxSeqs {
		batch := items[b:min(b+g.maxSeqs, len(items))]
		in := make([]*Tensor, len(batch))
		for i, it := range batch {
			y0, x0 := p.YStarts[it.tile/nx], p.XStarts[it.tile%nx]
			in[i] = zq.Slice(it.clip*c.chunkTokens(), p.ClipTokens, y0, p.TileH, x0, p.TileW)
		}
		out, took, err := g.DecodeTiles(in)
		if err != nil {
			return nil, st, err
		}
		st.Device += took
		st.Batches++
		st.Calls += len(batch)
		for i, it := range batch {
			if tiles[it.clip] == nil {
				tiles[it.clip] = make([]*Tensor, p.Tiles())
			}
			tiles[it.clip][it.tile] = out[i]
			if it.tile == p.Tiles()-1 {
				stitched[it.clip] = p.stitch(tiles[it.clip])
				tiles[it.clip] = nil
			}
		}
	}
	return c.assemble(stitched, p.Frames), st, nil
}

// PostQuantConv is the 1x1x1 conv between the latent and the decoder,
// applied on the host in float64: 24 × 24 multiply-adds a latent.
type PostQuantConv struct {
	C    int
	W, B []float32
}

func LoadPostQuantConv(dir string) (*PostQuantConv, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	n := cfg.LatentChannels
	w, err := f32(set, "post_quant_conv.weight", n*n)
	if err != nil {
		return nil, err
	}
	b, err := f32(set, "post_quant_conv.bias", n)
	if err != nil {
		return nil, err
	}
	return &PostQuantConv{C: n, W: w, B: b}, nil
}

func (q *PostQuantConv) Apply(z *Tensor) *Tensor {
	out := NewTensor(z.C, z.T, z.H, z.W)
	plane := z.T * z.H * z.W
	parallelFor(q.C, func(o int) {
		for i := 0; i < plane; i++ {
			s := float64(q.B[o])
			for c := 0; c < q.C; c++ {
				s += float64(q.W[o*q.C+c]) * float64(z.Data[c*plane+i])
			}
			out.Data[o*plane+i] = float32(s)
		}
	})
	return out
}

// callFlops is one tile-clip's useful work: every projection at its real
// K over its real rows, and the attention's two products.
func (g *GPU) callFlops() float64 {
	n, H, F := float64(g.seqLen), float64(g.H), float64(g.ffn)
	perBlock := 2*n*(4*H*H+3*H*F) + 4*n*n*H
	return float64(len(g.blocks))*perBlock + 2*n*H*float64(g.projOut.n)
}
