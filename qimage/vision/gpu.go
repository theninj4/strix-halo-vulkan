package vision

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The vision tower on the device — IMAGE.md Q8, the last piece of an edit
// still running on the CPU.
//
// The CPU tower in vision.go is the oracle and is also the reason this file
// exists: 27 blocks over a 1024² condition image is 4096 patches and tens of
// minutes, against a transformer that costs 90 s for the whole image.
//
// **Precision was measured before it was chosen** (TestFP16Ladder). This
// tower sits between the two conventions already in the tree: the VAE is
// fp32 because its activations reach 1e5 against half's 65504, and the DiT
// and text encoder are fp16 on the matrix cores because that is where their
// speed is. The tower's residual stream reaches absmax 1.3e4 — inside half's
// range but with three decimal digits left — and Q8.2 measured the tower
// amplifying perturbations by ~1600x. Running every linear with fp16
// operands and fp32 accumulation on the CPU oracle costs **rel 0.14 on the
// merged rows** and does not overflow, which is what said the matrix cores
// are usable here and what this port is gated against. For scale: the
// checkpoint's own bf16 pipeline deviates from fp32 by rel 2.0 on the text
// embedding it ultimately feeds.
//
// Almost every kernel is the DiT's. Two are new (qvit_bias_act, qvit_rope)
// and three reuses are worth naming because they are not obvious:
//
//   - **the affine LayerNorm is dit_final_norm plus algebra.** That kernel
//     computes LN(x) * scale and narrows to fp16; the ViT wants
//     LN(x) * gamma + beta feeding a linear W(.) + b. Since
//     W(LN(x)*gamma + beta) + b = W(LN(x)*gamma) + (W*beta + b), the beta is
//     folded into the following projection's bias at load time and the
//     kernel is used verbatim;
//   - **the residual add is dit_gate_add with its gate dropped** (aux2 = 0),
//     which is the path the DiT's two unmodulated blocks already take;
//   - **the patch merger's concatenation is free.** Four consecutive rows of
//     1152 in a row-major buffer *are* one row of 4608, so the output
//     merger normalises with a tight row stride and the GEMM reads the same
//     bytes four rows at a time, while the three deepstack mergers normalise
//     the 4608 directly.
//
// Attention is the *scalar* kernel, which takes its head width from a push
// constant and so runs this tower's 72 unchanged. The matrix-core one is
// built per head dim and 72 does not divide its 16-wide tile; padding to 80
// would work and is a percent, not a capability — measured in
// TestGPUTowerTiming before it is worth doing.

const (
	// ffnAlign is the GEMM tile's N granularity. The MLP is 4304 wide, which
	// no tile divides, so its weights are zero-padded to the next multiple —
	// the pad columns contribute nothing through GELU (which fixes zero) and
	// nothing through the second layer's zero rows.
	ffnAlign = 64
	// ldaPad keeps a K-strided fragment load off one DRAM channel (IDEAS
	// §2.3), as in the DiT.
	ldaPad = 128
	// attnTokenPad spaces the transposed keys' rows for the same reason.
	attnTokenPad = 64
)

// GPU runs the tower's graph on a Vulkan device for one patch grid.
type GPU struct {
	dev *vk.Device
	cfg Config

	// The staged budget, and the grid of the run in progress. The arenas are
	// sized for the first and every dispatch is bounded by the second.
	maxRows, maxMerged int
	gridH, gridW       int
	rows               int // patches
	merged             int // rows / 4

	dim, heads, headDim, ffn, ffnPad int
	mergeIn, mergeOut                int

	wbuf  *vk.Buffer // fp32: biases
	abuf  *vk.Buffer // fp32 activations
	hbuf  *vk.Buffer // fp16 GEMM A operands
	banks []*vk.Buffer

	pipes map[string]*vk.ComputePipeline
	gemms []map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	// activation offsets
	aX, aQ, aK, aV, aKT, aCtx, aY, aH, aPix, aPos uint32
	aCos, aSin, aGamma, aDeep, aMerged            uint32
	actElems, hElems                              int
	hA                                            uint32
	ldaA                                          int

	// weight offsets
	wPatchBias uint32
	patchBank  int
	patchOff   uint32
	cpu        *Model
	blocks     []gpuBlock
	mergers    []gpuMerger
	gammaOff   []uint32 // per uploaded gamma vector, into the activation arena
	gammas     []float32
}

type gpuBlock struct {
	bank           int
	wQ, wK, wV, wO uint32
	wFC1, wFC2     uint32
	bQ, bK, bV, bO uint32
	bFC1, bFC2     uint32
	gamma1, gamma2 uint32 // indices into gammaOff
	deepstack      int    // -1, or the merger that taps this block
}

type gpuMerger struct {
	bank       int
	wFC1, wFC2 uint32
	bFC1, bFC2 uint32
	gamma      uint32
	post       bool // normalise the 4608 concatenation rather than each 1152
	act        uint32
}

// pushConstants mirrors the PC block in shaders/dit_common.glsl.
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

const noW = 0xffffffff

// gemmBN is the N granularity of the GEMM kernel this port uses. Reg64 is
// the small-tile variant, which is what a 1152-wide model needs: the 256
// tile does not divide 1152.
const gemmBN = 64
const gemmBM = 64

// NewGPU stages the tower for patch grids of up to maxRows patches and builds
// its pipelines.
//
// It is a *budget* rather than a grid because a condition image's grid
// follows its aspect ratio: `calculate_dimensions` fixes the area and lets
// the sides fall where they will, so a served edit hands this tower a
// different pair of numbers every request. The arenas are laid out for the
// budget once and Forward takes the grid, exactly as the VAE's graphs take
// an image size under a staged ceiling.
func NewGPU(dev *vk.Device, cpu *Model, maxRows int) (*GPU, error) {
	cfg := cpu.Cfg
	if len(cpu.Blocks) != cfg.Depth {
		return nil, fmt.Errorf("vision: %d blocks loaded, the config has %d", len(cpu.Blocks), cfg.Depth)
	}
	merge := cfg.SpatialMergeSize * cfg.SpatialMergeSize
	if maxRows <= 0 || maxRows%merge != 0 {
		return nil, fmt.Errorf("vision: a patch budget of %d does not group into %ds", maxRows, merge)
	}
	g := &GPU{
		dev: dev, cfg: cfg, cpu: cpu,
		maxRows: maxRows, maxMerged: maxRows / merge,
		rows: maxRows, merged: maxRows / merge,
		dim: cfg.HiddenSize, heads: cfg.NumHeads, headDim: cfg.HeadDim(),
		ffn: cfg.IntermediateSize, mergeIn: cfg.HiddenSize * merge, mergeOut: cfg.OutHiddenSize,
		pipes: map[string]*vk.ComputePipeline{},
	}
	g.ffnPad = (g.ffn + ffnAlign - 1) &^ (ffnAlign - 1)
	if err := g.stageWeights(cpu); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.allocActivations(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	g.abuf.WriteFloat32At(int(g.aGamma), g.gammas)
	return g, nil
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
	for _, b := range append([]*vk.Buffer{g.abuf, g.hbuf, g.wbuf}, g.banks...) {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.gemms, g.mods, g.banks = map[string]*vk.ComputePipeline{}, nil, nil, nil
	g.abuf, g.hbuf, g.wbuf = nil, nil, nil
}

// WeightBytes and ActivationBytes are what the tower costs in device memory.
func (g *GPU) WeightBytes() int {
	n := g.wbuf.Size()
	for _, b := range g.banks {
		n += b.Size()
	}
	return n
}
func (g *GPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// f32arena accumulates the fp32 weight arena (biases and the folded betas).
type f32arena struct {
	data []float32
}

func (a *f32arena) put(v []float32) uint32 {
	off := uint32(len(a.data))
	a.data = append(a.data, v...)
	return off
}

// fold returns b + W*beta, which is what lets dit_final_norm — a norm with a
// scale and no shift — stand in for the tower's affine LayerNorm.
func fold(lin *Linear, beta []float32, outPad int) []float32 {
	out := make([]float32, outPad)
	for o := 0; o < lin.Out; o++ {
		var acc float64
		w := lin.Weight[o*lin.In : (o+1)*lin.In]
		for i, b := range beta {
			acc += float64(w[i]) * float64(b)
		}
		if lin.Bias != nil {
			acc += float64(lin.Bias[o])
		}
		out[o] = float32(acc)
	}
	return out
}

// tile4 repeats a 1152-wide beta across the merger's 4608-wide input, which
// is what the concatenation of four normalised rows does to it.
func tile4(beta []float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = beta[i%len(beta)]
	}
	return out
}

// padWeight zero-extends a [out, in] weight to [outPad, inPad].
func padWeight(w []float32, out, in, outPad, inPad int) []float32 {
	if out == outPad && in == inPad {
		return w
	}
	dst := make([]float32, outPad*inPad)
	for o := 0; o < out; o++ {
		copy(dst[o*inPad:o*inPad+in], w[o*in:(o+1)*in])
	}
	return dst
}

func (g *GPU) stageWeights(cpu *Model) error {
	var w32 f32arena
	// The gammas live in the *activation* arena, because dit_final_norm
	// reads its scale vector from there — in the DiT it is a per-step
	// modulation row, and here it is a constant uploaded once.
	gamma := func(v []float32) uint32 {
		off := uint32(len(g.gammas))
		g.gammas = append(g.gammas, v...)
		g.gammaOff = append(g.gammaOff, off)
		return uint32(len(g.gammaOff) - 1)
	}

	type staged struct {
		bank int
		off  uint32
		w    []float32
		n, k int
	}
	var pending []staged
	var bankElems []int
	cur := -1
	place := func(n, k int, weight []float32) (int, uint32) {
		need := n * k
		if cur < 0 || (bankElems[cur]+need)*2 > maxBankBytes {
			bankElems = append(bankElems, 0)
			cur = len(bankElems) - 1
		}
		off := uint32(bankElems[cur])
		bankElems[cur] += need
		pending = append(pending, staged{bank: cur, off: off, w: weight, n: n, k: k})
		return cur, off
	}

	// The patch embedding: a Conv3d whose kernel is exactly one patch, i.e. a
	// linear over the 1536 flattened values.
	pxBank, pxOff := place(g.dim, g.cfg.PatchElems(), cpu.PatchProj.Weight)
	g.wPatchBias = w32.put(cpu.PatchProj.Bias)
	g.blocks = make([]gpuBlock, len(cpu.Blocks))
	deepOf := map[int]int{}
	for j, idx := range g.cfg.DeepstackIndexes {
		deepOf[idx] = j
	}
	for i := range cpu.Blocks {
		b := &cpu.Blocks[i]
		blk := &g.blocks[i]
		blk.deepstack = -1
		if j, ok := deepOf[i]; ok && j < len(cpu.Deepstack) {
			blk.deepstack = j
		}
		blk.gamma1 = gamma(b.Norm1.Weight)
		blk.gamma2 = gamma(b.Norm2.Weight)

		// qkv is one [3*dim, dim] weight in the checkpoint and three GEMMs
		// here: the bank's tiling is per weight, so three tensors is simpler
		// and exactly as fast as three offsets into one.
		qkv := &b.Attn.QKV
		for s, dst := range []*uint32{&blk.wQ, &blk.wK, &blk.wV} {
			part := Linear{In: g.dim, Out: g.dim,
				Weight: qkv.Weight[s*g.dim*g.dim : (s+1)*g.dim*g.dim]}
			if qkv.Bias != nil {
				part.Bias = qkv.Bias[s*g.dim : (s+1)*g.dim]
			}
			bank, off := place(g.dim, g.dim, part.Weight)
			*dst = off
			blk.bank = bank
			folded := fold(&part, b.Norm1.Bias, g.dim)
			switch s {
			case 0:
				blk.bQ = w32.put(folded)
			case 1:
				blk.bK = w32.put(folded)
			default:
				blk.bV = w32.put(folded)
			}
		}
		_, blk.wO = place(g.dim, g.dim, b.Attn.Proj.Weight)
		blk.bO = w32.put(b.Attn.Proj.Bias)

		fc1 := padWeight(b.MLP.FC1.Weight, g.ffn, g.dim, g.ffnPad, g.dim)
		_, blk.wFC1 = place(g.ffnPad, g.dim, fc1)
		blk.bFC1 = w32.put(fold(&b.MLP.FC1, b.Norm2.Bias, g.ffnPad))
		fc2 := padWeight(b.MLP.FC2.Weight, g.dim, g.ffn, g.dim, g.ffnPad)
		_, blk.wFC2 = place(g.dim, g.ffnPad, fc2)
		blk.bFC2 = w32.put(b.MLP.FC2.Bias)
	}

	// The four mergers: three deepstack taps and the output one.
	all := append(append([]Merger{}, cpu.Deepstack...), cpu.Merger)
	g.mergers = make([]gpuMerger, len(all))
	for i := range all {
		m := &all[i]
		gm := &g.mergers[i]
		gm.post = m.PostShuffleNorm
		gm.gamma = gamma(m.Norm.Weight)
		gm.act = 2 // both mergers use the exact erf GELU
		beta := m.Norm.Bias
		if !gm.post {
			beta = tile4(m.Norm.Bias, g.mergeIn)
		}
		bank, off := place(g.mergeIn, g.mergeIn, m.FC1.Weight)
		gm.bank, gm.wFC1 = bank, off
		gm.bFC1 = w32.put(fold(&m.FC1, beta, g.mergeIn))
		_, gm.wFC2 = place(g.mergeOut, g.mergeIn, m.FC2.Weight)
		gm.bFC2 = w32.put(m.FC2.Bias)
	}

	var err error
	if g.wbuf, err = g.dev.NewBuffer(len(w32.data) * 4); err != nil {
		return fmt.Errorf("vision: fp32 weight arena: %w", err)
	}
	g.wbuf.WriteFloat32(w32.data)
	for _, n := range bankElems {
		b, err := g.dev.NewBuffer(n * 2)
		if err != nil {
			return fmt.Errorf("vision: fp16 bank (%d MB): %w", (n*2)>>20, err)
		}
		g.banks = append(g.banks, b)
	}
	for _, s := range pending {
		buf := make([]uint16, s.n*s.k)
		packB(buf, s.w, s.n, s.k)
		g.banks[s.bank].WriteUint16At(int(s.off), buf)
	}
	g.patchBank, g.patchOff = pxBank, pxOff
	return nil
}

// maxBankBytes is the device's storage-buffer ceiling, as in qimage/dit.
const maxBankBytes = 0xfffffffc

// coopMatTile is the cooperative-matrix tile the GEMM's B operand is packed
// for, as in qimage/dit's packB.
const coopMatTile = 16

// packB packs a row-major [n, k] fp32 weight into the fp16 bank in the
// fragment order dit_gemm reads. It is qimage/dit's, and the two are the
// same function over the same kernel.
func packB(dst []uint16, w []float32, n, k int) {
	const tile = coopMatTile
	kt := k / tile
	for i := 0; i < n; i++ {
		row := w[i*k : (i+1)*k]
		base := (i / tile) * kt * tile * tile
		lane := (i % tile) * tile
		for j, v := range row {
			dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
		}
	}
}

func (g *GPU) allocActivations() error {
	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	// Every row-shaped tensor is allocated to the GEMM's M tile, because the
	// kernel writes whole tiles: the rows past the patch count are garbage
	// that nothing reads, but they are written.
	rows := g.rowsPad()
	mrg := g.mergedPad()
	g.aPix = alloc(rows * g.cfg.PatchElems())
	g.aPos = alloc(rows * g.dim)
	g.aX = alloc(rows * g.dim)
	g.aQ = alloc(rows * g.dim)
	g.aK = alloc(rows * g.dim)
	g.aV = alloc(rows * g.dim)
	g.aCtx = alloc(rows * g.dim)
	g.aY = alloc(rows * g.dim)
	// aH holds both the MLP's hidden rows and, during a merger, its 4608-wide
	// first layer over a quarter as many rows — which is the smaller of the
	// two at every grid this tower runs.
	g.aH = alloc(max(rows*g.ffnPad, mrg*g.mergeIn))
	g.aKT = alloc(g.heads * g.headDim * (g.rows + attnTokenPad))
	g.aCos = alloc(g.rows * g.headDim)
	g.aSin = alloc(g.rows * g.headDim)
	g.aGamma = alloc(len(g.gammas))
	g.aDeep = alloc(3 * mrg * g.mergeOut)
	g.aMerged = alloc(mrg * g.mergeOut)

	// One fp16 A operand, sized for the widest reduction any GEMM makes.
	kMax := g.cfg.PatchElems()
	for _, k := range []int{g.dim, g.ffnPad, g.mergeIn} {
		if k > kMax {
			kMax = k
		}
	}
	g.ldaA = kMax + ldaPad
	g.hA = 0
	g.hElems = rows * g.ldaA

	var err error
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("vision: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("vision: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zero the fp16 arena once: every A operand's pad columns are read by a
	// reduction that stops at K only if they are zero when K is padded.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	return nil
}

func (g *GPU) pipeline(name string, spirv []byte, bufs []*vk.Buffer) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("vision: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers: bufs, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
	})
	if err != nil {
		return fmt.Errorf("vision: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

func (g *GPU) build() error {
	base := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf}
	for name, spirv := range map[string][]byte{
		"norm":      shaders.DiTFinalNorm,
		"add":       shaders.DiTGateAdd,
		"narrow":    shaders.DiTScaleF16,
		"transpose": shaders.DiTTransposeK,
		"attention": shaders.DiTAttention,
		"biasact":   shaders.QViTBiasAct,
		"rope":      shaders.QViTRoPE,
	} {
		if err := g.pipeline(name, spirv, base); err != nil {
			return err
		}
	}
	for b := range g.banks {
		mod, err := g.dev.NewShaderModule(shaders.DiTGEMMReg64Tiled)
		if err != nil {
			return fmt.Errorf("vision: gemm bank %d: %w", b, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[b]},
			PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
		})
		if err != nil {
			return fmt.Errorf("vision: gemm pipeline bank %d: %w", b, err)
		}
		g.gemms = append(g.gemms, map[string]*vk.ComputePipeline{"gemm": pipe})
	}
	return nil
}

// rowsPad is the patch count rounded up to the GEMM's M tile: the kernel
// writes whole tiles, so every row-shaped activation is allocated to it.
//
// At construction g.rows is the budget, which is what the arenas are sized
// by; in a run it is the grid's, which is what packs the deepstack features
// and the transposed keys more tightly inside the same space.
func (g *GPU) rowsPad() int   { return (g.rows + gemmBM - 1) &^ (gemmBM - 1) }
func (g *GPU) mergedPad() int { return (g.merged + gemmBM - 1) &^ (gemmBM - 1) }

// MaxRows is the staged patch budget.
func (g *GPU) MaxRows() int { return g.maxRows }

// Forward runs the tower over one image's patches on the device.
//
// The geometry — the interpolated position grid and the axial rope tables —
// is computed on the host and uploaded, exactly as the DiT uploads its
// modulation: both are functions of the patch grid rather than of any
// weight, they cost microseconds, and reproducing a bilinear resample of a
// learned table in a shader would be a second place for it to be wrong.
func (g *GPU) Forward(pixels *qwen.Mat, gridH, gridW int) (*Output, error) {
	merge := g.cfg.SpatialMergeSize * g.cfg.SpatialMergeSize
	rows := gridH * gridW
	if gridH <= 0 || gridW <= 0 || rows%merge != 0 {
		return nil, fmt.Errorf("vision: a %dx%d grid does not group into %ds", gridH, gridW, merge)
	}
	if rows > g.maxRows {
		return nil, fmt.Errorf("vision: a %dx%d grid is %d patches, staged for %d",
			gridH, gridW, rows, g.maxRows)
	}
	if pixels.Rows != rows || pixels.Cols != g.cfg.PatchElems() {
		return nil, fmt.Errorf("vision: pixels are %s, want [%d %d]", pixels, rows, g.cfg.PatchElems())
	}
	g.gridH, g.gridW, g.rows, g.merged = gridH, gridW, rows, rows/merge
	pos, err := g.cpu.positionEmbedding(g.gridH, g.gridW)
	if err != nil {
		return nil, err
	}
	rope := NewRope(&g.cfg, g.gridH, g.gridW)
	g.abuf.WriteFloat32At(int(g.aPix), pixels.Data)
	g.abuf.WriteFloat32At(int(g.aPos), pos.Data)
	g.abuf.WriteFloat32At(int(g.aCos), rope.Cos.Data)
	g.abuf.WriteFloat32At(int(g.aSin), rope.Sin.Data)

	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{
			Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes(),
		})
	}
	gemm := func(bank int, aOff, cOff, bOff uint32, m, n, k, lda int) error {
		if n%gemmBN != 0 {
			return fmt.Errorf("vision: GEMM tile %d does not divide N=%d", gemmBN, n)
		}
		mPad := (m + gemmBM - 1) &^ (gemmBM - 1)
		pc := pushConstants{
			InOff: aOff, OutOff: cOff, BOff: bOff,
			GemmM: uint32(mPad), GemmN: uint32(n), GemmK: uint32(k), LDA: uint32(lda),
		}
		d = append(d, vk.MultiDispatch{
			Pipeline: g.gemms[bank]["gemm"], GroupsX: uint32(n / gemmBN), GroupsY: uint32(mPad / gemmBM),
			PushConstants: pc.bytes(),
		})
		return nil
	}
	// norm is the affine LayerNorm: dit_final_norm's LN(x)*gamma into the
	// fp16 arena, with the beta already folded into the next bias.
	norm := func(src uint32, tokens, dim int, gammaIdx uint32, lda int) {
		add("norm", uint32(tokens), 1, pushConstants{
			InOff: src, OutOff: g.hA, Tokens: uint32(tokens), Dim: uint32(dim),
			Aux0: g.aGamma + g.gammaOff[gammaIdx],
			Eps:  math.Float32bits(float32(normEps)), LDA: uint32(lda),
		})
	}
	narrow := func(src uint32, tokens, dim, lda int) {
		add("narrow", uint32(tokens), 1, pushConstants{
			InOff: src, OutOff: g.hA, Tokens: uint32(tokens), Dim: uint32(dim), LDA: uint32(lda),
		})
	}
	biasact := func(dst, bias uint32, tokens, dim int, act uint32) {
		add("biasact", uint32(tokens), 1, pushConstants{
			OutOff: dst, WOff: bias, Tokens: uint32(tokens), Dim: uint32(dim), Aux0: act,
		})
	}
	residual := func(y, x uint32, tokens, dim int) {
		add("add", uint32(tokens), 1, pushConstants{
			InOff: y, OutOff: x, Tokens: uint32(tokens), Dim: uint32(dim), Aux2: 0,
		})
	}

	// ---- the patch embedding, plus the interpolated position grid.
	narrow(g.aPix, g.rows, g.cfg.PatchElems(), g.ldaA)
	if err := gemm(g.patchBank, g.hA, g.aX, g.patchOff, g.rows, g.dim, g.cfg.PatchElems(), g.ldaA); err != nil {
		return nil, err
	}
	biasact(g.aX, g.wPatchBias, g.rows, g.dim, 0)
	residual(g.aPos, g.aX, g.rows, g.dim)

	// ---- the 27 blocks.
	kStride := g.rows + attnTokenPad
	scale := math.Float32bits(float32(1 / math.Sqrt(float64(g.headDim))))
	for i := range g.blocks {
		b := &g.blocks[i]
		norm(g.aX, g.rows, g.dim, b.gamma1, g.ldaA)
		for _, pr := range []struct {
			w, bias, out uint32
		}{{b.wQ, b.bQ, g.aQ}, {b.wK, b.bK, g.aK}, {b.wV, b.bV, g.aV}} {
			if err := gemm(b.bank, g.hA, pr.out, pr.w, g.rows, g.dim, g.dim, g.ldaA); err != nil {
				return nil, err
			}
			biasact(pr.out, pr.bias, g.rows, g.dim, 0)
		}
		for _, t := range []uint32{g.aQ, g.aK} {
			add("rope", uint32(g.rows), uint32(g.heads), pushConstants{
				InOff: t, Tokens: uint32(g.rows), Dim: uint32(g.dim),
				Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
				WOff: g.aCos, Aux0: g.aSin,
			})
		}
		add("transpose", uint32((g.rows*g.heads*g.headDim+255)/256), 1, pushConstants{
			InOff: g.aK, OutOff: g.aKT, Tokens: uint32(g.rows), Dim: uint32(g.dim),
			Heads: uint32(g.heads), HeadDim: uint32(g.headDim), KStride: uint32(kStride),
		})
		add("attention", uint32(g.rows), uint32(g.heads), pushConstants{
			InOff: g.aQ, OutOff: g.aCtx, KOff: g.aKT, VOff: g.aV,
			Tokens: uint32(g.rows), Dim: uint32(g.dim),
			Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
			KStride: uint32(kStride), Scale: scale,
		})
		narrow(g.aCtx, g.rows, g.dim, g.ldaA)
		if err := gemm(b.bank, g.hA, g.aY, b.wO, g.rows, g.dim, g.dim, g.ldaA); err != nil {
			return nil, err
		}
		biasact(g.aY, b.bO, g.rows, g.dim, 0)
		residual(g.aY, g.aX, g.rows, g.dim)

		norm(g.aX, g.rows, g.dim, b.gamma2, g.ldaA)
		if err := gemm(b.bank, g.hA, g.aH, b.wFC1, g.rows, g.ffnPad, g.dim, g.ldaA); err != nil {
			return nil, err
		}
		biasact(g.aH, b.bFC1, g.rows, g.ffnPad, 1)
		narrow(g.aH, g.rows, g.ffnPad, g.ldaA)
		if err := gemm(b.bank, g.hA, g.aY, b.wFC2, g.rows, g.dim, g.ffnPad, g.ldaA); err != nil {
			return nil, err
		}
		biasact(g.aY, b.bFC2, g.rows, g.dim, 0)
		residual(g.aY, g.aX, g.rows, g.dim)

		if b.deepstack >= 0 {
			dst := g.aDeep + uint32(b.deepstack*g.mergedPad()*g.mergeOut)
			if err := g.merger(&d, add, gemm, norm, narrow, biasact, b.deepstack, dst); err != nil {
				return nil, err
			}
		}
	}
	if err := g.merger(&d, add, gemm, norm, narrow, biasact, len(g.mergers)-1, g.aMerged); err != nil {
		return nil, err
	}

	if err := g.run(d); err != nil {
		return nil, err
	}
	out := &Output{
		Merged: g.read(g.aMerged, g.merged, g.mergeOut),
		Last:   g.read(g.aX, g.rows, g.dim),
	}
	for j := 0; j < len(g.mergers)-1; j++ {
		out.Deepstack = append(out.Deepstack,
			g.read(g.aDeep+uint32(j*g.mergedPad()*g.mergeOut), g.merged, g.mergeOut))
	}
	return out, nil
}

// merger records one patch merger. The concatenation of four 1152-wide rows
// into one 4608-wide row is free in a row-major buffer, so the only
// difference between the output merger and the three deepstack ones is
// *where the norm runs*: over each 1152 row with a tight destination stride,
// or over the 4608 the four of them already form.
func (g *GPU) merger(
	d *[]vk.MultiDispatch,
	add func(string, uint32, uint32, pushConstants),
	gemm func(int, uint32, uint32, uint32, int, int, int, int) error,
	norm func(uint32, int, int, uint32, int),
	narrow func(uint32, int, int, int),
	biasact func(uint32, uint32, int, int, uint32),
	idx int, dst uint32,
) error {
	m := &g.mergers[idx]
	lda := g.ldaA
	if m.post {
		norm(g.aX, g.merged, g.mergeIn, m.gamma, lda)
	} else {
		// A tight stride, so that four normalised rows of 1152 sit
		// contiguously and the GEMM reads them as one row of 4608.
		lda = g.mergeIn
		norm(g.aX, g.rows, g.dim, m.gamma, g.dim)
	}
	if err := gemm(m.bank, g.hA, g.aH, m.wFC1, g.merged, g.mergeIn, g.mergeIn, lda); err != nil {
		return err
	}
	biasact(g.aH, m.bFC1, g.merged, g.mergeIn, m.act)
	narrow(g.aH, g.merged, g.mergeIn, g.ldaA)
	if err := gemm(m.bank, g.hA, dst, m.wFC2, g.merged, g.mergeOut, g.mergeIn, g.ldaA); err != nil {
		return err
	}
	biasact(dst, m.bFC2, g.merged, g.mergeOut, 0)
	return nil
}

// dispatchesPerSubmit caps how much work goes into one command buffer, for
// the reason qimage/vae gives: the whole graph in a single submit trips the
// driver's reset watchdog.
const dispatchesPerSubmit = 8

func (g *GPU) run(ds []vk.MultiDispatch) error {
	for i := 0; i < len(ds); i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, len(ds))
		if _, err := vk.DispatchMultiTimed(ds[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("vision: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

func (g *GPU) read(off uint32, rows, cols int) *qwen.Mat {
	out := qwen.NewMat(rows, cols)
	copy(out.Data, g.abuf.ReadFloat32At(int(off), rows*cols))
	return out
}
