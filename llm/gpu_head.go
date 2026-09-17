package llm

// The language-model head (LLM.md L6b): the one tensor of this model that no
// block owns.
//
// `output.weight` is [2560, 248320] Q8_0 — 675 MB in the checkpoint, 1.27 GB
// as the halves the matrix cores want — and it is the only weight in the
// vertical that is neither a mixer, a layer nor an expert. **L8 stops it
// being halves**: staged as int8 with one fp16 scale per 32 elements it is
// 0.68 GB, which is the checkpoint's own width, and the values the kernel
// multiplies are bit-identical (bank.go). It is this vertical's first Q8 bank
// because it is its simplest: one matrix, one dispatch, no fusion, and a
// tensor that is Q8_0 all the way through. It is also the
// only matmul here with nothing fused onto it: no norm before it (the final
// hyper-connection mixer *is* the output norm, which is why this checkpoint
// has no `output_norm.weight`), no epilogue after it. So it is llm_gemm.comp's
// MODE 2 — the plain rung the PLE block's key/value projection already uses —
// over a bank of its own.
//
// It runs on **one row**, not on the prompt. llama.cpp's graph ends with a
// `ggml_get_rows(inp_out_ids)` before the final mixer, so prefill computes
// `result_norm` and `result_output` for the last token alone; the 7-token
// trace's `0142_result_output.bin` is one row of 248320 floats and not seven.
// Doing anything else would be 508 MB of logits a graph for 511 rows nothing
// reads, and it would not be the reference's arithmetic to compare against.

import (
	"fmt"
	"unsafe"

	"time"

	"strix-halo-vulkan/gguf"
	"strix-halo-vulkan/vk"
)

// headBN is the plain GEMM's column block: WN=4 sixteen-wide tiles. The
// vocabulary has to be a whole number of them, which 248320 = 3880*64 is.
const headBN = attnBN

// headStageRows is how many rows of the head are dequantised at a time.
//
// The tensor is 675 MB of Q8_0 and 2.54 GB as f32, and there is no reason to
// hold the second of those: a slab of rows dequantises, narrows and tiles
// into exactly the contiguous slab of the bank those rows occupy, because
// the fragment tiling's row blocks are sixteen rows wide and this is a
// multiple of sixteen.
const headStageRows = 4096

// HeadGPU is the output projection: one fp16 bank, one fp16 A operand, one
// f32 logit arena.
type HeadGPU struct {
	// rec, when set, collects this block's dispatches into the pass's one
	// command buffer instead of submitting them (record.go).
	rec *recorder
	dev *vk.Device

	wbuf, abuf, hbuf, wbank *vk.Buffer
	pipes                   map[string]*vk.ComputePipeline
	mods                    []*vk.ShaderModule

	nEmbd, vocab            int
	tokens, arenaRows, rows int
	lda                     int
	// bank is which width `output.weight` is staged in: the fp16 tiling every
	// block used before L8, L8a's int8-plus-scales, or L8c-4's 4.5-bit
	// K-quant. It picks the pipeline build, the bank size and what stage()
	// writes, and nothing downstream of the dispatch.
	bank DenseBank
	// sim is the format the 4.5-bit bank quantises to — `q4_k/32` and a mode
	// — and is unread on the other two. It is a QuantSim because that is what
	// L8c-3 measured the format with, and both go through one encoder
	// (quantk.go), which is what makes "the bank is what was simulated" a
	// fact rather than a claim.
	sim QuantSim
	// gemv is the decode rung, or GEMVOff for llm_gemm.comp MODE 2. The head
	// is the one block L8d and L8e left on the GEMM — 194 GB/s on a 242 GB/s
	// bus at one row — so this is off by default and exists to price the
	// narrower bank's GEMV against a matrix big enough to fill the machine
	// from N alone.
	gemv GEMVKernel

	hXn  uint32
	aOut uint32

	gemm     GEMMKernel
	autoPlan bool
}

// NewHeadGPU stages `output.weight` in the fragment tiling, as L8's int8 and
// scales or as the halves that preceded them.
//
// maxRows is how many rows of logits the arena holds. One is what a prefill
// needs and what the reference computes; a caller that wants more is asking
// for 0.99 MB of f32 per row and says so here rather than at a run.
func NewHeadGPU(dev *vk.Device, nEmbd int, w *gguf.Tensor, maxRows int, q8 bool) (*HeadGPU, error) {
	return NewHeadGPUBank(dev, nEmbd, w, maxRows, bankOf(q8), QuantSim{})
}

// NewHeadGPUBank is the same, with the bank named rather than implied — and
// with the format, for the one bank that has one to choose.
func NewHeadGPUBank(dev *vk.Device, nEmbd int, w *gguf.Tensor, maxRows int,
	bank DenseBank, sim QuantSim) (*HeadGPU, error) {

	if maxRows <= 0 {
		return nil, fmt.Errorf("llm: head maxRows is %d", maxRows)
	}
	if len(w.Dims) != 2 || int(w.Dims[0]) != nEmbd {
		return nil, fmt.Errorf("llm: %s is %v, want [%d, vocab]", w.Name, w.Dims, nEmbd)
	}
	vocab := int(w.Dims[1])
	if vocab%headBN != 0 {
		return nil, fmt.Errorf("llm: vocabulary %d is not a whole %d-column block", vocab, headBN)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	if bank == BankQ4K {
		if err := q4kFits(vocab, nEmbd); err != nil {
			return nil, err
		}
		if sim.Off() {
			return nil, fmt.Errorf("llm: a q4_k head bank needs a format to quantise to")
		}
	}
	g := &HeadGPU{
		dev: dev, pipes: make(map[string]*vk.ComputePipeline),
		nEmbd: nEmbd, vocab: vocab, bank: bank, sim: sim,
		tokens: maxRows, rows: maxRows,
		lda:      nEmbd + gemmPad,
		gemm:     OutGEMMKernelFor(maxRows),
		gemv:     GEMVOff,
		autoPlan: true,
	}
	align := 1
	for _, v := range gemmBuildsFor(bank) {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxRows, align)
	if err := g.alloc(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(w); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

func (g *HeadGPU) alloc() error {
	var err error
	// Nothing in this block reads an fp32 weight, but the descriptor set is
	// the vertical's and binding 0 has to be something.
	if g.wbuf, err = g.dev.NewBuffer(64); err != nil {
		return fmt.Errorf("llm: head fp32 weight arena: %w", err)
	}
	g.aOut = 0
	if g.abuf, err = newArena(g.dev, g.arenaRows*g.vocab*4); err != nil {
		return fmt.Errorf("llm: head logit arena (%d MB): %w", (g.arenaRows*g.vocab*4)>>20, err)
	}
	g.hXn = 0
	if g.hbuf, err = newArena(g.dev, g.arenaRows*g.lda*2); err != nil {
		return fmt.Errorf("llm: head fp16 activation arena: %w", err)
	}
	// Zeroed once: the A operand's pad columns and a short run's pad rows,
	// which the kernel has no bounds check for.
	g.hbuf.Zero()
	g.abuf.Zero()

	size := g.vocab * g.nEmbd * 2
	switch g.bank {
	case BankQ8:
		size = q8Bytes(g.vocab, g.nEmbd)
	case BankQ4K:
		size = q4kBytes(g.vocab, g.nEmbd)
	}
	if g.wbank, err = g.dev.NewBuffer(size); err != nil {
		return fmt.Errorf("llm: head weight bank (%d MB): %w", size>>20, err)
	}
	return nil
}

func (g *HeadGPU) build() error {
	// The Q8 build names a sixth buffer: the bank again, as raw words, for
	// the tiles the scale plane at binding 3 belongs to (llm_common.glsl).
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.wbank, g.abuf}
	if g.bank != BankFP16 {
		bufs = append(bufs, g.wbank)
	}
	pcSize := uint32(unsafe.Sizeof(push{}))
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
		return fmt.Errorf("llm: the GEMM rungs need a pinned 64-wide subgroup")
	}
	for _, v := range gemmBuildsFor(g.bank) {
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("llm: shader head.%s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		})
		if err != nil {
			return fmt.Errorf("llm: pipeline head.%s: %w", v.name, err)
		}
		g.pipes[bankPipe(g.bank, v.name)] = pipe
	}
	// The decode rungs, on the quantised banks only: `llm_gemv.comp` has an
	// arm for each of them and none for the halves this block also stages.
	// KSLABS = 1 alone, because at N = 248320 the grid is 15520 workgroups
	// from the output width by itself — there is nothing for a split of K to
	// buy, and no partial-sum arena to pay for (D11).
	if g.bank != BankFP16 {
		spv, ok := gemvSPIRV[gemvBankPipe(GEMVK1, g.bank)]
		if !ok {
			return fmt.Errorf("llm: no GEMV build for the %s bank", g.bank)
		}
		mod, err := g.dev.NewShaderModule(spv)
		if err != nil {
			return fmt.Errorf("llm: shader head.gemv: %w", err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		})
		if err != nil {
			return fmt.Errorf("llm: pipeline head.gemv: %w", err)
		}
		g.pipes[gemvBankPipe(GEMVK1, g.bank)] = pipe
	}
	return nil
}

// stage dequantises the head a slab of rows at a time and tiles each slab
// straight into the bank.
//
// The slab boundary is a multiple of the fragment tiling's sixteen rows, so
// rows [r0, r0+n) of the matrix are halves [r0*nEmbd, (r0+n)*nEmbd) of the
// bank and the tiling of a slab is the tiling of the whole thing restricted
// to it — which is what keeps a 2.54 GB f32 copy of the head from ever
// existing.
func (g *HeadGPU) stage(w *gguf.Tensor) error {
	if headStageRows%coopMatTile != 0 {
		return fmt.Errorf("llm: head slab of %d rows is not whole %d-row tiles", headStageRows, coopMatTile)
	}
	rowBytes := int(w.RowBytes())
	// gguf.Dequantize *appends*, so the slab buffer is handed over with zero
	// length and its capacity is what keeps it from reallocating.
	f32 := make([]float32, 0, headStageRows*g.nEmbd)
	half := make([]uint16, headStageRows*g.nEmbd)
	// The Q8 bank's two planes. A slab's tiles are bytes [r0*nEmbd, ...) and
	// its scales are halves [r0*nEmbd/32, ...) of the plane behind them —
	// both contiguous, because the slab is a whole number of sixteen-row
	// tiles and the plane is k-major inside one.
	var qs []byte
	var sc []uint16
	if g.bank == BankQ8 {
		qs = make([]byte, headStageRows*g.nEmbd)
		sc = make([]uint16, headStageRows*g.nEmbd/q8Group)
	}
	// The 4.5-bit bank's two, on the same argument: a slab's nibbles are
	// bytes [r0*nEmbd/2, ...) and its records are [r0/16 * nsb * 256, ...),
	// both contiguous for the same reason.
	var q4 []byte
	var rec []byte
	var qw []float32
	nsb := g.nEmbd / (q4kSuper * 32)
	if !g.sim.Off() {
		var err error
		// The published matrix covers `blk.N.*` only, so the head has no
		// calibration data and this is nil — which is what makes `lm_head`
		// the one family L8c-3 measured at rtn on both forms. It is looked
		// up rather than assumed, because a bank that silently skipped
		// calibration is the bug sim.go's tally exists to catch.
		if qw, err = imatrixCols(w.Name, g.sim.Mode); err != nil {
			return err
		}
		if qw == nil && g.sim.Mode == "imatrix" {
			// ggml's own fallback: `quantize_row_q4_K_impl` returns
			// `quantize_row_q4_K_ref` on a null `quant_weights`, so an
			// uncovered tensor is round-to-nearest and not the *search* with
			// unit weights — a different arm, and one L8c-3 measured as
			// worse than either. sim.go's `ApplyTo` does the same thing for
			// the same reason.
			g.sim.Mode = "rtn"
		}
	}
	if g.bank == BankQ4K {
		q4 = make([]byte, headStageRows*g.nEmbd/2)
		rec = make([]byte, headStageRows/coopMatTile*nsb*coopMatTile*q4kRecord)
	}
	planeOff := g.vocab * g.nEmbd
	for r0 := 0; r0 < g.vocab; r0 += headStageRows {
		n := minInt(headStageRows, g.vocab-r0)
		src, err := gguf.Dequantize(w.Type, w.Data[r0*rowBytes:(r0+n)*rowBytes],
			int64(n*g.nEmbd), f32[:0])
		if err != nil {
			return fmt.Errorf("llm: head rows %d-%d: %w", r0, r0+n, err)
		}
		// L8c's width simulation. The head is the one dense weight that does
		// not reach the device through Model.F32 — it is 2.54 GB as floats
		// and is dequantised a slab at a time here so that no whole f32 copy
		// ever exists — so the hook has to be repeated rather than inherited.
		// A slab is whole rows, so the grouping along k is the same as it
		// would be for the matrix entire.
		if err := DensePlan().ApplyTo(w.Name, w.Type == gguf.Q8_0, src, g.nEmbd); err != nil {
			return fmt.Errorf("llm: head rows %d-%d: %w", r0, r0+n, err)
		}
		// **The simulation of a bank, on the bank it is a simulation of.**
		// A format given to the fp16 arm is L8c-1's round trip done
		// explicitly rather than through the environment, and it is what
		// lets `cmd/llm -head` hold the simulated q4_k and the real one in
		// one process and compare their logits element for element. On the
		// q4_k bank itself it would be the same quantisation applied twice,
		// so it is the fp16 arm's alone.
		if g.bank == BankFP16 && !g.sim.Off() {
			if err := g.sim.ApplyWeighted(src, g.nEmbd, qw); err != nil {
				return fmt.Errorf("llm: head rows %d-%d: %w", r0, r0+n, err)
			}
		}
		switch g.bank {
		case BankQ8:
			tileBQ8(qs[:n*g.nEmbd], sc[:n*g.nEmbd/q8Group], src, n, g.nEmbd,
				func(i int) int { return i })
			g.wbank.WriteBytesAt(r0*g.nEmbd, qs[:n*g.nEmbd])
			g.wbank.WriteUint16At(planeOff/2+r0*g.nEmbd/q8Group, sc[:n*g.nEmbd/q8Group])
		case BankQ4K:
			nr := n / coopMatTile * nsb * coopMatTile * q4kRecord
			if err := tileBQ4K(q4[:n*g.nEmbd/2], rec[:nr], src, n, g.nEmbd,
				func(i int) int { return i }, g.sim, qw); err != nil {
				return fmt.Errorf("llm: head rows %d-%d: %w", r0, r0+n, err)
			}
			g.wbank.WriteBytesAt(r0*g.nEmbd/2, q4[:n*g.nEmbd/2])
			g.wbank.WriteBytesAt(g.vocab*g.nEmbd/2+r0/coopMatTile*nsb*coopMatTile*q4kRecord, rec[:nr])
		default:
			tileB(half[:n*g.nEmbd], src, n, g.nEmbd, func(i int) int { return i })
			g.wbank.WriteUint16At(r0*g.nEmbd, half[:n*g.nEmbd])
		}
	}
	return nil
}

// SetKernel names the rung, and turns off the per-run choice.
func (g *HeadGPU) SetKernel(k GEMMKernel) error {
	if _, ok := gemmVariantFor(k); !ok {
		return fmt.Errorf("llm: no GEMM kernel %q (have %v)", k, GEMMKernels())
	}
	g.gemm, g.autoPlan = k, false
	return nil
}

// Kernel reports the rung in use.
func (g *HeadGPU) Kernel() GEMMKernel { return g.gemm }

// Vocab is the head's output width.
func (g *HeadGPU) Vocab() int { return g.vocab }

// MaxRows is how many rows of logits the arena holds — what NewHeadGPU was
// given, and the slab a caller that wants more than one row works in.
func (g *HeadGPU) MaxRows() int { return g.tokens }

// WeightBytes is the staged bank; ActivationBytes the two arenas.
func (g *HeadGPU) WeightBytes() int     { return g.wbank.Size() }
func (g *HeadGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }
func (g *HeadGPU) Buffers() int         { return 4 }

// Upload writes the rows to project: `result_norm`, [rows][nEmbd], narrowed
// into the GEMM's A layout.
func (g *HeadGPU) Upload(x []float32, rows int) error {
	if rows <= 0 || rows > g.tokens {
		return fmt.Errorf("llm: %d rows, the head arena is built for %d", rows, g.tokens)
	}
	if len(x) != rows*g.nEmbd {
		return fmt.Errorf("llm: head input is %d values, want %d", len(x), rows*g.nEmbd)
	}
	if g.gemv != GEMVOff && rows != 1 {
		return fmt.Errorf("llm: the GEMV reads one row, not %d (D15)", rows)
	}
	g.rows = rows
	if g.autoPlan {
		g.gemm = OutGEMMKernelFor(rows)
	}
	slab := make([]uint16, rows*g.lda)
	narrowRows(slab, x, rows, g.nEmbd, g.lda)
	g.hbuf.WriteUint16At(int(g.hXn), slab)
	return nil
}

// graph is the one dispatch.
func (g *HeadGPU) graph() ([]vk.MultiDispatch, []string) {
	v, _ := gemmVariantFor(g.gemm)
	m := roundUpInt(g.rows, v.bm)
	pc := push{
		Tokens: uint32(g.rows), NEmbd: uint32(g.nEmbd),
		XnOff: g.hXn, LDA: uint32(g.lda), OutOff: g.aOut, BOff: 0,
		GemmM: uint32(m), GemmN: uint32(g.vocab), GemmK: uint32(g.nEmbd),
		// No fp16 tail: `output.weight` is Q8_0 for all 248320 rows. The Q8
		// arm reads these two whatever the bank is (llm_common.glsl), so the
		// fp16 build gets them too rather than leaving a field that means
		// something only half the time.
		LowRank: uint32(g.vocab), GateOff: noW,
	}
	if g.gemv != GEMVOff {
		// The decode kernel (D15): one workgroup per sixteen output columns,
		// no row block at all, and at KSLABS = 1 no partial sums to reduce.
		// It reads one row of A, so the host refuses it above one — dropping
		// rows silently is the failure mode this rule exists for.
		return []vk.MultiDispatch{{
			Pipeline: g.pipes[gemvBankPipe(g.gemv, g.bank)],
			GroupsX:  1, GroupsY: uint32(g.vocab / coopMatTile),
			PushConstants: pc.bytes(),
		}}, []string{"head_gemv"}
	}
	return []vk.MultiDispatch{{
		Pipeline: g.pipes[bankPipe(g.bank, g.gemm)],
		GroupsX:  uint32(g.vocab / headBN), GroupsY: uint32(m / v.bm),
		PushConstants: pc.bytes(),
	}}, []string{"head"}
}

// SetGEMV puts the projection on `llm_gemv.comp` at one row, or GEMVOff back
// on the GEMM. Only KSLABS = 1 is built here (see build), and only on a
// quantised bank.
func (g *HeadGPU) SetGEMV(k GEMVKernel) error {
	if k == GEMVOff {
		g.gemv = k
		return nil
	}
	if g.bank == BankFP16 {
		return fmt.Errorf("llm: the GEMV has no fp16 arm")
	}
	if k != GEMVK1 {
		return fmt.Errorf("llm: the head builds GEMV rung %s only, not %s", GEMVK1, k)
	}
	if g.rows != 1 {
		return fmt.Errorf("llm: the GEMV reads one row, not %d", g.rows)
	}
	g.gemv = k
	return nil
}

// Run projects whatever Upload left in the A operand.
func (g *HeadGPU) Run() error {
	d, kinds := g.graph()
	if g.rec.add(ownHead, d) {
		return nil
	}
	if _, err := vk.DispatchMultiTimed(d, 1, 1, true); err != nil {
		return fmt.Errorf("llm: head dispatch (%s): %w", kinds[0], err)
	}
	return nil
}

// Profile times the dispatch on the GPU over iters back-to-back repetitions.
func (g *HeadGPU) Profile(iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	d, kinds := g.graph()
	el, err := vk.DispatchMultiTimed(d, 1, uint32(iters), true)
	if err != nil {
		return nil, fmt.Errorf("llm: head profile: %w", err)
	}
	return []Stage{{Kind: kinds[0], GPU: el / time.Duration(iters)}}, nil
}

// Logits is the run's output, [rows][vocab].
func (g *HeadGPU) Logits() []float32 {
	return g.abuf.ReadFloat32At(int(g.aOut), g.rows*g.vocab)
}

// Destroy releases every Vulkan object.
func (g *HeadGPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.wbank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// InPort is `result_norm` as the projection's A operand wants it: fp16
// [rows][lda], one row at prefill.
func (g *HeadGPU) InPort() Port {
	return Port{Buf: g.hbuf, Off: g.hXn, Stride: g.lda, Width: g.nEmbd,
		Rows: g.arenaRows, Half: true}
}

// Resize sets how many rows the projection covers, without writing them.
func (g *HeadGPU) Resize(rows int) error {
	if rows <= 0 || rows > g.tokens {
		return fmt.Errorf("llm: %d rows, the head arena is built for %d", rows, g.tokens)
	}
	if g.gemv != GEMVOff && rows != 1 {
		return fmt.Errorf("llm: the GEMV reads one row, not %d (D15)", rows)
	}
	g.rows = rows
	if g.autoPlan {
		g.gemm = OutGEMMKernelFor(rows)
	}
	return nil
}
