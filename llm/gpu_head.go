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

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	nEmbd, vocab            int
	tokens, arenaRows, rows int
	lda                     int
	// q8 is whether the bank is L8's int8-plus-scales or the fp16 tiling
	// every block used before it. It picks the pipeline build, the bank size
	// and what stage() writes, and nothing downstream of the dispatch.
	q8 bool

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
	g := &HeadGPU{
		dev: dev, pipes: make(map[string]*vk.ComputePipeline),
		nEmbd: nEmbd, vocab: vocab, q8: q8,
		tokens: maxRows, rows: maxRows,
		lda:      nEmbd + gemmPad,
		gemm:     OutGEMMKernelFor(maxRows),
		autoPlan: true,
	}
	align := 1
	for _, v := range gemmBuilds(q8) {
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

	bank := g.vocab * g.nEmbd * 2
	if g.q8 {
		bank = q8Bytes(g.vocab, g.nEmbd)
	}
	if g.bank, err = g.dev.NewBuffer(bank); err != nil {
		return fmt.Errorf("llm: head weight bank (%d MB): %w", bank>>20, err)
	}
	return nil
}

func (g *HeadGPU) build() error {
	// The Q8 build names a sixth buffer: the bank again, as raw words, for
	// the tiles the scale plane at binding 3 belongs to (llm_common.glsl).
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank, g.abuf}
	if g.q8 {
		bufs = append(bufs, g.bank)
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
	for _, v := range gemmBuilds(g.q8) {
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
		name := string(v.name)
		if g.q8 {
			name = q8Pipe(v.name)
		}
		g.pipes[name] = pipe
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
	if g.q8 {
		qs = make([]byte, headStageRows*g.nEmbd)
		sc = make([]uint16, headStageRows*g.nEmbd/q8Group)
	}
	planeOff := g.vocab * g.nEmbd
	for r0 := 0; r0 < g.vocab; r0 += headStageRows {
		n := minInt(headStageRows, g.vocab-r0)
		src, err := gguf.Dequantize(w.Type, w.Data[r0*rowBytes:(r0+n)*rowBytes],
			int64(n*g.nEmbd), f32[:0])
		if err != nil {
			return fmt.Errorf("llm: head rows %d-%d: %w", r0, r0+n, err)
		}
		if g.q8 {
			tileBQ8(qs[:n*g.nEmbd], sc[:n*g.nEmbd/q8Group], src, n, g.nEmbd,
				func(i int) int { return i })
			g.bank.WriteBytesAt(r0*g.nEmbd, qs[:n*g.nEmbd])
			g.bank.WriteUint16At(planeOff/2+r0*g.nEmbd/q8Group, sc[:n*g.nEmbd/q8Group])
			continue
		}
		tileB(half[:n*g.nEmbd], src, n, g.nEmbd, func(i int) int { return i })
		g.bank.WriteUint16At(r0*g.nEmbd, half[:n*g.nEmbd])
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

// WeightBytes is the staged bank; ActivationBytes the two arenas.
func (g *HeadGPU) WeightBytes() int     { return g.bank.Size() }
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
	pipe := string(g.gemm)
	if g.q8 {
		pipe = q8Pipe(g.gemm)
	}
	return []vk.MultiDispatch{{
		Pipeline: g.pipes[pipe],
		GroupsX:  uint32(g.vocab / headBN), GroupsY: uint32(m / v.bm),
		PushConstants: pc.bytes(),
	}}, []string{"head"}
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
	for _, b := range []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank} {
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
	g.rows = rows
	if g.autoPlan {
		g.gemm = OutGEMMKernelFor(rows)
	}
	return nil
}
