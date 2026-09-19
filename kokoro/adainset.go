package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// This file is the part of StyleTTS2 that is the same wherever it appears:
// `AdainResBlk1d`, the style-conditioned residual block, staged on the device.
//
// Two objects run stacks of it and they do not look alike from the outside.
// `GPUDecoder` is the vocoder's four blocks at 514, 1090, 1024 and 512
// channels, each of them reading its own output concatenated with 66 side
// channels (T4c). `GPUProsody` is the predictor's two parallel stacks of
// three at 512 and 256, with no side channels at all and a 1-wide projection
// on the end (T6b). What they share is every kernel and every arena rule —
// the padded row stride, the bordered fp16 operand, the two-phase AdaIN
// reduction — so the block machinery lives here and each owner supplies only
// the plumbing: where a block's input comes from and where its output goes.

// chanPad is what a channel count is rounded up to before it becomes a row
// stride on the device. It has to be a multiple of dit_gemm.comp's BK — the
// K-slab depth, BK_TILES*16 = 64 in every rung built here — because A_CONV=2
// carries the tap index across the K loop and rolls it over when the column
// offset reaches the padded width. Pad to less and a slab would straddle two
// taps; pad to a power of two instead and the decoder's 1090 channels would
// become 2048, nearly doubling the arithmetic.
const chanPad = 64

// decChunks is how many frame chunks an AdaIN reduction is split over.
// The generator's 64 is sized for a 15601-frame tensor; the blocks here run
// over 130 and 260, so eight chunks of 17 or 33 frames is already more
// workgroups than the machine needs at 1090 channels (five channel bands
// each).
const decChunks = 8

// dconvVariants is the convolution ladder these blocks run on: the same four
// rungs the generator sweeps, built with A_CONV=2.
//
// Which one wins is measured rather than carried over, because the shape is
// the other way up. The generator's convolutions are M in the thousands by
// N = 128; these are M = 130 or 260 by N = 256, 512 or 1024, with a K of
// 3*1152 — short, wide, and deep, which is the regime where a wave's share of
// the output is small and the K loop is everything.
var dconvVariants = []convVariant{
	{name: ConvReg32x32W32, spirv: shaders.KokoroDConvReg32x32W32, bm: 32, bn: 32, wave: 32},
	{name: ConvReg32x64W32, spirv: shaders.KokoroDConvReg32x64W32, bm: 32, bn: 64, wave: 32},
	{name: ConvReg64, spirv: shaders.KokoroDConvReg64, bm: 64, bn: 64},
	{name: ConvReg32x128, spirv: shaders.KokoroDConvReg32x128, bm: 32, bn: 128},
}

// DefaultDecoderKernel is the rung TestGPUDecoderLadder measures as the
// winner.
const DefaultDecoderKernel = ConvReg32x32W32

func dconvVariantFor(k ConvKernel) (convVariant, bool) {
	for _, v := range dconvVariants {
		if v.name == k {
			return v, true
		}
	}
	return convVariant{}, false
}

// decBlock is one AdainResBlk1d staged on the device.
//
// The shapes are per block because no stack of them is uniform: the decoder's
// first block is 514 channels wide and its last doubles its frame count, and
// the predictor's middle block does both at once. Nearly every offset here is
// a property of the block rather than of the set.
type decBlock struct {
	in, out    int // channel counts, unpadded
	inPad      int // in, rounded up to chanPad: the fp16 row stride
	frames     int // input frames
	conv       int // frames reaching conv1 — 2*frames when the block upsamples
	upsample   bool
	noSC       bool   // the shortcut is the block's input, unconvolved
	srcOff     uint32 // [frames, srcLDA] fp32 input
	srcLDA     uint32
	dstOff     uint32 // [conv, dstLDA] fp32 output
	dstLDA     uint32
	hSrc, hMid uint32 // fp16 A operands: the normalisation's and conv2's
	hPool      uint32 // the pool's output, when the block upsamples
	hSc        uint32 // the shortcut's narrow
	w1, w2     uint32 // fragment-tiled convolution weights
	w1x1       uint32
	pool       uint32 // [in, 3] fp32 depthwise weights
	poolBias   uint32
	bias       uint32 // conv2's bias plus conv1x1's, summed at staging
	gb1, gb2   uint32 // gamma then beta, written per utterance
	fc1, fc2   *Linear
}

// hArenas hands out bordered fp16 sub-arenas, one per (use, rows, stride).
//
// Each carries a zero border at both ends so the convolutions stay
// branchless, and each is written only in rows [0, rows) thereafter — so two
// blocks of the same shape can share one and the border survives. `use`
// distinguishes two sub-arenas of the same shape that are live at the same
// moment: a block's normalisation and its shortcut are both a narrow of a
// [frames, in] tensor, and they are different tensors.
type hArenas struct {
	off uint32
	at  map[[3]int]uint32
}

func (h *hArenas) take(use, rows, lda int) uint32 {
	if h.at == nil {
		h.at = map[[3]int]uint32{}
	}
	key := [3]int{use, rows, lda}
	if o, ok := h.at[key]; ok {
		return o
	}
	o := h.off + uint32(convBorder*lda)
	h.off += uint32((2*convBorder + roundUp(rows, framePad)) * lda)
	h.at[key] = o
	return o
}

// blockSet is a stack of AdainResBlk1d on the device: the pipelines they run
// on, the fp16 operand arena, the weight bank, and the scratch the AdaIN
// reduction lands in.
//
// It owns no input and no output. The embedding object fills in each block's
// srcOff/dstOff before calling allocBlocks, and everything after that —
// staging, the style, the dispatch graph — is the same for every stack.
type blockSet struct {
	dev    *vk.Device
	kernel ConvKernel
	blocks []decBlock

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline
	convs                  map[ConvKernel]*vk.ComputePipeline

	// Scratch in the fp32 arena, shared by every block: conv1's output, the
	// residual branch, the shortcut branch, and the AdaIN reduction's two
	// stages. The owner reserves these because it owns the arena.
	aMid, aR, aSc uint32
	aStat, aAff   uint32

	// Room at the end of the fp32 weight buffer for weights that belong to
	// the stack rather than to any block — the predictor's two 1-wide
	// projections. The owner sets extraW before allocBlocks and reads wExtra
	// after it.
	extraW int
	wExtra uint32

	hElems int
}

// Kernel is the rung the convolutions run on.
func (s *blockSet) Kernel() ConvKernel { return s.kernel }

// SetKernel switches rungs, which needs no restaging: the weights are
// fragment tiles and every rung reads those.
func (s *blockSet) SetKernel(k ConvKernel) error {
	if _, ok := s.convs[k]; !ok {
		return fmt.Errorf("kokoro: no block pipeline for %q", k)
	}
	s.kernel = k
	return nil
}

// allocBlocks lays out the fp16 arena and the weight bank over blocks whose
// shapes and fp32 wiring the caller has already filled in, and allocates the
// three buffers that hold them.
func (s *blockSet) allocBlocks() error {
	var h hArenas
	for i := range s.blocks {
		db := &s.blocks[i]
		db.hSrc = h.take(0, db.frames, db.inPad)
		db.hMid = h.take(1, db.conv, db.out)
		if db.noSC {
			// The shortcut is the block's own input, read where it already
			// is: no narrow, no convolution, no fp16 operand.
			db.hSc = noW
		} else {
			// A block's shortcut reads its *input*, not its normalisation, so
			// it needs a narrow of its own — at the doubled frame count when
			// the block upsamples, since the repeat is folded into that pass.
			db.hSc = h.take(2, db.conv, db.inPad)
		}
		if db.upsample {
			db.hPool = h.take(3, db.conv, db.inPad)
		}
	}
	s.hElems = int(h.off)

	var offW uint32
	for i := range s.blocks {
		db := &s.blocks[i]
		db.gb1 = offW
		offW += uint32(2 * db.in)
		db.gb2 = offW
		offW += uint32(2 * db.out)
		db.bias = offW
		offW += uint32(db.out)
		if db.upsample {
			db.pool = offW
			offW += uint32(3 * db.in)
			db.poolBias = offW
			offW += uint32(db.in)
		} else {
			db.pool, db.poolBias = noW, noW
		}
	}

	var halves int
	for i := range s.blocks {
		db := &s.blocks[i]
		db.w1 = uint32(halves)
		halves += db.out * 3 * db.inPad
		db.w2 = uint32(halves)
		halves += db.out * 3 * db.out
		if db.noSC {
			db.w1x1 = noW
			continue
		}
		db.w1x1 = uint32(halves)
		halves += db.out * db.inPad
	}

	s.wExtra = offW
	offW += uint32(s.extraW)

	var err error
	if s.hbuf, err = s.dev.NewBuffer(s.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: block fp16 arena: %w", err)
	}
	if s.wbuf, err = s.dev.NewBuffer(int(offW) * 4); err != nil {
		return fmt.Errorf("kokoro: block fp32 weights: %w", err)
	}
	if s.bank, err = s.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("kokoro: block weight bank: %w", err)
	}
	return nil
}

// build compiles the pipelines. It cannot run until the fp32 arena exists,
// because every pipeline binds it.
func (s *blockSet) build() error {
	arenas := []*vk.Buffer{s.wbuf, s.abuf, s.hbuf, s.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, p := range []struct {
		name  string
		spirv []byte
	}{
		{"stats", shaders.KokoroDStats},
		{"affine", shaders.KokoroAffine},
		{"act", shaders.KokoroDAct},
		{"narrow", shaders.KokoroDNarrow},
		{"repeat", shaders.KokoroDNarrowRepeat},
		{"shortcut", shaders.KokoroShortcut},
		{"pool", shaders.KokoroPool},
		{"proj", shaders.KokoroProj},
	} {
		mod, err := s.dev.NewShaderModule(p.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", p.name, err)
		}
		s.mods = append(s.mods, mod)
		pipe, err := s.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", p.name, err)
		}
		s.pipes[p.name] = pipe
	}
	for _, v := range dconvVariants {
		sp := spec
		sp.RequiredSubgroupSize = v.wave
		if v.wave != 0 {
			ok, err := canPinWave(s.dev, v.wave)
			if err != nil {
				return err
			}
			if !ok {
				continue
			}
		}
		mod, err := s.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", v.name, err)
		}
		s.mods = append(s.mods, mod)
		pipe, err := s.dev.NewPipeline(mod, sp)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", v.name, err)
		}
		s.convs[v.name] = pipe
	}
	if _, ok := s.convs[s.kernel]; !ok {
		return fmt.Errorf("kokoro: kernel %q is not available on this device", s.kernel)
	}
	return nil
}

// stageBlocks writes the convolution weights, the biases and the pool.
//
// conv1's bias is not staged, for the reason T4a found in the generator: its
// output goes straight into an AdaIN, which subtracts the per-channel mean
// over time, and a per-channel constant moves that mean by exactly itself.
// conv2's and conv1x1's both reach the block's output, and since they are
// added to the same sum they are staged as one vector.
//
// The caller passes `extra`, a function that may write more of the fp32
// weight buffer — the predictor's two 1-wide projections, which are weights
// of the stack rather than of any block.
func (s *blockSet) stageBlocks(src []*AdainResBlk1d, extra func(w32 []float32)) error {
	bank := make([]uint16, s.bank.Size()/2)
	w32 := make([]float32, s.wbuf.Size()/4)
	for i := range s.blocks {
		db := &s.blocks[i]
		b := src[i]
		packConvBPad(bank[db.w1:], b.Conv1.Weight, db.out, db.in, db.inPad, b.Conv1.Kernel)
		packConvBPad(bank[db.w2:], b.Conv2.Weight, db.out, db.out, db.out, b.Conv2.Kernel)
		if !db.noSC {
			packConvBPad(bank[db.w1x1:], b.Conv1x1.Weight, db.out, db.in, db.inPad, 1)
		}

		bias := w32[db.bias : db.bias+uint32(db.out)]
		if b.Conv2.Bias != nil {
			copy(bias, b.Conv2.Bias)
		}
		if b.Conv1x1 != nil && b.Conv1x1.Bias != nil {
			for j, v := range b.Conv1x1.Bias {
				bias[j] += v
			}
		}
		if b.Pool != nil {
			copy(w32[db.pool:], b.Pool.Weight)
			if b.Pool.Bias != nil {
				copy(w32[db.poolBias:], b.Pool.Bias)
			}
		}
	}
	if extra != nil {
		extra(w32)
	}
	s.bank.WriteUint16At(0, bank)
	s.wbuf.WriteFloat32At(0, w32)
	// The fp16 arena's borders and its padded columns have to be zero before
	// the first convolution reads them, and nothing writes them afterwards.
	s.hbuf.WriteUint16At(0, make([]uint16, s.hElems))
	return nil
}

// packConvBPad is packConvB over a padded input width: the K index is
// tap*inPad + in, so the columns between `in` and `inPad` are left zero and
// contribute nothing.
//
// The padding is what makes A_CONV=2 addressable — a 16-wide fragment stays
// inside one tap and a 64-wide K slab does too — and it is why the decoder's
// 1090 channels cost 1152 rather than the 2048 a power-of-two pad would.
func packConvBPad(dst []uint16, w []float32, out, in, inPad, taps int) {
	k := taps * inPad
	kt := k / coopMatTile
	parallelFor(out, func(o int) {
		base := (o / coopMatTile) * kt * coopMatTile * coopMatTile
		lane := (o % coopMatTile) * coopMatTile
		for i := 0; i < in; i++ {
			for j := 0; j < taps; j++ {
				kk := j*inPad + i
				dst[base+(kk/coopMatTile)*coopMatTile*coopMatTile+lane+kk%coopMatTile] =
					safetensors.F32ToF16(w[(o*in+i)*taps+j])
			}
		}
	})
}

// SetStyle computes every AdaIN's gamma and beta for one utterance, which is
// the whole cost of the style conditioning: two [2C, 128] projections of one
// vector per block, on the host, once.
func (s *blockSet) SetStyle(style []float32) error {
	for i := range s.blocks {
		db := &s.blocks[i]
		for _, p := range []struct {
			fc   *Linear
			off  uint32
			want int
		}{{db.fc1, db.gb1, 2 * db.in}, {db.fc2, db.gb2, 2 * db.out}} {
			if p.fc.Out != p.want {
				return fmt.Errorf("kokoro: style projection is %d wide, want %d", p.fc.Out, p.want)
			}
			s.wbuf.WriteFloat32At(int(p.off), p.fc.ApplyRow(make([]float32, p.want), style))
		}
	}
	return nil
}

// graph builds one block's dispatch sequence, with a label per dispatch.
//
// The sequence is eleven dispatches at most and eight at least: two AdaIN
// reductions of three passes each, two convolutions, the shortcut's narrow
// and convolution when it has one, the pool when the block upsamples, and the
// epilogue that averages the two branches.
func (s *blockSet) graph(block int) ([]vk.MultiDispatch, []string, error) {
	db := &s.blocks[block]
	v, ok := dconvVariantFor(s.kernel)
	if !ok {
		return nil, nil, fmt.Errorf("kokoro: no variant %q", s.kernel)
	}
	if db.out%v.bn != 0 {
		return nil, nil, fmt.Errorf("kokoro: tile width %d does not divide %d output channels", v.bn, db.out)
	}

	var dis []vk.MultiDispatch
	var kinds []string
	add := func(pipe *vk.ComputePipeline, kind string, gx, gy uint32, pc pushConstants) {
		dis = append(dis, vk.MultiDispatch{Pipeline: pipe, GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	base := pushConstants{Eps: math.Float32bits(float32(instanceNormEps))}

	// AdaIN over time, then the leaky rectifier, then the narrow into the
	// bordered arena the convolution reads — three passes over the tensor in
	// the reference and one here, plus the two-step reduction the column
	// direction needs.
	adain := func(kind string, src, ldSrc, dst, ldDst, gb uint32, c, rows int) {
		chunks := uint32(decChunks)
		perChunk := uint32(roundUp(rows, decChunks) / decChunks)
		bands := groups(c, 256)

		pc := base
		pc.Tokens, pc.Dim = uint32(rows), uint32(c)
		pc.InOff, pc.OutOff, pc.Aux0 = src, s.aStat, perChunk
		add(s.pipes["stats"], kind+" stats", chunks, bands, pc)

		pc = base
		pc.Tokens, pc.Dim = uint32(rows), uint32(c)
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = s.aStat, s.aAff, gb, chunks
		add(s.pipes["affine"], kind+" affine", 1, 1, pc)

		pc = base
		pc.Dim = uint32(c)
		pc.InOff, pc.OutOff, pc.Aux0, pc.Aux1, pc.LDA = src, dst, s.aAff, ldSrc, ldDst
		pc.Scale = math.Float32bits(0.2)
		add(s.pipes["act"], kind+" act", bands, uint32(rows), pc)
	}
	// One convolution, with the im2col implicit in the A operand's addressing
	// and the zero padding in the arena's border.
	conv := func(kind string, in, bOff, out uint32, lda, taps, rows, n int) {
		m := roundUp(rows, v.bm)
		pc := base
		pc.InOff = in - uint32((taps-1)/2*lda)
		pc.OutOff, pc.BOff = out, bOff
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(m), uint32(n), uint32(taps*lda)
		pc.LDA = uint32(lda)
		pc.Aux0, pc.Aux1 = uint32(lda), uint32(lda)
		dis = append(dis, vk.MultiDispatch{
			Pipeline: s.convs[s.kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, kind)
	}

	adain("norm1", db.srcOff, db.srcLDA, db.hSrc, uint32(db.inPad), db.gb1, db.in, db.frames)

	convIn := db.hSrc
	if db.upsample {
		pc := base
		pc.Dim, pc.LDA = uint32(db.in), uint32(db.inPad)
		pc.InOff, pc.OutOff, pc.WOff, pc.BOff = db.hSrc, db.hPool, db.pool, db.poolBias
		add(s.pipes["pool"], "pool", groups(db.in, 256), uint32(db.conv), pc)
		convIn = db.hPool
	}
	conv("conv1", convIn, db.w1, s.aMid, db.inPad, 3, db.conv, db.out)

	adain("norm2", s.aMid, uint32(db.out), db.hMid, uint32(db.out), db.gb2, db.out, db.conv)
	conv("conv2", db.hMid, db.w2, s.aR, db.out, 3, db.conv, db.out)

	// The shortcut reads the block's input rather than its normalisation. A
	// block that changes its channel count convolves it, which needs a narrow
	// of its own — with the nearest-neighbour repeat folded in when the block
	// upsamples. A block that does not is the identity on it, and the
	// epilogue reads the input where it already lies: the predictor's stacks
	// have two of those and the decoder none.
	scOff, scLDA := db.srcOff, db.srcLDA
	if !db.noSC {
		pc := base
		pc.Dim = uint32(db.in)
		pc.InOff, pc.OutOff, pc.Aux1, pc.LDA = db.srcOff, db.hSc, db.srcLDA, uint32(db.inPad)
		narrow := "narrow"
		if db.upsample {
			narrow = "repeat"
		}
		add(s.pipes[narrow], narrow, groups(db.in, 256), uint32(db.conv), pc)
		conv("conv1x1", db.hSc, db.w1x1, s.aSc, db.inPad, 1, db.conv, db.out)
		scOff, scLDA = s.aSc, uint32(db.out)
	}

	pc := base
	pc.Dim, pc.LDA = uint32(db.out), db.dstLDA
	pc.InOff, pc.Aux0, pc.OutOff, pc.WOff = s.aR, scOff, db.dstOff, db.bias
	pc.Aux1 = scLDA
	pc.Scale = math.Float32bits(float32(1 / math.Sqrt2))
	add(s.pipes["shortcut"], "shortcut", groups(db.out, 256), uint32(db.conv), pc)

	return dis, kinds, nil
}

// destroy releases every Vulkan object the set owns. The fp32 arena is not
// one of them: the embedding object owns it.
func (s *blockSet) destroy() {
	for _, p := range s.pipes {
		p.Destroy()
	}
	for _, p := range s.convs {
		p.Destroy()
	}
	for _, m := range s.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{s.wbuf, s.hbuf, s.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	s.pipes, s.convs, s.mods = nil, nil, nil
	s.wbuf, s.hbuf, s.bank = nil, nil, nil
}

// setFrames rewrites one block's frame counts for an utterance shorter than
// the one the arenas were built for. The shapes it does not touch — channel
// counts, row strides, every offset — are what make this legal: a [T, C]
// region read as [n, C] with n < T is its own prefix, and nothing in a block
// addresses a row from the end.
func (db *decBlock) setFrames(frames int) {
	db.frames = frames
	db.conv = frames
	if db.upsample {
		db.conv = 2 * frames
	}
}

// zeroBorders restores the zero padding past each block's live rows.
//
// A sub-arena carries a border at both ends so the convolutions stay
// branchless (see hArenas). The one at the *start* is written once at staging
// and never touched again, but the one at the end moves with the utterance:
// after a shorter run, the rows just past the last live one still hold what a
// longer utterance left there, and a 3-tap convolution reads one of them.
//
// So the invariant is restored here rather than in a kernel: convBorder rows
// of zeros after every live region, which is the widest tap reach in this
// package and 8 KB a block from the host. It is only worth doing when the
// frame count actually changed, which is the owner's call.
func (s *blockSet) zeroBorders() {
	var zero []uint16
	put := func(off uint32, rows, lda int) {
		if off == noW || lda <= 0 {
			return
		}
		n := convBorder * lda
		if len(zero) < n {
			zero = make([]uint16, n)
		}
		s.hbuf.WriteUint16At(int(off)+rows*lda, zero[:n])
	}
	for i := range s.blocks {
		db := &s.blocks[i]
		put(db.hSrc, db.frames, db.inPad)
		put(db.hMid, db.conv, db.out)
		if !db.noSC {
			put(db.hSc, db.conv, db.inPad)
		}
		if db.upsample {
			put(db.hPool, db.conv, db.inPad)
		}
	}
}
