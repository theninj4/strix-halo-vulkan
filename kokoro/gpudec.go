package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// chanPad is what a channel count is rounded up to before it becomes a row
// stride on the device. It has to be a multiple of dit_gemm.comp's BK — the
// K-slab depth, BK_TILES*16 = 64 in every rung built here — because A_CONV=2
// carries the tap index across the K loop and rolls it over when the column
// offset reaches the padded width. Pad to less and a slab would straddle two
// taps; pad to a power of two instead and the decoder's 1090 channels would
// become 2048, nearly doubling the arithmetic.
const chanPad = 64

// decChunks is how many frame chunks the decoder's AdaIN reduction is split
// over. The generator's 64 is sized for a 15601-frame tensor; the decoder's
// are 130 and 260, so eight chunks of 17 or 33 frames is already more
// workgroups than the machine needs at 1090 channels (five channel bands
// each).
const decChunks = 8

// dconvVariants is the decoder's convolution ladder: the same four rungs the
// generator sweeps, built with A_CONV=2.
//
// Which one wins is measured rather than carried over, because the shape is
// the other way up. The generator's convolutions are M in the thousands by
// N = 128; the decoder's are M = 130 or 260 by N = 512 or 1024, with a K of
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
// The shapes are per block because the decoder's are not uniform: the first
// block is 514 channels wide and the last one doubles its frame count on the
// way through, so nearly every offset here is a property of the block rather
// than of the set.
type decBlock struct {
	in, out    int // channel counts, unpadded
	inPad      int // in, rounded up to chanPad: the fp16 row stride
	frames     int // input frames
	conv       int // frames reaching conv1 — 2*frames when the block upsamples
	upsample   bool
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

// GPUDecoder runs the vocoder's decoder — upstream's `Decoder`, the four AdaIN
// residual blocks and the `encode` block before them — on the device.
//
// It is 54% of the vocoder's time on 5.5% of its arithmetic, which is what
// happens when the other 94.5% moves and this does not. The blocks are the
// same shape of thing the generator's are and run on the same kernels; what
// kept them on the host is three differences, all of them in this file:
//
//   - the channel counts are 514, 1090, 1024 and 512, and two of those are
//     not powers of two, so A_CONV=1's shift and mask do not address them.
//     A_CONV=2 and a padded row stride do.
//   - the rectifier is leaky rather than a Snake, and it runs *after* the
//     normalisation rather than instead of it.
//   - there is a shortcut — a 1x1 convolution, and on one block a depthwise
//     transposed one that doubles the frames — and the two branches are
//     averaged by 1/sqrt(2) rather than added.
//
// The concatenation upstream does per block is a row stride here. Every
// decode block reads its own output beside the phoneme residual and the two
// curves, and those 66 side channels are the same for all four, so they are
// written once into columns [1024, 1090) of the running arena and each
// block's epilogue writes its 1024 beside them.
type GPUDecoder struct {
	dev *vk.Device

	frames int // alignment frames
	kernel ConvKernel
	blocks []decBlock

	// The arena this decoder's fp32 space came from, when it is shared with
	// the generator's stages: then aOut *is* stage 0's input and the
	// [260, 512] between them never crosses the bus.
	shared *sharedArena

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline
	convs                  map[ConvKernel]*vk.ComputePipeline

	// Arena offsets, fp32.
	aEnc, aCat       uint32
	aMid, aR, aSc    uint32
	aOut             uint32
	aStat, aAff      uint32
	actElems, hElems int

	// The side channels, held so Apply can write one contiguous [T, 1090]
	// block rather than 130 strided ones.
	cat []float32

	// The blocks, held between alloc and finish when the arena is shared.
	src []*AdainResBlk1d

	outCh int // the last block's output width
}

// NewGPUDecoder stages the decoder for a given alignment frame count.
//
// The frame count is fixed at construction because the arenas are, exactly as
// the generator's stages are: the durations decide it, so the prosody has to
// have run before this is called.
func NewGPUDecoder(dev *vk.Device, v *Vocoder, frames int, kernel ConvKernel) (*GPUDecoder, error) {
	return newGPUDecoder(dev, v, frames, kernel, nil)
}

func newGPUDecoder(dev *vk.Device, v *Vocoder, frames int, kernel ConvKernel,
	sa *sharedArena) (*GPUDecoder, error) {
	if frames <= 0 {
		return nil, fmt.Errorf("kokoro: decoder over %d frames", frames)
	}
	blocks := append([]*AdainResBlk1d{v.Encode}, v.Decode...)
	for i, b := range blocks {
		if b.Conv1x1 == nil {
			return nil, fmt.Errorf("kokoro: decoder block %d has no shortcut convolution", i)
		}
		if b.Upsample && i != len(blocks)-1 {
			return nil, fmt.Errorf("kokoro: decoder block %d upsamples and is not last", i)
		}
	}
	d := &GPUDecoder{
		dev: dev, frames: frames, kernel: kernel, shared: sa,
		pipes: map[string]*vk.ComputePipeline{},
		convs: map[ConvKernel]*vk.ComputePipeline{},
		src:   blocks,
	}
	if err := d.alloc(blocks, v.ASRRes.Out); err != nil {
		d.Destroy()
		return nil, err
	}
	if sa != nil {
		// The buffer does not exist yet; the caller commits and calls finish.
		return d, nil
	}
	if err := d.finish(); err != nil {
		d.Destroy()
		return nil, err
	}
	return d, nil
}

// finish builds the pipelines and writes the weights, which cannot happen
// until every arena exists.
func (d *GPUDecoder) finish() error {
	if d.shared != nil {
		d.abuf = d.shared.buf
	}
	if err := d.build(); err != nil {
		return err
	}
	err := d.stage(d.src)
	d.src = nil
	return err
}

// Frames is the alignment frame count this decoder was built for.
func (d *GPUDecoder) Frames() int { return d.frames }

// Kernel is the rung the convolutions run on.
func (d *GPUDecoder) Kernel() ConvKernel { return d.kernel }

// SetKernel switches rungs, which needs no restaging: the weights are
// fragment tiles and every rung reads those.
func (d *GPUDecoder) SetKernel(k ConvKernel) error {
	if _, ok := d.convs[k]; !ok {
		return fmt.Errorf("kokoro: no decoder pipeline for %q", k)
	}
	d.kernel = k
	return nil
}

// alloc lays out the three arenas and fills in every block's offsets.
func (d *GPUDecoder) alloc(blocks []*AdainResBlk1d, asrResCh int) error {
	t := d.frames
	// Every fp32 region is sized for the padded *output* frame count, because
	// the convolution writes roundUp(frames, BM) rows and the tail of those
	// is the product of zeros.
	pad := roundUp(2*t, framePad)

	var base, off32 uint32
	if d.shared != nil {
		base = d.shared.next
	}
	take32 := func(n int) uint32 { o := base + off32; off32 += uint32(n); return o }

	if len(blocks) < 2 {
		return fmt.Errorf("kokoro: %d decoder blocks", len(blocks))
	}
	enc, cat := blocks[0], blocks[1]
	var maxC, maxOut int
	for _, b := range blocks {
		maxC = max(maxC, b.In)
		maxOut = max(maxOut, b.Out)
	}
	d.outCh = blocks[len(blocks)-1].Out

	d.aEnc = take32(roundUp(t, framePad) * enc.In)
	d.aCat = take32(roundUp(t, framePad) * cat.In)
	d.aMid = take32(pad * maxOut)
	d.aR = take32(pad * maxOut)
	d.aSc = take32(pad * maxOut)
	d.aOut = take32(pad * d.outCh)
	d.aStat = take32(decChunks * 2 * maxC)
	d.aAff = take32(2 * maxC)
	d.actElems = int(off32)

	// fp16 sub-arenas, one per (rows, stride) an A operand is read at. Each
	// carries a zero border at both ends so the convolutions stay branchless,
	// and each is written only in rows [0, frames) thereafter — so a sub-arena
	// can be shared by blocks of the same shape and the border survives.
	var offH uint32
	arenas := map[[3]int]uint32{}
	// `use` distinguishes two sub-arenas of the same shape that are live at
	// the same moment: a block's normalisation and its shortcut are both a
	// narrow of a [frames, in] tensor, and they are different tensors.
	takeH := func(use, rows, lda int) uint32 {
		key := [3]int{use, rows, lda}
		if o, ok := arenas[key]; ok {
			return o
		}
		o := offH + uint32(convBorder*lda)
		offH += uint32((2*convBorder + roundUp(rows, framePad)) * lda)
		arenas[key] = o
		return o
	}

	for _, b := range blocks {
		db := decBlock{
			in: b.In, out: b.Out, inPad: roundUp(b.In, chanPad),
			frames: t, conv: t, upsample: b.Upsample,
			fc1: b.Norm1.FC, fc2: b.Norm2.FC,
		}
		if b.Upsample {
			db.conv = 2 * t
		}
		if b == enc {
			db.srcOff, db.srcLDA = d.aEnc, uint32(enc.In)
		} else {
			db.srcOff, db.srcLDA = d.aCat, uint32(cat.In)
		}
		if b.Upsample {
			db.dstOff, db.dstLDA = d.aOut, uint32(b.Out)
		} else {
			db.dstOff, db.dstLDA = d.aCat, uint32(cat.In)
		}
		db.hSrc = takeH(0, db.frames, db.inPad)
		db.hMid = takeH(1, db.conv, db.out)
		// A block's shortcut reads its *input*, not its normalisation, so it
		// needs a narrow of its own — at the doubled frame count on the block
		// that upsamples, since the repeat is folded into that pass.
		db.hSc = takeH(2, db.conv, db.inPad)
		if b.Upsample {
			db.hPool = takeH(3, db.conv, db.inPad)
		}
		d.blocks = append(d.blocks, db)
	}
	d.hElems = int(offH)

	// The running arena's side channels: [asr_res | F0 | N], the same 66
	// columns for every block, written once by Apply.
	d.cat = make([]float32, roundUp(t, framePad)*cat.In)
	if want := blocks[1].In - blocks[0].Out; want != asrResCh+2 {
		return fmt.Errorf("kokoro: %d side channels, asr_res gives %d and the curves 2", want, asrResCh)
	}

	var offW uint32
	for i := range d.blocks {
		db := &d.blocks[i]
		b := blocks[i]
		db.gb1 = offW
		offW += uint32(2 * db.in)
		db.gb2 = offW
		offW += uint32(2 * db.out)
		db.bias = offW
		offW += uint32(db.out)
		if b.Pool != nil {
			db.pool = offW
			offW += uint32(3 * db.in)
			db.poolBias = offW
			offW += uint32(db.in)
		} else {
			db.pool, db.poolBias = noW, noW
		}
	}

	var halves int
	for i := range d.blocks {
		db := &d.blocks[i]
		db.w1 = uint32(halves)
		halves += db.out * 3 * db.inPad
		db.w2 = uint32(halves)
		halves += db.out * 3 * db.out
		db.w1x1 = uint32(halves)
		halves += db.out * db.inPad
	}

	var err error
	if d.shared != nil {
		d.shared.reserve(d.actElems)
	} else if d.abuf, err = d.dev.NewBuffer(d.actElems * 4); err != nil {
		return fmt.Errorf("kokoro: decoder fp32 arena: %w", err)
	}
	if d.hbuf, err = d.dev.NewBuffer(d.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: decoder fp16 arena: %w", err)
	}
	if d.wbuf, err = d.dev.NewBuffer(int(offW) * 4); err != nil {
		return fmt.Errorf("kokoro: decoder fp32 weights: %w", err)
	}
	if d.bank, err = d.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("kokoro: decoder weight bank: %w", err)
	}
	return nil
}

func (d *GPUDecoder) build() error {
	arenas := []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, s := range []struct {
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
	} {
		mod, err := d.dev.NewShaderModule(s.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", s.name, err)
		}
		d.mods = append(d.mods, mod)
		pipe, err := d.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", s.name, err)
		}
		d.pipes[s.name] = pipe
	}
	for _, v := range dconvVariants {
		s := spec
		s.RequiredSubgroupSize = v.wave
		if v.wave != 0 {
			ok, err := canPinWave(d.dev, v.wave)
			if err != nil {
				return err
			}
			if !ok {
				continue
			}
		}
		mod, err := d.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", v.name, err)
		}
		d.mods = append(d.mods, mod)
		pipe, err := d.dev.NewPipeline(mod, s)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", v.name, err)
		}
		d.convs[v.name] = pipe
	}
	if _, ok := d.convs[d.kernel]; !ok {
		return fmt.Errorf("kokoro: decoder kernel %q is not available on this device", d.kernel)
	}
	return nil
}

// stage writes the weights.
//
// conv1's bias is not staged, for the reason T4a found in the generator: its
// output goes straight into an AdaIN, which subtracts the per-channel mean
// over time, and a per-channel constant moves that mean by exactly itself.
// conv2's and conv1x1's both reach the block's output, and since they are
// added to the same sum they are staged as one vector.
func (d *GPUDecoder) stage(blocks []*AdainResBlk1d) error {
	bank := make([]uint16, d.bank.Size()/2)
	w32 := make([]float32, d.wbuf.Size()/4)
	for i := range d.blocks {
		db := &d.blocks[i]
		b := blocks[i]
		packConvBPad(bank[db.w1:], b.Conv1.Weight, db.out, db.in, db.inPad, b.Conv1.Kernel)
		packConvBPad(bank[db.w2:], b.Conv2.Weight, db.out, db.out, db.out, b.Conv2.Kernel)
		packConvBPad(bank[db.w1x1:], b.Conv1x1.Weight, db.out, db.in, db.inPad, 1)

		bias := w32[db.bias : db.bias+uint32(db.out)]
		if b.Conv2.Bias != nil {
			copy(bias, b.Conv2.Bias)
		}
		if b.Conv1x1.Bias != nil {
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
	d.bank.WriteUint16At(0, bank)
	d.wbuf.WriteFloat32At(0, w32)
	// The fp16 arena's borders and its padded columns have to be zero before
	// the first convolution reads them, and nothing writes them afterwards.
	d.hbuf.WriteUint16At(0, make([]uint16, d.hElems))
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
// the whole cost of the style conditioning: ten [2C, 128] projections of one
// vector, on the host, once.
func (d *GPUDecoder) SetStyle(style []float32) error {
	for i := range d.blocks {
		db := &d.blocks[i]
		for _, s := range []struct {
			fc   *Linear
			off  uint32
			want int
		}{{db.fc1, db.gb1, 2 * db.in}, {db.fc2, db.gb2, 2 * db.out}} {
			if s.fc.Out != s.want {
				return fmt.Errorf("kokoro: style projection is %d wide, want %d", s.fc.Out, s.want)
			}
			d.wbuf.WriteFloat32At(int(s.off), s.fc.ApplyRow(make([]float32, s.want), style))
		}
	}
	return nil
}

// graph builds one block's dispatch sequence, with a label per dispatch.
func (d *GPUDecoder) graph(block int) ([]vk.MultiDispatch, []string, error) {
	db := &d.blocks[block]
	v, ok := dconvVariantFor(d.kernel)
	if !ok {
		return nil, nil, fmt.Errorf("kokoro: no decoder variant %q", d.kernel)
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
		pc.InOff, pc.OutOff, pc.Aux0 = src, d.aStat, perChunk
		add(d.pipes["stats"], kind+" stats", chunks, bands, pc)

		pc = base
		pc.Tokens, pc.Dim = uint32(rows), uint32(c)
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = d.aStat, d.aAff, gb, chunks
		add(d.pipes["affine"], kind+" affine", 1, 1, pc)

		pc = base
		pc.Dim = uint32(c)
		pc.InOff, pc.OutOff, pc.Aux0, pc.Aux1, pc.LDA = src, dst, d.aAff, ldSrc, ldDst
		pc.Scale = math.Float32bits(0.2)
		add(d.pipes["act"], kind+" act", bands, uint32(rows), pc)
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
			Pipeline: d.convs[d.kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
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
		add(d.pipes["pool"], "pool", groups(db.in, 256), uint32(db.conv), pc)
		convIn = db.hPool
	}
	conv("conv1", convIn, db.w1, d.aMid, db.inPad, 3, db.conv, db.out)

	adain("norm2", d.aMid, uint32(db.out), db.hMid, uint32(db.out), db.gb2, db.out, db.conv)
	conv("conv2", db.hMid, db.w2, d.aR, db.out, 3, db.conv, db.out)

	// The shortcut reads the block's input rather than its normalisation, so
	// it needs a narrow of its own — with the nearest-neighbour repeat folded
	// in on the block that upsamples.
	pc := base
	pc.Dim = uint32(db.in)
	pc.InOff, pc.OutOff, pc.Aux1, pc.LDA = db.srcOff, db.hSc, db.srcLDA, uint32(db.inPad)
	narrow := "narrow"
	if db.upsample {
		narrow = "repeat"
	}
	add(d.pipes[narrow], narrow, groups(db.in, 256), uint32(db.conv), pc)
	conv("conv1x1", db.hSc, db.w1x1, d.aSc, db.inPad, 1, db.conv, db.out)

	pc = base
	pc.Dim, pc.LDA = uint32(db.out), db.dstLDA
	pc.InOff, pc.Aux0, pc.OutOff, pc.WOff = d.aR, d.aSc, db.dstOff, db.bias
	pc.Scale = math.Float32bits(float32(1 / math.Sqrt2))
	add(d.pipes["shortcut"], "shortcut", groups(db.out, 256), uint32(db.conv), pc)

	return dis, kinds, nil
}

// Upload writes the decoder's two inputs: the `encode` block's operand and the
// side channels every decode block reads beside its own output.
func (d *GPUDecoder) Upload(asr, f0c, nc, asrRes *Mat) error {
	t := d.frames
	enc := &d.blocks[0]
	cat := &d.blocks[1]
	if asr.Rows != t || f0c.Rows != t || nc.Rows != t || asrRes.Rows != t {
		return fmt.Errorf("kokoro: decoder inputs are %d/%d/%d/%d frames, want %d",
			asr.Rows, f0c.Rows, nc.Rows, asrRes.Rows, t)
	}
	if asr.Cols+f0c.Cols+nc.Cols != enc.in {
		return fmt.Errorf("kokoro: %d+%d+%d channels into a %d-wide encode block",
			asr.Cols, f0c.Cols, nc.Cols, enc.in)
	}
	e := make([]float32, t*enc.in)
	for r := 0; r < t; r++ {
		row := e[r*enc.in:]
		copy(row, asr.Row(r))
		copy(row[asr.Cols:], f0c.Row(r))
		copy(row[asr.Cols+f0c.Cols:], nc.Row(r))
	}
	d.abuf.WriteFloat32At(int(d.aEnc), e)

	// Columns [encode.Out, cat.in) are the concatenation every decode block
	// sees; the rest is the running value and is written on the device.
	side := enc.out
	for r := 0; r < t; r++ {
		row := d.cat[r*cat.in+side:]
		copy(row, asrRes.Row(r))
		copy(row[asrRes.Cols:], f0c.Row(r))
		copy(row[asrRes.Cols+f0c.Cols:], nc.Row(r))
	}
	d.abuf.WriteFloat32At(int(d.aCat), d.cat)
	return nil
}

// Download reads the decoder's output — [2*frames, 512], the generator's
// input.
func (d *GPUDecoder) Download() *Mat {
	return &Mat{Rows: 2 * d.frames, Cols: d.outCh,
		Data: d.abuf.ReadFloat32At(int(d.aOut), 2*d.frames*d.outCh)}
}

// Resident reports whether the decoder's output stays on the device — which
// it does when the generator's stages share its arena, and then Download is
// 0.5 MB of readback nobody needs.
func (d *GPUDecoder) Resident() bool { return d.shared != nil }

// Run executes the whole decoder over the uploaded inputs: five blocks, one
// submit, and nothing downloaded.
func (d *GPUDecoder) Run(asr, f0c, nc, asrRes *Mat) error {
	if err := d.Upload(asr, f0c, nc, asrRes); err != nil {
		return err
	}
	var dis []vk.MultiDispatch
	for i := range d.blocks {
		g, _, err := d.graph(i)
		if err != nil {
			return err
		}
		dis = append(dis, g...)
	}
	return submit(dis)
}

// Apply runs the whole decoder over one utterance: five blocks, one submit,
// one upload and one download.
func (d *GPUDecoder) Apply(asr, f0c, nc, asrRes *Mat) (*Mat, error) {
	if err := d.Run(asr, f0c, nc, asrRes); err != nil {
		return nil, err
	}
	return d.Download(), nil
}

// ApplyOne runs a single block over an already-uploaded input, which is what
// a correctness check wants and not what the pipeline does.
func (d *GPUDecoder) ApplyOne(block int) (*Mat, error) {
	dis, _, err := d.graph(block)
	if err != nil {
		return nil, err
	}
	if err := submit(dis); err != nil {
		return nil, err
	}
	db := &d.blocks[block]
	out := NewMat(db.conv, db.out)
	row := d.abuf.ReadFloat32At(int(db.dstOff), db.conv*int(db.dstLDA))
	for r := 0; r < db.conv; r++ {
		copy(out.Row(r), row[r*int(db.dstLDA):])
	}
	return out, nil
}

// Profile times every dispatch of the whole decoder on the device.
func (d *GPUDecoder) Profile() ([]Stage, error) {
	var out []Stage
	for i := range d.blocks {
		dis, kinds, err := d.graph(i)
		if err != nil {
			return nil, err
		}
		for j := range dis {
			t, err := vk.DispatchMultiTimed(dis[j:j+1], 1, 4, true)
			if err != nil {
				return nil, err
			}
			out = append(out, Stage{Kind: fmt.Sprintf("%d %s", i, kinds[j]), Time: t.Seconds() / 4})
		}
	}
	return out, nil
}

// Destroy releases every Vulkan object.
func (d *GPUDecoder) Destroy() {
	for _, p := range d.pipes {
		p.Destroy()
	}
	for _, p := range d.convs {
		p.Destroy()
	}
	for _, m := range d.mods {
		m.Destroy()
	}
	bufs := []*vk.Buffer{d.wbuf, d.hbuf, d.bank}
	if d.shared == nil {
		bufs = append(bufs, d.abuf)
	}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
	}
	d.pipes, d.convs, d.mods = nil, nil, nil
	d.wbuf, d.abuf, d.hbuf, d.bank = nil, nil, nil, nil
}
