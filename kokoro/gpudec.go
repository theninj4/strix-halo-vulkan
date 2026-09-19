package kokoro

import (
	"fmt"

	"strix-halo-vulkan/vk"
)

// GPUDecoder runs the vocoder's decoder — upstream's `Decoder`, the four AdaIN
// residual blocks and the `encode` block before them — on the device.
//
// It is 54% of the vocoder's time on 5.5% of its arithmetic, which is what
// happens when the other 94.5% moves and this does not. The blocks are the
// same shape of thing the generator's are and run on the same kernels; what
// kept them on the host is three differences, all of them in `blockSet`:
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
// What is left in this file is the plumbing, which is what makes a decoder a
// decoder rather than any other stack of the same block. The concatenation
// upstream does per block is a row stride here: every decode block reads its
// own output beside the phoneme residual and the two curves, and those 66
// side channels are the same for all four, so they are written once into
// columns [1024, 1090) of the running arena and each block's epilogue writes
// its 1024 beside them.
type GPUDecoder struct {
	blockSet

	frames int // alignment frames the arenas were built for
	run    int // alignment frames of the utterance running now

	// The arena this decoder's fp32 space came from, when it is shared with
	// the generator's stages: then aOut *is* stage 0's input and the
	// [260, 512] between them never crosses the bus.
	shared *sharedArena

	// Arena offsets, fp32. The scratch every block shares lives on blockSet.
	aEnc, aCat uint32
	aOut       uint32
	actElems   int

	// The side channels, held so Apply can write one contiguous [T, 1090]
	// block rather than 130 strided ones.
	cat []float32

	// The blocks, held between alloc and finish when the arena is shared.
	src []*AdainResBlk1d

	outCh int // the last block's output width
}

// NewGPUDecoder stages the decoder for alignment frame counts up to `frames`.
//
// The arenas are sized by that ceiling, not by the utterance: every region
// here is [T, C] row-major, so a shorter utterance is a prefix of each one and
// SetFrames is the whole of what it takes to run one. Staging is 85 ms and an
// utterance is 25, which is why the ceiling is a server's to pick once rather
// than a request's to pay (SPEECH.md T9).
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
		blockSet: blockSet{
			dev: dev, kernel: kernel,
			pipes: map[string]*vk.ComputePipeline{},
			convs: map[ConvKernel]*vk.ComputePipeline{},
		},
		frames: frames, run: frames, shared: sa, src: blocks,
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
	err := d.stageBlocks(d.src, nil)
	d.src = nil
	return err
}

// Frames is the alignment frame count this decoder was built for, which is
// the longest utterance it can run rather than the one it is running.
func (d *GPUDecoder) Frames() int { return d.frames }

// RunFrames is the utterance currently sized for.
func (d *GPUDecoder) RunFrames() int { return d.run }

// SetFrames sizes the decoder for one utterance, which must fit the ceiling
// the arenas were built for.
//
// Everything it changes is a dispatch extent or a push constant: the blocks
// keep their offsets, their strides and their weights, and the only memory
// touched is the fp16 border the shorter run leaves stale (see zeroBorders).
func (d *GPUDecoder) SetFrames(frames int) error {
	if frames <= 0 || frames > d.frames {
		return fmt.Errorf("kokoro: %d frames against a decoder staged for %d", frames, d.frames)
	}
	if frames == d.run {
		return nil
	}
	for i := range d.blocks {
		d.blocks[i].setFrames(frames)
	}
	d.zeroBorders()
	d.run = frames
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
	// The reduction runs over a block's input channels for norm1 and its
	// output channels for norm2, so the partials are sized for the wider of
	// the two over the whole stack.
	var maxC, maxOut int
	for _, b := range blocks {
		maxC = max(maxC, max(b.In, b.Out))
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
		d.blocks = append(d.blocks, db)
	}

	// The running arena's side channels: [asr_res | F0 | N], the same 66
	// columns for every block, written once by Apply.
	d.cat = make([]float32, roundUp(t, framePad)*cat.In)
	if want := blocks[1].In - blocks[0].Out; want != asrResCh+2 {
		return fmt.Errorf("kokoro: %d side channels, asr_res gives %d and the curves 2", want, asrResCh)
	}

	if err := d.allocBlocks(); err != nil {
		return err
	}
	if d.shared != nil {
		d.shared.reserve(d.actElems)
		return nil
	}
	buf, err := d.dev.NewBuffer(d.actElems * 4)
	if err != nil {
		return fmt.Errorf("kokoro: decoder fp32 arena: %w", err)
	}
	d.abuf = buf
	return nil
}

// Upload writes the decoder's two inputs: the `encode` block's operand and the
// side channels every decode block reads beside its own output.
func (d *GPUDecoder) Upload(asr, f0c, nc, asrRes *Mat) error {
	t := d.run
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
	// Only the live rows: the arena is sized for the ceiling, and what lies
	// past this utterance is read by nothing (see SetFrames).
	d.abuf.WriteFloat32At(int(d.aCat), d.cat[:t*cat.in])
	return nil
}

// Download reads the decoder's output — [2*frames, 512], the generator's
// input.
func (d *GPUDecoder) Download() *Mat {
	return &Mat{Rows: 2 * d.run, Cols: d.outCh,
		Data: d.abuf.ReadFloat32At(int(d.aOut), 2*d.run*d.outCh)}
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
	d.blockSet.destroy()
	if d.shared == nil && d.abuf != nil {
		d.abuf.Destroy()
	}
	d.abuf = nil
}
