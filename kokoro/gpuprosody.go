package kokoro

import (
	"fmt"
	"math"

	"strix-halo-vulkan/vk"
)

// GPUProsody runs the prosody predictor's F0 and energy stacks on the device:
// six `AdainResBlk1d` at 512 and 256 channels, and the two 1-wide projections
// that turn their output into curves (SPEECH.md T6b).
//
// Measured before it was written, which is T6a's lesson: the stacks are 40 ms
// of the 64 ms `Prosody` takes and a third of the whole phoneme side, against
// the shared LSTM's 24. They are also *the same block the decoder already
// runs*, on the same kernels, so what this file contains is plumbing and not
// arithmetic. Three things about the plumbing differ from `GPUDecoder`:
//
//   - two stacks, not one, branching off the same input. They are staged as
//     one set of six blocks whose first and fourth read the same tensor, so
//     both curves come out of a single submit.
//   - four of the six blocks have no shortcut convolution, because they do
//     not change their channel count. Their shortcut is the block's input
//     read where it already lies, which is three dispatches and an fp16
//     sub-arena that do not happen — see `decBlock.noSC`.
//   - the block that upsamples is the *middle* one rather than the last, so
//     the frame count doubles halfway down the stack and every region after
//     it is twice as long.
//
// The shared recurrence that feeds it stays on the host: it is T6c's problem,
// and it is a GEMV at M = 1 per timestep rather than anything this file's
// kernels are shaped for.
type GPUProsody struct {
	blockSet

	frames   int // alignment frames in; the curves come out at 2*frames
	perStack int

	// The arena this object's fp32 space came from, when it is shared with
	// the recurrences: then aIn *is* `shared`'s output and the [130, 512]
	// between them never crosses the bus.
	shared *sharedArena
	src    []*AdainResBlk1d // held between alloc and finish

	aIn      uint32    // [frames, inCh] fp32, the shared LSTM's output
	aOut     [2]uint32 // the two curves, one value a frame
	wProj    [2]uint32 // [inCh] weights, then one bias each
	bProj    [2]uint32
	projIn   int // the projections' input channel count
	inCh     int // the stacks' input channel count
	actElems int
}

// NewGPUProsody stages both stacks for a given alignment frame count.
//
// The frame count is fixed at construction, as everywhere else in this
// package: the durations decide it, so the duration head has to have run.
func NewGPUProsody(dev *vk.Device, p *Predictor, frames int, kernel ConvKernel) (*GPUProsody, error) {
	return newGPUProsody(dev, p, frames, kernel, nil)
}

func newGPUProsody(dev *vk.Device, p *Predictor, frames int, kernel ConvKernel,
	sa *sharedArena) (*GPUProsody, error) {
	if frames <= 0 {
		return nil, fmt.Errorf("kokoro: prosody stacks over %d frames", frames)
	}
	if len(p.F0) != len(p.N) || len(p.F0) == 0 {
		return nil, fmt.Errorf("kokoro: %d F0 blocks against %d N blocks", len(p.F0), len(p.N))
	}
	g := &GPUProsody{
		blockSet: blockSet{
			dev: dev, kernel: kernel,
			pipes: map[string]*vk.ComputePipeline{},
			convs: map[ConvKernel]*vk.ComputePipeline{},
		},
		frames: frames, perStack: len(p.F0), shared: sa,
	}
	g.src = append(append([]*AdainResBlk1d{}, p.F0...), p.N...)
	if err := g.alloc(g.src, p); err != nil {
		g.Destroy()
		return nil, err
	}
	if sa != nil {
		// The buffer does not exist yet; the caller commits and calls finish.
		return g, nil
	}
	if err := g.finish(p); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// finish builds the pipelines and writes the weights, which cannot happen
// until every arena exists.
func (g *GPUProsody) finish(p *Predictor) error {
	if g.shared != nil {
		g.abuf = g.shared.buf
	}
	if err := g.build(); err != nil {
		return err
	}
	err := g.stageBlocks(g.src, func(w32 []float32) {
		for i, proj := range [2]*Conv1D{p.F0Proj, p.NProj} {
			copy(w32[g.wProj[i]:], proj.Weight)
			if proj.Bias != nil {
				w32[g.bProj[i]] = proj.Bias[0]
			}
		}
	})
	g.src = nil
	return err
}

// Attach redirects both stacks' input to a region another object owns, so the
// [frames, 512] between the shared recurrence and here never crosses the bus.
func (g *GPUProsody) Attach(src uint32, lda int) {
	g.aIn = src
	for s := 0; s < 2; s++ {
		db := &g.blocks[s*g.perStack]
		db.srcOff, db.srcLDA = src, uint32(lda)
	}
}

// Resident reports whether this object's input is another's output, and so
// whether Apply has anything to upload.
func (g *GPUProsody) Resident() bool { return g.shared != nil }

// alloc lays out the fp32 arena, wires each block's input and output, and
// hands the block shapes to the shared machinery.
//
// Every block gets an output region of its own rather than a ping-pong pair.
// Two of them would do — a block's input is dead the moment its epilogue
// writes — but six regions of a [260, 256] tensor is 1 MB, and having the
// whole stack readable afterwards is what makes a block-by-block correctness
// check possible at all.
func (g *GPUProsody) alloc(src []*AdainResBlk1d, p *Predictor) error {
	g.inCh = src[0].In
	for i, b := range src {
		if b.In != g.inCh && i%g.perStack == 0 {
			return fmt.Errorf("kokoro: stack %d starts at %d channels, not %d", i/g.perStack, b.In, g.inCh)
		}
	}
	for _, proj := range [2]*Conv1D{p.F0Proj, p.NProj} {
		if proj.Out != 1 || proj.Kernel != 1 {
			return fmt.Errorf("kokoro: projection is %d wide over %d taps, want 1 and 1",
				proj.Out, proj.Kernel)
		}
	}
	g.projIn = p.F0Proj.In
	if p.NProj.In != g.projIn {
		return fmt.Errorf("kokoro: projections take %d and %d channels", g.projIn, p.NProj.In)
	}

	// Walk both stacks once to size the shared scratch: the widest output at
	// the longest frame count, and the widest channel count either
	// normalisation reduces over.
	var maxRegion, maxC, outFrames int
	for s := 0; s < 2; s++ {
		t := g.frames
		for i := 0; i < g.perStack; i++ {
			b := src[s*g.perStack+i]
			conv := t
			if b.Upsample {
				conv = 2 * t
			}
			maxRegion = max(maxRegion, roundUp(conv, framePad)*b.Out)
			maxC = max(maxC, max(b.In, b.Out))
			t = conv
		}
		outFrames = t
	}

	var base, off32 uint32
	if g.shared != nil {
		base = g.shared.next
	}
	take32 := func(n int) uint32 { o := base + off32; off32 += uint32(n); return o }

	g.aIn = take32(roundUp(g.frames, framePad) * g.inCh)
	outs := make([]uint32, len(src))
	for s := 0; s < 2; s++ {
		t := g.frames
		for i := 0; i < g.perStack; i++ {
			b := src[s*g.perStack+i]
			if b.Upsample {
				t = 2 * t
			}
			outs[s*g.perStack+i] = take32(roundUp(t, framePad) * b.Out)
		}
	}
	g.aMid = take32(maxRegion)
	g.aR = take32(maxRegion)
	g.aSc = take32(maxRegion)
	g.aStat = take32(decChunks * 2 * maxC)
	g.aAff = take32(2 * maxC)
	for s := 0; s < 2; s++ {
		g.aOut[s] = take32(roundUp(outFrames, framePad))
	}
	g.actElems = int(off32)

	for s := 0; s < 2; s++ {
		t := g.frames
		for i := 0; i < g.perStack; i++ {
			j := s*g.perStack + i
			b := src[j]
			db := decBlock{
				in: b.In, out: b.Out, inPad: roundUp(b.In, chanPad),
				frames: t, conv: t, upsample: b.Upsample,
				noSC: b.Conv1x1 == nil,
				fc1:  b.Norm1.FC, fc2: b.Norm2.FC,
			}
			if b.Upsample {
				db.conv = 2 * t
			}
			// A block with no shortcut convolution adds its input to its
			// residual directly, so the two have to be the same shape — which
			// is the condition under which upstream leaves the convolution
			// out in the first place.
			if db.noSC && (db.in != db.out || db.upsample) {
				return fmt.Errorf("kokoro: block %d is %d->%d channels with no shortcut convolution",
					j, db.in, db.out)
			}
			if i == 0 {
				db.srcOff, db.srcLDA = g.aIn, uint32(g.inCh)
			} else {
				db.srcOff, db.srcLDA = outs[j-1], uint32(src[j-1].Out)
			}
			db.dstOff, db.dstLDA = outs[j], uint32(b.Out)
			g.blocks = append(g.blocks, db)
			t = db.conv
		}
	}

	// The two projections' weights live past the blocks' in the fp32 weight
	// buffer: [inCh] and a bias each.
	g.extraW = 2 * (g.projIn + 1)
	if err := g.allocBlocks(); err != nil {
		return err
	}
	off := g.wExtra
	for s := 0; s < 2; s++ {
		g.wProj[s] = off
		off += uint32(g.projIn)
		g.bProj[s] = off
		off++
	}

	if g.shared != nil {
		g.shared.reserve(g.actElems)
		return nil
	}
	buf, err := g.dev.NewBuffer(g.actElems * 4)
	if err != nil {
		return fmt.Errorf("kokoro: prosody fp32 arena: %w", err)
	}
	g.abuf = buf
	return nil
}

// OutFrames is how long the two curves are — twice the alignment rate,
// 12.5 ms a frame, because the vocoder's F0 conditioning is decimated by a
// stride-2 convolution on the way in.
func (g *GPUProsody) OutFrames() int {
	return g.blocks[g.perStack-1].conv
}

// graph is the whole of both stacks: six blocks and two projections.
func (g *GPUProsody) graph() ([]vk.MultiDispatch, []string, error) {
	var dis []vk.MultiDispatch
	var kinds []string
	for i := range g.blocks {
		b, k, err := g.blockSet.graph(i)
		if err != nil {
			return nil, nil, err
		}
		dis = append(dis, b...)
		for _, s := range k {
			kinds = append(kinds, fmt.Sprintf("%d %s", i, s))
		}
	}
	for s := 0; s < 2; s++ {
		last := &g.blocks[(s+1)*g.perStack-1]
		pc := pushConstants{Eps: math.Float32bits(float32(instanceNormEps))}
		pc.Dim, pc.LDA = uint32(g.projIn), last.dstLDA
		pc.InOff, pc.OutOff = last.dstOff, g.aOut[s]
		pc.WOff, pc.BOff = g.wProj[s], g.bProj[s]
		dis = append(dis, vk.MultiDispatch{
			Pipeline: g.pipes["proj"], GroupsX: uint32(last.conv), GroupsY: 1,
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, []string{"F0 proj", "N proj"}[s])
	}
	return dis, kinds, nil
}

// Upload writes the shared recurrence's output, which is the only thing
// either stack reads.
func (g *GPUProsody) Upload(shared *Mat) error {
	if shared.Rows != g.frames || shared.Cols != g.inCh {
		return fmt.Errorf("kokoro: prosody input is %v, want [%d %d]", shared, g.frames, g.inCh)
	}
	g.abuf.WriteFloat32At(int(g.aIn), shared.Data)
	return nil
}

// Apply runs both stacks over one utterance: six blocks, two projections, one
// submit, one upload of 266 KB and one download of 2 KB.
//
// The asymmetry is the point. This device reads host-visible memory at
// 210 MB/s and writes it at 29 GB/s (T4a), so what leaves the device matters
// far more than what enters it — and the two curves are 2*frames floats
// between them, where the last block's activation they came from is [260, 256]
// and would cost 1.3 ms to fetch.
func (g *GPUProsody) Apply(shared *Mat) (f0, energy []float32, err error) {
	if shared != nil {
		if err := g.Upload(shared); err != nil {
			return nil, nil, err
		}
	}
	dis, _, err := g.graph()
	if err != nil {
		return nil, nil, err
	}
	if err := submit(dis); err != nil {
		return nil, nil, err
	}
	n := g.OutFrames()
	return g.abuf.ReadFloat32At(int(g.aOut[0]), n), g.abuf.ReadFloat32At(int(g.aOut[1]), n), nil
}

// ApplyOne runs the blocks up to and including `block` and reads that block's
// output, which is what a correctness check wants and not what the pipeline
// does.
func (g *GPUProsody) ApplyOne(shared *Mat, block int) (*Mat, error) {
	if err := g.Upload(shared); err != nil {
		return nil, err
	}
	var dis []vk.MultiDispatch
	// A stack's blocks depend only on the ones before them *in that stack*,
	// so running from its first is enough.
	for i := block / g.perStack * g.perStack; i <= block; i++ {
		d, _, err := g.blockSet.graph(i)
		if err != nil {
			return nil, err
		}
		dis = append(dis, d...)
	}
	if err := submit(dis); err != nil {
		return nil, err
	}
	db := &g.blocks[block]
	out := NewMat(db.conv, db.out)
	row := g.abuf.ReadFloat32At(int(db.dstOff), db.conv*int(db.dstLDA))
	for r := 0; r < db.conv; r++ {
		copy(out.Row(r), row[r*int(db.dstLDA):])
	}
	return out, nil
}

// Profile times every dispatch of both stacks on the device.
func (g *GPUProsody) Profile() ([]Stage, error) {
	dis, kinds, err := g.graph()
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(dis))
	for i := range dis {
		t, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 4, true)
		if err != nil {
			return nil, err
		}
		out = append(out, Stage{Kind: kinds[i], Time: t.Seconds() / 4})
	}
	return out, nil
}

// Destroy releases every Vulkan object.
func (g *GPUProsody) Destroy() {
	g.blockSet.destroy()
	if g.shared == nil && g.abuf != nil {
		g.abuf.Destroy()
	}
	g.abuf = nil
}
