package parakeet

// The transducer tail on Vulkan: SPEECH.md stage S8.
//
// S6 and S7 left the encoder at 20 ms of wall clock for an 11 s clip and the
// decode loop at 207 ms -- 82% of the pipeline, for 1.1 GFLOP. Nothing about
// that is arithmetic. The loop is 46 emissions, each one LSTM step and one
// 8198-wide projection, and what it costs is that every one of them is a
// round trip: the model is on the device, the decision is on the host, and
// the two alternate.
//
// # Why this is not like anything else in the engine
//
// Greedy transducer decoding is **sequential by construction**. The
// prediction network is a language model over the tokens it has already
// emitted, so step n+1's input is step n's output and there is no batch to
// find. Every other kernel in this repository is a throughput problem with a
// tile to pick; this one is a latency problem with a round trip to remove,
// and the ladder's answer -- a wider tile -- makes it worse rather than
// better.
//
// What follows from that:
//
//   - **M = 1, and it stays 1.** The four projections here run on the same
//     dit_gemm.comp rungs the encoder does, with M padded up to the narrowest
//     tile in the ladder. That costs arithmetic (16x of it) and *no
//     bandwidth*: a GEMM reads its B operand once whatever M is, and at 24
//     MFLOP against 18 MB per emission the B operand is the whole cost. A
//     dedicated GEMV would win the arithmetic back and it is not worth a
//     kernel until the round trips are gone.
//   - **The state stays resident.** h and c for both layers, the projected
//     encoder frames, the logits -- none of it crosses the bus. What crosses
//     per emission is 640 floats of embedding in (2.5 KB, because the
//     embedding table is a host-side gather rather than a device tensor) and
//     **two floats out**: the argmax over the vocabulary and the argmax over
//     the durations.
//   - **The control flow stays on the host.** The rule that joins the two
//     argmaxes -- a blank asking for a zero-frame jump is forced to one, or
//     the loop cannot terminate -- and the cursor arithmetic are decisions
//     between submits. A persistent kernel would remove the remaining fences
//     and is what a streaming API would want; this is the honest first
//     version, and it is already the difference between 207 ms and single
//     digits.
//
// # Where it lives
//
// Its own four buffers, in the roles dit_common.glsl already names: fp32
// weights, fp32 activations, fp16 activations, fp16 weight bank. That is why
// `dit_gemm.comp` and `parakeet_narrow_f16.comp` and `parakeet_sub_bias.comp`
// run here unchanged over a completely separate arena -- the binding layout
// is a contract about *roles*, not about which tensors.
//
// The encoder's hidden states cross the host between the two, which is not a
// cost this stage adds: `GPUEncoder.ApplyMel` already returns them, and 138
// rows of 1024 floats is 565 KB.

import (
	"fmt"
	"math"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// DecProj names one of the five matrices the tail multiplies by.
type DecProj string

const (
	DecEnc   DecProj = "enc.proj"
	DecLSTM0 DecProj = "lstm0"
	DecLSTM1 DecProj = "lstm1"
	DecPred  DecProj = "pred.proj"
	DecJoint DecProj = "joint"
)

// decProjOrder is every matrix, in the order the graph issues them.
var decProjOrder = []DecProj{DecEnc, DecLSTM0, DecLSTM1, DecPred, DecJoint}

// decLSTM is the projection name for an LSTM layer.
func decLSTM(l int) DecProj {
	if l == 0 {
		return DecLSTM0
	}
	return DecLSTM1
}

// DecPlan chooses a kernel per projection of the tail.
type DecPlan map[DecProj]GEMMKernel

// UniformDecPlan runs every projection on one kernel, which is what the
// ladder sweeps.
func UniformDecPlan(k GEMMKernel) DecPlan {
	p := DecPlan{}
	for _, r := range decProjOrder {
		p[r] = k
	}
	return p
}

// DefaultDecPlan is the narrowest tile in the ladder, which is what M = 1
// wants: every row of the tile above the first multiplies zeros, so the
// smallest BM is the least waste. Read off TestGPUDecLadder.
func DefaultDecPlan() DecPlan { return UniformDecPlan(GEMMReg16x64W32) }

// GPUDecoder runs the encoder projector, the prediction network, the joint
// and the TDT greedy loop on the device.
type GPUDecoder struct {
	dev *vk.Device
	m   *Model

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	gemms                  map[GEMMKernel]*vk.ComputePipeline
	kernels                map[GEMMKernel]gemmVariant
	mods                   []*vk.ShaderModule
	plan                   DecPlan

	frames, rowsPad  int
	rows, rowsRun    int
	encDim, hidden   int
	vocab, jointOut  int
	jointPad, layers int
	blank            int
	maxSymbols       int
	durations        []int

	ldaEnc, ldaLSTM, ldaHidden int

	wEncBias, wPredBias, wJointBias uint32
	wLSTMBias                       []uint32
	bOff                            map[DecProj]uint32

	aHidden, aEnc, aX               uint32
	aH, aC                          []uint32
	aGates, aPred, aLogits, aResult uint32
	hHidden, hLSTM, hPred, hSum     uint32

	// enc is the encoder this decoder reads its input straight out of, once
	// Attach has built the pipeline for it.
	enc     *GPUEncoder
	encPipe *vk.ComputePipeline

	// Steps counts the emissions of the last Decode and Submits the command
	// buffers it took, which is the measurement this stage is about.
	Steps, Submits int
}

// decTile is how many rows of an A operand the pre-passes write, and how the
// arenas are sized: the widest tile in the ladder, so a plan can be changed
// after construction without reallocating anything. Only the first row ever
// holds a vector; the rest are zeros, because the GEMM covers whole tiles.
const decTile = arenaAlign

// NewGPUDecoder stages the tail onto the device, sized for clips of at most
// maxFrames encoder frames.
//
// The weights come from the loaded Model, so a test that runs this against
// Model.Decode is comparing two implementations rather than two loads -- the
// same argument NewGPUEncoder makes.
func NewGPUDecoder(dev *vk.Device, m *Model, maxFrames int) (*GPUDecoder, error) {
	if maxFrames <= 0 {
		return nil, fmt.Errorf("parakeet: maxFrames is %d", maxFrames)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("parakeet: this device has no 16x16x16 fp16 cooperative matrix")
	}
	d := &GPUDecoder{
		dev:        dev,
		m:          m,
		pipes:      make(map[string]*vk.ComputePipeline),
		gemms:      make(map[GEMMKernel]*vk.ComputePipeline),
		kernels:    make(map[GEMMKernel]gemmVariant),
		frames:     maxFrames,
		blank:      m.Config.BlankTokenID,
		maxSymbols: m.Config.MaxSymbolsPerStep,
		durations:  m.Joint.Durations,
	}
	if err := d.check(); err != nil {
		return nil, err
	}
	d.rowsPad = roundUp(maxFrames, arenaAlign)
	d.rows, d.rowsRun = maxFrames, d.rowsPad
	d.ldaEnc = d.encDim + gemmPad
	d.ldaLSTM = 2*d.hidden + gemmPad
	d.ldaHidden = d.hidden + gemmPad
	// jointPad is the head's output width rounded up to the widest BN in the
	// ladder, so that any rung divides it. The rows past 8198 are staged as
	// zeros and never read: the two argmaxes cover [0, vocab) and
	// [vocab, vocab+durations).
	d.jointPad = roundUp(d.jointOut, 256)

	for _, v := range gemmVariants {
		d.kernels[v.name] = v
	}
	if err := d.SetPlan(DefaultDecPlan()); err != nil {
		return nil, err
	}
	if err := d.layout(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.build(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.stage(); err != nil {
		d.Destroy()
		return nil, err
	}
	return d, nil
}

// check reads the tail's shapes off the loaded model and holds them to what
// the kernels assume.
func (d *GPUDecoder) check() error {
	m := d.m
	switch {
	case m.Projector == nil || m.Prediction == nil || m.Joint == nil || m.Joint.Head == nil:
		return fmt.Errorf("parakeet: the model has no transducer tail")
	case m.Tokenizer == nil:
		return fmt.Errorf("parakeet: the model has no tokenizer")
	case len(m.Prediction.Layers) == 0:
		return fmt.Errorf("parakeet: the prediction network has no LSTM layers")
	case len(d.durations) == 0:
		return fmt.Errorf("parakeet: the joint has no durations")
	}
	d.encDim, d.hidden = m.Projector.In, m.Projector.Out
	d.layers = len(m.Prediction.Layers)
	d.vocab = m.Joint.Vocab
	d.jointOut = m.Joint.Head.Out
	if d.jointOut != d.vocab+len(d.durations) {
		return fmt.Errorf("parakeet: the joint head is %d wide, want %d tokens + %d durations",
			d.jointOut, d.vocab, len(d.durations))
	}
	if m.Prediction.Hidden != d.hidden || m.Prediction.Projector.In != d.hidden || m.Prediction.Projector.Out != d.hidden {
		return fmt.Errorf("parakeet: the prediction network is %d wide, the joint wants %d", m.Prediction.Hidden, d.hidden)
	}
	if m.Joint.Head.In != d.hidden {
		return fmt.Errorf("parakeet: the joint head takes %d features, the projector makes %d", m.Joint.Head.In, d.hidden)
	}
	for i, l := range m.Prediction.Layers {
		if l.Hidden != d.hidden || l.In != d.hidden {
			return fmt.Errorf("parakeet: LSTM layer %d is %d -> %d, want %d -> %d", i, l.In, l.Hidden, d.hidden, d.hidden)
		}
		if len(l.WIH) != 4*d.hidden*d.hidden || len(l.WHH) != 4*d.hidden*d.hidden || len(l.Bias) != 4*d.hidden {
			return fmt.Errorf("parakeet: LSTM layer %d has the wrong weight sizes", i)
		}
	}
	if d.blank < 0 || d.blank >= d.vocab {
		return fmt.Errorf("parakeet: blank token %d outside a vocabulary of %d", d.blank, d.vocab)
	}
	return nil
}

// decProjShape is [out, in] for a projection of the tail.
func (d *GPUDecoder) decProjShape(r DecProj) (int, int) {
	switch r {
	case DecEnc:
		return d.hidden, d.encDim
	case DecLSTM0, DecLSTM1:
		return 4 * d.hidden, 2 * d.hidden
	case DecPred:
		return d.hidden, d.hidden
	default:
		return d.jointPad, d.hidden
	}
}

// layout plans every offset and allocates the four buffers. Nothing is read
// here, so the whole arena is decided before a weight is touched.
func (d *GPUDecoder) layout() error {
	var off32, offA, offH, off16 uint32
	take := func(p *uint32, n int) uint32 { o := *p; *p += uint32(n); return o }

	d.wEncBias = take(&off32, d.hidden)
	d.wPredBias = take(&off32, d.hidden)
	d.wJointBias = take(&off32, d.jointPad)
	d.wLSTMBias = make([]uint32, d.layers)
	for i := range d.wLSTMBias {
		d.wLSTMBias[i] = take(&off32, 4*d.hidden)
	}

	d.bOff = make(map[DecProj]uint32, len(decProjOrder))
	for _, r := range decProjOrder {
		if r == DecLSTM1 && d.layers < 2 {
			continue
		}
		n, k := d.decProjShape(r)
		d.bOff[r] = take(&off16, n*k)
	}

	d.aHidden = take(&offA, d.rowsPad*d.encDim)
	d.aEnc = take(&offA, d.rowsPad*d.hidden)
	d.aX = take(&offA, d.hidden)
	d.aH, d.aC = make([]uint32, d.layers), make([]uint32, d.layers)
	for i := 0; i < d.layers; i++ {
		d.aH[i] = take(&offA, d.hidden)
		d.aC[i] = take(&offA, d.hidden)
	}
	d.aGates = take(&offA, decTile*4*d.hidden)
	d.aPred = take(&offA, decTile*d.hidden)
	d.aLogits = take(&offA, decTile*d.jointPad)
	d.aResult = take(&offA, 4)

	d.hHidden = take(&offH, d.rowsPad*d.ldaEnc)
	d.hLSTM = take(&offH, decTile*d.ldaLSTM)
	d.hPred = take(&offH, decTile*d.ldaHidden)
	d.hSum = take(&offH, decTile*d.ldaHidden)

	var err error
	if d.wbuf, err = d.dev.NewBuffer(int(off32) * 4); err != nil {
		return fmt.Errorf("parakeet: decoder fp32 weights: %w", err)
	}
	if d.abuf, err = d.dev.NewBuffer(int(offA) * 4); err != nil {
		return fmt.Errorf("parakeet: decoder fp32 activations: %w", err)
	}
	if d.hbuf, err = d.dev.NewBuffer(int(offH) * 2); err != nil {
		return fmt.Errorf("parakeet: decoder fp16 activations: %w", err)
	}
	if d.bank, err = d.dev.NewBuffer(int(off16) * 2); err != nil {
		return fmt.Errorf("parakeet: decoder fp16 weight bank (%d MB): %w", (int(off16)*2)>>20, err)
	}
	return nil
}

// build compiles every pipeline over the decoder's own four buffers, in the
// roles dit_common.glsl names them.
func (d *GPUDecoder) build() error {
	spec := vk.PipelineSpec{
		Buffers:          []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank},
		PushConstantSize: uint32(pushConstantSize),
	}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"narrow16", shaders.ParakeetNarrowF16},
		{"bias", shaders.ParakeetSubBias},
		{"lstmin", shaders.ParakeetLSTMIn},
		{"lstmgate", shaders.ParakeetLSTMGate},
		{"jointsum", shaders.ParakeetJointSum},
		{"argmax", shaders.ParakeetArgmax},
	} {
		mod, err := d.dev.NewShaderModule(s.spirv)
		if err != nil {
			return fmt.Errorf("parakeet: decoder shader %s: %w", s.name, err)
		}
		d.mods = append(d.mods, mod)
		pipe, err := d.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("parakeet: decoder pipeline %s: %w", s.name, err)
		}
		d.pipes[s.name] = pipe
	}
	for _, v := range gemmVariants {
		ok, err := canPinWave(d.dev, v.wave)
		if err != nil {
			return err
		}
		if !ok {
			delete(d.kernels, v.name)
			continue
		}
		gspec := spec
		gspec.RequiredSubgroupSize = v.wave
		mod, err := d.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("parakeet: decoder gemm %s: %w", v.name, err)
		}
		d.mods = append(d.mods, mod)
		pipe, err := d.dev.NewPipeline(mod, gspec)
		if err != nil {
			return fmt.Errorf("parakeet: decoder gemm %s: %w", v.name, err)
		}
		d.gemms[v.name] = pipe
	}
	for _, r := range decProjOrder {
		if _, ok := d.kernels[d.plan[r]]; !ok {
			return d.SetPlan(UniformDecPlan(GEMMReg32x64))
		}
	}
	return nil
}

// SetPlan changes which tile each projection runs on.
func (d *GPUDecoder) SetPlan(plan DecPlan) error {
	next := DecPlan{}
	for _, r := range decProjOrder {
		k, ok := plan[r]
		if !ok {
			return fmt.Errorf("parakeet: dec plan names no kernel for %s", r)
		}
		v, ok := d.kernels[k]
		if !ok {
			return fmt.Errorf("parakeet: no such projection kernel %q", k)
		}
		if v.layout != kernelLayout {
			return fmt.Errorf("parakeet: kernel %q reads B layout %d, the weights are staged as %d", k, v.layout, kernelLayout)
		}
		n, _ := d.decProjShape(r)
		if n%v.bn != 0 {
			return fmt.Errorf("parakeet: kernel %q has BN %d, which does not divide %s's %d columns", k, v.bn, r, n)
		}
		next[r] = k
	}
	d.plan = next
	return nil
}

// Plan is the kernel currently chosen for each projection.
func (d *GPUDecoder) Plan() DecPlan { return d.plan }

// stage fills the four buffers.
func (d *GPUDecoder) stage() error {
	m := d.m
	d.wbuf.WriteFloat32At(int(d.wEncBias), m.Projector.Bias)
	d.wbuf.WriteFloat32At(int(d.wPredBias), m.Prediction.Projector.Bias)
	// The head's bias is padded with zeros, as its weight is: the columns
	// past 8198 exist only so that every rung's BN divides the width.
	jb := make([]float32, d.jointPad)
	copy(jb, m.Joint.Head.Bias)
	d.wbuf.WriteFloat32At(int(d.wJointBias), jb)
	for i, l := range m.Prediction.Layers {
		d.wbuf.WriteFloat32At(int(d.wLSTMBias[i]), l.Bias)
	}

	for _, r := range decProjOrder {
		off, ok := d.bOff[r]
		if !ok {
			continue
		}
		n, k := d.decProjShape(r)
		var src []float32
		switch r {
		case DecEnc:
			src = m.Projector.Weight
		case DecLSTM0, DecLSTM1:
			l := m.Prediction.Layers[0]
			if r == DecLSTM1 {
				l = m.Prediction.Layers[1]
			}
			src = lstmWeight(l, d.hidden)
		case DecPred:
			src = m.Prediction.Projector.Weight
		default:
			src = make([]float32, n*k)
			copy(src, m.Joint.Head.Weight)
		}
		if len(src) != n*k {
			return fmt.Errorf("parakeet: %s has %d weights, want %d", r, len(src), n*k)
		}
		dst := make([]uint16, n*k)
		packB(dst, src, n, k)
		d.bank.WriteUint16At(int(off), dst)
	}
	return nil
}

// lstmWeight concatenates a layer's two matrices along the reduction axis, so
// that a step is one [4H, 2H] product against [x ; h] rather than two [4H, H]
// products summed.
//
// PyTorch stores them apart because CuDNN does, and the gate order -- input,
// forget, cell, output -- is the same in both, so the concatenation is per
// row and the row order is untouched.
func lstmWeight(l *LSTMLayer, h int) []float32 {
	out := make([]float32, 4*h*2*h)
	for g := 0; g < 4*h; g++ {
		copy(out[g*2*h:], l.WIH[g*h:(g+1)*h])
		copy(out[g*2*h+h:], l.WHH[g*h:(g+1)*h])
	}
	return out
}

// Destroy releases every Vulkan object.
func (d *GPUDecoder) Destroy() {
	for _, p := range d.pipes {
		p.Destroy()
	}
	for _, p := range d.gemms {
		p.Destroy()
	}
	if d.encPipe != nil {
		d.encPipe.Destroy()
		d.encPipe = nil
	}
	for _, m := range d.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	d.pipes, d.gemms, d.mods = nil, nil, nil
	d.wbuf, d.abuf, d.hbuf, d.bank = nil, nil, nil, nil
}

// gemm records one projection of the tail: C[m, n] = A[m, k] * B[n, k].
func (d *GPUDecoder) gemm(dis []vk.MultiDispatch, kinds []string, r DecProj, aOff, cOff uint32, m, lda int) ([]vk.MultiDispatch, []string, error) {
	v, ok := d.kernels[d.plan[r]]
	if !ok {
		return nil, nil, fmt.Errorf("parakeet: %s has no pipeline for %q", r, d.plan[r])
	}
	n, k := d.decProjShape(r)
	m = roundUp(m, v.bm)
	if m > decTile && r != DecEnc {
		return nil, nil, fmt.Errorf("parakeet: %s wants %d rows, the arena holds %d", r, m, decTile)
	}
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, d.bOff[r]
	pc.GemmM, pc.GemmN, pc.GemmK = uint32(m), uint32(n), uint32(k)
	pc.LDA = uint32(lda)
	dis = append(dis, vk.MultiDispatch{
		Pipeline: d.gemms[d.plan[r]], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
		PushConstants: pc.bytes(),
	})
	return dis, append(kinds, "gemm "+string(r)), nil
}

// Attach lets the decoder read the encoder's hidden states out of the
// *encoder's* arena, so that the [T, 1024] tensor between the two never
// crosses the bus.
//
// It is one more pipeline: the same narrowing shader, built over a descriptor
// set whose activation binding is the encoder's buffer instead of this one's.
// The binding layout is a contract about roles, so nothing else changes -- the
// same argument that lets the encoder's position GEMM read its B operand out
// of the fp16 activation arena.
//
// What it buys is measured rather than assumed: reading 552 KB back out of a
// device-local host-visible buffer runs at 0.2 GB/s on this part (stage 3c's
// finding, which is the same number at 63 MB), so the copy GPUEncoder.ApplyMel
// does is 2.4 ms -- 5% of the whole pipeline, and half again what the entire
// decode loop costs.
func (d *GPUDecoder) Attach(g *GPUEncoder) error {
	if g.Dim() != d.encDim {
		return fmt.Errorf("parakeet: the encoder is %d wide, the projector takes %d", g.Dim(), d.encDim)
	}
	if g.MaxFrames() > d.frames {
		return fmt.Errorf("parakeet: the encoder holds %d frames, the decoder was built for %d", g.MaxFrames(), d.frames)
	}
	mod, err := d.dev.NewShaderModule(shaders.ParakeetNarrowF16)
	if err != nil {
		return fmt.Errorf("parakeet: attached narrow: %w", err)
	}
	d.mods = append(d.mods, mod)
	pipe, err := d.dev.NewPipeline(mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{d.wbuf, g.abuf, d.hbuf, d.bank},
		PushConstantSize: uint32(pushConstantSize),
	})
	if err != nil {
		return fmt.Errorf("parakeet: attached narrow: %w", err)
	}
	d.enc, d.encPipe = g, pipe
	return nil
}

// Transcribe runs the whole model over one clip: the encoder in g, then the
// tail, with nothing crossing the bus but the mel spectrogram in and two
// floats per emission out.
//
// It needs Attach, because the point of it is that the hidden states never
// come back to the host.
func (d *GPUDecoder) Transcribe(mel *Mat, validMel int) (*Transcript, error) {
	if d.enc == nil {
		return nil, fmt.Errorf("parakeet: the decoder is not attached to an encoder")
	}
	if _, err := d.enc.RunMel(mel, validMel); err != nil {
		return nil, err
	}
	return d.DecodeResident()
}

// DecodeResident is the tail over whatever the attached encoder's residual
// stream currently holds, which is what a caller that wants the encoder and
// the loop timed apart uses.
func (d *GPUDecoder) DecodeResident() (*Transcript, error) {
	if d.enc == nil {
		return nil, fmt.Errorf("parakeet: the decoder is not attached to an encoder")
	}
	if err := d.projectResident(); err != nil {
		return nil, err
	}
	return d.loop(d.enc.Valid())
}

// projectResident is Project reading the encoder's own residual stream.
func (d *GPUDecoder) projectResident() error {
	g := d.enc
	if g.Rows() <= 0 || g.Rows() > d.frames {
		return fmt.Errorf("parakeet: the encoder holds %d frames; the decoder was built for at most %d", g.Rows(), d.frames)
	}
	d.rows = g.Rows()
	d.rowsRun = roundUp(d.rows, decTile)
	return submit(d.projectGraph(d.encPipe, g.TensorX()))
}

// Project runs the encoder projector over the encoder's hidden states,
// leaving [T, 640] in the arena for the loop to read a row of at a time.
//
// It is the one part of the tail that is a throughput problem: T rows at
// once, on the same GEMM the encoder runs eleven of per layer.
func (d *GPUDecoder) Project(hidden *Mat) error {
	if hidden.Cols != d.encDim {
		return fmt.Errorf("parakeet: hidden states are %s, want [rows %d]", hidden, d.encDim)
	}
	if hidden.Rows <= 0 || hidden.Rows > d.frames {
		return fmt.Errorf("parakeet: %d frames; the decoder was built for at most %d", hidden.Rows, d.frames)
	}
	d.rows = hidden.Rows
	d.rowsRun = roundUp(d.rows, decTile)
	d.abuf.WriteFloat32At(int(d.aHidden), hidden.Data)
	return submit(d.projectGraph(d.pipes["narrow16"], d.aHidden))
}

// projectGraph is the projector's three dispatches, over whichever arena the
// hidden states are in: narrow into the fp16 A operand, the GEMM, the bias.
func (d *GPUDecoder) projectGraph(narrow *vk.ComputePipeline, src uint32) []vk.MultiDispatch {
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.WOff = src, d.hHidden, noW
	pc.Tokens, pc.Aux1 = uint32(d.rows), uint32(d.rowsRun)
	pc.Dim, pc.LDA = uint32(d.encDim), uint32(d.ldaEnc)
	dis := []vk.MultiDispatch{{Pipeline: narrow, GroupsX: uint32(d.rowsRun), GroupsY: 1, PushConstants: pc.bytes()}}

	dis, _, err := d.gemm(dis, nil, DecEnc, d.hHidden, d.aEnc, d.rowsRun, d.ldaEnc)
	if err != nil {
		// Unreachable: SetPlan has already checked that the rung exists and
		// that its BN divides the projector's 640 columns.
		panic(err)
	}
	var pb pushConstants
	pb.InOff, pb.OutOff, pb.WOff = d.aEnc, d.aEnc, d.wEncBias
	pb.Tokens, pb.Dim, pb.Aux1 = uint32(d.rows), uint32(d.hidden), uint32(d.rows)
	pb.Scale = math.Float32bits(1)
	return append(dis, vk.MultiDispatch{Pipeline: d.pipes["bias"], GroupsX: uint32(d.rows), GroupsY: 1, PushConstants: pb.bytes()})
}

// predictGraph is one step of the prediction network: both LSTM layers and
// the projection that follows them, over whatever h and c the arena holds.
//
// The caller has already written the token's embedding row into aX, which is
// a 2.5 KB host write rather than a device gather: the embedding table is
// [8193, 640] and putting it on the device would cost 21 MB to save a copy
// that is 0.2 microseconds.
func (d *GPUDecoder) predictGraph() ([]vk.MultiDispatch, []string, error) {
	var dis []vk.MultiDispatch
	var kinds []string
	var err error
	for l := 0; l < d.layers; l++ {
		x := d.aX
		if l > 0 {
			x = d.aH[l-1]
		}
		var pc pushConstants
		pc.InOff, pc.Aux0, pc.OutOff = x, d.aH[l], d.hLSTM
		pc.Tokens, pc.Dim, pc.LDA = decTile, uint32(d.hidden), uint32(d.ldaLSTM)
		dis = append(dis, vk.MultiDispatch{Pipeline: d.pipes["lstmin"], GroupsX: decTile, GroupsY: 1, PushConstants: pc.bytes()})
		kinds = append(kinds, "lstm in")

		if dis, kinds, err = d.gemm(dis, kinds, decLSTM(l), d.hLSTM, d.aGates, 1, d.ldaLSTM); err != nil {
			return nil, nil, err
		}
		var pg pushConstants
		pg.InOff, pg.WOff = d.aGates, d.wLSTMBias[l]
		pg.OutOff, pg.Aux0, pg.Dim = d.aH[l], d.aC[l], uint32(d.hidden)
		dis = append(dis, vk.MultiDispatch{Pipeline: d.pipes["lstmgate"], GroupsX: groups(d.hidden, 256), GroupsY: 1, PushConstants: pg.bytes()})
		kinds = append(kinds, "lstm gate")
	}
	var pn pushConstants
	pn.InOff, pn.OutOff, pn.WOff = d.aH[d.layers-1], d.hPred, noW
	pn.Tokens, pn.Aux1 = 1, decTile
	pn.Dim, pn.LDA = uint32(d.hidden), uint32(d.ldaHidden)
	dis = append(dis, vk.MultiDispatch{Pipeline: d.pipes["narrow16"], GroupsX: decTile, GroupsY: 1, PushConstants: pn.bytes()})
	kinds = append(kinds, "narrow lstm")

	return d.gemm(dis, kinds, DecPred, d.hPred, d.aPred, 1, d.ldaHidden)
}

// jointGraph is one frame's decision: the two towers added and rectified, the
// 8198-wide head, and the two argmaxes.
func (d *GPUDecoder) jointGraph(t int) ([]vk.MultiDispatch, []string, error) {
	var dis []vk.MultiDispatch
	var kinds []string
	var pc pushConstants
	pc.InOff = d.aEnc + uint32(t*d.hidden)
	pc.Aux0, pc.WOff, pc.OutOff = d.aPred, d.wPredBias, d.hSum
	pc.Tokens, pc.Dim, pc.LDA = decTile, uint32(d.hidden), uint32(d.ldaHidden)
	dis = append(dis, vk.MultiDispatch{Pipeline: d.pipes["jointsum"], GroupsX: decTile, GroupsY: 1, PushConstants: pc.bytes()})
	kinds = append(kinds, "joint sum")

	var err error
	if dis, kinds, err = d.gemm(dis, kinds, DecJoint, d.hSum, d.aLogits, 1, d.ldaHidden); err != nil {
		return nil, nil, err
	}
	var pa pushConstants
	pa.InOff, pa.WOff, pa.OutOff = d.aLogits, d.wJointBias, d.aResult
	pa.Dim, pa.Aux0 = uint32(d.vocab), uint32(len(d.durations))
	dis = append(dis, vk.MultiDispatch{Pipeline: d.pipes["argmax"], GroupsX: 1, GroupsY: 1, PushConstants: pa.bytes()})
	return dis, append(kinds, "argmax"), nil
}

// Reset zeroes the prediction network's carried state, which is what the
// first step of a clip starts from.
func (d *GPUDecoder) Reset() {
	zero := make([]float32, d.hidden)
	for l := 0; l < d.layers; l++ {
		d.abuf.WriteFloat32At(int(d.aH[l]), zero)
		d.abuf.WriteFloat32At(int(d.aC[l]), zero)
	}
}

// Decode runs the whole tail over the encoder's hidden states: the projector,
// then the TDT greedy loop.
//
// The three rules are Model.Decode's, and they are here rather than in a
// shader because each of them is control flow over a decision the device has
// already made:
//
//   - the prediction network advances only on a *non-blank* emission, so a run
//     of blanks costs one joint each and no LSTM step at all;
//   - the encoder cursor advances by the predicted duration rather than by one
//     frame, and a blank predicting duration 0 is forced to 1 so the loop
//     cannot stall;
//   - decoding stops when the cursor passes the last valid frame, with
//     max_symbols_per_step frames' worth of emissions as the backstop.
func (d *GPUDecoder) Decode(hidden *Mat, valid int) (*Transcript, error) {
	if valid < 0 || valid > hidden.Rows {
		return nil, fmt.Errorf("parakeet: %d valid frames of %d", valid, hidden.Rows)
	}
	if err := d.Project(hidden); err != nil {
		return nil, err
	}
	return d.loop(valid)
}

// loop is the TDT greedy loop over whatever projected frames the arena holds.
func (d *GPUDecoder) loop(valid int) (*Transcript, error) {
	d.Reset()

	out := &Transcript{Frames: valid}
	limit := d.maxSymbols * valid
	prev, fresh := d.blank, true
	d.Steps, d.Submits = 0, 0

	for t := 0; t < valid && len(out.Steps) < limit; {
		var dis []vk.MultiDispatch
		if fresh || prev != d.blank {
			if err := d.writeEmbedding(prev); err != nil {
				return nil, err
			}
			step, _, err := d.predictGraph()
			if err != nil {
				return nil, err
			}
			dis, fresh = append(dis, step...), false
		}
		joint, _, err := d.jointGraph(t)
		if err != nil {
			return nil, err
		}
		dis = append(dis, joint...)
		if err := submit(dis); err != nil {
			return nil, err
		}
		d.Submits++

		res := d.abuf.ReadFloat32At(int(d.aResult), 2)
		token, di := int(res[0]), int(res[1])
		if token < 0 || token >= d.vocab {
			return nil, fmt.Errorf("parakeet: the joint chose token %d of %d", token, d.vocab)
		}
		if di < 0 || di >= len(d.durations) {
			return nil, fmt.Errorf("parakeet: the joint chose duration index %d of %d", di, len(d.durations))
		}
		duration := d.durations[di]
		if token == d.blank && duration == 0 {
			duration = 1
		}
		out.Steps = append(out.Steps, Step{Frame: t, Token: token, Duration: duration})
		if token != d.blank {
			out.Tokens = append(out.Tokens, token)
		}
		prev = token
		t += duration
	}
	d.Steps = len(out.Steps)

	text, err := d.m.Tokenizer.Decode(out.Tokens)
	if err != nil {
		return nil, err
	}
	out.Text = text
	return out, nil
}

// writeEmbedding copies one row of the embedding table into the arena as the
// first LSTM layer's input.
func (d *GPUDecoder) writeEmbedding(token int) error {
	p := d.m.Prediction
	if token < 0 || token >= p.Vocab {
		return fmt.Errorf("parakeet: token %d outside a vocabulary of %d", token, p.Vocab)
	}
	d.abuf.WriteFloat32At(int(d.aX), p.Embedding[token*d.hidden:(token+1)*d.hidden])
	return nil
}

// Read copies a [rows, cols] tensor out of the fp32 arena.
func (d *GPUDecoder) Read(off uint32, rows, cols int) *Mat {
	out := NewMat(rows, cols)
	copy(out.Data, d.abuf.ReadFloat32At(int(off), rows*cols))
	return out
}

// The tail's tensors, for the stagewise validation.
func (d *GPUDecoder) TensorEnc() uint32    { return d.aEnc }
func (d *GPUDecoder) TensorH(l int) uint32 { return d.aH[l] }
func (d *GPUDecoder) TensorC(l int) uint32 { return d.aC[l] }
func (d *GPUDecoder) TensorPred() uint32   { return d.aPred }
func (d *GPUDecoder) TensorLogits() uint32 { return d.aLogits }
func (d *GPUDecoder) Hidden() int          { return d.hidden }
func (d *GPUDecoder) Layers() int          { return d.layers }
func (d *GPUDecoder) JointOut() int        { return d.jointOut }
func (d *GPUDecoder) Rows() int            { return d.rows }

// WeightBytes is what the staged tail costs on the device and
// ActivationBytes what its arenas do.
func (d *GPUDecoder) WeightBytes() int     { return d.wbuf.Size() + d.bank.Size() }
func (d *GPUDecoder) ActivationBytes() int { return d.abuf.Size() + d.hbuf.Size() }

// Step advances the prediction network by one token and leaves its projected
// output in the arena, which is what the stagewise test walks.
func (d *GPUDecoder) Step(token int) error {
	if err := d.writeEmbedding(token); err != nil {
		return err
	}
	dis, _, err := d.predictGraph()
	if err != nil {
		return err
	}
	return submit(dis)
}

// Joint runs one frame's joint and argmax over whatever the prediction
// network last left in the arena.
func (d *GPUDecoder) Joint(t int) error {
	dis, _, err := d.jointGraph(t)
	if err != nil {
		return err
	}
	return submit(dis)
}

// Labels lists a full emission's dispatches in order, for the profiler.
func (d *GPUDecoder) Labels() []string {
	_, pred, err := d.predictGraph()
	if err != nil {
		return nil
	}
	_, joint, err := d.jointGraph(0)
	if err != nil {
		return nil
	}
	return append(pred, joint...)
}
