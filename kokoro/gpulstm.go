package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// lstmPad is the extra halves on the input projection's fp16 row stride, and
// it does the same two jobs bertPad does: it keeps the leading dimension off
// a 4 KB multiple (§2.3), and column `In` of every row holds a 1 so the
// summed bias rides in the weight bank as an extra column of K.
//
// 64 rather than one tile, because dit_gemm.comp's K loop steps by BK.
const lstmPad = 64

// lstmVariant is the rung the input projection runs on: M is the sequence
// length, fifty or a hundred and thirty, which is the narrow-M regime every
// other short-sequence GEMM in this package picked a 32-wide tile for.
var lstmVariant = convVariant{name: ConvKernel("gemm_reg32x32_w32"),
	spirv: shaders.DiTGEMMReg32x32TiledW32, bm: 32, bn: 32, wave: 32}

// StepKernel names one build of kokoro_lstm.comp, by how many cells a
// workgroup owns.
type StepKernel string

const (
	StepCells64U4 StepKernel = "lstm_c64_u4"
	StepCells128  StepKernel = "lstm_c128"
	StepCells64   StepKernel = "lstm_c64"
	StepCells32   StepKernel = "lstm_c32"
)

// stepVariants is the step ladder, over the two knobs the step kernel has.
//
// `cells` is how many cells a workgroup owns, which decides how many
// workgroups a step is spread across; `unroll` is how many taps of the
// reduction one thread has in flight. The sweep is worth reading in that
// order, because only one of them moves the number — see
// shaders/kokoro_lstm.comp.
var stepVariants = []struct {
	name          StepKernel
	spirv         []byte
	cells, unroll int
}{
	{StepCells64U4, shaders.KokoroLSTMC64U4, 64, 4},
	{StepCells128, shaders.KokoroLSTMC128, 128, 32},
	{StepCells64, shaders.KokoroLSTMC64, 64, 32},
	{StepCells32, shaders.KokoroLSTMC32, 32, 32},
}

// DefaultStepKernel is the rung TestGPULSTMLadder measures as the winner.
const DefaultStepKernel = StepCells64

// StepKernels lists the rungs, for a sweep.
func StepKernels() []StepKernel {
	out := make([]StepKernel, 0, len(stepVariants))
	for _, v := range stepVariants {
		out = append(out, v.name)
	}
	return out
}

// GPULSTM runs one single-layer bidirectional nn.LSTM on the device.
//
// Six of these are the whole of what is left on kokoro's phoneme side: after
// T6a put ALBERT on the device and T6b the AdaIN stacks, timing `Prosody`
// against the recurrences measured **73 ms of a 76 ms phoneme side** in these
// six objects. They are all the same shape — hidden 256 a direction, 1024
// gates, an input of 640 or 512 — and differ only in sequence length.
//
// # The shape of the problem, and the one algebraic move
//
// A recurrence is a latency problem: step t+1 cannot start until step t
// finished, so there is no tile to widen and no batch to find. What there
// *is* is a term that does not belong in the loop at all. Torch's step is
// `W_ih x_t + W_hh h_{t-1} + b`, and the first term depends on nothing the
// loop produces — so `P = X W_ih^T + b` is computed for the whole sequence up
// front, as one GEMM with **both directions' weights concatenated along N**,
// and the loop is left with `W_hh h`.
//
// That is worth more than the arithmetic it moves. The per-step B operand
// goes from `[4H, In+H]` at 1.8 MB to `[H, 4H]` at 0.5 MB, which is small
// enough to stay in the MALL across every step of every LSTM here
// (research/5.1b-mall-cliff-and-stride.md: 805-965 GB/s resident against 236
// from DRAM), and it removes the per-step pass that would otherwise have to
// assemble `[x_t | h_{t-1}]` as an fp16 operand.
//
// # What crosses the bus
//
// One upload of [T, In] and one download of [T, 2H] per call. The download is
// the expensive half — this device reads host-visible memory at 210 MB/s
// (T4a) — which is why `Attach` exists: chained through it, an LSTM's output
// *is* the next object's input and the only readback is the last one's.
type GPULSTM struct {
	dev *vk.Device

	in, hidden, maxT int
	tokens           int
	ldaIn            int // fp16 row stride of the projection's A operand

	// The arena this object's fp32 space came from, when it is shared.
	shared *sharedArena

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline
	gemm                   *vk.ComputePipeline

	// Arena offsets, fp32.
	aX, aP, aH, aState uint32
	actElems           int
	// Where the input is read from and the output written to. Both default to
	// aX and aH and are redirected by Attach.
	srcOff, srcLDA uint32
	dstOff, dstLDA uint32

	hA       uint32
	hElems   int
	bIH, wHH uint32

	// narrowW is how many of the `in` input channels come from the fp32
	// source; the rest are a constant written into the fp16 operand once.
	narrowW int
	tail    []float32
	leaky   bool

	step  StepKernel
	steps map[StepKernel]*vk.ComputePipeline
}

// NewGPULSTM stages one LSTM for sequences of at most maxT.
func NewGPULSTM(dev *vk.Device, l *LSTM, maxT int) (*GPULSTM, error) {
	return newGPULSTM(dev, l, maxT, nil)
}

func newGPULSTM(dev *vk.Device, l *LSTM, maxT int, sa *sharedArena) (*GPULSTM, error) {
	if maxT <= 0 {
		return nil, fmt.Errorf("kokoro: lstm over %d timesteps", maxT)
	}
	// One thread owns one cell and all four of its gates, so the hidden width
	// is the workgroup and cannot exceed it; the four-tap unroll of the
	// reduction wants it a multiple of four. Every LSTM in this model is 256.
	if l.Hidden > lstmThreads || l.Hidden%4 != 0 {
		return nil, fmt.Errorf("kokoro: lstm hidden %d, want a multiple of 4 at most %d",
			l.Hidden, lstmThreads)
	}
	g := &GPULSTM{
		dev: dev, in: l.In, hidden: l.Hidden, maxT: maxT, shared: sa,
		pipes: map[string]*vk.ComputePipeline{},
		steps: map[StepKernel]*vk.ComputePipeline{},
		step:  DefaultStepKernel,
	}
	g.ldaIn = l.In + lstmPad
	g.narrowW = l.In
	if err := g.alloc(); err != nil {
		g.Destroy()
		return nil, err
	}
	if sa != nil {
		// The buffer does not exist yet; the caller commits and calls finish.
		return g, nil
	}
	if err := g.finish(l); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// lstmThreads is kokoro_lstm.comp's workgroup, and therefore the widest
// hidden state it can run.
const lstmThreads = 256

func (g *GPULSTM) alloc() error {
	h4 := 4 * g.hidden
	rows := roundUp(g.maxT, framePad)

	var base, off uint32
	if g.shared != nil {
		base = g.shared.next
	}
	take := func(n int) uint32 { o := base + off; off += uint32(n); return o }
	g.aX = take(rows * g.in)
	g.aP = take(rows * 2 * h4)
	g.aH = take(rows * 2 * g.hidden)
	// Three vectors a direction: h double-buffered, because two workgroups
	// of one step cannot be ordered against each other, and c.
	g.aState = take(2 * 3 * g.hidden)
	g.actElems = int(off)
	g.srcOff, g.srcLDA = g.aX, uint32(g.in)
	g.dstOff, g.dstLDA = g.aH, uint32(2*g.hidden)

	g.hA = 0
	g.hElems = rows * g.ldaIn

	// The bank holds two layouts: the input projection as B_LAYOUT=2 fragment
	// tiles, both directions stacked along N, and the recurrent weights as
	// plain k-major rows the step kernel indexes itself.
	g.bIH = 0
	halves := 2 * h4 * g.ldaIn
	g.wHH = uint32(halves)
	halves += 2 * g.hidden * h4

	var err error
	if g.shared != nil {
		g.shared.reserve(g.actElems)
	} else if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("kokoro: lstm fp32 arena: %w", err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: lstm fp16 arena: %w", err)
	}
	// Nothing in this object needs an fp32 weight: the bias is a column of
	// the bank. The buffer exists because the pipeline layout binds four.
	if g.wbuf, err = g.dev.NewBuffer(4); err != nil {
		return fmt.Errorf("kokoro: lstm fp32 weights: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("kokoro: lstm weight bank: %w", err)
	}
	return nil
}

// finish builds the pipelines and writes the weights, which cannot happen
// until every arena exists.
func (g *GPULSTM) finish(l *LSTM) error {
	if g.shared != nil {
		g.abuf = g.shared.buf
	}
	if err := g.build(); err != nil {
		return err
	}
	return g.stage(l)
}

func (g *GPULSTM) build() error {
	arenas := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, n := range []struct {
		name  string
		spirv []byte
	}{{"narrow", shaders.KokoroDNarrow}, {"leaky", shaders.KokoroDLeaky}} {
		mod, err := g.dev.NewShaderModule(n.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", n.name, err)
		}
		g.mods = append(g.mods, mod)
		if g.pipes[n.name], err = g.dev.NewPipeline(mod, spec); err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", n.name, err)
		}
	}
	var err error
	for _, v := range stepVariants {
		if g.hidden%v.cells != 0 {
			continue
		}
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", v.name, err)
		}
		g.steps[v.name] = pipe
	}
	if _, ok := g.steps[g.step]; !ok {
		return fmt.Errorf("kokoro: no step kernel divides a hidden width of %d", g.hidden)
	}
	s := spec
	s.RequiredSubgroupSize = lstmVariant.wave
	ok, err := canPinWave(g.dev, lstmVariant.wave)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("kokoro: this device cannot pin a %d-wide subgroup", lstmVariant.wave)
	}
	gm, err := g.dev.NewShaderModule(lstmVariant.spirv)
	if err != nil {
		return fmt.Errorf("kokoro: shader %s: %w", lstmVariant.name, err)
	}
	g.mods = append(g.mods, gm)
	if g.gemm, err = g.dev.NewPipeline(gm, s); err != nil {
		return fmt.Errorf("kokoro: pipeline %s: %w", lstmVariant.name, err)
	}
	return nil
}

// stage writes both directions' weights.
//
// The input projections are concatenated along N — forward's 4H rows then
// reverse's — so one GEMM covers both, and the summed bias goes in as the
// column of K that `lstmPad` reserved. The recurrent weights are transposed
// to k-major, which is what makes a wave's 32 lanes read 32 consecutive
// halves inside the step loop.
func (g *GPULSTM) stage(l *LSTM) error {
	h4 := 4 * g.hidden
	if l.In != g.in || l.Hidden != g.hidden {
		return fmt.Errorf("kokoro: lstm is %d->%d, staged for %d->%d", l.In, l.Hidden, g.in, g.hidden)
	}
	dirs := [2]*LSTMDirection{l.Fwd, l.Rev}
	for _, d := range dirs {
		if len(d.WIH) != h4*g.in || len(d.WHH) != h4*g.hidden || len(d.Bias) != h4 {
			return fmt.Errorf("kokoro: lstm direction has the wrong weight sizes")
		}
	}

	bank := make([]uint16, g.bank.Size()/2)
	// [2*4H, In] and [2*4H] as one projection, so the concatenation along N
	// costs nothing at run time.
	wih := make([]float32, 2*h4*g.in)
	bias := make([]float32, 2*h4)
	for i, d := range dirs {
		copy(wih[i*h4*g.in:], d.WIH)
		copy(bias[i*h4:], d.Bias)
	}
	packLinearB(bank[g.bIH:], wih, bias, 2*h4, g.in, g.ldaIn)

	// [4H, H] row-major to [H, 4H] k-major, per direction.
	hh := bank[g.wHH:]
	for i, d := range dirs {
		dst := hh[i*g.hidden*h4:]
		parallelFor(g.hidden, func(k int) {
			for j := 0; j < h4; j++ {
				dst[k*h4+j] = safetensors.F32ToF16(d.WHH[j*g.hidden+k])
			}
		})
	}
	g.bank.WriteUint16At(0, bank)

	return g.writeOperand()
}

// writeOperand lays out everything in the fp16 A operand that is not data.
//
// Column `In` of every row is the 1 the summed bias rides on, and columns
// [narrowW, In) are whatever SetTail fixed — the style vector, for five of
// this model's six recurrences. The narrow pass only ever writes
// [0, narrowW), so both survive for the life of the object.
func (g *GPULSTM) writeOperand() error {
	h := make([]uint16, g.hElems)
	one := safetensors.F32ToF16(1)
	tail := make([]uint16, len(g.tail))
	for i, v := range g.tail {
		tail[i] = safetensors.F32ToF16(v)
	}
	for r := 0; r < roundUp(g.maxT, framePad); r++ {
		copy(h[r*g.ldaIn+g.narrowW:], tail)
		h[r*g.ldaIn+g.in] = one
	}
	g.hbuf.WriteUint16At(0, h)
	return nil
}

// SetTail fixes the last len(v) input channels to a constant for every
// timestep, so the fp32 source only has to carry the first `In - len(v)`.
//
// This is what the style concatenation becomes. Five of the six recurrences
// here take `[h | style]`, 512 channels of activation beside 128 that are the
// same in every row of every utterance — so torch's `Concat(h, s)` is a
// [T, 640] allocation and a copy per timestep, and here it is 128 halves
// written once into an operand whose row stride was already padded to hold
// them. **A constant input channel is part of the operand, not part of the
// data**, which is stage 9's bias trick one step further: the bias is the
// special case where the constant is 1.
//
// It also shortens the chain in front of it. The duration encoder's
// normalisations then run over a 512-wide row rather than the first 512
// columns of a 640-wide one, which is the shape parakeet's LayerNorm kernel
// already reads.
func (g *GPULSTM) SetTail(v []float32) error {
	if len(v) > g.in {
		return fmt.Errorf("kokoro: a tail of %d channels on a %d-wide input", len(v), g.in)
	}
	g.tail = append([]float32(nil), v...)
	g.narrowW = g.in - len(v)
	if g.srcOff == g.aX {
		g.srcLDA = uint32(g.narrowW)
	}
	return g.writeOperand()
}

// SetNarrowLeaky rectifies the fp32 source on its way into the operand, which
// is what the text encoder's recurrence wants: its input is the last
// convolution block's output and upstream applies a leaky ReLU to it. Fusing
// it into the narrow that was happening anyway saves a pass over [T, 512].
func (g *GPULSTM) SetNarrowLeaky(on bool) { g.leaky = on }

func stepVariantFor(k StepKernel) (struct {
	name          StepKernel
	spirv         []byte
	cells, unroll int
}, bool) {
	for _, v := range stepVariants {
		if v.name == k {
			return v, true
		}
	}
	return stepVariants[0], false
}

// StepKernel is the rung the timestep runs on.
func (g *GPULSTM) StepKernel() StepKernel { return g.step }

// SetStepKernel switches rungs, which needs no restaging: the recurrent
// weights are k-major rows and every rung reads those.
func (g *GPULSTM) SetStepKernel(k StepKernel) error {
	if _, ok := g.steps[k]; !ok {
		return fmt.Errorf("kokoro: no step pipeline for %q", k)
	}
	g.step = k
	return nil
}

// Attach redirects this LSTM's input to a region another object owns, so the
// [T, In] between them never crosses the bus. Both must live in the same
// shared arena.
func (g *GPULSTM) Attach(src uint32, lda int) {
	g.srcOff, g.srcLDA = src, uint32(lda)
}

// Output is where this LSTM writes, as an offset into the fp32 arena and a
// row stride, so a consumer can read it in place.
func (g *GPULSTM) Output() (uint32, int) { return g.dstOff, int(g.dstLDA) }

// In is the input width, and Hidden the width of one direction.
func (g *GPULSTM) In() int     { return g.in }
func (g *GPULSTM) Hidden() int { return g.hidden }

// graph is the whole recurrence: the narrow, the input projection, and one
// dispatch per timestep covering both directions.
//
// T+2 dispatches, all of them in one command buffer. Nothing here is decided
// between steps — the sequence length is known before the first one — which
// is the difference from S8's transducer loop and the reason there is no
// round trip in the middle.
func (g *GPULSTM) graph() ([]vk.MultiDispatch, []string, error) {
	t := uint32(g.tokens)
	m := uint32(roundUp(g.tokens, lstmVariant.bm))
	h4 := uint32(4 * g.hidden)
	if 2*h4%uint32(lstmVariant.bn) != 0 {
		return nil, nil, fmt.Errorf("kokoro: tile width %d does not divide %d gates",
			lstmVariant.bn, 2*h4)
	}

	var dis []vk.MultiDispatch
	var kinds []string
	add := func(pipe *vk.ComputePipeline, kind string, gx, gy uint32, pc pushConstants) {
		dis = append(dis, vk.MultiDispatch{Pipeline: pipe, GroupsX: gx, GroupsY: gy,
			PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	narrow := "narrow"
	if g.leaky {
		narrow = "leaky"
	}
	pc := pushConstants{Dim: uint32(g.narrowW), InOff: g.srcOff, OutOff: g.hA,
		Aux1: g.srcLDA, LDA: uint32(g.ldaIn), Scale: math.Float32bits(0.2)}
	add(g.pipes[narrow], narrow, groups(g.narrowW, 256), t, pc)

	pc = pushConstants{InOff: g.hA, OutOff: g.aP, BOff: g.bIH,
		GemmM: m, GemmN: 2 * h4, GemmK: uint32(g.ldaIn), LDA: uint32(g.ldaIn)}
	add(g.gemm, "project", 2*h4/uint32(lstmVariant.bn), m/uint32(lstmVariant.bm), pc)

	v, ok := stepVariantFor(g.step)
	if !ok {
		return nil, nil, fmt.Errorf("kokoro: no step variant %q", g.step)
	}
	bands := uint32(g.hidden / v.cells)
	for s := 0; s < g.tokens; s++ {
		pc = pushConstants{
			InOff: g.aP, LDA: 2 * h4,
			OutOff: g.dstOff, LDB: g.dstLDA,
			Aux0: g.aState, Aux1: uint32(s), Aux2: uint32(s % 2), BOff: g.wHH,
			Tokens: t, Dim: uint32(g.hidden),
		}
		add(g.steps[g.step], "step", bands, 2, pc)
	}
	return dis, kinds, nil
}

// Graph records the recurrence for a sequence of t timesteps, for a caller
// that is chaining several objects into one submit.
func (g *GPULSTM) Graph(t int) ([]vk.MultiDispatch, []string, error) {
	if t <= 0 || t > g.maxT {
		return nil, nil, fmt.Errorf("kokoro: %d timesteps against an arena for %d", t, g.maxT)
	}
	g.tokens = t
	g.Reset()
	return g.graph()
}

// Reset zeroes the carried state, which every call has to do: h and c start
// at zero in both directions, and this object is reused across utterances.
//
// It is a host *write* of 4 KB rather than a dispatch, because writes to
// device-local host-visible memory run at 29 GB/s where reads run at 210 MB/s
// (T4a) — the asymmetry that decides almost every boundary in this package.
func (g *GPULSTM) Reset() {
	g.abuf.WriteFloat32At(int(g.aState), make([]float32, 6*g.hidden))
}

// Upload writes the input sequence.
func (g *GPULSTM) Upload(x *Mat) error {
	if x.Cols != g.narrowW {
		return fmt.Errorf("kokoro: lstm reads %d channels from the arena, got %d",
			g.narrowW, x.Cols)
	}
	if x.Rows <= 0 || x.Rows > g.maxT {
		return fmt.Errorf("kokoro: %d timesteps against an arena for %d", x.Rows, g.maxT)
	}
	g.tokens = x.Rows
	g.abuf.WriteFloat32At(int(g.aX), x.Data)
	return nil
}

// Download reads the output, [T, 2H] with forward in the first H channels and
// reverse in the last — torch's concatenation order.
func (g *GPULSTM) Download() *Mat {
	out := NewMat(g.tokens, 2*g.hidden)
	row := g.abuf.ReadFloat32At(int(g.dstOff), g.tokens*int(g.dstLDA))
	for r := 0; r < g.tokens; r++ {
		copy(out.Row(r), row[r*int(g.dstLDA):])
	}
	return out
}

// Apply runs the whole recurrence over one sequence: one upload, one submit,
// one download.
func (g *GPULSTM) Apply(x *Mat) (*Mat, error) {
	if err := g.Upload(x); err != nil {
		return nil, err
	}
	g.Reset()
	dis, _, err := g.graph()
	if err != nil {
		return nil, err
	}
	if err := submit(dis); err != nil {
		return nil, err
	}
	return g.Download(), nil
}

// Profile times the narrow, the projection, one step and the whole loop on
// the device.
//
// One step rather than all of them, because they are identical and a table of
// three hundred identical rows says nothing the one row does not — but the
// loop is timed as a loop too, because a step measured alone is not a step
// measured on the critical path of three hundred barriers.
func (g *GPULSTM) Profile(t int) ([]Stage, error) {
	g.tokens = t
	g.Reset()
	dis, kinds, err := g.graph()
	if err != nil {
		return nil, err
	}
	// The first dispatch of a fresh object pays for caches nothing has warmed
	// and would otherwise be charged to whichever row happened to be first.
	if _, err := vk.DispatchMultiTimed(dis, 1, 2, true); err != nil {
		return nil, err
	}
	out := make([]Stage, 0, 4)
	for i := 0; i < 3 && i < len(dis); i++ {
		d, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 32, true)
		if err != nil {
			return nil, err
		}
		out = append(out, Stage{Kind: kinds[i], Time: d.Seconds() / 32})
	}
	d, err := g.LoopTime(4)
	if err != nil {
		return nil, err
	}
	return append(out, Stage{Kind: "loop", Time: d}), nil
}

// LoopTime is the GPU time of the whole recurrence — the narrow, the
// projection and every step — over `iters` runs, best-effort warmed.
func (g *GPULSTM) LoopTime(iters int) (float64, error) {
	dis, _, err := g.graph()
	if err != nil {
		return 0, err
	}
	if _, err := vk.DispatchMultiTimed(dis, 1, 1, true); err != nil {
		return 0, err
	}
	d, err := vk.DispatchMultiTimed(dis, 1, uint32(iters), true)
	if err != nil {
		return 0, err
	}
	return d.Seconds() / float64(iters), nil
}

// Destroy releases every Vulkan object.
func (g *GPULSTM) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, p := range g.steps {
		p.Destroy()
	}
	if g.gemm != nil {
		g.gemm.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	bufs := []*vk.Buffer{g.wbuf, g.hbuf, g.bank}
	if g.shared == nil {
		bufs = append(bufs, g.abuf)
	}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.steps, g.mods, g.gemm = nil, nil, nil, nil
	g.wbuf, g.abuf, g.hbuf, g.bank = nil, nil, nil, nil
}
