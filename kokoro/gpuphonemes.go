package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// GPUPhonemes is the whole of kokoro's phoneme side on the device: both
// encoders, the duration head, the length regulator and every recurrence
// between them (SPEECH.md T6d).
//
// # What this stage is, and what it is not
//
// T6c put the six bidirectional LSTMs on the device and took them from 73 ms
// to 6. But only **1.4 ms of that 6 was GPU time**: the rest was six
// readbacks, 776 KB at the 210 MB/s this device reads host-visible memory at,
// because each recurrence returned its output so the host could do the three
// lines between it and the next one. This object is those three lines.
//
// So almost nothing here is arithmetic. It is a chain of offsets, and the
// only new kernel in it is a gather.
//
// # The chain
//
//	dEn ─▶ lstm0 ─▶ norm0 ─▶ lstm1 ─▶ norm1 ─▶ lstm2 ─▶ norm2 ─┬─▶ lstm3 ─▶ (head, on the host)
//	                                                            │
//	                                        (the durations, on the host)
//	                                                            │
//	                                                            └─▶ gather ─▶ shared ─▶ F0/N
//
//	ids ─▶ embedding ─▶ (conv ─▶ norm ─▶ leaky) x3 ─▶ lstm4 ─▶ t_en
//
// Two things leave the device, both [T, 512] and 102 KB: the last
// recurrence's output on the duration path, and the text encoder's.
//
// **The duration head is on the host on purpose, and it is the one place in
// this model where fp16 is not good enough.** It was on the device first —
// one narrow and one GEMM, 13 KB back instead of 102 — and a duration moved.
// Not by rounding noise: by **0.29 of a frame**, where T6a's whole ALBERT port
// moved the largest of them by 0.005. The head's logits have an rms of 27, so
// an fp16 operand's thousandth is two hundredths of a logit, and a duration is
// the *sum of fifty sigmoids* of those logits — a soft count whose terms near
// zero have a derivative of a quarter and whose total is then **rounded**.
// Every frame boundary in the utterance is placed by that rounding. So the
// [T, 512] comes back and `Predictor.Durations` runs unchanged, in fp32, on
// the same code the CPU reference uses.
//
// Expanding `t_en` on the host is the cheaper direction for a different
// reason: the vocoder still takes its input from there, and [T, 512] is less
// to read than the [L, 512] a device-side expansion would produce — 102 KB
// against 266.
//
// # Three things that do not need a kernel
//
//   - **The style concatenation is part of the operand.** Five of the six
//     recurrences take `[h | style]`; see GPULSTM.SetTail. That removes a
//     [T, 640] allocation and a copy per timestep, and it leaves every
//     normalisation running over a 512-wide row, which is the shape
//     parakeet's LayerNorm kernel already reads.
//   - **AdaLayerNorm is parakeet's LayerNorm.** `(1+gamma) * norm(x) + beta`
//     with gamma and beta from a [2C, 128] projection of one vector is a
//     LayerNorm with a weight and a bias — computed on the host once per
//     utterance, exactly as every AdaIN in the vocoder already is.
//   - **The length regulator is a gather**, and the indices are floats in the
//     activation arena. See kokoro_gather.comp.
type GPUPhonemes struct {
	dev   *vk.Device
	arena *sharedArena

	cfg            *Config
	maxTok, frames int
	tokens         int
	hidden         int // a recurrence's output width, 2*Hidden

	// The six recurrences, in the order an utterance runs them.
	dur    [3]*GPULSTM
	durLST *GPULSTM
	shared *GPULSTM
	text   *GPULSTM

	// The F0 and energy stacks, which are the rest of the phoneme side. They
	// live here rather than beside the vocoder because their input is
	// `shared`'s output, and sharing an arena with it is what keeps the
	// [L, 512] between them off the bus.
	stacks *GPUProsody

	wbuf, hbuf, bank *vk.Buffer
	abuf             *vk.Buffer
	mods             []*vk.ShaderModule
	pipes            map[string]*vk.ComputePipeline
	conv             *vk.ComputePipeline

	// Arena offsets, fp32.
	aDEn      uint32
	aIdx, aEn uint32
	aTE       uint32
	actElems  int

	// fp16: the text encoder convolutions' bordered A operand.
	hTE            uint32
	ldTE           int
	hElems         int
	bConv          [3]uint32
	wNormG, wNormB [3]uint32 // the duration encoder's style affines
	wTEG, wTEB     [3]uint32 // the text encoder's own
	wTEBias        [3]uint32 // its convolution biases

	// The predictor, held so SetStyle can reach the AdaLayerNorms' fc.
	pred *Predictor
	te   *TextEncoder
}

// tePad is the extra halves on the text encoder convolutions' fp16 row
// stride. It is zero: the stride is the 512 channels themselves, which is
// already a multiple of dit_gemm.comp's BK, and A_CONV=2 needs the tap index
// to roll over at exactly the channel count.
const tePad = 0

// NewGPUPhonemes stages the whole phoneme side for a given alignment frame
// count.
//
// Only `shared` and the length regulator depend on that count; everything
// before them runs at the token rate and is sized for the model's own
// 512-token limit, so the object is staged once for any utterance the way
// PL-BERT is.
func NewGPUPhonemes(dev *vk.Device, m *Model, frames int) (*GPUPhonemes, error) {
	g := &GPUPhonemes{
		dev: dev, cfg: m.Config, frames: frames,
		maxTok: m.BERT.Config.MaxPositionEmbed,
		pipes:  map[string]*vk.ComputePipeline{},
		pred:   m.Predictor, te: m.TextEncoder,
	}
	if frames <= 0 {
		return nil, fmt.Errorf("kokoro: phoneme side over %d frames", frames)
	}
	g.arena = &sharedArena{dev: dev}
	if err := g.build(m); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// build lays out every arena, commits the one buffer they all point into, and
// stages the weights.
//
// The order is forced: the recurrences reserve their own space, this object
// reserves what sits between them, and only then does the buffer exist — so
// the pipelines, which bind it, are the last thing built.
func (g *GPUPhonemes) build(m *Model) error {
	p := m.Predictor
	g.hidden = 2 * p.Shared.Hidden

	// The recurrences, each reserving its own input projection and state.
	var err error
	for i := range g.dur {
		if g.dur[i], err = newGPULSTM(g.dev, p.TextEncoder.LSTMs[i], g.maxTok, g.arena); err != nil {
			return fmt.Errorf("kokoro: duration recurrence %d: %w", i, err)
		}
	}
	if g.durLST, err = newGPULSTM(g.dev, p.LSTM, g.maxTok, g.arena); err != nil {
		return fmt.Errorf("kokoro: duration head recurrence: %w", err)
	}
	if g.shared, err = newGPULSTM(g.dev, p.Shared, g.frames, g.arena); err != nil {
		return fmt.Errorf("kokoro: shared recurrence: %w", err)
	}
	if g.text, err = newGPULSTM(g.dev, m.TextEncoder.LSTM, g.maxTok, g.arena); err != nil {
		return fmt.Errorf("kokoro: text encoder recurrence: %w", err)
	}
	if g.stacks, err = newGPUProsody(g.dev, p, g.frames, DefaultDecoderKernel, g.arena); err != nil {
		return fmt.Errorf("kokoro: F0/N stacks: %w", err)
	}

	// This object's own fp32 regions.
	rows := roundUp(g.maxTok, framePad)
	g.aDEn = g.arena.reserve(rows * g.hidden)
	g.aIdx = g.arena.reserve(roundUp(g.frames, framePad))
	g.aEn = g.arena.reserve(roundUp(g.frames, framePad) * g.hidden)
	g.aTE = g.arena.reserve(rows * g.te.Channels)
	g.actElems = int(g.arena.next)

	// fp16: the convolutions' bordered A operand.
	g.ldTE = g.te.Channels + tePad
	g.hTE = uint32(convBorder * g.ldTE)
	g.hElems = (2*convBorder + rows) * g.ldTE

	// fp32 weights: three style affines, three of the text encoder's own, and
	// its three convolution biases.
	var offW uint32
	takeW := func(n int) uint32 { o := offW; offW += uint32(n); return o }
	c := g.te.Channels
	for i := range g.wNormG {
		g.wNormG[i], g.wNormB[i] = takeW(g.hidden), takeW(g.hidden)
	}
	for i := range g.wTEG {
		g.wTEG[i], g.wTEB[i] = takeW(c), takeW(c)
		g.wTEBias[i] = takeW(c)
	}

	// fp16 weight bank: the three convolutions.
	halves := 0
	for i := range g.bConv {
		g.bConv[i] = uint32(halves)
		halves += c * g.te.Convs[i].Kernel * g.ldTE
	}

	if err := g.arena.commit(); err != nil {
		return err
	}
	g.abuf = g.arena.buf
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: phoneme fp16 arena: %w", err)
	}
	if g.wbuf, err = g.dev.NewBuffer(int(offW) * 4); err != nil {
		return fmt.Errorf("kokoro: phoneme fp32 weights: %w", err)
	}
	if g.bank, err = g.dev.NewBuffer(halves * 2); err != nil {
		return fmt.Errorf("kokoro: phoneme weight bank: %w", err)
	}
	if err := g.pipelines(); err != nil {
		return err
	}
	for i, l := range []*GPULSTM{g.dur[0], g.dur[1], g.dur[2], g.durLST, g.shared, g.text} {
		var src *LSTM
		switch {
		case i < 3:
			src = p.TextEncoder.LSTMs[i]
		case i == 3:
			src = p.LSTM
		case i == 4:
			src = p.Shared
		default:
			src = m.TextEncoder.LSTM
		}
		if err := l.finish(src); err != nil {
			return fmt.Errorf("kokoro: staging recurrence %d: %w", i, err)
		}
	}
	if err := g.stacks.finish(p); err != nil {
		return fmt.Errorf("kokoro: staging the F0/N stacks: %w", err)
	}
	g.wire()
	return g.stage()
}

// wire points every recurrence at the tensor before it.
//
// This is the whole of what T6d is: six objects that each returned their
// output to the host now read the previous one's in place, and the only
// things that still cross the bus are the two the host genuinely decides
// from.
func (g *GPUPhonemes) wire() {
	g.dur[0].Attach(g.aDEn, g.hidden)
	for i := 1; i < len(g.dur); i++ {
		off, lda := g.dur[i-1].Output()
		g.dur[i].Attach(off, lda)
	}
	off, lda := g.dur[2].Output()
	g.durLST.Attach(off, lda)
	g.shared.Attach(g.aEn, g.hidden)
	g.text.Attach(g.aTE, g.te.Channels)
	g.text.SetNarrowLeaky(true)
	off, lda = g.shared.Output()
	g.stacks.Attach(off, lda)
}

// Encoded is where the duration encoder leaves its output — the tensor the
// length regulator gathers from — as an offset and a row stride.
func (g *GPUPhonemes) Encoded() (uint32, int) { return g.dur[2].Output() }

// Shared is where the recurrence after the length regulator leaves its
// output, which is what the F0/N stacks read.
func (g *GPUPhonemes) Shared() (uint32, int) { return g.shared.Output() }

// Arena is the fp32 buffer every region here lives in, so another object can
// join it.
func (g *GPUPhonemes) Arena() *sharedArena { return g.arena }

func (g *GPUPhonemes) pipelines() error {
	arenas := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"narrow", shaders.KokoroDNarrow},
		{"leaky", shaders.KokoroDLeaky},
		{"layernorm", shaders.ParakeetLayerNorm},
		{"bias", shaders.ParakeetSubBias},
		{"gather", shaders.KokoroGather},
	} {
		mod, err := g.dev.NewShaderModule(s.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", s.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, spec)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", s.name, err)
		}
		g.pipes[s.name] = pipe
	}
	// The text encoder's five-tap convolutions are the A_CONV=2 rung the
	// decoder's blocks run on.
	for _, v := range []struct {
		name  string
		into  **vk.ComputePipeline
		spirv []byte
		wave  uint32
	}{
		{"conv", &g.conv, shaders.KokoroDConvReg32x32W32, 32},
	} {
		s := spec
		s.RequiredSubgroupSize = v.wave
		ok, err := canPinWave(g.dev, v.wave)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("kokoro: this device cannot pin a %d-wide subgroup", v.wave)
		}
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		if *v.into, err = g.dev.NewPipeline(mod, s); err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", v.name, err)
		}
	}
	return nil
}

// stage writes everything that does not depend on the voice.
func (g *GPUPhonemes) stage() error {
	c := g.te.Channels
	bank := make([]uint16, g.bank.Size()/2)
	for i, cv := range g.te.Convs {
		if cv.In != c || cv.Out != c {
			return fmt.Errorf("kokoro: text encoder conv %d is %d->%d, want %d", i, cv.In, cv.Out, c)
		}
		packConvBPad(bank[g.bConv[i]:], cv.Weight, c, c, g.ldTE, cv.Kernel)
	}
	g.bank.WriteUint16At(0, bank)

	w32 := make([]float32, g.wbuf.Size()/4)
	for i, n := range g.te.Norms {
		copy(w32[g.wTEG[i]:], n.Weight)
		copy(w32[g.wTEB[i]:], n.Bias)
		if g.te.Convs[i].Bias != nil {
			copy(w32[g.wTEBias[i]:], g.te.Convs[i].Bias)
		}
	}
	g.wbuf.WriteFloat32At(0, w32)

	// The convolutions' bordered fp16 arena has to be zero before the first
	// one reads it, and the narrow only ever writes rows [0, T).
	g.hbuf.WriteUint16At(0, make([]uint16, g.hElems))
	return nil
}

// SetStyle conditions everything the predictor's 128-wide style vector
// touches: the three AdaLayerNorms between the duration encoder's
// recurrences, and the 128 constant input channels five of the six
// recurrences carry.
//
// It is the whole cost of the conditioning — three [2C, 128] projections of
// one vector plus six operand writes, on the host, once per utterance.
func (g *GPUPhonemes) SetStyle(style []float32) error {
	if len(style) != g.cfg.StyleDim {
		return fmt.Errorf("kokoro: style is %d wide, want %d", len(style), g.cfg.StyleDim)
	}
	w32 := make([]float32, 2*g.hidden)
	for i, n := range g.pred.TextEncoder.Norms {
		if n.FC.Out != 2*g.hidden {
			return fmt.Errorf("kokoro: style projection %d is %d wide, want %d",
				i, n.FC.Out, 2*g.hidden)
		}
		h := n.FC.ApplyRow(make([]float32, n.FC.Out), style)
		// AdaLayerNorm is (1+gamma)*norm(x) + beta, which is a LayerNorm with
		// a weight and a bias — and parakeet's kernel is the one that has
		// both.
		for j := 0; j < g.hidden; j++ {
			w32[j] = 1 + h[j]
			w32[g.hidden+j] = h[g.hidden+j]
		}
		g.wbuf.WriteFloat32At(int(g.wNormG[i]), w32)
	}
	for _, l := range []*GPULSTM{g.dur[0], g.dur[1], g.dur[2], g.durLST, g.shared} {
		if err := l.SetTail(style); err != nil {
			return err
		}
	}
	// The AdaIN blocks downstream take the *predictor's* half of the style
	// vector too, which is the same 128 numbers these five carry.
	return g.stacks.SetStyle(style)
}

// durationGraph is everything up to the logits: three LSTM/AdaLayerNorm pairs,
// one more recurrence, and the head.
func (g *GPUPhonemes) durationGraph() ([]vk.MultiDispatch, []string, error) {
	t := uint32(g.tokens)
	var dis []vk.MultiDispatch
	var kinds []string
	appendLSTM := func(name string, l *GPULSTM) error {
		d, k, err := l.Graph(g.tokens)
		if err != nil {
			return err
		}
		dis = append(dis, d...)
		for _, s := range k {
			kinds = append(kinds, name+" "+s)
		}
		return nil
	}
	for i := range g.dur {
		if err := appendLSTM(fmt.Sprintf("dur%d", i), g.dur[i]); err != nil {
			return nil, nil, err
		}
		// In place over the recurrence's own output: one workgroup a row, and
		// both reduction passes finish before anything is written.
		off, _ := g.dur[i].Output()
		dis = append(dis, vk.MultiDispatch{Pipeline: g.pipes["layernorm"],
			GroupsX: t, GroupsY: 1, PushConstants: pushConstants{
				Tokens: t, Dim: uint32(g.hidden), InOff: off, OutOff: off,
				WOff: g.wNormG[i], Aux0: g.wNormB[i],
				Eps: math.Float32bits(float32(layerNormEps)),
			}.bytes()})
		kinds = append(kinds, fmt.Sprintf("dur%d norm", i))
	}
	if err := appendLSTM("head", g.durLST); err != nil {
		return nil, nil, err
	}
	return dis, kinds, nil
}

// textGraph is the second path from the phonemes: three convolution blocks and
// a recurrence, none of which depends on the durations.
func (g *GPUPhonemes) textGraph() ([]vk.MultiDispatch, []string, error) {
	t := uint32(g.tokens)
	c := g.te.Channels
	m := uint32(roundUp(g.tokens, lstmVariant.bm))
	var dis []vk.MultiDispatch
	var kinds []string

	for i, cv := range g.te.Convs {
		// The first block narrows the embedding as it is; the others rectify
		// on the way in, which is the leaky ReLU that ends the block before.
		narrow := "narrow"
		if i > 0 {
			narrow = "leaky"
		}
		dis = append(dis, vk.MultiDispatch{Pipeline: g.pipes[narrow],
			GroupsX: groups(c, 256), GroupsY: t, PushConstants: pushConstants{
				Dim: uint32(c), InOff: g.aTE, OutOff: g.hTE,
				Aux1: uint32(c), LDA: uint32(g.ldTE), Scale: math.Float32bits(0.2),
			}.bytes()})
		kinds = append(kinds, fmt.Sprintf("te%d %s", i, narrow))

		taps := cv.Kernel
		dis = append(dis, vk.MultiDispatch{Pipeline: g.conv,
			GroupsX: uint32(c / lstmVariant.bn), GroupsY: m / uint32(lstmVariant.bm),
			PushConstants: pushConstants{
				InOff: g.hTE - uint32((taps-1)/2*g.ldTE), OutOff: g.aTE, BOff: g.bConv[i],
				GemmM: m, GemmN: uint32(c), GemmK: uint32(taps * g.ldTE),
				LDA: uint32(g.ldTE), Aux0: uint32(g.ldTE), Aux1: uint32(g.ldTE),
			}.bytes()})
		kinds = append(kinds, fmt.Sprintf("te%d conv", i))

		dis = append(dis, vk.MultiDispatch{Pipeline: g.pipes["bias"],
			GroupsX: t, GroupsY: 1, PushConstants: pushConstants{
				Tokens: t, Dim: uint32(c), InOff: g.aTE, OutOff: g.aTE,
				WOff: g.wTEBias[i], Aux1: t, Scale: math.Float32bits(1),
			}.bytes()})
		kinds = append(kinds, fmt.Sprintf("te%d bias", i))

		dis = append(dis, vk.MultiDispatch{Pipeline: g.pipes["layernorm"],
			GroupsX: t, GroupsY: 1, PushConstants: pushConstants{
				Tokens: t, Dim: uint32(c), InOff: g.aTE, OutOff: g.aTE,
				WOff: g.wTEG[i], Aux0: g.wTEB[i],
				Eps: math.Float32bits(float32(layerNormEps)),
			}.bytes()})
		kinds = append(kinds, fmt.Sprintf("te%d norm", i))
	}
	// The block's closing leaky ReLU is the recurrence's narrow.
	d, k, err := g.text.Graph(g.tokens)
	if err != nil {
		return nil, nil, err
	}
	dis = append(dis, d...)
	for _, s := range k {
		kinds = append(kinds, "text "+s)
	}
	return dis, kinds, nil
}

// sharedGraph is the length regulator and the recurrence after it, which are
// the only things that cannot be recorded until the durations exist.
func (g *GPUPhonemes) sharedGraph(frames int) ([]vk.MultiDispatch, []string, error) {
	if frames <= 0 || frames > g.frames {
		return nil, nil, fmt.Errorf("kokoro: %d frames against an arena for %d", frames, g.frames)
	}
	dis := []vk.MultiDispatch{{Pipeline: g.pipes["gather"],
		GroupsX: groups(g.hidden, 256), GroupsY: uint32(frames),
		PushConstants: pushConstants{
			Dim: uint32(g.hidden), Tokens: uint32(frames),
			InOff: mustOff(g.dur[2].Output()), LDA: uint32(g.hidden),
			OutOff: g.aEn, LDB: uint32(g.hidden), Aux0: g.aIdx,
		}.bytes()}}
	kinds := []string{"gather"}
	d, k, err := g.shared.Graph(frames)
	if err != nil {
		return nil, nil, err
	}
	dis = append(dis, d...)
	for _, s := range k {
		kinds = append(kinds, "shared "+s)
	}
	return dis, kinds, nil
}

func mustOff(off uint32, _ int) uint32 { return off }

// Encode runs everything that does not depend on the durations — the duration
// encoder, the recurrence before the head, and the whole text encoder — and
// returns the head's input.
//
// One submit for both paths: the text encoder depends on nothing the duration
// path produces, so the two are one command buffer and the second costs only
// its own dispatches.
func (g *GPUPhonemes) Encode(dEn *Mat, emb *Mat) (*Mat, error) {
	if dEn.Rows != emb.Rows {
		return nil, fmt.Errorf("kokoro: %d encoder rows against %d embedding rows", dEn.Rows, emb.Rows)
	}
	if dEn.Rows <= 0 || dEn.Rows > g.maxTok {
		return nil, fmt.Errorf("kokoro: %d tokens against an arena for %d", dEn.Rows, g.maxTok)
	}
	if dEn.Cols != g.hidden {
		return nil, fmt.Errorf("kokoro: the encoder gives %d channels, want %d", dEn.Cols, g.hidden)
	}
	if emb.Cols != g.te.Channels {
		return nil, fmt.Errorf("kokoro: the embedding gives %d channels, want %d",
			emb.Cols, g.te.Channels)
	}
	g.setTokens(dEn.Rows)
	g.abuf.WriteFloat32At(int(g.aDEn), dEn.Data)
	g.abuf.WriteFloat32At(int(g.aTE), emb.Data)

	dis, _, err := g.durationGraph()
	if err != nil {
		return nil, err
	}
	td, _, err := g.textGraph()
	if err != nil {
		return nil, err
	}
	if err := submit(append(dis, td...)); err != nil {
		return nil, err
	}

	return g.read(g.durLST), nil
}

// setTokens fixes the token count for an utterance and restores the zero
// padding past it.
//
// The text encoder's arena has always been sized for the *longest* utterance
// the position embedding allows rather than for this one — it is the one
// thing in this package that was bucketed before T9 — and its convolutions
// are five taps wide, so they read two rows past the last token. Those rows
// are zero for the life of a fresh object and hold the previous utterance's
// tokens after a longer one has run: before T9 no attachment survived two
// utterances, so nothing could observe it.
func (g *GPUPhonemes) setTokens(tokens int) {
	if tokens == g.tokens {
		return
	}
	g.tokens = tokens
	g.hbuf.WriteUint16At(int(g.hTE)+tokens*g.ldTE, make([]uint16, convBorder*g.ldTE))
}

// read pulls one recurrence's output back, [T, 2H].
func (g *GPUPhonemes) read(l *GPULSTM) *Mat {
	off, lda := l.Output()
	out := NewMat(g.tokens, g.hidden)
	row := g.abuf.ReadFloat32At(int(off), g.tokens*lda)
	for r := 0; r < g.tokens; r++ {
		copy(out.Row(r), row[r*lda:])
	}
	return out
}

// TextEncoded reads the second path's output, which the host expands by the
// durations on its way to the vocoder.
//
// [T, 512] rather than the [L, 512] the vocoder actually wants, because the
// expansion is a gather either way and 102 KB is cheaper to read than 266.
func (g *GPUPhonemes) TextEncoded() *Mat { return g.read(g.text) }

// Prosody runs everything after the durations: the length regulator, the
// recurrence across it, and both AdaIN stacks.
//
// One submit, and the only thing that comes back is the two curves — 2 KB for
// a stage that in T6b uploaded 266 and in T6c read 266 more.
//
// The durations arrive as frame counts and become one token index per frame,
// which is the only thing the host has to send back after deciding them.
func (g *GPUPhonemes) Prosody(durations []int) (f0, energy []float32, err error) {
	if len(durations) != g.tokens {
		return nil, nil, fmt.Errorf("kokoro: %d durations for %d tokens", len(durations), g.tokens)
	}
	idx := make([]float32, 0, g.frames)
	for t, d := range durations {
		for i := 0; i < d; i++ {
			idx = append(idx, float32(t))
		}
	}
	// The arenas are sized for a ceiling and the utterance is a prefix of
	// them, so the durations have only to fit (SPEECH.md T9).
	if len(idx) > g.frames {
		return nil, nil, &FramesOverflowError{Frames: len(idx), Staged: g.frames}
	}
	if err := g.stacks.SetFrames(len(idx)); err != nil {
		return nil, nil, err
	}
	g.abuf.WriteFloat32At(int(g.aIdx), idx)

	dis, _, err := g.sharedGraph(len(idx))
	if err != nil {
		return nil, nil, err
	}
	sd, _, err := g.stacks.graph()
	if err != nil {
		return nil, nil, err
	}
	if err := submit(append(dis, sd...)); err != nil {
		return nil, nil, err
	}
	n := g.stacks.OutFrames()
	return g.abuf.ReadFloat32At(int(g.stacks.aOut[0]), n),
		g.abuf.ReadFloat32At(int(g.stacks.aOut[1]), n), nil
}

// FramesOverflowError is an utterance longer than the attachment was staged
// for. It carries the frame count the durations came to, which is the only
// thing a caller needs to restage and try again — and the reason this is a
// type rather than a string.
type FramesOverflowError struct {
	Frames int // what the durations summed to
	Staged int // the ceiling the arenas were built for
}

func (e *FramesOverflowError) Error() string {
	return fmt.Sprintf("kokoro: the durations sum to %d frames, staged for %d", e.Frames, e.Staged)
}

// Profile times every dispatch of the phoneme side, by family.
func (g *GPUPhonemes) Profile(tokens, frames int) ([]Stage, error) {
	g.setTokens(tokens)
	var dis []vk.MultiDispatch
	var kinds []string
	for _, f := range []func() ([]vk.MultiDispatch, []string, error){
		g.durationGraph, g.textGraph,
		func() ([]vk.MultiDispatch, []string, error) { return g.sharedGraph(frames) },
		g.stacks.graph,
	} {
		d, k, err := f()
		if err != nil {
			return nil, err
		}
		dis, kinds = append(dis, d...), append(kinds, k...)
	}
	if _, err := vk.DispatchMultiTimed(dis, 1, 2, true); err != nil {
		return nil, err
	}
	// One row per family rather than per dispatch: 700 of them are steps.
	sum := map[string]float64{}
	var order []string
	for i := range dis {
		d, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 4, true)
		if err != nil {
			return nil, err
		}
		k := kinds[i]
		if _, ok := sum[k]; !ok {
			order = append(order, k)
		}
		sum[k] += d.Seconds() / 4
	}
	out := make([]Stage, 0, len(order))
	for _, k := range order {
		out = append(out, Stage{Kind: k, Time: sum[k]})
	}
	return out, nil
}

// Destroy releases every Vulkan object, including the recurrences'.
func (g *GPUPhonemes) Destroy() {
	if g.stacks != nil {
		g.stacks.Destroy()
		g.stacks = nil
	}
	for _, l := range []*GPULSTM{g.dur[0], g.dur[1], g.dur[2], g.durLST, g.shared, g.text} {
		if l != nil {
			l.Destroy()
		}
	}
	g.dur, g.durLST, g.shared, g.text = [3]*GPULSTM{}, nil, nil, nil
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, p := range []*vk.ComputePipeline{g.conv} {
		if p != nil {
			p.Destroy()
		}
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.wbuf, g.hbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	g.pipes, g.mods, g.conv = nil, nil, nil
	g.wbuf, g.hbuf, g.bank, g.abuf = nil, nil, nil, nil
	if g.arena != nil {
		g.arena.destroy()
		g.arena = nil
	}
}
