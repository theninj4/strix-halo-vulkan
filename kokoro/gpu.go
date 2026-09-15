package kokoro

import (
	"fmt"
	"math"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// pushConstants is dit_common.glsl's block, the same one parakeet's graph
// uses. One size across every pipeline is what lets vk.DispatchMultiTimed
// record a mixed sequence into a single command buffer.
type pushConstants struct {
	InOff, OutOff, WOff uint32
	Tokens, Dim, Heads  uint32
	HeadDim, Span       uint32
	KOff, VOff, KStride uint32
	Eps, Scale          uint32
	Aux0, Aux1, Aux2    uint32
	BOff                uint32
	GemmM, GemmN, GemmK uint32
	LDA, LDB            uint32
}

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

const (
	// coopMatTile is the cooperative-matrix extent, the only shape this
	// device reports.
	coopMatTile = 16
	// convBorder is the zero margin, in frames, at each end of the fp16
	// activation arena. A tap of a k-wide filter at dilation d reaches
	// (k-1)*d/2 frames either side; the widest in the generator is k=11 at
	// d=5, which is 25. Rounded to 32 so a bordered row block stays aligned.
	//
	// The border is what makes the convolution branchless: the padding is in
	// the data, so an out-of-range tap reads a zero that was already there.
	// It is stage 8's conv2d finding, one dimension down.
	convBorder = 32
	// framePad is what the frame count is rounded up to, so that any rung of
	// the ladder divides M. It is the widest BM built below.
	framePad = 64
	// noW is dit_common.glsl's NO_W.
	noW = 0xffffffff
	// statChunks is how many frame chunks the AdaIN reduction is split over.
	// It decides the parallelism of a pass that is otherwise one workgroup
	// per channel; 64 chunks of a 15601-frame tensor is 244 frames each,
	// which is enough work per workgroup to cover the launch.
	statChunks = 64
)

// ConvKernel names one build of dit_gemm.comp with A_CONV=1.
type ConvKernel string

const (
	ConvReg32x32W32 ConvKernel = "conv_reg32x32_w32"
	ConvReg32x64W32 ConvKernel = "conv_reg32x64_w32"
	ConvReg64       ConvKernel = "conv_reg64"
	ConvReg32x128   ConvKernel = "conv_reg32x128"
)

type convVariant struct {
	name   ConvKernel
	spirv  []byte
	bm, bn int
	wave   uint32 // 0 takes the driver's default, 64 here
}

// convVariants is the ladder. The vocoder's shape is new to this repository —
// M in the thousands and N of 128 or 256, where the DiT's is M=4096 by
// N=3840 and parakeet's M=138 by N=1024 — so which rung wins is measured
// rather than assumed (TestGPUConvLadder).
var convVariants = []convVariant{
	{name: ConvReg32x32W32, spirv: shaders.KokoroConvReg32x32W32, bm: 32, bn: 32, wave: 32},
	{name: ConvReg32x64W32, spirv: shaders.KokoroConvReg32x64W32, bm: 32, bn: 64, wave: 32},
	{name: ConvReg64, spirv: shaders.KokoroConvReg64, bm: 64, bn: 64},
	{name: ConvReg32x128, spirv: shaders.KokoroConvReg32x128, bm: 32, bn: 128},
}

// ConvKernels lists every rung, in ladder order.
func ConvKernels() []ConvKernel {
	out := make([]ConvKernel, 0, len(convVariants))
	for _, v := range convVariants {
		out = append(out, v.name)
	}
	return out
}

// DefaultConvKernel is the rung TestGPUConvLadder measures as the winner.
const DefaultConvKernel = ConvReg32x64W32

// upWeights is one ConvTranspose1d upsampler, staged as a single GEMM.
//
// At kernel = 2*stride — which both of the generator's upsamplers are — every
// output frame is reached by exactly two taps, so the transposed convolution
// is an ordinary convolution of two taps whose *output* is s times wider:
// column r*C_out + n of a [T+1, s*C_out] result is output frame q*s + r,
// channel n. Read back as [(T+1)*s, C_out] that is already the upsampled
// signal, offset by stride/2 because torch's padding drops those rows — so
// there is no strided store, no residue loop and no second kernel.
type upWeights struct {
	inCh, outCh int
	stride      int
	inFrames    int // T_in; the GEMM runs T_in+1 rows
	bOff        uint32
	bias        uint32
	pad         bool // the last stage's reflection pad
	ldaIn       int
	scale       float32          // folded into the weights, not the bias
	conv        *ConvTranspose1D // dropped after staging
}

// tailWeights is the vocoder's tail — `conv_post`, its two nonlinearities and
// the inverse transform — staged on the last upsampling stage.
//
// It is here rather than in an object of its own because it reads the stage's
// own running value and the pipelines are bound to the stage's four arenas.
// What it buys is the readback: with the tail on the device a stage returns
// 78000 samples, 312 KB, where the [15601, 128] activation it used to return
// is 8 MB — and that buffer reads back at 210 MB/s.
//
// Two scalars are folded into the weights rather than applied as passes. The
// generator averages its three residual blocks, and a leaky rectifier is
// positively homogeneous, so dividing `conv_post`'s *weights* by three (and
// not its bias) is the same function as dividing the activation by three
// first. And conv_post's N is 22, which is not a multiple of the 16-wide
// fragment tile, so it is padded to 32 with zero weight columns.
type tailWeights struct {
	nfft, hop, bins int
	frames, samples int
	n               int // conv_post's output width, padded to a tile
	taps, pad       int
	bank            uint32 // the padded, scaled conv_post weights
	bias            uint32 // [2*bins]
	tw              uint32 // cos, sin, window, window squared
	aPost, aRI      uint32
	aWav            uint32
	conv            *Conv1D // dropped after staging
	window          []float64
}

// blockWeights is one SnakeResBlock's staged offsets, three iterations deep.
type blockWeights struct {
	b1Off, b2Off [3]uint32 // fragment-tiled convolution weights
	k1, k2       [3]int    // taps
	d1           [3]int    // convs1 dilation; convs2 is always 1
	alpha1       [3]uint32 // fp32, per channel
	alpha2       [3]uint32
	bias2        [3]uint32 // convs2 bias; convs1's cancels in the AdaIN
	gb1, gb2     [3]uint32 // gamma then beta, written per utterance
	fc1, fc2     [3]*Linear
}

// GPUBlocks runs the vocoder's AdaIN residual blocks on the device.
//
// These are 97% of the vocoder's 160 GFLOP and 80% of a whole utterance, and
// they are one kernel family: every convolution in them is a k-tap filter
// over a channel-last [T, C] activation, which dit_gemm.comp's A_CONV build
// runs as an ordinary GEMM. What is left on the host — the two transposed
// convolutions, the excitation, the noise projections and the iSTFT — is 3%
// of the arithmetic.
//
// One of these holds every block at one (frames, channels) geometry, which is
// how the generator is shaped: three residual blocks and one noise block per
// upsampling stage, all at that stage's rate.
type GPUBlocks struct {
	dev *vk.Device

	frames, framesPad int
	channels, lda     int
	kernel            ConvKernel

	blocks []blockWeights

	// The upsampler that feeds the blocks, when this set owns one.
	up *upWeights
	// The tail, when this is the last stage.
	tail *tailWeights
	// The arena this stage's fp32 space came from, when it is shared. With
	// one set, aXin is the previous object's output and nothing is uploaded
	// or downloaded between them.
	shared   *sharedArena
	resident bool

	wbuf, abuf, hbuf, bank *vk.Buffer
	mods                   []*vk.ShaderModule
	pipes                  map[string]*vk.ComputePipeline
	convs                  map[ConvKernel]*vk.ComputePipeline

	// Arena offsets.
	aIn, aX, aSum     uint32
	aC, aStat, aAff   uint32
	aNoise, aUp, aXin uint32
	hA, hUp           uint32
	wNext             uint32
	bankNext          uint32
	actElems, hElems  int
}

// sharedArena is one fp32 buffer handed out to the decoder and to both
// generator stages, so that one object's output *is* the next one's input.
//
// The alternative is what T4a and T4b left: each object owns its arena, and a
// stage's result comes back to the host only to be written straight into the
// next one. Measured on this machine that is 2.3 ms for the decoder's
// [260, 512] and 14.4 ms for stage 0's [2600, 256] — 17 ms of a 54 ms
// vocoder, for two tensors that never needed to leave the device. A
// device-local host-visible buffer writes at 29 GB/s and reads at 176-219,
// which is the same 140x asymmetry T4a found; the answer is the same, one
// level up.
//
// Reservation is two-phase because the offsets have to be known before the
// buffer can be sized: every object lays itself out against a running total,
// and Commit allocates once at the end.
type sharedArena struct {
	dev  *vk.Device
	buf  *vk.Buffer
	next uint32
}

// reserve hands out n fp32 elements and returns their offset.
func (a *sharedArena) reserve(n int) uint32 {
	o := a.next
	a.next += uint32(n)
	return o
}

// commit allocates the one buffer every reservation points into.
func (a *sharedArena) commit() error {
	buf, err := a.dev.NewBuffer(int(a.next) * 4)
	if err != nil {
		return fmt.Errorf("kokoro: shared fp32 arena of %d MB: %w", a.next*4>>20, err)
	}
	a.buf = buf
	return nil
}

func (a *sharedArena) destroy() {
	if a.buf != nil {
		a.buf.Destroy()
		a.buf = nil
	}
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

func groups(n, per int) uint32 { return uint32((n + per - 1) / per) }

// NewGPUBlocks stages a set of residual blocks, all of the same shape.
func NewGPUBlocks(dev *vk.Device, blocks []*SnakeResBlock, frames int, kernel ConvKernel) (*GPUBlocks, error) {
	return newGPUBlocks(dev, blocks, frames, kernel, nil, nil, nil)
}

// NewGPUStage stages a whole generator upsampling stage: the transposed
// convolution that feeds it, the three residual blocks it averages and the
// noise block beside them.
//
// With the upsampler here, a stage is one upload of its input and one
// download of its output — the leaky rectifier, the upsampling, the bias, the
// reflection pad and the excitation add all happen between them.
func NewGPUStage(dev *vk.Device, blocks []*SnakeResBlock, up *ConvTranspose1D, inFrames, frames int,
	pad bool, kernel ConvKernel) (*GPUBlocks, error) {
	return newGPUStage(dev, blocks, up, inFrames, frames, pad, kernel, nil, nil)
}

// NewGPUStageWithTail is NewGPUStage for the *last* stage, with `conv_post`
// and the inverse transform attached — so the whole vocoder ends on the
// device and what comes back is the waveform.
func NewGPUStageWithTail(dev *vk.Device, blocks []*SnakeResBlock, up *ConvTranspose1D, inFrames, frames int,
	pad bool, kernel ConvKernel, g *Generator, sa *sharedArena) (*GPUBlocks, error) {
	if g.ISTFT == nil || g.ConvPost == nil {
		return nil, fmt.Errorf("kokoro: the generator has no tail to stage")
	}
	if !g.ISTFT.Center {
		return nil, fmt.Errorf("kokoro: this path assumes a centred inverse transform")
	}
	bins := g.ISTFT.Bins()
	if g.ConvPost.Out != 2*bins {
		return nil, fmt.Errorf("kokoro: conv_post is %d wide against %d bins", g.ConvPost.Out, bins)
	}
	return newGPUStage(dev, blocks, up, inFrames, frames, pad, kernel, &tailWeights{
		nfft: g.ISTFT.NFFT, hop: g.ISTFT.Hop, bins: bins,
		frames: frames, samples: g.ISTFT.Samples(frames),
		n:    roundUp(g.ConvPost.Out, coopMatTile*2),
		taps: g.ConvPost.Kernel, pad: g.ConvPost.Pad,
		conv: g.ConvPost, window: g.ISTFT.Window(),
	}, sa)
}

func newGPUStage(dev *vk.Device, blocks []*SnakeResBlock, up *ConvTranspose1D, inFrames, frames int,
	pad bool, kernel ConvKernel, tail *tailWeights, sa *sharedArena) (*GPUBlocks, error) {
	if up.Kernel != 2*up.Stride || up.Pad != up.Stride/2 {
		return nil, fmt.Errorf("kokoro: upsampler k=%d s=%d p=%d, this path needs k=2s and p=s/2",
			up.Kernel, up.Stride, up.Pad)
	}
	return newGPUBlocks(dev, blocks, frames, kernel, &upWeights{
		inCh: up.In, outCh: up.Out, stride: up.Stride, inFrames: inFrames, pad: pad, ldaIn: up.In,
		conv: up,
	}, tail, sa)
}

func newGPUBlocks(dev *vk.Device, blocks []*SnakeResBlock, frames int, kernel ConvKernel,
	up *upWeights, tail *tailWeights, sa *sharedArena) (*GPUBlocks, error) {
	if len(blocks) == 0 {
		return nil, fmt.Errorf("kokoro: no blocks")
	}
	c := blocks[0].Channels
	if c&(c-1) != 0 {
		return nil, fmt.Errorf("kokoro: %d channels is not a power of two", c)
	}
	g := &GPUBlocks{
		dev: dev, frames: frames, framesPad: roundUp(frames, framePad),
		channels: c, lda: c, kernel: kernel, up: up, tail: tail, shared: sa,
		pipes: map[string]*vk.ComputePipeline{},
		convs: map[ConvKernel]*vk.ComputePipeline{},
	}
	for _, b := range blocks {
		if b.Channels != c {
			return nil, fmt.Errorf("kokoro: blocks of %d and %d channels in one set", c, b.Channels)
		}
	}
	if err := g.alloc(blocks); err != nil {
		g.Destroy()
		return nil, err
	}
	if g.shared != nil {
		// The buffer does not exist yet: the caller reserves every object's
		// space, commits once, and then calls finish.
		return g, nil
	}
	if err := g.finish(blocks); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// finish builds the pipelines and writes the weights, which cannot happen
// until every arena exists.
func (g *GPUBlocks) finish(blocks []*SnakeResBlock) error {
	if g.shared != nil {
		g.abuf = g.shared.buf
	}
	if err := g.build(); err != nil {
		return err
	}
	return g.stage(blocks)
}

// Frames is the geometry this set was built for.
func (g *GPUBlocks) Frames() int { return g.frames }

// Channels is the width this set was built for.
func (g *GPUBlocks) Channels() int { return g.channels }

// Kernel is the rung the convolutions run on.
func (g *GPUBlocks) Kernel() ConvKernel { return g.kernel }

// SetKernel switches rungs. The weights are staged as fragment tiles, which
// every rung reads, so this needs no restaging — the same property that makes
// parakeet's GEMM ladder one load instead of eight.
func (g *GPUBlocks) SetKernel(k ConvKernel) error {
	if _, ok := g.convs[k]; !ok {
		return fmt.Errorf("kokoro: no pipeline for %q", k)
	}
	g.kernel = k
	return nil
}

func (g *GPUBlocks) alloc(blocks []*SnakeResBlock) error {
	c := g.channels
	var base, off32 uint32
	if g.shared != nil {
		base = g.shared.next
	}
	take32 := func(n int) uint32 { o := base + off32; off32 += uint32(n); return o }
	// The stage's input is held once and copied into the running value before
	// each block, because the generator runs three blocks over the *same*
	// input and averages them. Uploading it three times would cost more than
	// the blocks do.
	g.aIn = take32(g.framesPad * c)
	g.aX = take32(g.framesPad * c)
	g.aSum = take32(g.framesPad * c)
	g.aC = take32(g.framesPad * c)
	g.aStat = take32(statChunks * 2 * c)
	g.aAff = take32(2 * c)
	if g.up != nil {
		// The excitation's own residual block runs in place in this arena,
		// so the noise path never touches the running value.
		g.aNoise = take32(g.framesPad * c)
		g.aXin = take32(roundUp(g.up.inFrames, framePad) * g.up.inCh)
		// The upsampler writes (T_in+1) rows of stride*C, which read as
		// [(T_in+1)*stride, C] is the signal offset by stride/2.
		g.aUp = take32(roundUp(g.up.inFrames+1, framePad) * g.up.stride * c)
	}
	if t := g.tail; t != nil {
		// conv_post writes roundUp(frames, BM) rows of the padded width; the
		// rectangular spectrum and the waveform are exact.
		t.aPost = take32(g.framesPad * t.n)
		t.aRI = take32(t.frames * 2 * t.bins)
		t.aWav = take32(t.samples)
	}
	g.actElems = int(off32)

	// One fp16 A operand, bordered at both ends. Only one is live at a time:
	// the fused activation pass writes it and the convolution reads it.
	g.hElems = (2*convBorder + g.framesPad) * g.lda
	g.hA = uint32(convBorder * g.lda)
	if g.up != nil {
		// A second bordered arena, at the *input* width: the upsampler's two
		// taps read frames q-1 and q, so both ends need a zero row.
		g.hUp = uint32(g.hElems + convBorder*g.up.ldaIn)
		g.hElems += (2*convBorder + roundUp(g.up.inFrames+1, framePad)) * g.up.ldaIn
	}

	// fp32 weights: per block per iteration, two alphas, one bias and two
	// pairs of style coefficients.
	var offW uint32
	for range blocks {
		offW += 3 * uint32(5*c+2*c) // alpha1, alpha2, bias2, gb1, gb2
	}
	if g.up != nil {
		offW += uint32(g.up.outCh)
	}
	if t := g.tail; t != nil {
		offW += uint32(2*t.bins + 2*t.nfft*t.bins + 2*t.nfft)
	}
	g.wNext = 0

	var err error
	if g.shared != nil {
		g.shared.reserve(g.actElems)
	} else if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("kokoro: fp32 arena: %w", err)
	}
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("kokoro: fp16 arena: %w", err)
	}
	if g.wbuf, err = g.dev.NewBuffer(int(offW) * 4); err != nil {
		return fmt.Errorf("kokoro: fp32 weights: %w", err)
	}
	// Fragment tiles for every convolution: N = C rows of K = taps*C.
	var halves int
	for _, b := range blocks {
		for i := range b.Convs1 {
			halves += c * b.Convs1[i].Kernel * c
			halves += c * b.Convs2[i].Kernel * c
		}
	}
	if g.up != nil {
		halves += g.up.stride * g.up.outCh * 2 * g.up.inCh
	}
	if t := g.tail; t != nil {
		halves += t.n * t.taps * c
	}
	g.bank, err = g.dev.NewBuffer(halves * 2)
	if err != nil {
		return fmt.Errorf("kokoro: weight bank: %w", err)
	}
	return nil
}

func (g *GPUBlocks) build() error {
	arenas := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	spec := vk.PipelineSpec{Buffers: arenas, PushConstantSize: uint32(unsafe.Sizeof(pushConstants{}))}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"stats", shaders.KokoroStats},
		{"affine", shaders.KokoroAffine},
		{"act", shaders.KokoroAct},
		{"snake", shaders.KokoroActSnake},
		{"residual", shaders.KokoroResidual},
		{"copy", shaders.KokoroCopy},
		{"leaky", shaders.KokoroActLeaky},
		{"upadd", shaders.KokoroUpAdd},
		{"upaddpad", shaders.KokoroUpAddPad},
		{"post", shaders.KokoroPost},
		{"istft", shaders.KokoroISTFT},
	} {
		if err := g.pipeline(s.name, s.spirv, spec); err != nil {
			return err
		}
	}
	for _, v := range convVariants {
		s := spec
		s.RequiredSubgroupSize = v.wave
		if v.wave != 0 {
			ok, err := canPinWave(g.dev, v.wave)
			if err != nil {
				return err
			}
			if !ok {
				continue
			}
		}
		mod, err := g.dev.NewShaderModule(v.spirv)
		if err != nil {
			return fmt.Errorf("kokoro: shader %s: %w", v.name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := g.dev.NewPipeline(mod, s)
		if err != nil {
			return fmt.Errorf("kokoro: pipeline %s: %w", v.name, err)
		}
		g.convs[v.name] = pipe
	}
	if _, ok := g.convs[g.kernel]; !ok {
		return fmt.Errorf("kokoro: convolution kernel %q is not available on this device", g.kernel)
	}
	return nil
}

// canPinWave reports whether the device lets a compute pipeline demand the
// given subgroup size. The wave32 rungs are useless without it, and §6.2
// measured wave32 beating wave64 on every shape either vertical has run.
func canPinWave(dev *vk.Device, wave uint32) (bool, error) {
	if wave == 0 {
		return true, nil
	}
	if !dev.Features().SubgroupSizeControl {
		return false, nil
	}
	sgs, err := dev.Physical().SubgroupSizeControl()
	if err != nil {
		return false, fmt.Errorf("kokoro: subgroup size control: %w", err)
	}
	return sgs.Supported && wave >= sgs.MinSubgroupSize && wave <= sgs.MaxSubgroupSize, nil
}

func (g *GPUBlocks) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("kokoro: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("kokoro: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage writes the weights: fp16 fragment tiles for the convolutions, fp32
// for everything read elementwise.
//
// convs1's bias is not staged at all. Its output goes straight into an AdaIN,
// which subtracts the per-channel mean over time — a per-channel constant
// moves that mean by exactly itself and leaves the variance alone, so the
// bias cancels identically. TestGPUBiasCancels holds that to the CPU.
func (g *GPUBlocks) stage(blocks []*SnakeResBlock) error {
	c := g.channels
	bank := make([]uint16, g.bank.Size()/2)
	w32 := make([]float32, g.wbuf.Size()/4)
	var bankOff, wOff uint32
	takeW := func(v []float32) uint32 {
		o := wOff
		copy(w32[o:], v)
		wOff += uint32(len(v))
		return o
	}
	for _, b := range blocks {
		bw := blockWeights{}
		for i := range b.Convs1 {
			c1, c2 := b.Convs1[i], b.Convs2[i]
			if c1.In != c || c1.Out != c || c2.In != c || c2.Out != c {
				return fmt.Errorf("kokoro: resblock convolution %d->%d, want %d->%d", c1.In, c1.Out, c, c)
			}
			bw.k1[i], bw.k2[i], bw.d1[i] = c1.Kernel, c2.Kernel, c1.Dilation
			bw.b1Off[i] = bankOff
			packConvB(bank[bankOff:], c1.Weight, c, c, c1.Kernel)
			bankOff += uint32(c * c * c1.Kernel)
			bw.b2Off[i] = bankOff
			packConvB(bank[bankOff:], c2.Weight, c, c, c2.Kernel)
			bankOff += uint32(c * c * c2.Kernel)

			bw.alpha1[i] = takeW(b.Alpha1[i])
			bw.alpha2[i] = takeW(b.Alpha2[i])
			bw.bias2[i] = takeW(c2.Bias)
			// The style coefficients are written by SetStyle; reserve them.
			bw.gb1[i] = takeW(make([]float32, 2*c))
			bw.gb2[i] = takeW(make([]float32, 2*c))
			bw.fc1[i], bw.fc2[i] = b.Norm1[i].FC, b.Norm2[i].FC
		}
		g.blocks = append(g.blocks, bw)
	}
	if g.up != nil {
		u := g.up
		u.bias = takeW(u.conv.Bias)
		u.bOff = bankOff
		w := u.conv.Weight
		if u.scale != 0 {
			// The previous stage left its blocks summed rather than averaged,
			// and the rectifier in front of this upsampler is positively
			// homogeneous, so the division lives here. The bias is not
			// scaled: it is added after the projection, not before it.
			w = make([]float32, len(u.conv.Weight))
			for i, v := range u.conv.Weight {
				w[i] = v * u.scale
			}
		}
		packUpB(bank[bankOff:], w, u.inCh, u.outCh, u.stride)
		bankOff += uint32(u.stride * u.outCh * 2 * u.inCh)
		u.conv = nil
	}
	if t := g.tail; t != nil {
		// The generator averages three residual blocks and the rectifier in
		// front of conv_post is positively homogeneous, so the division by
		// three lives in these weights. The bias is not divided: it is added
		// after the projection, not before it.
		inv := float32(1) / float32(len(blocks)-1)
		w := make([]float32, len(t.conv.Weight))
		for i, v := range t.conv.Weight {
			w[i] = v * inv
		}
		bias := make([]float32, t.conv.Out)
		if t.conv.Bias != nil {
			copy(bias, t.conv.Bias)
		}
		t.bias = takeW(bias)

		// The twiddles and the window: [nfft, bins] of each, then the window
		// and its square. They depend on nothing but the geometry.
		tw := make([]float32, 2*t.nfft*t.bins+2*t.nfft)
		for n := 0; n < t.nfft; n++ {
			for b := 0; b < t.bins; b++ {
				th := 2 * math.Pi * float64(b*n) / float64(t.nfft)
				tw[n*t.bins+b] = float32(math.Cos(th))
				tw[t.nfft*t.bins+n*t.bins+b] = float32(math.Sin(th))
			}
			wn := float32(t.window[n])
			tw[2*t.nfft*t.bins+n] = wn
			tw[2*t.nfft*t.bins+t.nfft+n] = wn * wn
		}
		t.tw = takeW(tw)

		t.bank = bankOff
		packConvBPad(bank[bankOff:], w, t.conv.Out, g.channels, g.channels, t.taps)
		bankOff += uint32(t.n * t.taps * g.channels)
		t.conv, t.window = nil, nil
	}
	g.bank.WriteUint16At(0, bank)
	g.wbuf.WriteFloat32At(0, w32)
	// The fp16 arena's border has to be zero before the first convolution
	// reads it, and nothing writes it afterwards.
	g.hbuf.WriteUint16At(0, make([]uint16, g.hElems))
	return nil
}

// packConvB lays a [Out, In, taps] convolution weight out as the 16x16
// fragment tiles B_LAYOUT=2 reads, with the K index being tap*In + in — which
// is the order dit_gemm.comp's A_CONV walks the activation in.
func packConvB(dst []uint16, w []float32, out, in, taps int) {
	packConvBPad(dst, w, out, in, in, taps)
}

// SetStyle computes every AdaIN's gamma and beta for one utterance.
//
// This is the whole cost of the style conditioning on the device: nothing. The
// `fc` layers are [2C, 128] projections of a single vector, so they run once
// on the host and what reaches the graph is two numbers per channel — which is
// also why kokoro_affine.comp never sees the style at all.
func (g *GPUBlocks) SetStyle(style []float32) error {
	c := g.channels
	buf := make([]float32, 2*c)
	for _, bw := range g.blocks {
		for i := 0; i < 3; i++ {
			for _, s := range []struct {
				fc  *Linear
				off uint32
			}{{bw.fc1[i], bw.gb1[i]}, {bw.fc2[i], bw.gb2[i]}} {
				if s.fc.Out != 2*c {
					return fmt.Errorf("kokoro: style projection is %d wide, want %d", s.fc.Out, 2*c)
				}
				s.fc.ApplyRow(buf, style)
				g.wbuf.WriteFloat32At(int(s.off), buf)
			}
		}
	}
	return nil
}

// graph builds one block's dispatch sequence, with a label per dispatch.
func (g *GPUBlocks) graph(block int) ([]vk.MultiDispatch, []string, error) {
	return g.graphAt(block, g.aX)
}

// graphAt builds a block's sequence over a caller-chosen running value, so
// the excitation's block can run in its own arena rather than through the one
// the generator's three share.
func (g *GPUBlocks) graphAt(block int, aX uint32) ([]vk.MultiDispatch, []string, error) {
	bw := &g.blocks[block]
	c := uint32(g.channels)
	logC := uint32(math.Log2(float64(g.channels)))
	rows := uint32(g.frames)
	perChunk := uint32(roundUp(g.frames, statChunks) / statChunks)

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe *vk.ComputePipeline, kind string, gx uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: pipe, GroupsX: gx, GroupsY: 1, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	base := pushConstants{Tokens: rows, Dim: c, Aux2: logC, Eps: math.Float32bits(float32(instanceNormEps))}

	// AdaIN over time, then Snake, then the narrow into the bordered arena.
	adain := func(kind string, src, gb, alpha uint32) {
		pc := base
		pc.InOff, pc.OutOff, pc.Aux0 = src, g.aStat, perChunk
		add(g.pipes["stats"], kind+" stats", statChunks, pc)

		pc = base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = g.aStat, g.aAff, gb, statChunks
		add(g.pipes["affine"], kind+" affine", 1, pc)

		pc = base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = src, g.hA, alpha, g.aAff
		pc.LDA = uint32(g.lda)
		add(g.pipes["snake"], kind+" snake", groups(g.frames*g.channels, 256), pc)
	}
	// One convolution: C[t, o] = sum over taps and input channels, with the
	// im2col implicit in the A operand's addressing.
	conv := func(kind string, bOff uint32, taps, dilation int, out uint32) error {
		v, ok := convVariantFor(g.kernel)
		if !ok {
			return fmt.Errorf("kokoro: no variant %q", g.kernel)
		}
		if g.channels%v.bn != 0 {
			return fmt.Errorf("kokoro: tile width %d does not divide %d channels", v.bn, g.channels)
		}
		m := roundUp(g.frames, v.bm)
		pad := (taps - 1) * dilation / 2
		pc := base
		// inOff points at the row the first tap reads, which is `pad` frames
		// before the output frame — the border makes that address legal.
		pc.InOff = g.hA - uint32(pad*g.lda)
		pc.OutOff, pc.BOff = out, bOff
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(m), c, uint32(taps*g.channels)
		pc.LDA = uint32(g.lda)
		pc.Aux0, pc.Aux1, pc.Aux2 = c, uint32(dilation*g.lda), logC
		d = append(d, vk.MultiDispatch{
			Pipeline: g.convs[g.kernel], GroupsX: uint32(g.channels / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, kind)
		return nil
	}

	for i := 0; i < 3; i++ {
		adain(fmt.Sprintf("norm1.%d", i), aX, bw.gb1[i], bw.alpha1[i])
		if err := conv(fmt.Sprintf("conv1.%d k=%d d=%d", i, bw.k1[i], bw.d1[i]),
			bw.b1Off[i], bw.k1[i], bw.d1[i], g.aC); err != nil {
			return nil, nil, err
		}
		adain(fmt.Sprintf("norm2.%d", i), g.aC, bw.gb2[i], bw.alpha2[i])
		if err := conv(fmt.Sprintf("conv2.%d k=%d", i, bw.k2[i]),
			bw.b2Off[i], bw.k2[i], 1, g.aC); err != nil {
			return nil, nil, err
		}
		pc := base
		pc.InOff, pc.OutOff, pc.WOff = g.aC, aX, bw.bias2[i]
		add(g.pipes["residual"], fmt.Sprintf("residual.%d", i), groups(g.frames*g.channels, 256), pc)
	}
	return d, kinds, nil
}

func convVariantFor(k ConvKernel) (convVariant, bool) {
	for _, v := range convVariants {
		if v.name == k {
			return v, true
		}
	}
	return convVariant{}, false
}

// Upload writes the stage's input into the fp32 arena.
func (g *GPUBlocks) Upload(x *Mat) error {
	if x.Rows != g.frames || x.Cols != g.channels {
		return fmt.Errorf("kokoro: input %v, want [%d %d]", x, g.frames, g.channels)
	}
	g.abuf.WriteFloat32At(int(g.aIn), x.Data)
	return nil
}

// Download reads the running value back.
func (g *GPUBlocks) Download() *Mat {
	return &Mat{Rows: g.frames, Cols: g.channels,
		Data: g.abuf.ReadFloat32At(int(g.aX), g.frames*g.channels)}
}

// elementwise is one dispatch of a pass that walks [T, C] linearly.
func (g *GPUBlocks) elementwise(pipe string, in, out, w uint32) vk.MultiDispatch {
	pc := pushConstants{Tokens: uint32(g.frames), Dim: uint32(g.channels), InOff: in, OutOff: out, WOff: w}
	return vk.MultiDispatch{
		Pipeline: g.pipes[pipe], GroupsX: groups(g.frames*g.channels, 256), GroupsY: 1,
		PushConstants: pc.bytes(),
	}
}

// Run executes one block over the uploaded input, leaving the result in the
// running value.
func (g *GPUBlocks) Run(block int) error {
	d, _, err := g.graph(block)
	if err != nil {
		return err
	}
	d = append([]vk.MultiDispatch{g.elementwise("copy", g.aIn, g.aX, noW)}, d...)
	return submit(d)
}

// Average runs several blocks over the same input and averages them, which is
// the generator's multi-receptive-field fusion: three filters of 3, 7 and 11
// taps over one activation, summed and divided by three.
//
// Nothing crosses the bus between them. The input is copied into the running
// value on the device before each block and the sum accumulates in a third
// arena, so a stage is one upload and one download however many blocks it has.
func (g *GPUBlocks) Average(blocks []int) (*Mat, error) {
	if len(blocks) == 0 {
		return nil, fmt.Errorf("kokoro: no blocks to average")
	}
	var d []vk.MultiDispatch
	for j, b := range blocks {
		g0, _, err := g.graph(b)
		if err != nil {
			return nil, err
		}
		d = append(d, g.elementwise("copy", g.aIn, g.aX, noW))
		d = append(d, g0...)
		if j == 0 {
			d = append(d, g.elementwise("copy", g.aX, g.aSum, noW))
		} else {
			d = append(d, g.elementwise("residual", g.aX, g.aSum, noW))
		}
	}
	if err := submit(d); err != nil {
		return nil, err
	}
	out := &Mat{Rows: g.frames, Cols: g.channels,
		Data: g.abuf.ReadFloat32At(int(g.aSum), g.frames*g.channels)}
	inv := 1 / float32(len(blocks))
	for i := range out.Data {
		out.Data[i] *= inv
	}
	return out, nil
}

// ApplyOne runs a single block, which is what a correctness check wants and
// not what the pipeline does.
func (g *GPUBlocks) ApplyOne(block int, x *Mat) (*Mat, error) {
	if err := g.Upload(x); err != nil {
		return nil, err
	}
	if err := g.Run(block); err != nil {
		return nil, err
	}
	return g.Download(), nil
}

// perSubmit caps how many dispatches go into one command buffer, the same
// bound parakeet's graph uses.
const perSubmit = 240

func submit(d []vk.MultiDispatch) error {
	for len(d) > 0 {
		n := min(len(d), perSubmit)
		if _, err := vk.DispatchMultiTimed(d[:n], 1, 1, true); err != nil {
			return err
		}
		d = d[n:]
	}
	return nil
}

// Stage is one dispatch's label and measured time.
type Stage struct {
	Kind string
	Time float64 // seconds
}

// Profile times every dispatch of one block on the device.
func (g *GPUBlocks) Profile(block int) ([]Stage, error) {
	d, kinds, err := g.graph(block)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		t, err := vk.DispatchMultiTimed(d[i:i+1], 1, 4, true)
		if err != nil {
			return nil, err
		}
		out = append(out, Stage{Kind: kinds[i], Time: t.Seconds() / 4})
	}
	return out, nil
}

// Destroy releases every Vulkan object.
func (g *GPUBlocks) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, p := range g.convs {
		p.Destroy()
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
	g.pipes, g.convs, g.mods = nil, nil, nil
	g.wbuf, g.abuf, g.hbuf, g.bank = nil, nil, nil, nil
}

// AttachGPU stages the generator's eight residual blocks onto the device, one
// set per upsampling stage, and conditions them on a voice.
//
// The frame counts are fixed at construction because the arenas are, so this
// takes the alignment frame count the prosody predicted and derives the rest —
// the same chain TestSTFTGeometry pins: 2L frames into the generator, then the
// upsampling rates, then a reflection pad on the last stage.
func (m *Model) AttachGPU(dev *vk.Device, frames int, decStyle, predStyle []float32, kernel ConvKernel) error {
	g := m.Vocoder.Generator
	m.DetachGPU()

	// One fp32 arena for all three objects, laid out before any of them is
	// built: the decoder's output *is* stage 0's input and stage 0's is
	// stage 1's, so the two tensors between them — 0.5 MB and 2.5 MB — never
	// cross the bus. See sharedArena.
	sa := &sharedArena{dev: dev}

	// PL-BERT, which is 47% of the phoneme side and shares one weight group
	// across its twelve layers, so it is staged once for any utterance rather
	// than sized per clip like everything below.
	bert, err := NewGPUAlbert(dev, m.BERT, m.BERT.Config.MaxPositionEmbed)
	if err != nil {
		return fmt.Errorf("kokoro: staging PL-BERT: %w", err)
	}
	m.BERTGPU = bert

	// The rest of the phoneme side, on an arena of its own: the duration
	// encoder, the duration head, the length regulator, the text encoder and
	// the F0/N stacks, with every recurrence chained through the tensor
	// before it (SPEECH.md T6c, T6d). Two things come back from it — the
	// duration logits and the text encoder's output — and both are host
	// decisions rather than readbacks that could be removed.
	ph, err := NewGPUPhonemes(dev, m, frames)
	if err != nil {
		m.DetachGPU()
		return fmt.Errorf("kokoro: staging the phoneme side: %w", err)
	}
	m.PhonemesGPU = ph
	if err := ph.SetStyle(predStyle); err != nil {
		m.DetachGPU()
		return err
	}

	// The decoder first, because it is what decides the generator's input:
	// its last block doubles the frame count, which is where the 2*frames
	// below comes from (SPEECH.md T4c).
	dec, err2 := newGPUDecoder(dev, m.Vocoder, frames, DefaultDecoderKernel, sa)
	if err = err2; err != nil {
		return fmt.Errorf("kokoro: staging the decoder: %w", err)
	}
	m.Vocoder.GPU = dec

	n := 2 * frames
	for i := range g.Ups {
		in := n
		n = g.Ups[i].OutFrames(n)
		pad := i == len(g.Ups)-1
		if pad {
			n++ // the reflection pad
		}
		blocks := []*SnakeResBlock{
			g.ResBlocks[i*g.NumKernels], g.ResBlocks[i*g.NumKernels+1], g.ResBlocks[i*g.NumKernels+2],
			g.NoiseRes[i],
		}
		var gb *GPUBlocks
		var err error
		if i == len(g.Ups)-1 {
			gb, err = NewGPUStageWithTail(dev, blocks, g.Ups[i], in, n, pad, kernel, g, sa)
		} else {
			gb, err = newGPUStage(dev, blocks, g.Ups[i], in, n, pad, kernel, nil, sa)
		}
		if err != nil {
			m.DetachGPU()
			return fmt.Errorf("kokoro: staging generator stage %d: %w", i, err)
		}
		g.GPU = append(g.GPU, gb)
	}

	// The aliasing, which is the whole point of the shared arena. Each of
	// these pairs is the same [T, C] region under two names, so an object's
	// UploadInput has nothing to do.
	//
	// A stage leaves its three residual blocks *summed*, not averaged, so the
	// consumer carries the 1/NumKernels. It can, because every consumer is
	// linear or positively homogeneous: the next stage's upsampler takes it
	// into its weights (below, not its bias, which is added afterwards) and
	// the tail takes it into conv_post's.
	g.GPU[0].aXin, g.GPU[0].resident = dec.aOut, true
	for i := 1; i < len(g.GPU); i++ {
		g.GPU[i].aXin, g.GPU[i].resident = g.GPU[i-1].aSum, true
		g.GPU[i].up.scale = 1 / float32(g.NumKernels)
	}

	if err := sa.commit(); err != nil {
		m.DetachGPU()
		return err
	}
	m.Vocoder.arena = sa
	if err := dec.finish(); err != nil {
		m.DetachGPU()
		return fmt.Errorf("kokoro: staging the decoder: %w", err)
	}
	if err := dec.SetStyle(decStyle); err != nil {
		m.DetachGPU()
		return err
	}
	for i, gb := range g.GPU {
		blocks := []*SnakeResBlock{
			g.ResBlocks[i*g.NumKernels], g.ResBlocks[i*g.NumKernels+1], g.ResBlocks[i*g.NumKernels+2],
			g.NoiseRes[i],
		}
		if err := gb.finish(blocks); err != nil {
			m.DetachGPU()
			return fmt.Errorf("kokoro: staging generator stage %d: %w", i, err)
		}
		if err := gb.SetStyle(decStyle); err != nil {
			m.DetachGPU()
			return err
		}
	}
	return nil
}

// DetachGPU releases the staged blocks and puts the vocoder back on the CPU.
func (m *Model) DetachGPU() {
	g := m.Vocoder.Generator
	for _, gb := range g.GPU {
		gb.Destroy()
	}
	g.GPU = nil
	if m.Vocoder.GPU != nil {
		m.Vocoder.GPU.Destroy()
		m.Vocoder.GPU = nil
	}
	if m.Vocoder.arena != nil {
		m.Vocoder.arena.destroy()
		m.Vocoder.arena = nil
	}
	if m.BERTGPU != nil {
		m.BERTGPU.Destroy()
		m.BERTGPU = nil
	}
	if m.PhonemesGPU != nil {
		m.PhonemesGPU.Destroy()
		m.PhonemesGPU = nil
	}
}

// recurrences is every bidirectional LSTM in the model, in the order an
// utterance runs them.
func (m *Model) recurrences() []*LSTM {
	p := m.Predictor
	out := append([]*LSTM{}, p.TextEncoder.LSTMs...)
	return append(out, p.LSTM, p.Shared, m.TextEncoder.LSTM)
}

// packUpB lays a ConvTranspose1d weight out as the one GEMM described on
// upWeights: N = stride*outCh, K = 2*inCh.
//
// Output frame q*s + r reads input frames q-1 and q, with taps r+s and r. The
// A operand walks the two input rows in that order (the earlier one first,
// which is what makes the tap stride a positive lda), so tap 0 of the B
// operand is the *later* weight index.
func packUpB(dst []uint16, w []float32, in, out, stride int) {
	n, k := stride*out, 2*in
	kt := k / coopMatTile
	taps := 2 * stride
	parallelFor(n, func(row int) {
		r, o := row/out, row%out
		base := (row / coopMatTile) * kt * coopMatTile * coopMatTile
		lane := (row % coopMatTile) * coopMatTile
		for i := 0; i < in; i++ {
			for jj := 0; jj < 2; jj++ {
				m := r + stride
				if jj == 1 {
					m = r
				}
				kk := jj*in + i
				dst[base+(kk/coopMatTile)*coopMatTile*coopMatTile+lane+kk%coopMatTile] =
					safetensors.F32ToF16(w[(i*out+o)*taps+m])
			}
		}
	})
}

// UploadInput writes the stage's input — the previous stage's output, or the
// decoder's — into the upsampler's arena.
func (g *GPUBlocks) UploadInput(x *Mat) error {
	if g.up == nil {
		return fmt.Errorf("kokoro: this block set has no upsampler")
	}
	if x.Rows != g.up.inFrames || x.Cols != g.up.inCh {
		return fmt.Errorf("kokoro: stage input %v, want [%d %d]", x, g.up.inFrames, g.up.inCh)
	}
	g.abuf.WriteFloat32At(int(g.aXin), x.Data)
	return nil
}

// UploadNoise writes the excitation projection the host computed.
func (g *GPUBlocks) UploadNoise(x *Mat) error {
	if x.Rows != g.frames || x.Cols != g.channels {
		return fmt.Errorf("kokoro: excitation %v, want [%d %d]", x, g.frames, g.channels)
	}
	g.abuf.WriteFloat32At(int(g.aNoise), x.Data)
	return nil
}

// upGraph is the stage's front: rectify and narrow, upsample, then the fused
// bias, reflection pad and excitation add.
func (g *GPUBlocks) upGraph() ([]vk.MultiDispatch, []string, error) {
	u := g.up
	v, ok := convVariantFor(g.kernel)
	if !ok {
		return nil, nil, fmt.Errorf("kokoro: no variant %q", g.kernel)
	}
	n := u.stride * u.outCh
	if n%v.bn != 0 {
		return nil, nil, fmt.Errorf("kokoro: tile width %d does not divide %d output columns", v.bn, n)
	}
	m := roundUp(u.inFrames+1, v.bm)
	logIn := uint32(math.Log2(float64(u.inCh)))
	logOut := uint32(math.Log2(float64(g.channels)))

	pcLeaky := pushConstants{
		Tokens: uint32(u.inFrames), Dim: uint32(u.inCh), Aux2: logIn,
		InOff: g.aXin, OutOff: g.hUp, LDA: uint32(u.ldaIn),
		Scale: math.Float32bits(0.1),
	}
	pcUp := pushConstants{
		InOff: g.hUp - uint32(u.ldaIn), OutOff: g.aUp, BOff: u.bOff,
		GemmM: uint32(m), GemmN: uint32(n), GemmK: uint32(2 * u.inCh),
		LDA:  uint32(u.ldaIn),
		Aux0: uint32(u.inCh), Aux1: uint32(u.ldaIn), Aux2: logIn,
	}
	pcAdd := pushConstants{
		Tokens: uint32(g.frames), Dim: uint32(g.channels), Aux2: logOut,
		InOff:  g.aUp + uint32(u.stride/2*g.channels),
		OutOff: g.aIn, WOff: u.bias, Aux0: g.aNoise,
	}
	add := "upadd"
	if u.pad {
		add = "upaddpad"
	}
	d := []vk.MultiDispatch{
		{Pipeline: g.pipes["leaky"], GroupsX: groups(u.inFrames*u.inCh, 256), GroupsY: 1,
			PushConstants: pcLeaky.bytes()},
		{Pipeline: g.convs[g.kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pcUp.bytes()},
		{Pipeline: g.pipes[add], GroupsX: groups(g.frames*g.channels, 256), GroupsY: 1,
			PushConstants: pcAdd.bytes()},
	}
	return d, []string{"leaky", "upsample", "upadd"}, nil
}

// stageDispatches is a whole upsampling stage: the noise block, the
// upsampler, and the three residual blocks summed into g.aSum.
//
// The sum is left undivided. Whatever consumes it — the host, or the tail —
// carries the 1/NumKernels, because every consumer is either linear or
// positively homogeneous and so can absorb it into a weight.
func (g *GPUBlocks) stageDispatches() ([]vk.MultiDispatch, error) {
	if g.up == nil {
		return nil, fmt.Errorf("kokoro: this block set has no upsampler")
	}
	// The excitation's residual block runs in its own arena, in place, so it
	// can be recorded before the upsampler and never crosses the bus.
	noise, _, err := g.graphAt(noiseBlock, g.aNoise)
	if err != nil {
		return nil, err
	}
	front, _, err := g.upGraph()
	if err != nil {
		return nil, err
	}
	d := append(noise, front...)
	for j, b := range []int{0, 1, 2} {
		g0, _, err := g.graph(b)
		if err != nil {
			return nil, err
		}
		d = append(d, g.elementwise("copy", g.aIn, g.aX, noW))
		d = append(d, g0...)
		if j == 0 {
			d = append(d, g.elementwise("copy", g.aX, g.aSum, noW))
		} else {
			d = append(d, g.elementwise("residual", g.aX, g.aSum, noW))
		}
	}
	return d, nil
}

// RunStage runs a whole upsampling stage over the uploaded input and
// excitation. One submit, one download.
func (g *GPUBlocks) RunStage() (*Mat, error) {
	d, err := g.stageDispatches()
	if err != nil {
		return nil, err
	}
	if err := submit(d); err != nil {
		return nil, err
	}
	out := &Mat{Rows: g.frames, Cols: g.channels,
		Data: g.abuf.ReadFloat32At(int(g.aSum), g.frames*g.channels)}
	inv := 1 / float32(3)
	for i := range out.Data {
		out.Data[i] *= inv
	}
	return out, nil
}

// HasTail reports whether this stage carries conv_post and the inverse
// transform.
func (g *GPUBlocks) HasTail() bool { return g.tail != nil }

// Resident reports whether this stage's input and output are already in the
// arena its neighbours read — so there is nothing to upload and nothing to
// bring back.
func (g *GPUBlocks) Resident() bool { return g.shared != nil }

// RunStageResident runs the stage and leaves its result where the next one
// will read it. The sum is left undivided; see stageDispatches.
func (g *GPUBlocks) RunStageResident() error {
	d, err := g.stageDispatches()
	if err != nil {
		return err
	}
	return submit(d)
}

// Samples is how many the tail produces.
func (g *GPUBlocks) Samples() int {
	if g.tail == nil {
		return 0
	}
	return g.tail.samples
}

// RunStageWave runs the stage and its tail, and returns the waveform.
//
// This is the whole point of staging the tail: what comes back is 78000
// samples, 312 KB, where the stage's own [15601, 128] output is 8 MB — and
// the arena it would come out of reads at 210 MB/s, so the download was
// 38 ms of a 117 ms vocoder.
func (g *GPUBlocks) RunStageWave() ([]float32, error) {
	if g.tail == nil {
		return nil, fmt.Errorf("kokoro: this stage has no tail")
	}
	d, err := g.stageDispatches()
	if err != nil {
		return nil, err
	}
	t, _, err := g.tailGraph()
	if err != nil {
		return nil, err
	}
	if err := submit(append(d, t...)); err != nil {
		return nil, err
	}
	return g.abuf.ReadFloat32At(int(g.tail.aWav), g.tail.samples), nil
}

// tailRung is the widest rung of the ladder whose tile divides conv_post's
// padded output width, which is 32 — so the 64- and 128-wide rungs the
// residual blocks run on cannot be used here.
func (g *GPUBlocks) tailRung() (convVariant, bool) {
	var best convVariant
	var found bool
	for _, v := range convVariants {
		if _, ok := g.convs[v.name]; !ok || g.tail.n%v.bn != 0 {
			continue
		}
		if !found || v.bn > best.bn {
			best, found = v, true
		}
	}
	return best, found
}

// tailGraph is the four dispatches that turn the stage's running sum into
// samples: the rectifier and the narrow, conv_post, the epilogue that makes a
// rectangular spectrum of it, and the inverse transform.
func (g *GPUBlocks) tailGraph() ([]vk.MultiDispatch, []string, error) {
	t := g.tail
	v, ok := g.tailRung()
	if !ok {
		return nil, nil, fmt.Errorf("kokoro: no rung tiles %d output columns", t.n)
	}
	c := uint32(g.channels)
	logC := uint32(math.Log2(float64(g.channels)))
	m := roundUp(g.frames, v.bm)

	pcLeaky := pushConstants{
		Tokens: uint32(g.frames), Dim: c, Aux2: logC,
		InOff: g.aSum, OutOff: g.hA, LDA: uint32(g.lda),
		Scale: math.Float32bits(0.01), // F.leaky_relu's default slope
	}
	pcConv := pushConstants{
		InOff: g.hA - uint32(t.pad*g.lda), OutOff: t.aPost, BOff: t.bank,
		GemmM: uint32(m), GemmN: uint32(t.n), GemmK: uint32(t.taps * g.channels),
		LDA:  uint32(g.lda),
		Aux0: c, Aux1: c, Aux2: logC,
	}
	pcPost := pushConstants{
		InOff: t.aPost, OutOff: t.aRI, WOff: t.bias,
		Dim: uint32(t.bins), LDA: uint32(t.n), Aux0: uint32(t.nfft),
	}
	pcISTFT := pushConstants{
		InOff: t.aRI, OutOff: t.aWav, WOff: t.tw,
		Tokens: uint32(t.samples), Dim: uint32(t.bins),
		Aux0: uint32(t.nfft), Aux1: uint32(t.hop), Aux2: uint32(t.frames),
	}
	d := []vk.MultiDispatch{
		{Pipeline: g.pipes["leaky"], GroupsX: groups(g.frames*g.channels, 256), GroupsY: 1,
			PushConstants: pcLeaky.bytes()},
		{Pipeline: g.convs[v.name], GroupsX: uint32(t.n / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pcConv.bytes()},
		{Pipeline: g.pipes["post"], GroupsX: groups(t.bins, 256), GroupsY: uint32(t.frames),
			PushConstants: pcPost.bytes()},
		{Pipeline: g.pipes["istft"], GroupsX: groups(t.samples, 256), GroupsY: 1,
			PushConstants: pcISTFT.bytes()},
	}
	return d, []string{"leaky", "conv_post", "post", "istft"}, nil
}
