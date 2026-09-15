package llm

// The hyper-connection block on the device: LLM.md L2's kernel, against the
// CPU reference in hc.go and llama.cpp's own tensors behind it.
//
// L2a made this the vertical's first kernel rather than its last. Per-op
// attribution of llama.cpp's prefill graph puts 30.6% in elementwise glue,
// 12.2% in tiny-N F32 matmul and ~7% in this block's low-rank pair — over
// half the graph, against 1.6% for the gated DeltaNet that three quarters of
// the layers are made of. Almost none of it is arithmetic, so what this file
// is about is how few dispatches and how little DRAM traffic the block can be
// expressed in:
//
//	norm      res -> xn, fp16, in the GEMM's A layout        RMS_NORM + MUL + CONT
//	down      xn -> silu(lo/hc) and inject, one matmul       MUL_MAT x2 + SCALE + SILU
//	up        lo -> sigmoid -> collapsed into mixed          MUL_MAT + SIGMOID + MUL + 3 ADD
//	combine   res += out * 2*sigmoid(inject/hc)              SCALE + SIGMOID + REPEAT + MUL + ADD
//
// Four dispatches against about sixteen, and two tensors that never exist:
// `inject` is four more output columns on the down projection (the
// reference's single worst line — 95 dispatches a graph at 31.8 GFLOP/s,
// 10.3% of prefill) and the [10240, T] gate is consumed inside the up
// projection's epilogue instead of being written to DRAM, which at 512 tokens
// is 21 MB — the whole MALL — per mixer.
//
// The arena and staging machinery is zimage/qwen's, unchanged: four buffers
// bound to every pipeline, tensors addressed by offset, weights packed once
// into the §2.8 fragment tiling, M padded up to the tile because the kernels
// have no bounds checks.

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// coopMatTile is the cooperative-matrix extent, 16x16x16, which is the only
// shape this device reports.
const coopMatTile = 16

// gemmPad is the leading-dimension pad in halves, §2.3's 256 B, on every A
// operand: a K-strided fragment load issues 16 addresses one row stride
// apart, and both 10240 and 320 are multiples of 256.
const gemmPad = 128

// push mirrors the PC block of shaders/llm_common.glsl. One size for every
// pipeline in the vertical, because vk.DispatchMultiTimed records a sequence
// into a single command buffer only if every layout declares the same range —
// so the PLE block's fields sit here too, unused by the hyper-connection
// kernels and zero in their dispatches.
type push struct {
	ResOff, XnOff, LoOff, InjOff uint32
	OutOff, GammaOff, BOff       uint32
	GateOff                      uint32
	Tokens, NEmbd, HC, LowRank   uint32
	LDA, LDALo                   uint32
	GemmM, GemmN, GemmK          uint32
	Eps                          uint32

	KVOff, GatedOff, NormOff        uint32
	ConvOff, ConvOutOff             uint32
	GammaQOff, GammaCOff, Kern, Dil uint32
}

func (p push) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*push)(unsafe.Pointer(&out[0])) = p
	return out
}

// noW marks an optional tensor as absent, as NO_W does in the shader.
const noW = 0xffffffff

// HCKernel names one build of shaders/llm_gemm.comp. The two projections
// have different N and different epilogues, so they have separate ladders;
// within each, the rungs differ only in how many token rows a workgroup
// carries, which is what decides occupancy at a short prompt and reuse at a
// long one.
type HCKernel string

const (
	HCDownM1 HCKernel = "down_m1"
	HCDownM2 HCKernel = "down_m2"
	HCDownM4 HCKernel = "down_m4"
	HCUpM1   HCKernel = "up_m1"
	HCUpM2   HCKernel = "up_m2"
	HCUpM4   HCKernel = "up_m4"
)

// hcVariant is one build's geometry, which the host has to be told because BM
// and BN are compiled into the SPIR-V.
type hcVariant struct {
	name   HCKernel
	spirv  []byte
	mode   int
	bm, bn int
}

var hcVariants = []hcVariant{
	{name: HCDownM1, spirv: shaders.LLMHCDownM1, mode: 0, bm: 16, bn: 48},
	{name: HCDownM2, spirv: shaders.LLMHCDownM2, mode: 0, bm: 32, bn: 48},
	{name: HCDownM4, spirv: shaders.LLMHCDownM4, mode: 0, bm: 64, bn: 48},
	{name: HCUpM1, spirv: shaders.LLMHCUpM1, mode: 1, bm: 16, bn: 64},
	{name: HCUpM2, spirv: shaders.LLMHCUpM2, mode: 1, bm: 32, bn: 64},
	{name: HCUpM4, spirv: shaders.LLMHCUpM4, mode: 1, bm: 64, bn: 64},
}

// DownKernels and UpKernels list the rungs of each ladder, narrowest first.
func DownKernels() []HCKernel { return []HCKernel{HCDownM1, HCDownM2, HCDownM4} }
func UpKernels() []HCKernel   { return []HCKernel{HCUpM1, HCUpM2, HCUpM4} }

// DefaultPlan is the best single pair over the whole prompt range, measured
// by `cmd/llm -hc -ladder` (results/l2c_hc.csv). It is never more than 1.04x
// off the rung that wins at a given length, where the worst pair in the table
// is 1.21-1.29x off.
func DefaultPlan() (HCKernel, HCKernel) { return HCDownM2, HCUpM2 }

// PlanFor is the measured schedule: which pair wins at a given prompt length.
//
//	tokens   winner              runner-up
//	    64   m1/up_m4  1.00x     m1/up_m1  1.01x
//	   128   m1/up_m2  1.00x     m1/up_m4  1.03x
//	   256   m1/up_m2  1.00x     m1/up_m4  1.03x
//	   512   m2/up_m2  1.00x     m1/up_m2  1.01x
//	  1024   m2/up_m2  1.00x     m4/up_m2  1.03x
//	  2048   m2/up_m2  1.00x     m4/up_m2  1.00x
//
// The down projection's ladder moves because its cost barely depends on the
// token count at all — 252 us at 64 tokens and 255 at 512, because what it
// reads is 6.9 MB of weight per M block and not the activation — so the rung
// that wins is the one that fills the machine, until there are enough tokens
// that reading the weight fewer times matters more. The up projection's does
// not: BM=32 wins from 128 tokens on.
//
// The boundaries sit between measured points, and the cost of getting one
// wrong is at most 1.04x on either side.
func PlanFor(tokens int) (HCKernel, HCKernel) {
	switch {
	case tokens <= 96:
		return HCDownM1, HCUpM4
	case tokens <= 384:
		return HCDownM1, HCUpM2
	default:
		return HCDownM2, HCUpM2
	}
}

func hcVariantFor(k HCKernel) (hcVariant, bool) {
	for _, v := range hcVariants {
		if v.name == k {
			return v, true
		}
	}
	return hcVariant{}, false
}

// hcMixer is where one mixer's weights sit in the arenas.
type hcMixer struct {
	gamma uint32 // fp32 arena
	down  uint32 // fp16 bank: the fused [lowRank + hc, wide] matrix
	up    uint32 // fp16 bank: the permuted [wide, lowRank] matrix
}

// HCGPU runs hyper-connection mixers on the device. It holds however many
// mixers it was built with — at L2 that is the four dumped layers' worth, at
// L6 it will be all 97 — plus one set of activation arenas sized for the
// longest run.
type HCGPU struct {
	dev *vk.Device
	cfg HCConfig

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	mixers []hcMixer

	down, up HCKernel
	// autoPlan re-plans every run for its own length, using PlanFor. It is
	// free — every rung reads the same staged weight, so it changes which
	// pipeline a dispatch names and nothing else — and SetPlan turns it off,
	// because a caller that named a pair meant it.
	autoPlan bool

	// tokens is the longest run the arenas were built for, arenaRows its
	// rounding up to the widest tile any rung covers, and rows the length of
	// the run in progress.
	tokens, arenaRows, rows int
	lda, ldaLo              int

	// fp32 arena.
	aRes, aInject, aMixed, aOut, aGate uint32
	actElems                           int
	// fp16 arena.
	hXn, hLo uint32
	hElems   int

	// gate is whether the up projection's [wide, T] gate has somewhere to be
	// written. It is validation machinery — the whole point of the kernel is
	// that the tensor does not exist — so it costs an arena only when asked
	// for.
	gate bool
	// ctl is the construction-time options, kept because one of them changes
	// how the weights were staged and a reader of a wrong tensor should be
	// able to ask.
	ctl HCOpts
}

// downBN is the down ladder's column block. Its N is lowRank + one fragment
// tile = 336 = 21 tiles, and 21 is 3 x 7, so 48 is the widest block that
// divides it — which is why that ladder moves BM alone.
const downBN = 48

// injStride is the inject tensor's row stride: the down projection's last
// fragment tile, not hc. The kernel stores a whole 16-wide tile of which the
// first hc columns mean anything.
func (g *HCGPU) injStride() int { return coopMatTile }

// gemmN is the fused down projection's output width: lowRank plus the tile
// that carries inject, padded up to the down ladder's BN.
func (g *HCGPU) gemmN() int {
	return roundUpInt(g.cfg.LowRank+g.injStride(), downBN)
}

// HCOpts are the choices that change what the block allocates, which is why
// they are made at construction and not per run.
type HCOpts struct {
	// Gate makes the up projection write the 10240-wide gate as well as the
	// collapsed output, into an arena allocated for it. It exists so the
	// fused kernel can be checked against `hc_gate-N` tensor for tensor; the
	// graph the model runs leaves it off, which is the point of the fusion —
	// at 512 tokens that tensor is 21 MB a mixer, the whole MALL.
	Gate bool
	// UnpermutedUp stages the up projection in the checkpoint's own row
	// order while the kernel goes on reading the permuted one. It is a
	// negative control, in the spirit of zimage/qwen's: the collapse depends
	// on a *layout*, and a layout that is load-bearing has to be shown to be
	// — an unpermuted stage produces a perfectly plausible tensor, so nothing
	// but a test that demands it disagree can tell the two apart.
	UnpermutedUp bool
}

// NewHCGPU stages mixers onto the device and builds every pipeline the block
// needs. The weights are copied into the arenas here, so the caller may drop
// them afterwards.
func NewHCGPU(dev *vk.Device, cfg HCConfig, maxTokens int, mixers []HCWeights, opts HCOpts) (*HCGPU, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("llm: maxTokens is %d", maxTokens)
	}
	if len(mixers) == 0 {
		return nil, fmt.Errorf("llm: no mixers to stage")
	}
	// HC_STREAMS is compiled into the collapse and cut into the up
	// projection's weight permutation, so a checkpoint with another stream
	// count would be answered wrongly rather than slowly.
	if cfg.HC != 4 {
		return nil, fmt.Errorf("llm: the fused kernel is built for hc=4, this checkpoint says %d", cfg.HC)
	}
	if cfg.NEmbd%coopMatTile != 0 || cfg.LowRank%coopMatTile != 0 {
		return nil, fmt.Errorf("llm: n_embd %d and low rank %d must be multiples of %d",
			cfg.NEmbd, cfg.LowRank, coopMatTile)
	}
	ok, err := canWMMA(dev)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("llm: this device has no 16x16x16 fp16 cooperative matrix")
	}
	g := &HCGPU{
		dev: dev, cfg: cfg,
		pipes:  make(map[string]*vk.ComputePipeline),
		tokens: maxTokens, rows: maxTokens,
		gate: opts.Gate, ctl: opts,
		autoPlan: true,
		lda:      cfg.Wide() + gemmPad,
		ldaLo:    cfg.LowRank + gemmPad,
	}
	align := coopMatTile
	for _, v := range hcVariants {
		align = maxInt(align, v.bm)
	}
	g.arenaRows = roundUpInt(maxTokens, align)
	g.down, g.up = DefaultPlan()

	if err := g.alloc(len(mixers)); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.build(); err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(mixers); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// WantGate reports whether this block was built to materialise the gate.
func (g *HCGPU) WantGate() bool { return g.gate }

// alloc lays out the four arenas. Nothing is read here — every size follows
// from the config — so the whole layout can be checked before a weight is
// touched.
func (g *HCGPU) alloc(nMixers int) error {
	c := g.cfg
	rows := g.arenaRows

	total32 := nMixers * c.Wide()
	var err error
	if g.wbuf, err = g.dev.NewBuffer(total32 * 4); err != nil {
		return fmt.Errorf("llm: fp32 weight arena: %w", err)
	}

	alloc := func(n int) uint32 {
		off := uint32(g.actElems)
		g.actElems += (n + 63) &^ 63
		return off
	}
	g.aRes = alloc(rows * c.Wide())
	g.aMixed = alloc(rows * c.NEmbd)
	g.aOut = alloc(rows * c.NEmbd)
	g.aInject = alloc(rows * g.injStride())
	g.aGate = noW
	if g.gate {
		g.aGate = alloc(rows * c.Wide())
	}
	if g.abuf, err = g.dev.NewBuffer(g.actElems * 4); err != nil {
		return fmt.Errorf("llm: fp32 activation arena (%d MB): %w", (g.actElems*4)>>20, err)
	}

	halloc := func(n int) uint32 {
		off := uint32(g.hElems)
		g.hElems += (n + 63) &^ 63
		return off
	}
	g.hXn = halloc(rows * g.lda)
	g.hLo = halloc(rows * g.ldaLo)
	if g.hbuf, err = g.dev.NewBuffer(g.hElems * 2); err != nil {
		return fmt.Errorf("llm: fp16 activation arena (%d MB): %w", (g.hElems*2)>>20, err)
	}
	// Zeroed once. The pad columns of every A operand and the pad rows of a
	// short run come from here, and the kernels have no bounds checks.
	g.hbuf.WriteFloat32(make([]float32, g.hElems/2))
	g.abuf.WriteFloat32(make([]float32, g.actElems))

	perMixer := g.gemmN()*c.Wide() + c.Wide()*c.LowRank
	if g.bank, err = g.dev.NewBuffer(nMixers * perMixer * 2); err != nil {
		return fmt.Errorf("llm: fp16 weight bank (%d MB): %w", (nMixers*perMixer*2)>>20, err)
	}
	g.mixers = make([]hcMixer, nMixers)
	for i := range g.mixers {
		g.mixers[i] = hcMixer{
			gamma: uint32(i * c.Wide()),
			down:  uint32(i * perMixer),
			up:    uint32(i*perMixer + g.gemmN()*c.Wide()),
		}
	}
	return nil
}

// build compiles every pipeline over all four arenas, bound whether the
// shader declares them or not, so one descriptor layout and one push-constant
// size serve the whole sequence.
func (g *HCGPU) build() error {
	bufs := []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.bank}
	pcSize := uint32(unsafe.Sizeof(push{}))
	for name, spirv := range map[string][]byte{
		"norm":    shaders.LLMHCNorm,
		"combine": shaders.LLMHCCombine,
	} {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			return err
		}
	}
	// Every rung is built, so a ladder is one staging and a different
	// pipeline per dispatch rather than a reload.
	feat := g.dev.Features()
	sgs, err := g.dev.Physical().SubgroupSizeControl()
	if err != nil {
		return fmt.Errorf("llm: subgroup size control: %w", err)
	}
	if !feat.SubgroupSizeControl || !sgs.Supported || sgs.MaxSubgroupSize < 64 {
		return fmt.Errorf("llm: the GEMM rungs need a pinned 64-wide subgroup")
	}
	for _, v := range hcVariants {
		if v.mode == 0 && v.bn != downBN {
			return fmt.Errorf("llm: down rung %q has BN %d, but the fused N is padded to %d",
				v.name, v.bn, downBN)
		}
		if v.mode == 1 && v.bn%(coopMatTile*g.cfg.HC) != 0 {
			return fmt.Errorf("llm: up rung %q has BN %d, which is not whole feature blocks", v.name, v.bn)
		}
		if err := g.pipeline(string(v.name), v.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: 64,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (g *HCGPU) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := g.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("llm: shader %s: %w", name, err)
	}
	g.mods = append(g.mods, mod)
	pipe, err := g.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("llm: pipeline %s: %w", name, err)
	}
	g.pipes[name] = pipe
	return nil
}

// stage fills the arenas: gamma fp32, and the two packed fp16 matrices.
func (g *HCGPU) stage(mixers []HCWeights) error {
	c := g.cfg
	wide, lr, n := c.Wide(), c.LowRank, g.gemmN()
	for i, w := range mixers {
		if len(w.Norm) != wide {
			return fmt.Errorf("llm: mixer %d norm is %d, want %d", i, len(w.Norm), wide)
		}
		if len(w.Down) != lr*wide {
			return fmt.Errorf("llm: mixer %d down is %d, want %d", i, len(w.Down), lr*wide)
		}
		if len(w.Up) != wide*lr {
			return fmt.Errorf("llm: mixer %d up is %d, want %d", i, len(w.Up), wide*lr)
		}
		if w.Inject != nil && len(w.Inject) != c.HC*wide {
			return fmt.Errorf("llm: mixer %d inject is %d, want %d", i, len(w.Inject), c.HC*wide)
		}
		g.wbuf.WriteFloat32At(int(g.mixers[i].gamma), w.Norm)

		down := make([]uint16, n*wide)
		packDownB(down, w.Down, w.Inject, lr, c.HC, wide)
		g.bank.WriteUint16At(int(g.mixers[i].down), down)

		up := make([]uint16, wide*lr)
		if g.ctl.UnpermutedUp {
			tileB(up, w.Up, wide, lr, func(o int) int { return o })
		} else {
			packUpB(up, w.Up, wide, lr, c.NEmbd)
		}
		g.bank.WriteUint16At(int(g.mixers[i].up), up)
	}
	return nil
}

// packChunk is how many output rows one worker narrows at a time.
const packChunk = 64

// tileB writes one [n, k] row-major matrix into the §2.8 fragment tiling:
// tile (nt, kt) is 256 contiguous halves holding element (k, n) at
// (n%16)*16 + k%16, tiles ordered kt-fastest. row(i) says which *packed* row
// output row i of the source occupies, which is how the up projection's
// permutation is expressed without a second copy of the matrix.
func tileB(dst []uint16, src []float32, n, k int, row func(int) int) {
	const tile = coopMatTile
	kt := k / tile
	parallelFor((n+packChunk-1)/packChunk, func(ch int) {
		for i := ch * packChunk; i < minInt((ch+1)*packChunk, n); i++ {
			r := row(i)
			base := (r / tile) * kt * tile * tile
			lane := (r % tile) * tile
			for j, v := range src[i*k : (i+1)*k] {
				dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(v)
			}
		}
	})
}

// packDownB fuses the two projections that read the same xn into one weight.
//
// This is L2a's finding 3 made into a layout: `w_inject` is a [wide, hc] F32
// matrix applied to the *same* normalised activation `w_down` reads, and
// llama.cpp runs it as its own MUL_MAT at 31.8 GFLOP/s for 10.3% of prefill.
// Here it is rows lowRank..lowRank+hc of a matrix that already has lowRank of
// them, and the kernel tells them apart by which fragment tile a column falls
// in. The rows between hc and the tile, and any tile the ladder's BN pads on
// the end, are left zero.
func packDownB(dst []uint16, down, inject []float32, lowRank, hc, wide int) {
	tileB(dst, down, lowRank, wide, func(i int) int { return i })
	if inject != nil {
		tileB(dst, inject, hc, wide, func(i int) int { return lowRank + i })
	}
}

// packUpB permutes the up projection's output rows so that the four streams
// of one feature land in the same workgroup, which is what lets the kernel
// collapse the gate instead of writing it.
//
// Output row o of the checkpoint is stream c = o/nEmbd, feature i = o%nEmbd.
// It is packed at (i/16)*64 + c*16 + i%16, so a 64-column block is 16
// features x 4 streams and the kernel's epilogue reads column c*16 + r of the
// block for stream c of feature (block*16 + r).
func packUpB(dst []uint16, up []float32, wide, lowRank, nEmbd int) {
	const tile = coopMatTile
	hc := wide / nEmbd
	tileB(dst, up, wide, lowRank, func(o int) int {
		c, i := o/nEmbd, o%nEmbd
		return (i/tile)*(tile*hc) + c*tile + i%tile
	})
}

// SetPlan chooses which rung each projection runs on. Every rung is built and
// they all read the same staged weight, so this moves a pipeline and a tile
// and restages nothing.
func (g *HCGPU) SetPlan(down, up HCKernel) error {
	dv, ok := hcVariantFor(down)
	if !ok || dv.mode != 0 {
		return fmt.Errorf("llm: %q is not a down-projection kernel (have %v)", down, DownKernels())
	}
	uv, ok := hcVariantFor(up)
	if !ok || uv.mode != 1 {
		return fmt.Errorf("llm: %q is not an up-projection kernel (have %v)", up, UpKernels())
	}
	g.down, g.up = down, up
	g.autoPlan = false
	return nil
}

// Plan reports the rungs in use.
func (g *HCGPU) Plan() (down, up HCKernel) { return g.down, g.up }

// Mixers is how many mixers are staged and Tokens the longest run the arenas
// were built for.
func (g *HCGPU) Mixers() int { return len(g.mixers) }
func (g *HCGPU) Tokens() int { return g.tokens }

// WeightBytes is what the staged mixers cost on the device and
// ActivationBytes what the shared arenas cost.
func (g *HCGPU) WeightBytes() int     { return g.wbuf.Size() + g.bank.Size() }
func (g *HCGPU) ActivationBytes() int { return g.abuf.Size() + g.hbuf.Size() }

// Upload writes the wide residual for a run: [T][hc*nEmbd], which is ggml's
// [nEmbd, hc, T] read the same way round.
func (g *HCGPU) Upload(res []float32, nTok int) error {
	if nTok <= 0 || nTok > g.tokens {
		return fmt.Errorf("llm: %d tokens, arenas are built for %d", nTok, g.tokens)
	}
	if len(res) != nTok*g.cfg.Wide() {
		return fmt.Errorf("llm: residual is %d values, want %d", len(res), nTok*g.cfg.Wide())
	}
	g.rows = nTok
	if g.autoPlan {
		g.down, g.up = PlanFor(nTok)
	}
	g.abuf.WriteFloat32At(int(g.aRes), res)
	return nil
}

// UploadBlockOut writes the block output the combine scatters back.
func (g *HCGPU) UploadBlockOut(out []float32) error {
	if len(out) != g.rows*g.cfg.NEmbd {
		return fmt.Errorf("llm: block output is %d values, want %d", len(out), g.rows*g.cfg.NEmbd)
	}
	g.abuf.WriteFloat32At(int(g.aOut), out)
	return nil
}

// graph builds one mixer's dispatch sequence, with a label per dispatch, and
// is shared by Run and Profile so that what the profiler times is what a run
// executes. combine appends the scatter, which needs a block output.
func (g *HCGPU) graph(mixer int, combine bool) ([]vk.MultiDispatch, []string, error) {
	if mixer < 0 || mixer >= len(g.mixers) {
		return nil, nil, fmt.Errorf("llm: mixer %d of %d", mixer, len(g.mixers))
	}
	c := g.cfg
	m := g.mixers[mixer]
	dv, _ := hcVariantFor(g.down)
	uv, _ := hcVariantFor(g.up)

	base := push{
		ResOff: g.aRes, XnOff: g.hXn, LoOff: g.hLo, InjOff: g.aInject,
		OutOff: g.aMixed, GammaOff: m.gamma, GateOff: noW,
		Tokens: uint32(g.rows), NEmbd: uint32(c.NEmbd), HC: uint32(c.HC),
		LowRank: uint32(c.LowRank),
		LDA:     uint32(g.lda), LDALo: uint32(g.ldaLo),
		Eps: math.Float32bits(c.Eps),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	add("norm", "norm", uint32(g.rows), uint32(c.HC), base)

	down := base
	down.BOff = m.down
	down.GemmM, down.GemmN, down.GemmK = uint32(roundUpInt(g.rows, dv.bm)), uint32(g.gemmN()), uint32(c.Wide())
	add(string(g.down), "down", uint32(g.gemmN()/dv.bn), uint32(roundUpInt(g.rows, dv.bm)/dv.bm), down)

	up := base
	up.BOff = m.up
	up.GemmM, up.GemmN, up.GemmK = uint32(roundUpInt(g.rows, uv.bm)), uint32(c.Wide()), uint32(c.LowRank)
	if g.gate {
		up.GateOff = g.aGate
	}
	add(string(g.up), "up", uint32(c.Wide()/uv.bn), uint32(roundUpInt(g.rows, uv.bm)/uv.bm), up)

	if combine {
		cb := base
		cb.OutOff = g.aOut
		cb.GemmN = uint32(g.gemmN())
		add("combine", "combine", uint32(g.rows), uint32(c.HC), cb)
	}
	return d, kinds, nil
}

// perSubmit is how many dispatches go into one command buffer.
const perSubmit = 8

// Run executes one mixer over whatever Upload left in the residual.
func (g *HCGPU) Run(mixer int, combine bool) error {
	d, _, err := g.graph(mixer, combine)
	if err != nil {
		return err
	}
	for i := 0; i < len(d); i += perSubmit {
		j := minInt(i+perSubmit, len(d))
		if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("llm: mixer %d dispatch %d-%d: %w", mixer, i, j-1, err)
		}
	}
	return nil
}

// Stage is one timed dispatch of a profile.
type Stage struct {
	Kind string
	GPU  time.Duration
}

// Elapsed sums a profile.
func Elapsed(st []Stage) time.Duration {
	var total time.Duration
	for _, s := range st {
		total += s.GPU
	}
	return total
}

// Profile runs the same graph one dispatch at a time, each timed on the GPU
// over `iters` back-to-back repetitions. Wall clock around a run is not a
// measurement of the block: it carries the upload and the read-back, and this
// arena reads at 0.2 GB/s.
func (g *HCGPU) Profile(mixer int, combine bool, iters int) ([]Stage, error) {
	d, kinds, err := g.graph(mixer, combine)
	if err != nil {
		return nil, err
	}
	if iters <= 0 {
		iters = 1
	}
	out := make([]Stage, 0, len(d))
	for i := range d {
		dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, uint32(iters), true)
		if err != nil {
			return out, fmt.Errorf("llm: dispatch %d (%s): %w", i, kinds[i], err)
		}
		out = append(out, Stage{Kind: kinds[i], GPU: dur / time.Duration(iters)})
	}
	return out, nil
}

// ProfileSweep times each kind of dispatch across *every* staged mixer, and
// is the number to quote rather than Profile's.
//
// The difference is the cache. One mixer is 13.4 MB of fp16 weights, which
// sits inside the 32 MiB MALL, so repeating one dispatch measures a kernel
// reading L3 — where the graph this is a model of reads 97 different mixers
// and every one of them is cold. Sweeping the staged set puts that back, as
// long as the caller stages enough of them to overflow the MALL.
func (g *HCGPU) ProfileSweep(combine bool, iters int) ([]Stage, error) {
	if iters <= 0 {
		iters = 1
	}
	_, kinds, err := g.graph(0, combine)
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(kinds))
	for k := range kinds {
		var d []vk.MultiDispatch
		for m := range g.mixers {
			dm, _, err := g.graph(m, combine)
			if err != nil {
				return nil, err
			}
			d = append(d, dm[k])
		}
		var total time.Duration
		for i := 0; i < len(d); i += perSubmit {
			j := minInt(i+perSubmit, len(d))
			dur, err := vk.DispatchMultiTimed(d[i:j], 1, uint32(iters), true)
			if err != nil {
				return out, fmt.Errorf("llm: %s sweep %d-%d: %w", kinds[k], i, j-1, err)
			}
			total += dur
		}
		out = append(out, Stage{Kind: kinds[k], GPU: total / time.Duration(len(d)*iters)})
	}
	return out, nil
}

// The tensors a run leaves behind, in the layout hc.go's CPU reference uses,
// so a comparison is a call to compare and not a reshape.

// Mixed is the block's input, [T][nEmbd].
func (g *HCGPU) Mixed() []float32 { return g.abuf.ReadFloat32At(int(g.aMixed), g.rows*g.cfg.NEmbd) }

// Res is the wide residual, [T][hc*nEmbd] — what the combine updated.
func (g *HCGPU) Res() []float32 { return g.abuf.ReadFloat32At(int(g.aRes), g.rows*g.cfg.Wide()) }

// Inject is the scatter weights, [T][hc], gathered out of the tile-strided
// rows the down projection writes.
func (g *HCGPU) Inject() []float32 {
	raw := g.abuf.ReadFloat32At(int(g.aInject), g.rows*g.injStride())
	out := make([]float32, g.rows*g.cfg.HC)
	for t := 0; t < g.rows; t++ {
		copy(out[t*g.cfg.HC:(t+1)*g.cfg.HC], raw[t*g.injStride():])
	}
	return out
}

// Gate is the [T][hc*nEmbd] gate, and is empty unless the block was built to
// write it.
func (g *HCGPU) Gate() []float32 {
	if !g.gate {
		return nil
	}
	return g.abuf.ReadFloat32At(int(g.aGate), g.rows*g.cfg.Wide())
}

// Xn is the normalised activation, widened out of the fp16 arena. The row
// stride is the A operand's leading dimension, not the width.
func (g *HCGPU) Xn() []float32 {
	wide := g.cfg.Wide()
	raw := g.hbuf.ReadUint16At(int(g.hXn), (g.rows-1)*g.lda+wide)
	out := make([]float32, g.rows*wide)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.lda : t*g.lda+wide] {
			out[t*wide+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Lo is silu(down·xn / hc), the up projection's A operand, widened out of the
// fp16 arena.
func (g *HCGPU) Lo() []float32 {
	lr := g.cfg.LowRank
	raw := g.hbuf.ReadUint16At(int(g.hLo), (g.rows-1)*g.ldaLo+lr)
	out := make([]float32, g.rows*lr)
	for t := 0; t < g.rows; t++ {
		for i, h := range raw[t*g.ldaLo : t*g.ldaLo+lr] {
			out[t*lr+i] = safetensors.F16ToF32(h)
		}
	}
	return out
}

// Destroy releases every Vulkan object.
func (g *HCGPU) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{g.hbuf, g.abuf, g.wbuf, g.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// canWMMA reports whether this device has the cooperative-matrix shape the
// GEMM rungs are built for. A device reporting some other shape would run
// them *wrong* rather than slowly, so the answer is "no" rather than "try".
func canWMMA(dev *vk.Device) (bool, error) {
	feat := dev.Features()
	if !feat.Float16 || !feat.CoopMatrix {
		return false, nil
	}
	shapes, err := dev.Physical().CooperativeMatrixShapes()
	if err != nil {
		return false, fmt.Errorf("llm: cooperative-matrix shapes: %w", err)
	}
	for _, sh := range shapes {
		if sh.Scope == vk.ScopeSubgroup && sh.M == coopMatTile && sh.N == coopMatTile && sh.K == coopMatTile &&
			sh.AType == vk.ComponentFloat16 && sh.BType == vk.ComponentFloat16 &&
			sh.CType == vk.ComponentFloat32 && sh.ResultType == vk.ComponentFloat32 {
			return true, nil
		}
	}
	return false, nil
}

func roundUpInt(n, m int) int { return (n + m - 1) / m * m }

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
