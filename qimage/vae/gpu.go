package vae

import (
	"context"
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// The decoder on the device — IMAGE.md Q5g, the stage that stood between
// Q4's 2.27 s DiT step and a served image.
//
// The CPU decoder in vae.go is the oracle and is also the reason this file
// exists: it decodes 256x256 in ~20 s, and the graph is 16x larger at
// 1024x1024, so a served generation would have spent minutes downstream of
// a 91 s transformer.
//
// Almost none of it is new. z-image's decoder graph (zimage/vae/gpu.go)
// established the shape — one weight arena written once, one activation
// arena a bump allocator ping-pongs inside, "which tensor" as an offset
// rather than a descriptor rebind, and the whole decode recorded as a
// dispatch list — and its kernels do the work here unchanged: the
// register-blocked fp32 convolution, the 2x nearest upsample, the add, and
// the mid block's row-major attention with its transposed K. Three things
// in this VAE are genuinely not in that one, and each is one shader:
//
//   - the norm. RMS_norm here is a per-pixel L2 normalize across channels,
//     not a group norm, and it is always followed by a SiLU — qvae_chnorm
//     does both in one pass;
//   - the up blocks' parameter-free channel-to-space shortcut, whose
//     channel mapping depends on a video model's temporal factor even
//     though no temporal axis survives at T = 1 — qvae_dupup;
//   - the mid block is 1152 channels wide where z-image's is 512, which the
//     scalar attention kernel needs as a compile-time bound.
//
// The mid block's four projections are 1x1 convolutions, which is to say
// linears over the pixel rows, and to_qkv's [3C, C] weight is three of them
// stacked — so the graph reads it at three offsets rather than splitting the
// tensor. That is why there is no qkv convolution in the dispatch list.
//
// Everything here is fp32. Q0 measured this decoder's intermediates at
// absmax ~1e5–3e5 against fp16's 65504, so the matrix-core convolution that
// took z-image's decode from 3.18 s to 258 ms cannot simply be pointed at
// this graph; what it would take is priced in gpu_test.go's range
// measurement, and it is Q9's, not Q5g's.

// pushConstants mirrors the PC block in shaders/vae_common.glsl. Every
// pipeline in the graph declares the same block, because vk.DispatchMultiTimed
// records the whole thing into one command buffer and requires one
// push-constant size across it.
type pushConstants struct {
	InOff, OutOff    uint32
	C, H, W          uint32
	OC, KH, KW, Pad  uint32
	WOff, BOff       uint32
	Groups           uint32
	ResOff           uint32
	Aux0, Aux1, Aux2 uint32
	// The GEMM block, unused on this path but part of the layout: the block's
	// size is fixed by the shaders the graph borrows.
	GemmB, GemmM, GemmN, GemmK uint32
	LDA, LDB                   uint32
}

const noBias = 0xffffffff

// dispatchesPerSubmit caps how much work goes into one command buffer. The
// whole graph in a single submit trips the driver's reset watchdog and comes
// back as VK_ERROR_DEVICE_LOST; the same dispatches submitted in batches all
// complete. 4 is z-image's measured-free value and this graph's dispatches
// are no larger than that one's.
const dispatchesPerSubmit = 4

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// arena hands out element offsets inside one activation buffer, with a
// best-fit free list. It is zimage/vae's, for the reason that file gives:
// an up block's input is twice the channel count of what its resnets
// produce, so exact-size bucketing fills the arena with large blocks that
// can never serve the smaller requests that follow.
type arena struct {
	next uint32
	high uint32
	free []freeBlock
}

type freeBlock struct {
	off  uint32
	size uint32
}

func roundUp(elems int) uint32 { return uint32((elems + 63) &^ 63) }

func (a *arena) alloc(elems int) uint32 {
	n := roundUp(elems)
	best := -1
	for i, b := range a.free {
		if b.size >= n && (best < 0 || b.size < a.free[best].size) {
			best = i
		}
	}
	if best >= 0 {
		blk := a.free[best]
		a.free = append(a.free[:best], a.free[best+1:]...)
		if rem := blk.size - n; rem >= 64 {
			a.free = append(a.free, freeBlock{off: blk.off + n, size: rem})
		}
		return blk.off
	}
	off := a.next
	a.next += n
	if a.next > a.high {
		a.high = a.next
	}
	return off
}

// release returns a block, coalescing with neighbours and giving back to the
// bump pointer anything that now ends at it.
func (a *arena) release(off uint32, elems int) {
	blk := freeBlock{off: off, size: roundUp(elems)}
	for merged := true; merged; {
		merged = false
		for i, b := range a.free {
			if b.off+b.size == blk.off {
				blk.off, blk.size = b.off, b.size+blk.size
			} else if blk.off+blk.size == b.off {
				blk.size += b.size
			} else {
				continue
			}
			a.free = append(a.free[:i], a.free[i+1:]...)
			merged = true
			break
		}
	}
	a.free = append(a.free, blk)
	for {
		i := -1
		for j, b := range a.free {
			if b.off+b.size == a.next {
				i = j
				break
			}
		}
		if i < 0 {
			return
		}
		a.next = a.free[i].off
		a.free = append(a.free[:i], a.free[i+1:]...)
	}
}

func (a *arena) reset() {
	a.next = 0
	a.free = nil
}

// gpuWeights is the flat fp32 weight arena plus every tensor's offset in it.
type gpuWeights struct {
	data []float32
	off  map[string]uint32
}

func (g *gpuWeights) put(name string, v []float32) {
	g.off[name] = uint32(len(g.data))
	g.data = append(g.data, v...)
}

// midDim is the channel width the attention kernel was built for. The build
// is shaders.VAEAttentionDim1152 and the bound is a shared-array extent, so
// a checkpoint with a different mid width would silently write past it — it
// is refused at load instead.
const midDim = 1152

// engine is the Vulkan half of a graph on this device: the two buffers, the
// pipelines, the flattened weights and the activation arena. The decoder and
// the encoder are two graphs over one set of kernels, so this is what they
// share — the same split zimage/vae made between its own two directions.
type engine struct {
	dev *vk.Device

	wbuf *vk.Buffer
	abuf *vk.Buffer

	pipes   map[string]*vk.ComputePipeline
	mods    []*vk.ShaderModule
	weights *gpuWeights

	// The two kernels Q9b screened; see kernels.go. Both graphs resolve them
	// the same way, which is why they live here.
	conv convVariant
	mid  midVariant

	arena arena
}

// stage uploads the weights, sizes the arena and builds every pipeline.
func (e *engine) stage(dev *vk.Device, set map[string][]byte, arenaElems uint32) error {
	e.dev = dev
	var err error
	wbytes := len(e.weights.data) * 4
	if e.wbuf, err = dev.NewBuffer(wbytes); err != nil {
		return fmt.Errorf("qvae: weight buffer (%d MB): %w", wbytes>>20, err)
	}
	e.wbuf.WriteFloat32(e.weights.data)
	if e.abuf, err = dev.NewBuffer(int(arenaElems) * 4); err != nil {
		return fmt.Errorf("qvae: activation buffer (%d MB): %w", (int(arenaElems)*4)>>20, err)
	}
	bufs := []*vk.Buffer{e.wbuf, e.abuf}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	for name, spirv := range set {
		mod, err := dev.NewShaderModule(spirv)
		if err != nil {
			return fmt.Errorf("qvae: shader %s: %w", name, err)
		}
		e.mods = append(e.mods, mod)
		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("qvae: pipeline %s: %w", name, err)
		}
		e.pipes[name] = pipe
	}
	return nil
}

func (e *engine) destroy() {
	for _, p := range e.pipes {
		p.Destroy()
	}
	for _, m := range e.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{e.abuf, e.wbuf} {
		if b != nil {
			b.Destroy()
		}
	}
	e.pipes, e.mods = map[string]*vk.ComputePipeline{}, nil
	e.abuf, e.wbuf = nil, nil
}

// run submits a prefix of a recorded graph, to completion.
func (e *engine) run(ds []vk.MultiDispatch) error {
	return e.runContext(context.Background(), ds)
}

// runContext is run with a cancellation point, and the batching this graph
// already does for the driver's sake is what provides one.
//
// A submitted command buffer has no cancellation point inside it — the fence
// is waited on or the device is left mid-graph — so the finest granularity
// available is *between* submits. That is already every four dispatches
// (dispatchesPerSubmit, a watchdog constraint rather than a choice), which
// puts a 1024² decode's 117 dispatches at 30 batches of ~250 ms. So a client
// that hangs up during a decode stops paying for it within a quarter of a
// second instead of 7.5 s, and the device lock is released that much sooner
// for whoever is queued behind them.
//
// The error wraps ctx.Err(), so `errors.Is(err, context.Canceled)` holds all
// the way up to the handler, which is where it becomes "no answer" rather
// than a 500.
func (e *engine) runContext(ctx context.Context, ds []vk.MultiDispatch) error {
	for i := 0; i < len(ds); i += dispatchesPerSubmit {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("qvae: cancelled after %d of %d dispatches: %w", i, len(ds), err)
		}
		j := min(i+dispatchesPerSubmit, len(ds))
		if _, err := vk.DispatchMultiTimed(ds[i:j], 1, 1, true); err != nil {
			return fmt.Errorf("qvae: dispatch %d-%d: %w", i, j-1, err)
		}
	}
	return nil
}

// read copies one arena tensor back.
func (e *engine) read(t tensor) *Tensor {
	out := NewTensor(1, t.C, t.H, t.W)
	copy(out.Data, e.abuf.ReadFloat32At(int(t.off), t.elems()))
	return out
}

// ActivationBytes and WeightBytes are what the graph costs in device memory.
func (e *engine) ActivationBytes() int { return e.abuf.Size() }
func (e *engine) WeightBytes() int     { return e.wbuf.Size() }

// newEngine is an engine with its maps made but nothing staged, which is also
// what the sizing passes record against. The kernels are the screen's winners
// from the start, so that a sizing pass — which never calls chooseKernels,
// having no device and no options — records the graph the real one will run.
func newEngine() engine {
	e := engine{
		pipes:   make(map[string]*vk.ComputePipeline),
		weights: &gpuWeights{off: make(map[string]uint32)},
	}
	if err := e.chooseKernels(Options{}); err != nil {
		panic(err) // the defaults are constants; this cannot fail at runtime
	}
	return e
}

// GPUDecoder runs the decode graph on a Vulkan device.
type GPUDecoder struct {
	engine
	cpu *Decoder
}

// shaderSet is every pipeline the graph uses. Six of the eight are z-image's
// unchanged.
var shaderSet = map[string][]byte{
	"conv2d":     shaders.VAEConv2D,
	"add":        shaders.VAEAdd,
	"upsample2x": shaders.VAEUpsample2x,
	"to_rows":    shaders.VAENCHWToRows,
	"to_nchw":    shaders.VAERowsToNCHWAdd,
	"linear":     shaders.VAELinear,
	"transpose":  shaders.VAETranspose,
	"attention":  shaders.VAEAttentionDim1152,
	"chnorm":     shaders.QVAEChannelNorm,
	"dupup":      shaders.QVAEDupUp,
}

// NewGPUDecoder uploads a loaded decoder's weights, builds its pipelines and
// sizes the activation arena for a latentH x latentW latent — which is the
// largest the returned decoder will accept, since the arena is allocated
// once.
func NewGPUDecoder(dev *vk.Device, cpu *Decoder, latentH, latentW int) (*GPUDecoder, error) {
	return NewGPUDecoderOpts(dev, cpu, latentH, latentW, Options{})
}

// NewGPUDecoderOpts is NewGPUDecoder with the two screened kernels named,
// which is what TestGPUKernelScreen sweeps and what holds the new graph
// against the one it replaces.
func NewGPUDecoderOpts(dev *vk.Device, cpu *Decoder, latentH, latentW int, opt Options) (*GPUDecoder, error) {
	if got := cpu.Mid.Attn.QKV.InC; got != midDim {
		return nil, fmt.Errorf("qvae: mid block is %d channels, the attention kernel is built for %d", got, midDim)
	}
	g := &GPUDecoder{engine: newEngine(), cpu: cpu}
	if err := g.chooseKernels(opt); err != nil {
		return nil, err
	}
	g.flattenWeights()

	need, err := g.planSize(latentH, latentW)
	if err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(dev, g.kernelSet(shaderSet), need); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUDecoder) Destroy() { g.destroy() }

// flattenWeights copies every weight the graph reads into one arena, in the
// order the graph reads them. to_qkv stays one [3C, C] tensor: the three
// projections are its three row blocks, and the graph addresses them by
// offset.
func (g *GPUDecoder) flattenWeights() {
	w := g.weights
	conv := func(name string, c *Conv2D) {
		w.put(name+".weight", c.Weight)
		if c.Bias != nil {
			w.put(name+".bias", c.Bias)
		}
	}
	norm := func(name string, n *ChannelNorm) { w.put(name+".gamma", n.Gamma) }
	res := func(name string, r *ResBlock) {
		norm(name+".norm1", &r.Norm1)
		conv(name+".conv1", &r.Conv1)
		norm(name+".norm2", &r.Norm2)
		conv(name+".conv2", &r.Conv2)
		if r.Shortcut != nil {
			conv(name+".shortcut", r.Shortcut)
		}
	}

	d := g.cpu
	conv("post_quant", &d.PostQuant)
	conv("conv_in", &d.ConvIn)
	res("mid.r1", &d.Mid.Res1)
	norm("mid.attn.norm", &d.Mid.Attn.Norm)
	conv("mid.attn.qkv", &d.Mid.Attn.QKV)
	conv("mid.attn.proj", &d.Mid.Attn.Proj)
	res("mid.r2", &d.Mid.Res2)
	for i := range d.Ups {
		up := &d.Ups[i]
		for j := range up.Resnets {
			res(fmt.Sprintf("up.%d.r%d", i, j), &up.Resnets[j])
		}
		if up.Up != nil {
			conv(fmt.Sprintf("up.%d.upconv", i), &up.Up.Conv)
		}
	}
	norm("norm_out", &d.NormOut)
	conv("conv_out", &d.ConvOut)
}

// tensor is a shape plus its offset in the activation arena.
type tensor struct {
	off     uint32
	C, H, W int
}

func (t tensor) elems() int { return t.C * t.H * t.W }

// builder records a graph into a dispatch list. It is shared by both
// directions: the decoder and the encoder differ in what they record, not in
// how it is recorded.
type builder struct {
	e     *engine
	ar    *arena
	out   []vk.MultiDispatch
	kinds []string
	flops []float64

	pendingKind  string
	pendingFlops float64
	pendingIn    tensor

	// ins is each dispatch's input tensor where one convolution reads one
	// tensor, and the zero value otherwise. ConvInputRanges is its only
	// reader: what a matrix-core convolution would need to know about this
	// graph is whether the operand it narrows fits in fp16, and that is a
	// property of these tensors.
	ins []tensor

	// marks names the tensor each named stage leaves behind and the dispatch
	// that finishes it. A bump-allocated graph overwrites every intermediate
	// long before it ends, so the only way to look at one is to re-run the
	// prefix and stop — which is what RunTo does and what the stagewise gate
	// in gpu_test.go needs.
	marks map[string]tensor
	order []string
	stops []int

	err error
}

func (b *builder) mark(name string, t tensor) tensor {
	if b.marks != nil {
		b.marks[name] = t
		b.order = append(b.order, name)
		b.stops = append(b.stops, len(b.out))
	}
	return t
}

func (b *builder) label(kind string, flops float64) {
	b.pendingKind, b.pendingFlops = kind, flops
}

func groups(n, local int) uint32 { return uint32((n + local - 1) / local) }

func (b *builder) add(pipe string, gx uint32, pc pushConstants) {
	b.addXY(pipe, gx, 1, pc)
}

func (b *builder) addXY(pipe string, gx, gy uint32, pc pushConstants) {
	if b.err != nil {
		return
	}
	// planSize records the same graph with no pipelines built, purely to size
	// the arena, so a missing pipeline is an error only once there are any.
	p := b.e.pipes[pipe]
	if p == nil && len(b.e.pipes) > 0 {
		b.err = fmt.Errorf("qvae: no pipeline %q", pipe)
		return
	}
	b.out = append(b.out, vk.MultiDispatch{Pipeline: p, GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	kind := b.pendingKind
	if kind == "" {
		kind = pipe
	}
	b.kinds = append(b.kinds, kind)
	b.flops = append(b.flops, b.pendingFlops)
	b.ins = append(b.ins, b.pendingIn)
	b.pendingKind, b.pendingFlops, b.pendingIn = "", 0, tensor{}
}

func (b *builder) release(ts ...tensor) {
	for _, t := range ts {
		b.ar.release(t.off, t.elems())
	}
}

// wOff resolves a weight. A missing one is an error only once there are
// pipelines to dispatch — the same condition as addXY, and for the same
// reason: the sizing pass records the graph against neither.
func (b *builder) wOff(name string) uint32 {
	off, ok := b.e.weights.off[name]
	if !ok && b.err == nil && len(b.e.pipes) > 0 {
		b.err = fmt.Errorf("qvae: no weight %q", name)
	}
	return off
}

func (b *builder) bOff(name string) uint32 {
	if off, ok := b.e.weights.off[name]; ok {
		return off
	}
	return noBias
}

// conv appends a convolution producing a fresh tensor. Both shapes the
// decoder uses — 3x3 pad 1 and the 1x1 pad 0 shortcuts and projections —
// preserve H and W.
func (b *builder) conv(name string, c *Conv2D, x tensor) tensor {
	out := tensor{off: b.ar.alloc(c.OutC * x.H * x.W), C: c.OutC, H: x.H, W: x.W}
	b.label(fmt.Sprintf("conv%dx%d %d->%d @%dx%d", c.KH, c.KW, x.C, c.OutC, x.H, x.W),
		2*float64(c.OutC)*float64(x.H)*float64(x.W)*float64(x.C)*float64(c.KH)*float64(c.KW))
	b.pendingIn = x
	// The grid's y axis is output-channel blocks: the shader holds OC_BLOCK
	// accumulators per thread, which is where its arithmetic intensity is,
	// and which build supplies it is the screen's answer (kernels.go).
	pipe, ocBlock := b.e.convPipe(c.OutC)
	b.addXY(pipe, groups(out.H*out.W, 256), groups(c.OutC, ocBlock), pushConstants{
		InOff: x.off, OutOff: out.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		OC: uint32(c.OutC), KH: uint32(c.KH), KW: uint32(c.KW), Pad: uint32(c.Pad),
		WOff: b.wOff(name + ".weight"), BOff: b.bOff(name + ".bias"),
	})
	return out
}

// chnorm appends the channel norm, optionally with its SiLU fused. dst may
// be src: one thread owns one pixel's whole channel column, so writing it
// back over the read is safe and saves a tensor.
func (b *builder) chnorm(name string, silu bool, src, dst tensor) tensor {
	aux := uint32(0)
	if silu {
		aux = 1
	}
	b.label(fmt.Sprintf("chnorm %d @%dx%d", src.C, src.H, src.W), 0)
	b.add("chnorm", groups(src.H*src.W, 256), pushConstants{
		InOff: src.off, OutOff: dst.off,
		C: uint32(src.C), H: uint32(src.H), W: uint32(src.W),
		WOff: b.wOff(name + ".gamma"),
		Aux0: aux,
	})
	return dst
}

// addInto appends a += res.
func (b *builder) addInto(a, res tensor) tensor {
	b.add("add", groups(a.elems(), 256), pushConstants{
		InOff: a.off, OutOff: a.off, ResOff: res.off, Aux0: uint32(a.elems()),
	})
	return a
}

// resblock records one residual block. keepIn leaves the input allocated,
// which the up blocks need: their parameter-free shortcut reads the block's
// input after all three resnets have run.
//
// Every release happens after the dispatch that consumes the tensor is
// recorded and after the next tensor is allocated; releasing earlier would
// let the allocator hand the same block to a tensor still being read.
func (b *builder) resblock(name string, r *ResBlock, x tensor, keepIn bool) tensor {
	// norm1 writes into a fresh tensor rather than over x: x is the residual
	// and is read again at the end of the block.
	h := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.chnorm(name+".norm1", true, x, h)

	h1 := b.conv(name+".conv1", &r.Conv1, h)
	b.release(h)
	b.chnorm(name+".norm2", true, h1, h1)

	h2 := b.conv(name+".conv2", &r.Conv2, h1)
	b.release(h1)

	skip := x
	if r.Shortcut != nil {
		skip = b.conv(name+".shortcut", r.Shortcut, x)
	}
	out := b.addInto(h2, skip)
	if r.Shortcut != nil {
		b.release(skip)
	}
	if !keepIn {
		b.release(x)
	}
	return out
}

// projection records one of the mid block's four 1x1 projections over the
// pixel rows, on whichever kernel the screen chose. Both read the same push
// constants — a row-major [rows, inDim] operand, a PyTorch [outDim, inDim]
// weight and a bias — and differ only in the grid: one thread per output
// element, or one workgroup per 64x64 output tile.
func (b *builder) projection(src, dst tensor, h, w, inDim, outDim int, wOff, bOff uint32) {
	rows := h * w
	b.label(fmt.Sprintf("linear %d->%d x%d", inDim, outDim, rows),
		2*float64(rows)*float64(outDim)*float64(inDim))
	pc := pushConstants{
		InOff: src.off, OutOff: dst.off,
		C: uint32(inDim), H: uint32(h), W: uint32(w), OC: uint32(outDim),
		WOff: wOff, BOff: bOff,
	}
	if v := b.e.mid; v.tiled {
		b.addXY("linear", groups(rows, v.bm), groups(outDim, v.bn), pc)
		return
	}
	b.add("linear", groups(rows*outDim, 64), pc)
}

// attention records the mid block's single-head spatial attention. The three
// projections are the three row blocks of to_qkv's [3C, C] weight, read at
// offsets; the output projection is its own 1x1. Both run on vae_linear over
// the pixel rows, which is what a 1x1 convolution is once the tensor is
// transposed out of NCHW.
func (b *builder) attention(name string, a *Attention, x tensor) tensor {
	rows := x.H * x.W
	dim := x.C

	h := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.chnorm(name+".norm", false, x, h)

	seq := tensor{off: b.ar.alloc(rows * dim), C: dim, H: x.H, W: x.W}
	b.add("to_rows", groups(rows*dim, 256), pushConstants{
		InOff: h.off, OutOff: seq.off,
		C: uint32(dim), H: uint32(x.H), W: uint32(x.W),
	})
	b.release(h)

	qkvW, qkvB := b.wOff(name+".qkv.weight"), b.wOff(name+".qkv.bias")
	proj := func(which int) tensor {
		out := tensor{off: b.ar.alloc(rows * dim), C: dim, H: x.H, W: x.W}
		b.projection(seq, out, x.H, x.W, dim, dim,
			qkvW+uint32(which*dim*dim), qkvB+uint32(which*dim))
		return out
	}
	q, k, v := proj(0), proj(1), proj(2)
	b.release(seq)

	// Transpose K so the score loop reads consecutive keys at consecutive
	// addresses. The +64 pads the row stride off a multiple of the 4 KB
	// channel rotation (IDEAS §5.1b).
	kStride := rows + 64
	kT := tensor{off: b.ar.alloc(dim * kStride), C: dim, H: 1, W: kStride}
	b.label(fmt.Sprintf("transpose %dx%d", rows, dim), 0)
	b.add("transpose", groups(rows*dim, 256), pushConstants{
		InOff: k.off, OutOff: kT.off,
		C: uint32(dim), Aux0: uint32(rows), Aux1: uint32(kStride),
	})
	b.release(k)

	ctx := tensor{off: b.ar.alloc(rows * dim), C: dim, H: x.H, W: x.W}
	// Two passes of q.k plus the weighted sum of v, all over rows x rows.
	b.label(fmt.Sprintf("attention %d rows x %d", rows, dim),
		3*2*float64(rows)*float64(rows)*float64(dim))
	b.add("attention", uint32(rows), pushConstants{
		InOff: q.off, OutOff: ctx.off, ResOff: kT.off,
		C:      uint32(dim),
		Groups: uint32(kStride),
		Aux0:   uint32(rows), Aux1: math.Float32bits(float32(1 / math.Sqrt(float64(dim)))), Aux2: v.off,
	})
	b.release(q, kT, v)

	outRows := tensor{off: b.ar.alloc(rows * dim), C: dim, H: x.H, W: x.W}
	b.projection(ctx, outRows, x.H, x.W, dim, dim,
		b.wOff(name+".proj.weight"), b.bOff(name+".proj.bias"))
	b.release(ctx)

	res := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	// The bias is already in the projection above, so the fused add has none
	// of its own — and there is no push-constant value that means "unset", so
	// it is said explicitly.
	b.add("to_nchw", groups(x.elems(), 256), pushConstants{
		InOff: outRows.off, OutOff: res.off, ResOff: x.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		Aux0: noBias,
	})
	b.release(outRows, x)
	return res
}

// upsample is the learned half of an up block's resampling: nearest 2x, then
// a 3x3 convolution.
func (b *builder) upsample(name string, c *Conv2D, x tensor) tensor {
	up := tensor{off: b.ar.alloc(x.C * x.H * 2 * x.W * 2), C: x.C, H: x.H * 2, W: x.W * 2}
	b.add("upsample2x", groups(up.elems(), 256), pushConstants{
		InOff: x.off, OutOff: up.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
	})
	b.release(x)
	out := b.conv(name, c, up)
	b.release(up)
	return out
}

// dupup is the parameter-free shortcut around an up block.
func (b *builder) dupup(u *DupUp, x tensor) tensor {
	out := tensor{off: b.ar.alloc(u.Out * x.H * 2 * x.W * 2), C: u.Out, H: x.H * 2, W: x.W * 2}
	b.label(fmt.Sprintf("dupup %d->%d @%dx%d", u.In, u.Out, out.H, out.W), 0)
	b.add("dupup", groups(out.elems(), 256), pushConstants{
		InOff: x.off, OutOff: out.off,
		C: uint32(u.In), H: uint32(x.H), W: uint32(x.W), OC: uint32(u.Out),
		Aux0: uint32(u.FactorT),
	})
	return out
}

// build records the whole decode and returns the output tensor, which is
// still unclamped: the clamp to [-1, 1] rides on the readback.
func (b *builder) build(d *Decoder, latentH, latentW int) tensor {
	z := tensor{off: b.ar.alloc(d.PostQuant.InC * latentH * latentW), C: d.PostQuant.InC, H: latentH, W: latentW}
	x := b.conv("post_quant", &d.PostQuant, z)
	b.release(z)
	h := b.conv("conv_in", &d.ConvIn, x)
	b.release(x)
	b.mark("conv_in", h)

	h = b.resblock("mid.r1", &d.Mid.Res1, h, false)
	h = b.attention("mid.attn", &d.Mid.Attn, h)
	h = b.resblock("mid.r2", &d.Mid.Res2, h, false)
	b.mark("mid_block", h)

	for i := range d.Ups {
		up := &d.Ups[i]
		in := h
		keep := up.Shortcut != nil
		for j := range up.Resnets {
			// Only the first resnet reads the block's input, so only it has to
			// be told to keep it alive.
			h = b.resblock(fmt.Sprintf("up.%d.r%d", i, j), &up.Resnets[j], h, keep && j == 0)
		}
		if up.Up != nil {
			h = b.upsample(fmt.Sprintf("up.%d.upconv", i), &up.Up.Conv, h)
		}
		if up.Shortcut != nil {
			short := b.dupup(up.Shortcut, in)
			h = b.addInto(h, short)
			b.release(short, in)
		}
		b.mark(fmt.Sprintf("up_blocks.%d", i), h)
	}

	h = b.chnorm("norm_out", true, h, h)
	b.mark("nonlinearity", h)
	out := b.conv("conv_out", &d.ConvOut, h)
	b.release(h)
	return b.mark("conv_out", out)
}

// plan records the graph. A nil-pipelined decoder records it purely to size
// the arena or count the dispatches.
func (g *GPUDecoder) plan(ar *arena, latentH, latentW int, marks bool) (*builder, tensor, error) {
	b := &builder{e: &g.engine, ar: ar}
	if marks {
		b.marks = map[string]tensor{}
	}
	out := b.build(g.cpu, latentH, latentW)
	return b, out, b.err
}

// MaxStorageBufferBytes is this device's `maxStorageBufferRange`, 4 GiB - 4.
// The activation arena is one buffer, so it is also the ceiling on what a
// decode can be planned for — see ArenaBytes.
const MaxStorageBufferBytes = 4<<30 - 4

// ArenaBytes is how much activation memory a decode of this latent would
// need, without a device and without loading a weight. It is what sizes the
// ceiling Q6's geometry has to respect: the arena is a single storage
// buffer, and this device caps one of those at MaxStorageBufferBytes.
func ArenaBytes(cpu *Decoder, latentH, latentW int) (int, error) {
	g := &GPUDecoder{engine: newEngine(), cpu: cpu}
	n, err := g.planSize(latentH, latentW)
	if err != nil {
		return 0, err
	}
	return int(n) * 4, nil
}

// planSize finds how large the activation arena has to be, by recording the
// graph with no pipelines bound.
func (g *GPUDecoder) planSize(latentH, latentW int) (uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, latentH, latentW, false)
	g.pipes = saved
	if err != nil {
		return 0, err
	}
	return b.ar.high, nil
}

// Dispatches is the number of GPU dispatches one decode records.
func (g *GPUDecoder) Dispatches(latentH, latentW int) (int, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, latentH, latentW, false)
	g.pipes = saved
	if err != nil {
		return 0, err
	}
	return len(b.out), nil
}

// record builds the graph against the real arena and uploads the latent.
func (g *GPUDecoder) record(z *Tensor, marks bool) (*builder, tensor, error) {
	if z.N != 1 {
		return nil, tensor{}, fmt.Errorf("qvae: the GPU decoder takes one latent at a time, got N=%d", z.N)
	}
	if z.C != g.cpu.PostQuant.InC {
		return nil, tensor{}, fmt.Errorf("qvae: latent has %d channels, post_quant_conv takes %d", z.C, g.cpu.PostQuant.InC)
	}
	g.arena.reset()
	b, out, err := g.plan(&g.arena, z.H, z.W, marks)
	if err != nil {
		return nil, tensor{}, err
	}
	if int(b.ar.high)*4 > g.abuf.Size() {
		return nil, tensor{}, fmt.Errorf("qvae: graph needs %d MB of activations, arena is %d MB",
			(int(b.ar.high)*4)>>20, g.abuf.Size()>>20)
	}
	// The latent is the arena's first allocation, so it goes in at offset 0.
	g.abuf.WriteFloat32(z.Data)
	return b, out, nil
}

// Decode decodes one denormalized latent, clamped to [-1, 1] like the
// reference's. The whole graph is recorded once and submitted in batches.
func (g *GPUDecoder) Decode(ctx context.Context, z *Tensor) (*Tensor, error) {
	b, out, err := g.record(z, false)
	if err != nil {
		return nil, err
	}
	if err := g.runContext(ctx, b.out); err != nil {
		return nil, err
	}
	img := g.read(out)
	// The clamp is the reference's, and it is free here: the readback is
	// already walking every element.
	for i, v := range img.Data {
		if v > 1 {
			img.Data[i] = 1
		} else if v < -1 {
			img.Data[i] = -1
		}
	}
	return img, nil
}

// Stages names the graph's marked stages, in order. They are the same names
// the CPU decoder's Tap emits, so a stagewise comparison needs no mapping.
func (g *GPUDecoder) Stages(latentH, latentW int) ([]string, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, latentH, latentW, true)
	g.pipes = saved
	if err != nil {
		return nil, err
	}
	return b.order, nil
}

// RunTo re-runs the graph from the start and stops after the named stage,
// returning what it produced. A bump-allocated graph overwrites every
// intermediate long before it ends, so re-running the prefix is the only way
// to see one — the same facility, for the same reason, as zimage/vae's
// GPUEncoder.RunTo.
func (g *GPUDecoder) RunTo(z *Tensor, stage string) (*Tensor, error) {
	b, _, err := g.record(z, true)
	if err != nil {
		return nil, err
	}
	t, ok := b.marks[stage]
	if !ok {
		return nil, fmt.Errorf("qvae: no stage %q (have %v)", stage, b.order)
	}
	stop := 0
	for i, name := range b.order {
		if name == stage {
			stop = b.stops[i]
			break
		}
	}
	if err := g.run(b.out[:stop]); err != nil {
		return nil, err
	}
	return g.read(t), nil
}

// ConvInputRange is one convolution's input magnitude, as measured by
// running the graph up to it.
type ConvInputRange struct {
	Index  int
	Kind   string
	AbsMax float64
}

// ConvInputRanges reports, for every convolution in the graph, the largest
// magnitude in the tensor it reads.
//
// It answers the one question standing between this decode and z-image's
// matrix-core convolution, which took that operator from 3.18 s to 258 ms
// and is 69% of this one. That kernel narrows its B operand — the
// activation — to fp16 and accumulates in fp32, so what decides whether it
// can be pointed at a graph is whether the activations clear 65504. In
// z-image's decoder they peak at 497, 132x inside; Q0 measured this model's
// intermediates at absmax ~1e5–3e5, which is why the answer is measured
// here rather than inherited.
//
// It reads a tensor back per convolution over a write-combined buffer, so
// it is slow by construction and belongs at the reference latent size.
func (g *GPUDecoder) ConvInputRanges(z *Tensor) ([]ConvInputRange, error) {
	b, _, err := g.record(z, false)
	if err != nil {
		return nil, err
	}
	var out []ConvInputRange
	for i := range b.out {
		if in := b.ins[i]; in.C != 0 {
			var absMax float64
			for _, v := range g.abuf.ReadFloat32At(int(in.off), in.elems()) {
				if a := math.Abs(float64(v)); a > absMax {
					absMax = a
				}
			}
			out = append(out, ConvInputRange{Index: i, Kind: b.kinds[i], AbsMax: absMax})
		}
		if _, err := vk.DispatchMultiTimed(b.out[i:i+1], 1, 1, true); err != nil {
			return out, fmt.Errorf("qvae: dispatch %d (%s): %w", i, b.kinds[i], err)
		}
	}
	return out, nil
}

// Stage is one dispatch's identity and cost, as returned by Profile.
type Stage struct {
	Index int
	Kind  string
	GPU   time.Duration
	// Flops is the dispatch's multiply-accumulate count, x2.
	Flops float64
}

// Profile runs the graph one dispatch at a time, timing each on the GPU.
// Submitting separately costs a fence wait per dispatch, so the total is
// above what Decode measures — what it is for is the distribution.
func (g *GPUDecoder) Profile(z *Tensor) ([]Stage, error) {
	b, _, err := g.record(z, false)
	if err != nil {
		return nil, err
	}
	stages := make([]Stage, 0, len(b.out))
	for i, d := range b.out {
		dur, err := d.Pipeline.DispatchTimed(d.GroupsX, d.GroupsY, 1, 1, d.PushConstants)
		if err != nil {
			return stages, fmt.Errorf("qvae: dispatch %d (%s): %w", i, b.kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: b.kinds[i], GPU: dur, Flops: b.flops[i]})
	}
	return stages, nil
}
