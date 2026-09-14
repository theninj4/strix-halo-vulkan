package vae

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// pushConstants mirrors the PC block in shaders/vae_common.glsl. Every VAE
// pipeline declares the same block, because vk.DispatchMultiTimed records a
// whole graph into one command buffer and requires one push-constant size
// across it.
type pushConstants struct {
	InOff, OutOff    uint32
	C, H, W          uint32
	OC, KH, KW, Pad  uint32
	WOff, BOff       uint32
	Groups           uint32
	ResOff           uint32
	Aux0, Aux1, Aux2 uint32
}

const noBias = 0xffffffff

// dispatchesPerSubmit caps how much work goes into one command buffer. See
// Apply for why it is not simply "all of it", and why 8 is not safe either.
const dispatchesPerSubmit = 4

func (p pushConstants) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*pushConstants)(unsafe.Pointer(&out[0])) = p
	return out
}

// arena hands out element offsets inside one activation buffer. The decoder
// never frees: a decode is a straight line of at most a few dozen live
// tensors, and a bump allocator that a caller resets between runs is both
// simpler and more predictable than reference counting buffers.
type arena struct {
	next uint32
	high uint32
	// free holds released blocks, largest first. A decode is a straight line
	// of a few dozen tensors, so a best-fit search over a handful of free
	// blocks costs nothing and recovers almost everything a real liveness
	// analysis would.
	//
	// Exact-size bucketing is not enough here, and the reason is the shape
	// of this graph: an up block's input is twice the channel count of what
	// its resnets produce, so the arena fills with 1.07 GB blocks that can
	// never serve the 537 MB requests that follow. That alone was the
	// difference between 5.28 GB and the 4.29 GB a single Vulkan storage
	// buffer can address on this device.
	free []freeBlock
}

// freeBlock is a released span of the arena.
type freeBlock struct {
	off  uint32
	size uint32
}

func roundUp(elems int) uint32 { return uint32((elems + 63) &^ 63) }

func (a *arena) alloc(elems int) uint32 {
	// Keep every tensor 256-byte aligned. IDEAS §5.1b's rule is about row
	// strides rather than base addresses, but starting tensors on a channel
	// boundary costs nothing and keeps a later fp16 port from inheriting an
	// accidental alias.
	n := roundUp(elems)
	// Best fit: the smallest free block that still holds n. Taking the
	// largest instead strands the big blocks the next up-block needs.
	best := -1
	for i, b := range a.free {
		if b.size >= n && (best < 0 || b.size < a.free[best].size) {
			best = i
		}
	}
	if best >= 0 {
		blk := a.free[best]
		a.free = append(a.free[:best], a.free[best+1:]...)
		// Return the remainder so a large block is not consumed whole by a
		// small request.
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

// release returns a tensor's block to the free list, coalescing it with any
// neighbour so that repeated split/rejoin cycles do not fragment the arena.
// Calling it on a tensor that is still read produces silent corruption, so
// it is only ever called from the builder, which knows the graph's shape.
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
}

func (a *arena) reset() {
	a.next = 0
	a.free = nil
}

// gpuWeights is the flat weight arena plus the offset of every tensor the
// decoder needs, resolved once at load time.
type gpuWeights struct {
	data []float32
	off  map[string]uint32
}

func (g *gpuWeights) put(name string, v []float32) uint32 {
	off := uint32(len(g.data))
	g.data = append(g.data, v...)
	g.off[name] = off
	return off
}

// GPUDecoder runs the decoder graph on a Vulkan device.
type GPUDecoder struct {
	dev  *vk.Device
	cpu  *Decoder
	wbuf *vk.Buffer
	abuf *vk.Buffer

	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	weights *gpuWeights
	arena   arena

	// dispatches accumulates the recorded graph so a decode is one submit.
	dispatches []vk.MultiDispatch
}

// shaderSet is every pipeline the decoder graph uses.
var shaderSet = map[string][]byte{
	"conv2d":     shaders.VAEConv2D,
	"groupnorm":  shaders.VAEGroupNorm,
	"silu":       shaders.VAESiLU,
	"add":        shaders.VAEAdd,
	"upsample2x": shaders.VAEUpsample2x,
	"to_rows":    shaders.VAENCHWToRows,
	"to_nchw":    shaders.VAERowsToNCHWAdd,
	"linear":     shaders.VAELinear,
	"attention":  shaders.VAEAttention,
	"transpose":  shaders.VAETranspose,
}

// NewGPUDecoder uploads a loaded decoder's weights and builds its pipelines.
// latentH/latentW size the activation arena, which is allocated once for the
// largest tensor the graph will hold.
func NewGPUDecoder(dev *vk.Device, cpu *Decoder, latentH, latentW int) (*GPUDecoder, error) {
	g := &GPUDecoder{
		dev:     dev,
		cpu:     cpu,
		pipes:   make(map[string]*vk.ComputePipeline),
		weights: &gpuWeights{off: make(map[string]uint32)},
	}

	g.flattenWeights()

	wbytes := len(g.weights.data) * 4
	var err error
	if g.wbuf, err = dev.NewBuffer(wbytes); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: weight buffer (%d MB): %w", wbytes>>20, err)
	}
	g.wbuf.WriteFloat32(g.weights.data)

	// Measure the arena by recording the graph once with a throwaway plan.
	need, err := g.planSize(latentH, latentW)
	if err != nil {
		g.Destroy()
		return nil, err
	}
	if g.abuf, err = dev.NewBuffer(int(need) * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: activation buffer (%d MB): %w", (int(need)*4)>>20, err)
	}

	for name, spirv := range shaderSet {
		mod, err := dev.NewShaderModule(spirv)
		if err != nil {
			g.Destroy()
			return nil, fmt.Errorf("vae: shader %s: %w", name, err)
		}
		g.mods = append(g.mods, mod)
		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{g.wbuf, g.abuf},
			PushConstantSize: uint32(unsafe.Sizeof(pushConstants{})),
		})
		if err != nil {
			g.Destroy()
			return nil, fmt.Errorf("vae: pipeline %s: %w", name, err)
		}
		g.pipes[name] = pipe
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUDecoder) Destroy() {
	for _, p := range g.pipes {
		p.Destroy()
	}
	for _, m := range g.mods {
		m.Destroy()
	}
	if g.abuf != nil {
		g.abuf.Destroy()
	}
	if g.wbuf != nil {
		g.wbuf.Destroy()
	}
}

// flattenWeights copies every weight the graph reads into one arena, in the
// order the graph reads them.
func (g *GPUDecoder) flattenWeights() {
	w := g.weights
	conv := func(name string, c *Conv2D) {
		w.put(name+".weight", c.Weight)
		if c.Bias != nil {
			w.put(name+".bias", c.Bias)
		}
	}
	norm := func(name string, n *GroupNorm) {
		w.put(name+".weight", n.Weight)
		w.put(name+".bias", n.Bias)
	}
	lin := func(name string, l *Linear) {
		w.put(name+".weight", l.Weight)
		if l.Bias != nil {
			w.put(name+".bias", l.Bias)
		}
	}
	resnet := func(name string, r *ResnetBlock) {
		norm(name+".norm1", r.Norm1)
		conv(name+".conv1", r.Conv1)
		norm(name+".norm2", r.Norm2)
		conv(name+".conv2", r.Conv2)
		if r.Shortcut != nil {
			conv(name+".shortcut", r.Shortcut)
		}
	}

	conv("conv_in", g.cpu.ConvIn)
	resnet("mid.r1", g.cpu.Mid.Resnet1)
	norm("mid.attn.norm", g.cpu.Mid.Attn.GroupNorm)
	lin("mid.attn.q", g.cpu.Mid.Attn.Q)
	lin("mid.attn.k", g.cpu.Mid.Attn.K)
	lin("mid.attn.v", g.cpu.Mid.Attn.V)
	lin("mid.attn.out", g.cpu.Mid.Attn.Out)
	resnet("mid.r2", g.cpu.Mid.Resnet2)
	for i, up := range g.cpu.UpBlocks {
		for j, r := range up.Resnets {
			resnet(fmt.Sprintf("up.%d.r%d", i, j), r)
		}
		if up.Upsampler != nil {
			conv(fmt.Sprintf("up.%d.upconv", i), up.Upsampler)
		}
	}
	norm("conv_norm_out", g.cpu.ConvNormOut)
	conv("conv_out", g.cpu.ConvOut)
}

// tensor is a shape plus its offset in the activation arena.
type tensor struct {
	off     uint32
	C, H, W int
}

func (t tensor) elems() int { return t.C * t.H * t.W }

// builder records the decode graph into a dispatch list.
type builder struct {
	g     *GPUDecoder
	ar    *arena
	out   []vk.MultiDispatch
	kinds []string
	flops []float64

	pendingKind  string
	pendingFlops float64

	err error
}

// label names the next dispatch for the profiler and records its arithmetic.
func (b *builder) label(kind string, flops float64) {
	b.pendingKind, b.pendingFlops = kind, flops
}

func groups(n, local int) uint32 { return uint32((n + local - 1) / local) }

func (b *builder) add(pipe string, gx uint32, pc pushConstants) {
	b.addXY(pipe, gx, 1, pc)
}

// addXY records a dispatch with a two-dimensional grid. The convolution uses
// it: x covers spatial tiles and y covers output channels, so one workgroup
// owns one channel's filter and can stage it through shared memory.
func (b *builder) addXY(pipe string, gx, gy uint32, pc pushConstants) {
	if b.err != nil {
		return
	}
	// planSize records the same graph with no pipelines built, purely to
	// size the arena, so a missing pipeline is an error only once there are
	// any at all.
	p := b.g.pipes[pipe]
	if p == nil && len(b.g.pipes) > 0 {
		b.err = fmt.Errorf("vae: no pipeline %q", pipe)
		return
	}
	b.out = append(b.out, vk.MultiDispatch{Pipeline: p, GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
	kind := b.pendingKind
	if kind == "" {
		kind = pipe
	}
	b.kinds = append(b.kinds, kind)
	b.flops = append(b.flops, b.pendingFlops)
	b.pendingKind, b.pendingFlops = "", 0
}

// release marks tensors dead so their memory can back later ones.
func (b *builder) release(ts ...tensor) {
	for _, t := range ts {
		b.ar.release(t.off, t.elems())
	}
}

func (b *builder) wOff(name string) uint32 {
	off, ok := b.g.weights.off[name]
	if !ok && b.err == nil {
		b.err = fmt.Errorf("vae: no weight %q", name)
	}
	return off
}

func (b *builder) bOff(name string) uint32 {
	if off, ok := b.g.weights.off[name]; ok {
		return off
	}
	return noBias
}

// conv appends a convolution producing a fresh tensor.
func (b *builder) conv(name string, c *Conv2D, x tensor) tensor {
	out := tensor{off: b.ar.alloc(c.OutC * x.H * x.W), C: c.OutC, H: x.H, W: x.W}
	if c.Pad == 0 && c.KH == 1 {
		// 1x1 with no padding also preserves H and W.
		out.H, out.W = x.H, x.W
	}
	b.label(fmt.Sprintf("conv%dx%d %d->%d @%dx%d", c.KH, c.KW, x.C, c.OutC, x.H, x.W),
		2*float64(c.OutC)*float64(out.H)*float64(out.W)*float64(x.C)*float64(c.KH)*float64(c.KW))
	// The grid's y axis is output-channel *blocks*: the shader's OC_BLOCK
	// accumulators per thread are what give the kernel its arithmetic
	// intensity, so the channel count is divided by that here.
	const convOCBlock = 8
	b.addXY("conv2d", groups(out.H*out.W, 256), groups(c.OutC, convOCBlock), pushConstants{
		InOff: x.off, OutOff: out.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		OC: uint32(c.OutC), KH: uint32(c.KH), KW: uint32(c.KW), Pad: uint32(c.Pad),
		WOff: b.wOff(name + ".weight"), BOff: b.bOff(name + ".bias"),
	})
	return out
}

// norm appends a group norm in place.
func (b *builder) norm(name string, n *GroupNorm, x tensor) tensor {
	b.add("groupnorm", uint32(n.Groups), pushConstants{
		InOff: x.off, OutOff: x.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		WOff: b.wOff(name + ".weight"), BOff: b.wOff(name + ".bias"),
		Groups: uint32(n.Groups),
		Aux0:   math.Float32bits(float32(n.Eps)),
	})
	return x
}

// silu appends an in-place SiLU.
func (b *builder) silu(x tensor) tensor {
	b.add("silu", groups(x.elems(), 256), pushConstants{
		InOff: x.off, OutOff: x.off, Aux0: uint32(x.elems()),
	})
	return x
}

// addInto appends out = a + res, writing into a.
func (b *builder) addInto(a, res tensor) tensor {
	b.add("add", groups(a.elems(), 256), pushConstants{
		InOff: a.off, OutOff: a.off, ResOff: res.off, Aux0: uint32(a.elems()),
	})
	return a
}

func (b *builder) resnet(name string, r *ResnetBlock, x tensor) tensor {
	// norm1 writes into a fresh tensor rather than over x, because x is the
	// residual and is read again at the end of the block.
	//
	// Every release below happens *after* the dispatch that consumes the
	// tensor has been recorded, and after the next tensor has been
	// allocated. Releasing earlier would let the allocator hand the same
	// block to a tensor that is still being read from it.
	h := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.add("groupnorm", uint32(r.Norm1.Groups), pushConstants{
		InOff: x.off, OutOff: h.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		WOff: b.wOff(name + ".norm1.weight"), BOff: b.wOff(name + ".norm1.bias"),
		Groups: uint32(r.Norm1.Groups), Aux0: math.Float32bits(float32(r.Norm1.Eps)),
	})
	h = b.silu(h)

	h1 := b.conv(name+".conv1", r.Conv1, h)
	b.release(h)
	h1 = b.norm(name+".norm2", r.Norm2, h1)
	h1 = b.silu(h1)

	h2 := b.conv(name+".conv2", r.Conv2, h1)
	b.release(h1)

	skip := x
	if r.Shortcut != nil {
		skip = b.conv(name+".shortcut", r.Shortcut, x)
	}
	out := b.addInto(h2, skip)
	if r.Shortcut != nil {
		b.release(skip)
	}
	b.release(x)
	return out
}

func (b *builder) attention(name string, a *Attention, x tensor) tensor {
	rows := x.H * x.W
	dim := a.Q.Out

	h := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.add("groupnorm", uint32(a.GroupNorm.Groups), pushConstants{
		InOff: x.off, OutOff: h.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		WOff: b.wOff(name + ".norm.weight"), BOff: b.wOff(name + ".norm.bias"),
		Groups: uint32(a.GroupNorm.Groups), Aux0: math.Float32bits(float32(a.GroupNorm.Eps)),
	})

	seq := tensor{off: b.ar.alloc(rows * x.C), C: x.C, H: x.H, W: x.W}
	b.add("to_rows", groups(rows*x.C, 256), pushConstants{
		InOff: h.off, OutOff: seq.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
	})

	proj := func(pname string, l *Linear) tensor {
		out := tensor{off: b.ar.alloc(rows * l.Out), C: l.Out, H: x.H, W: x.W}
		b.label(fmt.Sprintf("linear %d->%d x%d", l.In, l.Out, rows),
			2*float64(rows)*float64(l.Out)*float64(l.In))
		b.add("linear", groups(rows*l.Out, 64), pushConstants{
			InOff: seq.off, OutOff: out.off,
			C: uint32(l.In), H: uint32(x.H), W: uint32(x.W), OC: uint32(l.Out),
			WOff: b.wOff(pname + ".weight"), BOff: b.bOff(pname + ".bias"),
		})
		return out
	}
	q := proj(name+".q", a.Q)
	k := proj(name+".k", a.K)
	v := proj(name+".v", a.V)

	// Transpose K so the score loop reads consecutive keys at consecutive
	// addresses (IDEAS §5.1b). The +64 pads the destination row stride off a
	// multiple of the 4 KB channel rotation, which is what puts the gcd
	// inside §5.1b's [128, 256] byte window.
	kStride := rows + 64
	kT := tensor{off: b.ar.alloc(dim * kStride), C: dim, H: 1, W: kStride}
	b.label(fmt.Sprintf("transpose %dx%d", rows, dim), 0)
	b.add("transpose", groups(rows*dim, 256), pushConstants{
		InOff: k.off, OutOff: kT.off,
		C: uint32(dim), Aux0: uint32(rows), Aux1: uint32(kStride),
	})

	ctx := tensor{off: b.ar.alloc(rows * dim), C: dim, H: x.H, W: x.W}
	// Two passes of q.k plus the weighted sum of v, all over rows x rows.
	b.label(fmt.Sprintf("attention %d rows x %d", rows, dim),
		3*2*float64(rows)*float64(rows)*float64(dim))
	b.add("attention", uint32(rows), pushConstants{
		InOff: q.off, OutOff: ctx.off, ResOff: kT.off,
		C:      uint32(dim),
		Groups: uint32(kStride),
		Aux0:   uint32(rows), Aux1: math.Float32bits(float32(a.Scale)), Aux2: v.off,
	})

	outRows := tensor{off: b.ar.alloc(rows * a.Out.Out), C: a.Out.Out, H: x.H, W: x.W}
	b.label(fmt.Sprintf("linear %d->%d x%d", a.Out.In, a.Out.Out, rows),
		2*float64(rows)*float64(a.Out.Out)*float64(a.Out.In))
	b.add("linear", groups(rows*a.Out.Out, 64), pushConstants{
		InOff: ctx.off, OutOff: outRows.off,
		C: uint32(a.Out.In), H: uint32(x.H), W: uint32(x.W), OC: uint32(a.Out.Out),
		WOff: b.wOff(name + ".out.weight"), BOff: b.bOff(name + ".out.bias"),
	})

	res := tensor{off: b.ar.alloc(x.elems()), C: x.C, H: x.H, W: x.W}
	b.add("to_nchw", groups(x.elems(), 256), pushConstants{
		InOff: outRows.off, OutOff: res.off, ResOff: x.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
	})
	b.release(h, seq, q, k, kT, v, ctx, outRows, x)
	return res
}

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

// build records the whole decode and returns the output tensor.
func (b *builder) build(latentH, latentW int) tensor {
	d := b.g.cpu
	x := tensor{off: b.ar.alloc(d.ConvIn.InC * latentH * latentW), C: d.ConvIn.InC, H: latentH, W: latentW}
	h := b.conv("conv_in", d.ConvIn, x)
	h = b.resnet("mid.r1", d.Mid.Resnet1, h)
	h = b.attention("mid.attn", d.Mid.Attn, h)
	h = b.resnet("mid.r2", d.Mid.Resnet2, h)
	for i, up := range d.UpBlocks {
		for j, r := range up.Resnets {
			h = b.resnet(fmt.Sprintf("up.%d.r%d", i, j), r, h)
		}
		if up.Upsampler != nil {
			h = b.upsample(fmt.Sprintf("up.%d.upconv", i), up.Upsampler, h)
		}
	}
	h = b.norm("conv_norm_out", d.ConvNormOut, h)
	h = b.silu(h)
	out := b.conv("conv_out", d.ConvOut, h)
	b.release(h)
	return out
}

// planSize records the graph with no pipelines bound, purely to find how
// large the activation arena has to be.
func (g *GPUDecoder) planSize(latentH, latentW int) (uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{g: g, ar: &arena{}}
	b.build(latentH, latentW)
	g.pipes = saved
	if b.err != nil {
		return 0, b.err
	}
	return b.ar.high, nil
}

// Apply decodes a latent on the GPU. The whole graph -- 100-odd dispatches
// -- is recorded into one command buffer and submitted once, with a barrier
// between every pair, because each stage reads what the previous wrote.
func (g *GPUDecoder) Apply(latent *Tensor) (*Tensor, error) {
	if latent.N != 1 {
		return nil, fmt.Errorf("vae: GPU decoder handles one image at a time, got N=%d", latent.N)
	}
	if latent.C != g.cpu.ConvIn.InC {
		return nil, fmt.Errorf("vae: latent has %d channels, conv_in takes %d", latent.C, g.cpu.ConvIn.InC)
	}

	g.arena.reset()
	b := &builder{g: g, ar: &g.arena}
	out := b.build(latent.H, latent.W)
	if b.err != nil {
		return nil, b.err
	}
	if int(b.ar.high)*4 > g.abuf.Size() {
		return nil, fmt.Errorf("vae: graph needs %d MB of activations, arena is %d MB",
			(int(b.ar.high)*4)>>20, g.abuf.Size()>>20)
	}

	// The input tensor is the arena's first allocation, so the latent goes
	// in at offset 0.
	g.abuf.WriteFloat32(latent.Data)

	// Submit in batches rather than as one command buffer. The whole graph
	// in a single submit is 5.5 s of GPU work at a 1024x1024 image, which
	// trips the driver's reset watchdog and comes back as VK_ERROR_DEVICE_LOST
	// -- the same 119 dispatches submitted separately all complete. Batching
	// keeps each submit short while still amortising the per-submit fence
	// wait over a useful number of dispatches.
	//
	// The batch is 4 and not 8 because the graph is not flat: at a 1024x1024
	// image the mid-block attention is **1.5 s in one dispatch** (16384 rows
	// of 512, 27% of the decode), so a batch of eight around it is over the
	// watchdog on its own. That size only shows up at the full image -- 8 is
	// fine at 512x512, which is the largest thing stage 2 measured -- so this
	// is a limit the pipeline found rather than the decoder. 4 and 2 are the
	// same wall clock to within 1%, so the halving costs nothing.
	//
	// Ordering and correctness are unaffected: DispatchMultiTimed blocks
	// until its batch completes, so a batch boundary is a stronger barrier
	// than the one already recorded between every pair of dispatches.
	for i := 0; i < len(b.out); i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, len(b.out))
		if _, err := vk.DispatchMultiTimed(b.out[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("vae: dispatch %d-%d: %w", i, j-1, err)
		}
	}

	res := NewTensor(1, out.C, out.H, out.W)
	copy(res.Data, g.abuf.ReadFloat32At(int(out.off), out.elems()))
	return res, nil
}

// Dispatches is the number of GPU dispatches one decode records, which is
// the figure IDEAS §4.1's 300 ns launch cost applies to.
func (g *GPUDecoder) Dispatches(latentH, latentW int) (int, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{g: g, ar: &arena{}}
	b.build(latentH, latentW)
	g.pipes = saved
	if b.err != nil {
		return 0, b.err
	}
	return len(b.out), nil
}

// ActivationBytes is the size of the activation arena.
func (g *GPUDecoder) ActivationBytes() int { return g.abuf.Size() }

// Stage is one dispatch's identity and cost, as returned by Profile.
type Stage struct {
	Index int
	Kind  string
	GPU   time.Duration
	// Flops is the multiply-accumulate count the dispatch performs, x2, so
	// it can be put against the ceilings in research/roofline terms.
	Flops float64
}

// Profile runs the graph one dispatch at a time, timing each on the GPU.
// Submitting separately costs a fence wait per dispatch, so the total here
// is above what Apply measures -- what it is for is the *distribution*, i.e.
// which stage to fix.
func (g *GPUDecoder) Profile(latent *Tensor) ([]Stage, error) {
	g.arena.reset()
	b := &builder{g: g, ar: &g.arena}
	out := b.build(latent.H, latent.W)
	if b.err != nil {
		return nil, b.err
	}
	_ = out
	g.abuf.WriteFloat32(latent.Data)

	stages := make([]Stage, 0, len(b.out))
	for i, d := range b.out {
		dur, err := d.Pipeline.DispatchTimed(d.GroupsX, d.GroupsY, 1, 1, d.PushConstants)
		if err != nil {
			return stages, fmt.Errorf("vae: dispatch %d (%s): %w", i, b.kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: b.kinds[i], GPU: dur, Flops: b.flops[i]})
	}
	return stages, nil
}
