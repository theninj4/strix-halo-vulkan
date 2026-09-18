package vae

import (
	"fmt"
	"os"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// taef1 on the device -- IMAGE.md I2.
//
// There is almost nothing here, and that is the point: the graph is 48
// convolutions, three nearest upsamples, 40 ReLUs and 10 adds, and every one
// of those but the ReLU is a kernel stage 8 already wrote and tuned for the
// full decoder. So this file is a weight staging, a graph, and a submit loop
// over the same `engine`.
//
// **The activation arena is host-cached and the full decoder's is not, and
// the measurement says it does not matter here.** A preview's whole output
// crosses back to the host -- 12.6 MB at 1024x1024, eight times per image
// rather than once -- and LLM.md L6b measured a mapped write-combined buffer
// reading at 0.18 GB/s against a HOST_CACHED one's ~25, which would make that
// readback 70 ms against an 87 ms decode. It is not, and stage 9's note
// (research/stage-9-head-and-tail.md) explains why: the 0.18 GB/s heap is the
// *device-local* one, whose budget on this part is about 8 GB, and a program
// past it gets ordinary cached system memory whichever type it asks for. A
// pipeline holding 25 GB is well past it.
//
// ZIMAGE_TINY_UNCACHED=1 is the control, and it measures the null result:
// 87.7/88.2/87.1 ms against the cached arena's 90.9/87.7/88.3 in a 25 GB
// process, and 87.7 against 86.6 in cmd/vaebench. The cached type is kept
// anyway because it is free -- the device reads every type at 236 GB/s
// (LLM.md L0b) -- and because the arithmetic *would* bite in a process small
// enough to stay in the device-local heap, which a future benchmark or a
// smaller deployment could be.

// DefaultTinyConvKernel is the convolution build taef1 gets, and it is not the
// full decoder's DefaultConvKernel.
//
// The reason is the output-channel tile. Stage 8's ladder chose a BM of 128
// because the full decoder's convolutions produce 128 to 512 channels; every
// convolution in taef1 produces 64, so half of a 128-row A tile is padding and
// the kernel does twice the work it needs to. Measured at a 1024x1024 image,
// `go run ./cmd/vaebench -tiny -sizes 128 -conv <k>`: 64x64_w32 86.1 ms,
// 64x64 88.6, 64x128 88.5, 128x64_w32 97.2, 128x128_w32 97.8, 256x64 126.4.
// The ordering is the same at 512x512 and 256x256, where it is worth less
// because the whole decode is.
const DefaultTinyConvKernel = Conv64x64W32

// GPUTiny runs taef1's decoder on a Vulkan device.
type GPUTiny struct {
	engine
	cpu *TinyDecoder

	arena  arena
	harena arena

	// maxH/maxW are the latent grid the arenas were sized for. A smaller one
	// runs in them; a larger one is refused, because the graph is re-recorded
	// per decode and would silently overrun.
	maxH, maxW int
}

// tinyShaderSet is every pipeline taef1's graph uses on the scalar path.
var tinyShaderSet = map[string][]byte{
	"conv2d":     shaders.VAEConv2D,
	"relu":       shaders.VAEReLU,
	"add":        shaders.VAEAdd,
	"upsample2x": shaders.VAEUpsample2x,
}

// NewGPUTiny stages taef1 on the device with the default convolution kernel.
// latentH/latentW are the largest latent grid it will be asked for, i.e. the
// image's sides over 8.
func NewGPUTiny(dev *vk.Device, cpu *TinyDecoder, latentH, latentW int) (*GPUTiny, error) {
	return NewGPUTinyOpts(dev, cpu, latentH, latentW, Options{})
}

// NewGPUTinyOpts is NewGPUTiny with the convolution kernel named. Only Conv is
// read: this decoder has no mid block, so there is no attention and no GEMM to
// choose.
func NewGPUTinyOpts(dev *vk.Device, cpu *TinyDecoder, latentH, latentW int, opt Options) (*GPUTiny, error) {
	if opt.Attn != "" || opt.GEMM != "" {
		return nil, fmt.Errorf("vae: taef1 has no attention or GEMM; Options.Attn/GEMM are not settable for it")
	}
	g := &GPUTiny{engine: newEngine(dev), cpu: cpu, maxH: latentH, maxW: latentW}
	if opt.Conv != ConvScalar && hasMatrixCores(dev) {
		ck := opt.Conv
		if ck == "" {
			ck = DefaultTinyConvKernel
		}
		v, ok := convVariantFor(ck)
		if !ok {
			g.Destroy()
			return nil, fmt.Errorf("vae: no conv kernel %q (have %v)", ck, ConvKernels())
		}
		g.conv = v
	} else if opt.Conv != "" && opt.Conv != ConvScalar {
		g.Destroy()
		return nil, fmt.Errorf("vae: conv kernel %q needs fp16, cooperative matrices and subgroup size control", opt.Conv)
	}

	g.flattenWeights()
	var err error
	if g.wbuf, err = dev.NewBuffer(len(g.weights.data) * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: taef1 weight buffer: %w", err)
	}
	g.wbuf.WriteFloat32(g.weights.data)

	if g.convCores() {
		w16 := g.stageConvWeights(nil)
		if g.w16buf, err = dev.NewBuffer(len(w16) * 2); err != nil {
			g.Destroy()
			return nil, fmt.Errorf("vae: taef1 fp16 weight buffer: %w", err)
		}
		g.w16buf.WriteUint16At(0, w16)
	} else if g.w16buf, err = dev.NewBuffer(64); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: taef1 fp16 weight placeholder: %w", err)
	}

	need, hneed, err := g.planSize(latentH, latentW)
	if err != nil {
		g.Destroy()
		return nil, err
	}
	// Host-cached: see the note at the top of this file, and
	// ZIMAGE_TINY_UNCACHED for the control that measures it.
	newArena := dev.NewHostCachedBuffer
	if os.Getenv("ZIMAGE_TINY_UNCACHED") != "" {
		newArena = dev.NewBuffer
	}
	if g.abuf, err = newArena(int(need) * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: taef1 activation buffer (%d MB): %w", (int(need)*4)>>20, err)
	}
	if g.hbuf, err = dev.NewBuffer(max(int(hneed), 64) * 2); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: taef1 fp16 activation buffer (%d MB): %w", (int(hneed)*2)>>20, err)
	}
	g.hbuf.Zero()

	set := map[string][]byte{}
	for name, spirv := range tinyShaderSet {
		set[name] = spirv
	}
	if g.convCores() {
		set["pack_conv"] = g.conv.packSpirv()
	}
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	for name, spirv := range set {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: g.bufs(), PushConstantSize: pcSize}); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	if g.convCores() {
		if err := g.pipeline("conv_wmma", g.conv.spirv, vk.PipelineSpec{
			Buffers: g.bufs(), PushConstantSize: pcSize, RequiredSubgroupSize: g.conv.wave,
		}); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUTiny) Destroy() { g.engine.destroy() }

// ConvKernel names the chosen convolution build, for a benchmark's output.
func (g *GPUTiny) ConvKernel() ConvKernel {
	if g.convCores() {
		return g.conv.name
	}
	return ConvScalar
}

// flattenWeights copies every filter into the fp32 arena in graph order, under
// the same "layers.N" names the checkpoint uses.
func (g *GPUTiny) flattenWeights() {
	w := g.weights
	conv := func(name string, c *Conv2D) {
		w.put(name+".weight", c.Weight)
		if c.Bias != nil {
			w.put(name+".bias", c.Bias)
		}
		g.convs = append(g.convs, convRef{name, c})
	}
	for _, l := range g.cpu.Layers {
		name := fmt.Sprintf("layers.%d", l.Index)
		switch {
		case l.Conv != nil:
			conv(name, l.Conv)
		case l.Block != nil:
			conv(name+".conv0", l.Block.Conv0)
			conv(name+".conv2", l.Block.Conv2)
			conv(name+".conv4", l.Block.Conv4)
		}
	}
	// The expanded biases the matrix-core convolution starts its accumulators
	// from. Three of taef1's convolutions have none -- the ones after the
	// upsamples are built with bias=False -- and those simply have no entry,
	// which convBiasOff already reads as NO_BIAS.
	if g.convCores() {
		for _, cr := range g.convs {
			if cr.conv.Bias != nil {
				w.put(cr.name+".bias16", expandBias(cr.conv.Bias, cr.conv.OutC))
			}
		}
	}
}

// relu appends an in-place ReLU.
func (b *builder) relu(x tensor) tensor {
	b.add("relu", groups(x.elems(), 256), pushConstants{
		InOff: x.off, OutOff: x.off, Aux0: uint32(x.elems()),
	})
	return x
}

// tinyBlock records one AutoencoderTinyBlock: three convolutions with a ReLU
// between them, added to the input, and a fourth ReLU over the sum.
//
// x is read at the end, so nothing may be allocated over it until then --
// same rule as builder.resnet, and the same reason. It is released here
// because the block consumes it: taef1's graph is a straight line.
func (b *builder) tinyBlock(name string, blk *TinyBlock, x tensor) tensor {
	h := b.conv(name+".conv0", blk.Conv0, x)
	h = b.relu(h)
	h2 := b.conv(name+".conv2", blk.Conv2, h)
	b.release(h)
	h2 = b.relu(h2)
	h3 := b.conv(name+".conv4", blk.Conv4, h2)
	b.release(h2)
	out := b.addInto(h3, x)
	b.release(x)
	return b.relu(out)
}

// tinyUpsample records the nearest 2x. taef1's upsample has no convolution
// attached -- the one that follows it is the next layer of the Sequential --
// which is the difference from builder.upsample.
func (b *builder) tinyUpsample(x tensor) tensor {
	up := tensor{off: b.ar.alloc(x.C * x.H * 2 * x.W * 2), C: x.C, H: x.H * 2, W: x.W * 2}
	b.add("upsample2x", groups(up.elems(), 256), pushConstants{
		InOff: x.off, OutOff: up.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
	})
	b.release(x)
	return up
}

// buildTiny records the whole decode and returns the output tensor.
//
// The clamp in front and the [0,1] -> [-1,1] behind are not in here: they are
// one multiply-add each over the smallest and the largest tensor in the graph,
// and both are already being copied across the bus, so the host does them for
// free on the way in and on the way out. A dispatch for either would cost a
// submit and save nothing.
func (b *builder) buildTiny(d *TinyDecoder, latentH, latentW int) tensor {
	h := tensor{off: b.ar.alloc(d.Cfg.LatentChannels * latentH * latentW),
		C: d.Cfg.LatentChannels, H: latentH, W: latentW}
	for _, l := range d.Layers {
		name := fmt.Sprintf("layers.%d", l.Index)
		switch {
		case l.Conv != nil:
			in := h
			h = b.conv(name, l.Conv, h)
			b.release(in)
		case l.Block != nil:
			h = b.tinyBlock(name, l.Block, h)
		case l.Upsample:
			h = b.tinyUpsample(h)
		case l.ReLU:
			h = b.relu(h)
		}
	}
	return h
}

// planSize records the graph with no pipelines bound, purely to size the two
// activation arenas.
func (g *GPUTiny) planSize(latentH, latentW int) (uint32, uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{e: &g.engine, ar: &arena{}, har: &arena{}}
	b.buildTiny(g.cpu, latentH, latentW)
	g.pipes = saved
	if b.err != nil {
		return 0, 0, b.err
	}
	return b.ar.high, b.har.high, nil
}

// Apply decodes a latent to an image in [-1, 1], the same contract as
// TinyDecoder.Apply and as GPUDecoder.Apply.
//
// The latent is the raw diffusion latent -- see the note in tiny.go -- so a
// caller previewing a pipeline's state hands over what the loop holds, with no
// scaling_factor in the way.
func (g *GPUTiny) Apply(latent *Tensor) (*Tensor, error) {
	if latent.N != 1 {
		return nil, fmt.Errorf("vae: taef1 decodes one image at a time, got N=%d", latent.N)
	}
	if latent.C != g.cpu.Cfg.LatentChannels {
		return nil, fmt.Errorf("vae: latent has %d channels, taef1 takes %d",
			latent.C, g.cpu.Cfg.LatentChannels)
	}
	if latent.H > g.maxH || latent.W > g.maxW {
		return nil, fmt.Errorf("vae: a %dx%d latent; taef1's arenas were built for %dx%d",
			latent.H, latent.W, g.maxH, g.maxW)
	}

	g.arena.reset()
	g.harena.reset()
	b := &builder{e: &g.engine, ar: &g.arena, har: &g.harena}
	out := b.buildTiny(g.cpu, latent.H, latent.W)
	if b.err != nil {
		return nil, b.err
	}
	if int(b.ar.high)*4 > g.abuf.Size() || int(b.har.high)*2 > g.hbuf.Size() {
		return nil, fmt.Errorf("vae: taef1 at %dx%d needs %d/%d MB of arena, has %d/%d",
			latent.H, latent.W, (int(b.ar.high)*4)>>20, (int(b.har.high)*2)>>20,
			g.abuf.Size()>>20, g.hbuf.Size()>>20)
	}

	// The clamp, applied on the way in. The latent is being copied to the
	// device anyway and it is the smallest tensor in the graph.
	mag := float32(g.cpu.Cfg.LatentMagnitude)
	clamped := make([]float32, latent.Len())
	for i, v := range latent.Data {
		clamped[i] = tanh32(v/mag) * mag
	}
	g.abuf.WriteFloat32(clamped)

	// Same batching as the full decoder, and for the same reason: a single
	// command buffer holding the whole graph is what tripped the driver's
	// reset watchdog there. taef1's dispatches are much smaller, but the cap
	// is free -- the fence wait is not what this decode is made of -- and a
	// cap that only holds for the shapes measured so far is not a cap.
	for i := 0; i < len(b.out); i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, len(b.out))
		if _, err := vk.DispatchMultiTimed(b.out[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("vae: taef1 dispatch %d-%d: %w", i, j-1, err)
		}
	}

	res := NewTensor(1, out.C, out.H, out.W)
	src := g.abuf.ReadFloat32At(int(out.off), out.elems())
	// [0, 1] to [-1, 1], on the way out and in the copy that was happening
	// anyway.
	for i, v := range src {
		res.Data[i] = v*2 - 1
	}
	return res, nil
}

// Dispatches is how many GPU dispatches one preview records.
func (g *GPUTiny) Dispatches(latentH, latentW int) (int, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{e: &g.engine, ar: &arena{}, har: &arena{}}
	b.buildTiny(g.cpu, latentH, latentW)
	g.pipes = saved
	if b.err != nil {
		return 0, b.err
	}
	return len(b.out), nil
}

// Profile runs the graph one dispatch at a time, timing each on the GPU. Same
// caveat as GPUDecoder.Profile: submitting separately costs a fence wait
// apiece, so what this is for is the distribution rather than the total.
func (g *GPUTiny) Profile(latent *Tensor) ([]Stage, error) {
	g.arena.reset()
	g.harena.reset()
	b := &builder{e: &g.engine, ar: &g.arena, har: &g.harena}
	b.buildTiny(g.cpu, latent.H, latent.W)
	if b.err != nil {
		return nil, b.err
	}
	mag := float32(g.cpu.Cfg.LatentMagnitude)
	clamped := make([]float32, latent.Len())
	for i, v := range latent.Data {
		clamped[i] = tanh32(v/mag) * mag
	}
	g.abuf.WriteFloat32(clamped)

	stages := make([]Stage, 0, len(b.out))
	for i, d := range b.out {
		dur, err := d.Pipeline.DispatchTimed(d.GroupsX, d.GroupsY, 1, 1, d.PushConstants)
		if err != nil {
			return stages, fmt.Errorf("vae: taef1 dispatch %d (%s): %w", i, b.kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: b.kinds[i], GPU: dur, Flops: b.flops[i]})
	}
	return stages, nil
}

// ActivationBytes, F16ActivationBytes and WeightBytes are what the preview
// decoder costs in residency, for the startup banner. They are separate
// numbers from the full decoder's because they are separate allocations: a
// server with previews turned on holds both.
func (g *GPUTiny) ActivationBytes() int    { return g.abuf.Size() }
func (g *GPUTiny) F16ActivationBytes() int { return g.hbuf.Size() }
func (g *GPUTiny) WeightBytes() (fp32, fp16 int) {
	return g.wbuf.Size(), g.w16buf.Size()
}
