package vae

import (
	"fmt"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// The AutoencoderKL encoder on the device -- IMAGE.md I7.
//
// Like tiny_gpu.go there is almost nothing here, and for the same reason: the
// graph is 27 convolutions, 8 resnets, one mid block with attention and three
// resolution changes, and every one of those but the last is a kernel the
// decoder already has. So this file is a weight staging, a graph and a submit
// loop over the same `engine`, and the mid block reaches the matrix cores
// through the same `builder.attentionWMMA` -- it is the *identical* block,
// 512 channels at the latent's resolution, in both directions.
//
// **The stride-2 convolution is not a new kernel.** The three downsamplers run
// their own filter at stride 1 on stage 8's implicit GEMM and then keep one
// pixel in four (builder.downsample). That does four times the arithmetic the
// strided form would, which at a 1024x1024 image is 700 GFLOP, about 18 ms of
// an encode -- against writing a second convolution kernel with a stride in
// its address arithmetic, a second packing layout to feed it and a second
// ladder to tune it. The identity behind it is measured rather than argued:
// TestStrideTwoIsStrideOneSubsampled.

// GPUEncoder runs the encoder graph on a Vulkan device.
type GPUEncoder struct {
	engine
	cpu *Encoder

	arena  arena
	harena arena

	// maxH/maxW are the *image* the arenas were sized for. A smaller one runs
	// in them; a larger one is refused, because the graph is re-recorded per
	// encode and would silently overrun.
	maxH, maxW int
}

// encoderShaderSet is every pipeline the encoder graph uses on the scalar
// path. It is the decoder's set with the upsample swapped for the
// downsample: nothing else about the two directions differs at this level.
var encoderShaderSet = map[string][]byte{
	"conv2d":       shaders.VAEConv2D,
	"groupnorm":    shaders.VAEGroupNorm,
	"silu":         shaders.VAESiLU,
	"add":          shaders.VAEAdd,
	"downsample2x": shaders.VAEDownsample2x,
	"to_rows":      shaders.VAENCHWToRows,
	"to_nchw":      shaders.VAERowsToNCHWAdd,
	"linear":       shaders.VAELinear,
	"attention":    shaders.VAEAttention,
	"transpose":    shaders.VAETranspose,
}

// NewGPUEncoder stages the encoder on the device with the default kernels.
// imgH/imgW are the largest image it will be asked for, in pixels.
func NewGPUEncoder(dev *vk.Device, cpu *Encoder, imgH, imgW int) (*GPUEncoder, error) {
	return NewGPUEncoderOpts(dev, cpu, imgH, imgW, Options{})
}

// NewGPUEncoderOpts is NewGPUEncoder with the kernels named, which is what the
// negative controls in the tests select.
func NewGPUEncoderOpts(dev *vk.Device, cpu *Encoder, imgH, imgW int, opt Options) (*GPUEncoder, error) {
	g := &GPUEncoder{engine: newEngine(dev), cpu: cpu, maxH: imgH, maxW: imgW}
	if err := g.engine.chooseKernels(opt); err != nil {
		g.Destroy()
		return nil, err
	}

	g.flattenWeights()
	wbytes := len(g.weights.data) * 4
	var err error
	if g.wbuf, err = dev.NewBuffer(wbytes); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: encoder weight buffer (%d MB): %w", wbytes>>20, err)
	}
	g.wbuf.WriteFloat32(g.weights.data)

	if g.f16() {
		var w16 []uint16
		if g.wmma() {
			w16 = g.stageProjections(w16, cpu.Mid.Attn)
		}
		if g.convCores() {
			w16 = g.stageConvWeights(w16)
		}
		if g.w16buf, err = dev.NewBuffer(max(len(w16), 32) * 2); err != nil {
			g.Destroy()
			return nil, fmt.Errorf("vae: encoder fp16 weight buffer (%d MB): %w", (len(w16)*2)>>20, err)
		}
		g.w16buf.WriteUint16At(0, w16)
	} else if g.w16buf, err = dev.NewBuffer(64); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: encoder fp16 weight placeholder: %w", err)
	}

	need, hneed, err := g.planSize(imgH, imgW)
	if err != nil {
		g.Destroy()
		return nil, err
	}
	if g.abuf, err = dev.NewBuffer(int(need) * 4); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: encoder activation buffer (%d MB): %w", (int(need)*4)>>20, err)
	}
	if g.hbuf, err = dev.NewBuffer(max(int(hneed), 64) * 2); err != nil {
		g.Destroy()
		return nil, fmt.Errorf("vae: encoder fp16 activation buffer (%d MB): %w", (int(hneed)*2)>>20, err)
	}
	// Same argument as the decoder's: both passes that write it cover their
	// own pad rows, so this is hygiene rather than an invariant -- but one
	// infinity in a row of A makes that whole row of the GEMM's output a NaN.
	g.hbuf.Zero()

	set := map[string][]byte{}
	for name, spirv := range encoderShaderSet {
		set[name] = spirv
	}
	if g.wmma() {
		for name, spirv := range wmmaShaderSet {
			set[name] = spirv
		}
	}
	if g.convCores() {
		set["pack_conv"] = g.conv.packSpirv()
	}
	bufs := g.bufs()
	pcSize := uint32(unsafe.Sizeof(pushConstants{}))
	for name, spirv := range set {
		if err := g.pipeline(name, spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	if g.wmma() {
		if err := g.pipeline("attn_wmma", g.attn.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: g.attn.wave,
		}); err != nil {
			g.Destroy()
			return nil, err
		}
		spec := vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize}
		if g.gemm.waves > 1 {
			spec.RequiredSubgroupSize = 64
		}
		if err := g.pipeline("gemm", g.gemm.spirv, spec); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	if g.convCores() {
		if err := g.pipeline("conv_wmma", g.conv.spirv, vk.PipelineSpec{
			Buffers: bufs, PushConstantSize: pcSize, RequiredSubgroupSize: g.conv.wave,
		}); err != nil {
			g.Destroy()
			return nil, err
		}
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUEncoder) Destroy() { g.engine.destroy() }

// Kernels names the graph's three chosen kernels, for a benchmark's output.
func (g *GPUEncoder) Kernels() (AttnKernel, GEMMKernel, ConvKernel) { return g.engine.kernels() }

// flattenWeights copies every weight the graph reads into one arena, in the
// order the graph reads them.
func (g *GPUEncoder) flattenWeights() {
	w := g.weights
	g.convs = nil
	conv := func(name string, c *Conv2D) {
		w.put(name+".weight", c.Weight)
		if c.Bias != nil {
			w.put(name+".bias", c.Bias)
		}
		g.convs = append(g.convs, convRef{name, c})
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
	for i, down := range g.cpu.DownBlocks {
		for j, r := range down.Resnets {
			resnet(fmt.Sprintf("down.%d.r%d", i, j), r)
		}
		if down.Downsampler != nil {
			conv(fmt.Sprintf("down.%d.downconv", i), down.Downsampler)
		}
	}
	resnet("mid.r1", g.cpu.Mid.Resnet1)
	norm("mid.attn.norm", g.cpu.Mid.Attn.GroupNorm)
	lin("mid.attn.q", g.cpu.Mid.Attn.Q)
	lin("mid.attn.k", g.cpu.Mid.Attn.K)
	lin("mid.attn.v", g.cpu.Mid.Attn.V)
	lin("mid.attn.out", g.cpu.Mid.Attn.Out)
	resnet("mid.r2", g.cpu.Mid.Resnet2)
	norm("conv_norm_out", g.cpu.ConvNormOut)
	conv("conv_out", g.cpu.ConvOut)

	if g.convCores() {
		for _, cr := range g.convs {
			if cr.conv.Bias != nil {
				w.put(cr.name+".bias16", expandBias(cr.conv.Bias, cr.conv.OutC))
			}
		}
	}
}

// downsample records one stride-2 convolution as a stride-1 one followed by
// vae_downsample2x.comp.
//
// **The filter is dispatched with Pad 1 and no stride**, which is not what the
// Conv2D says, and that substitution is the whole of the trick: output (oh,
// ow) of the strided form reads input pixels 2oh..2oh+2 by 2ow..2ow+2, which
// is what the stride-1 pad-1 form computes at (2oh+1, 2ow+1) -- at the far
// edge included, where the stride-1 form's own zero padding supplies the pixel
// diffusers' (0, 1, 0, 1) pad would have.
//
// It costs four times the convolution's arithmetic and a full-resolution
// temporary: at 1024x1024 the three downsamplers are 927 GFLOP instead of 232,
// about 18 ms of the encode at the measured 38 TFLOP/s. What it buys is that
// no kernel, no packed layout and no ladder in stage 8 has to learn about a
// stride. If the encode ever matters that is the first thing to take back.
func (b *builder) downsample(name string, c *Conv2D, x tensor) tensor {
	if c.Stride != 2 || c.KH != 3 || c.KW != 3 {
		b.err = fmt.Errorf("vae: downsample %s is %dx%d stride %d; the subsampling identity is for 3x3 stride 2",
			name, c.KH, c.KW, c.Stride)
		return x
	}
	dense := &Conv2D{InC: c.InC, OutC: c.OutC, KH: c.KH, KW: c.KW, Pad: 1, Weight: c.Weight, Bias: c.Bias}
	h := b.conv(name, dense, x)
	b.release(x)

	out := tensor{off: b.ar.alloc(c.OutC * (h.H / 2) * (h.W / 2)), C: c.OutC, H: h.H / 2, W: h.W / 2}
	b.label(fmt.Sprintf("downsample %dx%dx%d", h.C, h.H, h.W), 0)
	b.add("downsample2x", groups(out.elems(), 256), pushConstants{
		InOff: h.off, OutOff: out.off,
		C: uint32(h.C), H: uint32(h.H), W: uint32(h.W),
	})
	b.release(h)
	return out
}

// buildEncoder records the whole encode and returns the moments tensor,
// [2*LatentChannels, H/8, W/8]. The mode -- dropping the log-variance -- is
// done on the host, on the copy back that was happening anyway.
func (b *builder) buildEncoder(e *Encoder, imgH, imgW int) tensor {
	x := tensor{off: b.ar.alloc(e.ConvIn.InC * imgH * imgW), C: e.ConvIn.InC, H: imgH, W: imgW}
	h := b.mark("conv_in", b.conv("conv_in", e.ConvIn, x))
	b.release(x)
	for i, down := range e.DownBlocks {
		for j, r := range down.Resnets {
			h = b.mark(fmt.Sprintf("down.%d.resnets.%d", i, j),
				b.resnet(fmt.Sprintf("down.%d.r%d", i, j), r, h))
		}
		if down.Downsampler != nil {
			h = b.mark(fmt.Sprintf("down.%d.downsample", i),
				b.downsample(fmt.Sprintf("down.%d.downconv", i), down.Downsampler, h))
		}
	}
	h = b.mark("mid.resnets.0", b.resnet("mid.r1", e.Mid.Resnet1, h))
	if b.e.wmma() {
		h = b.attentionWMMA("mid.attn", e.Mid.Attn, h)
	} else {
		h = b.attention("mid.attn", e.Mid.Attn, h)
	}
	b.mark("mid.attn", h)
	h = b.mark("mid.resnets.1", b.resnet("mid.r2", e.Mid.Resnet2, h))
	h = b.mark("conv_norm_out", b.norm("conv_norm_out", e.ConvNormOut, h))
	h = b.silu(h)
	out := b.mark("conv_out", b.conv("conv_out", e.ConvOut, h))
	b.release(h)
	return out
}

// planSize records the graph with no pipelines bound, purely to size the two
// activation arenas.
func (g *GPUEncoder) planSize(imgH, imgW int) (uint32, uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{e: &g.engine, ar: &arena{}, har: &arena{}}
	b.buildEncoder(g.cpu, imgH, imgW)
	g.pipes = saved
	if b.err != nil {
		return 0, 0, b.err
	}
	return b.ar.high, b.har.high, nil
}

// Apply encodes an image to the posterior's parameters, [1, 2C, H/8, W/8].
// Encode is what a caller normally wants.
func (g *GPUEncoder) Apply(img *Tensor) (*Tensor, error) {
	b, out, err := g.record(img)
	if err != nil {
		return nil, err
	}
	g.abuf.WriteFloat32(img.Data)

	// Same batching as the decoder, and for the same reason: a single command
	// buffer holding the whole graph is what tripped the driver's reset
	// watchdog there.
	for i := 0; i < len(b.out); i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, len(b.out))
		if _, err := vk.DispatchMultiTimed(b.out[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("vae: encoder dispatch %d-%d: %w", i, j-1, err)
		}
	}
	res := NewTensor(1, out.C, out.H, out.W)
	copy(res.Data, g.abuf.ReadFloat32At(int(out.off), out.elems()))
	return res, nil
}

// Encode runs the encoder and takes the posterior's mode, which is the first
// half of Apply's channels. The copy back reads only that half, so the
// log-variance never crosses the bus.
func (g *GPUEncoder) Encode(img *Tensor) (*Tensor, error) {
	b, out, err := g.record(img)
	if err != nil {
		return nil, err
	}
	g.abuf.WriteFloat32(img.Data)
	for i := 0; i < len(b.out); i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, len(b.out))
		if _, err := vk.DispatchMultiTimed(b.out[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("vae: encoder dispatch %d-%d: %w", i, j-1, err)
		}
	}
	c := g.cpu.LatentChannels
	res := NewTensor(1, c, out.H, out.W)
	copy(res.Data, g.abuf.ReadFloat32At(int(out.off), c*out.H*out.W))
	return res, nil
}

// RunTo runs the graph as far as a named submodule and reads back what it
// produced, which is the only way to see an intermediate: the arena is a bump
// allocator and every tensor here is overwritten before the encode ends.
//
// The names are the ones reference/dump_vae_encoder.py dumps, so a stagewise
// test is a loop over the manifest rather than a table that has to be kept in
// step with it. zimage/dit's GPUBlock.RunTo is the same facility for the same
// reason.
func (g *GPUEncoder) RunTo(img *Tensor, name string) (*Tensor, error) {
	b, _, err := g.record(img)
	if err != nil {
		return nil, err
	}
	t, ok := b.marks[name]
	if !ok {
		return nil, fmt.Errorf("vae: the encoder has no stage %q (have %v)", name, b.order)
	}
	stop := 0
	for i, n := range b.order {
		if n == name {
			stop = b.stops[i]
			break
		}
	}
	g.abuf.WriteFloat32(img.Data)
	for i := 0; i < stop; i += dispatchesPerSubmit {
		j := min(i+dispatchesPerSubmit, stop)
		if _, err := vk.DispatchMultiTimed(b.out[i:j], 1, 1, true); err != nil {
			return nil, fmt.Errorf("vae: encoder dispatch %d-%d: %w", i, j-1, err)
		}
	}
	res := NewTensor(1, t.C, t.H, t.W)
	copy(res.Data, g.abuf.ReadFloat32At(int(t.off), t.elems()))
	return res, nil
}

// Stages names every submodule RunTo can stop at, in graph order.
func (g *GPUEncoder) Stages(imgH, imgW int) ([]string, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{e: &g.engine, ar: &arena{}, har: &arena{}, marks: map[string]tensor{}}
	b.buildEncoder(g.cpu, imgH, imgW)
	g.pipes = saved
	if b.err != nil {
		return nil, b.err
	}
	return b.order, nil
}

// record checks the image and builds the graph, which is everything Apply and
// Encode share.
func (g *GPUEncoder) record(img *Tensor) (*builder, tensor, error) {
	if img.N != 1 {
		return nil, tensor{}, fmt.Errorf("vae: the encoder handles one image at a time, got N=%d", img.N)
	}
	if img.C != g.cpu.ConvIn.InC {
		return nil, tensor{}, fmt.Errorf("vae: image has %d channels, conv_in takes %d", img.C, g.cpu.ConvIn.InC)
	}
	if img.H > g.maxH || img.W > g.maxW {
		return nil, tensor{}, fmt.Errorf("vae: a %dx%d image; the encoder's arenas were built for %dx%d",
			img.H, img.W, g.maxH, g.maxW)
	}
	// Every resolution change halves both sides three times over, and the
	// subsampling identity assumes an even side at each of them.
	if img.H%8 != 0 || img.W%8 != 0 {
		return nil, tensor{}, fmt.Errorf("vae: a %dx%d image; both sides must be a multiple of 8", img.H, img.W)
	}

	g.arena.reset()
	g.harena.reset()
	b := &builder{e: &g.engine, ar: &g.arena, har: &g.harena, marks: map[string]tensor{}}
	out := b.buildEncoder(g.cpu, img.H, img.W)
	if b.err != nil {
		return nil, tensor{}, b.err
	}
	if int(b.har.high)*2 > g.hbuf.Size() || int(b.ar.high)*4 > g.abuf.Size() {
		return nil, tensor{}, fmt.Errorf("vae: the encoder at %dx%d needs %d/%d MB of arena, has %d/%d",
			img.H, img.W, (int(b.ar.high)*4)>>20, (int(b.har.high)*2)>>20,
			g.abuf.Size()>>20, g.hbuf.Size()>>20)
	}
	return b, out, nil
}

// Profile runs the graph one dispatch at a time, timing each on the GPU. Same
// caveat as GPUDecoder.Profile: what it is for is the distribution.
func (g *GPUEncoder) Profile(img *Tensor) ([]Stage, error) {
	b, _, err := g.record(img)
	if err != nil {
		return nil, err
	}
	g.abuf.WriteFloat32(img.Data)
	stages := make([]Stage, 0, len(b.out))
	for i, d := range b.out {
		dur, err := d.Pipeline.DispatchTimed(d.GroupsX, d.GroupsY, 1, 1, d.PushConstants)
		if err != nil {
			return stages, fmt.Errorf("vae: encoder dispatch %d (%s): %w", i, b.kinds[i], err)
		}
		stages = append(stages, Stage{Index: i, Kind: b.kinds[i], GPU: dur, Flops: b.flops[i]})
	}
	return stages, nil
}

// Dispatches is how many GPU dispatches one encode records.
func (g *GPUEncoder) Dispatches(imgH, imgW int) (int, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b := &builder{e: &g.engine, ar: &arena{}, har: &arena{}}
	b.buildEncoder(g.cpu, imgH, imgW)
	g.pipes = saved
	if b.err != nil {
		return 0, b.err
	}
	return len(b.out), nil
}

// ActivationBytes, F16ActivationBytes and WeightBytes are what the encoder
// costs in residency, for the startup banner.
func (g *GPUEncoder) ActivationBytes() int    { return g.abuf.Size() }
func (g *GPUEncoder) F16ActivationBytes() int { return g.hbuf.Size() }
func (g *GPUEncoder) WeightBytes() (fp32, fp16 int) {
	return g.wbuf.Size(), g.w16buf.Size()
}
