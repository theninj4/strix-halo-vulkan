package vae

import (
	"context"
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
	zvae "strix-halo-vulkan/zimage/vae"
)

// The encoder on the device — IMAGE.md Q8, the stage Q5g deferred.
//
// Nothing t2i serves needs an encoder, which is why the decoder went to the
// GPU alone and this waited for edits. An edit needs it per request and per
// reference image: the CPU encoder is the oracle and costs minutes at a
// served condition size, which is the same argument that moved the decoder.
//
// It is a small file because the encoder is the decoder's graph read
// backwards over the same kernels. Two things differ and only one of them is
// new:
//
//   - **the stride-2 downsampler is not a new kernel**, and it is not even a
//     new idea: z-image's encoder found that a 3x3 stride-2 convolution with
//     diffusers' (0, 1, 0, 1) padding computes exactly what the stride-1
//     pad-1 form computes at the *odd* pixels, so it runs the ordinary
//     convolution and throws three pixels in four away. Qwen's downsampler
//     has the identical shape, and vae_downsample2x.comp — which keeps
//     (2h+1, 2w+1) — is reused unmodified. The discarded three quarters are
//     the price of not writing a second convolution;
//   - **AvgDown3D is new** (qvae_avgdown.comp), the mirror of the decoder's
//     DupUp: space folded into channels and the group averaged, with a
//     temporal factor that at T = 1 still averages in a zero frame and still
//     divides by the full group.
//
// The mid block is **768** channels wide here against the decoder's 1152 —
// base_dim 96 against decoder_base_dim 144 — so the attention kernel is a
// different build of the same source, and the width is checked at
// construction for the same reason the decoder checks its own.

// encoderShaderSet is the encoder's pipelines. Every one of them is the
// decoder's or z-image's except qvae_avgdown; the up-only kernels
// (upsample2x, dupup) are not built here.
var encoderShaderSet = map[string][]byte{
	"conv2d":       shaders.VAEConv2D,
	"add":          shaders.VAEAdd,
	"downsample2x": shaders.VAEDownsample2x,
	"to_rows":      shaders.VAENCHWToRows,
	"to_nchw":      shaders.VAERowsToNCHWAdd,
	"linear":       shaders.VAELinear,
	"transpose":    shaders.VAETranspose,
	"attention":    shaders.VAEAttentionDim768,
	"chnorm":       shaders.QVAEChannelNorm,
	"avgdown":      shaders.QVAEAvgDown,
}

// encMidDim is the channel width encoderShaderSet's attention build expects.
const encMidDim = 768

// GPUEncoder runs the encode graph on a Vulkan device.
type GPUEncoder struct {
	engine
	cpu *Encoder
}

// NewGPUEncoder uploads a loaded encoder's weights, builds its pipelines and
// sizes the activation arena for an imgH x imgW image — the largest the
// returned encoder will accept, since the arena is allocated once.
func NewGPUEncoder(dev *vk.Device, cpu *Encoder, imgH, imgW int) (*GPUEncoder, error) {
	if got := cpu.Mid.Attn.QKV.InC; got != encMidDim {
		return nil, fmt.Errorf("qvae: encoder mid block is %d channels, the attention kernel is built for %d", got, encMidDim)
	}
	g := &GPUEncoder{engine: newEngine(), cpu: cpu}
	g.flattenWeights()

	need, err := g.planSize(imgH, imgW)
	if err != nil {
		g.Destroy()
		return nil, err
	}
	if err := g.stage(dev, encoderShaderSet, need); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *GPUEncoder) Destroy() { g.destroy() }

// flattenWeights copies every weight the graph reads into one arena, in the
// order the graph reads them.
func (g *GPUEncoder) flattenWeights() {
	w := g.weights
	conv := func(name string, c *zvae.Conv2D) {
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

	e := g.cpu
	conv("conv_in", &e.ConvIn)
	for i := range e.Downs {
		down := &e.Downs[i]
		for j := range down.Resnets {
			res(fmt.Sprintf("down.%d.r%d", i, j), &down.Resnets[j])
		}
		if down.Down != nil {
			conv(fmt.Sprintf("down.%d.downconv", i), &down.Down.Conv)
		}
	}
	res("mid.r1", &e.Mid.Res1)
	norm("mid.attn.norm", &e.Mid.Attn.Norm)
	conv("mid.attn.qkv", &e.Mid.Attn.QKV)
	conv("mid.attn.proj", &e.Mid.Attn.Proj)
	res("mid.r2", &e.Mid.Res2)
	norm("norm_out", &e.NormOut)
	conv("conv_out", &e.ConvOut)
	conv("quant", &e.Quant)
}

// downsample records one stride-2 convolution as a stride-1 one followed by
// the subsampling kernel, which is z-image's identity on Qwen's filter: a
// 3x3 stride-2 convolution over a (0, 1, 0, 1)-padded input computes exactly
// what the 3x3 stride-1 pad-1 form computes at the odd pixels.
//
// It costs four times the arithmetic of a real stride-2 kernel and saves a
// second convolution shader. Whether that trade is still right is a Q9
// question, not this stage's: the same sentence is in zimage/vae, with the
// same reasoning and the same numbers waiting to be measured.
func (b *builder) downsample(name string, c *zvae.Conv2D, x tensor) tensor {
	if c.Stride != 2 || c.KH != 3 || c.KW != 3 || c.Pad != 0 || c.PadEnd != 1 {
		b.err = fmt.Errorf("qvae: downsample %s is %dx%d stride %d pad %d/%d; the subsampling identity is for a 3x3 stride-2 (0,1,0,1) filter",
			name, c.KH, c.KW, c.Stride, c.Pad, c.PadEnd)
		return x
	}
	// The same weights read as an ordinary symmetric-pad convolution.
	stride1 := *c
	stride1.Pad, stride1.PadEnd, stride1.Stride = 1, 0, 1
	h := b.conv(name, &stride1, x)

	out := tensor{off: b.ar.alloc(h.C * (h.H / 2) * (h.W / 2)), C: h.C, H: h.H / 2, W: h.W / 2}
	b.label(fmt.Sprintf("downsample %d @%dx%d", h.C, h.H, h.W), 0)
	b.add("downsample2x", groups(out.elems(), 256), pushConstants{
		InOff: h.off, OutOff: out.off,
		C: uint32(h.C), H: uint32(h.H), W: uint32(h.W),
	})
	b.release(h)
	return out
}

// avgdown is the parameter-free shortcut around a down block.
func (b *builder) avgdown(a *AvgDown, x tensor) tensor {
	fs := a.FactorS
	out := tensor{off: b.ar.alloc(a.Out * (x.H / fs) * (x.W / fs)), C: a.Out, H: x.H / fs, W: x.W / fs}
	b.label(fmt.Sprintf("avgdown %d->%d @%dx%d", a.In, a.Out, out.H, out.W), 0)
	b.add("avgdown", groups(out.elems(), 256), pushConstants{
		InOff: x.off, OutOff: out.off,
		C: uint32(a.In), H: uint32(x.H), W: uint32(x.W), OC: uint32(a.Out),
		Aux0: uint32(a.FactorT), Aux1: uint32(fs),
	})
	return out
}

// buildEncode records the whole encode and returns the quantized output —
// [2z, h, w], mean then logvar. Encode keeps the mean half, which is the
// posterior mode and is contiguous in NCHW.
func (b *builder) buildEncode(e *Encoder, imgH, imgW int) tensor {
	img := tensor{off: b.ar.alloc(e.ConvIn.InC * imgH * imgW), C: e.ConvIn.InC, H: imgH, W: imgW}
	h := b.conv("conv_in", &e.ConvIn, img)
	b.release(img)
	b.mark("conv_in", h)

	for i := range e.Downs {
		d := &e.Downs[i]
		in := h
		// The shortcut reads the block's *input*, so it is recorded before
		// the resnets overwrite it and the input is kept alive until it has.
		short := b.avgdown(&d.Shortcut, in)
		for j := range d.Resnets {
			h = b.resblock(fmt.Sprintf("down.%d.r%d", i, j), &d.Resnets[j], h, j == 0)
		}
		if d.Down != nil {
			h = b.downsample(fmt.Sprintf("down.%d.downconv", i), &d.Down.Conv, h)
		}
		h = b.addInto(h, short)
		b.release(short, in)
		b.mark(fmt.Sprintf("down_blocks.%d", i), h)
	}

	h = b.resblock("mid.r1", &e.Mid.Res1, h, false)
	h = b.attention("mid.attn", &e.Mid.Attn, h)
	h = b.resblock("mid.r2", &e.Mid.Res2, h, false)
	b.mark("mid_block", h)

	h = b.chnorm("norm_out", true, h, h)
	b.mark("nonlinearity", h)
	out := b.conv("conv_out", &e.ConvOut, h)
	b.release(h)
	b.mark("conv_out", out)
	q := b.conv("quant", &e.Quant, out)
	b.release(out)
	return b.mark("quant", q)
}

// plan records the graph. A pipeline-less encoder records it purely to size
// the arena or count the dispatches.
func (g *GPUEncoder) plan(ar *arena, imgH, imgW int, marks bool) (*builder, tensor, error) {
	b := &builder{e: &g.engine, ar: ar}
	if marks {
		b.marks = map[string]tensor{}
	}
	out := b.buildEncode(g.cpu, imgH, imgW)
	return b, out, b.err
}

func (g *GPUEncoder) planSize(imgH, imgW int) (uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, imgH, imgW, false)
	g.pipes = saved
	if err != nil {
		return 0, err
	}
	return b.ar.high, nil
}

// EncoderArenaBytes is how much activation memory an encode of this image
// would need, without a device and without loading a weight — the encoder's
// half of the ceiling arithmetic ArenaBytes does for the decoder.
func EncoderArenaBytes(cpu *Encoder, imgH, imgW int) (int, error) {
	g := &GPUEncoder{engine: newEngine(), cpu: cpu}
	n, err := g.planSize(imgH, imgW)
	if err != nil {
		return 0, err
	}
	return int(n) * 4, nil
}

// Dispatches is the number of GPU dispatches one encode records.
func (g *GPUEncoder) Dispatches(imgH, imgW int) (int, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, imgH, imgW, false)
	g.pipes = saved
	if err != nil {
		return 0, err
	}
	return len(b.out), nil
}

// record builds the graph against the real arena and uploads the image.
func (g *GPUEncoder) record(img *zvae.Tensor, marks bool) (*builder, tensor, error) {
	if img.N != 1 {
		return nil, tensor{}, fmt.Errorf("qvae: the GPU encoder takes one image at a time, got N=%d", img.N)
	}
	if img.C != g.cpu.ConvIn.InC {
		return nil, tensor{}, fmt.Errorf("qvae: image has %d channels, conv_in takes %d", img.C, g.cpu.ConvIn.InC)
	}
	g.arena.reset()
	b, out, err := g.plan(&g.arena, img.H, img.W, marks)
	if err != nil {
		return nil, tensor{}, err
	}
	if int(b.ar.high)*4 > g.abuf.Size() {
		return nil, tensor{}, fmt.Errorf("qvae: graph needs %d MB of activations, arena is %d MB",
			(int(b.ar.high)*4)>>20, g.abuf.Size()>>20)
	}
	// The image is the arena's first allocation, so it goes in at offset 0.
	g.abuf.WriteFloat32(img.Data)
	return b, out, nil
}

// Encode returns the posterior mode of one image in [-1, 1], which is the
// mean half of the quantized output and the only half an edit ever reads
// (IMAGE.md decision 3). The latents come back *raw* — Config.Normalize maps
// them into the DiT's space, exactly as the CPU encoder's caller does it.
func (g *GPUEncoder) Encode(ctx context.Context, img *zvae.Tensor) (*zvae.Tensor, error) {
	b, out, err := g.record(img, false)
	if err != nil {
		return nil, err
	}
	if err := g.runContext(ctx, b.out); err != nil {
		return nil, err
	}
	return g.read(tensor{off: out.off, C: out.C / 2, H: out.H, W: out.W}), nil
}

// Stages names the graph's marked stages, in order. They are the names the
// CPU encoder's Tap emits, so a stagewise comparison needs no mapping.
func (g *GPUEncoder) Stages(imgH, imgW int) ([]string, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	b, _, err := g.plan(&arena{}, imgH, imgW, true)
	g.pipes = saved
	if err != nil {
		return nil, err
	}
	return b.order, nil
}

// RunTo re-runs the graph from the start and stops after the named stage,
// returning what it produced. A bump-allocated graph overwrites every
// intermediate long before it ends, so re-running the prefix is the only way
// to see one.
func (g *GPUEncoder) RunTo(img *zvae.Tensor, stage string) (*zvae.Tensor, error) {
	b, _, err := g.record(img, true)
	if err != nil {
		return nil, err
	}
	t, ok := b.marks[stage]
	if !ok {
		return nil, fmt.Errorf("qvae: no stage %q in the encode graph", stage)
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
