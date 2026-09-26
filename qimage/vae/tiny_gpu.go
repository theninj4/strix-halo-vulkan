package vae

import (
	"context"
	"fmt"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// TAEQI2.1 on the device. The engine, the arena and the upsample are the full
// decoder's; the convolution is vae_conv2d.comp's EPILOGUE build, which fuses
// a block's ReLUs and its residual add into the store, so the whole graph is
// convolutions and upsamples — 38 dispatches, 35 of them convolutions.
//
// Everything is fp32 like the full decoder, though for a different reason:
// nothing here has been priced in fp16 and the graph is small enough not to
// need it (see TestTinyGPUTiming for what it costs against a DiT step).

// TinyKernel names a build of the epilogue convolution by its output-channel
// block.
type TinyKernel string

const (
	TinyOC16 TinyKernel = "oc16" // ic8
	TinyOC32 TinyKernel = "oc32" // ic16
	TinyOC64 TinyKernel = "oc64" // ic8
)

type tinyVariant struct {
	name    TinyKernel
	spirv   []byte
	ocBlock int
}

var tinyVariants = []tinyVariant{
	{TinyOC16, shaders.TAEConvOC16, 16},
	{TinyOC32, shaders.TAEConvOC32, 32},
	{TinyOC64, shaders.TAEConvOC64, 64},
}

// DefaultTinyKernel is TestTinyKernelScreen's winner at 1024²: 62.6 ms against
// oc32's 67.1 and oc16's 68.2, two runs agreeing, for the same bits. 64
// divides every width but conv_out's 16, which takes the oc16 build.
const DefaultTinyKernel = TinyOC64

// TinyKernels lists the arms, in the order they are screened.
func TinyKernels() []TinyKernel {
	out := make([]TinyKernel, 0, len(tinyVariants))
	for _, v := range tinyVariants {
		out = append(out, v.name)
	}
	return out
}

// The epilogue flags, aux0's bits in the EPILOGUE build.
const (
	epiResidual = 1
	epiReLU     = 2
)

// TinyGPUDecoder runs TAEQI2.1's decoder on a Vulkan device.
type TinyGPUDecoder struct {
	engine
	cpu  *TinyDecoder
	kern tinyVariant
}

// NewTinyGPUDecoder stages the decoder with its arena sized for a latentH x
// latentW latent, the largest it will then accept.
func NewTinyGPUDecoder(dev *vk.Device, cpu *TinyDecoder, latentH, latentW int) (*TinyGPUDecoder, error) {
	return NewTinyGPUDecoderKernel(dev, cpu, latentH, latentW, "")
}

// NewTinyGPUDecoderKernel is NewTinyGPUDecoder with the convolution build
// named; "" is the default.
func NewTinyGPUDecoderKernel(dev *vk.Device, cpu *TinyDecoder, latentH, latentW int, k TinyKernel) (*TinyGPUDecoder, error) {
	if k == "" {
		k = DefaultTinyKernel
	}
	g := &TinyGPUDecoder{engine: newEngine(), cpu: cpu}
	for _, v := range tinyVariants {
		if v.name == k {
			g.kern = v
		}
	}
	if g.kern.name == "" {
		return nil, fmt.Errorf("qvae: no tiny conv kernel %q (have %v)", k, TinyKernels())
	}
	g.flattenWeights()
	need, err := g.planSize(latentH, latentW)
	if err != nil {
		return nil, err
	}
	set := map[string][]byte{
		"tconv":      g.kern.spirv,
		"tconv16":    shaders.TAEConvOC16,
		"upsample2x": shaders.VAEUpsample2x,
	}
	if err := g.stage(dev, set, need); err != nil {
		g.Destroy()
		return nil, err
	}
	return g, nil
}

// Destroy releases every Vulkan object.
func (g *TinyGPUDecoder) Destroy() { g.destroy() }

// Kernel names the convolution build the decoder is running.
func (g *TinyGPUDecoder) Kernel() TinyKernel { return g.kern.name }

func (g *TinyGPUDecoder) flattenWeights() {
	w := g.weights
	conv := func(name string, c *Conv2D) {
		w.put(name+".weight", c.Weight)
		if c.Bias != nil {
			w.put(name+".bias", c.Bias)
		}
	}
	d := g.cpu
	conv("conv_in", &d.ConvIn)
	for s := range d.Stages {
		st := &d.Stages[s]
		for i := range st.Blocks {
			b := &st.Blocks[i]
			p := fmt.Sprintf("s%d.b%d", s, i)
			conv(p+".c0", &b.Conv0)
			conv(p+".c2", &b.Conv2)
			conv(p+".c4", &b.Conv4)
		}
		if st.Up != nil {
			conv(fmt.Sprintf("s%d.up", s), st.Up)
		}
	}
	conv("conv_out", &d.ConvOut)
}

// tconv records one epilogue convolution into a fresh tensor. res, when
// non-nil, is added before the optional ReLU; it must have the output's
// shape.
func (b *builder) tconv(kern tinyVariant, name string, c *Conv2D, x tensor, res *tensor, relu bool) tensor {
	out := tensor{off: b.ar.alloc(c.OutC * x.H * x.W), C: c.OutC, H: x.H, W: x.W}
	b.label(fmt.Sprintf("conv3x3 %d->%d @%dx%d", x.C, c.OutC, x.H, x.W),
		2*float64(c.OutC)*float64(x.H)*float64(x.W)*float64(x.C)*float64(c.KH)*float64(c.KW))
	pipe, ocBlock := "tconv", kern.ocBlock
	if c.OutC < ocBlock {
		pipe, ocBlock = "tconv16", 16
	}
	pc := pushConstants{
		InOff: x.off, OutOff: out.off,
		C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
		OC: uint32(c.OutC), KH: uint32(c.KH), KW: uint32(c.KW), Pad: uint32(c.Pad),
		WOff: b.wOff(name + ".weight"), BOff: b.bOff(name + ".bias"),
	}
	if res != nil {
		pc.ResOff = res.off
		pc.Aux0 |= epiResidual
	}
	if relu {
		pc.Aux0 |= epiReLU
	}
	b.addXY(pipe, groups(out.H*out.W, 256), groups(c.OutC, ocBlock), pc)
	return out
}

// buildTiny records the decode and returns conv_out's tensor, which is still
// the pre-shuffle [16, h, w]: the shuffle and the range change ride on the
// readback, as the full decoder's clamp does. Marks are named by the
// Sequential layer index that reference/dump_taeqi.py taps.
func (b *builder) buildTiny(kern tinyVariant, d *TinyDecoder, latentH, latentW int) tensor {
	z := tensor{off: b.ar.alloc(d.LatentChannels * latentH * latentW), C: d.LatentChannels, H: latentH, W: latentW}
	x := b.tconv(kern, "conv_in", &d.ConvIn, z, nil, true)
	b.release(z)
	b.mark("layer02", x)
	idx := 3
	for s := range d.Stages {
		st := &d.Stages[s]
		for i := range st.Blocks {
			blk := &st.Blocks[i]
			p := fmt.Sprintf("s%d.b%d", s, i)
			h0 := b.tconv(kern, p+".c0", &blk.Conv0, x, nil, true)
			h1 := b.tconv(kern, p+".c2", &blk.Conv2, h0, nil, true)
			b.release(h0)
			out := b.tconv(kern, p+".c4", &blk.Conv4, h1, &x, true)
			b.release(h1, x)
			x = b.mark(fmt.Sprintf("layer%02d", idx), out)
			idx++
		}
		if st.Up != nil {
			up := tensor{off: b.ar.alloc(x.C * x.H * 2 * x.W * 2), C: x.C, H: x.H * 2, W: x.W * 2}
			b.add("upsample2x", groups(up.elems(), 256), pushConstants{
				InOff: x.off, OutOff: up.off,
				C: uint32(x.C), H: uint32(x.H), W: uint32(x.W),
			})
			b.release(x)
			b.mark(fmt.Sprintf("layer%02d", idx), up)
			x = b.tconv(kern, fmt.Sprintf("s%d.up", s), st.Up, up, nil, false)
			b.release(up)
			b.mark(fmt.Sprintf("layer%02d", idx+1), x)
			idx += 2
		}
	}
	out := b.tconv(kern, "conv_out", &d.ConvOut, x, nil, false)
	b.release(x)
	return b.mark(fmt.Sprintf("layer%02d", idx), out)
}

func (g *TinyGPUDecoder) plan(ar *arena, latentH, latentW int, marks bool) (*builder, tensor, error) {
	b := &builder{e: &g.engine, ar: ar}
	if marks {
		b.marks = map[string]tensor{}
	}
	out := b.buildTiny(g.kern, g.cpu, latentH, latentW)
	return b, out, b.err
}

// planSize records the graph with no pipelines bound, to size the arena.
func (g *TinyGPUDecoder) planSize(latentH, latentW int) (uint32, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	defer func() { g.pipes = saved }()
	b, _, err := g.plan(&arena{}, latentH, latentW, false)
	if err != nil {
		return 0, err
	}
	if int(b.ar.high)*4 > MaxStorageBufferBytes {
		return 0, fmt.Errorf("qvae: a %dx%d tiny decode needs %d MB of activations, past the %d MB single-buffer limit",
			latentH, latentW, (int(b.ar.high)*4)>>20, MaxStorageBufferBytes>>20)
	}
	return b.ar.high, nil
}

// record plans against the real arena and uploads the clamped latent.
func (g *TinyGPUDecoder) record(z *Tensor, marks bool) (*builder, tensor, error) {
	if z.N != 1 || z.C != g.cpu.LatentChannels {
		return nil, tensor{}, fmt.Errorf("qvae: the tiny decoder takes one [1, %d, h, w] latent, got %s", g.cpu.LatentChannels, z)
	}
	g.arena.reset()
	b, out, err := g.plan(&g.arena, z.H, z.W, marks)
	if err != nil {
		return nil, tensor{}, err
	}
	if int(b.ar.high)*4 > g.abuf.Size() {
		return nil, tensor{}, fmt.Errorf("qvae: tiny graph needs %d MB of activations, arena is %d MB",
			(int(b.ar.high)*4)>>20, g.abuf.Size()>>20)
	}
	// The clamp is layer 0 and elementwise over a small tensor, so it runs
	// here; the latent is the arena's first allocation, at offset 0.
	clamped := append([]float32(nil), z.Data...)
	TinyClamp(clamped)
	g.abuf.WriteFloat32(clamped)
	return b, out, nil
}

// Decode maps a normalized latent [1, 64, h, w] to an RGBA image
// [1, 4, 16h, 16w] in [-1, 1].
func (g *TinyGPUDecoder) Decode(ctx context.Context, z *Tensor) (*Tensor, error) {
	b, out, err := g.record(z, false)
	if err != nil {
		return nil, err
	}
	if err := g.runContext(ctx, b.out); err != nil {
		return nil, err
	}
	return TinyToImage(g.read(out))
}

// RunTo re-runs the graph from the start and stops after the named layer
// ("layer02" ... "layer19"), returning what it produced.
func (g *TinyGPUDecoder) RunTo(z *Tensor, stage string) (*Tensor, error) {
	b, _, err := g.record(z, true)
	if err != nil {
		return nil, err
	}
	t, ok := b.marks[stage]
	if !ok {
		return nil, fmt.Errorf("qvae: no tiny stage %q (have %v)", stage, b.order)
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

// Stages names the marked layers in order.
func (g *TinyGPUDecoder) Stages(latentH, latentW int) ([]string, error) {
	saved := g.pipes
	g.pipes = map[string]*vk.ComputePipeline{}
	defer func() { g.pipes = saved }()
	b, _, err := g.plan(&arena{}, latentH, latentW, true)
	if err != nil {
		return nil, err
	}
	return b.order, nil
}
