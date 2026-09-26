package vae

// The video VAE's encoder on the device (VIDEO.md M10), which only an fl2va
// keyframe needs: one frame in, its posterior's moments out.
//
// The encoder is a causal 3-D CNN, and at one frame it collapses to 2-D:
// every temporal filter is padded with two zero frames in front and none
// behind, so the only tap that ever meets the frame is the last one, and the
// temporal stride-2 downsamplers map one padded frame of three back to one.
// What runs here is therefore each filter's `[:, :, 2]` slice as a 2-D
// convolution, and it is exactly what diffusers computes, not an
// approximation of it (`_encode` runs a single frame through `_encode_clip`
// alone, with no temporal chunking).
//
// Three things differ from the Qwen-Image encoder qimage/vae runs, and they
// are the whole of the new work:
//
//   - the padding is **reflect**, not zero: vae_conv2d.comp's REFLECT builds;
//   - the norms are **GroupNorm(32)** with an affine, a new kernel
//     (h3_groupnorm.comp), not a per-pixel RMS norm;
//   - the downsampler pads (0, 1, 0, 1) by reflection before a stride-2
//     filter. As with zero padding, that is the stride-1 pad-1 filter read
//     at the odd pixels: the only padded tap a stride-2 output reads is the
//     row (or column) past the end, which reflection maps to H − 2 in both
//     forms. So it is the same stride-1 build plus vae_downsample2x.comp.
//
// The graph is fp32 end to end: 0.18 B parameters, and a 256² tile is
// ~217 GFLOP of direct convolution. diffusers encodes in 256-pixel tiles
// with widened overlaps and blends them, and the tiling changes the result,
// so it is reproduced (the decoder's splitTiles and stitch, in latents).

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// encPC mirrors the PC block in shaders/vae_common.glsl, which every kernel
// this graph runs declares.
type encPC struct {
	InOff, OutOff    uint32
	C, H, W          uint32
	OC, KH, KW, Pad  uint32
	WOff, BOff       uint32
	Groups           uint32
	ResOff           uint32
	Aux0, Aux1, Aux2 uint32
	GemmB, GemmM     uint32
	GemmN, GemmK     uint32
	LDA, LDB         uint32
}

func (p encPC) bytes() []byte {
	out := make([]byte, unsafe.Sizeof(p))
	*(*encPC)(unsafe.Pointer(&out[0])) = p
	return out
}

// EncoderConfig is the encoder's half of vae/config.json.
type EncoderConfig struct {
	BlockOut       []int   `json:"block_out_channels"`
	LayersPerBlock int     `json:"layers_per_block"`
	NormGroups     int     `json:"norm_num_groups"`
	NormEps        float64 `json:"norm_eps"`
	PaddingMode    string  `json:"spatial_padding_mode"`
}

// EncConvKernel names one REFLECT build of vae_conv2d.comp.
type EncConvKernel string

const (
	EncConvOC16 EncConvKernel = "oc16"
	EncConvOC32 EncConvKernel = "oc32"
	EncConvOC48 EncConvKernel = "oc48"
)

var encConvVariants = map[EncConvKernel]struct {
	spirv   []byte
	ocBlock int
}{
	EncConvOC16: {shaders.H3EncConvOC16, 16},
	EncConvOC32: {shaders.H3EncConvOC32, 32},
	EncConvOC48: {shaders.H3EncConvOC48, 48},
}

// DefaultEncConv is the build the encoder runs unless told otherwise.
const DefaultEncConv = EncConvOC32

type encConv struct {
	in, out, k int
	w, b       uint32
}

type encNorm struct {
	c    int
	w, b uint32
}

type encResnet struct {
	n1, n2 encNorm
	c1, c2 encConv
	short  *encConv
}

type encLevel struct {
	resnets []encResnet
	down    *encConv
}

// Encoder is the encoder staged for tiles up to tileH × tileW pixels.
type Encoder struct {
	// Between, when set, is called between two submissions (a server
	// yields the device there).
	Between func() error

	dev          *vk.Device
	cfg          *Config
	ecfg         EncoderConfig
	ratio        int
	tileH, tileW int

	wbuf, abuf *vk.Buffer
	pipes      map[string]*vk.ComputePipeline
	mods       []*vk.ShaderModule
	ocBlock    int

	convIn   encConv
	levels   []encLevel
	normOut  encNorm
	convOut  encConv
	quant    encConv
	slot     uint32 // elements a slot holds
	slots    [5]uint32
	weights  []float32
	graphs   map[[2]int]*encGraph
	lastTook time.Duration
}

// encGraph is one tile shape's recorded dispatches.
type encGraph struct {
	d     []vk.MultiDispatch
	flops []float64
	out   uint32
	marks map[string]encMark
}

// encMark is where a stage's output sits and how many dispatches produce
// it: the slots are reused, so a stage is read by running the graph only
// that far.
type encMark struct {
	off     uint32
	c, h, w int
	n       int
}

// NewEncoder stages the encoder (0.7 GB of fp32 weights) for tiles up to
// tileH × tileW pixels. kernel "" is DefaultEncConv.
func NewEncoder(dev *vk.Device, dir string, tileH, tileW int, kernel EncConvKernel) (*Encoder, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	ecfg, err := loadEncoderConfig(dir)
	if err != nil {
		return nil, err
	}
	if kernel == "" {
		kernel = DefaultEncConv
	}
	variant, ok := encConvVariants[kernel]
	if !ok {
		return nil, fmt.Errorf("vae: no encoder conv build %q", kernel)
	}
	r := cfg.Spatial()
	if tileH%r != 0 || tileW%r != 0 {
		return nil, fmt.Errorf("vae: an encoder tile of %dx%d is not whole latents", tileW, tileH)
	}
	e := &Encoder{dev: dev, cfg: cfg, ecfg: ecfg, ratio: r, tileH: tileH, tileW: tileW,
		pipes: map[string]*vk.ComputePipeline{}, ocBlock: variant.ocBlock, graphs: map[[2]int]*encGraph{}}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	if err := e.load(set); err != nil {
		return nil, err
	}
	if err := e.stage(variant.spirv); err != nil {
		e.Destroy()
		return nil, err
	}
	return e, nil
}

func loadEncoderConfig(dir string) (EncoderConfig, error) {
	var c EncoderConfig
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, fmt.Errorf("vae: %s: %w", dir, err)
	}
	if len(c.BlockOut) == 0 || c.LayersPerBlock == 0 || c.NormGroups == 0 {
		return c, fmt.Errorf("vae: %s has no encoder config", dir)
	}
	if c.PaddingMode != "reflect" {
		return c, fmt.Errorf("vae: spatial_padding_mode %q; this encoder implements reflect", c.PaddingMode)
	}
	for _, ch := range c.BlockOut {
		if ch%c.NormGroups != 0 {
			return c, fmt.Errorf("vae: %d channels do not split into %d groups", ch, c.NormGroups)
		}
	}
	return c, nil
}

// put appends a tensor to the weight arena and returns its offset.
func (e *Encoder) put(v []float32) uint32 {
	off := uint32(len(e.weights))
	e.weights = append(e.weights, v...)
	return off
}

// conv loads a causal Conv3d as the 2-D filter of its last temporal tap.
func (e *Encoder) conv(set *safetensors.Set, name string, in, out, k int) (encConv, error) {
	w, err := f32(set, name+".weight", out*in*k*k*k)
	if err != nil {
		return encConv{}, err
	}
	b, err := f32(set, name+".bias", out)
	if err != nil {
		return encConv{}, err
	}
	taps := k * k
	w2 := make([]float32, out*in*taps)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			src := ((o*in+i)*k + (k - 1)) * taps
			copy(w2[(o*in+i)*taps:(o*in+i+1)*taps], w[src:src+taps])
		}
	}
	return encConv{in: in, out: out, k: k, w: e.put(w2), b: e.put(b)}, nil
}

func (e *Encoder) norm(set *safetensors.Set, name string, c int) (encNorm, error) {
	w, err := f32(set, name+".weight", c)
	if err != nil {
		return encNorm{}, err
	}
	b, err := f32(set, name+".bias", c)
	if err != nil {
		return encNorm{}, err
	}
	return encNorm{c: c, w: e.put(w), b: e.put(b)}, nil
}

func (e *Encoder) load(set *safetensors.Set) error {
	c := e.ecfg
	var err error
	if e.convIn, err = e.conv(set, "encoder.conv_in", 3, c.BlockOut[0], 3); err != nil {
		return err
	}
	in := c.BlockOut[0]
	for i, out := range c.BlockOut {
		var lv encLevel
		for j := 0; j < c.LayersPerBlock; j++ {
			p := fmt.Sprintf("encoder.down_blocks.%d.resnets.%d.", i, j)
			rin := out
			if j == 0 {
				rin = in
			}
			var r encResnet
			if r.n1, err = e.norm(set, p+"norm1", rin); err != nil {
				return err
			}
			if r.c1, err = e.conv(set, p+"conv1", rin, out, 3); err != nil {
				return err
			}
			if r.n2, err = e.norm(set, p+"norm2", out); err != nil {
				return err
			}
			if r.c2, err = e.conv(set, p+"conv2", out, out, 3); err != nil {
				return err
			}
			if rin != out {
				s, err := e.conv(set, p+"conv_shortcut", rin, out, 1)
				if err != nil {
					return err
				}
				r.short = &s
			}
			lv.resnets = append(lv.resnets, r)
		}
		name := fmt.Sprintf("encoder.down_blocks.%d.downsamplers.0.conv", i)
		if set.Has(name + ".weight") {
			if e.cfg.SpatialFactors[i] != 2 {
				return fmt.Errorf("vae: level %d downsamples by %d in space; the encoder implements 2", i, e.cfg.SpatialFactors[i])
			}
			d, err := e.conv(set, name, out, out, 3)
			if err != nil {
				return err
			}
			lv.down = &d
		} else if e.cfg.SpatialFactors[i] != 1 {
			return fmt.Errorf("vae: level %d downsamples by %d but has no downsampler", i, e.cfg.SpatialFactors[i])
		}
		e.levels = append(e.levels, lv)
		in = out
	}
	last := c.BlockOut[len(c.BlockOut)-1]
	if e.normOut, err = e.norm(set, "encoder.norm_out", last); err != nil {
		return err
	}
	moments := 2 * e.cfg.LatentChannels
	if e.convOut, err = e.conv(set, "encoder.conv_out", last, moments, 3); err != nil {
		return err
	}
	// quant_conv is a plain 1x1x1 Conv3d; its one tap is the last.
	if e.quant, err = e.conv(set, "quant_conv", moments, moments, 1); err != nil {
		return err
	}
	return nil
}

// slotElems is the largest tensor the graph holds at the staged tile: a
// level's widest activation at its resolution, including the stride-1
// output a downsampler subsamples.
func (e *Encoder) slotElems() int {
	h, w := e.tileH, e.tileW
	n := 3 * h * w
	for i, ch := range e.ecfg.BlockOut {
		n = max(n, ch*h*w)
		if e.levels[i].down != nil {
			h, w = h/2, w/2
		}
	}
	return n
}

func (e *Encoder) stage(convSPIRV []byte) error {
	var err error
	if e.wbuf, err = e.dev.NewBuffer(len(e.weights) * 4); err != nil {
		return fmt.Errorf("vae: encoder weights (%d MB): %w", len(e.weights)*4>>20, err)
	}
	e.wbuf.WriteFloat32(e.weights)
	e.weights = nil
	e.slot = uint32((e.slotElems() + 63) &^ 63)
	if e.abuf, err = e.dev.NewBuffer(int(e.slot) * len(e.slots) * 4); err != nil {
		return fmt.Errorf("vae: encoder activations: %w", err)
	}
	for i := range e.slots {
		e.slots[i] = uint32(i) * e.slot
	}
	bufs := []*vk.Buffer{e.wbuf, e.abuf}
	pcSize := uint32(unsafe.Sizeof(encPC{}))
	for name, spirv := range map[string][]byte{
		"conv": convSPIRV,
		"gn":   shaders.H3GroupNorm,
		"add":  shaders.VAEAdd,
		"sub":  shaders.VAEDownsample2x,
	} {
		mod, err := e.dev.NewShaderModule(spirv)
		if err != nil {
			return fmt.Errorf("vae: encoder shader %s: %w", name, err)
		}
		e.mods = append(e.mods, mod)
		p, err := e.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pcSize})
		if err != nil {
			return fmt.Errorf("vae: encoder pipeline %s: %w", name, err)
		}
		e.pipes[name] = p
	}
	return nil
}

// Destroy releases every Vulkan object.
func (e *Encoder) Destroy() {
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
	e.pipes, e.mods, e.abuf, e.wbuf = nil, nil, nil, nil
}

// WeightBytes and ActivationBytes are what the encoder holds on the device.
func (e *Encoder) WeightBytes() int     { return e.wbuf.Size() }
func (e *Encoder) ActivationBytes() int { return e.abuf.Size() }

func groups(n, local int) uint32 { return uint32((n + local - 1) / local) }

// graph records one tile's encode: pixels in slot 0, moments out.
//
// Five slots, each the size of the widest activation: the input and
// output of a resnet (a, and its sum), the two norm/conv scratches (b, c),
// and the pixels. A resnet is b = GN(a); c = conv1(b); b = GN(c);
// c = conv2(b); then a = a + c in place, or a = shortcut(a) + c through b.
func (e *Encoder) graph(h, w int) (*encGraph, error) {
	if g, ok := e.graphs[[2]int{h, w}]; ok {
		return g, nil
	}
	if h > e.tileH || w > e.tileW || h%e.ratio != 0 || w%e.ratio != 0 {
		return nil, fmt.Errorf("vae: an encoder tile of %dx%d, staged for %dx%d", w, h, e.tileW, e.tileH)
	}
	key := [2]int{h, w}
	g := &encGraph{marks: map[string]encMark{}}
	eps := math.Float32bits(float32(e.ecfg.NormEps))
	px, a, b, c, d := e.slots[0], e.slots[1], e.slots[2], e.slots[3], e.slots[4]
	conv := func(k encConv, in, out uint32, h, w int) {
		pad := uint32(0)
		if k.k == 3 {
			pad = 1
		}
		g.d = append(g.d, vk.MultiDispatch{Pipeline: e.pipes["conv"],
			GroupsX: groups(h*w, 256), GroupsY: groups(k.out, e.ocBlock),
			PushConstants: encPC{InOff: in, OutOff: out, C: uint32(k.in), H: uint32(h), W: uint32(w),
				OC: uint32(k.out), KH: uint32(k.k), KW: uint32(k.k), Pad: pad, WOff: k.w, BOff: k.b}.bytes()})
		g.flops = append(g.flops, 2*float64(k.in*k.out*k.k*k.k*h*w))
	}
	gn := func(n encNorm, in, out uint32, h, w int, silu bool) {
		s := uint32(0)
		if silu {
			s = 1
		}
		g.d = append(g.d, vk.MultiDispatch{Pipeline: e.pipes["gn"], GroupsX: uint32(e.ecfg.NormGroups), GroupsY: 1,
			PushConstants: encPC{InOff: in, OutOff: out, C: uint32(n.c), H: uint32(h), W: uint32(w),
				Groups: uint32(e.ecfg.NormGroups), WOff: n.w, BOff: n.b, Aux0: s, Aux1: eps}.bytes()})
		g.flops = append(g.flops, float64(8*n.c*h*w))
	}
	add := func(x, y, out uint32, n int) {
		g.d = append(g.d, vk.MultiDispatch{Pipeline: e.pipes["add"], GroupsX: groups(n, 256), GroupsY: 1,
			PushConstants: encPC{InOff: x, ResOff: y, OutOff: out, Aux0: uint32(n)}.bytes()})
		g.flops = append(g.flops, float64(n))
	}
	sub := func(in, out uint32, ch, h, w int) {
		g.d = append(g.d, vk.MultiDispatch{Pipeline: e.pipes["sub"], GroupsX: groups(ch*h*w/4, 256), GroupsY: 1,
			PushConstants: encPC{InOff: in, OutOff: out, C: uint32(ch), H: uint32(h), W: uint32(w)}.bytes()})
		g.flops = append(g.flops, float64(ch*h*w))
	}

	conv(e.convIn, px, a, h, w)
	g.marks["conv_in"] = encMark{a, e.convIn.out, h, w, len(g.d)}
	for i, lv := range e.levels {
		for j, r := range lv.resnets {
			gn(r.n1, a, b, h, w, true)
			conv(r.c1, b, c, h, w)
			gn(r.n2, c, b, h, w, true)
			conv(r.c2, b, c, h, w)
			n := r.c2.out * h * w
			if r.short != nil {
				conv(*r.short, a, b, h, w)
				add(b, c, a, n)
			} else {
				add(a, c, a, n)
			}
			if j == 0 {
				g.marks[fmt.Sprintf("down%d_r0", i)] = encMark{a, r.c2.out, h, w, len(g.d)}
			}
		}
		ch := e.ecfg.BlockOut[i]
		if lv.down != nil {
			conv(*lv.down, a, b, h, w)
			sub(b, a, ch, h, w)
			h, w = h/2, w/2
		}
		g.marks[fmt.Sprintf("down%d", i)] = encMark{a, ch, h, w, len(g.d)}
	}
	gn(e.normOut, a, b, h, w, true)
	g.marks["norm_out"] = encMark{b, e.normOut.c, h, w, len(g.d)}
	conv(e.convOut, b, c, h, w)
	conv(e.quant, c, d, h, w)
	g.out = d
	g.marks["moments"] = encMark{d, e.quant.out, h, w, len(g.d)}
	e.graphs[key] = g
	return g, nil
}

// submit budget, as the decoder's: a submission closes before its
// estimated device time passes 800 ms at a conservative rate.
const encRate = 1.5e12

func (e *Encoder) submit(g *encGraph, n int) (time.Duration, error) {
	var total time.Duration
	for i := 0; i < n; {
		j, est := i, time.Duration(0)
		for j < n {
			c := submitFlat + time.Duration(g.flops[j]/encRate*1e9)
			if j > i && est+c > submitBudget {
				break
			}
			est += c
			j++
		}
		if i > 0 && e.Between != nil {
			if err := e.Between(); err != nil {
				return total, err
			}
		}
		t, err := vk.DispatchMultiTimed(g.d[i:j], 1, 1, true)
		if err != nil {
			return total, fmt.Errorf("vae: encoder dispatch %d-%d: %w", i, j-1, err)
		}
		total += t
		i = j
	}
	return total, nil
}

// EncodeTile runs one tile, [3, 1, h, w] ImageNet-normalised, and returns
// its moments [2·C, 1, h/16, w/16]. stage, when not "", names an
// intermediate to return instead (conv_in, down{i}_r0, down{i}, norm_out).
func (e *Encoder) EncodeTile(x *Tensor, stage string) (*Tensor, error) {
	if x.C != 3 || x.T != 1 {
		return nil, fmt.Errorf("vae: the encoder takes one RGB frame, got %dx%d", x.C, x.T)
	}
	g, err := e.graph(x.H, x.W)
	if err != nil {
		return nil, err
	}
	if stage == "" {
		stage = "moments"
	}
	m, ok := g.marks[stage]
	if !ok {
		return nil, fmt.Errorf("vae: no encoder stage %q", stage)
	}
	e.abuf.WriteFloat32At(int(e.slots[0]), x.Data)
	took, err := e.submit(g, m.n)
	if err != nil {
		return nil, err
	}
	e.lastTook += took
	out := NewTensor(m.c, 1, m.h, m.w)
	copy(out.Data, e.abuf.ReadFloat32At(int(m.off), m.c*m.h*m.w))
	return out, nil
}

// Encode returns the moments of one whole frame, [3, 1, H, W]
// ImageNet-normalised, tiled and blended as `_encode_clip` does, and the
// device time it took.
func (e *Encoder) Encode(x *Tensor) (*Tensor, time.Duration, error) {
	r := e.ratio
	ys, th, yo := splitTiles(x.H, tileSize, tileOverlap, r)
	xs, tw, xo := splitTiles(x.W, tileSize, tileOverlap, r)
	e.lastTook = 0
	var tiles []*Tensor
	for _, y := range ys {
		for _, x0 := range xs {
			m, err := e.EncodeTile(x.Slice(0, 1, y, th, x0, tw), "")
			if err != nil {
				return nil, 0, err
			}
			tiles = append(tiles, m)
		}
	}
	p := &Plan{Height: x.H / r, Width: x.W / r}
	for _, y := range ys {
		p.YStarts = append(p.YStarts, y/r)
	}
	for _, x0 := range xs {
		p.XStarts = append(p.XStarts, x0/r)
	}
	// The blends run in latents: `_encode_clip` divides the pixel overlaps
	// by the compression ratio.
	for _, o := range yo {
		p.YOverlaps = append(p.YOverlaps, o/r)
	}
	for _, o := range xo {
		p.XOverlaps = append(p.XOverlaps, o/r)
	}
	return p.stitch(tiles), e.lastTook, nil
}

// NormalizePixels is the pipeline's input convention for one RGB frame,
// [H·W·3] bytes to [3, 1, H, W]: ((v / 255) − mean) / std, each step in
// float32 as torch computes it.
func NormalizePixels(rgb []byte, h, w int) (*Tensor, error) {
	if len(rgb) != h*w*3 {
		return nil, fmt.Errorf("vae: %d bytes for a %dx%d RGB frame", len(rgb), w, h)
	}
	x := NewTensor(3, 1, h, w)
	for c := 0; c < 3; c++ {
		m, s := PixelMean[c], PixelStd[c]
		plane := x.Data[c*h*w : (c+1)*h*w]
		for i := range plane {
			plane[i] = (float32(rgb[i*3+c])/255 - m) / s
		}
	}
	return x, nil
}

// TileExtent is the largest tile an encode of an h × w frame runs.
func TileExtent(h, w int) (th, tw int) {
	return min(h, tileSize), min(w, tileSize)
}
