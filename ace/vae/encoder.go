package vae

// The encoder (MUSIC.md A11a): diffusers' OobleckEncoder, 48 kHz stereo in,
// the posterior's 64 means and 64 scales a latent out. Every task that
// takes audio -- a reference's timbre, a cover's source, a repaint's --
// starts here.
//
// The graph is the decoder's mirrored: conv1 (k7, 2 → 128), five blocks
// (three residual units at dilation 1/3/9, then a Snake and a strided
// convolution of kernel 2s, stride s, padding s/2; s = 2, 4, 4, 6, 10,
// channels 128 → 128 → 256 → 512 → 1024 → 2048), a final Snake and conv2
// (k3, 2048 → 128). It runs on the same A_CONV GEMM, glue passes and
// bordered fp16 arena, and two things are new:
//
//   - **The strided convolution is a GEMM over overlapping rows.** Output
//     frame t reads input frames s·t − s/2 … s·t + 3s/2 − 1: 2s consecutive
//     rows of the channel-last arena, which is one contiguous run of 2s·C
//     values starting s·C further on for every t. So it is the A_CONV
//     kernel with the row stride set to s·C and a tap stride of C: no
//     im2col, no new shader.
//   - **conv1's two channels are padded to 16 and its 7 taps to 8** on the
//     host, as the upload lays the audio out, so a fragment stays inside one
//     tap and K is a whole number of the kernel's K tiles.
//   - **The audio and block 0's activations are carried as two fp16s**, hi
//     and lo (v − hi), against weights duplicated across the pair, so the
//     GEMM sums W·v to ~22 bits of v. The encoder is that sensitive at
//     48 kHz: rounding the audio alone to fp16 costs the posterior mean
//     45.5 dB, block 0's inputs 51 dB, and every later stage 72-87 dB
//     (torch, one site at a time). The audio's pair rides in conv1's
//     padding channels (free); block 0's convolutions read a 2C-wide operand
//     that shaders/ace_vae.comp's SPLIT pass writes (twice the FLOPs of a
//     third of the encoder).
//
// The encoder shares the decoder's activation buffers (abuf, hbuf): the
// two never run at once. Block 0's split operand is twice the decoder's
// widest fp16 stage, so the encoder's window is half the decoder's, and a
// tile's audio (the last one's remainder frames included) must fit it. Its
// margin is 16 latents: the receptive field is ~±7 (block 4's residual
// units reach ±3.9 latents, its strided convolution and conv2 one each).

import (
	"fmt"
	"math"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

// encWidths are the channels before encoder block i, and after the last.
var (
	encStrides = []int{2, 4, 4, 6, 10}
	encWidths  = []int{128, 128, 256, 512, 1024, 2048}
)

const (
	// inPad and inTaps are conv1's padded channel count and tap count.
	inPad, inTaps = 16, 8
	// Posterior is the encoder's output channels: means then scales.
	Posterior = 2 * Latent
	// encMargin is the overlap each side of an encode tile, in latents.
	encMargin = 16
)

type encBlock struct {
	res     [3]resUnit
	s       snake
	down    conv // K = 2s·in (2s·2in split)
	stride  int
	in, out int
	split   bool // its convolutions read hi | lo (block 0)
}

// Encoder is the VAE encoder on the device, borrowing a decoder's
// activation buffers.
type Encoder struct {
	// Between, when set, is called between tiles.
	Between func() error

	k      *GPU // the kernels, bound to this encoder's weights and dec's arenas
	window int  // latents a tile: half the decoder's
	conv1  conv
	conv2  conv
	blocks []encBlock
	final  snake
}

// NewEncoder stages the encoder from dir (the checkpoint's vae/) over dec's
// activation buffers, for tiles of up to dec.Window latents. It must not
// run while dec does.
func NewEncoder(dec *GPU, dir string) (*Encoder, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	k := &GPU{dev: dec.dev, Window: dec.Window, pipes: map[string]*vk.ComputePipeline{},
		abuf: dec.abuf, hbuf: dec.hbuf, aX: dec.aX, aC: dec.aC, aUp: dec.aUp, aOut: dec.aOut, aWav: dec.aWav, hA: dec.hA}
	// Block 0's split operand, (rows + border) · 2·128 fp16, has to fit the
	// decoder's arena, sized for (Window·1920 + 1 + border) · 128.
	e := &Encoder{k: k, window: (dec.Window*Hop + 1 - border) / (2 * Hop)}
	if e.window <= 2*encMargin+1 {
		return nil, fmt.Errorf("vae: a %d-latent decoder window leaves the encoder no core", dec.Window)
	}
	if err := e.stage(set); err != nil {
		e.Destroy()
		return nil, err
	}
	if err := k.build(); err != nil {
		e.Destroy()
		return nil, err
	}
	return e, nil
}

func (e *Encoder) stage(set *safetensors.Set) error {
	l := &loader{set: set}
	var bank []uint16
	var w32 []float32
	put32 := func(v []float32) uint32 {
		off := uint32(len(w32))
		w32 = append(w32, v...)
		return off
	}
	// A convolution [out, in, taps] as B[o, j·inP + c], with the input
	// channels padded to inP and the taps to tapsP (zero weights). dup is
	// where the lo half of a split operand starts (0: not split): the
	// weights are written there again.
	addConv := func(p string, out, in, taps, dil, inP, tapsP, dup int) conv {
		v := l.f32(p+".weight_v", out*in*taps)
		gg := l.f32(p+".weight_g", out)
		c := conv{in: inP, n: roundUp(out, kConv.bn), taps: tapsP, dil: dil}
		if l.err != nil {
			return c
		}
		w := weightNorm(gg, v)
		kk := tapsP * inP
		b := make([]float32, out*kk)
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				for j := 0; j < taps; j++ {
					b[o*kk+j*inP+i] = w[(o*in+i)*taps+j]
					if dup > 0 {
						b[o*kk+j*inP+dup+i] = w[(o*in+i)*taps+j]
					}
				}
			}
		}
		c.bOff = uint32(len(bank))
		buf := make([]uint16, c.n*kk)
		packB(buf, b, c.n, kk)
		bank = append(bank, buf...)
		c.bias = put32(l.f32(p+".bias", out))
		return c
	}
	addSnake := func(p string, ch int) snake {
		a := l.f32(p+".alpha", ch)
		b := l.f32(p+".beta", ch)
		if l.err != nil {
			return 0
		}
		v := make([]float32, 2*ch)
		for i := 0; i < ch; i++ {
			v[i] = float32(math.Exp(float64(a[i])))
			v[ch+i] = 1 / (float32(math.Exp(float64(b[i]))) + 1e-9)
		}
		return snake(put32(v))
	}

	// conv1 reads the audio's hi pair in channels 0-1 and its lo in 2-3.
	e.conv1 = addConv("encoder.conv1", encWidths[0], Channels, 7, 1, inPad, inTaps, Channels)
	for i, s := range encStrides {
		p := fmt.Sprintf("encoder.block.%d.", i)
		b := encBlock{stride: s, in: encWidths[i], out: encWidths[i+1], split: i == 0}
		inP, dup := b.in, 0
		if b.split {
			inP, dup = 2*b.in, b.in
		}
		for r, d := range dilation {
			q := fmt.Sprintf("%sres_unit%d.", p, r+1)
			b.res[r] = resUnit{
				s1: addSnake(q+"snake1", b.in),
				c1: addConv(q+"conv1", b.in, b.in, 7, d, inP, 7, dup),
				s2: addSnake(q+"snake2", b.in),
				c2: addConv(q+"conv2", b.in, b.in, 1, 1, inP, 1, dup),
			}
		}
		b.s = addSnake(p+"snake1", b.in)
		b.down = addConv(p+"conv1", b.out, b.in, 2*s, 1, inP, 2*s, dup)
		e.blocks = append(e.blocks, b)
	}
	last := encWidths[len(encWidths)-1]
	e.final = addSnake("encoder.snake1", last)
	e.conv2 = addConv("encoder.conv2", Posterior, last, 3, 1, last, 3, 0)
	if l.err != nil {
		return l.err
	}
	var err error
	if e.k.bank, err = e.k.dev.NewBuffer(len(bank) * 2); err != nil {
		return err
	}
	e.k.bank.WriteUint16At(0, bank)
	if e.k.wbuf, err = e.k.dev.NewBuffer(len(w32) * 4); err != nil {
		return err
	}
	e.k.wbuf.WriteFloat32At(0, w32)
	return nil
}

// Destroy releases the encoder's own objects (not the decoder's buffers).
func (e *Encoder) Destroy() {
	for _, p := range e.k.pipes {
		p.Destroy()
	}
	for _, m := range e.k.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{e.k.wbuf, e.k.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Bytes is what the encoder adds on the device.
func (e *Encoder) Bytes() int { return e.k.wbuf.Size() + e.k.bank.Size() }

// down records block b's strided convolution over rows of the arena into
// out: [rows/s, b.out] fp32, output frame t reading input rows
// s·t − s/2 … s·t + 3s/2 − 1 (of 2·in values when split).
func (gr *graph) down(kind string, b encBlock, rows int, out uint32) {
	C, s := b.down.in, b.stride
	m := roundUp(rows/s, kConv.bm)
	var pc pushConstants
	pc.InOff = gr.g.hA - uint32(s/2*C)
	pc.OutOff, pc.BOff = out, b.down.bOff
	pc.GemmM, pc.GemmN, pc.GemmK, pc.LDA = uint32(m), uint32(b.down.n), uint32(2*s*C), uint32(s*C)
	pc.Aux0, pc.Aux1, pc.Aux2 = uint32(C), uint32(C), log2(C)
	gr.d = append(gr.d, vk.MultiDispatch{Pipeline: gr.g.pipes["conv"], GroupsX: uint32(b.down.n / kConv.bn),
		GroupsY: uint32(m / kConv.bm), PushConstants: pc.bytes()})
	gr.kinds = append(gr.kinds, kind)
}

// tileGraph records the encode of n audio frames already in the arena,
// leaving the posterior [n/1920, 128] at aX.
func (e *Encoder) tileGraph(n int) *graph {
	g := e.k
	gr := &graph{g: g}
	rows := n
	gr.conv("conv1", e.conv1, rows, g.aC, kConv, "conv")
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, e.conv1.bias
	pc.Tokens, pc.Dim = uint32(rows), uint32(encWidths[0])
	gr.add("copy", "conv1 bias", groups(rows*encWidths[0]), pc)
	for bi, b := range e.blocks {
		snake := gr.snake
		if b.split {
			snake = gr.snakeSplit
		}
		for ri, r := range b.res {
			k := fmt.Sprintf("b%d r%d", bi, ri)
			snake(k+" snake1", g.aX, rows, b.in, r.s1, noW)
			gr.conv(k+" conv1", r.c1, rows, g.aC, kConv, "conv")
			snake(k+" snake2", g.aC, rows, b.in, r.s2, r.c1.bias)
			gr.conv(k+" conv2", r.c2, rows, g.aC, kConv, "conv")
			var pc pushConstants
			pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, r.c2.bias
			pc.Tokens, pc.Dim = uint32(rows), uint32(b.in)
			gr.add("residual", k+" add", groups(rows*b.in), pc)
		}
		snake(fmt.Sprintf("b%d snake", bi), g.aX, rows, b.in, b.s, noW)
		gr.down(fmt.Sprintf("b%d down", bi), b, rows, g.aC)
		rows /= b.stride
		var pc pushConstants
		pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, b.down.bias
		pc.Tokens, pc.Dim = uint32(rows), uint32(b.out)
		gr.add("copy", fmt.Sprintf("b%d down bias", bi), groups(rows*b.out), pc)
	}
	last := encWidths[len(encWidths)-1]
	gr.snake("final snake", g.aX, rows, last, e.final, noW)
	gr.conv("conv2", e.conv2, rows, g.aC, kConv, "conv")
	pc = pushConstants{}
	pc.InOff, pc.OutOff, pc.WOff = g.aC, g.aX, e.conv2.bias
	pc.Tokens, pc.Dim = uint32(rows), Posterior
	gr.add("copy", "conv2 bias", groups(rows*Posterior), pc)
	return gr
}

// snakeSplit is snake writing hi | lo rows of 2C (the SPLIT pass).
func (gr *graph) snakeSplit(kind string, x uint32, rows, ch int, s snake, bias uint32) {
	var pc pushConstants
	pc.InOff, pc.OutOff, pc.WOff, pc.KOff = x, gr.g.hA, uint32(s), bias
	pc.Tokens, pc.Dim, pc.Span, pc.Aux2 = uint32(rows), uint32(ch), border, log2(ch)
	gr.add("snake split", kind, groups((rows+border)*ch), pc)
}

// upload writes audio frames [from, to) of wav (interleaved stereo) into
// the arena as conv1's A operand: 16 channels a frame, the pair's fp16 hi,
// its lo (v − hi), and zeros, with zero rows after them.
func (e *Encoder) upload(wav []float32, from, to int) {
	n := to - from
	buf := make([]uint16, (n+border)*inPad)
	for i := 0; i < n; i++ {
		for c := 0; c < Channels; c++ {
			v := wav[(from+i)*Channels+c]
			hi := safetensors.F32ToF16(v)
			buf[i*inPad+c] = hi
			buf[i*inPad+Channels+c] = safetensors.F32ToF16(v - safetensors.F16ToF32(hi))
		}
	}
	e.k.hbuf.WriteUint16At(int(e.k.hA), buf)
}

// Encode turns interleaved 48 kHz stereo into the posterior's mean and
// scale, [T, 64] each (time-major) with T = frames/1920, in overlap-discard
// tiles of at most the window, and returns the device time. The frames
// past the last whole latent are read, as the unpadded convolutions read
// them; a latent's std is softplus(scale) + 1e-4 (Sample).
func (e *Encoder) Encode(wav []float32) (mean, scale []float32, took time.Duration, err error) {
	if len(wav)%Channels != 0 {
		return nil, nil, 0, fmt.Errorf("vae: %d values is not whole stereo frames", len(wav))
	}
	N := len(wav) / Channels
	T := N / Hop
	if T == 0 {
		return nil, nil, 0, fmt.Errorf("vae: %d frames is less than one latent (%d)", N, Hop)
	}
	W := e.window
	rem := N - T*Hop
	// A window holds at most W·1920 frames, the remainder included.
	fits := func(latents int) bool { return latents*Hop+rem <= W*Hop }
	core := W - 2*encMargin - 1
	untiled := fits(T)
	if untiled {
		core = T
	}
	mean = make([]float32, T*Latent)
	scale = make([]float32, T*Latent)
	for c0 := 0; c0 < T; c0 += core {
		c1 := min(c0+core, T)
		w0, w1 := max(0, c0-encMargin), min(T, c1+encMargin)
		if untiled {
			w0, w1 = 0, T
		}
		if c0 > 0 && e.Between != nil {
			if err := e.Between(); err != nil {
				return nil, nil, took, err
			}
		}
		f1 := w1 * Hop
		if w1 == T {
			f1 = N
		}
		e.upload(wav, w0*Hop, f1)
		gr := e.tileGraph(f1 - w0*Hop)
		t, err := vk.DispatchMultiTimed(gr.d, 1, 1, true)
		if err != nil {
			return nil, nil, took, fmt.Errorf("vae: encode tile at latent %d: %w", c0, err)
		}
		took += t
		post := e.k.abuf.ReadFloat32At(int(e.k.aX)+(c0-w0)*Posterior, (c1-c0)*Posterior)
		for i := 0; i < c1-c0; i++ {
			copy(mean[(c0+i)*Latent:(c0+i+1)*Latent], post[i*Posterior:i*Posterior+Latent])
			copy(scale[(c0+i)*Latent:(c0+i+1)*Latent], post[i*Posterior+Latent:(i+1)*Posterior])
		}
	}
	return mean, scale, took, nil
}

// Sample draws from the posterior as upstream's latent_dist.sample() does:
// mean + (softplus(scale) + 1e-4)·noise, noise one standard normal a value.
func Sample(mean, scale, noise []float32) []float32 {
	out := make([]float32, len(mean))
	for i, m := range mean {
		s := float64(scale[i])
		sp := s
		if s < 20 { // torch's softplus threshold
			sp = math.Log1p(math.Exp(s))
		}
		out[i] = m + float32(sp+1e-4)*noise[i]
	}
	return out
}
