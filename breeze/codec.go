package breeze

// The codec decoder on the device (B5): Qwen3-TTS's 12.5 Hz tokenizer
// decoder (qwen_tts Qwen3TTSTokenizerV2Decoder), 16 codes a frame to 1920
// samples at 24 kHz, streamed.
//
//	quantizer   16 codebooks of 2048 x 256, summed through two output
//	            projections (semantic, acoustic) to 512 -- on the host
//	pre_conv    causal k3 512 -> 1024, then the transformer's input_proj
//	            1024 -> 512: both linear, folded at load into one k3 conv
//	transformer 8 layers at 512 (16 heads of 64, rope 1e4, causal with a
//	            72-frame window, layer scales, SiLU GLU 1024), final norm
//	upsample    2 x (causal ConvTranspose k2 s2 at 1024, ConvNeXt: depthwise
//	            causal k7, LayerNorm, 4096 GELU, gamma); output_proj (512 ->
//	            1024) is folded into the first transposed conv
//	decoder     causal k7 1024 -> 1536; 4 blocks (SnakeBeta, causal
//	            ConvTranspose k2s s = 8, 5, 4, 3 halving the channels, three
//	            residual units at dilation 1, 3, 9); SnakeBeta, k7 96 -> 1,
//	            clamp
//
// Everything is causal, so a clip decodes in chunks of frames with each
// stage's left context carried: the transformer's KV cache (absolute
// positions), each convolution's last (k-1)·d input rows, each depthwise
// conv's last six. A chunked decode is the whole-clip decode's arithmetic,
// row for row, which TestCodecChunks holds to within fp32 summation order.
//
// The convolutions are dit_gemm's A_CONV=2 over a channel-last fp16 operand
// whose left border holds the carried rows (kokoro's machinery, SPEECH.md
// T4); a transposed convolution of kernel 2s and stride s is a 2-tap one with
// an s-times wider output (ace/vae's), and its causal trim is exactly the
// row it leaves for the next chunk. breeze_codec.comp has the rest.

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	// FrameSamples is the samples a frame decodes to; SampleRate their rate.
	FrameSamples = 1920
	SampleRate   = 24000
	// FrameRate is frames a second.
	FrameRate = 12.5

	cdQ      = 512 // quantizer output, transformer width
	cdLatent = 1024
	cdHeads  = 16
	cdHD     = 64
	cdFFN    = 1024
	cdLayers = 8
	cdWindow = 72
	cdEps    = 1e-5
	cdConvNX = 4096
	cdBM     = 64 // dit_gemm_reg64's tile
	cdPad    = 128
	cdBorder = 64 // rows of carried context before an operand's row 0
	cdSpan   = 64 // zero rows after an operand's data: the M tile's reach
)

var (
	cdRates  = []int{8, 5, 4, 3}
	cdWidths = []int{1536, 768, 384, 192, 96}
	cdDils   = []int{1, 3, 9}
)

type cdConv struct {
	b, bias      uint32 // bank (halves), weight arena (or noW)
	cin, cpad    int    // input channels, padded to the operand's row
	taps, dil, n int    // N, padded to the kernel's tile
	out          int    // real output channels
	narrow       bool   // the 32-wide tile
	state        uint32 // fp16 arena: the carried rows, (taps-1)·dil x cpad
}

func (c *cdConv) hist() int { return (c.taps - 1) * c.dil }

type cdSnake uint32

type cdRes struct {
	a1, a2 cdSnake
	c1, c2 cdConv
}

type cdBlock struct {
	snake  cdSnake
	up     cdConv
	upBias uint32
	rate   int
	res    [3]cdRes
}

type cdLayer struct {
	ln1, ln2, sAttn, sMLP  uint32 // weight arena
	qkv, o, gate, up, down uint32 // bank
	kc, vc                 uint32 // fp16 arena, [maxFrames][16][64]
}

type cdNeXt struct {
	up, upBias                 uint32 // bank: the 1-tap transposed conv; its bias
	dw, dwBias                 uint32
	lnW, lnB                   uint32
	pw1, pw1Bias, pw2, pw2Bias uint32
	gamma                      uint32
	state                      uint32 // fp32 arena: the depthwise conv's six carried rows
}

const noW = 0xffffffff

// Codec is the decoder, resident, with one stream's state.
type Codec struct {
	dev                    *vk.Device
	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule

	// Chunk is the most frames a Decode call takes; MaxFrames the stream's
	// length (the transformer's KV cache and rope).
	Chunk, MaxFrames int

	// quantizer tables, host: per codebook [2048][256], and the two output
	// projections [512][256]
	codebooks [Codebooks][]float32
	projFirst []float32
	projRest  []float32

	qin    cdConv
	layers []cdLayer
	norm   uint32
	rope   uint32
	next   [2]cdNeXt
	dec0   cdConv
	blocks []cdBlock
	final  cdSnake
	out    cdConv

	// fp32 arena
	aX, aY, aG, aU, aP, aWav uint32
	// fp16 arena
	hOp, hQ, hCtx, hF uint32
	ldT, ldCtx, ldF   int

	// the stream
	pos     int       // frames decoded
	qHist   []float32 // the last two quantizer rows, the k3 conv's context
	started bool
}

// LoadCodec stages the decoder from dir/audio_tokenizer for chunks of up to
// chunk frames and streams of up to maxFrames.
func LoadCodec(dev *vk.Device, dir string, chunk, maxFrames int) (*Codec, error) {
	if chunk <= 0 || maxFrames < chunk {
		return nil, fmt.Errorf("breeze: a codec for chunks of %d frames, streams of %d", chunk, maxFrames)
	}
	f, err := safetensors.Open(dir + "/audio_tokenizer/model.safetensors")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c := &Codec{dev: dev, pipes: map[string]*vk.ComputePipeline{}, Chunk: chunk, MaxFrames: maxFrames}
	if err := c.stage(f); err != nil {
		c.Destroy()
		return nil, err
	}
	if err := c.build(); err != nil {
		c.Destroy()
		return nil, err
	}
	c.Reset()
	return c, nil
}

type cdLoader struct {
	f   *safetensors.File
	err error
}

func (l *cdLoader) get(name string, n int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.f.Get("decoder." + name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err == nil && len(v) != n {
		err = fmt.Errorf("breeze: codec %s has %d values, want %d", name, len(v), n)
	}
	l.err = err
	return v
}

func (c *Codec) stage(f *safetensors.File) error {
	l := &cdLoader{f: f}
	var w32 []float32
	put := func(v []float32) uint32 {
		off := uint32(len(w32))
		w32 = append(w32, v...)
		for len(w32)%64 != 0 {
			w32 = append(w32, 0)
		}
		return off
	}
	var bank []uint16
	pack := func(b []float32, n, k int) uint32 {
		off := uint32(len(bank))
		buf := make([]uint16, n*k)
		tileB(buf, n, k, func(i int, dst []float32) {
			if (i+1)*k <= len(b) {
				copy(dst, b[i*k:(i+1)*k])
			} else {
				clear(dst)
			}
		})
		bank = append(bank, buf...)
		return off
	}
	halves := 0
	halloc := func(n int) uint32 {
		off := uint32(halves)
		halves += roundUp(n, 64)
		return off
	}
	snake := func(p string, ch int) cdSnake {
		a, b := l.get(p+".alpha", ch), l.get(p+".beta", ch)
		if l.err != nil {
			return 0
		}
		v := make([]float32, 2*ch)
		for i := 0; i < ch; i++ {
			v[i] = float32(math.Exp(float64(a[i])))
			v[ch+i] = 1 / (float32(math.Exp(float64(b[i]))) + 1e-9)
		}
		return cdSnake(put(v))
	}
	// conv lays w ([out][in][taps], Conv1d) out as B[o][j·cpad + c].
	conv := func(w, bias []float32, out, in, taps, dil int) cdConv {
		cv := cdConv{cin: in, cpad: roundUp(in, 64), taps: taps, dil: dil, out: out, bias: noW}
		cv.narrow = out%64 != 0
		tile := 64
		if cv.narrow {
			tile = 32
		}
		cv.n = roundUp(out, tile)
		k := taps * cv.cpad
		b := make([]float32, out*k)
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				for j := 0; j < taps; j++ {
					b[o*k+j*cv.cpad+i] = w[(o*in+i)*taps+j]
				}
			}
		}
		cv.b = pack(b, cv.n, k)
		if bias != nil {
			cv.bias = put(bias)
		}
		if h := cv.hist(); h > 0 {
			cv.state = halloc(h * cv.cpad)
		}
		return cv
	}

	// The quantizer: codebook = embedding_sum / max(cluster_usage, 1e-5).
	cb := func(p string) []float32 {
		sum, use := l.get(p+"._codebook.embedding_sum", 2048*256), l.get(p+"._codebook.cluster_usage", 2048)
		if l.err != nil {
			return nil
		}
		out := make([]float32, len(sum))
		for i := 0; i < 2048; i++ {
			u := max(use[i], 1e-5)
			for j := 0; j < 256; j++ {
				out[i*256+j] = sum[i*256+j] / u
			}
		}
		return out
	}
	c.codebooks[0] = cb("quantizer.rvq_first.vq.layers.0")
	for k := 1; k < Codebooks; k++ {
		c.codebooks[k] = cb(fmt.Sprintf("quantizer.rvq_rest.vq.layers.%d", k-1))
	}
	c.projFirst = l.get("quantizer.rvq_first.output_proj.weight", cdQ*256)
	c.projRest = l.get("quantizer.rvq_rest.output_proj.weight", cdQ*256)

	// pre_conv (k3, 512 -> 1024) then input_proj (1024 -> 512), folded:
	// W[o][c][j] = sum_i P[o][i] C[i][c][j], b = P bc + bp.
	pw, pb := l.get("pre_conv.conv.weight", cdLatent*cdQ*3), l.get("pre_conv.conv.bias", cdLatent)
	iw, ib := l.get("pre_transformer.input_proj.weight", cdQ*cdLatent), l.get("pre_transformer.input_proj.bias", cdQ)
	if l.err != nil {
		return l.err
	}
	fw := make([]float32, cdQ*cdQ*3)
	fb := make([]float32, cdQ)
	parallel(cdQ, func(o int) {
		acc := make([]float64, cdQ*3)
		bsum := float64(ib[o])
		for i := 0; i < cdLatent; i++ {
			p := float64(iw[o*cdLatent+i])
			bsum += p * float64(pb[i])
			row := pw[i*cdQ*3 : (i+1)*cdQ*3]
			for x, v := range row {
				acc[x] += p * float64(v)
			}
		}
		for x, v := range acc {
			fw[o*cdQ*3+x] = float32(v)
		}
		fb[o] = float32(bsum)
	})
	c.qin = conv(fw, fb, cdQ, cdQ, 3, 1)

	// The transformer.
	cos, sin := make([]float32, c.MaxFrames*cdHD/2), make([]float32, c.MaxFrames*cdHD/2)
	for p := 0; p < c.MaxFrames; p++ {
		for i := 0; i < cdHD/2; i++ {
			inv := float32(1 / math.Pow(1e4, float64(2*i)/cdHD))
			a := float64(float32(p) * inv)
			cos[p*cdHD/2+i], sin[p*cdHD/2+i] = float32(math.Cos(a)), float32(math.Sin(a))
		}
	}
	c.rope = put(append(cos, sin...))
	c.layers = make([]cdLayer, cdLayers)
	qw := cdHeads * cdHD
	for i := range c.layers {
		ly := &c.layers[i]
		p := fmt.Sprintf("pre_transformer.layers.%d.", i)
		ly.ln1 = put(l.get(p+"input_layernorm.weight", cdQ))
		ly.ln2 = put(l.get(p+"post_attention_layernorm.weight", cdQ))
		ly.sAttn = put(l.get(p+"self_attn_layer_scale.scale", cdQ))
		ly.sMLP = put(l.get(p+"mlp_layer_scale.scale", cdQ))
		q := l.get(p+"self_attn.q_proj.weight", qw*cdQ)
		k := l.get(p+"self_attn.k_proj.weight", qw*cdQ)
		v := l.get(p+"self_attn.v_proj.weight", qw*cdQ)
		if l.err != nil {
			return l.err
		}
		ly.qkv = pack(append(append(q, k...), v...), 3*qw, cdQ)
		ly.o = pack(l.get(p+"self_attn.o_proj.weight", cdQ*qw), cdQ, qw)
		ly.gate = pack(l.get(p+"mlp.gate_proj.weight", cdFFN*cdQ), cdFFN, cdQ)
		ly.up = pack(l.get(p+"mlp.up_proj.weight", cdFFN*cdQ), cdFFN, cdQ)
		ly.down = pack(l.get(p+"mlp.down_proj.weight", cdQ*cdFFN), cdQ, cdFFN)
		ly.kc = halloc(c.MaxFrames * qw)
		ly.vc = halloc(c.MaxFrames * qw)
	}
	c.norm = put(l.get("pre_transformer.norm.weight", cdQ))

	// The two upsampling stages. A ConvTranspose1d with kernel = stride = 2
	// has no overlap: output frame 2t + r is x[t] W[:, :, r], one GEMM
	// [T][2·1024] read as [2T][1024]. The first folds output_proj in.
	ow, ob := l.get("pre_transformer.output_proj.weight", cdLatent*cdQ), l.get("pre_transformer.output_proj.bias", cdLatent)
	for s := range c.next {
		p := fmt.Sprintf("upsample.%d.", s)
		uw, ub := l.get(p+"0.conv.weight", cdLatent*cdLatent*2), l.get(p+"0.conv.bias", cdLatent)
		if l.err != nil {
			return l.err
		}
		nx := &c.next[s]
		bias := make([]float32, 2*cdLatent)
		if s == 0 {
			// W'[r·1024 + o][j] = sum_i U[i][o][r] O[i][j]; b' = sum_i U[i][o][r] ob[i] + ub[o].
			b := make([]float32, 2*cdLatent*cdQ)
			parallel(2*cdLatent, func(row int) {
				r, o := row/cdLatent, row%cdLatent
				acc := make([]float64, cdQ)
				bs := float64(ub[o])
				for i := 0; i < cdLatent; i++ {
					u := float64(uw[(i*cdLatent+o)*2+r])
					bs += u * float64(ob[i])
					for j, v := range ow[i*cdQ : (i+1)*cdQ] {
						acc[j] += u * float64(v)
					}
				}
				for j, v := range acc {
					b[row*cdQ+j] = float32(v)
				}
				bias[row] = float32(bs)
			})
			nx.up = pack(b, 2*cdLatent, cdQ)
		} else {
			b := make([]float32, 2*cdLatent*cdLatent)
			for r := 0; r < 2; r++ {
				for o := 0; o < cdLatent; o++ {
					row := b[(r*cdLatent+o)*cdLatent:]
					for i := 0; i < cdLatent; i++ {
						row[i] = uw[(i*cdLatent+o)*2+r]
					}
					bias[r*cdLatent+o] = ub[o]
				}
			}
			nx.up = pack(b, 2*cdLatent, cdLatent)
		}
		nx.upBias = put(bias)
		nx.dw = put(l.get(p+"1.dwconv.conv.weight", cdLatent*7))
		nx.dwBias = put(l.get(p+"1.dwconv.conv.bias", cdLatent))
		nx.lnW = put(l.get(p+"1.norm.weight", cdLatent))
		nx.lnB = put(l.get(p+"1.norm.bias", cdLatent))
		nx.pw1 = pack(l.get(p+"1.pwconv1.weight", cdConvNX*cdLatent), cdConvNX, cdLatent)
		nx.pw1Bias = put(l.get(p+"1.pwconv1.bias", cdConvNX))
		nx.pw2 = pack(l.get(p+"1.pwconv2.weight", cdLatent*cdConvNX), cdLatent, cdConvNX)
		nx.pw2Bias = put(l.get(p+"1.pwconv2.bias", cdLatent))
		nx.gamma = put(l.get(p+"1.gamma", cdLatent))
	}

	// The decoder.
	c.dec0 = conv(l.get("decoder.0.conv.weight", cdWidths[0]*cdLatent*7), l.get("decoder.0.conv.bias", cdWidths[0]),
		cdWidths[0], cdLatent, 7, 1)
	for b, s := range cdRates {
		in, out := cdWidths[b], cdWidths[b+1]
		p := fmt.Sprintf("decoder.%d.block.", b+1)
		blk := cdBlock{rate: s, snake: snake(p+"0", in)}
		// ConvTranspose1d [in][out][2s], causal (the last s outputs
		// trimmed): output s·t + r is x[t] W[r] + x[t-1] W[r+s], a 2-tap
		// conv over (t-1, t) with row r·out + o of an s·out-wide output.
		uw := l.get(p+"1.conv.weight", in*out*2*s)
		if l.err != nil {
			return l.err
		}
		w := make([]float32, s*out*in*2)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				for i := 0; i < in; i++ {
					w[((r*out+o)*in+i)*2+0] = uw[(i*out+o)*2*s+r+s]
					w[((r*out+o)*in+i)*2+1] = uw[(i*out+o)*2*s+r]
				}
			}
		}
		blk.up = conv(w, nil, s*out, in, 2, 1)
		blk.upBias = put(l.get(p+"1.conv.bias", out))
		for r, d := range cdDils {
			q := fmt.Sprintf("%s%d.", p, r+2)
			blk.res[r] = cdRes{
				a1: snake(q+"act1", out),
				c1: conv(l.get(q+"conv1.conv.weight", out*out*7), l.get(q+"conv1.conv.bias", out), out, out, 7, d),
				a2: snake(q+"act2", out),
				c2: conv(l.get(q+"conv2.conv.weight", out*out), l.get(q+"conv2.conv.bias", out), out, out, 1, 1),
			}
		}
		c.blocks = append(c.blocks, blk)
	}
	last := cdWidths[len(cdWidths)-1]
	c.final = snake("decoder.5", last)
	c.out = conv(l.get("decoder.6.conv.weight", last*7), l.get("decoder.6.conv.bias", 1), 1, last, 7, 1)
	if l.err != nil {
		return l.err
	}

	// Activations, sized by a full chunk.
	n := c.Chunk
	rows := 4 * n // after the two upsamplings
	maxF := max(roundUp(rows, cdBM)*cdConvNX, roundUp(rows/2, cdBM)*2*cdLatent, roundUp(rows, cdBM)*cdWidths[0])
	for b, s := range cdRates {
		rows *= s
		maxF = max(maxF, roundUp(rows, cdBM)*cdWidths[b+1], roundUp(rows/s, cdBM)*s*cdWidths[b+1])
	}
	maxF = max(maxF, roundUp(rows, cdBM)*32)
	acts := 0
	aalloc := func(k int) uint32 {
		off := uint32(acts)
		acts += roundUp(k, 64)
		return off
	}
	// x carries six rows of history before row 0 for the depthwise conv.
	aalloc(6 * cdLatent)
	c.aX = aalloc(maxF)
	c.aY = aalloc(maxF)
	c.aG = aalloc(roundUp(n, cdBM) * cdFFN)
	c.aU = aalloc(roundUp(n, cdBM) * cdFFN)
	c.aP = aalloc(roundUp(n, cdBM) * 3 * qw)
	c.aWav = aalloc(rows)
	for s := range c.next {
		c.next[s].state = aalloc(6 * cdLatent)
	}
	var err error
	if c.abuf, err = c.dev.NewHostCachedBuffer(acts * 4); err != nil {
		return fmt.Errorf("breeze: codec fp32 arena (%d MB): %w", acts*4>>20, err)
	}
	c.ldT, c.ldCtx, c.ldF = cdQ+cdPad, qw+cdPad, cdConvNX+cdPad
	maxOp := roundUp(n, cdBM) * max(c.ldT, c.ldCtx, c.ldF)
	rows = 4 * n
	maxOp = max(maxOp, (roundUp(rows, cdBM)+cdSpan)*c.ldF)
	for b, s := range cdRates {
		maxOp = max(maxOp, (rows+cdSpan)*roundUp(cdWidths[b], 64))
		rows *= s
		maxOp = max(maxOp, (rows+cdSpan)*roundUp(cdWidths[b+1], 64))
	}
	// The operand: cdBorder rows of carried context at the widest row before
	// row 0 (a k7 tap at dilation 9 reaches 54 rows back).
	lead := cdBorder * roundUp(cdWidths[0], 64)
	c.hOp = halloc(lead+maxOp) + uint32(lead)
	c.hQ = halloc(roundUp(n, cdBM) * qw)
	c.hCtx = halloc(roundUp(n, cdBM) * c.ldCtx)
	c.hF = halloc(roundUp(n, cdBM) * c.ldT)
	if halves*2 > maxBankBytes || acts*4 > maxBankBytes {
		return fmt.Errorf("breeze: a %d-frame codec chunk is past one binding", n)
	}
	if c.hbuf, err = c.dev.NewBuffer(halves * 2); err != nil {
		return err
	}
	c.hbuf.Zero()
	if c.bank, err = c.dev.NewBuffer(len(bank) * 2); err != nil {
		return err
	}
	c.bank.WriteUint16At(0, bank)
	if c.wbuf, err = c.dev.NewBuffer(len(w32) * 4); err != nil {
		return err
	}
	c.wbuf.WriteFloat32At(0, w32)
	return nil
}

func (c *Codec) build() error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{c.wbuf, c.abuf, c.hbuf}
	withBank := []*vk.Buffer{c.wbuf, c.abuf, c.hbuf, c.bank}
	for _, p := range []struct {
		name  string
		spirv []byte
		bufs  []*vk.Buffer
		wave  uint32
	}{
		{"narrow", shaders.BreezeCodecNarrow, base, 0},
		{"bias", shaders.BreezeCodecBias, base, 0},
		{"ln", shaders.BreezeCodecLN, base, 0},
		{"gelu", shaders.BreezeCodecGELU, base, 0},
		{"resid", shaders.BreezeCodecResid, base, 0},
		{"dwconv", shaders.BreezeCodecDWConv, base, 0},
		{"copy32", shaders.BreezeCodecCopy32, base, 0},
		{"copy16", shaders.BreezeCodecCopy16, base, 0},
		{"out", shaders.BreezeCodecOut, base, 0},
		{"prep", shaders.BreezeCodecPrep, base, 0},
		{"attn", shaders.BreezeCodecAttn, base, 0},
		{"normf16", shaders.DiTNormScaleF16, base, 0},
		{"swiglu", shaders.DiTSwiGLUF16, base, 0},
		{"gemm", shaders.DiTGEMMReg64Tiled, withBank, 0},
		{"conv", shaders.KokoroDConvReg32x64W32, withBank, 32},
		{"conv narrow", shaders.KokoroDConvReg32x32W32, withBank, 32},
	} {
		mod, err := c.dev.NewShaderModule(p.spirv)
		if err != nil {
			return fmt.Errorf("breeze: codec shader %s: %w", p.name, err)
		}
		c.mods = append(c.mods, mod)
		pipe, err := c.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: p.bufs, PushConstantSize: pc, RequiredSubgroupSize: p.wave})
		if err != nil {
			return fmt.Errorf("breeze: codec pipeline %s: %w", p.name, err)
		}
		c.pipes[p.name] = pipe
	}
	return nil
}

// Destroy releases the codec's device objects.
func (c *Codec) Destroy() {
	for _, p := range c.pipes {
		p.Destroy()
	}
	for _, m := range c.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{c.wbuf, c.abuf, c.hbuf, c.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Reset starts a new stream: every carried state back to the zero padding.
func (c *Codec) Reset() {
	c.pos = 0
	c.qHist = make([]float32, 2*cdQ)
	c.started = false
}

// quantize is the quantizer's decode of a frame: [512].
func (c *Codec) quantize(dst []float32, f [16]int32) {
	var first, rest [256]float32
	copy(first[:], c.codebooks[0][int(f[0])*256:])
	for k := 1; k < Codebooks; k++ {
		e := c.codebooks[k][int(f[k])*256:]
		for j := range rest {
			rest[j] += e[j]
		}
	}
	for o := 0; o < cdQ; o++ {
		var s float32
		pf, pr := c.projFirst[o*256:], c.projRest[o*256:]
		for j := 0; j < 256; j++ {
			s += pf[j]*first[j] + pr[j]*rest[j]
		}
		dst[o] = s
	}
}

// Decode runs the next frames of the stream (at most Chunk) and returns
// their 1920 samples each.
func (c *Codec) Decode(frames [][16]int32) ([]float32, time.Duration, error) {
	n := len(frames)
	if n == 0 {
		return nil, 0, nil
	}
	if n > c.Chunk {
		return nil, 0, fmt.Errorf("breeze: %d frames in one codec chunk; at most %d", n, c.Chunk)
	}
	if c.pos+n > c.MaxFrames {
		return nil, 0, fmt.Errorf("breeze: the codec stream is %d frames at most", c.MaxFrames)
	}
	for _, f := range frames {
		for _, v := range f {
			if v < 0 || v >= Codes {
				return nil, 0, fmt.Errorf("breeze: code %d outside the codebook", v)
			}
		}
	}
	// The k3 conv's operand, with its two carried rows, straight from the
	// host: [2 + n][512] at the operand's row -2.
	q := make([]float32, (2+n)*cdQ)
	copy(q, c.qHist)
	parallel(n, func(i int) { c.quantize(q[(2+i)*cdQ:(3+i)*cdQ], frames[i]) })
	copy(c.qHist, q[n*cdQ:])
	op := make([]uint16, (2+n+cdSpan)*c.qin.cpad)
	for r := 0; r < 2+n; r++ {
		for j := 0; j < cdQ; j++ {
			op[r*c.qin.cpad+j] = safetensors.F32ToF16(q[r*cdQ+j])
		}
	}
	c.hbuf.WriteUint16At(int(c.hOp)-2*c.qin.cpad, op)
	if !c.started {
		for s := range c.next {
			c.abuf.ZeroFloat32At(int(c.next[s].state), 6*cdLatent)
		}
		c.zeroConvStates()
		c.started = true
	}

	g := c.graph(n)
	took, err := vk.DispatchMultiTimed(g, 1, 1, true)
	if err != nil {
		return nil, took, err
	}
	c.pos += n
	return c.abuf.ReadFloat32At(int(c.aWav), n*FrameSamples), took, nil
}

func (c *Codec) zeroConvStates() {
	z := func(cv *cdConv) {
		if h := cv.hist(); h > 0 {
			c.hbuf.ZeroUint16At(int(cv.state), h*cv.cpad)
		}
	}
	z(&c.dec0)
	z(&c.out)
	for b := range c.blocks {
		z(&c.blocks[b].up)
		for r := range c.blocks[b].res {
			z(&c.blocks[b].res[r].c1)
		}
	}
}

// parallel runs fn(0..n-1) over the cores.
func parallel(n int, fn func(i int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(w)
	for ; w > 0; w-- {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

func (c *Codec) graph(n int) []vk.MultiDispatch {
	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		d = append(d, vk.MultiDispatch{Pipeline: c.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	flat := func(pipe string, count int, p push) { add(pipe, groups(count, 256), 1, p) }
	gemm := func(b, aOff, cOff uint32, m, nOut, k, lda int) {
		mPad := roundUp(m, cdBM)
		add("gemm", uint32(nOut/cdBM), uint32(mPad/cdBM), push{InOff: aOff, OutOff: cOff, BOff: b,
			GemmM: uint32(mPad), GemmN: uint32(nOut), GemmK: uint32(k), LDA: uint32(lda)})
	}
	// conv runs cv over rows already at the operand (row 0 at hOp, stride
	// cpad), with its carried rows swapped in before and out after.
	conv := func(cv *cdConv, rows int, out uint32) {
		h := cv.hist()
		carry := cv != &c.qin // the k3 conv's context is the host's to write
		if h > 0 && carry {
			flat("copy16", h*cv.cpad, push{InOff: cv.state, OutOff: c.hOp - uint32(h*cv.cpad), Tokens: uint32(h * cv.cpad)})
		}
		pipe, bm, bn := "conv", 32, 64
		if cv.narrow {
			pipe, bn = "conv narrow", 32
		}
		m := roundUp(rows, bm)
		add(pipe, uint32(cv.n/bn), uint32(m/bm), push{InOff: c.hOp - uint32(h*cv.cpad), OutOff: out, BOff: cv.b,
			GemmM: uint32(m), GemmN: uint32(cv.n), GemmK: uint32(cv.taps * cv.cpad), LDA: uint32(cv.cpad),
			Aux0: uint32(cv.cpad), Aux1: uint32(cv.dil * cv.cpad)})
		if h > 0 && carry {
			flat("copy16", h*cv.cpad, push{InOff: c.hOp + uint32((rows-h)*cv.cpad), OutOff: cv.state, Tokens: uint32(h * cv.cpad)})
		}
	}
	narrow := func(in uint32, rows, ch, lda int, bias uint32, s cdSnake, snake bool) {
		w := uint32(noW)
		if snake {
			w = uint32(s)
		}
		flat("narrow", (rows+cdSpan)*lda, push{InOff: in, OutOff: c.hOp, KOff: bias, WOff: w,
			Tokens: uint32(rows), Span: cdSpan, Dim: uint32(ch), LDA: uint32(lda)})
	}
	bias := func(x uint32, rows, ch int, b uint32) {
		flat("bias", rows*ch, push{OutOff: x, KOff: b, Tokens: uint32(rows), Dim: uint32(ch)})
	}
	resid := func(x, y uint32, rows, ch int, g, b uint32) {
		flat("resid", rows*ch, push{OutOff: x, InOff: y, WOff: g, KOff: b, Tokens: uint32(rows), Dim: uint32(ch)})
	}

	// pre_conv + input_proj, folded: the residual stream of the transformer.
	conv(&c.qin, n, c.aX)
	bias(c.aX, n, cdQ, c.qin.bias)

	eps := math.Float32bits(cdEps)
	qw := cdHeads * cdHD
	norm := func(w uint32) {
		add("normf16", uint32(n), 1, push{InOff: c.aX, OutOff: c.hF, WOff: w, Tokens: uint32(n), Dim: cdQ,
			LDA: uint32(c.ldT), Eps: eps})
	}
	for _, ly := range c.layers {
		norm(ly.ln1)
		gemm(ly.qkv, c.hF, c.aP, n, 3*qw, cdQ, c.ldT)
		add("prep", uint32(n), 3*cdHeads, push{InOff: c.aP, OutOff: c.hQ, KOff: ly.kc, VOff: ly.vc, WOff: c.rope,
			KStride: uint32(c.MaxFrames), Aux0: uint32(c.pos), Tokens: uint32(n),
			Scale: math.Float32bits(float32(1 / math.Sqrt(cdHD)))})
		add("attn", uint32(n), cdHeads, push{InOff: c.hQ, KOff: ly.kc, VOff: ly.vc, OutOff: c.hCtx,
			LDA: uint32(c.ldCtx), Aux0: uint32(c.pos), Span: cdWindow, Tokens: uint32(n)})
		gemm(ly.o, c.hCtx, c.aY, n, cdQ, qw, c.ldCtx)
		resid(c.aX, c.aY, n, cdQ, ly.sAttn, noW)
		norm(ly.ln2)
		gemm(ly.gate, c.hF, c.aG, n, cdFFN, cdQ, c.ldT)
		gemm(ly.up, c.hF, c.aU, n, cdFFN, cdQ, c.ldT)
		add("swiglu", uint32(n), 1, push{InOff: c.aG, KOff: c.aU, OutOff: c.hCtx, Tokens: uint32(n), Dim: cdFFN,
			LDA: uint32(c.ldCtx), Scale: math.Float32bits(1)})
		gemm(ly.down, c.hCtx, c.aY, n, cdQ, cdFFN, c.ldCtx)
		resid(c.aX, c.aY, n, cdQ, ly.sMLP, noW)
	}
	norm(c.norm)

	// The upsampling: x is [rows][1024] at aX after each stage.
	rows := n
	for s := range c.next {
		nx := &c.next[s]
		src, k, lda := c.hF, cdQ, c.ldT
		if s > 0 {
			narrow(c.aX, rows, cdLatent, cdLatent, noW, 0, false)
			src, k, lda = c.hOp, cdLatent, cdLatent
		}
		// [rows][2048] is [2·rows][1024]: x, after its six carried rows.
		gemm(nx.up, src, c.aX, rows, 2*cdLatent, k, lda)
		rows *= 2
		bias(c.aX, rows/2, 2*cdLatent, nx.upBias)
		flat("copy32", 6*cdLatent, push{InOff: nx.state, OutOff: c.aX - 6*cdLatent, Tokens: 6 * cdLatent})
		flat("dwconv", rows*cdLatent, push{InOff: c.aX, OutOff: c.aY, WOff: nx.dw, KOff: nx.dwBias,
			Tokens: uint32(rows), Dim: cdLatent})
		flat("copy32", 6*cdLatent, push{InOff: c.aX + uint32((rows-6)*cdLatent), OutOff: nx.state, Tokens: 6 * cdLatent})
		add("ln", uint32(rows), 1, push{InOff: c.aY, OutOff: c.hOp, WOff: nx.lnW, KOff: nx.lnB, Tokens: uint32(rows),
			Dim: cdLatent, LDA: cdLatent})
		gemm(nx.pw1, c.hOp, c.aY, rows, cdConvNX, cdLatent, cdLatent)
		flat("gelu", rows*cdConvNX, push{InOff: c.aY, OutOff: c.hOp, KOff: nx.pw1Bias, Tokens: uint32(rows),
			Dim: cdConvNX, LDA: uint32(c.ldF)})
		gemm(nx.pw2, c.hOp, c.aY, rows, cdLatent, cdConvNX, c.ldF)
		resid(c.aX, c.aY, rows, cdLatent, nx.gamma, nx.pw2Bias)
	}

	// The decoder.
	narrow(c.aX, rows, cdLatent, c.dec0.cpad, noW, 0, false)
	conv(&c.dec0, rows, c.aX)
	bias(c.aX, rows, cdWidths[0], c.dec0.bias)
	for b := range c.blocks {
		blk := &c.blocks[b]
		in, out := cdWidths[b], cdWidths[b+1]
		narrow(c.aX, rows, in, blk.up.cpad, noW, blk.snake, true)
		conv(&blk.up, rows, c.aX)
		rows *= blk.rate
		bias(c.aX, rows, out, blk.upBias)
		for r := range blk.res {
			ru := &blk.res[r]
			narrow(c.aX, rows, out, ru.c1.cpad, noW, ru.a1, true)
			conv(&ru.c1, rows, c.aY)
			narrow(c.aY, rows, out, ru.c2.cpad, ru.c1.bias, ru.a2, true)
			conv(&ru.c2, rows, c.aY)
			resid(c.aX, c.aY, rows, out, noW, ru.c2.bias)
		}
	}
	last := cdWidths[len(cdWidths)-1]
	narrow(c.aX, rows, last, c.out.cpad, noW, c.final, true)
	conv(&c.out, rows, c.aY)
	flat("out", rows, push{InOff: c.aY, OutOff: c.aWav, KOff: c.out.bias, LDA: uint32(c.out.n), Tokens: uint32(rows)})
	return d
}
