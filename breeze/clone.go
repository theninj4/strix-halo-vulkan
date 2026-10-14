package breeze

// The codec's encoder (B8): a reference clip to the 16 codebooks a clone's
// prompt carries. It is HF's MimiModel.encode as qwen_tts drives it --
//
//	SEANet      causal conv k7 1 -> 64; for strides 4, 5, 6, 8: a residual
//	            block (ELU, conv k3 C -> C/2, ELU, conv k1 C/2 -> C, + x) and
//	            ELU, conv k=2s stride s C -> 2C; ELU, conv k3 1024 -> 512
//	            (24 kHz -> 25 Hz)
//	transformer 8 layers at 512 (LayerNorm, 8 heads of 64, rope 1e4, causal,
//	            layer scales, GELU MLP 2048)
//	downsample  causal conv k4 stride 2, replicate padding (-> 12.5 Hz)
//	quantizer   the split RVQ: one semantic codebook, then 15 acoustic ones
//	            residually, nearest by L2, each through its input projection
//
// -- cut to ceil(samples / 1920) frames. It runs on the host in fp32: a voice
// is created once, and the codes are nearest-neighbour choices that fp16
// operands would move. TestCodecEncoder holds them to the oracle exactly.

import (
	"fmt"
	"math"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

const (
	encRate   = 24000
	encFrames = 1920 // samples a 12.5 Hz frame
)

type mConv struct {
	w       []float32 // [out][in][k]
	b       []float32 // or nil
	in, out int
	k, s, d int
	repl    bool // replicate padding (the downsample), else zeros
}

type mLayer struct {
	ln1w, ln1b, ln2w, ln2b []float32
	q, k, v, o             []float32 // [512][512]
	fc1, fc2               []float32 // [2048][512], [512][2048]
	s1, s2                 []float32
}

type rvq struct {
	inProj []float32   // [256][512]
	books  [][]float32 // [2048][256] each
}

// CodecEncoder is the codec's encoder, on the host.
type CodecEncoder struct {
	seanet   []any // mConv, or elu{} between them, or mRes
	layers   []mLayer
	down     mConv
	semantic rvq
	acoustic rvq
}

type elu struct{}

type mRes struct{ c1, c2 mConv }

// LoadCodecEncoder reads dir/audio_tokenizer's encoder.
func LoadCodecEncoder(dir string) (*CodecEncoder, error) {
	f, err := safetensors.Open(filepath.Join(dir, "audio_tokenizer/model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lerr error
	get := func(name string, n int) []float32 {
		if lerr != nil {
			return nil
		}
		t, err := f.Get("encoder." + name)
		if err != nil {
			lerr = err
			return nil
		}
		v, err := t.F32(nil)
		if err == nil && len(v) != n {
			err = fmt.Errorf("breeze: codec encoder %s has %d values, want %d", name, len(v), n)
		}
		lerr = err
		return v
	}
	conv := func(p string, in, out, k, s, d int, bias bool) mConv {
		c := mConv{w: get(p+".conv.weight", out*in*k), in: in, out: out, k: k, s: s, d: d}
		if bias {
			c.b = get(p+".conv.bias", out)
		}
		return c
	}
	e := &CodecEncoder{}
	e.seanet = append(e.seanet, conv("encoder.layers.0", 1, 64, 7, 1, 1, true))
	ch, li := 64, 1
	for _, s := range []int{4, 5, 6, 8} {
		p := fmt.Sprintf("encoder.layers.%d.block.", li)
		e.seanet = append(e.seanet, mRes{conv(p+"1", ch, ch/2, 3, 1, 1, true), conv(p+"3", ch/2, ch, 1, 1, 1, true)})
		e.seanet = append(e.seanet, elu{}, conv(fmt.Sprintf("encoder.layers.%d", li+2), ch, 2*ch, 2*s, s, 1, true))
		ch *= 2
		li += 3
	}
	e.seanet = append(e.seanet, elu{}, conv(fmt.Sprintf("encoder.layers.%d", li+1), ch, 512, 3, 1, 1, true))
	for i := 0; i < 8; i++ {
		p := fmt.Sprintf("encoder_transformer.layers.%d.", i)
		e.layers = append(e.layers, mLayer{
			ln1w: get(p+"input_layernorm.weight", 512), ln1b: get(p+"input_layernorm.bias", 512),
			ln2w: get(p+"post_attention_layernorm.weight", 512), ln2b: get(p+"post_attention_layernorm.bias", 512),
			q: get(p+"self_attn.q_proj.weight", 512*512), k: get(p+"self_attn.k_proj.weight", 512*512),
			v: get(p+"self_attn.v_proj.weight", 512*512), o: get(p+"self_attn.o_proj.weight", 512*512),
			fc1: get(p+"mlp.fc1.weight", 2048*512), fc2: get(p+"mlp.fc2.weight", 512*2048),
			s1: get(p+"self_attn_layer_scale.scale", 512), s2: get(p+"mlp_layer_scale.scale", 512),
		})
	}
	e.down = conv("downsample", 512, 512, 4, 2, 1, false)
	e.down.repl = true
	book := func(p string) []float32 {
		sum, use := get(p+".codebook.embed_sum", 2048*256), get(p+".codebook.cluster_usage", 2048)
		if lerr != nil {
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
	q := "quantizer.semantic_residual_vector_quantizer"
	e.semantic = rvq{inProj: get(q+".input_proj.weight", 256*512), books: [][]float32{book(q + ".layers.0")}}
	q = "quantizer.acoustic_residual_vector_quantizer"
	e.acoustic = rvq{inProj: get(q+".input_proj.weight", 256*512)}
	for k := 0; k < Codebooks-1; k++ {
		e.acoustic.books = append(e.acoustic.books, book(fmt.Sprintf("%s.layers.%d", q, k)))
	}
	if lerr != nil {
		return nil, lerr
	}
	return e, nil
}

// apply is MimiConv1d.forward, causal: x is [in][T]; padding_total = (k-1)d
// + 1 - s on the left and the extra right padding that completes the last
// stride, zeros or (repl) the edge value.
func (c *mConv) apply(x [][]float32) [][]float32 {
	T := len(x[0])
	kEff := (c.k-1)*c.d + 1
	padTotal := kEff - c.s
	nFrames := math.Ceil(float64(T-kEff+padTotal)/float64(c.s)+1) - 1
	extra := int(nFrames)*c.s + kEff - padTotal - T
	L := T + padTotal + extra
	outT := (L-kEff)/c.s + 1
	at := func(i, t int) float32 { // x at padded index t
		t -= padTotal
		switch {
		case t < 0:
			if c.repl {
				return x[i][0]
			}
			return 0
		case t >= T:
			if c.repl {
				return x[i][T-1]
			}
			return 0
		}
		return x[i][t]
	}
	y := make([][]float32, c.out)
	parallel(c.out, func(o int) {
		row := make([]float32, outT)
		if c.b != nil {
			for t := range row {
				row[t] = c.b[o]
			}
		}
		for i := 0; i < c.in; i++ {
			xi := x[i]
			for j := 0; j < c.k; j++ {
				w := c.w[(o*c.in+i)*c.k+j]
				off := j*c.d - padTotal
				for t := range row {
					p := t*c.s + off
					if p >= 0 && p < T {
						row[t] += w * xi[p]
					} else {
						row[t] += w * at(i, p+padTotal)
					}
				}
			}
		}
		y[o] = row
	})
	return y
}

func eluInPlace(x [][]float32) {
	for _, r := range x {
		for t, v := range r {
			if v < 0 {
				r[t] = float32(math.Expm1(float64(v)))
			}
		}
	}
}

// matT is x [T][in] times w^T (w [out][in]): [T][out].
func matT(x [][]float32, w []float32, out int) [][]float32 {
	in := len(x[0])
	y := make([][]float32, len(x))
	parallel(len(x), func(t int) {
		r := make([]float32, out)
		for o := 0; o < out; o++ {
			var s float32
			wr := w[o*in : (o+1)*in]
			for i, v := range x[t] {
				s += v * wr[i]
			}
			r[o] = s
		}
		y[t] = r
	})
	return y
}

func layerNorm(x []float32, w, b []float32) []float32 {
	var mean, v float64
	for _, a := range x {
		mean += float64(a)
	}
	mean /= float64(len(x))
	for _, a := range x {
		d := float64(a) - mean
		v += d * d
	}
	inv := 1 / math.Sqrt(v/float64(len(x))+1e-5)
	out := make([]float32, len(x))
	for i, a := range x {
		out[i] = float32((float64(a)-mean)*inv)*w[i] + b[i]
	}
	return out
}

// transformer is MimiTransformerModel over [T][512] rows, causal over every
// earlier row (create_causal_mask: the config's sliding window is not
// applied on this path, which jfk's 275 rows check).
func (e *CodecEncoder) transformer(x [][]float32) [][]float32 {
	T := len(x)
	const H, hd = 8, 64
	cos, sin := make([][]float64, T), make([][]float64, T)
	for p := 0; p < T; p++ {
		cos[p], sin[p] = make([]float64, hd/2), make([]float64, hd/2)
		for i := 0; i < hd/2; i++ {
			inv := float32(1 / math.Pow(1e4, float64(2*i)/hd))
			a := float64(float32(p) * inv)
			cos[p][i], sin[p][i] = math.Cos(a), math.Sin(a)
		}
	}
	rope := func(v []float32, p int) {
		for h := 0; h < H; h++ {
			r := v[h*hd : (h+1)*hd]
			for i := 0; i < hd/2; i++ {
				a, b := float64(r[i]), float64(r[i+hd/2])
				r[i] = float32(a*cos[p][i] - b*sin[p][i])
				r[i+hd/2] = float32(b*cos[p][i] + a*sin[p][i])
			}
		}
	}
	for _, l := range e.layers {
		n := make([][]float32, T)
		for t := range x {
			n[t] = layerNorm(x[t], l.ln1w, l.ln1b)
		}
		q, k, v := matT(n, l.q, 512), matT(n, l.k, 512), matT(n, l.v, 512)
		for t := 0; t < T; t++ {
			rope(q[t], t)
			rope(k[t], t)
		}
		ctx := make([][]float32, T)
		parallel(T, func(t int) {
			c := make([]float32, 512)
			s := make([]float64, t+1)
			for h := 0; h < H; h++ {
				mx := math.Inf(-1)
				for j := 0; j <= t; j++ {
					var d float64
					for i := 0; i < hd; i++ {
						d += float64(q[t][h*hd+i]) * float64(k[j][h*hd+i])
					}
					s[j] = d / 8
					mx = math.Max(mx, s[j])
				}
				var z float64
				for j := range s {
					s[j] = math.Exp(s[j] - mx)
					z += s[j]
				}
				for i := 0; i < hd; i++ {
					var a float64
					for j := range s {
						a += s[j] * float64(v[j][h*hd+i])
					}
					c[h*hd+i] = float32(a / z)
				}
			}
			ctx[t] = c
		})
		o := matT(ctx, l.o, 512)
		for t := range x {
			for i := range x[t] {
				x[t][i] += l.s1[i] * o[t][i]
			}
			n[t] = layerNorm(x[t], l.ln2w, l.ln2b)
		}
		f := matT(n, l.fc1, 2048)
		for _, r := range f {
			for i, a := range r {
				r[i] = float32(0.5 * float64(a) * (1 + math.Erf(float64(a)/math.Sqrt2)))
			}
		}
		f = matT(f, l.fc2, 512)
		for t := range x {
			for i := range x[t] {
				x[t][i] += l.s2[i] * f[t][i]
			}
		}
	}
	return x
}

// nearest is the codebook row closest to v by L2 (torch.cdist, argmin: the
// first of equals).
func nearest(book, v []float32) int {
	best, bi := math.Inf(1), 0
	for c := 0; c < len(book)/256; c++ {
		row := book[c*256 : (c+1)*256]
		var d float64
		for j, a := range v {
			e := float64(a) - float64(row[j])
			d += e * e
		}
		if d < best {
			best, bi = d, c
		}
	}
	return bi
}

// Stages are the encoder's intermediates, for the gate.
type Stages struct {
	SEANet, Transformer, Downsample [][]float32 // [T][512]
}

// Encode turns 24 kHz mono samples into [frames][16] codes.
func (e *CodecEncoder) Encode(wav []float32, st *Stages) ([][16]int32, error) {
	if len(wav) == 0 {
		return nil, fmt.Errorf("breeze: an empty reference clip")
	}
	x := [][]float32{wav}
	for _, l := range e.seanet {
		switch l := l.(type) {
		case mConv:
			x = l.apply(x)
		case elu:
			eluInPlace(x)
		case mRes:
			h := make([][]float32, len(x))
			for i, r := range x {
				h[i] = append([]float32(nil), r...)
			}
			eluInPlace(h)
			h = l.c1.apply(h)
			eluInPlace(h)
			h = l.c2.apply(h)
			for i := range x {
				for t := range x[i] {
					x[i][t] += h[i][t]
				}
			}
		}
	}
	rows := transpose(x)
	if st != nil {
		st.SEANet = clone2(rows)
	}
	rows = e.transformer(rows)
	if st != nil {
		st.Transformer = clone2(rows)
	}
	d := transpose(e.down.apply(transpose(rows)))
	if st != nil {
		st.Downsample = clone2(d)
	}
	frames := (len(wav) + encFrames - 1) / encFrames
	if frames > len(d) {
		return nil, fmt.Errorf("breeze: %d samples gave %d frames, want %d", len(wav), len(d), frames)
	}
	codes := make([][16]int32, frames)
	parallel(frames, func(t int) {
		sem := matT([][]float32{d[t]}, e.semantic.inProj, 256)[0]
		codes[t][0] = int32(nearest(e.semantic.books[0], sem))
		res := matT([][]float32{d[t]}, e.acoustic.inProj, 256)[0]
		for k, book := range e.acoustic.books {
			c := nearest(book, res)
			codes[t][k+1] = int32(c)
			for j := range res {
				res[j] -= book[c*256+j]
			}
		}
	})
	return codes, nil
}

func transpose(x [][]float32) [][]float32 {
	out := make([][]float32, len(x[0]))
	for t := range out {
		out[t] = make([]float32, len(x))
		for i := range x {
			out[t][i] = x[i][t]
		}
	}
	return out
}

func clone2(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, r := range x {
		out[i] = append([]float32(nil), r...)
	}
	return out
}
