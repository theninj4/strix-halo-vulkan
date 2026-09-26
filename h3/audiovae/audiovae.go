// Package audiovae is MiniMax-H3's audio VAE decoder (VIDEO.md M6): a
// BigVGAN that turns 32-channel latents at 40 a second into a 32 kHz
// waveform, one mono codec applied to the left and right channels in turn.
//
// It runs on the CPU in fp32. The whole decode of a 5 s stereo clip is
// ~480 GFLOP of 1-D convolutions — 1% of a served request's device time —
// and diffusers keeps this model in fp32 on purpose (bf16 decodes come out
// ~20 dB quieter), so it is the one piece of the vertical that neither
// needed the device nor fp16 to start with. Activations are channel-last,
// [T, C], so every convolution is a set of dot products over contiguous
// channel vectors and every per-channel operation a pass over contiguous
// rows.
//
// The graph (`MiniMaxH3AudioBigVGANDecoder`): dec_in_proj (1x1, 32 → 2048),
// conv_pre (k7, → 1024), then seven stages of a transposed-conv upsampler
// (×5, ×5, ×2 ×5; channels halving to 8) and three AMP blocks (kernels 3, 7,
// 11) averaged; then an alias-free SnakeBeta, conv_post (k7, → 1) and a
// clamp to [-1, 1]. An AMP block is three (dilated conv, conv) pairs, each
// conv preceded by its own alias-free SnakeBeta: upsample ×2 through a
// 12-tap Kaiser-sinc, SnakeBeta, and the matching low-pass down ×2 — with
// replicate padding at both ends, which is the one thing about it that is
// easy to get plausibly wrong.
package audiovae

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// Config is audio_vae/config.json, the decoder's half.
type Config struct {
	LatentChannels int       `json:"latent_channels"`
	LatentDim      int       `json:"latent_dim"`
	DecoderDim     int       `json:"decoder_dim"`
	Rates          []int     `json:"decoder_rates"`
	Kernels        []int     `json:"decoder_kernel_sizes"`
	ResKernels     []int     `json:"resblock_kernel_sizes"`
	ResDilations   [][]int   `json:"resblock_dilation_sizes"`
	SamplingRate   int       `json:"sampling_rate"`
	LatentsMean    []float32 `json:"latents_mean"`
	LatentsStd     []float32 `json:"latents_std"`
}

func LoadConfig(dir string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	c := &Config{}
	if err := json.Unmarshal(raw, c); err != nil {
		return nil, fmt.Errorf("audiovae: %s: %w", dir, err)
	}
	if len(c.Rates) != len(c.Kernels) || len(c.ResKernels) != len(c.ResDilations) || len(c.LatentsMean) != c.LatentChannels {
		return nil, fmt.Errorf("audiovae: %s is not a MiniMax-H3 audio VAE config", dir)
	}
	return c, nil
}

// Hop is how many samples one latent decodes to: 800.
func (c *Config) Hop() int {
	h := 1
	for _, r := range c.Rates {
		h *= r
	}
	return h
}

// Mat is a channel-last activation, [Rows = T, Cols = C].
type Mat struct {
	Rows, Cols int
	Data       []float32
}

func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Conv is a Conv1d with weight_norm folded, repacked [K][Out][In] so each
// tap is a set of contiguous dot products against a channel-last row.
type Conv struct {
	In, Out, K, Dilation, Pad int
	W                         []float32 // [K][Out][In]
	B                         []float32 // nil when the checkpoint has none
}

// Apply is a stride-1 convolution with zero padding, output length
// T + 2·Pad − Dilation·(K − 1).
func (c *Conv) Apply(x *Mat) *Mat {
	T := x.Rows + 2*c.Pad - c.Dilation*(c.K-1)
	out := NewMat(T, c.Out)
	zero := make([]float32, c.In)
	row := func(t int) []float32 {
		if t < 0 || t >= x.Rows {
			return zero
		}
		return x.Row(t)
	}
	// Tiles of 4 output frames × 4 output channels, 16 accumulators: each
	// step of the reduction loads four inputs and four weights for sixteen
	// multiply-adds.
	const bt = 32
	parallelFor((T+bt-1)/bt, func(blk int) {
		t0, t1 := blk*bt, min((blk+1)*bt, T)
		for t := t0; t < t1; t += 4 {
			nt := min(4, t1-t)
			for o := 0; o < c.Out; o += 4 {
				no := min(4, c.Out-o)
				var acc [4][4]float32
				for k := 0; k < c.K; k++ {
					base := t - c.Pad + k*c.Dilation
					var xs [4][]float32
					for i := 0; i < 4; i++ {
						if i < nt {
							xs[i] = row(base + i)
						} else {
							xs[i] = zero
						}
					}
					var ws [4][]float32
					for j := 0; j < 4; j++ {
						oo := o + min(j, no-1)
						ws[j] = c.W[(k*c.Out+oo)*c.In : (k*c.Out+oo+1)*c.In]
					}
					dot4x4(&acc, xs, ws, c.In)
				}
				for i := 0; i < nt; i++ {
					dst := out.Row(t + i)
					for j := 0; j < no; j++ {
						v := acc[i][j]
						if c.B != nil {
							v += c.B[o+j]
						}
						dst[o+j] = v
					}
				}
			}
		}
	})
	return out
}

// dot4x4 accumulates acc[i][j] += x_i · w_j over their length, in sixteen
// scalars: Go keeps a local array in memory, not in registers.
func dot4x4(acc *[4][4]float32, xs [4][]float32, ws [4][]float32, n int) {
	x0, x1, x2, x3 := xs[0][:n], xs[1][:n], xs[2][:n], xs[3][:n]
	w0, w1, w2, w3 := ws[0][:n], ws[1][:n], ws[2][:n], ws[3][:n]
	var a00, a01, a02, a03, a10, a11, a12, a13 float32
	var a20, a21, a22, a23, a30, a31, a32, a33 float32
	for i := range x0 {
		v0, v1, v2, v3 := x0[i], x1[i], x2[i], x3[i]
		u0, u1, u2, u3 := w0[i], w1[i], w2[i], w3[i]
		a00 += v0 * u0
		a01 += v0 * u1
		a02 += v0 * u2
		a03 += v0 * u3
		a10 += v1 * u0
		a11 += v1 * u1
		a12 += v1 * u2
		a13 += v1 * u3
		a20 += v2 * u0
		a21 += v2 * u1
		a22 += v2 * u2
		a23 += v2 * u3
		a30 += v3 * u0
		a31 += v3 * u1
		a32 += v3 * u2
		a33 += v3 * u3
	}
	acc[0][0] += a00
	acc[0][1] += a01
	acc[0][2] += a02
	acc[0][3] += a03
	acc[1][0] += a10
	acc[1][1] += a11
	acc[1][2] += a12
	acc[1][3] += a13
	acc[2][0] += a20
	acc[2][1] += a21
	acc[2][2] += a22
	acc[2][3] += a23
	acc[3][0] += a30
	acc[3][1] += a31
	acc[3][2] += a32
	acc[3][3] += a33
}

// Up is a ConvTranspose1d upsampler, weight_norm folded, repacked
// [K][Out][In].
type Up struct {
	In, Out, K, Stride, Pad int
	W                       []float32
	B                       []float32
}

// Apply gathers rather than scatters: output frame j = t·stride + k − pad
// takes input frame t through tap k, so each output sums the (at most
// ⌈K/stride⌉) taps of its phase.
func (u *Up) Apply(x *Mat) *Mat {
	T := (x.Rows-1)*u.Stride - 2*u.Pad + u.K
	out := NewMat(T, u.Out)
	parallelFor(T, func(j int) {
		dst := out.Row(j)
		copy(dst, u.B)
		for k := 0; k < u.K; k++ {
			n := j + u.Pad - k
			if n < 0 || n%u.Stride != 0 || n/u.Stride >= x.Rows {
				continue
			}
			src := x.Row(n / u.Stride)
			for o := 0; o < u.Out; o++ {
				w := u.W[(k*u.Out+o)*u.In : (k*u.Out+o+1)*u.In]
				var s float32
				for i, v := range src {
					s += w[i] * v
				}
				dst[o] += s
			}
		}
	})
	return out
}

// Act is the alias-free SnakeBeta (`MiniMaxH3AudioActivation1d`).
type Act struct {
	A, IB    []float32 // exp(alpha), 1/(exp(beta) + 1e-9), per channel
	Up, Down []float32 // the 12-tap Kaiser-sinc filters, as stored
}

const (
	actRatio  = 2
	actKernel = 12
)

// Apply is upsample ×2 → SnakeBeta → low-pass ×2, each with the replicate
// padding diffusers uses.
//
// Up: the input is replicate-padded by pad = K/r − 1 = 5 a side, transposed-
// convolved at stride 2 (length 2(T+10−1)+12), scaled by 2 and cropped by
// 15 a side: sample j of the ×2 signal is 2·Σ f_up[m − 2u]·x[clamp(u − 5)]
// over m = j + 15. Down: the snake'd signal is replicate-padded 5 left and
// 6 right and convolved at stride 2: out[n] = Σ_k f_down[k]·s[clamp(2n + k
// − 5)].
func (a *Act) Apply(x *Mat) *Mat {
	T, C := x.Rows, x.Cols
	const pad = actKernel/actRatio - 1                 // 5
	const padL = pad*actRatio + (actKernel-actRatio)/2 // 15
	const downL = actKernel/2 - 1                      // 5
	s := NewMat(actRatio*T, C)
	parallelFor((s.Rows+63)/64, func(blk int) {
		for j := blk * 64; j < min((blk+1)*64, s.Rows); j++ {
			dst := s.Row(j)
			m := j + padL
			// u runs over the padded input's frames with 0 ≤ m − 2u < K.
			for u := max(0, (m-actKernel+2)/2); u <= m/2; u++ {
				k := m - 2*u
				if k < 0 || k >= actKernel {
					continue
				}
				f := a.Up[k]
				src := x.Row(min(max(u-pad, 0), T-1))
				for c, v := range src {
					dst[c] += f * v
				}
			}
			for c := range dst {
				v := float32(actRatio) * dst[c]
				sn := float32(math.Sin(float64(float32(a.A[c] * v))))
				dst[c] = v + a.IB[c]*float32(sn*sn)
			}
		}
	})
	out := NewMat(T, C)
	parallelFor((T+63)/64, func(blk int) {
		for n := blk * 64; n < min((blk+1)*64, T); n++ {
			dst := out.Row(n)
			for k := 0; k < actKernel; k++ {
				f := a.Down[k]
				src := s.Row(min(max(actRatio*n+k-downL, 0), s.Rows-1))
				for c, v := range src {
					dst[c] += f * v
				}
			}
		}
	})
	return out
}

// AMP is one anti-aliased multi-periodicity block: three (dilated conv,
// conv) pairs, each conv behind its own Act, each pair a residual.
type AMP struct {
	Convs1, Convs2 []*Conv
	Acts           []*Act // act1, act2 of pair 0, then of pair 1, …
}

func (b *AMP) Apply(x *Mat) *Mat {
	for p := range b.Convs1 {
		r := b.Convs1[p].Apply(b.Acts[2*p].Apply(x))
		r = b.Convs2[p].Apply(b.Acts[2*p+1].Apply(r))
		y := NewMat(x.Rows, x.Cols)
		for i, v := range r.Data {
			y.Data[i] = v + x.Data[i]
		}
		x = y
	}
	return x
}

// Decoder is the whole decode, fp32.
type Decoder struct {
	Cfg     *Config
	InProj  *Conv // dec_in_proj, 1x1
	Pre     *Conv
	Ups     []*Up
	Blocks  []*AMP // len(Ups) × len(ResKernels)
	PostAct *Act
	Post    *Conv
}

// Stage runs upsampler i and its averaged AMP blocks.
func (d *Decoder) Stage(i int, x *Mat) *Mat {
	h := d.Ups[i].Apply(x)
	nk := len(d.Cfg.ResKernels)
	outs := make([]*Mat, nk)
	for j := 0; j < nk; j++ {
		outs[j] = d.Blocks[i*nk+j].Apply(h)
	}
	acc := outs[0]
	for _, o := range outs[1:] {
		for k, v := range o.Data {
			acc.Data[k] += v
		}
	}
	for k := range acc.Data {
		acc.Data[k] /= float32(nk)
	}
	return acc
}

// Front is dec_in_proj then conv_pre over denormalised latents [T, 32].
func (d *Decoder) Front(z *Mat) *Mat { return d.Pre.Apply(d.InProj.Apply(z)) }

// Tail is the last activation, conv_post and the clamp, to samples.
func (d *Decoder) Tail(x *Mat) []float32 {
	y := d.Post.Apply(d.PostAct.Apply(x))
	for i, v := range y.Data {
		y.Data[i] = min(max(v, -1), 1)
	}
	return y.Data
}

// DecodeChannel decodes one channel's denormalised latents [T, 32] into
// T·800 samples.
func (d *Decoder) DecodeChannel(z *Mat) []float32 {
	h := d.Front(z)
	for i := range d.Ups {
		h = d.Stage(i, h)
	}
	return d.Tail(h)
}

// Decode takes the transformer's audio rows — [2·n, 32], the left
// channel's n latents then the right's, normalised — and returns the
// stereo waveform, left and right, n·800 samples each:
// `MiniMaxH3AfterDenoiseStep`'s unpacking, the decode step's
// denormalisation, and the codec over each channel.
func (d *Decoder) Decode(rows []float32) (left, right []float32, err error) {
	c := d.Cfg.LatentChannels
	if len(rows)%(2*c) != 0 {
		return nil, nil, fmt.Errorf("audiovae: %d values are not 2 channels of %d-wide latents", len(rows), c)
	}
	n := len(rows) / (2 * c)
	var ch [2][]float32
	for s := 0; s < 2; s++ {
		z := NewMat(n, c)
		for t := 0; t < n; t++ {
			src := rows[(s*n+t)*c : (s*n+t+1)*c]
			for k, v := range src {
				z.Data[t*c+k] = float32(v*d.Cfg.LatentsStd[k]) + d.Cfg.LatentsMean[k]
			}
		}
		ch[s] = d.DecodeChannel(z)
	}
	return ch[0], ch[1], nil
}

// Load reads the decoder from dir (the checkpoint's audio_vae/).
func Load(dir string) (*Decoder, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}
	d := &Decoder{Cfg: cfg}
	d.InProj = l.conv("dec_in_proj", cfg.LatentChannels, cfg.LatentDim, 1, 1, false, true)
	d.Pre = l.conv("decoder.conv_pre", cfg.LatentDim, cfg.DecoderDim, 7, 1, true, true)
	ch := cfg.DecoderDim
	for i, r := range cfg.Rates {
		k := cfg.Kernels[i]
		d.Ups = append(d.Ups, l.up(fmt.Sprintf("decoder.ups.%d.0", i), ch, ch/2, k, r))
		ch /= 2
		for j, rk := range cfg.ResKernels {
			p := fmt.Sprintf("decoder.resblocks.%d.", i*len(cfg.ResKernels)+j)
			b := &AMP{}
			for q, dil := range cfg.ResDilations[j] {
				b.Convs1 = append(b.Convs1, l.conv(fmt.Sprintf("%sconvs1.%d", p, q), ch, ch, rk, dil, true, true))
				b.Convs2 = append(b.Convs2, l.conv(fmt.Sprintf("%sconvs2.%d", p, q), ch, ch, rk, 1, true, true))
			}
			for q := 0; q < 2*len(cfg.ResDilations[j]); q++ {
				b.Acts = append(b.Acts, l.act(fmt.Sprintf("%sactivations.%d", p, q), ch))
			}
			d.Blocks = append(d.Blocks, b)
		}
	}
	d.PostAct = l.act("decoder.activation_post", ch)
	d.Post = l.conv("decoder.conv_post", ch, 1, 7, 1, true, false)
	if l.err != nil {
		return nil, l.err
	}
	return d, nil
}

type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) f32(name string, n int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
		return nil
	}
	if len(v) != n {
		l.err = fmt.Errorf("audiovae: %s has %d values, want %d", name, len(v), n)
		return nil
	}
	return v
}

// weightNorm folds weight_g · v/‖v‖, the norm over every axis but the
// first, in float64: v is [rows, rest].
func weightNorm(g, v []float32, rows int) []float32 {
	rest := len(v) / rows
	w := make([]float32, len(v))
	for r := 0; r < rows; r++ {
		var ss float64
		for _, x := range v[r*rest : (r+1)*rest] {
			ss += float64(x) * float64(x)
		}
		s := float64(g[r]) / math.Sqrt(ss)
		for i, x := range v[r*rest : (r+1)*rest] {
			w[r*rest+i] = float32(float64(x) * s)
		}
	}
	return w
}

// conv loads a Conv1d [out, in, k] ("same" padding at its dilation) and
// repacks it [k][out][in].
func (l *loader) conv(p string, in, out, k, dil int, wn, bias bool) *Conv {
	var w []float32
	if wn {
		w = weightNorm(l.f32(p+".weight_g", out), l.f32(p+".weight_v", out*in*k), out)
	} else {
		w = l.f32(p+".weight", out*in*k)
	}
	c := &Conv{In: in, Out: out, K: k, Dilation: dil, Pad: (k*dil - dil) / 2, W: make([]float32, len(w))}
	if l.err != nil {
		return c
	}
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			for t := 0; t < k; t++ {
				c.W[(t*out+o)*in+i] = w[(o*in+i)*k+t]
			}
		}
	}
	if bias {
		c.B = l.f32(p+".bias", out)
	}
	return c
}

// up loads a ConvTranspose1d [in, out, k] (weight_norm over out and k, per
// input channel) and repacks it [k][out][in].
func (l *loader) up(p string, in, out, k, stride int) *Up {
	w := weightNorm(l.f32(p+".weight_g", in), l.f32(p+".weight_v", in*out*k), in)
	u := &Up{In: in, Out: out, K: k, Stride: stride, Pad: (k - stride) / 2, W: make([]float32, len(w)), B: l.f32(p+".bias", out)}
	if l.err != nil {
		return u
	}
	for i := 0; i < in; i++ {
		for o := 0; o < out; o++ {
			for t := 0; t < k; t++ {
				u.W[(t*out+o)*in+i] = w[(i*out+o)*k+t]
			}
		}
	}
	return u
}

func (l *loader) act(p string, ch int) *Act {
	alpha, beta := l.f32(p+".act.alpha", ch), l.f32(p+".act.beta", ch)
	a := &Act{A: make([]float32, ch), IB: make([]float32, ch),
		Up: l.f32(p+".upsample.filter", actKernel), Down: l.f32(p+".downsample.lowpass.filter", actKernel)}
	if l.err != nil {
		return a
	}
	for c := 0; c < ch; c++ {
		a.A[c] = float32(math.Exp(float64(alpha[c])))
		eb := float32(math.Exp(float64(beta[c])))
		a.IB[c] = 1 / (eb + 1e-9)
	}
	return a
}

// parallelFor runs fn(0..n-1) over GOMAXPROCS workers.
func parallelFor(n int, fn func(i int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
