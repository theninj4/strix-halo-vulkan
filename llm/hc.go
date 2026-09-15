package llm

import "math"

// Hyper-connections: qwen3.8-flash-next's replacement for the residual
// stream. Instead of one n_embd-wide vector between layers there are HC of
// them, and every block reads a learned mix of the HC streams and writes back
// into all of them with learned per-stream weights.
//
// L2a made this the first thing to build rather than the last. In llama.cpp's
// prefill graph the block is:
//
//	elementwise glue            30.6%    2453 dispatches
//	tiny-N F32 matmul           12.2%    of which hc_*_inject alone is 10.3%
//	hc.down + hc.up             ~7%      at 6.9-9.4 TFLOP/s
//
// — over half the graph, against 1.6% for the gated DeltaNet that three
// quarters of the layers are made of. And the worst part of it is free:
// `inject` is a [10240, 4] F32 matrix applied to the *same* normalised
// activation `down` reads, so it is four more output columns on a matrix that
// already has 320. This file keeps them separate because it is the CPU
// reference and clarity wins; the GPU path fuses them.
//
// Layout, everywhere in this package: activations are [tokens][features]
// row-major float32, which is byte-identical to ggml's [features, tokens]
// because ggml's ne[0] is the fastest axis. The wide residual is
// [tokens][hc][nEmbd] for the same reason — ggml has it as [n_embd, hc, T].

// HCWeights is one hyper-connection mixer: the pair a layer has before
// attention and before the FFN, and the one at the head of the model.
//
// Shapes are as the GGUF states them, transposed into the row-major
// [out][in] a dot product wants:
//
//	Norm   [hc*nEmbd]        per-stream gamma, folded to (1+w) by the converter
//	Down   [lowRank][hc*nEmbd]
//	Up     [hc*nEmbd][lowRank]
//	Inject [hc][hc*nEmbd]    nil at the head, which does not inject
type HCWeights struct {
	Norm   []float32
	Down   []float32
	Up     []float32
	Inject []float32
}

// Numerics selects how a matmul against a Q8_0 weight is evaluated.
//
// It exists because the oracle is not exact. llama.cpp's Vulkan backend runs
// a Q8_0 matmul as an *integer* dot product: it quantises the activation to
// int8 in blocks of 32 first, so the reference's own answer carries the
// activation error L0d measured rather than the f32 one. Measured on this
// block's gate, at layer 0 of the real checkpoint:
//
//	exact f32 against llama.cpp    rms 3.143e-03, maxAbs 6.99e-02
//	int8 activations               rms 3.529e-05, maxAbs 5.72e-04   89x
//
// So Exact is the mathematical model and what the GPU port should aim at;
// RefQ8 is what reproduces llama.cpp, and it is the only setting under which
// a tight tolerance against the dump means anything.
type Numerics int

const (
	// Exact evaluates in float32 throughout.
	Exact Numerics = iota
	// RefQ8 quantises each matmul's activation to int8 in blocks of 32, a
	// symmetric scale of amax/127, as ggml's q8_1 row quantiser does.
	RefQ8
)

// HCConfig is the shape of the block, read from the checkpoint's metadata
// rather than hard-coded: hyper_connection.count and .low_rank.
type HCConfig struct {
	NEmbd   int
	HC      int
	LowRank int
	Eps     float32
	// Act selects the numerics of the two Q8_0 projections. The inject
	// projection is F32 in the checkpoint and is never quantised — which is
	// how the mismatch was localised in the first place: inject agreed to
	// 2.6e-06 while the gate beside it did not.
	Act Numerics
}

// Wide returns the width of the concatenated residual, hc*n_embd.
func (c HCConfig) Wide() int { return c.HC * c.NEmbd }

// HCMix is llama.cpp's build_hc_mix: grouped RMSNorm over the HC streams, a
// low-rank sigmoid gate applied to them, the gated streams collapsed by their
// mean, and — the part that is 10% of prefill in the reference — a projection
// of the same normalised activation down to HC scatter weights.
//
//	res    [T][hc][nEmbd]   the wide residual, read not written
//	mixed  [T][nEmbd]       the block input
//	inject [T][hc]          the scatter weights build_hc_combine will use
//
// xn is returned because it is the tensor everything else in the block reads
// and the tests compare it against `hc_norm-N`; a fused kernel would never
// materialise it.
func HCMix(c HCConfig, w HCWeights, res []float32, nTok int) (mixed, inject, xn, gate []float32) {
	wide := c.Wide()
	xn = make([]float32, nTok*wide)
	gate = make([]float32, nTok*wide)
	mixed = make([]float32, nTok*c.NEmbd)
	if w.Inject != nil {
		inject = make([]float32, nTok*c.HC)
	}
	lo := make([]float32, c.LowRank)
	invHC := 1 / float32(c.HC)
	var actBuf, loBuf []float32
	if c.Act == RefQ8 {
		actBuf = make([]float32, wide)
		loBuf = make([]float32, c.LowRank)
	}

	for t := 0; t < nTok; t++ {
		src := res[t*wide : (t+1)*wide]
		dst := xn[t*wide : (t+1)*wide]

		// Grouped RMSNorm: ggml_rms_norm reduces over ne[0] alone, which is
		// one stream of one token — HC independent norms, not one over the
		// whole 10240.
		for ch := 0; ch < c.HC; ch++ {
			x := src[ch*c.NEmbd : (ch+1)*c.NEmbd]
			var ss float64
			for _, v := range x {
				ss += float64(v) * float64(v)
			}
			scale := float32(1 / math.Sqrt(ss/float64(c.NEmbd)+float64(c.Eps)))
			g := w.Norm[ch*c.NEmbd : (ch+1)*c.NEmbd]
			o := dst[ch*c.NEmbd : (ch+1)*c.NEmbd]
			for i, v := range x {
				o[i] = v * scale * g[i]
			}
		}

		// The low-rank gate. The 1/hc before the SiLU is llama.cpp's, and it
		// is not a norm: it is folded into the down projection's output.
		matvec(lo, w.Down, c.act(dst, actBuf), c.LowRank, wide)
		for i, v := range lo {
			lo[i] = silu(v * invHC)
		}
		g := gate[t*wide : (t+1)*wide]
		matvec(g, w.Up, c.act(lo, loBuf), wide, c.LowRank)
		for i, v := range g {
			g[i] = sigmoid(v)
		}

		// Collapse the gated streams by their mean.
		m := mixed[t*c.NEmbd : (t+1)*c.NEmbd]
		for ch := 0; ch < c.HC; ch++ {
			o := dst[ch*c.NEmbd : (ch+1)*c.NEmbd]
			gg := g[ch*c.NEmbd : (ch+1)*c.NEmbd]
			for i := range m {
				m[i] += o[i] * gg[i]
			}
		}
		for i := range m {
			m[i] *= invHC
		}

		if w.Inject != nil {
			matvec(inject[t*c.HC:(t+1)*c.HC], w.Inject, dst, c.HC, wide)
		}
	}
	return mixed, inject, xn, gate
}

// HCCombine is llama.cpp's build_hc_combine: scatter one block's output back
// into every stream, each with its own weight.
//
// The 2*sigmoid centres the weights on 1, so an inject of zero leaves a plain
// residual add — which is why the block can be initialised to a no-op.
// res is updated in place.
func HCCombine(c HCConfig, res, blockOut, inject []float32, nTok int) {
	invHC := 1 / float32(c.HC)
	for t := 0; t < nTok; t++ {
		b := blockOut[t*c.NEmbd : (t+1)*c.NEmbd]
		for ch := 0; ch < c.HC; ch++ {
			wgt := 2 * sigmoid(inject[t*c.HC+ch]*invHC)
			r := res[(t*c.HC+ch)*c.NEmbd : (t*c.HC+ch+1)*c.NEmbd]
			for i := range r {
				r[i] += b[i] * wgt
			}
		}
	}
}

// HCInit is the wide residual's starting value: hc identical copies of the
// token embedding (llama.cpp's `hc_init`).
func HCInit(c HCConfig, embd []float32, nTok int) []float32 {
	out := make([]float32, nTok*c.Wide())
	for t := 0; t < nTok; t++ {
		e := embd[t*c.NEmbd : (t+1)*c.NEmbd]
		for ch := 0; ch < c.HC; ch++ {
			copy(out[(t*c.HC+ch)*c.NEmbd:], e)
		}
	}
	return out
}

// matvec computes y = W x for a row-major W of n rows and k columns.
func matvec(y, w, x []float32, n, k int) {
	for r := 0; r < n; r++ {
		row := w[r*k : (r+1)*k]
		var s float32
		for i, v := range row {
			s += v * x[i]
		}
		y[r] = s
	}
}

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

func silu(x float32) float32 { return x * sigmoid(x) }

// actQuantBlock is ggml's q8 row-quantiser block length.
const actQuantBlock = 32

// act returns the activation a matmul against a Q8_0 weight actually sees.
// Under Exact that is x itself; under RefQ8 it is x put through a symmetric
// int8 grid in blocks of 32, written into buf.
func (c HCConfig) act(x, buf []float32) []float32 {
	if c.Act != RefQ8 {
		return x
	}
	for b := 0; b < len(x); b += actQuantBlock {
		e := b + actQuantBlock
		if e > len(x) {
			e = len(x)
		}
		var amax float32
		for _, v := range x[b:e] {
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		if amax == 0 {
			copy(buf[b:e], x[b:e])
			continue
		}
		// ggml's quantize_q8_1.comp, exactly: the reciprocal is taken in
		// f32 and multiplied, and only the *stored* scale is fp16. Dividing
		// by a scale already rounded to fp16 is a different answer, and a
		// measurably worse fit to the reference.
		d := amax / 127
		dInv := 1 / d
		ds := f16Round(d)
		for i := b; i < e; i++ {
			buf[i] = ds * float32(math.Round(float64(x[i]*dInv)))
		}
	}
	return buf
}

// f16Round returns x rounded to the nearest IEEE half, round-half-to-even,
// which is how ggml stores a q8_1 block scale (`f16vec2 ds`).
func f16Round(x float32) float32 {
	b := math.Float32bits(x)
	sign := b & 0x80000000
	exp := int32((b>>23)&0xff) - 127
	man := b & 0x7fffff
	switch {
	case exp == 128: // Inf or NaN
		return x
	case exp > 15: // overflows half: saturate to Inf, as a conversion does
		return math.Float32frombits(sign | 0x7f800000)
	case exp < -25:
		return math.Float32frombits(sign)
	case exp < -14: // subnormal half
		shift := uint32(-exp-14) + 13
		full := man | 0x800000
		q := full >> shift
		rem := full & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && q&1 == 1) {
			q++
		}
		return math.Float32frombits(sign) + float32(q)*sub14
	default:
		q := man >> 13
		rem := man & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && q&1 == 1) {
			q++
		}
		e := uint32(exp+127)<<23 + q<<13 // carry out of the mantissa bumps the exponent
		return math.Float32frombits(sign | e)
	}
}

// sub14 is 2^-24, the spacing of subnormal halves.
const sub14 = float32(1.0) / (1 << 24)
