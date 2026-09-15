package llm

import "math"

// Gated DeltaNet — 36 of the 48 layers, and the last architectural piece of
// this model with no implementation anywhere in this repo to start from.
//
// The layer is a linear attention: instead of keeping every key and value it
// has ever seen, it keeps one [headDim, headDim] matrix per head and updates
// it in place. Per token and per head, with S the state, that is the delta
// rule with a forget gate:
//
//	S  <- S * exp(g)                 decay, one scalar for the whole head
//	u  <- v - kᵀS                    what the state already predicts for k
//	S  <- S + k (beta*u)ᵀ            write the correction back
//	o  <- qᵀS / sqrt(headDim)        read it out
//
// Everything in front of that is shaping: one fused [2560, 10240] projection
// produces q, k and v end to end, a depthwise causal convolution of kernel 4
// mixes each channel over four token positions, a SiLU, an L2 normalisation
// of q and k, and two tiny F32 projections produce the decay and the write
// strength. Everything behind it is a gated RMS norm against a second
// [2560, 6144] projection and the output matmul.
//
// Three things about this are worth saying before the code, because each is a
// place a reasonable reading of the paper gives a different answer from the
// reference implementation.
//
// **There are three head counts, not one, and the V heads are permuted.** q
// and k have `ssm.group_count` = 16 heads; v and the state have
// `ssm.time_step_rank` = 48. The reference's fused op maps v-head h to
// q/k-head `h % 16` — *modulo* — where ggml's own mul_mat broadcasting would
// have divided and given `h / 3`; the two disagree on 32 of the 48 heads, and
// `iq1 = iv1 % neq1` in `ggml_compute_forward_gated_delta_net_one_chunk` is
// the authority. `TestDeltaNetHeadMapIsModulo` is the negative control.
//
// But modulo is right **only because this checkpoint's V heads are not in the
// model's order**. The HF weights group them by key head — [G0_v0..v2,
// G1_v0..v2, …], which is `h / 3` — and llama.cpp's converter
// (`_LinearAttentionVReorderBase._reorder_v_heads`) transposes that grid into
// tiled order so ggml's cheap tiled broadcast lands on the right head. It
// does so to **seven** tensor families at once: `in_proj_qkv`'s V rows,
// `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d`'s V
// channels and `out_proj`'s columns. vLLM, which reads the HF weights
// directly, divides (`i_h // (H // Hg)` in flash-linear-attention's
// `chunk_fwd_kernel_o`), and the two agree on the function computed.
//
// So `h % 16` is a fact about the GGUF and not about the architecture, and
// `NewState`'s head order is the GGUF's. **Anything that sources weights from
// the bf16 checkpoint instead — which is L8's other option — has to apply
// that permutation itself, to all seven families, or it silently computes a
// different model in 36 of the 48 layers.**
//
// **The state is stored transposed.** ggml's `new_state` has ne[0] = i, the
// key axis, and ne[1] = j, the value axis, holding S[i][j] — so a row of it
// is a *column* of S, which is exactly the vector the three dot products
// want contiguous. This file keeps the reference's layout rather than the
// mathematically tidy one, because the GPU port will want it too.
//
// **Prefill is not chunked here.** `build_delta_net_chunking` exists in
// llama.cpp and is not what runs: `cparams.fused_gdn_ch` sends a multi-token
// batch to `ggml_gated_delta_net`, whose Vulkan kernel walks the tokens one
// at a time with one workgroup per head. That is why `q_conv_predelta` is
// dumped with 16 heads and not 48 — the non-fused path would have repeated it
// first. So the reference for L3 is the sequential recurrence, and the
// chunked form is a performance question for the GPU kernel rather than a
// correctness one.
//
// Layout, as everywhere in this package: [tokens][features] row-major.

// DeltaNetConfig is the layer's shape, read from the checkpoint's `ssm.*`
// keys. The names are ggml's rather than the paper's, and they do not mean
// what they say: `time_step_rank` is the value head count and
// `group_count` the key head count, neither of which is a rank or a group.
type DeltaNetConfig struct {
	NEmbd   int
	HeadDim int // ssm.state_size — the head width of q, k, v *and* the state
	NHeadK  int // ssm.group_count
	NHeadV  int // ssm.time_step_rank
	Conv    int // ssm.conv_kernel
	Inner   int // ssm.inner_size, which is NHeadV*HeadDim
	Eps     float32
	// Act selects the numerics of the three Q8_0 projections — the fused
	// qkv, the output gate z and the output — exactly as HCConfig.Act does.
	Act Numerics
	// F32MM selects the same for `ssm_alpha` and `ssm_beta`, which are F32
	// tensors and so take neither path. L2e-3 found that the reference
	// evaluates an F32 x F32 matmul on the fp16 matrix cores; whether that
	// applies to a projection this narrow is a measurement, and
	// TestDeltaNetGateNumerics is where it is made.
	F32MM Numerics
	// QKNorm selects the spelling of q and k's normalisation, which is a
	// property of the llama.cpp *build* and not of the model.
	QKNorm GDNNorm
}

// GDNNorm is the two spellings of `build_gdn_l2_norm`.
//
// This is the one place in the vertical where the oracle is not merely
// imprecise but has since been called wrong. The build that produced this
// repo's trace — and L1's `PPL 4.0340` and `tg128 25.15` — is `cff184438`,
// which predates llama.cpp's "models: fix GDN normalization from `max` to
// `rsqrt`" (5fdfa6282, #28068). Before that commit the layer used
// `ggml_l2_norm`, which divides by `max(|x|, eps)`; after it, an RMS norm
// with eps/n and a 1/sqrt(n) scale, which divides by `sqrt(|x|² + eps)`.
//
// At |x| near 1 and eps 1e-06 the two differ by about 5e-07 relative, so the
// fix changes nothing anyone would notice — but it is 15x the noise floor of
// this comparison, and `TestDeltaNetL2NormIsTheBuilds` measures it rather
// than leaving the residual unexplained.
type GDNNorm int

const (
	// L2Rsqrt is x/sqrt(|x|² + eps): the model, and llama.cpp since #28068.
	L2Rsqrt GDNNorm = iota
	// L2Max is x/max(|x|, eps): `ggml_l2_norm`, and what the build behind
	// this repo's trace and baseline actually ran.
	L2Max
)

// QKWidth is the fused projection's query half, and its key half: 16 heads of
// 128. VWidth is its value half, 48 of 128. ConvWidth is all three, which is
// the channel count the depthwise convolution runs over.
func (c DeltaNetConfig) QKWidth() int   { return c.NHeadK * c.HeadDim }
func (c DeltaNetConfig) VWidth() int    { return c.NHeadV * c.HeadDim }
func (c DeltaNetConfig) ConvWidth() int { return 2*c.QKWidth() + c.VWidth() }

// StateSize is one sequence's recurrent state: a [headDim, headDim] matrix
// per value head. 48*128*128 floats is 3.1 MB a layer, 113 MB over the 36
// linear layers — and it does not grow with the context, which is the whole
// point of the architecture.
func (c DeltaNetConfig) StateSize() int { return c.NHeadV * c.HeadDim * c.HeadDim }

// DeltaNetWeights is the layer's tensors, row-major [out][in].
type DeltaNetWeights struct {
	QKV   []float32 // [convWidth][nEmbd]  — blk.N.attn_qkv, Q8_0
	Z     []float32 // [inner][nEmbd]      — blk.N.attn_gate, Q8_0
	Out   []float32 // [nEmbd][inner]      — blk.N.ssm_out, Q8_0
	Alpha []float32 // [nHeadV][nEmbd]     — blk.N.ssm_alpha, F32
	Beta  []float32 // [nHeadV][nEmbd]     — blk.N.ssm_beta, F32
	// Conv1d is [convWidth][conv]: the taps of one channel are contiguous,
	// which is how the GGUF states it ([4, 10240], ne[0] fastest).
	Conv1d []float32
	// A is `ssm_a`, which the converter already stores as -exp(A_log): the
	// decay is a *multiply* here, not a negate-and-exponentiate.
	A      []float32 // [nHeadV]
	DTBias []float32 // [nHeadV] — blk.N.ssm_dt.bias
	Norm   []float32 // [headDim] — blk.N.ssm_norm, shared by all 48 heads
}

// DeltaNetState is what the layer carries between ubatches: the convolution's
// history and the recurrent state. Both are zero at the start of a sequence,
// which is why a prefill from position zero can pass nil.
//
// The layouts are the reference's, not the convenient ones. Conv is the last
// Conv-1 columns of the convolution's input with the *token* axis fastest,
// because that is what `ggml_ssm_conv` slides a window over. S is
// [head][j][i] holding S[i][j], the transposed form the fused op writes.
type DeltaNetState struct {
	Conv []float32 // [convWidth][conv-1], column 0 the oldest
	S    []float32 // [nHeadV][headDim][headDim]
}

// NewDeltaNetState allocates the zero state a fresh sequence starts from.
func NewDeltaNetState(c DeltaNetConfig) *DeltaNetState {
	return &DeltaNetState{
		Conv: make([]float32, c.ConvWidth()*(c.Conv-1)),
		S:    make([]float32, c.StateSize()),
	}
}

// Clone copies a state, which is what a chunk-boundary test needs and what a
// speculative decode will need at L9.
func (s *DeltaNetState) Clone() *DeltaNetState {
	return &DeltaNetState{
		Conv: append([]float32(nil), s.Conv...),
		S:    append([]float32(nil), s.S...),
	}
}

// DeltaNetTrace is every tensor llama.cpp names inside the layer, so a test
// can compare them one at a time instead of only the output. A fused kernel
// would materialise almost none of them.
type DeltaNetTrace struct {
	QKVMixed  []float32 // [T][convWidth] — `linear_attn_qkv_mixed`
	Z         []float32 // [T][inner]     — `z`
	Alpha     []float32 // [T][nHeadV]    — `alpha`, before the bias
	ASoftplus []float32 // [T][nHeadV]    — `a_softplus`
	Gate      []float32 // [T][nHeadV]    — `gate`, the log decay
	Beta      []float32 // [T][nHeadV]    — `beta`, before the sigmoid
	BetaSig   []float32 // [T][nHeadV]    — `beta_sigmoid`
	ConvRaw   []float32 // [T][convWidth] — `conv_output_raw`
	ConvSilu  []float32 // [T][convWidth] — `conv_output_silu`
	Q         []float32 // [T][nHeadK][headDim] — `q_conv`
	K         []float32 // [T][nHeadK][headDim] — `k_conv`
	V         []float32 // [T][nHeadV][headDim] — `v_conv_predelta`
	QNorm     []float32 // [T][nHeadK][headDim] — `q_conv_predelta`
	KNorm     []float32 // [T][nHeadK][headDim] — `k_conv_predelta`
	Out       []float32 // [T][nHeadV][headDim] — `attn_output`
	NewState  []float32 // [nHeadV][headDim][headDim] — `new_state`
	Final     []float32 // [T][inner]  — `final_output`, the gated norm
	Result    []float32 // [T][nEmbd]  — `linear_attn_out`
}

// DeltaNetLayer runs one linear-attention layer over nTok tokens of block
// input, advancing st in place. st may be nil, which is a fresh sequence.
func DeltaNetLayer(c DeltaNetConfig, w DeltaNetWeights, xn []float32, nTok int, st *DeltaNetState) *DeltaNetTrace {
	if st == nil {
		st = NewDeltaNetState(c)
	}
	cw, qk, vw := c.ConvWidth(), c.QKWidth(), c.VWidth()
	t := &DeltaNetTrace{
		QKVMixed:  make([]float32, nTok*cw),
		Z:         make([]float32, nTok*c.Inner),
		Alpha:     make([]float32, nTok*c.NHeadV),
		ASoftplus: make([]float32, nTok*c.NHeadV),
		Gate:      make([]float32, nTok*c.NHeadV),
		Beta:      make([]float32, nTok*c.NHeadV),
		BetaSig:   make([]float32, nTok*c.NHeadV),
		ConvRaw:   make([]float32, nTok*cw),
		ConvSilu:  make([]float32, nTok*cw),
		Q:         make([]float32, nTok*qk),
		K:         make([]float32, nTok*qk),
		V:         make([]float32, nTok*vw),
		Out:       make([]float32, nTok*vw),
		Final:     make([]float32, nTok*c.Inner),
		Result:    make([]float32, nTok*c.NEmbd),
	}

	// The four projections of the block input. The two Q8_0 ones see the
	// int8 activation; alpha and beta are F32 weights and take whatever
	// F32MM says.
	var actBuf, f32Buf []float32
	if c.Act == RefQ8 {
		actBuf = make([]float32, c.NEmbd)
	}
	if c.F32MM == RefQ8 {
		f32Buf = make([]float32, c.NEmbd)
	}
	aw, bw := w.Alpha, w.Beta
	if c.F32MM == RefQ8 {
		aw, bw = f16Copy(w.Alpha), f16Copy(w.Beta)
	}
	for i := 0; i < nTok; i++ {
		x := xn[i*c.NEmbd : (i+1)*c.NEmbd]
		a := quantAct(x, actBuf, c.Act)
		matvec(t.QKVMixed[i*cw:(i+1)*cw], w.QKV, a, cw, c.NEmbd)
		matvec(t.Z[i*c.Inner:(i+1)*c.Inner], w.Z, a, c.Inner, c.NEmbd)

		xf := x
		if c.F32MM == RefQ8 {
			xf = f16Into(f32Buf, x)
		}
		matvec(t.Alpha[i*c.NHeadV:(i+1)*c.NHeadV], aw, xf, c.NHeadV, c.NEmbd)
		matvec(t.Beta[i*c.NHeadV:(i+1)*c.NHeadV], bw, xf, c.NHeadV, c.NEmbd)
	}

	// The decay and the write strength. softplus saturates to the identity
	// above 20, which is ggml's own cutoff and not an approximation of it;
	// `ssm_a` is already negative, so `gate` is a log decay in [-inf, 0].
	for i := range t.Alpha {
		h := i % c.NHeadV
		t.ASoftplus[i] = softplus(t.Alpha[i] + w.DTBias[h])
		t.Gate[i] = t.ASoftplus[i] * w.A[h]
		t.BetaSig[i] = sigmoid(t.Beta[i])
	}

	// The depthwise causal convolution, kernel 4, one tap per channel per
	// offset, over the history in st.Conv followed by this batch. Then SiLU,
	// then the three views: q, k and v are column ranges of the same tensor.
	hist := c.Conv - 1
	for i := 0; i < nTok; i++ {
		out := t.ConvRaw[i*cw : (i+1)*cw]
		for ch := 0; ch < cw; ch++ {
			k := w.Conv1d[ch*c.Conv : (ch+1)*c.Conv]
			var s float32
			for j := 0; j < c.Conv; j++ {
				p := i + j - hist // token index; negative reads the history
				var x float32
				if p < 0 {
					x = st.Conv[ch*hist+hist+p]
				} else {
					x = t.QKVMixed[p*cw+ch]
				}
				s += x * k[j]
			}
			out[ch] = s
		}
		for ch, v := range out {
			t.ConvSilu[i*cw+ch] = silu(v)
		}
		copy(t.Q[i*qk:], t.ConvSilu[i*cw:i*cw+qk])
		copy(t.K[i*qk:], t.ConvSilu[i*cw+qk:i*cw+2*qk])
		copy(t.V[i*vw:], t.ConvSilu[i*cw+2*qk:(i+1)*cw])
	}

	// The new convolution history: the last Conv-1 columns of what it just
	// read, which is this batch's tail unless the batch is shorter than the
	// window.
	next := make([]float32, len(st.Conv))
	for ch := 0; ch < cw; ch++ {
		for p := 0; p < hist; p++ {
			src := nTok - hist + p
			if src < 0 {
				next[ch*hist+p] = st.Conv[ch*hist+hist+src]
			} else {
				next[ch*hist+p] = t.QKVMixed[src*cw+ch]
			}
		}
	}
	copy(st.Conv, next)

	// L2 normalisation of q and k over the head. Which of the two spellings
	// GDNNorm documents is used is the caller's, because it is the build's.
	t.QNorm = l2NormHeads(t.Q, c.HeadDim, c.NHeadK*nTok, c.Eps, c.QKNorm)
	t.KNorm = l2NormHeads(t.K, c.HeadDim, c.NHeadK*nTok, c.Eps, c.QKNorm)

	deltaRule(c, t, st, nTok, kHeadOfV(c))
	t.NewState = append([]float32(nil), st.S...)

	// The gated output norm: RMS over one value head against a gamma shared
	// by all 48, times sigmoid(z). The one numerical difference from
	// Qwen3.5's GDN is that this gate is a sigmoid and not a SiLU.
	for i := 0; i < nTok*c.NHeadV; i++ {
		x := t.Out[i*c.HeadDim : (i+1)*c.HeadDim]
		var ss float64
		for _, v := range x {
			ss += float64(v) * float64(v)
		}
		scale := float32(1 / math.Sqrt(ss/float64(c.HeadDim)+float64(c.Eps)))
		o := t.Final[i*c.HeadDim : (i+1)*c.HeadDim]
		z := t.Z[i*c.HeadDim : (i+1)*c.HeadDim]
		for j, v := range x {
			o[j] = v * scale * w.Norm[j] * sigmoid(z[j])
		}
	}

	var outBuf []float32
	if c.Act == RefQ8 {
		outBuf = make([]float32, c.Inner)
	}
	for i := 0; i < nTok; i++ {
		a := quantAct(t.Final[i*c.Inner:(i+1)*c.Inner], outBuf, c.Act)
		matvec(t.Result[i*c.NEmbd:(i+1)*c.NEmbd], w.Out, a, c.NEmbd, c.Inner)
	}
	return t
}

// deltaRule is the recurrence itself: `ggml_gated_delta_net`, one head at a
// time, one token at a time, in f32.
//
// The state is held as the reference holds it — s[j*D+i] = S[i][j] — so row j
// is the column of S that all three of the token's dot products read, and the
// outer-product update writes the same row. Every access here is contiguous
// and that is not an accident of the transpose; it is why the transpose is
// there.
// kOf is the value-head to key-head map, passed in rather than computed here
// so that TestDeltaNetHeadMapIsModulo can run the same recurrence with the
// other plausible map and show where it lands.
func deltaRule(c DeltaNetConfig, t *DeltaNetTrace, st *DeltaNetState, nTok int, kOf []int) {
	d := c.HeadDim
	scale := float32(1 / math.Sqrt(float64(d)))
	delta := make([]float32, d)
	for h := 0; h < c.NHeadV; h++ {
		hk := kOf[h]
		s := st.S[h*d*d : (h+1)*d*d]
		for i := 0; i < nTok; i++ {
			q := t.QNorm[(i*c.NHeadK+hk)*d : (i*c.NHeadK+hk+1)*d]
			k := t.KNorm[(i*c.NHeadK+hk)*d : (i*c.NHeadK+hk+1)*d]
			v := t.V[(i*c.NHeadV+h)*d : (i*c.NHeadV+h+1)*d]
			g := float32(math.Exp(float64(t.Gate[i*c.NHeadV+h])))
			beta := t.BetaSig[i*c.NHeadV+h]

			for j := range s {
				s[j] *= g
			}
			for j := 0; j < d; j++ {
				row := s[j*d : (j+1)*d]
				var sum float32
				for x, kv := range k {
					sum += row[x] * kv
				}
				delta[j] = (v[j] - sum) * beta
			}
			out := t.Out[(i*c.NHeadV+h)*d : (i*c.NHeadV+h+1)*d]
			for j := 0; j < d; j++ {
				row := s[j*d : (j+1)*d]
				dj := delta[j]
				var sum float32
				for x, qv := range q {
					row[x] += dj * k[x]
					sum += row[x] * qv
				}
				out[j] = sum * scale
			}
		}
	}
}

// kHeadOfV is the reference's head map: v-head h reads q/k-head h % NHeadK.
// Modulo and not divide — `iq1 = iv1 % neq1` in the fused op — because the
// converter has already permuted the V heads into tiled order for exactly
// this broadcast; see the note at the top of the file, and do not carry this
// map over to weights read from the HF checkpoint. The two maps agree on 16
// of the 48 heads, which is exactly enough for a wrong one to look nearly
// right.
func kHeadOfV(c DeltaNetConfig) []int {
	kOf := make([]int, c.NHeadV)
	for h := range kOf {
		kOf[h] = h % c.NHeadK
	}
	return kOf
}

// l2NormHeads normalises each of n rows of dim values to unit length,
// returning a new slice. Either spelling keeps a row of zeros finite: eps is
// inside the square root under L2Rsqrt and floors the divisor under L2Max.
func l2NormHeads(x []float32, dim, n int, eps float32, mode GDNNorm) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < n; r++ {
		row := x[r*dim : (r+1)*dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		var scale float32
		if mode == L2Max {
			scale = float32(1 / math.Max(math.Sqrt(ss), float64(eps)))
		} else {
			scale = float32(1 / math.Sqrt(ss+float64(eps)))
		}
		o := out[r*dim : (r+1)*dim]
		for i, v := range row {
			o[i] = v * scale
		}
	}
	return out
}

// softplus is ggml's, cutoff and all — including the part that looks like a
// bug and is load-bearing. `logf(1.0f + expf(x))` adds in **float32**, so
// every x below about -16.6 flushes to exactly zero rather than to exp(x),
// and `a_softplus-0` has one such zero among its 336 values. math.Log1p
// would return 2e-09 there and be wrong about the reference.
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log(float64(1 + float32(math.Exp(float64(x))))))
}

// f16Into rounds x into buf, the in-place form of f16Copy.
func f16Into(buf, x []float32) []float32 {
	for i, v := range x {
		buf[i] = f16Round(v)
	}
	return buf[:len(x)]
}
