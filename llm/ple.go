package llm

import "math"

// Per-layer embeddings: the n-gram block that runs once, at layer 1, between
// `hc_init` and the rest of the stack.
//
// It is the reason a quarter of UD-Q4_K_XL is a lookup table.
// `per_layer_token_embd` is 320 001 536 rows of 160 values — 28.80 GB, IQ4_NL
// — and a token reads **sixteen** of them: 1.41 KB. So the table is capacity
// and not bandwidth (D2), and llama.cpp agrees, creating it `TENSOR_READ_LAZY`
// and leaving it in the mapping.
//
// Which sixteen is a hash of the token and its two predecessors, computed on
// the host. For n = 2 and n = 3 the n-gram is mixed into one 64-bit value,
//
//	mixed = ctx[0]*mult[0]  then  mixed ^= ctx[j]*mult[j]  for j = 1..n-1
//
// and each of the 8 heads of that group takes `mixed % vocab[h] + offset[h]`
// — the same mixed value against sixteen near-prime vocabularies around
// 20 000 0xx, which is what spreads one n-gram over sixteen disjoint ranges of
// the table. An EOS anywhere in the window, or running off the start of the
// sequence, resets every earlier position to EOS; the token's own EOS does
// not cut its own context.
//
// What the block then does with those rows is a gated value, added into the
// wide residual twice:
//
//	key    = W_key  * emb          [hc*nEmbd]   grouped-normed
//	value  = W_val  * emb          [nEmbd]
//	query  = grouped_norm(res)     [hc*nEmbd]
//	s      = sum_i(key*query)/sqrt(nEmbd)       per stream
//	gate   = sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))
//	gated  = value (broadcast over the streams) * gate
//	conv   = silu(depthwise causal conv over tokens of grouped_norm(gated))
//	res   += gated + conv
//
// The signed square root before the sigmoid is the unusual part and it is not
// a normalisation: it compresses a dot product of two 2560-wide normalised
// vectors into a range a sigmoid can resolve.
//
// The convolution is depthwise over all 10240 channels, kernel 4, **dilated by
// the n-gram size**, so its taps reach 0, 3, 6 and 9 tokens back. That makes
// it the second recurrent thing in the model after DeltaNet: at decode it
// needs a 9-token history per sequence, which llama.cpp keeps in a row of the
// recurrent cache. On the device that history is `PLEGPU`'s ring (L7b); this
// file is prefill from position zero, so here it is still zeros — and
// `PLERows` below already takes the whole id list, so a continuing run needs
// nothing from it but a slice.

// PLEConfig is the block's shape and its hash constants, all read from the
// checkpoint's own metadata (`qwen4exp.ple.*`) rather than transcribed.
type PLEConfig struct {
	NEmbd   int
	HC      int
	HeadDim int // embedding_length_per_layer_input, 160
	NGram   int
	PerGram int // heads_per_ngram
	NHeads  int // (NGram-1) * PerGram
	Conv    int // conv_kernel
	Layers  []int
	EOS     int32
	Image   int32
	Mult    []uint64 // layer_multipliers, one per n-gram position
	Offsets []uint32 // head_offsets, one per head
	Vocabs  []uint32 // head_vocab_sizes, one per head
	Eps     float32
	// Act selects the numerics of the two Q8_0 projections, as in HCConfig:
	// llama.cpp evaluates them over int8 activations.
	Act Numerics
}

// Wide is the residual's width and EmbdWidth the gathered n-gram embedding's,
// which is NHeads x HeadDim = 2560 — the same as nEmbd, and the reason the
// key and value projections read it directly.
func (c PLEConfig) Wide() int      { return c.HC * c.NEmbd }
func (c PLEConfig) EmbdWidth() int { return c.NHeads * c.HeadDim }

// ConvHist is how far back the depthwise convolution reaches: (Conv-1)*NGram
// tokens, 9 for this checkpoint's kernel of 4 dilated by 3. It is the length
// of the ring a continuing run reads (L7b) and the number of rows a run has to
// leave behind it.
func (c PLEConfig) ConvHist() int { return (c.Conv - 1) * c.NGram }

// IsPLE reports whether a layer carries the block. The checkpoint lists one.
func (c PLEConfig) IsPLE(layer int) bool {
	for _, l := range c.Layers {
		if l == layer {
			return true
		}
	}
	return false
}

// PLEWeights is the block's tensors, in the row-major [out][in] a dot product
// wants:
//
//	Key       [hc*nEmbd][nEmbd]
//	Value     [nEmbd][nEmbd]
//	NormKey   [hc*nEmbd]   NormQuery, NormConv the same
//	Conv1d    [hc*nEmbd][conv]   channel-major, as the GGUF states it
type PLEWeights struct {
	Key       []float32
	Value     []float32
	NormKey   []float32
	NormQuery []float32
	NormConv  []float32
	Conv1d    []float32
}

// PLERows computes the row indices a prompt gathers: [T][NHeads], in the order
// llama.cpp's `ggml_get_rows` input carries them, so row (t*NHeads + h) of the
// gather is head h of token t.
//
// ids is a whole sequence from position zero. The predecessors of token i are
// ids[i-1] and ids[i-2]; before the start there are none, which reads as EOS
// and cuts everything earlier — the same as an EOS in the window.
func PLERows(c PLEConfig, ids []int32) []int32 {
	return PLERowsFrom(c, ids, 0)
}

// PLERowsFrom is PLERows for the tail of a sequence: the rows of positions
// `from` onwards, which is `(len(ids)-from)*NHeads` of them, and exactly what
// `PLERows(c, ids)[from*c.NHeads:]` returns.
//
// **It exists because the whole-sequence form is O(context) on a hot path.**
// Every run of the graph gathers the n-gram rows of the tokens it is about to
// push and throws the rest away, so a decode step at 128 000 cells hashed
// 128 000 positions and allocated 8.2 MB of them to use sixteen — 5.2 ms of a
// 44.3 ms token, and 4.2 of the 10.6 ms that token gains over one at depth
// zero. It was the largest *host* term in a deep decode step and it did not
// belong to the device at all.
//
// The rows are identical and not merely close, by construction: a position's
// row depends on `ids[i-NGram+1 .. i]` and nothing else — the `cut` flag is
// reset at the top of each position, so no state carries between them — and
// this walks the same positions with the same window. `from` below the window
// simply starts earlier. TestPLERowsFromIsTheTail is the gate.
func PLERowsFrom(c PLEConfig, ids []int32, from int) []int32 {
	if from < 0 {
		from = 0
	}
	if from > len(ids) {
		from = len(ids)
	}
	rows := make([]int32, (len(ids)-from)*c.NHeads)
	ctx := make([]uint64, c.NGram)
	eos := uint64(c.EOS)
	for i := from; i < len(ids); i++ {
		ctx[0] = uint64(ids[i])
		cut := false
		for s := 1; s < c.NGram; s++ {
			var t int64 = -1
			if !cut && i-s >= 0 {
				t = int64(ids[i-s])
			}
			// A missing predecessor, or an EOS, is EOS — and cuts every
			// position before it. The token's own id is never cut.
			if cut || t < 0 || uint64(t) == eos {
				cut = true
				ctx[s] = eos
			} else {
				ctx[s] = uint64(t)
			}
		}
		for n := 2; n <= c.NGram; n++ {
			mixed := ctx[0] * c.Mult[0]
			for j := 1; j < n; j++ {
				mixed ^= ctx[j] * c.Mult[j]
			}
			base := (n - 2) * c.PerGram
			for g := 0; g < c.PerGram; g++ {
				h := base + g
				rows[(i-from)*c.NHeads+h] = int32(mixed%uint64(c.Vocabs[h]) + uint64(c.Offsets[h]))
			}
		}
	}
	return rows
}

// PLEBlock is llama.cpp's build_ple. res is the wide residual and is updated
// in place; embd is the gathered n-gram embedding, [T][EmbdWidth].
//
// gate, gated and convOut are returned because they are what the trace holds
// (`ple_gate-1`, `ple_gated_value-1`, `ple_conv_out-1`); a fused kernel would
// materialise none of them.
func PLEBlock(c PLEConfig, w PLEWeights, res, embd []float32, nTok int) (gate, gated, convOut []float32) {
	wide, ew := c.Wide(), c.EmbdWidth()
	gate = make([]float32, nTok*c.HC)
	gated = make([]float32, nTok*wide)
	convOut = make([]float32, nTok*wide)

	key := make([]float32, wide)
	keyN := make([]float32, wide)
	query := make([]float32, wide)
	value := make([]float32, c.NEmbd)
	normed := make([]float32, nTok*wide)
	var actBuf []float32
	if c.Act == RefQ8 {
		actBuf = make([]float32, ew)
	}
	invSqrt := float32(1 / math.Sqrt(float64(c.NEmbd)))

	for t := 0; t < nTok; t++ {
		e := embd[t*ew : (t+1)*ew]
		// The two Q8_0 projections read the same gathered embedding, so they
		// quantise it once between them.
		a := quantAct(e, actBuf, c.Act)
		matvec(key, w.Key, a, wide, ew)
		matvec(value, w.Value, a, c.NEmbd, ew)

		groupedRMSNorm(keyN, key, w.NormKey, c.NEmbd, c.HC, c.Eps)
		groupedRMSNorm(query, res[t*wide:(t+1)*wide], w.NormQuery, c.NEmbd, c.HC, c.Eps)

		g := gated[t*wide : (t+1)*wide]
		for ch := 0; ch < c.HC; ch++ {
			var s float32
			k := keyN[ch*c.NEmbd : (ch+1)*c.NEmbd]
			q := query[ch*c.NEmbd : (ch+1)*c.NEmbd]
			for i := range k {
				s += k[i] * q[i]
			}
			s *= invSqrt
			// The signed square root: sgn(s) * sqrt(clamp(|s|, 1e-6, inf)).
			// ggml_sgn(0) is 0, so a zero score gates at exactly 0.5.
			mag := float32(math.Sqrt(math.Max(math.Abs(float64(s)), 1e-6)))
			gv := sigmoid(sign(s) * mag)
			gate[t*c.HC+ch] = gv
			// One value vector, broadcast across the streams.
			o := g[ch*c.NEmbd : (ch+1)*c.NEmbd]
			for i, v := range value {
				o[i] = v * gv
			}
		}
		groupedRMSNorm(normed[t*wide:(t+1)*wide], g, w.NormConv, c.NEmbd, c.HC, c.Eps)
	}

	// Depthwise causal convolution over the token axis, dilated by the n-gram
	// size: out[c, t] = sum_k w[c][k] * x[c, t - (K-1-k)*dilation]. The
	// history before position zero is zero, which is what the recurrent cache
	// holds at the start of a sequence.
	dil := c.NGram
	for t := 0; t < nTok; t++ {
		out := convOut[t*wide : (t+1)*wide]
		for k := 0; k < c.Conv; k++ {
			src := t - (c.Conv-1-k)*dil
			if src < 0 {
				continue
			}
			x := normed[src*wide : (src+1)*wide]
			for ch := 0; ch < wide; ch++ {
				out[ch] += w.Conv1d[ch*c.Conv+k] * x[ch]
			}
		}
		for i, v := range out {
			out[i] = silu(v)
		}
	}

	for i := range res {
		res[i] += gated[i] + convOut[i]
	}
	return gate, gated, convOut
}

func sign(x float32) float32 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return 0
}
