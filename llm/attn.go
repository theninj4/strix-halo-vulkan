package llm

import "math"

// The full-attention layer — 12 of the 48 — and the QSA indexer beside it.
//
// Three things here are not in any other transformer this repo has built.
//
// **One projection produces the query and its gate.** `attn_q` is
// [2560, 12288]: 24 heads of 256 query dims immediately followed by 256 gate
// dims, so the query is a strided view of it and the gate is the same view
// shifted by one head dim. The gate is applied as a sigmoid to the attention
// output, not to the query.
//
// **The rotary is interleaved M-RoPE.** `ggml_rope_multi` with
// `GGML_ROPE_TYPE_IMROPE`, sections [11, 11, 10, 0] over n_rot = 64 of the
// 256 head dims. For a *text* batch llama.cpp sets the t, h and w positions
// all equal to the token position and the e position to zero, and no sector
// of [11, 11, 10] ever selects e — so on text this is exactly NeoX rope, and
// `TestRoPEMultiIsNeoXOnText` says so as a checked claim rather than an
// assumption. The sections are implemented anyway, because an image batch is
// where they stop agreeing.
//
// **The indexer is a block-sparse scorer, not attention.** It projects the
// same block input to a 128-wide key and four 128-wide query heads, pools the
// keys over blocks of `compress_ratio` cells, scores every block against every
// query with a *rectified* per-head dot product, and takes the top
// `top_k + ratio - 1` cells. What that costs is 1.5% of a prefill graph
// (L2a); what it buys is that a step reads at most 2048 keys however long the
// context is.
//
// Layout, as everywhere in this package: [tokens][features] row-major, which
// is ggml's [features, tokens] read the same way round.

// AttnConfig is the layer's shape, from the checkpoint's own metadata.
type AttnConfig struct {
	NEmbd    int
	NHead    int
	NHeadKV  int
	HeadDim  int
	RopeDims int // n_rot: how many of HeadDim rotate
	RopeBase float32
	Sections [4]int
	Eps      float32
	// Act selects the numerics of the Q8_0 projections, as in HCConfig — and
	// here it carries one more thing the reference does: its KV cache is
	// **fp16**, so under RefQ8 the keys and values attention reads are
	// rounded to halves after the rotary, which is where they are written to
	// the cache. The indexer's own projections are BF16 and are never
	// quantised, which is why its tensors agree under either setting.
	Act Numerics

	// The indexer. Ratio is this layer's compress_ratios entry; zero means
	// the layer is dense and has no indexer at all.
	IdxHeads int
	IdxDim   int
	TopK     int
	Ratio    int
}

// QWidth is the fused query projection's output width: a query and a gate per
// head. Gate is that same width without the query half.
func (c AttnConfig) QWidth() int    { return c.NHead * c.HeadDim * 2 }
func (c AttnConfig) GateWidth() int { return c.NHead * c.HeadDim }
func (c AttnConfig) KVWidth() int   { return c.NHeadKV * c.HeadDim }

// AttnWeights is the layer's tensors, row-major [out][in].
type AttnWeights struct {
	Q     []float32 // [nHead*2*headDim][nEmbd]
	K, V  []float32 // [nHeadKV*headDim][nEmbd]
	O     []float32 // [nEmbd][nHead*headDim]
	QNorm []float32 // [headDim], per head
	KNorm []float32 // [headDim]

	IdxQ     []float32 // [idxHeads*idxDim][nEmbd], BF16 in the checkpoint
	IdxK     []float32 // [idxDim][nEmbd]
	IdxQNorm []float32 // [idxDim]
	IdxKNorm []float32 // [idxDim]
}

// AttnTrace is every tensor the layer produces that llama.cpp names, so a
// test can compare them one at a time instead of only the output.
type AttnTrace struct {
	Q, K, V     []float32 // [T][nHead][headDim] and [T][nHeadKV][headDim], after norm and rope
	Gate        []float32 // [T][nHead*headDim] — `gate_reshaped`
	GateSigmoid []float32 // `gate_sigmoid`
	Pregate     []float32 // `attn_pregate`, the attention output before the gate
	Gated       []float32 // `attn_gated`
	Out         []float32 // `attn_output`, [T][nEmbd]

	IdxKRaw      []float32 // [T][idxDim] — `indexer_k_raw`
	IdxKPooled   []float32 // [nBlocks][idxDim] — `indexer_k_pooled`
	IdxK         []float32 // [nBlocks][idxDim] — `indexer_k`, normed and rotated
	IdxQ         []float32 // [T][idxHeads][idxDim] — `indexer_q`
	IdxScore     []float32 // [T][nBlocks] — `indexer_score`, pre-bias
	IdxScoreCell []float32 // [T][nKV] — `indexer_score_tokens`, biased and masked
	TopK         []int32   // [T][width] — `indexer_top_k`
}

// AttnLayer runs one full-attention layer over a whole prompt from position
// zero, with a cache of nKV cells of which the first nTok are this prompt's.
//
// nKV is the reference's padded cell count and not the token count, and it
// has to be passed rather than assumed because the indexer's block structure,
// its bias and its top-k width are all cut against it — at 7 tokens in a
// 256-cell cache there is exactly *one* full block of 4 and 63 empty ones,
// and reproducing that is most of what the tests here check.
func AttnLayer(c AttnConfig, w AttnWeights, xn []float32, nTok, nKV int) *AttnTrace {
	t := &AttnTrace{}
	pos := make([]int32, nTok)
	for i := range pos {
		pos[i] = int32(i)
	}

	// The fused query projection, split per head into a query and a gate.
	qFull := make([]float32, nTok*c.QWidth())
	kv := make([]float32, nTok*c.KVWidth())
	t.Gate = make([]float32, nTok*c.GateWidth())
	t.Q = make([]float32, nTok*c.NHead*c.HeadDim)
	t.K = make([]float32, nTok*c.KVWidth())
	t.V = make([]float32, nTok*c.KVWidth())
	var actBuf []float32
	if c.Act == RefQ8 {
		actBuf = make([]float32, c.NEmbd)
	}
	for i := 0; i < nTok; i++ {
		a := quantAct(xn[i*c.NEmbd:(i+1)*c.NEmbd], actBuf, c.Act)
		matvec(qFull[i*c.QWidth():(i+1)*c.QWidth()], w.Q, a, c.QWidth(), c.NEmbd)
		matvec(kv[i*c.KVWidth():(i+1)*c.KVWidth()], w.K, a, c.KVWidth(), c.NEmbd)
		matvec(t.V[i*c.KVWidth():(i+1)*c.KVWidth()], w.V, a, c.KVWidth(), c.NEmbd)

		for h := 0; h < c.NHead; h++ {
			src := qFull[i*c.QWidth()+h*2*c.HeadDim:]
			copy(t.Q[(i*c.NHead+h)*c.HeadDim:], src[:c.HeadDim])
			copy(t.Gate[i*c.GateWidth()+h*c.HeadDim:], src[c.HeadDim:2*c.HeadDim])
		}
		copy(t.K[i*c.KVWidth():], kv[i*c.KVWidth():(i+1)*c.KVWidth()])
	}

	// Per-head RMS norm, then the rotary.
	headNorm(t.Q, w.QNorm, c.HeadDim, c.NHead*nTok, c.Eps)
	headNorm(t.K, w.KNorm, c.HeadDim, c.NHeadKV*nTok, c.Eps)
	RoPEMulti(c, t.Q, pos, c.NHead, c.HeadDim, nTok)
	RoPEMulti(c, t.K, pos, c.NHeadKV, c.HeadDim, nTok)
	if c.Act == RefQ8 {
		// Into the fp16 cache, which is what attention then reads.
		for i, v := range t.K {
			t.K[i] = f16Round(v)
		}
		for i, v := range t.V {
			t.V[i] = f16Round(v)
		}
	}

	if c.Ratio > 0 {
		t.indexer(c, w, xn, nTok, nKV)
	}

	// The attention itself. The mask is causal over the prompt, and the top-k
	// selection can only unmask cells the causal mask already allows — so
	// whenever the selection covers every visible cell, as it does at any
	// prompt shorter than top_k, this is dense causal attention.
	aq := t.Q
	if c.Act == RefQ8 {
		aq = f16Copy(t.Q) // the query meets an fp16 cache on fp16 matrix cores
	}
	t.Pregate = attention(c, aq, t.K, t.V, nTok, t.selected(c, nTok, nKV))

	t.GateSigmoid = make([]float32, len(t.Gate))
	t.Gated = make([]float32, len(t.Gate))
	for i, v := range t.Gate {
		t.GateSigmoid[i] = sigmoid(v)
		t.Gated[i] = t.Pregate[i] * t.GateSigmoid[i]
	}

	t.Out = make([]float32, nTok*c.NEmbd)
	var outBuf []float32
	if c.Act == RefQ8 {
		outBuf = make([]float32, c.GateWidth())
	}
	for i := 0; i < nTok; i++ {
		a := quantAct(t.Gated[i*c.GateWidth():(i+1)*c.GateWidth()], outBuf, c.Act)
		matvec(t.Out[i*c.NEmbd:(i+1)*c.NEmbd], w.O, a, c.NEmbd, c.GateWidth())
	}
	return t
}

// indexer is build_qsa_top_k: the block-pooled scorer that decides which cells
// attention is allowed to read.
//
// The cache layout is the part that has to be reproduced rather than invented.
// A block is `ratio` consecutive cell positions and it only exists if *all*
// `ratio` of them are occupied, so a prompt of 7 in a 256-cell cache makes one
// block of 4 and leaves cells 4-6 unpooled. `blk_cells` is zero-filled, so
// every block that does not exist pools cell 0 four times — which is why
// `indexer_k_pooled` rows 1..63 are all exactly the raw key of token 0, and
// why that is a check rather than a curiosity.
func (t *AttnTrace) indexer(c AttnConfig, w AttnWeights, xn []float32, nTok, nKV int) {
	nBlocks := (nKV + c.Ratio - 1) / c.Ratio
	nBid := nTok / c.Ratio // whole blocks only

	// The raw key, cached before the norm and the rotation — which is why
	// pooling can precede both.
	t.IdxKRaw = make([]float32, nTok*c.IdxDim)
	for i := 0; i < nTok; i++ {
		matvec(t.IdxKRaw[i*c.IdxDim:(i+1)*c.IdxDim], w.IdxK, xn[i*c.NEmbd:(i+1)*c.NEmbd], c.IdxDim, c.NEmbd)
	}

	// What pooling reads is the *cache*, and the indexer cache is fp16 like
	// the attention one — `llama_memory_hybrid_idx` passes the context's
	// type_k straight through. So the keys are rounded to halves between
	// being projected and being pooled, and modelling that is worth 280x
	// here: without it `indexer_k_pooled` sits at 3.5e-04 rms against
	// llama.cpp where `indexer_k_raw`, which is compared *before* the cache
	// write, is at 1.3e-06.
	cached := t.IdxKRaw
	if c.Act == RefQ8 {
		cached = make([]float32, len(t.IdxKRaw))
		for i, v := range t.IdxKRaw {
			cached[i] = f16Round(v)
		}
	}

	t.IdxKPooled = make([]float32, nBlocks*c.IdxDim)
	inv := 1 / float32(c.Ratio)
	for b := 0; b < nBlocks; b++ {
		dst := t.IdxKPooled[b*c.IdxDim : (b+1)*c.IdxDim]
		for s := 0; s < c.Ratio; s++ {
			cell := 0 // the zero fill: a block that does not exist pools cell 0
			if b < nBid {
				cell = b*c.Ratio + s
			}
			for i, v := range cached[cell*c.IdxDim : (cell+1)*c.IdxDim] {
				dst[i] += v
			}
		}
		for i := range dst {
			dst[i] *= inv
		}
	}

	// Norm and rotate. A block's position is the first cell position it
	// covers, and every block that does not exist keeps the zero fill.
	t.IdxK = append([]float32(nil), t.IdxKPooled...)
	headNorm(t.IdxK, w.IdxKNorm, c.IdxDim, nBlocks, c.Eps)
	blkPos := make([]int32, nBlocks)
	for b := 0; b < nBid; b++ {
		blkPos[b] = int32(b * c.Ratio)
	}
	RoPEMulti(c, t.IdxK, blkPos, 1, c.IdxDim, nBlocks)

	t.IdxQ = make([]float32, nTok*c.IdxHeads*c.IdxDim)
	pos := make([]int32, nTok)
	for i := 0; i < nTok; i++ {
		pos[i] = int32(i)
		matvec(t.IdxQ[i*c.IdxHeads*c.IdxDim:(i+1)*c.IdxHeads*c.IdxDim], w.IdxQ,
			xn[i*c.NEmbd:(i+1)*c.NEmbd], c.IdxHeads*c.IdxDim, c.NEmbd)
	}
	headNorm(t.IdxQ, w.IdxQNorm, c.IdxDim, c.IdxHeads*nTok, c.Eps)
	RoPEMulti(c, t.IdxQ, pos, c.IdxHeads, c.IdxDim, nTok)

	// Rectify each head's dot product before summing, as the DeepSeek
	// lightning indexer does: a head that disagrees contributes nothing
	// rather than cancelling a head that agrees.
	//
	// Both operands of this one are F32 tensors — and the reference still
	// evaluates it in **fp16**. That is the third thing this model's oracle
	// does that its graph does not say: L2b found int8 activations under a
	// Q8_0 weight, the indexer cache is fp16, and a plain f32 matmul on this
	// backend goes through the fp16 matrix cores. Modelling it is worth 180x
	// on the score (4.1e-03 rms to 2.3e-05) and nothing else explains the
	// gap — the same arithmetic in float64 lands in exactly the same place.
	sk, sq := t.IdxK, t.IdxQ
	if c.Act == RefQ8 {
		sk, sq = f16Copy(t.IdxK), f16Copy(t.IdxQ)
	}
	t.IdxScore = make([]float32, nTok*nBlocks)
	for i := 0; i < nTok; i++ {
		for b := 0; b < nBlocks; b++ {
			var s float32
			k := sk[b*c.IdxDim : (b+1)*c.IdxDim]
			for h := 0; h < c.IdxHeads; h++ {
				q := sq[(i*c.IdxHeads+h)*c.IdxDim:]
				var d float32
				for j := range k {
					d += q[j] * k[j]
				}
				if d > 0 {
					s += d
				}
			}
			t.IdxScore[i*nBlocks+b] = s
		}
	}

	// The bias, then the expansion to cells, then the causal mask.
	//
	// 1e9 is llama.cpp's "always visible" marker rather than an infinity, so
	// that it can never meet a -inf and make a nan: it marks the incomplete
	// tail — the cells after the last whole block — which the reference
	// always attends to. Blocks that do not exist are -inf, and the one spare
	// block that collects every unpooled cell carries the tail's 1e9.
	neg := float32(math.Inf(-1))
	t.IdxScoreCell = make([]float32, nTok*nKV)
	deadBid := nBid
	if nBid >= nBlocks {
		deadBid = nBlocks - 1
	}
	bias := make([]float32, nBlocks)
	for i := 0; i < nTok; i++ {
		tailStart := (i + 1) / c.Ratio * c.Ratio
		for b := 0; b < nBlocks; b++ {
			switch {
			case b >= nBid:
				bias[b] = neg
			case b*c.Ratio >= tailStart:
				bias[b] = 1e9
			default:
				bias[b] = 0
			}
		}
		if nBid < nBlocks {
			bias[deadBid] = 1e9
		}
		for j := 0; j < nKV; j++ {
			blk := deadBid
			if j < nBid*c.Ratio {
				blk = j / c.Ratio
			}
			v := t.IdxScore[i*nBlocks+blk] + bias[blk]
			if j > i || j >= nTok {
				v = neg // the causal mask, and every empty cell
			}
			t.IdxScoreCell[i*nKV+j] = v
		}
	}

	// The reference asks for whole blocks plus the incomplete tail.
	width := c.TopK + c.Ratio - 1
	if width > nKV {
		width = nKV
	}
	t.TopK = make([]int32, nTok*width)
	for i := 0; i < nTok; i++ {
		copy(t.TopK[i*width:(i+1)*width], topK(t.IdxScoreCell[i*nKV:(i+1)*nKV], width))
	}
}

// selected turns the top-k list into a per-token mask over the prompt's own
// cells. A cell the selection leaves out is unreadable however visible the
// causal mask says it is.
func (t *AttnTrace) selected(c AttnConfig, nTok, nKV int) []bool {
	if c.Ratio == 0 || t.TopK == nil {
		return nil
	}
	width := len(t.TopK) / nTok
	out := make([]bool, nTok*nTok)
	for i := 0; i < nTok; i++ {
		for _, cell := range t.TopK[i*width : (i+1)*width] {
			if int(cell) < nTok {
				out[i*nTok+int(cell)] = true
			}
		}
	}
	return out
}

// attention is causal grouped-query attention over the prompt. allow is the
// selection, or nil for dense.
func attention(c AttnConfig, q, k, v []float32, nTok int, allow []bool) []float32 {
	rep := c.NHead / c.NHeadKV
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	out := make([]float32, nTok*c.NHead*c.HeadDim)
	scores := make([]float32, nTok)
	for i := 0; i < nTok; i++ {
		for h := 0; h < c.NHead; h++ {
			kvh := h / rep
			qv := q[(i*c.NHead+h)*c.HeadDim:]
			n := 0
			for j := 0; j <= i; j++ {
				if allow != nil && !allow[i*nTok+j] {
					continue
				}
				kv := k[(j*c.NHeadKV+kvh)*c.HeadDim:]
				var s float32
				for d := 0; d < c.HeadDim; d++ {
					s += qv[d] * kv[d]
				}
				scores[n] = s * scale
				n++
			}
			// Softmax over the visible cells, then the value average.
			max := scores[0]
			for _, s := range scores[1:n] {
				if s > max {
					max = s
				}
			}
			var sum float32
			for j := 0; j < n; j++ {
				scores[j] = float32(math.Exp(float64(scores[j] - max)))
				sum += scores[j]
			}
			dst := out[(i*c.NHead+h)*c.HeadDim:]
			n = 0
			for j := 0; j <= i; j++ {
				if allow != nil && !allow[i*nTok+j] {
					continue
				}
				wgt := scores[n] / sum
				n++
				vv := v[(j*c.NHeadKV+kvh)*c.HeadDim:]
				for d := 0; d < c.HeadDim; d++ {
					dst[d] += wgt * vv[d]
				}
			}
		}
	}
	return out
}

// f16Copy is x with every value rounded to the nearest half, which is what
// the reference's matrix cores see.
func f16Copy(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = f16Round(v)
	}
	return out
}

// headNorm is RMS norm over each of n rows of dim values, with one shared
// gamma — which is what `build_norm` does to a [headDim, nHead, T] tensor.
func headNorm(x, gamma []float32, dim, n int, eps float32) {
	for r := 0; r < n; r++ {
		row := x[r*dim : (r+1)*dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		scale := float32(1 / math.Sqrt(ss/float64(dim)+float64(eps)))
		for i, v := range row {
			row[i] = v * scale * gamma[i]
		}
	}
}

// RoPEMulti is `ggml_rope_multi` with GGML_ROPE_TYPE_IMROPE, in place over
// [nTok][nHeads][headDim].
//
// Two things about it are easy to get wrong and are spelled out rather than
// assumed. The pairing is **NeoX**: dimension i rotates against i + n_rot/2,
// not against i+1, and the dims past n_rot are copied untouched. And the angle
// for pair i is `pos * base^(-2i/n_rot)` where *which position* is selected by
// the interleaved section rule — sector i%3 picks t, h or w, with a fourth
// section for an extra position that the text path never reaches.
func RoPEMulti(c AttnConfig, x []float32, pos []int32, nHeads, headDim, nTok int) {
	rot := c.RopeDims
	if rot > headDim {
		rot = headDim
	}
	half := rot / 2
	sect := c.Sections[0] + c.Sections[1] + c.Sections[2] + c.Sections[3]
	if sect == 0 {
		sect = half
	}
	scale := math.Pow(float64(c.RopeBase), -2/float64(rot))
	for t := 0; t < nTok; t++ {
		// A text batch carries the same position in t, h and w, and zero in
		// the fourth; an image batch is where they differ.
		p := [4]float64{float64(pos[t]), float64(pos[t]), float64(pos[t]), 0}
		theta := p
		for i := 0; i < half; i++ {
			sector := i % sect
			var th float64
			switch {
			case sector%3 == 0 && sector < 3*c.Sections[0]:
				th = theta[0]
			case sector%3 == 1 && sector < 3*c.Sections[1]:
				th = theta[1]
			case sector%3 == 2 && sector < 3*c.Sections[2]:
				th = theta[2]
			default:
				th = theta[3]
			}
			cos := float32(math.Cos(th))
			sin := float32(math.Sin(th))
			for h := 0; h < nHeads; h++ {
				row := x[(t*nHeads+h)*headDim:]
				x0, x1 := row[i], row[i+half]
				row[i] = x0*cos - x1*sin
				row[i+half] = x0*sin + x1*cos
			}
			for j := range theta {
				theta[j] *= scale
			}
		}
	}
}

// topK returns the indices of the n largest values, largest first.
//
// Ties are *not* resolved the way ggml_top_k resolves them, and at any prompt
// shorter than top_k most of a row is -inf, so the tail of this list is not
// comparable against `indexer_top_k` and the tests only compare its finite
// prefix. What matters downstream is the set, not the order.
func topK(x []float32, n int) []int32 {
	idx := make([]int32, len(x))
	for i := range idx {
		idx[i] = int32(i)
	}
	// A partial selection sort: n is the whole row here and the rows are
	// short, so this is the clearest thing that is also stable on ties.
	for i := 0; i < n; i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if x[idx[j]] > x[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	return idx[:n]
}
