package llm

import (
	"fmt"
	"math"
	"sort"

	"strix-halo-vulkan/gguf"
)

// The MoE block — LLM.md L5a — which is 97% of this checkpoint's parameters
// and **35.7% of llama.cpp's prefill graph**, the largest single line in
// L2a's attribution and the last block of the model without a kernel.
//
// `build_layer_ffn` in `src/models/qwen4exp.cpp` is two halves that are added
// together:
//
//	moe    router -> softmax -> top-10 of 512 -> normalise the ten weights
//	       -> gate and up per selected expert -> swiglu -> down -> weight
//	       -> sum the ten
//	shexp  an always-on dense FFN of the same width, times a sigmoid of its
//	       own one-column gate
//
// Three things about it are decided by the reference rather than by the
// architecture, and all three are checked below rather than assumed.
//
// **The router is F32 and its output is not.** `ffn_gate_inp` is a [2560, 512]
// F32 weight — 0.25 GB read every token for 0.06 B of parameters, L2a's other
// F32 scandal — and at a real ubatch it goes to the fp16 coopmat path like
// every other matmul wider than 8 columns (L3a-5). But L4a-5's census found
// `ffn_moe_logits` at **0.0%** exactly-representable halves where every
// quantised matmul is at 100.0%: the `.f16acc` accumulator is a property of
// the *quantised* kernels, so the router has fp16 operands and an f32
// accumulator. `shared_expert_gate` is one column wide and so is on the f32
// vector path entirely, which is why it is 0.0% for a different reason.
//
// **The ten weights are normalised, not scaled.** `norm_w` is true and
// `expert_weights_scale` is absent from the checkpoint, so the sequence is
// sum -> clamp to fp16's smallest normal -> divide, and there is no
// `ffn_moe_weights_scaled` node. The clamp is the reference's own guard
// against a zero denominator and it is reproduced rather than skipped.
//
// **The weights are applied after the FFN, not before.** `weight_before_ffn`
// is a llama4 special case; here `ffn_moe_down` is multiplied by the ten
// weights and only then summed.

// MoEConfig is the block's shape, read from the checkpoint's own metadata.
type MoEConfig struct {
	NEmbd       int
	NExpert     int // 512
	NExpertUsed int // 10
	FFNExpert   int // 640, one routed expert's hidden width
	FFNShared   int // 640, the always-on one's

	// Act selects the arithmetic, as it does for every other block: Exact is
	// the f32 model, RefQ8 is what llama.cpp computes. At prefill the gap
	// between them is smaller than the gap to the reference either way —
	// L4a-5 — because the reference accumulates the expert matmuls in fp16.
	Act Numerics
}

// MoEConfig returns the block's shape. Every layer has one, so unlike the
// attention and DeltaNet configs there is no "is this layer that kind"
// question to answer.
func (m *Model) MoEConfig() MoEConfig {
	c := m.Config
	return MoEConfig{
		NEmbd: c.NEmbd, NExpert: c.NExpert, NExpertUsed: c.NExpertUsed,
		FFNExpert: c.FFNExpert, FFNShared: c.FFNExpert,
	}
}

// ExpertBank is one of the three routed expert tensors held as the
// checkpoint's own quantised rows rather than as floats.
//
// It cannot be a `[]float32` the way every other weight in this package is.
// One layer's three banks are 2.52 billion weights — 10 GB dequantised, and
// 77 GB across the model — so the reference gathers the rows it needs and
// drops them, which is also what the GPU path will have to do. A bank is
// `[in, out, nExpert]` in GGUF order, so expert e's output row j is row
// `e*out + j` and is `in` values long.
type ExpertBank struct {
	T             *gguf.Tensor
	In, Out, NExp int
}

func newExpertBank(t *gguf.Tensor) (*ExpertBank, error) {
	if len(t.Dims) != 3 {
		return nil, fmt.Errorf("llm: %s has dims %v, want three axes", t.Name, t.Dims)
	}
	return &ExpertBank{T: t, In: int(t.Dims[0]), Out: int(t.Dims[1]), NExp: int(t.Dims[2])}, nil
}

// Expert dequantises one expert's whole [Out][In] matrix into dst, which is
// reused across calls because at 4096 tokens every one of the 512 experts is
// selected by something and a fresh 6.6 MB allocation per expert is most of
// the reference's time.
func (b *ExpertBank) Expert(e int, dst []float32) ([]float32, error) {
	if e < 0 || e >= b.NExp {
		return nil, fmt.Errorf("llm: expert %d of %d in %s", e, b.NExp, b.T.Name)
	}
	if cap(dst) < b.Out*b.In {
		dst = make([]float32, 0, b.Out*b.In)
	}
	dst = dst[:0]
	var err error
	for j := 0; j < b.Out; j++ {
		if dst, err = b.T.DequantizeRow(int64(e*b.Out+j), dst); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

// MoEWeights is one layer's FFN half. The three routed banks stay quantised;
// everything else is small enough to be floats.
type MoEWeights struct {
	Router     []float32 // [nExpert][nEmbd], F32 — 5.2 M weights, read every token
	SharedGate []float32 // [nEmbd], F32 — one output column
	GateShexp  []float32 // [ffnShared][nEmbd], Q8_0
	UpShexp    []float32 // [ffnShared][nEmbd]
	DownShexp  []float32 // [nEmbd][ffnShared]

	Gate, Up, Down *ExpertBank
}

// MoEWeights loads one layer's FFN half.
func (m *Model) MoEWeights(layer int) (MoEWeights, error) {
	var w MoEWeights
	name := func(s string) string { return fmt.Sprintf("blk.%d.%s.weight", layer, s) }
	var err error
	for _, t := range []struct {
		dst  *[]float32
		name string
	}{
		{&w.Router, "ffn_gate_inp"},
		{&w.SharedGate, "ffn_gate_inp_shexp"},
		{&w.GateShexp, "ffn_gate_shexp"},
		{&w.UpShexp, "ffn_up_shexp"},
		{&w.DownShexp, "ffn_down_shexp"},
	} {
		if *t.dst, err = m.F32(name(t.name)); err != nil {
			return w, err
		}
	}
	for _, t := range []struct {
		dst  **ExpertBank
		name string
	}{
		{&w.Gate, "ffn_gate_exps"},
		{&w.Up, "ffn_up_exps"},
		{&w.Down, "ffn_down_exps"},
	} {
		tn, err := m.Set.Get(name(t.name))
		if err != nil {
			return w, err
		}
		if *t.dst, err = newExpertBank(tn); err != nil {
			return w, err
		}
	}
	return w, nil
}

// MoETrace is every tensor llama.cpp names inside this block, in its own
// layouts, so a disagreement lands on a node of the reference's graph rather
// than on "the FFN".
type MoETrace struct {
	// Routing, over every token: [T][nExpert] and then [T][used].
	Logits            []float32
	Probs             []float32
	Argsort           []int32 // the full descending permutation, [T][nExpert]
	TopK              []int32 // its first `used` columns
	Weights           []float32
	WeightsSum        []float32 // [T]
	WeightsSumClamped []float32
	WeightsNorm       []float32

	// The routed experts, over whichever token range Experts was given:
	// [T][used][ffn] and then [T][used][nEmbd].
	Gate, Up, Swiglu []float32
	Down, Weighted   []float32
	MoEOut           []float32 // [T][nEmbd]

	// The shared expert, over the same range.
	ShGate, ShUp, ShSwiglu []float32 // [T][ffnShared]
	Shexp                  []float32 // [T][nEmbd]
	SharedGate             []float32 // [T]
	SharedGateSigmoid      []float32 // [T]
	ShexpGated             []float32 // [T][nEmbd]

	Out []float32 // [T][nEmbd] — ffn_out, the block's output
}

// Route runs everything up to the expert matmuls: the router, the softmax,
// the selection and the ten normalised weights. It is cheap — 5.2 M weights
// a token against the experts' 49 M — so it always runs over the whole
// prompt, which is what makes it comparable against a 4096-token dump.
func (t *MoETrace) Route(c MoEConfig, w MoEWeights, x []float32, nTok int) {
	nE, used := c.NExpert, c.NExpertUsed
	t.Logits = make([]float32, nTok*nE)
	t.Probs = make([]float32, nTok*nE)
	t.Argsort = make([]int32, nTok*nE)
	t.TopK = make([]int32, nTok*used)
	t.Weights = make([]float32, nTok*used)
	t.WeightsSum = make([]float32, nTok)
	t.WeightsSumClamped = make([]float32, nTok)
	t.WeightsNorm = make([]float32, nTok*used)

	parallel(nTok, func(lo, hi int) {
		idx := make([]int32, nE)
		for i := lo; i < hi; i++ {
			// The router is F32 x F32, which at any real ubatch is the fp16
			// coopmat path (L3a-5's 8-column threshold) with an f32
			// accumulator (L4a-5: its output is 0.0% exact halves).
			xi := x[i*c.NEmbd : (i+1)*c.NEmbd]
			if c.Act == RefQ8 {
				xi = f16Copy(xi)
			}
			logits := t.Logits[i*nE : (i+1)*nE]
			matvecW16(logits, w.Router, xi, nE, c.NEmbd, c.Act == RefQ8)

			probs := t.Probs[i*nE : (i+1)*nE]
			softmaxInto(probs, logits)

			for e := range idx {
				idx[e] = int32(e)
			}
			// ggml_argsort DESC, and its comparator is a plain `>` over an
			// index sequence — so equal probabilities keep ascending index
			// order, which a stable sort reproduces.
			sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
			copy(t.Argsort[i*nE:(i+1)*nE], idx)
			copy(t.TopK[i*used:(i+1)*used], idx[:used])

			var sum float32
			for k := 0; k < used; k++ {
				v := probs[idx[k]]
				t.Weights[i*used+k] = v
				sum += v
			}
			t.WeightsSum[i] = sum
			// The reference's guard against a zero denominator: clamp to
			// fp16's smallest normal, 2^-14.
			if sum < moeWeightsSumFloor {
				sum = moeWeightsSumFloor
			}
			t.WeightsSumClamped[i] = sum
			for k := 0; k < used; k++ {
				t.WeightsNorm[i*used+k] = t.Weights[i*used+k] / sum
			}
		}
	})
}

// moeWeightsSumFloor is `ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)` —
// the smallest normal half.
const moeWeightsSumFloor = 6.103515625e-5

// Experts runs the routed half over tokens [lo, hi) — the range, and not the
// whole prompt, because one (token, expert) pair is 4.9 M weights of Q4_K and
// Q5_1 to dequantise and a 4096-token prefill is 40 960 of them.
//
// It walks by **expert** rather than by token, gathering each selected
// expert's three matrices once and applying them to every token that chose
// it. That is the permutation a MoE kernel does for the same reason (§1.10's
// gather/combine), and here it is the difference between dequantising the
// bank once and dequantising it `used` times a token.
func (t *MoETrace) Experts(c MoEConfig, w MoEWeights, x []float32, lo, hi int) error {
	n, used, ff := hi-lo, c.NExpertUsed, c.FFNExpert
	t.Gate = make([]float32, n*used*ff)
	t.Up = make([]float32, n*used*ff)
	t.Swiglu = make([]float32, n*used*ff)
	t.Down = make([]float32, n*used*c.NEmbd)
	t.Weighted = make([]float32, n*used*c.NEmbd)
	t.MoEOut = make([]float32, n*c.NEmbd)

	// Which (token, slot) pairs chose each expert. 512 buckets over at most
	// n*used pairs, so this is the routing permutation itself.
	type slot struct{ tok, k int }
	byExpert := make([][]slot, c.NExpert)
	for i := lo; i < hi; i++ {
		for k := 0; k < used; k++ {
			e := int(t.TopK[i*used+k])
			byExpert[e] = append(byExpert[e], slot{i, k})
		}
	}

	var gateW, upW, downW []float32
	qbuf := make([]float32, c.NEmbd)
	qbuf2 := make([]float32, ff)
	var err error
	for e := 0; e < c.NExpert; e++ {
		if len(byExpert[e]) == 0 {
			continue
		}
		if gateW, err = w.Gate.Expert(e, gateW); err != nil {
			return err
		}
		if upW, err = w.Up.Expert(e, upW); err != nil {
			return err
		}
		if downW, err = w.Down.Expert(e, downW); err != nil {
			return err
		}
		for _, s := range byExpert[e] {
			base := ((s.tok-lo)*used + s.k) * ff
			xi := quantAct(x[s.tok*c.NEmbd:(s.tok+1)*c.NEmbd], qbuf, c.Act)
			gate := t.Gate[base : base+ff]
			up := t.Up[base : base+ff]
			matvec(gate, gateW, xi, ff, c.NEmbd)
			matvec(up, upW, xi, ff, c.NEmbd)
			sw := t.Swiglu[base : base+ff]
			for j := range sw {
				// ggml_swiglu_split(gate, up) = silu(gate) * up.
				sw[j] = silu(gate[j]) * up[j]
			}
			dbase := ((s.tok-lo)*used + s.k) * c.NEmbd
			down := t.Down[dbase : dbase+c.NEmbd]
			matvec(down, downW, quantAct(sw, qbuf2, c.Act), c.NEmbd, ff)
			wt := t.WeightsNorm[s.tok*used+s.k]
			for j := range down {
				t.Weighted[dbase+j] = down[j] * wt
			}
		}
	}
	// The ten are summed in slot order, which is the order the reference's
	// chain of adds takes them in.
	for i := 0; i < n; i++ {
		out := t.MoEOut[i*c.NEmbd : (i+1)*c.NEmbd]
		for k := 0; k < used; k++ {
			src := t.Weighted[(i*used+k)*c.NEmbd:]
			for j := range out {
				out[j] += src[j]
			}
		}
	}
	return nil
}

// Shared runs the always-on expert and its one-column sigmoid gate over
// tokens [lo, hi), and adds it to the routed half to give `ffn_out`.
func (t *MoETrace) Shared(c MoEConfig, w MoEWeights, x []float32, lo, hi int) {
	n, ffs := hi-lo, c.FFNShared
	t.ShGate = make([]float32, n*ffs)
	t.ShUp = make([]float32, n*ffs)
	t.ShSwiglu = make([]float32, n*ffs)
	t.Shexp = make([]float32, n*c.NEmbd)
	t.SharedGate = make([]float32, n)
	t.SharedGateSigmoid = make([]float32, n)
	t.ShexpGated = make([]float32, n*c.NEmbd)
	t.Out = make([]float32, n*c.NEmbd)

	parallel(n, func(a, b int) {
		qbuf := make([]float32, c.NEmbd)
		qbuf2 := make([]float32, ffs)
		for r := a; r < b; r++ {
			i := lo + r
			raw := x[i*c.NEmbd : (i+1)*c.NEmbd]
			xi := quantAct(raw, qbuf, c.Act)
			g := t.ShGate[r*ffs : (r+1)*ffs]
			u := t.ShUp[r*ffs : (r+1)*ffs]
			matvec(g, w.GateShexp, xi, ffs, c.NEmbd)
			matvec(u, w.UpShexp, xi, ffs, c.NEmbd)
			sw := t.ShSwiglu[r*ffs : (r+1)*ffs]
			for j := range sw {
				sw[j] = silu(g[j]) * u[j]
			}
			sh := t.Shexp[r*c.NEmbd : (r+1)*c.NEmbd]
			matvec(sh, w.DownShexp, quantAct(sw, qbuf2, c.Act), c.NEmbd, ffs)

			// One output column, so this is the f32 *vector* path however
			// long the prompt is (L3a-5): no fp16 operands here, which is
			// why L4a-5's census puts it at 0.0% exact halves beside the
			// quantised matmuls' 100.0%.
			var gate float32
			for j, v := range w.SharedGate {
				gate += v * raw[j]
			}
			t.SharedGate[r] = gate
			s := sigmoid(gate)
			t.SharedGateSigmoid[r] = s
			for j := range sh {
				t.ShexpGated[r*c.NEmbd+j] = sh[j] * s
				t.Out[r*c.NEmbd+j] = t.MoEOut[r*c.NEmbd+j] + sh[j]*s
			}
		}
	})
}

// MoEBlock is the whole block over tokens [lo, hi): routing over the whole
// prompt, because the selection of one token does not depend on another and
// the comparison wants every row of it, then the two halves over the range.
func MoEBlock(c MoEConfig, w MoEWeights, x []float32, nTok, lo, hi int) (*MoETrace, error) {
	t := &MoETrace{}
	t.Route(c, w, x, nTok)
	if err := t.Experts(c, w, x, lo, hi); err != nil {
		return nil, err
	}
	t.Shared(c, w, x, lo, hi)
	return t, nil
}

// softmaxInto is ggml_soft_max over one row: the max subtracted for range,
// then exp and a normalise.
func softmaxInto(dst, src []float32) {
	max := float32(math.Inf(-1))
	for _, v := range src {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range src {
		e := float32(math.Exp(float64(v - max)))
		dst[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range dst {
		dst[i] *= inv
	}
}

// matvecW16 is matvec with the weight optionally rounded to halves, which is
// what the coopmat path does to an F32 weight above the 8-column threshold.
// The accumulator stays f32, because the reference's does here.
func matvecW16(y, w, x []float32, n, k int, fp16 bool) {
	if !fp16 {
		matvec(y, w, x, n, k)
		return
	}
	for r := 0; r < n; r++ {
		row := w[r*k : (r+1)*k]
		var s float32
		for i, v := range row {
			s += f16Round(v) * x[i]
		}
		y[r] = s
	}
}
