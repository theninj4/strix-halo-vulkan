package parakeet

import (
	"fmt"
	"math"

	"strix-halo-vulkan/vk"
)

// layerGraph builds one FastConformer layer's dispatch sequence, with a label
// per dispatch. Run, Profile and RunTo share it, so what the profiler times
// is what Run runs, and the stack is one of these per layer.
//
// The layer is the macaron block: a half-weighted feed forward, attention, the
// convolution branch, a second half-weighted feed forward, and a norm on the
// way out. Each of the four branches is pre-normed, and the norm writes the
// fp16 A operand its first GEMM reads -- the narrowing is fused into the norm
// rather than being a pass of its own (dit_norm_scale_f16.comp's lesson).
//
// Nothing here touches memory, so asking for the labels costs nothing.
func (g *GPUEncoder) layerGraph(i int) ([]vk.MultiDispatch, []string, error) {
	w := &g.w[i]
	base := pushConstants{
		Tokens: uint32(g.rows), Dim: uint32(g.dim),
		Heads: uint32(g.heads), HeadDim: uint32(g.headDim),
	}

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}

	// A pre-norm: LayerNorm over the residual stream, straight into the fp16
	// arena as the branch's first A operand.
	norm := func(kind string, out uint32, n [2]uint32, lda int) {
		pc := base
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = g.aX, out, n[0], n[1]
		pc.LDA = uint32(lda)
		pc.Eps = math.Float32bits(float32(g.normEps))
		add("layernorm16", kind, uint32(g.rows), 1, pc)
	}
	// One projection: C[m, n] = A[m, k] * B[n, k], with B the staged weight.
	gemm := func(r Proj, aOff, cOff uint32, m, n, k, lda int) error {
		kernel := g.plan[r]
		v, ok := g.kernels[kernel]
		if !ok {
			return fmt.Errorf("parakeet: projection %s has no pipeline for %q", r, kernel)
		}
		if n%v.bn != 0 || m%v.bm != 0 {
			return fmt.Errorf("parakeet: %s tile %dx%d does not divide [%d %d]", r, v.bm, v.bn, m, n)
		}
		pc := base
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, w.bOff[r]
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(m), uint32(n), uint32(k)
		pc.LDA = uint32(lda)
		d = append(d, vk.MultiDispatch{
			Pipeline: g.gemms[kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, "gemm "+string(r))
		return nil
	}
	// x += scale * branch.
	residual := func(kind string, scale float32) {
		pc := base
		pc.InOff, pc.OutOff = g.aBranch, g.aX
		pc.Scale = math.Float32bits(scale)
		add("residual", kind, uint32(g.rows), 1, pc)
	}

	// ---- The first feed forward, at half weight (the macaron step).
	norm("norm ff1", g.hA, w.normFF1, g.ldaDim)
	if err := gemm(ProjFF1A, g.hA, g.aFF, g.rowsRun, g.ffn, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	pcSiLU := base
	pcSiLU.InOff, pcSiLU.OutOff = g.aFF, g.hFF
	pcSiLU.Dim, pcSiLU.LDA = uint32(g.ffn), uint32(g.ldaFFN)
	add("silu16", "silu ff1", uint32(g.rows), 1, pcSiLU)
	if err := gemm(ProjFF1B, g.hFF, g.aBranch, g.rowsRun, g.dim, g.ffn, g.ldaFFN); err != nil {
		return nil, nil, err
	}
	residual("resid ff1", 0.5)

	// ---- Attention. Two score matrices summed: the content term out of the
	// packed q/k planes, and the position term, which is three dispatches of
	// its own before the score kernel ever runs.
	norm("norm attn", g.hA, w.normAttn, g.ldaDim)
	for _, s := range []struct {
		r   Proj
		out uint32
	}{{ProjQ, g.aQ}, {ProjK, g.aK}, {ProjV, g.aV}} {
		if err := gemm(s.r, g.hA, s.out, g.rowsRun, g.dim, g.dim, g.ldaDim); err != nil {
			return nil, nil, err
		}
	}
	// q is packed twice, because the two score matrices want it in different
	// layouts and with different biases: as fragment tiles with bias_u and the
	// softmax scale for the content term, and as a plain fp16 A operand with
	// bias_v for the position term.
	scale := float32(1/math.Sqrt(float64(g.headDim))) * float32(log2e)
	pcQ := base
	pcQ.InOff, pcQ.OutOff, pcQ.WOff = g.aQ, g.hQ, w.biasU
	pcQ.Aux0, pcQ.Aux1 = 0, uint32(g.planeRows)
	pcQ.Scale = math.Float32bits(scale)
	packQ := "packbias"
	if g.ctl.noBiasU {
		packQ = "pack"
	}
	add(packQ, "pack q", groups(g.planeRows, coopMatTile), uint32(g.heads), pcQ)
	for _, s := range []struct {
		kind     string
		src, dst uint32
		mode     uint32
	}{{"pack k", g.aK, g.hK, 0}, {"pack v", g.aV, g.hV, 1}} {
		pc := base
		pc.InOff, pc.OutOff = s.src, s.dst
		pc.Aux0, pc.Aux1 = s.mode, uint32(g.planeRows)
		pc.Scale = math.Float32bits(1)
		add("pack", s.kind, groups(g.planeRows, coopMatTile), uint32(g.heads), pc)
	}
	pcQV := base
	pcQV.InOff, pcQV.OutOff, pcQV.WOff = g.aQ, g.hQV, w.biasV
	pcQV.LDA, pcQV.Aux1 = uint32(g.ldaDim), uint32(g.planeRows)
	add("narrow16", "bias v", uint32(g.planeRows), 1, pcQV)

	// The position projection: [2T-1, dim] x [dim, dim], the same weight for
	// every frame but a different one per layer, so it cannot be hoisted out
	// of the stack -- but it is independent of the activations, so it runs
	// here rather than inside the score kernel. At T=138 it is twice the work
	// of the q projection and at T=3000 it is 43x it.
	if err := gemm(ProjRelK, g.hPos, g.aRelK, g.nRun, g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	pcRK := base
	pcRK.InOff, pcRK.OutOff, pcRK.WOff = g.aRelK, g.hRelK, noW
	pcRK.Tokens, pcRK.Aux1 = uint32(g.nRun), uint32(g.nRun)
	pcRK.LDA = uint32(g.ldaDim)
	add("narrow16", "narrow rel_k", uint32(g.nRun), 1, pcRK)

	// The raw position scores, per head: [T, headDim] x [2T-1, headDim]^T.
	// This is the one GEMM in the encoder whose B operand is an activation,
	// so it reads the natural [N, ldb] layout out of the fp16 arena and runs
	// on its own pipeline; the heads are separate dispatches because the
	// reduction is per head and a single [T, dim] x [2T-1, dim] product would
	// sum over all eight of them.
	av := g.actGEMM
	if g.nRun%av.bn != 0 || g.planeRows%av.bm != 0 {
		return nil, nil, fmt.Errorf("parakeet: position tile %dx%d does not divide [%d %d]", av.bm, av.bn, g.planeRows, g.nRun)
	}
	for h := 0; h < g.heads; h++ {
		pc := base
		pc.InOff = g.hQV + uint32(h*g.headDim)
		pc.BOff = g.hRelK + uint32(h*g.headDim)
		pc.OutOff = g.aBD + uint32(h*g.planeRows*g.nRun)
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(g.planeRows), uint32(g.nRun), uint32(g.headDim)
		pc.LDA, pc.LDB = uint32(g.ldaDim), uint32(g.ldaDim)
		d = append(d, vk.MultiDispatch{
			Pipeline: g.actGEMMs[av.name], GroupsX: uint32(g.nRun / av.bn), GroupsY: uint32(g.planeRows / av.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, fmt.Sprintf("gemm pos %d", h))
	}
	// The shift: column j of row i holds the score for relative offset i-j,
	// read off a diagonal of the 2T-1 wide matrix, with the softmax scale
	// applied so that the sum with the content term is in log2 units.
	pcShift := base
	pcShift.InOff, pcShift.OutOff = g.aBD, g.aBias
	pcShift.Tokens, pcShift.KStride = uint32(g.valid), uint32(g.rows)
	pcShift.Aux0, pcShift.Aux1 = uint32(g.nRun), uint32(g.planeRows)
	pcShift.Scale = math.Float32bits(scale)
	add("relshift", "rel shift", uint32(g.planeRows), uint32(g.heads), pcShift)

	// The score kernel, with the position term as an additive bias. Its key
	// count is the clip's valid frames and its query count every frame, which
	// is what the CPU reference does: a padding frame attends to the real
	// ones and no real frame attends to it.
	pcAttn := base
	pcAttn.InOff, pcAttn.OutOff = g.hQ, g.hCtx
	pcAttn.KOff, pcAttn.VOff = g.hK, g.hV
	pcAttn.Tokens, pcAttn.KStride = uint32(g.valid), uint32(g.rows)
	pcAttn.Aux0, pcAttn.Aux1 = g.aBias, uint32(g.planeRows)
	pcAttn.LDA = uint32(g.ldaDim)
	add("attention", "attention", groups(g.rows, g.attn.rows()), uint32(g.heads), pcAttn)

	if err := gemm(ProjO, g.hCtx, g.aBranch, g.rowsRun, g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	residual("resid attn", 1)

	// ---- The convolution branch.
	norm("norm conv", g.hA, w.normConv, g.ldaDim)
	if err := gemm(ProjPW1, g.hA, g.aPW1, g.rowsRun, 2*g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	pcGLU := base
	pcGLU.InOff, pcGLU.OutOff = g.aPW1, g.aGLU
	pcGLU.Aux1 = uint32(g.valid)
	if g.ctl.noPadZero {
		pcGLU.Aux2 = 1
	}
	add("glu", "glu", uint32(g.rows), 1, pcGLU)
	pcDW := base
	pcDW.InOff, pcDW.OutOff, pcDW.WOff = g.aGLU, g.hPW, w.dw
	pcDW.Span, pcDW.LDA = uint32(g.convK), uint32(g.ldaDim)
	pcDW.Aux0, pcDW.Aux1 = w.bnScale, w.bnShift
	add("dwconv16", "dwconv", uint32(g.rows), 1, pcDW)
	if err := gemm(ProjPW2, g.hPW, g.aBranch, g.rowsRun, g.dim, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	residual("resid conv", 1)

	// ---- The second feed forward, at half weight, and the output norm.
	norm("norm ff2", g.hA, w.normFF2, g.ldaDim)
	if err := gemm(ProjFF2A, g.hA, g.aFF, g.rowsRun, g.ffn, g.dim, g.ldaDim); err != nil {
		return nil, nil, err
	}
	pcSiLU2 := pcSiLU
	add("silu16", "silu ff2", uint32(g.rows), 1, pcSiLU2)
	if err := gemm(ProjFF2B, g.hFF, g.aBranch, g.rowsRun, g.dim, g.ffn, g.ldaFFN); err != nil {
		return nil, nil, err
	}
	residual("resid ff2", 0.5)

	pcOut := base
	pcOut.InOff, pcOut.OutOff = g.aX, g.aX
	pcOut.WOff, pcOut.Aux0 = w.normOut[0], w.normOut[1]
	pcOut.Eps = math.Float32bits(float32(g.normEps))
	add("layernorm", "norm out", uint32(g.rows), 1, pcOut)

	return d, kinds, nil
}
