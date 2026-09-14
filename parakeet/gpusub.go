package parakeet

// The subsampling stack on Vulkan: SPEECH.md stage S7.
//
// S6 left the encoder's 24 conformer layers at 13.8 ms for an 11 s clip and
// the five convolutions in front of them at 92 ms on the host -- 2.8 GFLOP at
// 30 GFLOP/s, on a part that had just sustained 12.8 TFLOP/s. So this stage
// is not a tuning problem; it is the last piece of the encoder that was never
// written for the device at all.
//
// The stack is NeMo's `dw_striding`: a dense 1 -> 256 convolution at stride 2,
// then twice (depthwise 3x3 at stride 2, pointwise 1x1), with a ReLU after
// the dense one and after each pointwise one, and finally a linear that folds
// the surviving 256 channels x 16 mel bins into the encoder's 1024 features.
// The mel spectrogram goes in at [1101, 128] and 138 frames come out.
//
// # The layout is the design
//
// Held as PyTorch holds it -- [C, T, F], channel slowest -- every operator
// here is awkward: the depthwise convolution's lanes are a plane apart, the
// pointwise ones are not GEMMs in any useful sense, and the flatten is a
// transpose. Held channel-*last*, as [T, F, C], all three become things the
// engine already has:
//
//	stride-2 dense conv   one lane per output channel, nine scalar mel reads
//	                      shared by all 256 of them (parakeet_sub_conv0.comp)
//	stride-2 depthwise    256 lanes walking 256 contiguous floats per tap
//	                      (parakeet_sub_dw.comp)
//	pointwise 1x1         a [P, 256] x [256, 256] GEMM over positions, on the
//	                      ladder's own kernel, with no packing pass
//	flatten + linear      a contiguous copy into the A operand, and the
//	                      [T, 4096] x [4096, 1024] GEMM it always was, once
//	                      the weight's columns are permuted at upload
//
// That last line is the one that could have been a per-clip transpose and is
// instead a permutation of a [1024, 4096] weight done once
// (subLinearWeight). It is also the stage's sharpest failure mode, since the
// wrong order is a plausible tensor: the subFlatOrder control breaks exactly
// it.
//
// # What the arithmetic is
//
// Of the 2.8 GFLOP, 1.16 is the linear and 1.16 the first pointwise
// convolution; the two depthwise convolutions together are 0.05, and the
// dense one 0.16. So three quarters of the stack is on the GEMM ladder and
// the two kernels this file adds are there to be correct rather than fast.

import (
	"fmt"
	"math"

	"strix-halo-vulkan/vk"
)

// SubProj names one of the three matrices the subsampling stack multiplies
// by. They are kept apart from Proj because they are not per layer and
// because their shapes are nothing like a conformer projection's: the
// pointwise convolutions are M = 8832 at N = K = 256, where every projection
// in the encoder is M = the clip's frames at N, K >= 1024.
type SubProj string

const (
	SubPW1    SubProj = "sub.pw1"
	SubPW2    SubProj = "sub.pw2"
	SubLinear SubProj = "sub.linear"
)

// subProjOrder is every subsampling matrix, in the order the graph issues
// them.
var subProjOrder = []SubProj{SubPW1, SubPW2, SubLinear}

// SubPlan chooses a kernel per subsampling projection.
type SubPlan map[SubProj]GEMMKernel

// DefaultSubPlan is the measured schedule, from TestGPUSubLadder.
//
// The expectation going in was that these two shapes would disagree: the
// pointwise convolutions are 8832 rows deep at N = K = 256, which is the one
// place in this model where a wide four-wave tile has enough rows to fill,
// and the linear is 138 rows, the narrow-M regime PlanFor exists for. They do
// not. The same wave32 32x32 tile wins both, and the wide tile loses the
// shape it was supposed to own:
//
//	kernel                 pointwise      linear
//	reg32x32_bt16_w32       1.00x          1.00x
//	reg32x64_bt16_w32       1.00x          1.12x
//	reg16x64_bt16_w32       1.11x          1.25x
//	reg32x64_bt16           1.17x          1.67x
//	reg64_bt16              1.17x          1.90x
//	wg128x256_bt16_swz8     1.24x          2.80x
//
// Which is §6.2 again, on a third set of shapes: every wave32 rung beats
// every wave64 one, and the margin at 8832 rows is 1.17-1.24x rather than the
// 1.28x the conformer measures at 138. Depth does not rescue a wave64 tile
// here because N = 256 is only four of wg128x256's tiles wide, so a
// 128x256 workgroup is reading 256 K-deep rows of B to serve 128 rows of A.
func DefaultSubPlan() SubPlan { return UniformSubPlan(GEMMReg32x32W32) }

// UniformSubPlan runs every subsampling projection on one kernel, which is
// what the ladder sweeps.
func UniformSubPlan(k GEMMKernel) SubPlan {
	p := SubPlan{}
	for _, r := range subProjOrder {
		p[r] = k
	}
	return p
}

// subOutLength is the convolution's length recurrence for this stack's one
// strided geometry: kernel 3, stride 2, padding 1.
func subOutLength(n int) int { return (n+2-3)/2 + 1 }

// subGeom is the shape of the stack for one clip: the tensor extents after
// each of the three strided convolutions, and how many time rows of each are
// not padding.
//
// The two chains are the same recurrence over different starting points, and
// they disagree by a frame at each stage -- 1101 mel frames become a tensor
// 551 rows deep of which 550 are valid, then 276 of 275, then 138 of 138. It
// is the library's bookkeeping, not an arithmetic identity, and it is the
// reason every stage here masks.
type subGeom struct {
	melRows, melBins, validMel int
	t, f, v, p                 [3]int
}

func subGeometry(melRows, melBins, validMel int) subGeom {
	g := subGeom{melRows: melRows, melBins: melBins, validMel: validMel}
	t, f, v := melRows, melBins, validMel
	for i := range g.t {
		t, f, v = subOutLength(t), subOutLength(f), subOutLength(v)
		g.t[i], g.f[i], g.v[i], g.p[i] = t, f, v, t*f
	}
	return g
}

// subWeights is where the stack's weights sit in the arenas.
type subWeights struct {
	// fp32 arena: the three 3x3 filter banks as [9, C] tap-major, and the
	// five convolution biases plus the linear's.
	conv0, bias0     uint32
	dw1, biasDW1     uint32
	dw2, biasDW2     uint32
	biasPW1, biasPW2 uint32
	biasLin          uint32
	// fp16 bank: the two pointwise convolutions and the linear, as fragment
	// tiles.
	b map[SubProj]uint32
}

// checkSubsampling holds the loaded stack to the geometry the kernels assume.
// Anything else would run wrong rather than slowly, so it is an error at
// construction and not a fallback.
func (g *GPUEncoder) checkSubsampling(s *Subsampling) error {
	if s == nil || s.Linear == nil {
		return fmt.Errorf("parakeet: the encoder has no subsampling stack")
	}
	c := g.subChans
	want := []struct {
		in, out, kernel, stride int
		depthwise               bool
	}{
		{1, c, 3, 2, false},
		{c, c, 3, 2, true},
		{c, c, 1, 1, false},
		{c, c, 3, 2, true},
		{c, c, 1, 1, false},
	}
	if len(s.Convs) != len(want) {
		return fmt.Errorf("parakeet: the subsampling stack has %d convolutions, the kernels are written for %d",
			len(s.Convs), len(want))
	}
	for i, w := range want {
		cv := s.Convs[i]
		if cv.In != w.in || cv.Out != w.out || cv.Kernel != w.kernel || cv.Stride != w.stride || cv.Depthwise != w.depthwise {
			return fmt.Errorf("parakeet: subsampling conv %d is %d->%d k%d s%d (depthwise %v), want %d->%d k%d s%d (depthwise %v)",
				cv.Index, cv.In, cv.Out, cv.Kernel, cv.Stride, cv.Depthwise, w.in, w.out, w.kernel, w.stride, w.depthwise)
		}
		if cv.Bias == nil {
			return fmt.Errorf("parakeet: subsampling conv %d has no bias, and the epilogue expects one", cv.Index)
		}
	}
	flat := g.subChans * g.maxGeom.f[2]
	if s.Linear.In != flat || s.Linear.Out != g.dim {
		return fmt.Errorf("parakeet: the subsampling linear is [%d %d], want [%d %d]", s.Linear.Out, s.Linear.In, g.dim, flat)
	}
	if s.Linear.Bias == nil {
		return fmt.Errorf("parakeet: the subsampling linear has no bias, and the epilogue expects one")
	}
	return nil
}

// layoutSub plans the stack's weight offsets, appending to the two weight
// arenas the layers already claimed. It returns the two new totals.
func (g *GPUEncoder) layoutSub(off32, off16 uint32) (uint32, uint32) {
	c := uint32(g.subChans)
	take32 := func(n uint32) uint32 { o := off32; off32 += n; return o }
	w := &g.subW
	w.conv0, w.bias0 = take32(9*c), take32(c)
	w.dw1, w.biasDW1 = take32(9*c), take32(c)
	w.dw2, w.biasDW2 = take32(9*c), take32(c)
	w.biasPW1, w.biasPW2 = take32(c), take32(c)
	w.biasLin = take32(uint32(g.dim))

	w.b = make(map[SubProj]uint32, len(subProjOrder))
	for _, r := range subProjOrder {
		n, k := g.subProjShape(r)
		w.b[r] = off16
		off16 += uint32(n * k)
	}
	return off32, off16
}

// subProjShape is [out, in] for a subsampling matrix.
func (g *GPUEncoder) subProjShape(r SubProj) (int, int) {
	c := g.subChans
	switch r {
	case SubPW1, SubPW2:
		return c, c
	default:
		return g.dim, c * g.maxGeom.f[2]
	}
}

// allocSub lays out the stack's activations inside the two arenas, sized for
// the longest clip the encoder will take. It returns the two new totals.
//
// The dense convolution's output is the large one: at 138 encoder frames it
// is [551, 64, 256] fp32, 36 MB, against 26 MB for the whole conformer stack.
// It stays fp32 because it is written once and read nine times by a kernel
// whose reads are already fully covered, and because it is the tensor a
// stagewise failure is read out of.
func (g *GPUEncoder) allocSub(off32, off16 uint32) (uint32, uint32) {
	take32 := func(n int) uint32 { o := off32; off32 += uint32(n); return o }
	take16 := func(n int) uint32 { o := off16; off16 += uint32(n); return o }
	m, c := g.maxGeom, g.subChans
	p1, p2 := roundUp(m.p[1], arenaAlign), roundUp(m.p[2], arenaAlign)

	g.sMel = take32(m.melRows * m.melBins)
	g.sV0 = take32(m.p[0] * c)
	g.sV1 = take32(p1 * c)
	g.sV2 = take32(p2 * c)

	g.ldaSub = c + gemmPad
	g.ldaFlat = c*m.f[2] + gemmPad
	g.hS1 = take16(p1 * g.ldaSub)
	g.hS2 = take16(p2 * g.ldaSub)
	g.hFlat = take16(g.rowsPad * g.ldaFlat)
	return off32, off16
}

// stageSub fills the stack's weights.
func (g *GPUEncoder) stageSub(s *Subsampling) error {
	c := g.subChans
	for _, f := range []struct {
		conv *Conv2D
		w, b uint32
	}{
		{s.Convs[0], g.subW.conv0, g.subW.bias0},
		{s.Convs[1], g.subW.dw1, g.subW.biasDW1},
		{s.Convs[3], g.subW.dw2, g.subW.biasDW2},
	} {
		g.wbuf.WriteFloat32At(int(f.w), tapMajor(f.conv.Weight, c))
		g.wbuf.WriteFloat32At(int(f.b), f.conv.Bias)
	}
	g.wbuf.WriteFloat32At(int(g.subW.biasPW1), s.Convs[2].Bias)
	g.wbuf.WriteFloat32At(int(g.subW.biasPW2), s.Convs[4].Bias)
	g.wbuf.WriteFloat32At(int(g.subW.biasLin), s.Linear.Bias)

	for _, r := range subProjOrder {
		n, k := g.subProjShape(r)
		var src []float32
		switch r {
		case SubPW1:
			src = s.Convs[2].Weight
		case SubPW2:
			src = s.Convs[4].Weight
		default:
			src = subLinearWeight(s.Linear.Weight, n, k, c, g.ctl.subFlatOrder)
		}
		if len(src) != n*k {
			return fmt.Errorf("parakeet: subsampling %s has %d weights, want %d", r, len(src), n*k)
		}
		dst := make([]uint16, n*k)
		packB(dst, src, n, k)
		g.bank.WriteUint16At(int(g.subW.b[r]), dst)
	}
	return nil
}

// tapMajor transposes a [C, 1, 3, 3] depthwise filter bank into the [9, C]
// the two convolution kernels read: one tap's C channels contiguous, so that
// a workgroup's lanes load a tap in one request and keep it in registers for
// the whole row.
func tapMajor(w []float32, c int) []float32 {
	out := make([]float32, len(w))
	for o := 0; o < c; o++ {
		for k := 0; k < 9; k++ {
			out[k*c+o] = w[o*9+k]
		}
	}
	return out
}

// subLinearWeight permutes the subsampling linear's [out, C*F] weight from
// the feature order PyTorch's flatten produces to the one this stack's
// channel-last feature map already has.
//
// The module does `h.transpose(1, 2).reshape(B, T, -1)` on a [C, T, F]
// tensor, so its feature index is c*F + f -- the channel slower. The device
// holds the map as [T, F, C], whose feature index is f*C + c. Permuting the
// weight's columns once at upload makes the flatten a contiguous copy
// (parakeet_sub_flatten_f16.comp); doing it the other way round would be a
// transpose of the activation per clip.
//
// flip is the negative control, which skips the permutation and so asks the
// GEMM to read the feature axis in the wrong order. Nothing about the result
// looks wrong: it is a [138, 1024] tensor of ordinary magnitude.
func subLinearWeight(w []float32, out, in, c int, flip bool) []float32 {
	if flip {
		return w
	}
	f := in / c
	dst := make([]float32, len(w))
	parallelFor(out, func(o int) {
		src, row := w[o*in:(o+1)*in], dst[o*in:(o+1)*in]
		for ci := 0; ci < c; ci++ {
			for fi := 0; fi < f; fi++ {
				row[fi*c+ci] = src[ci*f+fi]
			}
		}
	})
	return dst
}

// SetSubPlan changes which tile each subsampling projection runs on.
func (g *GPUEncoder) SetSubPlan(plan SubPlan) error {
	next := SubPlan{}
	for _, r := range subProjOrder {
		k, ok := plan[r]
		if !ok {
			return fmt.Errorf("parakeet: sub plan names no kernel for %s", r)
		}
		v, ok := g.kernels[k]
		if !ok {
			return fmt.Errorf("parakeet: no such projection kernel %q", k)
		}
		if v.layout != kernelLayout {
			return fmt.Errorf("parakeet: kernel %q reads B layout %d, the weights are staged as %d", k, v.layout, kernelLayout)
		}
		next[r] = k
	}
	g.subPlan = next
	return nil
}

// SubPlanOf is the kernel currently chosen for each subsampling projection.
func (g *GPUEncoder) SubPlanOf() SubPlan { return g.subPlan }

// subGraph builds the stack's dispatch sequence, with a label per dispatch --
// ten dispatches for what the host took 92 ms to do.
//
// Nothing here is per layer, so unlike layerGraph this runs once per clip and
// its extents come from the run's subGeom rather than from the config.
func (g *GPUEncoder) subGraph() ([]vk.MultiDispatch, []string, error) {
	m := g.geom
	c := uint32(g.subChans)

	var d []vk.MultiDispatch
	var kinds []string
	add := func(pipe, kind string, gx, gy uint32, pc pushConstants) {
		d = append(d, vk.MultiDispatch{Pipeline: g.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		kinds = append(kinds, kind)
	}
	// valid is the masking bound a stage is given, which the negative control
	// widens to "everything" so that the padding at the end of the clip walks
	// into the encoder.
	valid := func(n int) uint32 {
		if g.ctl.subNoMask {
			return uint32(1 << 30)
		}
		return uint32(n)
	}
	// One of the two 3x3 convolutions: same extents, same masking, different
	// kernel and a different destination arena.
	conv := func(pipe, kind string, stage int, in, out, w, bias uint32, lda int) {
		var pc pushConstants
		pc.InOff, pc.OutOff, pc.WOff, pc.Aux0 = in, out, w, bias
		pc.Aux1 = valid(m.v[stage])
		pc.Tokens, pc.Heads, pc.Dim = uint32(m.t[stage]), uint32(m.f[stage]), c
		pc.LDA = uint32(lda)
		if stage == 0 {
			pc.Span, pc.KStride = uint32(m.melBins), uint32(m.melRows)
		} else {
			pc.Span, pc.KStride = uint32(m.f[stage-1]), uint32(m.t[stage-1])
		}
		add(pipe, kind, uint32(m.t[stage]), groups(m.f[stage], subFPer), pc)
	}
	// (x + bias) * scale over a [rows, dim] tensor, in place, with the
	// positions past the valid length zeroed.
	bias := func(kind string, off, b uint32, rows, dim, live int, scale float32) {
		var pc pushConstants
		pc.InOff, pc.OutOff, pc.WOff = off, off, b
		pc.Tokens, pc.Dim, pc.Aux1 = uint32(rows), uint32(dim), valid(live)
		pc.Scale = math.Float32bits(scale)
		add("subbias", kind, uint32(rows), 1, pc)
	}
	// One subsampling GEMM: C[m, n] = A[m, k] * B[n, k], B the staged weight.
	gemm := func(r SubProj, aOff, cOff uint32, m, n, k, lda int) error {
		kernel := g.subPlan[r]
		v, ok := g.kernels[kernel]
		if !ok {
			return fmt.Errorf("parakeet: subsampling %s has no pipeline for %q", r, kernel)
		}
		m = roundUp(m, v.bm)
		if n%v.bn != 0 {
			return fmt.Errorf("parakeet: %s tile %dx%d does not divide [%d %d]", r, v.bm, v.bn, m, n)
		}
		var pc pushConstants
		pc.InOff, pc.OutOff, pc.BOff = aOff, cOff, g.subW.b[r]
		pc.GemmM, pc.GemmN, pc.GemmK = uint32(m), uint32(n), uint32(k)
		pc.LDA = uint32(lda)
		d = append(d, vk.MultiDispatch{
			Pipeline: g.gemms[kernel], GroupsX: uint32(n / v.bn), GroupsY: uint32(m / v.bm),
			PushConstants: pc.bytes(),
		})
		kinds = append(kinds, "gemm "+string(r))
		return nil
	}

	// ---- The dense 1 -> 256 convolution, and the first dw/pw pair. The
	// depthwise one writes the pointwise one's fp16 A operand directly, since
	// there is no activation between them: the ReLU in the ModuleList sits
	// before the depthwise convolution, not after it, and this kernel applies
	// it on the way in.
	conv("subconv0", "sub conv0", 0, g.sMel, g.sV0, g.subW.conv0, g.subW.bias0, 0)
	conv("subdw", "sub dw1", 1, g.sV0, g.hS1, g.subW.dw1, g.subW.biasDW1, g.ldaSub)
	if err := gemm(SubPW1, g.hS1, g.sV1, m.p[1], g.subChans, g.subChans, g.ldaSub); err != nil {
		return nil, nil, err
	}
	bias("sub bias1", g.sV1, g.subW.biasPW1, m.p[1], g.subChans, m.v[1]*m.f[1], 1)

	// ---- The second pair, which takes the map down to [138, 16, 256].
	conv("subdw", "sub dw2", 2, g.sV1, g.hS2, g.subW.dw2, g.subW.biasDW2, g.ldaSub)
	if err := gemm(SubPW2, g.hS2, g.sV2, m.p[2], g.subChans, g.subChans, g.ldaSub); err != nil {
		return nil, nil, err
	}
	bias("sub bias2", g.sV2, g.subW.biasPW2, m.p[2], g.subChans, m.v[2]*m.f[2], 1)

	// ---- The flatten and the linear, straight into the residual stream. The
	// epilogue carries the encoder's scale_input as well as the bias, so the
	// first layer reads what Encoder.forward would have handed it.
	flat := g.subChans * m.f[2]
	var pcFlat pushConstants
	pcFlat.InOff, pcFlat.OutOff = g.sV2, g.hFlat
	pcFlat.Tokens, pcFlat.Dim, pcFlat.LDA = uint32(m.t[2]), uint32(flat), uint32(g.ldaFlat)
	add("subflatten", "sub flatten", uint32(m.t[2]), 1, pcFlat)

	rowsOut := roundUp(g.rows, max(g.kernels[g.subPlan[SubLinear]].bm, g.planBM))
	if err := gemm(SubLinear, g.hFlat, g.aX, rowsOut, g.dim, flat, g.ldaFlat); err != nil {
		return nil, nil, err
	}
	scale := float32(1)
	if g.scaleInput {
		scale = float32(math.Sqrt(float64(g.dim)))
	}
	bias("sub scale", g.aX, g.subW.biasLin, g.rows, g.dim, g.rows, scale)
	return d, kinds, nil
}

// subFPer is the output frequency bins one workgroup of the two convolution
// kernels covers, and has to match their FPER.
const subFPer = 8

// SubLabels lists the subsampling stack's dispatches in order.
func (g *GPUEncoder) SubLabels() []string {
	saved := g.geom
	if g.geom.melRows == 0 {
		g.geom = g.maxGeom
	}
	_, kinds, err := g.subGraph()
	g.geom = saved
	if err != nil {
		return nil
	}
	return kinds
}

// RunSub runs the subsampling stack alone, leaving every stage's tensor in
// the arena. Unlike the layers, no two of them share a buffer, so one run is
// enough for the whole stagewise walk.
func (g *GPUEncoder) RunSub() error {
	d, _, err := g.subGraph()
	if err != nil {
		return err
	}
	return submit(d)
}

// SubShape is the tensor extent and valid time rows after the stage'th
// strided convolution, for the stagewise validation.
func (g *GPUEncoder) SubShape(stage int) (t, f, valid int) {
	return g.geom.t[stage], g.geom.f[stage], g.geom.v[stage]
}

// The subsampling stack's tensors, in the order the graph writes them.
func (g *GPUEncoder) TensorMel() uint32      { return g.sMel }
func (g *GPUEncoder) TensorSubConv0() uint32 { return g.sV0 }
func (g *GPUEncoder) TensorSubDW1() uint32   { return g.hS1 }
func (g *GPUEncoder) TensorSubPW1() uint32   { return g.sV1 }
func (g *GPUEncoder) TensorSubDW2() uint32   { return g.hS2 }
func (g *GPUEncoder) TensorSubPW2() uint32   { return g.sV2 }
func (g *GPUEncoder) TensorSubFlat() uint32  { return g.hFlat }
func (g *GPUEncoder) LDASub() int            { return g.ldaSub }
func (g *GPUEncoder) LDAFlat() int           { return g.ldaFlat }
func (g *GPUEncoder) SubChans() int          { return g.subChans }

// SubFLOPs is the stack's multiply-add count for the clip in the arena,
// counting the five convolutions and the linear.
func (g *GPUEncoder) SubFLOPs() float64 {
	m, c := g.geom, float64(g.subChans)
	dense := 2 * float64(m.p[0]) * c * 9
	dw := 2 * float64(m.p[1]+m.p[2]) * c * 9
	pw := 2 * float64(m.p[1]+m.p[2]) * c * c
	lin := 2 * float64(m.t[2]) * c * float64(m.f[2]) * float64(g.dim)
	return dense + dw + pw + lin
}
