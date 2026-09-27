package layout

import (
	"fmt"
	"math"
	"sort"
)

// Tap sees a stage under the dump's name (reference/dump_doclayout.py): a
// map's values in [C, H, W] order or a matrix's in row order.
type Tap func(name string, v []float32)

// Output is the model's raw result for one 800x800 input.
type Output struct {
	// Logits are the last layer's class logits, [300][25].
	Logits *Rows
	// Boxes are the last layer's boxes, normalised cx, cy, w, h, [300][4].
	Boxes *Rows
	// Order is the order head's [300][300] logits; below and on the diagonal
	// they are -1e4.
	Order *Rows
	// Masks are the mask logits, [300] planes of MaskH x MaskW.
	Masks        *Rows
	MaskH, MaskW int
}

// Forward runs the model over pixels, [3][800][800] in [0, 1].
func (m *Model) Forward(pixels []float32, tap Tap) (*Output, error) {
	if len(pixels) != 3*InputSize*InputSize {
		return nil, fmt.Errorf("layout: %d values, want 3x%dx%d", len(pixels), InputSize, InputSize)
	}
	emit := func(name string, v []float32) {
		if tap != nil {
			tap(name, v)
		}
	}
	x := &Map{C: 3, H: InputSize, W: InputSize, D: pixels}
	x4, proj := m.backbone(x, emit)
	proj[2] = m.aifiMap(proj[2])
	emit("enc.aifi", proj[2].D)
	pan, mf := m.neck(proj, x4, emit)
	memory, shapes := m.memory(pan, emit)
	return m.head(memory, shapes, mf, nil, tap)
}

// projected is the head's products over the whole memory, when a device
// has computed them: the encoder output's linear (before the invalid rows
// are masked and before its norm) and each decoder layer's value projection.
type projected struct {
	encLin *Rows
	values [6]*Rows
}

// backbone is HGNetV2 and the encoder's input projections: the stride-4 map
// the mask head reads, and the stride 8, 16 and 32 maps at 256 channels.
func (m *Model) backbone(x *Map, emit func(string, []float32)) (*Map, [3]*Map) {
	s1 := m.stem[0].Apply(x)
	p1 := padBR(s1)
	a := m.stem[1].Apply(p1)
	a = m.stem[2].Apply(padBR(a))
	h := concat(maxPool2(p1), a)
	h = m.stem[3].Apply(h)
	h = m.stem[4].Apply(h)
	emit("bb.stem", h.D)
	var feats [4]*Map
	for s := range m.stages {
		st := &m.stages[s]
		if st.down != nil {
			h = st.down.Apply(h)
		}
		for b := range st.blocks {
			h = st.blocks[b].apply(h)
		}
		feats[s] = h
		emit(fmt.Sprintf("bb.stage%d", s+1), h.D)
	}

	var proj [3]*Map
	for i := range proj {
		proj[i] = m.encProj[i].Apply(feats[i+1])
		emit(fmt.Sprintf("enc.proj%d", i), proj[i].D)
	}
	return feats[0], proj
}

// aifiMap is the one transformer layer over the stride-32 map.
func (m *Model) aifiMap(top *Map) *Map {
	t := m.aifi.encoder(tokens(top), sinePos(top.H, top.W, dModel, 10000))
	return untokens(t, top.H, top.W)
}

// neck is the FPN, the PAN and the mask prototypes.
func (m *Model) neck(proj [3]*Map, x4 *Map, emit func(string, []float32)) ([]*Map, *Map) {
	// Top-down FPN from stride 32 to stride 8, then bottom-up PAN.
	fpn := []*Map{proj[2]}
	for i := 0; i < 2; i++ {
		lat := m.lateral[i].Apply(fpn[len(fpn)-1])
		fpn[len(fpn)-1] = lat
		fused := concat(upNearest2(lat), proj[1-i])
		fpn = append(fpn, m.fpn[i].apply(fused))
	}
	for i, j := 0, len(fpn)-1; i < j; i, j = i+1, j-1 {
		fpn[i], fpn[j] = fpn[j], fpn[i]
	}
	pan := []*Map{fpn[0]}
	for i := 0; i < 2; i++ {
		down := m.downsample[i].Apply(pan[len(pan)-1])
		pan = append(pan, m.pan[i].apply(concat(down, fpn[i+1])))
	}
	for i, p := range pan {
		emit(fmt.Sprintf("enc.pan%d", i), p.D)
	}

	// The mask prototypes at stride 4.
	var mf *Map
	for i, head := range m.scaleHeads {
		y := pan[i]
		for _, c := range head {
			y = c.Apply(y)
			if i > 0 {
				y = upBilinear2(y)
			}
		}
		if mf == nil {
			mf = y
		} else {
			mf = add(mf, y) // same size, so the reference's interpolate is the identity
		}
	}
	mf = upBilinear2(m.maskOut.Apply(mf))
	mf = add(mf, m.maskLat.Apply(x4))
	mf = m.maskProto.Apply(m.maskBase.Apply(mf))
	emit("enc.mask_feat", mf.D)
	return pan, mf
}

// memory is the decoder's input: each PAN level's projection, flattened
// and concatenated, [13125, 256].
func (m *Model) memory(pan []*Map, emit func(string, []float32)) (*Rows, [numLevels][2]int) {
	var levels [numLevels]*Map
	var shapes [numLevels][2]int
	memory := newRows(0, dModel)
	for l := range levels {
		levels[l] = m.decProj[l].Apply(pan[l])
		shapes[l] = [2]int{levels[l].H, levels[l].W}
		tk := tokens(levels[l])
		memory.V = append(memory.V, tk.V...)
		memory.N += tk.N
	}
	emit("dec.memory", memory.V)
	return memory, shapes
}

// head is everything from the memory and the mask prototypes on: the query
// selection, the mask-enhanced initial boxes, the decoder and the heads.
func (m *Model) head(memory *Rows, shapes [numLevels][2]int, mf *Map, pre *projected, tap Tap) (*Output, error) {
	emit := func(name string, v []float32) {
		if tap != nil {
			tap(name, v)
		}
	}
	anchors, valid := makeAnchors(shapes)
	var lin *Rows
	if pre != nil {
		// A masked row is zero, so its linear is the bias alone.
		lin = pre.encLin
		for i := 0; i < memory.N; i++ {
			if !valid[i] {
				copy(lin.Row(i), m.encOutput.B)
			}
		}
	} else {
		masked := newRows(memory.N, dModel)
		for i := 0; i < memory.N; i++ {
			if valid[i] {
				copy(masked.Row(i), memory.Row(i))
			}
		}
		lin = m.encOutput.Apply(masked)
	}
	outMem := m.encOutLN.Apply(lin)
	class := m.scoreHead.Apply(outMem)
	emit("enc.class", class.V)
	if tap != nil {
		coord := m.bboxHead.apply(outMem)
		for i := range coord.V {
			coord.V[i] += anchors[i]
		}
		emit("enc.coord", coord.V)
	}
	topk := topQueries(class, numQueries)
	if tap != nil {
		v := make([]float32, len(topk))
		for i, q := range topk {
			v[i] = float32(q)
		}
		emit("enc.topk", v)
	}
	target := newRows(numQueries, dModel)
	for i, q := range topk {
		copy(target.Row(i), outMem.Row(q))
	}

	// mask_enhanced: the initial boxes are the bounding boxes of the
	// selected queries' masks > 0.
	masks := m.maskQuery.apply(m.decNorm.Apply(target))
	protos := &Rows{N: mf.C, D: mf.H * mf.W, V: mf.D}
	enc := matmul(masks, protos)
	ref := newRows(numQueries, 4)
	for q := 0; q < numQueries; q++ {
		box := maskBox(enc.Row(q), mf.H, mf.W)
		for c := range box {
			ref.V[q*4+c] = float32(inverseSigmoid(box[c]))
		}
	}
	emit("dec.init_ref", ref.V)

	// ---- decoder
	value := make([]*Rows, len(m.layers))
	hidden := target
	for i := range ref.V {
		ref.V[i] = float32(sigmoid(float64(ref.V[i])))
	}
	refs := make([]float32, 0, len(m.layers)*numQueries*4)
	logits := make([]float32, 0, len(m.layers)*numQueries*numLabels)
	var last *Rows
	for i := range m.layers {
		L := &m.layers[i]
		pos := m.queryPos.apply(ref)
		if pre != nil {
			value[i] = pre.values[i]
		} else {
			value[i] = L.value.Apply(memory)
		}
		hidden = L.forward(hidden, pos, ref, value[i], shapes)
		emit(fmt.Sprintf("dec.layer%d", i), hidden.V)
		delta := m.bboxHead.apply(hidden)
		next := newRows(numQueries, 4)
		for k := range next.V {
			next.V[k] = float32(sigmoid(float64(delta.V[k]) + inverseSigmoid(float64(ref.V[k]))))
		}
		ref = next
		refs = append(refs, ref.V...)
		last = m.decNorm.Apply(hidden)
		logits = append(logits, m.scoreHead.Apply(last).V...)
	}
	emit("dec.ref", refs)
	emit("dec.logits", logits)

	out := &Output{
		Logits: &Rows{N: numQueries, D: numLabels, V: logits[len(logits)-numQueries*numLabels:]},
		Boxes:  ref,
		MaskH:  mf.H, MaskW: mf.W,
	}
	// The order head: the global pointer over the last layer's queries.
	qk := m.pointer.Apply(m.orderHead[len(m.layers)-1].Apply(last))
	const hs = 64
	out.Order = newRows(numQueries, numQueries)
	for i := 0; i < numQueries; i++ {
		qi := qk.Row(i)[:hs]
		for j := 0; j < numQueries; j++ {
			if j <= i {
				out.Order.V[i*numQueries+j] = -1e4
				continue
			}
			kj := qk.Row(j)[hs:]
			var acc float64
			for d := 0; d < hs; d++ {
				acc += float64(qi[d]) * float64(kj[d])
			}
			out.Order.V[i*numQueries+j] = float32(acc / 8)
		}
	}
	emit("order_logits", out.Order.V)
	out.Masks = matmul(m.maskQuery.apply(last), protos)
	emit("masks", out.Masks.V)
	return out, nil
}

// apply is one HGNetV2 basic layer.
func (b *basicLayer) apply(x *Map) *Map {
	outs := []*Map{x}
	h := x
	for _, convs := range b.layers {
		for _, c := range convs {
			h = c.Apply(h)
		}
		outs = append(outs, h)
	}
	y := b.excite.Apply(b.squeeze.Apply(concat(outs...)))
	if b.residual {
		y = add(y, x)
	}
	return y
}

// apply is a CSP-RepVGG layer: conv1 through three RepVGG blocks, plus conv2.
func (c *cspRep) apply(x *Map) *Map {
	a := c.conv1.Apply(x)
	for i := range c.bottlenecks {
		r := &c.bottlenecks[i]
		y := add(r.a.Apply(a), r.b.Apply(a))
		ActSiLU.apply(y.D)
		a = y
	}
	return add(a, c.conv2.Apply(x))
}

// sinePos is build_2d_sinusoidal_position_embedding: [sin_h | cos_h | sin_w
// | cos_w] per position, h-outer, in float64.
func sinePos(h, w, dim int, temp float64) *Rows {
	pd := dim / 4
	out := newRows(h*w, dim)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			row := out.Row(y*w + x)
			for i := 0; i < pd; i++ {
				omega := 1 / math.Pow(temp, float64(i)/float64(pd))
				row[i] = float32(math.Sin(float64(y) * omega))
				row[pd+i] = float32(math.Cos(float64(y) * omega))
				row[2*pd+i] = float32(math.Sin(float64(x) * omega))
				row[3*pd+i] = float32(math.Cos(float64(x) * omega))
			}
		}
	}
	return out
}

// attention is multi-head scaled dot-product attention: q, k, v are
// [N, heads*hd].
func attention(q, k, v *Rows, heads int) *Rows {
	hd := q.D / heads
	scale := 1 / math.Sqrt(float64(hd))
	out := newRows(q.N, q.D)
	parallel(q.N, func(i int) {
		s := make([]float64, k.N)
		for h := 0; h < heads; h++ {
			qi := q.Row(i)[h*hd : (h+1)*hd]
			mx := math.Inf(-1)
			for j := 0; j < k.N; j++ {
				kj := k.Row(j)[h*hd : (h+1)*hd]
				var acc float64
				for d := range qi {
					acc += float64(qi[d]) * float64(kj[d])
				}
				s[j] = acc * scale
				mx = max(mx, s[j])
			}
			var sum float64
			for j := range s {
				s[j] = math.Exp(s[j] - mx)
				sum += s[j]
			}
			dst := out.Row(i)[h*hd : (h+1)*hd]
			acc := make([]float64, hd)
			for j := range s {
				p := s[j] / sum
				vj := v.Row(j)[h*hd : (h+1)*hd]
				for d := range acc {
					acc[d] += p * float64(vj[d])
				}
			}
			for d := range dst {
				dst[d] = float32(acc[d])
			}
		}
	})
	return out
}

// selfAttn is the layer's self-attention with the position added to the
// queries' and keys' input (not the values'), residual and post-norm.
func (a *attnLayer) selfAttn(x, pos *Rows) *Rows {
	qk := addRows(x, pos)
	o := a.o.Apply(attention(a.q.Apply(qk), a.k.Apply(qk), a.v.Apply(x), numHeads))
	return a.ln1.Apply(addRows(x, o))
}

// encoder is an AIFI layer: self-attention, then a GELU MLP, post-norm.
func (a *attnLayer) encoder(x, pos *Rows) *Rows {
	x = a.selfAttn(x, pos)
	f := a.fc2.Apply(actRows(a.fc1.Apply(x), ActGELU))
	return a.ln2.Apply(addRows(x, f))
}

// forward is one decoder layer: self-attention, deformable cross-attention
// over the three levels, a ReLU MLP, each post-norm.
func (L *decLayer) forward(x, pos, ref, value *Rows, shapes [numLevels][2]int) *Rows {
	x = L.self.selfAttn(x, pos)
	q := addRows(x, pos)
	off := L.offsets.Apply(q)
	aw := L.weights.Apply(q)
	var starts [numLevels]int
	for l := 1; l < numLevels; l++ {
		starts[l] = starts[l-1] + shapes[l-1][0]*shapes[l-1][1]
	}
	const hd = dModel / numHeads
	sampled := newRows(x.N, dModel)
	parallel(x.N, func(i int) {
		r := ref.Row(i)
		o := off.Row(i)
		w := aw.Row(i)
		for h := 0; h < numHeads; h++ {
			// Softmax over the head's levels x points.
			ws := w[h*numLevels*numPoints : (h+1)*numLevels*numPoints]
			mx := math.Inf(-1)
			for _, v := range ws {
				mx = max(mx, float64(v))
			}
			var p [numLevels * numPoints]float64
			var sum float64
			for k, v := range ws {
				p[k] = math.Exp(float64(v) - mx)
				sum += p[k]
			}
			dst := sampled.Row(i)[h*hd : (h+1)*hd]
			var acc [hd]float64
			for l := 0; l < numLevels; l++ {
				H, W := shapes[l][0], shapes[l][1]
				for pt := 0; pt < numPoints; pt++ {
					k := (h*numLevels+l)*numPoints + pt
					lx := float64(r[0]) + float64(o[2*k])/numPoints*float64(r[2])*0.5
					ly := float64(r[1]) + float64(o[2*k+1])/numPoints*float64(r[3])*0.5
					wt := p[l*numPoints+pt] / sum
					// grid_sample, bilinear, zeros, align_corners=False.
					px, py := lx*float64(W)-0.5, ly*float64(H)-0.5
					x0, y0 := int(math.Floor(px)), int(math.Floor(py))
					fx, fy := px-float64(x0), py-float64(y0)
					for _, c := range [4]struct {
						x, y int
						w    float64
					}{{x0, y0, (1 - fx) * (1 - fy)}, {x0 + 1, y0, fx * (1 - fy)}, {x0, y0 + 1, (1 - fx) * fy}, {x0 + 1, y0 + 1, fx * fy}} {
						if c.x < 0 || c.x >= W || c.y < 0 || c.y >= H {
							continue
						}
						vr := value.Row(starts[l] + c.y*W + c.x)[h*hd : (h+1)*hd]
						s := wt * c.w
						for d := range acc {
							acc[d] += s * float64(vr[d])
						}
					}
				}
			}
			for d := range dst {
				dst[d] = float32(acc[d])
			}
		}
	})
	x = L.lnCross.Apply(addRows(x, L.output.Apply(sampled)))
	f := L.fc2.Apply(actRows(L.fc1.Apply(x), ActReLU))
	return L.lnFinal.Apply(addRows(x, f))
}

// makeAnchors is generate_anchors: each level's cell centres with a side of
// 0.05*2^level, as logits, and whether each is inside (0.01, 0.99) on every
// coordinate. An invalid anchor's logits are float32's max, as the
// reference's.
func makeAnchors(shapes [numLevels][2]int) ([]float32, []bool) {
	var anchors []float32
	var valid []bool
	for l, s := range shapes {
		H, W := s[0], s[1]
		wh := float32(0.05 * math.Pow(2, float64(l)))
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				a := [4]float32{(float32(x) + 0.5) / float32(W), (float32(y) + 0.5) / float32(H), wh, wh}
				ok := true
				for _, v := range a {
					if !(v > 0.01 && v < 0.99) {
						ok = false
					}
				}
				valid = append(valid, ok)
				for _, v := range a {
					if ok {
						anchors = append(anchors, float32(math.Log(float64(v/(1-v)))))
					} else {
						anchors = append(anchors, math.MaxFloat32)
					}
				}
			}
		}
	}
	return anchors, valid
}

// topQueries is the k memory rows with the largest class logit, largest
// first (torch.topk, sorted).
func topQueries(class *Rows, k int) []int {
	best := make([]float32, class.N)
	idx := make([]int, class.N)
	for i := range best {
		row := class.Row(i)
		b := row[0]
		for _, v := range row[1:] {
			b = max(b, v)
		}
		best[i], idx[i] = b, i
	}
	sort.SliceStable(idx, func(a, b int) bool { return best[idx[a]] > best[idx[b]] })
	return idx[:k]
}

// matmul is a [N, K] x [K, M] product.
func matmul(a, b *Rows) *Rows {
	out := newRows(a.N, b.D)
	parallel(a.N, func(i int) {
		dst := out.Row(i)
		for k, w := range a.Row(i) {
			src := b.Row(k)
			for j, v := range src {
				dst[j] += w * v
			}
		}
	})
	return out
}

// maskBox is mask_to_box_coordinate over one mask's logits > 0: the
// normalised cx, cy, w, h of its bounding box (x_max and y_max exclusive), or
// zeros if it is empty.
func maskBox(mask []float32, h, w int) [4]float64 {
	x0, y0, x1, y1 := w, h, -1, -1
	for y := 0; y < h; y++ {
		row := mask[y*w : (y+1)*w]
		for x, v := range row {
			if v > 0 {
				x0, y0 = min(x0, x), min(y0, y)
				x1, y1 = max(x1, x), max(y1, y)
			}
		}
	}
	if x1 < 0 {
		return [4]float64{}
	}
	// In float32, as the reference divides.
	fx0, fy0 := float32(x0)/float32(w), float32(y0)/float32(h)
	fx1, fy1 := float32(x1+1)/float32(w), float32(y1+1)/float32(h)
	return [4]float64{float64((fx0 + fx1) / 2), float64((fy0 + fy1) / 2), float64(fx1 - fx0), float64(fy1 - fy0)}
}
