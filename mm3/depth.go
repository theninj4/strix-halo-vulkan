package mm3

// The local LM (N2, upstream's rvq_depth_decoder): within a frame it samples
// the seven residual codes c1..c7 from the global LM's hidden and the
// frame's semantic code, and its seven hiddens are six-sevenths of the
// frame's condition.
//
// Upstream re-runs the whole depth sequence (at most 8 positions) for every
// code; this keeps a KV cache, so a frame is 7 passes: positions 0 and 1
// (the projected global hidden and the projected semantic-code embedding)
// as one pass of 4 rows, then one position a pass. Every pass is both CFG
// rows; a row's slot is its CFG row. The model has no rope (a learned table
// of 16 positions is added to the projected input) and no q/k norm, so the
// k and v projections write straight into fp32 caches, [pos][slot][4096],
// and mm3_depth_attn reads them.
//
// The weights are the global LM's banks' layout (fp16 or int8 fragment
// tiles); the seven heads stay fp16.

import (
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

const (
	dHeads   = 16
	dHD      = 256
	dFFN     = 6144
	dLayers  = 4
	dPos     = 16 // pos_embedding's rows
	dCodes   = 7  // c1..c7, one head each
	dVocab   = 1024
	dMaxRows = 4 // the first pass: positions 0 and 1, both slots
	dSlots   = 2
	dEps     = 1e-6
)

// depth projections: the four of attention, three of the feed forward, and
// the input projection, which runs once a pass before the layers.
const (
	dQ proj = iota
	dK
	dV
	dO
	dGate
	dUp
	dDown
	dIn
	dNProj
)

var dShape = [dNProj][2]int{
	dQ: {hidden, hidden}, dK: {hidden, hidden}, dV: {hidden, hidden}, dO: {hidden, hidden},
	dGate: {dFFN, hidden}, dUp: {dFFN, hidden}, dDown: {hidden, dFFN}, dIn: {hidden, hidden},
}

// dSlabs is each projection's split-K: K is 256 tiles, or 384 for down.
// int8 takes k32 throughout, as the global LM does (N3: a frame at k8 42.0
// ms, k16 40.5, k1 27.7, k32 25.4).
var (
	dSlabs   = [dNProj]int{dQ: 8, dK: 8, dV: 8, dO: 8, dGate: 4, dUp: 4, dDown: 8, dIn: 8}
	dSlabsQ8 = [dNProj]int{dQ: 32, dK: 32, dV: 32, dO: 32, dGate: 32, dUp: 32, dDown: 32, dIn: 32}
)

// headSlabs is the heads' split-K (1024 rows: 64 workgroups a slab).
const headSlabs = 8

type depthLayer struct {
	off               [dNProj - 1]uint32 // in the bank
	attnNorm, ffnNorm uint32             // fp32 arena
	kc, vc            uint32             // activation arena: [dPos][dSlots][4096]
}

// Depth is the local LM, resident.
type Depth struct {
	bankKind Bank
	dev      *vk.Device
	wbuf     *vk.Buffer
	abuf     *vk.Buffer
	hbuf     *vk.Buffer
	bank     *vk.Buffer
	headBank *vk.Buffer
	pipes    map[string]*vk.ComputePipeline
	mods     []*vk.ShaderModule

	l         [dLayers]depthLayer
	inOff     uint32 // the input projection, in the bank
	finalNorm uint32
	pos       []float32 // [dPos][4096]
	audioEmb  []uint16  // bf16 [7 * 1024][4096], on the host (the feedback)

	aX, aQ, aO, aGate, aUp, aFF, aPos, aMeta, aRes, aLogits, aUnscale uint32
	hA, hCtx, hFFN                                                    uint32
	ldaDim, ldaF                                                      int
}

// LoadDepth stages the local LM (models/MiniMax-Music3/rvq_depth_decoder).
func LoadDepth(dev *vk.Device, dir string, bank Bank) (*Depth, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	d := &Depth{bankKind: bank, dev: dev, pipes: map[string]*vk.ComputePipeline{},
		ldaDim: hidden + gemmPad, ldaF: dFFN + gemmPad}
	if err := d.layout(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.build(); err != nil {
		d.Destroy()
		return nil, err
	}
	if err := d.stage(set); err != nil {
		d.Destroy()
		return nil, err
	}
	return d, nil
}

func (d *Depth) layout() error {
	unit := 2
	if d.bankKind == BankQ8 {
		unit = 1
	}
	n := 0
	for i := range d.l {
		for p := dQ; p < dIn; p++ {
			d.l[i].off[p] = uint32(n / unit)
			n += d.bankKind.projBytes(dShape[p][0], dShape[p][1])
		}
	}
	d.inOff = uint32(n / unit)
	n += d.bankKind.projBytes(hidden, hidden)
	var err error
	if d.bank, err = d.dev.NewBuffer(n); err != nil {
		return fmt.Errorf("mm3: depth bank (%d MB): %w", n>>20, err)
	}
	if d.headBank, err = d.dev.NewBuffer(dCodes * dVocab * hidden * 2); err != nil {
		return err
	}

	w32 := 0
	for i := range d.l {
		d.l[i].attnNorm = uint32(w32)
		d.l[i].ffnNorm = uint32(w32 + hidden)
		w32 += 2 * hidden
	}
	d.finalNorm = uint32(w32)
	w32 += hidden
	if d.wbuf, err = d.dev.NewBuffer(w32 * 4); err != nil {
		return err
	}

	a := 0
	alloc := func(n int) uint32 {
		off := uint32(a)
		a += (n + 63) &^ 63
		return off
	}
	r := dMaxRows
	d.aX = alloc(r * hidden)
	d.aQ = alloc(r * hidden)
	d.aO = alloc(r * hidden)
	d.aGate = alloc(r * dFFN)
	d.aUp = alloc(r * dFFN)
	d.aFF = alloc(r * hidden)
	d.aPos = alloc(r * hidden)
	d.aMeta = alloc(r * 2)
	d.aRes = alloc(dMaxRows * 32 * dFFN)
	d.aLogits = alloc(dSlots * dVocab)
	d.aUnscale = alloc(hidden)
	for i := range d.l {
		d.l[i].kc = alloc(dPos * dSlots * hidden)
		d.l[i].vc = alloc(dPos * dSlots * hidden)
	}
	if d.abuf, err = d.dev.NewHostCachedBuffer(a * 4); err != nil {
		return err
	}
	unscale := make([]float32, hidden)
	for i := range unscale {
		unscale[i] = 1 / ffScale
	}
	d.abuf.WriteFloat32At(int(d.aUnscale), unscale)

	h := 0
	halloc := func(n int) uint32 {
		off := uint32(h)
		h += (n + 63) &^ 63
		return off
	}
	d.hA = halloc(r * d.ldaDim)
	d.hCtx = halloc(r * d.ldaDim)
	d.hFFN = halloc(r * d.ldaF)
	if d.hbuf, err = d.dev.NewHostCachedBuffer(h * 2); err != nil {
		return err
	}
	d.hbuf.Zero()
	return nil
}

func (d *Depth) pipeline(name string, spirv []byte, spec vk.PipelineSpec) error {
	mod, err := d.dev.NewShaderModule(spirv)
	if err != nil {
		return fmt.Errorf("mm3: shader %s: %w", name, err)
	}
	d.mods = append(d.mods, mod)
	p, err := d.dev.NewPipeline(mod, spec)
	if err != nil {
		return fmt.Errorf("mm3: pipeline %s: %w", name, err)
	}
	d.pipes[name] = p
	return nil
}

func (d *Depth) build() error {
	pc := uint32(pushBytes)
	bufs := []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank}
	for _, s := range []struct {
		name  string
		spirv []byte
	}{
		{"normf16", shaders.DiTNormScaleF16},
		{"swiglu", shaders.DiTSwiGLUF16},
		{"add", shaders.DiTGateAdd},
		{"attn", shaders.MM3DepthAttn},
	} {
		if err := d.pipeline(s.name, s.spirv, vk.PipelineSpec{Buffers: bufs, PushConstantSize: pc}); err != nil {
			return err
		}
	}
	gemvs, table := gemvSPIRV, dSlabs
	bb := bufs
	if d.bankKind == BankQ8 {
		gemvs, table = gemvQ8SPIRV, dSlabsQ8
		bb = []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank, d.abuf, d.bank}
	}
	used := map[int]bool{}
	for _, s := range table {
		used[s] = true
	}
	hb := []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.headBank}
	for r := 1; r <= dMaxRows; r++ {
		for s := range used {
			if err := d.pipeline(fmt.Sprintf("gemv_k%d_r%d", s, r), gemvs[s], vk.PipelineSpec{Buffers: bb,
				PushConstantSize: pc, SpecConstants: rowSpec(r), RequiredSubgroupSize: 64}); err != nil {
				return err
			}
		}
		if err := d.pipeline(fmt.Sprintf("head_r%d", r), gemvSPIRV[headSlabs], vk.PipelineSpec{Buffers: hb,
			PushConstantSize: pc, SpecConstants: rowSpec(r), RequiredSubgroupSize: 64}); err != nil {
			return err
		}
		for s, spirv := range gemvSumSPIRV {
			if err := d.pipeline(fmt.Sprintf("sum_k%d_r%d", s, r), spirv, vk.PipelineSpec{Buffers: bufs,
				PushConstantSize: pc, SpecConstants: rowSpec(r), RequiredSubgroupSize: 64}); err != nil {
				return err
			}
		}
	}
	return nil
}

func (d *Depth) stage(set *safetensors.Set) error {
	get := func(name string, shape ...int) ([]float32, error) {
		t, err := set.Get(name)
		if err != nil {
			return nil, err
		}
		if len(t.Shape) != len(shape) {
			return nil, fmt.Errorf("mm3: %s is %v, want %v", name, t.Shape, shape)
		}
		for i := range shape {
			if t.Shape[i] != shape[i] {
				return nil, fmt.Errorf("mm3: %s is %v, want %v", name, t.Shape, shape)
			}
		}
		return t.F32(nil)
	}
	put := func(off uint32, w []float32, n, k int) {
		if d.bankKind == BankQ8 {
			q, sc := tileBQ8(w, n, k)
			d.bank.WriteBytesAt(int(off), q)
			d.bank.WriteUint16At((int(off)+len(q))/2, sc)
			return
		}
		buf := make([]uint16, n*k)
		tileB(buf, n, k, func(r int, dst []float32) { copy(dst, w[r*k:(r+1)*k]) })
		d.bank.WriteUint16At(int(off), buf)
	}
	names := [dNProj - 1]string{"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "gate_proj", "up_proj", "down_proj"}
	for i := range d.l {
		for p := dQ; p < dIn; p++ {
			n, k := dShape[p][0], dShape[p][1]
			w, err := get(fmt.Sprintf("layers.%d.%s.weight", i, names[p]), n, k)
			if err != nil {
				return err
			}
			put(d.l[i].off[p], w, n, k)
		}
		for _, nm := range []struct {
			name string
			off  uint32
		}{{"input_layernorm", d.l[i].attnNorm}, {"post_attention_layernorm", d.l[i].ffnNorm}} {
			w, err := get(fmt.Sprintf("layers.%d.%s.weight", i, nm.name), hidden)
			if err != nil {
				return err
			}
			d.wbuf.WriteFloat32At(int(nm.off), w)
		}
	}
	w, err := get("projection.weight", hidden, hidden)
	if err != nil {
		return err
	}
	put(d.inOff, w, hidden, hidden)
	if w, err = get("norm.weight", hidden); err != nil {
		return err
	}
	d.wbuf.WriteFloat32At(int(d.finalNorm), w)
	if d.pos, err = get("pos_embedding.weight", dPos, hidden); err != nil {
		return err
	}
	heads := make([]uint16, dCodes*dVocab*hidden)
	for i := 0; i < dCodes; i++ {
		w, err := get(fmt.Sprintf("audio_heads.%d.weight", i), dVocab, hidden)
		if err != nil {
			return err
		}
		tileB(heads[i*dVocab*hidden:(i+1)*dVocab*hidden], dVocab, hidden, func(r int, dst []float32) {
			copy(dst, w[r*hidden:(r+1)*hidden])
		})
	}
	d.headBank.WriteUint16At(0, heads)
	emb, err := set.Get("audio_embeddings.weight")
	if err != nil {
		return err
	}
	if len(emb.Shape) != 2 || emb.Shape[0] != dCodes*dVocab || emb.Shape[1] != hidden || emb.DType != safetensors.BF16 {
		return fmt.Errorf("mm3: audio_embeddings is %v %s", emb.Shape, emb.DType)
	}
	d.audioEmb = make([]uint16, dCodes*dVocab*hidden)
	for i := range d.audioEmb {
		d.audioEmb[i] = binary.LittleEndian.Uint16(emb.Data[2*i:])
	}
	return nil
}

// Destroy releases every Vulkan object.
func (d *Depth) Destroy() {
	for _, p := range d.pipes {
		p.Destroy()
	}
	for _, m := range d.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{d.wbuf, d.abuf, d.hbuf, d.bank, d.headBank} {
		if b != nil {
			b.Destroy()
		}
	}
}

func (d *Depth) slabs(p proj) int {
	if d.bankKind == BankQ8 {
		return dSlabsQ8[p]
	}
	return dSlabs[p]
}

// gemv is one projection at n rows (split-K partials and their sum).
func (d *Depth) gemv(pipe string, slabs int, aOff uint32, lda int, cOff, bOff uint32, nOut, k, n int) []vk.MultiDispatch {
	kt := k / tile
	if kt%slabs != 0 || (kt/slabs)%4 != 0 {
		panic(fmt.Sprintf("mm3: %d slabs do not split K %d", slabs, k))
	}
	p := gemvPush{resOff: d.aRes, xnOff: aOff, outOff: cOff, bOff: bOff, lda: uint32(lda), gemmN: uint32(nOut), gemmK: uint32(k)}
	out := []vk.MultiDispatch{{Pipeline: d.pipes[pipe], GroupsX: uint32(slabs), GroupsY: uint32(nOut / tile), PushConstants: p.bytes()}}
	if slabs > 1 {
		out = append(out, vk.MultiDispatch{Pipeline: d.pipes[fmt.Sprintf("sum_k%d_r%d", slabs, n)],
			GroupsX: groups(nOut, 64), GroupsY: uint32(n), PushConstants: p.bytes()})
	}
	return out
}

// passGraph is one pass of n rows at positions [p0, p0 + n/2): the input
// projection and position add, the four layers, then the final norm and
// head code over the last two rows (the newest position, both slots).
func (d *Depth) passGraph(n, p0, code int) []vk.MultiDispatch {
	eps := math.Float32bits(dEps)
	var out []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		out = append(out, vk.MultiDispatch{Pipeline: d.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	proj := func(p proj, bOff uint32, aOff uint32, lda int, cOff uint32) {
		s := d.slabs(p)
		out = append(out, d.gemv(fmt.Sprintf("gemv_k%d_r%d", s, n), s, aOff, lda, cOff, bOff, dShape[p][0], dShape[p][1], n)...)
	}
	norm := func(wOff uint32) {
		add("normf16", uint32(n), 1, push{InOff: d.aX, OutOff: d.hA, WOff: wOff, Tokens: uint32(n), Dim: hidden,
			LDA: uint32(d.ldaDim), Eps: eps})
	}
	proj(dIn, d.inOff, d.hA, d.ldaDim, d.aX)
	add("add", uint32(n), 1, push{InOff: d.aPos, OutOff: d.aX, Tokens: uint32(n), Dim: hidden})
	cacheRow := uint32(p0 * dSlots * hidden)
	for i := range d.l {
		l := &d.l[i]
		norm(l.attnNorm)
		proj(dQ, l.off[dQ], d.hA, d.ldaDim, d.aQ)
		proj(dK, l.off[dK], d.hA, d.ldaDim, l.kc+cacheRow)
		proj(dV, l.off[dV], d.hA, d.ldaDim, l.vc+cacheRow)
		add("attn", uint32(n), dHeads, push{InOff: d.aQ, OutOff: d.hCtx, KOff: l.kc, VOff: l.vc, Aux0: d.aMeta,
			Span: dSlots, Scale: math.Float32bits(float32(1 / math.Sqrt(dHD))), Tokens: uint32(n), LDA: uint32(d.ldaDim)})
		proj(dO, l.off[dO], d.hCtx, d.ldaDim, d.aO)
		add("add", uint32(n), 1, push{InOff: d.aO, OutOff: d.aX, Tokens: uint32(n), Dim: hidden})
		norm(l.ffnNorm)
		proj(dGate, l.off[dGate], d.hA, d.ldaDim, d.aGate)
		proj(dUp, l.off[dUp], d.hA, d.ldaDim, d.aUp)
		add("swiglu", uint32(n), 1, push{InOff: d.aGate, KOff: d.aUp, OutOff: d.hFFN, Tokens: uint32(n), Dim: dFFN,
			LDA: uint32(d.ldaF), Scale: math.Float32bits(ffScale)})
		proj(dDown, l.off[dDown], d.hFFN, d.ldaF, d.aFF)
		add("add", uint32(n), 1, push{InOff: d.aFF, OutOff: d.aX, Tokens: uint32(n), Dim: hidden, Aux0: d.aUnscale, Aux2: 1})
	}
	// The newest position's two rows: final norm into hA rows 0-1, head.
	last := uint32((n - dSlots) * hidden)
	add("normf16", dSlots, 1, push{InOff: d.aX + last, OutOff: d.hA, WOff: d.finalNorm, Tokens: dSlots, Dim: hidden,
		LDA: uint32(d.ldaDim), Eps: eps})
	return append(out, d.gemv(fmt.Sprintf("head_r%d", dSlots), headSlabs, d.hA, d.ldaDim, d.aLogits,
		uint32(code*dVocab*hidden), dVocab, hidden, dSlots)...)
}

// DepthOut is a frame's local-LM result.
type DepthOut struct {
	Codes  [dCodes]int       // c1..c7
	Hidden [dCodes][]float32 // the conditional row's hidden before each code's head
	// Logits is each code's head over its 1024 codes, conditional row then
	// unconditional.
	Logits [dCodes][dSlots][]float32
}

// Frame runs the local LM for one frame. last is the global LM's hidden for
// the two CFG rows; sem is the semantic code's embedding (the global LM's
// table, LM.Embed(CodeOffset + code)). pick chooses code i (0 for c1) from
// its head's logits; a teacher-forced gate returns the oracle's.
func (d *Depth) Frame(last [dSlots][]float32, sem []float32, pick func(i int, cond, uncond []float32) int) (*DepthOut, time.Duration, error) {
	out := &DepthOut{}
	var total time.Duration
	for i := 0; i < dCodes; i++ {
		// Inputs: position 0 is each row's global hidden, position 1 the
		// semantic code's embedding, position p > 1 code c_{p-1}'s.
		var in [][]float32
		var meta []uint32
		p0, np := i+1, 1
		if i == 0 {
			p0, np = 0, 2
		}
		for p := p0; p < p0+np; p++ {
			for s := 0; s < dSlots; s++ {
				switch {
				case p == 0:
					in = append(in, last[s])
				case p == 1:
					in = append(in, sem)
				default:
					in = append(in, d.codeEmbed(p-2, out.Codes[p-2]))
				}
				meta = append(meta, uint32(s), uint32(p))
			}
		}
		n := len(in)
		h := make([]uint16, n*d.ldaDim)
		pos := make([]float32, n*hidden)
		for r, v := range in {
			if len(v) != hidden {
				return nil, total, fmt.Errorf("mm3: depth input %d is %d wide", r, len(v))
			}
			for j, x := range v {
				h[r*d.ldaDim+j] = safetensors.F32ToF16(x)
			}
			copy(pos[r*hidden:], d.pos[int(meta[2*r+1])*hidden:(int(meta[2*r+1])+1)*hidden])
		}
		d.hbuf.WriteUint16At(int(d.hA), h)
		d.abuf.WriteFloat32At(int(d.aPos), pos)
		d.abuf.WriteUint32At(int(d.aMeta), meta)
		dur, err := vk.DispatchMultiTimed(d.passGraph(n, p0, i), 1, 1, true)
		if err != nil {
			return nil, total, fmt.Errorf("mm3: depth pass %d: %w", i, err)
		}
		total += dur
		lg := d.abuf.ReadFloat32At(int(d.aLogits), dSlots*dVocab)
		out.Logits[i] = [dSlots][]float32{lg[:dVocab], lg[dVocab:]}
		hh := d.hbuf.ReadUint16At(int(d.hA), hidden)
		out.Hidden[i] = make([]float32, hidden)
		for j, v := range hh {
			out.Hidden[i][j] = safetensors.F16ToF32(v)
		}
		c := pick(i, out.Logits[i][0], out.Logits[i][1])
		if c < 0 || c >= dVocab {
			return nil, total, fmt.Errorf("mm3: code %d of c%d", c, i+1)
		}
		out.Codes[i] = c
	}
	return out, total, nil
}

// codeEmbed is residual code c_{i+1}'s embedding (audio_embeddings at
// i * 1024 + code).
func (d *Depth) codeEmbed(i, code int) []float32 {
	row := d.audioEmb[(i*dVocab+code)*hidden : (i*dVocab+code+1)*hidden]
	out := make([]float32, hidden)
	for j, v := range row {
		out[j] = bf16(v)
	}
	return out
}

// Feedback is the global LM's next input for a frame: the semantic code's
// embedding plus the seven residual codes', times 8^-1/2.
func (d *Depth) Feedback(sem []float32, codes [dCodes]int) []float32 {
	out := make([]float32, hidden)
	copy(out, sem)
	for i, c := range codes {
		row := d.audioEmb[(i*dVocab+c)*hidden : (i*dVocab+c+1)*hidden]
		for j, v := range row {
			out[j] += bf16(v)
		}
	}
	s := float32(1 / math.Sqrt(dCodes+1))
	for j := range out {
		out[j] *= s
	}
	return out
}
