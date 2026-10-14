package breeze

// The text encoder on the device (B2): T5Gemma2's encoder, 26 sandwich-normed
// layers at 1152 (4 q heads and 1 kv head of 256, q/k norms, GELU-tanh GLU
// at 6912), bidirectional, layers 5, 11, 17 and 23 full and the rest a
// 512-token window; then `text_encoder_proj` to the backbone's 2048. Each
// text segment of a prompt is encoded on its own from position 0
// (BreezeForConditionalGeneration.convert_input_ids_to_embeds).
//
// It is Rune's vision tower's arrangement (gemma4/vision.go R10) at another
// head shape: the DiT's fp16 GEMM for the projections, dit_gemm's GLU_GELU
// epilogue for gate and up, gemma4_vit's JOIN for the sandwich residual and
// the next norm in one dispatch. T5Gemma2's norms are (1 + w), so the
// weights are staged as 1 + w and the kernels' plain `w` reads them.
// breeze_enc.comp has the prep and the attention.
//
// The embedding table (302 M rows of bf16, 604 MB) stays on the host: a
// segment gathers a few dozen rows.

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
	encHidden = 1152
	encNQ     = 4
	encHD     = 256
	encQKV    = (encNQ + 2) * encHD
	encFFN    = 6912
	encLayers = 26
	encEps    = 1e-6
	encWindow = 512
	encBM     = 64 // dit_gemm_reg64's tile, and the GEGLU build's
	encLdaPad = 128
	gluGroup  = 32 // dit_gemm C_SWIGLU's gate/up interleave
	eoiToken  = 256000
)

// encFull is layer_types: every sixth layer attends to the whole segment.
func encFull(i int) bool { return i%6 == 5 }

type encLayer struct {
	preAttn, postAttn, preFF, postFF, qk uint32 // weight arena
	qkv, o, gu, down                     uint32 // bank, halves
}

// Encoder is the text encoder and its projection, resident.
type Encoder struct {
	dev   *vk.Device
	wbuf  *vk.Buffer
	abuf  *vk.Buffer
	hbuf  *vk.Buffer
	bank  *vk.Buffer
	pipes map[string]*vk.ComputePipeline
	mods  []*vk.ShaderModule

	embed []uint16 // bf16 [vocab][1152]
	eoi   []float32
	vocab int
	scale float32 // the embedding's sqrt(1152)

	maxTokens                      int
	layers                         []encLayer
	finalNorm, ropeSlide, ropeFull uint32
	proj                           uint32
	aX, aY, aP, aOut               uint32
	hA, hF, hQ, hK, hV             uint32
	lda, ldF, plane                int
}

// LoadEncoder stages the encoder for segments of up to maxTokens tokens.
func LoadEncoder(dev *vk.Device, dir string, maxTokens int) (*Encoder, error) {
	if maxTokens <= 0 || maxTokens > 2048 {
		return nil, fmt.Errorf("breeze: an encoder for %d tokens; 1 to 2048 (breeze_enc ATTN's MAXK)", maxTokens)
	}
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	e := &Encoder{dev: dev, pipes: map[string]*vk.ComputePipeline{}, maxTokens: maxTokens,
		scale: float32(math.Sqrt(encHidden))}
	if err := e.stage(set); err != nil {
		e.Destroy()
		return nil, err
	}
	if err := e.build(); err != nil {
		e.Destroy()
		return nil, err
	}
	return e, nil
}

func (e *Encoder) stage(set *safetensors.Set) error {
	emb, err := set.Get("text_encoder.embed_tokens.weight")
	if err != nil {
		return err
	}
	if len(emb.Shape) != 2 || emb.Shape[1] != encHidden || emb.DType != safetensors.BF16 {
		return fmt.Errorf("breeze: text embedding is %v %s", emb.Shape, emb.DType)
	}
	e.vocab = emb.Shape[0]
	e.embed = make([]uint16, e.vocab*encHidden)
	for i := range e.embed {
		e.embed[i] = binary.LittleEndian.Uint16(emb.Data[2*i:])
	}
	eoi, err := get(set, "text_encoder.embed_tokens.eoi_embedding", encHidden)
	if err != nil {
		return err
	}
	e.eoi = make([]float32, encHidden)
	for i := range e.eoi {
		e.eoi[i] = bfAt(eoi, i)
	}

	var w []float32
	walloc := func(v []float32) uint32 {
		off := uint32(len(w))
		w = append(w, v...)
		for len(w)%64 != 0 {
			w = append(w, 0)
		}
		return off
	}
	onePlus := func(name string, n int) ([]float32, error) {
		t, err := get(set, name, n)
		if err != nil {
			return nil, err
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = 1 + bfAt(t, i)
		}
		return v, nil
	}
	type pend struct {
		off  uint32
		rows func(i int, dst []float32)
		n, k int
	}
	var pending []pend
	bank := 0
	place := func(n, k int, rows func(i int, dst []float32)) uint32 {
		off := uint32(bank)
		bank += n * k
		pending = append(pending, pend{off, rows, n, k})
		return off
	}
	matrix := func(name string, n, k int) (*safetensors.Tensor, error) { return get(set, name, n, k) }

	e.layers = make([]encLayer, encLayers)
	for i := range e.layers {
		l := &e.layers[i]
		p := fmt.Sprintf("text_encoder.layers.%d.", i)
		for _, nw := range []struct {
			dst  *uint32
			name string
			n    int
		}{
			{&l.preAttn, "pre_self_attn_layernorm.weight", encHidden},
			{&l.postAttn, "post_self_attn_layernorm.weight", encHidden},
			{&l.preFF, "pre_feedforward_layernorm.weight", encHidden},
			{&l.postFF, "post_feedforward_layernorm.weight", encHidden},
		} {
			v, err := onePlus(p+nw.name, nw.n)
			if err != nil {
				return err
			}
			*nw.dst = walloc(v)
		}
		qn, err := onePlus(p+"self_attn.q_norm.weight", encHD)
		if err != nil {
			return err
		}
		kn, err := onePlus(p+"self_attn.k_norm.weight", encHD)
		if err != nil {
			return err
		}
		l.qk = walloc(append(qn, kn...))

		q, err := matrix(p+"self_attn.q_proj.weight", encNQ*encHD, encHidden)
		if err != nil {
			return err
		}
		k, err := matrix(p+"self_attn.k_proj.weight", encHD, encHidden)
		if err != nil {
			return err
		}
		v, err := matrix(p+"self_attn.v_proj.weight", encHD, encHidden)
		if err != nil {
			return err
		}
		l.qkv = place(encQKV, encHidden, func(i int, dst []float32) {
			src, r := q, i
			switch {
			case i >= (encNQ+1)*encHD:
				src, r = v, i-(encNQ+1)*encHD
			case i >= encNQ*encHD:
				src, r = k, i-encNQ*encHD
			}
			for j := range dst {
				dst[j] = bfAt(src, r*encHidden+j)
			}
		})
		o, err := matrix(p+"self_attn.o_proj.weight", encHidden, encNQ*encHD)
		if err != nil {
			return err
		}
		l.o = place(encHidden, encNQ*encHD, rowsOf(o, encNQ*encHD))
		gate, err := matrix(p+"mlp.gate_proj.weight", encFFN, encHidden)
		if err != nil {
			return err
		}
		up, err := matrix(p+"mlp.up_proj.weight", encFFN, encHidden)
		if err != nil {
			return err
		}
		// gate and up interleaved 32 rows at a time: the GLU epilogue's contract.
		l.gu = place(2*encFFN, encHidden, func(i int, dst []float32) {
			g, r := i/(2*gluGroup), i%(2*gluGroup)
			src := gate
			if r >= gluGroup {
				src, r = up, r-gluGroup
			}
			row := g*gluGroup + r
			for j := range dst {
				dst[j] = bfAt(src, row*encHidden+j)
			}
		})
		down, err := matrix(p+"mlp.down_proj.weight", encHidden, encFFN)
		if err != nil {
			return err
		}
		l.down = place(encHidden, encFFN, rowsOf(down, encFFN))
	}
	fn, err := onePlus("text_encoder.norm.weight", encHidden)
	if err != nil {
		return err
	}
	e.finalNorm = walloc(fn)
	pj, err := matrix("text_encoder_proj.weight", bbHidden, encHidden)
	if err != nil {
		return err
	}
	e.proj = place(bbHidden, encHidden, rowsOf(pj, encHidden))

	// The ropes: sliding layers theta 1e4, full layers theta 1e6 with linear
	// factor 8 (inv_freq / 8), cos then sin, [pos][128].
	for _, r := range []struct {
		dst          *uint32
		theta, scale float64
	}{{&e.ropeSlide, 1e4, 1}, {&e.ropeFull, 1e6, 8}} {
		inv := make([]float32, encHD/2)
		for i := range inv {
			inv[i] = float32(1/math.Pow(r.theta, float64(2*i)/encHD)) / float32(r.scale)
		}
		tab := make([]float32, 2*e.maxTokens*encHD/2)
		for p := 0; p < e.maxTokens; p++ {
			for i, f := range inv {
				a := float64(float32(p) * f)
				tab[p*encHD/2+i] = float32(math.Cos(a))
				tab[(e.maxTokens+p)*encHD/2+i] = float32(math.Sin(a))
			}
		}
		*r.dst = walloc(tab)
	}

	if e.wbuf, err = e.dev.NewBuffer(len(w) * 4); err != nil {
		return err
	}
	e.wbuf.WriteFloat32(w)
	if e.bank, err = e.dev.NewBuffer(bank * 2); err != nil {
		return fmt.Errorf("breeze: encoder bank (%d MB): %w", bank*2>>20, err)
	}
	for _, s := range pending {
		buf := make([]uint16, s.n*s.k)
		tileB(buf, s.n, s.k, s.rows)
		e.bank.WriteUint16At(int(s.off), buf)
	}

	R := roundUp(e.maxTokens, encBM)
	acts := 0
	aalloc := func(n int) uint32 {
		off := uint32(acts)
		acts += roundUp(n, 64)
		return off
	}
	e.aX = aalloc(R * encHidden)
	e.aY = aalloc(R * encHidden)
	e.aP = aalloc(R * encQKV)
	e.aOut = aalloc(R * bbHidden)
	if e.abuf, err = e.dev.NewHostCachedBuffer(acts * 4); err != nil {
		return err
	}
	e.lda = encHidden + encLdaPad // also holds the 1024-wide attention context
	e.ldF = encFFN + encLdaPad
	e.plane = R
	halves := 0
	halloc := func(n int) uint32 {
		off := uint32(halves)
		halves += roundUp(n, 64)
		return off
	}
	e.hA = halloc(R * e.lda)
	e.hF = halloc(R * e.ldF)
	e.hQ = halloc(encNQ * R * encHD)
	e.hK = halloc(R * encHD)
	e.hV = halloc(R * encHD)
	if e.hbuf, err = e.dev.NewBuffer(halves * 2); err != nil {
		return err
	}
	e.hbuf.Zero()
	return nil
}

// rowsOf reads row i of a [n][k] bf16 tensor.
func rowsOf(t *safetensors.Tensor, k int) func(i int, dst []float32) {
	return func(i int, dst []float32) {
		for j := range dst {
			dst[j] = bfAt(t, i*k+j)
		}
	}
}

func (e *Encoder) build() error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{e.wbuf, e.abuf, e.hbuf}
	withBank := []*vk.Buffer{e.wbuf, e.abuf, e.hbuf, e.bank}
	for _, p := range []struct {
		name  string
		spirv []byte
		bufs  []*vk.Buffer
		wave  uint32
	}{
		{"normf16", shaders.DiTNormScaleF16, base, 0},
		{"join", shaders.Gemma4ViTJoin, base, 0},
		{"prep", shaders.BreezeEncPrep, base, 0},
		{"attn", shaders.BreezeEncAttn, base, 0},
		{"gemm", shaders.DiTGEMMReg64Tiled, withBank, 0},
		{"geglu", shaders.Gemma4ViTGEMMGEGLU, withBank, 0},
	} {
		mod, err := e.dev.NewShaderModule(p.spirv)
		if err != nil {
			return fmt.Errorf("breeze: encoder shader %s: %w", p.name, err)
		}
		e.mods = append(e.mods, mod)
		pipe, err := e.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: p.bufs, PushConstantSize: pc, RequiredSubgroupSize: p.wave})
		if err != nil {
			return fmt.Errorf("breeze: encoder pipeline %s: %w", p.name, err)
		}
		e.pipes[p.name] = pipe
	}
	return nil
}

// Destroy releases the encoder's device objects.
func (e *Encoder) Destroy() {
	for _, p := range e.pipes {
		p.Destroy()
	}
	for _, m := range e.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{e.wbuf, e.abuf, e.hbuf, e.bank} {
		if b != nil {
			b.Destroy()
		}
	}
}

// Encode runs one text segment (ids from position 0) through the encoder
// and the projection: [n][2048] fp32, the backbone's input rows. If states
// is set it also gets the encoder's normed output ([n][1152], the fp16 the
// projection reads), for the gate.
func (e *Encoder) Encode(ids []int32, states *[]float32) ([]float32, time.Duration, error) {
	n := len(ids)
	if n == 0 || n > e.maxTokens {
		return nil, 0, fmt.Errorf("breeze: a text segment of %d tokens; 1 to %d", n, e.maxTokens)
	}
	x := make([]float32, n*encHidden)
	for r, id := range ids {
		if id < 0 || int(id) >= e.vocab {
			return nil, 0, fmt.Errorf("breeze: token %d outside the encoder's vocabulary", id)
		}
		row := x[r*encHidden : (r+1)*encHidden]
		if id == eoiToken {
			copy(row, e.eoi)
			continue
		}
		src := e.embed[int(id)*encHidden:]
		for j := range row {
			row[j] = bf16(src[j]) * e.scale
		}
	}
	e.abuf.WriteFloat32At(int(e.aX), x)

	var d []vk.MultiDispatch
	add := func(pipe string, gx, gy uint32, p push) {
		d = append(d, vk.MultiDispatch{Pipeline: e.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: p.bytes()})
	}
	mPad := roundUp(n, encBM)
	gemm := func(b, aOff, cOff uint32, nOut, k, lda int) {
		add("gemm", uint32(nOut/encBM), uint32(mPad/encBM), push{InOff: aOff, OutOff: cOff, BOff: b,
			GemmM: uint32(mPad), GemmN: uint32(nOut), GemmK: uint32(k), LDA: uint32(lda)})
	}
	eps := math.Float32bits(encEps)
	join := func(y, w, next uint32) {
		add("join", uint32(n), 1, push{InOff: y, OutOff: e.aX, WOff: w, KOff: next, BOff: e.hA,
			LDA: uint32(e.lda), Tokens: uint32(n), Dim: encHidden, Eps: eps})
	}
	add("normf16", uint32(n), 1, push{InOff: e.aX, OutOff: e.hA, WOff: e.layers[0].preAttn, Tokens: uint32(n),
		Dim: encHidden, LDA: uint32(e.lda), Eps: eps})
	for i, l := range e.layers {
		rope, slide := e.ropeSlide, uint32(1)
		if encFull(i) {
			rope, slide = e.ropeFull, 0
		}
		gemm(l.qkv, e.hA, e.aP, encQKV, encHidden, e.lda)
		add("prep", uint32(n), encNQ+2, push{InOff: e.aP, Aux0: encQKV, WOff: l.qk, BOff: rope,
			KStride: uint32(e.maxTokens), OutOff: e.hQ, KOff: e.hK, VOff: e.hV, Aux1: uint32(e.plane),
			Tokens: uint32(n), Eps: eps, Scale: math.Float32bits(float32(1 / math.Sqrt(encHD)))})
		add("attn", uint32(n), encNQ, push{InOff: e.hQ, KOff: e.hK, VOff: e.hV, OutOff: e.hA, LDA: uint32(e.lda),
			Aux1: uint32(e.plane), Aux2: slide, Tokens: uint32(n)})
		gemm(l.o, e.hA, e.aY, encHidden, encNQ*encHD, e.lda)
		join(e.aY, l.postAttn, l.preFF)
		add("geglu", uint32(2*encFFN/encBM), uint32(mPad/encBM), push{InOff: e.hA, OutOff: e.hF, BOff: l.gu,
			GemmM: uint32(mPad), GemmN: 2 * encFFN, GemmK: encHidden, LDA: uint32(e.lda),
			Aux0: uint32(e.ldF), Scale: math.Float32bits(1)})
		gemm(l.down, e.hF, e.aY, encHidden, encFFN, e.ldF)
		next := e.finalNorm
		if i+1 < len(e.layers) {
			next = e.layers[i+1].preAttn
		}
		join(e.aY, l.postFF, next)
	}
	gemm(e.proj, e.hA, e.aOut, bbHidden, encHidden, e.lda)

	var took time.Duration
	for i := 0; i < len(d); i += 64 {
		dt, err := vk.DispatchMultiTimed(d[i:min(i+64, len(d))], 1, 1, true)
		if err != nil {
			return nil, took, fmt.Errorf("breeze: encoder dispatch %d: %w", i, err)
		}
		took += dt
	}
	if states != nil {
		s := make([]float32, n*encHidden)
		for r := 0; r < n; r++ {
			h := e.hbuf.ReadUint16At(int(e.hA)+r*e.lda, encHidden)
			for j, v := range h {
				s[r*encHidden+j] = safetensors.F16ToF32(v)
			}
		}
		*states = s
	}
	return e.abuf.ReadFloat32At(int(e.aOut), n*bbHidden), took, nil
}
