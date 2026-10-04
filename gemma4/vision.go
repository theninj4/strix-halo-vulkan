package gemma4

// Rune's image input (research/rune-vertical.md R10): Gemma 4's vision tower
// and projector, from pixels to the rows that replace a prompt's image
// tokens.
//
// The processor is transformers' Gemma4ImageProcessor: RGB (alpha dropped),
// an aspect-preserving resize to the largest size whose sides are multiples
// of 48 (patch 16 x pool 3) inside 280 x 9 = 2520 patches, torch's uint8
// antialiased bicubic (llm/pixels.Resize), x/255, and 16x16 patches in
// (row, column, channel) order. The model's first step is 2(x - 0.5).
//
// The tower is 27 pre-and-post-normed layers at 1152 (16 heads of 72,
// q/k/v norms, a 2-D rope over the patch's x and y, scale 1, GEGLU-tanh
// 4304), bidirectional over one image's patches. Then a 3x3 average pool,
// x sqrt(1152), a fixed standardisation, an unscaled RMS norm and one
// projection to the text model's 2816: one row a soft token.
//
// It runs on the matrix cores in fp16 (the weights are bf16 in the
// checkpoint, ~1.1 GB as fp16), on kernels that are mostly other towers':
// the DiT's GEMM, and the Qwen ViT's WMMA attention, which pads 72-wide
// heads to 80 (qimage/vision explains the padding). gemma4_vit.comp has the
// rest.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// VisionConfig is config.json's vision_config, and the processor's numbers.
type VisionConfig struct {
	Hidden   int     `json:"hidden_size"`
	Inter    int     `json:"intermediate_size"`
	Layers   int     `json:"num_hidden_layers"`
	Heads    int     `json:"num_attention_heads"`
	KVHeads  int     `json:"num_key_value_heads"`
	HeadDim  int     `json:"head_dim"`
	Patch    int     `json:"patch_size"`
	Pool     int     `json:"pooling_kernel_size"`
	PosSize  int     `json:"position_embedding_size"`
	Eps      float64 `json:"rms_norm_eps"`
	Std      bool    `json:"standardize"`
	Clipped  bool    `json:"use_clipped_linears"`
	Act      string  `json:"hidden_activation"`
	MaxSoft  int     `json:"default_output_length"` // soft tokens an image, at most
	RopeBase float64
}

// LoadVisionConfig reads dir/config.json's vision_config and holds it to
// what this port implements.
func LoadVisionConfig(dir string) (*VisionConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var f struct {
		V *struct {
			VisionConfig
			Rope struct {
				Theta float64 `json:"rope_theta"`
			} `json:"rope_parameters"`
		} `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("gemma4: config.json: %w", err)
	}
	if f.V == nil {
		return nil, fmt.Errorf("gemma4: the checkpoint has no vision tower")
	}
	c := f.V.VisionConfig
	c.RopeBase = f.V.Rope.Theta
	switch {
	case c.Hidden != 1152 || c.Heads != 16 || c.HeadDim != 72 || c.KVHeads != c.Heads:
		return nil, fmt.Errorf("gemma4: vision tower %d x %d heads of %d: only 1152 x 16 x 72 is implemented", c.Hidden, c.Heads, c.HeadDim)
	case c.Pool != 3 || c.Patch != 16:
		return nil, fmt.Errorf("gemma4: vision pool %d, patch %d", c.Pool, c.Patch)
	case c.Clipped || !c.Std || c.Act != "gelu_pytorch_tanh":
		return nil, fmt.Errorf("gemma4: vision tower options not implemented (clipped %v, standardize %v, %s)", c.Clipped, c.Std, c.Act)
	}
	if c.MaxSoft == 0 {
		c.MaxSoft = 280
	}
	return &c, nil
}

// Patches is one image ready for the tower.
type Patches struct {
	Pixels []float32 // [N][3 * 16 * 16], the processor's pixel_values (x/255), unpadded
	GW, GH int       // the patch grid
	Soft   int       // soft tokens: N / 9
}

// N is the patch count.
func (p *Patches) N() int { return p.GW * p.GH }

// VisionSize is get_aspect_ratio_preserving_size: the (height, width) an
// image is resized to.
func VisionSize(h, w, patch, maxPatches, pool int) (int, int, error) {
	if h <= 0 || w <= 0 {
		return 0, 0, fmt.Errorf("gemma4: a %dx%d image", w, h)
	}
	targetPx := float64(maxPatches * patch * patch)
	factor := math.Sqrt(targetPx / float64(h*w))
	side := pool * patch
	th := int(math.Floor(factor*float64(h)/float64(side))) * side
	tw := int(math.Floor(factor*float64(w)/float64(side))) * side
	if th == 0 && tw == 0 {
		return 0, 0, fmt.Errorf("gemma4: a %dx%d image resizes to nothing", w, h)
	}
	maxSide := (maxPatches / (pool * pool)) * side
	if th == 0 {
		th = side
		tw = min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	} else if tw == 0 {
		tw = side
		th = min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	if float64(th*tw) > targetPx {
		return 0, 0, fmt.Errorf("gemma4: a %dx%d image resizes to %dx%d, past %d patches", w, h, tw, th, maxPatches)
	}
	return th, tw, nil
}

// Patchify is the processor over a decoded image.
func (c *VisionConfig) Patchify(img *pixels.RGB) (*Patches, error) {
	h, w, err := VisionSize(img.H, img.W, c.Patch, c.MaxSoft*c.Pool*c.Pool, c.Pool)
	if err != nil {
		return nil, err
	}
	r := img
	if h != img.H || w != img.W {
		r = pixels.Resize(img, w, h)
	}
	P := c.Patch
	p := &Patches{GW: w / P, GH: h / P}
	p.Soft = p.N() / (c.Pool * c.Pool)
	p.Pixels = make([]float32, p.N()*3*P*P)
	scale := float32(1.0 / 255)
	for py := range p.GH {
		for px := range p.GW {
			row := p.Pixels[(py*p.GW+px)*3*P*P:]
			for y := range P {
				for x := range P {
					src := r.Pix[((py*P+y)*w+px*P+x)*3:]
					for ch := range 3 {
						row[(y*P+x)*3+ch] = float32(src[ch]) * scale
					}
				}
			}
		}
	}
	return p, nil
}

const (
	vitBM, vitBN, vitBK = 64, 64, 64 // dit_gemm_reg64_bt16's tile
	vitHDPad            = 80
	vitLdaPad           = 128
	vitKeyBlock         = 64
	vitTokenAlign       = 128
	noW                 = 0xffffffff
)

type vitLayer struct {
	qkv, o, gu, down       uint32 // bank, halves
	in, postAttn, pre, pst uint32 // norms, wbuf
	qk                     uint32 // q_norm [72] then k_norm [72]
}

// Vision is the staged tower.
type Vision struct {
	Cfg *VisionConfig
	dev *vk.Device

	wbuf, abuf, hbuf, bank *vk.Buffer
	pipes                  map[string]*vk.ComputePipeline
	mods                   []*vk.ShaderModule
	layers                 []vitLayer
	patch, proj            uint32
	stdBias, stdScale      uint32
	posTable               *safetensors.Tensor // [2][PosSize][1152] bf16, read on the host
	textHidden             int

	maxRows, ffnPad, lda, planeTok int
	aX, aY, aP                     uint32
	aPos, aRope, aPool, aOut       uint32
	hA, hF, hQ, hK, hV             uint32
	ldF                            int
	wave                           uint32
}

const visionPre = "model.vision_tower."

// LoadVision stages the tower from an open checkpoint.
func LoadVision(dev *vk.Device, set *safetensors.Set, dir string, textHidden int) (*Vision, error) {
	c, err := LoadVisionConfig(dir)
	if err != nil {
		return nil, err
	}
	v := &Vision{Cfg: c, dev: dev, pipes: map[string]*vk.ComputePipeline{}, textHidden: textHidden}
	v.maxRows = c.MaxSoft * c.Pool * c.Pool
	v.ffnPad = roundUp(c.Inter, vitBN)
	if err := v.stage(set); err != nil {
		v.Destroy()
		return nil, err
	}
	if err := v.build(); err != nil {
		v.Destroy()
		return nil, err
	}
	return v, nil
}

func packF16(dst []uint16, w []float32, n, k int) {
	kt := k / tile
	for i := range n {
		row := w[i*k : (i+1)*k]
		base := (i / tile) * kt * tile * tile
		lane := (i % tile) * tile
		for j, x := range row {
			dst[base+(j/tile)*tile*tile+lane+j%tile] = safetensors.F32ToF16(x)
		}
	}
}

func (v *Vision) stage(set *safetensors.Set) error {
	c := v.Cfg
	D, hd, H := c.Hidden, c.HeadDim, c.Heads
	get := func(name string) ([]float32, error) {
		t, err := set.Get(name)
		if err != nil {
			return nil, err
		}
		return t.F32(nil)
	}
	var w []float32
	walloc := func(x []float32) uint32 {
		off := uint32(len(w))
		w = append(w, x...)
		for len(w)%64 != 0 {
			w = append(w, 0)
		}
		return off
	}
	type staged struct {
		off  int
		w    []float32
		n, k int
	}
	var pending []staged
	bankHalves := 0
	place := func(x []float32, n, k int) uint32 {
		off := bankHalves
		bankHalves += n * k
		pending = append(pending, staged{off, x, n, k})
		return uint32(off)
	}
	padRows := func(x []float32, n, k, nPad int) []float32 {
		out := make([]float32, nPad*k)
		copy(out, x[:n*k])
		return out
	}
	padCols := func(x []float32, n, k, kPad int) []float32 {
		out := make([]float32, n*kPad)
		for r := range n {
			copy(out[r*kPad:r*kPad+k], x[r*k:(r+1)*k])
		}
		return out
	}

	var err error
	if v.posTable, err = set.Get(visionPre + "patch_embedder.position_embedding_table"); err != nil {
		return err
	}
	pe, err := get(visionPre + "patch_embedder.input_proj.weight")
	if err != nil {
		return err
	}
	v.patch = place(pe, D, 3*c.Patch*c.Patch)
	sb, err := get(visionPre + "std_bias")
	if err != nil {
		return err
	}
	ss, err := get(visionPre + "std_scale")
	if err != nil {
		return err
	}
	v.stdBias, v.stdScale = walloc(sb), walloc(ss)
	pj, err := get("model.embed_vision.embedding_projection.weight")
	if err != nil {
		return err
	}
	v.proj = place(pj, v.textHidden, D)

	v.layers = make([]vitLayer, c.Layers)
	for i := range v.layers {
		l := &v.layers[i]
		p := fmt.Sprintf("%sencoder.layers.%d.", visionPre, i)
		var ws [13][]float32
		for k, name := range []string{"input_layernorm.weight", "post_attention_layernorm.weight",
			"pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
			"self_attn.q_norm.weight", "self_attn.k_norm.weight",
			"self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight",
			"self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight",
			"mlp.down_proj.linear.weight"} {
			if ws[k], err = get(p + name); err != nil {
				return err
			}
		}
		l.in, l.postAttn, l.pre, l.pst = walloc(ws[0]), walloc(ws[1]), walloc(ws[2]), walloc(ws[3])
		l.qk = walloc(append(append([]float32(nil), ws[4]...), ws[5]...))
		l.qkv = place(append(append(ws[6], ws[7]...), ws[8]...), 3*D, D)
		// o reads the attention's context at 80 columns a head, zeros at
		// each head's pads (qimage/vision's padHeads).
		o := make([]float32, D*H*vitHDPad)
		for r := range D {
			for h := range H {
				copy(o[(r*H+h)*vitHDPad:][:hd], ws[9][(r*H+h)*hd:][:hd])
			}
		}
		l.o = place(o, D, H*vitHDPad)
		// gate and up interleaved 32 rows at a time: the GLU epilogue's
		// contract (dit_gemm.comp C_SWIGLU, WN = 4).
		l.gu = place(interleaveGLU(padRows(ws[10], c.Inter, D, v.ffnPad), padRows(ws[11], c.Inter, D, v.ffnPad), v.ffnPad, D), 2*v.ffnPad, D)
		l.down = place(padCols(ws[12], D, c.Inter, v.ffnPad), D, v.ffnPad)
	}

	if v.wbuf, err = v.dev.NewBuffer(len(w) * 4); err != nil {
		return err
	}
	v.wbuf.WriteFloat32(w)
	if v.bank, err = v.dev.NewBuffer(bankHalves * 2); err != nil {
		return fmt.Errorf("gemma4: vision bank (%d MB): %w", bankHalves>>19, err)
	}
	for _, s := range pending {
		buf := make([]uint16, s.n*s.k)
		packF16(buf, s.w, s.n, s.k)
		v.bank.WriteUint16At(s.off, buf)
	}

	// Activations. Every row-shaped tensor is padded to the GEMM's M tile.
	R := roundUp(v.maxRows, 128) // the LDS GEMM's row tile
	var acts int
	aalloc := func(n int) uint32 {
		off := uint32(acts)
		acts += roundUp(n, 64)
		return off
	}
	v.aX = aalloc(R * D)
	v.aY = aalloc(R * D)
	v.aP = aalloc(R * 3 * D)
	v.aPos = aalloc(R * D)
	v.aRope = aalloc(R * 2 * hd)
	v.aPool = aalloc(roundUp(c.MaxSoft, vitBM) * D)
	v.aOut = aalloc(roundUp(c.MaxSoft, vitBM) * v.textHidden)
	if v.abuf, err = v.dev.NewBuffer(acts * 4); err != nil {
		return err
	}
	v.lda = max(3*c.Patch*c.Patch, D, H*vitHDPad, v.ffnPad) + vitLdaPad
	v.planeTok = roundUp(v.maxRows, vitTokenAlign)
	halves := R * v.lda
	v.hA = 0
	// The GLU's output, the down projection's A operand: rows padded to the
	// LDS build's 128.
	v.ldF = v.ffnPad + vitLdaPad
	v.hF = uint32(halves)
	halves += roundUp(v.maxRows, 128) * v.ldF
	plane := H * v.planeTok * vitHDPad
	v.hQ = uint32(halves)
	v.hK = v.hQ + uint32(plane)
	v.hV = v.hK + uint32(plane)
	halves += 3 * plane
	if v.hbuf, err = v.dev.NewBuffer(halves * 2); err != nil {
		return err
	}
	v.hbuf.Zero()
	return nil
}

func (v *Vision) build() error {
	pc := uint32(pushBytes)
	base := []*vk.Buffer{v.wbuf, v.abuf, v.hbuf}
	attn, wave := shaders.QViTAttnWMMAHD80, uint32(0)
	if sgs, err := v.dev.Physical().SubgroupSizeControl(); err == nil && v.dev.Features().SubgroupSizeControl &&
		sgs.Supported && sgs.MinSubgroupSize <= 32 && 32 <= sgs.MaxSubgroupSize {
		attn, wave = shaders.QViTAttnWMMAHD80W32, 32
	}
	v.wave = wave
	for _, p := range []struct {
		name  string
		spirv []byte
		bufs  []*vk.Buffer
		wave  uint32
	}{
		{"join", shaders.Gemma4ViTJoin, base, 0},
		{"prep", shaders.Gemma4ViTPrep, base, 64}, // one wave a (row, head): subgroupAdd is its reduction
		{"geglu", shaders.Gemma4ViTGEMMGEGLU, []*vk.Buffer{v.wbuf, v.abuf, v.hbuf, v.bank}, 0},
		{"geglu_lds", shaders.Gemma4ViTGEMMGEGLULDS, []*vk.Buffer{v.wbuf, v.abuf, v.hbuf, v.bank}, 32},
		{"pool", shaders.Gemma4ViTPool, base, 0},
		{"wmma", attn, base, wave},
		{"gemm", shaders.DiTGEMMReg64Tiled, []*vk.Buffer{v.wbuf, v.abuf, v.hbuf, v.bank}, 0},
	} {
		mod, err := v.dev.NewShaderModule(p.spirv)
		if err != nil {
			return fmt.Errorf("gemma4: vision shader %s: %w", p.name, err)
		}
		v.mods = append(v.mods, mod)
		pipe, err := v.dev.NewPipeline(mod, vk.PipelineSpec{Buffers: p.bufs, PushConstantSize: pc, RequiredSubgroupSize: p.wave})
		if err != nil {
			return fmt.Errorf("gemma4: vision pipeline %s: %w", p.name, err)
		}
		v.pipes[p.name] = pipe
	}
	return nil
}

// Destroy releases the tower's device objects.
func (v *Vision) Destroy() {
	for _, p := range v.pipes {
		p.Destroy()
	}
	for _, m := range v.mods {
		m.Destroy()
	}
	for _, b := range []*vk.Buffer{v.wbuf, v.abuf, v.hbuf, v.bank} {
		if b != nil {
			b.Destroy()
		}
	}
	v.pipes, v.mods = nil, nil
	v.wbuf, v.abuf, v.hbuf, v.bank = nil, nil, nil, nil
}

// Bytes is the tower's device memory.
func (v *Vision) Bytes() int {
	return v.wbuf.Size() + v.abuf.Size() + v.hbuf.Size() + v.bank.Size()
}

// VisionTap, when set on a Vision, sees the residual after the embedding
// ("embed"), after each layer ("layer%d"), and the pooled rows ("pooled"),
// for the gate. It costs a submit at each.
type VisionTap func(name string, rows []float32)

// Encode runs the tower over one image: [p.Soft][text hidden] rows.
func (v *Vision) Encode(p *Patches, tap VisionTap) ([]float32, error) {
	return v.encode(p, tap, nil)
}

// Profile runs the tower over one image with a timestamp after every
// dispatch: GPU time by dispatch label, summed over the layers.
func (v *Vision) Profile(p *Patches) (map[string]time.Duration, error) {
	out := map[string]time.Duration{}
	_, err := v.encode(p, nil, out)
	return out, err
}

func (v *Vision) encode(p *Patches, tap VisionTap, prof map[string]time.Duration) ([]float32, error) {
	c := v.Cfg
	D, hd, H := c.Hidden, c.HeadDim, c.Heads
	n := p.N()
	if n > v.maxRows || n%(c.Pool*c.Pool) != 0 || p.GW%c.Pool != 0 {
		return nil, fmt.Errorf("gemma4: a %dx%d patch grid does not fit the tower", p.GW, p.GH)
	}
	K := 3 * c.Patch * c.Patch

	// The host's share: the pixels as fp16 A rows (2(x - 0.5) first, the
	// model's own first step), the 2-D position embedding and the rope.
	a := make([]uint16, n*v.lda)
	for r := range n {
		for j := range K {
			a[r*v.lda+j] = safetensors.F32ToF16(2 * (p.Pixels[r*K+j] - 0.5))
		}
	}
	v.hbuf.WriteUint16At(int(v.hA), a)
	pos := make([]float32, n*D)
	rope := make([]float32, n*2*hd)
	quarter := hd / 4 // 18 frequencies an axis
	inv := make([]float32, quarter)
	for j := range quarter {
		inv[j] = float32(1 / math.Pow(c.RopeBase, float64(2*j)/float64(hd/2)))
	}
	tab := v.posTable.Data
	bf := func(axis, idx, i int) float32 {
		o := ((axis*c.PosSize+idx)*D + i) * 2
		return math.Float32frombits((uint32(tab[o]) | uint32(tab[o+1])<<8) << 16)
	}
	for r := range n {
		x, y := r%p.GW, r/p.GW
		for i := range D {
			pos[r*D+i] = bf(0, x, i) + bf(1, y, i)
		}
		cs, sn := rope[r*2*hd:], rope[r*2*hd+hd:]
		for j := range quarter {
			fx := float64(float32(x) * inv[j])
			fy := float64(float32(y) * inv[j])
			for _, k := range []int{j, j + quarter} {
				cs[k], sn[k] = float32(math.Cos(fx)), float32(math.Sin(fx))
				cs[2*quarter+k], sn[2*quarter+k] = float32(math.Cos(fy)), float32(math.Sin(fy))
			}
		}
	}
	v.abuf.WriteFloat32At(int(v.aPos), pos)
	v.abuf.WriteFloat32At(int(v.aRope), rope)

	var d []vk.MultiDispatch
	var labels []string
	type mark struct {
		at   int
		name string
		off  uint32
		rows int
		cols int
	}
	var marks []mark
	tapAt := func(name string, off uint32, rows, cols int) {
		if tap != nil {
			marks = append(marks, mark{len(d), name, off, rows, cols})
		}
	}
	label := ""
	add := func(pipe string, gx, gy uint32, pc push) {
		d = append(d, vk.MultiDispatch{Pipeline: v.pipes[pipe], GroupsX: gx, GroupsY: gy, PushConstants: pc.bytes()})
		if label != "" {
			labels, label = append(labels, label), ""
		} else {
			labels = append(labels, pipe)
		}
	}
	gemmLDA := func(b uint32, aOff, cOff uint32, m, nOut, k, lda int) {
		mPad := roundUp(m, vitBM)
		if label == "" {
			label = fmt.Sprintf("gemm n%d k%d", nOut, k)
		}
		add("gemm", uint32(nOut/vitBN), uint32(mPad/vitBM), push{InOff: aOff, OutOff: cOff, BOff: b,
			GemmM: uint32(mPad), GemmN: uint32(nOut), GemmK: uint32(k), LDA: uint32(lda)})
	}
	gemm := func(b uint32, aOff, cOff uint32, m, nOut, k int) { gemmLDA(b, aOff, cOff, m, nOut, k, v.lda) }
	// gate+up with the GELU-tanh GLU in the epilogue, into hF. The LDS-staged
	// wave32 build wants a full machine (~1000 rows and up, research §2.9).
	geglu := func(b uint32, m int) {
		pipe, bm, bn := "geglu", vitBM, vitBN
		if m >= 1024 && v.wave == 32 {
			pipe, bm, bn = "geglu_lds", 128, 256
		}
		mPad := roundUp(m, bm)
		label = "gemm gate+up glu"
		add(pipe, uint32(2*v.ffnPad/bn), uint32(mPad/bm), push{InOff: v.hA, OutOff: v.hF, BOff: b,
			GemmM: uint32(mPad), GemmN: uint32(2 * v.ffnPad), GemmK: uint32(D), LDA: uint32(v.lda),
			Aux0: uint32(v.ldF), Scale: math.Float32bits(1)})
	}
	eps := math.Float32bits(float32(c.Eps))
	join := func(y uint32, w uint32, next uint32, embed bool) {
		pc := push{InOff: y, OutOff: v.aX, WOff: w, KOff: next, BOff: v.hA, LDA: uint32(v.lda),
			Tokens: uint32(n), Dim: uint32(D), Eps: eps}
		if embed {
			pc.Aux0, pc.Aux1 = 1, v.aPos
		}
		add("join", uint32(n), 1, pc)
	}

	// The patch embedding plus the positions, and layer 0's input norm.
	gemm(v.patch, v.hA, v.aY, n, D, K)
	join(v.aY, noW, v.layers[0].in, true)
	tapAt("embed", v.aX, n, D)

	for i, l := range v.layers {
		gemm(l.qkv, v.hA, v.aP, n, 3*D, D)
		// Scale 1 (Gemma's attention), so q carries only log2(e): the
		// kernel's softmax is exp2. The rows past the image up to the key
		// block are zeroed by the same dispatch.
		add("prep", uint32(H), uint32(roundUp(n, vitKeyBlock)), push{InOff: v.aP, Aux0: uint32(3 * D),
			OutOff: v.hQ, KOff: v.hK, VOff: v.hV, Aux2: uint32(v.planeTok), Scale: math.Float32bits(1.4426950408889634),
			WOff: l.qk, Aux1: v.aRope, Tokens: uint32(n), Eps: eps})
		add("wmma", uint32((n+tile-1)/tile), uint32(H), push{InOff: v.hQ, KOff: v.hK, VOff: v.hV, OutOff: v.hA,
			LDA: uint32(v.lda), Tokens: uint32(n), Dim: uint32(D), Heads: uint32(H), Aux1: uint32(v.planeTok)})
		gemm(l.o, v.hA, v.aY, n, D, H*vitHDPad)
		join(v.aY, l.postAttn, l.pre, false)
		geglu(l.gu, n)
		gemmLDA(l.down, v.hF, v.aY, n, D, v.ffnPad, v.ldF)
		next := uint32(noW)
		if i+1 < len(v.layers) {
			next = v.layers[i+1].in
		}
		join(v.aY, l.pst, next, false)
		tapAt(fmt.Sprintf("layer%d", i), v.aX, n, D)
	}

	add("pool", uint32(p.Soft), 1, push{InOff: v.aX, OutOff: v.aPool, WOff: v.stdBias, KOff: v.stdScale,
		BOff: v.hA, LDA: uint32(v.lda), Tokens: uint32(p.Soft), Dim: uint32(D), Aux0: uint32(p.GW / c.Pool),
		Eps: eps, Scale: math.Float32bits(float32(math.Sqrt(float64(D))))})
	tapAt("pooled", v.aPool, p.Soft, D)
	gemm(v.proj, v.hA, v.aOut, p.Soft, v.textHidden, D)

	if prof != nil {
		for i := 0; i < len(d); i += 32 {
			j := min(i+32, len(d))
			_, marks, err := vk.DispatchMultiMarked(d[i:j], true)
			if err != nil {
				return nil, err
			}
			for k, m := range marks {
				prof[labels[i+k]] += m
			}
		}
		return nil, nil
	}
	for i := 0; i < len(d); {
		j := min(i+16, len(d))
		if len(marks) > 0 && marks[0].at < j {
			j = marks[0].at
		}
		if j > i {
			if _, err := vk.DispatchMultiTimed(d[i:j], 1, 1, true); err != nil {
				return nil, fmt.Errorf("gemma4: vision dispatch %d-%d: %w", i, j-1, err)
			}
		}
		for len(marks) > 0 && marks[0].at == j {
			m := marks[0]
			tap(m.name, v.abuf.ReadFloat32At(int(m.off), m.rows*m.cols))
			marks = marks[1:]
		}
		i = j
	}
	for _, m := range marks {
		tap(m.name, v.abuf.ReadFloat32At(int(m.off), m.rows*m.cols))
	}
	return v.abuf.ReadFloat32At(int(v.aOut), p.Soft*v.textHidden), nil
}
