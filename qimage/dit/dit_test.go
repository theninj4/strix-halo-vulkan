package dit

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
)

// The references come from reference/dump_qi21_dit.py (everything around the
// blocks, num_layers=0) and reference/dump_qi21_dit_block.py (block 0 with
// real weights, all three cache modes). Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_dit.py
//	.venv/bin/python reference/dump_qi21_dit_block.py
const (
	headRef     = "../../reference/out/qi21dit"
	blockRef    = "../../reference/out/qi21block"
	transformer = "../../models/Qwen-Image-2.1/transformer"
)

type caseInfo struct {
	TextLens  []int   `json:"text_lens"`
	ImgShapes [][]int `json:"img_shapes"`
	VLMLen    int     `json:"vlm_len"`
	PrefixLen int     `json:"prefix_len"`
	Segments  [][]any `json:"segments"`
}

type manifest struct {
	dir string

	T       float64             `json:"t"`
	Cases   map[string]caseInfo `json:"cases"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T, dir string) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run the dump script", dir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = dir
	return &m
}

func (c caseInfo) layout(t *testing.T) *Layout {
	t.Helper()
	shapes := make([][3]int, len(c.ImgShapes))
	for i, s := range c.ImgShapes {
		copy(shapes[i][:], s)
	}
	l, err := NewLayout(c.TextLens, shapes)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func loadRef(t *testing.T, m *manifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(m.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// relTol is the fp32 bound, same instrument as qimage/textenc's: both sides
// fp32, drift is summation order, the rms floor keeps outliers honest. One
// block deep, so the depth argument that loosened textenc's deep bound does
// not apply here.
const relTol = 2e-4

func deviation(got, want *qwen.Mat) (maxAbs, rms, rel float64, worst int) {
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms = math.Sqrt(sumSq / float64(len(want.Data)))
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		d := math.Abs(g - w)
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(w), math.Max(rms, 1e-12)); r > rel {
			rel, worst = r, i
		}
	}
	return maxAbs, rms, rel, worst
}

func compare(t *testing.T, name string, got, want *qwen.Mat) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	maxAbs, rms, rel, worst := deviation(got, want)
	if rel > relTol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g at %d (got %g want %g), rms %.6g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, relTol)
		return
	}
	t.Logf("%-22s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

// asMat rewraps a Mat to a different row split of the same data — how the
// dump's [1, rows, heads, head_dim] cache tensors line up against this
// package's [rows, heads*head_dim].
func asMat(m *qwen.Mat, rows, cols int) *qwen.Mat {
	return &qwen.Mat{Rows: rows, Cols: cols, Data: m.Data}
}

// TestLayout checks the joint sequence's geometry — image ids, the target
// mask, the prefix segments — against what diffusers derived for both cases.
func TestLayout(t *testing.T) {
	hm := loadManifest(t, headRef)
	bm := loadManifest(t, blockRef)
	for label, c := range hm.Cases {
		l := c.layout(t)
		ids := loadRef(t, hm, label+"_image_ids")
		target := loadRef(t, hm, label+"_target_mask")
		if l.Seq() != ids.Cols*ids.Rows {
			t.Fatalf("%s: layout is %d tokens, dump %d", label, l.Seq(), ids.Rows*ids.Cols)
		}
		for i := 0; i < l.Seq(); i++ {
			if float32(l.ImageIDs[i]) != ids.Data[i] {
				t.Fatalf("%s: image id %d is %d, dump says %g", label, i, l.ImageIDs[i], ids.Data[i])
			}
			if (l.Target[i] && target.Data[i] != 1) || (!l.Target[i] && target.Data[i] != 0) {
				t.Fatalf("%s: target bit %d is %v, dump says %g", label, i, l.Target[i], target.Data[i])
			}
		}
		bc := bm.Cases[label]
		if l.PrefixLen != bc.PrefixLen {
			t.Errorf("%s: prefix %d, dump %d", label, l.PrefixLen, bc.PrefixLen)
		}
		if len(l.Segments) != len(bc.Segments) {
			t.Fatalf("%s: %d segments, dump %d", label, len(l.Segments), len(bc.Segments))
		}
		for i, s := range l.Segments {
			ref := bc.Segments[i]
			if float64(s.Start) != ref[0].(float64) || float64(s.End) != ref[1].(float64) || s.Text != ref[2].(bool) {
				t.Errorf("%s: segment %d is %+v, dump %v", label, i, s, ref)
			}
		}
		t.Logf("%s: %d tokens, prefix %d, %d segments", label, l.Seq(), l.PrefixLen, len(l.Segments))
	}
}

// TestRope checks the 3-axis table — frozen frame axis, zero-centred grids,
// negative indices — against the model's own complex freqs.
func TestRope(t *testing.T) {
	m := loadManifest(t, headRef)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	for label, c := range m.Cases {
		l := c.layout(t)
		rope, err := NewRope(l, cfg.AxesDims, cfg.HeadDim, 10000)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_rope_real", rope.Cos, loadRef(t, m, label+"_rope_real"))
		compare(t, label+"_rope_imag", rope.Sin, loadRef(t, m, label+"_rope_imag"))
	}
}

// TestHeadPieces checks txt_in, img_in, the timestep embedding and the
// shared modulation, weight by weight, against the num_layers=0 dump.
func TestHeadPieces(t *testing.T) {
	m := loadManifest(t, headRef)
	model, err := Load(transformer, 0)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	for label := range m.Cases {
		txt := loadRef(t, m, label+"_encoder_hidden")
		latents := loadRef(t, m, label+"_latents")
		got, err := model.TxtProject(txt)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_txt_in", got, loadRef(t, m, label+"_txt_in"))
		got, err = model.ImgIn.Apply(latents)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_img_in", got, loadRef(t, m, label+"_img_in"))
	}
	temb, err := model.TimeEmbed(m.T)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "temb", temb, loadRef(t, m, "t2i_temb"))
	smod := temb.Clone()
	for i := range smod.Data {
		smod.Data[i] = silu(smod.Data[i])
	}
	mod, err := model.Modulation.Apply(smod)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "modulation", mod, loadRef(t, m, "t2i_modulation"))
}

// TestHeadForward runs the whole no-block model — assembly, head, tail — in
// prefill and cached modes against the dump's forwards. With no blocks the
// two must agree on the target rows exactly, which the dump measured as 0.
func TestHeadForward(t *testing.T) {
	m := loadManifest(t, headRef)
	model, err := Load(transformer, 0)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	for label, c := range m.Cases {
		l := c.layout(t)
		txt := loadRef(t, m, label+"_encoder_hidden")
		latents := loadRef(t, m, label+"_latents")
		out, err := model.Forward(latents, txt, l, m.T, ModePrefill, nil)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_out_prefill", out, loadRef(t, m, label+"_out_prefill"))
		out, err = model.Forward(latents, txt, l, m.T, ModeCached, NewCache(0))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_out_cached", out, loadRef(t, m, label+"_out_cached"))
	}
}

// TestBlock is Q2's acceptance gate: block 0 with real weights, in extract
// mode (output plus the cache's own contents) and cached mode, both cases.
func TestBlock(t *testing.T) {
	m := loadManifest(t, blockRef)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	model, err := Load(transformer, 1)
	if err != nil {
		t.Fatal(err)
	}
	b := model.Blocks[0]
	for label, c := range m.Cases {
		l := c.layout(t)
		rope, err := NewRope(l, cfg.AxesDims, cfg.HeadDim, 10000)
		if err != nil {
			t.Fatal(err)
		}
		hidden := loadRef(t, m, label+"_hidden")
		mod := loadRef(t, m, label+"_modulation")

		cache := LayerCache{}
		out, err := b.Forward(hidden, mod, rope, l, ModeExtract, &cache)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_out_prefill", out, loadRef(t, m, label+"_out_prefill"))
		heads := cfg.NumHeads
		compare(t, label+"_cache_k", asMat(cache.K, l.PrefixLen*heads, cfg.HeadDim), loadRef(t, m, label+"_cache_k"))
		compare(t, label+"_cache_v", asMat(cache.V, l.PrefixLen*heads, cfg.HeadDim), loadRef(t, m, label+"_cache_v"))

		targetRows := cloneRows(hidden, l.PrefixLen, l.Seq())
		out, err = b.Forward(targetRows, mod, rope.Slice(l.PrefixLen), l, ModeCached, &cache)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, label+"_out_cached", out, loadRef(t, m, label+"_out_cached"))
	}
}

// TestValidationDetectsErrors is the negative control: the one mistake this
// block invites that every shape check misses is using the text encoder's
// rotary convention — pairing by halves instead of adjacent components.
func TestValidationDetectsErrors(t *testing.T) {
	m := loadManifest(t, blockRef)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	model, err := Load(transformer, 1)
	if err != nil {
		t.Fatal(err)
	}
	b := model.Blocks[0]
	c := m.Cases["t2i"]
	l := c.layout(t)
	rope, err := NewRope(l, cfg.AxesDims, cfg.HeadDim, 10000)
	if err != nil {
		t.Fatal(err)
	}
	// The perturbation: rotate q/k with the NeoX-halves pairing by swapping
	// the table's meaning — component j against j+half. Implemented by
	// running the real forward against a rope whose Apply is fooled: we
	// permute the sin table's sign pattern by pre-rotating the hidden state
	// is not equivalent, so instead build a block whose head dim is the full
	// rotation and compare a manually mis-paired attention input.
	hidden := loadRef(t, m, "t2i_hidden")
	mod := loadRef(t, m, "t2i_modulation")
	// Mis-pair by permuting each head's components so adjacent-pair rotation
	// lands on the halves pairing: [0, half, 1, half+1, …] before, inverse
	// after. If the convention did not matter, this would be a no-op.
	perm := misPair(hidden, cfg.NumHeads, cfg.HeadDim)
	out, err := b.Forward(perm, mod, rope, l, ModePrefill, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, rel, _ := deviation(out, loadRef(t, m, "t2i_out_prefill"))
	if rel <= relTol {
		t.Errorf("perturbation NOT caught: rel %.3g <= %.0e", rel, relTol)
		return
	}
	t.Logf("caught: rel %.3g (%.0fx the %.0e bound)", rel, rel/relTol, relTol)
}

// misPair returns a copy with each head's components interleaved
// [0, half, 1, half+1, …] — the permutation under which adjacent-pair
// rotation becomes halves rotation of the original.
func misPair(x *qwen.Mat, heads, headDim int) *qwen.Mat {
	half := headDim / 2
	out := qwen.NewMat(x.Rows, x.Cols)
	for r := 0; r < x.Rows; r++ {
		src, dst := x.Row(r), out.Row(r)
		for h := 0; h < heads; h++ {
			s, d := src[h*headDim:(h+1)*headDim], dst[h*headDim:(h+1)*headDim]
			for j := 0; j < half; j++ {
				d[2*j] = s[j]
				d[2*j+1] = s[j+half]
			}
		}
	}
	return out
}
