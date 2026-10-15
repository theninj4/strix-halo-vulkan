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

// The references come from reference/dump_flux3_dit_block.py (one fp32 512²
// edit forward of diffusers' Flux3Transformer2DModel, hooked around the first
// mode block of each mode, single blocks 0 and 27 and norm_out) and
// reference/dump_flux3_textenc.py (its text input). Regenerate with
//
//	.venv-flux3/bin/python -I reference/dump_flux3_textenc.py
//	.venv-flux3/bin/python -I reference/dump_flux3_dit_block.py
const (
	blockDir = "../../reference/out/flux3block"
	textDir  = "../../reference/out/flux3textenc"
	modelDir = "../../models/flux-3-image-klein/transformer"
)

type dump struct {
	dir     string
	T       float64 `json:"t"`
	TNext   float64 `json:"t_next"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

// needModel skips when the checkpoint is absent; a load error past that is a failure.
func needModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s (%v)", modelDir, err)
	}
}

func loadDump(t *testing.T, dir string) *dump {
	t.Helper()
	needModel(t)
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v)", dir, err)
	}
	d := &dump{dir: dir}
	if err := json.Unmarshal(buf, d); err != nil {
		t.Fatal(err)
	}
	return d
}

func (d *dump) mat(t *testing.T, name string) *qwen.Mat {
	t.Helper()
	meta, ok := d.Tensors[name]
	if !ok {
		t.Fatalf("%s has no tensor %q", d.dir, name)
	}
	raw, err := os.ReadFile(filepath.Join(d.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	cols := meta.Shape[len(meta.Shape)-1]
	return &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: data}
}

// blockTol is the per-block fp32 bound qimage/dit held (both sides fp32,
// summation order only); X2 measures this model against it.
const blockTol = 4e-5

func compare(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	worst := 0
	for i := range want.Data {
		g, w := float64(got.Data[i]), float64(want.Data[i])
		d := math.Abs(g - w)
		maxAbs = math.Max(maxAbs, d)
		if r := d / math.Max(math.Abs(w), math.Max(rms, 1e-12)); r > rel {
			rel, worst = r, i
		}
	}
	if rel > tol || math.IsNaN(rel) {
		t.Errorf("%-24s %s: max abs %.4g, worst rel %.3g at %d (got %g want %g), rms %.4g > %.0e",
			name, got, maxAbs, rel, worst, got.Data[worst], want.Data[worst], rms, tol)
		return
	}
	t.Logf("%-24s %-12s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

func rowsOf(m *qwen.Mat, from, n int) *qwen.Mat {
	return &qwen.Mat{Rows: n, Cols: m.Cols, Data: m.Data[from*m.Cols : (from+n)*m.Cols]}
}

// modOf splits one row of a dumped temb_mod (shift|scale|gate).
func modOf(m *qwen.Mat, row int) Mod {
	r, d := m.Row(row), m.Cols/3
	return Mod{Shift: r[:d], Scale: r[d : 2*d], Gate: r[2*d:]}
}

// ropeOf narrows a dumped repeat-interleaved [S, 128] table to the [S, 64]
// pairs Rope holds, after checking the interleave really is a repeat.
func ropeOf(t *testing.T, d *dump, prefix string) *Rope {
	t.Helper()
	narrow := func(m *qwen.Mat) *qwen.Mat {
		out := qwen.NewMat(m.Rows, m.Cols/2)
		for r := 0; r < m.Rows; r++ {
			src, dst := m.Row(r), out.Row(r)
			for j := range dst {
				if src[2*j] != src[2*j+1] {
					t.Fatalf("%s: row %d pair %d is not repeated (%g, %g)", prefix, r, j, src[2*j], src[2*j+1])
				}
				dst[j] = src[2*j]
			}
		}
		return out
	}
	return &Rope{Cos: narrow(d.mat(t, prefix+"_cos")), Sin: narrow(d.mat(t, prefix+"_sin"))}
}

// TestRope builds every mode's table from the dumped ids and gates it
// against the model's own Flux3PosEmbed output.
func TestRope(t *testing.T) {
	d := loadDump(t, blockDir)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ ids, rope string }{
		{"txt_ids", "mode_txt0_in_image_rotary_emb"},
		{"img_ids", "mode_image0_in_image_rotary_emb"},
		{"cond_ids", "mode_cond0_in_image_rotary_emb"},
	} {
		got, err := NewRope(d.mat(t, c.ids), cfg.AxesDimsRope, cfg.RopeTheta)
		if err != nil {
			t.Fatal(err)
		}
		want := ropeOf(t, d, c.rope)
		compare(t, c.ids+" cos", got.Cos, want.Cos, 1e-6)
		compare(t, c.ids+" sin", got.Sin, want.Sin, 1e-6)
	}
}

// TestModulation gates the timestep embeddings and all six modulations
// against the rows the blocks were handed, and the output layer's temb.
func TestModulation(t *testing.T) {
	d := loadDump(t, blockDir)
	m, err := Load(modelDir, Parts{})
	if err != nil {
		t.Fatal(err)
	}
	mods, err := m.Modulations(d.T, d.TNext)
	if err != nil {
		t.Fatal(err)
	}
	flat := func(md Mod) *qwen.Mat {
		v := append(append(append([]float32(nil), md.Shift...), md.Scale...), md.Gate...)
		return &qwen.Mat{Rows: 1, Cols: len(v), Data: v}
	}
	compare(t, "image temb", &qwen.Mat{Rows: 1, Cols: len(mods.ImageTemb), Data: mods.ImageTemb},
		rowsOf(d.mat(t, "norm_out_in_temb"), 0, 1), blockTol)
	compare(t, "early txt", flat(mods.Early[Text]), rowsOf(d.mat(t, "mode_txt0_in_temb_mod"), 0, 1), blockTol)
	compare(t, "early image", flat(mods.Early[Image]), rowsOf(d.mat(t, "mode_image0_in_temb_mod"), 0, 1), blockTol)
	compare(t, "early cond", flat(mods.Early[Cond]), rowsOf(d.mat(t, "mode_cond0_in_temb_mod"), 0, 1), blockTol)
	content := d.mat(t, "single0_in_temb_mod") // image rows, then cond rows
	compare(t, "single txt", flat(mods.Single[Text]), rowsOf(d.mat(t, "single0_in_temb_mod_context"), 0, 1), blockTol)
	compare(t, "single image", flat(mods.Single[Image]), rowsOf(content, 0, 1), blockTol)
	compare(t, "single cond", flat(mods.Single[Cond]), rowsOf(content, content.Rows-1, 1), blockTol)
}

// TestModeBlocks runs the first mode block of each mode on its exact
// dumped inputs (hidden state, modulation, rotary table).
func TestModeBlocks(t *testing.T) {
	d := loadDump(t, blockDir)
	blocks, err := LoadBlocks(modelDir, "txt_mode_blocks.0", "mode_blocks.image.0", "mode_blocks.image_cond.0")
	if err != nil {
		t.Fatal(err)
	}
	for i, name := range []string{"mode_txt0", "mode_image0", "mode_cond0"} {
		x := d.mat(t, name+"_in_hidden_states")
		mod := modOf(d.mat(t, name+"_in_temb_mod"), 0)
		got, err := blocks[i].Forward(x, []Segment{{x.Rows, mod}}, ropeOf(t, d, name+"_in_image_rotary_emb"))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, name, got, d.mat(t, name+"_out"), blockTol)
	}
}

// TestSingleBlocks runs single blocks 0 and 27 on their exact inputs: the
// joint [text, image, cond] sequence with three modulations.
func TestSingleBlocks(t *testing.T) {
	d := loadDump(t, blockDir)
	blocks, err := LoadBlocks(modelDir, "single_transformer_blocks.0", "single_transformer_blocks.27")
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeOf(t, d, "single0_in_image_rotary_emb") // shared by every single block
	for i, name := range []string{"single0", "single27"} {
		txt, content := d.mat(t, name+"_in_encoder_hidden_states"), d.mat(t, name+"_in_hidden_states")
		cmod, tmod := d.mat(t, name+"_in_temb_mod"), d.mat(t, name+"_in_temb_mod_context")
		half := content.Rows / 2 // 1024 image rows, then 1024 cond rows
		segs := []Segment{{txt.Rows, modOf(tmod, 0)}, {half, modOf(cmod, 0)}, {half, modOf(cmod, half)}}
		got, err := blocks[i].Forward(stack(txt, content), segs, rope)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, name+" text", rowsOf(got, 0, txt.Rows), d.mat(t, name+"_out_encoder_hidden_states"), blockTol)
		compare(t, name+" content", rowsOf(got, txt.Rows, content.Rows), d.mat(t, name+"_out_hidden_states"), blockTol)
	}
}

// TestFinal gates norm_out on its exact inputs, then proj_out against the
// edit forward's velocity.
func TestFinal(t *testing.T) {
	d := loadDump(t, blockDir)
	m, err := Load(modelDir, Parts{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.FinalNorm(d.mat(t, "norm_out_in_hidden_states"), d.mat(t, "norm_out_in_temb").Row(0))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "norm_out", got, d.mat(t, "norm_out_out"), blockTol)
	v, err := m.ProjOut.Apply(d.mat(t, "norm_out_out"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "proj_out", v, d.mat(t, "edit_velocity"), blockTol)
}

// TestForward is the stack gate: the whole model from the dumped noise,
// text embedding, reference tokens and ids to the velocity, edit and t2i.
// ~22 GB of fp32 weights and several minutes of CPU.
func TestForward(t *testing.T) {
	if testing.Short() {
		t.Skip("whole model in fp32")
	}
	d := loadDump(t, blockDir)
	te := loadDump(t, textDir)
	m, err := Load(modelDir, Parts{Mode: 5, Single: 28})
	if err != nil {
		t.Fatal(err)
	}
	in := &Input{
		Img: d.mat(t, "noise"), Txt: te.mat(t, "short_prompt_embeds"),
		ImgIDs: d.mat(t, "img_ids"), TxtIDs: d.mat(t, "txt_ids"),
		T: d.T, TNext: d.TNext,
	}
	in.Cond, in.CondIDs = d.mat(t, "cond_tokens"), d.mat(t, "cond_ids")
	v, err := m.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "edit velocity", v, d.mat(t, "edit_velocity"), stackTol)
	in.Cond, in.CondIDs = nil, nil
	if v, err = m.Forward(in); err != nil {
		t.Fatal(err)
	}
	compare(t, "t2i velocity", v, d.mat(t, "t2i_velocity"), stackTol)
}

// stackTol is the 33-block bound. TestForward measured rel 2.5e-5 (edit)
// and 1.7e-5 (t2i) at 512² (X2, 2026-10-15) — inside the per-block bound, so
// error does not grow with depth here; 1e-4 keeps 4x of margin over that.
const stackTol = 1e-4

// relOf is compare's worst relative error, for the negative controls.
func relOf(got, want *qwen.Mat) float64 {
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		rel = math.Max(rel, d/math.Max(math.Abs(float64(want.Data[i])), rms))
	}
	return rel
}

// TestControls shows the gates above can fail: each control changes one
// thing the port could plausibly get wrong and must land far outside
// blockTol on the same dumped inputs.
func TestControls(t *testing.T) {
	d := loadDump(t, blockDir)
	m, err := Load(modelDir, Parts{})
	if err != nil {
		t.Fatal(err)
	}
	blocks, err := LoadBlocks(modelDir, "mode_blocks.image.0")
	if err != nil {
		t.Fatal(err)
	}
	x, want := d.mat(t, "mode_image0_in_hidden_states"), d.mat(t, "mode_image0_out")
	rope := ropeOf(t, d, "mode_image0_in_image_rotary_emb")
	good := modOf(d.mat(t, "mode_image0_in_temb_mod"), 0)

	check := func(name string, got *qwen.Mat) {
		t.Helper()
		r := relOf(got, want)
		if r < 100*blockTol {
			t.Errorf("control %q lands at rel %.3g — the gate cannot see it", name, r)
			return
		}
		t.Logf("control %-26s rel %.3g (%.0fx the bound)", name, r, r/blockTol)
	}

	// the flow map's jump target forgotten: modulation at (t, t)
	mods, err := m.Modulations(d.T, d.T)
	if err != nil {
		t.Fatal(err)
	}
	got, err := blocks[0].Forward(x, []Segment{{x.Rows, mods.Early[Image]}}, rope)
	if err != nil {
		t.Fatal(err)
	}
	check("t_next = t", got)

	// shift and scale exchanged (the blocks are shift-first, norm_out scale-first)
	swapped := Mod{Shift: good.Scale, Scale: good.Shift, Gate: good.Gate}
	if got, err = blocks[0].Forward(x, []Segment{{x.Rows, swapped}}, rope); err != nil {
		t.Fatal(err)
	}
	check("shift/scale swapped", got)

	// image ids unscaled (image_rope_scale 0.5 forgotten)
	ids := d.mat(t, "img_ids").Clone()
	for r := 0; r < ids.Rows; r++ {
		ids.Row(r)[1] *= 2
		ids.Row(r)[2] *= 2
	}
	unscaled, err := NewRope(ids, m.Cfg.AxesDimsRope, m.Cfg.RopeTheta)
	if err != nil {
		t.Fatal(err)
	}
	if got, err = blocks[0].Forward(x, []Segment{{x.Rows, good}}, unscaled); err != nil {
		t.Fatal(err)
	}
	check("ids unscaled", got)
}
