package dit

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference is reference/dump_h3_dit_block.py: diffusers' own modules
// for the timestep MLP, the input projections, the text refiner, blocks 0
// and 1 and the tail, in fp32, over a real 928-row layout. Regenerate with:
//
//	.venv/bin/python reference/dump_h3_dit_block.py
const (
	blockRef = "../../reference/out/h3block"
	modelDir = "../../models/MiniMax-H3/transformer"
)

type blockManifest struct {
	Text         int       `json:"text"`
	LatentFrames int       `json:"latent_frames"`
	LatentHeight int       `json:"latent_height"`
	LatentWidth  int       `json:"latent_width"`
	AudioLatents int       `json:"audio_latents"`
	Blocks       int       `json:"blocks"`
	Timestep     []float32 `json:"timestep"`
	Tensors      map[string]struct {
		Shape  []int   `json:"shape"`
		Count  int     `json:"count"`
		Absmax float64 `json:"absmax"`
	} `json:"tensors"`
}

func loadBlockRef(t *testing.T) *blockManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(blockRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_dit_block.py", blockRef, err)
	}
	var m blockManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func refMat(t *testing.T, m *blockManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(blockRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	data := make([]float32, meta.Count)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	rows, cols := 1, meta.Count
	if len(meta.Shape) >= 2 {
		cols = meta.Shape[len(meta.Shape)-1]
		rows = meta.Count / cols
	}
	return &qwen.Mat{Rows: rows, Cols: cols, Data: data}
}

// compare reports max abs error relative to the reference's absmax, and
// fails past tol. Everything here is fp32 against fp32, so what is left is
// summation order: tens of millions of products per value at most.
func compare(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: %v, want %v", name, got, want)
	}
	var maxAbs, ref, sq float64
	worst := 0
	for i := range got.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		sq += d * d
		if d > maxAbs {
			maxAbs, worst = d, i
		}
		ref = math.Max(ref, math.Abs(float64(want.Data[i])))
	}
	rel := maxAbs / ref
	rms := math.Sqrt(sq / float64(len(got.Data)))
	if rel > tol || math.IsNaN(rel) {
		t.Errorf("%-14s max abs %.4g at %d (got %g want %g) = %.3g of absmax %.4g > %.0e",
			name, maxAbs, worst, got.Data[worst], want.Data[worst], rel, ref, tol)
		return
	}
	t.Logf("%-14s max abs %.3g  rms %.3g  rel %.2g of absmax %.4g", name, maxAbs, rms, rel, ref)
}

// blockTol is fp32 noise relative to each tensor's absmax. Measured
// 2026-09-26: ≤ 8e-7 everywhere but temb, 5e-6 (torch's Sleef exp/cos/sin
// against Go's correctly rounded ones, as in h3/plan's rope tables).
const blockTol = 1e-5

// TestBlock is M3's gate: the transformer's front, the AdaLN tables, two
// blocks and the tail, each stage fed the reference's own input so a
// failure points at one piece; then the same chain run end to end on the
// Go side's own outputs.
func TestBlock(t *testing.T) {
	m := loadBlockRef(t)
	model, err := Load(modelDir, m.Blocks)
	if err != nil {
		t.Skipf("no transformer weights (%v)", err)
	}
	lay, err := plan.NewLayout(filled(m.Text, plan.TextTag), m.LatentFrames, m.LatentHeight, m.LatentWidth, m.AudioLatents, nil)
	if err != nil {
		t.Fatal(err)
	}
	cos, sin := lay.Rope(plan.InvFreq())
	rowT := toI32(refMat(t, m, "timestep_indices").Data)
	rowMod := RowMod(rowT, lay.Tags)

	temb, err := model.TimeEmbed(m.Timestep)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "temb", temb, refMat(t, m, "temb"), blockTol)
	temb = refMat(t, m, "temb")

	text, err := model.TextIn.Apply(refMat(t, m, "in_text"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "text_projected", text, refMat(t, m, "text_projected"), blockTol)
	text, err = model.Text(refMat(t, m, "in_text"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "text_refined", text, refMat(t, m, "text_refined"), blockTol)
	packed, err := model.Pack(lay, refMat(t, m, "text_refined"), refMat(t, m, "in_video"), refMat(t, m, "in_audio"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "packed", packed, refMat(t, m, "packed"), blockTol)

	in := refMat(t, m, "packed")
	for i, b := range model.Blocks {
		tab, err := b.Table(temb)
		if err != nil {
			t.Fatal(err)
		}
		// The reference's table is six [T·3, H] tensors stacked; ours is
		// T·3 rows of six. Same numbers, transposed.
		want := refMat(t, m, blockName(i, "adaln"))
		got := &qwen.Mat{Rows: want.Rows, Cols: want.Cols, Data: make([]float32, len(want.Data))}
		for r := 0; r < tab.Rows; r++ {
			for j := 0; j < 6; j++ {
				copy(got.Row(j*tab.Rows+r), tab.Vec(r, j))
			}
		}
		compare(t, blockName(i, "adaln"), got, want, blockTol)
		out, err := b.Forward(in, tab, rowMod, cos, sin)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, blockName(i, "out"), out, refMat(t, m, blockName(i, "out")), blockTol)
		in = refMat(t, m, blockName(i, "out"))
	}

	normed, video, audio, err := model.Tail(in, temb, rowT, lay)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "tail_normed", normed, refMat(t, m, "tail_normed"), blockTol)
	compare(t, "out_video", video, refMat(t, m, "out_video"), blockTol)
	compare(t, "out_audio", audio, refMat(t, m, "out_audio"), blockTol)

	// The heads alone, on the reference's own normed rows: separates the
	// heads' arithmetic from the norm's rounding they amplify.
	refNormed := refMat(t, m, "tail_normed")
	hv, err := model.ProjOut.Apply(gather(refNormed, lay.Video))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "head_video", hv, refMat(t, m, "out_video"), blockTol)
	ha, err := model.AudioProjOut.Apply(gather(refNormed, lay.Audio))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "head_audio", ha, refMat(t, m, "out_audio"), blockTol)
}

func blockName(i int, what string) string { return "block" + string(rune('0'+i)) + "_" + what }

func filled(n int, v int32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func toI32(f []float32) []int32 {
	out := make([]int32, len(f))
	for i, v := range f {
		out[i] = int32(v)
	}
	return out
}

// TestCheckArenas: the served and trained 5 s canvases fit, and the trained
// canvas at 14.4 s (104,966 rows) does not: its q/k/v planes alone are past
// one storage buffer. This is what M9 refuses at submit time.
func TestCheckArenas(t *testing.T) {
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skipf("no transformer config (%v)", err)
	}
	for _, c := range []struct {
		rows int
		ok   bool
	}{{15936, true}, {38247, true}, {74386, true}, {104966, false}} {
		err := CheckArenas(cfg, c.rows, 2048, 8192)
		if (err == nil) != c.ok {
			t.Errorf("%d rows: %v", c.rows, err)
		}
	}
}
