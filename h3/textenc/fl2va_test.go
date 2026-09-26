package textenc

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	qtextenc "strix-halo-vulkan/qimage/textenc"
	"strix-halo-vulkan/qimage/vision"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
)

// The reference is reference/dump_h3_fl2va.py's prep and text phases: the
// README fl2va prompt behind one keyframe ("f") and two ("fl", first and a
// cover-cropped last), both on a 256x448 canvas. Regenerate with:
//
//	.venv/bin/python reference/dump_h3_fl2va.py prep text   (~75 GB, ~3 min)
const fl2vaRef = "../../reference/out/h3fl2va"

type fl2vaManifest struct {
	Height, Width int
	Presentations map[string]struct {
		IDs   []int32  `json:"ids"`
		Tags  []int32  `json:"tags"`
		Grids [][3]int `json:"grids"`
	} `json:"presentations"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadFL2VA(t *testing.T) *fl2vaManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(fl2vaRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_fl2va.py prep text", fl2vaRef, err)
	}
	var m fl2vaManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func (m *fl2vaManifest) floats(t *testing.T, name string) ([]float32, []int) {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Skipf("the dump has no %q; run the phase that makes it", name)
	}
	raw, err := os.ReadFile(filepath.Join(fl2vaRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, meta.Count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, meta.Shape
}

func (m *fl2vaManifest) mat(t *testing.T, name string) *qwen.Mat {
	d, s := m.floats(t, name)
	return &qwen.Mat{Rows: s[0], Cols: s[1], Data: d}
}

// keyframe reads a dumped [H, W, 3] keyframe as bytes.
func (m *fl2vaManifest) keyframe(t *testing.T, name string) ([]byte, int, int) {
	d, s := m.floats(t, name)
	out := make([]byte, len(d))
	for i, v := range d {
		out[i] = byte(v)
	}
	return out, s[0], s[1]
}

func visionConfig(t *testing.T) *vision.Config {
	t.Helper()
	c, err := vision.LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder config (%v)", err)
	}
	return c
}

// TestPresentation holds the fl2va presentation — labels, vision blocks,
// the prompt — and its modality tags to the pipeline's, for one keyframe
// and two.
func TestPresentation(t *testing.T) {
	m := loadFL2VA(t)
	tok, err := tokenizer.Load(modelDir + "/tokenizer")
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	vc := visionConfig(t)
	for _, label := range []string{"f", "fl"} {
		want := m.Presentations[label]
		grids := make([]qtextenc.Grid, len(label))
		for i := range grids {
			grids[i] = ImageGrid(m.Height, m.Width, vc.PatchSize)
			if g := want.Grids[i]; g != [3]int{grids[i].T, grids[i].H, grids[i].W} {
				t.Fatalf("%s: grid %v, the processor's %v", label, grids[i], g)
			}
		}
		p, err := NewPresentation(tok, readmePrompt(t), grids, vc.SpatialMergeSize)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.IDs) != len(want.IDs) {
			t.Fatalf("%s: %d tokens, want %d", label, len(p.IDs), len(want.IDs))
		}
		for i := range want.IDs {
			if p.IDs[i] != want.IDs[i] || p.Tags[i] != want.Tags[i] {
				t.Fatalf("%s: token %d is %d/%d, want %d/%d", label, i, p.IDs[i], p.Tags[i], want.IDs[i], want.Tags[i])
			}
		}
		t.Logf("%s: %d tokens, %d image slots", label, len(p.IDs), len(p.Pads))
	}
}

func readmePrompt(t *testing.T) string {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(fl2vaRef, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return m.Prompt
}

// keyframePixels is both keyframes as the tower's pixel_values, in the
// processor's batched order.
func keyframePixels(t *testing.T, m *fl2vaManifest, vc *vision.Config) ([]*qwen.Mat, []qtextenc.Grid) {
	t.Helper()
	var pix []*qwen.Mat
	var grids []qtextenc.Grid
	for _, name := range []string{"key_first", "key_last"} {
		rgb, h, w := m.keyframe(t, name)
		planes, err := VisionPixels(rgb, h, w)
		if err != nil {
			t.Fatal(err)
		}
		p, gh, gw, err := vc.Patchify(planes, h, w)
		if err != nil {
			t.Fatal(err)
		}
		pix = append(pix, p)
		grids = append(grids, qtextenc.Grid{T: 1, H: gh, W: gw})
	}
	return pix, grids
}

// TestVisionPixels checks the processor's pixel_values bit for bit.
func TestVisionPixels(t *testing.T) {
	m := loadFL2VA(t)
	vc := visionConfig(t)
	pix, _ := keyframePixels(t, m, vc)
	want, _ := m.floats(t, "pixel_values_fl")
	got := append(append([]float32(nil), pix[0].Data...), pix[1].Data...)
	if len(got) != len(want) {
		t.Fatalf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("value %d: %v, want %v", i, got[i], want[i])
		}
	}
}

// towerTol bounds the tower's merged rows and deepstack features against
// the fp32 tower, relative to absmax. Measured 2026-09-26 on the two
// keyframes: merged 3.9e-2, deepstack 4.5e-3 / 1.8e-2 / 3.0e-2.
const towerTol = 0.1

// forcedTol bounds the conditioning with the oracle's fp32 tower rows fed
// in. Measured 2026-09-26: 9.3e-4 (f) and 3.3e-4 (fl), against t2va's
// 1.1e-4 on the README prompt; the image rows enter fp16 activations at
// the tower's scale, and the 1-D-positions control lands far past it.
const forcedTol = 2e-3

// deviceTol bounds the conditioning with the device's own tower in it. The
// tower's fp16 carries through: 7.4e-3 (f) and 1.0e-3 (fl) measured
// 2026-09-26, where the released bf16 pipeline is at 1.1–1.2e-2 against the
// same fp32 walk. The teacher-forced arm, fed the oracle's tower rows, is
// held to t2va's gpuTol.
const deviceTol = 1e-2

// TestGPUPresentation is the conditioner gate for fl2va: the tower on both
// keyframes against its fp32 run, then hidden_states[50] over the "f" and
// "fl" presentations against the fp32 walk, with the tower's own output
// scattered in. ~51 GB on the device.
func TestGPUPresentation(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 50 GB of fp16 banks and the vision tower")
	}
	m := loadFL2VA(t)
	vc := visionConfig(t)
	tok, err := tokenizer.Load(modelDir + "/tokenizer")
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder (%v)", err)
	}
	mrope, err := qtextenc.LoadMRope(encoder)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	defer done()

	pix, grids := keyframePixels(t, m, vc)
	cpuTower, err := vision.Load(encoder, vc, 0)
	if err != nil {
		t.Fatal(err)
	}
	tower, err := vision.NewGPU(dev, cpuTower, pix[0].Rows)
	if err != nil {
		t.Fatal(err)
	}
	defer tower.Destroy()
	var conds []qtextenc.Condition
	for i, p := range pix {
		start := time.Now()
		out, err := tower.Forward(context.Background(), p, grids[i].H, grids[i].W)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("tower %d: %d patches in %v", i, p.Rows, time.Since(start).Round(time.Millisecond))
		conds = append(conds, qtextenc.Condition{Merged: out.Merged, Deepstack: out.Deepstack})
	}
	// The dump batches both keyframes: rows of the first, then the second.
	check := func(name string, got []*qwen.Mat) {
		want := m.mat(t, name)
		all := &qwen.Mat{Rows: 0, Cols: got[0].Cols}
		for _, g := range got {
			all.Data = append(all.Data, g.Data...)
			all.Rows += g.Rows
		}
		rel, maxAbs, rms := gap(all, want)
		t.Logf("%-18s rel %.3g (max abs %.4g, rms %.3g)", name, rel, maxAbs, rms)
		// The tower is qimage/vision's fp16 device port, unchanged, whose
		// own gate prices fp16 at 0.13–0.56 of its measure on Qwen-Image's
		// cards; the encoder's output below is what this test gates.
		if rel > towerTol || math.IsNaN(rel) {
			t.Errorf("%s: rel %.3g", name, rel)
		}
	}
	check("vis_merged_fl", []*qwen.Mat{conds[0].Merged, conds[1].Merged})
	for l := range conds[0].Deepstack {
		check("vis_deepstack"+string(rune('0'+l))+"_fl", []*qwen.Mat{conds[0].Deepstack[l], conds[1].Deepstack[l]})
	}

	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	start := time.Now()
	g, err := qwen.NewGPUEncoder(dev, set, cfg, Layers, 1280, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged %d layers in %v", Layers, time.Since(start).Round(time.Millisecond))
	// The oracle's own fp32 tower rows, split per keyframe: the
	// teacher-forced arm, which isolates the encoder's handling of them
	// (scatter, deepstack, 3-D positions) from the tower's fp16.
	oracle := make([]qtextenc.Condition, 2)
	split := func(name string) [2]*qwen.Mat {
		all := m.mat(t, name)
		h := all.Rows / 2
		return [2]*qwen.Mat{
			{Rows: h, Cols: all.Cols, Data: all.Data[:h*all.Cols]},
			{Rows: h, Cols: all.Cols, Data: all.Data[h*all.Cols:]},
		}
	}
	merged := split("vis_merged_fl")
	for i := range oracle {
		oracle[i].Merged = merged[i]
	}
	for l := range conds[0].Deepstack {
		d := split("vis_deepstack" + string(rune('0'+l)) + "_fl")
		for i := range oracle {
			oracle[i].Deepstack = append(oracle[i].Deepstack, d[i])
		}
	}
	for _, label := range []string{"f", "fl"} {
		n := len(label)
		p, err := NewPresentation(tok, readmePrompt(t), grids[:n], vc.SpatialMergeSize)
		if err != nil {
			t.Fatal(err)
		}
		rope, err := p.MRope(cfg.HeadDim, cfg.RopeTheta, mrope)
		if err != nil {
			t.Fatal(err)
		}
		want := m.mat(t, label+"_fp32")
		brel, _, _ := gap(m.mat(t, label+"_bf16"), want)
		for _, arm := range []struct {
			name  string
			conds []qtextenc.Condition
			tol   float64
		}{{"teacher-forced", oracle[:n], forcedTol}, {"device tower", conds[:n], deviceTol}} {
			start := time.Now()
			out, _, err := p.EncodeGPU(g, rope, arm.conds, 0, nil)
			if err != nil {
				t.Fatal(err)
			}
			took := time.Since(start)
			rel, maxAbs, rms := gap(out, want)
			t.Logf("%-2s %4d tokens, %s, in %v: rel %.3g (max abs %.4g, rms %.3g); official bf16 rel %.3g",
				label, out.Rows, arm.name, took.Round(time.Millisecond), rel, maxAbs, rms, brel)
			if rel > arm.tol || math.IsNaN(rel) {
				t.Errorf("%s, %s: rel %.3g > %.0e", label, arm.name, rel, arm.tol)
			}
		}
		// The negative control: the same oracle rows under plain 1-D text
		// positions, i.e. an encoder that forgot the image blocks are 3-D.
		flat := *p.EditPrompt
		for a := range flat.Pos {
			flat.Pos[a] = make([]int32, len(p.IDs))
			for i := range flat.Pos[a] {
				flat.Pos[a][i] = int32(i)
			}
		}
		frope, err := flat.MRope(cfg.HeadDim, cfg.RopeTheta, mrope)
		if err != nil {
			t.Fatal(err)
		}
		out, _, err := p.EncodeGPU(g, frope, oracle[:n], 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		rel, _, rms := gap(out, want)
		t.Logf("%-2s negative control, 1-D positions: rel %.3g (rms %.3g)", label, rel, rms)
		if rel < 10*gpuTol {
			t.Errorf("%s: 1-D positions land at rel %.3g; the gate cannot tell them from 3-D", label, rel)
		}
	}
}
