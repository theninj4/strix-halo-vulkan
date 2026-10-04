package gemma4

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/decide"
	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/safetensors"
)

// The vision oracle: reference/dump_rune_vision.py (research/rune-vertical.md R10).
const visionOut = "../reference/out/rune/vision"

type visionMeta struct {
	Name       string
	Text       string
	System     string
	User       string
	IDs        []int32 `json:"ids"`
	Labels     []string
	LabelIDs   []int32   `json:"label_ids"`
	LabelLogit []float64 `json:"label_logits"`
	ProbsT1    []float64 `json:"probs_T1"`
	Images     []string
	Soft       []int `json:"soft_tokens"`
}

func visionCases(t *testing.T, dir string) []visionMeta {
	t.Helper()
	metas, _ := filepath.Glob(filepath.Join(dir, "*", "*", "meta.json"))
	if len(metas) == 0 {
		t.Skip("no " + dir + " (run reference/dump_rune_vision.py)")
	}
	var out []visionMeta
	for _, mp := range metas {
		raw, err := os.ReadFile(mp)
		if err != nil {
			t.Fatal(err)
		}
		var m visionMeta
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		out = append(out, m)
	}
	return out
}

func caseRows(t *testing.T, dir, name string) *safetensors.File {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(dir, name, "rows.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func refF32(t *testing.T, f *safetensors.File, name string) []float32 {
	t.Helper()
	ref, err := f.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	v, err := ref.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func loadImage(t *testing.T, path string) *pixels.RGB {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", path))
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := pixels.Decode(raw)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TestVisionSize holds the resize rule to the processor's branches.
func TestVisionSize(t *testing.T) {
	for _, c := range []struct{ h, w, wh, ww int }{
		{669, 1200, 576, 1056},
		{8, 3000, 48, 13440},
		{3000, 8, 13440, 48},
		{80, 100, 672, 864},
		{2016, 2880, 672, 960},
	} {
		h, w, err := VisionSize(c.h, c.w, 16, 2520, 3)
		if err != nil {
			t.Fatal(err)
		}
		if h != c.wh || w != c.ww || (h/16)*(w/16) > 2520 {
			t.Errorf("%dx%d: got %dx%d, want %dx%d", c.w, c.h, w, h, c.ww, c.wh)
		}
	}
}

// TestVisionPatchify is the processor's gate: the oracle's PNGs to the
// tower's pixel rows, bit for bit, and the patch grid.
func TestVisionPatchify(t *testing.T) {
	cfg, err := LoadVisionConfig(runeDir)
	if err != nil {
		t.Skip(err)
	}
	for _, m := range visionCases(t, visionOut) {
		t.Run(m.Name, func(t *testing.T) {
			f := caseRows(t, visionOut, m.Name)
			defer f.Close()
			for k, path := range m.Images {
				p, err := cfg.Patchify(loadImage(t, path))
				if err != nil {
					t.Fatal(err)
				}
				want := refF32(t, f, fmt.Sprintf("img%d.pixels", k))
				if len(want) != len(p.Pixels) {
					t.Fatalf("image %d: %d values, want %d", k, len(p.Pixels), len(want))
				}
				diff := 0
				for i := range want {
					if want[i] != p.Pixels[i] {
						diff++
					}
				}
				if diff > 0 {
					t.Errorf("image %d (%dx%d patches): %d of %d values differ", k, p.GW, p.GH, diff, len(want))
				}
				if p.Soft != m.Soft[k] {
					t.Errorf("image %d: %d soft tokens, want %d", k, p.Soft, m.Soft[k])
				}
			}
		})
	}
}

var (
	visOnce sync.Once
	visM    *Vision
	visErr  error
)

func loadVision(t *testing.T) *Vision {
	t.Helper()
	if testing.Short() {
		t.Skip("stages the vision tower; not in -short")
	}
	if _, err := os.Stat(filepath.Join(runeDir, "model-00011-of-00011.safetensors")); err != nil {
		t.Skip("no models/rune-26b-a4b")
	}
	if gpuM != nil && gpuM.Vis != nil {
		return gpuM.Vis
	}
	visOnce.Do(func() {
		dev, _ := newTestDevice(t)
		set, err := safetensors.OpenSet(runeDir)
		if err != nil {
			visErr = err
			return
		}
		visM, visErr = LoadVision(dev, set, runeDir, 2816)
	})
	if visErr != nil {
		t.Fatal(visErr)
	}
	return visM
}

func relRMS(got, want []float32) (rel, worst float64) {
	var e2, r2 float64
	for i := range want {
		d := float64(got[i] - want[i])
		e2 += d * d
		r2 += float64(want[i]) * float64(want[i])
		worst = max(worst, math.Abs(d))
	}
	return math.Sqrt(e2 / r2), worst
}

// TestVisionTower is the tower's gate against HF's fp32 tower over the same
// pixels: every layer's residual, the pooled rows and the projected
// features, by relative RMS.
func TestVisionTower(t *testing.T) {
	v := loadVision(t)
	for _, m := range visionCases(t, visionOut) {
		t.Run(m.Name, func(t *testing.T) {
			f := caseRows(t, visionOut, m.Name)
			defer f.Close()
			for k, path := range m.Images {
				p, err := v.Cfg.Patchify(loadImage(t, path))
				if err != nil {
					t.Fatal(err)
				}
				tap := func(name string, rows []float32) {
					ref := name
					if name != "embed" && name != "pooled" {
						var i int
						fmt.Sscanf(name, "layer%d", &i)
						if i%6 != 0 && i != v.Cfg.Layers-1 {
							return
						}
					}
					want := refF32(t, f, fmt.Sprintf("img%d.%s", k, ref))
					rel, worst := relRMS(rows, want)
					t.Logf("image %d %-8s rel rms %.2e worst %.3g", k, name, rel, worst)
				}
				feats, err := v.Encode(p, tap)
				if err != nil {
					t.Fatal(err)
				}
				want := refF32(t, f, fmt.Sprintf("img%d.features", k))
				rel, worst := relRMS(feats, want)
				t.Logf("image %d features rel rms %.2e worst %.3g (%d soft tokens)", k, rel, worst, p.Soft)
				if rel > 0.02 {
					t.Errorf("image %d features off by %.3g relative", k, rel)
				}
			}
		})
	}
}

// TestVisionPromptIDs holds the image prompt to the processor's ids: the
// user turn as ImageText then the text, through the tokenizer.
func TestVisionPromptIDs(t *testing.T) {
	tok, err := LoadTokenizer(runeDir)
	if err != nil {
		t.Skip(err)
	}
	for _, m := range visionCases(t, visionOut) {
		got := tok.Encode(DecisionPrompt(m.System, ImageText(m.Soft)+m.User))
		if !slices.Equal(got, m.IDs) {
			i := 0
			for i < min(len(got), len(m.IDs)) && got[i] == m.IDs[i] {
				i++
			}
			t.Errorf("%s: %d ids, want %d; first difference at %d: %v, want %v", m.Name, len(got), len(m.IDs), i,
				got[i:min(i+4, len(got))], m.IDs[i:min(i+4, len(m.IDs))])
		}
	}
}

// TestVisionOracle is R10's model gate: each case's images through the
// tower and the prompt through the text model, the state as one segment
// and the last token as a branch, layer by layer against HF's fp32
// forward (bidirectional image blocks on the sliding layers), then the
// probabilities. The bar is TestOracle's: the same choice, p within 0.05.
func TestVisionOracle(t *testing.T) {
	g := loadGPU(t)
	if os.Getenv("RUNE_CAUSAL_IMAGES") == "1" {
		causalImages = true // the control: the drift must grow
		defer func() { causalImages = false }()
	}
	bf16 := map[string]visionMeta{}
	if _, err := os.Stat(visionOut + "-bf16"); err == nil {
		for _, m := range visionCases(t, visionOut+"-bf16") {
			bf16[m.Name] = m
		}
	}
	for _, m := range visionCases(t, visionOut) {
		t.Run(m.Name, func(t *testing.T) {
			f := caseRows(t, visionOut, m.Name)
			defer f.Close()
			var soft [][]float32
			for _, path := range m.Images {
				raw, err := os.ReadFile(filepath.Join("..", path))
				if err != nil {
					t.Fatal(err)
				}
				pic, err := g.Vis.Cfg.NewPicture(raw)
				if err != nil {
					t.Fatal(err)
				}
				if err := g.EncodePictures([]*Picture{pic}); err != nil {
					t.Fatal(err)
				}
				soft = append(soft, pic.Feats)
			}
			ids := m.IDs
			L, H := len(ids), g.Cfg.Hidden
			p, err := g.NewBatch([]Segment{{State: ids[:L-1], Branches: [][]int32{ids[L-1:]}, Soft: soft}})
			if err != nil {
				t.Fatal(err)
			}
			g.Trace = func(i int) {
				if i%6 != 0 && i != g.Cfg.Layers-1 {
					return
				}
				want := refF32(t, f, fmt.Sprintf("layer%d", i))
				got := make([]float32, 0, L*H)
				for r := range L {
					got = append(got, g.Residual(r)...)
				}
				rel, worst := relRMS(got, want)
				t.Logf("layer %2d: rel rms %.2e worst %.3g", i, rel, worst)
			}
			defer func() { g.Trace = nil }()
			took, err := g.Forward(p, 0)
			if err != nil {
				t.Fatal(err)
			}
			h := g.FinalNorm(g.Residual(p.readouts[0]))
			z := g.LabelLogits(h, m.LabelIDs)
			pr, err := decide.Probabilities(z, 1)
			if err != nil {
				t.Fatal(err)
			}
			var worstP, worstB float64
			for k := range pr {
				worstP = max(worstP, math.Abs(pr[k]-m.ProbsT1[k]))
				if b, ok := bf16[m.Name]; ok {
					worstB = max(worstB, math.Abs(b.ProbsT1[k]-m.ProbsT1[k]))
				}
			}
			t.Logf("%d tokens in %v: p %.4f, want %.4f (|d| %.3g; bf16's own %.3g)", L, took, pr, m.ProbsT1, worstP, worstB)
			if worstP > 0.05 {
				t.Errorf("probabilities off by %.3g", worstP)
			}
			if argmax(pr) != argmax(m.ProbsT1) {
				t.Errorf("chose option %d, the reference %d", argmax(pr), argmax(m.ProbsT1))
			}
		})
	}
}

func argmax(v []float64) int {
	b := 0
	for i := range v {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}

// TestVisionProfile times the tower dispatch by dispatch over the oracle's
// first image (R10's speed table), and the whole Encode.
func TestVisionProfile(t *testing.T) {
	v := loadVision(t)
	cases := visionCases(t, visionOut)
	raw, err := os.ReadFile(filepath.Join("..", cases[0].Images[0]))
	if err != nil {
		t.Fatal(err)
	}
	pic, err := v.Cfg.NewPicture(raw)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := v.Encode(pic.Patches, nil); err != nil {
			t.Fatal(err)
		}
	}
	var best time.Duration
	for i := range 5 {
		t0 := time.Now()
		if _, err := v.Encode(pic.Patches, nil); err != nil {
			t.Fatal(err)
		}
		if d := time.Since(t0); i == 0 || d < best {
			best = d
		}
	}
	prof, err := v.Profile(pic.Patches)
	if err != nil {
		t.Fatal(err)
	}
	var total time.Duration
	keys := make([]string, 0, len(prof))
	for k, d := range prof {
		keys = append(keys, k)
		total += d
	}
	slices.SortFunc(keys, func(a, b string) int { return int(prof[b] - prof[a]) })
	t.Logf("%d patches: Encode %v wall, %v GPU", pic.Patches.N(), best, total)
	for _, k := range keys {
		t.Logf("  %-18s %7.2f ms", k, float64(prof[k].Microseconds())/1000)
	}
}
