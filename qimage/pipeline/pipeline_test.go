package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference comes from reference/dump_qi21_run.py: the whole diffusers
// pipeline in fp32 on CPU at 256²/4 steps, kv-cache on, the noise passed in
// packed so this side gets it as data. Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_run.py
const (
	runRef      = "../../reference/out/qi21run"
	transformer = "../../models/Qwen-Image-2.1/transformer"
)

type runManifest struct {
	Prompt  string `json:"prompt"`
	Size    int    `json:"size"`
	Steps   int    `json:"steps"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadRunManifest(t *testing.T) *runManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(runRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_run.py", runRef, err)
	}
	var m runManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadRunMat(t *testing.T, m *runManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(runRef, name+".bin"))
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

// stepTol is the fp32 bound for latents after a full 32-block step, set the
// way textenc's deepTol was: from measured drift. Both sides are fp32; the
// drift is summation order compounded through 32 blocks and the sampler's
// feedback, measured at rel 2.1e-5 after step 0 and 2.5e-4 after step 3
// (2026-09-20); the bound is 4x the worst measured step.
const stepTol = 1e-3

func compareStep(t *testing.T, name string, got, want *qwen.Mat) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	if rel > stepTol {
		t.Errorf("%s %s: max abs %.6g, worst rel %.3g, rms %.6g > %.0e", name, got, maxAbs, rel, rms, stepTol)
		return
	}
	t.Logf("%-16s %-14s max abs %.3g  rms %.4g  rel %.2g", name, got.String(), maxAbs, rms, rel)
}

// TestDenoiseOracle is Q3's latent-space gate: the full 32-block stack under
// the sampler, prefill + cached exactly as served, from the oracle's own
// noise and prompt embeddings, compared per step. Loads 28 GB of fp32
// weights and takes minutes per step; the image half of the gate lands with
// Q5's VAE decoder.
func TestDenoiseOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("loads 28 GB of fp32 weights and runs a four-step sample")
	}
	m := loadRunManifest(t)
	cfg, err := dit.LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	model, err := dit.Load(transformer, cfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	scfg, err := LoadSchedConfig("../../models/Qwen-Image-2.1/scheduler")
	if err != nil {
		t.Fatal(err)
	}

	noise := loadRunMat(t, m, "noise")
	txt := loadRunMat(t, m, "prompt_embeds")
	side := m.Size / 16
	lay, err := dit.NewLayout([]int{txt.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}

	final, err := Denoise(model, txt, lay, nil, noise, sched, func(i int, latents *qwen.Mat) {
		compareStep(t, fmt.Sprintf("step%d_latents", i), latents, loadRunMat(t, m, fmt.Sprintf("step%d_latents", i)))
	})
	if err != nil {
		t.Fatal(err)
	}
	compareStep(t, "final", final, loadRunMat(t, m, fmt.Sprintf("step%d_latents", m.Steps-1)))
}
