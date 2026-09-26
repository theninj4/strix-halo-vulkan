package dit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// The fl2va reference is reference/dump_h3_fl2va.py's dit phase: the same
// fp32 folded transformer as M4's, over a 5,709-row fl2va (the README fl2va
// prompt behind one first keyframe, 1,039 conditioning rows of which 114 are
// a vision block tagged video; 112 keyframe rows at t = 0.999 ahead of the
// 4,144 generated video rows; 414 audio), two forwards of N = 8.
// Regenerate with:
//
//	.venv/bin/python reference/dump_h3_fl2va.py prep vae text dit   (~85 GB, ~15 min)
const fl2vaRef = "../../reference/out/h3fl2va"

type fl2vaManifest struct {
	Presentations map[string]struct {
		Tags []int32 `json:"tags"`
	} `json:"presentations"`
	DiT struct {
		TextTokens   int `json:"text_tokens"`
		LatentFrames int `json:"latent_frames"`
		LatentHeight int `json:"latent_height"`
		LatentWidth  int `json:"latent_width"`
		AudioLatents int `json:"audio_latents"`
		Rows         int `json:"rows"`
		CondRows     int `json:"cond_rows"`
		Steps        int `json:"steps"`
		Forwards     []struct {
			Timestep float32   `json:"timestep"`
			Unique   []float32 `json:"unique"`
		} `json:"forwards"`
	} `json:"dit"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

// TestGPUFL2VA is M10's transformer gate: the keyframe rows and the
// presentation's video-tagged vision rows through the unchanged GPU
// transformer, teacher-forced against the oracle — blocks 0 and 49 alone,
// then both dumped forwards end to end, with only the generated rows
// stepped. ~44 GB on the device.
func TestGPUFL2VA(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 42 GB")
	}
	buf, err := os.ReadFile(filepath.Join(fl2vaRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v)", err)
	}
	var m fl2vaManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	d := m.DiT
	if len(d.Forwards) == 0 {
		t.Skip("the dump has no dit phase; run reference/dump_h3_fl2va.py dit")
	}
	ref := func(name string) *qwen.Mat { return readRef(t, fl2vaRef, m.Tensors, name) }
	lay, err := plan.NewLayout(m.Presentations["f"].Tags, d.LatentFrames, d.LatentHeight, d.LatentWidth, d.AudioLatents,
		[]plan.Anchor{plan.First})
	if err != nil {
		t.Fatal(err)
	}
	if len(lay.Pos) != d.Rows || lay.CondVideoRows != d.CondRows {
		t.Fatalf("layout %d rows (%d keyframe), the oracle's %d (%d)", len(lay.Pos), lay.CondVideoRows, d.Rows, d.CondRows)
	}
	vs, err := plan.NewSchedule(d.Steps, 12)
	if err != nil {
		t.Fatal(err)
	}
	as, err := plan.NewSchedule(d.Steps, 3)
	if err != nil {
		t.Fatal(err)
	}
	var tvals []float32
	rowTs := make([][]int32, len(d.Forwards))
	for f := range d.Forwards {
		u, idx := lay.RowTimesteps(vs.Timesteps[f], as.Timesteps[f])
		if len(u) != len(d.Forwards[f].Unique) {
			t.Fatalf("forward %d: timesteps %v, the oracle's %v", f, u, d.Forwards[f].Unique)
		}
		for i := range u {
			if u[i] != d.Forwards[f].Unique[i] {
				t.Fatalf("forward %d: timesteps %v, the oracle's %v", f, u, d.Forwards[f].Unique)
			}
			if indexOf(tvals, u[i]) < 0 {
				tvals = append(tvals, u[i])
			}
		}
		rowTs[f] = make([]int32, len(idx))
		for i, j := range idx {
			rowTs[f][i] = int32(indexOf(tvals, u[j]))
		}
	}
	start := time.Now()
	tabs, err := Tables(modelDir, tvals)
	if err != nil {
		t.Skipf("no weights (%v)", err)
	}
	t.Logf("AdaLN tables for %d timesteps %v in %v", len(tvals), tvals, time.Since(start).Round(time.Millisecond))

	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPU(dev, modelDir, len(lay.Pos), d.TextTokens, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.Begin(lay, ref("f_fp32"), tvals, tabs); err != nil {
		t.Fatal(err)
	}
	for _, b := range []int{0, 49} {
		out, err := g.Blocks(ref(fmt.Sprintf("f0_block%d_in", b)), rowTs[0], b, b+1)
		if err != nil {
			t.Fatal(err)
		}
		rel, rms := relGap(out, ref(fmt.Sprintf("f0_block%d_out", b)))
		t.Logf("block %d, teacher-forced: rel %.3g rms %.3g", b, rel, rms)
		if rel > 1e-3 || math.IsNaN(rel) {
			t.Errorf("block %d: rel %.3g", b, rel)
		}
	}

	cond := ref("cond_rows")
	video := &qwen.Mat{Rows: len(lay.Video), Cols: cond.Cols,
		Data: append(append([]float32(nil), cond.Data...), ref("noise_video").Data...)}
	audio := ref("noise_audio")
	gen := lay.CondVideoRows * video.Cols
	for f := range d.Forwards {
		if f > 0 {
			video, audio = ref(fmt.Sprintf("f%d_latents", f-1)), ref(fmt.Sprintf("f%d_audio", f-1))
		}
		v, a, took, err := g.Step(video, audio, rowTs[f])
		if err != nil {
			t.Fatal(err)
		}
		wantV := ref(fmt.Sprintf("f%d_v_video", f))
		rv, rvs := relGap(v, wantV)
		// The generated rows alone, which are what the sampler reads.
		rg, rgs := relGap(&qwen.Mat{Rows: v.Rows - lay.CondVideoRows, Cols: v.Cols, Data: v.Data[gen:]},
			&qwen.Mat{Rows: v.Rows - lay.CondVideoRows, Cols: v.Cols, Data: wantV.Data[gen:]})
		ra, ras := relGap(a, ref(fmt.Sprintf("f%d_v_audio", f)))
		t.Logf("forward %d (%v): video velocity rel %.3g rms %.3g (generated rows rel %.3g rms %.3g), audio rel %.3g rms %.3g",
			f, took.Round(time.Millisecond), rv, rvs, rg, rgs, ra, ras)
		vs.Step(f, v.Data[gen:], video.Data[gen:])
		as.Step(f, a.Data, audio.Data)
		relV, rmsV := relGap(video, ref(fmt.Sprintf("f%d_latents", f)))
		relA, rmsA := relGap(audio, ref(fmt.Sprintf("f%d_audio", f)))
		t.Logf("step %d: video latents rel %.3g rms %.3g; audio rel %.3g rms %.3g", f, relV, rmsV, relA, rmsA)
		if rmsV > stepRMS || rmsA > stepRMS || math.IsNaN(relV+relA) {
			t.Errorf("step %d: video latents rms %.3g, audio rms %.3g > %.0e", f, rmsV, rmsA, stepRMS)
		}
		for i := 0; i < gen; i++ {
			if video.Data[i] != cond.Data[i] {
				t.Fatalf("step %d moved keyframe value %d", f, i)
			}
		}
	}
}
