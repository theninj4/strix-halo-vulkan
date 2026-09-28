package layout

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/safetensors"
)

const (
	modelDir = "../../models/PP-DocLayoutV3"
	refDir   = "../../reference/out/doclayout"
)

type detection struct {
	Query   int       `json:"query"`
	Label   string    `json:"label"`
	LabelID int       `json:"label_id"`
	Score   float64   `json:"score"`
	Box     []float64 `json:"box"`
	Order   int       `json:"order"`
}

type record struct {
	Name       string      `json:"name"`
	Image      string      `json:"image"`
	Size       [2]int      `json:"size"`
	Threshold  float64     `json:"threshold"`
	Detections []detection `json:"detections"`
}

func records(t *testing.T) []record {
	t.Helper()
	paths, _ := filepath.Glob(filepath.Join(refDir, "*", "record.json"))
	if len(paths) == 0 {
		t.Skip("no reference; run reference/dump_doclayout.py")
	}
	var out []record
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var r record
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func refTensors(t *testing.T, name string) map[string][]float32 {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(refDir, name, "tensors.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string][]float32{}
	for _, n := range f.Names() {
		ten, err := f.Get(n)
		if err != nil {
			t.Fatal(err)
		}
		if out[n], err = ten.F32(nil); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// cpuTol bounds every stage of the fp32 port against HF's fp32: the worst
// measured over the five cases is 2.5e-5 (the formula's masks).
const cpuTol = 1e-4

// checkDetections holds regions to the dump's: the same (query, label)
// pairs in the same order, scores within scoreTol and boxes within boxTol px.
func checkDetections(t *testing.T, rec record, got []Region, scoreTol, boxTol float64) {
	t.Helper()
	if len(got) != len(rec.Detections) {
		t.Errorf("%d regions, the reference has %d", len(got), len(rec.Detections))
		return
	}
	var ds, db float64
	for i, w := range rec.Detections {
		g := got[i]
		// The regions come sorted by order, so the sequence is compared
		// everywhere; the absolute rank among all 300 queries only at fp32,
		// since a junk query's vote can move it by one under fp16. Even at
		// fp32 it may move by one: HF sums the votes in float32 and argsorts
		// unstably, so a near-tie with an undetected query is noise (the
		// OmniDocBench page reference/dump_doclayout.py --case figtab1 has
		// one, O10).
		if g.Query != w.Query || g.LabelID != w.LabelID || (scoreTol < 1e-3 && abs(g.Order-w.Order) > 1) {
			t.Errorf("region %d: query %d %s order %d, want query %d %s order %d",
				i, g.Query, g.Label, g.Order, w.Query, w.Label, w.Order)
			continue
		}
		ds = max(ds, math.Abs(g.Score-w.Score))
		for k := range g.Box {
			db = max(db, math.Abs(g.Box[k]-w.Box[k]))
		}
	}
	t.Logf("%d regions; worst score %.1e, worst box %.3f px", len(got), ds, db)
	if ds > scoreTol || db > boxTol {
		t.Errorf("scores or boxes moved: %.1e, %.3f px", ds, db)
	}
}

// TestPreprocess resizes each case's image and holds it to the dump's
// pixels: PNG must be exact, JPEG (Go's decoder, not libjpeg) is reported.
func TestPreprocess(t *testing.T) {
	for _, rec := range records(t) {
		data, err := os.ReadFile(filepath.Join("../..", rec.Image))
		if err != nil {
			t.Fatal(err)
		}
		img, format, err := pixels.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		got := Preprocess(img)
		want := refTensors(t, rec.Name)["pixels"]
		off, worst := 0, 0.0
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > 0 {
				off++
			}
			worst = max(worst, d*255)
		}
		t.Logf("%s (%s): %d of %d values off, max %.0f levels", rec.Name, format, off, len(want), worst)
		if format != "jpeg" && off != 0 {
			t.Errorf("%s: a %s must be exact", rec.Name, format)
		}
	}
}

var loaded *Model

func load(t *testing.T) *Model {
	t.Helper()
	if loaded == nil {
		m, err := Load(modelDir)
		if err != nil {
			t.Skipf("no checkpoint (%v)", err)
		}
		loaded = m
	}
	return loaded
}

// TestForwardCPU runs every dumped case from the dump's own pixels and
// reports every stage against HF's fp32.
func TestForwardCPU(t *testing.T) {
	m := load(t)
	for _, rec := range records(t) {
		t.Run(rec.Name, func(t *testing.T) {
			ref := refTensors(t, rec.Name)
			start := time.Now()
			worst := 0.0
			out, err := m.Forward(ref["pixels"], func(name string, got []float32) {
				want, ok := ref[name]
				if !ok {
					t.Errorf("%s: not in the dump", name)
					return
				}
				if len(want) != len(got) {
					t.Errorf("%s: %d values, dump has %d", name, len(got), len(want))
					return
				}
				if name == "enc.coord" {
					// An invalid anchor's logits are float32's max on both
					// sides; compare the valid rows.
					var g, w []float32
					for i := range want {
						if math.Abs(float64(want[i])) < 1e30 {
							g, w = append(g, got[i]), append(w, want[i])
						}
					}
					got, want = g, w
				}
				e := relErr(got, want)
				if name == "enc.topk" && e != 0 {
					t.Errorf("enc.topk: a different selection of queries")
				}
				worst = max(worst, e)
				if e > cpuTol {
					t.Errorf("%s: rel %.2e over %.0e", name, e, cpuTol)
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("forward %v, worst stage rel %.2e", time.Since(start).Round(time.Millisecond), worst)
			checkDetections(t, rec, out.Detect(rec.Size[1], rec.Size[0], rec.Threshold), 1e-4, 0.05)
		})
	}
}

// TestDetect runs every case end to end from its image file and holds the
// regions to HF's. The JPEG's decode is two levels off on 0.4% of values
// (TestPreprocess), which is let through if it moves nothing.
func TestDetect(t *testing.T) {
	m := load(t)
	for _, rec := range records(t) {
		t.Run(rec.Name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("../..", rec.Image))
			if err != nil {
				t.Fatal(err)
			}
			img, _, err := pixels.Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			got, err := m.Detect(img, rec.Threshold)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%v", time.Since(start).Round(time.Millisecond))
			checkDetections(t, rec, got, 1e-4, 0.05)
		})
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
