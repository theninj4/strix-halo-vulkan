package plan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const baseSamplerRef = "../../reference/out/acebasesampler"

type baseSamplerManifest struct {
	Request map[string]any `json:"request"`
	Sampler map[string]struct {
		T         int       `json:"T"`
		Steps     int       `json:"steps"`
		Shift     float64   `json:"shift"`
		Guidance  float64   `json:"guidance"`
		Interval  []float64 `json:"interval"`
		DCW       bool      `json:"dcw"`
		Timesteps []float64 `json:"timesteps"`
		Rows      []int     `json:"rows"`
	} `json:"sampler"`
	Tensors map[string]struct {
		Shape []int  `json:"shape"`
		Dtype string `json:"dtype"`
	} `json:"tensors"`
}

func loadBaseSampler(t *testing.T) (*baseSamplerManifest, *planManifest) {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(baseSamplerRef, "manifest.json"))
	if err != nil {
		t.Skipf("no base sampler oracle (run reference/dump_ace_base_sampler.py): %v", err)
	}
	var m baseSamplerManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	// readRaw reads through a planManifest: hand it this oracle's tensors.
	pm := &planManifest{dir: baseSamplerRef}
	pm.Tensors = m.Tensors
	return &m, pm
}

// TestBaseRequest: what upstream's generate_music hands the base model for
// a request at Gradio's pure-base settings is what DefaultGuidance and
// BaseSteps say.
func TestBaseRequest(t *testing.T) {
	m, _ := loadBaseSampler(t)
	r := m.Request
	for k, want := range map[string]any{
		"infer_steps": float64(BaseSteps), "diffusion_guidance_scale": DefaultGuidance.Scale,
		"use_adg": false, "cfg_interval_start": DefaultGuidance.IntervalStart, "cfg_interval_end": DefaultGuidance.End,
		"dcw_enabled": false, "infer_method": "ode", "sampler_mode": "euler",
	} {
		if r[k] != want {
			t.Errorf("%s = %v, want %v", k, r[k], want)
		}
	}
}

// TestBaseSampler: the base model's whole generate_audio loop over a stub
// decoder whose conditional row is 0.5·x + t·c and null row 0.4·x + t·d,
// over step counts, shifts, scales, an interval and DCW -- bit-exact, and
// the null row asked for exactly where upstream's guidance used it.
func TestBaseSampler(t *testing.T) {
	m, pm := loadBaseSampler(t)
	if len(m.Sampler) == 0 {
		t.Fatal("no sampler cases")
	}
	for label, s := range m.Sampler {
		sched := BaseSchedule(s.Steps, s.Shift)
		if len(sched) != len(s.Timesteps)+1 || sched[len(sched)-1] != 0 {
			t.Fatalf("%s: %d entries, want %d calls and a final 0", label, len(sched), len(s.Timesteps))
		}
		for i, want := range s.Timesteps {
			if float64(sched[i]) != want {
				t.Fatalf("%s: timestep %d = %v, want %v", label, i, sched[i], want)
			}
		}
		g := Guidance{Scale: s.Guidance, IntervalStart: s.Interval[0], End: s.Interval[1]}
		if s.DCW {
			g.DCW = DefaultDCW
		}
		c, d := f32s(t, pm, label+"_c"), f32s(t, pm, label+"_d")
		var nulls, wantNulls int
		for i, rows := range s.Rows {
			if rows == 2 && g.on(float32(s.Timesteps[i])) {
				wantNulls++
			}
		}
		stub := func(x []float32, tt float32, uncond bool) ([]float32, []float32, error) {
			cond := make([]float32, len(x))
			for i := range cond {
				cond[i] = 0.5*x[i] + float32(tt*c[i])
			}
			if !uncond {
				return cond, nil, nil
			}
			nulls++
			null := make([]float32, len(x))
			for i := range null {
				null[i] = float32(float32(0.4)*x[i]) + float32(tt*d[i])
			}
			return cond, null, nil
		}
		out, err := SampleGuided(f32s(t, pm, label+"_noise"), sched, g, stub, nil)
		if err != nil {
			t.Fatal(err)
		}
		if nulls != wantNulls {
			t.Errorf("%s: %d null rows, upstream used %d", label, nulls, wantNulls)
		}
		exact32(t, label, out, f32s(t, pm, label+"_out"))
	}
}

// TestCascadeSum: the cascade's bookkeeping at the lengths where its levels
// fold (16, 256, 4096 rows) and either side, against a plain sum of values
// that sum exactly in any order.
func TestCascadeSum(t *testing.T) {
	for _, rows := range []int{1, 15, 16, 17, 255, 256, 257, 4095, 4096, 4097, 7500} {
		a := make([]float64, rows*3)
		for i := range a {
			a[i] = float64(i%7) - 3
		}
		got := cascadeSum(a, rows, 3)
		for c := 0; c < 3; c++ {
			var want float64
			for r := 0; r < rows; r++ {
				want += a[r*3+c]
			}
			if got[c] != want {
				t.Errorf("rows %d col %d: %v, want %v", rows, c, got[c], want)
			}
		}
	}
}
