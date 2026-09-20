package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference comes from reference/dump_qi21_sched.py: diffusers' own
// FlowMatchEulerDiscreteScheduler under the pipeline's own calculate_shift,
// for five size/step cases, plus one Euler step over deterministic vectors.
// Regenerate with:
//
//	.venv/bin/python reference/dump_qi21_sched.py
const (
	schedRef = "../../reference/out/qi21sched"
	schedDir = "../../models/Qwen-Image-2.1/scheduler"
)

type schedManifest struct {
	Cases map[string]struct {
		Side   int     `json:"side"`
		Tokens int     `json:"tokens"`
		Steps  int     `json:"steps"`
		Mu     float64 `json:"mu"`
	} `json:"cases"`
	Tensors map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadSchedManifest(t *testing.T) *schedManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(schedRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_qi21_sched.py", schedRef, err)
	}
	var m schedManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadVec(t *testing.T, m *schedManifest, name string) []float32 {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(schedRef, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != meta.Count*4 {
		t.Fatalf("%s: %d bytes for %d float32", name, len(raw), meta.Count)
	}
	out := make([]float32, meta.Count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// schedTol is float32-cast noise: both sides compute the shift in higher
// precision elementwise and narrow once.
const schedTol = 2e-6

func compareVec(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var worst float64
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if r := d / math.Max(math.Abs(float64(want[i])), 1e-6); r > worst {
			worst = r
		}
	}
	if worst > schedTol {
		t.Errorf("%s: worst rel %.3g > %.0e", name, worst, schedTol)
		return
	}
	t.Logf("%-22s %3d values  rel %.2g", name, len(got), worst)
}

// TestSchedule checks mu, the shifted and terminal-stretched sigmas, and the
// timesteps for every dumped size/step case.
func TestSchedule(t *testing.T) {
	m := loadSchedManifest(t)
	cfg, err := LoadSchedConfig(schedDir)
	if err != nil {
		t.Skipf("no scheduler config at %s (%v)", schedDir, err)
	}
	for label, c := range m.Cases {
		if mu := cfg.Mu(c.Tokens); math.Abs(mu-c.Mu) > 1e-9 {
			t.Errorf("%s: mu %.9f, want %.9f", label, mu, c.Mu)
		}
		s, err := cfg.Timesteps(c.Steps, c.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		compareVec(t, label+"_sigmas", s.Sigmas, loadVec(t, m, label+"_sigmas"))
		compareVec(t, label+"_timesteps", s.Timesteps, loadVec(t, m, label+"_timesteps"))
	}
}

// TestStep checks one Euler move over the dump's deterministic vectors, and
// that the DiT's t input is timestep/1000.
func TestStep(t *testing.T) {
	m := loadSchedManifest(t)
	cfg, err := LoadSchedConfig(schedDir)
	if err != nil {
		t.Skipf("no scheduler config at %s (%v)", schedDir, err)
	}
	c := m.Cases["s256x4"]
	s, err := cfg.Timesteps(c.Steps, c.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	sample := loadVec(t, m, "step_sample")
	if err := s.Step(0, sample, loadVec(t, m, "step_model_out")); err != nil {
		t.Fatal(err)
	}
	compareVec(t, "step_result", sample, loadVec(t, m, "step_result"))

	modelT := loadVec(t, m, "step_model_t")
	got := make([]float32, s.Steps())
	for i := range got {
		got[i] = float32(s.T(i))
	}
	compareVec(t, "step_model_t", got, modelT)
}
