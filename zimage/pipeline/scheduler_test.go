package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The reference is reference/dump_zimage.py, which drives the checkpoint's own
// FlowMatchEulerDiscreteScheduler the way the pipeline does:
//
//	.venv/bin/python reference/dump_zimage.py
const (
	refDir       = "../../reference/out/zimage"
	runDir       = "../../reference/out/zimagerun"
	schedulerDir = "../../models/Z-Image-Turbo/scheduler"
)

type manifest struct {
	dir   string
	Steps int `json:"steps"`
	// Size and Prompt are dump_zimage_run.py's; dump_zimage.py carries
	// neither, which is how a test says which dump it wants.
	Size    int    `json:"size"`
	Prompt  string `json:"prompt"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadManifest(t *testing.T, dir string) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_zimage.py or dump_zimage_run.py", dir, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = dir
	return &m
}

func loadRef(t *testing.T, m *manifest, name string) []float32 {
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
	out := make([]float32, meta.Count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// schedTol is loose next to the rest of the pipeline's bounds because these
// are absolute comparisons of numbers of order 1 and 1000 rather than
// RMS-normalised tensors: diffusers builds the schedule in float32 and this
// builds it in float64, so the last bit of a timestep differs.
const schedTol = 1e-4

func newSchedule(t *testing.T, steps int) *FlowMatchEuler {
	t.Helper()
	cfg, err := LoadSchedulerConfig(schedulerDir)
	if err != nil {
		t.Skipf("no scheduler config at %s (%v)", schedulerDir, err)
	}
	s, err := NewFlowMatchEuler(steps, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestSchedule checks the eight sigmas, their timesteps and the conditioning
// value each step sends the transformer. The schedule is eight numbers that
// decide what every one of the 8 x 34 block runs is conditioned on, so a wrong
// one is a worse image and nothing else -- there is no shape to catch it.
func TestSchedule(t *testing.T) {
	m := loadManifest(t, refDir)
	s := newSchedule(t, m.Steps)

	for _, c := range []struct {
		name string
		got  []float64
	}{
		{"sigmas", s.Sigmas},
		{"timesteps", s.Timesteps},
		{"model_t", modelT(s)},
	} {
		want := loadRef(t, m, c.name)
		if len(c.got) != len(want) {
			t.Fatalf("%s: %d values, want %d", c.name, len(c.got), len(want))
		}
		var worst float64
		for i := range want {
			scale := math.Max(math.Abs(float64(want[i])), 1)
			if d := math.Abs(c.got[i]-float64(want[i])) / scale; d > worst {
				worst = d
			}
		}
		if worst > schedTol {
			t.Errorf("%s: worst relative difference %.3g\n got  %v\n want %v", c.name, worst, c.got, want)
			continue
		}
		t.Logf("%-10s worst %.2g", c.name, worst)
	}

	if s.Sigmas[len(s.Sigmas)-1] != 0 {
		t.Errorf("the schedule does not terminate at sigma 0: %v", s.Sigmas)
	}
}

func modelT(s *FlowMatchEuler) []float64 {
	out := make([]float64, s.Steps())
	for i := range out {
		out[i] = s.ModelT(i)
	}
	return out
}

// TestStep checks the Euler update itself against the scheduler's own.
func TestStep(t *testing.T) {
	m := loadManifest(t, refDir)
	s := newSchedule(t, m.Steps)

	sample := loadRef(t, m, "step_sample")
	out := loadRef(t, m, "step_model_out")
	want := loadRef(t, m, "step_result")

	got := append([]float32(nil), sample...)
	if err := s.Step(0, got, out); err != nil {
		t.Fatal(err)
	}
	var worst float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst = d
		}
	}
	if worst > schedTol {
		t.Fatalf("step: worst absolute difference %.3g", worst)
	}
	t.Logf("step       worst %.2g", worst)

	// The negative control: the step with the shifted schedule's *unshifted*
	// sigmas, which is the mistake a scheduler config invites -- the shift is
	// in the config and nothing about a sigma says whether it was applied.
	raw := &FlowMatchEuler{Sigmas: make([]float64, len(s.Sigmas)), TrainTimesteps: s.TrainTimesteps,
		Timesteps: s.Timesteps, Shift: 1}
	for i := range raw.Sigmas {
		raw.Sigmas[i] = s.Sigmas[i] / (s.Shift - (s.Shift-1)*s.Sigmas[i])
	}
	broken := append([]float32(nil), sample...)
	if err := raw.Step(0, broken, out); err != nil {
		t.Fatal(err)
	}
	var brokenWorst float64
	for i := range want {
		if d := math.Abs(float64(broken[i] - want[i])); d > brokenWorst {
			brokenWorst = d
		}
	}
	if brokenWorst <= schedTol {
		t.Errorf("the unshifted schedule is inside the bound (%.3g); the test would not catch it", brokenWorst)
	}
	t.Logf("unshifted  worst %.2g = %.0fx the bound", brokenWorst, brokenWorst/schedTol)
}

// TestStepCount checks the schedule at the lengths the CLI allows, since the
// linspace degenerates at one step and the terminal zero is what carries it.
func TestStepCount(t *testing.T) {
	for _, n := range []int{1, 2, 4, 8, 20, 50} {
		s := newSchedule(t, n)
		if s.Steps() != n || len(s.Sigmas) != n+1 {
			t.Fatalf("%d steps: %d timesteps, %d sigmas", n, s.Steps(), len(s.Sigmas))
		}
		for i := 0; i < n; i++ {
			if s.Sigmas[i] <= s.Sigmas[i+1] {
				t.Fatalf("%d steps: sigma %d (%g) does not exceed sigma %d (%g)", n, i, s.Sigmas[i], i+1, s.Sigmas[i+1])
			}
		}
		if s.Sigmas[0] != 1 {
			t.Errorf("%d steps: starts at sigma %g, want 1", n, s.Sigmas[0])
		}
	}
}
