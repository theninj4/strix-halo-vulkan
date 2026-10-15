package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"strix-halo-vulkan/flux3/dit"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference is reference/dump_flux3_run.py t2i: diffusers' modular
// pipeline, untouched, at 1024²/16 in fp32 on the CPU, hooked for every
// transformer call's latents, times and velocity. Regenerate with
//
//	.venv-flux3/bin/python -I reference/dump_flux3_run.py t2i
const (
	runDir   = "../../reference/out/flux3run_t2i"
	editDir  = "../../reference/out/flux3run_edit"
	modelDir = "../../models/flux-3-image-klein/transformer"
)

type run struct {
	dir    string
	Prompt string `json:"prompt"`
	Steps  int    `json:"steps"`
	Calls  []struct {
		Step   int     `json:"step"`
		T      float64 `json:"timestep"`
		Target float64 `json:"target_timestep"`
	} `json:"calls"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func loadRun(t *testing.T) *run {
	t.Helper()
	return loadRunDir(t, runDir)
}

// loadRunDir reads a run dump's manifest: runDir (t2i) or editDir.
func loadRunDir(t *testing.T, dir string) *run {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no run dump in %s (%v); run reference/dump_flux3_run.py", dir, err)
	}
	r := &run{dir: dir}
	if err := json.Unmarshal(buf, r); err != nil {
		t.Fatal(err)
	}
	return r
}

func (r *run) mat(t *testing.T, name string) *qwen.Mat {
	t.Helper()
	meta, ok := r.Tensors[name]
	if !ok {
		t.Fatalf("run dump has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(r.dir, name+".bin"))
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

// TestSchedule: the times the model was handed, and the jump targets, are
// reproduced exactly (the manifest holds float32 values widened to JSON).
func TestSchedule(t *testing.T) {
	r := loadRun(t)
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		t.Fatal(err)
	}
	if s.Steps() != len(r.Calls) {
		t.Fatalf("%d steps, the run made %d calls", s.Steps(), len(r.Calls))
	}
	for i, c := range r.Calls {
		if got := s.T(i); float64(got) != c.T {
			t.Errorf("step %d: t %.9g, the run passed %.9g", i, got, c.T)
		}
		if got := s.T(i + 1); float64(got) != c.Target {
			t.Errorf("step %d: target %.9g, the run passed %.9g", i, got, c.Target)
		}
	}
	t.Logf("17 times exact: %v", s.Sigmas)
}

// TestEulerSteps rebuilds every step's latents from the previous step's and
// its dumped velocity — the update alone, no DiT — and asks for bit equality.
func TestEulerSteps(t *testing.T) {
	r := loadRun(t)
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i+1 < r.Steps; i++ {
		x := r.mat(t, fmt.Sprintf("step%02d_latents", i))
		if err := s.Step(x, r.mat(t, fmt.Sprintf("step%02d_velocity", i)), i); err != nil {
			t.Fatal(err)
		}
		want := r.mat(t, fmt.Sprintf("step%02d_latents", i+1))
		diff := 0
		for j := range want.Data {
			if x.Data[j] != want.Data[j] {
				diff++
			}
		}
		if diff != 0 {
			t.Errorf("step %d → %d: %d of %d values differ", i, i+1, diff, len(want.Data))
		}
	}
}

// TestNoise: reproducible, bf16 values, standard normal.
func TestNoise(t *testing.T) {
	a, b := Noise(4096, 128, 7), Noise(4096, 128, 7)
	var sum, sq float64
	for i, v := range a.Data {
		if v != b.Data[i] {
			t.Fatalf("seed 7 is not reproducible at %d", i)
		}
		if math.Float32bits(v)&0xffff != 0 {
			t.Fatalf("value %d (%g) is not a bf16 value", i, v)
		}
		sum += float64(v)
		sq += float64(v) * float64(v)
	}
	n := float64(len(a.Data))
	mean, std := sum/n, math.Sqrt(sq/n-(sum/n)*(sum/n))
	if math.Abs(mean) > 0.01 || math.Abs(std-1) > 0.01 {
		t.Errorf("noise mean %.4f std %.4f", mean, std)
	}
	if Noise(4096, 128, 8).Data[0] == a.Data[0] && Noise(4096, 128, 8).Data[1] == a.Data[1] {
		t.Errorf("seeds 7 and 8 agree")
	}
}

// tfTol is the teacher-forced step bound: flux3/dit's whole-stack bound
// (measured 2.5e-5 at 512², X2), now at 1024² and every point of the
// schedule.
const tfTol = 1e-4

// TestTeacherForced feeds the dumped latents of three steps (first, middle,
// last) through the whole fp32 model and gates the velocity. ~22 GB and
// ~5 min of CPU a step.
func TestTeacherForced(t *testing.T) {
	if testing.Short() {
		t.Skip("whole model in fp32 at 1024²")
	}
	r := loadRun(t)
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", modelDir)
	}
	m, err := dit.Load(modelDir, dit.Parts{Mode: 5, Single: 28})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		t.Fatal(err)
	}
	in := &dit.Input{Txt: r.mat(t, "prompt_embeds"), ImgIDs: r.mat(t, "img_ids"), TxtIDs: r.mat(t, "txt_ids")}
	for _, i := range []int{0, r.Steps / 2, r.Steps - 1} {
		in.Img = r.mat(t, fmt.Sprintf("step%02d_latents", i))
		in.T, in.TNext = float64(s.T(i)), float64(s.T(i+1))
		v, err := m.Forward(in)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, fmt.Sprintf("step %d velocity", i), v, r.mat(t, fmt.Sprintf("step%02d_velocity", i)), tfTol)
	}
}

func compare(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	var sumSq float64
	for _, w := range want.Data {
		sumSq += float64(w) * float64(w)
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var maxAbs, rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		maxAbs = math.Max(maxAbs, d)
		rel = math.Max(rel, d/math.Max(math.Abs(float64(want.Data[i])), rms))
	}
	if rel > tol || math.IsNaN(rel) {
		t.Errorf("%s: max abs %.4g, worst rel %.3g, rms %.4g > %.0e", name, maxAbs, rel, rms, tol)
		return
	}
	t.Logf("%-18s max abs %.3g  rms %.4g  rel %.2g", name, maxAbs, rms, rel)
}

const (
	vaeDir     = "../../models/flux-3-image-klein/vae"
	vaeDumpDir = "../../reference/out/flux3vae"
)

// TestToVAE finishes the run's last step from its dump and de-normalises and
// unpatchifies it, asking for exactly the tensor the run's VAE received.
func TestToVAE(t *testing.T) {
	r := loadRun(t)
	b, err := LoadBN(vaeDir)
	if err != nil {
		t.Skipf("no vae at %s (%v)", vaeDir, err)
	}
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		t.Fatal(err)
	}
	last := r.Steps - 1
	x := r.mat(t, fmt.Sprintf("step%02d_latents", last))
	if err := s.Step(x, r.mat(t, fmt.Sprintf("step%02d_velocity", last)), last); err != nil {
		t.Fatal(err)
	}
	got, err := b.ToVAE(x, 64, 64)
	if err != nil {
		t.Fatal(err)
	}
	// v*std + mean cancels near zero, so the ulp that measures it is the
	// operands' — |v|*std + |mean| — not the result's.
	abs := &BN{Mean: make([]float32, len(b.Mean)), Std: b.Std}
	for i, m := range b.Mean {
		abs.Mean[i] = float32(math.Abs(float64(m)))
	}
	xa := x.Clone()
	for i, v := range xa.Data {
		xa.Data[i] = float32(math.Abs(float64(v)))
	}
	scale, err := abs.ToVAE(xa, 64, 64)
	if err != nil {
		t.Fatal(err)
	}
	ulpGateScaled(t, "vae_in", got, r.mat(t, "vae_in").Data, scale)
}

// bnUlps is the bridge's bound, in float32 ulps of each value. It is not 0
// because torch's AVX-512 float32 sqrt is not correctly rounded: it differs
// from IEEE sqrt in 29 of the 128 BatchNorm stds (torch 2.14.0+cpu, X3), and
// LoadBN takes the IEEE one. Everything else in the bridge is exact.
const bnUlps = 4

func ulpGate(t *testing.T, name string, got, want []float32) {
	t.Helper()
	ulpGateScaled(t, name, got, want, want)
}

// ulpGateScaled measures each difference in ulps of scale[i].
func ulpGateScaled(t *testing.T, name string, got, want, scale []float32) {
	t.Helper()
	worst, diff := 0.0, 0
	for i, w := range want {
		if got[i] == w {
			continue
		}
		diff++
		a := float32(math.Abs(float64(scale[i])))
		ulp := float64(math.Nextafter32(a, float32(math.Inf(1))) - a)
		worst = math.Max(worst, math.Abs(float64(got[i]-w))/ulp)
	}
	if worst > bnUlps {
		t.Errorf("%s: %d of %d values differ, worst by %.1f ulps > %d", name, diff, len(want), worst, bnUlps)
		return
	}
	t.Logf("%s: %d of %d values differ, worst by %.1f ulps", name, diff, len(want), worst)
}

// TestFromVAE normalises and patchifies the VAE oracle's encoded posterior
// mode (768x512) and asks for exactly its tokens.
func TestFromVAE(t *testing.T) {
	b, err := LoadBN(vaeDir)
	if err != nil {
		t.Skipf("no vae at %s (%v)", vaeDir, err)
	}
	buf, err := os.ReadFile(filepath.Join(vaeDumpDir, "manifest.json"))
	if err != nil {
		t.Skipf("no vae dump (%v)", err)
	}
	var man struct {
		Tensors map[string]struct{ Count int } `json:"tensors"`
	}
	if err := json.Unmarshal(buf, &man); err != nil {
		t.Fatal(err)
	}
	read := func(name string) []float32 {
		raw, err := os.ReadFile(filepath.Join(vaeDumpDir, name+".bin"))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	}
	got, err := b.FromVAE(read("enc_mode"), 32, 48)
	if err != nil {
		t.Fatal(err)
	}
	ulpGate(t, "enc_tokens", got.Data, read("enc_tokens"))
}
