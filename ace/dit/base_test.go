package dit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/zimage/qwen"
)

// The XL base checkpoint's gates (research/music-vertical.md A15c), against
// upstream's generate_audio at Gradio's base settings -- 32 steps, shift 3,
// CFG 7 through APG -- in fp32 on the CPU, priced against its bf16 run:
//
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_dit.py --model base full_metas
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_dit.py --model base --bf16 full_metas
const (
	baseDir     = "../../models/acestep-v15-xl-base"
	baseRef     = "../../reference/out/acedit_base"
	baseBF16Ref = "../../reference/out/acedit_base_bf16"
)

var sharedBase struct {
	once sync.Once
	g    *GPU
	err  error
	skip string
}

// baseGPU stages the base checkpoint once. It is a second ~10 GB staging
// beside sharedGPU's, so run the base tests on their own (-run Base).
func baseGPU(t *testing.T) *GPU {
	t.Helper()
	if testing.Short() {
		t.Skip("stages the whole model")
	}
	sharedBase.once.Do(func() {
		dev, skip := testDevice()
		if skip != "" {
			sharedBase.skip = skip
			return
		}
		if _, err := os.Stat(filepath.Join(baseDir, "config.json")); err != nil {
			sharedBase.skip = err.Error()
			return
		}
		sharedBase.g, sharedBase.err = NewGPU(dev, baseDir, 7500)
	})
	if sharedBase.skip != "" {
		t.Skip(sharedBase.skip)
	}
	if sharedBase.err != nil {
		t.Fatal(sharedBase.err)
	}
	return sharedBase.g
}

func loadBaseManifest(t *testing.T) *ditManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(baseRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_dit.py --model base", baseRef, err)
	}
	var m ditManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// TestBaseDiT is A15c's gate on full_metas (30 s, 375 tokens):
//
//   - the condition encoder from the plan's inputs (the base checkpoint
//     carries its own encoders);
//   - every forward's conditional and null rows, teacher-forced from the
//     oracle's x_t;
//   - the whole guided sample from the oracle's noise, through
//     plan.SampleGuided.
//
// The first forward's rows and the latents are held under half of
// upstream's own bf16 drift.
func TestBaseDiT(t *testing.T) {
	g := baseGPU(t)
	m := loadBaseManifest(t)
	if !g.Guided() {
		t.Fatal("the base checkpoint reads as turbo")
	}
	const label = "full_metas"
	c, ok := m.Cases[label]
	if !ok {
		t.Skipf("no %s case in %s", label, baseRef)
	}
	p := label + "_"
	enc, err := g.Encode(readMat(t, planRef, p+"text_hidden"), readMat(t, planRef, p+"lyric_embeds"), silence(t, 750))
	if err != nil {
		t.Fatal(err)
	}
	gate(t, "encoder states", enc, readMat(t, baseRef, p+"encoder_states"), 2e-2, 5e-3)
	enc = readMat(t, baseRef, p+"encoder_states")
	if err := g.Begin(enc); err != nil {
		t.Fatal(err)
	}
	if err := g.BeginNull(enc.Rows); err != nil {
		t.Fatal(err)
	}
	ctx := readMat(t, baseRef, p+"context_latents")
	for f, tf := range c.Timesteps {
		x := readMat(t, baseRef, fmt.Sprintf("%sx_in%d", p, f))
		for _, row := range []struct {
			name string
			step func(x, ctx *qwen.Mat, t float32) (*qwen.Mat, float64, error)
		}{
			{"v", func(x, ctx *qwen.Mat, t float32) (*qwen.Mat, float64, error) {
				v, d, err := g.Step(x, ctx, t)
				return v, d.Seconds(), err
			}},
			{"vnull", func(x, ctx *qwen.Mat, t float32) (*qwen.Mat, float64, error) {
				v, d, err := g.StepNull(x, ctx, t)
				return v, d.Seconds(), err
			}},
		} {
			v, took, err := row.step(x, ctx, tf)
			if err != nil {
				t.Fatal(err)
			}
			ref := fmt.Sprintf("%s%s%d", p, row.name, f)
			name := fmt.Sprintf("%s%d (%.0f ms)", row.name, f, took*1e3)
			rms := gate(t, name, v, readMat(t, baseRef, ref), 0.3, 0.1)
			if f == 0 {
				underBF16In(t, baseRef, baseBF16Ref, name, ref, rms)
			}
		}
	}
	noise := readMat(t, baseRef, p+"noise")
	T := noise.Rows
	out, err := plan.SampleGuided(noise.Data, plan.BaseSchedule(plan.BaseSteps, 3), plan.DefaultGuidance,
		func(x []float32, tt float32, uncond bool) ([]float32, []float32, error) {
			xm := &qwen.Mat{Rows: T, Cols: 64, Data: x}
			v, _, err := g.Step(xm, ctx, tt)
			if err != nil || !uncond {
				return v.Data, nil, err
			}
			n, _, err := g.StepNull(xm, ctx, tt)
			if err != nil {
				return nil, nil, err
			}
			return v.Data, n.Data, nil
		}, nil)
	if err != nil {
		t.Fatal(err)
	}
	rms := gate(t, "latents", &qwen.Mat{Rows: T, Cols: 64, Data: out}, readMat(t, baseRef, p+"latents"), 0.5, 0.15)
	underBF16In(t, baseRef, baseBF16Ref, "latents", p+"latents", rms)
}

// TestBaseNullCross is S1's gate: the null row with its cross-attention as
// a constant a layer against the control (the null planes run through the
// block), both against the oracle, at the first, a middle and the last
// forward; and the null forward's time either way (median of 7).
func TestBaseNullCross(t *testing.T) {
	g := baseGPU(t)
	m := loadBaseManifest(t)
	const label = "full_metas"
	c, ok := m.Cases[label]
	if !ok {
		t.Skipf("no %s case in %s", label, baseRef)
	}
	p := label + "_"
	enc := readMat(t, baseRef, p+"encoder_states")
	if err := g.Begin(enc); err != nil {
		t.Fatal(err)
	}
	if err := g.BeginNull(enc.Rows); err != nil {
		t.Fatal(err)
	}
	defer func() { g.nullConst = true }()
	ctx := readMat(t, baseRef, p+"context_latents")
	for _, f := range []int{0, len(c.Timesteps) / 2, len(c.Timesteps) - 1} {
		x := readMat(t, baseRef, fmt.Sprintf("%sx_in%d", p, f))
		want := readMat(t, baseRef, fmt.Sprintf("%svnull%d", p, f))
		var got [2]*qwen.Mat
		var ms [2]float64
		for i, on := range []bool{false, true} {
			g.nullConst = on
			var times []float64
			for range 7 {
				v, took, err := g.StepNull(x, ctx, c.Timesteps[f])
				if err != nil {
					t.Fatal(err)
				}
				got[i] = v
				times = append(times, took.Seconds()*1e3)
			}
			sort.Float64s(times)
			ms[i] = times[len(times)/2]
		}
		ctl := gate(t, fmt.Sprintf("vnull%d control (%.1f ms)", f, ms[0]), got[0], want, 0.3, 0.1)
		s1 := gate(t, fmt.Sprintf("vnull%d S1 (%.1f ms)", f, ms[1]), got[1], want, 0.3, 0.1)
		rel, rms := relGap(got[1], got[0])
		t.Logf("vnull%d S1 against the control: max %.2e rms %.2e; %.3fx", f, rel, rms, ms[0]/ms[1])
		if s1 > 1.5*ctl+1e-4 {
			t.Errorf("vnull%d: S1 rms %.2e against the control's %.2e", f, s1, ctl)
		}
	}
}

// TestBaseTiming is A15's measure of a forward by length (median of 5, the
// conditional row and the null row under S1), and the split by kind at the
// lengths in ACE_BASE_PROFILE (default 30 and 60 s): what S2 (the CFG pair
// as one forward of twice the rows) could win is a forward at 2n tokens
// against two at n.
func TestBaseTiming(t *testing.T) {
	g := baseGPU(t)
	loadBaseManifest(t)
	enc := readMat(t, baseRef, "full_metas_encoder_states")
	if err := g.Begin(enc); err != nil {
		t.Fatal(err)
	}
	if err := g.BeginNull(enc.Rows); err != nil {
		t.Fatal(err)
	}
	profile := map[int]bool{30: true, 60: true}
	if s := os.Getenv("ACE_BASE_PROFILE"); s != "" {
		profile = map[int]bool{}
		for _, f := range strings.Split(s, ",") {
			n, _ := strconv.Atoi(f)
			profile[n] = true
		}
	}
	for _, sec := range []int{15, 30, 60, 120, 240} {
		T := sec * 25
		x, ctx := qwen.NewMat(T, 64), qwen.NewMat(T, 128)
		for i := range x.Data {
			x.Data[i] = float32(math.Sin(float64(i)))
		}
		var med [2]float64
		for i, step := range []func(x, ctx *qwen.Mat, t float32) (*qwen.Mat, time.Duration, error){g.Step, g.StepNull} {
			var ms []float64
			for range 5 {
				_, took, err := step(x, ctx, 0.75)
				if err != nil {
					t.Fatal(err)
				}
				ms = append(ms, took.Seconds()*1e3)
			}
			sort.Float64s(ms)
			med[i] = ms[2]
		}
		t.Logf("%3d s (%4d tokens): cond %6.1f ms, null %6.1f ms, a CFG step %6.1f ms (%.3f ms a token-row)",
			sec, g.Tokens(T), med[0], med[1], med[0]+med[1], (med[0]+med[1])/float64(2*g.Tokens(T)))
		if !profile[sec] {
			continue
		}
		st, err := g.Profile(x, ctx, 0.75)
		if err != nil {
			t.Fatal(err)
		}
		by := map[string]time.Duration{}
		var total time.Duration
		for _, s := range st {
			by[s.Kind] += s.GPU
			total += s.GPU
		}
		kinds := make([]string, 0, len(by))
		for k := range by {
			kinds = append(kinds, k)
		}
		sort.Slice(kinds, func(i, j int) bool { return by[kinds[i]] > by[kinds[j]] })
		for _, k := range kinds {
			t.Logf("  %-18s %7.1f ms %5.1f%%", k, by[k].Seconds()*1e3, 100*by[k].Seconds()/total.Seconds())
		}
	}
}
