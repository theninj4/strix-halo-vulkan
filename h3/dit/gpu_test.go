package dit

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference is reference/dump_h3_dit.py: the whole transformer in fp32,
// AdaLN folded into per-forward tables, over a 5,095-row t2va (256x448 x 124
// frames, M2's 537-token README conditioning), N = 8. Regenerate with:
//
//	.venv/bin/python reference/dump_h3_dit.py   (~85 GB RSS, ~20 min)
const (
	ditRef     = "../../reference/out/h3dit"
	textencRef = "../../reference/out/h3textenc"
)

type ditManifest struct {
	TextTokens   int                  `json:"text_tokens"`
	LatentFrames int                  `json:"latent_frames"`
	LatentHeight int                  `json:"latent_height"`
	LatentWidth  int                  `json:"latent_width"`
	AudioLatents int                  `json:"audio_latents"`
	KeepBlocks   []int                `json:"keep_blocks"`
	BlockStats   []map[string]float64 `json:"block_stats"`
	Forwards     []struct {
		Timestep      float32   `json:"timestep"`
		AudioTimestep float32   `json:"audio_timestep"`
		Unique        []float32 `json:"unique"`
	} `json:"forwards"`
	Tensors map[string]struct {
		Shape []int `json:"shape"`
		Count int   `json:"count"`
	} `json:"tensors"`
}

func readRef(t *testing.T, dir string, tensors map[string]struct {
	Shape []int `json:"shape"`
	Count int   `json:"count"`
}, name string) *qwen.Mat {
	t.Helper()
	meta, ok := tensors[name]
	if !ok {
		t.Fatalf("%s has no tensor %q", dir, name)
	}
	raw, err := os.ReadFile(filepath.Join(dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	cols := meta.Shape[len(meta.Shape)-1]
	m := &qwen.Mat{Rows: meta.Count / cols, Cols: cols, Data: make([]float32, meta.Count)}
	for i := range m.Data {
		m.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return m
}

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("h3-dit-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// relGap is max abs error over the reference's absmax, and the rms error
// over the reference's rms.
func relGap(got, want *qwen.Mat) (rel, rmsRel float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want.Data[i])))
		sq += d * d
		refSq += float64(want.Data[i]) * float64(want.Data[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq)
}

// TestGPUForward is M7's gate on the device: the refiner, blocks teacher-
// forced one at a time at five depths of the stack, and forward 0 end to
// end, against the fp32 oracle — at chunk 2048, so the 5,095 rows run as
// three chunks and every chunk boundary is exercised. ~42 GB staged.
func TestGPUForward(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 42 GB")
	}
	buf, err := os.ReadFile(filepath.Join(ditRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_dit.py", ditRef, err)
	}
	var m ditManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	tbuf, err := os.ReadFile(filepath.Join(textencRef, "manifest.json"))
	if err != nil {
		t.Skip(err)
	}
	var tm ditManifest
	if err := json.Unmarshal(tbuf, &tm); err != nil {
		t.Fatal(err)
	}
	cond := readRef(t, textencRef, tm.Tensors, "readme_fp32")

	lay, err := plan.NewLayout(filled(m.TextTokens, plan.TextTag), m.LatentFrames, m.LatentHeight, m.LatentWidth, m.AudioLatents, nil)
	if err != nil {
		t.Fatal(err)
	}
	f0 := m.Forwards[0]
	uniq, rowT := lay.RowTimesteps(f0.Timestep, f0.AudioTimestep)
	t.Logf("layout %d rows; forward 0 timesteps %v", len(lay.Pos), uniq)

	start := time.Now()
	tabs, err := Tables(modelDir, uniq)
	if err != nil {
		t.Skipf("no transformer weights (%v)", err)
	}
	t.Logf("AdaLN tables for %d timestep(s) in %v", len(uniq), time.Since(start).Round(time.Millisecond))
	for _, b := range []int{0, 49} {
		want := readRef(t, ditRef, m.Tensors, fmt.Sprintf("adaln%d_f0", b))
		got := &qwen.Mat{Rows: want.Rows, Cols: want.Cols, Data: make([]float32, len(want.Data))}
		for r := 0; r < tabs[b].Rows; r++ {
			for j := 0; j < 6; j++ {
				copy(got.Row(j*tabs[b].Rows+r), tabs[b].Vec(r, j))
			}
		}
		rel, _ := relGap(got, want)
		t.Logf("adaln table %2d: rel %.2g", b, rel)
		if rel > 1e-5 {
			t.Errorf("block %d AdaLN table rel %.3g", b, rel)
		}
	}

	dev, done := newTestDevice(t)
	defer done()
	start = time.Now()
	g, err := NewGPUBank(dev, modelDir, 5120, 1024, 2048, testBank(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged in %v: weights %.1f GB, activations %.2f GB", time.Since(start).Round(time.Millisecond),
		float64(g.WeightBytes())/1e9, float64(g.ActivationBytes())/1e9)

	if err := g.Begin(lay, cond, uniq, tabs); err != nil {
		t.Fatal(err)
	}
	rel, rms := relGap(g.text, readRef(t, ditRef, m.Tensors, "text_refined"))
	t.Logf("refiner          rel %.3g  rms %.3g", rel, rms)
	if !(rel <= 1e-2) {
		t.Errorf("refined text rel %.3g", rel)
	}

	for _, b := range m.KeepBlocks {
		in := readRef(t, ditRef, m.Tensors, fmt.Sprintf("f0_block%d_in", b))
		want := readRef(t, ditRef, m.Tensors, fmt.Sprintf("f0_block%d_out", b))
		start := time.Now()
		got, err := g.Blocks(in, rowT, b, b+1)
		if err != nil {
			t.Fatal(err)
		}
		rel, rms := relGap(got, want)
		t.Logf("block %2d         rel %.3g  rms %.3g  (%v)", b, rel, rms, time.Since(start).Round(time.Millisecond))
		if rel > 1e-2 || math.IsNaN(rel) {
			t.Errorf("block %d rel %.3g", b, rel)
		}
	}

	v, a, took, err := g.Step(readRef(t, ditRef, m.Tensors, "noise_video"), readRef(t, ditRef, m.Tensors, "noise_audio"), rowT)
	if err != nil {
		t.Fatal(err)
	}
	relV, rmsV := relGap(v, readRef(t, ditRef, m.Tensors, "f0_v_video"))
	relA, rmsA := relGap(a, readRef(t, ditRef, m.Tensors, "f0_v_audio"))
	t.Logf("forward 0 in %v: video rel %.3g rms %.3g; audio rel %.3g rms %.3g", took.Round(time.Millisecond), relV, rmsV, relA, rmsA)
	if relV > 5e-2 || relA > 5e-2 || math.IsNaN(relV+relA) {
		t.Errorf("forward 0: video rel %.3g, audio rel %.3g", relV, relA)
	}

	// The down projection's K split (M11d) is the one-pass GEMM split in
	// time, so the same forward without it is the same bits.
	split := g.downSplit
	g.downSplit = 1
	v1, a1, _, err := g.Step(readRef(t, ditRef, m.Tensors, "noise_video"), readRef(t, ditRef, m.Tensors, "noise_audio"), rowT)
	g.downSplit = split
	if err != nil {
		t.Fatal(err)
	}
	if split < 2 {
		t.Errorf("the down projection is not split (%d): nothing compared", split)
	}
	for _, p := range [][2]*qwen.Mat{{v, v1}, {a, a1}} {
		for i := range p[0].Data {
			if math.Float32bits(p[0].Data[i]) != math.Float32bits(p[1].Data[i]) {
				t.Fatalf("down in %d K pieces differs from one pass at %d: %g against %g", split, i, p[0].Data[i], p[1].Data[i])
			}
		}
	}
	t.Logf("down projection in %d K pieces: bit-identical to one pass", split)

	// A second request after a full forward: the planes now hold 5,095 rows
	// of keys, and the refiner's 537-key attention reads a key block past
	// its own count. Before zeroKeyTail that turned every text row into NaN.
	if err := g.Begin(lay, cond, uniq, tabs); err != nil {
		t.Fatal(err)
	}
	if again, _ := relGap(g.text, readRef(t, ditRef, m.Tensors, "text_refined")); !(again <= 1e-2) {
		t.Errorf("refiner after a forward: rel %v", again)
	}
	// The negative control: without the zeroing, the same sequence fails --
	// on the plain kernel, whose row max reads the stale keys. The transposed
	// one (M11c) masks the key tail out of its max as well as out of P, so it
	// survives stale keys on its own; it keeps the zeroing, since P = 0 still
	// multiplies whatever the stale v rows hold.
	g.noKeyTail = true
	defer func() { g.noKeyTail, g.attnFixed = false, false }()
	sequence := func() float64 {
		if err := g.Begin(lay, cond, uniq, tabs); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := g.Step(readRef(t, ditRef, m.Tensors, "noise_video"), readRef(t, ditRef, m.Tensors, "noise_audio"), rowT); err != nil {
			t.Fatal(err)
		}
		if err := g.Begin(lay, cond, uniq, tabs); err != nil {
			t.Fatal(err)
		}
		again, _ := relGap(g.text, readRef(t, ditRef, m.Tensors, "text_refined"))
		return again
	}
	if g.attnT {
		t.Logf("transposed attention without zeroKeyTail: refiner after a forward rel %v", sequence())
	}
	if err := g.SetAttention(AttnVariant{QT: 1, KTIL: 4}); err != nil {
		t.Fatal(err)
	}
	if again := sequence(); again <= 1e-2 {
		t.Errorf("without zeroKeyTail the refiner still matches (rel %v): the check above proves nothing", again)
	} else {
		t.Logf("negative control: without zeroKeyTail the plain kernel's refiner after a forward is rel %v", again)
	}
}

// TestGPURun is the sampler on the device: all seven forwards of the
// oracle's N = 8 run, stepped by h3/plan's schedules, against the oracle's
// velocities and latents at every step. The AdaLN tables are built once over
// every timestep the request uses, as the served path will.
//
// It is **teacher-forced**: each forward starts from the oracle's latents.
// Free-running (H3_FREE=1) is logged, not gated, because TestGPUSensitivity
// measured the sampler's own dynamics growing a 1e-4 change in the starting
// noise to latents rms 0.096 by step 6 on the device alone — the same size
// as fp16-vs-fp32's free-running 0.121. A free-running comparison measures
// the model's sensitivity, not this port's precision.
func TestGPURun(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 42 GB")
	}
	r := newRunRef(t)
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUBank(dev, modelDir, 5120, 1024, 2048, testBank(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if os.Getenv("H3_FREE") != "" {
		r.run(t, g, true)
		return
	}
	steps := r.run(t, g, false)
	if g.bank == qwen.BankFP16 {
		for f, e := range steps {
			if e[0] > stepRMS || e[1] > stepRMS {
				t.Errorf("step %d: video latents rms %.3g, audio rms %.3g > %.0e", f, e[0], e[1], stepRMS)
			}
		}
		return
	}
	// The int8 bank is held to the released pipeline: each step within
	// q8OverBF16 of bf16's same teacher-forced step from the fp32 oracle.
	bf := bf16Steps(t)
	for f, e := range steps {
		t.Logf("step %d: video rms %.3g against bf16's %.3g, audio %.3g against %.3g", f, e[0], bf[f][0], e[1], bf[f][1])
		if e[0] > q8OverBF16*bf[f][0] || e[1] > q8OverBF16*bf[f][1] {
			t.Errorf("step %d: %s is past %.2gx bf16's distance from fp32 (video rms %.3g against %.3g, audio %.3g against %.3g)",
				f, g.bank, q8OverBF16, e[0], bf[f][0], e[1], bf[f][1])
		}
	}
}

// q8OverBF16 is how far past the bf16 pipeline's step error the int8 bank
// may land. Measured 2026-09-26: at or inside it on every step of both
// modalities (0.46x–0.99x; step 1's audio is the 0.99), so the headroom is
// for kernel changes that move the last digit, not for the bank.
const q8OverBF16 = 1.25

// bf16Steps is reference/dump_h3_dit_bf16.py's teacher-forced run: each
// step's video and audio latents rms against the fp32 oracle, at the
// precision the released pipeline runs the transformer.
func bf16Steps(t *testing.T) [][2]float64 {
	buf, err := os.ReadFile(filepath.Join(ditRef+"_bf16", "manifest.json"))
	if err != nil {
		t.Fatalf("the int8 gate is priced on the bf16 run (%v); run reference/dump_h3_dit_bf16.py", err)
	}
	var m struct {
		Forwards []struct {
			Latents, Audio [2]float64
		} `json:"forwards"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	out := make([][2]float64, len(m.Forwards))
	for f, fw := range m.Forwards {
		out[f] = [2]float64{fw.Latents[1], fw.Audio[1]}
	}
	return out
}

// runRef is the oracle's N = 8 run, set up for a sampler on the device: the
// layout, both schedules, the AdaLN tables built once over every timestep
// the request uses (as the served path does) and each forward's rows
// indexed into them.
type runRef struct {
	m      ditManifest
	lay    *plan.Layout
	vs, as *plan.Schedule
	tvals  []float32
	rowTs  [][]int32
	tabs   []*Table
	cond   *qwen.Mat
}

func newRunRef(t *testing.T) *runRef {
	buf, err := os.ReadFile(filepath.Join(ditRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v)", err)
	}
	r := &runRef{}
	if err := json.Unmarshal(buf, &r.m); err != nil {
		t.Fatal(err)
	}
	tbuf, err := os.ReadFile(filepath.Join(textencRef, "manifest.json"))
	if err != nil {
		t.Skip(err)
	}
	var tm ditManifest
	if err := json.Unmarshal(tbuf, &tm); err != nil {
		t.Fatal(err)
	}
	m := r.m
	if r.lay, err = plan.NewLayout(filled(m.TextTokens, plan.TextTag), m.LatentFrames, m.LatentHeight, m.LatentWidth, m.AudioLatents, nil); err != nil {
		t.Fatal(err)
	}
	if r.vs, err = plan.NewSchedule(len(m.Forwards)+1, 12); err != nil {
		t.Fatal(err)
	}
	if r.as, err = plan.NewSchedule(len(m.Forwards)+1, 3); err != nil {
		t.Fatal(err)
	}
	seen := map[float32]bool{}
	r.rowTs = make([][]int32, len(r.vs.Timesteps))
	for f := range r.vs.Timesteps {
		u, idx := r.lay.RowTimesteps(r.vs.Timesteps[f], r.as.Timesteps[f])
		for _, v := range u {
			if !seen[v] {
				seen[v] = true
				r.tvals = append(r.tvals, v)
			}
		}
		r.rowTs[f] = make([]int32, len(idx))
		for i, j := range idx {
			r.rowTs[f][i] = int32(indexOf(r.tvals, u[j]))
		}
	}
	start := time.Now()
	if r.tabs, err = Tables(modelDir, r.tvals); err != nil {
		t.Skipf("no weights (%v)", err)
	}
	t.Logf("AdaLN tables for %d timesteps in %v", len(r.tvals), time.Since(start).Round(time.Millisecond))
	r.cond = readRef(t, textencRef, tm.Tensors, "readme_fp32")
	return r
}

// run steps all seven forwards on g, teacher-forced unless free, logging
// every forward's velocity and step against the oracle, and returns the
// video and audio latents rms of every step. TestGPURun says why the gate
// is teacher-forced.
func (r *runRef) run(t *testing.T, g *GPU, free bool) (steps [][2]float64) {
	m := r.m
	if err := g.Begin(r.lay, r.cond, r.tvals, r.tabs); err != nil {
		t.Fatal(err)
	}
	video := readRef(t, ditRef, m.Tensors, "noise_video")
	audio := readRef(t, ditRef, m.Tensors, "noise_audio")
	var total time.Duration
	for f := range r.vs.Timesteps {
		if !free && f > 0 {
			video = readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_latents", f-1))
			audio = readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_audio", f-1))
		}
		v, a, took, err := g.Step(video, audio, r.rowTs[f])
		if err != nil {
			t.Fatal(err)
		}
		total += took
		rv, rvs := relGap(v, readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_v_video", f)))
		ra, ras := relGap(a, readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_v_audio", f)))
		t.Logf("forward %d velocity: video rel %.3g rms %.3g, audio rel %.3g rms %.3g", f, rv, rvs, ra, ras)
		r.vs.Step(f, v.Data, video.Data)
		r.as.Step(f, a.Data, audio.Data)
		relV, rmsV := relGap(video, readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_latents", f)))
		relA, rmsA := relGap(audio, readRef(t, ditRef, m.Tensors, fmt.Sprintf("f%d_audio", f)))
		t.Logf("step %d (t %.4f / %.4f) %v: video latents rel %.3g rms %.3g; audio rel %.3g rms %.3g",
			f, r.vs.Timesteps[f], r.as.Timesteps[f], took.Round(time.Millisecond), relV, rmsV, relA, rmsA)
		if math.IsNaN(relV + relA) {
			t.Fatalf("step %d is NaN", f)
		}
		steps = append(steps, [2]float64{rmsV, rmsA})
	}
	t.Logf("%d forwards in %v", len(r.vs.Timesteps), total.Round(time.Millisecond))
	return steps
}

// TestGPUQ8Ablation says where the int8 bank's error comes from (VIDEO.md
// M11a): the teacher-forced run with the named projections held at their
// int8 values and the rest fp16, one staging an arm. H3_Q8_ARMS is a
// semicolon-separated list of comma-separated projection sets ("" is the
// fp16 control, "all" every projection); the default is the control, all,
// and all but each one in turn.
func TestGPUQ8Ablation(t *testing.T) {
	if os.Getenv("H3_Q8_ABLATION") == "" {
		t.Skip("H3_Q8_ABLATION=1: one 40 GB staging and seven forwards an arm")
	}
	arms := []string{"", "all"}
	for _, p := range projOrder {
		arms = append(arms, "-"+projNames[p])
	}
	if s, ok := os.LookupEnv("H3_Q8_ARMS"); ok {
		arms = strings.Split(s, ";")
	}
	r := newRunRef(t)
	dev, done := newTestDevice(t)
	defer done()
	defer func() { simQ8 = nil }()
	for _, arm := range arms {
		simQ8 = map[proj]bool{}
		switch {
		case arm == "all":
			for _, p := range projOrder {
				simQ8[p] = true
			}
		case strings.HasPrefix(arm, "-"):
			for _, p := range projOrder {
				simQ8[p] = projNames[p] != arm[1:]
			}
		case arm != "":
			for _, name := range strings.Split(arm, ",") {
				for _, p := range projOrder {
					if projNames[p] == name {
						simQ8[p] = true
					}
				}
			}
		}
		g, err := NewGPU(dev, modelDir, 5120, 1024, 2048)
		if err != nil {
			t.Fatal(err)
		}
		steps := r.run(t, g, false)
		g.Destroy()
		var line string
		for _, e := range steps {
			line += fmt.Sprintf(" %.2e/%.2e", e[0], e[1])
		}
		t.Logf("ARM %-8q steps video/audio rms:%s", arm, line)
	}
}

func finite(xs []float32) bool {
	for _, x := range xs {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// stepRMS bounds one teacher-forced step's latents, rms error over the
// oracle's rms. Measured 2026-09-26: 7.9e-5 to 1.7e-3 over the seven steps.
const stepRMS = 3e-3

func indexOf(xs []float32, v float32) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return -1
}

// TestGPUSensitivity is the control TestGPURun's free-running numbers are
// read against: the device path run twice from the oracle's noise, once as
// is and once moved by eps of its own scale, compared with each other. It
// measures how fast the N = 8 sampler's own dynamics grow a difference,
// with no oracle and no fp16-vs-fp32 in it.
func TestGPUSensitivity(t *testing.T) {
	if testing.Short() || os.Getenv("H3_SENSITIVITY") == "" {
		t.Skip("a control, not a gate: set H3_SENSITIVITY=1 (stages 42 GB, ~2 min)")
	}
	buf, err := os.ReadFile(filepath.Join(ditRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump (%v)", err)
	}
	var m ditManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	tbuf, _ := os.ReadFile(filepath.Join(textencRef, "manifest.json"))
	var tm ditManifest
	if err := json.Unmarshal(tbuf, &tm); err != nil {
		t.Fatal(err)
	}
	lay, _ := plan.NewLayout(filled(m.TextTokens, plan.TextTag), m.LatentFrames, m.LatentHeight, m.LatentWidth, m.AudioLatents, nil)
	vs, _ := plan.NewSchedule(len(m.Forwards)+1, 12)
	as, _ := plan.NewSchedule(len(m.Forwards)+1, 3)
	var tvals []float32
	seen := map[float32]bool{}
	rowTs := make([][]int32, len(vs.Timesteps))
	for f := range vs.Timesteps {
		u, idx := lay.RowTimesteps(vs.Timesteps[f], as.Timesteps[f])
		for _, v := range u {
			if !seen[v] {
				seen[v] = true
				tvals = append(tvals, v)
			}
		}
		rowTs[f] = make([]int32, len(idx))
		for i, j := range idx {
			rowTs[f][i] = int32(indexOf(tvals, u[j]))
		}
	}
	tabs, err := Tables(modelDir, tvals)
	if err != nil {
		t.Skip(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUBank(dev, modelDir, 5120, 1024, 2048, testBank(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.Begin(lay, readRef(t, textencRef, tm.Tensors, "readme_fp32"), tvals, tabs); err != nil {
		t.Fatal(err)
	}
	run := func(eps, phase float64) []*qwen.Mat {
		video := readRef(t, ditRef, m.Tensors, "noise_video")
		audio := readRef(t, ditRef, m.Tensors, "noise_audio")
		for i := range video.Data {
			video.Data[i] *= 1 + float32(eps*math.Sin(float64(i)+phase))
		}
		var out []*qwen.Mat
		for f := range vs.Timesteps {
			v, a, _, err := g.Step(video, audio, rowTs[f])
			if err != nil {
				t.Fatal(err)
			}
			vs.Step(f, v.Data, video.Data)
			as.Step(f, a.Data, audio.Data)
			out = append(out, video.Clone())
		}
		return out
	}
	base := run(0, 0)
	for _, eps := range []float64{1e-4, 1e-3} {
		moved := run(eps, 0)
		for f := range moved {
			rel, rms := relGap(moved[f], base[f])
			t.Logf("eps %.0e step %d: device-vs-device video latents rel %.3g rms %.3g", eps, f, rel, rms)
		}
	}
	// H3_SENSITIVITY_DRAWS=n: the spread of the final step's distance over
	// n draws of the 1e-4 move (a phase each), which is what one
	// free-running comparison is a single sample of.
	draws, _ := strconv.Atoi(os.Getenv("H3_SENSITIVITY_DRAWS"))
	var finals []float64
	for d := 1; d <= draws; d++ {
		moved := run(1e-4, float64(d))
		_, rms := relGap(moved[len(moved)-1], base[len(base)-1])
		finals = append(finals, rms)
		t.Logf("draw %d: final video latents rms %.3g", d, rms)
	}
	if len(finals) > 0 {
		sort.Float64s(finals)
		t.Logf("%d draws of a 1e-4 move: final video latents rms min %.3g, median %.3g, max %.3g",
			len(finals), finals[0], finals[len(finals)/2], finals[len(finals)-1])
	}
}

// TestGPUShapes times one forward at the shapes a request actually takes —
// the served 864x480 and the trained 1344x768, 124 frames each, on the
// README conditioning — and checks the arena plan fits them. Set
// H3_SHAPES=1; it stages 42 GB plus arenas sized for the largest shape.
// The velocities are not compared (there is no oracle at these sizes);
// they are checked finite.
func TestGPUShapes(t *testing.T) {
	if testing.Short() || os.Getenv("H3_SHAPES") == "" {
		t.Skip("a timing, not a gate: set H3_SHAPES=1")
	}
	tbuf, err := os.ReadFile(filepath.Join(textencRef, "manifest.json"))
	if err != nil {
		t.Skip(err)
	}
	var tm ditManifest
	if err := json.Unmarshal(tbuf, &tm); err != nil {
		t.Fatal(err)
	}
	cond := readRef(t, textencRef, tm.Tensors, "readme_fp32")
	type shape struct{ h, w, frames int }
	shapes := []shape{{480, 864, 124}, {768, 1344, 124}}
	if os.Getenv("H3_SHAPES") == "480" {
		shapes = shapes[:1]
	}
	lays := make([]*plan.Layout, len(shapes))
	maxRows := 0
	for i, s := range shapes {
		lay, err := plan.NewLayout(filled(cond.Rows, plan.TextTag), plan.LatentFrames(s.frames), s.h/plan.SpatialCompression,
			s.w/plan.SpatialCompression, plan.AudioLatents(s.frames), nil)
		if err != nil {
			t.Fatal(err)
		}
		lays[i], maxRows = lay, max(maxRows, len(lay.Pos))
	}
	vs, _ := plan.NewSchedule(20, 12)
	as, _ := plan.NewSchedule(20, 3)
	tvals, _ := lays[0].RowTimesteps(vs.Timesteps[5], as.Timesteps[5])
	tabs, err := Tables(modelDir, tvals)
	if err != nil {
		t.Skip(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	chunk := 8192
	if c := os.Getenv("H3_CHUNK"); c != "" {
		fmt.Sscan(c, &chunk)
	}
	g, err := NewGPUBank(dev, modelDir, maxRows, 1024, chunk, testBank(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("staged for %d rows, chunk %d: activations %.2f GB", maxRows, chunk, float64(g.ActivationBytes())/1e9)
	// H3_ATTN_PLAIN=1 is the control arm: the plain attention builds, as
	// before M11c's transposed one.
	if os.Getenv("H3_ATTN_PLAIN") != "" {
		g.attnProbed = true
	}
	// H3_DOWN_SPLIT=n runs the down projection as n K pieces (M11d).
	if v := os.Getenv("H3_DOWN_SPLIT"); v != "" {
		fmt.Sscan(v, &g.downSplit)
	}
	for i, lay := range lays {
		if err := g.Begin(lay, cond, tvals, tabs); err != nil {
			t.Fatal(err)
		}
		_, rowT := lay.RowTimesteps(vs.Timesteps[5], as.Timesteps[5])
		rng := rand.New(rand.NewPCG(1, uint64(i)))
		video := qwen.NewMat(len(lay.Video), g.cfg.Patch())
		audio := qwen.NewMat(len(lay.Audio), g.cfg.AudioChannels)
		for j := range video.Data {
			video.Data[j] = float32(rng.NormFloat64())
		}
		for j := range audio.Data {
			audio.Data[j] = float32(rng.NormFloat64())
		}
		v, _, took, err := g.Step(video, audio, rowT)
		if err != nil {
			t.Fatal(err)
		}
		if !finite(v.Data) {
			// Bisect: the packed input, then block by block.
			gr, err := g.stepGraph(video, audio, rowT, 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := gr.submit(); err != nil {
				t.Fatal(err)
			}
			x := qwen.NewMat(len(lay.Pos), g.H)
			copy(x.Data, g.abuf.ReadFloat32At(int(g.aX), len(x.Data)))
			nbad, first, last := 0, -1, -1
			for r := 0; r < x.Rows; r++ {
				if !finite(x.Row(r)) {
					nbad++
					if first < 0 {
						first = r
					}
					last = r
				}
			}
			t.Logf("packed input: %d non-finite rows, %d..%d (text %d, audio %d..%d, video %d..)", nbad, first, last,
				len(lay.Text), lay.Audio[0], lay.Audio[len(lay.Audio)-1], lay.Video[0])
			for b := 0; b < len(g.blocks) && finite(x.Data); b++ {
				if x, err = g.Blocks(x, rowT, b, b+1); err != nil {
					t.Fatal(err)
				}
				bad, absmax := -1, 0.0
				for j, val := range x.Data {
					if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
						bad = j
						break
					}
					absmax = math.Max(absmax, math.Abs(float64(val)))
				}
				t.Logf("block %2d: absmax %.4g, first non-finite %d (row %d)", b, absmax, bad, bad/g.H)
			}
			t.Fatalf("%v: non-finite velocity", shapes[i])
		}
		if os.Getenv("H3_PROFILE") != "" {
			stages, err := g.Profile(video, audio, rowT)
			if err != nil {
				t.Fatal(err)
			}
			type agg struct {
				d     time.Duration
				fl    float64
				count int
			}
			byKind := map[string]*agg{}
			var order []string
			var total time.Duration
			for _, st := range stages {
				a, ok := byKind[st.Kind]
				if !ok {
					a = &agg{}
					byKind[st.Kind] = a
					order = append(order, st.Kind)
				}
				a.d += st.GPU
				a.fl += st.Flops
				a.count++
				total += st.GPU
			}
			sort.Slice(order, func(i, j int) bool { return byKind[order[i]].d > byKind[order[j]].d })
			for _, k := range order {
				a := byKind[k]
				rate := ""
				if a.fl > 0 {
					rate = fmt.Sprintf("%5.1f TFLOP/s", a.fl/a.d.Seconds()/1e12)
				}
				t.Logf("  %-18s %5d  %9v  %5.1f%%  %s", k, a.count, a.d.Round(time.Millisecond), 100*a.d.Seconds()/total.Seconds(), rate)
			}
		}
		L := float64(len(lay.Pos))
		fl := 2*19.27e9*L + 4*L*L*7168*50
		t.Logf("%dx%d x %d frames: %d rows, forward %v, %.1f TFLOP/s", shapes[i].w, shapes[i].h, shapes[i].frames,
			len(lay.Pos), took.Round(time.Millisecond), fl/took.Seconds()/1e12)
	}
}

// TestGPUAttentionScreen times every attention build over the same planes —
// block 49's, after one real forward at a given shape — and checks each
// against the default build's context. Set H3_SCREEN to the shape, e.g.
// 768x1344 or 480x864.
func TestGPUAttentionScreen(t *testing.T) {
	var h, w int
	if _, err := fmt.Sscanf(os.Getenv("H3_SCREEN"), "%dx%d", &h, &w); err != nil || testing.Short() {
		t.Skip("a screen, not a gate: set H3_SCREEN=<h>x<w>")
	}
	tbuf, err := os.ReadFile(filepath.Join(textencRef, "manifest.json"))
	if err != nil {
		t.Skip(err)
	}
	var tm ditManifest
	if err := json.Unmarshal(tbuf, &tm); err != nil {
		t.Fatal(err)
	}
	cond := readRef(t, textencRef, tm.Tensors, "readme_fp32")
	lay, err := plan.NewLayout(filled(cond.Rows, plan.TextTag), plan.LatentFrames(124), h/plan.SpatialCompression,
		w/plan.SpatialCompression, plan.AudioLatents(124), nil)
	if err != nil {
		t.Fatal(err)
	}
	vs, _ := plan.NewSchedule(20, 12)
	as, _ := plan.NewSchedule(20, 3)
	tvals, rowT := lay.RowTimesteps(vs.Timesteps[5], as.Timesteps[5])
	tabs, err := Tables(modelDir, tvals)
	if err != nil {
		t.Skip(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUBank(dev, modelDir, len(lay.Pos), 1024, 8192, testBank(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.Begin(lay, cond, tvals, tabs); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 1))
	video := qwen.NewMat(len(lay.Video), g.cfg.Patch())
	audio := qwen.NewMat(len(lay.Audio), g.cfg.AudioChannels)
	for j := range video.Data {
		video.Data[j] = float32(rng.NormFloat64())
	}
	for j := range audio.Data {
		audio.Data[j] = float32(rng.NormFloat64())
	}
	if _, _, _, err := g.Step(video, audio, rowT); err != nil {
		t.Fatal(err)
	}
	L := float64(len(lay.Pos))
	fl := 4 * L * L * float64(g.headDim*g.heads)
	// H3_ATTN_SPV=qt:ktil:path[,...] adds candidate builds to the screen.
	variants := AttnVariants()
	if env := os.Getenv("H3_ATTN_SPV"); env != "" {
		for _, arm := range strings.Split(env, ",") {
			var v AttnVariant
			f := strings.SplitN(arm, ":", 3)
			if len(f) != 3 {
				t.Fatalf("H3_ATTN_SPV arm %q: want qt:ktil:path", arm)
			}
			v.QT, _ = strconv.Atoi(f[0])
			v.KTIL, _ = strconv.Atoi(f[1])
			v.Tag = filepath.Base(f[2])
			spirv, err := os.ReadFile(f[2])
			if err != nil {
				t.Fatal(err)
			}
			if err := g.AddAttention(v, spirv); err != nil {
				t.Fatal(err)
			}
			variants = append(variants, v)
		}
	}
	if os.Getenv("H3_ATTN_ONLY_SPV") != "" {
		variants = variants[len(AttnVariants()):]
		variants = append([]AttnVariant{{QT: 1, KTIL: 4}}, variants...)
	}
	var base []uint16
	for _, v := range variants {
		if err := g.SetAttention(v); err != nil {
			t.Fatal(err)
		}
		var took time.Duration
		var ctx []uint16
		for rep := 0; rep < 2; rep++ {
			if took, ctx, err = g.TimeAttention(); err != nil {
				t.Fatal(err)
			}
		}
		if base == nil {
			base = ctx
		}
		var worst, d2, b2 float64
		for i := range ctx {
			d := float64(f16(ctx[i]) - f16(base[i]))
			worst = math.Max(worst, math.Abs(d))
			d2 += d * d
			b2 += float64(f16(base[i])) * float64(f16(base[i]))
		}
		t.Logf("QT%d KTIL%d T %-5v %s: %8v  %5.1f TFLOP/s  against QT1 KTIL4: max |d| %.3g, rms rel %.3g", v.QT, v.KTIL, v.T, v.Tag,
			took.Round(time.Microsecond), fl/took.Seconds()/1e12, worst, math.Sqrt(d2/b2))
	}
}

func f16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		return float32(math.Copysign(float64(man)*math.Pow(2, -24), float64(int32(sign))*-1+1))
	case exp == 31:
		return float32(math.Inf(1))
	}
	return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
}

// testBank is the block stack's bank H3_BANK names: fp16 (the default) or
// q8 (VIDEO.md M11a).
func testBank(t *testing.T) qwen.Bank {
	s := os.Getenv("H3_BANK")
	if s == "" {
		return qwen.BankFP16
	}
	b, err := qwen.ParseBank(s)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("bank %s", b)
	return b
}
