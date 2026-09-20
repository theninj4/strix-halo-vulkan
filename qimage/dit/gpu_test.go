package dit_test

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/qimage/pipeline"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The reference is the end-to-end oracle (reference/dump_qi21_run.py): the
// diffusers pipeline in fp32, whose per-step latents Q3's CPU sampler
// reproduces at 2.5e-4. The GPU is gated against the same dumps — the fp16
// matrix-core error dominates, so the CPU-vs-oracle gap does not muddy the
// bound.
const (
	runRef      = "../../reference/out/qi21run"
	editRef     = "../../reference/out/qi21edit"
	runRef1024  = "../../reference/out/qi21run1024"
	transformer = "../../models/Qwen-Image-2.1/transformer"
	scheduler   = "../../models/Qwen-Image-2.1/scheduler"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-dit-test")
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

type runManifest struct {
	dir    string
	Prompt string `json:"prompt"`
	Size   int    `json:"size"`
	Steps  int    `json:"steps"`
	// The edit oracle's extra geometry; absent from the t2i dumps.
	CondLatentShape []int   `json:"cond_latent_shape"`
	ImgShapes       [][]int `json:"img_shapes"`
	Tensors         map[string]struct {
		Shape []int   `json:"shape"`
		Count int     `json:"count"`
		Sum   float64 `json:"sum"`
	} `json:"tensors"`
}

func loadRun(t *testing.T, dir string) *runManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v)", dir, err)
	}
	var m runManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = dir
	return &m
}

func loadMat(t *testing.T, m *runManifest, name string) *qwen.Mat {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(m.dir, name+".bin"))
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

// The fp16 bounds, measured 2026-09-20 and split by what they bound:
//
//   - gpuStepTol bounds the *model*: teacher-forced, every step fed the
//     oracle's own latents, so nothing compounds. Measured flat at
//     1.5e-3–8e-3 across all 40 steps at 1024².
//   - gpuTrajTol bounds the *trajectory*: the free-running sampler drifts
//     from the oracle's path at a steady ~1.3e-3 of rel per step (the same
//     rate at 256² and 1024²), because a perturbed ODE diverges — the
//     phenomenon diffusers documents for cache-on/off, where one-ULP input
//     differences give visibly distinct but equally valid samples. 40 steps
//     land near 5e-2; the bound is ~2x that, and the *image* judgment
//     belongs to Q6's served eyeball, not to a latent norm.
const (
	gpuStepTol = 2e-2
	gpuTrajTol = 0.12
)

func stepRel(got, want *qwen.Mat) (maxAbs, rel float64) {
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if d > maxAbs {
			maxAbs = d
		}
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	return maxAbs, rel
}

// runOracle drives the GPU through one oracle run and returns the per-step
// wall clocks. teacher selects the regime: teacher-forced feeds every step
// the oracle's own latents (bounding the model, gpuStepTol per step);
// free-running lets the trajectory evolve (bounded only at gpuTrajTol,
// logged throughout).
func runOracle(t *testing.T, g *dit.GPU, m *runManifest, lay *dit.Layout, cond *qwen.Mat,
	sched *pipeline.Schedule, teacher bool) []time.Duration {

	t.Helper()
	embeds := loadMat(t, m, "prompt_embeds")
	if err := g.BeginImage(embeds, lay, cond); err != nil {
		t.Fatal(err)
	}
	latents := loadMat(t, m, "noise").Clone()

	var walls []time.Duration
	var rel float64
	for i := 0; i < sched.Steps(); i++ {
		if teacher && i > 0 {
			latents = loadMat(t, m, fmt.Sprintf("step%d_latents", i-1)).Clone()
		}
		start := time.Now()
		out, err := g.Step(latents, sched.T(i), i == 0)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		walls = append(walls, time.Since(start))
		if err := sched.Step(i, latents.Data, out.Data); err != nil {
			t.Fatal(err)
		}
		var maxAbs float64
		maxAbs, rel = stepRel(latents, loadMat(t, m, fmt.Sprintf("step%d_latents", i)))
		t.Logf("step %2d: %7.1f ms  max abs %.4g  rel %.3g", i, float64(walls[i].Microseconds())/1000, maxAbs, rel)
		if teacher && rel > gpuStepTol {
			t.Fatalf("step %d: teacher-forced rel %.3g > %.0e", i, rel, gpuStepTol)
		}
	}
	if !teacher && rel > gpuTrajTol {
		t.Fatalf("trajectory rel %.3g > %.2g after %d steps", rel, gpuTrajTol, sched.Steps())
	}
	return walls
}

// TestGPUOracle256 is Q4's correctness gate: both row regimes — the step-0
// prefill and the cached decodes — against the 256²/4-step oracle's latents,
// with the fp32 host scheduler anchoring each step.
func TestGPUOracle256(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks")
	}
	m := loadRun(t, runRef)
	dev, done := newTestDevice(t)
	defer done()

	side := m.Size / 16
	embeds := loadMat(t, m, "prompt_embeds")
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dit.NewGPU(dev, transformer, side*side+embeds.Rows, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("teacher-forced:")
	runOracle(t, g, m, lay, nil, sched, true)
	t.Log("free-running:")
	runOracle(t, g, m, lay, nil, sched, false)
}

// TestGPUOracle1024 is the model gate at the served shape and the wall-clock
// measurement: the full 1024²/40-step oracle, teacher-forced for the bound
// and free-running for the trajectory log.
func TestGPUOracle1024(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks and runs 40 full steps")
	}
	m := loadRun(t, runRef1024)
	dev, done := newTestDevice(t)
	defer done()

	side := m.Size / 16
	embeds := loadMat(t, m, "prompt_embeds")
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dit.NewGPU(dev, transformer, side*side+embeds.Rows, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("teacher-forced:")
	runOracle(t, g, m, lay, nil, sched, true)
	t.Log("free-running:")
	walls := runOracle(t, g, m, lay, nil, sched, false)
	var total, decode time.Duration
	for i, w := range walls {
		total += w
		if i > 0 {
			decode += w
		}
	}
	t.Logf("prefill %v; %d cached steps mean %v; total %v",
		walls[0], len(walls)-1, decode/time.Duration(len(walls)-1), total)
}

// TestGPUEditOracle is Q8.5's gate: the same two regimes over the *edit*
// oracle (reference/dump_qi21_edit.py), which is the t2i dump's deliberate
// twin — same size, same steps, same seed, one 256x256 condition image.
//
// It is a gate on the transformer alone: the prompt embedding, the condition
// latents and the noise are all the oracle's own, so what is under test is
// the four things an edit changes about this graph and nothing upstream of
// them — the condition latents going through img_in into the prefix, the
// prefix's block-causal mask repaired segment by segment, a prefix KV cache
// that no longer fits the activation arena, and a rotary table with a second
// image block in it.
//
// The bounds are Q4's, unchanged. They were measured on t2i and there is no
// reason an edit should need its own: the arithmetic per row is identical and
// the row count is what changed.
func TestGPUEditOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks")
	}
	m := loadRun(t, editRef)
	if len(m.CondLatentShape) != 3 {
		t.Skipf("no edit dump in %s; run reference/dump_qi21_edit.py", editRef)
	}
	dev, done := newTestDevice(t)
	defer done()

	embeds := loadMat(t, m, "prompt_embeds")
	cond := loadMat(t, m, "cond_latents")
	mask := loadMat(t, m, "image_pad_mask")
	pad := make([]bool, mask.Rows*mask.Cols)
	for i := range pad {
		pad[i] = mask.Data[i] != 0
	}
	side := m.Size / 16
	condShape := [3]int{m.CondLatentShape[0], m.CondLatentShape[1], m.CondLatentShape[2]}
	lay, err := dit.LayoutFromPadMask(pad, [][3]int{condShape}, [3]int{1, side, side})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d VLM rows -> %d prefix + %d target = %d joint rows, %d segments",
		embeds.Rows, lay.PrefixLen, side*side, lay.Seq(), len(lay.Segments))

	g, err := dit.NewGPU(dev, transformer, lay.Seq(), lay.PrefixLen, embeds.Rows)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	// The shift is a function of the *target* token count, not the joint
	// sequence: a bigger prefix does not move the schedule.
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("teacher-forced:")
	runOracle(t, g, m, lay, cond, sched, true)
	t.Log("free-running:")
	runOracle(t, g, m, lay, cond, sched, false)

	// The controls. Each breaks one of the four mechanisms, produces a
	// full-shaped answer and no error, and has to miss the bound.
	t.Run("controls", func(t *testing.T) {
		noise := loadMat(t, m, "noise")
		want := loadMat(t, m, "step0_latents")
		run := func(what string, lay *dit.Layout, cond *qwen.Mat) {
			t.Helper()
			if err := g.BeginImage(embeds, lay, cond); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			out, err := g.Step(noise.Clone(), sched.T(0), true)
			if err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			got := noise.Clone()
			if err := sched.Step(0, got.Data, out.Data); err != nil {
				t.Fatal(err)
			}
			_, rel := stepRel(got, want)
			if rel <= gpuStepTol {
				t.Errorf("%s still matches at rel %.3g, inside the %.0e bound", what, rel, gpuStepTol)
				return
			}
			t.Logf("control %-34s rel %.3g, %.0fx the bound", what, rel, rel/gpuStepTol)
		}
		// Zeroed condition latents: the prefix is still there, still
		// attended over, and carries no picture.
		zero := qwen.NewMat(cond.Rows, cond.Cols)
		run("condition latents zeroed", lay, zero)
		// The pad mask ignored: the same 85 embedding rows read as a flat
		// t2i prompt, so the condition image never enters the sequence at
		// all and its slots become text. Every shape still checks out.
		flat, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
		if err != nil {
			t.Fatal(err)
		}
		run("pad mask ignored, condition dropped", flat, nil)
		// And the refusal: the edit's own layout with no condition latents
		// to fill its image block is a shape error, not an answer.
		if err := g.BeginImage(embeds, lay, nil); err == nil {
			t.Error("an edit layout with no condition latents was accepted")
		}
		// And the mechanism this stage exists for: the prefix's rows
		// computed bidirectionally over the whole sequence, which is what
		// the matrix-core pass leaves behind before the segment repair.
		// Re-run with the repair suppressed.
		g.SetNoPrefixRepair(true)
		run("prefix mask not repaired", lay, cond)
		g.SetNoPrefixRepair(false)
	})
}

// TestGPUEditWallClock is what an edit's transformer costs at the served
// shape: a 1024² target conditioned on one 1024² reference image, which is
// what diffusers' own `output_resolution` default resizes a condition to.
//
// There is no oracle at this size — the fp32 reference would take hours — so
// this measures rather than gates, on random weights' worth of noise in the
// right shapes. What it exercises that the 256² gate cannot is the part that
// only exists at scale: a **4136-row prefix**, whose KV cache is 2.2 GB of
// fp16 across the 32 blocks and would be 4.34 GB of fp32, past this device's
// single-buffer limit.
func TestGPUEditWallClock(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks and a 2.2 GB KV cache")
	}
	dev, done := newTestDevice(t)
	defer done()

	const side, textA, textB = 64, 20, 20
	condShape := [3]int{1, side, side}
	target := [3]int{1, side, side}
	lay, err := dit.NewLayout([]int{textA, textB}, [][3]int{condShape, target})
	if err != nil {
		t.Fatal(err)
	}
	vlmRows := len(lay.VLMMask) - side*side/4
	t.Logf("%d VLM rows -> %d prefix + %d target = %d joint rows, %d segments",
		vlmRows, lay.PrefixLen, side*side, lay.Seq(), len(lay.Segments))

	g, err := dit.NewGPU(dev, transformer, lay.Seq(), lay.PrefixLen, vlmRows)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("resident: %d MB of weights, %d MB of activations, %d MB of KV cache",
		g.WeightBytes()>>20, g.ActivationBytes()>>20, g.CacheBytes()>>20)

	rng := rand.New(rand.NewSource(7))
	fill := func(m *qwen.Mat) *qwen.Mat {
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64())
		}
		return m
	}
	embeds := fill(qwen.NewMat(vlmRows, 4096))
	cond := fill(qwen.NewMat(side*side, 64))
	latents := fill(qwen.NewMat(side*side, 64))

	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(40, side*side)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.BeginImage(embeds, lay, cond); err != nil {
		t.Fatal(err)
	}
	const steps = 4
	var prefill time.Duration
	var cached time.Duration
	for i := 0; i < steps; i++ {
		start := time.Now()
		if _, err := g.Step(latents, sched.T(i), i == 0); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		w := time.Since(start)
		if i == 0 {
			prefill = w
		} else {
			cached += w
		}
		t.Logf("step %2d: %7.3f s", i, w.Seconds())
	}
	mean := cached / (steps - 1)
	t.Logf("prefill %.3f s; cached step mean %.3f s; a 40-step edit is %.1f s of transformer",
		prefill.Seconds(), mean.Seconds(), (prefill + 39*mean).Seconds())
}

// TestGPUStepProfile is where Q9 starts, and it starts here because this is
// where the time is: Q6 measured an image as 92% denoising steps, and Q4 got
// that graph running on z-image's measured winners *uncontested* — no shape
// in this model was ever screened against an alternative.
//
// So this attributes before anything is optimised, at the served shape and in
// both regimes the sampler uses. It reports per-kind totals with the
// arithmetic each did, and then the slowest individual dispatches, because a
// kind that is 20% spread over 64 dispatches and a kind that is 20% in two
// are different problems.
func TestGPUStepProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("profiles the full-size graph one dispatch at a time")
	}
	m := loadRun(t, runRef1024)
	dev, done := newTestDevice(t)
	defer done()

	side := m.Size / 16
	embeds := loadMat(t, m, "prompt_embeds")
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dit.NewGPU(dev, transformer, side*side+embeds.Rows, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.BeginImage(embeds, lay, nil); err != nil {
		t.Fatal(err)
	}
	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	latents := loadMat(t, m, "noise").Clone()

	// The prefill first, because it is the one that fills the cache — a
	// cached step profiled before it would attend over nothing.
	for _, c := range []struct {
		name    string
		prefill bool
	}{{"prefill", true}, {"cached step", false}} {
		stages, err := g.Profile(latents, sched.T(0), c.prefill)
		if err != nil {
			t.Fatal(err)
		}
		reportStages(t, c.name, stages)
	}
}

// reportStages aggregates a profile by dispatch kind and prints the
// distribution, then the individual dispatches at the top of it.
func reportStages(t *testing.T, what string, stages []dit.Stage) {
	t.Helper()
	type agg struct {
		d     time.Duration
		flops float64
		n     int
	}
	byOp := map[string]*agg{}
	var total time.Duration
	for _, s := range stages {
		op := s.Kind
		if i := strings.IndexByte(op, ' '); i > 0 && !strings.HasPrefix(op, "gemm") {
			op = op[:i]
		}
		a := byOp[op]
		if a == nil {
			a = &agg{}
			byOp[op] = a
		}
		a.d += s.GPU
		a.flops += s.Flops
		a.n++
		total += s.GPU
	}
	ops := make([]string, 0, len(byOp))
	for op := range byOp {
		ops = append(ops, op)
	}
	sort.Slice(ops, func(i, j int) bool { return byOp[ops[i]].d > byOp[ops[j]].d })
	t.Logf("%s: %v over %d dispatches", what, total.Round(time.Millisecond), len(stages))
	for _, op := range ops {
		a := byOp[op]
		rate := ""
		if a.flops > 0 {
			rate = fmt.Sprintf("%7.1f TFLOP/s", a.flops/a.d.Seconds()/1e12)
		}
		t.Logf("  %-18s %4d x  %9v  %5.1f%%  %s",
			op, a.n, a.d.Round(time.Millisecond), 100*float64(a.d)/float64(total), rate)
	}
	sort.Slice(stages, func(i, j int) bool { return stages[i].GPU > stages[j].GPU })
	t.Log("  slowest dispatches:")
	for _, s := range stages[:min(6, len(stages))] {
		t.Logf("    %-24s %9v", s.Kind, s.GPU.Round(time.Microsecond))
	}
}

// TestGPUGEMMScreen settles the hypothesis IMAGE.md's Q9 carried from
// z-image: that the crown GEMM's swizzle arm should be re-screened at this
// model's shapes.
//
// It matters because the profile above says the GEMMs are **69% of a step**,
// so a percent here is worth more than a percent anywhere else in the graph.
// z-image chose SWZ=8 at M=16384, K=12288; Qwen-Image-2.1 runs M=4096 with
// K of 4096 and 12288, and the swizzle is a mapping from workgroup id to
// tile — the one parameter whose best value depends on how the grid divides.
//
// One staging, four arms: every arm reads the same fragment-tiled weight, so
// SetBigKernel changes which pipeline a dispatch names and nothing else.
func TestGPUGEMMScreen(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks and runs several steps per arm")
	}
	m := loadRun(t, runRef1024)
	dev, done := newTestDevice(t)
	defer done()

	side := m.Size / 16
	embeds := loadMat(t, m, "prompt_embeds")
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dit.NewGPU(dev, transformer, side*side+embeds.Rows, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.BeginImage(embeds, lay, nil); err != nil {
		t.Fatal(err)
	}
	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	latents := loadMat(t, m, "noise").Clone()
	// The prefill once, so every arm is timed on a cached step against a
	// filled cache — the regime 39 of 40 steps run in.
	if _, err := g.Step(latents, sched.T(0), true); err != nil {
		t.Fatal(err)
	}

	type result struct {
		name string
		wall time.Duration
	}
	var results []result
	for _, k := range dit.BigKernels {
		if err := g.SetBigKernel(k); err != nil {
			t.Fatal(err)
		}
		// One step to warm, three to measure.
		if _, err := g.Step(latents, sched.T(1), false); err != nil {
			t.Fatal(err)
		}
		var total time.Duration
		const runs = 3
		for i := 0; i < runs; i++ {
			start := time.Now()
			if _, err := g.Step(latents, sched.T(1), false); err != nil {
				t.Fatal(err)
			}
			total += time.Since(start)
		}
		results = append(results, result{dit.KernelName(k), total / runs})
	}
	best := results[0]
	for _, r := range results[1:] {
		if r.wall < best.wall {
			best = r
		}
	}
	for _, r := range results {
		mark := ""
		if r.name == best.name {
			mark = "  <- best"
		}
		t.Logf("%-24s %8.3f ms  %+5.1f%%%s", r.name, float64(r.wall.Microseconds())/1000,
			100*(r.wall.Seconds()/best.wall.Seconds()-1), mark)
	}
}

// TestGPUPackScreen is the profile's other question. The fragment pack moves
// 102 MB in 2.3 ms — **44 GB/s** — where the SwiGLU dispatch beside it in the
// same block reaches ~190 on the same bus. The layout it writes is the reason
// the WMMA attention kernel is worth using at all and is not in question;
// what is, is that one token tile per workgroup is 2048 elements over 256
// threads, eight each, with two barriers around them. That is a launch-bound
// shape, not a bandwidth-bound one, and TPW tiles per workgroup is the
// one-line test of it.
//
// It screens the pack and the whole step together: a kernel that is 6.6% of a
// step can only be worth what the step says it is worth.
func TestGPUPackScreen(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 14 GB of fp16 banks and runs several steps per arm")
	}
	m := loadRun(t, runRef1024)
	dev, done := newTestDevice(t)
	defer done()

	side := m.Size / 16
	embeds := loadMat(t, m, "prompt_embeds")
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dit.NewGPU(dev, transformer, side*side+embeds.Rows, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.BeginImage(embeds, lay, nil); err != nil {
		t.Fatal(err)
	}
	scfg, err := pipeline.LoadSchedConfig(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	latents := loadMat(t, m, "noise").Clone()
	if _, err := g.Step(latents, sched.T(0), true); err != nil {
		t.Fatal(err)
	}

	// The pack's own dispatches, out of a profile, and the step around them.
	type result struct {
		tiles int
		pack  time.Duration
		wall  time.Duration
	}
	var results []result
	var want *qwen.Mat
	for _, n := range dit.PackTiles {
		if err := g.SetPackTiles(n); err != nil {
			t.Fatal(err)
		}
		stages, err := g.Profile(latents, sched.T(1), false)
		if err != nil {
			t.Fatal(err)
		}
		var pack time.Duration
		for _, s := range stages {
			if strings.HasPrefix(s.Kind, "pack ") {
				pack += s.GPU
			}
		}
		if _, err := g.Step(latents, sched.T(1), false); err != nil {
			t.Fatal(err)
		}
		var total time.Duration
		const runs = 3
		for i := 0; i < runs; i++ {
			start := time.Now()
			out, err := g.Step(latents, sched.T(1), false)
			if err != nil {
				t.Fatal(err)
			}
			total += time.Since(start)
			// Every arm writes the same bytes, so every arm has to produce
			// the same prediction — bit for bit, not to a tolerance. A
			// screen that only timed them could pick one that packs wrongly.
			if want == nil {
				want = out.Clone()
			} else if maxAbs, _ := stepRel(out, want); maxAbs != 0 {
				t.Fatalf("%d tiles a workgroup changed the output by %g; the pack's layout must be identical",
					n, maxAbs)
			}
		}
		results = append(results, result{n, pack, total / runs})
	}
	best := results[0]
	for _, r := range results[1:] {
		if r.wall < best.wall {
			best = r
		}
	}
	// 102 MB read+written per pack dispatch, 64 of them a step.
	const packBytes = 64 * 102e6
	for _, r := range results {
		mark := ""
		if r.tiles == best.tiles {
			mark = "  <- best"
		}
		t.Logf("%d tile(s)/workgroup: pack %6.1f ms (%5.0f GB/s), step %8.3f ms  %+5.1f%%%s",
			r.tiles, float64(r.pack.Microseconds())/1000, packBytes/r.pack.Seconds()/1e9,
			float64(r.wall.Microseconds())/1000, 100*(r.wall.Seconds()/best.wall.Seconds()-1), mark)
	}
}
