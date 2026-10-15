package dit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The GPU gates (X5) feed the device the same exact dumped inputs the CPU
// gates use. Every one stages the whole transformer (~11 GB fp16, ~1 min),
// so one staging is shared by the package's GPU tests.

const strixHaloDeviceID = 0x1586

// Staged for the largest dump the gates run: the 1024² edit (4096 image +
// 3920 reference + 80 text rows).
const (
	gpuMaxRows = 8192 + 128
	gpuMaxText = 512
	gpuChunk   = 4096
)

var (
	gpuOnce sync.Once
	gpuDev  *vk.Device
	gpuDiT  *GPU
	gpuErr  error
	gpuSkip string
)

func stagedGPU(t *testing.T) *GPU {
	t.Helper()
	if testing.Short() {
		t.Skip("a GPU gate: run without -short, and one package at a time (-p 1) -- packages in parallel share the ring, and the fp32 decoder's batches then pass amdgpu's 2 s watchdog")
	}
	needModel(t)
	gpuOnce.Do(func() {
		inst, err := vk.NewInstance("flux3-dit-test")
		if err != nil {
			gpuSkip = fmt.Sprintf("no Vulkan instance: %v", err)
			return
		}
		devices, err := inst.PhysicalDevices()
		if err != nil || len(devices) == 0 {
			gpuSkip = fmt.Sprintf("no Vulkan devices: %v", err)
			return
		}
		phys := &devices[0]
		for i := range devices {
			if devices[i].DeviceID == strixHaloDeviceID {
				phys = &devices[i]
			}
		}
		qf, err := phys.ComputeQueueFamily()
		if err != nil {
			gpuSkip = fmt.Sprintf("no compute queue: %v", err)
			return
		}
		sgs, err := phys.SubgroupSizeControl()
		if err != nil {
			gpuSkip = fmt.Sprintf("no subgroup size control: %v", err)
			return
		}
		if gpuDev, err = vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported}); err != nil {
			gpuSkip = fmt.Sprintf("no device: %v", err)
			return
		}
		gpuDiT, gpuErr = NewGPU(gpuDev, modelDir, gpuMaxRows, gpuMaxText, gpuChunk)
	})
	if gpuSkip != "" {
		t.Skip(gpuSkip)
	}
	if gpuErr != nil {
		t.Fatal(gpuErr)
	}
	return gpuDiT
}

// gpuGap is the GPU's distance from the fp32 reference: the worst element
// over the reference's absmax, and the rms error over the reference's rms.
// (compare's per-element relative error is an fp32 instrument; at fp16 a
// small element's relative error says nothing.)
func gpuGap(got, want *qwen.Mat) (rel, rmsRel float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if math.IsNaN(d) {
			return math.NaN(), math.NaN()
		}
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want.Data[i])))
		sq += d * d
		refSq += float64(want.Data[i]) * float64(want.Data[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq)
}

func gpuGate(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	rel, rms := gpuGap(got, want)
	if !(rms <= tol) {
		t.Errorf("%-22s max/absmax %.3g  rms rel %.3g > %.0e", name, rel, rms, tol)
		return
	}
	t.Logf("%-22s max/absmax %.3g  rms rel %.3g", name, rel, rms)
}

// gpuBlockTol bounds one block's rms error against fp32, on exact inputs.
// The residual passes through in fp32, so this is the fp16 branch's error
// over the whole output. X5 measured 3.9e-5 (text mode block) to 2.3e-4
// (image mode block); 1e-3 keeps 4x.
const gpuBlockTol = 1e-3

// TestGPUBlocks runs mode block 0 of each mode and single blocks 0 and 27 on
// the device from their exact dumped inputs.
func TestGPUBlocks(t *testing.T) {
	d := loadDump(t, blockDir)
	g := stagedGPU(t)
	for _, c := range []struct {
		name string
		w    blockW
	}{{"mode_txt0", g.mode[Text][0]}, {"mode_image0", g.mode[Image][0]}, {"mode_cond0", g.mode[Cond][0]}} {
		x := d.mat(t, c.name+"_in_hidden_states")
		mod := modOf(d.mat(t, c.name+"_in_temb_mod"), 0)
		got, err := g.runBlock(c.w, x, []Segment{{x.Rows, mod}}, ropeOf(t, d, c.name+"_in_image_rotary_emb"))
		if err != nil {
			t.Fatal(err)
		}
		gpuGate(t, c.name, got, d.mat(t, c.name+"_out"), gpuBlockTol)
	}
	rope := ropeOf(t, d, "single0_in_image_rotary_emb")
	for _, c := range []struct {
		name string
		w    blockW
	}{{"single0", g.single[0]}, {"single27", g.single[27]}} {
		txt, content := d.mat(t, c.name+"_in_encoder_hidden_states"), d.mat(t, c.name+"_in_hidden_states")
		cmod, tmod := d.mat(t, c.name+"_in_temb_mod"), d.mat(t, c.name+"_in_temb_mod_context")
		half := content.Rows / 2
		segs := []Segment{{txt.Rows, modOf(tmod, 0)}, {half, modOf(cmod, 0)}, {half, modOf(cmod, half)}}
		got, err := g.runBlock(c.w, stack(txt, content), segs, rope)
		if err != nil {
			t.Fatal(err)
		}
		gpuGate(t, c.name+" text", rowsOf(got, 0, txt.Rows), d.mat(t, c.name+"_out_encoder_hidden_states"), gpuBlockTol)
		gpuGate(t, c.name+" content", rowsOf(got, txt.Rows, content.Rows), d.mat(t, c.name+"_out_hidden_states"), gpuBlockTol)
	}
}

// gpuStepTol bounds a whole forward's velocity (rms) against fp32 on the
// same inputs: the teacher-forced bound. X5 measured 5.4e-4 / 5.8e-4 at
// 512² (t2i / edit) and 3.5e-4 to 9.7e-4 over the 16 steps at 1024²; 4e-3
// keeps 4x. qimage/dit's 2e-2 was too loose here: the RMS-pre-norm control
// read only 3x of it.
const gpuStepTol = 4e-3

// TestGPUForward is the stack on the device at 512², edit and t2i, against
// the dumped velocities.
func TestGPUForward(t *testing.T) {
	d := loadDump(t, blockDir)
	te := loadDump(t, textDir)
	g := stagedGPU(t)
	txt, txtIDs := te.mat(t, "short_prompt_embeds"), d.mat(t, "txt_ids")
	for _, c := range []struct {
		name       string
		cond, cids *qwen.Mat
		want       string
	}{
		{"edit", d.mat(t, "cond_tokens"), d.mat(t, "cond_ids"), "edit_velocity"},
		{"t2i", nil, nil, "t2i_velocity"},
	} {
		if err := g.Begin(txt, txtIDs, d.mat(t, "img_ids"), c.cond, c.cids); err != nil {
			t.Fatal(err)
		}
		v, took, err := g.Forward(d.mat(t, "noise"), d.T, d.TNext)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s forward: %v on the device", c.name, took)
		gpuGate(t, c.name+" velocity", v, d.mat(t, c.want), gpuStepTol)
	}
}

// TestGPUTeacherForced feeds every step's dumped latents of the 1024² t2i
// run, and of the one-reference edit (912x1152 from an 896x1120 reference:
// 8,104 joint rows), through the device and gates each velocity.
func TestGPUTeacherForced(t *testing.T) {
	for _, run := range []string{"t2i", "edit"} {
		dir := "../../reference/out/flux3run_" + run
		buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
		if err != nil {
			t.Skipf("no run dump in %s", dir)
		}
		var man struct {
			Calls []struct {
				T      float64 `json:"timestep"`
				Target float64 `json:"target_timestep"`
			} `json:"calls"`
		}
		if err := json.Unmarshal(buf, &man); err != nil {
			t.Fatal(err)
		}
		r := loadDump(t, dir)
		g := stagedGPU(t)
		var cond, cids *qwen.Mat
		if run == "edit" {
			cond, cids = r.mat(t, "cond_tokens"), r.mat(t, "cond_ids")
		}
		if err := g.Begin(r.mat(t, "prompt_embeds"), r.mat(t, "txt_ids"), r.mat(t, "img_ids"), cond, cids); err != nil {
			t.Fatal(err)
		}
		for i, c := range man.Calls {
			v, took, err := g.Forward(r.mat(t, fmt.Sprintf("step%02d_latents", i)), c.T, c.Target)
			if err != nil {
				t.Fatal(err)
			}
			gpuGate(t, fmt.Sprintf("%s step %2d (%.0f ms)", run, i, took.Seconds()*1e3), v,
				r.mat(t, fmt.Sprintf("step%02d_velocity", i)), gpuStepTol)
		}
	}
}

// TestGPUControls shows the device gates can fail: each control changes one
// thing the port could plausibly get wrong, and must land outside the bound
// the real graph meets.
func TestGPUControls(t *testing.T) {
	d := loadDump(t, blockDir)
	te := loadDump(t, textDir)
	g := stagedGPU(t)
	txt, txtIDs := te.mat(t, "short_prompt_embeds"), d.mat(t, "txt_ids")
	forward := func(cond, cids *qwen.Mat) *qwen.Mat {
		t.Helper()
		if err := g.Begin(txt, txtIDs, d.mat(t, "img_ids"), cond, cids); err != nil {
			t.Fatal(err)
		}
		v, _, err := g.Forward(d.mat(t, "noise"), d.T, d.TNext)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	check := func(name string, got *qwen.Mat, want string, bound float64) {
		t.Helper()
		_, rms := gpuGap(got, d.mat(t, want))
		if rms <= bound {
			t.Errorf("control %q lands at rms %.3g, inside %.0e — the gate cannot see it", name, rms, bound)
			return
		}
		t.Logf("control %-28s rms rel %.3g (%.0fx the bound)", name, rms, rms/bound)
	}

	// The pre-norm as an RMS norm: the mean not subtracted.
	norm := g.pipes["norm"]
	rms, err := newControlPipe(g, "h3_norm_mod")
	if err != nil {
		t.Fatal(err)
	}
	g.pipes["norm"] = rms
	v := forward(nil, nil)
	g.pipes["norm"] = norm
	check("RMS pre-norm", v, "t2i_velocity", gpuStepTol)

	// The key tail left as the previous pass wrote it: the edit's longer
	// passes first, then a t2i forward reading their rows past its keys.
	// Measured, not gated: on the transposed LDS build it moves nothing
	// (5.37e-4 either way, X5), because that build masks the tail out of the
	// row max itself; the zeroing is for the plain build, h3/dit's bug.
	forward(d.mat(t, "cond_tokens"), d.mat(t, "cond_ids"))
	g.noKeyTail = true
	v = forward(nil, nil)
	g.noKeyTail = false
	_, rmsTail := gpuGap(v, d.mat(t, "t2i_velocity"))
	t.Logf("control %-28s rms rel %.3g", "key tail not zeroed", rmsTail)
}

// TestGPUProfile times a 1024² t2i forward a dispatch at a time and sums
// by kind (FLUX3_PROFILE=1). Per-dispatch fences put the total above
// Forward's; the shares are what it is for.
func TestGPUProfile(t *testing.T) {
	if os.Getenv("FLUX3_PROFILE") == "" {
		t.Skip("FLUX3_PROFILE=1")
	}
	const dir = "../../reference/out/flux3run_t2i"
	r := loadDump(t, dir)
	g := stagedGPU(t)
	if err := g.Begin(r.mat(t, "prompt_embeds"), r.mat(t, "txt_ids"), r.mat(t, "img_ids"), nil, nil); err != nil {
		t.Fatal(err)
	}
	stages, err := g.Profile(r.mat(t, "step08_latents"), 0.87, 0.84)
	if err != nil {
		t.Fatal(err)
	}
	type sum struct {
		n     int
		sec   float64
		flops float64
	}
	by := map[string]*sum{}
	var order []string
	var total float64
	for _, s := range stages {
		k := s.Kind
		if by[k] == nil {
			by[k] = &sum{}
			order = append(order, k)
		}
		by[k].n++
		by[k].sec += s.GPU.Seconds()
		by[k].flops += s.Flops
		total += s.GPU.Seconds()
	}
	t.Logf("%d dispatches, %.1f ms", len(stages), total*1e3)
	for _, k := range order {
		s := by[k]
		rate := ""
		if s.flops > 0 {
			rate = fmt.Sprintf("%6.1f TFLOP/s", s.flops/s.sec/1e12)
		}
		t.Logf("%-16s %4d  %8.1f ms  %5.1f%%  %s", k, s.n, s.sec*1e3, 100*s.sec/total, rate)
	}
}
