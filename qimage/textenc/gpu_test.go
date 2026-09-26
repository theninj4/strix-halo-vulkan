package textenc

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("qi21-textenc-test")
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

// The fp16 bounds are measured, and the story behind them matters (all
// numbers 2026-09-20, TestGPULadder and the torch probes beside it):
//
// This checkpoint's residual stream grows a ~13k massive-activation channel
// at layer 6 (and again at 16), carries it flat for 18 layers, and *cancels*
// it in layers 34-35 — the two layers Z-Image's hidden_states[-2] never ran,
// which is why zimage/qwen's fp16 path could hold 6e-3 and this one cannot.
// Every reduced-precision pass through the FFN injects ~5e-4 of bulk
// rounding variance into that channel (keeping just the spike elements exact
// measurably does NOT help), and when the channel collapses the carried
// error surfaces: rel 0.024 on a real prompt, 0.21 elementwise on the
// all-template empty prompt whose rows are mostly cancellation residue.
//
// Priced against the model's own deployment: the *official* bf16 pipeline
// deviates from fp32 by rel 2.0 (max abs 204) through the same mechanism —
// two orders of magnitude beyond this path. If Q3/Q6 ever show prompt
// adherence suffering, the known lever is an fp32-activation GEMM arm for
// the encoder, and at prompt-sized M (~40 rows) the encoder is
// weight-bandwidth-bound, so that arm would cost little; it is not built
// because nothing yet shows it is needed.
const (
	gpuTol      = 3e-2 // en/cjk: measured 0.024-0.027
	gpuEmptyTol = 0.25 // empty prompt: measured 0.214, residue-dominated
)

// The int8 bank's bounds are on the rms of the difference from fp32, not on
// the worst element: a quantised bank moves every element a little, and the
// worst-element rel then names one residue element of the cancelled
// massive-activation channel (rel 0.87 at one element of an edit prompt whose
// rms diff is 0.27). They are priced against the released pipeline, whose
// bf16 encoder is reference/out/qi21textenc_bf16 (research/qimage-vertical.md
// Q13). Measured 2026-09-26, Q8KeepFP16 = {6, 16}:
//
//	              int8   bf16   int8 throughout
//	en            0.244  0.765  0.402
//	cjk           0.245  0.903  0.439
//	empty         2.58   2.34   3.61
//	edit, 1 ref   0.273
//	edit, 2 refs  0.307
//	edit layer 0  1.4e-3 (fp16 6.6e-5)
//
// The empty prompt is all template and cancellation residue (the fp16 path
// is rel 0.21 on it too), and it is the one place int8 is not inside bf16.
const (
	q8RMSTol      = 0.4  // en/cjk
	q8EmptyRMSTol = 3.0  // empty
	q8EditRMSTol  = 0.45 // edit prompts, one and two references
	q8LayerRMSTol = 5e-3 // edit layer 0
	// q8BF16Ratio bounds int8's rms diff by bf16's on the same prompt:
	// en/cjk measured 0.32/0.27. q8EmptyBF16Ratio is VIDEO.md M11a's 1.25.
	q8BF16Ratio      = 0.5
	q8EmptyBF16Ratio = 1.25
)

// gate holds got to fp32 at the bound for the bank under test: worst-element
// rel for fp16, rms diff for int8.
func gate(t *testing.T, name string, got, want *qwen.Mat, relTol, q8Tol float64) {
	t.Helper()
	if testBank(t) != qwen.BankQ8 {
		compareAt(t, name, got, want, relTol)
		return
	}
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	d := rmsDiff(got, want)
	if d > q8Tol || math.IsNaN(d) {
		t.Errorf("%s %s: q8 rms diff %.4g > %.3g", name, got, d, q8Tol)
		return
	}
	t.Logf("%-22s %-14s q8 rms diff %.4g (bound %.3g)", name, got.String(), d, q8Tol)
}

// testBank is the bank QIMAGE_BANK names: fp16 (the default) or q8.
func testBank(t *testing.T) qwen.Bank {
	t.Helper()
	s := os.Getenv("QIMAGE_BANK")
	if s == "" {
		return qwen.BankFP16
	}
	b, err := qwen.ParseBank(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// newBankEncoder stages the encoder in testBank's bank. Under q8 the layers
// held in fp16 are Q8KeepFP16, or QIMAGE_Q8_KEEP's comma list ("" for none,
// "-" for the int8 bank throughout).
func newBankEncoder(t *testing.T, dev *vk.Device, set *safetensors.Set, cfg *qwen.Config) *qwen.GPUEncoder {
	t.Helper()
	bank := testBank(t)
	keep := Q8KeepFP16
	if s, ok := os.LookupEnv("QIMAGE_Q8_KEEP"); ok {
		keep = nil
		for _, f := range strings.Split(s, ",") {
			if f == "" || f == "-" {
				continue
			}
			l, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			keep = append(keep, l)
		}
	}
	if bank != qwen.BankQ8 {
		keep = nil
	}
	start := time.Now()
	g, err := qwen.NewGPUEncoderMixed(dev, set, cfg, cfg.NumLayers, 512, nil, bank, keep)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("staged %s keeping %v in fp16 (%.2f GB) in %v", bank, keep, float64(g.WeightBytes())/1e9, time.Since(start).Round(time.Millisecond))
	return g
}

// TestGPUEncoder is Q1's GPU acceptance gate: zimage/qwen's GPUEncoder,
// unchanged, over the Qwen3-VL config — all 36 layers, no final norm — for
// the three dumped prompts. ~15 GB of fp16 banks, staged one layer at a
// time.
func TestGPUEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 15 GB of fp16 banks")
	}
	m := loadManifest(t)
	if _, err := os.Stat(encoder); err != nil {
		t.Skipf("no text encoder checkpoint at %s", encoder)
	}
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	dev, done := newTestDevice(t)
	defer done()
	g := newBankEncoder(t, dev, set, cfg)
	defer g.Destroy()
	bf := loadBF16Manifest(t)

	for _, label := range []string{"en", "cjk", "empty"} {
		p := m.Prompts[label]
		out, err := g.Forward(p.IDs)
		if err != nil {
			t.Fatal(err)
		}
		tol, q8Tol, ratio := gpuTol, q8RMSTol, q8BF16Ratio
		if label == "empty" {
			tol, q8Tol, ratio = gpuEmptyTol, q8EmptyRMSTol, q8EmptyBF16Ratio
		}
		if label == "en" && testBank(t) != qwen.BankQ8 {
			compareAt(t, "en_last_prenorm", out, loadRef(t, m, "en_last_prenorm"), tol)
		}
		embeds, err := Drop(out, m.DropIdx)
		if err != nil {
			t.Fatal(err)
		}
		want := loadRef(t, m, label+"_prompt_embeds")
		if bf != nil {
			b := loadRef(t, bf, label+"_prompt_embeds")
			gAbs, _, gRel, _ := deviation(embeds, want)
			bAbs, _, bRel, _ := deviation(b, want)
			g, bd := rmsDiff(embeds, want), rmsDiff(b, want)
			t.Logf("%-6s %s: max abs %.4g, rel %.3g, rms diff %.4g | official bf16: max abs %.4g, rel %.3g, rms diff %.4g",
				label, testBank(t), gAbs, gRel, g, bAbs, bRel, bd)
			if testBank(t) == qwen.BankQ8 && g > ratio*bd {
				t.Errorf("%s: q8 rms diff %.4g is %.2fx the official bf16's, past %.2fx", label, g, g/bd, ratio)
			}
		}
		gate(t, label+"_prompt_embeds", embeds, want, tol, q8Tol)
	}
}

// bf16RefDir is reference/dump_qi21_textenc_bf16.py's output: the released
// pipeline's own bf16 encoder on the same three prompts, which is the scale
// a quantised bank is priced on.
const bf16RefDir = "../../reference/out/qi21textenc_bf16"

// loadBF16Manifest is the bf16 dump, or nil (logged) when it was never run.
func loadBF16Manifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(bf16RefDir, "manifest.json"))
	if err != nil {
		t.Logf("no bf16 dump in %s; run reference/dump_qi21_textenc_bf16.py", bf16RefDir)
		return nil
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	m.dir = bf16RefDir
	return &m
}

// rmsDiff is the root mean square of got - want.
func rmsDiff(got, want *qwen.Mat) float64 {
	var s float64
	for i := range want.Data {
		d := float64(got.Data[i]) - float64(want.Data[i])
		s += d * d
	}
	return math.Sqrt(s / float64(len(want.Data)))
}
