package dit

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// The references are reference/dump_ace_plan.py (the encoders' inputs) and
// reference/dump_ace_dit.py (everything after them), upstream's own
// generate_audio in fp32 on the CPU:
//
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_plan.py
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_dit.py
const (
	modelDir = "../../models/acestep-v15-xl-turbo"
	planRef  = "../../reference/out/aceplan"
	ditRef   = "../../reference/out/acedit"
	// bf16Ref is the same requests run the way upstream serves them on CUDA
	// (bf16), from the fp32 run's noise: dump_ace_dit.py --bf16.
	bf16Ref = "../../reference/out/acedit_bf16"
)

type tensorMeta struct {
	Shape []int `json:"shape"`
}

type ditManifest struct {
	Cases map[string]struct {
		EncoderLen   int       `json:"encoder_len"`
		Timesteps    []float32 `json:"timesteps"`
		LatentLength int       `json:"latent_length"`
		Tokens       int       `json:"tokens"`
	} `json:"cases"`
	Tensors map[string]tensorMeta `json:"tensors"`
}

func readMat(t *testing.T, dir, name string) *qwen.Mat {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v)", dir, err)
	}
	var m struct {
		Tensors map[string]tensorMeta `json:"tensors"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("%s has no tensor %q", dir, name)
	}
	raw, err := os.ReadFile(filepath.Join(dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	cols := meta.Shape[len(meta.Shape)-1]
	n := len(raw) / 4
	out := &qwen.Mat{Rows: n / cols, Cols: cols, Data: make([]float32, n)}
	for i := range out.Data {
		out.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func loadDitManifest(t *testing.T) *ditManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(ditRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_dit.py", ditRef, err)
	}
	var m ditManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// relGap is the max abs error over the reference's absmax, and the rms
// error over the reference's rms.
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

func gate(t *testing.T, name string, got, want *qwen.Mat, maxRel, maxRMS float64) float64 {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: %v, want %v", name, got, want)
	}
	for _, v := range got.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: non-finite output", name)
		}
	}
	rel, rms := relGap(got, want)
	t.Logf("%-28s max %.2e  rms %.2e", name, rel, rms)
	if rel > maxRel || rms > maxRMS {
		t.Errorf("%s: max %.2e (limit %.0e), rms %.2e (limit %.0e)", name, rel, maxRel, rms, maxRMS)
	}
	return rms
}

// underBF16 holds a drift from the fp32 oracle to half of upstream's own
// bf16 drift on the same tensor, when the bf16 dump is there.
func underBF16(t *testing.T, name, ref string, rms float64) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(bf16Ref, ref+".bin")); err != nil {
		t.Logf("%s: no bf16 reference (%v)", name, err)
		return
	}
	_, bf := relGap(readMat(t, bf16Ref, ref), readMat(t, ditRef, ref))
	t.Logf("%-28s bf16 rms %.2e (%.1fx ours)", name, bf, bf/rms)
	if rms > bf/2 {
		t.Errorf("%s: rms %.2e is more than half of upstream bf16's %.2e", name, rms, bf)
	}
}

// TestHostTimestep: the two timestep MLPs, fp32 on the host.
func TestHostTimestep(t *testing.T) {
	m := loadDitManifest(t)
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Skip(err)
	}
	set, err := safetensors.OpenSet(modelDir)
	if err != nil {
		t.Skip(err)
	}
	defer set.Close()
	h, err := LoadHost(set, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t0 := m.Cases["full_metas"].Timesteps[0]
	e, p, err := h.Time.embed(t0)
	if err != nil {
		t.Fatal(err)
	}
	gate(t, "temb_t", &qwen.Mat{Rows: 1, Cols: len(e), Data: e}, readMat(t, ditRef, "full_metas_temb_t"), 1e-5, 1e-5)
	gate(t, "tproj_t", &qwen.Mat{Rows: 6, Cols: cfg.Hidden, Data: p}, readMat(t, ditRef, "full_metas_tproj_t"), 1e-5, 1e-5)
	e, p, err = h.TimeR.embed(0)
	if err != nil {
		t.Fatal(err)
	}
	gate(t, "temb_r", &qwen.Mat{Rows: 1, Cols: len(e), Data: e}, readMat(t, ditRef, "full_metas_temb_r"), 1e-5, 1e-5)
	gate(t, "tproj_r", &qwen.Mat{Rows: 6, Cols: cfg.Hidden, Data: p}, readMat(t, ditRef, "full_metas_tproj_r"), 1e-5, 1e-5)
}

const strixHaloDeviceID = 0x1586

var shared struct {
	once sync.Once
	g    *GPU
	err  error
	skip string
}

// sharedGPU stages the model once for every test in the package: ~10 GB of
// fp16 read out of 17 GB of fp32.
func sharedGPU(t *testing.T) *GPU {
	t.Helper()
	if testing.Short() {
		t.Skip("stages the whole model")
	}
	shared.once.Do(func() {
		inst, err := vk.NewInstance("ace-dit-test")
		if err != nil {
			shared.skip = fmt.Sprintf("no Vulkan instance: %v", err)
			return
		}
		devices, err := inst.PhysicalDevices()
		if err != nil || len(devices) == 0 {
			shared.skip = fmt.Sprintf("no Vulkan devices: %v", err)
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
			shared.skip = err.Error()
			return
		}
		sgs, err := phys.SubgroupSizeControl()
		if err != nil {
			shared.skip = err.Error()
			return
		}
		dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
		if err != nil {
			shared.skip = err.Error()
			return
		}
		if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
			shared.skip = err.Error()
			return
		}
		shared.g, shared.err = NewGPU(dev, modelDir, 7500)
	})
	if shared.skip != "" {
		t.Skip(shared.skip)
	}
	if shared.err != nil {
		t.Fatal(shared.err)
	}
	return shared.g
}

func silence(t *testing.T, rows int) *qwen.Mat {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(modelDir, "silence_latent.safetensors"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	tn, err := f.Get("silence_latent")
	if err != nil {
		t.Fatal(err)
	}
	v, err := tn.F32(nil)
	if err != nil {
		t.Fatal(err)
	}
	return &qwen.Mat{Rows: rows, Cols: 64, Data: v[:rows*64]}
}

var cases = []string{"full_metas", "defaults"}

// TestGPUEncode is A2's gate: both encoders teacher-forced a layer at a
// time, then the whole condition encoder from the plan's inputs to the
// packed sequence the DiT cross-attends to.
func TestGPUEncode(t *testing.T) {
	g := sharedGPU(t)
	loadDitManifest(t)
	for _, label := range cases {
		p := label + "_"
		for _, st := range []struct {
			s    Stack
			name string
			n    int
		}{{Lyric, "lyric", g.cfg.LyricLayers}, {Timbre, "timbre", g.cfg.TimbreLayers}} {
			for i := 1; i < st.n; i++ {
				x := readMat(t, ditRef, fmt.Sprintf("%s%s_layer%d", p, st.name, i-1))
				got, err := g.EncoderLayers(st.s, x, i, i+1)
				if err != nil {
					t.Fatal(err)
				}
				gate(t, fmt.Sprintf("%s %s layer %d", label, st.name, i), got,
					readMat(t, ditRef, fmt.Sprintf("%s%s_layer%d", p, st.name, i)), 2e-2, 5e-3)
			}
		}
		enc, err := g.Encode(readMat(t, planRef, p+"text_hidden"), readMat(t, planRef, p+"lyric_embeds"), silence(t, 750))
		if err != nil {
			t.Fatal(err)
		}
		gate(t, label+" encoder states", enc, readMat(t, ditRef, p+"encoder_states"), 2e-2, 5e-3)
	}
}

// TestGPUDiT is A3's gate: layers teacher-forced from the oracle's residual
// at five depths, every forward teacher-forced from the oracle's x_t, and
// the whole 8-step sample from the oracle's noise.
//
// The limits are measured, and one layer sets them. Layer 0's q/k norm
// weights reach 31.6, its logits are large, and fp16 q and k alone cost it
// 8e-3 of its output (a host fp32 layer with only q/k/v rounded to fp16
// reproduces the device's whole error; every other fp16 operand costs it
// ≤3e-4, and every other layer teacher-forced is ≤1e-3). That error rides
// the residual's 5.8e3 outlier into every later layer: 3-7% rms on the
// velocity, 5-9% on the latents. Upstream's own bf16 path (its CUDA
// default) drifts 3.5-16x further from fp32 on the same tensors, which
// underBF16 holds.
func TestGPUDiT(t *testing.T) {
	g := sharedGPU(t)
	m := loadDitManifest(t)
	for _, label := range cases {
		c := m.Cases[label]
		p := label + "_"
		if err := g.Begin(readMat(t, ditRef, p+"encoder_states")); err != nil {
			t.Fatal(err)
		}
		ctx := readMat(t, ditRef, p+"context_latents")
		t0 := c.Timesteps[0]
		for _, span := range [][3]int{{-1, 0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 16}, {15, 16, 32}} {
			in := "proj_in"
			if span[0] >= 0 {
				in = fmt.Sprintf("layer%d", span[0])
			}
			got, err := g.Layers(readMat(t, ditRef, p+in), t0, span[1], span[2])
			if err != nil {
				t.Fatal(err)
			}
			lim := [2]float64{2e-3, 1e-3}
			if span[1] == 0 {
				lim = [2]float64{2e-2, 1e-2}
			}
			gate(t, fmt.Sprintf("%s layers %d-%d", label, span[1], span[2]-1), got,
				readMat(t, ditRef, fmt.Sprintf("%slayer%d", p, span[2]-1)), lim[0], lim[1])
		}
		for f, tf := range c.Timesteps {
			v, took, err := g.Step(readMat(t, ditRef, fmt.Sprintf("%sx_in%d", p, f)), ctx, tf)
			if err != nil {
				t.Fatal(err)
			}
			name := fmt.Sprintf("%s v%d (%.0f ms)", label, f, took.Seconds()*1e3)
			rms := gate(t, name, v, readMat(t, ditRef, fmt.Sprintf("%sv%d", p, f)), 0.3, 0.1)
			if f == 0 {
				underBF16(t, name, p+"v0", rms) // later forwards' inputs differ in the bf16 run
			}
		}
		noise := readMat(t, ditRef, p+"noise")
		T := noise.Rows
		out, err := plan.Sample(noise.Data, plan.Schedule(3, nil), plan.DefaultDCW, func(x []float32, tt float32) ([]float32, error) {
			v, _, err := g.Step(&qwen.Mat{Rows: T, Cols: 64, Data: x}, ctx, tt)
			if err != nil {
				return nil, err
			}
			return v.Data, nil
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rms := gate(t, label+" latents", &qwen.Mat{Rows: T, Cols: 64, Data: out}, readMat(t, ditRef, p+"latents"), 0.5, 0.15)
		underBF16(t, label+" latents", p+"latents", rms)
	}
}

// TestGPUStepTiming measures one forward at 30 s, 2, 4 and 10 minutes of
// song (the oracle's condition sequence, random latents), and the profile's
// split by kind at 4 minutes.
func TestGPUStepTiming(t *testing.T) {
	g := sharedGPU(t)
	loadDitManifest(t)
	// ACE_DIT_BIG=w64|small runs the big projections on the wave64 128x256
	// build or the 64x64 rung instead (KERNELS.md G8's small-M screen).
	switch os.Getenv("ACE_DIT_BIG") {
	case "w64":
		g.big = gemmBigW64
	case "small":
		g.big = gemmSmall
	}
	defer func() { g.big = gemmBig }()
	if err := g.Begin(readMat(t, ditRef, "full_metas_encoder_states")); err != nil {
		t.Fatal(err)
	}
	for _, sec := range []int{30, 120, 240, 600} {
		T := sec * 25
		x, ctx := qwen.NewMat(T, 64), qwen.NewMat(T, 128)
		for i := range x.Data {
			x.Data[i] = float32(math.Sin(float64(i)))
		}
		var best time.Duration
		for r := 0; r < 3; r++ {
			_, took, err := g.Step(x, ctx, 0.75)
			if err != nil {
				t.Fatal(err)
			}
			if r == 0 || took < best {
				best = took
			}
		}
		t.Logf("%3d s (%4d tokens): %6.1f ms a forward, %5.2f s for 8", sec, g.Tokens(T), best.Seconds()*1e3, 8*best.Seconds())
		if sec != 240 {
			continue
		}
		st, err := g.Profile(x, ctx, 0.75)
		if err != nil {
			t.Fatal(err)
		}
		by := map[string]time.Duration{}
		fl := map[string]float64{}
		var total time.Duration
		for _, s := range st {
			by[s.Kind] += s.GPU
			fl[s.Kind] += s.Flops
			total += s.GPU
		}
		kinds := make([]string, 0, len(by))
		for k := range by {
			kinds = append(kinds, k)
		}
		sort.Slice(kinds, func(i, j int) bool { return by[kinds[i]] > by[kinds[j]] })
		for _, k := range kinds {
			tf := ""
			if fl[k] > 0 {
				tf = fmt.Sprintf("%5.1f TFLOP/s", fl[k]/by[k].Seconds()/1e12)
			}
			t.Logf("  %-18s %7.1f ms %5.1f%% %s", k, by[k].Seconds()*1e3, 100*by[k].Seconds()/total.Seconds(), tf)
		}
	}
}

const detokRef = "../../reference/out/acedetok"

func readI32(t *testing.T, path string) []int32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference (%v); run reference/dump_ace_detok.py", err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestGPUDetok is A4's gate: the detokenizer's second layer teacher-forced
// from the oracle's first, and real codes (the fp32 DiT's latents through
// the model's own audio tokenizer) to 25 Hz hints, whole and in chunks
// that do not divide the song.
func TestGPUDetok(t *testing.T) {
	g := sharedGPU(t)
	for _, label := range cases {
		p := label + "_"
		codes := readI32(t, filepath.Join(detokRef, p+"codes.bin"))
		x := readMat(t, detokRef, p+"layer0")
		got, err := g.DetokLayers(x, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		gate(t, label+" detok layer 1", got, readMat(t, detokRef, p+"layer1"), 1e-2, 1e-3)
		want := readMat(t, detokRef, p+"hints")
		for _, chunk := range []int{0, 97} {
			g.detokChunk = chunk
			h, err := g.Detokenize(codes)
			g.detokChunk = 0
			if err != nil {
				t.Fatal(err)
			}
			gate(t, fmt.Sprintf("%s hints (chunk %d)", label, chunk), h, want, 1e-2, 3e-3)
		}
	}
}
