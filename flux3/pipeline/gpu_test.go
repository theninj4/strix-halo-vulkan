package pipeline

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"strix-halo-vulkan/flux3/vae"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	if testing.Short() {
		t.Skip("a GPU gate: run without -short, and one package at a time (-p 1) -- packages in parallel share the ring, and the fp32 decoder's batches then pass amdgpu's 2 s watchdog")
	}
	inst, err := vk.NewInstance("flux3-pipeline-test")
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
		t.Skipf("no subgroup size control: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// rmsRel is the rms error over the reference's rms.
func rmsRel(got, want *qwen.Mat) float64 {
	var sq, ref float64
	for i, w := range want.Data {
		d := float64(got.Data[i]) - float64(w)
		sq += d * d
		ref += float64(w) * float64(w)
	}
	return math.Sqrt(sq / ref)
}

// trajTol bounds the free-running trajectory's final latent (rms) against
// the fp32 run's, and minPSNR the decoded picture against the run's PNG.
// X5 measured, all on the device (encoder, DiT, fp16 decoder), 1.95e-3 and
// 63.1 dB, a mean of 0.03 levels; with the dumped text embedding and the CPU
// decoder it was 1.34e-3 and 65.5 dB. The drift grows ~1.4x a step. 1e-2 and
// 45 dB keep margin for a different prompt's trajectory.
const (
	trajTol = 1e-2
	minPSNR = 45
)

// TestGPUFreeRun is X5's end-to-end gate: the 1024² t2i run's prompt
// *text* through the device encoder, its noise through 16 free-running
// steps of the device DiT, and the device decoder, compared with the
// picture the fp32 run wrote. Setting FLUX3_PNG writes the picture there.
func TestGPUFreeRun(t *testing.T) {
	r := loadRun(t)
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", modelDir)
	}
	dev, done := newTestDevice(t)
	defer done()
	p, err := NewGPU(dev, Config{Model: filepath.Dir(modelDir), MaxWidth: 1024, MaxHeight: 1024, MaxPrompt: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Destroy()
	var x *qwen.Mat
	img, tm, err := p.Generate(context.Background(), Request{
		Prompt: r.Prompt, Width: 1024, Height: 1024, Steps: r.Steps, Noise: r.mat(t, "step00_latents"),
		Progress: func(st Step) {
			x = st.Latent
			if st.Index+1 < r.Steps {
				t.Logf("after step %2d: latent rms rel %.3g", st.Index, rmsRel(x, r.mat(t, fmt.Sprintf("step%02d_latents", st.Index+1))))
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("encode %v, denoise %v (%v on the device), decode %v", tm.Encode.Round(time.Millisecond),
		tm.Denoise.Round(time.Millisecond), tm.Device.Round(time.Millisecond), tm.Decode.Round(time.Millisecond))
	checkFinal(t, r, x, img)
}

// checkFinal compares a free run's last latent with the dump's (its last
// step finished from the dump) and its picture with the run's PNG, and
// writes the picture to FLUX3_PNG when that is set.
func checkFinal(t *testing.T, r *run, x *qwen.Mat, img *vae.Tensor) {
	t.Helper()
	s, err := NewSchedule(r.Steps, Alpha)
	if err != nil {
		t.Fatal(err)
	}
	want := r.mat(t, fmt.Sprintf("step%02d_latents", r.Steps-1))
	if err := s.Step(want, r.mat(t, fmt.Sprintf("step%02d_velocity", r.Steps-1)), r.Steps-1); err != nil {
		t.Fatal(err)
	}
	if rel := rmsRel(x, want); !(rel <= trajTol) {
		t.Errorf("final latent rms rel %.3g > %.2g", rel, trajTol)
	} else {
		t.Logf("final latent rms rel %.3g", rel)
	}

	rgb := vae.RGB8(img)
	raw, err := os.ReadFile(filepath.Join(r.dir, "image.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4*len(rgb) {
		t.Fatalf("a %dx%d picture against the run's %d values", img.W, img.H, len(raw)/4)
	}
	var sq, sad float64
	for i, g := range rgb {
		d := float64(g) - float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		sq += d * d
		sad += math.Abs(d)
	}
	psnr := 10 * math.Log10(255*255/(sq/float64(len(rgb))))
	if psnr < minPSNR {
		t.Errorf("picture PSNR %.2f dB < %d (mean |diff| %.2f levels)", psnr, minPSNR, sad/float64(len(rgb)))
	} else {
		t.Logf("picture PSNR %.2f dB, mean |diff| %.2f levels", psnr, sad/float64(len(rgb)))
	}
	if path := os.Getenv("FLUX3_PNG"); path != "" {
		pic := image.NewNRGBA(image.Rect(0, 0, img.W, img.H))
		for i := 0; i < img.W*img.H; i++ {
			copy(pic.Pix[i*4:], rgb[i*3:i*3+3])
			pic.Pix[i*4+3] = 255
		}
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := png.Encode(f, pic); err != nil {
			t.Fatal(err)
		}
	}
}

// TestGPUEdit is X7's end-to-end gate: the edit run's prompt text and its
// reference picture (the figurine, prepared by PrepareReference -- on the
// grid already, so BFL's and diffusers' preparations agree), encoded on the
// device, with the run's noise through 16 free steps at the run's own
// 912x1152 (diffusers' size choice; BFL's default would be the
// reference's), against the fp32 run's latents and PNG. FLUX3_EDIT_SCALAR=1
// encodes the reference on the fp32 scalar path instead: the control for
// the fp16 encoder's 2.6e-3.
func TestGPUEdit(t *testing.T) {
	r := loadRunDir(t, editDir)
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", modelDir)
	}
	f, err := os.Open(filepath.Join(filepath.Dir(modelDir), "assets/1_figurine.png"))
	if err != nil {
		t.Skip(err)
	}
	ref, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	cfg := Config{Model: filepath.Dir(modelDir), MaxWidth: 912, MaxHeight: 1152, MaxPrompt: 512, Refs: 1}
	if os.Getenv("FLUX3_EDIT_SCALAR") == "1" {
		cfg.RefEncoder = vae.Options{Attn: vae.AttnScalar, Conv: vae.ConvScalar}
	}
	p, err := NewGPU(dev, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Destroy()

	// the reference's tokens and ids against the dump's
	refs, err := PrepareReferences([]image.Image{ref}, p.Area())
	if err != nil {
		t.Fatal(err)
	}
	cond, condIDs, err := p.encodeReferences(refs)
	if err != nil {
		t.Fatal(err)
	}
	if rel := rmsRel(cond, r.mat(t, "cond_tokens")); !(rel <= condTol) {
		t.Errorf("reference tokens rms rel %.3g > %.0e", rel, condTol)
	} else {
		t.Logf("reference tokens %dx%d: rms rel %.3g", cond.Rows, cond.Cols, rel)
	}
	if rel := rmsRel(condIDs, r.mat(t, "cond_ids")); rel != 0 {
		t.Errorf("reference ids differ: rms rel %.3g", rel)
	}

	var x *qwen.Mat
	img, tm, err := p.Generate(context.Background(), Request{
		Prompt: r.Prompt, Width: 912, Height: 1152, Steps: r.Steps, Noise: r.mat(t, "step00_latents"),
		References: []image.Image{ref},
		Progress: func(st Step) {
			x = st.Latent
			if st.Index+1 < r.Steps {
				t.Logf("after step %2d: latent rms rel %.3g", st.Index, rmsRel(x, r.mat(t, fmt.Sprintf("step%02d_latents", st.Index+1))))
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("references %v (%d tokens), encode %v, denoise %v (%v on the device), decode %v",
		tm.References.Round(time.Millisecond), tm.CondTokens, tm.Encode.Round(time.Millisecond),
		tm.Denoise.Round(time.Millisecond), tm.Device.Round(time.Millisecond), tm.Decode.Round(time.Millisecond))
	checkFinal(t, r, x, img)
}

// condTol bounds the device-encoded reference tokens against the dump's.
// X7 measured 1.8e-3 on the fp16 encoder (the picture 58.8 dB, mean 0.07
// levels) and 2.9e-6 on the fp32 scalar one (62.9 dB, 0.03 levels, +3.5 s).
const condTol = 5e-3
