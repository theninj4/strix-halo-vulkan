package pipeline

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"strix-halo-vulkan/h3/plan"
	"strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/tokenizer"
)

const (
	modelDir = "../../models/MiniMax-H3"
	ditRef   = "../../reference/out/h3dit"
	vaeRef   = "../../reference/out/h3vae"
	audioRef = "../../reference/out/h3audio"
)

func TestResolveDefaults(t *testing.T) {
	tok, err := tokenizer.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	r, err := Resolve(tok, &Request{Prompt: "A cat on a windowsill."})
	if err != nil {
		t.Fatal(err)
	}
	if r.Width != 864 || r.Height != 480 || r.Frames != 124 || r.LatentFrames != 37 || r.AudioLatents != 207 || r.Steps != 20 {
		t.Errorf("defaults resolve to %dx%d, %d frames (%d latents, %d audio), %d steps; want 864x480, 124 (37, 207), 20",
			r.Width, r.Height, r.Frames, r.LatentFrames, r.AudioLatents, r.Steps)
	}
	if rows := len(r.Layout.Pos); rows != len(r.Tokens)+2*207+37*15*27 {
		t.Errorf("%d rows", rows)
	}
	for _, bad := range []*Request{
		{Prompt: ""},
		{Prompt: "x", Seconds: 4},
		{Prompt: "x", Seconds: 16},
		{Prompt: "x", ShortEdge: 128},
		{Prompt: "x", Height: 480, Width: 860},
		{Prompt: "x", Steps: 1},
	} {
		if _, err := Resolve(tok, bad); err == nil {
			t.Errorf("%+v resolved", *bad)
		}
	}
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no reference (%v)", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func readmePrompt(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(modelDir, "scripts/readme/reproducible-768p-t2va-request.sh"))
	if err != nil {
		t.Skip(err)
	}
	s := strings.SplitN(string(body), "<<'JSON'\n", 2)[1]
	s = strings.SplitN(s, "\nJSON\n", 2)[0]
	var req struct{ Prompt string }
	if err := json.Unmarshal([]byte(s), &req); err != nil {
		t.Fatal(err)
	}
	return req.Prompt
}

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("h3-pipeline-test")
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
		if devices[i].DeviceID == 0x1586 {
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

// TestE2E is M8's gate: the README's Context-IR prompt, from the string,
// through every stage to an mp4, at M4's oracle shape (448×256, 124 frames,
// N = 8) and from M4's own noise. The run is free-running, so its latents
// are not held to a tolerance (M7: two runs of the *same* path whose noise
// differs by 1e-4 end 0.1 rms apart); the gate is what the result looks and
// sounds like against the oracle's own decode of its own run — PSNR of the
// frames and SNR of the soundtrack — plus the mux. ~60 GB peak, ~5 min.
// H3_E2E_MP4=path keeps the mp4.
func TestE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 50 GB, then 44")
	}
	videoNoise := readF32(t, filepath.Join(ditRef, "noise_video.bin"))
	audioNoise := readF32(t, filepath.Join(ditRef, "noise_audio.bin"))
	wantLat := readF32(t, filepath.Join(ditRef, "f6_latents.bin"))
	wantAud := readF32(t, filepath.Join(ditRef, "f6_audio.bin"))
	wantDec := readF32(t, filepath.Join(vaeRef, "full_dec.bin"))
	wantWave := readF32(t, filepath.Join(audioRef, "wave.bin"))

	dev, done := newTestDevice(t)
	defer done()
	p, err := New(dev, modelDir, Options{})
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	defer p.Close()
	req := &Request{Prompt: readmePrompt(t), Height: 256, Width: 448, Frames: 124, Steps: 8,
		VideoNoise: videoNoise, AudioNoise: audioNoise}
	start := time.Now()
	res, err := p.Generate(context.Background(), req, func(s Stage, d, n int) {
		t.Logf("%8.1f s  %s %d/%d", time.Since(start).Seconds(), s, d, n)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d tokens, %d rows; wall %v: %v; forwards %v on the device",
		len(res.Tokens), len(res.Layout.Pos), time.Since(start).Round(time.Second), res.Timings, res.StepDevice.Round(time.Millisecond))

	_, rmsV := gap(res.VideoLatents, wantLat)
	_, rmsA := gap(res.AudioLatents, wantAud)
	t.Logf("final latents, free-running against the oracle's: video rms %.3g, audio rms %.3g (of their rms)", rmsV, rmsA)

	// The oracle's frames, as ToRGB8 would make them.
	want := vae.ToRGB8(&vae.Tensor{C: 3, T: res.Resolved.Frames, H: 256, W: 448, Data: wantDec})
	var sq float64
	var n int
	for f := range want {
		for i, v := range want[f] {
			d := float64(v) - float64(res.Video[f][i])
			sq += d * d
			n++
		}
	}
	psnr := 10 * math.Log10(255*255/(sq/float64(n)))
	var sig, noise float64
	half := len(wantWave) / 2
	for i := 0; i < half; i++ {
		for _, pr := range [][2]float32{{res.Left[i], wantWave[i]}, {res.Right[i], wantWave[half+i]}} {
			sig += float64(pr[1]) * float64(pr[1])
			e := float64(pr[0] - pr[1])
			noise += e * e
		}
	}
	snr := 10 * math.Log10(sig/noise)
	t.Logf("against the oracle's decode: frames PSNR %.1f dB, soundtrack SNR %.1f dB", psnr, snr)
	if psnr < 20 || math.IsNaN(psnr) {
		t.Errorf("frames PSNR %.1f dB", psnr)
	}
	if math.IsNaN(rmsV+rmsA) || rmsV > 0.5 || rmsA > 0.5 {
		t.Errorf("latents rms %.3g / %.3g", rmsV, rmsA)
	}

	out := os.Getenv("H3_E2E_MP4")
	if out == "" {
		out = filepath.Join(t.TempDir(), "e2e.mp4")
	}
	if err := WriteMP4(out, res); err != nil {
		t.Fatal(err)
	}
	probe, err := exec.Command("ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,nb_frames,sample_rate,channels",
		"-of", "compact", out).Output()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s:\n%s", out, probe)
	if !strings.Contains(string(probe), "h264") || !strings.Contains(string(probe), "aac") ||
		!strings.Contains(string(probe), "nb_frames="+strconv.Itoa(plan.AlignFrames(124))) {
		t.Errorf("mp4 streams: %s", probe)
	}
}

func gap(got, want []float32) (rel, rmsRel float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want[i])))
		sq += d * d
		refSq += float64(want[i]) * float64(want[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq)
}
