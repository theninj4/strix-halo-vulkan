package pipeline

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"strix-halo-vulkan/ace/plan"
	h3vae "strix-halo-vulkan/h3/vae"
	"strix-halo-vulkan/vk"
)

// The references: reference/dump_ace_plan.py (requests, text encoder),
// dump_ace_dit.py (the fp32 DiT and its bf16 twin), dump_ace_vae.py (the
// decode and the -1 dBFS audio).
const (
	models  = "../../models"
	planRef = "../../reference/out/aceplan"
	ditRef  = "../../reference/out/acedit"
	bf16Ref = "../../reference/out/acedit_bf16"
	vaeRef  = "../../reference/out/acevae"
)

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

func request(t *testing.T, label string) *plan.Request {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(planRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v)", planRef, err)
	}
	var m struct {
		Cases map[string]struct {
			Params struct {
				Caption       string  `json:"caption"`
				Lyrics        string  `json:"lyrics"`
				BPM           int     `json:"bpm"`
				KeyScale      string  `json:"keyscale"`
				TimeSignature string  `json:"timesignature"`
				Duration      float64 `json:"duration"`
				Language      string  `json:"vocal_language"`
			} `json:"params"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	c, ok := m.Cases[label]
	if !ok {
		t.Fatalf("no case %q", label)
	}
	p := c.Params
	return &plan.Request{Caption: p.Caption, Lyrics: p.Lyrics, BPM: p.BPM, KeyScale: p.KeyScale,
		TimeSignature: p.TimeSignature, Duration: p.Duration, Language: p.Language}
}

// gap is the max abs error over the reference's absmax, the rms error over
// the reference's rms, and that as an SNR.
func gap(got, want []float32) (rel, rms, snr float64) {
	var maxAbs, ref, sq, refSq float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maxAbs = math.Max(maxAbs, d)
		ref = math.Max(ref, math.Abs(float64(want[i])))
		sq += d * d
		refSq += float64(want[i]) * float64(want[i])
	}
	return maxAbs / ref, math.Sqrt(sq / refSq), 10 * math.Log10(refSq/sq)
}

// channelLast reads a dumped [2, N] stereo tensor as interleaved samples.
func interleaved(t *testing.T, path string) []float32 {
	raw := readF32(t, path)
	n := len(raw) / 2
	out := make([]float32, len(raw))
	for i := 0; i < n; i++ {
		out[2*i], out[2*i+1] = raw[i], raw[n+i]
	}
	return out
}

// TestNoise: the seeded noise is the oracle's.
func TestNoise(t *testing.T) {
	want := readF32(t, filepath.Join(ditRef, "full_metas_noise.bin"))
	got, err := h3vae.TorchRandn(42, len(want))
	if err != nil {
		t.Fatal(err)
	}
	rel, _, _ := gap(got, want)
	t.Logf("seed-42 noise: max %.2e of absmax", rel)
	if rel > 1e-6 {
		t.Fatalf("noise differs by %.2e", rel)
	}
}

const strixHaloDeviceID = 0x1586

var shared struct {
	once sync.Once
	p    *Pipeline
	err  error
	skip string
}

func pipeline(t *testing.T) *Pipeline {
	t.Helper()
	if testing.Short() {
		t.Skip("stages the whole model")
	}
	shared.once.Do(func() {
		inst, err := vk.NewInstance("ace-pipeline-test")
		if err != nil {
			shared.skip = err.Error()
			return
		}
		devices, err := inst.PhysicalDevices()
		if err != nil || len(devices) == 0 {
			shared.skip = "no Vulkan devices"
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
		d := DefaultDirs(models)
		for _, dir := range []string{d.DiT, d.VAE, d.Text} {
			if _, err := os.Stat(dir); err != nil {
				shared.skip = err.Error()
				return
			}
		}
		shared.p, shared.err = New(dev, d, 240)
	})
	if shared.skip != "" {
		t.Skip(shared.skip)
	}
	if shared.err != nil {
		t.Fatal(shared.err)
	}
	return shared.p
}

// TestCondition: the caption's last hidden state through `embed` against
// upstream's text encoder, and the lyric rows, which are a lookup.
func TestCondition(t *testing.T) {
	p := pipeline(t)
	for _, label := range []string{"full_metas", "defaults", "long_caption", "instrumental_60"} {
		text, lyric, err := p.Condition(request(t, label))
		if err != nil {
			t.Fatal(err)
		}
		want := readF32(t, filepath.Join(planRef, label+"_text_hidden.bin"))
		if len(want) != len(text.Data) {
			t.Fatalf("%s: text hidden %v, want %d values", label, text, len(want))
		}
		rel, rms, _ := gap(text.Data, want)
		t.Logf("%-16s text hidden %v  max %.2e  rms %.2e", label, text, rel, rms)
		if rms > 5e-3 {
			t.Errorf("%s: text hidden rms %.2e", label, rms)
		}
		wantL := readF32(t, filepath.Join(planRef, label+"_lyric_embeds.bin"))
		if len(wantL) != len(lyric.Data) {
			t.Fatalf("%s: lyric embeds %v, want %d values", label, lyric, len(wantL))
		}
		for i := range wantL {
			if lyric.Data[i] != wantL[i] {
				t.Fatalf("%s: lyric embed %d = %v, want %v", label, i, lyric.Data[i], wantL[i])
			}
		}
	}
}

// TestGenerate is A6's gate: a request from the plan to the audio, seed 42,
// against the fp32 oracle's latents (held under half of upstream bf16's
// drift from them, as ace/dit's gate does) and its decoded, normalised
// audio; then the file through ffmpeg.
func TestGenerate(t *testing.T) {
	p := pipeline(t)
	for _, label := range []string{"full_metas", "defaults"} {
		r := request(t, label)
		res, err := p.Generate(r, Options{Seed: 42})
		if err != nil {
			t.Fatal(err)
		}
		tm := res.Timings
		t.Logf("%s: %.1f s of audio in %.2f s (text %.0f ms, encoders %.0f ms, DiT %.2f s, VAE %.2f s)",
			label, res.Seconds, tm.Total.Seconds(), tm.Text.Seconds()*1e3, tm.Encode.Seconds()*1e3, tm.DiT.Seconds(), tm.VAE.Seconds())
		want := readF32(t, filepath.Join(ditRef, label+"_latents.bin"))
		_, rms, _ := gap(res.Latents, want)
		bf := readF32(t, filepath.Join(bf16Ref, label+"_latents.bin"))
		_, bfRMS, _ := gap(bf, want)
		t.Logf("%s latents: rms %.2e from fp32; upstream bf16 %.2e (%.1fx)", label, rms, bfRMS, bfRMS/rms)
		if rms > bfRMS/2 {
			t.Errorf("%s: latents rms %.2e, more than half of bf16's %.2e", label, rms, bfRMS)
		}
		audio := interleaved(t, filepath.Join(vaeRef, label+"_final.bin"))
		if len(audio) != len(res.Audio) {
			t.Fatalf("%s: %d samples, want %d", label, len(res.Audio), len(audio))
		}
		_, arms, snr := gap(res.Audio, audio)
		t.Logf("%s audio: rms %.2e, SNR %.1f dB against the fp32 oracle's", label, arms, snr)
	}
	if _, err := exec.LookPath(FFmpeg); err != nil {
		t.Skip("no ffmpeg")
	}
	res, err := p.Generate(request(t, "full_metas"), Options{Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for ext := range Formats {
		path := filepath.Join(dir, "song."+ext)
		if err := Write(path, res.Audio); err != nil {
			t.Fatal(err)
		}
		st, err := os.Stat(path)
		if err != nil || st.Size() < 1000 {
			t.Fatalf("%s: %v %v", path, st, err)
		}
		t.Logf("%s: %d KB", filepath.Base(path), st.Size()>>10)
	}
}
