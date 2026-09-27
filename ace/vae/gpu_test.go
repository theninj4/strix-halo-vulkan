package vae

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strix-halo-vulkan/vk"
)

// The reference is reference/dump_ace_vae.py: diffusers' AutoencoderOobleck
// in fp32 on the fp32 DiT oracle's final latents.
//
//	HF_HUB_OFFLINE=1 .venv-acestep/bin/python reference/dump_ace_vae.py   (~45 s)
const (
	vaeDir = "../../models/Ace-Step1.5/vae"
	vaeRef = "../../reference/out/acevae"
	ditRef = "../../reference/out/acedit"
)

type manifest struct {
	Tensors map[string]struct {
		Shape []int `json:"shape"`
	} `json:"tensors"`
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

// channelLast reads a dumped [C, T] tensor as [T, C].
func channelLast(t *testing.T, m *manifest, name string) ([]float32, int, int) {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok || len(meta.Shape) != 2 {
		t.Fatalf("no [C, T] tensor %q", name)
	}
	C, T := meta.Shape[0], meta.Shape[1]
	raw := readF32(t, filepath.Join(vaeRef, name+".bin"))
	out := make([]float32, len(raw))
	for c := 0; c < C; c++ {
		for i := 0; i < T; i++ {
			out[i*C+c] = raw[c*T+i]
		}
	}
	return out, T, C
}

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

func check(t *testing.T, name string, got, want []float32, minSNR float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for _, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: non-finite output", name)
		}
	}
	rel, rms, snr := gap(got, want)
	t.Logf("%-26s max %.2e  rms %.2e  SNR %5.1f dB", name, rel, rms, snr)
	if snr < minSNR {
		t.Errorf("%s: SNR %.1f dB, want ≥ %.0f", name, snr, minSNR)
	}
}

const strixHaloDeviceID = 0x1586

func newGPU(t *testing.T, window int) (*GPU, func()) {
	t.Helper()
	inst, err := vk.NewInstance("ace-vae-test")
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
		t.Skip(err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skip(err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		t.Skip(err)
	}
	if _, err := os.Stat(filepath.Join(vaeDir, "config.json")); err != nil {
		t.Skip(err)
	}
	g, err := NewGPU(dev, vaeDir, window)
	if err != nil {
		dev.Destroy()
		inst.Destroy()
		t.Fatal(err)
	}
	return g, func() { g.Destroy(); dev.Destroy(); inst.Destroy() }
}

func loadManifest(t *testing.T) *manifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(vaeRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_ace_vae.py", vaeRef, err)
	}
	var m manifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// TestGPUStages: the decode of the 4 s excerpt, run up to the end of conv1
// and of each block and read back there, against the oracle's hooks; and
// the whole excerpt's output.
func TestGPUStages(t *testing.T) {
	m := loadManifest(t)
	g, done := newGPU(t, 512)
	defer done()
	z, n, _ := channelLast(t, m, "stage_in")
	g.upload(z, 0, n)
	gr := g.tileGraph(n)
	upTo := func(prefix string) int {
		last := -1
		for i, k := range gr.kinds {
			if strings.HasPrefix(k, prefix) {
				last = i
			}
		}
		return last
	}
	rows := n
	for i := -1; i < len(strides); i++ {
		name, prefix, ch := "stage_conv1", "conv1", widths[0]
		if i >= 0 {
			rows *= strides[i]
			name, prefix, ch = fmt.Sprintf("stage_block%d", i), fmt.Sprintf("b%d r2 add", i), widths[i+1]
		}
		end := upTo(prefix)
		g.upload(z, 0, n) // the last run's Snake passes wrote over it
		if _, err := vk.DispatchMultiTimed(gr.d[:end+1], 1, 1, true); err != nil {
			t.Fatal(err)
		}
		want, T, C := channelLast(t, m, name)
		if T != rows || C != ch {
			t.Fatalf("%s is [%d %d], the graph has [%d %d]", name, C, T, ch, rows)
		}
		check(t, name, g.abuf.ReadFloat32At(int(g.aX), rows*ch), want, 40)
	}
	got, _, err := g.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	want, _, _ := channelLast(t, m, "stage_out")
	check(t, "stage_out", got, want, 40)
}

// TestGPUDecode: both cases' whole decode, tiled at a 512-latent window,
// against the untiled oracle, and the -1 dBFS audio upstream writes.
func TestGPUDecode(t *testing.T) {
	m := loadManifest(t)
	g, done := newGPU(t, 512)
	defer done()
	for _, label := range []string{"full_metas", "defaults"} {
		z := readF32(t, filepath.Join(ditRef, label+"_latents.bin"))
		got, took, err := g.Decode(z)
		if err != nil {
			t.Fatal(err)
		}
		T := len(z) / Latent
		t.Logf("%s: %d latents (%.0f s of audio) in %.0f ms on the device", label, T, float64(T)/25, took.Seconds()*1e3)
		want, _, _ := channelLast(t, m, label+"_untiled")
		check(t, label+" untiled", got, want, 40)
		Normalize(got, -1)
		want, _, _ = channelLast(t, m, label+"_final")
		check(t, label+" final", got, want, 40)
	}
}
