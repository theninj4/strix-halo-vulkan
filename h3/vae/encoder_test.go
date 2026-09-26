package vae

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The reference is reference/dump_h3_fl2va.py's prep and vae phases: the
// README fl2va keyframe on a 256x448 canvas (and a portrait crop of it,
// cover-cropped as the `last` follower, and the keyframe again at 480x864),
// through diffusers' encoder in fp32. Regenerate with:
//
//	.venv/bin/python reference/dump_h3_fl2va.py prep vae   (~12 GB, ~20 s)
const fl2vaRef = "../../reference/out/h3fl2va"

type fl2vaTensors map[string]struct {
	Shape []int `json:"shape"`
	Count int   `json:"count"`
}

func loadFL2VA(t *testing.T) fl2vaTensors {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(fl2vaRef, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_h3_fl2va.py prep vae", fl2vaRef, err)
	}
	var m struct {
		Tensors fl2vaTensors `json:"tensors"`
	}
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m.Tensors["first_moments"]; !ok {
		t.Skip("the dump has no vae phase; run reference/dump_h3_fl2va.py vae")
	}
	return m.Tensors
}

// frame reads a dumped [C, H, W] (or [C, 1, H, W]) tensor as one frame.
func (m fl2vaTensors) frame(t *testing.T, name string) *Tensor {
	t.Helper()
	meta, ok := m[name]
	if !ok {
		t.Fatalf("no tensor %q in the dump", name)
	}
	s := meta.Shape
	if len(s) == 4 {
		s = []int{s[0], s[2], s[3]}
	}
	if len(s) != 3 {
		t.Fatalf("%s is %v, not a frame", name, meta.Shape)
	}
	return &Tensor{C: s[0], T: 1, H: s[1], W: s[2], Data: readF32(t, filepath.Join(fl2vaRef, name+".bin"), meta.Count)}
}

// rgb reads a dumped [H, W, 3] keyframe as bytes.
func (m fl2vaTensors) rgb(t *testing.T, name string) ([]byte, int, int) {
	t.Helper()
	meta := m[name]
	v := readF32(t, filepath.Join(fl2vaRef, name+".bin"), meta.Count)
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out, meta.Shape[0], meta.Shape[1]
}

// TestTorchRandn holds the posterior's draw to torch's own: seed 42 over
// the 256x448 and 480x864 latents (10,752 and 38,880 values, both whole
// blocks of 16).
func TestTorchRandn(t *testing.T) {
	m := loadFL2VA(t)
	for _, name := range []string{"first_post_noise", "big_post_noise"} {
		want := m.frame(t, name).Data
		got, err := TorchRandn(KeyframeSeed, len(want))
		if err != nil {
			t.Fatal(err)
		}
		var worst float64
		for i := range want {
			worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		}
		t.Logf("%s: %d values, max abs %.3g", name, len(want), worst)
		if worst > 2e-6 {
			t.Errorf("%s: max abs %.3g against torch's draw", name, worst)
		}
	}
	// The tail rule: n not a multiple of 16 redraws the last 16. A draw of
	// 17 is 16 transformed uniforms, then 16 more over [1, 17).
	a, _ := TorchRandn(KeyframeSeed, 16)
	b, _ := TorchRandn(KeyframeSeed, 17)
	if a[0] != b[0] || a[1] == b[1] {
		t.Errorf("tail redraw: n=16 %v, n=17 %v", a[:2], b[:2])
	}
}

// TestKeyframePixels checks the input convention against the dump's
// normalised pixels, bit for bit.
func TestKeyframePixels(t *testing.T) {
	m := loadFL2VA(t)
	rgb, h, w := m.rgb(t, "key_first")
	got, err := NormalizePixels(rgb, h, w)
	if err != nil {
		t.Fatal(err)
	}
	want := m.frame(t, "first_pixels")
	for i := range want.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("pixel %d: %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

// TestSampleLatent runs the host half — sample, fp16, normalise — on the
// oracle's own moments.
func TestSampleLatent(t *testing.T) {
	m := loadFL2VA(t)
	c := testConfig(t)
	for _, k := range []string{"first", "last", "big"} {
		z, err := c.SampleLatent(m.frame(t, k+"_moments"), KeyframeSeed, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := m.frame(t, k+"_latents")
		rel, rms := gap(z.Data, want.Data)
		t.Logf("%s: rel %.3g rms %.3g", k, rel, rms)
		if rel > 1e-3 {
			t.Errorf("%s: latents rel %.3g", k, rel)
		}
	}
}

func newTestEncoder(t *testing.T, kernel EncConvKernel) (*Encoder, func()) {
	t.Helper()
	if testing.Short() {
		t.Skip("stages the encoder on the device")
	}
	dev, done := newTestDevice(t)
	start := time.Now()
	e, err := NewEncoder(dev, vaeDir, 256, 256, kernel)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("staged %.2f GB of weights and %.2f GB of activations in %v",
		float64(e.WeightBytes())/1e9, float64(e.ActivationBytes())/1e9, time.Since(start).Round(time.Millisecond))
	return e, func() { e.Destroy(); done() }
}

// TestGPUEncoderStages teacher-forces nothing: it runs the first 256x256
// tile and compares every stage the dump hooked.
func TestGPUEncoderStages(t *testing.T) {
	m := loadFL2VA(t)
	e, done := newTestEncoder(t, "")
	defer done()
	tile := m.frame(t, "first_pixels").Slice(0, 1, 0, 256, 0, 256)
	stages := []string{"conv_in"}
	for i := range e.levels {
		stages = append(stages, "down"+itoa(i)+"_r0", "down"+itoa(i))
	}
	stages = append(stages, "norm_out", "moments")
	for _, s := range stages {
		got, err := e.EncodeTile(tile, s)
		if err != nil {
			t.Fatal(err)
		}
		want := m.frame(t, "tile_"+s)
		if s == "norm_out" {
			// The dump hooks the GroupNorm module; the graph fuses its SiLU.
			for i, v := range want.Data {
				want.Data[i] = v / (1 + float32(math.Exp(float64(-v))))
			}
		}
		if got.C != want.C || got.H != want.H || got.W != want.W {
			t.Fatalf("%s: %dx%dx%d, want %dx%dx%d", s, got.C, got.H, got.W, want.C, want.H, want.W)
		}
		rel, rms := gap(got.Data, want.Data)
		t.Logf("%-10s %4dx%3dx%3d rel %.3g rms %.3g", s, got.C, got.H, got.W, rel, rms)
		// The tail norm divides a small group variance out of down5's
		// ~5e-5, which shows as 2.8e-4 at its worst element (rms 8e-6);
		// conv_out averages it back to 1e-5 in the moments.
		bound := 1e-4
		if s == "norm_out" {
			bound = 1e-3
		}
		if rel > bound {
			t.Errorf("%s: rel %.3g", s, rel)
		}
	}
}

func itoa(i int) string { return string(rune('0' + i)) }

// TestGPUEncoderKeyframes encodes whole keyframes: 256x448 is two tiles
// with the minimum overlap, and 480x864 is 3x5 tiles with widened ones.
// Then the host samples, and the result is the anchor the pipeline packs.
func TestGPUEncoderKeyframes(t *testing.T) {
	m := loadFL2VA(t)
	c := testConfig(t)
	e, done := newTestEncoder(t, "")
	defer done()
	for _, k := range []string{"first", "last", "big"} {
		x := m.frame(t, k+"_pixels")
		start := time.Now()
		mom, took, err := e.Encode(x)
		if err != nil {
			t.Fatal(err)
		}
		wall := time.Since(start)
		rel, rms := gap(mom.Data, m.frame(t, k+"_moments").Data)
		z, err := c.SampleLatent(mom, KeyframeSeed, nil)
		if err != nil {
			t.Fatal(err)
		}
		zrel, zrms := gap(z.Data, m.frame(t, k+"_latents").Data)
		t.Logf("%-5s %dx%d: moments rel %.3g rms %.3g; latents rel %.3g rms %.3g; %v device, %v wall",
			k, x.W, x.H, rel, rms, zrel, zrms, took.Round(time.Millisecond), wall.Round(time.Millisecond))
		if rel > 1e-4 || zrel > 2e-3 {
			t.Errorf("%s: moments rel %.3g, latents rel %.3g", k, rel, zrel)
		}
	}
}

// TestGPUEncoderScreen times the REFLECT builds (H3_ENC_SCREEN=1) on the
// 480x864 keyframe and asserts they agree bit for bit.
func TestGPUEncoderScreen(t *testing.T) {
	if os.Getenv("H3_ENC_SCREEN") == "" {
		t.Skip("set H3_ENC_SCREEN=1")
	}
	m := loadFL2VA(t)
	x := m.frame(t, "big_pixels")
	var ref []float32
	for _, k := range []EncConvKernel{EncConvOC16, EncConvOC32, EncConvOC48} {
		e, done := newTestEncoder(t, k)
		var best time.Duration
		var out *Tensor
		for i := 0; i < 3; i++ {
			mom, took, err := e.Encode(x)
			if err != nil {
				t.Fatal(err)
			}
			if i == 0 || took < best {
				best = took
			}
			out = mom
		}
		done()
		same := ref == nil
		if ref == nil {
			ref = out.Data
		} else {
			same = true
			for i := range ref {
				if ref[i] != out.Data[i] {
					same = false
					break
				}
			}
		}
		t.Logf("%s: %v (15 tiles), bit-identical %v", k, best.Round(time.Millisecond), same)
	}
}
