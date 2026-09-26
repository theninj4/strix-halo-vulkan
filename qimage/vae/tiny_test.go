package vae

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"
)

// The reference is reference/dump_taeqi.py: taesd's own F16Decoder in fp32
// over the fp16 checkpoint, on the final latent of a real 40-step 1024² run
// and on a 24x40 crop of it with every Sequential layer tapped.
//
//	.venv/bin/python reference/dump_taeqi.py
const (
	tinyRefDir   = "../../reference/out/taeqi"
	tinyModelDir = "../../models/taeqi2_1"
)

// The gates, measured and then given room. Everything is fp32 on both sides,
// so what separates the port from torch is summation order through 20
// convolutions, and at the image — [-1, 1] after the clamp — that is well
// under one 8-bit level (0.0078).
const (
	tinyStageRel = 1e-4 // max abs over the tensor's own absmax, per layer
	tinyImageAbs = 2e-4 // max abs at the image
)

type tinyManifest struct {
	Tensors map[string]struct {
		Shape []int `json:"shape"`
	} `json:"tensors"`
	VsFull struct {
		RMS float64 `json:"rms"`
	} `json:"vs_full_vae"`
}

func loadTinyManifest(t *testing.T) *tinyManifest {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(tinyRefDir, "manifest.json"))
	if err != nil {
		t.Skipf("no reference dump in %s (%v); run reference/dump_taeqi.py", tinyRefDir, err)
	}
	var m tinyManifest
	if err := json.Unmarshal(buf, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func loadTinyRef(t *testing.T, m *tinyManifest, name string) *Tensor {
	t.Helper()
	meta, ok := m.Tensors[name]
	if !ok {
		t.Fatalf("reference has no tensor %q", name)
	}
	sh := meta.Shape
	if len(sh) != 4 {
		t.Fatalf("%s: shape %v, want 4-D", name, sh)
	}
	raw, err := os.ReadFile(filepath.Join(tinyRefDir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := NewTensor(sh[0], sh[1], sh[2], sh[3])
	if len(raw) != 4*out.Len() {
		t.Fatalf("%s: %d bytes for shape %v", name, len(raw), sh)
	}
	copy(out.Data, unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), out.Len()))
	return out
}

func loadTiny(t *testing.T) *TinyDecoder {
	t.Helper()
	d, err := LoadTinyDecoder(tinyModelDir)
	if err != nil {
		t.Skipf("no tiny decoder: %v", err)
	}
	return d
}

// tinyDiff is the max abs difference and the reference's absmax.
func tinyDiff(t *testing.T, name string, got, want *Tensor) (maxAbs, absMax float64) {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: got %s, want %s", name, got, want)
	}
	for i, w := range want.Data {
		if d := math.Abs(float64(got.Data[i]) - float64(w)); d > maxAbs {
			maxAbs = d
		}
		if a := math.Abs(float64(w)); a > absMax {
			absMax = a
		}
	}
	return maxAbs, absMax
}

func checkStage(t *testing.T, name string, got, want *Tensor) {
	t.Helper()
	d, a := tinyDiff(t, name, got, want)
	rel := d / math.Max(a, 1e-6)
	t.Logf("%-12s %-16s max abs %.3g (absmax %.3g, rel %.2g)", name, fmt.Sprint(got.Shape()[1:]), d, a, rel)
	if rel > tinyStageRel {
		t.Errorf("%s: rel %.3g past %.0e", name, rel, tinyStageRel)
	}
}

func checkImage(t *testing.T, name string, got, want *Tensor) {
	t.Helper()
	d, _ := tinyDiff(t, name, got, want)
	t.Logf("%s: max abs %.3g at the image (gate %.0e)", name, d, tinyImageAbs)
	if d > tinyImageAbs {
		t.Errorf("%s: max abs %.3g past %.0e", name, d, tinyImageAbs)
	}
}

// TestTinyDecoder is the Go port against taesd, layer by layer, on the
// rectangular crop.
func TestTinyDecoder(t *testing.T) {
	m := loadTinyManifest(t)
	d := loadTiny(t)
	z := loadTinyRef(t, m, "crop_latent")
	img, err := d.Decode(z, func(layer int, x *Tensor) {
		name := fmt.Sprintf("crop_layer%02d", layer)
		checkStage(t, name, x, loadTinyRef(t, m, name))
	})
	if err != nil {
		t.Fatal(err)
	}
	checkImage(t, "crop_image", img, loadTinyRef(t, m, "crop_image"))
}

// TestTinyNegativeControls are the two conventions a preview gets wrong
// while still producing a picture, each checked to be caught by the image
// gate: the shuffle's row and column offsets swapped (a transposed 2x2 block
// is invisible at a glance), and the latent denormalized first as the full
// decoder's is (the tanh clamp squashes it into something recognisable).
func TestTinyNegativeControls(t *testing.T) {
	m := loadTinyManifest(t)
	d := loadTiny(t)
	z := loadTinyRef(t, m, "crop_latent")
	want := loadTinyRef(t, m, "crop_image")

	t.Run("shuffle transposed", func(t *testing.T) {
		var pre *Tensor
		if _, err := d.Decode(z, func(layer int, x *Tensor) {
			if layer == 19 {
				pre = x
			}
		}); err != nil {
			t.Fatal(err)
		}
		// Swap channel k = i*2+j for j*2+i, i.e. 1 <-> 2 within every group.
		swapped := NewTensor(1, pre.C, pre.H, pre.W)
		for c := 0; c < pre.C; c++ {
			src := c
			switch c % 4 {
			case 1:
				src = c + 1
			case 2:
				src = c - 1
			}
			copy(swapped.Plane(0, c), pre.Plane(0, src))
		}
		bad, err := TinyToImage(swapped)
		if err != nil {
			t.Fatal(err)
		}
		dd, _ := tinyDiff(t, "shuffle", bad, want)
		if dd <= tinyImageAbs {
			t.Fatalf("a transposed shuffle decodes at max abs %.3g, inside the %.0e gate", dd, tinyImageAbs)
		}
		t.Logf("transposed shuffle: max abs %.3g, %.0fx the gate", dd, dd/tinyImageAbs)
	})

	t.Run("denormalized latent", func(t *testing.T) {
		cfg, err := LoadConfig(vaeDir)
		if err != nil {
			t.Skipf("no VAE config: %v", err)
		}
		raw := &Tensor{N: 1, C: z.C, H: z.H, W: z.W, Data: append([]float32(nil), z.Data...)}
		cfg.Denormalize(raw)
		bad, err := d.Decode(raw, nil)
		if err != nil {
			t.Fatal(err)
		}
		dd, _ := tinyDiff(t, "denorm", bad, want)
		if dd <= tinyImageAbs {
			t.Fatalf("a denormalized latent decodes at max abs %.3g, inside the %.0e gate", dd, tinyImageAbs)
		}
		t.Logf("denormalized latent: max abs %.3g, %.0fx the gate", dd, dd/tinyImageAbs)
	})
}

// TestTinyGPUDecoder is the device graph against the dump: every layer on the
// crop, then the full 1024² image.
func TestTinyGPUDecoder(t *testing.T) {
	m := loadTinyManifest(t)
	d := loadTiny(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	full := loadTinyRef(t, m, "latent")
	g, err := NewTinyGPUDecoder(dev, d, full.H, full.W)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	t.Logf("kernel %s, %d MB activations, %d MB weights", g.Kernel(), g.ActivationBytes()>>20, g.WeightBytes()>>20)

	z := loadTinyRef(t, m, "crop_latent")
	stages, err := g.Stages(z.H, z.W)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range stages {
		got, err := g.RunTo(z, s)
		if err != nil {
			t.Fatal(err)
		}
		checkStage(t, "crop_"+s, got, loadTinyRef(t, m, "crop_"+s))
	}
	img, err := g.Decode(t.Context(), z)
	if err != nil {
		t.Fatal(err)
	}
	checkImage(t, "crop_image", img, loadTinyRef(t, m, "crop_image"))

	img, err = g.Decode(t.Context(), full)
	if err != nil {
		t.Fatal(err)
	}
	checkImage(t, "image", img, loadTinyRef(t, m, "image"))
}

// TestTinyKernelScreen times the epilogue convolution's builds at 1024² and
// asserts they agree bit for bit: the block sizes change no accumulator's
// summation order, and a screen that only timed the arms could pick one that
// computes the wrong thing quickly.
func TestTinyKernelScreen(t *testing.T) {
	if testing.Short() {
		t.Skip("screens the full-size graph")
	}
	m := loadTinyManifest(t)
	d := loadTiny(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	z := loadTinyRef(t, m, "latent")

	var first *Tensor
	for _, k := range TinyKernels() {
		g, err := NewTinyGPUDecoderKernel(dev, d, z.H, z.W, k)
		if err != nil {
			t.Fatal(err)
		}
		var img *Tensor
		best := time.Duration(math.MaxInt64)
		for run := 0; run < 3; run++ {
			start := time.Now()
			if img, err = g.Decode(t.Context(), z); err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		g.Destroy()
		t.Logf("%-5s best of 3: %v", k, best.Round(100*time.Microsecond))
		if first == nil {
			first = img
			continue
		}
		for i := range first.Data {
			if first.Data[i] != img.Data[i] {
				t.Fatalf("%s differs from %s at %d: %g vs %g", k, TinyKernels()[0], i, img.Data[i], first.Data[i])
			}
		}
	}
}

// TestTinyGPUTiming is what a preview frame costs at the served sizes. Two
// runs, per the project's convention.
func TestTinyGPUTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("timing runs the full-size graph")
	}
	d := loadTiny(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	for _, side := range []int{512, 1024} {
		lat := side / TinyScale
		t.Run(fmt.Sprintf("%dx%d", side, side), func(t *testing.T) {
			g, err := NewTinyGPUDecoder(dev, d, lat, lat)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(g.Destroy)
			z := NewTensor(1, 64, lat, lat)
			for i := range z.Data {
				z.Data[i] = float32(math.Sin(float64(i) * 0.001))
			}
			for run := 0; run < 2; run++ {
				start := time.Now()
				if _, err := g.Decode(t.Context(), z); err != nil {
					t.Fatal(err)
				}
				t.Logf("run %d: %v (%d MB activations)", run, time.Since(start).Round(100*time.Microsecond), g.ActivationBytes()>>20)
			}
		})
	}
}
