package vae

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The reference is reference/dump_taef2.py: upstream taesd.py's flux_2
// decoder in fp32 over the t2i run's latent, every layer dumped.
const (
	taef2Dir  = "../../models/taef2"
	taef2Dump = "../../reference/out/taef2"
)

func loadTiny(t *testing.T) (*TinyDecoder, *dump) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(taef2Dir, "taef2.safetensors")); err != nil {
		t.Skipf("no taef2 at %s", taef2Dir)
	}
	d := loadDump(t, taef2Dump)
	dec, err := LoadTinyDecoder(taef2Dir, TAEF2Config())
	if err != nil {
		t.Fatal(err)
	}
	return dec, d
}

// TestTinyLayers runs every layer of the decoder on the dump's input to that
// layer (the clamp first) and gates its output; then the whole decode free.
func TestTinyLayers(t *testing.T) {
	dec, d := loadTiny(t)
	pools := 0
	for _, l := range dec.Layers {
		if l.Block != nil && l.Block.Pool != nil {
			pools++
		}
	}
	if pools != 3 {
		t.Fatalf("%d blocks with a pool branch, want taef2's 3", pools)
	}
	in := "clamp"
	for _, l := range dec.Layers {
		x := d.tensor(t, in)
		var got *Tensor
		var err error
		switch {
		case l.Conv != nil:
			got, err = l.Conv.Apply(x)
		case l.Block != nil:
			got, err = l.Block.Apply(x)
		case l.Upsample:
			got = UpsampleNearest2x(x)
		case l.ReLU:
			got = ReLUInPlace(x)
		}
		if err != nil {
			t.Fatal(err)
		}
		name := fmt.Sprintf("layer%02d", l.Index)
		compare(t, name, got, d.tensor(t, name), stageTol)
		in = name
	}
	img, err := dec.Apply(d.tensor(t, "latent"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "image (free)", img, d.tensor(t, "image"), stageTol)
}

// gpuTinyMinPSNR bounds the device preview against the oracle's picture.
const gpuTinyMinPSNR = 40

// TestGPUTiny decodes the oracle's latent on the device, fp16 matrix cores
// and the fp32 scalar control, against the oracle's image, and times both.
func TestGPUTiny(t *testing.T) {
	dec, d := loadTiny(t)
	dev, done := newTestDevice(t)
	defer done()
	z, want := d.tensor(t, "latent"), d.tensor(t, "image")
	for _, arm := range []struct {
		name string
		opt  Options
	}{{"fp16 cores", Options{}}, {"fp32 scalar", Options{Conv: ConvScalar}}} {
		g, err := NewGPUTinyOpts(dev, dec, z.H, z.W, arm.opt)
		if err != nil {
			t.Fatal(err)
		}
		var img *Tensor
		var best time.Duration
		for rep := 0; rep < 3; rep++ {
			start := time.Now()
			if img, err = g.Apply(z); err != nil {
				g.Destroy()
				t.Fatal(err)
			}
			if el := time.Since(start); rep == 0 || el < best {
				best = el
			}
		}
		g.Destroy()
		var sq, ref, maxAbs float64
		for i, v := range img.Data {
			w := float64(want.Data[i])
			// both sides clamp to [-1, 1] before 8 bits; compare there
			a, b := math.Max(-1, math.Min(1, float64(v))), math.Max(-1, math.Min(1, w))
			sq += (a - b) * (a - b)
			ref += b * b
			maxAbs = math.Max(maxAbs, math.Abs(a-b))
		}
		psnr := 10 * math.Log10(4/(sq/float64(len(img.Data)))) // range 2
		msg := "%-11s %v: max abs %.3g, rms rel %.3g, %.1f dB"
		if psnr < gpuTinyMinPSNR {
			t.Errorf(msg+" < %d dB", arm.name, best.Round(time.Millisecond), maxAbs, math.Sqrt(sq/ref), psnr, gpuTinyMinPSNR)
			continue
		}
		t.Logf(msg, arm.name, best.Round(time.Millisecond), maxAbs, math.Sqrt(sq/ref), psnr)
	}
}

// TestTinyConvention gates what taef2 is *fed*: on the t2i run's latent, the
// normalised input must decode close to the full VAE's PNG and the raw one
// (the first port's mistake: oversaturated previews) must not. The device
// fp16 path, so it is the served preview being measured.
func TestTinyConvention(t *testing.T) {
	dec, d := loadTiny(t)
	raw := loadDump(t, dumpDir).tensor(t, "dec_in")
	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUTiny(dev, dec, 128, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	psnr := func(z *Tensor) float64 {
		img, err := g.Apply(z)
		if err != nil {
			t.Fatal(err)
		}
		rawPNG, err := os.ReadFile(filepath.Join(runDir, "image.bin"))
		if err != nil {
			t.Fatal(err)
		}
		var sq float64
		for i, v := range RGB8(img) {
			w := float64(math.Float32frombits(binary.LittleEndian.Uint32(rawPNG[i*4:])))
			sq += (float64(v) - w) * (float64(v) - w)
		}
		return 10 * math.Log10(255*255/(sq/float64(img.H*img.W*3)))
	}
	norm, rawDB := psnr(d.tensor(t, "latent")), psnr(raw)
	if norm < 30 || rawDB > 25 {
		t.Errorf("taef2 vs the full VAE's picture: normalised input %.1f dB (want ≥ 30), raw %.1f dB (want ≤ 25)", norm, rawDB)
		return
	}
	t.Logf("taef2 vs the full VAE's picture: normalised input %.1f dB, raw (the control) %.1f dB", norm, rawDB)
}
