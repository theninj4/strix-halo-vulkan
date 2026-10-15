package vae

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// gpuEncodeMaxRMS bounds a device encode's rms relative distance from the
// fp32 dump, stagewise and free-running. Measured at X7 (768x512): fp16
// cores ≤ 2.9e-3 at every stage and 2.6e-3 on the mode (144 ms); fp32
// scalar ≤ 1.7e-6 (1.04 s). 6e-3 is 2x the fp16 arm.
const gpuEncodeMaxRMS = 6e-3

// rmsRel is ‖got − want‖ / ‖want‖.
func rmsRel(t *testing.T, got, want *Tensor) float64 {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("got %s, want %s", got, want)
	}
	var sq, ref float64
	for i, w := range want.Data {
		d := float64(got.Data[i]) - float64(w)
		if math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatalf("non-finite value at %d", i)
		}
		sq += d * d
		ref += float64(w) * float64(w)
	}
	return math.Sqrt(sq / ref)
}

// TestGPUEncode runs the device encoder on the dump's 768x512 picture, on
// the fp16 matrix-core path (the default) and on the fp32 scalar one: every
// dumped stage from the pixels (RunTo), then the free encode's mode, timed.
func TestGPUEncode(t *testing.T) {
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	defer done()
	img := d.tensor(t, "enc_in")
	// the dump's stage names, and the graph's mark that produces each
	stages := [][2]string{{"enc_conv_in", "conv_in"}}
	for i, down := range enc.DownBlocks {
		mark := fmt.Sprintf("down.%d.downsample", i)
		if down.Downsampler == nil {
			mark = fmt.Sprintf("down.%d.resnets.%d", i, len(down.Resnets)-1)
		}
		stages = append(stages, [2]string{fmt.Sprintf("enc_down_blocks_%d", i), mark})
	}
	stages = append(stages, [2]string{"enc_mid_block", "mid.resnets.1"}, [2]string{"enc_conv_norm_out", "conv_norm_out"},
		[2]string{"enc_conv_out", "conv_out"}, [2]string{"enc_quant_conv", "quant_conv"})
	for _, arm := range []struct {
		name string
		opt  Options
	}{
		{"fp16 cores", Options{}},
		{"fp32 scalar", Options{Attn: AttnScalar, Conv: ConvScalar}},
	} {
		g, err := NewGPUEncoderOpts(dev, enc, img.H, img.W, arm.opt)
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range stages {
			got, err := g.RunTo(img, s[1])
			if err != nil {
				g.Destroy()
				t.Fatal(err)
			}
			r := rmsRel(t, got, d.tensor(t, s[0]))
			if r > gpuEncodeMaxRMS {
				t.Errorf("%-11s %-18s rms rel %.3g > %.0e", arm.name, s[0], r, gpuEncodeMaxRMS)
				continue
			}
			t.Logf("%-11s %-18s rms rel %.3g", arm.name, s[0], r)
		}
		var mode *Tensor
		var best time.Duration
		for rep := 0; rep < 2; rep++ {
			start := time.Now()
			if mode, err = g.Encode(img); err != nil {
				g.Destroy()
				t.Fatal(err)
			}
			if el := time.Since(start); rep == 0 || el < best {
				best = el
			}
		}
		g.Destroy()
		r := rmsRel(t, mode, d.tensor(t, "enc_mode"))
		if r > gpuEncodeMaxRMS {
			t.Errorf("%-11s encode (free) rms rel %.3g > %.0e", arm.name, r, gpuEncodeMaxRMS)
			continue
		}
		t.Logf("%-11s encode (free) rms rel %.3g, %v", arm.name, r, best.Round(time.Millisecond))
	}
}
