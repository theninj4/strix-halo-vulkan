package vae

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestConvFP16Ladder is the measurement X4 left for X5 before choosing the
// decoder's device path: the t2i run's own 1024² latent decoded on the CPU
// with convolution operands narrowed to binary16 (fp32 accumulation, what a
// matrix core does), against the dump's fp32 decode and the run's PNG.
//
// Two arms, Qwen-Image's (its Q9b refused both on its own decoder, whose
// tail norm read a 2.6e5 residual): "3x3" narrows every 3x3 convolution,
// "all" the 1x1s too — post_quant_conv and the shortcuts, which read the
// raw residual stream. This decoder's activations peak at 457 (X0).
//
// A measurement, not a gate, except that an overflow fails it. ~3 min of
// CPU an arm.
func TestConvFP16Ladder(t *testing.T) {
	if testing.Short() || os.Getenv("FLUX3_LADDER") == "" {
		t.Skip("FLUX3_LADDER=1: two 1024² CPU decodes")
	}
	d := loadDump(t, dumpDir)
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(runDir, "image.bin"))
	if err != nil {
		t.Fatal(err)
	}
	want := d.tensor(t, "dec_out")
	for _, arm := range []struct {
		name string
		pick func(c *Conv2D) bool
	}{
		{"3x3", func(c *Conv2D) bool { return c.KH == 3 }},
		{"all", func(c *Conv2D) bool { return true }},
	} {
		FP16ConvOperands = arm.pick
		img, err := dec.Apply(d.tensor(t, "dec_in"))
		FP16ConvOperands = nil
		if err != nil {
			t.Fatal(err)
		}
		var maxAbs, sq, ref float64
		for i, v := range img.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: fp16 operands produced %v", arm.name, v)
			}
			dd := float64(v) - float64(want.Data[i])
			maxAbs = math.Max(maxAbs, math.Abs(dd))
			sq += dd * dd
			ref += float64(want.Data[i]) * float64(want.Data[i])
		}
		rgb := RGB8(img)
		diff, worst := 0, 0
		var lsq float64
		for i, g := range rgb {
			w := int(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
			if e := abs(int(g) - w); e > 0 {
				diff++
				worst = max(worst, e)
				lsq += float64(e * e)
			}
		}
		psnr := 10 * math.Log10(255*255/(lsq/float64(len(rgb))))
		t.Logf("%-4s decoded: max abs %.3g, rms rel %.3g; 8-bit: %d of %d levels differ (worst %d), %.1f dB",
			arm.name, maxAbs, math.Sqrt(sq/ref), diff, len(rgb), worst, psnr)
	}
}
