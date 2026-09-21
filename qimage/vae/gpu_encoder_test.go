package vae

import (
	"fmt"
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// loadGPUEncoder loads the CPU encoder and builds its device graph for one
// image size. The cleanup is registered rather than deferred for the reason
// loadGPU gives: cleanups run after the test body's defers, so the device
// must be torn down after the pipelines it owns.
func loadGPUEncoder(t *testing.T, dev *vk.Device, h, w int) (*Config, *Encoder, *GPUEncoder) {
	t.Helper()
	cfg, err := LoadConfig(vaeDir)
	if err != nil {
		t.Skipf("no VAE checkpoint at %s (%v)", vaeDir, err)
	}
	enc, err := LoadEncoder(vaeDir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUEncoder(dev, enc, h, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	return cfg, enc, g
}

// TestGPUEncoderStages walks the device graph stage by stage against the same
// dump the CPU encoder is gated on — conv_in, every down block, the mid
// block, the fused tail norm, conv_out, and the posterior mode both raw and
// normalised — for both of the dump's cases.
//
// The bounds are vae_test.go's, unchanged. This path is fp32 throughout, so
// summation order is all that separates it from the CPU port, and the two
// pieces that are genuinely new here — the AvgDown shortcut and the
// stride-2-as-subsampled-stride-1 downsampler — are structural: getting
// either wrong moves a down block by rel >= 1e-1, not by a rounding step.
func TestGPUEncoderStages(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	for label, c := range m.Cases {
		t.Run(label, func(t *testing.T) {
			cfg, _, g := loadGPUEncoder(t, dev, c.H, c.W)
			n, err := g.Dispatches(c.H, c.W)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: %dx%d, %d dispatches, %d MB activations, %d MB weights",
				label, c.H, c.W, n, g.ActivationBytes()>>20, g.WeightBytes()>>20)

			card := loadRef(t, m, label+"_card")
			stages, err := g.Stages(c.H, c.W)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range stages {
				if name == "quant" {
					continue // the dump stops at conv_out; the mode below covers it
				}
				got, err := g.RunTo(card, name)
				if err != nil {
					t.Fatal(err)
				}
				compare(t, label+"_enc_"+name, got, loadRef(t, m, label+"_enc_"+name))
			}

			mode, err := g.Encode(t.Context(), card)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, label+"_encoded_mode", mode, loadRef(t, m, label+"_encoded_mode"))
			cfg.Normalize(mode)
			compare(t, label+"_encoded_norm", mode, loadRef(t, m, label+"_encoded_norm"))
		})
	}
}

// TestGPUEncoderMatchesCPU is the same encode on both paths, end to end —
// the tighter instrument of the two, because it isolates what the device
// graph does differently from the Go one it was ported from rather than
// measuring both against the reference's own fp32 noise.
func TestGPUEncoderMatchesCPU(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	c, ok := m.Cases["s256"]
	if !ok {
		t.Skip("reference has no s256 case")
	}
	_, enc, g := loadGPUEncoder(t, dev, c.H, c.W)
	card := loadRef(t, m, "s256_card")

	want, err := enc.Encode(card)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Encode(t.Context(), card)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "s256_gpu_vs_cpu_encoded", got, want)
}

// TestGPUEncoderNegativeControls break the two pieces of this graph that are
// not the decoder's, and check the bound catches each. Both produce a
// correctly shaped answer and no error, which is why they have to be shown
// to fail rather than assumed to.
func TestGPUEncoderNegativeControls(t *testing.T) {
	m := loadManifest(t)
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	c, ok := m.Cases["s256"]
	if !ok {
		t.Skip("reference has no s256 case")
	}
	card := loadRef(t, m, "s256_card")

	// 1. The AvgDown shortcut's temporal factor. At T = 1 it moves no data,
	// but it decides which source values a group is made of *and* whether a
	// zero frame is averaged in — so setting it to 1 on a temporal block
	// halves nothing and doubles that shortcut. The break has to be measured
	// on a block that *has* a temporal factor: the checkpoint's
	// temperal_downsample is [false, true, true, true], so block 0 does not
	// and measuring there would show nothing however wrong the kernel was.
	t.Run("avgdown temporal factor", func(t *testing.T) {
		_, enc, _ := loadGPUEncoder(t, dev, c.H, c.W)
		broken := *enc
		broken.Downs = append([]DownBlock(nil), enc.Downs...)
		first := -1
		for i := range broken.Downs {
			if broken.Downs[i].Shortcut.FactorT != 1 {
				if first < 0 {
					first = i
				}
				broken.Downs[i].Shortcut.FactorT = 1
			}
		}
		if first < 0 {
			t.Skip("no down block has a temporal factor to break")
		}
		g, err := NewGPUEncoder(dev, &broken, c.H, c.W)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		stage := fmt.Sprintf("down_blocks.%d", first)
		control(t, "avgdown temporal factor", g, card,
			stage, loadRef(t, m, "s256_enc_"+stage))
	})

	// 2. The stride-2 identity's offset. Keeping (2h, 2w) instead of
	// (2h+1, 2w+1) is the plausible mistake — it is the subsampling a reader
	// would write without knowing the padding is (0, 1, 0, 1) — and it is
	// half a pixel of shift through every downsampler.
	t.Run("downsample offset", func(t *testing.T) {
		_, enc, _ := loadGPUEncoder(t, dev, c.H, c.W)
		broken := *enc
		broken.Downs = append([]DownBlock(nil), enc.Downs...)
		for i := range broken.Downs {
			if broken.Downs[i].Down == nil {
				continue
			}
			// A symmetric-pad stride-2 filter is what the shifted
			// subsampling computes, and the loader will not build one, so
			// the break is expressed as the conv the graph would need.
			conv := broken.Downs[i].Down.Conv
			conv.Pad, conv.PadEnd = 1, 0
			down := *broken.Downs[i].Down
			down.Conv = conv
			broken.Downs[i].Down = &down
		}
		if _, err := NewGPUEncoder(dev, &broken, c.H, c.W); err == nil {
			t.Error("a symmetric-pad stride-2 downsampler was accepted; the identity only holds for (0,1,0,1)")
		} else {
			t.Logf("control %-26s refused: %v", "downsample padding", err)
		}
	})
}

// control runs a deliberately broken graph to one stage and requires it to
// land far outside the bound the gate uses.
func control(t *testing.T, what string, g *GPUEncoder, card *Tensor, stage string, want *Tensor) {
	t.Helper()
	got, err := g.RunTo(card, stage)
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
	var sumSq float64
	for i := range want.Data {
		sumSq += float64(want.Data[i]) * float64(want.Data[i])
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	tol := tolFor("s256_enc_" + stage)
	if rel <= tol {
		t.Errorf("%s still matches at rel %.3g, inside the %.0e bound", what, rel, tol)
		return
	}
	t.Logf("control %-26s rel %.3g, %.0fx the bound", what, rel, rel/tol)
}

// TestGPUEncodeTiming is the number an edit's prefix costs per reference
// image, against the CPU encoder this replaces.
func TestGPUEncodeTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("times a full-size encode")
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	const side = 1024
	_, _, g := loadGPUEncoder(t, dev, side, side)
	img := NewTensor(1, 4, side, side)
	for i := range img.Data {
		img.Data[i] = float32((i%255))/127.5 - 1
	}
	n, err := g.Dispatches(side, side)
	if err != nil {
		t.Fatal(err)
	}
	for run := 0; run < 2; run++ {
		start := time.Now()
		if _, err := g.Encode(t.Context(), img); err != nil {
			t.Fatal(err)
		}
		t.Logf("%dx%d encode: %v over %d dispatches, %d MB activations",
			side, side, time.Since(start).Round(time.Millisecond), n, g.ActivationBytes()>>20)
	}
}
