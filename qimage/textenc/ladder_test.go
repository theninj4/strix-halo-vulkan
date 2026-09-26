package textenc

import (
	"fmt"
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// TestGPULadder walks the GPU encoder layer by layer against the CPU model
// (which TestEncoder pins to the reference) and logs each layer's deviation.
// It is the bisecting instrument for the fp16 path on this checkpoint, whose
// residual stream carries a ~13k massive-activation channel from layer 6 —
// something the Qwen3-4B this GPU path was built against never had. It loads
// ~50 GB (34 fp32 + 15 fp16), so it must run alone on the machine.
func TestGPULadder(t *testing.T) {
	if os.Getenv("QI21_LADDER") == "" {
		t.Skip("diagnostic; set QI21_LADDER=1 (loads ~50 GB and must run alone)")
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	ids := m.Prompts["en"].IDs

	cpu, err := qwen.LoadWith(encoder, cfg, cfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	tr := qwen.Trace{}
	if _, err := cpu.Forward(ids, tr); err != nil {
		t.Fatal(err)
	}

	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	dev, done := newTestDevice(t)
	defer done()
	g, err := qwen.NewGPUEncoder(dev, set, cfg, cfg.NumLayers, 512, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	for layer := 0; layer < cfg.NumLayers; layer++ {
		if err := g.RunTo(ids, layer, "resid ffn"); err != nil {
			t.Fatal(err)
		}
		got := g.Read(g.TensorX(), cfg.HiddenSize)
		want := tr[fmt.Sprintf("hidden_%d", layer+1)]
		maxAbs, rms, rel, _ := deviation(got, want)
		t.Logf("layer %2d: max abs %8.4f  rms %9.3f  rel %.3g", layer, maxAbs, rms, rel)
	}
}

// TestGPUBankLadder walks the QIMAGE_BANK encoder against the fp16 one layer
// by layer, both on the device, and logs each layer's deviation and the
// column it is worst in. It is the bisecting instrument for the int8 bank:
// the fp16 path is 20x tighter to fp32, so it serves as the reference, and
// the pair is ~22 GB rather than the fp32 ladder's ~50.
func TestGPUBankLadder(t *testing.T) {
	if os.Getenv("QI21_BANK_LADDER") == "" {
		t.Skip("diagnostic; set QI21_BANK_LADDER=1 and QIMAGE_BANK=q8 (stages ~22 GB)")
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	ids := m.Prompts["en"].IDs
	set, err := safetensors.OpenSet(encoder)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	dev, done := newTestDevice(t)
	defer done()
	ref, err := qwen.NewGPUEncoder(dev, set, cfg, cfg.NumLayers, 512, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Destroy()
	g := newBankEncoder(t, dev, set, cfg)
	defer g.Destroy()

	for layer := 0; layer < cfg.NumLayers; layer++ {
		for _, label := range []string{"resid attn", "resid ffn"} {
			if err := ref.RunTo(ids, layer, label); err != nil {
				t.Fatal(err)
			}
			want := ref.Read(ref.TensorX(), cfg.HiddenSize)
			if err := g.RunTo(ids, layer, label); err != nil {
				t.Fatal(err)
			}
			got := g.Read(g.TensorX(), cfg.HiddenSize)
			maxAbs, rms, rel, worst := deviation(got, want)
			t.Logf("layer %2d %-10s: max abs %8.4f  rms %9.3f  rel %.3g  worst col %d (want %.1f)",
				layer, label, maxAbs, rms, rel, worst%cfg.HiddenSize, want.Data[worst])
		}
	}
}
