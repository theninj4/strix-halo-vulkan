package pipeline

import (
	"math"
	"runtime"
	"testing"

	"strix-halo-vulkan/qimage/dit"
	"strix-halo-vulkan/qimage/textenc"
	qvae "strix-halo-vulkan/qimage/vae"
	"strix-halo-vulkan/zimage/qwen"
	"strix-halo-vulkan/zimage/tokenizer"
	zvae "strix-halo-vulkan/zimage/vae"
)

const (
	model     = "../../models/Qwen-Image-2.1"
	imgSpec   = 2.2e-2 // Z-Image's end-to-end precedent, in [0, 1] image space
	embedSpec = 1e-3   // textenc's own deep bound; localizes a text-side break
)

// TestEndToEndImage is the gate Q3 owed to Q5: prompt → image through this
// repo's own tokenizer, text encoder, DiT, scheduler and VAE decoder — only
// the initial noise comes from the oracle, as data — against the 256²/4-step
// reference image. The three component gates do not compose into this claim;
// this is the measurement. ~65 GB of fp32 weights pass through, sequenced so
// the peak stays near the text encoder's 34 GB; it must run alone on the
// machine (the full oracle run and this test together OOM'd it once).
func TestEndToEndImage(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the whole fp32 pipeline, ~10 min")
	}
	m := loadRunManifest(t)

	// --- text: tokenize and encode the oracle's prompt ourselves.
	tcfg, err := textenc.LoadConfig(model + "/text_encoder")
	if err != nil {
		t.Skipf("no checkpoint at %s (%v)", model, err)
	}
	tok, err := tokenizer.Load(model + "/processor")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := textenc.EncodePrompt(tok, m.Prompt)
	if err != nil {
		t.Fatal(err)
	}
	drop, err := textenc.DropTokens(tok)
	if err != nil {
		t.Fatal(err)
	}
	te, err := qwen.LoadWith(model+"/text_encoder", tcfg, tcfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	out, err := te.Forward(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	embeds, err := textenc.Drop(out, drop)
	if err != nil {
		t.Fatal(err)
	}
	embeds = embeds.Clone()
	te = nil
	runtime.GC()
	compareStepAt(t, "prompt_embeds", embeds, loadRunMat(t, m, "prompt_embeds"), embedSpec)

	// --- denoise from the oracle's noise.
	dcfg, err := dit.LoadConfig(model + "/transformer")
	if err != nil {
		t.Fatal(err)
	}
	dm, err := dit.Load(model+"/transformer", dcfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	side := m.Size / 16
	lay, err := dit.NewLayout([]int{embeds.Rows}, [][3]int{{1, side, side}})
	if err != nil {
		t.Fatal(err)
	}
	scfg, err := LoadSchedConfig(model + "/scheduler")
	if err != nil {
		t.Fatal(err)
	}
	sched, err := scfg.Timesteps(m.Steps, side*side)
	if err != nil {
		t.Fatal(err)
	}
	latents, err := Denoise(dm, embeds, lay, nil, loadRunMat(t, m, "noise"), sched, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm = nil
	runtime.GC()

	// --- decode: unpack [tokens, C] to [1, C, h, w], denormalize, decode.
	vcfg, err := qvae.LoadConfig(model + "/vae")
	if err != nil {
		t.Fatal(err)
	}
	dec, err := qvae.LoadDecoder(model+"/vae", vcfg)
	if err != nil {
		t.Fatal(err)
	}
	z := zvae.NewTensor(1, latents.Cols, side, side)
	for tk := 0; tk < latents.Rows; tk++ {
		row := latents.Row(tk)
		for c := 0; c < latents.Cols; c++ {
			z.Plane(0, c)[tk] = row[c]
		}
	}
	vcfg.Denormalize(z)
	img, err := dec.Decode(z)
	if err != nil {
		t.Fatal(err)
	}

	// --- against the oracle's image, in its own [0, 1], HWC layout.
	want := loadRunMat(t, m, "image") // [H*W, 4]
	var maxAbs, sumAbs float64
	for h := 0; h < img.H; h++ {
		for w := 0; w < img.W; w++ {
			for c := 0; c < img.C; c++ {
				got := (float64(img.Plane(0, c)[h*img.W+w]) + 1) / 2
				d := math.Abs(got - float64(want.Row(h*img.W+w)[c]))
				if d > maxAbs {
					maxAbs = d
				}
				sumAbs += d
			}
		}
	}
	mean := sumAbs / float64(img.C*img.H*img.W)
	if maxAbs > imgSpec {
		t.Errorf("image: max abs %.5f (mean %.6f) > %.3g against the oracle", maxAbs, mean, imgSpec)
		return
	}
	t.Logf("image: max abs %.5f, mean %.6f against the oracle (bound %.3g)", maxAbs, mean, imgSpec)
}

// compareStepAt is compareStep with an explicit bound.
func compareStepAt(t *testing.T, name string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", name, got, want)
	}
	var sumSq float64
	for i := range want.Data {
		w := float64(want.Data[i])
		sumSq += w * w
	}
	rms := math.Sqrt(sumSq / float64(len(want.Data)))
	var rel float64
	for i := range want.Data {
		d := math.Abs(float64(got.Data[i]) - float64(want.Data[i]))
		if r := d / math.Max(math.Abs(float64(want.Data[i])), math.Max(rms, 1e-12)); r > rel {
			rel = r
		}
	}
	if rel > tol {
		t.Fatalf("%s: worst rel %.3g > %.0e", name, rel, tol)
	}
	t.Logf("%-16s rel %.2g", name, rel)
}
