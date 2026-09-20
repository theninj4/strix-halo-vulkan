package textenc

import (
	"fmt"
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// Q8.4's gate: the edit text encoding on the device.
//
// The conditions are the **dump's own** merged and deepstack rows rather than
// our tower's, for the same reason the CPU path's two-image case uses them —
// it makes this a gate on the text encoding instead of on the pair, and
// qimage/vision already gates the tower against the same dump. What is under
// test here is only what EncodeGPU adds to Q1's GPUEncoder: the host scatter
// before the upload, the uploaded mrope table, and the deepstack additions
// between the first three layers.
//
// The bound is gpuEditTol, measured here and not borrowed from Q1: an edit
// prompt lands at 0.059 (one image) and 0.064 (two) where the t2i prompts
// measure 0.024-0.027. That is twice the t2i number, and the reason it is a
// bound rather than a bug is TestGPUEditLadder, which walks the same run
// against the CPU port layer by layer and finds **Q1's mechanism, unchanged
// and in the same place**:
//
//	layers 0-5    rel ~1.1e-3, flat — the scatter, the mrope table and the
//	              deepstack injection are exact to fp16 rounding
//	layer 6       the massive-activation channel appears (rms 0.98 -> 21.1)
//	              and carries an absolute error of 4.46 *flat*
//	layer 16      a second jump (rms 26.0), the carried error 8.83, and it
//	              stays 8.83 and in the same column (270) for 18 layers —
//	              carried, not accumulated
//	layers 34-35  the channel cancels (rms 27.8 -> 20.0 -> 13.0) and the
//	              carried error surfaces: rel 0.0049 -> 0.0146 -> 0.0532
//
// So the last two layers produce the whole of the deviation, exactly as Q1
// described for t2i. And the *absolute* deviation does not move at all
// between the two paths: the worst element is **2.38 in all three cases** —
// the t2i "en" prompt, the one-image edit and the two-image edit, measured in
// the same hour. What the extra 2.4x buys is a different worst *relative*
// element (0.69 on an element of 11.8) under a lower rms floor (10.05 against
// en's 13), which is the metric moving rather than the error.
// Priced the same way Q1 priced its own: the
// *official* bf16 pipeline deviates from fp32 by rel 2.0 on this checkpoint,
// 30x further than this. The known lever, if an edit's prompt adherence ever
// looks wrong, is still the fp32-activation GEMM arm Q1 priced and did not
// build.
//
// edit_layer0_out is gated at layerTol instead, which is 4x its measurement
// rather than 90x: the shallow reading is what keeps a real mistake in the
// three new mechanisms from hiding under the fp16 bound.
const (
	gpuEditTol = 1e-1 // measured 0.059 (one image) / 0.064 (two)
	layerTol   = 5e-3 // layer 0 measured 1.1e-3
)

func TestGPUEditEncoder(t *testing.T) {
	if testing.Short() {
		t.Skip("stages 15 GB of fp16 banks")
	}
	m := loadEditManifest(t)
	if _, err := os.Stat(encoder); err != nil {
		t.Skipf("no text encoder checkpoint at %s", encoder)
	}
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Fatal(err)
	}
	section, err := LoadMRope(encoder)
	if err != nil {
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

	// --- one condition image.
	p := editPrompt(t, m)
	one := []Condition{dumpedCondition(t, m, "vis")}
	rope, err := p.MRope(cfg.HeadDim, cfg.RopeTheta, section)
	if err != nil {
		t.Fatal(err)
	}
	// Layer 0's output, read off the device before its deepstack injection —
	// where the reference's own forward hook fires. It is the one
	// intermediate that says whether the scatter and the table are right
	// *before* 36 layers of fp16 have had a chance to explain a difference.
	embeds, mask, err := p.EncodeGPU(g, rope, one, m.DropIdx, func(layer int) error {
		if layer != 0 {
			return nil
		}
		compareAt(t, "edit_layer0_out", g.Read(g.TensorX(), cfg.HiddenSize),
			loadEditRef(t, m, "edit_layer0_out"), layerTol)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	compareAt(t, "edit_prompt_embeds", embeds, loadEditRef(t, m, "edit_prompt_embeds"), gpuEditTol)
	if got := countTrue(mask); got != m.PadMaskCount {
		t.Errorf("%d image rows in the mask, want %d", got, m.PadMaskCount)
	}

	// The two mechanisms the seams exist for, each switched off in turn.
	// Both run to completion and return a full-shaped answer, which is why
	// the bound has to be shown to catch them.
	want := loadEditRef(t, m, "edit_prompt_embeds")
	t.Run("controls", func(t *testing.T) {
		plain := qwen.NewRoPE(cfg.HeadDim, len(p.IDs), cfg.RopeTheta)
		out, _, err := p.EncodeGPU(g, plain, one, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		controlAt(t, "plain RoPE instead of mrope", out, want, gpuEditTol)

		bare := []Condition{{Merged: one[0].Merged}}
		out, _, err = p.EncodeGPU(g, rope, bare, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		controlAt(t, "no deepstack injection", out, want, gpuEditTol)
	})

	// --- two condition images, on the encoder that is already staged. The
	// second card is 192x384, so the run lengths, the position spans and the
	// two AddRows dispatches per layer all differ from the first's — and the
	// endpoint promises ten.
	t.Run("multi", func(t *testing.T) {
		mp := multiPrompt(t, m)
		mrope, err := mp.MRope(cfg.HeadDim, cfg.RopeTheta, section)
		if err != nil {
			t.Fatal(err)
		}
		two := []Condition{dumpedCondition(t, m, "vis"), dumpedCondition(t, m, "vis2")}
		embeds, mask, err := mp.EncodeGPU(g, mrope, two, m.DropIdx, nil)
		if err != nil {
			t.Fatal(err)
		}
		compareAt(t, "multi_prompt_embeds", embeds, loadEditRef(t, m, "multi_prompt_embeds"), gpuEditTol)
		if got := countTrue(mask); got != m.MultiPadMaskCount {
			t.Errorf("%d image rows in the mask, want %d", got, m.MultiPadMaskCount)
		}
		refMask := loadEditRef(t, m, "multi_image_pad_mask")
		for i, v := range mask {
			if v != (refMask.Data[i] != 0) {
				t.Fatalf("multi pad mask differs at %d", i)
			}
		}
		// Same rows, same count, wrong order — refused at the row count
		// rather than answered, exactly as the CPU path refuses it.
		if _, _, err := mp.EncodeGPU(g, mrope, []Condition{two[1], two[0]}, m.DropIdx, nil); err == nil {
			t.Error("encoding with the conditions swapped succeeded; the row counts should have refused it")
		}
	})
}

// dumpedCondition reads one condition image's tower output straight out of the
// reference — the merger's rows and the three deepstack features.
func dumpedCondition(t *testing.T, m *editManifest, prefix string) Condition {
	t.Helper()
	c := Condition{Merged: loadEditRef(t, m, prefix+"_merged")}
	for i := 0; i < 3; i++ {
		c.Deepstack = append(c.Deepstack, loadEditRef(t, m, fmt.Sprintf("%s_deepstack%d", prefix, i)))
	}
	return c
}

// controlAt is control against a bound the caller names, which the fp16 path
// needs: a control has to miss the gate *this* path is held to, not the fp32
// one.
func controlAt(t *testing.T, what string, got, want *qwen.Mat, tol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape %s, want %s", what, got, want)
	}
	_, _, rel, _ := deviation(got, want)
	if rel <= tol {
		t.Errorf("%s still matches at rel %.3g, inside the %.0e bound", what, rel, tol)
		return
	}
	t.Logf("control %-30s rel %.3g, %.0fx the bound", what, rel, rel/tol)
}
