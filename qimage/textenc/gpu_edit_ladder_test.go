package textenc

import (
	"fmt"
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// TestGPUEditLadder is TestGPULadder for the edit path: it walks the device's
// edit encoding layer by layer against the CPU port of the same encoding,
// which qimage/textenc's TestEditEncoder pins to the reference at 4.3e-4.
//
// It exists because the edit gate's fp16 deviation (0.059) is twice the t2i
// prompt's (0.024) on the same 36 layers, and the question a bound cannot
// answer is *where* the difference is born: a structural mistake in the
// scatter, the mrope table or the deepstack injection would show as a step,
// and this checkpoint's known fp16 mechanism — a massive-activation channel
// grown at layer 6 and cancelled in layers 34-35 — shows as a flat carry and
// a jump at the end.
//
// It also re-measures the t2i "en" prompt through the same staged encoder, so
// the comparison is same-hour rather than against a number in a file.
//
// ~50 GB (34 fp32 + 15 fp16); it must run alone on the machine.
func TestGPUEditLadder(t *testing.T) {
	if os.Getenv("QI21_LADDER") == "" {
		t.Skip("diagnostic; set QI21_LADDER=1 (loads ~50 GB and must run alone)")
	}
	m := loadEditManifest(t)
	tm := loadManifest(t)
	cfg, err := LoadConfig(encoder)
	if err != nil {
		t.Skipf("no text encoder checkpoint at %s (%v)", encoder, err)
	}
	section, err := LoadMRope(encoder)
	if err != nil {
		t.Fatal(err)
	}
	p := editPrompt(t, m)
	one := []Condition{dumpedCondition(t, m, "vis")}
	rope, err := p.MRope(cfg.HeadDim, cfg.RopeTheta, section)
	if err != nil {
		t.Fatal(err)
	}

	// The CPU side first, with its own trace.
	cpu, err := qwen.LoadWith(encoder, cfg, cfg.NumLayers)
	if err != nil {
		t.Fatal(err)
	}
	tr := qwen.Trace{}
	cpuEmbeds, _, err := p.Encode(cpu, rope, one, m.DropIdx, tr)
	if err != nil {
		t.Fatal(err)
	}
	compareAt(t, "cpu edit_prompt_embeds", cpuEmbeds, loadEditRef(t, m, "edit_prompt_embeds"), deepTol)

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

	// The same-hour t2i reading, for scale.
	enOut, err := g.Forward(tm.Prompts["en"].IDs)
	if err != nil {
		t.Fatal(err)
	}
	enEmbeds, err := Drop(enOut, tm.DropIdx)
	if err != nil {
		t.Fatal(err)
	}
	_, enRMS, enRel, _ := deviation(enEmbeds, loadRef(t, tm, "en_prompt_embeds"))
	t.Logf("t2i en prompt (%d rows): rel %.3g, rms %.4g", enEmbeds.Rows, enRel, enRMS)

	gpuEmbeds, _, err := p.EncodeGPU(g, rope, one, m.DropIdx, func(layer int) error {
		got := g.Read(g.TensorX(), cfg.HiddenSize)
		want := tr[fmt.Sprintf("hidden_%d", layer+1)]
		maxAbs, rms, rel, worst := deviation(got, want)
		// The massive-activation channel is a *column*: report which one the
		// worst element sits in, so a carried channel is visible as a
		// constant and a spread as noise.
		t.Logf("layer %2d: max abs %8.4f  rms %9.3f  rel %.3g  worst row %d col %d",
			layer, maxAbs, rms, rel, worst/cfg.HiddenSize, worst%cfg.HiddenSize)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	_, rms, rel, worst := deviation(gpuEmbeds, cpuEmbeds)
	t.Logf("edit prompt_embeds vs the CPU port: rel %.3g, rms %.4g, worst row %d col %d",
		rel, rms, worst/cfg.HiddenSize, worst%cfg.HiddenSize)
}
