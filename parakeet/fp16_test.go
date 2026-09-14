package parakeet

import (
	"testing"

	"strix-halo-vulkan/audio"
)

// TestF16SurvivesTheTranscript is SPEECH.md's open question asked of the
// cheapest thing that can answer it.
//
// The fp16 survey says nothing on the path overflows — the largest activation
// anywhere is 7284 against a limit of 65504 — but "nothing overflows" is not
// "the transcript is unchanged", and the transcript is the bar. fp16's
// resolution at 7000 is 4, and the residual stream after the first feed
// forward is exactly there, so the question is real.
//
// This narrows every matrix-core operand — weights in place, activations on
// the way into each projection — and leaves the norms, the residual stream
// and the front end in float32, which is the shape the Vulkan port will have.
// It then asks for the transcript.
//
// The model is mutated in place, so this test loads its own copy rather than
// the shared one.
func TestF16SurvivesTheTranscript(t *testing.T) {
	if testing.Short() {
		t.Skip("two full CPU runs of a 0.6 B model")
	}
	ref := loadManifest(t)
	m, err := Load(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}

	// fp32 first, from this same copy, so the comparison is against the same
	// arithmetic everywhere else.
	fp32, err := m.Transcribe(clip)
	if err != nil {
		t.Fatal(err)
	}
	if fp32.Text != ref.Decode.Text {
		t.Fatalf("the fp32 baseline already disagrees with the reference:\n got %q\nwant %q", fp32.Text, ref.Decode.Text)
	}

	m.SetF16(true)
	fp16, err := m.Transcribe(clip)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("fp32: %d emissions, %q", len(fp32.Steps), fp32.Text)
	t.Logf("fp16: %d emissions, %q", len(fp16.Steps), fp16.Text)
	if fp16.Text != fp32.Text {
		t.Errorf("fp16 changes the transcript\n got %q\nwant %q", fp16.Text, fp32.Text)
	}

	// The trace is the stricter comparison: the same words can be reached by
	// a different path through the lattice, and a port that changes the path
	// is a port whose timestamps have moved.
	same := len(fp16.Steps) == len(fp32.Steps)
	if same {
		for i := range fp16.Steps {
			if fp16.Steps[i] != fp32.Steps[i] {
				t.Logf("first divergence at step %d: fp16 %+v, fp32 %+v", i, fp16.Steps[i], fp32.Steps[i])
				same = false
				break
			}
		}
	}
	t.Logf("decode trace identical: %v", same)
}

// TestF16PerStageDrift measures where fp16 actually costs something, stage by
// stage, against the fp32 reference dump. It is the survey that tells the
// Vulkan port which tensor to watch, and it is the reason the bound in
// TestF16SurvivesTheTranscript is a transcript rather than a number.
func TestF16PerStageDrift(t *testing.T) {
	if testing.Short() {
		t.Skip("a full CPU run of a 0.6 B model")
	}
	ref := loadManifest(t)
	m, err := Load(modelDir)
	if err != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, err)
	}
	m.SetF16(true)
	mel, valid := features(t, m)

	dumped := map[int]string{0: "layer_out", 1: "hidden_1", 12: "hidden_12", ref.Encoder.Layers - 1: "hidden_23"}
	out, _, err := m.Encoder.forward(mel, valid, func(i int, h *Mat) {
		name, ok := dumped[i]
		if !ok {
			return
		}
		want, _ := loadRef(t, ref, name)
		d := compare(t, h.Data, want)
		t.Logf("layer %-2d %-12s max abs %.3g, rms %.4g (%.2g relative)", i, name, d.MaxAbs, d.RMS, d.MaxAbs/d.RMS)
	})
	if err != nil {
		t.Fatal(err)
	}
	want, _ := loadRef(t, ref, "encoder_out")
	d := compare(t, out.Data, want)
	t.Logf("encoder_out   max abs %.3g, rms %.4g (%.2g relative, against %.2g in fp32)",
		d.MaxAbs, d.RMS, d.MaxAbs/d.RMS, 1.7e-4)

	proj, err := m.Projector.Apply(out)
	if err != nil {
		t.Fatal(err)
	}
	want, _ = loadRef(t, ref, "encoder_projected")
	d = compare(t, proj.Data, want)
	t.Logf("encoder_projected max abs %.3g, rms %.4g (%.2g relative)", d.MaxAbs, d.RMS, d.MaxAbs/d.RMS)
}
