package kokoro

import (
	"testing"
	"time"
)

// bertTol is what the fp16 matrix-core path is held to against the CPU
// reference, as an rms difference over the tensor's own rms.
//
// Twelve layers deep with a post-LayerNorm after each half, so the error does
// not compound the way it would down a plain stack — the normalisation
// renormalises the scale every time. The bound is the measured worst times
// three.
const bertTol = 6e-3

// TestGPUAlbert runs PL-BERT on the device against the CPU reference, layer
// by layer.
//
// Layer by layer rather than end to end because the twelve layers share one
// weight group: a staging bug would look identical in all of them, and the
// per-layer drift is what says whether the fp16 error compounds or the
// post-norms hold it.
func TestGPUAlbert(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)

	ids := m.InputIDs
	g, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	_, hidden, err := model.BERT.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.BERT.Apply(ids)
	if err != nil {
		t.Fatal(err)
	}

	// One layer at a time, feeding the device its own output, so the drift
	// reported is cumulative exactly as the real graph's is.
	g.tokens = hidden.Rows
	g.abuf.WriteFloat32At(int(g.aX), hidden.Data)
	one, _ := g.graph()
	for i := range want {
		if err := submit(one); err != nil {
			t.Fatal(err)
		}
		got := &Mat{Rows: g.tokens, Cols: g.cfg.HiddenSize,
			Data: g.abuf.ReadFloat32At(int(g.aX), g.tokens*g.cfg.HiddenSize)}
		d := compare(t, got.Data, want[i].Data)
		if d.Rel() > bertTol {
			t.Errorf("layer %d: relative %.3g (max abs %.3g at %d)", i, d.Rel(), d.MaxAbs, d.At)
		} else if i == 0 || i == len(want)-1 {
			t.Logf("layer %2d: relative %.3g", i, d.Rel())
		}
	}
}

// TestGPUAlbertApply runs the whole encoder the way the pipeline does — one
// submit for twelve layers — and times it against the CPU.
func TestGPUAlbertApply(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	ids := m.InputIDs

	t0 := time.Now()
	want, err := model.BERT.Apply(ids)
	if err != nil {
		t.Fatal(err)
	}
	cpu := time.Since(t0)

	g, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	_, hidden, err := model.BERT.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Apply(hidden); err != nil { // warm
		t.Fatal(err)
	}
	t0 = time.Now()
	got, err := g.Apply(hidden)
	if err != nil {
		t.Fatal(err)
	}
	gpu := time.Since(t0)

	d := compare(t, got.Data, want[len(want)-1].Data)
	if d.Rel() > bertTol {
		t.Errorf("relative %.3g against the CPU path", d.Rel())
	}
	t.Logf("%d tokens: %.1f ms on the CPU, %.1f ms on the device — %.1fx; relative %.3g",
		len(ids), float64(cpu.Microseconds())/1000, float64(gpu.Microseconds())/1000,
		float64(cpu)/float64(gpu), d.Rel())
}

// TestGPUAlbertProfile prints where a layer's time goes, on GPU timestamps.
func TestGPUAlbertProfile(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	g, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	stages, err := g.Profile(len(m.InputIDs))
	if err != nil {
		t.Fatal(err)
	}
	var total float64
	for _, s := range stages {
		total += s.Time
		t.Logf("  %-16s %7.3f ms", s.Kind, s.Time*1000)
	}
	t.Logf("%-16s %7.3f ms a layer, %7.3f ms for twelve", "total", total*1000,
		total*1000*float64(g.cfg.NumHiddenLayers))
}

// TestGPUAlbertDurations is the test that matters more than the tensor
// comparison: whether the fp16 path changes any *duration*.
//
// The durations are integers — a sum of fifty sigmoids, rounded — and they
// decide the length of every phoneme and therefore the alignment of the whole
// waveform. A 2e-3 drift in ALBERT's output is nothing in the hidden states
// and everything if it flips one rounding, because the audio after that point
// is shifted and no sample-wise comparison of the two waveforms means
// anything afterwards. So the durations are compared as integers, not as a
// tolerance.
func TestGPUAlbertDurations(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	ids := m.InputIDs

	host, err := model.Prosody(ids, predStyle, 1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	model.BERTGPU = g
	defer func() { model.BERTGPU = nil }()

	got, err := model.Prosody(ids, predStyle, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Frames != host.Frames {
		t.Errorf("device gives %d frames, the host gives %d", got.Frames, host.Frames)
	}
	diff := 0
	for i := range host.Durations {
		if i < len(got.Durations) && got.Durations[i] != host.Durations[i] {
			diff++
			if diff <= 5 {
				t.Errorf("token %d (%d): %d frames on the device, %d on the host",
					i, ids[i], got.Durations[i], host.Durations[i])
			}
		}
	}
	// The unrounded durations say how close the flips were, which is what
	// decides whether this is a tolerance question or a correctness one.
	var maxRaw float64
	for i := range host.RawDur {
		if i < len(got.RawDur) {
			d := float64(got.RawDur[i] - host.RawDur[i])
			if d < 0 {
				d = -d
			}
			if d > maxRaw {
				maxRaw = d
			}
		}
	}
	t.Logf("%d of %d durations differ; largest unrounded drift %.4g frames",
		diff, len(host.Durations), maxRaw)
	d := compare(t, got.F0, host.F0)
	t.Logf("F0 curve relative %.3g", d.Rel())
	// That last number is small and its consequence is not, which is worth
	// stating rather than discovering later. The F0 curve drives the
	// excitation's phase accumulator, and by the end of a three-second
	// utterance that holds 1.3e5 radians (T3) — so 2e-4 of relative drift is
	// tens of radians of phase, and the two waveforms decorrelate at the
	// sample level while staying identical in alignment, spectrum and sound.
	// A waveform comparison between the two paths measures that, not this.
}
