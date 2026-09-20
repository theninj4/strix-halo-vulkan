package kokoro

import (
	"strconv"
	"strings"
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

// TestGPUAlbertScaling is the measurement R1 asked for and the one that
// justified the matrix-core port: PL-BERT cost 4 ms at fifty tokens and 75 at
// three hundred and twenty-one, and a layer's cost per token is what says
// which dispatch owned the difference.
//
// It runs both kernels, because the comparison is the finding. It is not a
// correctness test and asserts no absolute time -- a bound in milliseconds
// would fail on a busy machine and tell nobody anything. What it asserts is
// the *shape* of each: the scalar kernel's attention grows per token, because
// its dispatch is one workgroup per (head, query) and each one walks every
// key; the matrix-core kernel's does not, because a flash kernel streams the
// keys through registers at a fixed cost per query tile; and no projection's
// does either way, because a GEMM over T rows is T times the same work.
func TestGPUAlbertScaling(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	model := loadModel(t)
	g, err := NewGPUAlbert(dev, model.BERT, model.BERT.Config.MaxPositionEmbed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	// 50 is the reference utterance, 321 is R1's paragraph, 510 is the
	// model's own ceiling: the voice packs have 510 style rows, so an
	// utterance cannot have more phonemes than that.
	lengths := []int{50, 100, 200, 321, 410, 510}

	type run struct {
		name    string
		layer   []float64 // ms a layer
		attn    []float64 // ms of it in attention (packs included)
		perKind map[string][]float64
		kinds   []string
	}
	var runs []run
	for _, scalar := range []bool{true, false} {
		g.ScalarAttn = scalar
		r := run{name: "wmma", layer: make([]float64, len(lengths)),
			attn: make([]float64, len(lengths)), perKind: map[string][]float64{}}
		if scalar {
			r.name = "scalar"
		}
		for i, n := range lengths {
			stages, err := g.Profile(n)
			if err != nil {
				t.Fatal(err)
			}
			for _, st := range stages {
				if _, seen := r.perKind[st.Kind]; !seen {
					r.kinds = append(r.kinds, st.Kind)
					r.perKind[st.Kind] = make([]float64, len(lengths))
				}
				r.perKind[st.Kind][i] += st.Time * 1000
				r.layer[i] += st.Time * 1000
				// The packs are attention's cost too: they exist only to
				// feed it, so charging them anywhere else would make the
				// comparison flatter than it is.
				if st.Kind == "attention" || strings.HasPrefix(st.Kind, "pack ") {
					r.attn[i] += st.Time * 1000
				}
			}
		}
		runs = append(runs, r)
	}
	g.ScalarAttn = false

	header := func(label string) {
		line, a := "%-16s", []any{label}
		for _, n := range lengths {
			line += " %8d"
			a = append(a, n)
		}
		t.Logf(line, a...)
	}
	row := func(label string, vals []float64, mul float64, prec int) {
		line, a := "%-16s", []any{label}
		for i := range vals {
			line += " %8." + strconv.Itoa(prec) + "f"
			a = append(a, vals[i]*mul)
		}
		t.Logf(line, a...)
	}
	for _, r := range runs {
		header("us a layer, " + r.name)
		for _, k := range r.kinds {
			row("  "+k, r.perKind[k], 1000, 0)
		}
		row("  layer", r.layer, 1000, 0)
		twelve := make([]float64, len(lengths))
		for i := range twelve {
			twelve[i] = r.layer[i] * float64(g.cfg.NumHiddenLayers)
		}
		row("  12 layers, ms", twelve, 1, 1)
		per := make([]float64, len(lengths))
		for i := range per {
			per[i] = r.attn[i] / float64(lengths[i])
		}
		row("  attn, us/tok", per, 1000, 2)
	}
	speedup := make([]float64, len(lengths))
	for i := range speedup {
		speedup[i] = runs[0].layer[i] / runs[1].layer[i]
	}
	header("wmma speedup")
	row("  a layer", speedup, 1, 2)

	// The shape assertions, one per kernel. Attention per token at the
	// ceiling against attention per token at the reference length.
	for _, c := range []struct {
		r     run
		grows bool
	}{{runs[0], true}, {runs[1], false}} {
		last := len(lengths) - 1
		first := c.r.attn[0] / float64(lengths[0])
		end := c.r.attn[last] / float64(lengths[last])
		t.Logf("%s: attention is %.2f us/token at %d and %.2f at %d (%.1fx, sequence %.1fx)",
			c.r.name, first*1000, lengths[0], end*1000, lengths[last],
			end/first, float64(lengths[last])/float64(lengths[0]))
		switch {
		case c.grows && end <= 1.5*first:
			t.Errorf("%s: attention is no longer superlinear in the sequence, "+
				"so this test is measuring a different kernel", c.r.name)
		case !c.grows && end > 1.5*first:
			t.Errorf("%s: attention costs %.2f us/token at %d against %.2f at %d, "+
				"which is the growth the matrix-core port existed to remove",
				c.r.name, end*1000, lengths[last], first*1000, lengths[0])
		}
	}
	// And a projection's does not move either way: `ffn` is the widest GEMM
	// in the layer at N=2048, and T rows of it is T times one row of it.
	for _, r := range runs {
		ff := r.perKind["ffn"]
		last := len(lengths) - 1
		f0, f1 := ff[0]/float64(lengths[0]), ff[last]/float64(lengths[last])
		if f1 > 1.5*f0 {
			t.Errorf("%s: the feed-forward is %.2f us/token at %d and %.2f at %d: "+
				"a GEMM over more rows should cost the same per row",
				r.name, f0*1000, lengths[0], f1*1000, lengths[last])
		}
	}
}
