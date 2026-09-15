package kokoro

import (
	"testing"
	"time"
)

// lstmTol is what a recurrence's device path is held to as an rms difference
// against the reference dump over the tensor's own rms.
//
// Looser than the AdaIN blocks' 6e-3 would need to be and tighter than it is,
// for a reason worth stating: an LSTM's error does not average out along the
// sequence the way a convolution stack's does. Every step's fp16 product
// feeds the next step's state through a sigmoid and a tanh, so the question a
// bound answers here is whether the *recurrence* is stable, not whether one
// dot product is accurate. At 130 steps it measures a few parts in ten
// thousand, which says it is.
const lstmTol = 3e-3

// TestGPULSTM runs the prosody predictor's `shared` recurrence on the device
// against the reference dump.
//
// `shared` is the longest of the six — 130 frames rather than fifty tokens —
// so it is the one where a state that decayed slightly wrong would show. It
// is also the one whose input the dump records on both sides, which makes it
// the only one checkable in isolation without replaying the chain above it.
func TestGPULSTM(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	en := refMat(t, m, "en", true)

	g, err := NewGPULSTM(dev, model.Predictor.Shared, en.Rows)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	got, err := g.Apply(en)
	if err != nil {
		t.Fatal(err)
	}
	want := refMat(t, m, "shared_out", false)
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("device gives %v, reference has [%d %d]", got, want.Rows, want.Cols)
	}
	dv := compare(t, got.Data, want.Data)
	if dv.Rel() > lstmTol {
		t.Errorf("shared: relative %.3g (max abs %.3g at %d, rms %.3g)",
			dv.Rel(), dv.MaxAbs, dv.At, dv.RMS)
	} else {
		t.Logf("shared against the dump: relative %.3g", dv.Rel())
	}

	// And against the CPU path, which is the stricter comparison: the dump
	// and the CPU agree to 1e-5, so anything the device gets wrong shows up
	// here at full size rather than against a reference the host also misses.
	host, err := model.Predictor.Shared.Apply(en)
	if err != nil {
		t.Fatal(err)
	}
	dv = compare(t, got.Data, host.Data)
	if dv.Rel() > lstmTol {
		t.Errorf("shared against the CPU: relative %.3g", dv.Rel())
	} else {
		t.Logf("shared against the CPU: relative %.3g", dv.Rel())
	}
}

// TestGPULSTMAll runs all six of the model's recurrences on the device over
// the sequences they actually see, and times them against the host.
//
// All six against one, because they are the same object at three input widths
// and two sequence lengths — and because the one that is wrong would be the
// one whose weights were staged from the wrong direction, which a single
// check of `shared` could not catch.
func TestGPULSTMAll(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	p := model.Predictor

	en := refMat(t, m, "en", true)
	dur := refMat(t, m, "dur_encoded", false)
	cases := []struct {
		name string
		l    *LSTM
		x    *Mat
	}{
		{"dur 0", p.TextEncoder.LSTMs[0], refMat(t, m, "dur_norm_0", true)},
		{"dur 1", p.TextEncoder.LSTMs[1], refMat(t, m, "dur_norm_0", true)},
		{"dur 2", p.TextEncoder.LSTMs[2], refMat(t, m, "dur_norm_1", true)},
		{"duration head", p.LSTM, dur},
		{"shared", p.Shared, en},
		{"text encoder", model.TextEncoder.LSTM, nil},
	}
	// The text encoder's LSTM takes the convolution stack's output, which the
	// dump records channel-first and at the one input width that is not 640.
	cases[len(cases)-1].x = refMat(t, m, "te_cnn_2", true)

	var gpu, cpu time.Duration
	for _, c := range cases {
		if c.x.Cols != c.l.In {
			t.Fatalf("%s: input is %d wide, the LSTM takes %d", c.name, c.x.Cols, c.l.In)
		}
		g, err := NewGPULSTM(dev, c.l, c.x.Rows)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := g.Apply(c.x); err != nil { // warm
			g.Destroy()
			t.Fatal(err)
		}
		t0 := time.Now()
		got, err := g.Apply(c.x)
		if err != nil {
			g.Destroy()
			t.Fatal(err)
		}
		dg := time.Since(t0)
		g.Destroy()

		t0 = time.Now()
		want, err := c.l.Apply(c.x)
		if err != nil {
			t.Fatal(err)
		}
		dc := time.Since(t0)
		gpu, cpu = gpu+dg, cpu+dc

		dv := compare(t, got.Data, want.Data)
		if dv.Rel() > lstmTol {
			t.Errorf("%s: relative %.3g (max abs %.3g at %d)", c.name, dv.Rel(), dv.MaxAbs, dv.At)
		}
		t.Logf("%-14s %3d steps, %4d in    relative %.3g    %6.2f ms against %6.2f on the host",
			c.name, c.x.Rows, c.l.In, dv.Rel(),
			float64(dg.Microseconds())/1000, float64(dc.Microseconds())/1000)
	}
	t.Logf("%-14s                          six recurrences  %6.2f ms against %6.2f, %.1fx",
		"total", float64(gpu.Microseconds())/1000, float64(cpu.Microseconds())/1000,
		float64(cpu)/float64(gpu))
}

// TestGPULSTMProfile prints what the three kinds of dispatch cost: the narrow,
// the input projection lifted out of the loop, and one timestep.
//
// The step is the number that matters. Everything else about this stage is
// fixed cost paid twice per LSTM, and the step is paid 330 times across the
// model.
func TestGPULSTMProfile(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	model := loadModel(t)

	for _, c := range []struct {
		name string
		l    *LSTM
		t    int
	}{
		{"tokens", model.Predictor.LSTM, 50},
		{"frames", model.Predictor.Shared, 130},
	} {
		g, err := NewGPULSTM(dev, c.l, c.t)
		if err != nil {
			t.Fatal(err)
		}
		stages, err := g.Profile(c.t)
		if err != nil {
			g.Destroy()
			t.Fatal(err)
		}
		var step float64
		for _, s := range stages {
			t.Logf("%-8s %-10s %7.4f ms", c.name, s.Kind, s.Time*1000)
			if s.Kind == "step" {
				step = s.Time
			}
		}
		t.Logf("%-8s %-10s %7.4f ms over %d steps", c.name, "loop", step*float64(c.t)*1000, c.t)
		g.Destroy()
	}
}

// TestGPULSTMLadder measures the three builds of the step kernel over the two
// sequence lengths this model actually runs.
//
// The rung is a trade rather than a tuning knob and it is not obvious which
// way it goes. A step is 0.5 MB of weights against 0.26 MFLOP — 500 bytes a
// FLOP — so more workgroups means fewer bytes through each CU, which should
// win; against that, a workgroup of 128 threads has four waves to hide its
// own memory latency where one of 512 has sixteen, and the whole kernel is a
// chain of dependent loads. Measured over the loop, not over one dispatch: a
// step on the critical path of three hundred barriers is not the same thing
// as a step measured alone.
func TestGPULSTMLadder(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	model := loadModel(t)

	for _, c := range []struct {
		name string
		l    *LSTM
		n    int
	}{
		{"50 tokens", model.Predictor.LSTM, 50},
		{"130 frames", model.Predictor.Shared, 130},
	} {
		g, err := NewGPULSTM(dev, c.l, c.n)
		if err != nil {
			t.Fatal(err)
		}
		g.tokens = c.n
		g.Reset()
		for _, k := range StepKernels() {
			if err := g.SetStepKernel(k); err != nil {
				t.Logf("%-12s %-10s unavailable: %v", c.name, k, err)
				continue
			}
			d, err := g.LoopTime(8)
			if err != nil {
				g.Destroy()
				t.Fatal(err)
			}
			t.Logf("%-12s %-10s %7.3f ms   %6.1f us a step", c.name, k, d*1000,
				d*1e6/float64(c.n))
		}
		g.Destroy()
	}
}

// TestGPULSTMTail checks the trick that removes the style concatenation: five
// of the six recurrences take `[h | style]`, and the style half is the same in
// every row of every utterance, so it belongs in the operand rather than in
// the data.
//
// The check is against the CPU path over the *concatenated* input, because
// that is the thing the trick claims to be identical to — not against a
// reference that only ever saw one of the two arrangements.
func TestGPULSTMTail(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	m := loadManifest(t)
	model := loadModel(t)
	_, predStyle, err := model.Style(m.Voice, len([]rune(m.Phonemes)))
	if err != nil {
		t.Fatal(err)
	}
	l := model.Predictor.Shared
	en := refMat(t, m, "en", true)
	if en.Cols != l.In {
		t.Fatalf("en is %d wide, the recurrence takes %d", en.Cols, l.In)
	}
	// The last StyleDim columns of `en` are the broadcast style, which is
	// what makes them removable; check that before relying on it.
	for r := 0; r < en.Rows; r++ {
		for c, v := range predStyle {
			if got := en.Row(r)[l.In-len(predStyle)+c]; got != v {
				t.Fatalf("row %d column %d is %g, the style says %g", r, c, got, v)
			}
		}
	}

	want, err := l.Apply(en)
	if err != nil {
		t.Fatal(err)
	}

	g, err := NewGPULSTM(dev, l, en.Rows)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if err := g.SetTail(predStyle); err != nil {
		t.Fatal(err)
	}
	head := NewMat(en.Rows, l.In-len(predStyle))
	for r := 0; r < en.Rows; r++ {
		copy(head.Row(r), en.Row(r))
	}
	got, err := g.Apply(head)
	if err != nil {
		t.Fatal(err)
	}
	dv := compare(t, got.Data, want.Data)
	if dv.Rel() > lstmTol {
		t.Errorf("tail: relative %.3g (max abs %.3g at %d)", dv.Rel(), dv.MaxAbs, dv.At)
	} else {
		t.Logf("%d channels of style out of the data: relative %.3g", len(predStyle), dv.Rel())
	}
}
