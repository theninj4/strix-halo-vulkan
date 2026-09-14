package parakeet

import (
	"fmt"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// newTestDecoder stages the transducer tail on the device, sized for the
// reference's own clip.
func newTestDecoder(t *testing.T) (*GPUDecoder, *Model, *manifest, func()) {
	t.Helper()
	m := testModel(t)
	ref := loadManifest(t)
	dev, done := newTestDevice(t)
	d, err := NewGPUDecoder(dev, m, ref.Encoder.Frames)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("%d frames, %d MB of weights, %d MB of arenas, plan %s",
		ref.Encoder.Frames, d.WeightBytes()>>20, d.ActivationBytes()>>20, d.Plan()[DecJoint])
	return d, m, ref, func() { d.Destroy(); done() }
}

// decTol is what the tail is held to per stage, on the same
// rms-of-the-difference measure S6 settled on. The head's logits are the
// widest at 9e-4, which is one fp16 rounding of a 640-deep reduction and
// nothing else; the bound is set an order of magnitude above it.
const decTol = 5e-3

// TestGPUPredictionMatchesReference walks the prediction network on the
// device through the same two steps TestPredictionMatchesReference walks the
// CPU one: the blank start token from a zero state, then a real token from
// the state that leaves behind.
//
// One step would not exercise the recurrence, and it would hide a swapped
// gate order — the blank's embedding row is zero in this checkpoint, so the
// first step's gates come entirely from the bias.
func TestGPUPredictionMatchesReference(t *testing.T) {
	d, m, ref, done := newTestDecoder(t)
	defer done()

	check := func(name string, got []float32) {
		t.Helper()
		want, meta := loadRef(t, ref, name)
		if len(got) != meta.Count {
			t.Fatalf("%s: %d values, reference has %d %v", name, len(got), meta.Count, meta.Shape)
		}
		dv := compare(t, got, want)
		t.Logf("%-10s max abs %.3g, rms %.4g, %.2g relative", name, dv.MaxAbs, dv.RMS, dv.Rel())
		if dv.Rel() > decTol {
			t.Errorf("%s deviates by %.3g relative, more than %g", name, dv.Rel(), decTol)
		}
	}
	state := func(off func(int) uint32) []float32 {
		var flat []float32
		for l := 0; l < d.Layers(); l++ {
			flat = append(flat, d.Read(off(l), 1, d.Hidden()).Data...)
		}
		return flat
	}
	// The prediction projector's bias is not in this tensor: it is folded
	// into parakeet_joint_sum.comp, which already reads all 640 numbers, so
	// the arena holds W*h and the reference holds W*h + b.
	pred := func() []float32 {
		out := d.Read(d.TensorPred(), 1, d.Hidden()).Data
		for i := range out {
			out[i] += m.Prediction.Projector.Bias[i]
		}
		return out
	}

	d.Reset()
	if err := d.Step(m.Config.BlankTokenID); err != nil {
		t.Fatal(err)
	}
	check("dec_out", pred())
	check("dec_h", state(d.TensorH))
	check("dec_c", state(d.TensorC))

	if err := d.Step(1); err != nil {
		t.Fatal(err)
	}
	check("dec_out1", pred())
	check("dec_h1", state(d.TensorH))
	check("dec_c1", state(d.TensorC))
}

// TestGPUJointMatchesReference runs the projector, then the joint at the
// first frame against the first prediction state — the (t=0, u=0) cell of the
// transducer lattice, and the first thing the decode loop evaluates.
func TestGPUJointMatchesReference(t *testing.T) {
	d, m, ref, done := newTestDecoder(t)
	defer done()

	// The projector runs over the *reference's* encoder output, so that what
	// is being compared here is the tail and not the encoder.
	hidden, meta := loadRef(t, ref, "encoder_out")
	frames := meta.Shape[0]
	if err := d.Project(&Mat{Rows: frames, Cols: meta.Shape[1], Data: hidden}); err != nil {
		t.Fatal(err)
	}
	want, _ := loadRef(t, ref, "encoder_projected")
	dv := compare(t, d.Read(d.TensorEnc(), frames, d.Hidden()).Data, want)
	t.Logf("%-14s max abs %.3g, rms %.4g, %.2g relative", "encoder_projected", dv.MaxAbs, dv.RMS, dv.Rel())
	if dv.Rel() > decTol {
		t.Errorf("the projector deviates by %.3g relative, more than %g", dv.Rel(), decTol)
	}

	d.Reset()
	if err := d.Step(m.Config.BlankTokenID); err != nil {
		t.Fatal(err)
	}
	if err := d.Joint(0); err != nil {
		t.Fatal(err)
	}
	wantLogits, lm := loadRef(t, ref, "joint_logits")
	if lm.Count != d.JointOut() {
		t.Fatalf("the head is %d wide, reference has %d", d.JointOut(), lm.Count)
	}
	// The head's bias is folded into parakeet_argmax.comp, which already
	// reads all 8198 logits, so the arena holds W*a and the reference holds
	// W*a + b. The difference is not small -- most of this bias is around
	// -6.3 -- which is why it is worth the two lines rather than a wider
	// tolerance.
	got := d.Read(d.TensorLogits(), 1, d.JointOut()).Data
	for i := range got {
		got[i] += m.Joint.Head.Bias[i]
	}
	dv = compare(t, got, wantLogits)
	t.Logf("%-14s max abs %.3g, rms %.4g, %.2g relative", "joint_logits", dv.MaxAbs, dv.RMS, dv.Rel())
	if dv.Rel() > decTol {
		t.Errorf("the joint deviates by %.3g relative, more than %g", dv.Rel(), decTol)
	}

	// And the decision the loop actually consumes, which is the argmax and
	// not the tensor: the device's two reductions against the reference's
	// first emission.
	res := d.Read(d.aResult, 1, 2).Data
	first := ref.Decode.Trace[0]
	token, duration := int(res[0]), d.durations[int(res[1])]
	if token == d.blank && duration == 0 {
		duration = 1
	}
	if token != first.Token || duration != first.Duration {
		t.Errorf("first step emits (%d, %d), reference has (%d, %d)", token, duration, first.Token, first.Duration)
	}
}

// TestGPUDecodeMatchesReference runs the whole tail over the reference's own
// encoder output, so a difference here is the loop's and not the encoder's.
//
// The assertion is the trace and the string, which is the bound SPEECH.md set
// for this vertical: the transducer's decisions are discrete, so either every
// emission agrees or the port is wrong.
func TestGPUDecodeMatchesReference(t *testing.T) {
	d, _, ref, done := newTestDecoder(t)
	defer done()

	hidden, meta := loadRef(t, ref, "encoder_out")
	got, err := d.Decode(&Mat{Rows: meta.Shape[0], Cols: meta.Shape[1], Data: hidden}, ref.Encoder.ValidFrames)
	if err != nil {
		t.Fatal(err)
	}
	compareTrace(t, ref, got)
	if got.Text != ref.Decode.Text {
		t.Errorf("text\n got %q\nwant %q", got.Text, ref.Decode.Text)
	}
	t.Logf("%d emissions in %d submits: %q", d.Steps, d.Submits, truncate(got.Text, 60))
}

// TestGPUPipeline is the whole vertical on the device: mel in, transcript out,
// with nothing on the host but the front end, the cursor arithmetic and the
// tokenizer.
func TestGPUPipeline(t *testing.T) {
	g, m, mel, validMel, done := newTestMelEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	d, err := NewGPUDecoder(g.dev, m, g.MaxFrames())
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()

	hidden, valid, err := g.ApplyMel(mel, validMel)
	if err != nil {
		t.Fatal(err)
	}
	out, err := d.Decode(hidden, valid)
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != transcript {
		t.Errorf("transcript:\n got %q\nwant %q", out.Text, transcript)
	}
	compareTrace(t, ref, out)
	t.Logf("%d emissions, %d submits, %d MB of decoder weights", d.Steps, d.Submits, d.WeightBytes()>>20)
}

// TestGPUDecodeTiming is the measurement S8 exists for: where a clip's decode
// time goes, split between the device and the round trips around it.
//
// Wall clock is the number that matters here and GPU time is not, which
// inverts every other timing test in this repository. The loop is sequential
// by construction, so what it costs is a fence per emission and not a tile
// choice.
func TestGPUDecodeTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("several decodes of the whole clip")
	}
	d, _, ref, done := newTestDecoder(t)
	defer done()

	hidden, meta := loadRef(t, ref, "encoder_out")
	enc := &Mat{Rows: meta.Shape[0], Cols: meta.Shape[1], Data: hidden}
	if _, err := d.Decode(enc, ref.Encoder.ValidFrames); err != nil {
		t.Fatal(err)
	}
	best := time.Hour
	for i := 0; i < 3; i++ {
		t0 := time.Now()
		if _, err := d.Decode(enc, ref.Encoder.ValidFrames); err != nil {
			t.Fatal(err)
		}
		best = min(best, time.Since(t0))
	}
	t.Logf("%d emissions in %d submits: %v wall, %v an emission",
		d.Steps, d.Submits, best.Round(time.Microsecond), (best / time.Duration(d.Submits)).Round(time.Microsecond))

	// The same emissions timed on the device, dispatch by dispatch, which is
	// what says how much of the wall clock is the round trip.
	d.Reset()
	if err := d.Step(0); err != nil {
		t.Fatal(err)
	}
	pred, predKinds, err := d.predictGraph()
	if err != nil {
		t.Fatal(err)
	}
	joint, jointKinds, err := d.jointGraph(0)
	if err != nil {
		t.Fatal(err)
	}
	dis, kinds := append(pred, joint...), append(predKinds, jointKinds...)
	byKind := map[string]float64{}
	var gpu float64
	for rep := 0; rep < 3; rep++ {
		for i := range dis {
			dur, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 1, true)
			if err != nil {
				t.Fatal(err)
			}
			if rep == 0 || dur.Seconds() < byKind[kinds[i]] {
				byKind[kinds[i]] = dur.Seconds()
			}
		}
	}
	for _, v := range byKind {
		gpu += v
	}
	t.Logf("one full emission is %.1f µs on the device, against %.1f µs of wall clock",
		gpu*1e6, float64(best.Microseconds())/float64(d.Submits))
	for _, k := range d.Labels() {
		fmt.Printf("  %-16s %7.1f µs\n", k, byKind[k]*1e6)
	}
}

// TestGPUDecLadder times one full emission once per projection kernel, which
// is what DefaultDecPlan is read off.
//
// Every shape here is M = 1, so the question the ladder is asking is only how
// much a tile's unused rows cost — and unlike everywhere else in this
// repository the answer is not "nothing", because the weight is streamed once
// either way and the padding is pure arithmetic on top of it.
func TestGPUDecLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("one emission per rung")
	}
	d, _, ref, done := newTestDecoder(t)
	defer done()

	hidden, meta := loadRef(t, ref, "encoder_out")
	if err := d.Project(&Mat{Rows: meta.Shape[0], Cols: meta.Shape[1], Data: hidden}); err != nil {
		t.Fatal(err)
	}
	d.Reset()
	if err := d.Step(0); err != nil {
		t.Fatal(err)
	}
	type row struct {
		kernel GEMMKernel
		gpu    time.Duration
	}
	var rows []row
	for _, k := range GEMMKernels() {
		if err := d.SetPlan(UniformDecPlan(k)); err != nil {
			t.Logf("%-22s unavailable: %v", k, err)
			continue
		}
		pred, _, err := d.predictGraph()
		if err != nil {
			t.Fatal(err)
		}
		joint, _, err := d.jointGraph(0)
		if err != nil {
			t.Fatal(err)
		}
		dis := append(pred, joint...)
		best := time.Hour
		for rep := 0; rep < 3; rep++ {
			var total time.Duration
			for i := range dis {
				dur, err := vk.DispatchMultiTimed(dis[i:i+1], 1, 1, true)
				if err != nil {
					t.Fatal(err)
				}
				total += dur
			}
			best = min(best, total)
		}
		rows = append(rows, row{k, best})
	}
	if len(rows) == 0 {
		t.Fatal("no rung ran")
	}
	best := rows[0].gpu
	for _, r := range rows {
		best = min(best, r.gpu)
	}
	for _, r := range rows {
		fmt.Printf("  %-22s %8.1f µs  %5.2fx\n", r.kernel, float64(r.gpu.Microseconds()), float64(r.gpu)/float64(best))
	}
	if err := d.SetPlan(DefaultDecPlan()); err != nil {
		t.Fatal(err)
	}
}
