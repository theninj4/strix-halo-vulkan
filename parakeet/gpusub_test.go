package parakeet

import (
	"fmt"
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// newTestMelEncoder builds a GPU encoder sized for the fixture and returns
// the fixture's mel spectrogram, which is what the S7 path takes in.
//
// Unlike newTestEncoder it never runs the CPU subsampling stack, so a failure
// here is a failure of the device's.
func newTestMelEncoder(t *testing.T, ctl controls) (*GPUEncoder, *Model, *Mat, int, func()) {
	t.Helper()
	m := testModel(t)
	dev, done := newTestDevice(t)
	mel, validMel := features(t, m)
	frames := m.Encoder.Subsampling.ValidLength(mel.Rows)
	g, err := newGPUEncoder(dev, m.Encoder, frames, nil, ctl)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("%d mel frames (%d valid) -> %d encoder frames, %d MB of weights, %d MB of arenas, sub plan %s/%s",
		mel.Rows, validMel, frames, g.WeightBytes()>>20, g.ActivationBytes()>>20,
		g.SubPlanOf()[SubPW1], g.SubPlanOf()[SubLinear])
	return g, m, mel, validMel, func() { g.Destroy(); done() }
}

// chanFirst rewrites a [T, F, C] feature map -- the layout the device holds
// the subsampling stack in -- as the [C, T, F] the reference dumps.
//
// It is the whole difference between the two implementations expressed as
// twelve lines of host code, which is why it is here and not a kernel: the
// permutation costs a test one pass over a tensor and would cost the graph
// four.
func chanFirst(src []float32, tf, c int) []float32 {
	out := make([]float32, tf*c)
	for p := 0; p < tf; p++ {
		row := src[p*c : (p+1)*c]
		for ch, v := range row {
			out[ch*tf+p] = v
		}
	}
	return out
}

// TestGPUSubsampling walks the five convolutions and the linear on the device
// against the same reference volumes TestSubsamplingMatchesReference holds the
// CPU implementation to.
//
// No two stages share a buffer here -- unlike a conformer layer, whose four
// branches write one tensor -- so one run leaves every intermediate in the
// arena and the walk costs nothing beyond the reads.
func TestGPUSubsampling(t *testing.T) {
	g, _, mel, validMel, done := newTestMelEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	if err := g.UploadMel(mel, validMel); err != nil {
		t.Fatal(err)
	}
	if err := g.RunSub(); err != nil {
		t.Fatal(err)
	}
	if g.Rows() != ref.Encoder.Frames || g.Valid() != ref.Encoder.ValidFrames {
		t.Fatalf("%d frames (%d valid), reference has %d (%d valid)",
			g.Rows(), g.Valid(), ref.Encoder.Frames, ref.Encoder.ValidFrames)
	}

	c := g.SubChans()
	check := func(name string, stage int, got func(tf int) []float32) {
		t.Helper()
		tt, f, _ := g.SubShape(stage)
		want, meta := loadRef(t, ref, name)
		if len(meta.Shape) != 3 || meta.Shape[0] != c || meta.Shape[1] != tt || meta.Shape[2] != f {
			t.Fatalf("%s: the device has [%d %d %d], reference has %v", name, c, tt, f, meta.Shape)
		}
		d := compare(t, chanFirst(got(tt*f), tt*f, c), want)
		t.Logf("%-8s [%d %d %d]  max abs %.3g, rms %.4g, %.2g relative", name, c, tt, f, d.MaxAbs, d.RMS, d.Rel())
		if d.Rel() > gpuTol {
			t.Errorf("%s deviates by %.3g relative, more than %g", name, d.Rel(), gpuTol)
		}
	}
	fp32 := func(off uint32) func(tf int) []float32 {
		return func(tf int) []float32 { return g.ReadRows(off, tf, c).Data }
	}
	// The two depthwise convolutions write their pointwise partner's fp16 A
	// operand directly, so that is where they are read from: there is no fp32
	// copy of either anywhere in this graph.
	fp16 := func(off uint32) func(tf int) []float32 {
		return func(tf int) []float32 { return g.ReadF16(off, tf, c, g.LDASub()).Data }
	}
	check("sub_0", 0, fp32(g.TensorSubConv0()))
	check("sub_2", 1, fp16(g.TensorSubDW1()))
	check("sub_3", 1, fp32(g.TensorSubPW1()))
	check("sub_5", 2, fp16(g.TensorSubDW2()))
	check("sub_6", 2, fp32(g.TensorSubPW2()))

	// The residual stream the layers start from, which is the linear's output
	// times the encoder's scale_input -- folded into the same epilogue that
	// adds the bias, so the reference is scaled rather than the tensor.
	want, _ := loadRef(t, ref, "subsampled")
	scale := float32(1)
	if g.scaleInput {
		scale = float32(math.Sqrt(float64(g.Dim())))
	}
	scaled := make([]float32, len(want))
	for i, v := range want {
		scaled[i] = v * scale
	}
	d := compare(t, g.Read(g.TensorX(), g.Dim()).Data, scaled)
	t.Logf("%-8s [%d %d]  max abs %.3g, rms %.4g, %.2g relative", "subsampled", g.Rows(), g.Dim(), d.MaxAbs, d.RMS, d.Rel())
	if d.Rel() > gpuTol {
		t.Errorf("subsampled deviates by %.3g relative, more than %g", d.Rel(), gpuTol)
	}
}

// TestGPUMelTranscript is the end-to-end assertion for S7: the whole encoder
// on the device, mel in and hidden states out, decoded to the same words at
// the same frames with the same durations as the reference.
//
// It is the bound the port is held to, for the reason S6 established: fp16
// operands move the encoder output by percent and the transducer does not
// care, so a tensor bound cannot certify the path and the trace can.
func TestGPUMelTranscript(t *testing.T) {
	g, m, mel, validMel, done := newTestMelEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	hidden, valid, err := g.ApplyMel(mel, validMel)
	if err != nil {
		t.Fatal(err)
	}
	out, err := decodeHidden(m, hidden, valid)
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != transcript {
		t.Errorf("transcript:\n got %q\nwant %q", out.Text, transcript)
	}
	if len(out.Steps) != len(ref.Decode.Trace) {
		t.Fatalf("%d emissions, reference has %d", len(out.Steps), len(ref.Decode.Trace))
	}
	for i, s := range out.Steps {
		w := ref.Decode.Trace[i]
		if s.Frame != w.T || s.Token != w.Token || s.Duration != w.Duration {
			t.Fatalf("step %d: frame %d token %d +%d, reference has frame %d token %d +%d",
				i, s.Frame, s.Token, s.Duration, w.T, w.Token, w.Duration)
		}
	}

	// And the same encoder output as the host stack produces from the same
	// mel, which is the comparison that isolates S7 from S6: both paths run
	// the identical 24 layers, so what is left is the convolutions.
	g.HostSubsampling = true
	host, _, err := g.ApplyMel(mel, validMel)
	if err != nil {
		t.Fatal(err)
	}
	g.HostSubsampling = false
	d := compare(t, hidden.Data[:valid*hidden.Cols], host.Data[:valid*host.Cols])
	t.Logf("%d emissions, identical frames and durations; %.3g relative to the host stack", len(out.Steps), d.Rel())
	if d.Rel() > gpuOutTol {
		t.Errorf("the device's subsampling moves the encoder output by %.3g relative", d.Rel())
	}
}

// TestGPUSubDetectsErrors is S7's negative control: two things the stack does
// that a plausible-looking one would not.
//
// The flatten order is the one worth the test. The channel-last feature map
// this stage is built on makes the linear's A operand read f*C + c, where
// PyTorch's flatten produces c*F + f; the weight is permuted once at upload
// to match. Skipping that permutation is not an error anywhere -- the shapes
// agree, the GEMM runs, and a [138, 1024] tensor of ordinary magnitude comes
// out. Only the words change.
func TestGPUSubDetectsErrors(t *testing.T) {
	if testing.Short() {
		t.Skip("two more runs of the whole encoder")
	}
	m := testModel(t)
	dev, done := newTestDevice(t)
	defer done()
	mel, validMel := features(t, m)

	// Eight mel frames of padding past the clip's valid length, so that the
	// masking control has something to leak: the fixture is 1100 valid frames
	// of 1101 and three stride-2 convolutions turn that into 138 of 138.
	padded := NewMat(mel.Rows+64, mel.Cols)
	copy(padded.Data, mel.Data)
	for r := mel.Rows; r < padded.Rows; r++ {
		copy(padded.Row(r), mel.Row(mel.Rows-1))
	}
	frames := m.Encoder.Subsampling.ValidLength(padded.Rows)
	run := func(ctl controls) (*Mat, int, *Transcript) {
		t.Helper()
		g, err := newGPUEncoder(dev, m.Encoder, frames, nil, ctl)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		hidden, valid, err := g.ApplyMel(padded, validMel)
		if err != nil {
			t.Fatal(err)
		}
		out, err := decodeHidden(m, hidden, valid)
		if err != nil {
			t.Fatal(err)
		}
		return hidden, valid, out
	}
	base, valid, baseOut := run(controls{})
	if baseOut.Text != transcript {
		t.Fatalf("the padded clip transcribes as %q", truncate(baseOut.Text, 72))
	}
	live := func(h *Mat) []float32 { return h.Data[:valid*h.Cols] }
	intact := compare(t, live(base), mustRef(t, "encoder_out")).Rel()
	t.Logf("intact, padded clip: %.3g relative to the reference", intact)
	if intact > gpuOutTol {
		t.Fatalf("the padded clip's valid frames deviate by %.3g relative", intact)
	}

	for _, c := range []struct {
		name string
		ctl  controls
		what string
	}{
		{"no mask", controls{subNoMask: true}, "padding survives all three stride-2 convolutions"},
		{"flatten order", controls{subFlatOrder: true}, "the linear reads the feature axis channel-slower"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, _, out := run(c.ctl)
			d := compare(t, live(got), live(base))
			changed := out.Text != baseOut.Text
			t.Logf("%-14s %.3g relative, transcript %s: %q",
				c.name, d.Rel(), map[bool]string{true: "changed", false: "unchanged"}[changed],
				truncate(out.Text, 60))
			if d.Rel() < 10*intact {
				t.Errorf("%s (%s) moved the encoder output by only %.3g relative, which is inside 10x the intact %.3g",
					c.name, c.what, d.Rel(), intact)
			}
		})
	}
}

// TestGPUSubLadder times the subsampling stack once per projection kernel,
// which is what DefaultSubPlan is read off.
//
// The stack's two GEMM shapes are nothing like the conformer's and nothing
// like each other: the pointwise convolutions are [8832, 256] x [256, 256] --
// the one place in this model with enough rows to fill a wide four-wave tile
// -- and the linear is [138, 4096] x [4096, 1024], which is the narrow-M
// regime PlanFor exists for. So they are swept together and allowed to
// disagree.
func TestGPUSubLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("one run of the subsampling stack per rung")
	}
	g, _, mel, validMel, done := newTestMelEncoder(t, controls{})
	defer done()

	if err := g.UploadMel(mel, validMel); err != nil {
		t.Fatal(err)
	}
	// One untimed run: the first dispatch of a pipeline pays for its own
	// compilation and the first read of a weight comes off DRAM cold.
	if err := g.RunSub(); err != nil {
		t.Fatal(err)
	}
	type row struct {
		kernel GEMMKernel
		pw, ln time.Duration
	}
	var rows []row
	for _, k := range GEMMKernels() {
		if err := g.SetSubPlan(UniformSubPlan(k)); err != nil {
			t.Logf("%-22s unavailable: %v", k, err)
			continue
		}
		d, kinds, err := g.subGraph()
		if err != nil {
			t.Logf("%-22s %v", k, err)
			continue
		}
		var r row
		r.kernel = k
		for rep := 0; rep < 3; rep++ {
			var pw, ln time.Duration
			for i := range d {
				dur, err := vk.DispatchMultiTimed(d[i:i+1], 1, 1, true)
				if err != nil {
					t.Fatal(err)
				}
				switch kinds[i] {
				case "gemm " + string(SubPW1), "gemm " + string(SubPW2):
					pw += dur
				case "gemm " + string(SubLinear):
					ln += dur
				}
			}
			if rep == 0 || pw < r.pw {
				r.pw = pw
			}
			if rep == 0 || ln < r.ln {
				r.ln = ln
			}
		}
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		t.Fatal("no rung ran")
	}
	best := rows[0]
	for _, r := range rows {
		if r.pw < best.pw {
			best.pw = r.pw
		}
		if r.ln < best.ln {
			best.ln = r.ln
		}
	}
	fmt.Printf("  %-22s %12s %8s   %12s %8s\n", "kernel", "pointwise", "", "linear", "")
	for _, r := range rows {
		fmt.Printf("  %-22s %10.3f ms %6.2fx   %10.3f ms %6.2fx\n",
			r.kernel, float64(r.pw.Microseconds())/1e3, float64(r.pw)/float64(best.pw),
			float64(r.ln.Microseconds())/1e3, float64(r.ln)/float64(best.ln))
	}
	if err := g.SetSubPlan(DefaultSubPlan()); err != nil {
		t.Fatal(err)
	}
}

// TestGPUSubProfile times every dispatch of the subsampling stack, which is
// what says whether S7 closed the gap the S6 profile opened.
func TestGPUSubProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("one profiled run of the whole encoder")
	}
	g, _, mel, validMel, done := newTestMelEncoder(t, controls{})
	defer done()

	if _, _, err := g.ApplyMel(mel, validMel); err != nil {
		t.Fatal(err)
	}
	stages, _, err := g.ProfileMel(mel, validMel)
	if err != nil {
		t.Fatal(err)
	}
	var sub, layers float64
	byKind := map[string]float64{}
	for _, s := range stages {
		if s.Layer < 0 {
			sub += s.GPU.Seconds()
			byKind[s.Kind] += s.GPU.Seconds()
		} else {
			layers += s.GPU.Seconds()
		}
	}
	t.Logf("subsampling %.3f ms for %.2f GFLOP (%.0f GFLOP/s), layers %.3f ms, total %.3f ms",
		sub*1e3, g.SubFLOPs()/1e9, g.SubFLOPs()/sub/1e9, layers*1e3, (sub+layers)*1e3)
	for _, k := range g.SubLabels() {
		fmt.Printf("  %-18s %8.3f ms  %4.1f%%\n", k, byKind[k]*1e3, 100*byKind[k]/(sub+layers))
	}
}
