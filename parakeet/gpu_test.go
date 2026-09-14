package parakeet

import (
	"fmt"
	"math"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("parakeet-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// gpuTol is what the matrix-core path is held to per stage: the rms of the
// difference against the reference, as a fraction of the tensor's own rms
// (deviation.Rel).
//
// It is the *measured* worst stage times three. Every dispatch of layer 0
// lands between 3e-5 and 9.5e-4 relative, the widest being the attention
// context -- the one tensor whose inputs have been through a softmax, so a
// score's fp16 error is exponentiated before it reaches the output.
//
// The metric matters as much as the number. A max-abs bound against the rms,
// which is what the CPU stages use, reads 2-3% on tensors that are correct
// here: fp16 rounds every element by its own magnitude and the largest
// element of these tensors is 30x their rms, so that ratio measures the
// dynamic range and not the error. SPEECH.md's fp16 survey is the warning --
// it found 4.3% by exactly that metric at the encoder output, with the
// transcript and the whole decode trace unchanged.
const gpuTol = 3e-3

// gpuOutTol is the same measure at the end of the 24-layer chain, where the
// drift has accumulated: 2.1e-3 measured, and the growth is mostly the last
// layer's norm dividing by a small variance (its rms is 0.02 against the
// stack's 11).
const gpuOutTol = 1e-2

// gpuInput runs the front end and the subsampling stack on the CPU and
// returns the encoder's input -- the residual stream every layer starts from,
// scaled as Encoder.forward scales it.
//
// The subsampling is deliberately still on the CPU: it is 2.8 GFLOP of the
// encoder's 180 and it runs once per clip, so it is a correctness problem
// rather than a performance one (SPEECH.md, S6's three gaps), and keeping it
// here means a failure in a GPU test is a failure of the layer graph.
func gpuInput(t *testing.T, m *Model) (*Mat, int) {
	t.Helper()
	mel, valid := features(t, m)
	x, encValid, err := m.Encoder.Subsampling.Apply(mel, valid)
	if err != nil {
		t.Fatal(err)
	}
	if m.Encoder.Config.ScaleInput {
		s := float32(math.Sqrt(float64(m.Encoder.Config.HiddenSize)))
		for i := range x.Data {
			x.Data[i] *= s
		}
	}
	return x, encValid
}

func newTestEncoder(t *testing.T, ctl controls) (*GPUEncoder, *Model, *Mat, int, func()) {
	t.Helper()
	m := testModel(t)
	dev, done := newTestDevice(t)
	x, valid := gpuInput(t, m)
	g, err := newGPUEncoder(dev, m.Encoder, x.Rows, nil, ctl)
	if err != nil {
		done()
		t.Fatal(err)
	}
	t.Logf("%d frames (%d valid), %d layers, %d MB of weights, %d MB of arenas, attention %s, plan %s",
		x.Rows, valid, g.Layers(), g.WeightBytes()>>20, g.ActivationBytes()>>20, g.Attention(), g.Plan()[ProjQ])
	return g, m, x, valid, func() { g.Destroy(); done() }
}

// transposed reinterprets a [cols, rows] reference tensor as [rows, cols].
// The convolution branch's intermediates are dumped in PyTorch's [B, C, T]
// layout -- the branch transposes into it and back out again -- while every
// tensor in this package is [frames, features].
func transposed(src []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[r*cols+c] = src[c*rows+r]
		}
	}
	return out
}

// TestGPULayerStages walks layer 0 on the device, dispatch by dispatch,
// against the same reference tensors TestEncoderLayerStages holds the CPU
// implementation to.
//
// Every stage is reached with RunTo, which re-runs the prefix and stops: the
// layer's four branch outputs share one tensor and its four pre-norms share
// one fp16 operand, so a test that only ran the layer could compare a third of
// these. What it costs is 20 re-runs of a 0.3 ms graph.
func TestGPULayerStages(t *testing.T) {
	g, m, x, valid, done := newTestEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	check := func(label, name string, got func() []float32) {
		t.Helper()
		if err := g.RunTo(x, valid, 0, label); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		want, meta := loadRef(t, ref, name)
		data := got()
		if len(data) != meta.Count {
			t.Fatalf("%s: got %d values, reference has %v", name, len(data), meta.Shape)
		}
		d := compare(t, data, want)
		t.Logf("%-14s %-16s max abs %.3g, rms %.4g, %.2g relative", label, name, d.MaxAbs, d.RMS, d.Rel())
		if d.Rel() > gpuTol {
			t.Errorf("%s deviates by %.3g relative, more than %g", name, d.Rel(), gpuTol)
		}
	}
	rows, dim := g.Rows(), g.Dim()
	// The four pre-norms write the fp16 A operand their branch's first GEMM
	// reads, so that is where they are read from -- there is no fp32 copy of
	// a norm output anywhere in this graph.
	normA := func() []float32 { return g.ReadF16(g.TensorA(), rows, dim, g.LDA()).Data }
	fp32 := func(off uint32, cols int) func() []float32 {
		return func() []float32 { return g.Read(off, cols).Data }
	}

	check("norm ff1", "norm_ff1", normA)
	check("gemm ff1.linear2", "ff1", fp32(g.TensorBranch(), dim))
	check("resid ff1", "resid1", fp32(g.TensorX(), dim))

	check("norm attn", "norm_self_att", normA)
	check("gemm q", "q", fp32(g.TensorQ(), dim))
	check("gemm k", "k", fp32(g.TensorK(), dim))
	check("gemm v", "v", fp32(g.TensorV(), dim))
	check("gemm rel_k", "rel_k", func() []float32 {
		return g.ReadRows(g.TensorRelK(), g.PosRows(), dim).Data
	})
	// The position term, before and after the shift. The raw scores are one
	// [rows, 2T-1] plane per head inside a padded arena, and the shifted ones
	// carry log2(e) as well as the softmax scale, since the score kernel's
	// exponential is exp2 -- the reference's matrix_bd has only the scale.
	check("gemm pos 7", "matrix_bd_raw", func() []float32 {
		out := make([]float32, 0, g.Heads()*rows*g.PosRows())
		for h := 0; h < g.Heads(); h++ {
			plane := g.ReadRows(g.TensorBD()+uint32(h*g.PlaneRows()*g.PosPadded()), rows, g.PosPadded())
			for r := 0; r < rows; r++ {
				out = append(out, plane.Row(r)[:g.PosRows()]...)
			}
		}
		return out
	})
	check("rel shift", "matrix_bd", func() []float32 {
		out := make([]float32, 0, g.Heads()*rows*rows)
		for h := 0; h < g.Heads(); h++ {
			plane := g.ReadRows(g.TensorBias()+uint32(h*g.PlaneRows()*g.PlaneRows()), rows, g.PlaneRows())
			for r := 0; r < rows; r++ {
				for _, v := range plane.Row(r)[:rows] {
					out = append(out, v/float32(log2e))
				}
			}
		}
		return out
	})
	check("attention", "attn_ctx", func() []float32 {
		return g.ReadF16(g.TensorCtx(), rows, dim, g.LDA()).Data
	})
	check("gemm o", "attn_out", fp32(g.TensorBranch(), dim))
	check("resid attn", "resid2", fp32(g.TensorX(), dim))

	check("norm conv", "norm_conv", normA)
	// The convolution branch's intermediates are dumped channel-major.
	checkT := func(label, name string, cols int, got func() []float32) {
		t.Helper()
		if err := g.RunTo(x, valid, 0, label); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		want, meta := loadRef(t, ref, name)
		if meta.Shape[0] != cols || meta.Shape[1] != rows {
			t.Fatalf("%s: reference is %v, want [%d %d]", name, meta.Shape, cols, rows)
		}
		d := compare(t, got(), transposed(want, rows, cols))
		t.Logf("%-14s %-16s max abs %.3g, rms %.4g, %.2g relative", label, name, d.MaxAbs, d.RMS, d.Rel())
		if d.Rel() > gpuTol {
			t.Errorf("%s deviates by %.3g relative, more than %g", name, d.Rel(), gpuTol)
		}
	}
	checkT("gemm conv.pw1", "conv_pw1", 2*dim, fp32(g.TensorPW1(), 2*dim))
	checkT("glu", "conv_glu", dim, fp32(g.TensorGLU(), dim))
	// The depthwise convolution, its folded BatchNorm and silu are one pass,
	// so the comparison is against silu(conv_bn) rather than against either
	// tensor on its own -- which is also what makes the fold itself checked:
	// conv_bn is the reference's BatchNorm output, computed from the running
	// statistics this code folded away at load.
	{
		if err := g.RunTo(x, valid, 0, "dwconv"); err != nil {
			t.Fatal(err)
		}
		want, _ := loadRef(t, ref, "conv_bn")
		act := transposed(want, rows, dim)
		for i, v := range act {
			act[i] = SiLU(v)
		}
		got := g.ReadF16(g.TensorPWA(), rows, dim, g.LDA())
		d := compare(t, got.Data, act)
		t.Logf("%-14s %-16s max abs %.3g, rms %.4g, %.2g relative", "dwconv", "silu(conv_bn)", d.MaxAbs, d.RMS, d.Rel())
		if d.Rel() > gpuTol {
			t.Errorf("the depthwise branch deviates by %.3g relative", d.Rel())
		}
	}
	check("gemm conv.pw2", "conv_out", fp32(g.TensorBranch(), dim))
	check("resid conv", "resid3", fp32(g.TensorX(), dim))

	check("norm ff2", "norm_ff2", normA)
	check("gemm ff2.linear2", "ff2", fp32(g.TensorBranch(), dim))
	check("norm out", "layer_out", fp32(g.TensorX(), dim))

	_ = m
}

// TestGPUEncoderStack runs all 24 layers and compares the four hidden states
// the reference dumped plus the encoder's output, which is where a per-layer
// error inside the bound would show up as an accumulated one.
func TestGPUEncoderStack(t *testing.T) {
	g, _, x, valid, done := newTestEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	dumped := map[int]string{1: "hidden_1", 2: "hidden_2", 12: "hidden_12", g.Layers() - 1: "hidden_23"}
	for i := 0; i < g.Layers(); i++ {
		if err := g.RunTo(x, valid, i, "norm out"); err != nil {
			t.Fatal(err)
		}
		name, ok := dumped[i]
		if !ok {
			continue
		}
		want, _ := loadRef(t, ref, name)
		d := compare(t, g.Read(g.TensorX(), g.Dim()).Data, want)
		t.Logf("layer %-2d %-12s max abs %.3g, rms %.4g, %.2g relative", i, name, d.MaxAbs, d.RMS, d.Rel())
		if d.Rel() > gpuOutTol {
			t.Errorf("%s deviates by %.3g relative, more than %g", name, d.Rel(), gpuOutTol)
		}
	}
	want, _ := loadRef(t, ref, "encoder_out")
	d := compare(t, g.Read(g.TensorX(), g.Dim()).Data, want)
	t.Logf("encoder_out max abs %.3g, rms %.4g, %.2g relative", d.MaxAbs, d.RMS, d.Rel())
	if d.Rel() > gpuOutTol {
		t.Errorf("the encoder output deviates by %.3g relative, more than %g", d.Rel(), gpuOutTol)
	}
}

// TestGPUTranscript is the bound that counts.
//
// A tensor-level bound cannot certify this path: fp16 operands move the
// encoder output by 4.3% relative and the transducer does not care, which
// SPEECH.md established on the CPU before a shader was written. What decides
// whether the port is correct is whether the same words come out, at the same
// frames, with the same durations -- so this runs the encoder on the device,
// the projector and the TDT loop on the host, and compares the whole trace
// against the reference's, emission for emission.
func TestGPUTranscript(t *testing.T) {
	g, m, x, valid, done := newTestEncoder(t, controls{})
	defer done()
	ref := loadManifest(t)

	hidden, err := g.Apply(x, valid)
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
	t.Logf("%d emissions, identical frames and durations", len(out.Steps))
}

// decodeHidden is the CPU tail: the projector and the TDT greedy loop, which
// together are 7.8% of the reference's time and are S7's business rather than
// S6's.
func decodeHidden(m *Model, hidden *Mat, valid int) (*Transcript, error) {
	enc, err := m.Projector.Apply(hidden)
	if err != nil {
		return nil, err
	}
	return m.Decode(enc, valid)
}

// TestGPUDetectsErrors is the negative control: four things this graph does
// that a plausible-looking one would not, each broken on its own and each
// required to move the encoder's output by at least 10x the bound the intact
// graph meets.
//
// The clip is *padded* -- eight frames of silence past the 138 real ones --
// because one of the four only acts on padding, and the fixture has none: 138
// of its 138 frames are valid. Everything else is the same run.
//
// The assertion is the tensor and not the transcript, and that is a finding
// rather than a convenience. Two of these four leave the words alone: the
// transducer has enough margin to absorb a 24-layer stack normalised without
// its mean, and enough to absorb padding walking four frames back through
// every depthwise convolution. The transcript is what says the *port* is
// right (TestGPUTranscript); it is not sensitive enough to be what says a
// kernel is.
func TestGPUDetectsErrors(t *testing.T) {
	if testing.Short() {
		t.Skip("five more runs of the whole encoder")
	}
	m := testModel(t)
	dev, done := newTestDevice(t)
	defer done()
	x, valid := gpuInput(t, m)

	// The padded clip: the real frames, then eight frames of what the front
	// end produces for silence -- zeros before the subsampling, which is not
	// zeros after it, so this is padding the model could mistake for speech
	// if anything let it.
	padded := NewMat(x.Rows+8, x.Cols)
	copy(padded.Data, x.Data)
	for r := x.Rows; r < padded.Rows; r++ {
		copy(padded.Row(r), x.Row(x.Rows-1))
	}
	run := func(ctl controls) (*Mat, *Transcript) {
		t.Helper()
		g, err := newGPUEncoder(dev, m.Encoder, padded.Rows, nil, ctl)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		hidden, err := g.Apply(padded.Clone(), valid)
		if err != nil {
			t.Fatal(err)
		}
		out, err := decodeHidden(m, hidden, valid)
		if err != nil {
			t.Fatal(err)
		}
		return hidden, out
	}
	// The intact run over the padded clip, which must still transcribe the
	// clip: padding that changes the words would make every comparison below
	// meaningless.
	base, baseOut := run(controls{})
	if baseOut.Text != transcript {
		t.Fatalf("the padded clip transcribes as %q", truncate(baseOut.Text, 72))
	}
	// Only the valid frames are compared: what the encoder leaves in a
	// padding frame is not defined by the reference and not read by the
	// decoder.
	live := func(h *Mat) []float32 { return h.Data[:valid*h.Cols] }

	// How far the *intact* graph is from the reference at this tensor, which
	// is what each control has to miss by 10x. Bounding against the measured
	// drift rather than against gpuOutTol is the stronger statement: the
	// bound is already set several times above what the port achieves, and a
	// control that only had to clear the bound would be a weaker test the
	// better the port got.
	//
	// It is also the assertion that padding does not reach the valid frames:
	// this run has eight pad frames and the reference has none.
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
		{"no mean", controls{noMean: true}, "every LayerNorm becomes an RMS norm"},
		{"centre slice", controls{shiftSlice: true}, "the position scores are read down the middle instead of the diagonal"},
		{"no bias_u", controls{noBiasU: true}, "the content term loses its learned query bias"},
		{"no pad zero", controls{noPadZero: true}, "padding walks back into real frames through the depthwise kernel"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, out := run(c.ctl)
			d := compare(t, live(got), live(base))
			changed := out.Text != baseOut.Text
			t.Logf("%-13s %.3g relative, transcript %s: %q",
				c.name, d.Rel(), map[bool]string{true: "changed", false: "unchanged"}[changed],
				truncate(out.Text, 60))
			if d.Rel() < 10*intact {
				t.Errorf("%s (%s) moved the encoder output by only %.3g relative, which is inside 10x the intact %.3g",
					c.name, c.what, d.Rel(), intact)
			}
		})
	}

	// The mean is the weakest of the four by a long way -- 3.9% of the
	// encoder output and nothing at all to the transcript -- because a
	// residual stream 24 layers deep is nearly zero-mean, which is the very
	// thing that makes an RMS norm a plausible substitute. So it gets a
	// second check, at the tensor where it acts rather than at the end of the
	// chain: layer 0's first norm, whose input is the subsampled features
	// scaled by sqrt(1024) and is not zero-mean at all.
	t.Run("no mean, at the norm", func(t *testing.T) {
		want := mustRef(t, "norm_ff1")
		norm := func(ctl controls) float64 {
			g, err := newGPUEncoder(dev, m.Encoder, x.Rows, nil, ctl)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Destroy()
			if err := g.RunTo(x.Clone(), valid, 0, "norm ff1"); err != nil {
				t.Fatal(err)
			}
			return compare(t, g.ReadF16(g.TensorA(), g.Rows(), g.Dim(), g.LDA()).Data, want).Rel()
		}
		with, without := norm(controls{}), norm(controls{noMean: true})
		t.Logf("norm_ff1: %.3g relative with the mean, %.3g without", with, without)
		if without < 10*with {
			t.Errorf("dropping the mean moved norm_ff1 by %.3g relative, against %.3g intact", without, with)
		}
	})
}

// mustRef reads one reference tensor, for the comparisons that do not care
// about its metadata.
func mustRef(t *testing.T, name string) []float32 {
	t.Helper()
	data, _ := loadRef(t, loadManifest(t), name)
	return data
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// TestGPUProfile times every dispatch of the graph on the device and reports
// the encoder's cost by kind, which is what says where S7's optimisation
// should go.
func TestGPUProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("one profiled run of the whole encoder")
	}
	g, _, x, valid, done := newTestEncoder(t, controls{})
	defer done()

	// One untimed run first: the first dispatch of a pipeline pays for its
	// own compilation and the first read of a weight comes off DRAM cold.
	if _, err := g.Apply(x.Clone(), valid); err != nil {
		t.Fatal(err)
	}
	stages, _, err := g.Profile(x.Clone(), valid)
	if err != nil {
		t.Fatal(err)
	}
	byKind := map[string]struct {
		n int
		d float64
	}{}
	var total float64
	for _, s := range stages {
		e := byKind[s.Kind]
		e.n++
		e.d += s.GPU.Seconds()
		byKind[s.Kind] = e
		total += s.GPU.Seconds()
	}
	t.Logf("%d dispatches, %.3f ms of GPU time, %.0f GFLOP, %.1f TFLOP/s",
		len(stages), total*1e3, g.FLOPs(g.Rows())/1e9, g.FLOPs(g.Rows())/total/1e12)
	for _, k := range g.Labels() {
		e := byKind[k]
		if e.n == 0 {
			continue
		}
		fmt.Printf("  %-18s %3d  %7.3f ms  %4.1f%%\n", k, e.n, e.d*1e3, 100*e.d/total)
		delete(byKind, k)
	}
	for k, e := range byKind {
		fmt.Printf("  %-18s %3d  %7.3f ms  %4.1f%%\n", k, e.n, e.d*1e3, 100*e.d/total)
	}
}

// TestGPUGEMMLadder times the whole encoder once per projection kernel, at
// four clip lengths, which is what PlanFor's boundaries are read off.
//
// It is a ladder over the *graph* and not over a GEMM in isolation, and that
// is the point: results/shapes.csv measures one shape at a time with its
// operands hot, and this runs 1.15 GB of weights past the cores once per
// clip, so the rung that wins here is allowed to disagree with the rung that
// wins there.
//
// The longer clips are the fixture's frames repeated. Nothing about the
// timing depends on what the frames say -- every dispatch covers the same
// tiles whatever is in them -- and the alternative is a fixture per length.
func TestGPUGEMMLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("one run of the whole encoder per rung per length")
	}
	m := testModel(t)
	dev, done := newTestDevice(t)
	defer done()
	x, _ := gpuInput(t, m)

	const maxFrames = 1024
	g, err := newGPUEncoder(dev, m.Encoder, maxFrames, DefaultGEMMPlan(), controls{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("%d layers, %d MB of weights, %d MB of arenas at %d frames",
		g.Layers(), g.WeightBytes()>>20, g.ActivationBytes()>>20, maxFrames)

	// A clip of n frames, the fixture's repeated.
	clip := func(n int) *Mat {
		out := NewMat(n, x.Cols)
		for r := 0; r < n; r++ {
			copy(out.Row(r), x.Row(r%x.Rows))
		}
		return out
	}
	for _, frames := range []int{x.Rows, 384, 768, 1024} {
		in := clip(frames)
		best := time.Duration(math.MaxInt64)
		type row struct {
			kernel GEMMKernel
			d      time.Duration
			m      int
		}
		var rows []row
		for _, k := range GEMMKernels() {
			if err := g.SetPlan(UniformGEMMPlan(k)); err != nil {
				t.Fatalf("%s: %v", k, err)
			}
			d := time.Duration(math.MaxInt64)
			for rep := 0; rep < 3; rep++ {
				if err := g.Upload(in, frames); err != nil {
					t.Fatal(err)
				}
				start := time.Now()
				if err := g.Run(); err != nil {
					t.Fatalf("%s: %v", k, err)
				}
				d = min(d, time.Since(start))
			}
			rows = append(rows, row{k, d, g.RowsPadded()})
			best = min(best, d)
		}
		t.Logf("--- %d frames (%.1f s of audio)", frames, float64(frames)*8*0.01)
		for _, r := range rows {
			t.Logf("  %-20s M=%-5d %7.2f ms  %.2fx", r.kernel, r.m,
				float64(r.d.Microseconds())/1e3, float64(r.d)/float64(best))
		}
	}
}
