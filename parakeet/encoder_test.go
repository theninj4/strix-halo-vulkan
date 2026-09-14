package parakeet

import (
	"math"
	"strconv"
	"sync"
	"testing"

	"strix-halo-vulkan/audio"
)

// The encoder is 0.6 B parameters of fp32 read many times over, so loading it
// once for every test in the package is worth the global.
var (
	loadOnce  sync.Once
	sharedM   *Model
	sharedErr error
)

func testModel(t *testing.T) *Model {
	t.Helper()
	loadOnce.Do(func() { sharedM, sharedErr = Load(modelDir) })
	if sharedErr != nil {
		t.Skipf("no checkpoint in %s (%v)", modelDir, sharedErr)
	}
	return sharedM
}

// features runs the front end over the fixture, which every encoder test
// starts from.
func features(t *testing.T, m *Model) (*Mat, int) {
	t.Helper()
	clip, err := audio.ReadWAV(fixture)
	if err != nil {
		t.Fatal(err)
	}
	f, err := m.FrontEnd.Features(clip)
	if err != nil {
		t.Fatal(err)
	}
	return &Mat{Rows: f.Frames, Cols: f.Mels, Data: f.Data}, f.Valid
}

// encTol is the bound for the encoder's stages, relative to each stage's own
// rms.
//
// It is wider than the front end's because its inputs are: the mel features
// this runs on are the Go ones, which already differ from the reference's by
// 3e-4, and every stage amplifies that by whatever its weights do. A
// *relative* bound is the only one that means anything across stages whose
// rms spans 0.02 to 400, so that is what is measured and printed. The worst
// stage over the whole model is the encoder's output at 1.7e-4 — the last
// layer's norm divides by a small variance and magnifies the drift — and the
// bound is an order of magnitude above it. TestEncoderDetectsErrors pins the
// other side.
const encTol = 2e-3

func TestSubsamplingMatchesReference(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)
	mel, valid := features(t, m)

	// The reference's own names for the five convolutions, which are their
	// indices in the checkpoint's ModuleList.
	seen := map[int]bool{}
	out, encValid, err := m.Encoder.Subsampling.forward(mel, valid, func(c *Conv2D, v *Volume) {
		name := "sub_" + strconv.Itoa(c.Index)
		want, meta := loadRef(t, ref, name)
		if len(meta.Shape) != 3 || meta.Shape[0] != v.C || meta.Shape[1] != v.T || meta.Shape[2] != v.F {
			t.Fatalf("%s: got %v, reference has %v", name, v, meta.Shape)
		}
		d := compare(t, v.Data, want)
		t.Logf("%-8s %-16s max abs %.3g, rms %.4g", name, v.String(), d.MaxAbs, d.RMS)
		if d.MaxAbs > encTol*d.RMS {
			t.Errorf("%s deviates by %.3g, which is more than %g of its rms %.4g", name, d.MaxAbs, encTol, d.RMS)
		}
		seen[c.Index] = true
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{0, 2, 3, 5, 6} {
		if !seen[i] {
			t.Errorf("convolution %d never ran", i)
		}
	}
	if encValid != ref.Encoder.ValidFrames || out.Rows != ref.Encoder.Frames {
		t.Fatalf("%d frames (%d valid), reference has %d (%d valid)",
			out.Rows, encValid, ref.Encoder.Frames, ref.Encoder.ValidFrames)
	}

	want, _ := loadRef(t, ref, "subsampled")
	d := compare(t, out.Data, want)
	t.Logf("subsampled %s max abs %.3g, rms %.4g", out.String(), d.MaxAbs, d.RMS)
	if d.MaxAbs > encTol*d.RMS {
		t.Errorf("subsampled deviates by %.3g against an rms of %.4g", d.MaxAbs, d.RMS)
	}
}

// TestPositionEmbeddings pins the sinusoid layout, which is the kind of
// detail that produces a plausible-looking transcript of the wrong words: the
// sin and cos components are *interleaved* here, where every other positional
// scheme in this repository splits the vector in half.
func TestPositionEmbeddings(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)
	want, meta := loadRef(t, ref, "pos_embed")
	got := m.Encoder.PositionEmbeddings(ref.Encoder.Frames)
	if got.Rows != meta.Shape[0] || got.Cols != meta.Shape[1] {
		t.Fatalf("got %s, reference has %v", got, meta.Shape)
	}
	d := compare(t, got.Data, want)
	t.Logf("max abs %.3g over offsets +/-%d", d.MaxAbs, got.Rows/2)
	// The reference holds the angle in float32, where one ulp at an offset of
	// 137 frames is already 1e-5, so this bound is the *reference's*
	// precision rather than this code's — and it widens with the clip, since
	// the angle grows with the offset. Computing in float64 and rounding once
	// is the more accurate of the two.
	if d.MaxAbs > 1e-4 {
		t.Errorf("position embeddings deviate by %.3g", d.MaxAbs)
	}
}

// TestRelShift checks the relative-position shift on its own, against the
// reference's before and after: `matrix_bd_raw` is the [T, 2T-1] matrix of
// every query against every relative offset, and `matrix_bd` is the [T, T]
// matrix the scores are actually built from. Getting this wrong is the one
// way to build an attention that runs, produces finite numbers, and attends
// to the wrong frames.
func TestRelShift(t *testing.T) {
	ref := loadManifest(t)
	raw, rawMeta := loadRef(t, ref, "matrix_bd_raw")
	want, _ := loadRef(t, ref, "matrix_bd")
	heads, tq, p := rawMeta.Shape[0], rawMeta.Shape[1], rawMeta.Shape[2]
	if p != 2*tq-1 {
		t.Fatalf("reference has %d relative offsets for %d frames", p, tq)
	}
	// The reference's matrix_bd is already scaled by 1/sqrt(head_dim); the
	// shift itself is not, so the comparison applies the scale before
	// comparing.
	scale := float32(1 / math.Sqrt(float64(ref.Encoder.HeadDim)))

	for h := 0; h < heads; h++ {
		got := relShift(raw[h*tq*p:(h+1)*tq*p], tq, p)
		block := want[h*tq*tq : (h+1)*tq*tq]
		scaled := make([]float32, len(got))
		for i, v := range got {
			scaled[i] = v * scale
		}
		d := compare(t, scaled, block)
		if d.MaxAbs > 1e-3 {
			t.Fatalf("head %d: rel shift deviates by %.3g against an rms of %.4g", h, d.MaxAbs, d.RMS)
		}
		if h == 0 {
			t.Logf("head 0: max abs %.3g, rms %.4g, %d x %d from %d offsets",
				d.MaxAbs, d.RMS, tq, tq, p)
		}
	}
}

// TestEncoderLayerStages walks layer 0 stage by stage. Every tensor here is
// recomputed by the reference from the same weights, so a mismatch names the
// operator that caused it.
func TestEncoderLayerStages(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)
	mel, valid := features(t, m)

	x, encValid, err := m.Encoder.Subsampling.Apply(mel, valid)
	if err != nil {
		t.Fatal(err)
	}
	pos := m.Encoder.PositionEmbeddings(x.Rows)
	layer := m.Encoder.Layers[0]

	check := func(name string, got *Mat) {
		t.Helper()
		want, meta := loadRef(t, ref, name)
		if got.Rows*got.Cols != meta.Count {
			t.Fatalf("%s: got %s, reference has %v", name, got, meta.Shape)
		}
		d := compare(t, got.Data, want)
		t.Logf("%-16s %-12s max abs %.3g, rms %.4g", name, got.String(), d.MaxAbs, d.RMS)
		if d.MaxAbs > encTol*d.RMS {
			t.Errorf("%s deviates by %.3g against an rms of %.4g", name, d.MaxAbs, d.RMS)
		}
	}

	n1, err := layer.NormFF1.Apply(x)
	if err != nil {
		t.Fatal(err)
	}
	check("norm_ff1", n1)
	ff1, err := layer.FF1.Apply(n1)
	if err != nil {
		t.Fatal(err)
	}
	check("ff1", ff1)
	x.AddInPlace(ff1, 0.5)
	check("resid1", x)

	n2, _ := layer.NormAttn.Apply(x)
	check("norm_self_att", n2)
	q, _ := layer.Attn.Q.Apply(n2)
	check("q", q)
	k, _ := layer.Attn.K.Apply(n2)
	check("k", k)
	v, _ := layer.Attn.V.Apply(n2)
	check("v", v)
	relK, _ := layer.Attn.RelK.Apply(pos)
	check("rel_k", relK)

	attn, err := layer.Attn.Apply(n2, pos, encValid)
	if err != nil {
		t.Fatal(err)
	}
	check("attn_out", attn)
	x.AddInPlace(attn, 1)
	check("resid2", x)

	n3, _ := layer.NormConv.Apply(x)
	check("norm_conv", n3)
	conv, err := layer.Conv.Apply(n3, encValid)
	if err != nil {
		t.Fatal(err)
	}
	check("conv_out", conv)
	x.AddInPlace(conv, 1)
	check("resid3", x)

	n4, _ := layer.NormFF2.Apply(x)
	check("norm_ff2", n4)
	ff2, err := layer.FF2.Apply(n4)
	if err != nil {
		t.Fatal(err)
	}
	check("ff2", ff2)
	x.AddInPlace(ff2, 0.5)
	if err := layer.NormOut.ApplyInPlace(x); err != nil {
		t.Fatal(err)
	}
	check("layer_out", x)
}

// TestEncoderStack runs all 24 layers and compares the four the reference
// dumped plus the encoder's output, which is where a per-layer error that is
// individually inside the bound would show up as an accumulated one.
func TestEncoderStack(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole encoder is ~180 GFLOP on the CPU")
	}
	m := testModel(t)
	ref := loadManifest(t)
	mel, valid := features(t, m)

	dumped := map[int]string{1: "hidden_1", 2: "hidden_2", 12: "hidden_12", ref.Encoder.Layers - 1: "hidden_23"}
	out, encValid, err := m.Encoder.forward(mel, valid, func(i int, h *Mat) {
		name, ok := dumped[i]
		if !ok {
			return
		}
		want, _ := loadRef(t, ref, name)
		d := compare(t, h.Data, want)
		t.Logf("layer %-2d %-12s max abs %.3g, rms %.4g (%.2g relative)", i, name, d.MaxAbs, d.RMS, d.MaxAbs/d.RMS)
		if d.MaxAbs > encTol*d.RMS {
			t.Errorf("%s deviates by %.3g against an rms of %.4g", name, d.MaxAbs, d.RMS)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if encValid != ref.Encoder.ValidFrames {
		t.Errorf("%d valid frames, reference has %d", encValid, ref.Encoder.ValidFrames)
	}

	want, _ := loadRef(t, ref, "encoder_out")
	d := compare(t, out.Data, want)
	t.Logf("encoder_out %s max abs %.3g, rms %.4g (%.2g relative)", out.String(), d.MaxAbs, d.RMS, d.MaxAbs/d.RMS)
	if d.MaxAbs > encTol*d.RMS {
		t.Errorf("encoder output deviates by %.3g against an rms of %.4g", d.MaxAbs, d.RMS)
	}

	proj, err := m.Projector.Apply(out)
	if err != nil {
		t.Fatal(err)
	}
	want, _ = loadRef(t, ref, "encoder_projected")
	d = compare(t, proj.Data, want)
	t.Logf("encoder_projected %s max abs %.3g, rms %.4g", proj.String(), d.MaxAbs, d.RMS)
	if d.MaxAbs > encTol*d.RMS {
		t.Errorf("projected output deviates by %.3g against an rms of %.4g", d.MaxAbs, d.RMS)
	}
}

// TestEncoderDetectsErrors is the other end of encTol. Each mutation below is
// a way to build an encoder that runs, produces finite numbers of a plausible
// size, and is wrong — the relative-position shift dropped, the two content
// biases swapped, the GLU halves exchanged, the BatchNorm fold skipped — and
// each has to move layer 0's output by orders of magnitude more than the
// bound, or the comparisons above are not evidence of anything.
func TestEncoderDetectsErrors(t *testing.T) {
	m := testModel(t)
	ref := loadManifest(t)
	mel, valid := features(t, m)
	want, _ := loadRef(t, ref, "layer_out")
	layer := m.Encoder.Layers[0]

	run := func() deviation {
		x, encValid, err := m.Encoder.Subsampling.Apply(mel, valid)
		if err != nil {
			t.Fatal(err)
		}
		pos := m.Encoder.PositionEmbeddings(x.Rows)
		out, err := layer.Apply(x, pos, encValid)
		if err != nil {
			t.Fatal(err)
		}
		return compare(t, out.Data, want)
	}

	base := run()
	t.Logf("%-26s max abs %.3g, rms %.4g (%.2g relative)", "unmodified", base.MaxAbs, base.RMS, base.MaxAbs/base.RMS)
	if base.MaxAbs > encTol*base.RMS {
		t.Fatalf("the unmodified layer is already outside the bound")
	}

	cases := []struct {
		name    string
		breakIt func() func() // mutate, returning an undo
	}{
		{"swapped bias_u/bias_v", func() func() {
			a := layer.Attn
			a.BiasU, a.BiasV = a.BiasV, a.BiasU
			return func() { a.BiasU, a.BiasV = a.BiasV, a.BiasU }
		}},
		{"no content bias", func() func() {
			a := layer.Attn
			old := a.BiasU
			a.BiasU = make([]float32, len(old))
			return func() { a.BiasU = old }
		}},
		{"unfolded batchnorm", func() func() {
			c := layer.Conv
			scale, shift := c.BNScale, c.BNShift
			c.BNScale = make([]float32, len(scale))
			c.BNShift = make([]float32, len(shift))
			for i := range scale {
				c.BNScale[i] = 1
			}
			return func() { c.BNScale, c.BNShift = scale, shift }
		}},
		{"swapped GLU halves", func() func() {
			pw1 := layer.Conv.PW1
			ch := layer.Conv.Channels
			old := pw1.Weight
			swapped := make([]float32, len(old))
			copy(swapped, old[ch*pw1.In:])
			copy(swapped[ch*pw1.In:], old[:ch*pw1.In])
			pw1.Weight = swapped
			return func() { pw1.Weight = old }
		}},
		{"half-split position layout", func() func() {
			// RoPE's layout instead of the interleaved sin/cos this model
			// uses: the same numbers in a different order, which is exactly
			// the mistake a reader of the shapes alone would make.
			a := layer.Attn
			old := a.RelK.Weight
			shuffled := make([]float32, len(old))
			half := a.RelK.In / 2
			for o := 0; o < a.RelK.Out; o++ {
				src := old[o*a.RelK.In : (o+1)*a.RelK.In]
				dst := shuffled[o*a.RelK.In : (o+1)*a.RelK.In]
				for i := 0; i < half; i++ {
					dst[2*i], dst[2*i+1] = src[i], src[half+i]
				}
			}
			a.RelK.Weight = shuffled
			return func() { a.RelK.Weight = old }
		}},
	}

	for _, tc := range cases {
		undo := tc.breakIt()
		d := run()
		undo()
		t.Logf("%-26s max abs %.3g (%.0fx the bound)", tc.name, d.MaxAbs, d.MaxAbs/(encTol*base.RMS))
		if d.MaxAbs < 10*encTol*base.RMS {
			t.Errorf("%s moved layer 0 by only %.3g, which encTol=%g would not catch", tc.name, d.MaxAbs, encTol)
		}
	}
}

// TestRelShiftIsNotASlice pins what the shift actually does, against the
// thing it is most often mistaken for: taking the middle T columns of the
// 2T-1 wide matrix. The two agree on exactly one row.
func TestRelShiftIsNotASlice(t *testing.T) {
	const tq = 6
	p := 2*tq - 1
	in := make([]float32, tq*p)
	for i := range in {
		in[i] = float32(i)
	}
	got := relShift(in, tq, p)

	// Row i reads the 2T-1 offsets starting T-1-i places in, which is what
	// makes column j hold the score for relative offset i-j.
	for i := 0; i < tq; i++ {
		for j := 0; j < tq; j++ {
			off := tq - 1 - i + j
			var want float32
			if off >= 0 && off < p {
				want = in[i*p+off]
			}
			if got[i*tq+j] != want {
				t.Fatalf("row %d col %d = %v, want %v", i, j, got[i*tq+j], want)
			}
		}
	}
	// And the naive centre slice is only right on the row where the two
	// happen to coincide.
	agree := 0
	for i := 0; i < tq; i++ {
		same := true
		for j := 0; j < tq; j++ {
			if got[i*tq+j] != in[i*p+(tq-1)/2+j] {
				same = false
			}
		}
		if same {
			agree++
		}
	}
	if agree > 1 {
		t.Errorf("the centre slice agrees with the shift on %d rows", agree)
	}
}
