package dit

import (
	"math"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// Stage 6's tests. The reference is reference/dump_zimage.py, which
// instantiates the real ZImageTransformer2DModel with **no blocks** and runs
// its own patchify, embedders, position ids, final layer and unpatchify:
//
//	.venv/bin/python reference/dump_zimage.py
//
// Nothing here is arithmetically hard -- the whole head is 130 MB of small
// linear layers -- and that is the point. What it is full of is convention:
// which order a patch's 64 features are in, which position the caption starts
// at, whether the pad token replaces a row before or after the embedder,
// whether the final norm subtracts a mean. Every one of those produces a
// tensor of the right shape and a picture of noise, so each has a test and a
// control that breaks exactly it.
const refHeadDir = "../../reference/out/zimage"

// loadHead opens the checkpoint's non-block tensors alongside the dump.
func loadHead(t *testing.T) (*Head, *manifest, func()) {
	t.Helper()
	m := loadManifest(t, refHeadDir)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Skipf("no transformer checkpoint at %s (%v)", transformer, err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	h, err := LoadHead(set, cfg)
	if err != nil {
		set.Close()
		t.Fatal(err)
	}
	return h, m, func() { set.Close() }
}

// TestTimestepEmbedding checks the conditioning vector every modulated block
// is driven by. It is 256 numbers that feed all 34 blocks and the final
// layer, so a wrong one is wrong everywhere at once and looks like a bad
// checkpoint rather than a bad embedding.
func TestTimestepEmbedding(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	got, err := h.Timestep(m.T)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "adaln_input", &Mat{Rows: 1, Cols: len(got), Data: got}, loadRef(t, m, "adaln_input"))

	if h.TScale != m.TScale {
		t.Errorf("t_scale %g, reference used %g", h.TScale, m.TScale)
	}
}

// TestPatchify checks the feature order inside an image token. Both orders
// round-trip through Unpatchify, so the round trip proves nothing; only the
// reference says which of the two the x_embedder's weight expects.
func TestPatchify(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	latent := loadRef(t, m, "latent")
	got, err := h.Patchify(latent.Data, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "img_patches", got, loadRef(t, m, "img_patches"))
}

// TestUnpatchify checks the tail's inverse against the reference's, and then
// the round trip -- which catches nothing the first check does not, but says
// the two halves agree with each other as well as with diffusers.
func TestUnpatchify(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	final := loadRef(t, m, "final")
	got, err := h.Unpatchify(final, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	want := loadRef(t, m, "unpatchified")
	compare(t, "unpatchified", &Mat{Rows: 1, Cols: len(got), Data: got}, &Mat{Rows: 1, Cols: len(want.Data), Data: want.Data})

	latent := loadRef(t, m, "latent")
	patches, err := h.Patchify(latent.Data, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	back, err := h.Unpatchify(patches, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	for i := range back {
		if back[i] != latent.Data[i] {
			t.Fatalf("round trip differs at %d: %g vs %g", i, back[i], latent.Data[i])
		}
	}
}

// TestPositionIDs checks the rotary grid, both as ids and as the table they
// build. The ids are where the two streams are tied together: the caption
// starts at axis-0 position 1 and the image sits one past the caption's
// *padded* length, so a 40-token prompt and a 64-token one place the image
// differently and nothing downstream can tell.
func TestPositionIDs(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	ids, err := h.PositionIDs(m.Caption, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	img, cap := loadRef(t, m, "img_pos_ids"), loadRef(t, m, "cap_pos_ids")
	imgTokens := m.ImgTokens
	if len(ids) != imgTokens+PadTo(m.Caption) {
		t.Fatalf("%d ids, want %d image + %d caption", len(ids), imgTokens, PadTo(m.Caption))
	}
	// The reference's caption ids are longer than its caption stream --
	// _pad_with_ids builds them from the padded length and then appends
	// pad_len more, which _prepare_sequence truncates -- so only the first
	// PadTo(caption) of them are ever used.
	for i := 0; i < len(ids); i++ {
		var want []float32
		if i < imgTokens {
			want = img.Row(i)
		} else {
			want = cap.Row(i - imgTokens)
		}
		for a := 0; a < 3; a++ {
			if float32(ids[i][a]) != want[a] {
				t.Fatalf("id %d axis %d: got %d, want %g", i, a, ids[i][a], want[a])
			}
		}
	}

	rope, err := NewRoPE(ids, m.AxesDims, m.AxesLens, m.RopeTheta)
	if err != nil {
		t.Fatal(err)
	}
	pairs := rope.Pairs
	compare(t, "unified_freqs_cos", &Mat{Rows: len(ids), Cols: pairs, Data: rope.Cos}, loadRef(t, m, "unified_freqs_cos"))
	compare(t, "unified_freqs_sin", &Mat{Rows: len(ids), Cols: pairs, Data: rope.Sin}, loadRef(t, m, "unified_freqs_sin"))
}

// TestEmbedStreams checks both embedders and the padding that follows them.
// The caption's is the interesting one: it is the only place in the pipeline
// where the text encoder's output is consumed, its last 24 rows here are the
// learned pad token rather than anything derived from the prompt, and the
// pad token is applied *after* the projection.
func TestEmbedStreams(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	latent := loadRef(t, m, "latent")
	patches, err := h.Patchify(latent.Data, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	x, err := h.EmbedImage(patches)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "x_prepared", x, loadRef(t, m, "x_prepared"))

	capFeats := loadRef(t, m, "cap_feats")
	cap, err := h.EmbedCaption(capFeats)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "cap_prepared", cap, loadRef(t, m, "cap_prepared"))

	if cap.Rows != PadTo(m.Caption) {
		t.Errorf("caption stream is %d rows, want %d", cap.Rows, PadTo(m.Caption))
	}
	// The pad rows must be the learned token exactly, not a repeat of the
	// last caption row -- which is what diffusers pads the *input* with.
	for r := m.Caption; r < cap.Rows; r++ {
		for c, v := range cap.Row(r) {
			if v != h.CapPad[c] {
				t.Fatalf("caption pad row %d column %d is %g, want cap_pad_token %g", r, c, v, h.CapPad[c])
			}
		}
	}
}

// TestFinalLayer checks the tail's norm-scale-project. The reference feeds it
// a tensor of its own rather than a block's output, so what is under test is
// this layer and not the stack that already has an oracle.
func TestFinalLayer(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	stream := loadRef(t, m, "stream")
	adaln := loadRef(t, m, "adaln_input")
	got, err := h.Final(stream, adaln.Data)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "final", got, loadRef(t, m, "final"))
}

// TestHeadDetectsErrors is the negative control. Each case breaks one
// convention the tests above assert, in the way the checkpoint invites, and
// asserts the break is caught: without this the tests prove only that the
// head runs.
func TestHeadDetectsErrors(t *testing.T) {
	h, m, done := loadHead(t)
	defer done()

	latent := loadRef(t, m, "latent")
	capFeats := loadRef(t, m, "cap_feats")
	stream := loadRef(t, m, "stream")
	adaln := loadRef(t, m, "adaln_input")

	cases := []struct {
		name string
		got  func() *Mat
		want string
	}{{
		// The other RoPE-adjacent convention: sin first, cos second. Every
		// norm of the embedding is unchanged.
		"sin before cos", func() *Mat {
			freq := SinusoidalEmbedding(m.T*h.TScale, h.FreqDim)
			half := len(freq) / 2
			for i := 0; i < half; i++ {
				freq[i], freq[half+i] = freq[half+i], freq[i]
			}
			in := &Mat{Rows: 1, Cols: len(freq), Data: freq}
			mid, _ := h.TMLP0.Apply(in)
			for i, v := range mid.Data {
				mid.Data[i] = v / (1 + float32(math.Exp(float64(-v))))
			}
			out, _ := h.TMLP2.Apply(mid)
			return out
		}, "adaln_input",
	}, {
		// Channel-major patches: the same 64 numbers in the other order.
		"channel-major patch", func() *Mat {
			p, c := h.Patch, h.InChan
			ht, wt := m.Latent/p, m.Latent/p
			out := NewMat(ht*wt, h.PatchDim())
			for th := 0; th < ht; th++ {
				for tw := 0; tw < wt; tw++ {
					row := out.Row(th*wt + tw)
					for ch := 0; ch < c; ch++ {
						for ph := 0; ph < p; ph++ {
							for pw := 0; pw < p; pw++ {
								row[(ch*p+ph)*p+pw] = latent.Data[ch*m.Latent*m.Latent+(th*p+ph)*m.Latent+tw*p+pw]
							}
						}
					}
				}
			}
			return out
		}, "img_patches",
	}, {
		// The caption padded by repeating its last row, which is what
		// diffusers pads the embedder's *input* with, instead of by the
		// learned token.
		"repeat instead of pad token", func() *Mat {
			normed := capFeats.Clone()
			h.CapNorm.ApplyInPlace(normed)
			out, _ := h.CapProj.Apply(normed)
			total := PadTo(out.Rows)
			padded := NewMat(total, out.Cols)
			copy(padded.Data, out.Data)
			for r := out.Rows; r < total; r++ {
				copy(padded.Row(r), out.Row(out.Rows-1))
			}
			return padded
		}, "cap_prepared",
	}, {
		// An RMS norm in the final layer instead of a LayerNorm: the mean is
		// not subtracted. Every other norm in this model really is an RMS one.
		"final norm without the mean", func() *Mat {
			act := make([]float32, len(adaln.Data))
			for i, v := range adaln.Data {
				act[i] = v / (1 + float32(math.Exp(float64(-v))))
			}
			scaleMat, _ := h.FinalAda.Apply(&Mat{Rows: 1, Cols: len(act), Data: act})
			normed := NewMat(stream.Rows, stream.Cols)
			for r := 0; r < stream.Rows; r++ {
				src, dst := stream.Row(r), normed.Row(r)
				var sq float64
				for _, v := range src {
					sq += float64(v) * float64(v)
				}
				inv := 1 / math.Sqrt(sq/float64(len(src))+h.FinalEps)
				for i, v := range src {
					dst[i] = float32(float64(v)*inv) * (1 + scaleMat.Data[i])
				}
			}
			out, _ := h.FinalLin.Apply(normed)
			return out
		}, "final",
	}}

	for _, c := range cases {
		_, _, rel, _ := deviation(c.got(), loadRef(t, m, c.want))
		if rel <= relTol {
			t.Errorf("%s: rel %.3g is inside the %.0e bound; the test would not catch it", c.name, rel, relTol)
			continue
		}
		t.Logf("%-28s %-18s rel %.3g = %.0fx the bound", c.name, c.want, rel, rel/relTol)
	}

	// The two id conventions, checked as ids rather than as a tensor: the
	// caption starting at 0, and the image measured from the *unpadded*
	// caption length. Both are off by a constant that no norm can see.
	ids, err := h.PositionIDs(m.Caption, m.Latent, m.Latent)
	if err != nil {
		t.Fatal(err)
	}
	if ids[m.ImgTokens][0] != 1 {
		t.Errorf("first caption id is %d, want 1", ids[m.ImgTokens][0])
	}
	if got, want := ids[0][0], int32(PadTo(m.Caption)+1); got != want {
		t.Errorf("image axis-0 id is %d, want the padded caption length plus one, %d", got, want)
	}
	if PadTo(m.Caption) == m.Caption {
		t.Errorf("caption of %d tokens needs no padding; the dump does not exercise the case", m.Caption)
	}
}
