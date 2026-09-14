package dit

import (
	"math/rand"
	"testing"

	"strix-halo-vulkan/safetensors"
)

// Stage 9's tests. The head is 130 MB of small linear layers and none of it
// is arithmetically hard; what it is full of is convention, which is why
// head_test.go has a control per convention and why this file compares the
// device path against the *same* reference rather than against the host path
// alone. reference/out/zimage carries both ends of it -- `x_prepared` is the
// patch embedder's output and `final` the tail's -- so the GPU port is held
// to diffusers directly.
//
// Two things the reference cannot check, because its 256 image tokens are
// already a multiple of SeqMultiOf and its stream is the real one:
//
//   - the pad-token column, which is checked against the host path at a
//     token count that needs padding;
//   - that the final layer's norm subtracts a mean, which no tolerance
//     against a *correct* reference can prove is load-bearing. That is what
//     the NO_MEAN control is for.

// The two bounds, set from the measured drift rather than inherited from
// fp16RelTol, and the difference between them is the point.
//
// Both GEMMs narrow their A operand to fp16 and accumulate in fp32, but they
// are at opposite ends of the same trade. The embedder reduces over K = 66
// and its output has an RMS of 0.41, so what it measures (3.4e-3) is almost
// entirely the rounding of the *patches themselves*. The tail reduces over
// 3840 and its output has an RMS of 11, so the same per-element rounding
// averages away and it measures 8.6e-4.
//
// Inheriting fp16RelTol (1.5e-2) would have left the NO_MEAN control only 3x
// outside the bound, which is not a control. At 2e-3 it is 22x.
const (
	headEmbedTol = 6e-3
	headFinalTol = 2e-3
)

// loadGPUHead builds a one-block stack sized for the reference's sequence and
// attaches a device head to it. The block is there because a stack needs one
// to read its shapes from; nothing in these tests runs it.
func loadGPUHead(t *testing.T, noMean bool) (*GPUStack, *GPUHead, *Head, *manifest, func()) {
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
	head, err := LoadHead(set, cfg)
	if err != nil {
		set.Close()
		t.Fatal(err)
	}
	ids, err := head.PositionIDs(m.Caption, m.Latent, m.Latent)
	if err != nil {
		set.Close()
		t.Fatal(err)
	}
	rope, err := NewRoPE(ids, cfg.AxesDims, cfg.AxesLens, cfg.RopeTheta)
	if err != nil {
		set.Close()
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	g, err := newStack(dev, set, cfg, []blockSpec{specFor("layers.0")}, rope, len(ids), nil, blockControls{}, maxBankBytes)
	if err != nil {
		set.Close()
		done()
		t.Fatal(err)
	}
	h, err := newGPUHead(g, head, noMean)
	if err != nil {
		g.Destroy()
		set.Close()
		done()
		t.Fatal(err)
	}
	return g, h, head, m, func() {
		h.Destroy()
		g.Destroy()
		set.Close()
		done()
	}
}

// TestGPUHeadEmbed runs the patch embedder on the device against diffusers'
// own prepared image stream. What it is really checking is the bias column:
// dit_gemm.comp has no bias, so `y = Wx + b` is expressed as one more column
// of K holding 1 against one more row of B holding b, and a kernel that
// dropped it would be off by a constant per feature -- which is exactly the
// kind of error an RMS-normalised bound on a downstream tensor forgives.
func TestGPUHeadEmbed(t *testing.T) {
	g, h, _, m, done := loadGPUHead(t, false)
	defer done()

	patches := loadRef(t, m, "img_patches")
	if err := h.Embed(patches, patches.Rows); err != nil {
		t.Fatal(err)
	}
	got := g.Read(g.aX, g.dim)
	compareTol(t, "x_prepared", got, loadRef(t, m, "x_prepared"), headEmbedTol)
}

// TestGPUHeadEmbedPadding is the same GEMM at a token count that is not a
// multiple of SeqMultiOf, where the rows past the image have to come out as
// the learned x_pad_token. On the device they are a *second* extra column of
// K -- 1 for a padded row, 0 for a real one -- so the padding is produced by
// the same matrix multiply rather than written by the host, and this is the
// only test that says so. The oracle is the host path, which head_test.go
// holds to the reference.
func TestGPUHeadEmbedPadding(t *testing.T) {
	g, h, head, _, done := loadGPUHead(t, false)
	defer done()

	const tokens = 36
	rng := rand.New(rand.NewSource(9))
	patches := NewMat(tokens, head.PatchDim())
	for i := range patches.Data {
		patches.Data[i] = float32(rng.NormFloat64())
	}
	total := PadTo(tokens)
	if total == tokens {
		t.Fatalf("%d tokens needs no padding; this test has nothing to check", tokens)
	}
	want, err := head.EmbedImage(patches)
	if err != nil {
		t.Fatal(err)
	}
	if want.Rows != total {
		t.Fatalf("host padded to %d rows, want %d", want.Rows, total)
	}
	if err := h.Embed(patches, total); err != nil {
		t.Fatal(err)
	}
	compareTol(t, "padded embed", g.Read(g.aX, g.dim), want, headEmbedTol)
}

// TestGPUHeadFinal runs the tail on the device over the reference's own
// residual stream: the final layer's adaLN projection on the block's kernel,
// the LayerNorm-scale-narrow pass, and the projection back to a patch.
//
// diffusers runs the final layer over the whole unified sequence and
// discards the caption's rows in unpatchify, so the reference has 320 rows
// and so does this.
func TestGPUHeadFinal(t *testing.T) {
	g, h, head, m, done := loadGPUHead(t, false)
	defer done()

	stream := loadRef(t, m, "stream")
	adaln, err := head.Timestep(m.T)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(stream, 0, stream.Rows); err != nil {
		t.Fatal(err)
	}
	if err := g.SetAdaLN(adaln); err != nil {
		t.Fatal(err)
	}
	got, err := h.Final(stream.Rows)
	if err != nil {
		t.Fatal(err)
	}
	compareTol(t, "final", got, loadRef(t, m, "final"), headFinalTol)
}

// TestGPUHeadFinalControl builds the tail's norm as an RMS norm -- the mean
// left in -- which is what every other norm in this model is and therefore
// what a port writes by habit. head_test.go makes the same point about the
// host implementation; this is the device's copy of it, and it is here
// because the two norms differ by a single subtraction that no amount of
// agreement with a correct reference can prove matters.
func TestGPUHeadFinalControl(t *testing.T) {
	g, h, head, m, done := loadGPUHead(t, true)
	defer done()

	stream := loadRef(t, m, "stream")
	adaln, err := head.Timestep(m.T)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Upload(stream, 0, stream.Rows); err != nil {
		t.Fatal(err)
	}
	if err := g.SetAdaLN(adaln); err != nil {
		t.Fatal(err)
	}
	got, err := h.Final(stream.Rows)
	if err != nil {
		t.Fatal(err)
	}
	_, _, rel, _ := deviation(got, loadRef(t, m, "final"))
	if rel <= headFinalTol {
		t.Errorf("NO_MEAN: rel %.3g is inside the tolerance %.1e -- the control is not controlling anything", rel, headFinalTol)
		return
	}
	t.Logf("NO_MEAN rel %.3g, %.0fx the bound", rel, rel/headFinalTol)
}
