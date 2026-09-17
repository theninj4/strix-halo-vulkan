package pipeline

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"strix-halo-vulkan/zimage/dit"
	"strix-halo-vulkan/zimage/vae"
)

// The resolution is a per-run parameter and the arenas are a ceiling, so
// everything about resolving a request's size is arithmetic on a patch size
// and needs neither a checkpoint nor a device.
func TestGeomFor(t *testing.T) {
	// A pipeline built for 1024x1024, without anything in it: geomFor reads
	// the head's patch size and the two geometries and nothing else.
	max, err := newGeom(1024, 1024, 2)
	if err != nil {
		t.Fatal(err)
	}
	p := &Pipeline{head: &dit.Head{Patch: 2}, max: max, def: max}

	for _, c := range []struct {
		name          string
		w, h          int
		width, height int
		tokens        int
	}{
		{"the default", 0, 0, 1024, 1024, 4096},
		// One side alone means a square, which is what a client typing "512"
		// into a box means.
		{"width alone", 512, 0, 512, 512, 1024},
		{"height alone", 0, 512, 512, 512, 1024},
		{"smaller", 512, 512, 512, 512, 1024},
		{"landscape", 1024, 512, 1024, 512, 2048},
		{"portrait", 512, 1024, 512, 1024, 2048},
		{"the smallest there is", 16, 16, 16, 16, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			g, err := p.geomFor(c.w, c.h)
			if err != nil {
				t.Fatal(err)
			}
			if g.width != c.width || g.height != c.height {
				t.Errorf("%dx%d, want %dx%d", g.width, g.height, c.width, c.height)
			}
			if g.imgTokens != c.tokens {
				t.Errorf("%d image tokens, want %d", g.imgTokens, c.tokens)
			}
			if g.latentH != c.height/vaeScale || g.latentW != c.width/vaeScale {
				t.Errorf("latent %dx%d for a %dx%d image", g.latentH, g.latentW, c.width, c.height)
			}
			if g.imgTotal < g.imgTokens || g.imgTotal%dit.SeqMultiOf != 0 {
				t.Errorf("%d tokens padded to %d, which is not a multiple of %d",
					g.imgTokens, g.imgTotal, dit.SeqMultiOf)
			}
		})
	}
}

// The sizes that are refused, and why each one has to be refused rather than
// rounded: the arenas are what they are, and a silently different image is
// worse than an error.
func TestGeomForRefuses(t *testing.T) {
	max, err := newGeom(1024, 1024, 2)
	if err != nil {
		t.Fatal(err)
	}
	p := &Pipeline{head: &dit.Head{Patch: 2}, max: max, def: max}

	for _, c := range []struct {
		name, want string
		w, h       int
	}{
		{"off the grid", "multiple of 16", 1000, 1000},
		{"negative", "is not an image", -16, 16},
		{"past the arenas", "every side has to be inside it", 2048, 2048},
		{"past them on one side", "every side has to be inside it", 1024, 1088},
		// **The same area in another shape is refused, and that is the
		// finding rather than a conservative choice.** By the transformer's
		// arithmetic 512x2048 is the same 4096 rows as the square these
		// arenas were built as. TestSameAreaTallerShapeIsRefused measures why
		// it is not enough.
		{"the same area, taller", "every side has to be inside it", 512, 2048},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := p.geomFor(c.w, c.h); err == nil {
				t.Fatalf("%dx%d was accepted", c.w, c.h)
			} else if !strings.Contains(err.Error(), c.want) {
				t.Errorf("%v, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestOneePipelineManySizes is the claim the geometry arithmetic above cannot
// make on its own: that the three graphs actually run at a size other than the
// one their arenas were allocated for.
//
// It is the only place that can say so, because each stage's own tests build
// their arenas for the size they then run. What breaks here if the change is
// wrong is not subtle -- a transformer whose run length did not follow the
// upload reads the previous run's rows, and a VAE whose graph did not follow
// the latent writes past its arena -- but it is invisible to every test below
// this one.
//
// 256x256 is the size reference/dump_zimage_run.py uses and the reason is the
// same: the composition does not know how big the image is, so the cheapest
// size that exercises it is the right one.
func TestOnePipelineManySizes(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: 256, Height: 256, Steps: 2})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	const prompt = "a red fox sitting in fresh snow, photograph"
	for _, c := range []struct{ w, h int }{
		{256, 256}, // what the arenas were built for
		{128, 128}, // a quarter of the tokens
		{256, 128}, // and a shape they were not built as
		{128, 256},
	} {
		name := fmtSize(c.w, c.h)
		t.Run(name, func(t *testing.T) {
			img, tm, err := p.Run(Request{Prompt: prompt, Width: c.w, Height: c.h, Seed: 1})
			if err != nil {
				t.Fatal(err)
			}
			if img.C != 3 || img.H != c.h || img.W != c.w {
				t.Fatalf("decoded [%d %d %d], want [3 %d %d]", img.C, img.H, img.W, c.h, c.w)
			}
			if tm.Width != c.w || tm.Height != c.h {
				t.Errorf("timings say %s", fmtSize(tm.Width, tm.Height))
			}
			// A picture, not an arena: the VAE's output is in [-1, 1] and a
			// run that read stale rows or wrote past its graph comes back
			// constant, enormous or NaN.
			lo, hi := img.Data[0], img.Data[0]
			for _, v := range img.Data {
				if v != v {
					t.Fatal("NaN in the decoded image")
				}
				lo, hi = min(lo, v), max(hi, v)
			}
			if hi-lo < 0.1 {
				t.Errorf("the image is flat: [%g, %g]", lo, hi)
			}
			if lo < -4 || hi > 4 {
				t.Errorf("the image is outside any plausible range: [%g, %g]", lo, hi)
			}
			t.Logf("%s: %d image tokens, %d unified, %v", name, tm.Unified-tm.CapTotal, tm.Unified, tm.Total)
		})
	}

	// And the default path is the same path: naming the pipeline's own size
	// and naming nothing have to be the same image, or the resolution
	// plumbing has perturbed the case that used to be the only one.
	named, _, err := p.Run(Request{Prompt: prompt, Width: 256, Height: 256, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	implied, _, err := p.Run(Request{Prompt: prompt, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	if e := relL2(implied.Data, named.Data); e != 0 {
		t.Errorf("256x256 named and implied differ by %g", e)
	}
}

// TestRunRefusesAMismatchedLatent: GenerateFrom takes its size from the latent
// it is handed, so a latent from another size has to be an error and not a
// reinterpretation of the same floats as a different picture.
func TestRunRefusesAMismatchedLatent(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: 256, Height: 256, Steps: 1})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	small, err := p.NoiseFor(128, 128, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := p.Run(Request{Prompt: "p", Latents: small}); err == nil {
		t.Fatal("a 128x128 latent was accepted for a 256x256 run")
	} else if !strings.Contains(err.Error(), "want C*H*W") {
		t.Errorf("%v", err)
	}
	// The same latent with its own size named is fine.
	if _, _, err := p.Run(Request{Prompt: "p", Width: 128, Height: 128, Latents: small}); err != nil {
		t.Fatal(err)
	}
}

// TestSameAreaTallerShapeIsRefused is the measurement behind geomFor's rule,
// and the reason the rule is each side rather than the token count.
//
// The transformer would take 128x512 out of a 256x256 pipeline without
// noticing: the same 256 image tokens, the same rows, the same arenas. The
// VAE will not, and the reason is stage 8's layout -- its fp16 arena holds
// *blocked* copies of each convolution's input, and a block is padded on each
// axis separately, so redistributing the same area over a taller grid needs
// more of it -- and the decoder says so rather than overrunning, which is how
// this was found. (At 1024-scale the same 4:1 redistribution measures 33 MB
// against a 32 MB arena; the decoder's message rounds to whole MB, so the
// assertion here is that it refuses and not what the two numbers print as.)
//
// This test drives the decoder directly, past geomFor, because geomFor's job
// is now to refuse exactly this and a test that went through it would only be
// testing the comparison. Delete the rule and this test says what comes back
// instead -- which, before the rule existed, was a failure after the whole
// denoising loop had run.
func TestSameAreaTallerShapeIsRefused(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()

	p, err := New(dev, Options{Model: modelDir, Width: 256, Height: 256, Steps: 1})
	if err != nil {
		t.Skipf("cannot build the pipeline (%v)", err)
	}
	defer p.Destroy()

	// geomFor refuses it up front, which is the behaviour.
	if _, err := p.geomFor(128, 512); err == nil {
		t.Fatal("128x512 was accepted out of a 256x256 pipeline")
	}

	// And this is why: the same area, straight at the decoder, does not fit.
	// 32x64 latent is 128x512 pixels, the same 2048 latent cells as 32x32.
	square := vae.NewTensor(1, p.cfg.InChan, 32, 32)
	if _, err := p.dec.Apply(square); err != nil {
		t.Fatalf("the arena's own shape does not decode: %v", err)
	}
	tall := vae.NewTensor(1, p.cfg.InChan, 64, 16)
	if _, err := p.dec.Apply(tall); err == nil {
		t.Error("the same area in a 4:1 shape decoded; geomFor's rule could be the token count after all")
	} else {
		t.Logf("the same area, 4:1: %v", err)
	}
}

func fmtSize(w, h int) string { return fmt.Sprintf("%dx%d", w, h) }

func TestCheckSizeRejectsBeforeAnythingIsStaged(t *testing.T) {
	if err := checkSize(1024, 1024); err != nil {
		t.Fatal(err)
	}
	if err := checkSize(1024, 1020); err == nil {
		t.Error("1024x1020 was accepted")
	}
}

// ErrPromptTooLong is the one failure a caller can fix, so it is a value and
// not a message. A server maps it to a 400 and everything else here to a 500.
func TestPromptTooLongIsItsOwnError(t *testing.T) {
	err := promptTooLong(512, 700)
	if !errors.Is(err, ErrPromptTooLong) {
		t.Fatalf("%v does not wrap ErrPromptTooLong", err)
	}
	// And the message still carries both numbers, which is what the caller
	// needs in order to act on it.
	for _, want := range []string{"700", "512"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("%v does not mention %s", err, want)
		}
	}
}
