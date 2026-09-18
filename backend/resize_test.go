package backend

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// solid builds a w x h image of one colour.
func solid(w, h int, c color.RGBA) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

// TestFitImageIsAPartitionOfUnity is the property that makes a resampler
// correct before it is good: a flat image has to come out flat, at the same
// value, at every scale.
//
// It is the one check that catches a filter whose weights do not sum to 1 --
// which is what an unnormalised kernel, a dropped edge tap or a wrong support
// all look like, and all of which produce a picture that is merely slightly
// dark rather than obviously broken.
func TestFitImageIsAPartitionOfUnity(t *testing.T) {
	const v = 0.25 // in [-1, 1], i.e. 159/255
	px := uint8(math.Round((v + 1) * 127.5))
	want := float64(px)/127.5 - 1

	for _, c := range []struct{ sw, sh, dw, dh int }{
		{1024, 1024, 256, 256}, // a 4x reduction, where area averaging matters
		{100, 100, 512, 512},   // enlargement
		{640, 480, 512, 512},   // a crop as well as a scale
		{257, 129, 64, 32},     // odd sides, so the crop is not symmetric
		{16, 16, 16, 16},       // identity
	} {
		img := solid(c.sw, c.sh, color.RGBA{px, px, px, 255})
		got, err := fitImage(img, c.dw, c.dh)
		if err != nil {
			t.Fatalf("%dx%d -> %dx%d: %v", c.sw, c.sh, c.dw, c.dh, err)
		}
		if got.H != c.dh || got.W != c.dw || got.C != 3 {
			t.Errorf("%dx%d -> %dx%d gave %s", c.sw, c.sh, c.dw, c.dh, got)
			continue
		}
		worst := 0.0
		for _, g := range got.Data {
			if d := math.Abs(float64(g) - want); d > worst {
				worst = d
			}
		}
		// 1e-5 rather than exact: the weights are normalised in float32.
		if worst > 1e-5 {
			t.Errorf("%dx%d -> %dx%d: a flat %.4f came out up to %.3g away", c.sw, c.sh, c.dw, c.dh, want, worst)
		}
	}
}

// TestFitImageCentresTheCrop pins the policy rather than the arithmetic: a
// wide picture fitted to a square keeps its middle, and loses the same amount
// from each end.
//
// The negative control is built in -- a crop anchored at the origin, which is
// the one-character version of this bug, puts the marked column somewhere else
// entirely and the assertion is on *where* the marker lands.
func TestFitImageCentresTheCrop(t *testing.T) {
	// A 400x100 image: black, with one white column dead centre.
	img := image.NewRGBA(image.Rect(0, 0, 400, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 400; x++ {
			c := color.RGBA{0, 0, 0, 255}
			if x >= 198 && x < 202 {
				c = color.RGBA{255, 255, 255, 255}
			}
			img.Set(x, y, c)
		}
	}
	got, err := fitImage(img, 100, 100)
	if err != nil {
		t.Fatal(err)
	}
	// The brightest column of the result.
	best, bestX := float32(-2), -1
	row := got.Plane(0, 0)
	for x := 0; x < got.W; x++ {
		if v := row[50*got.W+x]; v > best {
			best, bestX = v, x
		}
	}
	if bestX < 48 || bestX > 51 {
		t.Errorf("the centre column landed at x=%d of %d; a centred crop puts it at 49-50", bestX, got.W)
	}
	if best < 0.9 {
		t.Errorf("the white column came out at %.3f, not near +1; the filter is smearing it away", best)
	}
}

// TestFitImageRefusesNothing covers the two degenerate inputs, which would
// otherwise divide by zero rather than answer.
func TestFitImageRefusesNothing(t *testing.T) {
	if _, err := fitImage(solid(8, 8, color.RGBA{}), 0, 16); err == nil {
		t.Error("a zero-width target was accepted")
	}
	if _, err := fitImage(image.NewRGBA(image.Rect(0, 0, 0, 0)), 16, 16); err == nil {
		t.Error("an empty source was accepted")
	}
}
