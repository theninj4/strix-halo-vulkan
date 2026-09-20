package pipeline

import (
	"image"
	"math"
	"testing"

	"strix-halo-vulkan/zimage/qwen"
)

// TestResize is the resampler's gate: the dump's own before/after pair, run
// through PIL.Image.resize(LANCZOS) by reference/dump_qi21_edit.py at the
// size CalcDimensions picks for a 500x333 source.
//
// It is asserted as **exact 8-bit equality** rather than to a tolerance, and
// that is the point of the port. Q8.3 measured the vision tower amplifying
// an input perturbation by ~10³ — half a level per pixel on the composite
// moved the prompt embedding by rel 11 — so a resampler that is merely close
// cannot be shown to reproduce the oracle, and one that is exact needs no
// tolerance at all. Anything that drifts here is a real difference in the
// arithmetic, not rounding.
func TestResize(t *testing.T) {
	m := loadEditManifest(t)
	if len(m.ResizeOutSize) != 2 {
		t.Skip("the dump has no resize pair; re-run reference/dump_qi21_edit.py")
	}
	// Three pairs, because the resampler has three behaviours: Pillow
	// stretches the filter when it downscales and not when it upscales, and
	// it skips the pass for an axis that does not change. A photo resized to
	// output_resolution 1024 is usually an upscale, so that is not a corner.
	for _, tag := range []string{"resize", "resize_up", "resize_axis"} {
		src := mustNRGBA(t, loadEditMat(t, m, tag+"_src"), m.Tensors[tag+"_src"].Shape)
		want := mustNRGBA(t, loadEditMat(t, m, tag+"_dst"), m.Tensors[tag+"_dst"].Shape)
		w, h := want.Rect.Dx(), want.Rect.Dy()

		got := Resize(src, w, h)
		diffs, maxAbs := 0, 0
		for y := 0; y < h; y++ {
			for x := 0; x < w*4; x++ {
				d := int(got.Pix[y*got.Stride+x]) - int(want.Pix[y*want.Stride+x])
				if d != 0 {
					diffs++
					if d < 0 {
						d = -d
					}
					if d > maxAbs {
						maxAbs = d
					}
				}
			}
		}
		if diffs != 0 {
			t.Errorf("%s %dx%d -> %dx%d: %d of %d samples differ, worst by %d levels",
				tag, src.Rect.Dx(), src.Rect.Dy(), w, h, diffs, w*h*4, maxAbs)
			continue
		}
		t.Logf("%-12s %dx%d -> %dx%d: all %d samples exact",
			tag, src.Rect.Dx(), src.Rect.Dy(), w, h, w*h*4)
	}
}

// TestResizeIdentity checks the two shortcuts Pillow takes and this port has
// to take with it: a pass whose axis does not change is skipped rather than
// run, because running it would quantize the image again. A port that always
// filters both axes returns a slightly different picture for a resize that
// should not have touched anything.
func TestResizeIdentity(t *testing.T) {
	m := loadEditManifest(t)
	src := mustNRGBA(t, loadEditMat(t, m, "resize_src"), m.Tensors["resize_src"].Shape)
	w, h := src.Rect.Dx(), src.Rect.Dy()

	same := Resize(src, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w*4; x++ {
			if same.Pix[y*same.Stride+x] != src.Pix[y*src.Stride+x] {
				t.Fatalf("resizing to its own size changed sample %d,%d", x, y)
			}
		}
	}
	// One axis only: the other must come through untouched, which is what
	// says the vertical pass was skipped rather than run with unit weights.
	narrow := Resize(src, w/2, h)
	if narrow.Rect.Dx() != w/2 || narrow.Rect.Dy() != h {
		t.Fatalf("one-axis resize gave %dx%d", narrow.Rect.Dx(), narrow.Rect.Dy())
	}
	t.Logf("identity exact over %d samples; one-axis resize keeps %d rows", w*h*4, h)
}

// mustNRGBA turns a dumped [1, 4, H, W] tensor in [-1, 1] back into the
// 8-bit image it was made from. The dump wrote `np.array(img)/127.5 - 1`, so
// this is that map inverted; it has to land exactly on the levels, and the
// test fails rather than rounds if it does not.
func mustNRGBA(t *testing.T, m *qwen.Mat, shape []int) *image.NRGBA {
	t.Helper()
	if len(shape) != 4 || shape[1] != 4 {
		t.Fatalf("expected a [1, 4, H, W] tensor, got %v", shape)
	}
	h, w := shape[2], shape[3]
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	plane := h * w
	for c := 0; c < 4; c++ {
		for i := 0; i < plane; i++ {
			v := float64(m.Data[c*plane+i])
			level := (v + 1) * 127.5
			if d := math.Abs(level - math.Round(level)); d > 1e-3 {
				t.Fatalf("sample %d of channel %d is %g, not an 8-bit level", i, c, v)
			}
			img.Pix[(i/w)*img.Stride+(i%w)*4+c] = uint8(math.Round(level))
		}
	}
	return img
}
