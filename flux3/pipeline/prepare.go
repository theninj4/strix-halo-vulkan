package pipeline

import (
	"fmt"
	"image"
	"image/color"
	"math"

	qpipe "strix-halo-vulkan/qimage/pipeline"
)

// Reference preparation for edits, ported from BFL's pipeline.py
// (reference/flux3/bfl; X0 finding 6: the server follows BFL, not
// diffusers). One reference is fitted under a pixel cap with Lanczos,
// reflect-padded to the 16 grid around its centre, and the output cropped
// back to the fitted box; several are each fitted under 1 MiP (sides
// floored) and centre-cropped to the grid. Gated bit-exact against BFL's own
// functions (TestPrepare, reference/dump_flux3_prepare.py).
//
// The resampler is qimage/pipeline.Resize, Pillow's LANCZOS to the 8-bit
// level (Q8.3). BFL's convert("RGB") drops alpha without compositing, so a
// picture is made opaque first, which Resize then passes through untouched.
//
// Not ported: ImageOps.exif_transpose. The decoders behind the API do not
// read EXIF, so a rotated phone photo is edited as it is stored.

const (
	// MaxReferences is the model's limit on an edit's pictures.
	MaxReferences = 10
	// MatchMaxPixels is BFL's cap on one reference's canvas, 4 MiP. The
	// server passes its staging's area instead (smaller: the decoder's
	// ceiling is 1232²), because the canvas is the output's default size.
	MatchMaxPixels = 4 << 20
	// MultiReferenceMaxPixels caps each picture of a multi-reference edit.
	MultiReferenceMaxPixels = 1 << 20
	minSide, maxSide        = 64, 4096
)

// Reference is one prepared picture on the 16 grid and, for a padded single
// reference, the fitted box inside it (empty when there is none to crop).
type Reference struct {
	Image *image.NRGBA
	Box   image.Rectangle
}

// opaque copies an image to NRGBA with every alpha 255: PIL's
// convert("RGB"), which keeps the (straight) colour channels and drops alpha.
func opaque(src image.Image) *image.NRGBA {
	b := src.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			c := color.NRGBAModel.Convert(src.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			i := y*out.Stride + x*4
			out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = c.R, c.G, c.B, 255
		}
	}
	return out
}

func ceil16(v int) int { return (v + SizeMultiple - 1) / SizeMultiple * SizeMultiple }

// ReferenceCanvas is BFL's reference_canvas with the cap a parameter: the
// fitted size of a w x h picture and its canvas (the fit rounded up to 16,
// sides 64-4096), the canvas at most maxPixels.
func ReferenceCanvas(w, h, maxPixels int) (fit, canvas image.Point) {
	scale := math.Min(1, math.Min(math.Sqrt(float64(maxPixels)/float64(w*h)), float64(maxSide)/float64(max(w, h))))
	for {
		// Python's round() is half-to-even.
		fit = image.Pt(max(1, int(math.RoundToEven(float64(w)*scale))), max(1, int(math.RoundToEven(float64(h)*scale))))
		canvas = image.Pt(min(maxSide, max(minSide, ceil16(fit.X))), min(maxSide, max(minSide, ceil16(fit.Y))))
		if canvas.X*canvas.Y <= maxPixels {
			return fit, canvas
		}
		scale *= 0.995
	}
}

// PrepareReference is BFL's prepare_reference with the cap a parameter: the
// picture fitted by ReferenceCanvas and padded to its canvas, centred, by
// numpy's "reflect" (or "symmetric" when a side is 16 or less, where reflect
// would need a pad shorter than the side).
func PrepareReference(src image.Image, maxPixels int) (Reference, error) {
	img := opaque(src)
	w, h := img.Rect.Dx(), img.Rect.Dy()
	if w == 0 || h == 0 {
		return Reference{}, fmt.Errorf("pipeline: an empty reference image")
	}
	fit, canvas := ReferenceCanvas(w, h, maxPixels)
	if fit.X != w || fit.Y != h {
		img = qpipe.Resize(img, fit.X, fit.Y)
		w, h = fit.X, fit.Y
	}
	if canvas.X == w && canvas.Y == h {
		return Reference{Image: img}, nil
	}
	left, top := (canvas.X-w)/2, (canvas.Y-h)/2
	symmetric := min(w, h) <= SizeMultiple
	out := image.NewNRGBA(image.Rect(0, 0, canvas.X, canvas.Y))
	for y := 0; y < canvas.Y; y++ {
		sy := mirror(y-top, h, symmetric)
		for x := 0; x < canvas.X; x++ {
			sx := mirror(x-left, w, symmetric)
			copy(out.Pix[y*out.Stride+x*4:][:4], img.Pix[sy*img.Stride+sx*4:])
		}
	}
	return Reference{Image: out, Box: image.Rect(left, top, left+w, top+h)}, nil
}

// mirror maps an index outside [0, n) back into it as numpy's pad modes do,
// pads longer than the side included (numpy pads in repeated reflections,
// which is the periodic extension): "reflect" mirrors about the edge pixels
// (period 2(n-1)), "symmetric" repeats them (period 2n).
func mirror(i, n int, symmetric bool) int {
	if symmetric {
		p := 2 * n
		i = ((i % p) + p) % p
		if i >= n {
			i = p - 1 - i
		}
		return i
	}
	if n == 1 {
		return 0
	}
	p := 2 * (n - 1)
	i = ((i % p) + p) % p
	if i >= n {
		i = p - i
	}
	return i
}

// PrepareMultiReference is BFL's prepare_multi_reference with the cap a
// parameter: past maxPixels the picture is scaled to it (sides floored, as
// Python's int()), then centre-cropped to the 16 grid.
func PrepareMultiReference(src image.Image, maxPixels int) (Reference, error) {
	img := opaque(src)
	w, h := img.Rect.Dx(), img.Rect.Dy()
	if w*h > maxPixels {
		scale := math.Sqrt(float64(maxPixels) / float64(w*h))
		w, h = int(float64(w)*scale), int(float64(h)*scale)
		img = qpipe.Resize(img, w, h)
	}
	cw, ch := w/SizeMultiple*SizeMultiple, h/SizeMultiple*SizeMultiple
	if cw == 0 || ch == 0 {
		return Reference{}, fmt.Errorf("pipeline: a %dx%d reference has no 16x16 tile to keep", w, h)
	}
	left, top := (w-cw)/2, (h-ch)/2
	out := image.NewNRGBA(image.Rect(0, 0, cw, ch))
	for y := 0; y < ch; y++ {
		copy(out.Pix[y*out.Stride:], img.Pix[(y+top)*img.Stride+left*4:][:cw*4])
	}
	return Reference{Image: out}, nil
}

// PrepareReferences prepares an edit's pictures as BFL's generate.py does:
// one by PrepareReference under maxPixels, several by PrepareMultiReference
// each under min(1 MiP, maxPixels).
func PrepareReferences(srcs []image.Image, maxPixels int) ([]Reference, error) {
	if len(srcs) > MaxReferences {
		return nil, fmt.Errorf("pipeline: %d reference images; the model takes at most %d", len(srcs), MaxReferences)
	}
	out := make([]Reference, len(srcs))
	for i, s := range srcs {
		var err error
		if len(srcs) == 1 {
			out[i], err = PrepareReference(s, maxPixels)
		} else {
			out[i], err = PrepareMultiReference(s, min(MultiReferenceMaxPixels, maxPixels))
		}
		if err != nil {
			return nil, fmt.Errorf("reference %d: %w", i, err)
		}
	}
	return out, nil
}

// ToTensor is BFL's to_tensor: 8-bit RGB to [1, 3, H, W] in [-1, 1] as
// x / 127.5 - 1.
func ToTensor(img *image.NRGBA) []float32 {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	out := make([]float32, 3*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p := img.Pix[y*img.Stride+x*4:]
			for c := 0; c < 3; c++ {
				out[(c*h+y)*w+x] = float32(p[c])/127.5 - 1
			}
		}
	}
	return out
}
