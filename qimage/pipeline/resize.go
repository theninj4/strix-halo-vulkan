package pipeline

// Pillow's resampler, ported exactly — IMAGE.md Q8.3.
//
// Every condition image a client sends is resized before either encoder sees
// it (CalcDimensions decides to what), and the pipeline does that with
// `VaeImageProcessor.resize`, which is `PIL.Image.resize(resample=LANCZOS)`
// — the processor's default, not bicubic and not torch's interpolate.
//
// **This has to be bit-exact, and that is a measured requirement rather than
// a preference.** Q8.3 found the vision tower amplifies an input
// perturbation by about 10³: compositing alpha over white in float instead
// of on the uint8 levels PIL uses moves half the pixels by half a level, and
// moves the prompt embedding by rel 11. A resampler that is merely close
// would do the same thing. So this reproduces Pillow's arithmetic rather
// than its intent — the same coefficient rounding into 22-bit fixed point,
// the same int32 accumulator with its rounding term, the same clip and shift,
// the same two passes with a **uint8 intermediate** between them, and the
// same skipping of a pass whose axis does not change.
//
// The filter is Lanczos with support 3, written the way Pillow writes it
// (including the asymmetric `-3 <= x < 3`, which is a no-op at the
// endpoints but is what the reference says).

import (
	"image"
	"math"
)

// precisionBits is Pillow's PRECISION_BITS: 32 - 8 - 2, the fixed-point
// shift its 8-bit resampler accumulates in.
const precisionBits = 32 - 8 - 2

// lanczosSupport is the filter's radius before the scale stretch.
const lanczosSupport = 3.0

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	x *= math.Pi
	return math.Sin(x) / x
}

func lanczos(x float64) float64 {
	if -3.0 <= x && x < 3.0 {
		return sinc(x) * sinc(x/3)
	}
	return 0
}

// roundUpFixed is Pillow's ROUND_UP applied to a coefficient scaled into
// fixed point: away from zero on a half, which is *not* Go's math.Round on
// negatives and not break-to-even either.
func roundUpFixed(k float64) int32 {
	f := k * (1 << precisionBits)
	if f >= 0 {
		return int32(f + 0.5)
	}
	return int32(f - 0.5)
}

// clip8 is the shift and clamp Pillow applies to every accumulated pixel.
func clip8(in int32) uint8 {
	v := in >> precisionBits
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// coeffs is one axis's resampling plan: for each output position, the first
// input index it reads, how many it reads, and the fixed-point weights.
type coeffs struct {
	ksize  int
	bounds [][2]int // {min, count}
	kk     []int32  // ksize weights per output position
}

// precomputeCoeffs is Pillow's precompute_coeffs over a full-image box.
//
// The filter is stretched by the *downscale* factor and not by the upscale
// one — that asymmetry is what makes Pillow's downscale antialiased and its
// upscale a plain interpolation, and it is the single thing most likely to
// be dropped by a port that reasons about the filter instead of copying it.
func precomputeCoeffs(inSize, outSize int) coeffs {
	scale := float64(inSize) / float64(outSize)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := lanczosSupport * filterscale
	ksize := int(math.Ceil(support))*2 + 1

	c := coeffs{ksize: ksize, bounds: make([][2]int, outSize), kk: make([]int32, outSize*ksize)}
	k := make([]float64, ksize)
	for xx := 0; xx < outSize; xx++ {
		center := (float64(xx) + 0.5) * scale
		ss := 1.0 / filterscale
		// C's cast truncates toward zero; below zero the clamp makes that
		// agree with a floor, so this is Pillow's expression verbatim.
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inSize {
			xmax = inSize
		}
		xmax -= xmin

		ww := 0.0
		for x := 0; x < xmax; x++ {
			w := lanczos((float64(x+xmin) - center + 0.5) * ss)
			k[x] = w
			ww += w
		}
		for x := 0; x < xmax; x++ {
			if ww != 0 {
				k[x] /= ww
			}
		}
		for x := xmax; x < ksize; x++ {
			k[x] = 0
		}
		for x := 0; x < ksize; x++ {
			c.kk[xx*ksize+x] = roundUpFixed(k[x])
		}
		c.bounds[xx] = [2]int{xmin, xmax}
	}
	return c
}

// Resize resamples an 8-bit RGBA image to w x h exactly as
// PIL.Image.resize(resample=LANCZOS) does.
//
// **An RGBA image is not resampled as four independent channels.**
// `Image.resize` converts it to "RGBa" — premultiplied alpha — resamples
// that, and converts back, so the colour of a transparent pixel cannot bleed
// into its neighbours. Both conversions are 8-bit and lossy, and skipping
// them leaves 28% of the samples up to four levels out, which is how this
// port first failed its gate. The premultiply rounds with Pillow's
// MULDIV255; the reverse is an integer divide that truncates.
//
// An axis whose size does not change is left alone rather than run through
// the filter, because Pillow skips that pass too and the pass is lossy: it
// quantizes back to 8 bits.
func Resize(src *image.NRGBA, w, h int) *image.NRGBA {
	sw := src.Rect.Dx()
	sh := src.Rect.Dy()
	if sw == w && sh == h {
		out := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			copy(out.Pix[y*out.Stride:], src.Pix[y*src.Stride:y*src.Stride+w*4])
		}
		return out
	}
	// At least one axis differs, so at least one pass runs and the result is
	// never the input.
	mid := premultiply(src)
	if sw != w {
		mid = resampleHorizontal(mid, w)
	}
	if sh != h {
		mid = resampleVertical(mid, h)
	}
	return unpremultiply(mid)
}

// premultiply is Pillow's "RGBA" -> "RGBa" conversion. Fully opaque and
// fully transparent pixels pass through untouched, which is not merely an
// optimisation: it is what keeps an opaque image bit-identical through the
// round trip.
func premultiply(src *image.NRGBA) *image.NRGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		in := src.Pix[y*src.Stride : y*src.Stride+w*4]
		dst := out.Pix[y*out.Stride : y*out.Stride+w*4]
		for x := 0; x < w*4; x += 4 {
			a := uint32(in[x+3])
			if a == 255 || a == 0 {
				copy(dst[x:x+4], in[x:x+4])
				continue
			}
			dst[x+0] = muldiv255(uint32(in[x+0]), a)
			dst[x+1] = muldiv255(uint32(in[x+1]), a)
			dst[x+2] = muldiv255(uint32(in[x+2]), a)
			dst[x+3] = in[x+3]
		}
	}
	return out
}

// unpremultiply is Pillow's "RGBa" -> "RGBA" conversion: 255*c/a with an
// integer divide, clipped.
func unpremultiply(src *image.NRGBA) *image.NRGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		in := src.Pix[y*src.Stride : y*src.Stride+w*4]
		dst := out.Pix[y*out.Stride : y*out.Stride+w*4]
		for x := 0; x < w*4; x += 4 {
			a := uint32(in[x+3])
			if a == 255 || a == 0 {
				copy(dst[x:x+4], in[x:x+4])
				continue
			}
			for c := 0; c < 3; c++ {
				v := 255 * uint32(in[x+c]) / a
				if v > 255 {
					v = 255
				}
				dst[x+c] = uint8(v)
			}
			dst[x+3] = in[x+3]
		}
	}
	return out
}

// muldiv255 is Pillow's MULDIV255: a*b/255 rounded, without a divide.
func muldiv255(a, b uint32) uint8 {
	t := a*b + 128
	return uint8(((t >> 8) + t) >> 8)
}

func resampleHorizontal(src *image.NRGBA, w int) *image.NRGBA {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	c := precomputeCoeffs(sw, w)
	out := image.NewNRGBA(image.Rect(0, 0, w, sh))
	for y := 0; y < sh; y++ {
		row := src.Pix[y*src.Stride : y*src.Stride+sw*4]
		dst := out.Pix[y*out.Stride : y*out.Stride+w*4]
		for xx := 0; xx < w; xx++ {
			xmin, count := c.bounds[xx][0], c.bounds[xx][1]
			k := c.kk[xx*c.ksize:]
			for ch := 0; ch < 4; ch++ {
				ss := int32(1) << (precisionBits - 1)
				for x := 0; x < count; x++ {
					ss += int32(row[(x+xmin)*4+ch]) * k[x]
				}
				dst[xx*4+ch] = clip8(ss)
			}
		}
	}
	return out
}

func resampleVertical(src *image.NRGBA, h int) *image.NRGBA {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	c := precomputeCoeffs(sh, h)
	out := image.NewNRGBA(image.Rect(0, 0, sw, h))
	for yy := 0; yy < h; yy++ {
		ymin, count := c.bounds[yy][0], c.bounds[yy][1]
		k := c.kk[yy*c.ksize:]
		dst := out.Pix[yy*out.Stride : yy*out.Stride+sw*4]
		for x := 0; x < sw; x++ {
			for ch := 0; ch < 4; ch++ {
				ss := int32(1) << (precisionBits - 1)
				for y := 0; y < count; y++ {
					ss += int32(src.Pix[(y+ymin)*src.Stride+x*4+ch]) * k[y]
				}
				dst[x*4+ch] = clip8(ss)
			}
		}
	}
	return out
}
