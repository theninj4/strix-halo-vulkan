package backend

import (
	"fmt"
	"image"
	"math"

	"strix-halo-vulkan/qimage/vae"
)

// Fitting a client's picture to the geometry a request resolved to -- the one
// piece of image processing this repository does that is not a model.
//
// It lives here rather than in `api` because it is arithmetic on pixels, and
// `api` is routing. It does not live in `zimage/vae` either: the encoder takes
// the image at the size being rendered and says so, because *how* a picture is
// fitted to a size is a policy question with three defensible answers, and a
// model that silently picked one would be making that decision where nobody
// could see it.
//
// **The policy is cover-crop**, which is CSS's `object-fit: cover`: scale until
// both sides are covered, then take the middle. The alternative, stretching,
// changes every shape in the picture, and an edit whose whole point is to keep
// the composition should not start by distorting it. The handler's default
// geometry is the input's own aspect rounded to a multiple of 16, so in the
// ordinary case the crop is at most fifteen pixels and this is a pure scale.

// fitImage resamples src to exactly dstW x dstH and returns it as the
// [1, 3, H, W] tensor in [-1, 1] the VAE's encoder takes.
//
// The filter is a separable triangle whose support widens when the image is
// being shrunk, which is the cheap form of area averaging: at a 4x reduction
// each output pixel is a weighted mean of about eight input pixels per axis
// rather than of two, so a downscale does not alias. Enlarging falls back to
// the same filter at radius 1, which is bilinear.
//
// It resamples in the encoder's own space rather than in linear light. That is
// a deliberate inheritance and not an oversight: the VAE was trained on sRGB
// pixels, so the picture it should be handed is the one an ordinary image
// viewer would show at this size.
func fitImage(src image.Image, dstW, dstH int) (*vae.Tensor, error) {
	if dstW <= 0 || dstH <= 0 {
		return nil, fmt.Errorf("backend: cannot fit an image to %dx%d", dstW, dstH)
	}
	b := src.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return nil, fmt.Errorf("backend: the input image is empty")
	}

	// The crop: the largest rectangle of the destination's shape that fits
	// inside the source, centred.
	scale := math.Max(float64(dstW)/float64(b.Dx()), float64(dstH)/float64(b.Dy()))
	cropW := min(b.Dx(), int(math.Round(float64(dstW)/scale)))
	cropH := min(b.Dy(), int(math.Round(float64(dstH)/scale)))
	x0 := b.Min.X + (b.Dx()-cropW)/2
	y0 := b.Min.Y + (b.Dy()-cropH)/2

	// The crop, read once into float32. Reading through image.Image's
	// interface is the slow part of this function -- an RGBA() call per pixel
	// -- and doing it once rather than once per output pixel is what keeps a
	// 4000x3000 upload off the clock.
	plane := cropW * cropH
	srcBuf := make([]float32, 3*plane)
	for y := 0; y < cropH; y++ {
		for x := 0; x < cropW; x++ {
			r, g, bl, _ := src.At(x0+x, y0+y).RGBA()
			i := y*cropW + x
			srcBuf[i] = float32(r) / 65535
			srcBuf[plane+i] = float32(g) / 65535
			srcBuf[2*plane+i] = float32(bl) / 65535
		}
	}

	// Horizontal, then vertical. Two passes of a separable filter is
	// O(n*radius) rather than the O(n*radius^2) a single 2-D pass would be,
	// and at a 4000-pixel input being fitted to 1024 the radius is 4.
	tmp := make([]float32, 3*dstW*cropH)
	hw := newWeights(cropW, dstW)
	for c := 0; c < 3; c++ {
		in := srcBuf[c*plane:]
		out := tmp[c*dstW*cropH:]
		for y := 0; y < cropH; y++ {
			hw.apply(in[y*cropW:(y+1)*cropW], out[y*dstW:(y+1)*dstW])
		}
	}

	dst := vae.NewTensor(1, 3, dstH, dstW)
	vw := newWeights(cropH, dstH)
	col := make([]float32, cropH)
	outCol := make([]float32, dstH)
	for c := 0; c < 3; c++ {
		in := tmp[c*dstW*cropH:]
		out := dst.Plane(0, c)
		for x := 0; x < dstW; x++ {
			for y := 0; y < cropH; y++ {
				col[y] = in[y*dstW+x]
			}
			vw.apply(col, outCol)
			for y := 0; y < dstH; y++ {
				// [0, 1] to [-1, 1] on the way out, and clamped: a triangle
				// filter has no negative lobes so this cannot overshoot, but
				// the encoder's contract is a range and stating it costs
				// nothing.
				v := outCol[y]*2 - 1
				out[y*dstW+x] = float32(math.Min(1, math.Max(-1, float64(v))))
			}
		}
	}
	return dst, nil
}

// weights is one axis of the separable resample: for each output sample, the
// contiguous run of input samples it draws from and their normalised weights.
// Precomputed once per axis, because every row of the image uses the same run.
type weights struct {
	start []int
	n     []int
	w     []float32
	max   int
}

func newWeights(srcN, dstN int) *weights {
	ratio := float64(srcN) / float64(dstN)
	// The filter's support in *source* units. Shrinking widens it, which is
	// what turns bilinear into an area average; enlarging leaves it at 1.
	support := math.Max(ratio, 1)
	maxN := int(math.Ceil(support*2)) + 2

	out := &weights{start: make([]int, dstN), n: make([]int, dstN), max: maxN}
	out.w = make([]float32, dstN*maxN)
	for i := 0; i < dstN; i++ {
		// The output sample's centre in source coordinates, at half-pixel
		// offsets so the mapping is symmetric: without the 0.5s a resample
		// shifts the picture by half an output pixel.
		centre := (float64(i)+0.5)*ratio - 0.5
		lo := int(math.Ceil(centre - support))
		hi := int(math.Floor(centre + support))
		var sum float32
		n := 0
		for j := lo; j <= hi && n < maxN; j++ {
			// Clamp at the edges rather than treating outside as zero, which
			// would darken the border.
			k := min(max(j, 0), srcN-1)
			t := math.Abs(float64(j)-centre) / support
			if t >= 1 {
				continue
			}
			wv := float32(1 - t)
			if n == 0 {
				out.start[i] = k
			} else if k != out.start[i]+n {
				// The clamp collapsed two taps onto one input sample; fold
				// the weight into the previous tap rather than skipping it,
				// so the row still sums to 1.
				out.w[i*maxN+n-1] += wv
				sum += wv
				continue
			}
			out.w[i*maxN+n] = wv
			sum += wv
			n++
		}
		if n == 0 {
			out.start[i], out.n[i] = min(max(int(math.Round(centre)), 0), srcN-1), 1
			out.w[i*maxN] = 1
			continue
		}
		out.n[i] = n
		for j := 0; j < n; j++ {
			out.w[i*maxN+j] /= sum
		}
	}
	return out
}

// apply resamples one line.
func (p *weights) apply(src, dst []float32) {
	for i := range dst {
		var sum float32
		base := i * p.max
		for j := 0; j < p.n[i]; j++ {
			sum += p.w[base+j] * src[p.start[i]+j]
		}
		dst[i] = sum
	}
}
