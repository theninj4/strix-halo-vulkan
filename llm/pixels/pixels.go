// Package pixels is qwen3.8-flash-next's image processor: an image in, the
// normalized [3, H, W] planes the vision tower's Patchify reads out
// (LLM-VISION.md V3).
//
// The processor is transformers' `Qwen2VLImageProcessor` on its torchvision
// backend, and every step of it is reproduced rather than approximated,
// because qimage's Q8.3 measured this tower amplifying an input perturbation
// by ~10³:
//
//  1. **RGB by dropping alpha.** `convert_rgb` is PIL's `convert("RGB")`,
//     which discards the alpha channel. It does not composite over white,
//     which is what Qwen-Image's pipeline does in front of the same tower.
//     The dump holds an RGBA card whose pixel_values equal its RGB twin's.
//  2. **smart_resize** to multiples of 32 (patch 16 x merge 2) inside
//     [min, max] pixels, using Python's `round()`, which rounds half to even.
//  3. **torch's uint8 antialiased bicubic.** On a CPU with AVX2 torchvision
//     hands a uint8 tensor to torch's native uint8 kernel. That kernel is
//     Pillow-SIMD's resampler: Pillow's coefficient plan (the same support
//     stretch on a downscale, the same bounds, the same normalization, cubic
//     a = -0.5), but quantized to **int16 at a per-axis precision**, the
//     largest that keeps the biggest weight under 2^15, where Pillow uses a
//     fixed 22 bits. Horizontal pass first, a uint8 intermediate, and an
//     axis whose size does not change is skipped. Pillow's own BICUBIC is
//     one level off on ~0.5% of samples. This is not.
//  4. **(x - 127.5) / 127.5 in float32**, which is the fused rescale and
//     normalize with mean = std = 0.5.
//
// JPEG is the one input this cannot make exact. Go's decoder and libjpeg's
// differ in the IDCT and the YCbCr conversion, so the same file can decode a
// level apart. The gates run on raw pixels. See LLM-VISION.md V3.
package pixels

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
)

// Processor is the processor's configuration: `preprocessor_config.json`.
type Processor struct {
	Factor    int // patch_size * merge_size, 32
	MinPixels int // size.shortest_edge
	MaxPixels int // size.longest_edge
}

// Default is qwen3.8-flash-next's preprocessor_config.json.
var Default = Processor{Factor: 32, MinPixels: 65536, MaxPixels: 16777216}

// roundHalfEven is Python 3's round() on a float.
func roundHalfEven(x float64) float64 { return math.RoundToEven(x) }

// SmartResize is transformers' smart_resize: the (height, width) an image is
// resized to.
func (p Processor) SmartResize(h, w int) (int, int, error) {
	if h <= 0 || w <= 0 {
		return 0, 0, fmt.Errorf("pixels: %dx%d image", w, h)
	}
	if float64(max(h, w))/float64(min(h, w)) > 200 {
		return 0, 0, fmt.Errorf("pixels: aspect ratio %dx%d is over 200", w, h)
	}
	f := float64(p.Factor)
	hb := int(roundHalfEven(float64(h)/f)) * p.Factor
	wb := int(roundHalfEven(float64(w)/f)) * p.Factor
	switch {
	case hb*wb > p.MaxPixels:
		beta := math.Sqrt(float64(h) * float64(w) / float64(p.MaxPixels))
		hb = max(p.Factor, int(math.Floor(float64(h)/beta/f))*p.Factor)
		wb = max(p.Factor, int(math.Floor(float64(w)/beta/f))*p.Factor)
	case hb*wb < p.MinPixels:
		beta := math.Sqrt(float64(p.MinPixels) / (float64(h) * float64(w)))
		hb = int(math.Ceil(float64(h)*beta/f)) * p.Factor
		wb = int(math.Ceil(float64(w)*beta/f)) * p.Factor
	}
	return hb, wb, nil
}

// RGB is an 8-bit image, interleaved RGB, row-major.
type RGB struct {
	W, H int
	Pix  []uint8
}

// FromImage is `convert("RGB")`: straight (non-premultiplied) colour with the
// alpha dropped. Going through NRGBA matters: Go's RGBA model is
// premultiplied, and reading a translucent pixel through it would darken it.
func FromImage(img image.Image) *RGB {
	b := img.Bounds()
	// Two stored forms carry straight colour that a round trip through Go's
	// premultiplied model would change: a palette entry with alpha (a fully
	// transparent one would come back black, where PIL keeps its RGB), and
	// 16-bit straight RGBA, which PIL narrows by taking the high byte.
	switch src := img.(type) {
	case *image.Paletted:
		out := &RGB{W: b.Dx(), H: b.Dy(), Pix: make([]uint8, b.Dx()*b.Dy()*3)}
		pal := make([][3]uint8, len(src.Palette))
		for i, c := range src.Palette {
			n := color.NRGBAModel.Convert(c).(color.NRGBA)
			if s, ok := c.(color.NRGBA); ok {
				n = s
			}
			pal[i] = [3]uint8{n.R, n.G, n.B}
		}
		for y := 0; y < out.H; y++ {
			for x := 0; x < out.W; x++ {
				idx := src.Pix[(y)*src.Stride+x]
				var p [3]uint8
				if int(idx) < len(pal) {
					p = pal[idx]
				}
				copy(out.Pix[(y*out.W+x)*3:], p[:])
			}
		}
		return out
	case *image.NRGBA64:
		out := &RGB{W: b.Dx(), H: b.Dy(), Pix: make([]uint8, b.Dx()*b.Dy()*3)}
		for y := 0; y < out.H; y++ {
			row := src.Pix[y*src.Stride:]
			for x := 0; x < out.W; x++ {
				for c := 0; c < 3; c++ {
					out.Pix[(y*out.W+x)*3+c] = row[x*8+c*2] // big-endian: the high byte
				}
			}
		}
		return out
	}
	n, ok := img.(*image.NRGBA)
	if !ok || n.Rect.Min != (image.Point{}) {
		n = image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(n, n.Rect, img, b.Min, draw.Src)
	}
	out := &RGB{W: b.Dx(), H: b.Dy(), Pix: make([]uint8, b.Dx()*b.Dy()*3)}
	for y := 0; y < out.H; y++ {
		src := n.Pix[y*n.Stride:]
		dst := out.Pix[y*out.W*3:]
		for x := 0; x < out.W; x++ {
			dst[x*3+0] = src[x*4+0]
			dst[x*3+1] = src[x*4+1]
			dst[x*3+2] = src[x*4+2]
		}
	}
	return out
}

// Planes resizes and normalizes: [3, H, W] float32, what Patchify reads.
func (p Processor) Planes(img *RGB) (planes []float32, h, w int, err error) {
	h, w, err = p.SmartResize(img.H, img.W)
	if err != nil {
		return nil, 0, 0, err
	}
	r := Resize(img, w, h)
	planes = make([]float32, 3*h*w)
	for i := 0; i < h*w; i++ {
		for c := 0; c < 3; c++ {
			planes[c*h*w+i] = (float32(r.Pix[i*3+c]) - 127.5) / 127.5
		}
	}
	return planes, h, w, nil
}

// Resize is torch's uint8 antialiased bicubic to w x h (see the package
// comment, step 3).
func Resize(src *RGB, w, h int) *RGB {
	out := src
	if w != src.W {
		out = resample(out, w, true)
	}
	if h != src.H {
		out = resample(out, h, false)
	}
	if out == src {
		out = &RGB{W: w, H: h, Pix: append([]uint8(nil), src.Pix...)}
	}
	return out
}

// cubic is the Keys kernel at a = -0.5, torch's and Pillow's bicubic.
func cubic(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// plan is one axis: for each output position the first input it reads, how
// many, and ksize int16-range weights at `prec` fractional bits.
type plan struct {
	ksize  int
	prec   uint
	bounds [][2]int
	kk     []int32
}

func newPlan(inSize, outSize int) plan {
	scale := float64(inSize) / float64(outSize)
	fs := math.Max(scale, 1)
	support := 2 * fs
	ksize := int(math.Ceil(support))*2 + 1
	p := plan{ksize: ksize, bounds: make([][2]int, outSize), kk: make([]int32, outSize*ksize)}
	k := make([]float64, outSize*ksize)
	maxW := math.Inf(-1)
	for xx := 0; xx < outSize; xx++ {
		center := (float64(xx) + 0.5) * scale
		// C's truncating cast; below zero the clamp makes it a floor.
		xmin := max(int(center-support+0.5), 0)
		n := min(int(center+support+0.5), inSize) - xmin
		row := k[xx*ksize : (xx+1)*ksize]
		var tot float64
		for x := 0; x < n; x++ {
			row[x] = cubic((float64(x+xmin) - center + 0.5) / fs)
			tot += row[x]
		}
		for x := 0; x < n; x++ {
			if tot != 0 {
				row[x] /= tot
			}
			maxW = math.Max(maxW, row[x])
		}
		p.bounds[xx] = [2]int{xmin, n}
	}
	// The largest precision whose biggest weight, doubled, stays under 2^15.
	for p.prec = 0; p.prec < 22; p.prec++ {
		if int(0.5+maxW*float64(int(1)<<(p.prec+1))) >= 1<<15 {
			break
		}
	}
	one := float64(int(1) << p.prec)
	for i, v := range k {
		if v < 0 {
			p.kk[i] = int32(-0.5 + v*one)
		} else {
			p.kk[i] = int32(0.5 + v*one)
		}
	}
	return p
}

// resample runs one pass along x (horizontal) or y, into uint8.
func resample(src *RGB, size int, horizontal bool) *RGB {
	var out *RGB
	var in int
	if horizontal {
		out = &RGB{W: size, H: src.H}
		in = src.W
	} else {
		out = &RGB{W: src.W, H: size}
		in = src.H
	}
	out.Pix = make([]uint8, out.W*out.H*3)
	p := newPlan(in, size)
	round := int32(1) << (p.prec - 1)
	clip := func(v int32) uint8 {
		v >>= p.prec
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return uint8(v)
	}
	for o := 0; o < size; o++ {
		xmin, n := p.bounds[o][0], p.bounds[o][1]
		kk := p.kk[o*p.ksize:][:n]
		if horizontal {
			for y := 0; y < src.H; y++ {
				row := src.Pix[y*src.W*3:]
				var r, g, b int32 = round, round, round
				for j, w := range kk {
					s := row[(xmin+j)*3:]
					r += int32(s[0]) * w
					g += int32(s[1]) * w
					b += int32(s[2]) * w
				}
				d := out.Pix[(y*out.W+o)*3:]
				d[0], d[1], d[2] = clip(r), clip(g), clip(b)
			}
			continue
		}
		for x := 0; x < src.W; x++ {
			var r, g, b int32 = round, round, round
			for j, w := range kk {
				s := src.Pix[((xmin+j)*src.W+x)*3:]
				r += int32(s[0]) * w
				g += int32(s[1]) * w
				b += int32(s[2]) * w
			}
			d := out.Pix[(o*out.W+x)*3:]
			d[0], d[1], d[2] = clip(r), clip(g), clip(b)
		}
	}
	return out
}
