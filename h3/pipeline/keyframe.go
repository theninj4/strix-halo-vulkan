package pipeline

// Keyframes (fl2va, VIDEO.md M10): the pictures a request pins the video's
// first and/or last frame to.
//
// A keyframe reaches the transformer twice. The conditioner sees it as a
// Qwen3-VL vision block inside the prompt (h3/textenc's Presentation), and
// the transformer sees it as anchor rows: the video VAE's encoding of the
// picture, a posterior draw under its own seed, noised to t = 0.999 and
// packed ahead of the generated video rows, where it stays for every step.
// Both copies are of the same picture on the same canvas, prepared here as
// MiniMaxH3ResizeStep prepares it.

import (
	"fmt"
	"image"
	"image/draw"
	"math"

	"strix-halo-vulkan/h3/plan"
	qpipe "strix-halo-vulkan/qimage/pipeline"
)

// keyframeList is a request's keyframes in packed order — first, then last
// — with their anchors.
func (req *Request) keyframeList() ([]image.Image, []plan.Anchor) {
	var imgs []image.Image
	var anchors []plan.Anchor
	if req.First != nil {
		imgs, anchors = append(imgs, req.First), append(anchors, plan.First)
	}
	if req.Last != nil {
		imgs, anchors = append(imgs, req.Last), append(anchors, plan.Last)
	}
	return imgs, anchors
}

// opaque is a picture as PIL's convert("RGB") leaves it: every pixel's
// colour with its alpha dropped, not composited over anything.
func opaque(src image.Image) *image.NRGBA {
	b := src.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	// An NRGBA picture is copied as stored: going through draw's
	// premultiplied colour model would lose the colour of transparent
	// pixels, which convert("RGB") keeps.
	if n, ok := src.(*image.NRGBA); ok {
		for y := 0; y < b.Dy(); y++ {
			copy(out.Pix[y*out.Stride:], n.Pix[n.PixOffset(b.Min.X, b.Min.Y+y):][:b.Dx()*4])
		}
	} else {
		draw.Draw(out, out.Rect, src, b.Min, draw.Src)
	}
	for i := 3; i < len(out.Pix); i += 4 {
		out.Pix[i] = 255
	}
	return out
}

// PlaceKeyframe puts one keyframe on an h × w canvas and returns it as
// H·W·3 RGB bytes. The first of a request's keyframes (the geometry anchor,
// whose aspect the canvas follows by default) is *stretched* onto it with
// PIL's LANCZOS; a second one is cover-cropped: scaled to cover the canvas
// with Python's rounding, LANCZOS, then centre-cropped as
// `(size − canvas) // 2`.
func PlaceKeyframe(src image.Image, index, h, w int) ([]byte, error) {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= 0 || sh <= 0 {
		return nil, fmt.Errorf("h3: an empty keyframe")
	}
	img := opaque(src)
	var placed *image.NRGBA
	if index == 0 || (sw == w && sh == h) {
		placed = qpipe.Resize(img, w, h)
	} else {
		scale := math.Max(float64(w)/float64(sw), float64(h)/float64(sh))
		rw := max(w, int(math.RoundToEven(float64(sw)*scale)))
		rh := max(h, int(math.RoundToEven(float64(sh)*scale)))
		resized := qpipe.Resize(img, rw, rh)
		left, top := max(0, (rw-w)/2), max(0, (rh-h)/2)
		placed = resized.SubImage(image.Rect(left, top, left+w, top+h)).(*image.NRGBA)
	}
	out := make([]byte, 0, h*w*3)
	for y := 0; y < h; y++ {
		row := placed.Pix[placed.PixOffset(placed.Rect.Min.X, placed.Rect.Min.Y+y):]
		for x := 0; x < w; x++ {
			out = append(out, row[x*4], row[x*4+1], row[x*4+2])
		}
	}
	return out, nil
}
