package ocr

import (
	"fmt"
	"math"

	"strix-halo-vulkan/llm/pixels"
)

// The processor's geometry (preprocessor_config.json): patch 14, merge 2,
// so every side is a multiple of 28.
const (
	PatchSize = 14
	MergeSize = 2
	Factor    = PatchSize * MergeSize
	// PatchElems is one patch's values: 3 channels x 14 x 14 (no temporal
	// axis; temporal_patch_size is 1).
	PatchElems = 3 * PatchSize * PatchSize

	// DefaultMinPixels and DefaultMaxPixels are the checkpoint's bounds:
	// 144 to 5,120 patches, 36 to 1,280 image tokens.
	DefaultMinPixels = 112896
	DefaultMaxPixels = 1003520
	// SpottingMaxPixels is the card's and PaddleX's bound for "Spotting:".
	SpottingMaxPixels = 1605632
)

// SmartResize is PaddleOCR-VL's smart_resize (the HF port's modular file):
// the size an h x w image is resized to.
//
// It is Qwen2-VL's rule with one step in front: **a side under 28 is raised
// to 28 and the other scaled to keep the aspect**, where Qwen2-VL's refuses
// the image. A thin formula or a one-line crop from the layout model reaches
// this branch, so it is not a corner case here. Rounding is Python's round(),
// half to even.
func SmartResize(h, w, minPixels, maxPixels int) (int, int, error) {
	if h <= 0 || w <= 0 {
		return 0, 0, fmt.Errorf("ocr: %dx%d image", w, h)
	}
	if h < Factor {
		w = int(math.RoundToEven(float64(w*Factor) / float64(h)))
		h = Factor
	}
	if w < Factor {
		h = int(math.RoundToEven(float64(h*Factor) / float64(w)))
		w = Factor
	}
	return pixels.Processor{Factor: Factor, MinPixels: minPixels, MaxPixels: maxPixels}.SmartResize(h, w)
}

// Image is a processed image: the patches the tower's patch embedding reads,
// in raster order, and the patch grid.
type Image struct {
	// Patches is [GridH*GridW, 3, 14, 14], row-major over the patch grid.
	Patches      []float32
	GridH, GridW int
}

// MergedH and MergedW are the grid of image tokens the language model sees.
func (im *Image) MergedH() int { return im.GridH / MergeSize }
func (im *Image) MergedW() int { return im.GridW / MergeSize }

// Process is the HF processor's PIL backend: smart_resize, Pillow BICUBIC on
// the 8-bit image, then rescale and normalise with its arithmetic, which is
// two steps and not the fused one: x·(1/255) in float64 narrowed to
// float32, then (v - 0.5) / 0.5 in float32.
func Process(img *pixels.RGB, minPixels, maxPixels int) (*Image, error) {
	h, w, err := SmartResize(img.H, img.W, minPixels, maxPixels)
	if err != nil {
		return nil, err
	}
	r := pixels.ResizePillow(img, w, h)
	gh, gw := h/PatchSize, w/PatchSize
	out := &Image{Patches: make([]float32, gh*gw*PatchElems), GridH: gh, GridW: gw}
	var lut [256]float32
	for v := range lut {
		s := float32(float64(v) * 0.00392156862745098)
		lut[v] = (s - 0.5) / 0.5
	}
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			dst := out.Patches[(py*gw+px)*PatchElems:]
			for c := 0; c < 3; c++ {
				for y := 0; y < PatchSize; y++ {
					row := r.Pix[((py*PatchSize+y)*w+px*PatchSize)*3:]
					for x := 0; x < PatchSize; x++ {
						dst[(c*PatchSize+y)*PatchSize+x] = lut[row[x*3+c]]
					}
				}
			}
		}
	}
	return out, nil
}
