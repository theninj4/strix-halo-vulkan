package page

import (
	"fmt"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr"
)

// imageLabels are the pipeline's image_labels at its defaults (no chart or
// seal recognition, no OCR for image blocks): IMAGE_LABELS + chart + seal.
// These regions are never recognised, only cropped for the markdown.
var imageLabels = map[string]bool{"image": true, "header_image": true, "footer_image": true, "chart": true, "seal": true}

// Block is a region after filtering, cropping and merging
// (paddleocr_vl/pipeline.py's blocks_for_img).
type Block struct {
	Label string
	Box   [4]int
	// Img is the crop; nil for a block merged into the one before it.
	Img *pixels.RGB
	// GroupID is the first block of a merged group, or -1.
	GroupID     int
	MergeAligns []string
}

// Entry is one recognition request (_paddleocr_vl_collect_page_vlm_entries_core).
type Entry struct {
	Block                int
	Img                  *pixels.RGB
	Prompt               string
	MinPixels, MaxPixels int
	// Figures are the tokens painted into a table's crop, for assembly to
	// swap back (tokenize_figure_of_table).
	Figures []figToken
}

// filterOverlap is paddleocr_vl/uilts.py filter_overlap_boxes in rect mode:
// "reference" regions go, as do regions under 6 px a side; of two that
// overlap by more than 0.7 of the smaller the smaller goes (an inline
// formula at 0.5, and always the formula), unless the pair is an image,
// table, seal or chart against something else that is not a table.
func filterOverlap(boxes []Box) []Box {
	var bs []Box
	for _, b := range boxes {
		if b.Label != "reference" {
			bs = append(bs, b)
		}
	}
	n := len(bs)
	area := make([]float64, n)
	for i, b := range bs {
		area[i] = abs64(float64((b.Coord[2] - b.Coord[0]) * (b.Coord[3] - b.Coord[1])))
	}
	overlap := func(i, j int) float64 {
		a, b := bs[i].Coord, bs[j].Coord
		iw := max(0, min(a[2], b[2])-max(a[0], b[0]))
		ih := max(0, min(a[3], b[3])-max(a[1], b[1]))
		small := min(area[i], area[j])
		if small <= 0 {
			return 0
		}
		return float64(iw*ih) / small
	}
	drop := map[int]bool{}
	for i := 0; i < n; i++ {
		c := bs[i].Coord
		if c[2]-c[0] < 6 || c[3]-c[1] < 6 {
			drop[i] = true
		}
		for j := i + 1; j < n; j++ {
			if drop[i] || drop[j] {
				continue
			}
			r := overlap(i, j)
			li, lj := bs[i].Label, bs[j].Label
			if li == "inline_formula" || lj == "inline_formula" {
				if r > 0.5 {
					if li == "inline_formula" {
						drop[i] = true
					}
					if lj == "inline_formula" {
						drop[j] = true
					}
					continue
				}
			}
			if r > 0.7 {
				visual := map[string]bool{"image": true, "table": true, "seal": true, "chart": true}
				if li != lj && (visual[li] || visual[lj]) {
					// labels <= {table, image, seal, chart}, or no table in the pair.
					if (li != "table" && lj != "table") || (visual[li] && visual[lj]) {
						continue
					}
				}
				if area[i] >= area[j] {
					drop[j] = true
				} else {
					drop[i] = true
				}
			}
		}
	}
	var out []Box
	for i, b := range bs {
		if !drop[i] {
			out = append(out, b)
		}
	}
	return out
}

func abs64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// crop is img[y0:y1, x0:x1] (CropByBoxes in rect mode).
func crop(img *pixels.RGB, c [4]int) *pixels.RGB {
	w, h := c[2]-c[0], c[3]-c[1]
	out := &pixels.RGB{W: w, H: h, Pix: make([]uint8, w*h*3)}
	for y := 0; y < h; y++ {
		copy(out.Pix[y*w*3:(y+1)*w*3], img.Pix[((c[1]+y)*img.W+c[0])*3:])
	}
	return out
}

// projOverlap is layout_parsing/utils.py calculate_projection_overlap_ratio,
// horizontal, union.
func projOverlap(a, b [4]int) float64 {
	o := min(a[2], b[2]) - max(a[0], b[0])
	if o <= 0 {
		return 0
	}
	return float64(o) / float64(max(a[2], b[2])-min(a[0], b[0]))
}

// overlapUnion is calculate_overlap_ratio's union mode.
func overlapUnion(a, b [4]int) float64 {
	iw := max(0, min(a[2], b[2])-max(a[0], b[0]))
	ih := max(0, min(a[3], b[3])-max(a[1], b[1]))
	inter := float64(iw * ih)
	ref := abs64(float64((a[2]-a[0])*(a[3]-a[1]))) + abs64(float64((b[2]-b[0])*(b[3]-b[1]))) - inter
	if ref == 0 {
		return 0
	}
	return inter / ref
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// mergeBlocks is paddleocr_vl/uilts.py merge_blocks: consecutive text
// blocks that continue each other -- side by side across a column gap, or
// one under the other aligned on exactly one edge with a non-merge block
// beside them -- are stacked into one image for one recognition, which
// the first block of the group carries.
func mergeBlocks(blocks []Block, nonMerge map[string]bool) []Block {
	type ib struct {
		idx int
		b   *Block
	}
	var toMerge []ib
	nonMergeIdx := map[int]bool{}
	for i := range blocks {
		if nonMerge[blocks[i].Label] {
			nonMergeIdx[i] = true
		} else {
			toMerge = append(toMerge, ib{i, &blocks[i]})
		}
	}
	aligned := func(a, b int) bool { return absInt(a-b) <= 5 }
	alignment := func(b, prev [4]int) string {
		switch {
		case aligned(b[0], prev[0]):
			return "left"
		case aligned(b[2], prev[2]):
			return "right"
		}
		return "center"
	}
	overlapsOther := func(bi, pi int) bool {
		p, b := blocks[pi].Box, blocks[bi].Box
		mb := [4]int{min(p[0], b[0]), min(p[1], b[1]), max(p[2], b[2]), max(p[3], b[3])}
		for idx := range blocks {
			if idx == bi || idx == pi || !nonMerge[blocks[idx].Label] {
				continue
			}
			if overlapUnion(mb, blocks[idx].Box) > 0 {
				return true
			}
		}
		return false
	}
	type group struct {
		indices []int
		aligns  []string
	}
	var groups []group
	var cur group
	for i, e := range toMerge {
		if len(cur.indices) == 0 {
			cur = group{indices: []int{e.idx}}
			continue
		}
		prev := toMerge[i-1]
		pb, bb := prev.b.Box, e.b.Box
		ih := projOverlap(bb, pb)
		cross := ih == 0 && e.b.Label == "text" && e.b.Label == prev.b.Label &&
			bb[0] > pb[2] && bb[1] < pb[3] &&
			float64(bb[0]-pb[2]) < float64(max(pb[2]-pb[0], bb[2]-bb[0]))*0.3
		updown := ih > 0 && e.b.Label == "text" && e.b.Label == prev.b.Label &&
			bb[3] >= pb[1] &&
			float64(absInt(bb[1]-pb[3])) < float64(max(pb[3]-pb[1], bb[3]-bb[1]))*0.5 &&
			(aligned(bb[0], pb[0]) != aligned(bb[2], pb[2])) &&
			overlapsOther(e.idx, prev.idx)
		if cross || updown {
			a := "center"
			if !cross {
				a = alignment(bb, pb)
			}
			cur.indices = append(cur.indices, e.idx)
			cur.aligns = append(cur.aligns, a)
		} else {
			groups = append(groups, cur)
			cur = group{indices: []int{e.idx}}
		}
	}
	if len(cur.indices) > 0 {
		groups = append(groups, cur)
	}

	var out []Block
	used := map[int]bool{}
	for idx := 0; idx < len(blocks); {
		found := false
		for _, g := range groups {
			start, end := g.indices[0], g.indices[0]
			for _, i := range g.indices {
				start, end = min(start, i), max(end, i)
			}
			free := true
			for _, i := range g.indices {
				if used[i] {
					free = false
				}
			}
			if idx != start || !free {
				continue
			}
			found = true
			var imgs []*pixels.RGB
			for _, i := range g.indices {
				imgs = append(imgs, blocks[i].Img)
			}
			w, h := 0, 0
			for _, im := range imgs {
				w, h = max(w, im.W), h+im.H
			}
			aspect := float64(h) / float64(w)
			if aspect >= 3 {
				for _, i := range g.indices {
					b := blocks[i]
					b.MergeAligns = nil
					out = append(out, b)
					used[i] = true
				}
			} else {
				m := mergeImages(imgs, g.aligns)
				for j, i := range g.indices {
					b := blocks[i]
					b.GroupID = g.indices[0]
					if j == 0 {
						b.Img, b.MergeAligns = m, append([]string{}, g.aligns...)
					} else {
						b.Img, b.MergeAligns = nil, nil
					}
					out = append(out, b)
					used[i] = true
				}
			}
			for n := start + 1; n < end; n++ {
				if nonMergeIdx[n] {
					out = append(out, blocks[n])
					used[n] = true
				}
			}
			idx = end + 1
			break
		}
		if found {
			continue
		}
		if nonMergeIdx[idx] && !used[idx] {
			out = append(out, blocks[idx])
			used[idx] = true
		}
		idx++
	}
	return out
}

// mergeImages is uilts.py merge_images: the images stacked top to bottom
// on a white canvas as wide as the widest, each step aligned left, right
// or centred (integer halving) against what is already stacked.
func mergeImages(imgs []*pixels.RGB, aligns []string) *pixels.RGB {
	if len(imgs) == 1 {
		return imgs[0]
	}
	xs := make([]int, len(imgs))
	mw := imgs[0].W
	for i := 1; i < len(imgs); i++ {
		sw := max(mw, imgs[i].W)
		var x1, x2 int
		switch aligns[i-1] {
		case "center":
			x1, x2 = (sw-mw)/2, (sw-imgs[i].W)/2
		case "right":
			x1, x2 = sw-mw, sw-imgs[i].W
		}
		for k := 0; k < i; k++ {
			xs[k] += x1
		}
		xs[i] = x2
		mw = sw
	}
	th := 0
	for _, im := range imgs {
		th += im.H
	}
	out := &pixels.RGB{W: mw, H: th, Pix: make([]uint8, mw*th*3)}
	for i := range out.Pix {
		out.Pix[i] = 255
	}
	y := 0
	for i, im := range imgs {
		for r := 0; r < im.H; r++ {
			copy(out.Pix[((y+r)*mw+xs[i])*3:], im.Pix[r*im.W*3:(r+1)*im.W*3])
		}
		y += im.H
	}
	return out
}

// cropMargin is uilts.py crop_margin: the bounding box of the pixels that
// are dark after stretching the grey levels to [0, 255] (at most 200 is
// dark), or the image unchanged if it is flat or has none.
func cropMargin(img *pixels.RGB) *pixels.RGB {
	n := img.W * img.H
	gray := make([]uint8, n)
	lo, hi := 255, 0
	for i := 0; i < n; i++ {
		// cv2 COLOR_BGR2GRAY on 8 bits: 14-bit fixed point, rounded.
		r, g, b := int(img.Pix[i*3]), int(img.Pix[i*3+1]), int(img.Pix[i*3+2])
		v := (r*4899 + g*9617 + b*1868 + (1 << 13)) >> 14
		gray[i] = uint8(v)
		lo, hi = min(lo, v), max(hi, v)
	}
	if lo == hi {
		return img
	}
	var lut [256]int
	for v := lo; v <= hi; v++ {
		lut[v] = int(float64(v-lo) / float64(hi-lo) * 255)
	}
	x0, y0, x1, y1 := img.W, img.H, -1, -1
	for y := 0; y < img.H; y++ {
		for x := 0; x < img.W; x++ {
			if lut[gray[y*img.W+x]] <= 200 {
				x0, y0 = min(x0, x), min(y0, y)
				x1, y1 = max(x1, x), max(y1, y)
			}
		}
	}
	if x1 < 0 {
		return img
	}
	return crop(img, [4]int{x0, y0, x1 + 1, y1 + 1})
}

// Pixel bounds a recognition takes, the pipeline's defaults for every label.
const (
	minPixels = ocr.DefaultMinPixels
	maxPixels = ocr.DefaultMaxPixels
)

// entries is _paddleocr_vl_collect_page_vlm_entries_core at the
// pipeline's defaults: every block with an image that is not an image
// label gets "OCR:", except tables ("Table Recognition:", figures inside a
// table tokenized), and formulas other than formula numbers ("Formula
// Recognition:", on the crop's margin-trimmed core).
//
// It also returns the figures inside tables, which PaddleX drops from the
// page's blocks (their content is the table's): gather_imgs' image, figure
// and seal regions, by construct_img_path.
func entries(blocks []Block, figures []Box) ([]Entry, map[string]bool) {
	var out []Entry
	drop := map[string]bool{}
	for j, b := range blocks {
		if imageLabels[b.Label] || b.Img == nil {
			continue
		}
		e := Entry{Block: j, Img: b.Img, Prompt: ocr.Tasks["ocr"], MinPixels: minPixels, MaxPixels: maxPixels}
		switch {
		case b.Label == "table":
			e.Prompt = ocr.Tasks["table"]
			var dropped []string
			e.Img, e.Figures, dropped = tokenizeFigures(b.Img, b.Box, figures)
			for _, p := range dropped {
				drop[p] = true
			}
		case containsFormula(b.Label) && b.Label != "formula_number":
			e.Prompt = ocr.Tasks["formula"]
			if c := cropMargin(b.Img); c.H > 2 && c.W > 2 {
				e.Img = c
			}
		}
		out = append(out, e)
	}
	return out, drop
}

// figureLabels are gather_imgs' labels (BLOCK_LABEL_MAP["image_labels"]).
var figureLabels = map[string]bool{"image": true, "figure": true, "seal": true}

// gatherFigures is layout_parsing/utils.py gather_imgs: the page's image,
// figure and seal regions that are not empty, in layout order.
func gatherFigures(boxes []Box) []Box {
	var out []Box
	for _, b := range boxes {
		if figureLabels[b.Label] && b.Coord[2] > b.Coord[0] && b.Coord[3] > b.Coord[1] {
			out = append(out, b)
		}
	}
	return out
}

// imgPath is layout_parsing/utils.py construct_img_path.
func imgPath(label string, c [4]int) string {
	return fmt.Sprintf("imgs/img_in_%s_box_%d_%d_%d_%d.jpg", label, c[0], c[1], c[2], c[3])
}

func containsFormula(label string) bool {
	return len(label) >= 7 && (label == "display_formula" || label == "inline_formula" || label == "formula")
}
