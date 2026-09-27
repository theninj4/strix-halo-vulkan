// Package page is PaddleOCR-VL's page pipeline (OCR.md O8): a page in,
// markdown and blocks out, from PP-DocLayoutV3 (ocr/layout) and the
// element-level engine (ocr), with PaddleX's glue between them ported
// function by function.
//
// The oracle is PaddleX 3.7.2 itself (reference/dump_ocr_page.py runs its
// functions, unmodified, on HF's layout), so each function here names the
// one it ports. Two PaddleX options are fixed and are the only departures
// from the pipeline's defaults:
//
//   - layout_shape_mode "rect": regions are their boxes, no polygons (the
//     default "auto" traces the masks with OpenCV and intersects polygons
//     with shapely; O8b). On the demo page 31 of 33 regions are rectangles
//     either way;
//   - the recognition model sees the raw crop pixels (PaddleX's "native"
//     backend), where its vllm-server client JPEG-encodes each crop.
//
// Arithmetic follows the numpy it ports: PaddleX's box arrays are float32
// and numpy 2's scalar rules keep comparisons against Python floats in
// float32 (the threshold, NMS, containment); the pipeline-level filters run
// on float64 or Python ints.
package page

import (
	"math"
	"sort"

	"strix-halo-vulkan/ocr/layout"
)

// Box is a region after PaddleX's layout post-processing
// (layout_analysis/processors.py restructured_boxes + update_order_index).
type Box struct {
	ClsID int
	Label string
	Score float32
	// Coord is x0, y0, x1, y1, clipped to the page, integers.
	Coord [4]int
	// Order is the reading order among the ordered labels, from 1; 0 for a
	// label PaddleX leaves unordered (SKIP_ORDER_LABELS).
	Order int
}

// Layout configuration, PaddleOCR-VL-1.6.yaml.
const layoutThreshold = 0.3

// mergeLarge are the classes whose layout_merge_bboxes_mode is "large": a
// box mostly inside one of these is dropped. The rest are "union" (kept).
var mergeLarge = []int{3, 5, 6, 15, 17} // chart, display_formula, doc_title, inline_formula, paragraph_title

// skipOrderLabels is layout_analysis/processors.py's SKIP_ORDER_LABELS.
var skipOrderLabels = map[string]bool{
	"figure_title": true, "vision_footnote": true, "image": true, "chart": true, "table": true,
	"header": true, "header_image": true, "footer": true, "footer_image": true, "footnote": true,
	"aside_text": true,
}

// row is one box as PaddleX's float32 array holds it: [cls, score, x0, y0,
// x1, y1, order].
type row struct {
	cls    int
	score  float32
	x0, y0 float32
	x1, y1 float32
	order  float32
}

// LayoutPost is LayoutAnalysisProcess.apply in rect mode, then
// update_order_index: detections is HF's post-processing at the layout
// threshold (layout.Output.Detect(w, h, 0.3)) for a w x h page.
func LayoutPost(detections []layout.Region, w, h int) []Box {
	rows := make([]row, 0, len(detections))
	for _, d := range detections {
		// np.round on the float32 coordinates: half to even.
		r := row{cls: d.LabelID, score: float32(d.Score), order: float32(d.Order)}
		c := [4]*float32{&r.x0, &r.y0, &r.x1, &r.y1}
		for k := range c {
			*c[k] = float32(math.RoundToEven(float64(float32(d.Box[k]))))
		}
		rows = append(rows, r)
	}
	// threshold, strictly above, in float32.
	kept := rows[:0]
	for _, r := range rows {
		if r.score > float32(layoutThreshold) && r.cls > -1 {
			kept = append(kept, r)
		}
	}
	rows = nms(kept)

	// Drop an image box larger than most of the page (filter_large_image).
	if len(rows) > 1 {
		thres := 0.93
		if w > h {
			thres = 0.82
		}
		limit := float32(thres * float64(w*h))
		var out []row
		for _, r := range rows {
			if r.cls == imageClass {
				x0, y0 := max(0, r.x0), max(0, r.y0)
				x1, y1 := min(float32(w), r.x1), min(float32(h), r.y1)
				if (x1-x0)*(y1-y0) > limit {
					continue
				}
			}
			out = append(out, r)
		}
		if len(out) > 0 {
			rows = out
		}
	}

	// layout_merge_bboxes_mode: for each "large" class, drop boxes 90% inside
	// a box of that class. formula_index is None (PaddleX looks up a label
	// "formula" this model does not have).
	keep := make([]bool, len(rows))
	for i := range keep {
		keep[i] = true
	}
	for _, cat := range mergeLarge {
		for i := range rows {
			for j := range rows {
				if i != j && rows[j].cls == cat && contained(rows[i], rows[j]) {
					keep[i] = false
				}
			}
		}
	}
	var merged []row
	for i, r := range rows {
		if keep[i] {
			merged = append(merged, r)
		}
	}
	// Sort by the order head's rank (np.argsort of a column of distinct
	// values: NMS has removed any second label of one query).
	sort.SliceStable(merged, func(a, b int) bool { return merged[a].order < merged[b].order })

	// restructured_boxes (the [1, 1] unclip is the identity on integer
	// coordinates), then update_order_index.
	var out []Box
	next := 1
	for _, r := range merged {
		x0 := int(max(0, r.x0))
		y0 := int(max(0, r.y0))
		x1 := int(min(float32(w), r.x1))
		y1 := int(min(float32(h), r.y1))
		if x1 <= x0 || y1 <= y0 {
			continue
		}
		b := Box{ClsID: r.cls, Label: layout.Labels[r.cls], Score: r.score, Coord: [4]int{x0, y0, x1, y1}}
		if !skipOrderLabels[b.Label] {
			b.Order = next
			next++
		}
		out = append(out, b)
	}
	return out
}

var imageClass = func() int {
	for i, l := range layout.Labels {
		if l == "image" {
			return i
		}
	}
	return -1
}()

// iou is object_detection/processors.py's, with its +1 pixel convention,
// in float32.
func iou(a, b row) float32 {
	ix0, iy0 := max(a.x0, b.x0), max(a.y0, b.y0)
	ix1, iy1 := min(a.x1, b.x1), min(a.y1, b.y1)
	inter := max(0, ix1-ix0+1) * max(0, iy1-iy0+1)
	aa := (a.x1 - a.x0 + 1) * (a.y1 - a.y0 + 1)
	ab := (b.x1 - b.x0 + 1) * (b.y1 - b.y0 + 1)
	return inter / float32(float64(aa+ab-inter))
}

// nms is object_detection/processors.py's, at iou_same 0.6 and iou_diff
// 0.98: highest score first, and a box survives another of its own class
// below 0.6 and of another class below 0.98. The survivors come back in
// score order, as PaddleX indexes them.
func nms(rows []row) []row {
	idx := make([]int, len(rows))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return rows[idx[a]].score > rows[idx[b]].score })
	var out []row
	for len(idx) > 0 {
		cur := rows[idx[0]]
		out = append(out, cur)
		var rest []int
		for _, i := range idx[1:] {
			th := float32(0.98)
			if rows[i].cls == cur.cls {
				th = 0.6
			}
			if iou(cur, rows[i]) < th {
				rest = append(rest, i)
			}
		}
		idx = rest
	}
	return out
}

// contained is is_contained: at least 90% of a's area inside b, float32.
func contained(a, b row) bool {
	area := (a.x1 - a.x0) * (a.y1 - a.y0)
	iw := max(0, min(a.x1, b.x1)-max(a.x0, b.x0))
	ih := max(0, min(a.y1, b.y1)-max(a.y0, b.y0))
	if area <= 0 {
		return false
	}
	return iw*ih/area >= float32(0.9)
}
