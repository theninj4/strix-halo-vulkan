package layout

import (
	"sort"

	"strix-halo-vulkan/llm/pixels"
)

// Preprocess is HF's PPDocLayoutV3ImageProcessor: an 800x800 bicubic resize
// without antialiasing on the 8-bit image (torch's uint8 kernel,
// pixels.ResizeNoAA; it stands in for PaddleX's cv2.INTER_CUBIC), then x/255,
// as [3][800][800]. Exact against HF on every PNG case.
func Preprocess(img *pixels.RGB) []float32 {
	const n = InputSize
	r := pixels.ResizeNoAA(img, n, n)
	out := make([]float32, 3*n*n)
	for i := 0; i < n*n; i++ {
		for c := 0; c < 3; c++ {
			// HF fuses rescale and normalize into (x - 0) / (1/(1/255)):
			// a division by 255, which is an ulp from a product by 1/255.
			out[c*n*n+i] = float32(r.Pix[i*3+c]) / 255
		}
	}
	return out
}

// Region is one detection in reading order.
type Region struct {
	Label   string
	LabelID int
	Score   float64
	// Box is x0, y0, x1, y1 in the original image's pixels, unclipped (the
	// model can place an edge a few pixels outside the page).
	Box [4]float64
	// Order is the region's rank among all 300 queries by the order head;
	// the regions come sorted by it.
	Order int
	Query int
}

// Detect is HF's post_process_object_detection without the polygons: every
// (query, label) pair whose sigmoid score reaches threshold, the boxes
// scaled to a w x h image, sorted by the order head's ranking.
func (o *Output) Detect(w, h int, threshold float64) []Region {
	order := orderSeq(o.Order)
	type cand struct {
		score float64
		idx   int
	}
	var cands []cand
	for i, v := range o.Logits.V {
		if s := sigmoid(float64(v)); s >= threshold {
			cands = append(cands, cand{s, i})
		}
	}
	// torch.topk over the flattened scores, largest first.
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].score > cands[b].score })
	out := make([]Region, 0, len(cands))
	for _, c := range cands {
		q, l := c.idx/numLabels, c.idx%numLabels
		b := o.Boxes.Row(q)
		cx, cy, bw, bh := float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])
		out = append(out, Region{
			Label: Labels[l], LabelID: l, Score: c.score, Query: q, Order: order[q],
			Box: [4]float64{(cx - bw/2) * float64(w), (cy - bh/2) * float64(h), (cx + bw/2) * float64(w), (cy + bh/2) * float64(h)},
		})
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Order < out[b].Order })
	return out
}

// orderSeq is _get_order_seqs: query j's votes are the probability that
// each earlier query precedes it plus the probability that it does not
// precede each later one, and its order is its rank by votes, fewest first.
func orderSeq(logits *Rows) []int {
	n := logits.N
	s := make([]float64, n*n)
	for i, v := range logits.V {
		s[i] = sigmoid(float64(v))
	}
	votes := make([]float64, n)
	for j := 0; j < n; j++ {
		var v float64
		for i := 0; i < j; i++ {
			v += s[i*n+j]
		}
		for i := j + 1; i < n; i++ {
			v += 1 - s[j*n+i]
		}
		votes[j] = v
	}
	ptr := make([]int, n)
	for i := range ptr {
		ptr[i] = i
	}
	sort.SliceStable(ptr, func(a, b int) bool { return votes[ptr[a]] < votes[ptr[b]] })
	rank := make([]int, n)
	for r, q := range ptr {
		rank[q] = r
	}
	return rank
}

// Detect runs a page through the model at the given score threshold.
func (m *Model) Detect(img *pixels.RGB, threshold float64) ([]Region, error) {
	out, err := m.Forward(Preprocess(img), nil)
	if err != nil {
		return nil, err
	}
	return out.Detect(img.W, img.H, threshold), nil
}
