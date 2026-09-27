package page

import (
	"context"
	"fmt"
	"time"

	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr"
	"strix-halo-vulkan/ocr/layout"
)

// visImageLabels are the labels whose crops the markdown shows as images
// (IMAGE_LABELS + seal + chart, at the defaults).
var visImageLabels = map[string]bool{"image": true, "header_image": true, "footer_image": true, "seal": true, "chart": true}

// maxNewTokens is the pipeline's default for a block.
const maxNewTokens = 4096

// assemble is _paddleocr_vl_assemble_parsing_results: each block's content
// from its recognition, post-processed by label, and an image path for the
// visual blocks; an image inside a table is dropped.
func assemble(blocks []Block, ents []Entry, texts []string, drop map[string]bool) []Result {
	byBlock := map[int]int{}
	for k, e := range ents {
		byBlock[e.Block] = k
	}
	var out []Result
	for j, b := range blocks {
		content := ""
		if k, ok := byBlock[j]; ok {
			r := texts[k]
			minCount := 50
			if b.Label == "table" {
				minCount = 5000
			}
			r = truncateRepetitive(r, minCount)
			r = formulaDelimiters(r, b.Label)
			if b.Label == "table" {
				if h := otslToHTML(r); h != "" {
					r = h
				}
			}
			content = r
		}
		res := Result{Label: b.Label, BBox: b.Box, Content: content, GroupID: b.GroupID}
		if visImageLabels[b.Label] && b.Img != nil {
			p := imgPath(b.Label, b.Box)
			if drop[p] {
				continue
			}
			res.Image, res.Img = p, b.Img
		}
		out = append(out, res)
	}
	return out
}

// resultSkipOrder is result.py's SKIP_ORDER_LABELS plus the markdown's
// ignored labels: the blocks that get no block_order.
var resultSkipOrder = func() map[string]bool {
	m := map[string]bool{}
	for l := range skipOrderLabels {
		m[l] = true
	}
	for l := range markdownIgnore {
		m[l] = true
	}
	return m
}()

// JSONBlock is one entry of PaddleX's parsing_res_list.
type JSONBlock struct {
	Label   string `json:"block_label"`
	Content string `json:"block_content"`
	BBox    [4]int `json:"block_bbox"`
	ID      int    `json:"block_id"`
	Order   *int   `json:"block_order"`
	GroupID int    `json:"group_id"`
}

// JSON is PaddleOCRVLResult._to_json's parsing_res_list.
func JSON(results []Result) []JSONBlock {
	out := make([]JSONBlock, len(results))
	next := 1
	for i, r := range results {
		jb := JSONBlock{Label: r.Label, Content: r.Content, BBox: r.BBox, ID: i, GroupID: r.GroupID}
		if jb.GroupID < 0 {
			jb.GroupID = i
		}
		if !resultSkipOrder[r.Label] {
			o := next
			jb.Order = &o
			next++
		}
		out[i] = jb
	}
	return out
}

// Page is one parsed page and how it got there.
type Page struct {
	Width, Height int
	// Boxes are the layout regions after PaddleX's post-processing.
	Boxes   []Box
	Blocks  []Block
	Entries []Entry
	// Texts are the raw recognitions, one an entry.
	Texts    []string
	Results  []Result
	Markdown string
	// Images are the markdown's pictures by path: the image, chart and seal
	// blocks' crops and every image, figure and seal region (gather_imgs),
	// as PaddleX's markdown_images.
	Images  map[string]*pixels.RGB
	Timings Timings
}

// Pruned is PaddleX's serving prunedResult for the page: the result JSON
// without input_path and page_index, at the pipeline's model settings.
// pageCount is nil for an image.
func (pg *Page) Pruned(pageCount *int) map[string]any {
	var boxes []map[string]any
	for _, b := range pg.Boxes {
		var order any
		if b.Order > 0 {
			order = b.Order
		}
		boxes = append(boxes, map[string]any{"cls_id": b.ClsID, "label": b.Label, "score": float64(b.Score),
			"coordinate": b.Coord, "order": order})
	}
	return map[string]any{
		"page_count": pageCount,
		"width":      pg.Width,
		"height":     pg.Height,
		"model_settings": map[string]any{
			"use_doc_preprocessor": false, "use_layout_detection": true, "use_chart_recognition": false,
			"use_seal_recognition": false, "use_ocr_for_image_block": false, "format_block_content": false,
			"merge_layout_blocks": true,
			"markdown_ignore_labels": []string{"number", "footnote", "header", "header_image", "footer",
				"footer_image", "aside_text"},
			"return_layout_polygon_points": false,
		},
		"parsing_res_list": JSON(pg.Results),
		"layout_det_res":   map[string]any{"boxes": boxes},
	}
}

// Timings is where a page's time went.
type Timings struct {
	Layout, Glue, Recognition time.Duration
}

// Parser runs pages: Layout is PP-DocLayoutV3 over the preprocessed page
// (the device trunk's or the CPU oracle's Forward), Recognize the
// element-level engine.
type Parser struct {
	Layout    func(pixels []float32) (*layout.Output, error)
	Recognize func(ctx context.Context, req ocr.Request) (*ocr.Result, error)
}

// Parse runs one page.
func (p *Parser) Parse(ctx context.Context, img *pixels.RGB) (*Page, error) {
	pg := &Page{Width: img.W, Height: img.H}
	t0 := time.Now()
	out, err := p.Layout(layout.Preprocess(img))
	if err != nil {
		return nil, err
	}
	pg.Boxes = LayoutPost(out.Detect(img.W, img.H, layoutThreshold), img.W, img.H)
	pg.Timings.Layout = time.Since(t0)

	t0 = time.Now()
	var figures []Box
	for _, b := range pg.Boxes {
		if figureLabels[b.Label] {
			figures = append(figures, b)
		}
	}
	for _, b := range filterOverlap(pg.Boxes) {
		pg.Blocks = append(pg.Blocks, Block{Label: b.Label, Box: b.Coord, Img: crop(img, b.Coord), GroupID: -1})
	}
	nonMerge := map[string]bool{"table": true}
	for l := range imageLabels {
		nonMerge[l] = true
	}
	pg.Blocks = mergeBlocks(pg.Blocks, nonMerge)
	ents, drop, err := entries(pg.Blocks, figures)
	if err != nil {
		return nil, err
	}
	pg.Entries = ents
	pg.Timings.Glue = time.Since(t0)

	t0 = time.Now()
	for _, e := range ents {
		res, err := p.Recognize(ctx, ocr.Request{Image: e.Img, Prompt: e.Prompt, MinPixels: e.MinPixels, MaxPixels: e.MaxPixels, MaxTokens: maxNewTokens})
		if err != nil {
			return nil, fmt.Errorf("page: block %d (%s): %w", e.Block, pg.Blocks[e.Block].Label, err)
		}
		pg.Texts = append(pg.Texts, res.Text)
	}
	pg.Timings.Recognition = time.Since(t0)
	pg.Results = assemble(pg.Blocks, ents, pg.Texts, drop)
	pg.Markdown = Markdown(pg.Results, img.W)
	pg.Images = map[string]*pixels.RGB{}
	for _, r := range pg.Results {
		if r.Image != "" {
			pg.Images[r.Image] = r.Img.(*pixels.RGB)
		}
	}
	for _, f := range figures {
		pg.Images[imgPath(f.Label, f.Coord)] = crop(img, f.Coord)
	}
	return pg, nil
}
