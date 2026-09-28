package backend

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"image"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm/pixels"
	"strix-halo-vulkan/ocr/page"
)

// Page parsing (OCR.md O9): the api.DocumentBackend half of OCR, on the
// same engine as the chat door. A PDF is rasterised by poppler's pdftoppm
// (decision 9, as ffmpeg muxes video) at PaddleX's scale: 2.0, i.e. 144 dpi,
// each side ceil(points x 2) pixels, which is what pypdfium2's render gives
// PaddleX's serving (the two renderers' anti-aliasing is not bit-identical).

// pdfScale is PaddleX's PDF_RENDER_SCALE.
const pdfScale = 2.0

// defaultMaxPages bounds one request's pages.
const defaultMaxPages = 100

var pdfPageSize = regexp.MustCompile(`^Page\s+(\d+) size:\s+([0-9.]+) x ([0-9.]+) pts`)

// pdfPages is each page's size in points, from pdfinfo.
func pdfPages(ctx context.Context, path string) ([][2]float64, error) {
	out, err := exec.CommandContext(ctx, "pdfinfo", "-f", "1", "-l", "100000", path).Output()
	if err != nil {
		return nil, fmt.Errorf("reading the PDF (pdfinfo): %v: %w", err, api.ErrUnsupported)
	}
	var sizes [][2]float64
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		m := pdfPageSize.FindStringSubmatch(sc.Text())
		if m == nil {
			continue
		}
		w, _ := strconv.ParseFloat(m[2], 64)
		h, _ := strconv.ParseFloat(m[3], 64)
		sizes = append(sizes, [2]float64{w, h})
	}
	if len(sizes) == 0 {
		return nil, fmt.Errorf("the PDF has no pages: %w", api.ErrUnsupported)
	}
	return sizes, nil
}

// renderPDFPage rasterises page i (from 0) at PaddleX's scale.
func renderPDFPage(ctx context.Context, path string, i int, size [2]float64) (*pixels.RGB, error) {
	w := int(math.Ceil(size[0] * pdfScale))
	h := int(math.Ceil(size[1] * pdfScale))
	n := strconv.Itoa(i + 1)
	out, err := exec.CommandContext(ctx, "pdftoppm", "-png", "-f", n, "-l", n,
		"-scale-to-x", strconv.Itoa(w), "-scale-to-y", strconv.Itoa(h), "-singlefile", path).Output()
	if err != nil {
		return nil, fmt.Errorf("rendering page %d (pdftoppm): %v", i, err)
	}
	img, _, err := pixels.Decode(out)
	if err != nil {
		return nil, fmt.Errorf("page %d: %v", i, err)
	}
	return img, nil
}

// rgbImage is a page crop as an image.Image, for the HTTP layer to encode.
func rgbImage(p *pixels.RGB) image.Image {
	out := image.NewNRGBA(image.Rect(0, 0, p.W, p.H))
	for i := 0; i < p.W*p.H; i++ {
		copy(out.Pix[i*4:], p.Pix[i*3:i*3+3])
		out.Pix[i*4+3] = 255
	}
	return out
}

// ParseDocument parses an image or the selected pages of a PDF.
func (b *OCR) ParseDocument(ctx context.Context, req *api.DocumentRequest) (*api.DocumentResult, error) {
	b.mu.Lock()
	eng, parser := b.eng, b.parser
	b.mu.Unlock()
	if eng == nil || parser == nil {
		return nil, fmt.Errorf("the OCR model is closed")
	}
	maxPages := b.opt.MaxPages
	if maxPages <= 0 {
		maxPages = defaultMaxPages
	}
	type pageIn struct {
		index int
		img   *pixels.RGB
		dpi   int
	}
	var pages []pageIn
	res := &api.DocumentResult{}
	if req.PDF {
		dir, err := os.MkdirTemp("", "ocr-pdf-")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(dir)
		path := filepath.Join(dir, "in.pdf")
		if err := os.WriteFile(path, req.Data, 0o600); err != nil {
			return nil, err
		}
		sizes, err := pdfPages(ctx, path)
		if err != nil {
			return nil, err
		}
		res.PDFPages = len(sizes)
		sel := req.Pages
		if sel == nil {
			for i := range sizes {
				sel = append(sel, i)
			}
		}
		if len(sel) > maxPages {
			return nil, fmt.Errorf("%d pages; this server parses at most %d a request: %w", len(sel), maxPages, api.ErrUnsupported)
		}
		for _, i := range sel {
			if i >= len(sizes) {
				return nil, fmt.Errorf("page %d of a %d-page PDF: %w", i, len(sizes), api.ErrUnsupported)
			}
			img, err := renderPDFPage(ctx, path, i, sizes[i])
			if err != nil {
				return nil, err
			}
			pages = append(pages, pageIn{i, img, int(pdfScale * 72)})
		}
	} else {
		img, _, err := pixels.Decode(req.Data)
		if err != nil {
			return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
		}
		pages = append(pages, pageIn{0, img, 0})
	}

	p := *parser
	p.Recognize = eng.RecognizeAll
	imgN, tblN := 0, 0
	for _, in := range pages {
		pg, err := p.Parse(ctx, in.img)
		if err != nil {
			return nil, fmt.Errorf("page %d: %w", in.index, err)
		}
		dp := api.DocumentPage{Index: in.index, Width: in.img.W, Height: in.img.H, DPI: in.dpi}
		md, tables, imgs := page.Mistral(pg.Results, req.SeparateTables, imgN, tblN)
		imgN, tblN = imgN+len(imgs), tblN+len(tables)
		dp.Mistral = md
		for _, t := range tables {
			dp.MistralTables = append(dp.MistralTables, api.DocumentTable{ID: t.ID, HTML: t.HTML})
		}
		for _, im := range imgs {
			dp.MistralImages = append(dp.MistralImages, api.DocumentImage{ID: im.ID, Box: im.Box, Image: rgbImage(im.Img.(*pixels.RGB))})
		}
		var head, foot []string
		for _, r := range pg.Results {
			switch r.Label {
			case "header":
				head = append(head, r.Content)
			case "footer":
				foot = append(foot, r.Content)
			}
		}
		dp.Header, dp.Footer = strings.Join(head, "\n"), strings.Join(foot, "\n")
		res.Pages = append(res.Pages, dp)
	}
	return res, nil
}
