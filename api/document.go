package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"

	"strix-halo-vulkan/util"
)

// Document parsing (OCR.md O9): a page image or a PDF in, markdown out, as
// POST /v1/ocr, Mistral's OCR API -- the closest thing document parsing has
// to an industry standard: `document` (`document_url` or `image_url`),
// `pages`, `include_image_base64`, `table_format`,
// `extract_header`/`extract_footer`; `pages[].markdown` with
// ![img-N.jpeg](img-N.jpeg) references, `images[]` with their boxes,
// `dimensions`, `usage_info`. What it does not do (file ids, annotations,
// markdown tables) it refuses by name rather than ignores.

// DocumentBackend parses documents.
type DocumentBackend interface {
	Backend
	ParseDocument(ctx context.Context, req *DocumentRequest) (*DocumentResult, error)
}

// DocumentRequest is one document.
type DocumentRequest struct {
	Data []byte
	// PDF is whether Data is a PDF; otherwise it is an image.
	PDF bool
	// Pages selects PDF pages, from 0; nil is every page.
	Pages []int
	// SeparateTables takes tables out of the markdown (table_format "html").
	SeparateTables bool
}

// DocumentResult is the parsed pages, in document order.
type DocumentResult struct {
	Pages []DocumentPage
	// PDFPages is how many pages the PDF has (0 for an image).
	PDFPages int
}

// DocumentPage is one page.
type DocumentPage struct {
	Index         int // in the document, from 0
	Width, Height int
	// DPI is the rendering resolution of a PDF page, 0 for an image.
	DPI int
	// Mistral is the page in Mistral's shape: PaddleX's plain markdown with
	// Mistral's references.
	Mistral        string
	MistralImages  []DocumentImage
	MistralTables  []DocumentTable
	Header, Footer string
}

// DocumentImage is a picture the Mistral markdown references.
type DocumentImage struct {
	ID    string
	Box   [4]int // x0, y0, x1, y1
	Image image.Image
}

// DocumentTable is a table out of the Mistral markdown.
type DocumentTable struct {
	ID, HTML string
}

func (s *Server) documentBackend(ctx context.Context, w http.ResponseWriter) bool {
	if s.Document == nil {
		notLoaded(ctx, w, "the document parser", "-ocr")
		return false
	}
	return true
}

// isPDF is the file's magic.
func isPDF(b []byte) bool { return bytes.HasPrefix(b, []byte("%PDF-")) }

// jpegBase64 is an image as base64 JPEG (quality 95, cv2.imencode's).
func jpegBase64(img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// ---- Mistral: POST /v1/ocr

type mistralDocument struct {
	Type        string          `json:"type"`
	DocumentURL string          `json:"document_url"`
	ImageURL    json.RawMessage `json:"image_url"`
	FileID      string          `json:"file_id"`
}

type mistralRequest struct {
	Model                    string          `json:"model"`
	ID                       string          `json:"id"`
	Document                 mistralDocument `json:"document"`
	Pages                    []int           `json:"pages"`
	IncludeImageBase64       bool            `json:"include_image_base64"`
	ImageLimit               *int            `json:"image_limit"`
	ImageMinSize             *int            `json:"image_min_size"`
	TableFormat              *string         `json:"table_format"`
	ExtractHeader            bool            `json:"extract_header"`
	ExtractFooter            bool            `json:"extract_footer"`
	BBoxAnnotationFormat     json.RawMessage `json:"bbox_annotation_format"`
	DocumentAnnotationFormat json.RawMessage `json:"document_annotation_format"`
}

type mistralImage struct {
	ID       string  `json:"id"`
	TopLeftX int     `json:"top_left_x"`
	TopLeftY int     `json:"top_left_y"`
	BottomX  int     `json:"bottom_right_x"`
	BottomY  int     `json:"bottom_right_y"`
	Base64   *string `json:"image_base64"`
}

type mistralTable struct {
	ID      string `json:"id"`
	Content string `json:"content"`
	Format  string `json:"format"`
}

type mistralPage struct {
	Index      int            `json:"index"`
	Markdown   string         `json:"markdown"`
	Images     []mistralImage `json:"images"`
	Tables     []mistralTable `json:"tables"`
	Hyperlinks []string       `json:"hyperlinks"`
	Header     *string        `json:"header"`
	Footer     *string        `json:"footer"`
	Dimensions struct {
		DPI    int `json:"dpi"`
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"dimensions"`
}

type mistralResponse struct {
	Pages              []mistralPage `json:"pages"`
	Model              string        `json:"model"`
	DocumentAnnotation *string       `json:"document_annotation"`
	UsageInfo          struct {
		PagesProcessed int `json:"pages_processed"`
		DocSizeBytes   int `json:"doc_size_bytes"`
	} `json:"usage_info"`
}

// handleOCR is Mistral's OCR endpoint.
func (s *Server) handleOCR(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if !s.documentBackend(ctx, w) {
		return
	}
	var req mistralRequest
	if !decodeJSON(ctx, w, r, &req) {
		return
	}
	var url string
	switch req.Document.Type {
	case "document_url":
		url = req.Document.DocumentURL
	case "image_url":
		// A string, or OpenAI's {"url": ...}.
		var sv string
		if json.Unmarshal(req.Document.ImageURL, &sv) != nil {
			var obj struct {
				URL string `json:"url"`
			}
			_ = json.Unmarshal(req.Document.ImageURL, &obj)
			sv = obj.URL
		}
		url = sv
	case "file":
		badRequest(ctx, w, "document.type \"file\" names an uploaded file_id; this server has no Files API, send a document_url or image_url")
		return
	default:
		badRequest(ctx, w, fmt.Sprintf("document.type %q; this server reads document_url and image_url", req.Document.Type))
		return
	}
	switch {
	case url == "":
		badRequest(ctx, w, "the document has no URL")
		return
	case len(req.BBoxAnnotationFormat) > 0 && string(req.BBoxAnnotationFormat) != "null",
		len(req.DocumentAnnotationFormat) > 0 && string(req.DocumentAnnotationFormat) != "null":
		badRequest(ctx, w, "annotation formats need a language model reading the page; this server's OCR does not annotate")
		return
	case req.TableFormat != nil && *req.TableFormat == "markdown":
		badRequest(ctx, w, "table_format \"markdown\": this model writes tables as HTML (with spans); ask for \"html\", or none for them inline")
		return
	case req.TableFormat != nil && *req.TableFormat != "html":
		badRequest(ctx, w, fmt.Sprintf("table_format %q; this server takes \"html\"", *req.TableFormat))
		return
	}
	data, err := util.FetchDocument(ctx, url)
	if err != nil {
		badRequest(ctx, w, "the document: "+err.Error())
		return
	}
	for _, p := range req.Pages {
		if p < 0 {
			badRequest(ctx, w, fmt.Sprintf("page %d; pages count from 0", p))
			return
		}
	}
	dreq := &DocumentRequest{Data: data, PDF: isPDF(data), Pages: req.Pages,
		SeparateTables: req.TableFormat != nil && *req.TableFormat == "html"}
	if !dreq.PDF && len(req.Pages) > 0 && (len(req.Pages) != 1 || req.Pages[0] != 0) {
		badRequest(ctx, w, "pages: an image is one page, page 0")
		return
	}
	res, err := s.Document.ParseDocument(ctx, dreq)
	if err != nil {
		backendError(ctx, w, "ocr", err)
		return
	}
	out := mistralResponse{Model: modelID(s.Document, req.Model), Pages: []mistralPage{}}
	out.UsageInfo.DocSizeBytes = len(data)
	out.UsageInfo.PagesProcessed = len(res.Pages)
	for _, pg := range res.Pages {
		mp := mistralPage{Index: pg.Index, Markdown: pg.Mistral, Images: []mistralImage{}, Tables: []mistralTable{}, Hyperlinks: []string{}}
		mp.Dimensions.DPI, mp.Dimensions.Width, mp.Dimensions.Height = pg.DPI, pg.Width, pg.Height
		for _, im := range pg.MistralImages {
			mi := mistralImage{ID: im.ID, TopLeftX: im.Box[0], TopLeftY: im.Box[1], BottomX: im.Box[2], BottomY: im.Box[3]}
			if req.IncludeImageBase64 {
				b, err := jpegBase64(im.Image)
				if err != nil {
					serverError(ctx, w, "ocr", err)
					return
				}
				b = "data:image/jpeg;base64," + b
				mi.Base64 = &b
			}
			mp.Images = append(mp.Images, mi)
		}
		for _, t := range pg.MistralTables {
			mp.Tables = append(mp.Tables, mistralTable{ID: t.ID, Content: t.HTML, Format: "html"})
		}
		if req.ExtractHeader && pg.Header != "" {
			h := pg.Header
			mp.Header = &h
		}
		if req.ExtractFooter && pg.Footer != "" {
			f := pg.Footer
			mp.Footer = &f
		}
		out.Pages = append(out.Pages, mp)
	}
	writeJSON(w, http.StatusOK, out)
}
