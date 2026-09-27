package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"strings"
	"time"

	"strix-halo-vulkan/util"
)

// Document parsing (OCR.md O9): a page image or a PDF in, markdown and
// blocks out, through two envelopes over one backend.
//
//   - POST /v1/ocr is Mistral's OCR API, the closest thing document parsing
//     has to an industry standard: `document` (`document_url` or
//     `image_url`), `pages`, `include_image_base64`, `table_format`,
//     `extract_header`/`extract_footer`; `pages[].markdown` with
//     ![img-N.jpeg](img-N.jpeg) references, `images[]` with their boxes,
//     `dimensions`, `usage_info`.
//   - POST /layout-parsing is PaddleX's own serving envelope for
//     PaddleOCR-VL (paddlex/inference/serving/schemas/paddleocr_vl.py), so
//     PaddleOCR's serving clients work unchanged: `file` (base64 or URL) and
//     `fileType`, `layoutParsingResults[]` of `prunedResult` and `markdown`,
//     `dataInfo`, in the `{logId, errorCode, errorMsg, result}` wrapper.
//
// What neither does, each refuses by name rather than ignores: polygons
// (layoutShapeMode other than "rect", OCR.md O8b), chart and seal
// recognition, document preprocessing, visualisations, export formats,
// Mistral's annotations and file ids.

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
	// MinPixels and MaxPixels override every block's resize bounds, 0 the
	// pipeline's; MaxTokens a block's generation cap, 0 the pipeline's.
	MinPixels, MaxPixels, MaxTokens int
	// RepetitionPenalty is vLLM's, 0 off.
	RepetitionPenalty float64
	// SeparateTables and the image numbering shape the Mistral markdown.
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
	// Markdown is PaddleX's pretty markdown, and Pruned its prunedResult.
	Markdown string
	Pruned   map[string]any
	// MarkdownImages are PaddleX's markdown_images by path.
	MarkdownImages map[string]image.Image
	// Mistral is the page in Mistral's shape.
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

// ---- PaddleX: POST /layout-parsing

// paddleRequest is InferRequest (schemas/paddleocr_vl.py). The fields this
// server implements are typed; every other one is kept raw so that a value
// other than its default is refused by name.
type paddleRequest struct {
	File                 string   `json:"file"`
	FileType             *int     `json:"fileType"`
	LayoutShapeMode      *string  `json:"layoutShapeMode"`
	RepetitionPenalty    *float64 `json:"repetitionPenalty"`
	Temperature          *float64 `json:"temperature"`
	TopP                 *float64 `json:"topP"`
	MinPixels            *int     `json:"minPixels"`
	MaxPixels            *int     `json:"maxPixels"`
	MaxNewTokens         *int     `json:"maxNewTokens"`
	PrettifyMarkdown     *bool    `json:"prettifyMarkdown"`
	ReturnMarkdownImages *bool    `json:"returnMarkdownImages"`
	LogID                string   `json:"logId"`

	rest map[string]json.RawMessage
}

// paddleDefaults are InferRequest's other fields and the values this
// server's pipeline is (null is always the pipeline's default).
var paddleDefaults = map[string]string{
	"useDocOrientationClassify": "false", "useDocUnwarping": "false", "useLayoutDetection": "true",
	"useChartRecognition": "false", "useSealRecognition": "false", "useOcrForImageBlock": "false",
	"layoutThreshold": "", "layoutNms": "true", "layoutUnclipRatio": "", "layoutMergeBboxesMode": "",
	"promptLabel": "", "formatBlockContent": "false", "mergeLayoutBlocks": "true", "markdownIgnoreLabels": "",
	"vlmExtraArgs": "", "showFormulaNumber": "false", "restructurePages": "false", "mergeTables": "true",
	"relevelTitles": "true", "visualize": "false", "outputFormats": "",
}

type paddleError struct {
	LogID     string `json:"logId"`
	ErrorCode int    `json:"errorCode"`
	ErrorMsg  string `json:"errorMsg"`
}

func paddleFail(ctx context.Context, w http.ResponseWriter, logID string, status int, msg string) {
	logf(ctx, "%d: %s", status, msg)
	writeJSON(w, status, paddleError{LogID: logID, ErrorCode: status, ErrorMsg: msg})
}

// handleLayoutParsing is PaddleX's /layout-parsing.
func (s *Server) handleLayoutParsing(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	logID := randomUUID()
	if s.Document == nil {
		paddleFail(ctx, w, logID, http.StatusNotImplemented, "the document parser is not loaded; start the server with -ocr")
		return
	}
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		if tooLarge(ctx, w, err) {
			return
		}
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "malformed JSON body: "+err.Error())
		return
	}
	b, _ := json.Marshal(raw)
	var req paddleRequest
	if err := json.Unmarshal(b, &req); err != nil {
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if req.LogID != "" {
		logID = req.LogID
	}
	for k, v := range raw {
		want, known := paddleDefaults[k]
		if !known {
			continue // a typed field, or one PaddleX's schema would reject anyway
		}
		got := strings.TrimSpace(string(v))
		if got == "null" || got == want || (k == "outputFormats" && (got == "[]")) {
			continue
		}
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity,
			fmt.Sprintf("`%s` %s is not implemented by this server's pipeline (it runs %s)", k, got, orDefault(want)))
		return
	}
	switch {
	case req.File == "":
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`file` is required")
		return
	case req.LayoutShapeMode != nil && *req.LayoutShapeMode != "rect":
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity,
			fmt.Sprintf("`layoutShapeMode` %q: this server's layout is rectangles only (\"rect\"); polygons are not implemented", *req.LayoutShapeMode))
		return
	case req.Temperature != nil && *req.Temperature != 0:
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`temperature`: this model decodes greedily (0)")
		return
	case req.TopP != nil && (*req.TopP <= 0 || *req.TopP > 1):
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`topP` must be > 0 and ≤ 1")
		return
	case req.RepetitionPenalty != nil && *req.RepetitionPenalty <= 0:
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`repetitionPenalty` must be > 0")
		return
	case req.MinPixels != nil && req.MaxPixels != nil && *req.MinPixels > *req.MaxPixels:
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`minPixels` cannot be greater than `maxPixels`")
		return
	}
	var data []byte
	var err error
	if strings.HasPrefix(req.File, "http://") || strings.HasPrefix(req.File, "https://") {
		data, err = util.FetchDocument(ctx, req.File)
	} else {
		data, err = base64.StdEncoding.DecodeString(req.File)
	}
	if err != nil {
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "Invalid input file: "+err.Error())
		return
	}
	pdf := isPDF(data)
	if req.FileType != nil && (*req.FileType == 0) != pdf {
		paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, fmt.Sprintf("`fileType` %d does not match the file", *req.FileType))
		return
	}
	dreq := &DocumentRequest{Data: data, PDF: pdf}
	if req.MinPixels != nil {
		dreq.MinPixels = *req.MinPixels
	}
	if req.MaxPixels != nil {
		dreq.MaxPixels = *req.MaxPixels
	}
	if req.MaxNewTokens != nil {
		dreq.MaxTokens = *req.MaxNewTokens
	}
	if req.RepetitionPenalty != nil {
		dreq.RepetitionPenalty = *req.RepetitionPenalty
	}
	start := time.Now()
	res, err := s.Document.ParseDocument(ctx, dreq)
	if err != nil {
		if errors.Is(err, ErrUnsupported) {
			paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, err.Error())
			return
		}
		if errors.Is(err, context.Canceled) {
			logf(ctx, "layout-parsing: client cancelled")
			return
		}
		paddleFail(ctx, w, logID, http.StatusInternalServerError, err.Error())
		return
	}
	logf(ctx, "layout-parsing: %d pages in %v", len(res.Pages), time.Since(start).Round(time.Millisecond))

	type markdownData struct {
		Text   string            `json:"text"`
		Images map[string]string `json:"images"`
	}
	type lpr struct {
		PrunedResult map[string]any    `json:"prunedResult"`
		Markdown     markdownData      `json:"markdown"`
		OutputImages map[string]string `json:"outputImages"`
		InputImage   *string           `json:"inputImage"`
	}
	var results []lpr
	for _, pg := range res.Pages {
		text := pg.Markdown
		if req.PrettifyMarkdown != nil && !*req.PrettifyMarkdown {
			paddleFail(ctx, w, logID, http.StatusUnprocessableEntity, "`prettifyMarkdown` false is not implemented")
			return
		}
		e := lpr{PrunedResult: pg.Pruned, Markdown: markdownData{Text: text}}
		if req.ReturnMarkdownImages == nil || *req.ReturnMarkdownImages {
			e.Markdown.Images = map[string]string{}
			for path, img := range pg.MarkdownImages {
				b, err := jpegBase64(img)
				if err != nil {
					paddleFail(ctx, w, logID, http.StatusInternalServerError, err.Error())
					return
				}
				e.Markdown.Images[path] = b
			}
		}
		results = append(results, e)
	}
	var dataInfo any
	if pdf {
		type pageInfo struct {
			Width  int `json:"width"`
			Height int `json:"height"`
		}
		pages := []pageInfo{}
		for _, pg := range res.Pages {
			pages = append(pages, pageInfo{pg.Width, pg.Height})
		}
		dataInfo = map[string]any{"numPages": len(pages), "pages": pages, "type": "pdf"}
	} else {
		dataInfo = map[string]any{"width": res.Pages[0].Width, "height": res.Pages[0].Height, "type": "image"}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"logId": logID, "errorCode": 0, "errorMsg": "Success",
		"result": map[string]any{"layoutParsingResults": results, "dataInfo": dataInfo},
	})
}

func orDefault(v string) string {
	if v == "" {
		return "its default"
	}
	return v
}

// randomUUID is a v4 UUID, PaddleX's logId.
func randomUUID() string {
	h := randomID() + randomID()
	return h[0:8] + "-" + h[8:12] + "-4" + h[13:16] + "-a" + h[17:20] + "-" + h[20:32]
}
