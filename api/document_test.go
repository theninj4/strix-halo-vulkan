package api

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"net/http"
	"strings"
	"testing"
)

// fakeDocs is a DocumentBackend that returns one page per request (or two
// for a PDF) and records the request.
type fakeDocs struct{ req *DocumentRequest }

func (f *fakeDocs) Models() []Model {
	return []Model{{ID: "PaddleOCR-VL-1.6-0.9B", Object: "model", OwnedBy: "PaddlePaddle"}}
}

func (f *fakeDocs) ParseDocument(_ context.Context, req *DocumentRequest) (*DocumentResult, error) {
	f.req = req
	n := 1
	if req.PDF {
		n = 2
	}
	res := &DocumentResult{}
	if req.PDF {
		res.PDFPages = 2
	}
	for i := 0; i < n; i++ {
		res.Pages = append(res.Pages, DocumentPage{
			Index: i, Width: 100, Height: 50,
			Mistral:       "# Title\n\n![img-0.jpeg](img-0.jpeg)",
			MistralImages: []DocumentImage{{ID: "img-0.jpeg", Box: [4]int{1, 2, 3, 4}, Image: image.NewGray(image.Rect(0, 0, 2, 2))}},
			Header:        "running head",
		})
	}
	return res, nil
}

var pngData = "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG not really"))

func TestOCREndpoint(t *testing.T) {
	f := &fakeDocs{}
	s := &Server{Document: f}
	for name, doc := range map[string]any{
		"document_url":     map[string]any{"type": "document_url", "document_url": pngData},
		"image_url string": map[string]any{"type": "image_url", "image_url": pngData},
		"image_url object": map[string]any{"type": "image_url", "image_url": map[string]any{"url": pngData}},
	} {
		rec := do(t, s, jsonRequest("POST", "/v1/ocr", map[string]any{"model": "mistral-ocr-latest", "document": doc,
			"include_image_base64": true, "extract_header": true}))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", name, rec.Code, rec.Body)
		}
		var out mistralResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		p := out.Pages[0]
		if out.Model != "PaddleOCR-VL-1.6-0.9B" || p.Markdown == "" || len(p.Images) != 1 || p.Images[0].Base64 == nil ||
			!strings.HasPrefix(*p.Images[0].Base64, "data:image/jpeg;base64,") || p.Images[0].BottomY != 4 ||
			p.Header == nil || *p.Header != "running head" || p.Dimensions.Width != 100 || out.UsageInfo.PagesProcessed != 1 {
			t.Errorf("%s: %s", name, rec.Body)
		}
	}
	for name, body := range map[string]map[string]any{
		"file id":     {"document": map[string]any{"type": "file", "file_id": "x"}},
		"annotation":  {"document": map[string]any{"type": "document_url", "document_url": pngData}, "document_annotation_format": map[string]any{"type": "json_schema"}},
		"md tables":   {"document": map[string]any{"type": "document_url", "document_url": pngData}, "table_format": "markdown"},
		"image pages": {"document": map[string]any{"type": "document_url", "document_url": pngData}, "pages": []int{1}},
		"no url":      {"document": map[string]any{"type": "document_url"}},
	} {
		if rec := do(t, s, jsonRequest("POST", "/v1/ocr", body)); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: %d, want 400: %s", name, rec.Code, rec.Body)
		}
	}
}
